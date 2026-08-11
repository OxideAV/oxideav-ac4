//! A-JCC parameter **extractor** (analysis side) — ETSI TS 103 190-2
//! §5.6 / §6.2.6, the encoder counterpart of the complete A-JCC decode
//! chain ([`crate::ajcc`] parameter decode + [`crate::ajcc_synth`]
//! Table 35 reconstruction).
//!
//! The extractor derives per-parameter-band quantised parameter rows
//! from QMF-domain statistics of the target output channels, then
//! assembles a complete `ajcc_data()` element with GOP-aware
//! differential coding:
//!
//! * **Quantisers** — exact inverses of the Table 30 dry
//!   (`q · Δ − 0,6`) and Table 31 wet (`q · Δ − 2,0`) dequantisers,
//!   plus alpha / beta through the Part 1 Tables 202-205 codebook
//!   lanes (the alpha raw-lane offset of `ajcc_huff_data()`'s plain
//!   `huff_decode` F0 included).
//! * **Per-band extraction** — the decoder reconstructs every output
//!   from **one** core channel per module (Table 37 / Table 38), so
//!   the natural core downmix per module is the plain output sum:
//!   dry gains sum to 1 and the wet coefficient rows sum to 0 in each
//!   module output group, making `x = Σ outputs` exact. Each dry gain
//!   is then the per-band least-squares projection `⟨z, x⟩ / ⟨x, x⟩`
//!   of its output onto the module input; each wet gain fills the
//!   projection residual's energy through the decorrelator model
//!   (`E[y] ≈ E[x]`), with the shared `wet3` coefficient pinned to 0 —
//!   an informative encoder choice (the TS specifies only the decoder
//!   mapping); alpha / beta come from the pair mid/side statistics
//!   exactly like the A-CPL extractors.
//! * **`ajcc_data()` assembly** — smooth interpolation, one parameter
//!   set per framing group, FREQ rows on I-frames and per-SET
//!   FREQ-vs-TIME differential selection priced by the real Annex
//!   A.1.2 codeword lengths on P-frames, with an encoder-side
//!   [`crate::ajcc::AjccState`] mirror advanced through the decoder's
//!   own [`crate::ajcc::ajcc_differential_decode`] so both sides stay
//!   in lockstep across a GOP.

use crate::acpl::{AcplHcbType, AcplQuantMode};
use crate::ajcc::{
    ajcc_differential_decode, encode_ajcc_deltas_freq, encode_ajcc_deltas_time, get_ajcc_hcb,
    AjccData, AjccDataType, AjccFramingData, AjccState, Ajced, AJCC_NUM_BANDS_TABLE,
};
use crate::ajoc::AjocDiffType;

/// One channel's QMF matrix in the encoder's `[sb][ts]` layout (as
/// produced by the streaming analysis banks).
pub type QmfMat = [Vec<(f32, f32)>];

// ---------------------------------------------------------------------
// Quantisers (Table 30 / 31 inverses + alpha / beta lanes)
// ---------------------------------------------------------------------

fn dry_wet_delta(qm: AcplQuantMode) -> f64 {
    match qm {
        AcplQuantMode::Fine => 0.1,
        AcplQuantMode::Coarse => 0.2,
    }
}

/// Quantise one A-JCC dry gain — exact inverse of
/// [`crate::ajcc::dequantize_ajcc_dry`] on the quantised grid, clamped
/// to the F0 codebook's index range.
pub fn quantize_ajcc_dry(v: f64, qm: AcplQuantMode) -> i32 {
    let hi = get_ajcc_hcb(AjccDataType::Dry, qm, AcplHcbType::F0)
        .len
        .len() as i32
        - 1;
    (((v + 0.6) / dry_wet_delta(qm)).round() as i32).clamp(0, hi)
}

/// Quantise one A-JCC wet gain — exact inverse of
/// [`crate::ajcc::dequantize_ajcc_wet`] on the quantised grid.
pub fn quantize_ajcc_wet(v: f64, qm: AcplQuantMode) -> i32 {
    let hi = get_ajcc_hcb(AjccDataType::Wet, qm, AcplHcbType::F0)
        .len
        .len() as i32
        - 1;
    (((v + 2.0) / dry_wet_delta(qm)).round() as i32).clamp(0, hi)
}

/// Quantise one A-JCC alpha to the **raw** F0 lane (`ajcc_huff_data()`
/// decodes the alpha F0 with a plain `huff_decode`, so the wire value
/// is the unsigned lane index — the signed Part 1 lane plus the F0
/// book's `cb_off`).
pub fn quantize_ajcc_alpha_raw(alpha: f32, qm: AcplQuantMode) -> i32 {
    let off = get_ajcc_hcb(AjccDataType::Alpha, qm, AcplHcbType::F0)
        .len
        .len() as i32
        / 2;
    crate::encoder_acpl3::quantise_alpha(alpha, qm) + off
}

/// Quantise one A-JCC beta magnitude to its signed lane (the in-tree
/// beta F0 books carry `cb_off = 0`, so the wire value is the signed
/// magnitude lane directly). The `ibeta` column coupling is resolved
/// by the decoder from the recovered alpha; the encoder quantises the
/// magnitude against the `ibeta = 0` column exactly like the A-CPL
/// extractors.
pub fn quantize_ajcc_beta(beta: f32, qm: AcplQuantMode) -> i32 {
    let q = crate::encoder_acpl3::quantise_beta_magnitude(beta.abs(), qm);
    if beta < 0.0 {
        -q
    } else {
        q
    }
}

// ---------------------------------------------------------------------
// Per-band statistics
// ---------------------------------------------------------------------

/// Per-parameter-band real cross-correlation `Σ Re(a · conj(b))` over
/// the whole frame, using the Part 1 Table 196/197 subband → band
/// mapping (§5.6.3.1).
fn band_cross(a: &QmfMat, b: &QmfMat, num_bands: u32) -> Vec<f64> {
    let mut out = vec![0.0f64; num_bands as usize];
    for (sb, row_a) in a.iter().enumerate() {
        let pb = crate::acpl::sb_to_pb(sb as u32, num_bands) as usize;
        let Some(row_b) = b.get(sb) else { continue };
        let mut acc = 0.0f64;
        for (va, vb) in row_a.iter().zip(row_b.iter()) {
            acc += va.0 as f64 * vb.0 as f64 + va.1 as f64 * vb.1 as f64;
        }
        if let Some(slot) = out.get_mut(pb) {
            *slot += acc;
        }
    }
    out
}

/// Per-parameter-band energy `Σ |a|²`.
fn band_energy(a: &QmfMat, num_bands: u32) -> Vec<f64> {
    band_cross(a, a, num_bands)
}

/// Linear combination of QMF matrices: `Σ gains[i] · mats[i]` (QMF
/// analysis is linear, so the combo of analyses equals the analysis of
/// the combo).
pub fn qmf_combo(mats: &[&QmfMat], gains: &[f32]) -> Vec<Vec<(f32, f32)>> {
    let num_sb = mats.iter().map(|m| m.len()).max().unwrap_or(0);
    let num_ts = mats
        .iter()
        .flat_map(|m| m.iter().map(|r| r.len()))
        .max()
        .unwrap_or(0);
    let mut out = vec![vec![(0.0f32, 0.0f32); num_ts]; num_sb];
    for (m, &g) in mats.iter().zip(gains.iter()) {
        for (sb, row) in m.iter().enumerate() {
            for (ts, v) in row.iter().enumerate() {
                out[sb][ts].0 += g * v.0;
                out[sb][ts].1 += g * v.1;
            }
        }
    }
    out
}

/// Relative energy floor below which a band is treated as silent and
/// coded with neutral parameters (dry 1/3-splits, zero alpha / beta /
/// wet).
const SILENT_BAND_FRACTION: f64 = 1e-9;

/// Quantised parameter rows for one A-JCC module-1 (`b_5fronts`,
/// Table 37) group: `dry1 / dry2` plus `wet1..wet3` per band.
struct Module1Rows {
    dry1: Vec<i32>,
    dry2: Vec<i32>,
    wet1: Vec<i32>,
    wet2: Vec<i32>,
    wet3: Vec<i32>,
}

/// Extract one Table 37 module's rows from its three target outputs
/// expressed in **module-output** scale (`za`, `zb`, `zc` — the
/// decoder's pre-√2 z signals) and the module input `x = za + zb + zc`.
fn extract_module1(
    za: &QmfMat,
    zb: &QmfMat,
    zc: &QmfMat,
    qm: AcplQuantMode,
    nb: u32,
) -> Module1Rows {
    let x = qmf_combo(&[za, zb, zc], &[1.0, 1.0, 1.0]);
    let e_x = band_energy(&x, nb);
    let total: f64 = e_x.iter().sum();
    let c_a = band_cross(za, &x, nb);
    let c_b = band_cross(zb, &x, nb);
    let e_b = band_energy(zb, nb);
    let e_c = band_energy(zc, nb);
    let n = nb as usize;
    let mut rows = Module1Rows {
        dry1: Vec::with_capacity(n),
        dry2: Vec::with_capacity(n),
        wet1: Vec::with_capacity(n),
        wet2: Vec::with_capacity(n),
        wet3: Vec::with_capacity(n),
    };
    for pb in 0..n {
        let ex = e_x[pb];
        if !(ex.is_finite()) || ex <= total * SILENT_BAND_FRACTION {
            // Silent band: neutral 1/3 dry split, no wet.
            rows.dry1.push(quantize_ajcc_dry(1.0 / 3.0, qm));
            rows.dry2.push(quantize_ajcc_dry(1.0 / 3.0, qm));
            rows.wet1.push(quantize_ajcc_wet(0.0, qm));
            rows.wet2.push(quantize_ajcc_wet(0.0, qm));
            rows.wet3.push(quantize_ajcc_wet(0.0, qm));
            continue;
        }
        let d1 = (c_a[pb] / ex).clamp(-0.6, 1.6);
        let d2 = (c_b[pb] / ex).clamp(-0.6, 1.6);
        let q1 = quantize_ajcc_dry(d1, qm);
        let q2 = quantize_ajcc_dry(d2, qm);
        let d1_dq = crate::ajcc::dequantize_ajcc_dry(q1, qm);
        let d2_dq = crate::ajcc::dequantize_ajcc_dry(q2, qm);
        let d3_dq = 1.0 - d1_dq - d2_dq;
        // Wet fill with wet3 = 0 (see module docs): zb's residual
        // energy rides `−s·wet2·y1`, zc's rides `−s·wet1·y0`
        // (`s = 1/√2`, `E[y] ≈ E[x]`).
        let res_b = (e_b[pb] - d2_dq * d2_dq * ex).max(0.0);
        let res_c = (e_c[pb] - d3_dq * d3_dq * ex).max(0.0);
        let w2 = (2.0 * res_b / ex).sqrt().clamp(0.0, 2.0);
        let w1 = (2.0 * res_c / ex).sqrt().clamp(0.0, 2.0);
        rows.dry1.push(q1);
        rows.dry2.push(q2);
        rows.wet1.push(quantize_ajcc_wet(w1, qm));
        rows.wet2.push(quantize_ajcc_wet(w2, qm));
        rows.wet3.push(quantize_ajcc_wet(0.0, qm));
    }
    rows
}

/// Quantised rows for one core-layout module (Table 38, `core_mode = 0`):
/// alpha / beta over the `(o0, o3)` front pair riding `x0` plus
/// dry / wet over the `(o1, o2, o4)` group riding `x1`.
struct Module2Rows {
    alpha: Vec<i32>,
    beta: Vec<i32>,
    dry1: Vec<i32>,
    dry2: Vec<i32>,
    wet1: Vec<i32>,
    wet2: Vec<i32>,
    wet3: Vec<i32>,
}

/// Extract one Table 38 (`core_mode = 0`) module's rows. Inputs are
/// the five module outputs in **z scale**: `o0` (front, unscaled),
/// `o1 / o2 / o4` (the x1 group: surround / back / top-back, each the
/// output channel ÷ √2) and `o3` (the α-coded top-front partner of
/// `o0`, also ÷ √2 — `z9` sits in the decoder's √2 range).
#[allow(clippy::too_many_arguments)]
fn extract_module2(
    o0: &QmfMat,
    o1: &QmfMat,
    o2: &QmfMat,
    o3: &QmfMat,
    o4: &QmfMat,
    qm_ab: AcplQuantMode,
    qm_dw: AcplQuantMode,
    nb: u32,
) -> Module2Rows {
    // x0 = o0 + o3 (the (1±α)/2 rows sum to 1, β rows cancel);
    // x1 = o1 + o2 + o4.
    let x0 = qmf_combo(&[o0, o3], &[1.0, 1.0]);
    let side = qmf_combo(&[o0, o3], &[1.0, -1.0]);
    let x1 = qmf_combo(&[o1, o2, o4], &[1.0, 1.0, 1.0]);
    let e_x0 = band_energy(&x0, nb);
    let e_side = band_energy(&side, nb);
    let c_side = band_cross(&side, &x0, nb);
    let e_x1 = band_energy(&x1, nb);
    let c_1 = band_cross(o1, &x1, nb);
    let c_2 = band_cross(o2, &x1, nb);
    let e_2 = band_energy(o2, nb);
    let e_4 = band_energy(o4, nb);
    let total0: f64 = e_x0.iter().sum();
    let total1: f64 = e_x1.iter().sum();
    let n = nb as usize;
    let mut rows = Module2Rows {
        alpha: Vec::with_capacity(n),
        beta: Vec::with_capacity(n),
        dry1: Vec::with_capacity(n),
        dry2: Vec::with_capacity(n),
        wet1: Vec::with_capacity(n),
        wet2: Vec::with_capacity(n),
        wet3: Vec::with_capacity(n),
    };
    for pb in 0..n {
        // Alpha / beta over the (o0, o3) pair: side = α·x0 + β·y0.
        let ex0 = e_x0[pb];
        if ex0.is_finite() && ex0 > total0 * SILENT_BAND_FRACTION {
            let alpha = (c_side[pb] / ex0).clamp(-2.0, 2.0) as f32;
            let a_q = crate::encoder_acpl3::quantise_alpha(alpha, qm_ab);
            let a_dq = crate::acpl_synth::dequantize_alpha_index(qm_ab, a_q).0 as f64;
            let beta_sq = e_side[pb] / ex0 - a_dq * a_dq;
            let beta = if beta_sq > 0.0 && beta_sq.is_finite() {
                (beta_sq.sqrt() as f32).clamp(0.0, 4.0)
            } else {
                0.0
            };
            rows.alpha.push(quantize_ajcc_alpha_raw(alpha, qm_ab));
            rows.beta.push(quantize_ajcc_beta(beta, qm_ab));
        } else {
            rows.alpha.push(quantize_ajcc_alpha_raw(0.0, qm_ab));
            rows.beta.push(0);
        }
        // Dry / wet over the (o1, o2, o4) group (same shape as the
        // Table 37 module: dry3 = 1 − dry1 − dry2 rides o4, wet3 = 0,
        // o2's residual on wet2, o4's on wet1).
        let ex1 = e_x1[pb];
        if !(ex1.is_finite()) || ex1 <= total1 * SILENT_BAND_FRACTION {
            rows.dry1.push(quantize_ajcc_dry(1.0 / 3.0, qm_dw));
            rows.dry2.push(quantize_ajcc_dry(1.0 / 3.0, qm_dw));
            rows.wet1.push(quantize_ajcc_wet(0.0, qm_dw));
            rows.wet2.push(quantize_ajcc_wet(0.0, qm_dw));
            rows.wet3.push(quantize_ajcc_wet(0.0, qm_dw));
            continue;
        }
        let d1 = (c_1[pb] / ex1).clamp(-0.6, 1.6);
        let d2 = (c_2[pb] / ex1).clamp(-0.6, 1.6);
        let q1 = quantize_ajcc_dry(d1, qm_dw);
        let q2 = quantize_ajcc_dry(d2, qm_dw);
        let d2_dq = crate::ajcc::dequantize_ajcc_dry(q2, qm_dw);
        let d3_dq = 1.0 - crate::ajcc::dequantize_ajcc_dry(q1, qm_dw) - d2_dq;
        let res_2 = (e_2[pb] - d2_dq * d2_dq * ex1).max(0.0);
        let res_4 = (e_4[pb] - d3_dq * d3_dq * ex1).max(0.0);
        let w2 = (2.0 * res_2 / ex1).sqrt().clamp(0.0, 2.0);
        let w1 = (2.0 * res_4 / ex1).sqrt().clamp(0.0, 2.0);
        rows.dry1.push(q1);
        rows.dry2.push(q2);
        rows.wet1.push(quantize_ajcc_wet(w1, qm_dw));
        rows.wet2.push(quantize_ajcc_wet(w2, qm_dw));
        rows.wet3.push(quantize_ajcc_wet(0.0, qm_dw));
    }
    rows
}

// ---------------------------------------------------------------------
// Extracted frame rows (roster order)
// ---------------------------------------------------------------------

/// Quantised per-band parameter rows for one frame in the
/// [`crate::ajcc::AjccData`] roster order.
pub struct AjccExtractedRows {
    /// Layout the rows were extracted for.
    pub b_5fronts: bool,
    /// Alpha SET rows (`[set][pb]`, raw F0 lane; empty for 5-fronts).
    pub alpha: Vec<Vec<i32>>,
    /// Beta SET rows (signed magnitude lane).
    pub beta: Vec<Vec<i32>>,
    /// Dry SET rows (4, or 8 for 5-fronts).
    pub dry: Vec<Vec<i32>>,
    /// Wet SET rows (6, or 12 for 5-fronts).
    pub wet: Vec<Vec<i32>>,
}

/// Extract the core-layout (`b_5fronts = 0`, `core_mode = 0`)
/// parameter rows from the 11 target output channels' QMF matrices in
/// the decoder's slot order `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl,
/// Tbr]`.
///
/// Module 0 (left) covers `(L, Tfl)` on alpha / beta and
/// `(Ls, Lb, Tbl)` on dry / wet; module 1 (right) the mirror set. The
/// decoder's √2 output range on z5..z12 means the module-scale signals
/// are the output channels ÷ √2 for everything except L / R.
pub fn extract_ajcc_core_rows(
    named_q: &[&QmfMat; 11],
    num_bands: u32,
    qm_ab: AcplQuantMode,
    qm_dw: AcplQuantMode,
) -> AjccExtractedRows {
    let isq2 = std::f32::consts::FRAC_1_SQRT_2;
    let z = |i: usize| qmf_combo(&[named_q[i]], &[isq2]);
    // Left module: o0 = L, o1 = Ls/√2, o2 = Lb/√2, o3 = Tfl/√2,
    // o4 = Tbl/√2.
    let (ls, lb, tfl, tbl) = (z(3), z(5), z(7), z(9));
    let left = extract_module2(named_q[0], &ls, &lb, &tfl, &tbl, qm_ab, qm_dw, num_bands);
    let (rs, rb, tfr, tbr) = (z(4), z(6), z(8), z(10));
    let right = extract_module2(named_q[1], &rs, &rb, &tfr, &tbr, qm_ab, qm_dw, num_bands);
    AjccExtractedRows {
        b_5fronts: false,
        alpha: vec![left.alpha, right.alpha],
        beta: vec![left.beta, right.beta],
        dry: vec![left.dry1, left.dry2, right.dry1, right.dry2],
        wet: vec![
            left.wet1, left.wet2, left.wet3, right.wet1, right.wet2, right.wet3,
        ],
    }
}

/// Extract the `b_5fronts = 1` parameter rows from the 13 target
/// output channels' QMF matrices in the decoder's slot order `[L, R,
/// C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
///
/// The four Table 37 modules cover `(L, Tfl, Lscr)`, `(R, Tfr, Rscr)`
/// (front groups — `qm_f`) and `(Ls, Lb, Ltb)`, `(Rs, Rb, Rtb)` (back
/// groups — `qm_b`); the z-scale signals are the output channels ÷ √2
/// on z5..z12 (surrounds / backs / tops) and unscaled on the fronts /
/// screens.
pub fn extract_ajcc_5fronts_rows(
    named_q: &[&QmfMat; 13],
    num_bands: u32,
    qm_f: AcplQuantMode,
    qm_b: AcplQuantMode,
) -> AjccExtractedRows {
    let isq2 = std::f32::consts::FRAC_1_SQRT_2;
    let z = |i: usize| qmf_combo(&[named_q[i]], &[isq2]);
    // Front modules: za = L (z0), zb = Tfl/√2 (z9), zc = Lscr (z3).
    let tfl = z(9);
    let m_lf = extract_module1(named_q[0], &tfl, named_q[3], qm_f, num_bands);
    let tfr = z(10);
    let m_rf = extract_module1(named_q[1], &tfr, named_q[4], qm_f, num_bands);
    // Back modules: za = Ls/√2 (z5), zb = Lb/√2 (z7), zc = Ltb/√2 (z11).
    let (ls, lb, ltb) = (z(5), z(7), z(11));
    let m_lb = extract_module1(&ls, &lb, &ltb, qm_b, num_bands);
    let (rs, rb, rtb) = (z(6), z(8), z(12));
    let m_rb = extract_module1(&rs, &rb, &rtb, qm_b, num_bands);
    AjccExtractedRows {
        b_5fronts: true,
        alpha: Vec::new(),
        beta: Vec::new(),
        dry: vec![
            m_lf.dry1, m_lf.dry2, m_rf.dry1, m_rf.dry2, m_lb.dry1, m_lb.dry2, m_rb.dry1, m_rb.dry2,
        ],
        wet: vec![
            m_lf.wet1, m_lf.wet2, m_lf.wet3, m_rf.wet1, m_rf.wet2, m_rf.wet3, m_lb.wet1, m_lb.wet2,
            m_lb.wet3, m_rb.wet1, m_rb.wet2, m_rb.wet3,
        ],
    }
}

// ---------------------------------------------------------------------
// ajcc_data() assembly with GOP-aware differential coding
// ---------------------------------------------------------------------

/// Encoder-side differential-coding state: a mirror of the decoder's
/// per-SET `ajcc_<SET>_q_prev` rows, advanced through the decoder's
/// own [`ajcc_differential_decode`] after every emitted frame.
#[derive(Debug, Default, Clone)]
pub struct AjccEncoderState {
    prev: Option<AjccState>,
}

impl AjccEncoderState {
    /// Fresh (unprimed) state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the cross-frame reference (e.g. on a layout change).
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

/// Bit cost of one Huffman symbol row.
fn row_bits(hcb: &crate::ajcc::AjccHcb, indices: impl Iterator<Item = i32>) -> Option<u32> {
    let mut bits = 0u32;
    for idx in indices {
        if idx < 0 || idx as usize >= hcb.len.len() {
            return None;
        }
        bits += hcb.len[idx as usize] as u32;
    }
    Some(bits)
}

/// Choose the cheaper differential direction for one SET row and
/// return the `ajcc_huff_data()` payload. `prev` is the decoder-side
/// previous quantised row for this SET (TIME allowed only when
/// `Some`). Falls back to FREQ when a TIME delta leaves the DT book.
fn encode_set_row(
    dt: AjccDataType,
    qm: AcplQuantMode,
    q_row: &[i32],
    prev: Option<&[i32]>,
) -> (AjocDiffType, Vec<i32>) {
    let freq = encode_ajcc_deltas_freq(q_row);
    let f0 = get_ajcc_hcb(dt, qm, AcplHcbType::F0);
    let df = get_ajcc_hcb(dt, qm, AcplHcbType::Df);
    let freq_bits = freq.split_first().and_then(|(&first, rest)| {
        let head = row_bits(&f0, std::iter::once(first))?;
        let tail = row_bits(&df, rest.iter().map(|&d| d + df.cb_off))?;
        Some(head + tail)
    });
    if let Some(prev) = prev {
        let time = encode_ajcc_deltas_time(q_row, prev);
        let dt_book = get_ajcc_hcb(dt, qm, AcplHcbType::Dt);
        let time_bits = row_bits(&dt_book, time.iter().map(|&d| d + dt_book.cb_off));
        match (freq_bits, time_bits) {
            (Some(fb), Some(tb)) if tb < fb => return (AjocDiffType::Time, time),
            (None, Some(_)) => return (AjocDiffType::Time, time),
            _ => {}
        }
    }
    (AjocDiffType::Freq, freq)
}

/// Configuration of one assembled `ajcc_data()` element.
#[derive(Debug, Clone, Copy)]
pub struct AjccBuildConfig {
    /// `ajcc_num_param_bands_id` (Table 108 index).
    pub num_param_bands_id: u8,
    /// `ajcc_core_mode` (core layout only; the extractor produces
    /// `core_mode = 0` parameters).
    pub core_mode: bool,
    /// Front / alpha-beta quant mode (`ajcc_qm_f` for 5-fronts,
    /// `ajcc_qm_ab` for the core layout).
    pub qm_first: AcplQuantMode,
    /// Back / dry-wet quant mode (`ajcc_qm_b` / `ajcc_qm_dw`).
    pub qm_second: AcplQuantMode,
}

impl Default for AjccBuildConfig {
    fn default() -> Self {
        AjccBuildConfig {
            num_param_bands_id: 0, // 15 bands — 1:1 with QMF sb 0..8
            core_mode: false,
            qm_first: AcplQuantMode::Fine,
            qm_second: AcplQuantMode::Fine,
        }
    }
}

/// Assemble a complete `ajcc_data()` element from extracted rows.
///
/// Framing: smooth interpolation, one parameter set per group. On
/// I-frames (or with an unprimed `state`) every SET goes out
/// FREQ-coded; on P-frames each SET independently picks FREQ vs TIME
/// by real codeword bit cost against the decoder-side previous rows.
/// `state` is advanced to this frame's rows (via the decoder's own
/// differential decode) so a following P-frame prices against exactly
/// what the decoder will hold.
pub fn build_ajcc_data(
    rows: &AjccExtractedRows,
    cfg: &AjccBuildConfig,
    state: &mut AjccEncoderState,
    b_iframe: bool,
) -> AjccData {
    let b_5fronts = rows.b_5fronts;
    let nb = AJCC_NUM_BANDS_TABLE[(cfg.num_param_bands_id & 3) as usize];
    let n_framing = if b_5fronts { 4 } else { 2 };
    let framing = vec![
        AjccFramingData {
            steep: false,
            num_param_sets: 1,
            param_timeslot: Vec::new(),
        };
        n_framing
    ];
    if b_iframe {
        state.prev = None;
    }
    // A stored state from the other layout has a different SET roster
    // and cannot serve as a TIME reference.
    if let Some(p) = &state.prev {
        if p.dry_prev.len() != rows.dry.len() || p.wet_prev.len() != rows.wet.len() {
            state.prev = None;
        }
    }
    let have_prev = state.prev.is_some();
    let mut prev = state
        .prev
        .take()
        .unwrap_or_else(|| AjccState::new(b_5fronts));

    let qm_dw = |set_idx: usize, is_dry: bool| -> AcplQuantMode {
        if b_5fronts {
            // dry SETs 0..4 are front (qm_f), 4..8 back (qm_b); wet
            // SETs 0..6 front, 6..12 back.
            let front = if is_dry { set_idx < 4 } else { set_idx < 6 };
            if front {
                cfg.qm_first
            } else {
                cfg.qm_second
            }
        } else {
            cfg.qm_second
        }
    };

    let mut data = AjccData {
        b_5fronts,
        b_no_dt: false,
        num_param_bands_id: cfg.num_param_bands_id & 3,
        num_bands: nb,
        core_mode: cfg.core_mode,
        qm_f: cfg.qm_first,
        qm_b: cfg.qm_second,
        qm_ab: cfg.qm_first,
        qm_dw: cfg.qm_second,
        framing,
        alpha: Vec::new(),
        beta: Vec::new(),
        dry: Vec::new(),
        wet: Vec::new(),
    };

    let encode_family = |sets: &[Vec<i32>],
                         dt: AjccDataType,
                         qm_of: &dyn Fn(usize) -> AcplQuantMode,
                         prev_rows: &mut [Vec<i32>]|
     -> Vec<Ajced> {
        sets.iter()
            .enumerate()
            .map(|(i, q_row)| {
                let qm = qm_of(i);
                let p = if have_prev {
                    Some(&prev_rows[i][..nb as usize])
                } else {
                    None
                };
                let (dir, deltas) = encode_set_row(dt, qm, q_row, p);
                let set: Ajced = vec![(dir, deltas)];
                // Advance the decoder-side mirror.
                ajcc_differential_decode(&set, nb as usize, &mut prev_rows[i]);
                set
            })
            .collect()
    };

    data.alpha = encode_family(
        &rows.alpha,
        AjccDataType::Alpha,
        &|_| cfg.qm_first,
        &mut prev.alpha_prev,
    );
    data.beta = encode_family(
        &rows.beta,
        AjccDataType::Beta,
        &|_| cfg.qm_first,
        &mut prev.beta_prev,
    );
    data.dry = encode_family(
        &rows.dry,
        AjccDataType::Dry,
        &|i| qm_dw(i, true),
        &mut prev.dry_prev,
    );
    data.wet = encode_family(
        &rows.wet,
        AjccDataType::Wet,
        &|i| qm_dw(i, false),
        &mut prev.wet_prev,
    );

    state.prev = Some(prev);
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ajcc::{
        decode_ajcc_parsed, dequantize_ajcc_dry, dequantize_ajcc_wet, parse_ajcc_data,
        write_ajcc_data,
    };
    use oxideav_core::bits::{BitReader, BitWriter};

    #[test]
    fn dry_wet_quantizers_invert_the_dequantizers_on_grid() {
        for qm in [AcplQuantMode::Fine, AcplQuantMode::Coarse] {
            let dry_hi = get_ajcc_hcb(AjccDataType::Dry, qm, AcplHcbType::F0)
                .len
                .len() as i32
                - 1;
            for q in 0..=dry_hi {
                let v = dequantize_ajcc_dry(q, qm);
                assert_eq!(quantize_ajcc_dry(v, qm), q, "dry {qm:?} q={q}");
            }
            let wet_hi = get_ajcc_hcb(AjccDataType::Wet, qm, AcplHcbType::F0)
                .len
                .len() as i32
                - 1;
            for q in 0..=wet_hi {
                let v = dequantize_ajcc_wet(q, qm);
                assert_eq!(quantize_ajcc_wet(v, qm), q, "wet {qm:?} q={q}");
            }
        }
    }

    #[test]
    fn alpha_raw_lane_roundtrips_through_the_decoder_bridge() {
        for qm in [AcplQuantMode::Fine, AcplQuantMode::Coarse] {
            for alpha in [-1.0f32, -0.5, 0.0, 0.25, 0.5, 1.0] {
                let raw = quantize_ajcc_alpha_raw(alpha, qm);
                let hi = get_ajcc_hcb(AjccDataType::Alpha, qm, AcplHcbType::F0)
                    .len
                    .len() as i32
                    - 1;
                assert!((0..=hi).contains(&raw), "raw lane in range");
                let (a_dq, _) = crate::ajcc::dequantize_ajcc_alpha_beta(raw, 0, qm);
                assert!(
                    (a_dq - alpha).abs() < 0.15,
                    "{qm:?} alpha {alpha} → lane {raw} → {a_dq}"
                );
            }
        }
    }

    /// Synthetic QMF matrix with one tone-like subband row.
    fn tone_mat(sb: usize, amp: f32, num_ts: usize) -> Vec<Vec<(f32, f32)>> {
        let mut m = vec![vec![(0.0f32, 0.0f32); num_ts]; 12];
        for (ts, v) in m[sb].iter_mut().enumerate() {
            let ph = 0.37f32 * ts as f32;
            *v = (amp * ph.cos(), amp * ph.sin());
        }
        m
    }

    fn scale_mat(m: &QmfMat, g: f32) -> Vec<Vec<(f32, f32)>> {
        qmf_combo(&[m], &[g])
    }

    #[test]
    fn module1_extraction_recovers_known_dry_gains() {
        // Build za/zb/zc as known dry fractions of a shared source in
        // one band (no decorrelated content → wet ≈ 0).
        let num_ts = 30;
        let x = tone_mat(2, 0.8, num_ts);
        let za = scale_mat(&x, 0.6);
        let zb = scale_mat(&x, 0.3);
        let zc = scale_mat(&x, 0.1);
        let rows = extract_module1(&za, &zb, &zc, AcplQuantMode::Fine, 15);
        // pb2 carries the content: dry1 = 0,6 → q = 12; dry2 = 0,3 → 9.
        assert_eq!(rows.dry1[2], 12);
        assert_eq!(rows.dry2[2], 9);
        // wet ≈ 0 → the Table 31 zero lane (q = 20 fine).
        assert_eq!(rows.wet1[2], 20);
        assert_eq!(rows.wet2[2], 20);
        // Silent band: neutral split.
        assert_eq!(
            rows.dry1[5],
            quantize_ajcc_dry(1.0 / 3.0, AcplQuantMode::Fine)
        );
    }

    #[test]
    fn module2_extraction_recovers_alpha_and_dry() {
        let num_ts = 30;
        // Front pair (o0, o3) fully correlated with o3 = 0,5·o0 in
        // band 1 → mid = 1,5·o0-unit; α = (1 − 0,5)/(1 + 0,5) = 1/3.
        let o0 = tone_mat(1, 0.6, num_ts);
        let o3 = scale_mat(&o0, 0.5);
        // x1 group in band 3: o1 = 0,7·src, o2 = 0,2·src, o4 = 0,1·src.
        let src = tone_mat(3, 0.9, num_ts);
        let o1 = scale_mat(&src, 0.7);
        let o2 = scale_mat(&src, 0.2);
        let o4 = scale_mat(&src, 0.1);
        let rows = extract_module2(
            &o0,
            &o1,
            &o2,
            &o3,
            &o4,
            AcplQuantMode::Fine,
            AcplQuantMode::Fine,
            15,
        );
        let a_raw = rows.alpha[1];
        let (a_dq, _) = crate::ajcc::dequantize_ajcc_alpha_beta(a_raw, 0, AcplQuantMode::Fine);
        assert!(
            (a_dq - 1.0 / 3.0).abs() < 0.1,
            "α ≈ 1/3 recovered (got {a_dq})"
        );
        assert_eq!(rows.beta[1], 0, "correlated pair → β = 0");
        assert_eq!(rows.dry1[3], quantize_ajcc_dry(0.7, AcplQuantMode::Fine));
        assert_eq!(rows.dry2[3], quantize_ajcc_dry(0.2, AcplQuantMode::Fine));
    }

    /// Full-roster synthetic rows for GOP tests.
    fn synthetic_rows(b_5fronts: bool, nb: usize, seed: i32) -> AjccExtractedRows {
        let row = |base: i32, lo: i32, hi: i32| -> Vec<i32> {
            (0..nb as i32)
                .map(|i| (base + (i * 7 + seed) % 5 - 2).clamp(lo, hi))
                .collect()
        };
        if b_5fronts {
            AjccExtractedRows {
                b_5fronts: true,
                alpha: Vec::new(),
                beta: Vec::new(),
                dry: (0..8).map(|i| row(8 + i, 0, 22)).collect(),
                wet: (0..12).map(|i| row(20 + i, 0, 40)).collect(),
            }
        } else {
            AjccExtractedRows {
                b_5fronts: false,
                // Beta rows stay in the non-negative magnitude lanes —
                // the F0 wire symbol is unsigned (negative values are
                // only reachable through DF/DT deltas), and the
                // extractor emits magnitudes.
                alpha: (0..2).map(|i| row(16 + i, 0, 32)).collect(),
                beta: (0..2).map(|i| row(2 + i, 0, 8)).collect(),
                dry: (0..4).map(|i| row(8 + i, 0, 22)).collect(),
                wet: (0..6).map(|i| row(20 + i, 0, 40)).collect(),
            }
        }
    }

    #[test]
    fn build_write_parse_decode_lockstep_over_gop() {
        for b_5fronts in [false, true] {
            let cfg = AjccBuildConfig::default();
            let nb = 15usize;
            let mut enc_state = AjccEncoderState::new();
            let mut dec_state = AjccState::new(b_5fronts);
            let mut saw_time = false;
            // Frame 1 changes the rows (FREQ or TIME, whichever is
            // cheaper); frames 2-3 hold them stationary, where the
            // all-zero TIME deltas are strictly cheapest.
            for (frame, (seed, b_iframe)) in [(0, true), (3, false), (3, false), (3, false)]
                .into_iter()
                .enumerate()
            {
                let rows = synthetic_rows(b_5fronts, nb, seed);
                let data = build_ajcc_data(&rows, &cfg, &mut enc_state, b_iframe);
                if frame > 0 {
                    saw_time |= data
                        .dry
                        .iter()
                        .chain(data.wet.iter())
                        .flat_map(|s| s.iter())
                        .any(|(d, _)| *d == AjocDiffType::Time);
                }
                // Wire round-trip.
                let mut bw = BitWriter::new();
                write_ajcc_data(&mut bw, &data).expect("write ajcc_data");
                let bytes = bw.finish();
                let mut br = BitReader::new(&bytes);
                let parsed = parse_ajcc_data(&mut br, b_5fronts).expect("parse ajcc_data");
                assert_eq!(parsed, data, "wire round-trip frame {frame}");
                // Decoder-side differential decode recovers exactly
                // the extractor's quantised rows.
                let decoded = decode_ajcc_parsed(parsed, &mut dec_state).expect("decode");
                for (i, set) in rows.dry.iter().enumerate() {
                    assert_eq!(
                        decoded.dry_q[i][0], *set,
                        "dry SET {i} frame {frame} (5fronts {b_5fronts})"
                    );
                }
                for (i, set) in rows.wet.iter().enumerate() {
                    assert_eq!(decoded.wet_q[i][0], *set, "wet SET {i} frame {frame}");
                }
                for (i, set) in rows.alpha.iter().enumerate() {
                    assert_eq!(decoded.alpha_q[i][0], *set, "alpha SET {i} frame {frame}");
                }
                for (i, set) in rows.beta.iter().enumerate() {
                    assert_eq!(decoded.beta_q[i][0], *set, "beta SET {i} frame {frame}");
                }
            }
            assert!(
                saw_time,
                "P-frames with near-stationary rows must pick TIME rows somewhere (5fronts {b_5fronts})"
            );
        }
    }

    #[test]
    fn iframe_resets_the_time_reference() {
        let cfg = AjccBuildConfig::default();
        let mut enc_state = AjccEncoderState::new();
        let rows = synthetic_rows(false, 15, 1);
        let _ = build_ajcc_data(&rows, &cfg, &mut enc_state, true);
        let data_p = build_ajcc_data(&rows, &cfg, &mut enc_state, false);
        // Stationary rows on a P-frame: TIME rows are all-zero deltas,
        // strictly cheaper than re-sending FREQ.
        assert!(data_p.dry.iter().all(|s| s
            .iter()
            .all(|(d, v)| *d == AjocDiffType::Time && v.iter().all(|&x| x == 0))));
        // A new I-frame must go back to FREQ everywhere.
        let data_i = build_ajcc_data(&rows, &cfg, &mut enc_state, true);
        assert!(data_i
            .dry
            .iter()
            .all(|s| s.iter().all(|(d, _)| *d == AjocDiffType::Freq)));
    }
}
