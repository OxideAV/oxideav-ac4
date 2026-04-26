//! A-CPL (Advanced Coupling) QMF synthesis math — ETSI TS 103 190-1
//! §5.7.7.
#![allow(clippy::excessive_precision, clippy::needless_range_loop)]
//!
//! What this module covers today:
//!
//! * **§5.7.7.7 Differential decoding + dequantization** —
//!   [`differential_decode`] (Pseudocode 121) absorbs the per-band
//!   Huffman deltas produced by `acpl_data_*ch()` and re-creates the
//!   absolute quantised parameter array `acpl_<SET>_q[ps][pb]` for both
//!   `DIFF_FREQ` and `DIFF_TIME` directions, chaining the
//!   `acpl_SET_q_prev` vector across parameter sets and AC-4 frames.
//!   The four index→float dequantisation tables ([`ALPHA_DQ_FINE`],
//!   [`ALPHA_DQ_COARSE`], [`BETA_DQ_FINE`], [`BETA_DQ_COARSE`] +
//!   [`IBETA_FINE`] / [`IBETA_COARSE`], Tables 203-206) lift the
//!   integer indices to real-valued α / β coefficients;
//!   [`dequantize_beta3`] / [`dequantize_gamma`] handle the simpler
//!   delta-multiplier paths from Tables 207 / 208. ALPHA / BETA / BETA3
//!   are coupled — the alpha index selects an `ibeta` row in Table 204
//!   / 206 that the corresponding beta lookup uses, so
//!   [`dequantize_alpha_beta`] returns both arrays in one shot.
//!
//! * **§5.7.7.3 Time interpolation** — [`interpolate`] (Pseudocode 109)
//!   linearly interpolates per-QMF-subsample (sb, ts) values from the
//!   per-parameter-set vectors using the `sb_to_pb` table-197 mapping,
//!   carrying the previous frame's `acpl_param_prev[sb]` array. Both
//!   smooth and steep interpolation modes are wired (smooth = linear
//!   between borders, steep = piecewise-constant at
//!   `acpl_param_timeslot[]`).
//!
//! * **§5.7.7.4.2 Decorrelator** — [`InputSignalModifier`] implements
//!   Pseudocode 111: a frequency-banded all-pass IIR (3 regions × 3
//!   coefficient sets per Tables 199 / 200 / 201) preceded by a
//!   per-region constant delay (Table 198: `7 / 10 / 12` slots for
//!   `k0 / k1 / k2`). Three modules `D0`, `D1`, `D2` are exposed via
//!   [`Decorrelator`] for the `u0 / u1 / u2` paths.
//!
//! * **§5.7.7.4.3 Transient ducker** — [`TransientDucker`]
//!   (Pseudocodes 112-114) computes the per-parameter-band peak-decay /
//!   smooth / smooth-peak-diff rolling state and applies the resulting
//!   `duck_gain[pb]` to the decorrelated signal via the `sb_to_pb`
//!   mapping.
//!
//! * **§5.7.7.5 Channel pair element** — [`acpl_module`]
//!   (Pseudocode 116) is the canonical 2-channel synthesis: it runs
//!   the M/S split below `acpl_qmf_band` and the mixed
//!   `0.5 * (x0*(1±α) ± y*β)` path above it. [`AcplCpeState`] bundles
//!   the previous-frame `acpl_alpha_prev` / `acpl_beta_prev` arrays
//!   plus the decorrelator + ducker state for the full
//!   `Pseudocode 115` channel-pair synthesis path.
//!
//! What is NOT covered yet (TS 103 190-1):
//!
//! * Multichannel `5_X_codec_mode = ASPX_ACPL_3` (Pseudocode 117 /
//!   Pseudocode 118) — needs `Transform`, `ACplModule2`, `ACplModule3`
//!   and the gamma-driven 3-module pipeline; ALPHA / BETA / BETA3 /
//!   GAMMA dequant tables are already in place to plug those in.
//! * Hooking the synthesis into the frame-level decoder pipeline (the
//!   asf walker still parses `acpl_data_*` but doesn't drive
//!   [`acpl_module`]).

use crate::acpl::{sb_to_pb, AcplHuffParam, AcplQuantMode};
use crate::qmf::NUM_QMF_SUBBANDS;

// =====================================================================
// §5.7.7.7 Tables 203 / 204 / 205 / 206 — ALPHA / BETA dequantisation
// =====================================================================

/// `alpha_dq[alpha_q]` for fine quantisation per Table 203.
/// Length 33 (`alpha_q ∈ 0..=32`).
#[rustfmt::skip]
pub const ALPHA_DQ_FINE: [f32; 33] = [
    -2.000000, -1.809375, -1.637500, -1.484375, -1.350000, -1.234375,
    -1.137500, -1.059375, -1.000000, -0.940625, -0.862500, -0.765625,
    -0.650000, -0.515625, -0.362500, -0.190625,  0.000000,  0.190625,
     0.362500,  0.515625,  0.650000,  0.765625,  0.862500,  0.940625,
     1.000000,  1.059375,  1.137500,  1.234375,  1.350000,  1.484375,
     1.637500,  1.809375,  2.000000,
];

/// `ibeta[alpha_q]` for fine quantisation per Table 203 (column 2).
/// Used to pick a column out of `BETA_DQ_FINE` for the matching beta
/// dequant.
#[rustfmt::skip]
pub const IBETA_FINE: [u8; 33] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

/// `alpha_dq[alpha_q]` for coarse quantisation per Table 205. Length 17.
#[rustfmt::skip]
pub const ALPHA_DQ_COARSE: [f32; 17] = [
    -2.000000, -1.637500, -1.350000, -1.137500, -1.000000, -0.862500,
    -0.650000, -0.362500,  0.000000,  0.362500,  0.650000,  0.862500,
     1.000000,  1.137500,  1.350000,  1.637500,  2.000000,
];

/// `ibeta[alpha_q]` for coarse quantisation per Table 205 (column 2).
#[rustfmt::skip]
pub const IBETA_COARSE: [u8; 17] = [
    0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0,
];

/// `beta_dq[beta_q][ibeta]` for fine quantisation per Table 204.
/// Indexed by `[beta_q][ibeta]` with `beta_q ∈ 0..=8`, `ibeta ∈ 0..=8`.
/// The spec table is signed via the `beta_q` ↔ huffman delta convention
/// (positive index = positive coefficient); here the table only carries
/// the magnitude — the caller flips the sign when `beta_q < 0`.
#[rustfmt::skip]
pub const BETA_DQ_FINE: [[f32; 9]; 9] = [
    // beta_q = 0
    [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
    // beta_q = 1
    [0.2375000, 0.2035449, 0.1729297, 0.1456543, 0.1217188, 0.1011230, 0.0838672, 0.0699512, 0.0593750],
    // beta_q = 2
    [0.5500000, 0.4713672, 0.4004688, 0.3373047, 0.2818750, 0.2341797, 0.1942188, 0.1619922, 0.1375000],
    // beta_q = 3
    [0.9375000, 0.8034668, 0.6826172, 0.5749512, 0.4804688, 0.3991699, 0.3310547, 0.2761230, 0.2343750],
    // beta_q = 4
    [1.4000000, 1.1998440, 1.0193750, 0.8585938, 0.7175000, 0.5960938, 0.4943750, 0.4123438, 0.3500000],
    // beta_q = 5
    [1.9375000, 1.6604980, 1.4107420, 1.1882319, 0.9929688, 0.8249512, 0.6841797, 0.5706543, 0.4843750],
    // beta_q = 6
    [2.5500000, 2.1854300, 1.8567190, 1.5638670, 1.3068750, 1.0857420, 0.9004688, 0.7510547, 0.6375000],
    // beta_q = 7
    [3.2375000, 2.7746389, 2.3573050, 1.9854980, 1.6592190, 1.3784670, 1.1432420, 0.9535449, 0.8093750],
    // beta_q = 8
    [4.0000000, 3.4281249, 2.9124999, 2.4531250, 2.0500000, 1.7031250, 1.4125000, 1.1781250, 1.0000000],
];

/// `beta_dq[beta_q][ibeta]` for coarse quantisation per Table 206.
/// Indexed by `[beta_q][ibeta]` with `beta_q ∈ 0..=4`, `ibeta ∈ 0..=4`.
#[rustfmt::skip]
pub const BETA_DQ_COARSE: [[f32; 5]; 5] = [
    // beta_q = 0
    [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
    // beta_q = 1
    [0.5500000, 0.4004688, 0.2818750, 0.1942188, 0.1375000],
    // beta_q = 2
    [1.4000000, 1.0193750, 0.7175000, 0.4943750, 0.3500000],
    // beta_q = 3
    [2.5500000, 1.8567190, 1.3068750, 0.9004688, 0.6375000],
    // beta_q = 4
    [4.0000000, 2.9124999, 2.0500000, 1.4125000, 1.0000000],
];

/// `beta3` dequantisation delta per Table 207. Multiply the recovered
/// `beta3_q` value by this delta to get `acpl_beta_3_dq`.
pub fn beta3_delta(qm: AcplQuantMode) -> f32 {
    match qm {
        AcplQuantMode::Fine => 0.125,
        AcplQuantMode::Coarse => 0.25,
    }
}

/// `gamma` dequantisation delta per Table 208 (`g1..g6`). Both modes
/// expressed as the spec's exact `n / 16384` ratios.
pub fn gamma_delta(qm: AcplQuantMode) -> f32 {
    match qm {
        AcplQuantMode::Fine => 1638.0 / 16384.0,
        AcplQuantMode::Coarse => 3276.0 / 16384.0,
    }
}

/// Look up `(alpha_dq, ibeta)` from a recovered alpha quantised index
/// per Table 203 (Fine) or Table 205 (Coarse).
///
/// The huffman pipeline already produces a *signed* `alpha_q` in the
/// range `-N/2 ..= +N/2` (where `N = ALPHA_DQ_*.len() - 1`) thanks to
/// the F0/DF/DT codebooks' `cb_off`. The dequant tables are addressed
/// by the *unsigned* lane index `alpha_q + cb_off`, where `cb_off =
/// N/2`. We re-add it here.
pub fn dequantize_alpha_index(qm: AcplQuantMode, alpha_q: i32) -> (f32, u8) {
    match qm {
        AcplQuantMode::Fine => {
            let lane = (alpha_q + 16).clamp(0, 32) as usize;
            (ALPHA_DQ_FINE[lane], IBETA_FINE[lane])
        }
        AcplQuantMode::Coarse => {
            let lane = (alpha_q + 8).clamp(0, 16) as usize;
            (ALPHA_DQ_COARSE[lane], IBETA_COARSE[lane])
        }
    }
}

/// Look up the dequantised beta from `beta_q` and the matching
/// `ibeta` produced by [`dequantize_alpha_index`].
///
/// Per Table 204 / 206, beta values are unsigned magnitudes — the
/// recovered `beta_q` carries the sign, so we flip the lookup once at
/// the end.
pub fn dequantize_beta_index(qm: AcplQuantMode, beta_q: i32, ibeta: u8) -> f32 {
    let mag = beta_q.unsigned_abs() as usize;
    let val = match qm {
        AcplQuantMode::Fine => {
            let row = mag.min(8);
            let col = (ibeta as usize).min(8);
            BETA_DQ_FINE[row][col]
        }
        AcplQuantMode::Coarse => {
            let row = mag.min(4);
            let col = (ibeta as usize).min(4);
            BETA_DQ_COARSE[row][col]
        }
    };
    if beta_q < 0 {
        -val
    } else {
        val
    }
}

// =====================================================================
// §5.7.7.7 Pseudocode 121 — differential decoding
// =====================================================================

/// Differential decoding state per parameter-set type
/// (`acpl_<SET>_q_prev`).
///
/// The spec mandates an initial all-zero `q_prev` for the first AC-4
/// frame, and carries the last `q[ps]` row forward across frames to
/// seed the next `DIFF_TIME` decode.
#[derive(Debug, Clone)]
pub struct AcplDiffState {
    /// Carries `acpl_<SET>_q[num_pset-1]` across calls. Length =
    /// `acpl_num_param_bands` once primed; the first call sees an
    /// empty vector and fills it.
    pub q_prev: Vec<i32>,
}

impl AcplDiffState {
    pub fn new() -> Self {
        Self { q_prev: Vec::new() }
    }

    /// Replace the running `q_prev` row.
    pub fn carry(&mut self, row: &[i32]) {
        self.q_prev.clear();
        self.q_prev.extend_from_slice(row);
    }
}

impl Default for AcplDiffState {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply `Pseudocode 121` differential decoding to the per-parameter-set
/// huffman-decoded deltas to recover the absolute `acpl_<SET>_q` array.
///
/// `params[ps]` is one parameter set's worth of huffman output (with
/// either `DIFF_FREQ` or `DIFF_TIME` direction); `num_bands` is
/// `acpl_num_param_bands`. The returned matrix is `num_pset`
/// rows × `num_bands` columns. The state `q_prev` is updated to the
/// last decoded row for forward chaining.
pub fn differential_decode(
    params: &[AcplHuffParam],
    num_bands: u32,
    state: &mut AcplDiffState,
) -> Vec<Vec<i32>> {
    let nb = num_bands as usize;
    if state.q_prev.len() != nb {
        state.q_prev = vec![0; nb];
    }
    let mut out = Vec::with_capacity(params.len());
    for p in params {
        // Per the spec the huffman delta vector should be exactly
        // `num_bands` long (per `acpl_data_*` framing); be defensive
        // for unusual lengths by taking what we can and zero-padding.
        let mut row = vec![0i32; nb];
        if !p.direction_time {
            // DIFF_FREQ — first value is absolute, rest accumulate.
            if !p.values.is_empty() {
                row[0] = p.values[0];
                for i in 1..nb {
                    let d = p.values.get(i).copied().unwrap_or(0);
                    row[i] = row[i - 1] + d;
                }
            }
        } else {
            // DIFF_TIME — each band reuses q_prev[i] + delta.
            for i in 0..nb {
                let d = p.values.get(i).copied().unwrap_or(0);
                row[i] = state.q_prev[i] + d;
            }
        }
        state.q_prev = row.clone();
        out.push(row);
    }
    out
}

// =====================================================================
// §5.7.7.7 Helpers — turn `acpl_<SET>_q` into dequantised f32 arrays
// =====================================================================

/// Dequantise one (or two) parameter sets' worth of (alpha, beta)
/// indices into their floating-point counterparts. The returned matrix
/// shape mirrors the input: `pset × num_bands`.
///
/// The shared `ibeta` linkage between Tables 203/204 (Fine) and
/// Tables 205/206 (Coarse) is honoured per the spec — beta is looked up
/// with the alpha column's `ibeta`.
pub fn dequantize_alpha_beta(
    alpha_q: &[Vec<i32>],
    beta_q: &[Vec<i32>],
    qm: AcplQuantMode,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    debug_assert_eq!(alpha_q.len(), beta_q.len());
    let mut alpha_dq = Vec::with_capacity(alpha_q.len());
    let mut beta_dq = Vec::with_capacity(beta_q.len());
    for ps in 0..alpha_q.len() {
        let n = alpha_q[ps].len();
        let mut a_row = vec![0.0f32; n];
        let mut b_row = vec![0.0f32; n];
        for i in 0..n {
            let (a_val, ibeta) = dequantize_alpha_index(qm, alpha_q[ps][i]);
            a_row[i] = a_val;
            let bq = beta_q[ps].get(i).copied().unwrap_or(0);
            b_row[i] = dequantize_beta_index(qm, bq, ibeta);
        }
        alpha_dq.push(a_row);
        beta_dq.push(b_row);
    }
    (alpha_dq, beta_dq)
}

/// Dequantise `beta3_q` to `acpl_beta_3_dq` per Table 207.
pub fn dequantize_beta3(beta3_q: &[Vec<i32>], qm: AcplQuantMode) -> Vec<Vec<f32>> {
    let delta = beta3_delta(qm);
    beta3_q
        .iter()
        .map(|row| row.iter().map(|&q| (q as f32) * delta).collect())
        .collect()
}

/// Dequantise `gamma_q` to `g_dq` per Table 208.
pub fn dequantize_gamma(gamma_q: &[Vec<i32>], qm: AcplQuantMode) -> Vec<Vec<f32>> {
    let delta = gamma_delta(qm);
    gamma_q
        .iter()
        .map(|row| row.iter().map(|&q| (q as f32) * delta).collect())
        .collect()
}

// =====================================================================
// §5.7.7.3 Pseudocode 109 — interpolate()
// =====================================================================

/// One frame's worth of dequantised parameter values across (pset, sb).
///
/// Indexed `[ps][sb]` after the spec's `interpolate()` rewrite (which
/// folds the `sb_to_pb()` lookup into the storage). The previous
/// frame's last-row values are carried in `prev` (length
/// `num_qmf_subbands`).
#[derive(Debug, Clone)]
pub struct InterpInputs<'a> {
    /// `acpl_param_dq[ps][sb]` — already expanded over `sb_to_pb()`.
    /// Outer length = `num_pset`, inner = `num_qmf_subbands`.
    pub by_pset: &'a [Vec<f32>],
    /// `acpl_param_prev[sb]` from the previous frame's last set.
    pub prev: &'a [f32],
}

/// `interpolate(acpl_param, num_pset, sb, ts)` per §5.7.7.3
/// Pseudocode 109.
///
/// `num_ts` is `num_qmf_timeslots`. `interpolation_type == 0` means
/// smooth, `== 1` means steep (driven by `acpl_param_timeslot[]` —
/// `param_timeslots` here).
#[allow(clippy::too_many_arguments)]
pub fn interpolate(
    inputs: &InterpInputs<'_>,
    num_pset: u32,
    sb: u32,
    ts: u32,
    num_ts: u32,
    interp_steep: bool,
    param_timeslots: &[u8],
) -> f32 {
    let sb_idx = sb as usize;
    if !interp_steep {
        // Smooth interpolation
        if num_pset == 1 {
            let p = inputs.by_pset[0].get(sb_idx).copied().unwrap_or(0.0);
            let prev = inputs.prev.get(sb_idx).copied().unwrap_or(0.0);
            let delta = p - prev;
            prev + ((ts + 1) as f32) * delta / (num_ts as f32)
        } else {
            // 2 parameter sets
            let ts_2 = num_ts / 2;
            let p0 = inputs.by_pset[0].get(sb_idx).copied().unwrap_or(0.0);
            let p1 = inputs.by_pset[1].get(sb_idx).copied().unwrap_or(0.0);
            if ts < ts_2 {
                let prev = inputs.prev.get(sb_idx).copied().unwrap_or(0.0);
                let delta = p0 - prev;
                prev + ((ts + 1) as f32) * delta / (ts_2 as f32)
            } else {
                let delta = p1 - p0;
                p0 + ((ts - ts_2 + 1) as f32) * delta / ((num_ts - ts_2) as f32)
            }
        }
    } else {
        // Steep interpolation
        if num_pset == 1 {
            let bound = param_timeslots.first().copied().unwrap_or(0) as u32;
            if ts < bound {
                inputs.prev.get(sb_idx).copied().unwrap_or(0.0)
            } else {
                inputs.by_pset[0].get(sb_idx).copied().unwrap_or(0.0)
            }
        } else {
            let bound0 = param_timeslots.first().copied().unwrap_or(0) as u32;
            let bound1 = param_timeslots.get(1).copied().unwrap_or(0) as u32;
            if ts < bound0 {
                inputs.prev.get(sb_idx).copied().unwrap_or(0.0)
            } else if ts < bound1 {
                inputs.by_pset[0].get(sb_idx).copied().unwrap_or(0.0)
            } else {
                inputs.by_pset[1].get(sb_idx).copied().unwrap_or(0.0)
            }
        }
    }
}

/// Expand a per-parameter-band array `pb_vals[pset][pb]` to a per-QMF-
/// subband array `sb_vals[pset][sb]` via `sb_to_pb()` (§5.7.7.2 Table
/// 197). This is the standard input to [`interpolate`]'s `by_pset` field.
pub fn expand_pb_to_sb(pb_vals: &[Vec<f32>], num_param_bands: u32) -> Vec<Vec<f32>> {
    pb_vals
        .iter()
        .map(|row| {
            let mut out = vec![0.0f32; NUM_QMF_SUBBANDS];
            for sb in 0..NUM_QMF_SUBBANDS {
                let pb = sb_to_pb(sb as u32, num_param_bands) as usize;
                out[sb] = row.get(pb).copied().unwrap_or(0.0);
            }
            out
        })
        .collect()
}

/// Update the `prev[sb]` array per §5.7.7.3 Pseudocode 110, using the
/// last parameter set in the current frame.
pub fn update_param_prev(prev: &mut Vec<f32>, last_set_sb: &[f32]) {
    prev.clear();
    prev.extend_from_slice(last_set_sb);
    // Pad / truncate to NUM_QMF_SUBBANDS so the next frame sees a
    // canonically-shaped state.
    prev.resize(NUM_QMF_SUBBANDS, 0.0);
}

// =====================================================================
// §5.7.7.4.2 Decorrelator — Tables 198 / 199 / 200 / 201 + Pseudocode
// 111
// =====================================================================

/// Per-subband region delay (Table 198).
pub fn region_delay(sb: u32) -> usize {
    match sb {
        0..=6 => 7,
        7..=22 => 10,
        _ => 12,
    }
}

/// Per-subband filter length (Table 198).
pub fn region_filter_length(sb: u32) -> usize {
    match sb {
        0..=6 => 7,
        7..=22 => 4,
        _ => 2,
    }
}

/// Decorrelator selector for the three parallel modules.
#[derive(Debug, Clone, Copy)]
pub enum DecorrelatorId {
    D0,
    D1,
    D2,
}

/// Region k0 coefficients (Table 199), indexed `[D][i]` with i in
/// `0..=7`. `D0` = column 0, `D1` = column 1, `D2` = column 2.
#[rustfmt::skip]
pub const A_K0: [[f64; 8]; 3] = [
    // D0
    [1.0000, 0.5306, -0.4533, -0.6248,  0.0424,  0.4237,  0.4311,  0.1688],
    // D1
    [1.0000, -0.4178, 0.1082, -0.2368, -0.1014, -0.1052, -0.3528,  0.4665],
    // D2
    [1.0000, 0.4007,  0.4747,  0.2611, -0.1211, -0.4248, -0.2989, -0.1932],
];

/// Region k1 coefficients (Table 200), indexed `[D][i]` with i in
/// `0..=4`.
#[rustfmt::skip]
pub const A_K1: [[f64; 5]; 3] = [
    [1.0000,  0.5561, -0.3039, -0.5024, -0.1850],
    [1.0000,  0.0425,  0.3235, -0.1556,  0.4958],
    [1.0000, -0.4361,  0.0345,  0.5215, -0.4178],
];

/// Region k2 coefficients (Table 201), indexed `[D][i]` with i in
/// `0..=2`.
#[rustfmt::skip]
pub const A_K2: [[f64; 3]; 3] = [
    [1.0000,  0.5773,  0.3321],
    [1.0000,  0.2327, -0.3901],
    [1.0000, -0.6057,  0.3804],
];

/// Look up the `a[i]` coefficient vector for the requested decorrelator
/// `D` and subband region (k0 / k1 / k2). Returned slice length matches
/// `region_filter_length(sb) + 1`.
pub fn region_coeffs(d: DecorrelatorId, sb: u32) -> &'static [f64] {
    let col = match d {
        DecorrelatorId::D0 => 0,
        DecorrelatorId::D1 => 1,
        DecorrelatorId::D2 => 2,
    };
    match sb {
        0..=6 => &A_K0[col][..],
        7..=22 => &A_K1[col][..],
        _ => &A_K2[col][..],
    }
}

/// Per-channel + per-decorrelator running state for
/// [`InputSignalModifier`]. Carries the tail of `x[ts-i-delay][sb]`
/// (input history) and `y[ts-i][sb]` (output history) needed by
/// Pseudocode 111. Sized to the maximum delay+filterLength = 12 + 2 =
/// 14 in the worst region; we allocate 24 per subband for slack.
#[derive(Debug, Clone)]
pub struct InputSignalModifier {
    /// Decorrelator id (D0/D1/D2).
    pub which: DecorrelatorId,
    /// `x_hist[sb][k]` = previous input sample at offset `k` slots in
    /// the past for QMF subband `sb`. Complex (re, im) pair.
    pub x_hist: Vec<Vec<(f32, f32)>>,
    /// `y_hist[sb][k]` = previous output sample at offset `k` slots in
    /// the past.
    pub y_hist: Vec<Vec<(f32, f32)>>,
}

impl InputSignalModifier {
    pub fn new(which: DecorrelatorId) -> Self {
        let depth = 24usize;
        Self {
            which,
            x_hist: vec![vec![(0.0, 0.0); depth]; NUM_QMF_SUBBANDS],
            y_hist: vec![vec![(0.0, 0.0); depth]; NUM_QMF_SUBBANDS],
        }
    }

    /// Reset the running history (e.g. on substream restart).
    pub fn reset(&mut self) {
        for h in &mut self.x_hist {
            for v in h {
                *v = (0.0, 0.0);
            }
        }
        for h in &mut self.y_hist {
            for v in h {
                *v = (0.0, 0.0);
            }
        }
    }

    /// Process one (sb, ts) sample of the input matrix `x`. Returns the
    /// decorrelated output `y[ts][sb]`. The caller pushes the input
    /// (re, im) and receives the corresponding output.
    pub fn process_sample(&mut self, sb: u32, x_in: (f32, f32)) -> (f32, f32) {
        let sb_idx = sb as usize;
        let delay = region_delay(sb);
        let filter_length = region_filter_length(sb);
        let a = region_coeffs(self.which, sb);
        // Push x_in into x_hist (offset 0 = brand-new sample).
        let xh = &mut self.x_hist[sb_idx];
        xh.rotate_right(1);
        xh[0] = x_in;
        let yh = &mut self.y_hist[sb_idx];

        // b[0] = a[filterLength], y = b[0] * x[ts-delay] / a[0]
        let b0 = a[filter_length];
        let a0 = a[0];
        let xd = if delay < xh.len() {
            xh[delay]
        } else {
            (0.0, 0.0)
        };
        let mut y_re = (b0 / a0) * xd.0 as f64;
        let mut y_im = (b0 / a0) * xd.1 as f64;
        for i in 1..=filter_length {
            let bi = a[filter_length - i];
            let xi_idx = i + delay;
            let xi = if xi_idx < xh.len() {
                xh[xi_idx]
            } else {
                (0.0, 0.0)
            };
            // y_hist[i-1] is the output at ts-i (since after rotate_right
            // we'll push current y at offset 0).
            let yi = if i - 1 < yh.len() {
                yh[i - 1]
            } else {
                (0.0, 0.0)
            };
            let ai = a[i];
            y_re += (bi * xi.0 as f64 - ai * yi.0 as f64) / a0;
            y_im += (bi * xi.1 as f64 - ai * yi.1 as f64) / a0;
        }
        let y_out = (y_re as f32, y_im as f32);
        // Push y_out into y_hist (rotate_right shifts older outputs up).
        yh.rotate_right(1);
        yh[0] = y_out;
        y_out
    }
}

/// Triplet of decorrelator modules for the three `u_x` paths used by
/// the multichannel A-CPL configurations.
#[derive(Debug, Clone)]
pub struct Decorrelator {
    pub d0: InputSignalModifier,
    pub d1: InputSignalModifier,
    pub d2: InputSignalModifier,
}

impl Decorrelator {
    pub fn new() -> Self {
        Self {
            d0: InputSignalModifier::new(DecorrelatorId::D0),
            d1: InputSignalModifier::new(DecorrelatorId::D1),
            d2: InputSignalModifier::new(DecorrelatorId::D2),
        }
    }
}

impl Default for Decorrelator {
    fn default() -> Self {
        Self::new()
    }
}

// =====================================================================
// §5.7.7.4.3 Transient ducker — Pseudocodes 112 / 113 / 114
// =====================================================================

/// Spec constants from Pseudocode 112.
pub const DUCKER_ALPHA: f32 = 0.76592833836465;
pub const DUCKER_ALPHA_SMOOTH: f32 = 0.25;
pub const DUCKER_GAMMA: f32 = 1.5;
pub const DUCKER_EPSILON: f32 = 1.0e-9;

/// `acpl_max_num_param_bands = 15` per Pseudocode 112.
pub const ACPL_MAX_NUM_PARAM_BANDS: usize = 15;

/// Persistent transient-ducker state across AC-4 frames.
#[derive(Debug, Clone)]
pub struct TransientDucker {
    pub p_peak_decay_prev: [f32; ACPL_MAX_NUM_PARAM_BANDS],
    pub p_smooth_prev: [f32; ACPL_MAX_NUM_PARAM_BANDS],
    pub smooth_peak_diff_prev: [f32; ACPL_MAX_NUM_PARAM_BANDS],
}

impl TransientDucker {
    pub fn new() -> Self {
        Self {
            p_peak_decay_prev: [0.0; ACPL_MAX_NUM_PARAM_BANDS],
            p_smooth_prev: [0.0; ACPL_MAX_NUM_PARAM_BANDS],
            smooth_peak_diff_prev: [0.0; ACPL_MAX_NUM_PARAM_BANDS],
        }
    }

    /// Reset state (called when `acpl_param_prev` is reinitialised).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Compute `duck_gain[pb]` from the per-pb energy array
    /// (Pseudocode 112). Updates the carrying state.
    pub fn update(
        &mut self,
        p_energy: &[f32; ACPL_MAX_NUM_PARAM_BANDS],
    ) -> [f32; ACPL_MAX_NUM_PARAM_BANDS] {
        let mut duck = [1.0f32; ACPL_MAX_NUM_PARAM_BANDS];
        for pb in 0..ACPL_MAX_NUM_PARAM_BANDS {
            let p_peak_decay = if DUCKER_ALPHA * self.p_peak_decay_prev[pb] < p_energy[pb] {
                p_energy[pb]
            } else {
                DUCKER_ALPHA * self.p_peak_decay_prev[pb]
            };
            let smooth = (1.0 - DUCKER_ALPHA_SMOOTH) * self.p_smooth_prev[pb]
                + DUCKER_ALPHA_SMOOTH * p_energy[pb];
            let smooth_peak_diff = (1.0 - DUCKER_ALPHA_SMOOTH) * self.smooth_peak_diff_prev[pb]
                + DUCKER_ALPHA_SMOOTH * (p_peak_decay - p_energy[pb]);
            let g = if DUCKER_GAMMA * smooth_peak_diff > smooth {
                smooth / (DUCKER_GAMMA * (smooth_peak_diff + DUCKER_EPSILON))
            } else {
                1.0
            };
            duck[pb] = g;
            self.p_peak_decay_prev[pb] = p_peak_decay;
            self.p_smooth_prev[pb] = smooth;
            self.smooth_peak_diff_prev[pb] = smooth_peak_diff;
        }
        duck
    }
}

impl Default for TransientDucker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute `p_energy[pb]` from one column of QMF subbands per
/// Pseudocode 113 (`pb` = parameter band, summed |x|² over the QMF
/// subbands mapping to that pb via Table 197).
pub fn compute_p_energy(
    x: &[(f32, f32); NUM_QMF_SUBBANDS],
    num_param_bands: u32,
) -> [f32; ACPL_MAX_NUM_PARAM_BANDS] {
    let mut e = [0.0f32; ACPL_MAX_NUM_PARAM_BANDS];
    for sb in 0..NUM_QMF_SUBBANDS as u32 {
        let pb = sb_to_pb(sb, num_param_bands) as usize;
        let (re, im) = x[sb as usize];
        e[pb] += re * re + im * im;
    }
    e
}

/// Apply the per-pb `duck_gain` to a single QMF column (Pseudocode 114).
pub fn apply_transient_ducker(
    x: &[(f32, f32); NUM_QMF_SUBBANDS],
    duck_gain: &[f32; ACPL_MAX_NUM_PARAM_BANDS],
    num_param_bands: u32,
) -> [(f32, f32); NUM_QMF_SUBBANDS] {
    let mut out = *x;
    for sb in 0..NUM_QMF_SUBBANDS as u32 {
        let pb = sb_to_pb(sb, num_param_bands) as usize;
        let g = duck_gain[pb];
        out[sb as usize].0 *= g;
        out[sb as usize].1 *= g;
    }
    out
}

// =====================================================================
// §5.7.7.5 Pseudocode 115 / 116 — channel-pair element synthesis
// =====================================================================

/// Persistent state for a single ACplModule (carries the previous
/// frame's `acpl_alpha_prev[sb]` / `acpl_beta_prev[sb]` and the
/// decorrelator + ducker scratch).
#[derive(Debug, Clone)]
pub struct AcplCpeState {
    pub alpha_prev_sb: Vec<f32>,
    pub beta_prev_sb: Vec<f32>,
    pub decorrelator: InputSignalModifier,
    pub ducker: TransientDucker,
}

impl AcplCpeState {
    pub fn new(which: DecorrelatorId) -> Self {
        Self {
            alpha_prev_sb: vec![0.0; NUM_QMF_SUBBANDS],
            beta_prev_sb: vec![0.0; NUM_QMF_SUBBANDS],
            decorrelator: InputSignalModifier::new(which),
            ducker: TransientDucker::new(),
        }
    }
}

/// Inputs to one full §5.7.7.5 channel-pair synthesis pass.
pub struct AcplCpeFrame<'a> {
    /// `x0[ts][sb]` — left / first-channel QMF input matrix
    /// (`num_qmf_timeslots` slots × 64 subbands). For ASPX_ACPL_2 the
    /// `x1` channel is absent (caller passes None).
    pub x0: &'a [[(f32, f32); NUM_QMF_SUBBANDS]],
    /// `x1[ts][sb]` — right / second-channel QMF input matrix. Some()
    /// for ASPX_ACPL_1, None for ASPX_ACPL_2.
    pub x1: Option<&'a [[(f32, f32); NUM_QMF_SUBBANDS]]>,
    /// `acpl_alpha_dq[pset][pb]` recovered values.
    pub alpha_dq: &'a [Vec<f32>],
    /// `acpl_beta_dq[pset][pb]` recovered values.
    pub beta_dq: &'a [Vec<f32>],
    /// `acpl_num_param_bands`.
    pub num_param_bands: u32,
    /// `acpl_qmf_band` from Table 59 — first subband at which the
    /// mixed-with-decorrelator path takes over from the M/S split.
    /// (For multichannel paths the spec sets this to 0.)
    pub acpl_qmf_band: u32,
    /// Steep interpolation flag.
    pub steep: bool,
    /// `acpl_param_timeslot[pset]` for the steep mode.
    pub param_timeslots: &'a [u8],
}

/// Output matrices from [`acpl_module`] — `(z0, z1)` per Pseudocode 116.
pub struct AcplCpeOutput {
    pub z0: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>,
    pub z1: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>,
}

/// `Pseudocode 116 ACplModule(acpl_alpha, acpl_beta, num_pset, x0, x1, y)`
/// — the two-channel A-CPL synthesis.
///
/// `y[ts][sb]` is the post-ducker decorrelated output of `x0` (provided
/// by [`run_pseudocode_115_pair`] below); `x1` may be all-zero for the
/// `ASPX_ACPL_2` mode (single-channel input).
pub fn acpl_module(
    frame: &AcplCpeFrame<'_>,
    y: &[[(f32, f32); NUM_QMF_SUBBANDS]],
) -> AcplCpeOutput {
    let num_ts = frame.x0.len();
    debug_assert_eq!(y.len(), num_ts);
    let num_pset = frame.alpha_dq.len() as u32;
    let alpha_sb = expand_pb_to_sb(frame.alpha_dq, frame.num_param_bands);
    let beta_sb = expand_pb_to_sb(frame.beta_dq, frame.num_param_bands);
    // Default prev to zeros for the first frame.
    let prev_alpha = vec![0.0f32; NUM_QMF_SUBBANDS];
    let prev_beta = vec![0.0f32; NUM_QMF_SUBBANDS];
    let alpha_inp = InterpInputs {
        by_pset: &alpha_sb,
        prev: &prev_alpha,
    };
    let beta_inp = InterpInputs {
        by_pset: &beta_sb,
        prev: &prev_beta,
    };

    let zero_col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
    let mut z0_out = Vec::with_capacity(num_ts);
    let mut z1_out = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let x0_col = frame.x0[ts];
        let x1_col = match frame.x1 {
            Some(matrix) => matrix[ts],
            None => zero_col,
        };
        let y_col = y[ts];
        let mut z0_col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        let mut z1_col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            let interp_a = interpolate(
                &alpha_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                frame.steep,
                frame.param_timeslots,
            );
            let interp_b = interpolate(
                &beta_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                frame.steep,
                frame.param_timeslots,
            );
            let sb_i = sb as usize;
            if sb < frame.acpl_qmf_band {
                let (x0r, x0i) = x0_col[sb_i];
                let (x1r, x1i) = x1_col[sb_i];
                z0_col[sb_i] = (0.5 * (x0r + x1r), 0.5 * (x0i + x1i));
                z1_col[sb_i] = (0.5 * (x0r - x1r), 0.5 * (x0i - x1i));
            } else {
                let (x0r, x0i) = x0_col[sb_i];
                let (yr, yi) = y_col[sb_i];
                let plus_a = 1.0 + interp_a;
                let minus_a = 1.0 - interp_a;
                z0_col[sb_i] = (
                    0.5 * (x0r * plus_a + yr * interp_b),
                    0.5 * (x0i * plus_a + yi * interp_b),
                );
                z1_col[sb_i] = (
                    0.5 * (x0r * minus_a - yr * interp_b),
                    0.5 * (x0i * minus_a - yi * interp_b),
                );
            }
        }
        z0_out.push(z0_col);
        z1_out.push(z1_col);
    }
    AcplCpeOutput {
        z0: z0_out,
        z1: z1_out,
    }
}

/// Run the complete §5.7.7.5 channel-pair element pipeline
/// (Pseudocode 115): scale inputs by 2, run the decorrelator + ducker
/// to get `y0`, then call [`acpl_module`].
///
/// `state` carries the `D0` decorrelator + ducker between calls (so
/// the IIR delay-line state survives across frames).
pub fn run_pseudocode_115_pair(state: &mut AcplCpeState, frame: AcplCpeFrame<'_>) -> AcplCpeOutput {
    // x0in = 2 * x0
    let num_ts = frame.x0.len();
    let mut x0in = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let mut col = frame.x0[ts];
        for sb in 0..NUM_QMF_SUBBANDS {
            col[sb].0 *= 2.0;
            col[sb].1 *= 2.0;
        }
        x0in.push(col);
    }
    // u0 = inputSignalModification(x0in)  (D0)
    let mut u0 = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let mut col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            col[sb as usize] = state.decorrelator.process_sample(sb, x0in[ts][sb as usize]);
        }
        u0.push(col);
    }
    // y0 = applyTransientDucker(u0)
    let mut y0 = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let p_energy = compute_p_energy(&u0[ts], frame.num_param_bands);
        let duck = state.ducker.update(&p_energy);
        y0.push(apply_transient_ducker(
            &u0[ts],
            &duck,
            frame.num_param_bands,
        ));
    }
    // Call acpl_module with x0in (note the spec: ACplModule receives
    // x0in, not raw x0).
    let mut x1in_owned: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>;
    let inner_x1: Option<&[[(f32, f32); NUM_QMF_SUBBANDS]]> = if let Some(x1) = frame.x1 {
        x1in_owned = Vec::with_capacity(num_ts);
        for ts in 0..num_ts {
            let mut col = x1[ts];
            for sb in 0..NUM_QMF_SUBBANDS {
                col[sb].0 *= 2.0;
                col[sb].1 *= 2.0;
            }
            x1in_owned.push(col);
        }
        Some(x1in_owned.as_slice())
    } else {
        None
    };
    let inner_frame = AcplCpeFrame {
        x0: &x0in,
        x1: inner_x1,
        alpha_dq: frame.alpha_dq,
        beta_dq: frame.beta_dq,
        num_param_bands: frame.num_param_bands,
        acpl_qmf_band: frame.acpl_qmf_band,
        steep: frame.steep,
        param_timeslots: frame.param_timeslots,
    };
    let out = acpl_module(&inner_frame, &y0);
    // Update prev arrays per Pseudocode 110 (last param set's expanded
    // sb values).
    if !frame.alpha_dq.is_empty() {
        let last_idx = frame.alpha_dq.len() - 1;
        let alpha_sb = expand_pb_to_sb(frame.alpha_dq, frame.num_param_bands);
        let beta_sb = expand_pb_to_sb(frame.beta_dq, frame.num_param_bands);
        update_param_prev(&mut state.alpha_prev_sb, &alpha_sb[last_idx]);
        update_param_prev(&mut state.beta_prev_sb, &beta_sb[last_idx]);
    }
    out
}

// =====================================================================
// §5.7.7 — top-level helpers wired into Ac4Decoder
// =====================================================================

use crate::acpl::{AcplConfig1ch, AcplData1ch, AcplInterpolationType};
use crate::qmf::{QmfAnalysisBank, QmfSynthesisBank};

/// Per-substream A-CPL persistent state — diff state for ALPHA and BETA
/// plus the channel-pair `AcplCpeState` (decorrelator + ducker + prev
/// arrays). Carried across AC-4 frames by the decoder.
#[derive(Debug, Clone)]
pub struct AcplSubstreamState {
    pub alpha_diff: AcplDiffState,
    pub beta_diff: AcplDiffState,
    pub cpe: AcplCpeState,
}

impl AcplSubstreamState {
    pub fn new() -> Self {
        Self {
            alpha_diff: AcplDiffState::new(),
            beta_diff: AcplDiffState::new(),
            cpe: AcplCpeState::new(DecorrelatorId::D0),
        }
    }
}

impl Default for AcplSubstreamState {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the §5.7.7 A-CPL channel-pair synthesis on a mono PCM input
/// (already ASPX-extended) and emit stereo PCM via QMF
/// analysis → A-CPL → QMF synthesis.
///
/// Spec wiring (ETSI TS 103 190-1):
///   * `pcm_in` — mono ASPX-extended PCM, length must be a multiple of
///     64 (one QMF slot = 64 samples).
///   * `cfg` — parsed `acpl_config_1ch()` (PARTIAL or FULL) for the
///     active substream (§4.2.13.1 Table 59).
///   * `data` — parsed `acpl_data_1ch()` (§4.2.13.3 Table 61).
///   * `state` — per-substream state carried across frames.
///
/// Returns `(left, right)` interleaved-friendly PCM buffers, each the
/// same length as `pcm_in`, or `None` if the input length isn't aligned
/// to a QMF slot boundary or the parameters are inconsistent.
pub fn run_acpl_1ch_pcm(
    pcm_in: &[f32],
    cfg: &AcplConfig1ch,
    data: &AcplData1ch,
    state: &mut AcplSubstreamState,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if pcm_in.is_empty() || pcm_in.len() % NUM_QMF_SUBBANDS != 0 {
        return None;
    }
    let n_slots = pcm_in.len() / NUM_QMF_SUBBANDS;
    if n_slots == 0 {
        return None;
    }
    // Forward QMF analysis on the input PCM. `process_block` already
    // returns the per-slot `[(re, im); 64]` columns we need.
    let mut ana = QmfAnalysisBank::new();
    let x0 = ana.process_block(pcm_in);
    // Differential decode + dequantize ALPHA / BETA.
    let alpha_q = differential_decode(&data.alpha1, cfg.num_param_bands, &mut state.alpha_diff);
    let beta_q = differential_decode(&data.beta1, cfg.num_param_bands, &mut state.beta_diff);
    let (alpha_dq, beta_dq) = dequantize_alpha_beta(&alpha_q, &beta_q, cfg.quant_mode);
    if alpha_dq.is_empty() {
        return None;
    }
    // Run the §5.7.7.5 channel-pair element. ASPX_ACPL_2 has x1 == None
    // (single ASPX channel — the spec's "1-channel A-CPL").
    let frame = AcplCpeFrame {
        x0: &x0,
        x1: None,
        alpha_dq: &alpha_dq,
        beta_dq: &beta_dq,
        num_param_bands: cfg.num_param_bands,
        acpl_qmf_band: cfg.qmf_band as u32,
        steep: matches!(
            data.framing.interpolation_type,
            AcplInterpolationType::Steep
        ),
        param_timeslots: &data.framing.param_timeslots,
    };
    let out = run_pseudocode_115_pair(&mut state.cpe, frame);
    // Inverse QMF synthesis for both output channels.
    let mut syn0 = QmfSynthesisBank::new();
    let mut syn1 = QmfSynthesisBank::new();
    let mut left = Vec::with_capacity(pcm_in.len());
    let mut right = Vec::with_capacity(pcm_in.len());
    for ts in 0..n_slots {
        left.extend_from_slice(&syn0.process_slot(&out.z0[ts]));
        right.extend_from_slice(&syn1.process_slot(&out.z1[ts]));
    }
    Some((left, right))
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acpl::AcplHuffParam;

    // ---------------- §5.7.7.7 dequantisation tables -----------------

    #[test]
    fn alpha_dq_fine_anchors_table_203() {
        // Table 203 anchors: index 0 = -2.0, index 16 = 0.0, index 32 = +2.0
        assert_eq!(ALPHA_DQ_FINE[0], -2.000000);
        assert_eq!(ALPHA_DQ_FINE[16], 0.000000);
        assert_eq!(ALPHA_DQ_FINE[32], 2.000000);
        // Anti-symmetry around the centre: alpha[16+k] = -alpha[16-k].
        for k in 1..=16 {
            let lo = ALPHA_DQ_FINE[16 - k];
            let hi = ALPHA_DQ_FINE[16 + k];
            assert!((lo + hi).abs() < 1e-6, "k={k} lo={lo} hi={hi}");
        }
    }

    #[test]
    fn alpha_dq_coarse_anchors_table_205() {
        assert_eq!(ALPHA_DQ_COARSE[0], -2.000000);
        assert_eq!(ALPHA_DQ_COARSE[8], 0.000000);
        assert_eq!(ALPHA_DQ_COARSE[16], 2.000000);
        for k in 1..=8 {
            let lo = ALPHA_DQ_COARSE[8 - k];
            let hi = ALPHA_DQ_COARSE[8 + k];
            assert!((lo + hi).abs() < 1e-6, "k={k} lo={lo} hi={hi}");
        }
    }

    #[test]
    fn ibeta_fine_table_203_column2() {
        assert_eq!(IBETA_FINE[0], 0);
        assert_eq!(IBETA_FINE[8], 8);
        assert_eq!(IBETA_FINE[16], 0);
        assert_eq!(IBETA_FINE[24], 8);
        assert_eq!(IBETA_FINE[32], 0);
    }

    #[test]
    fn ibeta_coarse_table_205_column2() {
        assert_eq!(IBETA_COARSE[0], 0);
        assert_eq!(IBETA_COARSE[4], 4);
        assert_eq!(IBETA_COARSE[8], 0);
        assert_eq!(IBETA_COARSE[12], 4);
        assert_eq!(IBETA_COARSE[16], 0);
    }

    #[test]
    fn beta_dq_fine_anchors_table_204() {
        // beta_q=0 row is all zeros.
        for col in 0..9 {
            assert_eq!(BETA_DQ_FINE[0][col], 0.0);
        }
        // Anchor: beta_q=8, ibeta=0 → 4.0; ibeta=8 → 1.0.
        assert_eq!(BETA_DQ_FINE[8][0], 4.0);
        assert_eq!(BETA_DQ_FINE[8][8], 1.0);
        // Strict monotone-decrease across ibeta for any non-zero beta_q.
        for q in 1..=8 {
            for c in 1..9 {
                assert!(
                    BETA_DQ_FINE[q][c] < BETA_DQ_FINE[q][c - 1],
                    "beta_q={q} col={c}"
                );
            }
        }
    }

    #[test]
    fn beta_dq_coarse_anchors_table_206() {
        assert_eq!(BETA_DQ_COARSE[0][0], 0.0);
        assert_eq!(BETA_DQ_COARSE[4][0], 4.0);
        assert_eq!(BETA_DQ_COARSE[4][4], 1.0);
        for q in 1..=4 {
            for c in 1..5 {
                assert!(
                    BETA_DQ_COARSE[q][c] < BETA_DQ_COARSE[q][c - 1],
                    "beta_q={q} col={c}"
                );
            }
        }
    }

    #[test]
    fn beta3_and_gamma_deltas_match_tables_207_208() {
        assert_eq!(beta3_delta(AcplQuantMode::Fine), 0.125);
        assert_eq!(beta3_delta(AcplQuantMode::Coarse), 0.25);
        assert!((gamma_delta(AcplQuantMode::Fine) - 1638.0 / 16384.0).abs() < 1e-9);
        assert!((gamma_delta(AcplQuantMode::Coarse) - 3276.0 / 16384.0).abs() < 1e-9);
    }

    #[test]
    fn dequantize_alpha_fine_round_trips_through_signed_index() {
        // alpha_q = 0 (Huffman F0 cb_off=16 → recovered i32=0) → lane
        // 16 → +0.0.
        let (a, ib) = dequantize_alpha_index(AcplQuantMode::Fine, 0);
        assert_eq!(a, 0.0);
        assert_eq!(ib, 0);
        // alpha_q = -16 (lane 0) → -2.0.
        let (a, ib) = dequantize_alpha_index(AcplQuantMode::Fine, -16);
        assert_eq!(a, -2.000000);
        assert_eq!(ib, 0);
        // alpha_q = +8 (lane 24) → +1.0.
        let (a, ib) = dequantize_alpha_index(AcplQuantMode::Fine, 8);
        assert_eq!(a, 1.000000);
        assert_eq!(ib, 8);
    }

    #[test]
    fn dequantize_beta_fine_uses_ibeta_column() {
        // beta_q=+1, ibeta=0 → +0.2375.
        let v = dequantize_beta_index(AcplQuantMode::Fine, 1, 0);
        assert!((v - 0.2375).abs() < 1e-6);
        // beta_q=-1, ibeta=0 → -0.2375.
        let v = dequantize_beta_index(AcplQuantMode::Fine, -1, 0);
        assert!((v + 0.2375).abs() < 1e-6);
        // beta_q=+8, ibeta=8 → +1.0.
        let v = dequantize_beta_index(AcplQuantMode::Fine, 8, 8);
        assert!((v - 1.0).abs() < 1e-6);
    }

    // ----------------- §5.7.7.7 differential decode ------------------

    #[test]
    fn differential_decode_freq_accumulates_from_seed() {
        // DIFF_FREQ: row[0]=values[0], row[i]=row[i-1]+values[i].
        let p = AcplHuffParam {
            values: vec![5, 1, -2, 3],
            direction_time: false,
        };
        let mut st = AcplDiffState::new();
        let out = differential_decode(&[p], 4, &mut st);
        assert_eq!(out, vec![vec![5, 6, 4, 7]]);
        assert_eq!(st.q_prev, vec![5, 6, 4, 7]);
    }

    #[test]
    fn differential_decode_time_uses_prev_then_carries() {
        // First set DIFF_FREQ → seed q_prev. Second set DIFF_TIME → adds
        // to q_prev band-by-band.
        let p1 = AcplHuffParam {
            values: vec![1, 2, 3],
            direction_time: false,
        };
        let p2 = AcplHuffParam {
            values: vec![10, -1, 0],
            direction_time: true,
        };
        let mut st = AcplDiffState::new();
        let out = differential_decode(&[p1, p2], 3, &mut st);
        // p1 → [1, 3, 6]
        // p2 (DIFF_TIME) → [1+10, 3-1, 6+0] = [11, 2, 6]
        assert_eq!(out, vec![vec![1, 3, 6], vec![11, 2, 6]]);
        assert_eq!(st.q_prev, vec![11, 2, 6]);
    }

    #[test]
    fn differential_decode_carries_across_frames() {
        // Frame 1 sets q_prev = [4, 5]. Frame 2 starts with DIFF_TIME
        // and should pick up q_prev from frame 1.
        let mut st = AcplDiffState::new();
        let p1 = AcplHuffParam {
            values: vec![4, 1],
            direction_time: false,
        };
        let _ = differential_decode(&[p1], 2, &mut st);
        assert_eq!(st.q_prev, vec![4, 5]);
        let p2 = AcplHuffParam {
            values: vec![1, -2],
            direction_time: true,
        };
        let out2 = differential_decode(&[p2], 2, &mut st);
        assert_eq!(out2, vec![vec![5, 3]]);
        assert_eq!(st.q_prev, vec![5, 3]);
    }

    // -------------- §5.7.7.7 dequantize_alpha_beta -------------------

    #[test]
    fn dequantize_alpha_beta_returns_two_synced_matrices() {
        // 1 param set, 2 bands. alpha_q = [0, 8] (fine, lanes 16+0=16
        // and 16+8=24 → 0.0 and 1.0; ibetas 0 and 8). beta_q = [+1, -1]
        // → +0.2375 (col 0), -0.0593750 (col 8).
        let alpha_q = vec![vec![0i32, 8]];
        let beta_q = vec![vec![1i32, -1]];
        let (a, b) = dequantize_alpha_beta(&alpha_q, &beta_q, AcplQuantMode::Fine);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].len(), 2);
        assert!((a[0][0] - 0.0).abs() < 1e-6);
        assert!((a[0][1] - 1.0).abs() < 1e-6);
        assert!((b[0][0] - 0.2375).abs() < 1e-6);
        assert!((b[0][1] + 0.0593750).abs() < 1e-6);
    }

    // ---------------- §5.7.7.3 interpolate ---------------------------

    #[test]
    fn interpolate_smooth_one_param_set_linear_ramp() {
        // 1 param set, prev=0.0, p=4.0, num_ts=4 → ts=0 → 1.0, ts=3 → 4.0
        let p_set = vec![vec![4.0f32; NUM_QMF_SUBBANDS]];
        let prev = vec![0.0f32; NUM_QMF_SUBBANDS];
        let inp = InterpInputs {
            by_pset: &p_set,
            prev: &prev,
        };
        for ts in 0..4u32 {
            let v = interpolate(&inp, 1, 0, ts, 4, false, &[]);
            let expected = (ts + 1) as f32;
            assert!(
                (v - expected).abs() < 1e-6,
                "ts={ts} v={v} expected={expected}"
            );
        }
    }

    #[test]
    fn interpolate_steep_falls_to_constants() {
        // 1 param set, steep, ts < timeslot[0] → prev, else current.
        let p_set = vec![vec![5.0f32; NUM_QMF_SUBBANDS]];
        let prev = vec![1.0f32; NUM_QMF_SUBBANDS];
        let inp = InterpInputs {
            by_pset: &p_set,
            prev: &prev,
        };
        // boundary at ts=2.
        assert_eq!(interpolate(&inp, 1, 0, 0, 4, true, &[2]), 1.0);
        assert_eq!(interpolate(&inp, 1, 0, 1, 4, true, &[2]), 1.0);
        assert_eq!(interpolate(&inp, 1, 0, 2, 4, true, &[2]), 5.0);
        assert_eq!(interpolate(&inp, 1, 0, 3, 4, true, &[2]), 5.0);
    }

    #[test]
    fn interpolate_smooth_two_sets_meets_at_midpoint() {
        // 2 sets: prev=0, p0=4, p1=8, num_ts=8 → ts_2=4.
        // ts in [0,4): linear 0→4 over ts_2=4 → ts=0 → 1, ts=3 → 4.
        // ts in [4,8): linear 4→8 over (8-4)=4 → ts=4 → 5, ts=7 → 8.
        let p_set = vec![
            vec![4.0f32; NUM_QMF_SUBBANDS],
            vec![8.0f32; NUM_QMF_SUBBANDS],
        ];
        let prev = vec![0.0f32; NUM_QMF_SUBBANDS];
        let inp = InterpInputs {
            by_pset: &p_set,
            prev: &prev,
        };
        for ts in 0..8u32 {
            let v = interpolate(&inp, 2, 0, ts, 8, false, &[]);
            let expected = (ts + 1) as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "ts={ts} v={v} expected={expected}"
            );
        }
    }

    // ---------------- expand_pb_to_sb -------------------------------

    #[test]
    fn expand_pb_to_sb_preserves_first_pb_for_low_subbands_15() {
        // Table 197 column 15: sb=0 → pb=0, sb=8 → pb=8, sb=63 → pb=14.
        let pb = vec![vec![
            0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        ]];
        let sb = expand_pb_to_sb(&pb, 15);
        assert_eq!(sb[0][0], 0.0);
        assert_eq!(sb[0][8], 8.0);
        assert_eq!(sb[0][9], 9.0);
        assert_eq!(sb[0][14], 11.0); // sb=14 → pb=11 (rows 14-17 → 11)
        assert_eq!(sb[0][63], 14.0);
    }

    // ----------------- §5.7.7.4.2 Decorrelator -----------------------

    #[test]
    fn region_delay_filter_length_match_table_198() {
        assert_eq!(region_delay(0), 7);
        assert_eq!(region_filter_length(0), 7);
        assert_eq!(region_delay(6), 7);
        assert_eq!(region_filter_length(6), 7);
        assert_eq!(region_delay(7), 10);
        assert_eq!(region_filter_length(7), 4);
        assert_eq!(region_delay(22), 10);
        assert_eq!(region_filter_length(22), 4);
        assert_eq!(region_delay(23), 12);
        assert_eq!(region_filter_length(23), 2);
        assert_eq!(region_delay(63), 12);
        assert_eq!(region_filter_length(63), 2);
    }

    #[test]
    fn region_coeffs_match_tables_199_200_201_first_row() {
        assert_eq!(region_coeffs(DecorrelatorId::D0, 0)[0], 1.0);
        assert!((region_coeffs(DecorrelatorId::D0, 0)[1] - 0.5306).abs() < 1e-12);
        assert!((region_coeffs(DecorrelatorId::D1, 7)[1] - 0.0425).abs() < 1e-12);
        assert!((region_coeffs(DecorrelatorId::D2, 23)[1] + 0.6057).abs() < 1e-12);
    }

    #[test]
    fn input_signal_modifier_zero_input_zero_output() {
        let mut m = InputSignalModifier::new(DecorrelatorId::D0);
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            let y = m.process_sample(sb, (0.0, 0.0));
            assert_eq!(y, (0.0, 0.0));
        }
    }

    #[test]
    fn input_signal_modifier_impulse_eventually_yields_nonzero() {
        // Push an impulse at ts=0 into sb=0 and run for >7+7=14 slots.
        // The IIR response should produce non-zero output by ts ≥ delay.
        let mut m = InputSignalModifier::new(DecorrelatorId::D0);
        let mut energy = 0.0f64;
        for ts in 0..32 {
            let x = if ts == 0 { (1.0f32, 0.0) } else { (0.0, 0.0) };
            let y = m.process_sample(0, x);
            energy += (y.0 as f64).powi(2) + (y.1 as f64).powi(2);
        }
        assert!(energy > 0.0, "decorrelator should emit non-zero IIR tail");
    }

    // ----------------- §5.7.7.4.3 Transient ducker -------------------

    #[test]
    fn transient_ducker_passes_silence_unmodified() {
        let mut d = TransientDucker::new();
        let p_e = [0.0f32; ACPL_MAX_NUM_PARAM_BANDS];
        let g = d.update(&p_e);
        for v in g.iter() {
            assert_eq!(*v, 1.0, "silence should yield gain 1.0 (no transient)");
        }
    }

    #[test]
    fn transient_ducker_attenuates_after_peak() {
        // Push a unit-energy spike followed by zero — the smoothed
        // decay should trigger ducking on the second slot.
        let mut d = TransientDucker::new();
        let mut spike = [0.0f32; ACPL_MAX_NUM_PARAM_BANDS];
        spike[0] = 1.0;
        let _ = d.update(&spike);
        // Now silence — peak_decay tail still > smooth, gamma branch
        // should fire and pull duck_gain[0] below 1.0.
        let zeros = [0.0f32; ACPL_MAX_NUM_PARAM_BANDS];
        let g = d.update(&zeros);
        assert!(g[0] < 1.0, "ducker should attenuate the post-peak slot");
        assert!(g[0] > 0.0);
    }

    #[test]
    fn compute_p_energy_aggregates_per_param_band() {
        // Subband 0 → pb 0 in 15-band config. Magnitude 3 at sb 0 →
        // p_energy[0] = 9.
        let mut x = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        x[0] = (3.0, 0.0);
        x[1] = (0.0, 4.0);
        let e = compute_p_energy(&x, 15);
        assert_eq!(e[0], 9.0);
        assert_eq!(e[1], 16.0);
        for pb in 2..ACPL_MAX_NUM_PARAM_BANDS {
            assert_eq!(e[pb], 0.0);
        }
    }

    #[test]
    fn apply_transient_ducker_scales_per_pb() {
        let x = [(2.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        let mut g = [1.0f32; ACPL_MAX_NUM_PARAM_BANDS];
        g[0] = 0.5;
        let out = apply_transient_ducker(&x, &g, 15);
        // sb 0 maps to pb 0 (gain 0.5) → 1.0, sb 1 maps to pb 1 (gain
        // 1.0) → 2.0.
        assert_eq!(out[0].0, 1.0);
        assert_eq!(out[1].0, 2.0);
    }

    // ----------------- §5.7.7.5 acpl_module --------------------------

    /// Synthetic stereo A-CPL frame: small sine signal in x0, zero in x1
    /// (ASPX_ACPL_2 mode), modest alpha/beta values, single param set.
    /// After running the synthesis we expect:
    ///   1) The output to differ from the input (cross-channel mixing
    ///      happened above acpl_qmf_band).
    ///   2) Below acpl_qmf_band the output is the M/S split of x0/x1.
    ///   3) The output stays within reasonable bounds (no NaN / inf).
    #[test]
    fn acpl_module_mixes_channels_above_qmf_band_aspx_acpl_2() {
        let num_ts = 32usize;
        let num_pb = 15u32;
        // Build x0 with a tone in subband 12 and 30, x1 silent.
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let zero = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for ts in 0..num_ts {
            let phase = (ts as f32) * 0.5;
            x0[ts][12] = (phase.cos(), phase.sin());
            x0[ts][30] = (0.5 * phase.cos(), 0.5 * phase.sin());
        }
        // Synthetic decorrelator output y = x0 (just for the math test).
        let y: Vec<_> = x0.clone();
        // alpha = +0.5 across bands, beta = +0.5 across bands.
        let alpha_dq = vec![vec![0.5f32; num_pb as usize]];
        let beta_dq = vec![vec![0.5f32; num_pb as usize]];
        let frame = AcplCpeFrame {
            x0: &x0,
            x1: None,
            alpha_dq: &alpha_dq,
            beta_dq: &beta_dq,
            num_param_bands: num_pb,
            acpl_qmf_band: 8, // M/S below sb=8, mixed above
            steep: false,
            param_timeslots: &[],
        };
        let out = acpl_module(&frame, &y);
        assert_eq!(out.z0.len(), num_ts);
        assert_eq!(out.z1.len(), num_ts);
        // Above acpl_qmf_band, with x1=0 and y=x0, the formula is
        //   z0 = 0.5 * (x0 * (1+a) + x0 * b) = 0.5 * x0 * (1+a+b)
        // With smooth interp, num_pset=1, prev=0, p=0.5:
        //   interp(ts) = (ts+1)/num_ts * 0.5
        // At ts=0 → interp = 0.5/32 ≈ 0.0156, scale = 0.5*(1 + 2*0.0156)
        //          = 0.5156 → z0 = 0.5156 * x0 ≠ x0.
        // At ts=num_ts-1 → interp = 0.5, scale = 1.0 → z0 = x0 (boundary).
        let early = 0u32;
        let interp_early = 0.5_f32 * ((early + 1) as f32) / (num_ts as f32);
        let scale_early = 0.5_f32 * (1.0 + 2.0 * interp_early);
        let (z0r, z0i) = out.z0[early as usize][12];
        let (x0r, x0i) = x0[early as usize][12];
        let exp_re = scale_early * x0r;
        let exp_im = scale_early * x0i;
        assert!(
            (z0r - exp_re).abs() < 1e-5,
            "z0r {z0r} vs {exp_re} (scale {scale_early})"
        );
        assert!((z0i - exp_im).abs() < 1e-5, "z0i {z0i} vs {exp_im}");
        // Demonstrate divergence: at ts=0 z0 ≠ x0 because scale ≠ 1.0.
        assert!(
            (z0r - x0r).abs() > 1e-4 || (z0i - x0i).abs() > 1e-4,
            "z0[early][12] should differ from x0 (mixing scale != 1.0) — got z0=({z0r},{z0i}) x0=({x0r},{x0i}) scale={scale_early}"
        );
        // Below acpl_qmf_band (sb < 8): with x1 = 0, z0 = 0.5*x0,
        // z1 = 0.5*x0.
        for sb in 0..8 {
            for ts in 0..num_ts {
                let (z0r, z0i) = out.z0[ts][sb];
                let (z1r, z1i) = out.z1[ts][sb];
                assert!((z0r - 0.5 * x0[ts][sb].0).abs() < 1e-6);
                assert!((z0i - 0.5 * x0[ts][sb].1).abs() < 1e-6);
                assert!((z1r - 0.5 * x0[ts][sb].0).abs() < 1e-6);
                assert!((z1i - 0.5 * x0[ts][sb].1).abs() < 1e-6);
            }
        }
        // No NaN / Inf in output.
        for ts in 0..num_ts {
            for sb in 0..NUM_QMF_SUBBANDS {
                assert!(out.z0[ts][sb].0.is_finite());
                assert!(out.z0[ts][sb].1.is_finite());
                assert!(out.z1[ts][sb].0.is_finite());
                assert!(out.z1[ts][sb].1.is_finite());
            }
        }
        // Silence the unused y matrix to suppress warnings.
        let _ = zero;
    }

    /// ASPX_ACPL_1: x1 ≠ 0 — stereo input. Verify the M/S split below
    /// `acpl_qmf_band` correctly produces (x0+x1)/2 and (x0-x1)/2.
    #[test]
    fn acpl_module_below_qmf_band_is_mid_side_split() {
        let num_ts = 8usize;
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x1 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            x0[ts][2] = (1.0, 0.0);
            x1[ts][2] = (0.0, 1.0);
        }
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let alpha_dq = vec![vec![0.0f32; 15]];
        let beta_dq = vec![vec![0.0f32; 15]];
        let frame = AcplCpeFrame {
            x0: &x0,
            x1: Some(&x1),
            alpha_dq: &alpha_dq,
            beta_dq: &beta_dq,
            num_param_bands: 15,
            acpl_qmf_band: 4,
            steep: false,
            param_timeslots: &[],
        };
        let out = acpl_module(&frame, &y);
        for ts in 0..num_ts {
            let z0 = out.z0[ts][2];
            let z1 = out.z1[ts][2];
            // (1+0j + 0+1j)/2 = (0.5, 0.5)
            assert!((z0.0 - 0.5).abs() < 1e-6);
            assert!((z0.1 - 0.5).abs() < 1e-6);
            // (1+0j - (0+1j))/2 = (0.5, -0.5)
            assert!((z1.0 - 0.5).abs() < 1e-6);
            assert!((z1.1 + 0.5).abs() < 1e-6);
        }
    }

    /// End-to-end check: run the full Pseudocode 115 pipeline (decorr
    /// + ducker + acpl_module) on a non-trivial frame. Verify nothing
    ///   blows up and the output differs from passthrough.
    #[test]
    fn run_pseudocode_115_pair_end_to_end() {
        let num_ts = 16usize;
        let num_pb = 9u32;
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            let phase = (ts as f32) * 0.3;
            for sb in 0..32 {
                x0[ts][sb] = (
                    0.1 * (phase + sb as f32 * 0.1).cos(),
                    0.1 * (phase + sb as f32 * 0.1).sin(),
                );
            }
        }
        let alpha_dq = vec![vec![0.3f32; num_pb as usize]];
        let beta_dq = vec![vec![0.3f32; num_pb as usize]];
        let mut state = AcplCpeState::new(DecorrelatorId::D0);
        let frame = AcplCpeFrame {
            x0: &x0,
            x1: None,
            alpha_dq: &alpha_dq,
            beta_dq: &beta_dq,
            num_param_bands: num_pb,
            acpl_qmf_band: 4,
            steep: false,
            param_timeslots: &[],
        };
        let out = run_pseudocode_115_pair(&mut state, frame);
        assert_eq!(out.z0.len(), num_ts);
        assert_eq!(out.z1.len(), num_ts);
        // Output should differ from the passthrough below acpl_qmf_band
        // (we doubled x0 inside).
        let mut diff_energy = 0.0f64;
        for ts in 0..num_ts {
            for sb in 0..NUM_QMF_SUBBANDS {
                assert!(out.z0[ts][sb].0.is_finite());
                assert!(out.z0[ts][sb].1.is_finite());
                assert!(out.z1[ts][sb].0.is_finite());
                assert!(out.z1[ts][sb].1.is_finite());
                let dr = out.z0[ts][sb].0 - x0[ts][sb].0;
                let di = out.z0[ts][sb].1 - x0[ts][sb].1;
                diff_energy += (dr as f64).powi(2) + (di as f64).powi(2);
            }
        }
        assert!(diff_energy > 0.0, "output should diverge from x0");
        // After a frame, prev arrays should be populated.
        assert_eq!(state.alpha_prev_sb.len(), NUM_QMF_SUBBANDS);
        assert_eq!(state.beta_prev_sb.len(), NUM_QMF_SUBBANDS);
        // alpha_dq=0.3 across pb → expand_pb_to_sb fills sb with 0.3.
        for sb in 0..NUM_QMF_SUBBANDS {
            assert!((state.alpha_prev_sb[sb] - 0.3).abs() < 1e-6);
            assert!((state.beta_prev_sb[sb] - 0.3).abs() < 1e-6);
        }
    }

    /// End-to-end: alpha=0, beta=0 with x1 = x0 → above acpl_qmf_band
    /// the formula collapses to z0 = 0.5*x0in = x0 (since x0in = 2*x0).
    #[test]
    fn acpl_module_zero_params_passthrough_above_qmf_band() {
        let num_ts = 4usize;
        let mut x0in = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            x0in[ts][20] = (2.0, 1.0); // already-doubled x0
        }
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let alpha_dq = vec![vec![0.0f32; 15]];
        let beta_dq = vec![vec![0.0f32; 15]];
        let frame = AcplCpeFrame {
            x0: &x0in,
            x1: None,
            alpha_dq: &alpha_dq,
            beta_dq: &beta_dq,
            num_param_bands: 15,
            acpl_qmf_band: 8,
            steep: false,
            param_timeslots: &[],
        };
        let out = acpl_module(&frame, &y);
        // Above qmf_band (sb=20), a=b=0, y=0 → z0 = 0.5*x0*(1+0) = 0.5*x0in
        // = (1.0, 0.5)
        for ts in 0..num_ts {
            let z = out.z0[ts][20];
            assert!((z.0 - 1.0).abs() < 1e-6);
            assert!((z.1 - 0.5).abs() < 1e-6);
            // z1 = 0.5*x0*(1-0) = 0.5*x0 = (1.0, 0.5)
            let z1 = out.z1[ts][20];
            assert!((z1.0 - 1.0).abs() < 1e-6);
            assert!((z1.1 - 0.5).abs() < 1e-6);
        }
    }

    // ----------------- run_acpl_1ch_pcm — PCM end-to-end --------------

    #[test]
    fn run_acpl_1ch_pcm_emits_two_channel_pcm_with_energy() {
        // End-to-end smoke test: feed a synthetic mono sine into the
        // §5.7.7.5 channel-pair pipeline (QMF analysis → A-CPL → QMF
        // synthesis) and assert both output channels carry energy and
        // come out the right length.
        use crate::acpl::{
            Acpl1chMode, AcplConfig1ch, AcplData1ch, AcplFramingData, AcplHuffParam,
            AcplInterpolationType, AcplQuantMode,
        };
        let _ = Acpl1chMode::Full; // silence unused-import warning
        let n_slots = 32usize;
        let n = n_slots * NUM_QMF_SUBBANDS;
        let mut pcm = vec![0.0f32; n];
        let f = 440.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let cfg = AcplConfig1ch {
            num_param_bands_id: 0,
            num_param_bands: 15,
            quant_mode: AcplQuantMode::Fine,
            qmf_band: 0,
        };
        // Hand-built parsed acpl_data_1ch — single param set, alpha
        // and beta carrying the F0 anchor index (huffman F0 cb_off
        // gives signed 0 here).
        let alpha = AcplHuffParam {
            values: vec![4i32; cfg.num_param_bands as usize],
            direction_time: false,
        };
        let beta = AcplHuffParam {
            values: vec![2i32; cfg.num_param_bands as usize],
            direction_time: false,
        };
        let data = AcplData1ch {
            framing: AcplFramingData {
                interpolation_type: AcplInterpolationType::Smooth,
                num_param_sets_cod: 0,
                num_param_sets: 1,
                param_timeslots: vec![],
            },
            alpha1: vec![alpha],
            beta1: vec![beta],
        };
        let mut state = AcplSubstreamState::new();
        let (left, right) = run_acpl_1ch_pcm(&pcm, &cfg, &data, &mut state).expect("synth runs");
        assert_eq!(left.len(), pcm.len());
        assert_eq!(right.len(), pcm.len());
        // Both channels should carry energy in the steady-state tail.
        let start = 1024usize;
        let e_l: f64 = left[start..].iter().map(|&s| (s as f64).powi(2)).sum();
        let e_r: f64 = right[start..].iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(e_l > 1e-6, "left channel silent (e={e_l})");
        assert!(e_r > 1e-6, "right channel silent (e={e_r})");
        // The two channels must differ — a non-zero alpha drives them
        // apart.
        let mut diffs = 0usize;
        for (l, r) in left[start..].iter().zip(right[start..].iter()) {
            if (l - r).abs() > 1e-6 {
                diffs += 1;
            }
        }
        assert!(
            diffs > (left.len() - start) / 4,
            "channels too similar (diffs={diffs})"
        );
    }

    #[test]
    fn run_acpl_1ch_pcm_rejects_misaligned_pcm() {
        use crate::acpl::{
            AcplConfig1ch, AcplData1ch, AcplFramingData, AcplInterpolationType, AcplQuantMode,
        };
        let pcm = vec![0.0f32; 65]; // not a multiple of 64
        let cfg = AcplConfig1ch {
            num_param_bands_id: 0,
            num_param_bands: 15,
            quant_mode: AcplQuantMode::Fine,
            qmf_band: 0,
        };
        let data = AcplData1ch {
            framing: AcplFramingData {
                interpolation_type: AcplInterpolationType::Smooth,
                num_param_sets_cod: 0,
                num_param_sets: 1,
                param_timeslots: vec![],
            },
            alpha1: vec![],
            beta1: vec![],
        };
        let mut state = AcplSubstreamState::new();
        assert!(run_acpl_1ch_pcm(&pcm, &cfg, &data, &mut state).is_none());
    }
}
