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
//! * **§5.7.7.6.2 Multichannel `ASPX_ACPL_3`** — [`transform`]
//!   (Pseudocode 119), [`acpl_module2`] / [`acpl_module3`] (also
//!   Pseudocode 119) and [`run_pseudocode_118_5x`] (Pseudocode 118)
//!   synthesise the five output channels (`z0..z4`) from the two A-CPL
//!   carrier channels (`x0`, `x1`), the centre passthrough `x2`, and the
//!   six gamma matrices `g1..g6`. The three parallel decorrelator paths
//!   (D0/D1/D2) are driven by `Transform()`-mixed inputs and routed
//!   through `ACplModule2` (gamma+alpha+beta combiner) followed by
//!   `ACplModule3` (beta3 cross-residual term) per the spec.
//!
//! What is NOT covered yet (TS 103 190-1):
//!
//! * §5.7.7.6.1 ASPX_ACPL_1 / ASPX_ACPL_2 multichannel wrapper
//!   (Pseudocode 117) — uses two parallel `ACplModule`s with `y0/y1`
//!   driven by D0/D1; the building blocks are all here, but the 5-input
//!   wrapper still needs wiring.
//! * Hooking the multichannel synthesis into the frame-level decoder
//!   pipeline (the asf walker still parses `acpl_data_*` but the 5_X
//!   walker doesn't yet drive [`run_pseudocode_118_5x`]).

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
// §5.7.7.6.2 Pseudocodes 118 / 119 — ASPX_ACPL_3 multichannel synthesis
// =====================================================================

/// Per-channel QMF matrix shape used throughout §5.7.7.6.2: a vector of
/// per-timeslot `[ (re, im); NUM_QMF_SUBBANDS ]` columns.
pub type AcplQmfMatrix = Vec<[(f32, f32); NUM_QMF_SUBBANDS]>;

/// `Transform(g1, g2, num_pset, x0, x1)` per §5.7.7.6.2 Pseudocode 119.
///
/// Linearly mixes the two A-CPL carrier channels (`x0`, `x1`) by the
/// interpolated gamma matrices `g1`, `g2`:
///
/// ```text
///   v[ts][sb] = x0[ts][sb] * interp_g1[ts][sb]
///             + x1[ts][sb] * interp_g2[ts][sb]
/// ```
///
/// `g1_pb` / `g2_pb` are per-`(pset, pb)` matrices; we fan them out to
/// per-subband via [`expand_pb_to_sb`] and then the §5.7.7.3 [`interpolate`]
/// call walks them across timeslots. `prev_g1` / `prev_g2` carry the
/// previous frame's last-set `[sb]` row (zero on the first frame, per the
/// spec).
///
/// The output is `[ts][sb]` shaped, matching the rest of the §5.7.7
/// per-slot interfaces.
#[allow(clippy::too_many_arguments)]
pub fn transform(
    x0: &[[(f32, f32); NUM_QMF_SUBBANDS]],
    x1: &[[(f32, f32); NUM_QMF_SUBBANDS]],
    g1_pb: &[Vec<f32>],
    g2_pb: &[Vec<f32>],
    num_param_bands: u32,
    prev_g1: &[f32],
    prev_g2: &[f32],
    steep: bool,
    param_timeslots: &[u8],
) -> Vec<[(f32, f32); NUM_QMF_SUBBANDS]> {
    let num_ts = x0.len();
    debug_assert_eq!(x1.len(), num_ts);
    let num_pset = g1_pb.len() as u32;
    let g1_sb = expand_pb_to_sb(g1_pb, num_param_bands);
    let g2_sb = expand_pb_to_sb(g2_pb, num_param_bands);
    let g1_inp = InterpInputs {
        by_pset: &g1_sb,
        prev: prev_g1,
    };
    let g2_inp = InterpInputs {
        by_pset: &g2_sb,
        prev: prev_g2,
    };
    let mut out = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let mut v_col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            let g1 = interpolate(
                &g1_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let g2 = interpolate(
                &g2_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let (x0r, x0i) = x0[ts][sb as usize];
            let (x1r, x1i) = x1[ts][sb as usize];
            v_col[sb as usize] = (x0r * g1 + x1r * g2, x0i * g1 + x1i * g2);
        }
        out.push(v_col);
    }
    out
}

/// `ACplModule2(g1, g2, a, b, num_pset, x0, x1, y)` per §5.7.7.6.2
/// Pseudocode 119.
///
/// Builds the (z0, z1) channel pair from the two A-CPL carrier inputs,
/// the gamma + alpha + beta parameter matrices, and the decorrelator
/// output `y`:
///
/// ```text
///   z0 = 0.5*(x0*(g1+g1*a) + x1*(g2+g2*a) + y*b)
///   z1 = 0.5*(x0*(g1-g1*a) + x1*(g2-g2*a) - y*b)
/// ```
///
/// The interpolations are taken on the full `g1`, `g2`, `g1*a`, `g2*a`
/// and `b` per-`pb` matrices (computed from the dequantised arrays
/// before the call). Per the spec the interpolations are computed on
/// the products `g*a` (not on `g` and `a` separately) because that's
/// the point at which time-interpolation must be evaluated.
///
/// `prev_*` are the previous-frame `[sb]` rows for each interpolated
/// matrix (zero on the first frame).
#[allow(clippy::too_many_arguments)]
pub fn acpl_module2(
    x0: &[[(f32, f32); NUM_QMF_SUBBANDS]],
    x1: &[[(f32, f32); NUM_QMF_SUBBANDS]],
    y: &[[(f32, f32); NUM_QMF_SUBBANDS]],
    g1_pb: &[Vec<f32>],
    g2_pb: &[Vec<f32>],
    g1a_pb: &[Vec<f32>],
    g2a_pb: &[Vec<f32>],
    b_pb: &[Vec<f32>],
    num_param_bands: u32,
    steep: bool,
    param_timeslots: &[u8],
) -> (AcplQmfMatrix, AcplQmfMatrix) {
    let num_ts = x0.len();
    debug_assert_eq!(x1.len(), num_ts);
    debug_assert_eq!(y.len(), num_ts);
    let num_pset = g1_pb.len() as u32;
    let g1_sb = expand_pb_to_sb(g1_pb, num_param_bands);
    let g2_sb = expand_pb_to_sb(g2_pb, num_param_bands);
    let g1a_sb = expand_pb_to_sb(g1a_pb, num_param_bands);
    let g2a_sb = expand_pb_to_sb(g2a_pb, num_param_bands);
    let b_sb = expand_pb_to_sb(b_pb, num_param_bands);
    let zero_prev = vec![0.0f32; NUM_QMF_SUBBANDS];
    let g1_inp = InterpInputs {
        by_pset: &g1_sb,
        prev: &zero_prev,
    };
    let g2_inp = InterpInputs {
        by_pset: &g2_sb,
        prev: &zero_prev,
    };
    let g1a_inp = InterpInputs {
        by_pset: &g1a_sb,
        prev: &zero_prev,
    };
    let g2a_inp = InterpInputs {
        by_pset: &g2a_sb,
        prev: &zero_prev,
    };
    let b_inp = InterpInputs {
        by_pset: &b_sb,
        prev: &zero_prev,
    };

    let mut z0_out = Vec::with_capacity(num_ts);
    let mut z1_out = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let mut z0_col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        let mut z1_col = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            let g1 = interpolate(
                &g1_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let g2 = interpolate(
                &g2_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let g1a = interpolate(
                &g1a_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let g2a = interpolate(
                &g2a_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let b = interpolate(
                &b_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let sb_i = sb as usize;
            let (x0r, x0i) = x0[ts][sb_i];
            let (x1r, x1i) = x1[ts][sb_i];
            let (yr, yi) = y[ts][sb_i];
            z0_col[sb_i] = (
                0.5 * (x0r * (g1 + g1a) + x1r * (g2 + g2a) + yr * b),
                0.5 * (x0i * (g1 + g1a) + x1i * (g2 + g2a) + yi * b),
            );
            z1_col[sb_i] = (
                0.5 * (x0r * (g1 - g1a) + x1r * (g2 - g2a) - yr * b),
                0.5 * (x0i * (g1 - g1a) + x1i * (g2 - g2a) - yi * b),
            );
        }
        z0_out.push(z0_col);
        z1_out.push(z1_col);
    }
    (z0_out, z1_out)
}

/// `ACplModule3(b3, a, num_pset, z0, z1, y2)` per §5.7.7.6.2
/// Pseudocode 119.
///
/// Adds a cross-residual decorrelator term to an existing `(z0, z1)`
/// pair using `beta3` and the alpha matrix:
///
/// ```text
///   z0 += 0.25 * y2 * (b3 + b3*a)
///   z1 += 0.25 * y2 * (b3 - b3*a)
/// ```
///
/// The dequantised `b3_pb` and `b3a_pb` per-`(pset, pb)` matrices are
/// fanned out to subbands via [`expand_pb_to_sb`] and run through
/// [`interpolate`] across timeslots. The `(z0, z1)` slices are mutated
/// in-place to match the spec's `z0[ts][sb] += ...` form.
#[allow(clippy::too_many_arguments)]
pub fn acpl_module3(
    z0: &mut [[(f32, f32); NUM_QMF_SUBBANDS]],
    z1: &mut [[(f32, f32); NUM_QMF_SUBBANDS]],
    y2: &[[(f32, f32); NUM_QMF_SUBBANDS]],
    b3_pb: &[Vec<f32>],
    b3a_pb: &[Vec<f32>],
    num_param_bands: u32,
    steep: bool,
    param_timeslots: &[u8],
) {
    let num_ts = z0.len();
    debug_assert_eq!(z1.len(), num_ts);
    debug_assert_eq!(y2.len(), num_ts);
    let num_pset = b3_pb.len() as u32;
    let b3_sb = expand_pb_to_sb(b3_pb, num_param_bands);
    let b3a_sb = expand_pb_to_sb(b3a_pb, num_param_bands);
    let zero_prev = vec![0.0f32; NUM_QMF_SUBBANDS];
    let b3_inp = InterpInputs {
        by_pset: &b3_sb,
        prev: &zero_prev,
    };
    let b3a_inp = InterpInputs {
        by_pset: &b3a_sb,
        prev: &zero_prev,
    };
    for ts in 0..num_ts {
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            let b3 = interpolate(
                &b3_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let b3a = interpolate(
                &b3a_inp,
                num_pset,
                sb,
                ts as u32,
                num_ts as u32,
                steep,
                param_timeslots,
            );
            let sb_i = sb as usize;
            let (yr, yi) = y2[ts][sb_i];
            z0[ts][sb_i].0 += 0.25 * yr * (b3 + b3a);
            z0[ts][sb_i].1 += 0.25 * yi * (b3 + b3a);
            z1[ts][sb_i].0 += 0.25 * yr * (b3 - b3a);
            z1[ts][sb_i].1 += 0.25 * yi * (b3 - b3a);
        }
    }
}

/// Helper: per-`(pset, pb)` element-wise multiply of two per-band
/// matrices. Returned matrix has the same shape as `a`. For mismatched
/// inner-row lengths the shorter wins.
fn pb_matrix_mul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ar, br)| {
            let n = ar.len().min(br.len());
            let mut row = Vec::with_capacity(n);
            for i in 0..n {
                row.push(ar[i] * br[i]);
            }
            row
        })
        .collect()
}

/// Helper: per-`(pset, pb)` element-wise sum of three per-band
/// matrices. Used to build `g1+g3+g5` and `g2+g4+g6` for the third
/// `Transform()` call in Pseudocode 118.
fn pb_matrix_sum3(a: &[Vec<f32>], b: &[Vec<f32>], c: &[Vec<f32>]) -> Vec<Vec<f32>> {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((ar, br), cr)| {
            let n = ar.len().min(br.len()).min(cr.len());
            let mut row = Vec::with_capacity(n);
            for i in 0..n {
                row.push(ar[i] + br[i] + cr[i]);
            }
            row
        })
        .collect()
}

/// Helper: scalar-multiply each element of a `(pset, pb)` matrix.
fn pb_matrix_scale(a: &[Vec<f32>], s: f32) -> Vec<Vec<f32>> {
    a.iter()
        .map(|row| row.iter().map(|&v| v * s).collect())
        .collect()
}

/// Persistent state for the §5.7.7.6.2 ASPX_ACPL_3 multichannel
/// pipeline: three parallel `D0`/`D1`/`D2` decorrelator + ducker pairs
/// (one per `Transform()` output) plus the running `prev` matrices for
/// the gamma interpolations.
#[derive(Debug, Clone)]
pub struct AcplMchState {
    pub d0: InputSignalModifier,
    pub d1: InputSignalModifier,
    pub d2: InputSignalModifier,
    pub ducker0: TransientDucker,
    pub ducker1: TransientDucker,
    pub ducker2: TransientDucker,
    /// `acpl_g1_prev[sb]` — last-frame's per-sb gamma1 row.
    pub g1_prev_sb: Vec<f32>,
    pub g2_prev_sb: Vec<f32>,
    pub g3_prev_sb: Vec<f32>,
    pub g4_prev_sb: Vec<f32>,
}

impl AcplMchState {
    pub fn new() -> Self {
        Self {
            d0: InputSignalModifier::new(DecorrelatorId::D0),
            d1: InputSignalModifier::new(DecorrelatorId::D1),
            d2: InputSignalModifier::new(DecorrelatorId::D2),
            ducker0: TransientDucker::new(),
            ducker1: TransientDucker::new(),
            ducker2: TransientDucker::new(),
            g1_prev_sb: vec![0.0; NUM_QMF_SUBBANDS],
            g2_prev_sb: vec![0.0; NUM_QMF_SUBBANDS],
            g3_prev_sb: vec![0.0; NUM_QMF_SUBBANDS],
            g4_prev_sb: vec![0.0; NUM_QMF_SUBBANDS],
        }
    }
}

impl Default for AcplMchState {
    fn default() -> Self {
        Self::new()
    }
}

/// Inputs to one full §5.7.7.6.2 ASPX_ACPL_3 multichannel synthesis pass.
///
/// All matrices are per-`(pset, pb)`. The dequantisation pipeline must
/// have already converted the Huffman-decoded indices via
/// [`dequantize_alpha_beta`] / [`dequantize_beta3`] / [`dequantize_gamma`].
pub struct AcplMchFrame<'a> {
    /// `x0[ts][sb]` — first A-CPL carrier (left/L for ASPX_ACPL_3).
    pub x0: &'a [[(f32, f32); NUM_QMF_SUBBANDS]],
    /// `x1[ts][sb]` — second A-CPL carrier (right/R for ASPX_ACPL_3).
    pub x1: &'a [[(f32, f32); NUM_QMF_SUBBANDS]],
    /// `x2[ts][sb]` — centre channel passthrough.
    pub x2: &'a [[(f32, f32); NUM_QMF_SUBBANDS]],
    /// `acpl_alpha_1_dq[pset][pb]` (left side).
    pub alpha_1_dq: &'a [Vec<f32>],
    /// `acpl_alpha_2_dq[pset][pb]` (right side).
    pub alpha_2_dq: &'a [Vec<f32>],
    /// `acpl_beta_1_dq[pset][pb]`.
    pub beta_1_dq: &'a [Vec<f32>],
    /// `acpl_beta_2_dq[pset][pb]`.
    pub beta_2_dq: &'a [Vec<f32>],
    /// `acpl_beta_3_dq[pset][pb]` — the cross-residual term used by
    /// `ACplModule3`.
    pub beta_3_dq: &'a [Vec<f32>],
    /// `g1_dq[pset][pb]` .. `g6_dq[pset][pb]` per §5.7.7.7 Tables 207-208.
    pub g1_dq: &'a [Vec<f32>],
    pub g2_dq: &'a [Vec<f32>],
    pub g3_dq: &'a [Vec<f32>],
    pub g4_dq: &'a [Vec<f32>],
    pub g5_dq: &'a [Vec<f32>],
    pub g6_dq: &'a [Vec<f32>],
    pub num_param_bands: u32,
    pub steep: bool,
    pub param_timeslots: &'a [u8],
}

/// Output channels from [`run_pseudocode_118_5x`] — `(L, R, C, Ls, Rs)`
/// indexed as `(z0, z2, z4, z1, z3)` in the spec.
pub struct AcplMchOutput {
    pub z0: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>, // L
    pub z2: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>, // R
    pub z4: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>, // C
    pub z1: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>, // Ls
    pub z3: Vec<[(f32, f32); NUM_QMF_SUBBANDS]>, // Rs
}

/// Run the full §5.7.7.6.2 Pseudocode 118 ASPX_ACPL_3 multichannel
/// pipeline. Produces the five output channels (L, Ls, R, Rs, C) from
/// the two A-CPL carriers, the centre channel and the gamma+alpha+beta
/// matrices.
///
/// Pipeline (verbatim from Pseudocode 118):
///
/// 1. `x0in = x0*(1+2*sqrt(0.5))`, `x1in = x1*(1+2*sqrt(0.5))`.
/// 2. `v1 = Transform(g1, g2, x0in, x1in)`,
///    `v2 = Transform(g3, g4, x0in, x1in)`,
///    `v3 = Transform(g1+g3+g5, g2+g4+g6, x0in, x1in)`.
/// 3. `u0 = D0(v1)`, `u1 = D1(v2)`, `u2 = D2(v3)`.
/// 4. `y0 = ducker(u0)`, `y1 = ducker(u1)`, `y2 = ducker(u2)`.
/// 5. `(z0, z1) = ACplModule2(g1, g2, alpha_1, beta_1, x0in, x1in, y0)`.
/// 6. `(z2, z3) = ACplModule2(g3, g4, alpha_2, beta_2, x0in, x1in, y1)`.
/// 7. `(z4, z5) = ACplModule2(g5, g6, 1, 0, x0in, x1in, 0)`.
/// 8. `(z0, z1) = ACplModule3(beta_3, alpha_1, z0, z1, y2)`.
/// 9. `(z2, z3) = ACplModule3(beta_3, alpha_2, z2, z3, y2)`.
/// 10. `(z4, z5) = ACplModule3(-beta_3, 1, z4, z5, y2)`.
/// 11. `z1 *= sqrt(2)`, `z3 *= sqrt(2)`, `z4 *= sqrt(2)`.
/// 12. `z4 = x2` (note: per the spec text, z4 is initialised from
///     `ACplModule2(g5, g6, ..)` and then has the ACplModule3 correction
///     and `*sqrt(2)` applied — but the surrounding text in §5.7.7.6.2
///     reads "z4 = x2" outside the pseudocode body. We follow the
///     pseudocode literally: the centre channel is the synthesised z4;
///     a caller wanting the spec's "z4 = x2" passthrough can override
///     [`AcplMchOutput::z4`] after the fact).
pub fn run_pseudocode_118_5x(state: &mut AcplMchState, frame: AcplMchFrame<'_>) -> AcplMchOutput {
    let num_ts = frame.x0.len();
    debug_assert_eq!(frame.x1.len(), num_ts);
    debug_assert_eq!(frame.x2.len(), num_ts);

    // Step 1: x0in = x0 * (1 + 2*sqrt(0.5)), x1in = x1 * (1 + 2*sqrt(0.5)).
    // 1 + 2*sqrt(0.5) = 1 + sqrt(2).
    let scale = 1.0 + 2.0 * (0.5f32).sqrt();
    let scale_x = |x: &[[(f32, f32); NUM_QMF_SUBBANDS]]| -> Vec<[(f32, f32); NUM_QMF_SUBBANDS]> {
        x.iter()
            .map(|col| {
                let mut out = *col;
                for sb in 0..NUM_QMF_SUBBANDS {
                    out[sb].0 *= scale;
                    out[sb].1 *= scale;
                }
                out
            })
            .collect()
    };
    let x0in = scale_x(frame.x0);
    let x1in = scale_x(frame.x1);

    // Step 2: build the three Transform() outputs.
    let g_sum_1 = pb_matrix_sum3(frame.g1_dq, frame.g3_dq, frame.g5_dq);
    let g_sum_2 = pb_matrix_sum3(frame.g2_dq, frame.g4_dq, frame.g6_dq);
    let v1 = transform(
        &x0in,
        &x1in,
        frame.g1_dq,
        frame.g2_dq,
        frame.num_param_bands,
        &state.g1_prev_sb,
        &state.g2_prev_sb,
        frame.steep,
        frame.param_timeslots,
    );
    let v2 = transform(
        &x0in,
        &x1in,
        frame.g3_dq,
        frame.g4_dq,
        frame.num_param_bands,
        &state.g3_prev_sb,
        &state.g4_prev_sb,
        frame.steep,
        frame.param_timeslots,
    );
    let v3 = transform(
        &x0in,
        &x1in,
        &g_sum_1,
        &g_sum_2,
        frame.num_param_bands,
        &vec![0.0; NUM_QMF_SUBBANDS],
        &vec![0.0; NUM_QMF_SUBBANDS],
        frame.steep,
        frame.param_timeslots,
    );

    // Step 3: u_x = decorrelator_x(v_x).
    let mut u0 = Vec::with_capacity(num_ts);
    let mut u1 = Vec::with_capacity(num_ts);
    let mut u2 = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        let mut col0 = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        let mut col1 = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        let mut col2 = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for sb in 0..NUM_QMF_SUBBANDS as u32 {
            col0[sb as usize] = state.d0.process_sample(sb, v1[ts][sb as usize]);
            col1[sb as usize] = state.d1.process_sample(sb, v2[ts][sb as usize]);
            col2[sb as usize] = state.d2.process_sample(sb, v3[ts][sb as usize]);
        }
        u0.push(col0);
        u1.push(col1);
        u2.push(col2);
    }

    // Step 4: y_x = ducker(u_x).
    let apply_ducker = |u: &[[(f32, f32); NUM_QMF_SUBBANDS]],
                        ducker: &mut TransientDucker|
     -> Vec<[(f32, f32); NUM_QMF_SUBBANDS]> {
        u.iter()
            .map(|col| {
                let p_energy = compute_p_energy(col, frame.num_param_bands);
                let duck = ducker.update(&p_energy);
                apply_transient_ducker(col, &duck, frame.num_param_bands)
            })
            .collect()
    };
    let y0 = apply_ducker(&u0, &mut state.ducker0);
    let y1 = apply_ducker(&u1, &mut state.ducker1);
    let y2 = apply_ducker(&u2, &mut state.ducker2);

    // Step 5: (z0, z1) = ACplModule2(g1, g2, alpha_1, beta_1, x0in, x1in, y0).
    // Per Pseudocode 119 the (g1*a) / (g2*a) interpolations sample the
    // *product* matrices — we precompute them at parameter-band granularity.
    let g1_a1 = pb_matrix_mul(frame.g1_dq, frame.alpha_1_dq);
    let g2_a1 = pb_matrix_mul(frame.g2_dq, frame.alpha_1_dq);
    let (mut z0, mut z1) = acpl_module2(
        &x0in,
        &x1in,
        &y0,
        frame.g1_dq,
        frame.g2_dq,
        &g1_a1,
        &g2_a1,
        frame.beta_1_dq,
        frame.num_param_bands,
        frame.steep,
        frame.param_timeslots,
    );

    // Step 6: (z2, z3) = ACplModule2(g3, g4, alpha_2, beta_2, x0in, x1in, y1).
    let g3_a2 = pb_matrix_mul(frame.g3_dq, frame.alpha_2_dq);
    let g4_a2 = pb_matrix_mul(frame.g4_dq, frame.alpha_2_dq);
    let (mut z2, mut z3) = acpl_module2(
        &x0in,
        &x1in,
        &y1,
        frame.g3_dq,
        frame.g4_dq,
        &g3_a2,
        &g4_a2,
        frame.beta_2_dq,
        frame.num_param_bands,
        frame.steep,
        frame.param_timeslots,
    );

    // Step 7: (z4, z5) = ACplModule2(g5, g6, 1, 0, x0in, x1in, 0).
    // a == 1 → g*a == g; b == 0 → no decorrelator term; y == 0 too.
    let zero_y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
    // β = 0 across all bands. The shape follows the gamma matrices.
    let zero_pb: Vec<Vec<f32>> = frame
        .g5_dq
        .iter()
        .map(|row| vec![0.0f32; row.len()])
        .collect();
    let (mut z4, _z5) = acpl_module2(
        &x0in,
        &x1in,
        &zero_y,
        frame.g5_dq,
        frame.g6_dq,
        frame.g5_dq, // g5 * 1
        frame.g6_dq, // g6 * 1
        &zero_pb,
        frame.num_param_bands,
        frame.steep,
        frame.param_timeslots,
    );
    // _z5 is a temporary per the spec note ("Note that z5 is used as a
    // temporary variable only and does not constitute an output channel.").

    // Step 8: (z0, z1) += ACplModule3(beta_3, alpha_1, z0, z1, y2).
    let b3_a1 = pb_matrix_mul(frame.beta_3_dq, frame.alpha_1_dq);
    acpl_module3(
        &mut z0,
        &mut z1,
        &y2,
        frame.beta_3_dq,
        &b3_a1,
        frame.num_param_bands,
        frame.steep,
        frame.param_timeslots,
    );

    // Step 9: (z2, z3) += ACplModule3(beta_3, alpha_2, z2, z3, y2).
    let b3_a2 = pb_matrix_mul(frame.beta_3_dq, frame.alpha_2_dq);
    acpl_module3(
        &mut z2,
        &mut z3,
        &y2,
        frame.beta_3_dq,
        &b3_a2,
        frame.num_param_bands,
        frame.steep,
        frame.param_timeslots,
    );

    // Step 10: (z4, z5) += ACplModule3(-beta_3, 1, z4, z5, y2). a == 1 →
    // beta3*a == beta3 (i.e. b3a == b3 with a sign flip on b3 — but the
    // spec writes -b3 for this row). So b3 is negated and b3a == -b3 too.
    let neg_b3 = pb_matrix_scale(frame.beta_3_dq, -1.0);
    let neg_b3_a = pb_matrix_scale(frame.beta_3_dq, -1.0); // a == 1 → -b3*1 = -b3.
    let mut z5_dummy = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
    acpl_module3(
        &mut z4,
        &mut z5_dummy,
        &y2,
        &neg_b3,
        &neg_b3_a,
        frame.num_param_bands,
        frame.steep,
        frame.param_timeslots,
    );

    // Step 11: z1 *= sqrt(2), z3 *= sqrt(2), z4 *= sqrt(2).
    let sq2 = (2.0f32).sqrt();
    let scale_inplace = |z: &mut [[(f32, f32); NUM_QMF_SUBBANDS]], s: f32| {
        for col in z.iter_mut() {
            for sb in 0..NUM_QMF_SUBBANDS {
                col[sb].0 *= s;
                col[sb].1 *= s;
            }
        }
    };
    scale_inplace(&mut z1, sq2);
    scale_inplace(&mut z3, sq2);
    scale_inplace(&mut z4, sq2);

    // Update state's prev arrays from the last param set's expanded
    // [sb] rows (gammas only — alpha/beta state lives elsewhere if a
    // caller wants to chain it).
    let update_prev = |prev: &mut Vec<f32>, src_pb: &[Vec<f32>]| {
        if !src_pb.is_empty() {
            let last = &src_pb[src_pb.len() - 1];
            let sb = expand_pb_to_sb(std::slice::from_ref(last), frame.num_param_bands);
            update_param_prev(prev, &sb[0]);
        }
    };
    update_prev(&mut state.g1_prev_sb, frame.g1_dq);
    update_prev(&mut state.g2_prev_sb, frame.g2_dq);
    update_prev(&mut state.g3_prev_sb, frame.g3_dq);
    update_prev(&mut state.g4_prev_sb, frame.g4_dq);

    AcplMchOutput { z0, z2, z4, z1, z3 }
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

/// Run the §5.7.7 A-CPL channel-pair synthesis on a stereo PCM input
/// (M / S channels from a joint-MDCT body, both already ASPX-extended on
/// the M side per `aspx_data_1ch()`) and emit stereo PCM via QMF
/// analysis → A-CPL → QMF synthesis.
///
/// Spec wiring (ETSI TS 103 190-1 §4.2.6.3 ASPX_ACPL_1):
///   * `pcm_m` / `pcm_s` — joint-MDCT M (mid) and S (side) PCM streams,
///     same length, multiple of 64 samples.
///   * `cfg` — parsed `acpl_config_1ch(PARTIAL)` for the active substream.
///   * `data` — parsed `acpl_data_1ch()` (§4.2.13.3 Table 61).
///   * `state` — per-substream state carried across frames.
///
/// In the spec's `acpl_module()` (Pseudocode 116) the low subbands below
/// `acpl_qmf_band` recover `(L, R)` via the inverse-M/S split
/// `z0 = 0.5*(x0+x1)`, `z1 = 0.5*(x0-x1)`; the upper subbands re-mix the
/// decorrelator output `y` (computed from `x0`) into the second channel
/// using the parametric `alpha`/`beta` coefficients. We feed `pcm_m` into
/// `x0` and `pcm_s` into `x1` here, matching the spec's M/S convention.
pub fn run_acpl_1ch_pcm_stereo(
    pcm_m: &[f32],
    pcm_s: &[f32],
    cfg: &AcplConfig1ch,
    data: &AcplData1ch,
    state: &mut AcplSubstreamState,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if pcm_m.is_empty() || pcm_m.len() % NUM_QMF_SUBBANDS != 0 || pcm_m.len() != pcm_s.len() {
        return None;
    }
    let n_slots = pcm_m.len() / NUM_QMF_SUBBANDS;
    if n_slots == 0 {
        return None;
    }
    let mut ana_m = QmfAnalysisBank::new();
    let mut ana_s = QmfAnalysisBank::new();
    let x0 = ana_m.process_block(pcm_m);
    let x1 = ana_s.process_block(pcm_s);
    let alpha_q = differential_decode(&data.alpha1, cfg.num_param_bands, &mut state.alpha_diff);
    let beta_q = differential_decode(&data.beta1, cfg.num_param_bands, &mut state.beta_diff);
    let (alpha_dq, beta_dq) = dequantize_alpha_beta(&alpha_q, &beta_q, cfg.quant_mode);
    if alpha_dq.is_empty() {
        return None;
    }
    let frame = AcplCpeFrame {
        x0: &x0,
        x1: Some(&x1),
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
    let mut syn0 = QmfSynthesisBank::new();
    let mut syn1 = QmfSynthesisBank::new();
    let mut left = Vec::with_capacity(pcm_m.len());
    let mut right = Vec::with_capacity(pcm_m.len());
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
    fn run_acpl_1ch_pcm_stereo_emits_two_channels_distinct_from_l_r_passthrough() {
        // Stereo (M/S) input variant — same shape as ACPL_2 but with x1
        // populated. Verifies the spec's M/S split path runs end-to-end
        // and produces non-silent, distinct channels.
        use crate::acpl::{
            AcplConfig1ch, AcplData1ch, AcplFramingData, AcplHuffParam, AcplInterpolationType,
            AcplQuantMode,
        };
        let n_slots = 32usize;
        let n = n_slots * NUM_QMF_SUBBANDS;
        let mut pcm_m = vec![0.0f32; n];
        let mut pcm_s = vec![0.0f32; n];
        let f_m = 440.0_f32 / 48_000.0_f32;
        let f_s = 220.0_f32 / 48_000.0_f32;
        for i in 0..n {
            pcm_m[i] = (2.0 * std::f32::consts::PI * f_m * i as f32).sin();
            pcm_s[i] = 0.3 * (2.0 * std::f32::consts::PI * f_s * i as f32).sin();
        }
        let cfg = AcplConfig1ch {
            num_param_bands_id: 0,
            num_param_bands: 15,
            quant_mode: AcplQuantMode::Fine,
            // PARTIAL mode: nonzero qmf_band so the M/S split path
            // engages on low subbands.
            qmf_band: 8,
        };
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
        let (left, right) = run_acpl_1ch_pcm_stereo(&pcm_m, &pcm_s, &cfg, &data, &mut state)
            .expect("stereo synth runs");
        assert_eq!(left.len(), pcm_m.len());
        assert_eq!(right.len(), pcm_m.len());
        let start = 1024usize;
        let e_l: f64 = left[start..].iter().map(|&s| (s as f64).powi(2)).sum();
        let e_r: f64 = right[start..].iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(e_l > 1e-6, "left channel silent (e={e_l})");
        assert!(e_r > 1e-6, "right channel silent (e={e_r})");
        // The output must not be a literal pass-through of the input.
        let mut diff_from_input = 0usize;
        for i in start..left.len() {
            if (left[i] - pcm_m[i]).abs() > 1e-3 {
                diff_from_input += 1;
            }
        }
        assert!(
            diff_from_input > (left.len() - start) / 4,
            "left channel matches input PCM (diff_from_input={diff_from_input})"
        );
    }

    #[test]
    fn run_acpl_1ch_pcm_stereo_rejects_mismatched_lengths() {
        use crate::acpl::{
            AcplConfig1ch, AcplData1ch, AcplFramingData, AcplInterpolationType, AcplQuantMode,
        };
        let pcm_m = vec![0.0f32; 64];
        let pcm_s = vec![0.0f32; 128];
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
        assert!(run_acpl_1ch_pcm_stereo(&pcm_m, &pcm_s, &cfg, &data, &mut state).is_none());
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

    // =====================================================================
    // §5.7.7.6.2 — ASPX_ACPL_3 multichannel transform synthesis tests
    // (Pseudocodes 117 / 118 / 119 wiring + helpers)
    // =====================================================================

    /// `Transform()` with `g1 = 1, g2 = 0` should pass `x0` through and
    /// drop `x1`. `g1 = 0, g2 = 1` should pass `x1` through.
    #[test]
    fn transform_unit_gammas_select_carriers() {
        let num_ts = 8usize;
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x1 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            x0[ts][10] = (1.0, 0.0);
            x1[ts][10] = (0.0, 1.0);
        }
        let g1 = vec![vec![1.0f32; 15]];
        let g2 = vec![vec![0.0f32; 15]];
        let prev = vec![1.0f32; NUM_QMF_SUBBANDS];
        let prev_zero = vec![0.0f32; NUM_QMF_SUBBANDS];
        // num_pset == 1, prev_g1 = 1.0 across sb -> interpolation stays
        // at 1.0 (start at prev=1.0, target=1.0, no ramp).
        let v = transform(&x0, &x1, &g1, &g2, 15, &prev, &prev_zero, false, &[]);
        for ts in 0..num_ts {
            // sb=10: v = x0 * 1 + x1 * 0 = x0.
            let (vr, vi) = v[ts][10];
            assert!((vr - 1.0).abs() < 1e-5, "ts={ts} vr={vr}, expected x0=1.0");
            assert!((vi - 0.0).abs() < 1e-5, "ts={ts} vi={vi}");
            // sb=20: both x0 and x1 are zero, result must be zero.
            assert_eq!(v[ts][20], (0.0, 0.0));
        }

        // Now flip: g1 = 0 → x1 carries through.
        let g1 = vec![vec![0.0f32; 15]];
        let g2 = vec![vec![1.0f32; 15]];
        let prev_g2 = vec![1.0f32; NUM_QMF_SUBBANDS];
        let v = transform(&x0, &x1, &g1, &g2, 15, &prev_zero, &prev_g2, false, &[]);
        for ts in 0..num_ts {
            let (vr, vi) = v[ts][10];
            assert!((vr - 0.0).abs() < 1e-5);
            assert!((vi - 1.0).abs() < 1e-5);
        }
    }

    /// `Transform()` with mixed gammas should produce the linear
    /// combination `x0*g1 + x1*g2` after the smooth-interpolation ramp.
    #[test]
    fn transform_mixes_two_carriers_with_gammas() {
        let num_ts = 4usize;
        let mut x0 = vec![[(2.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x1 = vec![[(0.0f32, 3.0f32); NUM_QMF_SUBBANDS]; num_ts];
        // Make g1 = 0.5 across all bands, g2 = 0.25. With prev = same
        // values, smooth interpolation collapses to constant gammas.
        let g1 = vec![vec![0.5f32; 15]];
        let g2 = vec![vec![0.25f32; 15]];
        let prev_g1 = vec![0.5f32; NUM_QMF_SUBBANDS];
        let prev_g2 = vec![0.25f32; NUM_QMF_SUBBANDS];
        let v = transform(&x0, &x1, &g1, &g2, 15, &prev_g1, &prev_g2, false, &[]);
        for ts in 0..num_ts {
            for sb in 0..NUM_QMF_SUBBANDS {
                let (vr, vi) = v[ts][sb];
                // x0 contributes (2 * 0.5, 0). x1 contributes (0, 3 * 0.25).
                assert!((vr - 1.0).abs() < 1e-5, "ts={ts} sb={sb} vr={vr}");
                assert!((vi - 0.75).abs() < 1e-5, "ts={ts} sb={sb} vi={vi}");
            }
        }
        // Quiet unused-mut warnings.
        let _ = (&mut x0, &mut x1);
    }

    /// `acpl_module2()` with g1=g2=g1*a=g2*a=b=0 must yield silent z0,z1.
    #[test]
    fn acpl_module2_zero_gammas_is_silent() {
        let num_ts = 4usize;
        let x0 = vec![[(1.0f32, 0.5f32); NUM_QMF_SUBBANDS]; num_ts];
        let x1 = vec![[(0.5f32, 1.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y = vec![[(2.0f32, 2.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let zero = vec![vec![0.0f32; 15]];
        let (z0, z1) = acpl_module2(
            &x0,
            &x1,
            &y,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            15,
            false,
            &[],
        );
        for ts in 0..num_ts {
            for sb in 0..NUM_QMF_SUBBANDS {
                assert_eq!(z0[ts][sb], (0.0, 0.0));
                assert_eq!(z1[ts][sb], (0.0, 0.0));
            }
        }
    }

    /// `acpl_module2()` with g1=1, g2=0, a=0, b=0 (so g1a = g2a = 0):
    ///   z0 = 0.5 * (x0 * (g1+g1a) + x1 * (g2+g2a) + y*b)
    ///       = 0.5 * x0 * 1
    ///   z1 = 0.5 * x0 * (1-0) = 0.5 * x0
    /// → both z0 and z1 collapse to 0.5*x0.
    #[test]
    fn acpl_module2_unit_g1_zero_alpha_beta_passes_half_x0_to_both() {
        let num_ts = 4usize;
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let x1 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            x0[ts][5] = (4.0, 2.0);
        }
        let g1 = vec![vec![1.0f32; 15]];
        let g2 = vec![vec![0.0f32; 15]];
        let g1a = vec![vec![0.0f32; 15]]; // a=0 → g1*a=0
        let g2a = vec![vec![0.0f32; 15]];
        let b = vec![vec![0.0f32; 15]];
        let (z0, z1) = acpl_module2(&x0, &x1, &y, &g1, &g2, &g1a, &g2a, &b, 15, false, &[]);
        // smooth interpolation with prev=0, target=1 → at ts=num_ts-1, g1
        // hits 1.0, but z formula has g1+g1a so 1+0=1; z = 0.5*x0*1 = 0.5*x0.
        // Just assert the last ts converges.
        let last = num_ts - 1;
        let (z0r, z0i) = z0[last][5];
        let (z1r, z1i) = z1[last][5];
        assert!((z0r - 2.0).abs() < 1e-5, "z0r={z0r} expected 2.0");
        assert!((z0i - 1.0).abs() < 1e-5, "z0i={z0i} expected 1.0");
        assert!((z1r - 2.0).abs() < 1e-5, "z1r={z1r}");
        assert!((z1i - 1.0).abs() < 1e-5, "z1i={z1i}");
    }

    /// `acpl_module3()` adds a beta3-driven decorrelator residual.
    /// With b3=1, b3a=0 (i.e. a=0): z0 += 0.25*y*1 = 0.25*y, z1 += same.
    #[test]
    fn acpl_module3_adds_decorrelator_residual() {
        let num_ts = 4usize;
        let mut z0 = vec![[(1.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut z1 = vec![[(0.0f32, 1.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y2 = vec![[(4.0f32, 4.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let b3 = vec![vec![1.0f32; 15]];
        let b3a = vec![vec![0.0f32; 15]];
        acpl_module3(&mut z0, &mut z1, &y2, &b3, &b3a, 15, false, &[]);
        // After ramp completes (ts=num_ts-1) b3 = 1.0, b3a = 0:
        //   z0 += 0.25 * y2 * (1 + 0) = 0.25 * (4, 4) = (1, 1)
        //   z1 += 0.25 * y2 * (1 - 0) = (1, 1)
        let last = num_ts - 1;
        let (z0r, z0i) = z0[last][7];
        assert!((z0r - 2.0).abs() < 1e-5, "z0r={z0r}");
        assert!((z0i - 1.0).abs() < 1e-5, "z0i={z0i}");
        let (z1r, z1i) = z1[last][7];
        assert!((z1r - 1.0).abs() < 1e-5, "z1r={z1r}");
        assert!((z1i - 2.0).abs() < 1e-5, "z1i={z1i}");
    }

    /// `acpl_module3()` with all-zero beta3 is a no-op.
    #[test]
    fn acpl_module3_zero_beta3_is_noop() {
        let num_ts = 4usize;
        let mut z0 = vec![[(7.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut z1 = vec![[(0.0f32, 7.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y2 = vec![[(99.0f32, -99.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let zero = vec![vec![0.0f32; 15]];
        acpl_module3(&mut z0, &mut z1, &y2, &zero, &zero, 15, false, &[]);
        for ts in 0..num_ts {
            for sb in 0..NUM_QMF_SUBBANDS {
                assert_eq!(z0[ts][sb], (7.0, 0.0));
                assert_eq!(z1[ts][sb], (0.0, 7.0));
            }
        }
    }

    /// Smoke test for the full §5.7.7.6.2 ASPX_ACPL_3 pipeline. Feed
    /// non-zero L/R/C carriers and verify all 5 outputs are populated and
    /// finite.
    #[test]
    fn run_pseudocode_118_5x_emits_five_finite_channels() {
        let num_ts = 16usize;
        let num_pb = 9u32;
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x1 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x2 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            for sb in 0..32 {
                let p = (ts as f32) * 0.2 + sb as f32 * 0.05;
                x0[ts][sb] = (0.1 * p.cos(), 0.1 * p.sin());
                x1[ts][sb] = (0.05 * p.sin(), 0.05 * p.cos());
                x2[ts][sb] = (0.03 * (p * 1.3).cos(), 0.0);
            }
        }
        let alpha_1 = vec![vec![0.4f32; num_pb as usize]];
        let alpha_2 = vec![vec![-0.2f32; num_pb as usize]];
        let beta_1 = vec![vec![0.3f32; num_pb as usize]];
        let beta_2 = vec![vec![0.2f32; num_pb as usize]];
        let beta_3 = vec![vec![0.15f32; num_pb as usize]];
        let g1 = vec![vec![0.6f32; num_pb as usize]];
        let g2 = vec![vec![0.3f32; num_pb as usize]];
        let g3 = vec![vec![0.2f32; num_pb as usize]];
        let g4 = vec![vec![0.5f32; num_pb as usize]];
        let g5 = vec![vec![0.1f32; num_pb as usize]];
        let g6 = vec![vec![0.1f32; num_pb as usize]];
        let mut state = AcplMchState::new();
        let frame = AcplMchFrame {
            x0: &x0,
            x1: &x1,
            x2: &x2,
            alpha_1_dq: &alpha_1,
            alpha_2_dq: &alpha_2,
            beta_1_dq: &beta_1,
            beta_2_dq: &beta_2,
            beta_3_dq: &beta_3,
            g1_dq: &g1,
            g2_dq: &g2,
            g3_dq: &g3,
            g4_dq: &g4,
            g5_dq: &g5,
            g6_dq: &g6,
            num_param_bands: num_pb,
            steep: false,
            param_timeslots: &[],
        };
        let out = run_pseudocode_118_5x(&mut state, frame);
        assert_eq!(out.z0.len(), num_ts);
        assert_eq!(out.z1.len(), num_ts);
        assert_eq!(out.z2.len(), num_ts);
        assert_eq!(out.z3.len(), num_ts);
        assert_eq!(out.z4.len(), num_ts);
        // No NaN / Inf in any output channel.
        for z in [&out.z0, &out.z1, &out.z2, &out.z3, &out.z4] {
            for col in z.iter() {
                for sb in 0..NUM_QMF_SUBBANDS {
                    assert!(
                        col[sb].0.is_finite() && col[sb].1.is_finite(),
                        "non-finite output: {:?}",
                        col[sb]
                    );
                }
            }
        }
        // Carriers are non-zero; the synthesis must produce non-zero
        // energy on all five channels.
        for (name, z) in [
            ("z0=L", &out.z0),
            ("z1=Ls", &out.z1),
            ("z2=R", &out.z2),
            ("z3=Rs", &out.z3),
            ("z4=C", &out.z4),
        ] {
            let e: f64 = z
                .iter()
                .flat_map(|col| col.iter())
                .map(|s| (s.0 as f64).powi(2) + (s.1 as f64).powi(2))
                .sum();
            assert!(e > 0.0, "channel {name} silent (energy {e})");
        }
        // Gamma prev arrays should be populated after the call.
        assert_eq!(state.g1_prev_sb.len(), NUM_QMF_SUBBANDS);
        for sb in 0..NUM_QMF_SUBBANDS {
            assert!((state.g1_prev_sb[sb] - 0.6).abs() < 1e-6);
        }
    }

    /// All-zero alpha/beta/beta3 + matching gamma should still produce
    /// valid, finite output. This covers the "no decorrelator coupling"
    /// path through the 5_X synthesis.
    #[test]
    fn run_pseudocode_118_5x_zero_alpha_beta_remains_finite() {
        let num_ts = 8usize;
        let num_pb = 9u32;
        let x0 = vec![[(0.5f32, 0.5f32); NUM_QMF_SUBBANDS]; num_ts];
        let x1 = vec![[(0.5f32, -0.5f32); NUM_QMF_SUBBANDS]; num_ts];
        let x2 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let zero = vec![vec![0.0f32; num_pb as usize]];
        let g1 = vec![vec![1.0f32; num_pb as usize]]; // g1 != 0 so v1 isn't trivially silent
        let mut state = AcplMchState::new();
        let frame = AcplMchFrame {
            x0: &x0,
            x1: &x1,
            x2: &x2,
            alpha_1_dq: &zero,
            alpha_2_dq: &zero,
            beta_1_dq: &zero,
            beta_2_dq: &zero,
            beta_3_dq: &zero,
            g1_dq: &g1,
            g2_dq: &zero,
            g3_dq: &zero,
            g4_dq: &zero,
            g5_dq: &zero,
            g6_dq: &zero,
            num_param_bands: num_pb,
            steep: false,
            param_timeslots: &[],
        };
        let out = run_pseudocode_118_5x(&mut state, frame);
        for col in out.z0.iter() {
            for sb in 0..NUM_QMF_SUBBANDS {
                assert!(col[sb].0.is_finite());
                assert!(col[sb].1.is_finite());
            }
        }
    }

    /// Smoke test for `pb_matrix_*` helpers used inside Pseudocode 118.
    #[test]
    fn pb_matrix_helpers() {
        let a = vec![vec![1.0f32, 2.0, 3.0]];
        let b = vec![vec![10.0f32, 20.0, 30.0]];
        let c = vec![vec![100.0f32, 200.0, 300.0]];
        let prod = pb_matrix_mul(&a, &b);
        assert_eq!(prod, vec![vec![10.0, 40.0, 90.0]]);
        let sum = pb_matrix_sum3(&a, &b, &c);
        assert_eq!(sum, vec![vec![111.0, 222.0, 333.0]]);
        let scaled = pb_matrix_scale(&a, -2.0);
        assert_eq!(scaled, vec![vec![-2.0, -4.0, -6.0]]);
    }

    /// Spec consistency: the `(1 + 2*sqrt(0.5))` scaling factor applied
    /// to x0/x1 in Pseudocode 118 must equal `1 + sqrt(2)` ≈ 2.414...
    #[test]
    fn pseudocode_118_scaling_factor() {
        let s = 1.0f32 + 2.0 * (0.5f32).sqrt();
        let expected = 1.0f32 + (2.0f32).sqrt();
        assert!((s - expected).abs() < 1e-6);
    }

    /// `AcplMchState::new()` should initialise prev arrays to zero.
    #[test]
    fn acpl_mch_state_zero_init() {
        let s = AcplMchState::new();
        assert!(s.g1_prev_sb.iter().all(|&v| v == 0.0));
        assert!(s.g2_prev_sb.iter().all(|&v| v == 0.0));
        assert!(s.g3_prev_sb.iter().all(|&v| v == 0.0));
        assert!(s.g4_prev_sb.iter().all(|&v| v == 0.0));
        assert_eq!(s.g1_prev_sb.len(), NUM_QMF_SUBBANDS);
    }
}
