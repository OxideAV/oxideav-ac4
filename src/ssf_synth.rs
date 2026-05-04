//! AC-4 Speech Spectral Frontend (SSF) synthesis chain.
//!
//! Implements the §5.2.3-5.2.7 PCM-side path that turns the
//! arithmetic-decoded indices in [`crate::ssf::SsfData`] into a vector
//! of `n_mdct` spectral lines per block (the `s_SSF,ch` output of
//! Figure 4 in §5.2.2). The IMDCT step itself lives in [`crate::mdct`]
//! — this module only produces the spectrum.
//!
//! Pseudocodes implemented:
//!
//! * §5.2.3.0a — Pseudocode 4a — `env_idx[band] -> env[band]`
//!   ([`decode_envelope`]).
//! * §5.2.3.0b — Pseudocode 4b — envelope interpolation (SHORT_STRIDE)
//!   ([`interpolate_envelope`]).
//! * §5.2.3.0c — Pseudocode 4c — `gain_idx[block] -> gain[block]`
//!   ([`decode_gains`]).
//! * §5.2.3.0d — Pseudocode 4d — envelope refinement (signal envelope +
//!   allocation envelope) ([`refine_envelope`]).
//! * §5.2.4.0a — Pseudocode 4e — predictor parameter calculation
//!   ([`decode_predictor`]).
//! * §5.2.5.2.1 — Pseudocode 26 — helper variable calculation
//!   ([`compute_helpers`]).
//! * §5.2.5.2.3 — Pseudocode 31 — lossless decoding (no-rfu path:
//!   `i_alloc_table` derived from `env_alloc_mod = env_alloc`)
//!   ([`build_alloc_table`]).
//! * §5.2.5.2.4 — Pseudocode 32 — inverse quantization
//!   ([`inverse_quantize_block`]); Pseudocode 33 `MmseLaplace` for the
//!   no-dither branch ([`mmse_laplace`]).
//! * §5.2.5.2.5 — Pseudocode 34 — heuristic inverse scaling (no-op when
//!   `f_gain_q == 1`) ([`inverse_heuristic_scale`]).
//! * §5.2.6 — Pseudocodes 35-37 — subband predictor
//!   ([`SubbandPredictorState::run`]).
//! * §5.2.7 — Pseudocode 38 — inverse flattening
//!   ([`inverse_flatten`]).
//! * §5.2.8.1 — Pseudocode 39 — C-matrix reconstruction from the
//!   quantized prediction-coefficient bytes in
//!   [`crate::ssf_pred_coeff`] ([`build_c_matrix`]).
//!
//! Pseudocodes 27 / 28 / 29 / 30 (Heuristic Scaling and `Map_dB_to_Lin`
//! / `Map_Lin_to_dB`) are deferred — when `f_pred_gain == 0` the spec
//! short-circuits to `env_alloc_mod == env_alloc` and `f_gain_q == 1`,
//! so the no-heuristic path covers any block with the predictor turned
//! off (the most common case in low-bitrate speech). The current SSF
//! walker decodes the coefficient indices using a *flat* `i_alloc_table`
//! seeded from `alloc_offset_bits` (see [`crate::ssf::parse_ssf_ac_data`]
//! for the in-spec degenerate case) — that matches what the no-heuristic
//! synthesis path consumes here.

use crate::ssf::{SsfBlock, SsfData, SsfGranule, StrideFlag};
use crate::ssf_ac::{idx_to_reconstruction, SsfRandGenState};
use crate::ssf_pred_coeff::{ssf_pred_coeff_mat, SSF_PRED_MAT_DIMS};
use crate::ssf_tables::{
    POST_GAIN_LUT, PRED_GAIN_QUANT_TAB, PRED_RFS_TABLE, PRED_RTS_TABLE, STEP_SIZES_Q4_15,
};

/// `NUM_SPEC_BUF` from §5.2.6 (subband predictor spectrum history depth).
pub const NUM_SPEC_BUF: usize = 5;

/// `NUM_ENV_BUF` from §5.2.6 (subband predictor envelope history depth).
pub const NUM_ENV_BUF: usize = 4;

/// `SSF_HIGH_FREQ_GAIN_THRESHOLD` from Pseudocode 4d — bands strictly
/// less than this get no gain applied.
pub const SSF_HIGH_FREQ_GAIN_THRESHOLD: usize = 2;

/// `ENV_MIN` / `ENV_MAX` from §5.2.3.0d.
pub const ENV_MIN: i32 = -64;
pub const ENV_MAX: i32 = 63;

/// `MIN_ALLOC_OFFSET` from Pseudocode 31.
pub const MIN_ALLOC_OFFSET: i32 = -21;

/// `ENV_MAX_2_MIN_OFFSET` from Pseudocode 31.
pub const ENV_MAX_2_MIN_OFFSET: i32 = 20;

/// `RFU_THRESHOLD` from Pseudocode 26.
pub const RFU_THRESHOLD: f32 = 0.75;

/// `ALLOC_DITHERING_THRESHOLD_SMALL` from Pseudocode 26.
pub const ALLOC_DITHERING_THRESHOLD_SMALL: i32 = 3;

/// `ALLOC_DITHERING_THRESHOLD_LARGE` from Pseudocode 26.
pub const ALLOC_DITHERING_THRESHOLD_LARGE: i32 = 5;

// === Pseudocode 4a: envelope decoding =======================================

/// §5.2.3.0a Pseudocode 4a: `env_idx[band] -> env[band]`.
///
/// `env_idx[0]` is the absolute envelope value already offset by
/// `ENV_BAND0_MIN = -28` upstream in [`crate::ssf::parse_ssf_ac_data`]
/// (it's stored that way on `SsfGranule::env_curr`), so this function
/// ONLY runs the differential `band >= 1` decode for the trailing
/// `num_bands - 1` indices.
pub fn decode_envelope(env_curr_offset: &[i32]) -> Vec<i32> {
    const ENV_DELTA_MIN: i32 = -16;
    let n = env_curr_offset.len();
    let mut env = vec![0i32; n];
    if n == 0 {
        return env;
    }
    env[0] = env_curr_offset[0];
    for band in 1..n {
        let delta = env_curr_offset[band] + ENV_DELTA_MIN;
        env[band] = env[band - 1] + delta;
    }
    env
}

// === Pseudocode 4b: envelope interpolation ==================================

/// §5.2.3.0b Pseudocode 4b: SHORT_STRIDE envelope interpolation.
///
/// Returns a `[block][band]` array of interpolated envelope values
/// (`Q.0` integer per the pseudocode). For LONG_STRIDE the caller must
/// instead set `env_interp[0] = env`; this function specifically
/// implements the SHORT_STRIDE path.
pub fn interpolate_envelope(env: &[i32], env_prev: &[i32], num_blocks: usize) -> Vec<Vec<i32>> {
    let n = env.len();
    let mut env_interp = vec![vec![0i32; n]; num_blocks];
    if num_blocks == 0 || n == 0 {
        return env_interp;
    }
    // The pseudocode is a fixed-point reformulation of the linear
    // interpolation `env_interp[block][band] = env_prev[band] +
    // (1 + block) / num_blocks * (env[band] - env_prev[band])` with
    // round-half-away-from-zero. We mirror the integer path exactly so
    // the result is bit-identical to the spec.
    let inv_num_blocks_q10: i32 = 256; // 1024 / 4 — only num_blocks == 4 in SHORT_STRIDE.
    for band in 0..n {
        let prev = if band < env_prev.len() {
            env_prev[band]
        } else {
            0
        };
        let left_delta_q10 = (env[band] - prev) * 1024;
        // iLeftSlope: (delta * inv_num_blocks) >> 10 in Q7.10.
        let left_slope_q10 = ((left_delta_q10 as i64 * inv_num_blocks_q10 as i64) >> 10) as i32;
        for (block, row) in env_interp.iter_mut().enumerate() {
            let interp_q10 = (1 + block as i32) * left_slope_q10 + prev * 1024;
            // Round half away from zero (the spec adds iHalf = 512 then
            // shifts right by 10).
            let rounded = if interp_q10 > 0 {
                (interp_q10 + 512) >> 10
            } else {
                -(((-interp_q10) + 512) >> 10)
            };
            row[band] = rounded;
        }
    }
    env_interp
}

// === Pseudocode 4c: gain decoding ===========================================

/// §5.2.3.0c Pseudocode 4c: `gain_idx[block] -> gain[block]`.
///
/// `gain_idx[block]` is `gain_bits[block] - 8` per `SsfBlock::gain_idx`.
/// LONG_STRIDE clamps `gain[0] = 1.0` (the gain index is forced to 0).
pub fn decode_gains(blocks: &[SsfBlock], stride_flag: StrideFlag) -> Vec<f32> {
    let num_blocks = blocks.len();
    let mut gain = vec![1.0_f32; num_blocks];
    if matches!(stride_flag, StrideFlag::LongStride) {
        // gain[0] = 1.0; nothing else to do — only one block.
        return gain;
    }
    for (block, slot) in gain.iter_mut().enumerate() {
        let idx = blocks[block].gain_idx() as f32;
        *slot = (10.0_f32).powf(idx * 0.1);
    }
    gain
}

// === Pseudocode 4d: envelope refinement =====================================

/// §5.2.3.0d Pseudocode 4d: produce `(env_alloc[block][band],
/// f_env_signal[block][band])`.
pub fn refine_envelope(
    env_interp: &[Vec<i32>],
    gain: &[f32],
    blocks: &[SsfBlock],
) -> (Vec<Vec<i32>>, Vec<Vec<f32>>) {
    let num_blocks = env_interp.len();
    let num_bands = env_interp.first().map(Vec::len).unwrap_or(0);
    let mut f_env_signal = vec![vec![0.0_f32; num_bands]; num_blocks];
    let mut env_alloc = vec![vec![0i32; num_bands]; num_blocks];
    for block in 0..num_blocks {
        for band in 0..num_bands {
            let interp = env_interp[block][band];
            let mut sig = (2.0_f32).powf(0.5 * interp as f32);
            if band >= SSF_HIGH_FREQ_GAIN_THRESHOLD && block < gain.len() {
                sig *= gain[block];
            }
            f_env_signal[block][band] = sig;
            let mut alloc = interp;
            if band >= SSF_HIGH_FREQ_GAIN_THRESHOLD && block < blocks.len() {
                let gain_idx = blocks[block].gain_idx();
                // round(2 * gain_idx / 3) — half-away-from-zero.
                let two = 2 * gain_idx;
                let q = if two >= 0 {
                    (two + 1) / 3
                } else {
                    -((-two + 1) / 3)
                };
                alloc += q;
                alloc = alloc.clamp(ENV_MIN, ENV_MAX);
            }
            env_alloc[block][band] = alloc;
        }
    }
    (env_alloc, f_env_signal)
}

// === Pseudocode 4e: predictor parameter calculation =========================

/// `(f_pred_gain[block], f_pred_lag[block])` from one block's parsed
/// `predictor_presence_flag` / `delta_flag` / `predictor_lag_*_bits`
/// + the per-channel `i_prev_pred_lag_idx` state.
///
/// Returns `(f_pred_gain, f_pred_lag)` and also mutates
/// `i_prev_pred_lag_idx` in place per Pseudocode 4e.
pub fn decode_predictor(
    block: &SsfBlock,
    block_idx: u32,
    start_block: u32,
    end_block: u32,
    i_prev_pred_lag_idx: &mut i32,
    n_mdct: u32,
) -> (f32, f32) {
    const PRED_LAG_DELTA_MIN: i32 = -8;
    let mut i_pred_lag_idx: i32 = 0;
    let mut f_pred_gain: f32 = 0.0;
    if block_idx >= start_block && block_idx < end_block && block.predictor_presence {
        let i_pred_gain_idx = block.pred_gain_idx.unwrap_or(0).clamp(0, 31);
        f_pred_gain = PRED_GAIN_QUANT_TAB[i_pred_gain_idx as usize];
        if block.delta_flag {
            i_pred_lag_idx = i32::from(block.predictor_lag_delta_bits)
                + *i_prev_pred_lag_idx
                + PRED_LAG_DELTA_MIN;
        } else {
            i_pred_lag_idx = i32::from(block.predictor_lag_bits);
        }
    }
    *i_prev_pred_lag_idx = i_pred_lag_idx;
    // f_pred_lag = 640 * 2^((i_pred_lag_idx - 509) / 170)
    let f_pred_lag = 640.0_f32 * (2.0_f32).powf((i_pred_lag_idx as f32 - 509.0) / 170.0);
    let _ = n_mdct;
    (f_pred_gain, f_pred_lag)
}

// === Pseudocode 26: helper variable calculation =============================

/// Helper variables produced by Pseudocode 26.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumHelpers {
    pub f_rfu: f32,
    pub i_alloc_dithering_threshold: i32,
    pub f_adaptive_noise_gain: f32,
    pub f_adaptive_noise_gain_var_pres: f32,
}

/// §5.2.5.2.1 Pseudocode 26.
pub fn compute_helpers(f_pred_gain: f32, variance_preserving: bool) -> SpectrumHelpers {
    let g = f_pred_gain;
    let f_rfu = if g < -1.0 {
        1.0
    } else if g < 0.0 {
        -g
    } else if g < 1.0 {
        g
    } else if g < 2.0 {
        2.0 - g
    } else {
        0.0
    };
    let mut i_alloc_dithering_threshold = if f_rfu > RFU_THRESHOLD {
        ALLOC_DITHERING_THRESHOLD_SMALL
    } else {
        ALLOC_DITHERING_THRESHOLD_LARGE
    };
    if variance_preserving {
        i_alloc_dithering_threshold = ALLOC_DITHERING_THRESHOLD_LARGE;
    }
    let f_adaptive_noise_gain_var_pres = (1.0 - f_rfu * f_rfu).max(0.0).sqrt();
    let f_adaptive_noise_gain = 1.0 - f_rfu;
    SpectrumHelpers {
        f_rfu,
        i_alloc_dithering_threshold,
        f_adaptive_noise_gain,
        f_adaptive_noise_gain_var_pres,
    }
}

// === Pseudocode 31: lossless decoding allocation table ======================

/// §5.2.5.2.3 Pseudocode 31 — derive `i_alloc_table[band]` from
/// `env_alloc_mod[band]` + `alloc_offset_bits`. The no-rfu path used
/// here sets `env_alloc_mod = env_alloc`.
pub fn build_alloc_table(env_alloc_mod: &[i32], alloc_offset_bits: u8) -> Vec<u32> {
    let n = env_alloc_mod.len();
    let i_alloc_offset = i32::from(alloc_offset_bits) + MIN_ALLOC_OFFSET;
    let mut i_max = env_alloc_mod.first().copied().unwrap_or(0);
    for &v in env_alloc_mod.iter().skip(1) {
        if v > i_max {
            i_max = v;
        }
    }
    i_max -= ENV_MAX_2_MIN_OFFSET;
    let mut tab = vec![0u32; n];
    for band in 0..n {
        let v = env_alloc_mod[band] - i_max + i_alloc_offset;
        let v = v.clamp(0, 20);
        tab[band] = v as u32;
    }
    tab
}

// === Pseudocode 33: MmseLaplace =============================================

/// §5.2.5.2.4 Pseudocode 33 — `MmseLaplace()` MMSE estimator for the
/// no-dither branch of the inverse quantizer.
pub fn mmse_laplace(f_mid_point: f32, f_step_size: f32) -> f32 {
    let s2 = std::f32::consts::SQRT_2;
    let f_upper = f_mid_point + f_step_size / 2.0;
    let f_lower = f_mid_point - f_step_size / 2.0;
    let f_pdf_lower = (s2 / 2.0) * (-f_lower.abs() * s2).exp();
    let f_pdf_upper = (s2 / 2.0) * (-f_upper.abs() * s2).exp();
    let f_pdf_lower = f_pdf_lower.max(0.0);
    let f_pdf_upper = f_pdf_upper.max(0.0);
    if f_lower > 0.0 {
        let num = f_pdf_upper * (s2 * f_upper + 1.0) - f_pdf_lower * (s2 * f_lower + 1.0);
        let den = s2 * (f_pdf_upper - f_pdf_lower);
        if den.abs() < 1e-12 {
            f_mid_point
        } else {
            num / den
        }
    } else if f_upper < 0.0 {
        let num = f_pdf_upper * (s2 * f_upper - 1.0) - f_pdf_lower * (s2 * f_lower - 1.0);
        let den = s2 * (f_pdf_upper - f_pdf_lower);
        if den.abs() < 1e-12 {
            f_mid_point
        } else {
            num / den
        }
    } else {
        let num = f_pdf_lower * (s2 * f_lower - 1.0) + f_pdf_upper * (s2 * f_upper + 1.0);
        let den = s2 * (f_pdf_lower + f_pdf_upper) - 2.0;
        if den.abs() < 1e-12 {
            f_mid_point
        } else {
            num / den
        }
    }
}

// === Pseudocode 32: inverse quantization ====================================

/// §5.2.5.2.4 Pseudocode 32 — produce `f_spec_invq[bin]` from
/// `i_quant_idx[bin]` for one block. `bands` is the inclusive
/// `(start_bin, end_bin)` per band.
#[allow(clippy::too_many_arguments)]
pub fn inverse_quantize_block(
    i_alloc_table: &[u32],
    bands: &[(usize, usize)],
    i_quant_idx: &[i32],
    helpers: &SpectrumHelpers,
    variance_preserving: bool,
    noise_rng: &mut SsfRandGenState,
    dither_rng: &mut SsfRandGenState,
    out: &mut [f32],
) {
    const I_MODEL_UNIT: f32 = (1u32 << 15) as f32;
    for (band_idx, &i_alloc) in i_alloc_table.iter().enumerate() {
        let (start_bin, end_bin) = bands[band_idx];
        let i_step_size = STEP_SIZES_Q4_15[i_alloc as usize];
        for bin in start_bin..=end_bin {
            if i_alloc == 0 {
                let mut s = noise_rng.random_noise_value();
                if variance_preserving && band_idx > 1 {
                    s *= helpers.f_adaptive_noise_gain_var_pres;
                } else {
                    s *= helpers.f_adaptive_noise_gain;
                }
                out[bin] = s;
            } else if (i_alloc as i32) < helpers.i_alloc_dithering_threshold {
                let i_dither = dither_rng.dither_value();
                let i_mid = idx_to_reconstruction(i_quant_idx[bin], i_dither, i_step_size);
                let f_mid = i_mid as f32 / I_MODEL_UNIT;
                let mut f_post_gain = POST_GAIN_LUT[(i_alloc - 1) as usize];
                if variance_preserving && band_idx > 1 {
                    let var_pres = f_post_gain.sqrt() * helpers.f_adaptive_noise_gain_var_pres;
                    if var_pres > f_post_gain {
                        f_post_gain = var_pres;
                    }
                }
                out[bin] = f_mid * f_post_gain;
            } else {
                let i_mid = idx_to_reconstruction(i_quant_idx[bin], 0, i_step_size);
                let f_mid = i_mid as f32 / I_MODEL_UNIT;
                let f_step = i_step_size as f32 / I_MODEL_UNIT;
                out[bin] = mmse_laplace(f_mid, f_step);
            }
        }
    }
}

// === Pseudocode 34: heuristic inverse scaling ===============================

/// §5.2.5.2.5 Pseudocode 34. With the no-rfu path `f_gain_q[band]` is
/// always 1.0, so the `f_gain_value = 1 / 1 = 1` multiplier is a
/// no-op; this helper is kept around for the future heuristic-scaling
/// landing.
pub fn inverse_heuristic_scale(spec_invq: &mut [f32], bands: &[(usize, usize)], f_gain_q: &[f32]) {
    for (band_idx, &(start_bin, end_bin)) in bands.iter().enumerate() {
        let g = if band_idx < f_gain_q.len() {
            f_gain_q[band_idx]
        } else {
            1.0
        };
        if (g - 1.0).abs() < 1e-9 {
            continue;
        }
        let inv = 1.0 / g;
        for slot in spec_invq.iter_mut().take(end_bin + 1).skip(start_bin) {
            *slot *= inv;
        }
    }
}

// === Pseudocode 39: C-matrix reconstruction ================================

/// §5.2.8.1 Pseudocode 39 — reconstruct the C-matrix for a given
/// `tab_idx` from the quantized prediction-coefficient bytes in
/// [`crate::ssf_pred_coeff`]. The quantized matrix has shape
/// `(2*Rf + 1) x 33 x Rt`, stored row-major in
/// `SSF_PRED_COEFF_MAT*[]` as `[k][eta][nu]` (Rt outer × 33 mid × (2Rf+1)
/// inner). Output is a `(2*Rf+1, 65, Rt)` float array indexed as
/// `[nu_idx][eta_idx][k]` where `nu_idx = nu + Rf` and `eta_idx = eta + 32`.
///
/// Returns a 3-d vector with the shape `[2*Rf+1][65][Rt]`.
pub fn build_c_matrix(tab_idx: usize) -> Option<Vec<Vec<Vec<f32>>>> {
    if tab_idx >= 37 {
        return None;
    }
    let rt = PRED_RTS_TABLE[tab_idx] as usize;
    let rf = PRED_RFS_TABLE[tab_idx] as usize;
    let two_rf_p1 = 2 * rf + 1;
    let raw = ssf_pred_coeff_mat(tab_idx)?;
    let (rows, cols) = SSF_PRED_MAT_DIMS[tab_idx];
    if rows != two_rf_p1 || cols != 33 * rt {
        return None;
    }
    // Allocate output [2*Rf+1][65][Rt] = float.
    let mut c = vec![vec![vec![0.0_f32; rt]; 65]; two_rf_p1];
    // Positive eta (0..=32). The raw bytes are stored row-major as
    // `(2*Rf+1) rows × (33 * Rt) cols`. Within each row of the raw
    // matrix the layout per Pseudocode 39 is `[k][eta]` — Rt outer ×
    // 33 inner. So the byte at `(nu_idx, k * 33 + eta)` maps to
    // `C[nu_idx][eta + 32][k]`.
    for (nu_idx, c_row) in c.iter_mut().enumerate().take(two_rf_p1) {
        for k in 0..rt {
            for eta in 0..33usize {
                let byte = raw[nu_idx * cols + k * 33 + eta] as i32;
                let recon = 1.1787855_f32 * (byte as f32 - 146.0) / 128.0;
                c_row[eta + 32][k] = recon;
            }
        }
    }
    // Negative eta. Pseudocode 39 mirror rule:
    //   C[nu][eta][k] = s * C[-nu][-eta][k] for eta in -32..0,
    //   where s starts as 1, then s *= -1 once per k iteration (so
    //   k=0 → s = -1, k=1 → s = +1, ...). Iterating by k is the
    //   spec-natural form; the 3-d aliasing prevents a clean
    //   iter_mut() refactor.
    #[allow(clippy::needless_range_loop)]
    for k in 0..rt {
        let s = if (k & 1) == 0 { -1.0_f32 } else { 1.0_f32 };
        for eta in -32_i32..0 {
            for nu in -(rf as i32)..=(rf as i32) {
                let nu_idx = (nu + rf as i32) as usize;
                let neg_nu_idx = (-nu + rf as i32) as usize;
                let eta_idx = (eta + 32) as usize;
                let neg_eta_idx = (-eta + 32) as usize;
                c[nu_idx][eta_idx][k] = s * c[neg_nu_idx][neg_eta_idx][k];
            }
        }
    }
    Some(c)
}

// === Pseudocode 35-37: subband predictor ====================================

/// State of the §5.2.6 subband predictor — `f_spec_buffer` (history of
/// `NUM_SPEC_BUF` previous spectra) and `f_env_buffer` (history of
/// `NUM_ENV_BUF` previous signal envelopes). Allocated lazily on first
/// `run` and reset on `clear`.
#[derive(Debug, Clone, Default)]
pub struct SubbandPredictorState {
    pub f_spec_buffer: Vec<Vec<f32>>, // NUM_SPEC_BUF entries.
    pub f_env_buffer: Vec<Vec<f32>>,  // NUM_ENV_BUF entries.
}

impl SubbandPredictorState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.f_spec_buffer.clear();
        self.f_env_buffer.clear();
    }

    /// §5.2.6 Pseudocode 35: shift the buffers, push new entries in.
    fn push_history(&mut self, f_spec_prev: &[f32], f_env_signal: &[f32]) {
        // f_spec_buffer: shift right (oldest dropped), push f_spec_prev at front.
        if self.f_spec_buffer.len() < NUM_SPEC_BUF {
            self.f_spec_buffer
                .resize(NUM_SPEC_BUF, vec![0.0_f32; f_spec_prev.len()]);
        }
        for i in (1..NUM_SPEC_BUF).rev() {
            self.f_spec_buffer[i] = self.f_spec_buffer[i - 1].clone();
        }
        self.f_spec_buffer[0] = f_spec_prev.to_vec();
        if self.f_env_buffer.len() < NUM_ENV_BUF {
            self.f_env_buffer
                .resize(NUM_ENV_BUF, vec![0.0_f32; f_env_signal.len()]);
        }
        for i in (1..NUM_ENV_BUF).rev() {
            self.f_env_buffer[i] = self.f_env_buffer[i - 1].clone();
        }
        self.f_env_buffer[0] = f_env_signal.to_vec();
    }

    /// §5.2.6 Pseudocodes 36 + 37 — produce `f_spec_pred[bin]`.
    ///
    /// Returns a `num_bins`-long vector. When the predictor is disabled
    /// (`f_pred_gain == 0`) returns a vector of zeros without touching
    /// the buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        f_spec_prev: &[f32],
        f_env_signal: &[f32],
        f_pred_gain: f32,
        f_pred_lag: f32,
        n_mdct: u32,
        bands: &[(usize, usize)],
        b_iframe: bool,
    ) -> Vec<f32> {
        let num_bins = f_spec_prev.len();
        // Always update history (the spec calls Pseudocode 35
        // unconditionally before the extraction).
        self.push_history(f_spec_prev, f_env_signal);
        if f_pred_gain == 0.0 || num_bins == 0 || n_mdct == 0 {
            return vec![0.0; num_bins];
        }
        // Pseudocode 36: model-based extraction.
        let mut f_period = f_pred_lag / n_mdct as f32;
        let k_s = if f_period > 81.0 / 32.0 {
            f_period -= 1.0;
            1usize
        } else {
            0usize
        };
        let tab_idx = if f_period <= 9.0 / 32.0 {
            0
        } else {
            ((16.0 * f_period + 0.5).floor() as i32 - 4).clamp(0, 36) as usize
        };
        let rt = PRED_RTS_TABLE[tab_idx] as usize;
        let rf = PRED_RFS_TABLE[tab_idx] as usize;
        let c = match build_c_matrix(tab_idx) {
            Some(c) => c,
            None => return vec![0.0; num_bins],
        };
        // Build Z[n][k]: n in -Rf..num_bins+Rf, k in 0..Rt.
        // Use offset indexing: z_idx = n + Rf. The 2-d aliasing across
        // separate rows of `z` (n + rf vs neg + rf) inside one k pass
        // doesn't fit a clean iter_mut refactor.
        let z_len = num_bins + 2 * rf;
        let mut z = vec![vec![0.0_f32; rt]; z_len];
        #[allow(clippy::needless_range_loop)]
        for k in 0..rt {
            let buf_idx = k + k_s;
            if buf_idx < self.f_spec_buffer.len() && self.f_spec_buffer[buf_idx].len() == num_bins {
                // Copy via temporary so we don't mut-borrow self twice.
                let copy: Vec<f32> = self.f_spec_buffer[buf_idx][..num_bins].to_vec();
                for (n, v) in copy.into_iter().enumerate() {
                    z[n + rf][k] = v;
                }
            }
            // Tail past num_bins is zero (already initialised).
            // Even reflection for n in -Rf..0: Z[n][k] = Z[-n-1][k].
            for n in (-(rf as i32))..0 {
                let neg = -n - 1; // 0..Rf
                z[(n + rf as i32) as usize][k] = z[(neg + rf as i32) as usize][k];
            }
        }
        // Extraction.
        let mut f_spec_extract = vec![0.0_f32; num_bins];
        for bin in 0..num_bins {
            let mut tmp = 0.0_f32;
            for nu in -(rf as i32)..=(rf as i32) {
                let mu = (2 * bin as i32 + nu + 1) as f32;
                let phi_unrounded = (f_period / 4.0) * mu;
                // round-half-to-even (the spec uses `round()` — we use
                // f32::round which is round-half-away-from-zero).
                let phi = phi_unrounded.round() - phi_unrounded;
                let f_int: i32 = if phi > f_period {
                    32
                } else if phi < -f_period {
                    -32
                } else {
                    let two_t = 2.0 * f_period;
                    let min_2t = if two_t < 1.0 { two_t } else { 1.0 };
                    if min_2t.abs() < 1e-12 {
                        0
                    } else {
                        (64.0 * phi / min_2t).round() as i32
                    }
                };
                let f_idx = (f_int + 32).clamp(0, 64) as usize;
                let nu_idx = (nu + rf as i32) as usize;
                // Pseudocode 36's `s = (bin % 2) ? s*(-1) : 1` resets s
                // to 1 at every iteration when bin is even, and flips
                // sign every k step when bin is odd.
                if bin % 2 == 0 {
                    for k in 0..rt {
                        let z_val = z[(bin as i32 + nu + rf as i32) as usize][k];
                        tmp += c[nu_idx][f_idx][k] * z_val;
                    }
                } else {
                    let mut s = 1.0_f32;
                    for k in 0..rt {
                        s *= -1.0;
                        let z_val = z[(bin as i32 + nu + rf as i32) as usize][k];
                        tmp += s * c[nu_idx][f_idx][k] * z_val;
                    }
                }
            }
            f_spec_extract[bin] = tmp;
        }
        // Pseudocode 37: shaper + prediction gain.
        let integer_lag_raw = (f_pred_lag / n_mdct as f32).round() as i32;
        let integer_lag = if b_iframe && integer_lag_raw > 0 {
            0usize
        } else {
            integer_lag_raw.max(0) as usize
        };
        let mut f_spec_pred = vec![0.0_f32; num_bins];
        let env_buf_len = self.f_env_buffer.len();
        let env_idx = integer_lag.min(env_buf_len.saturating_sub(1));
        for (band_idx, &(start_bin, end_bin)) in bands.iter().enumerate() {
            let env_v = self
                .f_env_buffer
                .get(env_idx)
                .and_then(|row| row.get(band_idx).copied())
                .unwrap_or(1.0);
            if env_v.abs() < 1e-12 {
                continue;
            }
            let f_envelope = 1.0 / env_v;
            for bin in start_bin..=end_bin {
                f_spec_pred[bin] = f_spec_extract[bin] * f_envelope * f_pred_gain;
            }
        }
        f_spec_pred
    }
}

// === Pseudocode 38: inverse flattening ======================================

/// §5.2.7 Pseudocode 38 — combine residual + predicted, apply signal
/// envelope per band. Output length = `num_bins`.
pub fn inverse_flatten(
    f_spec_res: &[f32],
    f_spec_pred: &[f32],
    f_env_signal: &[f32],
    bands: &[(usize, usize)],
) -> Vec<f32> {
    let num_bins = f_spec_res.len();
    let mut f_spec = vec![0.0_f32; num_bins];
    for bin in 0..num_bins {
        let p = if bin < f_spec_pred.len() {
            f_spec_pred[bin]
        } else {
            0.0
        };
        f_spec[bin] = f_spec_res[bin] + p;
    }
    for (band_idx, &(start_bin, end_bin)) in bands.iter().enumerate() {
        let g = if band_idx < f_env_signal.len() {
            f_env_signal[band_idx]
        } else {
            1.0
        };
        for slot in f_spec.iter_mut().take(end_bin + 1).skip(start_bin) {
            *slot *= g;
        }
    }
    f_spec
}

// === Top-level granule synthesis ============================================

/// Per-channel SSF synthesis state — the dither / noise RNGs, the
/// predictor lag history, the subband predictor's spec / env buffers
/// (§5.2.6), and a one-block latch for the previous block's `f_spec[]`
/// (the predictor's `f_spec_prev` input).
#[derive(Debug, Clone, Default)]
pub struct SsfSynthState {
    /// Dither RNG (Pseudocode 56) — independent of the bitstream walker's
    /// copy because the synthesis layer needs to advance it in step with
    /// `inverse_quantize_block`. Reset on every SSF-I-frame.
    pub dither_rng: SsfRandGenState,
    /// Noise RNG (Pseudocode 57) — same reset semantics.
    pub noise_rng: SsfRandGenState,
    /// Predictor lag history (`i_prev_pred_lag_idx`).
    pub prev_pred_lag_idx: i32,
    /// Subband predictor history.
    pub pred_state: SubbandPredictorState,
    /// Previous block's `f_spec[]` — the `f_spec_prev` input to
    /// Pseudocode 36 + 37.
    pub f_spec_prev: Vec<f32>,
}

impl SsfSynthState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset_iframe(&mut self) {
        self.dither_rng.reset();
        self.noise_rng.reset();
        self.pred_state.clear();
        self.f_spec_prev.clear();
        self.prev_pred_lag_idx = 0;
    }
}

/// Top-level SSF synthesis. Produces `num_blocks * n_mdct`-long
/// concatenated MDCT spectra (block-major, zero-padded out to `n_mdct`
/// after the `num_bins`-th coded line). Each `n_mdct`-chunk goes
/// straight into the AC-4 IMDCT.
pub fn synthesize_granule(
    granule: &SsfGranule,
    env_prev: &[i32],
    state: &mut SsfSynthState,
) -> Vec<f32> {
    let num_blocks = granule.num_blocks as usize;
    let n_mdct = granule.n_mdct as usize;
    let num_bins = granule.num_bins as usize;
    let num_bands = granule.num_bands as usize;

    // Reset RNG/predictor state on I-frames.
    if granule.b_iframe {
        state.reset_iframe();
    }

    // Pseudocode 4a — env_curr is already env_idx[0] absolute + delta-
    // decoded internally in `parse_ssf_ac_data`'s envelope loop. We
    // run Pseudocode 4a's chain explicitly here so we get true env[band]
    // (the walker leaves the partial decode in env_curr; we must
    // accumulate the delta chain).
    // env_curr[0] is already absolute (band-0 abs) per the walker; the
    // remaining entries are *raw symbol indices* (0..=32) that need
    // ENV_DELTA_MIN offset. We reconstruct env[] here.
    let env = decode_envelope(&granule.env_curr);

    // Pseudocode 4b/4c/4d — interpolation + gain + refinement.
    let env_interp: Vec<Vec<i32>> = match granule.stride_flag {
        StrideFlag::LongStride => vec![env.clone()],
        StrideFlag::ShortStride => {
            let prev_for_interp = if granule.b_iframe {
                granule
                    .env_startup
                    .as_deref()
                    .map(decode_envelope)
                    .unwrap_or_else(|| vec![0i32; num_bands])
            } else {
                env_prev.to_vec()
            };
            interpolate_envelope(&env, &prev_for_interp, num_blocks)
        }
    };
    let gain = decode_gains(&granule.blocks, granule.stride_flag);
    let (env_alloc, f_env_signal) = refine_envelope(&env_interp, &gain, &granule.blocks);

    // Build the bin layout (Pseudocode 7) again — the walker's layout
    // isn't exposed on SsfGranule but we can reconstruct it from
    // num_bands + n_mdct via the same path.
    let layout = match crate::ssf::SsfBinLayout::build(num_bands, granule.n_mdct) {
        Some(l) => l,
        None => return vec![0.0; num_blocks * n_mdct],
    };
    let bands: Vec<(usize, usize)> = (0..num_bands)
        .map(|b| (layout.start_bin[b] as usize, layout.end_bin[b] as usize))
        .collect();

    let mut out = vec![0.0_f32; num_blocks * n_mdct];
    for block_idx in 0..num_blocks {
        let blk = &granule.blocks[block_idx];
        // Pseudocode 4e — predictor.
        let (f_pred_gain, f_pred_lag) = decode_predictor(
            blk,
            block_idx as u32,
            granule.start_block,
            granule.end_block,
            &mut state.prev_pred_lag_idx,
            granule.n_mdct,
        );
        // Pseudocode 26 — helpers.
        let helpers = compute_helpers(f_pred_gain, blk.variance_preserving);
        // Pseudocode 31 — alloc table (no-rfu path: env_alloc_mod ==
        // env_alloc).
        let i_alloc_table = build_alloc_table(&env_alloc[block_idx], blk.alloc_offset_bits);
        // Pseudocode 32 — inverse quantization.
        let mut f_spec_invq = vec![0.0_f32; num_bins];
        inverse_quantize_block(
            &i_alloc_table,
            &bands,
            &blk.quant_idx,
            &helpers,
            blk.variance_preserving,
            &mut state.noise_rng,
            &mut state.dither_rng,
            &mut f_spec_invq,
        );
        // Pseudocode 34 — heuristic inverse scale (no-op when f_gain_q
        // is all 1s, which it is in the no-rfu path).
        let f_gain_q = vec![1.0_f32; num_bands];
        inverse_heuristic_scale(&mut f_spec_invq, &bands, &f_gain_q);
        let f_spec_res = f_spec_invq;
        // Pseudocode 35-37 — subband predictor. Uses f_spec_prev (the
        // previous block's *output*) — empty on the first I-block.
        let f_spec_prev_input = if state.f_spec_prev.len() == num_bins {
            state.f_spec_prev.clone()
        } else {
            vec![0.0_f32; num_bins]
        };
        let f_spec_pred = state.pred_state.run(
            &f_spec_prev_input,
            &f_env_signal[block_idx],
            f_pred_gain,
            f_pred_lag,
            granule.n_mdct,
            &bands,
            granule.b_iframe,
        );
        // Pseudocode 38 — inverse flattening.
        let f_spec = inverse_flatten(&f_spec_res, &f_spec_pred, &f_env_signal[block_idx], &bands);
        // Latch this block's output for the next block's predictor.
        state.f_spec_prev = f_spec.clone();
        // Lay into the block-major output (zero-padded out to n_mdct).
        let off = block_idx * n_mdct;
        let copy = f_spec.len().min(n_mdct);
        out[off..off + copy].copy_from_slice(&f_spec[..copy]);
    }
    out
}

/// Convenience: synthesize an `SsfData` (one or two granules) into a
/// flat MDCT-spectrum stream. Output length = sum of
/// `num_blocks * n_mdct` per granule.
pub fn synthesize_ssf_data(data: &SsfData, state: &mut SsfSynthState) -> Vec<f32> {
    let mut env_prev: Vec<i32> = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    for g in &data.granules {
        let spec = synthesize_granule(g, &env_prev, state);
        out.extend_from_slice(&spec);
        env_prev = decode_envelope(&g.env_curr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_envelope_handles_empty() {
        let env = decode_envelope(&[]);
        assert!(env.is_empty());
    }

    #[test]
    fn decode_envelope_chains_deltas() {
        // env_curr[0] is the absolute band-0 value (ENV_BAND0_MIN
        // pre-offset baked in upstream). Remaining entries are raw
        // symbols 0..=32; ENV_DELTA_MIN = -16 means symbol 16 → delta 0.
        let env = decode_envelope(&[-28, 16, 16, 17]);
        // band 0 = -28, band 1 += 16 - 16 = 0 → -28, band 2 += 0 → -28,
        // band 3 += 17 - 16 = 1 → -27.
        assert_eq!(env, vec![-28, -28, -28, -27]);
    }

    #[test]
    fn refine_envelope_low_bands_no_gain() {
        let env_interp = vec![vec![0i32, 0, 4, 4]];
        let gain = vec![2.0_f32];
        let blocks = vec![SsfBlock {
            gain_bits: 8 + 3,
            ..SsfBlock::default()
        }]; // gain_idx = 3.
        let (alloc, sig) = refine_envelope(&env_interp, &gain, &blocks);
        // Bands 0/1 below threshold → no gain.
        assert!((sig[0][0] - 1.0).abs() < 1e-5);
        assert!((sig[0][1] - 1.0).abs() < 1e-5);
        // Band 2 (>= threshold) → 2^(0.5*4) * gain = 4 * 2 = 8.
        assert!((sig[0][2] - 8.0).abs() < 1e-3);
        // Allocation: band 0/1 unchanged.
        assert_eq!(alloc[0][0], 0);
        assert_eq!(alloc[0][1], 0);
        // Band 2: 4 + round(2 * 3 / 3) = 4 + 2 = 6.
        assert_eq!(alloc[0][2], 6);
    }

    #[test]
    fn build_alloc_table_clamps_to_0_20() {
        // env_alloc_mod = [0, 5, 10, 15, 20], alloc_offset_bits = 21
        // → i_alloc_offset = 0, i_max = 20, then 20 - 20 = 0; for each
        // band: env - 0 + 0. Band 0 -> 0 (clamped), band 1 -> 5,
        // band 2 -> 10, band 3 -> 15, band 4 -> 20.
        let tab = build_alloc_table(&[0, 5, 10, 15, 20], 21);
        assert_eq!(tab, vec![0, 5, 10, 15, 20]);
    }

    #[test]
    fn build_alloc_table_negative_clips_to_zero() {
        // env_alloc_mod = [-50, -50], alloc_offset_bits = 0
        // → i_alloc_offset = -21, i_max = -50 - 20 = -70.
        // For each band: -50 - (-70) + (-21) = 20 - 21 = -1 -> clamp to 0.
        let tab = build_alloc_table(&[-50, -50], 0);
        assert_eq!(tab, vec![0, 0]);
    }

    #[test]
    fn helpers_no_predictor_gives_zero_rfu() {
        let h = compute_helpers(0.0, false);
        assert_eq!(h.f_rfu, 0.0);
        assert_eq!(h.f_adaptive_noise_gain, 1.0);
        assert!((h.f_adaptive_noise_gain_var_pres - 1.0).abs() < 1e-6);
        assert_eq!(
            h.i_alloc_dithering_threshold,
            ALLOC_DITHERING_THRESHOLD_LARGE
        );
    }

    #[test]
    fn helpers_pred_gain_in_unit_window() {
        let h = compute_helpers(0.5, false);
        assert!((h.f_rfu - 0.5).abs() < 1e-6);
        assert!((h.f_adaptive_noise_gain - 0.5).abs() < 1e-6);
    }

    #[test]
    fn predictor_off_returns_zero_gain_and_zero_lag_idx() {
        let mut prev = 100;
        let blk = SsfBlock {
            predictor_presence: false,
            ..SsfBlock::default()
        };
        let (g, _lag) = decode_predictor(&blk, 0, 0, 1, &mut prev, 960);
        assert_eq!(g, 0.0);
        // i_prev_pred_lag_idx is reset to 0 (block in live range, no
        // predictor presence path).
        assert_eq!(prev, 0);
    }

    #[test]
    fn predictor_with_lag_idx_resolves_to_pred_gain_quant() {
        let mut prev = 0;
        let blk = SsfBlock {
            predictor_presence: true,
            delta_flag: false,
            predictor_lag_bits: 200,
            pred_gain_idx: Some(0),
            ..SsfBlock::default()
        };
        let (g, _lag) = decode_predictor(&blk, 0, 0, 1, &mut prev, 960);
        assert!((g - PRED_GAIN_QUANT_TAB[0]).abs() < 1e-6);
        assert_eq!(prev, 200);
    }

    #[test]
    fn c_matrix_dimensions() {
        for tab_idx in 0..37 {
            let c = build_c_matrix(tab_idx).unwrap_or_else(|| panic!("tab_idx={tab_idx}"));
            let rt = PRED_RTS_TABLE[tab_idx] as usize;
            let rf = PRED_RFS_TABLE[tab_idx] as usize;
            assert_eq!(c.len(), 2 * rf + 1, "outer dim wrong for tab_idx={tab_idx}");
            assert_eq!(c[0].len(), 65, "mid dim wrong for tab_idx={tab_idx}");
            assert_eq!(c[0][0].len(), rt, "inner dim wrong for tab_idx={tab_idx}");
        }
    }

    #[test]
    fn c_matrix_negative_eta_mirror() {
        let c = build_c_matrix(0).expect("mat0");
        let rf = PRED_RFS_TABLE[0] as usize;
        let rt = PRED_RTS_TABLE[0] as usize;
        // Pseudocode 39: C[nu][eta][k] = s * C[-nu][-eta][k], where
        // s = -1 at k=0 for the first iteration and flips per k.
        #[allow(clippy::needless_range_loop)]
        for k in 0..rt {
            let s = if k % 2 == 0 { -1.0_f32 } else { 1.0 };
            for eta in 1..=32_i32 {
                for nu in -(rf as i32)..=(rf as i32) {
                    let pos_eta = eta;
                    let neg_eta = -eta;
                    let pos_nu_idx = (nu + rf as i32) as usize;
                    let neg_nu_idx = (-nu + rf as i32) as usize;
                    let pos = c[pos_nu_idx][(pos_eta + 32) as usize][k];
                    let neg = c[neg_nu_idx][(neg_eta + 32) as usize][k];
                    let expected = s * pos;
                    assert!(
                        (neg - expected).abs() < 1e-5,
                        "mirror failure k={k} nu={nu} eta={eta}: got {neg} expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn subband_predictor_passes_zero_pred_gain_through() {
        let mut s = SubbandPredictorState::new();
        let bands = vec![(0usize, 11usize)];
        let prev = vec![0.5_f32; 12];
        let env = vec![1.0_f32; 1];
        let out = s.run(&prev, &env, 0.0, 640.0, 960, &bands, true);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn subband_predictor_extraction_runs_without_panic() {
        // Smoke: with f_pred_gain != 0 and a small history, the
        // extraction loop should produce a finite output.
        let mut s = SubbandPredictorState::new();
        let bands = vec![(0usize, 11usize)];
        let prev = vec![0.1_f32; 12];
        let env = vec![1.0_f32; 1];
        // Push some history first by running with zero gain.
        s.run(&prev, &env, 0.0, 640.0, 960, &bands, true);
        let out = s.run(&prev, &env, 0.5, 640.0, 960, &bands, false);
        assert_eq!(out.len(), 12);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "bin {i} not finite: {v}");
        }
    }

    #[test]
    fn inverse_flatten_band_envelope_gain() {
        let res = vec![0.5_f32; 4];
        let pred = vec![0.0_f32; 4];
        let env = vec![2.0_f32];
        let bands = vec![(0usize, 3usize)];
        let f = inverse_flatten(&res, &pred, &env, &bands);
        for &v in &f {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn synthesize_granule_long_stride_zero_blocks_runs() {
        // A LONG_STRIDE I-granule with predictor disabled produces zero
        // PCM (all `i_alloc == 0` → noise from RNG; on this synthetic
        // path env_alloc is 0 throughout → i_alloc stays 0; output is
        // RNG-noise multiplied by f_env_signal).
        let granule = SsfGranule {
            b_iframe: true,
            stride_flag: StrideFlag::LongStride,
            num_bands: 12,
            start_block: 0,
            end_block: 0,
            num_blocks: 1,
            n_mdct: 960,
            num_bins: 140,
            env_curr_band0_bits: 0,
            env_startup_band0_bits: None,
            env_curr: vec![-28; 12],
            env_startup: None,
            blocks: vec![SsfBlock::default()],
            ac_bits_used: 30,
        };
        let mut state = SsfSynthState::new();
        let pcm_spec = synthesize_granule(&granule, &[], &mut state);
        assert_eq!(pcm_spec.len(), 960);
        // Output must be finite.
        for &v in &pcm_spec {
            assert!(v.is_finite());
        }
    }
}
