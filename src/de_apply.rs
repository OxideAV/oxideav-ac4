//! Dialogue-enhancement **application** (ETSI TS 103 190-1 §5.7.8).
//!
//! [`crate::de`] walks the `dialog_enhancement()` bitstream element;
//! this module turns a decoded [`DeConfig`] + [`DeData`] pair and the
//! user's enhancement gain G_DE into the §5.7.8.6 per-frame 3×3
//! enhancement matrices over the processed front-channel slots
//! (Table 171 order: Left, Right, Centre) and applies them, with the
//! §5.7.8.6 cross-frame interpolation, to QMF-domain channel data:
//!
//! * §5.7.8.3 dequantization — Table 209 (channel-independent) /
//!   Table 210 (cross-channel) parameter vectors, Table 172 mixing
//!   coefficients.
//! * §5.7.8.4 parameter bands — the Table 173 QMF-subband →
//!   dialogue-enhancement-band mapping (8 bands, QMF subbands 0..=40;
//!   higher subbands are outside the tool and pass through).
//! * §5.7.8.5 rendering — the r vector from `de_mix_coef1/2` with the
//!   energy-preserving last coefficient.
//! * §5.7.8.7 parametric channel-independent enhancement
//!   (`Y = (I + g·diag(p))·m`), including the M/S processing form for
//!   two processed channels with `de_ms_proc_flag`.
//! * §5.7.8.8 parametric cross-channel enhancement
//!   (`Y = (I + g·r·pᵀ)·m`).
//! * §5.7.8.9 waveform-parametric hybrid — the **parametric** half
//!   (`g_p = (1 − α_c)·g`); the isolated dialogue waveform channels
//!   are carried in a separate dialogue substream that this decoder
//!   does not consume, which §5.7.8.1 sanctions ("low-complexity
//!   decoders may use only the parametric data").
//!
//! The `de_keep_*` inheritance (§4.3.14.4.1/.3) and the §4.3.14.5.3
//! `de_par_prev` zero-default live in [`DeApplyState`], which also
//! carries the previous frame's matrices for interpolation.

use crate::de::{DeConfig, DeData, DeMethod, DE_NR_BANDS};
use crate::qmf::NUM_QMF_SUBBANDS;

/// Table 209 — dequantization vector for the channel-independent
/// enhancement modes (`de_par` index 0..=31).
pub const DE_PAR_DQ_CI: [f32; 32] = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5,
    3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
];

/// Table 172 — dequantization vector for `de_mix_coef1_idx` /
/// `de_mix_coef2_idx` (index 0..=31). The values are the printed
/// Table 172 entries verbatim (index 16 is the table's rounded
/// "0,7071", not the exact 1/√2).
#[allow(clippy::approx_constant)]
pub const DE_MIX_COEF_DQ: [f32; 32] = [
    0.0, 6.32e-3, 1.0e-2, 1.79e-2, 3.16e-2, 5.65e-2, 7.87e-2, 0.111, 0.156, 0.218, 0.303, 0.37,
    0.448, 0.533, 0.577, 0.622, 0.7071, 0.783, 0.846, 0.894, 0.929, 0.953, 0.976, 0.9877, 0.9938,
    0.9969, 0.9984, 0.9995, 0.99984, 0.99995, 0.99998, 1.0,
];

/// Table 173 — last QMF subband of each dialogue-enhancement parameter
/// band (`de_nr_bands = 8`; first subbands are 0, 1, 2, 4, 7, 11, 17,
/// 27).
pub const DE_BAND_LAST_SB: [usize; DE_NR_BANDS] = [0, 1, 3, 6, 10, 16, 26, 40];

/// §5.7.8.3 Table 209 lookup (channel-independent modes).
pub fn dequant_par_ci(idx: i32) -> f32 {
    DE_PAR_DQ_CI[idx.clamp(0, 31) as usize]
}

/// §5.7.8.3 Table 210 lookup (cross-channel modes): index −30..=30
/// maps linearly to −3,0..=3,0.
pub fn dequant_par_cc(idx: i32) -> f32 {
    idx.clamp(-30, 30) as f32 * 0.1
}

/// Table 173 mapping from QMF subband to dialogue-enhancement
/// parameter band. QMF subbands above 40 lie outside the tool's
/// banding and are passed through unprocessed.
pub fn sb_to_pb_de(sb: usize) -> Option<usize> {
    DE_BAND_LAST_SB.iter().position(|&last| sb <= last)
}

/// Per-band 3×3 enhancement matrices over the (Left, Right, Centre)
/// dialogue-enhancement slots — `h[band][row][col]`.
pub type DeBandMatrices = [[[f32; 3]; 3]; DE_NR_BANDS];

/// Identity matrices — the inactive-tool / §4.3.14.5.3 zero-parameter
/// state (`I + g·0`).
pub fn identity_matrices() -> DeBandMatrices {
    let mut h = [[[0.0f32; 3]; 3]; DE_NR_BANDS];
    for hb in h.iter_mut() {
        for (i, row) in hb.iter_mut().enumerate() {
            row[i] = 1.0;
        }
    }
    h
}

/// Which of the (Left, Right, Centre) slots the configuration
/// processes — Table 171: bit 2 = Left, bit 1 = Right, bit 0 = Centre.
pub fn processed_slots(cfg: &DeConfig) -> [bool; 3] {
    let c = cfg.channel_config & 0x7;
    [c & 0b100 != 0, c & 0b010 != 0, c & 0b001 != 0]
}

/// §5.7.8.7 / §5.7.8.8 gain: `g = 10^(G/20) − 1` with G clamped to the
/// configured `Gmax` (§4.3.14.3.2); hybrid modes scale by `(1 − α_c)`
/// (§5.7.8.9, parametric contribution).
fn parametric_gain(cfg: &DeConfig, data: &DeData, gain_db: f32) -> f32 {
    let g_db = gain_db.min(cfg.max_gain_db());
    let g = 10f32.powf(g_db / 20.0) - 1.0;
    match cfg.method {
        DeMethod::HybridChannelIndependent | DeMethod::HybridCrossChannel => {
            let alpha_c = data.signal_contribution.unwrap_or(0) as f32 / 31.0;
            (1.0 - alpha_c) * g
        }
        _ => g,
    }
}

/// Cross-frame state: the previous frame's matrices (interpolation
/// reference, identity when the tool was inactive), the last decoded
/// parameter rows (`de_keep_data_flag` / §4.3.14.5.3) and the last
/// mixing coefficients (`de_keep_pos_flag`).
#[derive(Debug, Clone, Default)]
pub struct DeApplyState {
    prev_h: Option<DeBandMatrices>,
    prev_par: Vec<[i32; DE_NR_BANDS]>,
    prev_mix: (Option<u8>, Option<u8>),
}

impl DeApplyState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to the inactive-tool state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Build the current frame's per-band matrices from a decoded
/// `dialog_enhancement()` payload, resolving the keep-flags against
/// `state` (which is updated with the resolved rows / coefficients).
pub fn build_frame_matrices(
    state: &mut DeApplyState,
    cfg: &DeConfig,
    data: &DeData,
    gain_db: f32,
) -> DeBandMatrices {
    let mut h = identity_matrices();
    let nr = cfg.nr_channels() as usize;
    if nr == 0 {
        state.prev_par.clear();
        return h;
    }
    // §4.3.14.4.3: keep_data inherits the previous rows; a fresh
    // payload replaces them. The decoded row count is
    // `de_nr_channels − de_ms_proc_flag`.
    let nr_rows = nr - usize::from(data.ms_proc_flag);
    if !data.de_par.is_empty() {
        state.prev_par = data.de_par.clone();
    }
    while state.prev_par.len() < nr_rows {
        // §4.3.14.5.3: undefined previous parameters are zero.
        state.prev_par.push([0i32; DE_NR_BANDS]);
    }
    let rows: Vec<[i32; DE_NR_BANDS]> = state.prev_par[..nr_rows].to_vec();
    // §4.3.14.4.1: keep_pos inherits the previous mixing coefficients.
    if data.mix_coef1_idx.is_some() {
        state.prev_mix = (data.mix_coef1_idx, data.mix_coef2_idx);
    }
    let (mix1, mix2) = state.prev_mix;
    let g = parametric_gain(cfg, data, gain_db);
    let slots = processed_slots(cfg);
    let slot_of_row: Vec<usize> = (0..3).filter(|&s| slots[s]).collect();
    let cross = matches!(
        cfg.method,
        DeMethod::CrossChannel | DeMethod::HybridCrossChannel
    );
    // §5.7.8.5 rendering vector (cross-channel modes).
    let r: Vec<f32> = if cross {
        match nr {
            1 => vec![1.0],
            2 => {
                let c1 = DE_MIX_COEF_DQ[mix1.unwrap_or(0) as usize & 31];
                vec![c1, (1.0 - c1 * c1).max(0.0).sqrt()]
            }
            _ => {
                let c1 = DE_MIX_COEF_DQ[mix1.unwrap_or(0) as usize & 31];
                let c2 = DE_MIX_COEF_DQ[mix2.unwrap_or(0) as usize & 31];
                vec![c1, c2, (1.0 - c1 * c1 - c2 * c2).max(0.0).sqrt()]
            }
        }
    } else {
        Vec::new()
    };
    for (band, hb) in h.iter_mut().enumerate() {
        if cross {
            // §5.7.8.8: H = I + g · r · pᵀ over the processed slots.
            for (i, &si) in slot_of_row.iter().enumerate() {
                for (j, &sj) in slot_of_row.iter().enumerate() {
                    let p_j = dequant_par_cc(rows[j][band]);
                    hb[si][sj] += g * r[i] * p_j;
                }
            }
        } else if data.ms_proc_flag && nr == 2 {
            // §5.7.8.7 M/S form: ½·[[1,1],[1,−1]]·diag(1+g·p₀, 1)·
            // [[1,1],[1,−1]] over the two processed slots.
            let a = 1.0 + g * dequant_par_ci(rows[0][band]);
            let (s0, s1) = (slot_of_row[0], slot_of_row[1]);
            hb[s0][s0] = 0.5 * (a + 1.0);
            hb[s0][s1] = 0.5 * (a - 1.0);
            hb[s1][s0] = 0.5 * (a - 1.0);
            hb[s1][s1] = 0.5 * (a + 1.0);
        } else {
            // §5.7.8.7: Y_i = m_i + g·p_i·m_i.
            for (i, &si) in slot_of_row.iter().enumerate() {
                hb[si][si] = 1.0 + g * dequant_par_ci(rows[i][band]);
            }
        }
    }
    h
}

/// Apply one frame of dialogue enhancement to the three
/// dialogue-enhancement slots' QMF data with the §5.7.8.6
/// interpolation:
///
/// ```text
///   H_DE(k, n) = (1 − (n+½)/N)·Ĥ(f−1) + ((n+½)/N)·Ĥ(f)
/// ```
///
/// `x[slot][ts][sb]` are the (Left, Right, Centre) slot matrices —
/// every column is transformed in place. Subbands above the Table 173
/// range pass through. The previous-frame reference defaults to the
/// identity (inactive tool / §4.3.14.5.3 zero parameters); `h_cur`
/// becomes the next frame's reference.
pub fn apply_frame_to_qmf(
    state: &mut DeApplyState,
    h_cur: DeBandMatrices,
    x: &mut [&mut Vec<[(f32, f32); NUM_QMF_SUBBANDS]>; 3],
) {
    let h_prev = state.prev_h.unwrap_or_else(identity_matrices);
    let num_ts = x[0].len();
    for ts in 0..num_ts {
        let w = (ts as f32 + 0.5) / num_ts.max(1) as f32;
        for sb in 0..NUM_QMF_SUBBANDS {
            let Some(band) = sb_to_pb_de(sb) else {
                continue;
            };
            let m: [(f32, f32); 3] = [x[0][ts][sb], x[1][ts][sb], x[2][ts][sb]];
            for (i, xi) in x.iter_mut().enumerate() {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for (j, &(mr, mi)) in m.iter().enumerate() {
                    let hij = (1.0 - w) * h_prev[band][i][j] + w * h_cur[band][i][j];
                    re += hij * mr;
                    im += hij * mi;
                }
                xi[ts][sb] = (re, im);
            }
        }
    }
    state.prev_h = Some(h_cur);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(method: DeMethod, max_gain: u8, channel_config: u8) -> DeConfig {
        DeConfig {
            method,
            max_gain,
            channel_config,
        }
    }

    fn data_rows(rows: Vec<[i32; DE_NR_BANDS]>) -> DeData {
        DeData {
            keep_pos_flag: false,
            mix_coef1_idx: None,
            mix_coef2_idx: None,
            keep_data_flag: false,
            ms_proc_flag: false,
            de_par: rows,
            signal_contribution: None,
        }
    }

    #[test]
    fn table_209_210_anchors() {
        assert_eq!(dequant_par_ci(0), 0.0);
        assert_eq!(dequant_par_ci(10), 1.0);
        assert_eq!(dequant_par_ci(16), 1.75);
        assert_eq!(dequant_par_ci(31), 9.0);
        assert_eq!(dequant_par_cc(0), 0.0);
        assert!((dequant_par_cc(-30) + 3.0).abs() < 1e-6);
        assert!((dequant_par_cc(30) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn table_173_band_mapping() {
        assert_eq!(sb_to_pb_de(0), Some(0));
        assert_eq!(sb_to_pb_de(1), Some(1));
        assert_eq!(sb_to_pb_de(2), Some(2));
        assert_eq!(sb_to_pb_de(3), Some(2));
        assert_eq!(sb_to_pb_de(4), Some(3));
        assert_eq!(sb_to_pb_de(7), Some(4));
        assert_eq!(sb_to_pb_de(11), Some(5));
        assert_eq!(sb_to_pb_de(17), Some(6));
        assert_eq!(sb_to_pb_de(27), Some(7));
        assert_eq!(sb_to_pb_de(40), Some(7));
        assert_eq!(sb_to_pb_de(41), None);
        assert_eq!(sb_to_pb_de(63), None);
    }

    #[test]
    fn channel_independent_diag_and_gmax_clamp() {
        // Centre-only (config 001), p = 1.0 everywhere, request 20 dB
        // against Gmax = (0+1)*3 = 3 dB → g = 10^(3/20) − 1.
        let c = cfg(DeMethod::ChannelIndependent, 0, 0b001);
        let d = data_rows(vec![[10i32; DE_NR_BANDS]]);
        let mut st = DeApplyState::new();
        let h = build_frame_matrices(&mut st, &c, &d, 20.0);
        let g = 10f32.powf(3.0 / 20.0) - 1.0;
        for hb in &h {
            assert!((hb[2][2] - (1.0 + g)).abs() < 1e-6);
            assert_eq!(hb[0][0], 1.0);
            assert_eq!(hb[1][1], 1.0);
            assert_eq!(hb[0][2], 0.0);
        }
    }

    #[test]
    fn ms_processing_two_channels() {
        // L+R (config 110) with de_ms_proc_flag: one Mid subset.
        let c = cfg(DeMethod::ChannelIndependent, 3, 0b110);
        let mut d = data_rows(vec![[10i32; DE_NR_BANDS]]);
        d.ms_proc_flag = true;
        let mut st = DeApplyState::new();
        let h = build_frame_matrices(&mut st, &c, &d, 6.0);
        let g = 10f32.powf(6.0 / 20.0) - 1.0;
        let a = 1.0 + g;
        for hb in &h {
            assert!((hb[0][0] - 0.5 * (a + 1.0)).abs() < 1e-6);
            assert!((hb[0][1] - 0.5 * (a - 1.0)).abs() < 1e-6);
            assert!((hb[1][0] - 0.5 * (a - 1.0)).abs() < 1e-6);
            assert!((hb[1][1] - 0.5 * (a + 1.0)).abs() < 1e-6);
            assert_eq!(hb[2][2], 1.0);
        }
    }

    #[test]
    fn cross_channel_rank_one_update() {
        // L+R (config 110), r from mix idx 16 (0,7071): H = I + g·r·pᵀ.
        let c = cfg(DeMethod::CrossChannel, 3, 0b110);
        let mut d = data_rows(vec![[10i32; DE_NR_BANDS], [-10i32; DE_NR_BANDS]]);
        d.mix_coef1_idx = Some(16);
        let mut st = DeApplyState::new();
        let h = build_frame_matrices(&mut st, &c, &d, 6.0);
        let g = 10f32.powf(6.0 / 20.0) - 1.0;
        let c1 = DE_MIX_COEF_DQ[16];
        let c2 = (1.0 - c1 * c1).max(0.0).sqrt();
        let (p0, p1) = (1.0f32, -1.0f32);
        for hb in &h {
            assert!((hb[0][0] - (1.0 + g * c1 * p0)).abs() < 1e-5);
            assert!((hb[0][1] - g * c1 * p1).abs() < 1e-5);
            assert!((hb[1][0] - g * c2 * p0).abs() < 1e-5);
            assert!((hb[1][1] - (1.0 + g * c2 * p1)).abs() < 1e-5);
            assert_eq!(hb[2][2], 1.0);
        }
    }

    #[test]
    fn hybrid_parametric_gain_scales_by_signal_contribution() {
        let c = cfg(DeMethod::HybridChannelIndependent, 3, 0b001);
        let mut d = data_rows(vec![[10i32; DE_NR_BANDS]]);
        d.signal_contribution = Some(31); // α_c = 1 → parametric g = 0.
        let mut st = DeApplyState::new();
        let h = build_frame_matrices(&mut st, &c, &d, 6.0);
        for hb in &h {
            assert!((hb[2][2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn keep_data_and_keep_pos_inherit_previous_frame() {
        let c = cfg(DeMethod::CrossChannel, 3, 0b110);
        let mut d = data_rows(vec![[5i32; DE_NR_BANDS], [7i32; DE_NR_BANDS]]);
        d.mix_coef1_idx = Some(20);
        let mut st = DeApplyState::new();
        let h1 = build_frame_matrices(&mut st, &c, &d, 6.0);
        // Keep-frame: parse leaves de_par empty + mix coefs None.
        let mut keep = data_rows(Vec::new());
        keep.keep_pos_flag = true;
        keep.keep_data_flag = true;
        let h2 = build_frame_matrices(&mut st, &c, &keep, 6.0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn interpolation_ramps_from_identity_and_settles() {
        let c = cfg(DeMethod::ChannelIndependent, 3, 0b100);
        let d = data_rows(vec![[10i32; DE_NR_BANDS]]);
        let mut st = DeApplyState::new();
        let g = 10f32.powf(6.0 / 20.0) - 1.0;
        let num_ts = 30usize;
        let ones = || vec![[(1.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let (mut l, mut r, mut cch) = (ones(), ones(), ones());
        let h = build_frame_matrices(&mut st, &c, &d, 6.0);
        apply_frame_to_qmf(&mut st, h, &mut [&mut l, &mut r, &mut cch]);
        // Frame 1 ramps from identity: the first timeslot is barely
        // boosted, the last one nearly fully.
        assert!(l[0][0].0 < 1.0 + g * 0.1);
        assert!(l[num_ts - 1][0].0 > 1.0 + g * 0.9);
        // Unprocessed slots and out-of-range subbands untouched.
        assert_eq!(r[0][0], (1.0, 0.0));
        assert_eq!(cch[num_ts - 1][5], (1.0, 0.0));
        assert_eq!(l[num_ts - 1][41], (1.0, 0.0));
        // Frame 2 interpolates cur↔cur: exact everywhere.
        let (mut l2, mut r2, mut c2) = (ones(), ones(), ones());
        let h2 = build_frame_matrices(&mut st, &c, &d, 6.0);
        apply_frame_to_qmf(&mut st, h2, &mut [&mut l2, &mut r2, &mut c2]);
        for col in l2.iter().take(num_ts) {
            assert!((col[3].0 - (1.0 + g)).abs() < 1e-5);
        }
    }
}
