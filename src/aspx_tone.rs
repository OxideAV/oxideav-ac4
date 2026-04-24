//! A-SPX tone generator (§5.7.6.4.4) + Table 196 SineTable.
//!
//! This module implements the ETSI TS 103 190-1 V1.4.1 A-SPX tone
//! generator tool (§5.7.6.4.4, Pseudocodes 104/105) plus the 4-entry
//! `SineTable` from Table 196. The tone generator adds complex
//! sinusoids into the HF part of the QMF matrix at positions driven
//! by the `aspx_hfgen_iwc_*` bitstream fields (`tic_used_in_slot` for
//! time-interleaved tones, plus `aspx_add_harmonic` via
//! `sine_idx_sb` from §5.7.6.4.2.1 Pseudocode 92).
//!
//! ## Data flow
//!
//! The caller is expected to have already derived:
//! * The frequency tables (sbx / sbz / num_sb_aspx) from §5.7.6.3.1.
//! * The signal envelope borders `atsg_sig[]` from Pseudocode 76
//!   (Table 194 for FIXFIX).
//! * The sine-level matrix `sine_lev_sb_adj[sb][atsg]` (shape
//!   `num_sb_aspx × num_atsg_sig`) from the envelope-adjustment tool
//!   (§5.7.6.4.2 Pseudocode 94 / Pseudocode 100 boost).
//!
//! Combined with the TIC (time-interleaved waveform coding) slot
//! bitmap from `aspx_hfgen_iwc_1ch / _2ch`, this module builds
//! `qmf_sine[sb][ts]` for the current interval and exposes an adder
//! that the HF signal assembler (§5.7.6.4.5 Pseudocode 108) can use.

/// Table 196 — SineTable for the tone generator tool.
///
/// `SINE_TABLE[k] = (SineTable_RE(k), SineTable_IM(k))` for
/// `k ∈ {0, 1, 2, 3}`. These are the four quadrant unit phases:
/// `+1`, `+i`, `-1`, `-i`.
pub const SINE_TABLE: [(f32, f32); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// Output of the A-SPX tone generator — a `qmf_sine[sb][ts]` complex
/// matrix plus the stateful `sine_idx_prev` used to seed the next
/// A-SPX interval's `sine_idx()` helper (Pseudocode 105).
#[derive(Debug, Clone, Default)]
pub struct QmfSine {
    /// `qmf_sine[sb][ts]` — complex sinusoid matrix. Absolute subband
    /// indexing (0..num_qmf_subbands). Entries outside the A-SPX
    /// range `sbx..sbz` or outside the current interval stay zero.
    pub qmf_sine: Vec<Vec<(f32, f32)>>,
    /// Last `sine_idx` produced during the generator run, keyed by
    /// absolute subband.
    pub last_indices: Vec<Vec<u32>>,
}

/// State carried across A-SPX intervals for the tone generator.
#[derive(Debug, Clone, Default)]
pub struct ToneGenState {
    /// `sine_idx_prev[sb][ts]` — per Pseudocode 105, the last
    /// `sine_idx` from the previous A-SPX interval. `None` on the
    /// first frame (`first_frame == 1` in Pseudocode 105).
    pub prev: Option<Vec<Vec<u32>>>,
}

impl ToneGenState {
    /// Construct a fresh state (first-frame, `first_frame == 1`).
    pub fn new() -> Self {
        Self { prev: None }
    }

    /// Reset the state to the `first_frame == 1` initial condition.
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

/// Compute `sine_idx(sb, ts)` per ETSI TS 103 190-1 §5.7.6.4.4
/// Pseudocode 105.
///
/// ```text
/// sine_idx(sb, ts)
/// {
///     if (first_frame) {
///         index = 1;
///     } else {
///         index = (sine_idx_prev[sb][ts] + 1) % 4;
///     }
///     index += ts - atsg_sig[0];
///     return index % 4;
/// }
/// ```
///
/// `ts` here is a QMF time slot — the outer loop in Pseudocode 104
/// runs `ts` from `atsg_sig[0]*num_ts_in_ats` to
/// `atsg_sig[num_atsg_sig]*num_ts_in_ats`, and the `ts - atsg_sig[0]`
/// expression there treats `atsg_sig[0]` in A-SPX timeslots. The
/// caller passes `atsg_sig0_qmf_slots = atsg_sig[0] * num_ts_in_ats`
/// so this function operates entirely in QMF slots.
#[inline]
pub fn sine_idx(ts: u32, atsg_sig0_qmf_slots: u32, sine_idx_prev: Option<u32>) -> u32 {
    let base = match sine_idx_prev {
        Some(p) => (p + 1) % 4,
        None => 1,
    };
    let offset = ts.saturating_sub(atsg_sig0_qmf_slots);
    (base + offset) % 4
}

/// Generate the complex tone QMF matrix per ETSI TS 103 190-1
/// §5.7.6.4.4 Pseudocode 104.
///
/// * `sine_lev_sb_adj[sb][atsg_sig]` — per-subband, per-signal-
///   envelope boosted sine level (shape `num_sb_aspx × num_atsg_sig`).
/// * `atsg_sig` — signal-envelope borders (A-SPX timeslots,
///   `num_atsg_sig + 1` entries).
/// * `num_ts_in_ats` — 1 or 2 (Table 192).
/// * `num_qmf_subbands` — almost always 64 in AC-4.
/// * `sbx` — crossover subband.
/// * `num_sb_aspx` — `sbz - sbx`.
/// * `state` — cross-interval sine index state (advance on return).
///
/// The per-slot level is looked up on `atsg_sig`; the `(-1)^(sb+sbx)`
/// factor on the imaginary side comes straight from Pseudocode 104:
/// `qmf_sine_IM[sb][ts] *= pow(-1, sb+sbx) * SineTable_IM[idx]`.
#[allow(clippy::too_many_arguments)]
pub fn generate_qmf_sine(
    sine_lev_sb_adj: &[Vec<f32>],
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
    num_qmf_subbands: u32,
    sbx: u32,
    num_sb_aspx: u32,
    state: &mut ToneGenState,
) -> QmfSine {
    let num_atsg_sig = atsg_sig.len().saturating_sub(1);
    if num_atsg_sig == 0 || num_sb_aspx == 0 {
        return QmfSine::default();
    }
    let ts_start = atsg_sig[0].saturating_mul(num_ts_in_ats);
    let ts_end = atsg_sig[num_atsg_sig].saturating_mul(num_ts_in_ats);
    let ts_total = ts_end as usize;
    let mut qmf_sine: Vec<Vec<(f32, f32)>> = (0..num_qmf_subbands)
        .map(|_| vec![(0.0_f32, 0.0_f32); ts_total])
        .collect();
    let mut last_indices: Vec<Vec<u32>> = (0..num_qmf_subbands)
        .map(|_| vec![0_u32; ts_total])
        .collect();
    let atsg_sig0_qmf = atsg_sig[0].saturating_mul(num_ts_in_ats);
    let mut atsg: usize = 0;
    for ts in ts_start..ts_end {
        while atsg + 1 < num_atsg_sig && ts >= atsg_sig[atsg + 1].saturating_mul(num_ts_in_ats) {
            atsg += 1;
        }
        for sb in 0..(num_sb_aspx as usize) {
            let sb_abs = sb + sbx as usize;
            if sb_abs >= qmf_sine.len() {
                break;
            }
            let prev = state.prev.as_ref().and_then(|mat| {
                mat.get(sb_abs)
                    .and_then(|row| row.get(ts as usize).copied())
            });
            let idx = sine_idx(ts, atsg_sig0_qmf, prev);
            last_indices[sb_abs][ts as usize] = idx;
            let (st_re, st_im) = SINE_TABLE[idx as usize];
            let lev = sine_lev_sb_adj
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            // Pseudocode 104:
            //   qmf_sine_RE[sb][ts] = lev * SineTable_RE[idx]
            //   qmf_sine_IM[sb][ts] = lev * pow(-1, sb+sbx) *
            //                         SineTable_IM[idx]
            let sign = if (sb_abs) & 1 == 0 { 1.0_f32 } else { -1.0_f32 };
            qmf_sine[sb_abs][ts as usize] = (lev * st_re, lev * sign * st_im);
        }
    }
    state.prev = Some(last_indices.clone());
    QmfSine {
        qmf_sine,
        last_indices,
    }
}

/// Add the generated `qmf_sine` into an existing QMF high-band matrix
/// `y` in place per ETSI TS 103 190-1 §5.7.6.4.5 Pseudocode 108.
pub fn add_qmf_sine(
    y: &mut [Vec<(f32, f32)>],
    qmf_sine: &QmfSine,
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
    sbx: u32,
    sbz: u32,
) {
    let num_atsg_sig = atsg_sig.len().saturating_sub(1);
    if num_atsg_sig == 0 {
        return;
    }
    let ts_start = atsg_sig[0].saturating_mul(num_ts_in_ats) as usize;
    let ts_end = atsg_sig[num_atsg_sig].saturating_mul(num_ts_in_ats) as usize;
    for sb in (sbx as usize)..(sbz as usize) {
        if sb >= y.len() || sb >= qmf_sine.qmf_sine.len() {
            break;
        }
        let src = &qmf_sine.qmf_sine[sb];
        let dst = &mut y[sb];
        let hi = ts_end.min(dst.len()).min(src.len());
        for ts in ts_start..hi {
            dst[ts].0 += src[ts].0;
            dst[ts].1 += src[ts].1;
        }
    }
}

/// Build a `sine_lev_sb_adj` matrix that places a sinusoid of unit
/// level at every QMF subband marked `true` in `tone_mask_sb`. Zero
/// elsewhere. Intended as a simple bridge between the parsed
/// `aspx_hfgen_iwc_*` TIC / FIC / harmonic flags and the tone
/// generator.
///
/// For the per-timeslot `tic_used_in_slot[ts]` bitmap alone, pass a
/// mask that sets every row of the matrix the same way — that's
/// enough to produce audible tones in the HF band. A full
/// implementation drives `sine_idx_sb` through §5.7.6.4.2.1
/// Pseudocode 92 / 93 which combines `aspx_add_harmonic` with the
/// signal-envelope subband-group structure.
pub fn level_matrix_from_flags(
    tone_mask_sb: &[bool],
    num_atsg_sig: u32,
    level: f32,
) -> Vec<Vec<f32>> {
    tone_mask_sb
        .iter()
        .map(|&on| {
            let v = if on { level } else { 0.0 };
            vec![v; num_atsg_sig as usize]
        })
        .collect()
}

// ---------------------------------------------------------------------
// Combined HF assembler entry point (§5.7.6.4.5 Pseudocodes 106/107/108)
// ---------------------------------------------------------------------

/// Assemble the final A-SPX HF matrix from a tile-copied / envelope-
/// adjusted high band `y`, the noise generator output `qmf_noise`,
/// and the tone generator output `qmf_sine`, in-place on `y`.
///
/// Implements the additive parts of §5.7.6.4.5:
/// * Pseudocode 107 — add `qmf_noise` into `y`.
/// * Pseudocode 108 — add `qmf_sine` into `y`.
///
/// The gain-multiplication step of Pseudocode 106 is the
/// responsibility of the envelope-adjustment tool (§5.7.6.4.2) — the
/// caller is expected to have already multiplied `y` by
/// `sig_gain_sb_adj[sb][atsg]`. This routine only handles the
/// additive noise and tone contributions.
pub fn hf_assemble(
    y: &mut [Vec<(f32, f32)>],
    qmf_noise: &crate::aspx_noise::QmfNoise,
    qmf_sine: &QmfSine,
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
    sbx: u32,
    sbz: u32,
) {
    crate::aspx_noise::add_qmf_noise(y, qmf_noise, atsg_sig, num_ts_in_ats, sbx, sbz);
    add_qmf_sine(y, qmf_sine, atsg_sig, num_ts_in_ats, sbx, sbz);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_table_matches_table_196() {
        assert_eq!(SINE_TABLE[0], (1.0, 0.0));
        assert_eq!(SINE_TABLE[1], (0.0, 1.0));
        assert_eq!(SINE_TABLE[2], (-1.0, 0.0));
        assert_eq!(SINE_TABLE[3], (0.0, -1.0));
    }

    #[test]
    fn sine_idx_first_frame_starts_at_1() {
        // first_frame branch: index = 1, then index += ts - atsg_sig[0].
        // With atsg_sig[0] * num_ts_in_ats = 0:
        //   sine_idx(0) = 1
        //   sine_idx(1) = 2
        //   sine_idx(2) = 3
        //   sine_idx(3) = 0 (wraps)
        //   sine_idx(4) = 1
        assert_eq!(sine_idx(0, 0, None), 1);
        assert_eq!(sine_idx(1, 0, None), 2);
        assert_eq!(sine_idx(2, 0, None), 3);
        assert_eq!(sine_idx(3, 0, None), 0);
        assert_eq!(sine_idx(4, 0, None), 1);
    }

    #[test]
    fn sine_idx_continues_from_prev() {
        // Non-first-frame: index = (prev + 1) % 4, then += ts - start.
        // prev = 2, ts = 0, start = 0 -> (2+1)%4 + 0 = 3
        // prev = 3, ts = 0, start = 0 -> (3+1)%4 + 0 = 0
        // prev = 1, ts = 5, start = 0 -> 2 + 5 = 7 % 4 = 3
        assert_eq!(sine_idx(0, 0, Some(2)), 3);
        assert_eq!(sine_idx(0, 0, Some(3)), 0);
        assert_eq!(sine_idx(5, 0, Some(1)), 3);
    }

    #[test]
    fn sine_idx_offset_by_atsg_start() {
        // atsg_sig0 QMF slot = 4 -> idx computed relative to that.
        // first_frame branch, ts = 4 -> 1 + 0 = 1.
        // ts = 5 -> 1 + 1 = 2.
        assert_eq!(sine_idx(4, 4, None), 1);
        assert_eq!(sine_idx(5, 4, None), 2);
        assert_eq!(sine_idx(7, 4, None), 0);
    }

    #[test]
    fn generator_places_tones_only_in_marked_subbands() {
        // Two atsg time slots, one signal envelope covering both.
        // 4-wide A-SPX range, sbx = 8.
        let atsg_sig = vec![0_u32, 2];
        // Tone at sb=8, 11; zero at 9, 10.
        let tone_mask_sb = vec![true, false, false, true];
        let sine_lev = level_matrix_from_flags(&tone_mask_sb, 1, 1.0);
        let mut state = ToneGenState::new();
        let out = generate_qmf_sine(&sine_lev, &atsg_sig, 1, 64, 8, 4, &mut state);
        // Zero outside the A-SPX range.
        for sb in 0..8 {
            for ts in 0..2 {
                assert_eq!(out.qmf_sine[sb][ts], (0.0, 0.0));
            }
        }
        // sb=9, 10 have zero level -> zero output.
        for sb in 9..=10 {
            for ts in 0..2 {
                assert_eq!(out.qmf_sine[sb][ts], (0.0, 0.0));
            }
        }
        // sb=8: first_frame -> idx(ts=0)=1 -> (0,1); idx(ts=1)=2 -> (-1,0).
        // sb=8 is even => sign = +1 for IM.
        // level 1.0. So:
        //   ts=0: RE=1*0=0, IM=1*1*1=1
        //   ts=1: RE=1*-1=-1, IM=1*1*0=0
        assert!((out.qmf_sine[8][0].0 - 0.0).abs() < 1e-6);
        assert!((out.qmf_sine[8][0].1 - 1.0).abs() < 1e-6);
        assert!((out.qmf_sine[8][1].0 - (-1.0)).abs() < 1e-6);
        assert!((out.qmf_sine[8][1].1 - 0.0).abs() < 1e-6);
        // sb=11 is odd => sign = -1 for IM.
        //   ts=0: RE=1*0=0, IM=1*(-1)*1 = -1
        //   ts=1: RE=1*(-1)=-1, IM=0
        assert!((out.qmf_sine[11][0].0 - 0.0).abs() < 1e-6);
        assert!((out.qmf_sine[11][0].1 - (-1.0)).abs() < 1e-6);
        assert!((out.qmf_sine[11][1].0 - (-1.0)).abs() < 1e-6);
        assert!((out.qmf_sine[11][1].1 - 0.0).abs() < 1e-6);
        assert!(state.prev.is_some());
    }

    #[test]
    fn generator_scales_by_level() {
        // Single tone at sb=4, level 3.5, duration 4 slots.
        let atsg_sig = vec![0_u32, 4];
        let sine_lev = vec![vec![3.5_f32]; 2];
        let mut state = ToneGenState::new();
        let out = generate_qmf_sine(&sine_lev, &atsg_sig, 1, 64, 4, 2, &mut state);
        // sb=4 (even, sign=+1). idx walks: 1, 2, 3, 0.
        //   ts=0 idx=1 -> (0, 1) * 3.5 = (0, 3.5)
        //   ts=1 idx=2 -> (-1, 0) * 3.5 = (-3.5, 0)
        //   ts=2 idx=3 -> (0, -1) * 3.5 = (0, -3.5)
        //   ts=3 idx=0 -> (1, 0) * 3.5 = (3.5, 0)
        assert!((out.qmf_sine[4][0].1 - 3.5).abs() < 1e-5);
        assert!((out.qmf_sine[4][1].0 + 3.5).abs() < 1e-5);
        assert!((out.qmf_sine[4][2].1 + 3.5).abs() < 1e-5);
        assert!((out.qmf_sine[4][3].0 - 3.5).abs() < 1e-5);
    }

    #[test]
    fn generator_picks_correct_sig_envelope() {
        // Two signal envelopes with different levels on the same tone.
        let atsg_sig = vec![0_u32, 2, 4];
        // sb 0: envelope 0 level=1, envelope 1 level=4.
        let sine_lev = vec![vec![1.0_f32, 4.0_f32], vec![0.0_f32, 0.0_f32]];
        let mut state = ToneGenState::new();
        let out = generate_qmf_sine(&sine_lev, &atsg_sig, 1, 64, 4, 2, &mut state);
        // First two slots use level 1. First slot: idx=1 (+0,+1),
        // sb=4 even.
        assert!((out.qmf_sine[4][0].1 - 1.0).abs() < 1e-5);
        // Last two slots use level 4. Slot 2: idx=3 -> (0,-1) * 4 (sign +1)
        assert!((out.qmf_sine[4][2].1 + 4.0).abs() < 1e-5);
    }

    #[test]
    fn add_qmf_sine_adds_to_existing_matrix() {
        let atsg_sig = vec![0_u32, 2];
        let sine_lev = vec![vec![1.0_f32]; 2];
        let mut state = ToneGenState::new();
        let out = generate_qmf_sine(&sine_lev, &atsg_sig, 1, 64, 4, 2, &mut state);
        let mut y: Vec<Vec<(f32, f32)>> = (0..64).map(|_| vec![(0.1_f32, 0.2_f32); 2]).collect();
        add_qmf_sine(&mut y, &out, &atsg_sig, 1, 4, 6);
        // Unchanged outside 4..6.
        for sb in 0..4 {
            assert_eq!(y[sb][0], (0.1, 0.2));
        }
        for sb in 6..64 {
            assert_eq!(y[sb][0], (0.1, 0.2));
        }
        // Inside 4..6: tone added on top.
        // sb=4 even (+1 sign), idx(0)=1 -> (0,1) * 1 = (0,1).
        // y[4][0] = (0.1 + 0, 0.2 + 1).
        assert!((y[4][0].0 - 0.1).abs() < 1e-5);
        assert!((y[4][0].1 - 1.2).abs() < 1e-5);
    }

    #[test]
    fn level_matrix_from_flags_preserves_shape() {
        let tone_mask_sb = vec![true, false, true, true];
        let m = level_matrix_from_flags(&tone_mask_sb, 3, 2.5);
        assert_eq!(m.len(), 4);
        assert_eq!(m[0], vec![2.5_f32; 3]);
        assert_eq!(m[1], vec![0.0_f32; 3]);
        assert_eq!(m[2], vec![2.5_f32; 3]);
        assert_eq!(m[3], vec![2.5_f32; 3]);
    }

    #[test]
    fn hf_assemble_adds_both_noise_and_tone() {
        use crate::aspx_noise::{generate_qmf_noise, NoiseGenState};
        let atsg_sig = vec![0_u32, 2];
        let atsg_noise = vec![0_u32, 2];
        let noise_lev = vec![vec![1.0_f32]; 2];
        let sine_lev = vec![vec![1.0_f32]; 2];
        let mut ns = NoiseGenState::new();
        let mut ts = ToneGenState::new();
        let noise = generate_qmf_noise(&noise_lev, &atsg_sig, &atsg_noise, 1, 64, 4, 2, &mut ns);
        let sine = generate_qmf_sine(&sine_lev, &atsg_sig, 1, 64, 4, 2, &mut ts);
        let mut y: Vec<Vec<(f32, f32)>> = (0..64).map(|_| vec![(0.0_f32, 0.0_f32); 2]).collect();
        hf_assemble(&mut y, &noise, &sine, &atsg_sig, 1, 4, 6);
        // Contribution at sb=4, ts=0:
        //   noise: ASPX_NOISE_TABLE[noise_idx(4, 0, 0, 1, 2, None)]
        //          noise_idx = 0 + 2*0 + 5 = 5 -> ASPX_NOISE_TABLE[5]
        //   tone:  sb=4 even, first_frame, ts=0 -> idx=1 -> (0, 1)
        let idx = crate::aspx_noise::noise_idx(4, 0, 0, 1, 2, None);
        let (nre, nim) = crate::aspx_noise::ASPX_NOISE_TABLE[idx as usize];
        assert!((y[4][0].0 - (nre + 0.0)).abs() < 1e-5);
        assert!((y[4][0].1 - (nim + 1.0)).abs() < 1e-5);
        // Outside A-SPX range — still zero.
        for sb in 0..4 {
            for t in 0..2 {
                assert_eq!(y[sb][t], (0.0, 0.0));
            }
        }
        for sb in 6..64 {
            for t in 0..2 {
                assert_eq!(y[sb][t], (0.0, 0.0));
            }
        }
    }

    #[test]
    fn tone_gen_state_resets() {
        let mut state = ToneGenState::new();
        let atsg_sig = vec![0_u32, 1];
        let sine_lev = vec![vec![1.0_f32]; 2];
        let _ = generate_qmf_sine(&sine_lev, &atsg_sig, 1, 64, 2, 2, &mut state);
        assert!(state.prev.is_some());
        state.reset();
        assert!(state.prev.is_none());
    }
}
