//! Encoder-side `aspx_tna_mode` selection (A-SPX subband tonal-to-noise
//! ratio adjustment / inverse-filtering mode).
//!
//! ETSI TS 103 190-1 §4.3.10.6.1 (Table 131) defines the **decode**
//! meaning of the 2-bit `aspx_tna_mode[sbg]`:
//!
//! | `aspx_tna_mode` | Meaning  |
//! | --------------- | -------- |
//! | 0               | None     |
//! | 1               | Light    |
//! | 2               | Moderate |
//! | 3               | Heavy    |
//!
//! and §5.7.6.4.1.2–§5.7.6.4.1.4 (Pseudocodes 85–89) define how the
//! decoder uses it: the per-noise-subband-group chirp factor
//! `tabNewChirp[prev][curr]` (Table 195) scales the order-2 complex LPC
//! residual filter (`alpha0`, `alpha1`) that is applied to the low band
//! *before* it is transposed into the high band:
//!
//! ```text
//!   Q_high[sb_high][ts]  = Q_low_ext[p][n]
//!   Q_high[sb_high][ts] += chirp_arr[g]   * alpha0[p] * Q_low_ext[p][n-2]
//!   Q_high[sb_high][ts] += chirp_arr[g]^2 * alpha1[p] * Q_low_ext[p][n-4]
//! ```
//!
//! The **selection** of `aspx_tna_mode` is an encoder decision the spec
//! leaves to the implementer (it is purely informative — any value in
//! `0..=3` produces a decodable bitstream). The amount of adjustment is,
//! per §4.3.10.6.1, *"proportional to a value calculated based on the
//! values of `aspx_tna_mode`"* — i.e. heavier modes apply more inverse
//! filtering of the tonal low-band carryover into the high band. A
//! principled clean-room encoder therefore signals a **heavier mode where
//! the low band is more tonal** (more strongly predictable by the order-2
//! complex predictor), because a tonal low band copied verbatim into the
//! high band would create spurious tonal artifacts that the chirp-scaled
//! LPC residual is designed to suppress.
//!
//! The tonality measure is taken directly from the spec's own analysis:
//! the magnitude energy of the `alpha0` / `alpha1` predictor coefficients
//! computed by [`crate::aspx_tns::compute_alphas`] from the
//! [`crate::aspx_tns::compute_covariance`] matrix of the low band. A
//! near-zero predictor means the band is noise-like (mode `None`); a
//! strong predictor (large coefficient energy) means the band is tonal
//! (heavier mode). Mapping per low subband is aggregated into each noise
//! subband group by mirroring the Pseudocode-89 high-band walk that maps
//! each high subband `sb_high` to its source low subband `p` and noise
//! group `g`.
//!
//! This module computes only the *encoder's signalling decision*; the
//! decoder-side chirp / LPC math lives in [`crate::aspx_tns`] and is the
//! ground truth this selection is calibrated against.

use crate::aspx::{derive_patch_tables, AspxFrequencyTables, AspxMasterFreqScale, AspxPatchTables};
use crate::aspx_tns::{compute_alphas, compute_covariance, Alphas};

/// Predictor-coefficient-energy thresholds that partition the
/// `|alpha0|^2 + |alpha1|^2` tonality measure into the four
/// `aspx_tna_mode` modes.
///
/// The `alpha` coefficients are bounded by the spec's `|alpha| < 4`
/// clamp (Pseudocode 87 prose: *"If either of the magnitudes of alpha0
/// and alpha1 is greater than or equal to 4, both coefficients are set to
/// 0"*), so the combined energy `|alpha0|^2 + |alpha1|^2` lies in
/// `[0, 32)`. We carve that range into ascending bands:
///
/// * `energy < LIGHT`            → mode 0 (None)     — noise-like band
/// * `LIGHT  <= energy < MOD`    → mode 1 (Light)
/// * `MOD    <= energy < HEAVY`  → mode 2 (Moderate)
/// * `energy >= HEAVY`           → mode 3 (Heavy)    — strongly tonal
///
/// The thresholds are encoder-tuning constants (the choice does not
/// affect bitstream validity), picked so a flat/white low band stays at
/// `None`, a mildly resonant band lands at `Light`, and a near-pure tone
/// (predictor coefficient magnitudes approaching the clamp) reaches
/// `Heavy`.
pub const TNA_LIGHT_THRESHOLD: f64 = 0.25;
/// See [`TNA_LIGHT_THRESHOLD`].
pub const TNA_MODERATE_THRESHOLD: f64 = 1.0;
/// See [`TNA_LIGHT_THRESHOLD`].
pub const TNA_HEAVY_THRESHOLD: f64 = 4.0;

/// Map a single low-subband predictor-coefficient energy to an
/// `aspx_tna_mode` in `0..=3` using the [`TNA_LIGHT_THRESHOLD`] /
/// [`TNA_MODERATE_THRESHOLD`] / [`TNA_HEAVY_THRESHOLD`] partition.
#[inline]
pub fn energy_to_mode(energy: f64) -> u8 {
    if energy >= TNA_HEAVY_THRESHOLD {
        3
    } else if energy >= TNA_MODERATE_THRESHOLD {
        2
    } else if energy >= TNA_LIGHT_THRESHOLD {
        1
    } else {
        0
    }
}

/// Per-low-subband predictor-coefficient energy `|alpha0|^2 +
/// |alpha1|^2`, the tonality proxy that drives mode selection.
///
/// `alphas.0` / `.1` are the complex `alpha0[sb]` / `alpha1[sb]`
/// coefficients from [`compute_alphas`]; the returned vector has the same
/// length (one entry per low subband).
pub fn predictor_energy(alphas: &Alphas) -> Vec<f64> {
    let (a0, a1) = alphas;
    let n = a0.len().max(a1.len());
    let mut out = vec![0.0_f64; n];
    for (sb, slot) in out.iter_mut().enumerate() {
        let e0 = a0
            .get(sb)
            .map(|&(re, im)| re as f64 * re as f64 + im as f64 * im as f64)
            .unwrap_or(0.0);
        let e1 = a1
            .get(sb)
            .map(|&(re, im)| re as f64 * re as f64 + im as f64 * im as f64)
            .unwrap_or(0.0);
        *slot = e0 + e1;
    }
    out
}

/// Aggregate per-low-subband tonality energies into per-noise-subband-
/// group energies by mirroring the Pseudocode-89 high-band generation
/// walk.
///
/// For each high subband `sb_high` from `sbx` upward, the walk recovers
/// the source low subband `p = sbg_patch_start_sb[i] + sb` and the active
/// noise group `g` (advanced whenever `sbg_noise[g+1] == sb_high`). The
/// low-subband energy `low_energy[p]` is accumulated into group `g`, then
/// averaged over the number of contributing high subbands so each group's
/// value is comparable regardless of how many subbands it spans.
///
/// Returns `num_sbg_noise` averaged energies (groups that receive no
/// contribution — e.g. when the patch tables are degenerate — are `0.0`,
/// which maps to mode `None`).
pub fn aggregate_noise_group_energy(
    low_energy: &[f64],
    patches: &AspxPatchTables,
    sbg_noise: &[u32],
    sbx: u32,
) -> Vec<f64> {
    let num_sbg_noise = sbg_noise.len().saturating_sub(1);
    let mut sums = vec![0.0_f64; num_sbg_noise];
    let mut counts = vec![0u32; num_sbg_noise];
    if num_sbg_noise == 0 {
        return sums;
    }

    // Pseudocode-89 high-band walk: iterate patches, then subbands within
    // each patch, tracking the running high subband and noise group.
    let mut sum_sb_patches: u32 = 0;
    let mut g: usize = 0;
    for i in 0..patches.num_sbg_patches as usize {
        let patch_num_sb = patches.sbg_patch_num_sb.get(i).copied().unwrap_or(0);
        let patch_start = patches.sbg_patch_start_sb.get(i).copied().unwrap_or(0);
        for sb in 0..patch_num_sb {
            let sb_high = sbx + sum_sb_patches + sb;
            // Advance the noise group when we cross its upper border.
            while g + 1 < sbg_noise.len() && sbg_noise[g + 1] == sb_high {
                g += 1;
            }
            if g >= num_sbg_noise {
                break;
            }
            let p = (patch_start + sb) as usize;
            let e = low_energy.get(p).copied().unwrap_or(0.0);
            sums[g] += e;
            counts[g] += 1;
        }
        sum_sb_patches += patch_num_sb;
    }

    for (s, c) in sums.iter_mut().zip(counts.iter()) {
        if *c > 0 {
            *s /= *c as f64;
        }
    }
    sums
}

/// Select the per-noise-subband-group `aspx_tna_mode` vector for one
/// A-SPX carrier from its extended low-band QMF samples.
///
/// `q_low_ext[sb][ts]` is the per-low-subband extended QMF matrix
/// (`(re, im)` pairs) as produced by [`crate::aspx_tns::build_q_low_ext`]
/// — i.e. with the `ts_offset_hfadj` look-back prepended. `tables` and
/// `patches` are the derived A-SPX frequency / patch tables for the
/// active `aspx_config`.
///
/// Returns a `num_sbg_noise`-length vector of modes in `0..=3` suitable
/// for the `tna_mode` field of
/// [`crate::encoder_acpl3::AspxHfgenIwc1ChPayload`] /
/// [`crate::encoder_acpl3::AspxHfgenIwc2ChPayload`].
pub fn select_tna_mode_from_qmf(
    q_low_ext: &[Vec<(f32, f32)>],
    tables: &AspxFrequencyTables,
    patches: &AspxPatchTables,
) -> Vec<u8> {
    let sba = tables.sba;
    let cov = compute_covariance(q_low_ext, sba);
    let alphas = compute_alphas(&cov);
    let low_energy = predictor_energy(&alphas);
    let group_energy =
        aggregate_noise_group_energy(&low_energy, patches, &tables.sbg_noise, tables.sbx);
    group_energy.iter().map(|&e| energy_to_mode(e)).collect()
}

/// Convenience wrapper around [`select_tna_mode_from_qmf`] that derives
/// the [`AspxPatchTables`] internally from the A-SPX frequency tables.
///
/// `base_samp_freq_is_48` is `true` for the 48 kHz sample-rate family
/// (the only family currently wired in the TOC-driven pipeline, matching
/// the decoder's [`crate::decoder`] call site). The high-/low-resolution
/// master-frequency-scale flag is taken from `master_freq_scale`.
pub fn select_tna_mode(
    q_low_ext: &[Vec<(f32, f32)>],
    tables: &AspxFrequencyTables,
    master_freq_scale: AspxMasterFreqScale,
    base_samp_freq_is_48: bool,
) -> Vec<u8> {
    let is_highres = matches!(master_freq_scale, AspxMasterFreqScale::HighRes);
    let patches = derive_patch_tables(
        &tables.sbg_master,
        tables.num_sbg_master,
        tables.sba,
        tables.sbx,
        tables.num_sb_aspx,
        base_samp_freq_is_48,
        is_highres,
    );
    if patches.num_sbg_patches == 0 {
        // Degenerate patch table -> no inverse filtering signalled.
        return vec![0u8; tables.sbg_noise.len().saturating_sub(1)];
    }
    select_tna_mode_from_qmf(q_low_ext, tables, &patches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aspx_tns::Alphas;

    #[test]
    fn energy_thresholds_partition_modes() {
        assert_eq!(energy_to_mode(0.0), 0);
        assert_eq!(energy_to_mode(0.1), 0);
        assert_eq!(energy_to_mode(TNA_LIGHT_THRESHOLD - 1e-9), 0);
        assert_eq!(energy_to_mode(TNA_LIGHT_THRESHOLD), 1);
        assert_eq!(energy_to_mode(0.5), 1);
        assert_eq!(energy_to_mode(TNA_MODERATE_THRESHOLD - 1e-9), 1);
        assert_eq!(energy_to_mode(TNA_MODERATE_THRESHOLD), 2);
        assert_eq!(energy_to_mode(2.0), 2);
        assert_eq!(energy_to_mode(TNA_HEAVY_THRESHOLD - 1e-9), 2);
        assert_eq!(energy_to_mode(TNA_HEAVY_THRESHOLD), 3);
        assert_eq!(energy_to_mode(10.0), 3);
    }

    #[test]
    fn energy_to_mode_is_monotone() {
        // Mode is non-decreasing in energy.
        let mut prev = 0u8;
        let mut e = 0.0;
        while e < 8.0 {
            let m = energy_to_mode(e);
            assert!(m >= prev, "mode decreased at energy {e}");
            prev = m;
            e += 0.05;
        }
    }

    #[test]
    fn predictor_energy_sums_squared_magnitudes() {
        let a0 = vec![(1.0_f32, 0.0), (0.0, 2.0), (0.0, 0.0)];
        let a1 = vec![(0.0_f32, 1.0), (3.0, 0.0), (0.0, 0.0)];
        let alphas: Alphas = (a0, a1);
        let e = predictor_energy(&alphas);
        assert_eq!(e.len(), 3);
        // sb0: |1+0j|^2 + |0+1j|^2 = 1 + 1 = 2.
        assert!((e[0] - 2.0).abs() < 1e-12);
        // sb1: |0+2j|^2 + |3+0j|^2 = 4 + 9 = 13.
        assert!((e[1] - 13.0).abs() < 1e-12);
        // sb2: all-zero predictor -> energy 0 (noise-like).
        assert_eq!(e[2], 0.0);
    }

    #[test]
    fn aggregate_walks_patches_into_noise_groups() {
        // Two patches, each two subbands. Low subbands 0..4 with energies
        // chosen so the two noise groups separate cleanly.
        let low_energy = vec![0.0, 0.0, 5.0, 5.0];
        // sbx = 4. Patch 0: start_sb=0, num_sb=2 -> high 4,5 source low 0,1.
        // Patch 1: start_sb=2, num_sb=2 -> high 6,7 source low 2,3.
        let patches = AspxPatchTables {
            sbg_patches: vec![4, 6, 8],
            num_sbg_patches: 2,
            sbg_patch_num_sb: vec![2, 2],
            sbg_patch_start_sb: vec![0, 2],
        };
        // Noise borders: group 0 covers high [4,6), group 1 covers [6,8).
        let sbg_noise = vec![4, 6, 8];
        let agg = aggregate_noise_group_energy(&low_energy, &patches, &sbg_noise, 4);
        assert_eq!(agg.len(), 2);
        // Group 0 averages low[0],low[1] = (0+0)/2 = 0.
        assert!((agg[0] - 0.0).abs() < 1e-12);
        // Group 1 averages low[2],low[3] = (5+5)/2 = 5.
        assert!((agg[1] - 5.0).abs() < 1e-12);
        // -> modes None, Heavy.
        let modes: Vec<u8> = agg.iter().map(|&e| energy_to_mode(e)).collect();
        assert_eq!(modes, vec![0, 3]);
    }

    #[test]
    fn empty_noise_groups_yield_no_modes() {
        let patches = AspxPatchTables {
            sbg_patches: vec![4],
            num_sbg_patches: 0,
            sbg_patch_num_sb: vec![],
            sbg_patch_start_sb: vec![],
        };
        let agg = aggregate_noise_group_energy(&[1.0, 2.0], &patches, &[4], 4);
        assert!(agg.is_empty());
    }

    #[test]
    fn tonal_low_band_selects_heavier_mode_than_noise() {
        use crate::aspx::{
            derive_aspx_frequency_tables, AspxConfig, AspxFreqResMode, AspxMasterFreqScale,
            AspxQuantStep,
        };
        use crate::aspx_tns::build_q_low_ext;

        // A representative 48 kHz high-res aspx_config.
        let cfg = AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::High,
        };
        let tables = derive_aspx_frequency_tables(&cfg, 0).expect("frequency tables");
        let sba = tables.sba as usize;
        let n_ts = 32usize;

        // Build a strongly tonal low band: each subband carries a pure
        // rotating phasor (perfectly predictable -> large alphas).
        let tonal: Vec<Vec<(f32, f32)>> = (0..sba)
            .map(|sb| {
                (0..n_ts)
                    .map(|ts| {
                        let theta = 0.3 * (sb as f32 + 1.0) * ts as f32;
                        (theta.cos(), theta.sin())
                    })
                    .collect()
            })
            .collect();
        // Build a noise-like low band: pseudo-random unpredictable samples.
        let mut state: u32 = 0x1234_5678;
        let mut rng = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let noisy: Vec<Vec<(f32, f32)>> = (0..sba)
            .map(|_| (0..n_ts).map(|_| (rng(), rng())).collect())
            .collect();

        let tonal_ext = build_q_low_ext(&tonal, &[], tables.sba);
        let noisy_ext = build_q_low_ext(&noisy, &[], tables.sba);

        let modes_tonal = select_tna_mode(&tonal_ext, &tables, AspxMasterFreqScale::HighRes, true);
        let modes_noisy = select_tna_mode(&noisy_ext, &tables, AspxMasterFreqScale::HighRes, true);

        assert_eq!(modes_tonal.len(), tables.counts.num_sbg_noise as usize);
        assert_eq!(modes_noisy.len(), tables.counts.num_sbg_noise as usize);
        // Every mode is a valid 2-bit value.
        assert!(modes_tonal.iter().all(|&m| m <= 3));
        assert!(modes_noisy.iter().all(|&m| m <= 3));
        // The tonal band must signal at least as much inverse filtering as
        // the noise band overall, and strictly more on the sum (a tonal
        // carrier is what TNS is designed to suppress).
        let sum_tonal: u32 = modes_tonal.iter().map(|&m| m as u32).sum();
        let sum_noisy: u32 = modes_noisy.iter().map(|&m| m as u32).sum();
        assert!(
            sum_tonal > sum_noisy,
            "tonal sum {sum_tonal} should exceed noisy sum {sum_noisy}"
        );
        // The noise band should be mostly None.
        assert!(sum_noisy <= modes_noisy.len() as u32);
    }

    #[test]
    fn group_with_no_contribution_is_zero_mode() {
        // A noise group whose high subbands never appear in the patch walk
        // (degenerate patch table) gets energy 0 -> mode None.
        let patches = AspxPatchTables {
            sbg_patches: vec![4, 6],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![2],
            sbg_patch_start_sb: vec![0],
        };
        // Two noise groups but the patch only fills the first.
        let sbg_noise = vec![4, 6, 8];
        let agg = aggregate_noise_group_energy(&[9.0, 9.0], &patches, &sbg_noise, 4);
        assert_eq!(agg.len(), 2);
        assert!(agg[0] > 0.0);
        assert_eq!(agg[1], 0.0);
        assert_eq!(energy_to_mode(agg[1]), 0);
    }
}
