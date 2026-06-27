//! Encoder-side `aspx_add_harmonic` selection (A-SPX missing-harmonic /
//! sinusoid restoration).
//!
//! ETSI TS 103 190-1 §4.2.12.6 (`aspx_hfgen_iwc_1ch` / `_2ch`) carries an
//! optional per-high-resolution-signal-subband-group flag vector
//! `aspx_add_harmonic[sbg]`. §5.7.6.4.2.1 Pseudocode 92
//! ([`crate::aspx::derive_sine_idx_sb`]) turns each set flag into a
//! sinusoid placed at the **middle subband** of that signal subband group
//! (`sb_mid = (sba + sbz) / 2`, integer-truncated); §5.7.6.4.4
//! Pseudocodes 104/105 ([`crate::aspx_tone`]) then injects a unit-phase
//! complex sinusoid into the regenerated HF QMF matrix at that subband,
//! level-matched by the signal envelope (`sine_lev_sb`).
//!
//! The A-SPX HF generator transposes the **low** band up into the high
//! band; that transposition reproduces the noise-like spectral envelope
//! but does **not** reproduce a discrete tonal partial that exists in the
//! original HF band yet has no counterpart at the corresponding low-band
//! transposition source. `aspx_add_harmonic` is the encoder's signal that
//! such a missing harmonic should be re-synthesised by the tone
//! generator. The decision is, per §4.2.12.6, an **encoder analysis
//! choice** — the field is informative for bitstream validity (the
//! decoder restores a sinusoid wherever the flag is set and the SineTable
//! phase walk dictates), so any vector in `{false, true}^num_sbg` decodes.
//!
//! ## Clean-room encoder analysis
//!
//! A discrete sinusoid in the QMF domain concentrates its energy in a
//! single subband, so within a high-res signal subband group it shows a
//! high **spectral crest** — the ratio of the peak (here the decoder's
//! placement subband `sb_mid`) per-subband energy to the group's mean
//! per-subband energy. A noise-like (flat) group has a crest near 1; a
//! group dominated by one tonal partial has a crest equal to the number
//! of subbands in the group (all energy in one bin). We therefore set
//! `aspx_add_harmonic[sbg] = true` when:
//!
//! * the group spans at least two subbands (a single-subband group has a
//!   trivial crest of 1 and carries no resolvable "missing harmonic"
//!   distinct from its own envelope), **and**
//! * the per-subband energy at `sb_mid` exceeds
//!   [`AH_CREST_THRESHOLD`] × the group's mean per-subband energy, **and**
//! * the group carries non-trivial energy (above [`AH_ENERGY_FLOOR`]
//!   relative to the whole A-SPX band peak), so silence / numerical-noise
//!   groups never spuriously request a tone.
//!
//! The crest measure is **level-independent** (it is a ratio of energies
//! within the same group), matching the level-independence of the decoder
//! sinusoid (whose absolute level comes from the signal envelope, not from
//! `aspx_add_harmonic`). Like [`crate::aspx_tna_select`], this module
//! computes only the encoder's signalling decision; the decoder-side
//! sinusoid math lives in [`crate::aspx::derive_sine_idx_sb`] +
//! [`crate::aspx_tone`] and is the ground truth this selection serves.
//!
//! Refs: ETSI TS 103 190-1 §4.2.12.6 (`aspx_hfgen_iwc`), §5.7.6.4.2.1
//! Pseudocode 92 (`sine_idx_sb` / `sb_mid`), §5.7.6.4.4 Pseudocodes
//! 104/105 (tone generator).

/// Crest-factor threshold partitioning a tonal group (request a harmonic)
/// from a noise-like group (no harmonic). A group's crest is
/// `energy[sb_mid] / mean(energy over group)`; a perfectly flat group has
/// crest 1, a single-bin tone has crest = group width. The threshold of
/// `2.0` requires the placement subband to carry at least twice the
/// group's average per-subband energy — i.e. a clearly dominant partial —
/// before signalling a missing harmonic.
///
/// This is an encoder-tuning constant; it does not affect bitstream
/// validity (the decoder restores a sinusoid wherever the flag is set).
pub const AH_CREST_THRESHOLD: f64 = 2.0;

/// Relative energy floor below which a signal subband group is treated as
/// silent and never requests a harmonic, expressed as a fraction of the
/// loudest group's mean per-subband energy across the A-SPX band. Guards
/// against a numerically "peaky" but inaudible group (e.g. dither in an
/// otherwise empty band) spuriously setting `aspx_add_harmonic`.
pub const AH_ENERGY_FLOOR: f64 = 1.0e-4;

/// Per-high-res-signal-subband-group total energy + middle-subband energy,
/// the two quantities the crest test consumes.
#[derive(Debug, Clone, Copy, Default)]
pub struct SbgTonality {
    /// Sum of per-subband energy across the group (Σ over `[sba, sbz)`).
    pub total_energy: f64,
    /// Per-subband energy at the decoder's placement subband
    /// `sb_mid = (sba + sbz) / 2`.
    pub mid_energy: f64,
    /// Number of subbands in the group (`sbz - sba`, clamped to the A-SPX
    /// range).
    pub num_sb: u32,
}

impl SbgTonality {
    /// Mean per-subband energy across the group (`total / num_sb`), or
    /// `0.0` for an empty group.
    #[inline]
    pub fn mean_energy(&self) -> f64 {
        if self.num_sb == 0 {
            0.0
        } else {
            self.total_energy / self.num_sb as f64
        }
    }

    /// Spectral crest `mid_energy / mean_energy`. A flat group → ~1; a
    /// single-bin tone at `sb_mid` → `num_sb`. `0.0` for an empty / silent
    /// group.
    #[inline]
    pub fn crest(&self) -> f64 {
        let mean = self.mean_energy();
        if mean <= 0.0 {
            0.0
        } else {
            self.mid_energy / mean
        }
    }
}

/// Per-subband energy (summed over all time slots in the frame) of the HF
/// QMF matrix `q_high[absolute_sb][ts]`, indexed by absolute subband.
///
/// Returned vector length equals `q_high.len()`. Subbands the matrix does
/// not cover contribute `0.0`.
pub fn per_subband_energy(q_high: &[Vec<(f32, f32)>]) -> Vec<f64> {
    q_high
        .iter()
        .map(|row| {
            row.iter()
                .map(|&(re, im)| re as f64 * re as f64 + im as f64 * im as f64)
                .sum::<f64>()
        })
        .collect()
}

/// Reduce a per-absolute-subband energy vector to per-high-res-signal-
/// subband-group [`SbgTonality`] entries over the `sbg_sig_highres`
/// borders.
///
/// `sbg_sig_highres` is the high-resolution signal subband-group border
/// list (`crate::aspx::AspxFrequencyTables::sbg_sig_highres`), absolute
/// subbands, length `num_sbg_sig_highres + 1`. `sbx` is the A-SPX
/// cross-over subband; subbands below `sbx` are clamped out (they are not
/// part of the regenerated band).
///
/// Returns `num_sbg_sig_highres` entries. The placement subband per group
/// is `sb_mid = (sba + sbz) / 2` (integer-truncated, matching
/// [`crate::aspx::derive_sine_idx_sb`]).
pub fn sbg_tonalities(sb_energy: &[f64], sbg_sig_highres: &[u32], sbx: u32) -> Vec<SbgTonality> {
    let num_sbg = sbg_sig_highres.len().saturating_sub(1);
    let mut out = Vec::with_capacity(num_sbg);
    for sbg in 0..num_sbg {
        let sba = sbg_sig_highres[sbg].max(sbx) as usize;
        let sbz = sbg_sig_highres[sbg + 1].max(sbx) as usize;
        if sbz <= sba {
            out.push(SbgTonality::default());
            continue;
        }
        // sb_mid mirrors derive_sine_idx_sb: (sba + sbz) / 2 over the raw
        // (un-clamped) borders, integer-truncated.
        let sb_mid = (sbg_sig_highres[sbg] as usize + sbg_sig_highres[sbg + 1] as usize) / 2;
        let mut total = 0.0_f64;
        for sb in sba..sbz {
            total += sb_energy.get(sb).copied().unwrap_or(0.0);
        }
        let mid_energy = sb_energy.get(sb_mid).copied().unwrap_or(0.0);
        out.push(SbgTonality {
            total_energy: total,
            mid_energy,
            num_sb: (sbz - sba) as u32,
        });
    }
    out
}

/// Decide the per-high-res-signal-subband-group `aspx_add_harmonic`
/// vector from group tonalities.
///
/// A group requests a harmonic (`true`) iff it spans ≥ 2 subbands, carries
/// energy above the relative [`AH_ENERGY_FLOOR`] (vs. the loudest group's
/// mean per-subband energy), and has a spectral crest at or above
/// [`AH_CREST_THRESHOLD`]. Returns a `tonalities.len()` boolean vector.
pub fn select_add_harmonic_from_tonalities(tonalities: &[SbgTonality]) -> Vec<bool> {
    // Reference level: the loudest group's mean per-subband energy. The
    // floor is taken relative to this so a quiet but isolated tone in an
    // otherwise empty band is not flagged.
    let max_mean = tonalities
        .iter()
        .map(|t| t.mean_energy())
        .fold(0.0_f64, f64::max);
    let floor = max_mean * AH_ENERGY_FLOOR;
    tonalities
        .iter()
        .map(|t| t.num_sb >= 2 && t.mean_energy() > floor && t.crest() >= AH_CREST_THRESHOLD)
        .collect()
}

/// Select the per-high-res-signal-subband-group `aspx_add_harmonic`
/// vector for one A-SPX carrier straight from its HF QMF matrix.
///
/// `q_high[absolute_sb][ts]` is the encoder's QMF-analysis output over the
/// regenerated high band (the same matrix the real-envelope extractor
/// consumes). `sbg_sig_highres` / `sbx` come from the carrier's A-SPX
/// frequency tables.
///
/// Returns a `num_sbg_sig_highres`-length boolean vector ready for the
/// `add_harmonic` field of
/// [`crate::encoder_acpl3::AspxHfgenIwc1ChPayload`] /
/// [`crate::encoder_acpl3::AspxHfgenIwc2ChPayload`].
pub fn select_add_harmonic(
    q_high: &[Vec<(f32, f32)>],
    sbg_sig_highres: &[u32],
    sbx: u32,
) -> Vec<bool> {
    let sb_energy = per_subband_energy(q_high);
    let tonalities = sbg_tonalities(&sb_energy, sbg_sig_highres, sbx);
    select_add_harmonic_from_tonalities(&tonalities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_group_has_unit_crest_no_harmonic() {
        // Four subbands, equal energy → crest 1 → no harmonic.
        let t = SbgTonality {
            total_energy: 4.0,
            mid_energy: 1.0,
            num_sb: 4,
        };
        assert!((t.mean_energy() - 1.0).abs() < 1e-12);
        assert!((t.crest() - 1.0).abs() < 1e-12);
        assert!(!select_add_harmonic_from_tonalities(&[t])[0]);
    }

    #[test]
    fn single_bin_tone_requests_harmonic() {
        // All energy at sb_mid over a 4-subband group → crest 4 → harmonic.
        let t = SbgTonality {
            total_energy: 4.0,
            mid_energy: 4.0,
            num_sb: 4,
        };
        assert!((t.crest() - 4.0).abs() < 1e-12);
        assert!(select_add_harmonic_from_tonalities(&[t])[0]);
    }

    #[test]
    fn single_subband_group_never_requests() {
        // num_sb == 1 → never flagged even with all energy at the bin.
        let t = SbgTonality {
            total_energy: 5.0,
            mid_energy: 5.0,
            num_sb: 1,
        };
        assert!(!select_add_harmonic_from_tonalities(&[t])[0]);
    }

    #[test]
    fn silent_group_never_requests() {
        let loud = SbgTonality {
            total_energy: 1000.0,
            mid_energy: 1000.0,
            num_sb: 2,
        };
        // A second group at the per-subband floor with a "peak" should be
        // gated out by AH_ENERGY_FLOOR relative to the loud group.
        let tiny = SbgTonality {
            total_energy: 1e-9,
            mid_energy: 1e-9,
            num_sb: 2,
        };
        let sel = select_add_harmonic_from_tonalities(&[loud, tiny]);
        assert!(sel[0]);
        assert!(!sel[1]);
    }

    #[test]
    fn per_subband_energy_sums_time_slots() {
        // sb0: two slots of (1,0) → 2.0 ; sb1: (0,2) → 4.0.
        let q = vec![vec![(1.0, 0.0), (1.0, 0.0)], vec![(0.0, 2.0), (0.0, 0.0)]];
        let e = per_subband_energy(&q);
        assert!((e[0] - 2.0).abs() < 1e-9);
        assert!((e[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn sbg_tonalities_picks_mid_subband() {
        // Group [4,8): sb_mid = 6. Put a tone at sb 6.
        let mut e = vec![0.1_f64; 16];
        e[6] = 10.0;
        let sbg = [4u32, 8];
        let t = sbg_tonalities(&e, &sbg, 4);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].num_sb, 4);
        assert!((t[0].mid_energy - 10.0).abs() < 1e-9);
        // total = 10.0 + 3*0.1 = 10.3, mean = 2.575, crest ≈ 3.88 → harmonic.
        assert!(t[0].crest() > AH_CREST_THRESHOLD);
        assert!(select_add_harmonic_from_tonalities(&t)[0]);
    }

    #[test]
    fn select_add_harmonic_end_to_end() {
        // Build a q_high where the [4,8) group has a tone at sb_mid=6 and
        // the [8,12) group is flat.
        let mut q = vec![vec![(0.0_f32, 0.0_f32); 4]; 16];
        for row in q.iter_mut().skip(4) {
            for cell in row.iter_mut() {
                *cell = (0.3, 0.0);
            }
        }
        // Tone at sb 6: large magnitude.
        for cell in q[6].iter_mut() {
            *cell = (5.0, 0.0);
        }
        let sbg = [4u32, 8, 12, 16];
        let sel = select_add_harmonic(&q, &sbg, 4);
        assert_eq!(sel.len(), 3);
        assert!(sel[0], "tonal group [4,8) should request a harmonic");
        assert!(!sel[1], "flat group [8,12) should not");
        assert!(!sel[2], "flat group [12,16) should not");
    }
}
