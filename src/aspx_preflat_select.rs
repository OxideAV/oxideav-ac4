//! Encoder-side `aspx_preflat` selection (A-SPX spectral pre-flattening).
//!
//! ETSI TS 103 190-1 Table 121 defines the **decode** meaning of the
//! single-bit `aspx_preflat` element (carried once per `aspx_config()`,
//! Table 50):
//!
//! | `aspx_preflat` | Meaning                  |
//! | -------------- | ------------------------ |
//! | 0              | Pre-flattening not used  |
//! | 1              | Pre-flattening used      |
//!
//! and §5.7.6.4.1.1–§5.7.6.4.1.2 (Pseudocode 85,
//! [`crate::aspx_tns::compute_preflat_gains`]) define how the decoder uses
//! it. Before the subband tonal-to-noise-ratio adjustment, the decoder
//! fits a third-order polynomial to the dB spectral envelope of the source
//! low band `Q_low` used for HF generation, turning the overall spectral
//! **slope** of that source range into a per-low-subband gain vector
//! `gain_vec[sb] = 10^((mean_energy − slope[sb]) / 20)`. When
//! `aspx_preflat == 1` the **inverse** of that gain is multiplied into the
//! patched high band (`Q_high[sb_high][ts] *= 1 / gain_vec[p]`,
//! Pseudocode 89), flattening the overall tilt the transposition would
//! otherwise carry up from the source range into the regenerated band.
//!
//! The **selection** of `aspx_preflat` is an encoder decision the spec
//! leaves to the implementer: the field is informative for bitstream
//! validity (the decoder applies the matching gain whenever the flag is
//! set), so either value produces a decodable bitstream. The per-subband-
//! group SIGNAL envelope is restored *after* pre-flattening
//! ([`crate::aspx::AspxEnvelopeAdjuster`] runs on the pre-flattened tile),
//! so enabling the flag re-shapes the spectrum **within** each subband
//! group while the group's total energy stays pinned to the signalled
//! envelope — i.e. the decision changes the decoded PCM (the intra-group
//! spectral shape) without disturbing the envelope-level fidelity.
//!
//! ## Clean-room encoder analysis
//!
//! Pre-flattening is only worth signalling when the HF-generation source
//! range — the low band `Q_low` — carries a **strong overall spectral
//! tilt**. A source range that is already spectrally flat fits a flat
//! polynomial, so `slope[sb]` is near-constant, every `gain_vec[sb]` is
//! near `1`, and `1 / gain_vec` leaves the tile essentially unchanged:
//! flattening a flat band is a no-op. A steeply sloped source range fits a
//! tilted polynomial, so the gains span a wide range and the inverse gain
//! materially de-tilts the transposed tile.
//!
//! We therefore reuse the decoder's exact Pseudocode-85 gain vector as the
//! ground truth and measure its **dynamic range in dB**:
//!
//! ```text
//!   spread_db = 20 · log10( max(gain_vec) / min(gain_vec) )
//! ```
//!
//! Because `gain_vec[sb] = 10^((mean_energy − slope[sb]) / 20)`, this
//! spread is exactly the peak-to-peak excursion of the fitted slope in dB
//! (`max(slope) − min(slope)`), independent of `mean_energy` — i.e. it is
//! **level-independent at audible levels** (a quiet and a loud copy of the
//! same signal fit the same shape and select the same flag) and zero for a
//! perfectly flat source range. (Pseudocode 85's `10·log10(pow_env + 1)`
//! regularizer makes the equality asymptotic in level: at sub-unit QMF
//! amplitudes the `+1` floor intentionally suppresses the fitted slope, so
//! near-silent bands are flattened less — a property of the decoder math
//! this selection faithfully tracks.) When the spread reaches
//! [`PREFLAT_SLOPE_THRESHOLD_DB`] the
//! source range tilts enough that flattening the tile is worthwhile and we
//! set `aspx_preflat = 1`.
//!
//! This module computes only the encoder's signalling decision; the
//! decoder-side polynomial fit / gain application lives in
//! [`crate::aspx_tns::compute_preflat_gains`] / [`crate::aspx_tns::hf_tile_tns`]
//! and is the ground truth this selection serves.
//!
//! Refs: ETSI TS 103 190-1 Table 121 (`aspx_preflat`), Table 50
//! (`aspx_config`), §5.7.6.4.1.1–§5.7.6.4.1.2 Pseudocode 85
//! (pre-flattening control data), §5.7.6.4.1.4 Pseudocode 89 (gain
//! application).

use crate::aspx_tns::compute_preflat_gains;

/// Peak-to-peak slope excursion (in dB) of the source-low-band spectral
/// envelope at which the encoder begins signalling `aspx_preflat = 1`.
///
/// The measure is `20·log10(max(gain_vec) / min(gain_vec))`, which equals
/// the fitted slope's max-minus-min in dB. A perfectly flat source range
/// yields `0`; a source range whose fitted overall tilt spans ≥ this many
/// dB across the low band carries enough tilt that de-tilting the
/// transposed HF tile is worthwhile.
///
/// `12 dB` requires a clearly sloped source range — roughly a factor-of-4
/// peak-to-trough excursion in the fitted slope — before flattening. This
/// is an encoder-tuning constant; it does not affect bitstream validity
/// (the decoder applies the matching gain wherever the flag is set).
pub const PREFLAT_SLOPE_THRESHOLD_DB: f64 = 12.0;

/// Peak-to-peak dynamic range, in dB, of a pre-flattening gain vector.
///
/// Returns `20·log10(max / min)` over the strictly-positive gains. Gains
/// that are non-finite or `≤ 0` are skipped (the decoder treats a zero
/// gain as "no divide", so it carries no tilt information). Returns `0.0`
/// when fewer than two usable gains remain (a single subband carries no
/// resolvable slope).
pub fn gain_spread_db(gain_vec: &[f32]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    let mut count = 0_usize;
    for &g in gain_vec {
        let g = g as f64;
        if g.is_finite() && g > 0.0 {
            if g < lo {
                lo = g;
            }
            if g > hi {
                hi = g;
            }
            count += 1;
        }
    }
    if count < 2 || lo <= 0.0 || !hi.is_finite() {
        return 0.0;
    }
    20.0 * (hi / lo).log10()
}

/// Decide `aspx_preflat` from a pre-flattening gain vector.
///
/// Returns `true` (signal pre-flattening) iff the gain spread
/// ([`gain_spread_db`]) reaches [`PREFLAT_SLOPE_THRESHOLD_DB`] — i.e. the
/// source low band carries a strong enough overall spectral tilt that
/// de-tilting the transposed HF tile is worthwhile.
pub fn select_preflat_from_gains(gain_vec: &[f32]) -> bool {
    gain_spread_db(gain_vec) >= PREFLAT_SLOPE_THRESHOLD_DB
}

/// Select `aspx_preflat` for one A-SPX carrier straight from its QMF low
/// band.
///
/// `q_low[sb][ts]` is the carrier's complex QMF low-band matrix (subbands
/// `0..sbx`, the same source range the decoder's HF generator transposes).
/// `sbx` is the number of low-band subbands. `atsg_sig` are the SIGNAL
/// A-SPX time-slot-group borders and `num_ts_in_ats` the QMF slots per
/// A-SPX slot — both feeding the decoder's exact Pseudocode-85 envelope
/// window. The function reuses
/// [`crate::aspx_tns::compute_preflat_gains`] (the decoder ground truth) to
/// obtain the gain vector, then thresholds its dB spread.
///
/// Returns `false` when the source range is too short / flat to fit a
/// slope (`compute_preflat_gains` then yields all-`1` gains and a `0 dB`
/// spread).
pub fn select_preflat(
    q_low: &[Vec<(f32, f32)>],
    sbx: u32,
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
) -> bool {
    let gains = compute_preflat_gains(q_low, sbx, atsg_sig, num_ts_in_ats);
    select_preflat_from_gains(&gains)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_gains_have_zero_spread_no_preflat() {
        let g = vec![1.0_f32; 8];
        assert!((gain_spread_db(&g)).abs() < 1e-12);
        assert!(!select_preflat_from_gains(&g));
    }

    #[test]
    fn spread_is_twenty_log10_ratio() {
        // max/min = 10 → 20·log10(10) = 20 dB.
        let g = vec![1.0_f32, 10.0, 3.0, 0.5_f32 * 2.0];
        // min=1.0, max=10.0 → 20 dB.
        assert!((gain_spread_db(&g) - 20.0).abs() < 1e-9);
        assert!(select_preflat_from_gains(&g));
    }

    #[test]
    fn spread_just_below_threshold_no_preflat() {
        // max/min = 10^(11/20) ≈ 3.548 → 11 dB < 12 dB threshold.
        let ratio = 10.0_f64.powf(11.0 / 20.0) as f32;
        let g = vec![1.0_f32, ratio];
        assert!((gain_spread_db(&g) - 11.0).abs() < 1e-4);
        assert!(!select_preflat_from_gains(&g));
    }

    #[test]
    fn spread_at_threshold_requests_preflat() {
        let ratio = 10.0_f64.powf(PREFLAT_SLOPE_THRESHOLD_DB / 20.0) as f32;
        let g = vec![1.0_f32, ratio];
        assert!(select_preflat_from_gains(&g));
    }

    #[test]
    fn nonpositive_and_nonfinite_gains_skipped() {
        // Zeros / NaN / inf are ignored; the two valid gains (1, 10) give
        // 20 dB.
        let g = vec![0.0_f32, f32::NAN, 1.0, f32::INFINITY, 10.0, -3.0];
        assert!((gain_spread_db(&g) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn single_usable_gain_has_zero_spread() {
        let g = vec![0.0_f32, 5.0, f32::NAN];
        assert!((gain_spread_db(&g)).abs() < 1e-12);
        assert!(!select_preflat_from_gains(&g));
    }

    #[test]
    fn level_independence_above_the_regularizer_floor() {
        // The spec's Pseudocode-85 dB envelope carries a `+1` regularizer
        // (`10·log10(pow_env + 1)`) that keeps the log finite for silent
        // subbands. Well above that floor (i.e. for audible levels), scaling
        // every QMF sample by a constant only shifts the whole dB envelope
        // by a constant, leaving the fitted slope shape — hence the gain
        // spread and the flag — unchanged. Build a sloped low band at an
        // audible base level and a 10x-louder copy; both must select the
        // same flag and report a near-identical spread. (At sub-unit
        // amplitudes the `+1` floor intentionally suppresses the slope, so
        // the equality is asymptotic in level, matching the decoder.)
        let sbx = 8u32;
        let num_ts_in_ats = 2u32;
        let atsg_sig = [0u32, 4];
        let ts = (atsg_sig[1] * num_ts_in_ats) as usize;
        let mut base = vec![Vec::new(); sbx as usize];
        for (sb, row) in base.iter_mut().enumerate() {
            // Geometric tilt across subbands at an audible base level:
            // amplitude 1000 · 0.75^sb (top subband still ≈ 100 ≫ √1).
            let amp = 1000.0_f32 * 0.75_f32.powi(sb as i32);
            *row = vec![(amp, 0.0); ts];
        }
        let loud: Vec<Vec<(f32, f32)>> = base
            .iter()
            .map(|r| r.iter().map(|&(re, im)| (re * 10.0, im * 10.0)).collect())
            .collect();
        let db = gain_spread_db(&compute_preflat_gains(&base, sbx, &atsg_sig, num_ts_in_ats));
        let dl = gain_spread_db(&compute_preflat_gains(&loud, sbx, &atsg_sig, num_ts_in_ats));
        assert!(
            (db - dl).abs() < 0.5,
            "level dependence above floor: base {db} dB vs loud {dl} dB"
        );
        assert_eq!(
            select_preflat_from_gains(&compute_preflat_gains(&base, sbx, &atsg_sig, num_ts_in_ats)),
            select_preflat_from_gains(&compute_preflat_gains(&loud, sbx, &atsg_sig, num_ts_in_ats)),
        );
    }

    #[test]
    fn steep_tilt_selects_preflat_flat_band_does_not() {
        let sbx = 8u32;
        let num_ts_in_ats = 2u32;
        let atsg_sig = [0u32, 4];
        let ts = (atsg_sig[1] * num_ts_in_ats) as usize;
        // Steeply tilted source range at an audible level (each subband
        // 2x quieter, top subband still ≫ the +1 regularizer floor).
        let steep: Vec<Vec<(f32, f32)>> = (0..sbx as usize)
            .map(|sb| {
                let amp = 4000.0_f32 * 0.5_f32.powi(sb as i32);
                vec![(amp, 0.0); ts]
            })
            .collect();
        assert!(select_preflat(&steep, sbx, &atsg_sig, num_ts_in_ats));
        // Flat source range → no preflat.
        let flat: Vec<Vec<(f32, f32)>> =
            (0..sbx as usize).map(|_| vec![(1000.0, 0.0); ts]).collect();
        assert!(!select_preflat(&flat, sbx, &atsg_sig, num_ts_in_ats));
    }
}
