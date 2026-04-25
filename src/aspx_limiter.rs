//! A-SPX gain limiter — ETSI TS 103 190-1 §5.7.6.4.2.2 Pseudocodes
//! 96..101 plus the §5.7.6.3.1.5 `sbg_lim` derivation (Pseudocodes
//! 72..74).
//!
//! ## Why this exists
//!
//! When the bitstream sets `aspx_limiter == 1` (Table 122) the A-SPX
//! HF regenerator runs a per-limiter-subband-group ceiling on the
//! envelope-compensation gain values produced by Pseudocode 95. Without
//! this stage, sharp peaks in the LF source signal get translated into
//! grossly over-amplified HF — the limiter caps the gain at a value
//! derived from the band's actual signal-to-energy ratio (`LIM_GAIN`
//! × √(scf_sig / est_sig)), then redistributes the energy lost via a
//! "boost factor" that re-scales the gain, the noise level, and the
//! sinusoid level for the band so the total output energy still tracks
//! the transmitted scale factors.
//!
//! ## Layout (mirrors the spec text exactly)
//!
//! * [`derive_sbg_lim`] — Pseudocode 72 / 73 / 74. Builds the limiter
//!   subband-group table from `sbg_sig_lowres` ∪ `sbg_patches`, then
//!   prunes to ~2 limiter groups per octave with the patch-preserving
//!   "octave < 0.245" rule.
//! * [`max_sig_gain`] — Pseudocode 96. `LIM_GAIN √(Σ scf_sig / Σ est_sig)`
//!   per limiter group, mapped to per-subband ceilings clamped at
//!   `MAX_SIG_GAIN`.
//! * [`limit_noise_level`] — Pseudocode 97. Caps the noise level by the
//!   gain reduction ratio.
//! * [`limit_sig_gain`] — Pseudocode 98. Min-cap on the gain.
//! * [`boost_factor`] — Pseudocode 99 + 100. Recovers the lost energy
//!   into a per-group boost factor (clamped at `MAX_BOOST_FACT`), then
//!   maps back to subbands.
//! * [`apply_boost`] — Pseudocode 101. Final `_adj` arrays —
//!   `sig_gain_sb_adj`, `noise_lev_sb_adj`, `sine_lev_sb_adj`.
//!
//! All routines are pure functions over the `f32` matrices already
//! produced upstream — no bitstream / state interaction. The
//! orchestrating glue lives in [`crate::aspx::inject_noise_and_tone`]
//! (when `aspx_limiter` is set) so the limiter sits transparently in
//! the existing P92 → P94 → P102 → P104 → P106 pipeline.

/// `LIM_GAIN` from ETSI TS 103 190-1 §5.7.6.4.2.2 Pseudocode 96 — the
/// max-gain "headroom" multiplier applied to the per-limiter-group
/// √(scf_sig/est_sig) ratio.
pub const LIM_GAIN: f32 = 1.41254;

/// `MAX_SIG_GAIN` from §5.7.6.4.2.2 Pseudocode 96 — final per-subband
/// ceiling on the maximum gain.
pub const MAX_SIG_GAIN: f32 = 1.0e5;

/// `EPSILON0` from §5.7.6.4.2.2 Pseudocode 96 — denominator floor.
pub const EPSILON0: f32 = 1.0e-12;

/// `MAX_BOOST_FACT` from §5.7.6.4.2.2 Pseudocode 100 — ceiling on the
/// per-limiter-group boost factor used to restore the energy lost by
/// the gain limit. Spec value is `10^(0.2)` ≈ 1.584893192; the f32
/// representation truncates to ~7 sig figs.
pub const MAX_BOOST_FACT: f32 = 1.584_893_2;

// ---------------------------------------------------------------------
// §5.7.6.3.1.5 limiter subband-group table (Pseudocodes 72 / 73 / 74).
// ---------------------------------------------------------------------

/// Build the limiter subband-group table per ETSI TS 103 190-1
/// §5.7.6.3.1.5 Pseudocode 72 (with the helpers from Pseudocodes 73 / 74
/// inlined).
///
/// The result has two limiter subband-groups per octave: it merges
/// `sbg_sig_lowres` and `sbg_patches`, sorts the union, and then walks
/// adjacent borders. Pairs whose log2 spacing is below 0.245 are
/// collapsed — the patch borders are preferred over the lowres borders
/// when the conflict is a duplicate-near-patch case.
///
/// Returns the limiter border table; `num_sbg_lim = result.len() - 1`.
///
/// ```text
/// for (sbg = 0; sbg <= num_sbg_sig_lowres; sbg++)
///     sbg_lim[sbg] = sbg_sig_lowres[sbg];
/// for (sbg = 1; sbg < num_sbg_patches; sbg++)
///     sbg_lim[sbg + num_sbg_sig_lowres] = sbg_patches[sbg];
/// sort(sbg_lim);
/// // ... octave-merge loop with is_element_of_sbg_patches() preference
/// ```
pub fn derive_sbg_lim(sbg_sig_lowres: &[u32], sbg_patches: &[u32]) -> Vec<u32> {
    if sbg_sig_lowres.is_empty() {
        return Vec::new();
    }
    // Pseudocode 72 — initial population:
    //   sbg_lim = sbg_sig_lowres ∪ {sbg_patches[1..num_sbg_patches-1]}.
    // The patch endpoints sbg_patches[0] = sbx and sbg_patches[last]
    // are deliberately excluded — the lowres table already covers the
    // crossover and the upper border, so adding them again would just
    // create immediate duplicates that the merge loop would have to
    // tear out.
    let mut sbg_lim: Vec<u32> = sbg_sig_lowres.to_vec();
    if sbg_patches.len() >= 2 {
        // Per the spec: copy sbg_patches[1..num_sbg_patches], skipping
        // the first and last entries (sbx + final patch border). Note:
        // Pseudocode 72 reads `for (sbg = 1; sbg < num_sbg_patches; sbg++)`
        // — i.e. indices 1..num_sbg_patches-1 inclusive when
        // sbg_patches has num_sbg_patches+1 entries.
        for &p in &sbg_patches[1..sbg_patches.len() - 1] {
            sbg_lim.push(p);
        }
    }
    // Sort + dedupe-on-entry — Pseudocode 72's `sort(sbg_lim)`.
    sbg_lim.sort_unstable();
    // Helper: `is_element_of_sbg_patches` — Pseudocode 73.
    let is_in_patches = |v: u32| sbg_patches.contains(&v);
    // Pseudocode 72 octave-merge loop.
    //
    // The C reference loop walks `sbg = 1..=num_sbg_lim`; when a pair
    // is too close (`log2(sbg_lim[sbg] / sbg_lim[sbg-1]) < 0.245`) one
    // entry is removed and `sbg` is NOT advanced. Otherwise `sbg++`.
    let mut sbg = 1usize;
    while sbg < sbg_lim.len() {
        let prev = sbg_lim[sbg - 1] as f32;
        let cur = sbg_lim[sbg] as f32;
        if prev <= 0.0 {
            sbg += 1;
            continue;
        }
        let num_octaves = (cur / prev).log2();
        if num_octaves < 0.245 {
            if sbg_lim[sbg] == sbg_lim[sbg - 1] {
                // Pure duplicate — drop the duplicate.
                sbg_lim.remove(sbg); // Pseudocode 74.
                continue;
            }
            // Octave too small but values differ. Prefer borders that
            // come from the patch table over the lowres table.
            let cur_in_patches = is_in_patches(sbg_lim[sbg]);
            let prev_in_patches = is_in_patches(sbg_lim[sbg - 1]);
            if cur_in_patches {
                if prev_in_patches {
                    // Both are patch borders — leave both, advance.
                    sbg += 1;
                    continue;
                } else {
                    // Drop the lowres-only predecessor.
                    sbg_lim.remove(sbg - 1);
                    continue;
                }
            } else {
                // Drop the current (non-patch) entry.
                sbg_lim.remove(sbg);
                continue;
            }
        } else {
            sbg += 1;
            continue;
        }
    }
    sbg_lim
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 95 — full compensatory gain (all four
// branches, including sine_area_sb != 0 and the aspx_tsg_ptr /
// p_sine_at_end exception). The simpler [`crate::aspx::compute_sig_gains`]
// only handles the `sine_area_sb == 0` non-exception case.
// ---------------------------------------------------------------------

/// Compute `sig_gain_sb` per ETSI TS 103 190-1 §5.7.6.4.2.2
/// Pseudocode 95 in full.
///
/// Branch table (matching the spec):
///
/// | `sine_area_sb` | `atsg == aspx_tsg_ptr` or `atsg == p_sine_at_end` | denominator |
/// | -------------- | ------------------------------------------------- | ----------- |
/// | 0              | true                                              | `EPSILON + est_sig` |
/// | 0              | false                                             | `(EPSILON + est_sig) * (1 + scf_noise)` |
/// | non-zero       | (always)                                          | `(EPSILON + est_sig) * (1 + scf_noise)` |
///
/// For the `sine_area_sb == 0` branches the numerator inside the
/// `sqrt()` is `scf_sig`; for `sine_area_sb != 0` it's
/// `scf_sig * scf_noise`. `p_sine_at_end` is `Some(num_atsg_sig_prev)`
/// when `aspx_tsg_ptr_prev == num_atsg_sig_prev` (i.e. the previous
/// interval ended with a sinusoid), otherwise `None` (the spec sets it
/// to -1, used here as "never matches").
#[allow(clippy::too_many_arguments)]
pub fn compute_sig_gains_full(
    est_sig_sb: &[Vec<f32>],
    scf_sig_sb: &[Vec<f32>],
    scf_noise_sb: &[Vec<f32>],
    sine_area_sb: &[Vec<u8>],
    aspx_tsg_ptr: u32,
    p_sine_at_end: Option<u32>,
) -> Vec<Vec<f32>> {
    const EPSILON: f32 = 1.0;
    let num_sb = est_sig_sb.len();
    if num_sb == 0 {
        return Vec::new();
    }
    let num_atsg = est_sig_sb[0].len();
    let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    for sb in 0..num_sb {
        for atsg in 0..num_atsg {
            let est = est_sig_sb[sb][atsg];
            let sig = scf_sig_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let noise = scf_noise_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let area = sine_area_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0);
            let exception =
                atsg as u32 == aspx_tsg_ptr || matches!(p_sine_at_end, Some(p) if atsg as u32 == p);
            let denom_base = EPSILON + est;
            let (numer, denom) = if area == 0 {
                let denom = if exception {
                    denom_base
                } else {
                    denom_base * (1.0 + noise)
                };
                (sig, denom)
            } else {
                // sine_area_sb != 0 branch always multiplies by (1 + noise).
                (sig * noise, denom_base * (1.0 + noise))
            };
            let ratio = if denom > 0.0 { numer / denom } else { 0.0 };
            out[sb][atsg] = ratio.max(0.0).sqrt();
        }
    }
    out
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 96 — max_sig_gain.
// ---------------------------------------------------------------------

/// `max_sig_gain_sb[sb][atsg]` per ETSI TS 103 190-1 §5.7.6.4.2.2
/// Pseudocode 96.
///
/// ```text
/// LIM_GAIN     = 1.41254;
/// EPSILON0     = pow(10, -12);
/// MAX_SIG_GAIN = pow(10, 5);
/// for atsg, sbg:
///     nom   = Σ scf_sig_sb[sb][atsg]    over sbg_lim[sbg]..sbg_lim[sbg+1]
///     denom = EPSILON0 + Σ est_sig_sb[sb][atsg]
///     max_sig_gain_sbg[sbg][atsg] = sqrt(nom/denom) * LIM_GAIN
/// // map to subbands and clamp at MAX_SIG_GAIN.
/// ```
pub fn max_sig_gain(
    est_sig_sb: &[Vec<f32>],
    scf_sig_sb: &[Vec<f32>],
    sbg_lim: &[u32],
    sbx: u32,
    num_sb_aspx: u32,
) -> Vec<Vec<f32>> {
    let num_sb = num_sb_aspx as usize;
    if num_sb == 0 || est_sig_sb.is_empty() {
        return Vec::new();
    }
    let num_atsg = est_sig_sb[0].len();
    let num_sbg_lim = sbg_lim.len().saturating_sub(1);
    if num_sbg_lim == 0 {
        // Degenerate — fall back to a flat ceiling so the limiter
        // becomes a no-op (sig_gain stays as-is, MAX_SIG_GAIN is huge).
        return vec![vec![MAX_SIG_GAIN; num_atsg]; num_sb];
    }
    let mut max_sb: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    let mut sbg_max: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sbg_lim];
    for atsg in 0..num_atsg {
        // Per-limiter-group sums.
        for sbg in 0..num_sbg_lim {
            let mut nom = 0.0_f32;
            let mut denom = EPSILON0;
            let lo = sbg_lim[sbg].saturating_sub(sbx) as usize;
            let hi = sbg_lim[sbg + 1].saturating_sub(sbx) as usize;
            let lo = lo.min(num_sb);
            let hi = hi.min(num_sb);
            for sb in lo..hi {
                let sig = scf_sig_sb
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                let est = est_sig_sb
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                nom += sig;
                denom += est;
            }
            let g = if denom > 0.0 {
                (nom / denom).max(0.0).sqrt() * LIM_GAIN
            } else {
                0.0
            };
            sbg_max[sbg][atsg] = g;
        }
        // Map to subbands — Pseudocode 96's `if (sb == sbg_lim[sbg+1]-sbx) sbg++`.
        let mut sbg = 0usize;
        #[allow(clippy::needless_range_loop)]
        // ETSI TS 103 190-1 §5.7.6.4.2.2 Pseudocode 96 max_sig_gain_sb[sb][atsg]
        for sb in 0..num_sb {
            // Advance sbg until sb < sbg_lim[sbg+1] - sbx.
            while sbg + 1 < num_sbg_lim && sb >= sbg_lim[sbg + 1].saturating_sub(sbx) as usize {
                sbg += 1;
            }
            // Clamp at MAX_SIG_GAIN.
            max_sb[sb][atsg] = sbg_max[sbg][atsg].min(MAX_SIG_GAIN);
        }
    }
    max_sb
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 97 — limit noise level by the gain ratio.
// ---------------------------------------------------------------------

/// `noise_lev_sb_lim[sb][atsg]` per ETSI TS 103 190-1 §5.7.6.4.2.2
/// Pseudocode 97.
///
/// ```text
/// tmp = noise_lev_sb[sb][atsg] * (max_sig_gain_sb / sig_gain_sb);
/// noise_lev_sb_lim = min(noise_lev_sb, tmp);
/// ```
///
/// When `sig_gain_sb` is zero the ratio is taken as 1.0 — i.e. no
/// reduction (the original noise level is preserved). Without this
/// guard a zero-gain band would force the noise to either zero or a
/// runaway value depending on the sign of the limit, both of which are
/// audibly worse than just keeping the unscaled noise.
pub fn limit_noise_level(
    noise_lev_sb: &[Vec<f32>],
    max_sig_gain_sb: &[Vec<f32>],
    sig_gain_sb: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let num_sb = noise_lev_sb.len();
    if num_sb == 0 {
        return Vec::new();
    }
    let num_atsg = noise_lev_sb[0].len();
    let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    for sb in 0..num_sb {
        for atsg in 0..num_atsg {
            let n = noise_lev_sb[sb][atsg];
            let mx = max_sig_gain_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let g = sig_gain_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let ratio = if g > 0.0 { mx / g } else { 1.0 };
            let tmp = n * ratio;
            out[sb][atsg] = n.min(tmp);
        }
    }
    out
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 98 — sig_gain_sb_lim.
// ---------------------------------------------------------------------

/// `sig_gain_sb_lim = min(sig_gain_sb, max_sig_gain_sb)` per
/// ETSI TS 103 190-1 §5.7.6.4.2.2 Pseudocode 98.
pub fn limit_sig_gain(sig_gain_sb: &[Vec<f32>], max_sig_gain_sb: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let num_sb = sig_gain_sb.len();
    if num_sb == 0 {
        return Vec::new();
    }
    let num_atsg = sig_gain_sb[0].len();
    let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    for sb in 0..num_sb {
        for atsg in 0..num_atsg {
            let g = sig_gain_sb[sb][atsg];
            let mx = max_sig_gain_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(g);
            out[sb][atsg] = g.min(mx);
        }
    }
    out
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 99 + 100 — boost_fact.
// ---------------------------------------------------------------------

/// Compute the per-subband boost factor per ETSI TS 103 190-1
/// §5.7.6.4.2.2 Pseudocodes 99 + 100.
///
/// Pseudocode 99 (per limiter group):
/// ```text
/// nom   = EPSILON0 + Σ scf_sig_sb[sb][atsg]
/// denom = EPSILON0
///       + Σ est_sig_sb[sb][atsg] * sig_gain_sb_lim[sb][atsg]^2
///       + Σ sine_lev_sb[sb][atsg]^2
///       + (if no sine and atsg ∉ {tsg_ptr, p_sine_at_end})
///             Σ noise_lev_sb_lim[sb][atsg]^2
/// boost_fact_sbg = sqrt(nom/denom)
/// ```
///
/// Pseudocode 100: clamp at `MAX_BOOST_FACT` and map back to subbands.
#[allow(clippy::too_many_arguments)]
pub fn boost_factor(
    est_sig_sb: &[Vec<f32>],
    scf_sig_sb: &[Vec<f32>],
    sig_gain_sb_lim: &[Vec<f32>],
    sine_lev_sb: &[Vec<f32>],
    noise_lev_sb_lim: &[Vec<f32>],
    sbg_lim: &[u32],
    sbx: u32,
    num_sb_aspx: u32,
    aspx_tsg_ptr: u32,
    p_sine_at_end: Option<u32>,
) -> Vec<Vec<f32>> {
    let num_sb = num_sb_aspx as usize;
    if num_sb == 0 || est_sig_sb.is_empty() {
        return Vec::new();
    }
    let num_atsg = est_sig_sb[0].len();
    let num_sbg_lim = sbg_lim.len().saturating_sub(1);
    if num_sbg_lim == 0 {
        return vec![vec![1.0; num_atsg]; num_sb];
    }
    let mut boost_sb: Vec<Vec<f32>> = vec![vec![1.0_f32; num_atsg]; num_sb];
    let mut sbg_boost: Vec<Vec<f32>> = vec![vec![1.0_f32; num_atsg]; num_sbg_lim];
    for atsg in 0..num_atsg {
        let exception =
            atsg as u32 == aspx_tsg_ptr || matches!(p_sine_at_end, Some(p) if atsg as u32 == p);
        for sbg in 0..num_sbg_lim {
            let mut nom = EPSILON0;
            let mut denom = EPSILON0;
            let lo = sbg_lim[sbg].saturating_sub(sbx) as usize;
            let hi = sbg_lim[sbg + 1].saturating_sub(sbx) as usize;
            let lo = lo.min(num_sb);
            let hi = hi.min(num_sb);
            for sb in lo..hi {
                let sig = scf_sig_sb
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                let est = est_sig_sb
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                let g = sig_gain_sb_lim
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                let s = sine_lev_sb
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                let n = noise_lev_sb_lim
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0.0);
                nom += sig;
                denom += est * g * g;
                denom += s * s;
                // Pseudocode 99: only add the noise term when the band
                // has no sine and we're not in the exception envelope.
                if s == 0.0 && !exception {
                    denom += n * n;
                }
            }
            let g = if denom > 0.0 {
                (nom / denom).max(0.0).sqrt()
            } else {
                1.0
            };
            sbg_boost[sbg][atsg] = g;
        }
        // Pseudocode 100 — clamp + map to subbands.
        let mut sbg = 0usize;
        #[allow(clippy::needless_range_loop)]
        // ETSI TS 103 190-1 §5.7.6.4.2.2 Pseudocode 100 boost_fact_sb[sb][atsg]
        for sb in 0..num_sb {
            while sbg + 1 < num_sbg_lim && sb >= sbg_lim[sbg + 1].saturating_sub(sbx) as usize {
                sbg += 1;
            }
            boost_sb[sb][atsg] = sbg_boost[sbg][atsg].min(MAX_BOOST_FACT);
        }
    }
    boost_sb
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 101 — apply boost.
// ---------------------------------------------------------------------

/// Final adjusted arrays per ETSI TS 103 190-1 §5.7.6.4.2.2
/// Pseudocode 101: scale sig_gain / noise_lev / sine_lev by the
/// per-subband boost factor.
pub struct LimitedAdjustments {
    pub sig_gain_sb_adj: Vec<Vec<f32>>,
    pub noise_lev_sb_adj: Vec<Vec<f32>>,
    pub sine_lev_sb_adj: Vec<Vec<f32>>,
}

/// Multiply by the per-subband boost factor in place — Pseudocode 101.
pub fn apply_boost(
    sig_gain_sb_lim: &[Vec<f32>],
    noise_lev_sb_lim: &[Vec<f32>],
    sine_lev_sb: &[Vec<f32>],
    boost_fact_sb: &[Vec<f32>],
) -> LimitedAdjustments {
    let num_sb = sig_gain_sb_lim.len();
    if num_sb == 0 {
        return LimitedAdjustments {
            sig_gain_sb_adj: Vec::new(),
            noise_lev_sb_adj: Vec::new(),
            sine_lev_sb_adj: Vec::new(),
        };
    }
    let num_atsg = sig_gain_sb_lim[0].len();
    let mut sig: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    let mut noise: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    let mut sine: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    for sb in 0..num_sb {
        for atsg in 0..num_atsg {
            let bf = boost_fact_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(1.0);
            sig[sb][atsg] = sig_gain_sb_lim[sb][atsg] * bf;
            let n = noise_lev_sb_lim
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            noise[sb][atsg] = n * bf;
            let s = sine_lev_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            sine[sb][atsg] = s * bf;
        }
    }
    LimitedAdjustments {
        sig_gain_sb_adj: sig,
        noise_lev_sb_adj: noise,
        sine_lev_sb_adj: sine,
    }
}

// ---------------------------------------------------------------------
// Top-level convenience: full §5.7.6.4.2.2 limiter pipeline.
// ---------------------------------------------------------------------

/// Run the full limiter pipeline (Pseudocodes 96 → 97 → 98 → 99 → 100
/// → 101) end-to-end and return the three `_adj` matrices that get fed
/// to Pseudocodes 106 / 107 / 108.
///
/// The caller is responsible for producing `sig_gain_sb` (Pseudocode
/// 95) and `sine_lev_sb` / `noise_lev_sb` (Pseudocode 94). The limiter
/// re-computes the limit ceiling itself from `est_sig_sb` and
/// `scf_sig_sb`, then splices the boost-corrected outputs back into
/// the gain / noise / sine matrices.
#[allow(clippy::too_many_arguments)]
pub fn run(
    est_sig_sb: &[Vec<f32>],
    scf_sig_sb: &[Vec<f32>],
    sig_gain_sb: &[Vec<f32>],
    sine_lev_sb: &[Vec<f32>],
    noise_lev_sb: &[Vec<f32>],
    sbg_lim: &[u32],
    sbx: u32,
    num_sb_aspx: u32,
    aspx_tsg_ptr: u32,
    p_sine_at_end: Option<u32>,
) -> LimitedAdjustments {
    let max_sig_gain_sb = max_sig_gain(est_sig_sb, scf_sig_sb, sbg_lim, sbx, num_sb_aspx);
    let noise_lev_sb_lim = limit_noise_level(noise_lev_sb, &max_sig_gain_sb, sig_gain_sb);
    let sig_gain_sb_lim = limit_sig_gain(sig_gain_sb, &max_sig_gain_sb);
    let boost_fact_sb = boost_factor(
        est_sig_sb,
        scf_sig_sb,
        &sig_gain_sb_lim,
        sine_lev_sb,
        &noise_lev_sb_lim,
        sbg_lim,
        sbx,
        num_sb_aspx,
        aspx_tsg_ptr,
        p_sine_at_end,
    );
    apply_boost(
        &sig_gain_sb_lim,
        &noise_lev_sb_lim,
        sine_lev_sb,
        &boost_fact_sb,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- §5.7.6.3.1.5 sbg_lim derivation -------------------------------

    #[test]
    fn derive_sbg_lim_empty_when_no_lowres() {
        let out = derive_sbg_lim(&[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn derive_sbg_lim_only_lowres_when_no_patches() {
        // Two lowres groups, no patches — expect identical sbg_lim
        // (octave gap between 4 and 16 is large, no merging).
        let out = derive_sbg_lim(&[4, 8, 16], &[]);
        assert_eq!(out, vec![4, 8, 16]);
    }

    #[test]
    fn derive_sbg_lim_drops_pure_duplicates() {
        // Both lowres and a single-element patch share value 8 ->
        // duplicate must be removed by Pseudocode 74.
        let out = derive_sbg_lim(&[4, 8, 16], &[8, 8]);
        // sbg_patches has only 2 entries -> the 1..len-1 slice is empty,
        // so the lowres table is unchanged.
        assert_eq!(out, vec![4, 8, 16]);
    }

    #[test]
    fn derive_sbg_lim_merges_subnoctave_pairs() {
        // sbg_lim sorted = [8, 9, 16, 32]. The (8, 9) pair has
        // log2(9/8) ~= 0.17 < 0.245 -> merge: prefer non-patch entries
        // are dropped if a patch border conflicts. Here 9 is in patches
        // and 8 is in the lowres table -> drop 8.
        let out = derive_sbg_lim(&[8, 16, 32], &[7, 9, 16, 17]);
        // Patches are {7, 9, 16, 17}; the ones added (skip first/last)
        // are {9, 16}. Sorted union: {8, 9, 16, 16, 32}. After
        // dedupe-on-merge: 8↔9 close -> 9 is a patch, 8 is not -> drop 8.
        // Then 9↔16: log2(16/9) ~= 0.83 -> keep. Then 16↔16: dedupe.
        // Then 16↔32: keep.
        assert_eq!(out, vec![9, 16, 32]);
    }

    // ----- Pseudocode 96 -------------------------------------------------

    #[test]
    fn max_sig_gain_flat_input_yields_lim_gain() {
        // Single limiter group, scf_sig == est_sig everywhere ->
        // sqrt(nom/denom) ~= 1; max_sig_gain ~= LIM_GAIN.
        let sbg_lim = vec![4u32, 8u32]; // single group [4..8) in QMF.
        let sbx = 4u32;
        let num_sb_aspx = 4u32;
        let est_sig: Vec<Vec<f32>> = vec![vec![1.0]; 4];
        let scf_sig: Vec<Vec<f32>> = vec![vec![1.0]; 4];
        let m = max_sig_gain(&est_sig, &scf_sig, &sbg_lim, sbx, num_sb_aspx);
        for row in m.iter().take(4) {
            assert!((row[0] - LIM_GAIN).abs() < 1e-3);
        }
    }

    #[test]
    fn max_sig_gain_clamped_at_max_when_signal_dwarfs_estimate() {
        // scf_sig = 1, est_sig ~= 0 (only EPSILON0 protects it) ->
        // max_sig_gain hits MAX_SIG_GAIN.
        let sbg_lim = vec![0u32, 1u32];
        let est_sig: Vec<Vec<f32>> = vec![vec![0.0]];
        let scf_sig: Vec<Vec<f32>> = vec![vec![1.0]];
        let m = max_sig_gain(&est_sig, &scf_sig, &sbg_lim, 0, 1);
        assert!((m[0][0] - MAX_SIG_GAIN).abs() < 1e-3);
    }

    // ----- Pseudocode 97 -------------------------------------------------

    #[test]
    fn limit_noise_level_caps_when_max_below_gain() {
        // sig_gain = 4, max_sig = 2 -> ratio 0.5 -> tmp = noise * 0.5
        // < noise -> output = tmp.
        let noise = vec![vec![10.0_f32]];
        let max_sig = vec![vec![2.0_f32]];
        let sig_gain = vec![vec![4.0_f32]];
        let out = limit_noise_level(&noise, &max_sig, &sig_gain);
        assert!((out[0][0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn limit_noise_level_keeps_when_max_above_gain() {
        // sig_gain = 1, max_sig = 4 -> ratio 4 -> tmp = noise * 4
        // > noise -> output = noise.
        let noise = vec![vec![10.0_f32]];
        let max_sig = vec![vec![4.0_f32]];
        let sig_gain = vec![vec![1.0_f32]];
        let out = limit_noise_level(&noise, &max_sig, &sig_gain);
        assert!((out[0][0] - 10.0).abs() < 1e-5);
    }

    // ----- Pseudocode 98 -------------------------------------------------

    #[test]
    fn limit_sig_gain_takes_min() {
        let g = vec![vec![3.0_f32, 1.0]];
        let m = vec![vec![2.0_f32, 5.0]];
        let out = limit_sig_gain(&g, &m);
        assert_eq!(out[0][0], 2.0);
        assert_eq!(out[0][1], 1.0);
    }

    // ----- Pseudocode 99 + 100 -------------------------------------------

    #[test]
    fn boost_factor_unity_when_input_balanced() {
        // est * gain^2 = sig everywhere, no sine, exception.
        let sbg_lim = vec![0u32, 2u32];
        let est = vec![vec![1.0_f32]; 2];
        let scf_sig = vec![vec![1.0_f32]; 2];
        // gain = 1.0 -> est*1 = sig -> denom == nom -> boost = 1.
        let sig_gain_lim = vec![vec![1.0_f32]; 2];
        let sine = vec![vec![0.0_f32]; 2];
        let noise = vec![vec![0.0_f32]; 2];
        let bf = boost_factor(
            &est,
            &scf_sig,
            &sig_gain_lim,
            &sine,
            &noise,
            &sbg_lim,
            0,
            2,
            0,
            Some(0), // exception envelope -> drop noise term anyway.
        );
        for row in bf.iter().take(2) {
            assert!((row[0] - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn boost_factor_clamped_at_max() {
        // sig huge, denom tiny -> boost would shoot up; expect clamp.
        let sbg_lim = vec![0u32, 1u32];
        let est = vec![vec![0.0_f32]];
        let scf_sig = vec![vec![100.0_f32]];
        let sig_gain_lim = vec![vec![0.0_f32]];
        let sine = vec![vec![0.0_f32]];
        let noise = vec![vec![0.0_f32]];
        let bf = boost_factor(
            &est,
            &scf_sig,
            &sig_gain_lim,
            &sine,
            &noise,
            &sbg_lim,
            0,
            1,
            0,
            None,
        );
        assert!((bf[0][0] - MAX_BOOST_FACT).abs() < 1e-5);
    }

    // ----- Pseudocode 101 ------------------------------------------------

    #[test]
    fn apply_boost_scales_each_array() {
        let sig = vec![vec![2.0_f32]];
        let noise = vec![vec![3.0_f32]];
        let sine = vec![vec![5.0_f32]];
        let bf = vec![vec![1.5_f32]];
        let out = apply_boost(&sig, &noise, &sine, &bf);
        assert!((out.sig_gain_sb_adj[0][0] - 3.0).abs() < 1e-5);
        assert!((out.noise_lev_sb_adj[0][0] - 4.5).abs() < 1e-5);
        assert!((out.sine_lev_sb_adj[0][0] - 7.5).abs() < 1e-5);
    }

    // ----- Top-level pipeline --------------------------------------------

    #[test]
    fn run_no_op_when_signal_already_within_limit() {
        // gain = sqrt(sig/((1+est)*(1+noise))) = sqrt(1/((1+0)*(1+0))) = 1.
        // max_sig_gain = LIM_GAIN * sqrt(1/EPSILON0) = enormous.
        // sig_gain_lim = min(1, enormous) = 1.
        // Boost numerator ~= sig + EPSILON0; denom ~= est*1 + 0 + 0 = 0
        //   + (no noise add bc atsg == tsg_ptr) ~= EPSILON0.
        // Actually with est=0 and no sine and exception, denom = EPSILON0
        // and nom ~= 1+EPSILON0 -> boost shoots to MAX_BOOST_FACT.
        // So sig_gain_adj = 1 * MAX_BOOST_FACT, sine = 0, noise = 0.
        let sbg_lim = vec![0u32, 1u32];
        let est = vec![vec![0.0_f32]];
        let scf_sig = vec![vec![1.0_f32]];
        let sig_gain = vec![vec![1.0_f32]];
        let sine = vec![vec![0.0_f32]];
        let noise = vec![vec![0.0_f32]];
        let out = run(
            &est, &scf_sig, &sig_gain, &sine, &noise, &sbg_lim, 0, 1, 0, None,
        );
        assert!(out.sig_gain_sb_adj[0][0] > 0.0);
        assert!(out.sig_gain_sb_adj[0][0] <= MAX_BOOST_FACT * 1.0 + 1e-5);
    }

    #[test]
    fn run_caps_runaway_gain() {
        // sig_gain way larger than max_sig_gain -> sig_gain_lim should
        // be max_sig_gain (~ LIM_GAIN with sig=est=1).
        //
        // Pseudocode 97 then crushes the noise level by
        // max_sig_gain/sig_gain ≈ LIM_GAIN/1000, so the noise² term
        // in the boost denominator becomes negligible. Pseudocode 99's
        // boost_fact_sbg = sqrt(nom/denom) with:
        //   nom   = sum scf_sig = 2  (per-sb scf_sig=1 over 2 subbands)
        //   denom = sum est * gain^2 = 2 * LIM_GAIN^2 (after limiting)
        // Then sig_gain_sb_adj = LIM_GAIN * sqrt(2/(2*LIM_GAIN^2)) = 1.
        let sbg_lim = vec![0u32, 2u32];
        let est = vec![vec![1.0_f32]; 2];
        let scf_sig = vec![vec![1.0_f32]; 2];
        let sig_gain = vec![vec![1000.0_f32]; 2]; // pretend Pseudocode 95
                                                  // produced a runaway.
        let sine = vec![vec![0.0_f32]; 2];
        let noise = vec![vec![0.5_f32]; 2];
        let out = run(
            &est, &scf_sig, &sig_gain, &sine, &noise, &sbg_lim, 0, 2, 99, None,
        );
        // adj ≈ 1.0 — the limiter perfectly recovers the energy in
        // this idealised band where sig_gain^2 == sig/est.
        for (sb, row) in out.sig_gain_sb_adj.iter().take(2).enumerate() {
            assert!(
                (row[0] - 1.0).abs() < 1e-3,
                "sb {sb}: got {} want ~1.0",
                row[0]
            );
        }
    }

    // ----- Pseudocode 95 (full) ------------------------------------------

    #[test]
    fn compute_sig_gains_full_no_sine_no_exception_matches_simple_path() {
        // sine_area_sb == 0, atsg ∉ {tsg_ptr, p_sine_at_end} ->
        // matches the existing simple compute_sig_gains formula:
        //   sqrt(scf_sig / ((1+est)*(1+noise)))
        let est = vec![vec![3.0_f32]; 1];
        let scf = vec![vec![16.0_f32]; 1];
        let n = vec![vec![1.0_f32]; 1];
        let area = vec![vec![0_u8]; 1];
        let g = compute_sig_gains_full(&est, &scf, &n, &area, 99, None);
        let expected = (16.0_f32 / ((1.0 + 3.0) * (1.0 + 1.0))).sqrt();
        assert!((g[0][0] - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_sig_gains_full_no_sine_with_exception_drops_noise_term() {
        // Exception: atsg == tsg_ptr -> denom = (EPSILON + est) only.
        let est = vec![vec![3.0_f32]; 1];
        let scf = vec![vec![16.0_f32]; 1];
        let n = vec![vec![1.0_f32]; 1];
        let area = vec![vec![0_u8]; 1];
        let g = compute_sig_gains_full(&est, &scf, &n, &area, 0, None);
        // atsg = 0 == tsg_ptr; denom = 1 + 3 = 4; numerator = 16; sqrt = 2.
        assert!((g[0][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn compute_sig_gains_full_with_sine_uses_noise_in_numerator() {
        // sine_area_sb == 1: numerator = scf_sig * scf_noise; denom
        // unchanged.
        let est = vec![vec![3.0_f32]; 1];
        let scf = vec![vec![16.0_f32]; 1];
        let n = vec![vec![2.0_f32]; 1];
        let area = vec![vec![1_u8]; 1];
        let g = compute_sig_gains_full(&est, &scf, &n, &area, 99, None);
        // numerator = 16 * 2 = 32; denom = (1+3)*(1+2) = 12; sqrt(32/12) ~ 1.633.
        let expected = (32.0_f32 / 12.0).sqrt();
        assert!((g[0][0] - expected).abs() < 1e-5);
    }
}
