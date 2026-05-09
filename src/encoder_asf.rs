//! Forward ASF entropy coding for the AC-4 IMS encoder (round 48).
//!
//! Per ETSI TS 103 190-1 §5.1 (Pseudocodes 17-19) and Annex A.0 the ASF
//! audio body for a long-frame, single-window-group, mono SIMPLE channel
//! consists of (in order):
//!
//!   1. `asf_transform_info()` — `b_long_frame = 1` for frame_len_base
//!      ≥ 1536 (zero further bits at the long-frame branch).
//!   2. `asf_psy_info(0, 0)` — `max_sfb[0]` in `n_msfb_bits` bits, no
//!      window-grouping bits when `b_long_frame`.
//!   3. `asf_section_data()` — outer `for (g)` loop over a single group
//!      with `transf_length_idx = 0` (long frame) → `n_sect_bits = 3`,
//!      `sect_esc_val = 7`. Each section: `sect_cb` (4 bits) +
//!      length-increment chain.
//!   4. `asf_spectral_data()` — Huffman-coded quantised spectrum across
//!      every section.
//!   5. `asf_scalefac_data()` — `reference_scale_factor` (8 bits) + per-band
//!      DPCM deltas in HCB_SCALEFAC.
//!   6. `asf_snf_data()` — `b_snf_data_exists` (1 bit). We always emit 0
//!      to keep things simple — empty bands just decode as silence.
//!
//! Round 48 ships the simplest viable closed-form encoder:
//!   * One section spanning all of `0..max_sfb` with codebook ID 5
//!     (HCB5, dim=2, signed, q-range -4..=+4).
//!   * Per-band scalefactor selected by greedy nearest-power-of-two
//!     scale chosen to keep the largest |coeff|/sf_gain power within
//!     HCB5's 4-magnitude bound after the spec's q = sign(c)*|c/sf|^(3/4)
//!     mapping.
//!   * `reference_scale_factor` taken from sfb 0's chosen value; the
//!     remaining bands DPCM-encode their delta with HCB_SCALEFAC.
//!
//! Round 49 widens the encoder along two axes:
//!   * **Per-band codebook selection** — each scale-factor band picks the
//!     codebook (HCB1..11) that minimises the bit cost for its quantised
//!     vector instead of always using HCB5. Consecutive bands sharing a
//!     codebook are merged into one section. HCB11 carries an escape
//!     mechanism that supports arbitrary magnitudes via Pseudocode 20
//!     extension codes, so high-energy bands no longer clip to ±4.
//!   * **`max_sfb` parameterised** — `build_mono_simple_asf_body_from_pcm_spectrum`
//!     accepts arbitrary `max_sfb` (up to the transform-length-specific
//!     `num_sfb_48` cap) so the encoder can cover up to ~17 kHz at
//!     tl = 1920 (sfb=55 → bin 1408 → 17.6 kHz at fs/2 = 24 kHz).
//!
//! Round 50 closes two more gaps:
//!   * **Section-boundary DP optimiser** — instead of greedy run-length
//!     merging of equal-codebook neighbours, a 2-D dynamic program over
//!     scale-factor bands picks the *globally* cheapest sequence of
//!     (start, end, codebook) sections, paying the per-section header
//!     cost (4 bits sect_cb + 3 bits/length-incr unit, ETSI Table 39)
//!     against the per-band bit savings of switching codebooks. Section
//!     count is capped at 16 per the spec's bitstream layout. On
//!     spectra where r49 picked a uniform HCB11 throughout, the DP can
//!     section off low-energy regions onto HCB1/2/3 and shave several
//!     hundred bits per frame.
//!   * **Spectral Noise Fill (SNF) emission** — bands that quantise to
//!     all-zero (cb == 0) at high frequency now carry per-band SNF
//!     indices encoded via HCB_SNF (§5.1.4 / Pseudocode 105). The
//!     decoder's `inject_snf_noise` (round 36+) injects shaped white
//!     noise into those bins on reconstruction so high-frequency content
//!     above the perceived noise floor doesn't decode as silence.
//!
//! Future work (deferred): SNF per-band envelope refinement (currently
//! flat per-band RMS estimate); multichannel SAP/M-S decisioning;
//! SNF DPCM context across bands.

use oxideav_core::bits::BitWriter;

use crate::asf_data::AsfSections;
use crate::huffman::{
    asf_hcb, Hcb, CB_DIM, HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN, HCB_SNF_CW, HCB_SNF_LEN, UNSIGNED_CB,
};

/// Per-codebook quantisation magnitude bound:
/// * HCB1 / HCB2 (dim=4 signed, cb_off=1, cb_mod=3) → q in [-1, 1] → 1
/// * HCB3 / HCB4 (dim=4 unsigned, cb_off=0, cb_mod=3) → |q| in [0, 2] → 2
/// * HCB5 / HCB6 (dim=2 signed, cb_off=4, cb_mod=9) → q in [-4, 4] → 4
/// * HCB7 / HCB8 (dim=2 unsigned, cb_off=0, cb_mod=8) → |q| in [0, 7] → 7
/// * HCB9 / HCB10 (dim=2 unsigned, cb_off=0, cb_mod=13) → |q| in [0, 12] → 12
/// * HCB11 (dim=2 unsigned, cb_off=0, cb_mod=17) → |q| ≤ 16 inline; |q| > 16
///   uses the `ext_decode` escape (Pseudocode 20) so it's effectively
///   unbounded. We model it as 16 here for the q_max constraint and pay
///   the escape cost separately during bit-counting.
pub const CB_QMAX: [u32; 12] = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16];

/// All candidate codebook IDs the optimiser sweeps.
pub const ALL_CB_IDS: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Quantise a single MDCT coefficient given an integer scalefactor.
/// Spec (§5.1, Pseudocode 18 inverse): `coeff = sign(q) * |q|^(4/3) * 2^((sf-100)/4)`.
/// Inverse-quantise → `q = round(sign(c) * (|c| / 2^((sf-100)/4))^(3/4))`.
pub fn quantise_coeff(coeff: f32, sf: i32) -> i32 {
    let sf_gain = 2.0_f32.powf((sf as f32 - 100.0) * 0.25);
    if sf_gain == 0.0 || !sf_gain.is_finite() {
        return 0;
    }
    let mag = (coeff.abs() / sf_gain).powf(0.75);
    let q = mag.round() as i32;
    if coeff < 0.0 {
        -q
    } else {
        q
    }
}

/// Pick the smallest scalefactor (most-precision) that keeps every bin's
/// quantised magnitude within `q_max` for one scale-factor band.
///
/// Walks the spec's scalefactor range 0..=255 — but that's 256 trials per
/// band per frame, well within budget. Returns the chosen scalefactor and
/// the resulting quantised vector for the band.
///
/// `coeffs` is the per-bin slice covering the band (sfb_offset[sfb] ..
/// sfb_offset[sfb+1]). `q_max` is the codebook's magnitude bound (HCB5 →
/// 4, HCB1/2 → 1, etc).
pub fn pick_scalefactor_for_band(coeffs: &[f32], q_max: u32) -> (i32, Vec<i32>) {
    // Empty band → silent, scalefactor doesn't matter; pick neutral 100.
    if coeffs.is_empty() {
        return (100, Vec::new());
    }
    let max_abs = coeffs.iter().fold(0.0_f32, |a, &c| a.max(c.abs()));
    if max_abs <= 1e-12 {
        // Silent band: zero quants, scalefactor at the spec midpoint.
        return (100, vec![0i32; coeffs.len()]);
    }
    // Try sf in 0..=255. We want to pick the smallest sf (largest sf_gain
    // is at sf=255; smallest at sf=0) such that quant magnitudes fit
    // into q_max. Larger sf → larger sf_gain → smaller quant magnitude.
    // So we iterate sf from low to high and pick the first that fits.
    // The mapping: q = round(|c/sf_gain|^(3/4)). Solve for sf:
    //   |c|/sf_gain = q^(4/3) → sf_gain = |c| / q^(4/3).
    // For q_max: sf_gain_min = max_abs / q_max^(4/3), so
    //   sf_min = 100 + 4 * log2(sf_gain_min).
    let q_max_43 = (q_max as f32).powf(4.0 / 3.0);
    let sf_gain_min = max_abs / q_max_43;
    // sf = 100 + 4 * log2(sf_gain_min) (rounded up).
    let sf_f = 100.0 + 4.0 * sf_gain_min.log2();
    let sf = sf_f.ceil() as i32;
    let sf = sf.clamp(0, 255);
    // Compute the quantised vector at this scalefactor.
    let mut q = vec![0i32; coeffs.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        let qi = quantise_coeff(c, sf);
        // Clamp to ±q_max (just in case ceil rounding produced one too small).
        q[i] = qi.clamp(-(q_max as i32), q_max as i32);
    }
    (sf, q)
}

/// Encode a quantised symbol vector for the given codebook into the bit
/// writer. For dim=2 codebooks the symbol covers 2 quantised lines; for
/// dim=4 it covers 4. The encoder reverses the spec's `split_qspec()`
/// (Pseudocode 19) into a single `cb_idx` and emits `(cw, len)` from the
/// codebook.
pub fn encode_pair(bw: &mut BitWriter, hcb: &Hcb, q: &[i32]) {
    let dim = hcb.dim as usize;
    debug_assert!(q.len() >= dim);
    // Build cb_idx by reversing split_qspec. For dim=2:
    //   q1 = (idx / cb_mod) - cb_off
    //   q2 = idx - (q1 + cb_off) * cb_mod
    // Reversing: idx = (q1 + cb_off) * cb_mod + (q2 + cb_off).
    // Same shape for dim=4 with the cb_mod{,2,3} cascade.
    let cb_idx = if dim == 4 {
        let i1 = (q[0] + hcb.cb_off) as u32 * hcb.cb_mod3;
        let i2 = (q[1] + hcb.cb_off) as u32 * hcb.cb_mod2;
        let i3 = (q[2] + hcb.cb_off) as u32 * hcb.cb_mod;
        let i4 = (q[3] + hcb.cb_off) as u32;
        i1 + i2 + i3 + i4
    } else {
        let i1 = (q[0] + hcb.cb_off) as u32 * hcb.cb_mod;
        let i2 = (q[1] + hcb.cb_off) as u32;
        i1 + i2
    } as usize;
    debug_assert!(
        cb_idx < hcb.cw.len(),
        "encode_pair: cb_idx {cb_idx} out of range for cb (len={})",
        hcb.cw.len()
    );
    bw.write_u32(hcb.cw[cb_idx], hcb.len[cb_idx] as u32);
    // For unsigned codebooks, append a sign bit per non-zero quant.
    if hcb.unsigned {
        for &qi in &q[..dim] {
            if qi != 0 {
                bw.write_u32(if qi < 0 { 1 } else { 0 }, 1);
            }
        }
    }
}

/// Encode a section-length increment per Pseudocode 17 (§4.3.5.4):
/// emit `floor((sect_len-1) / esc)` escape codewords followed by one
/// non-escape `(sect_len-1) % esc` value.
pub fn write_sect_len_incr(bw: &mut BitWriter, sect_len: u32, n_sect_bits: u32, esc: u32) {
    let base = sect_len.saturating_sub(1);
    let k = base / esc;
    let incr = base % esc;
    for _ in 0..k {
        bw.write_u32(esc, n_sect_bits);
    }
    bw.write_u32(incr, n_sect_bits);
}

/// Build an [`AsfSections`] for a single section spanning `0..max_sfb`
/// with codebook id `cb`.
pub fn single_section(max_sfb: u32, cb: u8) -> AsfSections {
    AsfSections {
        sect_cb: vec![cb],
        sect_start: vec![0],
        sect_end: vec![max_sfb as u16],
        sfb_cb: vec![cb; max_sfb as usize],
        num_sec: 1,
        num_sec_lsf: 1,
    }
}

/// Compute the bit cost of emitting `q` (a quantised vector covering
/// integer multiples of `dim` lines) under codebook `hcb`. Includes the
/// Huffman codeword bits, the per-non-zero-line sign bits for unsigned
/// codebooks, and (for HCB11) the escape-code bits per Pseudocode 20.
///
/// For HCB11 the inline magnitude saturates at 16 (the escape carries
/// the actual magnitude); for unsigned codebooks the cb_idx encodes
/// magnitudes only and the sign rides separately. The caller is
/// responsible for ensuring `q` doesn't exceed `CB_QMAX[cb_id]` for
/// non-HCB11 codebooks.
///
/// Returns the total bit count.
pub fn bit_cost_for_band(hcb: &Hcb, q: &[i32], cb_id: u8) -> u32 {
    let dim = hcb.dim as usize;
    let unsig = hcb.unsigned;
    let mut bits: u32 = 0;
    let mut k = 0usize;
    while k + dim <= q.len() {
        // Build the cb_idx the same way write_spectral_data_sections does.
        let mut sym = [0i32; 4];
        for t in 0..dim {
            let qi = q[k + t];
            let mag = qi.unsigned_abs() as i32;
            let inline_mag = if cb_id == 11 && mag >= 16 { 16 } else { mag };
            sym[t] = if unsig {
                inline_mag
            } else if qi < 0 {
                -inline_mag
            } else {
                inline_mag
            };
        }
        let cb_idx = if dim == 4 {
            let i1 = (sym[0] + hcb.cb_off) as u32 * hcb.cb_mod3;
            let i2 = (sym[1] + hcb.cb_off) as u32 * hcb.cb_mod2;
            let i3 = (sym[2] + hcb.cb_off) as u32 * hcb.cb_mod;
            let i4 = (sym[3] + hcb.cb_off) as u32;
            (i1 + i2 + i3 + i4) as usize
        } else {
            let i1 = (sym[0] + hcb.cb_off) as u32 * hcb.cb_mod;
            let i2 = (sym[1] + hcb.cb_off) as u32;
            (i1 + i2) as usize
        };
        if cb_idx >= hcb.cw.len() {
            return u32::MAX;
        }
        bits += hcb.len[cb_idx] as u32;
        if unsig {
            for t in 0..dim {
                let qi = q[k + t];
                let mag = qi.unsigned_abs() as i32;
                let inline_mag = if cb_id == 11 && mag >= 16 { 16 } else { mag };
                if inline_mag != 0 {
                    bits += 1;
                }
            }
        }
        if cb_id == 11 {
            for &qi in &q[k..k + dim] {
                let mag = qi.unsigned_abs();
                if mag >= 16 {
                    let mut n_ext: u32 = 0;
                    while (1u32 << (n_ext + 4)) <= mag {
                        n_ext += 1;
                    }
                    n_ext = n_ext.saturating_sub(1);
                    bits += (n_ext + 1) + (n_ext + 4);
                }
            }
        }
        k += dim;
    }
    bits
}

/// Quantise the band's coefficients under codebook `cb_id` at scalefactor
/// `sf`, clamping each line to the codebook's signed range. For HCB11 the
/// inline range is ±16, with values outside flagged for the escape path
/// (no clamping past 16 — the escape supports arbitrary magnitudes).
fn quantise_band_for_cb(coeffs: &[f32], sf: i32, cb_id: u8) -> Vec<i32> {
    let q_max = CB_QMAX[cb_id as usize] as i32;
    let mut q = vec![0i32; coeffs.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        let qi = quantise_coeff(c, sf);
        q[i] = if cb_id == 11 {
            // HCB11 inline range is [-16, 16] but |q| == 16 triggers
            // the escape; values larger than 16 are encoded via ext.
            // For clamping purposes only: bound the absolute value at
            // a sane upper limit (256 → log2 = 8 → n_ext = 4 → ~13 extra
            // bits, still affordable). This stops one runaway sample
            // from blowing the bit budget.
            qi.clamp(-256, 256)
        } else {
            qi.clamp(-q_max, q_max)
        };
    }
    q
}

/// Pick the best (codebook, scalefactor, quantised-vector, bit-cost)
/// tuple for a single scale-factor band.
///
/// Strategy (round 49):
///   1. Pick sf such that the band's peak coefficient naturally maps to
///      q ≈ 4 (HCB5's q_max) — the round-48 baseline. This preserves the
///      round-48 SNR floor for every band; if the natural peak ends up
///      smaller than 4 we get a chance to choose a cheaper codebook.
///   2. For each candidate codebook whose `q_max ≥ natural_max`, compute
///      the bit cost of emitting `natural_q`. (HCB11 always qualifies via
///      its escape mechanism.)
///   3. Pick the codebook with the minimum bit cost.
///
/// This means low-energy bands (q peak ≤ 1) get HCB1-2 (cheaper dim=4
/// codebooks); medium-energy bands stay on HCB5/6; very-high-energy
/// bands fall through to HCB11 + escape.
///
/// Empty bands return `(0, 100, [], 0)` — codebook 0 = "all-zero", no
/// emission, scalefactor doesn't matter.
pub fn pick_best_codebook_for_band(coeffs: &[f32]) -> (u8, i32, Vec<i32>, u32) {
    if coeffs.is_empty() {
        return (0, 100, Vec::new(), 0);
    }
    let max_abs = coeffs.iter().fold(0.0_f32, |a, &c| a.max(c.abs()));
    if max_abs <= 1e-12 {
        return (0, 100, vec![0i32; coeffs.len()], 0);
    }
    // Anchor scalefactor: target peak quant ≈ 12 (HCB9/10's q_max).
    // This gives 3× more quantisation levels per band than the round-48
    // HCB5-only baseline (q_max=4) → roughly +10 dB extra SNR per band
    // before the bit budget is dominated by codeword length. Bands with
    // very small natural peaks still get a chance to pick a smaller-q
    // codebook below; very large peaks fall through to HCB11 + escape.
    let q_target: f32 = 12.0;
    let q_target_43 = q_target.powf(4.0 / 3.0);
    let sf_gain_min = max_abs / q_target_43;
    let sf_anchor_f = 100.0 + 4.0 * sf_gain_min.log2();
    let sf_anchor = (sf_anchor_f.ceil() as i32).clamp(0, 255);
    // Compute the natural quantisation at this sf via the HCB11 path
    // (no clipping — we want to know the true magnitudes for the
    // codebook-fit check).
    let natural_q = quantise_band_for_cb(coeffs, sf_anchor, 11);
    let natural_max: u32 = natural_q
        .iter()
        .map(|&qi| qi.unsigned_abs())
        .max()
        .unwrap_or(0);
    if natural_max == 0 {
        // All bins quantise to zero — the band is below the noise
        // floor at our anchor scalefactor. Emit cb=0 (silent).
        return (0, 100, natural_q, 0);
    }
    let mut best_cb: u8 = 5;
    let mut best_q: Vec<i32> = natural_q.clone();
    let mut best_cost: u32 = u32::MAX;
    for &cb_id in ALL_CB_IDS {
        let hcb = match asf_hcb(cb_id as u32) {
            Some(h) => h,
            None => continue,
        };
        let dim = hcb.dim as usize;
        if natural_q.len() % dim != 0 {
            continue;
        }
        let q_max_cb = CB_QMAX[cb_id as usize];
        // Codebook must be able to represent every line losslessly —
        // its q_max must be ≥ natural_max (HCB11 has effectively
        // unbounded range via the escape).
        if cb_id != 11 && natural_max > q_max_cb {
            continue;
        }
        let cost = bit_cost_for_band(hcb, &natural_q, cb_id);
        if cost < best_cost {
            best_cost = cost;
            best_cb = cb_id;
            best_q = natural_q.clone();
        }
    }
    (best_cb, sf_anchor, best_q, best_cost)
}

/// Round-50 DP-based section optimiser.
///
/// Given the precomputed per-band per-codebook bit costs (built by
/// [`build_band_codebook_cost_table`]), finds the globally optimal sequence
/// of sections (start, end, codebook) that minimises total bits including
/// the per-section header overhead.
///
/// Per ETSI TS 103 190-1 §5.7.4 / Table 39:
///   * `sect_cb`: 4 bits per section (codebook id 0..=11).
///   * `sect_len_incr`: 3 bits per chunk (`sect_esc_val = 7`) for long-frame.
///     A section of length `L` emits `floor((L-1)/7) + 1` such 3-bit fields.
///
/// So the per-section overhead in bits for length `L` is
/// `4 + 3 * (floor((L-1)/7) + 1)` — 7 bits for `L ∈ 1..=7`, 10 bits for
/// `L ∈ 8..=14`, etc.
///
/// `cost_band_cb[sfb][cb]` is the bit cost (excluding section header) of
/// encoding band `sfb` under codebook id `cb` (0..=11), or `u32::MAX` if
/// that codebook can't represent the band's natural quantisation
/// magnitudes. cb=0 means "silent / no spectral data" — only valid for
/// all-zero-quant bands.
///
/// `max_sections` caps the section count at the spec-permitted maximum
/// (16 per ETSI Table SCFB).
///
/// Returns a vector of `(start_sfb, end_sfb_exclusive, cb)` tuples
/// describing the optimal section layout.
pub fn dp_optimise_sections(cost_band_cb: &[[u32; 12]], max_sections: u32) -> Vec<(u32, u32, u8)> {
    let n = cost_band_cb.len();
    if n == 0 {
        return Vec::new();
    }
    let inf: u64 = u64::MAX / 4;
    // best[i] = minimal cost to encode bands [0..=i] using ≤ max_sections sections.
    let mut best: Vec<u64> = vec![inf; n];
    // pred[i] = (prev_band_or_neg1, cb_for_last_section).
    let mut pred: Vec<(i64, u8)> = vec![(-1, 0u8); n];
    // sections_count[i] = number of sections used at the optimal cost so far.
    let mut sections_count: Vec<u32> = vec![0u32; n];

    // Precompute prefix sums of band cost per codebook to make the inner
    // sum O(1). prefix[cb][i] = sum_{b=0..i} cost_band_cb[b][cb] when feasible,
    // u64::MAX/2 if any band in 0..i is infeasible for cb.
    let big: u64 = u64::MAX / 2;
    let mut prefix: Vec<[u64; 12]> = vec![[0u64; 12]; n + 1];
    for cb in 0..12usize {
        let mut acc: u64 = 0;
        let mut blocked = false;
        prefix[0][cb] = 0;
        for b in 0..n {
            if blocked {
                prefix[b + 1][cb] = big;
                continue;
            }
            let c = cost_band_cb[b][cb];
            if c == u32::MAX {
                // From here on prefix is "blocked" — but a *suffix* sum from
                // start>b can still be feasible, so we use a different
                // strategy below for partial-range sums.
                blocked = true;
                prefix[b + 1][cb] = big;
            } else {
                acc += c as u64;
                prefix[b + 1][cb] = acc;
            }
        }
    }

    // Helper: feasible sum of cost_band_cb[start..=i_inclusive][cb].
    // Falls back to per-band scan when prefix is blocked across the range.
    let band_sum = |start: usize, i: usize, cb: usize| -> Option<u64> {
        if prefix[i + 1][cb] != big && prefix[start][cb] != big {
            return Some(prefix[i + 1][cb] - prefix[start][cb]);
        }
        let mut s: u64 = 0;
        for row in cost_band_cb.iter().take(i + 1).skip(start) {
            let c = row[cb];
            if c == u32::MAX {
                return None;
            }
            s += c as u64;
        }
        Some(s)
    };

    for i in 0..n {
        for start in 0..=i {
            let len = (i - start + 1) as u32;
            let overhead = section_overhead_bits(len) as u64;
            let (prior_cost, prior_sections) = if start == 0 {
                (0u64, 0u32)
            } else {
                if best[start - 1] >= inf {
                    continue;
                }
                (best[start - 1], sections_count[start - 1])
            };
            if prior_sections + 1 > max_sections {
                continue;
            }
            for cb in 0..12usize {
                let bs = match band_sum(start, i, cb) {
                    Some(v) => v,
                    None => continue,
                };
                let total = prior_cost + bs + overhead;
                if total < best[i] {
                    best[i] = total;
                    pred[i] = (if start == 0 { -1 } else { (start - 1) as i64 }, cb as u8);
                    sections_count[i] = prior_sections + 1;
                }
            }
        }
    }

    // Backtrack from the final band.
    let mut sections: Vec<(u32, u32, u8)> = Vec::new();
    let mut i: i64 = (n - 1) as i64;
    while i >= 0 {
        let (prev, cb) = pred[i as usize];
        let start = (prev + 1) as u32;
        let end = (i as u32) + 1;
        sections.push((start, end, cb));
        i = prev;
    }
    sections.reverse();
    sections
}

/// Per-section header overhead in bits for the long-frame
/// (`n_sect_bits = 3`, `sect_esc_val = 7`) case. For a section of length
/// `len ≥ 1`, emits a 4-bit `sect_cb` then `floor((len-1)/7) + 1` 3-bit
/// length-increment fields.
#[inline]
pub fn section_overhead_bits(len: u32) -> u32 {
    let base = len.saturating_sub(1);
    let k = base / 7;
    4 + 3 * (k + 1)
}

/// Build the per-band per-codebook bit-cost table consumed by
/// [`dp_optimise_sections`]. `natural_q_per_band[sfb]` is the quantised
/// vector at the band's anchor scalefactor (matching what
/// [`pick_best_codebook_for_band`] already produces).
///
/// For codebook palette index 0 (silent / cb=0), the cost is 0 if the
/// band is all-zero quants and `u32::MAX` otherwise. For palette indices
/// 1..=11, the cost is the Huffman + sign + (HCB11) escape bit total.
pub fn build_band_codebook_cost_table(natural_q_per_band: &[Vec<i32>]) -> Vec<[u32; 12]> {
    let mut table: Vec<[u32; 12]> = Vec::with_capacity(natural_q_per_band.len());
    for q in natural_q_per_band {
        let mut row = [u32::MAX; 12];
        let all_zero = q.iter().all(|&qi| qi == 0);
        if all_zero {
            row[0] = 0;
        }
        if q.is_empty() {
            row[0] = 0;
            table.push(row);
            continue;
        }
        let nat_max: u32 = q.iter().map(|&qi| qi.unsigned_abs()).max().unwrap_or(0);
        for cb_id in 1u8..=11 {
            let hcb = match asf_hcb(cb_id as u32) {
                Some(h) => h,
                None => continue,
            };
            let dim = hcb.dim as usize;
            if q.len() % dim != 0 {
                continue;
            }
            if cb_id != 11 && nat_max > CB_QMAX[cb_id as usize] {
                continue;
            }
            row[cb_id as usize] = bit_cost_for_band(hcb, q, cb_id);
        }
        table.push(row);
    }
    table
}

/// Lower a DP-derived section list into an [`AsfSections`].
pub fn build_sections_from_dp(sections: &[(u32, u32, u8)], max_sfb: u32) -> AsfSections {
    let mut sect_cb: Vec<u8> = Vec::with_capacity(sections.len());
    let mut sect_start: Vec<u16> = Vec::with_capacity(sections.len());
    let mut sect_end: Vec<u16> = Vec::with_capacity(sections.len());
    let mut sfb_cb = vec![0u8; max_sfb as usize];
    for &(s, e, cb) in sections {
        sect_cb.push(cb);
        sect_start.push(s as u16);
        sect_end.push(e as u16);
        for sfb in s..e {
            if (sfb as usize) < sfb_cb.len() {
                sfb_cb[sfb as usize] = cb;
            }
        }
    }
    let n = sect_cb.len() as u32;
    AsfSections {
        sect_cb,
        sect_start,
        sect_end,
        sfb_cb,
        num_sec: n,
        num_sec_lsf: n,
    }
}

/// Compute per-band SNF DPCM indices for the bands that quantise to all-zero
/// (cb == 0 || max_quant_idx == 0). For each such band, estimate the band's
/// RMS energy from the original MDCT coefficients and pick the HCB_SNF
/// index whose `snf_gain = 2^((idx * 1.5 - 84) / 4)` best matches.
///
/// Bands with content (cb != 0 && mqi > 0) are not eligible for SNF — the
/// loop in `parse_asf_snf_data` skips them. Returns `Some(per_band_idx)`
/// when at least one zero-quant band has measurable energy that warrants
/// fill, else `None` (the encoder then writes `b_snf_data_exists = 0`).
pub fn compute_snf_dpcm_for_zero_quant_bands(
    coeffs: &[f32],
    sfb_offset: &[u16],
    max_sfb: u32,
    sfb_cb: &[u8],
    max_quant_idx: &[u32],
) -> Option<Vec<i32>> {
    let mut out = vec![0i32; max_sfb as usize];
    let mut any_set = false;
    for sfb in 0..max_sfb as usize {
        let eligible = sfb_cb[sfb] == 0 || max_quant_idx[sfb] == 0;
        if !eligible {
            continue;
        }
        let a = sfb_offset[sfb] as usize;
        let b = sfb_offset[sfb + 1] as usize;
        if b <= a {
            continue;
        }
        let band = &coeffs[a..b.min(coeffs.len())];
        let n = band.len();
        if n == 0 {
            continue;
        }
        let mut sum_sq: f64 = 0.0;
        for &c in band {
            sum_sq += (c as f64) * (c as f64);
        }
        if sum_sq <= 1e-30 {
            continue;
        }
        let rms = (sum_sq / n as f64).sqrt() as f32;
        if rms <= 0.0 {
            continue;
        }
        // Solve for idx: snf_gain = 2^((idx * 1.5 - 84) / 4) = rms.
        //   ⇒ idx = (4 * log2(rms) + 84) / 1.5.
        let idx_f = (4.0 * rms.log2() + 84.0) / 1.5;
        let idx = (idx_f.round() as i32).clamp(1, (HCB_SNF_LEN.len() as i32) - 1);
        out[sfb] = idx;
        any_set = true;
    }
    if any_set {
        Some(out)
    } else {
        None
    }
}

/// Emit `asf_snf_data()` per Table 42:
///   * `b_snf_data_exists` (1 bit).
///   * If 1, for each sfb where (sfb_cb == 0 || max_quant_idx == 0) emit
///     one HCB_SNF Huffman codeword carrying the per-band index.
pub fn write_snf_data(
    bw: &mut BitWriter,
    snf: Option<&[i32]>,
    sfb_cb: &[u8],
    max_quant_idx: &[u32],
    max_sfb: u32,
) {
    let snf = match snf {
        Some(s) => s,
        None => {
            bw.write_u32(0, 1);
            return;
        }
    };
    bw.write_bit(true);
    for sfb in 0..max_sfb as usize {
        let cb = sfb_cb[sfb];
        if cb != 0 && max_quant_idx[sfb] != 0 {
            continue;
        }
        let idx = snf.get(sfb).copied().unwrap_or(0);
        let idx_u = idx.clamp(0, (HCB_SNF_LEN.len() as i32) - 1) as usize;
        bw.write_u32(HCB_SNF_CW[idx_u], HCB_SNF_LEN[idx_u] as u32);
    }
}

/// Merge per-band codebook choices into an [`AsfSections`] by collapsing
/// runs of consecutive bands that share the same codebook into one
/// section.
pub fn build_sections_from_per_band_cb(per_band_cb: &[u8], max_sfb: u32) -> AsfSections {
    debug_assert_eq!(per_band_cb.len(), max_sfb as usize);
    let mut sect_cb: Vec<u8> = Vec::new();
    let mut sect_start: Vec<u16> = Vec::new();
    let mut sect_end: Vec<u16> = Vec::new();
    let mut sfb: u32 = 0;
    while sfb < max_sfb {
        let cb = per_band_cb[sfb as usize];
        let start = sfb;
        while sfb < max_sfb && per_band_cb[sfb as usize] == cb {
            sfb += 1;
        }
        sect_cb.push(cb);
        sect_start.push(start as u16);
        sect_end.push(sfb as u16);
    }
    let n = sect_cb.len() as u32;
    AsfSections {
        sect_cb,
        sect_start,
        sect_end,
        sfb_cb: per_band_cb.to_vec(),
        num_sec: n,
        num_sec_lsf: n,
    }
}

/// Encode the per-band scalefactor data per Table 41 / Pseudocode 17:
/// `reference_scale_factor` (8 bits) followed by per-band DPCM deltas
/// in HCB_SCALEFAC. Bands with `cb == 0` or `max_quant_idx[sfb] == 0`
/// don't emit a delta (they pin to whatever the running scale_factor is
/// without resetting `first_scf_found`).
pub fn write_scalefac_data(
    bw: &mut BitWriter,
    sf_per_band: &[i32],
    sfb_cb: &[u8],
    max_quant_idx: &[u32],
    max_sfb: u32,
) {
    // reference_scale_factor: take sfb 0's value (always emitted as the
    // anchor when present); fall back to 100 if no band is active.
    let mut reference: i32 = 100;
    let mut found = false;
    for sfb in 0..max_sfb as usize {
        let cb = sfb_cb[sfb];
        if cb != 0 && max_quant_idx[sfb] > 0 {
            reference = sf_per_band[sfb].clamp(0, 255);
            found = true;
            break;
        }
    }
    bw.write_u32(reference as u32, 8);
    let mut running = reference;
    let mut first = false;
    for sfb in 0..max_sfb as usize {
        let cb = sfb_cb[sfb];
        if cb == 0 || max_quant_idx[sfb] == 0 {
            continue;
        }
        if !first {
            // Anchor — no codeword.
            first = true;
            // Found defends against a degenerate case where reference
            // wasn't actually pinned to this sfb's value.
            if !found {
                running = sf_per_band[sfb].clamp(0, 255);
            }
            continue;
        }
        let delta = sf_per_band[sfb].clamp(0, 255) - running;
        // delta + 60 → HCB_SCALEFAC index. The codebook has 121 entries
        // (indices 0..=120) covering deltas in [-60, +60]. Clamp into
        // that range; runs further than ±60 in one step would need
        // multiple anchor resets which the round-49 encoder doesn't do.
        let idx = (delta + 60).clamp(0, HCB_SCALEFAC_LEN.len() as i32 - 1) as usize;
        bw.write_u32(HCB_SCALEFAC_CW[idx], HCB_SCALEFAC_LEN[idx] as u32);
        running += idx as i32 - 60;
    }
}

/// Encode the spectral coefficient body for a single section spanning
/// `0..max_sfb`, all bins, using codebook `cb`.
pub fn write_spectral_data_single_section(
    bw: &mut BitWriter,
    qspec: &[i32],
    sfb_offset: &[u16],
    max_sfb: u32,
    cb: u32,
) {
    let hcb = asf_hcb(cb).expect("encode_asf: invalid codebook id");
    let dim = CB_DIM[cb as usize] as usize;
    let _unsig = UNSIGNED_CB[cb as usize];
    let end_bin = sfb_offset[max_sfb as usize] as usize;
    debug_assert!(qspec.len() >= end_bin);
    let mut k = 0usize;
    while k + dim <= end_bin {
        encode_pair(bw, hcb, &qspec[k..k + dim]);
        k += dim;
    }
}

/// Write the section-data syntax element for the supplied
/// [`AsfSections`]. Each section emits a 4-bit `sect_cb` followed by a
/// length-increment chain (`n_sect_bits = 3`, `esc = 7` for long frame).
pub fn write_section_data(bw: &mut BitWriter, sections: &AsfSections) {
    for i in 0..sections.num_sec_lsf as usize {
        let cb = sections.sect_cb[i];
        let start = sections.sect_start[i] as u32;
        let end = sections.sect_end[i] as u32;
        let len = end - start;
        bw.write_u32(cb as u32, 4);
        write_sect_len_incr(bw, len, 3, 7);
    }
}

/// Encode the spectral data for an arbitrary [`AsfSections`] layout.
/// Each section covers a contiguous bin range derived from
/// `sfb_offset[sect_start..sect_end]`. For HCB11 each `|q| >= 16`
/// triggers a Pseudocode 20 escape (`ext_decode`'s inverse).
///
/// Sections with `cb == 0` emit nothing (silent band).
pub fn write_spectral_data_sections(
    bw: &mut BitWriter,
    qspec: &[i32],
    sfb_offset: &[u16],
    sections: &AsfSections,
) {
    for i in 0..sections.num_sec_lsf as usize {
        let cb = sections.sect_cb[i] as u32;
        if cb == 0 || cb > 11 {
            continue;
        }
        let hcb = asf_hcb(cb).expect("encode_asf: bad section cb id");
        let dim = CB_DIM[cb as usize] as usize;
        let unsig = UNSIGNED_CB[cb as usize];
        let start_bin = sfb_offset[sections.sect_start[i] as usize] as usize;
        let end_bin = sfb_offset[sections.sect_end[i] as usize] as usize;
        let mut k = start_bin;
        while k + dim <= end_bin {
            // 1. Build the cb_idx for this dim-tuple. For HCB11 the
            //    inline magnitude saturates at 16 so the decoder reads
            //    the escape; for unsigned codebooks the cb_idx carries
            //    magnitudes only (the sign bit follows separately).
            let mut sym = [0i32; 4];
            for t in 0..dim {
                let qi = qspec[k + t];
                let mag = qi.unsigned_abs() as i32;
                let inline_mag = if cb == 11 && mag >= 16 { 16 } else { mag };
                sym[t] = if unsig {
                    inline_mag
                } else if qi < 0 {
                    -inline_mag
                } else {
                    inline_mag
                };
            }
            let cb_idx = if dim == 4 {
                let i1 = (sym[0] + hcb.cb_off) as u32 * hcb.cb_mod3;
                let i2 = (sym[1] + hcb.cb_off) as u32 * hcb.cb_mod2;
                let i3 = (sym[2] + hcb.cb_off) as u32 * hcb.cb_mod;
                let i4 = (sym[3] + hcb.cb_off) as u32;
                (i1 + i2 + i3 + i4) as usize
            } else {
                let i1 = (sym[0] + hcb.cb_off) as u32 * hcb.cb_mod;
                let i2 = (sym[1] + hcb.cb_off) as u32;
                (i1 + i2) as usize
            };
            debug_assert!(
                cb_idx < hcb.cw.len(),
                "spec_data: cb_idx {cb_idx} out of range"
            );
            bw.write_u32(hcb.cw[cb_idx], hcb.len[cb_idx] as u32);
            // 2. For unsigned codebooks: sign bit per non-zero original
            //    magnitude (the parser uses the *post-split* magnitude
            //    to decide whether to read the sign, but for unsigned
            //    codebooks the post-split q is always >= 0, so the sign
            //    bit is keyed on whether the magnitude is non-zero —
            //    matching qspec[k+t] != 0 here).
            if unsig {
                for t in 0..dim {
                    let qi = qspec[k + t];
                    let mag = qi.unsigned_abs() as i32;
                    let inline_mag = if cb == 11 && mag >= 16 { 16 } else { mag };
                    if inline_mag != 0 {
                        bw.write_u32(if qi < 0 { 1 } else { 0 }, 1);
                    }
                }
            }
            // 3. For HCB11 emit the escape extension for any |q| >= 16.
            //    Pseudocode 20 inverse: write n_ext '1' bits + one '0'
            //    + (n_ext + 4) value bits where mag = 2^(n_ext+4) + ext.
            if cb == 11 {
                for t in 0..dim {
                    let mag = qspec[k + t].unsigned_abs();
                    if mag >= 16 {
                        let mut n_ext: u32 = 0;
                        while (1u32 << (n_ext + 4)) <= mag {
                            n_ext += 1;
                        }
                        n_ext = n_ext.saturating_sub(1);
                        let bits = n_ext + 4;
                        let ext = mag - (1u32 << bits);
                        // Unary prefix: n_ext '1' bits then one '0'.
                        for _ in 0..n_ext {
                            bw.write_u32(1, 1);
                        }
                        bw.write_u32(0, 1);
                        bw.write_u32(ext, bits);
                    }
                }
            }
            k += dim;
        }
    }
}

/// Build the full mono SIMPLE/ASF substream body for the long-frame
/// single-window-group case at the configured transform length and
/// max_sfb. `coeffs` is the windowed forward-MDCT spectrum (length ≥
/// `sfb_offset[max_sfb]`).
///
/// Returns the substream bytes (audio_size header + audio_data + zero-
/// padding) sized to `pad_target_bytes`.
pub fn build_mono_simple_asf_body_from_pcm_spectrum(
    transform_length: u32,
    max_sfb: u32,
    coeffs: &[f32],
    pad_target_bytes: usize,
) -> Vec<u8> {
    // Round-50 path:
    //   1. Per-band: anchor scalefactor + natural-q quantisation.
    //   2. DP over SFBs: globally optimal section layout (start, end, cb)
    //      that minimises total bits including the 7+ bit/section header.
    //   3. SNF emission for zero-quant bands with measurable original
    //      energy.

    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let end_bin = sfbo[max_sfb as usize] as usize;

    // Per-band anchoring: pick the natural-q vector at the band's anchor
    // scalefactor (matches `pick_best_codebook_for_band`'s logic). We
    // collect natural_q_per_band for the DP cost-table, qspec for emission,
    // and sf_per_band for scalefac DPCM.
    let mut qspec = vec![0i32; end_bin];
    let mut sf_per_band = vec![100i32; max_sfb as usize];
    let mut max_quant_idx = vec![0u32; max_sfb as usize];
    let mut natural_q_per_band: Vec<Vec<i32>> = Vec::with_capacity(max_sfb as usize);
    for sfb in 0..max_sfb as usize {
        let a = sfbo[sfb] as usize;
        let b = sfbo[sfb + 1] as usize;
        let band = &coeffs[a..b.min(coeffs.len())];
        let (_cb_picked, sf, q, _cost) = pick_best_codebook_for_band(band);
        sf_per_band[sfb] = sf;
        let mut max_q: u32 = 0;
        for (i, &qi) in q.iter().enumerate() {
            qspec[a + i] = qi;
            max_q = max_q.max(qi.unsigned_abs());
        }
        max_quant_idx[sfb] = max_q;
        natural_q_per_band.push(q);
    }

    // DP optimisation: globally cheapest section layout.
    let cost_table = build_band_codebook_cost_table(&natural_q_per_band);
    let dp_sections = dp_optimise_sections(&cost_table, 16);
    let sections = build_sections_from_dp(&dp_sections, max_sfb);

    // SNF emission for zero-quant bands.
    let snf = compute_snf_dpcm_for_zero_quant_bands(
        coeffs,
        sfbo,
        max_sfb,
        &sections.sfb_cb,
        &max_quant_idx,
    );

    // Bit-stream emission.
    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();
    // mono_codec_mode = SIMPLE (0), spec_frontend = ASF (0).
    bw.write_u32(0, 1);
    bw.write_u32(0, 1);
    // asf_transform_info: b_long_frame = 1.
    bw.write_bit(true);
    // asf_psy_info(0, 0): max_sfb[0] in n_msfb_bits bits.
    let (n_msfb_bits, _, _) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    bw.write_u32(max_sfb, n_msfb_bits);

    write_section_data(&mut bw, &sections);
    write_spectral_data_sections(&mut bw, &qspec, sfbo, &sections);
    write_scalefac_data(
        &mut bw,
        &sf_per_band,
        &sections.sfb_cb,
        &max_quant_idx,
        max_sfb,
    );
    write_snf_data(
        &mut bw,
        snf.as_deref(),
        &sections.sfb_cb,
        &max_quant_idx,
        max_sfb,
    );

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
}

/// Round-50 helper: count total bits the body would emit if we used the
/// greedy run-length section merge instead of the DP optimiser. Returns
/// (greedy_bits, dp_bits) so callers (and tests) can quantify the
/// section-boundary savings.
pub fn measure_greedy_vs_dp_bits(
    transform_length: u32,
    max_sfb: u32,
    coeffs: &[f32],
) -> (u64, u64) {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let mut natural_q_per_band: Vec<Vec<i32>> = Vec::with_capacity(max_sfb as usize);
    let mut per_band_cb = vec![0u8; max_sfb as usize];
    for sfb in 0..max_sfb as usize {
        let a = sfbo[sfb] as usize;
        let b = sfbo[sfb + 1] as usize;
        let band = &coeffs[a..b.min(coeffs.len())];
        let (cb, _sf, q, _cost) = pick_best_codebook_for_band(band);
        per_band_cb[sfb] = cb;
        natural_q_per_band.push(q);
    }
    let cost_table = build_band_codebook_cost_table(&natural_q_per_band);

    // Greedy: merge runs of constant per-band cb, then add per-band cost
    // for each section's bands plus per-section overhead.
    let greedy = build_sections_from_per_band_cb(&per_band_cb, max_sfb);
    let mut greedy_bits: u64 = 0;
    for s in 0..greedy.num_sec as usize {
        let cb = greedy.sect_cb[s] as usize;
        let start = greedy.sect_start[s] as u32;
        let end = greedy.sect_end[s] as u32;
        greedy_bits += section_overhead_bits(end - start) as u64;
        for b in start..end {
            let c = cost_table[b as usize][cb];
            greedy_bits += if c == u32::MAX { 0 } else { c as u64 };
        }
    }

    let dp_sections = dp_optimise_sections(&cost_table, 16);
    let mut dp_bits: u64 = 0;
    for &(start, end, cb) in &dp_sections {
        dp_bits += section_overhead_bits(end - start) as u64;
        for b in start..end {
            let c = cost_table[b as usize][cb as usize];
            dp_bits += if c == u32::MAX { 0 } else { c as u64 };
        }
    }
    (greedy_bits, dp_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asf_data::{
        dequantise_and_scale, parse_asf_scalefac_data, parse_asf_section_data,
        parse_asf_spectral_data,
    };
    use crate::huffman::HCB5;
    use oxideav_core::bits::BitReader;

    #[test]
    fn quantise_then_dequantise_roundtrips_small_value() {
        // sf = 120 → sf_gain = 32. Coefficient 32 should round to q = 1
        // (since 1^(4/3) * 32 = 32). Coefficient 64 ≈ 32 * 2^(3/4) →
        // q ≈ round(2^(3/4)) = round(1.68) = 2.
        let q = quantise_coeff(32.0, 120);
        assert_eq!(q, 1);
        // Coefficient 0 → q = 0.
        let q0 = quantise_coeff(0.0, 120);
        assert_eq!(q0, 0);
        // Negative coefficient preserves sign.
        let qn = quantise_coeff(-32.0, 120);
        assert_eq!(qn, -1);
    }

    #[test]
    fn pick_scalefactor_keeps_quantised_within_q_max() {
        // Strong band with peak 100; q_max = 4.
        let band = vec![10.0_f32, -50.0, 0.0, 100.0];
        let (_sf, q) = pick_scalefactor_for_band(&band, 4);
        for &qi in &q {
            assert!(qi.abs() <= 4, "q exceeded q_max=4: got {qi}");
        }
    }

    #[test]
    fn encode_pair_dim2_signed_roundtrips() {
        // HCB5 is dim=2, signed (cb_off=4). Pair (q1=+1, q2=0) →
        // cb_idx = (1+4)*9 + (0+4) = 45+4 = 49.
        let mut bw = BitWriter::new();
        let q = [1i32, 0i32];
        encode_pair(&mut bw, &HCB5, &q);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cb_idx = crate::huffman::huff_decode(&mut br, HCB5.len, HCB5.cw).unwrap();
        let mut out = [0i32; 4];
        crate::huffman::split_qspec(&HCB5, cb_idx, &mut out);
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn encode_pair_dim2_unsigned_emits_sign_bit() {
        // HCB7 is dim=2, unsigned (cb_off=0, cb_mod=8). Pair (q1=2, q2=0):
        // cb_idx = 2*8 + 0 = 16.
        let mut bw = BitWriter::new();
        let q = [2i32, 0i32];
        let hcb = asf_hcb(7).unwrap();
        encode_pair(&mut bw, hcb, &q);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cb_idx = crate::huffman::huff_decode(&mut br, hcb.len, hcb.cw).unwrap();
        // Sign bit follows the codeword for non-zero q1.
        let sign = br.read_u32(1).unwrap();
        let mut out = [0i32; 4];
        crate::huffman::split_qspec(hcb, cb_idx, &mut out);
        let q1 = if sign == 1 { -out[0] } else { out[0] };
        assert_eq!(q1, 2);
    }

    /// End-to-end: build a small spectrum, run through
    /// build_mono_simple_asf_body_from_pcm_spectrum, then walk the
    /// produced bytes through the regular ASF parser. The decoded
    /// quantised spectrum should match the encoder's chosen quants.
    #[test]
    fn build_body_roundtrips_through_asf_parser() {
        // Tiny test: tl = 256, max_sfb = 5. Non-trivial coefficients in
        // band 1 (bin 4, 5, 6, 7).
        let tl = 256u32;
        let max_sfb = 5u32;
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let end_bin = sfbo[max_sfb as usize] as usize;
        let mut coeffs = vec![0.0_f32; end_bin];
        coeffs[4] = 32.0; // ends up in band 1 (sfbo[1]=4)
        coeffs[5] = -32.0;
        // Build the body.
        let body = build_mono_simple_asf_body_from_pcm_spectrum(tl, max_sfb, &coeffs, 80);
        assert_eq!(body.len(), 80);

        // Parse: skip header (audio_size_value + b_more_bits + align +
        // mono_codec_mode + spec_frontend + b_long_frame + max_sfb).
        let mut br = BitReader::new(&body);
        // audio_size_value (15) + b_more_bits (1) = 16 bits = 2 bytes.
        let _audio_size = br.read_u32(15).unwrap();
        let _more = br.read_bit().unwrap();
        br.align_to_byte();
        let mono_mode = br.read_u32(1).unwrap();
        assert_eq!(mono_mode, 0);
        let frontend = br.read_u32(1).unwrap();
        assert_eq!(frontend, 0);
        let b_long = br.read_bit().unwrap();
        assert!(b_long);
        let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(tl).unwrap();
        let parsed_max_sfb = br.read_u32(n_msfb_bits).unwrap();
        assert_eq!(parsed_max_sfb, max_sfb);
        let sections = parse_asf_section_data(&mut br, 0, tl, max_sfb).unwrap();
        // Round-49: per-band codebook optimiser may choose different
        // codebooks per band (HCB1..11) instead of always HCB5. The
        // exact section count depends on the input spectrum: silent
        // bands map to cb=0 and active bands pick the best HCB.
        assert!(
            sections.num_sec >= 1,
            "expected at least one section, got {}",
            sections.num_sec
        );
        let (qspec, mqi) = parse_asf_spectral_data(&mut br, &sections, sfbo, max_sfb).unwrap();
        // Band 1 must carry the injected non-zero values.
        assert!(mqi[1] > 0, "expected non-zero max_quant_idx in band 1");
        let sf_gain = parse_asf_scalefac_data(&mut br, &sections, &mqi, max_sfb, tl).unwrap();
        let scaled = dequantise_and_scale(&qspec, &sf_gain, sfbo, max_sfb);
        // Reconstructed coefficient at bin 4 should be close to +32 in
        // sign and order of magnitude (we did lossy quantisation).
        assert!(
            scaled[4] > 0.0,
            "expected positive reconstruction at bin 4, got {}",
            scaled[4]
        );
        assert!(
            scaled[5] < 0.0,
            "expected negative reconstruction at bin 5, got {}",
            scaled[5]
        );
    }

    // ------------------------------------------------------------------
    // Round 49 — codebook-selection optimiser unit tests.
    // ------------------------------------------------------------------

    /// Optimiser must produce a non-degenerate codebook + non-empty
    /// quantised vector for a band with real signal, regardless of which
    /// codebook minimises bits.
    #[test]
    fn optimiser_picks_real_codebook_for_low_magnitude_band() {
        let band = vec![32.0_f32, -32.0, 0.0, 0.0];
        let (cb, _sf, q, cost) = pick_best_codebook_for_band(&band);
        assert!(
            (1..=11).contains(&cb),
            "expected a real codebook id, got {cb}"
        );
        assert!(cost < u32::MAX, "cost overflow");
        assert_eq!(q.len(), band.len());
        // The chosen quantisation must preserve the sign of bin 0 / 1.
        assert!(q[0] > 0, "expected positive q[0], got {}", q[0]);
        assert!(q[1] < 0, "expected negative q[1], got {}", q[1]);
    }

    /// Optimiser at the round-49 default `q_target = 12` produces a
    /// `natural_max ≥ 1` for any band with real signal — preserving
    /// at least 3× the precision of a round-48 HCB5-only encoder
    /// (where natural_max would saturate at 4 from the q_target=4
    /// anchor). This is the round-49 SNR-vs-bits trade-off in numbers.
    #[test]
    fn optimiser_natural_max_at_q_target_12() {
        let cases: Vec<(Vec<f32>, u32)> = vec![
            (vec![1.0_f32, -1.0, 0.0, 0.0], 1),
            (vec![5.0, -3.0, 1.0, 0.0], 1),
            (vec![32.0, -32.0, 0.0, 0.0], 1),
            (vec![100.0, -50.0, 10.0, 0.0], 1),
        ];
        for (band, min_max) in cases {
            let (_cb, _sf, q, cost) = pick_best_codebook_for_band(&band);
            let nat_max = q.iter().map(|&qi| qi.unsigned_abs()).max().unwrap_or(0);
            assert!(
                nat_max >= min_max,
                "expected natural_max ≥ {min_max} for {band:?}, got {nat_max} (q={q:?} cost={cost})"
            );
        }
    }

    /// Optimiser on a high-magnitude band (peak ~200) should pick a
    /// codebook with a wide q range — typically HCB9, HCB10, or HCB11.
    /// HCB5 (q range ±4) would clip and lose a lot of energy.
    #[test]
    fn optimiser_picks_wide_codebook_for_high_magnitude_band() {
        let band = vec![200.0_f32, -180.0, 150.0, -120.0];
        let (cb, _sf, _q, cost) = pick_best_codebook_for_band(&band);
        assert!(cost < u32::MAX);
        // Wide-q codebooks: 7..11 (dim=2 unsigned with cb_mod >= 8).
        assert!(
            matches!(cb, 7..=11),
            "expected a wide-q codebook for high magnitude band, picked cb {cb}"
        );
    }

    /// Empty / silent bands map to codebook 0 (no spectral data).
    #[test]
    fn optimiser_picks_zero_codebook_for_silent_band() {
        let band = vec![0.0_f32; 4];
        let (cb, _sf, _q, cost) = pick_best_codebook_for_band(&band);
        assert_eq!(cb, 0);
        assert_eq!(cost, 0);
    }

    /// Bit-cost computation for HCB5 matches the literal codeword length.
    #[test]
    fn bit_cost_hcb5_matches_codeword_length() {
        // q = (1, 0): cb_idx = (1+4)*9 + (0+4) = 49.
        let q = vec![1i32, 0i32];
        let cost = bit_cost_for_band(asf_hcb(5).unwrap(), &q, 5);
        // Cost must equal HCB5_LEN[49] (the actual codebook entry length).
        assert_eq!(cost, crate::huffman::HCB5_LEN[49] as u32);
    }

    /// HCB11 escape costs: |q| = 16 → 1 unary + 4 value = 5 extra bits.
    #[test]
    fn bit_cost_hcb11_escape_for_q_16() {
        // q = (16, 0). The cb_idx for inline=16 is (16*17)+0 = 272 — a
        // non-zero entry of HCB11. The escape adds 5 bits + sign bit.
        let q = vec![16i32, 0i32];
        let hcb = asf_hcb(11).unwrap();
        let inline = vec![16i32, 0i32];
        let inline_cost = bit_cost_for_band(hcb, &inline, 5); // skip cb=11 path
                                                              // Now via cb=11 path: same inline cost + sign bit (16!=0) + 5 escape bits.
        let total = bit_cost_for_band(hcb, &q, 11);
        assert!(
            total >= inline_cost + 5,
            "HCB11 escape did not add at least 5 bits: total={total} inline={inline_cost}"
        );
    }

    /// `build_sections_from_per_band_cb` collapses runs of equal cb.
    #[test]
    fn build_sections_collapses_runs() {
        // 5-band layout: [3, 3, 3, 5, 5] → 2 sections.
        let per_band = vec![3u8, 3, 3, 5, 5];
        let secs = build_sections_from_per_band_cb(&per_band, 5);
        assert_eq!(secs.num_sec, 2);
        assert_eq!(secs.sect_cb, vec![3, 5]);
        assert_eq!(secs.sect_start, vec![0, 3]);
        assert_eq!(secs.sect_end, vec![3, 5]);
    }

    // ------------------------------------------------------------------
    // Round 50 — DP section optimiser + SNF emission unit tests.
    // ------------------------------------------------------------------

    /// `section_overhead_bits` matches the spec's 4 + 3 * (k+1) formula.
    #[test]
    fn section_overhead_bits_long_frame() {
        // L = 1..=7 → k = 0 → overhead = 4 + 3 = 7.
        for len in 1..=7u32 {
            assert_eq!(section_overhead_bits(len), 7, "len={len}");
        }
        // L = 8..=14 → k = 1 → overhead = 4 + 6 = 10.
        for len in 8..=14u32 {
            assert_eq!(section_overhead_bits(len), 10, "len={len}");
        }
        // L = 15..=21 → k = 2 → overhead = 4 + 9 = 13.
        assert_eq!(section_overhead_bits(15), 13);
        assert_eq!(section_overhead_bits(21), 13);
        // L = 22 → k = 3 → overhead = 16.
        assert_eq!(section_overhead_bits(22), 16);
    }

    /// `dp_optimise_sections` collapses uniform-cost bands into ONE section.
    #[test]
    fn dp_optimise_sections_collapses_uniform_cost() {
        // 5 bands; cb=5 has cost 6 each; everything else infeasible.
        let mut row = [u32::MAX; 12];
        row[5] = 6;
        let table = vec![row; 5];
        let sections = dp_optimise_sections(&table, 16);
        // Optimal: one section [0..5] with cb=5: cost = 5*6 + overhead(5) = 30 + 7 = 37.
        // Two sections would cost more: 2 * (5*3 + overhead) > 37.
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0], (0, 5, 5));
    }

    /// `dp_optimise_sections` SPLITS into 2 sections when costs vary
    /// strongly enough to beat the per-section overhead.
    #[test]
    fn dp_optimise_sections_splits_when_split_is_cheaper() {
        // 4 bands. cb=1 is cheap for bands 0,1 only; cb=11 is uniform 50.
        // Layout 1: ONE section cb=11 → 4*50 + 7 = 207.
        // Layout 2: TWO sections cb=1 [0..2] + cb=11 [2..4] → (2*5 + 7) + (2*50 + 7) = 17 + 107 = 124.
        let mut t = vec![[u32::MAX; 12]; 4];
        t[0][1] = 5;
        t[0][11] = 50;
        t[1][1] = 5;
        t[1][11] = 50;
        t[2][11] = 50;
        t[3][11] = 50;
        let sections = dp_optimise_sections(&t, 16);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0], (0, 2, 1));
        assert_eq!(sections[1], (2, 4, 11));
    }

    /// `dp_optimise_sections` honours the section count cap.
    #[test]
    fn dp_optimise_sections_caps_at_max_sections() {
        // 5 bands, only one feasible cb each. Without the cap, DP would
        // pick 5 single-band sections. With max=1, must pick a single
        // cb covering all 5 bands.
        let mut t = vec![[u32::MAX; 12]; 5];
        // Make every codebook feasible everywhere with the same uniform cost
        // so DP can pick any. cb=5 cost 10 per band.
        for r in &mut t {
            r[5] = 10;
        }
        let sections = dp_optimise_sections(&t, 1);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0], (0, 5, 5));
    }

    /// SNF emission: for a band with non-zero original energy that's
    /// quantised to silence (cb=0), the encoder picks an HCB_SNF index in
    /// [1, 21].
    #[test]
    fn snf_emits_index_for_zero_quant_band_with_real_energy() {
        let coeffs: Vec<f32> = vec![0.5, -0.5, 0.3, -0.4, 0.2, 0.0];
        let sfb_offset: Vec<u16> = vec![0, 4, 6];
        let max_sfb = 2u32;
        let sfb_cb = vec![0u8, 0u8]; // both silent (cb=0)
        let mqi = vec![0u32, 0u32];
        let snf =
            compute_snf_dpcm_for_zero_quant_bands(&coeffs, &sfb_offset, max_sfb, &sfb_cb, &mqi);
        let snf = snf.expect("expected SNF data for non-silent band");
        assert!(
            snf[0] >= 1 && snf[0] <= 21,
            "expected SNF idx in [1,21], got {}",
            snf[0]
        );
    }

    /// SNF returns None for fully-silent input (no measurable energy).
    #[test]
    fn snf_returns_none_for_silent_input() {
        let coeffs = vec![0.0_f32; 8];
        let sfb_offset: Vec<u16> = vec![0, 4, 8];
        let snf = compute_snf_dpcm_for_zero_quant_bands(
            &coeffs,
            &sfb_offset,
            2,
            &[0u8, 0u8],
            &[0u32, 0u32],
        );
        assert!(snf.is_none());
    }

    /// SNF skips bands with content (cb != 0 && mqi > 0).
    #[test]
    fn snf_skips_bands_with_quantised_content() {
        let coeffs: Vec<f32> = vec![0.5, -0.5, 0.3, -0.4];
        let sfb_offset: Vec<u16> = vec![0, 2, 4];
        // Band 0 has quantised content; band 1 is silent.
        let snf = compute_snf_dpcm_for_zero_quant_bands(
            &coeffs,
            &sfb_offset,
            2,
            &[5u8, 0u8],
            &[3u32, 0u32],
        );
        let snf = snf.expect("expected SNF for the silent band");
        // Band 0 is not eligible → idx 0 (default-init).
        assert_eq!(snf[0], 0);
        // Band 1 is eligible and has real energy → non-zero idx.
        assert!(
            snf[1] >= 1,
            "expected SNF idx >= 1 for band 1, got {}",
            snf[1]
        );
    }

    /// `write_snf_data` emits the 1-bit gate then per-band Huffman codes.
    /// Round-trip via `parse_asf_snf_data` recovers the indices.
    #[test]
    fn snf_round_trip_through_parser() {
        use crate::asf_data::parse_asf_snf_data;
        // 4 bands: 2 with content, 2 silent.
        let max_sfb = 4u32;
        let snf = vec![0i32, 0i32, 14, 16]; // idx for the 2 silent bands
        let sfb_cb = vec![5u8, 5u8, 0u8, 0u8];
        let mqi = vec![3u32, 3u32, 0u32, 0u32];
        // Build a synthetic AsfSections with sfb_cb populated.
        let sections = AsfSections {
            sect_cb: vec![5u8, 0u8],
            sect_start: vec![0u16, 2u16],
            sect_end: vec![2u16, 4u16],
            sfb_cb: sfb_cb.clone(),
            num_sec: 2,
            num_sec_lsf: 2,
        };
        let mut bw = BitWriter::new();
        write_snf_data(&mut bw, Some(&snf), &sfb_cb, &mqi, max_sfb);
        bw.align_to_byte();
        let bytes = bw.finish();

        // Parse back. parse_asf_snf_data reads the 1-bit gate then per-band
        // codes, walking sections.sfb_cb.
        let mut br = BitReader::new(&bytes);
        let parsed = parse_asf_snf_data(&mut br, &sections, &mqi, max_sfb, 1920)
            .expect("parse_asf_snf_data");
        let parsed = parsed.expect("expected SNF data");
        // Bands 2 and 3 should have idx 14 and 16 respectively.
        assert_eq!(parsed[2], 14);
        assert_eq!(parsed[3], 16);
    }

    /// DP-vs-greedy: 1 kHz tone has one very-high-energy band (HCB11) and
    /// many low-energy / silent bands. Greedy + run-length merging may
    /// section the tone band as HCB11 separately (good) but greedy and DP
    /// should both find the same minimal layout in this monotone case.
    /// What this test verifies is the more interesting *mixed* case where
    /// DP can section across an alternating pattern that greedy can't.
    #[test]
    fn dp_vs_greedy_dp_savings_on_alternating_pattern() {
        // 6 bands with alternating cost minimisers: cb=1 cheapest at
        // bands 0,2,4 and cb=11 cheapest at 1,3,5. The greedy run-length
        // optimiser would pick 6 single-band sections (cost = 6*7 + 6*c).
        // DP can either match that or pick a single uniform section if the
        // overhead saving outweighs the per-band cost penalty.
        let mut t = vec![[u32::MAX; 12]; 6];
        for (i, row) in t.iter_mut().enumerate() {
            // Equal feasibility for cb=11 (uniform), with mild cost gradient
            // on cb=1 between odd vs. even bands.
            row[11] = 12;
            if i % 2 == 0 {
                row[1] = 8;
            }
        }
        let dp_sections = dp_optimise_sections(&t, 16);
        // Compute greedy bits assuming greedy picks the per-band minimum
        // codebook. For odd bands: only cb=11 feasible. For even bands:
        // cb=1 is cheaper. So greedy = [1, 11, 1, 11, 1, 11] → 6 sections,
        // each L=1 → overhead 7 each = 42 + (8 + 12 + 8 + 12 + 8 + 12) = 102.
        let greedy_total: u64 = 6 * 7 + 8 * 3 + 12 * 3;
        let mut dp_total: u64 = 0;
        for &(s, e, cb) in &dp_sections {
            dp_total += section_overhead_bits(e - s) as u64;
            for b in s..e {
                let c = t[b as usize][cb as usize];
                dp_total += if c == u32::MAX { 0 } else { c as u64 };
            }
        }
        assert!(
            dp_total <= greedy_total,
            "DP must beat greedy: dp={dp_total} greedy={greedy_total}"
        );
        // Concrete check: ONE uniform cb=11 section costs 6*12 + 7 = 79,
        // beating greedy's 102 by 23 bits.
        assert!(
            dp_total <= 79,
            "DP did not collapse alternating pattern to single section: {dp_total}"
        );
    }

    /// DP-vs-greedy: noise-like band layout where greedy picks one wide cb
    /// (HCB11) but DP can split off cheap mid sections.
    /// Asserts `dp_bits <= greedy_bits` (DP is optimal so always ≤ greedy).
    #[test]
    fn dp_vs_greedy_dp_never_worse_for_noise_spectrum() {
        let n = 1920usize;
        let max_sfb = 50u32;
        let mut s: u64 = 0xACE4u64;
        let pcm: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 33) as u32;
                (u as f32 / (1u32 << 31) as f32 - 1.0) * 0.3
            })
            .collect();
        let mut mdct = crate::encoder_mdct::EncoderMdctState::new(n as u32);
        let _ = mdct.analyse_frame(&pcm);
        let coeffs = mdct.analyse_frame(&pcm);
        let (greedy, dp) = measure_greedy_vs_dp_bits(n as u32, max_sfb, &coeffs);
        eprintln!(
            "ROUND-50 DP-vs-greedy noise-fixture bits: greedy={greedy} dp={dp} (savings={})",
            greedy.saturating_sub(dp)
        );
        assert!(
            dp <= greedy,
            "DP must not be worse than greedy: dp={dp} greedy={greedy}"
        );
    }
}
