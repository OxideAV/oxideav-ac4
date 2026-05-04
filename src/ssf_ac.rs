//! AC-4 Speech Spectral Frontend (SSF) arithmetic decoder.
//!
//! Implements ETSI TS 103 190-1 V1.4.1 §5.2.8 — the binary arithmetic
//! decoder used inside `ssf_ac_data()` (Table 46) to decode envelope
//! quantisation indices, predictor-gain indices, and quantised MDCT
//! coefficient indices for an SSF granule.
//!
//! Pseudocodes implemented:
//!
//! * Pseudocode 41 — `SSF_MODEL_BITS` / `SSF_RANGE_BITS` /
//!   `SSF_OFFSET_BITS` constants.
//! * Pseudocode 42 — [`AcState`] (the `ac_state` struct).
//! * Pseudocode 43 — [`AcState::init`] (`AcDecoderInit()`).
//! * Pseudocode 44 — [`AcState::decode_symbol_ext_cdf`]
//!   (`AcDecodeSymbolExtCdf()`).
//! * Pseudocode 45 — [`AcState::decode_target`] (`AcDecodeTarget()`).
//! * Pseudocode 46 — [`AcState::decode`] (`AcDecode()`).
//! * Pseudocode 47 — [`AcState::decode_finish`] (`AcDecodeFinish()`).
//! * Pseudocode 48 — [`decode_envelope_indices`].
//! * Pseudocode 49 — [`decode_predictor_gain`].
//! * Pseudocode 50 — [`decode_coefficient_indices`].
//! * Pseudocode 51 — [`AcState::decode_symbol_calc_cdf`] (CDF
//!   computation path for transform-coefficient symbols).
//! * Pseudocode 52 — [`idx_to_reconstruction`] (`Idx2Reconstruction()`).
//! * Pseudocode 53 — [`cdf_est`] (`CdfEst()`).
//! * Pseudocodes 54-57 — [`SsfRandGenState`] / [`SsfRandGenState::reset`] /
//!   [`SsfRandGenState::dither_value`] /
//!   [`SsfRandGenState::random_noise_value`].
//!
//! All tables are sourced from [`crate::ssf_tables`]:
//! [`ENVELOPE_CDF_LUT`], [`PREDICTOR_GAIN_CDF_LUT`], [`CDF_TABLE`],
//! [`STEP_SIZES_Q4_15`], [`AC_COEFF_MAX_INDEX`], [`DITHER_TABLE`],
//! [`RANDOM_NOISE_TABLE`].
//!
//! The arithmetic decoder is initialized once per SSF granule when
//! parsing reaches `ssf_ac_data()`. After all envelope, predictor-gain
//! and coefficient symbols have been decoded the caller invokes
//! [`AcState::decode_finish`] which returns the total number of
//! arithmetic-decoded bits so the bitstream cursor can be aligned.
//!
//! This is a clean-room transcription of the spec pseudocode — no
//! reference encoder or external decoder source consulted.

use oxideav_core::bits::BitReader;

use crate::ssf_tables::{
    AC_COEFF_MAX_INDEX, CDF_TABLE, ENVELOPE_CDF_LUT, PREDICTOR_GAIN_CDF_LUT, STEP_SIZES_Q4_15,
};

// === Pseudocode 41: AC decoder constants =====================================

/// Number of model bits (`SSF_MODEL_BITS`). All CDFs are Q0.15.
pub const SSF_MODEL_BITS: u32 = 15;

/// Model unit (`SSF_MODEL_UNIT = 1 << SSF_MODEL_BITS`).
pub const SSF_MODEL_UNIT: u32 = 1u32 << SSF_MODEL_BITS;

/// Number of range bits (`SSF_RANGE_BITS`).
pub const SSF_RANGE_BITS: u32 = 30;

/// Half of the range unit (`SSF_THRESHOLD_LARGE = 1 << (SSF_RANGE_BITS-1)`).
pub const SSF_THRESHOLD_LARGE: u32 = 1u32 << (SSF_RANGE_BITS - 1);

/// Quarter of the range unit (`SSF_THRESHOLD_SMALL = 1 << (SSF_RANGE_BITS-2)`).
pub const SSF_THRESHOLD_SMALL: u32 = 1u32 << (SSF_RANGE_BITS - 2);

/// Offset bits (`SSF_OFFSET_BITS`).
pub const SSF_OFFSET_BITS: u32 = 14;

// === Pseudocode 42: ac_state =================================================

/// State of the SSF arithmetic decoder.
///
/// Mirrors `ac_state` from Pseudocode 42. The bitstream itself is not
/// owned by the state; each [`AcState`] method takes a `&mut BitReader<'_>`
/// argument so the caller controls the byte source. `bits_consumed`
/// tracks the absolute bitstream-bit count read so far so
/// [`AcState::decode_finish`] can compute termination bits exactly as in
/// Pseudocode 47.
#[derive(Debug, Clone)]
pub struct AcState {
    /// `uiLow`.
    pub low: u32,
    /// `uiRange`.
    pub range: u32,
    /// `uiOffset`.
    pub offset: u32,
    /// `uiOffset2`.
    pub offset2: u32,
    /// Total bitstream bits read so far through this state.
    pub bits_consumed: u32,
}

impl AcState {
    /// Pseudocode 43: `AcDecoderInit()` — read the first
    /// `SSF_RANGE_BITS` bits from the bitstream into `uiOffset`.
    pub fn init(br: &mut BitReader<'_>) -> Result<Self, AcError> {
        let mut s = AcState {
            low: 0,
            range: SSF_THRESHOLD_LARGE,
            offset: 0,
            offset2: 0,
            bits_consumed: 0,
        };
        // Read SSF_RANGE_BITS bits MSB-first.
        let first = br.read_bit().map_err(|_| AcError::BitstreamUnderflow)?;
        s.offset = u32::from(first);
        s.bits_consumed = 1;
        for _ in 1..SSF_RANGE_BITS {
            let b = br.read_bit().map_err(|_| AcError::BitstreamUnderflow)?;
            s.offset = (s.offset << 1) | u32::from(b);
            s.bits_consumed += 1;
        }
        s.offset2 = s.offset;
        Ok(s)
    }

    /// Pseudocode 45: `AcDecodeTarget()` — compute `uiTarget` for the
    /// next symbol.
    pub fn decode_target(&self) -> u32 {
        let range = self.range >> SSF_MODEL_BITS;
        let tmp = 1u32 << SSF_OFFSET_BITS;
        let num_shifts = if range < tmp {
            SSF_MODEL_BITS
        } else {
            SSF_MODEL_BITS - 1
        };
        let mut num = self.offset;
        let den = range << num_shifts;
        let mut target: u32 = 0;
        let mut idx = num_shifts;
        while idx > 0 {
            if num >= den {
                num = num.wrapping_sub(den);
                target = target.wrapping_add(1);
            }
            num = num.wrapping_shl(1);
            target = target.wrapping_shl(1);
            idx -= 1;
        }
        if num >= den {
            num = num.wrapping_sub(den);
            target = target.wrapping_add(1);
        }
        let _ = num;
        if target >= SSF_MODEL_UNIT {
            target = SSF_MODEL_UNIT - 1;
        }
        target
    }

    /// Pseudocode 46: `AcDecode()` — advance the arithmetic decoder
    /// after a symbol has been chosen. Reads zero or more bits from
    /// the bitstream as needed to renormalise `uiRange`.
    pub fn decode(
        &mut self,
        cdf_low: u32,
        cdf_high: u32,
        br: &mut BitReader<'_>,
    ) -> Result<(), AcError> {
        let range = self.range >> SSF_MODEL_BITS;
        let tmp1 = range.wrapping_mul(cdf_low);
        self.offset = self.offset.wrapping_sub(tmp1);
        if cdf_high < SSF_MODEL_UNIT {
            let tmp2 = cdf_high.wrapping_sub(cdf_low);
            self.range = range.wrapping_mul(tmp2);
        } else {
            self.range = self.range.wrapping_sub(tmp1);
        }
        // Renormalise.
        while self.range <= SSF_THRESHOLD_SMALL {
            let bit = br.read_bit().map_err(|_| AcError::BitstreamUnderflow)?;
            self.bits_consumed += 1;
            self.range = self.range.wrapping_shl(1);
            self.offset = self.offset.wrapping_shl(1);
            self.offset = self.offset.wrapping_add(u32::from(bit));
            self.offset2 = self.offset2.wrapping_shl(1);
            if self.offset & 1 != 0 {
                self.offset2 = self.offset2.wrapping_add(1);
            }
        }
        Ok(())
    }

    /// Pseudocode 44: `AcDecodeSymbolExtCdf()` — decode one symbol
    /// using a static CDF table, returning the chosen symbol index in
    /// `[min_symbol, max_symbol]`. The table is indexed by
    /// `(symbol_idx - min_symbol)` and must therefore be at least
    /// `max_symbol - min_symbol + 2` entries long.
    pub fn decode_symbol_ext_cdf(
        &mut self,
        cdf: &[u32],
        min_symbol: i32,
        max_symbol: i32,
        br: &mut BitReader<'_>,
    ) -> Result<i32, AcError> {
        let target = self.decode_target();
        for sym in min_symbol..=max_symbol {
            let idx = (sym - min_symbol) as usize;
            let cdf_low = cdf[idx];
            let cdf_high = cdf[idx + 1];
            if target < cdf_high && target >= cdf_low {
                self.decode(cdf_low, cdf_high, br)?;
                return Ok(sym);
            }
        }
        Err(AcError::SymbolNotFound)
    }

    /// Pseudocode 44 + 51: decode one transform-coefficient symbol
    /// using the *computed* CDF (`CdfEst` of `Idx2Reconstruction`'s
    /// half-step neighbours). `i_step_size` and `i_dither_val` are the
    /// per-band step size and per-bin dither value from §5.2.8.3.
    pub fn decode_symbol_calc_cdf(
        &mut self,
        i_step_size: i32,
        i_dither_val: i32,
        max_idx: i32,
        br: &mut BitReader<'_>,
    ) -> Result<i32, AcError> {
        let target = self.decode_target();
        for sym in 0..=max_idx {
            let (cdf_low, cdf_high) = compute_coeff_cdf(sym, i_step_size, i_dither_val);
            if target < cdf_high && target >= cdf_low {
                self.decode(cdf_low, cdf_high, br)?;
                return Ok(sym);
            }
        }
        Err(AcError::SymbolNotFound)
    }

    /// Pseudocode 47: `AcDecodeFinish()` — number of bits the
    /// arithmetic decoder consumed from the bitstream including the
    /// termination bits. Caller subtracts this from the actual byte
    /// position to determine how many overhead bits to skip.
    pub fn decode_finish(&mut self) -> u32 {
        let mut res = self.bits_consumed.saturating_sub(SSF_RANGE_BITS);
        // Determine the termination bits.
        self.low = self.offset2 & (SSF_THRESHOLD_LARGE - 1);
        let tmp1 = SSF_THRESHOLD_LARGE.wrapping_sub(self.offset);
        self.low = self.low.wrapping_add(tmp1);
        let mut chosen_bit_idx: u32 = 0;
        for bit_idx in 1..=SSF_RANGE_BITS {
            let rev_idx = SSF_RANGE_BITS - bit_idx;
            let mut const_up_fact = 1u32 << rev_idx;
            const_up_fact = const_up_fact.wrapping_sub(1);
            let tmp1 = self.low.wrapping_add(const_up_fact);
            let bits = tmp1 >> rev_idx;
            let val = bits.wrapping_shl(rev_idx);
            let tmp1b = val.wrapping_add(const_up_fact);
            let mut tmp2 = self.range.wrapping_sub(1);
            tmp2 = tmp2.wrapping_add(self.low);
            if self.low <= val && tmp1b <= tmp2 {
                chosen_bit_idx = bit_idx;
                break;
            }
        }
        res = res.wrapping_add(chosen_bit_idx);
        res
    }
}

/// Errors raised by the SSF arithmetic decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcError {
    /// Ran out of bits while pulling more from the bitstream.
    BitstreamUnderflow,
    /// `AcDecodeSymbolExtCdf()` could not locate the target value
    /// within the supplied CDF range. Indicates either a malformed
    /// bitstream or an off-by-one in the CDF table.
    SymbolNotFound,
}

// === Pseudocode 48: envelope decoder =========================================

/// Pseudocode 48: decode `num_bands - 1` envelope quantization indices
/// using [`ENVELOPE_CDF_LUT`]. The first index is *not* arithmetic-
/// coded — it comes from `env_curr_band0_bits` in `ssf_st_data()`
/// (Table 45), so the caller fills `out[0]` and we fill `out[1..]`.
pub fn decode_envelope_indices(
    state: &mut AcState,
    num_bands: usize,
    out: &mut [i32],
    br: &mut BitReader<'_>,
) -> Result<(), AcError> {
    if out.len() < num_bands {
        return Err(AcError::SymbolNotFound);
    }
    let cdf: Vec<u32> = ENVELOPE_CDF_LUT.iter().map(|&x| x as u32).collect();
    // Pseudocode 48 picks one symbol per (idx = 1..num_bands) using a
    // 33-entry CDF (range 0..=32). The previous range `0..=31` capped
    // the symbol space one short.
    for slot in out.iter_mut().take(num_bands).skip(1) {
        *slot = state.decode_symbol_ext_cdf(&cdf, 0, 32, br)?;
    }
    Ok(())
}

// === Pseudocode 49: predictor-gain decoder ===================================

/// Pseudocode 49: `arithmetic_decode_pred()` — decode a single
/// predictor-gain index in `0..=32` using [`PREDICTOR_GAIN_CDF_LUT`]
/// (33 entries gives 32 symbol slots with one trailing CDF terminator).
pub fn decode_predictor_gain(state: &mut AcState, br: &mut BitReader<'_>) -> Result<i32, AcError> {
    state.decode_symbol_ext_cdf(&PREDICTOR_GAIN_CDF_LUT, 0, 32, br)
}

// === Pseudocode 50: coefficient index decoder ================================

/// Pseudocode 50: `arithmetic_decode_coeffs()` — decode the quantised
/// MDCT coefficient indices for one block. For each band the caller
/// supplies the allocation table entry; bands with `i_alloc == 0`
/// emit zero indices, others pull `(end_bin - start_bin + 1)` symbols
/// from the arithmetic decoder using the computed-CDF path.
pub fn decode_coefficient_indices(
    state: &mut AcState,
    i_alloc_table: &[u32],
    bands: &[(usize, usize)], // (start_bin, end_bin) — inclusive
    i_dither: &[i32],
    out: &mut [i32],
    br: &mut BitReader<'_>,
) -> Result<(), AcError> {
    for (band_idx, &i_alloc) in i_alloc_table.iter().enumerate() {
        let (start_bin, end_bin) = bands[band_idx];
        if i_alloc == 0 {
            for slot in out.iter_mut().take(end_bin + 1).skip(start_bin) {
                *slot = 0;
            }
        } else {
            let i_alloc = i_alloc as usize;
            let i_step_size = STEP_SIZES_Q4_15[i_alloc];
            let i_max_idx = (AC_COEFF_MAX_INDEX[i_alloc] as i32) + 1;
            for bin in start_bin..=end_bin {
                let i_dither_val = i_dither[bin];
                out[bin] =
                    state.decode_symbol_calc_cdf(i_step_size, i_dither_val, i_max_idx, br)?;
            }
        }
    }
    Ok(())
}

// === Pseudocode 51 + 52 + 53: CDF computation for coefficients ==============

/// Pseudocode 52: `Idx2Reconstruction()` — convert a quantization
/// index into a reconstruction value (Q?.15) given the dither value
/// and step size.
pub fn idx_to_reconstruction(i_index: i32, i_dither_value: i32, i_step_size: i32) -> i32 {
    // subtract the dither
    let i_tmp1 = i_index.wrapping_shl(15);
    let i_reconstruction = i_tmp1.wrapping_sub(i_dither_value);
    // multiply times the step size
    let i_tmp2 = i_reconstruction >> 15; // sign-preserving
    let i_tmp1 = i_tmp2.wrapping_shl(15);
    let mut i_tmp1 = i_reconstruction.wrapping_sub(i_tmp1);
    i_tmp1 >>= 3; // sign-preserving
    let mut recon = i_tmp1.wrapping_mul(i_step_size);
    recon >>= 12; // sign-preserving
    let i_tmp1 = i_tmp2.wrapping_mul(i_step_size);
    recon = recon.wrapping_add(i_tmp1);
    recon
}

/// Pseudocode 53: `CdfEst()` — read a CDF value from [`CDF_TABLE`]
/// given a reconstruction value `iInVal` (Q4.15 → Q.0 via shift-right
/// 10, then offset by +352 to land in `[0, 704]`).
pub fn cdf_est(i_in_val: i32) -> u32 {
    let i_idx = (i_in_val >> 10) + 352;
    // Clamp defensively — well-formed inputs land within [0, 704].
    let i_idx = i_idx.clamp(0, (CDF_TABLE.len() as i32) - 1);
    CDF_TABLE[i_idx as usize] as u32
}

/// Pseudocode 51: derive `(uiCdfLow, uiCdfHigh)` for one coefficient
/// symbol from `iSymbolIdx`, `i_step_size`, `i_dither_val`.
pub fn compute_coeff_cdf(i_symbol_idx: i32, i_step_size: i32, i_dither_val: i32) -> (u32, u32) {
    const I_MAX_VALUE: i32 = 327_680;
    let i_midpoint = idx_to_reconstruction(i_symbol_idx, i_dither_val, i_step_size);
    let i_half_step_size = i_step_size >> 1;
    let mut i_left = i_midpoint.wrapping_sub(i_half_step_size);
    let mut i_right = i_left.wrapping_add(i_step_size);
    if i_left < -I_MAX_VALUE {
        i_left = -I_MAX_VALUE;
    }
    if i_right > I_MAX_VALUE {
        i_right = I_MAX_VALUE;
    }
    let cdf_low = cdf_est(i_left);
    let cdf_high = cdf_est(i_right);
    (cdf_low, cdf_high)
}

// === Pseudocode 54-57: random number generator ==============================

/// SSF random number generator state (`ssf_rndgen_state` from
/// Pseudocode 54). Two instances are spun up per granule — one for the
/// dither sequence, one for the noise sequence. All four counters are
/// `u8` (implicit modulo 256).
#[derive(Debug, Clone, Default)]
pub struct SsfRandGenState {
    pub offset_a: u8,
    pub offset_b: u8,
    pub state_idx: u8,
    pub current_idx: u8,
}

impl SsfRandGenState {
    /// Pseudocode 55: `ResetRandGenState()` — initialize / reset the
    /// random number generator at the beginning of an SSF-I-frame.
    pub fn reset(&mut self) {
        self.offset_a = 0;
        self.offset_b = 0;
        self.state_idx = 1;
        self.current_idx = 0;
    }

    fn step(&mut self) {
        // Pseudocode 56/57 state-update tail. C source has `psS->uiStateIdx
        // = psS->uiStateIdx++;` which (post-increment-assigned-back) is
        // a documented quirk that effectively leaves state_idx unchanged
        // before adding offset_b + offset_a below. Same shape inside
        // the if-block for `current_idx`. We elide both no-op
        // self-assignments here.
        self.offset_a = self.offset_a.wrapping_add(1);
        if self.offset_a == 255 {
            self.offset_b = self.offset_b.wrapping_add(1);
            self.offset_a = 0;
        }
        self.current_idx = self.current_idx.wrapping_add(self.offset_a);
        self.state_idx = self
            .state_idx
            .wrapping_add(self.offset_b)
            .wrapping_add(self.offset_a);
    }

    /// Pseudocode 56: `GetDitherValue()` — pull one dither sample
    /// from [`crate::ssf_tables::DITHER_TABLE`] then advance state.
    pub fn dither_value(&mut self) -> i32 {
        let res = crate::ssf_tables::DITHER_TABLE[self.current_idx as usize];
        self.step();
        res
    }

    /// Pseudocode 57: `GetRandomNoiseValue()` — sum of two
    /// [`crate::ssf_tables::RANDOM_NOISE_TABLE`] entries (current +
    /// state) then advance state.
    pub fn random_noise_value(&mut self) -> f32 {
        let f_res1 = crate::ssf_tables::RANDOM_NOISE_TABLE[self.current_idx as usize];
        let f_res2 = crate::ssf_tables::RANDOM_NOISE_TABLE[self.state_idx as usize];
        let res = f_res1 + f_res2;
        self.step();
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: constants line up with Pseudocode 41.
    #[test]
    fn constants_match_pseudocode_41() {
        assert_eq!(SSF_MODEL_BITS, 15);
        assert_eq!(SSF_MODEL_UNIT, 0x8000);
        assert_eq!(SSF_RANGE_BITS, 30);
        assert_eq!(SSF_THRESHOLD_LARGE, 1u32 << 29);
        assert_eq!(SSF_THRESHOLD_SMALL, 1u32 << 28);
        assert_eq!(SSF_OFFSET_BITS, 14);
    }

    /// Pseudocode 43: after `init`, `low == 0`, `range == THRESHOLD_LARGE`,
    /// `offset == offset2`, and exactly `SSF_RANGE_BITS` bits have been
    /// pulled.
    #[test]
    fn ac_state_init_pulls_range_bits() {
        // Pattern: 30 bits, alternating 1010...
        let mut bytes = vec![0u8; 4];
        // Construct the first 30 bits = 1010 1010 1010 1010 1010 1010 1010 10
        // Pack MSB-first.
        // 32-bit alternating-1010 pattern, top 30 bits used.
        let pattern: u32 = 0xAAAA_AAAA;
        bytes[0] = (pattern >> 24) as u8;
        bytes[1] = (pattern >> 16) as u8;
        bytes[2] = (pattern >> 8) as u8;
        bytes[3] = pattern as u8;
        let mut br = BitReader::new(&bytes);
        let s = AcState::init(&mut br).expect("init OK");
        assert_eq!(s.low, 0);
        assert_eq!(s.range, SSF_THRESHOLD_LARGE);
        assert_eq!(s.bits_consumed, SSF_RANGE_BITS);
        assert_eq!(s.offset, s.offset2);
        // First 30 bits of the alternating pattern produce a known offset.
        let mut expected: u32 = 0;
        for i in 0..30 {
            let bit = (pattern >> (31 - i)) & 1;
            expected = (expected << 1) | bit;
        }
        assert_eq!(s.offset, expected);
    }

    /// Pseudocode 53: `CdfEst(0)` should land at index 352 of [`CDF_TABLE`].
    #[test]
    fn cdf_est_centre_index() {
        let centre = cdf_est(0);
        assert_eq!(centre, CDF_TABLE[352] as u32);
    }

    /// Pseudocode 53: `CdfEst()` is monotone non-decreasing in `iInVal`.
    #[test]
    fn cdf_est_monotone() {
        let samples: Vec<u32> = (-100_000..=100_000).step_by(2_000).map(cdf_est).collect();
        for w in samples.windows(2) {
            assert!(w[0] <= w[1], "CdfEst not monotone: {} > {}", w[0], w[1]);
        }
    }

    /// Pseudocode 55: `reset()` zeros offsets and primes state_idx = 1.
    #[test]
    fn rng_reset_initial_state() {
        let mut g = SsfRandGenState {
            offset_a: 7,
            offset_b: 3,
            state_idx: 9,
            current_idx: 11,
        };
        g.reset();
        assert_eq!(g.offset_a, 0);
        assert_eq!(g.offset_b, 0);
        assert_eq!(g.state_idx, 1);
        assert_eq!(g.current_idx, 0);
    }

    /// Pseudocode 56: first dither sample is `DITHER_TABLE[0]` =
    /// `0x3200` (anchor from Annex C.10).
    #[test]
    fn rng_first_dither_value_is_table_zero() {
        let mut g = SsfRandGenState::default();
        g.reset();
        let v = g.dither_value();
        assert_eq!(v, 0x3200);
    }

    /// Pseudocode 56: pulling 256 dither values does not panic and
    /// covers a wide range. Statistically the resulting set should
    /// span much of the Q0.15 range.
    #[test]
    fn rng_dither_pull_256() {
        let mut g = SsfRandGenState::default();
        g.reset();
        let mut samples = Vec::with_capacity(256);
        for _ in 0..256 {
            samples.push(g.dither_value());
        }
        let min = *samples.iter().min().unwrap();
        let max = *samples.iter().max().unwrap();
        assert!(min < 0x1000, "range too narrow at the bottom: min={}", min);
        assert!(max > 0x7000, "range too narrow at the top: max={}", max);
    }

    /// Pseudocode 57: random-noise samples sum to roughly zero mean.
    #[test]
    fn rng_noise_zero_mean() {
        let mut g = SsfRandGenState::default();
        g.reset();
        let mut sum = 0.0_f64;
        const N: usize = 1024;
        for _ in 0..N {
            sum += g.random_noise_value() as f64;
        }
        let mean = sum / N as f64;
        assert!(mean.abs() < 0.5, "noise mean = {} too far from 0", mean);
    }

    /// Pseudocode 52: zero index + zero dither + nonzero step = 0.
    #[test]
    fn idx2recon_zero() {
        assert_eq!(idx_to_reconstruction(0, 0, 0x8000), 0);
    }

    /// Pseudocode 52: positive index moves reconstruction up.
    #[test]
    fn idx2recon_monotone_in_index() {
        let r0 = idx_to_reconstruction(0, 0, 0x100);
        let r1 = idx_to_reconstruction(1, 0, 0x100);
        let r2 = idx_to_reconstruction(2, 0, 0x100);
        assert!(
            r0 < r1 && r1 < r2,
            "expected monotone, got {r0}, {r1}, {r2}"
        );
    }

    /// Pseudocode 51: in the *non-saturated* region the CDF interval
    /// `[cdf_low, cdf_high)` is well-formed (`low <= high`). Outside
    /// the `±I_MAX_VALUE = 327680` reconstruction window the spec only
    /// clamps `iLeft` on the negative side and `iRight` on the positive
    /// side, so symbols whose reconstruction overflows produce a
    /// degenerate (zero-probability) interval; those are correctly
    /// skipped by the decoder loop because `target < cdf_high &&
    /// target >= cdf_low` is unsatisfiable.
    #[test]
    fn coeff_cdf_low_le_high_in_non_saturated_region() {
        let dither = 0x3200;
        for sym in 0..=2 {
            for &alloc in [5usize, 10, 15, 20].iter() {
                let step = STEP_SIZES_Q4_15[alloc];
                let (lo, hi) = compute_coeff_cdf(sym, step, dither);
                assert!(lo <= hi, "sym={sym} alloc={alloc} lo={lo} hi={hi}");
            }
        }
    }

    /// Round-trip: feed `AcDecode` a CDF (low=0 high=MODEL_UNIT) which
    /// is the all-symbol-degenerate case and verify it doesn't crash
    /// the renormalisation loop. Useful as a smoke test for the
    /// arithmetic core.
    #[test]
    fn ac_decode_full_range_does_not_loop_forever() {
        // 64 bits of payload, all zeros after the initial 30 init bits.
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let mut s = AcState::init(&mut br).expect("init OK");
        // CDF [0, MODEL_UNIT] => decoder picks the only symbol and
        // renormalises until range > THRESHOLD_SMALL.
        s.decode(0, SSF_MODEL_UNIT, &mut br).expect("decode OK");
        // Range should now be > THRESHOLD_SMALL.
        assert!(s.range > SSF_THRESHOLD_SMALL);
    }
}
