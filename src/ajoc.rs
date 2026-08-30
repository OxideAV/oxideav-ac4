//! A-JOC (Advanced Joint Object Coding) parameter processing —
//! ETSI TS 103 190-2 §5.7 + §6.2.5 / §6.3.6.
//!
//! A-JOC reconstructs `num_umx_signals` output audio objects from a
//! smaller set of `num_dmx_signals` jointly-coded downmix objects plus
//! a bank of `ajoc_num_decorr` decorrelators, using two time-variant,
//! frequency-dependent reconstruction submatrices: a *dry* matrix
//! factoring the downmix inputs onto the outputs and a *wet* matrix
//! factoring the decorrelator outputs onto the outputs (§5.7.3.6.1).
//!
//! What this module lands today (all of it normative, bit-exact, and
//! independent of the per-codeword Huffman tables):
//!
//! * **Parameter band ↔ QMF subband mapping** (§5.7.3.1 Table 42) —
//!   [`sb_to_pb`] maps each of the 64 QMF subbands to an A-JOC
//!   parameter band for all eight `ajoc_num_bands` configurations
//!   (23 / 15 / 12 / 9 / 7 / 5 / 3 / 1), and [`num_bands_from_code`]
//!   resolves `ajoc_num_bands_code` per Table 100.
//!
//! * **Differential decoding** (§5.7.3.2 Table 43) —
//!   [`differential_decode_dry`] / [`differential_decode_wet`] recover
//!   the absolute quantised parameter arrays `mtx_dry_q` / `mtx_wet_q`
//!   from the Huffman-decoded deltas, covering DIFF_FREQ (running
//!   modulo-`nquant` sum from band 0), DIFF_TIME (per-band add to the
//!   previous frame's quantised values), and the `ajoc_sparse_select`
//!   "absent" default (`(nquant - 1) / 2`, i.e. the zero-centre index).
//!   The `mtx_*_q_prev` state is carried via [`AjocDiffState`].
//!
//! * **Dequantization** (§5.7.3.3 Tables 44-47) —
//!   [`dequantize_dry`] / [`dequantize_wet`] lift the integer indices
//!   to real-valued matrix coefficients. The four quantizers are
//!   uniform: dry coarse spans ±5,0048828 over 51 levels, dry fine
//!   over 101, wet coarse ±2,001953125 over 21, wet fine over 41. The
//!   closed-form steps are validated bit-exactly against the documented
//!   table endpoints in the unit tests.
//!
//! * **Time interpolation** (§5.7.3.4 Table 48) —
//!   [`AjocInterpolator`] implements `ajoc_interpolate()`: a linear ramp
//!   of `ajoc_ramp_len` QMF time slots that re-arms its delta increment
//!   whenever `ts == ajoc_start_pos[dp]`, supporting interpolation over
//!   multiple frame boundaries (max ramp 64 slots).
//!
//! * **Signal reconstruction** (§5.7.3.6 Table 49, dry path) —
//!   [`reconstruct_dry`] forms `z[o] = Σ_ch mtx_dry[o][ch] · x[ch]`
//!   over the QMF subband / time-slot grid, honouring
//!   `ajoc_object_present[o]`. The decorrelator (wet) contribution is
//!   structured in [`reconstruct`] and gated behind the supplied
//!   decorrelator output matrices.
//!
//! * **Full spatial reconstruction** (§5.7.3.5 + §5.7.3.6 Table 49) —
//!   [`ajoc_reconstruct`] is the complete Table 49 driver: it accumulates
//!   the §5.7.3.6.2 decorrelation-input pre-matrix
//!   `D[de][ch] = Σ_o |mtx_wet_dq[o][de]| · mtx_dry_dq[o][ch]`
//!   ([`pre_matrix_param`]), then walks the `(ts, sb)` grid interpolating
//!   the dry / wet / pre tracks (per-track [`AjocInterpolator`] carried
//!   across frames in [`AjocReconState`]), forms the decorrelator inputs
//!   `u = pre · x`, decorrelates them with the §5.7.3.5 cyclic-0,2,1
//!   decorrelator bank (reusing the part-1 §5.7.7.4.2
//!   [`crate::acpl_synth::InputSignalModifier`]), and sums the dry
//!   (`x · mtx_dry`) + wet (`y · mtx_wet`) contributions into the
//!   reconstructed output objects `z[ts][sb][o]`. This closes the A-JOC
//!   decode chain from dequantized matrices to output QMF objects.
//!
//! Config-layer bitstream parsers ([`parse_ajoc_ctrl_info`],
//! [`parse_ajoc_data_point_info`], [`parse_ajoc_bed_info`],
//! [`parse_ajoc_dmx_de_data`], [`AjocCtrlInfo`]) walk the §6.2.5
//! side-information that does not depend on the per-codeword Huffman
//! codebooks.
//!
//! The per-codeword `ajoc_huff_data()` decode (§6.2.5.5) lives in
//! [`crate::ajoc_huffman`] over the Annex A.1.1 `AJOC_HCB_*` codebooks
//! (formerly a docs gap; the numeric arrays landed from the ETSI Part 2
//! electronic-attachment table file). Its `(diff_type, a_huff_data)`
//! output drops straight into the differential decoder below.

#![allow(clippy::excessive_precision, clippy::needless_range_loop)]

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// `DRY` / `WET` selector for the two reconstruction submatrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AjocMatrixKind {
    /// Dry submatrix — factors downmix inputs onto outputs.
    Dry,
    /// Wet submatrix — factors decorrelator outputs onto outputs.
    Wet,
}

/// Quantization mode for an A-JOC reconstructed object
/// (`ajoc_quant_select`, Table 101).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AjocQuantMode {
    /// `ajoc_quant_select == 0` — fine quantization (more levels).
    Fine,
    /// `ajoc_quant_select == 1` — coarse quantization (fewer levels).
    Coarse,
}

impl AjocQuantMode {
    /// Resolve from the 1-bit `ajoc_quant_select` field.
    pub fn from_bit(b: bool) -> Self {
        if b {
            AjocQuantMode::Coarse
        } else {
            AjocQuantMode::Fine
        }
    }

    /// Number of quantization levels `nquant` for the given matrix kind.
    ///
    /// Per §5.7.3.2 Table 43: dry uses 51 (coarse) / 101 (fine), wet
    /// uses 21 (coarse) / 41 (fine).
    pub fn nquant(self, kind: AjocMatrixKind) -> u32 {
        match (kind, self) {
            (AjocMatrixKind::Dry, AjocQuantMode::Coarse) => 51,
            (AjocMatrixKind::Dry, AjocQuantMode::Fine) => 101,
            (AjocMatrixKind::Wet, AjocQuantMode::Coarse) => 21,
            (AjocMatrixKind::Wet, AjocQuantMode::Fine) => 41,
        }
    }

    /// The zero-centre quantised index `(nquant - 1) / 2` used as the
    /// "matrix element absent" default (sparse path, §5.7.3.2).
    pub fn zero_index(self, kind: AjocMatrixKind) -> u32 {
        (self.nquant(kind) - 1) / 2
    }
}

// ---------------------------------------------------------------------
// §5.7.3.1 / Table 100 — parameter-band count and sb → pb mapping
// ---------------------------------------------------------------------

/// The eight supported `ajoc_num_bands` values, indexed by the order
/// they appear across Table 42's columns (descending).
pub const AJOC_NUM_BANDS_VALUES: [u32; 8] = [23, 15, 12, 9, 7, 5, 3, 1];

/// Resolve `ajoc_num_bands` from the 3-bit `ajoc_num_bands_code`
/// (§6.3.6.2.3 Table 100).
pub fn num_bands_from_code(code: u8) -> Result<u32> {
    AJOC_NUM_BANDS_VALUES
        .get(code as usize)
        .copied()
        .ok_or_else(|| Error::invalid("ac4: ajoc_num_bands_code out of range"))
}

/// A-JOC parameter band to QMF subband mapping (§5.7.3.1 Table 42).
///
/// Row index = QMF subband (0..=63); column index selects the
/// `ajoc_num_bands` configuration in the order of
/// [`AJOC_NUM_BANDS_VALUES`]. Subband ranges that share a parameter
/// band in the spec table are expanded per-subband here.
#[rustfmt::skip]
const AJOC_SB_TO_PB: [[u8; 8]; 64] = [
    //  23  15  12   9   7   5   3   1   <- ajoc_num_bands
    [    0,  0,  0,  0,  0,  0,  0,  0 ], // sb 0
    [    1,  1,  1,  1,  1,  1,  0,  0 ], // sb 1
    [    2,  2,  2,  2,  2,  1,  0,  0 ], // sb 2
    [    3,  3,  3,  3,  2,  2,  1,  0 ], // sb 3
    [    4,  4,  4,  3,  3,  2,  1,  0 ], // sb 4
    [    5,  5,  4,  4,  3,  2,  1,  0 ], // sb 5
    [    6,  6,  5,  4,  3,  2,  1,  0 ], // sb 6
    [    7,  7,  5,  5,  3,  2,  1,  0 ], // sb 7
    [    8,  8,  6,  5,  4,  2,  1,  0 ], // sb 8
    [    9,  9,  6,  6,  4,  3,  1,  0 ], // sb 9
    [   10,  9,  6,  6,  4,  3,  1,  0 ], // sb 10
    [   11, 10,  7,  6,  4,  3,  1,  0 ], // sb 11
    [   12, 10,  7,  6,  4,  3,  1,  0 ], // sb 12
    [   12, 10,  7,  6,  4,  3,  1,  0 ], // sb 13
    [   13, 11,  8,  7,  5,  3,  2,  0 ], // sb 14
    [   13, 11,  8,  7,  5,  3,  2,  0 ], // sb 15
    [   14, 11,  8,  7,  5,  3,  2,  0 ], // sb 16
    [   14, 11,  8,  7,  5,  3,  2,  0 ], // sb 17
    [   15, 12,  9,  7,  5,  3,  2,  0 ], // sb 18
    [   15, 12,  9,  7,  5,  3,  2,  0 ], // sb 19
    [   16, 12,  9,  7,  5,  3,  2,  0 ], // sb 20
    [   16, 12,  9,  7,  5,  3,  2,  0 ], // sb 21
    [   16, 12,  9,  7,  5,  3,  2,  0 ], // sb 22
    [   17, 13, 10,  8,  6,  4,  2,  0 ], // sb 23
    [   17, 13, 10,  8,  6,  4,  2,  0 ], // sb 24
    [   17, 13, 10,  8,  6,  4,  2,  0 ], // sb 25
    [   18, 13, 10,  8,  6,  4,  2,  0 ], // sb 26
    [   18, 13, 10,  8,  6,  4,  2,  0 ], // sb 27
    [   18, 13, 10,  8,  6,  4,  2,  0 ], // sb 28
    [   18, 13, 10,  8,  6,  4,  2,  0 ], // sb 29
    [   19, 13, 10,  8,  6,  4,  2,  0 ], // sb 30
    [   19, 13, 10,  8,  6,  4,  2,  0 ], // sb 31
    [   19, 13, 10,  8,  6,  4,  2,  0 ], // sb 32
    [   19, 13, 10,  8,  6,  4,  2,  0 ], // sb 33
    [   19, 13, 10,  8,  6,  4,  2,  0 ], // sb 34
    [   20, 14, 11,  8,  6,  4,  2,  0 ], // sb 35
    [   20, 14, 11,  8,  6,  4,  2,  0 ], // sb 36
    [   20, 14, 11,  8,  6,  4,  2,  0 ], // sb 37
    [   20, 14, 11,  8,  6,  4,  2,  0 ], // sb 38
    [   20, 14, 11,  8,  6,  4,  2,  0 ], // sb 39
    [   20, 14, 11,  8,  6,  4,  2,  0 ], // sb 40
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 41
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 42
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 43
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 44
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 45
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 46
    [   21, 14, 11,  8,  6,  4,  2,  0 ], // sb 47
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 48
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 49
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 50
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 51
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 52
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 53
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 54
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 55
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 56
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 57
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 58
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 59
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 60
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 61
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 62
    [   22, 14, 11,  8,  6,  4,  2,  0 ], // sb 63
];

/// Column index into [`AJOC_SB_TO_PB`] for a given `ajoc_num_bands`.
fn band_config_column(num_bands: u32) -> Result<usize> {
    AJOC_NUM_BANDS_VALUES
        .iter()
        .position(|&v| v == num_bands)
        .ok_or_else(|| Error::invalid("ac4: unsupported ajoc_num_bands"))
}

/// Map a QMF subband (0..=63) to its A-JOC parameter band for the given
/// `ajoc_num_bands` configuration (§5.7.3.1 Table 42).
pub fn sb_to_pb(sb: u32, num_bands: u32) -> Result<u32> {
    if sb >= 64 {
        return Err(Error::invalid("ac4: A-JOC QMF subband out of range"));
    }
    let col = band_config_column(num_bands)?;
    Ok(AJOC_SB_TO_PB[sb as usize][col] as u32)
}

// ---------------------------------------------------------------------
// §5.7.3.3 Tables 44-47 — dequantization
// ---------------------------------------------------------------------

// The four A-JOC dequantizers are uniform: a single step size and a
// zero-centred index. The constants below reproduce the documented
// table endpoints bit-exactly (verified in the unit tests against the
// Table 44-47 first / last / centre entries).

/// Dry coarse quantizer step (Table 44: 0.20019531..., 51 levels).
const DRY_COARSE_STEP: f64 = 5.0048828 / 25.0;
/// Dry fine quantizer step (Table 45: 0.10009766..., 101 levels).
const DRY_FINE_STEP: f64 = 5.00488281 / 50.0;
/// Wet coarse quantizer step (Table 46: 0.200195313..., 21 levels).
const WET_COARSE_STEP: f64 = 2.001953125 / 10.0;
/// Wet fine quantizer step (Table 47: 0.10009766..., 41 levels).
const WET_FINE_STEP: f64 = 2.00195313 / 20.0;

/// Dequantize a dry matrix index `mtx_dry_q` (§5.7.3.3 Tables 44 / 45).
pub fn dequantize_dry(q: u32, mode: AjocQuantMode) -> f64 {
    let (step, centre) = match mode {
        AjocQuantMode::Coarse => (DRY_COARSE_STEP, 25.0),
        AjocQuantMode::Fine => (DRY_FINE_STEP, 50.0),
    };
    (q as f64 - centre) * step
}

/// Dequantize a wet matrix index `mtx_wet_q` (§5.7.3.3 Tables 46 / 47).
pub fn dequantize_wet(q: u32, mode: AjocQuantMode) -> f64 {
    let (step, centre) = match mode {
        AjocQuantMode::Coarse => (WET_COARSE_STEP, 10.0),
        AjocQuantMode::Fine => (WET_FINE_STEP, 20.0),
    };
    (q as f64 - centre) * step
}

// ---------------------------------------------------------------------
// §5.7.3.2 Table 43 — differential decoding
// ---------------------------------------------------------------------

/// Direction of the per-band differential coding for one data point
/// (`diff_type`, §6.3.6.5.1 Table 103).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AjocDiffType {
    /// `diff_type == 0` — frequency-differential coding.
    Freq,
    /// `diff_type == 1` — time-differential coding.
    Time,
}

impl AjocDiffType {
    /// Resolve from the 1-bit `diff_type` field.
    pub fn from_bit(b: bool) -> Self {
        if b {
            AjocDiffType::Time
        } else {
            AjocDiffType::Freq
        }
    }
}

/// Per-object running `mtx_*_q_prev` state used by DIFF_TIME decoding
/// across data points and across AC-4 frames (§5.7.3.2).
#[derive(Clone, Debug, Default)]
pub struct AjocDiffState {
    /// `mtx_dry_q_prev[o][ch][pb]`.
    pub dry_prev: Vec<Vec<Vec<i32>>>,
    /// `mtx_wet_q_prev[o][de][pb]`.
    pub wet_prev: Vec<Vec<Vec<i32>>>,
}

impl AjocDiffState {
    /// Allocate previous-value state for the given dimensions, all zero.
    pub fn new(num_umx: usize, num_dmx: usize, num_decorr: usize, num_bands: usize) -> Self {
        AjocDiffState {
            dry_prev: vec![vec![vec![0; num_bands]; num_dmx]; num_umx],
            wet_prev: vec![vec![vec![0; num_bands]; num_decorr]; num_umx],
        }
    }
}

/// Recover one channel's absolute quantised dry values for a single
/// data point from the Huffman deltas (§5.7.3.2 Table 43).
///
/// * `deltas` — `mix_mtx_dry[o][dp][ch][0..num_bands]` Huffman output.
/// * `prev` — `mtx_dry_q_prev[o][ch][0..num_bands]`; updated in place to
///   this data point's result on return (so the caller chains it to the
///   next data point / frame).
/// * `present` — `false` when `ajoc_sparse_select == 1` and this
///   channel's `ajoc_sparse_mask_dry` bit is 0 (matrix element absent);
///   the band values default to the zero-centre index.
pub fn differential_decode_dry(
    deltas: &[i32],
    diff_type: AjocDiffType,
    mode: AjocQuantMode,
    present: bool,
    prev: &mut [i32],
) -> Vec<i32> {
    differential_decode(deltas, diff_type, mode, AjocMatrixKind::Dry, present, prev)
}

/// Recover one decorrelator's absolute quantised wet values for a
/// single data point (§5.7.3.2 Table 43). See [`differential_decode_dry`].
pub fn differential_decode_wet(
    deltas: &[i32],
    diff_type: AjocDiffType,
    mode: AjocQuantMode,
    present: bool,
    prev: &mut [i32],
) -> Vec<i32> {
    differential_decode(deltas, diff_type, mode, AjocMatrixKind::Wet, present, prev)
}

fn differential_decode(
    deltas: &[i32],
    diff_type: AjocDiffType,
    mode: AjocQuantMode,
    kind: AjocMatrixKind,
    present: bool,
    prev: &mut [i32],
) -> Vec<i32> {
    let num_bands = prev.len();
    let nquant = mode.nquant(kind) as i32;
    let mut out = vec![0i32; num_bands];

    if !present {
        let zero = mode.zero_index(kind) as i32;
        for v in out.iter_mut() {
            *v = zero;
        }
    } else {
        match diff_type {
            AjocDiffType::Freq => {
                // DIFF_FREQ: band 0 absolute, running modulo-nquant sum.
                if num_bands > 0 {
                    out[0] = deltas.first().copied().unwrap_or(0);
                    for pb in 1..num_bands {
                        let d = deltas.get(pb).copied().unwrap_or(0);
                        // Euclidean modulo keeps the index in [0, nquant).
                        out[pb] = (out[pb - 1] + d).rem_euclid(nquant);
                    }
                }
            }
            AjocDiffType::Time => {
                // DIFF_TIME: per-band add to the previous frame's value.
                for pb in 0..num_bands {
                    let d = deltas.get(pb).copied().unwrap_or(0);
                    out[pb] = prev[pb] + d;
                }
            }
        }
    }

    // mtx_*_q_prev[o][.] = mtx_*_q[o][dp][.]
    prev.copy_from_slice(&out);
    out
}

// ---------------------------------------------------------------------
// §5.7.3.4 Table 48 — time interpolation
// ---------------------------------------------------------------------

/// Linear-ramp interpolator for a single matrix element track
/// (`ajoc_interpolate()`, §5.7.3.4 Table 48).
///
/// The ramp advances by a fixed `delta_inc` per QMF time slot while
/// `curr_ramp_len <= target_ramp_len`, then holds. Whenever the
/// caller-supplied time slot `ts` equals an `ajoc_start_pos[dp]`, the
/// increment is re-armed toward that data point's target value over
/// `ajoc_ramp_len[dp]` slots.
///
/// The `curr_ramp_len` / `target_ramp_len` reset that the Table 49
/// driver performs at the *end* of each time-slot iteration
/// (`if ts == ajoc_start_pos[dp]: curr_ramp_len = 0; target_ramp_len =
/// ajoc_ramp_len[dp]`) is exposed separately as [`AjocInterpolator::end_of_slot`]
/// because that reset is shared across all `(o, ch, de)` tracks for a
/// given `(ts, sb)` — the caller invokes [`AjocInterpolator::step`] for
/// each track then [`AjocInterpolator::end_of_slot`] once per track at
/// the slot boundary.
#[derive(Clone, Debug, Default)]
pub struct AjocInterpolator {
    prev_value: f64,
    delta_inc: f64,
    curr_ramp_len: u32,
    target_ramp_len: u32,
}

impl AjocInterpolator {
    /// Start an interpolator seeded with the previous frame's tail value.
    pub fn new(prev_value: f64) -> Self {
        AjocInterpolator {
            prev_value,
            delta_inc: 0.0,
            curr_ramp_len: 0,
            target_ramp_len: 0,
        }
    }

    /// Advance one QMF time slot per `ajoc_interpolate()` (Table 48):
    /// emit `prev + delta_inc` while still ramping (else hold), then
    /// re-arm `delta_inc` if `ts` hits a data point's `ajoc_start_pos`.
    ///
    /// `targets[dp]` is the data point's dequantized target value,
    /// `start_pos[dp]` its ramp start slot, `ramp_len[dp]` its ramp
    /// length in slots.
    pub fn step(&mut self, ts: u32, targets: &[f64], start_pos: &[u32], ramp_len: &[u32]) -> f64 {
        let interpolated = if self.curr_ramp_len <= self.target_ramp_len {
            let v = self.prev_value + self.delta_inc;
            self.prev_value = v;
            self.curr_ramp_len = self.curr_ramp_len.saturating_add(1);
            v
        } else {
            self.prev_value
        };

        for dp in 0..targets.len() {
            if start_pos.get(dp).copied() == Some(ts) {
                let rl = ramp_len.get(dp).copied().unwrap_or(1).max(1) as f64;
                self.delta_inc = (targets[dp] - self.prev_value) / rl;
            }
        }

        interpolated
    }

    /// Apply the Table 49 end-of-time-slot ramp reset: if `ts` equals a
    /// data point's `ajoc_start_pos`, restart the ramp counter and set
    /// the target length to that data point's `ajoc_ramp_len`.
    pub fn end_of_slot(&mut self, ts: u32, start_pos: &[u32], ramp_len: &[u32]) {
        for dp in 0..start_pos.len() {
            if start_pos[dp] == ts {
                self.curr_ramp_len = 0;
                self.target_ramp_len = ramp_len.get(dp).copied().unwrap_or(1);
            }
        }
    }
}

// ---------------------------------------------------------------------
// §5.7.3.6 Table 49 — signal reconstruction (dry path)
// ---------------------------------------------------------------------

/// Reconstruct the dry contribution to one output object's QMF sample
/// `z[o](ts, sb)` from the downmix inputs (§5.7.3.6.1):
///
/// `z = Σ_ch mtx_dry[o][ch] · x[ch]`
///
/// `mtx_dry` is the per-channel dry coefficient for this `(o, ts, sb)`
/// after interpolation + ungrouping; `x` the downmix input samples for
/// this `(ts, sb)`. Returns 0 when the object is not present.
pub fn reconstruct_dry(object_present: bool, mtx_dry: &[f64], x: &[f64]) -> f64 {
    if !object_present {
        return 0.0;
    }
    mtx_dry.iter().zip(x.iter()).map(|(&m, &xc)| m * xc).sum()
}

/// Reconstruct the full output sample `z[o](ts, sb)` including the wet
/// (decorrelator) contribution (§5.7.3.6.1):
///
/// `z = Σ_ch mtx_dry[o][ch]·x[ch] + Σ_de mtx_wet[o][de]·y[de]`
///
/// `y` is the decorrelator output bank for this `(ts, sb)`.
pub fn reconstruct(
    object_present: bool,
    mtx_dry: &[f64],
    x: &[f64],
    mtx_wet: &[f64],
    y: &[f64],
) -> f64 {
    if !object_present {
        return 0.0;
    }
    let dry: f64 = mtx_dry.iter().zip(x.iter()).map(|(&m, &xc)| m * xc).sum();
    let wet: f64 = mtx_wet.iter().zip(y.iter()).map(|(&m, &yd)| m * yd).sum();
    dry + wet
}

// ---------------------------------------------------------------------
// §5.7.3.6 Table 49 — full A-JOC spatial reconstruction driver
// ---------------------------------------------------------------------

use crate::acpl_synth::{DecorrelatorId, InputSignalModifier};

/// A complex QMF sample `(re, im)`.
pub type Cplx = (f64, f64);

/// Per-data-point dequantized A-JOC reconstruction matrices for one AC-4
/// frame, addressed in parameter-band space (the output of differential
/// decode + dequantize, §5.7.3.2 / §5.7.3.3 — *before* the ts/sb
/// interpolation + ungrouping done by [`ajoc_reconstruct`]).
///
/// The dimensions follow the Table 49 NOTE: `dry_dq[o][dp][ch][pb]` and
/// `wet_dq[o][dp][de][pb]`. `num_bands[o]` may differ per object, but
/// every row is sized to that object's `ajoc_num_bands[o]`.
#[derive(Clone, Debug, Default)]
pub struct AjocDequantMatrices {
    /// `mtx_dry_dq[o][dp][ch][pb]`.
    pub dry_dq: Vec<Vec<Vec<Vec<f64>>>>,
    /// `mtx_wet_dq[o][dp][de][pb]`.
    pub wet_dq: Vec<Vec<Vec<Vec<f64>>>>,
    /// `ajoc_num_bands[o]` for each output object.
    pub num_bands: Vec<u32>,
}

/// Geometry of one A-JOC reconstruction call.
#[derive(Clone, Copy, Debug)]
pub struct AjocGeometry {
    /// `num_dmx_signals` — number of jointly-coded downmix input objects.
    pub num_dmx: usize,
    /// `num_umx_signals` — number of reconstructed output objects.
    pub num_umx: usize,
    /// `ajoc_num_decorr` — number of parallel decorrelator instances.
    pub num_decorr: usize,
    /// `num_qmf_timeslots` for this frame.
    pub num_timeslots: usize,
    /// `num_qmf_subbands` (64 in AC-4 A-JOC).
    pub num_subbands: usize,
}

/// Map an A-JOC decorrelator index `de` to one of the three physical
/// decorrelator modules per §5.7.3.5: the three available decorrelators
/// are used cyclically `0, 2, 1, 0, 2, 1, 0` as `de` increases.
pub fn decorr_module_for(de: usize) -> DecorrelatorId {
    match de % 3 {
        0 => DecorrelatorId::D0,
        1 => DecorrelatorId::D2,
        _ => DecorrelatorId::D1,
    }
}

/// Persistent A-JOC reconstruction state carried across AC-4 frames: the
/// per-track linear-ramp interpolators for the dry / wet / pre matrices,
/// plus the `ajoc_num_decorr` decorrelator history banks.
///
/// Track layout (matching the Table 49 NOTE dimensions, addressed in
/// QMF-subband space because interpolation runs per `(ts, sb)`):
/// * `dry[o][ch][sb]`, `wet[o][de][sb]`, `pre[de][ch][sb]`.
#[derive(Clone, Debug)]
pub struct AjocReconState {
    dry: Vec<Vec<Vec<AjocInterpolator>>>,
    wet: Vec<Vec<Vec<AjocInterpolator>>>,
    pre: Vec<Vec<Vec<AjocInterpolator>>>,
    decorr: Vec<InputSignalModifier>,
    geom: (usize, usize, usize, usize),
}

impl AjocReconState {
    /// Allocate fresh reconstruction state for the given geometry. All
    /// interpolators start at 0 (matching the spec's `*_prev` arrays,
    /// which are zero on the first frame) and the decorrelator histories
    /// are empty.
    pub fn new(geom: &AjocGeometry) -> Self {
        let mk = || AjocInterpolator::new(0.0);
        let dry = (0..geom.num_umx)
            .map(|_| {
                (0..geom.num_dmx)
                    .map(|_| (0..geom.num_subbands).map(|_| mk()).collect())
                    .collect()
            })
            .collect();
        let wet = (0..geom.num_umx)
            .map(|_| {
                (0..geom.num_decorr)
                    .map(|_| (0..geom.num_subbands).map(|_| mk()).collect())
                    .collect()
            })
            .collect();
        let pre = (0..geom.num_decorr)
            .map(|_| {
                (0..geom.num_dmx)
                    .map(|_| (0..geom.num_subbands).map(|_| mk()).collect())
                    .collect()
            })
            .collect();
        let decorr = (0..geom.num_decorr)
            .map(|de| InputSignalModifier::new(decorr_module_for(de)))
            .collect();
        AjocReconState {
            dry,
            wet,
            pre,
            decorr,
            geom: (
                geom.num_dmx,
                geom.num_umx,
                geom.num_decorr,
                geom.num_subbands,
            ),
        }
    }
}

/// Compute the §5.7.3.6.2 decorrelation-input pre-matrix parameters
/// `mtx_pre_param[de][ch][pb]` from the dequantized dry / wet matrices,
/// per the first loop of Table 49:
///
/// `mtx_pre_param[pb][de][ch] += |mtx_wet_dq[o][de][pb]| * mtx_dry_dq[o][ch][pb]`
///
/// summed over the output objects `o` (= `|Csub2'^T| × Csub1`,
/// §5.7.3.6.2). The result is addressed `[de][ch][pb]` with `pb`
/// running over `num_bands` (the maximum band count across objects;
/// objects with fewer bands contribute only to their own range).
fn pre_matrix_param(
    matrices: &AjocDequantMatrices,
    dp: usize,
    geom: &AjocGeometry,
    num_bands: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut pre = vec![vec![vec![0.0f64; num_bands]; geom.num_dmx]; geom.num_decorr];
    for o in 0..geom.num_umx {
        let nb = matrices.num_bands.get(o).copied().unwrap_or(0) as usize;
        let dry_o = match matrices.dry_dq.get(o).and_then(|d| d.get(dp)) {
            Some(d) => d,
            None => continue,
        };
        let wet_o = match matrices.wet_dq.get(o).and_then(|w| w.get(dp)) {
            Some(w) => w,
            None => continue,
        };
        for pb in 0..nb {
            for ch in 0..geom.num_dmx {
                let dry = dry_o
                    .get(ch)
                    .and_then(|r| r.get(pb))
                    .copied()
                    .unwrap_or(0.0);
                for de in 0..geom.num_decorr {
                    let wet = wet_o
                        .get(de)
                        .and_then(|r| r.get(pb))
                        .copied()
                        .unwrap_or(0.0);
                    pre[de][ch][pb] += wet.abs() * dry;
                }
            }
        }
    }
    pre
}

/// TS 103 190-2 §5.8.2.4 — dialogue enhancement for **core decoding**
/// of A-JOC content, applied in place on the downmix QMF signals
/// `x[ts][sb][ch]`:
///
/// ```text
///   y_ch(ts, sb) = Σ_dlg H_M(ts, sb)[ch][dlg] · (H_A(ts, sb)[dlg] · x(ts, sb)) + x_ch(ts, sb)
/// ```
///
/// `H_A` (Pseudocode 23) is the interpolated dry matrix of the
/// main-dialogue objects (`dlg_idx`) — the same §5.7.3.4 interpolators
/// the full path steps, on `state` — with the rows scaled by the
/// core-decoding `de_gain = 10^(G/20) − 1` (Pseudocode 22 applied to
/// mtx_dry); `H_M` (Pseudocode 24) is the per-subband
/// `de_dlg_dmx_coeff[dlg][ch]` grid, interpolated over the frame
/// between the previous frame's `H'_M` (`h_m_prev`) and this frame's
/// (`h_m_cur`) with weight `(ts+1) / num_qmf_timeslots`. Missing
/// coefficient rows read as zero (no contribution).
#[allow(clippy::too_many_arguments)]
pub fn ajoc_core_de_apply(
    x: &mut [Vec<Vec<Cplx>>],
    matrices: &AjocDequantMatrices,
    data_point_info: &AjocDataPointInfo,
    geom: &AjocGeometry,
    state: &mut AjocReconState,
    dlg_idx: &[u32],
    de_gain: f64,
    h_m_prev: &[Vec<f64>],
    h_m_cur: &[Vec<f64>],
) {
    if dlg_idx.is_empty() || de_gain <= 0.0 {
        return;
    }
    let num_dp = data_point_info.num_dpoints.max(1) as usize;
    let start_pos = &data_point_info.start_pos;
    let ramp_len = &data_point_info.ramp_len;
    let num_ts = geom.num_timeslots.min(x.len());
    let coef = |grid: &[Vec<f64>], d: usize, ch: usize| -> f64 {
        grid.get(d).and_then(|r| r.get(ch)).copied().unwrap_or(0.0)
    };
    for (ts, x_ts) in x.iter_mut().enumerate().take(num_ts) {
        let w = (ts as f64 + 1.0) / geom.num_timeslots.max(1) as f64;
        for (sb, x_col) in x_ts.iter_mut().enumerate().take(geom.num_subbands) {
            // a[d] = de_gain · Σ_ch mtx_dry[dlg][ch] · x[ch]
            let mut a: Vec<Cplx> = Vec::with_capacity(dlg_idx.len());
            for &o in dlg_idx {
                let o = o as usize;
                let nb = matrices.num_bands.get(o).copied().unwrap_or(0);
                let mut acc = (0.0f64, 0.0f64);
                for ch in 0..geom.num_dmx {
                    let targets: Vec<f64> = (0..num_dp)
                        .map(|dp| {
                            matrices
                                .dry_dq
                                .get(o)
                                .and_then(|d| d.get(dp))
                                .and_then(|d| d.get(ch))
                                .map(|row| band_value(row, sb, nb))
                                .unwrap_or(0.0)
                        })
                        .collect();
                    let m = match state.dry.get_mut(o).and_then(|r| r.get_mut(ch)) {
                        Some(row) if sb < row.len() => {
                            row[sb].step(ts as u32, &targets, start_pos, ramp_len)
                        }
                        _ => 0.0,
                    };
                    let xc = x_col.get(ch).copied().unwrap_or((0.0, 0.0));
                    acc.0 += m * xc.0;
                    acc.1 += m * xc.1;
                }
                a.push((acc.0 * de_gain, acc.1 * de_gain));
            }
            for (ch, xc) in x_col.iter_mut().enumerate().take(geom.num_dmx) {
                let mut add = (0.0f64, 0.0f64);
                for (d, ad) in a.iter().enumerate() {
                    let h = (1.0 - w) * coef(h_m_prev, d, ch) + w * coef(h_m_cur, d, ch);
                    add.0 += h * ad.0;
                    add.1 += h * ad.1;
                }
                xc.0 += add.0;
                xc.1 += add.1;
            }
        }
        // Table 49 end-of-time-slot ramp re-arm on the interpolators
        // this pass stepped (the dialogue objects' dry rows).
        for &o in dlg_idx {
            if let Some(rows) = state.dry.get_mut(o as usize) {
                for row in rows.iter_mut() {
                    for interp in row.iter_mut() {
                        interp.end_of_slot(ts as u32, start_pos, ramp_len);
                    }
                }
            }
        }
    }
}

/// Write one `de_dlg_dmx_coeff_idx` (§6.3.6.6.4 Table 106) from its
/// index `k` in `0..=15` (`k/15` is the coefficient): `0` → `0`,
/// `15` → `1111`, otherwise `1` + 4 bits of `k − 1`. The exact inverse
/// of [`read_de_dlg_dmx_coeff`].
pub fn write_de_dlg_dmx_coeff_idx(bw: &mut oxideav_core::bits::BitWriter, k: u32) {
    match k {
        0 => bw.write_bit(false),
        15.. => bw.write_u32(0b1111, 4),
        k => {
            bw.write_bit(true);
            bw.write_u32(k - 1, 4);
        }
    }
}

/// Write `ajoc_dmx_de_data(num_dmx_signals, num_umx_signals)`
/// (§6.2.3.5) — the inverse of [`parse_ajoc_dmx_de_data`]. The
/// coefficient grid is emitted from `de_dlg_dmx_coeff` (values on the
/// `k/15` grid, rounded to the nearest index) when
/// `keep_dmx_de_coeffs == false`, one row per main-dialogue object of
/// `de_main_dlg_mask`.
pub fn write_ajoc_dmx_de_data(
    bw: &mut oxideav_core::bits::BitWriter,
    de: &AjocDmxDeData,
    num_dmx_signals: u32,
    num_umx_signals: u32,
) {
    bw.write_bit(de.dmx_de_cfg);
    bw.write_bit(de.keep_dmx_de_coeffs);
    if de.dmx_de_cfg {
        bw.write_u32(de.de_max_gain & 3, 2);
        bw.write_u32(de.de_main_dlg_mask, num_umx_signals);
    }
    if !de.keep_dmx_de_coeffs {
        let (num_dlg_obj, _) = dlg_obj(de.de_main_dlg_mask, num_umx_signals);
        for dio in 0..num_dlg_obj as usize {
            for dmxo in 0..num_dmx_signals as usize {
                let v = de
                    .de_dlg_dmx_coeff
                    .get(dio)
                    .and_then(|r| r.get(dmxo))
                    .copied()
                    .unwrap_or(0.0);
                let k = (v.clamp(0.0, 1.0) * 15.0).round() as u32;
                write_de_dlg_dmx_coeff_idx(bw, k);
            }
        }
    }
}

/// Per-track target value for a given parameter band, ungrouped to a QMF
/// subband via [`sb_to_pb`]. Returns 0 when the band is out of range.
fn band_value(row: &[f64], sb: usize, num_bands: u32) -> f64 {
    match sb_to_pb(sb as u32, num_bands) {
        Ok(pb) => row.get(pb as usize).copied().unwrap_or(0.0),
        Err(_) => 0.0,
    }
}

/// Run the full §5.7.3.6 / Table 49 A-JOC spatial reconstruction for one
/// AC-4 frame, transforming the `num_dmx` downmix QMF signals into the
/// `num_umx` reconstructed output objects.
///
/// * `x[ts][sb][ch]` — input downmix QMF samples.
/// * `matrices` — per-data-point dequantized dry / wet matrices.
/// * `data_point_info` — `ajoc_start_pos` / `ajoc_ramp_len` per data point.
/// * `object_present[o]` — gates the per-object reconstruction.
/// * `state` — interpolator + decorrelator history carried across frames.
///
/// Returns `z[ts][sb][o]`, the reconstructed output objects. Mirrors the
/// pseudocode order exactly: pre-matrix-param accumulation, then for each
/// `(ts, sb)` interpolate the dry / wet / pre tracks, form the
/// decorrelator inputs `u = pre · x`, decorrelate `y = D(u)`, and sum the
/// dry (`x · mtx_dry`) and wet (`y · mtx_wet`) contributions into `z`.
pub fn ajoc_reconstruct(
    x: &[Vec<Vec<Cplx>>],
    matrices: &AjocDequantMatrices,
    data_point_info: &AjocDataPointInfo,
    object_present: &[bool],
    geom: &AjocGeometry,
    state: &mut AjocReconState,
) -> Vec<Vec<Vec<Cplx>>> {
    ajoc_reconstruct_de(
        x,
        matrices,
        data_point_info,
        object_present,
        geom,
        state,
        None,
    )
}

/// TS 103 190-2 §5.8.2.3 Pseudocode 22 `ajoc_de_process()`: scale the
/// dry and wet matrix rows of the main-dialogue objects (`dlg_idx`,
/// from [`dlg_obj`]) by `de_gain` — the full-decoding dialogue
/// enhancement. Per the pseudocode's guard the scaling applies only
/// when `de_gain > 1`.
pub fn ajoc_de_process(matrices: &mut AjocDequantMatrices, dlg_idx: &[u32], de_gain: f64) {
    if de_gain <= 1.0 {
        return;
    }
    for &o in dlg_idx {
        let o = o as usize;
        if let Some(dry) = matrices.dry_dq.get_mut(o) {
            for v in dry.iter_mut().flatten().flatten() {
                *v *= de_gain;
            }
        }
        if let Some(wet) = matrices.wet_dq.get_mut(o) {
            for v in wet.iter_mut().flatten().flatten() {
                *v *= de_gain;
            }
        }
    }
}

/// [`ajoc_reconstruct`] with the §5.8.2.3 dialogue-enhancement hook
/// of Pseudocode 18: `de = Some((dlg_idx, de_gain))` runs
/// [`ajoc_de_process`] on the dequantized matrices **after** the
/// decorrelator pre-matrix parameters are formed (they follow the
/// unscaled matrices) and before the interpolation / reconstruction.
pub fn ajoc_reconstruct_de(
    x: &[Vec<Vec<Cplx>>],
    matrices: &AjocDequantMatrices,
    data_point_info: &AjocDataPointInfo,
    object_present: &[bool],
    geom: &AjocGeometry,
    state: &mut AjocReconState,
    de: Option<(&[u32], f64)>,
) -> Vec<Vec<Vec<Cplx>>> {
    let num_dp = data_point_info.num_dpoints.max(1) as usize;
    let start_pos = &data_point_info.start_pos;
    let ramp_len = &data_point_info.ramp_len;

    // Maximum band count over all objects, for the pre-matrix-param grid.
    let max_bands = matrices.num_bands.iter().copied().max().unwrap_or(0) as usize;

    // §5.7.3.6.2 decorrelation-input pre-matrix parameter, per data point.
    let pre_param: Vec<Vec<Vec<Vec<f64>>>> = (0..num_dp)
        .map(|dp| pre_matrix_param(matrices, dp, geom, max_bands))
        .collect();

    // §5.8.2.3: b_ajoc_de_process → ajoc_de_process() on the matrices
    // feeding the interpolation (the pre-matrix parameters above stay
    // on the unscaled matrices, as in Pseudocode 18).
    let de_scaled: AjocDequantMatrices;
    let matrices: &AjocDequantMatrices = match de {
        Some((dlg_idx, de_gain)) if de_gain > 1.0 && !dlg_idx.is_empty() => {
            let mut m = matrices.clone();
            ajoc_de_process(&mut m, dlg_idx, de_gain);
            de_scaled = m;
            &de_scaled
        }
        _ => matrices,
    };

    // Build per-track target arrays (target per data point) addressed by
    // QMF subband. For each `(o, ch, sb)` the dry target at data point dp
    // is `mtx_dry_dq[o][dp][ch][sb_to_pb(sb)]`, etc.
    let mut z = vec![vec![vec![(0.0, 0.0); geom.num_umx]; geom.num_subbands]; geom.num_timeslots];

    for ts in 0..geom.num_timeslots {
        for sb in 0..geom.num_subbands {
            // Interpolate dry tracks → mtx_dry[o][ch].
            let mut mtx_dry = vec![vec![0.0f64; geom.num_dmx]; geom.num_umx];
            for o in 0..geom.num_umx {
                let nb = matrices.num_bands.get(o).copied().unwrap_or(0);
                for ch in 0..geom.num_dmx {
                    let targets: Vec<f64> = (0..num_dp)
                        .map(|dp| {
                            matrices
                                .dry_dq
                                .get(o)
                                .and_then(|d| d.get(dp))
                                .and_then(|d| d.get(ch))
                                .map(|row| band_value(row, sb, nb))
                                .unwrap_or(0.0)
                        })
                        .collect();
                    mtx_dry[o][ch] =
                        state.dry[o][ch][sb].step(ts as u32, &targets, start_pos, ramp_len);
                }
            }
            // Interpolate wet tracks → mtx_wet[o][de].
            let mut mtx_wet = vec![vec![0.0f64; geom.num_decorr]; geom.num_umx];
            for o in 0..geom.num_umx {
                let nb = matrices.num_bands.get(o).copied().unwrap_or(0);
                for de in 0..geom.num_decorr {
                    let targets: Vec<f64> = (0..num_dp)
                        .map(|dp| {
                            matrices
                                .wet_dq
                                .get(o)
                                .and_then(|w| w.get(dp))
                                .and_then(|w| w.get(de))
                                .map(|row| band_value(row, sb, nb))
                                .unwrap_or(0.0)
                        })
                        .collect();
                    mtx_wet[o][de] =
                        state.wet[o][de][sb].step(ts as u32, &targets, start_pos, ramp_len);
                }
            }
            // Interpolate pre tracks → mtx_pre[de][ch].
            let mut mtx_pre = vec![vec![0.0f64; geom.num_dmx]; geom.num_decorr];
            for de in 0..geom.num_decorr {
                for ch in 0..geom.num_dmx {
                    let targets: Vec<f64> = (0..num_dp)
                        .map(|dp| {
                            let pb = sb_to_pb(sb as u32, max_bands.max(1) as u32).unwrap_or(0);
                            pre_param[dp][de][ch]
                                .get(pb as usize)
                                .copied()
                                .unwrap_or(0.0)
                        })
                        .collect();
                    mtx_pre[de][ch] =
                        state.pre[de][ch][sb].step(ts as u32, &targets, start_pos, ramp_len);
                }
            }

            // Decorrelation pre-matrix: u[de] = Σ_ch mtx_pre[de][ch] · x[ch].
            // Then decorrelate: y[de] = ajoc_decorrelate(u[de]).
            let mut y = vec![(0.0f64, 0.0f64); geom.num_decorr];
            for de in 0..geom.num_decorr {
                let mut u_re = 0.0f64;
                let mut u_im = 0.0f64;
                for ch in 0..geom.num_dmx {
                    let xc = x
                        .get(ts)
                        .and_then(|r| r.get(sb))
                        .and_then(|r| r.get(ch))
                        .copied()
                        .unwrap_or((0.0, 0.0));
                    u_re += mtx_pre[de][ch] * xc.0;
                    u_im += mtx_pre[de][ch] * xc.1;
                }
                let yd = state.decorr[de].process_sample(sb as u32, (u_re as f32, u_im as f32));
                y[de] = (yd.0 as f64, yd.1 as f64);
            }

            // Reconstruct: z[o] = Σ_ch x[ch]·mtx_dry[o][ch] + Σ_de y[de]·mtx_wet[o][de].
            for o in 0..geom.num_umx {
                if !object_present.get(o).copied().unwrap_or(false) {
                    continue;
                }
                let mut z_re = 0.0f64;
                let mut z_im = 0.0f64;
                for ch in 0..geom.num_dmx {
                    let xc = x
                        .get(ts)
                        .and_then(|r| r.get(sb))
                        .and_then(|r| r.get(ch))
                        .copied()
                        .unwrap_or((0.0, 0.0));
                    z_re += xc.0 * mtx_dry[o][ch];
                    z_im += xc.1 * mtx_dry[o][ch];
                }
                for de in 0..geom.num_decorr {
                    z_re += y[de].0 * mtx_wet[o][de];
                    z_im += y[de].1 * mtx_wet[o][de];
                }
                z[ts][sb][o] = (z_re, z_im);
            }
        }

        // End-of-time-slot ramp reset (Table 49 trailer): re-arm every
        // track's ramp counter at each data point's `ajoc_start_pos`.
        for o in 0..geom.num_umx {
            for ch in 0..geom.num_dmx {
                for sb in 0..geom.num_subbands {
                    state.dry[o][ch][sb].end_of_slot(ts as u32, start_pos, ramp_len);
                }
            }
            for de in 0..geom.num_decorr {
                for sb in 0..geom.num_subbands {
                    state.wet[o][de][sb].end_of_slot(ts as u32, start_pos, ramp_len);
                }
            }
        }
        for de in 0..geom.num_decorr {
            for ch in 0..geom.num_dmx {
                for sb in 0..geom.num_subbands {
                    state.pre[de][ch][sb].end_of_slot(ts as u32, start_pos, ramp_len);
                }
            }
        }
    }

    let _ = state.geom;
    z
}

// ---------------------------------------------------------------------
// §6.2.5 config-layer bitstream parsers (Huffman-independent)
// ---------------------------------------------------------------------

/// Parsed `ajoc_data_point_info()` (§6.2.5.4).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AjocDataPointInfo {
    /// `ajoc_num_dpoints` — number of A-JOC data points (0, 1 or 2).
    pub num_dpoints: u32,
    /// `ajoc_start_pos[dp]` — first QMF time slot of each ramp (5 bits).
    pub start_pos: Vec<u32>,
    /// `ajoc_ramp_len[dp]` = `ajoc_ramp_len_minus1[dp] + 1` (6-bit field).
    pub ramp_len: Vec<u32>,
}

/// Walk `ajoc_data_point_info()` (§6.2.5.4 / §6.3.6.4).
pub fn parse_ajoc_data_point_info(br: &mut BitReader<'_>) -> Result<AjocDataPointInfo> {
    let num_dpoints = br.read_u32(2)?;
    let mut start_pos = Vec::with_capacity(num_dpoints as usize);
    let mut ramp_len = Vec::with_capacity(num_dpoints as usize);
    for _ in 0..num_dpoints {
        start_pos.push(br.read_u32(5)?);
        ramp_len.push(br.read_u32(6)? + 1);
    }
    Ok(AjocDataPointInfo {
        num_dpoints,
        start_pos,
        ramp_len,
    })
}

/// Parsed per-object A-JOC control info (§6.2.5.2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AjocCtrlInfo {
    /// `ajoc_decorr_enable[d]` for each of `ajoc_num_decorr` decorrelators.
    pub decorr_enable: Vec<bool>,
    /// `ajoc_object_present[o]` for each of `num_umx_signals` objects.
    pub object_present: Vec<bool>,
    /// `ajoc_data_point_info()` payload.
    pub data_point_info: AjocDataPointInfo,
    /// `ajoc_num_bands_code[o]` (only meaningful for present objects).
    pub num_bands_code: Vec<u8>,
    /// Resolved `ajoc_num_bands[o]` (Table 100).
    pub num_bands: Vec<u32>,
    /// `ajoc_quant_select[o]`.
    pub quant_select: Vec<AjocQuantMode>,
    /// `ajoc_sparse_select[o]`.
    pub sparse_select: Vec<bool>,
    /// `ajoc_mix_mtx_dry_present[o][ch]` (sparse path only).
    pub mix_mtx_dry_present: Vec<Vec<bool>>,
    /// `ajoc_mix_mtx_wet_present[o][d]` (sparse path only).
    pub mix_mtx_wet_present: Vec<Vec<bool>>,
}

/// Walk `ajoc_ctrl_info(num_dmx_signals, ajoc_num_decorr,
/// num_umx_signals)` (§6.2.5.2).
pub fn parse_ajoc_ctrl_info(
    br: &mut BitReader<'_>,
    num_dmx_signals: u32,
    ajoc_num_decorr: u32,
    num_umx_signals: u32,
) -> Result<AjocCtrlInfo> {
    let num_umx = num_umx_signals as usize;
    let num_dmx = num_dmx_signals as usize;
    let num_decorr = ajoc_num_decorr as usize;

    let mut decorr_enable = Vec::with_capacity(num_decorr);
    for _ in 0..ajoc_num_decorr {
        decorr_enable.push(br.read_bit()?);
    }

    let mut object_present = Vec::with_capacity(num_umx);
    for _ in 0..num_umx_signals {
        object_present.push(br.read_bit()?);
    }

    let data_point_info = parse_ajoc_data_point_info(br)?;

    let mut num_bands_code = vec![0u8; num_umx];
    let mut num_bands = vec![0u32; num_umx];
    let mut quant_select = vec![AjocQuantMode::Fine; num_umx];
    let mut sparse_select = vec![false; num_umx];
    let mut mix_mtx_dry_present = vec![Vec::new(); num_umx];
    let mut mix_mtx_wet_present = vec![Vec::new(); num_umx];

    if data_point_info.num_dpoints != 0 {
        for o in 0..num_umx {
            if !object_present[o] {
                continue;
            }
            let code = br.read_u32(3)? as u8;
            num_bands_code[o] = code;
            num_bands[o] = num_bands_from_code(code)?;
            quant_select[o] = AjocQuantMode::from_bit(br.read_bit()?);
            let sparse = br.read_bit()?;
            sparse_select[o] = sparse;
            if sparse {
                let mut dry = Vec::with_capacity(num_dmx);
                for _ in 0..num_dmx {
                    dry.push(br.read_bit()?);
                }
                mix_mtx_dry_present[o] = dry;

                let mut wet = Vec::with_capacity(num_decorr);
                for d in 0..num_decorr {
                    if decorr_enable[d] {
                        wet.push(br.read_bit()?);
                    } else {
                        wet.push(false);
                    }
                }
                mix_mtx_wet_present[o] = wet;
            }
        }
    }

    Ok(AjocCtrlInfo {
        decorr_enable,
        object_present,
        data_point_info,
        num_bands_code,
        num_bands,
        quant_select,
        sparse_select,
        mix_mtx_dry_present,
        mix_mtx_wet_present,
    })
}

/// Parsed `ajoc_bed_info()` (§6.2.3.6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AjocBedInfo {
    /// `b_non_bed_obj_present`.
    pub non_bed_obj_present: bool,
    /// `num_bed_obj_ajoc` (only present when `non_bed_obj_present`).
    pub num_bed_obj: u32,
    /// Number of bits consumed (used to adjust the OAMD extension skip).
    pub bits_read: u32,
}

/// Walk `ajoc_bed_info()` (§6.2.3.6). Returns the parsed flags plus the
/// number of bits consumed (the `audio_data_ajoc()` OAMD-extension skip
/// subtracts this from its byte-aligned skip budget).
pub fn parse_ajoc_bed_info(br: &mut BitReader<'_>) -> Result<AjocBedInfo> {
    let non_bed_obj_present = br.read_bit()?;
    let mut bits_read = 1;
    let mut num_bed_obj = 0;
    if non_bed_obj_present {
        num_bed_obj = br.read_u32(3)?;
        bits_read += 3;
    }
    Ok(AjocBedInfo {
        non_bed_obj_present,
        num_bed_obj,
        bits_read,
    })
}

/// Decode one `de_dlg_dmx_coeff_idx` element to its `de_dlg_dmx_coeff`
/// value (§6.3.6.6.4 Table 106), a variable-length prefix code.
///
/// Returns `k/15` for `0b10000 + k` (k = 0..=13), `0` for `0b0`, and
/// `1` for `0b1111`.
pub fn read_de_dlg_dmx_coeff(br: &mut BitReader<'_>) -> Result<f64> {
    // 0b0 -> 0
    if !br.read_bit()? {
        return Ok(0.0);
    }
    // leading 1; next three bits distinguish 0b1111 (=> 1) from the
    // 0b10xxx / 0b11xxx family (5-bit codes).
    let b1 = br.read_bit()?;
    let b2 = br.read_bit()?;
    let b3 = br.read_bit()?;
    if b1 && b2 && b3 {
        // 0b1111
        return Ok(1.0);
    }
    let b4 = br.read_bit()?;
    // code = 1 bbbb with bbbb = (b1 b2 b3 b4); the table runs
    // 0b10000 (=1/15) .. 0b11101 (=14/15).
    let low = ((b1 as u32) << 3) | ((b2 as u32) << 2) | ((b3 as u32) << 1) | (b4 as u32);
    // low ranges 0b0000..0b1101 here (0b1110 / 0b1111 are the 0b1111
    // 4-bit escape handled above), giving k = low + 1 in 1..=14.
    let k = low + 1;
    if k > 14 {
        return Err(Error::invalid("ac4: invalid de_dlg_dmx_coeff_idx"));
    }
    Ok(k as f64 / 15.0)
}

/// Parsed `ajoc_dmx_de_data()` config (§6.2.3.5). The per-dialogue-object
/// downmix coefficient grid is parsed when `keep_dmx_de_coeffs == false`
/// and the caller supplies `num_dlg_objs` (derived from `de_main_dlg_mask`
/// per §6.3.6.6.3 `dlg_obj()`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AjocDmxDeData {
    /// `b_dmx_de_cfg`.
    pub dmx_de_cfg: bool,
    /// `b_keep_dmx_de_coeffs`.
    pub keep_dmx_de_coeffs: bool,
    /// `de_max_gain` (2 bits, only present when `dmx_de_cfg`).
    pub de_max_gain: u32,
    /// `de_main_dlg_mask` (`num_umx_signals` bits, only when `dmx_de_cfg`).
    pub de_main_dlg_mask: u32,
    /// `de_dlg_dmx_coeff[dio][dmxo]` decoded values (only when
    /// `keep_dmx_de_coeffs == false`).
    pub de_dlg_dmx_coeff: Vec<Vec<f64>>,
}

/// Derive `(num_dlg_obj, dlg_idx)` from `de_main_dlg_mask`
/// (§6.3.6.6.3 Table 105 `dlg_obj()`).
pub fn dlg_obj(de_main_dlg_mask: u32, num_umx_signals: u32) -> (u32, Vec<u32>) {
    let mut num_dlg_obj = 0;
    let mut dlg_idx = Vec::new();
    for obj in 0..num_umx_signals {
        // The mask carries at most the lowest 32 flag bits (objects
        // whose flag position falls outside a u32 read as 0).
        let bit = num_umx_signals - obj - 1;
        if bit < 32 && de_main_dlg_mask & (1u32 << bit) != 0 {
            dlg_idx.push(obj);
            num_dlg_obj += 1;
        }
    }
    (num_dlg_obj, dlg_idx)
}

/// Walk `ajoc_dmx_de_data(num_dmx_signals, num_umx_signals)` (§6.2.3.5).
pub fn parse_ajoc_dmx_de_data(
    br: &mut BitReader<'_>,
    num_dmx_signals: u32,
    num_umx_signals: u32,
) -> Result<AjocDmxDeData> {
    let dmx_de_cfg = br.read_bit()?;
    let keep_dmx_de_coeffs = br.read_bit()?;
    let mut de_max_gain = 0;
    let mut de_main_dlg_mask = 0;
    if dmx_de_cfg {
        de_max_gain = br.read_u32(2)?;
        // de_main_dlg_flag[] is one bit per upmix signal; the count
        // may exceed a u32 read (up to 256 objects), so consume the
        // flags bit-by-bit and keep the lowest 32 mask positions
        // (matching `dlg_obj`).
        for i in 0..num_umx_signals {
            let flag = br.read_bit()?;
            let bit = num_umx_signals - i - 1;
            if flag && bit < 32 {
                de_main_dlg_mask |= 1u32 << bit;
            }
        }
    }
    let mut de_dlg_dmx_coeff = Vec::new();
    if !keep_dmx_de_coeffs {
        let (num_dlg_objs, _) = dlg_obj(de_main_dlg_mask, num_umx_signals);
        for _dio in 0..num_dlg_objs {
            let mut row = Vec::with_capacity(num_dmx_signals as usize);
            for _dmxo in 0..num_dmx_signals {
                row.push(read_de_dlg_dmx_coeff(br)?);
            }
            de_dlg_dmx_coeff.push(row);
        }
    }
    Ok(AjocDmxDeData {
        dmx_de_cfg,
        keep_dmx_de_coeffs,
        de_max_gain,
        de_main_dlg_mask,
        de_dlg_dmx_coeff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_bands_table_100() {
        assert_eq!(num_bands_from_code(0).unwrap(), 23);
        assert_eq!(num_bands_from_code(1).unwrap(), 15);
        assert_eq!(num_bands_from_code(7).unwrap(), 1);
        assert!(num_bands_from_code(8).is_err());
    }

    #[test]
    fn sb_to_pb_endpoints() {
        // sb 0 always maps to pb 0; the highest sb maps to the highest pb.
        for &nb in &AJOC_NUM_BANDS_VALUES {
            assert_eq!(sb_to_pb(0, nb).unwrap(), 0);
            assert_eq!(sb_to_pb(63, nb).unwrap(), nb - 1);
        }
    }

    #[test]
    fn sb_to_pb_monotonic_and_complete() {
        // Each configuration must be non-decreasing in sb and must cover
        // every parameter band 0..num_bands exactly.
        for &nb in &AJOC_NUM_BANDS_VALUES {
            let mut last = 0;
            let mut seen = vec![false; nb as usize];
            for sb in 0..64 {
                let pb = sb_to_pb(sb, nb).unwrap();
                assert!(pb >= last, "nb={nb} sb={sb} pb={pb} < {last}");
                assert!(pb < nb, "nb={nb} sb={sb} pb={pb} >= {nb}");
                seen[pb as usize] = true;
                last = pb;
            }
            assert!(seen.iter().all(|&s| s), "nb={nb} skipped a parameter band");
        }
    }

    #[test]
    fn dequant_dry_endpoints() {
        // Table 44 (coarse): 0 -> -5,0048828, 25 -> 0, 50 -> +5,0048828.
        assert!((dequantize_dry(0, AjocQuantMode::Coarse) + 5.0048828).abs() < 1e-6);
        assert!(dequantize_dry(25, AjocQuantMode::Coarse).abs() < 1e-9);
        assert!((dequantize_dry(50, AjocQuantMode::Coarse) - 5.0048828).abs() < 1e-6);
        // Table 45 (fine): 0 -> -5,00488281, 50 -> 0, 100 -> +5,00488281.
        assert!((dequantize_dry(0, AjocQuantMode::Fine) + 5.00488281).abs() < 1e-6);
        assert!(dequantize_dry(50, AjocQuantMode::Fine).abs() < 1e-9);
        assert!((dequantize_dry(100, AjocQuantMode::Fine) - 5.00488281).abs() < 1e-6);
    }

    #[test]
    fn dequant_dry_interior() {
        // Table 44 entry 9 = -3,203125, entry 41 = +3,203125.
        assert!((dequantize_dry(9, AjocQuantMode::Coarse) + 3.203125).abs() < 1e-6);
        assert!((dequantize_dry(41, AjocQuantMode::Coarse) - 3.203125).abs() < 1e-6);
        // Table 45 entry 30 = -2,00195313.
        assert!((dequantize_dry(30, AjocQuantMode::Fine) + 2.00195313).abs() < 1e-6);
    }

    #[test]
    fn dequant_wet_endpoints() {
        // Table 46 (coarse): 0 -> -2,001953125, 10 -> 0, 20 -> +2,001953125.
        assert!((dequantize_wet(0, AjocQuantMode::Coarse) + 2.001953125).abs() < 1e-6);
        assert!(dequantize_wet(10, AjocQuantMode::Coarse).abs() < 1e-9);
        assert!((dequantize_wet(20, AjocQuantMode::Coarse) - 2.001953125).abs() < 1e-6);
        // Table 47 (fine): 0 -> -2,00195313, 20 -> 0, 40 -> +2,001953125.
        assert!((dequantize_wet(0, AjocQuantMode::Fine) + 2.00195313).abs() < 1e-6);
        assert!(dequantize_wet(20, AjocQuantMode::Fine).abs() < 1e-9);
        assert!((dequantize_wet(40, AjocQuantMode::Fine) - 2.001953125).abs() < 1e-6);
    }

    #[test]
    fn dequant_wet_interior() {
        // Table 47 entry 15 = -0,50048828, entry 25 = +0,500488281.
        assert!((dequantize_wet(15, AjocQuantMode::Fine) + 0.50048828).abs() < 1e-6);
        assert!((dequantize_wet(25, AjocQuantMode::Fine) - 0.500488281).abs() < 1e-6);
    }

    #[test]
    fn nquant_and_zero_index() {
        assert_eq!(AjocQuantMode::Coarse.nquant(AjocMatrixKind::Dry), 51);
        assert_eq!(AjocQuantMode::Fine.nquant(AjocMatrixKind::Dry), 101);
        assert_eq!(AjocQuantMode::Coarse.nquant(AjocMatrixKind::Wet), 21);
        assert_eq!(AjocQuantMode::Fine.nquant(AjocMatrixKind::Wet), 41);
        // Zero-centre indices match the dequant-to-0 indices above.
        assert_eq!(AjocQuantMode::Coarse.zero_index(AjocMatrixKind::Dry), 25);
        assert_eq!(AjocQuantMode::Fine.zero_index(AjocMatrixKind::Dry), 50);
        assert_eq!(AjocQuantMode::Coarse.zero_index(AjocMatrixKind::Wet), 10);
        assert_eq!(AjocQuantMode::Fine.zero_index(AjocMatrixKind::Wet), 20);
    }

    #[test]
    fn diff_freq_running_sum_modulo() {
        // DIFF_FREQ: band 0 = abs, then running modulo-nquant sum.
        let deltas = [10, 5, -3, 50];
        let mut prev = vec![0i32; 4];
        let out = differential_decode_dry(
            &deltas,
            AjocDiffType::Freq,
            AjocQuantMode::Coarse, // nquant = 51
            true,
            &mut prev,
        );
        // 10, 15, 12, (12+50)%51 = 11
        assert_eq!(out, vec![10, 15, 12, 11]);
        // prev updated to out.
        assert_eq!(prev, out);
    }

    #[test]
    fn diff_time_adds_to_prev() {
        let deltas = [1, -2, 3];
        let mut prev = vec![20, 20, 20];
        let out = differential_decode_dry(
            &deltas,
            AjocDiffType::Time,
            AjocQuantMode::Fine,
            true,
            &mut prev,
        );
        assert_eq!(out, vec![21, 18, 23]);
        assert_eq!(prev, out);
    }

    #[test]
    fn diff_absent_defaults_to_zero_index() {
        let deltas = [99, 99, 99];
        let mut prev = vec![7, 7, 7];
        let out = differential_decode_wet(
            &deltas,
            AjocDiffType::Freq,
            AjocQuantMode::Coarse, // wet nquant = 21, zero idx = 10
            false,
            &mut prev,
        );
        assert_eq!(out, vec![10, 10, 10]);
        assert_eq!(prev, out);
    }

    #[test]
    fn interpolate_ramp_then_hold() {
        // Ramp toward 4 with a 4-slot ramp armed at ts 0, then hold.
        // Driver order per Table 49: step() for each track, then
        // end_of_slot() at the slot boundary.
        let mut interp = AjocInterpolator::new(0.0);
        let targets = [4.0];
        let start = [0u32];
        let ramp = [4u32];
        let mut vals = Vec::new();
        for ts in 0..8 {
            vals.push(interp.step(ts, &targets, &start, &ramp));
            interp.end_of_slot(ts, &start, &ramp);
        }
        // ts 0 emits the pre-arm value (0); the increment (delta = 1.0)
        // is armed during the same step, the ramp counter reset at the
        // slot boundary, then the value climbs by 1.0 per slot.
        assert!((vals[0] - 0.0).abs() < 1e-9);
        assert!((vals[1] - 1.0).abs() < 1e-9);
        assert!((vals[2] - 2.0).abs() < 1e-9);
        assert!((vals[3] - 3.0).abs() < 1e-9);
        assert!((vals[4] - 4.0).abs() < 1e-9);
        // The `curr_ramp_len <= target_ramp_len` boundary applies one
        // final increment (curr 0..=4 = five updates), per the literal
        // Table 48 pseudocode, before the value holds.
        assert!((vals[5] - 5.0).abs() < 1e-9);
        // Once curr_ramp_len exceeds target_ramp_len the value holds.
        assert!((vals[6] - vals[5]).abs() < 1e-9);
        assert!((vals[7] - vals[5]).abs() < 1e-9);
    }

    #[test]
    fn reconstruct_dry_sum() {
        let mtx = [0.5, -0.25, 1.0];
        let x = [2.0, 4.0, -1.0];
        // 0.5*2 + (-0.25)*4 + 1.0*(-1) = 1 - 1 - 1 = -1
        assert!((reconstruct_dry(true, &mtx, &x) + 1.0).abs() < 1e-9);
        assert_eq!(reconstruct_dry(false, &mtx, &x), 0.0);
    }

    #[test]
    fn reconstruct_dry_plus_wet() {
        let mtx_dry = [1.0, 0.0];
        let x = [3.0, 9.0];
        let mtx_wet = [0.5];
        let y = [4.0];
        // 1*3 + 0*9 + 0.5*4 = 3 + 2 = 5
        assert!((reconstruct(true, &mtx_dry, &x, &mtx_wet, &y) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn dlg_obj_mask() {
        // num_umx_signals = 4, mask 0b1010 selects objects 0 and 2.
        let (n, idx) = dlg_obj(0b1010, 4);
        assert_eq!(n, 2);
        assert_eq!(idx, vec![0, 2]);
    }

    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                out[i / 8] |= 0x80 >> (i % 8);
            }
        }
        out
    }

    #[test]
    fn de_dlg_dmx_coeff_prefix_code() {
        // 0b0 -> 0
        let bytes = pack_bits(&[0]);
        let mut br = BitReader::new(&bytes);
        assert!((read_de_dlg_dmx_coeff(&mut br).unwrap()).abs() < 1e-12);

        // 0b10000 -> 1/15
        let bytes = pack_bits(&[1, 0, 0, 0, 0]);
        let mut br = BitReader::new(&bytes);
        assert!((read_de_dlg_dmx_coeff(&mut br).unwrap() - 1.0 / 15.0).abs() < 1e-12);

        // 0b11101 -> 14/15
        let bytes = pack_bits(&[1, 1, 1, 0, 1]);
        let mut br = BitReader::new(&bytes);
        assert!((read_de_dlg_dmx_coeff(&mut br).unwrap() - 14.0 / 15.0).abs() < 1e-12);

        // 0b1111 -> 1
        let bytes = pack_bits(&[1, 1, 1, 1]);
        let mut br = BitReader::new(&bytes);
        assert!((read_de_dlg_dmx_coeff(&mut br).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn parse_data_point_info_roundtrip() {
        // num_dpoints = 1, start_pos = 0b00011 (3), ramp_len_minus1 = 0b000101 (5).
        let bits = [
            0, 1, /*start*/ 0, 0, 0, 1, 1, /*ramp*/ 0, 0, 0, 1, 0, 1,
        ];
        let bytes = pack_bits(&bits);
        let mut br = BitReader::new(&bytes);
        let dpi = parse_ajoc_data_point_info(&mut br).unwrap();
        assert_eq!(dpi.num_dpoints, 1);
        assert_eq!(dpi.start_pos, vec![3]);
        assert_eq!(dpi.ramp_len, vec![6]); // minus1 + 1
    }

    #[test]
    fn parse_bed_info_present_and_absent() {
        // b_non_bed_obj_present = 0
        let bytes = pack_bits(&[0]);
        let mut br = BitReader::new(&bytes);
        let bi = parse_ajoc_bed_info(&mut br).unwrap();
        assert!(!bi.non_bed_obj_present);
        assert_eq!(bi.bits_read, 1);

        // present = 1, num_bed_obj_ajoc = 0b101 = 5
        let bytes = pack_bits(&[1, 1, 0, 1]);
        let mut br = BitReader::new(&bytes);
        let bi = parse_ajoc_bed_info(&mut br).unwrap();
        assert!(bi.non_bed_obj_present);
        assert_eq!(bi.num_bed_obj, 5);
        assert_eq!(bi.bits_read, 4);
    }

    #[test]
    fn decorr_module_cyclic_021() {
        // §5.7.3.5: three decorrelators used cyclically 0, 2, 1, 0, 2, 1, 0.
        let ids: Vec<_> = (0..7).map(decorr_module_for).collect();
        let as_num = |d: &DecorrelatorId| match d {
            DecorrelatorId::D0 => 0,
            DecorrelatorId::D1 => 1,
            DecorrelatorId::D2 => 2,
        };
        let seq: Vec<_> = ids.iter().map(as_num).collect();
        assert_eq!(seq, vec![0, 2, 1, 0, 2, 1, 0]);
    }

    #[test]
    fn pre_matrix_param_sum_over_objects() {
        // 2 umx objects, 2 dmx, 1 decorr, 1 band.
        // dry[o][dp][ch][pb], wet[o][dp][de][pb].
        // D[de][ch] = Σ_o |wet[o][de]| · dry[o][ch].
        let matrices = AjocDequantMatrices {
            dry_dq: vec![
                vec![vec![vec![1.0], vec![2.0]]],  // o0: ch0=1, ch1=2
                vec![vec![vec![0.5], vec![-1.0]]], // o1: ch0=0.5, ch1=-1
            ],
            wet_dq: vec![
                vec![vec![vec![3.0]]],  // o0: de0 = 3
                vec![vec![vec![-2.0]]], // o1: de0 = -2
            ],
            num_bands: vec![1, 1],
        };
        let geom = AjocGeometry {
            num_dmx: 2,
            num_umx: 2,
            num_decorr: 1,
            num_timeslots: 1,
            num_subbands: 64,
        };
        let pre = pre_matrix_param(&matrices, 0, &geom, 1);
        // D[0][0] = |3|·1 + |-2|·0.5 = 3 + 1 = 4
        // D[0][1] = |3|·2 + |-2|·(-1) = 6 - 2 = 4
        assert!((pre[0][0][0] - 4.0).abs() < 1e-9);
        assert!((pre[0][1][0] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn reconstruct_dry_only_tracks_interpolated_coeff() {
        // Single object, single downmix, no decorrelators, single data
        // point. The output equals the input times the interpolated dry
        // coefficient; on a converged plateau the interpolator settles at
        // the §5.7.3.4 Table 48 value `target·(1 + 1/ramp_len)` (the
        // documented one-extra-increment overshoot, see
        // `interpolate_ramp_then_hold`). Here target = 1.0, ramp_len = 4 →
        // plateau coefficient 1.25, input 2.0 → output 2.5.
        let geom = AjocGeometry {
            num_dmx: 1,
            num_umx: 1,
            num_decorr: 0,
            num_timeslots: 8,
            num_subbands: 64,
        };
        let matrices = AjocDequantMatrices {
            dry_dq: vec![vec![vec![vec![1.0]]]],
            wet_dq: vec![vec![vec![]]],
            num_bands: vec![1],
        };
        let dpi = AjocDataPointInfo {
            num_dpoints: 1,
            start_pos: vec![0],
            ramp_len: vec![4],
        };
        let mut x = vec![vec![vec![(0.0f64, 0.0f64); 1]; 64]; 8];
        for ts in 0..8 {
            x[ts][5][0] = (2.0, 0.0);
        }
        let mut state = AjocReconState::new(&geom);
        let z = ajoc_reconstruct(&x, &matrices, &dpi, &[true], &geom, &mut state);
        // Converged plateau coefficient = 1.0·(1 + 1/4) = 1.25; ×2.0 input.
        assert!((z[7][5][0].0 - 2.5).abs() < 1e-6, "got {:?}", z[7][5][0]);
        // The output is purely real (input has zero imaginary part).
        assert!(z[7][5][0].1.abs() < 1e-9);
        // A subband with no input stays zero.
        assert!(z[7][10][0].0.abs() < 1e-9);
        // Early ramp slot is below the plateau (monotone climb).
        assert!(z[1][5][0].0 < z[7][5][0].0);
    }

    #[test]
    fn reconstruct_absent_object_zeroed() {
        let geom = AjocGeometry {
            num_dmx: 1,
            num_umx: 1,
            num_decorr: 0,
            num_timeslots: 2,
            num_subbands: 64,
        };
        let matrices = AjocDequantMatrices {
            dry_dq: vec![vec![vec![vec![1.0]]]],
            wet_dq: vec![vec![vec![]]],
            num_bands: vec![1],
        };
        let dpi = AjocDataPointInfo {
            num_dpoints: 1,
            start_pos: vec![0],
            ramp_len: vec![1],
        };
        let mut x = vec![vec![vec![(5.0f64, 0.0f64); 1]; 64]; 2];
        x[0][0][0] = (5.0, 0.0);
        let mut state = AjocReconState::new(&geom);
        // object_present = false → all outputs zero despite non-zero dry.
        let z = ajoc_reconstruct(&x, &matrices, &dpi, &[false], &geom, &mut state);
        for ts in 0..2 {
            for sb in 0..64 {
                assert_eq!(z[ts][sb][0], (0.0, 0.0));
            }
        }
    }

    #[test]
    fn reconstruct_wet_adds_decorrelator_energy() {
        // With a non-zero wet matrix and a non-zero pre-matrix, the
        // decorrelator path injects energy distinct from the pure dry
        // passthrough: the output should differ from the dry-only result.
        let geom = AjocGeometry {
            num_dmx: 1,
            num_umx: 1,
            num_decorr: 1,
            num_timeslots: 8,
            num_subbands: 64,
        };
        let matrices = AjocDequantMatrices {
            // dry = 1.0, wet = 0.5 — so pre-param = |0.5|·1.0 = 0.5.
            dry_dq: vec![vec![vec![vec![1.0]]]],
            wet_dq: vec![vec![vec![vec![0.5]]]],
            num_bands: vec![1],
        };
        let dpi = AjocDataPointInfo {
            num_dpoints: 1,
            start_pos: vec![0],
            ramp_len: vec![1],
        };
        // Drive subband 3 (decorrelator region k0, delay 7) with a steady
        // input. The decorrelator's delayed tap means y[de] becomes
        // non-zero only once the input history fills the delay line, so a
        // late time slot carries wet energy injected through the
        // pre-matrix → decorrelate → wet-matrix chain.
        let geom = AjocGeometry {
            num_timeslots: 16,
            ..geom
        };
        let mut x = vec![vec![vec![(0.0f64, 0.0f64); 1]; 64]; 16];
        for ts in 0..16 {
            x[ts][3][0] = (1.0, 0.0);
        }
        let mut state = AjocReconState::new(&geom);
        let z = ajoc_reconstruct(&x, &matrices, &dpi, &[true], &geom, &mut state);
        // Dry-only contribution (coefficient plateau 2.0 with ramp_len 1):
        // input 1.0 → 2.0. The full output differs once the decorrelator
        // tail (delay 7) arrives and the wet matrix injects it.
        let dry_only = 2.0f64;
        let late_diff: f64 = (8..16).map(|ts| (z[ts][3][0].0 - dry_only).abs()).sum();
        assert!(late_diff > 1e-6, "expected decorrelator wet energy");
    }

    #[test]
    fn parse_ctrl_info_non_sparse() {
        // ajoc_num_decorr = 1, num_umx = 2, num_dmx = 2.
        // decorr_enable[0] = 1
        // object_present = [1, 0]
        // data_point_info: num_dpoints = 1, start=0, ramp_minus1=0
        // object 0: num_bands_code = 0b111 (=>1 band), quant_select=0, sparse=0
        let mut bits = Vec::new();
        bits.push(1u8); // decorr_enable[0]
        bits.extend_from_slice(&[1, 0]); // object_present
        bits.extend_from_slice(&[0, 1]); // num_dpoints = 1
        bits.extend_from_slice(&[0, 0, 0, 0, 0]); // start_pos = 0
        bits.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // ramp_len_minus1 = 0
        bits.extend_from_slice(&[1, 1, 1]); // num_bands_code = 7 -> 1 band
        bits.push(0); // quant_select = fine
        bits.push(0); // sparse_select = 0
        let bytes = pack_bits(&bits);
        let mut br = BitReader::new(&bytes);
        let ci = parse_ajoc_ctrl_info(&mut br, 2, 1, 2).unwrap();
        assert_eq!(ci.decorr_enable, vec![true]);
        assert_eq!(ci.object_present, vec![true, false]);
        assert_eq!(ci.data_point_info.num_dpoints, 1);
        assert_eq!(ci.num_bands[0], 1);
        assert_eq!(ci.quant_select[0], AjocQuantMode::Fine);
        assert!(!ci.sparse_select[0]);
    }
}
