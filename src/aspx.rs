//! AC-4 Advanced Spectral Extension (A-SPX) parameter parsing.
//!
//! A-SPX is AC-4's spectral-bandwidth-extension tool (ETSI TS 103 190-1
//! §4.2.12 / §4.3.10 / §5.7.6): the core codec carries the low band and
//! A-SPX reconstructs the high band in the QMF domain from a compact
//! envelope-energy sidecar and a small set of control parameters.
//!
//! This module implements:
//!
//! * **`aspx_config()`** (Table 50, §4.2.12.1) — the 15-bit I-frame
//!   configuration block.
//! * **`companding_control()`** (Table 49, §4.2.11) — the per-channel
//!   companding sidecar.
//! * **`aspx_framing()`** (Table 53, §4.2.12.4) — the per-channel
//!   envelope framing. This block is variable-width: its size depends
//!   on the `aspx_int_class` (a 1/2/3-bit prefix code selecting one of
//!   FIXFIX / FIXVAR / VARFIX / VARVAR), on the config fields
//!   `aspx_num_env_bits_fixfix` and `aspx_freq_res_mode`, on whether
//!   the frame is an I-frame (only I-frames carry the leading
//!   `aspx_var_bord_left` for VARFIX / VARVAR), and on a tool-level
//!   `num_aspx_timeslots` flag that selects between 1- and 2-bit fields
//!   for `aspx_num_rel_*` / `aspx_rel_bord_*` (Note 1 in Table 53).
//!   Together these fully determine the signal- and noise-envelope
//!   count (`aspx_num_env`, `aspx_num_noise`) for each channel — the
//!   inputs the envelope Huffman payload will be parameterised by.
//!
//! * **`aspx_delta_dir()`** (Table 54, §4.2.12.5) — one bit per signal
//!   envelope plus one per noise envelope.
//! * **`aspx_hfgen_iwc_1ch()`** (Table 55, §4.2.12.6) — 1-channel
//!   HF-generation control: `tna_mode[0..num_sbg_noise]`,
//!   `aspx_ah_present` + optional `add_harmonic[0..num_sbg_sig_highres]`,
//!   `aspx_fic_present` + optional `fic_used_in_sfb[0..]`, and
//!   `aspx_tic_present` + optional
//!   `tic_used_in_slot[0..num_aspx_timeslots]`. The three sbg / ats
//!   counts come from the A-SPX freq-scale derivation and are passed
//!   in from the caller.
//! * **`aspx_hfgen_iwc_2ch()`** (Table 56, §4.2.12.7) — 2-channel
//!   analogue with explicit per-channel `aspx_ah_left` / `_right` and
//!   `aspx_fic_left` / `_right` gates, plus `aspx_tic_copy` which
//!   lets the encoder mirror the left-channel TIC pattern into the
//!   right. Mirrors the pseudocode in Table 56 byte-for-byte.
//!
//! We deliberately stop short of:
//!
//! * `aspx_ec_data()` and the A-SPX Huffman tables in Annex A.2 —
//!   envelope and noise scale factors are Huffman-coded with dedicated
//!   codebooks (different ones per delta direction, quantization step,
//!   signal vs noise etc.) and would double the crate's line count.
//!   See the module-level roadmap in `lib.rs`.
//!
//! Even so, `parse_aspx_config()` is useful on its own: it unblocks the
//! `mono_codec_mode == ASPX` I-frame path in the outer `audio_data()`
//! walker, so the substream walker no longer bails out silently on
//! ASPX substreams. It just stops one syntax element earlier — after
//! the config — instead of at the `mono_codec_mode` flag.

use oxideav_core::bits::BitReader;
use oxideav_core::Result;

/// A-SPX frequency-resolution transmission mode (Table 124).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxFreqResMode {
    /// `aspx_freq_res` is signalled explicitly in `aspx_framing()`.
    Signalled,
    /// Defaults to low resolution.
    Low,
    /// Defaults to values dependent on signal envelope duration.
    DurationDependent,
    /// Defaults to high resolution.
    High,
}

impl AspxFreqResMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b11 {
            0 => Self::Signalled,
            1 => Self::Low,
            2 => Self::DurationDependent,
            _ => Self::High,
        }
    }
}

/// A-SPX master frequency-table scale (Table 119). `false` == low-bit-
/// rate scale-factor table (`sbg_template_lowres`), `true` == high-bit-
/// rate table (`sbg_template_highres`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxMasterFreqScale {
    LowRes,
    HighRes,
}

impl AspxMasterFreqScale {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::LowRes
        } else {
            Self::HighRes
        }
    }
}

/// A-SPX initial envelope quantization step (Table 118). Used as the
/// seed for `aspx_qmode_env[ch]`; re-initialised by `aspx_data_1ch()` /
/// `aspx_data_2ch()` when the interval class is FIXFIX with exactly one
/// envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxQuantStep {
    /// 1.5 dB step size.
    Fine,
    /// 3.0 dB step size.
    Coarse,
}

impl AspxQuantStep {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::Fine
        } else {
            Self::Coarse
        }
    }
}

/// Parsed `aspx_config()` (ETSI TS 103 190-1 §4.2.12.1, Table 50).
///
/// Total wire size is 15 bits: a single I-frame-sticky block that
/// configures the A-SPX processor for the substream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AspxConfig {
    /// `aspx_quant_mode_env` — 1 bit, Table 118.
    pub quant_mode_env: AspxQuantStep,
    /// `aspx_start_freq` — 3 bits, index into scale-factor subband group
    /// table counting up in 2-subband steps from the first subband
    /// (§4.3.10.1.2).
    pub start_freq: u8,
    /// `aspx_stop_freq` — 2 bits, index counting down in 2-subband
    /// steps from the last subband (§4.3.10.1.3).
    pub stop_freq: u8,
    /// `aspx_master_freq_scale` — 1 bit, Table 119.
    pub master_freq_scale: AspxMasterFreqScale,
    /// `aspx_interpolation` — 1 bit, Table 120.
    pub interpolation: bool,
    /// `aspx_preflat` — 1 bit, Table 121.
    pub preflat: bool,
    /// `aspx_limiter` — 1 bit, Table 122.
    pub limiter: bool,
    /// `aspx_noise_sbg` — 2 bits. Input to
    /// `numNoiseSbgroups = aspx_noise_sbg + 1` (§5.7.6.3.1.3).
    pub noise_sbg: u8,
    /// `aspx_num_env_bits_fixfix` — 1 bit, Table 123. When 0 the FIXFIX
    /// `tmp_num_env` field is 1 bit wide (1 or 2 envelopes); when 1
    /// it's 2 bits wide (1, 2 or 4 envelopes).
    pub num_env_bits_fixfix: u8,
    /// `aspx_freq_res_mode` — 2 bits, Table 124.
    pub freq_res_mode: AspxFreqResMode,
}

impl AspxConfig {
    /// Bit width of this config element on the wire.
    pub const BITS: u32 = 15;

    /// Returns `numNoiseSbgroups` per §5.7.6.3.1.3:
    /// `numNoiseSbgroups = aspx_noise_sbg + 1` (so one of 1..=4).
    pub fn num_noise_sbgroups(&self) -> u32 {
        self.noise_sbg as u32 + 1
    }

    /// Returns how many bits the `tmp_num_env` field in a FIXFIX
    /// `aspx_framing()` will take (1 or 2) per §4.3.10.1.9.
    pub fn fixfix_tmp_num_env_bits(&self) -> u32 {
        self.num_env_bits_fixfix as u32 + 1
    }

    /// Returns `true` when the configuration calls for explicit
    /// `aspx_freq_res` bits in `aspx_framing()` (Table 124 row 0).
    pub fn signals_freq_res(&self) -> bool {
        matches!(self.freq_res_mode, AspxFreqResMode::Signalled)
    }
}

/// Parse `aspx_config()` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 50 (§4.2.12.1). Consumes exactly 15 bits.
pub fn parse_aspx_config(br: &mut BitReader<'_>) -> Result<AspxConfig> {
    // Field order straight from Table 50.
    let quant_mode_env = AspxQuantStep::from_bit(br.read_u32(1)?);
    let start_freq = br.read_u32(3)? as u8;
    let stop_freq = br.read_u32(2)? as u8;
    let master_freq_scale = AspxMasterFreqScale::from_bit(br.read_u32(1)?);
    let interpolation = br.read_bit()?;
    let preflat = br.read_bit()?;
    let limiter = br.read_bit()?;
    let noise_sbg = br.read_u32(2)? as u8;
    let num_env_bits_fixfix = br.read_u32(1)? as u8;
    let freq_res_mode = AspxFreqResMode::from_u32(br.read_u32(2)?);
    Ok(AspxConfig {
        quant_mode_env,
        start_freq,
        stop_freq,
        master_freq_scale,
        interpolation,
        preflat,
        limiter,
        noise_sbg,
        num_env_bits_fixfix,
        freq_res_mode,
    })
}

/// Parsed `companding_control()` (ETSI TS 103 190-1 §4.2.11, Table 49).
///
/// Companding is a per-channel transient-response tool that toggles
/// short-time gain compression/expansion. This struct captures the
/// flags exactly as the bitstream carries them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompandingControl {
    /// Present only for `num_chan > 1`; absent for mono.
    pub sync_flag: Option<bool>,
    /// `b_compand_on[ch]` — length is `1` when `sync_flag == true`, else
    /// `num_chan`.
    pub compand_on: Vec<bool>,
    /// `b_compand_avg` — only present when at least one channel had
    /// companding off.
    pub compand_avg: Option<bool>,
}

/// Parse `companding_control(num_chan)` at the current bit-reader
/// position. `num_chan` matches the caller's companding-grouping arg:
/// 1 for single-channel calls, 2 for stereo, 3 / 5 for multi-channel.
pub fn parse_companding_control(
    br: &mut BitReader<'_>,
    num_chan: u32,
) -> Result<CompandingControl> {
    let sync_flag = if num_chan > 1 {
        Some(br.read_bit()?)
    } else {
        None
    };
    let nc = match sync_flag {
        Some(true) => 1,
        _ => num_chan,
    };
    let mut compand_on = Vec::with_capacity(nc as usize);
    let mut b_need_avg = false;
    for _ in 0..nc {
        let bit = br.read_bit()?;
        if !bit {
            b_need_avg = true;
        }
        compand_on.push(bit);
    }
    let compand_avg = if b_need_avg {
        Some(br.read_bit()?)
    } else {
        None
    };
    Ok(CompandingControl {
        sync_flag,
        compand_on,
        compand_avg,
    })
}

/// A-SPX interval class (ETSI TS 103 190-1 §4.3.10.4.1, Table 126).
///
/// Encoded on the wire as a 1/2/3-bit variable-length prefix code:
/// `0b0` → FIXFIX, `0b10` → FIXVAR, `0b110` → VARFIX, `0b111` → VARVAR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxIntClass {
    FixFix,
    FixVar,
    VarFix,
    VarVar,
}

impl AspxIntClass {
    /// Read `aspx_int_class` from the bit-reader using the prefix code
    /// in Table 126. Consumes between 1 and 3 bits.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        if !br.read_bit()? {
            return Ok(Self::FixFix); // 0
        }
        if !br.read_bit()? {
            return Ok(Self::FixVar); // 10
        }
        if !br.read_bit()? {
            Ok(Self::VarFix) // 110
        } else {
            Ok(Self::VarVar) // 111
        }
    }

    /// Number of bits this class occupies on the wire.
    pub fn bits(self) -> u32 {
        match self {
            Self::FixFix => 1,
            Self::FixVar => 2,
            Self::VarFix | Self::VarVar => 3,
        }
    }
}

/// Parsed per-channel `aspx_framing()` output (ETSI TS 103 190-1
/// §4.2.12.4 Table 53 / §4.3.10.4).
///
/// This holds the framing elements for a single A-SPX channel: the
/// interval class, the signal- and noise-envelope counts, the freq-res
/// vector (empty when `aspx_freq_res_mode != Signalled`), and the raw
/// variable-border fields that feed later A-SPX stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspxFraming {
    /// `aspx_int_class[ch]`.
    pub int_class: AspxIntClass,
    /// `aspx_num_env[ch]` — number of signal envelopes, derived from
    /// `tmp_num_env` (FIXFIX) or from `num_rel_left + num_rel_right + 1`
    /// (otherwise). Bounded by Table 128: 4 for FIXFIX, 5 otherwise.
    pub num_env: u32,
    /// `aspx_num_noise[ch]` — 1 if `num_env == 1`, else 2
    /// (§4.3.10.4.11).
    pub num_noise: u32,
    /// `aspx_freq_res[ch][env]`. Populated only when
    /// `aspx_freq_res_mode == Signalled`; empty otherwise.
    pub freq_res: Vec<bool>,
    /// `aspx_var_bord_left[ch]` — present for VARFIX / VARVAR and only
    /// in an I-frame (per Table 53).
    pub var_bord_left: Option<u8>,
    /// `aspx_var_bord_right[ch]` — present for FIXVAR / VARVAR.
    pub var_bord_right: Option<u8>,
    /// `aspx_num_rel_left[ch]`. Zero for FIXFIX / FIXVAR.
    pub num_rel_left: u8,
    /// `aspx_num_rel_right[ch]`. Zero for FIXFIX / VARFIX.
    pub num_rel_right: u8,
    /// `aspx_rel_bord_left[ch][rel]`. Empty when `num_rel_left == 0`.
    pub rel_bord_left: Vec<u8>,
    /// `aspx_rel_bord_right[ch][rel]`. Empty when `num_rel_right == 0`.
    pub rel_bord_right: Vec<u8>,
    /// `aspx_tsg_ptr[ch]` — transient-pointer, present for every class
    /// except FIXFIX. `ptr_bits = ceil(log2(num_env + 2))`.
    pub tsg_ptr: Option<u8>,
}

/// Parse `aspx_framing(ch)` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 53 (§4.2.12.4).
///
/// * `cfg` — the previously parsed `aspx_config()`, which drives the
///   FIXFIX envelope-count field width and the freq-res signalling.
/// * `b_iframe` — from `ac4_substream_info()`; gates `aspx_var_bord_left`
///   for VARFIX / VARVAR.
/// * `num_aspx_timeslots_over_8` — `true` when `num_aspx_timeslots > 8`;
///   selects 2-bit vs 1-bit fields for `aspx_num_rel_*` and
///   `aspx_rel_bord_*` (Note 1 in Table 53).
pub fn parse_aspx_framing(
    br: &mut BitReader<'_>,
    cfg: &AspxConfig,
    b_iframe: bool,
    num_aspx_timeslots_over_8: bool,
) -> Result<AspxFraming> {
    // Width of the Note-1 fields: aspx_num_rel_*, aspx_rel_bord_*.
    let note1_bits: u32 = if num_aspx_timeslots_over_8 { 2 } else { 1 };

    let int_class = AspxIntClass::read(br)?;
    let mut num_rel_left: u8 = 0;
    let mut num_rel_right: u8 = 0;
    let mut var_bord_left: Option<u8> = None;
    let mut var_bord_right: Option<u8> = None;
    let mut rel_bord_left: Vec<u8> = Vec::new();
    let mut rel_bord_right: Vec<u8> = Vec::new();
    // Signal-envelope count. FIXFIX sets this directly from tmp_num_env;
    // the other classes compute it after the branch.
    let num_env: u32;
    // Freq-res vector. Signalled in-band only when the config opted in.
    let mut freq_res: Vec<bool> = Vec::new();

    match int_class {
        AspxIntClass::FixFix => {
            // envbits = aspx_num_env_bits_fixfix + 1
            let envbits = cfg.fixfix_tmp_num_env_bits();
            let tmp_num_env = br.read_u32(envbits)?;
            // aspx_num_env = 1 << tmp_num_env
            num_env = 1u32 << tmp_num_env;
            if cfg.signals_freq_res() {
                freq_res.push(br.read_bit()?);
            }
        }
        AspxIntClass::FixVar => {
            var_bord_right = Some(br.read_u32(2)? as u8);
            num_rel_right = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_right {
                rel_bord_right.push(br.read_u32(note1_bits)? as u8);
            }
            num_env = u32::from(num_rel_left) + u32::from(num_rel_right) + 1;
        }
        AspxIntClass::VarVar => {
            if b_iframe {
                var_bord_left = Some(br.read_u32(2)? as u8);
            }
            num_rel_left = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_left {
                rel_bord_left.push(br.read_u32(note1_bits)? as u8);
            }
            var_bord_right = Some(br.read_u32(2)? as u8);
            num_rel_right = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_right {
                rel_bord_right.push(br.read_u32(note1_bits)? as u8);
            }
            num_env = u32::from(num_rel_left) + u32::from(num_rel_right) + 1;
        }
        AspxIntClass::VarFix => {
            if b_iframe {
                var_bord_left = Some(br.read_u32(2)? as u8);
            }
            num_rel_left = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_left {
                rel_bord_left.push(br.read_u32(note1_bits)? as u8);
            }
            num_env = u32::from(num_rel_left) + u32::from(num_rel_right) + 1;
        }
    }

    let mut tsg_ptr: Option<u8> = None;
    if !matches!(int_class, AspxIntClass::FixFix) {
        // ptr_bits = ceil(log2(num_env + 2)). "log" here is a float log
        // without rounding, so result = bits needed to represent
        // num_env + 2 - 1 = num_env + 1 (for powers of two). We use
        // u32::next_power_of_two / leading_zeros to evaluate ceil_log2.
        let ptr_bits = ceil_log2(num_env + 2);
        tsg_ptr = Some(br.read_u32(ptr_bits)? as u8);
        if cfg.signals_freq_res() {
            freq_res.reserve(num_env as usize);
            for _ in 0..num_env {
                freq_res.push(br.read_bit()?);
            }
        }
    }

    let num_noise = if num_env > 1 { 2 } else { 1 };

    Ok(AspxFraming {
        int_class,
        num_env,
        num_noise,
        freq_res,
        var_bord_left,
        var_bord_right,
        num_rel_left,
        num_rel_right,
        rel_bord_left,
        rel_bord_right,
        tsg_ptr,
    })
}

/// Parsed `aspx_delta_dir(ch)` (ETSI TS 103 190-1 §4.2.12.5, Table 54).
///
/// Two bit-arrays gating how the matching `aspx_ec_data()` Huffman
/// codebook interprets its deltas:
///
/// * `aspx_sig_delta_dir[ch][env]` (length `aspx_num_env[ch]`): per
///   signal-envelope direction flag.
/// * `aspx_noise_delta_dir[ch][env]` (length `aspx_num_noise[ch]`):
///   per noise-envelope direction flag.
///
/// Convention (per §4.3.10.5 / Table 130): `false` means DF
/// (delta-frequency / F0 depending on position), `true` means DT
/// (delta-time).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxDeltaDir {
    pub sig_delta_dir: Vec<bool>,
    pub noise_delta_dir: Vec<bool>,
}

/// Parse `aspx_delta_dir(ch)` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 54 (§4.2.12.5). Reads
/// `aspx_num_env[ch] + aspx_num_noise[ch]` bits in total.
pub fn parse_aspx_delta_dir(
    br: &mut BitReader<'_>,
    framing: &AspxFraming,
) -> Result<AspxDeltaDir> {
    let mut sig = Vec::with_capacity(framing.num_env as usize);
    for _ in 0..framing.num_env {
        sig.push(br.read_bit()?);
    }
    let mut noise = Vec::with_capacity(framing.num_noise as usize);
    for _ in 0..framing.num_noise {
        noise.push(br.read_bit()?);
    }
    Ok(AspxDeltaDir {
        sig_delta_dir: sig,
        noise_delta_dir: noise,
    })
}

/// Parsed `aspx_hfgen_iwc_1ch()` (ETSI TS 103 190-1 §4.2.12.6,
/// Table 55) — the 1-channel HF-generation + interleaved-waveform
/// coding element carried after `aspx_delta_dir(0)` on the
/// `aspx_data_1ch()` path.
///
/// Field semantics (§4.3.10.6):
///
/// * `tna_mode[n]` (2 bits) — subband-wise tonal-to-noise adjustment
///   selector for each of `num_sbg_noise` noise subband groups.
/// * `add_harmonic[n]` (1 bit, optional) — per-highres-subband add-
///   harmonics flag; gated by `aspx_ah_present`.
/// * `fic_used_in_sfb[n]` (1 bit, optional) — per-highres-subband
///   frequency-interleaved-coding flag; gated by `aspx_fic_present`.
/// * `tic_used_in_slot[n]` (1 bit, optional) — per-A-SPX-timeslot
///   time-interleaved-coding flag; gated by `aspx_tic_present`.
///
/// When a gate bit is 0 the corresponding vector stays all-zero per
/// the pseudocode's explicit initialisation loops.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxHfgenIwc1Ch {
    pub tna_mode: Vec<u8>,
    pub ah_present: bool,
    pub add_harmonic: Vec<bool>,
    pub fic_present: bool,
    pub fic_used_in_sfb: Vec<bool>,
    pub tic_present: bool,
    pub tic_used_in_slot: Vec<bool>,
}

/// Parse `aspx_hfgen_iwc_1ch()` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 55 (§4.2.12.6).
///
/// * `num_sbg_noise` — number of noise subband groups (§5.7.6.3.1.3).
/// * `num_sbg_sig_highres` — number of high-resolution signal subband
///   groups (§5.7.6.3.1.2).
/// * `num_aspx_timeslots` — A-SPX time slots per frame (§5.7.6.3.3.0,
///   Pseudocode 75a).
///
/// All three come from the A-SPX freq-scale derivation which is
/// driven by `aspx_config` + the frame's sample-rate family; this
/// parser takes them as ready-to-go counts so it stays decoupled from
/// that future derivation.
pub fn parse_aspx_hfgen_iwc_1ch(
    br: &mut BitReader<'_>,
    num_sbg_noise: u32,
    num_sbg_sig_highres: u32,
    num_aspx_timeslots: u32,
) -> Result<AspxHfgenIwc1Ch> {
    let mut tna_mode = Vec::with_capacity(num_sbg_noise as usize);
    for _ in 0..num_sbg_noise {
        tna_mode.push(br.read_u32(2)? as u8);
    }
    // aspx_ah_present + optional per-sbg add_harmonic flags.
    let ah_present = br.read_bit()?;
    let mut add_harmonic = vec![false; num_sbg_sig_highres as usize];
    if ah_present {
        for ah in add_harmonic.iter_mut() {
            *ah = br.read_bit()?;
        }
    }
    // aspx_fic_present + optional per-sbg fic_used_in_sfb flags.
    let fic_present = br.read_bit()?;
    let mut fic_used_in_sfb = vec![false; num_sbg_sig_highres as usize];
    if fic_present {
        for f in fic_used_in_sfb.iter_mut() {
            *f = br.read_bit()?;
        }
    }
    // aspx_tic_present + optional per-timeslot tic_used_in_slot flags.
    let tic_present = br.read_bit()?;
    let mut tic_used_in_slot = vec![false; num_aspx_timeslots as usize];
    if tic_present {
        for t in tic_used_in_slot.iter_mut() {
            *t = br.read_bit()?;
        }
    }
    Ok(AspxHfgenIwc1Ch {
        tna_mode,
        ah_present,
        add_harmonic,
        fic_present,
        fic_used_in_sfb,
        tic_present,
        tic_used_in_slot,
    })
}

/// Returns `ceil(log2(n))` for `n >= 1`. For `n == 1` returns 0, i.e.
/// "zero bits needed". Used to size `aspx_tsg_ptr` (`ptr_bits`).
fn ceil_log2(n: u32) -> u32 {
    debug_assert!(n >= 1);
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Return `num_qmf_timeslots = frame_length / num_qmf_subbands` per
/// ETSI TS 103 190-1 §5.7.3.2 (Table 189). `num_qmf_subbands` is fixed
/// at 64 for AC-4. Valid inputs are the eight base-rate
/// `frame_length` values: 2048 / 1920 / 1536 / 1024 / 960 / 768 / 512
/// / 384 — for which Table 189 gives 32 / 30 / 24 / 16 / 15 / 12 / 8 /
/// 6. We compute the quotient directly (which matches Table 189 for
/// any of those inputs) rather than hard-coding the table.
pub fn num_qmf_timeslots(frame_len_base: u32) -> u32 {
    frame_len_base / 64
}

/// Return `num_ts_in_ats` — how many QMF time slots make up one A-SPX
/// time slot — per ETSI TS 103 190-1 §5.7.6.3.3.0 (Table 192). Two for
/// `frame_length >= 1536`, one for shorter frames.
pub fn num_ts_in_ats(frame_len_base: u32) -> u32 {
    if frame_len_base >= 1536 {
        2
    } else {
        1
    }
}

/// Return `num_aspx_timeslots = num_qmf_timeslots / num_ts_in_ats` per
/// ETSI TS 103 190-1 §5.7.6.3.3.0 Pseudocode 75a. Valid only for the
/// eight base-rate `frame_length` values in Table 189.
pub fn num_aspx_timeslots(frame_len_base: u32) -> u32 {
    num_qmf_timeslots(frame_len_base) / num_ts_in_ats(frame_len_base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[allow(clippy::too_many_arguments)]
    fn build_aspx_config_bits(
        qmode: u32,
        start: u32,
        stop: u32,
        scale: u32,
        interp: bool,
        preflat: bool,
        limiter: bool,
        noise_sbg: u32,
        num_env_bits_fixfix: u32,
        freq_res_mode: u32,
    ) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write_u32(qmode, 1);
        bw.write_u32(start, 3);
        bw.write_u32(stop, 2);
        bw.write_u32(scale, 1);
        bw.write_bit(interp);
        bw.write_bit(preflat);
        bw.write_bit(limiter);
        bw.write_u32(noise_sbg, 2);
        bw.write_u32(num_env_bits_fixfix, 1);
        bw.write_u32(freq_res_mode, 2);
        bw.align_to_byte();
        bw.finish()
    }

    #[test]
    fn aspx_config_bit_order_matches_table_50() {
        // quant_mode_env=1 (coarse/3dB), start_freq=5, stop_freq=2,
        // master_freq_scale=1 (highres), interpolation=1, preflat=0,
        // limiter=1, noise_sbg=3, num_env_bits_fixfix=1,
        // freq_res_mode=2 (duration-dependent default).
        let bytes = build_aspx_config_bits(1, 5, 2, 1, true, false, true, 3, 1, 2);
        let mut br = BitReader::new(&bytes);
        let cfg = parse_aspx_config(&mut br).unwrap();
        assert_eq!(cfg.quant_mode_env, AspxQuantStep::Coarse);
        assert_eq!(cfg.start_freq, 5);
        assert_eq!(cfg.stop_freq, 2);
        assert_eq!(cfg.master_freq_scale, AspxMasterFreqScale::HighRes);
        assert!(cfg.interpolation);
        assert!(!cfg.preflat);
        assert!(cfg.limiter);
        assert_eq!(cfg.noise_sbg, 3);
        assert_eq!(cfg.num_env_bits_fixfix, 1);
        assert_eq!(cfg.freq_res_mode, AspxFreqResMode::DurationDependent);
    }

    #[test]
    fn aspx_config_exactly_15_bits() {
        // Pack a config plus a known sentinel bit right after; verify
        // the parser left the cursor at the 16th bit (index 15).
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1);
        bw.write_u32(0, 3);
        bw.write_u32(0, 2);
        bw.write_u32(0, 1);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_u32(0, 2);
        bw.write_u32(0, 1);
        bw.write_u32(0, 2);
        // Sentinel bit.
        bw.write_bit(true);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let _ = parse_aspx_config(&mut br).unwrap();
        assert_eq!(br.bit_position(), 15);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_config_helpers() {
        let cfg = AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::Signalled,
        };
        assert_eq!(cfg.num_noise_sbgroups(), 1);
        assert_eq!(cfg.fixfix_tmp_num_env_bits(), 1);
        assert!(cfg.signals_freq_res());
        let cfg2 = AspxConfig {
            noise_sbg: 3,
            num_env_bits_fixfix: 1,
            freq_res_mode: AspxFreqResMode::High,
            ..cfg
        };
        assert_eq!(cfg2.num_noise_sbgroups(), 4);
        assert_eq!(cfg2.fixfix_tmp_num_env_bits(), 2);
        assert!(!cfg2.signals_freq_res());
    }

    #[test]
    fn companding_control_mono() {
        // num_chan = 1: no sync_flag; one b_compand_on; no avg when on.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_compand_on[0] = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 1).unwrap();
        assert!(cc.sync_flag.is_none());
        assert_eq!(cc.compand_on, vec![true]);
        assert!(cc.compand_avg.is_none());
    }

    #[test]
    fn companding_control_stereo_sync_needs_avg() {
        // num_chan = 2, sync_flag = 1 -> single compand_on; set it 0
        // (companding off) -> needs avg.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // sync_flag
        bw.write_bit(false); // compand_on[0] = 0
        bw.write_bit(true); // compand_avg = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 2).unwrap();
        assert_eq!(cc.sync_flag, Some(true));
        assert_eq!(cc.compand_on, vec![false]);
        assert_eq!(cc.compand_avg, Some(true));
    }

    #[test]
    fn companding_control_stereo_no_sync() {
        // num_chan = 2, sync_flag = 0 -> per-channel flags.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // sync_flag
        bw.write_bit(true); // compand_on[0] = 1
        bw.write_bit(true); // compand_on[1] = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 2).unwrap();
        assert_eq!(cc.sync_flag, Some(false));
        assert_eq!(cc.compand_on, vec![true, true]);
        assert!(cc.compand_avg.is_none());
    }

    #[test]
    fn companding_control_5ch_one_off() {
        // num_chan = 5, sync_flag = 0, channels = [1,1,0,1,1] -> avg
        // required.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // sync
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // ch2 off
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // compand_avg = 0
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 5).unwrap();
        assert_eq!(cc.sync_flag, Some(false));
        assert_eq!(cc.compand_on, vec![true, true, false, true, true]);
        assert_eq!(cc.compand_avg, Some(false));
    }

    // --- aspx_framing tests ------------------------------------------

    /// Build a default `AspxConfig` parameterised on the two fields
    /// `parse_aspx_framing` actually consults.
    fn test_config(num_env_bits_fixfix: u8, freq_res_mode: AspxFreqResMode) -> AspxConfig {
        AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix,
            freq_res_mode,
        }
    }

    #[test]
    fn aspx_int_class_prefix_code_matches_table_126() {
        // 0b0 -> FIXFIX
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(false);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::FixFix);
        assert_eq!(br.bit_position(), 1);

        // 0b10 -> FIXVAR
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(true);
            bw.write_bit(false);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::FixVar);
        assert_eq!(br.bit_position(), 2);

        // 0b110 -> VARFIX
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(true);
            bw.write_bit(true);
            bw.write_bit(false);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::VarFix);
        assert_eq!(br.bit_position(), 3);

        // 0b111 -> VARVAR
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(true);
            bw.write_bit(true);
            bw.write_bit(true);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::VarVar);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn aspx_framing_fixfix_signalled_freq_res() {
        // aspx_num_env_bits_fixfix = 1 (envbits = 2), freq_res_mode =
        // Signalled. Bits: int_class=0 (FIXFIX, 1 bit); tmp_num_env=10
        // (2 bits, value 2 -> num_env = 1 << 2 = 4); aspx_freq_res[0]=1
        // (1 bit). Total = 4 bits. Sentinel follows.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // FIXFIX
        bw.write_u32(2, 2); // tmp_num_env = 2
        bw.write_bit(true); // freq_res[0]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(1, AspxFreqResMode::Signalled);
        let f = parse_aspx_framing(&mut br, &cfg, true, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::FixFix);
        assert_eq!(f.num_env, 4);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.freq_res, vec![true]);
        assert_eq!(f.var_bord_left, None);
        assert_eq!(f.var_bord_right, None);
        assert_eq!(f.num_rel_left, 0);
        assert_eq!(f.num_rel_right, 0);
        assert!(f.rel_bord_left.is_empty());
        assert!(f.rel_bord_right.is_empty());
        assert_eq!(f.tsg_ptr, None);
        // Cursor lands on the sentinel bit.
        assert_eq!(br.bit_position(), 4);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_fixfix_narrow_envbits_no_freq_res() {
        // aspx_num_env_bits_fixfix = 0 (envbits = 1), freq_res_mode =
        // High (NOT signalled). Bits: FIXFIX=0 (1 bit), tmp_num_env=1
        // (1 bit, num_env = 2). Total = 2 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // FIXFIX
        bw.write_bit(true); // tmp_num_env = 1 -> num_env = 2
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::High);
        let f = parse_aspx_framing(&mut br, &cfg, false, true).unwrap();
        assert_eq!(f.int_class, AspxIntClass::FixFix);
        assert_eq!(f.num_env, 2);
        assert_eq!(f.num_noise, 2);
        assert!(f.freq_res.is_empty());
        assert_eq!(br.bit_position(), 2);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_fixvar_ts_over_8_with_two_rel() {
        // FIXVAR prefix = 0b10 (2 bits). num_aspx_timeslots > 8 so
        // note-1 fields are 2 bits. var_bord_right = 0b11 (2 bits),
        // num_rel_right = 2 (2 bits), rel_bord_right = [0b01, 0b10]
        // (2 bits each). Then branch ends -> num_env = 0 + 2 + 1 = 3,
        // ptr_bits = ceil(log2(5)) = 3. Then since freq_res_mode is
        // Signalled, read 3 freq_res bits. tsg_ptr = 0b101, freq_res =
        // [1,0,1]. Total = 2+2+2+2+2+3+3 = 16 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(false); // FIXVAR
        bw.write_u32(0b11, 2); // var_bord_right
        bw.write_u32(2, 2); // num_rel_right
        bw.write_u32(0b01, 2); // rel_bord_right[0]
        bw.write_u32(0b10, 2); // rel_bord_right[1]
        bw.write_u32(0b101, 3); // tsg_ptr
        bw.write_bit(true); // freq_res[0]
        bw.write_bit(false); // freq_res[1]
        bw.write_bit(true); // freq_res[2]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Signalled);
        let f = parse_aspx_framing(&mut br, &cfg, true, true).unwrap();
        assert_eq!(f.int_class, AspxIntClass::FixVar);
        assert_eq!(f.num_env, 3);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.var_bord_right, Some(0b11));
        assert_eq!(f.num_rel_right, 2);
        assert_eq!(f.rel_bord_right, vec![0b01, 0b10]);
        assert_eq!(f.num_rel_left, 0);
        assert!(f.rel_bord_left.is_empty());
        assert_eq!(f.var_bord_left, None);
        assert_eq!(f.tsg_ptr, Some(0b101));
        assert_eq!(f.freq_res, vec![true, false, true]);
        assert_eq!(br.bit_position(), 16);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_varfix_iframe_ts_short() {
        // VARFIX prefix = 0b110 (3 bits). b_iframe = true so
        // var_bord_left is read (2 bits). num_aspx_timeslots <= 8 so
        // note-1 fields are 1 bit. num_rel_left = 1 (1 bit),
        // rel_bord_left = [1] (1 bit). Branch ends -> num_env = 2,
        // ptr_bits = ceil(log2(4)) = 2. freq_res_mode = Low (not
        // signalled) so no freq_res read. Total bits = 3+2+1+1+2 = 9.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // VARFIX
        bw.write_u32(0b10, 2); // var_bord_left
        bw.write_u32(1, 1); // num_rel_left
        bw.write_u32(1, 1); // rel_bord_left[0]
        bw.write_u32(0b11, 2); // tsg_ptr
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Low);
        let f = parse_aspx_framing(&mut br, &cfg, true, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::VarFix);
        assert_eq!(f.num_env, 2);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.var_bord_left, Some(0b10));
        assert_eq!(f.num_rel_left, 1);
        assert_eq!(f.rel_bord_left, vec![1]);
        assert_eq!(f.num_rel_right, 0);
        assert!(f.rel_bord_right.is_empty());
        assert_eq!(f.var_bord_right, None);
        assert_eq!(f.tsg_ptr, Some(0b11));
        assert!(f.freq_res.is_empty());
        assert_eq!(br.bit_position(), 9);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_varfix_non_iframe_omits_var_bord_left() {
        // Same as above but b_iframe = false -> var_bord_left is NOT
        // present in the bitstream. Bits: VARFIX=0b110 (3),
        // num_rel_left=0 (1), tsg_ptr on num_env=1 -> ptr_bits =
        // ceil(log2(3)) = 2. Total = 3+1+2 = 6 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // VARFIX
        bw.write_u32(0, 1); // num_rel_left = 0
        bw.write_u32(0b10, 2); // tsg_ptr
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Low);
        let f = parse_aspx_framing(&mut br, &cfg, false, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::VarFix);
        assert_eq!(f.num_env, 1);
        assert_eq!(f.num_noise, 1);
        assert_eq!(f.var_bord_left, None);
        assert_eq!(f.num_rel_left, 0);
        assert_eq!(f.tsg_ptr, Some(0b10));
        assert_eq!(br.bit_position(), 6);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_varvar_iframe_symmetric() {
        // VARVAR prefix = 0b111 (3 bits). b_iframe = true.
        // num_aspx_timeslots <= 8 (1-bit note-1 fields).
        //   var_bord_left = 0b01 (2)
        //   num_rel_left = 1 (1) -> rel_bord_left = [0] (1)
        //   var_bord_right = 0b11 (2)
        //   num_rel_right = 1 (1) -> rel_bord_right = [1] (1)
        // num_env = 1+1+1 = 3, ptr_bits = ceil(log2(5)) = 3, tsg_ptr =
        // 0b100. freq_res_mode = Signalled -> 3 freq_res bits.
        // Total = 3 + 2 + 1 + 1 + 2 + 1 + 1 + 3 + 3 = 17 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(true); // VARVAR
        bw.write_u32(0b01, 2); // var_bord_left
        bw.write_u32(1, 1); // num_rel_left
        bw.write_u32(0, 1); // rel_bord_left[0]
        bw.write_u32(0b11, 2); // var_bord_right
        bw.write_u32(1, 1); // num_rel_right
        bw.write_u32(1, 1); // rel_bord_right[0]
        bw.write_u32(0b100, 3); // tsg_ptr
        bw.write_bit(false); // freq_res[0]
        bw.write_bit(true); // freq_res[1]
        bw.write_bit(false); // freq_res[2]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Signalled);
        let f = parse_aspx_framing(&mut br, &cfg, true, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::VarVar);
        assert_eq!(f.num_env, 3);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.var_bord_left, Some(0b01));
        assert_eq!(f.num_rel_left, 1);
        assert_eq!(f.rel_bord_left, vec![0]);
        assert_eq!(f.var_bord_right, Some(0b11));
        assert_eq!(f.num_rel_right, 1);
        assert_eq!(f.rel_bord_right, vec![1]);
        assert_eq!(f.tsg_ptr, Some(0b100));
        assert_eq!(f.freq_res, vec![false, true, false]);
        assert_eq!(br.bit_position(), 17);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn ceil_log2_matches_spec_ptr_bits_formula() {
        // Spec: ptr_bits = ceil(log(num_env + 2) / log(2)).
        // num_env = 1 -> log2(3) = ~1.58 -> 2
        // num_env = 2 -> log2(4) = 2    -> 2
        // num_env = 3 -> log2(5) = ~2.32 -> 3
        // num_env = 4 -> log2(6) = ~2.58 -> 3
        // num_env = 5 -> log2(7) = ~2.81 -> 3
        assert_eq!(ceil_log2(1 + 2), 2);
        assert_eq!(ceil_log2(2 + 2), 2);
        assert_eq!(ceil_log2(3 + 2), 3);
        assert_eq!(ceil_log2(4 + 2), 3);
        assert_eq!(ceil_log2(5 + 2), 3);
    }

    #[test]
    fn num_qmf_timeslots_matches_table_189() {
        assert_eq!(num_qmf_timeslots(2048), 32);
        assert_eq!(num_qmf_timeslots(1920), 30);
        assert_eq!(num_qmf_timeslots(1536), 24);
        assert_eq!(num_qmf_timeslots(1024), 16);
        assert_eq!(num_qmf_timeslots(960), 15);
        assert_eq!(num_qmf_timeslots(768), 12);
        assert_eq!(num_qmf_timeslots(512), 8);
        assert_eq!(num_qmf_timeslots(384), 6);
    }

    #[test]
    fn num_ts_in_ats_matches_table_192() {
        // >= 1536 -> 2, else 1.
        assert_eq!(num_ts_in_ats(2048), 2);
        assert_eq!(num_ts_in_ats(1920), 2);
        assert_eq!(num_ts_in_ats(1536), 2);
        assert_eq!(num_ts_in_ats(1024), 1);
        assert_eq!(num_ts_in_ats(960), 1);
        assert_eq!(num_ts_in_ats(768), 1);
        assert_eq!(num_ts_in_ats(512), 1);
        assert_eq!(num_ts_in_ats(384), 1);
    }

    #[test]
    fn aspx_delta_dir_reads_num_env_plus_num_noise_bits() {
        // Framing with num_env=3, num_noise=2 -> 5 bits total.
        let framing = AspxFraming {
            int_class: AspxIntClass::VarVar,
            num_env: 3,
            num_noise: 2,
            freq_res: Vec::new(),
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: Vec::new(),
            rel_bord_right: Vec::new(),
            tsg_ptr: None,
        };
        // Write: sig = [1,0,1], noise = [0,1], sentinel.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let dd = parse_aspx_delta_dir(&mut br, &framing).unwrap();
        assert_eq!(dd.sig_delta_dir, vec![true, false, true]);
        assert_eq!(dd.noise_delta_dir, vec![false, true]);
        assert_eq!(br.bit_position(), 5);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_delta_dir_single_env_single_noise() {
        // Minimal case: num_env=1, num_noise=1 -> 2 bits.
        let framing = AspxFraming {
            int_class: AspxIntClass::FixFix,
            num_env: 1,
            num_noise: 1,
            freq_res: Vec::new(),
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: Vec::new(),
            rel_bord_right: Vec::new(),
            tsg_ptr: None,
        };
        let mut bw = BitWriter::new();
        bw.write_bit(false); // sig[0] = 0
        bw.write_bit(true); // noise[0] = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let dd = parse_aspx_delta_dir(&mut br, &framing).unwrap();
        assert_eq!(dd.sig_delta_dir, vec![false]);
        assert_eq!(dd.noise_delta_dir, vec![true]);
    }

    #[test]
    fn aspx_hfgen_iwc_1ch_all_gates_off() {
        // num_sbg_noise=2 -> 4 bits (tna_mode[0..=1], 2 bits each).
        // ah_present=0, fic_present=0, tic_present=0 — 3 more bits.
        // Total = 7 bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0b01, 2); // tna_mode[0] = 1
        bw.write_u32(0b10, 2); // tna_mode[1] = 2
        bw.write_bit(false); // ah_present
        bw.write_bit(false); // fic_present
        bw.write_bit(false); // tic_present
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_1ch(&mut br, 2, 3, 16).unwrap();
        assert_eq!(h.tna_mode, vec![1, 2]);
        assert!(!h.ah_present);
        assert!(!h.fic_present);
        assert!(!h.tic_present);
        // Gated-off vectors stay all-zero at their declared length.
        assert_eq!(h.add_harmonic, vec![false; 3]);
        assert_eq!(h.fic_used_in_sfb, vec![false; 3]);
        assert_eq!(h.tic_used_in_slot, vec![false; 16]);
        assert_eq!(br.bit_position(), 7);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_hfgen_iwc_1ch_all_gates_on() {
        // num_sbg_noise=1, num_sbg_sig_highres=2, num_aspx_timeslots=3.
        // tna_mode[0] = 3 (2b)
        // ah_present=1 + add_harmonic=[1,0] (3b)
        // fic_present=1 + fic_used_in_sfb=[0,1] (3b)
        // tic_present=1 + tic_used_in_slot=[1,0,1] (4b)
        // Total = 2 + 3 + 3 + 4 = 12 bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0b11, 2);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_1ch(&mut br, 1, 2, 3).unwrap();
        assert_eq!(h.tna_mode, vec![3]);
        assert!(h.ah_present);
        assert_eq!(h.add_harmonic, vec![true, false]);
        assert!(h.fic_present);
        assert_eq!(h.fic_used_in_sfb, vec![false, true]);
        assert!(h.tic_present);
        assert_eq!(h.tic_used_in_slot, vec![true, false, true]);
        assert_eq!(br.bit_position(), 12);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn num_aspx_timeslots_matches_pseudocode_75a() {
        // num_aspx_timeslots = num_qmf_timeslots / num_ts_in_ats
        assert_eq!(num_aspx_timeslots(2048), 16); // 32 / 2
        assert_eq!(num_aspx_timeslots(1920), 15); // 30 / 2
        assert_eq!(num_aspx_timeslots(1536), 12); // 24 / 2
        assert_eq!(num_aspx_timeslots(1024), 16); // 16 / 1
        assert_eq!(num_aspx_timeslots(960), 15); // 15 / 1
        assert_eq!(num_aspx_timeslots(768), 12); // 12 / 1
        assert_eq!(num_aspx_timeslots(512), 8); // 8 / 1
        assert_eq!(num_aspx_timeslots(384), 6); // 6 / 1
    }
}
