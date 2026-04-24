//! AC-4 Advanced Spectral Extension (A-SPX) parameter parsing.
//!
//! A-SPX is AC-4's spectral-bandwidth-extension tool (ETSI TS 103 190-1
//! §4.2.12 / §4.3.10 / §5.7.6): the core codec carries the low band and
//! A-SPX reconstructs the high band in the QMF domain from a compact
//! envelope-energy sidecar and a small set of control parameters.
//!
//! This module implements the **configuration block** `aspx_config()`
//! (Table 50, §4.2.12.1) plus `companding_control()` (Table 49,
//! §4.2.11). Together those are everything the bitstream carries in the
//! I-frame header for an A-SPX substream **before** the envelope /
//! noise Huffman payload (`aspx_data_1ch()` / `aspx_data_2ch()`) — i.e.
//! enough to classify a stream, expose its configuration for downstream
//! tooling, and determine how many bits the envelope payload will be
//! parameterised by.
//!
//! We deliberately stop short of:
//!
//! * `aspx_framing()` (§4.2.12.4) — bit-count is dynamic, driven by
//!   `aspx_num_env_bits_fixfix`, `aspx_freq_res_mode`, and the interval
//!   class.
//! * `aspx_delta_dir()`, `aspx_hfgen_iwc_*()`, `aspx_ec_data()` and the
//!   A-SPX Huffman tables in Annex A.2 — envelope and noise scale
//!   factors are Huffman-coded with dedicated codebooks (different ones
//!   per delta direction, quantization step, signal vs noise etc.) and
//!   would double the crate's line count. See the module-level roadmap
//!   in `lib.rs`.
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
}
