//! AC-4 Audio Spectral Frontend (ASF) substream baseline.
//!
//! This module implements the framing of an `ac4_substream()` element
//! (ETSI TS 103 190-1 §4.3.4) and the outer shell of the `audio_data()`
//! dispatcher (§4.2.5) for the mono and stereo channel modes. It parses
//! the top-level codec-mode flags, `asf_transform_info()` (§4.2.8.1 /
//! §4.3.6.1), and `asf_psy_info()` (§4.2.8.2 / §4.3.6.2) so the decoder
//! can describe which tools each substream uses (`SIMPLE` / `ASPX` /
//! `ASPX_ACPL_*`), what transform length is in play, how many MDCT
//! windows are present, the window grouping, and the per-group
//! `max_sfb`. It stops short of the actual spectral-coefficient decoding,
//! Huffman tables, and MDCT synthesis.
//!
//! What we deliberately do **not** do yet (and why):
//!
//! * `asf_section_data`, `asf_spectral_data`, `asf_scalefac_data`,
//!   `asf_snf_data` — all Huffman-driven; Huffman tables (§5.1) have
//!   not been transcribed yet.
//! * `ssf_data` (Speech Spectral Frontend) — gated by the same Huffman
//!   / arithmetic-coded layer.
//! * A-SPX (`aspx_config`, `aspx_data_*`) and A-CPL. These are the
//!   bandwidth-extension and inter-channel-coupling tools; each carries
//!   its own Huffman suites.
//!
//! Those components remain marked TODO and the substream walker simply
//! consumes opaque bits from the `audio_size` budget after the outer
//! `asf_transform_info()` is parsed. The decoder continues to emit
//! silence until the tools are filled in.
//!
//! Nothing in here panics on malformed input — every read is
//! result-propagated.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::asf_data;
use crate::aspx;
use crate::sfb_offset;
use crate::tables;
use crate::toc::variable_bits;

/// `mono_codec_mode` values (Table 93).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoCodecMode {
    Simple,
    Aspx,
}

impl MonoCodecMode {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::Simple
        } else {
            Self::Aspx
        }
    }
}

/// `stereo_codec_mode` values (Table 95).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoCodecMode {
    Simple,
    Aspx,
    AspxAcpl1,
    AspxAcpl2,
}

impl StereoCodecMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b11 {
            0 => Self::Simple,
            1 => Self::Aspx,
            2 => Self::AspxAcpl1,
            _ => Self::AspxAcpl2,
        }
    }
}

/// `spec_frontend` values (Table 94).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecFrontend {
    /// Audio Spectral Frontend — MDCT-based.
    Asf,
    /// Speech Spectral Frontend — arithmetic-coded MDCT with LPC envelope.
    Ssf,
}

impl SpecFrontend {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::Asf
        } else {
            Self::Ssf
        }
    }
}

/// ASF transform-length info (`asf_transform_info()`, §4.2.8.1 + §4.3.6.1).
#[derive(Debug, Clone, Copy, Default)]
pub struct AsfTransformInfo {
    /// `b_long_frame` — only meaningful when `frame_len_base >= 1536`.
    pub b_long_frame: bool,
    /// `transf_length[0]` / `transf_length[1]` — 2-bit indices. For a
    /// long frame the two are implicitly equal and hold the long-frame
    /// transform length from Table 99. For `frame_len_base < 1536`
    /// there's only a single `transf_length` field; we mirror it into
    /// both slots.
    pub transf_length: [u32; 2],
    /// Resolved transform length in samples for `transf_length[0]`.
    pub transform_length_0: u32,
    /// Resolved transform length for `transf_length[1]`.
    pub transform_length_1: u32,
}

/// Parsed `asf_psy_info()` (ETSI TS 103 190-1 §4.2.8.2 + §4.3.6.2) —
/// the scale-factor-band organisation for one or two transform-length
/// windows.
#[derive(Debug, Clone, Default)]
pub struct AsfPsyInfo {
    /// Derived from the spec: `transf_length[0]` differed from
    /// `transf_length[1]`, triggering the dual-window branch.
    pub b_different_framing: bool,
    /// `max_sfb[0]` or `max_sfb_side[0]` depending on `b_side_limited`.
    pub max_sfb_0: u32,
    /// `max_sfb_side[0]` when `b_dual_maxsfb` is set (else 0).
    pub max_sfb_side_0: u32,
    /// `max_sfb[1]` — only populated when `b_different_framing`.
    pub max_sfb_1: u32,
    /// `max_sfb_side[1]` — only populated when `b_different_framing`
    /// and `b_dual_maxsfb`.
    pub max_sfb_side_1: u32,
    /// Raw `scale_factor_grouping[i]` bits.
    pub scale_factor_grouping: Vec<u8>,
    /// Derived: `num_windows` (1 for long frames).
    pub num_windows: u32,
    /// Derived: `num_window_groups`.
    pub num_window_groups: u32,
}

/// Table 109 row lookup — `n_grp_bits` for `frame_len_base ≥ 1536` and
/// `b_long_frame == 0`, keyed by `(transf_length[0], transf_length[1])`.
pub fn n_grp_bits_ge_1536(tl0: u32, tl1: u32) -> u32 {
    match (tl0 & 0b11, tl1 & 0b11) {
        (0, 0) => 15,
        (0, 1) => 10,
        (0, 2) => 8,
        (0, 3) => 7,
        (1, 0) => 10,
        (1, 1) => 7,
        (1, 2) => 4,
        (1, 3) => 3,
        (2, 0) => 8,
        (2, 1) => 4,
        (2, 2) => 3,
        (2, 3) => 1,
        (3, 0) => 7,
        (3, 1) => 3,
        (3, 2) => 1,
        (3, 3) => 1,
        _ => 0,
    }
}

/// Table 110 row lookup — `n_grp_bits` for `frame_len_base < 1536`,
/// keyed by `(frame_len_base, transf_length)`. Returns 0 for unknown
/// combinations.
pub fn n_grp_bits_lt_1536(frame_len_base: u32, tl: u32) -> u32 {
    match (frame_len_base, tl & 0b11) {
        (1024, 0) | (960, 0) | (768, 0) => 7,
        (1024, 1) | (960, 1) | (768, 1) => 3,
        (1024, 2) | (960, 2) | (768, 2) => 1,
        (1024, 3) | (960, 3) | (768, 3) => 0,
        (512, 0) | (384, 0) => 3,
        (512, 1) | (384, 1) => 1,
        (512, 2) | (384, 2) => 0,
        _ => 0,
    }
}

/// Parse `asf_psy_info(b_dual_maxsfb, b_side_limited)` at the current
/// position. `transform_info` is the previously parsed
/// `asf_transform_info()` result; `frame_len_base` the TOC's
/// `frame_length` at the base rate.
///
/// Returns the scale-factor-band shape plus the parsed
/// `scale_factor_grouping[i]` bits. Derives `num_windows` /
/// `num_window_groups` per Pseudocode 3 in §4.3.6.2.6.
pub fn parse_asf_psy_info(
    br: &mut BitReader<'_>,
    transform_info: &AsfTransformInfo,
    frame_len_base: u32,
    b_dual_maxsfb: bool,
    b_side_limited: bool,
) -> Result<AsfPsyInfo> {
    // b_different_framing is derived, not coded.
    let b_different_framing = frame_len_base >= 1536
        && !transform_info.b_long_frame
        && transform_info.transf_length[0] != transform_info.transf_length[1];
    let mut info = AsfPsyInfo {
        b_different_framing,
        ..AsfPsyInfo::default()
    };

    // Resolve n_msfb_bits / n_side_bits from Table 106 for the primary
    // transform length.
    let (nm0, ns0, _nml0) = tables::n_msfb_bits_48(transform_info.transform_length_0)
        .ok_or_else(|| Error::invalid("ac4: asf_psy_info: unsupported transform_length[0]"))?;

    if b_side_limited {
        info.max_sfb_0 = br.read_u32(ns0)?;
    } else {
        info.max_sfb_0 = br.read_u32(nm0)?;
        if b_dual_maxsfb {
            info.max_sfb_side_0 = br.read_u32(nm0)?;
        }
    }

    if info.b_different_framing {
        let (nm1, ns1, _) = tables::n_msfb_bits_48(transform_info.transform_length_1)
            .ok_or_else(|| Error::invalid("ac4: asf_psy_info: unsupported transform_length[1]"))?;
        if b_side_limited {
            info.max_sfb_1 = br.read_u32(ns1)?;
        } else {
            info.max_sfb_1 = br.read_u32(nm1)?;
            if b_dual_maxsfb {
                info.max_sfb_side_1 = br.read_u32(nm1)?;
            }
        }
    }

    // Determine n_grp_bits per Table 109 / 110 and spec §4.3.6.2.4.
    let n_grp_bits = if transform_info.b_long_frame {
        // Long frames: no grouping bits.
        0
    } else if frame_len_base >= 1536 {
        n_grp_bits_ge_1536(
            transform_info.transf_length[0],
            transform_info.transf_length[1],
        )
    } else {
        n_grp_bits_lt_1536(frame_len_base, transform_info.transf_length[0])
    };

    // Read scale_factor_grouping[i] bits.
    let mut grouping = Vec::with_capacity(n_grp_bits as usize);
    for _ in 0..n_grp_bits {
        grouping.push(br.read_u32(1)? as u8);
    }
    info.scale_factor_grouping = grouping;

    // Derive num_windows / num_window_groups per Pseudocode 3 —
    // simplified: for long frames it's 1/1; for non-long with equal
    // transform lengths, num_windows = n_grp_bits + 1.
    if transform_info.b_long_frame {
        info.num_windows = 1;
        info.num_window_groups = 1;
    } else {
        info.num_windows = n_grp_bits + 1;
        info.num_window_groups = 1;
        // For the equal-transform-length case each grouping==0 bit
        // starts a new group. For b_different_framing the pseudocode
        // inserts an unconditional boundary at the half-frame mark.
        for &b in &info.scale_factor_grouping {
            if b == 0 {
                info.num_window_groups += 1;
            }
        }
        // Safety cap — malformed streams shouldn't explode.
        if info.num_windows == 0 {
            info.num_windows = 1;
        }
    }

    Ok(info)
}

/// Per-substream tool summary — what the decoder can learn by walking
/// the outer layers of `audio_data()` without touching Huffman tables.
#[derive(Debug, Clone, Default)]
pub struct SubstreamTools {
    /// Channel mode that drove the `audio_data()` switch. Copied from
    /// the parent `ac4_substream_info()`.
    pub channel_mode_channels: u16,
    /// Mono codec mode for channel_mode == 0.
    pub mono_mode: Option<MonoCodecMode>,
    /// Stereo codec mode for channel_mode == 1.
    pub stereo_mode: Option<StereoCodecMode>,
    /// `spec_frontend` for the primary (mid / left / mono) channel.
    pub spec_frontend_primary: Option<SpecFrontend>,
    /// `spec_frontend` for the secondary (side / right) channel if the
    /// substream is dual-MDCT-frontend.
    pub spec_frontend_secondary: Option<SpecFrontend>,
    /// Parsed `asf_transform_info()` for the primary channel when
    /// spec_frontend == ASF.
    pub transform_info_primary: Option<AsfTransformInfo>,
    /// Parsed `asf_transform_info()` for the secondary channel.
    pub transform_info_secondary: Option<AsfTransformInfo>,
    /// Parsed `asf_psy_info()` for the primary channel.
    pub psy_info_primary: Option<AsfPsyInfo>,
    /// Parsed `asf_psy_info()` for the secondary channel.
    pub psy_info_secondary: Option<AsfPsyInfo>,
    /// `b_enable_mdct_stereo_proc` — stereo MDCT joint processing flag.
    pub mdct_stereo_proc: bool,
    /// Dequantised + scaled spectral coefficients for the primary
    /// channel (length = `sfb_offset[max_sfb]`). `None` when the
    /// Huffman-driven data path didn't run (non-SIMPLE / non-ASF /
    /// short-frame grouping cases).
    pub scaled_spec_primary: Option<Vec<f32>>,
    /// Dequantised + scaled spectral coefficients for the secondary
    /// (right) channel in a stereo SIMPLE substream. `None` for mono
    /// or non-decoded stereo cases.
    pub scaled_spec_secondary: Option<Vec<f32>>,
    /// Per-scale-factor-band M/S flags (`ms_used[sfb]`). Populated for
    /// `b_enable_mdct_stereo_proc == 1` joint-stereo frames. `None`
    /// otherwise. Length equals the decoded `max_sfb` for the shared
    /// window.
    pub ms_used: Option<Vec<bool>>,
    /// Parsed `aspx_config()` when the substream's codec mode selected
    /// one of the A-SPX paths (mono ASPX or stereo ASPX / ASPX_ACPL_*)
    /// **and** the current frame is an I-frame. `aspx_config` is only
    /// present in I-frames (§4.2.6.1 / §4.2.6.3); predictive frames
    /// inherit the previous I-frame's config. For non-I-frames or for
    /// SIMPLE substreams this stays `None`.
    pub aspx_config: Option<aspx::AspxConfig>,
    /// Parsed `companding_control()` when the substream's codec mode
    /// is one of the ASPX paths. Captured from the bitstream in the
    /// outer `audio_data()` walker.
    pub companding: Option<aspx::CompandingControl>,
    /// Parsed `aspx_framing(0)` for the primary channel. Populated when
    /// the substream's codec mode is one of the ASPX paths and an
    /// `aspx_config` is in scope (either from this frame if it's an
    /// I-frame or carried over from a prior I-frame — we only wire the
    /// I-frame path here so non-I-frame ASPX bitstreams stay at
    /// companding_control until config state plumbing arrives).
    pub aspx_framing_primary: Option<aspx::AspxFraming>,
    /// Parsed `aspx_framing(1)` for the secondary channel in a stereo
    /// ASPX substream. Only present when `aspx_balance == 0` (the
    /// secondary channel has its own framing per Table 52); when
    /// `aspx_balance == 1` the secondary envelope data reuses the
    /// primary's framing.
    pub aspx_framing_secondary: Option<aspx::AspxFraming>,
    /// `aspx_balance` — 1-bit flag from `aspx_data_2ch()` (Table 52).
    /// Present only in stereo ASPX substreams; otherwise `None`.
    pub aspx_balance: Option<bool>,
    /// `aspx_xover_subband_offset` — 3-bit I-frame-sticky field that
    /// leads `aspx_data_1ch` / `aspx_data_2ch` (Tables 51 / 52). Only
    /// populated for I-frames; carries the crossover-subband offset
    /// used to seed `sbx` in §5.7.6.3.1.2.
    pub aspx_xover_subband_offset: Option<u8>,
    /// `aspx_delta_dir(0)` — per-envelope delta-direction bits for the
    /// primary channel (Table 54). Populated when the walker reaches
    /// `aspx_data_1ch() / aspx_data_2ch()` and the framing decoded
    /// successfully.
    pub aspx_delta_dir_primary: Option<aspx::AspxDeltaDir>,
    /// `aspx_delta_dir(1)` — per-envelope delta-direction bits for
    /// the secondary channel in a stereo ASPX substream. `None` for
    /// mono.
    pub aspx_delta_dir_secondary: Option<aspx::AspxDeltaDir>,
    /// `aspx_qmode_env[0]` — the effective envelope quant step for
    /// the primary channel after the `FIXFIX && num_env == 1`
    /// clamp-to-0 override in Tables 51 / 52.
    pub aspx_qmode_env_primary: Option<aspx::AspxQuantStep>,
    /// `aspx_qmode_env[1]` — same, for the secondary channel.
    pub aspx_qmode_env_secondary: Option<aspx::AspxQuantStep>,
    /// Derived A-SPX frequency tables (§5.7.6.3.1). Populated on any
    /// I-frame ASPX substream that parses `aspx_config` plus
    /// `aspx_xover_subband_offset` without hitting a bailout.
    pub aspx_frequency_tables: Option<aspx::AspxFrequencyTables>,
    /// Parsed `aspx_hfgen_iwc_1ch()` for the mono ASPX path (Table 55).
    pub aspx_hfgen_iwc_1ch: Option<aspx::AspxHfgenIwc1Ch>,
    /// Parsed `aspx_hfgen_iwc_2ch()` for the stereo ASPX path (Table 56).
    pub aspx_hfgen_iwc_2ch: Option<aspx::AspxHfgenIwc2Ch>,
    /// `aspx_data_sig[0]` — per-envelope Huffman-decoded signal
    /// envelopes for the primary channel.
    pub aspx_data_sig_primary: Option<Vec<aspx::AspxHuffEnv>>,
    /// `aspx_data_sig[1]` — per-envelope signal envelopes for the
    /// secondary channel in a stereo ASPX substream.
    pub aspx_data_sig_secondary: Option<Vec<aspx::AspxHuffEnv>>,
    /// `aspx_data_noise[0]` — per-envelope Huffman-decoded noise
    /// envelopes for the primary channel.
    pub aspx_data_noise_primary: Option<Vec<aspx::AspxHuffEnv>>,
    /// `aspx_data_noise[1]` — per-envelope noise envelopes for the
    /// secondary channel.
    pub aspx_data_noise_secondary: Option<Vec<aspx::AspxHuffEnv>>,
    /// Parsed `acpl_config_1ch(PARTIAL)` (§4.2.13.1 Table 59) for
    /// stereo `ASPX_ACPL_1` I-frame substreams. `None` for SIMPLE /
    /// mono ASPX / stereo ASPX paths.
    pub acpl_config_1ch_partial: Option<crate::acpl::AcplConfig1ch>,
    /// Parsed `acpl_config_1ch(FULL)` (§4.2.13.1 Table 59) for
    /// stereo `ASPX_ACPL_2` I-frame substreams. `None` otherwise.
    pub acpl_config_1ch_full: Option<crate::acpl::AcplConfig1ch>,
}

/// Result of walking a single `ac4_substream()` payload.
#[derive(Debug, Clone, Default)]
pub struct Ac4SubstreamInfo {
    /// `audio_size_value` in bytes (post variable_bits extension).
    pub audio_size: u32,
    /// Byte position (relative to the start of `ac4_substream()`) where
    /// `audio_data()` begins.
    pub audio_data_offset: u32,
    /// Tool summary — what we parsed from the outer `audio_data()`
    /// layers before bailing to opaque consumption.
    pub tools: SubstreamTools,
}

/// Resolve a `transf_length` index into an actual transform length in
/// samples for a given `frame_len_base` and base sample rate family.
///
/// Covers Tables 99 (long frames), 100 (non-long, 44.1/48 kHz,
/// `frame_len_base >= 1536`) and 103 (non-long, 44.1/48 kHz,
/// `frame_len_base < 1536`) for the base-rate path. 96 kHz and 192 kHz
/// (Tables 101 / 102 / 104 / 105) reach via HSF extension — not wired
/// in this baseline.
pub fn resolve_transf_length(frame_len_base: u32, b_long_frame: bool, idx: u32) -> u32 {
    // Long-frame branch — Table 99, 44.1/48 kHz column.
    if b_long_frame {
        return frame_len_base;
    }
    if frame_len_base >= 1536 {
        // Table 100 — rows are frame_len_base, columns are transf_length[i].
        match (frame_len_base, idx & 0b11) {
            (2048, 0) => 128,
            (2048, 1) => 256,
            (2048, 2) => 512,
            (2048, 3) => 1024,
            (1920, 0) => 120,
            (1920, 1) => 240,
            (1920, 2) => 480,
            (1920, 3) => 960,
            (1536, 0) => 96,
            (1536, 1) => 192,
            (1536, 2) => 384,
            (1536, 3) => 768,
            _ => 0,
        }
    } else {
        // Table 103 — short-ish frames.
        match (frame_len_base, idx & 0b11) {
            (1024, 0) => 128,
            (1024, 1) => 256,
            (1024, 2) => 512,
            (1024, 3) => 1024,
            (960, 0) => 120,
            (960, 1) => 240,
            (960, 2) => 480,
            (960, 3) => 960,
            (768, 0) => 96,
            (768, 1) => 192,
            (768, 2) => 384,
            (768, 3) => 768,
            (512, 0) => 128,
            (512, 1) => 256,
            (512, 2) => 512,
            (384, 0) => 96,
            (384, 1) => 192,
            (384, 2) => 384,
            _ => 0,
        }
    }
}

/// Parse `asf_transform_info()` at the current reader position. The
/// `frame_len_base` comes from the TOC.
pub fn parse_asf_transform_info(
    br: &mut BitReader<'_>,
    frame_len_base: u32,
) -> Result<AsfTransformInfo> {
    // Table 37 / §4.3.6.1.
    if frame_len_base >= 1536 {
        let b_long_frame = br.read_bit()?;
        if b_long_frame {
            // Long-frame transform length equals frame_len_base (at the
            // base sample rate family).
            let tl = resolve_transf_length(frame_len_base, true, 0);
            Ok(AsfTransformInfo {
                b_long_frame: true,
                transf_length: [0, 0],
                transform_length_0: tl,
                transform_length_1: tl,
            })
        } else {
            let t0 = br.read_u32(2)?;
            let t1 = br.read_u32(2)?;
            Ok(AsfTransformInfo {
                b_long_frame: false,
                transf_length: [t0, t1],
                transform_length_0: resolve_transf_length(frame_len_base, false, t0),
                transform_length_1: resolve_transf_length(frame_len_base, false, t1),
            })
        }
    } else {
        let t0 = br.read_u32(2)?;
        let len = resolve_transf_length(frame_len_base, false, t0);
        Ok(AsfTransformInfo {
            b_long_frame: false,
            transf_length: [t0, t0],
            transform_length_0: len,
            transform_length_1: len,
        })
    }
}

/// Parse the outer layers of `audio_data(channel_mode, b_iframe)` for
/// the mono channel mode. Fills in `tools.mono_mode`,
/// `tools.spec_frontend_primary`, and `tools.transform_info_primary`
/// when the mode is SIMPLE/ASPX and the frontend is ASF.
///
/// Returns after parsing `sf_info` — the `sf_data` payload is left
/// untouched for the caller to skip (via `audio_size`).
pub fn parse_mono_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    // §4.2.6.1 single_channel_element(b_iframe):
    //   mono_codec_mode; 1 bit
    //   if (b_iframe && mono_codec_mode == ASPX) { aspx_config(); }
    //   if (mono_codec_mode == SIMPLE) {
    //       mono_data(0);
    //   } else {
    //       companding_control(1);
    //       mono_data(0);
    //       aspx_data_1ch();
    //   }
    let mode_bit = br.read_u32(1)?;
    let mode = MonoCodecMode::from_bit(mode_bit);
    tools.mono_mode = Some(mode);
    // ASPX path (§4.2.6.1): if b_iframe, read aspx_config(); then
    // companding_control(1); then mono_data(0); then aspx_data_1ch().
    // We stop after companding_control + mono_data(0) — the aspx_data
    // Huffman body needs Annex A.2 tables we haven't transcribed.
    if mode != MonoCodecMode::Simple {
        if b_iframe {
            tools.aspx_config = Some(aspx::parse_aspx_config(br)?);
        }
        tools.companding = Some(aspx::parse_companding_control(br, 1)?);
        // Fall through into mono_data(0) — same outer shell as SIMPLE.
    }
    // mono_data(b_lfe=0):
    //   spec_frontend;    1 bit
    //   sf_info(spec_frontend, 0, 0);
    //   sf_data(spec_frontend);
    let sf_bit = br.read_u32(1)?;
    let frontend = SpecFrontend::from_bit(sf_bit);
    tools.spec_frontend_primary = Some(frontend);
    if !b_iframe {
        // The transform-info for non-I-frames still runs on the first
        // I-frame's state but the syntax still reads it — keep parsing.
    }
    if let SpecFrontend::Asf = frontend {
        let ti = parse_asf_transform_info(br, frame_len_base)?;
        tools.transform_info_primary = Some(ti);
        // asf_psy_info(b_dual_maxsfb=0, b_side_limited=0) for a mono
        // channel — per sf_info(spec_frontend, 0, 0).
        if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
            tools.psy_info_primary = Some(psy);
            // For the single-window-group / long-frame case we can now
            // walk asf_section_data + asf_spectral_data +
            // asf_scalefac_data + asf_snf_data and produce scaled
            // spectral coefficients.
            let mut body_ok = false;
            if ti.b_long_frame && tools.psy_info_primary.as_ref().unwrap().num_window_groups == 1 {
                let psy_ref = tools.psy_info_primary.as_ref().unwrap().clone();
                if let Some(scaled) = decode_asf_long_mono_body(br, &ti, &psy_ref) {
                    tools.scaled_spec_primary = Some(scaled);
                    body_ok = true;
                }
            }
            // §4.2.6.1: for the ASPX path, aspx_data_1ch() follows
            // mono_data(0). Parse its leading xover-subband-offset +
            // aspx_framing(0) here. Only runs when:
            //   * we're on an I-frame — both xover-subband-offset and
            //     aspx_config (which drives aspx_framing) are I-frame-
            //     sticky;
            //   * mono_data(0) was fully decoded, so the bitreader sits
            //     at the start of aspx_data_1ch();
            //   * aspx_config was parsed into tools above.
            if mode != MonoCodecMode::Simple && b_iframe && body_ok {
                if let Some(cfg) = tools.aspx_config {
                    let xover = br.read_u32(3)? as u8;
                    tools.aspx_xover_subband_offset = Some(xover);
                    let nats = aspx::num_aspx_timeslots(frame_len_base);
                    let framing = aspx::parse_aspx_framing(br, &cfg, b_iframe, nats > 8)?;
                    // Per Table 51: `aspx_qmode_env[0] = aspx_quant_mode_env`
                    // by default, but FIXFIX with a single envelope drops
                    // back to 0 (1.5 dB / Fine) regardless of the config.
                    let qmode = if matches!(framing.int_class, aspx::AspxIntClass::FixFix)
                        && framing.num_env == 1
                    {
                        aspx::AspxQuantStep::Fine
                    } else {
                        cfg.quant_mode_env
                    };
                    tools.aspx_qmode_env_primary = Some(qmode);
                    // aspx_delta_dir(0) follows aspx_framing(0) per
                    // Table 51 — fully determined by framing.num_env /
                    // num_noise.
                    let dd = aspx::parse_aspx_delta_dir(br, &framing)?;
                    // Derive the §5.7.6.3.1 frequency tables and use
                    // them to parse aspx_hfgen_iwc_1ch() (Table 55)
                    // then two aspx_ec_data() calls (SIGNAL + NOISE)
                    // per the aspx_data_1ch() body. Any derivation
                    // error silently skips this block without clearing
                    // the framing / delta_dir already stashed.
                    if let Ok(tables) = aspx::derive_aspx_frequency_tables(&cfg, xover as u32) {
                        let hfgen = aspx::parse_aspx_hfgen_iwc_1ch(
                            br,
                            cfg.num_noise_sbgroups(),
                            tables.counts.num_sbg_sig_highres,
                            nats,
                        )?;
                        tools.aspx_hfgen_iwc_1ch = Some(hfgen);
                        // Signal envelopes — per Table 51, stereo_mode
                        // is LEVEL (literal `0` in the Table 51 body)
                        // for the 1-channel path.
                        let sig = aspx::parse_aspx_ec_data(
                            br,
                            aspx::AspxDataType::Signal,
                            framing.num_env,
                            &framing.freq_res,
                            qmode,
                            aspx::AspxStereoMode::Level,
                            &dd.sig_delta_dir,
                            tables.counts,
                        )?;
                        tools.aspx_data_sig_primary = Some(sig);
                        // Noise envelopes — per Table 51 freq_res /
                        // qmode are `0` (both ignored by the NOISE
                        // branch of Pseudocode 79).
                        let noise = aspx::parse_aspx_ec_data(
                            br,
                            aspx::AspxDataType::Noise,
                            framing.num_noise,
                            &[],
                            aspx::AspxQuantStep::Fine,
                            aspx::AspxStereoMode::Level,
                            &dd.noise_delta_dir,
                            tables.counts,
                        )?;
                        tools.aspx_data_noise_primary = Some(noise);
                        tools.aspx_frequency_tables = Some(tables);
                    }
                    tools.aspx_delta_dir_primary = Some(dd);
                    tools.aspx_framing_primary = Some(framing);
                }
            }
        }
    }
    Ok(())
}

/// Decode the `sf_data` body for a mono, long-frame, single-window-
/// group ASF substream. Returns the dequantised + scaled spectral
/// coefficients for the frame.
///
/// On any Huffman / bitstream error we return `None` — the caller
/// falls back to silence. Uses the decoder's own max_sfb (from
/// psy_info) rather than num_sfb_48 since the latter is only a cap.
fn decode_asf_long_mono_body(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<Vec<f32>> {
    let tl = ti.transform_length_0;
    let tl_idx = ti.transf_length[0];
    let max_sfb_cap = tables::num_sfb_48(tl)?;
    let max_sfb = psy.max_sfb_0.min(max_sfb_cap);
    if max_sfb == 0 {
        return None;
    }
    let sfbo = sfb_offset::sfb_offset_48(tl)?;
    // asf_section_data.
    let sections = asf_data::parse_asf_section_data(br, tl_idx, tl, max_sfb).ok()?;
    // asf_spectral_data.
    let (qspec, mqi) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    // asf_scalefac_data.
    let sf_gain = asf_data::parse_asf_scalefac_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // asf_snf_data — consume the bits but ignore the output for now
    // (noise fill to be added later).
    let _snf = asf_data::parse_asf_snf_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // Dequantise + scale.
    let scaled = asf_data::dequantise_and_scale(&qspec, &sf_gain, sfbo, max_sfb);
    Some(scaled)
}

/// Parse the outer layers of a stereo `audio_data()` element
/// (channel_pair_element / stereo_data).
///
/// Only the most common case — `stereo_codec_mode == SIMPLE` with the
/// `stereo_data()` body — is walked to the `sf_info` level. Other
/// modes (ASPX / ASPX_ACPL_1 / ASPX_ACPL_2) set the tool enum and stop.
pub fn parse_stereo_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    // §4.2.6.3 channel_pair_element(b_iframe):
    //   stereo_codec_mode;        2 bits
    let mode_bits = br.read_u32(2)?;
    let mode = StereoCodecMode::from_u32(mode_bits);
    tools.stereo_mode = Some(mode);

    if mode != StereoCodecMode::Simple {
        // §4.2.6.3: for b_iframe all ASPX stereo modes start with an
        // aspx_config(); ASPX_ACPL_{1,2} also carry an acpl_config_1ch
        // (PARTIAL / FULL) right after — both are now parsed (the
        // A-CPL config is only 6 bits in PARTIAL mode, 3 in FULL).
        // Each stereo mode then runs companding_control with a
        // mode-specific channel count (2 for ASPX, 1 for ASPX_ACPL_*),
        // which we also surface. The per-channel stereo_data /
        // sf_data / aspx_data / acpl_data bodies remain Huffman-gated
        // and unhandled at the outer-walker level — the dedicated
        // [`crate::acpl`] parsers can be invoked directly once the
        // surrounding QMF / aspx_data state machine is wired.
        if b_iframe {
            tools.aspx_config = Some(aspx::parse_aspx_config(br)?);
            match mode {
                StereoCodecMode::AspxAcpl1 => {
                    tools.acpl_config_1ch_partial = Some(crate::acpl::parse_acpl_config_1ch(
                        br,
                        crate::acpl::Acpl1chMode::Partial,
                    )?);
                    // acpl_data_1ch() body still gated on
                    // surrounding state machine (QMF subband layout
                    // + sb_to_pb mapping not wired) — bail here so we
                    // don't misread companding_control() out of place.
                    return Ok(());
                }
                StereoCodecMode::AspxAcpl2 => {
                    tools.acpl_config_1ch_full = Some(crate::acpl::parse_acpl_config_1ch(
                        br,
                        crate::acpl::Acpl1chMode::Full,
                    )?);
                    return Ok(());
                }
                _ => {}
            }
        }
        let nc = match mode {
            StereoCodecMode::Aspx => 2,
            _ => 1,
        };
        tools.companding = Some(aspx::parse_companding_control(br, nc)?);
        // For `StereoCodecMode::Aspx` (Table 22) the stereo_data()
        // body follows companding_control(2), then aspx_data_2ch()
        // closes the element. We parse stereo_data() with the same
        // shared decoder as SIMPLE, then attempt to read the leading
        // xover-offset + aspx_framing(0)[ + aspx_balance + framing(1)]
        // of aspx_data_2ch(). Only runs when:
        //   * we're on an I-frame, and
        //   * the stereo_data() body decoded cleanly (bitreader is at
        //     the right place).
        if matches!(mode, StereoCodecMode::Aspx) {
            let body_ok = parse_stereo_data_body(br, tools, frame_len_base);
            if b_iframe && body_ok {
                if let Some(cfg) = tools.aspx_config {
                    let xover = br.read_u32(3)? as u8;
                    tools.aspx_xover_subband_offset = Some(xover);
                    let nats = aspx::num_aspx_timeslots(frame_len_base);
                    let framing_ch0 = aspx::parse_aspx_framing(br, &cfg, b_iframe, nats > 8)?;
                    // Per Table 52: `aspx_qmode_env[0] = aspx_qmode_env[1]
                    // = aspx_quant_mode_env` then clamp to 0 on FIXFIX +
                    // num_env == 1.
                    let qmode_ch0 = if matches!(framing_ch0.int_class, aspx::AspxIntClass::FixFix)
                        && framing_ch0.num_env == 1
                    {
                        aspx::AspxQuantStep::Fine
                    } else {
                        cfg.quant_mode_env
                    };
                    tools.aspx_qmode_env_primary = Some(qmode_ch0);
                    // Table 52: aspx_balance (1 bit). If 0, aspx_framing(1)
                    // follows for channel 1; otherwise channel 1 reuses
                    // channel 0's framing.
                    let balance = br.read_bit()?;
                    tools.aspx_balance = Some(balance);
                    let framing_ch1_ref;
                    if !balance {
                        let framing_ch1 = aspx::parse_aspx_framing(br, &cfg, b_iframe, nats > 8)?;
                        // Per Table 52 the ch1 qmode is recomputed
                        // against the ch1 framing (and re-clamped on
                        // FIXFIX + num_env == 1).
                        let qmode_ch1 =
                            if matches!(framing_ch1.int_class, aspx::AspxIntClass::FixFix)
                                && framing_ch1.num_env == 1
                            {
                                aspx::AspxQuantStep::Fine
                            } else {
                                cfg.quant_mode_env
                            };
                        tools.aspx_qmode_env_secondary = Some(qmode_ch1);
                        tools.aspx_framing_secondary = Some(framing_ch1);
                        framing_ch1_ref = tools.aspx_framing_secondary.as_ref();
                    } else {
                        // Shared framing; copy the ch0 qmode across.
                        tools.aspx_qmode_env_secondary = Some(qmode_ch0);
                        framing_ch1_ref = Some(&framing_ch0);
                    }
                    // aspx_delta_dir(0) then aspx_delta_dir(1) per
                    // Table 52.
                    let dd0 = aspx::parse_aspx_delta_dir(br, &framing_ch0)?;
                    let f_ch1 = framing_ch1_ref.unwrap_or(&framing_ch0);
                    let dd1 = aspx::parse_aspx_delta_dir(br, f_ch1)?;
                    // §5.7.6.3.1 derivation feeds aspx_hfgen_iwc_2ch()
                    // (Table 56) then four aspx_ec_data() calls
                    // (ch0/ch1 SIGNAL, ch0/ch1 NOISE) per Table 52.
                    if let Ok(tables) = aspx::derive_aspx_frequency_tables(&cfg, xover as u32) {
                        let hfgen = aspx::parse_aspx_hfgen_iwc_2ch(
                            br,
                            balance,
                            cfg.num_noise_sbgroups(),
                            tables.counts.num_sbg_sig_highres,
                            nats,
                        )?;
                        tools.aspx_hfgen_iwc_2ch = Some(hfgen);
                        let qmode_ch1_effective =
                            tools.aspx_qmode_env_secondary.unwrap_or(qmode_ch0);
                        // ch0 SIGNAL: stereo_mode = LEVEL (Table 52).
                        let sig0 = aspx::parse_aspx_ec_data(
                            br,
                            aspx::AspxDataType::Signal,
                            framing_ch0.num_env,
                            &framing_ch0.freq_res,
                            qmode_ch0,
                            aspx::AspxStereoMode::Level,
                            &dd0.sig_delta_dir,
                            tables.counts,
                        )?;
                        tools.aspx_data_sig_primary = Some(sig0);
                        // ch1 SIGNAL: BALANCE when aspx_balance == 1
                        // else LEVEL (Table 52).
                        let sm_ch1 = if balance {
                            aspx::AspxStereoMode::Balance
                        } else {
                            aspx::AspxStereoMode::Level
                        };
                        let sig1 = aspx::parse_aspx_ec_data(
                            br,
                            aspx::AspxDataType::Signal,
                            f_ch1.num_env,
                            &f_ch1.freq_res,
                            qmode_ch1_effective,
                            sm_ch1,
                            &dd1.sig_delta_dir,
                            tables.counts,
                        )?;
                        tools.aspx_data_sig_secondary = Some(sig1);
                        // ch0 NOISE.
                        let noise0 = aspx::parse_aspx_ec_data(
                            br,
                            aspx::AspxDataType::Noise,
                            framing_ch0.num_noise,
                            &[],
                            aspx::AspxQuantStep::Fine,
                            aspx::AspxStereoMode::Level,
                            &dd0.noise_delta_dir,
                            tables.counts,
                        )?;
                        tools.aspx_data_noise_primary = Some(noise0);
                        // ch1 NOISE mirrors ch1 SIGNAL stereo_mode.
                        let noise1 = aspx::parse_aspx_ec_data(
                            br,
                            aspx::AspxDataType::Noise,
                            f_ch1.num_noise,
                            &[],
                            aspx::AspxQuantStep::Fine,
                            sm_ch1,
                            &dd1.noise_delta_dir,
                            tables.counts,
                        )?;
                        tools.aspx_data_noise_secondary = Some(noise1);
                        tools.aspx_frequency_tables = Some(tables);
                    }
                    tools.aspx_delta_dir_primary = Some(dd0);
                    tools.aspx_delta_dir_secondary = Some(dd1);
                    tools.aspx_framing_primary = Some(framing_ch0);
                }
            }
        }
        return Ok(());
    }

    // SIMPLE path: just `stereo_data()`.
    let _ = parse_stereo_data_body(br, tools, frame_len_base);
    let _ = b_iframe; // reserved for later Huffman-state keying.
    Ok(())
}

/// Walk `stereo_data()` (§4.2.6.3 / Table 22) into `tools`. Returns
/// `true` when the body decoded cleanly enough for the bitreader to sit
/// at the spec's end-of-body position (i.e. the caller is safe to
/// continue reading downstream elements like `aspx_data_2ch()`). A
/// return of `false` means some inner Huffman-gated decode bailed; the
/// bitreader position after the call is indeterminate.
fn parse_stereo_data_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    frame_len_base: u32,
) -> bool {
    // stereo_data():
    //   if (b_enable_mdct_stereo_proc) {
    //       spec_frontend_l = ASF; spec_frontend_r = ASF;
    //       sf_info(ASF, 0, 0);
    //       chparam_info();
    //   } else {
    //       spec_frontend_l;   1 bit
    //       sf_info(spec_frontend_l, 0, 0);
    //       spec_frontend_r;   1 bit
    //       sf_info(spec_frontend_r, 0, 0);
    //   }
    //   sf_data(spec_frontend_l);
    //   sf_data(spec_frontend_r);
    let b_mdct_stereo = match br.read_bit() {
        Ok(v) => v,
        Err(_) => return false,
    };
    tools.mdct_stereo_proc = b_mdct_stereo;
    if b_mdct_stereo {
        tools.spec_frontend_primary = Some(SpecFrontend::Asf);
        tools.spec_frontend_secondary = Some(SpecFrontend::Asf);
        let ti = match parse_asf_transform_info(br, frame_len_base) {
            Ok(t) => t,
            Err(_) => return false,
        };
        tools.transform_info_primary = Some(ti);
        tools.transform_info_secondary = Some(ti);
        // stereo_data() with b_enable_mdct_stereo_proc==1 calls
        // sf_info(ASF, 0, 0) which fires asf_psy_info(0, 0) over the
        // shared transform window.
        let psy = match parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
            Ok(p) => p,
            Err(_) => return false,
        };
        tools.psy_info_primary = Some(psy.clone());
        // Joint-stereo MDCT path: one shared section / psy layout,
        // two residual channel spectra, followed by a per-sfb
        // ms_used[] flag array. Only walk the long-frame, single
        // window-group body; other shapes stay at outer-layer only.
        if ti.b_long_frame && psy.num_window_groups == 1 {
            match decode_asf_long_stereo_joint_body(br, &ti, &psy) {
                Some((l, r, ms)) => {
                    tools.scaled_spec_primary = Some(l);
                    tools.scaled_spec_secondary = Some(r);
                    tools.ms_used = Some(ms);
                    true
                }
                None => false,
            }
        } else {
            // Outer layers parsed but Huffman body not walked — the
            // bitreader isn't at the end of stereo_data(), so downstream
            // elements aren't safe to parse.
            false
        }
    } else {
        let l = match br.read_u32(1) {
            Ok(v) => SpecFrontend::from_bit(v),
            Err(_) => return false,
        };
        tools.spec_frontend_primary = Some(l);
        let mut ti_l: Option<AsfTransformInfo> = None;
        let mut psy_l: Option<AsfPsyInfo> = None;
        if let SpecFrontend::Asf = l {
            let ti = match parse_asf_transform_info(br, frame_len_base) {
                Ok(t) => t,
                Err(_) => return false,
            };
            tools.transform_info_primary = Some(ti);
            ti_l = Some(ti);
            if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                tools.psy_info_primary = Some(psy.clone());
                psy_l = Some(psy);
            } else {
                return false;
            }
        }
        let r = match br.read_u32(1) {
            Ok(v) => SpecFrontend::from_bit(v),
            Err(_) => return false,
        };
        tools.spec_frontend_secondary = Some(r);
        let mut ti_r: Option<AsfTransformInfo> = None;
        let mut psy_r: Option<AsfPsyInfo> = None;
        if let SpecFrontend::Asf = r {
            let ti = match parse_asf_transform_info(br, frame_len_base) {
                Ok(t) => t,
                Err(_) => return false,
            };
            tools.transform_info_secondary = Some(ti);
            ti_r = Some(ti);
            if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                tools.psy_info_secondary = Some(psy.clone());
                psy_r = Some(psy);
            } else {
                return false;
            }
        }
        // Split-MDCT stereo: two independent ASF spectra. Decode each if
        // long-frame + single-window-group. Any malformed body is
        // surfaced as a decode miss (None) rather than a hard error.
        let mut body_ok = true;
        if let (Some(ti), Some(psy)) = (ti_l, psy_l.as_ref()) {
            if ti.b_long_frame && psy.num_window_groups == 1 {
                tools.scaled_spec_primary = decode_asf_long_mono_body(br, &ti, psy);
                if tools.scaled_spec_primary.is_none() {
                    body_ok = false;
                }
            } else {
                body_ok = false;
            }
        }
        if let (Some(ti), Some(psy)) = (ti_r, psy_r.as_ref()) {
            if ti.b_long_frame && psy.num_window_groups == 1 {
                tools.scaled_spec_secondary = decode_asf_long_mono_body(br, &ti, psy);
                if tools.scaled_spec_secondary.is_none() {
                    body_ok = false;
                }
            } else {
                body_ok = false;
            }
        }
        body_ok
    }
}

/// Decode the `sf_data` body for a stereo, long-frame, single-window-
/// group ASF substream with `b_enable_mdct_stereo_proc == 1` (joint
/// MDCT processing — §7.4 / §7.5).
///
/// Layout inferred from the spec:
///   - shared asf_section_data (one section partition drives both
///     channels' codebook assignment).
///   - two asf_spectral_data bodies (residuals for channel L and R,
///     which are actually M and S when ms_used is set).
///   - shared asf_scalefac_data (the scalefactors are shared so the
///     joint quant step is uniform across both channels).
///   - per-sfb ms_used[sfb] flag — one bit per active band.
///   - asf_snf_data consumed but not injected.
///
/// Returns `(left_scaled, right_scaled, ms_used)` on success.
fn decode_asf_long_stereo_joint_body(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
    let tl = ti.transform_length_0;
    let tl_idx = ti.transf_length[0];
    let max_sfb_cap = tables::num_sfb_48(tl)?;
    let max_sfb = psy.max_sfb_0.min(max_sfb_cap);
    if max_sfb == 0 {
        return None;
    }
    let sfbo = sfb_offset::sfb_offset_48(tl)?;
    // Shared asf_section_data.
    let sections = asf_data::parse_asf_section_data(br, tl_idx, tl, max_sfb).ok()?;
    // L / M channel residuals.
    let (q_l, mqi_l) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    // R / S channel residuals.
    let (q_r, mqi_r) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    // Shared scalefactors — compute max_quant_idx as the band-wise max
    // over both channels so the scalefactor DPCM state tracks bands
    // that have any energy at all.
    let mqi: Vec<u32> = mqi_l
        .iter()
        .zip(mqi_r.iter())
        .map(|(a, b)| (*a).max(*b))
        .collect();
    let sf_gain = asf_data::parse_asf_scalefac_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // Per-sfb ms_used flag array. Only active bands (cb != 0 and
    // max_quant_idx > 0) carry a bit per §7.5 Pseudocode 77; other
    // bands default to false.
    let mut ms_used = vec![false; max_sfb as usize];
    for sfb in 0..max_sfb as usize {
        let cb = sections.sfb_cb[sfb];
        if cb == 0 || mqi[sfb] == 0 {
            continue;
        }
        ms_used[sfb] = br.read_bit().ok()?;
    }
    // asf_snf_data consumed but not currently injected.
    let _ = asf_data::parse_asf_snf_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // Dequantise + scale both channels with the shared gain.
    let mut scaled_l = asf_data::dequantise_and_scale(&q_l, &sf_gain, sfbo, max_sfb);
    let mut scaled_r = asf_data::dequantise_and_scale(&q_r, &sf_gain, sfbo, max_sfb);
    // Apply inverse M/S per §7.5: L = M + S, R = M - S for bands with
    // ms_used == true.
    for (sfb, &used) in ms_used.iter().enumerate() {
        if !used {
            continue;
        }
        let a = sfbo[sfb] as usize;
        let b = sfbo[sfb + 1] as usize;
        let bmax = b.min(scaled_l.len()).min(scaled_r.len());
        for k in a..bmax {
            let m = scaled_l[k];
            let s = scaled_r[k];
            scaled_l[k] = m + s;
            scaled_r[k] = m - s;
        }
    }
    Some((scaled_l, scaled_r, ms_used))
}

/// Walk an `ac4_substream()` payload. `substream_bytes` is the slice
/// covering the substream as pointed to by `substream_index_table()`.
/// The walker reads the outer framing and the initial `audio_data()`
/// flags; on success returns an [`Ac4SubstreamInfo`] with the tool
/// summary. On malformed input — which for a stub walker mostly means
/// running off the end of the bitstream — the function returns a
/// best-effort info with whatever was parsed before the error.
///
/// `channels` is the channel count taken from the parent
/// `ac4_substream_info()`. `b_iframe` likewise. `frame_len_base` is
/// `frame_length` at the base sample rate (i.e. the TOC's
/// `frame_length` entry for 48 kHz and 44.1 kHz).
pub fn walk_ac4_substream(
    substream_bytes: &[u8],
    channels: u16,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<Ac4SubstreamInfo> {
    if substream_bytes.is_empty() {
        return Err(Error::invalid("ac4: empty substream"));
    }
    let mut br = BitReader::new(substream_bytes);

    // §4.3.4.1 audio_size_value — 15-bit value, optional variable_bits(7)
    // extension gated by b_more_bits.
    let audio_size_short = br.read_u32(15)?;
    let b_more_bits = br.read_bit()?;
    let audio_size = if b_more_bits {
        audio_size_short + (variable_bits(&mut br, 7)? << 15)
    } else {
        audio_size_short
    };

    // byte_align to enter audio_data().
    br.align_to_byte();
    let audio_data_offset = br.byte_position() as u32;

    // Parse the outer layers of audio_data(channel_mode, b_iframe).
    let mut tools = SubstreamTools {
        channel_mode_channels: channels,
        ..Default::default()
    };
    match channels {
        1 => parse_mono_audio_data_outer(&mut br, &mut tools, b_iframe, frame_len_base)?,
        2 => parse_stereo_audio_data_outer(&mut br, &mut tools, b_iframe, frame_len_base)?,
        // 3.0 / 5.0 / 5.1 / 7.x paths are coding-config-dependent; their
        // outer walkers live behind the same Huffman gate as ASF's
        // spectral data. For the baseline we record the channel count
        // and bail.
        _ => {}
    }

    Ok(Ac4SubstreamInfo {
        audio_size,
        audio_data_offset,
        tools,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[test]
    fn resolve_transf_length_long_frame_table_99() {
        // Long frame at 44.1/48 kHz returns frame_len_base directly.
        assert_eq!(resolve_transf_length(2048, true, 0), 2048);
        assert_eq!(resolve_transf_length(1920, true, 3), 1920);
    }

    #[test]
    fn resolve_transf_length_table_100_rows() {
        assert_eq!(resolve_transf_length(2048, false, 0), 128);
        assert_eq!(resolve_transf_length(2048, false, 1), 256);
        assert_eq!(resolve_transf_length(2048, false, 2), 512);
        assert_eq!(resolve_transf_length(2048, false, 3), 1024);
        assert_eq!(resolve_transf_length(1920, false, 2), 480);
        assert_eq!(resolve_transf_length(1536, false, 1), 192);
    }

    #[test]
    fn resolve_transf_length_table_103_rows() {
        assert_eq!(resolve_transf_length(1024, false, 3), 1024);
        assert_eq!(resolve_transf_length(960, false, 2), 480);
        assert_eq!(resolve_transf_length(512, false, 0), 128);
        assert_eq!(resolve_transf_length(384, false, 2), 384);
    }

    #[test]
    fn asf_transform_info_long_frame_path() {
        // frame_len_base = 1920, b_long_frame = 1.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = parse_asf_transform_info(&mut br, 1920).unwrap();
        assert!(ti.b_long_frame);
        assert_eq!(ti.transform_length_0, 1920);
        assert_eq!(ti.transform_length_1, 1920);
    }

    #[test]
    fn asf_transform_info_short_pair() {
        // frame_len_base = 1920, b_long_frame = 0, transf_length[0] = 2,
        // transf_length[1] = 3. -> transform lengths 480, 960.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_long_frame
        bw.write_u32(2, 2);
        bw.write_u32(3, 2);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = parse_asf_transform_info(&mut br, 1920).unwrap();
        assert!(!ti.b_long_frame);
        assert_eq!(ti.transf_length, [2, 3]);
        assert_eq!(ti.transform_length_0, 480);
        assert_eq!(ti.transform_length_1, 960);
    }

    fn build_mono_substream() -> Vec<u8> {
        // Minimal ac4_substream() body: audio_size = 10, no extension,
        // byte_align, then audio_data() for channel_mode=mono,
        // b_iframe=1: mono_codec_mode=0 (SIMPLE), spec_frontend=0 (ASF),
        // asf_transform_info() for frame_len_base=1920 long-frame.
        let mut bw = BitWriter::new();
        // audio_size_value (15 bits) = 10.
        bw.write_u32(10, 15);
        // b_more_bits = 0.
        bw.write_bit(false);
        // byte_align to enter audio_data.
        bw.align_to_byte();
        // mono_codec_mode = 0 (SIMPLE).
        bw.write_u32(0, 1);
        // mono_data(0) body:
        //   spec_frontend = 0 (ASF).
        bw.write_u32(0, 1);
        //   sf_info(ASF, 0, 0) -> asf_transform_info():
        //     frame_len_base >= 1536 path; b_long_frame = 1.
        bw.write_bit(true);
        //   asf_psy_info(b_dual_maxsfb=0, b_side_limited=0):
        //   long-frame @ 1920 => n_msfb_bits = 6; max_sfb[0] = 40.
        bw.write_u32(40, 6);
        //   asf_section_data / spectral / scalefac / snf left opaque.
        bw.align_to_byte();
        // Pad up to "audio_size"-worth of bytes.
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn walk_mono_asf_substream_extracts_tools() {
        let bytes = build_mono_substream();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        assert_eq!(info.audio_size, 10);
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Simple));
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Asf));
        let ti = info.tools.transform_info_primary.unwrap();
        assert!(ti.b_long_frame);
        assert_eq!(ti.transform_length_0, 1920);
        let psy = info.tools.psy_info_primary.as_ref().unwrap();
        assert_eq!(psy.max_sfb_0, 40);
        assert_eq!(psy.num_windows, 1);
        assert_eq!(psy.num_window_groups, 1);
    }

    fn build_stereo_simple_substream() -> Vec<u8> {
        let mut bw = BitWriter::new();
        // audio_size = 20.
        bw.write_u32(20, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // stereo_codec_mode = SIMPLE (0b00).
        bw.write_u32(0, 2);
        // stereo_data(): b_enable_mdct_stereo_proc = 0.
        bw.write_bit(false);
        // spec_frontend_l = 0 (ASF), long-frame, max_sfb[0] = 30.
        bw.write_u32(0, 1);
        bw.write_bit(true); // b_long_frame
        bw.write_u32(30, 6);
        // spec_frontend_r = 0 (ASF), long-frame, max_sfb[0] = 32.
        bw.write_u32(0, 1);
        bw.write_bit(true);
        bw.write_u32(32, 6);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn walk_stereo_simple_substream_extracts_two_frontends() {
        let bytes = build_stereo_simple_substream();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.audio_size, 20);
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::Simple));
        assert!(!info.tools.mdct_stereo_proc);
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Asf));
        assert_eq!(info.tools.spec_frontend_secondary, Some(SpecFrontend::Asf));
        assert!(info.tools.transform_info_primary.is_some());
        assert!(info.tools.transform_info_secondary.is_some());
        let psy_l = info.tools.psy_info_primary.as_ref().unwrap();
        let psy_r = info.tools.psy_info_secondary.as_ref().unwrap();
        assert_eq!(psy_l.max_sfb_0, 30);
        assert_eq!(psy_r.max_sfb_0, 32);
    }

    #[test]
    fn walk_stereo_mdct_joint_shares_transform_info() {
        let mut bw = BitWriter::new();
        bw.write_u32(20, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0, 2); // SIMPLE
        bw.write_bit(true); // b_enable_mdct_stereo_proc
        bw.write_bit(true); // long-frame
        bw.write_u32(25, 6); // max_sfb[0] = 25.
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert!(info.tools.mdct_stereo_proc);
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Asf));
        assert_eq!(info.tools.spec_frontend_secondary, Some(SpecFrontend::Asf));
        assert_eq!(
            info.tools
                .transform_info_primary
                .unwrap()
                .transform_length_0,
            info.tools
                .transform_info_secondary
                .unwrap()
                .transform_length_0
        );
        let psy = info.tools.psy_info_primary.as_ref().unwrap();
        assert_eq!(psy.max_sfb_0, 25);
    }

    #[test]
    fn asf_psy_info_long_frame_path() {
        // Manually drive parse_asf_psy_info with a known transform_info.
        let ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0, 0],
            transform_length_0: 1920,
            transform_length_1: 1920,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(50, 6); // max_sfb[0]
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        assert!(!psy.b_different_framing);
        assert_eq!(psy.max_sfb_0, 50);
        assert_eq!(psy.num_windows, 1);
        assert_eq!(psy.num_window_groups, 1);
        assert!(psy.scale_factor_grouping.is_empty());
    }

    #[test]
    fn asf_psy_info_short_pair_with_grouping() {
        // frame_len_base = 1920, transf_length = [2, 2] (480 both).
        // From Table 109: n_grp_bits = 3. n_msfb_bits at 480 = 6.
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [2, 2],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(20, 6); // max_sfb[0]
                             // scale_factor_grouping bits (3): 1, 0, 1 => one new group at bit 1.
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        // b_different_framing = false (transf_length[0] == transf_length[1]).
        assert!(!psy.b_different_framing);
        assert_eq!(psy.max_sfb_0, 20);
        assert_eq!(psy.scale_factor_grouping, vec![1, 0, 1]);
        // num_windows = n_grp_bits + 1 = 4; one zero-bit => 2 groups.
        assert_eq!(psy.num_windows, 4);
        assert_eq!(psy.num_window_groups, 2);
    }

    #[test]
    fn asf_psy_info_different_framing_path() {
        // transf_length[0] = 1 (240 @ 1920), transf_length[1] = 2 (480).
        // n_grp_bits (Table 109) = 4.
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [1, 2],
            transform_length_0: 240,
            transform_length_1: 480,
        };
        let mut bw = BitWriter::new();
        // n_msfb_bits at 240 = 5 (Table 106).
        bw.write_u32(10, 5); // max_sfb[0]
                             // b_different_framing triggers:
                             // n_msfb_bits at 480 = 6.
        bw.write_u32(15, 6); // max_sfb[1]
                             // scale_factor_grouping bits (4): 0,0,0,0
        bw.write_u32(0, 4);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        assert!(psy.b_different_framing);
        assert_eq!(psy.max_sfb_0, 10);
        assert_eq!(psy.max_sfb_1, 15);
        assert_eq!(psy.num_windows, 5);
    }

    #[test]
    fn walk_mono_aspx_substream_parses_config_and_companding() {
        // mono_codec_mode = 1 (ASPX) on an I-frame: the walker now
        // consumes aspx_config() (15 bits) and companding_control(1).
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15); // audio_size = 5
        bw.write_bit(false); // b_more_bits = 0
        bw.align_to_byte();
        // mono_codec_mode = 1 (ASPX)
        bw.write_u32(1, 1);
        // aspx_config: quant_mode=0, start=3, stop=1, scale=1, interp=1,
        // preflat=0, limiter=1, noise_sbg=2, num_env_bits_fixfix=0,
        // freq_res_mode=0.
        bw.write_u32(0, 1);
        bw.write_u32(3, 3);
        bw.write_u32(1, 2);
        bw.write_u32(1, 1);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_u32(2, 2);
        bw.write_u32(0, 1);
        bw.write_u32(0, 2);
        // companding_control(1): no sync_flag; compand_on[0] = 1.
        bw.write_bit(true);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        // aspx_config captured.
        let cfg = info.tools.aspx_config.unwrap();
        assert_eq!(cfg.start_freq, 3);
        assert_eq!(cfg.stop_freq, 1);
        assert_eq!(cfg.noise_sbg, 2);
        assert_eq!(cfg.num_noise_sbgroups(), 3);
        assert!(cfg.interpolation);
        assert!(!cfg.preflat);
        assert!(cfg.limiter);
        assert!(cfg.signals_freq_res());
        // companding captured.
        let cc = info.tools.companding.as_ref().unwrap();
        assert_eq!(cc.compand_on, vec![true]);
        assert!(cc.sync_flag.is_none());
        assert!(cc.compand_avg.is_none());
    }

    #[test]
    fn walk_stereo_aspx_substream_parses_config() {
        // stereo_codec_mode = 01 (ASPX) on an I-frame: read
        // aspx_config() then companding_control(2).
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b01, 2); // ASPX
                               // aspx_config: all-zero fields.
        bw.write_u32(0, 15);
        // companding_control(2): sync_flag=0, compand_on=[1,1].
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::Aspx));
        let cfg = info.tools.aspx_config.unwrap();
        assert_eq!(cfg.start_freq, 0);
        let cc = info.tools.companding.as_ref().unwrap();
        assert_eq!(cc.sync_flag, Some(false));
        assert_eq!(cc.compand_on, vec![true, true]);
    }

    #[test]
    fn walk_stereo_aspx_non_iframe_skips_config() {
        // Predictive (b_iframe = 0) ASPX frame: aspx_config not present;
        // we go straight to companding_control.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b01, 2); // ASPX
                               // No aspx_config — go straight to companding_control(2).
        bw.write_bit(true); // sync_flag = 1 -> single compand_on
        bw.write_bit(true); // compand_on[0] = 1
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, false, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::Aspx));
        assert!(info.tools.aspx_config.is_none());
        let cc = info.tools.companding.as_ref().unwrap();
        assert_eq!(cc.sync_flag, Some(true));
        assert_eq!(cc.compand_on, vec![true]);
    }

    /// Write sect_len_incr per §4.2.7.1 Table 80 (section-length
    /// expansion). Duplicated from `decoder.rs` tests for crate-local use.
    fn write_sect_len_incr(bw: &mut BitWriter, sect_len: u32, n_sect_bits: u32, esc: u32) {
        let base = sect_len.saturating_sub(1);
        let k = base / esc;
        let incr = base % esc;
        for _ in 0..k {
            bw.write_u32(esc, n_sect_bits);
        }
        bw.write_u32(incr, n_sect_bits);
    }

    #[test]
    fn walk_mono_aspx_iframe_reads_framing_after_mono_data() {
        // Full-path mono ASPX I-frame substream:
        //   audio_size_value (15), b_more_bits (1), byte_align.
        //   mono_codec_mode = 1 (ASPX)
        //   aspx_config() — 15 bits.
        //   companding_control(1) — 1 bit.
        //   mono_data(0):
        //     spec_frontend = 0 (ASF)
        //     asf_transform_info(): b_long_frame = 1
        //     asf_psy_info(): max_sfb[0] = 10 (6 bits)
        //     sf_data body (section + spectral zeros + scalefac 120 + snf off)
        //   aspx_data_1ch():
        //     aspx_xover_subband_offset = 3 (3 bits)
        //     aspx_framing(0): FIXFIX (1 bit) + tmp_num_env=1 (1 bit since
        //     aspx_num_env_bits_fixfix=0) -> num_env = 2.
        // We set aspx_freq_res_mode to High (not Signalled) so no
        // freq_res bits follow, keeping the fixture small.
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(200, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // mono_codec_mode = ASPX
        bw.write_u32(1, 1);
        // aspx_config(): quant_mode=0, start=0, stop=0, scale=0,
        // interp=0, preflat=0, limiter=0, noise_sbg=0,
        // num_env_bits_fixfix=0, freq_res_mode=3 (High -> not signalled).
        bw.write_u32(0, 1); // quant_mode_env
        bw.write_u32(0, 3); // start_freq
        bw.write_u32(0, 2); // stop_freq
        bw.write_u32(0, 1); // master_freq_scale
        bw.write_bit(false); // interpolation
        bw.write_bit(false); // preflat
        bw.write_bit(false); // limiter
        bw.write_u32(0, 2); // noise_sbg
        bw.write_u32(0, 1); // num_env_bits_fixfix = 0 -> 1-bit tmp_num_env
        bw.write_u32(3, 2); // freq_res_mode = 3 (High)
                            // companding_control(1): compand_on[0] = 1, no avg.
        bw.write_bit(true);
        // mono_data(0): spec_frontend = 0 (ASF).
        bw.write_u32(0, 1);
        // asf_transform_info: b_long_frame = 1.
        bw.write_bit(true);
        // asf_psy_info: max_sfb[0] = 10 in 6 bits.
        bw.write_u32(10, 6);
        // sf_data body (all-zero spectra):
        //   section: cb_idx = 5 (4 bits) + sect_len for max_sfb=10 via
        //   n_sect_bits=3 (long frame at 1920), esc=7.
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, 10, 3, 7);
        // spectral pairs — cb 5 is dim=2, so pairs = end_line / 2.
        // sfb_offset_48 @ 1920 index 10 = ?
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end_line = sfbo[10] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        let pairs = end_line / 2;
        for _ in 0..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32); // zero pair
        }
        // scalefac: reference = 120 (8 bits). All bands are empty, so
        // no dpcm follows.
        bw.write_u32(120, 8);
        // snf: b_snf_data_exists = 0.
        bw.write_u32(0, 1);
        // aspx_data_1ch():
        //   aspx_xover_subband_offset = 3 (3 bits).
        bw.write_u32(3, 3);
        //   aspx_framing(0): FIXFIX = '0' (1 bit); tmp_num_env = 1
        //   (1 bit) -> num_env = 1 << 1 = 2.
        bw.write_bit(false); // FIXFIX
        bw.write_bit(true); // tmp_num_env = 1
        bw.align_to_byte();
        while bw.byte_len() < 220 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        // Sanity: we got through the outer layers.
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        assert!(info.tools.aspx_config.is_some());
        assert!(info.tools.companding.is_some());
        assert!(info.tools.transform_info_primary.is_some());
        // aspx_data_1ch path fired:
        assert_eq!(info.tools.aspx_xover_subband_offset, Some(3));
        let framing = info.tools.aspx_framing_primary.expect("framing");
        assert_eq!(framing.int_class, aspx::AspxIntClass::FixFix);
        assert_eq!(framing.num_env, 2);
        assert_eq!(framing.num_noise, 2);
        assert!(framing.freq_res.is_empty()); // freq_res_mode = High
                                              // Stereo sidecar untouched on mono.
        assert!(info.tools.aspx_framing_secondary.is_none());
        assert!(info.tools.aspx_balance.is_none());
        // aspx_delta_dir(0) followed on the same path: consumed
        // num_env + num_noise = 4 bits — all zeros from the padding.
        let dd = info.tools.aspx_delta_dir_primary.expect("delta dir");
        assert_eq!(dd.sig_delta_dir, vec![false; 2]);
        assert_eq!(dd.noise_delta_dir, vec![false; 2]);
        // FIXFIX + num_env > 1, so the config's quant_mode_env carries
        // through (and the config had it as Fine).
        assert_eq!(
            info.tools.aspx_qmode_env_primary,
            Some(aspx::AspxQuantStep::Fine)
        );
    }

    #[test]
    fn walk_mono_aspx_non_iframe_does_not_emit_framing() {
        // Non-I-frame ASPX substream: no aspx_config in the bitstream,
        // so the walker cannot safely consume aspx_data_1ch (xover
        // offset is also I-frame-sticky). Framing stays None.
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(200, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(1, 1); // mono_codec_mode = ASPX
                            // No aspx_config (b_iframe = false).
        bw.write_bit(true); // companding_control(1): compand_on[0] = 1.
        bw.write_u32(0, 1); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame = 1
        bw.write_u32(10, 6); // max_sfb
        bw.write_u32(5, 4); // cb
        write_sect_len_incr(&mut bw, 10, 3, 7);
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end_line = sfbo[10] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        for _ in 0..(end_line / 2) {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        bw.write_u32(120, 8);
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 220 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, false, 1920).unwrap();
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        // No aspx_config (non-I-frame), so no framing either.
        assert!(info.tools.aspx_config.is_none());
        assert!(info.tools.aspx_framing_primary.is_none());
        assert!(info.tools.aspx_xover_subband_offset.is_none());
    }

    /// End-to-end mono ASPX I-frame substream: hand-build
    /// aspx_hfgen_iwc_1ch plus two aspx_ec_data() payloads and verify
    /// the walker captures them. This is the round-6 acceptance test
    /// for parse_aspx_ec_data's wiring through §5.7.6.3.1 derivation.
    #[test]
    fn walk_mono_aspx_iframe_reads_ec_data_end_to_end() {
        use crate::aspx::{
            ASPX_HCB_ENV_LEVEL_15_DF, ASPX_HCB_ENV_LEVEL_15_F0, ASPX_HCB_NOISE_LEVEL_F0,
        };
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(500, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(1, 1); // mono_codec_mode = ASPX
                            // aspx_config(): quant_mode=0, start=3, stop=3, scale=1
                            // (highres -> 22 - 6 - 6 = 10 master groups), freq_res_mode=3
                            // (High -> no per-env freq_res bits), other flags zero.
        bw.write_u32(0, 1); // quant_mode_env
        bw.write_u32(3, 3); // start_freq
        bw.write_u32(3, 2); // stop_freq
        bw.write_u32(1, 1); // master_freq_scale = HighRes
        bw.write_bit(false); // interpolation
        bw.write_bit(false); // preflat
        bw.write_bit(false); // limiter
        bw.write_u32(0, 2); // noise_sbg
        bw.write_u32(0, 1); // num_env_bits_fixfix
        bw.write_u32(3, 2); // freq_res_mode = High
                            // companding_control(1): compand_on[0] = 1.
        bw.write_bit(true);
        // mono_data(0): ASF frontend, long frame, max_sfb=10, zero
        // spectra + scalefac 120 + no snf.
        bw.write_u32(0, 1); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, 10, 3, 7);
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end_line = sfbo[10] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        for _ in 0..(end_line / 2) {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        bw.write_u32(120, 8); // reference scalefac
        bw.write_u32(0, 1); // b_snf_data_exists
                            // aspx_data_1ch(): xover=7 -> num_sbg_sig_highres = 10 - 7 = 3
                            // and num_sbg_sig_lowres = 2. num_sbg_noise clamps to 1 since
                            // aspx_noise_sbg = 0. FIXFIX + tmp_num_env=0 -> num_env = 1.
        bw.write_u32(7, 3); // aspx_xover_subband_offset
        bw.write_bit(false); // FIXFIX
        bw.write_bit(false); // tmp_num_env = 0
                             // aspx_delta_dir(0): sig[0]=0, noise[0]=0 (FREQ for both).
        bw.write_bit(false);
        bw.write_bit(false);
        // aspx_hfgen_iwc_1ch: tna_mode[0] = 0 (2 bits), ah_present=0,
        // fic_present=0, tic_present=0 (3 bits).
        bw.write_u32(0, 2);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(false);
        // aspx_data_sig[0]: num_env=1, high-res (3 subband groups).
        // F0 symbol 30 + two DF zero-deltas (index 70).
        let f0_cb = &ASPX_HCB_ENV_LEVEL_15_F0;
        let df_cb = &ASPX_HCB_ENV_LEVEL_15_DF;
        bw.write_u32(f0_cb.cw[30], f0_cb.len[30] as u32);
        bw.write_u32(df_cb.cw[70], df_cb.len[70] as u32);
        bw.write_u32(df_cb.cw[70], df_cb.len[70] as u32);
        // aspx_data_noise[0]: num_noise=1 (since num_env==1), num_sbg_noise=1.
        // One F0 codeword from the NOISE_LEVEL_F0 table.
        let noise_f0 = &ASPX_HCB_NOISE_LEVEL_F0;
        bw.write_u32(noise_f0.cw[6], noise_f0.len[6] as u32);
        bw.align_to_byte();
        while bw.byte_len() < 520 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        // §5.7.6.3.1 derivation results.
        let tables = info.tools.aspx_frequency_tables.as_ref().expect("tables");
        assert_eq!(tables.num_sbg_master, 10);
        assert_eq!(tables.sba, 24);
        assert_eq!(tables.sbz, 44);
        assert_eq!(tables.counts.num_sbg_sig_highres, 3);
        assert_eq!(tables.counts.num_sbg_sig_lowres, 2);
        assert_eq!(tables.counts.num_sbg_noise, 1);
        // hfgen: all gates off.
        let hfgen = info.tools.aspx_hfgen_iwc_1ch.as_ref().expect("hfgen");
        assert_eq!(hfgen.tna_mode, vec![0]);
        assert!(!hfgen.ah_present);
        assert!(!hfgen.fic_present);
        assert!(!hfgen.tic_present);
        // Signal envelope: F0=30, DF=0, DF=0.
        let sig = info.tools.aspx_data_sig_primary.as_ref().expect("sig");
        assert_eq!(sig.len(), 1);
        assert_eq!(sig[0].values, vec![30, 0, 0]);
        assert!(!sig[0].direction_time);
        // Noise envelope: single F0 with delta = 6.
        let noise = info.tools.aspx_data_noise_primary.as_ref().expect("noise");
        assert_eq!(noise.len(), 1);
        assert_eq!(noise[0].values, vec![6]);
        assert!(!noise[0].direction_time);
    }

    #[test]
    fn walk_stereo_aspx_acpl1_parses_acpl_config_partial() {
        // stereo_codec_mode = 10 (ASPX_ACPL_1): aspx_config + then
        // acpl_config_1ch(PARTIAL) (= 6 bits: 2 bands_id + 1 quant +
        // 3 qmf_band_minus1). We bail before companding_control —
        // companding stays None.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b10, 2); // ASPX_ACPL_1
        bw.write_u32(0, 15); // aspx_config (all zero)
                             // acpl_config_1ch(PARTIAL): id=1 (12 bands), coarse, qmf_minus1=3
        bw.write_u32(0b01, 2);
        bw.write_u32(1, 1);
        bw.write_u32(0b011, 3);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::AspxAcpl1));
        assert!(info.tools.aspx_config.is_some());
        let acpl = info
            .tools
            .acpl_config_1ch_partial
            .expect("acpl_config_1ch_partial should be set");
        assert_eq!(acpl.num_param_bands_id, 1);
        assert_eq!(acpl.num_param_bands, 12);
        assert_eq!(acpl.quant_mode, crate::acpl::AcplQuantMode::Coarse);
        assert_eq!(acpl.qmf_band, 4);
        assert!(info.tools.acpl_config_1ch_full.is_none());
        // We bail before companding_control — companding stays None.
        assert!(info.tools.companding.is_none());
    }

    #[test]
    fn walk_stereo_aspx_acpl2_parses_acpl_config_full() {
        // stereo_codec_mode = 11 (ASPX_ACPL_2): aspx_config + then
        // acpl_config_1ch(FULL) (= 3 bits: 2 bands_id + 1 quant). Bails
        // before companding.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b11, 2); // ASPX_ACPL_2
        bw.write_u32(0, 15); // aspx_config (all zero)
                             // acpl_config_1ch(FULL): id=2 (9 bands), fine, no qmf_band field
        bw.write_u32(0b10, 2);
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::AspxAcpl2));
        assert!(info.tools.aspx_config.is_some());
        let acpl = info
            .tools
            .acpl_config_1ch_full
            .expect("acpl_config_1ch_full should be set");
        assert_eq!(acpl.num_param_bands_id, 2);
        assert_eq!(acpl.num_param_bands, 9);
        assert_eq!(acpl.quant_mode, crate::acpl::AcplQuantMode::Fine);
        // qmf_band stays 0 for FULL mode (not transmitted).
        assert_eq!(acpl.qmf_band, 0);
        assert!(info.tools.acpl_config_1ch_partial.is_none());
        assert!(info.tools.companding.is_none());
    }
}
