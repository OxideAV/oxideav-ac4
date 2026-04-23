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
    let mut info = AsfPsyInfo::default();

    // b_different_framing is derived, not coded.
    info.b_different_framing = frame_len_base >= 1536
        && !transform_info.b_long_frame
        && transform_info.transf_length[0] != transform_info.transf_length[1];

    // Resolve n_msfb_bits / n_side_bits from Table 106 for the primary
    // transform length.
    let (nm0, ns0, _nml0) = tables::n_msfb_bits_48(transform_info.transform_length_0).ok_or_else(
        || Error::invalid("ac4: asf_psy_info: unsupported transform_length[0]"),
    )?;

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
    // aspx_config() / companding_control() / aspx_data_1ch() parsing are
    // not implemented — see module docs. They're gated by mode==ASPX;
    // for mode==SIMPLE we only have mono_data(0).
    if mode != MonoCodecMode::Simple {
        // ASPX path requires Huffman / QMF state we don't have.
        return Ok(());
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
        }
        // asf_section_data / asf_spectral_data / asf_scalefac_data /
        // asf_snf_data bodies pending Huffman tables — not decoded yet.
    }
    Ok(())
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
        // ASPX / ASPX_ACPL paths need aspx/acpl configs — not parsed
        // yet.
        return Ok(());
    }

    // SIMPLE path: just `stereo_data()`.
    //
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
    let b_mdct_stereo = br.read_bit()?;
    tools.mdct_stereo_proc = b_mdct_stereo;
    if b_mdct_stereo {
        tools.spec_frontend_primary = Some(SpecFrontend::Asf);
        tools.spec_frontend_secondary = Some(SpecFrontend::Asf);
        let ti = parse_asf_transform_info(br, frame_len_base)?;
        tools.transform_info_primary = Some(ti);
        tools.transform_info_secondary = Some(ti);
        // stereo_data() with b_enable_mdct_stereo_proc==1 calls
        // sf_info(ASF, 0, 0) which fires asf_psy_info(0, 0) over the
        // shared transform window.
        if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
            tools.psy_info_primary = Some(psy);
        }
        // chparam_info() + sf_data() bodies pending.
    } else {
        let l = SpecFrontend::from_bit(br.read_u32(1)?);
        tools.spec_frontend_primary = Some(l);
        if let SpecFrontend::Asf = l {
            let ti = parse_asf_transform_info(br, frame_len_base)?;
            tools.transform_info_primary = Some(ti);
            if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                tools.psy_info_primary = Some(psy);
            }
        }
        let r = SpecFrontend::from_bit(br.read_u32(1)?);
        tools.spec_frontend_secondary = Some(r);
        if let SpecFrontend::Asf = r {
            let ti = parse_asf_transform_info(br, frame_len_base)?;
            tools.transform_info_secondary = Some(ti);
            if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                tools.psy_info_secondary = Some(psy);
            }
        }
    }
    let _ = b_iframe; // reserved for later Huffman-state keying.
    Ok(())
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
    fn walk_mono_aspx_substream_records_mode_and_bails() {
        // mono_codec_mode = 1 (ASPX) — we stop before aspx_config, so
        // mono_mode is set but no frontend/transform_info.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(1, 1); // ASPX
        bw.align_to_byte();
        while bw.byte_len() < 16 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        assert!(info.tools.spec_frontend_primary.is_none());
        assert!(info.tools.transform_info_primary.is_none());
    }
}
