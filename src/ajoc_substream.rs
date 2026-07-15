//! A-JOC object-audio substream body — ETSI TS 103 190-2 §6.2.3.4
//! `audio_data_ajoc()` and §6.2.4.4 `var_channel_element()`.
//!
//! This module joins the previously landed pieces into the complete
//! substream-body walk for A-JOC-coded object audio:
//!
//! * **`var_channel_element()`** (§6.2.4.4) carries the coded downmix
//!   signals: an optional LFE `mono_data(1)`, `floor(n/2)` ×
//!   `two_channel_data()` pairs, and the odd-count tail
//!   (`mono_data(0)`, pair + mono, or `three_channel_data()` selected
//!   by `var_coding_config`), plus the A-SPX layer (`aspx_config()` /
//!   `companding_control()` / per-pair `aspx_data_2ch()` + odd
//!   `aspx_data_1ch()`) when `var_codec_mode == 1` (Table 99).
//! * **`audio_data_ajoc()`** (§6.2.3.4) wraps the downmix with the
//!   OAMD side information (core-decode `oamd_timing_data()` +
//!   `oamd_dyndata_single()`), the `ajoc()` parameter element
//!   (§6.2.5.1 — Huffman rows → differential decode → dequantized
//!   matrices via [`crate::ajoc_data::decode_ajoc`]),
//!   `ajoc_dmx_de_data()` (§6.2.3.5), and the full-decode OAMD tail.
//!   The `b_static_dmx == 1` form delegates to the part-1
//!   `5_X_channel_element()` walker instead.
//!
//! Write-side helpers emit a parse-equivalent `audio_data_ajoc()` for
//! the SIMPLE (`var_codec_mode == 0`) downmix form so the encoder can
//! produce complete A-JOC substreams and the tests can round-trip the
//! whole chain bitstream → dequantized matrices + downmix spectra.

use crate::ajoc::{
    parse_ajoc_bed_info, parse_ajoc_dmx_de_data, AjocBedInfo, AjocDiffState, AjocDmxDeData,
};
use crate::ajoc_data::{decode_ajoc, AjocFrame};
use crate::asf::SubstreamTools;
use crate::aspx::{
    parse_aspx_config, parse_companding_control, AspxConfig, CompandingControl, FiveXAspxTrailer,
};
use crate::mch::{
    parse_mono_data, parse_three_channel_data, parse_two_channel_data, MonoLfeData,
    ThreeChannelData, TwoChannelData,
};
use crate::oamd::{
    parse_oamd_dyndata_single, parse_oamd_timing_data, write_oamd_dyndata_single,
    write_oamd_timing_data, InfoStatus, OaSampleOffset, OamdDynDataSingle, OamdTimingBlock,
    OamdTimingData, ObjType, ObjectInfoBlock, RampDuration,
};
use crate::toc::variable_bits;
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

// =====================================================================
// var_channel_element (§6.2.4.4)
// =====================================================================

/// Odd-signal-count tail of `var_channel_element()`.
#[derive(Debug, Clone)]
pub enum VarOddTail {
    /// `n_dmx_signals == 1` — a single `mono_data(0)`.
    Mono(MonoLfeData),
    /// `var_coding_config == 0` — final pair + `mono_data(0)`.
    PairPlusMono {
        /// The final `two_channel_data()`.
        pair: TwoChannelData,
        /// The trailing `mono_data(0)`.
        mono: MonoLfeData,
    },
    /// `var_coding_config == 1` — `three_channel_data()`.
    Three(ThreeChannelData),
}

/// Parsed `var_channel_element(b_iframe, n_dmx_signals, b_has_lfe)`.
#[derive(Debug, Clone, Default)]
pub struct VarChannelElement {
    /// `var_codec_mode` — false = Simple, true = A-SPX (Table 99).
    pub aspx_mode: bool,
    /// `aspx_config()` (A-SPX mode I-frames).
    pub aspx_config: Option<AspxConfig>,
    /// `companding_control()` (A-SPX mode, `n_dmx_signals <= 5`).
    pub companding: Option<CompandingControl>,
    /// LFE `mono_data(1)` when `b_has_lfe`.
    pub lfe: Option<MonoLfeData>,
    /// Leading `two_channel_data()` pairs (all `n_pairs` for even
    /// counts, `n_pairs - 1` for odd counts > 1).
    pub pairs: Vec<TwoChannelData>,
    /// Odd-count tail.
    pub odd_tail: Option<Box<VarOddTail>>,
    /// Per-pair A-SPX payloads (`aspx_data_2ch()`, A-SPX mode only) —
    /// captured as decode-ready trailers; `None` when the capture
    /// guards tripped (the bits were still consumed).
    pub aspx_pair_trailers: Vec<Option<Box<FiveXAspxTrailer>>>,
    /// Odd-channel `aspx_data_1ch()` payload (A-SPX mode only).
    pub aspx_odd_trailer: Option<Box<FiveXAspxTrailer>>,
}

impl VarChannelElement {
    /// Decoded fullband MDCT spectra in signal order (LFE excluded).
    /// `None` entries are channels whose `sf_data` body the walker
    /// could not decode (short/grouped frames).
    pub fn fullband_spectra(&self) -> Vec<Option<&[f32]>> {
        let mut out: Vec<Option<&[f32]>> = Vec::new();
        for pair in &self.pairs {
            for ch in 0..2 {
                out.push(
                    pair.scaled_spec_per_channel
                        .get(ch)
                        .and_then(|s| s.as_deref()),
                );
            }
        }
        match self.odd_tail.as_deref() {
            None => {}
            Some(VarOddTail::Mono(m)) => out.push(m.scaled_spec.as_deref()),
            Some(VarOddTail::PairPlusMono { pair, mono }) => {
                for ch in 0..2 {
                    out.push(
                        pair.scaled_spec_per_channel
                            .get(ch)
                            .and_then(|s| s.as_deref()),
                    );
                }
                out.push(mono.scaled_spec.as_deref());
            }
            Some(VarOddTail::Three(t)) => {
                for ch in 0..3 {
                    out.push(t.scaled_spec_per_channel.get(ch).and_then(|s| s.as_deref()));
                }
            }
        }
        out
    }

    /// Decoded LFE spectrum, when present and decodable.
    pub fn lfe_spectrum(&self) -> Option<&[f32]> {
        self.lfe.as_ref().and_then(|m| m.scaled_spec.as_deref())
    }
}

/// Parse `var_channel_element(b_iframe, n_dmx_signals, b_has_lfe)` per
/// §6.2.4.4. `n_fb_signals` is the fullband signal count (the syntax
/// argument `n_dmx_signals`); the LFE, when present, is carried in
/// addition to it.
///
/// For P-frames in A-SPX mode the I-frame `aspx_config()` is absent —
/// pass it via `sticky_aspx` (with the sticky
/// `aspx_xover_subband_offset`) or the parse fails.
pub fn parse_var_channel_element(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    n_fb_signals: u32,
    b_has_lfe: bool,
    frame_len_base: u32,
    sticky_aspx: Option<(&AspxConfig, u8)>,
) -> Result<VarChannelElement> {
    let mut out = VarChannelElement {
        aspx_mode: br.read_bit()?,
        ..Default::default()
    };
    let b_isodd = n_fb_signals % 2 == 1;
    let n_pairs = n_fb_signals / 2;
    if out.aspx_mode {
        if b_iframe {
            out.aspx_config = Some(parse_aspx_config(br)?);
        }
        if n_fb_signals <= 5 {
            out.companding = Some(parse_companding_control(br, n_fb_signals)?);
        }
    }
    if b_has_lfe {
        out.lfe = Some(parse_mono_data(br, true, frame_len_base)?);
    }
    if b_isodd {
        if n_fb_signals == 1 {
            out.odd_tail = Some(Box::new(VarOddTail::Mono(parse_mono_data(
                br,
                false,
                frame_len_base,
            )?)));
        } else {
            for _ in 0..n_pairs.saturating_sub(1) {
                out.pairs.push(parse_two_channel_data(br, frame_len_base)?);
            }
            let var_coding_config = br.read_bit()?;
            if !var_coding_config {
                let pair = parse_two_channel_data(br, frame_len_base)?;
                let mono = parse_mono_data(br, false, frame_len_base)?;
                out.odd_tail = Some(Box::new(VarOddTail::PairPlusMono { pair, mono }));
            } else {
                out.odd_tail = Some(Box::new(VarOddTail::Three(parse_three_channel_data(
                    br,
                    frame_len_base,
                )?)));
            }
        }
    } else {
        for _ in 0..n_pairs {
            out.pairs.push(parse_two_channel_data(br, frame_len_base)?);
        }
    }
    if out.aspx_mode {
        let (cfg, sticky_xover) = match (&out.aspx_config, sticky_aspx) {
            (Some(c), _) => (*c, None),
            (None, Some((c, x))) => (*c, Some(x)),
            (None, None) => {
                return Err(Error::invalid(
                    "ac4: non-iframe var_channel_element A-SPX without sticky config",
                ))
            }
        };
        // Scratch tools shared across the captures — carries the
        // sticky xover for P-frames (the capture helpers snapshot /
        // restore around each parse).
        let mut scratch = SubstreamTools {
            aspx_xover_subband_offset: sticky_xover,
            ..Default::default()
        };
        for _ in 0..n_pairs {
            let t = crate::asf::capture_aspx_data_2ch_trailer(
                br,
                &mut scratch,
                &cfg,
                b_iframe,
                frame_len_base,
            );
            if t.is_none() {
                return Err(Error::invalid(
                    "ac4: var_channel_element aspx_data_2ch parse failed",
                ));
            }
            out.aspx_pair_trailers.push(t.map(Box::new));
        }
        if b_isodd {
            let t = crate::asf::capture_aspx_data_1ch_trailer(
                br,
                &mut scratch,
                &cfg,
                b_iframe,
                frame_len_base,
            );
            if t.is_none() {
                return Err(Error::invalid(
                    "ac4: var_channel_element aspx_data_1ch parse failed",
                ));
            }
            out.aspx_odd_trailer = t.map(Box::new);
        }
    }
    Ok(out)
}

// =====================================================================
// audio_data_ajoc (§6.2.3.4)
// =====================================================================

/// Parsed `audio_data_ajoc()` element.
#[derive(Debug, Clone)]
pub struct AudioDataAjoc {
    /// `b_static_dmx == 1` path: the part-1 `5_X_channel_element()`
    /// scratch (5.0 / 5.1 per `b_lfe`).
    pub static_chan_tools: Option<Box<SubstreamTools>>,
    /// `dmx_active_signals_mask` when `b_some_signals_inactive`
    /// (LSB-first as read, `n_fullband_dmx_signals` bits; a 0 flag =
    /// coded silence per §6.3.4.2).
    pub dmx_active_signals_mask: Option<u32>,
    /// The coded downmix (dynamic form).
    pub var_element: Option<VarChannelElement>,
    /// Core-decode OAMD timing (`b_dmx_timing`).
    pub dmx_timing: Option<OamdTimingData>,
    /// Core-decode per-object OAMD (dynamic form only).
    pub dmx_dyndata: Option<OamdDynDataSingle>,
    /// OAMD extension envelope: parsed `ajoc_bed_info()` + skipped
    /// bits (`b_oamd_extension_present`).
    pub oamd_extension: Option<(AjocBedInfo, u32)>,
    /// The decoded `ajoc()` element — ctrl info, Huffman payload, and
    /// dequantized dry/wet matrices.
    pub ajoc_frame: AjocFrame,
    /// `ajoc_dmx_de_data()` (§6.2.3.5).
    pub dmx_de: AjocDmxDeData,
    /// Full-decode OAMD timing (`b_umx_timing`).
    pub umx_timing: Option<OamdTimingData>,
    /// `b_derive_timing_from_dmx` (present when `b_umx_timing == 0`).
    pub b_derive_timing_from_dmx: Option<bool>,
    /// Full-decode per-object OAMD.
    pub umx_dyndata: OamdDynDataSingle,
}

/// Downmix-signal descriptor needed to parse `audio_data_ajoc()` —
/// the fields of `ac4_substream_info_ajoc()` the body depends on.
#[derive(Debug, Clone)]
pub struct AjocBodyParams {
    /// `b_lfe`.
    pub b_lfe: bool,
    /// `b_static_dmx`.
    pub b_static_dmx: bool,
    /// `n_fullband_dmx_signals`.
    pub n_fullband_dmx_signals: u32,
    /// `n_fullband_upmix_signals`.
    pub n_fullband_upmix_signals: u32,
    /// Object types for the downmix set (LFE first when present).
    pub obj_type_dmx: Vec<ObjType>,
    /// Object types for the upmix set (LFE first when present).
    pub obj_type_umx: Vec<ObjType>,
}

impl AjocBodyParams {
    /// Derive the body parameters from a parsed TOC descriptor.
    pub fn from_substream_info(info: &crate::toc::AjocSubstreamInfo) -> Self {
        AjocBodyParams {
            b_lfe: info.b_lfe,
            b_static_dmx: info.b_static_dmx,
            n_fullband_dmx_signals: info.n_fullband_dmx_signals,
            n_fullband_upmix_signals: info.n_fullband_upmix_signals,
            obj_type_dmx: info.obj_type_dmx(),
            obj_type_umx: info.obj_type_umx(),
        }
    }

    fn is_lfe(&self, n: usize) -> Vec<bool> {
        let mut v = vec![false; n];
        if self.b_lfe && !v.is_empty() {
            v[0] = true;
        }
        v
    }
}

/// Parse `audio_data_ajoc(n_fb_upmix_signals, b_static_dmx,
/// n_fb_dmx_signals, b_lfe, b_iframe)` per §6.2.3.4.
///
/// `b_alternative` comes from the presentation substream
/// (`ac4_presentation_substream_info()`, §6.2.1.12); `ajoc_state`
/// carries the §5.7.3.2 differential-decode reference across frames.
/// `sticky_aspx` supplies the previous I-frame's `aspx_config` +
/// `aspx_xover_subband_offset` for `b_iframe == 0` A-SPX downmixes
/// (Tables 51/52 omit both on non-I-frames).
pub fn parse_audio_data_ajoc(
    br: &mut BitReader<'_>,
    params: &AjocBodyParams,
    b_iframe: bool,
    b_alternative: bool,
    frame_len_base: u32,
    ajoc_state: &mut AjocDiffState,
    sticky_aspx: Option<(&AspxConfig, u8)>,
) -> Result<AudioDataAjoc> {
    let mut static_chan_tools = None;
    let mut dmx_active_signals_mask = None;
    let mut var_element = None;
    let mut dmx_timing = None;
    let mut dmx_dyndata = None;
    if params.b_static_dmx {
        // audio_data_chan(b_lfe ? 5.1 : 5.0, b_iframe).
        let mut tools = Box::new(SubstreamTools {
            channel_mode_channels: if params.b_lfe { 6 } else { 5 },
            ..Default::default()
        });
        crate::mch::parse_5x_audio_data_outer(
            br,
            &mut tools,
            params.b_lfe,
            b_iframe,
            frame_len_base,
        )?;
        static_chan_tools = Some(tools);
    } else {
        let n_dmx_signals = params.n_fullband_dmx_signals + u32::from(params.b_lfe);
        if br.read_bit()? {
            // b_some_signals_inactive.
            dmx_active_signals_mask = Some(br.read_u32(params.n_fullband_dmx_signals)?);
        }
        var_element = Some(parse_var_channel_element(
            br,
            b_iframe,
            params.n_fullband_dmx_signals,
            params.b_lfe,
            frame_len_base,
            sticky_aspx,
        )?);
        if br.read_bit()? {
            // b_dmx_timing.
            dmx_timing = Some(parse_oamd_timing_data(br)?);
        }
        let n_blocks = dmx_timing
            .as_ref()
            .map(|t| t.num_obj_info_blocks())
            .unwrap_or(0);
        let is_lfe = params.is_lfe(n_dmx_signals as usize);
        if params.obj_type_dmx.len() != n_dmx_signals as usize {
            return Err(Error::invalid("ac4: obj_type_dmx length mismatch"));
        }
        dmx_dyndata = Some(parse_oamd_dyndata_single(
            br,
            n_blocks,
            b_iframe,
            b_alternative,
            &params.obj_type_dmx,
            &is_lfe,
        )?);
    }
    // b_oamd_extension_present — the envelope embeds ajoc_bed_info().
    let oamd_extension = if !params.b_static_dmx && br.read_bit()? {
        let total = (variable_bits(br, 3)? + 1) * 8;
        let start = br.bit_position();
        let bed_info = parse_ajoc_bed_info(br)?;
        let used = (br.bit_position() - start) as u32;
        if used > total {
            return Err(Error::invalid(
                "ac4: ajoc_bed_info exceeded the OAMD extension envelope",
            ));
        }
        let remain = total - used;
        for _ in 0..remain {
            let _ = br.read_bit()?;
        }
        Some((bed_info, remain))
    } else {
        None
    };
    // ajoc(n_fb_dmx_signals, n_fb_upmix_signals).
    let ajoc_frame = decode_ajoc(
        br,
        params.n_fullband_dmx_signals,
        params.n_fullband_upmix_signals,
        ajoc_state,
    )?;
    // ajoc_dmx_de_data(n_fb_dmx_signals, n_fb_upmix_signals).
    let dmx_de = parse_ajoc_dmx_de_data(
        br,
        params.n_fullband_dmx_signals,
        params.n_fullband_upmix_signals,
    )?;
    // Full-decode timing + dyndata.
    let mut umx_timing = None;
    let mut b_derive_timing_from_dmx = None;
    if br.read_bit()? {
        umx_timing = Some(parse_oamd_timing_data(br)?);
    } else {
        b_derive_timing_from_dmx = Some(br.read_bit()?);
    }
    let n_umx_signals = params.n_fullband_upmix_signals + u32::from(params.b_lfe);
    let n_blocks = umx_timing
        .as_ref()
        .or(dmx_timing.as_ref())
        .map(|t| t.num_obj_info_blocks())
        .unwrap_or(0);
    if params.obj_type_umx.len() != n_umx_signals as usize {
        return Err(Error::invalid("ac4: obj_type_umx length mismatch"));
    }
    let is_lfe_umx = params.is_lfe(n_umx_signals as usize);
    let umx_dyndata = parse_oamd_dyndata_single(
        br,
        n_blocks,
        b_iframe,
        b_alternative,
        &params.obj_type_umx,
        &is_lfe_umx,
    )?;
    Ok(AudioDataAjoc {
        static_chan_tools,
        dmx_active_signals_mask,
        var_element,
        dmx_timing,
        dmx_dyndata,
        oamd_extension,
        ajoc_frame,
        dmx_de,
        umx_timing,
        b_derive_timing_from_dmx,
        umx_dyndata,
    })
}

// =====================================================================
// Bitstream → object-PCM decode chain (§4.8.3.13 + §5.7)
// =====================================================================

/// Reconstructed A-JOC PCM: the `num_umx` fullband objects
/// (`[o][samples]`) plus the decoded LFE PCM when the substream
/// carries one.
pub type AjocObjectPcm = (Vec<Vec<f32>>, Option<Vec<f32>>);

/// [`AjocObjectPcm`] plus the parsed `audio_data_ajoc()` body and the
/// post-audio `metadata(…, sus_ver = 1)` element (§6.2.2.2).
pub type AjocSubstreamDecode = (
    Vec<Vec<f32>>,
    Option<Vec<f32>>,
    AudioDataAjoc,
    Option<crate::metadata::MetadataV2>,
);

/// Extract the static-downmix 5.X core's fullband spectra in the
/// canonical `[L, R, C, Ls, Rs]` order from the parsed
/// `audio_data_chan(b_lfe ? 5.1 : 5.0)` tools (part-1 Table 180
/// channel mappings, SIMPLE codec mode). The A-SPX / A-CPL 5.X codec
/// modes are parsed upstream but their carrier synthesis is not wired
/// into the object path.
fn static_dmx_spectra(tools: &SubstreamTools) -> Result<[Option<&[f32]>; 5]> {
    use crate::mch::{FiveXCodecMode, FiveXCodingConfig};
    if !matches!(tools.five_x_mode, Some(FiveXCodecMode::Simple)) {
        return Err(Error::unsupported(
            "ac4: static-dmx A-JOC core beyond the SIMPLE codec mode not wired yet",
        ));
    }
    fn pair_ch(p: &TwoChannelData, ch: usize) -> Option<&[f32]> {
        p.scaled_spec_per_channel.get(ch).and_then(|s| s.as_deref())
    }
    let mut out: [Option<&[f32]>; 5] = [None; 5];
    match tools.five_x_coding_config {
        Some(FiveXCodingConfig::Cfg3Five) => {
            if let Some(five) = tools.five_channel_data.as_ref() {
                // Table 180 cfg3: five_channel_data → [L, R, C, Ls, Rs].
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = five
                        .scaled_spec_per_channel
                        .get(i)
                        .and_then(|s| s.as_deref());
                }
            }
        }
        Some(FiveXCodingConfig::Cfg0Stereo2plusMono) => {
            // Table 180 cfg0: pair order depends on b_2ch_mode
            // (L/R + Ls/Rs, or L/Ls + R/Rs); trailing mono is C.
            if tools.two_channel_data.len() >= 2 {
                let (p0, p1) = (&tools.two_channel_data[0], &tools.two_channel_data[1]);
                if tools.b_2ch_mode == Some(true) {
                    out[0] = pair_ch(p0, 0);
                    out[3] = pair_ch(p0, 1);
                    out[1] = pair_ch(p1, 0);
                    out[4] = pair_ch(p1, 1);
                } else {
                    out[0] = pair_ch(p0, 0);
                    out[1] = pair_ch(p0, 1);
                    out[3] = pair_ch(p1, 0);
                    out[4] = pair_ch(p1, 1);
                }
            }
            out[2] = tools
                .cfg0_centre_mono
                .as_ref()
                .and_then(|m| m.scaled_spec.as_deref());
        }
        Some(FiveXCodingConfig::Cfg1ThreeStereo) => {
            // Table 180 cfg1: three_channel_data → [L, R, C],
            // two_channel_data → [Ls, Rs].
            if let Some(three) = tools.three_channel_data.as_ref() {
                for (i, slot) in out.iter_mut().take(3).enumerate() {
                    *slot = three
                        .scaled_spec_per_channel
                        .get(i)
                        .and_then(|s| s.as_deref());
                }
            }
            if let Some(p) = tools.two_channel_data.first() {
                out[3] = pair_ch(p, 0);
                out[4] = pair_ch(p, 1);
            }
        }
        Some(FiveXCodingConfig::Cfg2FourMono) => {
            // Table 180 cfg2: four_channel_data → [L, R, Ls, Rs],
            // trailing mono → C.
            if let Some(four) = tools.four_channel_data.as_ref() {
                for (i, slot) in [0usize, 1, 3, 4].into_iter().enumerate() {
                    out[slot] = four
                        .scaled_spec_per_channel
                        .get(i)
                        .and_then(|s| s.as_deref());
                }
            }
            out[2] = tools
                .cfg2_back_mono
                .as_ref()
                .and_then(|m| m.scaled_spec.as_deref());
        }
        _ => {
            return Err(Error::invalid(
                "ac4: static-dmx core without a coding_config",
            ))
        }
    }
    Ok(out)
}

/// Persistent A-JOC object-substream decoder state: the §5.7.3.2
/// differential reference, the §5.7.3.4-6 reconstruction state
/// (interpolators + decorrelator histories), the per-downmix-channel
/// IMDCT overlap-add tails, and the per-object QMF synthesis banks.
pub struct AjocSubstreamDecoder {
    diff_state: AjocDiffState,
    recon: Option<(crate::ajoc::AjocGeometry, crate::ajoc::AjocReconState)>,
    overlap: Vec<Vec<f32>>,
    prev_transform_length: u32,
    analysis: Vec<crate::qmf::QmfAnalysisBank>,
    synthesis: Vec<crate::qmf::QmfSynthesisBank>,
    num_dmx: usize,
    num_umx: usize,
    /// `de_config()` carried from a prior I-frame's post-audio
    /// `metadata()` — needed to parse `dialog_enhancement()` on
    /// dependent frames.
    prev_de: Option<crate::de::DeConfig>,
    /// I-frame-sticky `(aspx_config, aspx_xover_subband_offset)` for
    /// the A-SPX downmix form (Tables 51/52 omit both on non-I-frames;
    /// single-xover-per-substream convention).
    sticky_aspx: Option<(AspxConfig, u8)>,
    /// Per-downmix-channel A-SPX extension state (envelope / noise /
    /// tone / TNS carry-over).
    aspx_ext_state: Vec<crate::aspx::AspxChannelExtState>,
}

impl AjocSubstreamDecoder {
    /// Allocate decoder state for `num_dmx` fullband downmix signals
    /// and `num_umx` reconstructed objects (from the TOC descriptor).
    pub fn new(num_dmx: usize, num_umx: usize) -> Self {
        AjocSubstreamDecoder {
            // ajoc_num_decorr is a 3-bit field — size the differential
            // state to its maximum so any frame fits.
            diff_state: crate::ajoc_data::new_ajoc_diff_state(num_umx, num_dmx, 7),
            recon: None,
            overlap: Vec::new(),
            prev_transform_length: 0,
            analysis: (0..num_dmx)
                .map(|_| crate::qmf::QmfAnalysisBank::new())
                .collect(),
            synthesis: (0..num_umx)
                .map(|_| crate::qmf::QmfSynthesisBank::new())
                .collect(),
            num_dmx,
            num_umx,
            prev_de: None,
            sticky_aspx: None,
            aspx_ext_state: Vec::new(),
        }
    }

    /// Access the parse-side differential state (for feeding
    /// [`parse_audio_data_ajoc`] directly).
    pub fn diff_state_mut(&mut self) -> &mut AjocDiffState {
        &mut self.diff_state
    }

    fn imdct_channel_f32(&mut self, ch: usize, scaled: &[f32], n: usize) -> Vec<f32> {
        if self.prev_transform_length != n as u32 {
            self.overlap.clear();
            self.prev_transform_length = n as u32;
        }
        while self.overlap.len() <= ch {
            self.overlap.push(vec![0.0f32; n]);
        }
        if self.overlap[ch].len() != n {
            self.overlap[ch] = vec![0.0f32; n];
        }
        let mut x = vec![0.0f32; n];
        let copy = scaled.len().min(n);
        x[..copy].copy_from_slice(&scaled[..copy]);
        let y = crate::mdct::imdct(&x);
        let window = crate::mdct::kbd_window(n as u32);
        crate::mdct::imdct_olap_symmetric(&y, &window, &mut self.overlap[ch])
    }

    /// Decode one parsed `audio_data_ajoc()` frame to reconstructed
    /// object PCM (`[o][samples]`, `num_umx` fullband objects) plus
    /// the decoded LFE PCM when the substream carries one (the LFE is
    /// coded directly in the downmix `mono_data(1)` / the 5.1 core and
    /// bypasses the A-JOC spatial reconstruction).
    ///
    /// Covers all three downmix forms: the dynamic SIMPLE form, the
    /// dynamic **A-SPX** form (per-channel QMF-domain bandwidth
    /// extension from the captured `aspx_data_*` payloads before the
    /// spatial reconstruction; companding gains are not applied), and
    /// the **`b_static_dmx`** 5.X core (SIMPLE coding configs — the
    /// `audio_data_chan(5.0/5.1)` spectra feed the reconstruction in
    /// `[L, R, C, Ls, Rs]` order). Per-channel IMDCT → QMF →
    /// §5.7.3.6 Table 49 spatial reconstruction → per-object QMF
    /// synthesis.
    pub fn decode_frame_pcm(
        &mut self,
        ajoc: &AudioDataAjoc,
        frame_len_base: u32,
    ) -> Result<AjocObjectPcm> {
        let n = frame_len_base as usize;
        // Downmix spectra in channel order + the LFE body, from either
        // the static 5.X core or the dynamic var_channel_element.
        let (spectra, lfe_data): (Vec<Option<&[f32]>>, Option<&MonoLfeData>) =
            if let Some(tools) = ajoc.static_chan_tools.as_deref() {
                (
                    static_dmx_spectra(tools)?.to_vec(),
                    tools.lfe_mono_data.as_ref(),
                )
            } else {
                let elem = ajoc.var_element.as_ref().ok_or_else(|| {
                    Error::invalid("ac4: audio_data_ajoc without a downmix element")
                })?;
                (elem.fullband_spectra(), elem.lfe.as_ref())
            };
        if spectra.len() != self.num_dmx {
            return Err(Error::invalid("ac4: downmix channel count mismatch"));
        }
        // 0. The LFE channel (when present) is carried directly as a
        // long-frame ASF mono body: IMDCT it on its dedicated overlap
        // slot (index num_dmx). An undecodable LFE body degrades to
        // silence (`None`) like the channel-based paths.
        let lfe_pcm = match lfe_data.and_then(|l| l.scaled_spec.as_deref().map(|s| (l, s))) {
            Some((lfe, spec))
                if lfe
                    .transform_info
                    .as_ref()
                    .is_some_and(|ti| ti.transform_length_0 as usize == n) =>
            {
                let spec = spec.to_vec();
                let slot = self.num_dmx;
                Some(self.imdct_channel_f32(slot, &spec, n))
            }
            _ => None,
        };
        // 1. IMDCT each downmix channel.
        let spectra: Vec<Vec<f32>> = spectra
            .iter()
            .map(|spec| {
                spec.map(<[f32]>::to_vec)
                    .ok_or_else(|| Error::unsupported("ac4: undecodable downmix channel body"))
            })
            .collect::<Result<_>>()?;
        let mut pcm_dmx: Vec<Vec<f32>> = Vec::with_capacity(self.num_dmx);
        for (ch, spec) in spectra.iter().enumerate() {
            pcm_dmx.push(self.imdct_channel_f32(ch, spec, n));
        }
        // 2. QMF-analyse each channel into x[ts][sb][ch]. For the
        // A-SPX downmix form each channel is bandwidth-extended in the
        // QMF domain from its captured payload (Tables 51/52; the
        // §5.7.5 companding gain is not applied on this route — the
        // in-crate writer signals compand-off).
        let num_sb = crate::qmf::NUM_QMF_SUBBANDS;
        let num_ts = n / num_sb;
        let mut x = vec![vec![vec![(0.0f64, 0.0f64); self.num_dmx]; num_sb]; num_ts];
        let aspx_elem = ajoc
            .var_element
            .as_ref()
            .filter(|e| e.aspx_mode)
            .map(|e| {
                let cfg = e
                    .aspx_config
                    .or(self.sticky_aspx.map(|(c, _)| c))
                    .ok_or_else(|| Error::invalid("ac4: A-SPX downmix without a config"))?;
                Ok::<_, Error>((e, cfg))
            })
            .transpose()?;
        for ch in 0..self.num_dmx {
            // A-SPX extension path: payload mapping pair p → channels
            // (2p, 2p+1), trailing aspx_data_1ch → the odd channel.
            if let Some((elem, cfg)) = &aspx_elem {
                let n_paired = 2 * elem.aspx_pair_trailers.len();
                let trailer = if ch < n_paired {
                    elem.aspx_pair_trailers[ch / 2]
                        .as_deref()
                        .map(|t| (t, ch % 2 == 1))
                } else {
                    elem.aspx_odd_trailer.as_deref().map(|t| (t, false))
                };
                if let Some((trailer, is_secondary)) = trailer {
                    while self.aspx_ext_state.len() <= ch {
                        self.aspx_ext_state
                            .push(crate::aspx::AspxChannelExtState::new());
                    }
                    let chd = if is_secondary {
                        trailer.secondary.as_ref().unwrap_or(&trailer.primary)
                    } else {
                        &trailer.primary
                    };
                    if let Some((q, _sb0, _sb1)) = crate::decoder::Ac4Decoder::aspx_extend_to_qmf(
                        &pcm_dmx[ch],
                        &trailer.frequency_tables,
                        cfg,
                        Some(&chd.framing),
                        Some(&chd.data_sig),
                        Some(&chd.data_noise),
                        Some(chd.qmode_env),
                        Some(&chd.delta_dir),
                        chd.add_harmonic.as_deref(),
                        chd.tna_mode.as_deref(),
                        &mut self.aspx_ext_state[ch],
                        crate::aspx::num_ts_in_ats(frame_len_base.max(1)),
                    ) {
                        for (ts, x_ts) in x.iter_mut().enumerate().take(num_ts) {
                            for (sb, row) in q.iter().enumerate().take(num_sb) {
                                let (re, im) = row.get(ts).copied().unwrap_or((0.0, 0.0));
                                x_ts[sb][ch] = (re as f64, im as f64);
                            }
                        }
                        continue;
                    }
                }
            }
            let slots = self.analysis[ch].process_block(&pcm_dmx[ch]);
            for (ts, slot) in slots.iter().take(num_ts).enumerate() {
                for (sb, &(re, im)) in slot.iter().enumerate() {
                    x[ts][sb][ch] = (re as f64, im as f64);
                }
            }
        }
        // 3. Spatial reconstruction.
        let frame = &ajoc.ajoc_frame;
        let geom = crate::ajoc::AjocGeometry {
            num_dmx: self.num_dmx,
            num_umx: self.num_umx,
            num_decorr: frame.num_decorr as usize,
            num_timeslots: num_ts,
            num_subbands: num_sb,
        };
        let state_matches = matches!(
            &self.recon,
            Some((g, _)) if g.num_dmx == geom.num_dmx
                && g.num_umx == geom.num_umx
                && g.num_decorr == geom.num_decorr
                && g.num_subbands == geom.num_subbands
        );
        if !state_matches {
            self.recon = Some((geom, crate::ajoc::AjocReconState::new(&geom)));
        }
        let recon = &mut self.recon.as_mut().expect("recon state just ensured").1;
        let z = crate::ajoc::ajoc_reconstruct(
            &x,
            &frame.matrices,
            &frame.ctrl.data_point_info,
            &frame.ctrl.object_present,
            &geom,
            recon,
        );
        // 4. QMF-synthesise each object.
        let mut out = Vec::with_capacity(self.num_umx);
        for o in 0..self.num_umx {
            let mut pcm = Vec::with_capacity(num_ts * num_sb);
            for z_ts in z.iter().take(num_ts) {
                let mut slot = [(0.0f32, 0.0f32); crate::qmf::NUM_QMF_SUBBANDS];
                for (sb, s) in slot.iter_mut().enumerate() {
                    let (re, im) = z_ts[sb][o];
                    *s = (re as f32, im as f32);
                }
                let row = self.synthesis[o].process_slot(&slot);
                pcm.extend_from_slice(&row);
            }
            out.push(pcm);
        }
        Ok((out, lfe_pcm))
    }

    /// Parse + decode one complete part-2 `ac4_substream()` body
    /// (§6.2.2.2: `audio_size` + byte-aligned `audio_data_ajoc()` +
    /// `metadata(…, sus_ver = 1)`) to object PCM. Returns the PCM
    /// grid, the decoded LFE PCM (when the substream carries an LFE),
    /// the parsed body, and the post-audio metadata element.
    ///
    /// Metadata channel-mode binding: per TS 103 190-2 §6.2.7.2 the
    /// `sus_ver = 1` `basic_metadata()` gates its downmix-history /
    /// phase90 fields on `channel_mode`; an A-JOC object substream
    /// signals no channel mode, so neither the `== stereo` nor the
    /// `> stereo` gate applies — the walk uses `channel_mode = MONO`
    /// (which opens no channel-gated field) and the `sus_ver = 1`
    /// stereo-dmx block is absent by definition (the bed/custom
    /// downmix data is authoritative for objects). Substreams whose
    /// payload ends exactly at the audio envelope surface `None`.
    pub fn decode_substream_pcm(
        &mut self,
        substream_bytes: &[u8],
        params: &AjocBodyParams,
        b_iframe: bool,
        b_alternative: bool,
        frame_len_base: u32,
    ) -> Result<AjocSubstreamDecode> {
        let mut br = BitReader::new(substream_bytes);
        // §6.2.2.2: audio_size_value (15) + b_more_bits escape.
        let mut audio_size = br.read_u32(15)?;
        if br.read_bit()? {
            audio_size += variable_bits(&mut br, 7)? << 15;
        }
        br.align_to_byte();
        let audio_start = br.byte_position();
        let sticky = self.sticky_aspx;
        let ajoc = parse_audio_data_ajoc(
            &mut br,
            params,
            b_iframe,
            b_alternative,
            frame_len_base,
            &mut self.diff_state,
            sticky.as_ref().map(|(c, x)| (c, *x)),
        )?;
        // Harvest the I-frame-sticky A-SPX config + xover for the
        // P-frames that follow (single-xover convention — the first
        // payload's offset stands for the element).
        if b_iframe {
            if let Some(elem) = ajoc.var_element.as_ref().filter(|e| e.aspx_mode) {
                let first = elem
                    .aspx_pair_trailers
                    .first()
                    .and_then(|t| t.as_deref())
                    .or(elem.aspx_odd_trailer.as_deref());
                if let (Some(cfg), Some(t)) = (elem.aspx_config, first) {
                    self.sticky_aspx = Some((cfg, t.xover));
                }
            }
        }
        // The walk must stay inside the announced audio_size envelope
        // (the remainder up to it is fill_bits + byte_align).
        let consumed = br.bit_position().div_ceil(8) - audio_start as u64;
        if consumed > audio_size as u64 {
            return Err(Error::invalid(
                "ac4: audio_data_ajoc overran the announced audio_size",
            ));
        }
        // §6.2.2.2: fill_bits + byte_align pad out to audio_size, then
        // metadata(b_alternative, b_ajoc = 1, b_audio_ndot, sus_ver = 1).
        let meta_off = audio_start + audio_size as usize;
        let metadata = if meta_off < substream_bytes.len() {
            let mut mbr = BitReader::new(&substream_bytes[meta_off..]);
            let meta = crate::metadata::parse_metadata_v2(
                &mut mbr,
                crate::metadata::channel_mode::MONO,
                b_iframe,
                false,
                self.prev_de,
            )?;
            if let Some(cfg) = meta.dialog_enhancement.config {
                self.prev_de = Some(cfg);
            }
            Some(meta)
        } else {
            None
        };
        let (pcm, lfe) = self.decode_frame_pcm(&ajoc, frame_len_base)?;
        Ok((pcm, lfe, ajoc, metadata))
    }
}

// =====================================================================
// Write-side helpers (SIMPLE downmix form)
// =====================================================================

/// Write one long-frame single-window-group ASF `sf_data` body
/// (sections + spectra + scalefactors + SNF) for `coeffs` at
/// `max_sfb` bands, mirroring the decode path of
/// `decode_mch_sf_data_channels`.
pub(crate) fn write_asf_body_from_spectrum(
    bw: &mut BitWriter,
    coeffs: &[f32],
    sfbo: &[u16],
    max_sfb: u32,
) {
    use crate::encoder_asf::{
        build_band_codebook_cost_table, build_sections_from_dp,
        compute_snf_dpcm_for_zero_quant_bands, dp_optimise_sections, pick_best_codebook_for_band,
        write_scalefac_data, write_section_data, write_snf_data, write_spectral_data_sections,
    };
    let end_bin = sfbo[max_sfb as usize] as usize;
    let mut qspec = vec![0i32; end_bin];
    let mut sf_per_band = vec![100i32; max_sfb as usize];
    let mut max_quant_idx = vec![0u32; max_sfb as usize];
    let mut natural_q_per_band: Vec<Vec<i32>> = Vec::with_capacity(max_sfb as usize);
    for sfb in 0..max_sfb as usize {
        let a = sfbo[sfb] as usize;
        let b = (sfbo[sfb + 1] as usize).min(coeffs.len());
        let band = &coeffs[a..b.max(a)];
        let (_cb, sf, q, _cost) = pick_best_codebook_for_band(band);
        sf_per_band[sfb] = sf;
        let mut max_q = 0u32;
        for (i, &qi) in q.iter().enumerate() {
            qspec[a + i] = qi;
            max_q = max_q.max(qi.unsigned_abs());
        }
        max_quant_idx[sfb] = max_q;
        natural_q_per_band.push(q);
    }
    let cost_table = build_band_codebook_cost_table(&natural_q_per_band);
    let dp_sections = dp_optimise_sections(&cost_table, 16);
    let sections = build_sections_from_dp(&dp_sections, max_sfb);
    let snf = compute_snf_dpcm_for_zero_quant_bands(
        coeffs,
        sfbo,
        max_sfb,
        &sections.sfb_cb,
        &max_quant_idx,
    );
    write_section_data(bw, &sections);
    write_spectral_data_sections(bw, &qspec, sfbo, &sections);
    write_scalefac_data(bw, &sf_per_band, &sections.sfb_cb, &max_quant_idx, max_sfb);
    write_snf_data(
        bw,
        snf.as_deref(),
        &sections.sfb_cb,
        &max_quant_idx,
        max_sfb,
    );
}

/// Write a SIMPLE long-frame `two_channel_data()` (Table 26) carrying
/// the two channels' MDCT spectra with independent coding
/// (`sap_mode = 0`).
pub fn write_two_channel_data_simple(
    bw: &mut BitWriter,
    left: &[f32],
    right: &[f32],
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    // asf_transform_info(): b_long_frame = 1.
    bw.write_bit(true);
    // asf_psy_info(): max_sfb.
    bw.write_u32(max_sfb, n_msfb_bits);
    // chparam_info(): sap_mode = 0 (independent channels).
    bw.write_u32(0, 2);
    write_asf_body_from_spectrum(bw, left, sfbo, max_sfb);
    write_asf_body_from_spectrum(bw, right, sfbo, max_sfb);
    Ok(())
}

/// Write a SIMPLE long-frame `mono_data(0)` (Table 21) carrying one
/// channel's MDCT spectrum via the ASF frontend.
pub fn write_mono_data_simple(
    bw: &mut BitWriter,
    coeffs: &[f32],
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    // spec_frontend = ASF (0).
    bw.write_bit(false);
    // asf_transform_info(): b_long_frame = 1.
    bw.write_bit(true);
    // sf_info(): max_sfb.
    bw.write_u32(max_sfb, n_msfb_bits);
    write_asf_body_from_spectrum(bw, coeffs, sfbo, max_sfb);
    Ok(())
}

/// Write a SIMPLE long-frame LFE `mono_data(1)` (Table 21): no
/// `spec_frontend` bit, `sf_info_lfe()` with the Table 106
/// `n_msfbl_bits` width for `max_sfb`, then the standard long-frame
/// ASF body.
pub fn write_mono_data_lfe_simple(
    bw: &mut BitWriter,
    coeffs: &[f32],
    transform_length: u32,
    max_sfb_lfe: u32,
) -> Result<()> {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    let (_nm, _ns, n_msfbl_bits) = crate::tables::n_msfb_bits_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    if n_msfbl_bits == 0 {
        return Err(Error::invalid(
            "ac4: transform_length not permitted for LFE",
        ));
    }
    if max_sfb_lfe >= (1 << n_msfbl_bits) {
        return Err(Error::invalid("ac4: LFE max_sfb out of range"));
    }
    // asf_transform_info(): b_long_frame = 1 (forced for LFE).
    bw.write_bit(true);
    // sf_info_lfe(): max_sfb (n_msfbl_bits wide).
    bw.write_u32(max_sfb_lfe, n_msfbl_bits);
    write_asf_body_from_spectrum(bw, coeffs, sfbo, max_sfb_lfe);
    Ok(())
}

/// Write a SIMPLE `var_channel_element()` (§6.2.4.4,
/// `var_codec_mode = 0`) for `spectra.len()` fullband signals, with an
/// optional leading LFE `mono_data(1)` — `lfe` is
/// `(coeffs, max_sfb_lfe)` and must match the descriptor's `b_lfe`.
pub fn write_var_channel_element_simple_lfe(
    bw: &mut BitWriter,
    lfe: Option<(&[f32], u32)>,
    spectra: &[&[f32]],
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let n = spectra.len();
    if n == 0 {
        return Err(Error::invalid("ac4: var_channel_element needs signals"));
    }
    // var_codec_mode = Simple.
    bw.write_bit(false);
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        write_mono_data_lfe_simple(bw, coeffs, transform_length, max_sfb_lfe)?;
    }
    write_var_channel_element_simple_tail(bw, spectra, transform_length, max_sfb)
}

/// Write a SIMPLE `var_channel_element()` (§6.2.4.4,
/// `var_codec_mode = 0`, no LFE) for `spectra.len()` fullband signals.
pub fn write_var_channel_element_simple(
    bw: &mut BitWriter,
    spectra: &[&[f32]],
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let n = spectra.len();
    if n == 0 {
        return Err(Error::invalid("ac4: var_channel_element needs signals"));
    }
    // var_codec_mode = Simple.
    bw.write_bit(false);
    write_var_channel_element_simple_tail(bw, spectra, transform_length, max_sfb)
}

fn write_var_channel_element_simple_tail(
    bw: &mut BitWriter,
    spectra: &[&[f32]],
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let n = spectra.len();
    let n_pairs = n / 2;
    if n % 2 == 1 {
        if n == 1 {
            write_mono_data_simple(bw, spectra[0], transform_length, max_sfb)?;
        } else {
            for p in 0..n_pairs - 1 {
                write_two_channel_data_simple(
                    bw,
                    spectra[2 * p],
                    spectra[2 * p + 1],
                    transform_length,
                    max_sfb,
                )?;
            }
            // var_coding_config = 0: pair + mono.
            bw.write_bit(false);
            write_two_channel_data_simple(
                bw,
                spectra[n - 3],
                spectra[n - 2],
                transform_length,
                max_sfb,
            )?;
            write_mono_data_simple(bw, spectra[n - 1], transform_length, max_sfb)?;
        }
    } else {
        for p in 0..n_pairs {
            write_two_channel_data_simple(
                bw,
                spectra[2 * p],
                spectra[2 * p + 1],
                transform_length,
                max_sfb,
            )?;
        }
    }
    Ok(())
}

/// A minimal I-frame-compatible `oamd_timing_data()`: zero sample
/// offset, one object-info block with a zero ramp.
pub fn minimal_oamd_timing() -> OamdTimingData {
    OamdTimingData {
        sample_offset: OaSampleOffset::Zero,
        blocks: vec![OamdTimingBlock {
            block_offset_factor: 0,
            ramp_duration: RampDuration::Zero,
        }],
    }
}

/// An all-inactive `oamd_dyndata_single()` payload: one
/// `object_info_block` per object with `b_object_not_active = 1`.
pub fn inactive_dyndata(n_objs: usize, n_blocks: usize) -> OamdDynDataSingle {
    let block = ObjectInfoBlock {
        b_object_not_active: true,
        basic_status: InfoStatus::Default,
        basic_info: None,
        render_status: InfoStatus::Default,
        render_info: None,
        add_table_data: None,
    };
    OamdDynDataSingle {
        object_blocks: vec![vec![block; n_blocks]; n_objs],
        alt: None,
    }
}

/// Write a complete dynamic-form `audio_data_ajoc()` with a SIMPLE
/// downmix, minimal OAMD (one inactive info block per object), no DE
/// coefficients, and the given `ajoc()` element payload.
///
/// `dmx_spectra` carries the fullband downmix MDCT spectra. When
/// `params.b_lfe` is set, `lfe` supplies the LFE `mono_data(1)` MDCT
/// spectrum plus its `max_sfb` (Table 106 `n_msfbl_bits` domain).
#[allow(clippy::too_many_arguments)]
pub fn write_audio_data_ajoc_simple(
    bw: &mut BitWriter,
    params: &AjocBodyParams,
    dmx_spectra: &[&[f32]],
    lfe: Option<(&[f32], u32)>,
    transform_length: u32,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<()> {
    if params.b_static_dmx {
        return Err(Error::invalid(
            "ac4: SIMPLE audio_data_ajoc writer covers the dynamic form",
        ));
    }
    if params.b_lfe != lfe.is_some() {
        return Err(Error::invalid(
            "ac4: LFE spectrum presence must match the descriptor's b_lfe",
        ));
    }
    if dmx_spectra.len() != params.n_fullband_dmx_signals as usize {
        return Err(Error::invalid("ac4: dmx spectra count mismatch"));
    }
    // b_some_signals_inactive = 0.
    bw.write_bit(false);
    write_var_channel_element_simple_lfe(bw, lfe, dmx_spectra, transform_length, max_sfb)?;
    write_ajoc_dyn_oamd_mid(bw, params, b_iframe)?;
    write_ajoc_tail(bw, params, num_decorr, ctrl, qmats, b_iframe, enc_state)
}

/// Dynamic-form middle section of `audio_data_ajoc()`: `b_dmx_timing`
/// plus minimal core OAMD (one inactive info block per downmix
/// object) plus `b_oamd_extension_present = 0`.
fn write_ajoc_dyn_oamd_mid(
    bw: &mut BitWriter,
    params: &AjocBodyParams,
    b_iframe: bool,
) -> Result<()> {
    // b_dmx_timing = 1 + minimal timing (1 block).
    bw.write_bit(true);
    let timing = minimal_oamd_timing();
    write_oamd_timing_data(bw, &timing)?;
    let n_dmx = params.n_fullband_dmx_signals as usize + usize::from(params.b_lfe);
    let is_lfe_dmx = params.is_lfe(n_dmx);
    write_oamd_dyndata_single(
        bw,
        &inactive_dyndata(n_dmx, 1),
        b_iframe,
        &params.obj_type_dmx,
        &is_lfe_dmx,
    )?;
    // b_oamd_extension_present = 0.
    bw.write_bit(false);
    Ok(())
}

/// Shared tail of `audio_data_ajoc()`: the `ajoc()` element,
/// `ajoc_dmx_de_data()`, and the full-decode OAMD timing + dyndata.
fn write_ajoc_tail(
    bw: &mut BitWriter,
    params: &AjocBodyParams,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<()> {
    // ajoc().
    crate::encoder_ajoc::encode_ajoc(bw, num_decorr, ctrl, qmats, b_iframe, enc_state)?;
    // ajoc_dmx_de_data(): b_dmx_de_cfg = 0, b_keep_dmx_de_coeffs = 1.
    bw.write_bit(false);
    bw.write_bit(true);
    // b_umx_timing = 0, b_derive_timing_from_dmx = 1.
    bw.write_bit(false);
    bw.write_bit(true);
    let n_umx = params.n_fullband_upmix_signals as usize + usize::from(params.b_lfe);
    let is_lfe_umx = params.is_lfe(n_umx);
    write_oamd_dyndata_single(
        bw,
        &inactive_dyndata(n_umx, 1),
        b_iframe,
        &params.obj_type_umx,
        &is_lfe_umx,
    )?;
    Ok(())
}

/// Write a complete **static-downmix** `audio_data_ajoc()`
/// (`b_static_dmx == 1`): the part-1 `audio_data_chan(b_lfe ? 5.1 :
/// 5.0)` core in the SIMPLE / Cfg3Five form (Table 25 + Table 29,
/// spectra in `[L, R, C, Ls, Rs]` order, optional LFE `mono_data(1)`)
/// followed by the `ajoc()` tail. Per §6.2.3.4 the static form
/// carries no signals-inactive mask, no var element, and no core OAMD
/// block.
#[allow(clippy::too_many_arguments)]
pub fn write_audio_data_ajoc_static(
    bw: &mut BitWriter,
    params: &AjocBodyParams,
    chan_spectra: &[&[f32]; 5],
    lfe: Option<(&[f32], u32)>,
    transform_length: u32,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<()> {
    if !params.b_static_dmx || params.n_fullband_dmx_signals != 5 {
        return Err(Error::invalid(
            "ac4: static audio_data_ajoc writer needs b_static_dmx with 5 core signals",
        ));
    }
    if params.b_lfe != lfe.is_some() {
        return Err(Error::invalid(
            "ac4: LFE spectrum presence must match the descriptor's b_lfe",
        ));
    }
    // audio_data_chan(5.X): 5_X_codec_mode = SIMPLE (3 bits), no
    // I-frame config block, optional LFE, coding_config = 3.
    bw.write_u32(0, 3);
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        write_mono_data_lfe_simple(bw, coeffs, transform_length, max_sfb_lfe)?;
    }
    bw.write_u32(3, 2);
    crate::ice::write_five_channel_data_simple(bw, chan_spectra, transform_length, max_sfb)?;
    write_ajoc_tail(bw, params, num_decorr, ctrl, qmats, b_iframe, enc_state)
}

/// Write a complete dynamic-form `audio_data_ajoc()` whose downmix is
/// the **A-SPX** `var_channel_element()` form (`var_codec_mode = 1`):
/// I-frame `aspx_config()`, compand-off `companding_control()`
/// (`n_fb_dmx_signals <= 5`), the paired channel data, and one
/// floored-noise single-envelope `aspx_data_2ch()` per pair (see
/// [`crate::ice::write_ice_body_ajcc`] on the noise floor). Even
/// fullband signal counts only.
#[allow(clippy::too_many_arguments)]
pub fn write_audio_data_ajoc_aspx(
    bw: &mut BitWriter,
    params: &AjocBodyParams,
    dmx_spectra: &[&[f32]],
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    transform_length: u32,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<()> {
    if params.b_static_dmx {
        return Err(Error::invalid(
            "ac4: A-SPX audio_data_ajoc writer covers the dynamic form",
        ));
    }
    if params.b_lfe != lfe.is_some() {
        return Err(Error::invalid(
            "ac4: LFE spectrum presence must match the descriptor's b_lfe",
        ));
    }
    let n = params.n_fullband_dmx_signals as usize;
    if dmx_spectra.len() != n || n == 0 || n % 2 != 0 {
        return Err(Error::invalid(
            "ac4: A-SPX downmix writer covers even fullband signal counts",
        ));
    }
    // b_some_signals_inactive = 0.
    bw.write_bit(false);
    // var_channel_element(): var_codec_mode = A-SPX.
    bw.write_bit(true);
    if b_iframe {
        crate::encoder_acpl3::write_aspx_config(bw, aspx_cfg);
    }
    if n <= 5 {
        // companding_control(n): sync_flag = 1, b_compand_on[0] = 0,
        // b_compand_avg = 0.
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(false);
    }
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        write_mono_data_lfe_simple(bw, coeffs, transform_length, max_sfb_lfe)?;
    }
    for p in 0..n / 2 {
        write_two_channel_data_simple(
            bw,
            dmx_spectra[2 * p],
            dmx_spectra[2 * p + 1],
            transform_length,
            max_sfb,
        )?;
    }
    // Per-pair aspx_data_2ch(): single-envelope FIXFIX with a floored
    // noise envelope (qnoise = 30) so the regenerated band carries no
    // decoder-side noise-floor energy.
    let tables = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0)
        .map_err(|_| Error::invalid("ac4: aspx frequency-table derivation failed"))?;
    let counts = tables.counts;
    let num_sbg_sig = if aspx_cfg.signals_freq_res() {
        counts.num_sbg_sig_lowres
    } else {
        counts.num_sbg_sig_highres
    } as usize;
    let num_sbg_noise = counts.num_sbg_noise as usize;
    let sig_row = vec![0i32; num_sbg_sig.max(1)];
    let mut noise_row = vec![0i32; num_sbg_noise.max(1)];
    noise_row[0] = 30;
    let bal_sig = vec![0i32; num_sbg_sig.max(1)];
    let bal_noise = vec![0i32; num_sbg_noise.max(1)];
    let tna = vec![0u8; num_sbg_noise];
    let ch_level = crate::encoder_acpl3::AspxRealEnvelopeChannel {
        sig: &sig_row,
        noise: &noise_row,
    };
    let ch_bal = crate::encoder_acpl3::AspxRealEnvelopeChannel {
        sig: &bal_sig,
        noise: &bal_noise,
    };
    for _ in 0..n / 2 {
        crate::encoder_acpl3::write_aspx_data_2ch_real_envelope_tna_ah_framed(
            bw,
            aspx_cfg,
            ch_level,
            ch_bal,
            &tna,
            &[],
            &[],
            b_iframe,
        )
        .map_err(Error::invalid)?;
    }
    write_ajoc_dyn_oamd_mid(bw, params, b_iframe)?;
    write_ajoc_tail(bw, params, num_decorr, ctrl, qmats, b_iframe, enc_state)
}

/// Assemble a complete part-2 `ac4_substream()` (§6.2.2.2) around a
/// SIMPLE `audio_data_ajoc()` body: `audio_size` header, byte-aligned
/// audio data, zero fill to `audio_size`. See
/// [`write_audio_data_ajoc_simple`] for the body arguments.
#[allow(clippy::too_many_arguments)]
pub fn write_ajoc_substream_simple(
    params: &AjocBodyParams,
    dmx_spectra: &[&[f32]],
    lfe: Option<(&[f32], u32)>,
    transform_length: u32,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<Vec<u8>> {
    let mut body = BitWriter::new();
    write_audio_data_ajoc_simple(
        &mut body,
        params,
        dmx_spectra,
        lfe,
        transform_length,
        max_sfb,
        num_decorr,
        ctrl,
        qmats,
        b_iframe,
        enc_state,
    )?;
    wrap_ajoc_substream(body)
}

/// Wrap an `audio_data_ajoc()` body into a complete part-2
/// `ac4_substream()`: `audio_size` header, byte-aligned audio data,
/// and the minimal `metadata(…, sus_ver = 1)` tail.
pub fn wrap_ajoc_substream(body: BitWriter) -> Result<Vec<u8>> {
    let mut body = body;
    body.align_to_byte();
    let body_bytes = body.into_bytes();
    let audio_size = body_bytes.len() as u32;
    if audio_size >= 1 << 15 {
        return Err(Error::invalid("ac4: A-JOC audio_data too large"));
    }
    let mut bw = BitWriter::new();
    bw.write_u32(audio_size, 15);
    bw.write_bit(false); // b_more_bits
    bw.align_to_byte();
    let mut out = bw.into_bytes();
    out.extend_from_slice(&body_bytes);
    // §6.2.2.2 tail: metadata(b_alternative = 0, b_ajoc = 1,
    // b_audio_ndot, sus_ver = 1) + byte_align — the minimal form
    // (object substreams carry no channel-gated basic_metadata fields
    // and no drc_frame per §6.2.7.1/2).
    let mut mw = BitWriter::new();
    mw.write_bit(false); // basic_metadata(…, 1): b_more_basic_metadata
    mw.write_bit(false); // extended_metadata(…, 1): b_dialog
    mw.write_bit(false); // b_channels_classifier
    mw.write_bit(false); // b_event_probability
    mw.write_u32(1, 7); // tools_metadata_size = 1 (DE only — no DRC)
    mw.write_bit(false); // b_more_bits
    mw.write_bit(false); // dialog_enhancement(): b_de_data_present
    mw.write_bit(false); // b_emdf_payloads_substream
    mw.align_to_byte();
    out.extend_from_slice(&mw.into_bytes());
    Ok(out)
}

/// Emit a complete v2 `raw_ac4_frame()` carrying one A-JOC object
/// substream: `ac4_toc()` (single presentation, single substream
/// group, `b_channel_coded = 0`, `b_ajoc = 1` with an all-dynamic
/// `ac4_substream_info_ajoc()`) followed by the byte-aligned
/// `ac4_substream()` from [`write_ajoc_substream_simple`].
///
/// Fixed shape: `fs_index = 1` (48 kHz), `frame_rate_index = 1`
/// (1920-sample frames), no LFE, no HSF extension.
#[allow(clippy::too_many_arguments)]
pub fn encode_ajoc_raw_frame(
    sequence_counter: u32,
    params: &AjocBodyParams,
    dmx_spectra: &[&[f32]],
    lfe: Option<(&[f32], u32)>,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<Vec<u8>> {
    if params.b_static_dmx {
        return Err(Error::invalid(
            "ac4: frame writer covers the dynamic A-JOC form",
        ));
    }
    if params.b_lfe != lfe.is_some() {
        return Err(Error::invalid(
            "ac4: LFE spectrum presence must match the descriptor's b_lfe",
        ));
    }
    let mut frame = write_ajoc_toc(sequence_counter, params, b_iframe)?;
    let substream = write_ajoc_substream_simple(
        params,
        dmx_spectra,
        lfe,
        1920,
        max_sfb,
        num_decorr,
        ctrl,
        qmats,
        b_iframe,
        enc_state,
    )?;
    frame.extend_from_slice(&substream);
    Ok(frame)
}

/// Emit the byte-aligned v2 `ac4_toc()` for a single A-JOC object
/// substream (single presentation / single substream group,
/// `b_channel_coded = 0`, `b_ajoc = 1`, `fs_index = 1`,
/// `frame_rate_index = 1`). Handles both the dynamic and the
/// `b_static_dmx` descriptor forms of `ac4_substream_info_ajoc()`.
pub fn write_ajoc_toc(
    sequence_counter: u32,
    params: &AjocBodyParams,
    b_iframe: bool,
) -> Result<Vec<u8>> {
    if params.n_fullband_upmix_signals >= 16 || params.n_fullband_dmx_signals > 16 {
        return Err(Error::invalid("ac4: frame writer signal-count limits"));
    }
    let mut bw = BitWriter::new();
    // ---- ac4_toc() (§6.2.1.1) ----
    bw.write_u32(2, 2); // bitstream_version = 2
    bw.write_u32(sequence_counter & 0x3FF, 10);
    bw.write_bit(false); // b_wait_frames
    bw.write_u32(1, 1); // fs_index = 48 kHz
    bw.write_u32(1, 4); // frame_rate_index = 1 (1920 samples)
    bw.write_bit(b_iframe); // b_iframe_global
    bw.write_bit(true); // b_single_presentation
    bw.write_bit(false); // b_payload_base
    bw.write_bit(false); // b_program_id
                         // ---- ac4_presentation_v1_info() (§6.2.1.3) ----
    bw.write_bit(true); // b_single_substream_group
    bw.write_bit(false); // presentation_version() = 0
    bw.write_u32(0, 3); // mdcompat
    bw.write_bit(false); // b_presentation_id
    bw.write_bit(false); // frame_rate_multiply_info: b_multiplier = 0
                         // emdf_info(): version 0, key_id 0, no payloads-substream, no more.
    bw.write_u32(0, 2);
    bw.write_u32(0, 3);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false); // b_presentation_filter
    bw.write_u32(0, 3); // ac4_sgi_specifier(): group_index = 0
    bw.write_bit(false); // b_pre_virtualized
    bw.write_bit(false); // b_add_emdf_substreams
                         // ac4_presentation_substream_info(): b_alternative = 0,
                         // b_pres_ndot = b_iframe (§6.3.2.11.2 — "no
                         // dependency over time"), substream_index = 0.
    bw.write_bit(false);
    bw.write_bit(b_iframe);
    bw.write_u32(0, 2);
    // ---- ac4_substream_group_info() (§6.2.1.6) ----
    bw.write_bit(true); // b_substreams_present
    bw.write_bit(false); // b_hsf_ext
    bw.write_bit(true); // b_single_substream
    bw.write_bit(false); // b_channel_coded
    bw.write_bit(false); // b_oamd_substream
    bw.write_bit(true); // b_ajoc
                        // ac4_substream_info_ajoc() (§6.2.1.9):
    bw.write_bit(params.b_lfe); // b_lfe
    bw.write_bit(params.b_static_dmx); // b_static_dmx
    if !params.b_static_dmx {
        bw.write_u32(params.n_fullband_dmx_signals - 1, 4);
        bw.write_bit(true); // dmx bed_dyn_obj_assignment: b_dyn_objects_only
    }
    bw.write_bit(false); // b_oamd_common_data_present
    bw.write_u32(params.n_fullband_upmix_signals - 1, 4);
    bw.write_bit(true); // umx bed_dyn_obj_assignment: b_dyn_objects_only
    bw.write_bit(false); // b_sf_multiplier (fs_index == 1)
    bw.write_bit(false); // b_bitrate_info
    bw.write_bit(b_iframe); // b_audio_ndot = b_iframe (frame_rate_factor = 1)
    bw.write_u32(0, 2); // substream_index
    bw.write_bit(false); // b_content_type
                         // ---- substream_index_table() ----
    bw.write_u32(1, 2); // n_substreams = 1
    bw.write_bit(false); // b_size_present = 0 (spans to frame end)
    bw.align_to_byte();
    Ok(bw.into_bytes())
}

/// Emit a complete v2 `raw_ac4_frame()` carrying one **static-downmix**
/// A-JOC object substream (`b_static_dmx = 1` — the 5-signal 5.X core;
/// see [`write_audio_data_ajoc_static`]).
#[allow(clippy::too_many_arguments)]
pub fn encode_ajoc_raw_frame_static(
    sequence_counter: u32,
    params: &AjocBodyParams,
    chan_spectra: &[&[f32]; 5],
    lfe: Option<(&[f32], u32)>,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<Vec<u8>> {
    let mut frame = write_ajoc_toc(sequence_counter, params, b_iframe)?;
    let mut body = BitWriter::new();
    write_audio_data_ajoc_static(
        &mut body,
        params,
        chan_spectra,
        lfe,
        1920,
        max_sfb,
        num_decorr,
        ctrl,
        qmats,
        b_iframe,
        enc_state,
    )?;
    frame.extend_from_slice(&wrap_ajoc_substream(body)?);
    Ok(frame)
}

/// Emit a complete v2 `raw_ac4_frame()` carrying one A-JOC object
/// substream whose downmix uses the **A-SPX** `var_channel_element()`
/// form (see [`write_audio_data_ajoc_aspx`]).
#[allow(clippy::too_many_arguments)]
pub fn encode_ajoc_raw_frame_aspx(
    sequence_counter: u32,
    params: &AjocBodyParams,
    dmx_spectra: &[&[f32]],
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<Vec<u8>> {
    let mut frame = write_ajoc_toc(sequence_counter, params, b_iframe)?;
    let mut body = BitWriter::new();
    write_audio_data_ajoc_aspx(
        &mut body,
        params,
        dmx_spectra,
        aspx_cfg,
        lfe,
        1920,
        max_sfb,
        num_decorr,
        ctrl,
        qmats,
        b_iframe,
        enc_state,
    )?;
    frame.extend_from_slice(&wrap_ajoc_substream(body)?);
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ajoc::{AjocDataPointInfo, AjocQuantMode};
    use crate::ajoc_data::new_ajoc_diff_state;
    use crate::encoder_ajoc::AjocQuantMatrices;
    use crate::oamd::ObjType;

    const TL: u32 = 1920;
    const MAX_SFB: u32 = 20;

    fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
        let sfbo = crate::sfb_offset::sfb_offset_48(TL).unwrap();
        let end = sfbo[MAX_SFB as usize] as usize;
        let mut v = vec![0.0f32; end];
        v[bin.min(end - 1)] = amp;
        v[bin.min(end - 1) / 2] = amp * 0.25;
        v
    }

    fn simple_ctrl(num_umx: usize, num_dmx: usize, num_decorr: usize) -> crate::ajoc::AjocCtrlInfo {
        crate::ajoc::AjocCtrlInfo {
            decorr_enable: vec![true; num_decorr],
            object_present: vec![true; num_umx],
            data_point_info: AjocDataPointInfo {
                num_dpoints: 1,
                start_pos: vec![0],
                ramp_len: vec![16],
            },
            num_bands_code: vec![7; num_umx],
            num_bands: vec![1; num_umx],
            quant_select: vec![AjocQuantMode::Fine; num_umx],
            sparse_select: vec![false; num_umx],
            mix_mtx_dry_present: vec![vec![true; num_dmx]; num_umx],
            mix_mtx_wet_present: vec![vec![true; num_decorr]; num_umx],
        }
    }

    #[test]
    fn var_channel_element_simple_round_trips_even_count() {
        let s0 = tone_spectrum(12, 40.0);
        let s1 = tone_spectrum(30, 25.0);
        let s2 = tone_spectrum(55, 18.0);
        let s3 = tone_spectrum(90, 12.0);
        let spectra: Vec<&[f32]> = vec![&s0, &s1, &s2, &s3];
        let mut bw = BitWriter::new();
        write_var_channel_element_simple(&mut bw, &spectra, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let elem = parse_var_channel_element(&mut br, true, 4, false, TL, None).unwrap();
        assert!(!elem.aspx_mode);
        assert_eq!(elem.pairs.len(), 2);
        let got = elem.fullband_spectra();
        assert_eq!(got.len(), 4);
        for (i, spec) in got.iter().enumerate() {
            let spec = spec.expect("decodable channel body");
            // The written tone must survive quantisation at its bin.
            let orig = &spectra[i];
            let peak_bin = orig
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap()
                .0;
            assert!(
                spec[peak_bin].abs() > 0.0,
                "channel {i} lost its tone at bin {peak_bin}"
            );
        }
    }

    #[test]
    fn var_channel_element_simple_round_trips_odd_counts() {
        // n = 1: bare mono.
        let s0 = tone_spectrum(20, 30.0);
        let spectra: Vec<&[f32]> = vec![&s0];
        let mut bw = BitWriter::new();
        write_var_channel_element_simple(&mut bw, &spectra, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let elem = parse_var_channel_element(&mut br, true, 1, false, TL, None).unwrap();
        assert_eq!(elem.fullband_spectra().len(), 1);
        assert!(matches!(
            elem.odd_tail.as_deref(),
            Some(VarOddTail::Mono(_))
        ));

        // n = 5: one pair + var_coding_config = 0 (pair + mono).
        let s: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 13 * i, 20.0)).collect();
        let spectra: Vec<&[f32]> = s.iter().map(|v| v.as_slice()).collect();
        let mut bw = BitWriter::new();
        write_var_channel_element_simple(&mut bw, &spectra, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let elem = parse_var_channel_element(&mut br, true, 5, false, TL, None).unwrap();
        assert_eq!(elem.pairs.len(), 1);
        assert!(matches!(
            elem.odd_tail.as_deref(),
            Some(VarOddTail::PairPlusMono { .. })
        ));
        assert_eq!(elem.fullband_spectra().len(), 5);
    }

    #[test]
    fn audio_data_ajoc_simple_round_trips_matrices_and_spectra() {
        let num_dmx = 4usize;
        let num_umx = 6usize;
        let num_decorr = 2usize;
        let params = AjocBodyParams {
            b_lfe: false,
            b_static_dmx: false,
            n_fullband_dmx_signals: num_dmx as u32,
            n_fullband_upmix_signals: num_umx as u32,
            obj_type_dmx: vec![ObjType::Dyn; num_dmx],
            obj_type_umx: vec![ObjType::Dyn; num_umx],
        };
        let ctrl = simple_ctrl(num_umx, num_dmx, num_decorr);
        // Real-valued matrices on the 1-band grid.
        let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|o| {
                vec![(0..num_dmx)
                    .map(|ch| vec![0.1 * (o as f64 + 1.0) - 0.05 * ch as f64])
                    .collect()]
            })
            .collect();
        let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|o| {
                vec![(0..num_decorr)
                    .map(|de| vec![0.05 * (o as f64) + 0.02 * de as f64])
                    .collect()]
            })
            .collect();
        let qmats = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);

        let s: Vec<Vec<f32>> = (0..num_dmx)
            .map(|i| tone_spectrum(15 + 20 * i, 30.0))
            .collect();
        let spectra: Vec<&[f32]> = s.iter().map(|v| v.as_slice()).collect();

        let mut enc_state = new_ajoc_diff_state(num_umx, num_dmx, num_decorr);
        let mut bw = BitWriter::new();
        write_audio_data_ajoc_simple(
            &mut bw,
            &params,
            &spectra,
            None,
            TL,
            MAX_SFB,
            num_decorr as u32,
            &ctrl,
            &qmats,
            true,
            &mut enc_state,
        )
        .unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();

        let mut dec_state = new_ajoc_diff_state(num_umx, num_dmx, num_decorr);
        let mut br = BitReader::new(&bytes);
        let ajoc =
            parse_audio_data_ajoc(&mut br, &params, true, false, TL, &mut dec_state, None).unwrap();

        // Downmix spectra survive.
        let elem = ajoc.var_element.as_ref().unwrap();
        assert_eq!(elem.fullband_spectra().len(), num_dmx);
        assert!(elem
            .fullband_spectra()
            .iter()
            .all(|s| s.map(|v| v.iter().any(|&x| x != 0.0)).unwrap_or(false)));

        // OAMD side info round-trips.
        assert_eq!(ajoc.dmx_timing, Some(minimal_oamd_timing()));
        assert_eq!(ajoc.dmx_dyndata, Some(inactive_dyndata(num_dmx, 1)));
        assert_eq!(ajoc.umx_timing, None);
        assert_eq!(ajoc.b_derive_timing_from_dmx, Some(true));
        assert_eq!(ajoc.umx_dyndata, inactive_dyndata(num_umx, 1));
        assert!(!ajoc.dmx_de.dmx_de_cfg);
        assert!(ajoc.dmx_de.keep_dmx_de_coeffs);

        // The decoded dequantized matrices match the quantized encoder
        // grid exactly (1 band, Fine quantizers; quantize_dry returns
        // the absolute Table 45 index).
        let m = &ajoc.ajoc_frame.matrices;
        for o in 0..num_umx {
            for ch in 0..num_dmx {
                let want = crate::ajoc::dequantize_dry(
                    qmats.dry_q[o][0][ch][0] as u32,
                    AjocQuantMode::Fine,
                );
                let got = m.dry_dq[o][0][ch][0];
                assert!(
                    (got - want).abs() < 1e-12,
                    "dry[{o}][{ch}] got {got} want {want}"
                );
            }
        }
    }

    /// Full bitstream-to-PCM chain: two encoded I-frames through
    /// `AjocSubstreamDecoder`, with a dry selector matrix routing
    /// downmix channel `o % num_dmx` to object `o`. Frame 2 (all
    /// interpolators settled at target) must match a reference chain
    /// built from the same downmix PCM through identical QMF banks
    /// scaled by the dequantized coefficient.
    #[test]
    fn ajoc_substream_end_to_end_bitstream_to_object_pcm() {
        let num_dmx = 2usize;
        let num_umx = 3usize;
        let num_decorr = 1usize;
        let params = AjocBodyParams {
            b_lfe: false,
            b_static_dmx: false,
            n_fullband_dmx_signals: num_dmx as u32,
            n_fullband_upmix_signals: num_umx as u32,
            obj_type_dmx: vec![ObjType::Dyn; num_dmx],
            obj_type_umx: vec![ObjType::Dyn; num_umx],
        };
        let ctrl = simple_ctrl(num_umx, num_dmx, num_decorr);
        // Selector dry matrix (object o <- channel o % num_dmx); wet 0.
        let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|o| {
                vec![(0..num_dmx)
                    .map(|ch| vec![if ch == o % num_dmx { 1.0 } else { 0.0 }])
                    .collect()]
            })
            .collect();
        let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|_| vec![vec![vec![0.0]; num_decorr]])
            .collect();
        let qmats = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);
        // The dequantized selector gains actually applied per object.
        let gain: Vec<f64> = (0..num_umx)
            .map(|o| {
                crate::ajoc::dequantize_dry(
                    qmats.dry_q[o][0][o % num_dmx][0] as u32,
                    AjocQuantMode::Fine,
                )
            })
            .collect();

        let s: Vec<Vec<f32>> = (0..num_dmx)
            .map(|i| tone_spectrum(18 + 25 * i, 35.0))
            .collect();
        let spectra: Vec<&[f32]> = s.iter().map(|v| v.as_slice()).collect();

        let mut enc_state = new_ajoc_diff_state(num_umx, num_dmx, 7);
        let mut dec = AjocSubstreamDecoder::new(num_dmx, num_umx);

        // Reference chain state: IMDCT overlap + QMF banks per dmx
        // channel (analysis -> synthesis identity, then scale).
        let mut ref_overlap = vec![vec![0.0f32; TL as usize]; num_dmx];
        let mut ref_analysis: Vec<_> = (0..num_dmx)
            .map(|_| crate::qmf::QmfAnalysisBank::new())
            .collect();
        let mut ref_synthesis: Vec<_> = (0..num_dmx)
            .map(|_| crate::qmf::QmfSynthesisBank::new())
            .collect();
        let window = crate::mdct::kbd_window(TL);

        let mut last_objects: Vec<Vec<f32>> = Vec::new();
        let mut last_reference: Vec<Vec<f32>> = Vec::new();
        for _frame in 0..4 {
            let bytes = write_ajoc_substream_simple(
                &params,
                &spectra,
                None,
                TL,
                MAX_SFB,
                num_decorr as u32,
                &ctrl,
                &qmats,
                true,
                &mut enc_state,
            )
            .unwrap();
            let (objects, _lfe, parsed, metadata) = dec
                .decode_substream_pcm(&bytes, &params, true, false, TL)
                .unwrap();
            // The §6.2.2.2 post-audio metadata(…, sus_ver = 1) tail
            // parses to the minimal element the writer emits.
            let meta = metadata.expect("post-audio metadata present");
            assert!(!meta.basic.more_basic_metadata);
            assert!(!meta.b_dialog);
            assert_eq!(meta.tools_metadata_size, 1);
            assert!(!meta.dialog_enhancement.data_present);
            assert!(meta.emdf_payloads_substream.is_none());
            assert_eq!(objects.len(), num_umx);

            // Reference: the DECODED downmix spectra (post ASF
            // quantisation) -> IMDCT -> QMF round trip.
            let elem = parsed.var_element.as_ref().unwrap();
            let dec_spectra: Vec<Vec<f32>> = elem
                .fullband_spectra()
                .iter()
                .map(|sp| sp.unwrap().to_vec())
                .collect();
            let mut ref_ch: Vec<Vec<f32>> = Vec::with_capacity(num_dmx);
            for ch in 0..num_dmx {
                let y = crate::mdct::imdct(&{
                    let mut x = vec![0.0f32; TL as usize];
                    let copy = dec_spectra[ch].len().min(x.len());
                    x[..copy].copy_from_slice(&dec_spectra[ch][..copy]);
                    x
                });
                let pcm = crate::mdct::imdct_olap_symmetric(&y, &window, &mut ref_overlap[ch]);
                let slots = ref_analysis[ch].process_block(&pcm);
                let mut out = Vec::with_capacity(pcm.len());
                for slot in &slots {
                    out.extend_from_slice(&ref_synthesis[ch].process_slot(slot));
                }
                ref_ch.push(out);
            }
            last_reference = (0..num_umx)
                .map(|o| {
                    ref_ch[o % num_dmx]
                        .iter()
                        .map(|&v| v * gain[o] as f32)
                        .collect()
                })
                .collect();
            last_objects = objects;
        }

        // Frame 4: interpolators settled (the Table-48 one-extra-
        // increment plateau decays geometrically across frames) —
        // decoder output equals the reference selector chain.
        for o in 0..num_umx {
            let obj = &last_objects[o];
            let refo = &last_reference[o];
            assert_eq!(obj.len(), TL as usize);
            let peak = refo.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            assert!(peak > 1e-4, "reference object {o} is silent");
            let mut max_err = 0.0f32;
            for (a, b) in obj.iter().zip(refo.iter()) {
                max_err = max_err.max((a - b).abs());
            }
            assert!(
                max_err < peak * 5e-3,
                "object {o}: max_err {max_err} vs peak {peak}"
            );
        }
    }
}
