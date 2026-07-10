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
use crate::aspx::{parse_aspx_config, parse_companding_control, AspxConfig, CompandingControl};
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
    /// one [`SubstreamTools`] scratch per pair.
    pub aspx_pair_tools: Vec<Box<SubstreamTools>>,
    /// Odd-channel `aspx_data_1ch()` payload (A-SPX mode only).
    pub aspx_odd_tools: Option<Box<SubstreamTools>>,
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
            (Some(c), _) => (c, None),
            (None, Some((c, x))) => (c, Some(x)),
            (None, None) => {
                return Err(Error::invalid(
                    "ac4: non-iframe var_channel_element A-SPX without sticky config",
                ))
            }
        };
        for _ in 0..n_pairs {
            let mut tools = Box::<SubstreamTools>::default();
            if let Some(x) = sticky_xover {
                tools.aspx_xover_subband_offset = Some(x);
            }
            crate::asf::parse_aspx_data_2ch_body(br, &mut tools, cfg, b_iframe, frame_len_base)?;
            out.aspx_pair_tools.push(tools);
        }
        if b_isodd {
            let mut tools = Box::<SubstreamTools>::default();
            if let Some(x) = sticky_xover {
                tools.aspx_xover_subband_offset = Some(x);
            }
            crate::asf::parse_aspx_data_1ch_body(br, &mut tools, cfg, b_iframe, frame_len_base)?;
            out.aspx_odd_tools = Some(tools);
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
pub fn parse_audio_data_ajoc(
    br: &mut BitReader<'_>,
    params: &AjocBodyParams,
    b_iframe: bool,
    b_alternative: bool,
    frame_len_base: u32,
    ajoc_state: &mut AjocDiffState,
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
            None,
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
// Write-side helpers (SIMPLE downmix form)
// =====================================================================

/// Write one long-frame single-window-group ASF `sf_data` body
/// (sections + spectra + scalefactors + SNF) for `coeffs` at
/// `max_sfb` bands, mirroring the decode path of
/// `decode_mch_sf_data_channels`.
fn write_asf_body_from_spectrum(bw: &mut BitWriter, coeffs: &[f32], sfbo: &[u16], max_sfb: u32) {
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
/// `dmx_spectra` carries the fullband downmix MDCT spectra
/// (`params.b_lfe` must be false — the SIMPLE writer does not emit an
/// LFE `mono_data(1)`).
#[allow(clippy::too_many_arguments)]
pub fn write_audio_data_ajoc_simple(
    bw: &mut BitWriter,
    params: &AjocBodyParams,
    dmx_spectra: &[&[f32]],
    transform_length: u32,
    max_sfb: u32,
    num_decorr: u32,
    ctrl: &crate::ajoc::AjocCtrlInfo,
    qmats: &crate::encoder_ajoc::AjocQuantMatrices,
    b_iframe: bool,
    enc_state: &mut AjocDiffState,
) -> Result<()> {
    if params.b_static_dmx || params.b_lfe {
        return Err(Error::invalid(
            "ac4: SIMPLE audio_data_ajoc writer covers the dynamic non-LFE form",
        ));
    }
    if dmx_spectra.len() != params.n_fullband_dmx_signals as usize {
        return Err(Error::invalid("ac4: dmx spectra count mismatch"));
    }
    // b_some_signals_inactive = 0.
    bw.write_bit(false);
    write_var_channel_element_simple(bw, dmx_spectra, transform_length, max_sfb)?;
    // b_dmx_timing = 1 + minimal timing (1 block).
    bw.write_bit(true);
    let timing = minimal_oamd_timing();
    write_oamd_timing_data(bw, &timing)?;
    let n_dmx = params.n_fullband_dmx_signals as usize;
    let is_lfe_dmx = vec![false; n_dmx];
    write_oamd_dyndata_single(
        bw,
        &inactive_dyndata(n_dmx, 1),
        b_iframe,
        &params.obj_type_dmx,
        &is_lfe_dmx,
    )?;
    // b_oamd_extension_present = 0.
    bw.write_bit(false);
    // ajoc().
    crate::encoder_ajoc::encode_ajoc(bw, num_decorr, ctrl, qmats, b_iframe, enc_state)?;
    // ajoc_dmx_de_data(): b_dmx_de_cfg = 0, b_keep_dmx_de_coeffs = 1.
    bw.write_bit(false);
    bw.write_bit(true);
    // b_umx_timing = 0, b_derive_timing_from_dmx = 1.
    bw.write_bit(false);
    bw.write_bit(true);
    let n_umx = params.n_fullband_upmix_signals as usize;
    let is_lfe_umx = vec![false; n_umx];
    write_oamd_dyndata_single(
        bw,
        &inactive_dyndata(n_umx, 1),
        b_iframe,
        &params.obj_type_umx,
        &is_lfe_umx,
    )?;
    Ok(())
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
            parse_audio_data_ajoc(&mut br, &params, true, false, TL, &mut dec_state).unwrap();

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
}
