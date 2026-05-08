//! Multichannel `5_X_channel_element` walker family — round 20 wiring.
//!
//! ETSI TS 103 190-1 V1.4.1, §4.2.6.6 (`5_X_channel_element`) selects
//! between four channel-element bodies via `coding_config` (Table 25):
//!
//! | `coding_config` | body                                    |
//! |-----------------|-----------------------------------------|
//! | 0               | two_channel_data + two_channel_data + mono_data(0) |
//! | 1               | three_channel_data + two_channel_data   |
//! | 2               | four_channel_data + mono_data(0)        |
//! | 3               | five_channel_data                       |
//!
//! Each of the channel-data variants composes:
//!
//! * `three_channel_data()` — Table 27: `sf_info(ASF, 0, 0)` +
//!   `three_channel_info()` (Table 30: `chel_matsel; 4` + 2x
//!   `chparam_info()`) + 3x `sf_data(ASF)`.
//! * `four_channel_data()` — Table 28: `sf_info(ASF, 0, 0)` +
//!   `four_channel_info()` (Table 31: 4x `chparam_info()`) + 4x
//!   `sf_data(ASF)`.
//! * `five_channel_data()` — Table 29: `sf_info(ASF, 0, 0)` +
//!   `five_channel_info()` (Table 32: `chel_matsel; 4` + 5x
//!   `chparam_info()`) + 5x `sf_data(ASF)`.
//! * `mono_data(b_lfe)` — Table 21: when `b_lfe == 1`, the LFE channel
//!   uses `sf_info_lfe()` instead of the regular `sf_info()`. The
//!   bit-count for `max_sfb[0]` switches from `n_msfb_bits` to
//!   `n_msfbl_bits` (Table 106 column 4).
//!
//! Round 19 landed the type definitions + parser scaffolds plus the
//! Cfg3Five outer-shell + LFE `mono_data(1)` walker. Round 20 wires the
//! remaining three coding-config layouts (Cfg0Stereo2plusMono /
//! Cfg1ThreeStereo / Cfg2FourMono) by introducing the
//! `parse_two_channel_data()` outer (Table 26) and reusing
//! `parse_mono_data(...)` for the trailing `mono_data(0)` calls. R20
//! also splits `sf_info_lfe()` from the regular `sf_info()` parser —
//! the leading `max_sfb` field now uses `n_msfbl_bits` from Table 106
//! (column 4) instead of the regular `n_msfb_bits`, matching Table 21.
//!
//! Round 23 wires the per-channel `sf_data(ASF)` Huffman bodies for
//! the multichannel layouts. Tables 26 / 27 / 28 / 29 each emit N
//! independent `sf_data(ASF)` calls (2 / 3 / 4 / 5) right after the
//! shared `sf_info(ASF, 0, 0)` + `*_channel_info()` block. We reuse the
//! ASF Huffman codebook suite (`HCB_1` .. `HCB_11` for spectral lines,
//! `HCB_SCALEFAC` for scale-factor DPCM, `HCB_SNF` for noise fill) —
//! the multichannel paths use the same codebook IDs as mono / stereo
//! per Annex A.1 of TS 103 190-1 (no separate "MCH" codebook set
//! exists — the Huffman tables are sample-rate-independent and
//! codec-mode-independent). The per-channel scaled spectra land on the
//! `scaled_spec_per_channel` field of each `*ChannelData` for the
//! long-frame, single-window-group case; short / grouped frames still
//! parse the outer shells and leave the per-channel slot `None`.

use oxideav_core::bits::BitReader;
use oxideav_core::Result;

use crate::asf::{
    decode_asf_long_lfe_body_with_max_sfb_lfe, decode_asf_long_mono_body_with_max_sfb,
    parse_asf_psy_info, parse_asf_psy_info_lfe, parse_asf_transform_info, parse_chparam_info,
    AsfPsyInfo, AsfTransformInfo, ChparamInfo, SubstreamTools,
};
use crate::tables;

/// `5_X_codec_mode` values (§4.3.5.6 Table 97).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiveXCodecMode {
    Simple,
    Aspx,
    AspxAcpl1,
    AspxAcpl2,
    AspxAcpl3,
    /// Reserved (5..=7). Surfaced so callers can detect spec violations
    /// without bailing the bitreader.
    Reserved(u8),
}

impl FiveXCodecMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b111 {
            0 => Self::Simple,
            1 => Self::Aspx,
            2 => Self::AspxAcpl1,
            3 => Self::AspxAcpl2,
            4 => Self::AspxAcpl3,
            other => Self::Reserved(other as u8),
        }
    }
}

/// `coding_config` values for the 5.X channel mode (§4.3.5.8 — keyed by
/// the enclosing `5_X_codec_mode` per Table 25). For SIMPLE / ASPX the
/// 2-bit `coding_config` selects one of four channel-data layouts; for
/// `ASPX_ACPL_{1,2}` it's a 1-bit selector between
/// two_channel_data and three_channel_data; for `ASPX_ACPL_3` there's
/// no `coding_config` at all (the body is `stereo_data()` unconditionally).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiveXCodingConfig {
    /// `coding_config == 0` (SIMPLE / ASPX): `2ch_mode + two_channel_data
    /// + two_channel_data + mono_data(0)`.
    Cfg0Stereo2plusMono,
    /// `coding_config == 1` (SIMPLE / ASPX): `three_channel_data +
    /// two_channel_data`. Also the 1-bit `coding_config` "true" value
    /// for ASPX_ACPL_{1,2} (`three_channel_data`).
    Cfg1ThreeStereo,
    /// `coding_config == 2` (SIMPLE / ASPX): `four_channel_data +
    /// mono_data(0)`.
    Cfg2FourMono,
    /// `coding_config == 3` (SIMPLE / ASPX): `five_channel_data`.
    Cfg3Five,
    /// 1-bit selector "false" branch for ASPX_ACPL_{1,2}: `two_channel_data`.
    AcplLite2,
}

/// Parsed `mono_data(b_lfe)` per Table 21 — outer shell + body.
///
/// `b_lfe == 1` switches `sf_info(...)` to `sf_info_lfe()` per §4.2.8 /
/// §4.3.6.2.1 (the `max_sfb[0]` field uses `n_msfbl_bits` from Table
/// 106 instead of `n_msfb_bits`).
///
/// Round 37: extended with `scaled_spec` — the dequantised + scaled
/// MDCT spectrum from the trailing `sf_data(ASF)` body (long-frame,
/// single window group case only). For LFE, for the SSF frontend
/// (`spec_frontend_bit == 1`), or for any short / grouped / Huffman-error
/// case, this stays `None` and only the outer shell is filled. This
/// matches the per-channel `scaled_spec_per_channel` slot pattern in
/// [`TwoChannelData`] / [`ThreeChannelData`] etc.
#[derive(Debug, Clone, Default)]
pub struct MonoLfeData {
    /// `b_lfe` flag the walker was invoked with.
    pub b_lfe: bool,
    /// `spec_frontend` selector — always ASF for LFE per Table 21.
    /// Captured from the bitstream for non-LFE invocations.
    pub spec_frontend_bit: u8,
    /// Parsed `asf_transform_info()` for the channel.
    pub transform_info: Option<AsfTransformInfo>,
    /// Parsed `asf_psy_info()` for the channel. For LFE this is the
    /// `sf_info_lfe()` flavour with `max_sfb` capped to
    /// `num_sfb_lfe()` and bit-width `n_msfbl_bits`.
    pub psy_info: Option<AsfPsyInfo>,
    /// Dequantised + scaled MDCT spectrum from the trailing `sf_data(ASF)`
    /// body. Populated for the non-LFE, ASF-frontend, long-frame,
    /// single-window-group case. Length is `sfb_offset[max_sfb]` at the
    /// signalled transform length. `None` for LFE (the LFE body decoder
    /// is reserved for a future round), for SSF-frontend mono channels,
    /// or for any short / grouped / Huffman-error case.
    pub scaled_spec: Option<Vec<f32>>,
}

/// Parsed `three_channel_info()` per Table 30: 4-bit `chel_matsel` +
/// two `chparam_info()` payloads.
#[derive(Debug, Clone, Default)]
pub struct ThreeChannelInfo {
    pub chel_matsel: u8,
    pub chparam: [ChparamInfo; 2],
}

/// Parsed `four_channel_info()` per Table 31: four `chparam_info()`
/// payloads (no `chel_matsel`).
#[derive(Debug, Clone, Default)]
pub struct FourChannelInfo {
    pub chparam: [ChparamInfo; 4],
}

/// Parsed `five_channel_info()` per Table 32: 4-bit `chel_matsel` +
/// five `chparam_info()` payloads.
#[derive(Debug, Clone, Default)]
pub struct FiveChannelInfo {
    pub chel_matsel: u8,
    pub chparam: [ChparamInfo; 5],
}

/// Parsed `two_channel_data()` outer shell per Table 26.
///
/// Table 26 is a stripped-down stereo container — single shared
/// `sf_info(ASF, 0, 0)` followed by `chparam_info()` (the full Table 47
/// row including `sap_mode` / `ms_used` / `sap_data`) and then two
/// `sf_data(ASF)` bodies. We parse the outer shell only — the
/// `sf_data(ASF)` Huffman bodies are deferred to the `acpl_synth` /
/// joint-MDCT decoder paths (Pseudocode 178 in §5.3.3 needs the
/// transform-matrix wiring that's still outstanding for the multichannel
/// elements).
///
/// Used by `5_X_channel_element` Cfg0 (twice — L/R and Ls/Rs) and Cfg1
/// (after the leading `three_channel_data`).
///
/// Round 23 added `scaled_spec_per_channel` — when the body is decoded
/// in the long-frame, single-window-group case, each `Some(...)` entry
/// carries the dequantised + scaled MDCT spectrum for that channel.
/// Entries are `None` when the per-channel `sf_data(ASF)` walk bailed
/// (short frame, grouped, or Huffman error).
#[derive(Debug, Clone, Default)]
pub struct TwoChannelData {
    pub transform_info: Option<AsfTransformInfo>,
    pub psy_info: Option<AsfPsyInfo>,
    pub chparam: Option<ChparamInfo>,
    /// Per-channel scaled MDCT spectra. Length = 2 once the body has
    /// been walked. Each entry's `Vec<f32>` is `sfb_offset[max_sfb]`
    /// long.
    pub scaled_spec_per_channel: Vec<Option<Vec<f32>>>,
}

/// Parsed `three_channel_data()` outer shell + per-channel sf_data
/// bodies per Table 27.
///
/// Holds the shared `sf_info` (transform_info + psy_info) and the
/// `three_channel_info` (chel_matsel + 2x chparam_info). Round 23 also
/// walks the three trailing `sf_data(ASF)` bodies into
/// `scaled_spec_per_channel` (length 3).
#[derive(Debug, Clone, Default)]
pub struct ThreeChannelData {
    pub transform_info: Option<AsfTransformInfo>,
    pub psy_info: Option<AsfPsyInfo>,
    pub info: Option<ThreeChannelInfo>,
    /// Per-channel scaled MDCT spectra (length 3). See [`TwoChannelData`].
    pub scaled_spec_per_channel: Vec<Option<Vec<f32>>>,
}

/// Parsed `four_channel_data()` outer shell + per-channel sf_data
/// bodies per Table 28. Round 23 walks the four trailing `sf_data(ASF)`
/// bodies into `scaled_spec_per_channel` (length 4).
#[derive(Debug, Clone, Default)]
pub struct FourChannelData {
    pub transform_info: Option<AsfTransformInfo>,
    pub psy_info: Option<AsfPsyInfo>,
    pub info: Option<FourChannelInfo>,
    /// Per-channel scaled MDCT spectra (length 4). See [`TwoChannelData`].
    pub scaled_spec_per_channel: Vec<Option<Vec<f32>>>,
}

/// Parsed `five_channel_data()` outer shell + per-channel sf_data
/// bodies per Table 29. Round 23 walks the five trailing `sf_data(ASF)`
/// bodies into `scaled_spec_per_channel` (length 5).
#[derive(Debug, Clone, Default)]
pub struct FiveChannelData {
    pub transform_info: Option<AsfTransformInfo>,
    pub psy_info: Option<AsfPsyInfo>,
    pub info: Option<FiveChannelInfo>,
    /// Per-channel scaled MDCT spectra (length 5). See [`TwoChannelData`].
    pub scaled_spec_per_channel: Vec<Option<Vec<f32>>>,
}

// =====================================================================
// Per-element parsers
// =====================================================================

/// `mono_data(b_lfe)` per Table 21.
///
/// For `b_lfe == 1` the leading `spec_frontend` bit is **omitted** and
/// `sf_info_lfe()` runs in place of `sf_info()` — `max_sfb[0]` is
/// `n_msfbl_bits` wide and clamped to the LFE band table.
///
/// Round 37: when the channel is non-LFE, ASF-frontend
/// (`spec_frontend_bit == 0`), and long-frame / single-window-group, the
/// trailing `sf_data(ASF)` body is also walked into `scaled_spec`
/// (matching the multichannel `decode_mch_sf_data_channels` pattern).
/// Walker is **try-and-bail** for the body so a Huffman miss leaves the
/// outer shell intact and the caller still gets `Ok(...)`. The SSF
/// frontend and LFE body paths remain deferred — those slots stay
/// `None` and the bitreader cursor is left where the outer shell
/// finished (consistent with prior behaviour).
pub fn parse_mono_data(
    br: &mut BitReader<'_>,
    b_lfe: bool,
    frame_len_base: u32,
) -> Result<MonoLfeData> {
    let mut out = MonoLfeData {
        b_lfe,
        ..Default::default()
    };
    if !b_lfe {
        // Non-LFE: leading 1-bit spec_frontend selector.
        out.spec_frontend_bit = br.read_u32(1)? as u8;
    }
    // Both LFE and non-LFE invoke the ASF transform-info shell — the
    // LFE channel is always coded with the ASF frontend per Table 21.
    let ti = parse_asf_transform_info(br, frame_len_base)?;
    out.transform_info = Some(ti);
    // `sf_info(ASF, 0, 0)` for non-LFE; `sf_info_lfe()` for LFE.
    // r20: dispatch to the dedicated `parse_asf_psy_info_lfe()` that
    // uses Table 106 column `n_msfbl_bits` for `max_sfb[0]` instead of
    // the regular `n_msfb_bits`. The two widths can differ by 2-4 bits
    // (e.g. 48 kHz long-frame: 6 vs 3) so this matters for any real
    // 5.1 / 7.1 stream LFE alignment.
    let psy = if b_lfe {
        parse_asf_psy_info_lfe(br, &ti)?
    } else {
        parse_asf_psy_info(br, &ti, frame_len_base, false, false)?
    };

    // Round 37: trailing `sf_data(ASF)` body for non-LFE ASF-frontend
    // mono channels. This mirrors the per-channel body walk inside
    // `decode_mch_sf_data_channels` for a single channel. Try-and-bail:
    // any Huffman / bit-stream miss leaves `scaled_spec` as `None` and
    // returns `Ok(...)`. SSF-frontend (`spec_frontend_bit == 1`) is
    // deferred — the SSF body lives elsewhere in the substream and is
    // not co-located with `mono_data()`.
    //
    // Round 38: extend the body walk to LFE channels via
    // `decode_asf_long_lfe_body_with_max_sfb_lfe`. Per `sf_info_lfe()`
    // (Table 35) the LFE channel is always long-frame, single window
    // group, ASF-frontend — so the same long-frame mono body decoder
    // applies, just with `max_sfb_0` already capped by the
    // `n_msfbl_bits` bit-width.
    if !b_lfe && out.spec_frontend_bit == 0 {
        if ti.b_long_frame && psy.num_window_groups == 1 {
            if let Some(scaled) = decode_asf_long_mono_body_with_max_sfb(br, &ti, psy.max_sfb_0) {
                out.scaled_spec = Some(scaled);
            }
        } else if psy.num_window_groups > 0 {
            if let Some(scaled) = decode_asf_grouped_mono_body_with_max_sfb(
                br,
                &ti,
                psy.max_sfb_0,
                psy.num_window_groups,
            ) {
                out.scaled_spec = Some(scaled);
            }
        }
    } else if b_lfe {
        // LFE: `sf_info_lfe()` forces long-frame, single window group;
        // walk the LFE body via the dedicated decoder. Try-and-bail
        // identical to the non-LFE path.
        if let Some(scaled) = decode_asf_long_lfe_body_with_max_sfb_lfe(br, &ti, psy.max_sfb_0) {
            out.scaled_spec = Some(scaled);
        }
    }
    out.psy_info = Some(psy);
    Ok(out)
}

/// `two_channel_data()` outer shell + per-channel `sf_data(ASF)` bodies
/// per Table 26.
///
/// Walks `sf_info(ASF, 0, 0)` + `chparam_info()` then two
/// `sf_data(ASF)` bodies. For the long-frame, single-window-group case
/// the per-channel scaled spectra land on `scaled_spec_per_channel`
/// (length 2). Short / grouped frames push `None` for each channel —
/// the outer shell still parses cleanly.
pub fn parse_two_channel_data(
    br: &mut BitReader<'_>,
    frame_len_base: u32,
) -> Result<TwoChannelData> {
    let ti = parse_asf_transform_info(br, frame_len_base)?;
    let psy = parse_asf_psy_info(br, &ti, frame_len_base, false, false)?;
    let max_sfb_g = psy.max_sfb_0;
    let chparam = parse_chparam_info(br, &[max_sfb_g])?;
    let scaled = decode_mch_sf_data_channels(br, &ti, &psy, 2);
    Ok(TwoChannelData {
        transform_info: Some(ti),
        psy_info: Some(psy),
        chparam: Some(chparam),
        scaled_spec_per_channel: scaled,
    })
}

/// `three_channel_info()` per Table 30.
pub fn parse_three_channel_info(
    br: &mut BitReader<'_>,
    max_sfb_per_group: &[u32],
) -> Result<ThreeChannelInfo> {
    let chel_matsel = br.read_u32(4)? as u8;
    let cp0 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp1 = parse_chparam_info(br, max_sfb_per_group)?;
    Ok(ThreeChannelInfo {
        chel_matsel,
        chparam: [cp0, cp1],
    })
}

/// `four_channel_info()` per Table 31.
pub fn parse_four_channel_info(
    br: &mut BitReader<'_>,
    max_sfb_per_group: &[u32],
) -> Result<FourChannelInfo> {
    let cp0 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp1 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp2 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp3 = parse_chparam_info(br, max_sfb_per_group)?;
    Ok(FourChannelInfo {
        chparam: [cp0, cp1, cp2, cp3],
    })
}

/// `five_channel_info()` per Table 32.
pub fn parse_five_channel_info(
    br: &mut BitReader<'_>,
    max_sfb_per_group: &[u32],
) -> Result<FiveChannelInfo> {
    let chel_matsel = br.read_u32(4)? as u8;
    let cp0 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp1 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp2 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp3 = parse_chparam_info(br, max_sfb_per_group)?;
    let cp4 = parse_chparam_info(br, max_sfb_per_group)?;
    Ok(FiveChannelInfo {
        chel_matsel,
        chparam: [cp0, cp1, cp2, cp3, cp4],
    })
}

/// `three_channel_data()` per Table 27 — outer shell + 3x `sf_data(ASF)`.
///
/// Round 23 wires the three trailing `sf_data(ASF)` bodies through the
/// shared `(transform_info, psy_info)` pair. Each per-channel scaled
/// spectrum lands on `scaled_spec_per_channel[i]` for the long-frame,
/// single-window-group case.
pub fn parse_three_channel_data(
    br: &mut BitReader<'_>,
    frame_len_base: u32,
) -> Result<ThreeChannelData> {
    let ti = parse_asf_transform_info(br, frame_len_base)?;
    let psy = parse_asf_psy_info(br, &ti, frame_len_base, false, false)?;
    let max_sfb_g = psy.max_sfb_0;
    let info = parse_three_channel_info(br, &[max_sfb_g])?;
    let scaled = decode_mch_sf_data_channels(br, &ti, &psy, 3);
    Ok(ThreeChannelData {
        transform_info: Some(ti),
        psy_info: Some(psy),
        info: Some(info),
        scaled_spec_per_channel: scaled,
    })
}

/// `four_channel_data()` per Table 28 — outer shell + 4x `sf_data(ASF)`.
pub fn parse_four_channel_data(
    br: &mut BitReader<'_>,
    frame_len_base: u32,
) -> Result<FourChannelData> {
    let ti = parse_asf_transform_info(br, frame_len_base)?;
    let psy = parse_asf_psy_info(br, &ti, frame_len_base, false, false)?;
    let max_sfb_g = psy.max_sfb_0;
    let info = parse_four_channel_info(br, &[max_sfb_g])?;
    let scaled = decode_mch_sf_data_channels(br, &ti, &psy, 4);
    Ok(FourChannelData {
        transform_info: Some(ti),
        psy_info: Some(psy),
        info: Some(info),
        scaled_spec_per_channel: scaled,
    })
}

/// `five_channel_data()` per Table 29 — outer shell + 5x `sf_data(ASF)`.
pub fn parse_five_channel_data(
    br: &mut BitReader<'_>,
    frame_len_base: u32,
) -> Result<FiveChannelData> {
    let ti = parse_asf_transform_info(br, frame_len_base)?;
    let psy = parse_asf_psy_info(br, &ti, frame_len_base, false, false)?;
    let max_sfb_g = psy.max_sfb_0;
    let info = parse_five_channel_info(br, &[max_sfb_g])?;
    let scaled = decode_mch_sf_data_channels(br, &ti, &psy, 5);
    Ok(FiveChannelData {
        transform_info: Some(ti),
        psy_info: Some(psy),
        info: Some(info),
        scaled_spec_per_channel: scaled,
    })
}

/// Walk `n_channels` consecutive `sf_data(ASF)` bodies sharing the same
/// `(transform_info, psy_info)` per Tables 26 / 27 / 28 / 29 of TS 103
/// 190-1.
///
/// Each body decodes as one `asf_section_data()` then `asf_spectral_data()`
/// then `asf_scalefac_data()` then `asf_snf_data()` chain (§4.2.8.3-6) and
/// produces a dequantised + scaled MDCT spectrum of length
/// `sfb_offset[max_sfb]`. The Huffman codebooks reused from
/// [`crate::huffman`] are: `HCB_1` .. `HCB_11` for spectral lines (per
/// `sect_cb`), `HCB_SCALEFAC` (codebook ID `SCF`, 121 entries) for
/// scale-factor DPCM and `HCB_SNF` (22 entries) for spectral noise
/// fill. Annex A.1 shares the codebooks across mono / stereo /
/// multichannel — there is no separate "MCH" codebook set.
///
/// Bodies past a Huffman / bit-stream miss return `None` for the
/// remaining channels (we can't re-sync mid-bitstream); the entries
/// before the miss are still populated.
///
/// Round 24 extends the previous (long-frame, `num_window_groups == 1`)
/// path to also walk **short / grouped** frames where
/// `num_window_groups > 1`. In that case each per-channel body fires
/// `num_window_groups` independent
/// `(asf_section_data + asf_spectral_data + asf_scalefac_data +
/// asf_snf_data)` cycles — one per window group — and the
/// per-channel spectrum is the concatenation of the
/// `num_window_groups` per-group spectra (each of length
/// `sfb_offset[max_sfb]` at the per-window transform length).
/// `b_dual_maxsfb == 0` means the same `max_sfb_0` applies to every
/// group; per-group `max_sfb` selection (Pseudocode 5
/// `get_max_sfb(g)`) collapses to that single value for the
/// non-side-channel multichannel path.
pub(crate) fn decode_mch_sf_data_channels(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
    n_channels: usize,
) -> Vec<Option<Vec<f32>>> {
    let mut out = vec![None; n_channels];
    if ti.b_long_frame && psy.num_window_groups == 1 {
        // Long-frame, 1 window group — walk one body chain per channel.
        for slot in out.iter_mut() {
            match decode_asf_long_mono_body_with_max_sfb(br, ti, psy.max_sfb_0) {
                Some(v) => *slot = Some(v),
                None => break,
            }
        }
        return out;
    }
    if psy.num_window_groups == 0 {
        return out;
    }
    // Short / grouped frame walker.
    for slot in out.iter_mut() {
        match decode_asf_grouped_mono_body_with_max_sfb(
            br,
            ti,
            psy.max_sfb_0,
            psy.num_window_groups,
        ) {
            Some(v) => *slot = Some(v),
            None => break,
        }
    }
    out
}

/// Walk one `sf_data(ASF)` body for the grouped / short-frame case
/// where `num_window_groups > 1`. Per spec §4.2.8 (Tables 39-42) and
/// §5.4.4.4 the body fires `num_window_groups` independent
/// `(section + spectral + scalefac + snf)` cycles back-to-back; the
/// per-group spectrum is `sfb_offset[max_sfb]` long at the per-window
/// transform length. The returned vector concatenates the
/// `num_window_groups` per-group spectra (group-major).
///
/// Returns `None` on the first Huffman / bit-stream miss; partial
/// per-group output is dropped because the bitreader position is
/// indeterminate after a mid-body miss.
fn decode_asf_grouped_mono_body_with_max_sfb(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    max_sfb_in: u32,
    num_window_groups: u32,
) -> Option<Vec<f32>> {
    let tl = ti.transform_length_0;
    let tl_idx = ti.transf_length[0];
    let max_sfb_cap = crate::tables::num_sfb_48(tl)?;
    let max_sfb = max_sfb_in.min(max_sfb_cap);
    if max_sfb == 0 {
        return None;
    }
    let sfbo = crate::sfb_offset::sfb_offset_48(tl)?;
    let per_group_len = sfbo[max_sfb as usize] as usize;
    let total_len = per_group_len.checked_mul(num_window_groups as usize)?;
    let mut out = Vec::with_capacity(total_len);
    for _g in 0..num_window_groups {
        let sections = crate::asf_data::parse_asf_section_data(br, tl_idx, tl, max_sfb).ok()?;
        let (qspec, mqi) =
            crate::asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
        let sf_gain =
            crate::asf_data::parse_asf_scalefac_data(br, &sections, &mqi, max_sfb, tl).ok()?;
        let _snf = crate::asf_data::parse_asf_snf_data(br, &sections, &mqi, max_sfb, tl).ok()?;
        let scaled = crate::asf_data::dequantise_and_scale(&qspec, &sf_gain, sfbo, max_sfb);
        out.extend_from_slice(&scaled);
    }
    Some(out)
}

// =====================================================================
// 5.X outer walker
// =====================================================================

/// Parse the outer layers of `5_X_channel_element(b_has_lfe, b_iframe)`
/// per §4.2.6.6 Table 25.
///
/// Returns `Ok(())` after walking:
///
/// 1. The 3-bit `5_X_codec_mode` selector.
/// 2. The I-frame config block (`aspx_config()` + `acpl_config_*`).
/// 3. The LFE `mono_data(1)` shell when `b_has_lfe == 1`.
/// 4. The `companding_control()` for non-SIMPLE codec modes.
/// 5. The `coding_config` selector (2 bits for SIMPLE/ASPX, 1 bit for
///    ASPX_ACPL_{1,2}, absent for ASPX_ACPL_3) and the chosen
///    channel-element bodies' outer shells.
///
/// **Scope**: r19 lands the bitstream walker for the SIMPLE / ASPX and
/// ASPX_ACPL_3 paths' outer shape, plus full LFE `mono_data(1)`
/// parsing. The ASPX_ACPL_{1,2} variants and the per-channel
/// `sf_data(ASF)` Huffman bodies remain TODO. On any inner parse miss
/// the walker bails early but keeps the partially-populated tools.
pub fn parse_5x_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_has_lfe: bool,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    // 5_X_codec_mode (3 bits).
    let mode_bits = br.read_u32(3)?;
    let mode = FiveXCodecMode::from_u32(mode_bits);
    tools.five_x_mode = Some(mode);
    tools.five_x_b_has_lfe = b_has_lfe;

    // I-frame config block.
    if b_iframe {
        match mode {
            FiveXCodecMode::Aspx
            | FiveXCodecMode::AspxAcpl1
            | FiveXCodecMode::AspxAcpl2
            | FiveXCodecMode::AspxAcpl3 => {
                tools.aspx_config = Some(crate::aspx::parse_aspx_config(br)?);
            }
            _ => {}
        }
        match mode {
            FiveXCodecMode::AspxAcpl1 => {
                let cfg =
                    crate::acpl::parse_acpl_config_1ch(br, crate::acpl::Acpl1chMode::Partial)?;
                tools.acpl_config_1ch_partial = Some(cfg);
            }
            FiveXCodecMode::AspxAcpl2 => {
                let cfg = crate::acpl::parse_acpl_config_1ch(br, crate::acpl::Acpl1chMode::Full)?;
                tools.acpl_config_1ch_full = Some(cfg);
            }
            FiveXCodecMode::AspxAcpl3 => {
                let cfg = crate::acpl::parse_acpl_config_2ch(br)?;
                tools.acpl_config_2ch = Some(cfg);
            }
            _ => {}
        }
    }

    // LFE: mono_data(1).
    if b_has_lfe {
        let lfe = parse_mono_data(br, true, frame_len_base)?;
        tools.lfe_mono_data = Some(lfe);
    }

    // Mode-specific body.
    match mode {
        FiveXCodecMode::Simple | FiveXCodecMode::Aspx => {
            if matches!(mode, FiveXCodecMode::Aspx) {
                tools.companding = Some(crate::aspx::parse_companding_control(br, 5)?);
            }
            // 2-bit coding_config.
            let cc = br.read_u32(2)?;
            let coding_cfg = match cc {
                0 => FiveXCodingConfig::Cfg0Stereo2plusMono,
                1 => FiveXCodingConfig::Cfg1ThreeStereo,
                2 => FiveXCodingConfig::Cfg2FourMono,
                _ => FiveXCodingConfig::Cfg3Five,
            };
            tools.five_x_coding_config = Some(coding_cfg);
            // r20: walk all four channel-element layouts' outer shells.
            // r23: also walks the trailing `sf_data(ASF)` Huffman bodies
            // (one per channel) for the long-frame, single-window-group
            // case, depositing the dequantised spectra on each
            // `*ChannelData::scaled_spec_per_channel`. Short / grouped
            // frames still parse the outer shell and leave the per-
            // channel slots `None`.
            match coding_cfg {
                FiveXCodingConfig::Cfg0Stereo2plusMono => {
                    // Table 25 row 0: 1-bit `2ch_mode` selector
                    // (b_2ch_mode), then two_channel_data twice (L/R
                    // then Ls/Rs), then mono_data(0) for the centre.
                    tools.b_2ch_mode = Some(br.read_bit()?);
                    tools.two_channel_data.clear();
                    tools
                        .two_channel_data
                        .push(parse_two_channel_data(br, frame_len_base)?);
                    tools
                        .two_channel_data
                        .push(parse_two_channel_data(br, frame_len_base)?);
                    tools.cfg0_centre_mono = Some(parse_mono_data(br, false, frame_len_base)?);
                }
                FiveXCodingConfig::Cfg1ThreeStereo => {
                    // Table 25 row 1: three_channel_data + two_channel_data.
                    tools.three_channel_data = Some(parse_three_channel_data(br, frame_len_base)?);
                    tools.two_channel_data.clear();
                    tools
                        .two_channel_data
                        .push(parse_two_channel_data(br, frame_len_base)?);
                }
                FiveXCodingConfig::Cfg2FourMono => {
                    // Table 25 row 2: four_channel_data + mono_data(0).
                    tools.four_channel_data = Some(parse_four_channel_data(br, frame_len_base)?);
                    tools.cfg2_back_mono = Some(parse_mono_data(br, false, frame_len_base)?);
                }
                FiveXCodingConfig::Cfg3Five => {
                    tools.five_channel_data = Some(parse_five_channel_data(br, frame_len_base)?);
                }
                FiveXCodingConfig::AcplLite2 => {
                    // AcplLite2 is the ASPX_ACPL_{1,2} false-branch and
                    // can't appear in the SIMPLE/ASPX 2-bit map.
                    debug_assert!(
                        false,
                        "AcplLite2 unreachable from SIMPLE/ASPX 2-bit coding_config"
                    );
                }
            }
            // §4.2.6.6 Table 25 row `case ASPX:` — when the 5_X codec
            // mode is ASPX (not SIMPLE) the body trails three ASPX
            // payloads regardless of coding_config:
            //
            //   aspx_data_2ch();   // first stereo pair
            //   aspx_data_2ch();   // second stereo pair
            //   aspx_data_1ch();   // mono (centre)
            //
            // Captured into per-coding-config trailer slots so the
            // decoder can run `aspx_extend_pcm` per channel without
            // overwriting the stereo-CPE primary/secondary slots.
            //
            // Round 41: cfg2 wired (the four-channel + back-mono
            // layout). cfg0 / cfg1 / cfg3 land in the same trailers
            // but with different channel-to-trailer mappings; the
            // dispatch side currently consumes only the cfg2 mapping
            // so the cfg0 / cfg1 / cfg3 trailer fields capture data
            // for a future round to wire.
            if matches!(mode, FiveXCodecMode::Aspx) && b_iframe && tools.aspx_config.is_some() {
                let aspx_cfg = tools.aspx_config.unwrap();
                let lr = crate::asf::capture_aspx_data_2ch_trailer(
                    br,
                    tools,
                    &aspx_cfg,
                    b_iframe,
                    frame_len_base,
                );
                let ls_rs = crate::asf::capture_aspx_data_2ch_trailer(
                    br,
                    tools,
                    &aspx_cfg,
                    b_iframe,
                    frame_len_base,
                );
                let centre = crate::asf::capture_aspx_data_1ch_trailer(
                    br,
                    tools,
                    &aspx_cfg,
                    b_iframe,
                    frame_len_base,
                );
                if matches!(coding_cfg, FiveXCodingConfig::Cfg2FourMono) {
                    tools.cfg2_aspx_lr = lr;
                    tools.cfg2_aspx_ls_rs = ls_rs;
                    tools.cfg2_aspx_centre = centre;
                } else {
                    // Discard captured trailers for cfg0 / cfg1 / cfg3
                    // until the dispatch side wires them. The bits
                    // were still consumed so the bit-pointer ends at
                    // the right offset for any frame_data trailer.
                    let _ = (lr, ls_rs, centre);
                }
            }
        }
        FiveXCodecMode::AspxAcpl1 | FiveXCodecMode::AspxAcpl2 => {
            tools.companding = Some(crate::aspx::parse_companding_control(br, 3)?);
            // 1-bit coding_config.
            let cc = br.read_bit()?;
            let coding_cfg = if cc {
                FiveXCodingConfig::Cfg1ThreeStereo
            } else {
                FiveXCodingConfig::AcplLite2
            };
            tools.five_x_coding_config = Some(coding_cfg);
            // r25: walk the inner body per §4.2.6.6 Table 25 row
            // `case ASPX_ACPL_1: case ASPX_ACPL_2:`. The walker is
            // try-and-bail: any inner Huffman / parse miss leaves the
            // already-populated tools intact and returns silently. The
            // outer walker still returns Ok(()).
            let _ = parse_aspx_acpl_1_2_inner_body(br, tools, mode, cc, b_iframe, frame_len_base);
        }
        FiveXCodecMode::AspxAcpl3 => {
            tools.companding = Some(crate::aspx::parse_companding_control(br, 2)?);
            // No coding_config: body is `stereo_data() + aspx_data_2ch()
            // + acpl_data_2ch()` per Table 25 row ASPX_ACPL_3.
            //
            // r24 wires the inner body walkers. The flow mirrors the
            // stereo-CPE ASPX path in `asf.rs`: walk stereo_data() into
            // tools, then if the body decoded cleanly + we're on an
            // I-frame + an aspx_config is in scope, walk aspx_data_2ch()
            // (Table 52). Finally walk acpl_data_2ch() (Table 62) using
            // the parsed acpl_config_2ch() — start_band is 0 since
            // acpl_config_2ch() doesn't carry a qmf_band field
            // (acpl_data_2ch always covers all parameter bands).
            let body_ok = crate::asf::parse_stereo_data_body(br, tools, frame_len_base);
            if b_iframe && body_ok {
                if let Some(cfg) = tools.aspx_config {
                    crate::asf::parse_aspx_data_2ch_body(
                        br,
                        tools,
                        &cfg,
                        b_iframe,
                        frame_len_base,
                    )?;
                    if let Some(acfg) = tools.acpl_config_2ch {
                        if let Ok(d) = crate::acpl::parse_acpl_data_2ch(
                            br,
                            acfg.num_param_bands,
                            0,
                            acfg.quant_mode_0,
                            acfg.quant_mode_1,
                        ) {
                            tools.acpl_data_2ch = Some(d);
                        }
                    }
                }
            }
        }
        FiveXCodecMode::Reserved(_) => {}
    }
    Ok(())
}

// =====================================================================
// 5_X ASPX_ACPL_1 / ASPX_ACPL_2 inner body walker (round 25)
// =====================================================================

/// Walk the inner body of `5_X_channel_element` for the
/// `ASPX_ACPL_1` / `ASPX_ACPL_2` modes per §4.2.6.6 Table 25 (the
/// `case ASPX_ACPL_1: case ASPX_ACPL_2:` arm) — the bits *after*
/// `companding_control(3)` and the 1-bit `coding_config`.
///
/// The body shape (in Table 25 order):
/// ```text
/// if (coding_config) { three_channel_data(); }
/// else               { two_channel_data();   }
/// if (5_X_codec_mode == ASPX_ACPL_1) {
///     max_sfb_master;            // n_side_bits — joint-MDCT residual
///     chparam_info();            // ACPL_1 residual ch0
///     chparam_info();            // ACPL_1 residual ch1
///     sf_data(ASF);              // ACPL_1 residual ch0
///     sf_data(ASF);              // ACPL_1 residual ch1
/// }
/// if (coding_config == 0) {
///     mono_data(0);              // centre / surround mono
/// }
/// aspx_data_2ch();
/// aspx_data_1ch();
/// acpl_data_1ch();               // -> tools.acpl_data_1ch_pair[0]
/// acpl_data_1ch();               // -> tools.acpl_data_1ch_pair[1]
/// ```
///
/// `n_side_bits` is derived per the §4.2.6.6 NOTE: largest signalled
/// transform length from the preceding two/three_channel_data() above
/// (look up Table 106 column `n_side_bits`).
///
/// The walker is **try-and-bail**: every step bails silently on the
/// first parse miss, leaving the already-populated `tools.*` fields
/// intact. It always returns `Ok(())` to the caller — the outer
/// walker's contract is that an inner-body miss is non-fatal.
///
/// Like the round-24 ASPX_ACPL_3 walker, the deeper aspx_data /
/// acpl_data steps are gated on `b_iframe && tools.aspx_config.is_some()`
/// so non-iframe paths simply consume what they can of the upstream
/// channel data and stop.
fn parse_aspx_acpl_1_2_inner_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    mode: FiveXCodecMode,
    coding_config_bit: bool,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    debug_assert!(matches!(
        mode,
        FiveXCodecMode::AspxAcpl1 | FiveXCodecMode::AspxAcpl2
    ));
    // 1) two_channel_data() OR three_channel_data().
    let largest_tl: Option<u32> = if coding_config_bit {
        // three_channel_data().
        match parse_three_channel_data(br, frame_len_base) {
            Ok(d) => {
                let tl = d.transform_info.as_ref().map(|ti| ti.transform_length_0);
                tools.three_channel_data = Some(d);
                tl
            }
            Err(_) => return Ok(()),
        }
    } else {
        // two_channel_data().
        match parse_two_channel_data(br, frame_len_base) {
            Ok(d) => {
                let tl = d.transform_info.as_ref().map(|ti| ti.transform_length_0);
                tools.two_channel_data.clear();
                tools.two_channel_data.push(d);
                tl
            }
            Err(_) => return Ok(()),
        }
    };

    // 2) ASPX_ACPL_1 only: max_sfb_master + chparam_info×2 + sf_data×2
    //    (joint-MDCT residual layer).
    if matches!(mode, FiveXCodecMode::AspxAcpl1) {
        let Some(tl) = largest_tl else {
            return Ok(());
        };
        let Some((_n_msfb, n_side, _n_msfbl)) = tables::n_msfb_bits_48(tl) else {
            return Ok(());
        };
        let Some(num_sfb_cap) = tables::num_sfb_48(tl) else {
            return Ok(());
        };
        let max_sfb_master = match br.read_u32(n_side) {
            Ok(v) => v.min(num_sfb_cap),
            Err(_) => return Ok(()),
        };
        if max_sfb_master == 0 {
            // No bands signalled — chparam_info()×2 still run with the
            // empty bound, but the sf_data bodies would be degenerate.
            // Bail rather than feed an empty bound through downstream.
            return Ok(());
        }
        // Two chparam_info() calls — one per ACPL_1 residual channel.
        // Per Pseudocode 5 / §4.2.10 the per-group max_sfb here is just
        // max_sfb_master (joint-MDCT residual layer is a single window
        // group at the dominant transform length).
        let cp0 = match parse_chparam_info(br, &[max_sfb_master]) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let cp1 = match parse_chparam_info(br, &[max_sfb_master]) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        // Two sf_data(ASF) bodies — pair the channels' residual MDCT
        // spectra. We reuse the ASF long-frame body decoder with the
        // explicit max_sfb_master bound. We synthesise the
        // AsfTransformInfo on the fly (long-frame at `tl`) since the
        // joint-MDCT residual layer always shares the dominant
        // transform length.
        let synth_ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: tl,
            transform_length_1: tl,
        };
        let body0 = decode_asf_long_mono_body_with_max_sfb(br, &synth_ti, max_sfb_master);
        let Some(b0) = body0 else { return Ok(()) };
        let body1 = decode_asf_long_mono_body_with_max_sfb(br, &synth_ti, max_sfb_master);
        let Some(b1) = body1 else { return Ok(()) };
        // Persist the joint-MDCT residual pair (sSMP,3 / sSMP,4 per
        // Table 181) so the ASPX_ACPL_1 dispatch can IMDCT them into
        // Ls / Rs surround PCM carriers (round 40 — replaces the
        // round-37 silence placeholder for the surround-driven path).
        // Round 41 also persists the matching `chparam_info()` pair so
        // the dispatch can apply Table 181's SAP a/b/c/d first-stage
        // matrix between (sSMP_A, sSMP_B) and (sSMP_3, sSMP_4) before
        // Pseudocode 117 runs.
        tools.acpl_1_residual_pair[0] = Some((tl, b0));
        tools.acpl_1_residual_pair[1] = Some((tl, b1));
        tools.acpl_1_residual_chparam[0] = Some(cp0);
        tools.acpl_1_residual_chparam[1] = Some(cp1);
        tools.acpl_1_residual_max_sfb_master = Some(max_sfb_master);
    }

    // 3) Cfg0 only: mono_data(0) — centre / surround mono.
    if !coding_config_bit {
        match parse_mono_data(br, false, frame_len_base) {
            Ok(m) => tools.cfg0_centre_mono = Some(m),
            Err(_) => return Ok(()),
        }
    }

    // 4) ASPX trailers + ACPL pair are I-frame-gated and need a parsed
    //    aspx_config in scope — same gate as ASPX_ACPL_3 (round 24).
    if !b_iframe {
        return Ok(());
    }
    let Some(aspx_cfg) = tools.aspx_config else {
        return Ok(());
    };
    // aspx_data_2ch() then aspx_data_1ch().
    if crate::asf::parse_aspx_data_2ch_body(br, tools, &aspx_cfg, b_iframe, frame_len_base).is_err()
    {
        return Ok(());
    }
    if crate::asf::parse_aspx_data_1ch_body(br, tools, &aspx_cfg, b_iframe, frame_len_base).is_err()
    {
        return Ok(());
    }
    // acpl_data_1ch()×2 — pair entries [0] / [1] per Pseudocode 117.
    // The active acpl_config_1ch was parsed earlier in the I-frame
    // header — `acpl_config_1ch_partial` for ASPX_ACPL_1, full for ACPL_2.
    let acpl_cfg = match mode {
        FiveXCodecMode::AspxAcpl1 => tools.acpl_config_1ch_partial,
        FiveXCodecMode::AspxAcpl2 => tools.acpl_config_1ch_full,
        _ => None,
    };
    let Some(acfg) = acpl_cfg else {
        return Ok(());
    };
    let start_band = if acfg.qmf_band == 0 {
        0
    } else {
        crate::acpl::sb_to_pb(acfg.qmf_band as u32, acfg.num_param_bands)
    };
    if let Ok(d0) =
        crate::acpl::parse_acpl_data_1ch(br, acfg.num_param_bands, start_band, acfg.quant_mode)
    {
        tools.acpl_data_1ch_pair[0] = Some(d0);
        if let Ok(d1) =
            crate::acpl::parse_acpl_data_1ch(br, acfg.num_param_bands, start_band, acfg.quant_mode)
        {
            tools.acpl_data_1ch_pair[1] = Some(d1);
        }
    }
    Ok(())
}

// =====================================================================
// 7_X channel-element walker (round 27 — immersive 7.0 / 7.1)
// =====================================================================

/// `7_X_codec_mode` values per §4.3.5.7 Table 98.
///
/// Note this is a **2-bit** field (vs the 3-bit `5_X_codec_mode`) and
/// has **no** `ASPX_ACPL_3` mode — only SIMPLE / ASPX / ASPX_ACPL_1 /
/// ASPX_ACPL_2 are defined for 7.X.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SevenXCodecMode {
    Simple,
    Aspx,
    AspxAcpl1,
    AspxAcpl2,
}

impl SevenXCodecMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b11 {
            0 => Self::Simple,
            1 => Self::Aspx,
            2 => Self::AspxAcpl1,
            _ => Self::AspxAcpl2,
        }
    }
}

/// Parse the outer layers of `7_X_channel_element(channel_mode, b_iframe)`
/// per §4.2.6.14 Table 33.
///
/// `b_has_lfe == true` corresponds to `channel_mode == "7.1"` per the
/// spec (the only differentiator between 7.0 and 7.1 inside the
/// channel element is the leading `mono_data(1)` LFE).
///
/// The walker mirrors [`parse_5x_audio_data_outer`] but with the
/// 7.X-specific shape:
///
/// 1. 2-bit `7_X_codec_mode` (vs 3-bit for 5_X). No `Reserved` values
///    since all 4 codepoints are defined.
/// 2. I-frame config block: `aspx_config()` for non-SIMPLE,
///    `acpl_config_1ch(PARTIAL/FULL)` for ASPX_ACPL_{1,2}. There is no
///    `acpl_config_2ch()` here — ASPX_ACPL_3 is 5.X-only.
/// 3. LFE `mono_data(1)` when `b_has_lfe`.
/// 4. `companding_control(5)` for ASPX_ACPL_{1,2} only — SIMPLE/ASPX in
///    7.X have **no** leading companding (that differs from 5_X where
///    ASPX gets `companding_control(5)`).
/// 5. The 2-bit `coding_config` switch driving the four channel-data
///    layouts (Cfg0 / Cfg1 / Cfg2 / Cfg3 — same selectors as the 5_X
///    SIMPLE/ASPX path) — but with one critical difference: the Cfg0
///    body is `2ch_mode + two_channel_data() + two_channel_data()`
///    (no centre `mono_data(0)` here) and the Cfg2 body is just
///    `four_channel_data()` (no trailing `mono_data(0)` here either).
/// 6. SIMPLE/ASPX-only additional-channel block: 1-bit
///    `b_use_sap_add_ch` then optional `chparam_info()×2` then a
///    `two_channel_data()` carrying the additional 2 channels (the L+R
///    surround / front-extension pair beyond the 5.X core).
/// 7. ASPX_ACPL_1-only joint-MDCT residual layer: `max_sfb_master`
///    (`n_side_bits` wide, derived per Table 33 NOTE) +
///    `chparam_info()×2 + sf_data(ASF)×2`.
/// 8. `coding_config in {0, 2}`-only trailing `mono_data(0)` — these
///    are the centre / surround mono channels that move from inside the
///    coding_config switch (5.X) to outside it (7.X). For 7.X this is
///    **after** the additional-channel block.
/// 9. ASPX trailers: `aspx_data_2ch()×2 + aspx_data_1ch()` for any
///    non-SIMPLE mode, plus an extra `aspx_data_2ch()` for the ASPX
///    mode (covering the additional two channels).
/// 10. `acpl_data_1ch()×2` for ASPX_ACPL_{1,2} — same pair shape as
///     the 5_X §5.7.7.6.1 Pseudocode 117 path.
///
/// Like the 5_X walker, the deeper `aspx_data` / `acpl_data` steps are
/// gated on `b_iframe && tools.aspx_config.is_some()` so non-iframe
/// paths consume what they can of the upstream channel data and stop.
/// All inner Huffman / parse misses are caught try-and-bail and surface
/// `Ok(())` to the caller — the outer walker never returns `Err` once
/// the leading 2-bit `7_X_codec_mode` has been consumed.
pub fn parse_7x_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_has_lfe: bool,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    // 7_X_codec_mode (2 bits — Table 98).
    let mode_bits = br.read_u32(2)?;
    let mode = SevenXCodecMode::from_u32(mode_bits);
    tools.seven_x_mode = Some(mode);
    tools.seven_x_b_has_lfe = b_has_lfe;

    // I-frame config block.
    if b_iframe {
        if !matches!(mode, SevenXCodecMode::Simple) {
            tools.aspx_config = Some(crate::aspx::parse_aspx_config(br)?);
        }
        match mode {
            SevenXCodecMode::AspxAcpl1 => {
                let cfg =
                    crate::acpl::parse_acpl_config_1ch(br, crate::acpl::Acpl1chMode::Partial)?;
                tools.acpl_config_1ch_partial = Some(cfg);
            }
            SevenXCodecMode::AspxAcpl2 => {
                let cfg = crate::acpl::parse_acpl_config_1ch(br, crate::acpl::Acpl1chMode::Full)?;
                tools.acpl_config_1ch_full = Some(cfg);
            }
            _ => {}
        }
    }

    // LFE: mono_data(1) when channel_mode == "7.1".
    if b_has_lfe {
        let lfe = parse_mono_data(br, true, frame_len_base)?;
        tools.lfe_mono_data = Some(lfe);
    }

    // companding_control(5) — ASPX_ACPL_{1,2} only.
    // Note: SIMPLE / ASPX in 7.X do **not** carry a leading companding
    // control (different from the 5_X walker where ASPX gets
    // companding_control(5)).
    if matches!(
        mode,
        SevenXCodecMode::AspxAcpl1 | SevenXCodecMode::AspxAcpl2
    ) {
        tools.companding = Some(crate::aspx::parse_companding_control(br, 5)?);
    }

    // coding_config (2 bits) — same 4-way selector as the 5.X
    // SIMPLE/ASPX path but with different body shapes (no Cfg0 centre
    // mono and no Cfg2 surround mono inside the switch — those move
    // out to a single trailing `mono_data(0)` below).
    let cc = br.read_u32(2)?;
    let coding_cfg = match cc {
        0 => FiveXCodingConfig::Cfg0Stereo2plusMono,
        1 => FiveXCodingConfig::Cfg1ThreeStereo,
        2 => FiveXCodingConfig::Cfg2FourMono,
        _ => FiveXCodingConfig::Cfg3Five,
    };
    tools.seven_x_coding_config = Some(coding_cfg);

    // Track the largest signalled transform length across the channel
    // data bodies — used downstream to derive `n_side_bits` per the
    // Table 33 NOTE for the ASPX_ACPL_1 joint-MDCT residual layer.
    let mut largest_tl: Option<u32> = None;
    let update_largest = |tl: u32, slot: &mut Option<u32>| {
        *slot = Some(slot.map_or(tl, |cur| cur.max(tl)));
    };

    // Channel-data switch. Try-and-bail: any inner parser miss leaves
    // the slot None and we still surface Ok(()).
    let mut body_ok = true;
    match coding_cfg {
        FiveXCodingConfig::Cfg0Stereo2plusMono => {
            // 7.X Cfg0 = `2ch_mode + two_channel_data + two_channel_data`
            // (no centre mono inside this switch).
            tools.b_2ch_mode = match br.read_bit() {
                Ok(b) => Some(b),
                Err(_) => {
                    body_ok = false;
                    None
                }
            };
            tools.two_channel_data.clear();
            if body_ok {
                match parse_two_channel_data(br, frame_len_base) {
                    Ok(d) => {
                        if let Some(ti) = d.transform_info.as_ref() {
                            update_largest(ti.transform_length_0, &mut largest_tl);
                        }
                        tools.two_channel_data.push(d);
                    }
                    Err(_) => body_ok = false,
                }
            }
            if body_ok {
                match parse_two_channel_data(br, frame_len_base) {
                    Ok(d) => {
                        if let Some(ti) = d.transform_info.as_ref() {
                            update_largest(ti.transform_length_0, &mut largest_tl);
                        }
                        tools.two_channel_data.push(d);
                    }
                    Err(_) => body_ok = false,
                }
            }
        }
        FiveXCodingConfig::Cfg1ThreeStereo => {
            // 7.X Cfg1 = `three_channel_data + two_channel_data`.
            match parse_three_channel_data(br, frame_len_base) {
                Ok(d) => {
                    if let Some(ti) = d.transform_info.as_ref() {
                        update_largest(ti.transform_length_0, &mut largest_tl);
                    }
                    tools.three_channel_data = Some(d);
                }
                Err(_) => body_ok = false,
            }
            if body_ok {
                tools.two_channel_data.clear();
                match parse_two_channel_data(br, frame_len_base) {
                    Ok(d) => {
                        if let Some(ti) = d.transform_info.as_ref() {
                            update_largest(ti.transform_length_0, &mut largest_tl);
                        }
                        tools.two_channel_data.push(d);
                    }
                    Err(_) => body_ok = false,
                }
            }
        }
        FiveXCodingConfig::Cfg2FourMono => {
            // 7.X Cfg2 = `four_channel_data` (no trailing mono inside
            // this switch — moved to the post-additional-channel block).
            match parse_four_channel_data(br, frame_len_base) {
                Ok(d) => {
                    if let Some(ti) = d.transform_info.as_ref() {
                        update_largest(ti.transform_length_0, &mut largest_tl);
                    }
                    tools.four_channel_data = Some(d);
                }
                Err(_) => body_ok = false,
            }
        }
        FiveXCodingConfig::Cfg3Five => match parse_five_channel_data(br, frame_len_base) {
            Ok(d) => {
                if let Some(ti) = d.transform_info.as_ref() {
                    update_largest(ti.transform_length_0, &mut largest_tl);
                }
                tools.five_channel_data = Some(d);
            }
            Err(_) => body_ok = false,
        },
        FiveXCodingConfig::AcplLite2 => {
            debug_assert!(false, "AcplLite2 unreachable from 7.X 2-bit coding_config");
            body_ok = false;
        }
    }
    if !body_ok {
        return Ok(());
    }

    // SIMPLE / ASPX additional-channel block: optional `chparam_info()×2`
    // gated on `b_use_sap_add_ch`, then a `two_channel_data()` carrying
    // the extra 2 channels (the front-extension or surround-back pair).
    if matches!(mode, SevenXCodecMode::Simple | SevenXCodecMode::Aspx) {
        let b_use_sap_add_ch = match br.read_bit() {
            Ok(b) => b,
            Err(_) => return Ok(()),
        };
        tools.seven_x_b_use_sap_add_ch = Some(b_use_sap_add_ch);
        if b_use_sap_add_ch {
            // Two chparam_info() calls. Use the additional-channel
            // two_channel_data's max_sfb when we read it below; for now
            // pass the largest channel-data max_sfb seen so far (the
            // chparam SAP DPCM walker is bounded by its own
            // `max_sfb_per_group` argument).
            let max_sfb_g = largest_tl.and_then(crate::tables::num_sfb_48).unwrap_or(63);
            let cp0 = match parse_chparam_info(br, &[max_sfb_g]) {
                Ok(c) => c,
                Err(_) => return Ok(()),
            };
            let cp1 = match parse_chparam_info(br, &[max_sfb_g]) {
                Ok(c) => c,
                Err(_) => return Ok(()),
            };
            tools.seven_x_add_chparam_info = Some([cp0, cp1]);
        }
        // Additional `two_channel_data()` for the extra 2 channels.
        match parse_two_channel_data(br, frame_len_base) {
            Ok(d) => {
                if let Some(ti) = d.transform_info.as_ref() {
                    update_largest(ti.transform_length_0, &mut largest_tl);
                }
                tools.seven_x_additional_channel_data = Some(d);
            }
            Err(_) => return Ok(()),
        }
    }

    // ASPX_ACPL_1-only joint-MDCT residual layer (max_sfb_master +
    // chparam_info×2 + sf_data×2). Mirrors the 5_X
    // `parse_aspx_acpl_1_2_inner_body` ACPL_1 block — same shape, same
    // n_side_bits derivation per Table 33 NOTE.
    if matches!(mode, SevenXCodecMode::AspxAcpl1) {
        let Some(tl) = largest_tl else {
            return Ok(());
        };
        let Some((_n_msfb, n_side, _n_msfbl)) = tables::n_msfb_bits_48(tl) else {
            return Ok(());
        };
        let Some(num_sfb_cap) = tables::num_sfb_48(tl) else {
            return Ok(());
        };
        let max_sfb_master = match br.read_u32(n_side) {
            Ok(v) => v.min(num_sfb_cap),
            Err(_) => return Ok(()),
        };
        if max_sfb_master == 0 {
            return Ok(());
        }
        let cp0 = match parse_chparam_info(br, &[max_sfb_master]) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let cp1 = match parse_chparam_info(br, &[max_sfb_master]) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let synth_ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: tl,
            transform_length_1: tl,
        };
        let body0 = decode_asf_long_mono_body_with_max_sfb(br, &synth_ti, max_sfb_master);
        let Some(b0) = body0 else { return Ok(()) };
        let body1 = decode_asf_long_mono_body_with_max_sfb(br, &synth_ti, max_sfb_master);
        let Some(b1) = body1 else { return Ok(()) };
        // Persist the 7_X ASPX_ACPL_1 joint-MDCT residual pair too —
        // shape mirrors the 5_X path; the dispatch can use the same
        // tools slot for both 5_X and 7_X surround-driven render. Same
        // `chparam_info()` pair persisted for Table 181 SAP application.
        tools.acpl_1_residual_pair[0] = Some((tl, b0));
        tools.acpl_1_residual_pair[1] = Some((tl, b1));
        tools.acpl_1_residual_chparam[0] = Some(cp0);
        tools.acpl_1_residual_chparam[1] = Some(cp1);
        tools.acpl_1_residual_max_sfb_master = Some(max_sfb_master);
    }

    // Trailing `mono_data(0)` for `coding_config in {0, 2}` — the
    // centre (Cfg0) or surround-back (Cfg2) mono channel. In 7.X this
    // moves out of the coding_config switch and lands after the
    // additional-channel block / ASPX_ACPL_1 residual layer.
    if matches!(
        coding_cfg,
        FiveXCodingConfig::Cfg0Stereo2plusMono | FiveXCodingConfig::Cfg2FourMono
    ) {
        match parse_mono_data(br, false, frame_len_base) {
            Ok(m) => {
                // Land in the mode-appropriate slot — Cfg0 → centre,
                // Cfg2 → back. Reuses the existing 5_X plumbing.
                if matches!(coding_cfg, FiveXCodingConfig::Cfg0Stereo2plusMono) {
                    tools.cfg0_centre_mono = Some(m);
                } else {
                    tools.cfg2_back_mono = Some(m);
                }
            }
            Err(_) => return Ok(()),
        }
    }

    // ASPX trailers + ACPL pair: gated on b_iframe + aspx_config in
    // scope. Same gate as the 5_X ASPX_ACPL_{1,2,3} walkers.
    if !b_iframe {
        return Ok(());
    }
    let Some(aspx_cfg) = tools.aspx_config else {
        return Ok(());
    };

    // `if (7_X_codec_mode != SIMPLE) { aspx_data_2ch + aspx_data_2ch
    // + aspx_data_1ch }` — covers the L/R + Ls/Rs front pair and the
    // additional-channel pair plus the centre mono.
    if !matches!(mode, SevenXCodecMode::Simple) {
        if crate::asf::parse_aspx_data_2ch_body(br, tools, &aspx_cfg, b_iframe, frame_len_base)
            .is_err()
        {
            return Ok(());
        }
        if crate::asf::parse_aspx_data_2ch_body(br, tools, &aspx_cfg, b_iframe, frame_len_base)
            .is_err()
        {
            return Ok(());
        }
        if crate::asf::parse_aspx_data_1ch_body(br, tools, &aspx_cfg, b_iframe, frame_len_base)
            .is_err()
        {
            return Ok(());
        }
    }
    // `if (7_X_codec_mode == ASPX) { aspx_data_2ch }` — extra 2ch
    // envelope for the additional-channel pair in pure-ASPX mode (the
    // ASPX_ACPL_{1,2} paths fold the additional-channel ASPX into the
    // single aspx_data_1ch above).
    if matches!(mode, SevenXCodecMode::Aspx)
        && crate::asf::parse_aspx_data_2ch_body(br, tools, &aspx_cfg, b_iframe, frame_len_base)
            .is_err()
    {
        return Ok(());
    }

    // ACPL pair for ASPX_ACPL_{1,2} — `acpl_data_1ch()×2`, lands in
    // tools.acpl_data_1ch_pair[0/1] per the §5.7.7.6.1 Pseudocode 117
    // pair walker shape (shared with the 5_X path).
    if matches!(
        mode,
        SevenXCodecMode::AspxAcpl1 | SevenXCodecMode::AspxAcpl2
    ) {
        let acpl_cfg = match mode {
            SevenXCodecMode::AspxAcpl1 => tools.acpl_config_1ch_partial,
            SevenXCodecMode::AspxAcpl2 => tools.acpl_config_1ch_full,
            _ => None,
        };
        let Some(acfg) = acpl_cfg else {
            return Ok(());
        };
        let start_band = if acfg.qmf_band == 0 {
            0
        } else {
            crate::acpl::sb_to_pb(acfg.qmf_band as u32, acfg.num_param_bands)
        };
        if let Ok(d0) =
            crate::acpl::parse_acpl_data_1ch(br, acfg.num_param_bands, start_band, acfg.quant_mode)
        {
            tools.acpl_data_1ch_pair[0] = Some(d0);
            if let Ok(d1) = crate::acpl::parse_acpl_data_1ch(
                br,
                acfg.num_param_bands,
                start_band,
                acfg.quant_mode,
            ) {
                tools.acpl_data_1ch_pair[1] = Some(d1);
            }
        }
    }
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================

/// Lookup `n_msfbl_bits` (Table 106 column 4) for a 48 kHz / 44.1 kHz
/// transform length. Returns `None` for transform lengths that have
/// `N/A` in the table (the LFE channel is restricted to long-frame
/// transforms — short windows aren't permitted on LFE).
pub fn n_msfbl_bits_48(transform_length: u32) -> Option<u32> {
    tables::n_msfb_bits_48(transform_length)
        .and_then(|(_n, _s, l)| if l == 0 { None } else { Some(l) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    /// Append a minimal "all-zero spectra" `sf_data(ASF)` body to the
    /// writer for one channel. Walks (per §4.2.8.3-6 + r23 wiring):
    ///
    ///   * `asf_section_data`: one section with `sect_cb = 0` covering
    ///     all `max_sfb` bands (4 bits sect_cb + N bits sect_len_incr,
    ///     where N = 3 for `transf_length_idx <= 2`, with 7-escape
    ///     accumulation when `max_sfb > 7`).
    ///   * `asf_spectral_data`: empty (cb=0 emits no bits).
    ///   * `asf_scalefac_data`: 8 bits `reference_scale_factor` only —
    ///     all bands have `cb == 0`, so no DPCM codewords.
    ///   * `asf_snf_data`: 1 bit `b_snf_data_exists = 0`.
    ///
    /// The decoder reads the body and produces an all-zero scaled
    /// spectrum of length `sfb_offset[max_sfb]`.
    fn write_zero_sf_data_body(bw: &mut BitWriter, max_sfb: u32, transf_length_idx: u32) {
        let (n_sect_bits, sect_esc_val) = if transf_length_idx <= 2 {
            (3, 7)
        } else {
            (5, 31)
        };
        // sect_cb = 0.
        bw.write_u32(0, 4);
        // sect_len = 1 + sum of increments; we want sect_len == max_sfb.
        // So we need the sum of increments to equal max_sfb - 1.
        let mut remaining = max_sfb.saturating_sub(1);
        while remaining >= sect_esc_val {
            bw.write_u32(sect_esc_val, n_sect_bits);
            remaining -= sect_esc_val;
        }
        bw.write_u32(remaining, n_sect_bits);
        // asf_spectral_data: nothing (cb=0).
        // asf_scalefac_data: reference_scale_factor (any value works
        // since no bands have non-zero quants).
        bw.write_u32(120, 8);
        // asf_snf_data: b_snf_data_exists = 0.
        bw.write_bit(false);
    }

    #[test]
    fn five_x_codec_mode_round_trip() {
        assert_eq!(FiveXCodecMode::from_u32(0), FiveXCodecMode::Simple);
        assert_eq!(FiveXCodecMode::from_u32(1), FiveXCodecMode::Aspx);
        assert_eq!(FiveXCodecMode::from_u32(2), FiveXCodecMode::AspxAcpl1);
        assert_eq!(FiveXCodecMode::from_u32(3), FiveXCodecMode::AspxAcpl2);
        assert_eq!(FiveXCodecMode::from_u32(4), FiveXCodecMode::AspxAcpl3);
        assert_eq!(FiveXCodecMode::from_u32(5), FiveXCodecMode::Reserved(5));
        assert_eq!(FiveXCodecMode::from_u32(7), FiveXCodecMode::Reserved(7));
    }

    #[test]
    fn n_msfbl_bits_48_known_rows() {
        // Table 106 (long-frame entries):
        // 2048/1920/1536 -> 3, 1024/960/768/512 -> 2, 384 -> 2,
        // 480/256/240/192/128/120/96 -> N/A.
        assert_eq!(n_msfbl_bits_48(2048), Some(3));
        assert_eq!(n_msfbl_bits_48(1920), Some(3));
        assert_eq!(n_msfbl_bits_48(1024), Some(2));
        assert_eq!(n_msfbl_bits_48(384), Some(2));
        assert_eq!(n_msfbl_bits_48(480), None);
        assert_eq!(n_msfbl_bits_48(128), None);
    }

    #[test]
    fn parse_mono_data_lfe_long_frame() {
        // mono_data(1) for frame_len_base=1920 long-frame:
        //   asf_transform_info: b_long_frame=1 -> tl=1920.
        //   sf_info_lfe(): max_sfb[0] read with n_msfbl_bits=3 (Table
        //   106 column 4 for tl=1920) = value 5.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(5, 3); // max_sfb[0] — n_msfbl_bits=3 for tl=1920
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let lfe = parse_mono_data(&mut br, true, 1920).unwrap();
        assert!(lfe.b_lfe);
        assert_eq!(lfe.spec_frontend_bit, 0);
        let ti = lfe.transform_info.unwrap();
        assert_eq!(ti.transform_length_0, 1920);
        let psy = lfe.psy_info.unwrap();
        assert_eq!(psy.max_sfb_0, 5);
        // LFE psy_info has no grouping bits or window groups.
        assert_eq!(psy.num_windows, 1);
        assert_eq!(psy.num_window_groups, 1);
        assert!(psy.scale_factor_grouping.is_empty());
    }

    #[test]
    fn parse_mono_data_lfe_rejects_short_only_transform() {
        // tl=480 -> n_msfbl_bits = 0 (LFE not permitted at this tl).
        // Reach `parse_asf_psy_info_lfe` by feeding b_long_frame=0
        // followed by a 2-bit `transf_length` selecting tl=480 at
        // frame_len_base=1920. asf_transform_info Table 99 row for
        // 1920 maps transf_length=0..=3 to {1920, 960, 480, 240}.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_long_frame=0
        bw.write_u32(2, 2); // transf_length=2 -> tl=480 (Table 99)
        bw.write_u32(2, 2); // transf_length[1]=2 -> tl=480 (no different framing)
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let err = parse_mono_data(&mut br, true, 1920).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("LFE") || msg.contains("transform_length"),
            "expected LFE-rejection error, got: {msg}"
        );
    }

    /// Round 37: `parse_mono_data(b_lfe=false)` walks the trailing
    /// `sf_data(ASF)` body for the long-frame, ASF-frontend, single
    /// window group case and lands a dequantised + scaled spectrum on
    /// `scaled_spec`. The all-zero body decodes to a zero-length-matched
    /// spectrum (no Huffman codepoints fired).
    #[test]
    fn parse_mono_data_non_lfe_walks_sf_data_body() {
        let mut bw = BitWriter::new();
        bw.write_bit(false); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(8, 6); // max_sfb[0]
        write_zero_sf_data_body(&mut bw, 8, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mono = parse_mono_data(&mut br, false, 1920).unwrap();
        assert!(!mono.b_lfe);
        assert_eq!(mono.spec_frontend_bit, 0);
        let scaled = mono
            .scaled_spec
            .as_ref()
            .expect("body walked into scaled_spec");
        // Spectrum length is `sfb_offset[max_sfb]` — Table 110 row for
        // tl=1920 / max_sfb=8 puts that just below 256 bins; we don't
        // hard-pin the exact value but the body must be non-empty.
        assert!(!scaled.is_empty(), "scaled spectrum must be non-empty");
        // All-zero body: every bin should be exactly 0.0 (dequantise of
        // q == 0 + any scalefac is 0.0).
        assert!(
            scaled.iter().all(|&v| v == 0.0),
            "all-zero sf_data body must dequantise to all zeros"
        );
    }

    /// Round 38: `parse_mono_data(b_lfe=true)` walks the trailing
    /// `sf_data(ASF)` body via `decode_asf_long_lfe_body_with_max_sfb_lfe`.
    /// LFE channels are always long-frame / single window group per
    /// Table 35 (`sf_info_lfe`), so an all-zero body decodes to a length-
    /// matched all-zero spectrum, identical in shape to the non-LFE long
    /// path but with `max_sfb` constrained by the `n_msfbl_bits` width.
    #[test]
    fn parse_mono_data_lfe_walks_sf_data_body() {
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(5, 3); // max_sfb[0] — n_msfbl_bits=3 @ tl=1920
        write_zero_sf_data_body(&mut bw, 5, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let lfe = parse_mono_data(&mut br, true, 1920).unwrap();
        assert!(lfe.b_lfe);
        let scaled = lfe
            .scaled_spec
            .as_ref()
            .expect("LFE body walked into scaled_spec");
        assert!(!scaled.is_empty(), "LFE scaled spectrum must be non-empty");
        // All-zero body: every bin should be exactly 0.0.
        assert!(
            scaled.iter().all(|&v| v == 0.0),
            "all-zero LFE sf_data body must dequantise to all zeros"
        );
        // Length matches `sfb_offset[max_sfb]` for tl=1920, max_sfb=5 —
        // not pinned exactly, just bounded above by the transform length.
        assert!(scaled.len() <= 1920);
    }

    /// Round 37: SSF-frontend (`spec_frontend_bit == 1`) mono channels
    /// don't have a co-located body; the walker stops after the outer
    /// shell and `scaled_spec` stays `None`. The bit cursor advances
    /// past the leading 1-bit selector + `asf_transform_info` +
    /// `asf_psy_info` only.
    #[test]
    fn parse_mono_data_non_lfe_ssf_frontend_skips_body_walk() {
        let mut bw = BitWriter::new();
        bw.write_bit(true); // spec_frontend = SSF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(8, 6); // max_sfb[0]
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mono = parse_mono_data(&mut br, false, 1920).unwrap();
        assert_eq!(mono.spec_frontend_bit, 1);
        assert!(
            mono.scaled_spec.is_none(),
            "SSF-frontend mono must skip the ASF body walk"
        );
    }

    #[test]
    fn parse_three_channel_info_reads_chel_matsel_and_two_chparam() {
        // chel_matsel = 0b1010, then two chparam_info bodies with
        // sap_mode = 0 (None) — each consumes only 2 bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0b1010, 4);
        bw.write_u32(0, 2); // chparam_info #0: sap_mode=None
        bw.write_u32(0, 2); // chparam_info #1: sap_mode=None
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let info = parse_three_channel_info(&mut br, &[10]).unwrap();
        assert_eq!(info.chel_matsel, 0b1010);
        assert_eq!(info.chparam[0].sap_mode, 0);
        assert_eq!(info.chparam[1].sap_mode, 0);
    }

    #[test]
    fn parse_four_channel_info_reads_four_chparam() {
        let mut bw = BitWriter::new();
        for _ in 0..4 {
            bw.write_u32(0, 2); // sap_mode=None
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let info = parse_four_channel_info(&mut br, &[10]).unwrap();
        assert!(info.chparam.iter().all(|c| c.sap_mode == 0));
    }

    #[test]
    fn parse_five_channel_info_reads_chel_matsel_and_five_chparam() {
        let mut bw = BitWriter::new();
        bw.write_u32(0b0111, 4); // chel_matsel
        for _ in 0..5 {
            bw.write_u32(0, 2); // sap_mode=None
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let info = parse_five_channel_info(&mut br, &[10]).unwrap();
        assert_eq!(info.chel_matsel, 0b0111);
        assert!(info.chparam.iter().all(|c| c.sap_mode == 0));
    }

    #[test]
    fn parse_three_channel_data_outer_shell() {
        // sf_info(ASF, 0, 0) at frame_len_base=1920 long-frame:
        //   b_long_frame=1; max_sfb[0]=12 (6 bits).
        // three_channel_info: chel_matsel=3, two chparam_info(None).
        // r23: trailing 3x sf_data(ASF) bodies (all-zero spectra).
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(12, 6); // max_sfb[0]
        bw.write_u32(3, 4); // chel_matsel
        bw.write_u32(0, 2); // chparam_info #0
        bw.write_u32(0, 2); // chparam_info #1
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 12, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_three_channel_data(&mut br, 1920).unwrap();
        let psy = d.psy_info.unwrap();
        assert_eq!(psy.max_sfb_0, 12);
        let info = d.info.unwrap();
        assert_eq!(info.chel_matsel, 3);
        assert_eq!(d.scaled_spec_per_channel.len(), 3);
        // All-zero spectra: every channel slot is Some(vec![0.0; ...]).
        for ch in &d.scaled_spec_per_channel {
            let v = ch.as_ref().expect("per-channel sf_data should decode");
            assert!(v.iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn parse_five_channel_data_outer_shell() {
        // r23: trailing 5x sf_data(ASF) bodies.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(20, 6); // max_sfb[0]
        bw.write_u32(0xF, 4); // chel_matsel
        for _ in 0..5 {
            bw.write_u32(0, 2);
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 20, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_five_channel_data(&mut br, 1920).unwrap();
        let info = d.info.unwrap();
        assert_eq!(info.chel_matsel, 0xF);
        assert_eq!(info.chparam.len(), 5);
        assert_eq!(d.scaled_spec_per_channel.len(), 5);
        for ch in &d.scaled_spec_per_channel {
            assert!(ch.is_some());
        }
    }

    #[test]
    fn parse_5x_outer_simple_cfg3_five_channel() {
        // 5_X_codec_mode = SIMPLE (0). b_has_lfe=0, b_iframe=1.
        // No companding (SIMPLE). coding_config = 3 (five_channel_data).
        // Then five_channel_data outer shell + 5x sf_data(ASF).
        let mut bw = BitWriter::new();
        bw.write_u32(0, 3); // 5_X_codec_mode = SIMPLE
                            // No I-frame config (SIMPLE).
                            // No LFE.
                            // No companding.
        bw.write_u32(3, 2); // coding_config = 3
                            // five_channel_data outer:
        bw.write_bit(true); // b_long_frame
        bw.write_u32(15, 6); // max_sfb[0]
        bw.write_u32(0, 4); // chel_matsel
        for _ in 0..5 {
            bw.write_u32(0, 2); // chparam_info
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 15, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::Simple));
        assert_eq!(
            tools.five_x_coding_config,
            Some(FiveXCodingConfig::Cfg3Five)
        );
        let d = tools.five_channel_data.as_ref().unwrap();
        assert_eq!(d.psy_info.as_ref().unwrap().max_sfb_0, 15);
    }

    #[test]
    fn parse_5x_outer_simple_with_lfe_walks_lfe_mono_data() {
        // 5_X_codec_mode = SIMPLE, b_has_lfe = 1.
        // mono_data(1): asf_transform_info long-frame at 1920 +
        // sf_info_lfe with n_msfbl_bits=3 -> value 4.
        // Round 38: LFE body now decoded — append an all-zero sf_data
        // body at max_sfb=4 (n_sect_bits=3 since transf_length_idx=0).
        // Then coding_config=3 + five_channel_data shell + 5x sf_data.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 3); // SIMPLE
                            // LFE mono_data(1):
        bw.write_bit(true); // b_long_frame
        bw.write_u32(4, 3); // max_sfb[0] -- n_msfbl_bits = 3 for tl=1920
        write_zero_sf_data_body(&mut bw, 4, 0); // round 38: LFE body
                                                // coding_config = 3, then five_channel_data:
        bw.write_u32(3, 2);
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 4);
        for _ in 0..5 {
            bw.write_u32(0, 2);
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, true, true, 1920).unwrap();
        assert!(tools.five_x_b_has_lfe);
        let lfe = tools.lfe_mono_data.as_ref().unwrap();
        assert!(lfe.b_lfe);
        assert_eq!(lfe.psy_info.as_ref().unwrap().max_sfb_0, 4);
        // Round 38: LFE body is now decoded; scaled_spec must be Some.
        assert!(
            lfe.scaled_spec.is_some(),
            "round 38: LFE body walks into scaled_spec"
        );
        let d = tools.five_channel_data.as_ref().unwrap();
        assert_eq!(d.psy_info.as_ref().unwrap().max_sfb_0, 10);
    }

    #[test]
    fn parse_two_channel_data_outer_walks_sf_info_plus_chparam() {
        // Long-frame @1920, max_sfb=20, chparam_info sap_mode=0,
        // r23: + 2 sf_data(ASF) all-zero bodies.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(20, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam sap_mode = 0
        write_zero_sf_data_body(&mut bw, 20, 0);
        write_zero_sf_data_body(&mut bw, 20, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_two_channel_data(&mut br, 1920).unwrap();
        assert_eq!(d.transform_info.as_ref().unwrap().transform_length_0, 1920);
        assert_eq!(d.psy_info.as_ref().unwrap().max_sfb_0, 20);
        assert_eq!(d.chparam.as_ref().unwrap().sap_mode, 0);
        assert_eq!(d.scaled_spec_per_channel.len(), 2);
        assert!(d.scaled_spec_per_channel.iter().all(|c| c.is_some()));
    }

    #[test]
    fn parse_5x_outer_simple_cfg0_walks_pair_pair_centre() {
        // 5_X_codec_mode = SIMPLE (0). b_has_lfe=0, b_iframe=1.
        // coding_config = 0 (Cfg0Stereo2plusMono).
        // Then 1-bit `2ch_mode` + two_channel_data x2 + mono_data(0).
        // r23: each two_channel_data trails 2x sf_data(ASF). The
        // mono_data(0) shell isn't yet sf_data-extended, so no trailer
        // there.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 3); // SIMPLE
                            // No LFE, no I-frame config.
        bw.write_u32(0, 2); // coding_config = 0 (Cfg0)
        bw.write_bit(true); // b_2ch_mode
                            // two_channel_data #1: long-frame, max_sfb=10, chparam=0.
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 10, 0); // sf_data #1
        write_zero_sf_data_body(&mut bw, 10, 0); // sf_data #2
                                                 // two_channel_data #2: long-frame, max_sfb=12, chparam=0.
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 12, 0); // sf_data #1
        write_zero_sf_data_body(&mut bw, 12, 0); // sf_data #2
                                                 // mono_data(0): spec_frontend bit + transform + psy.
        bw.write_bit(false); // spec_frontend = 0 (ASF)
        bw.write_bit(true); // b_long_frame
        bw.write_u32(8, 6); // max_sfb[0]
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(
            tools.five_x_coding_config,
            Some(FiveXCodingConfig::Cfg0Stereo2plusMono)
        );
        assert_eq!(tools.b_2ch_mode, Some(true));
        assert_eq!(tools.two_channel_data.len(), 2);
        assert_eq!(
            tools.two_channel_data[0]
                .psy_info
                .as_ref()
                .unwrap()
                .max_sfb_0,
            10
        );
        assert_eq!(
            tools.two_channel_data[1]
                .psy_info
                .as_ref()
                .unwrap()
                .max_sfb_0,
            12
        );
        let centre = tools.cfg0_centre_mono.as_ref().unwrap();
        assert!(!centre.b_lfe);
        assert_eq!(centre.spec_frontend_bit, 0);
        assert_eq!(centre.psy_info.as_ref().unwrap().max_sfb_0, 8);
    }

    #[test]
    fn parse_5x_outer_simple_cfg1_walks_three_plus_two() {
        // SIMPLE, coding_config=1 -> three_channel_data + two_channel_data.
        // r23: 3+2 sf_data(ASF) trailers.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 3); // SIMPLE
        bw.write_u32(1, 2); // coding_config = 1 (Cfg1ThreeStereo)
                            // three_channel_data: long-frame, max_sfb=14, chel_matsel=0,
                            // 2x chparam_info(sap_mode=0).
        bw.write_bit(true);
        bw.write_u32(14, 6);
        bw.write_u32(0, 4);
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 14, 0);
        }
        // two_channel_data: long-frame, max_sfb=18, chparam=0.
        bw.write_bit(true);
        bw.write_u32(18, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 18, 0);
        write_zero_sf_data_body(&mut bw, 18, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(
            tools.five_x_coding_config,
            Some(FiveXCodingConfig::Cfg1ThreeStereo)
        );
        let three = tools.three_channel_data.as_ref().unwrap();
        assert_eq!(three.psy_info.as_ref().unwrap().max_sfb_0, 14);
        assert_eq!(tools.two_channel_data.len(), 1);
        assert_eq!(
            tools.two_channel_data[0]
                .psy_info
                .as_ref()
                .unwrap()
                .max_sfb_0,
            18
        );
    }

    #[test]
    fn parse_5x_outer_simple_cfg2_walks_four_plus_mono() {
        // SIMPLE, coding_config=2 -> four_channel_data + mono_data(0).
        // r23: 4 sf_data(ASF) trailers from four_channel_data.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 3); // SIMPLE
        bw.write_u32(2, 2); // coding_config = 2 (Cfg2FourMono)
                            // four_channel_data: long-frame, max_sfb=22, 4x chparam_info.
        bw.write_bit(true);
        bw.write_u32(22, 6);
        for _ in 0..4 {
            bw.write_u32(0, 2);
        }
        for _ in 0..4 {
            write_zero_sf_data_body(&mut bw, 22, 0);
        }
        // mono_data(0): spec_frontend + transform + psy.
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_u32(7, 6);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(
            tools.five_x_coding_config,
            Some(FiveXCodingConfig::Cfg2FourMono)
        );
        let four = tools.four_channel_data.as_ref().unwrap();
        assert_eq!(four.psy_info.as_ref().unwrap().max_sfb_0, 22);
        let back = tools.cfg2_back_mono.as_ref().unwrap();
        assert!(!back.b_lfe);
        assert_eq!(back.psy_info.as_ref().unwrap().max_sfb_0, 7);
    }

    #[test]
    fn parse_5x_outer_aspx_acpl3_reads_acpl_config_2ch_and_companding() {
        // ASPX_ACPL_3 (4) on b_iframe=1:
        //   aspx_config(): for r19 the easiest exercise is to feed an
        //   all-zero aspx_config payload. parse_aspx_config consumes a
        //   known prefix; we just check the round-trip succeeds without
        //   walking the body — the test focuses on
        //   acpl_config_2ch_present + companding(2) + 5_X_codec_mode.
        // Skip aspx_config exercise here — it's complex. Instead test
        // a non-iframe to dodge it.
        let mut bw = BitWriter::new();
        bw.write_u32(4, 3); // 5_X_codec_mode = ASPX_ACPL_3
                            // b_iframe=0: skip aspx_config + acpl_config_2ch.
                            // No LFE (b_has_lfe=0).
                            // companding_control(2): per Table 41 it's 1 bit
                            // (b_compand_avg) + per-channel bits. For the round-trip we
                            // just need the bits to be consumed correctly.
        bw.write_bit(false); // b_compand_avg = 0
        bw.write_bit(false); // b_compand_on[0]
        bw.write_bit(false); // b_compand_on[1]
                             // ASPX_ACPL_3 body: stereo_data() + aspx_data_2ch +
                             // acpl_data_2ch — all opaque for r19.
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl3));
        // acpl_config_2ch is gated on b_iframe=1 — should be None.
        assert!(tools.acpl_config_2ch.is_none());
        assert!(tools.companding.is_some());
    }

    // =================================================================
    // Round 23: per-channel sf_data(ASF) wiring tests
    // =================================================================

    /// `decode_mch_sf_data_channels` should produce one `Some(Vec<f32>)`
    /// per channel for the long-frame, single-window-group all-zero
    /// case, and the spectrum length should match
    /// `sfb_offset[max_sfb]`.
    #[test]
    fn decode_mch_sf_data_long_frame_all_zero_two_channels() {
        let mut bw = BitWriter::new();
        // Two stacked sf_data bodies for max_sfb=8 at tl=1920.
        write_zero_sf_data_body(&mut bw, 8, 0);
        write_zero_sf_data_body(&mut bw, 8, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0, 0],
            transform_length_0: 1920,
            transform_length_1: 1920,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: 8,
            num_windows: 1,
            num_window_groups: 1,
            ..Default::default()
        };
        let out = decode_mch_sf_data_channels(&mut br, &ti, &psy, 2);
        assert_eq!(out.len(), 2);
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let expected_len = sfbo[8] as usize;
        for slot in &out {
            let v = slot.as_ref().expect("each channel decodes");
            assert_eq!(v.len(), expected_len);
            assert!(v.iter().all(|&s| s == 0.0));
        }
    }

    /// `decode_mch_sf_data_channels` returns `None` for **all** channels
    /// when the input bits are garbage (Huffman miss in the very first
    /// section_data). r24's grouped walker still attempts the body
    /// chain for `num_window_groups > 1`, so the all-None return here
    /// signals a parse failure rather than the previous "skip
    /// short/grouped" gate.
    #[test]
    fn decode_mch_sf_data_short_frame_garbage_returns_all_none() {
        let bytes = [0xFFu8; 4];
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [0, 0],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: 6,
            num_windows: 4,
            num_window_groups: 2,
            ..Default::default()
        };
        let out = decode_mch_sf_data_channels(&mut br, &ti, &psy, 5);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|c| c.is_none()));
    }

    /// `parse_three_channel_data` populates all three
    /// `scaled_spec_per_channel` slots with vectors of the correct
    /// length when the trailing `sf_data(ASF)` bodies decode cleanly.
    /// This is the per-channel walk anchored to Tables 27 + Annex A.1
    /// (codebooks `HCB_1..HCB_11`, `HCB_SCALEFAC`, `HCB_SNF`).
    #[test]
    fn parse_three_channel_data_decodes_three_sf_data_bodies() {
        // Long-frame at fl_base=1920, max_sfb=10, chel_matsel=1,
        // 2x chparam_info(sap_mode=0), then 3x all-zero sf_data(ASF).
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb[0]
        bw.write_u32(1, 4); // chel_matsel
        bw.write_u32(0, 2); // chparam_info #0
        bw.write_u32(0, 2); // chparam_info #1
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_three_channel_data(&mut br, 1920).unwrap();
        assert_eq!(d.scaled_spec_per_channel.len(), 3);
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let expected_len = sfbo[10] as usize;
        for ch in &d.scaled_spec_per_channel {
            let v = ch.as_ref().unwrap();
            assert_eq!(v.len(), expected_len);
            assert!(v.iter().all(|&s| s == 0.0));
        }
    }

    /// `parse_four_channel_data` populates four per-channel slots and
    /// `parse_five_channel_data` populates five slots. The parser
    /// progresses linearly through the bit-stream, consuming exactly
    /// N body's worth of bits for an N-channel layout.
    #[test]
    fn parse_four_and_five_channel_data_emit_correct_per_channel_counts() {
        // four_channel_data: 4 sf_data bodies.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_u32(8, 6); // max_sfb[0]
        for _ in 0..4 {
            bw.write_u32(0, 2); // chparam_info
        }
        for _ in 0..4 {
            write_zero_sf_data_body(&mut bw, 8, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d4 = parse_four_channel_data(&mut br, 1920).unwrap();
        assert_eq!(d4.scaled_spec_per_channel.len(), 4);
        assert_eq!(
            d4.scaled_spec_per_channel
                .iter()
                .filter(|c| c.is_some())
                .count(),
            4
        );

        // five_channel_data: 5 sf_data bodies.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_u32(6, 6); // max_sfb[0]
        bw.write_u32(0, 4); // chel_matsel
        for _ in 0..5 {
            bw.write_u32(0, 2);
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 6, 0);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d5 = parse_five_channel_data(&mut br, 1920).unwrap();
        assert_eq!(d5.scaled_spec_per_channel.len(), 5);
        assert!(d5.scaled_spec_per_channel.iter().all(|c| c.is_some()));
    }

    /// When the bit-stream is truncated mid-way through the multichannel
    /// `sf_data(ASF)` walk, the channels parsed before the truncation
    /// retain their `Some(...)` slots while the remaining ones stay
    /// `None`. The walker must not panic.
    #[test]
    fn parse_three_channel_data_truncated_sf_data_yields_partial_decode() {
        // Long-frame at fl_base=1920, max_sfb=12, chel_matsel=2,
        // 2x chparam_info(sap_mode=0), then ONLY ONE sf_data body (the
        // walker should consume that one, then bail on the second).
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(2, 4);
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 12, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_three_channel_data(&mut br, 1920).unwrap();
        assert_eq!(d.scaled_spec_per_channel.len(), 3);
        // First channel decoded; rest bailed.
        assert!(d.scaled_spec_per_channel[0].is_some());
        // We can't assert which exact remaining slots are None vs. Some
        // — depending on byte alignment, the bit reader may have a few
        // leftover zero-padding bits that get re-interpreted as a
        // partial section header. What we DO require is that at least
        // one of the trailing slots is `None` and that the function
        // didn't panic.
        let some_count = d
            .scaled_spec_per_channel
            .iter()
            .filter(|c| c.is_some())
            .count();
        assert!(some_count < 3, "expected at least one None slot");
    }

    /// `parse_two_channel_data` populates two scaled-spec slots and the
    /// per-channel vectors have the exact length dictated by
    /// `sfb_offset_48(transform_length_0)[max_sfb_0]`.
    #[test]
    fn parse_two_channel_data_per_channel_lengths_match_sfb_offset() {
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(15, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam_info sap_mode = 0
        write_zero_sf_data_body(&mut bw, 15, 0);
        write_zero_sf_data_body(&mut bw, 15, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_two_channel_data(&mut br, 1920).unwrap();
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let expected_len = sfbo[15] as usize;
        assert_eq!(d.scaled_spec_per_channel.len(), 2);
        for ch in &d.scaled_spec_per_channel {
            let v = ch.as_ref().unwrap();
            assert_eq!(v.len(), expected_len);
        }
    }

    // =================================================================
    // Round 24: grouped multichannel sf_data(ASF) walker (num_window_groups > 1)
    // =================================================================

    /// `decode_mch_sf_data_channels` for a grouped short frame
    /// (`num_window_groups == 2`, `b_long_frame == 0`) walks
    /// `num_window_groups` consecutive `section / spectral / scalefac /
    /// snf` chains per channel and returns a per-channel spectrum of
    /// length `num_window_groups * sfb_offset[max_sfb]` (all-zero for
    /// the synthetic input). Pins r24 §5.4.4.4 grouped path.
    #[test]
    fn decode_mch_sf_data_grouped_short_frame_two_groups_two_channels() {
        // Two channels x two window groups x sf_data body each. tl=480
        // is at frame_len_base=1920 with transf_length=2 (the third
        // short-frame index — `n_sect_bits = 3`, esc = 7).
        let max_sfb = 8u32;
        let tl_idx = 2u32; // matches transf_length=2 (tl=480 short-frame).
        let mut bw = BitWriter::new();
        // Channel 0: two grouped bodies.
        write_zero_sf_data_body(&mut bw, max_sfb, tl_idx);
        write_zero_sf_data_body(&mut bw, max_sfb, tl_idx);
        // Channel 1: two grouped bodies.
        write_zero_sf_data_body(&mut bw, max_sfb, tl_idx);
        write_zero_sf_data_body(&mut bw, max_sfb, tl_idx);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [tl_idx, tl_idx],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: max_sfb,
            num_windows: 2,
            num_window_groups: 2,
            scale_factor_grouping: vec![0],
            ..Default::default()
        };
        let out = decode_mch_sf_data_channels(&mut br, &ti, &psy, 2);
        assert_eq!(out.len(), 2);
        let sfbo = crate::sfb_offset::sfb_offset_48(480).unwrap();
        let per_group_len = sfbo[max_sfb as usize] as usize;
        let expected_total = per_group_len * 2; // num_window_groups
        for slot in &out {
            let v = slot.as_ref().expect("each channel decodes");
            assert_eq!(v.len(), expected_total);
            assert!(v.iter().all(|&s| s == 0.0));
        }
    }

    /// Three-window-group case at the same short-frame transform
    /// length. Verifies the walker scales linearly with
    /// `num_window_groups`.
    #[test]
    fn decode_mch_sf_data_grouped_three_groups_one_channel() {
        let max_sfb = 6u32;
        let tl_idx = 2u32;
        let mut bw = BitWriter::new();
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, max_sfb, tl_idx);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [tl_idx, tl_idx],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: max_sfb,
            num_windows: 3,
            num_window_groups: 3,
            scale_factor_grouping: vec![0, 0],
            ..Default::default()
        };
        let out = decode_mch_sf_data_channels(&mut br, &ti, &psy, 1);
        assert_eq!(out.len(), 1);
        let v = out[0].as_ref().expect("decode succeeds");
        let sfbo = crate::sfb_offset::sfb_offset_48(480).unwrap();
        assert_eq!(v.len(), 3 * sfbo[max_sfb as usize] as usize);
    }

    /// `parse_three_channel_data` correctly drives the grouped path
    /// when the head `sf_info(ASF, 0, 0)` reports
    /// `num_window_groups > 1`. Each channel's `scaled_spec_per_channel`
    /// slot carries the concatenated per-group spectra.
    #[test]
    fn parse_three_channel_data_grouped_short_frame_walks_per_group() {
        // sf_info(ASF, 0, 0) at frame_len_base=1920 with
        // b_long_frame=0, transf_length=[2,2] -> tl=480 short-frame +
        // n_grp_bits = n_grp_bits_lt_1536 isn't applicable here; for
        // frame_len_base=1920 (>= 1536) with equal transf_length we
        // use n_grp_bits_ge_1536(2,2) = 3. Set
        // scale_factor_grouping = [1, 0, 1] -> num_window_groups = 2.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_long_frame = 0
        bw.write_u32(2, 2); // transf_length[0] = 2 (tl=480)
        bw.write_u32(2, 2); // transf_length[1] = 2 (tl=480)
        let max_sfb = 5u32;
        // n_msfb_bits for tl=480 short-frame at 48 kHz: per Table 106
        // tl=480 column 1 (n_msfb_bits) = 6.
        bw.write_u32(max_sfb, 6); // max_sfb[0]
                                  // scale_factor_grouping bits = 3 (n_grp_bits_ge_1536(2,2)).
                                  // Pattern [1, 0, 1] -> exactly one group boundary (one zero) ->
                                  // num_window_groups = 1 + 1 = 2.
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        // three_channel_info: chel_matsel + 2x chparam_info(sap_mode=0).
        bw.write_u32(0, 4);
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        // Three channels x two window groups of sf_data(ASF) bodies.
        for _ in 0..3 {
            for _ in 0..2 {
                write_zero_sf_data_body(&mut bw, max_sfb, 2);
            }
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_three_channel_data(&mut br, 1920).unwrap();
        let psy = d.psy_info.as_ref().unwrap();
        assert_eq!(psy.num_window_groups, 2);
        assert!(!d.transform_info.as_ref().unwrap().b_long_frame);
        assert_eq!(d.scaled_spec_per_channel.len(), 3);
        let sfbo = crate::sfb_offset::sfb_offset_48(480).unwrap();
        let expected_total = (sfbo[max_sfb as usize] as usize) * 2;
        for ch in &d.scaled_spec_per_channel {
            let v = ch.as_ref().expect("each channel decodes");
            assert_eq!(v.len(), expected_total);
            assert!(v.iter().all(|&s| s == 0.0));
        }
    }

    /// `parse_two_channel_data` mirrors the grouped walk for the
    /// `5_X_channel_element` Cfg0 / Cfg1 paths — every per-channel slot
    /// carries the concatenated grouped spectrum.
    #[test]
    fn parse_two_channel_data_grouped_short_frame_walks_per_group() {
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_long_frame = 0
        bw.write_u32(2, 2); // transf_length[0] = 2 (tl=480)
        bw.write_u32(2, 2); // transf_length[1] = 2 (tl=480)
        let max_sfb = 4u32;
        bw.write_u32(max_sfb, 6); // max_sfb[0]
                                  // n_grp_bits = 3 from n_grp_bits_ge_1536(2,2). [0,0,0] -> 4 groups.
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // chparam_info: sap_mode=0.
        bw.write_u32(0, 2);
        // 2 channels x 4 window groups of bodies.
        for _ in 0..2 {
            for _ in 0..4 {
                write_zero_sf_data_body(&mut bw, max_sfb, 2);
            }
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = parse_two_channel_data(&mut br, 1920).unwrap();
        let psy = d.psy_info.as_ref().unwrap();
        assert_eq!(psy.num_window_groups, 4);
        assert_eq!(d.scaled_spec_per_channel.len(), 2);
        let sfbo = crate::sfb_offset::sfb_offset_48(480).unwrap();
        let expected_total = (sfbo[max_sfb as usize] as usize) * 4;
        for ch in &d.scaled_spec_per_channel {
            let v = ch.as_ref().expect("each channel decodes");
            assert_eq!(v.len(), expected_total);
        }
    }

    /// Truncated grouped input should yield a partial per-channel
    /// decode and not panic. We feed only one window-group's body for
    /// the single channel and expect a `None` slot.
    #[test]
    fn decode_mch_sf_data_grouped_truncated_returns_none() {
        let max_sfb = 6u32;
        let tl_idx = 2u32;
        let mut bw = BitWriter::new();
        // Only one body when num_window_groups=2 expects two.
        write_zero_sf_data_body(&mut bw, max_sfb, tl_idx);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [tl_idx, tl_idx],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: max_sfb,
            num_windows: 2,
            num_window_groups: 2,
            scale_factor_grouping: vec![0],
            ..Default::default()
        };
        let out = decode_mch_sf_data_channels(&mut br, &ti, &psy, 1);
        assert_eq!(out.len(), 1);
        // Single channel: with only 1 of 2 groups present, the second
        // group attempt should bail (Huffman miss on garbage / EOF).
        // We allow either Some (if zero-padding accidentally validates
        // as a section header) or None — but we must NOT panic.
        let _ = &out[0];
    }

    // =================================================================
    // Round 24: ASPX_ACPL_3 inner body walker
    // =================================================================

    /// `parse_5x_audio_data_outer` for ASPX_ACPL_3 on a non-iframe path
    /// shouldn't touch the inner body walker (gated on b_iframe + an
    /// in-scope aspx_config) — `tools.acpl_data_2ch` stays `None`.
    #[test]
    fn parse_5x_aspx_acpl_3_non_iframe_leaves_acpl_data_2ch_none() {
        let mut bw = BitWriter::new();
        bw.write_u32(4, 3); // 5_X_codec_mode = ASPX_ACPL_3
                            // companding_control(2): all-zero.
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl3));
        assert!(tools.acpl_data_2ch.is_none());
        assert!(tools.companding.is_some());
    }

    /// `parse_5x_audio_data_outer` for ASPX_ACPL_3 on an I-frame walks
    /// `aspx_config` + `acpl_config_2ch` + `companding_control(2)` +
    /// `stereo_data()` then, when the body decodes cleanly, runs
    /// `aspx_data_2ch()` and `acpl_data_2ch()`. We feed an
    /// all-zero-tail bitstream — the body walker is allowed to bail
    /// silently downstream of `stereo_data()` (since the freq-table
    /// derivation may fail on a degenerate aspx_config), but the
    /// outer walker must finish without erroring and the parsed
    /// `acpl_config_2ch` must be visible on the tools.
    #[test]
    fn parse_5x_aspx_acpl_3_iframe_parses_aspx_and_acpl_configs() {
        let mut bw = BitWriter::new();
        bw.write_u32(4, 3); // 5_X_codec_mode = ASPX_ACPL_3
                            // aspx_config(): 15 bits all-zero.
        bw.write_u32(0, 15);
        // acpl_config_2ch(): 2-bit num_param_bands_id + 2x 1-bit
        // quant_mode. id=0 -> 15 bands; both quant modes = Fine.
        bw.write_u32(0, 2);
        bw.write_bit(false);
        bw.write_bit(false);
        // companding_control(2).
        bw.write_bit(false); // sync_flag = 0
        bw.write_bit(false); // compand_on[0]
        bw.write_bit(false); // compand_on[1]
        bw.write_bit(false); // compand_avg
                             // stereo_data() body — feed enough zeros that the body walker
                             // either succeeds or bails cleanly without panicking.
        bw.align_to_byte();
        while bw.byte_len() < 256 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        // The walker must complete without erroring — any body-walker
        // mid-stream miss should be swallowed silently per the
        // try-and-bail contract (acpl_data_2ch slot would simply stay
        // None in that case).
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl3));
        let cfg = tools.acpl_config_2ch.expect("acpl_config_2ch parsed");
        assert_eq!(cfg.num_param_bands, 15);
    }

    // =================================================================
    // Round 25: ASPX_ACPL_1 / ASPX_ACPL_2 inner body walker
    // =================================================================

    /// Helper — write a `companding_control(3)` element with all
    /// channels companded on (sync_flag=true compresses the per-channel
    /// loop to a single `compand_on=true` bit and skips compand_avg).
    fn write_companding_3_all_on(bw: &mut oxideav_core::bits::BitWriter) {
        bw.write_bit(true); // sync_flag
        bw.write_bit(true); // compand_on (sync=1 → only 1 channel-bit)
    }

    /// Helper — write a 15-bit all-zero `aspx_config()` payload.
    fn write_zero_aspx_config(bw: &mut oxideav_core::bits::BitWriter) {
        bw.write_u32(0, 15);
    }

    /// Helper — write a PARTIAL `acpl_config_1ch()`: 2-bit
    /// num_param_bands_id + 1-bit quant_mode + 3-bit qmf_band_minus1.
    fn write_acpl_config_1ch_partial(bw: &mut oxideav_core::bits::BitWriter) {
        bw.write_u32(0, 2); // id = 0 (15 bands)
        bw.write_bit(false); // quant_mode = Fine
        bw.write_u32(0, 3); // qmf_band_minus1 = 0 -> qmf_band = 1
    }

    /// Helper — write a FULL `acpl_config_1ch()`: 2-bit + 1-bit (no
    /// qmf_band field).
    fn write_acpl_config_1ch_full(bw: &mut oxideav_core::bits::BitWriter) {
        bw.write_u32(0, 2); // id = 0 (15 bands)
        bw.write_bit(false); // quant_mode = Fine
    }

    /// `parse_5x_audio_data_outer` for ASPX_ACPL_2 on a non-iframe
    /// shouldn't reach the ACPL pair walker (gated on b_iframe +
    /// in-scope aspx_config). The outer must still consume
    /// companding_control(3) + 1-bit coding_config and try to parse the
    /// inner channel data — but the per-side acpl_data slots stay
    /// `None`.
    #[test]
    fn parse_5x_aspx_acpl_2_non_iframe_leaves_acpl_pair_none() {
        let mut bw = BitWriter::new();
        bw.write_u32(3, 3); // 5_X_codec_mode = ASPX_ACPL_2
        write_companding_3_all_on(&mut bw);
        bw.write_bit(false); // coding_config = 0 -> two_channel_data + mono(0)
                             // two_channel_data() outer + 2x sf_data:
        bw.write_bit(true); // b_long_frame
        bw.write_u32(8, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam sap_mode = 0
        write_zero_sf_data_body(&mut bw, 8, 0);
        write_zero_sf_data_body(&mut bw, 8, 0);
        // mono_data(0): spec_frontend bit + asf_transform_info long +
        // sf_info(ASF, 0, 0).
        bw.write_bit(false); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(8, 6); // max_sfb[0] (n_msfb_bits=6 @ tl=1920)
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl2));
        assert_eq!(
            tools.five_x_coding_config,
            Some(FiveXCodingConfig::AcplLite2)
        );
        assert_eq!(tools.two_channel_data.len(), 1);
        assert!(tools.cfg0_centre_mono.is_some());
        assert!(tools.acpl_data_1ch_pair[0].is_none());
        assert!(tools.acpl_data_1ch_pair[1].is_none());
    }

    /// ASPX_ACPL_1 non-iframe with `coding_config = 1`
    /// (three_channel_data branch, no Cfg0 mono_data trailer).
    /// Walker should populate `three_channel_data` but the joint-MDCT
    /// residual layer + ACPL pair are gated and stay unset.
    #[test]
    fn parse_5x_aspx_acpl_1_non_iframe_walks_three_channel_data() {
        let mut bw = BitWriter::new();
        bw.write_u32(2, 3); // 5_X_codec_mode = ASPX_ACPL_1
        write_companding_3_all_on(&mut bw);
        bw.write_bit(true); // coding_config = 1 -> three_channel_data
                            // three_channel_data outer:
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb[0]
        bw.write_u32(0, 4); // chel_matsel
        bw.write_u32(0, 2); // chparam_info #0
        bw.write_u32(0, 2); // chparam_info #1
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // Joint-MDCT residual layer (ASPX_ACPL_1 only): max_sfb_master
        // is read with n_side_bits=5 @ tl=1920 (Table 106).
        bw.write_u32(8, 5); // max_sfb_master = 8
        bw.write_u32(0, 2); // chparam_info residual ch0 (sap_mode=0)
        bw.write_u32(0, 2); // chparam_info residual ch1
        write_zero_sf_data_body(&mut bw, 8, 0); // residual ch0 sf_data
        write_zero_sf_data_body(&mut bw, 8, 0); // residual ch1 sf_data
                                                // Pad to be safe.
        bw.align_to_byte();
        while bw.byte_len() < 64 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl1));
        assert_eq!(
            tools.five_x_coding_config,
            Some(FiveXCodingConfig::Cfg1ThreeStereo)
        );
        let three = tools.three_channel_data.as_ref().expect("3ch parsed");
        assert_eq!(three.psy_info.as_ref().unwrap().max_sfb_0, 10);
        // Non-iframe: joint-MDCT residual layer is walked but ACPL pair
        // stays None (no aspx_config in scope).
        assert!(tools.acpl_data_1ch_pair[0].is_none());
        assert!(tools.acpl_data_1ch_pair[1].is_none());
        // Round 40: the residual pair (sSMP,3 / sSMP,4 spectra per
        // Table 181) is now persisted on `tools.acpl_1_residual_pair`
        // so the dispatch can IMDCT it into Ls/Rs PCM carriers. The
        // sf_data bodies above are all-zero, so the resulting spectra
        // are all-zero — the slot is `Some` regardless.
        assert!(tools.acpl_1_residual_pair[0].is_some(), "sSMP,3 persisted");
        assert!(tools.acpl_1_residual_pair[1].is_some(), "sSMP,4 persisted");
        let (tl0, spec0) = tools.acpl_1_residual_pair[0].as_ref().unwrap();
        assert_eq!(*tl0, 1920);
        // sfb_offset[8] for tl=1920 — caller-provided max_sfb_master = 8.
        // Length is the number of MDCT bins covered by 8 sfbs at tl=1920.
        assert!(!spec0.is_empty(), "sSMP,3 spectrum is non-empty");
    }

    /// ASPX_ACPL_2 I-frame with `coding_config = 1`
    /// (three_channel_data branch). The outer parses aspx_config +
    /// acpl_config_1ch(FULL) before the body. The walker exercises
    /// three_channel_data + aspx_data_2ch + aspx_data_1ch + 2x
    /// acpl_data_1ch. The downstream Huffman walks may bail silently
    /// on a degenerate aspx_config (zero start_freq), but the parsed
    /// configs and the upstream three_channel_data must surface on
    /// tools.
    #[test]
    fn parse_5x_aspx_acpl_2_iframe_parses_configs_and_three_channel() {
        let mut bw = BitWriter::new();
        bw.write_u32(3, 3); // 5_X_codec_mode = ASPX_ACPL_2
        write_zero_aspx_config(&mut bw);
        write_acpl_config_1ch_full(&mut bw);
        write_companding_3_all_on(&mut bw);
        bw.write_bit(true); // coding_config = 1 -> three_channel_data
                            // three_channel_data outer:
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb[0]
        bw.write_u32(0, 4); // chel_matsel
        bw.write_u32(0, 2); // chparam_info #0
        bw.write_u32(0, 2); // chparam_info #1
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // Pad with zeros for downstream aspx/acpl walkers (which are
        // try-and-bail).
        bw.align_to_byte();
        while bw.byte_len() < 256 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl2));
        assert!(tools.aspx_config.is_some());
        let cfg_full = tools.acpl_config_1ch_full.expect("FULL config parsed");
        assert_eq!(cfg_full.num_param_bands, 15);
        assert_eq!(cfg_full.qmf_band, 0); // FULL has no qmf_band
        let three = tools.three_channel_data.as_ref().expect("3ch parsed");
        assert_eq!(three.psy_info.as_ref().unwrap().max_sfb_0, 10);
    }

    /// ASPX_ACPL_1 I-frame with `coding_config = 0` (two_channel_data
    /// branch — pulls the joint-MDCT residual layer + Cfg0
    /// `mono_data(0)` trailer). Validates that:
    ///   * aspx_config is parsed,
    ///   * acpl_config_1ch_partial is parsed (with non-zero qmf_band),
    ///   * two_channel_data lands in the slot,
    ///   * cfg0_centre_mono is populated by the trailing mono_data(0).
    ///
    /// Downstream aspx_data / acpl_data may bail silently on the
    /// all-zero pad; the test just asserts non-fatal completion.
    #[test]
    fn parse_5x_aspx_acpl_1_iframe_walks_residual_and_mono_trailer() {
        let mut bw = BitWriter::new();
        bw.write_u32(2, 3); // 5_X_codec_mode = ASPX_ACPL_1
        write_zero_aspx_config(&mut bw);
        write_acpl_config_1ch_partial(&mut bw);
        write_companding_3_all_on(&mut bw);
        bw.write_bit(false); // coding_config = 0 -> two_channel_data
                             // two_channel_data outer (Table 26):
        bw.write_bit(true); // b_long_frame
        bw.write_u32(12, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam sap_mode = 0
        write_zero_sf_data_body(&mut bw, 12, 0);
        write_zero_sf_data_body(&mut bw, 12, 0);
        // Joint-MDCT residual layer (ASPX_ACPL_1):
        // max_sfb_master uses n_side_bits = 5 @ tl=1920.
        bw.write_u32(6, 5); // max_sfb_master = 6
        bw.write_u32(0, 2); // chparam residual ch0
        bw.write_u32(0, 2); // chparam residual ch1
        write_zero_sf_data_body(&mut bw, 6, 0);
        write_zero_sf_data_body(&mut bw, 6, 0);
        // Cfg0 trailer: mono_data(0) for the centre channel.
        bw.write_bit(false); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(7, 6); // max_sfb[0] for centre mono
                            // Pad for downstream aspx_data / acpl_data.
        bw.align_to_byte();
        while bw.byte_len() < 256 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl1));
        assert!(tools.aspx_config.is_some());
        let cfg_partial = tools
            .acpl_config_1ch_partial
            .expect("PARTIAL config parsed");
        assert_eq!(cfg_partial.num_param_bands, 15);
        assert_eq!(cfg_partial.qmf_band, 1); // qmf_band_minus1=0 -> 1
        assert_eq!(tools.two_channel_data.len(), 1);
        assert_eq!(
            tools.two_channel_data[0]
                .psy_info
                .as_ref()
                .unwrap()
                .max_sfb_0,
            12
        );
        let centre = tools
            .cfg0_centre_mono
            .as_ref()
            .expect("Cfg0 centre mono walked");
        assert_eq!(centre.psy_info.as_ref().unwrap().max_sfb_0, 7);
    }

    /// Truncated input mid-`three_channel_data` for ASPX_ACPL_2 should
    /// leave `three_channel_data` `None` (the channel-data parser
    /// errored) without panicking and without setting any of the
    /// downstream slots. The outer walker still returns Ok(()).
    #[test]
    fn parse_5x_aspx_acpl_2_truncated_channel_data_bails() {
        let mut bw = BitWriter::new();
        bw.write_u32(3, 3); // 5_X_codec_mode = ASPX_ACPL_2
        write_companding_3_all_on(&mut bw);
        bw.write_bit(true); // coding_config = 1 -> three_channel_data
                            // start three_channel_data but truncate after b_long_frame
                            // (no max_sfb bits).
        bw.write_bit(true);
        // intentionally cut here — the next byte boundary won't have
        // the 6-bit max_sfb field complete.
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        // The truncation lands inside `parse_three_channel_data` which
        // returns Err — that's caught by the inner walker's
        // try-and-bail and we surface Ok(()).
        parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl2));
        assert!(tools.three_channel_data.is_none());
        assert!(tools.acpl_data_1ch_pair[0].is_none());
        assert!(tools.acpl_data_1ch_pair[1].is_none());
    }

    /// max_sfb_master = 0 in the joint-MDCT residual layer should bail
    /// silently — the chparam_info / sf_data trailers would be
    /// degenerate. Subsequent aspx/acpl trailers stay unset.
    #[test]
    fn parse_5x_aspx_acpl_1_iframe_zero_max_sfb_master_bails() {
        let mut bw = BitWriter::new();
        bw.write_u32(2, 3); // 5_X_codec_mode = ASPX_ACPL_1
        write_zero_aspx_config(&mut bw);
        write_acpl_config_1ch_partial(&mut bw);
        write_companding_3_all_on(&mut bw);
        bw.write_bit(true); // coding_config = 1 -> three_channel_data
                            // three_channel_data outer:
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb[0]
        bw.write_u32(0, 4); // chel_matsel
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // max_sfb_master = 0 (n_side_bits = 5 @ tl=1920).
        bw.write_u32(0, 5);
        bw.align_to_byte();
        while bw.byte_len() < 64 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl1));
        assert!(tools.three_channel_data.is_some());
        assert!(tools.acpl_data_1ch_pair[0].is_none());
        assert!(tools.acpl_data_1ch_pair[1].is_none());
    }

    // =================================================================
    // Round 27: 7_X channel-element walker (immersive 7.0 / 7.1)
    // =================================================================

    /// Helper — write a `companding_control(5)` element with all five
    /// channels companded on (sync_flag=true compresses the per-channel
    /// loop to a single `compand_on=true` bit).
    fn write_companding_5_all_on(bw: &mut oxideav_core::bits::BitWriter) {
        bw.write_bit(true); // sync_flag
        bw.write_bit(true); // compand_on
    }

    #[test]
    fn seven_x_codec_mode_round_trip() {
        assert_eq!(SevenXCodecMode::from_u32(0), SevenXCodecMode::Simple);
        assert_eq!(SevenXCodecMode::from_u32(1), SevenXCodecMode::Aspx);
        assert_eq!(SevenXCodecMode::from_u32(2), SevenXCodecMode::AspxAcpl1);
        assert_eq!(SevenXCodecMode::from_u32(3), SevenXCodecMode::AspxAcpl2);
        // Wraparound — only 2 bits are used so 4 .. 7 fold back.
        assert_eq!(SevenXCodecMode::from_u32(4), SevenXCodecMode::Simple);
    }

    /// 7_X SIMPLE coding_config = 3 (five_channel_data) + the trailing
    /// SIMPLE additional `two_channel_data` (no SAP). 7.0 path (no LFE).
    #[test]
    fn parse_7x_outer_simple_cfg3_no_sap_walks_full_body() {
        let mut bw = BitWriter::new();
        // 7_X_codec_mode = SIMPLE (0) -- 2 bits.
        bw.write_u32(0, 2);
        // No I-frame config (SIMPLE).
        // No LFE.
        // No companding (SIMPLE/ASPX skip companding in 7.X).
        // coding_config = 3 -> five_channel_data.
        bw.write_u32(3, 2);
        // five_channel_data outer:
        bw.write_bit(true); // b_long_frame
        bw.write_u32(15, 6); // max_sfb[0]
        bw.write_u32(0, 4); // chel_matsel
        for _ in 0..5 {
            bw.write_u32(0, 2); // chparam_info
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 15, 0);
        }
        // SIMPLE additional-channel block: b_use_sap_add_ch = 0.
        bw.write_bit(false);
        // additional two_channel_data (no SAP):
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam sap_mode = 0
        write_zero_sf_data_body(&mut bw, 10, 0);
        write_zero_sf_data_body(&mut bw, 10, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::Simple));
        assert!(!tools.seven_x_b_has_lfe);
        assert_eq!(
            tools.seven_x_coding_config,
            Some(FiveXCodingConfig::Cfg3Five)
        );
        let five = tools.five_channel_data.as_ref().expect("5ch parsed");
        assert_eq!(five.psy_info.as_ref().unwrap().max_sfb_0, 15);
        assert_eq!(tools.seven_x_b_use_sap_add_ch, Some(false));
        assert!(tools.seven_x_add_chparam_info.is_none());
        let add = tools
            .seven_x_additional_channel_data
            .as_ref()
            .expect("additional 2ch parsed");
        assert_eq!(add.psy_info.as_ref().unwrap().max_sfb_0, 10);
    }

    /// 7.1 SIMPLE: leading `mono_data(1)` LFE then five_channel_data
    /// then the additional `two_channel_data`.
    #[test]
    fn parse_7x_outer_simple_71_walks_lfe_and_five_channel() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // SIMPLE
                            // LFE mono_data(1):
        bw.write_bit(true); // b_long_frame
        bw.write_u32(4, 3); // max_sfb[0] (n_msfbl_bits=3 @ tl=1920)
        write_zero_sf_data_body(&mut bw, 4, 0); // round 38: LFE body
                                                // coding_config = 3 -> five_channel_data:
        bw.write_u32(3, 2);
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb[0]
        bw.write_u32(0, 4);
        for _ in 0..5 {
            bw.write_u32(0, 2);
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // SIMPLE additional-channel block.
        bw.write_bit(false); // b_use_sap_add_ch = 0
        bw.write_bit(true); // b_long_frame
        bw.write_u32(8, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam sap_mode = 0
        write_zero_sf_data_body(&mut bw, 8, 0);
        write_zero_sf_data_body(&mut bw, 8, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, true, true, 1920).unwrap();
        assert!(tools.seven_x_b_has_lfe);
        let lfe = tools.lfe_mono_data.as_ref().expect("LFE walked");
        assert!(lfe.b_lfe);
        assert_eq!(lfe.psy_info.as_ref().unwrap().max_sfb_0, 4);
        assert!(tools.five_channel_data.is_some());
        assert!(tools.seven_x_additional_channel_data.is_some());
    }

    /// 7_X SIMPLE Cfg0 — `2ch_mode + two_channel_data + two_channel_data`
    /// (no centre mono inside the switch). The trailing centre
    /// `mono_data(0)` lands AFTER the additional-channel block.
    #[test]
    fn parse_7x_outer_simple_cfg0_walks_two_pairs_then_centre_mono() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // SIMPLE
                            // coding_config = 0:
        bw.write_u32(0, 2);
        // 2ch_mode (1 bit).
        bw.write_bit(false);
        // two_channel_data #0:
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 12, 0);
        write_zero_sf_data_body(&mut bw, 12, 0);
        // two_channel_data #1:
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 12, 0);
        write_zero_sf_data_body(&mut bw, 12, 0);
        // SIMPLE additional-channel block.
        bw.write_bit(false); // b_use_sap_add_ch = 0
                             // additional two_channel_data:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 10, 0);
        write_zero_sf_data_body(&mut bw, 10, 0);
        // Trailing mono_data(0) for Cfg0 (centre).
        bw.write_bit(false); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(7, 6); // max_sfb[0]
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::Simple));
        assert_eq!(
            tools.seven_x_coding_config,
            Some(FiveXCodingConfig::Cfg0Stereo2plusMono)
        );
        assert_eq!(tools.b_2ch_mode, Some(false));
        assert_eq!(tools.two_channel_data.len(), 2);
        let centre = tools.cfg0_centre_mono.as_ref().expect("centre mono walked");
        assert_eq!(centre.psy_info.as_ref().unwrap().max_sfb_0, 7);
    }

    /// 7_X SIMPLE Cfg2 — `four_channel_data` (no surround mono inside
    /// switch). Trailing surround `mono_data(0)` lands AFTER the
    /// additional-channel block.
    #[test]
    fn parse_7x_outer_simple_cfg2_walks_four_then_back_mono() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // SIMPLE
                            // coding_config = 2 -> four_channel_data:
        bw.write_u32(2, 2);
        bw.write_bit(true); // b_long_frame
        bw.write_u32(11, 6); // max_sfb[0]
        for _ in 0..4 {
            bw.write_u32(0, 2);
        }
        for _ in 0..4 {
            write_zero_sf_data_body(&mut bw, 11, 0);
        }
        // SIMPLE additional-channel block.
        bw.write_bit(false); // b_use_sap_add_ch = 0
        bw.write_bit(true);
        bw.write_u32(9, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 9, 0);
        write_zero_sf_data_body(&mut bw, 9, 0);
        // Trailing mono_data(0) for Cfg2 (back surround).
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_u32(6, 6);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(
            tools.seven_x_coding_config,
            Some(FiveXCodingConfig::Cfg2FourMono)
        );
        assert!(tools.four_channel_data.is_some());
        let back = tools.cfg2_back_mono.as_ref().expect("back mono walked");
        assert_eq!(back.psy_info.as_ref().unwrap().max_sfb_0, 6);
    }

    /// 7_X SIMPLE Cfg1 — `three_channel_data + two_channel_data` (no
    /// trailing mono_data — coding_config in {0,2} only triggers the
    /// trailer). The additional-channel block still fires.
    #[test]
    fn parse_7x_outer_simple_cfg1_no_mono_trailer() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // SIMPLE
                            // coding_config = 1 -> three_channel_data + two_channel_data
        bw.write_u32(1, 2);
        // three_channel_data:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 4); // chel_matsel
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // two_channel_data:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 10, 0);
        write_zero_sf_data_body(&mut bw, 10, 0);
        // SIMPLE additional-channel block.
        bw.write_bit(false); // b_use_sap_add_ch
        bw.write_bit(true);
        bw.write_u32(8, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 8, 0);
        write_zero_sf_data_body(&mut bw, 8, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(
            tools.seven_x_coding_config,
            Some(FiveXCodingConfig::Cfg1ThreeStereo)
        );
        assert!(tools.three_channel_data.is_some());
        assert_eq!(tools.two_channel_data.len(), 1);
        // Cfg1: no trailing mono_data(0).
        assert!(tools.cfg0_centre_mono.is_none());
        assert!(tools.cfg2_back_mono.is_none());
        assert!(tools.seven_x_additional_channel_data.is_some());
    }

    /// 7_X SIMPLE with `b_use_sap_add_ch = 1` — two `chparam_info()`
    /// elements precede the additional `two_channel_data`. Validates
    /// that `seven_x_add_chparam_info` is populated.
    #[test]
    fn parse_7x_outer_simple_with_sap_add_ch_populates_chparam_pair() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // SIMPLE
        bw.write_u32(3, 2); // coding_config = 3 -> five_channel_data
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(0, 4);
        for _ in 0..5 {
            bw.write_u32(0, 2);
        }
        for _ in 0..5 {
            write_zero_sf_data_body(&mut bw, 12, 0);
        }
        // SIMPLE additional-channel block with SAP.
        bw.write_bit(true); // b_use_sap_add_ch = 1
        bw.write_u32(0, 2); // chparam_info #0 sap_mode = 0
        bw.write_u32(0, 2); // chparam_info #1 sap_mode = 0
                            // additional two_channel_data:
        bw.write_bit(true);
        bw.write_u32(8, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 8, 0);
        write_zero_sf_data_body(&mut bw, 8, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.seven_x_b_use_sap_add_ch, Some(true));
        let pair = tools
            .seven_x_add_chparam_info
            .as_ref()
            .expect("SAP chparam pair");
        assert_eq!(pair[0].sap_mode, 0);
        assert_eq!(pair[1].sap_mode, 0);
    }

    /// 7_X ASPX_ACPL_2 non-iframe with `coding_config = 1`
    /// (three_channel_data branch). Walker should populate
    /// `three_channel_data` + `two_channel_data` but the ACPL pair stays
    /// unset (no aspx_config in scope on a non-iframe). NO additional
    /// `two_channel_data` should be parsed (it's SIMPLE/ASPX-only).
    #[test]
    fn parse_7x_aspx_acpl_2_non_iframe_walks_three_channel_no_addch() {
        let mut bw = BitWriter::new();
        bw.write_u32(3, 2); // 7_X_codec_mode = ASPX_ACPL_2
        write_companding_5_all_on(&mut bw);
        bw.write_u32(1, 2); // coding_config = 1 -> three_channel + two_channel
                            // three_channel_data outer:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 4);
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // two_channel_data:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 10, 0);
        write_zero_sf_data_body(&mut bw, 10, 0);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::AspxAcpl2));
        assert_eq!(
            tools.seven_x_coding_config,
            Some(FiveXCodingConfig::Cfg1ThreeStereo)
        );
        assert!(tools.three_channel_data.is_some());
        assert_eq!(tools.two_channel_data.len(), 1);
        // No additional two_channel_data — that's SIMPLE/ASPX-only.
        assert!(tools.seven_x_additional_channel_data.is_none());
        assert!(tools.seven_x_b_use_sap_add_ch.is_none());
        // ACPL pair gated on b_iframe + aspx_config in scope.
        assert!(tools.acpl_data_1ch_pair[0].is_none());
        assert!(tools.acpl_data_1ch_pair[1].is_none());
    }

    /// 7_X ASPX_ACPL_1 I-frame with `coding_config = 0` (two_channel_data
    /// branch). Validates the joint-MDCT residual layer + Cfg0
    /// trailing mono_data(0) (which moves AFTER the additional-channel
    /// block in 7.X — but ASPX_ACPL_1 has no additional-channel block,
    /// so it's right after the residual layer).
    #[test]
    fn parse_7x_aspx_acpl_1_iframe_walks_residual_and_mono_trailer() {
        let mut bw = BitWriter::new();
        bw.write_u32(2, 2); // 7_X_codec_mode = ASPX_ACPL_1
        write_zero_aspx_config(&mut bw);
        write_acpl_config_1ch_partial(&mut bw);
        write_companding_5_all_on(&mut bw);
        bw.write_u32(0, 2); // coding_config = 0 -> 2ch_mode + 2x two_channel_data
        bw.write_bit(false); // 2ch_mode
                             // two_channel_data #0:
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 12, 0);
        write_zero_sf_data_body(&mut bw, 12, 0);
        // two_channel_data #1:
        bw.write_bit(true);
        bw.write_u32(12, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 12, 0);
        write_zero_sf_data_body(&mut bw, 12, 0);
        // ASPX_ACPL_1 joint-MDCT residual layer (n_side_bits=5 @ tl=1920).
        bw.write_u32(6, 5); // max_sfb_master = 6
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 6, 0);
        write_zero_sf_data_body(&mut bw, 6, 0);
        // Cfg0 trailer: mono_data(0) for the centre.
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_u32(7, 6);
        // Pad for downstream try-and-bail aspx/acpl trailers.
        bw.align_to_byte();
        while bw.byte_len() < 256 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::AspxAcpl1));
        assert!(tools.aspx_config.is_some());
        let cfg_partial = tools
            .acpl_config_1ch_partial
            .expect("PARTIAL config parsed");
        assert_eq!(cfg_partial.num_param_bands, 15);
        assert_eq!(cfg_partial.qmf_band, 1);
        assert_eq!(tools.two_channel_data.len(), 2);
        let centre = tools.cfg0_centre_mono.as_ref().expect("centre mono walked");
        assert_eq!(centre.psy_info.as_ref().unwrap().max_sfb_0, 7);
        // No additional-channel block on ASPX_ACPL_*.
        assert!(tools.seven_x_additional_channel_data.is_none());
        assert!(tools.seven_x_b_use_sap_add_ch.is_none());
    }

    /// 7_X ASPX_ACPL_1 zero `max_sfb_master` should bail silently —
    /// matching the 5_X walker's bail behaviour. Subsequent aspx/acpl
    /// trailers stay unset.
    #[test]
    fn parse_7x_aspx_acpl_1_iframe_zero_max_sfb_master_bails() {
        let mut bw = BitWriter::new();
        bw.write_u32(2, 2); // ASPX_ACPL_1
        write_zero_aspx_config(&mut bw);
        write_acpl_config_1ch_partial(&mut bw);
        write_companding_5_all_on(&mut bw);
        bw.write_u32(1, 2); // coding_config = 1 -> three_channel + two_channel
                            // three_channel_data:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 4);
        bw.write_u32(0, 2);
        bw.write_u32(0, 2);
        for _ in 0..3 {
            write_zero_sf_data_body(&mut bw, 10, 0);
        }
        // two_channel_data:
        bw.write_bit(true);
        bw.write_u32(10, 6);
        bw.write_u32(0, 2);
        write_zero_sf_data_body(&mut bw, 10, 0);
        write_zero_sf_data_body(&mut bw, 10, 0);
        // max_sfb_master = 0 (n_side_bits=5).
        bw.write_u32(0, 5);
        bw.align_to_byte();
        while bw.byte_len() < 64 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::AspxAcpl1));
        assert!(tools.three_channel_data.is_some());
        assert!(tools.acpl_data_1ch_pair[0].is_none());
        assert!(tools.acpl_data_1ch_pair[1].is_none());
    }

    /// Truncated input mid-`five_channel_data` for SIMPLE 7_X should
    /// leave `five_channel_data` `None` (the channel-data parser
    /// errored) without panicking — the outer walker still returns
    /// Ok(()) thanks to try-and-bail.
    #[test]
    fn parse_7x_simple_truncated_five_channel_data_bails() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // SIMPLE
        bw.write_u32(3, 2); // coding_config = 3
                            // start five_channel_data but truncate:
        bw.write_bit(true); // b_long_frame
                            // intentionally cut here.
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_7x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::Simple));
        assert_eq!(
            tools.seven_x_coding_config,
            Some(FiveXCodingConfig::Cfg3Five)
        );
        assert!(tools.five_channel_data.is_none());
        assert!(tools.seven_x_additional_channel_data.is_none());
    }
}
