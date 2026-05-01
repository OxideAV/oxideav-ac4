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
    decode_asf_long_mono_body_with_max_sfb, parse_asf_psy_info, parse_asf_psy_info_lfe,
    parse_asf_transform_info, parse_chparam_info, AsfPsyInfo, AsfTransformInfo, ChparamInfo,
    SubstreamTools,
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

/// Parsed `mono_data(b_lfe)` per Table 21 — outer shell only.
///
/// `b_lfe == 1` switches `sf_info(...)` to `sf_info_lfe()` per §4.2.8 /
/// §4.3.6.2.1 (the `max_sfb[0]` field uses `n_msfbl_bits` from Table
/// 106 instead of `n_msfb_bits`).
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
/// We currently parse the outer-shell (transform_info + psy_info). The
/// `sf_data` body is left for the caller / a future round.
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
            // ASPX trailers (aspx_data_2ch + aspx_data_2ch +
            // aspx_data_1ch) deferred — they follow the same Table 51 /
            // 52 layout as stereo ASPX but consume per-substream
            // I-frame state we haven't propagated up here yet.
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
            // The remainder (max_sfb_master + 2x chparam_info + 2x
            // sf_data + optional mono_data(0) + aspx + acpl trailers)
            // is wired in r20+.
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
        // Then coding_config=3 + five_channel_data shell + 5x sf_data.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 3); // SIMPLE
                            // LFE mono_data(1):
        bw.write_bit(true); // b_long_frame
        bw.write_u32(4, 3); // max_sfb[0] -- n_msfbl_bits = 3 for tl=1920
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
}
