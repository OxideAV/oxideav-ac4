//! Outer §4.2.14.1 `metadata()` walker — TS 103 190-1 V1.4.1 Table 66.
//!
//! `metadata(b_iframe)` carries:
//!
//! 1. `basic_metadata(channel_mode)` — §4.2.14.2 Table 67
//! 2. `extended_metadata(channel_mode, b_associated, b_dialog)` — §4.2.14.4
//!    Table 69
//! 3. `tools_metadata_size_value` (7 bits) + optional
//!    `variable_bits(3) << 7` extension via the `b_more_bits` flag —
//!    announces the bit-size of the DRC + DE payload that follows.
//! 4. `drc_frame(b_iframe)` — §4.2.14.5 (handed off to
//!    [`crate::drc::parse_drc_frame`])
//! 5. `dialog_enhancement(b_iframe)` — §4.2.14.11 (handed off to
//!    [`crate::de::parse_dialog_enhancement`])
//! 6. `if (b_emdf_payloads_substream)` — `emdf_payloads_substream()`
//!
//! The A-CPL parameter payload itself does **not** live inside
//! `metadata()` per Table 66 — A-CPL data is carried by `audio_data()`
//! through `acpl_data_1ch()` / `acpl_data_2ch()` (already wired in
//! [`crate::asf::walk_ac4_substream`]). The `metadata()` walker is
//! therefore the home for DRC / DE only on the metadata side.
//!
//! Per §4.3.12.1.1 `tools_metadata_size` is a hint of the size in bits
//! of the DRC + DE payload. After dispatching the two parsers we
//! reconcile the consumed bit count against this size and skip any
//! trailing reserved bits, providing forward compatibility against
//! future extensions inside the size envelope.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::de::{parse_dialog_enhancement, write_dialog_enhancement, DeConfig, DialogEnhancement};
use crate::drc::{
    nr_drc_channels, nr_drc_subframes, parse_drc_frame, write_drc_frame, DrcChannelInfo, DrcConfig,
    DrcFrame,
};
use crate::emdf::{parse_emdf_payloads_substream, EmdfPayloadsSubstream};
use crate::toc::variable_bits;

// ---------------------------------------------------------------------
// channel_mode helpers — §4.3.3.4.1 Table 85
// ---------------------------------------------------------------------

/// Numeric `channel_mode` value (post-prefix decoding by
/// [`crate::toc::decode_channel_mode`]). The §4.3.12.2 table refers to
/// the symbolic names — these helpers map them.
pub mod channel_mode {
    pub const MONO: u32 = 0;
    pub const STEREO: u32 = 1;
    /// `3.0` — Centre + L/R.
    pub const C_LR: u32 = 2;
    /// `5.0`.
    pub const FIVE_0: u32 = 3;
    /// `5.1`.
    pub const FIVE_1: u32 = 4;
    /// `7.x` family extends from index 5 upwards.
    pub const SEVEN_X_FIRST: u32 = 5;
    pub const SEVEN_X_LAST: u32 = 10;
    /// `7.0.4` / `7.1.4` etc. (>= 11 covers the immersive layouts).
    pub const SEVEN_X_FOUR_FIRST: u32 = 11;
}

/// Indicates whether the channel layout has a centre channel (per
/// `channel_mode_contains_c()`).
fn channel_mode_contains_c(channel_mode: u32) -> bool {
    // Mono is itself the centre; stereo has no centre. Everything else
    // in the AC-4 listed configurations carries a centre. We treat any
    // "no-centre multi-channel" exotic as carrying a centre by default
    // for forward compatibility.
    channel_mode != channel_mode::MONO && channel_mode != channel_mode::STEREO
}

/// `channel_mode_contains_lr()` — true once stereo or richer.
fn channel_mode_contains_lr(channel_mode: u32) -> bool {
    channel_mode >= channel_mode::STEREO
}

/// `channel_mode_contains_LsRs()` — surrounds present (5.x and up).
fn channel_mode_contains_lsrs(channel_mode: u32) -> bool {
    channel_mode >= channel_mode::FIVE_0
}

/// `channel_mode_contains_LbRb()` — back surrounds (7.x family
/// including the .4 layouts).
fn channel_mode_contains_lbrb(channel_mode: u32) -> bool {
    channel_mode >= channel_mode::SEVEN_X_FIRST
}

/// `channel_mode_contains_LwRw()` — wide surrounds (only certain 7.x
/// layouts; we pick the 9.x and up immersive bucket as the conservative
/// home for it).
fn channel_mode_contains_lwrw(channel_mode: u32) -> bool {
    channel_mode >= channel_mode::SEVEN_X_FOUR_FIRST
}

/// `channel_mode_contains_TflTfr()` — top fronts (only the 7.x.4 and
/// 9.1.4 layouts).
fn channel_mode_contains_tfltfr(channel_mode: u32) -> bool {
    channel_mode >= channel_mode::SEVEN_X_FOUR_FIRST
}

/// `channel_mode_contains_Lfe()` — true when the layout has an LFE.
/// 5.1 (4), 7.1 family (6/8/10), 7.1.4 family (12/14/...).
fn channel_mode_contains_lfe(channel_mode: u32) -> bool {
    matches!(channel_mode, 4 | 6 | 8 | 10 | 12 | 14)
}

// ---------------------------------------------------------------------
// further_loudness_info — §4.2.14.3 Table 68
// ---------------------------------------------------------------------

/// Decoded `further_loudness_info()` (§4.2.14.3 Table 68). We only
/// surface the load-bearing scalar fields plus a flag mask for the
/// Booleans so callers can distinguish "absent" from "present + zero".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FurtherLoudnessInfo {
    pub loudness_version: u8,
    pub loud_prac_type: u8,
    pub dialgate_prac_type: Option<u8>,
    pub loudcorr_type: Option<bool>,
    pub loudrelgat: Option<u16>,
    pub loudspchgat: Option<u16>,
    pub loudspchgat_dialgate_prac_type: Option<u8>,
    pub loudstrm3s: Option<u16>,
    pub max_loudstrm3s: Option<u16>,
    pub truepk: Option<u16>,
    pub max_truepk: Option<u16>,
    pub prgmbndy: Option<u32>,
    pub b_end_or_start: Option<bool>,
    pub prgmbndy_offset: Option<u16>,
    pub lra: Option<u16>,
    pub lra_prac_type: Option<u8>,
    pub loudmntry: Option<u16>,
    pub max_loudmntry: Option<u16>,
}

fn parse_further_loudness_info(br: &mut BitReader<'_>) -> Result<FurtherLoudnessInfo> {
    let mut v = FurtherLoudnessInfo::default();
    let mut lv = br.read_u32(2)? as u8;
    if lv == 3 {
        let extra = br.read_u32(4)? as u8;
        // Per Table 68: loudness_version += extended_loudness_version.
        lv = lv.wrapping_add(extra);
    }
    v.loudness_version = lv;
    v.loud_prac_type = br.read_u32(4)? as u8;
    if v.loud_prac_type != 0 {
        if br.read_bit()? {
            v.dialgate_prac_type = Some(br.read_u32(3)? as u8);
        }
        v.loudcorr_type = Some(br.read_bit()?);
    }
    if br.read_bit()? {
        v.loudrelgat = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.loudspchgat = Some(br.read_u32(11)? as u16);
        v.loudspchgat_dialgate_prac_type = Some(br.read_u32(3)? as u8);
    }
    if br.read_bit()? {
        v.loudstrm3s = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.max_loudstrm3s = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.truepk = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.max_truepk = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        // prgmbndy: read unary-bit count then shift.
        let mut prgmbndy: u32 = 1;
        loop {
            let bit = br.read_u32(1)?;
            if bit == 1 {
                break;
            }
            prgmbndy <<= 1;
            if prgmbndy > (1u32 << 30) {
                return Err(Error::invalid(
                    "ac4: further_loudness_info prgmbndy unary overflow",
                ));
            }
        }
        v.prgmbndy = Some(prgmbndy);
        v.b_end_or_start = Some(br.read_bit()?);
        if br.read_bit()? {
            v.prgmbndy_offset = Some(br.read_u32(11)? as u16);
        }
    }
    if br.read_bit()? {
        v.lra = Some(br.read_u32(10)? as u16);
        v.lra_prac_type = Some(br.read_u32(3)? as u8);
    }
    if br.read_bit()? {
        v.loudmntry = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.max_loudmntry = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        // b_extension: e_bits_size (5b) [+ variable_bits(4) if 31]
        // then e_bits_size bits of opaque extension payload.
        let mut sz = br.read_u32(5)?;
        if sz == 31 {
            sz = sz.checked_add(variable_bits(br, 4)?).ok_or_else(|| {
                Error::invalid("ac4: further_loudness_info extension size overflow")
            })?;
        }
        // Skip opaque extension bits.
        skip_n_bits(br, sz)?;
    }
    Ok(v)
}

// ---------------------------------------------------------------------
// basic_metadata — §4.2.14.2 Table 67
// ---------------------------------------------------------------------

/// Decoded `basic_metadata(channel_mode)` (§4.2.14.2 Table 67).
///
/// Only the load-bearing scalars are surfaced; `more_basic_metadata`
/// indicates whether the post-`b_more_basic_metadata` block was
/// transmitted (the gated optional fields all flow into the same
/// `Option`-bearing fields below).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BasicMetadata {
    pub dialnorm_bits: u8,
    pub more_basic_metadata: bool,
    pub further_loudness_info: Option<FurtherLoudnessInfo>,
    // Stereo-mode previous downmix info (channel_mode == stereo branch):
    pub pre_dmixtyp_2ch: Option<u8>,
    pub phase90_info_2ch: Option<u8>,
    // Multi-channel branch (channel_mode > stereo):
    pub loro_centre_mixgain: Option<u8>,
    pub loro_surround_mixgain: Option<u8>,
    pub loro_dmx_loud_corr: Option<u8>,
    pub ltrt_centre_mixgain: Option<u8>,
    pub ltrt_surround_mixgain: Option<u8>,
    pub ltrt_dmx_loud_corr: Option<u8>,
    pub lfe_mixgain: Option<u8>,
    pub preferred_dmx_method: Option<u8>,
    pub pre_dmixtyp_5ch: Option<u8>,
    pub pre_upmixtyp_5ch: Option<u8>,
    pub pre_upmixtyp_3_4: Option<u8>,
    pub pre_upmixtyp_3_2_2: Option<u8>,
    pub phase90_info_mc: Option<u8>,
    pub b_surround_attenuation_known: Option<bool>,
    pub b_lfe_attenuation_known: Option<bool>,
    pub dc_block_on: Option<bool>,
}

/// Walk `basic_metadata(channel_mode)` per §4.2.14.2 Table 67.
pub fn parse_basic_metadata(br: &mut BitReader<'_>, channel_mode: u32) -> Result<BasicMetadata> {
    let mut v = BasicMetadata {
        dialnorm_bits: br.read_u32(7)? as u8,
        ..Default::default()
    };
    let b_more_basic_metadata = br.read_bit()?;
    v.more_basic_metadata = b_more_basic_metadata;
    if !b_more_basic_metadata {
        return Ok(v);
    }
    if br.read_bit()? {
        // b_further_loudness_info.
        v.further_loudness_info = Some(parse_further_loudness_info(br)?);
    }
    if channel_mode == channel_mode::STEREO {
        if br.read_bit()? {
            // b_prev_dmx_info.
            v.pre_dmixtyp_2ch = Some(br.read_u32(3)? as u8);
            v.phase90_info_2ch = Some(br.read_u32(2)? as u8);
        }
    } else if channel_mode > channel_mode::STEREO {
        if br.read_bit()? {
            // b_dmx_coeff.
            v.loro_centre_mixgain = Some(br.read_u32(3)? as u8);
            v.loro_surround_mixgain = Some(br.read_u32(3)? as u8);
            if br.read_bit()? {
                // b_loro_dmx_loud_corr.
                v.loro_dmx_loud_corr = Some(br.read_u32(5)? as u8);
            }
            if br.read_bit()? {
                // b_ltrt_mixinfo.
                v.ltrt_centre_mixgain = Some(br.read_u32(3)? as u8);
                v.ltrt_surround_mixgain = Some(br.read_u32(3)? as u8);
            }
            if br.read_bit()? {
                // b_ltrt_dmx_loud_corr.
                v.ltrt_dmx_loud_corr = Some(br.read_u32(5)? as u8);
            }
            if channel_mode_contains_lfe(channel_mode) && br.read_bit()? {
                // b_lfe_mixinfo.
                v.lfe_mixgain = Some(br.read_u32(5)? as u8);
            }
            v.preferred_dmx_method = Some(br.read_u32(2)? as u8);
        }
        // 5.x branch.
        if matches!(channel_mode, channel_mode::FIVE_0 | channel_mode::FIVE_1) {
            if br.read_bit()? {
                v.pre_dmixtyp_5ch = Some(br.read_u32(3)? as u8);
            }
            if br.read_bit()? {
                v.pre_upmixtyp_5ch = Some(br.read_u32(4)? as u8);
            }
        }
        if (channel_mode::SEVEN_X_FIRST..=channel_mode::SEVEN_X_LAST).contains(&channel_mode)
            && br.read_bit()?
        {
            if channel_mode <= 6 {
                v.pre_upmixtyp_3_4 = Some(br.read_u32(2)? as u8);
            } else if (9..=10).contains(&channel_mode) {
                v.pre_upmixtyp_3_2_2 = Some(br.read_u32(1)? as u8);
            }
        }
        v.phase90_info_mc = Some(br.read_u32(2)? as u8);
        v.b_surround_attenuation_known = Some(br.read_bit()?);
        v.b_lfe_attenuation_known = Some(br.read_bit()?);
    }
    if br.read_bit()? {
        // b_dc_blocking.
        v.dc_block_on = Some(br.read_bit()?);
    }
    Ok(v)
}

/// Write `further_loudness_info()` per §4.2.14.3 Table 68, the inverse of
/// [`parse_further_loudness_info`].
///
/// The decoder discards the optional opaque `b_extension` payload, so
/// this writer always emits `b_extension = 0` (extension absent) — a
/// spec-valid canonical form that re-decodes to the same struct. All
/// other `Option` fields are gated by their presence flags; an
/// inconsistency (e.g. `loudspchgat` set without its paired
/// `loudspchgat_dialgate_prac_type`) raises `Error::invalid`.
pub fn write_further_loudness_info(
    bw: &mut oxideav_core::bits::BitWriter,
    v: &FurtherLoudnessInfo,
) -> Result<()> {
    // loudness_version: 2-bit base, escape via 4-bit extension when the
    // value is >= 3. The decoder forms lv = 3 + extended (wrapping), so a
    // value >= 3 encodes as base 3 + (value - 3) in the 4-bit field.
    if v.loudness_version >= 3 {
        let extra = v.loudness_version - 3;
        if extra > 0x0F {
            return Err(Error::invalid(
                "ac4: loudness_version extension exceeds 4 bits",
            ));
        }
        bw.write_u32(3, 2);
        bw.write_u32(extra as u32, 4);
    } else {
        bw.write_u32(v.loudness_version as u32, 2);
    }

    bw.write_u32(v.loud_prac_type as u32, 4);
    if v.loud_prac_type != 0 {
        match v.dialgate_prac_type {
            Some(d) => {
                bw.write_bit(true);
                bw.write_u32(d as u32, 3);
            }
            None => bw.write_bit(false),
        }
        let corr = v.loudcorr_type.ok_or_else(|| {
            Error::invalid("ac4: loudcorr_type required when loud_prac_type != 0")
        })?;
        bw.write_bit(corr);
    }

    write_opt_u(bw, v.loudrelgat.map(|x| x as u32), 11);

    match (v.loudspchgat, v.loudspchgat_dialgate_prac_type) {
        (Some(g), Some(d)) => {
            bw.write_bit(true);
            bw.write_u32(g as u32, 11);
            bw.write_u32(d as u32, 3);
        }
        (None, None) => bw.write_bit(false),
        _ => {
            return Err(Error::invalid(
                "ac4: loudspchgat and its dialgate_prac_type must both be present or absent",
            ))
        }
    }

    write_opt_u(bw, v.loudstrm3s.map(|x| x as u32), 11);
    write_opt_u(bw, v.max_loudstrm3s.map(|x| x as u32), 11);
    write_opt_u(bw, v.truepk.map(|x| x as u32), 11);
    write_opt_u(bw, v.max_truepk.map(|x| x as u32), 11);

    match v.prgmbndy {
        Some(p) => {
            if !p.is_power_of_two() {
                return Err(Error::invalid(
                    "ac4: prgmbndy must be a power of two (unary code)",
                ));
            }
            bw.write_bit(true);
            // prgmbndy = 1 << num_zeros; emit num_zeros '0' bits then '1'.
            let num_zeros = p.trailing_zeros();
            for _ in 0..num_zeros {
                bw.write_bit(false);
            }
            bw.write_bit(true);
            let eos = v
                .b_end_or_start
                .ok_or_else(|| Error::invalid("ac4: b_end_or_start required with prgmbndy"))?;
            bw.write_bit(eos);
            write_opt_u(bw, v.prgmbndy_offset.map(|x| x as u32), 11);
        }
        None => bw.write_bit(false),
    }

    match (v.lra, v.lra_prac_type) {
        (Some(lra), Some(pt)) => {
            bw.write_bit(true);
            bw.write_u32(lra as u32, 10);
            bw.write_u32(pt as u32, 3);
        }
        (None, None) => bw.write_bit(false),
        _ => {
            return Err(Error::invalid(
                "ac4: lra and lra_prac_type must both be present or absent",
            ))
        }
    }

    write_opt_u(bw, v.loudmntry.map(|x| x as u32), 11);
    write_opt_u(bw, v.max_loudmntry.map(|x| x as u32), 11);

    // b_extension: always 0 (canonical, extension-free form).
    bw.write_bit(false);
    Ok(())
}

/// Write `basic_metadata(channel_mode)` per §4.2.14.2 Table 67, the
/// inverse of [`parse_basic_metadata`].
///
/// The `channel_mode` MUST match the one used to decode `v`, since the
/// syntax branches on it. `Option` fields that contradict the branch the
/// channel_mode selects are ignored; required-but-absent fields within an
/// active branch raise `Error::invalid`.
pub fn write_basic_metadata(
    bw: &mut oxideav_core::bits::BitWriter,
    v: &BasicMetadata,
    channel_mode: u32,
) -> Result<()> {
    bw.write_u32(v.dialnorm_bits as u32, 7);
    bw.write_bit(v.more_basic_metadata);
    if !v.more_basic_metadata {
        return Ok(());
    }

    match &v.further_loudness_info {
        Some(fli) => {
            bw.write_bit(true);
            write_further_loudness_info(bw, fli)?;
        }
        None => bw.write_bit(false),
    }

    if channel_mode == channel_mode::STEREO {
        match (v.pre_dmixtyp_2ch, v.phase90_info_2ch) {
            (Some(d), Some(p)) => {
                bw.write_bit(true); // b_prev_dmx_info
                bw.write_u32(d as u32, 3);
                bw.write_u32(p as u32, 2);
            }
            (None, None) => bw.write_bit(false),
            _ => {
                return Err(Error::invalid(
                    "ac4: pre_dmixtyp_2ch and phase90_info_2ch must both be present or absent",
                ))
            }
        }
    } else if channel_mode > channel_mode::STEREO {
        // b_dmx_coeff block.
        if v.loro_centre_mixgain.is_some() {
            bw.write_bit(true);
            bw.write_u32(
                v.loro_centre_mixgain
                    .ok_or_else(|| Error::invalid("ac4: loro_centre_mixgain"))?
                    as u32,
                3,
            );
            bw.write_u32(
                v.loro_surround_mixgain
                    .ok_or_else(|| Error::invalid("ac4: loro_surround_mixgain"))?
                    as u32,
                3,
            );
            write_opt_u(bw, v.loro_dmx_loud_corr.map(|x| x as u32), 5);
            match (v.ltrt_centre_mixgain, v.ltrt_surround_mixgain) {
                (Some(c), Some(s)) => {
                    bw.write_bit(true);
                    bw.write_u32(c as u32, 3);
                    bw.write_u32(s as u32, 3);
                }
                (None, None) => bw.write_bit(false),
                _ => {
                    return Err(Error::invalid(
                        "ac4: ltrt centre/surround mixgain must both be present or absent",
                    ))
                }
            }
            write_opt_u(bw, v.ltrt_dmx_loud_corr.map(|x| x as u32), 5);
            if channel_mode_contains_lfe(channel_mode) {
                write_opt_u(bw, v.lfe_mixgain.map(|x| x as u32), 5);
            }
            bw.write_u32(
                v.preferred_dmx_method
                    .ok_or_else(|| Error::invalid("ac4: preferred_dmx_method"))?
                    as u32,
                2,
            );
        } else {
            bw.write_bit(false);
        }

        if matches!(channel_mode, channel_mode::FIVE_0 | channel_mode::FIVE_1) {
            write_opt_u(bw, v.pre_dmixtyp_5ch.map(|x| x as u32), 3);
            write_opt_u(bw, v.pre_upmixtyp_5ch.map(|x| x as u32), 4);
        }

        if (channel_mode::SEVEN_X_FIRST..=channel_mode::SEVEN_X_LAST).contains(&channel_mode) {
            // b_upmixtyp_7ch — present iff one of the gated fields is set.
            let has = v.pre_upmixtyp_3_4.is_some() || v.pre_upmixtyp_3_2_2.is_some();
            bw.write_bit(has);
            if has {
                if channel_mode <= 6 {
                    bw.write_u32(
                        v.pre_upmixtyp_3_4
                            .ok_or_else(|| Error::invalid("ac4: pre_upmixtyp_3_4"))?
                            as u32,
                        2,
                    );
                } else if (9..=10).contains(&channel_mode) {
                    bw.write_u32(
                        v.pre_upmixtyp_3_2_2
                            .ok_or_else(|| Error::invalid("ac4: pre_upmixtyp_3_2_2"))?
                            as u32,
                        1,
                    );
                }
            }
        }

        bw.write_u32(
            v.phase90_info_mc
                .ok_or_else(|| Error::invalid("ac4: phase90_info_mc required in mc branch"))?
                as u32,
            2,
        );
        bw.write_bit(
            v.b_surround_attenuation_known
                .ok_or_else(|| Error::invalid("ac4: b_surround_attenuation_known"))?,
        );
        bw.write_bit(
            v.b_lfe_attenuation_known
                .ok_or_else(|| Error::invalid("ac4: b_lfe_attenuation_known"))?,
        );
    }

    match v.dc_block_on {
        Some(on) => {
            bw.write_bit(true);
            bw.write_bit(on);
        }
        None => bw.write_bit(false),
    }
    Ok(())
}

/// Helper: write a 1-bit presence flag then the `n`-bit value when
/// `value.is_some()`.
fn write_opt_u(bw: &mut oxideav_core::bits::BitWriter, value: Option<u32>, n: u32) {
    match value {
        Some(x) => {
            bw.write_bit(true);
            bw.write_u32(x, n);
        }
        None => bw.write_bit(false),
    }
}

// ---------------------------------------------------------------------
// extended_metadata — §4.2.14.4 Table 69
// ---------------------------------------------------------------------

/// Decoded `extended_metadata(channel_mode, b_associated, b_dialog)`
/// (§4.2.14.4 Table 69).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtendedMetadata {
    pub scale_main: Option<u8>,
    pub scale_main_centre: Option<u8>,
    pub scale_main_front: Option<u8>,
    pub pan_associated: Option<u8>,
    pub dialog_max_gain: Option<u8>,
    pub pan_dialog: Option<u8>,
    pub pan_dialog_pair: Option<(u8, u8)>,
    pub pan_signal_selector: Option<u8>,
    pub b_c_active: Option<bool>,
    pub b_c_has_dialog: Option<bool>,
    pub b_l_active: Option<bool>,
    pub b_l_has_dialog: Option<bool>,
    pub b_r_active: Option<bool>,
    pub b_r_has_dialog: Option<bool>,
    pub b_ls_active: Option<bool>,
    pub b_rs_active: Option<bool>,
    pub b_lb_active: Option<bool>,
    pub b_rb_active: Option<bool>,
    pub b_lw_active: Option<bool>,
    pub b_rw_active: Option<bool>,
    pub b_vhl_active: Option<bool>,
    pub b_vhr_active: Option<bool>,
    pub b_lfe_active: Option<bool>,
    /// `b_channels_classifier` (§4.2.14.4) — true when the channels
    /// classifier block was transmitted. Tracked explicitly because for
    /// layouts with no classifiable channels (e.g. mono) the block emits
    /// no inner fields, making its presence otherwise inferrable only
    /// from this flag.
    pub b_channels_classifier: bool,
    pub event_probability: Option<u8>,
}

/// Walk `extended_metadata(channel_mode, b_associated, b_dialog)` per
/// §4.2.14.4 Table 69.
pub fn parse_extended_metadata(
    br: &mut BitReader<'_>,
    channel_mode: u32,
    b_associated: bool,
    b_dialog: bool,
) -> Result<ExtendedMetadata> {
    let mut v = ExtendedMetadata::default();
    if b_associated {
        if br.read_bit()? {
            v.scale_main = Some(br.read_u32(8)? as u8);
        }
        if br.read_bit()? {
            v.scale_main_centre = Some(br.read_u32(8)? as u8);
        }
        if br.read_bit()? {
            v.scale_main_front = Some(br.read_u32(8)? as u8);
        }
        if channel_mode == channel_mode::MONO {
            v.pan_associated = Some(br.read_u32(8)? as u8);
        }
    }
    if b_dialog {
        if br.read_bit()? {
            v.dialog_max_gain = Some(br.read_u32(2)? as u8);
        }
        if br.read_bit()? {
            if channel_mode == channel_mode::MONO {
                v.pan_dialog = Some(br.read_u32(8)? as u8);
            } else {
                let a = br.read_u32(8)? as u8;
                let b = br.read_u32(8)? as u8;
                v.pan_dialog_pair = Some((a, b));
                v.pan_signal_selector = Some(br.read_u32(2)? as u8);
            }
        }
    }
    if br.read_bit()? {
        // b_channels_classifier.
        v.b_channels_classifier = true;
        if channel_mode_contains_c(channel_mode) && br.read_bit()? {
            // b_c_active.
            v.b_c_active = Some(true);
            v.b_c_has_dialog = Some(br.read_bit()?);
        }
        if channel_mode_contains_lr(channel_mode) {
            if br.read_bit()? {
                v.b_l_active = Some(true);
                v.b_l_has_dialog = Some(br.read_bit()?);
            }
            if br.read_bit()? {
                v.b_r_active = Some(true);
                v.b_r_has_dialog = Some(br.read_bit()?);
            }
        }
        if channel_mode_contains_lsrs(channel_mode) {
            v.b_ls_active = Some(br.read_bit()?);
            v.b_rs_active = Some(br.read_bit()?);
        }
        if channel_mode_contains_lbrb(channel_mode) {
            v.b_lb_active = Some(br.read_bit()?);
            v.b_rb_active = Some(br.read_bit()?);
        }
        if channel_mode_contains_lwrw(channel_mode) {
            v.b_lw_active = Some(br.read_bit()?);
            v.b_rw_active = Some(br.read_bit()?);
        }
        if channel_mode_contains_tfltfr(channel_mode) {
            v.b_vhl_active = Some(br.read_bit()?);
            v.b_vhr_active = Some(br.read_bit()?);
        }
        if channel_mode_contains_lfe(channel_mode) {
            v.b_lfe_active = Some(br.read_bit()?);
        }
    }
    if br.read_bit()? {
        // b_event_probability.
        v.event_probability = Some(br.read_u32(4)? as u8);
    }
    Ok(v)
}

/// Write `extended_metadata(channel_mode, b_associated, b_dialog)` per
/// §4.2.14.4 Table 69, the inverse of [`parse_extended_metadata`].
///
/// `channel_mode`, `b_associated`, and `b_dialog` MUST match the values
/// used to decode `v`. The channels-classifier block is emitted iff
/// `v.b_channels_classifier` is set; within it, the c/l/r channels use
/// the active-flag / has-dialog nesting while the Ls/Rs and higher pairs
/// carry an unconditional active bit per their presence in the layout.
pub fn write_extended_metadata(
    bw: &mut oxideav_core::bits::BitWriter,
    v: &ExtendedMetadata,
    channel_mode: u32,
    b_associated: bool,
    b_dialog: bool,
) -> Result<()> {
    if b_associated {
        write_opt_u(bw, v.scale_main.map(|x| x as u32), 8);
        write_opt_u(bw, v.scale_main_centre.map(|x| x as u32), 8);
        write_opt_u(bw, v.scale_main_front.map(|x| x as u32), 8);
        if channel_mode == channel_mode::MONO {
            bw.write_u32(
                v.pan_associated
                    .ok_or_else(|| Error::invalid("ac4: pan_associated required for mono assoc"))?
                    as u32,
                8,
            );
        }
    }

    if b_dialog {
        write_opt_u(bw, v.dialog_max_gain.map(|x| x as u32), 2);
        // b_pan_dialog gate: present iff pan_dialog (mono) or the pair is
        // set.
        let has_pan = v.pan_dialog.is_some() || v.pan_dialog_pair.is_some();
        bw.write_bit(has_pan);
        if has_pan {
            if channel_mode == channel_mode::MONO {
                bw.write_u32(
                    v.pan_dialog
                        .ok_or_else(|| Error::invalid("ac4: pan_dialog required for mono dialog"))?
                        as u32,
                    8,
                );
            } else {
                let (a, b) = v
                    .pan_dialog_pair
                    .ok_or_else(|| Error::invalid("ac4: pan_dialog_pair required for mc dialog"))?;
                bw.write_u32(a as u32, 8);
                bw.write_u32(b as u32, 8);
                bw.write_u32(
                    v.pan_signal_selector
                        .ok_or_else(|| Error::invalid("ac4: pan_signal_selector required"))?
                        as u32,
                    2,
                );
            }
        }
    }

    bw.write_bit(v.b_channels_classifier);
    if v.b_channels_classifier {
        if channel_mode_contains_c(channel_mode) {
            match v.b_c_active {
                Some(true) => {
                    bw.write_bit(true);
                    bw.write_bit(
                        v.b_c_has_dialog
                            .ok_or_else(|| Error::invalid("ac4: b_c_has_dialog required"))?,
                    );
                }
                _ => bw.write_bit(false),
            }
        }
        if channel_mode_contains_lr(channel_mode) {
            match v.b_l_active {
                Some(true) => {
                    bw.write_bit(true);
                    bw.write_bit(
                        v.b_l_has_dialog
                            .ok_or_else(|| Error::invalid("ac4: b_l_has_dialog required"))?,
                    );
                }
                _ => bw.write_bit(false),
            }
            match v.b_r_active {
                Some(true) => {
                    bw.write_bit(true);
                    bw.write_bit(
                        v.b_r_has_dialog
                            .ok_or_else(|| Error::invalid("ac4: b_r_has_dialog required"))?,
                    );
                }
                _ => bw.write_bit(false),
            }
        }
        if channel_mode_contains_lsrs(channel_mode) {
            bw.write_bit(
                v.b_ls_active
                    .ok_or_else(|| Error::invalid("ac4: b_ls_active"))?,
            );
            bw.write_bit(
                v.b_rs_active
                    .ok_or_else(|| Error::invalid("ac4: b_rs_active"))?,
            );
        }
        if channel_mode_contains_lbrb(channel_mode) {
            bw.write_bit(
                v.b_lb_active
                    .ok_or_else(|| Error::invalid("ac4: b_lb_active"))?,
            );
            bw.write_bit(
                v.b_rb_active
                    .ok_or_else(|| Error::invalid("ac4: b_rb_active"))?,
            );
        }
        if channel_mode_contains_lwrw(channel_mode) {
            bw.write_bit(
                v.b_lw_active
                    .ok_or_else(|| Error::invalid("ac4: b_lw_active"))?,
            );
            bw.write_bit(
                v.b_rw_active
                    .ok_or_else(|| Error::invalid("ac4: b_rw_active"))?,
            );
        }
        if channel_mode_contains_tfltfr(channel_mode) {
            bw.write_bit(
                v.b_vhl_active
                    .ok_or_else(|| Error::invalid("ac4: b_vhl_active"))?,
            );
            bw.write_bit(
                v.b_vhr_active
                    .ok_or_else(|| Error::invalid("ac4: b_vhr_active"))?,
            );
        }
        if channel_mode_contains_lfe(channel_mode) {
            bw.write_bit(
                v.b_lfe_active
                    .ok_or_else(|| Error::invalid("ac4: b_lfe_active"))?,
            );
        }
    }

    write_opt_u(bw, v.event_probability.map(|x| x as u32), 4);
    Ok(())
}

// ---------------------------------------------------------------------
// metadata() — §4.2.14.1 Table 66 outer walker
// ---------------------------------------------------------------------

/// Per-substream parser state that survives across frames so non-I
/// frames can decode against the previous I-frame's configuration.
#[derive(Debug, Clone, Default)]
pub struct MetadataState {
    /// Last successfully parsed `drc_config()` (carried across non-I
    /// frames per §4.2.14.5 Table 70).
    pub prev_drc_config: Option<DrcConfig>,
    /// Last successfully parsed `de_config()` (carried across non-I
    /// frames per §4.2.14.11 Table 76).
    pub prev_de_config: Option<DeConfig>,
}

/// Decoded `metadata(b_iframe)` (§4.2.14.1 Table 66).
#[derive(Debug, Clone)]
pub struct Metadata {
    /// `basic_metadata(channel_mode)` (always present per Table 66).
    pub basic: BasicMetadata,
    /// `extended_metadata(channel_mode, b_associated, b_dialog)`
    /// (always present per Table 66 — `b_associated` / `b_dialog` are
    /// caller-supplied substream-level flags).
    pub extended: ExtendedMetadata,
    /// Resolved `tools_metadata_size` in bits — sum of the 7-bit base
    /// field and the optional `variable_bits(3) << 7` extension.
    pub tools_metadata_size: u32,
    /// Decoded `drc_frame()`.
    pub drc: DrcFrame,
    /// Decoded `dialog_enhancement()`.
    pub dialog_enhancement: DialogEnhancement,
    /// `b_emdf_payloads_substream` flag — true when the optional
    /// `emdf_payloads_substream()` element is present in the bitstream.
    pub emdf_payloads_substream_present: bool,
    /// Decoded `emdf_payloads_substream()` (Table 18) when the
    /// `b_emdf_payloads_substream` flag was set; otherwise `None`.
    /// Per §4.3.15.1 the terminator payload is consumed but not
    /// surfaced as an entry.
    pub emdf_payloads_substream: Option<EmdfPayloadsSubstream>,
    /// Number of bits left over inside the `tools_metadata_size`
    /// envelope after DRC + DE were consumed; the walker skips them
    /// to maintain bit-alignment for the next substream element.
    pub tools_metadata_trailing_bits: u32,
}

/// Caller context for the outer walker.
#[derive(Debug, Clone, Copy)]
pub struct MetadataContext {
    /// `channel_mode` from `ac4_substream_info()` (passed to
    /// `basic_metadata` / `extended_metadata`).
    pub channel_mode: u32,
    /// `b_iframe` for the substream (drives `drc_config` /
    /// `de_config` presence).
    pub b_iframe: bool,
    /// `b_associated` flag (substream-level — `extended_metadata`).
    pub b_associated: bool,
    /// `b_dialog` flag (substream-level — `extended_metadata`).
    pub b_dialog: bool,
    /// AC-4 frame_length in samples — drives `nr_drc_subframes`
    /// (§4.3.13.7.2 Table 169).
    pub frame_length: u32,
}

/// Walk `metadata(b_iframe)` per §4.2.14.1 Table 66.
///
/// `prev_state` carries forward `drc_config()` and `de_config()` from
/// earlier I-frames — both are required when their respective parsers
/// see `b_*_present == 1` on a non-I-frame. Pass a fresh
/// [`MetadataState`] on the very first call.
pub fn parse_metadata(
    br: &mut BitReader<'_>,
    ctx: MetadataContext,
    prev_state: &MetadataState,
) -> Result<Metadata> {
    let basic = parse_basic_metadata(br, ctx.channel_mode)?;
    let extended = parse_extended_metadata(br, ctx.channel_mode, ctx.b_associated, ctx.b_dialog)?;
    let mut tools_size = br.read_u32(7)?;
    if br.read_bit()? {
        // b_more_bits → tools_metadata_size += variable_bits(3) << 7.
        let extra = variable_bits(br, 3)?;
        tools_size = tools_size
            .checked_add(
                extra
                    .checked_shl(7)
                    .ok_or_else(|| Error::invalid("ac4: tools_metadata_size shift overflow"))?,
            )
            .ok_or_else(|| Error::invalid("ac4: tools_metadata_size overflow"))?;
    }

    // Snapshot bit position so we can reconcile against tools_size
    // afterwards.
    let tools_start_bit = br.bit_position();

    let drc_chan_info = DrcChannelInfo::new(
        nr_drc_channels(ctx.channel_mode),
        nr_drc_subframes(ctx.frame_length).unwrap_or(1),
    );
    let drc = parse_drc_frame(
        br,
        ctx.b_iframe,
        drc_chan_info,
        prev_state.prev_drc_config.as_ref(),
    )?;
    let dialog_enhancement = parse_dialog_enhancement(br, ctx.b_iframe, prev_state.prev_de_config)?;

    // Reconcile against tools_metadata_size: skip any trailing bits the
    // bitstream announced but the parsers didn't consume (forward
    // compat). Underrun is a hard error — that means the parser read
    // *past* the announced envelope.
    let consumed = (br.bit_position() - tools_start_bit) as u32;
    if consumed > tools_size {
        return Err(Error::invalid(
            "ac4: drc_frame + dialog_enhancement consumed more than tools_metadata_size bits",
        ));
    }
    let trailing = tools_size - consumed;
    skip_n_bits(br, trailing)?;

    let emdf_payloads_substream_present = br.read_bit()?;
    let emdf_payloads_substream = if emdf_payloads_substream_present {
        // §4.2.4.4 emdf_payloads_substream() — Table 18.
        Some(parse_emdf_payloads_substream(br)?)
    } else {
        None
    };

    Ok(Metadata {
        basic,
        extended,
        tools_metadata_size: tools_size,
        drc,
        dialog_enhancement,
        emdf_payloads_substream_present,
        emdf_payloads_substream,
        tools_metadata_trailing_bits: trailing,
    })
}

/// Write `metadata(b_iframe)` per §4.2.14.1 Table 66, the inverse of
/// [`parse_metadata`].
///
/// Reproduces the full element order: `basic_metadata`,
/// `extended_metadata`, the 7-bit `tools_metadata_size` base plus its
/// `b_more_bits` / `variable_bits(3) << 7` extension, then `drc_frame` +
/// `dialog_enhancement` inside the announced size envelope (re-emitting
/// the recorded `tools_metadata_trailing_bits` of zero filler), and
/// finally the `b_emdf_payloads_substream` flag + optional
/// `emdf_payloads_substream()`.
///
/// `ctx` MUST match the context used to decode `meta`. The recorded
/// `tools_metadata_size` is honoured as-is; an inconsistency between the
/// recorded trailing-bit count and the actual DRC+DE size raises
/// `Error::invalid`.
pub fn write_metadata(
    bw: &mut oxideav_core::bits::BitWriter,
    meta: &Metadata,
    ctx: MetadataContext,
) -> Result<()> {
    write_basic_metadata(bw, &meta.basic, ctx.channel_mode)?;
    write_extended_metadata(
        bw,
        &meta.extended,
        ctx.channel_mode,
        ctx.b_associated,
        ctx.b_dialog,
    )?;

    // tools_metadata_size: 7-bit base + optional variable_bits(3) << 7.
    let tools_size = meta.tools_metadata_size;
    let base = tools_size & 0x7F;
    let high = tools_size >> 7;
    bw.write_u32(base, 7);
    if high > 0 {
        bw.write_bit(true);
        crate::toc::write_variable_bits(bw, 3, high);
    } else {
        bw.write_bit(false);
    }

    let tools_start = bw.bit_position();

    let drc_chan_info = DrcChannelInfo::new(
        nr_drc_channels(ctx.channel_mode),
        nr_drc_subframes(ctx.frame_length).unwrap_or(1),
    );
    write_drc_frame(bw, &meta.drc, ctx.b_iframe, drc_chan_info)?;
    write_dialog_enhancement(bw, &meta.dialog_enhancement, ctx.b_iframe)?;

    let consumed = (bw.bit_position() - tools_start) as u32;
    if consumed + meta.tools_metadata_trailing_bits != tools_size {
        return Err(Error::invalid(
            "ac4: write_metadata DRC+DE size + trailing bits != tools_metadata_size",
        ));
    }
    // Emit the trailing reserved (zero) bits to reach the envelope size.
    let mut trailing = meta.tools_metadata_trailing_bits;
    while trailing >= 32 {
        bw.write_u32(0, 32);
        trailing -= 32;
    }
    if trailing > 0 {
        bw.write_u32(0, trailing);
    }

    bw.write_bit(meta.emdf_payloads_substream_present);
    if meta.emdf_payloads_substream_present {
        let sub = meta.emdf_payloads_substream.as_ref().ok_or_else(|| {
            Error::invalid("ac4: emdf_payloads_substream_present but substream is None")
        })?;
        crate::emdf::write_emdf_payloads_substream(bw, sub)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

/// Skip `n` bits from the reader — the BitReader API tops out at u32
/// chunks, so split into 16-bit slices for safety.
fn skip_n_bits(br: &mut BitReader<'_>, mut n: u32) -> Result<()> {
    while n >= 16 {
        br.skip(16)?;
        n -= 16;
    }
    if n > 0 {
        br.skip(n)?;
    }
    Ok(())
}

// =====================================================================
// sus_ver = 1 metadata variants — ETSI TS 103 190-2 §6.2.7
// =====================================================================

/// Decoded `further_loudness_info(sus_ver = 1, b_presentation_ldn)`
/// (TS 103 190-2 §6.2.7.3). The version/practice header and the
/// `prgmbndy` block only exist when `b_presentation_ldn`; otherwise a
/// bare `b_loudcorr_dialgate` replaces the header. The `sus_ver >= 1`
/// tail carries `rtll_comp`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FurtherLoudnessInfoV2 {
    /// The `b_presentation_ldn` context this element was coded with.
    pub b_presentation_ldn: bool,
    /// Bare `b_loudcorr_dialgate` (headerless form,
    /// `b_presentation_ldn == false`).
    pub b_loudcorr_dialgate: Option<bool>,
    /// The shared field set (header + measures + prgmbndy where the
    /// context admits them).
    pub info: FurtherLoudnessInfo,
    /// 8-bit `rtll_comp` (`sus_ver >= 1` tail, `b_rtllcomp` gate).
    pub rtll_comp: Option<u8>,
}

/// Walk `further_loudness_info(sus_ver = 1, b_presentation_ldn)` per
/// TS 103 190-2 §6.2.7.3.
pub fn parse_further_loudness_info_v2(
    br: &mut BitReader<'_>,
    b_presentation_ldn: bool,
) -> Result<FurtherLoudnessInfoV2> {
    let mut v = FurtherLoudnessInfoV2 {
        b_presentation_ldn,
        ..Default::default()
    };
    if b_presentation_ldn {
        let mut lv = br.read_u32(2)? as u8;
        if lv == 3 {
            lv = lv.wrapping_add(br.read_u32(4)? as u8);
        }
        v.info.loudness_version = lv;
        v.info.loud_prac_type = br.read_u32(4)? as u8;
        if v.info.loud_prac_type != 0 {
            if br.read_bit()? {
                v.info.dialgate_prac_type = Some(br.read_u32(3)? as u8);
            }
            v.info.loudcorr_type = Some(br.read_bit()?);
        }
    } else {
        v.b_loudcorr_dialgate = Some(br.read_bit()?);
    }
    if br.read_bit()? {
        v.info.loudrelgat = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.info.loudspchgat = Some(br.read_u32(11)? as u16);
        v.info.loudspchgat_dialgate_prac_type = Some(br.read_u32(3)? as u8);
    }
    if br.read_bit()? {
        v.info.loudstrm3s = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.info.max_loudstrm3s = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.info.truepk = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.info.max_truepk = Some(br.read_u32(11)? as u16);
    }
    if b_presentation_ldn && br.read_bit()? {
        let mut prgmbndy: u32 = 1;
        loop {
            if br.read_u32(1)? == 1 {
                break;
            }
            prgmbndy <<= 1;
            if prgmbndy > (1u32 << 30) {
                return Err(Error::invalid("ac4: v2 prgmbndy unary overflow"));
            }
        }
        v.info.prgmbndy = Some(prgmbndy);
        v.info.b_end_or_start = Some(br.read_bit()?);
        if br.read_bit()? {
            v.info.prgmbndy_offset = Some(br.read_u32(11)? as u16);
        }
    }
    if br.read_bit()? {
        v.info.lra = Some(br.read_u32(10)? as u16);
        v.info.lra_prac_type = Some(br.read_u32(3)? as u8);
    }
    if br.read_bit()? {
        v.info.loudmntry = Some(br.read_u32(11)? as u16);
    }
    if br.read_bit()? {
        v.info.max_loudmntry = Some(br.read_u32(11)? as u16);
    }
    // sus_ver >= 1 tail: b_rtllcomp + b_extension.
    if br.read_bit()? {
        v.rtll_comp = Some(br.read_u32(8)? as u8);
    }
    if br.read_bit()? {
        let mut sz = br.read_u32(5)?;
        if sz == 31 {
            sz = sz
                .checked_add(variable_bits(br, 4)?)
                .ok_or_else(|| Error::invalid("ac4: v2 loudness extension size overflow"))?;
        }
        skip_n_bits(br, sz)?;
    }
    Ok(v)
}

/// Write `further_loudness_info(sus_ver = 1, b_presentation_ldn)` —
/// exact inverse of [`parse_further_loudness_info_v2`] (canonical
/// extension-free form).
pub fn write_further_loudness_info_v2(
    bw: &mut oxideav_core::bits::BitWriter,
    v: &FurtherLoudnessInfoV2,
) -> Result<()> {
    if v.b_presentation_ldn {
        if v.info.loudness_version >= 3 {
            let extra = v.info.loudness_version - 3;
            if extra > 0x0F {
                return Err(Error::invalid("ac4: loudness_version extension too big"));
            }
            bw.write_u32(3, 2);
            bw.write_u32(extra as u32, 4);
        } else {
            bw.write_u32(v.info.loudness_version as u32, 2);
        }
        bw.write_u32(v.info.loud_prac_type as u32, 4);
        if v.info.loud_prac_type != 0 {
            match v.info.dialgate_prac_type {
                Some(d) => {
                    bw.write_bit(true);
                    bw.write_u32(d as u32, 3);
                }
                None => bw.write_bit(false),
            }
            let corr = v
                .info
                .loudcorr_type
                .ok_or_else(|| Error::invalid("ac4: loudcorr_type required"))?;
            bw.write_bit(corr);
        }
    } else {
        let dg = v
            .b_loudcorr_dialgate
            .ok_or_else(|| Error::invalid("ac4: headerless form needs b_loudcorr_dialgate"))?;
        bw.write_bit(dg);
    }
    write_opt_u(bw, v.info.loudrelgat.map(|x| x as u32), 11);
    match (v.info.loudspchgat, v.info.loudspchgat_dialgate_prac_type) {
        (Some(g), Some(d)) => {
            bw.write_bit(true);
            bw.write_u32(g as u32, 11);
            bw.write_u32(d as u32, 3);
        }
        (None, None) => bw.write_bit(false),
        _ => return Err(Error::invalid("ac4: loudspchgat pair mismatch")),
    }
    write_opt_u(bw, v.info.loudstrm3s.map(|x| x as u32), 11);
    write_opt_u(bw, v.info.max_loudstrm3s.map(|x| x as u32), 11);
    write_opt_u(bw, v.info.truepk.map(|x| x as u32), 11);
    write_opt_u(bw, v.info.max_truepk.map(|x| x as u32), 11);
    if v.b_presentation_ldn {
        match v.info.prgmbndy {
            Some(p) => {
                if !p.is_power_of_two() {
                    return Err(Error::invalid("ac4: prgmbndy must be a power of two"));
                }
                bw.write_bit(true);
                for _ in 0..p.trailing_zeros() {
                    bw.write_bit(false);
                }
                bw.write_bit(true);
                let eos = v
                    .info
                    .b_end_or_start
                    .ok_or_else(|| Error::invalid("ac4: b_end_or_start required"))?;
                bw.write_bit(eos);
                write_opt_u(bw, v.info.prgmbndy_offset.map(|x| x as u32), 11);
            }
            None => bw.write_bit(false),
        }
    } else if v.info.prgmbndy.is_some() {
        return Err(Error::invalid(
            "ac4: prgmbndy only valid with b_presentation_ldn",
        ));
    }
    match (v.info.lra, v.info.lra_prac_type) {
        (Some(lra), Some(pt)) => {
            bw.write_bit(true);
            bw.write_u32(lra as u32, 10);
            bw.write_u32(pt as u32, 3);
        }
        (None, None) => bw.write_bit(false),
        _ => return Err(Error::invalid("ac4: lra pair mismatch")),
    }
    write_opt_u(bw, v.info.loudmntry.map(|x| x as u32), 11);
    write_opt_u(bw, v.info.max_loudmntry.map(|x| x as u32), 11);
    match v.rtll_comp {
        Some(r) => {
            bw.write_bit(true);
            bw.write_u32(r as u32, 8);
        }
        None => bw.write_bit(false),
    }
    bw.write_bit(false); // b_extension: canonical extension-free form.
    Ok(())
}

/// Decoded `basic_metadata(channel_mode, sus_ver = 1)`
/// (TS 103 190-2 §6.2.7.2). The `sus_ver = 1` form has no `dialnorm`
/// and replaces `b_further_loudness_info` with the substream-loudness
/// pair; the `b_stereo_dmx_coeff` downmix block is `sus_ver = 0` only.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BasicMetadataV2 {
    /// `b_more_basic_metadata`.
    pub more_basic_metadata: bool,
    /// 8-bit `substream_loudness_bits` (`b_substream_loudness_info`).
    pub substream_loudness_bits: Option<u8>,
    /// `further_loudness_info(1, 0)`
    /// (`b_further_substream_loudness_info`).
    pub further_loudness_info: Option<FurtherLoudnessInfoV2>,
    /// Stereo branch: `pre_dmixtyp_2ch` + `phase90_info_2ch`.
    pub pre_dmixtyp_2ch: Option<u8>,
    /// See [`BasicMetadataV2::pre_dmixtyp_2ch`].
    pub phase90_info_2ch: Option<u8>,
    /// 5.X branch.
    pub pre_dmixtyp_5ch: Option<u8>,
    /// 5.X branch.
    pub pre_upmixtyp_5ch: Option<u8>,
    /// 7.X (3/4/0) branch.
    pub pre_upmixtyp_3_4: Option<u8>,
    /// 7.X (3/2/2) branch.
    pub pre_upmixtyp_3_2_2: Option<u8>,
    /// Multichannel tail.
    pub phase90_info_mc: Option<u8>,
    /// Multichannel tail.
    pub b_surround_attenuation_known: Option<bool>,
    /// Multichannel tail.
    pub b_lfe_attenuation_known: Option<bool>,
    /// `b_dc_blocking` payload.
    pub dc_block_on: Option<bool>,
}

/// Walk `basic_metadata(channel_mode, sus_ver = 1)` per TS 103 190-2
/// §6.2.7.2.
pub fn parse_basic_metadata_v2(
    br: &mut BitReader<'_>,
    channel_mode: u32,
) -> Result<BasicMetadataV2> {
    let mut v = BasicMetadataV2 {
        more_basic_metadata: br.read_bit()?,
        ..Default::default()
    };
    if !v.more_basic_metadata {
        return Ok(v);
    }
    if br.read_bit()? {
        // b_substream_loudness_info.
        v.substream_loudness_bits = Some(br.read_u32(8)? as u8);
        if br.read_bit()? {
            // b_further_substream_loudness_info.
            v.further_loudness_info = Some(parse_further_loudness_info_v2(br, false)?);
        }
    }
    if channel_mode == channel_mode::STEREO {
        if br.read_bit()? {
            v.pre_dmixtyp_2ch = Some(br.read_u32(3)? as u8);
            v.phase90_info_2ch = Some(br.read_u32(2)? as u8);
        }
    } else if channel_mode > channel_mode::STEREO {
        // (The b_stereo_dmx_coeff block is sus_ver == 0 only.)
        if matches!(channel_mode, channel_mode::FIVE_0 | channel_mode::FIVE_1) {
            if br.read_bit()? {
                v.pre_dmixtyp_5ch = Some(br.read_u32(3)? as u8);
            }
            if br.read_bit()? {
                v.pre_upmixtyp_5ch = Some(br.read_u32(4)? as u8);
            }
        }
        if (channel_mode::SEVEN_X_FIRST..=channel_mode::SEVEN_X_LAST).contains(&channel_mode)
            && br.read_bit()?
        {
            if channel_mode <= 6 {
                v.pre_upmixtyp_3_4 = Some(br.read_u32(2)? as u8);
            } else if (9..=10).contains(&channel_mode) {
                v.pre_upmixtyp_3_2_2 = Some(br.read_u32(1)? as u8);
            }
        }
        v.phase90_info_mc = Some(br.read_u32(2)? as u8);
        v.b_surround_attenuation_known = Some(br.read_bit()?);
        v.b_lfe_attenuation_known = Some(br.read_bit()?);
    }
    if br.read_bit()? {
        v.dc_block_on = Some(br.read_bit()?);
    }
    Ok(v)
}

/// Write `basic_metadata(channel_mode, sus_ver = 1)` — exact inverse of
/// [`parse_basic_metadata_v2`].
pub fn write_basic_metadata_v2(
    bw: &mut oxideav_core::bits::BitWriter,
    v: &BasicMetadataV2,
    channel_mode: u32,
) -> Result<()> {
    bw.write_bit(v.more_basic_metadata);
    if !v.more_basic_metadata {
        return Ok(());
    }
    match v.substream_loudness_bits {
        Some(bits) => {
            bw.write_bit(true);
            bw.write_u32(bits as u32, 8);
            match &v.further_loudness_info {
                Some(f) => {
                    bw.write_bit(true);
                    write_further_loudness_info_v2(bw, f)?;
                }
                None => bw.write_bit(false),
            }
        }
        None => {
            if v.further_loudness_info.is_some() {
                return Err(Error::invalid(
                    "ac4: v2 further loudness needs substream_loudness_bits",
                ));
            }
            bw.write_bit(false);
        }
    }
    if channel_mode == channel_mode::STEREO {
        match (v.pre_dmixtyp_2ch, v.phase90_info_2ch) {
            (Some(d), Some(p)) => {
                bw.write_bit(true);
                bw.write_u32(d as u32, 3);
                bw.write_u32(p as u32, 2);
            }
            (None, None) => bw.write_bit(false),
            _ => return Err(Error::invalid("ac4: stereo prev-dmx pair mismatch")),
        }
    } else if channel_mode > channel_mode::STEREO {
        if matches!(channel_mode, channel_mode::FIVE_0 | channel_mode::FIVE_1) {
            match v.pre_dmixtyp_5ch {
                Some(d) => {
                    bw.write_bit(true);
                    bw.write_u32(d as u32, 3);
                }
                None => bw.write_bit(false),
            }
            match v.pre_upmixtyp_5ch {
                Some(u) => {
                    bw.write_bit(true);
                    bw.write_u32(u as u32, 4);
                }
                None => bw.write_bit(false),
            }
        }
        if (channel_mode::SEVEN_X_FIRST..=channel_mode::SEVEN_X_LAST).contains(&channel_mode) {
            if channel_mode <= 6 {
                match v.pre_upmixtyp_3_4 {
                    Some(u) => {
                        bw.write_bit(true);
                        bw.write_u32(u as u32, 2);
                    }
                    None => bw.write_bit(false),
                }
            } else if (9..=10).contains(&channel_mode) {
                match v.pre_upmixtyp_3_2_2 {
                    Some(u) => {
                        bw.write_bit(true);
                        bw.write_u32(u as u32, 1);
                    }
                    None => bw.write_bit(false),
                }
            } else {
                bw.write_bit(false);
            }
        }
        let p90 = v
            .phase90_info_mc
            .ok_or_else(|| Error::invalid("ac4: phase90_info_mc required"))?;
        bw.write_u32(p90 as u32, 2);
        bw.write_bit(v.b_surround_attenuation_known.unwrap_or(false));
        bw.write_bit(v.b_lfe_attenuation_known.unwrap_or(false));
    }
    match v.dc_block_on {
        Some(on) => {
            bw.write_bit(true);
            bw.write_bit(on);
        }
        None => bw.write_bit(false),
    }
    Ok(())
}

/// Decoded `metadata(b_alternative, b_ajoc, b_iframe, sus_ver = 1)`
/// (TS 103 190-2 §6.2.7.1) — the object-substream metadata form: no
/// `drc_frame()`, `extended_metadata` carries its own `b_dialog` flag,
/// and the `b_alternative && !b_ajoc` OAMD branch is not taken on
/// A-JOC substreams.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataV2 {
    /// `basic_metadata(channel_mode, 1)`.
    pub basic: BasicMetadataV2,
    /// The `b_dialog` flag read at the head of
    /// `extended_metadata(…, sus_ver = 1)`.
    pub b_dialog: bool,
    /// `extended_metadata` payload.
    pub extended: ExtendedMetadata,
    /// Announced tools envelope size in bits.
    pub tools_metadata_size: u32,
    /// `dialog_enhancement(b_iframe)`.
    pub dialog_enhancement: DialogEnhancement,
    /// Announced-but-unparsed trailing tools bits.
    pub tools_metadata_trailing_bits: u32,
    /// `b_emdf_payloads_substream` + payload.
    pub emdf_payloads_substream: Option<EmdfPayloadsSubstream>,
}

/// Walk `metadata(b_alternative = 0 | b_ajoc = 1, b_iframe,
/// sus_ver = 1)` per TS 103 190-2 §6.2.7.1 for an A-JOC object
/// substream. (The `b_alternative && !b_ajoc` inline
/// `oamd_dyndata_single()` branch needs the direct-coded object
/// context and is rejected here.)
pub fn parse_metadata_v2(
    br: &mut BitReader<'_>,
    channel_mode: u32,
    b_iframe: bool,
    b_alternative_and_not_ajoc: bool,
    prev_de: Option<crate::de::DeConfig>,
) -> Result<MetadataV2> {
    if b_alternative_and_not_ajoc {
        return Err(Error::unsupported(
            "ac4: metadata v2 inline OAMD for direct-coded alternative substreams",
        ));
    }
    let basic = parse_basic_metadata_v2(br, channel_mode)?;
    // extended_metadata(…, sus_ver = 1): leading b_dialog flag, no
    // b_associated block.
    let b_dialog = br.read_bit()?;
    let extended = parse_extended_metadata(br, channel_mode, false, b_dialog)?;
    let mut tools_size = br.read_u32(7)?;
    if br.read_bit()? {
        let extra = variable_bits(br, 3)?;
        tools_size = tools_size
            .checked_add(
                extra
                    .checked_shl(7)
                    .ok_or_else(|| Error::invalid("ac4: tools size shift overflow"))?,
            )
            .ok_or_else(|| Error::invalid("ac4: tools size overflow"))?;
    }
    let tools_start = br.bit_position();
    // sus_ver == 1: no drc_frame().
    let dialog_enhancement = parse_dialog_enhancement(br, b_iframe, prev_de)?;
    let consumed = (br.bit_position() - tools_start) as u32;
    if consumed > tools_size {
        return Err(Error::invalid(
            "ac4: v2 dialog_enhancement consumed more than tools_metadata_size",
        ));
    }
    let trailing = tools_size - consumed;
    skip_n_bits(br, trailing)?;
    let emdf_payloads_substream = if br.read_bit()? {
        Some(parse_emdf_payloads_substream(br)?)
    } else {
        None
    };
    Ok(MetadataV2 {
        basic,
        b_dialog,
        extended,
        tools_metadata_size: tools_size,
        dialog_enhancement,
        tools_metadata_trailing_bits: trailing,
        emdf_payloads_substream,
    })
}

/// Write `metadata(…, sus_ver = 1)` — exact inverse of
/// [`parse_metadata_v2`].
pub fn write_metadata_v2(
    bw: &mut oxideav_core::bits::BitWriter,
    meta: &MetadataV2,
    channel_mode: u32,
    b_iframe: bool,
) -> Result<()> {
    write_basic_metadata_v2(bw, &meta.basic, channel_mode)?;
    bw.write_bit(meta.b_dialog);
    write_extended_metadata(bw, &meta.extended, channel_mode, false, meta.b_dialog)?;
    if meta.tools_metadata_size < (1 << 7) {
        bw.write_u32(meta.tools_metadata_size, 7);
        bw.write_bit(false);
    } else {
        bw.write_u32(meta.tools_metadata_size & 0x7F, 7);
        bw.write_bit(true);
        crate::toc::write_variable_bits(bw, 3, meta.tools_metadata_size >> 7);
    }
    let start = bw.bit_position();
    write_dialog_enhancement(bw, &meta.dialog_enhancement, b_iframe)?;
    let used = (bw.bit_position() - start) as u32;
    if used + meta.tools_metadata_trailing_bits != meta.tools_metadata_size {
        return Err(Error::invalid(
            "ac4: v2 tools envelope inconsistent with recorded trailing bits",
        ));
    }
    for _ in 0..meta.tools_metadata_trailing_bits {
        bw.write_bit(false);
    }
    match &meta.emdf_payloads_substream {
        Some(e) => {
            bw.write_bit(true);
            crate::emdf::write_emdf_payloads_substream(bw, e)?;
        }
        None => bw.write_bit(false),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    // ------------------------------------------------------------------
    // channel_mode_contains_* helpers — sanity check Table 85 mapping.
    // ------------------------------------------------------------------

    #[test]
    fn channel_mode_lfe_mapping() {
        // 5.1 = 4, 7.1 family = 6/8/10, 7.1.4 family = 12/14.
        for &m in &[4u32, 6, 8, 10, 12, 14] {
            assert!(
                channel_mode_contains_lfe(m),
                "channel_mode {m} should have LFE"
            );
        }
        for &m in &[0u32, 1, 2, 3, 5, 7, 9, 11] {
            assert!(
                !channel_mode_contains_lfe(m),
                "channel_mode {m} should not have LFE"
            );
        }
    }

    #[test]
    fn channel_mode_centre_mapping() {
        // Stereo never has explicit centre; mono is the centre itself
        // (returns false in the helper).
        assert!(!channel_mode_contains_c(0));
        assert!(!channel_mode_contains_c(1));
        // 3.0 / 5.0 / 5.1 / 7.x all carry an explicit C.
        for &m in &[2u32, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
            assert!(
                channel_mode_contains_c(m),
                "channel_mode {m} should carry C"
            );
        }
    }

    // ------------------------------------------------------------------
    // basic_metadata round-trip.
    // ------------------------------------------------------------------

    #[test]
    fn basic_metadata_minimal_no_extra_bits() {
        // dialnorm_bits = 0x40 (= -16 dBFS), b_more_basic_metadata = 0.
        let mut bw = BitWriter::new();
        bw.write_u32(0x40, 7);
        bw.write_bit(false);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let bm = parse_basic_metadata(&mut br, channel_mode::MONO).unwrap();
        assert_eq!(bm.dialnorm_bits, 0x40);
        assert!(!bm.more_basic_metadata);
        assert!(bm.further_loudness_info.is_none());
        assert!(bm.dc_block_on.is_none());
    }

    #[test]
    fn basic_metadata_stereo_with_prev_dmx_info() {
        // dialnorm_bits = 0, b_more = 1, b_further = 0,
        // stereo branch: b_prev_dmx_info = 1 -> pre_dmixtyp_2ch = 5,
        // phase90_info_2ch = 2; b_dc_blocking = 0.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm_bits
        bw.write_bit(true); // b_more_basic_metadata
        bw.write_bit(false); // b_further_loudness_info
        bw.write_bit(true); // b_prev_dmx_info
        bw.write_u32(5, 3); // pre_dmixtyp_2ch
        bw.write_u32(2, 2); // phase90_info_2ch
        bw.write_bit(false); // b_dc_blocking
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let bm = parse_basic_metadata(&mut br, channel_mode::STEREO).unwrap();
        assert_eq!(bm.dialnorm_bits, 0);
        assert!(bm.more_basic_metadata);
        assert_eq!(bm.pre_dmixtyp_2ch, Some(5));
        assert_eq!(bm.phase90_info_2ch, Some(2));
        assert!(bm.dc_block_on.is_none());
    }

    #[test]
    fn basic_metadata_5_1_with_dmx_coeff_and_lfe() {
        // 5.1 (channel_mode=4): b_more=1, b_further=0, b_dmx_coeff=1
        // -> loro_centre=2, loro_surround=4, b_loro_dmx_loud_corr=0,
        //    b_ltrt_mixinfo=0, b_ltrt_dmx_loud_corr=0, b_lfe_mixinfo=1
        //    -> lfe_mixgain=15, preferred_dmx_method=1.
        // 5.x branch: b_predmxtyp_5ch=0, b_preupmixtyp_5ch=0.
        // (channel_mode==4 is not in 5..=10 so 7.x block is skipped.)
        // phase90_info_mc=1, b_surround=1, b_lfe=0, b_dc_blocking=0.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(true); // more
        bw.write_bit(false); // b_further
        bw.write_bit(true); // b_dmx_coeff
        bw.write_u32(2, 3); // loro_centre
        bw.write_u32(4, 3); // loro_surround
        bw.write_bit(false); // b_loro_dmx_loud_corr
        bw.write_bit(false); // b_ltrt_mixinfo
        bw.write_bit(false); // b_ltrt_dmx_loud_corr
        bw.write_bit(true); // b_lfe_mixinfo
        bw.write_u32(15, 5); // lfe_mixgain
        bw.write_u32(1, 2); // preferred_dmx_method
        bw.write_bit(false); // b_predmxtyp_5ch
        bw.write_bit(false); // b_preupmixtyp_5ch
        bw.write_u32(1, 2); // phase90_info_mc
        bw.write_bit(true); // b_surround_attenuation_known
        bw.write_bit(false); // b_lfe_attenuation_known
        bw.write_bit(false); // b_dc_blocking
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let bm = parse_basic_metadata(&mut br, channel_mode::FIVE_1).unwrap();
        assert_eq!(bm.loro_centre_mixgain, Some(2));
        assert_eq!(bm.loro_surround_mixgain, Some(4));
        assert!(bm.loro_dmx_loud_corr.is_none());
        assert!(bm.ltrt_centre_mixgain.is_none());
        assert_eq!(bm.lfe_mixgain, Some(15));
        assert_eq!(bm.preferred_dmx_method, Some(1));
        assert_eq!(bm.phase90_info_mc, Some(1));
        assert_eq!(bm.b_surround_attenuation_known, Some(true));
        assert_eq!(bm.b_lfe_attenuation_known, Some(false));
    }

    // ------------------------------------------------------------------
    // extended_metadata round-trip.
    // ------------------------------------------------------------------

    #[test]
    fn extended_metadata_no_flags() {
        // b_associated=0, b_dialog=0; only b_channels_classifier=0,
        // b_event_probability=0 → 2 zero bits.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let em = parse_extended_metadata(&mut br, channel_mode::STEREO, false, false).unwrap();
        assert!(em.event_probability.is_none());
        assert!(em.b_l_active.is_none());
    }

    #[test]
    fn extended_metadata_associated_mono_with_pan() {
        // b_associated=1: b_scale_main=1 -> 0xAB, b_scale_main_centre=0,
        // b_scale_main_front=0, channel_mode==mono → pan_associated=0xCD.
        // b_dialog=0.
        // b_channels_classifier=0, b_event_probability=0.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_scale_main
        bw.write_u32(0xAB, 8);
        bw.write_bit(false); // b_scale_main_centre
        bw.write_bit(false); // b_scale_main_front
        bw.write_u32(0xCD, 8); // pan_associated
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let em = parse_extended_metadata(&mut br, channel_mode::MONO, true, false).unwrap();
        assert_eq!(em.scale_main, Some(0xAB));
        assert_eq!(em.pan_associated, Some(0xCD));
        assert!(em.dialog_max_gain.is_none());
    }

    // ------------------------------------------------------------------
    // Outer metadata() walker.
    // ------------------------------------------------------------------

    fn write_minimal_drc_absent(bw: &mut BitWriter) {
        // b_drc_present = 0.
        bw.write_bit(false);
    }

    fn write_minimal_de_absent(bw: &mut BitWriter) {
        // b_de_data_present = 0.
        bw.write_bit(false);
    }

    // -----------------------------------------------------------------
    // sus_ver = 1 (TS 103 190-2 §6.2.7) variants
    // -----------------------------------------------------------------

    fn v2_round_trips(raw: &[u8], channel_mode: u32, b_iframe: bool) -> MetadataV2 {
        let mut br = BitReader::new(raw);
        let meta = parse_metadata_v2(&mut br, channel_mode, b_iframe, false, None).unwrap();
        let mut bw = BitWriter::new();
        write_metadata_v2(&mut bw, &meta, channel_mode, b_iframe).unwrap();
        bw.align_to_byte();
        let rebuilt = bw.finish();
        let mut br2 = BitReader::new(&rebuilt);
        let meta2 = parse_metadata_v2(&mut br2, channel_mode, b_iframe, false, None).unwrap();
        assert_eq!(meta, meta2);
        meta
    }

    #[test]
    fn metadata_v2_minimal_mono_round_trips() {
        // basic_metadata v2: NO dialnorm — b_more = 0 is the first bit.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_more_basic_metadata
        bw.write_bit(false); // extended: b_dialog = 0
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(1, 7); // tools_metadata_size = 1 (DE only — no DRC)
        bw.write_bit(false); // b_more_bits
        write_minimal_de_absent(&mut bw);
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let raw = bw.finish();
        let meta = v2_round_trips(&raw, channel_mode::MONO, true);
        assert!(!meta.basic.more_basic_metadata);
        assert!(!meta.b_dialog);
        assert_eq!(meta.tools_metadata_size, 1);
    }

    #[test]
    fn metadata_v2_substream_loudness_and_rtll_round_trips() {
        // Stereo, b_more = 1 with the substream-loudness pair and the
        // headerless further_loudness_info(1, 0) carrying rtll_comp.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_more_basic_metadata
        bw.write_bit(true); // b_substream_loudness_info
        bw.write_u32(0xA5, 8); // substream_loudness_bits
        bw.write_bit(true); // b_further_substream_loudness_info
                            // further_loudness_info(sus_ver = 1, b_presentation_ldn = 0):
        bw.write_bit(true); // b_loudcorr_dialgate (headerless form)
        bw.write_bit(false); // b_loudrelgat
        bw.write_bit(false); // b_loudspchgat
        bw.write_bit(true); // b_loudstrm3s
        bw.write_u32(321, 11);
        bw.write_bit(false); // b_max_loudstrm3s
        bw.write_bit(false); // b_truepk
        bw.write_bit(false); // b_max_truepk
                             // (no prgmbndy block — b_presentation_ldn = 0)
        bw.write_bit(false); // b_lra
        bw.write_bit(false); // b_loudmntry
        bw.write_bit(false); // b_max_loudmntry
        bw.write_bit(true); // b_rtllcomp
        bw.write_u32(77, 8); // rtll_comp
        bw.write_bit(false); // b_extension
                             // stereo branch: b_prev_dmx_info = 1.
        bw.write_bit(true);
        bw.write_u32(5, 3); // pre_dmixtyp_2ch
        bw.write_u32(2, 2); // phase90_info_2ch
        bw.write_bit(false); // b_dc_blocking
                             // extended_metadata v2: b_dialog = 1 with a max gain.
        bw.write_bit(true); // b_dialog
        bw.write_bit(true); // b_dialog_max_gain
        bw.write_u32(2, 2); // dialog_max_gain
        bw.write_bit(false); // b_pan_dialog_present
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(1, 7); // tools_metadata_size
        bw.write_bit(false); // b_more_bits
        write_minimal_de_absent(&mut bw);
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let raw = bw.finish();
        let meta = v2_round_trips(&raw, channel_mode::STEREO, true);
        assert_eq!(meta.basic.substream_loudness_bits, Some(0xA5));
        let fli = meta.basic.further_loudness_info.as_ref().unwrap();
        assert!(!fli.b_presentation_ldn);
        assert_eq!(fli.b_loudcorr_dialgate, Some(true));
        assert_eq!(fli.info.loudstrm3s, Some(321));
        assert_eq!(fli.rtll_comp, Some(77));
        assert_eq!(meta.basic.pre_dmixtyp_2ch, Some(5));
        assert!(meta.b_dialog);
        assert_eq!(meta.extended.dialog_max_gain, Some(2));
    }

    #[test]
    fn further_loudness_info_v2_presentation_form_round_trips() {
        // b_presentation_ldn = 1: full header + prgmbndy + rtll tail.
        let v = FurtherLoudnessInfoV2 {
            b_presentation_ldn: true,
            b_loudcorr_dialgate: None,
            info: FurtherLoudnessInfo {
                loudness_version: 5,
                loud_prac_type: 2,
                dialgate_prac_type: Some(3),
                loudcorr_type: Some(true),
                loudrelgat: Some(1000),
                prgmbndy: Some(8),
                b_end_or_start: Some(true),
                prgmbndy_offset: Some(99),
                lra: Some(500),
                lra_prac_type: Some(1),
                ..Default::default()
            },
            rtll_comp: Some(200),
        };
        let mut bw = BitWriter::new();
        write_further_loudness_info_v2(&mut bw, &v).unwrap();
        bw.write_u32(0, 7);
        let raw = bw.into_bytes();
        let mut br = BitReader::new(&raw);
        let got = parse_further_loudness_info_v2(&mut br, true).unwrap();
        assert_eq!(got, v);
    }

    #[test]
    fn basic_metadata_v2_multichannel_tail_round_trips() {
        // 5.1: no stereo-dmx block (sus_ver = 1), 5.X types + tail.
        let v = BasicMetadataV2 {
            more_basic_metadata: true,
            pre_dmixtyp_5ch: Some(4),
            pre_upmixtyp_5ch: Some(9),
            phase90_info_mc: Some(3),
            b_surround_attenuation_known: Some(true),
            b_lfe_attenuation_known: Some(false),
            dc_block_on: Some(true),
            ..Default::default()
        };
        let mut bw = BitWriter::new();
        write_basic_metadata_v2(&mut bw, &v, channel_mode::FIVE_1).unwrap();
        bw.write_u32(0, 7);
        let raw = bw.into_bytes();
        let mut br = BitReader::new(&raw);
        let got = parse_basic_metadata_v2(&mut br, channel_mode::FIVE_1).unwrap();
        assert_eq!(got, v);
    }

    #[test]
    fn metadata_walker_minimal_iframe_mono_no_payload() {
        // basic_metadata(mono): dialnorm 0x40, more=0.
        // extended_metadata(mono, b_assoc=0, b_dialog=0): only the
        // trailing channels_classifier and event_probability flags (2x 0).
        // tools_metadata_size = 2 (drc 1 bit + de 1 bit), b_more_bits=0.
        // drc_frame: b_drc_present=0; dialog_enhancement: b_de_data_present=0.
        // b_emdf_payloads_substream = 0.
        let mut bw = BitWriter::new();
        bw.write_u32(0x40, 7); // dialnorm
        bw.write_bit(false); // more_basic_metadata
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(2, 7); // tools_metadata_size_value = 2
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        let state = MetadataState::default();
        let m = parse_metadata(&mut br, ctx, &state).unwrap();
        assert_eq!(m.basic.dialnorm_bits, 0x40);
        assert_eq!(m.tools_metadata_size, 2);
        assert!(!m.drc.b_drc_present);
        assert!(!m.dialog_enhancement.data_present);
        assert!(!m.emdf_payloads_substream_present);
        assert_eq!(m.tools_metadata_trailing_bits, 0);
    }

    #[test]
    fn metadata_walker_more_bits_extension() {
        // tools_metadata_size_value = 2, b_more_bits=1 with a small
        // variable_bits(3) of 0 → +0 << 7 = +0. So tools_size stays 2.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(2, 7); // tools_metadata_size_value
        bw.write_bit(true); // b_more_bits = 1
        bw.write_u32(0, 3); // variable_bits(3) value=0
        bw.write_bit(false); // b_read_more = 0 -> done
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        let m = parse_metadata(&mut br, ctx, &MetadataState::default()).unwrap();
        // value=0 + read_more=0 → variable_bits returns 0; tools size = 2 + (0<<7) = 2.
        assert_eq!(m.tools_metadata_size, 2);
    }

    #[test]
    fn metadata_walker_trailing_bits_skipped() {
        // tools_metadata_size = 6 (2 actual + 4 forward-compat
        // reserved bits). After DRC + DE consume 2 bits, the walker
        // skips 4 zero bits and continues.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(6, 7); // tools_metadata_size_value = 6
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        // 4 trailing reserved bits inside the size envelope.
        bw.write_u32(0xF, 4);
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        let m = parse_metadata(&mut br, ctx, &MetadataState::default()).unwrap();
        assert_eq!(m.tools_metadata_trailing_bits, 4);
        assert!(!m.emdf_payloads_substream_present);
    }

    #[test]
    fn metadata_walker_dispatches_drc_with_curve_and_de() {
        // I-frame with a real DRC frame (one mode, default profile so
        // implicit curve flag, gainset absent) AND a DE config + data.
        // Mono channel: nr_drc_channels=1, frame_length=512 → subframes=2.
        // - DRC payload: b_drc_present(1) + drc_decoder_nr_modes(3)=0
        //   + drc_decoder_mode_id(3)=0 + drc_repeat_profile_flag(0)
        //   + drc_default_profile_flag(1) + drc_eac3_profile(3)=0
        //   + drc_reset_flag(1)=0 + drc_reserved(2)=0
        //   = 1+3+3+1+1+3+1+2 = 15 bits.
        // - DE payload: b_de_data_present(1) + de_method(2)=0 +
        //   de_max_gain(2)=0 + de_channel_config(3)=0
        //   = 8 bits (channel_config=0 → no further per-channel data).
        // tools_metadata_size = 15 + 8 = 23 bits.
        let mut bw = BitWriter::new();
        // basic + extended (minimal mono).
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
                             // tools_metadata_size = 23.
        bw.write_u32(23, 7);
        bw.write_bit(false); // b_more_bits
                             // DRC frame.
        bw.write_bit(true); // b_drc_present
        bw.write_u32(0, 3); // drc_decoder_nr_modes = 0 (1 mode)
        bw.write_u32(0, 3); // drc_decoder_mode_id = 0
        bw.write_bit(false); // drc_repeat_profile_flag
        bw.write_bit(true); // drc_default_profile_flag
        bw.write_u32(0, 3); // drc_eac3_profile
        bw.write_bit(false); // drc_reset_flag
        bw.write_u32(0, 2); // drc_reserved
                            // DE.
        bw.write_bit(true); // b_de_data_present
        bw.write_u32(0, 2); // de_method = 0 (ChannelIndependent)
        bw.write_u32(0, 2); // de_max_gain
        bw.write_u32(0, 3); // de_channel_config = 0 → 0 channels
                            // EMDF flag.
        bw.write_bit(false);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 512,
        };
        let m = parse_metadata(&mut br, ctx, &MetadataState::default()).unwrap();
        assert_eq!(m.tools_metadata_size, 23);
        assert!(m.drc.b_drc_present);
        let cfg = m.drc.config.as_ref().expect("drc config present");
        assert_eq!(cfg.modes.len(), 1);
        assert!(m.dialog_enhancement.data_present);
        let de_cfg = m.dialog_enhancement.config.expect("de config present");
        assert_eq!(de_cfg.channel_config, 0);
        assert_eq!(m.tools_metadata_trailing_bits, 0);
    }

    #[test]
    fn metadata_walker_chains_prev_state_for_p_frame() {
        // Build: I-frame establishes drc_config + de_config; P-frame
        // re-uses them (b_iframe=0).
        // For simplicity we simulate just the P-frame here against a
        // populated MetadataState — the I-frame test above already
        // exercises the normal initial path.
        // P-frame DRC: b_drc_present=1, drc_data():
        //   curve_present(1)=1 (because prev mode has compression_curve_flag=true),
        //   drc_reset_flag(1)=0, drc_reserved(2)=0.
        //   That's 1+1+1+2 = 5 bits.
        // P-frame DE: b_de_data_present=1, de_config_flag(1)=0 (re-use
        //   prev), then de_data: channel_config=0 → 0 channels: nothing
        //   else read.
        //   That's 1+1 = 2 bits (since I-frame condition is false).
        // tools_metadata_size = 5 + 2 = 7.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(7, 7); // tools_metadata_size = 7
        bw.write_bit(false); // b_more_bits
                             // DRC P-frame.
        bw.write_bit(true); // b_drc_present
        bw.write_bit(false); // drc_reset_flag
        bw.write_u32(0, 2); // drc_reserved
                            // DE P-frame.
        bw.write_bit(true); // b_de_data_present
        bw.write_bit(false); // de_config_flag (re-use prev)
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);

        // Build up a previous state mirroring the I-frame above.
        let prev_drc = DrcConfig {
            drc_decoder_nr_modes: 0,
            drc_eac3_profile: 0,
            modes: vec![crate::drc::DrcDecoderMode {
                drc_decoder_mode_id: 0,
                drc_output_level_from: None,
                drc_output_level_to: None,
                drc_repeat_profile_flag: false,
                drc_repeat_id: None,
                drc_default_profile_flag: Some(true),
                drc_compression_curve_flag: true,
                compression_curve: None,
                drc_gains_config: None,
            }],
        };
        let prev_de = DeConfig {
            method: crate::de::DeMethod::ChannelIndependent,
            max_gain: 0,
            channel_config: 0,
        };
        let state = MetadataState {
            prev_drc_config: Some(prev_drc),
            prev_de_config: Some(prev_de),
        };

        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: false,
            b_associated: false,
            b_dialog: false,
            frame_length: 512,
        };
        let m = parse_metadata(&mut br, ctx, &state).unwrap();
        assert!(m.drc.b_drc_present);
        // P-frame: no fresh config returned, but data_present should be true.
        assert!(m.drc.config.is_none());
        assert!(m.drc.data.is_some());
        assert!(m.dialog_enhancement.data_present);
        assert!(!m.dialog_enhancement.config_flag);
        assert_eq!(m.tools_metadata_size, 7);
    }

    #[test]
    fn metadata_walker_emdf_present_terminator_only_is_ok() {
        // Same minimal payload but with b_emdf_payloads_substream = 1
        // followed by a bare terminator (5-bit emdf_payload_id == 0)
        // and the trailing byte_align. Walker now decodes through it
        // and returns an empty payload list rather than erroring out.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(2, 7); // tools_metadata_size = 2
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        bw.write_bit(true); // b_emdf_payloads_substream = 1
                            // emdf_payloads_substream(): just the terminator id == 0, then
                            // the byte_align inside parse_emdf_payloads_substream.
        bw.write_u32(0, 5);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        let m = parse_metadata(&mut br, ctx, &MetadataState::default()).unwrap();
        assert!(m.emdf_payloads_substream_present);
        let sub = m
            .emdf_payloads_substream
            .as_ref()
            .expect("present flag set, payload missing");
        assert!(sub.payloads.is_empty());
    }

    #[test]
    fn metadata_walker_emdf_present_with_one_payload_round_trips() {
        // Same minimal frame followed by emdf_payloads_substream() with
        // one minimal payload (id = 9, empty bytes, b_discard_unknown_payload = 1).
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(2, 7); // tools_metadata_size = 2
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        bw.write_bit(true); // b_emdf_payloads_substream = 1
                            // Payload 1: id=9, all-zero config gates + discard=1, size=0.
        bw.write_u32(9, 5);
        bw.write_bit(false); // b_smpoffst
        bw.write_bit(false); // b_duration
        bw.write_bit(false); // b_groupid
        bw.write_bit(false); // b_codecdata
        bw.write_bit(true); // b_discard_unknown_payload
                            // emdf_payload_size = variable_bits(8) → just 8 bits of zero
                            // and the 1-bit "more" terminator.
        bw.write_u32(0, 8);
        bw.write_bit(false); // more
                             // Terminator id == 0.
        bw.write_u32(0, 5);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        let m = parse_metadata(&mut br, ctx, &MetadataState::default()).unwrap();
        let sub = m.emdf_payloads_substream.as_ref().unwrap();
        assert_eq!(sub.payloads.len(), 1);
        assert_eq!(sub.payloads[0].emdf_payload_id, 9);
        assert!(sub.payloads[0].payload_bytes.is_empty());
    }

    // ------------------------------------------------------------------
    // further_loudness_info round-trip — tiny smoke test.
    // ------------------------------------------------------------------

    #[test]
    fn further_loudness_info_minimum() {
        // loudness_version=0, loud_prac_type=0 (skip nested ifs),
        // all booleans = 0 except b_extension=0. That's 2+4+11 zero
        // bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // loudness_version
        bw.write_u32(0, 4); // loud_prac_type
        for _ in 0..11 {
            bw.write_bit(false); // 11 boolean gates all zero
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let v = parse_further_loudness_info(&mut br).unwrap();
        assert_eq!(v.loudness_version, 0);
        assert_eq!(v.loud_prac_type, 0);
        assert!(v.loudrelgat.is_none());
        assert!(v.loudmntry.is_none());
    }

    // ------------------------------------------------------------------
    // Write-side round-trip — §4.2.14.2/.3 Table 67/68
    // ------------------------------------------------------------------

    fn fli_round_trips(v: &FurtherLoudnessInfo) {
        let mut bw = BitWriter::new();
        write_further_loudness_info(&mut bw, v).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = parse_further_loudness_info(&mut br).unwrap();
        assert_eq!(&got, v);
    }

    #[test]
    fn write_fli_default_round_trips() {
        fli_round_trips(&FurtherLoudnessInfo::default());
    }

    #[test]
    fn write_fli_full_round_trips() {
        let v = FurtherLoudnessInfo {
            loudness_version: 9, // exercises the 4-bit extension (>= 3)
            loud_prac_type: 5,
            dialgate_prac_type: Some(4),
            loudcorr_type: Some(true),
            loudrelgat: Some(1500),
            loudspchgat: Some(700),
            loudspchgat_dialgate_prac_type: Some(2),
            loudstrm3s: Some(800),
            max_loudstrm3s: Some(900),
            truepk: Some(1000),
            max_truepk: Some(1100),
            prgmbndy: Some(1 << 5),
            b_end_or_start: Some(true),
            prgmbndy_offset: Some(123),
            lra: Some(456),
            lra_prac_type: Some(3),
            loudmntry: Some(200),
            max_loudmntry: Some(300),
        };
        fli_round_trips(&v);
    }

    #[test]
    fn write_fli_prac_type_zero_skips_nested() {
        // loud_prac_type == 0 → dialgate/loudcorr fields not transmitted.
        let v = FurtherLoudnessInfo {
            loudness_version: 1,
            loud_prac_type: 0,
            loudrelgat: Some(42),
            ..Default::default()
        };
        fli_round_trips(&v);
    }

    fn bm_round_trips(v: &BasicMetadata, channel_mode: u32) {
        let mut bw = BitWriter::new();
        write_basic_metadata(&mut bw, v, channel_mode).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = parse_basic_metadata(&mut br, channel_mode).unwrap();
        assert_eq!(&got, v);
    }

    #[test]
    fn write_bm_minimal_no_more_round_trips() {
        let v = BasicMetadata {
            dialnorm_bits: 100,
            more_basic_metadata: false,
            ..Default::default()
        };
        bm_round_trips(&v, channel_mode::MONO);
    }

    #[test]
    fn write_bm_stereo_prev_dmx_round_trips() {
        let v = BasicMetadata {
            dialnorm_bits: 64,
            more_basic_metadata: true,
            pre_dmixtyp_2ch: Some(5),
            phase90_info_2ch: Some(2),
            dc_block_on: Some(true),
            ..Default::default()
        };
        bm_round_trips(&v, channel_mode::STEREO);
    }

    #[test]
    fn write_bm_51_full_dmx_round_trips() {
        let v = BasicMetadata {
            dialnorm_bits: 70,
            more_basic_metadata: true,
            further_loudness_info: Some(FurtherLoudnessInfo {
                loudness_version: 1,
                loud_prac_type: 2,
                dialgate_prac_type: Some(1),
                loudcorr_type: Some(false),
                ..Default::default()
            }),
            loro_centre_mixgain: Some(3),
            loro_surround_mixgain: Some(4),
            loro_dmx_loud_corr: Some(7),
            ltrt_centre_mixgain: Some(2),
            ltrt_surround_mixgain: Some(5),
            ltrt_dmx_loud_corr: Some(9),
            lfe_mixgain: Some(11), // 5.1 has LFE
            preferred_dmx_method: Some(2),
            pre_dmixtyp_5ch: Some(6),
            pre_upmixtyp_5ch: Some(10),
            phase90_info_mc: Some(1),
            b_surround_attenuation_known: Some(true),
            b_lfe_attenuation_known: Some(false),
            dc_block_on: Some(false),
            ..Default::default()
        };
        bm_round_trips(&v, channel_mode::FIVE_1);
    }

    #[test]
    fn write_bm_7x_upmix_round_trips() {
        // channel_mode 5 (7.0 family, <= 6) → pre_upmixtyp_3_4 path.
        let v = BasicMetadata {
            dialnorm_bits: 55,
            more_basic_metadata: true,
            preferred_dmx_method: Some(1),
            loro_centre_mixgain: Some(1),
            loro_surround_mixgain: Some(2),
            pre_upmixtyp_3_4: Some(3),
            phase90_info_mc: Some(2),
            b_surround_attenuation_known: Some(false),
            b_lfe_attenuation_known: Some(false),
            ..Default::default()
        };
        bm_round_trips(&v, 5);
    }

    #[test]
    fn write_bm_mc_no_dmx_coeff_round_trips() {
        // channel_mode > stereo but b_dmx_coeff = 0 (loro fields absent).
        let v = BasicMetadata {
            dialnorm_bits: 33,
            more_basic_metadata: true,
            phase90_info_mc: Some(0),
            b_surround_attenuation_known: Some(true),
            b_lfe_attenuation_known: Some(true),
            ..Default::default()
        };
        bm_round_trips(&v, channel_mode::C_LR);
    }

    #[test]
    fn write_bm_rejects_inconsistent_stereo() {
        let v = BasicMetadata {
            dialnorm_bits: 1,
            more_basic_metadata: true,
            pre_dmixtyp_2ch: Some(1),
            phase90_info_2ch: None, // mismatched
            ..Default::default()
        };
        let mut bw = BitWriter::new();
        let err = write_basic_metadata(&mut bw, &v, channel_mode::STEREO).unwrap_err();
        assert!(err.to_string().contains("phase90_info_2ch"), "got: {err}");
    }

    fn em_round_trips(v: &ExtendedMetadata, channel_mode: u32, assoc: bool, dialog: bool) {
        let mut bw = BitWriter::new();
        write_extended_metadata(&mut bw, v, channel_mode, assoc, dialog).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = parse_extended_metadata(&mut br, channel_mode, assoc, dialog).unwrap();
        assert_eq!(&got, v);
    }

    #[test]
    fn write_em_empty_round_trips() {
        em_round_trips(
            &ExtendedMetadata::default(),
            channel_mode::MONO,
            false,
            false,
        );
    }

    #[test]
    fn write_em_associated_mono_round_trips() {
        let v = ExtendedMetadata {
            scale_main: Some(0xAB),
            scale_main_centre: Some(0xCD),
            scale_main_front: None,
            pan_associated: Some(0x42),
            ..Default::default()
        };
        em_round_trips(&v, channel_mode::MONO, true, false);
    }

    #[test]
    fn write_em_dialog_pair_round_trips() {
        let v = ExtendedMetadata {
            dialog_max_gain: Some(2),
            pan_dialog_pair: Some((0x11, 0x22)),
            pan_signal_selector: Some(1),
            ..Default::default()
        };
        em_round_trips(&v, channel_mode::FIVE_1, false, true);
    }

    #[test]
    fn write_em_dialog_mono_pan_round_trips() {
        let v = ExtendedMetadata {
            dialog_max_gain: Some(1),
            pan_dialog: Some(0x77),
            ..Default::default()
        };
        em_round_trips(&v, channel_mode::MONO, false, true);
    }

    #[test]
    fn write_em_classifier_51_round_trips() {
        // 5.1: c/l/r + ls/rs + lfe present.
        let v = ExtendedMetadata {
            b_channels_classifier: true,
            b_c_active: Some(true),
            b_c_has_dialog: Some(true),
            b_l_active: None, // inactive
            b_r_active: Some(true),
            b_r_has_dialog: Some(false),
            b_ls_active: Some(true),
            b_rs_active: Some(false),
            b_lfe_active: Some(true),
            event_probability: Some(9),
            ..Default::default()
        };
        em_round_trips(&v, channel_mode::FIVE_1, false, false);
    }

    #[test]
    fn write_em_classifier_present_but_empty_for_mono_round_trips() {
        // Mono has no classifiable channels, so the block emits nothing —
        // the explicit flag is what carries its presence.
        let v = ExtendedMetadata {
            b_channels_classifier: true,
            ..Default::default()
        };
        em_round_trips(&v, channel_mode::MONO, false, false);
    }

    // ------------------------------------------------------------------
    // Outer metadata() write-side round-trip — §4.2.14.1 Table 66
    // ------------------------------------------------------------------

    /// Parse a hand-built metadata bitstream, write it back, re-parse, and
    /// assert structural equality (write∘parse identity).
    fn metadata_round_trips(raw: &[u8], ctx: MetadataContext, state: &MetadataState) {
        let mut br = BitReader::new(raw);
        let first = parse_metadata(&mut br, ctx, state).unwrap();
        let mut bw = BitWriter::new();
        write_metadata(&mut bw, &first, ctx).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br2 = BitReader::new(&bytes);
        let second = parse_metadata(&mut br2, ctx, state).unwrap();
        // DrcFrame/Metadata derive PartialEq; compare the load-bearing
        // fields directly.
        assert_eq!(first.basic, second.basic);
        assert_eq!(first.extended, second.extended);
        assert_eq!(first.tools_metadata_size, second.tools_metadata_size);
        assert_eq!(first.drc, second.drc);
        assert_eq!(first.dialog_enhancement, second.dialog_enhancement);
        assert_eq!(
            first.emdf_payloads_substream,
            second.emdf_payloads_substream
        );
        assert_eq!(
            first.tools_metadata_trailing_bits,
            second.tools_metadata_trailing_bits
        );
    }

    #[test]
    fn write_metadata_minimal_iframe_mono_round_trips() {
        let mut bw = BitWriter::new();
        bw.write_u32(0x40, 7); // dialnorm
        bw.write_bit(false); // more_basic_metadata
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(2, 7); // tools_metadata_size
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let raw = bw.finish();
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        metadata_round_trips(&raw, ctx, &MetadataState::default());
    }

    #[test]
    fn write_metadata_trailing_bits_round_trips() {
        // tools_metadata_size = 6 → 2 consumed + 4 reserved zero bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(6, 7); // tools_metadata_size = 6
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        for _ in 0..4 {
            bw.write_bit(false); // reserved trailing
        }
        bw.write_bit(false); // b_emdf_payloads_substream
        bw.align_to_byte();
        let raw = bw.finish();
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        metadata_round_trips(&raw, ctx, &MetadataState::default());
    }

    #[test]
    fn write_metadata_with_emdf_payload_round_trips() {
        let mut bw = BitWriter::new();
        bw.write_u32(0x20, 7); // dialnorm
        bw.write_bit(false); // more
        bw.write_bit(false); // b_channels_classifier
        bw.write_bit(false); // b_event_probability
        bw.write_u32(2, 7); // tools_metadata_size
        bw.write_bit(false); // b_more_bits
        write_minimal_drc_absent(&mut bw);
        write_minimal_de_absent(&mut bw);
        bw.write_bit(true); // b_emdf_payloads_substream
                            // emdf_payloads_substream(): one payload id 7, minimal config,
                            // 1 byte, then terminator + byte_align.
        bw.write_u32(7, 5); // emdf_payload_id
        bw.write_bit(false); // b_smpoffst
        bw.write_bit(false); // b_duration
        bw.write_bit(false); // b_groupid
        bw.write_bit(false); // b_codecdata
        bw.write_bit(true); // b_discard_unknown_payload
        bw.write_u32(1, 8); // emdf_payload_size = 1 (variable_bits(8))
        bw.write_bit(false); // continuation for variable_bits
        bw.write_u32(0xAB, 8); // payload byte
        bw.write_u32(0, 5); // terminator
        bw.align_to_byte();
        let raw = bw.finish();
        let ctx = MetadataContext {
            channel_mode: channel_mode::MONO,
            b_iframe: true,
            b_associated: false,
            b_dialog: false,
            frame_length: 1024,
        };
        metadata_round_trips(&raw, ctx, &MetadataState::default());
    }
}
