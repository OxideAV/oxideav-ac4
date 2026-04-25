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

use crate::de::{parse_dialog_enhancement, DeConfig, DialogEnhancement};
use crate::drc::{
    nr_drc_channels, nr_drc_subframes, parse_drc_frame, DrcChannelInfo, DrcConfig, DrcFrame,
};
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
    /// `b_emdf_payloads_substream` flag (the payload itself is skipped
    /// today — full EMDF parsing is out of scope for round 15).
    pub emdf_payloads_substream_present: bool,
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
    if emdf_payloads_substream_present {
        // §4.2.4.4 emdf_payloads_substream() is gated by EMDF semantics
        // that aren't yet implemented in this crate — return an error
        // rather than silently mis-aligning the bitstream.
        return Err(Error::invalid(
            "ac4: emdf_payloads_substream() parsing is not yet implemented",
        ));
    }

    Ok(Metadata {
        basic,
        extended,
        tools_metadata_size: tools_size,
        drc,
        dialog_enhancement,
        emdf_payloads_substream_present,
        tools_metadata_trailing_bits: trailing,
    })
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
    fn metadata_walker_emdf_present_errors_until_implemented() {
        // Same minimal payload but with b_emdf_payloads_substream = 1
        // → walker errors with "not yet implemented" rather than
        // mis-aligning.
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
        let err = parse_metadata(&mut br, ctx, &MetadataState::default()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("emdf"), "expected EMDF error, got: {msg}");
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
}
