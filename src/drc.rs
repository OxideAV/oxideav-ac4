//! Dynamic Range Control (DRC) metadata parsing — §4.2.14.5..10 syntax,
//! §4.3.13 semantics. Pure parser: walks the bitstream and surfaces the
//! decoded fields, plus the per-(channel, subframe, band) DRC gain
//! matrix decoded via the Annex A.5 Huffman codebook
//! ([`crate::drc_huffman`]).
//!
//! Entry point: [`parse_drc_frame`] (§4.2.14.5 Table 70). It consumes
//! `b_drc_present`, optionally `drc_config()` on I-frames, and the
//! per-frame `drc_data()` payload that carries the gain stream.
//!
//! Out of scope here:
//! * Application of the gains to PCM samples (§5.7.9 DRC tool).
//! * E-AC-3 transcoding profile lookup beyond surfacing the
//!   [`DrcConfig::drc_eac3_profile`] field.
//! * Skipped `drc2_bits` payload (`drc_version > 0`) — bits are read
//!   and counted but their content is not interpreted.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::drc_huffman::drc_huff_decode_diff;
use crate::toc::variable_bits;

/// Maximum number of DRC decoder modes per `drc_decoder_nr_modes` (§4.3.13.2.1):
/// the field is 3 bits and the mode count is `value + 1`, so 1..=8.
pub const DRC_MAX_DECODER_MODES: usize = 8;

/// Decoded DRC compression curve (§4.2.14.8 Table 73, §4.3.13.4 semantics).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DrcCompressionCurve {
    pub drc_lev_nullband_low: u8,
    pub drc_lev_nullband_high: u8,
    pub drc_gain_max_boost: u8,
    pub drc_lev_max_boost: Option<u8>,
    pub drc_nr_boost_sections: Option<u8>,
    pub drc_gain_section_boost: Option<u8>,
    pub drc_lev_section_boost: Option<u8>,
    pub drc_gain_max_cut: u8,
    pub drc_lev_max_cut: Option<u8>,
    pub drc_nr_cut_sections: Option<u8>,
    pub drc_gain_section_cut: Option<u8>,
    pub drc_lev_section_cut: Option<u8>,
    pub drc_tc_default_flag: bool,
    /// `Some(_)` when `!drc_tc_default_flag`: per-section time constants.
    pub time_constants: Option<DrcTimeConstants>,
}

/// Optional per-decoder-mode time-constant block (§4.3.13.4.14..21).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DrcTimeConstants {
    pub drc_tc_attack: u8,
    pub drc_tc_release: u8,
    pub drc_tc_attack_fast: u8,
    pub drc_tc_release_fast: u8,
    pub drc_adaptive_smoothing_flag: bool,
    pub drc_attack_threshold: Option<u8>,
    pub drc_release_threshold: Option<u8>,
}

/// One `drc_decoder_mode_config()` element (§4.2.14.7 Table 72).
#[derive(Debug, Clone, PartialEq)]
pub struct DrcDecoderMode {
    pub drc_decoder_mode_id: u8,
    pub drc_output_level_from: Option<u8>,
    pub drc_output_level_to: Option<u8>,
    pub drc_repeat_profile_flag: bool,
    /// `Some(id)` when `drc_repeat_profile_flag` is set.
    pub drc_repeat_id: Option<u8>,
    /// `drc_default_profile_flag`. Only present when `!repeat_profile_flag`.
    pub drc_default_profile_flag: Option<bool>,
    /// True when this mode carries (or repeats) a compression curve;
    /// false when it carries gainset entries via `drc_gains_config`.
    pub drc_compression_curve_flag: bool,
    /// `Some(curve)` when this mode transmits its own compression curve.
    pub compression_curve: Option<DrcCompressionCurve>,
    /// `Some(value)` when `!compression_curve_flag` for this mode (gainset).
    pub drc_gains_config: Option<u8>,
}

/// Decoded `drc_config()` (§4.2.14.6 Table 71).
#[derive(Debug, Clone, PartialEq)]
pub struct DrcConfig {
    /// Per spec: number of decoder modes is `drc_decoder_nr_modes + 1`,
    /// so this field stores the *raw 3-bit value*. Use [`Self::nr_modes`]
    /// for the count.
    pub drc_decoder_nr_modes: u8,
    pub drc_eac3_profile: u8,
    pub modes: Vec<DrcDecoderMode>,
}

impl DrcConfig {
    /// Returns `drc_decoder_nr_modes + 1` per §4.3.13.2.1.
    pub fn nr_modes(&self) -> usize {
        self.drc_decoder_nr_modes as usize + 1
    }
}

/// Decoded `drc_gains()` payload (§4.2.14.10 Table 75).
///
/// `drc_gain[ch][sf][band]` flattened in `(ch, band, sf)` walk order
/// per the spec loop. Index helper: [`DrcGains::idx`].
#[derive(Debug, Clone, PartialEq)]
pub struct DrcGains {
    pub mode_id: u8,
    pub nr_drc_channels: u8,
    pub nr_drc_subframes: u8,
    pub nr_drc_bands: u8,
    /// 7-bit `drc_gain_val` (§4.3.13.6.1); `drc_gain[0][0][0] = val - 64 [dB]`.
    pub drc_gain_val: u8,
    /// All decoded gains as 7-bit unsigned values (-64 dB offset applies).
    /// Length = `nr_drc_channels * nr_drc_subframes * nr_drc_bands`.
    pub drc_gain: Vec<u8>,
}

impl DrcGains {
    /// Walk-order flattened index for `(ch, sf, band)` — matches the
    /// spec's three nested loops: `for ch { for band { for sf {} } }`.
    pub fn idx(&self, ch: usize, sf: usize, band: usize) -> usize {
        let bands = self.nr_drc_bands as usize;
        let sfs = self.nr_drc_subframes as usize;
        ch * bands * sfs + band * sfs + sf
    }

    /// `drc_gain[ch][sf][band]` in dB per §4.3.13.6.1 (`val - 64`).
    pub fn gain_db(&self, ch: usize, sf: usize, band: usize) -> i32 {
        self.drc_gain[self.idx(ch, sf, band)] as i32 - 64
    }
}

/// Decoded `drc_data()` (§4.2.14.9 Table 74) plus framing info that the
/// decoder needs in order to apply the gains.
#[derive(Debug, Clone, PartialEq)]
pub struct DrcData {
    /// One `DrcGains` per decoder mode that has `compression_curve_flag == 0`.
    /// Modes with curves don't carry per-frame gains.
    pub gainsets: Vec<DrcGains>,
    /// Set when at least one mode in this frame is curve-driven.
    pub curve_present: bool,
    /// Only meaningful when `curve_present`.
    pub drc_reset_flag: bool,
    /// Reserved 2 bits (§4.3.13.5.6) — surfaced for round-trip fidelity.
    pub drc_reserved: u8,
}

/// Top-level decoded `drc_frame()` (§4.2.14.5 Table 70).
#[derive(Debug, Clone, PartialEq)]
pub struct DrcFrame {
    pub b_drc_present: bool,
    /// I-frame configuration. `Some` when `b_drc_present && b_iframe`.
    pub config: Option<DrcConfig>,
    /// Per-frame data. `Some` when `b_drc_present`.
    pub data: Option<DrcData>,
}

/// Channel-config descriptor needed to decode `drc_data()` because the
/// gain-set walk depends on `nr_drc_channels` (§4.3.13.7.1 Table 168)
/// and `nr_drc_subframes` (§4.3.13.7.2 Table 169).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrcChannelInfo {
    /// From Table 168 — depends on the speaker configuration of the frame.
    pub nr_drc_channels: u8,
    /// From Table 169 — depends on the AC-4 frame length.
    pub nr_drc_subframes: u8,
}

impl DrcChannelInfo {
    pub fn new(nr_drc_channels: u8, nr_drc_subframes: u8) -> Self {
        Self {
            nr_drc_channels,
            nr_drc_subframes,
        }
    }
}

/// Map an `nr_drc_subframes` value from the AC-4 frame length per
/// Table 169 (§4.3.13.7.2). Returns `None` for frame lengths not in
/// the table.
pub fn nr_drc_subframes(frame_length: u32) -> Option<u8> {
    match frame_length {
        384 => Some(1),
        512 => Some(2),
        768 | 960 => Some(3),
        1024 => Some(4),
        1536 | 1920 => Some(6),
        2048 => Some(8),
        _ => None,
    }
}

/// Map `nr_drc_channels` per Table 168 (§4.3.13.7.1). The `channel_mode`
/// here is the decoded `channel_mode` value from `ac4_substream_info()`
/// (mono = 0, stereo = 1, 5.1 = 4 .. etc per Annex H). Falls back to
/// the most common groups when the configuration is exotic; callers
/// expecting deterministic behaviour for unhandled modes should
/// override this.
pub fn nr_drc_channels(channel_mode: u32) -> u8 {
    // Table 168 maps Mono/Stereo to 1, the rest of the listed configs
    // (5.1 / 7.1 variants) to 3. The spec is silent about other
    // channel modes; for them we default to 1 wideband group and let
    // the bitstream's drc_gainset_size carry the truth.
    match channel_mode {
        0 | 1 => 1, // Mono, Stereo
        _ => 3,     // 5.1, 7.1 family per Table 168
    }
}

/// Map `drc_gains_config` value to `nr_drc_bands` per Table 163.
pub fn nr_drc_bands(drc_gains_config: u8) -> u8 {
    match drc_gains_config {
        0 | 1 => 1,
        2 => 2,
        3 => 4,
        _ => 1, // Reserved values are out of range for a 2-bit field.
    }
}

/// `drc_frame(b_iframe)` per §4.2.14.5 Table 70.
///
/// `b_iframe` controls whether `drc_config()` is present (only on
/// I-frames). `chan_info` provides the channel-configuration-dependent
/// `nr_drc_channels` / `nr_drc_subframes` constants needed to decode
/// the gainset payload. `prev_config` is the configuration carried by
/// a prior I-frame in this stream — when `b_drc_present` is true on a
/// non-I-frame, the parser falls back on `prev_config` to know how
/// many decoder modes (and which `compression_curve_flag` /
/// `drc_gains_config` they have) are in flight.
pub fn parse_drc_frame(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    chan_info: DrcChannelInfo,
    prev_config: Option<&DrcConfig>,
) -> Result<DrcFrame> {
    let b_drc_present = br.read_bit()?;
    if !b_drc_present {
        return Ok(DrcFrame {
            b_drc_present: false,
            config: None,
            data: None,
        });
    }

    let config = if b_iframe {
        Some(parse_drc_config(br)?)
    } else {
        None
    };

    let active = config.as_ref().or(prev_config).ok_or_else(|| {
        Error::invalid(
            "ac4: drc_frame on non-I-frame with no prior drc_config — \
             cannot decode drc_data() without per-mode flags",
        )
    })?;

    let data = parse_drc_data(br, active, chan_info)?;

    Ok(DrcFrame {
        b_drc_present: true,
        config,
        data: Some(data),
    })
}

/// `drc_config()` per §4.2.14.6 Table 71.
pub fn parse_drc_config(br: &mut BitReader<'_>) -> Result<DrcConfig> {
    let drc_decoder_nr_modes = br.read_u32(3)? as u8;
    let mode_count = drc_decoder_nr_modes as usize + 1;
    let mut modes = Vec::with_capacity(mode_count);
    for _ in 0..mode_count {
        modes.push(parse_drc_decoder_mode_config(br, &modes)?);
    }
    let drc_eac3_profile = br.read_u32(3)? as u8;
    Ok(DrcConfig {
        drc_decoder_nr_modes,
        drc_eac3_profile,
        modes,
    })
}

/// `drc_decoder_mode_config()` per §4.2.14.7 Table 72.
///
/// The `drc_repeat_profile_flag` path duplicates the
/// `drc_compression_curve_flag` value from the referenced (already
/// decoded) mode, so callers must pass the modes decoded so far via
/// `prior_modes`. If the reference mode is missing the function
/// errors out — that's a malformed bitstream.
fn parse_drc_decoder_mode_config(
    br: &mut BitReader<'_>,
    prior_modes: &[DrcDecoderMode],
) -> Result<DrcDecoderMode> {
    let drc_decoder_mode_id = br.read_u32(3)? as u8;
    let (drc_output_level_from, drc_output_level_to) = if drc_decoder_mode_id > 3 {
        (Some(br.read_u32(5)? as u8), Some(br.read_u32(5)? as u8))
    } else {
        (None, None)
    };

    let drc_repeat_profile_flag = br.read_bit()?;
    if drc_repeat_profile_flag {
        let drc_repeat_id = br.read_u32(3)? as u8;
        // Per spec: drc_compression_curve_flag[mode_id]
        //          = drc_compression_curve_flag[drc_repeat_id].
        let referenced = prior_modes
            .iter()
            .find(|m| m.drc_decoder_mode_id == drc_repeat_id)
            .ok_or_else(|| Error::invalid("ac4: drc_repeat_id refers to undeclared mode"))?;
        Ok(DrcDecoderMode {
            drc_decoder_mode_id,
            drc_output_level_from,
            drc_output_level_to,
            drc_repeat_profile_flag: true,
            drc_repeat_id: Some(drc_repeat_id),
            drc_default_profile_flag: None,
            drc_compression_curve_flag: referenced.drc_compression_curve_flag,
            compression_curve: None, // Curve lives on the referenced mode.
            drc_gains_config: referenced.drc_gains_config,
        })
    } else {
        let drc_default_profile_flag = br.read_bit()?;
        if drc_default_profile_flag {
            // Default-profile mode: compression_curve_flag is implicitly
            // 1 per §4.2.14.7 last `else { ... = 1; }` branch.
            Ok(DrcDecoderMode {
                drc_decoder_mode_id,
                drc_output_level_from,
                drc_output_level_to,
                drc_repeat_profile_flag: false,
                drc_repeat_id: None,
                drc_default_profile_flag: Some(true),
                drc_compression_curve_flag: true,
                compression_curve: None,
                drc_gains_config: None,
            })
        } else {
            let drc_compression_curve_flag = br.read_bit()?;
            if drc_compression_curve_flag {
                let curve = parse_drc_compression_curve(br)?;
                Ok(DrcDecoderMode {
                    drc_decoder_mode_id,
                    drc_output_level_from,
                    drc_output_level_to,
                    drc_repeat_profile_flag: false,
                    drc_repeat_id: None,
                    drc_default_profile_flag: Some(false),
                    drc_compression_curve_flag: true,
                    compression_curve: Some(curve),
                    drc_gains_config: None,
                })
            } else {
                let drc_gains_config = br.read_u32(2)? as u8;
                Ok(DrcDecoderMode {
                    drc_decoder_mode_id,
                    drc_output_level_from,
                    drc_output_level_to,
                    drc_repeat_profile_flag: false,
                    drc_repeat_id: None,
                    drc_default_profile_flag: Some(false),
                    drc_compression_curve_flag: false,
                    compression_curve: None,
                    drc_gains_config: Some(drc_gains_config),
                })
            }
        }
    }
}

/// `drc_compression_curve()` per §4.2.14.8 Table 73.
pub fn parse_drc_compression_curve(br: &mut BitReader<'_>) -> Result<DrcCompressionCurve> {
    let drc_lev_nullband_low = br.read_u32(4)? as u8;
    let drc_lev_nullband_high = br.read_u32(4)? as u8;
    let drc_gain_max_boost = br.read_u32(4)? as u8;
    let (drc_lev_max_boost, drc_nr_boost_sections, drc_gain_section_boost, drc_lev_section_boost) =
        if drc_gain_max_boost > 0 {
            let lmb = br.read_u32(5)? as u8;
            let nbs = br.read_u32(1)? as u8;
            if nbs > 0 {
                (
                    Some(lmb),
                    Some(nbs),
                    Some(br.read_u32(4)? as u8),
                    Some(br.read_u32(5)? as u8),
                )
            } else {
                (Some(lmb), Some(nbs), None, None)
            }
        } else {
            (None, None, None, None)
        };

    let drc_gain_max_cut = br.read_u32(5)? as u8;
    let (drc_lev_max_cut, drc_nr_cut_sections, drc_gain_section_cut, drc_lev_section_cut) =
        if drc_gain_max_cut > 0 {
            let lmc = br.read_u32(6)? as u8;
            let ncs = br.read_u32(1)? as u8;
            if ncs > 0 {
                (
                    Some(lmc),
                    Some(ncs),
                    Some(br.read_u32(5)? as u8),
                    Some(br.read_u32(5)? as u8),
                )
            } else {
                (Some(lmc), Some(ncs), None, None)
            }
        } else {
            (None, None, None, None)
        };

    let drc_tc_default_flag = br.read_bit()?;
    let time_constants = if !drc_tc_default_flag {
        let drc_tc_attack = br.read_u32(8)? as u8;
        let drc_tc_release = br.read_u32(8)? as u8;
        let drc_tc_attack_fast = br.read_u32(8)? as u8;
        let drc_tc_release_fast = br.read_u32(8)? as u8;
        let drc_adaptive_smoothing_flag = br.read_bit()?;
        let (drc_attack_threshold, drc_release_threshold) = if drc_adaptive_smoothing_flag {
            (Some(br.read_u32(5)? as u8), Some(br.read_u32(5)? as u8))
        } else {
            (None, None)
        };
        Some(DrcTimeConstants {
            drc_tc_attack,
            drc_tc_release,
            drc_tc_attack_fast,
            drc_tc_release_fast,
            drc_adaptive_smoothing_flag,
            drc_attack_threshold,
            drc_release_threshold,
        })
    } else {
        None
    };

    Ok(DrcCompressionCurve {
        drc_lev_nullband_low,
        drc_lev_nullband_high,
        drc_gain_max_boost,
        drc_lev_max_boost,
        drc_nr_boost_sections,
        drc_gain_section_boost,
        drc_lev_section_boost,
        drc_gain_max_cut,
        drc_lev_max_cut,
        drc_nr_cut_sections,
        drc_gain_section_cut,
        drc_lev_section_cut,
        drc_tc_default_flag,
        time_constants,
    })
}

/// `drc_data()` per §4.2.14.9 Table 74.
pub fn parse_drc_data(
    br: &mut BitReader<'_>,
    config: &DrcConfig,
    chan_info: DrcChannelInfo,
) -> Result<DrcData> {
    let mut curve_present = false;
    let mut gainsets = Vec::new();

    for mode in &config.modes {
        let mode_id = mode.drc_decoder_mode_id;
        if !mode.drc_compression_curve_flag {
            // Gainset payload: drc_gainset_size (6 bits, optionally
            // extended via variable_bits(2)), drc_version (2 bits),
            // then drc_gains() if version <= 1, then optional
            // drc2_bits filler.
            let mut drc_gainset_size = br.read_u32(6)?;
            let b_more_bits = br.read_bit()?;
            if b_more_bits {
                drc_gainset_size += variable_bits(br, 2)? << 6;
            }
            let drc_version = br.read_u32(2)?;
            let bit_pos_before = br.bit_position();
            let gainset = if drc_version <= 1 {
                let gains =
                    parse_drc_gains(br, mode_id, mode.drc_gains_config.unwrap_or(0), chan_info)?;
                Some(gains)
            } else {
                None
            };
            let used_bits = (br.bit_position() - bit_pos_before) as u32;
            // drc_gainset_size is the bit-count of the drc_gains payload
            // *plus* the drc_version bits per the inner equation
            // `bits_left = drc_gainset_size - 2 - used_bits`. Skip any
            // residual drc2_bits.
            if drc_version >= 1 {
                let bits_left = drc_gainset_size
                    .checked_sub(2)
                    .and_then(|v| v.checked_sub(used_bits))
                    .ok_or_else(|| {
                        Error::invalid(
                            "ac4: drc_gainset_size accounting underflow \
                             (drc_gains used more bits than declared)",
                        )
                    })?;
                if bits_left > 0 {
                    br.skip(bits_left)?;
                }
            }
            if let Some(g) = gainset {
                gainsets.push(g);
            }
        } else {
            curve_present = true;
        }
    }

    let (drc_reset_flag, drc_reserved) = if curve_present {
        let reset = br.read_bit()?;
        let reserved = br.read_u32(2)? as u8;
        (reset, reserved)
    } else {
        (false, 0)
    };

    Ok(DrcData {
        gainsets,
        curve_present,
        drc_reset_flag,
        drc_reserved,
    })
}

/// `drc_gains(mode)` per §4.2.14.10 Table 75.
///
/// The first gain `drc_gain[0][0][0]` is sent as a 7-bit
/// `drc_gain_val` (offset by -64 dB at use-time). All subsequent
/// (ch, band, sf) gains are deltas decoded via the Annex A.5 codebook
/// (`huff_decode_diff(DRC_HCB, drc_gain_code)`). The spec resets
/// `ref_drc_gain` to `drc_gain[ch][0][band]` between bands and to
/// `drc_gain[ch][0][0]` between channels.
pub fn parse_drc_gains(
    br: &mut BitReader<'_>,
    mode_id: u8,
    drc_gains_config_value: u8,
    chan_info: DrcChannelInfo,
) -> Result<DrcGains> {
    let drc_gain_val = br.read_u32(7)? as u8;
    let nr_drc_bands = nr_drc_bands(drc_gains_config_value);

    // `drc_gains_config == 0` means a single wideband gain shared by
    // all channels (just the 7-bit drc_gain_val).
    if drc_gains_config_value == 0 {
        return Ok(DrcGains {
            mode_id,
            nr_drc_channels: 1,
            nr_drc_subframes: 1,
            nr_drc_bands: 1,
            drc_gain_val,
            drc_gain: vec![drc_gain_val],
        });
    }

    let nr_ch = chan_info.nr_drc_channels as usize;
    let nr_sf = chan_info.nr_drc_subframes as usize;
    let nr_bd = nr_drc_bands as usize;
    if nr_ch == 0 || nr_sf == 0 || nr_bd == 0 {
        return Err(Error::invalid(
            "ac4: drc_gains() invalid (ch, sf, band) dimensions",
        ));
    }

    let mut gains = vec![0u8; nr_ch * nr_bd * nr_sf];
    // gain[ch][sf][band] = ref + diff
    // Walk order matches the spec: ch outer, band middle, sf inner.
    let mut ref_gain: i32 = drc_gain_val as i32;
    let mut first = true;
    let idx =
        |ch: usize, sf: usize, band: usize| -> usize { ch * nr_bd * nr_sf + band * nr_sf + sf };

    for ch in 0..nr_ch {
        // Reset reference between channels per spec — see the trailing
        // `ref_drc_gain = drc_gain[ch][0][0];` after the channel loop.
        // For ch == 0 this is the seed value drc_gain_val.
        for band in 0..nr_bd {
            // Reset reference between bands per spec — see the trailing
            // `ref_drc_gain = drc_gain[ch][0][band];` after the band
            // loop. For band == 0 we keep the channel reset.
            for sf in 0..nr_sf {
                if first {
                    // First slot is the seed drc_gain[0][0][0].
                    gains[idx(ch, sf, band)] = drc_gain_val;
                    first = false;
                } else {
                    let diff = drc_huff_decode_diff(br)?;
                    let g = ref_gain + diff;
                    if !(0..=127).contains(&g) {
                        return Err(Error::invalid(
                            "ac4: DRC gain out of 7-bit range after diff",
                        ));
                    }
                    gains[idx(ch, sf, band)] = g as u8;
                }
                ref_gain = gains[idx(ch, sf, band)] as i32;
            }
            // Spec reset at end of band: ref = drc_gain[ch][0][band].
            ref_gain = gains[idx(ch, 0, band)] as i32;
        }
        // Spec reset at end of channel: ref = drc_gain[ch][0][0].
        ref_gain = gains[idx(ch, 0, 0)] as i32;
    }

    Ok(DrcGains {
        mode_id,
        nr_drc_channels: chan_info.nr_drc_channels,
        nr_drc_subframes: chan_info.nr_drc_subframes,
        nr_drc_bands,
        drc_gain_val,
        drc_gain: gains,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drc_huffman::{DRC_HCB_CW, DRC_HCB_LEN};
    use oxideav_core::bits::BitWriter;

    fn write_zero_diff(bw: &mut BitWriter) {
        // index 127 -> diff 0
        bw.write_u32(DRC_HCB_CW[127], DRC_HCB_LEN[127] as u32);
    }

    fn write_plus_one_diff(bw: &mut BitWriter) {
        // index 128 -> diff +1
        bw.write_u32(DRC_HCB_CW[128], DRC_HCB_LEN[128] as u32);
    }

    fn write_minus_one_diff(bw: &mut BitWriter) {
        // index 126 -> diff -1
        bw.write_u32(DRC_HCB_CW[126], DRC_HCB_LEN[126] as u32);
    }

    #[test]
    fn nr_drc_subframes_table_169() {
        assert_eq!(nr_drc_subframes(384), Some(1));
        assert_eq!(nr_drc_subframes(512), Some(2));
        assert_eq!(nr_drc_subframes(768), Some(3));
        assert_eq!(nr_drc_subframes(960), Some(3));
        assert_eq!(nr_drc_subframes(1024), Some(4));
        assert_eq!(nr_drc_subframes(1536), Some(6));
        assert_eq!(nr_drc_subframes(1920), Some(6));
        assert_eq!(nr_drc_subframes(2048), Some(8));
        assert_eq!(nr_drc_subframes(999), None);
    }

    #[test]
    fn nr_drc_bands_table_163() {
        assert_eq!(nr_drc_bands(0), 1);
        assert_eq!(nr_drc_bands(1), 1);
        assert_eq!(nr_drc_bands(2), 2);
        assert_eq!(nr_drc_bands(3), 4);
    }

    #[test]
    fn parse_drc_frame_absent() {
        // b_drc_present = 0 -> no payload at all.
        let mut bw = BitWriter::new();
        bw.write_bit(false);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let info = DrcChannelInfo::new(1, 1);
        let frame = parse_drc_frame(&mut br, true, info, None).unwrap();
        assert!(!frame.b_drc_present);
        assert!(frame.config.is_none());
        assert!(frame.data.is_none());
    }

    #[test]
    fn parse_drc_frame_one_mode_wideband() {
        // I-frame with one decoder mode, default profile (so curve flag
        // implicitly 1, no gainset payload), no curve transmitted.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_drc_present
                            // drc_config():
        bw.write_u32(0, 3); // drc_decoder_nr_modes = 0 -> 1 mode
                            // mode 0:
        bw.write_u32(2, 3); // drc_decoder_mode_id = 2 (Portable Speakers)
                            // mode_id <= 3 so no output_level_from/to.
        bw.write_bit(false); // drc_repeat_profile_flag = 0
        bw.write_bit(true); // drc_default_profile_flag = 1 -> implicit curve flag
        bw.write_u32(1, 3); // drc_eac3_profile = 1 (Film standard)
                            // drc_data(): mode 0 has compression_curve_flag = 1 ->
                            // curve_present = 1, no gainsets, then drc_reset_flag (1b) +
                            // drc_reserved (2b).
        bw.write_bit(false); // drc_reset_flag
        bw.write_u32(0, 2); // drc_reserved
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let info = DrcChannelInfo::new(1, 1);
        let frame = parse_drc_frame(&mut br, true, info, None).unwrap();
        assert!(frame.b_drc_present);
        let cfg = frame.config.as_ref().expect("config present");
        assert_eq!(cfg.nr_modes(), 1);
        assert_eq!(cfg.modes[0].drc_decoder_mode_id, 2);
        assert_eq!(
            cfg.modes[0].drc_default_profile_flag,
            Some(true),
            "default profile"
        );
        assert!(cfg.modes[0].drc_compression_curve_flag);
        assert_eq!(cfg.drc_eac3_profile, 1);
        let data = frame.data.as_ref().expect("data present");
        assert!(data.curve_present);
        assert!(data.gainsets.is_empty());
        assert!(!data.drc_reset_flag);
    }

    #[test]
    fn parse_drc_frame_with_gainset_wideband() {
        // I-frame with one decoder mode that carries gainset
        // (drc_gains_config = 1 -> 1 wideband band, channel-dependent
        // gains). Channel info: 1 DRC channel, 2 subframes -> 1*1*2 = 2
        // gain entries. First is drc_gain_val = 64 (=> 0 dB), then one
        // diff.
        let drc_gain_val: u8 = 64;
        // drc_gainset_size accounting: drc_version (2b) + the gain
        // payload bits = drc_gain_val (7) + 1 diff (3 bits for +1) = 10.
        // So drc_gainset_size = 12 (= 2 + 10 inner bits, but the spec
        // formulation is `bits_left = drc_gainset_size - 2 - used_bits`,
        // so size includes drc_version's 2 bits). Thus we set size to
        // 12 (under 64 so single 6-bit field). Total bits = 12, version
        // skipped because 0.
        let drc_gainset_size: u32 = 2 /* drc_version */ + 7 /* drc_gain_val */ + 3 /* +1 diff */;

        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_drc_present
                            // drc_config:
        bw.write_u32(0, 3); // drc_decoder_nr_modes = 0 -> 1 mode
        bw.write_u32(0, 3); // drc_decoder_mode_id = 0 (Home Theatre)
        bw.write_bit(false); // drc_repeat_profile_flag
        bw.write_bit(false); // drc_default_profile_flag = 0
        bw.write_bit(false); // drc_compression_curve_flag = 0 -> gainset
        bw.write_u32(1, 2); // drc_gains_config = 1 (wideband, ch-dep)
        bw.write_u32(0, 3); // drc_eac3_profile = 0 (None)
                            // drc_data:
        bw.write_u32(drc_gainset_size, 6); // drc_gainset_size_value
        bw.write_bit(false); // b_more_bits = 0
        bw.write_u32(0, 2); // drc_version = 0
                            // drc_gains payload starts here:
        bw.write_u32(drc_gain_val as u32, 7);
        write_plus_one_diff(&mut bw);
        bw.align_to_byte();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        // Frame length 512 -> 2 subframes (Table 169).
        let info = DrcChannelInfo::new(1, 2);
        let frame = parse_drc_frame(&mut br, true, info, None).unwrap();
        let data = frame.data.as_ref().expect("data");
        assert!(!data.curve_present);
        assert_eq!(data.gainsets.len(), 1);
        let g = &data.gainsets[0];
        assert_eq!(g.nr_drc_channels, 1);
        assert_eq!(g.nr_drc_subframes, 2);
        assert_eq!(g.nr_drc_bands, 1);
        assert_eq!(g.drc_gain_val, 64);
        assert_eq!(g.drc_gain.len(), 2);
        // Gain[0][0][0] = drc_gain_val = 64 -> 0 dB.
        assert_eq!(g.gain_db(0, 0, 0), 0);
        // Gain[0][1][0] = ref(64) + diff(+1) = 65 -> +1 dB.
        assert_eq!(g.gain_db(0, 1, 0), 1);
    }

    #[test]
    fn parse_drc_frame_gainset_2band_2ch_2sf() {
        // 2 channels x 2 subframes x 2 bands = 8 slots, 7 deltas after
        // the seed.
        let drc_gain_val: u8 = 64;
        // Walk order: ch0 band0 sf0 (seed), ch0 band0 sf1 (+1),
        // ch0 band1 sf0 (-1), ch0 band1 sf1 (0), ch1 band0 sf0 (+1),
        // ch1 band0 sf1 (-1), ch1 band1 sf0 (0), ch1 band1 sf1 (+1).
        // After ch0 band1, ref resets to ch0 band1 sf0 = 63.
        // After ch1 (final), no further reset needed.
        let total_diff_bits = 3 + 3 + 2 + 3 + 3 + 2 + 3; // 19 bits

        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_drc_present
        bw.write_u32(0, 3); // drc_decoder_nr_modes = 0 -> 1 mode
        bw.write_u32(0, 3); // drc_decoder_mode_id = 0
        bw.write_bit(false); // drc_repeat_profile_flag
        bw.write_bit(false); // drc_default_profile_flag = 0
        bw.write_bit(false); // drc_compression_curve_flag = 0
        bw.write_u32(2, 2); // drc_gains_config = 2 (2 freq bands)
        bw.write_u32(0, 3); // drc_eac3_profile

        let drc_gainset_size: u32 = 2 + 7 + total_diff_bits;
        bw.write_u32(drc_gainset_size, 6);
        bw.write_bit(false); // b_more_bits
        bw.write_u32(0, 2); // drc_version = 0
        bw.write_u32(drc_gain_val as u32, 7);
        // 7 diffs in walk order:
        write_plus_one_diff(&mut bw);
        write_minus_one_diff(&mut bw);
        write_zero_diff(&mut bw);
        write_plus_one_diff(&mut bw);
        write_minus_one_diff(&mut bw);
        write_zero_diff(&mut bw);
        write_plus_one_diff(&mut bw);
        bw.align_to_byte();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let info = DrcChannelInfo::new(2, 2);
        let frame = parse_drc_frame(&mut br, true, info, None).unwrap();
        let g = &frame.data.as_ref().unwrap().gainsets[0];
        assert_eq!(g.nr_drc_bands, 2);
        assert_eq!(g.drc_gain.len(), 8);

        // Reproduce the expected walk in software:
        //   ref starts at drc_gain_val = 64.
        //   ch=0 band=0 sf=0 -> seed 64
        //   ch=0 band=0 sf=1 -> 64 + 1 = 65
        //   reset ref to gain[0][0][0] = 64 after band 0.
        //   ch=0 band=1 sf=0 -> 64 + (-1) = 63
        //   ch=0 band=1 sf=1 -> 63 + 0  = 63
        //   reset ref to gain[0][0][1] = 63 after band 1.
        //   reset ref to gain[0][0][0] = 64 after channel.
        //   ch=1 band=0 sf=0 -> 64 + 1 = 65
        //   ch=1 band=0 sf=1 -> 65 + (-1) = 64
        //   reset ref to gain[1][0][0] = 65 after band 0.
        //   ch=1 band=1 sf=0 -> 65 + 0 = 65
        //   ch=1 band=1 sf=1 -> 65 + 1 = 66
        assert_eq!(g.gain_db(0, 0, 0), 0); // 64 -> 0 dB
        assert_eq!(g.gain_db(0, 1, 0), 1); // 65
        assert_eq!(g.gain_db(0, 0, 1), -1); // 63
        assert_eq!(g.gain_db(0, 1, 1), -1); // 63
        assert_eq!(g.gain_db(1, 0, 0), 1); // 65
        assert_eq!(g.gain_db(1, 1, 0), 0); // 64
        assert_eq!(g.gain_db(1, 0, 1), 1); // 65
        assert_eq!(g.gain_db(1, 1, 1), 2); // 66
    }

    #[test]
    fn parse_drc_compression_curve_minimal() {
        // Minimal compression curve: nullband_low=0, nullband_high=0,
        // gain_max_boost=0 (no boost section), gain_max_cut=0 (no cut
        // section), tc_default_flag=1 (no time-constants block).
        let mut bw = BitWriter::new();
        bw.write_u32(0, 4); // drc_lev_nullband_low
        bw.write_u32(0, 4); // drc_lev_nullband_high
        bw.write_u32(0, 4); // drc_gain_max_boost
        bw.write_u32(0, 5); // drc_gain_max_cut
        bw.write_bit(true); // drc_tc_default_flag
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let curve = parse_drc_compression_curve(&mut br).unwrap();
        assert_eq!(curve.drc_lev_nullband_low, 0);
        assert_eq!(curve.drc_gain_max_boost, 0);
        assert!(curve.drc_lev_max_boost.is_none());
        assert!(curve.time_constants.is_none());
        assert!(curve.drc_tc_default_flag);
    }

    #[test]
    fn parse_drc_compression_curve_full_with_boost_and_cut() {
        // Boost section + cut section + adaptive smoothing.
        let mut bw = BitWriter::new();
        bw.write_u32(3, 4); // nullband_low
        bw.write_u32(5, 4); // nullband_high
        bw.write_u32(7, 4); // gain_max_boost > 0
        bw.write_u32(15, 5); // lev_max_boost
        bw.write_u32(1, 1); // nr_boost_sections > 0
        bw.write_u32(9, 4); // gain_section_boost
        bw.write_u32(20, 5); // lev_section_boost
        bw.write_u32(11, 5); // gain_max_cut > 0
        bw.write_u32(33, 6); // lev_max_cut
        bw.write_u32(1, 1); // nr_cut_sections > 0
        bw.write_u32(13, 5); // gain_section_cut
        bw.write_u32(17, 5); // lev_section_cut
        bw.write_bit(false); // tc_default_flag = 0
        bw.write_u32(20, 8); // tc_attack
        bw.write_u32(75, 8); // tc_release
        bw.write_u32(2, 8); // tc_attack_fast
        bw.write_u32(50, 8); // tc_release_fast
        bw.write_bit(true); // adaptive_smoothing_flag
        bw.write_u32(15, 5); // attack_threshold
        bw.write_u32(20, 5); // release_threshold
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let curve = parse_drc_compression_curve(&mut br).unwrap();
        assert_eq!(curve.drc_lev_nullband_low, 3);
        assert_eq!(curve.drc_gain_max_boost, 7);
        assert_eq!(curve.drc_lev_max_boost, Some(15));
        assert_eq!(curve.drc_nr_boost_sections, Some(1));
        assert_eq!(curve.drc_gain_section_boost, Some(9));
        assert_eq!(curve.drc_lev_section_boost, Some(20));
        assert_eq!(curve.drc_gain_max_cut, 11);
        assert_eq!(curve.drc_lev_max_cut, Some(33));
        let tc = curve.time_constants.as_ref().unwrap();
        assert_eq!(tc.drc_tc_attack, 20);
        assert!(tc.drc_adaptive_smoothing_flag);
        assert_eq!(tc.drc_attack_threshold, Some(15));
        assert_eq!(tc.drc_release_threshold, Some(20));
    }

    #[test]
    fn parse_drc_decoder_mode_repeat_profile() {
        // Two-mode config: mode 0 carries a curve, mode 1 repeats it
        // via drc_repeat_profile_flag.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_drc_present
                            // drc_config:
        bw.write_u32(1, 3); // drc_decoder_nr_modes = 1 -> 2 modes

        // Mode 0: id=0, default profile (implicit curve flag).
        bw.write_u32(0, 3); // mode_id
        bw.write_bit(false); // drc_repeat_profile_flag
        bw.write_bit(true); // drc_default_profile_flag

        // Mode 1: id=1, repeat profile of mode 0.
        bw.write_u32(1, 3); // mode_id
        bw.write_bit(true); // drc_repeat_profile_flag
        bw.write_u32(0, 3); // drc_repeat_id = 0 -> mode 0

        bw.write_u32(0, 3); // drc_eac3_profile
                            // drc_data: both modes have curve_flag = 1 (implicit), so
                            // gainsets are empty; just the curve_present trailer.
        bw.write_bit(true); // drc_reset_flag
        bw.write_u32(2, 2); // drc_reserved
        bw.align_to_byte();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let info = DrcChannelInfo::new(1, 1);
        let frame = parse_drc_frame(&mut br, true, info, None).unwrap();
        let cfg = frame.config.unwrap();
        assert_eq!(cfg.nr_modes(), 2);
        assert!(cfg.modes[0].drc_compression_curve_flag);
        assert!(cfg.modes[1].drc_repeat_profile_flag);
        assert_eq!(cfg.modes[1].drc_repeat_id, Some(0));
        assert!(cfg.modes[1].drc_compression_curve_flag); // inherited
        let data = frame.data.unwrap();
        assert!(data.curve_present);
        assert!(data.drc_reset_flag);
        assert_eq!(data.drc_reserved, 2);
    }

    #[test]
    fn parse_drc_frame_non_iframe_uses_prev_config() {
        // Non-I-frame: drc_frame skips drc_config(). Caller supplies
        // prev_config so the parser knows which modes carry gainsets.
        let drc_gain_val: u8 = 50;
        let drc_gainset_size: u32 = 2 + 7;

        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_drc_present
                            // No drc_config() because !b_iframe.
        bw.write_u32(drc_gainset_size, 6); // drc_gainset_size_value
        bw.write_bit(false); // b_more_bits
        bw.write_u32(0, 2); // drc_version
        bw.write_u32(drc_gain_val as u32, 7); // drc_gain_val
        bw.align_to_byte();
        let bytes = bw.finish();

        let prev = DrcConfig {
            drc_decoder_nr_modes: 0,
            drc_eac3_profile: 0,
            modes: vec![DrcDecoderMode {
                drc_decoder_mode_id: 0,
                drc_output_level_from: None,
                drc_output_level_to: None,
                drc_repeat_profile_flag: false,
                drc_repeat_id: None,
                drc_default_profile_flag: Some(false),
                drc_compression_curve_flag: false,
                compression_curve: None,
                drc_gains_config: Some(0),
            }],
        };

        let mut br = BitReader::new(&bytes);
        let info = DrcChannelInfo::new(1, 1);
        let frame = parse_drc_frame(&mut br, false, info, Some(&prev)).unwrap();
        assert!(frame.config.is_none(), "no config on non-I-frame");
        let data = frame.data.unwrap();
        assert_eq!(data.gainsets.len(), 1);
        let g = &data.gainsets[0];
        assert_eq!(g.drc_gain_val, 50);
        // gain[0][0][0] = 50 - 64 = -14 dB.
        assert_eq!(g.gain_db(0, 0, 0), -14);
    }

    #[test]
    fn parse_drc_frame_non_iframe_without_prev_config_errors() {
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let info = DrcChannelInfo::new(1, 1);
        assert!(parse_drc_frame(&mut br, false, info, None).is_err());
    }
}
