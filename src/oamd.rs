//! Object audio metadata (OAMD) — ETSI TS 103 190-2 §6.2.8 + §6.3.9.
//!
//! OAMD carries the per-object rendering metadata that accompanies the
//! object-audio substreams: timing/synchronization offsets
//! (`oamd_timing_data`, §6.2.8.2), per-object property update blocks
//! (`object_info_block` §6.2.8.5 with `object_basic_info` §6.2.8.6 and
//! `object_render_info` §6.2.8.7), the per-substream dynamic-data
//! wrappers (`oamd_dyndata_single` §6.2.8.3 / `oamd_dyndata_multi`
//! §6.2.8.4), the common data (`oamd_common_data` §6.2.8.1 with `trim`
//! §6.2.8.9 and `bed_render_info` §6.2.8.8), and the extended-precision
//! position refinements (`add_per_object_md` §6.2.8.10, `ext_prec_pos`
//! §6.2.8.11, `ext_prec_alt_pos` §6.2.8.12).
//!
//! Every parser has an exact bit-inverse writer so a decoded structure
//! round-trips to a parse-equivalent bitstream (the crate-wide metadata
//! symmetry convention). Value semantics are surfaced where the spec
//! defines them: sample offsets (Tables 125/126), ramp durations
//! (Tables 127-129), object gain (Tables 134-136), object priority
//! (§6.3.9.7.6), room-anchored positions (§6.3.9.8.4), alternative
//! gains (Table 131), and extended-precision offsets (Tables 146m-o).
//!
//! The `stereo_dmx_coeff()` call inside `bed_render_info()` (§6.2.8.8)
//! has no syntax box of its own; its field layout is the factored-out
//! form of the identical inline `b_stereo_dmx_coeff` block in
//! `custom_dmx_data()` (§6.2.9.2) and is handled by
//! [`crate::dmx_coeff`]. Because its LFE sub-block is gated on the
//! invoking context's LFE presence, `parse_bed_render_info` /
//! `parse_oamd_common_data` take a `bed_has_lfe` argument.

use crate::toc::{variable_bits, write_variable_bits};
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

// =====================================================================
// Object types (§4.2 + §6.3.2.8/6.3.2.10)
// =====================================================================

/// AC-4 object type — drives which OAMD fields exist for an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjType {
    /// Dynamic object — position via 3-D coordinates.
    Dyn,
    /// Bed object — position via speaker assignment.
    Bed,
    /// Intermediate-spatial-format object — stacked-ring assignment.
    Isf,
}

// =====================================================================
// oamd_timing_data (§6.2.8.2, semantics §6.3.9.3)
// =====================================================================

/// `oa_sample_offset_type` / `oa_sample_offset_code` resolution
/// (Tables 125 + 126).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OaSampleOffset {
    /// Type prefix `0b0` — offset is 0 samples.
    Zero,
    /// Type prefix `0b10` — Table 126 codeword. Code `0b0` = 16,
    /// `0b10` = 8, `0b11` = 24 samples.
    Code8,
    /// See [`OaSampleOffset::Code8`].
    Code16,
    /// See [`OaSampleOffset::Code8`].
    Code24,
    /// Type prefix `0b11` — explicit 5-bit `oa_sample_offset`.
    Explicit(u8),
}

impl OaSampleOffset {
    /// Resolved offset in audio samples.
    pub fn samples(self) -> u32 {
        match self {
            OaSampleOffset::Zero => 0,
            OaSampleOffset::Code8 => 8,
            OaSampleOffset::Code16 => 16,
            OaSampleOffset::Code24 => 24,
            OaSampleOffset::Explicit(v) => v as u32,
        }
    }
}

/// Table 129: `ramp_duration_table` → ramp duration in audio samples.
pub const RAMP_DURATION_TABLE: [u16; 16] = [
    32, 64, 128, 256, 320, 480, 1000, 1001, 1024, 1600, 1601, 1602, 1920, 2000, 2002, 2048,
];

/// `ramp_duration_code` resolution (Tables 127-129).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampDuration {
    /// Code `0b00` — 0 samples.
    Zero,
    /// Code `0b01` — 512 samples.
    D512,
    /// Code `0b10` — 1 536 samples.
    D1536,
    /// Code `0b11` + `b_use_ramp_table = 1` — Table 129 index.
    Table(u8),
    /// Code `0b11` + `b_use_ramp_table = 0` — explicit 11-bit value.
    Explicit(u16),
}

impl RampDuration {
    /// Resolved ramp duration in audio samples.
    pub fn samples(self) -> u32 {
        match self {
            RampDuration::Zero => 0,
            RampDuration::D512 => 512,
            RampDuration::D1536 => 1536,
            RampDuration::Table(i) => RAMP_DURATION_TABLE[(i & 0xF) as usize] as u32,
            RampDuration::Explicit(v) => v as u32,
        }
    }
}

/// One per-block timing entry inside `oamd_timing_data()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OamdTimingBlock {
    /// 6-bit `block_offset_factor` — per-block timing offset.
    pub block_offset_factor: u8,
    /// Resolved ramp duration.
    pub ramp_duration: RampDuration,
}

/// Parsed `oamd_timing_data()` (§6.2.8.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OamdTimingData {
    /// Resolved sample offset.
    pub sample_offset: OaSampleOffset,
    /// One entry per `object_info_block` (`num_obj_info_blocks`, 3 bits).
    pub blocks: Vec<OamdTimingBlock>,
}

impl OamdTimingData {
    /// `num_obj_info_blocks` — the per-object block count announced by
    /// this timing element.
    pub fn num_obj_info_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// Parse `oamd_timing_data()` per §6.2.8.2.
pub fn parse_oamd_timing_data(br: &mut BitReader<'_>) -> Result<OamdTimingData> {
    // oa_sample_offset_type prefix: 0b0 / 0b10 / 0b11 (Table 125).
    let sample_offset = if !br.read_bit()? {
        OaSampleOffset::Zero
    } else if !br.read_bit()? {
        // 0b10 → oa_sample_offset_code prefix (Table 126).
        if !br.read_bit()? {
            OaSampleOffset::Code16
        } else if !br.read_bit()? {
            OaSampleOffset::Code8
        } else {
            OaSampleOffset::Code24
        }
    } else {
        // 0b11 → explicit 5-bit offset.
        OaSampleOffset::Explicit(br.read_u32(5)? as u8)
    };
    let num_blocks = br.read_u32(3)?;
    let mut blocks = Vec::with_capacity(num_blocks as usize);
    for _ in 0..num_blocks {
        let block_offset_factor = br.read_u32(6)? as u8;
        let ramp_duration = match br.read_u32(2)? {
            0b00 => RampDuration::Zero,
            0b01 => RampDuration::D512,
            0b10 => RampDuration::D1536,
            _ => {
                if br.read_bit()? {
                    RampDuration::Table(br.read_u32(4)? as u8)
                } else {
                    RampDuration::Explicit(br.read_u32(11)? as u16)
                }
            }
        };
        blocks.push(OamdTimingBlock {
            block_offset_factor,
            ramp_duration,
        });
    }
    Ok(OamdTimingData {
        sample_offset,
        blocks,
    })
}

/// Write `oamd_timing_data()` — exact inverse of
/// [`parse_oamd_timing_data`].
pub fn write_oamd_timing_data(bw: &mut BitWriter, t: &OamdTimingData) -> Result<()> {
    match t.sample_offset {
        OaSampleOffset::Zero => bw.write_bit(false),
        OaSampleOffset::Code16 => {
            bw.write_u32(0b10, 2);
            bw.write_bit(false);
        }
        OaSampleOffset::Code8 => {
            bw.write_u32(0b10, 2);
            bw.write_u32(0b10, 2);
        }
        OaSampleOffset::Code24 => {
            bw.write_u32(0b10, 2);
            bw.write_u32(0b11, 2);
        }
        OaSampleOffset::Explicit(v) => {
            bw.write_u32(0b11, 2);
            bw.write_u32(v as u32, 5);
        }
    }
    if t.blocks.len() > 7 {
        return Err(Error::invalid("ac4: num_obj_info_blocks > 7"));
    }
    bw.write_u32(t.blocks.len() as u32, 3);
    for b in &t.blocks {
        bw.write_u32(b.block_offset_factor as u32, 6);
        match b.ramp_duration {
            RampDuration::Zero => bw.write_u32(0b00, 2),
            RampDuration::D512 => bw.write_u32(0b01, 2),
            RampDuration::D1536 => bw.write_u32(0b10, 2),
            RampDuration::Table(i) => {
                bw.write_u32(0b11, 2);
                bw.write_bit(true);
                bw.write_u32(i as u32, 4);
            }
            RampDuration::Explicit(v) => {
                bw.write_u32(0b11, 2);
                bw.write_bit(false);
                bw.write_u32(v as u32, 11);
            }
        }
    }
    Ok(())
}

// =====================================================================
// object_basic_info (§6.2.8.6, semantics §6.3.9.7)
// =====================================================================

/// `basic_info_md` prefix code (Table 134).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasicInfoMd {
    /// `0b0` — non-default gain, default priority.
    Gain,
    /// `0b10` — non-default gain, non-default priority.
    GainPriority,
    /// `0b11` — default gain, non-default priority.
    Priority,
}

/// `object_gain_code` resolution (Tables 135/136).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectGain {
    /// `0b0` + 6-bit `object_gain_value`: 0..=14 → `15 - v` dB,
    /// 15..=63 → `14 - v` dB.
    Value(u8),
    /// `0b10` — −∞ dB (mute).
    NegInf,
    /// `0b11` — reuse the previous object's gain.
    PrevObject,
}

impl ObjectGain {
    /// Gain in dB. `None` for −∞ / previous-object reuse.
    pub fn db(self) -> Option<i32> {
        match self {
            ObjectGain::Value(v) if v <= 14 => Some(15 - v as i32),
            ObjectGain::Value(v) => Some(14 - v as i32),
            _ => None,
        }
    }
}

/// Parsed `object_basic_info()` (§6.2.8.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectBasicInfo {
    /// `b_default_basic_info_md` — when true everything below is
    /// defaulted (0 dB gain, priority 1) and absent from the stream.
    pub b_default: bool,
    /// Coding selector (present when `b_default == false`).
    pub md: Option<BasicInfoMd>,
    /// Gain (present for [`BasicInfoMd::Gain`] / `GainPriority`).
    pub gain: Option<ObjectGain>,
    /// 5-bit `object_priority_code`; `object_priority = code / 31`.
    pub priority_code: Option<u8>,
}

impl Default for ObjectBasicInfo {
    fn default() -> Self {
        ObjectBasicInfo {
            b_default: true,
            md: None,
            gain: None,
            priority_code: None,
        }
    }
}

/// Parse `object_basic_info()` per §6.2.8.6.
pub fn parse_object_basic_info(br: &mut BitReader<'_>) -> Result<ObjectBasicInfo> {
    let b_default = br.read_bit()?;
    if b_default {
        return Ok(ObjectBasicInfo::default());
    }
    // basic_info_md prefix: 0b0 / 0b10 / 0b11 (Table 134).
    let md = if !br.read_bit()? {
        BasicInfoMd::Gain
    } else if !br.read_bit()? {
        BasicInfoMd::GainPriority
    } else {
        BasicInfoMd::Priority
    };
    let gain = if matches!(md, BasicInfoMd::Gain | BasicInfoMd::GainPriority) {
        // object_gain_code prefix: 0b0 / 0b10 / 0b11 (Table 135).
        Some(if !br.read_bit()? {
            ObjectGain::Value(br.read_u32(6)? as u8)
        } else if !br.read_bit()? {
            ObjectGain::NegInf
        } else {
            ObjectGain::PrevObject
        })
    } else {
        None
    };
    let priority_code = if matches!(md, BasicInfoMd::GainPriority | BasicInfoMd::Priority) {
        Some(br.read_u32(5)? as u8)
    } else {
        None
    };
    Ok(ObjectBasicInfo {
        b_default,
        md: Some(md),
        gain,
        priority_code,
    })
}

/// Write `object_basic_info()` — exact inverse of
/// [`parse_object_basic_info`].
pub fn write_object_basic_info(bw: &mut BitWriter, b: &ObjectBasicInfo) -> Result<()> {
    bw.write_bit(b.b_default);
    if b.b_default {
        return Ok(());
    }
    let md =
        b.md.ok_or_else(|| Error::invalid("ac4: non-default object_basic_info without md"))?;
    match md {
        BasicInfoMd::Gain => bw.write_bit(false),
        BasicInfoMd::GainPriority => bw.write_u32(0b10, 2),
        BasicInfoMd::Priority => bw.write_u32(0b11, 2),
    }
    if matches!(md, BasicInfoMd::Gain | BasicInfoMd::GainPriority) {
        match b
            .gain
            .ok_or_else(|| Error::invalid("ac4: basic_info_md expects a gain"))?
        {
            ObjectGain::Value(v) => {
                bw.write_bit(false);
                bw.write_u32(v as u32, 6);
            }
            ObjectGain::NegInf => bw.write_u32(0b10, 2),
            ObjectGain::PrevObject => bw.write_u32(0b11, 2),
        }
    }
    if matches!(md, BasicInfoMd::GainPriority | BasicInfoMd::Priority) {
        let p = b
            .priority_code
            .ok_or_else(|| Error::invalid("ac4: basic_info_md expects a priority"))?;
        bw.write_u32(p as u32, 5);
    }
    Ok(())
}

// =====================================================================
// object_render_info (§6.2.8.7, semantics §6.3.9.8)
// =====================================================================

/// Room-anchored position payload (§6.3.9.8.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPosition {
    /// Differential update: 3-bit two's-complement deltas applied to the
    /// previous block's standard-precision position (÷62, ÷62, ÷15).
    Diff {
        /// `diff_pos3D_X` (−4..=3).
        x: i8,
        /// `diff_pos3D_Y` (−4..=3).
        y: i8,
        /// `diff_pos3D_Z` (−4..=3).
        z: i8,
    },
    /// Absolute position: `x/62`, `y/62`, `sign · z/15`.
    Abs {
        /// 6-bit `pos3D_X`.
        x: u8,
        /// 6-bit `pos3D_Y`.
        y: u8,
        /// `pos3D_Z_sign` — true = +1.
        z_sign: bool,
        /// 4-bit `pos3D_Z`.
        z: u8,
    },
}

/// Zone-mask section of `object_render_info()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderZone {
    /// `b_grouped_zone_defaults` — true = all defaults, nothing follows.
    pub b_grouped_defaults: bool,
    /// 3-bit `group_zone_mask` (present when not defaulted).
    pub group_zone_mask: Option<u8>,
    /// 3-bit `zone_mask` (present when `group_zone_mask & 0b001`).
    pub zone_mask: Option<u8>,
}

/// Object-width subsection (`group_other_mask & 0b0001`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectWidth {
    /// `object_width_mode == 0` — single 5-bit code.
    Uniform(u8),
    /// `object_width_mode == 1` — per-axis 5-bit codes.
    PerAxis {
        /// `object_width_X_code`.
        x: u8,
        /// `object_width_Y_code`.
        y: u8,
        /// `object_width_Z_code`.
        z: u8,
    },
}

/// Divergence subsection (`group_other_mask & 0b1000`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectDivergence {
    /// `object_div_mode == 0b00` — 2-bit table index.
    Table(u8),
    /// `object_div_mode == 0b01` — no further payload.
    Mode01,
    /// `object_div_mode & 0b10` — 6-bit code (modes 0b10 / 0b11).
    Code {
        /// The 2-bit mode actually coded (0b10 or 0b11).
        mode: u8,
        /// `object_div_code`.
        code: u8,
    },
}

/// Other-properties section of `object_render_info()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderOtherProps {
    /// `b_grouped_other_defaults` — true = all defaults, nothing follows.
    pub b_grouped_defaults: bool,
    /// 4-bit `group_other_mask` (present when not defaulted).
    pub group_other_mask: Option<u8>,
    /// Width subsection (`mask & 0b0001`).
    pub width: Option<ObjectWidth>,
    /// `object_screen_factor_code` (3) + `object_depth_factor` (2)
    /// (`mask & 0b0010`).
    pub screen: Option<(u8, u8)>,
    /// Distance subsection (`mask & 0b0100`): `None` inside `Some` =
    /// object at infinity, else 4-bit `obj_distance_factor_code`.
    pub distance: Option<Option<u8>>,
    /// Divergence subsection (`mask & 0b1000`).
    pub divergence: Option<ObjectDivergence>,
}

/// Parsed `object_render_info()` (§6.2.8.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectRenderInfo {
    /// True when invoked with `object_render_info_status == ALL_NEW`
    /// (the three presence flags are implicit and absent).
    pub all_new: bool,
    /// `b_obj_render_position_present` (implicit true when `all_new`).
    pub b_position_present: bool,
    /// `b_obj_render_zone_present` (implicit true when `all_new`).
    pub b_zone_present: bool,
    /// `b_obj_render_otherprops_present` (implicit true when `all_new`).
    pub b_otherprops_present: bool,
    /// Position payload.
    pub position: Option<RenderPosition>,
    /// Zone payload.
    pub zone: Option<RenderZone>,
    /// Other-properties payload.
    pub otherprops: Option<RenderOtherProps>,
}

/// Parse `object_render_info(status, b_no_delta)` per §6.2.8.7.
///
/// `all_new` mirrors `object_render_info_status == ALL_NEW` (the only
/// other status reaching this element is `PART_REUSE`).
pub fn parse_object_render_info(
    br: &mut BitReader<'_>,
    all_new: bool,
    b_no_delta: bool,
) -> Result<ObjectRenderInfo> {
    let (b_position_present, b_zone_present, b_otherprops_present) = if all_new {
        (true, true, true)
    } else {
        // object_render_info_mask section — transmitted in the order
        // otherprops, zone, position per the syntax box.
        let other = br.read_bit()?;
        let zone = br.read_bit()?;
        let pos = br.read_bit()?;
        (pos, zone, other)
    };
    let position = if b_position_present {
        let b_diff = if b_no_delta { false } else { br.read_bit()? };
        Some(if b_diff {
            let sx = br.read_u32(3)? as i8;
            let sy = br.read_u32(3)? as i8;
            let sz = br.read_u32(3)? as i8;
            // 3-bit two's complement.
            let s = |v: i8| if v >= 4 { v - 8 } else { v };
            RenderPosition::Diff {
                x: s(sx),
                y: s(sy),
                z: s(sz),
            }
        } else {
            RenderPosition::Abs {
                x: br.read_u32(6)? as u8,
                y: br.read_u32(6)? as u8,
                z_sign: br.read_bit()?,
                z: br.read_u32(4)? as u8,
            }
        })
    } else {
        None
    };
    let zone = if b_zone_present {
        let b_grouped_defaults = br.read_bit()?;
        if b_grouped_defaults {
            Some(RenderZone {
                b_grouped_defaults,
                group_zone_mask: None,
                zone_mask: None,
            })
        } else {
            let mask = br.read_u32(3)? as u8;
            let zone_mask = if mask & 0b001 != 0 {
                Some(br.read_u32(3)? as u8)
            } else {
                None
            };
            Some(RenderZone {
                b_grouped_defaults,
                group_zone_mask: Some(mask),
                zone_mask,
            })
        }
    } else {
        None
    };
    let otherprops = if b_otherprops_present {
        let b_grouped_defaults = br.read_bit()?;
        if b_grouped_defaults {
            Some(RenderOtherProps {
                b_grouped_defaults,
                ..Default::default()
            })
        } else {
            let mask = br.read_u32(4)? as u8;
            let width = if mask & 0b0001 != 0 {
                Some(if !br.read_bit()? {
                    ObjectWidth::Uniform(br.read_u32(5)? as u8)
                } else {
                    ObjectWidth::PerAxis {
                        x: br.read_u32(5)? as u8,
                        y: br.read_u32(5)? as u8,
                        z: br.read_u32(5)? as u8,
                    }
                })
            } else {
                None
            };
            let screen = if mask & 0b0010 != 0 {
                Some((br.read_u32(3)? as u8, br.read_u32(2)? as u8))
            } else {
                None
            };
            let distance = if mask & 0b0100 != 0 {
                Some(if br.read_bit()? {
                    None // object at infinity
                } else {
                    Some(br.read_u32(4)? as u8)
                })
            } else {
                None
            };
            let divergence = if mask & 0b1000 != 0 {
                let mode = br.read_u32(2)? as u8;
                Some(match mode {
                    0b00 => ObjectDivergence::Table(br.read_u32(2)? as u8),
                    0b01 => ObjectDivergence::Mode01,
                    _ => ObjectDivergence::Code {
                        mode,
                        code: br.read_u32(6)? as u8,
                    },
                })
            } else {
                None
            };
            Some(RenderOtherProps {
                b_grouped_defaults,
                group_other_mask: Some(mask),
                width,
                screen,
                distance,
                divergence,
            })
        }
    } else {
        None
    };
    Ok(ObjectRenderInfo {
        all_new,
        b_position_present,
        b_zone_present,
        b_otherprops_present,
        position,
        zone,
        otherprops,
    })
}

/// Write `object_render_info()` — exact inverse of
/// [`parse_object_render_info`].
pub fn write_object_render_info(
    bw: &mut BitWriter,
    r: &ObjectRenderInfo,
    b_no_delta: bool,
) -> Result<()> {
    if !r.all_new {
        bw.write_bit(r.b_otherprops_present);
        bw.write_bit(r.b_zone_present);
        bw.write_bit(r.b_position_present);
    }
    if r.b_position_present {
        let pos = r
            .position
            .ok_or_else(|| Error::invalid("ac4: render position announced but missing"))?;
        match pos {
            RenderPosition::Diff { x, y, z } => {
                if b_no_delta {
                    return Err(Error::invalid("ac4: diff position on a no-delta block"));
                }
                bw.write_bit(true);
                for v in [x, y, z] {
                    bw.write_u32((v & 0x7) as u32, 3);
                }
            }
            RenderPosition::Abs { x, y, z_sign, z } => {
                if !b_no_delta {
                    bw.write_bit(false);
                }
                bw.write_u32(x as u32, 6);
                bw.write_u32(y as u32, 6);
                bw.write_bit(z_sign);
                bw.write_u32(z as u32, 4);
            }
        }
    }
    if r.b_zone_present {
        let zone = r
            .zone
            .ok_or_else(|| Error::invalid("ac4: render zone announced but missing"))?;
        bw.write_bit(zone.b_grouped_defaults);
        if !zone.b_grouped_defaults {
            let mask = zone
                .group_zone_mask
                .ok_or_else(|| Error::invalid("ac4: zone mask missing"))?;
            bw.write_u32(mask as u32, 3);
            if mask & 0b001 != 0 {
                let zm = zone
                    .zone_mask
                    .ok_or_else(|| Error::invalid("ac4: zone_mask missing"))?;
                bw.write_u32(zm as u32, 3);
            }
        }
    }
    if r.b_otherprops_present {
        let op = r
            .otherprops
            .ok_or_else(|| Error::invalid("ac4: render otherprops announced but missing"))?;
        bw.write_bit(op.b_grouped_defaults);
        if !op.b_grouped_defaults {
            let mask = op
                .group_other_mask
                .ok_or_else(|| Error::invalid("ac4: other mask missing"))?;
            bw.write_u32(mask as u32, 4);
            if mask & 0b0001 != 0 {
                match op
                    .width
                    .ok_or_else(|| Error::invalid("ac4: width missing"))?
                {
                    ObjectWidth::Uniform(w) => {
                        bw.write_bit(false);
                        bw.write_u32(w as u32, 5);
                    }
                    ObjectWidth::PerAxis { x, y, z } => {
                        bw.write_bit(true);
                        bw.write_u32(x as u32, 5);
                        bw.write_u32(y as u32, 5);
                        bw.write_u32(z as u32, 5);
                    }
                }
            }
            if mask & 0b0010 != 0 {
                let (sf, df) = op
                    .screen
                    .ok_or_else(|| Error::invalid("ac4: screen factors missing"))?;
                bw.write_u32(sf as u32, 3);
                bw.write_u32(df as u32, 2);
            }
            if mask & 0b0100 != 0 {
                match op
                    .distance
                    .ok_or_else(|| Error::invalid("ac4: distance missing"))?
                {
                    None => bw.write_bit(true),
                    Some(code) => {
                        bw.write_bit(false);
                        bw.write_u32(code as u32, 4);
                    }
                }
            }
            if mask & 0b1000 != 0 {
                match op
                    .divergence
                    .ok_or_else(|| Error::invalid("ac4: divergence missing"))?
                {
                    ObjectDivergence::Table(t) => {
                        bw.write_u32(0b00, 2);
                        bw.write_u32(t as u32, 2);
                    }
                    ObjectDivergence::Mode01 => bw.write_u32(0b01, 2),
                    ObjectDivergence::Code { mode, code } => {
                        if mode & 0b10 == 0 {
                            return Err(Error::invalid("ac4: divergence code needs mode 0b1x"));
                        }
                        bw.write_u32(mode as u32, 2);
                        bw.write_u32(code as u32, 6);
                    }
                }
            }
        }
    }
    Ok(())
}

// =====================================================================
// ext_prec_pos (§6.2.8.11) + add_per_object_md (§6.2.8.10)
// =====================================================================

/// Table 146m-o: 2-bit extended-precision offset value → signed step.
pub fn ext_prec_value(code: u8) -> i32 {
    match code & 0b11 {
        0b00 => 1,
        0b01 => 2,
        0b10 => -1,
        _ => -2,
    }
}

/// Parsed `ext_prec_pos()` (§6.2.8.11) — per-axis 2-bit refinement
/// codes, present per the 3-bit presence array (index 2 = X, 1 = Y,
/// 0 = Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtPrecPos {
    /// `ext_prec_pos3D_X` code.
    pub x: Option<u8>,
    /// `ext_prec_pos3D_Y` code.
    pub y: Option<u8>,
    /// `ext_prec_pos3D_Z` code.
    pub z: Option<u8>,
}

/// Parse `ext_prec_pos()` per §6.2.8.11.
pub fn parse_ext_prec_pos(br: &mut BitReader<'_>) -> Result<ExtPrecPos> {
    let presence = br.read_u32(3)?;
    let x = if presence & 0b100 != 0 {
        Some(br.read_u32(2)? as u8)
    } else {
        None
    };
    let y = if presence & 0b010 != 0 {
        Some(br.read_u32(2)? as u8)
    } else {
        None
    };
    let z = if presence & 0b001 != 0 {
        Some(br.read_u32(2)? as u8)
    } else {
        None
    };
    Ok(ExtPrecPos { x, y, z })
}

/// Write `ext_prec_pos()` — exact inverse of [`parse_ext_prec_pos`].
pub fn write_ext_prec_pos(bw: &mut BitWriter, e: &ExtPrecPos) -> Result<()> {
    let mut presence = 0u32;
    if e.x.is_some() {
        presence |= 0b100;
    }
    if e.y.is_some() {
        presence |= 0b010;
    }
    if e.z.is_some() {
        presence |= 0b001;
    }
    bw.write_u32(presence, 3);
    for v in [e.x, e.y, e.z].into_iter().flatten() {
        bw.write_u32(v as u32, 2);
    }
    Ok(())
}

/// Parsed `add_per_object_md()` (§6.2.8.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AddPerObjectMd {
    /// `b_obj_trim_disable`.
    pub b_obj_trim_disable: bool,
    /// Extended-precision refinement — only reachable for active
    /// dynamic objects; `None` when the gate bits were absent or 0.
    pub ext_prec_pos: Option<ExtPrecPos>,
}

/// Parse `add_per_object_md(b_object_not_active, b_dynamic_object)` per
/// §6.2.8.10.
pub fn parse_add_per_object_md(
    br: &mut BitReader<'_>,
    b_object_not_active: bool,
    b_dynamic_object: bool,
) -> Result<AddPerObjectMd> {
    let b_obj_trim_disable = br.read_bit()?;
    let ext_prec_pos = if !b_object_not_active && b_dynamic_object && br.read_bit()? {
        Some(parse_ext_prec_pos(br)?)
    } else {
        None
    };
    Ok(AddPerObjectMd {
        b_obj_trim_disable,
        ext_prec_pos,
    })
}

/// Write `add_per_object_md()` — exact inverse of
/// [`parse_add_per_object_md`].
pub fn write_add_per_object_md(
    bw: &mut BitWriter,
    md: &AddPerObjectMd,
    b_object_not_active: bool,
    b_dynamic_object: bool,
) -> Result<()> {
    bw.write_bit(md.b_obj_trim_disable);
    if !b_object_not_active && b_dynamic_object {
        match &md.ext_prec_pos {
            Some(e) => {
                bw.write_bit(true);
                write_ext_prec_pos(bw, e)?;
            }
            None => bw.write_bit(false),
        }
    } else if md.ext_prec_pos.is_some() {
        return Err(Error::invalid(
            "ac4: ext_prec_pos only valid for active dynamic objects",
        ));
    }
    Ok(())
}

// =====================================================================
// object_info_block (§6.2.8.5, semantics §6.3.9.6)
// =====================================================================

/// `object_basic_info_status` / `object_render_info_status` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoStatus {
    /// Metadata defaulted (inactive object / non-dynamic render info).
    Default,
    /// Freshly transmitted.
    AllNew,
    /// Fully reused from the previous block.
    Reuse,
    /// Partially reused (render info only).
    PartReuse,
}

/// Parsed `object_info_block()` (§6.2.8.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectInfoBlock {
    /// `b_object_not_active`.
    pub b_object_not_active: bool,
    /// Resolved `object_basic_info_status`.
    pub basic_status: InfoStatus,
    /// `object_basic_info()` payload (status `AllNew` only).
    pub basic_info: Option<ObjectBasicInfo>,
    /// Resolved `object_render_info_status`.
    pub render_status: InfoStatus,
    /// `object_render_info()` payload (status `AllNew` / `PartReuse`).
    pub render_info: Option<ObjectRenderInfo>,
    /// `b_add_table_data` payload: per-object additional metadata plus
    /// the number of announced-but-skipped trailing bits.
    pub add_table_data: Option<(AddPerObjectMd, u32)>,
}

/// Parse `object_info_block(b_no_delta, b_dynamic_object)` per §6.2.8.5.
pub fn parse_object_info_block(
    br: &mut BitReader<'_>,
    b_no_delta: bool,
    b_dynamic_object: bool,
) -> Result<ObjectInfoBlock> {
    let b_object_not_active = br.read_bit()?;
    let basic_status = if b_object_not_active {
        InfoStatus::Default
    } else if b_no_delta {
        InfoStatus::AllNew
    } else if br.read_bit()? {
        InfoStatus::Reuse
    } else {
        InfoStatus::AllNew
    };
    let basic_info = if basic_status == InfoStatus::AllNew {
        Some(parse_object_basic_info(br)?)
    } else {
        None
    };
    let render_status = if b_object_not_active {
        InfoStatus::Default
    } else if b_dynamic_object {
        if b_no_delta {
            InfoStatus::AllNew
        } else if br.read_bit()? {
            InfoStatus::Reuse
        } else if br.read_bit()? {
            InfoStatus::PartReuse
        } else {
            InfoStatus::AllNew
        }
    } else {
        InfoStatus::Default
    };
    let render_info = if matches!(render_status, InfoStatus::AllNew | InfoStatus::PartReuse) {
        Some(parse_object_render_info(
            br,
            render_status == InfoStatus::AllNew,
            b_no_delta,
        )?)
    } else {
        None
    };
    let add_table_data = if br.read_bit()? {
        let atd_size = br.read_u32(4)? + 1;
        let start = br.bit_position();
        let md = parse_add_per_object_md(br, b_object_not_active, b_dynamic_object)?;
        let used = (br.bit_position() - start) as u32;
        let total = 8 * atd_size;
        if used > total {
            return Err(Error::invalid(
                "ac4: add_per_object_md exceeded add_table_data envelope",
            ));
        }
        let remain = total - used;
        for _ in 0..remain {
            let _ = br.read_bit()?;
        }
        Some((md, remain))
    } else {
        None
    };
    Ok(ObjectInfoBlock {
        b_object_not_active,
        basic_status,
        basic_info,
        render_status,
        render_info,
        add_table_data,
    })
}

/// Write `object_info_block()` — exact inverse of
/// [`parse_object_info_block`].
pub fn write_object_info_block(
    bw: &mut BitWriter,
    blk: &ObjectInfoBlock,
    b_no_delta: bool,
    b_dynamic_object: bool,
) -> Result<()> {
    bw.write_bit(blk.b_object_not_active);
    if !blk.b_object_not_active && !b_no_delta {
        bw.write_bit(blk.basic_status == InfoStatus::Reuse);
    }
    if blk.basic_status == InfoStatus::AllNew {
        let bi = blk
            .basic_info
            .as_ref()
            .ok_or_else(|| Error::invalid("ac4: ALL_NEW basic info missing"))?;
        write_object_basic_info(bw, bi)?;
    }
    if !blk.b_object_not_active && b_dynamic_object && !b_no_delta {
        match blk.render_status {
            InfoStatus::Reuse => bw.write_bit(true),
            InfoStatus::PartReuse => {
                bw.write_bit(false);
                bw.write_bit(true);
            }
            InfoStatus::AllNew => {
                bw.write_bit(false);
                bw.write_bit(false);
            }
            InfoStatus::Default => {
                return Err(Error::invalid(
                    "ac4: DEFAULT render status on an active dynamic object",
                ))
            }
        }
    }
    if matches!(
        blk.render_status,
        InfoStatus::AllNew | InfoStatus::PartReuse
    ) {
        let ri = blk
            .render_info
            .as_ref()
            .ok_or_else(|| Error::invalid("ac4: render info missing"))?;
        write_object_render_info(bw, ri, b_no_delta)?;
    }
    match &blk.add_table_data {
        None => bw.write_bit(false),
        Some((md, remain)) => {
            bw.write_bit(true);
            // Re-derive atd_size from the payload width + recorded
            // filler.
            let mut probe = BitWriter::new();
            write_add_per_object_md(&mut probe, md, blk.b_object_not_active, b_dynamic_object)?;
            let used = probe.bit_position() as u32;
            let total = used + remain;
            if total == 0 || total % 8 != 0 || total > 16 * 8 {
                return Err(Error::invalid(
                    "ac4: add_table_data payload + filler is not a valid byte envelope",
                ));
            }
            bw.write_u32(total / 8 - 1, 4);
            write_add_per_object_md(bw, md, blk.b_object_not_active, b_dynamic_object)?;
            for _ in 0..*remain {
                bw.write_bit(false);
            }
        }
    }
    Ok(())
}

// =====================================================================
// oamd_dyndata_single / _multi (§6.2.8.3 / §6.2.8.4)
// =====================================================================

/// Alternative-set per-data-point payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AltDataPoint {
    /// 6-bit `alt_obj_gain` (`b_alt_gain` gate). Table 131:
    /// 0..=62 → `14 − alt_gain` dB, 63 → −∞.
    pub alt_gain: Option<u8>,
    /// Alternative position (DYN non-LFE objects only).
    pub alt_position: Option<(u8, u8, bool, u8)>,
}

/// One alternative data set inside `oamd_dyndata_single()`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AltDataSet {
    /// `b_keep` — previous alternative values still valid.
    pub b_keep: bool,
    /// `b_common_data` (present when `!b_keep` and `obj_type[0] != ISF`).
    pub b_common_data: Option<bool>,
    /// Per-data-point payloads (1 when common/ISF, else `n_objs`).
    pub points: Vec<AltDataPoint>,
    /// `b_additional_data` payload: parsed `ext_prec_alt_pos` (one entry
    /// per DYN non-LFE object when `b_keep == 0`) + skipped filler bits.
    pub additional: Option<(Vec<Option<ExtPrecPos>>, u32)>,
}

/// Alternative-properties tail of `oamd_dyndata_single()`
/// (`b_alternative == 1`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AltProperties {
    /// `b_ducking_disabled`.
    pub b_ducking_disabled: bool,
    /// `object_sound_category` (2-bit + `variable_bits(2)` escape).
    pub object_sound_category: u32,
    /// The alternative data sets (2-bit + `variable_bits(2)` escape).
    pub sets: Vec<AltDataSet>,
}

/// Parsed `oamd_dyndata_single()` (§6.2.8.3).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OamdDynDataSingle {
    /// `object_info_block` grid — `blocks[obj][blk]`.
    pub object_blocks: Vec<Vec<ObjectInfoBlock>>,
    /// Alternative-properties tail (present when `b_alternative`).
    pub alt: Option<AltProperties>,
}

fn is_dynamic(obj_type: ObjType, b_lfe: bool) -> bool {
    obj_type == ObjType::Dyn && !b_lfe
}

/// Parse `oamd_dyndata_single(n_objs, n_blocks, b_iframe,
/// b_alternative, obj_type[], b_lfe[])` per §6.2.8.3.
pub fn parse_oamd_dyndata_single(
    br: &mut BitReader<'_>,
    n_blocks: usize,
    b_iframe: bool,
    b_alternative: bool,
    obj_type: &[ObjType],
    b_lfe: &[bool],
) -> Result<OamdDynDataSingle> {
    if obj_type.len() != b_lfe.len() {
        return Err(Error::invalid("ac4: obj_type / b_lfe length mismatch"));
    }
    let n_objs = obj_type.len();
    let mut object_blocks = Vec::with_capacity(n_objs);
    for i in 0..n_objs {
        let b_dynamic_object = is_dynamic(obj_type[i], b_lfe[i]);
        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let b_no_delta = b_iframe && b == 0;
            blocks.push(parse_object_info_block(br, b_no_delta, b_dynamic_object)?);
        }
        object_blocks.push(blocks);
    }
    let alt = if b_alternative {
        let b_ducking_disabled = br.read_bit()?;
        let mut object_sound_category = br.read_u32(2)?;
        if object_sound_category == 3 {
            object_sound_category = object_sound_category
                .checked_add(variable_bits(br, 2)?)
                .ok_or_else(|| {
                    oxideav_core::Error::invalid("ac4: variable_bits escape overflow")
                })?;
        }
        let mut n_alt_data_sets = br.read_u32(2)?;
        if n_alt_data_sets == 3 {
            n_alt_data_sets = n_alt_data_sets
                .checked_add(variable_bits(br, 2)?)
                .ok_or_else(|| {
                    oxideav_core::Error::invalid("ac4: variable_bits escape overflow")
                })?;
        }
        let mut sets = Vec::with_capacity(n_alt_data_sets as usize);
        for _ in 0..n_alt_data_sets {
            let b_keep = br.read_bit()?;
            let mut b_common_data = None;
            let mut points = Vec::new();
            if !b_keep {
                let n_data_points = if obj_type.first() == Some(&ObjType::Isf) {
                    1
                } else {
                    let common = br.read_bit()?;
                    b_common_data = Some(common);
                    if common {
                        1
                    } else {
                        n_objs
                    }
                };
                for dp in 0..n_data_points {
                    let ot = obj_type[dp.min(n_objs.saturating_sub(1))];
                    let lfe = b_lfe[dp.min(n_objs.saturating_sub(1))];
                    let mut point = AltDataPoint::default();
                    match ot {
                        ObjType::Bed | ObjType::Isf => {
                            if br.read_bit()? {
                                point.alt_gain = Some(br.read_u32(6)? as u8);
                            }
                        }
                        ObjType::Dyn => {
                            if br.read_bit()? {
                                point.alt_gain = Some(br.read_u32(6)? as u8);
                            }
                            if !lfe && br.read_bit()? {
                                point.alt_position = Some((
                                    br.read_u32(6)? as u8,
                                    br.read_u32(6)? as u8,
                                    br.read_bit()?,
                                    br.read_u32(4)? as u8,
                                ));
                            }
                        }
                    }
                    points.push(point);
                }
            }
            let additional = if br.read_bit()? {
                let total = variable_bits(br, 2)?
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(8))
                    .ok_or_else(|| Error::invalid("ac4: oamd extension size overflow"))?;
                let start = br.bit_position();
                // ext_prec_alt_pos(n_objs, b_keep, obj_type, b_lfe).
                let mut per_obj = Vec::with_capacity(n_objs);
                if !b_keep {
                    for i in 0..n_objs {
                        if is_dynamic(obj_type[i], b_lfe[i]) {
                            if br.read_bit()? {
                                per_obj.push(Some(parse_ext_prec_pos(br)?));
                            } else {
                                per_obj.push(None);
                            }
                        }
                    }
                }
                let used = (br.bit_position() - start) as u32;
                if used > total {
                    return Err(Error::invalid(
                        "ac4: ext_prec_alt_pos exceeded additional-data envelope",
                    ));
                }
                let remain = total - used;
                for _ in 0..remain {
                    let _ = br.read_bit()?;
                }
                Some((per_obj, remain))
            } else {
                None
            };
            sets.push(AltDataSet {
                b_keep,
                b_common_data,
                points,
                additional,
            });
        }
        Some(AltProperties {
            b_ducking_disabled,
            object_sound_category,
            sets,
        })
    } else {
        None
    };
    Ok(OamdDynDataSingle { object_blocks, alt })
}

/// Write `oamd_dyndata_single()` — exact inverse of
/// [`parse_oamd_dyndata_single`].
pub fn write_oamd_dyndata_single(
    bw: &mut BitWriter,
    d: &OamdDynDataSingle,
    b_iframe: bool,
    obj_type: &[ObjType],
    b_lfe: &[bool],
) -> Result<()> {
    if d.object_blocks.len() != obj_type.len() || obj_type.len() != b_lfe.len() {
        return Err(Error::invalid("ac4: dyndata object count mismatch"));
    }
    for (i, blocks) in d.object_blocks.iter().enumerate() {
        let b_dynamic_object = is_dynamic(obj_type[i], b_lfe[i]);
        for (b, blk) in blocks.iter().enumerate() {
            let b_no_delta = b_iframe && b == 0;
            write_object_info_block(bw, blk, b_no_delta, b_dynamic_object)?;
        }
    }
    if let Some(alt) = &d.alt {
        bw.write_bit(alt.b_ducking_disabled);
        if alt.object_sound_category >= 3 {
            bw.write_u32(3, 2);
            write_variable_bits(bw, 2, alt.object_sound_category - 3);
        } else {
            bw.write_u32(alt.object_sound_category, 2);
        }
        let n_sets = alt.sets.len() as u32;
        if n_sets >= 3 {
            bw.write_u32(3, 2);
            write_variable_bits(bw, 2, n_sets - 3);
        } else {
            bw.write_u32(n_sets, 2);
        }
        for set in &alt.sets {
            bw.write_bit(set.b_keep);
            if !set.b_keep {
                if obj_type.first() != Some(&ObjType::Isf) {
                    let common = set
                        .b_common_data
                        .ok_or_else(|| Error::invalid("ac4: b_common_data missing"))?;
                    bw.write_bit(common);
                }
                for (dp, point) in set.points.iter().enumerate() {
                    let idx = dp.min(obj_type.len().saturating_sub(1));
                    let ot = obj_type[idx];
                    let lfe = b_lfe[idx];
                    match point.alt_gain {
                        Some(g) => {
                            bw.write_bit(true);
                            bw.write_u32(g as u32, 6);
                        }
                        None => bw.write_bit(false),
                    }
                    if ot == ObjType::Dyn && !lfe {
                        match point.alt_position {
                            Some((x, y, zs, z)) => {
                                bw.write_bit(true);
                                bw.write_u32(x as u32, 6);
                                bw.write_u32(y as u32, 6);
                                bw.write_bit(zs);
                                bw.write_u32(z as u32, 4);
                            }
                            None => bw.write_bit(false),
                        }
                    } else if point.alt_position.is_some() {
                        return Err(Error::invalid(
                            "ac4: alt position only valid for DYN non-LFE objects",
                        ));
                    }
                }
            }
            match &set.additional {
                None => bw.write_bit(false),
                Some((per_obj, remain)) => {
                    bw.write_bit(true);
                    let mut probe = BitWriter::new();
                    write_ext_prec_alt_pos(&mut probe, per_obj, set.b_keep)?;
                    let used = probe.bit_position() as u32;
                    let total = used + remain;
                    if total == 0 || total % 8 != 0 {
                        return Err(Error::invalid(
                            "ac4: alt additional-data envelope is not whole bytes",
                        ));
                    }
                    write_variable_bits(bw, 2, total / 8 - 1);
                    write_ext_prec_alt_pos(bw, per_obj, set.b_keep)?;
                    for _ in 0..*remain {
                        bw.write_bit(false);
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_ext_prec_alt_pos(
    bw: &mut BitWriter,
    per_obj: &[Option<ExtPrecPos>],
    b_keep: bool,
) -> Result<()> {
    if b_keep {
        if !per_obj.is_empty() {
            return Err(Error::invalid("ac4: ext_prec_alt_pos with b_keep"));
        }
        return Ok(());
    }
    for entry in per_obj {
        match entry {
            Some(e) => {
                bw.write_bit(true);
                write_ext_prec_pos(bw, e)?;
            }
            None => bw.write_bit(false),
        }
    }
    Ok(())
}

/// Parsed `oamd_dyndata_multi()` (§6.2.8.4) — object-info blocks for
/// the objects **not** already covered by A-JOC-carried dynamic data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OamdDynDataMulti {
    /// `object_info_block` grid for the non-A-JOC objects, in object
    /// order (skipped objects have empty rows).
    pub object_blocks: Vec<Vec<ObjectInfoBlock>>,
}

/// Parse `oamd_dyndata_multi(n_objs, n_blocks, b_iframe, obj_type[],
/// b_lfe[], b_ajoc_coded[])` per §6.2.8.4.
pub fn parse_oamd_dyndata_multi(
    br: &mut BitReader<'_>,
    n_blocks: usize,
    b_iframe: bool,
    obj_type: &[ObjType],
    b_lfe: &[bool],
    b_ajoc_coded: &[bool],
) -> Result<OamdDynDataMulti> {
    if obj_type.len() != b_lfe.len() || obj_type.len() != b_ajoc_coded.len() {
        return Err(Error::invalid("ac4: dyndata_multi length mismatch"));
    }
    let mut object_blocks = Vec::with_capacity(obj_type.len());
    for i in 0..obj_type.len() {
        if b_ajoc_coded[i] {
            object_blocks.push(Vec::new());
            continue;
        }
        let b_dynamic_object = is_dynamic(obj_type[i], b_lfe[i]);
        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let b_no_delta = b_iframe && b == 0;
            blocks.push(parse_object_info_block(br, b_no_delta, b_dynamic_object)?);
        }
        object_blocks.push(blocks);
    }
    Ok(OamdDynDataMulti { object_blocks })
}

/// Write `oamd_dyndata_multi()` — exact inverse of
/// [`parse_oamd_dyndata_multi`].
pub fn write_oamd_dyndata_multi(
    bw: &mut BitWriter,
    d: &OamdDynDataMulti,
    b_iframe: bool,
    obj_type: &[ObjType],
    b_lfe: &[bool],
    b_ajoc_coded: &[bool],
) -> Result<()> {
    if d.object_blocks.len() != obj_type.len() {
        return Err(Error::invalid("ac4: dyndata_multi object count mismatch"));
    }
    for (i, blocks) in d.object_blocks.iter().enumerate() {
        if b_ajoc_coded[i] {
            if !blocks.is_empty() {
                return Err(Error::invalid("ac4: blocks present for A-JOC object"));
            }
            continue;
        }
        let b_dynamic_object = is_dynamic(obj_type[i], b_lfe[i]);
        for (b, blk) in blocks.iter().enumerate() {
            let b_no_delta = b_iframe && b == 0;
            write_object_info_block(bw, blk, b_no_delta, b_dynamic_object)?;
        }
    }
    Ok(())
}

// =====================================================================
// oamd_substream (§6.2.2.4)
// =====================================================================

/// Parsed standalone `oamd_substream()` (§6.2.2.4) — the OAMD-only
/// substream referenced by `oamd_substream_info()` (§6.2.1.13).
#[derive(Debug, Clone, PartialEq)]
pub struct OamdSubstream {
    /// `oamd_common_data()` (`b_oamd_common_data_present`).
    pub common_data: Option<OamdCommonData>,
    /// `oamd_timing_data()` (`b_oamd_timing_present`).
    pub timing: Option<OamdTimingData>,
    /// `oamd_dyndata_multi(...)` — present iff `b_alternative == 0`.
    pub dyndata: Option<OamdDynDataMulti>,
}

/// Caller context for [`parse_oamd_substream`] — the presentation- and
/// object-descriptor-level quantities the syntax gates on.
#[derive(Debug, Clone, Copy)]
pub struct OamdSubstreamContext<'a> {
    /// `b_alternative` from `ac4_presentation_substream_info()`.
    pub b_alternative: bool,
    /// `b_oamd_ndot` from `oamd_substream_info()` — true when this
    /// frame's OAMD decodes independently (§6.3.2.12.1), i.e. the
    /// I-frame flag for `oamd_dyndata_multi`.
    pub b_oamd_ndot: bool,
    /// The bed context's LFE presence (gates the
    /// `stereo_dmx_coeff()` LFE sub-block inside `oamd_common_data`).
    pub bed_has_lfe: bool,
    /// `num_obj_info_blocks` carried from a prior frame's
    /// `oamd_timing_data()`; used when `b_oamd_timing_present == 0`.
    pub prev_num_obj_info_blocks: usize,
    /// Per-object types (all `n_objs` objects).
    pub obj_type: &'a [ObjType],
    /// Per-object LFE flags.
    pub b_lfe: &'a [bool],
    /// Per-object `b_ajoc_coded` flags (A-JOC-carried objects are
    /// skipped by `oamd_dyndata_multi`).
    pub b_ajoc_coded: &'a [bool],
}

/// Parse `oamd_substream()` per §6.2.2.4. The element is byte-aligned
/// at entry and exit (§6.2.2.1).
pub fn parse_oamd_substream(
    br: &mut BitReader<'_>,
    ctx: &OamdSubstreamContext<'_>,
) -> Result<OamdSubstream> {
    let common_data = if br.read_bit()? {
        Some(parse_oamd_common_data(br, ctx.bed_has_lfe)?)
    } else {
        None
    };
    let timing = if br.read_bit()? {
        Some(parse_oamd_timing_data(br)?)
    } else {
        None
    };
    let dyndata = if !ctx.b_alternative {
        let n_blocks = timing
            .as_ref()
            .map(|t| t.num_obj_info_blocks())
            .unwrap_or(ctx.prev_num_obj_info_blocks);
        Some(parse_oamd_dyndata_multi(
            br,
            n_blocks,
            ctx.b_oamd_ndot,
            ctx.obj_type,
            ctx.b_lfe,
            ctx.b_ajoc_coded,
        )?)
    } else {
        None
    };
    br.align_to_byte();
    Ok(OamdSubstream {
        common_data,
        timing,
        dyndata,
    })
}

/// Write `oamd_substream()` — exact inverse of
/// [`parse_oamd_substream`] under the same context.
pub fn write_oamd_substream(
    bw: &mut BitWriter,
    s: &OamdSubstream,
    ctx: &OamdSubstreamContext<'_>,
) -> Result<()> {
    match &s.common_data {
        Some(c) => {
            bw.write_bit(true);
            write_oamd_common_data(bw, c, ctx.bed_has_lfe)?;
        }
        None => bw.write_bit(false),
    }
    match &s.timing {
        Some(t) => {
            bw.write_bit(true);
            write_oamd_timing_data(bw, t)?;
        }
        None => bw.write_bit(false),
    }
    match (&s.dyndata, ctx.b_alternative) {
        (Some(d), false) => {
            write_oamd_dyndata_multi(
                bw,
                d,
                ctx.b_oamd_ndot,
                ctx.obj_type,
                ctx.b_lfe,
                ctx.b_ajoc_coded,
            )?;
        }
        (None, true) => {}
        (Some(_), true) => {
            return Err(Error::invalid(
                "ac4: oamd_substream dyndata needs b_alternative == 0",
            ));
        }
        (None, false) => {
            return Err(Error::invalid(
                "ac4: oamd_substream requires dyndata when b_alternative == 0",
            ));
        }
    }
    bw.align_to_byte();
    Ok(())
}

// =====================================================================
// trim (§6.2.8.9) + bed_render_info (§6.2.8.8) + tool elements
// =====================================================================

/// `NUM_TRIM_CONFIGS` per §6.3.9.10.4.
pub const NUM_TRIM_CONFIGS: usize = 9;

/// Per-config trim balance payload (`b_default_trim == 0`,
/// `b_disable_trim == 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TrimBalance {
    /// 4-bit `trim_centre` (`presence[4]`).
    pub trim_centre: Option<u8>,
    /// 4-bit `trim_surround` (`presence[3]`).
    pub trim_surround: Option<u8>,
    /// 4-bit `trim_height` (`presence[2]`).
    pub trim_height: Option<u8>,
    /// `bal3D_Y_sign_tb_code` (1) + `bal3D_Y_amount_tb` (4)
    /// (`presence[1]`).
    pub bal_tb: Option<(bool, u8)>,
    /// `bal3D_Y_sign_lis_code` (1) + `bal3D_Y_amount_lis` (4)
    /// (`presence[0]`).
    pub bal_lis: Option<(bool, u8)>,
}

/// Per-config trim entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimConfig {
    /// `b_default_trim == 1`.
    Default,
    /// `b_default_trim == 0`, `b_disable_trim == 1`.
    Disabled,
    /// Explicit balance payload.
    Balance(TrimBalance),
}

/// Parsed `trim()` (§6.2.8.9).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Trim {
    /// Present when `b_trim_present`.
    pub payload: Option<TrimPayload>,
}

/// Body of `trim()` when present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimPayload {
    /// 2-bit `warp_mode`.
    pub warp_mode: u8,
    /// 2-bit reserved field (preserved verbatim).
    pub reserved: u8,
    /// 2-bit `global_trim_mode`.
    pub global_trim_mode: u8,
    /// Per-config entries (`global_trim_mode == 0b10` only —
    /// `NUM_TRIM_CONFIGS` entries).
    pub configs: Vec<TrimConfig>,
}

/// Parse `trim()` per §6.2.8.9.
pub fn parse_trim(br: &mut BitReader<'_>) -> Result<Trim> {
    if !br.read_bit()? {
        return Ok(Trim { payload: None });
    }
    let warp_mode = br.read_u32(2)? as u8;
    let reserved = br.read_u32(2)? as u8;
    let global_trim_mode = br.read_u32(2)? as u8;
    let mut configs = Vec::new();
    if global_trim_mode == 0b10 {
        for _ in 0..NUM_TRIM_CONFIGS {
            if br.read_bit()? {
                configs.push(TrimConfig::Default);
            } else if br.read_bit()? {
                configs.push(TrimConfig::Disabled);
            } else {
                let presence = br.read_u32(5)?;
                let mut bal = TrimBalance::default();
                if presence & 0b10000 != 0 {
                    bal.trim_centre = Some(br.read_u32(4)? as u8);
                }
                if presence & 0b01000 != 0 {
                    bal.trim_surround = Some(br.read_u32(4)? as u8);
                }
                if presence & 0b00100 != 0 {
                    bal.trim_height = Some(br.read_u32(4)? as u8);
                }
                if presence & 0b00010 != 0 {
                    bal.bal_tb = Some((br.read_bit()?, br.read_u32(4)? as u8));
                }
                if presence & 0b00001 != 0 {
                    bal.bal_lis = Some((br.read_bit()?, br.read_u32(4)? as u8));
                }
                configs.push(TrimConfig::Balance(bal));
            }
        }
    }
    Ok(Trim {
        payload: Some(TrimPayload {
            warp_mode,
            reserved,
            global_trim_mode,
            configs,
        }),
    })
}

/// Write `trim()` — exact inverse of [`parse_trim`].
pub fn write_trim(bw: &mut BitWriter, t: &Trim) -> Result<()> {
    match &t.payload {
        None => bw.write_bit(false),
        Some(p) => {
            bw.write_bit(true);
            bw.write_u32(p.warp_mode as u32, 2);
            bw.write_u32(p.reserved as u32, 2);
            bw.write_u32(p.global_trim_mode as u32, 2);
            if p.global_trim_mode == 0b10 {
                if p.configs.len() != NUM_TRIM_CONFIGS {
                    return Err(Error::invalid("ac4: trim needs 9 configs"));
                }
                for cfg in &p.configs {
                    match cfg {
                        TrimConfig::Default => bw.write_bit(true),
                        TrimConfig::Disabled => {
                            bw.write_bit(false);
                            bw.write_bit(true);
                        }
                        TrimConfig::Balance(bal) => {
                            bw.write_bit(false);
                            bw.write_bit(false);
                            let mut presence = 0u32;
                            if bal.trim_centre.is_some() {
                                presence |= 0b10000;
                            }
                            if bal.trim_surround.is_some() {
                                presence |= 0b01000;
                            }
                            if bal.trim_height.is_some() {
                                presence |= 0b00100;
                            }
                            if bal.bal_tb.is_some() {
                                presence |= 0b00010;
                            }
                            if bal.bal_lis.is_some() {
                                presence |= 0b00001;
                            }
                            bw.write_u32(presence, 5);
                            for v in [bal.trim_centre, bal.trim_surround, bal.trim_height]
                                .into_iter()
                                .flatten()
                            {
                                bw.write_u32(v as u32, 4);
                            }
                            for (sign, amount) in [bal.bal_tb, bal.bal_lis].into_iter().flatten() {
                                bw.write_bit(sign);
                                bw.write_u32(amount as u32, 4);
                            }
                        }
                    }
                }
            } else if !p.configs.is_empty() {
                return Err(Error::invalid(
                    "ac4: trim configs only valid for global_trim_mode 0b10",
                ));
            }
        }
    }
    Ok(())
}

/// `tool_tb_to_f_s_b()` (§6.2.8.13) — top-back routing, three-way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolThreeWay {
    /// `b_*_to_front == 1` — 3-bit primary gain code.
    Front(u8),
    /// `b_*_to_front == 0`, `b_*_to_side == 1` — 3-bit side gain code.
    Side(u8),
    /// Both selectors 0 — 3-bit fallback gain code.
    Other(u8),
}

/// `tool_tb_to_f_s()` / `tool_tf_to_f_s()` (§6.2.8.14/16) — two-way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolTwoWay {
    /// `b_*_to_front == 1` — 3-bit gain code.
    Front(u8),
    /// `b_*_to_front == 0` — 3-bit gain code.
    Other(u8),
}

pub(crate) fn parse_tool_three_way(br: &mut BitReader<'_>) -> Result<ToolThreeWay> {
    if br.read_bit()? {
        Ok(ToolThreeWay::Front(br.read_u32(3)? as u8))
    } else if br.read_bit()? {
        Ok(ToolThreeWay::Side(br.read_u32(3)? as u8))
    } else {
        Ok(ToolThreeWay::Other(br.read_u32(3)? as u8))
    }
}

pub(crate) fn write_tool_three_way(bw: &mut BitWriter, t: ToolThreeWay) {
    match t {
        ToolThreeWay::Front(g) => {
            bw.write_bit(true);
            bw.write_u32(g as u32, 3);
        }
        ToolThreeWay::Side(g) => {
            bw.write_bit(false);
            bw.write_bit(true);
            bw.write_u32(g as u32, 3);
        }
        ToolThreeWay::Other(g) => {
            bw.write_bit(false);
            bw.write_bit(false);
            bw.write_u32(g as u32, 3);
        }
    }
}

pub(crate) fn parse_tool_two_way(br: &mut BitReader<'_>) -> Result<ToolTwoWay> {
    if br.read_bit()? {
        Ok(ToolTwoWay::Front(br.read_u32(3)? as u8))
    } else {
        Ok(ToolTwoWay::Other(br.read_u32(3)? as u8))
    }
}

pub(crate) fn write_tool_two_way(bw: &mut BitWriter, t: ToolTwoWay) {
    match t {
        ToolTwoWay::Front(g) => {
            bw.write_bit(true);
            bw.write_u32(g as u32, 3);
        }
        ToolTwoWay::Other(g) => {
            bw.write_bit(false);
            bw.write_u32(g as u32, 3);
        }
    }
}

/// Custom-downmix payload of `bed_render_info()`
/// (`b_cdmx_data_present == 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BedCdmxData {
    /// 3-bit `gain_w_to_f_code` (`b_cdmx_w_to_f`).
    pub gain_w_to_f: Option<u8>,
    /// 3-bit `gain_b4_to_b2_code` (`b_cdmx_b4_to_b2`).
    pub gain_b4_to_b2: Option<u8>,
    /// Top-middle channel tools (`b_tm_ch_present`).
    pub tm: Option<(Option<ToolThreeWay>, Option<ToolTwoWay>)>,
    /// Top-back channel tools (`b_tb_ch_present`).
    pub tb: Option<(Option<ToolThreeWay>, Option<ToolTwoWay>)>,
    /// Top-front channel tools (`b_tf_ch_present`).
    pub tf: Option<(Option<ToolThreeWay>, Option<ToolTwoWay>)>,
    /// 3-bit `gain_tfb_to_tm_code` (present when tb or tf present and
    /// `b_cdmx_tfb_to_tm`).
    pub gain_tfb_to_tm: Option<u8>,
}

/// Parsed `bed_render_info()` (§6.2.8.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BedRenderInfo {
    /// `stereo_dmx_coeff()` when `b_stereo_dmx_coeff == 1` (only
    /// meaningful when [`BedRenderInfo::payload`] is present).
    pub stereo_dmx_coeff: Option<crate::dmx_coeff::StereoDmxCoeff>,
    /// Present when `b_bed_render_info == 1`.
    pub payload: Option<BedCdmxData>,
}

/// Parse `bed_render_info()` per §6.2.8.8.
///
/// The `stereo_dmx_coeff()` sub-element has no dedicated syntax box in
/// the TS; the field layout is the factored-out form of the identical
/// inline `b_stereo_dmx_coeff` block in `custom_dmx_data()` (§6.2.9.2)
/// — see [`crate::dmx_coeff`]. Its LFE sub-block is gated on the bed's
/// LFE presence, passed as `bed_has_lfe`.
pub fn parse_bed_render_info(br: &mut BitReader<'_>, bed_has_lfe: bool) -> Result<BedRenderInfo> {
    if !br.read_bit()? {
        return Ok(BedRenderInfo {
            stereo_dmx_coeff: None,
            payload: None,
        });
    }
    let stereo_dmx_coeff = if br.read_bit()? {
        Some(crate::dmx_coeff::parse_stereo_dmx_coeff(br, bed_has_lfe)?)
    } else {
        None
    };
    let mut data = BedCdmxData::default();
    if br.read_bit()? {
        // b_cdmx_data_present.
        if br.read_bit()? {
            data.gain_w_to_f = Some(br.read_u32(3)? as u8);
        }
        if br.read_bit()? {
            data.gain_b4_to_b2 = Some(br.read_u32(3)? as u8);
        }
        if br.read_bit()? {
            // b_tm_ch_present → t2_to_f_s_b / t2_to_f_s.
            let a = if br.read_bit()? {
                Some(parse_tool_three_way(br)?)
            } else {
                None
            };
            let b = if br.read_bit()? {
                Some(parse_tool_two_way(br)?)
            } else {
                None
            };
            data.tm = Some((a, b));
        }
        if br.read_bit()? {
            // b_tb_ch_present → tb_to_f_s_b / tb_to_f_s.
            let a = if br.read_bit()? {
                Some(parse_tool_three_way(br)?)
            } else {
                None
            };
            let b = if br.read_bit()? {
                Some(parse_tool_two_way(br)?)
            } else {
                None
            };
            data.tb = Some((a, b));
        }
        if br.read_bit()? {
            // b_tf_ch_present → tf_to_f_s_b / tf_to_f_s.
            let a = if br.read_bit()? {
                Some(parse_tool_three_way(br)?)
            } else {
                None
            };
            let b = if br.read_bit()? {
                Some(parse_tool_two_way(br)?)
            } else {
                None
            };
            data.tf = Some((a, b));
        }
        if (data.tb.is_some() || data.tf.is_some()) && br.read_bit()? {
            data.gain_tfb_to_tm = Some(br.read_u32(3)? as u8);
        }
    }
    Ok(BedRenderInfo {
        stereo_dmx_coeff,
        payload: Some(data),
    })
}

/// Write `bed_render_info()` — exact inverse of
/// [`parse_bed_render_info`] under the same `bed_has_lfe` context.
pub fn write_bed_render_info(
    bw: &mut BitWriter,
    b: &BedRenderInfo,
    bed_has_lfe: bool,
) -> Result<()> {
    match &b.payload {
        None => {
            if b.stereo_dmx_coeff.is_some() {
                return Err(Error::invalid(
                    "ac4: stereo_dmx_coeff requires b_bed_render_info",
                ));
            }
            bw.write_bit(false);
        }
        Some(data) => {
            bw.write_bit(true);
            match &b.stereo_dmx_coeff {
                Some(c) => {
                    bw.write_bit(true);
                    crate::dmx_coeff::write_stereo_dmx_coeff(bw, c, bed_has_lfe)?;
                }
                None => bw.write_bit(false),
            }
            let has_cdmx = data.gain_w_to_f.is_some()
                || data.gain_b4_to_b2.is_some()
                || data.tm.is_some()
                || data.tb.is_some()
                || data.tf.is_some()
                || data.gain_tfb_to_tm.is_some();
            bw.write_bit(has_cdmx);
            if has_cdmx {
                match data.gain_w_to_f {
                    Some(g) => {
                        bw.write_bit(true);
                        bw.write_u32(g as u32, 3);
                    }
                    None => bw.write_bit(false),
                }
                match data.gain_b4_to_b2 {
                    Some(g) => {
                        bw.write_bit(true);
                        bw.write_u32(g as u32, 3);
                    }
                    None => bw.write_bit(false),
                }
                for pair in [&data.tm, &data.tb, &data.tf] {
                    match pair {
                        None => bw.write_bit(false),
                        Some((a, b2)) => {
                            bw.write_bit(true);
                            match a {
                                Some(t) => {
                                    bw.write_bit(true);
                                    write_tool_three_way(bw, *t);
                                }
                                None => bw.write_bit(false),
                            }
                            match b2 {
                                Some(t) => {
                                    bw.write_bit(true);
                                    write_tool_two_way(bw, *t);
                                }
                                None => bw.write_bit(false),
                            }
                        }
                    }
                }
                if data.tb.is_some() || data.tf.is_some() {
                    match data.gain_tfb_to_tm {
                        Some(g) => {
                            bw.write_bit(true);
                            bw.write_u32(g as u32, 3);
                        }
                        None => bw.write_bit(false),
                    }
                } else if data.gain_tfb_to_tm.is_some() {
                    return Err(Error::invalid(
                        "ac4: gain_tfb_to_tm needs tb or tf channels present",
                    ));
                }
            }
        }
    }
    Ok(())
}

// =====================================================================
// oamd_common_data (§6.2.8.1)
// =====================================================================

/// Additional-data envelope of `oamd_common_data()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonAddData {
    /// Parsed `trim()`.
    pub trim: Trim,
    /// Parsed `bed_render_info()`.
    pub bed_render_info: BedRenderInfo,
    /// Announced-but-unconsumed bits skipped after the two elements.
    pub filler_bits: u32,
}

/// Parsed `oamd_common_data()` (§6.2.8.1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OamdCommonData {
    /// 5-bit `master_screen_size_ratio_code`; `None` when
    /// `b_default_screen_size_ratio == 1`.
    pub master_screen_size_ratio_code: Option<u8>,
    /// `b_bed_object_chan_distribute`.
    pub b_bed_object_chan_distribute: bool,
    /// `b_additional_data` payload.
    pub add_data: Option<CommonAddData>,
}

/// Parse `oamd_common_data()` per §6.2.8.1. `bed_has_lfe` gates the
/// LFE sub-block of `bed_render_info()`'s `stereo_dmx_coeff()`.
pub fn parse_oamd_common_data(br: &mut BitReader<'_>, bed_has_lfe: bool) -> Result<OamdCommonData> {
    let b_default_ratio = br.read_bit()?;
    let master_screen_size_ratio_code = if !b_default_ratio {
        Some(br.read_u32(5)? as u8)
    } else {
        None
    };
    let b_bed_object_chan_distribute = br.read_bit()?;
    let add_data = if br.read_bit()? {
        let mut add_data_bytes = br.read_u32(1)? + 1;
        if add_data_bytes == 2 {
            add_data_bytes = add_data_bytes
                .checked_add(variable_bits(br, 2)?)
                .ok_or_else(|| Error::invalid("ac4: oamd add_data size overflow"))?;
        }
        let total = add_data_bytes
            .checked_mul(8)
            .ok_or_else(|| Error::invalid("ac4: oamd add_data size overflow"))?;
        let start = br.bit_position();
        let trim = parse_trim(br)?;
        let bed_render_info = parse_bed_render_info(br, bed_has_lfe)?;
        let used = (br.bit_position() - start) as u32;
        if used > total {
            return Err(Error::invalid(
                "ac4: trim + bed_render_info exceeded oamd add-data envelope",
            ));
        }
        let filler_bits = total - used;
        for _ in 0..filler_bits {
            let _ = br.read_bit()?;
        }
        Some(CommonAddData {
            trim,
            bed_render_info,
            filler_bits,
        })
    } else {
        None
    };
    Ok(OamdCommonData {
        master_screen_size_ratio_code,
        b_bed_object_chan_distribute,
        add_data,
    })
}

/// Write `oamd_common_data()` — exact inverse of
/// [`parse_oamd_common_data`] under the same `bed_has_lfe` context.
pub fn write_oamd_common_data(
    bw: &mut BitWriter,
    c: &OamdCommonData,
    bed_has_lfe: bool,
) -> Result<()> {
    match c.master_screen_size_ratio_code {
        None => bw.write_bit(true),
        Some(code) => {
            bw.write_bit(false);
            bw.write_u32(code as u32, 5);
        }
    }
    bw.write_bit(c.b_bed_object_chan_distribute);
    match &c.add_data {
        None => bw.write_bit(false),
        Some(a) => {
            bw.write_bit(true);
            let mut probe = BitWriter::new();
            write_trim(&mut probe, &a.trim)?;
            write_bed_render_info(&mut probe, &a.bed_render_info, bed_has_lfe)?;
            let used = probe.bit_position() as u32;
            let total = used + a.filler_bits;
            if total == 0 || total % 8 != 0 {
                return Err(Error::invalid(
                    "ac4: oamd add-data envelope is not whole bytes",
                ));
            }
            let add_data_bytes = total / 8;
            if add_data_bytes < 1 {
                return Err(Error::invalid("ac4: oamd add-data too small"));
            }
            if add_data_bytes == 1 {
                bw.write_u32(0, 1);
            } else {
                bw.write_u32(1, 1);
                write_variable_bits(bw, 2, add_data_bytes - 2);
            }
            write_trim(bw, &a.trim)?;
            write_bed_render_info(bw, &a.bed_render_info, bed_has_lfe)?;
            for _ in 0..a.filler_bits {
                bw.write_bit(false);
            }
        }
    }
    Ok(())
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_timing(t: &OamdTimingData) {
        let mut bw = BitWriter::new();
        write_oamd_timing_data(&mut bw, t).unwrap();
        bw.write_u32(0, 7); // trailing guard
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_oamd_timing_data(&mut br).unwrap();
        assert_eq!(&got, t);
    }

    #[test]
    fn timing_data_round_trips_all_offset_forms() {
        for offset in [
            OaSampleOffset::Zero,
            OaSampleOffset::Code8,
            OaSampleOffset::Code16,
            OaSampleOffset::Code24,
            OaSampleOffset::Explicit(19),
        ] {
            round_trip_timing(&OamdTimingData {
                sample_offset: offset,
                blocks: vec![OamdTimingBlock {
                    block_offset_factor: 33,
                    ramp_duration: RampDuration::D512,
                }],
            });
        }
    }

    #[test]
    fn timing_data_round_trips_all_ramp_forms() {
        for ramp in [
            RampDuration::Zero,
            RampDuration::D512,
            RampDuration::D1536,
            RampDuration::Table(9),
            RampDuration::Explicit(2047),
        ] {
            round_trip_timing(&OamdTimingData {
                sample_offset: OaSampleOffset::Zero,
                blocks: vec![OamdTimingBlock {
                    block_offset_factor: 0,
                    ramp_duration: ramp,
                }],
            });
        }
    }

    #[test]
    fn timing_semantics_match_tables() {
        assert_eq!(OaSampleOffset::Code8.samples(), 8);
        assert_eq!(OaSampleOffset::Code16.samples(), 16);
        assert_eq!(OaSampleOffset::Code24.samples(), 24);
        assert_eq!(OaSampleOffset::Explicit(31).samples(), 31);
        assert_eq!(RampDuration::Zero.samples(), 0);
        assert_eq!(RampDuration::D512.samples(), 512);
        assert_eq!(RampDuration::D1536.samples(), 1536);
        // Table 129 endpoints + interior.
        assert_eq!(RampDuration::Table(0).samples(), 32);
        assert_eq!(RampDuration::Table(7).samples(), 1001);
        assert_eq!(RampDuration::Table(15).samples(), 2048);
        assert_eq!(RampDuration::Explicit(2047).samples(), 2047);
    }

    #[test]
    fn object_gain_table_136() {
        // 0..=14 → 15 − v dB; 15..=63 → 14 − v dB.
        assert_eq!(ObjectGain::Value(0).db(), Some(15));
        assert_eq!(ObjectGain::Value(14).db(), Some(1));
        assert_eq!(ObjectGain::Value(15).db(), Some(-1));
        assert_eq!(ObjectGain::Value(63).db(), Some(-49));
        assert_eq!(ObjectGain::NegInf.db(), None);
    }

    fn round_trip_basic(b: &ObjectBasicInfo) {
        let mut bw = BitWriter::new();
        write_object_basic_info(&mut bw, b).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(&parse_object_basic_info(&mut br).unwrap(), b);
    }

    #[test]
    fn object_basic_info_round_trips() {
        round_trip_basic(&ObjectBasicInfo::default());
        round_trip_basic(&ObjectBasicInfo {
            b_default: false,
            md: Some(BasicInfoMd::Gain),
            gain: Some(ObjectGain::Value(20)),
            priority_code: None,
        });
        round_trip_basic(&ObjectBasicInfo {
            b_default: false,
            md: Some(BasicInfoMd::GainPriority),
            gain: Some(ObjectGain::NegInf),
            priority_code: Some(31),
        });
        round_trip_basic(&ObjectBasicInfo {
            b_default: false,
            md: Some(BasicInfoMd::Priority),
            gain: None,
            priority_code: Some(7),
        });
    }

    fn round_trip_render(r: &ObjectRenderInfo, b_no_delta: bool) {
        let mut bw = BitWriter::new();
        write_object_render_info(&mut bw, r, b_no_delta).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_object_render_info(&mut br, r.all_new, b_no_delta).unwrap();
        assert_eq!(&got, r);
    }

    #[test]
    fn object_render_info_round_trips_abs_position_all_new() {
        round_trip_render(
            &ObjectRenderInfo {
                all_new: true,
                b_position_present: true,
                b_zone_present: true,
                b_otherprops_present: true,
                position: Some(RenderPosition::Abs {
                    x: 31,
                    y: 0,
                    z_sign: true,
                    z: 15,
                }),
                zone: Some(RenderZone {
                    b_grouped_defaults: true,
                    group_zone_mask: None,
                    zone_mask: None,
                }),
                otherprops: Some(RenderOtherProps {
                    b_grouped_defaults: true,
                    ..Default::default()
                }),
            },
            true,
        );
    }

    #[test]
    fn object_render_info_round_trips_diff_position_and_props() {
        round_trip_render(
            &ObjectRenderInfo {
                all_new: false,
                b_position_present: true,
                b_zone_present: true,
                b_otherprops_present: true,
                position: Some(RenderPosition::Diff { x: -4, y: 3, z: -1 }),
                zone: Some(RenderZone {
                    b_grouped_defaults: false,
                    group_zone_mask: Some(0b111),
                    zone_mask: Some(0b101),
                }),
                otherprops: Some(RenderOtherProps {
                    b_grouped_defaults: false,
                    group_other_mask: Some(0b1111),
                    width: Some(ObjectWidth::PerAxis { x: 1, y: 2, z: 3 }),
                    screen: Some((5, 2)),
                    distance: Some(Some(9)),
                    divergence: Some(ObjectDivergence::Code {
                        mode: 0b10,
                        code: 45,
                    }),
                }),
            },
            false,
        );
    }

    #[test]
    fn object_render_info_round_trips_partial_mask() {
        round_trip_render(
            &ObjectRenderInfo {
                all_new: false,
                b_position_present: false,
                b_zone_present: false,
                b_otherprops_present: true,
                position: None,
                zone: None,
                otherprops: Some(RenderOtherProps {
                    b_grouped_defaults: false,
                    group_other_mask: Some(0b1000),
                    divergence: Some(ObjectDivergence::Table(2)),
                    ..Default::default()
                }),
            },
            false,
        );
    }

    #[test]
    fn ext_prec_pos_round_trips_and_dequantizes() {
        for e in [
            ExtPrecPos::default(),
            ExtPrecPos {
                x: Some(0b01),
                y: None,
                z: Some(0b10),
            },
            ExtPrecPos {
                x: Some(0b11),
                y: Some(0b00),
                z: Some(0b01),
            },
        ] {
            let mut bw = BitWriter::new();
            write_ext_prec_pos(&mut bw, &e).unwrap();
            bw.write_u32(0, 7);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(parse_ext_prec_pos(&mut br).unwrap(), e);
        }
        // Tables 146m-o.
        assert_eq!(ext_prec_value(0b00), 1);
        assert_eq!(ext_prec_value(0b01), 2);
        assert_eq!(ext_prec_value(0b10), -1);
        assert_eq!(ext_prec_value(0b11), -2);
    }

    fn round_trip_block(blk: &ObjectInfoBlock, b_no_delta: bool, b_dynamic: bool) {
        let mut bw = BitWriter::new();
        write_object_info_block(&mut bw, blk, b_no_delta, b_dynamic).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_object_info_block(&mut br, b_no_delta, b_dynamic).unwrap();
        assert_eq!(&got, blk);
    }

    #[test]
    fn object_info_block_inactive_defaults() {
        round_trip_block(
            &ObjectInfoBlock {
                b_object_not_active: true,
                basic_status: InfoStatus::Default,
                basic_info: None,
                render_status: InfoStatus::Default,
                render_info: None,
                add_table_data: None,
            },
            true,
            true,
        );
    }

    #[test]
    fn object_info_block_iframe_dynamic_all_new() {
        round_trip_block(
            &ObjectInfoBlock {
                b_object_not_active: false,
                basic_status: InfoStatus::AllNew,
                basic_info: Some(ObjectBasicInfo::default()),
                render_status: InfoStatus::AllNew,
                render_info: Some(ObjectRenderInfo {
                    all_new: true,
                    b_position_present: true,
                    b_zone_present: true,
                    b_otherprops_present: true,
                    position: Some(RenderPosition::Abs {
                        x: 62,
                        y: 31,
                        z_sign: false,
                        z: 5,
                    }),
                    zone: Some(RenderZone {
                        b_grouped_defaults: true,
                        group_zone_mask: None,
                        zone_mask: None,
                    }),
                    otherprops: Some(RenderOtherProps {
                        b_grouped_defaults: true,
                        ..Default::default()
                    }),
                }),
                add_table_data: None,
            },
            true,
            true,
        );
    }

    #[test]
    fn object_info_block_pframe_reuse_paths() {
        // Basic REUSE + render REUSE on a P-frame block.
        round_trip_block(
            &ObjectInfoBlock {
                b_object_not_active: false,
                basic_status: InfoStatus::Reuse,
                basic_info: None,
                render_status: InfoStatus::Reuse,
                render_info: None,
                add_table_data: None,
            },
            false,
            true,
        );
        // PART_REUSE render with a masked update.
        round_trip_block(
            &ObjectInfoBlock {
                b_object_not_active: false,
                basic_status: InfoStatus::AllNew,
                basic_info: Some(ObjectBasicInfo::default()),
                render_status: InfoStatus::PartReuse,
                render_info: Some(ObjectRenderInfo {
                    all_new: false,
                    b_position_present: true,
                    b_zone_present: false,
                    b_otherprops_present: false,
                    position: Some(RenderPosition::Diff { x: 1, y: -2, z: 0 }),
                    zone: None,
                    otherprops: None,
                }),
                add_table_data: None,
            },
            false,
            true,
        );
    }

    #[test]
    fn object_info_block_static_object_has_default_render() {
        // Non-dynamic object: render status is DEFAULT, no reuse bits.
        round_trip_block(
            &ObjectInfoBlock {
                b_object_not_active: false,
                basic_status: InfoStatus::AllNew,
                basic_info: Some(ObjectBasicInfo {
                    b_default: false,
                    md: Some(BasicInfoMd::Gain),
                    gain: Some(ObjectGain::PrevObject),
                    priority_code: None,
                }),
                render_status: InfoStatus::Default,
                render_info: None,
                add_table_data: None,
            },
            true,
            false,
        );
    }

    #[test]
    fn object_info_block_add_table_data_envelope() {
        round_trip_block(
            &ObjectInfoBlock {
                b_object_not_active: false,
                basic_status: InfoStatus::AllNew,
                basic_info: Some(ObjectBasicInfo::default()),
                render_status: InfoStatus::AllNew,
                render_info: Some(ObjectRenderInfo {
                    all_new: true,
                    b_position_present: true,
                    b_zone_present: true,
                    b_otherprops_present: true,
                    position: Some(RenderPosition::Abs {
                        x: 10,
                        y: 20,
                        z_sign: true,
                        z: 0,
                    }),
                    zone: Some(RenderZone {
                        b_grouped_defaults: true,
                        group_zone_mask: None,
                        zone_mask: None,
                    }),
                    otherprops: Some(RenderOtherProps {
                        b_grouped_defaults: true,
                        ..Default::default()
                    }),
                }),
                add_table_data: Some((
                    AddPerObjectMd {
                        b_obj_trim_disable: true,
                        ext_prec_pos: Some(ExtPrecPos {
                            x: Some(0b10),
                            y: Some(0b01),
                            z: None,
                        }),
                    },
                    // 1 + 1 + 3 + 2 + 2 = 9 bits used → 16-bit envelope
                    // leaves 7 filler bits.
                    7,
                )),
            },
            true,
            true,
        );
    }

    fn simple_active_block(all_new_pos_x: u8) -> ObjectInfoBlock {
        ObjectInfoBlock {
            b_object_not_active: false,
            basic_status: InfoStatus::AllNew,
            basic_info: Some(ObjectBasicInfo::default()),
            render_status: InfoStatus::AllNew,
            render_info: Some(ObjectRenderInfo {
                all_new: true,
                b_position_present: true,
                b_zone_present: true,
                b_otherprops_present: true,
                position: Some(RenderPosition::Abs {
                    x: all_new_pos_x,
                    y: 0,
                    z_sign: true,
                    z: 0,
                }),
                zone: Some(RenderZone {
                    b_grouped_defaults: true,
                    group_zone_mask: None,
                    zone_mask: None,
                }),
                otherprops: Some(RenderOtherProps {
                    b_grouped_defaults: true,
                    ..Default::default()
                }),
            }),
            add_table_data: None,
        }
    }

    fn bed_block() -> ObjectInfoBlock {
        ObjectInfoBlock {
            b_object_not_active: false,
            basic_status: InfoStatus::AllNew,
            basic_info: Some(ObjectBasicInfo::default()),
            render_status: InfoStatus::Default,
            render_info: None,
            add_table_data: None,
        }
    }

    #[test]
    fn dyndata_single_round_trips_mixed_objects() {
        let obj_type = [ObjType::Bed, ObjType::Dyn, ObjType::Dyn];
        let b_lfe = [false, true, false];
        let d = OamdDynDataSingle {
            object_blocks: vec![
                vec![bed_block()],
                // DYN + LFE ⇒ not a dynamic object per §6.2.8.3.
                vec![bed_block()],
                vec![simple_active_block(42)],
            ],
            alt: None,
        };
        let mut bw = BitWriter::new();
        write_oamd_dyndata_single(&mut bw, &d, true, &obj_type, &b_lfe).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_oamd_dyndata_single(&mut br, 1, true, false, &obj_type, &b_lfe).unwrap();
        assert_eq!(got, d);
    }

    #[test]
    fn dyndata_single_round_trips_alternative_sets() {
        let obj_type = [ObjType::Dyn, ObjType::Dyn];
        let b_lfe = [false, false];
        let d = OamdDynDataSingle {
            object_blocks: vec![vec![simple_active_block(1)], vec![simple_active_block(2)]],
            alt: Some(AltProperties {
                b_ducking_disabled: true,
                object_sound_category: 1,
                sets: vec![
                    AltDataSet {
                        b_keep: true,
                        b_common_data: None,
                        points: vec![],
                        additional: None,
                    },
                    AltDataSet {
                        b_keep: false,
                        b_common_data: Some(false),
                        points: vec![
                            AltDataPoint {
                                alt_gain: Some(63),
                                alt_position: Some((10, 20, false, 3)),
                            },
                            AltDataPoint {
                                alt_gain: None,
                                alt_position: None,
                            },
                        ],
                        additional: Some((
                            vec![
                                Some(ExtPrecPos {
                                    x: Some(0b01),
                                    y: None,
                                    z: None,
                                }),
                                None,
                            ],
                            // 1+3+2 + 1 = 7 bits → 8-bit envelope, 1
                            // filler bit.
                            1,
                        )),
                    },
                ],
            }),
        };
        let mut bw = BitWriter::new();
        write_oamd_dyndata_single(&mut bw, &d, true, &obj_type, &b_lfe).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_oamd_dyndata_single(&mut br, 1, true, true, &obj_type, &b_lfe).unwrap();
        assert_eq!(got, d);
    }

    #[test]
    fn dyndata_single_escaped_sound_category_round_trips() {
        let obj_type = [ObjType::Isf];
        let b_lfe = [false];
        let d = OamdDynDataSingle {
            object_blocks: vec![vec![bed_block()]],
            alt: Some(AltProperties {
                b_ducking_disabled: false,
                object_sound_category: 7, // 3 + variable_bits escape
                sets: vec![AltDataSet {
                    b_keep: false,
                    // ISF first object ⇒ no b_common_data bit, 1 point.
                    b_common_data: None,
                    points: vec![AltDataPoint {
                        alt_gain: Some(0),
                        alt_position: None,
                    }],
                    additional: None,
                }],
            }),
        };
        let mut bw = BitWriter::new();
        write_oamd_dyndata_single(&mut bw, &d, true, &obj_type, &b_lfe).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_oamd_dyndata_single(&mut br, 1, true, true, &obj_type, &b_lfe).unwrap();
        assert_eq!(got, d);
    }

    #[test]
    fn dyndata_multi_skips_ajoc_coded_objects() {
        let obj_type = [ObjType::Dyn, ObjType::Dyn, ObjType::Bed];
        let b_lfe = [false, false, false];
        let b_ajoc = [true, false, true];
        let d = OamdDynDataMulti {
            object_blocks: vec![Vec::new(), vec![simple_active_block(9)], Vec::new()],
        };
        let mut bw = BitWriter::new();
        write_oamd_dyndata_multi(&mut bw, &d, true, &obj_type, &b_lfe, &b_ajoc).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_oamd_dyndata_multi(&mut br, 1, true, &obj_type, &b_lfe, &b_ajoc).unwrap();
        assert_eq!(got, d);
    }

    #[test]
    fn trim_round_trips_all_modes() {
        for t in [
            Trim { payload: None },
            Trim {
                payload: Some(TrimPayload {
                    warp_mode: 1,
                    reserved: 0,
                    global_trim_mode: 0b01,
                    configs: vec![],
                }),
            },
            Trim {
                payload: Some(TrimPayload {
                    warp_mode: 2,
                    reserved: 3,
                    global_trim_mode: 0b10,
                    configs: vec![
                        TrimConfig::Default,
                        TrimConfig::Disabled,
                        TrimConfig::Balance(TrimBalance {
                            trim_centre: Some(4),
                            trim_surround: None,
                            trim_height: Some(15),
                            bal_tb: Some((true, 8)),
                            bal_lis: None,
                        }),
                        TrimConfig::Default,
                        TrimConfig::Default,
                        TrimConfig::Default,
                        TrimConfig::Default,
                        TrimConfig::Default,
                        TrimConfig::Default,
                    ],
                }),
            },
        ] {
            let mut bw = BitWriter::new();
            write_trim(&mut bw, &t).unwrap();
            bw.write_u32(0, 7);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(parse_trim(&mut br).unwrap(), t);
        }
    }

    #[test]
    fn bed_render_info_round_trips() {
        for bed_has_lfe in [false, true] {
            for b in [
                BedRenderInfo {
                    stereo_dmx_coeff: None,
                    payload: None,
                },
                BedRenderInfo {
                    stereo_dmx_coeff: None,
                    payload: Some(BedCdmxData::default()),
                },
                BedRenderInfo {
                    stereo_dmx_coeff: Some(crate::dmx_coeff::StereoDmxCoeff {
                        loro_centre_mixgain: 4,
                        loro_surround_mixgain: 2,
                        ltrt_mixgains: Some((1, 6)),
                        lfe_mixgain: if bed_has_lfe { Some(9) } else { None },
                        preferred_dmx_method: 1,
                    }),
                    payload: Some(BedCdmxData::default()),
                },
                BedRenderInfo {
                    stereo_dmx_coeff: None,
                    payload: Some(BedCdmxData {
                        gain_w_to_f: Some(3),
                        gain_b4_to_b2: None,
                        tm: Some((Some(ToolThreeWay::Front(1)), Some(ToolTwoWay::Other(2)))),
                        tb: Some((Some(ToolThreeWay::Side(5)), None)),
                        tf: Some((None, Some(ToolTwoWay::Front(7)))),
                        gain_tfb_to_tm: Some(4),
                    }),
                },
            ] {
                let mut bw = BitWriter::new();
                write_bed_render_info(&mut bw, &b, bed_has_lfe).unwrap();
                bw.write_u32(0, 7);
                let bytes = bw.into_bytes();
                let mut br = BitReader::new(&bytes);
                assert_eq!(parse_bed_render_info(&mut br, bed_has_lfe).unwrap(), b);
            }
        }
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)] // groups mirror the field widths
    fn bed_render_info_stereo_dmx_coeff_parses_custom_dmx_form() {
        // b_bed_render_info = 1, b_stereo_dmx_coeff = 1, then the
        // §6.2.9.2 inline block: loro_centre = 0b010, loro_surround =
        // 0b100, b_ltrt_mixinfo = 0, (no LFE context), preferred = 0b01,
        // then b_cdmx_data_present = 0.
        let bytes = [0b11_010_100u8, 0b0_01_0_0000];
        let mut br = BitReader::new(&bytes);
        let b = parse_bed_render_info(&mut br, false).unwrap();
        let c = b.stereo_dmx_coeff.expect("stereo_dmx_coeff present");
        assert_eq!(c.loro_centre_mixgain, 2);
        assert_eq!(c.loro_surround_mixgain, 4);
        assert!(c.ltrt_mixgains.is_none());
        assert!(c.lfe_mixgain.is_none());
        assert_eq!(c.preferred_dmx_method, 1);
        assert_eq!(b.payload, Some(BedCdmxData::default()));
    }

    #[test]
    fn oamd_common_data_round_trips() {
        for c in [
            OamdCommonData::default(),
            OamdCommonData {
                master_screen_size_ratio_code: Some(17),
                b_bed_object_chan_distribute: true,
                add_data: None,
            },
            OamdCommonData {
                master_screen_size_ratio_code: None,
                b_bed_object_chan_distribute: false,
                add_data: Some(CommonAddData {
                    trim: Trim { payload: None },
                    bed_render_info: BedRenderInfo {
                        stereo_dmx_coeff: None,
                        payload: None,
                    },
                    // 2 bits used → 8-bit envelope leaves 6 filler bits.
                    filler_bits: 6,
                }),
            },
            OamdCommonData {
                master_screen_size_ratio_code: None,
                b_bed_object_chan_distribute: false,
                add_data: Some(CommonAddData {
                    trim: Trim {
                        payload: Some(TrimPayload {
                            warp_mode: 0,
                            reserved: 0,
                            global_trim_mode: 0b00,
                            configs: vec![],
                        }),
                    },
                    bed_render_info: BedRenderInfo {
                        stereo_dmx_coeff: None,
                        payload: Some(BedCdmxData {
                            gain_w_to_f: Some(1),
                            ..Default::default()
                        }),
                    },
                    // trim 7 + bri 1+1+1+1+3+1+... compute: trim()
                    // present=1+2+2+2=7; bri: 1 (present) + 1 (stereo=0)
                    // + 1 (cdmx=1) + 1+3 (w_to_f) + 1 (b4b2=0) + 3×1
                    // (tm/tb/tf=0) = 11; total 18 → 24-bit envelope
                    // leaves 6 filler bits.
                    filler_bits: 6,
                }),
            },
        ] {
            let mut bw = BitWriter::new();
            write_oamd_common_data(&mut bw, &c, false).unwrap();
            bw.write_u32(0, 7);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(parse_oamd_common_data(&mut br, false).unwrap(), c);
        }
    }

    #[test]
    fn oamd_substream_round_trips_all_shapes() {
        // Two dynamic objects, one A-JOC-coded (skipped by the multi
        // walk), one plain.
        let obj_type = [ObjType::Dyn, ObjType::Dyn];
        let b_lfe = [false, false];
        let b_ajoc_coded = [true, false];
        let timing = OamdTimingData {
            sample_offset: OaSampleOffset::Zero,
            blocks: vec![OamdTimingBlock {
                block_offset_factor: 5,
                ramp_duration: RampDuration::D512,
            }],
        };
        for (b_alternative, with_common, with_timing) in [
            (false, false, true),
            (false, true, true),
            (true, false, false),
            (true, true, true),
        ] {
            let ctx = OamdSubstreamContext {
                b_alternative,
                b_oamd_ndot: true,
                bed_has_lfe: false,
                prev_num_obj_info_blocks: 1,
                obj_type: &obj_type,
                b_lfe: &b_lfe,
                b_ajoc_coded: &b_ajoc_coded,
            };
            let n_blocks = 1;
            let dyndata = (!b_alternative).then(|| {
                // Build a canonical dyndata by writing the flag bits
                // through the real writer: one non-A-JOC object with
                // one no-delta block (b_object_not_active form).
                let mut dbw = BitWriter::new();
                // object_info_block(b_no_delta = 1): b_object_not_active
                // = 1 → object_basic_info implied defaults... use the
                // parser to produce the canonical struct instead.
                dbw.write_bit(true); // b_object_not_active
                dbw.write_u32(0, 7);
                let dbytes = dbw.into_bytes();
                let mut dbr = BitReader::new(&dbytes);
                parse_oamd_dyndata_multi(&mut dbr, n_blocks, true, &obj_type, &b_lfe, &b_ajoc_coded)
                    .unwrap()
            });
            let s = OamdSubstream {
                common_data: with_common.then(OamdCommonData::default),
                timing: with_timing.then(|| timing.clone()),
                dyndata,
            };
            let mut bw = BitWriter::new();
            write_oamd_substream(&mut bw, &s, &ctx).unwrap();
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let got = parse_oamd_substream(&mut br, &ctx).unwrap();
            assert_eq!(got, s, "alt={b_alternative} common={with_common}");
            assert_eq!(br.bit_position() % 8, 0);
        }
    }

    #[test]
    fn oamd_substream_uses_prev_block_count_without_timing() {
        // b_alternative = 0 and no timing → num_obj_info_blocks falls
        // back to the carried prev count (here 0 → empty block rows).
        let obj_type = [ObjType::Dyn];
        let b_lfe = [false];
        let b_ajoc_coded = [false];
        let ctx = OamdSubstreamContext {
            b_alternative: false,
            b_oamd_ndot: true,
            bed_has_lfe: false,
            prev_num_obj_info_blocks: 0,
            obj_type: &obj_type,
            b_lfe: &b_lfe,
            b_ajoc_coded: &b_ajoc_coded,
        };
        let s = OamdSubstream {
            common_data: None,
            timing: None,
            dyndata: Some(OamdDynDataMulti {
                object_blocks: vec![Vec::new()],
            }),
        };
        let mut bw = BitWriter::new();
        write_oamd_substream(&mut bw, &s, &ctx).unwrap();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(parse_oamd_substream(&mut br, &ctx).unwrap(), s);
    }

    #[test]
    fn oamd_common_data_round_trips_stereo_dmx_coeff_with_lfe() {
        // trim absent (1 bit) + bed_render_info with stereo_dmx_coeff
        // under an LFE-carrying bed.
        let c = OamdCommonData {
            master_screen_size_ratio_code: None,
            b_bed_object_chan_distribute: false,
            add_data: Some(CommonAddData {
                trim: Trim { payload: None },
                bed_render_info: BedRenderInfo {
                    stereo_dmx_coeff: Some(crate::dmx_coeff::StereoDmxCoeff {
                        loro_centre_mixgain: 4,
                        loro_surround_mixgain: 4,
                        ltrt_mixgains: None,
                        lfe_mixgain: Some(21),
                        preferred_dmx_method: 2,
                    }),
                    payload: Some(BedCdmxData::default()),
                },
                // trim 1 + bri (1 + 1 + 15-bit sdc + 1 cdmx=0) = 19
                // bits → 24-bit envelope leaves 5 filler bits.
                filler_bits: 5,
            }),
        };
        let mut bw = BitWriter::new();
        write_oamd_common_data(&mut bw, &c, true).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(parse_oamd_common_data(&mut br, true).unwrap(), c);
    }
}
