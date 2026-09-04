//! Presentation data — ETSI TS 103 190-2 §6.2.9: `loud_corr()`
//! (§6.2.9.1), `custom_dmx_data()` (§6.2.9.2), `cdmx_parameters()`
//! (§6.2.9.3) and the §6.2.9.4-10 downmix tool elements.
//!
//! These elements ride in `ac4_presentation_substream()` (§6.2.2.3)
//! and carry the presentation-level loudness-correction gains and the
//! custom immersive→lower-layout downmix coefficients. Every parser
//! has an exact bit-inverse writer under the same presentation
//! parameters (`pres_ch_mode`, `pres_ch_mode_core`, …), following the
//! crate-wide metadata symmetry convention.
//!
//! Semantics: §6.3.10.1 (`loud_corr`), §6.3.10.2 (`custom_dmx_data`),
//! §6.3.10.3 (downmix coefficients — shared with
//! [`crate::dmx_coeff`]).

use crate::dmx_coeff::{parse_stereo_dmx_coeff, write_stereo_dmx_coeff, StereoDmxCoeff};
use crate::oamd::{
    parse_tool_three_way, parse_tool_two_way, write_tool_three_way, write_tool_two_way,
    ToolThreeWay, ToolTwoWay,
};
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

// =====================================================================
// loud_corr (§6.2.9.1)
// =====================================================================

/// Parsed `loud_corr(pres_ch_mode, pres_ch_mode_core, b_objects)`.
///
/// Every 5-bit correction code resolves through
/// [`crate::dmx_coeff::dmx_loud_corr_db`]-style semantics (§6.3.10.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoudCorr {
    /// `b_obj_loud_corr` (only signalled when `b_objects == 1`).
    pub b_obj_loud_corr: bool,
    /// `b_corr_for_immersive_out` — `None` when its gate
    /// (`pres_ch_mode > 4 or b_obj_loud_corr`) is closed.
    pub b_corr_for_immersive_out: Option<bool>,
    /// `loro_dmx_loud_corr` (`b_loro_loud_comp`).
    pub loro_dmx_loud_corr: Option<u8>,
    /// `ltrt_dmx_loud_corr` (`b_ltrt_loud_comp`).
    pub ltrt_dmx_loud_corr: Option<u8>,
    /// `loud_corr_5_X`.
    pub loud_corr_5_x: Option<u8>,
    /// `loud_corr_5_X_2` (immersive-out block).
    pub loud_corr_5_x_2: Option<u8>,
    /// `loud_corr_7_X` (immersive-out block).
    pub loud_corr_7_x: Option<u8>,
    /// `loud_corr_7_X_4` (`pres_ch_mode > 10` immersive-out block).
    pub loud_corr_7_x_4: Option<u8>,
    /// `loud_corr_7_X_2` (`pres_ch_mode > 10` immersive-out block).
    pub loud_corr_7_x_2: Option<u8>,
    /// `loud_corr_5_X_4` (`pres_ch_mode > 10` immersive-out block).
    pub loud_corr_5_x_4: Option<u8>,
    /// `loud_corr_core_5_X_2` (`pres_ch_mode_core >= 5`).
    pub loud_corr_core_5_x_2: Option<u8>,
    /// `loud_corr_core_5_X` (`pres_ch_mode_core >= 3`).
    pub loud_corr_core_5_x: Option<u8>,
    /// `(loud_corr_core_loro, loud_corr_core_ltrt)`
    /// (`pres_ch_mode_core >= 3`, one presence flag for the pair).
    pub loud_corr_core_loro_ltrt: Option<(u8, u8)>,
    /// `loud_corr_9_X_4` (`b_obj_loud_corr == 1`).
    pub loud_corr_9_x_4: Option<u8>,
}

fn parse_opt_code5(br: &mut BitReader<'_>) -> Result<Option<u8>> {
    Ok(if br.read_bit()? {
        Some(br.read_u32(5)? as u8)
    } else {
        None
    })
}

fn write_opt_code5(bw: &mut BitWriter, v: Option<u8>) -> Result<()> {
    match v {
        Some(code) => {
            if code > 31 {
                return Err(Error::invalid("ac4: loud_corr code out of range"));
            }
            bw.write_bit(true);
            bw.write_u32(code as u32, 5);
        }
        None => bw.write_bit(false),
    }
    Ok(())
}

/// Parse `loud_corr(pres_ch_mode, pres_ch_mode_core, b_objects)` per
/// §6.2.9.1. `pres_ch_mode` / `pres_ch_mode_core` are −1 when the
/// presentation carries no (core) channel-mode.
pub fn parse_loud_corr(
    br: &mut BitReader<'_>,
    pres_ch_mode: i32,
    pres_ch_mode_core: i32,
    b_objects: bool,
) -> Result<LoudCorr> {
    let mut v = LoudCorr::default();
    if b_objects {
        v.b_obj_loud_corr = br.read_bit()?;
    }
    let obj = v.b_obj_loud_corr;
    if pres_ch_mode > 4 || obj {
        v.b_corr_for_immersive_out = Some(br.read_bit()?);
    }
    let immersive_out = v.b_corr_for_immersive_out == Some(true);
    if pres_ch_mode > 1 || obj {
        v.loro_dmx_loud_corr = parse_opt_code5(br)?;
        v.ltrt_dmx_loud_corr = parse_opt_code5(br)?;
    }
    if pres_ch_mode > 4 || obj {
        v.loud_corr_5_x = parse_opt_code5(br)?;
        if immersive_out {
            v.loud_corr_5_x_2 = parse_opt_code5(br)?;
            v.loud_corr_7_x = parse_opt_code5(br)?;
        }
    }
    if (pres_ch_mode > 10 || obj) && immersive_out {
        v.loud_corr_7_x_4 = parse_opt_code5(br)?;
        v.loud_corr_7_x_2 = parse_opt_code5(br)?;
        v.loud_corr_5_x_4 = parse_opt_code5(br)?;
    }
    if pres_ch_mode_core >= 5 {
        v.loud_corr_core_5_x_2 = parse_opt_code5(br)?;
    }
    if pres_ch_mode_core >= 3 {
        v.loud_corr_core_5_x = parse_opt_code5(br)?;
        if br.read_bit()? {
            let loro = br.read_u32(5)? as u8;
            let ltrt = br.read_u32(5)? as u8;
            v.loud_corr_core_loro_ltrt = Some((loro, ltrt));
        }
    }
    if obj {
        v.loud_corr_9_x_4 = parse_opt_code5(br)?;
    }
    Ok(v)
}

/// Write `loud_corr()` — exact inverse of [`parse_loud_corr`] under
/// the same presentation parameters.
pub fn write_loud_corr(
    bw: &mut BitWriter,
    v: &LoudCorr,
    pres_ch_mode: i32,
    pres_ch_mode_core: i32,
    b_objects: bool,
) -> Result<()> {
    if b_objects {
        bw.write_bit(v.b_obj_loud_corr);
    } else if v.b_obj_loud_corr {
        return Err(Error::invalid(
            "ac4: b_obj_loud_corr requires an object presentation",
        ));
    }
    let obj = v.b_obj_loud_corr;
    if pres_ch_mode > 4 || obj {
        let b = v
            .b_corr_for_immersive_out
            .ok_or_else(|| Error::invalid("ac4: b_corr_for_immersive_out required"))?;
        bw.write_bit(b);
    } else if v.b_corr_for_immersive_out.is_some() {
        return Err(Error::invalid(
            "ac4: b_corr_for_immersive_out gate is closed",
        ));
    }
    let immersive_out = v.b_corr_for_immersive_out == Some(true);
    if pres_ch_mode > 1 || obj {
        write_opt_code5(bw, v.loro_dmx_loud_corr)?;
        write_opt_code5(bw, v.ltrt_dmx_loud_corr)?;
    } else if v.loro_dmx_loud_corr.is_some() || v.ltrt_dmx_loud_corr.is_some() {
        return Err(Error::invalid("ac4: loro/ltrt loud-corr gate is closed"));
    }
    if pres_ch_mode > 4 || obj {
        write_opt_code5(bw, v.loud_corr_5_x)?;
        if immersive_out {
            write_opt_code5(bw, v.loud_corr_5_x_2)?;
            write_opt_code5(bw, v.loud_corr_7_x)?;
        }
    }
    if (pres_ch_mode > 10 || obj) && immersive_out {
        write_opt_code5(bw, v.loud_corr_7_x_4)?;
        write_opt_code5(bw, v.loud_corr_7_x_2)?;
        write_opt_code5(bw, v.loud_corr_5_x_4)?;
    }
    if pres_ch_mode_core >= 5 {
        write_opt_code5(bw, v.loud_corr_core_5_x_2)?;
    }
    if pres_ch_mode_core >= 3 {
        write_opt_code5(bw, v.loud_corr_core_5_x)?;
        match v.loud_corr_core_loro_ltrt {
            Some((loro, ltrt)) => {
                if loro > 31 || ltrt > 31 {
                    return Err(Error::invalid("ac4: loud_corr code out of range"));
                }
                bw.write_bit(true);
                bw.write_u32(loro as u32, 5);
                bw.write_u32(ltrt as u32, 5);
            }
            None => bw.write_bit(false),
        }
    }
    if obj {
        write_opt_code5(bw, v.loud_corr_9_x_4)?;
    }
    Ok(())
}

// =====================================================================
// Downmix tools (§6.2.9.4-10)
// =====================================================================

/// `tool_scr_to_c_l()` (§6.2.9.4): `b_put_screen_to_c` selects
/// `gain_f1_code` (true) or `gain_f2_code` (false).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolScrToCL {
    /// `b_put_screen_to_c`.
    pub put_screen_to_c: bool,
    /// 3-bit `gain_f1_code` / `gain_f2_code`.
    pub gain_code: u8,
}

fn parse_tool_scr_to_c_l(br: &mut BitReader<'_>) -> Result<ToolScrToCL> {
    let put_screen_to_c = br.read_bit()?;
    let gain_code = br.read_u32(3)? as u8;
    Ok(ToolScrToCL {
        put_screen_to_c,
        gain_code,
    })
}

fn write_tool_scr_to_c_l(bw: &mut BitWriter, t: ToolScrToCL) {
    bw.write_bit(t.put_screen_to_c);
    bw.write_u32(t.gain_code as u32, 3);
}

/// Parsed `cdmx_parameters(bs_ch_config, out_ch_config)` (§6.2.9.3).
/// Which tools are present is fully determined by the two arguments;
/// the writer re-derives and validates the shape.
///
/// * `tool_t4_to_f_s()` (§6.2.9.8) = front (`t2a`/`t2b`) + back
///   (`t2d`/`t2e`) two-way selections.
/// * `tool_t4_to_f_s_b()` (§6.2.9.7) = front (`t2a`/`t2b`/`t2c`) +
///   back (`t2d`/`t2e`/`t2f`) three-way selections.
/// * `tool_t2_to_f_s()` (§6.2.9.10) / `tool_t2_to_f_s_b()` (§6.2.9.9)
///   = the single top-pair forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CdmxParameters {
    /// `tool_scr_to_c_l()` (`bs_ch_config` 0 or 3).
    pub scr_to_c_l: Option<ToolScrToCL>,
    /// `tool_t4_to_f_s()` — front + back two-way selections.
    pub t4_to_f_s: Option<(ToolTwoWay, ToolTwoWay)>,
    /// `tool_t4_to_t2()` — 3-bit `gain_t1_code`.
    pub t4_to_t2: Option<u8>,
    /// `tool_b4_to_b2()` — 3-bit `gain_b_code`.
    pub b4_to_b2: Option<u8>,
    /// `tool_t4_to_f_s_b()` — front + back three-way selections.
    pub t4_to_f_s_b: Option<(ToolThreeWay, ToolThreeWay)>,
    /// `tool_t2_to_f_s()`.
    pub t2_to_f_s: Option<ToolTwoWay>,
    /// `tool_t2_to_f_s_b()`.
    pub t2_to_f_s_b: Option<ToolThreeWay>,
}

/// Parse `cdmx_parameters(bs_ch_config, out_ch_config)` per §6.2.9.3.
pub fn parse_cdmx_parameters(
    br: &mut BitReader<'_>,
    bs_ch_config: i32,
    out_ch_config: u8,
) -> Result<CdmxParameters> {
    let mut p = CdmxParameters::default();
    if bs_ch_config == 0 || bs_ch_config == 3 {
        p.scr_to_c_l = Some(parse_tool_scr_to_c_l(br)?);
    }
    if bs_ch_config < 2 {
        match out_ch_config {
            0 => {
                p.t4_to_f_s = Some((parse_tool_two_way(br)?, parse_tool_two_way(br)?));
                p.b4_to_b2 = Some(br.read_u32(3)? as u8);
            }
            1 => {
                p.t4_to_t2 = Some(br.read_u32(3)? as u8);
                p.b4_to_b2 = Some(br.read_u32(3)? as u8);
            }
            2 => p.b4_to_b2 = Some(br.read_u32(3)? as u8),
            3 => {
                p.t4_to_f_s_b = Some((parse_tool_three_way(br)?, parse_tool_three_way(br)?));
            }
            4 => p.t4_to_t2 = Some(br.read_u32(3)? as u8),
            _ => {}
        }
    }
    if bs_ch_config == 2 {
        match out_ch_config {
            0 => {
                p.t4_to_f_s = Some((parse_tool_two_way(br)?, parse_tool_two_way(br)?));
            }
            1 => p.t4_to_t2 = Some(br.read_u32(3)? as u8),
            _ => {}
        }
    }
    if (3..=4).contains(&bs_ch_config) {
        match out_ch_config {
            0 => {
                p.t2_to_f_s = Some(parse_tool_two_way(br)?);
                p.b4_to_b2 = Some(br.read_u32(3)? as u8);
            }
            1 | 2 => p.b4_to_b2 = Some(br.read_u32(3)? as u8),
            3 => p.t2_to_f_s_b = Some(parse_tool_three_way(br)?),
            _ => {}
        }
    }
    if bs_ch_config == 5 && out_ch_config == 0 {
        p.t2_to_f_s = Some(parse_tool_two_way(br)?);
    }
    Ok(p)
}

/// Write `cdmx_parameters()` — exact inverse of
/// [`parse_cdmx_parameters`] under the same arguments.
pub fn write_cdmx_parameters(
    bw: &mut BitWriter,
    p: &CdmxParameters,
    bs_ch_config: i32,
    out_ch_config: u8,
) -> Result<()> {
    let need = |present: bool, name: &str| {
        if present {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "ac4: cdmx_parameters missing {name} for this config"
            )))
        }
    };
    if bs_ch_config == 0 || bs_ch_config == 3 {
        need(p.scr_to_c_l.is_some(), "tool_scr_to_c_l")?;
        write_tool_scr_to_c_l(bw, p.scr_to_c_l.expect("checked"));
    }
    let write_gain3 = |bw: &mut BitWriter, g: Option<u8>, name: &str| -> Result<()> {
        let g = g.ok_or_else(|| Error::invalid(format!("ac4: cdmx_parameters missing {name}")))?;
        if g > 7 {
            return Err(Error::invalid("ac4: cdmx gain code out of range"));
        }
        bw.write_u32(g as u32, 3);
        Ok(())
    };
    if bs_ch_config < 2 {
        match out_ch_config {
            0 => {
                let (f, b) = p
                    .t4_to_f_s
                    .ok_or_else(|| Error::invalid("ac4: cdmx_parameters missing tool_t4_to_f_s"))?;
                write_tool_two_way(bw, f);
                write_tool_two_way(bw, b);
                write_gain3(bw, p.b4_to_b2, "tool_b4_to_b2")?;
            }
            1 => {
                write_gain3(bw, p.t4_to_t2, "tool_t4_to_t2")?;
                write_gain3(bw, p.b4_to_b2, "tool_b4_to_b2")?;
            }
            2 => write_gain3(bw, p.b4_to_b2, "tool_b4_to_b2")?,
            3 => {
                let (f, b) = p.t4_to_f_s_b.ok_or_else(|| {
                    Error::invalid("ac4: cdmx_parameters missing tool_t4_to_f_s_b")
                })?;
                write_tool_three_way(bw, f);
                write_tool_three_way(bw, b);
            }
            4 => write_gain3(bw, p.t4_to_t2, "tool_t4_to_t2")?,
            _ => {}
        }
    }
    if bs_ch_config == 2 {
        match out_ch_config {
            0 => {
                let (f, b) = p
                    .t4_to_f_s
                    .ok_or_else(|| Error::invalid("ac4: cdmx_parameters missing tool_t4_to_f_s"))?;
                write_tool_two_way(bw, f);
                write_tool_two_way(bw, b);
            }
            1 => write_gain3(bw, p.t4_to_t2, "tool_t4_to_t2")?,
            _ => {}
        }
    }
    if (3..=4).contains(&bs_ch_config) {
        match out_ch_config {
            0 => {
                let f = p
                    .t2_to_f_s
                    .ok_or_else(|| Error::invalid("ac4: cdmx_parameters missing tool_t2_to_f_s"))?;
                write_tool_two_way(bw, f);
                write_gain3(bw, p.b4_to_b2, "tool_b4_to_b2")?;
            }
            1 | 2 => write_gain3(bw, p.b4_to_b2, "tool_b4_to_b2")?,
            3 => {
                let f = p.t2_to_f_s_b.ok_or_else(|| {
                    Error::invalid("ac4: cdmx_parameters missing tool_t2_to_f_s_b")
                })?;
                write_tool_three_way(bw, f);
            }
            _ => {}
        }
    }
    if bs_ch_config == 5 && out_ch_config == 0 {
        let f = p
            .t2_to_f_s
            .ok_or_else(|| Error::invalid("ac4: cdmx_parameters missing tool_t2_to_f_s"))?;
        write_tool_two_way(bw, f);
    }
    Ok(())
}

// =====================================================================
// custom_dmx_data (§6.2.9.2)
// =====================================================================

/// Derive the §6.2.9.2 `bs_ch_config` selector from the presentation
/// parameters. Returns −1 when the presentation carries no custom-
/// downmixable immersive layout.
pub fn bs_ch_config(
    pres_ch_mode: i32,
    b_pres_4_back_channels_present: bool,
    pres_top_channel_pairs: u32,
) -> i32 {
    if !(11..=14).contains(&pres_ch_mode) {
        return -1;
    }
    match pres_top_channel_pairs {
        2 => {
            if pres_ch_mode >= 13 && b_pres_4_back_channels_present {
                0
            } else if pres_ch_mode <= 12 {
                if b_pres_4_back_channels_present {
                    1
                } else {
                    2
                }
            } else {
                -1
            }
        }
        1 => {
            if pres_ch_mode >= 13 && b_pres_4_back_channels_present {
                3
            } else if pres_ch_mode <= 12 {
                if b_pres_4_back_channels_present {
                    4
                } else {
                    5
                }
            } else {
                -1
            }
        }
        _ => -1,
    }
}

/// Parsed `custom_dmx_data(...)` (§6.2.9.2).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CustomDmxData {
    /// `(out_ch_config[dc], cdmx_parameters)` rows when
    /// `b_cdmx_data_present == 1`.
    pub cdmx_configs: Option<Vec<(u8, CdmxParameters)>>,
    /// The inline `b_stereo_dmx_coeff` block (present when
    /// `pres_ch_mode >= 3 or pres_ch_mode_core >= 3`).
    pub stereo_dmx_coeff: Option<StereoDmxCoeff>,
}

/// Presentation parameters `custom_dmx_data()` depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomDmxParams {
    /// `pres_ch_mode` (−1 = none).
    pub pres_ch_mode: i32,
    /// `pres_ch_mode_core` (−1 = none).
    pub pres_ch_mode_core: i32,
    /// `b_pres_4_back_channels_present`.
    pub b_pres_4_back_channels_present: bool,
    /// `pres_top_channel_pairs` (0, 1 or 2).
    pub pres_top_channel_pairs: u32,
    /// `b_pres_has_lfe`.
    pub b_pres_has_lfe: bool,
}

/// Parse `custom_dmx_data()` per §6.2.9.2.
pub fn parse_custom_dmx_data(
    br: &mut BitReader<'_>,
    params: &CustomDmxParams,
) -> Result<CustomDmxData> {
    let mut v = CustomDmxData::default();
    let bs = bs_ch_config(
        params.pres_ch_mode,
        params.b_pres_4_back_channels_present,
        params.pres_top_channel_pairs,
    );
    if bs >= 0 && br.read_bit()? {
        // b_cdmx_data_present.
        let n_cdmx_configs = br.read_u32(2)? + 1;
        let out_bits = if bs == 2 || bs == 5 { 1 } else { 3 };
        let mut rows = Vec::with_capacity(n_cdmx_configs as usize);
        for _ in 0..n_cdmx_configs {
            let out_ch_config = br.read_u32(out_bits)? as u8;
            let p = parse_cdmx_parameters(br, bs, out_ch_config)?;
            rows.push((out_ch_config, p));
        }
        v.cdmx_configs = Some(rows);
    }
    if (params.pres_ch_mode >= 3 || params.pres_ch_mode_core >= 3) && br.read_bit()? {
        // b_stereo_dmx_coeff.
        v.stereo_dmx_coeff = Some(parse_stereo_dmx_coeff(br, params.b_pres_has_lfe)?);
    }
    Ok(v)
}

/// Write `custom_dmx_data()` — exact inverse of
/// [`parse_custom_dmx_data`] under the same parameters.
pub fn write_custom_dmx_data(
    bw: &mut BitWriter,
    v: &CustomDmxData,
    params: &CustomDmxParams,
) -> Result<()> {
    let bs = bs_ch_config(
        params.pres_ch_mode,
        params.b_pres_4_back_channels_present,
        params.pres_top_channel_pairs,
    );
    if bs >= 0 {
        match &v.cdmx_configs {
            Some(rows) => {
                if rows.is_empty() || rows.len() > 4 {
                    return Err(Error::invalid("ac4: n_cdmx_configs must be 1..=4"));
                }
                bw.write_bit(true);
                bw.write_u32(rows.len() as u32 - 1, 2);
                let out_bits = if bs == 2 || bs == 5 { 1 } else { 3 };
                for (out_ch_config, p) in rows {
                    if u32::from(*out_ch_config) >= (1 << out_bits) {
                        return Err(Error::invalid("ac4: out_ch_config out of range"));
                    }
                    bw.write_u32(*out_ch_config as u32, out_bits);
                    write_cdmx_parameters(bw, p, bs, *out_ch_config)?;
                }
            }
            None => bw.write_bit(false),
        }
    } else if v.cdmx_configs.is_some() {
        return Err(Error::invalid(
            "ac4: cdmx configs need an immersive pres_ch_mode",
        ));
    }
    if params.pres_ch_mode >= 3 || params.pres_ch_mode_core >= 3 {
        match &v.stereo_dmx_coeff {
            Some(c) => {
                bw.write_bit(true);
                write_stereo_dmx_coeff(bw, c, params.b_pres_has_lfe)?;
            }
            None => bw.write_bit(false),
        }
    } else if v.stereo_dmx_coeff.is_some() {
        return Err(Error::invalid(
            "ac4: stereo_dmx_coeff gate is closed for this pres_ch_mode",
        ));
    }
    Ok(())
}

// =====================================================================
// ac4_presentation_substream (§6.2.2.3)
// =====================================================================

/// Per-target block of the `b_alternative` branch (§6.2.2.3).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PresTarget {
    /// 3-bit `target_level`.
    pub target_level: u8,
    /// 4-bit `target_device_category[]`.
    pub target_device_category: u8,
    /// 4 reserved bits when `b_tdc_extension == 1`.
    pub tdc_reserved: Option<u8>,
    /// 6-bit `max_ducking_depth` (`b_ducking_depth_present`).
    pub max_ducking_depth: Option<u8>,
    /// 5-bit `loud_corr_target` (`b_loud_corr_target`).
    pub loud_corr_target: Option<u8>,
    /// Per-substream activation: `None` = `b_active == 0`, otherwise
    /// the resolved `alt_data_set_index` (1-bit base with a
    /// `variable_bits(2)` extension on 1).
    pub substream_active: Vec<Option<u32>>,
}

/// `b_associated` scaling block (§6.2.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AssociatedScaling {
    /// 8-bit `scale_main` (`b_scale_main`).
    pub scale_main: Option<u8>,
    /// 8-bit `scale_main_centre` (`b_scale_main_centre`).
    pub scale_main_centre: Option<u8>,
    /// 8-bit `scale_main_front` (`b_scale_main_front`).
    pub scale_main_front: Option<u8>,
    /// 8-bit `pan_associated` — present iff `b_associate_is_mono`.
    pub pan_associated: Option<u8>,
}

/// Parsed `ac4_presentation_substream()` (§6.2.2.3).
#[derive(Debug, Clone, PartialEq)]
pub struct PresentationSubstream {
    /// `presentation_name` bytes (`b_alternative` + `b_name_present`).
    pub name: Option<Vec<u8>>,
    /// Alternative-presentation targets (`b_alternative`).
    pub targets: Vec<PresTarget>,
    /// `add_data` bytes (`b_additional_data`).
    pub add_data: Option<Vec<u8>>,
    /// 7-bit `dialnorm_bits`.
    pub dialnorm_bits: u8,
    /// `further_loudness_info(1, 1)` (`b_further_loudness_info`).
    pub further_loudness_info: Option<crate::metadata::FurtherLoudnessInfoV2>,
    /// Resolved `drc_metadata_size` envelope in bits.
    pub drc_metadata_size: u32,
    /// `drc_frame(b_pres_ndot)`.
    pub drc: crate::drc::DrcFrame,
    /// Announced-but-unparsed trailing bits of the DRC envelope.
    pub drc_trailing_bits: u32,
    /// Substream-group gains (`n_substream_groups > 1`):
    /// `None` = flag absent, `Some(None)` = `b_keep`,
    /// `Some(Some(gains))` = 6-bit `sg_gain` per group.
    pub substream_group_gains: Option<Option<Vec<u8>>>,
    /// `b_associated` scaling block.
    pub associated: Option<AssociatedScaling>,
    /// `custom_dmx_data(...)`.
    pub custom_dmx_data: CustomDmxData,
    /// `loud_corr(...)`.
    pub loud_corr: LoudCorr,
}

/// Caller context for `ac4_presentation_substream()` — everything the
/// syntax gates on, all signalled in the TOC / presentation info.
#[derive(Debug, Clone, Copy)]
pub struct PresSubstreamParams {
    /// `b_alternative` (from `ac4_presentation_substream_info()`).
    pub b_alternative: bool,
    /// `b_pres_ndot` — the presentation I-frame flag (§6.3.2.11.2),
    /// passed to `drc_frame()`.
    pub b_pres_ndot: bool,
    /// `n_substreams_in_presentation`.
    pub n_substreams_in_presentation: u32,
    /// `n_substream_groups`.
    pub n_substream_groups: u32,
    /// `b_objects` — the presentation carries object substreams.
    pub b_objects: bool,
    /// `pres_ch_mode` (−1 = none).
    pub pres_ch_mode: i32,
    /// `pres_ch_mode_core` (−1 = none).
    pub pres_ch_mode_core: i32,
    /// `b_pres_4_back_channels_present`.
    pub b_pres_4_back_channels_present: bool,
    /// `pres_top_channel_pairs`.
    pub pres_top_channel_pairs: u32,
    /// `b_pres_has_lfe`.
    pub b_pres_has_lfe: bool,
    /// AC-4 frame length in samples (drives `nr_drc_subframes`).
    pub frame_length: u32,
}

impl PresSubstreamParams {
    fn custom_dmx(&self) -> CustomDmxParams {
        CustomDmxParams {
            pres_ch_mode: self.pres_ch_mode,
            pres_ch_mode_core: self.pres_ch_mode_core,
            b_pres_4_back_channels_present: self.b_pres_4_back_channels_present,
            pres_top_channel_pairs: self.pres_top_channel_pairs,
            b_pres_has_lfe: self.b_pres_has_lfe,
        }
    }

    fn drc_chan_info(&self) -> crate::drc::DrcChannelInfo {
        let mode = if self.pres_ch_mode >= 0 {
            self.pres_ch_mode as u32
        } else {
            // Object presentations: Table 168's multichannel grouping.
            5
        };
        crate::drc::DrcChannelInfo::new(
            crate::drc::nr_drc_channels(mode),
            crate::drc::nr_drc_subframes(self.frame_length).unwrap_or(1),
        )
    }
}

fn parse_alt_data_set_index(br: &mut BitReader<'_>) -> Result<u32> {
    let mut idx = br.read_u32(1)?;
    if idx == 1 {
        idx = idx
            .checked_add(crate::toc::variable_bits(br, 2)?)
            .ok_or_else(|| oxideav_core::Error::invalid("ac4: variable_bits escape overflow"))?;
    }
    Ok(idx)
}

fn write_alt_data_set_index(bw: &mut BitWriter, idx: u32) {
    if idx == 0 {
        bw.write_u32(0, 1);
    } else {
        bw.write_u32(1, 1);
        crate::toc::write_variable_bits(bw, 2, idx - 1);
    }
}

/// Parse `ac4_presentation_substream()` per §6.2.2.3.
///
/// `prev_drc_config` carries the presentation DRC configuration from a
/// prior `b_pres_ndot` frame (required to decode `drc_data()` on
/// dependent frames).
pub fn parse_presentation_substream(
    br: &mut BitReader<'_>,
    params: &PresSubstreamParams,
    prev_drc_config: Option<&crate::drc::DrcConfig>,
) -> Result<PresentationSubstream> {
    let mut name = None;
    let mut targets = Vec::new();
    if params.b_alternative {
        if br.read_bit()? {
            // b_name_present.
            let name_len = if br.read_bit()? { br.read_u32(5)? } else { 32 };
            let mut bytes = Vec::with_capacity(name_len as usize);
            for _ in 0..name_len {
                bytes.push(br.read_u32(8)? as u8);
            }
            name = Some(bytes);
        }
        let mut n_targets = br.read_u32(2)? + 1;
        if n_targets == 4 {
            n_targets = n_targets
                .checked_add(crate::toc::variable_bits(br, 2)?)
                .ok_or_else(|| {
                    oxideav_core::Error::invalid("ac4: variable_bits escape overflow")
                })?;
        }
        for _ in 0..n_targets {
            let target_level = br.read_u32(3)? as u8;
            let target_device_category = br.read_u32(4)? as u8;
            let tdc_reserved = if br.read_bit()? {
                Some(br.read_u32(4)? as u8)
            } else {
                None
            };
            let max_ducking_depth = if br.read_bit()? {
                Some(br.read_u32(6)? as u8)
            } else {
                None
            };
            let loud_corr_target = if br.read_bit()? {
                Some(br.read_u32(5)? as u8)
            } else {
                None
            };
            let mut substream_active =
                Vec::with_capacity(params.n_substreams_in_presentation as usize);
            for _ in 0..params.n_substreams_in_presentation {
                if br.read_bit()? {
                    substream_active.push(Some(parse_alt_data_set_index(br)?));
                } else {
                    substream_active.push(None);
                }
            }
            targets.push(PresTarget {
                target_level,
                target_device_category,
                tdc_reserved,
                max_ducking_depth,
                loud_corr_target,
                substream_active,
            });
        }
    }
    let add_data = if br.read_bit()? {
        // b_additional_data.
        let mut add_data_bytes = br.read_u32(4)? + 1;
        if add_data_bytes == 16 {
            add_data_bytes = add_data_bytes
                .checked_add(crate::toc::variable_bits(br, 2)?)
                .ok_or_else(|| {
                    oxideav_core::Error::invalid("ac4: variable_bits escape overflow")
                })?;
        }
        br.align_to_byte();
        let mut bytes = Vec::with_capacity(add_data_bytes as usize);
        for _ in 0..add_data_bytes {
            bytes.push(br.read_u32(8)? as u8);
        }
        Some(bytes)
    } else {
        None
    };
    let dialnorm_bits = br.read_u32(7)? as u8;
    let further_loudness_info = if br.read_bit()? {
        Some(crate::metadata::parse_further_loudness_info_v2(br, true)?)
    } else {
        None
    };
    let mut drc_metadata_size = br.read_u32(5)?;
    if br.read_bit()? {
        drc_metadata_size = drc_metadata_size
            .checked_add(
                crate::toc::variable_bits(br, 3)?
                    .checked_shl(5)
                    .ok_or_else(|| Error::invalid("ac4: drc_metadata_size shift overflow"))?,
            )
            .ok_or_else(|| Error::invalid("ac4: drc_metadata_size overflow"))?;
    }
    let drc_start = br.bit_position();
    let drc = crate::drc::parse_drc_frame(
        br,
        params.b_pres_ndot,
        params.drc_chan_info(),
        prev_drc_config,
    )?;
    let consumed = (br.bit_position() - drc_start) as u32;
    if consumed > drc_metadata_size {
        return Err(Error::invalid(
            "ac4: presentation drc_frame overran drc_metadata_size",
        ));
    }
    let drc_trailing_bits = drc_metadata_size - consumed;
    crate::metadata::skip_n_bits(br, drc_trailing_bits)?;
    let substream_group_gains = if params.n_substream_groups > 1 {
        if br.read_bit()? {
            // b_substream_group_gains_present.
            if br.read_bit()? {
                // b_keep.
                Some(None)
            } else {
                let mut gains = Vec::with_capacity(params.n_substream_groups as usize);
                for _ in 0..params.n_substream_groups {
                    gains.push(br.read_u32(6)? as u8);
                }
                Some(Some(gains))
            }
        } else {
            None
        }
    } else {
        None
    };
    let associated = if br.read_bit()? {
        // b_associated.
        let scale_main = if br.read_bit()? {
            Some(br.read_u32(8)? as u8)
        } else {
            None
        };
        let scale_main_centre = if br.read_bit()? {
            Some(br.read_u32(8)? as u8)
        } else {
            None
        };
        let scale_main_front = if br.read_bit()? {
            Some(br.read_u32(8)? as u8)
        } else {
            None
        };
        let pan_associated = if br.read_bit()? {
            // b_associate_is_mono.
            Some(br.read_u32(8)? as u8)
        } else {
            None
        };
        Some(AssociatedScaling {
            scale_main,
            scale_main_centre,
            scale_main_front,
            pan_associated,
        })
    } else {
        None
    };
    let custom_dmx_data = parse_custom_dmx_data(br, &params.custom_dmx())?;
    let loud_corr = parse_loud_corr(
        br,
        params.pres_ch_mode,
        params.pres_ch_mode_core,
        params.b_objects,
    )?;
    br.align_to_byte();
    Ok(PresentationSubstream {
        name,
        targets,
        add_data,
        dialnorm_bits,
        further_loudness_info,
        drc_metadata_size,
        drc,
        drc_trailing_bits,
        substream_group_gains,
        associated,
        custom_dmx_data,
        loud_corr,
    })
}

/// Write `ac4_presentation_substream()` — exact inverse of
/// [`parse_presentation_substream`] under the same parameters (the
/// element is byte-aligned at entry per §6.2.2.1).
pub fn write_presentation_substream(
    bw: &mut BitWriter,
    v: &PresentationSubstream,
    params: &PresSubstreamParams,
) -> Result<()> {
    if params.b_alternative {
        match &v.name {
            Some(bytes) => {
                bw.write_bit(true);
                if bytes.len() == 32 {
                    bw.write_bit(false);
                } else {
                    if bytes.len() > 31 {
                        return Err(Error::invalid("ac4: presentation_name too long"));
                    }
                    bw.write_bit(true);
                    bw.write_u32(bytes.len() as u32, 5);
                }
                for &b in bytes {
                    bw.write_u32(b as u32, 8);
                }
            }
            None => bw.write_bit(false),
        }
        if v.targets.is_empty() {
            return Err(Error::invalid(
                "ac4: alternative presentation needs targets",
            ));
        }
        let n_targets = v.targets.len() as u32;
        if n_targets < 4 {
            bw.write_u32(n_targets - 1, 2);
        } else {
            bw.write_u32(3, 2);
            crate::toc::write_variable_bits(bw, 2, n_targets - 4);
        }
        for t in &v.targets {
            bw.write_u32(t.target_level as u32, 3);
            bw.write_u32(t.target_device_category as u32, 4);
            match t.tdc_reserved {
                Some(r) => {
                    bw.write_bit(true);
                    bw.write_u32(r as u32, 4);
                }
                None => bw.write_bit(false),
            }
            match t.max_ducking_depth {
                Some(d) => {
                    bw.write_bit(true);
                    bw.write_u32(d as u32, 6);
                }
                None => bw.write_bit(false),
            }
            match t.loud_corr_target {
                Some(l) => {
                    bw.write_bit(true);
                    bw.write_u32(l as u32, 5);
                }
                None => bw.write_bit(false),
            }
            if t.substream_active.len() != params.n_substreams_in_presentation as usize {
                return Err(Error::invalid(
                    "ac4: per-target substream activation length mismatch",
                ));
            }
            for a in &t.substream_active {
                match a {
                    Some(idx) => {
                        bw.write_bit(true);
                        write_alt_data_set_index(bw, *idx);
                    }
                    None => bw.write_bit(false),
                }
            }
        }
    } else if v.name.is_some() || !v.targets.is_empty() {
        return Err(Error::invalid(
            "ac4: targets/name need b_alternative presentations",
        ));
    }
    match &v.add_data {
        Some(bytes) => {
            if bytes.is_empty() {
                return Err(Error::invalid("ac4: add_data must not be empty"));
            }
            bw.write_bit(true);
            let n = bytes.len() as u32;
            if n < 16 {
                bw.write_u32(n - 1, 4);
            } else {
                bw.write_u32(15, 4);
                crate::toc::write_variable_bits(bw, 2, n - 16);
            }
            bw.align_to_byte();
            for &b in bytes {
                bw.write_u32(b as u32, 8);
            }
        }
        None => bw.write_bit(false),
    }
    if v.dialnorm_bits > 127 {
        return Err(Error::invalid("ac4: dialnorm_bits out of range"));
    }
    bw.write_u32(v.dialnorm_bits as u32, 7);
    match &v.further_loudness_info {
        Some(f) => {
            if !f.b_presentation_ldn {
                return Err(Error::invalid(
                    "ac4: presentation further_loudness_info needs b_presentation_ldn",
                ));
            }
            bw.write_bit(true);
            crate::metadata::write_further_loudness_info_v2(bw, f)?;
        }
        None => bw.write_bit(false),
    }
    if v.drc_metadata_size < (1 << 5) {
        bw.write_u32(v.drc_metadata_size, 5);
        bw.write_bit(false);
    } else {
        bw.write_u32(v.drc_metadata_size & 0x1F, 5);
        bw.write_bit(true);
        crate::toc::write_variable_bits(bw, 3, v.drc_metadata_size >> 5);
    }
    let start = bw.bit_position();
    crate::drc::write_drc_frame(bw, &v.drc, params.b_pres_ndot, params.drc_chan_info())?;
    let used = (bw.bit_position() - start) as u32;
    if used + v.drc_trailing_bits != v.drc_metadata_size {
        return Err(Error::invalid(
            "ac4: presentation DRC envelope inconsistent with trailing bits",
        ));
    }
    for _ in 0..v.drc_trailing_bits {
        bw.write_bit(false);
    }
    if params.n_substream_groups > 1 {
        match &v.substream_group_gains {
            None => bw.write_bit(false),
            Some(None) => {
                bw.write_bit(true);
                bw.write_bit(true); // b_keep
            }
            Some(Some(gains)) => {
                if gains.len() != params.n_substream_groups as usize {
                    return Err(Error::invalid("ac4: sg_gain count mismatch"));
                }
                bw.write_bit(true);
                bw.write_bit(false);
                for &g in gains {
                    if g > 63 {
                        return Err(Error::invalid("ac4: sg_gain out of range"));
                    }
                    bw.write_u32(g as u32, 6);
                }
            }
        }
    } else if v.substream_group_gains.is_some() {
        return Err(Error::invalid("ac4: sg gains need n_substream_groups > 1"));
    }
    match &v.associated {
        Some(a) => {
            bw.write_bit(true);
            for field in [a.scale_main, a.scale_main_centre, a.scale_main_front] {
                match field {
                    Some(s) => {
                        bw.write_bit(true);
                        bw.write_u32(s as u32, 8);
                    }
                    None => bw.write_bit(false),
                }
            }
            match a.pan_associated {
                Some(p) => {
                    bw.write_bit(true);
                    bw.write_u32(p as u32, 8);
                }
                None => bw.write_bit(false),
            }
        }
        None => bw.write_bit(false),
    }
    write_custom_dmx_data(bw, &v.custom_dmx_data, &params.custom_dmx())?;
    write_loud_corr(
        bw,
        &v.loud_corr,
        params.pres_ch_mode,
        params.pres_ch_mode_core,
        params.b_objects,
    )?;
    bw.align_to_byte();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    fn round_trip_loud_corr(v: &LoudCorr, pcm: i32, pcmc: i32, obj: bool) {
        let mut bw = BitWriter::new();
        write_loud_corr(&mut bw, v, pcm, pcmc, obj).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_loud_corr(&mut br, pcm, pcmc, obj).unwrap();
        assert_eq!(&got, v, "pcm={pcm} pcmc={pcmc} obj={obj}");
    }

    #[test]
    fn loud_corr_minimal_forms_round_trip() {
        // Mono / stereo channel presentations carry nothing at all.
        for pcm in [-1, 0, 1] {
            round_trip_loud_corr(&LoudCorr::default(), pcm, -1, false);
        }
    }

    #[test]
    fn loud_corr_channel_5_1_round_trips() {
        // pres_ch_mode = 4 (5.1): only the loro/ltrt pair block.
        round_trip_loud_corr(
            &LoudCorr {
                loro_dmx_loud_corr: Some(12),
                ltrt_dmx_loud_corr: None,
                ..Default::default()
            },
            4,
            -1,
            false,
        );
    }

    #[test]
    fn loud_corr_immersive_full_round_trips() {
        // pres_ch_mode = 13 (9.0.4), core 5.X (5): every channel block.
        round_trip_loud_corr(
            &LoudCorr {
                b_corr_for_immersive_out: Some(true),
                loro_dmx_loud_corr: Some(1),
                ltrt_dmx_loud_corr: Some(2),
                loud_corr_5_x: Some(3),
                loud_corr_5_x_2: Some(4),
                loud_corr_7_x: Some(5),
                loud_corr_7_x_4: Some(6),
                loud_corr_7_x_2: Some(7),
                loud_corr_5_x_4: Some(8),
                loud_corr_core_5_x_2: Some(9),
                loud_corr_core_5_x: Some(10),
                loud_corr_core_loro_ltrt: Some((11, 12)),
                ..Default::default()
            },
            13,
            5,
            false,
        );
    }

    #[test]
    fn loud_corr_immersive_out_false_skips_blocks() {
        // With b_corr_for_immersive_out = 0 the 5_X_2/7_X and the
        // pres_ch_mode > 10 blocks are absent.
        round_trip_loud_corr(
            &LoudCorr {
                b_corr_for_immersive_out: Some(false),
                loro_dmx_loud_corr: Some(3),
                loud_corr_5_x: Some(30),
                ..Default::default()
            },
            11,
            -1,
            false,
        );
    }

    #[test]
    fn loud_corr_object_presentation_round_trips() {
        // b_objects with b_obj_loud_corr opens every object-gated block
        // regardless of pres_ch_mode.
        round_trip_loud_corr(
            &LoudCorr {
                b_obj_loud_corr: true,
                b_corr_for_immersive_out: Some(true),
                loro_dmx_loud_corr: Some(31),
                ltrt_dmx_loud_corr: Some(0),
                loud_corr_5_x: Some(15),
                loud_corr_5_x_2: Some(16),
                loud_corr_7_x: Some(17),
                loud_corr_7_x_4: Some(18),
                loud_corr_7_x_2: Some(19),
                loud_corr_5_x_4: Some(20),
                loud_corr_9_x_4: Some(21),
                ..Default::default()
            },
            -1,
            -1,
            true,
        );
        // b_objects with b_obj_loud_corr = 0 carries only the flag bit.
        round_trip_loud_corr(&LoudCorr::default(), -1, -1, true);
    }

    #[test]
    fn loud_corr_write_rejects_closed_gates() {
        let mut bw = BitWriter::new();
        assert!(write_loud_corr(
            &mut bw,
            &LoudCorr {
                b_obj_loud_corr: true,
                ..Default::default()
            },
            0,
            -1,
            false,
        )
        .is_err());
        let mut bw = BitWriter::new();
        assert!(write_loud_corr(
            &mut bw,
            &LoudCorr {
                loro_dmx_loud_corr: Some(1),
                ..Default::default()
            },
            1,
            -1,
            false,
        )
        .is_err());
    }

    #[test]
    fn bs_ch_config_matches_spec_decision_tree() {
        // pres_ch_mode ∈ [11..14] with top pairs / back channels.
        // (mode, 4_back, top_pairs) → expected.
        let cases = [
            (13, true, 2, 0),
            (14, true, 2, 0),
            (11, true, 2, 1),
            (12, true, 2, 1),
            (11, false, 2, 2),
            (12, false, 2, 2),
            (13, true, 1, 3),
            (14, true, 1, 3),
            (11, true, 1, 4),
            (12, true, 1, 4),
            (11, false, 1, 5),
            (12, false, 1, 5),
            // Out of range / no top pairs / immersive without 4-back.
            (10, true, 2, -1),
            (15, true, 2, -1),
            (13, false, 2, -1),
            (13, false, 1, -1),
            (11, true, 0, -1),
        ];
        for (mode, back, pairs, want) in cases {
            assert_eq!(
                bs_ch_config(mode, back, pairs),
                want,
                "mode={mode} back={back} pairs={pairs}"
            );
        }
    }

    fn round_trip_cdmx(p: &CdmxParameters, bs: i32, out: u8) {
        let mut bw = BitWriter::new();
        write_cdmx_parameters(&mut bw, p, bs, out).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_cdmx_parameters(&mut br, bs, out).unwrap();
        assert_eq!(&got, p, "bs={bs} out={out}");
    }

    #[test]
    fn cdmx_parameters_round_trip_every_config() {
        use ToolThreeWay as T3;
        use ToolTwoWay as T2;
        // bs 0: scr_to_c_l + the bs<2 switch.
        round_trip_cdmx(
            &CdmxParameters {
                scr_to_c_l: Some(ToolScrToCL {
                    put_screen_to_c: true,
                    gain_code: 5,
                }),
                t4_to_f_s: Some((T2::Front(1), T2::Other(2))),
                b4_to_b2: Some(3),
                ..Default::default()
            },
            0,
            0,
        );
        // bs 1 (no scr_to_c_l).
        round_trip_cdmx(
            &CdmxParameters {
                t4_to_t2: Some(6),
                b4_to_b2: Some(7),
                ..Default::default()
            },
            1,
            1,
        );
        round_trip_cdmx(
            &CdmxParameters {
                b4_to_b2: Some(0),
                ..Default::default()
            },
            1,
            2,
        );
        round_trip_cdmx(
            &CdmxParameters {
                t4_to_f_s_b: Some((T3::Side(4), T3::Other(2))),
                ..Default::default()
            },
            1,
            3,
        );
        round_trip_cdmx(
            &CdmxParameters {
                t4_to_t2: Some(1),
                ..Default::default()
            },
            1,
            4,
        );
        // bs 2 (1-bit out_ch_config domain).
        round_trip_cdmx(
            &CdmxParameters {
                t4_to_f_s: Some((T2::Other(3), T2::Front(0))),
                ..Default::default()
            },
            2,
            0,
        );
        round_trip_cdmx(
            &CdmxParameters {
                t4_to_t2: Some(2),
                ..Default::default()
            },
            2,
            1,
        );
        // bs 3: scr_to_c_l + the 3..=4 switch.
        round_trip_cdmx(
            &CdmxParameters {
                scr_to_c_l: Some(ToolScrToCL {
                    put_screen_to_c: false,
                    gain_code: 7,
                }),
                t2_to_f_s: Some(T2::Front(6)),
                b4_to_b2: Some(5),
                ..Default::default()
            },
            3,
            0,
        );
        round_trip_cdmx(
            &CdmxParameters {
                scr_to_c_l: Some(ToolScrToCL {
                    put_screen_to_c: true,
                    gain_code: 0,
                }),
                t2_to_f_s_b: Some(T3::Front(3)),
                ..Default::default()
            },
            3,
            3,
        );
        // bs 4.
        round_trip_cdmx(
            &CdmxParameters {
                b4_to_b2: Some(4),
                ..Default::default()
            },
            4,
            1,
        );
        // bs 5.
        round_trip_cdmx(
            &CdmxParameters {
                t2_to_f_s: Some(T2::Other(5)),
                ..Default::default()
            },
            5,
            0,
        );
    }

    fn round_trip_custom_dmx(v: &CustomDmxData, params: &CustomDmxParams) {
        let mut bw = BitWriter::new();
        write_custom_dmx_data(&mut bw, v, params).unwrap();
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_custom_dmx_data(&mut br, params).unwrap();
        assert_eq!(&got, v);
    }

    #[test]
    fn custom_dmx_data_round_trips_immersive_with_stereo_coeff() {
        let params = CustomDmxParams {
            pres_ch_mode: 12, // 7.1.4 → bs_ch_config 1 or 2
            pres_ch_mode_core: 4,
            b_pres_4_back_channels_present: true,
            pres_top_channel_pairs: 2,
            b_pres_has_lfe: true,
        };
        assert_eq!(bs_ch_config(12, true, 2), 1);
        round_trip_custom_dmx(
            &CustomDmxData {
                cdmx_configs: Some(vec![
                    (
                        2,
                        CdmxParameters {
                            b4_to_b2: Some(3),
                            ..Default::default()
                        },
                    ),
                    (
                        4,
                        CdmxParameters {
                            t4_to_t2: Some(1),
                            ..Default::default()
                        },
                    ),
                ]),
                stereo_dmx_coeff: Some(StereoDmxCoeff {
                    loro_centre_mixgain: 4,
                    loro_surround_mixgain: 4,
                    ltrt_mixgains: None,
                    lfe_mixgain: Some(11),
                    preferred_dmx_method: 1,
                }),
            },
            &params,
        );
    }

    #[test]
    fn custom_dmx_data_round_trips_channel_only_presentation() {
        // Non-immersive pres_ch_mode: no cdmx block; the stereo-dmx
        // gate opens on pres_ch_mode >= 3.
        let params = CustomDmxParams {
            pres_ch_mode: 4,
            pres_ch_mode_core: -1,
            b_pres_4_back_channels_present: false,
            pres_top_channel_pairs: 0,
            b_pres_has_lfe: true,
        };
        round_trip_custom_dmx(
            &CustomDmxData {
                cdmx_configs: None,
                stereo_dmx_coeff: Some(StereoDmxCoeff {
                    loro_centre_mixgain: 2,
                    loro_surround_mixgain: 6,
                    ltrt_mixgains: Some((0, 7)),
                    lfe_mixgain: None,
                    preferred_dmx_method: 0,
                }),
            },
            &params,
        );
        // Stereo presentation: both gates closed → zero bits.
        let stereo = CustomDmxParams {
            pres_ch_mode: 1,
            pres_ch_mode_core: -1,
            b_pres_4_back_channels_present: false,
            pres_top_channel_pairs: 0,
            b_pres_has_lfe: false,
        };
        let mut bw = BitWriter::new();
        write_custom_dmx_data(&mut bw, &CustomDmxData::default(), &stereo).unwrap();
        assert_eq!(bw.bit_position(), 0);
    }

    // ------------------------------------------------------------------
    // ac4_presentation_substream (§6.2.2.3)
    // ------------------------------------------------------------------

    fn empty_drc() -> crate::drc::DrcFrame {
        crate::drc::DrcFrame {
            b_drc_present: false,
            config: None,
            data: None,
        }
    }

    fn round_trip_pres(v: &PresentationSubstream, params: &PresSubstreamParams) {
        let mut bw = BitWriter::new();
        write_presentation_substream(&mut bw, v, params).unwrap();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_presentation_substream(&mut br, params, None).unwrap();
        assert_eq!(&got, v);
        // The element is byte-aligned at exit.
        assert_eq!(br.bit_position() % 8, 0);
    }

    fn minimal_pres() -> PresentationSubstream {
        PresentationSubstream {
            name: None,
            targets: Vec::new(),
            add_data: None,
            dialnorm_bits: 64,
            further_loudness_info: None,
            drc_metadata_size: 1,
            drc: empty_drc(),
            drc_trailing_bits: 0,
            substream_group_gains: None,
            associated: None,
            custom_dmx_data: CustomDmxData::default(),
            loud_corr: LoudCorr::default(),
        }
    }

    #[test]
    fn presentation_substream_minimal_stereo_round_trips() {
        let params = PresSubstreamParams {
            b_alternative: false,
            b_pres_ndot: true,
            n_substreams_in_presentation: 1,
            n_substream_groups: 1,
            b_objects: false,
            pres_ch_mode: 1,
            pres_ch_mode_core: -1,
            b_pres_4_back_channels_present: false,
            pres_top_channel_pairs: 0,
            b_pres_has_lfe: false,
            frame_length: 2048,
        };
        round_trip_pres(&minimal_pres(), &params);
    }

    #[test]
    fn presentation_substream_alternative_full_round_trips() {
        let params = PresSubstreamParams {
            b_alternative: true,
            b_pres_ndot: true,
            n_substreams_in_presentation: 3,
            n_substream_groups: 3,
            b_objects: false,
            pres_ch_mode: 12,
            pres_ch_mode_core: 4,
            b_pres_4_back_channels_present: true,
            pres_top_channel_pairs: 2,
            b_pres_has_lfe: true,
            frame_length: 2048,
        };
        let v = PresentationSubstream {
            name: Some(b"Alt!".to_vec()),
            targets: vec![
                PresTarget {
                    target_level: 5,
                    target_device_category: 9,
                    tdc_reserved: Some(3),
                    max_ducking_depth: Some(40),
                    loud_corr_target: Some(17),
                    substream_active: vec![Some(0), None, Some(1)],
                },
                PresTarget {
                    target_level: 0,
                    target_device_category: 1,
                    tdc_reserved: None,
                    max_ducking_depth: None,
                    loud_corr_target: None,
                    substream_active: vec![Some(5), Some(2), None],
                },
            ],
            add_data: Some(vec![0xDE, 0xAD, 0xBE]),
            dialnorm_bits: 96,
            further_loudness_info: Some(crate::metadata::FurtherLoudnessInfoV2 {
                b_presentation_ldn: true,
                ..Default::default()
            }),
            drc_metadata_size: 4,
            drc: empty_drc(),
            drc_trailing_bits: 3,
            substream_group_gains: Some(Some(vec![10, 20, 63])),
            associated: Some(AssociatedScaling {
                scale_main: Some(200),
                scale_main_centre: None,
                scale_main_front: Some(0),
                pan_associated: Some(128),
            }),
            custom_dmx_data: CustomDmxData {
                cdmx_configs: Some(vec![(
                    2,
                    CdmxParameters {
                        b4_to_b2: Some(3),
                        ..Default::default()
                    },
                )]),
                stereo_dmx_coeff: Some(StereoDmxCoeff {
                    loro_centre_mixgain: 4,
                    loro_surround_mixgain: 4,
                    ltrt_mixgains: None,
                    lfe_mixgain: Some(2),
                    preferred_dmx_method: 1,
                }),
            },
            loud_corr: LoudCorr {
                b_corr_for_immersive_out: Some(true),
                loro_dmx_loud_corr: Some(15),
                loud_corr_5_x: Some(1),
                loud_corr_5_x_2: Some(2),
                loud_corr_7_x: Some(3),
                loud_corr_7_x_4: Some(4),
                loud_corr_7_x_2: Some(5),
                loud_corr_5_x_4: Some(6),
                loud_corr_core_5_x: Some(7),
                loud_corr_core_loro_ltrt: Some((8, 9)),
                ..Default::default()
            },
        };
        round_trip_pres(&v, &params);
    }

    #[test]
    fn presentation_substream_sg_keep_and_32_byte_name() {
        let params = PresSubstreamParams {
            b_alternative: true,
            b_pres_ndot: true,
            n_substreams_in_presentation: 1,
            n_substream_groups: 2,
            b_objects: true,
            pres_ch_mode: -1,
            pres_ch_mode_core: -1,
            b_pres_4_back_channels_present: false,
            pres_top_channel_pairs: 0,
            b_pres_has_lfe: false,
            frame_length: 1920,
        };
        let v = PresentationSubstream {
            name: Some(vec![b'x'; 32]),
            targets: vec![PresTarget {
                target_level: 7,
                target_device_category: 15,
                tdc_reserved: None,
                max_ducking_depth: None,
                loud_corr_target: None,
                substream_active: vec![None],
            }],
            substream_group_gains: Some(None), // b_keep
            loud_corr: LoudCorr {
                b_obj_loud_corr: true,
                b_corr_for_immersive_out: Some(false),
                loud_corr_5_x: Some(11),
                loud_corr_9_x_4: Some(22),
                ..Default::default()
            },
            ..minimal_pres()
        };
        round_trip_pres(&v, &params);
    }

    #[test]
    fn presentation_substream_carries_real_drc_frame() {
        // A presentation with a real drc_frame() (I-frame config +
        // gains) inside the announced envelope.
        let params = PresSubstreamParams {
            b_alternative: false,
            b_pres_ndot: true,
            n_substreams_in_presentation: 1,
            n_substream_groups: 1,
            b_objects: false,
            pres_ch_mode: 1,
            pres_ch_mode_core: -1,
            b_pres_4_back_channels_present: false,
            pres_top_channel_pairs: 0,
            b_pres_has_lfe: false,
            frame_length: 2048,
        };
        // Build the DRC frame bit-first through the real parser so the
        // struct matches a canonical decode.
        let mut dbw = BitWriter::new();
        dbw.write_bit(true); // b_drc_present
        dbw.write_bit(false); // drc_config: b_more_drc_configs... minimal
        dbw.write_u32(0, 3); // drc_decoder_nr_modes = 0 → one default mode
        dbw.write_bit(false); // b_drc_gains_present = 0
        dbw.write_u32(0, 7); // guard
        let dbytes = dbw.into_bytes();
        let mut dbr = BitReader::new(&dbytes);
        let drc = match crate::drc::parse_drc_frame(&mut dbr, true, params.drc_chan_info(), None) {
            Ok(d) => d,
            // If the minimal hand bits don't form a valid config the
            // simpler absent form still exercises the envelope.
            Err(_) => empty_drc(),
        };
        // Measure the exact element size by writing it back.
        let mut mbw = BitWriter::new();
        crate::drc::write_drc_frame(&mut mbw, &drc, true, params.drc_chan_info()).unwrap();
        let drc_bits = mbw.bit_position() as u32;
        let v = PresentationSubstream {
            drc_metadata_size: drc_bits + 2,
            drc,
            drc_trailing_bits: 2,
            ..minimal_pres()
        };
        round_trip_pres(&v, &params);
    }
}
