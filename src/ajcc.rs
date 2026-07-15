//! A-JCC (Advanced Joint Channel Coding) — ETSI TS 103 190-2 §6.2.6
//! bitstream layer + §5.6.3.2 differential decoding / dequantization.
//!
//! A-JCC represents multichannel (9.X.4-style) content as a
//! five-channel core plus parametric side information (§5.6.1). The
//! side information is a fixed roster of parameter SETs (§5.6.3.2):
//! per-channel dry / wet gains plus, in the non-`b_5fronts` layout,
//! A-CPL-style alpha / beta prediction parameters.
//!
//! This module lands the complete parameter path:
//!
//! * **Codebooks** — the twelve `AJCC_HCB_<DRY|WET>_<COARSE|FINE>_<F0|DF|DT>`
//!   Huffman codebooks (Annex A.1.2 Tables A.13-A.24, numeric arrays
//!   from the ETSI companion table file, verified complete prefix
//!   codes). ALPHA / BETA parameters reuse the Part 1 A-CPL codebooks
//!   via [`crate::acpl::get_acpl_hcb`] exactly as `get_ajcc_hcb()`
//!   (§6.3.7.3.2 Table 116) prescribes.
//! * **`ajcc_huff_data()`** (§6.2.6.4) — F0 absolute (plain
//!   `huff_decode`, i.e. the raw codeword index) + DF differences on
//!   the frequency path, DT differences on the time path; `b_no_dt`
//!   elides the per-set `diff_type` bit (Table 107).
//! * **`ajcc_framing_data()`** (§6.2.6.2) — smooth vs steep
//!   interpolation, 1 or 2 parameter sets, steep switch timeslots.
//! * **`ajced()` / `ajcc_data()`** (§6.2.6.3 / §6.2.6.1) — the full
//!   element for both layouts (`b_5fronts` 5.x.4-front and the
//!   core-mode 5-channel layout with alpha / beta).
//! * **Differential decode** (§5.6.3.2 Table 29) — per-SET running
//!   frequency sums (plain addition — A-JCC has *no* modulo, unlike
//!   the A-JOC Table 43) and time deltas against `ajcc_<SET>_q_prev`
//!   carried across parameter sets and AC-4 frames in [`AjccState`].
//! * **Dequantization** — dry `q·Δ − 0,6` (Table 30), wet `q·Δ − 2,0`
//!   (Table 31) with Δ = 0,1 (fine) / 0,2 (coarse); alpha / beta
//!   through the Part 1 Tables 202-205 machinery already in
//!   [`crate::acpl_synth`] (the `ibeta` coupling included).
//!
//! Exact encoder inverses ([`write_ajcc_data`] etc.) mirror every
//! parser for the roundtrip / GOP tests.
//!
//! Not yet wired (follow-up): the §5.6.3.3 Table 32 interpolator and
//! the §5.6.3.5.2/§5.6.3.5.3 full/core reconstruction matrices that
//! turn these dequantized parameters plus the five core channels into
//! output QMF channels.

use crate::acpl::{get_acpl_hcb, AcplDataType, AcplHcbType, AcplQuantMode};
use crate::acpl_synth::{dequantize_alpha_index, dequantize_beta_index};
use crate::ajoc::AjocDiffType;
use crate::huffman::huff_decode;
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

include!("ajcc_hcb_tables.rs");

/// Table 108: `ajcc_num_bands_table` indexed by
/// `ajcc_num_param_bands_id`.
pub const AJCC_NUM_BANDS_TABLE: [u32; 4] = [15, 12, 9, 7];

/// `data_type` argument of `get_ajcc_hcb()` (§6.3.7.3.2 Table 116).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AjccDataType {
    /// A-CPL-style prediction coefficient (Part 1 codebooks).
    Alpha,
    /// A-CPL-style prediction coefficient (Part 1 codebooks).
    Beta,
    /// Dry gain (Annex A.1.2 codebooks).
    Dry,
    /// Wet (decorrelator) gain (Annex A.1.2 codebooks).
    Wet,
}

/// One selected Huffman codebook handle (same shape as
/// [`crate::acpl::AcplHcb`]): lengths, codewords, `cb_off`.
pub struct AjccHcb {
    /// Codebook name (auditing / errors).
    pub name: &'static str,
    /// Per-symbol code lengths.
    pub len: &'static [u8],
    /// Per-symbol codewords.
    pub cw: &'static [u32],
    /// `huff_decode_diff()` offset.
    pub cb_off: i32,
}

/// `get_ajcc_hcb(data_type, quant_mode, hcb_type)` — §6.3.7.3.2
/// Table 116. ALPHA / BETA delegate to the Part 1 A-CPL codebooks
/// (`ACPL_HCB_<data_type>_…`, TS 103 190-1 Annex A.3); DRY / WET use
/// the Part 2 Annex A.1.2 books.
pub fn get_ajcc_hcb(dt: AjccDataType, qm: AcplQuantMode, ht: AcplHcbType) -> AjccHcb {
    use AcplHcbType::*;
    use AcplQuantMode::*;
    match dt {
        AjccDataType::Alpha | AjccDataType::Beta => {
            let acpl_dt = if dt == AjccDataType::Alpha {
                AcplDataType::Alpha
            } else {
                AcplDataType::Beta
            };
            let h = get_acpl_hcb(acpl_dt, qm, ht);
            AjccHcb {
                name: h.name,
                len: h.len,
                cw: h.cw,
                cb_off: h.cb_off,
            }
        }
        AjccDataType::Dry => match (qm, ht) {
            (Coarse, F0) => AjccHcb {
                name: "AJCC_HCB_DRY_COARSE_F0",
                len: &AJCC_HCB_DRY_COARSE_F0_LEN,
                cw: &AJCC_HCB_DRY_COARSE_F0_CW,
                cb_off: 0,
            },
            (Fine, F0) => AjccHcb {
                name: "AJCC_HCB_DRY_FINE_F0",
                len: &AJCC_HCB_DRY_FINE_F0_LEN,
                cw: &AJCC_HCB_DRY_FINE_F0_CW,
                cb_off: 0,
            },
            (Coarse, Df) => AjccHcb {
                name: "AJCC_HCB_DRY_COARSE_DF",
                len: &AJCC_HCB_DRY_COARSE_DF_LEN,
                cw: &AJCC_HCB_DRY_COARSE_DF_CW,
                cb_off: 11,
            },
            (Fine, Df) => AjccHcb {
                name: "AJCC_HCB_DRY_FINE_DF",
                len: &AJCC_HCB_DRY_FINE_DF_LEN,
                cw: &AJCC_HCB_DRY_FINE_DF_CW,
                cb_off: 22,
            },
            (Coarse, Dt) => AjccHcb {
                name: "AJCC_HCB_DRY_COARSE_DT",
                len: &AJCC_HCB_DRY_COARSE_DT_LEN,
                cw: &AJCC_HCB_DRY_COARSE_DT_CW,
                cb_off: 11,
            },
            (Fine, Dt) => AjccHcb {
                name: "AJCC_HCB_DRY_FINE_DT",
                len: &AJCC_HCB_DRY_FINE_DT_LEN,
                cw: &AJCC_HCB_DRY_FINE_DT_CW,
                cb_off: 22,
            },
        },
        AjccDataType::Wet => match (qm, ht) {
            (Coarse, F0) => AjccHcb {
                name: "AJCC_HCB_WET_COARSE_F0",
                len: &AJCC_HCB_WET_COARSE_F0_LEN,
                cw: &AJCC_HCB_WET_COARSE_F0_CW,
                cb_off: 0,
            },
            (Fine, F0) => AjccHcb {
                name: "AJCC_HCB_WET_FINE_F0",
                len: &AJCC_HCB_WET_FINE_F0_LEN,
                cw: &AJCC_HCB_WET_FINE_F0_CW,
                cb_off: 0,
            },
            (Coarse, Df) => AjccHcb {
                name: "AJCC_HCB_WET_COARSE_DF",
                len: &AJCC_HCB_WET_COARSE_DF_LEN,
                cw: &AJCC_HCB_WET_COARSE_DF_CW,
                cb_off: 20,
            },
            (Fine, Df) => AjccHcb {
                name: "AJCC_HCB_WET_FINE_DF",
                len: &AJCC_HCB_WET_FINE_DF_LEN,
                cw: &AJCC_HCB_WET_FINE_DF_CW,
                cb_off: 40,
            },
            (Coarse, Dt) => AjccHcb {
                name: "AJCC_HCB_WET_COARSE_DT",
                len: &AJCC_HCB_WET_COARSE_DT_LEN,
                cw: &AJCC_HCB_WET_COARSE_DT_CW,
                cb_off: 20,
            },
            (Fine, Dt) => AjccHcb {
                name: "AJCC_HCB_WET_FINE_DT",
                len: &AJCC_HCB_WET_FINE_DT_LEN,
                cw: &AJCC_HCB_WET_FINE_DT_CW,
                cb_off: 40,
            },
        },
    }
}

/// Decode one `ajcc_huff_data(data_type, data_bands, quant_mode,
/// b_no_dt)` element (§6.2.6.4). Same shape as the A-JOC layer: the
/// frequency path decodes band 0 with `huff_decode` (raw index) and
/// the rest with `huff_decode_diff` (index − cb_off); the time path is
/// all `huff_decode_diff` over the DT book.
pub fn ajcc_huff_data(
    br: &mut BitReader<'_>,
    data_type: AjccDataType,
    data_bands: usize,
    quant_mode: AcplQuantMode,
    b_no_dt: bool,
) -> Result<(AjocDiffType, Vec<i32>)> {
    let diff_type = if b_no_dt {
        AjocDiffType::Freq
    } else {
        AjocDiffType::from_bit(br.read_bit()?)
    };
    let mut a_huff_data = Vec::with_capacity(data_bands);
    match diff_type {
        AjocDiffType::Freq => {
            if data_bands > 0 {
                let hcb = get_ajcc_hcb(data_type, quant_mode, AcplHcbType::F0);
                a_huff_data.push(huff_decode(br, hcb.len, hcb.cw)? as i32);
                let hcb = get_ajcc_hcb(data_type, quant_mode, AcplHcbType::Df);
                for _ in 1..data_bands {
                    a_huff_data.push(huff_decode(br, hcb.len, hcb.cw)? as i32 - hcb.cb_off);
                }
            }
        }
        AjocDiffType::Time => {
            let hcb = get_ajcc_hcb(data_type, quant_mode, AcplHcbType::Dt);
            for _ in 0..data_bands {
                a_huff_data.push(huff_decode(br, hcb.len, hcb.cw)? as i32 - hcb.cb_off);
            }
        }
    }
    Ok((diff_type, a_huff_data))
}

/// Encode one `ajcc_huff_data()` element — exact inverse of
/// [`ajcc_huff_data`].
pub fn write_ajcc_huff_data(
    bw: &mut BitWriter,
    data_type: AjccDataType,
    quant_mode: AcplQuantMode,
    b_no_dt: bool,
    diff_type: AjocDiffType,
    a_huff_data: &[i32],
) -> Result<()> {
    let put = |bw: &mut BitWriter, hcb: &AjccHcb, idx: i32| -> Result<()> {
        if idx < 0 || idx as usize >= hcb.len.len() {
            return Err(Error::invalid("ac4: A-JCC Huffman symbol out of range"));
        }
        bw.write_u32(hcb.cw[idx as usize], hcb.len[idx as usize] as u32);
        Ok(())
    };
    if b_no_dt {
        if diff_type == AjocDiffType::Time {
            return Err(Error::invalid(
                "ac4: A-JCC time-differential coding forbidden when b_no_dt",
            ));
        }
    } else {
        bw.write_bit(diff_type == AjocDiffType::Time);
    }
    match diff_type {
        AjocDiffType::Freq => {
            if let Some((&f0, rest)) = a_huff_data.split_first() {
                let hcb = get_ajcc_hcb(data_type, quant_mode, AcplHcbType::F0);
                put(bw, &hcb, f0)?;
                let hcb = get_ajcc_hcb(data_type, quant_mode, AcplHcbType::Df);
                for &d in rest {
                    put(bw, &hcb, d + hcb.cb_off)?;
                }
            }
        }
        AjocDiffType::Time => {
            let hcb = get_ajcc_hcb(data_type, quant_mode, AcplHcbType::Dt);
            for &d in a_huff_data {
                put(bw, &hcb, d + hcb.cb_off)?;
            }
        }
    }
    Ok(())
}

/// Parsed `ajcc_framing_data()` (§6.2.6.2).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AjccFramingData {
    /// `ajcc_interpolation_type` — false = smooth, true = steep
    /// (Table 114).
    pub steep: bool,
    /// `ajcc_num_param_sets` = `ajcc_num_param_sets_code + 1`
    /// (Table 115; 1 or 2).
    pub num_param_sets: u32,
    /// `ajcc_param_timeslot[ps]` — steep switch slots (5 bits each,
    /// only transmitted for steep interpolation).
    pub param_timeslot: Vec<u32>,
}

/// Walk `ajcc_framing_data()` (§6.2.6.2).
pub fn parse_ajcc_framing_data(br: &mut BitReader<'_>) -> Result<AjccFramingData> {
    let steep = br.read_bit()?;
    let num_param_sets = br.read_u32(1)? + 1;
    let mut param_timeslot = Vec::new();
    if steep {
        for _ in 0..num_param_sets {
            param_timeslot.push(br.read_u32(5)?);
        }
    }
    Ok(AjccFramingData {
        steep,
        num_param_sets,
        param_timeslot,
    })
}

/// Write `ajcc_framing_data()` — exact inverse of
/// [`parse_ajcc_framing_data`].
pub fn write_ajcc_framing_data(bw: &mut BitWriter, fd: &AjccFramingData) -> Result<()> {
    if fd.num_param_sets < 1 || fd.num_param_sets > 2 {
        return Err(Error::invalid("ac4: ajcc_num_param_sets out of range"));
    }
    bw.write_bit(fd.steep);
    bw.write_u32(fd.num_param_sets - 1, 1);
    if fd.steep {
        if fd.param_timeslot.len() != fd.num_param_sets as usize {
            return Err(Error::invalid("ac4: ajcc_param_timeslot count mismatch"));
        }
        for &ts in &fd.param_timeslot {
            if ts > 31 {
                return Err(Error::invalid("ac4: ajcc_param_timeslot out of range"));
            }
            bw.write_u32(ts, 5);
        }
    }
    Ok(())
}

/// One `ajced()` result (§6.2.6.3): `num_ps` Huffman-decoded parameter
/// sets, each `(diff_type, a_huff_data[0..num_bands])`.
pub type Ajced = Vec<(AjocDiffType, Vec<i32>)>;

/// Walk `ajced(data_type, data_bands, quant_mode, b_no_dt, num_ps)`
/// (§6.2.6.3).
pub fn parse_ajced(
    br: &mut BitReader<'_>,
    data_type: AjccDataType,
    data_bands: usize,
    quant_mode: AcplQuantMode,
    b_no_dt: bool,
    num_ps: u32,
) -> Result<Ajced> {
    (0..num_ps)
        .map(|_| ajcc_huff_data(br, data_type, data_bands, quant_mode, b_no_dt))
        .collect()
}

/// Write `ajced()` — exact inverse of [`parse_ajced`].
pub fn write_ajced(
    bw: &mut BitWriter,
    data_type: AjccDataType,
    quant_mode: AcplQuantMode,
    b_no_dt: bool,
    sets: &Ajced,
) -> Result<()> {
    for (dt, vals) in sets {
        write_ajcc_huff_data(bw, data_type, quant_mode, b_no_dt, *dt, vals)?;
    }
    Ok(())
}

/// Parsed `ajcc_data(b_5fronts)` element (§6.2.6.1).
///
/// The parameter SETs are stored in the spec's transmission order:
///
/// * `b_5fronts`: `dry` = `[dry1f, dry2f, dry3f, dry4f, dry1b, dry2b,
///   dry3b, dry4b]`, `wet` = `[wet1f..wet6f, wet1b..wet6b]`,
///   `framing` = `[nps_lf, nps_rf, nps_lb, nps_rb]`, no alpha / beta.
/// * otherwise: `dry` = `[dry1..dry4]`, `wet` = `[wet1..wet6]`,
///   `alpha` = `[alpha1, alpha2]`, `beta` = `[beta1, beta2]`,
///   `framing` = `[nps_l, nps_r]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AjccData {
    /// Layout selector (caller-supplied, from the channel config).
    pub b_5fronts: bool,
    /// `b_no_dt` (Table 107).
    pub b_no_dt: bool,
    /// `ajcc_num_param_bands_id` (Table 108 index).
    pub num_param_bands_id: u8,
    /// Resolved `num_bands`.
    pub num_bands: u32,
    /// `ajcc_core_mode` (Table 109; only when `!b_5fronts`).
    pub core_mode: bool,
    /// `ajcc_qm_f` / `ajcc_qm_b` (only when `b_5fronts`).
    pub qm_f: AcplQuantMode,
    /// See [`AjccData::qm_f`].
    pub qm_b: AcplQuantMode,
    /// `ajcc_qm_ab` / `ajcc_qm_dw` (only when `!b_5fronts`).
    pub qm_ab: AcplQuantMode,
    /// See [`AjccData::qm_ab`].
    pub qm_dw: AcplQuantMode,
    /// Framing data in transmission order (see struct docs).
    pub framing: Vec<AjccFramingData>,
    /// Alpha SETs (empty for `b_5fronts`).
    pub alpha: Vec<Ajced>,
    /// Beta SETs (empty for `b_5fronts`).
    pub beta: Vec<Ajced>,
    /// Dry SETs.
    pub dry: Vec<Ajced>,
    /// Wet SETs.
    pub wet: Vec<Ajced>,
}

/// The per-SET plan of one layout: which framing entry and which quant
/// mode every dry / wet / alpha / beta SET uses (from the §6.2.6.1
/// listing).
struct AjccPlan {
    /// Framing index per dry SET.
    dry: &'static [usize],
    /// Framing index per wet SET.
    wet: &'static [usize],
    /// Framing index per alpha / beta SET (empty for 5-fronts).
    ab: &'static [usize],
}

fn ajcc_plan(b_5fronts: bool) -> AjccPlan {
    if b_5fronts {
        AjccPlan {
            // dry1f dry2f dry3f dry4f dry1b dry2b dry3b dry4b
            dry: &[0, 0, 1, 1, 2, 2, 3, 3],
            // wet1f..3f (lf) wet4f..6f (rf) wet1b..3b (lb) wet4b..6b (rb)
            wet: &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            ab: &[],
        }
    } else {
        AjccPlan {
            // dry1 dry2 (l) dry3 dry4 (r)
            dry: &[0, 0, 1, 1],
            // wet1..3 (l) wet4..6 (r)
            wet: &[0, 0, 0, 1, 1, 1],
            // alpha1 (l) alpha2 (r); beta likewise
            ab: &[0, 1],
        }
    }
}

/// Quant mode of a dry / wet SET from its framing index.
fn dry_wet_qm(data: &AjccData, framing_idx: usize) -> AcplQuantMode {
    if data.b_5fronts {
        if framing_idx < 2 {
            data.qm_f
        } else {
            data.qm_b
        }
    } else {
        data.qm_dw
    }
}

/// Walk `ajcc_data(b_5fronts)` (§6.2.6.1).
pub fn parse_ajcc_data(br: &mut BitReader<'_>, b_5fronts: bool) -> Result<AjccData> {
    let b_no_dt = br.read_bit()?;
    let num_param_bands_id = br.read_u32(2)? as u8;
    let num_bands = AJCC_NUM_BANDS_TABLE[num_param_bands_id as usize];
    let nb = num_bands as usize;

    let mut data = AjccData {
        b_5fronts,
        b_no_dt,
        num_param_bands_id,
        num_bands,
        ..Default::default()
    };

    if b_5fronts {
        data.qm_f = AcplQuantMode::from_bit(br.read_bit()?);
        data.qm_b = AcplQuantMode::from_bit(br.read_bit()?);
    } else {
        data.core_mode = br.read_bit()?;
        data.qm_ab = AcplQuantMode::from_bit(br.read_bit()?);
        data.qm_dw = AcplQuantMode::from_bit(br.read_bit()?);
    }

    let num_framing = if b_5fronts { 4 } else { 2 };
    for _ in 0..num_framing {
        data.framing.push(parse_ajcc_framing_data(br)?);
    }
    let nps = |idx: usize| data.framing[idx].num_param_sets;

    let plan = ajcc_plan(b_5fronts);
    for &fi in plan.ab {
        data.alpha.push(parse_ajced(
            br,
            AjccDataType::Alpha,
            nb,
            data.qm_ab,
            b_no_dt,
            nps(fi),
        )?);
    }
    for &fi in plan.ab {
        data.beta.push(parse_ajced(
            br,
            AjccDataType::Beta,
            nb,
            data.qm_ab,
            b_no_dt,
            nps(fi),
        )?);
    }
    for &fi in plan.dry {
        let qm = dry_wet_qm(&data, fi);
        data.dry.push(parse_ajced(
            br,
            AjccDataType::Dry,
            nb,
            qm,
            b_no_dt,
            nps(fi),
        )?);
    }
    for &fi in plan.wet {
        let qm = dry_wet_qm(&data, fi);
        data.wet.push(parse_ajced(
            br,
            AjccDataType::Wet,
            nb,
            qm,
            b_no_dt,
            nps(fi),
        )?);
    }
    Ok(data)
}

/// Write `ajcc_data()` — exact inverse of [`parse_ajcc_data`].
pub fn write_ajcc_data(bw: &mut BitWriter, data: &AjccData) -> Result<()> {
    bw.write_bit(data.b_no_dt);
    if data.num_param_bands_id > 3 {
        return Err(Error::invalid("ac4: ajcc_num_param_bands_id out of range"));
    }
    bw.write_u32(data.num_param_bands_id as u32, 2);

    if data.b_5fronts {
        bw.write_bit(data.qm_f == AcplQuantMode::Coarse);
        bw.write_bit(data.qm_b == AcplQuantMode::Coarse);
    } else {
        bw.write_bit(data.core_mode);
        bw.write_bit(data.qm_ab == AcplQuantMode::Coarse);
        bw.write_bit(data.qm_dw == AcplQuantMode::Coarse);
    }

    let num_framing = if data.b_5fronts { 4 } else { 2 };
    if data.framing.len() != num_framing {
        return Err(Error::invalid("ac4: ajcc framing count mismatch"));
    }
    for fd in &data.framing {
        write_ajcc_framing_data(bw, fd)?;
    }

    let plan = ajcc_plan(data.b_5fronts);
    if data.dry.len() != plan.dry.len()
        || data.wet.len() != plan.wet.len()
        || data.alpha.len() != plan.ab.len()
        || data.beta.len() != plan.ab.len()
    {
        return Err(Error::invalid("ac4: ajcc SET roster mismatch"));
    }
    let check_ps = |sets: &Ajced, fi: usize| -> Result<()> {
        if sets.len() != data.framing[fi].num_param_sets as usize {
            return Err(Error::invalid("ac4: ajcc parameter-set count mismatch"));
        }
        Ok(())
    };
    for (i, &fi) in plan.ab.iter().enumerate() {
        check_ps(&data.alpha[i], fi)?;
        write_ajced(
            bw,
            AjccDataType::Alpha,
            data.qm_ab,
            data.b_no_dt,
            &data.alpha[i],
        )?;
    }
    for (i, &fi) in plan.ab.iter().enumerate() {
        check_ps(&data.beta[i], fi)?;
        write_ajced(
            bw,
            AjccDataType::Beta,
            data.qm_ab,
            data.b_no_dt,
            &data.beta[i],
        )?;
    }
    for (i, &fi) in plan.dry.iter().enumerate() {
        check_ps(&data.dry[i], fi)?;
        write_ajced(
            bw,
            AjccDataType::Dry,
            dry_wet_qm(data, fi),
            data.b_no_dt,
            &data.dry[i],
        )?;
    }
    for (i, &fi) in plan.wet.iter().enumerate() {
        check_ps(&data.wet[i], fi)?;
        write_ajced(
            bw,
            AjccDataType::Wet,
            dry_wet_qm(data, fi),
            data.b_no_dt,
            &data.wet[i],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// §5.6.3.2 — differential decoding + dequantization
// ---------------------------------------------------------------------

/// Table 29: recover the absolute quantised values for one SET from
/// its Huffman-decoded parameter sets. The frequency path is a plain
/// running sum (no modulo); the time path adds to `prev`, which is the
/// last parameter set of the previous frame (or of this SET's previous
/// parameter set). `prev` is updated to each parameter set in turn.
pub fn ajcc_differential_decode(sets: &Ajced, num_bands: usize, prev: &mut [i32]) -> Vec<Vec<i32>> {
    let mut out = Vec::with_capacity(sets.len());
    for (diff_type, deltas) in sets {
        let mut q = vec![0i32; num_bands];
        match diff_type {
            AjocDiffType::Freq => {
                if num_bands > 0 {
                    q[0] = deltas.first().copied().unwrap_or(0);
                    for i in 1..num_bands {
                        q[i] = q[i - 1] + deltas.get(i).copied().unwrap_or(0);
                    }
                }
            }
            AjocDiffType::Time => {
                for i in 0..num_bands {
                    q[i] = prev[i] + deltas.get(i).copied().unwrap_or(0);
                }
            }
        }
        prev[..num_bands].copy_from_slice(&q);
        out.push(q);
    }
    out
}

/// Table 30: dequantize one A-JCC dry index — `q · Δ − 0,6` with
/// Δ = 0,1 (fine) / 0,2 (coarse).
pub fn dequantize_ajcc_dry(q: i32, qm: AcplQuantMode) -> f64 {
    let delta = match qm {
        AcplQuantMode::Fine => 0.1,
        AcplQuantMode::Coarse => 0.2,
    };
    q as f64 * delta - 0.6
}

/// Table 31: dequantize one A-JCC wet index — `q · Δ − 2,0`.
pub fn dequantize_ajcc_wet(q: i32, qm: AcplQuantMode) -> f64 {
    let delta = match qm {
        AcplQuantMode::Fine => 0.1,
        AcplQuantMode::Coarse => 0.2,
    };
    q as f64 * delta - 2.0
}

/// Dequantize one A-JCC alpha / beta pair (§5.6.3.2, Part 1
/// Tables 202-205 with the `ibeta` coupling).
///
/// A-JCC's `ajcc_huff_data()` produces the alpha F0 as a *raw* index
/// (plain `huff_decode`), while the Part 1 lookup in
/// [`crate::acpl_synth`] is addressed in the signed lane
/// (`index − cb_off`); the bridge subtracts the alpha F0 book's
/// `cb_off` (8 coarse / 16 fine). Beta F0 books have `cb_off = 0` so
/// the raw index is already the signed magnitude lane.
pub fn dequantize_ajcc_alpha_beta(alpha_q: i32, beta_q: i32, qm: AcplQuantMode) -> (f32, f32) {
    let alpha_off = get_ajcc_hcb(AjccDataType::Alpha, qm, AcplHcbType::F0)
        .len
        .len() as i32
        / 2;
    let (alpha_dq, ibeta) = dequantize_alpha_index(qm, alpha_q - alpha_off);
    let beta_dq = dequantize_beta_index(qm, beta_q, ibeta);
    (alpha_dq, beta_dq)
}

/// Valid quantised range for one SET type: `0 ..= max` for the
/// unsigned families (dry / wet / alpha raw lanes) and `-max ..= max`
/// for beta (signed magnitude).
fn ajcc_q_range(dt: AjccDataType, qm: AcplQuantMode) -> (i32, i32) {
    let f0_len = get_ajcc_hcb(dt, qm, AcplHcbType::F0).len.len() as i32;
    match dt {
        AjccDataType::Beta => (-(f0_len - 1), f0_len - 1),
        _ => (0, f0_len - 1),
    }
}

/// Persistent A-JCC `ajcc_<SET>_q_prev` state (§5.6.3.2), one row per
/// SET in the [`AjccData`] roster order, sized to the maximum band
/// count (15).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AjccState {
    /// Per-alpha-SET previous quantised row.
    pub alpha_prev: Vec<Vec<i32>>,
    /// Per-beta-SET previous quantised row.
    pub beta_prev: Vec<Vec<i32>>,
    /// Per-dry-SET previous quantised row.
    pub dry_prev: Vec<Vec<i32>>,
    /// Per-wet-SET previous quantised row.
    pub wet_prev: Vec<Vec<i32>>,
}

impl AjccState {
    /// Fresh all-zero state for the given layout (§5.6.3.2: "when
    /// decoding the first AC-4 frame, all elements … shall be 0").
    pub fn new(b_5fronts: bool) -> Self {
        let plan = ajcc_plan(b_5fronts);
        let max_nb = AJCC_NUM_BANDS_TABLE[0] as usize;
        AjccState {
            alpha_prev: vec![vec![0; max_nb]; plan.ab.len()],
            beta_prev: vec![vec![0; max_nb]; plan.ab.len()],
            dry_prev: vec![vec![0; max_nb]; plan.dry.len()],
            wet_prev: vec![vec![0; max_nb]; plan.wet.len()],
        }
    }
}

/// Fully decoded A-JCC parameters for one frame: absolute quantised
/// values (`*_q[set][ps][pb]`) and dequantized values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AjccDecoded {
    /// The parsed bitstream element.
    pub data: AjccData,
    /// Quantised alpha per SET / parameter set / band.
    pub alpha_q: Vec<Vec<Vec<i32>>>,
    /// Quantised beta.
    pub beta_q: Vec<Vec<Vec<i32>>>,
    /// Quantised dry.
    pub dry_q: Vec<Vec<Vec<i32>>>,
    /// Quantised wet.
    pub wet_q: Vec<Vec<Vec<i32>>>,
    /// Dequantized alpha (Tables 202/204).
    pub alpha_dq: Vec<Vec<Vec<f32>>>,
    /// Dequantized beta (Tables 203/205, `ibeta`-coupled to alpha).
    pub beta_dq: Vec<Vec<Vec<f32>>>,
    /// Dequantized dry (Table 30).
    pub dry_dq: Vec<Vec<Vec<f64>>>,
    /// Dequantized wet (Table 31).
    pub wet_dq: Vec<Vec<Vec<f64>>>,
}

fn decode_family(
    sets: &[Ajced],
    dt: AjccDataType,
    qm: AcplQuantMode,
    num_bands: usize,
    prev: &mut [Vec<i32>],
) -> Result<Vec<Vec<Vec<i32>>>> {
    let (lo, hi) = ajcc_q_range(dt, qm);
    let mut out = Vec::with_capacity(sets.len());
    for (i, s) in sets.iter().enumerate() {
        let q = ajcc_differential_decode(s, num_bands, &mut prev[i]);
        for row in &q {
            if row.iter().any(|&v| v < lo || v > hi) {
                return Err(Error::invalid("ac4: A-JCC quantised value out of range"));
            }
        }
        out.push(q);
    }
    Ok(out)
}

/// Decode a complete `ajcc_data()` element and run the §5.6.3.2
/// differential decode + dequantization. `state` carries the per-SET
/// `ajcc_<SET>_q_prev` rows across frames.
pub fn decode_ajcc(
    br: &mut BitReader<'_>,
    b_5fronts: bool,
    state: &mut AjccState,
) -> Result<AjccDecoded> {
    let data = parse_ajcc_data(br, b_5fronts)?;
    decode_ajcc_parsed(data, state)
}

/// Run the §5.6.3.2 differential decode + dequantization over an
/// already-parsed `ajcc_data()` element (e.g. one stored on a parsed
/// `immersive_channel_element`). `state` carries the per-SET
/// `ajcc_<SET>_q_prev` rows across frames.
pub fn decode_ajcc_parsed(data: AjccData, state: &mut AjccState) -> Result<AjccDecoded> {
    let b_5fronts = data.b_5fronts;
    let nb = data.num_bands as usize;

    let alpha_q = decode_family(
        &data.alpha,
        AjccDataType::Alpha,
        data.qm_ab,
        nb,
        &mut state.alpha_prev,
    )?;
    let beta_q = decode_family(
        &data.beta,
        AjccDataType::Beta,
        data.qm_ab,
        nb,
        &mut state.beta_prev,
    )?;

    let plan = ajcc_plan(b_5fronts);
    let mut dry_q = Vec::with_capacity(data.dry.len());
    for (i, s) in data.dry.iter().enumerate() {
        let qm = dry_wet_qm(&data, plan.dry[i]);
        dry_q.extend(decode_family(
            std::slice::from_ref(s),
            AjccDataType::Dry,
            qm,
            nb,
            &mut state.dry_prev[i..=i],
        )?);
    }
    let mut wet_q = Vec::with_capacity(data.wet.len());
    for (i, s) in data.wet.iter().enumerate() {
        let qm = dry_wet_qm(&data, plan.wet[i]);
        wet_q.extend(decode_family(
            std::slice::from_ref(s),
            AjccDataType::Wet,
            qm,
            nb,
            &mut state.wet_prev[i..=i],
        )?);
    }

    // Dequantize. Alpha / beta are ibeta-coupled per (set, ps, band).
    let mut alpha_dq = Vec::with_capacity(alpha_q.len());
    let mut beta_dq = Vec::with_capacity(beta_q.len());
    for (a_set, b_set) in alpha_q.iter().zip(beta_q.iter()) {
        let mut a_out = Vec::with_capacity(a_set.len());
        let mut b_out = Vec::with_capacity(b_set.len());
        for (a_ps, b_ps) in a_set.iter().zip(b_set.iter()) {
            let mut a_row = Vec::with_capacity(a_ps.len());
            let mut b_row = Vec::with_capacity(b_ps.len());
            for (&aq, &bq) in a_ps.iter().zip(b_ps.iter()) {
                let (a, b) = dequantize_ajcc_alpha_beta(aq, bq, data.qm_ab);
                a_row.push(a);
                b_row.push(b);
            }
            a_out.push(a_row);
            b_out.push(b_row);
        }
        alpha_dq.push(a_out);
        beta_dq.push(b_out);
    }
    let dry_dq = dry_q
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let qm = dry_wet_qm(&data, plan.dry[i]);
            s.iter()
                .map(|ps| ps.iter().map(|&q| dequantize_ajcc_dry(q, qm)).collect())
                .collect()
        })
        .collect();
    let wet_dq = wet_q
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let qm = dry_wet_qm(&data, plan.wet[i]);
            s.iter()
                .map(|ps| ps.iter().map(|&q| dequantize_ajcc_wet(q, qm)).collect())
                .collect()
        })
        .collect();

    Ok(AjccDecoded {
        data,
        alpha_q,
        beta_q,
        dry_q,
        wet_q,
        alpha_dq,
        beta_dq,
        dry_dq,
        wet_dq,
    })
}

/// Derive freq-direction `a_huff_data` from an absolute quantised row
/// (plain differences — the Table 29 inverse; no modulo in A-JCC).
pub fn encode_ajcc_deltas_freq(q: &[i32]) -> Vec<i32> {
    q.iter()
        .enumerate()
        .map(|(i, &v)| if i == 0 { v } else { v - q[i - 1] })
        .collect()
}

/// Derive time-direction `a_huff_data`: `q[i] − prev[i]`.
pub fn encode_ajcc_deltas_time(q: &[i32], prev: &[i32]) -> Vec<i32> {
    q.iter().zip(prev.iter()).map(|(&v, &p)| v - p).collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    fn all_ajcc_books() -> Vec<(&'static str, AjccHcb, usize, i32)> {
        use AcplHcbType::*;
        use AcplQuantMode::*;
        use AjccDataType::*;
        // (label, book, expected length, expected cb_off) per Annex A.1.2.
        vec![
            ("A.13", get_ajcc_hcb(Dry, Coarse, F0), 12, 0),
            ("A.14", get_ajcc_hcb(Dry, Fine, F0), 23, 0),
            ("A.15", get_ajcc_hcb(Dry, Coarse, Df), 23, 11),
            ("A.16", get_ajcc_hcb(Dry, Fine, Df), 45, 22),
            ("A.17", get_ajcc_hcb(Dry, Coarse, Dt), 23, 11),
            ("A.18", get_ajcc_hcb(Dry, Fine, Dt), 45, 22),
            ("A.19", get_ajcc_hcb(Wet, Coarse, F0), 21, 0),
            ("A.20", get_ajcc_hcb(Wet, Fine, F0), 41, 0),
            ("A.21", get_ajcc_hcb(Wet, Coarse, Df), 41, 20),
            ("A.22", get_ajcc_hcb(Wet, Fine, Df), 81, 40),
            ("A.23", get_ajcc_hcb(Wet, Coarse, Dt), 41, 20),
            ("A.24", get_ajcc_hcb(Wet, Fine, Dt), 81, 40),
        ]
    }

    #[test]
    fn ajcc_codebooks_match_annex_a12_and_are_prefix_complete() {
        for (label, hcb, n, off) in all_ajcc_books() {
            assert_eq!(hcb.len.len(), n, "{label} length");
            assert_eq!(hcb.cw.len(), n, "{label} cw length");
            assert_eq!(hcb.cb_off, off, "{label} cb_off");
            let kraft: u64 = hcb.len.iter().map(|&l| 1u64 << (32 - l as u64)).sum();
            assert_eq!(kraft, 1u64 << 32, "{label} Kraft sum");
            for i in 0..n {
                for j in (i + 1)..n {
                    let (li, ci) = (hcb.len[i] as u32, hcb.cw[i]);
                    let (lj, cj) = (hcb.len[j] as u32, hcb.cw[j]);
                    let (ls, cs, ll, cl) = if li <= lj {
                        (li, ci, lj, cj)
                    } else {
                        (lj, cj, li, ci)
                    };
                    assert!(ls != ll || cs != cl, "{label} dup {i}/{j}");
                    assert!(cl >> (ll - ls) != cs, "{label} prefix {i}/{j}");
                }
            }
        }
    }

    #[test]
    fn alpha_beta_delegate_to_part1_acpl_books() {
        use AcplHcbType::*;
        use AcplQuantMode::*;
        let a = get_ajcc_hcb(AjccDataType::Alpha, Coarse, F0);
        let p1 = get_acpl_hcb(AcplDataType::Alpha, Coarse, F0);
        assert_eq!(a.name, p1.name);
        assert_eq!(a.cb_off, p1.cb_off);
        assert_eq!(a.len.len(), 17);
        let b = get_ajcc_hcb(AjccDataType::Beta, Fine, Dt);
        let p1 = get_acpl_hcb(AcplDataType::Beta, Fine, Dt);
        assert_eq!(b.name, p1.name);
        assert_eq!(b.len.len(), p1.len.len());
    }

    #[test]
    fn ajcc_huff_data_roundtrips_all_types() {
        // (data_type, qm, freq row within grid, time deltas)
        let cases = [
            (AjccDataType::Dry, AcplQuantMode::Coarse, vec![5, 11, 0, 3]),
            (AjccDataType::Wet, AcplQuantMode::Fine, vec![20, 0, 40, 33]),
            (
                AjccDataType::Alpha,
                AcplQuantMode::Coarse,
                vec![8, 16, 0, 9],
            ),
            (AjccDataType::Beta, AcplQuantMode::Fine, vec![0, 3, 8, 1]),
        ];
        for (dt, qm, q) in cases {
            let freq = encode_ajcc_deltas_freq(&q);
            let mut bw = BitWriter::new();
            write_ajcc_huff_data(&mut bw, dt, qm, false, AjocDiffType::Freq, &freq).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let (got_dt, got) = ajcc_huff_data(&mut br, dt, 4, qm, false).unwrap();
            assert_eq!(got_dt, AjocDiffType::Freq);
            assert_eq!(got, freq, "{dt:?}");

            // Time direction with signed deltas.
            let deltas = vec![-2, 0, 1, -1];
            let mut bw = BitWriter::new();
            write_ajcc_huff_data(&mut bw, dt, qm, false, AjocDiffType::Time, &deltas).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let (got_dt, got) = ajcc_huff_data(&mut br, dt, 4, qm, false).unwrap();
            assert_eq!(got_dt, AjocDiffType::Time);
            assert_eq!(got, deltas, "{dt:?}");
        }
    }

    #[test]
    fn framing_data_roundtrip() {
        for fd in [
            AjccFramingData {
                steep: false,
                num_param_sets: 1,
                param_timeslot: vec![],
            },
            AjccFramingData {
                steep: false,
                num_param_sets: 2,
                param_timeslot: vec![],
            },
            AjccFramingData {
                steep: true,
                num_param_sets: 2,
                param_timeslot: vec![7, 23],
            },
        ] {
            let mut bw = BitWriter::new();
            write_ajcc_framing_data(&mut bw, &fd).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(parse_ajcc_framing_data(&mut br).unwrap(), fd);
        }
    }

    /// Build a valid AjccData from per-family constant quantised rows.
    fn build_data(b_5fronts: bool, nb_id: u8, frame: i32, prev: Option<&AjccState>) -> AjccData {
        let nb = AJCC_NUM_BANDS_TABLE[nb_id as usize] as usize;
        let plan = ajcc_plan(b_5fronts);
        let framing = vec![
            AjccFramingData {
                steep: false,
                num_param_sets: 1,
                param_timeslot: vec![],
            };
            if b_5fronts { 4 } else { 2 }
        ];
        let mut data = AjccData {
            b_5fronts,
            b_no_dt: prev.is_none(),
            num_param_bands_id: nb_id,
            num_bands: nb as u32,
            core_mode: false,
            // Only the layout's own quant fields may differ from the
            // Default (untransmitted fields stay at their parse-side
            // defaults so the roundtrip equality holds).
            qm_f: if b_5fronts {
                AcplQuantMode::Coarse
            } else {
                AcplQuantMode::default()
            },
            qm_b: AcplQuantMode::Fine,
            qm_ab: if b_5fronts {
                AcplQuantMode::default()
            } else {
                AcplQuantMode::Coarse
            },
            qm_dw: AcplQuantMode::Fine,
            framing,
            ..Default::default()
        };
        // Deterministic quantised rows per SET.
        let row = |base: i32, span: i32, k: usize| -> Vec<i32> {
            (0..nb)
                .map(|pb| base + ((pb as i32 + k as i32 + frame) % (span.max(1))))
                .collect()
        };
        let mk = |q: &[i32], prev_row: Option<&[i32]>| -> Ajced {
            match prev_row {
                Some(p) => vec![(
                    AjocDiffType::Time,
                    encode_ajcc_deltas_time(q, &p[..q.len()]),
                )],
                None => vec![(AjocDiffType::Freq, encode_ajcc_deltas_freq(q))],
            }
        };
        for (i, _) in plan.ab.iter().enumerate() {
            let q = row(6, 5, i);
            data.alpha
                .push(mk(&q, prev.map(|s| s.alpha_prev[i].as_slice())));
            let qb = row(0, 4, i);
            data.beta
                .push(mk(&qb, prev.map(|s| s.beta_prev[i].as_slice())));
        }
        for (i, &fi) in plan.dry.iter().enumerate() {
            let qm = dry_wet_qm(&data, fi);
            let span = if qm == AcplQuantMode::Coarse { 12 } else { 23 };
            let q = row(0, span, i);
            data.dry
                .push(mk(&q, prev.map(|s| s.dry_prev[i].as_slice())));
        }
        for (i, &fi) in plan.wet.iter().enumerate() {
            let qm = dry_wet_qm(&data, fi);
            let span = if qm == AcplQuantMode::Coarse { 21 } else { 41 };
            let q = row(0, span, i);
            data.wet
                .push(mk(&q, prev.map(|s| s.wet_prev[i].as_slice())));
        }
        data
    }

    #[test]
    fn ajcc_data_bitstream_roundtrip_both_layouts() {
        for b_5fronts in [false, true] {
            let data = build_data(b_5fronts, 2, 0, None); // 9 bands
            let mut bw = BitWriter::new();
            write_ajcc_data(&mut bw, &data).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let got = parse_ajcc_data(&mut br, b_5fronts).unwrap();
            assert_eq!(got, data, "b_5fronts={b_5fronts}");
        }
    }

    #[test]
    fn table_29_freq_is_plain_sum_no_modulo() {
        let sets: Ajced = vec![(AjocDiffType::Freq, vec![10, 5, -8, 2])];
        let mut prev = vec![0i32; 4];
        let q = ajcc_differential_decode(&sets, 4, &mut prev);
        assert_eq!(q, vec![vec![10, 15, 7, 9]]);
        assert_eq!(prev, vec![10, 15, 7, 9]);
        // Two parameter sets: the second chains off the first via prev.
        let sets: Ajced = vec![
            (AjocDiffType::Freq, vec![3, 1, 1, 1]),
            (AjocDiffType::Time, vec![1, -1, 0, 2]),
        ];
        let mut prev = vec![0i32; 4];
        let q = ajcc_differential_decode(&sets, 4, &mut prev);
        assert_eq!(q[0], vec![3, 4, 5, 6]);
        assert_eq!(q[1], vec![4, 3, 5, 8]);
        assert_eq!(prev, q[1]);
    }

    #[test]
    fn dry_wet_dequant_tables_30_31() {
        // Fine: Δ = 0,1; Coarse: Δ = 0,2. Dry offset −0,6; wet −2,0.
        assert!((dequantize_ajcc_dry(0, AcplQuantMode::Fine) + 0.6).abs() < 1e-12);
        assert!((dequantize_ajcc_dry(22, AcplQuantMode::Fine) - 1.6).abs() < 1e-12);
        assert!((dequantize_ajcc_dry(11, AcplQuantMode::Coarse) - 1.6).abs() < 1e-12);
        assert!((dequantize_ajcc_wet(0, AcplQuantMode::Coarse) + 2.0).abs() < 1e-12);
        assert!((dequantize_ajcc_wet(20, AcplQuantMode::Coarse) - 2.0).abs() < 1e-12);
        assert!((dequantize_ajcc_wet(40, AcplQuantMode::Fine) - 2.0).abs() < 1e-12);
        assert!(dequantize_ajcc_wet(20, AcplQuantMode::Fine).abs() < 1e-12);
    }

    #[test]
    fn alpha_beta_dequant_bridges_raw_lane() {
        // Raw centre lane (8 coarse / 16 fine) is alpha_q = 0 signed →
        // alpha_dq = 0 per Part 1 Table 203/205 centre.
        let (a, _) = dequantize_ajcc_alpha_beta(8, 0, AcplQuantMode::Coarse);
        assert!(a.abs() < 1e-6, "coarse centre alpha {a}");
        let (a, _) = dequantize_ajcc_alpha_beta(16, 0, AcplQuantMode::Fine);
        assert!(a.abs() < 1e-6, "fine centre alpha {a}");
        // Beta magnitude 0 dequantizes to the ibeta-selected zero row.
        let (_, b) = dequantize_ajcc_alpha_beta(8, 0, AcplQuantMode::Coarse);
        let (_, b2) = dequantize_ajcc_alpha_beta(0, 0, AcplQuantMode::Coarse);
        // Different alpha lanes may select different ibeta columns; both
        // must be finite table entries.
        assert!(b.is_finite() && b2.is_finite());
    }

    #[test]
    fn gop_decode_chain_matches_across_frames() {
        // Frame 0 (I-style, b_no_dt = 1, freq) then frame 1 coded
        // DIFF_TIME against frame 0: the decoded quantised values must
        // equal the intended absolute rows on both layouts.
        for b_5fronts in [false, true] {
            let mut state = AjccState::new(b_5fronts);

            let data0 = build_data(b_5fronts, 1, 0, None); // 12 bands
            let mut bw = BitWriter::new();
            write_ajcc_data(&mut bw, &data0).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let dec0 = decode_ajcc(&mut br, b_5fronts, &mut state).unwrap();

            // The I-frame freq rows reconstruct the intended absolutes:
            // check one dry SET explicitly.
            let nb = data0.num_bands as usize;
            let freq0 = &data0.dry[0][0].1;
            let mut want = vec![0i32; nb];
            want[0] = freq0[0];
            for i in 1..nb {
                want[i] = want[i - 1] + freq0[i];
            }
            assert_eq!(dec0.dry_q[0][0], want);

            // P-frame built against the carried state.
            let data1 = build_data(b_5fronts, 1, 3, Some(&state));
            let mut bw = BitWriter::new();
            write_ajcc_data(&mut bw, &data1).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let dec1 = decode_ajcc(&mut br, b_5fronts, &mut state).unwrap();

            // Every row of frame 1 is time-coded; the decoded absolutes
            // are prev + delta = the frame-1 target rows by construction.
            for (i, s) in data1.dry.iter().enumerate() {
                assert_eq!(s[0].0, AjocDiffType::Time);
                let target: Vec<i32> = dec0.dry_q[i][0]
                    .iter()
                    .zip(s[0].1.iter())
                    .map(|(&p, &d)| p + d)
                    .collect();
                assert_eq!(dec1.dry_q[i][0], target, "dry set {i}");
            }
            // Dequant sanity on the wet family (within ±2 by design).
            for set in &dec1.wet_dq {
                for ps in set {
                    for &v in ps {
                        assert!((-2.0..=2.0).contains(&v), "wet dq {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn out_of_range_running_sum_rejected() {
        // A freq row that pushes the dry index past its grid must fail
        // decode_ajcc's range validation.
        let mut data = build_data(false, 3, 0, None); // 7 bands
                                                      // dry SET 0, fine grid (0..=22): force an escape upward.
        data.dry[0] = vec![(AjocDiffType::Freq, vec![22, 5, 0, 0, 0, 0, 0])];
        let mut bw = BitWriter::new();
        write_ajcc_data(&mut bw, &data).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut state = AjccState::new(false);
        assert!(decode_ajcc(&mut br, false, &mut state).is_err());
    }
}
