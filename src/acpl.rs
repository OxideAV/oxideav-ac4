//! Advanced Coupling (A-CPL) parser plumbing — ETSI TS 103 190-1
//! §4.2.13 / §4.3.11.
//!
//! What this module covers today:
//!
//! * `acpl_config_1ch()` (§4.2.13.1, Table 59) — `acpl_num_param_bands_id`
//!   (Table 143), `acpl_quant_mode` (Table 144), and the optional
//!   PARTIAL-mode `acpl_qmf_band_minus1` 3-bit field.
//! * `acpl_config_2ch()` (§4.2.13.2, Table 60) — same `bands_id` plus
//!   two `quant_mode_{0,1}` bits (§4.3.11.2.2 / §4.3.11.2.3 — _0 drives
//!   ALPHA / BETA / BETA3 codebooks, _1 drives the GAMMA family).
//! * `acpl_framing_data()` (§4.2.13.5, Table 63) — interpolation type
//!   (Table 145) + `num_param_sets_cod` (Table 146) + per-set
//!   `acpl_param_timeslot[5]`.
//! * `acpl_huff_data()` (§4.2.13.7, Table 65) — `diff_type` plus the
//!   `huff_decode_diff` loop over the chosen `(data_type, quant_mode,
//!   F0|DF|DT)` codebook from §A.3 (Tables A.34..A.57). The
//!   `get_acpl_hcb()` lookup is implemented per §4.3.11.6.1
//!   Pseudocode 8.
//! * `acpl_ec_data()` (§4.2.13.6, Table 64) — wraps `acpl_huff_data()`
//!   in the per-parameter-set loop driven by
//!   `acpl_num_param_sets_cod + 1`.
//!
//! What is NOT covered yet:
//!
//! * `acpl_data_1ch()` / `acpl_data_2ch()` (§4.2.13.3 / §4.2.13.4) —
//!   the outer wrapper that assembles a frame's worth of `(alpha,
//!   beta[, beta3, gamma1..6])` parameter sets — those are scaffolded
//!   in [`parse_acpl_data_1ch`] / [`parse_acpl_data_2ch`] but the
//!   recovered values aren't wired into the QMF bandwidth-extension
//!   pipeline yet.
//! * The dequantisation tables that map the per-band Huffman index
//!   onto an actual α / β / γ coefficient.
//! * Steep-interpolation timeslot smoothing.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::acpl_huffman;

// =====================================================================
// §4.3.11.6.1 Pseudocode 8 — get_acpl_hcb()
// =====================================================================

/// `data_type` argument to `get_acpl_hcb()` (§4.3.11.6.1, Pseudocode 8).
/// The four families pick distinct codebook triples per §A.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcplDataType {
    Alpha,
    Beta,
    Beta3,
    Gamma,
}

/// `quant_mode` argument per Table 144 / §4.3.11.1.3.
///
/// `0 = Fine`, `1 = Coarse` per the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcplQuantMode {
    Fine,
    Coarse,
}

impl AcplQuantMode {
    /// Spec mapping per Table 144: bit value `0` → Fine, `1` → Coarse.
    pub fn from_bit(b: bool) -> Self {
        if b {
            AcplQuantMode::Coarse
        } else {
            AcplQuantMode::Fine
        }
    }
}

/// `hcb_type` argument per Pseudocode 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcplHcbType {
    F0,
    Df,
    Dt,
}

/// `acpl_interpolation_type` per Table 145 / §4.3.11.5.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcplInterpolationType {
    Smooth,
    Steep,
}

/// A single A-CPL Huffman codebook handle: `(name, len[], cw[], cb_off)`.
///
/// Pulled from [`crate::acpl_huffman`]. Decoded values are
/// `symbol_index - cb_off` (per §4.2.13.7 Table 65, every Huffman call
/// in `acpl_huff_data()` goes through `huff_decode_diff` regardless of
/// F0 vs DF vs DT — for the F0 codebooks where `cb_off = 0` this is a
/// no-op; for the GAMMA F0 codebooks `cb_off` is non-zero (10 / 20)
/// and the offset matters).
#[derive(Debug, Clone, Copy)]
pub struct AcplHcb {
    pub name: &'static str,
    pub len: &'static [u8],
    pub cw: &'static [u32],
    pub cb_off: i32,
}

impl AcplHcb {
    /// Decode one Huffman codeword from `br` and return
    /// `symbol_index - cb_off`.
    ///
    /// Uses the same correctness-first "try one entry at a time,
    /// widest-match-wins" walk as the A-SPX / DRC decoders (the
    /// codebooks are small enough that a generic linear scan is
    /// fine; an O(1) lookup table is a future optimisation).
    pub fn decode_delta(&self, br: &mut BitReader<'_>) -> Result<i32> {
        debug_assert_eq!(self.len.len(), self.cw.len());
        let mut code: u32 = 0;
        let mut width: u8 = 0;
        // Widest codeword across all 24 codebooks is 18 bits
        // (ACPL_HCB_GAMMA_FINE_DT). 24-bit cap is defensive.
        while width < 24 {
            let b = br.read_u32(1)?;
            code = (code << 1) | b;
            width += 1;
            for (i, &l) in self.len.iter().enumerate() {
                if l == width && self.cw[i] == code {
                    return Ok(i as i32 - self.cb_off);
                }
            }
        }
        Err(Error::invalid("ac4: no matching A-CPL Huffman codeword"))
    }
}

/// `get_acpl_hcb(data_type, quant_mode, hcb_type)` per
/// §4.3.11.6.1 Pseudocode 8. Returns the matching A-CPL Huffman
/// codebook handle that `acpl_huff_data()` should pull from.
pub fn get_acpl_hcb(dt: AcplDataType, qm: AcplQuantMode, ht: AcplHcbType) -> AcplHcb {
    use acpl_huffman::*;
    use AcplDataType::*;
    use AcplHcbType::*;
    use AcplQuantMode::*;
    match (dt, qm, ht) {
        // ALPHA — Tables A.34..A.39
        (Alpha, Coarse, F0) => AcplHcb {
            name: "ACPL_HCB_ALPHA_COARSE_F0",
            len: ACPL_HCB_ALPHA_COARSE_F0_LEN,
            cw: ACPL_HCB_ALPHA_COARSE_F0_CW,
            cb_off: 0,
        },
        (Alpha, Fine, F0) => AcplHcb {
            name: "ACPL_HCB_ALPHA_FINE_F0",
            len: ACPL_HCB_ALPHA_FINE_F0_LEN,
            cw: ACPL_HCB_ALPHA_FINE_F0_CW,
            cb_off: 0,
        },
        (Alpha, Coarse, Df) => AcplHcb {
            name: "ACPL_HCB_ALPHA_COARSE_DF",
            len: ACPL_HCB_ALPHA_COARSE_DF_LEN,
            cw: ACPL_HCB_ALPHA_COARSE_DF_CW,
            cb_off: 16,
        },
        (Alpha, Fine, Df) => AcplHcb {
            name: "ACPL_HCB_ALPHA_FINE_DF",
            len: ACPL_HCB_ALPHA_FINE_DF_LEN,
            cw: ACPL_HCB_ALPHA_FINE_DF_CW,
            cb_off: 32,
        },
        (Alpha, Coarse, Dt) => AcplHcb {
            name: "ACPL_HCB_ALPHA_COARSE_DT",
            len: ACPL_HCB_ALPHA_COARSE_DT_LEN,
            cw: ACPL_HCB_ALPHA_COARSE_DT_CW,
            cb_off: 16,
        },
        (Alpha, Fine, Dt) => AcplHcb {
            name: "ACPL_HCB_ALPHA_FINE_DT",
            len: ACPL_HCB_ALPHA_FINE_DT_LEN,
            cw: ACPL_HCB_ALPHA_FINE_DT_CW,
            cb_off: 32,
        },
        // BETA — Tables A.40..A.45
        (Beta, Coarse, F0) => AcplHcb {
            name: "ACPL_HCB_BETA_COARSE_F0",
            len: ACPL_HCB_BETA_COARSE_F0_LEN,
            cw: ACPL_HCB_BETA_COARSE_F0_CW,
            cb_off: 0,
        },
        (Beta, Fine, F0) => AcplHcb {
            name: "ACPL_HCB_BETA_FINE_F0",
            len: ACPL_HCB_BETA_FINE_F0_LEN,
            cw: ACPL_HCB_BETA_FINE_F0_CW,
            cb_off: 0,
        },
        (Beta, Coarse, Df) => AcplHcb {
            name: "ACPL_HCB_BETA_COARSE_DF",
            len: ACPL_HCB_BETA_COARSE_DF_LEN,
            cw: ACPL_HCB_BETA_COARSE_DF_CW,
            cb_off: 4,
        },
        (Beta, Fine, Df) => AcplHcb {
            name: "ACPL_HCB_BETA_FINE_DF",
            len: ACPL_HCB_BETA_FINE_DF_LEN,
            cw: ACPL_HCB_BETA_FINE_DF_CW,
            cb_off: 8,
        },
        (Beta, Coarse, Dt) => AcplHcb {
            name: "ACPL_HCB_BETA_COARSE_DT",
            len: ACPL_HCB_BETA_COARSE_DT_LEN,
            cw: ACPL_HCB_BETA_COARSE_DT_CW,
            cb_off: 4,
        },
        (Beta, Fine, Dt) => AcplHcb {
            name: "ACPL_HCB_BETA_FINE_DT",
            len: ACPL_HCB_BETA_FINE_DT_LEN,
            cw: ACPL_HCB_BETA_FINE_DT_CW,
            cb_off: 8,
        },
        // BETA3 — Tables A.46..A.51
        (Beta3, Coarse, F0) => AcplHcb {
            name: "ACPL_HCB_BETA3_COARSE_F0",
            len: ACPL_HCB_BETA3_COARSE_F0_LEN,
            cw: ACPL_HCB_BETA3_COARSE_F0_CW,
            cb_off: 0,
        },
        (Beta3, Fine, F0) => AcplHcb {
            name: "ACPL_HCB_BETA3_FINE_F0",
            len: ACPL_HCB_BETA3_FINE_F0_LEN,
            cw: ACPL_HCB_BETA3_FINE_F0_CW,
            cb_off: 0,
        },
        (Beta3, Coarse, Df) => AcplHcb {
            name: "ACPL_HCB_BETA3_COARSE_DF",
            len: ACPL_HCB_BETA3_COARSE_DF_LEN,
            cw: ACPL_HCB_BETA3_COARSE_DF_CW,
            cb_off: 8,
        },
        (Beta3, Fine, Df) => AcplHcb {
            name: "ACPL_HCB_BETA3_FINE_DF",
            len: ACPL_HCB_BETA3_FINE_DF_LEN,
            cw: ACPL_HCB_BETA3_FINE_DF_CW,
            cb_off: 16,
        },
        (Beta3, Coarse, Dt) => AcplHcb {
            name: "ACPL_HCB_BETA3_COARSE_DT",
            len: ACPL_HCB_BETA3_COARSE_DT_LEN,
            cw: ACPL_HCB_BETA3_COARSE_DT_CW,
            cb_off: 8,
        },
        (Beta3, Fine, Dt) => AcplHcb {
            name: "ACPL_HCB_BETA3_FINE_DT",
            len: ACPL_HCB_BETA3_FINE_DT_LEN,
            cw: ACPL_HCB_BETA3_FINE_DT_CW,
            cb_off: 16,
        },
        // GAMMA — Tables A.52..A.57
        (Gamma, Coarse, F0) => AcplHcb {
            name: "ACPL_HCB_GAMMA_COARSE_F0",
            len: ACPL_HCB_GAMMA_COARSE_F0_LEN,
            cw: ACPL_HCB_GAMMA_COARSE_F0_CW,
            cb_off: 10,
        },
        (Gamma, Fine, F0) => AcplHcb {
            name: "ACPL_HCB_GAMMA_FINE_F0",
            len: ACPL_HCB_GAMMA_FINE_F0_LEN,
            cw: ACPL_HCB_GAMMA_FINE_F0_CW,
            cb_off: 20,
        },
        (Gamma, Coarse, Df) => AcplHcb {
            name: "ACPL_HCB_GAMMA_COARSE_DF",
            len: ACPL_HCB_GAMMA_COARSE_DF_LEN,
            cw: ACPL_HCB_GAMMA_COARSE_DF_CW,
            cb_off: 20,
        },
        (Gamma, Fine, Df) => AcplHcb {
            name: "ACPL_HCB_GAMMA_FINE_DF",
            len: ACPL_HCB_GAMMA_FINE_DF_LEN,
            cw: ACPL_HCB_GAMMA_FINE_DF_CW,
            cb_off: 40,
        },
        (Gamma, Coarse, Dt) => AcplHcb {
            name: "ACPL_HCB_GAMMA_COARSE_DT",
            len: ACPL_HCB_GAMMA_COARSE_DT_LEN,
            cw: ACPL_HCB_GAMMA_COARSE_DT_CW,
            cb_off: 20,
        },
        (Gamma, Fine, Dt) => AcplHcb {
            name: "ACPL_HCB_GAMMA_FINE_DT",
            len: ACPL_HCB_GAMMA_FINE_DT_LEN,
            cw: ACPL_HCB_GAMMA_FINE_DT_CW,
            cb_off: 40,
        },
    }
}

// =====================================================================
// §4.3.11.1.2 Table 143 — acpl_num_param_bands_id mapping
// =====================================================================

/// Map the 2-bit `acpl_num_param_bands_id` to the actual band count
/// (§4.3.11.1.2, Table 143).
pub fn num_param_bands_from_id(id: u32) -> u32 {
    match id & 0b11 {
        0 => 15,
        1 => 12,
        2 => 9,
        _ => 7, // id == 3
    }
}

// =====================================================================
// §5.7.7.2 Table 197 — acpl_param_band ↔ QMF subband mapping
// =====================================================================

/// `sb_to_pb(sb, num_param_bands)` per §5.7.7.2 Table 197.
///
/// Maps a QMF subband `sb` (0..=63) to the parameter-band index for one
/// of the four `acpl_num_param_bands` configurations (15, 12, 9, 7). The
/// table is given in row-form per the spec — we encode the row break
/// points and resolve the right column by `num_param_bands`.
///
/// Returns the parameter-band index (always < `num_param_bands`).
pub fn sb_to_pb(sb: u32, num_param_bands: u32) -> u32 {
    // Row layout from Table 197:
    //  rows 0..=8 are 1:1 with sb (each row covers one sb).
    //  Then merged rows: 9-10, 11-13, 14-17, 18-22, 23-34, 35-63.
    //
    // Per-column param-band assignments (sb_groups[i] is the row index):
    //   row →  (15, 12, 9, 7)
    //    0 → ( 0,  0, 0, 0)
    //    1 → ( 1,  1, 1, 1)
    //    2 → ( 2,  2, 2, 2)
    //    3 → ( 3,  3, 3, 2)
    //    4 → ( 4,  4, 3, 3)
    //    5 → ( 5,  4, 4, 3)
    //    6 → ( 6,  5, 4, 3)
    //    7 → ( 7,  5, 5, 3)
    //    8 → ( 8,  6, 5, 4)
    //  9-10 → ( 9,  6, 6, 4)
    // 11-13 → (10,  7, 6, 4)
    // 14-17 → (11,  8, 7, 5)
    // 18-22 → (12,  9, 7, 5)
    // 23-34 → (13, 10, 8, 6)
    // 35-63 → (14, 11, 8, 6)

    // Determine row index from sb.
    let row: u32 = match sb {
        0..=8 => sb,
        9..=10 => 9,
        11..=13 => 10,
        14..=17 => 11,
        18..=22 => 12,
        23..=34 => 13,
        _ => 14, // 35..=63 (and clamp anything beyond)
    };

    // Resolve column by num_param_bands.
    let table_row: [u32; 4] = match row {
        0 => [0, 0, 0, 0],
        1 => [1, 1, 1, 1],
        2 => [2, 2, 2, 2],
        3 => [3, 3, 3, 2],
        4 => [4, 4, 3, 3],
        5 => [5, 4, 4, 3],
        6 => [6, 5, 4, 3],
        7 => [7, 5, 5, 3],
        8 => [8, 6, 5, 4],
        9 => [9, 6, 6, 4],
        10 => [10, 7, 6, 4],
        11 => [11, 8, 7, 5],
        12 => [12, 9, 7, 5],
        13 => [13, 10, 8, 6],
        _ => [14, 11, 8, 6],
    };

    let col: usize = match num_param_bands {
        15 => 0,
        12 => 1,
        9 => 2,
        7 => 3,
        // For unknown band counts, default to the closest spec column.
        n if n >= 13 => 0,
        n if n >= 10 => 1,
        n if n >= 8 => 2,
        _ => 3,
    };
    table_row[col]
}

// =====================================================================
// §4.2.13.1 / §4.2.13.2 — acpl_config_*
// =====================================================================

/// Selector for `acpl_config_1ch(acpl_1ch_mode)` per Table 59 caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acpl1chMode {
    /// PARTIAL — `acpl_qmf_band_minus1` is present.
    Partial,
    /// FULL — no QMF-band extra field.
    Full,
}

/// Parsed `acpl_config_1ch()` per §4.2.13.1 Table 59.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcplConfig1ch {
    pub num_param_bands_id: u8,
    pub num_param_bands: u32,
    pub quant_mode: AcplQuantMode,
    /// `acpl_qmf_band` per Table 59 — only meaningful when
    /// `mode == Partial`. Set to 0 for FULL (per the table init).
    pub qmf_band: u8,
}

/// `acpl_config_1ch(acpl_1ch_mode)` per §4.2.13.1 Table 59.
pub fn parse_acpl_config_1ch(br: &mut BitReader<'_>, mode: Acpl1chMode) -> Result<AcplConfig1ch> {
    let id = br.read_u32(2)? as u8;
    let qm = AcplQuantMode::from_bit(br.read_bit()?);
    let mut qmf_band = 0u8;
    if matches!(mode, Acpl1chMode::Partial) {
        // acpl_qmf_band = acpl_qmf_band_minus1 + 1 (3 bits).
        let m1 = br.read_u32(3)? as u8;
        qmf_band = m1 + 1;
    }
    Ok(AcplConfig1ch {
        num_param_bands_id: id,
        num_param_bands: num_param_bands_from_id(id as u32),
        quant_mode: qm,
        qmf_band,
    })
}

/// Parsed `acpl_config_2ch()` per §4.2.13.2 Table 60.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcplConfig2ch {
    pub num_param_bands_id: u8,
    pub num_param_bands: u32,
    /// Drives the ALPHA / BETA / BETA3 codebook quant mode
    /// (§4.3.11.2.2).
    pub quant_mode_0: AcplQuantMode,
    /// Drives the GAMMA codebook quant mode (§4.3.11.2.3).
    pub quant_mode_1: AcplQuantMode,
}

/// `acpl_config_2ch()` per §4.2.13.2 Table 60.
pub fn parse_acpl_config_2ch(br: &mut BitReader<'_>) -> Result<AcplConfig2ch> {
    let id = br.read_u32(2)? as u8;
    let qm0 = AcplQuantMode::from_bit(br.read_bit()?);
    let qm1 = AcplQuantMode::from_bit(br.read_bit()?);
    Ok(AcplConfig2ch {
        num_param_bands_id: id,
        num_param_bands: num_param_bands_from_id(id as u32),
        quant_mode_0: qm0,
        quant_mode_1: qm1,
    })
}

// =====================================================================
// §4.2.13.5 — acpl_framing_data
// =====================================================================

/// Parsed `acpl_framing_data()` per §4.2.13.5 Table 63.
#[derive(Debug, Clone)]
pub struct AcplFramingData {
    pub interpolation_type: AcplInterpolationType,
    /// `acpl_num_param_sets_cod` — 1-bit: 0 → 1 set, 1 → 2 sets per
    /// frame (Table 146 / §4.3.11.5.2).
    pub num_param_sets_cod: u8,
    /// Resolved `acpl_num_param_sets` (1 or 2).
    pub num_param_sets: u32,
    /// 5-bit timeslot indices, only present when
    /// `interpolation_type == Steep`. Length is `num_param_sets`.
    pub param_timeslots: Vec<u8>,
}

pub fn parse_acpl_framing_data(br: &mut BitReader<'_>) -> Result<AcplFramingData> {
    let interp_bit = br.read_bit()?;
    let interp = if interp_bit {
        AcplInterpolationType::Steep
    } else {
        AcplInterpolationType::Smooth
    };
    let nps_cod = br.read_bit()?;
    let nps_cod_val = if nps_cod { 1u8 } else { 0u8 };
    let num_param_sets = (nps_cod_val as u32) + 1;
    let mut slots = Vec::new();
    if matches!(interp, AcplInterpolationType::Steep) {
        for _ in 0..num_param_sets {
            slots.push(br.read_u32(5)? as u8);
        }
    }
    Ok(AcplFramingData {
        interpolation_type: interp,
        num_param_sets_cod: nps_cod_val,
        num_param_sets,
        param_timeslots: slots,
    })
}

// =====================================================================
// §4.2.13.7 — acpl_huff_data
// =====================================================================

/// One A-CPL Huffman parameter set: per-band recovered values from
/// `acpl_huff_data()` (§4.2.13.7 Table 65).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcplHuffParam {
    /// Per-band recovered deltas (`a_huff_data[start..data_bands]`).
    /// First entry is the F0 / DT seed, subsequent are DF or DT
    /// deltas depending on `direction_time`.
    pub values: Vec<i32>,
    /// `false = DIFF_FREQ` (F0 + DF), `true = DIFF_TIME` (DT only).
    pub direction_time: bool,
}

/// `acpl_huff_data(data_type, data_bands, start_band, quant_mode)` per
/// §4.2.13.7 Table 65.
///
/// The Huffman direction (`diff_type` bit) is read first; then either
/// the FREQ branch (one F0-codebook value at `start_band`, then DF
/// values for `start_band+1 .. data_bands`) or the TIME branch (DT
/// values for `start_band .. data_bands`) runs.
///
/// `data_bands` is the *upper bound* (exclusive). Per the spec the
/// loops index `i` from `start_band` to `data_bands` exclusive.
pub fn parse_acpl_huff_data(
    br: &mut BitReader<'_>,
    data_type: AcplDataType,
    data_bands: u32,
    start_band: u32,
    quant_mode: AcplQuantMode,
) -> Result<AcplHuffParam> {
    if start_band > data_bands {
        return Err(Error::invalid(
            "ac4: acpl_huff_data start_band > data_bands",
        ));
    }
    let diff_type = br.read_bit()?;
    let mut values: Vec<i32> = Vec::with_capacity((data_bands - start_band) as usize);
    if !diff_type {
        // DIFF_FREQ
        if data_bands > start_band {
            let hcb_f0 = get_acpl_hcb(data_type, quant_mode, AcplHcbType::F0);
            values.push(hcb_f0.decode_delta(br)?);
        }
        if data_bands > start_band + 1 {
            let hcb_df = get_acpl_hcb(data_type, quant_mode, AcplHcbType::Df);
            for _ in (start_band + 1)..data_bands {
                values.push(hcb_df.decode_delta(br)?);
            }
        }
    } else {
        // DIFF_TIME
        let hcb_dt = get_acpl_hcb(data_type, quant_mode, AcplHcbType::Dt);
        for _ in start_band..data_bands {
            values.push(hcb_dt.decode_delta(br)?);
        }
    }
    Ok(AcplHuffParam {
        values,
        direction_time: diff_type,
    })
}

// =====================================================================
// §4.2.13.6 — acpl_ec_data
// =====================================================================

/// `acpl_ec_data(data_type, data_bands, start_band, quant_mode)` per
/// §4.2.13.6 Table 64. Repeats `acpl_huff_data()` for each parameter
/// set in this frame.
pub fn parse_acpl_ec_data(
    br: &mut BitReader<'_>,
    data_type: AcplDataType,
    data_bands: u32,
    start_band: u32,
    quant_mode: AcplQuantMode,
    num_param_sets: u32,
) -> Result<Vec<AcplHuffParam>> {
    let mut out = Vec::with_capacity(num_param_sets as usize);
    for _ in 0..num_param_sets {
        out.push(parse_acpl_huff_data(
            br, data_type, data_bands, start_band, quant_mode,
        )?);
    }
    Ok(out)
}

// =====================================================================
// §4.2.13.3 / §4.2.13.4 — acpl_data_*
// =====================================================================

/// Parsed `acpl_data_1ch()` per §4.2.13.3 Table 61: framing data plus
/// `(alpha1, beta1)` parameter-set vectors.
#[derive(Debug, Clone)]
pub struct AcplData1ch {
    pub framing: AcplFramingData,
    pub alpha1: Vec<AcplHuffParam>,
    pub beta1: Vec<AcplHuffParam>,
}

/// `acpl_data_1ch()` per §4.2.13.3 Table 61.
///
/// Spec call: `acpl_alpha1 = acpl_ec_data(ALPHA, num_bands, start,
/// acpl_quant_mode)` and likewise for `beta1`. `num_bands` and `start`
/// come from the active `acpl_config_1ch()` (`num_param_bands` and
/// `acpl_param_band` respectively — `acpl_param_band` is derived from
/// `acpl_qmf_band` via `sb_to_pb()` in PARTIAL mode, else 0).
pub fn parse_acpl_data_1ch(
    br: &mut BitReader<'_>,
    num_bands: u32,
    start_band: u32,
    quant_mode: AcplQuantMode,
) -> Result<AcplData1ch> {
    let framing = parse_acpl_framing_data(br)?;
    let alpha1 = parse_acpl_ec_data(
        br,
        AcplDataType::Alpha,
        num_bands,
        start_band,
        quant_mode,
        framing.num_param_sets,
    )?;
    let beta1 = parse_acpl_ec_data(
        br,
        AcplDataType::Beta,
        num_bands,
        start_band,
        quant_mode,
        framing.num_param_sets,
    )?;
    Ok(AcplData1ch {
        framing,
        alpha1,
        beta1,
    })
}

/// Parsed `acpl_data_2ch()` per §4.2.13.4 Table 62: framing plus the
/// full set of `(alpha1, alpha2, beta1, beta2, beta3, gamma1..gamma6)`
/// parameter vectors. `quant_mode_0` drives ALPHA / BETA / BETA3,
/// `quant_mode_1` drives GAMMA.
#[derive(Debug, Clone)]
pub struct AcplData2ch {
    pub framing: AcplFramingData,
    pub alpha1: Vec<AcplHuffParam>,
    pub alpha2: Vec<AcplHuffParam>,
    pub beta1: Vec<AcplHuffParam>,
    pub beta2: Vec<AcplHuffParam>,
    pub beta3: Vec<AcplHuffParam>,
    pub gamma1: Vec<AcplHuffParam>,
    pub gamma2: Vec<AcplHuffParam>,
    pub gamma3: Vec<AcplHuffParam>,
    pub gamma4: Vec<AcplHuffParam>,
    pub gamma5: Vec<AcplHuffParam>,
    pub gamma6: Vec<AcplHuffParam>,
}

/// `acpl_data_2ch()` per §4.2.13.4 Table 62.
pub fn parse_acpl_data_2ch(
    br: &mut BitReader<'_>,
    num_bands: u32,
    start_band: u32,
    quant_mode_0: AcplQuantMode,
    quant_mode_1: AcplQuantMode,
) -> Result<AcplData2ch> {
    let framing = parse_acpl_framing_data(br)?;
    let nps = framing.num_param_sets;
    let ec_alpha = |br: &mut BitReader<'_>| -> Result<Vec<AcplHuffParam>> {
        parse_acpl_ec_data(
            br,
            AcplDataType::Alpha,
            num_bands,
            start_band,
            quant_mode_0,
            nps,
        )
    };
    let ec_beta = |br: &mut BitReader<'_>| -> Result<Vec<AcplHuffParam>> {
        parse_acpl_ec_data(
            br,
            AcplDataType::Beta,
            num_bands,
            start_band,
            quant_mode_0,
            nps,
        )
    };
    let ec_beta3 = |br: &mut BitReader<'_>| -> Result<Vec<AcplHuffParam>> {
        parse_acpl_ec_data(
            br,
            AcplDataType::Beta3,
            num_bands,
            start_band,
            quant_mode_0,
            nps,
        )
    };
    let ec_gamma = |br: &mut BitReader<'_>| -> Result<Vec<AcplHuffParam>> {
        parse_acpl_ec_data(
            br,
            AcplDataType::Gamma,
            num_bands,
            start_band,
            quant_mode_1,
            nps,
        )
    };
    let alpha1 = ec_alpha(br)?;
    let alpha2 = ec_alpha(br)?;
    let beta1 = ec_beta(br)?;
    let beta2 = ec_beta(br)?;
    let beta3 = ec_beta3(br)?;
    let gamma1 = ec_gamma(br)?;
    let gamma2 = ec_gamma(br)?;
    let gamma3 = ec_gamma(br)?;
    let gamma4 = ec_gamma(br)?;
    let gamma5 = ec_gamma(br)?;
    let gamma6 = ec_gamma(br)?;
    Ok(AcplData2ch {
        framing,
        alpha1,
        alpha2,
        beta1,
        beta2,
        beta3,
        gamma1,
        gamma2,
        gamma3,
        gamma4,
        gamma5,
        gamma6,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    /// `get_acpl_hcb()` must return the right codebook by name for
    /// every `(data_type, quant_mode, hcb_type)` combination. Sanity
    /// check the lookup matrix doesn't have copy-paste errors.
    #[test]
    fn get_acpl_hcb_returns_named_codebook_per_combination() {
        for &(dt, dt_name) in &[
            (AcplDataType::Alpha, "ALPHA"),
            (AcplDataType::Beta, "BETA"),
            (AcplDataType::Beta3, "BETA3"),
            (AcplDataType::Gamma, "GAMMA"),
        ] {
            for &(qm, qm_name) in &[
                (AcplQuantMode::Coarse, "COARSE"),
                (AcplQuantMode::Fine, "FINE"),
            ] {
                for &(ht, ht_name) in &[
                    (AcplHcbType::F0, "F0"),
                    (AcplHcbType::Df, "DF"),
                    (AcplHcbType::Dt, "DT"),
                ] {
                    let hcb = get_acpl_hcb(dt, qm, ht);
                    let want = format!("ACPL_HCB_{dt_name}_{qm_name}_{ht_name}");
                    assert_eq!(hcb.name, want, "dt={dt_name} qm={qm_name} ht={ht_name}");
                    assert_eq!(hcb.len.len(), hcb.cw.len());
                }
            }
        }
    }

    #[test]
    fn quant_mode_from_bit_matches_table_144() {
        // Table 144: 0 → Fine, 1 → Coarse.
        assert_eq!(AcplQuantMode::from_bit(false), AcplQuantMode::Fine);
        assert_eq!(AcplQuantMode::from_bit(true), AcplQuantMode::Coarse);
    }

    #[test]
    fn num_param_bands_from_id_matches_table_143() {
        assert_eq!(num_param_bands_from_id(0), 15);
        assert_eq!(num_param_bands_from_id(1), 12);
        assert_eq!(num_param_bands_from_id(2), 9);
        assert_eq!(num_param_bands_from_id(3), 7);
    }

    /// Encode one Huffman codeword for ALPHA_COARSE_F0 at index = 8
    /// (cb_off = 0 → recovered delta = 8). Index 8 has the shortest
    /// codeword in that table (len=1, cw=0).
    #[test]
    fn alpha_coarse_f0_decodes_zero_delta() {
        let hcb = get_acpl_hcb(AcplDataType::Alpha, AcplQuantMode::Coarse, AcplHcbType::F0);
        // Spec table A.34: cb_off = 0; index 8 is "shortest" (len=1, cw=0).
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(hcb.decode_delta(&mut br).unwrap(), 8);
    }

    /// Encode the cb_off=16 delta-zero codeword for ALPHA_COARSE_DF
    /// (index = 16 → recovered delta = 0). Per the table that's
    /// (len=1, cw=0).
    #[test]
    fn alpha_coarse_df_decodes_zero_delta_at_cb_off() {
        let hcb = get_acpl_hcb(AcplDataType::Alpha, AcplQuantMode::Coarse, AcplHcbType::Df);
        // cb_off = 16, index 16 has the shortest codeword.
        assert_eq!(hcb.cb_off, 16);
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(hcb.decode_delta(&mut br).unwrap(), 0);
    }

    /// GAMMA_FINE_DT cb_off = 40. Index 40 has the shortest codeword
    /// (len=1, cw=0); recovered delta should be 0.
    #[test]
    fn gamma_fine_dt_decodes_zero_delta_at_cb_off() {
        let hcb = get_acpl_hcb(AcplDataType::Gamma, AcplQuantMode::Fine, AcplHcbType::Dt);
        assert_eq!(hcb.cb_off, 40);
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(hcb.decode_delta(&mut br).unwrap(), 0);
    }

    /// `parse_acpl_config_1ch(FULL)` — only `id` (2 bits) +
    /// `quant_mode` (1 bit). qmf_band stays 0.
    #[test]
    fn config_1ch_full_reads_3_bits() {
        let mut bw = BitWriter::new();
        bw.write_u32(0b10, 2); // id = 2 → 9 bands
        bw.write_u32(0b1, 1); // quant_mode = COARSE
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = parse_acpl_config_1ch(&mut br, Acpl1chMode::Full).unwrap();
        assert_eq!(cfg.num_param_bands_id, 2);
        assert_eq!(cfg.num_param_bands, 9);
        assert_eq!(cfg.quant_mode, AcplQuantMode::Coarse);
        assert_eq!(cfg.qmf_band, 0);
    }

    /// `parse_acpl_config_1ch(PARTIAL)` — adds 3 bits for
    /// `acpl_qmf_band_minus1`. qmf_band = minus1 + 1.
    #[test]
    fn config_1ch_partial_reads_qmf_band() {
        let mut bw = BitWriter::new();
        bw.write_u32(0b00, 2); // id = 0 → 15 bands
        bw.write_u32(0b0, 1); // quant_mode = FINE
        bw.write_u32(0b101, 3); // minus1 = 5 → qmf_band = 6
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = parse_acpl_config_1ch(&mut br, Acpl1chMode::Partial).unwrap();
        assert_eq!(cfg.num_param_bands_id, 0);
        assert_eq!(cfg.num_param_bands, 15);
        assert_eq!(cfg.quant_mode, AcplQuantMode::Fine);
        assert_eq!(cfg.qmf_band, 6);
    }

    #[test]
    fn config_2ch_reads_both_quant_modes() {
        let mut bw = BitWriter::new();
        bw.write_u32(0b11, 2); // id = 3 → 7 bands
        bw.write_u32(0b1, 1); // quant_mode_0 = COARSE
        bw.write_u32(0b0, 1); // quant_mode_1 = FINE
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = parse_acpl_config_2ch(&mut br).unwrap();
        assert_eq!(cfg.num_param_bands_id, 3);
        assert_eq!(cfg.num_param_bands, 7);
        assert_eq!(cfg.quant_mode_0, AcplQuantMode::Coarse);
        assert_eq!(cfg.quant_mode_1, AcplQuantMode::Fine);
    }

    /// `acpl_framing_data()` SMOOTH — interp_type=0, nps_cod=0 →
    /// 2 bits total, no timeslots.
    #[test]
    fn framing_smooth_one_param_set() {
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1); // interp = SMOOTH
        bw.write_u32(0, 1); // num_param_sets_cod = 0 → 1 set
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let framing = parse_acpl_framing_data(&mut br).unwrap();
        assert_eq!(framing.interpolation_type, AcplInterpolationType::Smooth);
        assert_eq!(framing.num_param_sets, 1);
        assert!(framing.param_timeslots.is_empty());
    }

    /// `acpl_framing_data()` STEEP / 2 sets → reads 2 timeslot codes.
    #[test]
    fn framing_steep_two_param_sets_reads_two_timeslots() {
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // interp = STEEP
        bw.write_u32(1, 1); // num_param_sets_cod = 1 → 2 sets
        bw.write_u32(7, 5); // ts[0] = 7
        bw.write_u32(23, 5); // ts[1] = 23
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let framing = parse_acpl_framing_data(&mut br).unwrap();
        assert_eq!(framing.interpolation_type, AcplInterpolationType::Steep);
        assert_eq!(framing.num_param_sets, 2);
        assert_eq!(framing.param_timeslots, vec![7, 23]);
    }

    /// Round-trip: encode an ALPHA FREQ-direction band stream and
    /// decode it back through `parse_acpl_huff_data()`. Verifies the
    /// F0 / DF codebook routing AND the diff_type bit handling.
    #[test]
    fn huff_data_freq_alpha_round_trips() {
        let hcb_f0 = get_acpl_hcb(AcplDataType::Alpha, AcplQuantMode::Coarse, AcplHcbType::F0);
        let hcb_df = get_acpl_hcb(AcplDataType::Alpha, AcplQuantMode::Coarse, AcplHcbType::Df);
        // F0 cb_off = 0, want delta = 8 → encode index 8 (len=1, cw=0).
        // DF cb_off = 16, want delta 0, +1, -1 → indexes 16, 17, 15.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1); // diff_type = 0 → DIFF_FREQ
        bw.write_u32(hcb_f0.cw[8], hcb_f0.len[8] as u32);
        bw.write_u32(hcb_df.cw[16], hcb_df.len[16] as u32);
        bw.write_u32(hcb_df.cw[17], hcb_df.len[17] as u32);
        bw.write_u32(hcb_df.cw[15], hcb_df.len[15] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        // Read 4 bands — 1 F0 + 3 DF.
        let p = parse_acpl_huff_data(&mut br, AcplDataType::Alpha, 4, 0, AcplQuantMode::Coarse)
            .unwrap();
        assert!(!p.direction_time);
        assert_eq!(p.values, vec![8, 0, 1, -1]);
    }

    /// Round-trip: TIME direction (DT only) for BETA. Encode 3 bands.
    #[test]
    fn huff_data_time_beta_round_trips() {
        let hcb_dt = get_acpl_hcb(AcplDataType::Beta, AcplQuantMode::Fine, AcplHcbType::Dt);
        // BETA_FINE_DT cb_off = 8, idx 8 = (1, 0) → delta 0; idx 9 = (2, 2) → +1.
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // diff_type = 1 → DIFF_TIME
        bw.write_u32(hcb_dt.cw[8], hcb_dt.len[8] as u32);
        bw.write_u32(hcb_dt.cw[9], hcb_dt.len[9] as u32);
        bw.write_u32(hcb_dt.cw[7], hcb_dt.len[7] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let p =
            parse_acpl_huff_data(&mut br, AcplDataType::Beta, 3, 0, AcplQuantMode::Fine).unwrap();
        assert!(p.direction_time);
        assert_eq!(p.values, vec![0, 1, -1]);
    }

    /// `acpl_ec_data()` runs `parse_acpl_huff_data` once per param set.
    #[test]
    fn ec_data_runs_once_per_param_set() {
        let hcb_f0 = get_acpl_hcb(AcplDataType::Beta, AcplQuantMode::Coarse, AcplHcbType::F0);
        // BETA_COARSE_F0[5]: idx 0=(1,0), so delta = 0.
        // 2 param sets, each: diff_type bit + one F0 codeword.
        let mut bw = BitWriter::new();
        for _ in 0..2 {
            bw.write_u32(0, 1); // DIFF_FREQ
            bw.write_u32(hcb_f0.cw[0], hcb_f0.len[0] as u32); // delta = 0
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let sets = parse_acpl_ec_data(
            &mut br,
            AcplDataType::Beta,
            1, // 1 band only
            0,
            AcplQuantMode::Coarse,
            2, // 2 param sets
        )
        .unwrap();
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].values, vec![0]);
        assert_eq!(sets[1].values, vec![0]);
    }

    /// End-to-end `acpl_data_1ch()` synthetic frame: SMOOTH framing,
    /// 1 param set, 2 bands, ALPHA + BETA both in DIFF_TIME mode.
    #[test]
    fn data_1ch_smooth_one_set_two_bands() {
        let alpha_dt = get_acpl_hcb(AcplDataType::Alpha, AcplQuantMode::Coarse, AcplHcbType::Dt);
        let beta_dt = get_acpl_hcb(AcplDataType::Beta, AcplQuantMode::Coarse, AcplHcbType::Dt);
        let mut bw = BitWriter::new();
        // framing: SMOOTH, 1 param set
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // alpha1 set 0: DT, 2 bands at idx cb_off and cb_off+1
        bw.write_u32(1, 1); // DIFF_TIME
        bw.write_u32(alpha_dt.cw[16], alpha_dt.len[16] as u32);
        bw.write_u32(alpha_dt.cw[17], alpha_dt.len[17] as u32);
        // beta1 set 0: DT, 2 bands at idx cb_off and cb_off-1
        bw.write_u32(1, 1); // DIFF_TIME
        bw.write_u32(beta_dt.cw[4], beta_dt.len[4] as u32);
        bw.write_u32(beta_dt.cw[3], beta_dt.len[3] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let data = parse_acpl_data_1ch(&mut br, 2, 0, AcplQuantMode::Coarse).unwrap();
        assert_eq!(data.framing.num_param_sets, 1);
        assert_eq!(data.alpha1.len(), 1);
        assert_eq!(data.alpha1[0].values, vec![0, 1]);
        assert_eq!(data.beta1.len(), 1);
        assert_eq!(data.beta1[0].values, vec![0, -1]);
    }

    // -------------------------------------------------------------
    // §5.7.7.2 Table 197 — sb_to_pb mapping
    // -------------------------------------------------------------

    #[test]
    fn sb_to_pb_15_bands_first_nine_are_one_to_one() {
        for sb in 0..=8u32 {
            assert_eq!(sb_to_pb(sb, 15), sb, "sb={sb} 15-band should be identity");
        }
    }

    #[test]
    fn sb_to_pb_15_bands_grouped_rows() {
        assert_eq!(sb_to_pb(9, 15), 9);
        assert_eq!(sb_to_pb(10, 15), 9);
        assert_eq!(sb_to_pb(11, 15), 10);
        assert_eq!(sb_to_pb(13, 15), 10);
        assert_eq!(sb_to_pb(14, 15), 11);
        assert_eq!(sb_to_pb(17, 15), 11);
        assert_eq!(sb_to_pb(18, 15), 12);
        assert_eq!(sb_to_pb(22, 15), 12);
        assert_eq!(sb_to_pb(23, 15), 13);
        assert_eq!(sb_to_pb(34, 15), 13);
        assert_eq!(sb_to_pb(35, 15), 14);
        assert_eq!(sb_to_pb(63, 15), 14);
    }

    #[test]
    fn sb_to_pb_12_bands_table_197() {
        // Selected anchor cells from Table 197's "12" column.
        assert_eq!(sb_to_pb(0, 12), 0);
        assert_eq!(sb_to_pb(3, 12), 3);
        assert_eq!(sb_to_pb(4, 12), 4);
        assert_eq!(sb_to_pb(5, 12), 4);
        assert_eq!(sb_to_pb(6, 12), 5);
        assert_eq!(sb_to_pb(7, 12), 5);
        assert_eq!(sb_to_pb(8, 12), 6);
        assert_eq!(sb_to_pb(10, 12), 6);
        assert_eq!(sb_to_pb(11, 12), 7);
        assert_eq!(sb_to_pb(13, 12), 7);
        assert_eq!(sb_to_pb(17, 12), 8);
        assert_eq!(sb_to_pb(22, 12), 9);
        assert_eq!(sb_to_pb(34, 12), 10);
        assert_eq!(sb_to_pb(63, 12), 11);
    }

    #[test]
    fn sb_to_pb_9_bands_table_197() {
        // Selected anchor cells from the "9" column.
        assert_eq!(sb_to_pb(0, 9), 0);
        assert_eq!(sb_to_pb(3, 9), 3);
        assert_eq!(sb_to_pb(4, 9), 3);
        assert_eq!(sb_to_pb(5, 9), 4);
        assert_eq!(sb_to_pb(7, 9), 5);
        assert_eq!(sb_to_pb(10, 9), 6);
        assert_eq!(sb_to_pb(13, 9), 6);
        assert_eq!(sb_to_pb(17, 9), 7);
        assert_eq!(sb_to_pb(22, 9), 7);
        assert_eq!(sb_to_pb(34, 9), 8);
        assert_eq!(sb_to_pb(63, 9), 8);
    }

    #[test]
    fn sb_to_pb_7_bands_table_197() {
        // Selected anchor cells from the "7" column.
        assert_eq!(sb_to_pb(0, 7), 0);
        assert_eq!(sb_to_pb(2, 7), 2);
        assert_eq!(sb_to_pb(3, 7), 2);
        assert_eq!(sb_to_pb(4, 7), 3);
        assert_eq!(sb_to_pb(7, 7), 3);
        assert_eq!(sb_to_pb(8, 7), 4);
        assert_eq!(sb_to_pb(10, 7), 4);
        assert_eq!(sb_to_pb(13, 7), 4);
        assert_eq!(sb_to_pb(17, 7), 5);
        assert_eq!(sb_to_pb(22, 7), 5);
        assert_eq!(sb_to_pb(34, 7), 6);
        assert_eq!(sb_to_pb(63, 7), 6);
    }

    #[test]
    fn sb_to_pb_image_below_num_param_bands() {
        // Property: for any (sb, npb) in the spec set, sb_to_pb returns
        // a value strictly less than npb.
        for &npb in &[15u32, 12, 9, 7] {
            for sb in 0..=63u32 {
                let pb = sb_to_pb(sb, npb);
                assert!(pb < npb, "sb_to_pb({sb}, {npb}) = {pb} >= {npb}");
            }
        }
    }
}
