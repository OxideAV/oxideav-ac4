//! Stereo downmix coefficients — ETSI TS 103 190-2 `stereo_dmx_coeff()`
//! and the TS 103 190-1 §4.3.12.2 code → gain mappings.
//!
//! `stereo_dmx_coeff()` is invoked from `bed_render_info()` (TS
//! 103 190-2 §6.2.8.8) but has no dedicated syntax box in either part
//! of the TS; its field layout is the factored-out form of the
//! identical inline `b_stereo_dmx_coeff` block that appears in
//! `custom_dmx_data()` (TS 103 190-2 §6.2.9.2):
//!
//! ```text
//! stereo_dmx_coeff()
//! {
//!   loro_centre_mixgain;                     3
//!   loro_surround_mixgain;                   3
//!   b_ltrt_mixinfo;                          1
//!   if (b_ltrt_mixinfo == 1) {
//!     ltrt_centre_mixgain;                   3
//!     ltrt_surround_mixgain;                 3
//!   }
//!   if (b_pres_has_lfe == 1) {
//!     b_lfe_mixinfo;                         1
//!     if (b_lfe_mixinfo == 1) {
//!       lfe_mixgain;                         5
//!     }
//!   }
//!   preferred_dmx_method;                    2
//! }
//! ```
//!
//! The `basic_metadata()` copy of the block (TS 103 190-1 §4.2.14.2
//! Table 67, handled in [`crate::metadata`]) additionally interleaves
//! the two downmix loudness-correction pairs; those fields are **not**
//! part of `stereo_dmx_coeff()` as invoked from the object/bed
//! rendering domain.
//!
//! Semantics: TS 103 190-1 §4.3.12.2.7-§4.3.12.2.19 (Tables 149 /
//! 149a / 150).

use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

/// Parsed `stereo_dmx_coeff()` element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StereoDmxCoeff {
    /// 3-bit `loro_centre_mixgain` (Table 149).
    pub loro_centre_mixgain: u8,
    /// 3-bit `loro_surround_mixgain` (Table 149a).
    pub loro_surround_mixgain: u8,
    /// `(ltrt_centre_mixgain, ltrt_surround_mixgain)` when
    /// `b_ltrt_mixinfo == 1`. When absent, the LtRt coefficients are
    /// identical to the LoRo coefficients (§4.3.12.2.12).
    pub ltrt_mixgains: Option<(u8, u8)>,
    /// 5-bit `lfe_mixgain` when `b_lfe_mixinfo == 1`. Only present in
    /// the bitstream when the invoking context has an LFE channel
    /// (`b_pres_has_lfe`).
    pub lfe_mixgain: Option<u8>,
    /// 2-bit `preferred_dmx_method` (Table 150).
    pub preferred_dmx_method: u8,
}

impl StereoDmxCoeff {
    /// Effective LtRt centre mixgain code — falls back to the LoRo
    /// code when `b_ltrt_mixinfo == 0` (§4.3.12.2.12).
    pub fn ltrt_centre_mixgain(&self) -> u8 {
        self.ltrt_mixgains
            .map(|(c, _)| c)
            .unwrap_or(self.loro_centre_mixgain)
    }

    /// Effective LtRt surround mixgain code — falls back to the LoRo
    /// code when `b_ltrt_mixinfo == 0` (§4.3.12.2.12).
    pub fn ltrt_surround_mixgain(&self) -> u8 {
        self.ltrt_mixgains
            .map(|(_, s)| s)
            .unwrap_or(self.loro_surround_mixgain)
    }
}

/// Parse `stereo_dmx_coeff()`. `has_lfe` is the invoking context's
/// LFE presence (`b_pres_has_lfe` / `channel_mode_contains_Lfe()`),
/// which gates the `b_lfe_mixinfo` bit.
pub fn parse_stereo_dmx_coeff(br: &mut BitReader<'_>, has_lfe: bool) -> Result<StereoDmxCoeff> {
    let loro_centre_mixgain = br.read_u32(3)? as u8;
    let loro_surround_mixgain = br.read_u32(3)? as u8;
    let ltrt_mixgains = if br.read_bit()? {
        // b_ltrt_mixinfo.
        Some((br.read_u32(3)? as u8, br.read_u32(3)? as u8))
    } else {
        None
    };
    let lfe_mixgain = if has_lfe && br.read_bit()? {
        // b_lfe_mixinfo.
        Some(br.read_u32(5)? as u8)
    } else {
        None
    };
    let preferred_dmx_method = br.read_u32(2)? as u8;
    Ok(StereoDmxCoeff {
        loro_centre_mixgain,
        loro_surround_mixgain,
        ltrt_mixgains,
        lfe_mixgain,
        preferred_dmx_method,
    })
}

/// Write `stereo_dmx_coeff()` — exact inverse of
/// [`parse_stereo_dmx_coeff`] under the same `has_lfe` context.
pub fn write_stereo_dmx_coeff(bw: &mut BitWriter, c: &StereoDmxCoeff, has_lfe: bool) -> Result<()> {
    if c.loro_centre_mixgain > 7
        || c.loro_surround_mixgain > 7
        || c.preferred_dmx_method > 3
        || matches!(c.ltrt_mixgains, Some((a, b)) if a > 7 || b > 7)
        || matches!(c.lfe_mixgain, Some(g) if g > 31)
    {
        return Err(Error::invalid("ac4: stereo_dmx_coeff field out of range"));
    }
    bw.write_u32(c.loro_centre_mixgain as u32, 3);
    bw.write_u32(c.loro_surround_mixgain as u32, 3);
    match c.ltrt_mixgains {
        Some((ctr, sur)) => {
            bw.write_bit(true);
            bw.write_u32(ctr as u32, 3);
            bw.write_u32(sur as u32, 3);
        }
        None => bw.write_bit(false),
    }
    if has_lfe {
        match c.lfe_mixgain {
            Some(g) => {
                bw.write_bit(true);
                bw.write_u32(g as u32, 5);
            }
            None => bw.write_bit(false),
        }
    } else if c.lfe_mixgain.is_some() {
        return Err(Error::invalid(
            "ac4: lfe_mixgain requires an LFE-carrying context",
        ));
    }
    bw.write_u32(c.preferred_dmx_method as u32, 2);
    Ok(())
}

// =====================================================================
// Code → gain mappings (TS 103 190-1 §4.3.12.2, Tables 149/149a/150)
// =====================================================================

/// Default downmix gain when no mixgains have been transmitted
/// (§4.3.12.2.8/9: −3,0 dB ≙ the Table 149 "0,707" linear step).
pub const DEFAULT_MIXGAIN_LINEAR: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `{loro,ltrt}_centre_mixgain` code → linear gain (Table 149). The
/// table's linear column steps in exact quarter-powers of two
/// (1,414 / 1,189 / 1,000 / 0,841 / 0,707 / 0,595 / 0,500) — the dB
/// column (+3,0 … −6,0 in 1,5 dB steps) is its rounding. Code 7 is
/// −∞ dB → 0,0.
pub fn centre_mixgain_linear(code: u8) -> Option<f32> {
    match code {
        0..=6 => Some(2.0f32.powf((2.0 - code as f32) / 4.0)),
        7 => Some(0.0),
        _ => None,
    }
}

/// `{loro,ltrt}_surround_mixgain` code → linear gain (Table 149a).
/// Codes 0 and 1 are reserved (`None`); codes 2..=6 step in exact
/// quarter-powers of two (1,000 … 0,500) and code 7 is −∞ dB → 0,0.
pub fn surround_mixgain_linear(code: u8) -> Option<f32> {
    match code {
        2..=6 => Some(2.0f32.powf((2.0 - code as f32) / 4.0)),
        7 => Some(0.0),
        _ => None,
    }
}

/// `lfe_mixgain` code → `Lfe_mg` gain in dB (§4.3.12.2.18):
/// `Lfe_mg = 5,5 − lfe_mixgain` (range −25,5 dB … +5,5 dB).
pub fn lfe_mixgain_db(code: u8) -> f32 {
    5.5 - code as f32
}

/// `{loro,ltrt}_dmx_loud_corr` code → correction gain in dB
/// (§4.3.12.2.11/16): `(15 − code) / 2`; the reserved value 31
/// indicates 0 dB.
pub fn dmx_loud_corr_db(code: u8) -> f32 {
    if code == 31 {
        0.0
    } else {
        (15.0 - code as f32) / 2.0
    }
}

/// Convert a dB gain to linear (−∞ dB → 0,0).
pub fn db_to_linear(db: f32) -> f32 {
    if db == f32::NEG_INFINITY {
        0.0
    } else {
        10.0f32.powf(db / 20.0)
    }
}

/// `preferred_dmx_method` resolution (Table 150): `None` = not
/// indicated, `Some(false)` = LoRo coefficients, `Some(true)` = LtRt
/// coefficients (both codes 2 and 3).
pub fn preferred_dmx_uses_ltrt(code: u8) -> Option<bool> {
    match code {
        1 => Some(false),
        2 | 3 => Some(true),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    fn round_trip(c: &StereoDmxCoeff, has_lfe: bool) {
        let mut bw = BitWriter::new();
        write_stereo_dmx_coeff(&mut bw, c, has_lfe).unwrap();
        bw.write_u32(0, 7); // trailing guard
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_stereo_dmx_coeff(&mut br, has_lfe).unwrap();
        assert_eq!(&got, c);
    }

    #[test]
    fn stereo_dmx_coeff_round_trips_all_shapes() {
        for has_lfe in [false, true] {
            for ltrt in [None, Some((1u8, 6u8))] {
                for lfe in [None, if has_lfe { Some(17u8) } else { None }] {
                    round_trip(
                        &StereoDmxCoeff {
                            loro_centre_mixgain: 4,
                            loro_surround_mixgain: 2,
                            ltrt_mixgains: ltrt,
                            lfe_mixgain: lfe,
                            preferred_dmx_method: 3,
                        },
                        has_lfe,
                    );
                }
            }
        }
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)] // groups mirror the field widths
    fn stereo_dmx_coeff_bit_layout_is_exact() {
        // loro_centre = 0b100, loro_surround = 0b010, b_ltrt = 1,
        // ltrt_centre = 0b011, ltrt_surround = 0b110, b_lfe_mixinfo = 1,
        // lfe_mixgain = 0b01010, preferred = 0b10.
        let mut bw = BitWriter::new();
        write_stereo_dmx_coeff(
            &mut bw,
            &StereoDmxCoeff {
                loro_centre_mixgain: 4,
                loro_surround_mixgain: 2,
                ltrt_mixgains: Some((3, 6)),
                lfe_mixgain: Some(10),
                preferred_dmx_method: 2,
            },
            true,
        )
        .unwrap();
        // 3 + 3 + 1 + 3 + 3 + 1 + 5 + 2 = 21 bits.
        assert_eq!(bw.bit_position(), 21);
        let bytes = bw.into_bytes();
        assert_eq!(bytes, vec![0b100_010_1_0, 0b11_110_1_01, 0b010_10_000]);
    }

    #[test]
    fn lfe_mixgain_rejected_without_lfe_context() {
        let mut bw = BitWriter::new();
        let err = write_stereo_dmx_coeff(
            &mut bw,
            &StereoDmxCoeff {
                lfe_mixgain: Some(1),
                ..Default::default()
            },
            false,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("LFE"));
    }

    #[test]
    fn ltrt_falls_back_to_loro() {
        let c = StereoDmxCoeff {
            loro_centre_mixgain: 5,
            loro_surround_mixgain: 6,
            ..Default::default()
        };
        assert_eq!(c.ltrt_centre_mixgain(), 5);
        assert_eq!(c.ltrt_surround_mixgain(), 6);
        let c2 = StereoDmxCoeff {
            ltrt_mixgains: Some((1, 2)),
            ..c
        };
        assert_eq!(c2.ltrt_centre_mixgain(), 1);
        assert_eq!(c2.ltrt_surround_mixgain(), 2);
    }

    #[test]
    fn table_149_centre_mixgain_matches_documented_linear_values() {
        // Table 149 linear column: 1,414 / 1,189 / 1,000 / 0,841 /
        // 0,707 / 0,595 / 0,500 / 0,000.
        let expect = [1.414, 1.189, 1.000, 0.841, 0.707, 0.595, 0.500, 0.000];
        for (code, &lin) in expect.iter().enumerate() {
            let got = centre_mixgain_linear(code as u8).unwrap();
            assert!(
                (got - lin).abs() < 5e-4,
                "code {code}: {got} vs table {lin}"
            );
        }
        assert!(centre_mixgain_linear(8).is_none());
    }

    #[test]
    fn table_149a_surround_mixgain_reserves_low_codes() {
        assert!(surround_mixgain_linear(0).is_none());
        assert!(surround_mixgain_linear(1).is_none());
        let expect = [1.000, 0.841, 0.707, 0.595, 0.500, 0.000];
        for (i, &lin) in expect.iter().enumerate() {
            let code = (i + 2) as u8;
            let got = surround_mixgain_linear(code).unwrap();
            assert!(
                (got - lin).abs() < 5e-4,
                "code {code}: {got} vs table {lin}"
            );
        }
    }

    #[test]
    fn default_mixgain_is_minus_3_db_step() {
        assert!((DEFAULT_MIXGAIN_LINEAR - 0.707).abs() < 5e-4);
    }

    #[test]
    fn lfe_mixgain_db_range() {
        assert_eq!(lfe_mixgain_db(0), 5.5);
        assert_eq!(lfe_mixgain_db(31), -25.5);
    }

    #[test]
    fn dmx_loud_corr_db_mapping() {
        assert_eq!(dmx_loud_corr_db(15), 0.0);
        assert_eq!(dmx_loud_corr_db(0), 7.5);
        assert_eq!(dmx_loud_corr_db(30), -7.5);
        // Reserved value 31 indicates 0 dB.
        assert_eq!(dmx_loud_corr_db(31), 0.0);
    }

    #[test]
    fn table_150_preferred_dmx_method() {
        assert_eq!(preferred_dmx_uses_ltrt(0), None);
        assert_eq!(preferred_dmx_uses_ltrt(1), Some(false));
        assert_eq!(preferred_dmx_uses_ltrt(2), Some(true));
        assert_eq!(preferred_dmx_uses_ltrt(3), Some(true));
    }
}
