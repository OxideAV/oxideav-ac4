//! A-JOC Huffman codeword layer — ETSI TS 103 190-2 §6.2.5.5
//! `ajoc_huff_data()` + §6.3.6.5.2 Table 104 `get_ajoc_hcb()` over the
//! Annex A.1.1 codebooks (Tables A.1-A.12).
//!
//! Each A-JOC matrix row (one downmix channel of the dry matrix or one
//! decorrelator of the wet matrix, per data point) is coded as
//! `data_bands` Huffman codewords:
//!
//! * **DIFF_FREQ** (`diff_type == 0`): band 0 uses the `F0` codebook and
//!   `huff_decode()` (Part 1 §4.3.6.4.2 — the decoded value is the
//!   codeword *index*); bands 1.. use the `DF` codebook and
//!   `huff_decode_diff()` (Part 1 §4.3.10.8.3 — index minus the
//!   codebook's `cb_off`).
//! * **DIFF_TIME** (`diff_type == 1`): every band uses the `DT` codebook
//!   with `huff_decode_diff()`.
//!
//! When `b_dfonly` is set by `ajoc_data()` (`dp == 0 && ajoc_b_nodt`,
//! §6.2.5.3) the `diff_type` bit is not transmitted and frequency
//! coding is implied.
//!
//! The twelve codebooks are selected as
//! `AJOC_HCB_<data_type>_<quant_mode>_<hcb_type>` (Table 104) with
//! `data_type ∈ {DRY, WET}`, `quant_mode ∈ {COARSE, FINE}`,
//! `hcb_type ∈ {F0, DF, DT}`. The numeric `_LEN` / `_CW` arrays are
//! normative constants from the ETSI electronic attachment (see
//! `ajoc_hcb_tables.rs`); every book is a complete prefix code, so the
//! encoder in [`write_ajoc_huff_data`] is exact and self-inverse.

use crate::ajoc::{AjocDiffType, AjocMatrixKind, AjocQuantMode};
use crate::huffman::huff_decode;
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

include!("ajoc_hcb_tables.rs");

/// One A-JOC Huffman codebook: per-symbol code lengths and codewords
/// plus the `cb_off` subtracted by `huff_decode_diff()`
/// (Part 1 §4.3.10.8.3).
pub struct AjocHcb {
    /// Per-symbol code lengths in bits.
    pub len: &'static [u8],
    /// Per-symbol codewords (MSB-first, `len` bits each).
    pub cw: &'static [u32],
    /// Codebook offset: `huff_decode_diff()` returns `index - cb_off`.
    pub cb_off: i32,
}

/// `hcb_type` selector of Table 104.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AjocHcbType {
    /// First-band absolute codebook (frequency-differential path).
    F0,
    /// Frequency-difference codebook (bands 1.. of the freq path).
    Df,
    /// Time-difference codebook (all bands of the time path).
    Dt,
}

// Static codebook descriptors (one per Annex A.1.1 table).
static HCB_DRY_COARSE_F0: AjocHcb = AjocHcb {
    len: &AJOC_HCB_DRY_COARSE_F0_LEN,
    cw: &AJOC_HCB_DRY_COARSE_F0_CW,
    cb_off: 0,
};
static HCB_DRY_FINE_F0: AjocHcb = AjocHcb {
    len: &AJOC_HCB_DRY_FINE_F0_LEN,
    cw: &AJOC_HCB_DRY_FINE_F0_CW,
    cb_off: 0,
};
static HCB_DRY_COARSE_DF: AjocHcb = AjocHcb {
    len: &AJOC_HCB_DRY_COARSE_DF_LEN,
    cw: &AJOC_HCB_DRY_COARSE_DF_CW,
    cb_off: 0,
};
static HCB_DRY_FINE_DF: AjocHcb = AjocHcb {
    len: &AJOC_HCB_DRY_FINE_DF_LEN,
    cw: &AJOC_HCB_DRY_FINE_DF_CW,
    cb_off: 0,
};
static HCB_DRY_COARSE_DT: AjocHcb = AjocHcb {
    len: &AJOC_HCB_DRY_COARSE_DT_LEN,
    cw: &AJOC_HCB_DRY_COARSE_DT_CW,
    cb_off: 50,
};
static HCB_DRY_FINE_DT: AjocHcb = AjocHcb {
    len: &AJOC_HCB_DRY_FINE_DT_LEN,
    cw: &AJOC_HCB_DRY_FINE_DT_CW,
    cb_off: 100,
};
static HCB_WET_COARSE_F0: AjocHcb = AjocHcb {
    len: &AJOC_HCB_WET_COARSE_F0_LEN,
    cw: &AJOC_HCB_WET_COARSE_F0_CW,
    cb_off: 0,
};
static HCB_WET_FINE_F0: AjocHcb = AjocHcb {
    len: &AJOC_HCB_WET_FINE_F0_LEN,
    cw: &AJOC_HCB_WET_FINE_F0_CW,
    cb_off: 0,
};
static HCB_WET_COARSE_DF: AjocHcb = AjocHcb {
    len: &AJOC_HCB_WET_COARSE_DF_LEN,
    cw: &AJOC_HCB_WET_COARSE_DF_CW,
    cb_off: 0,
};
static HCB_WET_FINE_DF: AjocHcb = AjocHcb {
    len: &AJOC_HCB_WET_FINE_DF_LEN,
    cw: &AJOC_HCB_WET_FINE_DF_CW,
    cb_off: 0,
};
static HCB_WET_COARSE_DT: AjocHcb = AjocHcb {
    len: &AJOC_HCB_WET_COARSE_DT_LEN,
    cw: &AJOC_HCB_WET_COARSE_DT_CW,
    cb_off: 20,
};
static HCB_WET_FINE_DT: AjocHcb = AjocHcb {
    len: &AJOC_HCB_WET_FINE_DT_LEN,
    cw: &AJOC_HCB_WET_FINE_DT_CW,
    cb_off: 40,
};

/// `get_ajoc_hcb(data_type, quant_mode, hcb_type)` — §6.3.6.5.2
/// Table 104: select the `AJOC_HCB_<data_type>_<quant_mode>_<hcb_type>`
/// codebook.
pub fn get_ajoc_hcb(
    data_type: AjocMatrixKind,
    quant_mode: AjocQuantMode,
    hcb_type: AjocHcbType,
) -> &'static AjocHcb {
    use AjocHcbType::*;
    use AjocMatrixKind::*;
    use AjocQuantMode::*;
    match (data_type, quant_mode, hcb_type) {
        (Dry, Coarse, F0) => &HCB_DRY_COARSE_F0,
        (Dry, Fine, F0) => &HCB_DRY_FINE_F0,
        (Dry, Coarse, Df) => &HCB_DRY_COARSE_DF,
        (Dry, Fine, Df) => &HCB_DRY_FINE_DF,
        (Dry, Coarse, Dt) => &HCB_DRY_COARSE_DT,
        (Dry, Fine, Dt) => &HCB_DRY_FINE_DT,
        (Wet, Coarse, F0) => &HCB_WET_COARSE_F0,
        (Wet, Fine, F0) => &HCB_WET_FINE_F0,
        (Wet, Coarse, Df) => &HCB_WET_COARSE_DF,
        (Wet, Fine, Df) => &HCB_WET_FINE_DF,
        (Wet, Coarse, Dt) => &HCB_WET_COARSE_DT,
        (Wet, Fine, Dt) => &HCB_WET_FINE_DT,
    }
}

/// `huff_decode_diff(hcb, hcw)` — Part 1 §4.3.10.8.3: decode one
/// codeword and return its index minus the codebook offset.
pub fn huff_decode_diff(br: &mut BitReader<'_>, hcb: &AjocHcb) -> Result<i32> {
    Ok(huff_decode(br, hcb.len, hcb.cw)? as i32 - hcb.cb_off)
}

/// Decode one `ajoc_huff_data(data_type, data_bands, quant_select,
/// b_dfonly)` element (§6.2.5.5): the `diff_type` bit (unless
/// `b_dfonly`) followed by `data_bands` Huffman codewords.
///
/// Returns `(diff_type, a_huff_data)` where `a_huff_data[0]` is the
/// absolute quantised index for the frequency path (F0 codebook) and
/// the remaining entries are differences — exactly the `mix_mtx_*`
/// input expected by the §5.7.3.2 Table 43 differential decoder
/// ([`crate::ajoc::differential_decode_dry`] /
/// [`crate::ajoc::differential_decode_wet`]).
pub fn ajoc_huff_data(
    br: &mut BitReader<'_>,
    data_type: AjocMatrixKind,
    data_bands: usize,
    quant_select: AjocQuantMode,
    b_dfonly: bool,
) -> Result<(AjocDiffType, Vec<i32>)> {
    let diff_type = if b_dfonly {
        AjocDiffType::Freq
    } else {
        AjocDiffType::from_bit(br.read_bit()?)
    };
    let mut a_huff_data = Vec::with_capacity(data_bands);
    match diff_type {
        AjocDiffType::Freq => {
            if data_bands > 0 {
                let hcb = get_ajoc_hcb(data_type, quant_select, AjocHcbType::F0);
                a_huff_data.push(huff_decode(br, hcb.len, hcb.cw)? as i32);
                let hcb = get_ajoc_hcb(data_type, quant_select, AjocHcbType::Df);
                for _ in 1..data_bands {
                    a_huff_data.push(huff_decode_diff(br, hcb)?);
                }
            }
        }
        AjocDiffType::Time => {
            let hcb = get_ajoc_hcb(data_type, quant_select, AjocHcbType::Dt);
            for _ in 0..data_bands {
                a_huff_data.push(huff_decode_diff(br, hcb)?);
            }
        }
    }
    Ok((diff_type, a_huff_data))
}

/// Write one codeword for symbol index `idx` of `hcb`.
fn write_codeword(bw: &mut BitWriter, hcb: &AjocHcb, idx: i32) -> Result<()> {
    if idx < 0 || idx as usize >= hcb.len.len() {
        return Err(Error::invalid("ac4: A-JOC Huffman symbol out of range"));
    }
    let i = idx as usize;
    bw.write_u32(hcb.cw[i], hcb.len[i] as u32);
    Ok(())
}

/// Encode one `ajoc_huff_data()` element — the exact inverse of
/// [`ajoc_huff_data`]. `a_huff_data` uses the decoder convention
/// (`[0]` absolute for the frequency path, differences elsewhere).
///
/// Errors if a value is outside its codebook's symbol range (freq
/// differences must fit `[-cb_off_df, len_df - 1 - cb_off_df]`, i.e.
/// the encoder is expected to fold them modulo `nquant` per the
/// Table 43 running-sum inverse before calling this), or if
/// `diff_type == Time` while `b_dfonly` forbids the time direction.
pub fn write_ajoc_huff_data(
    bw: &mut BitWriter,
    data_type: AjocMatrixKind,
    quant_select: AjocQuantMode,
    b_dfonly: bool,
    diff_type: AjocDiffType,
    a_huff_data: &[i32],
) -> Result<()> {
    if b_dfonly {
        if diff_type == AjocDiffType::Time {
            return Err(Error::invalid(
                "ac4: A-JOC time-differential coding forbidden when b_dfonly",
            ));
        }
    } else {
        bw.write_bit(diff_type == AjocDiffType::Time);
    }
    match diff_type {
        AjocDiffType::Freq => {
            if let Some((&f0, rest)) = a_huff_data.split_first() {
                let hcb = get_ajoc_hcb(data_type, quant_select, AjocHcbType::F0);
                write_codeword(bw, hcb, f0)?;
                let hcb = get_ajoc_hcb(data_type, quant_select, AjocHcbType::Df);
                for &d in rest {
                    write_codeword(bw, hcb, d + hcb.cb_off)?;
                }
            }
        }
        AjocDiffType::Time => {
            let hcb = get_ajoc_hcb(data_type, quant_select, AjocHcbType::Dt);
            for &d in a_huff_data {
                write_codeword(bw, hcb, d + hcb.cb_off)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_books() -> Vec<(&'static str, &'static AjocHcb, usize, i32)> {
        // (name, book, expected length, expected cb_off) per Annex A.1.1.
        vec![
            ("DRY_COARSE_F0", &HCB_DRY_COARSE_F0, 51, 0),
            ("DRY_FINE_F0", &HCB_DRY_FINE_F0, 101, 0),
            ("DRY_COARSE_DF", &HCB_DRY_COARSE_DF, 51, 0),
            ("DRY_FINE_DF", &HCB_DRY_FINE_DF, 101, 0),
            ("DRY_COARSE_DT", &HCB_DRY_COARSE_DT, 101, 50),
            ("DRY_FINE_DT", &HCB_DRY_FINE_DT, 201, 100),
            ("WET_COARSE_F0", &HCB_WET_COARSE_F0, 21, 0),
            ("WET_FINE_F0", &HCB_WET_FINE_F0, 41, 0),
            ("WET_COARSE_DF", &HCB_WET_COARSE_DF, 21, 0),
            ("WET_FINE_DF", &HCB_WET_FINE_DF, 41, 0),
            ("WET_COARSE_DT", &HCB_WET_COARSE_DT, 41, 20),
            ("WET_FINE_DT", &HCB_WET_FINE_DT, 81, 40),
        ]
    }

    #[test]
    fn codebook_metadata_matches_annex_a11() {
        for (name, hcb, n, off) in all_books() {
            assert_eq!(hcb.len.len(), n, "{name} length");
            assert_eq!(hcb.cw.len(), n, "{name} cw length");
            assert_eq!(hcb.cb_off, off, "{name} cb_off");
        }
    }

    #[test]
    fn codebooks_are_complete_prefix_codes() {
        for (name, hcb, _, _) in all_books() {
            // Kraft sum == 1 exactly: Σ 2^(32-len) == 2^32.
            let kraft: u64 = hcb.len.iter().map(|&l| 1u64 << (32 - l as u64)).sum();
            assert_eq!(kraft, 1u64 << 32, "{name} Kraft sum");
            // Codewords fit their lengths and are pairwise prefix-free.
            for i in 0..hcb.len.len() {
                let (li, ci) = (hcb.len[i] as u32, hcb.cw[i]);
                assert!((1..=32).contains(&li), "{name}[{i}] length");
                assert!(li == 32 || ci < (1u32 << li), "{name}[{i}] overflow");
                for j in (i + 1)..hcb.len.len() {
                    let (lj, cj) = (hcb.len[j] as u32, hcb.cw[j]);
                    let (ls, cs, ll, cl) = if li <= lj {
                        (li, ci, lj, cj)
                    } else {
                        (lj, cj, li, ci)
                    };
                    assert!(ls != ll || cs != cl, "{name}: duplicate codeword {i}/{j}");
                    assert!(cl >> (ll - ls) != cs, "{name}: codeword {i} prefixes {j}");
                }
            }
        }
    }

    #[test]
    fn every_symbol_roundtrips_through_the_bitstream() {
        for (name, hcb, n, _) in all_books() {
            let mut bw = BitWriter::new();
            for i in 0..n {
                bw.write_u32(hcb.cw[i], hcb.len[i] as u32);
            }
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            for i in 0..n {
                let got = huff_decode(&mut br, hcb.len, hcb.cw).unwrap();
                assert_eq!(got as usize, i, "{name} symbol {i}");
            }
        }
    }

    #[test]
    fn table_104_selects_the_right_book() {
        use AjocHcbType::*;
        use AjocMatrixKind::*;
        use AjocQuantMode::*;
        // Each (data_type, quant_mode, hcb_type) picks the book whose
        // symbol count matches nquant (F0/DF) or 2·nquant - 1 (DT).
        for kind in [Dry, Wet] {
            for mode in [Coarse, Fine] {
                let nq = mode.nquant(kind) as usize;
                assert_eq!(get_ajoc_hcb(kind, mode, F0).len.len(), nq);
                assert_eq!(get_ajoc_hcb(kind, mode, Df).len.len(), nq);
                let dt = get_ajoc_hcb(kind, mode, Dt);
                assert_eq!(dt.len.len(), 2 * nq - 1);
                assert_eq!(dt.cb_off, nq as i32 - 1);
            }
        }
    }

    #[test]
    fn huff_data_freq_roundtrip() {
        // 5 bands, dry coarse: band 0 absolute (0..=50), then DF values.
        let values = vec![25, 3, 0, 50, 12];
        let mut bw = BitWriter::new();
        bw.write_bit(true); // marker to prove bit alignment carries over
        write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Dry,
            AjocQuantMode::Coarse,
            false,
            AjocDiffType::Freq,
            &values,
        )
        .unwrap();
        bw.write_bit(true); // trailing marker
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert!(br.read_bit().unwrap());
        let (dt, got) = ajoc_huff_data(
            &mut br,
            AjocMatrixKind::Dry,
            5,
            AjocQuantMode::Coarse,
            false,
        )
        .unwrap();
        assert_eq!(dt, AjocDiffType::Freq);
        assert_eq!(got, values);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn huff_data_time_roundtrip_signed_deltas() {
        // Wet fine DT: deltas in [-40, 40].
        let values = vec![-40, -1, 0, 1, 40, -7];
        let mut bw = BitWriter::new();
        write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Wet,
            AjocQuantMode::Fine,
            false,
            AjocDiffType::Time,
            &values,
        )
        .unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let (dt, got) =
            ajoc_huff_data(&mut br, AjocMatrixKind::Wet, 6, AjocQuantMode::Fine, false).unwrap();
        assert_eq!(dt, AjocDiffType::Time);
        assert_eq!(got, values);
    }

    #[test]
    fn huff_data_dfonly_skips_diff_type_bit() {
        let values = vec![10, 2];
        let mut bw = BitWriter::new();
        write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Wet,
            AjocQuantMode::Coarse,
            true,
            AjocDiffType::Freq,
            &values,
        )
        .unwrap();
        let with_dfonly = bw.finish();

        let mut bw = BitWriter::new();
        write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Wet,
            AjocQuantMode::Coarse,
            false,
            AjocDiffType::Freq,
            &values,
        )
        .unwrap();
        let without = bw.finish();

        // The explicit diff_type bit costs exactly one leading bit.
        let mut br = BitReader::new(&with_dfonly);
        let (dt, got) =
            ajoc_huff_data(&mut br, AjocMatrixKind::Wet, 2, AjocQuantMode::Coarse, true).unwrap();
        assert_eq!(dt, AjocDiffType::Freq);
        assert_eq!(got, values);

        let mut br = BitReader::new(&without);
        assert!(!br.read_bit().unwrap()); // diff_type = 0 transmitted
                                          // Time direction is rejected under b_dfonly.
        let mut bw = BitWriter::new();
        assert!(write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Wet,
            AjocQuantMode::Coarse,
            true,
            AjocDiffType::Time,
            &values,
        )
        .is_err());
    }

    #[test]
    fn out_of_range_symbols_rejected() {
        let mut bw = BitWriter::new();
        // Dry coarse F0 has 51 symbols (0..=50).
        assert!(write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Dry,
            AjocQuantMode::Coarse,
            false,
            AjocDiffType::Freq,
            &[51],
        )
        .is_err());
        // Wet coarse DT deltas are limited to [-20, 20].
        assert!(write_ajoc_huff_data(
            &mut bw,
            AjocMatrixKind::Wet,
            AjocQuantMode::Coarse,
            false,
            AjocDiffType::Time,
            &[21],
        )
        .is_err());
    }
}
