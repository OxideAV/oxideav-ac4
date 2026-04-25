//! Dialogue Enhancement (DE) Huffman codebooks (Annex A.4, Tables
//! A.58..A.61 of ETSI TS 103 190-1 V1.4.1).
//!
//! Four codebooks ship with the AC-4 spec for the dialogue-enhancement
//! parametric data. They are organised as two `(method_parity)` ×
//! two `(coding_kind)` matrix:
//!
//! ```text
//!   table_idx ∈ { 0, 1 }      (de_method % 2)
//!   coding    ∈ { ABS, DIFF }
//! ```
//!
//! Both `de_abs_huffman` (§4.3.14.5.4 Pseudocode 16) and
//! `de_diff_huffman` (§4.3.14.5.5 Pseudocode 17) return
//! `huff_decode_diff(hcb, code) = symbol_index - cb_off`. The
//! per-codebook offsets per Tables A.58..A.61 are:
//!
//! ```text
//!   DE_HCB_ABS_0   codebook_length=32   cb_off=0
//!   DE_HCB_DIFF_0  codebook_length=63   cb_off=31
//!   DE_HCB_ABS_1   codebook_length=61   cb_off=30
//!   DE_HCB_DIFF_1  codebook_length=121  cb_off=60
//! ```
//!
//! For ABS_0 the `cb_off` is zero so the recovered value is exactly the
//! symbol index in `0..=31`; for ABS_1 the offset shifts the recovered
//! value into `-30..=+30`. For both DIFF codebooks the centre of the
//! table maps to `0` (the no-change delta).
//!
//! The four tables are transcribed verbatim from the normative ETSI
//! accompaniment file `ts_10319001v010401p0-tables.c` (lines 795..855
//! of that file). They are uncopyrightable numeric constants from the
//! published specification.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

// ============================================================================
// DE_HCB_ABS_0 (Table A.58)
// ============================================================================

/// `cb_off` for `DE_HCB_ABS_0` per Annex A.4 Table A.58.
pub const DE_HCB_ABS_0_OFFSET: i32 = 0;

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.58 — DE_HCB_ABS_0_LEN[32]
pub static DE_HCB_ABS_0_LEN: &[u8] = &[
    3, 3, 4, 4, 5, 5, 5, 5, 4, 4, 3, 7, 7, 7, 8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8,
];

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.58 — DE_HCB_ABS_0_CW[32]
pub static DE_HCB_ABS_0_CW: &[u32] = &[
    0x000006, 0x000003, 0x000009, 0x000001, 0x00001f, 0x00001d, 0x000015, 0x000017, 0x000000,
    0x000003, 0x000002, 0x000079, 0x000015, 0x000010, 0x000029, 0x000078, 0x00005a, 0x000020,
    0x00003d, 0x000039, 0x000022, 0x000021, 0x000028, 0x00002c, 0x000029, 0x000038, 0x000023,
    0x00000b, 0x000009, 0x00005b, 0x000011, 0x000028,
];

// ============================================================================
// DE_HCB_DIFF_0 (Table A.59)
// ============================================================================

/// `cb_off` for `DE_HCB_DIFF_0` per Annex A.4 Table A.59.
pub const DE_HCB_DIFF_0_OFFSET: i32 = 31;

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.59 — DE_HCB_DIFF_0_LEN[63]
pub static DE_HCB_DIFF_0_LEN: &[u8] = &[
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 10, 10, 9,
    8, 8, 7, 6, 5, 4, 3, 1, 3, 5, 5, 6, 7, 7, 7, 8, 8, 8, 10, 10, 10, 11, 11, 11, 11, 12, 12, 13,
    13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13,
];

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.59 — DE_HCB_DIFF_0_CW[63]
pub static DE_HCB_DIFF_0_CW: &[u32] = &[
    0x0002b0, 0x0002b1, 0x0002b4, 0x0002b5, 0x0002b7, 0x0002b6, 0x0002ba, 0x0002bb, 0x0002bc,
    0x00079d, 0x00148d, 0x0014b1, 0x00079e, 0x00079f, 0x00148f, 0x0014b6, 0x0014b4, 0x0014b5,
    0x00014c, 0x000159, 0x0003cc, 0x00002a, 0x00014a, 0x00003d, 0x00000b, 0x000057, 0x000028,
    0x000003, 0x000002, 0x000004, 0x000003, 0x000001, 0x000001, 0x00000b, 0x000000, 0x000006,
    0x00002a, 0x00000e, 0x000004, 0x000053, 0x00001f, 0x000056, 0x000149, 0x000078, 0x000028,
    0x000297, 0x000290, 0x0000f2, 0x000052, 0x000522, 0x0000a7, 0x000a59, 0x0003cd, 0x00015c,
    0x0014b7, 0x0014b0, 0x00148e, 0x00148c, 0x00079c, 0x0002bf, 0x0002bd, 0x0002be, 0x00014d,
];

// ============================================================================
// DE_HCB_ABS_1 (Table A.60)
// ============================================================================

/// `cb_off` for `DE_HCB_ABS_1` per Annex A.4 Table A.60.
pub const DE_HCB_ABS_1_OFFSET: i32 = 30;

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.60 — DE_HCB_ABS_1_LEN[61]
pub static DE_HCB_ABS_1_LEN: &[u8] = &[
    9, 12, 12, 12, 12, 12, 12, 11, 10, 11, 11, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 7,
    6, 6, 6, 5, 1, 4, 5, 5, 5, 5, 5, 5, 6, 6, 5, 6, 7, 6, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 10,
    11, 11, 12, 12, 10,
];

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.60 — DE_HCB_ABS_1_CW[61]
pub static DE_HCB_ABS_1_CW: &[u32] = &[
    0x00015c, 0x000aea, 0x000aeb, 0x000c56, 0x000c78, 0x000c79, 0x000e51, 0x000427, 0x000210,
    0x00062a, 0x00063d, 0x000212, 0x00021d, 0x00031d, 0x0002bb, 0x000314, 0x000390, 0x000395,
    0x00010b, 0x00015f, 0x00018b, 0x0001cb, 0x000086, 0x0000ab, 0x0000c6, 0x000054, 0x000020,
    0x000024, 0x000038, 0x00001e, 0x000000, 0x00000d, 0x00001f, 0x000017, 0x000016, 0x000014,
    0x000013, 0x000011, 0x00003b, 0x00003a, 0x000019, 0x000025, 0x000073, 0x000030, 0x000056,
    0x0000c4, 0x0000aa, 0x0001c9, 0x00015e, 0x00010f, 0x00010a, 0x000391, 0x00031c, 0x00021c,
    0x000729, 0x000211, 0x000574, 0x000426, 0x000e50, 0x000c57, 0x00031f,
];

// ============================================================================
// DE_HCB_DIFF_1 (Table A.61)
// ============================================================================

/// `cb_off` for `DE_HCB_DIFF_1` per Annex A.4 Table A.61.
pub const DE_HCB_DIFF_1_OFFSET: i32 = 60;

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.61 — DE_HCB_DIFF_1_LEN[121]
pub static DE_HCB_DIFF_1_LEN: &[u8] = &[
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 13, 13, 13,
    13, 13, 13, 12, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 10, 10, 10, 10, 9,
    9, 9, 8, 8, 7, 7, 7, 6, 6, 5, 4, 3, 1, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
    11, 11, 11, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
];

// FROM: ts_10319001v010401p0-tables.c §A.4 Table A.61 — DE_HCB_DIFF_1_CW[121]
pub static DE_HCB_DIFF_1_CW: &[u32] = &[
    0x001cf0, 0x001cbd, 0x001cbc, 0x001ccc, 0x001cb8, 0x001ccb, 0x001ccf, 0x001cc6, 0x001cca,
    0x001cc9, 0x001cc8, 0x001cce, 0x001cc5, 0x001cd8, 0x001cc4, 0x001cdf, 0x001cd1, 0x001cd5,
    0x001ce2, 0x001ce3, 0x000c21, 0x001cfc, 0x001cf1, 0x001cf2, 0x001cf6, 0x001cf4, 0x001cf7,
    0x000c20, 0x000c24, 0x000c25, 0x000729, 0x000d1e, 0x000d1f, 0x000d59, 0x000d5b, 0x000d5f,
    0x000e6d, 0x000e77, 0x00061b, 0x000613, 0x0006a8, 0x0006ae, 0x000730, 0x00030a, 0x00030c,
    0x000355, 0x000396, 0x00016c, 0x0001a2, 0x0001bb, 0x0000d0, 0x0000dc, 0x00005a, 0x000069,
    0x00006f, 0x000031, 0x000038, 0x000019, 0x00000a, 0x000004, 0x000000, 0x00000f, 0x00001d,
    0x000017, 0x000036, 0x00002c, 0x00006b, 0x000060, 0x0000e4, 0x0000d4, 0x0000b7, 0x0001ba,
    0x000187, 0x00016d, 0x000395, 0x00030b, 0x00073e, 0x000728, 0x0006a9, 0x00068d, 0x00068e,
    0x000e7f, 0x000611, 0x000d5e, 0x000d5a, 0x000d58, 0x000d19, 0x000d18, 0x000c34, 0x000c35,
    0x000e74, 0x001cfd, 0x001cf3, 0x001cf5, 0x001cec, 0x001ced, 0x001cea, 0x001ce7, 0x001ce5,
    0x001ce6, 0x001ce4, 0x001cdd, 0x001ce1, 0x001ce0, 0x001cd4, 0x001cde, 0x001cd7, 0x001cdc,
    0x001cd2, 0x001cd6, 0x001ccd, 0x001cd3, 0x001cd9, 0x001cbf, 0x001cbe, 0x001cd0, 0x001cbb,
    0x001cba, 0x001cb9, 0x001cc7, 0x001ceb,
];

// ============================================================================
// Decoder helpers
// ============================================================================

/// Generic Huffman decoder over a `(len[], cw[])` table pair: read MSB-
/// first bits one at a time, match against any `(len_i, cw_i)` of the
/// same length, then return `index - cb_off`.
///
/// Maximum codeword width across all four DE codebooks is 14 bits
/// (`DE_HCB_DIFF_0`); we cap defensively at 16.
fn huff_decode_diff_de(
    br: &mut BitReader<'_>,
    len_table: &[u8],
    cw_table: &[u32],
    cb_off: i32,
) -> Result<i32> {
    debug_assert_eq!(len_table.len(), cw_table.len());
    let mut code: u32 = 0;
    let mut width: u8 = 0;
    while width < 16 {
        let b = br.read_u32(1)?;
        code = (code << 1) | b;
        width += 1;
        for (i, &l) in len_table.iter().enumerate() {
            if l == width && cw_table[i] == code {
                return Ok(i as i32 - cb_off);
            }
        }
    }
    Err(Error::invalid("ac4: no matching DE_HCB codeword"))
}

/// `de_abs_huffman(table_idx, code)` — Pseudocode 16, §4.3.14.5.4.
/// Picks `DE_HCB_ABS_0` when `table_idx == 0`, else `DE_HCB_ABS_1`,
/// returns `huff_decode_diff(hcb, code) = symbol_index - cb_off`.
pub fn de_abs_huffman(br: &mut BitReader<'_>, table_idx: u32) -> Result<i32> {
    if table_idx == 0 {
        huff_decode_diff_de(br, DE_HCB_ABS_0_LEN, DE_HCB_ABS_0_CW, DE_HCB_ABS_0_OFFSET)
    } else {
        huff_decode_diff_de(br, DE_HCB_ABS_1_LEN, DE_HCB_ABS_1_CW, DE_HCB_ABS_1_OFFSET)
    }
}

/// `de_diff_huffman(table_idx, code)` — Pseudocode 17, §4.3.14.5.5.
/// Picks `DE_HCB_DIFF_0` when `table_idx == 0`, else `DE_HCB_DIFF_1`,
/// returns `huff_decode_diff(hcb, code) = symbol_index - cb_off`.
pub fn de_diff_huffman(br: &mut BitReader<'_>, table_idx: u32) -> Result<i32> {
    if table_idx == 0 {
        huff_decode_diff_de(
            br,
            DE_HCB_DIFF_0_LEN,
            DE_HCB_DIFF_0_CW,
            DE_HCB_DIFF_0_OFFSET,
        )
    } else {
        huff_decode_diff_de(
            br,
            DE_HCB_DIFF_1_LEN,
            DE_HCB_DIFF_1_CW,
            DE_HCB_DIFF_1_OFFSET,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    // ------------------------------------------------------------------------
    // Sanity: table sizes match the spec headers.
    // ------------------------------------------------------------------------

    #[test]
    fn table_sizes_match_spec() {
        // Tables A.58..A.61 codebook_length values.
        assert_eq!(DE_HCB_ABS_0_LEN.len(), 32);
        assert_eq!(DE_HCB_ABS_0_CW.len(), 32);
        assert_eq!(DE_HCB_DIFF_0_LEN.len(), 63);
        assert_eq!(DE_HCB_DIFF_0_CW.len(), 63);
        assert_eq!(DE_HCB_ABS_1_LEN.len(), 61);
        assert_eq!(DE_HCB_ABS_1_CW.len(), 61);
        assert_eq!(DE_HCB_DIFF_1_LEN.len(), 121);
        assert_eq!(DE_HCB_DIFF_1_CW.len(), 121);
    }

    // ------------------------------------------------------------------------
    // Kraft inequality: sum_i 2^-len_i == 1 for a complete prefix code.
    // ------------------------------------------------------------------------

    fn kraft_sum_equals_one(len: &[u8]) -> bool {
        let denom: u128 = 1u128 << 32;
        let mut sum: u128 = 0;
        for &l in len {
            sum += denom >> l as u128;
        }
        sum == denom
    }

    #[test]
    fn de_hcb_abs_0_kraft_sum_equals_one() {
        assert!(kraft_sum_equals_one(DE_HCB_ABS_0_LEN));
    }

    #[test]
    fn de_hcb_diff_0_kraft_sum_equals_one() {
        assert!(kraft_sum_equals_one(DE_HCB_DIFF_0_LEN));
    }

    #[test]
    fn de_hcb_abs_1_kraft_sum_equals_one() {
        assert!(kraft_sum_equals_one(DE_HCB_ABS_1_LEN));
    }

    #[test]
    fn de_hcb_diff_1_kraft_sum_equals_one() {
        assert!(kraft_sum_equals_one(DE_HCB_DIFF_1_LEN));
    }

    // ------------------------------------------------------------------------
    // Codeword fits in declared length.
    // ------------------------------------------------------------------------

    fn codewords_fit(len: &[u8], cw: &[u32]) -> bool {
        for (i, (&l, &c)) in len.iter().zip(cw.iter()).enumerate() {
            if l == 0 || l > 16 {
                eprintln!("DE_HCB[{i}] length {l} out of range");
                return false;
            }
            let max = (1u64 << l as u64) - 1;
            if c as u64 > max {
                eprintln!("DE_HCB[{i}] cw 0x{c:x} exceeds {l}-bit limit");
                return false;
            }
        }
        true
    }

    #[test]
    fn all_codewords_fit_in_declared_length() {
        assert!(codewords_fit(DE_HCB_ABS_0_LEN, DE_HCB_ABS_0_CW));
        assert!(codewords_fit(DE_HCB_DIFF_0_LEN, DE_HCB_DIFF_0_CW));
        assert!(codewords_fit(DE_HCB_ABS_1_LEN, DE_HCB_ABS_1_CW));
        assert!(codewords_fit(DE_HCB_DIFF_1_LEN, DE_HCB_DIFF_1_CW));
    }

    // ------------------------------------------------------------------------
    // Prefix-code property: pair-wise no collision and no shorter-prefixes-
    // longer relationship across the codebook.
    // ------------------------------------------------------------------------

    fn is_prefix_code(len: &[u8], cw: &[u32]) -> bool {
        for i in 0..len.len() {
            let li = len[i];
            let ci = cw[i];
            for j in (i + 1)..len.len() {
                let lj = len[j];
                let cj = cw[j];
                if li == lj {
                    if ci == cj {
                        eprintln!("collision at {i},{j}: len {li} cw 0x{ci:x}");
                        return false;
                    }
                } else {
                    let (short_len, short_cw, long_len, long_cw) = if li < lj {
                        (li, ci, lj, cj)
                    } else {
                        (lj, cj, li, ci)
                    };
                    let prefix = long_cw >> (long_len - short_len);
                    if prefix == short_cw {
                        eprintln!(
                            "prefix conflict at {i},{j}: short={short_len}b cw=0x{short_cw:x}, long={long_len}b cw=0x{long_cw:x}"
                        );
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn de_hcb_abs_0_is_prefix_code() {
        assert!(is_prefix_code(DE_HCB_ABS_0_LEN, DE_HCB_ABS_0_CW));
    }

    #[test]
    fn de_hcb_diff_0_is_prefix_code() {
        assert!(is_prefix_code(DE_HCB_DIFF_0_LEN, DE_HCB_DIFF_0_CW));
    }

    #[test]
    fn de_hcb_abs_1_is_prefix_code() {
        assert!(is_prefix_code(DE_HCB_ABS_1_LEN, DE_HCB_ABS_1_CW));
    }

    #[test]
    fn de_hcb_diff_1_is_prefix_code() {
        assert!(is_prefix_code(DE_HCB_DIFF_1_LEN, DE_HCB_DIFF_1_CW));
    }

    // ------------------------------------------------------------------------
    // End-to-end decode: write a codeword via BitWriter, read back the
    // expected `index - cb_off` via the public helpers.
    // ------------------------------------------------------------------------

    fn encode_and_decode_abs(table_idx: u32, sym_index: usize) -> i32 {
        let (len_table, cw_table, cb_off) = if table_idx == 0 {
            (DE_HCB_ABS_0_LEN, DE_HCB_ABS_0_CW, DE_HCB_ABS_0_OFFSET)
        } else {
            (DE_HCB_ABS_1_LEN, DE_HCB_ABS_1_CW, DE_HCB_ABS_1_OFFSET)
        };
        let l = len_table[sym_index] as u32;
        let cw = cw_table[sym_index];
        let mut bw = BitWriter::new();
        bw.write_u32(cw, l);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = de_abs_huffman(&mut br, table_idx).unwrap();
        // Sanity: the decoded value should equal sym_index - cb_off.
        assert_eq!(got, sym_index as i32 - cb_off);
        got
    }

    fn encode_and_decode_diff(table_idx: u32, sym_index: usize) -> i32 {
        let (len_table, cw_table, cb_off) = if table_idx == 0 {
            (DE_HCB_DIFF_0_LEN, DE_HCB_DIFF_0_CW, DE_HCB_DIFF_0_OFFSET)
        } else {
            (DE_HCB_DIFF_1_LEN, DE_HCB_DIFF_1_CW, DE_HCB_DIFF_1_OFFSET)
        };
        let l = len_table[sym_index] as u32;
        let cw = cw_table[sym_index];
        let mut bw = BitWriter::new();
        bw.write_u32(cw, l);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = de_diff_huffman(&mut br, table_idx).unwrap();
        assert_eq!(got, sym_index as i32 - cb_off);
        got
    }

    #[test]
    fn de_abs_0_decode_first_index_is_zero_offset() {
        // ABS_0 cb_off = 0 so symbol_index == returned value.
        assert_eq!(encode_and_decode_abs(0, 0), 0);
        assert_eq!(encode_and_decode_abs(0, 8), 8);
        assert_eq!(encode_and_decode_abs(0, 31), 31);
    }

    #[test]
    fn de_abs_1_decode_centre_is_zero() {
        // ABS_1 cb_off = 30 so symbol_index 30 maps to 0; index 30 has the
        // shortest codeword (len=1, cw=0x0).
        assert_eq!(DE_HCB_ABS_1_LEN[30], 1);
        assert_eq!(DE_HCB_ABS_1_CW[30], 0x0);
        assert_eq!(encode_and_decode_abs(1, 30), 0);
    }

    #[test]
    fn de_abs_1_decode_extremes() {
        assert_eq!(encode_and_decode_abs(1, 0), -30);
        assert_eq!(encode_and_decode_abs(1, 60), 30);
    }

    #[test]
    fn de_diff_0_decode_centre_is_zero() {
        // DIFF_0 cb_off = 31 so symbol_index 31 maps to 0; index 31 has
        // the shortest codeword (len=1, cw=0x1).
        assert_eq!(DE_HCB_DIFF_0_LEN[31], 1);
        assert_eq!(DE_HCB_DIFF_0_CW[31], 0x1);
        assert_eq!(encode_and_decode_diff(0, 31), 0);
    }

    #[test]
    fn de_diff_0_decode_neighbours() {
        assert_eq!(encode_and_decode_diff(0, 30), -1);
        assert_eq!(encode_and_decode_diff(0, 32), 1);
    }

    #[test]
    fn de_diff_1_decode_centre_is_zero() {
        // DIFF_1 cb_off = 60 so symbol_index 60 maps to 0; len=1, cw=0x0.
        assert_eq!(DE_HCB_DIFF_1_LEN[60], 1);
        assert_eq!(DE_HCB_DIFF_1_CW[60], 0x0);
        assert_eq!(encode_and_decode_diff(1, 60), 0);
    }

    #[test]
    fn de_diff_1_decode_extremes() {
        assert_eq!(encode_and_decode_diff(1, 0), -60);
        assert_eq!(encode_and_decode_diff(1, 120), 60);
    }

    #[test]
    fn de_back_to_back_decode() {
        // Concatenate three codewords from DE_HCB_DIFF_0: +1, 0, -1.
        let mut bw = BitWriter::new();
        // index 32 -> +1
        bw.write_u32(DE_HCB_DIFF_0_CW[32], DE_HCB_DIFF_0_LEN[32] as u32);
        // index 31 -> 0
        bw.write_u32(DE_HCB_DIFF_0_CW[31], DE_HCB_DIFF_0_LEN[31] as u32);
        // index 30 -> -1
        bw.write_u32(DE_HCB_DIFF_0_CW[30], DE_HCB_DIFF_0_LEN[30] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(de_diff_huffman(&mut br, 0).unwrap(), 1);
        assert_eq!(de_diff_huffman(&mut br, 0).unwrap(), 0);
        assert_eq!(de_diff_huffman(&mut br, 0).unwrap(), -1);
    }
}
