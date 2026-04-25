//! DRC Huffman codebook (Annex A.5, Table A.62 of ETSI TS 103 190-1).
//!
//! Single 255-entry codebook `DRC_HCB` for delta-coded DRC gain values.
//! Per spec §4.3.13.6.2 the codeword maps to `cb_idx - cb_off` where
//! `cb_off = 127`, so the recovered diff lies in `-127..=+127`.
//!
//! Used by `drc_gains()` (§4.2.14.10, Table 75): the first per-frame DRC
//! gain is sent as a 7-bit `drc_gain_val` (giving `drc_gain - 64 dB`),
//! and every subsequent per-(channel, subframe, band) gain is sent as a
//! Huffman-coded delta against the previous gain — `drc_gain_code` in
//! the bitstream, decoded by `huff_decode_diff(DRC_HCB, drc_gain_code)`
//! per §4.3.10.8.3. The "diff" value is then added to the running
//! reference gain.
//!
//! The table values below are transcribed verbatim from the normative
//! ETSI accompaniment file `ts_10319001v010401p0-tables.c` (Annex A.5,
//! lines 859..889 of that file). They are uncopyrightable numeric
//! constants from the published specification.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// `cb_off` for `DRC_HCB` per Annex A.5 Table A.62.
pub const DRC_HCB_OFFSET: i32 = 127;

// FROM: ts_10319001v010401p0-tables.c §A.5 Table A.62 — DRC_HCB_LEN[255]
pub static DRC_HCB_LEN: &[u8] = &[
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 9, 9, 8, 8, 7, 7,
    6, 6, 5, 4, 3, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12,
];

// FROM: ts_10319001v010401p0-tables.c §A.5 Table A.62 — DRC_HCB_CW[255]
pub static DRC_HCB_CW: &[u32] = &[
    0x000e12, 0x000e11, 0x000e10, 0x000e0f, 0x000e0e, 0x000e0d, 0x000e0c, 0x000e0b, 0x000e0a,
    0x000e09, 0x000e08, 0x000e07, 0x000e06, 0x000e05, 0x000e04, 0x000e03, 0x000e02, 0x000e01,
    0x000e00, 0x00053f, 0x00053e, 0x00053d, 0x00053c, 0x00053b, 0x00053a, 0x000539, 0x000538,
    0x000537, 0x000536, 0x000535, 0x000534, 0x000533, 0x000532, 0x000531, 0x000530, 0x00052f,
    0x00052e, 0x00052d, 0x00052c, 0x00052b, 0x00052a, 0x000529, 0x000528, 0x000527, 0x000526,
    0x000525, 0x000524, 0x000523, 0x000522, 0x000521, 0x000520, 0x00051f, 0x00051e, 0x00051d,
    0x00051c, 0x00051b, 0x00051a, 0x000519, 0x000518, 0x000517, 0x000516, 0x000515, 0x000514,
    0x000513, 0x000512, 0x000511, 0x000510, 0x00050f, 0x00050e, 0x00050d, 0x00050c, 0x00050b,
    0x00050a, 0x000509, 0x000508, 0x000507, 0x000506, 0x000505, 0x000504, 0x000503, 0x000502,
    0x000501, 0x000500, 0x0004ff, 0x0004fe, 0x0004fd, 0x0004fc, 0x0004fb, 0x0004fa, 0x0004f9,
    0x0004f8, 0x0004f7, 0x0004f6, 0x0004f5, 0x0004f4, 0x0004f3, 0x0004f2, 0x0004f1, 0x0004f0,
    0x0004ef, 0x0004ee, 0x0004ed, 0x0004ec, 0x0004eb, 0x0004ea, 0x0004e9, 0x0004e8, 0x000e13,
    0x0000ba, 0x0000bb, 0x00005c, 0x000385, 0x000016, 0x0001c3, 0x00002f, 0x000054, 0x000055,
    0x000056, 0x000022, 0x000023, 0x00000d, 0x000009, 0x00000b, 0x000039, 0x00001d, 0x00000f,
    0x000003, 0x000002, 0x000006, 0x000003, 0x000000, 0x00000b, 0x000008, 0x000015, 0x000007,
    0x000071, 0x000014, 0x000012, 0x000013, 0x00000c, 0x00000a, 0x000010, 0x000008, 0x0004e7,
    0x0004e6, 0x0004e5, 0x0004e4, 0x0004e3, 0x0004e2, 0x0004e1, 0x0004e0, 0x0004df, 0x0004de,
    0x0004dd, 0x0004dc, 0x0004db, 0x0004da, 0x0004d9, 0x0004d8, 0x0004d7, 0x0004d6, 0x0004d5,
    0x0004d4, 0x0004d3, 0x0004d2, 0x0004d1, 0x0004d0, 0x0004cf, 0x0004ce, 0x0004cd, 0x0004cc,
    0x0004cb, 0x0004ca, 0x0004c9, 0x0004c8, 0x0004c7, 0x0004c6, 0x0004c5, 0x0004c4, 0x0004c3,
    0x0004c2, 0x0004c1, 0x0004c0, 0x0004bf, 0x0004be, 0x0004bd, 0x0004bc, 0x0004bb, 0x0004ba,
    0x0004b9, 0x0004b8, 0x0004b7, 0x0004b6, 0x0004b5, 0x0004b4, 0x0004b3, 0x0004b2, 0x0004b1,
    0x0004b0, 0x0004af, 0x0004ae, 0x0004ad, 0x0004ac, 0x0004ab, 0x0004aa, 0x0004a9, 0x0004a8,
    0x0004a7, 0x0004a6, 0x0004a5, 0x0004a4, 0x0004a3, 0x0004a2, 0x0004a1, 0x0004a0, 0x00049f,
    0x00049e, 0x00049d, 0x00049c, 0x00049b, 0x00049a, 0x000499, 0x000498, 0x000497, 0x000496,
    0x000495, 0x000494, 0x000493, 0x000492, 0x000491, 0x000490, 0x00048f, 0x00048e, 0x00048d,
    0x00048c, 0x00048b, 0x00048a, 0x000489, 0x000488, 0x000487, 0x000486, 0x000485, 0x000484,
    0x000483, 0x000482, 0x000481, 0x000480, 0x0002bf, 0x0002be, 0x0002bd, 0x0002bc, 0x0002bb,
    0x0002ba, 0x0002b9, 0x0002b8,
];

/// Decode `huff_decode_diff(DRC_HCB, drc_gain_code)` per §4.3.10.8.3
/// and §4.3.13.6.2: read MSB-first bits until the accumulated codeword
/// matches an entry of the same length, then return `index - cb_off`.
///
/// Maximum codeword width is 12 bits (per Table A.62 inspection); we cap
/// the loop at 16 defensively.
pub fn drc_huff_decode_diff(br: &mut BitReader<'_>) -> Result<i32> {
    debug_assert_eq!(DRC_HCB_LEN.len(), DRC_HCB_CW.len());
    let mut code: u32 = 0;
    let mut width: u8 = 0;
    while width < 16 {
        let b = br.read_u32(1)?;
        code = (code << 1) | b;
        width += 1;
        for (i, &l) in DRC_HCB_LEN.iter().enumerate() {
            if l == width && DRC_HCB_CW[i] == code {
                return Ok(i as i32 - DRC_HCB_OFFSET);
            }
        }
    }
    Err(Error::invalid("ac4: no matching DRC_HCB codeword"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[test]
    fn drc_hcb_lengths_and_codewords_have_same_count() {
        // Table A.62 declares codebook_length == 255.
        assert_eq!(DRC_HCB_LEN.len(), 255);
        assert_eq!(DRC_HCB_CW.len(), 255);
    }

    /// Kraft inequality: for a complete binary prefix code the sum of
    /// 2^-len_i over all symbols equals exactly 1. A failure here means
    /// the codebook was transcribed incorrectly (or is not a complete
    /// prefix code, which would be a bug in the spec).
    #[test]
    fn drc_hcb_kraft_sum_equals_one() {
        let mut sum_num: u128 = 0;
        // Use 64-bit denominator headroom: max length is 12 in this
        // codebook so 2^(64-12) is plenty of dynamic range.
        let denom: u128 = 1u128 << 32;
        for &l in DRC_HCB_LEN {
            sum_num += denom >> l as u128;
        }
        assert_eq!(sum_num, denom, "DRC_HCB Kraft sum != 1");
    }

    /// Verify every (length, codeword) pair is within bounds: codeword
    /// must fit in `length` bits.
    #[test]
    fn drc_hcb_codewords_fit_in_declared_length() {
        for (i, (&l, &c)) in DRC_HCB_LEN.iter().zip(DRC_HCB_CW.iter()).enumerate() {
            assert!(l > 0 && l <= 16, "DRC_HCB[{i}] length out of range");
            let max = (1u64 << l as u64) - 1;
            assert!(
                c as u64 <= max,
                "DRC_HCB[{i}] codeword 0x{c:x} exceeds {l}-bit limit"
            );
        }
    }

    /// Verify the prefix-code property pair-by-pair: no two codewords
    /// of equal length collide, and no codeword is a prefix of another
    /// (longer) codeword.
    #[test]
    fn drc_hcb_is_prefix_code() {
        for i in 0..DRC_HCB_LEN.len() {
            let li = DRC_HCB_LEN[i];
            let ci = DRC_HCB_CW[i];
            for j in (i + 1)..DRC_HCB_LEN.len() {
                let lj = DRC_HCB_LEN[j];
                let cj = DRC_HCB_CW[j];
                if li == lj {
                    assert_ne!(
                        ci, cj,
                        "DRC_HCB collision at {i},{j}: same length {li} cw 0x{ci:x}"
                    );
                } else {
                    let (short_len, short_cw, long_len, long_cw) = if li < lj {
                        (li, ci, lj, cj)
                    } else {
                        (lj, cj, li, ci)
                    };
                    let prefix = long_cw >> (long_len - short_len);
                    assert_ne!(
                        prefix, short_cw,
                        "DRC_HCB prefix conflict at {i},{j}: \
                         {short_len}-bit cw 0x{short_cw:x} prefixes \
                         {long_len}-bit cw 0x{long_cw:x}"
                    );
                }
            }
        }
    }

    #[test]
    fn drc_hcb_decode_zero_diff() {
        // Index 127 has the shortest codeword (len=2, cw=0x2 == "10")
        // and represents the diff = 0 (delta against the running gain).
        assert_eq!(DRC_HCB_LEN[127], 2);
        assert_eq!(DRC_HCB_CW[127], 0x2);
        let mut bw = BitWriter::new();
        bw.write_u32(DRC_HCB_CW[127], DRC_HCB_LEN[127] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), 0);
    }

    #[test]
    fn drc_hcb_decode_plus_one_diff() {
        // Index 128 -> diff = +1. Per the table that's len=3, cw=0x6
        // (binary "110").
        assert_eq!(DRC_HCB_LEN[128], 3);
        assert_eq!(DRC_HCB_CW[128], 0x6);
        let mut bw = BitWriter::new();
        bw.write_u32(DRC_HCB_CW[128], DRC_HCB_LEN[128] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), 1);
    }

    #[test]
    fn drc_hcb_decode_minus_one_diff() {
        // Index 126 -> diff = -1. Per the table len=3, cw=0x3 ("011").
        assert_eq!(DRC_HCB_LEN[126], 3);
        assert_eq!(DRC_HCB_CW[126], 0x3);
        let mut bw = BitWriter::new();
        bw.write_u32(DRC_HCB_CW[126], DRC_HCB_LEN[126] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), -1);
    }

    #[test]
    fn drc_hcb_decode_extreme_negative() {
        // Index 0 -> diff = -127 (cb_off = 127 minus index 0).
        // It's a 12-bit codeword 0xe12 ("111000010010").
        assert_eq!(DRC_HCB_LEN[0], 12);
        assert_eq!(DRC_HCB_CW[0], 0xe12);
        let mut bw = BitWriter::new();
        bw.write_u32(DRC_HCB_CW[0], DRC_HCB_LEN[0] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), -127);
    }

    #[test]
    fn drc_hcb_decode_extreme_positive() {
        // Index 254 -> diff = +127 (last entry, cw=0x2b8 len=12).
        assert_eq!(DRC_HCB_LEN[254], 12);
        assert_eq!(DRC_HCB_CW[254], 0x2b8);
        let mut bw = BitWriter::new();
        bw.write_u32(DRC_HCB_CW[254], DRC_HCB_LEN[254] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), 127);
    }

    #[test]
    fn drc_hcb_decode_back_to_back() {
        // Two codewords concatenated in the bitstream: diff +1 then
        // diff 0. Decoder should consume exactly the right number of
        // bits per call.
        let mut bw = BitWriter::new();
        bw.write_u32(DRC_HCB_CW[128], DRC_HCB_LEN[128] as u32);
        bw.write_u32(DRC_HCB_CW[127], DRC_HCB_LEN[127] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), 1);
        assert_eq!(drc_huff_decode_diff(&mut br).unwrap(), 0);
    }
}
