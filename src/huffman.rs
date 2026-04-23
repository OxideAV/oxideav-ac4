//! ASF Huffman codebooks from ETSI TS 103 190-1 Annex A.
//!
//! The codebook `_LEN[]` and `_CW[]` arrays are transcribed verbatim
//! from the normative accompaniment file `ts_10319001v010401p0.zip`
//! (`ts_103190_tables.c`) that ETSI publishes alongside the spec — per
//! Annex A.0 of the spec, the tables "are available in the accompanying
//! file". They are normative numeric constants (uncopyrightable facts).
//!
//! Each codebook `n` has entries `i = 0 .. codebook_length-1`. The
//! encoder maps a source symbol to `i`; the decoder reads
//! `ASF_HCB_n_LEN[i]` bits from the stream as MSB-first codeword and
//! compares against `ASF_HCB_n_CW[i]`. The first match wins.
//!
//! The 11 spectral codebooks (ASF_HCB_1..ASF_HCB_11) have
//! codebook-specific dimension / modulus / offset metadata used to
//! split a decoded symbol back into per-line quantised values (§5.1.2,
//! Pseudocode 19). See [`Hcb`].
//!
//! The decoder here is the simple "read one bit, try one codebook entry"
//! reference implementation. Sufficient for correctness; not optimised.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// Metadata for one spectral Huffman codebook — dimensions, moduli,
/// offset and sign-flag per Tables A.2..A.15.
pub struct Hcb {
    pub len: &'static [u8],
    pub cw: &'static [u32],
    pub dim: u8,
    pub cb_mod: u32,
    pub cb_mod2: u32,
    pub cb_mod3: u32,
    pub cb_off: i32,
    pub unsigned: bool,
}

/// Decode the shortest matching codeword from `br` against `(lens, cws)`.
///
/// Reference algorithm (Pseudocode 19 of §5.1.2.2): accumulate bits
/// MSB-first, and after each bit check whether any entry of length
/// equal to the accumulator width matches. Returns the index `i` of
/// the first matching entry.
pub fn huff_decode(br: &mut BitReader<'_>, lens: &[u8], cws: &[u32]) -> Result<u32> {
    debug_assert_eq!(lens.len(), cws.len());
    let mut code: u32 = 0;
    let mut width: u8 = 0;
    // Max Huffman length across all AC-4 ASF tables is 17 (SCALEFAC);
    // we cap at 32 defensively.
    while width < 32 {
        let b = br.read_u32(1)?;
        code = (code << 1) | b;
        width += 1;
        for (i, &l) in lens.iter().enumerate() {
            if l == width && cws[i] == code {
                return Ok(i as u32);
            }
        }
    }
    Err(Error::invalid("ac4: no matching Huffman codeword"))
}

include!("huffman_tables.rs");

/// ASF spectral codebooks (1..=11). Index 0 is a sentinel "all-zero"
/// band and carries no bitstream data.
pub fn asf_hcb(cb_idx: u32) -> Option<&'static Hcb> {
    match cb_idx {
        1 => Some(&HCB1),
        2 => Some(&HCB2),
        3 => Some(&HCB3),
        4 => Some(&HCB4),
        5 => Some(&HCB5),
        6 => Some(&HCB6),
        7 => Some(&HCB7),
        8 => Some(&HCB8),
        9 => Some(&HCB9),
        10 => Some(&HCB10),
        11 => Some(&HCB11),
        _ => None,
    }
}

/// Split one decoded spectral-codebook symbol into its N quantised
/// lines per Pseudocode 19 in §5.1.2.2. Writes 2 or 4 outputs depending
/// on `dim`.
pub fn split_qspec(hcb: &Hcb, mut cb_idx: u32, out: &mut [i32]) {
    if hcb.dim == 4 {
        debug_assert!(out.len() >= 4);
        let q1 = (cb_idx / hcb.cb_mod3) as i32 - hcb.cb_off;
        cb_idx -= (q1 + hcb.cb_off) as u32 * hcb.cb_mod3;
        let q2 = (cb_idx / hcb.cb_mod2) as i32 - hcb.cb_off;
        cb_idx -= (q2 + hcb.cb_off) as u32 * hcb.cb_mod2;
        let q3 = (cb_idx / hcb.cb_mod) as i32 - hcb.cb_off;
        cb_idx -= (q3 + hcb.cb_off) as u32 * hcb.cb_mod;
        let q4 = cb_idx as i32 - hcb.cb_off;
        out[0] = q1;
        out[1] = q2;
        out[2] = q3;
        out[3] = q4;
    } else {
        debug_assert!(out.len() >= 2);
        let q1 = (cb_idx / hcb.cb_mod) as i32 - hcb.cb_off;
        cb_idx -= (q1 + hcb.cb_off) as u32 * hcb.cb_mod;
        let q2 = cb_idx as i32 - hcb.cb_off;
        out[0] = q1;
        out[1] = q2;
    }
}

/// Extension-code decoder for codebook 11, Pseudocode 20 in §5.1.2.2:
/// unary prefix giving N_ext, then (N_ext + 4) magnitude bits ->
/// 2^(N_ext+4) + ext_val.
pub fn ext_decode(br: &mut BitReader<'_>) -> Result<u32> {
    let mut n_ext: u32 = 0;
    loop {
        let b = br.read_u32(1)?;
        if b == 0 {
            break;
        }
        n_ext += 1;
    }
    let bits = n_ext + 4;
    let ext_val = br.read_u32(bits)?;
    Ok((1u32 << (n_ext + 4)) + ext_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[test]
    fn scalefac_roundtrip_index_60() {
        // The DC element of ASF_HCB_SCALEFAC is at index 60 with a
        // 1-bit codeword "0" (len=1, cw=0x1). Writing one '1' bit
        // should decode to 60.
        let mut bw = BitWriter::new();
        bw.write_u32(HCB_SCALEFAC_CW[60], HCB_SCALEFAC_LEN[60] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let idx = huff_decode(&mut br, HCB_SCALEFAC_LEN, HCB_SCALEFAC_CW).unwrap();
        assert_eq!(idx, 60);
    }

    #[test]
    fn snf_short_codeword() {
        // ASF_HCB_SNF index 13 has len=3, cw=0x4. Decode should
        // return 13.
        let mut bw = BitWriter::new();
        bw.write_u32(HCB_SNF_CW[13], HCB_SNF_LEN[13] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let idx = huff_decode(&mut br, HCB_SNF_LEN, HCB_SNF_CW).unwrap();
        assert_eq!(idx, 13);
    }

    #[test]
    fn hcb1_known_entry() {
        // HCB1 index 40 has the shortest codeword (len=1, cw=1).
        let mut bw = BitWriter::new();
        bw.write_u32(HCB1.cw[40], HCB1.len[40] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let idx = huff_decode(&mut br, HCB1.len, HCB1.cw).unwrap();
        assert_eq!(idx, 40);
    }

    #[test]
    fn split_qspec_dim2_unsigned() {
        // HCB5 is dim=2, cb_mod=9, cb_off=4. An index of 40 (center)
        // should map to (0, 0).
        let mut out = [0i32; 4];
        split_qspec(&HCB5, 40, &mut out);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn split_qspec_dim4_signed() {
        // HCB1 has dim=4, cb_mod=3, cb_off=1; index 40 is the centre
        // (0,0,0,0).
        let mut out = [0i32; 4];
        split_qspec(&HCB1, 40, &mut out);
        assert_eq!(out, [0, 0, 0, 0]);
    }

    #[test]
    fn ext_decode_n0_v0() {
        // n_ext = 0 -> unary "0" (one zero bit), then 4 value bits = 0.
        // Output should be 2^4 + 0 = 16.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1); // unary terminator
        bw.write_u32(0, 4); // 4 value bits
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(ext_decode(&mut br).unwrap(), 16);
    }

    #[test]
    fn ext_decode_n1_v5() {
        // n_ext=1: unary "1,0", then 5 value bits = 5.
        // Output should be 2^5 + 5 = 37.
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(5, 5);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(ext_decode(&mut br).unwrap(), 37);
    }
}
