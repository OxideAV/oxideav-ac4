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

/// Binary decode tree of one `(lens, cws)` codebook: `nodes[n][bit]`
/// is `0` for "no codeword continues here", a positive node index, or
/// `−(index + 1)` for the codeword `index` ending on that bit.
struct HuffTree {
    nodes: Vec<[i32; 2]>,
}

impl HuffTree {
    fn build(lens: &[u8], cws: &[u32]) -> Self {
        let mut nodes: Vec<[i32; 2]> = vec![[0, 0]];
        for (i, (&l, &cw)) in lens.iter().zip(cws).enumerate() {
            if l == 0 || l > 32 {
                continue;
            }
            let mut n = 0usize;
            for depth in (0..l).rev() {
                let bit = ((cw >> depth) & 1) as usize;
                let slot = nodes[n][bit];
                if depth == 0 {
                    // The shortest match wins, and among equal
                    // codewords the first entry: a leaf already here
                    // stays; a longer codeword routed through here
                    // gives way.
                    if slot >= 0 {
                        nodes[n][bit] = -(i as i32 + 1);
                    }
                } else if slot < 0 {
                    // A shorter codeword ends on this prefix.
                    break;
                } else if slot == 0 {
                    nodes.push([0, 0]);
                    let next = nodes.len() - 1;
                    nodes[n][bit] = next as i32;
                    n = next;
                } else {
                    n = slot as usize;
                }
            }
        }
        Self { nodes }
    }
}

thread_local! {
    /// Decode trees by codebook identity (the address of its static
    /// codeword table), built on first use.
    static TREES: std::cell::RefCell<Vec<(HuffKey, std::rc::Rc<HuffTree>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Identity of a static codebook: the addresses of its codeword and
/// length tables and their length.
type HuffKey = (usize, usize, usize);

fn tree_for(lens: &'static [u8], cws: &'static [u32]) -> std::rc::Rc<HuffTree> {
    let key: HuffKey = (cws.as_ptr() as usize, lens.as_ptr() as usize, cws.len());
    TREES.with(|t| {
        let mut t = t.borrow_mut();
        if let Some((_, tree)) = t.iter().find(|(k, _)| *k == key) {
            return tree.clone();
        }
        let tree = std::rc::Rc::new(HuffTree::build(lens, cws));
        t.push((key, tree.clone()));
        tree
    })
}

/// Decode the shortest matching codeword from `br` against `(lens, cws)`.
///
/// Reference algorithm (Pseudocode 19 of §5.1.2.2): accumulate bits
/// MSB-first until an entry of the accumulated length matches; returns
/// the index of that entry. The search walks a per-codebook binary
/// tree (one step per bit) instead of scanning the table after every
/// bit; the codebooks are static tables, so a tree is built once per
/// thread and codebook.
pub fn huff_decode(
    br: &mut BitReader<'_>,
    lens: &'static [u8],
    cws: &'static [u32],
) -> Result<u32> {
    debug_assert_eq!(lens.len(), cws.len());
    let tree = tree_for(lens, cws);
    let mut n = 0usize;
    // Max Huffman length across all AC-4 tables is 17 (SCALEFAC); the
    // tree is at most 32 deep.
    for _ in 0..32 {
        let b = br.read_u32(1)? as usize;
        let slot = tree.nodes[n][b];
        if slot < 0 {
            return Ok((-slot - 1) as u32);
        }
        if slot == 0 {
            break;
        }
        n = slot as usize;
    }
    Err(Error::invalid("ac4: no matching Huffman codeword"))
}

/// Table-scan form of [`huff_decode`] — the reference the tree walk is
/// tested against.
#[doc(hidden)]
pub fn huff_decode_scan(br: &mut BitReader<'_>, lens: &[u8], cws: &[u32]) -> Result<u32> {
    let mut code: u32 = 0;
    let mut width: u8 = 0;
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
    // The escape prefix is a unary run; a hostile stream can run it
    // past the 32-bit reader (and past any representable spectral
    // value). Bound it so `bits` stays a valid read width and the
    // `(1 << bits) + ext_val` sum cannot overflow.
    if n_ext > 20 {
        return Err(Error::invalid("ac4: ext_decode escape run too long"));
    }
    let bits = n_ext + 4;
    let ext_val = br.read_u32(bits)?;
    Ok((1u32 << bits) + ext_val)
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

    /// All 13 ASF Huffman codebooks (SCALEFAC, SNF, HCB1..HCB11) paired up
    /// for a single sweep test rather than 13 individual tests.
    fn all_asf_tables() -> Vec<(&'static str, &'static [u8], &'static [u32])> {
        vec![
            ("ASF_HCB_SCALEFAC", HCB_SCALEFAC_LEN, HCB_SCALEFAC_CW),
            ("ASF_HCB_SNF", HCB_SNF_LEN, HCB_SNF_CW),
            ("ASF_HCB_1", HCB1_LEN, HCB1_CW),
            ("ASF_HCB_2", HCB2_LEN, HCB2_CW),
            ("ASF_HCB_3", HCB3_LEN, HCB3_CW),
            ("ASF_HCB_4", HCB4_LEN, HCB4_CW),
            ("ASF_HCB_5", HCB5_LEN, HCB5_CW),
            ("ASF_HCB_6", HCB6_LEN, HCB6_CW),
            ("ASF_HCB_7", HCB7_LEN, HCB7_CW),
            ("ASF_HCB_8", HCB8_LEN, HCB8_CW),
            ("ASF_HCB_9", HCB9_LEN, HCB9_CW),
            ("ASF_HCB_10", HCB10_LEN, HCB10_CW),
            ("ASF_HCB_11", HCB11_LEN, HCB11_CW),
        ]
    }

    /// Roundtrip the shortest entry of every ASF codebook through
    /// `huff_decode`. Pairs with `all_asf_tables_decode_last_entry` for
    /// coverage at both ends of the bit-width range.
    #[test]
    fn all_asf_tables_decode_shortest_entry() {
        for (name, lens, cws) in all_asf_tables() {
            let (sym_idx, &min_len) = lens
                .iter()
                .enumerate()
                .min_by_key(|&(_, &l)| l)
                .expect("non-empty codebook");
            let cw = cws[sym_idx];

            let mut bw = BitWriter::new();
            bw.write_u32(cw, min_len as u32);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let got = huff_decode(&mut br, lens, cws)
                .unwrap_or_else(|e| panic!("{name}: decode failed at sym_idx={sym_idx}: {e:?}"));
            assert_eq!(
                got as usize, sym_idx,
                "{name}: decoded {got}, expected {sym_idx} (cw=0x{cw:x}, len={min_len})"
            );
        }
    }

    /// Roundtrip the last entry (often a long codeword at the tail) of
    /// every ASF codebook.
    #[test]
    fn all_asf_tables_decode_last_entry() {
        for (name, lens, cws) in all_asf_tables() {
            let last = lens.len() - 1;
            let l = lens[last];
            let cw = cws[last];

            let mut bw = BitWriter::new();
            bw.write_u32(cw, l as u32);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let got = huff_decode(&mut br, lens, cws)
                .unwrap_or_else(|e| panic!("{name}: decode failed for last entry: {e:?}"));
            assert_eq!(
                got as usize, last,
                "{name}: decoded {got}, expected {last} (cw=0x{cw:x}, len={l})"
            );
        }
    }

    #[test]
    fn tree_decode_matches_the_table_scan_on_every_asf_codebook() {
        let mut books: Vec<(&'static [u8], &'static [u32])> = vec![
            (HCB_SCALEFAC_LEN, HCB_SCALEFAC_CW),
            (HCB_SNF_LEN, HCB_SNF_CW),
        ];
        for cb in 1..=11 {
            let h = asf_hcb(cb).unwrap();
            books.push((h.len, h.cw));
        }
        for (lens, cws) in books {
            // Every codeword decodes to its own index …
            for (i, (&l, &cw)) in lens.iter().zip(cws).enumerate() {
                let mut bw = BitWriter::new();
                bw.write_u32(cw, l as u32);
                bw.write_u32(0, 24);
                bw.align_to_byte();
                let bytes = bw.finish();
                let a = huff_decode(&mut BitReader::new(&bytes), lens, cws).unwrap();
                let b = huff_decode_scan(&mut BitReader::new(&bytes), lens, cws).unwrap();
                assert_eq!(a, b);
                assert_eq!(a as usize, i);
            }
            // … and arbitrary bit patterns agree, errors included.
            let mut seed = 0x2545_F491u32;
            for _ in 0..2000 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let bytes = seed.to_be_bytes();
                let mut ra = BitReader::new(&bytes);
                let mut rb = BitReader::new(&bytes);
                let a = huff_decode(&mut ra, lens, cws).ok();
                let b = huff_decode_scan(&mut rb, lens, cws).ok();
                assert_eq!(a, b, "pattern {seed:08x}");
            }
        }
    }
}
