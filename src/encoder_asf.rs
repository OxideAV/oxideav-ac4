//! Forward ASF entropy coding for the AC-4 IMS encoder (round 48).
//!
//! Per ETSI TS 103 190-1 §5.1 (Pseudocodes 17-19) and Annex A.0 the ASF
//! audio body for a long-frame, single-window-group, mono SIMPLE channel
//! consists of (in order):
//!
//!   1. `asf_transform_info()` — `b_long_frame = 1` for frame_len_base
//!      ≥ 1536 (zero further bits at the long-frame branch).
//!   2. `asf_psy_info(0, 0)` — `max_sfb[0]` in `n_msfb_bits` bits, no
//!      window-grouping bits when `b_long_frame`.
//!   3. `asf_section_data()` — outer `for (g)` loop over a single group
//!      with `transf_length_idx = 0` (long frame) → `n_sect_bits = 3`,
//!      `sect_esc_val = 7`. Each section: `sect_cb` (4 bits) +
//!      length-increment chain.
//!   4. `asf_spectral_data()` — Huffman-coded quantised spectrum across
//!      every section.
//!   5. `asf_scalefac_data()` — `reference_scale_factor` (8 bits) + per-band
//!      DPCM deltas in HCB_SCALEFAC.
//!   6. `asf_snf_data()` — `b_snf_data_exists` (1 bit). We always emit 0
//!      to keep things simple — empty bands just decode as silence.
//!
//! Round 48 ships the simplest viable closed-form encoder:
//!   * One section spanning all of `0..max_sfb` with codebook ID 5
//!     (HCB5, dim=2, signed, q-range -4..=+4).
//!   * Per-band scalefactor selected by greedy nearest-power-of-two
//!     scale chosen to keep the largest |coeff|/sf_gain power within
//!     HCB5's 4-magnitude bound after the spec's q = sign(c)*|c/sf|^(3/4)
//!     mapping.
//!   * `reference_scale_factor` taken from sfb 0's chosen value; the
//!     remaining bands DPCM-encode their delta with HCB_SCALEFAC.
//!
//! Future work (deferred): codebook-selection optimiser (try HCB1..11
//! per section, pick min-bits), section-boundary optimiser (split bands
//! by codebook), spectral noise fill.

use oxideav_core::bits::BitWriter;

use crate::asf_data::AsfSections;
use crate::huffman::{asf_hcb, Hcb, CB_DIM, HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN, UNSIGNED_CB};

/// Quantise a single MDCT coefficient given an integer scalefactor.
/// Spec (§5.1, Pseudocode 18 inverse): `coeff = sign(q) * |q|^(4/3) * 2^((sf-100)/4)`.
/// Inverse-quantise → `q = round(sign(c) * (|c| / 2^((sf-100)/4))^(3/4))`.
pub fn quantise_coeff(coeff: f32, sf: i32) -> i32 {
    let sf_gain = 2.0_f32.powf((sf as f32 - 100.0) * 0.25);
    if sf_gain == 0.0 || !sf_gain.is_finite() {
        return 0;
    }
    let mag = (coeff.abs() / sf_gain).powf(0.75);
    let q = mag.round() as i32;
    if coeff < 0.0 {
        -q
    } else {
        q
    }
}

/// Pick the smallest scalefactor (most-precision) that keeps every bin's
/// quantised magnitude within `q_max` for one scale-factor band.
///
/// Walks the spec's scalefactor range 0..=255 — but that's 256 trials per
/// band per frame, well within budget. Returns the chosen scalefactor and
/// the resulting quantised vector for the band.
///
/// `coeffs` is the per-bin slice covering the band (sfb_offset[sfb] ..
/// sfb_offset[sfb+1]). `q_max` is the codebook's magnitude bound (HCB5 →
/// 4, HCB1/2 → 1, etc).
pub fn pick_scalefactor_for_band(coeffs: &[f32], q_max: u32) -> (i32, Vec<i32>) {
    // Empty band → silent, scalefactor doesn't matter; pick neutral 100.
    if coeffs.is_empty() {
        return (100, Vec::new());
    }
    let max_abs = coeffs.iter().fold(0.0_f32, |a, &c| a.max(c.abs()));
    if max_abs <= 1e-12 {
        // Silent band: zero quants, scalefactor at the spec midpoint.
        return (100, vec![0i32; coeffs.len()]);
    }
    // Try sf in 0..=255. We want to pick the smallest sf (largest sf_gain
    // is at sf=255; smallest at sf=0) such that quant magnitudes fit
    // into q_max. Larger sf → larger sf_gain → smaller quant magnitude.
    // So we iterate sf from low to high and pick the first that fits.
    // The mapping: q = round(|c/sf_gain|^(3/4)). Solve for sf:
    //   |c|/sf_gain = q^(4/3) → sf_gain = |c| / q^(4/3).
    // For q_max: sf_gain_min = max_abs / q_max^(4/3), so
    //   sf_min = 100 + 4 * log2(sf_gain_min).
    let q_max_43 = (q_max as f32).powf(4.0 / 3.0);
    let sf_gain_min = max_abs / q_max_43;
    // sf = 100 + 4 * log2(sf_gain_min) (rounded up).
    let sf_f = 100.0 + 4.0 * sf_gain_min.log2();
    let sf = sf_f.ceil() as i32;
    let sf = sf.clamp(0, 255);
    // Compute the quantised vector at this scalefactor.
    let mut q = vec![0i32; coeffs.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        let qi = quantise_coeff(c, sf);
        // Clamp to ±q_max (just in case ceil rounding produced one too small).
        q[i] = qi.clamp(-(q_max as i32), q_max as i32);
    }
    (sf, q)
}

/// Encode a quantised symbol vector for the given codebook into the bit
/// writer. For dim=2 codebooks the symbol covers 2 quantised lines; for
/// dim=4 it covers 4. The encoder reverses the spec's `split_qspec()`
/// (Pseudocode 19) into a single `cb_idx` and emits `(cw, len)` from the
/// codebook.
pub fn encode_pair(bw: &mut BitWriter, hcb: &Hcb, q: &[i32]) {
    let dim = hcb.dim as usize;
    debug_assert!(q.len() >= dim);
    // Build cb_idx by reversing split_qspec. For dim=2:
    //   q1 = (idx / cb_mod) - cb_off
    //   q2 = idx - (q1 + cb_off) * cb_mod
    // Reversing: idx = (q1 + cb_off) * cb_mod + (q2 + cb_off).
    // Same shape for dim=4 with the cb_mod{,2,3} cascade.
    let cb_idx = if dim == 4 {
        let i1 = (q[0] + hcb.cb_off) as u32 * hcb.cb_mod3;
        let i2 = (q[1] + hcb.cb_off) as u32 * hcb.cb_mod2;
        let i3 = (q[2] + hcb.cb_off) as u32 * hcb.cb_mod;
        let i4 = (q[3] + hcb.cb_off) as u32;
        i1 + i2 + i3 + i4
    } else {
        let i1 = (q[0] + hcb.cb_off) as u32 * hcb.cb_mod;
        let i2 = (q[1] + hcb.cb_off) as u32;
        i1 + i2
    } as usize;
    debug_assert!(
        cb_idx < hcb.cw.len(),
        "encode_pair: cb_idx {cb_idx} out of range for cb (len={})",
        hcb.cw.len()
    );
    bw.write_u32(hcb.cw[cb_idx], hcb.len[cb_idx] as u32);
    // For unsigned codebooks, append a sign bit per non-zero quant.
    if hcb.unsigned {
        for &qi in &q[..dim] {
            if qi != 0 {
                bw.write_u32(if qi < 0 { 1 } else { 0 }, 1);
            }
        }
    }
}

/// Encode a section-length increment per Pseudocode 17 (§4.3.5.4):
/// emit `floor((sect_len-1) / esc)` escape codewords followed by one
/// non-escape `(sect_len-1) % esc` value.
pub fn write_sect_len_incr(bw: &mut BitWriter, sect_len: u32, n_sect_bits: u32, esc: u32) {
    let base = sect_len.saturating_sub(1);
    let k = base / esc;
    let incr = base % esc;
    for _ in 0..k {
        bw.write_u32(esc, n_sect_bits);
    }
    bw.write_u32(incr, n_sect_bits);
}

/// Build an [`AsfSections`] for a single section spanning `0..max_sfb`
/// with codebook id `cb`.
pub fn single_section(max_sfb: u32, cb: u8) -> AsfSections {
    AsfSections {
        sect_cb: vec![cb],
        sect_start: vec![0],
        sect_end: vec![max_sfb as u16],
        sfb_cb: vec![cb; max_sfb as usize],
        num_sec: 1,
        num_sec_lsf: 1,
    }
}

/// Encode the per-band scalefactor data per Table 41 / Pseudocode 17:
/// `reference_scale_factor` (8 bits) followed by per-band DPCM deltas
/// in HCB_SCALEFAC. Bands with `cb == 0` or `max_quant_idx[sfb] == 0`
/// don't emit a delta (they pin to whatever the running scale_factor is
/// without resetting `first_scf_found`).
pub fn write_scalefac_data(
    bw: &mut BitWriter,
    sf_per_band: &[i32],
    sfb_cb: &[u8],
    max_quant_idx: &[u32],
    max_sfb: u32,
) {
    // reference_scale_factor: take sfb 0's value (always emitted as the
    // anchor when present); fall back to 100 if no band is active.
    let mut reference: i32 = 100;
    let mut found = false;
    for sfb in 0..max_sfb as usize {
        let cb = sfb_cb[sfb];
        if cb != 0 && max_quant_idx[sfb] > 0 {
            reference = sf_per_band[sfb].clamp(0, 255);
            found = true;
            break;
        }
    }
    bw.write_u32(reference as u32, 8);
    let mut running = reference;
    let mut first = false;
    for sfb in 0..max_sfb as usize {
        let cb = sfb_cb[sfb];
        if cb == 0 || max_quant_idx[sfb] == 0 {
            continue;
        }
        if !first {
            // Anchor — no codeword.
            first = true;
            // Found defends against a degenerate case where reference
            // wasn't actually pinned to this sfb's value.
            if !found {
                running = sf_per_band[sfb].clamp(0, 255);
            }
            continue;
        }
        let delta = sf_per_band[sfb].clamp(0, 255) - running;
        // delta + 60 → HCB_SCALEFAC index. Clamp to the legal range.
        let idx = (delta + 60).clamp(0, 124) as usize;
        bw.write_u32(HCB_SCALEFAC_CW[idx], HCB_SCALEFAC_LEN[idx] as u32);
        running += idx as i32 - 60;
    }
}

/// Encode the spectral coefficient body for a single section spanning
/// `0..max_sfb`, all bins, using codebook `cb`.
pub fn write_spectral_data_single_section(
    bw: &mut BitWriter,
    qspec: &[i32],
    sfb_offset: &[u16],
    max_sfb: u32,
    cb: u32,
) {
    let hcb = asf_hcb(cb).expect("encode_asf: invalid codebook id");
    let dim = CB_DIM[cb as usize] as usize;
    let _unsig = UNSIGNED_CB[cb as usize];
    let end_bin = sfb_offset[max_sfb as usize] as usize;
    debug_assert!(qspec.len() >= end_bin);
    let mut k = 0usize;
    while k + dim <= end_bin {
        encode_pair(bw, hcb, &qspec[k..k + dim]);
        k += dim;
    }
}

/// Build the full mono SIMPLE/ASF substream body for the long-frame
/// single-window-group case at the configured transform length and
/// max_sfb. `coeffs` is the windowed forward-MDCT spectrum (length ≥
/// `sfb_offset[max_sfb]`).
///
/// Returns the substream bytes (audio_size header + audio_data + zero-
/// padding) sized to `pad_target_bytes`.
pub fn build_mono_simple_asf_body_from_pcm_spectrum(
    transform_length: u32,
    max_sfb: u32,
    coeffs: &[f32],
    pad_target_bytes: usize,
) -> Vec<u8> {
    let cb: u8 = 5; // HCB5 — dim=2, signed, q-range -4..=+4
    let q_max = 4u32;

    // 1. Per-band scalefactor + quantisation.
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let end_bin = sfbo[max_sfb as usize] as usize;
    let mut qspec = vec![0i32; end_bin];
    let mut sf_per_band = vec![100i32; max_sfb as usize];
    let mut max_quant_idx = vec![0u32; max_sfb as usize];
    for sfb in 0..max_sfb as usize {
        let a = sfbo[sfb] as usize;
        let b = sfbo[sfb + 1] as usize;
        let band = &coeffs[a..b.min(coeffs.len())];
        let (sf, q) = pick_scalefactor_for_band(band, q_max);
        sf_per_band[sfb] = sf;
        let mut max_q: u32 = 0;
        for (i, &qi) in q.iter().enumerate() {
            qspec[a + i] = qi;
            max_q = max_q.max(qi.unsigned_abs());
        }
        max_quant_idx[sfb] = max_q;
    }

    // 2. Bit-stream emission.
    let mut bw = BitWriter::new();
    // audio_size_value (15 b) + b_more_bits (1 b). We declare the size
    // as the pad target — the decoder reads exactly that many bytes for
    // the substream body.
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();
    // mono_codec_mode = SIMPLE (0), spec_frontend = ASF (0).
    bw.write_u32(0, 1);
    bw.write_u32(0, 1);
    // asf_transform_info: b_long_frame = 1.
    bw.write_bit(true);
    // asf_psy_info(0, 0): max_sfb[0] in n_msfb_bits bits.
    let (n_msfb_bits, _, _) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    bw.write_u32(max_sfb, n_msfb_bits);

    // asf_section_data: one section spanning 0..max_sfb with cb=5.
    bw.write_u32(cb as u32, 4);
    write_sect_len_incr(&mut bw, max_sfb, 3, 7);

    // asf_spectral_data.
    write_spectral_data_single_section(&mut bw, &qspec, sfbo, max_sfb, cb as u32);

    // asf_scalefac_data.
    let sections = single_section(max_sfb, cb);
    write_scalefac_data(
        &mut bw,
        &sf_per_band,
        &sections.sfb_cb,
        &max_quant_idx,
        max_sfb,
    );

    // asf_snf_data: b_snf_data_exists = 0.
    bw.write_u32(0, 1);

    bw.align_to_byte();
    // Pad with zeros to the announced size.
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asf_data::{
        dequantise_and_scale, parse_asf_scalefac_data, parse_asf_section_data,
        parse_asf_spectral_data,
    };
    use crate::huffman::HCB5;
    use oxideav_core::bits::BitReader;

    #[test]
    fn quantise_then_dequantise_roundtrips_small_value() {
        // sf = 120 → sf_gain = 32. Coefficient 32 should round to q = 1
        // (since 1^(4/3) * 32 = 32). Coefficient 64 ≈ 32 * 2^(3/4) →
        // q ≈ round(2^(3/4)) = round(1.68) = 2.
        let q = quantise_coeff(32.0, 120);
        assert_eq!(q, 1);
        // Coefficient 0 → q = 0.
        let q0 = quantise_coeff(0.0, 120);
        assert_eq!(q0, 0);
        // Negative coefficient preserves sign.
        let qn = quantise_coeff(-32.0, 120);
        assert_eq!(qn, -1);
    }

    #[test]
    fn pick_scalefactor_keeps_quantised_within_q_max() {
        // Strong band with peak 100; q_max = 4.
        let band = vec![10.0_f32, -50.0, 0.0, 100.0];
        let (_sf, q) = pick_scalefactor_for_band(&band, 4);
        for &qi in &q {
            assert!(qi.abs() <= 4, "q exceeded q_max=4: got {qi}");
        }
    }

    #[test]
    fn encode_pair_dim2_signed_roundtrips() {
        // HCB5 is dim=2, signed (cb_off=4). Pair (q1=+1, q2=0) →
        // cb_idx = (1+4)*9 + (0+4) = 45+4 = 49.
        let mut bw = BitWriter::new();
        let q = [1i32, 0i32];
        encode_pair(&mut bw, &HCB5, &q);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cb_idx = crate::huffman::huff_decode(&mut br, HCB5.len, HCB5.cw).unwrap();
        let mut out = [0i32; 4];
        crate::huffman::split_qspec(&HCB5, cb_idx, &mut out);
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn encode_pair_dim2_unsigned_emits_sign_bit() {
        // HCB7 is dim=2, unsigned (cb_off=0, cb_mod=8). Pair (q1=2, q2=0):
        // cb_idx = 2*8 + 0 = 16.
        let mut bw = BitWriter::new();
        let q = [2i32, 0i32];
        let hcb = asf_hcb(7).unwrap();
        encode_pair(&mut bw, hcb, &q);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cb_idx = crate::huffman::huff_decode(&mut br, hcb.len, hcb.cw).unwrap();
        // Sign bit follows the codeword for non-zero q1.
        let sign = br.read_u32(1).unwrap();
        let mut out = [0i32; 4];
        crate::huffman::split_qspec(hcb, cb_idx, &mut out);
        let q1 = if sign == 1 { -out[0] } else { out[0] };
        assert_eq!(q1, 2);
    }

    /// End-to-end: build a small spectrum, run through
    /// build_mono_simple_asf_body_from_pcm_spectrum, then walk the
    /// produced bytes through the regular ASF parser. The decoded
    /// quantised spectrum should match the encoder's chosen quants.
    #[test]
    fn build_body_roundtrips_through_asf_parser() {
        // Tiny test: tl = 256, max_sfb = 5. Non-trivial coefficients in
        // band 1 (bin 4, 5, 6, 7).
        let tl = 256u32;
        let max_sfb = 5u32;
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let end_bin = sfbo[max_sfb as usize] as usize;
        let mut coeffs = vec![0.0_f32; end_bin];
        coeffs[4] = 32.0; // ends up in band 1 (sfbo[1]=4)
        coeffs[5] = -32.0;
        // Build the body.
        let body = build_mono_simple_asf_body_from_pcm_spectrum(tl, max_sfb, &coeffs, 80);
        assert_eq!(body.len(), 80);

        // Parse: skip header (audio_size_value + b_more_bits + align +
        // mono_codec_mode + spec_frontend + b_long_frame + max_sfb).
        let mut br = BitReader::new(&body);
        // audio_size_value (15) + b_more_bits (1) = 16 bits = 2 bytes.
        let _audio_size = br.read_u32(15).unwrap();
        let _more = br.read_bit().unwrap();
        br.align_to_byte();
        let mono_mode = br.read_u32(1).unwrap();
        assert_eq!(mono_mode, 0);
        let frontend = br.read_u32(1).unwrap();
        assert_eq!(frontend, 0);
        let b_long = br.read_bit().unwrap();
        assert!(b_long);
        let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(tl).unwrap();
        let parsed_max_sfb = br.read_u32(n_msfb_bits).unwrap();
        assert_eq!(parsed_max_sfb, max_sfb);
        let sections = parse_asf_section_data(&mut br, 0, tl, max_sfb).unwrap();
        assert_eq!(sections.num_sec, 1);
        assert_eq!(sections.sect_cb, vec![5u8]);
        let (qspec, mqi) = parse_asf_spectral_data(&mut br, &sections, sfbo, max_sfb).unwrap();
        // Band 1 must carry the injected non-zero values.
        assert!(mqi[1] > 0, "expected non-zero max_quant_idx in band 1");
        let sf_gain = parse_asf_scalefac_data(&mut br, &sections, &mqi, max_sfb, tl).unwrap();
        let scaled = dequantise_and_scale(&qspec, &sf_gain, sfbo, max_sfb);
        // Reconstructed coefficient at bin 4 should be close to +32 in
        // sign and order of magnitude (we did lossy quantisation).
        assert!(
            scaled[4] > 0.0,
            "expected positive reconstruction at bin 4, got {}",
            scaled[4]
        );
        assert!(
            scaled[5] < 0.0,
            "expected negative reconstruction at bin 5, got {}",
            scaled[5]
        );
    }
}
