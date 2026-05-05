//! ASF data-payload parsers — the Huffman-driven body of
//! `asf_section_data()`, `asf_spectral_data()`, `asf_scalefac_data()`
//! and `asf_snf_data()` (ETSI TS 103 190-1 §4.2.8.3 / §4.2.8.4 /
//! §4.2.8.5 / §4.2.8.6 and §5.1.2 / §5.1.3).
//!
//! These consume the Huffman-coded section-codebook, spectral,
//! scalefactor and noise-fill data and produce the per-band state the
//! decoder needs to drive dequantisation and MDCT synthesis. The
//! Huffman tables live in [`crate::huffman`]; the sfb offsets live in
//! [`crate::sfb_offset`].
//!
//! Both the long-frame (`num_window_groups == 1`) and the short-frame
//! grouped (`num_window_groups > 1`) cases are supported. The
//! grouped variants (`*_grouped`) take per-group transform-length and
//! `max_sfb` arrays and follow Tables 39-42's outer
//! `for (g = 0; g < num_window_groups; g++)` loops with a *single*
//! `reference_scale_factor` (8 bits) and *single* `b_snf_data_exists`
//! (1 bit) at the head of `asf_scalefac_data()` / `asf_snf_data()`.
//! `first_scf_found` carries across groups so the scalefactor DPCM
//! state is continuous over the whole frame.
//!
//! HSF extension (`asf_hsf_spectral_data()`, Table 42a) is still
//! pending.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::huffman::{
    asf_hcb, ext_decode, huff_decode, split_qspec, CB_DIM, HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN,
    HCB_SNF_CW, HCB_SNF_LEN, UNSIGNED_CB,
};
use crate::tables::num_sfb_48;

/// Decoded section information for one window group.
#[derive(Debug, Default, Clone)]
pub struct AsfSections {
    /// `sect_cb[s]` — Huffman codebook ID used for section s (0..=11).
    pub sect_cb: Vec<u8>,
    /// `sect_start[s]` — first sfb in section s (inclusive).
    pub sect_start: Vec<u16>,
    /// `sect_end[s]` — one past the last sfb in section s (exclusive).
    pub sect_end: Vec<u16>,
    /// `sfb_cb[sfb]` — per-scale-factor-band codebook id, filled in from
    /// the section table. Length = max_sfb.
    pub sfb_cb: Vec<u8>,
    /// `num_sec` — total number of sections.
    pub num_sec: u32,
    /// `num_sec_lsf` — number of sections whose `sect_end` fits inside
    /// `num_sfb_48(transform_length)` (the low-sample-frequency extent).
    pub num_sec_lsf: u32,
}

/// Parse `asf_section_data()` for a single window group (Table 39).
///
/// `transf_length_idx` is the 2-bit transform-length index (Table 100 /
/// 103) that drives `n_sect_bits`. `transform_length` is the resolved
/// length used for the `num_sfb_48` cap. `max_sfb` is the group's
/// per-group max scale factor band.
pub fn parse_asf_section_data(
    br: &mut BitReader<'_>,
    transf_length_idx: u32,
    transform_length: u32,
    max_sfb: u32,
) -> Result<AsfSections> {
    let (n_sect_bits, sect_esc_val) = if transf_length_idx <= 2 {
        (3u32, 7u32)
    } else {
        (5u32, 31u32)
    };
    let num_sfb = num_sfb_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: asf_section_data: unsupported transform_length"))?;

    let mut out = AsfSections {
        sfb_cb: vec![0u8; max_sfb as usize],
        ..Default::default()
    };
    let mut k: u32 = 0;
    while k < max_sfb {
        let sect_cb = br.read_u32(4)? as u8;
        // Spec pseudocode: sect_len = 1; while (sect_len_incr == esc)
        // { sect_len += esc; } sect_len += sect_len_incr;
        // We read increments until we see a non-escape value.
        let mut sect_len: u32 = 0;
        loop {
            let incr = br.read_u32(n_sect_bits)?;
            sect_len += incr;
            if incr != sect_esc_val {
                break;
            }
        }
        // A sect_len of 0 still means "length 1" per the spec's
        // sect_len = 1 initial assignment: the counter starts at 1 and
        // only accumulates escapes. So add 1 implicitly.
        sect_len += 1;
        let sect_start = k;
        let mut sect_end = k + sect_len;
        // Fix up when sect_end straddles num_sfb_48 boundary: split
        // into an LSF and HSF section.
        if sect_start < num_sfb && sect_end > num_sfb {
            out.sect_cb.push(sect_cb);
            out.sect_start.push(sect_start as u16);
            out.sect_end.push(num_sfb as u16);
            out.num_sec_lsf = out.sect_cb.len() as u32;
            // The HSF portion becomes the next section with the same
            // sect_cb.
            out.sect_cb.push(sect_cb);
            out.sect_start.push(num_sfb as u16);
            out.sect_end.push(sect_end as u16);
            for sfb in sect_start..sect_end {
                if (sfb as usize) < out.sfb_cb.len() {
                    out.sfb_cb[sfb as usize] = sect_cb;
                }
            }
            k += sect_len;
            continue;
        }
        if sect_end > max_sfb {
            sect_end = max_sfb;
        }
        out.sect_cb.push(sect_cb);
        out.sect_start.push(sect_start as u16);
        out.sect_end.push(sect_end as u16);
        for sfb in sect_start..sect_end {
            if (sfb as usize) < out.sfb_cb.len() {
                out.sfb_cb[sfb as usize] = sect_cb;
            }
        }
        k += sect_len;
    }
    out.num_sec = out.sect_cb.len() as u32;
    if out.num_sec_lsf == 0 {
        out.num_sec_lsf = out.num_sec;
    }
    Ok(out)
}

/// Parse `asf_spectral_data()` for a single window group and return the
/// vector of quantised spectral lines `quant_spec[0..end_line]`, along
/// with per-sfb `max_quant_idx[sfb]`.
///
/// `sfb_offset` maps sfb boundary -> bin index; `sections` drives which
/// codebook to use per segment.
pub fn parse_asf_spectral_data(
    br: &mut BitReader<'_>,
    sections: &AsfSections,
    sfb_offset: &[u16],
    max_sfb: u32,
) -> Result<(Vec<i32>, Vec<u32>)> {
    let end_bin = sfb_offset[max_sfb as usize] as usize;
    let mut quant_spec = vec![0i32; end_bin];
    let mut max_quant_idx = vec![0u32; max_sfb as usize];
    for i in 0..sections.num_sec_lsf as usize {
        let cb = sections.sect_cb[i] as u32;
        if cb == 0 || cb > 11 {
            continue;
        }
        let hcb =
            asf_hcb(cb).ok_or_else(|| Error::invalid("ac4: asf_spectral_data: bad codebook"))?;
        let dim = CB_DIM[cb as usize];
        let unsig = UNSIGNED_CB[cb as usize];
        let sect_start_line = sfb_offset[sections.sect_start[i] as usize] as usize;
        let sect_end_line = sfb_offset[sections.sect_end[i] as usize] as usize;
        let mut k = sect_start_line;
        let mut tmp = [0i32; 4];
        while k < sect_end_line {
            let cb_idx = huff_decode(br, hcb.len, hcb.cw)?;
            split_qspec(hcb, cb_idx, &mut tmp);
            let step = dim as usize;
            for t in 0..step {
                let mut q = tmp[t];
                if unsig && q != 0 {
                    let s = br.read_u32(1)?;
                    if s == 1 {
                        q = -q;
                    }
                }
                if cb == 11 && q.unsigned_abs() == 16 {
                    let ext = ext_decode(br)?;
                    // sign was already applied if unsigned; re-apply
                    // sign via sign of q.
                    q = if q.is_negative() {
                        -(ext as i32)
                    } else {
                        ext as i32
                    };
                }
                if k + t < quant_spec.len() {
                    quant_spec[k + t] = q;
                }
            }
            k += step;
        }
    }
    // Compute max_quant_idx per sfb.
    for sfb in 0..max_sfb as usize {
        let a = sfb_offset[sfb] as usize;
        let b = sfb_offset[sfb + 1] as usize;
        let mut m: u32 = 0;
        for &q in &quant_spec[a..b.min(quant_spec.len())] {
            m = m.max(q.unsigned_abs());
        }
        max_quant_idx[sfb] = m;
    }
    Ok((quant_spec, max_quant_idx))
}

/// Parse `asf_scalefac_data()` for a single window group. Returns a
/// per-sfb vector of scale-factor-gain values (`2^((sf - 100) / 4)`) as
/// f32. Bands with no scale factor (codebook 0 or all-zero lines) get
/// a gain of 0.0.
pub fn parse_asf_scalefac_data(
    br: &mut BitReader<'_>,
    sections: &AsfSections,
    max_quant_idx: &[u32],
    max_sfb: u32,
    transform_length: u32,
) -> Result<Vec<f32>> {
    let num_sfb_lsf =
        num_sfb_48(transform_length).ok_or_else(|| Error::invalid("ac4: scalefac: bad tl"))?;
    let reference_scale_factor = br.read_u32(8)?;
    let mut sf_gain = vec![0.0_f32; max_sfb as usize];
    let mut scale_factor: i32 = reference_scale_factor as i32;
    let mut first_scf_found = false;
    let max_sfb_eff = max_sfb.min(num_sfb_lsf);
    for sfb in 0..max_sfb_eff as usize {
        let cb = sections.sfb_cb[sfb];
        if cb == 0 || max_quant_idx[sfb] == 0 {
            continue;
        }
        if first_scf_found {
            let cw_idx = huff_decode(br, HCB_SCALEFAC_LEN, HCB_SCALEFAC_CW)?;
            // dpcm: diff = cw_idx - 60 (index offset for the codebook).
            scale_factor += cw_idx as i32 - 60;
        } else {
            first_scf_found = true;
        }
        // sf_gain[sfb] = 2^((scale_factor - 100) / 4).
        let sf = scale_factor;
        let exp = (sf as f32 - 100.0) * 0.25;
        sf_gain[sfb] = 2.0_f32.powf(exp);
    }
    Ok(sf_gain)
}

/// Parse `asf_snf_data(b_iframe)` — spectral noise fill data. Returns
/// a per-sfb vector of noise-fill gains (0 for bands without SNF).
///
/// The spec defines the gain formula in §5.1.4: band RMS energy is
/// seeded from the decoded SNF index, then random Gaussian noise
/// scaled and injected into zero bands. We just decode and surface the
/// per-band dpcm_snf indices; downstream synthesis will consume them.
pub fn parse_asf_snf_data(
    br: &mut BitReader<'_>,
    sections: &AsfSections,
    max_quant_idx: &[u32],
    max_sfb: u32,
    transform_length: u32,
) -> Result<Option<Vec<i32>>> {
    let b_snf_data_exists = br.read_bit()?;
    if !b_snf_data_exists {
        return Ok(None);
    }
    let num_sfb_lsf =
        num_sfb_48(transform_length).ok_or_else(|| Error::invalid("ac4: snf: bad tl"))?;
    let mut dpcm_snf = vec![0i32; max_sfb as usize];
    let max_sfb_eff = max_sfb.min(num_sfb_lsf);
    for sfb in 0..max_sfb_eff as usize {
        let cb = sections.sfb_cb[sfb];
        if cb == 0 || max_quant_idx[sfb] == 0 {
            let idx = huff_decode(br, HCB_SNF_LEN, HCB_SNF_CW)?;
            dpcm_snf[sfb] = idx as i32;
        }
    }
    Ok(Some(dpcm_snf))
}

/// Parse `asf_section_data()` per Table 39 with the spec's outer
/// `for (g = 0; g < num_window_groups; g++)` loop. Returns one
/// [`AsfSections`] per group.
///
/// `transf_length_idx_per_group[g]` is the 2-bit transform-length
/// index used for group g (drives `n_sect_bits`). `transform_length_per_group[g]`
/// is the resolved transform length used for the `num_sfb_48` cap.
/// `max_sfb_per_group[g]` is the per-group max_sfb from Pseudocode 5
/// (`get_max_sfb(g)`).
///
/// All three slices must have length `num_window_groups`. For the
/// equal-transform-length non-different-framing case this collapses
/// to the same value repeated.
pub fn parse_asf_section_data_grouped(
    br: &mut BitReader<'_>,
    transf_length_idx_per_group: &[u32],
    transform_length_per_group: &[u32],
    max_sfb_per_group: &[u32],
) -> Result<Vec<AsfSections>> {
    let n = transf_length_idx_per_group.len();
    if n == 0 || transform_length_per_group.len() != n || max_sfb_per_group.len() != n {
        return Err(Error::invalid(
            "ac4: asf_section_data_grouped: inconsistent per-group slice lengths",
        ));
    }
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        out.push(parse_asf_section_data(
            br,
            transf_length_idx_per_group[g],
            transform_length_per_group[g],
            max_sfb_per_group[g],
        )?);
    }
    Ok(out)
}

/// Parse `asf_spectral_data()` per Table 40 with the spec's outer
/// `for (g = 0; g < num_window_groups; g++)` loop. Returns
/// `(quant_spec_per_group, max_quant_idx_per_group)`.
///
/// `sfb_offset_per_group[g]` is the per-group `sfb_offset[]` table
/// (Annex B at the per-group transform length). `sections_per_group`
/// must have one entry per group; `max_sfb_per_group[g]` ditto.
#[allow(clippy::type_complexity)]
pub fn parse_asf_spectral_data_grouped(
    br: &mut BitReader<'_>,
    sections_per_group: &[AsfSections],
    sfb_offset_per_group: &[&[u16]],
    max_sfb_per_group: &[u32],
) -> Result<(Vec<Vec<i32>>, Vec<Vec<u32>>)> {
    let n = sections_per_group.len();
    if sfb_offset_per_group.len() != n || max_sfb_per_group.len() != n {
        return Err(Error::invalid(
            "ac4: asf_spectral_data_grouped: inconsistent per-group slice lengths",
        ));
    }
    let mut q_per_g = Vec::with_capacity(n);
    let mut mqi_per_g = Vec::with_capacity(n);
    for g in 0..n {
        let (q, mqi) = parse_asf_spectral_data(
            br,
            &sections_per_group[g],
            sfb_offset_per_group[g],
            max_sfb_per_group[g],
        )?;
        q_per_g.push(q);
        mqi_per_g.push(mqi);
    }
    Ok((q_per_g, mqi_per_g))
}

/// Parse `asf_scalefac_data()` per Table 41 with the spec's
/// `reference_scale_factor` (8 bits) consumed *once*, then the outer
/// `for (g = 0; g < num_window_groups; g++)` loop with `first_scf_found`
/// shared across groups (so the DPCM state runs continuously over the
/// whole frame). Returns one per-sfb scale-factor-gain vector per group.
///
/// `transform_length_per_group[g]` drives the per-group `num_sfb_48`
/// cap on the LSF range. `max_quant_idx_per_group[g]` is the per-group
/// per-sfb max-quant-index that `parse_asf_spectral_data_grouped`
/// produced.
pub fn parse_asf_scalefac_data_grouped(
    br: &mut BitReader<'_>,
    sections_per_group: &[AsfSections],
    max_quant_idx_per_group: &[Vec<u32>],
    max_sfb_per_group: &[u32],
    transform_length_per_group: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let n = sections_per_group.len();
    if max_quant_idx_per_group.len() != n
        || max_sfb_per_group.len() != n
        || transform_length_per_group.len() != n
    {
        return Err(Error::invalid(
            "ac4: asf_scalefac_data_grouped: inconsistent per-group slice lengths",
        ));
    }
    let reference_scale_factor = br.read_u32(8)?;
    let mut scale_factor: i32 = reference_scale_factor as i32;
    let mut first_scf_found = false;
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let max_sfb = max_sfb_per_group[g];
        let tl = transform_length_per_group[g];
        let num_sfb_lsf = num_sfb_48(tl)
            .ok_or_else(|| Error::invalid("ac4: scalefac_grouped: bad transform_length"))?;
        let max_sfb_eff = max_sfb.min(num_sfb_lsf);
        let mqi = &max_quant_idx_per_group[g];
        let sections = &sections_per_group[g];
        let mut sf_gain = vec![0.0_f32; max_sfb as usize];
        for sfb in 0..max_sfb_eff as usize {
            let cb = sections.sfb_cb[sfb];
            if cb == 0 || mqi[sfb] == 0 {
                continue;
            }
            if first_scf_found {
                let cw_idx = huff_decode(br, HCB_SCALEFAC_LEN, HCB_SCALEFAC_CW)?;
                scale_factor += cw_idx as i32 - 60;
            } else {
                first_scf_found = true;
            }
            let sf = scale_factor;
            let exp = (sf as f32 - 100.0) * 0.25;
            sf_gain[sfb] = 2.0_f32.powf(exp);
        }
        out.push(sf_gain);
    }
    Ok(out)
}

/// Parse `asf_snf_data()` per Table 42 with the spec's
/// `b_snf_data_exists` (1 bit) consumed *once* at the head, then the
/// outer `for (g = 0; g < num_window_groups; g++)` loop. Returns
/// `Some(per_group_dpcm_snf)` when present, `None` when absent.
pub fn parse_asf_snf_data_grouped(
    br: &mut BitReader<'_>,
    sections_per_group: &[AsfSections],
    max_quant_idx_per_group: &[Vec<u32>],
    max_sfb_per_group: &[u32],
    transform_length_per_group: &[u32],
) -> Result<Option<Vec<Vec<i32>>>> {
    let n = sections_per_group.len();
    if max_quant_idx_per_group.len() != n
        || max_sfb_per_group.len() != n
        || transform_length_per_group.len() != n
    {
        return Err(Error::invalid(
            "ac4: asf_snf_data_grouped: inconsistent per-group slice lengths",
        ));
    }
    let b_snf_data_exists = br.read_bit()?;
    if !b_snf_data_exists {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let max_sfb = max_sfb_per_group[g];
        let tl = transform_length_per_group[g];
        let num_sfb_lsf = num_sfb_48(tl)
            .ok_or_else(|| Error::invalid("ac4: snf_grouped: bad transform_length"))?;
        let max_sfb_eff = max_sfb.min(num_sfb_lsf);
        let mqi = &max_quant_idx_per_group[g];
        let sections = &sections_per_group[g];
        let mut dpcm_snf = vec![0i32; max_sfb as usize];
        for sfb in 0..max_sfb_eff as usize {
            let cb = sections.sfb_cb[sfb];
            if cb == 0 || mqi[sfb] == 0 {
                let idx = huff_decode(br, HCB_SNF_LEN, HCB_SNF_CW)?;
                dpcm_snf[sfb] = idx as i32;
            }
        }
        out.push(dpcm_snf);
    }
    Ok(Some(out))
}

/// Apply Pseudocode 18 (dequantisation reconstruction) to the raw
/// quantised spectral lines and scale them by `sf_gain[sfb]`.
pub fn dequantise_and_scale(
    quant_spec: &[i32],
    sf_gain: &[f32],
    sfb_offset: &[u16],
    max_sfb: u32,
) -> Vec<f32> {
    let end_bin = sfb_offset[max_sfb as usize] as usize;
    let mut scaled = vec![0.0_f32; end_bin];
    for sfb in 0..max_sfb as usize {
        let gain = sf_gain[sfb];
        if gain == 0.0 {
            continue;
        }
        let a = sfb_offset[sfb] as usize;
        let b = sfb_offset[sfb + 1] as usize;
        for k in a..b.min(end_bin) {
            let q = quant_spec[k];
            // rec_spec = sign(q) * |q|^(4/3).
            let rec = (q.unsigned_abs() as f32).powf(4.0 / 3.0);
            let rec_signed = if q < 0 { -rec } else { rec };
            scaled[k] = gain * rec_signed;
        }
    }
    scaled
}

/// §5.1.4 Spectral noise fill — inject shaped white noise into zero-energy
/// spectral bins using the per-band `dpcm_snf[]` indices parsed by
/// [`parse_asf_snf_data`].
///
/// The gain for band `sfb` is `2^((snf_idx * 1.5 - 84) / 4)`, which
/// matches the `sf_gain` formula but with the SNF index substituted for
/// the scalefactor index and a 1.5 step-size (vs 1.0 for scalefacs).
///
/// Noise is injected into every bin `k` in `sfb` where `scaled[k] == 0.0`
/// (no energy from the MDCT coefficient path). The pseudo-random number
/// sequence uses a 16-bit LCG with multiplier 69069 + addend 1 (same
/// convention as the ASF random-sign generator in §5.1.2.3).
///
/// `snf_data`: per-band SNF indices from `parse_asf_snf_data` (length
/// `max_sfb`). Bands with index 0 contribute no noise. `sfb_offset` and
/// `max_sfb` must match the ones used to produce `scaled`. `rng_state` is
/// updated across calls so the noise sequence continues frame-to-frame;
/// callers should reset it on I-frames.
pub fn inject_snf_noise(
    scaled: &mut [f32],
    snf_data: &[i32],
    sfb_offset: &[u16],
    max_sfb: u32,
    rng_state: &mut u32,
) {
    for sfb in 0..max_sfb as usize {
        let idx = match snf_data.get(sfb) {
            Some(&v) if v > 0 => v as u32,
            _ => continue,
        };
        // SNF gain: step size = 1.5 dB, base offset equivalent to sf=84.
        // gain = 2^((idx * 1.5 - 84) / 4.0)
        let snf_gain: f32 = 2.0_f32.powf((idx as f32 * 1.5 - 84.0) / 4.0);
        let a = sfb_offset.get(sfb).copied().unwrap_or(0) as usize;
        let b = sfb_offset.get(sfb + 1).copied().unwrap_or(0) as usize;
        for k in a..b.min(scaled.len()) {
            if scaled[k] == 0.0 {
                // 16-bit LCG: next = (69069 * state + 1) mod 2^32
                *rng_state = rng_state.wrapping_mul(69069).wrapping_add(1);
                // Sign from bit 15 of the state, magnitude = 1.
                let sign: f32 = if (*rng_state >> 15) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                scaled[k] = sign * snf_gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    fn encode_scalefac_idx(bw: &mut BitWriter, idx: usize) {
        bw.write_u32(
            crate::huffman::HCB_SCALEFAC_CW[idx],
            crate::huffman::HCB_SCALEFAC_LEN[idx] as u32,
        );
    }

    #[test]
    fn section_parser_single_cb_zero_covers_all_bands() {
        // max_sfb = 5, transf_length_idx = 0, transform_length = 256.
        // sect_cb = 0 (4 bits = 0), sect_len_incr = (max_sfb - 1) = 4
        // (3 bits), non-escape. => one section spanning sfb 0..5.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 4); // sect_cb
        bw.write_u32(4, 3); // sect_len_incr (non-escape) -> sect_len = 5
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let s = parse_asf_section_data(&mut br, 0, 256, 5).unwrap();
        assert_eq!(s.num_sec, 1);
        assert_eq!(s.sect_cb, vec![0u8]);
        assert_eq!(s.sect_start, vec![0u16]);
        assert_eq!(s.sect_end, vec![5u16]);
        assert_eq!(s.sfb_cb, vec![0u8; 5]);
    }

    #[test]
    fn section_parser_two_sections_escape() {
        // max_sfb = 10, transf_length_idx = 0 -> n_sect_bits = 3, esc = 7.
        // Section 0: sect_cb = 3 (non-zero), sect_len = 3.
        //   -> sect_len_incr = 2 (non-escape).
        // Section 1: sect_cb = 5, sect_len = 7.
        //   -> sect_len_incr = 6 (non-escape).
        let mut bw = BitWriter::new();
        bw.write_u32(3, 4);
        bw.write_u32(2, 3);
        bw.write_u32(5, 4);
        bw.write_u32(6, 3);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let s = parse_asf_section_data(&mut br, 0, 256, 10).unwrap();
        assert_eq!(s.num_sec, 2);
        assert_eq!(s.sect_cb, vec![3u8, 5u8]);
        assert_eq!(s.sect_start, vec![0u16, 3u16]);
        assert_eq!(s.sect_end, vec![3u16, 10u16]);
    }

    #[test]
    fn spectral_parser_all_zero_section() {
        // max_sfb = 2 at transform_length = 256 (num_sfb=20). Section
        // cb = 0 covers both. No Huffman bits needed in spectral data.
        let sections = AsfSections {
            sect_cb: vec![0u8],
            sect_start: vec![0u16],
            sect_end: vec![2u16],
            sfb_cb: vec![0u8; 2],
            num_sec: 1,
            num_sec_lsf: 1,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(0, 8); // ensure byte exists
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let sfb_offset = crate::sfb_offset::sfb_offset_48(256).unwrap();
        let (qspec, mqi) = parse_asf_spectral_data(&mut br, &sections, sfb_offset, 2).unwrap();
        let end_bin = sfb_offset[2] as usize;
        assert_eq!(qspec.len(), end_bin);
        assert!(qspec.iter().all(|&v| v == 0));
        assert_eq!(mqi, vec![0u32, 0u32]);
    }

    #[test]
    fn scalefac_parser_yields_gain_when_mqi_nonzero() {
        // max_sfb = 3 at transform_length = 256. Section 0 covers all
        // with cb=5 (non-zero). max_quant_idx[1] = 3.
        // reference_scale_factor = 120 -> first sfb with cb!=0 and
        // mqi>0 pins scale_factor at 120, gain = 2^((120-100)/4) = 32.0.
        let sections = AsfSections {
            sfb_cb: vec![5u8, 5u8, 5u8],
            ..AsfSections::default()
        };
        let mqi = vec![0u32, 3u32, 0u32];
        let mut bw = BitWriter::new();
        bw.write_u32(120, 8); // reference_scale_factor
                              // Only sfb=1 has mqi>0 and cb!=0 -> first_scf_found is set,
                              // no codeword emitted. sfb=0: mqi=0 -> skip. sfb=2: mqi=0 ->
                              // skip. So no huffman data needed.
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let gains = parse_asf_scalefac_data(&mut br, &sections, &mqi, 3, 256).unwrap();
        assert_eq!(gains[0], 0.0);
        assert!((gains[1] - 32.0).abs() < 1e-3);
        assert_eq!(gains[2], 0.0);
    }

    #[test]
    fn scalefac_parser_reads_dpcm_for_subsequent_bands() {
        // max_sfb = 3. sfb_cb = [5, 5, 5]. mqi = [3, 3, 3]. Reference
        // sf = 120. First band anchors at 120. Second band reads one
        // SCALEFAC codeword — use idx = 60 (codeword "1"). Scale
        // factor moves by cw_idx - 60 = 0, stays at 120. Third band
        // reads another codeword idx=63 (len=4, cw=0x4): delta = 3,
        // scale = 123.
        let sections = AsfSections {
            sfb_cb: vec![5u8, 5u8, 5u8],
            ..AsfSections::default()
        };
        let mqi = vec![3u32, 3u32, 3u32];
        let mut bw = BitWriter::new();
        bw.write_u32(120, 8);
        encode_scalefac_idx(&mut bw, 60);
        encode_scalefac_idx(&mut bw, 63);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let gains = parse_asf_scalefac_data(&mut br, &sections, &mqi, 3, 256).unwrap();
        assert!((gains[0] - 32.0).abs() < 1e-2);
        assert!((gains[1] - 32.0).abs() < 1e-2);
        assert!((gains[2] - 2.0_f32.powf((123.0 - 100.0) / 4.0)).abs() < 1e-2);
    }

    #[test]
    fn dequantise_scales_correctly() {
        let qspec = vec![0i32, 0, 2, -2, 0, 0, 0, 0];
        let sfb_offset = [0u16, 2, 4, 8];
        let sf_gain = vec![0.0_f32, 1.0, 0.5];
        let out = dequantise_and_scale(&qspec, &sf_gain, &sfb_offset, 3);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        let exp_mag = 2.0_f32.powf(4.0 / 3.0);
        assert!((out[2] - exp_mag).abs() < 1e-4);
        assert!((out[3] + exp_mag).abs() < 1e-4);
        assert!(out[4..].iter().all(|&v| v == 0.0));
    }

    /// Grouped section_data walker: two groups, both all-zero CB-0
    /// covering all bands. Each group emits the same 4-bit cb=0 +
    /// n_sect_bits length-incr (no escape) chain.
    #[test]
    fn section_grouped_walks_two_groups_all_zero_sections() {
        // Per group: max_sfb = 5, transf_length_idx = 0 -> n_sect_bits = 3, esc = 7.
        // sect_cb = 0 (4 bits), sect_len_incr = 4 (3 bits, non-escape).
        // Two groups => 2 * (4+3) = 14 bits.
        let mut bw = BitWriter::new();
        for _ in 0..2 {
            bw.write_u32(0, 4); // sect_cb
            bw.write_u32(4, 3); // sect_len_incr -> sect_len = 5
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let s = parse_asf_section_data_grouped(
            &mut br,
            &[0u32, 0u32],
            &[256u32, 256u32],
            &[5u32, 5u32],
        )
        .unwrap();
        assert_eq!(s.len(), 2);
        for sg in &s {
            assert_eq!(sg.num_sec, 1);
            assert_eq!(sg.sect_cb, vec![0u8]);
            assert_eq!(sg.sect_start, vec![0u16]);
            assert_eq!(sg.sect_end, vec![5u16]);
        }
    }

    /// Grouped scalefac_data: ONE 8-bit reference at the head, then
    /// per-group DPCM with shared `first_scf_found`. With all-zero
    /// `max_quant_idx`, no DPCM codewords are emitted — only the
    /// 8-bit reference.
    #[test]
    fn scalefac_grouped_consumes_single_reference_then_no_dpcm_when_mqi_all_zero() {
        let sections = vec![
            AsfSections {
                sfb_cb: vec![5u8, 5u8],
                ..AsfSections::default()
            },
            AsfSections {
                sfb_cb: vec![5u8, 5u8],
                ..AsfSections::default()
            },
        ];
        let mqi_per_g = vec![vec![0u32, 0u32], vec![0u32, 0u32]];
        let mut bw = BitWriter::new();
        bw.write_u32(120, 8); // reference_scale_factor
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let gains_per_g = parse_asf_scalefac_data_grouped(
            &mut br,
            &sections,
            &mqi_per_g,
            &[2u32, 2u32],
            &[256u32, 256u32],
        )
        .unwrap();
        assert_eq!(gains_per_g.len(), 2);
        for g in &gains_per_g {
            assert_eq!(g.len(), 2);
            assert!(g.iter().all(|&v| v == 0.0));
        }
    }

    /// Grouped scalefac_data: `first_scf_found` carries across groups —
    /// the second group's first active band emits a DPCM codeword
    /// (rather than anchoring on the reference).
    #[test]
    fn scalefac_grouped_first_scf_found_carries_across_groups() {
        let sections = vec![
            AsfSections {
                sfb_cb: vec![5u8, 5u8],
                ..AsfSections::default()
            },
            AsfSections {
                sfb_cb: vec![5u8, 5u8],
                ..AsfSections::default()
            },
        ];
        // Group 0: sfb 0 active (cb!=0, mqi>0) -> first_scf_found set,
        //   no codeword emitted.
        // Group 1: sfb 0 active -> DPCM codeword emitted.
        let mqi_per_g = vec![vec![3u32, 0u32], vec![3u32, 0u32]];
        let mut bw = BitWriter::new();
        bw.write_u32(120, 8); // reference
                              // One DPCM codeword for group 1's first active band: idx=60
                              // -> delta 0 -> stays at 120.
        bw.write_u32(
            crate::huffman::HCB_SCALEFAC_CW[60],
            crate::huffman::HCB_SCALEFAC_LEN[60] as u32,
        );
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let gains_per_g = parse_asf_scalefac_data_grouped(
            &mut br,
            &sections,
            &mqi_per_g,
            &[2u32, 2u32],
            &[256u32, 256u32],
        )
        .unwrap();
        assert!((gains_per_g[0][0] - 32.0).abs() < 1e-2);
        assert!((gains_per_g[1][0] - 32.0).abs() < 1e-2);
    }

    /// Grouped snf_data: ONE 1-bit gate at the head; when 0 returns
    /// `None` regardless of group count.
    #[test]
    fn snf_grouped_single_gate_at_head_returns_none_when_absent() {
        let sections = vec![AsfSections::default(); 3];
        let mqi_per_g = vec![vec![0u32; 0]; 3];
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_snf_data_exists = 0
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let snf =
            parse_asf_snf_data_grouped(&mut br, &sections, &mqi_per_g, &[0u32; 3], &[256u32; 3])
                .unwrap();
        assert!(snf.is_none());
    }

    // ------------------------------------------------------------------
    // §5.1.4 inject_snf_noise — spectral noise fill
    // ------------------------------------------------------------------

    #[test]
    fn inject_snf_fills_zero_bins_only() {
        // SFB 0 spans bins [0..4], SNF idx=56 (gain > 0).
        // bin 1 is non-zero → must not be overwritten.
        let sfb_offset: Vec<u16> = vec![0, 4, 8];
        let snf_data: Vec<i32> = vec![56, 0]; // sfb0 has idx, sfb1 is zero→skip
        let mut scaled = vec![0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng: u32 = 1;
        inject_snf_noise(&mut scaled, &snf_data, &sfb_offset, 2, &mut rng);
        // bin 1 must be unchanged.
        assert_eq!(scaled[1], 1.0);
        // bins 0, 2, 3 must be non-zero (filled with SNF gain × ±1).
        let snf_gain = 2.0_f32.powf((56.0 * 1.5 - 84.0) / 4.0);
        assert!((scaled[0].abs() - snf_gain).abs() < 1e-5);
        assert!((scaled[2].abs() - snf_gain).abs() < 1e-5);
        assert!((scaled[3].abs() - snf_gain).abs() < 1e-5);
        // sfb1 had idx=0 (≤0) so bins 4..8 must stay zero.
        for &s in &scaled[4..] {
            assert_eq!(s, 0.0, "sfb1 bins must remain zero");
        }
    }

    #[test]
    fn inject_snf_gain_formula_idx56() {
        // idx=56: gain = 2^((56*1.5-84)/4) = 2^((84-84)/4) = 2^0 = 1.0
        let snf_gain = 2.0_f32.powf((56.0_f32 * 1.5 - 84.0) / 4.0);
        assert!((snf_gain - 1.0).abs() < 1e-5, "idx=56 gain must be 1.0");
    }

    #[test]
    fn inject_snf_lcg_state_advances() {
        // Each zero bin should advance the RNG and produce a deterministic result.
        let sfb_offset: Vec<u16> = vec![0, 2];
        let snf_data: Vec<i32> = vec![56];
        let mut scaled = vec![0.0_f32; 2];
        let mut rng: u32 = 0x1234_5678;
        inject_snf_noise(&mut scaled, &snf_data, &sfb_offset, 1, &mut rng);
        // Both bins should be ±1.0 (since idx=56 → gain=1.0).
        assert!((scaled[0].abs() - 1.0).abs() < 1e-5);
        assert!((scaled[1].abs() - 1.0).abs() < 1e-5);
        // RNG advanced exactly twice.
        let mut expected_rng = 0x1234_5678u32;
        expected_rng = expected_rng.wrapping_mul(69069).wrapping_add(1);
        let sign0: f32 = if (expected_rng >> 15) & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        assert_eq!(scaled[0], sign0 * 1.0);
        expected_rng = expected_rng.wrapping_mul(69069).wrapping_add(1);
        let sign1: f32 = if (expected_rng >> 15) & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        assert_eq!(scaled[1], sign1 * 1.0);
    }
}
