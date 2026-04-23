//! Normative AC-4 lookup tables (ETSI TS 103 190-1 Annex B and §4.3.6).
//!
//! These are the data-driven side of the ASF: transform length →
//! `num_sfb`, codeword widths for `max_sfb[i]` / `max_sfb_side[i]` /
//! `max_sfb_master`, and the section-length-increment width.
//!
//! The tables are verbatim transcriptions of the spec — no processing,
//! no FFmpeg / reference-code influence. The comments cite the table
//! numbers so an auditor can cross-check.

/// `num_sfb` for 44.1 kHz or 48 kHz sampling frequency (Table B.1).
///
/// Returns `None` for transform lengths that aren't defined at 44.1/48
/// kHz.
pub fn num_sfb_48(transform_length: u32) -> Option<u32> {
    Some(match transform_length {
        2048 => 63,
        1920 => 61,
        1536 => 55,
        1024 => 49,
        960 => 49,
        768 => 43,
        512 => 36,
        480 => 36,
        384 => 33,
        256 => 20,
        240 => 20,
        192 => 18,
        128 => 14,
        120 => 14,
        96 => 12,
        _ => return None,
    })
}

/// `num_sfb` for 96 kHz sampling frequency (Table B.2).
pub fn num_sfb_96(transform_length: u32) -> Option<u32> {
    Some(match transform_length {
        4096 => 79,
        3840 => 76,
        3072 => 67,
        2048 => 57,
        1920 => 57,
        1536 => 49,
        1024 => 44,
        920 => 44,
        768 => 39,
        512 => 28,
        480 => 28,
        384 => 24,
        256 => 22,
        240 => 22,
        192 => 18,
        _ => return None,
    })
}

/// `num_sfb` for 192 kHz sampling frequency (Table B.3).
pub fn num_sfb_192(transform_length: u32) -> Option<u32> {
    Some(match transform_length {
        8192 => 111,
        7680 => 106,
        6144 => 91,
        4096 => 73,
        3840 => 72,
        3072 => 61,
        2048 => 60,
        1920 => 59,
        1536 => 51,
        1024 => 36,
        960 => 36,
        768 => 30,
        512 => 30,
        480 => 30,
        384 => 24,
        _ => return None,
    })
}

/// Widths of `max_sfb[i]`, `max_sfb_side[i]` and `max_sfbl_bits`
/// (Table 106, 44.1/48 kHz family) for a given transform length.
///
/// Returns `(n_msfb_bits, n_side_bits, n_msfbl_bits)`. Fields that are
/// "N/A" in the spec are returned as 0 — the caller should know whether
/// it's allowed to reach for `n_msfbl_bits` based on `b_long_frame`.
pub fn n_msfb_bits_48(transform_length: u32) -> Option<(u32, u32, u32)> {
    Some(match transform_length {
        2048 => (6, 5, 3),
        1920 => (6, 5, 3),
        1536 => (6, 5, 3),
        1024 => (6, 5, 2),
        960 => (6, 5, 2),
        768 => (6, 5, 2),
        512 => (6, 5, 2),
        480 => (6, 5, 0),
        384 => (6, 4, 2),
        256 => (5, 4, 0),
        240 => (5, 4, 0),
        192 => (5, 3, 0),
        128 => (4, 3, 0),
        120 => (4, 3, 0),
        96 => (4, 3, 0),
        _ => return None,
    })
}

/// `n_sect_bits` per Pseudocode 6 in §4.3.6.3.2. Returns 3 for
/// `transf_length_group <= 2` and 5 otherwise. The input is the
/// `transf_length` *index* (0..=3), not the resolved transform length.
#[inline]
pub fn n_sect_bits(transf_length_idx: u32) -> u32 {
    if transf_length_idx <= 2 {
        3
    } else {
        5
    }
}

// TODO: `sfb_offset_48()` — Tables B.4 / B.5 / B.6. These are long
// column-per-transform-length tables (63, 61, 55, 49, 49, 43, 36, 36,
// 33 entries etc.) and are only meaningful once asf_section_data()
// and asf_spectral_data() decoding lands. They are deliberately
// deferred so that when they're needed they can be transcribed with
// a proper value-level cross-check against the spec rather than a
// blind dump.

/// `sect_esc_val` companion to [`n_sect_bits`]: `(1 << n_sect_bits) - 1`.
#[inline]
pub fn sect_esc_val(transf_length_idx: u32) -> u32 {
    (1 << n_sect_bits(transf_length_idx)) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_sfb_48_samples() {
        assert_eq!(num_sfb_48(2048), Some(63));
        assert_eq!(num_sfb_48(1920), Some(61));
        assert_eq!(num_sfb_48(1024), Some(49));
        assert_eq!(num_sfb_48(960), Some(49));
        assert_eq!(num_sfb_48(512), Some(36));
        assert_eq!(num_sfb_48(96), Some(12));
        assert_eq!(num_sfb_48(1000), None);
    }

    #[test]
    fn n_msfb_bits_48_samples() {
        assert_eq!(n_msfb_bits_48(2048), Some((6, 5, 3)));
        assert_eq!(n_msfb_bits_48(480), Some((6, 5, 0)));
        assert_eq!(n_msfb_bits_48(384), Some((6, 4, 2)));
        assert_eq!(n_msfb_bits_48(128), Some((4, 3, 0)));
        assert_eq!(n_msfb_bits_48(999), None);
    }

    #[test]
    fn n_sect_bits_split() {
        assert_eq!(n_sect_bits(0), 3);
        assert_eq!(n_sect_bits(2), 3);
        assert_eq!(n_sect_bits(3), 5);
        assert_eq!(sect_esc_val(0), 7);
        assert_eq!(sect_esc_val(3), 31);
    }
}
