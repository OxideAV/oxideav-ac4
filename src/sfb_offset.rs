//! Scale-factor-band offset tables (Annex B.4-B.7 of ETSI TS 103 190-1).
//!
//! Each table entry is `sfb_offset[sfb]` — the bin index at which scale
//! factor band `sfb` starts. Band `sfb` spans
//! `[sfb_offset[sfb] .. sfb_offset[sfb+1])`.
//!
//! Tables B.4, B.5, B.6 and B.7 are laid out in the spec as a matrix
//! with one column per transform length. We split them into
//! per-transform-length vectors here for direct lookup.
//!
//! Only the 44,1 / 48 kHz family is exposed — HSF extension (96 / 192
//! kHz) reaches the same tables via block-size mapping but isn't
//! wired in yet.
//!
//! Numeric data transcribed verbatim from the spec (uncopyrightable
//! facts).

/// `sfb_offset[sfb]` for 48 kHz / 44.1 kHz, transform length 2048.
/// Length = num_sfb_48(2048) + 1 = 64 entries (0..=63).
///
/// Table B.4, column "2 048@44,1 / 2 048@48".
pub const SFB_OFFSET_2048: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664,
    1728, 1792, 1856, 1920, 1984, 2048,
];

/// Table B.4, column "1 920@48". 62 entries (num_sfb_48(1920) + 1 = 62).
pub const SFB_OFFSET_1920: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664,
    1728, 1792, 1856, 1920,
];

// Note: at 1920 the last several sfb_offset entries (sfb=56 onward)
// follow the "1 920@48" column. Our transcription ends at 1920 which
// equals the transform length.

/// Table B.4, column "1 536@48". 56 entries (num_sfb_48(1536) + 1 = 56).
pub const SFB_OFFSET_1536: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536,
];

/// Table B.5, column "1 024@44,1 / 1 024@48". 50 entries (49+1).
pub const SFB_OFFSET_1024: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// Table B.5, column "960@48". 50 entries (49+1).
pub const SFB_OFFSET_960: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960,
];

/// Table B.5, column "768@48". 44 entries (43+1).
pub const SFB_OFFSET_768: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768,
];

/// Table B.6, column "512@44,1 / 512@48". 37 entries (36+1).
pub const SFB_OFFSET_512: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 76, 84, 92, 100, 112, 124,
    136, 148, 164, 184, 208, 236, 268, 300, 332, 364, 396, 428, 460, 512,
];

/// Table B.6, column "480@48". 37 entries (36+1).
pub const SFB_OFFSET_480: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 76, 84, 92, 100, 112, 124,
    136, 148, 164, 184, 208, 236, 268, 300, 332, 364, 396, 428, 460, 480,
];

/// Table B.6, column "384@48". 34 entries (33+1).
pub const SFB_OFFSET_384: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 76, 84, 92, 100, 112, 124,
    136, 148, 164, 184, 208, 236, 268, 300, 332, 364, 384,
];

/// Table B.7, column "256@44,1 / 256@48". 21 entries (20+1).
pub const SFB_OFFSET_256: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128, 148, 172, 196, 224, 256,
];

/// Table B.7, column "240@48". 21 entries (20+1).
pub const SFB_OFFSET_240: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128, 148, 172, 196, 224, 240,
];

/// Table B.7, column "192@48". 19 entries (18+1).
pub const SFB_OFFSET_192: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128, 148, 172, 192,
];

/// Table B.7, column "128@44,1 / 128@48". 15 entries (14+1).
pub const SFB_OFFSET_128: &[u16] = &[0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

/// Table B.7, column "120@48". 15 entries (14+1).
pub const SFB_OFFSET_120: &[u16] = &[0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 120];

/// Table B.7, column "96@48". 13 entries (12+1).
pub const SFB_OFFSET_96: &[u16] = &[0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96];

/// Look up the `sfb_offset[]` vector for the given transform length at
/// 48 kHz / 44.1 kHz.
pub fn sfb_offset_48(transform_length: u32) -> Option<&'static [u16]> {
    Some(match transform_length {
        2048 => SFB_OFFSET_2048,
        1920 => SFB_OFFSET_1920,
        1536 => SFB_OFFSET_1536,
        1024 => SFB_OFFSET_1024,
        960 => SFB_OFFSET_960,
        768 => SFB_OFFSET_768,
        512 => SFB_OFFSET_512,
        480 => SFB_OFFSET_480,
        384 => SFB_OFFSET_384,
        256 => SFB_OFFSET_256,
        240 => SFB_OFFSET_240,
        192 => SFB_OFFSET_192,
        128 => SFB_OFFSET_128,
        120 => SFB_OFFSET_120,
        96 => SFB_OFFSET_96,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::num_sfb_48;

    fn assert_consistent(len: u32, offs: &[u16]) {
        // Length must equal num_sfb + 1 with last entry = transform length.
        let expected_n = num_sfb_48(len).unwrap() as usize + 1;
        assert_eq!(offs.len(), expected_n, "length mismatch for {len}");
        assert_eq!(*offs.first().unwrap(), 0);
        assert_eq!(*offs.last().unwrap() as u32, len);
        // Monotonically increasing.
        for w in offs.windows(2) {
            assert!(w[0] < w[1], "non-monotone at {len}: {:?}", w);
        }
    }

    #[test]
    fn sfb_offset_tables_consistent() {
        for &tl in &[
            2048u32, 1920, 1536, 1024, 960, 768, 512, 480, 384, 256, 240, 192, 128, 120, 96,
        ] {
            let o = sfb_offset_48(tl).unwrap();
            assert_consistent(tl, o);
        }
    }

    #[test]
    fn sfb_offset_48_lookup() {
        let t = sfb_offset_48(1920).unwrap();
        assert_eq!(t[0], 0);
        assert_eq!(t[t.len() - 1], 1920);
        assert!(sfb_offset_48(999).is_none());
    }
}
