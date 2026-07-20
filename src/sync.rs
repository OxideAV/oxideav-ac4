//! AC-4 sync-frame helpers (TS 103 190-1 Annex G).
//!
//! The wire framing around a `raw_ac4_frame()` payload is:
//!
//! ```text
//!   sync_word  (2 bytes, 0xAC40 or 0xAC41)
//!   frame_size (2 bytes; escape to 3 bytes if == 0xFFFF)
//!   raw_ac4_frame()   (frame_size bytes)
//!   crc_word   (2 bytes, only when sync_word == 0xAC41)
//! ```
//!
//! `find_sync_frame` scans a byte slice for `0xAC40`/`0xAC41`, reads
//! `frame_size`, and returns the slice covering the `raw_ac4_frame()`
//! payload plus the total framed length so callers can advance.
//!
//! MP4 / TS-4 / ADTS-in-MP4 containers hand us raw payloads directly
//! with no sync word, so the decoder supports both paths.

use oxideav_core::{Error, Result};

pub const SYNC_WORD_PLAIN: u16 = 0xAC40;
pub const SYNC_WORD_CRC: u16 = 0xAC41;

/// Result of a successful framing scan.
#[derive(Debug, Clone, Copy)]
pub struct SyncFrame<'a> {
    /// The raw `raw_ac4_frame()` payload.
    pub payload: &'a [u8],
    /// True when the sync word was `0xAC41` (CRC-protected).
    pub crc_protected: bool,
    /// The transmitted `crc_word` (0xAC41 frames only).
    pub crc_word: Option<u16>,
    /// CRC verification per Annex G.4.2 — the protected payload is
    /// `frame_size` + `raw_ac4_frame` (the sync word is excluded);
    /// processing it followed by `crc_word` through the register must
    /// yield 0x0000. `None` for unprotected (0xAC40) frames.
    pub crc_valid: Option<bool>,
    /// Offset of the sync word within the input slice.
    pub sync_offset: usize,
    /// Total bytes consumed starting at `sync_offset`.
    pub total_len: usize,
}

/// Find the next AC-4 sync frame in `data`. Returns `None` if no valid
/// sync word is found or if the frame extends past the buffer.
/// Parse a sync frame anchored at the **start** of `data` — the
/// packet-boundary form used by the decoder's per-packet dispatch. A
/// packet either is sync-wrapped (sync word first) or is a bare
/// `raw_ac4_frame`; scanning for a sync word at arbitrary offsets
/// (see [`find_sync_frame`]) would let a chance `0xAC40` byte pair
/// inside a bare frame's payload hijack the parse.
pub fn parse_sync_frame_at_start(data: &[u8]) -> Option<SyncFrame<'_>> {
    if data.len() < 4 {
        return None;
    }
    let sync = u16::from_be_bytes([data[0], data[1]]);
    if sync != SYNC_WORD_PLAIN && sync != SYNC_WORD_CRC {
        return None;
    }
    try_parse_frame_at(data, 0).ok()
}

pub fn find_sync_frame(data: &[u8]) -> Option<SyncFrame<'_>> {
    if data.len() < 4 {
        return None;
    }
    let mut i = 0usize;
    while i + 4 <= data.len() {
        let sync = u16::from_be_bytes([data[i], data[i + 1]]);
        if sync == SYNC_WORD_PLAIN || sync == SYNC_WORD_CRC {
            if let Ok(frame) = try_parse_frame_at(data, i) {
                return Some(frame);
            }
        }
        i += 1;
    }
    None
}

fn try_parse_frame_at(data: &[u8], offset: usize) -> Result<SyncFrame<'_>> {
    // sync (2) + fs_short (2).
    if offset + 4 > data.len() {
        return Err(Error::invalid("ac4: sync frame truncated"));
    }
    let sync = u16::from_be_bytes([data[offset], data[offset + 1]]);
    let crc_protected = sync == SYNC_WORD_CRC;
    if !crc_protected && sync != SYNC_WORD_PLAIN {
        return Err(Error::invalid("ac4: not a sync word"));
    }
    let fs_short = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as u32;
    let (frame_size, header_len) = if fs_short == 0xFFFF {
        if offset + 7 > data.len() {
            return Err(Error::invalid("ac4: extended frame_size truncated"));
        }
        let fs_ext = ((data[offset + 4] as u32) << 16)
            | ((data[offset + 5] as u32) << 8)
            | (data[offset + 6] as u32);
        (fs_ext, 7)
    } else {
        (fs_short, 4)
    };
    let crc_len = if crc_protected { 2 } else { 0 };
    let payload_start = offset + header_len;
    let payload_end = payload_start + frame_size as usize;
    if payload_end + crc_len > data.len() {
        return Err(Error::invalid("ac4: payload extends past buffer"));
    }
    let (crc_word, crc_valid) = if crc_protected {
        let word = u16::from_be_bytes([data[payload_end], data[payload_end + 1]]);
        // Annex G.4.2: the protected payload is the frame_size element
        // + raw_ac4_frame; feeding it followed by crc_word through the
        // register yields 0x0000 ⇔ crc16(protected) == crc_word.
        let computed = crc16(&data[offset + 2..payload_end]);
        (Some(word), Some(computed == word))
    } else {
        (None, None)
    };
    Ok(SyncFrame {
        payload: &data[payload_start..payload_end],
        crc_protected,
        crc_word,
        crc_valid,
        sync_offset: offset,
        total_len: payload_end + crc_len - offset,
    })
}

/// Wrap a `raw_ac4_frame()` payload in Annex G sync framing. When
/// `with_crc` the 0xAC41 form is used and the Annex G.4.2 `crc_word`
/// (over `frame_size` + payload) is appended.
pub fn wrap_sync_frame(payload: &[u8], with_crc: bool) -> Vec<u8> {
    let sync = if with_crc {
        SYNC_WORD_CRC
    } else {
        SYNC_WORD_PLAIN
    };
    let mut out = Vec::with_capacity(payload.len() + 9);
    out.extend_from_slice(&sync.to_be_bytes());
    let protected_start = out.len();
    if payload.len() >= 0xFFFF {
        out.extend_from_slice(&0xFFFFu16.to_be_bytes());
        let fs = payload.len() as u32;
        out.push(((fs >> 16) & 0xFF) as u8);
        out.push(((fs >> 8) & 0xFF) as u8);
        out.push((fs & 0xFF) as u8);
    } else {
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    }
    out.extend_from_slice(payload);
    if with_crc {
        let word = crc16(&out[protected_start..]);
        out.extend_from_slice(&word.to_be_bytes());
    }
    out
}

/// Compute the AC-4 frame CRC-16 (generator x^16 + x^15 + x^2 + 1,
/// init 0x0000, no reflection, no final XOR) over `input` — used when
/// the caller wants to verify the 0xAC41 trailer.
pub fn crc16(input: &[u8]) -> u16 {
    const POLY: u32 = 0x8005;
    let mut crc: u32 = 0x0000;
    for &b in input {
        crc ^= (b as u32) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    (crc & 0xFFFF) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_sync() {
        // Plain 0xAC40 with frame_size=3, payload [0x11,0x22,0x33].
        let data = [0xAC, 0x40, 0x00, 0x03, 0x11, 0x22, 0x33];
        let f = find_sync_frame(&data).expect("should sync");
        assert_eq!(f.payload, &[0x11, 0x22, 0x33]);
        assert!(!f.crc_protected);
        assert_eq!(f.total_len, 7);
    }

    #[test]
    fn parse_extended_size() {
        // Extended frame_size escape. frame_size written as 0xFFFF + 24 bits.
        let fs: u32 = 0x10_0000;
        let mut data = vec![0xAC, 0x40, 0xFF, 0xFF];
        data.push(((fs >> 16) & 0xFF) as u8);
        data.push(((fs >> 8) & 0xFF) as u8);
        data.push((fs & 0xFF) as u8);
        data.extend(std::iter::repeat(0u8).take(fs as usize));
        let f = find_sync_frame(&data).expect("should sync");
        assert_eq!(f.payload.len(), fs as usize);
        assert_eq!(f.total_len, fs as usize + 7);
    }

    #[test]
    fn crc16_zero_empty() {
        // Empty input with zero initial register → 0.
        assert_eq!(crc16(&[]), 0x0000);
    }

    #[test]
    fn crc_frame_verifies_and_flags_corruption() {
        let payload = [0x11u8, 0x22, 0x33, 0x44, 0x55];
        let wrapped = wrap_sync_frame(&payload, true);
        let f = find_sync_frame(&wrapped).expect("should sync");
        assert!(f.crc_protected);
        assert_eq!(f.payload, &payload);
        assert_eq!(f.crc_valid, Some(true));
        // G.4.2 self-check: protected payload followed by crc_word
        // yields 0x0000.
        let end = wrapped.len();
        assert_eq!(crc16(&wrapped[2..end]), 0x0000);
        // Corrupt one payload byte: the same crc_word no longer
        // matches.
        let mut bad = wrapped.clone();
        bad[5] ^= 0x01;
        let f = find_sync_frame(&bad).expect("framing still parses");
        assert_eq!(f.crc_valid, Some(false));
        // Corrupt the crc_word itself.
        let mut bad = wrapped;
        let last = bad.len() - 1;
        bad[last] ^= 0x80;
        let f = find_sync_frame(&bad).expect("framing still parses");
        assert_eq!(f.crc_valid, Some(false));
    }

    #[test]
    fn plain_frame_has_no_crc_fields() {
        let payload = [0xAAu8, 0xBB];
        let wrapped = wrap_sync_frame(&payload, false);
        let f = find_sync_frame(&wrapped).expect("should sync");
        assert!(!f.crc_protected);
        assert_eq!(f.crc_word, None);
        assert_eq!(f.crc_valid, None);
        assert_eq!(f.payload, &payload);
    }
}
