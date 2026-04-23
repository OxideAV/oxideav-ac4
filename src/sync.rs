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
    /// Offset of the sync word within the input slice.
    pub sync_offset: usize,
    /// Total bytes consumed starting at `sync_offset`.
    pub total_len: usize,
}

/// Find the next AC-4 sync frame in `data`. Returns `None` if no valid
/// sync word is found or if the frame extends past the buffer.
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
    Ok(SyncFrame {
        payload: &data[payload_start..payload_end],
        crc_protected,
        sync_offset: offset,
        total_len: payload_end + crc_len - offset,
    })
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
}
