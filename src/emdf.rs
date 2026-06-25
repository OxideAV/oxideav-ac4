//! EMDF payloads substream parser — TS 103 190-1 V1.4.1.
//!
//! Implements `emdf_payloads_substream()` (Table 18, §4.2.4.4) and
//! `emdf_payload_config()` (Table 79, §4.2.14.14) end-to-end. Stops at
//! the terminator payload (`emdf_payload_id == 0`) and byte-aligns the
//! reader on exit per the spec's trailing `byte_align`.
//!
//! EMDF payload IDs other than 0 are not assigned semantic meaning by
//! TS 103 190-1 itself — table 174 (§4.3.15.1.1) defers them to the
//! AC-4 EMDF datatype registry [i.14]. The parser therefore captures
//! `emdf_payload_id` as an opaque numeric tag and the `emdf_payload_byte[]`
//! sequence as a `Vec<u8>` so callers (or downstream tools-of-tomorrow)
//! can route on the registered ID. The carried bytes do **not** include
//! the `emdf_payload_size` field nor any `emdf_payload_config()` fields,
//! per §4.3.15.1.2 ("the value of the emdf_payload_size element shall be
//! equal to the number of bytes in the following payload, excluding the
//! emdf_payload_size field and all fields in emdf_payload_config").
//!
//! ## Defensive parsing
//!
//! - Hard cap on payload count (`MAX_EMDF_PAYLOADS = 64`) so a runaway
//!   `emdf_payload_id != 0` field can't pin the parser in an unbounded
//!   loop on malformed input.
//! - Hard cap on each payload's byte size (`MAX_EMDF_PAYLOAD_BYTES =
//!   65_536`) to bound memory use even if `variable_bits(8)` decodes a
//!   pathological value.
//! - All sub-parsers are infallible-on-EOF only via the wrapped
//!   `BitReader`: an over-read bubbles up as `Error::invalid`.

use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

use crate::toc::{variable_bits, write_variable_bits};

/// Hard upper bound on the number of payloads a single
/// `emdf_payloads_substream()` may carry. Real bitstreams sit well
/// under this; the cap exists to bound runaway loops on malformed
/// input.
pub const MAX_EMDF_PAYLOADS: usize = 64;

/// Hard upper bound on a single payload's `emdf_payload_byte[]`
/// length. `emdf_payload_size` is `variable_bits(8)` so the syntax
/// can technically carry > 4 GiB; we refuse anything beyond 64 KiB
/// per payload.
pub const MAX_EMDF_PAYLOAD_BYTES: usize = 65_536;

/// `emdf_payload_config()` per §4.2.14.14 Table 79.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmdfPayloadConfig {
    /// `b_smpoffst` (§4.3.15.2.1).
    pub b_smpoffst: bool,
    /// `smpoffst` value (variable_bits(11)) when `b_smpoffst == 1`.
    pub smpoffst: Option<u32>,
    /// `b_duration` (§4.3.15.2.3).
    pub b_duration: bool,
    /// `duration` value (variable_bits(11)) when `b_duration == 1`.
    pub duration: Option<u32>,
    /// `b_groupid` (§4.3.15.2.5).
    pub b_groupid: bool,
    /// `groupid` value (variable_bits(2)) when `b_groupid == 1`.
    pub groupid: Option<u32>,
    /// `b_codecdata` (§4.3.15.2.7) — bitstreams that conform to the
    /// present document shall set this to false; we surface the bit
    /// for round-trip fidelity.
    pub b_codecdata: bool,
    /// `codecdata` (8 bits) when `b_codecdata == 1`.
    pub codecdata: Option<u8>,
    /// `b_discard_unknown_payload` (§4.3.15.2.9).
    pub b_discard_unknown_payload: bool,
    /// `b_payload_frame_aligned` (§4.3.15.2.10) — present only when
    /// `b_discard_unknown_payload == 0 && b_smpoffst == 0`.
    pub b_payload_frame_aligned: Option<bool>,
    /// `b_create_duplicate` (§4.3.15.2.11) — present only when
    /// `b_payload_frame_aligned == 1`.
    pub b_create_duplicate: Option<bool>,
    /// `b_remove_duplicate` (§4.3.15.2.12) — present only when
    /// `b_payload_frame_aligned == 1`.
    pub b_remove_duplicate: Option<bool>,
    /// `priority` (5 bits) — present when
    /// `b_smpoffst == 1 || b_payload_frame_aligned == 1`.
    pub priority: Option<u8>,
    /// `proc_allowed` (2 bits) — present in the same gate as `priority`.
    pub proc_allowed: Option<u8>,
}

/// One decoded EMDF payload (Table 18 inner loop body).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmdfPayload {
    /// Resolved `emdf_payload_id` value — 5-bit base plus the
    /// `variable_bits(5)` extension when the base equalled 31. The
    /// terminator (id == 0) is **not** surfaced as a payload entry.
    pub emdf_payload_id: u32,
    /// Decoded `emdf_payload_config()`.
    pub config: EmdfPayloadConfig,
    /// `emdf_payload_byte[0..emdf_payload_size]` verbatim.
    pub payload_bytes: Vec<u8>,
}

/// Decoded `emdf_payloads_substream()` (Table 18, §4.2.4.4).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmdfPayloadsSubstream {
    pub payloads: Vec<EmdfPayload>,
}

/// Walk `emdf_payloads_substream()` per §4.2.4.4 Table 18.
///
/// Returns when the terminator payload (id == 0) is read; the spec
/// guarantees that one terminator is always present. The trailing
/// `byte_align` is consumed before returning so the caller continues
/// at a byte boundary.
pub fn parse_emdf_payloads_substream(br: &mut BitReader<'_>) -> Result<EmdfPayloadsSubstream> {
    let mut payloads = Vec::new();

    loop {
        if payloads.len() >= MAX_EMDF_PAYLOADS {
            return Err(Error::invalid(
                "ac4: emdf_payloads_substream() exceeded MAX_EMDF_PAYLOADS — malformed?",
            ));
        }

        // Per Table 18: while (emdf_payload_id != 0) { ... }
        //
        //   emdf_payload_id ........ 5
        //   if (emdf_payload_id == 31) { emdf_payload_id += variable_bits(5); }
        //
        // We read the base 5 bits unconditionally; on the loop's first
        // iteration this is the very first read. On later iterations
        // it implements the `while` predicate.
        let id_base = br.read_u32(5)?;
        if id_base == 0 {
            // Terminator. Per §4.3.15.1, all emdf_payload_config fields
            // and emdf_payload_size are 0 in this case — but in the
            // actual syntax the terminator is just the 5-bit id == 0
            // breaking out of the while loop. The loop body that would
            // read emdf_payload_config()/emdf_payload_size is skipped.
            break;
        }

        let emdf_payload_id = if id_base == 31 {
            31u32
                .checked_add(variable_bits(br, 5)?)
                .ok_or_else(|| Error::invalid("ac4: emdf_payload_id extension overflow"))?
        } else {
            id_base
        };

        let config = parse_emdf_payload_config(br)?;

        let emdf_payload_size = variable_bits(br, 8)? as usize;
        if emdf_payload_size > MAX_EMDF_PAYLOAD_BYTES {
            return Err(Error::invalid(
                "ac4: emdf_payload_size exceeded MAX_EMDF_PAYLOAD_BYTES — malformed?",
            ));
        }

        let mut payload_bytes = Vec::with_capacity(emdf_payload_size);
        for _ in 0..emdf_payload_size {
            payload_bytes.push(br.read_u32(8)? as u8);
        }

        payloads.push(EmdfPayload {
            emdf_payload_id,
            config,
            payload_bytes,
        });
    }

    // Trailing `byte_align` per Table 18.
    byte_align(br)?;

    Ok(EmdfPayloadsSubstream { payloads })
}

/// Walk `emdf_payload_config()` per §4.2.14.14 Table 79.
pub fn parse_emdf_payload_config(br: &mut BitReader<'_>) -> Result<EmdfPayloadConfig> {
    let b_smpoffst = br.read_bit()?;
    let smpoffst = if b_smpoffst {
        Some(variable_bits(br, 11)?)
    } else {
        None
    };

    let b_duration = br.read_bit()?;
    let duration = if b_duration {
        Some(variable_bits(br, 11)?)
    } else {
        None
    };

    let b_groupid = br.read_bit()?;
    let groupid = if b_groupid {
        Some(variable_bits(br, 2)?)
    } else {
        None
    };

    let b_codecdata = br.read_bit()?;
    let codecdata = if b_codecdata {
        Some(br.read_u32(8)? as u8)
    } else {
        None
    };

    let b_discard_unknown_payload = br.read_bit()?;
    let mut b_payload_frame_aligned = None;
    let mut b_create_duplicate = None;
    let mut b_remove_duplicate = None;
    let mut priority = None;
    let mut proc_allowed = None;

    if !b_discard_unknown_payload {
        if !b_smpoffst {
            let aligned = br.read_bit()?;
            b_payload_frame_aligned = Some(aligned);
            if aligned {
                b_create_duplicate = Some(br.read_bit()?);
                b_remove_duplicate = Some(br.read_bit()?);
            }
        }

        let aligned_or_offset = b_smpoffst || b_payload_frame_aligned.unwrap_or(false);
        if aligned_or_offset {
            priority = Some(br.read_u32(5)? as u8);
            proc_allowed = Some(br.read_u32(2)? as u8);
        }
    }

    Ok(EmdfPayloadConfig {
        b_smpoffst,
        smpoffst,
        b_duration,
        duration,
        b_groupid,
        groupid,
        b_codecdata,
        codecdata,
        b_discard_unknown_payload,
        b_payload_frame_aligned,
        b_create_duplicate,
        b_remove_duplicate,
        priority,
        proc_allowed,
    })
}

/// Write `emdf_payload_config()` per §4.2.14.14 Table 79, the exact
/// inverse of [`parse_emdf_payload_config`].
///
/// The `Option` fields must be consistent with their gating bit flags
/// (e.g. `smpoffst.is_some()` iff `b_smpoffst`); an inconsistency yields
/// `Error::invalid` rather than emitting a malformed field. Values that
/// the syntax does not carry (because their gate is false) are ignored.
pub fn write_emdf_payload_config(bw: &mut BitWriter, cfg: &EmdfPayloadConfig) -> Result<()> {
    bw.write_bit(cfg.b_smpoffst);
    if cfg.b_smpoffst {
        let v = cfg
            .smpoffst
            .ok_or_else(|| Error::invalid("ac4: b_smpoffst set but smpoffst is None"))?;
        write_variable_bits(bw, 11, v);
    }

    bw.write_bit(cfg.b_duration);
    if cfg.b_duration {
        let v = cfg
            .duration
            .ok_or_else(|| Error::invalid("ac4: b_duration set but duration is None"))?;
        write_variable_bits(bw, 11, v);
    }

    bw.write_bit(cfg.b_groupid);
    if cfg.b_groupid {
        let v = cfg
            .groupid
            .ok_or_else(|| Error::invalid("ac4: b_groupid set but groupid is None"))?;
        write_variable_bits(bw, 2, v);
    }

    bw.write_bit(cfg.b_codecdata);
    if cfg.b_codecdata {
        let v = cfg
            .codecdata
            .ok_or_else(|| Error::invalid("ac4: b_codecdata set but codecdata is None"))?;
        bw.write_u32(v as u32, 8);
    }

    bw.write_bit(cfg.b_discard_unknown_payload);
    if !cfg.b_discard_unknown_payload {
        if !cfg.b_smpoffst {
            let aligned = cfg.b_payload_frame_aligned.ok_or_else(|| {
                Error::invalid("ac4: b_payload_frame_aligned required when !discard && !smpoffst")
            })?;
            bw.write_bit(aligned);
            if aligned {
                let create = cfg.b_create_duplicate.ok_or_else(|| {
                    Error::invalid("ac4: b_create_duplicate required when frame_aligned")
                })?;
                let remove = cfg.b_remove_duplicate.ok_or_else(|| {
                    Error::invalid("ac4: b_remove_duplicate required when frame_aligned")
                })?;
                bw.write_bit(create);
                bw.write_bit(remove);
            }
        }

        let aligned_or_offset = cfg.b_smpoffst || cfg.b_payload_frame_aligned.unwrap_or(false);
        if aligned_or_offset {
            let priority = cfg
                .priority
                .ok_or_else(|| Error::invalid("ac4: priority required in this gate"))?;
            let proc_allowed = cfg
                .proc_allowed
                .ok_or_else(|| Error::invalid("ac4: proc_allowed required in this gate"))?;
            bw.write_u32(priority as u32, 5);
            bw.write_u32(proc_allowed as u32, 2);
        }
    }

    Ok(())
}

/// Write `emdf_payloads_substream()` per §4.2.4.4 Table 18, the exact
/// inverse of [`parse_emdf_payloads_substream`].
///
/// Each payload's `emdf_payload_id` must be `>= 1` (id 0 is the
/// terminator and is appended automatically) and is written with the
/// 5-bit-base / `variable_bits(5)`-escape form. `payload_bytes.len()`
/// becomes `emdf_payload_size = variable_bits(8)`. The trailing
/// `byte_align` is emitted so the writer stops on a byte boundary.
pub fn write_emdf_payloads_substream(
    bw: &mut BitWriter,
    substream: &EmdfPayloadsSubstream,
) -> Result<()> {
    if substream.payloads.len() > MAX_EMDF_PAYLOADS {
        return Err(Error::invalid(
            "ac4: emdf_payloads_substream exceeds MAX_EMDF_PAYLOADS",
        ));
    }

    for payload in &substream.payloads {
        if payload.emdf_payload_id == 0 {
            return Err(Error::invalid(
                "ac4: emdf_payload_id 0 is reserved as the terminator",
            ));
        }
        if payload.payload_bytes.len() > MAX_EMDF_PAYLOAD_BYTES {
            return Err(Error::invalid(
                "ac4: emdf_payload_byte[] exceeds MAX_EMDF_PAYLOAD_BYTES",
            ));
        }

        // emdf_payload_id: 5-bit base, escape via variable_bits(5) when
        // the resolved id is >= 31.
        if payload.emdf_payload_id >= 31 {
            bw.write_u32(31, 5);
            write_variable_bits(bw, 5, payload.emdf_payload_id - 31);
        } else {
            bw.write_u32(payload.emdf_payload_id, 5);
        }

        write_emdf_payload_config(bw, &payload.config)?;

        write_variable_bits(bw, 8, payload.payload_bytes.len() as u32);
        for &b in &payload.payload_bytes {
            bw.write_u32(b as u32, 8);
        }
    }

    // Terminator: emdf_payload_id == 0.
    bw.write_u32(0, 5);
    // Trailing byte_align.
    bw.align_to_byte();

    Ok(())
}

/// `byte_align` — consume 0..7 zero-fill bits to reach the next byte
/// boundary.
fn byte_align(br: &mut BitReader<'_>) -> Result<()> {
    let pos = br.bit_position();
    let extra = (8 - (pos & 7)) & 7;
    if extra > 0 {
        br.skip(extra as u32)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[test]
    fn variable_bits_round_trip_small_values() {
        for v in [0u32, 1, 2, 7, 8, 15, 16, 17, 31, 32, 100, 255, 256, 1023] {
            let mut bw = BitWriter::new();
            write_variable_bits(&mut bw, 2, v);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(variable_bits(&mut br, 2).unwrap(), v, "v={v}");
        }
    }

    #[test]
    fn empty_substream_is_just_terminator() {
        // 5-bit emdf_payload_id == 0 → terminator; then byte_align.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 5);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        assert!(parsed.payloads.is_empty());
    }

    /// Build a minimal `emdf_payload_config()` with everything off,
    /// then `b_discard_unknown_payload = 1` so no further fields exist.
    fn write_minimal_config(bw: &mut BitWriter) {
        bw.write_bit(false); // b_smpoffst
        bw.write_bit(false); // b_duration
        bw.write_bit(false); // b_groupid
        bw.write_bit(false); // b_codecdata
        bw.write_bit(true); // b_discard_unknown_payload
    }

    #[test]
    fn one_payload_with_minimal_config_round_trips() {
        let mut bw = BitWriter::new();
        // emdf_payload_id = 7
        bw.write_u32(7, 5);
        write_minimal_config(&mut bw);
        // emdf_payload_size = 3 (variable_bits(8))
        write_variable_bits(&mut bw, 8, 3);
        // payload bytes
        bw.write_u32(0xAA, 8);
        bw.write_u32(0xBB, 8);
        bw.write_u32(0xCC, 8);
        // terminator
        bw.write_u32(0, 5);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        assert_eq!(parsed.payloads.len(), 1);
        let p = &parsed.payloads[0];
        assert_eq!(p.emdf_payload_id, 7);
        assert_eq!(p.payload_bytes, vec![0xAA, 0xBB, 0xCC]);
        assert!(p.config.b_discard_unknown_payload);
        assert!(p.config.priority.is_none());
        assert!(p.config.proc_allowed.is_none());
    }

    #[test]
    fn extended_payload_id_via_variable_bits_5() {
        // base id = 31 triggers the extension. variable_bits(5) value
        // = 9 → emdf_payload_id = 31 + 9 = 40.
        let mut bw = BitWriter::new();
        bw.write_u32(31, 5);
        write_variable_bits(&mut bw, 5, 9);
        write_minimal_config(&mut bw);
        write_variable_bits(&mut bw, 8, 0); // empty payload
        bw.write_u32(0, 5); // terminator
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        assert_eq!(parsed.payloads.len(), 1);
        assert_eq!(parsed.payloads[0].emdf_payload_id, 40);
        assert!(parsed.payloads[0].payload_bytes.is_empty());
    }

    #[test]
    fn config_with_all_optional_fields_round_trips() {
        // b_smpoffst = 1 with smpoffst = 5 → priority/proc_allowed
        // gated on (smpoffst || aligned).
        let mut bw = BitWriter::new();
        bw.write_u32(1, 5); // payload id
        bw.write_bit(true); // b_smpoffst
        write_variable_bits(&mut bw, 11, 5); // smpoffst = 5
        bw.write_bit(true); // b_duration
        write_variable_bits(&mut bw, 11, 12); // duration = 12
        bw.write_bit(true); // b_groupid
        write_variable_bits(&mut bw, 2, 2); // groupid = 2
        bw.write_bit(true); // b_codecdata
        bw.write_u32(0xA5, 8); // codecdata
        bw.write_bit(false); // b_discard_unknown_payload
                             // b_smpoffst is 1, so the inner "if (b_smpoffst == 0)" branch
                             // is skipped — `b_payload_frame_aligned` is NOT in the bitstream.
                             // Then the `if (b_smpoffst == 1 || ...)` branch fires:
        bw.write_u32(11, 5); // priority
        bw.write_u32(2, 2); // proc_allowed

        write_variable_bits(&mut bw, 8, 1); // size = 1
        bw.write_u32(0xEE, 8);

        bw.write_u32(0, 5); // terminator
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        assert_eq!(parsed.payloads.len(), 1);
        let cfg = &parsed.payloads[0].config;
        assert!(cfg.b_smpoffst);
        assert_eq!(cfg.smpoffst, Some(5));
        assert!(cfg.b_duration);
        assert_eq!(cfg.duration, Some(12));
        assert!(cfg.b_groupid);
        assert_eq!(cfg.groupid, Some(2));
        assert!(cfg.b_codecdata);
        assert_eq!(cfg.codecdata, Some(0xA5));
        assert!(!cfg.b_discard_unknown_payload);
        // b_smpoffst was 1 so frame_aligned wasn't read.
        assert!(cfg.b_payload_frame_aligned.is_none());
        assert_eq!(cfg.priority, Some(11));
        assert_eq!(cfg.proc_allowed, Some(2));
        assert_eq!(parsed.payloads[0].payload_bytes, vec![0xEE]);
    }

    #[test]
    fn frame_aligned_branch_with_create_remove_duplicate() {
        // b_smpoffst = 0, b_discard_unknown_payload = 0,
        // b_payload_frame_aligned = 1 → both create + remove duplicate
        // bits + priority + proc_allowed all appear.
        let mut bw = BitWriter::new();
        bw.write_u32(2, 5); // payload id
        bw.write_bit(false); // b_smpoffst
        bw.write_bit(false); // b_duration
        bw.write_bit(false); // b_groupid
        bw.write_bit(false); // b_codecdata
        bw.write_bit(false); // b_discard_unknown_payload
        bw.write_bit(true); // b_payload_frame_aligned
        bw.write_bit(true); // b_create_duplicate
        bw.write_bit(false); // b_remove_duplicate
        bw.write_u32(7, 5); // priority
        bw.write_u32(1, 2); // proc_allowed

        write_variable_bits(&mut bw, 8, 0); // empty payload

        bw.write_u32(0, 5); // terminator
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        let cfg = &parsed.payloads[0].config;
        assert_eq!(cfg.b_payload_frame_aligned, Some(true));
        assert_eq!(cfg.b_create_duplicate, Some(true));
        assert_eq!(cfg.b_remove_duplicate, Some(false));
        assert_eq!(cfg.priority, Some(7));
        assert_eq!(cfg.proc_allowed, Some(1));
    }

    #[test]
    fn multiple_payloads_in_sequence() {
        let mut bw = BitWriter::new();
        for (id, payload) in [(1u32, vec![0x01, 0x02]), (5u32, vec![0xFF])] {
            bw.write_u32(id, 5);
            write_minimal_config(&mut bw);
            write_variable_bits(&mut bw, 8, payload.len() as u32);
            for b in &payload {
                bw.write_u32(*b as u32, 8);
            }
        }
        bw.write_u32(0, 5); // terminator
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        assert_eq!(parsed.payloads.len(), 2);
        assert_eq!(parsed.payloads[0].emdf_payload_id, 1);
        assert_eq!(parsed.payloads[0].payload_bytes, vec![0x01, 0x02]);
        assert_eq!(parsed.payloads[1].emdf_payload_id, 5);
        assert_eq!(parsed.payloads[1].payload_bytes, vec![0xFF]);
    }

    #[test]
    fn malformed_oversize_payload_count_rejected() {
        // Build > MAX_EMDF_PAYLOADS payloads (each minimal) — parser
        // should bail before exhausting the buffer.
        let mut bw = BitWriter::new();
        for _ in 0..(MAX_EMDF_PAYLOADS + 1) {
            bw.write_u32(1, 5);
            write_minimal_config(&mut bw);
            write_variable_bits(&mut bw, 8, 0);
        }
        bw.write_u32(0, 5);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let err = parse_emdf_payloads_substream(&mut br).unwrap_err();
        assert!(err.to_string().contains("MAX_EMDF_PAYLOADS"), "got: {err}");
    }

    /// Encode a substream, decode it back, assert structural equality.
    fn assert_substream_round_trips(substream: &EmdfPayloadsSubstream) {
        let mut bw = BitWriter::new();
        write_emdf_payloads_substream(&mut bw, substream).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_emdf_payloads_substream(&mut br).unwrap();
        assert_eq!(&parsed, substream);
    }

    #[test]
    fn write_empty_substream_round_trips() {
        assert_substream_round_trips(&EmdfPayloadsSubstream::default());
    }

    #[test]
    fn write_minimal_payload_round_trips() {
        let substream = EmdfPayloadsSubstream {
            payloads: vec![EmdfPayload {
                emdf_payload_id: 7,
                config: EmdfPayloadConfig {
                    b_discard_unknown_payload: true,
                    ..Default::default()
                },
                payload_bytes: vec![0xAA, 0xBB, 0xCC],
            }],
        };
        assert_substream_round_trips(&substream);
    }

    #[test]
    fn write_extended_id_round_trips() {
        // id 40 (>= 31) exercises the variable_bits(5) escape on write.
        let substream = EmdfPayloadsSubstream {
            payloads: vec![EmdfPayload {
                emdf_payload_id: 40,
                config: EmdfPayloadConfig {
                    b_discard_unknown_payload: true,
                    ..Default::default()
                },
                payload_bytes: Vec::new(),
            }],
        };
        assert_substream_round_trips(&substream);
    }

    #[test]
    fn write_full_config_round_trips() {
        // b_smpoffst gate → priority/proc_allowed present, no
        // frame-aligned bit.
        let substream = EmdfPayloadsSubstream {
            payloads: vec![EmdfPayload {
                emdf_payload_id: 1,
                config: EmdfPayloadConfig {
                    b_smpoffst: true,
                    smpoffst: Some(5),
                    b_duration: true,
                    duration: Some(12),
                    b_groupid: true,
                    groupid: Some(2),
                    b_codecdata: true,
                    codecdata: Some(0xA5),
                    b_discard_unknown_payload: false,
                    b_payload_frame_aligned: None,
                    b_create_duplicate: None,
                    b_remove_duplicate: None,
                    priority: Some(11),
                    proc_allowed: Some(2),
                },
                payload_bytes: vec![0xEE],
            }],
        };
        assert_substream_round_trips(&substream);
    }

    #[test]
    fn write_frame_aligned_config_round_trips() {
        let substream = EmdfPayloadsSubstream {
            payloads: vec![EmdfPayload {
                emdf_payload_id: 2,
                config: EmdfPayloadConfig {
                    b_discard_unknown_payload: false,
                    b_payload_frame_aligned: Some(true),
                    b_create_duplicate: Some(true),
                    b_remove_duplicate: Some(false),
                    priority: Some(7),
                    proc_allowed: Some(1),
                    ..Default::default()
                },
                payload_bytes: Vec::new(),
            }],
        };
        assert_substream_round_trips(&substream);
    }

    #[test]
    fn write_frame_aligned_false_no_priority_round_trips() {
        // !discard, !smpoffst, frame_aligned = 0 → priority/proc_allowed
        // are NOT in the bitstream.
        let substream = EmdfPayloadsSubstream {
            payloads: vec![EmdfPayload {
                emdf_payload_id: 3,
                config: EmdfPayloadConfig {
                    b_discard_unknown_payload: false,
                    b_payload_frame_aligned: Some(false),
                    priority: None,
                    proc_allowed: None,
                    ..Default::default()
                },
                payload_bytes: vec![0x10, 0x20],
            }],
        };
        assert_substream_round_trips(&substream);
    }

    #[test]
    fn write_multiple_payloads_round_trips() {
        let substream = EmdfPayloadsSubstream {
            payloads: vec![
                EmdfPayload {
                    emdf_payload_id: 1,
                    config: EmdfPayloadConfig {
                        b_discard_unknown_payload: true,
                        ..Default::default()
                    },
                    payload_bytes: vec![0x01, 0x02],
                },
                EmdfPayload {
                    emdf_payload_id: 5,
                    config: EmdfPayloadConfig {
                        b_discard_unknown_payload: true,
                        ..Default::default()
                    },
                    payload_bytes: vec![0xFF],
                },
            ],
        };
        assert_substream_round_trips(&substream);
    }

    #[test]
    fn write_round_trips_against_existing_decoded_fixtures() {
        // Re-decode each of the hand-built fixtures from the decode tests,
        // re-encode, and re-decode — asserting the encoder reproduces a
        // parse-equivalent bitstream (write∘parse∘write∘parse identity).
        let mut bw = BitWriter::new();
        bw.write_u32(1, 5);
        bw.write_bit(true); // b_smpoffst
        write_variable_bits(&mut bw, 11, 5);
        bw.write_bit(false); // b_duration
        bw.write_bit(false); // b_groupid
        bw.write_bit(false); // b_codecdata
        bw.write_bit(false); // b_discard_unknown_payload
        bw.write_u32(11, 5); // priority
        bw.write_u32(2, 2); // proc_allowed
        write_variable_bits(&mut bw, 8, 1);
        bw.write_u32(0xEE, 8);
        bw.write_u32(0, 5); // terminator
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let first = parse_emdf_payloads_substream(&mut br).unwrap();
        assert_substream_round_trips(&first);
    }

    #[test]
    fn write_rejects_inconsistent_config() {
        // b_smpoffst set but smpoffst None must error, not panic.
        let cfg = EmdfPayloadConfig {
            b_smpoffst: true,
            smpoffst: None,
            ..Default::default()
        };
        let mut bw = BitWriter::new();
        let err = write_emdf_payload_config(&mut bw, &cfg).unwrap_err();
        assert!(err.to_string().contains("smpoffst"), "got: {err}");
    }

    #[test]
    fn write_rejects_terminator_id() {
        let substream = EmdfPayloadsSubstream {
            payloads: vec![EmdfPayload {
                emdf_payload_id: 0,
                config: EmdfPayloadConfig::default(),
                payload_bytes: Vec::new(),
            }],
        };
        let mut bw = BitWriter::new();
        let err = write_emdf_payloads_substream(&mut bw, &substream).unwrap_err();
        assert!(err.to_string().contains("terminator"), "got: {err}");
    }
}
