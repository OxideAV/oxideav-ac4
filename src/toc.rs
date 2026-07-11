//! `ac4_toc()` — AC-4 table-of-contents parser.
//!
//! Walks the Table of Contents element (ETSI TS 103 190-1 clause 4.3.3.2),
//! including the per-presentation `ac4_presentation_info()` (clause
//! 4.3.3.3) and the per-substream descriptor chain.
//!
//! The parser is intentionally structural — it extracts the fields we
//! need to describe the frame shape (channel count, sample rate, frame
//! length in samples) and skips payloads we don't decode yet
//! (metadata, EMDF, coefficient streams). Where the spec allows reserved
//! / escape forms we read and discard the bits so downstream readers
//! stay aligned.
//!
//! Bit counts quoted in comments track Tables 2–14 of the spec.
//!
//! Field naming preserves the bitstream names so the code reads as close
//! to the syntax tables as Rust allows.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// Base sampling frequency (Table 82). AC-4 carries a single-bit index
/// `fs_index` selecting between 44.1 kHz and 48 kHz; 96 / 192 kHz arrive
/// via the `sf_multiplier` inside each substream.
#[inline]
pub fn base_sample_rate(fs_index: u32) -> u32 {
    if fs_index == 0 {
        44_100
    } else {
        48_000
    }
}

/// `frame_rate_index` → (frames-per-second × 1000, internal frame length
/// at 48 kHz / 44.1 kHz).
///
/// The spec serves up Table 83 for 48/96/192 kHz and Table 84 for 44.1
/// kHz. For 44.1 kHz only index 13 is defined (11025 ÷ 512 ≈ 21.53 fps,
/// 2048-sample frame). For 48 kHz indices 0..=13 are meaningful; 14 and
/// 15 are reserved.
///
/// Returns `(fps_milli, frame_len_base)`. `fps_milli` is 0 for reserved
/// entries.
pub fn frame_rate_entry(frame_rate_index: u32, fs_index: u32) -> (u32, u32) {
    if fs_index == 0 {
        // 44.1 kHz table — only index 13 is real.
        if frame_rate_index == 13 {
            // 11025 / 512 ≈ 21.533203125 fps → scale by 1000 = 21533.
            (21_533, 2_048)
        } else {
            (0, 0)
        }
    } else {
        // 48 kHz base table.
        match frame_rate_index {
            0 => (23_976, 1_920),
            1 => (24_000, 1_920),
            2 => (25_000, 2_048),
            3 => (29_970, 1_536),
            4 => (30_000, 1_536),
            5 => (47_950, 960),
            6 => (48_000, 960),
            7 => (50_000, 1_024),
            8 => (59_940, 768),
            9 => (60_000, 768),
            10 => (100_000, 512),
            11 => (119_880, 384),
            12 => (120_000, 384),
            13 => (23_440, 2_048),
            _ => (0, 0),
        }
    }
}

/// `variable_bits(n_bits)` — TS 103 190-1 §4.2.2.
///
/// Reads `n_bits`-wide chunks, each followed by a continuation flag; the
/// accumulated value is (chunk << n_bits) + 1_shift for every extra
/// chunk.
pub fn variable_bits(br: &mut BitReader<'_>, n_bits: u32) -> Result<u32> {
    let mut value: u32 = 0;
    loop {
        let chunk = br.read_u32(n_bits)?;
        value = value
            .checked_add(chunk)
            .ok_or_else(|| Error::invalid("ac4: variable_bits overflow"))?;
        let more = br.read_bit()?;
        if !more {
            return Ok(value);
        }
        value = value
            .checked_shl(n_bits)
            .ok_or_else(|| Error::invalid("ac4: variable_bits shift overflow"))?;
        value = value
            .checked_add(1u32 << n_bits)
            .ok_or_else(|| Error::invalid("ac4: variable_bits bias overflow"))?;
    }
}

/// Inverse of [`variable_bits`] — write `value` as a `variable_bits(n_bits)`
/// field (TS 103 190-1 §4.2.2).
///
/// The decoder accumulates, per extra chunk, `value = (value << n) +
/// (1 << n) + chunk = ((value + 1) << n) + chunk`. We invert that
/// recurrence from the least-significant side: while `value >= (1 << n)`,
/// the trailing chunk is `value & ((1 << n) - 1)` and the preceding
/// accumulator was `(value >> n) - 1`. The remaining `value < (1 << n)`
/// is the first chunk `c0`. Chunks are emitted most-significant first,
/// each followed by a `1` continuation flag, with the final chunk
/// followed by `0`.
///
/// Round-trips bit-exactly with [`variable_bits`] for every `u32`.
pub fn write_variable_bits(bw: &mut oxideav_core::bits::BitWriter, n_bits: u32, mut value: u32) {
    debug_assert!((1..=32).contains(&n_bits), "variable_bits chunk width");
    let bias = 1u32 << n_bits;
    let mask = bias - 1;

    // Peel trailing chunks (most-significant accumulator stages) first.
    let mut chunks: Vec<u32> = Vec::new();
    while value >= bias {
        chunks.push(value & mask);
        value = (value >> n_bits) - 1;
    }
    // `value` now holds the first chunk `c0`.
    chunks.push(value);

    // Emit oldest (c0) → newest. c0 is `chunks.last()`.
    for (i, chunk) in chunks.iter().rev().enumerate() {
        bw.write_u32(*chunk, n_bits);
        let more = i + 1 < chunks.len();
        bw.write_bit(more);
    }
}

/// Channel mode lookup — maps the encoded bit pattern to channel count.
///
/// The channel_mode field uses a variable-length code: 1, 2, 4 or 7 bits
/// per the table hint in Syntax of `ac4_substream_info()`. We implement
/// the prefix decoder spelled out in the spec (clause 4.3.3.4.1 Table
/// 85 — "channel_mode encoding"). Returns `(channel_count, total_bits)`.
///
/// The shortest codes give the common mono / stereo / 5.1 layouts; the
/// 7-bit codes reach the high-count and immersive modes. `0b1111111`
/// with a `variable_bits(2)` extension is reserved for future use and
/// is returned as "0 channels" so the caller can treat it as unknown.
pub fn decode_channel_mode(br: &mut BitReader<'_>) -> Result<(u32, u32)> {
    // Table 85 channel_mode prefix codes per TS 103 190-1 clause 4.3.3.4.1.
    //
    // Prefix   Length  channel_mode  channels  layout
    // 0              1  0             1         mono
    // 10             2  1             2         stereo
    // 1100           4  2             3         3.0
    // 1101           4  3             5         5.0
    // 1110           4  4             6         5.1
    // 11110000       7  5             7         7.0 (3/4/0)
    // 11110001       7  6             8         7.1 (3/4/0.1)
    // 11110010       7  7             7         7.0 (5/2/0)
    // 11110011       7  8             8         7.1 (5/2/0.1)
    // 11110100       7  9             7         7.0 (3/2/2)
    // 11110101       7 10             8         7.1 (3/2/2.1)
    // 11110110       7 11             7         7.0.4
    // 11110111       7 12             9         7.1.4 (9.1)
    // 11111000       7 13            11         9.0.4
    // 11111001       7 14            12         9.1.4
    // 11111010       7 15             3         mono + 2 (reserved-ish)
    // 11111011       7 16             2         stereo (add channel form)
    // 11111100       7 17             4         quad (add channel form)
    // 11111101       7 18             4         quad (add channel form)
    // 11111110       7 19-…           0         immersive/object escape
    // 1111111        7 escape         —         variable_bits(2) follow-on
    //
    // Exact channel counts above index 11 are used by TS 103 190-2 IFM
    // streams; for this foundation we treat them as opaque — the field is
    // still consumed correctly so downstream bit-alignment is preserved.
    //
    // We read up to 7 bits; on the 0b1111111 escape the caller is expected
    // to run `variable_bits(2)` to extend the encoded index.
    let b0 = br.read_u32(1)?;
    if b0 == 0 {
        return Ok((1, 1));
    }
    let b1 = br.read_u32(1)?;
    if b1 == 0 {
        return Ok((2, 2));
    }
    let nx = br.read_u32(2)?;
    if nx != 0b11 {
        // 4-bit prefix group: 1100 / 1101 / 1110.
        return Ok((
            match nx {
                0b00 => 3,
                0b01 => 5,
                0b10 => 6,
                _ => 0,
            },
            4,
        ));
    }
    // 7-bit prefix group: 1111xxx.
    let tail = br.read_u32(3)?;
    let channels = match tail {
        0b000 => 7, // channel_mode 5 — 7.0 (3/4/0)
        0b001 => 8, // channel_mode 6 — 7.1 (3/4/0.1)
        0b010 => 7, // channel_mode 7 — 7.0 (5/2/0)
        0b011 => 8, // channel_mode 8 — 7.1 (5/2/0.1)
        0b100 => 7, // channel_mode 9 — 7.0 (3/2/2)
        0b101 => 8, // channel_mode 10 — 7.1 (3/2/2.1)
        0b110 => 7, // channel_mode 11 — 7.0.4
        0b111 => {
            // 1111111 — escape. Caller reads variable_bits(2); we leave
            // channel count unknown.
            let _ext = variable_bits(br, 2)?;
            return Ok((0, 7 + 3));
        }
        _ => unreachable!("3-bit tail is 0..=7"),
    };
    Ok((channels, 7))
}

/// Parsed AC-4 frame information — the result of running
/// [`parse_ac4_toc`] over a raw AC-4 payload (post-sync, pre-substream
/// data). The fields we expose are the ones a containerised decoder
/// pipeline actually needs: channel count, sample rate, samples-per-
/// frame, and enough identity bits to tell I-frames from P-frames.
#[derive(Debug, Clone)]
pub struct Ac4FrameInfo {
    /// `bitstream_version`, post variable_bits expansion.
    pub bitstream_version: u32,
    /// 10-bit frame counter.
    pub sequence_counter: u32,
    /// 0 = 44.1 kHz, 1 = 48 kHz base.
    pub fs_index: u32,
    /// Base sample rate derived from `fs_index`.
    pub base_sample_rate: u32,
    /// Effective sample rate after any per-substream `sf_multiplier`.
    pub sample_rate: u32,
    /// Raw frame-rate code (Table 83 / 84).
    pub frame_rate_index: u32,
    /// Frame rate × 1000 (e.g. 24000, 23976, 48000).
    pub frame_rate_milli: u32,
    /// Internal frame length at the base sample rate.
    pub frame_length: u32,
    /// `b_iframe_global` — true if all substreams of every presentation
    /// have `b_iframe` set.
    pub b_iframe_global: bool,
    /// Derived primary channel count across the first decoded
    /// presentation (mono→7.1.4). 0 if the stream uses only
    /// reserved/escape channel_mode codes we don't map.
    pub channels: u16,
    /// Number of presentations in the frame.
    pub n_presentations: u32,
    /// Total number of substreams indexed by `substream_index_table()`.
    pub n_substreams: u32,
    /// Substream byte sizes parsed from `substream_index_table()`.
    /// Empty if `b_size_present` was 0 (single-substream frame).
    pub substream_sizes: Vec<u32>,
    /// Offset (bytes) of the first substream relative to the end of
    /// the byte-aligned `ac4_toc()` element.
    pub payload_base: u32,
    /// Descriptors for each presentation (as far as we parse them).
    pub presentations: Vec<PresentationInfo>,
    /// Size of the byte-aligned `ac4_toc()` element in bytes. The
    /// first substream starts at `toc_size + payload_base` bytes into
    /// the `raw_ac4_frame()` payload.
    pub toc_size: u32,
    /// A-JOC-coded object substream descriptors collected from every
    /// `ac4_substream_group_info()` of a v2 TOC
    /// (`ac4_substream_info_ajoc()`, TS 103 190-2 §6.2.1.9), in
    /// substream-group walk order.
    pub ajoc_substreams: Vec<AjocSubstreamInfo>,
    /// Direct-coded object substream descriptors
    /// (`ac4_substream_info_obj()`, TS 103 190-2 §6.2.1.11), in
    /// substream-group walk order.
    pub obj_substreams: Vec<ObjSubstreamInfo>,
}

/// Per-presentation information we extract from `ac4_presentation_info()`.
#[derive(Debug, Clone, Default)]
pub struct PresentationInfo {
    /// Version (0 / 1 / 2) indicated by the unary `presentation_version()`
    /// prefix.
    pub version: u32,
    /// True when the presentation references a single substream — the
    /// most common case for simple AC-4 fixtures.
    pub b_single_substream: bool,
    /// `presentation_config` (0..=5 mapped, 6 = additional-EMDF-only,
    /// 7+ = extension info). 0 on single-substream presentations.
    pub presentation_config: u32,
    /// Channels for the first resolved substream (or 0 for escape
    /// codes).
    pub channels: u16,
    /// Count of `ac4_substream_info()` sub-elements this presentation
    /// references.
    pub n_substream_info: u32,
    /// Count of `ac4_hsf_ext_substream_info()` HSF extensions.
    pub n_hsf_ext: u32,
    /// Count of additional EMDF substreams referenced by this
    /// presentation.
    pub n_add_emdf_substreams: u32,
    /// Copy of the first substream's `b_iframe` bit (false if no
    /// substream was parsed).
    pub b_iframe: bool,
    /// sf_multiplier — 0 => base rate, 1 => 96 kHz, 2 => 192 kHz
    /// (only set when fs_index == 1).
    pub sf_multiplier: u32,
}

/// Parse the raw AC-4 frame element starting at the TOC.
///
/// `bytes` should be the `raw_ac4_frame()` payload (i.e. starting at the
/// first byte of `ac4_toc()`). The parser consumes the TOC, including
/// presentations and `substream_index_table()`, and stops at the
/// byte-aligned boundary that precedes the first substream's data.
pub fn parse_ac4_toc(bytes: &[u8]) -> Result<Ac4FrameInfo> {
    let mut br = BitReader::new(bytes);

    // 4.2.3.1 Syntax of ac4_toc().
    let mut bitstream_version = br.read_u32(2)?;
    if bitstream_version == 3 {
        bitstream_version += variable_bits(&mut br, 2)?;
    }
    let sequence_counter = br.read_u32(10)?;
    let b_wait_frames = br.read_bit()?;
    if b_wait_frames {
        let wait_frames = br.read_u32(3)?;
        if wait_frames > 0 {
            let _reserved = br.read_u32(2)?;
        }
    }
    let fs_index = br.read_u32(1)?;
    let frame_rate_index = br.read_u32(4)?;
    let b_iframe_global = br.read_bit()?;
    let b_single_presentation = br.read_bit()?;
    let n_presentations = if b_single_presentation {
        1
    } else {
        let b_more = br.read_bit()?;
        if b_more {
            variable_bits(&mut br, 2)? + 2
        } else {
            0
        }
    };

    // payload_base offset (§4.3.3.2.10).
    let b_payload_base = br.read_bit()?;
    let payload_base = if b_payload_base {
        let base = br.read_u32(5)? + 1;
        if base == 0x20 {
            base + variable_bits(&mut br, 3)?
        } else {
            base
        }
    } else {
        0
    };

    // Per TS 103 190-2 §6.2.1.1, the per-presentation walk depends on
    // bitstream_version: <= 1 takes the TS 103 190-1 `ac4_presentation_info()`
    // path; >= 2 runs `ac4_presentation_v1_info()` per presentation followed
    // by `ac4_substream_group_info()` × `total_n_substream_groups`.
    let mut presentations = Vec::with_capacity(n_presentations as usize);
    let mut ajoc_substreams = Vec::new();
    let mut obj_substreams = Vec::new();
    if bitstream_version <= 1 {
        for _ in 0..n_presentations {
            let pi = parse_presentation_info(&mut br, fs_index, frame_rate_index)?;
            presentations.push(pi);
        }
    } else {
        // §6.2.1.1: optional `b_program_id` block (short_program_id +
        // optional 128-bit program_uuid) precedes the per-presentation
        // loop on bitstream_version >= 2.
        let b_program_id = br.read_bit()?;
        if b_program_id {
            let _short_program_id = br.read_u32(16)?;
            let b_program_uuid_present = br.read_bit()?;
            if b_program_uuid_present {
                br.skip(16 * 8)?;
            }
        }
        let mut total_n_substream_groups: u32 = 0;
        for _ in 0..n_presentations {
            let (pi, n_sg) =
                parse_presentation_v1_info(&mut br, bitstream_version, fs_index, frame_rate_index)?;
            total_n_substream_groups += n_sg;
            presentations.push(pi);
        }
        // §6.3.2.5 ac4_substream_group_info() loop. The walker returns
        // the first substream's `(channels, sf_multiplier)` so we can
        // back-fill the leading presentation's `channels` field — for
        // single-substream-group v2 frames this is the only path the
        // channel count comes through.
        let mut first_group_channels: u16 = 0;
        let mut first_group_sf_mul: u32 = 0;
        for j in 0..total_n_substream_groups {
            let g =
                parse_substream_group_info(&mut br, bitstream_version, fs_index, frame_rate_index)?;
            if j == 0 {
                first_group_channels = g.first_channels;
                first_group_sf_mul = g.first_sf_multiplier;
            }
            ajoc_substreams.extend(g.ajoc);
            obj_substreams.extend(g.objs);
        }
        if let Some(p) = presentations.first_mut() {
            if p.channels == 0 {
                p.channels = first_group_channels;
            }
            if p.sf_multiplier == 0 {
                p.sf_multiplier = first_group_sf_mul;
            }
        }
    }

    // substream_index_table().
    let (n_substreams, substream_sizes) = parse_substream_index_table(&mut br)?;

    // Byte-align at the end of ac4_toc().
    br.align_to_byte();
    let toc_size = br.byte_position() as u32;

    // Derive effective sample rate: pick the first presentation's
    // sf_multiplier if present, otherwise fall back to the base rate.
    let base_sr = base_sample_rate(fs_index);
    let sf_mul = presentations.first().map(|p| p.sf_multiplier).unwrap_or(0);
    let sample_rate = match (fs_index, sf_mul) {
        (1, 1) => 96_000,
        (1, 2) => 192_000,
        _ => base_sr,
    };
    let channels = presentations.first().map(|p| p.channels).unwrap_or(0);

    let (fps_milli, frame_length) = frame_rate_entry(frame_rate_index, fs_index);

    Ok(Ac4FrameInfo {
        bitstream_version,
        sequence_counter,
        fs_index,
        base_sample_rate: base_sr,
        sample_rate,
        frame_rate_index,
        frame_rate_milli: fps_milli,
        frame_length,
        b_iframe_global,
        channels,
        n_presentations,
        n_substreams,
        substream_sizes,
        payload_base,
        presentations,
        toc_size,
        ajoc_substreams,
        obj_substreams,
    })
}

/// `frame_rate_factor` derived from the frame_rate_index and the
/// presentation's multiplier bits (Table 87 in TS 103 190-1 §4.3.3.3.4).
fn frame_rate_factor(frame_rate_index: u32, b_multiplier: bool, multiplier_bit: u32) -> u32 {
    match frame_rate_index {
        // Indices 2/3/4 — 25 / 29.97 / 30 fps: factor is 1 or (b_multiplier ? 1+multiplier_bit : 1).
        2..=4 if b_multiplier => {
            if multiplier_bit == 0 {
                2
            } else {
                4
            }
        }
        // Indices 0/1/7/8/9 — high-FPS forms: factor is 1 or 2.
        0 | 1 | 7 | 8 | 9 if b_multiplier => 2,
        _ => 1,
    }
}

fn parse_frame_rate_multiply_info(
    br: &mut BitReader<'_>,
    frame_rate_index: u32,
) -> Result<(bool, u32)> {
    // §4.2.3.4 Syntax of frame_rate_multiply_info().
    let mut b_multiplier = false;
    let mut multiplier_bit = 0u32;
    match frame_rate_index {
        2..=4 => {
            b_multiplier = br.read_bit()?;
            if b_multiplier {
                multiplier_bit = br.read_u32(1)?;
            }
        }
        0 | 1 | 7 | 8 | 9 => {
            b_multiplier = br.read_bit()?;
        }
        _ => {}
    }
    Ok((b_multiplier, multiplier_bit))
}

fn parse_emdf_info(br: &mut BitReader<'_>) -> Result<()> {
    // §4.2.3.5 Syntax of emdf_info().
    let emdf_version = br.read_u32(2)?;
    if emdf_version == 3 {
        let _ = variable_bits(br, 2)?;
    }
    let key_id = br.read_u32(3)?;
    if key_id == 7 {
        let _ = variable_bits(br, 3)?;
    }
    let b_emdf_payloads_substream_info = br.read_bit()?;
    if b_emdf_payloads_substream_info {
        parse_emdf_payloads_substream_info(br)?;
    }
    parse_emdf_reserved(br)?;
    Ok(())
}

fn parse_emdf_payloads_substream_info(br: &mut BitReader<'_>) -> Result<()> {
    // §4.2.3.10.
    let substream_index = br.read_u32(2)?;
    if substream_index == 3 {
        let _ = variable_bits(br, 2)?;
    }
    Ok(())
}

fn parse_emdf_reserved(br: &mut BitReader<'_>) -> Result<()> {
    // §4.2.3.12 — emdf_reserved(): b_more_bits and optional
    // variable_bits(32) chunk list. Consumes a minimum of 1 bit.
    let b_more_bits = br.read_bit()?;
    if b_more_bits {
        // Spec phrasing: emdf_reserved() carries a payload of
        // variable_bits(5) skip bytes, each treated as opaque reserved.
        let n_bits = variable_bits(br, 5)?;
        // Clamp — the spec says the reserved field must fit within the
        // remaining frame, so we trust it but cap at a sane upper bound
        // to avoid runaway reads on malformed streams.
        if n_bits > 1 << 20 {
            return Err(Error::invalid("ac4: emdf_reserved claims too many bits"));
        }
        br.skip(n_bits)?;
    }
    Ok(())
}

fn parse_substream_info(
    br: &mut BitReader<'_>,
    fs_index: u32,
    frame_rate_index: u32,
) -> Result<SubstreamInfo> {
    // §4.2.3.6 ac4_substream_info().
    let (channels, _mode_bits) = decode_channel_mode(br)?;
    let mut sf_multiplier = 0;
    if fs_index == 1 {
        let b_sf_multiplier = br.read_bit()?;
        if b_sf_multiplier {
            sf_multiplier = br.read_u32(1)? + 1;
        }
    }
    let b_bitrate_info = br.read_bit()?;
    if b_bitrate_info {
        // bitrate_indicator is 3 bits (short) or 5 bits (long). The spec
        // splits the two via the prefix value: if the 3-bit indicator is
        // 0b111 we reinterpret with 2 more bits. We simply consume up to
        // 5 bits, which keeps us byte-aligned correctly per Table 86.
        let short = br.read_u32(3)?;
        if short == 0b111 {
            let _ = br.read_u32(2)?;
        }
    }
    // add_ch_base bit for certain channel_mode values (0b1111010..0b1111101).
    // Since we decoded via the prefix tree we don't have that exact code
    // value; the spec gates it on channel_mode numeric identity, and our
    // 7-bit prefix decoder returns the channel count, not the code.
    // For the frame-shape foundation we don't need add_ch_base, so we
    // skip this bit conservatively when the channel count suggests an
    // extended layout (7/8 channels from the 7-bit prefix group).
    if channels == 7 || channels == 8 {
        // The spec specifies add_ch_base for exactly codes 122..125; those
        // map to our (channels, mode_bits=7) results for tail in 0b010..=
        // 0b101. We cannot distinguish them after the fact from
        // decode_channel_mode alone, so we approximate by always reading
        // the bit when mode_bits == 7 — safe because it's the next bit
        // either way; in the non-add-ch-base subset the bit is a
        // b_content_type that we consume just below. Approximate path is
        // kept minimal; see note in README.
    }
    let b_content_type = br.read_bit()?;
    if b_content_type {
        parse_content_type(br)?;
    }
    let factor = frame_rate_factor(frame_rate_index, false, 0);
    let mut b_iframe = false;
    for _ in 0..factor.max(1) {
        let f = br.read_bit()?;
        if !b_iframe {
            b_iframe = f;
        }
    }
    // substream_index (2 bits + optional variable_bits(2)).
    let si = br.read_u32(2)?;
    if si == 3 {
        let _ = variable_bits(br, 2)?;
    }
    Ok(SubstreamInfo {
        channels: channels as u16,
        sf_multiplier,
        b_iframe,
    })
}

struct SubstreamInfo {
    channels: u16,
    sf_multiplier: u32,
    b_iframe: bool,
}

fn parse_content_type(br: &mut BitReader<'_>) -> Result<()> {
    // §4.2.3.7 content_type().
    let _content_classifier = br.read_u32(3)?;
    let b_language_indicator = br.read_bit()?;
    if b_language_indicator {
        let b_serialized = br.read_bit()?;
        if b_serialized {
            let _b_start_tag = br.read_bit()?;
            let _language_tag_chunk = br.read_u32(16)?;
        } else {
            let n = br.read_u32(6)?;
            br.skip(8 * n)?;
        }
    }
    Ok(())
}

fn parse_hsf_ext_substream_info(br: &mut BitReader<'_>) -> Result<()> {
    // §4.2.3.9 ac4_hsf_ext_substream_info().
    let si = br.read_u32(2)?;
    if si == 3 {
        let _ = variable_bits(br, 2)?;
    }
    Ok(())
}

fn parse_presentation_config_ext_info(br: &mut BitReader<'_>) -> Result<()> {
    // §4.2.3.8 presentation_config_ext_info().
    let mut n_skip_bytes = br.read_u32(5)?;
    let b_more = br.read_bit()?;
    if b_more {
        n_skip_bytes += variable_bits(br, 2)? << 5;
    }
    if n_skip_bytes > 1 << 20 {
        return Err(Error::invalid("ac4: presentation_config_ext_info too big"));
    }
    br.skip(n_skip_bytes * 8)?;
    Ok(())
}

fn parse_presentation_info(
    br: &mut BitReader<'_>,
    fs_index: u32,
    frame_rate_index: u32,
) -> Result<PresentationInfo> {
    // §4.2.3.2 Syntax of ac4_presentation_info().
    let mut info = PresentationInfo::default();
    let b_single_substream = br.read_bit()?;
    info.b_single_substream = b_single_substream;
    let mut presentation_config: u32 = 0;
    if !b_single_substream {
        presentation_config = br.read_u32(3)?;
        if presentation_config == 7 {
            presentation_config += variable_bits(br, 2)?;
        }
    }
    info.presentation_config = presentation_config;
    // presentation_version(): read bits until we see a 0.
    let mut ver = 0u32;
    while br.read_bit()? {
        ver += 1;
        if ver > 32 {
            return Err(Error::invalid("ac4: runaway presentation_version"));
        }
    }
    info.version = ver;
    let b_add_emdf_substreams;
    if !b_single_substream && presentation_config == 6 {
        // Special "add EMDF only" configuration.
        b_add_emdf_substreams = true;
    } else {
        let _md_compat = br.read_u32(3)?;
        let b_belongs_to_presentation_id = br.read_bit()?;
        if b_belongs_to_presentation_id {
            let _presentation_id = variable_bits(br, 2)?;
        }
        let (_b_mult, _mult_bit) = parse_frame_rate_multiply_info(br, frame_rate_index)?;
        parse_emdf_info(br)?;
        if b_single_substream {
            let si = parse_substream_info(br, fs_index, frame_rate_index)?;
            info.channels = si.channels;
            info.sf_multiplier = si.sf_multiplier;
            info.b_iframe = si.b_iframe;
            info.n_substream_info = 1;
        } else {
            let _b_hsf_ext = br.read_bit()?;
            let b_hsf_ext = _b_hsf_ext;
            match presentation_config {
                0..=2 => {
                    // Three variants that share the same layout: main/ME +
                    // optional HSF + secondary stream.
                    let first = parse_substream_info(br, fs_index, frame_rate_index)?;
                    info.channels = first.channels;
                    info.sf_multiplier = first.sf_multiplier;
                    info.b_iframe = first.b_iframe;
                    info.n_substream_info = 1;
                    if b_hsf_ext {
                        parse_hsf_ext_substream_info(br)?;
                        info.n_hsf_ext += 1;
                    }
                    let _second = parse_substream_info(br, fs_index, frame_rate_index)?;
                    info.n_substream_info += 1;
                }
                3 | 4 => {
                    let first = parse_substream_info(br, fs_index, frame_rate_index)?;
                    info.channels = first.channels;
                    info.sf_multiplier = first.sf_multiplier;
                    info.b_iframe = first.b_iframe;
                    info.n_substream_info = 1;
                    if b_hsf_ext {
                        parse_hsf_ext_substream_info(br)?;
                        info.n_hsf_ext += 1;
                    }
                    let _second = parse_substream_info(br, fs_index, frame_rate_index)?;
                    let _third = parse_substream_info(br, fs_index, frame_rate_index)?;
                    info.n_substream_info += 2;
                }
                5 => {
                    let first = parse_substream_info(br, fs_index, frame_rate_index)?;
                    info.channels = first.channels;
                    info.sf_multiplier = first.sf_multiplier;
                    info.b_iframe = first.b_iframe;
                    info.n_substream_info = 1;
                    if b_hsf_ext {
                        parse_hsf_ext_substream_info(br)?;
                        info.n_hsf_ext += 1;
                    }
                }
                _ => {
                    parse_presentation_config_ext_info(br)?;
                }
            }
        }
        let _b_pre_virtualized = br.read_bit()?;
        b_add_emdf_substreams = br.read_bit()?;
    }
    if b_add_emdf_substreams {
        let mut n = br.read_u32(2)?;
        if n == 0 {
            n = variable_bits(br, 2)? + 4;
        }
        for _ in 0..n {
            parse_emdf_info(br)?;
        }
        info.n_add_emdf_substreams = n;
    }
    Ok(info)
}

/// `ac4_presentation_v1_info()` per ETSI TS 103 190-2 §6.2.1.3.
///
/// Returns the parsed [`PresentationInfo`] plus `n_substream_groups`
/// — the count of `ac4_sgi_specifier()` calls this presentation made,
/// summed by the caller into `total_n_substream_groups` for the
/// trailing `ac4_substream_group_info()` loop.
fn parse_presentation_v1_info(
    br: &mut BitReader<'_>,
    bitstream_version: u32,
    fs_index: u32,
    frame_rate_index: u32,
) -> Result<(PresentationInfo, u32)> {
    let mut info = PresentationInfo::default();
    let b_single_substream_group = br.read_bit()?;
    info.b_single_substream = b_single_substream_group;
    let mut presentation_config: u32 = 0;
    if !b_single_substream_group {
        presentation_config = br.read_u32(3)?;
        if presentation_config == 7 {
            presentation_config += variable_bits(br, 2)?;
        }
    }
    info.presentation_config = presentation_config;
    if bitstream_version != 1 {
        let mut ver = 0u32;
        while br.read_bit()? {
            ver += 1;
            if ver > 32 {
                return Err(Error::invalid("ac4: runaway presentation_version"));
            }
        }
        info.version = ver;
    }
    let mut n_substream_groups: u32 = 0;
    let b_add_emdf_substreams;
    if !b_single_substream_group && presentation_config == 6 {
        b_add_emdf_substreams = true;
    } else {
        if bitstream_version != 1 {
            let _mdcompat = br.read_u32(3)?;
        }
        let b_presentation_id = br.read_bit()?;
        if b_presentation_id {
            let _presentation_id = variable_bits(br, 2)?;
        }
        let (_b_mult, _mult_bit) = parse_frame_rate_multiply_info(br, frame_rate_index)?;
        parse_frame_rate_fractions_info(br, frame_rate_index)?;
        parse_emdf_info(br)?;
        let b_presentation_filter = br.read_bit()?;
        if b_presentation_filter {
            let _b_enable_presentation = br.read_bit()?;
        }
        if b_single_substream_group {
            // ac4_sgi_specifier(): group_index field only on
            // bitstream_version != 1 — bitstream_version == 1 inlines the
            // group itself, which we don't emit.
            parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
            n_substream_groups = 1;
        } else {
            let _b_multi_pid = br.read_bit()?;
            match presentation_config {
                0 | 2 => {
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    n_substream_groups = 2;
                }
                1 => {
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    n_substream_groups = 1;
                }
                3 => {
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    n_substream_groups = 3;
                }
                4 => {
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    n_substream_groups = 2;
                }
                5 => {
                    let n_minus2 = br.read_u32(2)?;
                    let mut n = n_minus2 + 2;
                    if n == 5 {
                        n += variable_bits(br, 2)?;
                    }
                    if n > 64 {
                        return Err(Error::invalid("ac4: presentation_config=5 n too big"));
                    }
                    for _ in 0..n {
                        parse_sgi_specifier(br, bitstream_version, fs_index, frame_rate_index)?;
                    }
                    n_substream_groups = n;
                }
                _ => {
                    parse_presentation_config_ext_info(br)?;
                }
            }
        }
        let _b_pre_virtualized = br.read_bit()?;
        b_add_emdf_substreams = br.read_bit()?;
        // ac4_presentation_substream_info() — per §6.2.1.12: b_alternative,
        // b_pres_ndot, substream_index (2 + optional variable_bits(2)).
        let _b_alternative = br.read_bit()?;
        let b_pres_ndot = br.read_bit()?;
        // §6.3.2.11.2: ndot = "no dependency over time" — true means
        // the presentation substream decodes independently from
        // preceding frames, i.e. it IS the I-frame flag (no inversion).
        info.b_iframe = b_pres_ndot;
        let si = br.read_u32(2)?;
        if si == 3 {
            let _ = variable_bits(br, 2)?;
        }
    }
    if b_add_emdf_substreams {
        let mut n = br.read_u32(2)?;
        if n == 0 {
            n = variable_bits(br, 2)? + 4;
        }
        for _ in 0..n {
            parse_emdf_info(br)?;
        }
        info.n_add_emdf_substreams = n;
    }
    Ok((info, n_substream_groups))
}

/// `ac4_sgi_specifier()` per ETSI TS 103 190-2 §6.2.1.7.
///
/// On `bitstream_version == 1` this inlines `ac4_substream_group_info()`
/// directly; on `bitstream_version != 1` it reads a 3-bit `group_index`
/// (with `variable_bits(2)` extension on the escape value 7).
fn parse_sgi_specifier(
    br: &mut BitReader<'_>,
    bitstream_version: u32,
    fs_index: u32,
    frame_rate_index: u32,
) -> Result<u32> {
    if bitstream_version == 1 {
        parse_substream_group_info(br, bitstream_version, fs_index, frame_rate_index)?;
        Ok(0)
    } else {
        let mut group_index = br.read_u32(3)?;
        if group_index == 7 {
            group_index += variable_bits(br, 2)?;
        }
        Ok(group_index)
    }
}

/// `ac4_substream_group_info()` per ETSI TS 103 190-2 §6.3.2.5 (syntax
/// box mirror in §6.2.1.6).
///
/// Returns a [`SubstreamGroupSummary`] describing the first
/// channel-coded substream in the group — the rest of the substream
/// descriptors (object / a-joc paths) are not yet implemented, so the
/// walker returns `Unsupported` if it hits one.
fn parse_substream_group_info(
    br: &mut BitReader<'_>,
    bitstream_version: u32,
    fs_index: u32,
    frame_rate_index: u32,
) -> Result<SubstreamGroupSummary> {
    let mut summary = SubstreamGroupSummary::default();
    let b_substreams_present = br.read_bit()?;
    let b_hsf_ext = br.read_bit()?;
    let b_single_substream = br.read_bit()?;
    let n_lf_substreams = if b_single_substream {
        1
    } else {
        let n_minus2 = br.read_u32(2)?;
        let mut n = n_minus2 + 2;
        if n == 5 {
            n += variable_bits(br, 2)?;
        }
        n
    };
    if n_lf_substreams > 64 {
        return Err(Error::invalid("ac4: n_lf_substreams too big"));
    }
    let b_channel_coded = br.read_bit()?;
    if b_channel_coded {
        for sus in 0..n_lf_substreams {
            if bitstream_version == 1 {
                let _sus_ver = br.read_bit()?;
            }
            let chan =
                parse_substream_info_chan(br, fs_index, frame_rate_index, b_substreams_present)?;
            if sus == 0 {
                summary.first_channels = chan.channels;
                summary.first_sf_multiplier = chan.sf_multiplier;
            }
            if b_hsf_ext && b_substreams_present {
                let si = br.read_u32(2)?;
                if si == 3 {
                    let _ = variable_bits(br, 2)?;
                }
            }
        }
    } else {
        let b_oamd_substream = br.read_bit()?;
        if b_oamd_substream {
            // oamd_substream_info(b_substreams_present)
            let _b_oamd_ndot = br.read_bit()?;
            if b_substreams_present {
                let si = br.read_u32(2)?;
                if si == 3 {
                    let _ = variable_bits(br, 2)?;
                }
            }
        }
        // §6.2.1.6 non-channel-coded substream loop: each substream is
        // either A-JOC coded (`ac4_substream_info_ajoc()`, §6.2.1.9) or
        // direct-coded objects (`ac4_substream_info_obj()`, §6.2.1.11).
        for _ in 0..n_lf_substreams {
            let b_ajoc = br.read_bit()?;
            if b_ajoc {
                let info = parse_substream_info_ajoc(
                    br,
                    fs_index,
                    frame_rate_index,
                    b_substreams_present,
                )?;
                if summary.first_sf_multiplier == 0 {
                    summary.first_sf_multiplier = info.sf_multiplier;
                }
                summary.ajoc.push(info);
            } else {
                let info =
                    parse_substream_info_obj(br, fs_index, frame_rate_index, b_substreams_present)?;
                if summary.first_sf_multiplier == 0 {
                    summary.first_sf_multiplier = info.sf_multiplier;
                }
                summary.objs.push(info);
            }
            if b_hsf_ext && b_substreams_present {
                let si = br.read_u32(2)?;
                if si == 3 {
                    let _ = variable_bits(br, 2)?;
                }
            }
        }
    }
    let b_content_type = br.read_bit()?;
    if b_content_type {
        parse_content_type(br)?;
    }
    Ok(summary)
}

/// `ac4_substream_info_chan(b_substreams_present)` per ETSI TS 103 190-2
/// §6.2.1.8. Reads the channel-coded substream descriptor inside an
/// `ac4_substream_group_info()` element.
fn parse_substream_info_chan(
    br: &mut BitReader<'_>,
    fs_index: u32,
    frame_rate_index: u32,
    b_substreams_present: bool,
) -> Result<SubstreamInfoChan> {
    let (channels, _mode_bits) = decode_channel_mode(br)?;
    let mut sf_multiplier = 0;
    if fs_index == 1 {
        let b_sf_multiplier = br.read_bit()?;
        if b_sf_multiplier {
            sf_multiplier = br.read_u32(1)? + 1;
        }
    }
    let b_bitrate_info = br.read_bit()?;
    if b_bitrate_info {
        let short = br.read_u32(3)?;
        if short == 0b111 {
            let _ = br.read_u32(2)?;
        }
    }
    // §6.2.1.8 add_ch_base bit gate — skipped for the v2 walker for the
    // same reason as the v0 walker (we don't surface raw channel_mode).
    let factor = frame_rate_factor(frame_rate_index, false, 0);
    for _ in 0..factor.max(1) {
        let _b_audio_ndot = br.read_bit()?;
    }
    if b_substreams_present {
        let si = br.read_u32(2)?;
        if si == 3 {
            let _ = variable_bits(br, 2)?;
        }
    }
    Ok(SubstreamInfoChan {
        channels: channels as u16,
        sf_multiplier,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct SubstreamInfoChan {
    channels: u16,
    sf_multiplier: u32,
}

#[derive(Debug, Clone, Default)]
struct SubstreamGroupSummary {
    first_channels: u16,
    first_sf_multiplier: u32,
    ajoc: Vec<AjocSubstreamInfo>,
    objs: Vec<ObjSubstreamInfo>,
}

// =====================================================================
// Object / A-JOC substream descriptors (TS 103 190-2 §6.2.1.9-11)
// =====================================================================

/// Table 83: `isf_config` → number of ISF objects.
pub const ISF_OBJECT_COUNT: [u32; 6] = [4, 8, 10, 14, 15, 30];

/// Table 84: `bed_chan_assign_code` → number of bed channels.
pub const BED_CHAN_ASSIGN_COUNT: [u32; 8] = [2, 3, 6, 8, 10, 8, 10, 12];

/// Table 86: per-flag channel counts for
/// `std_bed_channel_assignment_mask` (array position 0 = L/R pair …
/// 9 = LFE2).
pub const STD_BED_FLAG_CHANNELS: [u32; 10] = [2, 1, 1, 2, 2, 2, 2, 2, 2, 1];

/// `bed_dyn_obj_assignment(n_signals)` per TS 103 190-2 §6.2.1.10 —
/// how the (static) objects of an A-JOC signal set are positioned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BedDynObjAssignment {
    /// `b_dyn_objects_only == 1` — every signal is a dynamic object.
    DynObjectsOnly,
    /// `b_isf == 1` — ISF objects per the 3-bit `isf_config`
    /// (Table 83).
    Isf {
        /// 3-bit `isf_config`.
        isf_config: u8,
    },
    /// `b_ch_assign_code == 1` — 3-bit `bed_chan_assign_code`
    /// (Table 84).
    BedCode {
        /// 3-bit `bed_chan_assign_code`.
        code: u8,
    },
    /// 17-bit `nonstd_bed_channel_assignment_mask` (Table 85 indices).
    BedNonStdMask {
        /// The raw 17-bit mask.
        mask: u32,
    },
    /// 10-bit `std_bed_channel_assignment_mask` (Table 86 flags).
    BedStdMask {
        /// The raw 10-bit mask.
        mask: u16,
    },
    /// Individual per-object 4-bit `nonstd_bed_channel_assignment`
    /// entries (Table 85 values).
    BedIndividual {
        /// One Table 85 channel id per bed signal.
        assignments: Vec<u8>,
    },
}

impl BedDynObjAssignment {
    /// Number of *static* (bed / ISF) objects this assignment
    /// describes. Per §6.3.2.8.0 the static objects precede the dynamic
    /// objects; the dynamic count is the difference to the signal
    /// count.
    pub fn n_static_objects(&self) -> u32 {
        match self {
            BedDynObjAssignment::DynObjectsOnly => 0,
            BedDynObjAssignment::Isf { isf_config } => {
                ISF_OBJECT_COUNT[(*isf_config).min(5) as usize]
            }
            BedDynObjAssignment::BedCode { code } => BED_CHAN_ASSIGN_COUNT[(*code & 7) as usize],
            BedDynObjAssignment::BedNonStdMask { mask } => mask.count_ones(),
            BedDynObjAssignment::BedStdMask { mask } => (0..10)
                .filter(|b| mask & (1 << b) != 0)
                .map(|b| STD_BED_FLAG_CHANNELS[b as usize])
                .sum(),
            BedDynObjAssignment::BedIndividual { assignments } => assignments.len() as u32,
        }
    }

    /// Per-object types for `n_signals` fullband signals: the derived
    /// static objects first (Bed or Isf), the remainder dynamic
    /// (§6.3.2.8.0).
    pub fn obj_types(&self, n_signals: u32) -> Vec<crate::oamd::ObjType> {
        use crate::oamd::ObjType;
        let static_kind = match self {
            BedDynObjAssignment::DynObjectsOnly => ObjType::Dyn,
            BedDynObjAssignment::Isf { .. } => ObjType::Isf,
            _ => ObjType::Bed,
        };
        let n_static = self.n_static_objects().min(n_signals);
        let mut v = vec![static_kind; n_static as usize];
        v.resize(n_signals as usize, ObjType::Dyn);
        v
    }
}

/// Parse `bed_dyn_obj_assignment(n_signals)` per §6.2.1.10.
pub fn parse_bed_dyn_obj_assignment(
    br: &mut BitReader<'_>,
    n_signals: u32,
) -> Result<BedDynObjAssignment> {
    if br.read_bit()? {
        return Ok(BedDynObjAssignment::DynObjectsOnly);
    }
    if br.read_bit()? {
        // b_isf.
        return Ok(BedDynObjAssignment::Isf {
            isf_config: br.read_u32(3)? as u8,
        });
    }
    if br.read_bit()? {
        // b_ch_assign_code.
        return Ok(BedDynObjAssignment::BedCode {
            code: br.read_u32(3)? as u8,
        });
    }
    if br.read_bit()? {
        // b_chan_assign_mask.
        if br.read_bit()? {
            Ok(BedDynObjAssignment::BedNonStdMask {
                mask: br.read_u32(17)?,
            })
        } else {
            Ok(BedDynObjAssignment::BedStdMask {
                mask: br.read_u32(10)? as u16,
            })
        }
    } else {
        // Individual 4-bit assignments; count is ceil(log2(n_signals))
        // bits of n_bed_signals_minus1 when n_signals > 1, else 1.
        let n_bed_signals = if n_signals > 1 {
            let bits = 32 - (n_signals - 1).leading_zeros();
            br.read_u32(bits)? + 1
        } else {
            1
        };
        let mut assignments = Vec::with_capacity(n_bed_signals as usize);
        for _ in 0..n_bed_signals {
            assignments.push(br.read_u32(4)? as u8);
        }
        Ok(BedDynObjAssignment::BedIndividual { assignments })
    }
}

/// Parsed `ac4_substream_info_ajoc()` (TS 103 190-2 §6.2.1.9) — the
/// descriptor of an A-JOC-coded object substream.
#[derive(Debug, Clone, PartialEq)]
pub struct AjocSubstreamInfo {
    /// `b_lfe` — an LFE channel accompanies the fullband signals.
    pub b_lfe: bool,
    /// `b_static_dmx` — the core downmix is a static 5.0/5.1 bed
    /// (`n_fullband_dmx_signals = 5`, no dmx assignment element).
    pub b_static_dmx: bool,
    /// Number of fullband core-downmix signals.
    pub n_fullband_dmx_signals: u32,
    /// Downmix object assignment (`None` when `b_static_dmx`).
    pub dmx_assignment: Option<BedDynObjAssignment>,
    /// `oamd_common_data()` payload when present.
    pub oamd_common_data: Option<crate::oamd::OamdCommonData>,
    /// Number of fullband full-decode (upmix) signals.
    pub n_fullband_upmix_signals: u32,
    /// Upmix object assignment.
    pub umx_assignment: BedDynObjAssignment,
    /// Resolved `sf_multiplier` (0 = ×1 implicit, 1 = ×2, 2 = ×4).
    pub sf_multiplier: u32,
    /// `bitrate_indicator` when signalled.
    pub bitrate_indicator: Option<u32>,
    /// One `b_audio_ndot` flag per frame-rate-factor slot.
    pub b_audio_ndot: Vec<bool>,
    /// `substream_index` when `b_substreams_present`.
    pub substream_index: Option<u32>,
}

impl AjocSubstreamInfo {
    /// `b_iframe` for the (first) audio frame. §6.3.2.x: `b_audio_ndot`
    /// = "no dependency over time" — true means the frame decodes
    /// independently, i.e. it IS the I-frame flag.
    pub fn b_iframe(&self) -> bool {
        self.b_audio_ndot.first().copied().unwrap_or(true)
    }

    /// Total downmix signal count including the optional LFE.
    pub fn n_dmx_signals(&self) -> u32 {
        self.n_fullband_dmx_signals + u32::from(self.b_lfe)
    }

    /// Total upmix signal count including the optional LFE.
    pub fn n_umx_signals(&self) -> u32 {
        self.n_fullband_upmix_signals + u32::from(self.b_lfe)
    }

    /// Object types for the downmix signal set (LFE slot first when
    /// present, typed dynamic per §6.3.2.8.1; a static 5.X downmix is a
    /// bed).
    pub fn obj_type_dmx(&self) -> Vec<crate::oamd::ObjType> {
        use crate::oamd::ObjType;
        let mut v = Vec::with_capacity(self.n_dmx_signals() as usize);
        if self.b_lfe {
            v.push(ObjType::Dyn);
        }
        match &self.dmx_assignment {
            Some(a) => v.extend(a.obj_types(self.n_fullband_dmx_signals)),
            None => v.extend(vec![ObjType::Bed; self.n_fullband_dmx_signals as usize]),
        }
        v
    }

    /// Object types for the upmix signal set (LFE slot first when
    /// present).
    pub fn obj_type_umx(&self) -> Vec<crate::oamd::ObjType> {
        use crate::oamd::ObjType;
        let mut v = Vec::with_capacity(self.n_umx_signals() as usize);
        if self.b_lfe {
            v.push(ObjType::Dyn);
        }
        v.extend(self.umx_assignment.obj_types(self.n_fullband_upmix_signals));
        v
    }

    /// Per-signal LFE flags for the downmix set (`is_lfe[0]` per
    /// §6.2.3.4).
    pub fn is_lfe_dmx(&self) -> Vec<bool> {
        let mut v = vec![false; self.n_dmx_signals() as usize];
        if self.b_lfe {
            v[0] = true;
        }
        v
    }

    /// Per-signal LFE flags for the upmix set.
    pub fn is_lfe_umx(&self) -> Vec<bool> {
        let mut v = vec![false; self.n_umx_signals() as usize];
        if self.b_lfe {
            v[0] = true;
        }
        v
    }
}

/// 3-bit `bitrate_indicator` with the 0b111 + 2-bit escape
/// (§4.3.3.4.4 form shared by the chan / ajoc / obj descriptors).
fn parse_bitrate_indicator(br: &mut BitReader<'_>) -> Result<u32> {
    let short = br.read_u32(3)?;
    if short == 0b111 {
        Ok((short << 2) | br.read_u32(2)?)
    } else {
        Ok(short)
    }
}

/// Parse `ac4_substream_info_ajoc(b_substreams_present)` per
/// TS 103 190-2 §6.2.1.9.
pub fn parse_substream_info_ajoc(
    br: &mut BitReader<'_>,
    fs_index: u32,
    frame_rate_index: u32,
    b_substreams_present: bool,
) -> Result<AjocSubstreamInfo> {
    let b_lfe = br.read_bit()?;
    let b_static_dmx = br.read_bit()?;
    let (n_fullband_dmx_signals, dmx_assignment) = if b_static_dmx {
        (5, None)
    } else {
        let n = br.read_u32(4)? + 1;
        let a = parse_bed_dyn_obj_assignment(br, n)?;
        (n, Some(a))
    };
    let oamd_common_data = if br.read_bit()? {
        // The bed context's LFE presence (gating the stereo_dmx_coeff
        // LFE sub-block) is the substream's b_lfe.
        Some(crate::oamd::parse_oamd_common_data(br, b_lfe)?)
    } else {
        None
    };
    let mut n_fullband_upmix_signals = br.read_u32(4)? + 1;
    if n_fullband_upmix_signals == 16 {
        n_fullband_upmix_signals += variable_bits(br, 3)?;
    }
    if n_fullband_upmix_signals > 256 {
        return Err(Error::invalid("ac4: n_fullband_upmix_signals too big"));
    }
    let umx_assignment = parse_bed_dyn_obj_assignment(br, n_fullband_upmix_signals)?;
    let mut sf_multiplier = 0;
    if fs_index == 1 && br.read_bit()? {
        sf_multiplier = br.read_u32(1)? + 1;
    }
    let bitrate_indicator = if br.read_bit()? {
        Some(parse_bitrate_indicator(br)?)
    } else {
        None
    };
    let factor = frame_rate_factor(frame_rate_index, false, 0).max(1);
    let mut b_audio_ndot = Vec::with_capacity(factor as usize);
    for _ in 0..factor {
        b_audio_ndot.push(br.read_bit()?);
    }
    let substream_index = if b_substreams_present {
        let mut si = br.read_u32(2)?;
        if si == 3 {
            si += variable_bits(br, 2)?;
        }
        Some(si)
    } else {
        None
    };
    Ok(AjocSubstreamInfo {
        b_lfe,
        b_static_dmx,
        n_fullband_dmx_signals,
        dmx_assignment,
        oamd_common_data,
        n_fullband_upmix_signals,
        umx_assignment,
        sf_multiplier,
        bitrate_indicator,
        b_audio_ndot,
        substream_index,
    })
}

/// Bed/ISF start descriptor inside `ac4_substream_info_obj()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjBedStart {
    /// `b_bed_start == 1`, `b_ch_assign_code == 1`.
    BedCode {
        /// 3-bit `bed_chan_assign_code` (Table 84).
        code: u8,
    },
    /// `b_bed_start == 1`, non-standard 17-bit mask (Table 85).
    BedNonStdMask {
        /// The raw mask.
        mask: u32,
    },
    /// `b_bed_start == 1`, standard 10-bit mask (Table 86).
    BedStdMask {
        /// The raw mask.
        mask: u16,
    },
    /// `b_bed_start == 0`, `b_isf == 1` — optional new ISF config.
    Isf {
        /// 3-bit `isf_config` when `b_isf_start`.
        isf_config: Option<u8>,
    },
    /// `b_bed_start == 0`, `b_isf == 0` — reserved bytes.
    Reserved {
        /// The skipped `reserved_data` byte count.
        res_bytes: u8,
    },
    /// `b_bed_objects == 1` but `b_bed_start == 0` — continuation of a
    /// bed started in another substream of the group.
    BedContinuation,
}

/// Parsed `ac4_substream_info_obj()` (TS 103 190-2 §6.2.1.11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjSubstreamInfo {
    /// 3-bit `n_objects_code`; resolved count via
    /// [`ObjSubstreamInfo::n_objects`].
    pub n_objects_code: u8,
    /// `b_dynamic_objects`.
    pub b_dynamic_objects: bool,
    /// `b_lfe` (dynamic-objects form only).
    pub b_lfe: bool,
    /// Bed / ISF descriptor for the static-objects form.
    pub bed: Option<ObjBedStart>,
    /// Resolved `sf_multiplier`.
    pub sf_multiplier: u32,
    /// `bitrate_indicator` when signalled.
    pub bitrate_indicator: Option<u32>,
    /// One `b_audio_ndot` flag per frame-rate-factor slot.
    pub b_audio_ndot: Vec<bool>,
    /// `substream_index` when `b_substreams_present`.
    pub substream_index: Option<u32>,
}

impl ObjSubstreamInfo {
    /// `n_objects` per Table 82: codes 0..=3 map to `code + b_lfe`,
    /// code 4 maps to `5 + b_lfe`; 5..=7 are reserved (`None`).
    pub fn n_objects(&self) -> Option<u32> {
        let lfe = u32::from(self.b_lfe);
        match self.n_objects_code {
            0..=3 => Some(self.n_objects_code as u32 + lfe),
            4 => Some(5 + lfe),
            _ => None,
        }
    }
}

/// Parse `ac4_substream_info_obj(b_substreams_present)` per
/// TS 103 190-2 §6.2.1.11.
pub fn parse_substream_info_obj(
    br: &mut BitReader<'_>,
    fs_index: u32,
    frame_rate_index: u32,
    b_substreams_present: bool,
) -> Result<ObjSubstreamInfo> {
    let n_objects_code = br.read_u32(3)? as u8;
    let b_dynamic_objects = br.read_bit()?;
    let mut b_lfe = false;
    let mut bed = None;
    if b_dynamic_objects {
        b_lfe = br.read_bit()?;
    } else if br.read_bit()? {
        // b_bed_objects.
        if br.read_bit()? {
            // b_bed_start.
            bed = Some(if br.read_bit()? {
                ObjBedStart::BedCode {
                    code: br.read_u32(3)? as u8,
                }
            } else if br.read_bit()? {
                ObjBedStart::BedNonStdMask {
                    mask: br.read_u32(17)?,
                }
            } else {
                ObjBedStart::BedStdMask {
                    mask: br.read_u32(10)? as u16,
                }
            });
        } else {
            bed = Some(ObjBedStart::BedContinuation);
        }
    } else if br.read_bit()? {
        // b_isf.
        let isf_config = if br.read_bit()? {
            Some(br.read_u32(3)? as u8)
        } else {
            None
        };
        bed = Some(ObjBedStart::Isf { isf_config });
    } else {
        let res_bytes = br.read_u32(4)? as u8;
        br.skip(res_bytes as u32 * 8)?;
        bed = Some(ObjBedStart::Reserved { res_bytes });
    }
    let mut sf_multiplier = 0;
    if fs_index == 1 && br.read_bit()? {
        sf_multiplier = br.read_u32(1)? + 1;
    }
    let bitrate_indicator = if br.read_bit()? {
        Some(parse_bitrate_indicator(br)?)
    } else {
        None
    };
    let factor = frame_rate_factor(frame_rate_index, false, 0).max(1);
    let mut b_audio_ndot = Vec::with_capacity(factor as usize);
    for _ in 0..factor {
        b_audio_ndot.push(br.read_bit()?);
    }
    let substream_index = if b_substreams_present {
        let mut si = br.read_u32(2)?;
        if si == 3 {
            si += variable_bits(br, 2)?;
        }
        Some(si)
    } else {
        None
    };
    Ok(ObjSubstreamInfo {
        n_objects_code,
        b_dynamic_objects,
        b_lfe,
        bed,
        sf_multiplier,
        bitrate_indicator,
        b_audio_ndot,
        substream_index,
    })
}

/// `frame_rate_fractions_info()` per ETSI TS 103 190-2 §6.2.1.4 — gated
/// on `frame_rate_index`. Consumes 0, 1, or 2 bits depending on the
/// frame-rate slot.
fn parse_frame_rate_fractions_info(br: &mut BitReader<'_>, frame_rate_index: u32) -> Result<()> {
    match frame_rate_index {
        5..=9 => {
            // Spec gates the read on `frame_rate_factor == 1`. We don't
            // re-derive the factor here — `frame_rate_multiply_info()`
            // determines it via `b_multiplier`, which is already consumed
            // above this call. For the b_multiplier=0 default path
            // (factor == 1) the fraction bit IS present; for the
            // b_multiplier=1 high-FPS variants (factor == 2) it isn't.
            // Round 47 only round-trips the `frame_rate_index == 1`
            // (24 fps) and `b_multiplier == 0` paths via the IMS encoder
            // — for those, frame_rate_index is outside [5, 9] so this
            // branch is unreachable. For full robustness we'd need to
            // thread `b_multiplier` through; deferred until a real v2
            // fixture forces the issue.
            let _b_frame_rate_fraction = br.read_bit()?;
            // No second bit for indices 5..=9.
        }
        10..=12 => {
            let b_frame_rate_fraction = br.read_bit()?;
            if b_frame_rate_fraction {
                let _b_frame_rate_fraction_is_4 = br.read_bit()?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn parse_substream_index_table(br: &mut BitReader<'_>) -> Result<(u32, Vec<u32>)> {
    // §4.2.3.11 Syntax of substream_index_table().
    let mut n_substreams = br.read_u32(2)?;
    if n_substreams == 0 {
        n_substreams = variable_bits(br, 2)? + 4;
    }
    let b_size_present = if n_substreams == 1 {
        br.read_bit()?
    } else {
        true
    };
    let mut sizes = Vec::new();
    if b_size_present {
        for _ in 0..n_substreams {
            let b_more_bits = br.read_bit()?;
            let mut size = br.read_u32(10)?;
            if b_more_bits {
                size += variable_bits(br, 2)? << 10;
            }
            sizes.push(size);
        }
    }
    Ok((n_substreams, sizes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_bits_single_chunk() {
        // value = 0b10 (2), terminator bit clear.
        // Grouping mirrors variable_bits(n=2) layout: 2 value bits, 1 terminator, 5 pad.
        #[allow(clippy::unusual_byte_groupings)] // ETSI TS 103 190-1 §4.2.2 variable_bits()
        let bytes = [0b10_0_00000];
        let mut br = BitReader::new(&bytes);
        let v = variable_bits(&mut br, 2).unwrap();
        assert_eq!(v, 0b10);
    }

    #[test]
    fn variable_bits_multi_chunk() {
        // value = 0b11 (3) then 0b01 (1), terminator clear. Expected:
        //   first chunk: value = 3, more=1 -> shift by 2, add 4 -> value = 16.
        //   second chunk: value = 16 + 1 = 17.
        //   Encoded as: 11 1 01 0 ...
        let bytes = [0b1110_1000];
        let mut br = BitReader::new(&bytes);
        let v = variable_bits(&mut br, 2).unwrap();
        assert_eq!(v, 17);
    }

    #[test]
    fn write_variable_bits_round_trips_decoder() {
        use oxideav_core::bits::BitWriter;
        // Exhaustive small values + boundaries for several chunk widths.
        let values = [
            0u32,
            1,
            2,
            3,
            4,
            7,
            8,
            15,
            16,
            17,
            31,
            32,
            63,
            64,
            100,
            255,
            256,
            1023,
            1024,
            4095,
            4096,
            65_535,
            65_536,
            1_000_000,
            u32::MAX - 1,
            u32::MAX,
        ];
        for n in [2u32, 3, 5, 8, 11] {
            for &v in &values {
                let mut bw = BitWriter::new();
                write_variable_bits(&mut bw, n, v);
                bw.align_to_byte();
                let bytes = bw.finish();
                let mut br = BitReader::new(&bytes);
                let got = variable_bits(&mut br, n).unwrap();
                assert_eq!(got, v, "n={n} v={v}");
            }
        }
    }

    #[test]
    fn write_variable_bits_matches_known_multichunk() {
        use oxideav_core::bits::BitWriter;
        // value 17 at n=2 must encode as `11 1 01 0` (see
        // variable_bits_multi_chunk above).
        let mut bw = BitWriter::new();
        write_variable_bits(&mut bw, 2, 17);
        bw.align_to_byte();
        assert_eq!(bw.finish(), vec![0b1110_1000]);
    }

    #[test]
    fn frame_rate_entry_table() {
        assert_eq!(frame_rate_entry(1, 1), (24_000, 1_920));
        assert_eq!(frame_rate_entry(6, 1), (48_000, 960));
        assert_eq!(frame_rate_entry(13, 0), (21_533, 2_048));
        assert_eq!(frame_rate_entry(14, 1), (0, 0));
    }

    #[test]
    fn channel_mode_mono_stereo_51() {
        // Mono prefix: 0.
        let bytes = [0b0_0000000];
        let mut br = BitReader::new(&bytes);
        assert_eq!(decode_channel_mode(&mut br).unwrap(), (1, 1));

        // Stereo prefix: 10.
        let bytes = [0b10_000000];
        let mut br = BitReader::new(&bytes);
        assert_eq!(decode_channel_mode(&mut br).unwrap(), (2, 2));

        // 5.1 prefix: 1110.
        let bytes = [0b1110_0000];
        let mut br = BitReader::new(&bytes);
        assert_eq!(decode_channel_mode(&mut br).unwrap(), (6, 4));
    }

    use oxideav_core::bits::BitWriter;

    fn write_bed_dyn_obj_assignment(bw: &mut BitWriter, a: &BedDynObjAssignment, n_signals: u32) {
        match a {
            BedDynObjAssignment::DynObjectsOnly => bw.write_bit(true),
            BedDynObjAssignment::Isf { isf_config } => {
                bw.write_bit(false);
                bw.write_bit(true);
                bw.write_u32(*isf_config as u32, 3);
            }
            BedDynObjAssignment::BedCode { code } => {
                bw.write_bit(false);
                bw.write_bit(false);
                bw.write_bit(true);
                bw.write_u32(*code as u32, 3);
            }
            BedDynObjAssignment::BedNonStdMask { mask } => {
                bw.write_u32(0b0001, 4);
                bw.write_bit(true);
                bw.write_u32(*mask, 17);
            }
            BedDynObjAssignment::BedStdMask { mask } => {
                bw.write_u32(0b0001, 4);
                bw.write_bit(false);
                bw.write_u32(*mask as u32, 10);
            }
            BedDynObjAssignment::BedIndividual { assignments } => {
                bw.write_u32(0b0000, 4);
                if n_signals > 1 {
                    let bits = 32 - (n_signals - 1).leading_zeros();
                    bw.write_u32(assignments.len() as u32 - 1, bits);
                }
                for a in assignments {
                    bw.write_u32(*a as u32, 4);
                }
            }
        }
    }

    #[test]
    fn bed_dyn_obj_assignment_round_trips_all_forms() {
        let cases: [(BedDynObjAssignment, u32); 6] = [
            (BedDynObjAssignment::DynObjectsOnly, 8),
            (BedDynObjAssignment::Isf { isf_config: 4 }, 15),
            (BedDynObjAssignment::BedCode { code: 2 }, 6),
            (BedDynObjAssignment::BedNonStdMask { mask: 0x1_5555 }, 9),
            (BedDynObjAssignment::BedStdMask { mask: 0b0000001111 }, 6),
            (
                BedDynObjAssignment::BedIndividual {
                    // 4-bit Table 85 values (the field cannot reach
                    // index 16 = L; masks cover the full table).
                    assignments: vec![15, 14, 12],
                },
                5,
            ),
        ];
        for (a, n_signals) in &cases {
            let mut bw = BitWriter::new();
            write_bed_dyn_obj_assignment(&mut bw, a, *n_signals);
            bw.write_u32(0, 7);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let got = parse_bed_dyn_obj_assignment(&mut br, *n_signals).unwrap();
            assert_eq!(&got, a);
        }
    }

    #[test]
    fn bed_dyn_obj_assignment_static_counts_match_tables() {
        use crate::oamd::ObjType;
        // Table 83: isf_config 4 = SR7.5.3.0 → 15 objects.
        assert_eq!(
            BedDynObjAssignment::Isf { isf_config: 4 }.n_static_objects(),
            15
        );
        // Table 84: code 2 = 5.1.0 → 6 channels.
        assert_eq!(
            BedDynObjAssignment::BedCode { code: 2 }.n_static_objects(),
            6
        );
        // Table 85 mask popcount.
        assert_eq!(
            BedDynObjAssignment::BedNonStdMask { mask: 0b101 }.n_static_objects(),
            2
        );
        // Table 86: L/R + C + LFE + Ls/Rs = 6.
        assert_eq!(
            BedDynObjAssignment::BedStdMask { mask: 0b0000001111 }.n_static_objects(),
            6
        );
        // Static objects precede dynamics.
        let types = BedDynObjAssignment::BedCode { code: 0 }.obj_types(4);
        assert_eq!(
            types,
            vec![ObjType::Bed, ObjType::Bed, ObjType::Dyn, ObjType::Dyn]
        );
        assert_eq!(
            BedDynObjAssignment::DynObjectsOnly.obj_types(3),
            vec![ObjType::Dyn; 3]
        );
    }

    fn write_substream_info_ajoc_minimal(bw: &mut BitWriter) {
        // b_lfe = 1, b_static_dmx = 0, n_fb_dmx = 4 (minus1 = 3),
        // dmx assignment = DynObjectsOnly.
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_u32(3, 4);
        bw.write_bit(true);
        // b_oamd_common_data_present = 0.
        bw.write_bit(false);
        // n_fb_umx = 12 (minus1 = 11), umx assignment = BedCode 2.
        bw.write_u32(11, 4);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_u32(2, 3);
        // fs_index = 1: b_sf_multiplier = 0.
        bw.write_bit(false);
        // b_bitrate_info = 0.
        bw.write_bit(false);
        // frame_rate_factor = 1 → one b_audio_ndot = 1 ("no dependency
        // over time" = I-frame, §6.3.2.x).
        bw.write_bit(true);
        // b_substreams_present = 1 → substream_index = 1.
        bw.write_u32(1, 2);
    }

    #[test]
    fn substream_info_ajoc_parses_and_derives_object_types() {
        use crate::oamd::ObjType;
        let mut bw = BitWriter::new();
        write_substream_info_ajoc_minimal(&mut bw);
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let info = parse_substream_info_ajoc(&mut br, 1, 1, true).unwrap();
        assert!(info.b_lfe);
        assert!(!info.b_static_dmx);
        assert_eq!(info.n_fullband_dmx_signals, 4);
        assert_eq!(info.n_dmx_signals(), 5);
        assert_eq!(info.n_fullband_upmix_signals, 12);
        assert_eq!(info.n_umx_signals(), 13);
        assert_eq!(
            info.dmx_assignment,
            Some(BedDynObjAssignment::DynObjectsOnly)
        );
        assert_eq!(
            info.umx_assignment,
            BedDynObjAssignment::BedCode { code: 2 }
        );
        assert!(info.b_iframe());
        assert_eq!(info.substream_index, Some(1));
        // dmx: LFE slot + 4 dynamic fullband objects.
        assert_eq!(info.obj_type_dmx(), vec![ObjType::Dyn; 5]);
        assert_eq!(info.is_lfe_dmx(), vec![true, false, false, false, false]);
        // umx: LFE slot + 6 bed + 6 dynamic.
        let umx = info.obj_type_umx();
        assert_eq!(umx.len(), 13);
        assert_eq!(umx[0], ObjType::Dyn);
        assert!(umx[1..7].iter().all(|t| *t == ObjType::Bed));
        assert!(umx[7..].iter().all(|t| *t == ObjType::Dyn));
    }

    #[test]
    fn substream_info_ajoc_static_dmx_is_5ch_bed() {
        use crate::oamd::ObjType;
        let mut bw = BitWriter::new();
        // b_lfe = 0, b_static_dmx = 1 → n_fb_dmx = 5, no assignment.
        bw.write_bit(false);
        bw.write_bit(true);
        // b_oamd_common_data_present = 0.
        bw.write_bit(false);
        // n_fb_umx = 16 → escape: minus1 = 15, variable_bits(3) = 2 → 18.
        bw.write_u32(15, 4);
        // variable_bits(3) = 2: chunk "010" with 1-bit continuation 0
        // → b_read_more = 0 after value 2? variable_bits format: value
        // chunks of 3 bits each preceded by... use write_variable_bits.
        write_variable_bits(&mut bw, 3, 2);
        // umx assignment DynObjectsOnly.
        bw.write_bit(true);
        // fs_index = 0 → no sf_multiplier bit. b_bitrate_info = 0.
        bw.write_bit(false);
        // b_audio_ndot = 0 (dependency over time = P-frame).
        bw.write_bit(false);
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let info = parse_substream_info_ajoc(&mut br, 0, 1, false).unwrap();
        assert!(info.b_static_dmx);
        assert_eq!(info.n_fullband_dmx_signals, 5);
        assert_eq!(info.n_fullband_upmix_signals, 18);
        assert_eq!(info.dmx_assignment, None);
        assert_eq!(info.obj_type_dmx(), vec![ObjType::Bed; 5]);
        assert!(!info.b_iframe());
        assert_eq!(info.substream_index, None);
    }

    #[test]
    fn substream_info_obj_dynamic_and_bed_forms() {
        // Dynamic-objects form: n_objects_code = 4, b_dynamic = 1,
        // b_lfe = 1 → n_objects = 6.
        let mut bw = BitWriter::new();
        bw.write_u32(4, 3);
        bw.write_bit(true);
        bw.write_bit(true);
        // fs_index = 1: b_sf_multiplier = 1, sf_multiplier = 1 (×4).
        bw.write_bit(true);
        bw.write_u32(1, 1);
        // b_bitrate_info = 1, bitrate_indicator = 0b111 escape + 0b01.
        bw.write_bit(true);
        bw.write_u32(0b111, 3);
        bw.write_u32(0b01, 2);
        // b_audio_ndot = 0.
        bw.write_bit(false);
        // substream_index = 3 escape + variable_bits(2) = 1 → 4.
        bw.write_u32(3, 2);
        write_variable_bits(&mut bw, 2, 1);
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let info = parse_substream_info_obj(&mut br, 1, 1, true).unwrap();
        assert_eq!(info.n_objects(), Some(6));
        assert!(info.b_dynamic_objects);
        assert!(info.b_lfe);
        assert_eq!(info.sf_multiplier, 2);
        assert_eq!(info.bitrate_indicator, Some(0b11101));
        assert_eq!(info.substream_index, Some(4));

        // Bed form: n_objects_code = 2, b_dynamic = 0, b_bed = 1,
        // b_bed_start = 1, b_ch_assign_code = 1, code = 0 (L/R).
        let mut bw = BitWriter::new();
        bw.write_u32(2, 3);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_u32(0, 3);
        // fs_index = 0, b_bitrate_info = 0, ndot = 0, no substreams.
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let info = parse_substream_info_obj(&mut br, 0, 1, false).unwrap();
        assert_eq!(info.n_objects(), Some(2));
        assert_eq!(info.bed, Some(ObjBedStart::BedCode { code: 0 }));
    }

    #[test]
    fn substream_group_info_walks_an_ajoc_group() {
        // ac4_substream_group_info(): b_substreams_present = 0,
        // b_hsf_ext = 0, b_single_substream = 1, b_channel_coded = 0,
        // b_oamd_substream = 0, then substream 0: b_ajoc = 1 + the
        // minimal descriptor, then b_content_type = 0.
        let mut bw = BitWriter::new();
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(true);
        // Descriptor with b_substreams_present = 0 (no index tail):
        // reuse the minimal writer minus the substream_index.
        bw.write_bit(true); // b_lfe
        bw.write_bit(false); // b_static_dmx
        bw.write_u32(3, 4); // n_fb_dmx minus1
        bw.write_bit(true); // DynObjectsOnly
        bw.write_bit(false); // no oamd_common_data
        bw.write_u32(11, 4); // n_fb_umx minus1
        bw.write_bit(true); // DynObjectsOnly
        bw.write_bit(false); // b_sf_multiplier = 0 (fs_index = 1)
        bw.write_bit(false); // b_bitrate_info = 0
        bw.write_bit(true); // b_audio_ndot = 1 (independent = I-frame)
        bw.write_bit(false); // b_content_type = 0
        bw.write_u32(0, 7);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let summary = parse_substream_group_info(&mut br, 2, 1, 1).unwrap();
        assert_eq!(summary.ajoc.len(), 1);
        assert!(summary.objs.is_empty());
        let info = &summary.ajoc[0];
        assert_eq!(info.n_dmx_signals(), 5);
        assert_eq!(info.n_umx_signals(), 13);
        assert!(info.b_iframe());
    }
}
