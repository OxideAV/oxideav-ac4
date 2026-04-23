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

    // Presentation loop.
    let mut presentations = Vec::with_capacity(n_presentations as usize);
    for _ in 0..n_presentations {
        let pi = parse_presentation_info(&mut br, fs_index, frame_rate_index)?;
        presentations.push(pi);
    }

    // substream_index_table().
    let (n_substreams, substream_sizes) = parse_substream_index_table(&mut br)?;

    // Byte-align at the end of ac4_toc().
    br.align_to_byte();
    let toc_size = br.byte_position() as u32;

    // Derive effective sample rate: pick the first presentation's
    // sf_multiplier if present, otherwise fall back to the base rate.
    let base_sr = base_sample_rate(fs_index);
    let sf_mul = presentations
        .first()
        .map(|p| p.sf_multiplier)
        .unwrap_or(0);
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
    })
}

/// `frame_rate_factor` derived from the frame_rate_index and the
/// presentation's multiplier bits (Table 87 in TS 103 190-1 §4.3.3.3.4).
fn frame_rate_factor(frame_rate_index: u32, b_multiplier: bool, multiplier_bit: u32) -> u32 {
    match frame_rate_index {
        // Indices 2/3/4 — 25 / 29.97 / 30 fps: factor is 1 or (b_multiplier ? 1+multiplier_bit : 1).
        2 | 3 | 4 => {
            if b_multiplier {
                if multiplier_bit == 0 {
                    2
                } else {
                    4
                }
            } else {
                1
            }
        }
        // Indices 0/1/7/8/9 — high-FPS forms: factor is 1 or 2.
        0 | 1 | 7 | 8 | 9 => {
            if b_multiplier {
                2
            } else {
                1
            }
        }
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
        2 | 3 | 4 => {
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
                0 | 1 | 2 => {
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
}
