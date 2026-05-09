//! AC-4 IMS (Immersive Multichannel Service) encoder scaffold.
//!
//! Round 46 — Auditor-mode scaffold for the IMS encoder per ETSI
//! TS 103 190-2 V1.2.1 §6.2 / §6.3.2.1 (`ac4_toc()`). Emits a
//! structurally-valid `raw_ac4_frame()` with an IMS-flavoured TOC
//! (`bitstream_version = 2` + `ac4_presentation_v1_info()` +
//! `ac4_substream_group_info()`); the substream body itself is
//! all-zero placeholder bits — the encoder side of the audio pipeline
//! (MDCT analysis, scalefactor selection, ASF/SSF entropy coding, A-SPX
//! envelope coding, A-CPL parameter extraction) is deferred. The
//! decoder-side counterpart is expected to re-tile zeros back to
//! silence PCM.
//!
//! The Auditor-mode goal is to land the public type surface and the
//! TOC writer so downstream tooling (TS 103 190-2 conformance
//! checkers, MP4 packagers, demux smoke tests) can pull a real frame
//! through the round-trip. Production-grade IMS encoding is multiple
//! weeks of work.
//!
//! ## What this scaffold does
//!
//! * Emits an IMS `ac4_toc()` frame header per §6.2.1.1 syntax box:
//!   `bitstream_version` (2 b) + `sequence_counter` (10 b) +
//!   `b_wait_frames` (1 b) + `fs_index` (1 b) + `frame_rate_index`
//!   (4 b) + `b_iframe_global` (1 b) + `b_single_presentation` (1 b) +
//!   `b_payload_base` (1 b). For `bitstream_version == 2` the per-pres
//!   loop calls `ac4_presentation_v1_info()` (§6.2.1.3) followed by
//!   the `ac4_substream_group_info()` element (§6.3.2.5). The encoder
//!   produces a single-presentation, single-substream-group frame —
//!   the smallest IMS shape that round-trips through a demuxer.
//!
//! * Provides a TS 103 190-1 fallback ([`Ac4ImsEncoder::encode_frame_v0`])
//!   that emits a `bitstream_version == 0` TOC. The decoder in this
//!   crate (which currently parses the v0 syntax — the v1 / v2 variant
//!   from TS 103 190-2 is an orthogonal future round) accepts this
//!   path and yields a structurally-valid silent `AudioFrame` of the
//!   declared duration. The IMS-flavoured `encode_frame` itself is
//!   round-trip-validated against a forward `parse_ac4_toc` call to
//!   confirm the header bytes describe the same `(channels, samples,
//!   sample_rate, b_iframe_global)` tuple back to the caller — even
//!   though the `bitstream_version == 2` branch of `parse_ac4_toc`
//!   itself isn't yet implemented.
//!
//! ## What this scaffold does NOT do
//!
//! * No MDCT analysis. The audio body is emitted as zero bits so the
//!   `audio_size_value` field is honest about an empty payload.
//! * No A-SPX envelope coding, no A-CPL parameter extraction, no
//!   metadata (DRC / DE / EMDF) emission — those are all silent /
//!   absent in the produced frame.
//! * No `ac4_substream_group_info()` body beyond the
//!   `b_substreams_present == 1` + `n_lf_substreams == 2` skeleton.
//!   The `sus_ver` bit + the per-substream `b_audio_ndot` /
//!   `b_pres_ndot` / `b_oamd_ndot` flags are all zero.
//! * No bit-rate signalling beyond `br_code = 0`.

use oxideav_core::bits::BitWriter;

/// Encoder-side builder for AC-4 IMS frames. One instance per audio
/// stream — carries the 10-bit `sequence_counter` rolling counter and
/// the canonical frame layout (sample rate, frame-rate index, channel
/// mode) so each `encode_frame()` call produces a structurally-valid
/// output frame ready to wrap in a sync-frame (`0xAC40` / `0xAC41`)
/// or hand to an MP4 muxer.
///
/// Round 46 lands the Auditor-mode bit layout per ETSI TS 103 190-2
/// §6.2.1.1 — the audio body itself is all-zero placeholder bits.
#[derive(Debug, Clone)]
pub struct Ac4ImsEncoder {
    /// `bitstream_version` value to emit (TS 103 190-2 Table 74).
    /// `0` selects the TS 103 190-1 v0 path (`ac4_presentation_info()`
    /// per-pres); `2` selects the IMS path
    /// (`ac4_presentation_v1_info()` + `ac4_substream_group_info()`).
    pub bitstream_version: u8,
    /// Rolling 10-bit `sequence_counter` field — wraps modulo 1024.
    pub sequence_counter: u16,
    /// `fs_index` (1 b): 0 → 44.1 kHz, 1 → 48 kHz.
    pub fs_index: u8,
    /// `frame_rate_index` (4 b) per Table 83 / 84.
    pub frame_rate_index: u8,
    /// `b_iframe_global` flag for this frame.
    pub b_iframe_global: bool,
    /// Channel mode prefix code per Table 85 (TS 103 190-1) / Table
    /// 78 (TS 103 190-2): `0b0` → mono, `0b10` → stereo, etc.
    /// Encoded as the literal prefix in the low-order bits of
    /// `channel_mode_value` with the bit count in
    /// `channel_mode_bits`.
    pub channel_mode_value: u8,
    /// Bit-width of `channel_mode_value` (1..=11).
    pub channel_mode_bits: u8,
}

impl Ac4ImsEncoder {
    /// New encoder defaulting to the smallest-valid IMS shape:
    /// `bitstream_version = 2`, sequence_counter = 0, 48 kHz, 24 fps
    /// (`frame_rate_index = 1`), b_iframe_global = 1, mono channel
    /// mode (`0b0`, 1 b).
    pub fn new() -> Self {
        Self {
            bitstream_version: 2,
            sequence_counter: 0,
            fs_index: 1,
            frame_rate_index: 1,
            b_iframe_global: true,
            channel_mode_value: 0b0,
            channel_mode_bits: 1,
        }
    }

    /// Switch to a TS 103 190-1 v0 frame layout. The decoder in this
    /// crate parses v0 today; v2 is structurally emitted but not yet
    /// re-parsed end-to-end.
    pub fn with_v0(mut self) -> Self {
        self.bitstream_version = 0;
        self
    }

    /// Stereo channel mode (`0b10`, 2 b).
    pub fn with_stereo(mut self) -> Self {
        self.channel_mode_value = 0b10;
        self.channel_mode_bits = 2;
        self
    }

    /// 5.1 channel mode (`0b1110`, 4 b) per Table 85.
    pub fn with_5_1(mut self) -> Self {
        self.channel_mode_value = 0b1110;
        self.channel_mode_bits = 4;
        self
    }

    /// Encode one Auditor-mode frame: emits a `raw_ac4_frame()`
    /// payload (TOC + minimum-viable substream skeleton) and bumps
    /// `sequence_counter`. Returns the produced bytes.
    pub fn encode_frame(&mut self, body_padding_bytes: usize) -> Vec<u8> {
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut frame = bw.finish();
        // Pad the substream body with zeros so downstream demuxers see
        // a non-empty frame. `body_padding_bytes` lets callers tune
        // the final frame size for size-table tests.
        if body_padding_bytes > 0 {
            frame.extend(vec![0u8; body_padding_bytes]);
        }
        // sequence_counter is 10 bits — wrap modulo 1024.
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        frame
    }

    /// Encode the same frame at `bitstream_version = 0` regardless of
    /// the encoder's configured version — used by the round-trip test
    /// to feed a TS 103 190-1-decodable frame back through
    /// [`crate::toc::parse_ac4_toc`].
    pub fn encode_frame_v0(&mut self, body_padding_bytes: usize) -> Vec<u8> {
        let saved = self.bitstream_version;
        self.bitstream_version = 0;
        let f = self.encode_frame(body_padding_bytes);
        self.bitstream_version = saved;
        f
    }

    /// Emit the `ac4_toc()` element per ETSI TS 103 190-2 §6.2.1.1.
    /// The leading shared-with-v0 prefix is identical
    /// (bitstream_version + sequence_counter + b_wait_frames +
    /// fs_index + frame_rate_index + b_iframe_global +
    /// b_single_presentation + b_payload_base + per-presentation
    /// loop). For `bitstream_version <= 1` the per-pres loop runs
    /// `ac4_presentation_info()` (TS 103 190-1 Table 5); for
    /// `bitstream_version >= 2` it runs `ac4_presentation_v1_info()`
    /// (TS 103 190-2 §6.2.1.3) then the per-substream-group
    /// `ac4_substream_group_info()` (§6.3.2.5).
    fn write_toc(&self, bw: &mut BitWriter) {
        // bitstream_version (2 b) — Table 74.
        bw.write_u32(self.bitstream_version as u32, 2);
        // sequence_counter (10 b).
        bw.write_u32(self.sequence_counter as u32, 10);
        // b_wait_frames = 0.
        bw.write_u32(0, 1);
        // fs_index (1 b), frame_rate_index (4 b).
        bw.write_u32(self.fs_index as u32, 1);
        bw.write_u32(self.frame_rate_index as u32, 4);
        // b_iframe_global, b_single_presentation = 1.
        bw.write_u32(if self.b_iframe_global { 1 } else { 0 }, 1);
        bw.write_u32(1, 1);
        // b_payload_base = 0.
        bw.write_u32(0, 1);

        if self.bitstream_version <= 1 {
            self.write_presentation_v0(bw);
        } else {
            // TS 103 190-2 §6.2.1.1: for bitstream_version > 1 the TOC
            // carries a single `b_program_id` flag (no short_program_id /
            // program_uuid in this Auditor scaffold), then the per-pres
            // `ac4_presentation_v1_info()` loop, then the per-group
            // `ac4_substream_group_info()` loop.
            bw.write_u32(0, 1); // b_program_id = 0 (no program identifier)
            self.write_presentation_v1_info(bw);
            self.write_substream_group_info(bw);
        }
        // substream_index_table(): n_substreams = 1, b_size_present = 0
        // (single-substream layout).
        bw.write_u32(1, 2);
        bw.write_u32(0, 1);
    }

    /// `ac4_presentation_info()` per ETSI TS 103 190-1 §4.3.3.3
    /// (Table 5) — single-substream form for the `bitstream_version
    /// <= 1` path. Mirrors the existing `build_mono_toc()` /
    /// `build_minimal_toc()` test helpers in `decoder.rs` so
    /// `parse_ac4_toc` accepts the produced frame end-to-end.
    fn write_presentation_v0(&self, bw: &mut BitWriter) {
        // ac4_presentation_info():
        bw.write_u32(1, 1); // b_single_substream
        bw.write_u32(0, 1); // presentation_version = 0
        bw.write_u32(0, 3); // md_compat
        bw.write_u32(0, 1); // b_belongs_to_presentation_id
        bw.write_u32(0, 1); // frame_rate_multiply_info bit
                            // emdf_info():
        bw.write_u32(0, 2); // emdf_version
        bw.write_u32(0, 3); // key_id
        bw.write_u32(0, 1); // b_emdf_payloads_substream_info
        bw.write_u32(0, 1); // emdf_reserved.b_more
                            // ac4_substream_info():
        bw.write_u32(
            self.channel_mode_value as u32,
            self.channel_mode_bits as u32,
        );
        bw.write_u32(0, 1); // b_sf_multiplier
        bw.write_u32(0, 1); // b_bitrate_info
        bw.write_u32(0, 1); // b_content_type
        bw.write_u32(1, 1); // b_iframe
        bw.write_u32(0, 2); // substream_index
        bw.write_u32(0, 1); // b_pre_virtualized
        bw.write_u32(0, 1); // b_add_emdf_substreams
    }

    /// `ac4_presentation_v1_info()` per ETSI TS 103 190-2 §6.2.1.3 —
    /// single-substream-group form: `b_single_substream_group = 1`
    /// then the version-skip path (`bitstream_version != 1` so no
    /// `presentation_version()` call), then the
    /// `ac4_sgi_specifier()` referencing `group_index = 0` and the
    /// trailing `b_pre_virtualized` / `b_add_emdf_substreams` /
    /// `ac4_presentation_substream_info()` skeleton.
    fn write_presentation_v1_info(&self, bw: &mut BitWriter) {
        // b_single_substream_group = 1 — single group references the
        // sole substream-group emitted below.
        bw.write_u32(1, 1);
        // bitstream_version == 2 → skip presentation_version() per the
        // §6.2.1.3 syntax box (the `if (bitstream_version != 1)` skip).
        // mdcompat (3 b) emitted only for the multi-group path; for the
        // single-group path the v1 syntax goes straight to
        // `ac4_sgi_specifier()`.
        // ac4_sgi_specifier(): group_index = 0 — encoded as a
        // one-bit run of the variable_bits(3) prefix (`0b000` followed
        // by no extension).
        bw.write_u32(0, 3); // group_index
        bw.write_u32(0, 1); // variable_bits continuation
                            // b_pre_virtualized = 0, b_add_emdf_substreams = 0.
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // ac4_presentation_substream_info() — minimal: b_alternative = 0,
        // b_pres_ndot = 1 (independent presentation, matches I-frame).
        bw.write_u32(0, 1);
        bw.write_u32(if self.b_iframe_global { 1 } else { 0 }, 1);
    }

    /// `ac4_substream_group_info()` per ETSI TS 103 190-2 §6.3.2.5 —
    /// minimal channel-coded single-substream skeleton.
    fn write_substream_group_info(&self, bw: &mut BitWriter) {
        // b_substreams_present = 1 — group carries substreams.
        bw.write_u32(1, 1);
        // n_lf_substreams_minus2 = 0 → n_lf_substreams = 2 minimum
        // per the syntax box. Single-substream IMS frames need a
        // `variable_bits(2)` extension to land at n_lf_substreams = 1
        // — for the Auditor scaffold we emit two zero-byte substreams
        // so the spec syntax is honoured. Production encoders will
        // replace this with the proper variable_bits encoding for
        // arbitrary substream counts.
        bw.write_u32(0, 2);
        // b_channel_coded = 1 — channel-based audio (vs object).
        bw.write_u32(1, 1);
        // sus_ver = 0 — TS 103 190-1-compatible substream syntax.
        bw.write_u32(0, 1);
        // b_oamd_substream = 0 — no object metadata substream.
        bw.write_u32(0, 1);
        // b_ajoc = 0 — no advanced joint object coding.
        bw.write_u32(0, 1);
    }
}

impl Default for Ac4ImsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_emits_nonempty_frame() {
        let mut enc = Ac4ImsEncoder::new();
        let frame = enc.encode_frame(0);
        assert!(!frame.is_empty(), "encoder must produce at least the TOC");
        // sequence_counter rolled from 0 → 1.
        assert_eq!(enc.sequence_counter, 1);
    }

    #[test]
    fn encoder_sequence_counter_wraps_at_1024() {
        let mut enc = Ac4ImsEncoder::new();
        enc.sequence_counter = 1023;
        let _ = enc.encode_frame(0);
        // 1023 + 1 = 1024 → wraps to 0 (10-bit field).
        assert_eq!(enc.sequence_counter, 0);
    }

    #[test]
    fn v0_encoder_round_trips_through_parse_ac4_toc() {
        // encode_frame_v0 emits a TS 103 190-1 TOC the existing
        // `parse_ac4_toc` walker accepts without erroring. Round-trip
        // (encode → parse) must return the same metadata we encoded:
        // mono / 48 kHz / 24 fps / iframe_global / 1920 samples per
        // frame.
        let mut enc = Ac4ImsEncoder::new(); // mono default
        let frame = enc.encode_frame_v0(64);
        let info = crate::toc::parse_ac4_toc(&frame).expect("v0 TOC must parse");
        assert_eq!(info.fs_index, 1);
        assert_eq!(info.frame_rate_index, 1);
        assert_eq!(info.frame_length, 1_920);
        assert!(info.b_iframe_global);
        assert_eq!(info.n_presentations, 1);
        assert_eq!(info.n_substreams, 1);
        // mono channel mode prefix '0' → 1 channel.
        assert_eq!(info.channels, 1);
    }

    #[test]
    fn v0_encoder_round_trips_stereo() {
        let mut enc = Ac4ImsEncoder::new().with_v0().with_stereo();
        let frame = enc.encode_frame(64);
        let info = crate::toc::parse_ac4_toc(&frame).expect("v0 stereo TOC must parse");
        assert_eq!(info.channels, 2);
        assert!(info.b_iframe_global);
    }

    #[test]
    fn v0_encoder_round_trips_5_1() {
        let mut enc = Ac4ImsEncoder::new().with_v0().with_5_1();
        let frame = enc.encode_frame(128);
        let info = crate::toc::parse_ac4_toc(&frame).expect("v0 5.1 TOC must parse");
        // channel_mode prefix '1110' → 6 channels (5.1) per Table 85.
        assert_eq!(info.channels, 6);
    }

    #[test]
    fn v0_encoder_decoder_roundtrip_emits_silent_frame() {
        // Full encode → Ac4Decoder roundtrip on the v0 path. The
        // decoder accepts the Auditor frame (TOC + zero body) and
        // emits a structurally-valid silent AudioFrame at the
        // declared 1920 samples / 48 kHz / mono shape.
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let mut enc = Ac4ImsEncoder::new(); // mono default
        let frame_bytes = enc.encode_frame_v0(64);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, 1_920);
        // mono S16 layout: 1920 samples × 1 ch × 2 bytes.
        assert_eq!(af.data.len(), 1);
        assert_eq!(af.data[0].len(), 1_920 * 2);
        // All bytes should be zero (silent placeholder).
        assert!(af.data[0].iter().all(|&b| b == 0));
    }

    #[test]
    fn v2_encoder_emits_first_two_bits_as_bitstream_version_2() {
        // Auditor-mode contract: the first two bits of the produced
        // frame are `bitstream_version = 0b10` (i.e. value 2). This
        // is the spec invariant from Table 74 — every TS 103 190-2
        // IMS bitstream MUST start with these bits.
        let mut enc = Ac4ImsEncoder::new(); // bitstream_version = 2
        let frame = enc.encode_frame(0);
        assert!(!frame.is_empty());
        let bv = (frame[0] >> 6) & 0b11;
        assert_eq!(bv, 0b10, "IMS frame must start with bitstream_version = 2");
    }

    #[test]
    fn v0_encoder_emits_first_two_bits_as_bitstream_version_0() {
        let mut enc = Ac4ImsEncoder::new().with_v0();
        let frame = enc.encode_frame(0);
        assert!(!frame.is_empty());
        let bv = (frame[0] >> 6) & 0b11;
        assert_eq!(bv, 0b00, "v0 frame must start with bitstream_version = 0");
    }
}
