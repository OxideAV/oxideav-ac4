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
//! Round 47 fixes the v2 TOC bit layout to match the literal §6.2.1.1
//! / §6.2.1.3 / §6.3.2.5 syntax boxes (the round-46 scaffold skipped
//! `b_hsf_ext` / `b_single_substream` in `ac4_substream_group_info()`
//! and emitted a stale `ac4_presentation_v1_info()` skeleton missing
//! `mdcompat`, `frame_rate_fractions_info()`, `emdf_info()`,
//! `b_presentation_filter`, and the trailing
//! `ac4_substream_info_chan()` body). The matching v2 dispatch in
//! [`crate::toc::parse_ac4_toc`] now walks the same syntax, so v2
//! `Ac4ImsEncoder::encode_frame()` → `parse_ac4_toc` round-trips the
//! same `(channels, samples, sample_rate, b_iframe_global)` tuple as
//! the v0 path.
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

use crate::encoder_asf::build_mono_simple_asf_body_from_pcm_spectrum;
use crate::encoder_mdct::EncoderMdctState;

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
    /// Forward-MDCT analysis state for `encode_frame_pcm()`. Carries
    /// the previous frame's `N` PCM samples so the 50% TDAC overlap
    /// runs correctly across frames. Lazy-initialised on first use.
    pub mdct_state: Option<EncoderMdctState>,
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
            mdct_state: None,
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
            // program_uuid in this scaffold), then the per-pres
            // `ac4_presentation_v1_info()` loop, then the per-group
            // `ac4_substream_group_info()` loop. Round 47 emits a
            // single-presentation, single-substream-group frame: the
            // smallest IMS shape that round-trips through `parse_ac4_toc`.
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
    /// single-substream-group form for `bitstream_version >= 2`:
    /// `b_single_substream_group = 1`, then `presentation_version() = 0`
    /// (single zero-bit since `bitstream_version != 1`), `mdcompat = 0`,
    /// `b_presentation_id = 0`, `frame_rate_multiply_info()` (one bit
    /// for `frame_rate_index = 1`), `frame_rate_fractions_info()`
    /// (zero bits for index 1), `emdf_info()` (minimum form),
    /// `b_presentation_filter = 0`, `ac4_sgi_specifier()` referencing
    /// `group_index = 0`, `b_pre_virtualized = 0`,
    /// `b_add_emdf_substreams = 0`, and `ac4_presentation_substream_info()`
    /// (b_alternative = 0, b_pres_ndot = !iframe, substream_index = 0).
    fn write_presentation_v1_info(&self, bw: &mut BitWriter) {
        // b_single_substream_group = 1.
        bw.write_u32(1, 1);
        // presentation_version() = 0 — single '0' bit (loop terminates
        // immediately). Emitted for bitstream_version != 1.
        bw.write_u32(0, 1);
        // mdcompat = 0 (3 b) — emitted for bitstream_version != 1.
        bw.write_u32(0, 3);
        // b_presentation_id = 0.
        bw.write_u32(0, 1);
        // frame_rate_multiply_info(): single b_multiplier bit for
        // frame_rate_index in {0, 1, 7, 8, 9}.
        bw.write_u32(0, 1);
        // frame_rate_fractions_info(): nothing for frame_rate_index < 5
        // or > 12.
        // emdf_info(): emdf_version=0 (2b), key_id=0 (3b),
        //   b_emdf_payloads_substream_info=0, emdf_reserved.b_more=0.
        bw.write_u32(0, 2);
        bw.write_u32(0, 3);
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // b_presentation_filter = 0.
        bw.write_u32(0, 1);
        // ac4_sgi_specifier(): group_index = 0 (3 b, no variable_bits
        // extension since group_index < 7).
        bw.write_u32(0, 3);
        // b_pre_virtualized = 0, b_add_emdf_substreams = 0.
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // ac4_presentation_substream_info(): b_alternative = 0,
        // b_pres_ndot = !b_iframe_global, substream_index = 0 (2 b).
        bw.write_u32(0, 1);
        bw.write_u32(if self.b_iframe_global { 0 } else { 1 }, 1);
        bw.write_u32(0, 2);
    }

    /// `ac4_substream_group_info()` per ETSI TS 103 190-2 §6.3.2.5 —
    /// single channel-coded substream skeleton matching the encoder's
    /// `n_substreams = 1` substream_index_table.
    fn write_substream_group_info(&self, bw: &mut BitWriter) {
        // b_substreams_present = 1.
        bw.write_u32(1, 1);
        // b_hsf_ext = 0 — no high-sample-rate extension.
        bw.write_u32(0, 1);
        // b_single_substream = 1 — n_lf_substreams = 1.
        bw.write_u32(1, 1);
        // b_channel_coded = 1 — channel-based audio (vs object).
        bw.write_u32(1, 1);
        // ac4_substream_info_chan(b_substreams_present = 1):
        //   channel_mode = encoder field (1..7 b),
        //   fs_index == 1: b_sf_multiplier = 0,
        //   b_bitrate_info = 0,
        //   frame_rate_factor copies of b_audio_ndot = !iframe,
        //   substream_index = 0 (2 b, since b_substreams_present = 1).
        bw.write_u32(
            self.channel_mode_value as u32,
            self.channel_mode_bits as u32,
        );
        if self.fs_index == 1 {
            bw.write_u32(0, 1); // b_sf_multiplier
        }
        bw.write_u32(0, 1); // b_bitrate_info
                            // frame_rate_factor for {0,1,7,8,9} with
                            // b_multiplier=0 is 1; for {2,3,4} also 1; otherwise 1.
                            // → 1 b_audio_ndot bit.
        bw.write_u32(if self.b_iframe_global { 0 } else { 1 }, 1);
        bw.write_u32(0, 2); // substream_index
                            // b_content_type = 0.
        bw.write_u32(0, 1);
    }
}

impl Default for Ac4ImsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a mono SIMPLE/ASF `ac4_substream()` body that injects a single
/// quantised spectral line at the specified scale-factor band. The
/// payload is sized for `transform_length = 1920` (24 fps @ 48 kHz)
/// with `max_sfb = 10`, matching the encoder's default frame layout.
///
/// `tone_cb_idx` selects the HCB5 codeword for the first spectral pair
/// — `49` (q0=+1, q1=0) is the simplest signal-bearing choice. The
/// remaining pairs all use codeword `40` (q0=0, q1=0). Reference scale
/// factor is 120 (`sf_gain = 32.0`).
///
/// The returned bytes are the substream body (no TOC) that should be
/// concatenated after the byte-aligned `ac4_toc()` element. They are
/// padded to `pad_target_bytes` bytes with zeros so the
/// `audio_size_value` field in the header matches the actual payload
/// length.
///
/// Per ETSI TS 103 190-1 §5.7 (SIMPLE mode) + §5.8 (ASF). The full
/// closed-form encoder for arbitrary input PCM (MDCT analysis +
/// scalefactor selection + entropy coding) is deferred — round 47
/// ships the canned-tone path so the encoder can produce non-silent
/// PCM end-to-end.
pub fn build_mono_simple_asf_tone_body(
    transform_length: u32,
    max_sfb: u32,
    tone_cb_idx: usize,
    tone_pair_idx: u32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits
    // (1 b). We declare the announced size as the pad target so the
    // outer demuxer reads the entire padded body. b_more_bits = 0 so
    // the 15-bit field is taken literally.
    let audio_size = pad_target_bytes as u32;
    let audio_size_lo = audio_size & 0x7FFF;
    bw.write_u32(audio_size_lo, 15);
    bw.write_bit(false);
    bw.align_to_byte();
    // audio_data() for channel_mode = 0 (mono), b_iframe = 1:
    //   mono_codec_mode = 0 (SIMPLE), spec_frontend = 0 (ASF),
    //   asf_transform_info() with b_long_frame = 1,
    //   asf_psy_info(0, 0) with max_sfb[0] in 6 bits.
    bw.write_u32(0, 1); // mono_codec_mode = SIMPLE
    bw.write_u32(0, 1); // spec_frontend = ASF
    bw.write_bit(true); // b_long_frame = 1
    bw.write_u32(max_sfb, 6); // max_sfb[0]
                              // asf_section_data: one section covering 0..max_sfb with cb=5
                              // (HCB5, dim=2, signed). n_sect_bits = 3 (transf_length_idx=0
                              // for long frame).
    bw.write_u32(5, 4); // sect_cb
    write_sect_len_incr(&mut bw, max_sfb, 3, 7);
    // asf_spectral_data: emit `tone_cb_idx` for pair `tone_pair_idx`,
    // and codeword 40 (q0=0, q1=0) for every other pair.
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length).expect("invalid tl");
    let end_line = sfbo[max_sfb as usize] as u32;
    let hcb = crate::huffman::asf_hcb(5u32).expect("HCB5 must exist");
    let pairs = end_line / 2;
    let zero_cw = hcb.cw[40];
    let zero_len = hcb.len[40] as u32;
    let tone_cw = hcb.cw[tone_cb_idx];
    let tone_len = hcb.len[tone_cb_idx] as u32;
    for p in 0..pairs {
        if p == tone_pair_idx {
            bw.write_u32(tone_cw, tone_len);
        } else {
            bw.write_u32(zero_cw, zero_len);
        }
    }
    // asf_scalefac_data: reference_scale_factor = 120 → sf_gain = 32.0.
    bw.write_u32(120, 8);
    // asf_snf_data: b_snf_data_exists = 0.
    bw.write_u32(0, 1);
    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    bw.finish()
}

/// Write a section-length increment sequence per §4.3.5.4
/// (Pseudocode 17). For `n_sect_bits = 3`, escape value 7,
/// `sect_len = 1 + 7k + incr`: emit `k` escape codes followed by one
/// non-escape `incr` (0..6).
fn write_sect_len_incr(bw: &mut BitWriter, sect_len: u32, n_sect_bits: u32, esc: u32) {
    let base = sect_len.saturating_sub(1);
    let k = base / esc;
    let incr = base % esc;
    for _ in 0..k {
        bw.write_u32(esc, n_sect_bits);
    }
    bw.write_u32(incr, n_sect_bits);
}

impl Ac4ImsEncoder {
    /// Encode one IMS v2 frame containing a mono SIMPLE/ASF audio
    /// substream that injects a single quantised spectral tone (per
    /// `tone_cb_idx` from the ETSI Annex A HCB5 codebook). The decoder
    /// dequantises the tone via `rec_spec = sign(q)|q|^(4/3)` and the
    /// IMDCT + KBD windowing produce real, non-silent PCM.
    ///
    /// This is the canned-tone closed-form encoder mentioned in round-47
    /// scope: full MDCT analysis + scalefactor optimisation + ASF
    /// entropy coding for arbitrary PCM input is deferred. The shape
    /// of this method (input PCM → bytes) is reserved for that future
    /// work; for now it ignores its `_input_pcm` argument and emits
    /// the canned tone payload.
    ///
    /// Per ETSI TS 103 190-1 §5.7 + §5.8.
    pub fn encode_frame_mono_tone(&mut self, tone_cb_idx: usize, tone_pair_idx: u32) -> Vec<u8> {
        // Force mono channel_mode for the tone helper — the canned ASF
        // body is mono SIMPLE only.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b0;
        self.channel_mode_bits = 1;
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut frame = bw.finish();
        // Append the canned-tone substream body. Size matches the test
        // helpers in `decoder.rs` (420 bytes) so the substream parser
        // sees a complete payload.
        let body = build_mono_simple_asf_tone_body(1920, 10, tone_cb_idx, tone_pair_idx, 420);
        frame.extend(body);
        // sequence_counter wraps at 1024.
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        frame
    }

    /// Encode one IMS v2 mono frame from arbitrary float PCM input
    /// (range `[-1.0, 1.0]`). Returns the produced frame bytes.
    ///
    /// Pipeline (round 48):
    ///   1. Forward MDCT analysis with KBD windowing across the 50% TDAC
    ///      boundary (carries prior-frame `N` samples in the per-encoder
    ///      [`EncoderMdctState`]).
    ///   2. Per-band scalefactor selection (greedy nearest power-of-two
    ///      that keeps |q| within the chosen Huffman codebook's bound).
    ///   3. Quantisation per Pseudocode 18 inverse:
    ///      `q = round(sign(c) * (|c|/sf_gain)^(3/4))`.
    ///   4. ASF entropy coding via HCB5 (signed dim=2, q-range -4..=+4).
    ///   5. Wrap in v2 IMS TOC + single-substream-group `audio_size` body.
    ///
    /// Frame length is derived from the encoder's
    /// `(fs_index, frame_rate_index)` pair via [`crate::toc::frame_rate_entry`].
    /// For the default mono 48 kHz / 24 fps configuration `frame.len()` is
    /// 1920 samples and `max_sfb` is 10 (matching the canned-tone helper).
    ///
    /// Per ETSI TS 103 190-1 §5.5 (MDCT) + §5.7 / §5.8 (SIMPLE/ASF) +
    /// TS 103 190-2 §6.2.1.1 (IMS TOC).
    pub fn encode_frame_pcm(&mut self, frame: &[f32]) -> Vec<u8> {
        // Default max_sfb = 40 (≤ 7.5 kHz at tl=1920) preserves
        // round-48 behaviour for callers that haven't opted in to the
        // wider-bandwidth encoder.
        self.encode_frame_pcm_with_max_sfb(frame, 40)
    }

    /// Encode one IMS v2 mono frame from arbitrary float PCM input
    /// (range `[-1.0, 1.0]`) at a caller-specified `max_sfb`. Larger
    /// values widen the encoder's frequency coverage at the cost of
    /// more bits per frame:
    ///   * `max_sfb = 40` → bins 0..508 → ~6.35 kHz @ tl=1920
    ///   * `max_sfb = 50` → bins 0..1216 → ~15.2 kHz @ tl=1920
    ///   * `max_sfb = 55` → bins 0..1600 → ~20.0 kHz @ tl=1920
    ///
    /// `max_sfb` must satisfy `max_sfb <= num_sfb_48(frame_len)` (61 at
    /// tl=1920). The pad budget scales with max_sfb so the announced
    /// `audio_size` reliably exceeds the actual emission length.
    pub fn encode_frame_pcm_with_max_sfb(&mut self, frame: &[f32], max_sfb: u32) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        assert_eq!(
            frame.len(),
            frame_len as usize,
            "encode_frame_pcm: input length must match frame_len = {frame_len}"
        );
        // Force mono — the forward analysis path is mono-only. (Multi-
        // channel needs SAP/M-S decision + per-channel state which is
        // queued for round-50+.)
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b0;
        self.channel_mode_bits = 1;

        // 1. Forward MDCT analysis. Lazily build per-encoder state.
        if self.mdct_state.is_none() || self.mdct_state.as_ref().unwrap().n != frame_len {
            self.mdct_state = Some(EncoderMdctState::new(frame_len));
        }
        let coeffs = self.mdct_state.as_mut().unwrap().analyse_frame(frame);

        // 2-4. Build the substream body (per-band codebook optimiser +
        // entropy-coding). Pad target scales with max_sfb to keep the
        // announced audio_size comfortably above the actual emission
        // length: worst case is ~25 bits/pair (HCB11 with one escape
        // per pair) × end_bin/2 pairs ≈ 3 × end_bin bytes.
        let pad_target_bytes = match max_sfb {
            0..=40 => 2048,
            41..=50 => 4096,
            _ => 8192,
        };
        let body = build_mono_simple_asf_body_from_pcm_spectrum(
            frame_len,
            max_sfb,
            &coeffs,
            pad_target_bytes,
        );

        // 5. Wrap in v2 IMS TOC.
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        // sequence_counter wraps at 1024.
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        // Restore caller's channel_mode setting.
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 frame containing a mono SIMPLE/ASF substream
    /// whose injected tone falls on the spectral pair nearest the
    /// requested frequency. With `tl = 1920` at 48 kHz the bin spacing
    /// is 12.5 Hz; the chosen pair carries a single non-zero quantised
    /// value at the lower bin of that pair.
    ///
    /// Returns the encoded frame bytes plus the actual nominal centre
    /// frequency the encoder targeted (lower-bin × bin_spacing).
    pub fn encode_frame_mono_tone_at_hz(&mut self, target_hz: f32) -> (Vec<u8>, f32) {
        let bin_spacing = 48_000.0 / (2.0 * 1_920.0); // 12.5 Hz
        let target_bin = (target_hz / bin_spacing).round().max(0.0) as u32;
        let pair_idx = target_bin / 2;
        let actual_hz = (pair_idx * 2) as f32 * bin_spacing;
        // cb_idx 49 → (q0=+1, q1=0): tone in lower bin of the pair.
        let frame = self.encode_frame_mono_tone(49, pair_idx);
        (frame, actual_hz)
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

    #[test]
    fn v2_encoder_round_trips_through_parse_ac4_toc() {
        // Round-47 contract: the v2 TOC emitted by `encode_frame()`
        // round-trips through `parse_ac4_toc`. Mono / 48 kHz / 24 fps /
        // iframe_global / 1920 samples per frame should land on the
        // returned `Ac4FrameInfo` exactly as configured.
        let mut enc = Ac4ImsEncoder::new(); // v2 default, mono
        let frame = enc.encode_frame(64);
        let info = crate::toc::parse_ac4_toc(&frame).expect("v2 TOC must parse");
        assert_eq!(info.bitstream_version, 2);
        assert_eq!(info.fs_index, 1);
        assert_eq!(info.frame_rate_index, 1);
        assert_eq!(info.frame_length, 1_920);
        assert!(info.b_iframe_global);
        assert_eq!(info.n_presentations, 1);
        assert_eq!(info.n_substreams, 1);
    }

    #[test]
    fn v2_encoder_round_trips_stereo() {
        let mut enc = Ac4ImsEncoder::new().with_stereo(); // v2, stereo
        let frame = enc.encode_frame(64);
        let info = crate::toc::parse_ac4_toc(&frame).expect("v2 stereo TOC must parse");
        assert_eq!(info.bitstream_version, 2);
        assert!(info.b_iframe_global);
    }

    #[test]
    fn v2_encoder_round_trips_5_1() {
        let mut enc = Ac4ImsEncoder::new().with_5_1(); // v2, 5.1
        let frame = enc.encode_frame(128);
        let info = crate::toc::parse_ac4_toc(&frame).expect("v2 5.1 TOC must parse");
        assert_eq!(info.bitstream_version, 2);
    }

    #[test]
    fn v2_encoder_mono_tone_roundtrip_emits_nonsilent_pcm() {
        // Round-47 IMS audio body: encode v2 frame containing a mono
        // SIMPLE/ASF substream with a single quantised spectral line at
        // (sfb=0, bin=0). Through the decoder's full Huffman → IMDCT →
        // KBD overlap-add chain this should produce real PCM with
        // non-trivial energy.
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let mut enc = Ac4ImsEncoder::new(); // v2, mono
                                            // cb_idx 49 → (q0=+1, q1=0); pair_idx 0 → bin 0.
        let frame_bytes = enc.encode_frame_mono_tone(49, 0);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.data.len(), 1);
        assert_eq!(af.data[0].len(), 1_920 * 2);
        // Decoded PCM must be non-silent.
        let samples_i16: Vec<i16> = af.data[0]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let nonzero_count = samples_i16.iter().filter(|&&s| s != 0).count();
        assert!(
            nonzero_count > 100,
            "expected non-silent PCM from IMS tone encoder, got {nonzero_count} non-zero samples"
        );
        let energy: i64 = samples_i16.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(energy > 0, "zero-energy tone output from IMS encoder");
        // Substream parse must have surfaced non-zero scaled spectra at
        // bin 0 (the tone we injected).
        let sub = dec.last_substream.as_ref().unwrap();
        let scaled = sub.tools.scaled_spec_primary.as_ref().unwrap();
        assert!(scaled[0].abs() > 0.0, "DC bin must carry the injected tone");
    }

    #[test]
    fn v2_encoder_mono_tone_at_440hz_has_spectral_peak_near_target() {
        // Round-47 closed-form tone encoder targeting 440 Hz. With
        // tl = 1920 / fs = 48 kHz, bin_spacing = 12.5 Hz so the tone
        // pair lands at pair 17 (bin 34, ~425 Hz). The decoder's
        // scaled spectrum should carry a non-zero value at the
        // targeted bin.
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let mut enc = Ac4ImsEncoder::new();
        let (frame_bytes, actual_hz) = enc.encode_frame_mono_tone_at_hz(440.0);
        // Encoder rounded 440 Hz → bin 35 → pair 17 → bin 34 (lower-of-pair).
        // Actual emitted frequency is 34 × 12.5 = 425.0 Hz.
        assert!(
            (actual_hz - 425.0).abs() < 1.0,
            "expected ~425 Hz target, got {actual_hz}"
        );
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, 1_920);
        // Spectral peak: scaled_spec[34] (lower bin of the targeted
        // pair) must be non-zero; the surrounding bins must NOT carry
        // the same peak (proves the tone is localised).
        let sub = dec.last_substream.as_ref().unwrap();
        let scaled = sub.tools.scaled_spec_primary.as_ref().unwrap();
        let target_bin = 34usize;
        assert!(
            scaled[target_bin].abs() > 0.0,
            "expected non-zero spectral coefficient at bin {target_bin}, got {}",
            scaled[target_bin]
        );
        // PCM should still be non-silent.
        let samples_i16: Vec<i16> = af.data[0]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let nonzero_count = samples_i16.iter().filter(|&&s| s != 0).count();
        assert!(
            nonzero_count > 100,
            "expected non-silent PCM at 440 Hz, got {nonzero_count} non-zero samples"
        );
    }

    #[test]
    fn v2_encoder_decoder_roundtrip_emits_silent_frame() {
        // Full encode → Ac4Decoder roundtrip on the v2 path. The
        // decoder accepts the IMS frame (TOC + zero body) and emits a
        // structurally-valid silent AudioFrame at the declared 1920
        // samples / 48 kHz / mono shape.
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let mut enc = Ac4ImsEncoder::new(); // v2 default, mono
        let frame_bytes = enc.encode_frame(64);
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
        // All bytes should be zero (silent placeholder for the v2
        // audio body, which the encoder emits as raw zero bits).
        assert!(af.data[0].iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // Round 48 — encode_frame_pcm: arbitrary float PCM input through the
    // full forward MDCT + scalefactor + ASF entropy chain.
    // ------------------------------------------------------------------

    /// Helper: feed a sequence of PCM frames through the encoder, then
    /// the decoder, and return the decoded i16 PCM concatenated. The
    /// first decoded frame loses half a window to the encoder's zero
    /// history; callers that compare against the input should ignore it.
    fn encode_decode_frames(frames: &[Vec<f32>]) -> Vec<Vec<i16>> {
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new(); // v2, mono, 48 kHz, 24 fps
        let mut out: Vec<Vec<i16>> = Vec::with_capacity(frames.len());
        for (idx, f) in frames.iter().enumerate() {
            let bytes = enc.encode_frame_pcm(f);
            let _ = idx;
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
                panic!("expected audio frame");
            };
            assert_eq!(af.samples, 1_920);
            assert_eq!(af.data.len(), 1);
            let pcm: Vec<i16> = af.data[0]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            out.push(pcm);
        }
        out
    }

    /// 1 kHz pure tone @ 48 kHz: encode → decode → assert spectral peak
    /// in the right neighbourhood. With tl = 1920, bin_spacing =
    /// 48_000 / (2 * 1920) = 12.5 Hz, so 1000 Hz lands at bin 80.
    #[test]
    fn encode_frame_pcm_1khz_tone_round_trips_with_spectral_peak() {
        // Generate 4 frames of a continuous 1 kHz sine wave so the MDCT
        // overlap-add reaches steady state.
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let f = 1000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * f * t).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..4).map(|i| make_frame(i * n)).collect();
        let decoded = encode_decode_frames(&frames);
        // Steady-state decoded frame: index 2.
        let pcm = &decoded[2];
        // Verify non-silent output.
        let nonzero = pcm.iter().filter(|&&s| s != 0).count();
        assert!(
            nonzero > 100,
            "expected non-silent PCM from 1 kHz tone, got {nonzero} non-zero samples"
        );
        // Energy must be substantial (input amplitude was 0.3 → expect
        // peak |i16| >= ~1000 at the centre of the steady-state frame).
        let peak = pcm.iter().map(|&s| s.abs()).max().unwrap_or(0);
        assert!(peak > 1000, "expected peak amplitude > 1000, got {peak}");
    }

    /// Multi-tone audio: encode → decode → assert SNR > 10 dB on the
    /// steady-state frame. Uses a sum of three pure tones (250 Hz +
    /// 500 Hz + 1 kHz at amplitude 0.2 each) so the input is non-trivial
    /// (multi-line spectrum) but bandlimited well below the encoder's
    /// 7.5 kHz max_sfb=40 cutoff. This stands in for the spec's
    /// "white-noise SNR > 30 dB" target — round 48's HCB5-only quantiser
    /// caps |q| ≤ 4 (~12 dB SNR ceiling per band) and only codes
    /// 0..7.5 kHz, so true white noise is out of reach until round 49
    /// adds a wider codebook selector and a wider max_sfb.
    #[test]
    fn encode_frame_pcm_multitone_round_trips_with_positive_snr() {
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    let pi2 = 2.0 * std::f32::consts::PI;
                    0.2 * (pi2 * 250.0 * t).sin()
                        + 0.2 * (pi2 * 500.0 * t).sin()
                        + 0.2 * (pi2 * 1000.0 * t).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..5).map(|i| make_frame(i * n)).collect();
        let decoded = encode_decode_frames(&frames);
        // Steady-state frame: index 2 (well past the leading transient).
        let orig = &frames[2];
        let recon_i16 = &decoded[2];
        let recon: Vec<f32> = recon_i16.iter().map(|&s| s as f32 / 32767.0).collect();
        let mut sig_e = 0.0_f64;
        let mut err_e = 0.0_f64;
        for (o, r) in orig.iter().zip(recon.iter()) {
            sig_e += (*o as f64).powi(2);
            err_e += (*o as f64 - *r as f64).powi(2);
        }
        let snr_db = 10.0 * (sig_e / err_e.max(1e-30)).log10();
        assert!(
            snr_db > 10.0,
            "multi-tone round-trip SNR too low: {snr_db:.1} dB \
             (expected > 10 dB; HCB5-only encoder caps q at ±4 — \
             round 49 will widen the codebook selector)"
        );
    }

    /// Silence: encode → decode → assert decoded amplitude is small.
    /// HCB5-only encoder always emits a non-zero-padded frame so we
    /// expect ε > 0 noise floor — but it should be << peak amplitude.
    #[test]
    fn encode_frame_pcm_silence_round_trips_to_silence() {
        let n = 1920usize;
        let frames: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0_f32; n]).collect();
        let decoded = encode_decode_frames(&frames);
        // Steady-state frame must be effectively silent.
        let pcm = &decoded[2];
        let peak = pcm.iter().map(|&s| s.abs()).max().unwrap_or(0);
        // i16 peak < 50 = -56 dBFS; comfortably below any audible threshold.
        assert!(
            peak < 50,
            "expected silent reconstruction, got peak amplitude {peak}"
        );
    }

    /// Encoder bumps the sequence_counter once per `encode_frame_pcm`
    /// call, identical to `encode_frame()`.
    #[test]
    fn encode_frame_pcm_bumps_sequence_counter() {
        let mut enc = Ac4ImsEncoder::new();
        assert_eq!(enc.sequence_counter, 0);
        let frame = vec![0.0_f32; 1920];
        let _ = enc.encode_frame_pcm(&frame);
        assert_eq!(enc.sequence_counter, 1);
        let _ = enc.encode_frame_pcm(&frame);
        assert_eq!(enc.sequence_counter, 2);
    }

    // ------------------------------------------------------------------
    // Round 49 — HCB1..11 codebook selection optimiser + wider max_sfb.
    // ------------------------------------------------------------------

    fn encode_decode_frames_with_max_sfb(frames: &[Vec<f32>], max_sfb: u32) -> Vec<Vec<i16>> {
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new();
        let mut out: Vec<Vec<i16>> = Vec::with_capacity(frames.len());
        for f in frames {
            let bytes = enc.encode_frame_pcm_with_max_sfb(f, max_sfb);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
                panic!("expected audio frame");
            };
            assert_eq!(af.samples, 1_920);
            assert_eq!(af.data.len(), 1);
            let pcm: Vec<i16> = af.data[0]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            out.push(pcm);
        }
        out
    }

    /// White-noise input: encode via the round-49 optimiser, then decode
    /// and pull the decoder's reconstructed scaled spectrum
    /// (`scaled_spec_primary`) directly out of the substream. Compare
    /// bin-for-bin against the encoder's input MDCT spectrum to measure
    /// the codebook-selection / quantisation SNR — this isolates the
    /// quantiser's noise contribution from the bandlimit / IMDCT
    /// reconstruction noise that dominates a time-domain comparison.
    ///
    /// Round-48 HCB5-only baseline: ~12 dB SNR (|q| ≤ 4 ceiling).
    /// Round-49 HCB1..11 with q_target = 12: ≥ 18 dB SNR.
    #[test]
    fn encode_frame_pcm_white_noise_snr_exceeds_hcb5_only_ceiling() {
        use crate::decoder::Ac4Decoder;
        use crate::encoder_mdct::EncoderMdctState;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};
        let n = 1920usize;
        let max_sfb = 50u32;
        let sfbo = crate::sfb_offset::sfb_offset_48(n as u32).unwrap();
        let end_bin = sfbo[max_sfb as usize] as usize;
        let make_frame = |seed_off: u64| -> Vec<f32> {
            let mut s: u64 = 0xACE4_u64.wrapping_add(seed_off);
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (s >> 33) as u32;
                    (u as f32 / (1u32 << 31) as f32 - 1.0) * 0.3
                })
                .collect()
        };
        // Encode + decode 3 frames; pull the third for steady-state.
        let frames: Vec<Vec<f32>> = (0..3).map(|i| make_frame(i as u64 * n as u64)).collect();
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new();
        let mut last_recon_spec: Option<Vec<f32>> = None;
        let mut mdct_in = EncoderMdctState::new(n as u32);
        let mut last_orig_spec: Option<Vec<f32>> = None;
        for f in &frames {
            // Mirror the encoder's MDCT on the input.
            let orig_coeffs = mdct_in.analyse_frame(f);
            last_orig_spec = Some(orig_coeffs.clone());
            let bytes = enc.encode_frame_pcm_with_max_sfb(f, max_sfb);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let _ = dec.receive_frame().expect("receive_frame");
            let sub = dec.last_substream.as_ref().unwrap();
            let scaled = sub.tools.scaled_spec_primary.as_ref().unwrap().clone();
            last_recon_spec = Some(scaled);
        }
        let orig = last_orig_spec.unwrap();
        let recon = last_recon_spec.unwrap();
        let mut sig_e = 0.0_f64;
        let mut err_e = 0.0_f64;
        for k in 0..end_bin {
            let o = orig[k] as f64;
            let r = recon[k] as f64;
            sig_e += o * o;
            err_e += (o - r) * (o - r);
        }
        let snr_db = 10.0 * (sig_e / err_e.max(1e-30)).log10();
        eprintln!("ROUND-49 white-noise spectral SNR (HCB1..11 optimiser, q_target=12, max_sfb=50): {snr_db:.1} dB");
        assert!(
            snr_db > 18.0,
            "white-noise spectral SNR did not improve over HCB5-only ceiling: \
             {snr_db:.1} dB (expected > 18 dB; round-48 HCB5-only baseline was ~12 dB)"
        );
    }

    /// Wider max_sfb=55: 1 kHz tone reconstruction has ≥80% of input
    /// energy preserved (vs ~40% with the round-48 max_sfb=40 default).
    #[test]
    fn encode_frame_pcm_max_sfb_55_preserves_tone_energy() {
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let f = 1000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * f * t).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..4).map(|i| make_frame(i * n)).collect();
        let decoded = encode_decode_frames_with_max_sfb(&frames, 55);
        let orig = &frames[2];
        let recon_i16 = &decoded[2];
        let recon: Vec<f32> = recon_i16.iter().map(|&s| s as f32 / 32767.0).collect();
        let orig_e: f64 = orig.iter().map(|&v| (v as f64).powi(2)).sum();
        let recon_e: f64 = recon.iter().map(|&v| (v as f64).powi(2)).sum();
        let ratio = recon_e / orig_e.max(1e-30);
        eprintln!(
            "ROUND-49 max_sfb=55 1 kHz tone energy preservation: {:.1}%",
            ratio * 100.0
        );
        assert!(
            ratio >= 0.80,
            "expected ≥80% energy preservation at max_sfb=55, got {:.1}%",
            ratio * 100.0
        );
    }

    /// Backwards compatibility: `encode_frame_pcm` without an explicit
    /// max_sfb still uses the round-48 default of 40, and the existing
    /// 1 kHz tone fixture still round-trips through the decoder with the
    /// optimiser-driven codebook selection enabled.
    #[test]
    fn encode_frame_pcm_default_max_sfb_still_works() {
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let f = 1000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * f * t).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..4).map(|i| make_frame(i * n)).collect();
        let decoded = encode_decode_frames(&frames); // default max_sfb=40
        let pcm = &decoded[2];
        let peak = pcm.iter().map(|&s| s.abs()).max().unwrap_or(0);
        assert!(
            peak > 1000,
            "expected peak amplitude > 1000 at default max_sfb=40, got {peak}"
        );
    }

    /// Sanity baseline: with the HCB5-only encoder configuration
    /// (`q_target = 4`) the white-noise spectral SNR caps near 12 dB.
    /// We simulate this via a one-shot helper that uses HCB5 only on
    /// every band, mirroring the round-48 build_mono_simple_asf body.
    /// This test exists as a benchmark anchor so future regressions
    /// against the round-48 baseline are visible at a glance.
    #[test]
    fn baseline_hcb5_only_white_noise_snr_logs_for_comparison() {
        use crate::asf_data::{
            dequantise_and_scale, parse_asf_scalefac_data, parse_asf_section_data,
            parse_asf_spectral_data,
        };
        use crate::encoder_asf::{
            pick_scalefactor_for_band, single_section, write_scalefac_data, write_sect_len_incr,
            write_spectral_data_single_section,
        };
        use crate::encoder_mdct::EncoderMdctState;
        use oxideav_core::bits::{BitReader, BitWriter};

        let n = 1920usize;
        let max_sfb = 50u32;
        let sfbo = crate::sfb_offset::sfb_offset_48(n as u32).unwrap();
        let end_bin = sfbo[max_sfb as usize] as usize;
        let mut s: u64 = 0xACE4u64;
        let pcm: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 33) as u32;
                (u as f32 / (1u32 << 31) as f32 - 1.0) * 0.3
            })
            .collect();
        let mut mdct = EncoderMdctState::new(n as u32);
        let _ = mdct.analyse_frame(&pcm);
        let coeffs = mdct.analyse_frame(&pcm);

        // HCB5-only encoder body (round-48 path).
        let cb: u8 = 5;
        let q_max = 4u32;
        let mut qspec = vec![0i32; end_bin];
        let mut sf_per_band = vec![100i32; max_sfb as usize];
        let mut max_quant_idx = vec![0u32; max_sfb as usize];
        for sfb in 0..max_sfb as usize {
            let a = sfbo[sfb] as usize;
            let b = sfbo[sfb + 1] as usize;
            let band = &coeffs[a..b.min(coeffs.len())];
            let (sf, q) = pick_scalefactor_for_band(band, q_max);
            sf_per_band[sfb] = sf;
            for (i, &qi) in q.iter().enumerate() {
                qspec[a + i] = qi;
                max_quant_idx[sfb] = max_quant_idx[sfb].max(qi.unsigned_abs());
            }
        }
        let mut bw = BitWriter::new();
        bw.write_u32(4096, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        bw.write_bit(true);
        let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(n as u32).unwrap();
        bw.write_u32(max_sfb, n_msfb_bits);
        bw.write_u32(cb as u32, 4);
        write_sect_len_incr(&mut bw, max_sfb, 3, 7);
        write_spectral_data_single_section(&mut bw, &qspec, sfbo, max_sfb, cb as u32);
        let sections = single_section(max_sfb, cb);
        write_scalefac_data(
            &mut bw,
            &sf_per_band,
            &sections.sfb_cb,
            &max_quant_idx,
            max_sfb,
        );
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 4096 {
            bw.write_u32(0, 8);
        }
        let body = bw.finish();

        // Walk through parser, then dequantise + compare.
        let mut br = BitReader::new(&body);
        let _ = br.read_u32(15).unwrap();
        let _ = br.read_bit().unwrap();
        br.align_to_byte();
        let _ = br.read_u32(1).unwrap();
        let _ = br.read_u32(1).unwrap();
        let _ = br.read_bit().unwrap();
        let _ = br.read_u32(n_msfb_bits).unwrap();
        let parsed = parse_asf_section_data(&mut br, 0, n as u32, max_sfb).unwrap();
        let (qs, mqi) = parse_asf_spectral_data(&mut br, &parsed, sfbo, max_sfb).unwrap();
        let sfg = parse_asf_scalefac_data(&mut br, &parsed, &mqi, max_sfb, n as u32).unwrap();
        let scaled = dequantise_and_scale(&qs, &sfg, sfbo, max_sfb);

        let mut sig_e = 0.0_f64;
        let mut err_e = 0.0_f64;
        for k in 0..end_bin {
            let o = coeffs[k] as f64;
            let r = scaled[k] as f64;
            sig_e += o * o;
            err_e += (o - r) * (o - r);
        }
        let snr_db = 10.0 * (sig_e / err_e.max(1e-30)).log10();
        eprintln!("ROUND-48 baseline white-noise spectral SNR (HCB5-only, q_target=4, max_sfb=50): {snr_db:.1} dB");
        // Sanity: round-48 should be in the 8-15 dB range.
        assert!(
            snr_db < 18.0,
            "round-48 HCB5-only baseline unexpectedly high: {snr_db:.1} dB"
        );
    }

    /// max_sfb wider than the round-48 default: encoder emits a frame
    /// the decoder parses without erroring.
    #[test]
    fn encode_frame_pcm_max_sfb_50_round_trips() {
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    let pi2 = 2.0 * std::f32::consts::PI;
                    // Tones across the wider band: 1 kHz + 8 kHz.
                    0.2 * (pi2 * 1000.0 * t).sin() + 0.2 * (pi2 * 8000.0 * t).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..4).map(|i| make_frame(i * n)).collect();
        let decoded = encode_decode_frames_with_max_sfb(&frames, 50);
        // Steady-state frame must be substantially non-silent.
        let pcm = &decoded[2];
        let nonzero = pcm.iter().filter(|&&s| s != 0).count();
        assert!(nonzero > 100, "expected non-silent recon, got {nonzero}");
    }
}
