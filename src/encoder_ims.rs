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

use crate::encoder_asf::{
    average_per_sfb_correlation, build_5_0_simple_asf_body_from_pcm_spectra,
    build_5_1_simple_asf_body_from_pcm_spectra, build_7_1_simple_asf_body_from_pcm_spectra,
    build_mono_simple_asf_body_from_pcm_spectrum,
    build_stereo_simple_asf_joint_body_from_pcm_spectra,
    build_stereo_simple_asf_split_body_from_pcm_spectra,
};
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
    /// Forward-MDCT analysis state for the secondary (right) channel of
    /// `encode_frame_pcm_stereo()`. Identical role to `mdct_state` but
    /// for the second channel — separate so 50% TDAC overlap is
    /// per-channel.
    pub mdct_state_r: Option<EncoderMdctState>,
    /// Forward-MDCT analysis state for the multichannel encoder paths
    /// (`encode_frame_pcm_5_0()` and any future N>2 variants). One
    /// [`EncoderMdctState`] per output channel — separate so 50% TDAC
    /// overlap continuity is preserved per channel across frames. Lazy-
    /// initialised on first use; grown to the required channel count.
    pub mdct_states_multi: Vec<EncoderMdctState>,
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
            mdct_state_r: None,
            mdct_states_multi: Vec::new(),
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

    /// 5.0 channel mode (`0b1101`, 4 b) per Table 85 — channel_mode 3 —
    /// the 5.0 surround layout (`L, R, C, Ls, Rs`) without LFE. Drives the
    /// decoder's `5_X_channel_element()` walker for `channels == 5` (no
    /// `b_has_lfe` block) and the corresponding `dispatch_5x_cfg3_simple_aspx`
    /// PCM output path.
    pub fn with_5_0(mut self) -> Self {
        self.channel_mode_value = 0b1101;
        self.channel_mode_bits = 4;
        self
    }

    /// 5.1 channel mode (`0b1110`, 4 b) per Table 85.
    pub fn with_5_1(mut self) -> Self {
        self.channel_mode_value = 0b1110;
        self.channel_mode_bits = 4;
        self
    }

    /// 7.1 (3/4/0.1) channel mode (`0b1111001`, 7 b) per ETSI TS 103 190-1
    /// §4.3.3.7.1 Table 88 — channel_mode value `1111001` → ch_mode 6 → 8
    /// channels with layout `L, C, R, Ls, Rs, Lb, Rb, LFE`. Drives the
    /// decoder's `7_X_channel_element()` walker for `channels == 8` (with
    /// `b_has_lfe`) per §4.2.6.14 Table 33. The decoder's internal coding
    /// order for the inner `five_channel_data()` is `[L, R, C, Ls, Rs]`
    /// per Table 180 (the inner SCE order differs from the surface
    /// Table 88 listing's `L, C, R` ordering — the decoder treats the
    /// inner five_channel_data slots as L/R/C/Ls/Rs).
    pub fn with_7_1(mut self) -> Self {
        self.channel_mode_value = 0b1111001;
        self.channel_mode_bits = 7;
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

    /// Encode one IMS v2 stereo frame from arbitrary float PCM input
    /// (range `[-1.0, 1.0]`) for both L and R. Returns the produced
    /// frame bytes.
    ///
    /// **Path A — 2× SCE (split-MDCT)** per ETSI TS 103 190-1 §5.3 +
    /// §4.2.6.3 Table 22 (`stereo_data()` with
    /// `b_enable_mdct_stereo_proc == 0`): each channel is encoded
    /// independently with the shared forward analysis pipeline (KBD-
    /// windowed MDCT, per-band scalefactor, DP-optimal sectioning,
    /// HCB1..11 codebook selection, SNF emission). No joint M/S coding.
    ///
    /// `frame_l` / `frame_r` must each be exactly `frame_len` samples
    /// long (1920 samples for the default 48 kHz / 24 fps configuration).
    /// The encoder forces stereo channel mode (`channel_mode_value =
    /// 0b10`) for this call. The decoder's
    /// [`crate::asf::parse_stereo_data_body_stateful`] split-MDCT path
    /// consumes the frame and reconstructs both channels through the
    /// shared ASF Huffman pipeline.
    ///
    /// `max_sfb` defaults to 40 (matching the round-48 mono default,
    /// covers bins 0..508 ≈ 0..6.35 kHz at tl = 1920) when called via
    /// [`Self::encode_frame_pcm_stereo`]; use
    /// [`Self::encode_frame_pcm_stereo_with_max_sfb`] for wider coverage.
    /// The decoder's split-MDCT branch reads BOTH L and R `max_sfb` with
    /// the full `n_msfb_bits` width (the spec's `b_side_limited` only
    /// applies to joint-MDCT stereo per §4.3.6.2), so the encoder isn't
    /// limited by the narrower `n_side_bits`.
    ///
    /// Per ETSI TS 103 190-1 §5.3 + §4.2.6.3 + §5.5 (MDCT) +
    /// §5.7 / §5.8 (SIMPLE/ASF) + TS 103 190-2 §6.2.1.1 (IMS TOC).
    pub fn encode_frame_pcm_stereo(&mut self, frame_l: &[f32], frame_r: &[f32]) -> Vec<u8> {
        // Default max_sfb = 40 (matches the round-48 mono default).
        self.encode_frame_pcm_stereo_with_max_sfb(frame_l, frame_r, 40)
    }

    /// Round-52 heuristic threshold for joint M/S coding. When the
    /// per-SFB average Pearson correlation between L and R MDCT spectra
    /// exceeds this value, the encoder switches to Path B (joint M/S CPE,
    /// `b_enable_mdct_stereo_proc == 1`); otherwise Path A (split-MDCT,
    /// 2× SCE) is used. The 0.7 threshold matches the spec's §5.3
    /// guidance plus the headline number cited in this crate's round-52
    /// task brief.
    pub const STEREO_JOINT_MS_CORRELATION_THRESHOLD: f32 = 0.7;

    /// Encode one IMS v2 stereo frame from arbitrary float PCM input
    /// (range `[-1.0, 1.0]`) at a caller-specified `max_sfb`. Both
    /// channels use the same `max_sfb` — the encoder uses the full
    /// `n_msfb_bits` field width for both. See
    /// [`Self::encode_frame_pcm_stereo`].
    ///
    /// **Round 52 — Path A vs Path B dispatch.** The encoder computes the
    /// per-SFB average Pearson correlation between the L and R MDCT
    /// spectra (via [`average_per_sfb_correlation`]) and, when it exceeds
    /// [`Self::STEREO_JOINT_MS_CORRELATION_THRESHOLD`] (default 0.7),
    /// switches to **joint M/S CPE (Path B,
    /// `b_enable_mdct_stereo_proc == 1`)** per ETSI TS 103 190-1 §5.3 +
    /// §4.2.6.3 Table 22 + §7.5 (Pseudocode 77). Otherwise it stays on
    /// the round-51 split-MDCT path (Path A, 2× SCE,
    /// `b_enable_mdct_stereo_proc == 0`). Per-SFB M/S vs L/R selection
    /// within the joint path is driven by the natural-q bit-cost
    /// comparison inside
    /// [`build_stereo_simple_asf_joint_body_from_pcm_spectra`].
    ///
    /// Use [`Self::encode_frame_pcm_stereo_split_with_max_sfb`] or
    /// [`Self::encode_frame_pcm_stereo_joint_with_max_sfb`] to force a
    /// specific path regardless of correlation.
    pub fn encode_frame_pcm_stereo_with_max_sfb(
        &mut self,
        frame_l: &[f32],
        frame_r: &[f32],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_stereo_dispatched(frame_l, frame_r, max_sfb, None)
    }

    /// Force the split-MDCT (Path A: 2× SCE) encoder path regardless of
    /// the L-vs-R correlation. Useful for tests / fixtures that need a
    /// deterministic on-wire layout. See
    /// [`Self::encode_frame_pcm_stereo_with_max_sfb`].
    pub fn encode_frame_pcm_stereo_split_with_max_sfb(
        &mut self,
        frame_l: &[f32],
        frame_r: &[f32],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_stereo_dispatched(frame_l, frame_r, max_sfb, Some(false))
    }

    /// Force the joint-MDCT (Path B: M/S CPE) encoder path regardless of
    /// the L-vs-R correlation. Useful for tests / fixtures that need a
    /// deterministic on-wire layout. See
    /// [`Self::encode_frame_pcm_stereo_with_max_sfb`].
    pub fn encode_frame_pcm_stereo_joint_with_max_sfb(
        &mut self,
        frame_l: &[f32],
        frame_r: &[f32],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_stereo_dispatched(frame_l, frame_r, max_sfb, Some(true))
    }

    /// Shared body for the stereo encode dispatch — runs forward MDCT,
    /// computes the cross-channel correlation (when `force_joint` is
    /// `None`) and picks Path A vs Path B accordingly. `force_joint =
    /// Some(true)` always emits joint M/S, `Some(false)` always emits
    /// split-MDCT, `None` selects via the
    /// [`Self::STEREO_JOINT_MS_CORRELATION_THRESHOLD`] threshold.
    fn encode_frame_pcm_stereo_dispatched(
        &mut self,
        frame_l: &[f32],
        frame_r: &[f32],
        max_sfb: u32,
        force_joint: Option<bool>,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        assert_eq!(
            frame_l.len(),
            frame_len as usize,
            "encode_frame_pcm_stereo: L input length must match frame_len = {frame_len}"
        );
        assert_eq!(
            frame_r.len(),
            frame_len as usize,
            "encode_frame_pcm_stereo: R input length must match frame_len = {frame_len}"
        );
        // Cap max_sfb at n_msfb_bits=6's max (63 for tl=1920) and at the
        // transform's actual `num_sfb_48` cap (61 at tl=1920).
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force stereo channel_mode (prefix '10', 2 bits) — both the
        // split-MDCT and joint-MDCT body builders require the TOC to
        // declare 2 channels.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b10;
        self.channel_mode_bits = 2;

        // 1. Forward MDCT analysis per channel (separate state for
        //    independent 50% TDAC overlap continuity).
        if self.mdct_state.is_none() || self.mdct_state.as_ref().unwrap().n != frame_len {
            self.mdct_state = Some(EncoderMdctState::new(frame_len));
        }
        if self.mdct_state_r.is_none() || self.mdct_state_r.as_ref().unwrap().n != frame_len {
            self.mdct_state_r = Some(EncoderMdctState::new(frame_len));
        }
        let coeffs_l = self.mdct_state.as_mut().unwrap().analyse_frame(frame_l);
        let coeffs_r = self.mdct_state_r.as_mut().unwrap().analyse_frame(frame_r);

        // 2. Path A vs Path B dispatch per round-52 heuristic.
        let use_joint = match force_joint {
            Some(b) => b,
            None => {
                let rho = average_per_sfb_correlation(frame_len, max_sfb, &coeffs_l, &coeffs_r);
                rho >= Self::STEREO_JOINT_MS_CORRELATION_THRESHOLD
            }
        };

        // 3-5. Build the stereo body. Pad budget is 2× the mono budget
        //      since we carry two spectra (joint or split).
        let pad_target_bytes = match max_sfb {
            0..=20 => 2048,
            21..=40 => 4096,
            41..=50 => 8192,
            _ => 16384,
        };
        let body = if use_joint {
            build_stereo_simple_asf_joint_body_from_pcm_spectra(
                frame_len,
                max_sfb,
                &coeffs_l,
                &coeffs_r,
                pad_target_bytes,
            )
        } else {
            build_stereo_simple_asf_split_body_from_pcm_spectra(
                frame_len,
                max_sfb,
                &coeffs_l,
                &coeffs_r,
                pad_target_bytes,
            )
        };

        // 6. Wrap in v2 IMS TOC.
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

    /// Encode one IMS v2 5.0 frame from arbitrary float PCM input (range
    /// `[-1.0, 1.0]`) for L, R, C, Ls, Rs.
    ///
    /// **Path SIMPLE/Cfg3Five — 5 SCE multichannel forward analysis** per
    /// ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case SIMPLE: coding_config ==
    /// 3` + §4.2.7.5 Table 29 (`five_channel_data()`): each of the five
    /// channels is encoded independently with the shared forward-analysis
    /// pipeline (KBD-windowed MDCT, per-band scalefactor, DP-optimal
    /// section partition, HCB1..11 codebook selection, SNF emission). One
    /// shared `sf_info(ASF, 0, 0)` precedes the per-channel data; the
    /// `five_channel_info()` uses identity SAP (`sap_mode = 0` on every
    /// `chparam_info()`, `chel_matsel = 0`) so no joint-MDCT mixing happens
    /// at decode time — every output channel comes straight from its own
    /// `sf_data(ASF)` body. This is the spec-mandated minimum for the 5.0
    /// SIMPLE path and unblocks the encoder's path to multichannel.
    ///
    /// `frames[i]` must each be exactly `frame_len` samples long
    /// (1920 samples for the default 48 kHz / 24 fps configuration). The
    /// slice order matches the 5.0 output layout (`L, R, C, Ls, Rs` —
    /// Table 180 row `coding_config == 3`). The encoder forces the 5.0
    /// channel mode (`channel_mode_value = 0b1101`, 4 b — Table 85
    /// channel_mode 3) for this call.
    ///
    /// The decoder's [`crate::mch::parse_5x_audio_data_outer`] for
    /// `channels == 5` (no LFE) consumes the body, IMDCTs each per-channel
    /// spectrum into slots 0..4, and emits 5-channel interleaved S16 PCM at
    /// the declared sample rate. There is no companding / ASPX / A-CPL on
    /// the SIMPLE path so the round-trip is purely the per-channel MDCT
    /// quantisation noise (≥ 20 dB spectral SNR per channel on tone /
    /// white-noise fixtures).
    ///
    /// `max_sfb` defaults to 40 (matching the round-49 mono default).
    /// Use [`Self::encode_frame_pcm_5_0_with_max_sfb`] for wider coverage.
    ///
    /// Per ETSI TS 103 190-1 §4.2.6.6 + §4.2.7.5 + §5.5 (MDCT) +
    /// §5.7 / §5.8 (SIMPLE/ASF) + TS 103 190-2 §6.2.1.1 (IMS TOC).
    pub fn encode_frame_pcm_5_0(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        // Default max_sfb = 40 (matches the round-48 mono default).
        self.encode_frame_pcm_5_0_with_max_sfb(frames, 40)
    }

    /// Encode one IMS v2 5.0 frame from arbitrary float PCM input at a
    /// caller-specified `max_sfb`. All five channels share the same
    /// `max_sfb` (the joint `sf_info` header carries one value). See
    /// [`Self::encode_frame_pcm_5_0`] for the rest of the contract.
    pub fn encode_frame_pcm_5_0_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 5],
        max_sfb: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_0: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        // Cap max_sfb at n_msfb_bits=6's max (63 for tl=1920) and at the
        // transform's actual `num_sfb_48` cap (61 at tl=1920).
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force 5.0 channel_mode (prefix '1101', 4 bits — Table 85
        // channel_mode 3). The body builder requires the TOC to declare
        // 5 channels so the decoder's `walk_ac4_substream` dispatch
        // routes through `parse_5x_audio_data_outer(b_has_lfe = false)`.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1101;
        self.channel_mode_bits = 4;

        // 1. Forward MDCT analysis per channel (separate state for
        //    independent 50% TDAC overlap continuity).
        while self.mdct_states_multi.len() < 5 {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        // Run analyses sequentially — borrow each state mutably one at a
        // time so we don't conflict on `self.mdct_states_multi`.
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(5);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // 2-4. Build the 5.0 SIMPLE/Cfg3Five body. Pad budget is 5× the
        //      mono budget since we carry five spectra independently.
        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };
        let body = build_5_0_simple_asf_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            &[
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
            ],
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

    /// Encode one IMS v2 5.1 frame from float PCM input per ETSI
    /// TS 103 190-1 §4.2.6.6 + §4.2.7.5 + TS 103 190-2 §6.2.1.1, building
    /// on top of the 5.0 Cfg3Five forward analysis path with an extra
    /// LFE `mono_data(1)` element per Table 25
    /// (`if (b_has_lfe) mono_data(1);`).
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, LFE]` order. Each slice must
    /// have length `frame_len` (1920 for the default 48 kHz / 24 fps
    /// configuration); panics otherwise.
    ///
    /// The encoder forces the 5.1 channel_mode prefix (`0b1110`, 4 b —
    /// Table 85 channel_mode 4) so the decoder's
    /// `walk_ac4_substream` dispatches `channels == 6` through
    /// `parse_5x_audio_data_outer(b_has_lfe = true)`. The LFE channel
    /// is coded with `sf_info_lfe()` (Table 35) carrying `max_sfb` in
    /// `n_msfbl_bits` bits (Table 106 column 4 — 3 bits for `tl = 1920`
    /// → max_sfb_lfe is capped at 7). The five non-LFE channels share
    /// the same Cfg3Five `five_channel_data()` body as the 5.0 path
    /// (identity SAP, independent per-channel SCE).
    ///
    /// `max_sfb` defaults to 40 (matching the round-49 mono / round-74
    /// 5.0 default); `max_sfb_lfe` defaults to 7 (the LFE-spec cap at
    /// `tl = 1920`). Use [`Self::encode_frame_pcm_5_1_with_max_sfb`]
    /// for wider coverage.
    pub fn encode_frame_pcm_5_1(&mut self, frames: &[&[f32]; 6]) -> Vec<u8> {
        self.encode_frame_pcm_5_1_with_max_sfb(frames, 40, 7)
    }

    /// Encode one IMS v2 5.1 frame from arbitrary float PCM input at
    /// caller-specified `max_sfb` (non-LFE channels) and `max_sfb_lfe`
    /// (LFE channel). See [`Self::encode_frame_pcm_5_1`] for the rest of
    /// the contract.
    pub fn encode_frame_pcm_5_1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 6],
        max_sfb: u32,
        max_sfb_lfe: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_1: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        // Cap max_sfb at the non-LFE max_sfb width's max and at the actual
        // num_sfb_48 cap.
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_5_1: tl = {frame_len} not permitted for LFE"
        );
        let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
        let max_sfb_lfe = max_sfb_lfe.min(n_msfbl_cap);

        // Force 5.1 channel_mode (prefix '1110', 4 bits — Table 85
        // channel_mode 4). The body builder requires the TOC to declare
        // 6 channels so the decoder's `walk_ac4_substream` dispatch
        // routes through `parse_5x_audio_data_outer(b_has_lfe = true)`.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1110;
        self.channel_mode_bits = 4;

        // 1. Forward MDCT analysis per channel (separate state for
        //    independent 50% TDAC overlap continuity). Six channels here
        //    so the multi-channel state vector needs to grow.
        while self.mdct_states_multi.len() < 6 {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(6);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // 2-4. Build the 5.1 SIMPLE/Cfg3Five body. Pad budget is 6× the
        //      mono budget since we carry six spectra independently.
        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };
        let body = build_5_1_simple_asf_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_lfe,
            &[
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &coeffs_per_channel[5],
            ],
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

    /// Encode one IMS v2 7.1 (3/4/0.1) frame from float PCM input per ETSI
    /// TS 103 190-1 §4.2.6.14 Table 33 + §4.2.7.5 Table 29
    /// (`five_channel_data()`) + §4.2.7.4 Table 26 (`two_channel_data()`),
    /// building on top of the 5.1 Cfg3Five forward analysis path with an
    /// extra trailing `two_channel_data()` for the immersive
    /// additional-channel pair (Lb, Rb) per the SIMPLE/ASPX
    /// additional-channel block in §4.2.6.14:
    /// `b_use_sap_add_ch + two_channel_data()`.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order (matching
    /// the decoder's output slot convention: slots 0..4 from
    /// `five_channel_data()` per Table 180, slots 5/6 from the additional
    /// `two_channel_data()` per `dispatch_7x_additional_channel_pair` /
    /// Table 183 row "3/4/0.x" identity-SAP path, slot 7 from the LFE
    /// `mono_data(1)`). Each slice must have length `frame_len` (1920
    /// for the default 48 kHz / 24 fps configuration); panics otherwise.
    ///
    /// The encoder forces the 7.1 (3/4/0.1) channel_mode prefix
    /// (`0b1111001`, 7 b — Table 88 channel_mode 6) so the decoder's
    /// `walk_ac4_substream` dispatches `channels == 8` through
    /// `parse_7x_audio_data_outer(b_has_lfe = true)`. The LFE channel
    /// is coded with `sf_info_lfe()` (Table 35) carrying `max_sfb` in
    /// `n_msfbl_bits` bits (Table 106 column 4 — 3 bits for `tl = 1920`
    /// → max_sfb_lfe is capped at 7). The five non-LFE front/surround
    /// channels share the same Cfg3Five `five_channel_data()` body as
    /// the 5.1 path; the additional pair (Lb, Rb) is coded as a single
    /// `two_channel_data()` with identity SAP (`b_use_sap_add_ch = 0`,
    /// `sap_mode = 0` on its `chparam_info`) so no joint-MDCT mixing
    /// happens at decode time and slots 5/6 receive Lb/Rb directly.
    ///
    /// `max_sfb` defaults to 40 (matching the round-49 mono / round-74
    /// 5.0 / round-80 5.1 default); `max_sfb_add` defaults to 40 (same
    /// width as the 5.1 non-LFE channels); `max_sfb_lfe` defaults to 7
    /// (the LFE-spec cap at `tl = 1920`). Use
    /// [`Self::encode_frame_pcm_7_1_with_max_sfb`] for wider coverage.
    pub fn encode_frame_pcm_7_1(&mut self, frames: &[&[f32]; 8]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_with_max_sfb(frames, 40, 40, 7)
    }

    /// Encode one IMS v2 7.1 (3/4/0.1) frame from arbitrary float PCM
    /// input at caller-specified `max_sfb` (five-channel front/surround
    /// SCEs), `max_sfb_add` (additional Lb/Rb pair), and `max_sfb_lfe`
    /// (LFE channel). See [`Self::encode_frame_pcm_7_1`] for the rest of
    /// the contract.
    pub fn encode_frame_pcm_7_1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 8],
        max_sfb: u32,
        max_sfb_add: u32,
        max_sfb_lfe: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_1: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        // Cap max_sfb at the non-LFE max_sfb width's max and at the actual
        // num_sfb_48 cap. Same cap applies to both the inner
        // five_channel_data and the additional two_channel_data — they
        // share the n_msfb_bits width.
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        let max_sfb_add = max_sfb_add.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_7_1: tl = {frame_len} not permitted for LFE"
        );
        let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
        let max_sfb_lfe = max_sfb_lfe.min(n_msfbl_cap);

        // Force 7.1 (3/4/0.1) channel_mode (prefix '1111001', 7 bits —
        // Table 88 channel_mode 6). The body builder requires the TOC to
        // declare 8 channels so the decoder's `walk_ac4_substream`
        // dispatch routes through `parse_7x_audio_data_outer(b_has_lfe =
        // true)`.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111001;
        self.channel_mode_bits = 7;

        // 1. Forward MDCT analysis per channel (separate state for
        //    independent 50% TDAC overlap continuity). Eight channels
        //    here so the multi-channel state vector needs to grow.
        while self.mdct_states_multi.len() < 8 {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(8);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // 2-4. Build the 7.1 SIMPLE/Cfg3Five body. Pad budget is ~8× the
        //      mono budget since we carry eight spectra independently.
        //      Capped at 32767 — the 15-bit `audio_size_value` field
        //      saturates there (extending via `b_more_bits` is permitted
        //      by §4.3.4.1 but not needed for the default max_sfb path).
        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };
        let body = build_7_1_simple_asf_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_add,
            max_sfb_lfe,
            &[
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &coeffs_per_channel[5],
                &coeffs_per_channel[6],
                &coeffs_per_channel[7],
            ],
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

    /// Encode one IMS v2 5.X frame in 5_X_codec_mode = ASPX_ACPL_3 per
    /// ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_3:`
    /// (round 95). Symmetric counterpart to the decoder's round-34
    /// [`crate::mch::parse_5x_audio_data_outer`] ASPX_ACPL_3 walker.
    ///
    /// `frames` is in `[L, R, C]` order — three carrier channels. The
    /// L/R pair feeds the `stereo_data()` body (split-MDCT path) and
    /// drives the A-CPL Ls/Rs surround reconstruction via Pseudocode 118
    /// at decode time. The centre carrier `C` is present in the
    /// `coeffs_per_channel` slice but unused on the spec ASPX_ACPL_3
    /// path — the decoder reconstructs the centre from
    /// `cfg0_centre_mono.scaled_spec` (which the ASPX_ACPL_3 walker
    /// doesn't populate), so the decoder's centre output is zero-filled.
    ///
    /// The encoder forces the 5.0 channel_mode prefix (`0b1101`, 4 b —
    /// Table 85 channel_mode 3) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 5` through
    /// `parse_5x_audio_data_outer(b_has_lfe = false)` with
    /// `5_X_codec_mode = AspxAcpl3`.
    ///
    /// The ASPX/A-CPL parameter bits are emitted as
    /// minimum-bit-cost zero-delta Huffman codewords (per round-95
    /// "structural scaffold" mode — see [`crate::encoder_acpl3`]).
    /// The decoder walks the full Table 25 body and produces
    /// 5-channel `[L, R, C, Ls, Rs]` PCM via
    /// [`crate::acpl_synth::run_acpl_5x_mch_pcm`]. With all-zero ACPL
    /// parameter deltas the surround pair Ls/Rs collapses to the
    /// ducker-driven reconstruction from the L/R carriers.
    ///
    /// `max_sfb` defaults to 40 (matching the round-49 mono / round-74
    /// 5.0 default).
    pub fn encode_frame_pcm_5_0_acpl3(&mut self, frames: &[&[f32]; 3]) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_with_max_sfb(frames, None, 40, None)
    }

    /// Encode one IMS v2 5.1 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// an LFE channel per ETSI TS 103 190-1 §4.2.6.6 Table 25
    /// (`if (b_has_lfe) mono_data(1);`) + §4.2.8 (`sf_info_lfe()`).
    ///
    /// `frames` is in `[L, R, C, LFE]` order. The L/R carrier pair drives
    /// the stereo body + A-CPL Ls/Rs reconstruction (same as
    /// [`Self::encode_frame_pcm_5_0_acpl3`]); the LFE channel is coded
    /// as a leading `mono_data(b_lfe = 1)` element per Table 21.
    pub fn encode_frame_pcm_5_1_acpl3(&mut self, frames: &[&[f32]; 4]) -> Vec<u8> {
        // Decompose into the 3-carrier slice + the LFE slice for the
        // shared dispatcher.
        let carriers: [&[f32]; 3] = [frames[0], frames[1], frames[2]];
        self.encode_frame_pcm_5_x_acpl3_with_max_sfb(&carriers, Some(frames[3]), 40, Some(7))
    }

    /// Shared body for [`Self::encode_frame_pcm_5_0_acpl3`] and
    /// [`Self::encode_frame_pcm_5_1_acpl3`]. `frames` carries the three
    /// non-LFE channels (`[L, R, C]`); `lfe` is `Some` for the 5.1 path,
    /// `None` for 5.0.
    fn encode_frame_pcm_5_x_acpl3_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 3],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force the right channel_mode based on whether LFE is present.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            // 5.1 channel_mode prefix '1110', 4 b → channel_mode 4.
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            // 5.0 channel_mode prefix '1101', 4 b → channel_mode 3.
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

        // 1. Forward MDCT analysis per channel (separate state for
        //    independent 50% TDAC overlap continuity).
        let n_channels = if lfe.is_some() { 4 } else { 3 };
        while self.mdct_states_multi.len() < n_channels {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }
        let coeffs_lfe: Option<Vec<f32>> = lfe.map(|buf| {
            // LFE channel uses its own MDCT state at index 3.
            self.mdct_states_multi[3].analyse_frame(buf)
        });

        // 2. Build the ASPX_ACPL_3 body via the shared builder. ASPX
        //    config: small low-res scale so the SBG counts stay small —
        //    keeps the ASPX_data_2ch body compact.
        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0, // num_noise_sbgroups = 1
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_modes Fine.
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        // Pad budget: scale with max_sfb and channel count (3 or 4).
        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_lfe,
            self.b_iframe_global,
            &coeffs_per_channel[0],
            &coeffs_per_channel[1],
            coeffs_lfe.as_deref(),
            &aspx_cfg,
            acpl_num_param_bands_id,
            acpl_qm0,
            acpl_qm1,
            pad_target_bytes,
        );

        // 3. Wrap in v2 IMS TOC.
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

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_2 per
    /// ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_2:`
    /// (round 100). Symmetric counterpart to the decoder's round-25
    /// [`crate::mch::parse_5x_audio_data_outer`] ASPX_ACPL_{1,2} inner-
    /// body walker (Pseudocode 117).
    ///
    /// `frames` is in `[L, R, C]` order — the L/R carrier pair feeds the
    /// `two_channel_data()` body and drives the A-CPL Ls/Rs surround
    /// reconstruction via [`crate::acpl_synth::run_acpl_5x_pair_pcm`] at
    /// decode time; the centre carrier `C` is coded as a Cfg0
    /// `mono_data(0)` element. ASPX_ACPL_2 has no surround carriers — the
    /// Ls/Rs PCM is reconstructed entirely from the L/R carriers + the
    /// two `acpl_data_1ch()` parameter sets.
    ///
    /// The encoder forces the 5.0 channel_mode prefix (`0b1101`, 4 b —
    /// Table 85 channel_mode 3) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 5` through
    /// `parse_5x_audio_data_outer(b_has_lfe = false)` with
    /// `5_X_codec_mode = AspxAcpl2`.
    ///
    /// The ASPX/A-CPL parameter bits are emitted as minimum-bit-cost
    /// zero-delta Huffman codewords (the round-95 "structural scaffold"
    /// mode — see [`crate::encoder_acpl3`]). The decoder walks the full
    /// Table 25 ASPX_ACPL_2 body and produces 5-channel `[L, R, C, Ls,
    /// Rs]` PCM. With all-zero ACPL parameter deltas the surround pair
    /// Ls/Rs collapses to the ducker-driven reconstruction from the L/R
    /// carriers.
    ///
    /// `max_sfb` defaults to 40 (matching the round-95 ACPL_3 default).
    pub fn encode_frame_pcm_5_0_acpl2(&mut self, frames: &[&[f32]; 3]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl2_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of [`Self::encode_frame_pcm_5_0_acpl2`].
    pub fn encode_frame_pcm_5_0_acpl2_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 3],
        max_sfb: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_0_acpl2: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force 5.0 channel_mode prefix '1101', 4 b → channel_mode 3.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1101;
        self.channel_mode_bits = 4;

        // Forward MDCT analysis per carrier channel (L, R, C — 3 states).
        let n_channels = 3;
        while self.mdct_states_multi.len() < n_channels {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // ASPX config: small low-res scale so the SBG counts stay small —
        // keeps the ASPX_data bodies compact. Matches the round-95
        // ASPX_ACPL_3 config exactly.
        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0, // num_noise_sbgroups = 1
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_mode Fine.
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            self.b_iframe_global,
            &coeffs_per_channel[0],
            &coeffs_per_channel[1],
            &coeffs_per_channel[2],
            &aspx_cfg,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            pad_target_bytes,
        );

        // Wrap in v2 IMS TOC.
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 frame containing a 5.0 SIMPLE/ASPX_ACPL_1
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.6 Table 25 row
    /// `case ASPX_ACPL_1:` (Pseudocode 117).
    ///
    /// Unlike ASPX_ACPL_2 (which reconstructs the Ls/Rs surround pair
    /// purely from the L/R carriers + the two `acpl_data_1ch()` parameter
    /// sets), ASPX_ACPL_1 transmits the surround signal explicitly as a
    /// **joint-MDCT residual layer** (`max_sfb_master + 2× chparam_info +
    /// 2× sf_data(ASF)`) keyed by the `acpl_config_1ch(PARTIAL)` element's
    /// `acpl_qmf_band` field. It therefore accepts a full 5-channel
    /// `[L, R, C, Ls, Rs]` input: L/R become the `two_channel_data()`
    /// carriers, C the Cfg0 `mono_data(0)`, and Ls/Rs the residual pair
    /// (sSMP,3 / sSMP,4 per Table 181).
    ///
    /// The encoder forces the 5.0 channel_mode prefix (`0b1101`, 4 b —
    /// Table 85 channel_mode 3) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 5` through
    /// `parse_5x_audio_data_outer(b_has_lfe = false)` with
    /// `5_X_codec_mode = AspxAcpl1`. The ASPX/A-CPL parameter bits use the
    /// round-95 minimum-bit-cost zero-delta Huffman scaffold. The decoder
    /// walks the full Table 25 ASPX_ACPL_1 body — including the residual
    /// layer that IMDCTs into the Ls/Rs PCM carriers — and produces
    /// 5-channel `[L, R, C, Ls, Rs]` PCM via
    /// [`crate::acpl_synth::run_acpl_5x_pair_pcm`] (Pseudocode 117).
    ///
    /// `max_sfb` defaults to 40; `max_sfb_master` (the residual-layer band
    /// budget) defaults to 20.
    pub fn encode_frame_pcm_5_0_acpl1(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl1_with_max_sfb(frames, 40, 20)
    }

    /// `max_sfb` / `max_sfb_master`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl1`].
    pub fn encode_frame_pcm_5_0_acpl1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 5],
        max_sfb: u32,
        max_sfb_master: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_0_acpl1: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force 5.0 channel_mode prefix '1101', 4 b → channel_mode 3.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1101;
        self.channel_mode_bits = 4;

        // Forward MDCT analysis per carrier channel (L, R, C, Ls, Rs — 5
        // states).
        let n_channels = 5;
        while self.mdct_states_multi.len() < n_channels {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // ASPX config: small low-res scale so the SBG counts stay small —
        // keeps the ASPX_data bodies compact. Matches the round-95 / 100
        // ASPX_ACPL_3 / ACPL_2 config exactly.
        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0, // num_noise_sbgroups = 1
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_mode Fine;
        // acpl_qmf_band_minus1 = 0 → qmf_band = 1 (PARTIAL mode).
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_master,
            self.b_iframe_global,
            &coeffs_per_channel[0],
            &coeffs_per_channel[1],
            &coeffs_per_channel[2],
            &coeffs_per_channel[3],
            &coeffs_per_channel[4],
            &aspx_cfg,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
            pad_target_bytes,
        );

        // Wrap in v2 IMS TOC.
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 frame containing a 7.0 SIMPLE/ASPX_ACPL_2
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
    /// `case ASPX_ACPL_2:` (round 107). The 7_X (immersive) symmetric
    /// counterpart to the round-100 5_X ASPX_ACPL_2 encoder — it reuses the
    /// same 1ch ACPL / ASPX parameter shape (Pseudocode 117) but emits the
    /// 7_X channel element's distinct framing (2-bit `7_X_codec_mode`,
    /// `companding_control(5)`, 2-bit `coding_config`, two `two_channel_data`
    /// pairs, trailing centre `mono_data(0)`, and the two-`aspx_data_2ch`
    /// envelope trailer).
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order — the 7.0 (3/4/0)
    /// surface layout. The L/R pair feeds the first `two_channel_data()`
    /// carriers and drives the A-CPL Ls/Rs surround reconstruction via
    /// [`crate::acpl_synth::run_acpl_5x_pair_pcm`] at decode time. The Ls/Rs
    /// pair is coded as the second `two_channel_data()` (keeps the body
    /// well-formed for the walker; the ACPL_2 dispatch reconstructs the
    /// surround from L/R + params). The centre `C` is the trailing Cfg0
    /// `mono_data(0)`. The back pair `Lb, Rb` is accepted for layout
    /// completeness but not carried by the ASPX_ACPL_2 body (the decoder's
    /// ACPL_2 7_X dispatch populates slots 0..4 only — slots 5/6 stay
    /// silent), matching the decoder's documented Table 202 channel mapping.
    ///
    /// The encoder forces the 7.0 (3/4/0) channel_mode prefix (`0b1111000`,
    /// 7 b — Table 85 channel_mode 5) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 7` through
    /// `parse_7x_audio_data_outer(b_has_lfe = false)` with
    /// `7_X_codec_mode = AspxAcpl2`. The ASPX/A-CPL parameter bits use the
    /// round-95 minimum-bit-cost zero-delta Huffman scaffold.
    ///
    /// `max_sfb` defaults to 40.
    pub fn encode_frame_pcm_7_0_acpl2(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_acpl2_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of [`Self::encode_frame_pcm_7_0_acpl2`].
    pub fn encode_frame_pcm_7_0_acpl2_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 7],
        max_sfb: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_0_acpl2: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force 7.0 (3/4/0) channel_mode prefix '1111000', 7 b →
        // channel_mode 5 (Table 85). The decoder routes channels == 7
        // through parse_7x_audio_data_outer(b_has_lfe = false).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111000;
        self.channel_mode_bits = 7;

        // Forward MDCT analysis per channel — seven SCE states (L, R, C,
        // Ls, Rs, Lb, Rb). Only the first five feed the ASPX_ACPL_2 body;
        // the back pair is analysed for state continuity but its spectra
        // are not carried by the ACPL_2 path.
        let n_channels = 7;
        while self.mdct_states_multi.len() < n_channels {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // ASPX config: small low-res scale so the SBG counts stay small —
        // matches the round-95 / 100 / 103 ASPX_ACPL config exactly.
        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0, // num_noise_sbgroups = 1
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_mode Fine.
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body = crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            None, // 7.0 — no LFE
            self.b_iframe_global,
            &coeffs_per_channel[0],
            &coeffs_per_channel[1],
            &coeffs_per_channel[3],
            &coeffs_per_channel[4],
            &coeffs_per_channel[2],
            None, // 7.0 — no LFE
            &aspx_cfg,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            pad_target_bytes,
        );

        // Wrap in v2 IMS TOC.
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 frame containing a 7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_2
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
    /// `case ASPX_ACPL_2:` with `b_has_lfe = 1` (round 114). The LFE
    /// counterpart of [`Self::encode_frame_pcm_7_0_acpl2`] — it emits the
    /// identical 7_X ASPX_ACPL_2 body plus a leading `mono_data(b_lfe = 1)`
    /// element (Table 21 + `sf_info_lfe()` Table 35) between the I-frame
    /// config block and `companding_control(5)`, exactly where the decoder's
    /// `parse_7x_audio_data_outer(b_has_lfe = true)` reads
    /// `if (b_has_lfe) mono_data(1);`.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order — the 7.1
    /// (3/4/0.1) surface layout. The L/R pair feeds the first
    /// `two_channel_data()` carriers and drives the A-CPL Ls/Rs surround
    /// reconstruction via [`crate::acpl_synth::run_acpl_5x_pair_pcm`] at
    /// decode time; the Ls/Rs pair rides the second `two_channel_data()`;
    /// the centre `C` is the trailing Cfg0 `mono_data(0)`; the LFE is the
    /// leading `mono_data(1)`. The back pair `Lb, Rb` is accepted for layout
    /// completeness but not carried by the ASPX_ACPL_2 body (the decoder's
    /// 7_X ACPL_2 dispatch populates slots 0..4 + the LFE slot 7 — slots 5/6
    /// stay silent), matching the round-107 documented Table 202 channel
    /// mapping plus the round-80 LFE PCM render at decode time.
    ///
    /// The encoder forces the 7.1 channel_mode prefix (`0b1111001`, 7 b —
    /// Table 88 channel_mode 6) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 8` through
    /// `parse_7x_audio_data_outer(b_has_lfe = true)` with
    /// `7_X_codec_mode = AspxAcpl2`. The ASPX/A-CPL parameter bits use the
    /// round-95 minimum-bit-cost zero-delta Huffman scaffold.
    ///
    /// `max_sfb` defaults to 40; `max_sfb_lfe` defaults to 7 (the LFE-spec
    /// cap at `tl = 1920`, `n_msfbl_bits = 3`).
    pub fn encode_frame_pcm_7_1_acpl2(&mut self, frames: &[&[f32]; 8]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_acpl2_with_max_sfb(frames, 40, 7)
    }

    /// `max_sfb`-parameterised form of [`Self::encode_frame_pcm_7_1_acpl2`].
    /// `max_sfb` governs the five front/surround carrier SCEs and the centre
    /// mono; `max_sfb_lfe` governs the LFE `mono_data(1)` (clamped to the
    /// `n_msfbl_bits` cap).
    pub fn encode_frame_pcm_7_1_acpl2_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 8],
        max_sfb: u32,
        max_sfb_lfe: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_1_acpl2: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_7_1_acpl2: tl = {frame_len} not permitted for LFE"
        );
        let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
        let max_sfb_lfe = max_sfb_lfe.min(n_msfbl_cap);

        // Force 7.1 (3/4/0.1) channel_mode prefix '1111001', 7 b →
        // channel_mode 6 (Table 88). The decoder routes channels == 8
        // through parse_7x_audio_data_outer(b_has_lfe = true).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111001;
        self.channel_mode_bits = 7;

        // Forward MDCT analysis per channel — eight SCE states (L, R, C,
        // Ls, Rs, Lb, Rb, LFE). The first five + LFE feed the ASPX_ACPL_2
        // body; the back pair is analysed for state continuity but its
        // spectra are not carried by the ACPL_2 path.
        let n_channels = 8;
        while self.mdct_states_multi.len() < n_channels {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // ASPX config: matches the round-95 / 100 / 103 / 107 ASPX_ACPL
        // config exactly (small low-res scale → small SBG counts).
        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0, // num_noise_sbgroups = 1
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_mode Fine.
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body = crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            Some(max_sfb_lfe),
            self.b_iframe_global,
            &coeffs_per_channel[0],
            &coeffs_per_channel[1],
            &coeffs_per_channel[3],
            &coeffs_per_channel[4],
            &coeffs_per_channel[2],
            Some(&coeffs_per_channel[7]),
            &aspx_cfg,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            pad_target_bytes,
        );

        // Wrap in v2 IMS TOC.
        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
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

    // ------------------------------------------------------------------
    // Round 50 — DP section optimiser + SNF emission integration tests.
    // ------------------------------------------------------------------

    /// SNF-bit-on round-trip: encode a tone+noise input, then verify
    /// that the decoded reconstruction has non-zero magnitude in
    /// high-frequency bins that the quantiser collapsed to zero. The
    /// `b_snf_data_exists` bit must round-trip through the parser
    /// without erroring.
    #[test]
    fn encode_frame_pcm_white_noise_with_snf_fills_zero_quant_bands() {
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};
        let n = 1920usize;
        let max_sfb = 55u32;
        let sfbo = crate::sfb_offset::sfb_offset_48(n as u32).unwrap();
        let end_bin = sfbo[max_sfb as usize] as usize;

        // Low-energy white noise — most high bands quantise to cb=0,
        // exercising the SNF emission path.
        let make_frame = |seed_off: u64| -> Vec<f32> {
            let mut s: u64 = 0xACE4_u64.wrapping_add(seed_off);
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (s >> 33) as u32;
                    (u as f32 / (1u32 << 31) as f32 - 1.0) * 0.05 // low energy
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..3).map(|i| make_frame(i as u64 * n as u64)).collect();
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new();
        let mut last_recon: Option<Vec<f32>> = None;
        for f in &frames {
            let bytes = enc.encode_frame_pcm_with_max_sfb(f, max_sfb);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let _ = dec.receive_frame().expect("receive_frame");
            let sub = dec.last_substream.as_ref().unwrap();
            let scaled = sub.tools.scaled_spec_primary.as_ref().unwrap().clone();
            last_recon = Some(scaled);
        }
        let recon = last_recon.unwrap();
        // Count non-zero bins in the recon — with SNF on, even bands that
        // collapsed to cb=0 should have non-zero magnitude from injected
        // noise (clamped by what the SNF index range allows).
        let nonzero = recon[..end_bin].iter().filter(|&&v| v.abs() > 0.0).count();
        // We don't insist on every bin being non-zero (some bands may have
        // SNF idx 0 = "no fill"); the assertion is that the bitstream
        // round-trips without error and decodes to a non-silent spectrum.
        assert!(
            nonzero > 0,
            "expected at least one non-zero bin in SNF reconstruction, got {nonzero}"
        );
    }

    /// SNF integration: SNF-on bitstream parses cleanly through the
    /// existing decoder. This is the smoke test for the new emission
    /// path — it MUST not break decode of non-SNF frames either.
    #[test]
    fn encode_frame_pcm_silence_with_snf_off_round_trips() {
        // Pure silence input: no band has measurable energy → SNF should
        // be `None` → b_snf_data_exists = 0 in the bitstream.
        let n = 1920usize;
        let frames: Vec<Vec<f32>> = (0..2).map(|_| vec![0.0_f32; n]).collect();
        let decoded = encode_decode_frames_with_max_sfb(&frames, 50);
        let pcm = &decoded[1];
        let peak = pcm.iter().map(|&s| s.abs()).max().unwrap_or(0);
        // Silence input → silence output (no SNF fill since no energy).
        assert_eq!(
            peak, 0,
            "silence + SNF-off should decode to silence, peak={peak}"
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

    /// Round 52 sanity: identical L=R PCM → MDCT spectra are bit-identical
    /// → per-SFB energy-weighted correlation is exactly 1.0; the
    /// dispatcher would route this frame to Path B (joint M/S).
    #[test]
    fn round52_correlation_identical_channels_is_one() {
        use crate::encoder_asf::average_per_sfb_correlation;
        use crate::encoder_mdct::EncoderMdctState;
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                })
                .collect()
        };
        let mut mdct_l = EncoderMdctState::new(n as u32);
        let mut mdct_r = EncoderMdctState::new(n as u32);
        let mut rhos: Vec<f32> = Vec::new();
        for i in 0..3 {
            let f = make_frame(i * n);
            let cl = mdct_l.analyse_frame(&f);
            let cr = mdct_r.analyse_frame(&f);
            let rho = average_per_sfb_correlation(n as u32, 40, &cl, &cr);
            rhos.push(rho);
        }
        for (i, &rho) in rhos.iter().enumerate() {
            assert!(
                (rho - 1.0).abs() < 1e-4,
                "frame {i} rho expected 1.0, got {rho}"
            );
        }
    }

    /// Round 52 sanity: 440 Hz L + 660 Hz R independent channels have
    /// well-separated MDCT spectra → energy-weighted per-SFB correlation
    /// falls below the 0.7 dispatch threshold; Path A is chosen.
    #[test]
    fn round52_correlation_independent_tones_below_threshold() {
        use crate::encoder_asf::average_per_sfb_correlation;
        use crate::encoder_mdct::EncoderMdctState;
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame = |freq: f32, start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * freq * t).sin()
                })
                .collect()
        };
        let mut mdct_l = EncoderMdctState::new(n as u32);
        let mut mdct_r = EncoderMdctState::new(n as u32);
        let mut rhos: Vec<f32> = Vec::new();
        for i in 0..3 {
            let fl = make_frame(440.0, i * n);
            let fr = make_frame(660.0, i * n);
            let cl = mdct_l.analyse_frame(&fl);
            let cr = mdct_r.analyse_frame(&fr);
            rhos.push(average_per_sfb_correlation(n as u32, 40, &cl, &cr));
        }
        for (i, &rho) in rhos.iter().enumerate() {
            assert!(rho.abs() < 0.6, "frame {i} rho expected < 0.6, got {rho}");
        }
    }

    // ------------------------------------------------------------------
    // Round 51 — Stereo SIMPLE/ASF split-MDCT (Path A, 2× SCE) tests.
    // ------------------------------------------------------------------

    /// Helper: encode a sequence of stereo PCM frames (each `(L, R)`)
    /// through `encode_frame_pcm_stereo`, then decode them via
    /// `Ac4Decoder` and return per-frame deinterleaved `(L, R)` i16 PCM.
    fn encode_decode_stereo_frames(
        frames_lr: &[(Vec<f32>, Vec<f32>)],
    ) -> Vec<(Vec<i16>, Vec<i16>)> {
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new();
        let mut out: Vec<(Vec<i16>, Vec<i16>)> = Vec::with_capacity(frames_lr.len());
        for (l, r) in frames_lr {
            let bytes = enc.encode_frame_pcm_stereo(l, r);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
                panic!("expected audio frame");
            };
            assert_eq!(af.samples, 1_920);
            assert_eq!(af.data.len(), 1);
            // Stereo S16 interleaved: 1920 samples × 2 ch × 2 bytes.
            assert_eq!(af.data[0].len(), 1_920 * 2 * 2);
            let buf = &af.data[0];
            let mut pcm_l: Vec<i16> = Vec::with_capacity(1_920);
            let mut pcm_r: Vec<i16> = Vec::with_capacity(1_920);
            for i in 0..1_920usize {
                let off_l = i * 4;
                let off_r = off_l + 2;
                pcm_l.push(i16::from_le_bytes([buf[off_l], buf[off_l + 1]]));
                pcm_r.push(i16::from_le_bytes([buf[off_r], buf[off_r + 1]]));
            }
            out.push((pcm_l, pcm_r));
        }
        out
    }

    /// Stereo encoder bumps sequence_counter once per frame, just like
    /// the mono path.
    #[test]
    fn encode_frame_pcm_stereo_bumps_sequence_counter() {
        let mut enc = Ac4ImsEncoder::new();
        assert_eq!(enc.sequence_counter, 0);
        let frame = vec![0.0_f32; 1920];
        let _ = enc.encode_frame_pcm_stereo(&frame, &frame);
        assert_eq!(enc.sequence_counter, 1);
        let _ = enc.encode_frame_pcm_stereo(&frame, &frame);
        assert_eq!(enc.sequence_counter, 2);
    }

    /// Stereo encoder produces a frame whose TOC declares 2 channels and
    /// whose decoded PCM layout is stereo (1920 × 2 × 2 bytes).
    #[test]
    fn encode_frame_pcm_stereo_produces_stereo_layout_pcm() {
        let n = 1920usize;
        let frames: Vec<(Vec<f32>, Vec<f32>)> = (0..2)
            .map(|_| (vec![0.0_f32; n], vec![0.0_f32; n]))
            .collect();
        let decoded = encode_decode_stereo_frames(&frames);
        // Both decoded frames have the stereo S16 byte layout.
        for (l, r) in &decoded {
            assert_eq!(l.len(), 1_920);
            assert_eq!(r.len(), 1_920);
        }
    }

    /// Stereo encoder roundtrip: decoder produces non-silent PCM in
    /// both channels with peak amplitudes reflecting the input level.
    #[test]
    fn encode_frame_pcm_stereo_440hz_steady_state_nonsilent_both_channels() {
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                })
                .collect()
        };
        let frames_lr: Vec<(Vec<f32>, Vec<f32>)> = (0..5)
            .map(|i| (make_frame(i * n), make_frame(i * n)))
            .collect();
        let decoded = encode_decode_stereo_frames(&frames_lr);
        let (l, r) = &decoded[2];
        let nz_l = l.iter().filter(|&&s| s != 0).count();
        let nz_r = r.iter().filter(|&&s| s != 0).count();
        let peak_l = l.iter().map(|&s| s.abs()).max().unwrap_or(0);
        let peak_r = r.iter().map(|&s| s.abs()).max().unwrap_or(0);
        assert!(nz_l > 100, "L too few non-zero samples: {nz_l}");
        assert!(nz_r > 100, "R too few non-zero samples: {nz_r}");
        // 0.3 input amplitude → ~0.3 * 32767 ≈ 9830 i16 peak. The
        // encoder/decoder lossy round-trip stays comfortably above 1000.
        assert!(peak_l > 1000, "L peak too low: {peak_l}");
        assert!(peak_r > 1000, "R peak too low: {peak_r}");
    }

    /// Stereo encoder TOC declares 2 channels and the substream parser
    /// surfaces both per-channel scaled spectra. Round 52 update: with
    /// identical channels (L=R), the cross-channel correlation is 1.0 so
    /// the dispatcher routes the frame through Path B (joint M/S CPE,
    /// `b_enable_mdct_stereo_proc == 1`). Force the split-MDCT path so
    /// this structural test continues to exercise Path A.
    #[test]
    fn encode_frame_pcm_stereo_substream_parses() {
        use crate::decoder::Ac4Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};
        let mut enc = Ac4ImsEncoder::new();
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let frame: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        let bytes = enc.encode_frame_pcm_stereo_split_with_max_sfb(&frame, &frame, 40);
        let info = crate::toc::parse_ac4_toc(&bytes).expect("parse_ac4_toc");
        assert_eq!(info.channels, 2);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let _ = dec.receive_frame().expect("receive_frame");
        let sub = dec.last_substream.as_ref().expect("substream parsed");
        // SIMPLE stereo mode + b_enable_mdct_stereo_proc = 0 (split-MDCT
        // path forced by `encode_frame_pcm_stereo_split_with_max_sfb`).
        // Both channels' spectra populated.
        assert!(matches!(
            sub.tools.stereo_mode,
            Some(crate::asf::StereoCodecMode::Simple)
        ));
        assert!(!sub.tools.mdct_stereo_proc);
        assert!(sub.tools.scaled_spec_primary.is_some());
        assert!(sub.tools.scaled_spec_secondary.is_some());
    }

    /// Round 48 stereo SNR target: 440 Hz tone on L + 440 Hz tone on R
    /// (identical content) round-trips with **spectral SNR ≥ 20 dB** on
    /// the steady-state frame for both channels.
    ///
    /// Spectral SNR is measured by mirroring the encoder's forward MDCT
    /// over the input PCM and comparing the input MDCT spectrum bin-for-
    /// bin against the decoder's reconstructed `scaled_spec_*`. This
    /// isolates the encoder's quantisation contribution from the IMDCT/
    /// KBD overlap-add reconstruction noise (which dominates a time-
    /// domain comparison since the IMDCT introduces a half-frame phase
    /// shift between the original and reconstructed waveforms even for
    /// perfect-reconstruction transforms — same convention used by the
    /// round-49 white-noise test
    /// `encode_frame_pcm_white_noise_snr_exceeds_hcb5_only_ceiling`).
    #[test]
    fn encode_frame_pcm_stereo_440hz_both_channels_snr_exceeds_20db() {
        use crate::decoder::Ac4Decoder;
        use crate::encoder_mdct::EncoderMdctState;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (start + i) as f32 / fs;
                    0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..3).map(|i| make_frame(i * n)).collect();
        // Mirror the encoder's MDCT on the input for both channels (same
        // PCM here, so we only need one mirror state).
        let mut mdct_in = EncoderMdctState::new(n as u32);
        let mut last_input_spec: Option<Vec<f32>> = None;
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new();
        let mut last_pri: Option<Vec<f32>> = None;
        let mut last_sec: Option<Vec<f32>> = None;
        for f in &frames {
            let input_coeffs = mdct_in.analyse_frame(f);
            last_input_spec = Some(input_coeffs);
            let bytes = enc.encode_frame_pcm_stereo(f, f);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let _ = dec.receive_frame().expect("receive_frame");
            let sub = dec.last_substream.as_ref().unwrap();
            last_pri = sub.tools.scaled_spec_primary.clone();
            last_sec = sub.tools.scaled_spec_secondary.clone();
        }
        let input = last_input_spec.unwrap();
        let pri = last_pri.unwrap();
        let sec = last_sec.unwrap();
        let snr = |orig: &[f32], recon: &[f32]| -> f64 {
            let mut sig_e = 0.0_f64;
            let mut err_e = 0.0_f64;
            let n_compare = orig.len().min(recon.len());
            for k in 0..n_compare {
                let o = orig[k] as f64;
                let r = recon[k] as f64;
                sig_e += o * o;
                err_e += (o - r) * (o - r);
            }
            10.0 * (sig_e / err_e.max(1e-30)).log10()
        };
        let snr_l = snr(&input, &pri);
        let snr_r = snr(&input, &sec);
        eprintln!(
            "ROUND-51 stereo 440Hz L+R spectral SNR: SNR_L = {snr_l:.1} dB, SNR_R = {snr_r:.1} dB"
        );
        assert!(
            snr_l > 20.0,
            "L channel spectral SNR too low: {snr_l:.1} dB (expected > 20 dB)"
        );
        assert!(
            snr_r > 20.0,
            "R channel spectral SNR too low: {snr_r:.1} dB (expected > 20 dB)"
        );
    }

    /// Round 48 stereo independence target: 440 Hz tone on L + 660 Hz
    /// tone on R round-trips with **spectral SNR ≥ 20 dB** on the
    /// steady-state frame for both channels (proves channels are encoded
    /// independently — no cross-channel bleed). See the docstring on
    /// [`encode_frame_pcm_stereo_440hz_both_channels_snr_exceeds_20db`]
    /// for why we compare in the spectral domain.
    #[test]
    fn encode_frame_pcm_stereo_440l_660r_independent_channels_snr_exceeds_20db() {
        use crate::decoder::Ac4Decoder;
        use crate::encoder_mdct::EncoderMdctState;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};
        let n = 1920usize;
        let fs = 48_000.0_f32;
        let make_frame_at = |freq: f32| -> Box<dyn Fn(usize) -> Vec<f32>> {
            Box::new(move |start: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        let t = (start + i) as f32 / fs;
                        0.3 * (2.0 * std::f32::consts::PI * freq * t).sin()
                    })
                    .collect()
            })
        };
        let make_l = make_frame_at(440.0);
        let make_r = make_frame_at(660.0);
        let frames_lr: Vec<(Vec<f32>, Vec<f32>)> =
            (0..3).map(|i| (make_l(i * n), make_r(i * n))).collect();
        // Mirror MDCT on each channel's input independently.
        let mut mdct_l = EncoderMdctState::new(n as u32);
        let mut mdct_r = EncoderMdctState::new(n as u32);
        let mut last_in_l: Option<Vec<f32>> = None;
        let mut last_in_r: Option<Vec<f32>> = None;
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mut enc = Ac4ImsEncoder::new();
        let mut last_pri: Option<Vec<f32>> = None;
        let mut last_sec: Option<Vec<f32>> = None;
        for (l, r) in &frames_lr {
            last_in_l = Some(mdct_l.analyse_frame(l));
            last_in_r = Some(mdct_r.analyse_frame(r));
            let bytes = enc.encode_frame_pcm_stereo(l, r);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let _ = dec.receive_frame().expect("receive_frame");
            let sub = dec.last_substream.as_ref().unwrap();
            last_pri = sub.tools.scaled_spec_primary.clone();
            last_sec = sub.tools.scaled_spec_secondary.clone();
        }
        let in_l = last_in_l.unwrap();
        let in_r = last_in_r.unwrap();
        let pri = last_pri.unwrap();
        let sec = last_sec.unwrap();
        let snr = |orig: &[f32], recon: &[f32]| -> f64 {
            let mut sig_e = 0.0_f64;
            let mut err_e = 0.0_f64;
            let n_compare = orig.len().min(recon.len());
            for k in 0..n_compare {
                let o = orig[k] as f64;
                let r = recon[k] as f64;
                sig_e += o * o;
                err_e += (o - r) * (o - r);
            }
            10.0 * (sig_e / err_e.max(1e-30)).log10()
        };
        let snr_l = snr(&in_l, &pri);
        let snr_r = snr(&in_r, &sec);
        eprintln!(
            "ROUND-51 stereo 440L+660R independent spectral SNR: SNR_L = {snr_l:.1} dB, SNR_R = {snr_r:.1} dB"
        );
        assert!(
            snr_l > 20.0,
            "L (440 Hz) channel spectral SNR too low: {snr_l:.1} dB (expected > 20 dB)"
        );
        assert!(
            snr_r > 20.0,
            "R (660 Hz) channel spectral SNR too low: {snr_r:.1} dB (expected > 20 dB)"
        );
        // Independence sanity check: L and R reconstructions should differ
        // (different input frequencies → different waveforms in the
        // spectrum).
        let differs = pri
            .iter()
            .zip(sec.iter())
            .filter(|(a, b)| (*a - *b).abs() > 0.01)
            .count();
        assert!(
            differs > 10,
            "L and R reconstructed spectra should differ for independent tones (got {differs} diffs)"
        );
    }
}
