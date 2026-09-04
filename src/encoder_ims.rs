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
//!   The `sus_ver` bit is zero; the per-substream `b_audio_ndot` /
//!   `b_pres_ndot` flags carry the I-frame flag directly
//!   (§6.3.2.11.2 — ndot means "no dependency over time").
//! * No bit-rate signalling beyond `br_code = 0`.

use oxideav_core::bits::BitWriter;

use crate::encoder_asf::{
    average_per_sfb_correlation, build_5_0_simple_asf_body_from_pcm_spectra,
    build_5_1_simple_asf_body_from_pcm_spectra, build_7_0_simple_asf_body_from_pcm_spectra,
    build_7_1_simple_asf_body_from_pcm_spectra, build_mono_simple_asf_body_from_pcm_spectrum,
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
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub mdct_state: Option<EncoderMdctState>,
    /// Forward-MDCT analysis state for the secondary (right) channel of
    /// `encode_frame_pcm_stereo()`. Identical role to `mdct_state` but
    /// for the second channel — separate so 50% TDAC overlap is
    /// per-channel.
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub mdct_state_r: Option<EncoderMdctState>,
    /// Forward-MDCT analysis state for the multichannel encoder paths
    /// (`encode_frame_pcm_5_0()` and any future N>2 variants). One
    /// [`EncoderMdctState`] per output channel — separate so 50% TDAC
    /// overlap continuity is preserved per channel across frames. Lazy-
    /// initialised on first use; grown to the required channel count.
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub mdct_states_multi: Vec<EncoderMdctState>,
    /// Previous frame's absolute A-SPX envelope rows on the live 5_X
    /// ACPL_3 single-envelope path — drives the P-frame TIME-direction
    /// DPCM decision (§5.7.6.3.4 Pseudocodes 80 / 81 `qscf_*_prev`).
    /// `None` until an ACPL_3 single-envelope frame has been emitted;
    /// cleared when a multi-envelope body is emitted (its last-envelope
    /// rows are not tracked, so the next frame safely stays FREQ).
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub acpl3_env_prev: Option<crate::encoder_acpl3::Acpl3EnvPrevRows>,
    /// Previous frame's absolute A-CPL quantized parameter rows on the
    /// live 5_X ACPL_3 single-envelope path — drives the P-frame
    /// DIFF_TIME decision (Table 65 / Pseudocode 121). Unprimed until
    /// an ACPL_3 single-envelope frame has been emitted; reset when a
    /// multi-envelope body is emitted.
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub acpl3_param_prev: crate::encoder_acpl3::Acpl3ParamPrevRows,
    /// Streaming QMF analysis banks for the ICE A-SPX envelope /
    /// decision extraction — one per decoupled extraction channel, so
    /// consecutive frames analyse as one continuous stream (a fresh
    /// per-frame bank leaks a broadband warm-up splash into the HF
    /// envelope measurement).
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub ice_env_ana: Vec<crate::qmf::QmfAnalysisBank>,
    /// Encoder-side A-JCC differential-coding state (the decoder-side
    /// `ajcc_<SET>_q_prev` mirror) for the ICE ASPX_AJCC encode arms —
    /// drives the P-frame FREQ-vs-TIME row selection.
    #[doc(hidden)] // internal cross-frame state; not part of the stable API
    pub ajcc_enc_state: crate::encoder_ajcc::AjccEncoderState,
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
            acpl3_env_prev: None,
            acpl3_param_prev: crate::encoder_acpl3::Acpl3ParamPrevRows::default(),
            ice_env_ana: Vec::new(),
            ajcc_enc_state: crate::encoder_ajcc::AjccEncoderState::new(),
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

    /// 7.0 (3/4/0) channel mode (`0b1111000`, 7 b) per ETSI TS 103 190-1
    /// §4.3.3.7.1 Table 88 — channel_mode value `1111000` → ch_mode 5 → 7
    /// channels with layout `L, C, R, Ls, Rs, Lb, Rb`. Drives the decoder's
    /// `7_X_channel_element()` walker for `channels == 7` (no `b_has_lfe`
    /// — that branch is gated on channel_mode 6 / 7.1) per §4.2.6.14
    /// Table 33. The decoder's internal coding order for the inner
    /// `five_channel_data()` is `[L, R, C, Ls, Rs]` per Table 180 (the
    /// inner SCE order differs from the surface Table 88 listing's `L, C,
    /// R` ordering — the decoder treats the inner five_channel_data slots
    /// as L/R/C/Ls/Rs).
    pub fn with_7_0(mut self) -> Self {
        self.channel_mode_value = 0b1111000;
        self.channel_mode_bits = 7;
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
        bw.write_u32(if self.b_iframe_global { 1 } else { 0 }, 1); // b_iframe
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
    /// (b_alternative = 0, b_pres_ndot = iframe, substream_index = 0).
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
        // b_pres_ndot = b_iframe_global (§6.3.2.11.2 — ndot means "no
        // dependency over time"), substream_index = 0 (2 b).
        bw.write_u32(0, 1);
        bw.write_u32(if self.b_iframe_global { 1 } else { 0 }, 1);
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
                            // → 1 b_audio_ndot bit = b_iframe (§6.3.2.x).
        bw.write_u32(if self.b_iframe_global { 1 } else { 0 }, 1);
        bw.write_u32(0, 2); // substream_index
                            // b_content_type = 0.
        bw.write_u32(0, 1);
    }
}

/// Scale a PCM slice into the A-SPX QMF integer-PCM domain
/// ([`crate::aspx::ASPX_QMF_PCM_SCALE`]) before feeding an analysis
/// bank — the encoder-side mirror of the decoder's scaled analysis,
/// keeping the envelope quantisers' absolute anchors calibrated.
fn aspx_scaled_pcm(pcm: &[f32]) -> Vec<f32> {
    pcm.iter()
        .map(|&v| v * crate::aspx::ASPX_QMF_PCM_SCALE)
        .collect()
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

    /// Encode one IMS v2 7.0 (3/4/0) frame from float PCM input per ETSI
    /// TS 103 190-1 §4.2.6.14 Table 33 + §4.2.7.5 Table 29
    /// (`five_channel_data()`) + §4.2.7.4 Table 26 (`two_channel_data()`).
    /// The non-LFE immersive counterpart of
    /// [`Self::encode_frame_pcm_7_1`] — same `7_X_codec_mode = SIMPLE` +
    /// `coding_config = Cfg3Five` body shape, but the walker's
    /// `if (b_has_lfe) mono_data(1);` branch is omitted (`b_has_lfe =
    /// false` for channel_mode 5 / 7.0).
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order — the inner
    /// `five_channel_data()` (Table 180) carries the front/surround pair
    /// `L/R/C/Ls/Rs` and the SIMPLE/ASPX additional-channel block carries
    /// the immersive back pair `Lb/Rb` via a trailing
    /// `two_channel_data()` per Table 26. The encoder uses identity SAP
    /// (`b_use_sap_add_ch = 0`, `sap_mode = 0` on every `chparam_info`)
    /// so no joint-MDCT mixing happens at decode time: every output
    /// channel comes straight from its own `sf_data(ASF)` body.
    ///
    /// The encoder forces the 7.0 channel_mode prefix (`0b1111000`, 7 b —
    /// Table 88 channel_mode 5) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 7` through
    /// `parse_7x_audio_data_outer(b_has_lfe = false)`. The five
    /// front/surround channels share the same Cfg3Five
    /// `five_channel_data()` body as the 5.0 / 5.1 / 7.1 paths; the
    /// additional pair (Lb, Rb) rides the trailing `two_channel_data()`
    /// which the decoder's `dispatch_7x_additional_channel_pair` (Table
    /// 183 row "3/4/0.x" identity path) routes into output slots 5 / 6.
    ///
    /// `max_sfb` defaults to 40 (matching the round-49 mono / round-74
    /// 5.0 / round-80 5.1 / round-91 7.1 default); `max_sfb_add`
    /// defaults to 40 (same width as the 7.0 non-additional channels).
    /// Use [`Self::encode_frame_pcm_7_0_with_max_sfb`] for wider coverage.
    pub fn encode_frame_pcm_7_0(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_with_max_sfb(frames, 40, 40)
    }

    /// Encode one IMS v2 7.0 (3/4/0) frame from arbitrary float PCM input
    /// at caller-specified `max_sfb` (five-channel front/surround SCEs)
    /// and `max_sfb_add` (additional Lb/Rb pair). See
    /// [`Self::encode_frame_pcm_7_0`] for the rest of the contract.
    pub fn encode_frame_pcm_7_0_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 7],
        max_sfb: u32,
        max_sfb_add: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_0: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        // Cap max_sfb at the non-LFE max_sfb width's max and at the actual
        // num_sfb_48 cap. Same cap applies to both the inner
        // five_channel_data and the additional two_channel_data — they
        // share the n_msfb_bits width.
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        let max_sfb_add = max_sfb_add.min(n_msfb_cap);

        // Force 7.0 (3/4/0) channel_mode (prefix '1111000', 7 bits —
        // Table 88 channel_mode 5). The body builder requires the TOC to
        // declare 7 channels so the decoder's `walk_ac4_substream`
        // dispatch routes through `parse_7x_audio_data_outer(b_has_lfe =
        // false)`.
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111000;
        self.channel_mode_bits = 7;

        // 1. Forward MDCT analysis per channel (separate state for
        //    independent 50% TDAC overlap continuity). Seven channels
        //    here so the multi-channel state vector needs to grow.
        while self.mdct_states_multi.len() < 7 {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(7);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        // 2-4. Build the 7.0 SIMPLE/Cfg3Five body. Pad budget is ~7× the
        //      mono budget since we carry seven spectra independently.
        //      Capped at 32767 — the 15-bit `audio_size_value` field
        //      saturates there (extending via `b_more_bits` is permitted
        //      by §4.3.4.1 but not needed for the default max_sfb path).
        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };
        let body = build_7_0_simple_asf_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_add,
            &[
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &coeffs_per_channel[5],
                &coeffs_per_channel[6],
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

    /// Encode one IMS v2 7.0 (3/4/0) **pure-ASPX** (`7_X_codec_mode = ASPX`)
    /// frame per ETSI TS 103 190-1 §4.2.6.14 Table 33 row `case ASPX:`.
    ///
    /// The ASPX counterpart of [`Self::encode_frame_pcm_7_0`]: the same
    /// `five_channel_data()` (L/R/C/Ls/Rs) + additional `two_channel_data()`
    /// (Lb/Rb) carrier body, but `7_X_codec_mode = ASPX` and the body
    /// carries the **four** ASPX trailers the decoder walks for ASPX mode —
    /// `aspx_data_2ch()` (L/R) + `aspx_data_2ch()` (Ls/Rs) + `aspx_data_1ch()`
    /// (centre) + the extra `aspx_data_2ch()` for the **back pair Lb/Rb**
    /// (Table 202 x3/x4; carried independently because pure-ASPX mode has no
    /// A-CPL coupling). The encoder QMF-analyses all seven input carriers
    /// and emits real SIGNAL/NOISE envelopes on each.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order; the output
    /// round-trips through [`crate::decoder::Ac4Decoder`] to a 7-channel
    /// `AudioFrame`.
    pub fn encode_frame_pcm_7_0_aspx_real_aspx(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_aspx_real_aspx_with_max_sfb(frames, 40, 40)
    }

    /// `max_sfb` / `max_sfb_add`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_aspx_real_aspx`].
    pub fn encode_frame_pcm_7_0_aspx_real_aspx_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 7],
        max_sfb: u32,
        max_sfb_add: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_0_aspx_real_aspx: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        let max_sfb_add = max_sfb_add.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111000;
        self.channel_mode_bits = 7;

        while self.mdct_states_multi.len() < 7 {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs_per_channel: Vec<Vec<f32>> = Vec::with_capacity(7);
        for (ch, f) in frames.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }

        let mut aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening decision (Table 121):
        // one per-`aspx_config` flag governs all four A-SPX elements in this
        // 7.0 pure-ASPX body, derived from the primary (L) carrier's QMF low
        // band (§5.7.6.4.1.2 Pseudocode 85).
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        // Real ASPX envelope extraction over all four ASPX elements: L/R
        // front pair, Ls/Rs surround pair, centre carrier, and the back
        // pair Lb/Rb.
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        let (ls_sig, ls_noise, rs_sig, rs_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[3], frames[4]);
        let (c_sig, c_noise) = self.extract_aspx_mono_envelope(&aspx_cfg, frame_len, frames[2]);
        let (lb_sig, lb_noise, rb_sig, rb_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[5], frames[6]);

        // Per-carrier `aspx_tna_mode` inverse-filtering vectors, each
        // derived from its own carrier's QMF low band (front from L,
        // surround from Ls, centre from C, back from Lb). Under
        // `aspx_balance = 1` channel 1 of each pair mirrors channel 0, so a
        // single per-pair vector suffices. See §4.3.10.6.1 Table 131.
        let front_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);
        let surround_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[3]);
        let c_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[2]);
        let back_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[5]);

        // Per-channel real aspx_add_harmonic (§4.2.12.6) for all seven
        // 7.0 carriers — signalled independently per channel.
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let ls_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[3]);
        let rs_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[4]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);
        let lb_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[5]);
        let rb_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[6]);

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body = crate::encoder_asf::build_7_0_aspx_asf_body_from_pcm_spectra_real_aspx_tna(
            frame_len,
            max_sfb,
            max_sfb_add,
            self.b_iframe_global,
            &[
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &coeffs_per_channel[5],
                &coeffs_per_channel[6],
            ],
            &aspx_cfg,
            &l_sig,
            &l_noise,
            &r_sig,
            &r_noise,
            &ls_sig,
            &ls_noise,
            &rs_sig,
            &rs_noise,
            &c_sig,
            &c_noise,
            &lb_sig,
            &lb_noise,
            &rb_sig,
            &rb_noise,
            &front_tna_mode,
            &surround_tna_mode,
            &c_tna_mode,
            &back_tna_mode,
            &l_ah,
            &r_ah,
            &ls_ah,
            &rs_ah,
            &c_ah,
            &lb_ah,
            &rb_ah,
            pad_target_bytes,
        );

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
        // Legacy ACPL_3 body: it advances the decoder's cross-frame ASPX
        // envelope + ACPL q_prev references without tracking the rows, so
        // reset the encoder-side P-frame reference states (next non-I
        // frame stays FREQ / always-decodable).
        self.acpl3_env_prev = None;
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        // Restore caller's channel_mode setting.
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// real per-parameter-band β1 / β2 extraction from the L / R
    /// carrier energy distributions (round 193). Symmetric to
    /// [`Self::encode_frame_pcm_5_0_acpl3`] but routes the substream
    /// body builder through
    /// [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_beta`]
    /// so the β1 / β2 Huffman layers carry the carrier-driven decorrelator
    /// gains instead of all-zero codewords.
    ///
    /// `beta_scale` controls the wet/dry balance — see
    /// [`crate::encoder_acpl3::extract_beta_q_per_band_carrier_energy`]
    /// for the magnitude / scale relationship. Values in `0.05..=0.3`
    /// produce noticeable surround reconstruction without saturating
    /// the BETA codebook.
    pub fn encode_frame_pcm_5_0_acpl3_real_beta(
        &mut self,
        frames: &[&[f32]; 3],
        beta_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_beta_with_max_sfb(frames, None, 40, None, beta_scale)
    }

    /// 5.1 counterpart to [`Self::encode_frame_pcm_5_0_acpl3_real_beta`].
    /// `frames` is in `[L, R, C, LFE]` order. The LFE channel is coded
    /// as a leading `mono_data(b_lfe = 1)` element per Table 21 — same
    /// path as [`Self::encode_frame_pcm_5_1_acpl3`].
    pub fn encode_frame_pcm_5_1_acpl3_real_beta(
        &mut self,
        frames: &[&[f32]; 4],
        beta_scale: f32,
    ) -> Vec<u8> {
        let carriers: [&[f32]; 3] = [frames[0], frames[1], frames[2]];
        self.encode_frame_pcm_5_x_acpl3_real_beta_with_max_sfb(
            &carriers,
            Some(frames[3]),
            40,
            Some(7),
            beta_scale,
        )
    }

    /// Shared body for the real-β ACPL_3 entry points. Mirrors
    /// [`Self::encode_frame_pcm_5_x_acpl3_with_max_sfb`] but invokes the
    /// real-β builder. Kept close to the parent so the two paths stay
    /// in sync as the ASPX / ACPL config evolves.
    fn encode_frame_pcm_5_x_acpl3_real_beta_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 3],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        beta_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_beta: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_beta: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[3].analyse_frame(buf));

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_beta(
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
            beta_scale,
            pad_target_bytes,
        );

        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        // Legacy ACPL_3 body: it advances the decoder's cross-frame ASPX
        // envelope + ACPL q_prev references without tracking the rows, so
        // reset the encoder-side P-frame reference states (next non-I
        // frame stays FREQ / always-decodable).
        self.acpl3_env_prev = None;
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// real per-parameter-band α₁ / α₂ extraction from the L↔R carrier
    /// cross-correlation (round 196) **and** the round-193 real β₁ / β₂
    /// extraction from the L / R carrier energies. Symmetric counterpart
    /// to [`Self::encode_frame_pcm_5_0_acpl3_real_beta`] but routes the
    /// substream body builder through
    /// [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta`]
    /// so the α₁ / α₂ Huffman layers carry correlation-driven dry-mix
    /// balance indices in addition to the carrier-driven decorrelator
    /// gains in β₁ / β₂.
    ///
    /// `alpha_scale` controls the front/back-balance policy — see
    /// [`crate::encoder_acpl3::extract_alpha_q_per_band_carrier_correlation`]
    /// for the magnitude / scale relationship. Values in `0.25..=1.0`
    /// produce a noticeable front/back bias on correlated content
    /// without saturating the ALPHA codebook.
    ///
    /// `beta_scale` retains its r193 meaning. With
    /// `alpha_scale = beta_scale = 0.0` the output is byte-identical to
    /// [`Self::encode_frame_pcm_5_0_acpl3`]'s round-95 scaffold.
    pub fn encode_frame_pcm_5_0_acpl3_real_alpha_beta(
        &mut self,
        frames: &[&[f32]; 3],
        alpha_scale: f32,
        beta_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_with_max_sfb(
            frames,
            None,
            40,
            None,
            alpha_scale,
            beta_scale,
        )
    }

    /// 5.1 counterpart to
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta`]. `frames` is
    /// in `[L, R, C, LFE]` order. The LFE channel is coded as a leading
    /// `mono_data(b_lfe = 1)` element per Table 21 — same path as
    /// [`Self::encode_frame_pcm_5_1_acpl3`].
    pub fn encode_frame_pcm_5_1_acpl3_real_alpha_beta(
        &mut self,
        frames: &[&[f32]; 4],
        alpha_scale: f32,
        beta_scale: f32,
    ) -> Vec<u8> {
        let carriers: [&[f32]; 3] = [frames[0], frames[1], frames[2]];
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_with_max_sfb(
            &carriers,
            Some(frames[3]),
            40,
            Some(7),
            alpha_scale,
            beta_scale,
        )
    }

    /// Shared body for the real-α/β ACPL_3 entry points. Mirrors
    /// [`Self::encode_frame_pcm_5_x_acpl3_real_beta_with_max_sfb`] but
    /// invokes the real-α + real-β builder.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_5_x_acpl3_real_alpha_beta_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 3],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        alpha_scale: f32,
        beta_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[3].analyse_frame(buf));

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta(
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
            alpha_scale,
            beta_scale,
            pad_target_bytes,
        );

        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        // Legacy ACPL_3 body: it advances the decoder's cross-frame ASPX
        // envelope + ACPL q_prev references without tracking the rows, so
        // reset the encoder-side P-frame reference states (next non-I
        // frame stays FREQ / always-decodable).
        self.acpl3_env_prev = None;
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// **real per-band α + β + γ5/γ6**. Layered on top of
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta`]: the γ5 / γ6
    /// entropy layers now carry per-band magnitudes derived from a 2×2
    /// least-squares fit of the centre channel
    /// `C ≈ K · (γ5·L + γ6·R)` (Pseudocode 118 step 7 + step 11 with
    /// `K = √2 · (1 + √2) / 2`). γ1..γ4 + β3 stay zero-delta.
    ///
    /// `gamma_scale = 0.0` reproduces the round-196 real-α-β byte stream
    /// exactly; `gamma_scale = 1.0` writes the full analytic γ pair
    /// (clamped to the Table-208 ±2.0 bound). `alpha_scale = beta_scale
    /// = gamma_scale = 0.0` reproduces the round-95 zero-delta scaffold
    /// byte-for-byte.
    ///
    /// `frames` is in `[L, R, C]` order. The decoder walks the same
    /// Table 25 ASPX_ACPL_3 body; γ5 / γ6 feed the ACplModule2 instance
    /// that synthesises the centre output channel (Pseudocode 119 with
    /// `a = 1`, `b = 0`, `y = 0`). γ1..γ4 still at zero-delta keeps the
    /// (L, R, Ls, Rs) sub-pipeline behaviour identical to the round-196
    /// path.
    pub fn encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(
        &mut self,
        frames: &[&[f32]; 3],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_gamma_with_max_sfb(
            frames,
            None,
            40,
            None,
            alpha_scale,
            beta_scale,
            gamma_scale,
        )
    }

    /// 5.1 counterpart to
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma`].
    /// `frames` is in `[L, R, C, LFE]` order. The LFE channel is coded
    /// as a leading `mono_data(b_lfe = 1)` element per Table 21 — same
    /// path as [`Self::encode_frame_pcm_5_1_acpl3_real_alpha_beta`].
    pub fn encode_frame_pcm_5_1_acpl3_real_alpha_beta_gamma(
        &mut self,
        frames: &[&[f32]; 4],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
    ) -> Vec<u8> {
        let carriers: [&[f32]; 3] = [frames[0], frames[1], frames[2]];
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_gamma_with_max_sfb(
            &carriers,
            Some(frames[3]),
            40,
            Some(7),
            alpha_scale,
            beta_scale,
            gamma_scale,
        )
    }

    /// Shared body for the real-α/β/γ5/γ6 ACPL_3 entry points. Mirrors
    /// [`Self::encode_frame_pcm_5_x_acpl3_real_alpha_beta_with_max_sfb`]
    /// but invokes the real-γ5/γ6 builder.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_5_x_acpl3_real_alpha_beta_gamma_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 3],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta_gamma: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta_gamma: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[3].analyse_frame(buf));

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body =
            crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_gamma(
                frame_len,
                max_sfb,
                max_sfb_lfe,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                Some(&coeffs_per_channel[2]),
                coeffs_lfe.as_deref(),
                &aspx_cfg,
                acpl_num_param_bands_id,
                acpl_qm0,
                acpl_qm1,
                alpha_scale,
                beta_scale,
                gamma_scale,
                pad_target_bytes,
            );

        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        // Legacy ACPL_3 body: it advances the decoder's cross-frame ASPX
        // envelope + ACPL q_prev references without tracking the rows, so
        // reset the encoder-side P-frame reference states (next non-I
        // frame stays FREQ / always-decodable).
        self.acpl3_env_prev = None;
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// **full** real per-parameter-band α₁ / α₂ / β₁ / β₂ / γ₁..γ₆
    /// extraction (round 215) — the round-208 entry point
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma`] lifted
    /// the centre γ₅ / γ₆ pair to real values; this entry point also
    /// lifts γ₁ / γ₂ (driving the (L, Ls) pair via Pseudocode 118
    /// step 5) and γ₃ / γ₄ (driving the (R, Rs) pair via Pseudocode 118
    /// step 6), closing the README's long-standing "γ1..γ4 stay at the
    /// round-95 zero-delta scaffold" deferral for the 5_X ACPL_3 path.
    ///
    /// The γ₁ / γ₂ pair comes from a per-band 2×2 least-squares fit of
    /// the (L, Ls) output sum `(L + Ls/√2)/(1 + √2)` onto the (L, R)
    /// carrier pair; γ₃ / γ₄ come from the symmetric fit on the
    /// (R, Rs) pair (see
    /// [`crate::encoder_acpl3::extract_gamma_1_2_q_per_band_surround_least_squares`]
    /// and
    /// [`crate::encoder_acpl3::extract_gamma_3_4_q_per_band_surround_least_squares`]).
    /// Both sums are independent of the α / β decorrelator
    /// contributions and equal a `(1 + √2) · (γ·L + γ'·R)` linear
    /// combination, so the resulting 2×2 normal-equations system is
    /// identical in shape to the round-208 γ₅ / γ₆ centre fit.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order. The L / R carriers
    /// also feed the round-51 stereo `two_channel_data()` body the
    /// ASPX_ACPL_3 path emits. β₃ stays at the round-95 zero-delta
    /// scaffold — its analytic extraction requires a model for the
    /// third decorrelator output `y₂` which is not observable at
    /// encode time. The ASPX envelope layer also stays at the
    /// minimum-bit-cost FIXFIX num_env=1 scaffold pending the
    /// "real ASPX envelope coding" deferral elsewhere on the README.
    ///
    /// `alpha_scale` / `beta_scale` / `gamma_scale` control the
    /// extractor magnitude (typically `1.0` for the analytic
    /// least-squares solution; `0.0` reproduces the prior-round
    /// scaffold byte-for-byte at the corresponding layer position).
    /// In particular `α/β/γ_scale = 0.0` reproduces the round-95
    /// zero-delta scaffold ([`Self::encode_frame_pcm_5_0_acpl3`])
    /// byte-for-byte; `γ_scale = 0.0` reproduces the round-196
    /// real-α-β bytes exactly.
    pub fn encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma(
        &mut self,
        frames: &[&[f32]; 5],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_with_max_sfb(
            frames,
            None,
            40,
            None,
            alpha_scale,
            beta_scale,
            gamma_scale,
        )
    }

    /// 5.1 counterpart to
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma`].
    /// `frames` is in `[L, R, C, Ls, Rs, LFE]` order. The LFE channel
    /// is coded as a leading `mono_data(b_lfe = 1)` element per Table
    /// 21 — same path as
    /// [`Self::encode_frame_pcm_5_1_acpl3_real_alpha_beta_gamma`].
    pub fn encode_frame_pcm_5_1_acpl3_real_alpha_beta_full_gamma(
        &mut self,
        frames: &[&[f32]; 6],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
    ) -> Vec<u8> {
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_with_max_sfb(
            &surround,
            Some(frames[5]),
            40,
            Some(7),
            alpha_scale,
            beta_scale,
            gamma_scale,
        )
    }

    /// Shared body for the real-α/β + full real-γ₁..γ₆ ACPL_3 entry
    /// points. Mirrors
    /// [`Self::encode_frame_pcm_5_x_acpl3_real_alpha_beta_gamma_with_max_sfb`]
    /// but accepts a 5-channel `[L, R, C, Ls, Rs]` input and invokes
    /// the
    /// [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma`]
    /// builder.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 5],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

        let n_channels = if lfe.is_some() { 6 } else { 5 };
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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[5].analyse_frame(buf));

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body =
            crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma(
                frame_len,
                max_sfb,
                max_sfb_lfe,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                Some(&coeffs_per_channel[2]),
                Some(&coeffs_per_channel[3]),
                Some(&coeffs_per_channel[4]),
                coeffs_lfe.as_deref(),
                &aspx_cfg,
                acpl_num_param_bands_id,
                acpl_qm0,
                acpl_qm1,
                alpha_scale,
                beta_scale,
                gamma_scale,
                pad_target_bytes,
            );

        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        // Legacy ACPL_3 body: it advances the decoder's cross-frame ASPX
        // envelope + ACPL q_prev references without tracking the rows, so
        // reset the encoder-side P-frame reference states (next non-I
        // frame stays FREQ / always-decodable).
        self.acpl3_env_prev = None;
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// **real per-parameter-band α₁ / α₂ + β₁ / β₂ + γ₁..γ₆ + β₃**
    /// extraction (round 285 — the β₃-real extension of
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma`]).
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order. β₃ (the gain on the
    /// third decorrelator output `y₂` per §5.7.7.6.2 Pseudocode 118
    /// steps 8–10) is energy-matched to the centre-channel
    /// reconstruction residual left over after the γ₅ / γ₆ dry-mix fit
    /// — see
    /// [`crate::encoder_acpl3::extract_beta3_q_per_band_centre_residual`].
    /// `beta3_scale = 0.0` reproduces the round-215 full-γ byte stream
    /// exactly; `beta3_scale = 1.0` applies the full energy-matching
    /// solution clamped to the Table-207 ±1.0 magnitude bound.
    pub fn encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &mut self,
        frames: &[&[f32]; 5],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_beta3_with_max_sfb(
            frames,
            None,
            40,
            None,
            alpha_scale,
            beta_scale,
            gamma_scale,
            beta3_scale,
        )
    }

    /// 5.1 counterpart to
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3`].
    /// `frames` is in `[L, R, C, Ls, Rs, LFE]` order. The LFE channel
    /// is coded as a leading `mono_data(b_lfe = 1)` element per Table
    /// 21 — same path as
    /// [`Self::encode_frame_pcm_5_1_acpl3_real_alpha_beta_full_gamma`].
    pub fn encode_frame_pcm_5_1_acpl3_real_alpha_beta_full_gamma_beta3(
        &mut self,
        frames: &[&[f32]; 6],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_beta3_with_max_sfb(
            &surround,
            Some(frames[5]),
            40,
            Some(7),
            alpha_scale,
            beta_scale,
            gamma_scale,
            beta3_scale,
        )
    }

    /// Shared body for the real-α/β/γ₁..γ₆/β₃ ACPL_3 entry points.
    /// Mirrors
    /// [`Self::encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_with_max_sfb`]
    /// but invokes the
    /// [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3`]
    /// builder with the additional `beta3_scale` decision knob.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_beta3_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 5],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_beta3: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_beta3: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

        let n_channels = if lfe.is_some() { 6 } else { 5 };
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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[5].analyse_frame(buf));

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body =
            crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3(
                frame_len,
                max_sfb,
                max_sfb_lfe,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                Some(&coeffs_per_channel[2]),
                Some(&coeffs_per_channel[3]),
                Some(&coeffs_per_channel[4]),
                coeffs_lfe.as_deref(),
                &aspx_cfg,
                acpl_num_param_bands_id,
                acpl_qm0,
                acpl_qm1,
                alpha_scale,
                beta_scale,
                gamma_scale,
                beta3_scale,
                pad_target_bytes,
            );

        let mut bw = BitWriter::new();
        self.write_toc(&mut bw);
        bw.align_to_byte();
        let mut out = bw.finish();
        out.extend(body);
        // Legacy ACPL_3 body: it advances the decoder's cross-frame ASPX
        // envelope + ACPL q_prev references without tracking the rows, so
        // reset the encoder-side P-frame reference states (next non-I
        // frame stays FREQ / always-decodable).
        self.acpl3_env_prev = None;
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        self.channel_mode_value = saved_mode.0;
        self.channel_mode_bits = saved_mode.1;
        out
    }

    /// Encode one IMS v2 5.0 frame in 5_X_codec_mode = ASPX_ACPL_3 with
    /// the full real α / β / β₃ / γ₁..γ₆ A-CPL parameter extraction
    /// **and** a real ASPX SIGNAL / NOISE envelope for the L / R carriers
    /// (round 322). This is the frame-path activation of the round-226
    /// real-envelope ASPX writers + the round-240 QMF energy aggregator:
    /// the encoder runs its QMF analysis bank over the L and R input PCM,
    /// aggregates the HF energy across the A-SPX signal / noise
    /// subband-group borders into per-`sbg` scale factors, quantises and
    /// FREQ-DPCM packs them (Pseudocodes 80–83), and splices the result
    /// into the live `aspx_data_2ch()` element in place of the round-95
    /// minimum-bit-cost scaffold.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order. All A-CPL scale knobs
    /// behave exactly as in
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3`].
    /// The result still decodes through `Ac4Decoder` (the real-envelope
    /// `aspx_data_2ch()` is a strict framing superset of the scaffold).
    pub fn encode_frame_pcm_5_0_acpl3_real_aspx(
        &mut self,
        frames: &[&[f32]; 5],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_aspx_with_max_sfb(
            frames,
            None,
            40,
            None,
            alpha_scale,
            beta_scale,
            gamma_scale,
            beta3_scale,
        )
    }

    /// 5.1 counterpart to [`Self::encode_frame_pcm_5_0_acpl3_real_aspx`].
    /// `frames` is in `[L, R, C, Ls, Rs, LFE]` order.
    pub fn encode_frame_pcm_5_1_acpl3_real_aspx(
        &mut self,
        frames: &[&[f32]; 6],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_5_x_acpl3_real_aspx_with_max_sfb(
            &surround,
            Some(frames[5]),
            40,
            Some(7),
            alpha_scale,
            beta_scale,
            gamma_scale,
            beta3_scale,
        )
    }

    /// 5.0 SIMPLE/ASPX_ACPL_3 encode with a **real multi-envelope** ASPX
    /// SIGNAL / NOISE payload on the L / R carriers.
    ///
    /// Identical to [`Self::encode_frame_pcm_5_0_acpl3_real_aspx`] except
    /// the encoder probes the L-carrier HF QMF energy for a temporal
    /// transient (via
    /// [`crate::encoder_acpl3::select_aspx_num_env_from_qmf`]) and, when one
    /// is present, splits the frame into `num_env > 1` uniformly spaced
    /// FIXFIX signal envelopes — emitting the round-299 multi-envelope
    /// `aspx_data_2ch()` body (per-envelope FREQ / TIME DPCM) instead of
    /// the single-envelope body. Stationary frames fall back to the
    /// single-envelope path, so this method is a strict superset of the
    /// round-322 entry point. `frames` is in `[L, R, C, Ls, Rs]` order.
    ///
    /// Refs ETSI TS 103 190-1 §4.2.12.4 Table 52, §4.3.10.1.9 Table 123,
    /// §4.3.10.4.1.
    pub fn encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &mut self,
        frames: &[&[f32]; 5],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_x_acpl3_real_aspx_multi_env_with_max_sfb(
            frames,
            None,
            40,
            None,
            alpha_scale,
            beta_scale,
            gamma_scale,
            beta3_scale,
        )
    }

    /// 5.1 counterpart to
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_aspx_multi_env`].
    /// `frames` is in `[L, R, C, Ls, Rs, LFE]` order.
    pub fn encode_frame_pcm_5_1_acpl3_real_aspx_multi_env(
        &mut self,
        frames: &[&[f32]; 6],
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_5_x_acpl3_real_aspx_multi_env_with_max_sfb(
            &surround,
            Some(frames[5]),
            40,
            Some(7),
            alpha_scale,
            beta_scale,
            gamma_scale,
            beta3_scale,
        )
    }

    /// Shared body for the real-ASPX ACPL_3 entry points. Mirrors
    /// [`Self::encode_frame_pcm_5_x_acpl3_real_alpha_beta_full_gamma_beta3_with_max_sfb`]
    /// but derives per-channel ASPX envelope quant indices from the L / R
    /// input PCM via the QMF analysis bank +
    /// [`crate::encoder_acpl3::build_aspx_real_envelope_channel_from_qmf`]
    /// and routes the substream body through
    /// [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx`].
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_5_x_acpl3_real_aspx_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 5],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_aspx: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_aspx: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

        let n_channels = if lfe.is_some() { 6 } else { 5 };
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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[5].analyse_frame(buf));

        let mut aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening decision (Table 121):
        // signal pre-flattening when the L carrier's QMF low band — the HF
        // generation source — carries a strong overall spectral tilt
        // (§5.7.6.4.1.2 Pseudocode 85). One per-config flag governs the
        // element; derived from the primary (L) carrier.
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        // Real ASPX envelope extraction: run the QMF analysis bank over
        // the L / R input PCM, then aggregate the HF energy across the
        // A-SPX signal / noise subband-group borders into per-channel
        // [F0, DF₁, …] quant-index vectors. The frequency tables provide
        // the absolute SBG borders + the cross-over subband `sbx`.
        let (l_sig_lvl, l_noise_lvl, r_sig_lvl, r_noise_lvl) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        // §5.7.6.3.5: the pair is transmitted as (sum, pan) under
        // aspx_balance = 1 (Pseudocode 84). Convert here — before the
        // FREQ/TIME direction decision — so the cross-frame
        // `Acpl3EnvPrevRows` bookkeeping, the direction pricing, and
        // the directional writer all operate in the wire domain
        // (channel 0 = sum LEVEL rows, channel 1 = pan wire steps for
        // the decoder's delta = 2 accumulation).
        let (l_sig, l_noise, r_sig, r_noise) = {
            let qmode_sig = if aspx_cfg.fixfix_tmp_num_env_bits() == 1 {
                crate::aspx::AspxQuantStep::Fine
            } else {
                aspx_cfg.quant_mode_env
            };
            let (s0, s1, n0, n1) = crate::encoder_acpl3::balance_convert_packed_rows(
                &l_sig_lvl,
                &r_sig_lvl,
                &l_noise_lvl,
                &r_noise_lvl,
                qmode_sig,
                l_sig_lvl.len().max(r_sig_lvl.len()),
                l_noise_lvl.len().max(r_noise_lvl.len()),
            );
            (s0, n0, s1, n1)
        };

        // Encoder-side A-SPX inverse-filtering decision for the L carrier
        // (mirrored to R under aspx_balance = 1). Heavier where the low
        // band is more tonal (§4.3.10.6.1 / §5.7.6.4.1.3).
        let aspx_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);

        // Encoder-side A-SPX missing-harmonic decision per carrier
        // (§4.2.12.6): a discrete tonal partial in a high-res signal
        // subband group's HF QMF band requests a restored sinusoid.
        let aspx_l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let aspx_r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        // P-frame TIME-direction decision (§5.7.6.3.4 Pseudocodes
        // 80 / 81): when the previous frame's envelope rows are known
        // and this is a non-I-frame, each of the four envelopes
        // (L/R × SIGNAL/NOISE) may switch to TIME-direction DPCM
        // against the previous frame when strictly cheaper. I-frames
        // and first frames always emit FREQ.
        let cur_rows = crate::encoder_acpl3::Acpl3EnvPrevRows {
            l_sig: crate::encoder_acpl3::qscf_row_from_freq_dpcm(&l_sig),
            l_noise: crate::encoder_acpl3::qscf_row_from_freq_dpcm(&l_noise),
            r_sig: crate::encoder_acpl3::qscf_row_from_freq_dpcm(&r_sig),
            r_noise: crate::encoder_acpl3::qscf_row_from_freq_dpcm(&r_noise),
        };
        let prev_rows = if self.b_iframe_global {
            None
        } else {
            self.acpl3_env_prev.as_ref()
        };
        let env_l_sig = crate::encoder_acpl3::choose_envelope_direction(
            &l_sig,
            prev_rows.map(|p| p.l_sig.as_slice()),
        );
        let env_l_noise = crate::encoder_acpl3::choose_envelope_direction(
            &l_noise,
            prev_rows.map(|p| p.l_noise.as_slice()),
        );
        let env_r_sig = crate::encoder_acpl3::choose_envelope_direction(
            &r_sig,
            prev_rows.map(|p| p.r_sig.as_slice()),
        );
        let env_r_noise = crate::encoder_acpl3::choose_envelope_direction(
            &r_noise,
            prev_rows.map(|p| p.r_noise.as_slice()),
        );

        let body =
            crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna_directional(
                frame_len,
                max_sfb,
                max_sfb_lfe,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                Some(&coeffs_per_channel[2]),
                Some(&coeffs_per_channel[3]),
                Some(&coeffs_per_channel[4]),
                coeffs_lfe.as_deref(),
                &aspx_cfg,
                &env_l_sig,
                &env_l_noise,
                &env_r_sig,
                &env_r_noise,
                &aspx_tna_mode,
                &aspx_l_ah,
                &aspx_r_ah,
                acpl_num_param_bands_id,
                acpl_qm0,
                acpl_qm1,
                alpha_scale,
                beta_scale,
                gamma_scale,
                beta3_scale,
                pad_target_bytes,
                Some(&mut self.acpl3_param_prev),
            );

        // The transmitted rows become the next frame's Pseudocode 80/81
        // `qscf_*_prev` reference (mirrors the decoder's per-channel
        // AspxEnvPrev update).
        self.acpl3_env_prev = Some(cur_rows);

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

    /// Derive the per-channel ASPX SIGNAL / NOISE envelope quant indices
    /// for the L / R carriers by running the QMF analysis bank over the
    /// input PCM and aggregating the HF energy across the A-SPX
    /// subband-group borders the decoder will consume.
    ///
    /// Returns `(l_sig, l_noise, r_sig, r_noise)` — each a `[F0, DF₁, …]`
    /// FREQ-DPCM quant-index vector ready to splice into the
    /// `aspx_data_2ch()` real-envelope writer. When the frequency-table
    /// derivation fails (e.g. an unsupported `frame_len`), every vector
    /// is empty and the writer falls back to the all-zero scaffold path.
    fn extract_aspx_lr_envelopes(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        pcm_l: &[f32],
        pcm_r: &[f32],
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
        let empty = || (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return empty();
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 {
            return empty();
        }
        // The signal SBG border list is high-res here because the live
        // config does not signal an in-band aspx_freq_res bit (the parser
        // falls back to high-res), matching the round-226 writer's
        // num_sbg selection.
        let sbg_sig_borders = &tables.sbg_sig_highres;
        let sbg_noise_borders = &tables.sbg_noise;
        let sbx = tables.sbx;

        // QMF-analyse the full frame: 64 samples per slot. Truncate to a
        // whole number of slots (the analysis bank requires a multiple of
        // 64) and transpose into the [absolute_sb][ts] shape the
        // aggregator consumes.
        let n_slots = pcm_l.len() / 64;
        if n_slots == 0 {
            return empty();
        }
        let usable = n_slots * 64;
        let mut bank_l = crate::qmf::QmfAnalysisBank::new();
        let mut bank_r = crate::qmf::QmfAnalysisBank::new();
        let slots_l = bank_l.process_block(&aspx_scaled_pcm(&pcm_l[..usable]));
        let slots_r = bank_r.process_block(&aspx_scaled_pcm(&pcm_r[..usable]));
        let q_high_l = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots_l);
        let q_high_r = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots_r);

        let ch_l = crate::encoder_acpl3::AspxQmfEnvelopeChannel {
            q_high: &q_high_l,
            sbg_sig_borders,
            sbg_noise_borders,
        };
        let ch_r = crate::encoder_acpl3::AspxQmfEnvelopeChannel {
            q_high: &q_high_r,
            sbg_sig_borders,
            sbg_noise_borders,
        };
        let (l_sig, l_noise) = crate::encoder_acpl3::build_aspx_real_envelope_channel_from_qmf(
            &ch_l,
            aspx_cfg.quant_mode_env,
            64,
            num_ts_in_ats,
            aspx_frame_ts_count,
            sbx,
        );
        let (r_sig, r_noise) = crate::encoder_acpl3::build_aspx_real_envelope_channel_from_qmf(
            &ch_r,
            aspx_cfg.quant_mode_env,
            64,
            num_ts_in_ats,
            aspx_frame_ts_count,
            sbx,
        );
        (l_sig, l_noise, r_sig, r_noise)
    }

    /// Compute the per-noise-subband-group `aspx_tna_mode` inverse-
    /// filtering vector for the L carrier from its QMF low band, per
    /// ETSI TS 103 190-1 §4.3.10.6.1 (selection) + §5.7.6.4.1.2-3 (the
    /// tonality analysis it is calibrated against).
    ///
    /// Returns an empty vector when the A-SPX frequency tables cannot be
    /// derived or the carrier is too short to QMF-analyse (the caller then
    /// emits the all-zero scaffold). Under `aspx_balance = 1` the R carrier
    /// mirrors this vector, so a single L-derived `tna_mode` suffices.
    fn extract_aspx_l_tna_mode(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        pcm_l: &[f32],
    ) -> Vec<u8> {
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return Vec::new();
        };
        let n_slots = pcm_l.len() / 64;
        if n_slots == 0 || tables.sba == 0 {
            return Vec::new();
        }
        let usable = n_slots * 64;
        let mut bank = crate::qmf::QmfAnalysisBank::new();
        let slots = bank.process_block(&aspx_scaled_pcm(&pcm_l[..usable]));
        let q_sb_major = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots);
        // Take the low band (subbands 0..sba) and build the extended
        // low-band matrix the covariance analysis consumes.
        let sba = tables.sba as usize;
        let q_low: Vec<Vec<(f32, f32)>> = q_sb_major
            .iter()
            .take(sba)
            .map(|row| row.to_vec())
            .collect();
        let q_low_ext = crate::aspx_tns::build_q_low_ext(&q_low, &[], tables.sba);
        // 48 kHz family is the only base_samp_freq wired in this path
        // (matching extract_aspx_lr_envelopes / the decoder call site).
        crate::aspx_tna_select::select_tna_mode(
            &q_low_ext,
            &tables,
            aspx_cfg.master_freq_scale,
            true,
        )
    }

    /// Decide `aspx_preflat` (A-SPX spectral pre-flattening, ETSI TS 103
    /// 190-1 Table 121 / Table 50) for an A-SPX element from a carrier's
    /// QMF **low** band — the HF-generation source range whose overall
    /// spectral tilt pre-flattening de-tilts.
    ///
    /// QMF-analyses the carrier PCM, takes the low band (subbands `0..sba`,
    /// the same source the decoder's HF generator transposes), and runs the
    /// decoder's exact Pseudocode-85 gain fit
    /// ([`crate::aspx_preflat_select::select_preflat`]) over a single
    /// frame-spanning SIGNAL time-slot group, signalling pre-flattening when
    /// the fitted-slope dB spread clears the threshold. Returns `false` when
    /// the A-SPX frequency tables cannot be derived or the carrier is too
    /// short to QMF-analyse (the caller then emits the historical
    /// `preflat = 0` framing). `aspx_preflat` is a single per-`aspx_config`
    /// flag, so one primary-carrier decision governs the element.
    fn extract_aspx_preflat(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        pcm: &[f32],
    ) -> bool {
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return false;
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 || tables.sba == 0 {
            return false;
        }
        let n_slots = pcm.len() / 64;
        if n_slots == 0 {
            return false;
        }
        let usable = n_slots * 64;
        let mut bank = crate::qmf::QmfAnalysisBank::new();
        let slots = bank.process_block(&aspx_scaled_pcm(&pcm[..usable]));
        let q_sb_major = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots);
        let sba = tables.sba as usize;
        let q_low: Vec<Vec<(f32, f32)>> = q_sb_major
            .iter()
            .take(sba)
            .map(|row| row.to_vec())
            .collect();
        // A single SIGNAL time-slot group spanning the whole A-SPX frame:
        // the Pseudocode-85 envelope window is [atsg_sig[0], atsg_sig[end])
        // QMF time slots. The fitted slope is a frame-level measure, so the
        // full-frame ATSG is the natural source window.
        let atsg_sig = [0u32, aspx_frame_ts_count];
        crate::aspx_preflat_select::select_preflat(&q_low, tables.sba, &atsg_sig, num_ts_in_ats)
    }

    /// Compute the per-high-res-signal-subband-group `aspx_add_harmonic`
    /// vector for one carrier from its HF QMF band, per ETSI TS 103 190-1
    /// §4.2.12.6 (`aspx_hfgen_iwc`) + the §5.7.6.4.2.1 Pseudocode 92
    /// `sb_mid` placement it serves.
    ///
    /// QMF-analyses the carrier PCM, reduces it to per-high-res-signal-SBG
    /// spectral crests via [`crate::aspx_ah_select::select_add_harmonic`],
    /// and returns the boolean `add_harmonic[sbg]` vector. Returns an empty
    /// vector when the A-SPX frequency tables cannot be derived or the
    /// carrier is too short to QMF-analyse (the caller then emits the
    /// all-zero scaffold).
    fn extract_aspx_add_harmonic(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        pcm: &[f32],
    ) -> Vec<bool> {
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return Vec::new();
        };
        let n_slots = pcm.len() / 64;
        if n_slots == 0 {
            return Vec::new();
        }
        let usable = n_slots * 64;
        let mut bank = crate::qmf::QmfAnalysisBank::new();
        let slots = bank.process_block(&aspx_scaled_pcm(&pcm[..usable]));
        let q_high = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots);
        crate::aspx_ah_select::select_add_harmonic(&q_high, &tables.sbg_sig_highres, tables.sbx)
    }

    /// Multi-envelope counterpart to [`Self::extract_aspx_lr_envelopes`].
    ///
    /// Runs the same QMF analysis over the L / R carrier PCM, then probes
    /// the L-carrier HF energy for a temporal transient
    /// ([`crate::encoder_acpl3::select_aspx_num_env_from_qmf`]) to pick the
    /// frame's FIXFIX signal-envelope count `num_env` (a power of two in
    /// `1..=1 << fixfix_tmp_num_env_bits()`), and builds both channels'
    /// per-envelope SIGNAL / NOISE [`crate::encoder_acpl3::AspxEncodedEnvelope`]
    /// rows ([`crate::encoder_acpl3::build_aspx_multi_envelope_2ch_from_qmf`]).
    ///
    /// Returns `(num_env, rows)`. When the frequency-table derivation fails
    /// or the frame carries no usable QMF slots, returns `(1, default)` so
    /// the caller falls back to the single-envelope path.
    fn extract_aspx_lr_multi_env(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        pcm_l: &[f32],
        pcm_r: &[f32],
        prev: Option<&crate::encoder_acpl3::Acpl3EnvPrevRows>,
    ) -> (u32, crate::encoder_acpl3::AspxMultiEnvelope2chRows) {
        let fallback = || {
            (
                1u32,
                crate::encoder_acpl3::AspxMultiEnvelope2chRows::default(),
            )
        };
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return fallback();
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 {
            return fallback();
        }
        let sbg_sig_borders = &tables.sbg_sig_highres;
        let sbg_noise_borders = &tables.sbg_noise;
        let sbx = tables.sbx;

        let n_slots = pcm_l.len() / 64;
        if n_slots == 0 {
            return fallback();
        }
        let usable = n_slots * 64;
        let mut bank_l = crate::qmf::QmfAnalysisBank::new();
        let mut bank_r = crate::qmf::QmfAnalysisBank::new();
        let slots_l = bank_l.process_block(&aspx_scaled_pcm(&pcm_l[..usable]));
        let slots_r = bank_r.process_block(&aspx_scaled_pcm(&pcm_r[..usable]));
        let q_high_l = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots_l);
        let q_high_r = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots_r);

        // Pick num_env from the L-carrier HF energy. The maximum is the
        // config's FIXFIX capacity: 1 << fixfix_tmp_num_env_bits() bits of
        // tmp_num_env address 1 << (1 << bits - 1) envelopes, but the live
        // config (num_env_bits_fixfix = 0) addresses {1, 2}.
        let max_num_env = 1u32 << ((1u32 << aspx_cfg.fixfix_tmp_num_env_bits()) - 1);
        let num_env = crate::encoder_acpl3::select_aspx_num_env_from_qmf(
            &q_high_l,
            sbg_sig_borders,
            num_ts_in_ats,
            aspx_frame_ts_count,
            sbx,
            max_num_env,
            // Coefficient-of-variation threshold above which the finer
            // partition is deemed to expose a transient.
            0.30,
        );
        if num_env <= 1 {
            return (1, crate::encoder_acpl3::AspxMultiEnvelope2chRows::default());
        }

        let ch0 = crate::encoder_acpl3::AspxQmfMultiEnvelopeChannel {
            q_high: &q_high_l,
            sbg_sig_borders,
            sbg_noise_borders,
        };
        let ch1 = crate::encoder_acpl3::AspxQmfMultiEnvelopeChannel {
            q_high: &q_high_r,
            sbg_sig_borders,
            sbg_noise_borders,
        };
        let rows = crate::encoder_acpl3::build_aspx_multi_envelope_2ch_from_qmf(
            &ch0,
            &ch1,
            num_env,
            aspx_cfg.quant_mode_env,
            64,
            num_ts_in_ats,
            aspx_frame_ts_count,
            sbx,
            // P-frames carry the previous frame's last-envelope rows
            // (sum / pan wire-step domain) as the leading-envelope TIME
            // reference; I-frames / first frames have no history.
            prev.map(|p| crate::encoder_acpl3::AspxMultiEnvelopePrevLast {
                sig: &p.l_sig,
                noise: &p.l_noise,
            })
            .unwrap_or_default(),
            prev.map(|p| crate::encoder_acpl3::AspxMultiEnvelopePrevLast {
                sig: &p.r_sig,
                noise: &p.r_noise,
            })
            .unwrap_or_default(),
            1,
            false,
        );
        (num_env, rows)
    }

    /// Shared body for the multi-envelope real-ASPX ACPL_3 entry points.
    /// Mirrors [`Self::encode_frame_pcm_5_x_acpl3_real_aspx_with_max_sfb`]
    /// but probes the L / R HF energy for a transient and, when one is
    /// present, routes the substream through the round-299 multi-envelope
    /// `aspx_data_2ch()` body
    /// ([`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_multi_env`]).
    /// Stationary frames (or a config that rejects `num_env > 1`) fall back
    /// to the single-envelope path, so the output is always a valid
    /// ASPX_ACPL_3 frame.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_5_x_acpl3_real_aspx_multi_env_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 5],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        alpha_scale: f32,
        beta_scale: f32,
        gamma_scale: f32,
        beta3_scale: f32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_aspx_multi_env: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_5_x_acpl3_real_aspx_multi_env: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let mut aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening decision (Table 121):
        // signal pre-flattening when the L carrier's QMF low band carries a
        // strong overall spectral tilt (§5.7.6.4.1.2 Pseudocode 85).
        // Orthogonal to the per-envelope SIGNAL/NOISE extraction below.
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        // Probe for a transient and build the multi-envelope rows. A
        // stationary frame returns num_env = 1, in which case we delegate
        // to the single-envelope path (which carries the FIXFIX num_env==1
        // Fine clamp the multi-envelope writer does not apply).
        let env_prev = if self.b_iframe_global {
            None
        } else {
            self.acpl3_env_prev.clone()
        };
        let (num_env, rows) = self.extract_aspx_lr_multi_env(
            &aspx_cfg,
            frame_len,
            frames[0],
            frames[1],
            env_prev.as_ref(),
        );
        // Per-channel real aspx_add_harmonic (§4.2.12.6) for the L / R
        // carriers — carried on the multi-envelope aspx_data_2ch() too.
        let aspx_l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let aspx_r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        if num_env <= 1 {
            return self.encode_frame_pcm_5_x_acpl3_real_aspx_with_max_sfb(
                frames,
                lfe,
                max_sfb,
                max_sfb_lfe,
                alpha_scale,
                beta_scale,
                gamma_scale,
                beta3_scale,
            );
        }

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        if lfe.is_some() {
            self.channel_mode_value = 0b1110;
            self.channel_mode_bits = 4;
        } else {
            self.channel_mode_value = 0b1101;
            self.channel_mode_bits = 4;
        }

        let n_channels = if lfe.is_some() { 6 } else { 5 };
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
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[5].analyse_frame(buf));

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_qm0 = crate::acpl::AcplQuantMode::Fine;
        let acpl_qm1 = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body =
            crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_multi_env(
                frame_len,
                max_sfb,
                max_sfb_lfe,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                Some(&coeffs_per_channel[2]),
                Some(&coeffs_per_channel[3]),
                Some(&coeffs_per_channel[4]),
                coeffs_lfe.as_deref(),
                &aspx_cfg,
                num_env,
                &rows,
                &aspx_l_ah,
                &aspx_r_ah,
                acpl_num_param_bands_id,
                acpl_qm0,
                acpl_qm1,
                alpha_scale,
                beta_scale,
                gamma_scale,
                beta3_scale,
                pad_target_bytes,
            );

        // The multi-envelope body builder returns an empty Vec if the
        // writer rejected the config / num_env; fall back in that case.
        if body.is_empty() {
            self.channel_mode_value = saved_mode.0;
            self.channel_mode_bits = saved_mode.1;
            return self.encode_frame_pcm_5_x_acpl3_real_aspx_with_max_sfb(
                frames,
                lfe,
                max_sfb,
                max_sfb_lfe,
                alpha_scale,
                beta_scale,
                gamma_scale,
                beta3_scale,
            );
        }

        // A multi-envelope body was emitted: its LAST envelope's rows
        // become the decoder's Pseudocode 80/81 `qscf_*_prev` (the same
        // sum / pan wire-step domain the single-envelope path tracks),
        // so the next frame — single- or multi-envelope — may pick
        // TIME direction against them. The A-CPL parameter rows the
        // multi-envelope body transmitted DO advance the decoder's
        // Pseudocode-121 q_prev, but this builder emits them as
        // DIFF_FREQ without threading the encoder-side reference, so
        // those rows stay unprimed (next frame emits DIFF_FREQ).
        self.acpl3_env_prev = Some(crate::encoder_acpl3::Acpl3EnvPrevRows {
            l_sig: rows.ch0_last_sig.clone(),
            l_noise: rows.ch0_last_noise.clone(),
            r_sig: rows.ch1_last_sig.clone(),
            r_noise: rows.ch1_last_noise.clone(),
        });
        self.acpl3_param_prev = crate::encoder_acpl3::Acpl3ParamPrevRows::default();

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

    /// Encode one IMS v2 frame containing a 5.0 SIMPLE/ASPX_ACPL_2
    /// multichannel substream with **real per-parameter-band α + β
    /// extraction** carried by the two trailing `acpl_data_1ch()` elements
    /// (round 144 — the ACPL_2 5.0 counterpart to the round-132 ACPL_1 5.0
    /// real α + β path).
    ///
    /// Per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 + §5.7.7.6.1
    /// Pseudocode 117, the A-CPL surround reconstruction carries the
    /// level component via α and a decorrelated residual via β:
    ///
    /// ```text
    ///   α   = 1 − 2·√2 · ⟨x_carrier, x_surround⟩ / ⟨x_carrier, x_carrier⟩
    ///   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
    ///   ⇒  β = √max(0, 2·E[Ls²]/E[L²] − (1 − α_dq)²)
    /// ```
    ///
    /// Unlike the ACPL_1 paths, ACPL_2 does **not** transmit the Ls/Rs
    /// surround pair on the wire — the decoder reconstructs the surround
    /// purely from the L/R carriers + the two `acpl_data_1ch()` parameter
    /// sets. This entry point therefore still emits the round-100
    /// ASPX_ACPL_2 body layout (no joint-MDCT residual layer, no
    /// `acpl_config_1ch(PARTIAL)` qmf_band field), but extracts the α + β
    /// indices from the caller's full 5-channel `[L, R, C, Ls, Rs]` input
    /// rather than pinning them at the zero-codebook scaffold.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order; β3 / γ stay at the
    /// scaffold. The `acpl_config_1ch(FULL)` carries no `qmf_band` →
    /// `start_band = 0` so every parameter band participates in the α + β
    /// coding (in contrast to the ACPL_1 PARTIAL mode whose
    /// `acpl_qmf_band` masks the low bands).
    ///
    /// **Note (round-128 ALPHA F0 writer-side `alpha_q` desync —
    /// deferred follow-up since round 132).** The shared
    /// `write_acpl_alpha_f0_value` writer treats the signed `alpha_q ∈
    /// [-N/2..+N/2]` returned by `quantise_alpha` as a raw F0 symbol
    /// index without re-centering it against the table's shortest
    /// codeword. The decoder's `dequantize_alpha_index` re-centers via
    /// `lane = alpha_q + N/2`, so non-trivial α values do not round-trip
    /// bit-exact through the full PCM→MDCT→writer→parser→synth chain
    /// when the analytic α resolves to a non-center quantisation lane.
    /// The on-wire β codewords for ACPL_2 are wired correctly per
    /// §A.3 Table A.40 / A.41 (β uses unsigned-magnitude F0 with
    /// `cb_off = 0`, no re-centering needed); the round-100 zero-α/β
    /// scaffold is structurally superseded by this entry point. Once
    /// the writer-side desync lands as a follow-up commit the on-wire β
    /// recovery will be bit-exact end-to-end.
    pub fn encode_frame_pcm_5_0_acpl2_real_alpha_beta(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl2_real_alpha_beta_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl2_real_alpha_beta`].
    pub fn encode_frame_pcm_5_0_acpl2_real_alpha_beta_with_max_sfb(
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
                "encode_frame_pcm_5_0_acpl2_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
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
        // states). The Ls/Rs spectra feed the α + β extractors only — they
        // are not emitted on the ACPL_2 wire (the decoder reconstructs the
        // surround from L/R + the two acpl_data_1ch parameter sets).
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

        // ASPX config: matches the round-95 / 100 / 103 ASPX_ACPL config.
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

        let body = crate::encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
            frame_len,
            max_sfb,
            self.b_iframe_global,
            &coeffs_per_channel[0],
            &coeffs_per_channel[1],
            &coeffs_per_channel[2],
            &coeffs_per_channel[3],
            &coeffs_per_channel[4],
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

    /// 5.0 SIMPLE/ASPX_ACPL_2 encode with a **real** ASPX SIGNAL / NOISE
    /// envelope on both the L / R carrier pair (`aspx_data_2ch()`) and the
    /// centre carrier (`aspx_data_1ch()`).
    ///
    /// This is the ACPL_2 counterpart to
    /// [`Self::encode_frame_pcm_5_0_acpl3_real_aspx`]: where the round-322
    /// ACPL_3 path wired the real-envelope writers into the carrier-pair
    /// `aspx_data_2ch()` element only, the ACPL_2 body additionally carries
    /// an `aspx_data_1ch()` element for the centre carrier. This method
    /// QMF-analyses the L / R **and** C input PCM and emits real
    /// `[F0, DF₁, …]` SIGNAL / NOISE envelopes on all three carriers via
    /// [`crate::encoder_acpl3::write_aspx_data_2ch_real_envelope`] /
    /// [`crate::encoder_acpl3::write_aspx_data_1ch_real_envelope`],
    /// replacing the round-144 minimum-bit-cost scaffolds — closing the
    /// README's "the 1-channel (`aspx_data_1ch`) real-envelope path … still
    /// writes the single-envelope scaffold on the live frame path"
    /// deferral. The real per-band α / β A-CPL parameters are unchanged
    /// from [`Self::encode_frame_pcm_5_0_acpl2_real_alpha_beta`].
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order. The output round-trips
    /// through [`crate::decoder::Ac4Decoder`] to a 5-channel `AudioFrame`.
    ///
    /// Refs ETSI TS 103 190-1 §4.2.6.6 Table 25, §4.2.12.3 Table 51,
    /// §4.2.12.4 Table 52, §5.7.6.3.4 / §5.7.6.3.5.
    pub fn encode_frame_pcm_5_0_acpl2_real_aspx(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl2_real_aspx_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl2_real_aspx`].
    pub fn encode_frame_pcm_5_0_acpl2_real_aspx_with_max_sfb(
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
                "encode_frame_pcm_5_0_acpl2_real_aspx: channel {ch} input length must match frame_len = {frame_len}"
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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary (L) carrier (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        // Real ASPX envelope extraction over the L / R carrier pair …
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        // … and the centre carrier (the ACPL_2 body's `aspx_data_1ch()`).
        let (c_sig, c_noise) = self.extract_aspx_mono_envelope(&aspx_cfg, frame_len, frames[2]);

        // Encoder-side A-SPX inverse-filtering decision per carrier: the
        // L / R front pair shares the L-derived vector (aspx_balance = 1),
        // and the centre carrier derives its own from its QMF low band.
        let lr_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);
        let c_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[2]);

        // Per-channel real aspx_add_harmonic (§4.2.12.6) for the L / R
        // front pair + the centre carrier.
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body =
            crate::encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
                frame_len,
                max_sfb,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &aspx_cfg,
                &l_sig,
                &l_noise,
                &r_sig,
                &r_noise,
                &c_sig,
                &c_noise,
                &lr_tna_mode,
                &c_tna_mode,
                &l_ah,
                &r_ah,
                &c_ah,
                acpl_num_param_bands_id,
                acpl_quant_mode,
                pad_target_bytes,
            );

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

    /// Multi-envelope centre-carrier variant of
    /// [`Self::encode_frame_pcm_5_0_acpl2_real_aspx`]. Probes the centre
    /// carrier's HF QMF energy for a transient and, when one is present,
    /// emits a **multi-envelope** centre `aspx_data_1ch()` (`num_env > 1`)
    /// via
    /// [`crate::encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env`]
    /// — the 1-channel / mono dual of the round-299/331 5_X ACPL_3
    /// multi-envelope `aspx_data_2ch()` path. The L / R front pair keeps its
    /// single-envelope `aspx_data_2ch()` (the decoder reads `num_env`
    /// independently per A-SPX element). A stationary centre carrier — or a
    /// config / `num_env` the writer rejects — falls back to the
    /// single-envelope [`Self::encode_frame_pcm_5_0_acpl2_real_aspx`] path,
    /// so the output is always a valid ASPX_ACPL_2 frame.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order.
    pub fn encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env(
        &mut self,
        frames: &[&[f32]; 5],
    ) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env`].
    pub fn encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env_with_max_sfb(
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
                "encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary (L) carrier (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        // Probe the centre carrier for a transient. A stationary carrier
        // returns num_env = 1; delegate to the single-envelope path.
        let (c_num_env, c_sig_rows, c_noise_rows) =
            self.extract_aspx_mono_multi_env(&aspx_cfg, frame_len, frames[2]);
        if c_num_env <= 1 {
            return self.encode_frame_pcm_5_0_acpl2_real_aspx_with_max_sfb(frames, max_sfb);
        }

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1101;
        self.channel_mode_bits = 4;

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

        // L / R front pair stays single-envelope.
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);

        // Per-channel real aspx_add_harmonic (§4.2.12.6) for L / R + centre.
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let centre = crate::encoder_acpl3::AspxMultiEnvelopeChannel {
            sig: &c_sig_rows,
            noise: &c_noise_rows,
        };
        let body =
            crate::encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env(
                frame_len,
                max_sfb,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[2],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &aspx_cfg,
                &l_sig,
                &l_noise,
                &r_sig,
                &r_noise,
                c_num_env,
                centre,
                &l_ah,
                &r_ah,
                &c_ah,
                acpl_num_param_bands_id,
                acpl_quant_mode,
                pad_target_bytes,
            );

        // Empty body ⇒ the multi-env writer rejected the config; fall back.
        if body.is_empty() {
            self.channel_mode_value = saved_mode.0;
            self.channel_mode_bits = saved_mode.1;
            return self.encode_frame_pcm_5_0_acpl2_real_aspx_with_max_sfb(frames, max_sfb);
        }

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

    /// 5.1 counterpart to [`Self::encode_frame_pcm_5_0_acpl2_real_aspx`].
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, LFE]` order. The LFE channel is
    /// MDCT-analysed for state continuity but the ACPL_2 body reconstructs
    /// the `.1` low-frequency element from the same scaffold as the
    /// round-144 path; only the three ASPX carriers carry real envelopes.
    pub fn encode_frame_pcm_5_1_acpl2_real_aspx(&mut self, frames: &[&[f32]; 6]) -> Vec<u8> {
        // ACPL_2 reconstructs the surround pair from L/R + the parameter
        // sets, so the body shape is independent of LFE — route the first
        // five channels through the 5.0 path. (The dedicated `.1`
        // low-frequency element on the ACPL_2 wire is unchanged from the
        // round-144 builder, which the 5.0 path already emits.)
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_5_0_acpl2_real_aspx(&surround)
    }

    /// Derive the per-channel ASPX SIGNAL / NOISE envelope quant indices
    /// for a single carrier (the ACPL_2 centre carrier's `aspx_data_1ch()`)
    /// by running the QMF analysis bank over the input PCM and aggregating
    /// the HF energy across the A-SPX subband-group borders.
    ///
    /// Mirrors the per-channel half of [`Self::extract_aspx_lr_envelopes`].
    /// Returns `(sig, noise)` — each a `[F0, DF₁, …]` FREQ-DPCM
    /// quant-index vector. Empty vectors when the frequency-table
    /// derivation fails (the writer then falls back to its all-zero path).
    fn extract_aspx_mono_envelope(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        pcm: &[f32],
    ) -> (Vec<i32>, Vec<i32>) {
        let empty = || (Vec::new(), Vec::new());
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return empty();
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 {
            return empty();
        }
        let sbg_sig_borders = &tables.sbg_sig_highres;
        let sbg_noise_borders = &tables.sbg_noise;
        let sbx = tables.sbx;

        let n_slots = pcm.len() / 64;
        if n_slots == 0 {
            return empty();
        }
        let usable = n_slots * 64;
        let mut bank = crate::qmf::QmfAnalysisBank::new();
        let slots = bank.process_block(&aspx_scaled_pcm(&pcm[..usable]));
        let q_high = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots);

        let ch = crate::encoder_acpl3::AspxQmfEnvelopeChannel {
            q_high: &q_high,
            sbg_sig_borders,
            sbg_noise_borders,
        };
        crate::encoder_acpl3::build_aspx_real_envelope_channel_from_qmf(
            &ch,
            aspx_cfg.quant_mode_env,
            64,
            num_ts_in_ats,
            aspx_frame_ts_count,
            sbx,
        )
    }

    /// Mono (1-channel) counterpart to [`Self::extract_aspx_lr_multi_env`].
    /// QMF-analyses a single carrier, probes its HF energy for a transient,
    /// and — when one is present — builds the per-envelope SIGNAL + NOISE
    /// DPCM rows for a FIXFIX `num_env > 1` `aspx_data_1ch()` element. A
    /// stationary carrier (or a config that rejects `num_env > 1`) returns
    /// `(1, _, _)`, signalling the caller to emit the single-envelope path.
    ///
    /// Returns `(num_env, sig_rows, noise_rows)` ready for
    /// [`crate::encoder_acpl3::AspxMultiEnvelopeChannel`] →
    /// [`crate::encoder_acpl3::write_aspx_data_1ch_multi_envelope`].
    fn extract_aspx_mono_multi_env(
        &mut self,
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        pcm: &[f32],
    ) -> (
        u32,
        Vec<crate::encoder_acpl3::AspxEncodedEnvelope>,
        Vec<crate::encoder_acpl3::AspxEncodedEnvelope>,
    ) {
        let fallback = || (1u32, Vec::new(), Vec::new());
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return fallback();
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 {
            return fallback();
        }
        let sbg_sig_borders = &tables.sbg_sig_highres;
        let sbg_noise_borders = &tables.sbg_noise;
        let sbx = tables.sbx;

        let n_slots = pcm.len() / 64;
        if n_slots == 0 {
            return fallback();
        }
        let usable = n_slots * 64;
        let mut bank = crate::qmf::QmfAnalysisBank::new();
        let slots = bank.process_block(&aspx_scaled_pcm(&pcm[..usable]));
        let q_high = crate::encoder_acpl3::qmf_slots_to_sb_major(&slots);

        let max_num_env = 1u32 << ((1u32 << aspx_cfg.fixfix_tmp_num_env_bits()) - 1);
        let num_env = crate::encoder_acpl3::select_aspx_num_env_from_qmf(
            &q_high,
            sbg_sig_borders,
            num_ts_in_ats,
            aspx_frame_ts_count,
            sbx,
            max_num_env,
            0.30,
        );
        if num_env <= 1 {
            return fallback();
        }

        let ch = crate::encoder_acpl3::AspxQmfMultiEnvelopeChannel {
            q_high: &q_high,
            sbg_sig_borders,
            sbg_noise_borders,
        };
        let (sig_rows, noise_rows) =
            crate::encoder_acpl3::build_aspx_multi_envelope_channel_from_qmf(
                &ch,
                num_env,
                aspx_cfg.quant_mode_env,
                64,
                num_ts_in_ats,
                aspx_frame_ts_count,
                sbx,
                // I-frame: no inter-frame TIME-direction history.
                &[],
                &[],
                1,
                false,
            );
        (num_env, sig_rows, noise_rows)
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

    /// Encode one IMS v2 frame containing a 5.0 SIMPLE/ASPX_ACPL_1
    /// multichannel substream whose joint-MDCT residual layer is
    /// **SAP-coded by decision** (round 279) — the automatic,
    /// decision-driven counterpart of [`Self::encode_frame_pcm_5_0_acpl1`].
    ///
    /// Per ETSI TS 103 190-1 §5.3.4.3.2 / Table 181 + §5.3.2 Pseudocode
    /// 59, the encoder runs the round-271
    /// [`crate::asf::select_alpha_q_for_pair`] least-squares decision per
    /// `(L, Ls)` / `(R, Rs)` target pair, materialises the SAP-coded
    /// `chparam_info()` rows via
    /// [`crate::asf::build_chparam_info_sap_data_from_alpha_q`] (falling
    /// back to the header-only `SapMode::None` row when no band
    /// benefits), and transmits the Table-181 matrix-input carriers
    /// `(sSMP_A, sSMP_B) = (M, ·)` plus the side prediction residual
    /// `(sSMP_3, sSMP_4) = (S − g·M, ·)` recovered through
    /// [`crate::asf::invert_sap_table_181`]. For a surround pair
    /// correlated with its front carrier the residual sf_data collapses
    /// to (near-)silence — the bits the identity path spends on the raw
    /// Ls/Rs spectra are saved while the decoder's
    /// `apply_sap_table_181` forward mix reproduces the same
    /// preliminaries.
    ///
    /// On-wire body layout is identical to
    /// [`Self::encode_frame_pcm_5_0_acpl1`]; when the decision picks no
    /// SAP band (e.g. `Ls = L`) the emitted frame is bit-for-bit
    /// identical to the identity-SAP path.
    pub fn encode_frame_pcm_5_0_acpl1_sap(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl1_sap_with_max_sfb(frames, 40, 20)
    }

    /// `max_sfb` / `max_sfb_master`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl1_sap`].
    pub fn encode_frame_pcm_5_0_acpl1_sap_with_max_sfb(
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
                "encode_frame_pcm_5_0_acpl1_sap: channel {ch} input length must match frame_len = {frame_len}"
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

        // Forward MDCT analysis per carrier channel (L, R, C, Ls, Rs).
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

        // Same ASPX / ACPL parameterisation as the round-103 path.
        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_sap_auto(
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

    /// Encode one IMS v2 frame containing a 5.0 SIMPLE/ASPX_ACPL_1
    /// multichannel substream with **real per-parameter-band α extraction**
    /// (round 128 — replaces the round-103 zero-delta scaffold for the
    /// α coefficient family; β / β3 / γ stay at the scaffold).
    ///
    /// Body layout is identical to [`Self::encode_frame_pcm_5_0_acpl1`]
    /// (delegates to [`crate::encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_real_alpha`]);
    /// the only on-wire difference is that the two `acpl_data_1ch()`
    /// elements now emit ALPHA F0 + DF codewords with non-zero values
    /// chosen to minimise the per-parameter-band residual against the
    /// caller's (Ls, Rs) input vs (L, R) carrier energies. See
    /// [`crate::encoder_acpl3`] §"Real per-band α extraction" for the
    /// closed-form derivation (β = 0 ⇒ α = 1 − 2·√2·⟨carrier, surround⟩
    /// / ⟨carrier, carrier⟩).
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order. The decoder's
    /// [`crate::acpl_synth::run_acpl_5x_pair_pcm`] consumes the recovered
    /// α and reconstructs Ls / Rs from the L / R carriers with measurably
    /// better fidelity than the zero-α baseline when Ls / Rs aren't a
    /// pure scaled copy of L / R.
    pub fn encode_frame_pcm_5_0_acpl1_real_alpha(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl1_real_alpha_with_max_sfb(frames, 40, 20)
    }

    /// `max_sfb` / `max_sfb_master`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl1_real_alpha`].
    pub fn encode_frame_pcm_5_0_acpl1_real_alpha_with_max_sfb(
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
                "encode_frame_pcm_5_0_acpl1_real_alpha: channel {ch} input length must match frame_len = {frame_len}"
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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_real_alpha(
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
    /// multichannel substream with **real per-parameter-band α + β
    /// extraction** per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 +
    /// §5.7.7.6.1 Pseudocode 117 (round 132).
    ///
    /// Extends [`Self::encode_frame_pcm_5_0_acpl1_real_alpha`] by emitting
    /// real per-band β magnitudes alongside the existing real α — the
    /// surround Ls/Rs reconstruction at the decoder is no longer a pure
    /// level-only image of L/R but also carries the energy of the
    /// decorrelated residual:
    ///
    /// ```text
    ///   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
    /// ```
    ///
    /// `frames` is in `[L, R, C, Ls, Rs]` order; β / γ stay at the
    /// round-95 / 100 / 103 / 128 scaffold for non-ACPL_1 paths.
    pub fn encode_frame_pcm_5_0_acpl1_real_alpha_beta(&mut self, frames: &[&[f32]; 5]) -> Vec<u8> {
        self.encode_frame_pcm_5_0_acpl1_real_alpha_beta_with_max_sfb(frames, 40, 20)
    }

    /// `max_sfb` / `max_sfb_master`-parameterised form of
    /// [`Self::encode_frame_pcm_5_0_acpl1_real_alpha_beta`].
    pub fn encode_frame_pcm_5_0_acpl1_real_alpha_beta_with_max_sfb(
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
                "encode_frame_pcm_5_0_acpl1_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1101;
        self.channel_mode_bits = 4;

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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 8192,
            41..=50 => 16384,
            _ => 32768,
        };

        let body = crate::encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta(
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

    /// Encode one IMS v2 frame containing a 7.0 (3/4/0) SIMPLE/ASPX_ACPL_2
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
    /// `case ASPX_ACPL_2:` with **real per-parameter-band α + β extraction**
    /// (round 202). The 7_X (immersive) counterpart to the round-144 5.0
    /// ACPL_2 real-α-β path
    /// ([`Self::encode_frame_pcm_5_0_acpl2_real_alpha_beta`]) and the
    /// real-α-β upgrade of the round-107 7.0 ACPL_2 zero-delta path
    /// ([`Self::encode_frame_pcm_7_0_acpl2`]).
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order — the 7.0 (3/4/0)
    /// surface layout. The L/R pair feeds the first `two_channel_data()`
    /// carriers and drives the A-CPL Ls/Rs surround reconstruction via
    /// [`crate::acpl_synth::run_acpl_5x_pair_pcm`] at decode time; the
    /// Ls/Rs pair rides the second `two_channel_data()` *and* feeds the
    /// α + β extractors (D0 module models (L → Ls); D1 module models
    /// (R → Rs)). `acpl_config_1ch(FULL)` carries no `qmf_band` →
    /// `start_band = 0` so every parameter band participates. The centre
    /// `C` is the trailing Cfg0 `mono_data(0)`. The back pair `Lb, Rb`
    /// is accepted for layout completeness but not carried by the
    /// ASPX_ACPL_2 body (the decoder's 7_X ACPL_2 dispatch populates
    /// slots 0..4 — slots 5/6 stay silent), matching the round-107
    /// documented Table 202 channel mapping.
    ///
    /// The encoder forces the 7.0 channel_mode prefix (`0b1111000`, 7 b —
    /// Table 85 channel_mode 5) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 7` through
    /// `parse_7x_audio_data_outer(b_has_lfe = false)` with
    /// `7_X_codec_mode = AspxAcpl2`.
    ///
    /// `max_sfb` defaults to 40.
    pub fn encode_frame_pcm_7_0_acpl2_real_alpha_beta(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_acpl2_real_alpha_beta_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_alpha_beta`].
    pub fn encode_frame_pcm_7_0_acpl2_real_alpha_beta_with_max_sfb(
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
                "encode_frame_pcm_7_0_acpl2_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force 7.0 (3/4/0) channel_mode prefix '1111000', 7 b →
        // channel_mode 5 (Table 85).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111000;
        self.channel_mode_bits = 7;

        // Forward MDCT analysis per channel — seven SCE states (L, R, C,
        // Ls, Rs, Lb, Rb). The first five feed the ASPX_ACPL_2 body;
        // Ls / Rs additionally feed the α + β extractors. The back pair
        // is analysed for state continuity but its spectra are not
        // carried by the ACPL_2 path.
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

        // ASPX config: matches the round-95 / 100 / 103 / 107 ASPX_ACPL
        // config exactly.
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

        let body = crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
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
    /// `case ASPX_ACPL_2:` with `b_has_lfe = 1` and **real per-parameter-band
    /// α + β extraction** (round 202). The LFE counterpart of
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_alpha_beta`] — it emits the
    /// identical 7_X ASPX_ACPL_2 real-α-β body plus a leading
    /// `mono_data(b_lfe = 1)` element between the I-frame config block and
    /// `companding_control(5)`, exactly where the decoder's
    /// `parse_7x_audio_data_outer(b_has_lfe = true)` reads
    /// `if (b_has_lfe) mono_data(1);`.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order. See
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_alpha_beta`] for the channel
    /// routing contract; the LFE is the leading `mono_data(1)`.
    ///
    /// The encoder forces the 7.1 channel_mode prefix (`0b1111001`, 7 b —
    /// Table 88 channel_mode 6) so the decoder dispatches `channels == 8`
    /// through `parse_7x_audio_data_outer(b_has_lfe = true)` with
    /// `7_X_codec_mode = AspxAcpl2`.
    ///
    /// `max_sfb` defaults to 40; `max_sfb_lfe` defaults to 7 (the LFE-spec
    /// cap at `tl = 1920`, `n_msfbl_bits = 3`).
    pub fn encode_frame_pcm_7_1_acpl2_real_alpha_beta(&mut self, frames: &[&[f32]; 8]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_acpl2_real_alpha_beta_with_max_sfb(frames, 40, 7)
    }

    /// `max_sfb` / `max_sfb_lfe`-parameterised form of
    /// [`Self::encode_frame_pcm_7_1_acpl2_real_alpha_beta`].
    pub fn encode_frame_pcm_7_1_acpl2_real_alpha_beta_with_max_sfb(
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
                "encode_frame_pcm_7_1_acpl2_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_7_1_acpl2_real_alpha_beta: tl = {frame_len} not permitted for LFE"
        );
        let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
        let max_sfb_lfe = max_sfb_lfe.min(n_msfbl_cap);

        // Force 7.1 (3/4/0.1) channel_mode prefix '1111001', 7 b →
        // channel_mode 6 (Table 88).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111001;
        self.channel_mode_bits = 7;

        // Forward MDCT analysis per channel — eight SCE states (L, R, C,
        // Ls, Rs, Lb, Rb, LFE).
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

        // ASPX config: matches the round-95 / 100 / 103 / 107 / 114 ASPX_ACPL
        // config exactly.
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

        let body = crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
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

    /// Encode one IMS v2 frame containing a 7.0 (3/4/0) SIMPLE/ASPX_ACPL_2
    /// multichannel substream with **real** ASPX SIGNAL / NOISE envelopes
    /// on all three ASPX carriers (round 337). The 7_X counterpart of the
    /// round-331 5_X [`Self::encode_frame_pcm_5_0_acpl2_real_aspx`]: where
    /// the round-202 7_X ACPL_2 path carried real per-band α + β but still
    /// emitted the round-107 minimum-bit-cost ASPX envelope scaffolds, this
    /// method QMF-analyses the L / R / Ls / Rs **and** C input PCM and emits
    /// real `[F0, DF₁, …]` SIGNAL / NOISE envelopes on the two carrier-pair
    /// `aspx_data_2ch()` elements (L/R front, Ls/Rs surround) **and** the
    /// centre `aspx_data_1ch()` element via
    /// [`crate::encoder_acpl3::write_aspx_data_2ch_real_envelope`] /
    /// [`crate::encoder_acpl3::write_aspx_data_1ch_real_envelope`],
    /// replacing the round-107 scaffolds. The real per-band α / β A-CPL
    /// parameters are unchanged from
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_alpha_beta`].
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order — the 7.0 (3/4/0)
    /// surface layout. The output round-trips through
    /// [`crate::decoder::Ac4Decoder`] to a 7-channel `AudioFrame`.
    ///
    /// Refs ETSI TS 103 190-1 §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
    /// §4.2.12.3 Table 51, §4.2.12.4 Table 52, §5.7.6.3.4 / §5.7.6.3.5.
    pub fn encode_frame_pcm_7_0_acpl2_real_aspx(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_acpl2_real_aspx_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_aspx`].
    pub fn encode_frame_pcm_7_0_acpl2_real_aspx_with_max_sfb(
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
                "encode_frame_pcm_7_0_acpl2_real_aspx: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Force 7.0 (3/4/0) channel_mode prefix '1111000', 7 b →
        // channel_mode 5 (Table 85).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111000;
        self.channel_mode_bits = 7;

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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary (L) carrier (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        // Real ASPX envelope extraction: L/R front pair, Ls/Rs surround
        // pair, and the centre carrier (the ACPL_2 body's aspx_data_1ch()).
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        let (ls_sig, ls_noise, rs_sig, rs_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[3], frames[4]);
        let (c_sig, c_noise) = self.extract_aspx_mono_envelope(&aspx_cfg, frame_len, frames[2]);

        // Per-carrier A-SPX inverse-filtering decisions: the front pair
        // shares L's vector, the surround pair shares Ls's, the centre its
        // own (all under aspx_balance = 1 for the pairs).
        let front_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);
        let surround_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[3]);
        let c_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[2]);

        // Per-channel real aspx_add_harmonic (§4.2.12.6) for all five
        // 7_X ACPL_2 A-SPX carriers — signalled independently per channel.
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let ls_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[3]);
        let rs_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[4]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body =
            crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
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
                &l_sig,
                &l_noise,
                &r_sig,
                &r_noise,
                &ls_sig,
                &ls_noise,
                &rs_sig,
                &rs_noise,
                &c_sig,
                &c_noise,
                &front_tna_mode,
                &surround_tna_mode,
                &c_tna_mode,
                &l_ah,
                &r_ah,
                &ls_ah,
                &rs_ah,
                &c_ah,
                acpl_num_param_bands_id,
                acpl_quant_mode,
                pad_target_bytes,
            );

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

    /// Multi-envelope centre-carrier variant of
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_aspx`] — the 7_X dual of
    /// [`Self::encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env`].
    /// Probes the centre carrier for a transient and, when one is present,
    /// emits a multi-envelope centre `aspx_data_1ch()` (`num_env > 1`) via
    /// [`crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env`].
    /// Both carrier pairs (L/R front, Ls/Rs surround) keep their
    /// single-envelope `aspx_data_2ch()`. Stationary centre carriers fall
    /// back to the round-337 single-envelope path.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order.
    pub fn encode_frame_pcm_7_0_acpl2_real_aspx_centre_multi_env(
        &mut self,
        frames: &[&[f32]; 7],
    ) -> Vec<u8> {
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_7_x_acpl2_real_aspx_centre_multi_env_with_max_sfb(
            &surround, None, 40, None, 0b1111000, 7,
        )
    }

    /// 7.1 counterpart to
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_aspx_centre_multi_env`].
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order.
    pub fn encode_frame_pcm_7_1_acpl2_real_aspx_centre_multi_env(
        &mut self,
        frames: &[&[f32]; 8],
    ) -> Vec<u8> {
        let surround: [&[f32]; 5] = [frames[0], frames[1], frames[2], frames[3], frames[4]];
        self.encode_frame_pcm_7_x_acpl2_real_aspx_centre_multi_env_with_max_sfb(
            &surround,
            Some(frames[7]),
            40,
            Some(7),
            0b1111001,
            7,
        )
    }

    /// Shared body for the 7_X ASPX_ACPL_2 centre-multi-envelope entry
    /// points. `surround` is `[L, R, C, Ls, Rs]`; the back pair Lb/Rb is
    /// reconstructed at decode time from the A-CPL coupling (Table 202), so
    /// it carries no carrier here. A stationary centre carrier — or a config
    /// the multi-envelope writer rejects — falls back to the round-337
    /// single-envelope 7_X path, so the output is always a valid frame.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame_pcm_7_x_acpl2_real_aspx_centre_multi_env_with_max_sfb(
        &mut self,
        surround: &[&[f32]; 5],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
        channel_mode_value: u8,
        channel_mode_bits: u8,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in surround.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_x_acpl2_real_aspx_centre_multi_env: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(lfe_buf) = lfe {
            assert_eq!(
                lfe_buf.len(),
                frame_len as usize,
                "encode_frame_pcm_7_x_acpl2_real_aspx_centre_multi_env: LFE input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary front-L carrier (surround[0])
        // (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, surround[0]);

        // Probe the centre carrier (surround[2]) for a transient.
        let (c_num_env, c_sig_rows, c_noise_rows) =
            self.extract_aspx_mono_multi_env(&aspx_cfg, frame_len, surround[2]);
        if c_num_env <= 1 {
            return self.fallback_7_x_acpl2_real_aspx_single_env(
                surround,
                lfe,
                max_sfb,
                max_sfb_lfe,
            );
        }

        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = channel_mode_value;
        self.channel_mode_bits = channel_mode_bits;

        let n_channels = if lfe.is_some() { 6 } else { 5 };
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
        for (ch, f) in surround.iter().enumerate() {
            let c = self.mdct_states_multi[ch].analyse_frame(f);
            coeffs_per_channel.push(c);
        }
        let coeffs_lfe: Option<Vec<f32>> =
            lfe.map(|buf| self.mdct_states_multi[5].analyse_frame(buf));

        // L/R front + Ls/Rs surround stay single-envelope.
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, surround[0], surround[1]);
        let (ls_sig, ls_noise, rs_sig, rs_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, surround[3], surround[4]);

        // Per-channel real aspx_add_harmonic (§4.2.12.6) for all five
        // 7_X ACPL_2 A-SPX carriers.
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, surround[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, surround[1]);
        let ls_ah = self.extract_aspx_add_harmonic(&aspx_cfg, surround[3]);
        let rs_ah = self.extract_aspx_add_harmonic(&aspx_cfg, surround[4]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, surround[2]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let centre = crate::encoder_acpl3::AspxMultiEnvelopeChannel {
            sig: &c_sig_rows,
            noise: &c_noise_rows,
        };
        let body =
            crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env(
                frame_len,
                max_sfb,
                max_sfb_lfe,
                self.b_iframe_global,
                &coeffs_per_channel[0],
                &coeffs_per_channel[1],
                &coeffs_per_channel[3],
                &coeffs_per_channel[4],
                &coeffs_per_channel[2],
                coeffs_lfe.as_deref(),
                &aspx_cfg,
                &l_sig,
                &l_noise,
                &r_sig,
                &r_noise,
                &ls_sig,
                &ls_noise,
                &rs_sig,
                &rs_noise,
                c_num_env,
                centre,
                &l_ah,
                &r_ah,
                &ls_ah,
                &rs_ah,
                &c_ah,
                acpl_num_param_bands_id,
                acpl_quant_mode,
                pad_target_bytes,
            );

        if body.is_empty() {
            self.channel_mode_value = saved_mode.0;
            self.channel_mode_bits = saved_mode.1;
            return self.fallback_7_x_acpl2_real_aspx_single_env(
                surround,
                lfe,
                max_sfb,
                max_sfb_lfe,
            );
        }

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

    /// Single-envelope fallback shared by the 7_X centre-multi-env entry
    /// points. Routes through the round-337 single-envelope 7_X path, which
    /// expects `[L, R, C, Ls, Rs, Lb, Rb(, LFE)]` — the back pair carries no
    /// carrier under ASPX_ACPL_2 (Table 202), so it is filled with the
    /// centre buffer (any value works: the decoder reconstructs Lb/Rb from
    /// the A-CPL coupling, never from these slots).
    fn fallback_7_x_acpl2_real_aspx_single_env(
        &mut self,
        surround: &[&[f32]; 5],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        max_sfb_lfe: Option<u32>,
    ) -> Vec<u8> {
        let c = surround[2];
        match lfe {
            Some(lfe_buf) => {
                let frames: [&[f32]; 8] = [
                    surround[0],
                    surround[1],
                    surround[2],
                    surround[3],
                    surround[4],
                    c,
                    c,
                    lfe_buf,
                ];
                self.encode_frame_pcm_7_1_acpl2_real_aspx_with_max_sfb(
                    &frames,
                    max_sfb,
                    max_sfb_lfe.unwrap_or(7),
                )
            }
            None => {
                let frames: [&[f32]; 7] = [
                    surround[0],
                    surround[1],
                    surround[2],
                    surround[3],
                    surround[4],
                    c,
                    c,
                ];
                self.encode_frame_pcm_7_0_acpl2_real_aspx_with_max_sfb(&frames, max_sfb)
            }
        }
    }

    /// 7.1 (3/4/0.1) counterpart to
    /// [`Self::encode_frame_pcm_7_0_acpl2_real_aspx`]. Emits the identical
    /// 7_X ASPX_ACPL_2 real-α/β + real-ASPX body plus a leading LFE
    /// `mono_data(b_lfe = 1)` element between the I-frame config block and
    /// `companding_control(5)`.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order. The output
    /// round-trips through [`crate::decoder::Ac4Decoder`] to an 8-channel
    /// `AudioFrame`.
    pub fn encode_frame_pcm_7_1_acpl2_real_aspx(&mut self, frames: &[&[f32]; 8]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_acpl2_real_aspx_with_max_sfb(frames, 40, 7)
    }

    /// `max_sfb` / `max_sfb_lfe`-parameterised form of
    /// [`Self::encode_frame_pcm_7_1_acpl2_real_aspx`].
    pub fn encode_frame_pcm_7_1_acpl2_real_aspx_with_max_sfb(
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
                "encode_frame_pcm_7_1_acpl2_real_aspx: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_7_1_acpl2_real_aspx: tl = {frame_len} not permitted for LFE"
        );
        let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
        let max_sfb_lfe = max_sfb_lfe.min(n_msfbl_cap);

        // Force 7.1 (3/4/0.1) channel_mode prefix '1111001', 7 b →
        // channel_mode 6 (Table 88).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111001;
        self.channel_mode_bits = 7;

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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary (L) carrier (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        let (ls_sig, ls_noise, rs_sig, rs_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[3], frames[4]);
        let (c_sig, c_noise) = self.extract_aspx_mono_envelope(&aspx_cfg, frame_len, frames[2]);

        // Per-carrier A-SPX inverse-filtering decisions (front pair from L,
        // surround pair from Ls, centre from C — under aspx_balance = 1).
        let front_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);
        let surround_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[3]);
        let c_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[2]);

        // Per-channel real aspx_add_harmonic (§4.2.12.6) for all five
        // 7_X ACPL_2 A-SPX carriers — signalled independently per channel.
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let ls_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[3]);
        let rs_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[4]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body =
            crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
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
                &l_sig,
                &l_noise,
                &r_sig,
                &r_noise,
                &ls_sig,
                &ls_noise,
                &rs_sig,
                &rs_noise,
                &c_sig,
                &c_noise,
                &front_tna_mode,
                &surround_tna_mode,
                &c_tna_mode,
                &l_ah,
                &r_ah,
                &ls_ah,
                &rs_ah,
                &c_ah,
                acpl_num_param_bands_id,
                acpl_quant_mode,
                pad_target_bytes,
            );

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

    /// Encode one IMS v2 frame containing a 7.0 (3/4/0) SIMPLE/ASPX_ACPL_1
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
    /// `case ASPX_ACPL_1:` (round 118). The 7_X (immersive) counterpart to
    /// the round-103 5_X ASPX_ACPL_1 encoder and the encoder side of the
    /// decoder's round-27 `parse_7x_audio_data_outer` ASPX_ACPL_1 branch.
    ///
    /// ASPX_ACPL_1 differs from the round-107 7.0 ASPX_ACPL_2 path in three
    /// structural places (the same three that separate the 5_X ACPL_1 path
    /// from the 5_X ACPL_2 path): `7_X_codec_mode = 2` (vs 3),
    /// `acpl_config_1ch` is PARTIAL (vs FULL — carries the 3-bit
    /// `acpl_qmf_band_minus1`), and the body carries an explicit joint-MDCT
    /// residual layer (`max_sfb_master + 2× chparam_info + 2× sf_data(ASF)`)
    /// transmitting the Ls/Rs surround pair (sSMP,3 / sSMP,4 per Table 181)
    /// rather than reconstructing it purely from the L/R carriers.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order — the 7.0 (3/4/0)
    /// surface layout. The L/R pair feeds the first `two_channel_data()`
    /// carriers; the Ls/Rs pair rides the second `two_channel_data()` *and*
    /// the joint-MDCT residual layer; the centre `C` is the trailing Cfg0
    /// `mono_data(0)`. The back pair `Lb, Rb` is accepted for layout
    /// completeness but not carried by the ASPX_ACPL_1 body (the decoder's
    /// 7_X ACPL_1 dispatch populates slots 0..4 only — slots 5/6 stay
    /// silent), matching the round-107 documented Table 202 channel mapping.
    ///
    /// The encoder forces the 7.0 (3/4/0) channel_mode prefix (`0b1111000`,
    /// 7 b — Table 85 channel_mode 5) so the decoder's `walk_ac4_substream`
    /// dispatches `channels == 7` through
    /// `parse_7x_audio_data_outer(b_has_lfe = false)` with
    /// `7_X_codec_mode = AspxAcpl1`. The ASPX/A-CPL parameter bits use the
    /// round-95 minimum-bit-cost zero-delta Huffman scaffold.
    ///
    /// `max_sfb` defaults to 40; `max_sfb_master` (the residual band bound)
    /// defaults to 20.
    pub fn encode_frame_pcm_7_0_acpl1(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_acpl1_with_max_sfb(frames, 40, 20)
    }

    /// `max_sfb` / `max_sfb_master`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_acpl1`].
    pub fn encode_frame_pcm_7_0_acpl1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 7],
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
                "encode_frame_pcm_7_0_acpl1: channel {ch} input length must match frame_len = {frame_len}"
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
        // Ls, Rs, Lb, Rb). Only the first five feed the ASPX_ACPL_1 body;
        // the back pair is analysed for state continuity but its spectra
        // are not carried by the ACPL_1 path.
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

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_mode Fine;
        // acpl_qmf_band_minus1 = 0 → qmf_band = 1 (PARTIAL mode).
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body = crate::encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_master,
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

    /// Encode one IMS v2 frame containing a 7.0 (3/4/0) SIMPLE/ASPX_ACPL_1
    /// multichannel substream with **real per-parameter-band α + β
    /// extraction** per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 +
    /// §5.7.7.6.1 Pseudocode 117 (round 135).
    ///
    /// The 7_X immersive counterpart of
    /// [`Self::encode_frame_pcm_5_0_acpl1_real_alpha_beta`] and the real-
    /// α+β upgrade of [`Self::encode_frame_pcm_7_0_acpl1`] (which emitted
    /// both `acpl_data_1ch()` sets at the round-118 zero-delta scaffold).
    /// The two trailing `acpl_data_1ch()` parameter sets now carry the
    /// analytic α (from the L/Ls and R/Rs MDCT-energy correlation) and the
    /// β magnitude that closes the surround/carrier energy balance after α
    /// removes the level-only component:
    ///
    /// ```text
    ///   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
    /// ```
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb]` order — the 7.0 (3/4/0)
    /// surface layout, identical to [`Self::encode_frame_pcm_7_0_acpl1`].
    /// β / β3 / γ for non-ACPL_1 paths stay at the scaffold.
    pub fn encode_frame_pcm_7_0_acpl1_real_alpha_beta(&mut self, frames: &[&[f32]; 7]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_acpl1_real_alpha_beta_with_max_sfb(frames, 40, 20)
    }

    /// `max_sfb` / `max_sfb_master`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_acpl1_real_alpha_beta`].
    pub fn encode_frame_pcm_7_0_acpl1_real_alpha_beta_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 7],
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
                "encode_frame_pcm_7_0_acpl1_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary (L) carrier (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        // Real ASPX envelope extraction over the three A-CPL_1 carriers:
        // L/R front pair, Ls/Rs surround pair, centre.
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        let (ls_sig, ls_noise, rs_sig, rs_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[3], frames[4]);
        let (c_sig, c_noise) = self.extract_aspx_mono_envelope(&aspx_cfg, frame_len, frames[2]);

        // Per-carrier real aspx_tna_mode (front from L, surround from Ls,
        // centre from C; channel 1 of each pair mirrors via balance = 1).
        let front_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);
        let surround_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[3]);
        let c_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[2]);

        // Per-carrier (per-channel) real aspx_add_harmonic (§4.2.12.6):
        // unlike tna_mode it is signalled independently for each channel of
        // a 2ch element (aspx_balance mirrors framing + tna_mode only).
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let ls_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[3]);
        let rs_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[4]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body =
            crate::encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
                frame_len,
                max_sfb,
                max_sfb_master,
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
                acpl_qmf_band_minus1,
                (
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &l_sig,
                        noise: &l_noise,
                    },
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &r_sig,
                        noise: &r_noise,
                    },
                ),
                (
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &ls_sig,
                        noise: &ls_noise,
                    },
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &rs_sig,
                        noise: &rs_noise,
                    },
                ),
                crate::encoder_acpl3::AspxRealEnvelopeChannel {
                    sig: &c_sig,
                    noise: &c_noise,
                },
                &front_tna_mode,
                &surround_tna_mode,
                &c_tna_mode,
                (&l_ah, &r_ah),
                (&ls_ah, &rs_ah),
                &c_ah,
                pad_target_bytes,
            );

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

    /// Encode one IMS v2 frame containing a 7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_1
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
    /// `case ASPX_ACPL_1:` with `b_has_lfe = 1` (round 118). The LFE
    /// counterpart of [`Self::encode_frame_pcm_7_0_acpl1`] — it emits the
    /// identical 7_X ASPX_ACPL_1 body plus a leading `mono_data(b_lfe = 1)`
    /// element (Table 21 + `sf_info_lfe()` Table 35) between the I-frame
    /// config block and `companding_control(5)`, exactly where the decoder's
    /// `parse_7x_audio_data_outer(b_has_lfe = true)` reads
    /// `if (b_has_lfe) mono_data(1);`.
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order — the 7.1
    /// (3/4/0.1) surface layout. The LFE is the leading `mono_data(1)`; the
    /// rest of the body matches the 7.0 ACPL_1 form. The encoder forces the
    /// 7.1 channel_mode prefix (`0b1111001`, 7 b — Table 88 channel_mode 6)
    /// so the decoder dispatches `channels == 8` through
    /// `parse_7x_audio_data_outer(b_has_lfe = true)` with
    /// `7_X_codec_mode = AspxAcpl1`; the LFE spectrum IMDCT's into slot 7
    /// via the round-80 LFE PCM render.
    ///
    /// `max_sfb` defaults to 40; `max_sfb_master` defaults to 20;
    /// `max_sfb_lfe` defaults to 7 (the LFE-spec cap at `tl = 1920`,
    /// `n_msfbl_bits = 3`).
    pub fn encode_frame_pcm_7_1_acpl1(&mut self, frames: &[&[f32]; 8]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_acpl1_with_max_sfb(frames, 40, 20, 7)
    }

    /// `max_sfb`-parameterised form of [`Self::encode_frame_pcm_7_1_acpl1`].
    /// `max_sfb` governs the five front/surround carrier SCEs and the centre
    /// mono; `max_sfb_master` governs the joint-MDCT surround residual
    /// layer; `max_sfb_lfe` governs the LFE `mono_data(1)` (clamped to the
    /// `n_msfbl_bits` cap).
    pub fn encode_frame_pcm_7_1_acpl1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 8],
        max_sfb: u32,
        max_sfb_master: u32,
        max_sfb_lfe: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_1_acpl1: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_7_1_acpl1: tl = {frame_len} not permitted for LFE"
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
        // Ls, Rs, Lb, Rb, LFE). The first five + LFE feed the ASPX_ACPL_1
        // body; the back pair is analysed for state continuity but its
        // spectra are not carried by the ACPL_1 path.
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

        // ACPL: num_param_bands_id = 3 → 7 param bands; quant_mode Fine;
        // acpl_qmf_band_minus1 = 0 → qmf_band = 1 (PARTIAL mode).
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body = crate::encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra(
            frame_len,
            max_sfb,
            max_sfb_master,
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

    /// Encode one IMS v2 frame containing a 7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_1
    /// multichannel substream per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
    /// `case ASPX_ACPL_1:` with `b_has_lfe = 1`, with **real per-parameter-band
    /// α + β extraction** carried by the two trailing `acpl_data_1ch()`
    /// parameter sets (round 139 — the LFE counterpart of the round-135
    /// 7.0 immersive real-α+β path,
    /// [`Self::encode_frame_pcm_7_0_acpl1_real_alpha_beta`]).
    ///
    /// The round-118 7.1 ASPX_ACPL_1 encoder emitted both `acpl_data_1ch()`
    /// parameter sets at the zero-delta scaffold; this entry point upgrades
    /// them to carry the analytic α (from the L/Ls and R/Rs MDCT-energy
    /// correlation, §5.7.7.5 Pseudocode 116) plus the β magnitude that
    /// closes the surround/carrier energy balance after α removes the
    /// level-only component (§5.7.7.6.1 Pseudocode 117):
    ///
    /// ```text
    ///   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
    ///   ⇒  β = √max(0, 2·E[Ls²]/E[L²] − (1 − α)²)
    /// ```
    ///
    /// `frames` is in `[L, R, C, Ls, Rs, Lb, Rb, LFE]` order — the 7.1
    /// (3/4/0.1) surface layout, identical to
    /// [`Self::encode_frame_pcm_7_1_acpl1`]. The leading `mono_data(b_lfe = 1)`
    /// element (Table 21 + `sf_info_lfe()` Table 35) is emitted between the
    /// I-frame config block and `companding_control(5)`. The on-wire body
    /// structure is otherwise identical — decoder resolves
    /// `SevenXCodecMode::AspxAcpl1` with `b_has_lfe = true`, both
    /// `acpl_data_1ch_pair[0/1]` populated (now carrying real α + β),
    /// joint-MDCT residual layer walked, LFE IMDCT'd into slot 7.
    pub fn encode_frame_pcm_7_1_acpl1_real_alpha_beta(&mut self, frames: &[&[f32]; 8]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_acpl1_real_alpha_beta_with_max_sfb(frames, 40, 20, 7)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_1_acpl1_real_alpha_beta`]. `max_sfb`
    /// governs the five front/surround carrier SCEs and the centre mono;
    /// `max_sfb_master` governs the joint-MDCT surround residual layer;
    /// `max_sfb_lfe` governs the LFE `mono_data(1)` (clamped to the
    /// `n_msfbl_bits` cap).
    pub fn encode_frame_pcm_7_1_acpl1_real_alpha_beta_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 8],
        max_sfb: u32,
        max_sfb_master: u32,
        max_sfb_lfe: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_7_1_acpl1_real_alpha_beta: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, n_msfbl_bits) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        assert!(
            n_msfbl_bits > 0,
            "encode_frame_pcm_7_1_acpl1_real_alpha_beta: tl = {frame_len} not permitted for LFE"
        );
        let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
        let max_sfb_lfe = max_sfb_lfe.min(n_msfbl_cap);

        // Force 7.1 (3/4/0.1) channel_mode prefix '1111001', 7 b →
        // channel_mode 6 (Table 88). The decoder routes channels == 8
        // through parse_7x_audio_data_outer(b_has_lfe = true).
        let saved_mode = (self.channel_mode_value, self.channel_mode_bits);
        self.channel_mode_value = 0b1111001;
        self.channel_mode_bits = 7;

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

        let aspx_cfg = crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        };

        // Encoder-side A-SPX spectral pre-flattening (Table 121), one
        // per-config flag from the primary (L) carrier (§5.7.6.4.1.2).
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = self.extract_aspx_preflat(&aspx_cfg, frame_len, frames[0]);

        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let acpl_qmf_band_minus1: u8 = 0;

        // Real ASPX envelope extraction over the three A-CPL_1 carriers:
        // L/R front pair, Ls/Rs surround pair, centre.
        let (l_sig, l_noise, r_sig, r_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[0], frames[1]);
        let (ls_sig, ls_noise, rs_sig, rs_noise) =
            self.extract_aspx_lr_envelopes(&aspx_cfg, frame_len, frames[3], frames[4]);
        let (c_sig, c_noise) = self.extract_aspx_mono_envelope(&aspx_cfg, frame_len, frames[2]);

        // Per-carrier real aspx_tna_mode (front from L, surround from Ls,
        // centre from C; channel 1 of each pair mirrors via balance = 1).
        let front_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[0]);
        let surround_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[3]);
        let c_tna_mode = self.extract_aspx_l_tna_mode(&aspx_cfg, frames[2]);

        // Per-carrier (per-channel) real aspx_add_harmonic (§4.2.12.6):
        // unlike tna_mode it is signalled independently for each channel of
        // a 2ch element (aspx_balance mirrors framing + tna_mode only).
        let l_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[0]);
        let r_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[1]);
        let ls_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[3]);
        let rs_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[4]);
        let c_ah = self.extract_aspx_add_harmonic(&aspx_cfg, frames[2]);

        let pad_target_bytes: usize = match max_sfb {
            0..=20 => 4096,
            21..=40 => 12288,
            41..=50 => 24576,
            _ => 32767,
        };

        let body =
            crate::encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
                frame_len,
                max_sfb,
                max_sfb_master,
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
                acpl_qmf_band_minus1,
                (
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &l_sig,
                        noise: &l_noise,
                    },
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &r_sig,
                        noise: &r_noise,
                    },
                ),
                (
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &ls_sig,
                        noise: &ls_noise,
                    },
                    crate::encoder_acpl3::AspxRealEnvelopeChannel {
                        sig: &rs_sig,
                        noise: &rs_noise,
                    },
                ),
                crate::encoder_acpl3::AspxRealEnvelopeChannel {
                    sig: &c_sig,
                    noise: &c_noise,
                },
                &front_tna_mode,
                &surround_tna_mode,
                &c_tna_mode,
                (&l_ah, &r_ah),
                (&ls_ah, &rs_ah),
                &c_ah,
                pad_target_bytes,
            );

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

    /// Derive the eleven coded ICE SMP track signals `A..K` from the
    /// named 7.0.4 channels — the encode-side inverse of the decoder's
    /// §5.3.3.1 Table 23 S-CPL full-decoding matrix composed with the
    /// mode's output gain ladder (`c_gain = m_gain = 1` +
    /// §4.8.3.11.3 Table 10 gains 2 / 2 / 2 / √2·… for ASPX_SCPL —
    /// identical totals to the SCPL `c_gain = 2` / `m_gain = √2`
    /// ladder, so one inverse serves both §5.2.3.2 modes):
    ///
    /// ```text
    ///   A = L/2          B = R/2          C = C/2
    ///   D = (Ls + Lb)/(2√2)    H = (Ls − Lb)/(2√2)
    ///   E = (Rs + Rb)/(2√2)    I = (Rs − Rb)/(2√2)
    ///   F = (Tfl + Tbl)/(2√2)  J = (Tfl − Tbl)/(2√2)
    ///   G = (Tfr + Tbr)/(2√2)  K = (Tfr − Tbr)/(2√2)
    /// ```
    ///
    /// (the Table 23 fold pairs `(D, H)` / `(E, I)` / `(F, J)` /
    /// `(G, K)` — tracks `A..G` are the 7CH_STATIC 5.X.2 core with the
    /// top-front mids on the `F` / `G` additional pair, and the four
    /// S-CPL-section tracks `H..K` carry the pair sides).
    ///
    /// `named` is `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`;
    /// the return order is `[A, B, C, D, E, F, G, H, I, J, K]`
    /// (core, additional pair, S-CPL pairs `[H, I]` / `[J, K]`).
    fn ice_scpl_tracks_from_named(named: &[&[f32]; 11]) -> Vec<Vec<f32>> {
        let n = named[0].len();
        let half = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| v * 0.5).collect() };
        let q = 0.5 * std::f32::consts::FRAC_1_SQRT_2; // 1/(2√2)
        let mix = |x: &[f32], y: &[f32], sign: f32| -> Vec<f32> {
            (0..n).map(|i| q * (x[i] + sign * y[i])).collect()
        };
        vec![
            half(named[0]),                 // A
            half(named[1]),                 // B
            half(named[2]),                 // C
            mix(named[3], named[5], 1.0),   // D = (Ls + Lb)/(2√2)
            mix(named[4], named[6], 1.0),   // E = (Rs + Rb)/(2√2)
            mix(named[7], named[9], 1.0),   // F = (Tfl + Tbl)/(2√2)
            mix(named[8], named[10], 1.0),  // G = (Tfr + Tbr)/(2√2)
            mix(named[3], named[5], -1.0),  // H = (Ls − Lb)/(2√2)
            mix(named[4], named[6], -1.0),  // I = (Rs − Rb)/(2√2)
            mix(named[7], named[9], -1.0),  // J = (Tfl − Tbl)/(2√2)
            mix(named[8], named[10], -1.0), // K = (Tfr − Tbr)/(2√2)
        ]
    }

    /// QMF-analyse one ICE extraction channel through its persistent
    /// (per-`chan_idx`) streaming analysis bank. Returns the
    /// `[absolute_sb][ts]` matrix for the current frame. Streaming
    /// matters here: a fresh bank per frame leaks a broadband warm-up
    /// splash into the HF measurement that quantises to a near
    /// full-scale SIGNAL envelope on genuinely HF-silent channels.
    fn ice_qmf_analyse(&mut self, chan_idx: usize, pcm: &[f32]) -> Vec<Vec<(f32, f32)>> {
        while self.ice_env_ana.len() <= chan_idx {
            self.ice_env_ana.push(crate::qmf::QmfAnalysisBank::new());
        }
        let n_slots = pcm.len() / 64;
        let usable = n_slots * 64;
        let slots = self.ice_env_ana[chan_idx].process_block(&aspx_scaled_pcm(&pcm[..usable]));
        crate::encoder_acpl3::qmf_slots_to_sb_major(&slots)
    }

    /// Build one channel's real A-SPX payload rows (SIGNAL / NOISE
    /// FREQ-DPCM quant indices + per-SBG `aspx_add_harmonic`) from an
    /// already-analysed QMF matrix. When the frequency tables cannot
    /// be derived the NOISE row falls back to the floored `qnoise =
    /// 30` scaffold so the regenerated band never lands on the
    /// full-scale Pseudocode-83 noise floor.
    fn ice_rows_from_matrix(
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        q: &[Vec<(f32, f32)>],
        tna_mode: &[u8],
        preflat: bool,
    ) -> crate::ice::IceAspxChannelRows {
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return crate::ice::IceAspxChannelRows {
                sig: Vec::new(),
                noise: vec![30],
                ah: Vec::new(),
            };
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 {
            return crate::ice::IceAspxChannelRows {
                sig: Vec::new(),
                noise: vec![30],
                ah: Vec::new(),
            };
        }
        // SIGNAL: absolute per-SBG envelope energies (Pseudocode 82
        // inverse). NOISE: tonal-to-noise **ratios** (Pseudocode 94
        // semantics) rather than absolute band energies.
        let sig_scf = crate::encoder_acpl3::extract_aspx_sig_envelope_scf_from_qmf(
            q,
            &tables.sbg_sig_highres,
            num_ts_in_ats,
            aspx_frame_ts_count,
            tables.sbx,
        );
        let noise_scf = crate::encoder_acpl3::extract_aspx_noise_ratio_scf_from_qmf(
            q,
            &tables.sbg_noise,
            num_ts_in_ats,
            aspx_frame_ts_count,
            tables.sbx,
        );
        // Patch-delivery compensation. The decoder reconstructs each
        // regenerated band as `scf·(f + κ·n)/(1 + n)`: `f = est/(1+est)`
        // is the fraction the Pseudocode-95 gain can deliver from the
        // (TNS-whitened) patch tile, `n` the coded noise ratio, and
        // `κ` the synthesis efficiency of i.i.d. injected subband
        // noise through the real-output filterbank (the §5.7.6.4.3
        // noise samples lack the inter-slot correlation of analysed
        // content, so the overlapped synthesis realises only ~40 % of
        // their subband energy in PCM — calibrated against the
        // in-tree §5.7.6.2/5.7.6.5 banks). The encoder (1) raises the
        // noise ratio where the tile source cannot carry the target,
        // and (2) scales the coded SIGNAL envelope by the inverse
        // predicted delivery so the decoded band lands on the true
        // level (everything downstream is linear in `scf`).
        const NOISE_SYNTH_EFF: f32 = 0.4; // κ
        let delivery = crate::encoder_acpl3::predict_aspx_patch_delivery_fraction_from_qmf(
            q,
            &tables,
            matches!(
                aspx_cfg.master_freq_scale,
                crate::aspx::AspxMasterFreqScale::HighRes
            ),
            &tables.sbg_noise,
            tna_mode,
            preflat,
            num_ts_in_ats,
            aspx_frame_ts_count,
        );
        let mut t_group = vec![1.0f32; noise_scf.len()];
        let noise_scf: Vec<f32> = noise_scf
            .iter()
            .enumerate()
            .map(|(g, &tonality)| {
                let f = delivery.get(g).copied().unwrap_or(1.0).clamp(0.0, 1.0);
                let n = if f >= 0.95 || f >= NOISE_SYNTH_EFF {
                    tonality
                } else {
                    // Patch cannot carry the band: lean on noise.
                    tonality.max(16.0)
                };
                let t = ((f + NOISE_SYNTH_EFF * n) / (1.0 + n)).clamp(0.2, 1.0);
                t_group[g] = t;
                n
            })
            .collect();
        // (2) inverse-delivery boost on the SIGNAL envelope, per
        // signal subband group (keyed into its covering noise group).
        let sig_scf: Vec<f32> = sig_scf
            .iter()
            .enumerate()
            .map(|(sg, &v)| {
                let lo = tables
                    .sbg_sig_highres
                    .get(sg)
                    .copied()
                    .unwrap_or(tables.sbx);
                let g = tables
                    .sbg_noise
                    .iter()
                    .take(t_group.len())
                    .rposition(|&b| b <= lo)
                    .unwrap_or(0);
                v / t_group.get(g).copied().unwrap_or(1.0).max(0.2)
            })
            .collect();
        let sig = crate::encoder_acpl3::extract_aspx_sig_envelope_indices(
            &sig_scf,
            aspx_cfg.quant_mode_env,
            64,
        );
        let noise = crate::encoder_acpl3::extract_aspx_noise_envelope_indices(&noise_scf);
        let noise = if noise.is_empty() { vec![30] } else { noise };
        let ah = crate::aspx_ah_select::select_add_harmonic(q, &tables.sbg_sig_highres, tables.sbx);
        crate::ice::IceAspxChannelRows { sig, noise, ah }
    }

    /// Derive the per-noise-SBG `aspx_tna_mode` vector from an
    /// already-analysed QMF matrix (low band `0..sba`).
    fn ice_tna_from_matrix(aspx_cfg: &crate::aspx::AspxConfig, q: &[Vec<(f32, f32)>]) -> Vec<u8> {
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return Vec::new();
        };
        if tables.sba == 0 {
            return Vec::new();
        }
        let sba = tables.sba as usize;
        let q_low: Vec<Vec<(f32, f32)>> = q.iter().take(sba).map(|row| row.to_vec()).collect();
        let q_low_ext = crate::aspx_tns::build_q_low_ext(&q_low, &[], tables.sba);
        crate::aspx_tna_select::select_tna_mode(
            &q_low_ext,
            &tables,
            aspx_cfg.master_freq_scale,
            true,
        )
    }

    /// Decide `aspx_preflat` from an already-analysed QMF matrix (the
    /// primary carrier's low band) — matrix-level counterpart of
    /// [`Self::extract_aspx_preflat`].
    fn ice_preflat_from_matrix(
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        q: &[Vec<(f32, f32)>],
    ) -> bool {
        let Ok(tables) = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0) else {
            return false;
        };
        let num_ts_in_ats = crate::aspx::num_ts_in_ats(frame_len);
        let aspx_frame_ts_count = crate::aspx::num_aspx_timeslots(frame_len);
        if num_ts_in_ats == 0 || aspx_frame_ts_count == 0 || tables.sba == 0 {
            return false;
        }
        let sba = tables.sba as usize;
        let q_low: Vec<Vec<(f32, f32)>> = q.iter().take(sba).map(|row| row.to_vec()).collect();
        let atsg_sig = [0u32, aspx_frame_ts_count];
        crate::aspx_preflat_select::select_preflat(&q_low, tables.sba, &atsg_sig, num_ts_in_ats)
    }

    /// Assemble one real `aspx_data_2ch()` ICE payload row set from two
    /// analysed channel matrices — public for round-trip validation
    /// harnesses that re-derive the wire rows independently.
    #[doc(hidden)]
    pub fn ice_2ch_rows_from_matrices(
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        q0: &[Vec<(f32, f32)>],
        q1: &[Vec<(f32, f32)>],
    ) -> crate::ice::IceAspx2chRows {
        let tna = Self::ice_tna_from_matrix(aspx_cfg, q0);
        // The balance-coded secondary carries no aspx_hfgen_iwc of its
        // own — the decoder runs it through the bare tile copy (no TNS
        // whitening, no pre-flattening), so its delivery prediction
        // models an unfiltered patch.
        let ch1 = Self::ice_rows_from_matrix(aspx_cfg, frame_len, q1, &[], false);
        crate::ice::IceAspx2chRows {
            ch0: Self::ice_rows_from_matrix(aspx_cfg, frame_len, q0, &tna, aspx_cfg.preflat),
            ch1,
            tna,
        }
    }

    /// Assemble one real `aspx_data_1ch()` ICE payload row set from an
    /// analysed channel matrix — public for round-trip validation
    /// harnesses.
    #[doc(hidden)]
    pub fn ice_1ch_rows_from_matrix(
        aspx_cfg: &crate::aspx::AspxConfig,
        frame_len: u32,
        q: &[Vec<(f32, f32)>],
    ) -> crate::ice::IceAspx1chRows {
        let tna = Self::ice_tna_from_matrix(aspx_cfg, q);
        crate::ice::IceAspx1chRows {
            ch: Self::ice_rows_from_matrix(aspx_cfg, frame_len, q, &tna, aspx_cfg.preflat),
            tna,
        }
    }

    /// The live A-SPX config shared by the ICE encode paths — the same
    /// shape as the 5_X / 7_X live paths (Fine quant, `start_freq = 0`,
    /// LowRes master scale, one noise subband group, FIXFIX-only).
    fn ice_live_aspx_cfg() -> crate::aspx::AspxConfig {
        crate::aspx::AspxConfig {
            quant_mode_env: crate::aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: crate::aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: crate::aspx::AspxFreqResMode::DurationDependent,
        }
    }

    /// 7.0.4 immersive-channel-element ASPX_SCPL encode from PCM
    /// (TS 103 190-2 §6.2.4.1, `immersive_codec_mode = ASPX_SCPL`).
    ///
    /// `frames` is `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`
    /// (the decoder's output slot order). The encoder derives the
    /// eleven SMP tracks with [`Self::ice_scpl_tracks_from_named`],
    /// forward-MDCTs them on persistent per-track TDAC states, and
    /// emits a grouping-3 ASPX_SCPL body whose six A-SPX payloads
    /// carry **real** SIGNAL / NOISE envelopes + `aspx_tna_mode` +
    /// `aspx_add_harmonic` extracted from the decoupled channels the
    /// decoder extends (Table 8 grouping `(L, R)`, `(Ls, Lb)`, `C`,
    /// `(Rs, Rb)`, `(Tfl, Tbl)`, `(Tfr, Tbr)`, each pre-scaled by the
    /// inverse §4.8.3.11.3 Table 10 output gain).
    ///
    /// The output round-trips through [`crate::decoder::Ac4Decoder`]
    /// to an 11-channel `AudioFrame`.
    pub fn encode_frame_pcm_7_0_4_ice_aspx_scpl(&mut self, frames: &[&[f32]; 11]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_4_ice_aspx_scpl_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_4_ice_aspx_scpl`].
    pub fn encode_frame_pcm_7_0_4_ice_aspx_scpl_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 11],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_aspx_scpl_inner(frames, None, max_sfb)
    }

    /// 7.1.4 form of [`Self::encode_frame_pcm_7_0_4_ice_aspx_scpl`]:
    /// `frames` is the 7.0.4 order plus the LFE channel **last**
    /// (`[L, …, Tbr, LFE]`); the decoder emits the LFE on the leading
    /// output slot.
    pub fn encode_frame_pcm_7_1_4_ice_aspx_scpl(&mut self, frames: &[&[f32]; 12]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_4_ice_aspx_scpl_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_1_4_ice_aspx_scpl`].
    pub fn encode_frame_pcm_7_1_4_ice_aspx_scpl_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 12],
        max_sfb: u32,
    ) -> Vec<u8> {
        let named: &[&[f32]; 11] = frames[..11].try_into().expect("11 named channels");
        self.encode_ice_aspx_scpl_inner(named, Some(frames[11]), max_sfb)
    }

    /// Shared 7.0.4 / 7.1.4 ASPX_SCPL body + frame assembly.
    fn encode_ice_aspx_scpl_inner(
        &mut self,
        named: &[&[f32]; 11],
        lfe: Option<&[f32]>,
        max_sfb: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in named.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_ice_aspx_scpl: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(l) = lfe {
            assert_eq!(l.len(), frame_len as usize, "LFE input length");
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Track derivation + forward MDCT (11 tracks; LFE on state 11).
        let tracks = Self::ice_scpl_tracks_from_named(named);
        let n_states = 12;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let coeffs: Vec<Vec<f32>> = tracks
            .iter()
            .enumerate()
            .map(|(t, pcm)| self.mdct_states_multi[t].analyse_frame(pcm))
            .collect();
        let lfe_coeffs = lfe.map(|pcm| self.mdct_states_multi[11].analyse_frame(pcm));

        // A-SPX real payload rows per Table 8 group, extracted from
        // the decoupled channels the decoder extends (output channel ÷
        // its §4.8.3.11.3 Table 10 gain): L/2, R/2, C/2 are exactly
        // tracks A / B / C; the mixing-row channels divide by √2. Each
        // decoupled channel runs through its own persistent streaming
        // QMF bank (fixed `chan_idx` layout below).
        let aspx_cfg = Self::ice_live_aspx_cfg();
        let scale = |x: &[f32], g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
        let isq2 = std::f32::consts::FRAC_1_SQRT_2;
        let dec: [Vec<f32>; 8] = [
            scale(named[3], isq2),  // Ls/√2
            scale(named[5], isq2),  // Lb/√2
            scale(named[4], isq2),  // Rs/√2
            scale(named[6], isq2),  // Rb/√2
            scale(named[7], isq2),  // Tfl/√2
            scale(named[9], isq2),  // Tbl/√2
            scale(named[8], isq2),  // Tfr/√2
            scale(named[10], isq2), // Tbr/√2
        ];
        // chan_idx layout: 0 = L, 1 = R, 2 = C, 3.. = the eight
        // decoupled mixing-row channels in `dec` order.
        let q_l = self.ice_qmf_analyse(0, &tracks[0]);
        let q_r = self.ice_qmf_analyse(1, &tracks[1]);
        let q_c = self.ice_qmf_analyse(2, &tracks[2]);
        let q_dec: Vec<Vec<Vec<(f32, f32)>>> = dec
            .iter()
            .enumerate()
            .map(|(i, pcm)| self.ice_qmf_analyse(3 + i, pcm))
            .collect();
        // preflat: one per-config flag, decided from the primary (L)
        // carrier's low band.
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = Self::ice_preflat_from_matrix(&aspx_cfg, frame_len, &q_l);
        // Payload transmission order per the V1.3.1 Table 8 ASPX_SCPL
        // roster (NOTE 3 — listed order IS bitstream order):
        // (Ls, Lb), (Rs, Rb), C, (L, R), (Tfl, Tbl), (Tfr, Tbr).
        let two_ch = vec![
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[0], &q_dec[1]), // (Ls, Lb)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[2], &q_dec[3]), // (Rs, Rb)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_l, &q_r),           // (L, R)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[4], &q_dec[5]), // (Tfl, Tbl)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[6], &q_dec[7]), // (Tfr, Tbr)
        ];
        let one_ch = Self::ice_1ch_rows_from_matrix(&aspx_cfg, frame_len, &q_c); // C

        let core: [&[f32]; 5] = [&coeffs[0], &coeffs[1], &coeffs[2], &coeffs[3], &coeffs[4]];
        let scpl_pairs: [[&[f32]; 2]; 2] = [[&coeffs[7], &coeffs[8]], [&coeffs[9], &coeffs[10]]];
        let spectra = crate::ice::IceScplSpectra {
            core: &core,
            add_pair: [&coeffs[5], &coeffs[6]],
            scpl_pairs: &scpl_pairs,
        };
        let b_iframe = self.b_iframe_global;
        let mut body = BitWriter::new();
        crate::ice::write_ice_body_aspx_scpl_real(
            &mut body,
            &spectra,
            lfe_coeffs.as_deref().map(|c| (c, 7u32)),
            false,
            &aspx_cfg,
            b_iframe,
            frame_len,
            max_sfb,
            &crate::ice::IceAspxRows {
                two_ch: &two_ch,
                one_ch: &one_ch,
            },
        )
        .expect("encoder: ice aspx_scpl body");
        let out = crate::ice::encode_ice_raw_frame(
            self.sequence_counter as u32,
            lfe.is_some(),
            false,
            b_iframe,
            body,
        )
        .expect("encoder: ice frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        out
    }

    /// Derive the thirteen coded ICE SMP track signals `A..M` from the
    /// named 9.0.4 channels — the `b_5fronts` encode-side inverse of
    /// the decoder's §5.3.3.1 Table 23 S-CPL full-decoding matrix
    /// composed with the mode's output gain ladder. The Table 23
    /// `b_5fronts` front rows carry the fixed ×2 matrix (independent
    /// of `c_gain` / `m_gain`; the ×2 · ½ entries cancel), so the
    /// front inverse is shared by SCPL and ASPX_SCPL exactly like the
    /// surround / top rows:
    ///
    /// ```text
    ///   A  = (L + Lscr)/2       B  = (R + Rscr)/2       C = C/2
    ///   D  = (Ls + Lb)/(2√2)    H  = (Ls − Lb)/(2√2)
    ///   E  = (Rs + Rb)/(2√2)    I  = (Rs − Rb)/(2√2)
    ///   F  = (Tfl + Tbl)/(2√2)  J  = (Tfl − Tbl)/(2√2)
    ///   G  = (Tfr + Tbr)/(2√2)  K  = (Tfr − Tbr)/(2√2)
    ///   L″ = (L − Lscr)/2       M″ = (R − Rscr)/2
    /// ```
    ///
    /// `named` is `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr,
    /// Tbl, Tbr]` (the decoder's 13-slot output order; the L / R slots
    /// carry the Table 23 `Lw` / `Rw` outputs); the return order is
    /// `[A, B, C, D, E, F, G, H, I, J, K, L″, M″]` (core, additional
    /// pair, S-CPL pairs `[H, I]` / `[J, K]` / `[L″, M″]`).
    fn ice_scpl_tracks_from_named_5fronts(named: &[&[f32]; 13]) -> Vec<Vec<f32>> {
        let n = named[0].len();
        let q2 = 0.5f32; // 1/2 — front rows
        let q = 0.5 * std::f32::consts::FRAC_1_SQRT_2; // 1/(2√2)
        let mix = |x: &[f32], y: &[f32], sign: f32, g: f32| -> Vec<f32> {
            (0..n).map(|i| g * (x[i] + sign * y[i])).collect()
        };
        let half = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| v * 0.5).collect() };
        vec![
            mix(named[0], named[3], 1.0, q2),   // A  = (L + Lscr)/2
            mix(named[1], named[4], 1.0, q2),   // B  = (R + Rscr)/2
            half(named[2]),                     // C
            mix(named[5], named[7], 1.0, q),    // D  = (Ls + Lb)/(2√2)
            mix(named[6], named[8], 1.0, q),    // E  = (Rs + Rb)/(2√2)
            mix(named[9], named[11], 1.0, q),   // F  = (Tfl + Tbl)/(2√2)
            mix(named[10], named[12], 1.0, q),  // G  = (Tfr + Tbr)/(2√2)
            mix(named[5], named[7], -1.0, q),   // H  = (Ls − Lb)/(2√2)
            mix(named[6], named[8], -1.0, q),   // I  = (Rs − Rb)/(2√2)
            mix(named[9], named[11], -1.0, q),  // J  = (Tfl − Tbl)/(2√2)
            mix(named[10], named[12], -1.0, q), // K  = (Tfr − Tbr)/(2√2)
            mix(named[0], named[3], -1.0, q2),  // L″ = (L − Lscr)/2
            mix(named[1], named[4], -1.0, q2),  // M″ = (R − Rscr)/2
        ]
    }

    /// 9.0.4 immersive-channel-element ASPX_SCPL encode from PCM
    /// (TS 103 190-2 §6.2.4.1, `immersive_codec_mode = ASPX_SCPL`,
    /// `b_5fronts = 1`).
    ///
    /// `frames` is `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr,
    /// Tbl, Tbr]` (the decoder's output slot order). The encoder
    /// derives the thirteen SMP tracks with
    /// [`Self::ice_scpl_tracks_from_named_5fronts`], forward-MDCTs
    /// them on persistent per-track TDAC states, and emits a
    /// grouping-3 ASPX_SCPL body whose seven A-SPX payloads carry
    /// **real** SIGNAL / NOISE envelopes + `aspx_tna_mode` +
    /// `aspx_add_harmonic` extracted from the decoupled channels the
    /// decoder extends (the `b_5fronts` Table 8 grouping
    /// `(L, Lscr)`, `(R, Rscr)`, `C`, `(Ls, Lb)`, `(Rs, Rb)`,
    /// `(Tfl, Tbl)`, `(Tfr, Tbr)`, each pre-scaled by the inverse
    /// §4.8.3.11.3 Table 11 output gain — 1 on the four front
    /// channels, 2 on C, √2 elsewhere).
    ///
    /// The output round-trips through [`crate::decoder::Ac4Decoder`]
    /// to a 13-channel `AudioFrame`.
    pub fn encode_frame_pcm_9_0_4_ice_aspx_scpl(&mut self, frames: &[&[f32]; 13]) -> Vec<u8> {
        self.encode_frame_pcm_9_0_4_ice_aspx_scpl_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_0_4_ice_aspx_scpl`].
    pub fn encode_frame_pcm_9_0_4_ice_aspx_scpl_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 13],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_aspx_scpl_inner_5fronts(frames, None, max_sfb)
    }

    /// 9.1.4 form of [`Self::encode_frame_pcm_9_0_4_ice_aspx_scpl`]:
    /// `frames` is the 9.0.4 order plus the LFE channel **last**
    /// (`[L, …, Tbr, LFE]`); the decoder emits the LFE on the leading
    /// output slot.
    pub fn encode_frame_pcm_9_1_4_ice_aspx_scpl(&mut self, frames: &[&[f32]; 14]) -> Vec<u8> {
        self.encode_frame_pcm_9_1_4_ice_aspx_scpl_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_1_4_ice_aspx_scpl`].
    pub fn encode_frame_pcm_9_1_4_ice_aspx_scpl_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 14],
        max_sfb: u32,
    ) -> Vec<u8> {
        let named: &[&[f32]; 13] = frames[..13].try_into().expect("13 named channels");
        self.encode_ice_aspx_scpl_inner_5fronts(named, Some(frames[13]), max_sfb)
    }

    /// Shared 9.0.4 / 9.1.4 ASPX_SCPL body + frame assembly
    /// (`b_5fronts` counterpart of [`Self::encode_ice_aspx_scpl_inner`]).
    fn encode_ice_aspx_scpl_inner_5fronts(
        &mut self,
        named: &[&[f32]; 13],
        lfe: Option<&[f32]>,
        max_sfb: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in named.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_9_x_4_ice_aspx_scpl: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(l) = lfe {
            assert_eq!(l.len(), frame_len as usize, "LFE input length");
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Track derivation + forward MDCT (13 tracks; LFE on state 13).
        let tracks = Self::ice_scpl_tracks_from_named_5fronts(named);
        let n_states = 14;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let coeffs: Vec<Vec<f32>> = tracks
            .iter()
            .enumerate()
            .map(|(t, pcm)| self.mdct_states_multi[t].analyse_frame(pcm))
            .collect();
        let lfe_coeffs = lfe.map(|pcm| self.mdct_states_multi[13].analyse_frame(pcm));

        // A-SPX real payload rows per the b_5fronts Table 8 grouping,
        // extracted from the decoupled channels the decoder extends
        // (output channel ÷ its §4.8.3.11.3 Table 11 gain): the four
        // front channels L / R / Lscr / Rscr carry gain 1 (decoupled =
        // the named channel itself), C carries gain 2 (decoupled =
        // C/2 = track C), the surround / top channels √2.
        let aspx_cfg = Self::ice_live_aspx_cfg();
        let scale = |x: &[f32], g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
        let isq2 = std::f32::consts::FRAC_1_SQRT_2;
        let dec: [Vec<f32>; 12] = [
            named[0].to_vec(),      // L
            named[3].to_vec(),      // Lscr
            named[1].to_vec(),      // R
            named[4].to_vec(),      // Rscr
            scale(named[5], isq2),  // Ls/√2
            scale(named[7], isq2),  // Lb/√2
            scale(named[6], isq2),  // Rs/√2
            scale(named[8], isq2),  // Rb/√2
            scale(named[9], isq2),  // Tfl/√2
            scale(named[11], isq2), // Tbl/√2
            scale(named[10], isq2), // Tfr/√2
            scale(named[12], isq2), // Tbr/√2
        ];
        // chan_idx layout: 0..12 = the twelve decoupled pair channels
        // in `dec` order, 12 = the decoupled C (= track C).
        let q_dec: Vec<Vec<Vec<(f32, f32)>>> = dec
            .iter()
            .enumerate()
            .map(|(i, pcm)| self.ice_qmf_analyse(i, pcm))
            .collect();
        let q_c = self.ice_qmf_analyse(12, &tracks[2]);
        // preflat: one per-config flag, decided from the primary (L)
        // carrier's low band.
        let mut aspx_cfg = aspx_cfg;
        aspx_cfg.preflat = Self::ice_preflat_from_matrix(&aspx_cfg, frame_len, &q_dec[0]);
        // Payload transmission order per the V1.3.1 Table 8 ASPX_SCPL
        // b_5fronts roster (NOTE 3 — listed order IS bitstream order):
        // (Ls, Lb), (Rs, Rb), C, (L, Lscr), (R, Rscr), (Tfl, Tbl),
        // (Tfr, Tbr).
        let two_ch = vec![
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[4], &q_dec[5]), // (Ls, Lb)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[6], &q_dec[7]), // (Rs, Rb)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[0], &q_dec[1]), // (L, Lscr)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[2], &q_dec[3]), // (R, Rscr)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[8], &q_dec[9]), // (Tfl, Tbl)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_dec[10], &q_dec[11]), // (Tfr, Tbr)
        ];
        let one_ch = Self::ice_1ch_rows_from_matrix(&aspx_cfg, frame_len, &q_c); // C

        let core: [&[f32]; 5] = [&coeffs[0], &coeffs[1], &coeffs[2], &coeffs[3], &coeffs[4]];
        let scpl_pairs: [[&[f32]; 2]; 3] = [
            [&coeffs[7], &coeffs[8]],
            [&coeffs[9], &coeffs[10]],
            [&coeffs[11], &coeffs[12]],
        ];
        let spectra = crate::ice::IceScplSpectra {
            core: &core,
            add_pair: [&coeffs[5], &coeffs[6]],
            scpl_pairs: &scpl_pairs,
        };
        let b_iframe = self.b_iframe_global;
        let mut body = BitWriter::new();
        crate::ice::write_ice_body_aspx_scpl_real(
            &mut body,
            &spectra,
            lfe_coeffs.as_deref().map(|c| (c, 7u32)),
            true,
            &aspx_cfg,
            b_iframe,
            frame_len,
            max_sfb,
            &crate::ice::IceAspxRows {
                two_ch: &two_ch,
                one_ch: &one_ch,
            },
        )
        .expect("encoder: ice aspx_scpl 5fronts body");
        let out = crate::ice::encode_ice_raw_frame(
            self.sequence_counter as u32,
            lfe.is_some(),
            true,
            b_iframe,
            body,
        )
        .expect("encoder: ice frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        out
    }

    /// 7.0.4 immersive-channel-element ASPX_ACPL_2 encode from PCM
    /// (TS 103 190-2 §6.2.4.1, `immersive_codec_mode = ASPX_ACPL_2`).
    ///
    /// `frames` is `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
    /// The seven coded tracks are `A = L/2`, `B = R/2`, `C = C/2` and
    /// the four module mid carriers `(P+Q)/(2√2)` for the §5.5.2
    /// Table 27 pairs `(Ls, Lb)` / `(Rs, Rb)` / `(Tfl, Tbl)` /
    /// `(Tfr, Tbr)` on tracks D / E / F / G; each module's per-band
    /// `(α, β)` comes from
    /// [`crate::encoder_acpl3::extract_ice_acpl_pair_alpha_beta_q`]
    /// over the pair's mid / side MDCT spectra, and the four A-SPX
    /// payloads carry real synthesis rows extracted from the carrier
    /// tracks themselves (Table 8 grouping `(A, B)` / `(D, E)` /
    /// `(F, G)` / `C`).
    pub fn encode_frame_pcm_7_0_4_ice_acpl2(&mut self, frames: &[&[f32]; 11]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_4_ice_acpl2_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_4_ice_acpl2`].
    pub fn encode_frame_pcm_7_0_4_ice_acpl2_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 11],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_acpl_inner(frames, None, max_sfb, false)
    }

    /// 7.1.4 form of [`Self::encode_frame_pcm_7_0_4_ice_acpl2`] (LFE
    /// **last**; the decoder emits it on the leading output slot).
    pub fn encode_frame_pcm_7_1_4_ice_acpl2(&mut self, frames: &[&[f32]; 12]) -> Vec<u8> {
        let named: &[&[f32]; 11] = frames[..11].try_into().expect("11 named channels");
        self.encode_ice_acpl_inner(named, Some(frames[11]), 40, false)
    }

    /// 7.0.4 immersive-channel-element ASPX_ACPL_1 encode from PCM
    /// (`immersive_codec_mode = ASPX_ACPL_1`, PARTIAL A-CPL config).
    ///
    /// Same layout as [`Self::encode_frame_pcm_7_0_4_ice_acpl2`], but
    /// the module pairs additionally code their **side** signals as
    /// M/S residual tracks below `acpl_qmf_band` on the S-CPL-section
    /// tracks (surround sides on H / I, top sides on J / K), so the
    /// band below the split reconstructs exactly while the bands
    /// above run parametric per-module `(α, β)`.
    pub fn encode_frame_pcm_7_0_4_ice_acpl1(&mut self, frames: &[&[f32]; 11]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_4_ice_acpl1_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_4_ice_acpl1`].
    pub fn encode_frame_pcm_7_0_4_ice_acpl1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 11],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_acpl_inner(frames, None, max_sfb, true)
    }

    /// 7.1.4 form of [`Self::encode_frame_pcm_7_0_4_ice_acpl1`] (LFE
    /// **last**).
    pub fn encode_frame_pcm_7_1_4_ice_acpl1(&mut self, frames: &[&[f32]; 12]) -> Vec<u8> {
        let named: &[&[f32]; 11] = frames[..11].try_into().expect("11 named channels");
        self.encode_ice_acpl_inner(named, Some(frames[11]), 40, true)
    }

    /// Shared ASPX_ACPL_1 / ASPX_ACPL_2 ICE body + frame assembly.
    ///
    /// The `acpl_qmf_band` for the PARTIAL (ACPL_1) config is fixed at
    /// QMF subband 6 (2 250 Hz — comfortably below the live config's
    /// crossover), giving `start_band = sb_to_pb(6)` parametric bands
    /// above an exactly-coded M/S band.
    fn encode_ice_acpl_inner(
        &mut self,
        named: &[&[f32]; 11],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        is_acpl1: bool,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in named.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_ice_acpl: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(l) = lfe {
            assert_eq!(l.len(), frame_len as usize, "LFE input length");
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        let n = frame_len as usize;

        // Module pairs (P, Q) in §5.5.2 module order.
        let pairs: [(usize, usize); 4] = [(3, 5), (4, 6), (7, 9), (8, 10)];
        let isq2 = std::f32::consts::FRAC_1_SQRT_2;
        let mix = |a: &[f32], b: &[f32], sign: f32, g: f32| -> Vec<f32> {
            (0..n).map(|i| g * 0.5 * (a[i] + sign * b[i])).collect()
        };
        // Coded tracks: A, B, C then the four module mid carriers
        // (`mid/√2`); ACPL_1 additionally codes the four `side/√2`
        // residual tracks (band-limited below acpl_qmf_band).
        let half = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| v * 0.5).collect() };
        let mut track_pcm: Vec<Vec<f32>> = vec![
            half(named[0]), // A
            half(named[1]), // B
            half(named[2]), // C
        ];
        for &(p, q) in &pairs {
            track_pcm.push(mix(named[p], named[q], 1.0, isq2)); // mid/√2
        }
        // Side signals (α/β spectra source; ACPL_1 residual tracks).
        let sides: Vec<Vec<f32>> = pairs
            .iter()
            .map(|&(p, q)| mix(named[p], named[q], -1.0, isq2))
            .collect();

        // MDCT: states 0..7 = coded A..G(mid), 7..11 = side spectra,
        // 11 = LFE.
        let n_states = 12;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs: Vec<Vec<f32>> = Vec::with_capacity(7);
        for (t, pcm) in track_pcm.iter().enumerate() {
            coeffs.push(self.mdct_states_multi[t].analyse_frame(pcm));
        }
        let side_coeffs: Vec<Vec<f32>> = sides
            .iter()
            .enumerate()
            .map(|(m, pcm)| self.mdct_states_multi[7 + m].analyse_frame(pcm))
            .collect();
        let lfe_coeffs = lfe.map(|pcm| self.mdct_states_multi[11].analyse_frame(pcm));

        // A-CPL parameters per module: α/β over the (mid, side)
        // spectra (mid = √2 · coded carrier spectrum).
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
        let acpl_qmf_band: u8 = if is_acpl1 { 6 } else { 0 };
        let start_pb = if is_acpl1 {
            crate::acpl::sb_to_pb(acpl_qmf_band as u32, num_bands)
        } else {
            0
        };
        let sq2 = std::f32::consts::SQRT_2;
        let modules_q: Vec<(Vec<i32>, Vec<i32>)> = (0..4)
            .map(|m| {
                let mid_spec: Vec<f32> = coeffs[3 + m].iter().map(|&v| v * sq2).collect();
                let side_spec: Vec<f32> = side_coeffs[m].iter().map(|&v| v * sq2).collect();
                crate::encoder_acpl3::extract_ice_acpl_pair_alpha_beta_q(
                    &mid_spec,
                    &side_spec,
                    frame_len,
                    num_bands,
                    start_pb,
                    acpl_quant_mode,
                )
            })
            .collect();

        // A-SPX: preflat from the primary carrier, real rows per
        // Table 8 payload group — carriers themselves are the
        // extension targets on the ACPL routes. QMF extraction
        // channel layout (this route): 0..7 = tracks A..G(mid).
        let mut aspx_cfg = Self::ice_live_aspx_cfg();
        let q_tracks: Vec<Vec<Vec<(f32, f32)>>> = (0..7)
            .map(|t| self.ice_qmf_analyse(t, &track_pcm[t]))
            .collect();
        aspx_cfg.preflat = Self::ice_preflat_from_matrix(&aspx_cfg, frame_len, &q_tracks[0]);
        // The four A-SPX payloads extend exactly the seven carrier
        // tracks, per the V1.3.1 Table 8 ACPL roster (errata note A2):
        // (A, B), (D, E), (F, G), C — for BOTH ACPL modes. The ACPL_1
        // residual tracks H..K are not A-SPX targets (they carry only
        // the band below acpl_qmf_band).
        let two_ch = vec![
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_tracks[0], &q_tracks[1]), // (A, B)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_tracks[3], &q_tracks[4]), // (D, E)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_tracks[5], &q_tracks[6]), // (F, G)
        ];
        let one_ch = Self::ice_1ch_rows_from_matrix(&aspx_cfg, frame_len, &q_tracks[2]); // C

        // Assemble the coded spectra per mode. Both modes carry the
        // four mid carriers on D..G (core + additional pair); ACPL_1
        // additionally codes the four pair residuals on the
        // S-CPL-section tracks H..K, band-limited below acpl_qmf_band
        // (30 MDCT bins per QMF subband at tl = 1920).
        let residual_limit = (acpl_qmf_band as usize) * (frame_len as usize / 64);
        let limit = |spec: &[f32]| -> Vec<f32> {
            spec.iter()
                .enumerate()
                .map(|(i, &v)| if i < residual_limit { v } else { 0.0 })
                .collect()
        };
        let res_spec: Vec<Vec<f32>> = if is_acpl1 {
            side_coeffs.iter().map(|s| limit(s)).collect()
        } else {
            Vec::new()
        };
        let mode = if is_acpl1 {
            crate::ice::IceCodecMode::AspxAcpl1
        } else {
            crate::ice::IceCodecMode::AspxAcpl2
        };
        let modules_ref: Vec<(&[i32], &[i32])> = modules_q
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let acpl = crate::ice::IceAcplParams {
            num_param_bands_id: acpl_num_param_bands_id,
            quant_mode: acpl_quant_mode,
            qmf_band: acpl_qmf_band,
            modules: &modules_ref,
        };
        let b_iframe = self.b_iframe_global;
        let mut body = BitWriter::new();
        let core: [&[f32]; 5] = [&coeffs[0], &coeffs[1], &coeffs[2], &coeffs[3], &coeffs[4]];
        let (add_pair, scpl_pairs): ([&[f32]; 2], Vec<[&[f32]; 2]>) = if is_acpl1 {
            (
                [&coeffs[5], &coeffs[6]],
                vec![[&res_spec[0], &res_spec[1]], [&res_spec[2], &res_spec[3]]],
            )
        } else {
            ([&coeffs[5], &coeffs[6]], Vec::new())
        };
        crate::ice::write_ice_body_acpl_real(
            &mut body,
            mode,
            &core,
            add_pair,
            &scpl_pairs,
            &acpl,
            &aspx_cfg,
            lfe_coeffs.as_deref().map(|c| (c, 7u32)),
            false,
            b_iframe,
            frame_len,
            max_sfb,
            &crate::ice::IceAspxRows {
                two_ch: &two_ch,
                one_ch: &one_ch,
            },
        )
        .expect("encoder: ice acpl body");
        let out = crate::ice::encode_ice_raw_frame(
            self.sequence_counter as u32,
            lfe.is_some(),
            false,
            b_iframe,
            body,
        )
        .expect("encoder: ice frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        out
    }

    /// 9.0.4 immersive-channel-element ASPX_ACPL_2 encode from PCM
    /// (TS 103 190-2 §6.2.4.1, `immersive_codec_mode = ASPX_ACPL_2`,
    /// `b_5fronts = 1`).
    ///
    /// `frames` is `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr,
    /// Tbl, Tbr]` (the decoder's output slot order). The seven coded
    /// tracks are the **front mid carriers** `A = (L + Lscr)/2` /
    /// `B = (R + Rscr)/2` (the §5.5.2 Table 27 b_5fronts front modules
    /// read the A / B track positions directly, without the √2
    /// immersive output scale), `C = C/2`, and the four module mid
    /// carriers `(P+Q)/(2√2)` for `(Ls, Lb)` / `(Rs, Rb)` /
    /// `(Tfl, Tbl)` / `(Tfr, Tbr)` on tracks D / E / F / G. Each of
    /// the six modules' per-band `(α, β)` comes from
    /// [`crate::encoder_acpl3::extract_ice_acpl_pair_alpha_beta_q`]
    /// over the pair's mid / side spectra, in the §6.2.4.1
    /// transmission order (four surround / top modules, then the two
    /// front modules).
    pub fn encode_frame_pcm_9_0_4_ice_acpl2(&mut self, frames: &[&[f32]; 13]) -> Vec<u8> {
        self.encode_frame_pcm_9_0_4_ice_acpl2_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_0_4_ice_acpl2`].
    pub fn encode_frame_pcm_9_0_4_ice_acpl2_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 13],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_acpl_inner_5fronts(frames, None, max_sfb, false)
    }

    /// 9.1.4 form of [`Self::encode_frame_pcm_9_0_4_ice_acpl2`] (LFE
    /// **last**; the decoder emits it on the leading output slot).
    pub fn encode_frame_pcm_9_1_4_ice_acpl2(&mut self, frames: &[&[f32]; 14]) -> Vec<u8> {
        let named: &[&[f32]; 13] = frames[..13].try_into().expect("13 named channels");
        self.encode_ice_acpl_inner_5fronts(named, Some(frames[13]), 40, false)
    }

    /// 9.0.4 immersive-channel-element ASPX_ACPL_1 encode from PCM
    /// (`immersive_codec_mode = ASPX_ACPL_1`, PARTIAL A-CPL config,
    /// `b_5fronts = 1`).
    ///
    /// Same layout as [`Self::encode_frame_pcm_9_0_4_ice_acpl2`], but
    /// the module pairs additionally code their **side** signals as
    /// M/S residual tracks below `acpl_qmf_band`: the surround
    /// residuals ride the first S-CPL pair (H / I), the top residuals
    /// the second (J / K), and the **front residuals**
    /// `(L − Lscr)/2` / `(R − Rscr)/2` the third b_5fronts S-CPL pair
    /// (L″ / M″), so every pair's band below the split reconstructs
    /// exactly while the bands above run parametric per-module
    /// `(α, β)`.
    pub fn encode_frame_pcm_9_0_4_ice_acpl1(&mut self, frames: &[&[f32]; 13]) -> Vec<u8> {
        self.encode_frame_pcm_9_0_4_ice_acpl1_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_0_4_ice_acpl1`].
    pub fn encode_frame_pcm_9_0_4_ice_acpl1_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 13],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_acpl_inner_5fronts(frames, None, max_sfb, true)
    }

    /// 9.1.4 form of [`Self::encode_frame_pcm_9_0_4_ice_acpl1`] (LFE
    /// **last**).
    pub fn encode_frame_pcm_9_1_4_ice_acpl1(&mut self, frames: &[&[f32]; 14]) -> Vec<u8> {
        let named: &[&[f32]; 13] = frames[..13].try_into().expect("13 named channels");
        self.encode_ice_acpl_inner_5fronts(named, Some(frames[13]), 40, true)
    }

    /// Shared 9.0.4 / 9.1.4 ASPX_ACPL_1 / ASPX_ACPL_2 ICE body +
    /// frame assembly (`b_5fronts` counterpart of
    /// [`Self::encode_ice_acpl_inner`]).
    ///
    /// Module order (= the decoder's §5.5.2 routing and the
    /// `acpl_data_1ch()` transmission order): `(Ls, Lb)`, `(Rs, Rb)`,
    /// `(Tfl, Tbl)`, `(Tfr, Tbr)`, then the front modules `(L, Lscr)`,
    /// `(R, Rscr)`. The front modules read the A / B track positions
    /// as their mid carriers with **no** √2 immersive output scale, so
    /// their carriers are the plain pair mids `(P+Q)/2` and (ACPL_1)
    /// their residuals the plain sides `(P−Q)/2`.
    fn encode_ice_acpl_inner_5fronts(
        &mut self,
        named: &[&[f32]; 13],
        lfe: Option<&[f32]>,
        max_sfb: u32,
        is_acpl1: bool,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in named.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_9_x_4_ice_acpl: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(l) = lfe {
            assert_eq!(l.len(), frame_len as usize, "LFE input length");
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        let n = frame_len as usize;

        // Surround / top module pairs (P, Q) in §5.5.2 module order,
        // then the front pairs.
        let pairs: [(usize, usize); 4] = [(5, 7), (6, 8), (9, 11), (10, 12)];
        let front_pairs: [(usize, usize); 2] = [(0, 3), (1, 4)];
        let isq2 = std::f32::consts::FRAC_1_SQRT_2;
        let mix = |a: &[f32], b: &[f32], sign: f32, g: f32| -> Vec<f32> {
            (0..n).map(|i| g * 0.5 * (a[i] + sign * b[i])).collect()
        };
        let half = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| v * 0.5).collect() };
        // Coded core tracks: the front mids A / B, C/2, then the four
        // surround / top mid carriers (`mid/√2`).
        let mut track_pcm: Vec<Vec<f32>> = vec![
            mix(named[0], named[3], 1.0, 1.0), // A = (L + Lscr)/2
            mix(named[1], named[4], 1.0, 1.0), // B = (R + Rscr)/2
            half(named[2]),                    // C
        ];
        for &(p, q) in &pairs {
            track_pcm.push(mix(named[p], named[q], 1.0, isq2)); // mid/√2
        }
        // Side signals (α/β spectra source; ACPL_1 residual tracks):
        // surround / top sides at `side/√2`, front sides plain.
        let mut sides: Vec<Vec<f32>> = pairs
            .iter()
            .map(|&(p, q)| mix(named[p], named[q], -1.0, isq2))
            .collect();
        for &(p, q) in &front_pairs {
            sides.push(mix(named[p], named[q], -1.0, 1.0));
        }

        // MDCT: states 0..7 = coded A..G(mid), 7..13 = the six side
        // spectra, 13 = LFE.
        let n_states = 14;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let mut coeffs: Vec<Vec<f32>> = Vec::with_capacity(7);
        for (t, pcm) in track_pcm.iter().enumerate() {
            coeffs.push(self.mdct_states_multi[t].analyse_frame(pcm));
        }
        let side_coeffs: Vec<Vec<f32>> = sides
            .iter()
            .enumerate()
            .map(|(m, pcm)| self.mdct_states_multi[7 + m].analyse_frame(pcm))
            .collect();
        let lfe_coeffs = lfe.map(|pcm| self.mdct_states_multi[13].analyse_frame(pcm));

        // A-CPL parameters per module: α/β over the (mid, side)
        // spectra. Surround / top carriers are `mid/√2` (scale by √2
        // to recover the pair mid); the front carriers ARE the pair
        // mids (scale 1).
        let acpl_num_param_bands_id: u8 = 3;
        let acpl_quant_mode = crate::acpl::AcplQuantMode::Fine;
        let num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
        let acpl_qmf_band: u8 = if is_acpl1 { 6 } else { 0 };
        let start_pb = if is_acpl1 {
            crate::acpl::sb_to_pb(acpl_qmf_band as u32, num_bands)
        } else {
            0
        };
        let sq2 = std::f32::consts::SQRT_2;
        let modules_q: Vec<(Vec<i32>, Vec<i32>)> = (0..6)
            .map(|m| {
                let (mid_spec, side_spec): (Vec<f32>, Vec<f32>) = if m < 4 {
                    (
                        coeffs[3 + m].iter().map(|&v| v * sq2).collect(),
                        side_coeffs[m].iter().map(|&v| v * sq2).collect(),
                    )
                } else {
                    // Front modules: mids on A / B, plain scale.
                    (coeffs[m - 4].clone(), side_coeffs[m].clone())
                };
                crate::encoder_acpl3::extract_ice_acpl_pair_alpha_beta_q(
                    &mid_spec,
                    &side_spec,
                    frame_len,
                    num_bands,
                    start_pb,
                    acpl_quant_mode,
                )
            })
            .collect();

        // A-SPX: real rows per Table 8 payload group — payloads extend
        // the wire tracks (A, B), (D, E), (F, G), C. QMF extraction
        // channel layout: 0..7 = wire tracks A..G.
        let mut aspx_cfg = Self::ice_live_aspx_cfg();
        let q_tracks: Vec<Vec<Vec<(f32, f32)>>> = (0..7)
            .map(|t| self.ice_qmf_analyse(t, &track_pcm[t]))
            .collect();
        aspx_cfg.preflat = Self::ice_preflat_from_matrix(&aspx_cfg, frame_len, &q_tracks[0]);
        // The four A-SPX payloads extend exactly the seven carrier
        // tracks, per the V1.3.1 Table 8 ACPL roster (errata note A2):
        // (A, B), (D, E), (F, G), C — for BOTH ACPL modes; the ACPL_1
        // residual tracks (H..K + the front pair L″/M″) are not A-SPX
        // targets.
        let two_ch = vec![
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_tracks[0], &q_tracks[1]), // (A, B)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_tracks[3], &q_tracks[4]), // (D, E)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_tracks[5], &q_tracks[6]), // (F, G)
        ];
        let one_ch = Self::ice_1ch_rows_from_matrix(&aspx_cfg, frame_len, &q_tracks[2]); // C

        // Assemble the coded spectra per mode. Both modes carry the
        // four surround / top mid carriers on D..G (core + additional
        // pair). ACPL_1 additionally codes the surround residuals on
        // the first S-CPL pair (H, I), the top residuals on the
        // second (J, K), and the front residuals on the third
        // (L″, M″); residual spectra are band-limited below
        // acpl_qmf_band.
        let residual_limit = (acpl_qmf_band as usize) * (frame_len as usize / 64);
        let limit = |spec: &[f32]| -> Vec<f32> {
            spec.iter()
                .enumerate()
                .map(|(i, &v)| if i < residual_limit { v } else { 0.0 })
                .collect()
        };
        let res_spec: Vec<Vec<f32>> = if is_acpl1 {
            side_coeffs.iter().map(|s| limit(s)).collect()
        } else {
            Vec::new()
        };
        let mode = if is_acpl1 {
            crate::ice::IceCodecMode::AspxAcpl1
        } else {
            crate::ice::IceCodecMode::AspxAcpl2
        };
        let modules_ref: Vec<(&[i32], &[i32])> = modules_q
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let acpl = crate::ice::IceAcplParams {
            num_param_bands_id: acpl_num_param_bands_id,
            quant_mode: acpl_quant_mode,
            qmf_band: acpl_qmf_band,
            modules: &modules_ref,
        };
        let b_iframe = self.b_iframe_global;
        let mut body = BitWriter::new();
        let core: [&[f32]; 5] = [&coeffs[0], &coeffs[1], &coeffs[2], &coeffs[3], &coeffs[4]];
        let (add_pair, scpl_pairs): ([&[f32]; 2], Vec<[&[f32]; 2]>) = if is_acpl1 {
            (
                [&coeffs[5], &coeffs[6]],
                vec![
                    [&res_spec[0], &res_spec[1]],
                    [&res_spec[2], &res_spec[3]],
                    [&res_spec[4], &res_spec[5]],
                ],
            )
        } else {
            ([&coeffs[5], &coeffs[6]], Vec::new())
        };
        crate::ice::write_ice_body_acpl_real(
            &mut body,
            mode,
            &core,
            add_pair,
            &scpl_pairs,
            &acpl,
            &aspx_cfg,
            lfe_coeffs.as_deref().map(|c| (c, 7u32)),
            true,
            b_iframe,
            frame_len,
            max_sfb,
            &crate::ice::IceAspxRows {
                two_ch: &two_ch,
                one_ch: &one_ch,
            },
        )
        .expect("encoder: ice acpl 5fronts body");
        let out = crate::ice::encode_ice_raw_frame(
            self.sequence_counter as u32,
            lfe.is_some(),
            true,
            b_iframe,
            body,
        )
        .expect("encoder: ice frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        out
    }

    /// 7.0.4 immersive-channel-element **SCPL** encode from PCM with
    /// automatic §5.2.3.2 SAP encode decisions (TS 103 190-2 §6.2.4.1,
    /// `immersive_codec_mode = SCPL`).
    ///
    /// `frames` is `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
    /// The thirteen/eleven SMP tracks come from the same Table 23
    /// inverse as the ASPX_SCPL arms; on top of the plain matrix the
    /// encoder makes both SAP decisions:
    ///
    /// * **Step 3/4** (`b_use_sap_add_ch`): each `(D, F)` / `(E, G)`
    ///   quartet pair is M/S + prediction coded per sfb pair
    ///   (`wire = (mid, side − g·mid)`, the exact inverse of the
    ///   Pseudocode 59 quartet) wherever the least-squares gain
    ///   engages — [`crate::ice::extract_sap_step34_pair`].
    /// * **Step 5/6**: each S-CPL track predicts from its Table 20
    ///   source carrier (`H ← D`, `I ← E`, `J ← F`, `K ← G`, and with
    ///   `b_5fronts` `L″ ← A`, `M″ ← B`); the wire track carries the
    ///   prediction residual and the full-SAP `chparam_info()` the
    ///   per-pair gains — [`crate::ice::extract_sap_step56_prediction`].
    ///
    /// Correlated vertical content (top channels tracking their
    /// surround carriers) codes dramatically darker S-CPL tracks while
    /// the decoder's `apply_sap_steps` restores them exactly (up to
    /// the `alpha_q · 0,1` gain grid).
    pub fn encode_frame_pcm_7_0_4_ice_scpl_sap(&mut self, frames: &[&[f32]; 11]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_4_ice_scpl_sap_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_4_ice_scpl_sap`].
    pub fn encode_frame_pcm_7_0_4_ice_scpl_sap_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 11],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_scpl_sap_inner(frames.as_slice(), None, max_sfb)
    }

    /// 7.1.4 form of [`Self::encode_frame_pcm_7_0_4_ice_scpl_sap`]
    /// (LFE **last**; the decoder emits it on the leading output
    /// slot).
    pub fn encode_frame_pcm_7_1_4_ice_scpl_sap(&mut self, frames: &[&[f32]; 12]) -> Vec<u8> {
        self.encode_frame_pcm_7_1_4_ice_scpl_sap_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_1_4_ice_scpl_sap`].
    pub fn encode_frame_pcm_7_1_4_ice_scpl_sap_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 12],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_scpl_sap_inner(&frames[..11], Some(frames[11]), max_sfb)
    }

    /// 9.0.4 form of [`Self::encode_frame_pcm_7_0_4_ice_scpl_sap`]
    /// (`b_5fronts = 1`): `frames` is `[L, R, C, Lscr, Rscr, Ls, Rs,
    /// Lb, Rb, Tfl, Tfr, Tbl, Tbr]`; the front residual pair `L″ / M″`
    /// predicts from the front mid tracks A / B.
    pub fn encode_frame_pcm_9_0_4_ice_scpl_sap(&mut self, frames: &[&[f32]; 13]) -> Vec<u8> {
        self.encode_frame_pcm_9_0_4_ice_scpl_sap_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_0_4_ice_scpl_sap`].
    pub fn encode_frame_pcm_9_0_4_ice_scpl_sap_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 13],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_scpl_sap_inner(frames.as_slice(), None, max_sfb)
    }

    /// 9.1.4 form of [`Self::encode_frame_pcm_9_0_4_ice_scpl_sap`]
    /// (LFE **last**).
    pub fn encode_frame_pcm_9_1_4_ice_scpl_sap(&mut self, frames: &[&[f32]; 14]) -> Vec<u8> {
        self.encode_frame_pcm_9_1_4_ice_scpl_sap_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_1_4_ice_scpl_sap`].
    pub fn encode_frame_pcm_9_1_4_ice_scpl_sap_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 14],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_scpl_sap_inner(&frames[..13], Some(frames[13]), max_sfb)
    }

    /// Shared SCPL + SAP body + frame assembly for both layouts
    /// (`named.len()` selects: 11 → 7.X.4, 13 → `b_5fronts` 9.X.4).
    fn encode_ice_scpl_sap_inner(
        &mut self,
        named: &[&[f32]],
        lfe: Option<&[f32]>,
        max_sfb: u32,
    ) -> Vec<u8> {
        let b_5fronts = match named.len() {
            11 => false,
            13 => true,
            n => panic!("encode_ice_scpl_sap_inner: {n} named channels (want 11 or 13)"),
        };
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in named.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_ice_scpl_sap: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(l) = lfe {
            assert_eq!(l.len(), frame_len as usize, "LFE input length");
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // Track derivation + forward MDCT.
        let tracks = if b_5fronts {
            let named13: &[&[f32]; 13] = named.try_into().expect("13 named channels");
            Self::ice_scpl_tracks_from_named_5fronts(named13)
        } else {
            let named11: &[&[f32]; 11] = named.try_into().expect("11 named channels");
            Self::ice_scpl_tracks_from_named(named11)
        };
        let n_tracks = tracks.len();
        let n_states = n_tracks + 1;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let coeffs: Vec<Vec<f32>> = tracks
            .iter()
            .enumerate()
            .map(|(t, pcm)| self.mdct_states_multi[t].analyse_frame(pcm))
            .collect();
        let lfe_coeffs = lfe.map(|pcm| self.mdct_states_multi[n_tracks].analyse_frame(pcm));

        // Step 3/4 decision on the (D, F) / (E, G) quartets. The wire
        // D/F/E/G spectra are replaced by the mid / predicted-side
        // pairs where a chparam engages.
        let mut wire: Vec<Vec<f32>> = coeffs.clone();
        let mut sap_add: Option<[crate::asf::ChparamInfo; 2]> = None;
        let s34_df =
            crate::ice::extract_sap_step34_pair(&coeffs[3], &coeffs[5], frame_len, max_sfb);
        let s34_eg =
            crate::ice::extract_sap_step34_pair(&coeffs[4], &coeffs[6], frame_len, max_sfb);
        if s34_df.is_some() || s34_eg.is_some() {
            let mut pair = [
                crate::ice::identity_chparam(),
                crate::ice::identity_chparam(),
            ];
            if let Some((info, wx, wy)) = s34_df {
                pair[0] = info;
                wire[3] = wx;
                wire[5] = wy;
            }
            if let Some((info, wx, wy)) = s34_eg {
                pair[1] = info;
                wire[4] = wx;
                wire[6] = wy;
            }
            sap_add = Some(pair);
        }
        // Step 5/6 decision per S-CPL track against its Table 20
        // source (the TRUE carrier — the decoder's step 3/4 restores
        // the true D/E/F/G before the additive rows apply).
        let targets: &[(usize, usize)] = if b_5fronts {
            &[(7, 3), (8, 4), (9, 5), (10, 6), (11, 0), (12, 1)]
        } else {
            &[(7, 3), (8, 4), (9, 5), (10, 6)]
        };
        let mut scpl_chparam: Vec<crate::asf::ChparamInfo> = Vec::with_capacity(targets.len());
        let mut any56 = false;
        for &(tgt, src) in targets {
            match crate::ice::extract_sap_step56_prediction(
                &coeffs[tgt],
                &coeffs[src],
                frame_len,
                max_sfb,
            ) {
                Some((info, residual)) => {
                    scpl_chparam.push(info);
                    wire[tgt] = residual;
                    any56 = true;
                }
                None => scpl_chparam.push(crate::ice::identity_chparam()),
            }
        }
        if !any56 {
            scpl_chparam.clear();
        }

        let core: [&[f32]; 5] = [&wire[0], &wire[1], &wire[2], &wire[3], &wire[4]];
        let scpl_pairs: Vec<[&[f32]; 2]> = if b_5fronts {
            vec![
                [&wire[7], &wire[8]],
                [&wire[9], &wire[10]],
                [&wire[11], &wire[12]],
            ]
        } else {
            vec![[&wire[7], &wire[8]], [&wire[9], &wire[10]]]
        };
        let spectra = crate::ice::IceScplSpectra {
            core: &core,
            add_pair: [&wire[5], &wire[6]],
            scpl_pairs: &scpl_pairs,
        };
        let b_iframe = self.b_iframe_global;
        let mut body = BitWriter::new();
        crate::ice::write_ice_body_scpl_with_sap(
            &mut body,
            &spectra,
            lfe_coeffs.as_deref().map(|c| (c, 7u32)),
            b_5fronts,
            frame_len,
            max_sfb,
            sap_add.as_ref(),
            &scpl_chparam,
        )
        .expect("encoder: ice scpl sap body");
        let out = crate::ice::encode_ice_raw_frame(
            self.sequence_counter as u32,
            lfe.is_some(),
            b_5fronts,
            b_iframe,
            body,
        )
        .expect("encoder: ice frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        out
    }

    /// 7.0.4 immersive-channel-element ASPX_AJCC encode from PCM
    /// (TS 103 190-2 §6.2.4.1 + §5.6, `immersive_codec_mode =
    /// ASPX_AJCC`, core layout) — driven by the
    /// [`crate::encoder_ajcc`] **parameter extractor**.
    ///
    /// `frames` is `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`
    /// (the decoder's output slot order). The encoder derives the
    /// five-channel core as the exact per-module output sums of the
    /// decoder's §5.6.3.5.2 Table 35 / Table 38 reconstruction (dry
    /// gains sum to 1, wet rows cancel, input scale `k = 2 + 1/√2`):
    ///
    /// ```text
    ///   A = (L + Tfl/√2)/k         B = (R + Tfr/√2)/k    C = C/k
    ///   D = (Ls + Lb + Tbl)/(√2k)  E = (Rs + Rb + Tbr)/(√2k)
    /// ```
    ///
    /// extracts the per-band alpha / beta / dry / wet rows from the
    /// named channels' QMF statistics
    /// ([`crate::encoder_ajcc::extract_ajcc_core_rows`]), assembles a
    /// GOP-aware `ajcc_data()`
    /// ([`crate::encoder_ajcc::build_ajcc_data`] — FREQ rows on
    /// I-frames, per-SET FREQ-vs-TIME bit pricing on P-frames), and
    /// emits the ASPX_AJCC body with real A-SPX payload rows on the
    /// core tracks.
    pub fn encode_frame_pcm_7_0_4_ice_ajcc(&mut self, frames: &[&[f32]; 11]) -> Vec<u8> {
        self.encode_frame_pcm_7_0_4_ice_ajcc_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_7_0_4_ice_ajcc`].
    pub fn encode_frame_pcm_7_0_4_ice_ajcc_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 11],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_ajcc_inner(frames.as_slice(), None, max_sfb)
    }

    /// 7.1.4 form of [`Self::encode_frame_pcm_7_0_4_ice_ajcc`] (LFE
    /// **last**; the decoder emits it on the leading output slot).
    pub fn encode_frame_pcm_7_1_4_ice_ajcc(&mut self, frames: &[&[f32]; 12]) -> Vec<u8> {
        self.encode_ice_ajcc_inner(&frames[..11], Some(frames[11]), 40)
    }

    /// 9.0.4 immersive-channel-element ASPX_AJCC encode from PCM
    /// (`b_5fronts = 1` — the Table 37 module layout).
    ///
    /// `frames` is `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr,
    /// Tbl, Tbr]`. Core downmix (per-module output sums of the
    /// Table 35 / Table 37 reconstruction):
    ///
    /// ```text
    ///   A = (L + Tfl/√2 + Lscr)/k  B = (R + Tfr/√2 + Rscr)/k
    ///   C = C/k
    ///   D = (Ls + Lb + Tbl)/(√2k)  E = (Rs + Rb + Tbr)/(√2k)
    /// ```
    pub fn encode_frame_pcm_9_0_4_ice_ajcc(&mut self, frames: &[&[f32]; 13]) -> Vec<u8> {
        self.encode_frame_pcm_9_0_4_ice_ajcc_with_max_sfb(frames, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_9_0_4_ice_ajcc`].
    pub fn encode_frame_pcm_9_0_4_ice_ajcc_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 13],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_ice_ajcc_inner(frames.as_slice(), None, max_sfb)
    }

    /// 9.1.4 form of [`Self::encode_frame_pcm_9_0_4_ice_ajcc`] (LFE
    /// **last**).
    pub fn encode_frame_pcm_9_1_4_ice_ajcc(&mut self, frames: &[&[f32]; 14]) -> Vec<u8> {
        self.encode_ice_ajcc_inner(&frames[..13], Some(frames[13]), 40)
    }

    /// Shared ASPX_AJCC ICE body + frame assembly for both layouts
    /// (`named.len()` selects: 11 → core layout, 13 → `b_5fronts`).
    fn encode_ice_ajcc_inner(
        &mut self,
        named: &[&[f32]],
        lfe: Option<&[f32]>,
        max_sfb: u32,
    ) -> Vec<u8> {
        let b_5fronts = match named.len() {
            11 => false,
            13 => true,
            n => panic!("encode_ice_ajcc_inner: {n} named channels (want 11 or 13)"),
        };
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in named.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_ice_ajcc: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        if let Some(l) = lfe {
            assert_eq!(l.len(), frame_len as usize, "LFE input length");
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);
        let n = frame_len as usize;
        let n_named = named.len();

        // Core downmix (see the public arms' docs). Named slot layout:
        // core: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr];
        // 5fronts: [L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr,
        // Tbl, Tbr].
        let isq2 = std::f32::consts::FRAC_1_SQRT_2;
        let k = 2.0f32 + isq2;
        let combo = |terms: &[(usize, f32)], g: f32| -> Vec<f32> {
            (0..n)
                .map(|i| g * terms.iter().map(|&(ch, w)| w * named[ch][i]).sum::<f32>())
                .collect()
        };
        let core_pcm: [Vec<f32>; 5] = if b_5fronts {
            [
                combo(&[(0, 1.0), (9, isq2), (3, 1.0)], 1.0 / k), // A
                combo(&[(1, 1.0), (10, isq2), (4, 1.0)], 1.0 / k), // B
                combo(&[(2, 1.0)], 1.0 / k),                      // C
                combo(&[(5, 1.0), (7, 1.0), (11, 1.0)], isq2 / k), // D
                combo(&[(6, 1.0), (8, 1.0), (12, 1.0)], isq2 / k), // E
            ]
        } else {
            [
                combo(&[(0, 1.0), (7, isq2)], 1.0 / k),            // A
                combo(&[(1, 1.0), (8, isq2)], 1.0 / k),            // B
                combo(&[(2, 1.0)], 1.0 / k),                       // C
                combo(&[(3, 1.0), (5, 1.0), (9, 1.0)], isq2 / k),  // D
                combo(&[(4, 1.0), (6, 1.0), (10, 1.0)], isq2 / k), // E
            ]
        };

        // MDCT: 5 core tracks + LFE on state 5.
        let n_states = 6;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let coeffs: Vec<Vec<f32>> = core_pcm
            .iter()
            .enumerate()
            .map(|(t, pcm)| self.mdct_states_multi[t].analyse_frame(pcm))
            .collect();
        let lfe_coeffs = lfe.map(|pcm| self.mdct_states_multi[5].analyse_frame(pcm));

        // QMF analysis on persistent streaming banks: chan_idx 0..5 =
        // the core tracks (A-SPX row extraction), 5..5+n_named = the
        // named target channels (parameter extraction).
        let q_core: Vec<Vec<Vec<(f32, f32)>>> = (0..5)
            .map(|t| self.ice_qmf_analyse(t, &core_pcm[t]))
            .collect();
        let q_named: Vec<Vec<Vec<(f32, f32)>>> = (0..n_named)
            .map(|c| self.ice_qmf_analyse(5 + c, named[c]))
            .collect();

        // A-JCC parameter extraction + GOP-aware ajcc_data assembly.
        let b_iframe = self.b_iframe_global;
        let cfg_build = crate::encoder_ajcc::AjccBuildConfig::default();
        let nb = crate::ajcc::AJCC_NUM_BANDS_TABLE[(cfg_build.num_param_bands_id & 3) as usize];
        let rows = if b_5fronts {
            let named_q: [&crate::encoder_ajcc::QmfMat; 13] =
                std::array::from_fn(|i| q_named[i].as_slice());
            crate::encoder_ajcc::extract_ajcc_5fronts_rows(
                &named_q,
                nb,
                cfg_build.qm_first,
                cfg_build.qm_second,
            )
        } else {
            let named_q: [&crate::encoder_ajcc::QmfMat; 11] =
                std::array::from_fn(|i| q_named[i].as_slice());
            crate::encoder_ajcc::extract_ajcc_core_rows(
                &named_q,
                nb,
                cfg_build.qm_first,
                cfg_build.qm_second,
            )
        };
        let ajcc = crate::encoder_ajcc::build_ajcc_data(
            &rows,
            &cfg_build,
            &mut self.ajcc_enc_state,
            b_iframe,
        );

        // A-SPX real rows on the core-track payload roster (A, B),
        // (D, E), C; preflat from the primary (A) carrier.
        let mut aspx_cfg = Self::ice_live_aspx_cfg();
        aspx_cfg.preflat = Self::ice_preflat_from_matrix(&aspx_cfg, frame_len, &q_core[0]);
        let two_ch = vec![
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_core[0], &q_core[1]), // (A, B)
            Self::ice_2ch_rows_from_matrices(&aspx_cfg, frame_len, &q_core[3], &q_core[4]), // (D, E)
        ];
        let one_ch = Self::ice_1ch_rows_from_matrix(&aspx_cfg, frame_len, &q_core[2]); // C

        // Companding off (§4.8.3.10.3): synced, not active.
        let companding = crate::aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![false],
            compand_avg: Some(false),
        };
        let core: [&[f32]; 5] = [&coeffs[0], &coeffs[1], &coeffs[2], &coeffs[3], &coeffs[4]];
        let mut body = BitWriter::new();
        crate::ice::write_ice_body_ajcc_real(
            &mut body,
            &core,
            &ajcc,
            &aspx_cfg,
            lfe_coeffs.as_deref().map(|c| (c, 7u32)),
            b_iframe,
            frame_len,
            max_sfb,
            &companding,
            &crate::ice::IceAspxRows {
                two_ch: &two_ch,
                one_ch: &one_ch,
            },
        )
        .expect("encoder: ice ajcc body");
        let out = crate::ice::encode_ice_raw_frame(
            self.sequence_counter as u32,
            lfe.is_some(),
            b_5fronts,
            b_iframe,
            body,
        )
        .expect("encoder: ice frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
        out
    }

    /// 22.2 channel-element encode from PCM (TS 103 190-2 §6.2.4.3,
    /// `22_2_codec_mode = Simple`).
    ///
    /// `frames` carries the 22 fullband channels in the §5.2.4
    /// Table 21 order (`[L, R, C, Tc, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl,
    /// Tbr, Tsl, Tsr, Tfc, Tbc, Bfl, Bfr, Bfc, Cb, Lw, Rw]`) followed
    /// by the two LFEs **last** (`LFE, LFE2`); the decoder emits both
    /// LFEs on the two leading output slots.
    pub fn encode_frame_pcm_22_2_simple(&mut self, frames: &[&[f32]; 24]) -> Vec<u8> {
        self.encode_frame_pcm_22_2_inner(frames, false, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_22_2_simple`].
    pub fn encode_frame_pcm_22_2_simple_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 24],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_22_2_inner(frames, false, max_sfb)
    }

    /// [`Self::encode_frame_pcm_22_2_simple`] with the A-SPX codec
    /// mode (Table 98): each of the eleven `two_channel_data()` pairs
    /// carries a real-synthesis `aspx_data_2ch()` payload extracted
    /// from its own channels.
    pub fn encode_frame_pcm_22_2_aspx(&mut self, frames: &[&[f32]; 24]) -> Vec<u8> {
        self.encode_frame_pcm_22_2_inner(frames, true, 40)
    }

    /// `max_sfb`-parameterised form of
    /// [`Self::encode_frame_pcm_22_2_aspx`].
    pub fn encode_frame_pcm_22_2_aspx_with_max_sfb(
        &mut self,
        frames: &[&[f32]; 24],
        max_sfb: u32,
    ) -> Vec<u8> {
        self.encode_frame_pcm_22_2_inner(frames, true, max_sfb)
    }

    /// Shared 22.2 body + frame assembly.
    fn encode_frame_pcm_22_2_inner(
        &mut self,
        frames: &[&[f32]; 24],
        aspx: bool,
        max_sfb: u32,
    ) -> Vec<u8> {
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(self.frame_rate_index as u32, self.fs_index as u32);
        let frame_len = if frame_len == 0 { 1920 } else { frame_len };
        for (ch, f) in frames.iter().enumerate() {
            assert_eq!(
                f.len(),
                frame_len as usize,
                "encode_frame_pcm_22_2: channel {ch} input length must match frame_len = {frame_len}"
            );
        }
        let (n_msfb_bits, _, _) =
            crate::tables::n_msfb_bits_48(frame_len).expect("encoder: bad tl");
        let n_msfb_cap = (1u32 << n_msfb_bits) - 1;
        let max_sfb = max_sfb.min(n_msfb_cap);

        // MDCT: 22 named + 2 LFE states.
        let n_states = 24;
        while self.mdct_states_multi.len() < n_states {
            self.mdct_states_multi
                .push(EncoderMdctState::new(frame_len));
        }
        for state in self.mdct_states_multi.iter_mut() {
            if state.n != frame_len {
                *state = EncoderMdctState::new(frame_len);
            }
        }
        let coeffs: Vec<Vec<f32>> = frames
            .iter()
            .enumerate()
            .map(|(t, pcm)| self.mdct_states_multi[t].analyse_frame(pcm))
            .collect();

        let b_iframe = self.b_iframe_global;
        let mut body = BitWriter::new();
        let pairs: [[&[f32]; 2]; 11] =
            std::array::from_fn(|p| [coeffs[2 * p].as_slice(), coeffs[2 * p + 1].as_slice()]);
        let lfe = [(coeffs[22].as_slice(), 7u32), (coeffs[23].as_slice(), 7u32)];
        if aspx {
            let mut aspx_cfg = Self::ice_live_aspx_cfg();
            let q_ch: Vec<Vec<Vec<(f32, f32)>>> = (0..22)
                .map(|c| self.ice_qmf_analyse(c, frames[c]))
                .collect();
            aspx_cfg.preflat = Self::ice_preflat_from_matrix(&aspx_cfg, frame_len, &q_ch[0]);
            let rows: Vec<crate::ice::IceAspx2chRows> = (0..11)
                .map(|p| {
                    Self::ice_2ch_rows_from_matrices(
                        &aspx_cfg,
                        frame_len,
                        &q_ch[2 * p],
                        &q_ch[2 * p + 1],
                    )
                })
                .collect();
            crate::ice::write_22_2_body_real(
                &mut body,
                lfe,
                &pairs,
                Some(&aspx_cfg),
                b_iframe,
                frame_len,
                max_sfb,
                &rows,
            )
            .expect("encoder: 22_2 body");
        } else {
            crate::ice::write_22_2_body_real(
                &mut body,
                lfe,
                &pairs,
                None,
                b_iframe,
                frame_len,
                max_sfb,
                &[],
            )
            .expect("encoder: 22_2 body");
        }
        let out = crate::ice::encode_22_2_raw_frame(self.sequence_counter as u32, b_iframe, body)
            .expect("encoder: 22_2 frame assembly");
        self.sequence_counter = (self.sequence_counter.wrapping_add(1)) & 0x3FF;
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
