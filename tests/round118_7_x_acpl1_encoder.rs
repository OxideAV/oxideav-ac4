//! Round 118 — 7.0 / 7.1 (3/4/0(.1)) SIMPLE/ASPX_ACPL_1 multichannel
//! encoder integration test. Verifies the encoder's emitted 7_X
//! ASPX_ACPL_1 frame round-trips through `Ac4Decoder` end-to-end and
//! produces 7- / 8-channel PCM output.
//!
//! Per ETSI TS 103 190-1 §4.2.6.14 Table 33 row `case ASPX_ACPL_1:`. The
//! encoder side of this codec mode is the round-118 deliverable; the
//! decoder side (the `parse_7x_audio_data_outer` ASPX_ACPL_1 walker,
//! including the joint-MDCT residual layer, Pseudocode 117) landed in
//! round 27.
//!
//! The 7_X (immersive) counterpart to the round-103 5_X ASPX_ACPL_1 path.
//! ASPX_ACPL_1 differs from the round-107/114 7_X ASPX_ACPL_2 path in
//! three structural places: `7_X_codec_mode = 2` (vs 3),
//! `acpl_config_1ch` is PARTIAL (vs FULL — carries the 3-bit
//! `acpl_qmf_band_minus1`), and the body carries an explicit joint-MDCT
//! residual layer (`max_sfb_master + 2× chparam_info + 2× sf_data(ASF)`)
//! transmitting the Ls/Rs surround pair (sSMP,3 / sSMP,4 per Table 181).

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn make_tone_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// The encoder emits a v2 IMS frame with 7_X_codec_mode = ASPX_ACPL_1 for
/// 7.0 input PCM. The Ac4Decoder accepts the frame, walks the
/// `parse_7x_audio_data_outer` ASPX_ACPL_1 branch end-to-end (including
/// the joint-MDCT residual layer), and returns an AudioFrame with
/// 7-channel interleaved S16 PCM at the declared sample rate.
#[test]
fn encode_7_0_acpl1_produces_7_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let lb = make_tone_frame(1320.0, 0.3);
    let rb = make_tone_frame(1540.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl1(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(
        !frame_bytes.is_empty(),
        "encoder must produce non-empty output"
    );

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(
        af.samples, 1920,
        "frame must be 1920 samples (48k / 24 fps)"
    );
    assert_eq!(af.data.len(), 1, "single-substream output expected");
    // 7 channel × 1920 samples × 2 bytes (S16).
    assert_eq!(
        af.data[0].len(),
        1920 * 7 * 2,
        "7-channel S16 interleaved PCM expected"
    );
}

/// The encoder always advances `sequence_counter` per call.
#[test]
fn encode_7_0_acpl1_advances_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let _f0 = enc.encode_frame_pcm_7_0_acpl1(&[&z, &z, &z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 1);
    let _f1 = enc.encode_frame_pcm_7_0_acpl1(&[&z, &z, &z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 2);
}

/// The decoder's parsed `last_substream.tools` must report
/// `seven_x_mode = AspxAcpl1`, the PARTIAL `acpl_config_1ch` (with a
/// non-zero `qmf_band`), both `two_channel_data` pairs, the persisted
/// joint-MDCT residual pair, the trailing Cfg0 centre mono, and both
/// `acpl_data_1ch` parameter sets — confirms the encoder's full Table 33
/// ASPX_ACPL_1 body is walked end-to-end.
#[test]
fn encode_7_0_acpl1_decoder_resolves_full_body() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let lb = make_tone_frame(1320.0, 0.3);
    let rb = make_tone_frame(1540.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl1(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl1),
        "decoder must resolve 7_X_codec_mode = AspxAcpl1"
    );
    assert!(
        !sub.tools.seven_x_b_has_lfe,
        "7.0 dispatch must register b_has_lfe = false"
    );
    let cfg = sub
        .tools
        .acpl_config_1ch_partial
        .expect("decoder must parse acpl_config_1ch(PARTIAL) from the I-frame block");
    assert!(
        cfg.qmf_band >= 1,
        "PARTIAL config must carry a non-zero qmf_band (qmf_band_minus1 + 1)"
    );
    assert_eq!(
        sub.tools.two_channel_data.len(),
        2,
        "decoder must walk both two_channel_data() pairs (L/R + Ls/Rs)"
    );
    assert!(
        sub.tools.acpl_1_residual_max_sfb_master.is_some(),
        "decoder must read max_sfb_master from the joint-MDCT residual layer"
    );
    assert!(
        sub.tools.acpl_1_residual_pair[0].is_some(),
        "decoder must persist the sSMP,3 residual spectrum"
    );
    assert!(
        sub.tools.acpl_1_residual_pair[1].is_some(),
        "decoder must persist the sSMP,4 residual spectrum"
    );
    assert!(
        sub.tools.cfg0_centre_mono.is_some(),
        "decoder must walk the trailing Cfg0 centre mono_data(0)"
    );
    assert!(
        sub.tools.acpl_data_1ch_pair[0].is_some(),
        "decoder must parse the first acpl_data_1ch() (Pseudocode 117 D0)"
    );
    assert!(
        sub.tools.acpl_data_1ch_pair[1].is_some(),
        "decoder must parse the second acpl_data_1ch() (Pseudocode 117 D1)"
    );
}

/// Silence input round-trips cleanly: a 7.0 ASPX_ACPL_1 frame built from
/// all-zero PCM still produces a 7-channel S16 frame.
#[test]
fn encode_7_0_acpl1_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl1(&[&z, &z, &z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2);
}

/// The encoder emits a v2 IMS frame with 7_X_codec_mode = ASPX_ACPL_1 +
/// LFE for 7.1 input PCM. The decoder walks the
/// `parse_7x_audio_data_outer(b_has_lfe = true)` ASPX_ACPL_1 branch
/// end-to-end and returns an 8-channel interleaved S16 PCM frame.
#[test]
fn encode_7_1_acpl1_produces_8_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let lb = make_tone_frame(1320.0, 0.3);
    let rb = make_tone_frame(1540.0, 0.3);
    let lfe = make_tone_frame(60.0, 0.4);
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl1(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert!(
        !frame_bytes.is_empty(),
        "encoder must produce non-empty output"
    );

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(
        af.samples, 1920,
        "frame must be 1920 samples (48k / 24 fps)"
    );
    assert_eq!(af.data.len(), 1, "single-substream output expected");
    // 8 channel × 1920 samples × 2 bytes (S16).
    assert_eq!(
        af.data[0].len(),
        1920 * 8 * 2,
        "8-channel S16 interleaved PCM expected"
    );
}

/// The 7.1 decoder must resolve `seven_x_mode = AspxAcpl1`,
/// `seven_x_b_has_lfe = true`, a parsed `lfe_mono_data`, the PARTIAL
/// config, both `two_channel_data` pairs, the residual pair, the centre
/// mono, and both `acpl_data_1ch` parameter sets.
#[test]
fn encode_7_1_acpl1_decoder_resolves_lfe_and_full_body() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let lb = make_tone_frame(1320.0, 0.3);
    let rb = make_tone_frame(1540.0, 0.3);
    let lfe = make_tone_frame(60.0, 0.4);
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl1(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl1),
        "decoder must resolve 7_X_codec_mode = AspxAcpl1"
    );
    assert!(
        sub.tools.seven_x_b_has_lfe,
        "decoder must register b_has_lfe = true for the 7.1 dispatch"
    );
    assert!(
        sub.tools.lfe_mono_data.is_some(),
        "decoder must parse the leading LFE mono_data(1)"
    );
    assert!(
        sub.tools.acpl_config_1ch_partial.is_some(),
        "decoder must parse acpl_config_1ch(PARTIAL) from the I-frame block"
    );
    assert_eq!(
        sub.tools.two_channel_data.len(),
        2,
        "decoder must walk both two_channel_data() pairs"
    );
    assert!(
        sub.tools.acpl_1_residual_pair[0].is_some(),
        "decoder must persist the sSMP,3 residual spectrum"
    );
    assert!(
        sub.tools.acpl_1_residual_pair[1].is_some(),
        "decoder must persist the sSMP,4 residual spectrum"
    );
    assert!(
        sub.tools.cfg0_centre_mono.is_some(),
        "decoder must walk the trailing Cfg0 centre mono_data(0)"
    );
    assert!(
        sub.tools.acpl_data_1ch_pair[0].is_some(),
        "decoder must parse the first acpl_data_1ch() (Pseudocode 117 D0)"
    );
    assert!(
        sub.tools.acpl_data_1ch_pair[1].is_some(),
        "decoder must parse the second acpl_data_1ch() (Pseudocode 117 D1)"
    );
}

/// A non-silent LFE tone (60 Hz) round-trips to a non-silent reconstructed
/// LFE channel — the decoder IMDCTs `tools.lfe_mono_data.scaled_spec` into
/// the trailing slot 7 (round 80 LFE render).
#[test]
fn encode_7_1_acpl1_lfe_slot_is_non_silent() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let lfe = make_tone_frame(60.0, 0.5);
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl1(&[&z, &z, &z, &z, &z, &z, &z, &lfe]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
    // Deinterleave the LFE slot (channel index 7 of 8) and check it's
    // non-silent.
    let bytes = &af.data[0];
    let mut max_abs: i32 = 0;
    for i in 0..1920usize {
        let off = (i * 8 + 7) * 2;
        let s = i16::from_le_bytes([bytes[off], bytes[off + 1]]) as i32;
        max_abs = max_abs.max(s.abs());
    }
    assert!(
        max_abs > 0,
        "LFE slot 7 must be non-silent for a 60 Hz LFE tone (got max |sample| = {max_abs})"
    );
}

/// The `_with_max_sfb` form accepts a smaller residual band budget and a
/// caller-chosen LFE budget and still round-trips, recovering the clamped
/// `max_sfb_master`.
#[test]
fn encode_7_1_acpl1_with_small_residual_budget_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let tone = make_tone_frame(500.0, 0.2);
    let lfe = make_tone_frame(50.0, 0.4);
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl1_with_max_sfb(
        &[&tone, &tone, &tone, &tone, &tone, &tone, &tone, &lfe],
        24,
        6,
        7,
    );
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl1)
    );
    assert_eq!(
        sub.tools.acpl_1_residual_max_sfb_master,
        Some(6),
        "decoder must recover the encoder's clamped max_sfb_master"
    );
}
