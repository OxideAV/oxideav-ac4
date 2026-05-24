//! Round 114 — 7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_2 multichannel encoder
//! integration test. Verifies the encoder's emitted 7_X ASPX_ACPL_2 frame
//! *with* an LFE channel round-trips through `Ac4Decoder` end-to-end and
//! produces 8-channel PCM output whose LFE slot is rendered.
//!
//! Per ETSI TS 103 190-1 §4.2.6.14 Table 33 row `case ASPX_ACPL_2:` with
//! `b_has_lfe = 1`. The LFE counterpart of the round-107 7.0 encoder:
//! the body is identical except a leading `mono_data(b_lfe = 1)` element
//! (§4.2.6.5 Table 21 + §4.2.8 `sf_info_lfe()` Table 35) sits between the
//! I-frame config block and `companding_control(5)`, exactly where the
//! decoder's `parse_7x_audio_data_outer(b_has_lfe = true)` reads
//! `if (b_has_lfe) mono_data(1);`. The 7.1 channel_mode prefix
//! (`0b1111001`, 7 b — Table 88 channel_mode 6) routes the decoder's
//! `walk_ac4_substream` dispatch through `channels == 8`, which already
//! parses the LFE and renders it into slot 7 (round 80).

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

/// The encoder emits a v2 IMS frame with 7_X_codec_mode = ASPX_ACPL_2 +
/// LFE for 7.1 input PCM. The Ac4Decoder accepts the frame, walks the
/// `parse_7x_audio_data_outer(b_has_lfe = true)` ASPX_ACPL_2 branch
/// end-to-end, and returns an AudioFrame with 8-channel interleaved S16
/// PCM at the declared sample rate.
#[test]
fn encode_7_1_acpl2_produces_8_channel_audio_frame() {
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
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl2(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
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

/// The encoder always advances `sequence_counter` per call.
#[test]
fn encode_7_1_acpl2_advances_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let _f0 = enc.encode_frame_pcm_7_1_acpl2(&[&z, &z, &z, &z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 1);
    let _f1 = enc.encode_frame_pcm_7_1_acpl2(&[&z, &z, &z, &z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 2);
}

/// The decoder's parsed `last_substream.tools` must report
/// `seven_x_mode = AspxAcpl2`, `seven_x_b_has_lfe = true`, and a resolved
/// `lfe_mono_data`, in addition to the full round-107 7.0 body (FULL
/// acpl config, both `two_channel_data` pairs, the trailing Cfg0 centre
/// mono, and both `acpl_data_1ch` parameter sets) — confirms the encoder's
/// full Table 33 ASPX_ACPL_2 + LFE body is walked end-to-end.
#[test]
fn encode_7_1_acpl2_decoder_resolves_lfe_and_full_body() {
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
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl2(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl2),
        "decoder must resolve 7_X_codec_mode = AspxAcpl2"
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
        sub.tools.acpl_config_1ch_full.is_some(),
        "decoder must parse acpl_config_1ch(FULL) from the I-frame block"
    );
    assert_eq!(
        sub.tools.two_channel_data.len(),
        2,
        "decoder must walk both two_channel_data() pairs"
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
    assert!(
        sub.tools.acpl_1_residual_pair[0].is_none(),
        "ASPX_ACPL_2 must not carry a joint-MDCT residual layer"
    );
}

/// A non-silent LFE tone (60 Hz) round-trips to a non-silent reconstructed
/// LFE channel — the decoder IMDCTs `tools.lfe_mono_data.scaled_spec` into
/// the trailing slot 7 (round 80 LFE render). Confirms the LFE element the
/// encoder added isn't merely parsed but actually rendered to PCM.
#[test]
fn encode_7_1_acpl2_lfe_slot_is_non_silent() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let lfe = make_tone_frame(60.0, 0.5);
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl2(&[&z, &z, &z, &z, &z, &z, &z, &lfe]);
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

/// Silence input round-trips cleanly: a 7.1 ASPX_ACPL_2 frame built from
/// all-zero PCM still produces an 8-channel S16 frame.
#[test]
fn encode_7_1_acpl2_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl2(&[&z, &z, &z, &z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
}

/// `encode_frame_pcm_7_1_acpl2_with_max_sfb` accepts a wider band budget
/// and a caller-chosen LFE band budget, and still round-trips end-to-end.
#[test]
fn encode_7_1_acpl2_with_wide_max_sfb_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let tone = make_tone_frame(500.0, 0.2);
    let lfe = make_tone_frame(50.0, 0.4);
    let frame_bytes = enc.encode_frame_pcm_7_1_acpl2_with_max_sfb(
        &[&tone, &tone, &tone, &tone, &tone, &tone, &tone, &lfe],
        48,
        7,
    );
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
}
