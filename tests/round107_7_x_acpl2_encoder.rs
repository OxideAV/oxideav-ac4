//! Round 107 — 7.0 SIMPLE/ASPX_ACPL_2 multichannel encoder integration
//! test. Verifies the encoder's emitted 7_X ASPX_ACPL_2 frame round-trips
//! through `Ac4Decoder` end-to-end and produces 7-channel PCM output.
//!
//! Per ETSI TS 103 190-1 §4.2.6.14 Table 33 row `case ASPX_ACPL_2:`. The
//! 7_X (immersive) encoder side of this codec mode is the round-107
//! deliverable; the decoder side (the `parse_7x_audio_data_outer` walker)
//! landed in round 27. The 7_X channel element reuses the same 1ch ACPL /
//! ASPX parameter shape as the round-100 5_X ASPX_ACPL_2 path (Pseudocode
//! 117) but differs in its framing: 2-bit `7_X_codec_mode`,
//! `companding_control(5)`, 2-bit `coding_config`, two `two_channel_data`
//! pairs (L/R + Ls/Rs), a trailing centre `mono_data(0)`, and a
//! two-`aspx_data_2ch` envelope trailer.

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

/// The encoder emits a v2 IMS frame with 7_X_codec_mode = ASPX_ACPL_2 for
/// 7.0 input PCM. The Ac4Decoder accepts the frame, walks the
/// `parse_7x_audio_data_outer` ASPX_ACPL_2 branch end-to-end, and returns
/// an AudioFrame with 7-channel interleaved S16 PCM at the declared sample
/// rate.
#[test]
fn encode_7_0_acpl2_produces_7_channel_audio_frame() {
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
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
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
fn encode_7_0_acpl2_advances_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let _f0 = enc.encode_frame_pcm_7_0_acpl2(&[&z, &z, &z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 1);
    let _f1 = enc.encode_frame_pcm_7_0_acpl2(&[&z, &z, &z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 2);
}

/// The decoder's parsed `last_substream.tools.seven_x_mode` must report
/// `AspxAcpl2` after receiving the encoder's 7_X ASPX_ACPL_2 frame, and
/// the I-frame header `acpl_config_1ch(FULL)`, both `two_channel_data`
/// pairs, the trailing Cfg0 centre `mono_data(0)`, and both
/// `acpl_data_1ch()` parameter sets must reach the dispatch — confirms the
/// encoder's full Table 33 ASPX_ACPL_2 body is walked end-to-end.
#[test]
fn encode_7_0_acpl2_decoder_resolves_aspx_acpl_2_mode() {
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
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl2),
        "decoder must resolve 7_X_codec_mode = AspxAcpl2 after parsing the encoder's frame"
    );
    assert!(
        sub.tools.acpl_config_1ch_full.is_some(),
        "decoder must parse acpl_config_1ch(FULL) from the encoder's I-frame block"
    );
    // The 7_X Cfg0 path carries two two_channel_data() pairs (L/R + Ls/Rs).
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
    // ASPX_ACPL_2 has no joint-MDCT residual layer (ASPX_ACPL_1 only).
    assert!(
        sub.tools.acpl_1_residual_pair[0].is_none(),
        "ASPX_ACPL_2 must not carry a joint-MDCT residual layer"
    );
}

/// Silence input round-trips cleanly: a 7.0 ASPX_ACPL_2 frame built from
/// all-zero PCM still produces a 7-channel S16 frame (the synthesis path
/// runs even when carriers are silent).
#[test]
fn encode_7_0_acpl2_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2(&[&z, &z, &z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2);
}

/// `encode_frame_pcm_7_0_acpl2_with_max_sfb` accepts a wider band budget
/// and still round-trips end-to-end to a 7-channel frame.
#[test]
fn encode_7_0_acpl2_with_wide_max_sfb_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let tone = make_tone_frame(500.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2_with_max_sfb(
        &[&tone, &tone, &tone, &tone, &tone, &tone, &tone],
        48,
    );
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2);
}
