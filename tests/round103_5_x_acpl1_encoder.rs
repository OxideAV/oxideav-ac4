//! Round 103 — 5_X SIMPLE/ASPX_ACPL_1 multichannel encoder integration
//! test. Verifies the encoder's emitted ASPX_ACPL_1 frame round-trips
//! through `Ac4Decoder` end-to-end and produces 5-channel PCM output.
//!
//! Per ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_1:`. The
//! encoder side of this codec mode is the round-103 deliverable; the
//! decoder side (the `parse_aspx_acpl_1_2_inner_body` walker, including
//! the joint-MDCT residual layer, Pseudocode 117) landed in round 25.
//!
//! ASPX_ACPL_1 differs from ASPX_ACPL_2 (round 100) in two structural
//! places: (1) `acpl_config_1ch` is PARTIAL (carries the 3-bit
//! `acpl_qmf_band_minus1` field FULL omits); (2) the body carries an
//! explicit joint-MDCT residual layer (`max_sfb_master + 2× chparam_info
//! + 2× sf_data(ASF)`) transmitting the Ls/Rs surround pair (sSMP,3 /
//! sSMP,4 per Table 181) rather than reconstructing it purely from the
//! L/R carriers. The encoder therefore takes a 5-channel
//! `[L, R, C, Ls, Rs]` input.

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

/// The encoder emits a v2 IMS frame with 5_X_codec_mode = ASPX_ACPL_1
/// for 5.0 input PCM. The Ac4Decoder accepts the frame, walks the
/// `parse_aspx_acpl_1_2_inner_body` ASPX_ACPL_1 branch end-to-end
/// (including the joint-MDCT residual layer), and returns an AudioFrame
/// with 5-channel interleaved S16 PCM at the declared sample rate.
#[test]
fn encode_5_0_acpl1_produces_5_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1(&[&l, &r, &c, &ls, &rs]);
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
    // 5 channel × 1920 samples × 2 bytes (S16).
    assert_eq!(
        af.data[0].len(),
        1920 * 5 * 2,
        "5-channel S16 interleaved PCM expected"
    );
}

/// The encoder always advances `sequence_counter` per call.
#[test]
fn encode_5_0_acpl1_advances_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let _f0 = enc.encode_frame_pcm_5_0_acpl1(&[&z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 1);
    let _f1 = enc.encode_frame_pcm_5_0_acpl1(&[&z, &z, &z, &z, &z]);
    assert_eq!(enc.sequence_counter, 2);
}

/// The decoder's parsed `last_substream.tools.five_x_mode` must report
/// `AspxAcpl1` after receiving an ASPX_ACPL_1 frame, the I-frame header
/// `acpl_config_1ch(PARTIAL)` (with non-zero `qmf_band`) must reach the
/// dispatch, the joint-MDCT residual pair must be persisted, and the two
/// `acpl_data_1ch()` parameter sets must parse — confirms the encoder's
/// full Table 25 ASPX_ACPL_1 body is walked end-to-end.
#[test]
fn encode_5_0_acpl1_decoder_resolves_aspx_acpl_1_mode() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(oxideav_ac4::mch::FiveXCodecMode::AspxAcpl1),
        "decoder must resolve 5_X_codec_mode = AspxAcpl1 after parsing the encoder's frame"
    );
    let cfg = sub
        .tools
        .acpl_config_1ch_partial
        .expect("decoder must parse acpl_config_1ch(PARTIAL) from the encoder's I-frame block");
    assert!(
        cfg.qmf_band >= 1,
        "PARTIAL config must carry a non-zero qmf_band (qmf_band_minus1 + 1)"
    );
    // The Cfg0 path (coding_config == 0) feeds two_channel_data() + a
    // centre mono_data(0).
    assert_eq!(
        sub.tools.two_channel_data.len(),
        1,
        "decoder must walk the two_channel_data() L/R carriers"
    );
    assert!(
        sub.tools.cfg0_centre_mono.is_some(),
        "decoder must walk the Cfg0 centre mono_data(0)"
    );
    // The ASPX_ACPL_1-only joint-MDCT residual layer must be persisted.
    assert!(
        sub.tools.acpl_1_residual_max_sfb_master.is_some(),
        "decoder must read max_sfb_master from the residual layer"
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
        sub.tools.acpl_data_1ch_pair[0].is_some(),
        "decoder must parse the first acpl_data_1ch() (Pseudocode 117 D0)"
    );
    assert!(
        sub.tools.acpl_data_1ch_pair[1].is_some(),
        "decoder must parse the second acpl_data_1ch() (Pseudocode 117 D1)"
    );
}

/// Silence input round-trips cleanly: a 5.0 ASPX_ACPL_1 frame built from
/// all-zero PCM still produces a 5-channel S16 frame (the synthesis path
/// runs even when carriers are silent).
#[test]
fn encode_5_0_acpl1_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1(&[&z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
}

/// The `max_sfb` / `max_sfb_master`-parameterised form accepts a smaller
/// residual-layer band budget and still round-trips to 5-channel PCM.
#[test]
fn encode_5_0_acpl1_with_small_residual_budget_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    // max_sfb = 24, max_sfb_master = 6 (small surround residual budget).
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_with_max_sfb(&[&l, &r, &c, &ls, &rs], 24, 6);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(oxideav_ac4::mch::FiveXCodecMode::AspxAcpl1)
    );
    assert_eq!(
        sub.tools.acpl_1_residual_max_sfb_master,
        Some(6),
        "decoder must recover the encoder's clamped max_sfb_master"
    );
}
