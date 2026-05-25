//! Round 139 — 7_X (7.1 with LFE) SIMPLE/ASPX_ACPL_1 multichannel
//! encoder with **real per-parameter-band α + β extraction**.
//!
//! The LFE counterpart of the round-135 7.0 immersive real-α+β path
//! ([`Ac4ImsEncoder::encode_frame_pcm_7_0_acpl1_real_alpha_beta`]). The
//! round-118 7.1 ASPX_ACPL_1 encoder emitted both `acpl_data_1ch()`
//! parameter sets at the zero-delta scaffold; round 139 upgrades them
//! to carry the analytic α (from the L/Ls and R/Rs MDCT-energy
//! correlation, §5.7.7.5 Pseudocode 116) plus the β magnitude that
//! closes the surround/carrier energy balance after α removes the
//! level-only component (§5.7.7.6.1 Pseudocode 117):
//!
//! ```text
//!   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
//!   ⇒  β = √max(0, 2·E[Ls²]/E[L²] − (1 − α)²)
//! ```
//!
//! These tests confirm:
//!
//!   1. The new entry point produces an 8-channel AudioFrame round-trip.
//!   2. The decoder resolves the frame to `SevenXCodecMode::AspxAcpl1`
//!      with `b_has_lfe = true`, the leading LFE `mono_data(1)` parsed,
//!      both `acpl_data_1ch_pair` slots populated and the joint-MDCT
//!      residual layer walked.
//!   3. A non-silent 60 Hz LFE tone round-trips to a non-silent
//!      reconstructed LFE channel (slot 7).
//!   4. Silence input yields all-zero β_q in every band.
//!   5. The encoder is bit-deterministic for matched inputs.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn make_tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

fn make_tone_phase(freq: f32, amp: f32, phase: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t + phase).sin()
        })
        .collect()
}

#[test]
fn encode_7_1_acpl1_real_alpha_beta_produces_8_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let lb = make_tone(1320.0, 0.3);
    let rb = make_tone(1540.0, 0.3);
    let lfe = make_tone(60.0, 0.4);
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert!(
        !frame_bytes.is_empty(),
        "real-α+β 7.1 encoder must produce non-empty output"
    );

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(
        af.data[0].len(),
        1920 * 8 * 2,
        "8-channel S16 interleaved PCM expected"
    );
}

#[test]
fn encode_7_1_acpl1_real_alpha_beta_decoder_resolves_lfe_and_full_body() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let lb = make_tone(1320.0, 0.3);
    let rb = make_tone(1540.0, 0.3);
    let lfe = make_tone(60.0, 0.4);
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
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
        "decoder must walk both two_channel_data() pairs (L/R + Ls/Rs)"
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

/// Loud anti-phase surround drives the extractor into the β-positive
/// regime (E[Ls²] > 0.5·E[L²]·(1−α)²). The 7.1 frame must still
/// round-trip to an 8-channel AudioFrame and resolve as AspxAcpl1 with
/// LFE present — confirming the extra β layer doesn't break the 7.1
/// substream-body layout or the IMDCT path.
#[test]
fn encode_7_1_acpl1_real_alpha_beta_loud_antiphase_surround_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(1500.0, 0.3);
    let r = make_tone(1800.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone_phase(1500.0, 0.9, std::f32::consts::PI);
    let rs = make_tone_phase(1800.0, 0.9, std::f32::consts::PI);
    let lb = make_tone(1320.0, 0.3);
    let rb = make_tone(1540.0, 0.3);
    let lfe = make_tone(60.0, 0.4);

    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl1)
    );
    assert!(sub.tools.seven_x_b_has_lfe);
}

/// A non-silent LFE tone (60 Hz) round-trips through the real-α+β 7.1
/// path to a non-silent reconstructed LFE channel — the decoder IMDCTs
/// `tools.lfe_mono_data.scaled_spec` into the trailing slot 7 (round 80
/// LFE render). The real-α+β changes touch only the trailing
/// `acpl_data_1ch()` parameter sets, not the LFE element, so the slot-7
/// non-silence invariant must hold.
#[test]
fn encode_7_1_acpl1_real_alpha_beta_lfe_slot_is_non_silent() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let lfe = make_tone(60.0, 0.5);
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&z, &z, &z, &z, &z, &z, &z, &lfe]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
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

/// Silence input round-trips through the real-α+β 7.1 path — both α
/// and β default to the zero-codebook index in every band of both
/// `acpl_data_1ch()` parameter sets.
#[test]
fn encode_7_1_acpl1_real_alpha_beta_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&z, &z, &z, &z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    for (idx, pair) in sub.tools.acpl_data_1ch_pair.iter().enumerate() {
        let p = pair.as_ref().unwrap_or_else(|| panic!("pair {idx} parsed"));
        let all_beta_zero = p.beta1.iter().all(|ps| ps.values.iter().all(|&v| v == 0));
        assert!(
            all_beta_zero,
            "silence input must produce all-zero beta_q in pair {idx}; got {:?}",
            p.beta1
        );
    }
}

/// Encoder is bit-deterministic for matched inputs and fresh state.
#[test]
fn encode_7_1_acpl1_real_alpha_beta_is_deterministic() {
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let lb = make_tone(1320.0, 0.3);
    let rb = make_tone(1540.0, 0.3);
    let lfe = make_tone(60.0, 0.4);
    let mut enc1 = Ac4ImsEncoder::new();
    let mut enc2 = Ac4ImsEncoder::new();
    let f1 =
        enc1.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    let f2 =
        enc2.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert_eq!(
        f1, f2,
        "encoder must be deterministic for matched inputs and fresh state"
    );
}
