//! Round 202 — 7.0 / 7.1 SIMPLE/ASPX_ACPL_2 multichannel encoder with
//! **real per-parameter-band α + β extraction**.
//!
//! The 7_X (immersive) counterpart to round 144 (which added real α + β
//! to the 5.0 ACPL_2 path) and the real-α-β upgrade of the round-107 /
//! 114 zero-delta 7_X ASPX_ACPL_2 encoder. ACPL_2 does **not** transmit
//! the Ls/Rs surround pair on the wire — the decoder reconstructs the
//! surround from the L/R carriers + the two `acpl_data_1ch()` parameter
//! sets per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 + §5.7.7.6.1
//! Pseudocode 117:
//!
//! ```text
//!   z0 = 0.5 · (x0·(1+α) + y·β)        // recovers L  carrier
//!   z1 = 0.5 · (x0·(1−α) − y·β)        // recovers Ls (then ·√2)
//! ```
//!
//! D0 module models (L → Ls); D1 module models (R → Rs).
//! `acpl_config_1ch(FULL)` carries no `qmf_band` → `start_band = 0` so
//! every parameter band participates (in contrast to the ACPL_1 PARTIAL
//! mode whose `acpl_qmf_band` masks the low bands).
//!
//! These tests confirm:
//!
//!   1. The new 7.0 / 7.1 entry points produce 7- / 8-channel AudioFrame
//!      round-trips.
//!   2. The decoder resolves the encoder's frame to
//!      `SevenXCodecMode::AspxAcpl2` and persists both
//!      `acpl_data_1ch_pair` slots.
//!   3. Loud-surround vs silence-surround inputs produce materially
//!      different bytes (the round-107 / 114 zero-delta scaffold would
//!      produce identical α / β codewords regardless of surround input).
//!   4. Silence input round-trips with β_q = 0 in every band.
//!   5. Encoder is bit-deterministic for matched inputs and fresh state.
//!   6. Direct body-builder probe diverges from the round-107 zero-delta
//!      scaffold byte stream when the caller's Ls/Rs spectra are
//!      non-trivial.

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

// ====================================================================
// 7.0 ACPL_2 real-α-β
// ====================================================================

#[test]
fn encode_7_0_acpl2_real_alpha_beta_produces_7_channel_audio_frame() {
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
    let frame_bytes =
        enc.encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(
        !frame_bytes.is_empty(),
        "real-α+β 7.0 ACPL_2 encoder must produce non-empty output"
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
        1920 * 7 * 2,
        "7-channel S16 interleaved PCM expected"
    );
}

#[test]
fn encode_7_0_acpl2_real_alpha_beta_decoder_resolves_aspx_acpl_2_mode() {
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
    let frame_bytes =
        enc.encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl2)
    );
    assert!(sub.tools.acpl_data_1ch_pair[0].is_some());
    assert!(sub.tools.acpl_data_1ch_pair[1].is_some());
}

/// Loud-surround vs silence-surround input → materially different bytes.
/// The round-107 zero-delta scaffold emits identical α/β codewords
/// regardless of surround input; the real-α-β path must not.
#[test]
fn encode_7_0_acpl2_real_alpha_beta_loud_surround_produces_different_bytes_than_silence() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let lb = make_tone(1320.0, 0.3);
    let rb = make_tone(1540.0, 0.3);
    let silence = vec![0.0f32; N];
    let ls_loud = make_tone(880.0, 0.95);
    let rs_loud = make_tone(1100.0, 0.95);

    let frame_silence = enc_a
        .encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &silence, &silence, &lb, &rb]);
    let frame_loud = enc_b
        .encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls_loud, &rs_loud, &lb, &rb]);
    // Substream bodies are tightly sized (audio_size = exact body
    // length), so the two frames need not share a length; the byte-stream
    // inequality below is the real check.
    assert_ne!(
        frame_silence, frame_loud,
        "real-α+β 7.0 ACPL_2 encoder must produce different bytes when surround content differs"
    );
}

/// Silence input round-trips with β_q = 0 in every band on pair0 (no
/// surround energy to model).
#[test]
fn encode_7_0_acpl2_real_alpha_beta_silence_round_trips_with_zero_beta_q() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&z, &z, &z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let pair0 = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("pair0 parsed");
    let all_beta_zero = pair0
        .beta1
        .iter()
        .all(|ps| ps.values.iter().all(|&v| v == 0));
    assert!(
        all_beta_zero,
        "silence input must produce all-zero beta_q; got {:?}",
        pair0.beta1
    );
}

/// Encoder is bit-deterministic for matched inputs and fresh state.
#[test]
fn encode_7_0_acpl2_real_alpha_beta_is_deterministic() {
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let lb = make_tone(1320.0, 0.3);
    let rb = make_tone(1540.0, 0.3);
    let mut enc1 = Ac4ImsEncoder::new();
    let mut enc2 = Ac4ImsEncoder::new();
    let f1 = enc1.encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let f2 = enc2.encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert_eq!(
        f1, f2,
        "encoder must be deterministic for matched inputs and fresh state"
    );
}

// ====================================================================
// 7.1 (LFE) ACPL_2 real-α-β
// ====================================================================

#[test]
fn encode_7_1_acpl2_real_alpha_beta_produces_8_channel_audio_frame() {
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
        enc.encode_frame_pcm_7_1_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert!(
        !frame_bytes.is_empty(),
        "real-α+β 7.1 ACPL_2 encoder must produce non-empty output"
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
fn encode_7_1_acpl2_real_alpha_beta_decoder_resolves_aspx_acpl_2_mode() {
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
        enc.encode_frame_pcm_7_1_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl2)
    );
    assert!(sub.tools.acpl_data_1ch_pair[0].is_some());
    assert!(sub.tools.acpl_data_1ch_pair[1].is_some());
}

/// 7.1 silence round-trips with β_q = 0 across pair0.
#[test]
fn encode_7_1_acpl2_real_alpha_beta_silence_round_trips_with_zero_beta_q() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl2_real_alpha_beta(&[&z, &z, &z, &z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let pair0 = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("pair0 parsed");
    let all_beta_zero = pair0
        .beta1
        .iter()
        .all(|ps| ps.values.iter().all(|&v| v == 0));
    assert!(
        all_beta_zero,
        "silence 7.1 input must produce all-zero beta_q; got {:?}",
        pair0.beta1
    );
}

/// Encoder is bit-deterministic for matched 7.1 inputs and fresh state.
#[test]
fn encode_7_1_acpl2_real_alpha_beta_is_deterministic() {
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
        enc1.encode_frame_pcm_7_1_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    let f2 =
        enc2.encode_frame_pcm_7_1_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert_eq!(
        f1, f2,
        "encoder must be deterministic for matched 7.1 inputs and fresh state"
    );
}

// ====================================================================
// Direct body-builder probe — diverges from r107 zero-delta scaffold
// ====================================================================

/// Direct body-builder probe: confirm the new 7_X builder produces a
/// byte stream that differs from the round-107 zero-delta scaffold when
/// the caller supplies non-trivial Ls/Rs. (Same wire schedule; only the
/// α + β codewords differ.)
#[test]
fn build_7_x_acpl2_body_real_alpha_beta_diverges_from_scaffold_for_nonzero_surround() {
    use oxideav_ac4::acpl::AcplQuantMode;
    use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
    use oxideav_ac4::encoder_acpl3::{
        build_7_x_acpl2_body_from_pcm_spectra,
        build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta,
    };

    let tl = 1920u32;
    let max_sfb = 40u32;
    let mut coeffs_l = vec![0.0f32; tl as usize];
    let mut coeffs_r = vec![0.0f32; tl as usize];
    let mut coeffs_c = vec![0.0f32; tl as usize];
    let mut coeffs_ls = vec![0.0f32; tl as usize];
    let mut coeffs_rs = vec![0.0f32; tl as usize];
    for bin in 50..400 {
        coeffs_l[bin] = 0.5;
        coeffs_r[bin] = 0.4;
        coeffs_c[bin] = 0.2;
        coeffs_ls[bin] = 2.5; // β must fire — E_s/E_c = 25
        coeffs_rs[bin] = 2.5;
    }
    let aspx_cfg = AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        start_freq: 0,
        stop_freq: 0,
        master_freq_scale: AspxMasterFreqScale::LowRes,
        interpolation: false,
        preflat: false,
        limiter: false,
        noise_sbg: 0,
        num_env_bits_fixfix: 0,
        freq_res_mode: AspxFreqResMode::DurationDependent,
    };
    let scaffold = build_7_x_acpl2_body_from_pcm_spectra(
        tl,
        max_sfb,
        None,
        true,
        &coeffs_l,
        &coeffs_r,
        &coeffs_ls,
        &coeffs_rs,
        &coeffs_c,
        None,
        &aspx_cfg,
        3,
        AcplQuantMode::Fine,
        12288,
    );
    let real = build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
        tl,
        max_sfb,
        None,
        true,
        &coeffs_l,
        &coeffs_r,
        &coeffs_ls,
        &coeffs_rs,
        &coeffs_c,
        None,
        &aspx_cfg,
        3,
        AcplQuantMode::Fine,
        12288,
    );
    // Substream bodies are tightly sized (audio_size = exact body
    // length), so the two frames need not share a length; the byte-stream
    // inequality below is the real check.
    assert_ne!(
        scaffold, real,
        "real-α+β 7_X builder must produce a different byte stream than the round-107 scaffold when the caller's Ls/Rs spectra are non-trivial"
    );
}
