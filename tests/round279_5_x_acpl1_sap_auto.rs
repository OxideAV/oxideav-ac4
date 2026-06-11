//! Round 279 — decision-driven SAP-coded ASPX_ACPL_1 residual layer,
//! end-to-end through `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_sap`
//! and `Ac4Decoder`.
//!
//! Per ETSI TS 103 190-1 §5.3.4.3.2 / Table 181 + §5.3.2 Pseudocode 59:
//! the joint-MDCT residual layer's two `chparam_info()` payloads drive
//! two independent 2x2 SAP systems mapping the transmitted
//! `(sSMP_A, sSMP_3)` / `(sSMP_B, sSMP_4)` tracks to the preliminary
//! `(L, Ls)` / `(R, Rs)` pairs. Round 271 landed the per-band `alpha_q`
//! least-squares decision driver (`select_alpha_q_for_pair`); round 279
//! wires it into the body builder: the encoder now derives the SAP-coded
//! `chparam_info()` rows from the input spectra, transmits the Table-181
//! matrix-input carriers `(M, ·)` via `two_channel_data()` and the side
//! prediction residual `(S − g·M, ·)` via the residual `sf_data` pair.
//!
//! Measurable claim pinned here: for a surround pair correlated with its
//! front carrier (`Ls = κ·L`) the optimal projection `g* = (1−κ)/(1+κ)`
//! collapses the transmitted residual to (near-)silence, while the
//! identity path (round 103) spends a full sf_data body on the raw `Ls`
//! spectrum. When prediction offers no benefit (`Ls = L` ⇒ `g* = 0`) the
//! decision falls back to header-only `SapMode::None` rows and the
//! emitted frame is bit-for-bit identical to the identity path.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::mch::FiveXCodecMode;
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

/// The SAP-auto encoder emits a v2 IMS frame with 5_X_codec_mode =
/// ASPX_ACPL_1 that the decoder walks end-to-end into a 5-channel
/// interleaved S16 AudioFrame.
#[test]
fn encode_5_0_acpl1_sap_produces_5_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.2);
    // Surrounds correlated with their fronts: Ls = 0.5·L, Rs = 0.25·R.
    let ls: Vec<f32> = l.iter().map(|v| v * 0.5).collect();
    let rs: Vec<f32> = r.iter().map(|v| v * 0.25).collect();
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_sap(&[&l, &r, &c, &ls, &rs]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(
        af.data[0].len(),
        1920 * 5 * 2,
        "5-channel S16 interleaved PCM expected"
    );
}

/// The decoder recovers the decision-derived SAP-coded chparam rows
/// (`sap_mode == 3`) on both residual slots for correlated surround
/// input, and the transmitted residual energy collapses versus the
/// identity path's raw-`Ls` residual on the same input.
#[test]
fn acpl1_sap_recovers_sap_rows_and_shrinks_residual_vs_identity() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.2);
    let ls: Vec<f32> = l.iter().map(|v| v * 0.5).collect();
    let rs: Vec<f32> = r.iter().map(|v| v * 0.25).collect();
    let frames: [&[f32]; 5] = [&l, &r, &c, &ls, &rs];

    // SAP-auto path.
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec_sap = Ac4Decoder::new(&params);
    let mut enc_sap = Ac4ImsEncoder::new();
    let bytes_sap = enc_sap.encode_frame_pcm_5_0_acpl1_sap(&frames);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes_sap);
    dec_sap.send_packet(&pkt).expect("send_packet");
    let _ = dec_sap.receive_frame().expect("receive_frame");
    let sub = dec_sap.last_substream.as_ref().expect("substream parsed");
    assert_eq!(sub.tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl1));
    let cp0 = sub.tools.acpl_1_residual_chparam[0]
        .as_ref()
        .expect("residual chparam[0] recovered");
    let cp1 = sub.tools.acpl_1_residual_chparam[1]
        .as_ref()
        .expect("residual chparam[1] recovered");
    assert_eq!(cp0.sap_mode, 3, "pair 0 must carry the SAP-coded row");
    assert_eq!(cp1.sap_mode, 3, "pair 1 must carry the SAP-coded row");
    let (_tl, s3_sap) = sub.tools.acpl_1_residual_pair[0]
        .as_ref()
        .expect("SAP residual pair[0]");
    let e_sap: f64 = s3_sap.iter().map(|v| (*v as f64) * (*v as f64)).sum();

    // Identity (round-103) path on the same input.
    let mut dec_id = Ac4Decoder::new(&params);
    let mut enc_id = Ac4ImsEncoder::new();
    let bytes_id = enc_id.encode_frame_pcm_5_0_acpl1(&frames);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes_id);
    dec_id.send_packet(&pkt).expect("send_packet");
    let _ = dec_id.receive_frame().expect("receive_frame");
    let sub_id = dec_id.last_substream.as_ref().expect("substream parsed");
    let id_cp0 = sub_id.tools.acpl_1_residual_chparam[0]
        .as_ref()
        .expect("identity chparam[0]");
    assert_eq!(id_cp0.sap_mode, 0, "identity path keeps SapMode::None");
    let (_tl, s3_id) = sub_id.tools.acpl_1_residual_pair[0]
        .as_ref()
        .expect("identity residual pair[0]");
    let e_id: f64 = s3_id.iter().map(|v| (*v as f64) * (*v as f64)).sum();

    assert!(
        e_id > 0.0,
        "identity path must transmit non-silent raw Ls residual"
    );
    assert!(
        e_sap < 0.1 * e_id,
        "SAP residual energy must collapse vs identity: e_sap = {e_sap}, e_id = {e_id}"
    );
}

/// When SAP prediction offers no benefit (`Ls = L`, `Rs = R` ⇒ zero
/// side energy ⇒ `g* = 0` on every band) the decision falls back to
/// header-only `SapMode::None` rows and the emitted frame is
/// bit-for-bit identical to the round-103 identity path.
#[test]
fn acpl1_sap_no_benefit_input_matches_identity_path_bytes() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.2);
    let frames: [&[f32]; 5] = [&l, &r, &c, &l, &r];

    let mut enc_sap = Ac4ImsEncoder::new();
    let bytes_sap = enc_sap.encode_frame_pcm_5_0_acpl1_sap(&frames);
    let mut enc_id = Ac4ImsEncoder::new();
    let bytes_id = enc_id.encode_frame_pcm_5_0_acpl1(&frames);
    assert_eq!(
        bytes_sap, bytes_id,
        "no-SAP-band input must encode bit-identically to the identity path"
    );
}

/// The SAP-auto encoder advances `sequence_counter` per call like every
/// other encode entry point.
#[test]
fn encode_5_0_acpl1_sap_advances_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let f0 = enc.encode_frame_pcm_5_0_acpl1_sap(&[&z, &z, &z, &z, &z]);
    let f1 = enc.encode_frame_pcm_5_0_acpl1_sap(&[&z, &z, &z, &z, &z]);
    assert_ne!(f0, f1, "sequence counter must advance between frames");
}
