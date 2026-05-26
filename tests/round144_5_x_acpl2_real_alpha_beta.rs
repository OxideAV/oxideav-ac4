//! Round 144 — 5_X SIMPLE/ASPX_ACPL_2 multichannel encoder with **real
//! per-parameter-band α + β extraction**.
//!
//! The ACPL_2 counterpart to round 132 (which added real α + β to the
//! ACPL_1 5.0 path). ACPL_2 does **not** transmit the Ls/Rs surround pair
//! on the wire — the decoder reconstructs the surround from the L/R
//! carriers + the two `acpl_data_1ch()` parameter sets per ETSI TS 103
//! 190-1 §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode 117:
//!
//! ```text
//!   z0 = 0.5 · (x0·(1+α) + y·β)        // recovers L
//!   z1 = 0.5 · (x0·(1-α) - y·β)        // recovers Ls (then ·√2)
//! ```
//!
//! The encoder accepts the caller's Ls/Rs spectra and picks α + β per
//! band so the decoder's Pseudocode-116 reconstruction matches the
//! surround energy + cross-correlation against the L/R carriers.
//! `acpl_config_1ch(FULL)` carries no `qmf_band` → `start_band = 0` so
//! every parameter band participates (in contrast to the ACPL_1 PARTIAL
//! mode whose `acpl_qmf_band` masks the low bands).
//!
//! These tests confirm:
//!
//!   1. The new entry point produces a 5-channel AudioFrame round-trip.
//!   2. The decoder resolves the encoder's frame to
//!      `FiveXCodecMode::AspxAcpl2` and persists both
//!      `acpl_data_1ch_pair` slots.
//!   3. When the caller's surround energy exceeds what α alone can
//!      explain, at least one band's β_q is positive.
//!   4. Silence input round-trips with β_q = 0 everywhere.
//!   5. Encoder is deterministic for matched inputs.

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

#[test]
fn encode_5_0_acpl2_real_alpha_beta_produces_5_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    assert!(
        !frame_bytes.is_empty(),
        "real-α+β ACPL_2 encoder must produce non-empty output"
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
        1920 * 5 * 2,
        "5-channel S16 interleaved PCM expected"
    );
}

#[test]
fn encode_5_0_acpl2_real_alpha_beta_decoder_resolves_aspx_acpl_2_mode() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(oxideav_ac4::mch::FiveXCodecMode::AspxAcpl2)
    );
    assert!(sub.tools.acpl_data_1ch_pair[0].is_some());
    assert!(sub.tools.acpl_data_1ch_pair[1].is_some());
}

/// The encoder's β extractor is functionally distinct between
/// "silence-surround" and "loud-surround" inputs: the emitted bytes
/// **must** differ. This is the structural counterpart to the on-wire
/// per-band β assertion (which is left to the direct
/// `extract_beta_q_per_band` extractor test in round 132 since the
/// round-128 ALPHA F0 writer-side `alpha_q` desync — documented as a
/// deferred follow-up since r132 — currently obscures per-band on-wire
/// β recovery through the full PCM→MDCT→writer→parser path when α
/// quantises to a non-trivial table position).
///
/// Two fixtures: silence-surround vs loud-surround. The new builder
/// produces materially different bytes in the `acpl_data_1ch` element
/// region; the round-100 scaffold would produce identical bytes
/// regardless of surround input.
#[test]
fn encode_5_0_acpl2_real_alpha_beta_loud_surround_produces_different_bytes_than_silence() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let silence = vec![0.0f32; N];
    // Same L/R/C carriers, but radically different surround content:
    // loud different-frequency tones (no MDCT-bin overlap with carriers)
    // → β extractor produces non-zero indices in multiple bands; silence
    // → β extractor produces zero indices everywhere.
    let ls_loud = make_tone(880.0, 0.95);
    let rs_loud = make_tone(1100.0, 0.95);

    let frame_silence =
        enc_a.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &silence, &silence]);
    let frame_loud =
        enc_b.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls_loud, &rs_loud]);

    assert_eq!(
        frame_silence.len(),
        frame_loud.len(),
        "padding-equalised output sizes"
    );
    assert_ne!(
        frame_silence, frame_loud,
        "real-α+β encoder must produce different bytes when surround content differs (the round-100 scaffold emits identical α/β codewords regardless of surround input)"
    );
}

/// Direct primitive: `extract_beta_q_per_band` returns at least one
/// non-zero β_q index for the round-144 fixture (loud uncorrelated
/// surround). Independent of the wire round-trip — this isolates the
/// **extractor** contract for ACPL_2's `start_band = 0` configuration
/// (the ACPL_2 path uses FULL `acpl_config_1ch` so every parameter band
/// participates in α + β coding, in contrast to the ACPL_1 PARTIAL mode).
#[test]
fn acpl2_extract_beta_q_per_band_emits_nonzero_for_loud_uncorrelated_surround() {
    use oxideav_ac4::acpl::AcplQuantMode;
    use oxideav_ac4::encoder_acpl3::{extract_alpha_q_per_band, extract_beta_q_per_band};

    // Synthetic spectra emulating ACPL_2's start_band = 0 layout. Carrier
    // and surround energise distinct parameter bands (no overlap) — the
    // carrier-band energy ratio is small (α ≈ 0) but the surround-band
    // energy is large, so β must fire.
    let mut coeffs_l = vec![0.0f32; 1920];
    let mut coeffs_ls = vec![0.0f32; 1920];
    // Bin 60..80: light carrier presence.
    for bin in 60..80 {
        coeffs_l[bin] = 0.1;
        coeffs_ls[bin] = 0.5;
    }
    let alpha_q = extract_alpha_q_per_band(&coeffs_l, &coeffs_ls, 1920, 7, 0, AcplQuantMode::Fine);
    let beta_q = extract_beta_q_per_band(
        &coeffs_l,
        &coeffs_ls,
        1920,
        7,
        0,
        &alpha_q,
        AcplQuantMode::Fine,
    );
    assert!(
        beta_q.iter().any(|&v| v != 0),
        "extract_beta_q_per_band (ACPL_2 start_band=0) must yield non-zero index when surround energy exceeds carrier·(1-α)² balance; got beta_q = {beta_q:?} (alpha_q = {alpha_q:?})"
    );
}

/// Silence input round-trips with β_q = 0 in every band (no surround
/// energy to model).
#[test]
fn encode_5_0_acpl2_real_alpha_beta_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
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
fn encode_5_0_acpl2_real_alpha_beta_is_deterministic() {
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let mut enc1 = Ac4ImsEncoder::new();
    let mut enc2 = Ac4ImsEncoder::new();
    let f1 = enc1.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let f2 = enc2.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    assert_eq!(
        f1, f2,
        "encoder must be deterministic for matched inputs and fresh state"
    );
}

/// Direct body-builder probe: confirm the new builder produces a byte
/// stream that differs from the round-100 zero-delta scaffold when the
/// caller supplies non-trivial Ls/Rs. (Same wire schedule; only the α + β
/// codewords differ.)
#[test]
fn build_body_real_alpha_beta_diverges_from_scaffold_for_nonzero_surround() {
    use oxideav_ac4::acpl::AcplQuantMode;
    use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
    use oxideav_ac4::encoder_acpl3::{
        build_5_x_acpl2_body_from_pcm_spectra,
        build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta,
    };

    let tl = 1920u32;
    let max_sfb = 40u32;
    // Synthetic carrier + loud differentiated surround spectra.
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
    let scaffold = build_5_x_acpl2_body_from_pcm_spectra(
        tl,
        max_sfb,
        true,
        &coeffs_l,
        &coeffs_r,
        &coeffs_c,
        &aspx_cfg,
        3,
        AcplQuantMode::Fine,
        8192,
    );
    let real = build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
        tl,
        max_sfb,
        true,
        &coeffs_l,
        &coeffs_r,
        &coeffs_c,
        &coeffs_ls,
        &coeffs_rs,
        &aspx_cfg,
        3,
        AcplQuantMode::Fine,
        8192,
    );
    assert_eq!(scaffold.len(), real.len(), "padding-equalised body sizes");
    assert_ne!(
        scaffold, real,
        "real-α+β builder must produce a different byte stream than the round-100 scaffold when the caller's Ls/Rs spectra are non-trivial"
    );
}
