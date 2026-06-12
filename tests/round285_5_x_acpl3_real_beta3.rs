//! Round 285 — 5_X SIMPLE/ASPX_ACPL_3 encoder with real per-parameter-
//! band β₃ extraction (the third-decorrelator gain), layered on top of
//! the round-215 real α₁ / α₂ + β₁ / β₂ + γ₁..γ₆ extractors. Closes the
//! round-215 "β₃ stays at the round-95 zero-delta scaffold" deferral
//! for the 5_X ACPL_3 path.
//!
//! ### Background
//!
//! In §5.7.7.6.2 Pseudocode 118 the centre output channel is
//!
//! ```text
//!   z4  = 0.5 · (γ₅·x0in + γ₆·x1in)                    (step 7, dry)
//!   z4 += 0.25 · y₂ · (−β₃ − β₃·1) = −0.5 · β₃ · y₂    (step 10, wet)
//!   C   = √2 · z4                                       (step 11)
//! ```
//!
//! where `y₂` is the ducked third-decorrelator output driven by
//! `v₃ = (γ₁+γ₃+γ₅)·x0in + (γ₂+γ₄+γ₆)·x1in` (step 2). `y₂` itself is
//! unobservable at encode time (it is decoder-side decorrelator
//! state), but its *energy* is — the decorrelator + ducker chain is
//! energy-preserving in steady state, so `E[y₂²] ≈ E[v₃²]` is fully
//! determined by the carrier spectra and the quantised γ matrix the
//! encoder is emitting. Energy-matching the wet centre contribution
//! `0.5 · β₃² · E[y₂²]` against the dry-fit residual
//! `E_res = Σ (C − K·(γ₅·L + γ₆·R))²` (with `K = 1 + √(1/2)`, using
//! the quantised γ₅ / γ₆ the decoder will actually apply) gives the
//! encoder decision `β₃ = √(2 · E_res / E[v₃²])`, quantised per
//! Table 207 (`beta3_q = round(β₃ / beta3_delta)`, `±cb_off` clamp).
//!
//! ### What this round measures
//!
//! 1. Round-trip — the real-β₃ encoder's output is accepted by
//!    `Ac4Decoder` and yields a 5-channel (5.0) AudioFrame.
//! 2. Round-trip — 5.1 input yields a 6-channel AudioFrame.
//! 3. Decode-side β₃ recovery — a builder-level body parses back
//!    through `parse_5x_audio_data_outer` and the recovered
//!    `acpl_data_2ch().beta3` layer differential-decodes (Pseudocode
//!    121) to exactly the per-band `beta3_q` row the extractor probe
//!    computes for the same inputs.
//! 4. `beta3_scale = 0.0` reproduces the round-215 full-γ IMS byte
//!    stream exactly (real β₃ is opt-in; the all-zero β₃ row emits the
//!    zero-delta scaffold codewords).
//! 5. A centre carrying content the γ dry mix cannot capture produces
//!    different encoded bytes with `beta3_scale = 1.0` than with
//!    `beta3_scale = 0.0` (the β₃ layer is live on the wire).
//! 6. The encoder is bit-deterministic for matched inputs + fresh
//!    state.

use oxideav_ac4::acpl::AcplQuantMode;
use oxideav_ac4::acpl_synth::{differential_decode, AcplDiffState};
use oxideav_ac4::asf::SubstreamTools;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{
    build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3,
    extract_beta3_q_per_band_centre_residual, extract_gamma_1_2_q_per_band_surround_least_squares,
    extract_gamma_3_4_q_per_band_surround_least_squares,
    extract_gamma_5_6_q_per_band_centre_least_squares,
};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::mch::{parse_5x_audio_data_outer, FiveXCodecMode};
use oxideav_core::bits::BitReader;
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

/// The real-α/β/γ₁..γ₆/β₃ 5.0 encoder produces a frame that the
/// decoder accepts and decodes to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_beta3_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2, "5-channel S16 interleaved");
}

/// The 5.1 entry point round-trips to a 6-channel AudioFrame.
#[test]
fn encode_5_1_acpl3_real_beta3_round_trips_to_6_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lfe = make_tone_frame(60.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_1_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs, &lfe],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 6 * 2, "6-channel S16 interleaved");
}

/// Builder-level decode-side pin: the emitted `acpl_data_2ch()` β₃
/// layer parses back through the decoder's Table-25 ASPX_ACPL_3 walker
/// and Pseudocode-121 differential decoding to exactly the per-band
/// `beta3_q` row the extractor computes for the same spectra.
#[test]
fn acpl3_real_beta3_body_recovers_beta3_q_through_decoder_walker() {
    let tl = 1920u32;
    let nb = 7u32; // acpl_num_param_bands_id = 3 → 7 bands.
    let qm = AcplQuantMode::Fine;

    // Deterministic synthetic spectra with an uncaptured centre (the
    // centre carries content in bins the L/R dry mix cannot model).
    let l: Vec<f32> = (0..tl as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let r: Vec<f32> = (0..tl as usize)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.07)
        .collect();
    let c: Vec<f32> = (0..tl as usize)
        .map(|i| if i % 5 == 1 { 1.5 } else { -0.6 })
        .collect();
    let ls: Vec<f32> = (0..tl as usize)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.08)
        .collect();
    let rs: Vec<f32> = (0..tl as usize)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.06)
        .collect();

    // Expected β₃ row — same extraction chain the builder runs.
    let (g1_q, g2_q) =
        extract_gamma_1_2_q_per_band_surround_least_squares(&l, &r, &ls, tl, nb, 0, 1.0, qm);
    let (g3_q, g4_q) =
        extract_gamma_3_4_q_per_band_surround_least_squares(&l, &r, &rs, tl, nb, 0, 1.0, qm);
    let (g5_q, g6_q) =
        extract_gamma_5_6_q_per_band_centre_least_squares(&l, &r, &c, tl, nb, 0, 1.0, qm);
    let expected_beta3_q = extract_beta3_q_per_band_centre_residual(
        &l, &r, &c, &g1_q, &g2_q, &g3_q, &g4_q, &g5_q, &g6_q, tl, nb, 0, 1.0, qm, qm,
    );
    assert!(
        expected_beta3_q.iter().any(|&q| q != 0),
        "fixture must exercise a non-zero β₃ row: {expected_beta3_q:?}"
    );

    let aspx_cfg = oxideav_ac4::aspx::AspxConfig {
        quant_mode_env: oxideav_ac4::aspx::AspxQuantStep::Fine,
        start_freq: 0,
        stop_freq: 0,
        master_freq_scale: oxideav_ac4::aspx::AspxMasterFreqScale::LowRes,
        interpolation: false,
        preflat: false,
        limiter: false,
        noise_sbg: 0,
        num_env_bits_fixfix: 0,
        freq_res_mode: oxideav_ac4::aspx::AspxFreqResMode::DurationDependent,
    };
    let body = build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3(
        tl,
        40,
        None,
        true,
        &l,
        &r,
        Some(&c),
        Some(&ls),
        Some(&rs),
        None,
        &aspx_cfg,
        3,
        qm,
        qm,
        0.5,
        0.1,
        1.0,
        1.0,
        8192,
    );

    // The body leads with the 2-byte ac4_substream() audio_size header;
    // the 5_X channel-element walker starts right after it.
    let mut br = BitReader::new(&body[2..]);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, true, tl).expect("walker");
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl3));
    let data = tools
        .acpl_data_2ch
        .as_ref()
        .expect("acpl_data_2ch parsed from the I-frame body");

    let mut state = AcplDiffState::new();
    let rows = differential_decode(&data.beta3, nb, &mut state);
    assert_eq!(rows.len(), 1, "one parameter set");
    assert_eq!(
        rows[0], expected_beta3_q,
        "decoder-recovered beta3_q must match the encoder's extraction"
    );
}

/// `beta3_scale = 0.0` reproduces the round-215 real-α/β/full-γ IMS
/// byte stream exactly — real β₃ is opt-in.
#[test]
fn encode_5_0_acpl3_beta3_zero_scale_matches_round215_bytes() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_legacy = Ac4ImsEncoder::new();
    let legacy = enc_legacy.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
    );
    let mut enc_new = Ac4ImsEncoder::new();
    let with_beta3_off = enc_new.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        0.0,
    );
    assert_eq!(legacy, with_beta3_off);
}

/// A centre carrying content the γ dry mix cannot capture flips β₃
/// live on the wire: `beta3_scale = 1.0` bytes differ from
/// `beta3_scale = 0.0` bytes for the same input.
#[test]
fn encode_5_0_acpl3_beta3_layer_is_live_for_uncaptured_centre() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    // Centre tone far from both carriers — the per-band dry fit leaves
    // a non-trivial residual in the centre-active bands.
    let c = make_tone_frame(3000.0, 0.4);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_off = Ac4ImsEncoder::new();
    let bytes_off = enc_off.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        0.0,
    );
    let mut enc_on = Ac4ImsEncoder::new();
    let bytes_on = enc_on.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_eq!(bytes_off.len(), bytes_on.len(), "padding-equalised sizes");
    assert_ne!(
        bytes_off, bytes_on,
        "β₃ = real must change the on-wire acpl_data_2ch() payload"
    );
}

/// The encoder is bit-deterministic for matched inputs + fresh state.
#[test]
fn encode_5_0_acpl3_real_beta3_is_deterministic() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(3000.0, 0.4);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_a = Ac4ImsEncoder::new();
    let a = enc_a.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let mut enc_b = Ac4ImsEncoder::new();
    let b = enc_b.encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_eq!(a, b);
}
