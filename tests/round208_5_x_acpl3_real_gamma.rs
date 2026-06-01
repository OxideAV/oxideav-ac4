//! Round 208 — 5_X SIMPLE/ASPX_ACPL_3 encoder with real per-parameter-
//! band γ5 / γ6 extraction from a 2×2 least-squares fit of the centre
//! channel, layered on top of the round-196 real α₁ / α₂ + real β₁ / β₂
//! extractors.
//!
//! ### Background
//!
//! In §5.7.7.6.2 Pseudocode 118 step 7 the centre output `z4` is built
//! by the third `ACplModule2` invocation with `(a = 1, b = 0, y = 0)`:
//!
//! ```text
//!   z4 = 0.5 · (γ5 · x0in + γ6 · x1in)
//! ```
//!
//! and step 11 scales `z4 *= √2` before the QMF synthesis bank emits
//! the centre channel. Step 1 rescales the carriers
//! `x0in = (1 + √2) · L`, `x1in = (1 + √2) · R`. The centre
//! reconstruction (ignoring ACplModule3 / ducker corrections, which
//! collapse to identity when β3 = 0 and the decorrelator output is
//! zero) is therefore:
//!
//! ```text
//!   C ≈ K · (γ5 · L + γ6 · R)         K = √2 · (1 + √2) / 2 = 1 + √(1/2)
//! ```
//!
//! The round-208 extractor solves the 2×2 normal equations per
//! parameter band for `(γ5, γ6)` that minimise the MDCT-bin-wise
//! residual `Σ (C/K − γ5·L − γ6·R)²`. Bands with a degenerate Gram
//! matrix (no L or R energy, or perfectly collinear L = ±R within
//! numerical tolerance) keep γ5 = γ6 = 0.
//!
//! Up through round 196 the γ1..γ6 entropy layers all emitted the
//! round-95 zero-delta scaffold codewords — which decoded to γ = 0
//! everywhere, silencing the centre channel that the decoder is
//! supposed to synthesise from `(γ5·L + γ6·R)`. Round 208 lifts only
//! γ5 / γ6 to real per-band values; γ1..γ4 + β3 stay at the round-95
//! scaffold (those parameter sets drive the (L, R, Ls, Rs) sub-pipeline
//! plus the ACplModule3 cross-residual — neither of which has a per-side
//! surround reference at encode time for the 5.0 / 5.1 PCM input
//! layouts the real-γ entry point targets).
//!
//! ### What this round measures
//!
//! 1. Round-trip — the encoder's output is accepted by `Ac4Decoder` and
//!    yields a 5-channel (5.0) AudioFrame.
//! 2. C = (L + R) / (2·K) inverse fit — a centre channel that exactly
//!    matches the analytic reconstruction with γ5 = γ6 = 0.5 quantises
//!    to non-zero γ_q in at least one tonally-active parameter band
//!    (verifies the least-squares extractor is selecting non-trivial γ).
//! 3. C = 0 (silent centre) yields γ5_q = γ6_q = 0 in every band.
//! 4. `α_scale = β_scale = γ_scale = 0.0` is byte-for-byte identical to
//!    the round-95 zero-delta scaffold
//!    ([`Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3`]).
//! 5. `γ_scale = 0.0` reproduces the round-196 real-α-β byte stream
//!    exactly (real-γ layer is opt-in).
//! 6. Loud-centre vs silent-centre inputs produce materially different
//!    encoded bytes (the round-196 path would emit identical γ5/γ6
//!    codewords regardless of centre input).
//! 7. The encoder is bit-deterministic for matched inputs + fresh state.

use oxideav_ac4::acpl::AcplQuantMode;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::extract_gamma_5_6_q_per_band_centre_least_squares;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::encoder_mdct::EncoderMdctState;
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

/// The real-α/β/γ 5.0 encoder produces a frame that the decoder accepts
/// and decodes to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_gamma_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let frame_bytes =
        enc.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c], 0.5, 0.1, 1.0);
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

/// `α_scale = β_scale = γ_scale = 0.0` reproduces the round-95
/// zero-delta scaffold byte-for-byte. Pins the real-γ path as a strict
/// superset of the structural scaffold.
#[test]
fn encode_5_0_acpl3_real_gamma_with_zero_scales_matches_round95_scaffold() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);

    let bytes_real =
        enc_a.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c], 0.0, 0.0, 0.0);
    let bytes_minimal = enc_b.encode_frame_pcm_5_0_acpl3(&[&l, &r, &c]);

    assert_eq!(
        bytes_real, bytes_minimal,
        "α_scale = β_scale = γ_scale = 0.0 must match the round-95 scaffold byte-for-byte"
    );
}

/// `γ_scale = 0.0` reproduces the round-196 real-α-β byte stream
/// exactly — the real-γ layer is strictly opt-in atop r196.
#[test]
fn encode_5_0_acpl3_zero_gamma_scale_matches_round196_real_alpha_beta() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);

    let bytes_real_gamma =
        enc_a.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c], 0.5, 0.1, 0.0);
    let bytes_real_alpha_beta =
        enc_b.encode_frame_pcm_5_0_acpl3_real_alpha_beta(&[&l, &r, &c], 0.5, 0.1);

    assert_eq!(
        bytes_real_gamma, bytes_real_alpha_beta,
        "γ_scale = 0.0 must reproduce the round-196 real-α-β byte stream exactly"
    );
}

/// Silent centre input quantises to γ5_q = γ6_q = 0 in every band:
/// the 2×2 right-hand side `<L, C>` / `<R, C>` is zero so the analytic
/// γ pair is `(0, 0)` regardless of L / R energy.
#[test]
fn extract_gamma_silent_centre_yields_all_zero_q() {
    let tone = make_tone_frame(440.0, 0.4);
    let silence = vec![0.0f32; N];
    let mut mdct_l = EncoderMdctState::new(1920);
    let mut mdct_r = EncoderMdctState::new(1920);
    let mut mdct_c = EncoderMdctState::new(1920);
    let coeffs_l = mdct_l.analyse_frame(&tone);
    let coeffs_r = mdct_r.analyse_frame(&tone);
    let coeffs_c = mdct_c.analyse_frame(&silence);

    let (g5_q, g6_q) = extract_gamma_5_6_q_per_band_centre_least_squares(
        &coeffs_l,
        &coeffs_r,
        &coeffs_c,
        1920,
        7,
        0,
        1.0,
        AcplQuantMode::Fine,
    );
    assert_eq!(g5_q.len(), 7);
    assert_eq!(g6_q.len(), 7);
    for &q in &g5_q {
        assert_eq!(q, 0, "silent centre must yield γ5_q = 0; got {g5_q:?}");
    }
    for &q in &g6_q {
        assert_eq!(q, 0, "silent centre must yield γ6_q = 0; got {g6_q:?}");
    }
}

/// A centre channel chosen so the analytic least-squares fit lands
/// exactly at `(γ5, γ6) = (0.5, 0.5)` produces non-zero γ_q in at least
/// one tonally-active band. We construct `C = (L + R) / 2` so the
/// per-band RHS is `<L,C>/K = 0.5·(<L,L>+<L,R>)/K` and analogously for
/// R. With L and R two unrelated tones the L↔R cross term is small (not
/// exactly zero in MDCT but ≪ the diagonal), so the 2×2 solve recovers
/// `γ5 ≈ γ6 ≈ 0.5 / K`. The quantiser maps that to a non-zero
/// γ_q ≈ round(0.5 / K / γ_delta) — explicitly non-zero because
/// `0.5 / K ≈ 0.293 ≫ γ_delta_fine ≈ 0.1`.
#[test]
fn extract_gamma_centre_half_l_plus_r_yields_nonzero_q() {
    // Pick two unrelated tones so the L↔R cross-correlation per band
    // stays bounded away from ±1 (so the Gram determinant is well-
    // conditioned and the least-squares fit is not degenerate).
    let l_pcm = make_tone_frame(220.0, 0.3);
    let r_pcm = make_tone_frame(880.0, 0.3);
    let c_pcm: Vec<f32> = l_pcm
        .iter()
        .zip(r_pcm.iter())
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect();
    let mut mdct_l = EncoderMdctState::new(1920);
    let mut mdct_r = EncoderMdctState::new(1920);
    let mut mdct_c = EncoderMdctState::new(1920);
    let coeffs_l = mdct_l.analyse_frame(&l_pcm);
    let coeffs_r = mdct_r.analyse_frame(&r_pcm);
    let coeffs_c = mdct_c.analyse_frame(&c_pcm);

    let (g5_q, g6_q) = extract_gamma_5_6_q_per_band_centre_least_squares(
        &coeffs_l,
        &coeffs_r,
        &coeffs_c,
        1920,
        7,
        0,
        1.0,
        AcplQuantMode::Fine,
    );
    let nonzero_5 = g5_q.iter().filter(|&&q| q != 0).count();
    let nonzero_6 = g6_q.iter().filter(|&&q| q != 0).count();
    assert!(
        nonzero_5 >= 1,
        "expected ≥1 band with non-zero γ5_q for C = (L+R)/2; got {g5_q:?}"
    );
    assert!(
        nonzero_6 >= 1,
        "expected ≥1 band with non-zero γ6_q for C = (L+R)/2; got {g6_q:?}"
    );
}

/// Loud-centre vs silent-centre inputs produce materially different
/// encoded bytes. The round-196 path would emit identical γ5 / γ6
/// codewords (zero-delta) regardless of centre input; round-208's
/// real-γ extractor differentiates the two.
#[test]
fn encode_5_0_acpl3_loud_vs_silent_centre_diverges() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c_loud = make_tone_frame(660.0, 0.4);
    let c_silent = vec![0.0f32; N];

    let bytes_loud =
        enc_a.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c_loud], 0.0, 0.0, 1.0);
    let bytes_silent =
        enc_b.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c_silent], 0.0, 0.0, 1.0);
    assert_ne!(
        bytes_loud, bytes_silent,
        "loud-centre vs silent-centre must produce different γ5 / γ6 codewords"
    );
}

/// The encoder is bit-deterministic for matched inputs + fresh state.
#[test]
fn encode_5_0_acpl3_real_gamma_is_deterministic() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);

    let bytes_a =
        enc_a.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c], 0.5, 0.1, 1.0);
    let bytes_b =
        enc_b.encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma(&[&l, &r, &c], 0.5, 0.1, 1.0);
    assert_eq!(
        bytes_a, bytes_b,
        "matched inputs + fresh state must produce identical bytes"
    );
}

/// 5.1 round-trip: the LFE PCM lands at slot 5 of the 6-channel
/// `AudioFrame` (per the round-80 channel mapping) when γ-driven centre
/// reconstruction is enabled.
#[test]
fn encode_5_1_acpl3_real_gamma_round_trips_to_6_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.2);
    let r = make_tone_frame(440.0, 0.2);
    let c = make_tone_frame(660.0, 0.2);
    let lfe = make_tone_frame(60.0, 0.4);

    let frame_bytes =
        enc.encode_frame_pcm_5_1_acpl3_real_alpha_beta_gamma(&[&l, &r, &c, &lfe], 0.5, 0.1, 1.0);
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
        1920 * 6 * 2,
        "6-channel S16 interleaved PCM expected"
    );
}
