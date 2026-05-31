//! Round 196 — 5_X SIMPLE/ASPX_ACPL_3 encoder with real per-parameter-
//! band α₁ / α₂ extraction from the L↔R carrier cross-correlation,
//! layered on top of the round-193 real β₁ / β₂ extraction.
//!
//! ### Background
//!
//! The round-193 ASPX_ACPL_3 encoder
//! ([`Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_beta`]) lifted the
//! `acpl_beta_1_dq` / `acpl_beta_2_dq` layers from the round-95
//! zero-delta scaffold to carrier-energy-driven magnitudes; α₁ / α₂
//! still emitted zero-delta codewords. With α = 0 in every band the
//! Pseudocode 119 ACplModule2 reduces to
//!
//! ```text
//!   z0 = 0.5·(γ·x0 + γ·x1) + 0.5·β·y0
//!   z1 = 0.5·(γ·x0 + γ·x1) − 0.5·β·y0
//! ```
//!
//! i.e. the dry mix is split evenly between front and back.
//!
//! Round 196 promotes α₁ / α₂ to **real per-band L↔R correlation**:
//! `α[pb] = α_scale · ρ(L, R)[pb]` clamped to the ALPHA_DQ table bound.
//! The ACplModule2 then becomes
//!
//! ```text
//!   z0 = 0.5·(1+α)·(γ·x0 + γ·x1) + 0.5·β·y0
//!   z1 = 0.5·(1−α)·(γ·x0 + γ·x1) − 0.5·β·y0
//! ```
//!
//! so mono-like (highly-correlated) bands push α toward +1 — biasing
//! more dry energy to the front pair — and decorrelated bands stay near
//! α = 0.
//!
//! ### What this round measures
//!
//! 1. Round-trip — the encoder's output is accepted by `Ac4Decoder` and
//!    yields a 5-channel (5.0) AudioFrame.
//! 2. Identity case — `L = R` yields per-band ρ = +1 → α = α_scale · 1,
//!    quantising to the lane closest to `α_scale` (clamped to ±2.0).
//! 3. Anti-correlation case — `L = −R` yields ρ = −1 → α = −α_scale.
//! 4. `α_scale = 0.0 ∧ β_scale = 0.0` is byte-for-byte identical to the
//!    round-95 zero-delta scaffold
//!    ([`Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3`]) — pinned
//!    regression invariant.

use oxideav_ac4::acpl::AcplQuantMode;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::extract_alpha_q_per_band_carrier_correlation;
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

/// The real-α/β 5.0 encoder produces a frame that the decoder accepts
/// and decodes to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_alpha_beta_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl3_real_alpha_beta(&[&l, &r, &c], 0.5, 0.1);
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

/// Perfect L = R correlation yields ρ = +1 → α_q saturates toward the
/// positive lane. With `α_scale = 1.0` the analytic α is 1.0 per band,
/// which `quantise_alpha` (nearest-neighbour against ALPHA_DQ_FINE)
/// resolves to lane 24 = 1.0 → `α_q = 24 − 16 = 8`.
#[test]
fn extract_alpha_perfect_l_r_correlation_yields_positive_alpha_q() {
    let tone = make_tone_frame(440.0, 0.4);
    let mut mdct = EncoderMdctState::new(1920);
    let coeffs = mdct.analyse_frame(&tone);
    // L = R: feed the same MDCT spectrum into both slots.
    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        &coeffs,
        &coeffs,
        1920,
        7, // num_param_bands_id = 3 → 7 bands
        0,
        1.0,
        AcplQuantMode::Fine,
    );
    assert_eq!(alpha_q.len(), 7);
    // Every band with non-zero E[L²]·E[R²] must land at lane 24
    // (= 1.0) → α_q = +8. Bands with zero energy (silence outside the
    // tone's spectral support) return 0.
    let tone_active: Vec<i32> = alpha_q.iter().copied().filter(|&q| q != 0).collect();
    assert!(
        !tone_active.is_empty(),
        "expected at least one band with non-zero α_q for a tonal input, got {alpha_q:?}"
    );
    for q in &tone_active {
        assert_eq!(
            *q, 8,
            "perfect ρ = +1 with α_scale = 1.0 must quantise to α_q = +8 (lane 24 = 1.0); got {alpha_q:?}"
        );
    }
}

/// Anti-correlation `L = −R` yields ρ = −1 → α saturates toward the
/// negative lane. With `α_scale = 1.0` the analytic α is −1.0 per band,
/// quantising to lane 8 = −1.0 → `α_q = 8 − 16 = −8`.
#[test]
fn extract_alpha_perfect_anti_correlation_yields_negative_alpha_q() {
    let tone = make_tone_frame(440.0, 0.4);
    let mut mdct_l = EncoderMdctState::new(1920);
    let mut mdct_r = EncoderMdctState::new(1920);
    let coeffs_l = mdct_l.analyse_frame(&tone);
    let anti: Vec<f32> = tone.iter().map(|&x| -x).collect();
    let coeffs_r = mdct_r.analyse_frame(&anti);

    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        &coeffs_l,
        &coeffs_r,
        1920,
        7,
        0,
        1.0,
        AcplQuantMode::Fine,
    );
    let active: Vec<i32> = alpha_q.iter().copied().filter(|&q| q != 0).collect();
    assert!(
        !active.is_empty(),
        "expected at least one band with non-zero α_q for an anti-correlated input, got {alpha_q:?}"
    );
    for q in &active {
        assert_eq!(
            *q, -8,
            "perfect ρ = −1 with α_scale = 1.0 must quantise to α_q = −8 (lane 8 = −1.0); got {alpha_q:?}"
        );
    }
}

/// `α_scale = 0.0` AND `β_scale = 0.0` reproduces the round-95
/// zero-delta scaffold byte-for-byte. This pins the real-α/β path as a
/// strict superset of the structural scaffold.
#[test]
fn encode_5_0_acpl3_real_alpha_beta_with_zero_scales_matches_round95_scaffold() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);

    let bytes_real = enc_a.encode_frame_pcm_5_0_acpl3_real_alpha_beta(&[&l, &r, &c], 0.0, 0.0);
    let bytes_minimal = enc_b.encode_frame_pcm_5_0_acpl3(&[&l, &r, &c]);

    assert_eq!(
        bytes_real, bytes_minimal,
        "α_scale = β_scale = 0.0 must match the round-95 scaffold byte-for-byte"
    );
}
