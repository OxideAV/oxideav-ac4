//! Round 193 — 5_X SIMPLE/ASPX_ACPL_3 encoder with real per-parameter-
//! band β1 / β2 extraction from L / R carrier energies.
//!
//! ### Background
//!
//! The round-95 ASPX_ACPL_3 encoder emits a structurally-valid 5-channel
//! `5_X_channel_element()` whose Table 62 `acpl_data_2ch()` body codes
//! all 11 parameter sets (α1, α2, β1, β2, β3, γ1..γ6) as zero-delta
//! Huffman codewords. With α = β = β3 = 0 and γ = 0, the §5.7.7.6.2
//! Pseudocode 118 / 119 synthesis collapses to a trivial mix that
//! produces silent surround outputs from non-silent carrier inputs —
//! the encoder is structurally correct but perceptually inert.
//!
//! Real per-band β1 / β2 extraction (this round) is the first step
//! toward perceptually-meaningful ASPX_ACPL_3 reconstruction. With
//! α1 = α2 = 0 and β3 = 0 the ACplModule2 (Pseudocode 119) for the
//! first parameter set reduces to
//!
//! ```text
//!   z0[ts][sb] = 0.5 · ( x0[ts][sb]·g1 + x1[ts][sb]·g2 + y0[ts][sb]·β1 )
//!   z1[ts][sb] = 0.5 · ( x0[ts][sb]·g1 + x1[ts][sb]·g2 − y0[ts][sb]·β1 )
//! ```
//!
//! and analogously `(z2, z3)` with β2 driving the second ACplModule2
//! instance. Non-zero β1 / β2 inject the decorrelator output `y` into
//! the surround reconstruction, producing the energy-bearing
//! decorrelated content that gives Ls / Rs their spaciousness.
//!
//! ### What this round measures
//!
//! These tests sweep the round-193
//! `encode_frame_pcm_5_0_acpl3_real_beta` /
//! `encode_frame_pcm_5_1_acpl3_real_beta` paths and assert:
//!
//!   1. Round-trip — the encoder's output is accepted by `Ac4Decoder`
//!      and yields a 5-channel (5.0) or 6-channel (5.1) AudioFrame.
//!   2. Silent input → β_q = 0 — the extractor short-circuits to zero
//!      indices, exactly matching the round-95 zero-delta scaffold for
//!      that codebook position.
//!   3. Non-silent input + non-zero `beta_scale` → at least one β_q
//!      lane is non-zero in each of β1, β2.
//!   4. `beta_scale = 0.0` is byte-for-byte identical to the round-95
//!      `encode_frame_pcm_5_0_acpl3` output — pinned regression invariant
//!      so the real-β path stays a strict superset of the structural
//!      scaffold.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::extract_beta_q_per_band_carrier_energy;
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

fn make_silence_frame() -> Vec<f32> {
    vec![0.0_f32; N]
}

/// The real-β 5.0 encoder produces a frame that the decoder accepts and
/// decodes to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_beta_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl3_real_beta(&[&l, &r, &c], 0.1);
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
    assert_eq!(
        af.data[0].len(),
        1920 * 5 * 2,
        "5-channel S16 interleaved PCM expected"
    );
}

/// The real-β 5.1 encoder produces a frame that the decoder accepts and
/// decodes to a 6-channel AudioFrame (5.0 + LFE).
#[test]
fn encode_5_1_acpl3_real_beta_round_trips_to_6_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let lfe = make_tone_frame(60.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_1_acpl3_real_beta(&[&l, &r, &c, &lfe], 0.1);
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

/// Silent carrier inputs short-circuit `extract_beta_q_per_band_carrier_energy`
/// to all-zero β_q — the per-band energy is zero so the extractor returns
/// the codebook origin index (0) for every band.
#[test]
fn extract_beta_silent_carrier_yields_all_zero_indices() {
    use oxideav_ac4::acpl::AcplQuantMode;
    let silence = vec![0.0_f32; N];
    let beta_q = extract_beta_q_per_band_carrier_energy(
        &silence,
        1920,
        7, // num_param_bands_id = 3 → 7 bands
        0,
        0.1,
        AcplQuantMode::Fine,
    );
    assert_eq!(beta_q.len(), 7);
    for (pb, &q) in beta_q.iter().enumerate() {
        assert_eq!(q, 0, "expected β_q[{pb}] = 0 for silent carrier, got {q}");
    }
}

/// A tone-bearing carrier with `scale > 0` yields at least one non-zero
/// β_q lane (the band containing the tone's MDCT peak crosses the
/// codebook's first lane boundary).
#[test]
fn extract_beta_tonal_carrier_with_non_zero_scale_yields_non_zero_indices() {
    use oxideav_ac4::acpl::AcplQuantMode;
    use oxideav_ac4::encoder_mdct::EncoderMdctState;
    // Use a strong tone so the per-band energy comfortably crosses the
    // first BETA Fine lane boundary (mid-point between 0.0 and 0.2375 =
    // 0.11875 → β_q = 1 when scale·√E > 0.11875).
    let tone = make_tone_frame(440.0, 0.6);
    let mut mdct = EncoderMdctState::new(1920);
    let coeffs = mdct.analyse_frame(&tone);

    let beta_q =
        extract_beta_q_per_band_carrier_energy(&coeffs, 1920, 7, 0, 0.3, AcplQuantMode::Fine);
    assert_eq!(beta_q.len(), 7);
    let total_non_zero: usize = beta_q.iter().filter(|&&q| q > 0).count();
    assert!(
        total_non_zero >= 1,
        "expected at least one non-zero β_q lane for a tonal carrier, got {beta_q:?}"
    );
}

/// `beta_scale = 0.0` reproduces the round-95 zero-delta scaffold
/// byte-for-byte. This pins the real-β path as a strict superset of the
/// structural scaffold — any future encoder change that perturbs this
/// invariant must be caught explicitly.
#[test]
fn encode_5_0_acpl3_real_beta_with_zero_scale_matches_round95_scaffold() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);

    let bytes_real_beta = enc_a.encode_frame_pcm_5_0_acpl3_real_beta(&[&l, &r, &c], 0.0);
    let bytes_minimal = enc_b.encode_frame_pcm_5_0_acpl3(&[&l, &r, &c]);

    assert_eq!(
        bytes_real_beta,
        bytes_minimal,
        "beta_scale = 0.0 must match the round-95 scaffold byte-for-byte; \
         lengths: real_beta = {}, minimal = {}",
        bytes_real_beta.len(),
        bytes_minimal.len()
    );
}

/// Silent inputs at any `beta_scale` produce a frame identical to the
/// round-95 scaffold — the carrier-energy extractor returns 0 in every
/// band when the carriers carry no signal.
#[test]
fn encode_5_0_acpl3_real_beta_with_silent_inputs_matches_scaffold() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let s = make_silence_frame();

    let bytes_real_beta = enc_a.encode_frame_pcm_5_0_acpl3_real_beta(&[&s, &s, &s], 0.3);
    let bytes_minimal = enc_b.encode_frame_pcm_5_0_acpl3(&[&s, &s, &s]);

    assert_eq!(
        bytes_real_beta, bytes_minimal,
        "silent inputs must produce scaffold-identical output regardless of beta_scale"
    );
}

/// Non-silent inputs at `beta_scale > 0` diverge from the round-95
/// scaffold (different bytes) — the new β1 / β2 codewords occupy
/// different bit-positions than the zero-delta DF codewords.
#[test]
fn encode_5_0_acpl3_real_beta_with_tonal_inputs_diverges_from_scaffold() {
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.6);
    let r = make_tone_frame(440.0, 0.6);
    let c = make_tone_frame(660.0, 0.6);

    let bytes_real_beta = enc_a.encode_frame_pcm_5_0_acpl3_real_beta(&[&l, &r, &c], 0.3);
    let bytes_minimal = enc_b.encode_frame_pcm_5_0_acpl3(&[&l, &r, &c]);

    assert_ne!(
        bytes_real_beta, bytes_minimal,
        "real-β path must differ from scaffold for tonal inputs at non-zero beta_scale"
    );
    // Padded substream so byte length stays equal (both paths use the
    // same `pad_target_bytes`); only the body-bit content changes.
    assert_eq!(
        bytes_real_beta.len(),
        bytes_minimal.len(),
        "padded substream length must match the scaffold's"
    );
}
