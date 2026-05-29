//! Round 132 — 5_X SIMPLE/ASPX_ACPL_1 multichannel encoder with **real
//! per-parameter-band α + β extraction**.
//!
//! Per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode
//! 117, the A-CPL reconstruction above `acpl_qmf_band` is:
//!
//! ```text
//!   z0 = 0.5 · (x0·(1+α) + y·β)        // recovers L
//!   z1 = 0.5 · (x0·(1-α) - y·β)        // recovers Ls (then ·√2)
//! ```
//!
//! Round 128 extracted α while pinning β = 0 (pure level-only image).
//! Round 132 adds a per-band β magnitude derived from the surround/
//! carrier energy ratio after the level component is removed:
//!
//! ```text
//!   E[Ls²]  ≈  0.5 · E[L²] · ((1 − α)² + β²)
//!   ⇒  β = √max(0, 2·E[Ls²]/E[L²] − (1 − α)²)
//! ```
//!
//! These tests confirm:
//!
//!   1. The new entry point produces a 5-channel AudioFrame round-trip.
//!   2. The decoder resolves the encoder's frame to
//!      `FiveXCodecMode::AspxAcpl1` and persists both
//!      `acpl_data_1ch_pair` slots.
//!   3. When the caller's surround energy exceeds what α alone can
//!      explain (`E[Ls²] > 0.5·E[L²]·(1-α)²`), at least one band's
//!      β_q is positive (the round-128 path always emits β_q = 0).

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
fn encode_5_0_acpl1_real_alpha_beta_produces_5_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    assert!(
        !frame_bytes.is_empty(),
        "real-α+β encoder must produce non-empty output"
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
fn encode_5_0_acpl1_real_alpha_beta_decoder_resolves_aspx_acpl_1_mode() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(oxideav_ac4::mch::FiveXCodecMode::AspxAcpl1)
    );
    assert!(sub.tools.acpl_data_1ch_pair[0].is_some());
    assert!(sub.tools.acpl_data_1ch_pair[1].is_some());
}

/// Direct extractor probe: when the surround MDCT energy in a band
/// exceeds what α alone can model, `extract_beta_q_per_band` returns a
/// non-zero β_q in that band.
///
/// This guards the encoder's analytic primitive directly — independent
/// of the on-wire emission / decoder round-trip path. The on-wire test
/// below builds on this.
#[test]
fn extract_beta_q_emits_nonzero_for_carrier_only_input() {
    use oxideav_ac4::encoder_acpl3::{extract_alpha_q_per_band, extract_beta_q_per_band};

    // Forward-MDCT a 1500 Hz carrier and a much louder 1500 Hz surround
    // (same frequency, 90° phase shift → MDCT cross-correlation small,
    // surround energy much larger than carrier energy).
    let _l = make_tone(1500.0, 0.5);
    let _ls: Vec<f32> = (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            0.95 * (2.0 * std::f32::consts::PI * 1500.0 * t + std::f32::consts::FRAC_PI_2).sin()
        })
        .collect();
    // Skip running MDCT here — just inject synthetic spectra.
    let mut coeffs_l = vec![0.0f32; 1920];
    let mut coeffs_ls = vec![0.0f32; 1920];
    // bin 120 = sb 4 (pb 3 in 7-band) — give it strong differential.
    coeffs_l[120] = 1.0;
    coeffs_ls[120] = 5.0; // E_s/E_c = 25 → β² = 50 − (1−α)² huge.
    let alpha_q = extract_alpha_q_per_band(
        &coeffs_l,
        &coeffs_ls,
        1920,
        7,
        1,
        oxideav_ac4::acpl::AcplQuantMode::Fine,
    );
    let beta_q = extract_beta_q_per_band(
        &coeffs_l,
        &coeffs_ls,
        1920,
        7,
        1,
        &alpha_q,
        oxideav_ac4::acpl::AcplQuantMode::Fine,
    );
    // pb 3 corresponds to sb 4 → bin 120 (120 * 64 / 1920 = 4).
    // Expect a non-zero β_q at pb 3.
    assert!(
        beta_q.iter().any(|&v| v != 0),
        "extract_beta_q_per_band must yield non-zero index when surround energy ≫ carrier; got {beta_q:?} (alpha_q = {alpha_q:?})"
    );
}

/// On-wire byte-exact round-trip of the `acpl_data_1ch()` element: the
/// encoder's `write_acpl_data_1ch_real_alpha_beta_bytes` emits an
/// element carrying real per-band α + β indices, and
/// `acpl::parse_acpl_data_1ch` recovers them exactly through the F0 + DF
/// codebooks (Tables A.34 / A.35 / A.40 / A.41) + DIFF_FREQ
/// reconstruction.
///
/// This isolates the α+β coding contract from the full-substream-body
/// layout (which the structural tests above cover) — it confirms that
/// the β magnitudes the extractor produces survive the wire intact and
/// that the round-128 zero-β scaffold is genuinely superseded for the
/// non-zero case.
#[test]
fn acpl_data_1ch_real_alpha_beta_round_trips_byte_exact() {
    use oxideav_ac4::acpl::{parse_acpl_data_1ch, AcplQuantMode};
    use oxideav_ac4::encoder_acpl3::write_acpl_data_1ch_real_alpha_beta_bytes;
    use oxideav_core::bits::BitReader;

    let num_bands = 7u32;
    let start_band = 1u32;
    let qm = AcplQuantMode::Fine;
    // Bands 0 is below start; bands 1..7 carry data.
    let alpha_q = [0i32, 16, 4, -3, 0, 8, -8];
    let beta_q = [0i32, 8, 5, 0, 3, 7, 2];

    let bytes =
        write_acpl_data_1ch_real_alpha_beta_bytes(num_bands, start_band, qm, &alpha_q, &beta_q);
    assert!(!bytes.is_empty());

    let mut br = BitReader::new(&bytes);
    let parsed = parse_acpl_data_1ch(&mut br, num_bands, start_band, qm).expect("parse");

    // alpha1 / beta1 each carry one parameter set (num_param_sets = 1).
    assert_eq!(parsed.alpha1.len(), 1);
    assert_eq!(parsed.beta1.len(), 1);

    // Round 181 (spec-aligned Pseudocode 121 layout): the parser
    // returns `AcplHuffParam.values` indexed by full param-band number
    // `pb in 0..num_bands`. Positions `[0..start_band)` are zero, the
    // F0 codeword's signed `alpha_q[start_band]` lands at
    // `values[start_band]`, and DF deltas occupy
    // `[start_band+1..num_bands)`. Apply Pseudocode 121's DIFF_FREQ
    // accumulation across the full `num_bands` array and compare to the
    // encoder's per-band input directly.
    let pseudocode_121_diff_freq = |vals: &[i32]| -> Vec<i32> {
        let mut row = vec![0i32; vals.len()];
        if !vals.is_empty() {
            row[0] = vals[0];
            for i in 1..vals.len() {
                row[i] = row[i - 1] + vals[i];
            }
        }
        row
    };
    let alpha_recovered = pseudocode_121_diff_freq(&parsed.alpha1[0].values);
    let beta_recovered = pseudocode_121_diff_freq(&parsed.beta1[0].values);
    assert_eq!(
        alpha_recovered,
        alpha_q.to_vec(),
        "α indices must round-trip byte-exact at spec param-band indexing"
    );
    assert_eq!(
        beta_recovered,
        beta_q.to_vec(),
        "β indices must round-trip byte-exact at spec param-band indexing"
    );
    // Confirm β actually carries non-zero values (vs the round-128 path).
    assert!(
        beta_recovered.iter().any(|&v| v != 0),
        "test fixture must exercise non-zero β"
    );
}

/// The full PCM-path encoder produces a frame that round-trips to a
/// 5-channel AudioFrame when the surround is a loud anti-phase copy of
/// the carrier (the input that drives the extractor's β-positive
/// regime). This confirms the extra β layer doesn't break the
/// substream-body layout or the decoder's IMDCT path.
fn make_tone_phase(freq: f32, amp: f32, phase: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t + phase).sin()
        })
        .collect()
}

#[test]
fn encode_5_0_acpl1_real_alpha_beta_loud_antiphase_surround_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(1500.0, 0.3);
    let r = make_tone(1800.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone_phase(1500.0, 0.9, std::f32::consts::PI);
    let rs = make_tone_phase(1800.0, 0.9, std::f32::consts::PI);

    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(oxideav_ac4::mch::FiveXCodecMode::AspxAcpl1)
    );
}

/// Silence input round-trips through the real-α+β path — both α and
/// β default to the zero-codebook index in every band.
#[test]
fn encode_5_0_acpl1_real_alpha_beta_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
    // Silence in → β_q = 0 in every band (no surround energy to model).
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

/// When Ls is a pure scaled copy of L (β = 0 case — all surround energy
/// is α-explainable), the real-α+β path's β_q should match the round-128
/// path's β_q = 0 in every band (or close to it). This is a regression
/// guard that confirms the β extractor doesn't spuriously fire for the
/// level-only case round 128 already handles.
#[test]
fn encode_5_0_acpl1_real_alpha_beta_zero_beta_for_pure_level_only() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    // Ls = 0.3 · L (pure level redistribution — α alone should explain
    // it). With perfect quantisation and the analytic α the residual
    // is zero. In practice ibeta-column-0 quantisation introduces a
    // small bias; allow up to ~half the bands to round non-zero.
    let l = make_tone(220.0, 0.5);
    let r = make_tone(440.0, 0.5);
    let c = make_tone(660.0, 0.3);
    let scale: f32 = 0.3;
    let ls: Vec<f32> = l.iter().map(|x| x * scale).collect();
    let rs: Vec<f32> = r.iter().map(|x| x * scale).collect();
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let pair0 = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("pair0 parsed");
    // For Ls = 0.3·L, the per-band ratio is constant 0.09; the energy
    // balance gives β² = 0.18 − (1−α)². With α optimal that's negative
    // → β = 0 by the max(0, ·) clamp. We assert pair0 has at most a few
    // bands with β_q != 0 — the residual extractor is meant to fire
    // *only* when the level-only image is structurally incomplete.
    let nonzero_count: usize = pair0
        .beta1
        .iter()
        .map(|ps| ps.values.iter().filter(|&&v| v != 0).count())
        .sum();
    let total_bands: usize = pair0.beta1.iter().map(|ps| ps.values.len()).sum();
    assert!(
        nonzero_count * 2 <= total_bands,
        "pure level-only surround (Ls = 0.3·L) should yield mostly β_q = 0; \
         got {} / {} non-zero bands, beta1 = {:?}",
        nonzero_count,
        total_bands,
        pair0.beta1
    );
}

/// Encoder is bit-deterministic for matched inputs and fresh state.
#[test]
fn encode_5_0_acpl1_real_alpha_beta_is_deterministic() {
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.3);
    let rs = make_tone(1100.0, 0.3);
    let mut enc1 = Ac4ImsEncoder::new();
    let mut enc2 = Ac4ImsEncoder::new();
    let f1 = enc1.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let f2 = enc2.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    assert_eq!(
        f1, f2,
        "encoder must be deterministic for matched inputs and fresh state"
    );
}
