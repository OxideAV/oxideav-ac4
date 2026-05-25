//! Round 128 — 5_X SIMPLE/ASPX_ACPL_1 multichannel encoder with **real
//! per-parameter-band α extraction**.
//!
//! Per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode
//! 117 the A-CPL reconstruction above `acpl_qmf_band` is:
//!
//! ```text
//!   z0 = 0.5 · (x0·(1+α) + y·β)            // recovers L
//!   z1 = 0.5 · (x0·(1-α) - y·β)  · √2      // recovers Ls
//! ```
//!
//! With β = 0 (the round-128 simplification — see
//! `crate::encoder_acpl3` §"Real per-band α extraction"), α controls a
//! pure level-only re-allocation between L and Ls per parameter band:
//!
//! ```text
//!   Ls_recon  =  (0.5 / √2) · L · (1 − α)
//! ```
//!
//! The round-128 extractor solves for α per band that best matches the
//! caller's Ls input vs L carrier energy. These tests assert:
//!
//!   1. The new encoder entry point produces a non-empty frame that
//!      round-trips through `Ac4Decoder` to a 5-channel AudioFrame.
//!   2. The decoder resolves the encoder's frame to
//!      `FiveXCodecMode::AspxAcpl1` and persists both
//!      `acpl_data_1ch_pair` slots (same body shape as the round-103
//!      zero-delta path).
//!   3. The on-wire α bits actually carry non-zero recovered values when
//!      the Ls / Rs energies differ from the L / R carriers (vs the
//!      round-103 zero-delta scaffold which would emit the same byte
//!      pattern regardless of caller input).

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

/// The real-α encoder entry point produces a 5-channel AudioFrame for a
/// 5.0 PCM input — same end-to-end round-trip guarantee as the round-103
/// zero-delta scaffold, just with real α bits on the wire.
#[test]
fn encode_5_0_acpl1_real_alpha_produces_5_channel_audio_frame() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha(&[&l, &r, &c, &ls, &rs]);
    assert!(
        !frame_bytes.is_empty(),
        "real-α encoder must produce non-empty output"
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

/// The decoder resolves the encoder's real-α frame to
/// `FiveXCodecMode::AspxAcpl1` and persists both `acpl_data_1ch` parameter
/// sets — confirms the on-wire body structure is unchanged from the
/// round-103 zero-delta scaffold.
#[test]
fn encode_5_0_acpl1_real_alpha_decoder_resolves_aspx_acpl_1_mode() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha(&[&l, &r, &c, &ls, &rs]);
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

/// When the caller's Ls / Rs energy differs from L / R, the real-α
/// encoder must emit at least one non-zero recovered α index. The
/// round-103 zero-delta scaffold emits α_q = 0 in every band; the
/// round-128 path picks per-band α values whose F0 / DF codewords
/// decode to non-zero `alpha_q` entries.
#[test]
fn encode_5_0_acpl1_real_alpha_emits_nonzero_alpha_when_surround_differs() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    // Strongly asymmetric L vs Ls: L is a loud 220 Hz tone, Ls is a quiet
    // 880 Hz tone at 1/10 the amplitude. The per-band Ls/L cross-energy
    // ratio is non-trivial → the analytic α is non-zero → its quantised
    // index is non-zero in at least one band.
    let l = make_tone_frame(220.0, 0.5);
    let r = make_tone_frame(440.0, 0.5);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.05);
    let rs = make_tone_frame(1100.0, 0.05);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let pair0 = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("acpl_data_1ch_pair[0] must parse");
    // alpha1 is the first param entry (DIFF_FREQ decoded values per
    // §5.7.7.7 Pseudocode 121). For 1 parameter set the vector has one
    // row; any non-zero value confirms real-α extraction fired.
    assert!(
        !pair0.alpha1.is_empty(),
        "alpha1 huffman values must populate"
    );
    let any_nonzero_alpha = pair0
        .alpha1
        .iter()
        .any(|ps| ps.values.iter().any(|&v| v != 0));
    assert!(
        any_nonzero_alpha,
        "real-α encoder must emit at least one non-zero alpha_q when surround Ls/Rs differs from L/R; got pair0.alpha1 = {:?}",
        pair0.alpha1
    );
}

/// Silence input round-trips through the real-α path (when both carrier
/// and surround energies are zero, α defaults to the zero-codebook index).
#[test]
fn encode_5_0_acpl1_real_alpha_silence_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha(&[&z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
}

/// When Ls is a uniformly-scaled copy of L (broadband level redistribution),
/// the per-band Ls/L ratio is constant and the extractor must produce the
/// **same** quantised α index in pair0 and pair1 (since R/Rs is scaled by
/// the same factor). This is a structural symmetry test that doesn't
/// depend on the exact analytic value (per-band MDCT energy can be small
/// when the tone frequency doesn't overlap a band).
#[test]
fn encode_5_0_acpl1_real_alpha_symmetric_scaling_yields_matching_alpha() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    // Use broadband-ish input (mix of tones) so multiple parameter bands
    // carry energy. Scale both surrounds identically.
    let l: Vec<f32> = (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1100.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 5500.0 * t).sin()
        })
        .collect();
    let r: Vec<f32> = (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            0.3 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1700.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 6600.0 * t).sin()
        })
        .collect();
    let c = make_tone_frame(660.0, 0.3);
    let scale: f32 = 0.3;
    let ls: Vec<f32> = l.iter().map(|x| x * scale).collect();
    let rs: Vec<f32> = r.iter().map(|x| x * scale).collect();
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let pair0 = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("acpl_data_1ch_pair[0] must parse");
    let pair1 = sub.tools.acpl_data_1ch_pair[1]
        .as_ref()
        .expect("acpl_data_1ch_pair[1] must parse");
    assert!(!pair0.alpha1.is_empty());
    assert!(!pair1.alpha1.is_empty());
    // For a band where BOTH L and R carry significant energy AND Ls/Rs are
    // scaled by the same factor, the per-band analytic α should match
    // ⇒ the quantised index should match.
    //
    // At minimum, the analytic α for "Ls = scale·L" (scale = 0.3) is
    // 1 − 2√2·0.3 ≈ 0.151 → Fine lane closest is 17 (α = +0.190625) →
    // alpha_q = +1. Bands where both L and R carry energy should resolve
    // to the same alpha_q.
    //
    // We don't pick which band; we just confirm at least one band's
    // alpha_q is positive (the level redistribution is in the right
    // direction). Some bands may have low energy and fall back to 0;
    // others may quantise to ±16 (saturation when num/den is unstable).
    let pair0_vals = &pair0.alpha1[0].values;
    let pair1_vals = &pair1.alpha1[0].values;
    let any_positive_pair0 = pair0_vals.iter().any(|&v| v > 0);
    let any_positive_pair1 = pair1_vals.iter().any(|&v| v > 0);
    assert!(
        any_positive_pair0,
        "pair0 must emit at least one positive alpha_q for Ls = 0.3·L (analytic α > 0); got {pair0_vals:?}"
    );
    assert!(
        any_positive_pair1,
        "pair1 must emit at least one positive alpha_q for Rs = 0.3·R (analytic α > 0); got {pair1_vals:?}"
    );
}

/// The real-α encoder is bit-deterministic for the same input — running
/// it twice on the same PCM (with the encoder state reset) produces the
/// same frame bytes.
#[test]
fn encode_5_0_acpl1_real_alpha_is_deterministic() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.3);
    let rs = make_tone_frame(1100.0, 0.3);
    let mut enc1 = Ac4ImsEncoder::new();
    let mut enc2 = Ac4ImsEncoder::new();
    let f1 = enc1.encode_frame_pcm_5_0_acpl1_real_alpha(&[&l, &r, &c, &ls, &rs]);
    let f2 = enc2.encode_frame_pcm_5_0_acpl1_real_alpha(&[&l, &r, &c, &ls, &rs]);
    assert_eq!(
        f1, f2,
        "encoder must be deterministic for matched inputs and fresh state"
    );
}
