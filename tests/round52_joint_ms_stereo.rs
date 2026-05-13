//! Round 52 — joint M/S stereo CPE (Path B,
//! `b_enable_mdct_stereo_proc == 1`) integration tests.
//!
//! Per ETSI TS 103 190-1 §5.3 (channel_count > 1) + §4.2.6.3 Table 22
//! (`stereo_data()`) + §7.5 (Pseudocode 77 joint stereo): the encoder
//! routes a stereo PCM pair through the joint M/S CPE path when the
//! per-SFB average cross-channel correlation exceeds
//! `Ac4ImsEncoder::STEREO_JOINT_MS_CORRELATION_THRESHOLD` (0.7), else it
//! stays on the round-51 split-MDCT path (Path A: 2× SCE).
//!
//! Test coverage:
//!   1. Matched L=R 440 Hz tone — heuristic picks joint M/S, M-channel
//!      q-target bump pushes spectral SNR ≥ 28 dB.
//!   2. Independent L=440/R=660 — heuristic picks Path A (split-MDCT),
//!      decoded SNR matches the round-51 baseline (≥ 24 dB).
//!   3. Half-correlated stereo (L vs (L+R)/2 image) — joint path, SNR
//!      ≥ 26 dB.
//!   4. Round-trip through `Ac4Decoder` — codec accepts the joint
//!      bitstream cleanly (parsed substream's
//!      `tools.mdct_stereo_proc == true` + both per-channel scaled
//!      spectra are populated).

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::encoder_mdct::EncoderMdctState;
use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};

/// Spectral SNR between two MDCT-spectrum slices, comparing bin-for-bin
/// up to the shorter of the two lengths.
fn spectral_snr_db(orig: &[f32], recon: &[f32]) -> f64 {
    let mut sig_e = 0.0_f64;
    let mut err_e = 0.0_f64;
    let n_compare = orig.len().min(recon.len());
    for k in 0..n_compare {
        let o = orig[k] as f64;
        let r = recon[k] as f64;
        sig_e += o * o;
        err_e += (o - r) * (o - r);
    }
    10.0 * (sig_e / err_e.max(1e-30)).log10()
}

/// Steady-state probe — (input L/R MDCT spectra, decoded L/R scaled
/// spectra, whether the dispatcher picked Path B for the last frame).
type SteadyStateProbe = ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>), bool);

/// Encode + decode `frames_lr` through the dispatcher (auto Path A vs
/// Path B). Returns the steady-state (index 2) decoded primary +
/// secondary scaled MDCT spectra plus the matching encoder-side input
/// MDCT spectra mirrored from the same inputs.
fn encode_decode_stereo_collect_steady_state(
    frames_lr: &[(Vec<f32>, Vec<f32>)],
) -> SteadyStateProbe {
    let n = frames_lr[0].0.len() as u32;
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let mut mdct_l = EncoderMdctState::new(n);
    let mut mdct_r = EncoderMdctState::new(n);
    let mut last_in_l: Vec<f32> = Vec::new();
    let mut last_in_r: Vec<f32> = Vec::new();
    let mut last_pri: Vec<f32> = Vec::new();
    let mut last_sec: Vec<f32> = Vec::new();
    let mut last_mdct_stereo_proc = false;
    for (l, r) in frames_lr {
        last_in_l = mdct_l.analyse_frame(l);
        last_in_r = mdct_r.analyse_frame(r);
        let bytes = enc.encode_frame_pcm_stereo(l, r);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let _ = dec.receive_frame().expect("receive_frame");
        let sub = dec.last_substream.as_ref().expect("substream");
        last_mdct_stereo_proc = sub.tools.mdct_stereo_proc;
        last_pri = sub.tools.scaled_spec_primary.clone().expect("pri spec");
        last_sec = sub.tools.scaled_spec_secondary.clone().expect("sec spec");
    }
    (
        (last_in_l, last_in_r),
        (last_pri, last_sec),
        last_mdct_stereo_proc,
    )
}

/// Fixture: 440 Hz sine at amplitude `amp` starting at sample index
/// `start`, evaluated at 48 kHz.
fn sine_frame(freq: f32, amp: f32, start: usize, n: usize) -> Vec<f32> {
    let fs = 48_000.0_f32;
    (0..n)
        .map(|i| {
            let t = (start + i) as f32 / fs;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Test 1: matched 440 Hz L=R. The joint M/S heuristic picks Path B
/// (`b_enable_mdct_stereo_proc == 1`); the M-channel carries virtually
/// all the joint signal while S is near-silent, so the encoder's
/// round-52 `q_target` bump tightens the M quantisation step and the
/// decoder reconstructs both L and R at ≥ 28 dB spectral SNR — beating
/// round-51's 24.8 dB on this fixture.
#[test]
fn round52_matched_stereo_joint_ms_snr_exceeds_28db() {
    let n = 1920usize;
    let frames_lr: Vec<(Vec<f32>, Vec<f32>)> = (0..3)
        .map(|i| {
            let f = sine_frame(440.0, 0.3, i * n, n);
            (f.clone(), f)
        })
        .collect();
    let ((in_l, _in_r), (pri, sec), used_joint) =
        encode_decode_stereo_collect_steady_state(&frames_lr);
    assert!(
        used_joint,
        "matched L=R fixture should route through Path B (joint M/S)"
    );
    let snr_l = spectral_snr_db(&in_l, &pri);
    let snr_r = spectral_snr_db(&in_l, &sec);
    eprintln!(
        "ROUND-52 matched 440 Hz L=R joint M/S spectral SNR: SNR_L = {snr_l:.1} dB, SNR_R = {snr_r:.1} dB"
    );
    eprintln!(
        "  bin 34 in_l={:.2} pri={:.2}; bin 35 in_l={:.2} pri={:.2}; bin 36 in_l={:.2} pri={:.2}",
        in_l.get(34).unwrap_or(&0.0),
        pri.get(34).unwrap_or(&0.0),
        in_l.get(35).unwrap_or(&0.0),
        pri.get(35).unwrap_or(&0.0),
        in_l.get(36).unwrap_or(&0.0),
        pri.get(36).unwrap_or(&0.0),
    );
    assert!(
        snr_l > 28.0,
        "L channel spectral SNR too low: {snr_l:.1} dB (expected > 28 dB)"
    );
    assert!(
        snr_r > 28.0,
        "R channel spectral SNR too low: {snr_r:.1} dB (expected > 28 dB)"
    );
}

/// Test 2: independent 440 Hz L + 660 Hz R. The two channels' MDCT
/// spectra concentrate energy in disjoint SFBs → the round-52 energy-
/// weighted correlation falls well below the 0.7 dispatch threshold and
/// the heuristic routes the frame through Path A (split-MDCT, 2× SCE),
/// preserving the round-51 SNR floor (≥ 24 dB per channel) for
/// uncorrelated content.
#[test]
fn round52_independent_stereo_routes_via_split_path_a() {
    let n = 1920usize;
    let frames_lr: Vec<(Vec<f32>, Vec<f32>)> = (0..3)
        .map(|i| {
            (
                sine_frame(440.0, 0.3, i * n, n),
                sine_frame(660.0, 0.3, i * n, n),
            )
        })
        .collect();
    let ((in_l, in_r), (pri, sec), used_joint) =
        encode_decode_stereo_collect_steady_state(&frames_lr);
    assert!(
        !used_joint,
        "independent L=440/R=660 fixture should route through Path A (split-MDCT)"
    );
    let snr_l = spectral_snr_db(&in_l, &pri);
    let snr_r = spectral_snr_db(&in_r, &sec);
    eprintln!(
        "ROUND-52 independent 440 L + 660 R split-MDCT spectral SNR: SNR_L = {snr_l:.1} dB, SNR_R = {snr_r:.1} dB"
    );
    assert!(
        snr_l > 24.0,
        "L (440 Hz) channel spectral SNR too low: {snr_l:.1} dB (expected > 24 dB)"
    );
    assert!(
        snr_r > 24.0,
        "R (660 Hz) channel spectral SNR too low: {snr_r:.1} dB (expected > 24 dB)"
    );
}

/// Test 3: half-correlated stereo — L = amp1·sin(440), R = amp2·sin(440)
/// with amp1 ≠ amp2 (level imbalance, no phase or frequency shift).
/// M = (L+R)/2 carries the average tone; S = (L-R)/2 carries the level
/// imbalance. With amp1 = 0.3, amp2 = 0.36 the S energy is roughly 1%
/// of M's, so the frame-level "matched-channels" gate triggers the
/// q_target bump (frame_e_s/(e_m+e_s) ≈ 0.003 → q_target ≈ 15.9). Both
/// L and R reconstructions clear ≥ 26 dB spectral SNR — between the
/// pure-matched (28+) and the fully-independent (24+) regimes.
#[test]
fn round52_half_correlated_stereo_joint_ms_snr_exceeds_26db() {
    let n = 1920usize;
    let frames_lr: Vec<(Vec<f32>, Vec<f32>)> = (0..3)
        .map(|i| {
            let l = sine_frame(440.0, 0.30, i * n, n);
            let r = sine_frame(440.0, 0.36, i * n, n);
            (l, r)
        })
        .collect();
    let ((in_l, in_r), (pri, sec), used_joint) =
        encode_decode_stereo_collect_steady_state(&frames_lr);
    let snr_l = spectral_snr_db(&in_l, &pri);
    let snr_r = spectral_snr_db(&in_r, &sec);
    eprintln!(
        "ROUND-52 half-correlated stereo joint = {used_joint}, spectral SNR: SNR_L = {snr_l:.1} dB, SNR_R = {snr_r:.1} dB"
    );
    assert!(
        used_joint,
        "level-imbalanced 440 Hz fixture should route through Path B (joint M/S)"
    );
    assert!(
        snr_l > 26.0,
        "L channel spectral SNR too low: {snr_l:.1} dB (expected > 26 dB)"
    );
    assert!(
        snr_r > 26.0,
        "R channel spectral SNR too low: {snr_r:.1} dB (expected > 26 dB)"
    );
}

/// Test 4: round-trip through `Ac4Decoder` — the joint-MDCT bitstream
/// the encoder emits is consumed cleanly by the existing decoder and
/// produces structurally-valid stereo S16 PCM (1920 samples × 2 ch ×
/// 2 bytes interleaved, both channels non-silent with peaks reflecting
/// the input level). Forces the joint path so the test isn't dependent
/// on the correlation-heuristic dispatch.
#[test]
fn round52_joint_ms_full_pcm_roundtrip_through_ac4decoder() {
    let n = 1920usize;
    let frames_lr: Vec<(Vec<f32>, Vec<f32>)> = (0..5)
        .map(|i| {
            let f = sine_frame(440.0, 0.3, i * n, n);
            (f.clone(), f)
        })
        .collect();

    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let mut decoded_pcm_pairs: Vec<(Vec<i16>, Vec<i16>)> = Vec::new();
    for (l, r) in &frames_lr {
        // Force the joint M/S path so this test always exercises Path B.
        let bytes = enc.encode_frame_pcm_stereo_joint_with_max_sfb(l, r, 40);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let frame = dec.receive_frame().expect("receive_frame");
        let oxideav_core::Frame::Audio(af) = frame else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.data.len(), 1);
        // Stereo S16 interleaved: 1920 × 2 ch × 2 bytes = 7680.
        assert_eq!(af.data[0].len(), 1_920 * 2 * 2);
        let buf = &af.data[0];
        let mut pcm_l: Vec<i16> = Vec::with_capacity(1_920);
        let mut pcm_r: Vec<i16> = Vec::with_capacity(1_920);
        for i in 0..1_920usize {
            let off_l = i * 4;
            let off_r = off_l + 2;
            pcm_l.push(i16::from_le_bytes([buf[off_l], buf[off_l + 1]]));
            pcm_r.push(i16::from_le_bytes([buf[off_r], buf[off_r + 1]]));
        }
        decoded_pcm_pairs.push((pcm_l, pcm_r));
        // Decoder must surface the joint-MDCT flag.
        let sub = dec.last_substream.as_ref().expect("substream");
        assert!(
            sub.tools.mdct_stereo_proc,
            "decoder must see b_enable_mdct_stereo_proc == 1 on forced-joint frame"
        );
        assert!(sub.tools.scaled_spec_primary.is_some());
        assert!(sub.tools.scaled_spec_secondary.is_some());
    }
    // Use frame index 2 (steady state — frames 0,1 lose half a window
    // each to the encoder's TDAC startup).
    let (pcm_l, pcm_r) = &decoded_pcm_pairs[2];
    let nz_l = pcm_l.iter().filter(|&&s| s != 0).count();
    let nz_r = pcm_r.iter().filter(|&&s| s != 0).count();
    // i32 peak so abs() doesn't overflow on i16::MIN.
    let peak_l = pcm_l.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0);
    let peak_r = pcm_r.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0);
    assert!(nz_l > 100, "L too few non-zero samples: {nz_l}");
    assert!(nz_r > 100, "R too few non-zero samples: {nz_r}");
    // Input amplitude 0.3 → expected peak ≈ 9830 i16. The lossy
    // encoder/decoder pipeline must stay comfortably above 1000.
    assert!(peak_l > 1000, "L peak too low: {peak_l}");
    assert!(peak_r > 1000, "R peak too low: {peak_r}");
    // The encoder forced Path B per frame; the encoder must have emitted
    // a single substream per frame (no oversize panic, no truncation).
}
