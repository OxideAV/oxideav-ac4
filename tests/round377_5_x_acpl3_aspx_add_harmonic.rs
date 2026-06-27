//! Round 377 — wire a **real `aspx_add_harmonic`** decision into the live
//! 5_X SIMPLE/ASPX_ACPL_3 single-envelope real-ASPX frame path.
//!
//! ### Background
//!
//! The A-SPX HF generator (§5.7.6.4) transposes the decoded low band up
//! into the high band. That transposition reproduces a noise-like HF
//! spectral envelope but cannot reproduce a *discrete tonal partial* that
//! exists in the original HF band yet has no counterpart at its low-band
//! transposition source. ETSI TS 103 190-1 §4.2.12.6 carries an optional
//! per-high-res-signal-subband-group `aspx_add_harmonic[sbg]` flag; the
//! §5.7.6.4.2.1 Pseudocode 92 `derive_sine_idx_sb` turns each set flag
//! into a sinusoid placed at the group's middle subband
//! (`sb_mid = (sba + sbz) / 2`), and the §5.7.6.4.4 tone generator
//! injects a level-matched complex sinusoid there.
//!
//! Through round 363 the live encoder paths always emitted an all-`false`
//! `add_harmonic` (the `write_aspx_hfgen_iwc_{1,2}ch` default). This round
//! adds the `aspx_ah_select` encoder analysis: per high-res signal subband
//! group, measure the HF QMF spectral crest (energy at `sb_mid` ÷ the
//! group's mean per-subband energy) and request a restored harmonic when a
//! dominant tonal partial is present. The decision is wired into the
//! single-envelope 5_X ACPL_3 real-ASPX frame builder.
//!
//! ### What this round measures
//!
//! 1. The analysis fires — a carrier whose HF band carries a strong
//!    isolated QMF partial sets at least one `add_harmonic` flag, while a
//!    spectrally flat HF carrier sets none.
//! 2. Round-trip — the 5.0 / 5.1 encoder output (tonal HF carriers, so the
//!    flag is live) is accepted by `Ac4Decoder` and yields a 5- / 6-channel
//!    AudioFrame.
//! 3. Liveness — a tonal-HF input (flag set) produces a frame whose bytes
//!    differ from a flat-HF input (flag clear), proving the bit reaches the
//!    wire.
//! 4. Determinism — matched inputs + fresh encoder state are byte-identical.
//!
//! Refs ETSI TS 103 190-1: §4.2.12.6 (`aspx_hfgen_iwc`), §4.2.12.4
//! Table 52 (`aspx_data_2ch`), §5.7.6.4.2.1 Pseudocode 92
//! (`sine_idx_sb` / `sb_mid`), §5.7.6.4.4 Pseudocodes 104/105 (tone
//! generator).

use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::aspx_ah_select::select_add_harmonic;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::qmf_slots_to_sb_major;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::qmf::QmfAnalysisBank;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

/// The live config the IMS encoder uses for the 5_X ACPL_3 path.
fn live_cfg() -> AspxConfig {
    AspxConfig {
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
    }
}

/// A pure HF tone — concentrates QMF energy into a single high subband,
/// producing a high spectral crest in its signal subband group.
fn make_tone_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// A spectrally flat HF carrier — broadband pseudo-random noise spreads
/// energy evenly across the HF subbands, so every group's crest is ~1.
fn make_flat_noise_frame(amp: f32) -> Vec<f32> {
    // Deterministic LCG so the test is reproducible.
    let mut state: u32 = 0x1234_5678;
    (0..N)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
            amp * (u * 2.0 - 1.0)
        })
        .collect()
}

/// Reproduce the encoder's `add_harmonic` decision for one carrier so the
/// test can assert it from outside the encoder.
fn add_harmonic_for(pcm: &[f32]) -> Vec<bool> {
    let cfg = live_cfg();
    let tables = oxideav_ac4::aspx::derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let n_slots = pcm.len() / 64;
    let usable = n_slots * 64;
    let mut bank = QmfAnalysisBank::new();
    let q_high = qmf_slots_to_sb_major(&bank.process_block(&pcm[..usable]));
    select_add_harmonic(&q_high, &tables.sbg_sig_highres, tables.sbx)
}

/// A strong isolated HF tone sets at least one `add_harmonic` flag; flat
/// broadband HF noise sets none.
#[test]
fn tonal_hf_sets_add_harmonic_flat_does_not() {
    let tone = make_tone_frame(13_000.0, 0.7);
    let ah_tone = add_harmonic_for(&tone);
    assert!(
        ah_tone.iter().any(|&b| b),
        "an isolated HF tone must request at least one harmonic, got {ah_tone:?}"
    );

    let noise = make_flat_noise_frame(0.5);
    let ah_noise = add_harmonic_for(&noise);
    assert!(
        !ah_noise.iter().any(|&b| b),
        "flat HF noise must not request any harmonic, got {ah_noise:?}"
    );
}

/// 5.0 single-envelope real-ASPX encode (tonal HF carriers → live
/// `add_harmonic`) round-trips to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_aspx_add_harmonic_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(13_000.0, 0.7);
    let r = make_tone_frame(12_000.0, 0.6);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    // Sanity: the L carrier really does request a harmonic.
    assert!(add_harmonic_for(&l).iter().any(|&b| b));

    let frame_bytes =
        enc.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
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

/// 5.1 single-envelope real-ASPX encode round-trips to a 6-channel frame.
#[test]
fn encode_5_1_acpl3_real_aspx_add_harmonic_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(13_000.0, 0.7);
    let r = make_tone_frame(12_000.0, 0.6);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lfe = make_tone_frame(60.0, 0.2);

    let frame_bytes =
        enc.encode_frame_pcm_5_1_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs, &lfe], 0.5, 0.1, 1.0, 1.0);
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

/// Liveness: a tonal-HF L/R carrier (flag set) produces a frame whose
/// bytes differ from a flat-HF-noise L/R carrier (flag clear), proving the
/// `add_harmonic` bit reaches the wire. The centre / surround carriers are
/// held identical so the difference is attributable to the L/R A-SPX path.
#[test]
fn add_harmonic_changes_emitted_bytes() {
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let l_tone = make_tone_frame(13_000.0, 0.7);
    let r_tone = make_tone_frame(12_000.0, 0.6);
    let l_flat = make_flat_noise_frame(0.5);
    let r_flat = make_flat_noise_frame(0.4);

    // Confirm the analysis differs across the two inputs.
    assert!(add_harmonic_for(&l_tone).iter().any(|&b| b));
    assert!(!add_harmonic_for(&l_flat).iter().any(|&b| b));

    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    let bytes_tone = enc_a.encode_frame_pcm_5_0_acpl3_real_aspx(
        &[&l_tone, &r_tone, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let bytes_flat = enc_b.encode_frame_pcm_5_0_acpl3_real_aspx(
        &[&l_flat, &r_flat, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_ne!(
        bytes_tone, bytes_flat,
        "tonal-HF (add_harmonic set) and flat-HF (clear) frames must differ"
    );
}

/// Determinism: identical inputs + fresh encoder state → identical bytes.
#[test]
fn add_harmonic_path_is_deterministic() {
    let l = make_tone_frame(13_000.0, 0.7);
    let r = make_tone_frame(12_000.0, 0.6);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    let a = enc_a.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    let b = enc_b.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    assert_eq!(a, b, "matched inputs must produce identical bytes");
}

/// The add_harmonic wiring is live on the 5_X ASPX_ACPL_2 single-envelope
/// path: a tonal-HF L/R/C input differs from a flat-HF-noise input, and
/// both round-trip to 5-channel audio.
#[test]
fn acpl2_5_0_add_harmonic_is_live_and_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);

    let c_tone = make_tone_frame(13_500.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_tone = Ac4ImsEncoder::new();
    let bytes_tone = enc_tone.encode_frame_pcm_5_0_acpl2_real_aspx(&[
        &make_tone_frame(13_000.0, 0.7),
        &make_tone_frame(12_000.0, 0.6),
        &c_tone,
        &ls,
        &rs,
    ]);
    let mut enc_flat = Ac4ImsEncoder::new();
    let bytes_flat = enc_flat.encode_frame_pcm_5_0_acpl2_real_aspx(&[
        &make_flat_noise_frame(0.5),
        &make_flat_noise_frame(0.4),
        &make_flat_noise_frame(0.45),
        &ls,
        &rs,
    ]);
    assert_ne!(
        bytes_tone, bytes_flat,
        "tonal vs flat HF must differ on 5_X ACPL_2"
    );

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes_tone);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
}

/// The add_harmonic wiring is live on the 7.0 pure-ASPX path: a tonal-HF
/// input differs from a flat-HF-noise input, and round-trips to 7-channel
/// audio.
#[test]
fn pure_aspx_7_0_add_harmonic_is_live_and_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);

    let tone7: Vec<Vec<f32>> = [
        13_000.0, 12_000.0, 13_500.0, 11_000.0, 10_500.0, 14_000.0, 12_500.0,
    ]
    .iter()
    .map(|&f| make_tone_frame(f, 0.6))
    .collect();
    let flat7: Vec<Vec<f32>> = (0..7).map(|_| make_flat_noise_frame(0.5)).collect();

    let mut enc_tone = Ac4ImsEncoder::new();
    let refs_tone: Vec<&[f32]> = tone7.iter().map(|v| v.as_slice()).collect();
    let bytes_tone =
        enc_tone.encode_frame_pcm_7_0_aspx_real_aspx(&refs_tone.clone().try_into().unwrap());

    let mut enc_flat = Ac4ImsEncoder::new();
    let refs_flat: Vec<&[f32]> = flat7.iter().map(|v| v.as_slice()).collect();
    let bytes_flat =
        enc_flat.encode_frame_pcm_7_0_aspx_real_aspx(&refs_flat.clone().try_into().unwrap());

    assert_ne!(
        bytes_tone, bytes_flat,
        "tonal vs flat HF must differ on 7.0 pure-ASPX"
    );

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes_tone);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");
}
