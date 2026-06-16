//! Round 327 — wire the **multi-envelope** (`num_env > 1`) real-ASPX
//! payload into the live 5_X SIMPLE/ASPX_ACPL_3 frame path, closing the
//! README's "the multi-envelope QMF builders … exist and round-trip in
//! isolation but are not yet wired into the frame emission" deferral for
//! the 5_X ACPL_3 path.
//!
//! ### Background
//!
//! Round 322 wired the single-envelope real-ASPX SIGNAL / NOISE writer
//! (`write_aspx_data_2ch_real_envelope`, FIXFIX `num_env = 1`) into the
//! live 5_X ACPL_3 frame builder. The round-299 multi-envelope writer
//! (`write_aspx_data_2ch_multi_envelope`), the round-316 QMF→rows builder
//! (`build_aspx_multi_envelope_2ch_from_qmf`), and the round-310 envelope
//! selector (`select_aspx_num_env_from_qmf`) have all existed and round-
//! tripped in isolation, but the live frame emission still always emitted
//! a single FIXFIX envelope.
//!
//! This round adds `encode_frame_pcm_5_{0,1}_acpl3_real_aspx_multi_env`:
//! the encoder QMF-analyses the L / R carriers, probes the L-carrier HF
//! energy for a temporal transient
//! (`select_aspx_num_env_from_qmf`), and — when one is present — splits
//! the frame into `num_env = 2` uniformly spaced FIXFIX signal envelopes,
//! emitting the multi-envelope `aspx_data_2ch()` body. Stationary frames
//! fall back to the single-envelope path, so the method is a strict
//! superset of the round-322 entry point.
//!
//! ### What this round measures
//!
//! 1. Transient drives the selector — an L carrier whose HF energy is
//!    concentrated in the second half of the frame makes
//!    `select_aspx_num_env_from_qmf` return `2`.
//! 2. Round-trip — the multi-envelope 5.0 encoder output is accepted by
//!    `Ac4Decoder` and yields a 5-channel AudioFrame.
//! 3. Round-trip — 5.1 input yields a 6-channel AudioFrame.
//! 4. Multi-envelope liveness — a transient input drives a 2-envelope
//!    FIXFIX framing whose `aspx_data_2ch()` byte region differs from the
//!    single-envelope (round-322) frame for the same input.
//! 5. Stationary fallback — a stationary input produces a frame
//!    byte-identical to the round-322 single-envelope path.
//! 6. Determinism — matched inputs + fresh encoder state produce
//!    identical bytes.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.6 Table 25, §4.2.12.4 Table 52,
//! §4.3.10.1.9 Table 123, §4.3.10.4.1, §5.7.6.3.4 / §5.7.6.3.5
//! Pseudocodes 80–83, §5.7.6.4.2.1 Pseudocodes 90–91.

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, num_aspx_timeslots, num_ts_in_ats, AspxConfig, AspxFreqResMode,
    AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{qmf_slots_to_sb_major, select_aspx_num_env_from_qmf};
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

fn make_tone_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// A transient HF carrier: a high-frequency tone gated almost entirely
/// into the second half of the frame, so its per-envelope QMF energy
/// varies strongly across the FIXFIX `num_env = 2` partition.
fn make_hf_transient_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            let gate = if i >= N / 2 { 1.0 } else { 0.02 };
            gate * amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Reproduce the encoder's `num_env` selection for an L carrier so the
/// test can assert the driver fires.
fn select_num_env_for(pcm_l: &[f32]) -> u32 {
    let cfg = live_cfg();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let frame_len = N as u32;
    let nts = num_ts_in_ats(frame_len);
    let aspx_frame_ts = num_aspx_timeslots(frame_len);
    let n_slots = pcm_l.len() / 64;
    let usable = n_slots * 64;
    let mut bank = QmfAnalysisBank::new();
    let q_high = qmf_slots_to_sb_major(&bank.process_block(&pcm_l[..usable]));
    // max_num_env = 1 << ((1 << fixfix_tmp_num_env_bits()) - 1) = 2 here.
    let max_num_env = 1u32 << ((1u32 << cfg.fixfix_tmp_num_env_bits()) - 1);
    select_aspx_num_env_from_qmf(
        &q_high,
        &tables.sbg_sig_highres,
        nts,
        aspx_frame_ts,
        tables.sbx,
        max_num_env,
        0.30,
    )
}

/// The transient HF carrier drives the selector to a 2-envelope FIXFIX
/// partition; a steady **full-frame** HF tone (uniform HF energy across
/// the partition) stays at 1.
#[test]
fn transient_drives_num_env_to_two() {
    let transient = make_hf_transient_frame(14000.0, 0.6);
    assert_eq!(
        select_num_env_for(&transient),
        2,
        "a second-half HF transient must select num_env = 2"
    );
    // A constant-amplitude HF tone spanning the whole frame has uniform
    // per-envelope HF energy → low coefficient of variation → num_env = 1.
    let steady_hf = make_tone_frame(14000.0, 0.5);
    assert_eq!(
        select_num_env_for(&steady_hf),
        1,
        "a steady full-frame HF tone must stay at num_env = 1"
    );
}

/// 5.0 multi-envelope encode round-trips to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_aspx_multi_env_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    // Transient L/R carriers so the multi-envelope path is taken.
    let l = make_hf_transient_frame(14000.0, 0.6);
    let r = make_hf_transient_frame(11000.0, 0.5);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
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

/// 5.1 multi-envelope encode round-trips to a 6-channel AudioFrame.
#[test]
fn encode_5_1_acpl3_real_aspx_multi_env_round_trips_to_6_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_hf_transient_frame(14000.0, 0.6);
    let r = make_hf_transient_frame(11000.0, 0.5);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lfe = make_tone_frame(60.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_1_acpl3_real_aspx_multi_env(
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

/// Multi-envelope liveness: a transient input that selects `num_env = 2`
/// produces a frame distinct from the single-envelope (round-322) path.
#[test]
fn multi_env_frame_differs_from_single_env_for_transient() {
    let l = make_hf_transient_frame(14000.0, 0.6);
    let r = make_hf_transient_frame(11000.0, 0.5);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    // Precondition: the transient L carrier selects num_env = 2.
    assert_eq!(select_num_env_for(&l), 2);

    let mut enc_multi = Ac4ImsEncoder::new();
    let multi = enc_multi.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let mut enc_single = Ac4ImsEncoder::new();
    let single = enc_single.encode_frame_pcm_5_0_acpl3_real_aspx(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_eq!(multi.len(), single.len(), "same padded substream length");
    assert_ne!(
        multi, single,
        "the 2-envelope FIXFIX body must reach the wire distinct from the \
         single-envelope scaffold for a transient input"
    );
}

/// Stationary fallback: a steady (non-transient) input selects
/// `num_env = 1`, so the multi-envelope entry point produces a frame
/// byte-identical to the round-322 single-envelope path.
#[test]
fn stationary_input_falls_back_to_single_env_bytes() {
    // A constant-amplitude HF carrier — uniform per-envelope HF energy
    // keeps the selector at num_env = 1.
    let l = make_tone_frame(14000.0, 0.5);
    let r = make_tone_frame(12000.0, 0.4);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    // Precondition: the steady HF L carrier stays at num_env = 1.
    assert_eq!(select_num_env_for(&l), 1);

    let mut enc_multi = Ac4ImsEncoder::new();
    let multi = enc_multi.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let mut enc_single = Ac4ImsEncoder::new();
    let single = enc_single.encode_frame_pcm_5_0_acpl3_real_aspx(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_eq!(
        multi, single,
        "a stationary frame must fall back byte-for-byte to the \
         single-envelope path"
    );
}

/// Determinism: matched transient inputs + fresh encoder state produce
/// identical bytes across repeated invocations.
#[test]
fn encode_5_0_acpl3_real_aspx_multi_env_is_byte_deterministic() {
    let l = make_hf_transient_frame(14000.0, 0.6);
    let r = make_hf_transient_frame(11000.0, 0.5);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let run = || -> Vec<u8> {
        let mut enc = Ac4ImsEncoder::new();
        enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
            &[&l, &r, &c, &ls, &rs],
            0.5,
            0.1,
            1.0,
            1.0,
        )
    };
    assert_eq!(run(), run());
}
