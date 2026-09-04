//! Round 340 — wire the **mono multi-envelope** (`num_env > 1`) real-ASPX
//! payload into the live 5_X ASPX_ACPL_2 frame path's centre carrier,
//! closing the README's "the live `aspx_data_1ch()` path remains
//! single-envelope (`num_env = 1`)" deferral.
//!
//! ### Background
//!
//! Round 331 wired the single-envelope real-ASPX centre `aspx_data_1ch()`
//! (FIXFIX `num_env = 1`, via `write_aspx_data_1ch_real_envelope`) into the
//! live 5_X ASPX_ACPL_2 frame builder. The round-299 mono multi-envelope
//! writer (`write_aspx_data_1ch_multi_envelope`), the round-310 QMF→rows
//! builder (`build_aspx_multi_envelope_channel_from_qmf`), and the round-310
//! envelope selector (`select_aspx_num_env_from_qmf`) have all existed and
//! round-tripped in isolation, but the live 1-channel A-SPX element still
//! always emitted a single FIXFIX envelope.
//!
//! This round adds `encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env`:
//! the encoder QMF-analyses the **centre** carrier, probes its HF energy
//! for a temporal transient (`select_aspx_num_env_from_qmf`), and — when one
//! is present — splits the frame into `num_env = 2` uniformly spaced FIXFIX
//! signal envelopes, emitting the multi-envelope `aspx_data_1ch()` body for
//! the centre while the L / R front pair keeps its single-envelope
//! `aspx_data_2ch()`. Stationary centre carriers fall back to the round-331
//! single-envelope path, so the method is a strict superset of that entry
//! point.
//!
//! ### What this round measures
//!
//! 1. Transient drives the selector — a centre carrier whose HF energy is
//!    concentrated in the second half of the frame makes
//!    `select_aspx_num_env_from_qmf` return `2`.
//! 2. Round-trip — the multi-envelope 5.0 encoder output is accepted by
//!    `Ac4Decoder` and yields a 5-channel AudioFrame.
//! 3. Multi-envelope liveness — a transient centre carrier drives a
//!    2-envelope FIXFIX framing whose body differs from the single-envelope
//!    (round-331) frame for the same input.
//! 4. Stationary fallback — a stationary centre carrier produces a frame
//!    byte-identical to the round-331 single-envelope path.
//! 5. Determinism — matched inputs + fresh encoder state produce identical
//!    bytes.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.6 Table 25 (`case ASPX_ACPL_2:`),
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.3.10.1.9 Table 123,
//! §4.3.10.4.11 (`aspx_num_env`), §5.7.6.3.4 / §5.7.6.3.5 Pseudocodes 80–83.

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

/// The live config the IMS encoder uses for the 5_X ACPL_2 path.
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

/// Reproduce the encoder's `num_env` selection for a single carrier so the
/// test can assert the driver fires.
fn select_num_env_for(pcm: &[f32]) -> u32 {
    let cfg = live_cfg();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let frame_len = N as u32;
    let nts = num_ts_in_ats(frame_len);
    let aspx_frame_ts = num_aspx_timeslots(frame_len);
    let n_slots = pcm.len() / 64;
    let usable = n_slots * 64;
    let mut bank = QmfAnalysisBank::new();
    let q_high = qmf_slots_to_sb_major(&bank.process_block(&pcm[..usable]));
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

/// The transient HF centre carrier drives the selector to a 2-envelope
/// FIXFIX partition; a steady full-frame HF tone stays at 1.
#[test]
fn centre_transient_drives_num_env_to_two() {
    let transient = make_hf_transient_frame(14000.0, 0.6);
    assert_eq!(
        select_num_env_for(&transient),
        2,
        "a second-half HF transient must select num_env = 2"
    );
    let steady_hf = make_tone_frame(14000.0, 0.5);
    assert_eq!(
        select_num_env_for(&steady_hf),
        1,
        "a steady full-frame HF tone must stay at num_env = 1"
    );
}

/// 5.0 centre-multi-envelope encode round-trips to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl2_centre_multi_env_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    // Transient centre carrier so the multi-envelope path is taken.
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    // Precondition: the centre carrier selects num_env = 2.
    assert_eq!(select_num_env_for(&c), 2);

    let frame_bytes =
        enc.encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs]);
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

/// Multi-envelope liveness: a transient centre carrier that selects
/// `num_env = 2` produces a frame distinct from the single-envelope
/// (round-331) path.
#[test]
fn centre_multi_env_frame_differs_from_single_env_for_transient() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    assert_eq!(select_num_env_for(&c), 2);

    let mut enc_multi = Ac4ImsEncoder::new();
    let multi =
        enc_multi.encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs]);
    let mut enc_single = Ac4ImsEncoder::new();
    let single = enc_single.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    // Substream bodies are tightly sized (audio_size = exact body
    // length), so the two frames need not share a length; the byte-stream
    // inequality below is the real check.
    assert_ne!(
        multi, single,
        "the 2-envelope FIXFIX centre body must reach the wire distinct from \
         the single-envelope scaffold for a transient centre carrier"
    );
}

/// Stationary fallback: a steady (non-transient) centre carrier selects
/// `num_env = 1`, so the multi-envelope entry point produces a frame
/// byte-identical to the round-331 single-envelope path.
#[test]
fn stationary_centre_falls_back_to_single_env_bytes() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    // A constant-amplitude HF centre carrier keeps the selector at 1.
    let c = make_tone_frame(14000.0, 0.5);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    assert_eq!(select_num_env_for(&c), 1);

    let mut enc_multi = Ac4ImsEncoder::new();
    let multi =
        enc_multi.encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs]);
    let mut enc_single = Ac4ImsEncoder::new();
    let single = enc_single.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    assert_eq!(
        multi, single,
        "a stationary centre carrier must fall back byte-for-byte to the \
         single-envelope path"
    );
}

/// Determinism: matched transient inputs + fresh encoder state produce
/// identical bytes across repeated invocations.
#[test]
fn encode_5_0_acpl2_centre_multi_env_is_byte_deterministic() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let run = || -> Vec<u8> {
        let mut enc = Ac4ImsEncoder::new();
        enc.encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs])
    };
    assert_eq!(run(), run());
}
