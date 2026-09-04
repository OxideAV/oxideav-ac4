//! Round 340 (part 2) — extend the mono multi-envelope live A-SPX path to
//! the **7_X** ASPX_ACPL_2 centre carrier.
//!
//! The 7_X dual of `round340_5_x_acpl2_centre_multi_env`: the encoder
//! QMF-analyses the centre carrier, probes its HF energy for a transient,
//! and — when one is present — emits a multi-envelope centre
//! `aspx_data_1ch()` (`num_env = 2`) for the 7_X ASPX_ACPL_2 body while both
//! carrier pairs (L/R front, Ls/Rs surround) keep their single-envelope
//! `aspx_data_2ch()`. Stationary centre carriers fall back to the round-337
//! single-envelope 7_X path.
//!
//! `frames` order is `[L, R, C, Ls, Rs, Lb, Rb(, LFE)]`; per Table 202 the
//! back pair Lb/Rb is reconstructed at decode time from the A-CPL coupling
//! (`z1`/`z3` decorrelator outputs), so it carries no independent carrier
//! under ASPX_ACPL_2.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
//! Table 202 (7_X_channel_element A-CPL channel mapping), §4.2.12.3
//! Table 51, §4.3.10.4.11.

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

fn make_hf_transient_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            let gate = if i >= N / 2 { 1.0 } else { 0.02 };
            gate * amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

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

/// 7.0 centre-multi-envelope encode round-trips to a 7-channel AudioFrame.
#[test]
fn encode_7_0_acpl2_centre_multi_env_round_trips_to_7_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lb = make_tone_frame(700.0, 0.2);
    let rb = make_tone_frame(900.0, 0.2);

    assert_eq!(select_num_env_for(&c), 2);

    let frame_bytes = enc
        .encode_frame_pcm_7_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");
}

/// 7.1 centre-multi-envelope encode round-trips to an 8-channel AudioFrame.
#[test]
fn encode_7_1_acpl2_centre_multi_env_round_trips_to_8_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lb = make_tone_frame(700.0, 0.2);
    let rb = make_tone_frame(900.0, 0.2);
    let lfe = make_tone_frame(60.0, 0.2);

    let frame_bytes = enc.encode_frame_pcm_7_1_acpl2_real_aspx_centre_multi_env(&[
        &l, &r, &c, &ls, &rs, &lb, &rb, &lfe,
    ]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2, "8-channel S16 interleaved");
}

/// Multi-envelope liveness: a transient centre carrier produces a frame
/// distinct from the round-337 single-envelope 7.0 path.
#[test]
fn centre_multi_env_7_0_differs_from_single_env_for_transient() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lb = make_tone_frame(700.0, 0.2);
    let rb = make_tone_frame(900.0, 0.2);

    assert_eq!(select_num_env_for(&c), 2);

    let mut enc_multi = Ac4ImsEncoder::new();
    let multi = enc_multi
        .encode_frame_pcm_7_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let mut enc_single = Ac4ImsEncoder::new();
    let single = enc_single.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    // Substream bodies are tightly sized (audio_size = exact body
    // length), so the two frames need not share a length; the byte-stream
    // inequality below is the real check.
    assert_ne!(
        multi, single,
        "the 2-envelope FIXFIX centre body must reach the wire distinct from \
         the single-envelope scaffold for a transient centre carrier"
    );
}

/// Stationary fallback: a steady centre carrier falls back byte-for-byte to
/// the round-337 single-envelope 7.0 path. The single-env path fills Lb/Rb
/// with the centre buffer, which is fine — the decoder reconstructs Lb/Rb
/// from the A-CPL coupling (Table 202), never from those carrier slots — so
/// the fallback reproduces it exactly by passing the same Lb=Rb=C.
#[test]
fn stationary_centre_7_0_falls_back_to_single_env_bytes() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_tone_frame(14000.0, 0.5); // steady HF → num_env = 1
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    assert_eq!(select_num_env_for(&c), 1);

    let mut enc_multi = Ac4ImsEncoder::new();
    let multi = enc_multi
        .encode_frame_pcm_7_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs, &c, &c]);
    // The single-env fallback fills Lb=Rb=C, so reproduce that here.
    let mut enc_single = Ac4ImsEncoder::new();
    let single = enc_single.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &c, &c]);
    assert_eq!(
        multi, single,
        "a stationary centre carrier must fall back byte-for-byte to the \
         single-envelope path"
    );
}

/// Determinism: matched transient inputs + fresh encoder state produce
/// identical bytes across repeated invocations.
#[test]
fn encode_7_0_acpl2_centre_multi_env_is_byte_deterministic() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_hf_transient_frame(14000.0, 0.6);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lb = make_tone_frame(700.0, 0.2);
    let rb = make_tone_frame(900.0, 0.2);
    let run = || -> Vec<u8> {
        let mut enc = Ac4ImsEncoder::new();
        enc.encode_frame_pcm_7_0_acpl2_real_aspx_centre_multi_env(&[&l, &r, &c, &ls, &rs, &lb, &rb])
    };
    assert_eq!(run(), run());
}
