//! Round 358 (part 2) — wire a **real** per-noise-subband-group
//! `aspx_tna_mode` (A-SPX inverse filtering) into the live 7_X
//! SIMPLE/ASPX_ACPL_2 frame path, on all three A-SPX carriers.
//!
//! ### Background
//!
//! The 7_X ASPX_ACPL_2 body carries three A-SPX trailers — `aspx_data_2ch()`
//! for the L / R front pair, `aspx_data_2ch()` for the Ls / Rs surround
//! pair, and `aspx_data_1ch()` for the centre carrier (matching the
//! decoder's `parse_7x_audio_data_outer` trailer walk). Through round 351
//! every one of those three emitted the all-zero `aspx_tna_mode` scaffold.
//!
//! This part wires the real `aspx_tna_mode` into the live 7_X ACPL_2 path:
//! `encode_frame_pcm_7_{0,1}_acpl2_real_aspx` now derives a tna_mode vector
//! for the front pair (from L, mirrored to R under `aspx_balance = 1`), the
//! surround pair (from Ls, mirrored to Rs), and the centre carrier (from C),
//! routing all three through the new
//! `build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna`.
//!
//! ### What this part measures
//!
//! 1. Round-trip — the live 7.0 + 7.1 ACPL_2 encoders (now emitting real
//!    `aspx_tna_mode` for structured input) still decode to a 7- / 8-channel
//!    AudioFrame.
//! 2. Wire liveness — the `_tna` body with non-zero tna_mode differs from
//!    the all-zero-tna body.
//! 3. Determinism — matched inputs + fresh encoder state are byte-equal.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
//! (`aspx_data_2ch()`), §4.3.10.6.1 Table 131 (`aspx_tna_mode`).

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::aspx_tna_select::select_tna_mode;
use oxideav_ac4::aspx_tns::build_q_low_ext;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{
    build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna, qmf_slots_to_sb_major,
};
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

fn tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

fn multitone(amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * ((2.0 * std::f32::consts::PI * 3000.0 * t).sin()
                + (2.0 * std::f32::consts::PI * 5000.0 * t).sin()
                + (2.0 * std::f32::consts::PI * 7000.0 * t).sin())
        })
        .collect()
}

fn tna_mode_for(cfg: &AspxConfig, pcm: &[f32]) -> Vec<u8> {
    let tables = derive_aspx_frequency_tables(cfg, 0).expect("freq tables");
    let n_slots = pcm.len() / 64;
    let usable = n_slots * 64;
    let mut bank = QmfAnalysisBank::new();
    let q_sb_major = qmf_slots_to_sb_major(&bank.process_block(&pcm[..usable]));
    let sba = tables.sba as usize;
    let q_low: Vec<Vec<(f32, f32)>> = q_sb_major.iter().take(sba).map(|r| r.to_vec()).collect();
    let q_low_ext = build_q_low_ext(&q_low, &[], tables.sba);
    select_tna_mode(&q_low_ext, &tables, cfg.master_freq_scale, true)
}

/// Live 7.0 ACPL_2 encode (real tna_mode on all three carriers) round-trips.
#[test]
fn live_7_0_acpl2_real_aspx_with_tna_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let lb = tone(440.0, 0.2);
    let rb = tone(550.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");
}

/// Live 7.1 ACPL_2 encode round-trips to an 8-channel AudioFrame.
#[test]
fn live_7_1_acpl2_real_aspx_with_tna_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let lb = tone(440.0, 0.2);
    let rb = tone(550.0, 0.2);
    let lfe = tone(60.0, 0.4);
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2, "8-channel S16 interleaved");
}

/// The `_tna` body with non-zero per-carrier tna_mode differs from the
/// all-zero body; empty and explicit-all-zero tna paths agree.
#[test]
fn tna_reaches_the_7x_acpl2_body() {
    let tl = 1920u32;
    let cfg = live_cfg();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);

    let front_tna = tna_mode_for(&cfg, &l);
    let surround_tna = tna_mode_for(&cfg, &ls);
    let c_tna = tna_mode_for(&cfg, &c);
    assert!(
        front_tna
            .iter()
            .chain(&surround_tna)
            .chain(&c_tna)
            .any(|&m| m != 0),
        "at least one carrier must select a non-None tna_mode"
    );

    let counts = derive_aspx_frequency_tables(&cfg, 0).unwrap().counts;
    let sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![0i32; counts.num_sbg_noise as usize];

    let build = |f: &[u8], s: &[u8], cc: &[u8]| {
        build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
            tl,
            40,
            None,
            true,
            &l,
            &r,
            &ls,
            &rs,
            &c,
            None,
            &cfg,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            f,
            s,
            cc,
            3,
            oxideav_ac4::acpl::AcplQuantMode::Fine,
            12288,
        )
    };

    let body_tna = build(&front_tna, &surround_tna, &c_tna);
    let body_zero = build(&[], &[], &[]);
    assert_ne!(
        body_tna, body_zero,
        "non-zero tna_mode must change the body bytes"
    );
    let zeros = vec![0u8; counts.num_sbg_noise as usize];
    assert_eq!(
        body_zero,
        build(&zeros, &zeros, &zeros),
        "all-zero tna == empty tna"
    );
}

/// Matched inputs + fresh encoder state produce byte-identical output.
#[test]
fn live_7x_acpl2_tna_path_is_byte_deterministic() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let lb = tone(440.0, 0.2);
    let rb = tone(550.0, 0.2);

    let mut e1 = Ac4ImsEncoder::new();
    let mut e2 = Ac4ImsEncoder::new();
    let b1 = e1.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let b2 = e2.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert_eq!(b1, b2);
}
