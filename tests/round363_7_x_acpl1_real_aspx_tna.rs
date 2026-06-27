//! Round 363 (part 2) — wire **real ASPX envelopes + a real
//! per-noise-subband-group `aspx_tna_mode`** (A-SPX inverse filtering) into
//! the live 7.0 / 7.1 ASPX_ACPL_1 frame path, on all three A-SPX carriers.
//!
//! ### Background
//!
//! The 7_X ASPX_ACPL_1 body carries three A-SPX trailers — `aspx_data_2ch()`
//! for the L / R front pair, `aspx_data_2ch()` for the Ls / Rs surround
//! pair, and `aspx_data_1ch()` for the centre carrier — followed by two
//! `acpl_data_1ch()` (the D0 / D1 coupling modules). Through round 358 the
//! ACPL_1 path emitted the round-118 minimum-bit-cost ASPX scaffold
//! (`write_aspx_data_*_minimal`) on all three carriers while the ACPL_2 /
//! pure-ASPX paths had already moved to real envelopes + real tna_mode.
//!
//! This part wires real ASPX into the live 7_X ACPL_1 path:
//! `encode_frame_pcm_7_{0,1}_acpl1_real_alpha_beta` now QMF-analyses the
//! three carriers, emits real per-sbg SIGNAL / NOISE envelopes, and derives
//! an independent `aspx_tna_mode` per carrier (front from L, surround from
//! Ls, centre from C), routing all three through the new
//! `build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna`.
//!
//! ### What this measures
//!
//! 1. Round-trip — the live 7.0 + 7.1 ACPL_1 encoders (now emitting real
//!    ASPX) still decode to a 7- / 8-channel AudioFrame.
//! 2. Wire liveness — the real-ASPX/tna body differs from the all-zero-tna,
//!    empty-envelope body; structured multitone selects a non-None mode.
//! 3. Determinism — matched inputs + fresh encoder state are byte-equal.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.14 Table 33 (`case ASPX_ACPL_1:`),
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
//! (`aspx_data_2ch()`), §4.3.10.6.1 Table 131 (`aspx_tna_mode`).

use oxideav_ac4::acpl::AcplQuantMode;
use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::aspx_tna_select::select_tna_mode;
use oxideav_ac4::aspx_tns::build_q_low_ext;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{
    build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna, qmf_slots_to_sb_major,
    AspxRealEnvelopeChannel,
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
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * (i as f32 / FS)).sin())
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

/// Live 7.0 ACPL_1 real-α/β + real-ASPX encode round-trips to 7 channels.
#[test]
fn live_7_0_acpl1_real_aspx_with_tna_round_trips() {
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
    let frame_bytes =
        enc.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");
}

/// Live 7.1 ACPL_1 real-α/β + real-ASPX encode round-trips to 8 channels.
#[test]
fn live_7_1_acpl1_real_aspx_with_tna_round_trips() {
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
        enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2, "8-channel S16 interleaved");
}

/// Structured multitone selects a non-None tna_mode, and the real-ASPX/tna
/// body differs from the all-zero, empty-envelope body (empty == all-zero).
#[test]
fn real_aspx_tna_reaches_the_7x_acpl1_body() {
    let tl = 1920u32;
    let cfg = live_cfg();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let coeffs = vec![0.0f32; 1920];

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
    let sig = vec![3i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![1i32; counts.num_sbg_noise as usize];
    let z_sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let z_noise = vec![0i32; counts.num_sbg_noise as usize];

    let build = |env_sig: &[i32], env_noise: &[i32], f: &[u8], s: &[u8], cc: &[u8]| {
        let ch = AspxRealEnvelopeChannel {
            sig: env_sig,
            noise: env_noise,
        };
        build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
            tl,
            40,
            20,
            None,
            true,
            &l,
            &r,
            &ls,
            &rs,
            &c,
            None,
            &cfg,
            3,
            AcplQuantMode::Fine,
            0,
            (ch, ch),
            (ch, ch),
            ch,
            f,
            s,
            cc,
            (&[], &[]),
            (&[], &[]),
            &[],
            12288,
        )
    };

    let body_real = build(&sig, &noise_v, &front_tna, &surround_tna, &c_tna);
    let body_zero = build(&z_sig, &z_noise, &[], &[], &[]);
    assert_ne!(
        body_real, body_zero,
        "real envelopes + non-zero tna must change the body bytes"
    );
    let zeros = vec![0u8; counts.num_sbg_noise as usize];
    assert_eq!(
        body_zero,
        build(&z_sig, &z_noise, &zeros, &zeros, &zeros),
        "all-zero tna == empty tna for zero envelopes"
    );

    // Reference unused symmetric carriers so the wiring intent is explicit.
    let _ = (&r, &rs, &coeffs);
}

/// Matched inputs + fresh encoder state produce byte-identical output.
#[test]
fn live_7_0_acpl1_real_aspx_tna_path_is_byte_deterministic() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let lb = tone(440.0, 0.2);
    let rb = tone(550.0, 0.2);

    let mut e1 = Ac4ImsEncoder::new();
    let mut e2 = Ac4ImsEncoder::new();
    let b1 = e1.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let b2 = e2.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert_eq!(b1, b2);
}
