//! Round 363 — wire a **real** per-noise-subband-group `aspx_tna_mode`
//! (A-SPX inverse filtering) into the live 7.0 **pure-ASPX** frame path,
//! on all four A-SPX carriers.
//!
//! ### Background
//!
//! The 7.0 pure-ASPX body (`7_X_codec_mode == ASPX`, no A-CPL coupling)
//! carries four A-SPX trailers per Table 33 — `aspx_data_2ch()` for the
//! L / R front pair, `aspx_data_2ch()` for the Ls / Rs surround pair,
//! `aspx_data_1ch()` for the centre carrier, and the extra
//! `aspx_data_2ch()` for the back pair Lb / Rb (Table 202 x3/x4). Through
//! round 358 every one of those four emitted the all-zero `aspx_tna_mode`
//! scaffold while the A-CPL-coupled paths had already moved to a real mode.
//!
//! This round wires the real `aspx_tna_mode` into the live 7.0 pure-ASPX
//! path: `encode_frame_pcm_7_0_aspx_real_aspx` now derives an independent
//! tna_mode vector for the front pair (from L, mirrored to R under
//! `aspx_balance = 1`), the surround pair (from Ls), the centre carrier
//! (from C), and the back pair (from Lb), routing all four through the new
//! `build_7_0_aspx_asf_body_from_pcm_spectra_real_aspx_tna`.
//!
//! ### What this measures
//!
//! 1. Round-trip — the live 7.0 pure-ASPX encoder (now emitting real
//!    `aspx_tna_mode` for structured input) still decodes to a 7-channel
//!    AudioFrame.
//! 2. Wire liveness — the `_tna` body with non-zero tna_mode differs from
//!    the all-zero-tna body, and structured multitone input selects at
//!    least one non-None mode.
//! 3. Determinism — matched inputs + fresh encoder state are byte-equal.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.14 Table 33 (`case ASPX:`), Table 202,
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
//! (`aspx_data_2ch()`), §4.3.10.6.1 Table 131 (`aspx_tna_mode`).

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::aspx_tna_select::select_tna_mode;
use oxideav_ac4::aspx_tns::build_q_low_ext;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::qmf_slots_to_sb_major;
use oxideav_ac4::encoder_asf::build_7_0_aspx_asf_body_from_pcm_spectra_real_aspx_tna;
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

/// Live 7.0 pure-ASPX encode (real tna_mode on all four carriers)
/// round-trips to a 7-channel AudioFrame.
#[test]
fn live_7_0_pure_aspx_with_tna_round_trips() {
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
    let frame_bytes = enc.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");
}

/// Structured multitone input selects at least one non-None tna_mode, and
/// the `_tna` body with that vector differs from the all-zero body
/// (empty and explicit-all-zero tna paths agree).
#[test]
fn tna_reaches_the_7_0_pure_aspx_body() {
    let tl = 1920u32;
    let cfg = live_cfg();

    // Distinct structured content per carrier so the four derived vectors
    // are independently meaningful.
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let lb = multitone(0.18);
    let rb = multitone(0.16);

    let front_tna = tna_mode_for(&cfg, &l);
    let surround_tna = tna_mode_for(&cfg, &ls);
    let c_tna = tna_mode_for(&cfg, &c);
    let back_tna = tna_mode_for(&cfg, &lb);
    assert!(
        front_tna
            .iter()
            .chain(&surround_tna)
            .chain(&c_tna)
            .chain(&back_tna)
            .any(|&m| m != 0),
        "at least one carrier must select a non-None tna_mode"
    );

    let counts = derive_aspx_frequency_tables(&cfg, 0).unwrap().counts;
    let sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![0i32; counts.num_sbg_noise as usize];
    let coeffs = vec![0.0f32; 1920];

    let build = |f: &[u8], s: &[u8], cc: &[u8], b: &[u8]| {
        build_7_0_aspx_asf_body_from_pcm_spectra_real_aspx_tna(
            tl,
            40,
            40,
            true,
            &[
                &coeffs, &coeffs, &coeffs, &coeffs, &coeffs, &coeffs, &coeffs,
            ],
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
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            f,
            s,
            cc,
            b,
            12288,
        )
    };

    let body_tna = build(&front_tna, &surround_tna, &c_tna, &back_tna);
    let body_zero = build(&[], &[], &[], &[]);
    assert_ne!(
        body_tna, body_zero,
        "non-zero tna_mode must change the body bytes"
    );
    let zeros = vec![0u8; counts.num_sbg_noise as usize];
    assert_eq!(
        body_zero,
        build(&zeros, &zeros, &zeros, &zeros),
        "all-zero tna == empty tna"
    );

    // Silence anywhere maps to all-None on that carrier (no usable predictor).
    let silence = vec![0.0f32; N];
    assert!(
        tna_mode_for(&cfg, &silence).iter().all(|&m| m == 0),
        "silence -> all-None tna_mode"
    );

    // Unused R/Rs/Rb derivations referenced to keep the wiring honest.
    let _ = (&r, &rs, &rb);
}

/// Matched inputs + fresh encoder state produce byte-identical output.
#[test]
fn live_7_0_pure_aspx_tna_path_is_byte_deterministic() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = multitone(0.22);
    let rs = multitone(0.2);
    let lb = tone(440.0, 0.2);
    let rb = tone(550.0, 0.2);

    let mut e1 = Ac4ImsEncoder::new();
    let mut e2 = Ac4ImsEncoder::new();
    let b1 = e1.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let b2 = e2.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert_eq!(b1, b2);
}
