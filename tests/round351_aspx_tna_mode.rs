//! Round 351 — encoder-side `aspx_tna_mode` (A-SPX inverse-filtering)
//! wired into the live 5_X SIMPLE/ASPX_ACPL_3 frame path.
//!
//! ### Background
//!
//! Every live ASPX frame path through round 347 emitted the
//! `aspx_hfgen_iwc_{1,2}ch()` element as the all-zero scaffold
//! (`aspx_tna_mode = 0` for every noise subband group, all three
//! present-gates 0), even though the decoder fully parses and applies the
//! §5.7.6.4.1.3 chirp / order-2 LPC inverse-filtering driven by
//! `aspx_tna_mode` (ETSI TS 103 190-1 §4.3.10.6.1 Table 131 —
//! None / Light / Moderate / Heavy).
//!
//! This round adds the encoder-side selection (`aspx_tna_select`): a
//! level-independent predictor-strength measure (`|alpha0|^2 + |alpha1|^2`
//! from the decoder's own `compute_covariance` / `compute_alphas`)
//! aggregated per noise subband group via the Pseudocode-89 high-band walk
//! and thresholded into a mode, routed into the live 5_X ACPL_3 path via
//! `build_5_x_acpl3_body_..._real_aspx_tna` +
//! `write_aspx_data_2ch_real_envelope_tna`.
//!
//! ### What this round measures
//!
//! 1. Round-trip — the live 5.0 ACPL_3 encoder (now emitting a real
//!    `aspx_tna_mode` for structured input) still decodes to a 5-channel
//!    AudioFrame.
//! 2. Selection sanity — silence selects no inverse filtering; a strongly
//!    structured carrier selects a non-zero `aspx_tna_mode`; the measure
//!    is level-independent.
//! 3. Wire liveness — the `_tna` body with a non-zero `aspx_tna_mode`
//!    differs from the all-zero-tna body and round-trips through
//!    `parse_aspx_hfgen_iwc_2ch` (channel-0 modes mirrored to channel 1
//!    under `aspx_balance = 1`).
//! 4. Determinism — matched inputs + fresh encoder state are byte-equal.
//!
//! Refs ETSI TS 103 190-1: §4.3.10.6.1 Table 131, §4.2.12.7 Table 56,
//! §5.7.6.4.1.2–§5.7.6.4.1.4 Pseudocodes 86–89.

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, parse_aspx_hfgen_iwc_2ch, AspxConfig, AspxFreqResMode,
    AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::aspx_tna_select::select_tna_mode;
use oxideav_ac4::aspx_tns::build_q_low_ext;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{
    build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna,
    qmf_slots_to_sb_major,
};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::qmf::QmfAnalysisBank;
use oxideav_core::bits::BitReader;
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

fn tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// A broadband multi-component carrier — structured low band that engages
/// the order-2 predictor (selects a non-zero `aspx_tna_mode`).
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

/// Reproduce the encoder's L-carrier tna_mode selection from raw PCM.
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

/// Live 5.0 ACPL_3 encode (now emitting real tna_mode) round-trips.
#[test]
fn live_5_0_acpl3_real_aspx_with_tna_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    // Structured carriers → the encoder signals non-trivial inverse
    // filtering; the decode must still succeed.
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.3);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let frame_bytes =
        enc.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2, "5-channel S16 interleaved");
}

/// Selection sanity: silence → all-None; a structured carrier → non-zero
/// `aspx_tna_mode`; the measure is level-independent.
#[test]
fn tna_selection_sanity() {
    let cfg = live_cfg();
    let modes_silence = tna_mode_for(&cfg, &vec![0.0f32; N]);
    let modes_mt = tna_mode_for(&cfg, &multitone(0.3));
    let modes_mt_loud = tna_mode_for(&cfg, &multitone(0.9));

    assert_eq!(modes_silence.len(), modes_mt.len());
    assert!(modes_silence.iter().chain(&modes_mt).all(|&m| m <= 3));

    assert!(
        modes_silence.iter().all(|&m| m == 0),
        "silence selects no inverse filtering (got {modes_silence:?})"
    );
    assert!(
        modes_mt.iter().any(|&m| m != 0),
        "structured carrier must select a non-None mode (got {modes_mt:?})"
    );
    // Level-independence: a 3x-louder copy selects the same mode vector.
    assert_eq!(
        modes_mt, modes_mt_loud,
        "mode selection must be level-independent"
    );
}

/// The `_tna` body with a non-zero `aspx_tna_mode` differs from the
/// all-zero-tna body and round-trips through `parse_aspx_hfgen_iwc_2ch`.
#[test]
fn tna_reaches_the_wire_and_round_trips() {
    let tl = 1920u32;
    let cfg = live_cfg();
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);

    let tna = tna_mode_for(&cfg, &l);
    assert!(
        tna.iter().any(|&m| m != 0),
        "structured carrier must select at least one non-None tna_mode (got {tna:?})"
    );

    let counts = derive_aspx_frequency_tables(&cfg, 0).unwrap().counts;
    let sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![0i32; counts.num_sbg_noise as usize];

    let build = |tna_mode: &[u8]| {
        build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna(
            tl,
            40,
            None,
            true,
            &l,
            &r,
            Some(&c),
            Some(&ls),
            Some(&rs),
            None,
            &cfg,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            tna_mode,
            &[],
            &[],
            3,
            oxideav_ac4::acpl::AcplQuantMode::Fine,
            oxideav_ac4::acpl::AcplQuantMode::Fine,
            0.5,
            0.1,
            1.0,
            1.0,
            8192,
        )
    };

    let body_tna = build(&tna);
    let body_zero = build(&[]);
    assert_ne!(
        body_tna, body_zero,
        "non-zero tna_mode must change the body bytes"
    );
    // The empty and explicit-all-zero tna paths agree.
    let body_zero2 = build(&vec![0u8; counts.num_sbg_noise as usize]);
    assert_eq!(body_zero, body_zero2, "all-zero tna == empty tna");

    // Re-emit just the aspx_data_2ch() element through the same writer the
    // body uses and confirm parse recovery (the body-bytes-differ
    // assertion above already proves the field reaches the live body).
    use oxideav_ac4::encoder_acpl3::{
        write_aspx_data_2ch_real_envelope_tna, AspxRealEnvelopeChannel,
    };
    use oxideav_core::bits::BitWriter;
    let mut bw = BitWriter::new();
    write_aspx_data_2ch_real_envelope_tna(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise_v,
        },
        AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise_v,
        },
        &tna,
    )
    .expect("aspx config");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    br.read_u32(3).unwrap(); // xover
    br.read_bit().unwrap(); // FIXFIX prefix
    let envbits = cfg.fixfix_tmp_num_env_bits();
    if envbits > 0 {
        br.read_u32(envbits).unwrap();
    }
    if cfg.signals_freq_res() {
        br.read_bit().unwrap();
    }
    let balance = br.read_bit().unwrap();
    assert!(balance);
    br.read_bit().unwrap(); // ch0 sig delta_dir
    br.read_bit().unwrap(); // ch0 noise delta_dir
    br.read_bit().unwrap(); // ch1 sig delta_dir
    br.read_bit().unwrap(); // ch1 noise delta_dir
    let hfgen = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        balance,
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    )
    .unwrap();
    assert_eq!(hfgen.tna_mode[0], tna, "ch0 tna_mode recovered");
    assert_eq!(hfgen.tna_mode[1], tna, "balance=1 mirrors ch0 -> ch1");
}

/// Matched inputs + fresh encoder state produce byte-identical output.
#[test]
fn live_tna_path_is_byte_deterministic() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.3);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);

    let mut e1 = Ac4ImsEncoder::new();
    let mut e2 = Ac4ImsEncoder::new();
    let b1 = e1.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    let b2 = e2.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    assert_eq!(b1, b2);
}
