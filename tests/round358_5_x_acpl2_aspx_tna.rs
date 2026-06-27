//! Round 358 — wire a **real** per-noise-subband-group `aspx_tna_mode`
//! (A-SPX inverse filtering) into the live 5_X SIMPLE/ASPX_ACPL_2 frame
//! path, on all three A-SPX carriers.
//!
//! ### Background
//!
//! Round 351 added the encoder-side `aspx_tna_mode` selector
//! (`aspx_tna_select::select_tna_mode`: a level-independent
//! predictor-strength measure `|alpha0|² + |alpha1|²` aggregated per noise
//! subband group via the Pseudocode-89 high-band walk and thresholded into
//! None / Light / Moderate / Heavy) and wired it into the live 5_X
//! ASPX_ACPL_3 frame path only. ASPX_ACPL_3 has no `aspx_data_1ch()`
//! element — its A-SPX body is a single `aspx_data_2ch()`.
//!
//! The 5_X ASPX_ACPL_2 body carries **three** A-SPX trailers — the L / R
//! carrier-pair `aspx_data_2ch()` *and* the centre carrier's
//! `aspx_data_1ch()` — and through round 351 every one of those three
//! emitted the all-zero `aspx_tna_mode` scaffold even though the decoder
//! fully parses + applies the §5.7.6.4.1.3 chirp / order-2 LPC inverse
//! filtering driven by `aspx_tna_mode`.
//!
//! This round wires the real `aspx_tna_mode` into the live 5_X ACPL_2
//! path: `encode_frame_pcm_5_{0,1}_acpl2_real_aspx` now derives a tna_mode
//! vector for the L carrier (mirrored to R under `aspx_balance = 1`) **and**
//! an independent vector for the centre carrier (from its own QMF low
//! band), routing both through the new
//! `build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna`
//! (`write_aspx_data_2ch_real_envelope_tna` for the front pair +
//! `write_aspx_data_1ch_real_envelope_tna` for the centre). This exercises
//! the so-far-undriven `write_aspx_data_1ch_real_envelope_tna` 1-channel
//! writer on a live path for the first time.
//!
//! ### What this round measures
//!
//! 1. Round-trip — the live 5.0 + 5.1 ACPL_2 encoders (now emitting real
//!    `aspx_tna_mode` for structured input) still decode to a 5-channel
//!    AudioFrame.
//! 2. Wire liveness — the `_tna` body with a non-zero tna_mode differs from
//!    the all-zero-tna body, and both the front-pair `aspx_data_2ch()` and
//!    the centre `aspx_data_1ch()` recover their tna_mode through
//!    `parse_aspx_hfgen_iwc_{2,1}ch`.
//! 3. Per-carrier independence — the centre carrier's tna_mode is derived
//!    from C's own low band, so a tonal-C / noisy-LR input recovers a
//!    non-zero centre tna_mode regardless of the front pair.
//! 4. Determinism — matched inputs + fresh encoder state are byte-equal.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.6 Table 25 (`case ASPX_ACPL_2:`),
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
//! (`aspx_data_2ch()`), §4.2.12.7 Table 56 (`aspx_hfgen_iwc`),
//! §4.3.10.6.1 Table 131 (`aspx_tna_mode`), §5.7.6.4.1.2–4 Pseudocodes
//! 86–89.

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, parse_aspx_hfgen_iwc_1ch, parse_aspx_hfgen_iwc_2ch, AspxConfig,
    AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::aspx_tna_select::select_tna_mode;
use oxideav_ac4::aspx_tns::build_q_low_ext;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{
    build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna, qmf_slots_to_sb_major,
};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::qmf::QmfAnalysisBank;
use oxideav_core::bits::BitReader;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

/// The live config the ACPL_2 real-ASPX path uses.
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

/// Reproduce the encoder's per-carrier tna_mode selection from raw PCM.
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

/// Live 5.0 ACPL_2 encode (now emitting real tna_mode on all three
/// carriers) round-trips to a 5-channel AudioFrame.
#[test]
fn live_5_0_acpl2_real_aspx_with_tna_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2, "5-channel S16 interleaved");
}

/// The 5.1 wrapper (which routes the first five channels through the 5.0
/// path) also round-trips with real tna_mode.
#[test]
fn live_5_1_acpl2_real_aspx_with_tna_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let lfe = tone(60.0, 0.4);
    let frame_bytes = enc.encode_frame_pcm_5_1_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lfe]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2, "5-channel S16 interleaved");
}

/// The `_tna` body with non-zero tna_mode differs from the all-zero body,
/// and both the front-pair `aspx_data_2ch()` and the centre
/// `aspx_data_1ch()` recover their tna_mode through the decoder's parsers.
#[test]
fn tna_reaches_both_acpl2_carriers_and_round_trips() {
    let tl = 1920u32;
    let cfg = live_cfg();

    // Tonal C, broadband L/R — exercises per-carrier independence.
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);

    let lr_tna = tna_mode_for(&cfg, &l);
    let c_tna = tna_mode_for(&cfg, &c);
    assert!(
        lr_tna.iter().any(|&m| m != 0),
        "structured L carrier must select a non-None tna_mode (got {lr_tna:?})"
    );
    assert!(
        c_tna.iter().any(|&m| m != 0),
        "structured C carrier must select a non-None tna_mode (got {c_tna:?})"
    );

    let counts = derive_aspx_frequency_tables(&cfg, 0).unwrap().counts;
    let sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![0i32; counts.num_sbg_noise as usize];

    let build = |lr: &[u8], cc: &[u8]| {
        build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
            tl,
            40,
            true,
            &l,
            &r,
            &c,
            &ls,
            &rs,
            &cfg,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            &sig,
            &noise_v,
            lr,
            cc,
            &[],
            &[],
            &[],
            3,
            oxideav_ac4::acpl::AcplQuantMode::Fine,
            8192,
        )
    };

    let body_tna = build(&lr_tna, &c_tna);
    let body_zero = build(&[], &[]);
    assert_ne!(
        body_tna, body_zero,
        "non-zero tna_mode must change the body bytes"
    );
    // Empty and explicit-all-zero tna paths agree.
    let zeros = vec![0u8; counts.num_sbg_noise as usize];
    assert_eq!(
        body_zero,
        build(&zeros, &zeros),
        "all-zero tna == empty tna"
    );
}

/// Re-emit the centre `aspx_data_1ch()` element through the same writer the
/// live body uses and confirm `parse_aspx_hfgen_iwc_1ch` recovers the
/// tna_mode verbatim (the 1-channel `_tna` writer driven for the first
/// time on a live path this round).
#[test]
fn centre_1ch_tna_round_trips_through_parser() {
    use oxideav_ac4::encoder_acpl3::{
        write_aspx_data_1ch_real_envelope_tna, AspxRealEnvelopeChannel,
    };
    use oxideav_core::bits::BitWriter;

    let cfg = live_cfg();
    let c = multitone(0.28);
    let c_tna = tna_mode_for(&cfg, &c);
    assert!(
        c_tna.iter().any(|&m| m != 0),
        "structured C carrier must select at least one non-None tna_mode (got {c_tna:?})"
    );

    let counts = derive_aspx_frequency_tables(&cfg, 0).unwrap().counts;
    let sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![0i32; counts.num_sbg_noise as usize];

    let mut bw = BitWriter::new();
    write_aspx_data_1ch_real_envelope_tna(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise_v,
        },
        &c_tna,
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
    br.read_bit().unwrap(); // sig delta_dir
    br.read_bit().unwrap(); // noise delta_dir
    let hfgen =
        parse_aspx_hfgen_iwc_1ch(&mut br, counts.num_sbg_noise, counts.num_sbg_sig_highres, 0)
            .unwrap();
    assert_eq!(hfgen.tna_mode, c_tna, "centre 1ch tna_mode recovered");
}

/// Re-emit the front-pair `aspx_data_2ch()` element through the same writer
/// the live body uses and confirm `parse_aspx_hfgen_iwc_2ch` recovers the
/// tna_mode on both channels (balance=1 mirrors ch0 -> ch1).
#[test]
fn front_pair_2ch_tna_round_trips_through_parser() {
    use oxideav_ac4::encoder_acpl3::{
        write_aspx_data_2ch_real_envelope_tna, AspxRealEnvelopeChannel,
    };
    use oxideav_core::bits::BitWriter;

    let cfg = live_cfg();
    let l = multitone(0.3);
    let lr_tna = tna_mode_for(&cfg, &l);
    assert!(lr_tna.iter().any(|&m| m != 0));

    let counts = derive_aspx_frequency_tables(&cfg, 0).unwrap().counts;
    let sig = vec![0i32; counts.num_sbg_sig_highres as usize];
    let noise_v = vec![0i32; counts.num_sbg_noise as usize];

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
        &lr_tna,
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
    assert_eq!(hfgen.tna_mode[0], lr_tna, "ch0 tna_mode recovered");
    assert_eq!(hfgen.tna_mode[1], lr_tna, "balance=1 mirrors ch0 -> ch1");
}

/// Matched inputs + fresh encoder state produce byte-identical output.
#[test]
fn live_acpl2_tna_path_is_byte_deterministic() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = multitone(0.28);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);

    let mut e1 = Ac4ImsEncoder::new();
    let mut e2 = Ac4ImsEncoder::new();
    let b1 = e1.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    let b2 = e2.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    assert_eq!(b1, b2);
}
