//! Round 435 — A-SPX **balance stereo joint decoding** (ETSI TS
//! 103 190-1 §5.7.6.3.4-5), end-to-end.
//!
//! An `aspx_data_2ch()` element with `aspx_balance = 1` carries the
//! pair as a jointly coded (sum, balance) pair: the balance channel's
//! envelopes accumulate with `delta = 2` (Pseudocode 80/81
//! `(ch == 1 && aspx_balance == 1)` arms) and both channels dequantize
//! through the Pseudocode 84 joint formulas (`PAN_OFFSET = 12`).
//! Historically the in-tree chain read the secondary channel as
//! absolute LEVEL rows through the BALANCE codebooks; this round wires
//! the spec convention on both sides. These tests pin:
//!
//! 1. Wire-level agreement — a body emitted by the (now converting)
//!    real-envelope writer parses back and joint-decodes to each
//!    channel's original scale factors, including a hard HF pan.
//! 2. Cross-frame TIME direction on the balance channel — the
//!    `delta = 2` accumulation carries across A-SPX intervals through
//!    the per-channel `AspxEnvPrev` state.
//! 3. Full-chain pan survival — a 5.0 ACPL_3 real-ASPX encode with an
//!    HF-loud L carrier and an HF-quiet R carrier decodes with the HF
//!    asymmetry intact (the joint decode preserves the pan the
//!    Pseudocode 84 inverse encoded).
//!
//! Refs ETSI TS 103 190-1 §4.2.12.4 Table 52, §4.3.10.3.2 Table 125,
//! §5.7.6.3.4 Pseudocodes 80/81, §5.7.6.3.5 Pseudocodes 82-84.

use oxideav_ac4::aspx::{
    decode_scf_balance_pair, derive_aspx_frequency_tables, num_aspx_timeslots,
    parse_aspx_delta_dir, parse_aspx_ec_data, parse_aspx_framing, parse_aspx_hfgen_iwc_2ch,
    AspxConfig, AspxDataType, AspxEnvPrev, AspxFreqResMode, AspxHuffEnv, AspxMasterFreqScale,
    AspxQuantStep, AspxStereoMode, ASPX_PAN_OFFSET,
};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{write_aspx_data_2ch_real_envelope, AspxRealEnvelopeChannel};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn small_cfg() -> AspxConfig {
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

/// Parse a written `aspx_data_2ch()` body back to the four envelope
/// vectors (ch0 sig, ch1 sig, ch0 noise, ch1 noise) plus the shared
/// framing's freq_res, asserting `aspx_balance == 1` on the wire.
#[allow(clippy::type_complexity)]
fn parse_body(
    bytes: &[u8],
    cfg: &AspxConfig,
) -> (
    Vec<AspxHuffEnv>,
    Vec<AspxHuffEnv>,
    Vec<AspxHuffEnv>,
    Vec<AspxHuffEnv>,
    Vec<bool>,
) {
    let tables = derive_aspx_frequency_tables(cfg, 0).expect("tables");
    let counts = tables.counts;
    let mut br = BitReader::new(bytes);
    let _xover = br.read_u32(3).unwrap();
    let nats = num_aspx_timeslots(2048);
    let framing = parse_aspx_framing(&mut br, cfg, true, nats > 8).expect("framing");
    let balance = br.read_bit().unwrap();
    assert!(balance, "writer emits aspx_balance = 1");
    let dd0 = parse_aspx_delta_dir(&mut br, &framing).expect("dd0");
    let dd1 = parse_aspx_delta_dir(&mut br, &framing).expect("dd1");
    let _hfgen = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        balance,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen");
    let sig0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        AspxQuantStep::Fine,
        AspxStereoMode::Level,
        &dd0.sig_delta_dir,
        counts,
    )
    .expect("sig0");
    let sig1 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        AspxQuantStep::Fine,
        AspxStereoMode::Balance,
        &dd1.sig_delta_dir,
        counts,
    )
    .expect("sig1");
    let noise0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Noise,
        framing.num_noise,
        &[],
        AspxQuantStep::Fine,
        AspxStereoMode::Level,
        &dd0.noise_delta_dir,
        counts,
    )
    .expect("noise0");
    let noise1 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Noise,
        framing.num_noise,
        &[],
        AspxQuantStep::Fine,
        AspxStereoMode::Balance,
        &dd1.noise_delta_dir,
        counts,
    )
    .expect("noise1");
    (sig0, sig1, noise0, noise1, framing.freq_res.clone())
}

/// Wire-level agreement: (L, R) LEVEL rows with a hard pan go through
/// the converting writer, parse back, and joint-decode (Pseudocode
/// 80/81 delta = 1/2 + Pseudocode 84) to each channel's original
/// scale factors within the (sum, pan) grid's quantization error.
#[test]
fn writer_output_joint_decodes_to_per_channel_levels() {
    let cfg = small_cfg();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("tables");
    let counts = tables.counts;
    let n_sig = counts.num_sbg_sig_highres as usize;
    let n_noise = counts.num_sbg_noise as usize;

    // Hard pan: L 12 quant units (9 dB Fine) above R, both varying
    // over frequency. Absolute LEVEL rows.
    let l_abs: Vec<i32> = (0..n_sig as i32).map(|i| 14 + (i % 3)).collect();
    let r_abs: Vec<i32> = (0..n_sig as i32).map(|i| 2 + (i % 2)).collect();
    let ln_abs: Vec<i32> = (0..n_noise as i32).map(|i| 2 + i).collect();
    let rn_abs: Vec<i32> = (0..n_noise as i32).map(|i| 6 + i).collect();

    // Pack to [F0, DF...] rows for the writer.
    let pack = |abs: &[i32]| -> Vec<i32> {
        let mut out = Vec::with_capacity(abs.len());
        let mut prev = 0;
        for &v in abs {
            out.push(v - prev);
            prev = v;
        }
        out
    };
    let mut bw = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &pack(&l_abs),
            noise: &pack(&ln_abs),
        },
        AspxRealEnvelopeChannel {
            sig: &pack(&r_abs),
            noise: &pack(&rn_abs),
        },
    )
    .expect("writer");
    bw.align_to_byte();
    let bytes = bw.finish();

    let (sig0, sig1, noise0, noise1, freq_res) = parse_body(&bytes, &cfg);

    // Joint decode exactly as the decoder's pair-level path does.
    let mut prev_a = AspxEnvPrev::default();
    let mut prev_b = AspxEnvPrev::default();
    let (scf_a, scf_b) = decode_scf_balance_pair(
        &tables,
        &sig0,
        &noise0,
        &sig1,
        &noise1,
        AspxQuantStep::Fine,
        &freq_res,
        &mut prev_a,
        &mut prev_b,
    );

    // Recovered per-channel scale factors match the LEVEL-domain
    // inputs within the joint grid's quantization error (< 2 dB).
    for sbg in 0..n_sig {
        let want_l = 64.0 * 2f32.powf(l_abs[sbg] as f32 / 2.0);
        let want_r = 64.0 * 2f32.powf(r_abs[sbg] as f32 / 2.0);
        let err_l = 10.0 * (scf_a.scf_sig_sbg[sbg][0] / want_l).log10().abs();
        let err_r = 10.0 * (scf_b.scf_sig_sbg[sbg][0] / want_r).log10().abs();
        assert!(
            err_l < 2.0 && err_r < 2.0,
            "sig sbg {sbg}: L err {err_l} dB, R err {err_r} dB"
        );
    }
    for sbg in 0..n_noise {
        let want_l = 2f32.powi(6 - ln_abs[sbg]);
        let want_r = 2f32.powi(6 - rn_abs[sbg]);
        let err_l = 10.0 * (scf_a.scf_noise_sbg[sbg][0] / want_l).log10().abs();
        let err_r = 10.0 * (scf_b.scf_noise_sbg[sbg][0] / want_r).log10().abs();
        assert!(
            err_l < 2.5 && err_r < 2.5,
            "noise sbg {sbg}: L err {err_l} dB, R err {err_r} dB"
        );
    }
    // The 9 dB pan itself survives: L/R recovered ratio in every
    // signal group is at least 6 dB.
    for sbg in 0..n_sig {
        let ratio_db = 10.0 * (scf_a.scf_sig_sbg[sbg][0] / scf_b.scf_sig_sbg[sbg][0]).log10();
        assert!(ratio_db > 6.0, "pan lost at sbg {sbg}: {ratio_db} dB");
    }
}

/// Cross-frame TIME direction on the balance channel: the second
/// interval's TIME rows accumulate with `delta = 2` against the
/// balance channel's own `AspxEnvPrev` (quantized domain), exactly as
/// Pseudocode 80/81 specify for `(ch == 1 && aspx_balance == 1)`.
#[test]
fn balance_time_rows_accumulate_delta2_across_intervals() {
    let cfg = small_cfg();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("tables");
    let n_sig = tables.sbg_sig_highres.len() - 1;
    let n_noise = tables.sbg_noise.len() - 1;

    let freq_env = |f0: i32, n: usize| -> Vec<AspxHuffEnv> {
        let mut values = vec![0i32; n];
        values[0] = f0;
        vec![AspxHuffEnv {
            values,
            direction_time: false,
        }]
    };
    let time_env = |d: i32, n: usize| -> Vec<AspxHuffEnv> {
        vec![AspxHuffEnv {
            values: vec![d; n],
            direction_time: true,
        }]
    };

    let mut prev_a = AspxEnvPrev::default();
    let mut prev_b = AspxEnvPrev::default();
    // Interval 1 (FREQ): sum F0 = 8, pan wire F0 = 9 ⇒ qscf_b = 18.
    let (_, _) = decode_scf_balance_pair(
        &tables,
        &freq_env(8, n_sig),
        &freq_env(3, n_noise),
        &freq_env(9, n_sig),
        &freq_env(6, n_noise),
        AspxQuantStep::Fine,
        &[],
        &mut prev_a,
        &mut prev_b,
    );
    assert_eq!(prev_a.qscf_sig_last[0], 8);
    assert_eq!(prev_b.qscf_sig_last[0], 18, "delta = 2 accumulation");
    assert_eq!(prev_b.qscf_noise_last[0], 12);

    // Interval 2 (TIME): sum DT = +1, pan DT = +2 wire steps ⇒
    // qscf_b = 18 + 2·2 = 22.
    let (scf_a2, scf_b2) = decode_scf_balance_pair(
        &tables,
        &time_env(1, n_sig),
        &time_env(0, n_noise),
        &time_env(2, n_sig),
        &time_env(0, n_noise),
        AspxQuantStep::Fine,
        &[],
        &mut prev_a,
        &mut prev_b,
    );
    assert_eq!(prev_a.qscf_sig_last[0], 9);
    assert_eq!(prev_b.qscf_sig_last[0], 22, "TIME delta = 2 across frames");
    // qscf_b/a = 11 < PAN_OFFSET ⇒ still panned toward B, but the
    // scf ratio moved by exactly 2 quant units (3 dB Fine) versus
    // interval 1's pan of 18/2 = 9.
    let ratio_db = 10.0 * (scf_a2.scf_sig_sbg[0][0] / scf_b2.scf_sig_sbg[0][0]).log10();
    let expect_db = 10.0 * 2f32.powf((22.0 / 2.0) - ASPX_PAN_OFFSET as f32).log10();
    assert!(
        (ratio_db - expect_db).abs() < 0.01,
        "pan ratio {ratio_db} dB != expected {expect_db} dB"
    );
}

fn tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Band energy above `cut_hz`, via a coarse DFT projection.
fn hf_energy(pcm: &[f32], cut_hz: f32) -> f32 {
    let n = pcm.len();
    let mut e = 0.0f32;
    let mut k = (cut_hz * n as f32 / FS).ceil() as usize;
    let k_max = n / 2;
    while k < k_max {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, &v) in pcm.iter().enumerate() {
            let ph = 2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
            re += v * ph.cos();
            im -= v * ph.sin();
        }
        e += re * re + im * im;
        // Coarse stride keeps the O(n²) projection affordable.
        k += 16;
    }
    e
}

/// Full-chain pan survival: a 5.0 ACPL_3 real-ASPX encode whose L
/// carrier is HF-loud while R is HF-quiet must decode with the HF
/// asymmetry intact — the Pseudocode 84 inverse encodes the pan and
/// the joint decode restores it.
#[test]
fn acpl3_full_chain_preserves_hf_pan() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    // L: LF carrier + strong HF tone. R: LF carrier only.
    let l: Vec<f32> = tone(400.0, 0.3)
        .iter()
        .zip(tone(15_000.0, 0.4).iter())
        .map(|(a, b)| a + b)
        .collect();
    let r = tone(420.0, 0.3);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);

    // Several frames so the streaming QMF banks settle.
    let mut last: Vec<Vec<f32>> = Vec::new();
    for f in 0..5 {
        let bytes =
            enc.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
        let pkt = Packet::new(f, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("packet accepted");
        let Frame::Audio(af) = dec.receive_frame().expect("frame") else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, N as u32, "frame length");
        // De-interleave S16 to per-channel f32.
        let buf = &af.data[0];
        assert_eq!(buf.len(), N * 5 * 2, "interleaved buffer size");
        let mut chans: Vec<Vec<f32>> = (0..5).map(|_| Vec::with_capacity(N)).collect();
        for i in 0..N {
            for (ch, out) in chans.iter_mut().enumerate() {
                let idx = (i * 5 + ch) * 2;
                let s = i16::from_le_bytes([buf[idx], buf[idx + 1]]);
                out.push(s as f32 / 32768.0);
            }
        }
        last = chans;
    }
    let e_l = hf_energy(&last[0], 13_000.0);
    let e_r = hf_energy(&last[1], 13_000.0);
    eprintln!("ROUND-435 balance pan: decoded HF energy L {e_l:.6}, R {e_r:.6}");
    assert!(
        e_l > 4.0 * e_r,
        "HF pan lost through the joint balance chain: L {e_l}, R {e_r}"
    );
}
