//! Round 316 — stereo QMF → multi-envelope ASPX builder.
//!
//! The round-310 QMF→multi-envelope builder
//! ([`oxideav_ac4::encoder_acpl3::build_aspx_multi_envelope_channel_from_qmf`])
//! bridges an HF QMF matrix to the per-envelope `AspxEncodedEnvelope`
//! rows the round-299 1-channel writer
//! ([`oxideav_ac4::encoder_acpl3::write_aspx_data_1ch_multi_envelope`])
//! consumes — but only for a single channel. The round-299 **2-channel**
//! coupled writer
//! ([`oxideav_ac4::encoder_acpl3::write_aspx_data_2ch_multi_envelope`])
//! had no QMF builder feeding it; callers had to hand-pack both channels'
//! `qscf` matrices.
//!
//! This round closes that gap with
//! [`oxideav_ac4::encoder_acpl3::build_aspx_multi_envelope_2ch_from_qmf`]:
//! the stereo dual that runs the round-310 per-channel pipeline (uniform
//! FIXFIX partition → Pseudocode-82/83 quant → Pseudocode-80/81 DPCM)
//! independently on each channel's QMF matrix and returns both channels'
//! rows ready to drop into the coupled writer. The LEVEL (ch0) / BALANCE
//! (ch1) codebook-family split is the writer's job — the builder's
//! quantisation and packing are channel-symmetric.
//!
//! The verification path threads the builder output through the
//! 2-channel writer and the decoder's `aspx_data_2ch()` framing +
//! per-channel `parse_aspx_ec_data` parsers, then asserts the recovered
//! per-`(sbg, atsg)` `qscf` matrix for each channel equals what the
//! builder's own quantiser produced from that channel's QMF energy.
//!
//! Refs ETSI TS 103 190-1 §4.2.12.4 Table 52 (LEVEL / BALANCE coupled
//! SIGNAL coding), §4.3.10.4.1 (FIXFIX uniform envelope spacing),
//! §4.3.10.4.11 (aspx_num_env / aspx_num_noise), §5.7.6.3.4 Pseudocodes
//! 80 / 81, §5.7.6.3.5 Pseudocodes 82 / 83, §5.7.6.4.2.1 Pseudocodes
//! 90 / 91.

use oxideav_ac4::aspx::{
    delta_decode_noise, delta_decode_sig, derive_aspx_frequency_tables, num_aspx_timeslots,
    num_ts_in_ats, parse_aspx_delta_dir, parse_aspx_ec_data, parse_aspx_framing,
    parse_aspx_hfgen_iwc_2ch, AspxConfig, AspxDataType, AspxFreqResMode, AspxIntClass,
    AspxMasterFreqScale, AspxQuantStep, AspxStereoMode,
};
use oxideav_ac4::encoder_acpl3::{
    aggregate_qmf_to_sbg_atsg_uniform, balance_encode_noise_matrix, balance_encode_sig_matrix,
    build_aspx_multi_envelope_2ch_from_qmf, dpcm_encode_qscf_envelopes,
    quantize_noise_energy_matrix, quantize_sig_energy_matrix, write_aspx_data_2ch_multi_envelope,
    AspxMultiEnvelope2chRows, AspxMultiEnvelopeChannel, AspxMultiEnvelopePrevLast,
    AspxQmfMultiEnvelopeChannel,
};
use oxideav_core::bits::{BitReader, BitWriter};

const N_SUBBANDS: u32 = 64;
const SBX: u32 = 0;

/// FIXFIX config with `num_env_bits_fixfix = 1` (tmp_num_env 2 b →
/// num_env up to 8) and `signals_freq_res() == false`.
fn multi_cfg() -> AspxConfig {
    AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        start_freq: 0,
        stop_freq: 0,
        master_freq_scale: AspxMasterFreqScale::LowRes,
        interpolation: false,
        preflat: false,
        limiter: false,
        noise_sbg: 0,
        num_env_bits_fixfix: 1,
        freq_res_mode: AspxFreqResMode::DurationDependent,
    }
}

/// Flat per-cell QMF energy matrix `[abs_sb][ts]`, amplitude `amp`.
fn flat_qmf(num_sb: usize, num_ts: usize, amp: f32) -> Vec<Vec<(f32, f32)>> {
    vec![vec![(amp, 0.0); num_ts]; num_sb]
}

/// Loud-second-half QMF matrix: a temporal transient whose quiet half
/// sits above the `n_subbands = 64` SIGNAL F0 floor (`amp >= 8` ⇒ energy
/// `>= 64`).
fn transient_qmf(
    num_sb: usize,
    num_ts: usize,
    base_amp: f32,
    loud_ratio: f32,
) -> Vec<Vec<(f32, f32)>> {
    let mut q = vec![vec![(0.0_f32, 0.0_f32); num_ts]; num_sb];
    for row in q.iter_mut() {
        for (ts, cell) in row.iter_mut().enumerate() {
            let amp = if ts >= num_ts / 2 {
                base_amp * loud_ratio
            } else {
                base_amp
            };
            *cell = (amp, 0.0);
        }
    }
    q
}

/// Shared frame geometry for `frame_len_base = 2048`.
struct Geo {
    nts: u32,
    num_ats: u32,
    num_ts: usize,
}

fn geo() -> Geo {
    let frame_len_base: u32 = 2048;
    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    Geo {
        nts,
        num_ats,
        num_ts: (num_ats * nts) as usize,
    }
}

/// Re-parse a coupled `aspx_data_2ch()` body's framing skeleton + both
/// channels' SIGNAL / NOISE ec_data, returning the four decoded envelope
/// vectors `(sig0, sig1, noise0, noise1)`.
#[allow(clippy::type_complexity)]
fn parse_2ch_body(
    bytes: &[u8],
    cfg: &AspxConfig,
    num_env: u32,
) -> (
    Vec<oxideav_ac4::aspx::AspxHuffEnv>,
    Vec<oxideav_ac4::aspx::AspxHuffEnv>,
    Vec<oxideav_ac4::aspx::AspxHuffEnv>,
    Vec<oxideav_ac4::aspx::AspxHuffEnv>,
) {
    let g = geo();
    let tables = derive_aspx_frequency_tables(cfg, 0).expect("freq tables");
    let counts = tables.counts;

    let mut br = BitReader::new(bytes);
    assert_eq!(br.read_u32(3).unwrap(), 0, "xover = 0");
    let framing = parse_aspx_framing(&mut br, cfg, true, g.num_ats > 8).expect("framing");
    assert!(matches!(framing.int_class, AspxIntClass::FixFix));
    assert_eq!(framing.num_env, num_env, "num_env");
    assert_eq!(framing.num_noise, 2, "num_noise = 2 when num_env > 1");
    let balance = br.read_bit().unwrap();
    assert!(balance, "aspx_balance = 1 (ch1 reuses ch0 framing)");
    let dd0 = parse_aspx_delta_dir(&mut br, &framing).expect("dd0");
    let dd1 = parse_aspx_delta_dir(&mut br, &framing).expect("dd1");
    let _hfgen = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        balance,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        g.num_ats,
    )
    .expect("hfgen 2ch");

    let qmode = cfg.quant_mode_env;
    let sig0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        qmode,
        AspxStereoMode::Level,
        &dd0.sig_delta_dir,
        counts,
    )
    .expect("ch0 sig");
    let sig1 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        qmode,
        AspxStereoMode::Balance,
        &dd1.sig_delta_dir,
        counts,
    )
    .expect("ch1 sig");
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
    .expect("ch0 noise");
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
    .expect("ch1 noise");
    (sig0, sig1, noise0, noise1)
}

/// Full stereo chain: two QMF matrices → 2ch builder → coupled writer →
/// decoder → each channel's recovered `qscf` matrix equals what that
/// channel's QMF energy quantised to.
#[test]
fn build_2ch_from_qmf_round_trips_both_channels() {
    let cfg = multi_cfg();
    let delta = 1;
    let num_env = 2u32;
    let g = geo();

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    // Both channels at per-cell energy = n_subbands = 64 (amplitude 8 ⇒
    // squared magnitude 64). At that energy the SIGNAL inverse
    // (`round(2·log2(64/64)) = 0`) and the NOISE inverse
    // (`round(6 − log2(64)) = 0`) both quantise to qscf 0, which sits
    // inside both the LEVEL and BALANCE F0 codebooks' addressable ranges
    // (all `cb_off = 0`, only non-negative F0 indices) — so the coupled
    // body round-trips byte-cleanly through the writer's F0 / DF clamps on
    // both the LEVEL (ch0) and BALANCE (ch1) channels. The two channels'
    // distinctness is exercised separately in `transient_2ch_round_trips`
    // and `matches_per_channel_1ch_builder`; here the goal is verifying
    // the stereo wiring recovers each channel's own qscf exactly.
    let q0 = flat_qmf(max_sb, g.num_ts, 8.0);
    let q1 = flat_qmf(max_sb, g.num_ts, 8.0);

    let ch0 = AspxQmfMultiEnvelopeChannel {
        q_high: &q0,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let ch1 = AspxQmfMultiEnvelopeChannel {
        q_high: &q1,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };

    let rows: AspxMultiEnvelope2chRows = build_aspx_multi_envelope_2ch_from_qmf(
        &ch0,
        &ch1,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        g.nts,
        g.num_ats,
        SBX,
        AspxMultiEnvelopePrevLast::default(),
        AspxMultiEnvelopePrevLast::default(),
        delta,
        false,
    );
    assert_eq!(rows.ch0_sig.len(), num_env as usize, "ch0 one sig row/env");
    assert_eq!(rows.ch1_sig.len(), num_env as usize, "ch1 one sig row/env");
    assert_eq!(rows.ch0_noise.len(), 2, "ch0 two noise envelopes");
    assert_eq!(rows.ch1_noise.len(), 2, "ch1 two noise envelopes");

    // Independently compute the expected qscf matrices per channel.
    let expect_sig = |q: &[Vec<(f32, f32)>]| {
        let e =
            aggregate_qmf_to_sbg_atsg_uniform(q, &sbg_sig_borders, num_env, g.num_ats, g.nts, SBX);
        quantize_sig_energy_matrix(&e, cfg.quant_mode_env, N_SUBBANDS)
    };
    let expect_noise = |q: &[Vec<(f32, f32)>]| {
        let e = aggregate_qmf_to_sbg_atsg_uniform(q, &sbg_noise_borders, 2, g.num_ats, g.nts, SBX);
        quantize_noise_energy_matrix(&e)
    };
    // §5.7.6.3.5: the coupled pair transmits (sum, pan) — the builder
    // applies the Pseudocode 84 inverse cell-wise, so the wire carries
    // the converted matrices (ch1 in pan wire steps; the decoder's
    // delta = 2 accumulation recovers qscf_b = 2 · pan).
    let (exp_sum_sig, exp_pan_sig) =
        balance_encode_sig_matrix(&expect_sig(&q0), &expect_sig(&q1), cfg.quant_mode_env);
    let (exp_sum_noise, exp_pan_noise) =
        balance_encode_noise_matrix(&expect_noise(&q0), &expect_noise(&q1));

    // Emit the coupled body.
    let mut bw = BitWriter::new();
    write_aspx_data_2ch_multi_envelope(
        &mut bw,
        &cfg,
        num_env,
        AspxMultiEnvelopeChannel {
            sig: &rows.ch0_sig,
            noise: &rows.ch0_noise,
        },
        AspxMultiEnvelopeChannel {
            sig: &rows.ch1_sig,
            noise: &rows.ch1_noise,
        },
    )
    .expect("2ch multi-envelope writer");
    bw.align_to_byte();
    let bytes = bw.finish();

    let (sig0, sig1, noise0, noise1) = parse_2ch_body(&bytes, &cfg, num_env);

    // Compare in the transmitted-symbol domain: delta = 1 accumulation
    // recovers the sum matrix on ch0 and the pan wire-step matrix on
    // ch1.
    let qscf_sig0 = delta_decode_sig(&sig0, num_sbg_sig, &[], delta);
    let qscf_sig1 = delta_decode_sig(&sig1, num_sbg_sig, &[], delta);
    let qscf_noise0 = delta_decode_noise(&noise0, num_sbg_noise, &[], delta);
    let qscf_noise1 = delta_decode_noise(&noise1, num_sbg_noise, &[], delta);

    for sbg in 0..num_sbg_sig as usize {
        for atsg in 0..num_env as usize {
            assert_eq!(
                qscf_sig0[sbg][atsg], exp_sum_sig[sbg][atsg],
                "ch0 (sum) SIGNAL sbg={sbg} atsg={atsg}"
            );
            assert_eq!(
                qscf_sig1[sbg][atsg], exp_pan_sig[sbg][atsg],
                "ch1 (pan) SIGNAL sbg={sbg} atsg={atsg}"
            );
        }
    }
    for sbg in 0..num_sbg_noise as usize {
        for atsg in 0..2usize {
            assert_eq!(
                qscf_noise0[sbg][atsg], exp_sum_noise[sbg][atsg],
                "ch0 (sum) NOISE sbg={sbg} atsg={atsg}"
            );
            assert_eq!(
                qscf_noise1[sbg][atsg], exp_pan_noise[sbg][atsg],
                "ch1 (pan) NOISE sbg={sbg} atsg={atsg}"
            );
        }
    }
}

/// A per-channel transient produces a temporally varying SIGNAL qscf that
/// survives the coupled writer + decoder chain on both LEVEL and BALANCE
/// channels.
#[test]
fn transient_2ch_round_trips() {
    let cfg = multi_cfg();
    let delta = 1;
    let num_env = 2u32;
    let g = geo();

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    // Both channels transient; quiet half >= n_subbands so SIGNAL qscf >= 0.
    let q0 = transient_qmf(max_sb, g.num_ts, 10.0, 4.0);
    let q1 = transient_qmf(max_sb, g.num_ts, 12.0, 3.0);
    let ch0 = AspxQmfMultiEnvelopeChannel {
        q_high: &q0,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let ch1 = AspxQmfMultiEnvelopeChannel {
        q_high: &q1,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };

    let rows = build_aspx_multi_envelope_2ch_from_qmf(
        &ch0,
        &ch1,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        g.nts,
        g.num_ats,
        SBX,
        AspxMultiEnvelopePrevLast::default(),
        AspxMultiEnvelopePrevLast::default(),
        delta,
        false,
    );

    let expect_sig = |q: &[Vec<(f32, f32)>]| {
        let e =
            aggregate_qmf_to_sbg_atsg_uniform(q, &sbg_sig_borders, num_env, g.num_ats, g.nts, SBX);
        quantize_sig_energy_matrix(&e, cfg.quant_mode_env, N_SUBBANDS)
    };
    let exp_sig0 = expect_sig(&q0);
    let exp_sig1 = expect_sig(&q1);
    // Loud-second-half ⇒ env 1 qscf strictly above env 0 on lead SBG.
    assert!(
        exp_sig0[0][1] > exp_sig0[0][0],
        "ch0 transient is time-varying"
    );
    assert!(
        exp_sig1[0][1] > exp_sig1[0][0],
        "ch1 transient is time-varying"
    );
    // Precondition: every expected SIGNAL qscf is in the F0 range.
    assert!(exp_sig0.iter().all(|r| r.iter().all(|&v| v >= 0)));
    assert!(exp_sig1.iter().all(|r| r.iter().all(|&v| v >= 0)));
    // §5.7.6.3.5 wire domain: (sum, pan) via the Pseudocode 84 inverse.
    // The sum matrix inherits the transient (both channels are loud in
    // the second half).
    let (exp_sum_sig, exp_pan_sig) =
        balance_encode_sig_matrix(&exp_sig0, &exp_sig1, cfg.quant_mode_env);
    assert!(
        exp_sum_sig[0][1] > exp_sum_sig[0][0],
        "sum channel keeps the transient"
    );

    let mut bw = BitWriter::new();
    write_aspx_data_2ch_multi_envelope(
        &mut bw,
        &cfg,
        num_env,
        AspxMultiEnvelopeChannel {
            sig: &rows.ch0_sig,
            noise: &rows.ch0_noise,
        },
        AspxMultiEnvelopeChannel {
            sig: &rows.ch1_sig,
            noise: &rows.ch1_noise,
        },
    )
    .expect("writer");
    bw.align_to_byte();
    let bytes = bw.finish();

    let (sig0, sig1, _noise0, _noise1) = parse_2ch_body(&bytes, &cfg, num_env);
    let qscf_sig0 = delta_decode_sig(&sig0, num_sbg_sig, &[], delta);
    let qscf_sig1 = delta_decode_sig(&sig1, num_sbg_sig, &[], delta);
    for sbg in 0..num_sbg_sig as usize {
        for atsg in 0..num_env as usize {
            assert_eq!(
                qscf_sig0[sbg][atsg], exp_sum_sig[sbg][atsg],
                "ch0 (sum) sbg{sbg} env{atsg}"
            );
            assert_eq!(
                qscf_sig1[sbg][atsg], exp_pan_sig[sbg][atsg],
                "ch1 (pan) sbg{sbg} env{atsg}"
            );
        }
    }
    let _ = num_sbg_noise;
}

/// `force_freq = true` makes every envelope on both channels FREQ.
#[test]
fn force_freq_emits_all_freq_rows_both_channels() {
    let cfg = multi_cfg();
    let num_env = 4u32;
    let g = geo();

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_sbg_sig = tables.counts.num_sbg_sig_highres;
    let num_sbg_noise = tables.counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    let q0 = flat_qmf(max_sb, g.num_ts, 0.4);
    let q1 = transient_qmf(max_sb, g.num_ts, 10.0, 3.0);
    let ch0 = AspxQmfMultiEnvelopeChannel {
        q_high: &q0,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let ch1 = AspxQmfMultiEnvelopeChannel {
        q_high: &q1,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };

    let rows = build_aspx_multi_envelope_2ch_from_qmf(
        &ch0,
        &ch1,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        g.nts,
        g.num_ats,
        SBX,
        AspxMultiEnvelopePrevLast::default(),
        AspxMultiEnvelopePrevLast::default(),
        1,
        true,
    );
    assert!(
        rows.ch0_sig.iter().all(|r| !r.direction_time),
        "ch0 sig FREQ"
    );
    assert!(
        rows.ch1_sig.iter().all(|r| !r.direction_time),
        "ch1 sig FREQ"
    );
    assert!(
        rows.ch0_noise.iter().all(|r| !r.direction_time),
        "ch0 noise FREQ"
    );
    assert!(
        rows.ch1_noise.iter().all(|r| !r.direction_time),
        "ch1 noise FREQ"
    );
}

/// The 2ch builder is the composition of the per-channel Pseudocode
/// 82/83 quantisation, the §5.7.6.3.5 (sum, pan) conversion
/// (Pseudocode 84 inverse, cell-wise), and the Pseudocode 80/81 DPCM
/// packing in the transmitted-symbol domain.
#[test]
fn matches_per_channel_1ch_builder() {
    let cfg = multi_cfg();
    let num_env = 2u32;
    let g = geo();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    let q0 = transient_qmf(max_sb, g.num_ts, 10.0, 4.0);
    let q1 = flat_qmf(max_sb, g.num_ts, 8.0);
    let ch0 = AspxQmfMultiEnvelopeChannel {
        q_high: &q0,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let ch1 = AspxQmfMultiEnvelopeChannel {
        q_high: &q1,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };

    let rows = build_aspx_multi_envelope_2ch_from_qmf(
        &ch0,
        &ch1,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        g.nts,
        g.num_ats,
        SBX,
        AspxMultiEnvelopePrevLast::default(),
        AspxMultiEnvelopePrevLast::default(),
        1,
        false,
    );

    // Compose the same pipeline by hand: per-channel quant matrices →
    // (sum, pan) conversion → DPCM packing.
    let quant_sig = |q: &[Vec<(f32, f32)>]| {
        let e =
            aggregate_qmf_to_sbg_atsg_uniform(q, &sbg_sig_borders, num_env, g.num_ats, g.nts, SBX);
        quantize_sig_energy_matrix(&e, cfg.quant_mode_env, N_SUBBANDS)
    };
    let quant_noise = |q: &[Vec<(f32, f32)>]| {
        let e = aggregate_qmf_to_sbg_atsg_uniform(q, &sbg_noise_borders, 2, g.num_ats, g.nts, SBX);
        quantize_noise_energy_matrix(&e)
    };
    let (sum_sig, pan_sig) =
        balance_encode_sig_matrix(&quant_sig(&q0), &quant_sig(&q1), cfg.quant_mode_env);
    let (sum_noise, pan_noise) = balance_encode_noise_matrix(&quant_noise(&q0), &quant_noise(&q1));
    let s0 = dpcm_encode_qscf_envelopes(&sum_sig, &[], 1, false);
    let n0 = dpcm_encode_qscf_envelopes(&sum_noise, &[], 1, false);
    let s1 = dpcm_encode_qscf_envelopes(&pan_sig, &[], 1, false);
    let n1 = dpcm_encode_qscf_envelopes(&pan_noise, &[], 1, false);

    assert_eq!(rows.ch0_sig, s0, "ch0 sig matches composed pipeline");
    assert_eq!(rows.ch0_noise, n0, "ch0 noise matches composed pipeline");
    assert_eq!(rows.ch1_sig, s1, "ch1 sig matches composed pipeline");
    assert_eq!(rows.ch1_noise, n1, "ch1 noise matches composed pipeline");
}

/// Determinism: identical stereo QMF input produces identical builder
/// output.
#[test]
fn builder_2ch_is_deterministic() {
    let cfg = multi_cfg();
    let num_env = 2u32;
    let g = geo();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_sbg_sig = tables.counts.num_sbg_sig_highres;
    let num_sbg_noise = tables.counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    let q0 = transient_qmf(max_sb, g.num_ts, 10.0, 3.0);
    let q1 = transient_qmf(max_sb, g.num_ts, 9.0, 2.0);
    let ch0 = AspxQmfMultiEnvelopeChannel {
        q_high: &q0,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let ch1 = AspxQmfMultiEnvelopeChannel {
        q_high: &q1,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let run = || {
        build_aspx_multi_envelope_2ch_from_qmf(
            &ch0,
            &ch1,
            num_env,
            cfg.quant_mode_env,
            N_SUBBANDS,
            g.nts,
            g.num_ats,
            SBX,
            AspxMultiEnvelopePrevLast::default(),
            AspxMultiEnvelopePrevLast::default(),
            1,
            false,
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a.ch0_sig, b.ch0_sig);
    assert_eq!(a.ch1_sig, b.ch1_sig);
    assert_eq!(a.ch0_noise, b.ch0_noise);
    assert_eq!(a.ch1_noise, b.ch1_noise);
}
