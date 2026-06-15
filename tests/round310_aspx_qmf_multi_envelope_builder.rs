//! Round 310 — QMF → multi-envelope ASPX builder + envelope-count
//! selection.
//!
//! The round-240 QMF aggregator
//! ([`oxideav_ac4::encoder_acpl3::aggregate_qmf_to_sbg_atsg`]) reduces an
//! HF QMF matrix to a per-`(sbg, atsg)` energy matrix, and the round-292
//! packer ([`oxideav_ac4::encoder_acpl3::dpcm_encode_qscf_envelopes`])
//! turns a per-`(sbg, atsg)` quant matrix into the per-envelope
//! `AspxEncodedEnvelope` rows the round-299 multi-envelope body writers
//! consume. Until this round nothing bridged the two for `num_env > 1`,
//! and nothing chose the per-frame envelope count from input energy.
//!
//! This round closes that follow-up with
//! [`oxideav_ac4::encoder_acpl3::build_aspx_multi_envelope_channel_from_qmf`]
//! (QMF → uniform FIXFIX partition → Pseudocode-82/83 quant →
//! Pseudocode-80/81 DPCM packing) and
//! [`oxideav_ac4::encoder_acpl3::select_aspx_num_env_from_qmf`] (transient
//! detection picks the FIXFIX `num_env`). The verification path threads
//! the builder output through the round-299 writer and the decoder
//! parsers, then asserts the recovered per-`(sbg, atsg)` `qscf` matrix
//! equals what the builder's own quantiser produced from the QMF energy.
//!
//! Refs ETSI TS 103 190-1 §4.3.10.4.1 (FIXFIX uniform envelope spacing),
//! §4.3.10.4.11 (aspx_num_env / aspx_num_noise), §4.3.10.1.9 (Table 123),
//! §5.7.6.3.4 Pseudocodes 80 / 81, §5.7.6.3.5 Pseudocodes 82 / 83,
//! §5.7.6.4.2.1 Pseudocodes 90 / 91.

use oxideav_ac4::aspx::{
    delta_decode_noise, delta_decode_sig, derive_aspx_frequency_tables, num_aspx_timeslots,
    num_ts_in_ats, parse_aspx_delta_dir, parse_aspx_ec_data, parse_aspx_framing,
    parse_aspx_hfgen_iwc_1ch, AspxConfig, AspxDataType, AspxFreqResMode, AspxIntClass,
    AspxMasterFreqScale, AspxQuantStep, AspxStereoMode,
};
use oxideav_ac4::encoder_acpl3::{
    aggregate_qmf_to_sbg_atsg_uniform, build_aspx_multi_envelope_channel_from_qmf,
    fixfix_uniform_atsg_borders, quantize_noise_energy_matrix, quantize_sig_energy_matrix,
    select_aspx_num_env_from_qmf, write_aspx_data_1ch_multi_envelope, AspxMultiEnvelopeChannel,
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

/// Build a synthetic HF QMF matrix `[abs_sb][ts]` where the energy in the
/// *second* time-half is `loud_ratio` times the first half — a temporal
/// transient. `num_sb` absolute subbands, `num_ts` time slots. `base_amp`
/// sets the quiet-half amplitude; the SIGNAL F0 codebook only addresses
/// `qscf >= 0` (`scf >= n_subbands = 64`, i.e. squared magnitude >= 64),
/// so round-trip callers pass a `base_amp` large enough to keep the quiet
/// half's per-cell energy above that floor.
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

/// A flat QMF matrix — constant energy across time (no transient).
fn flat_qmf(num_sb: usize, num_ts: usize, amp: f32) -> Vec<Vec<(f32, f32)>> {
    vec![vec![(amp, 0.0); num_ts]; num_sb]
}

/// `fixfix_uniform_atsg_borders` partitions the frame evenly and the last
/// border closes at the exact frame length (remainder folded into the
/// trailing envelope).
#[test]
fn uniform_atsg_borders_partition_evenly() {
    assert_eq!(fixfix_uniform_atsg_borders(16, 1), vec![0, 16]);
    assert_eq!(fixfix_uniform_atsg_borders(16, 2), vec![0, 8, 16]);
    // Divisible: even spans, last border closes at the frame length.
    assert_eq!(fixfix_uniform_atsg_borders(16, 4), vec![0, 4, 8, 12, 16]);
    // Non-divisible: 15 / 4 = 3, trailing envelope absorbs the remainder.
    assert_eq!(fixfix_uniform_atsg_borders(15, 4), vec![0, 3, 6, 9, 15]);
    assert!(fixfix_uniform_atsg_borders(16, 0).is_empty());
}

/// The selector picks `num_env > 1` for a transient frame and `1` for a
/// stationary frame.
#[test]
fn selector_detects_transient() {
    let frame_len_base: u32 = 2048;
    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    let num_ts = (num_ats * nts) as usize;
    let sbg_borders: Vec<u32> = (0..=8).collect();

    let transient = transient_qmf(8, num_ts, 1.0, 50.0);
    let n_t = select_aspx_num_env_from_qmf(&transient, &sbg_borders, nts, num_ats, SBX, 8, 0.3);
    assert!(
        n_t >= 2 && n_t.is_power_of_two(),
        "transient frame selects num_env >= 2 (got {n_t})"
    );

    let flat = flat_qmf(8, num_ts, 0.5);
    let n_f = select_aspx_num_env_from_qmf(&flat, &sbg_borders, nts, num_ats, SBX, 8, 0.3);
    assert_eq!(n_f, 1, "stationary frame selects num_env = 1");

    // Degenerate inputs return 1.
    assert_eq!(
        select_aspx_num_env_from_qmf(&flat, &sbg_borders, nts, 1, SBX, 8, 0.3),
        1,
        "frame < 2 ATSs → 1"
    );
    assert_eq!(
        select_aspx_num_env_from_qmf(&flat, &sbg_borders, nts, num_ats, SBX, 1, 0.3),
        1,
        "max_num_env < 2 → 1"
    );
}

/// Full chain: QMF → builder → 1ch multi-envelope writer → decoder
/// parser → delta-decoder recovers the exact `qscf[sbg][atsg]` matrix the
/// builder's own quantiser produced from the QMF energy.
#[test]
fn build_from_qmf_round_trips_through_writer_and_decoder() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let delta = 1;
    let num_env = 2u32;
    let num_noise = 2u32;

    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    let num_ts = (num_ats * nts) as usize;

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    // SBG border lists matching the derived counts (absolute subbands).
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    // Flat per-cell energy of exactly `n_subbands = 64` (amplitude 8 →
    // squared magnitude 64). At that energy both the Pseudocode-82 signal
    // inverse (`round(2·log2(64/64)) = 0`) and the Pseudocode-83 noise
    // inverse (`round(6 − log2(64)) = 0`) quantise to qscf 0, which sits
    // inside both F0 codebooks' addressable ranges — so the full chain
    // round-trips byte-cleanly through the writer's F0 / DF clamps.
    // (The time-varying quant path is exercised separately in
    // `transient_frame_yields_time_varying_qscf` and the SIGNAL-only
    // round-trip in `sig_transient_round_trips`.)
    let q_high = flat_qmf(max_sb, num_ts, 8.0);
    let ch = AspxQmfMultiEnvelopeChannel {
        q_high: &q_high,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };

    let (sig_rows, noise_rows) = build_aspx_multi_envelope_channel_from_qmf(
        &ch,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        nts,
        num_ats,
        SBX,
        &[],
        &[],
        delta,
        false,
    );
    assert_eq!(sig_rows.len(), num_env as usize, "one sig row per envelope");
    assert_eq!(noise_rows.len(), num_noise as usize, "two noise envelopes");

    // Independently compute the expected qscf matrices the same way the
    // builder does — these are what the decoder must recover.
    let sig_energy =
        aggregate_qmf_to_sbg_atsg_uniform(&q_high, &sbg_sig_borders, num_env, num_ats, nts, SBX);
    let noise_energy = aggregate_qmf_to_sbg_atsg_uniform(
        &q_high,
        &sbg_noise_borders,
        num_noise,
        num_ats,
        nts,
        SBX,
    );
    let expected_sig = quantize_sig_energy_matrix(&sig_energy, cfg.quant_mode_env, N_SUBBANDS);
    let expected_noise = quantize_noise_energy_matrix(&noise_energy);

    // Emit + re-parse.
    let mut bw = BitWriter::new();
    write_aspx_data_1ch_multi_envelope(
        &mut bw,
        &cfg,
        num_env,
        AspxMultiEnvelopeChannel {
            sig: &sig_rows,
            noise: &noise_rows,
        },
    )
    .expect("1ch multi-envelope writer");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    assert_eq!(br.read_u32(3).unwrap(), 0, "xover");
    let framing = parse_aspx_framing(&mut br, &cfg, true, num_ats > 8).expect("framing");
    assert!(matches!(framing.int_class, AspxIntClass::FixFix));
    assert_eq!(framing.num_env, num_env);
    assert_eq!(framing.num_noise, num_noise);
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("delta_dir");
    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        num_ats,
    )
    .expect("hfgen");

    let sig_dec = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        cfg.quant_mode_env,
        AspxStereoMode::Level,
        &dd.sig_delta_dir,
        counts,
    )
    .expect("sig ec_data");
    let noise_dec = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Noise,
        framing.num_noise,
        &[],
        AspxQuantStep::Fine,
        AspxStereoMode::Level,
        &dd.noise_delta_dir,
        counts,
    )
    .expect("noise ec_data");

    let sig_qscf = delta_decode_sig(&sig_dec, num_sbg_sig, &[], delta);
    let noise_qscf = delta_decode_noise(&noise_dec, num_sbg_noise, &[], delta);

    for sbg in 0..num_sbg_sig as usize {
        for atsg in 0..num_env as usize {
            assert_eq!(
                sig_qscf[sbg][atsg], expected_sig[sbg][atsg],
                "SIGNAL qscf mismatch at sbg={sbg} atsg={atsg}"
            );
        }
    }
    for sbg in 0..num_sbg_noise as usize {
        for atsg in 0..num_noise as usize {
            assert_eq!(
                noise_qscf[sbg][atsg], expected_noise[sbg][atsg],
                "NOISE qscf mismatch at sbg={sbg} atsg={atsg}"
            );
        }
    }
}

/// SIGNAL-only transient round-trip: a loud-second-half QMF matrix whose
/// quiet half sits at energy ≥ `n_subbands` keeps every SIGNAL qscf ≥ 0
/// (inside the F0 codebook range), so the time-varying SIGNAL envelopes
/// recover exactly through the writer + decoder chain.
#[test]
fn sig_transient_round_trips() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let delta = 1;
    let num_env = 2u32;
    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    let num_ts = (num_ats * nts) as usize;

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    // Quiet half energy = 100 (> 64) so SIGNAL qscf >= 0; loud half × 4.
    let q_high = transient_qmf(max_sb, num_ts, 10.0, 4.0);
    let ch = AspxQmfMultiEnvelopeChannel {
        q_high: &q_high,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let (sig_rows, noise_rows) = build_aspx_multi_envelope_channel_from_qmf(
        &ch,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        nts,
        num_ats,
        SBX,
        &[],
        &[],
        delta,
        false,
    );

    let sig_energy =
        aggregate_qmf_to_sbg_atsg_uniform(&q_high, &sbg_sig_borders, num_env, num_ats, nts, SBX);
    let expected_sig = quantize_sig_energy_matrix(&sig_energy, cfg.quant_mode_env, N_SUBBANDS);
    // Precondition: every expected SIGNAL qscf is in the F0 range.
    assert!(
        expected_sig.iter().all(|r| r.iter().all(|&v| v >= 0)),
        "fixture keeps SIGNAL qscf >= 0"
    );

    let mut bw = BitWriter::new();
    write_aspx_data_1ch_multi_envelope(
        &mut bw,
        &cfg,
        num_env,
        AspxMultiEnvelopeChannel {
            sig: &sig_rows,
            noise: &noise_rows,
        },
    )
    .expect("writer");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    br.read_u32(3).unwrap();
    let framing = parse_aspx_framing(&mut br, &cfg, true, num_ats > 8).expect("framing");
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("delta_dir");
    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        num_ats,
    )
    .expect("hfgen");
    let sig_dec = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        cfg.quant_mode_env,
        AspxStereoMode::Level,
        &dd.sig_delta_dir,
        counts,
    )
    .expect("sig ec_data");
    let sig_qscf = delta_decode_sig(&sig_dec, num_sbg_sig, &[], delta);
    for sbg in 0..num_sbg_sig as usize {
        for atsg in 0..num_env as usize {
            assert_eq!(
                sig_qscf[sbg][atsg], expected_sig[sbg][atsg],
                "SIGNAL transient qscf mismatch at sbg={sbg} atsg={atsg}"
            );
        }
    }
}

/// The builder's transient frame produces a *temporally varying* qscf
/// matrix: the second envelope (loud half) has a higher leading-SBG qscf
/// than the first.
#[test]
fn transient_frame_yields_time_varying_qscf() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let num_env = 2u32;
    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    let num_ts = (num_ats * nts) as usize;

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_sbg_sig = tables.counts.num_sbg_sig_highres;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let max_sb = (num_sbg_sig + 1) as usize;

    let q_high = transient_qmf(max_sb, num_ts, 10.0, 4.0);
    let sig_energy =
        aggregate_qmf_to_sbg_atsg_uniform(&q_high, &sbg_sig_borders, num_env, num_ats, nts, SBX);
    let qscf = quantize_sig_energy_matrix(&sig_energy, cfg.quant_mode_env, N_SUBBANDS);

    // Loud second half ⇒ env 1 qscf strictly greater than env 0 on the
    // leading subband group (16× energy = 4 octaves = +8 Fine steps).
    assert!(
        qscf[0][1] > qscf[0][0],
        "transient: env1 qscf ({}) > env0 qscf ({})",
        qscf[0][1],
        qscf[0][0]
    );
}

/// `force_freq = true` reproduces the all-FREQ scaffold: every envelope
/// row carries `direction_time == false`.
#[test]
fn force_freq_emits_all_freq_rows() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let num_env = 4u32;
    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    let num_ts = (num_ats * nts) as usize;

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_sbg_sig = tables.counts.num_sbg_sig_highres;
    let num_sbg_noise = tables.counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    let q_high = flat_qmf(max_sb, num_ts, 0.4);
    let ch = AspxQmfMultiEnvelopeChannel {
        q_high: &q_high,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let (sig_rows, noise_rows) = build_aspx_multi_envelope_channel_from_qmf(
        &ch,
        num_env,
        cfg.quant_mode_env,
        N_SUBBANDS,
        nts,
        num_ats,
        SBX,
        &[],
        &[],
        1,
        true,
    );
    assert!(
        sig_rows.iter().all(|r| !r.direction_time),
        "force_freq: every sig envelope is FREQ"
    );
    assert!(
        noise_rows.iter().all(|r| !r.direction_time),
        "force_freq: every noise envelope is FREQ"
    );
}

/// Determinism: identical QMF input produces identical builder output.
#[test]
fn builder_is_deterministic() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let num_env = 2u32;
    let num_ats = num_aspx_timeslots(frame_len_base);
    let nts = num_ts_in_ats(frame_len_base);
    let num_ts = (num_ats * nts) as usize;

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_sbg_sig = tables.counts.num_sbg_sig_highres;
    let num_sbg_noise = tables.counts.num_sbg_noise;
    let sbg_sig_borders: Vec<u32> = (0..=num_sbg_sig).collect();
    let sbg_noise_borders: Vec<u32> = (0..=num_sbg_noise).collect();
    let max_sb = (num_sbg_sig.max(num_sbg_noise) + 1) as usize;

    let q_high = transient_qmf(max_sb, num_ts, 10.0, 3.0);
    let ch = AspxQmfMultiEnvelopeChannel {
        q_high: &q_high,
        sbg_sig_borders: &sbg_sig_borders,
        sbg_noise_borders: &sbg_noise_borders,
    };
    let run = || {
        build_aspx_multi_envelope_channel_from_qmf(
            &ch,
            num_env,
            cfg.quant_mode_env,
            N_SUBBANDS,
            nts,
            num_ats,
            SBX,
            &[],
            &[],
            1,
            false,
        )
    };
    assert_eq!(run(), run(), "builder is deterministic");
}
