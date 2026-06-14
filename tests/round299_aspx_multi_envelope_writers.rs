//! Round 299 — multi-envelope (`num_env > 1`) ASPX `aspx_data_1ch()` /
//! `aspx_data_2ch()` body writers.
//!
//! Until now the encoder's real-envelope body writers
//! (`write_aspx_data_{1,2}ch_real_envelope`, round 226) only emitted a
//! single FIXFIX envelope (`num_env == 1`, FREQ direction). The round-292
//! packer ([`oxideav_ac4::encoder_acpl3::dpcm_encode_qscf_envelopes`])
//! already produces a full multi-envelope `AspxEncodedEnvelope` sequence
//! with per-envelope FREQ / TIME direction selection, but nothing
//! consumed it at the body-writer level. This round closes that gap with
//! [`oxideav_ac4::encoder_acpl3::write_aspx_data_2ch_multi_envelope`] and
//! [`write_aspx_data_1ch_multi_envelope`] — FIXFIX bodies with
//! `tmp_num_env` set so the decoder derives `num_env = 1 << tmp_num_env`,
//! per-envelope `aspx_delta_dir` bits, and per-envelope SIGNAL / NOISE
//! ec_data honouring each envelope's chosen direction.
//!
//! Verification path mirrors round 226 but with `num_env = 2`:
//!
//! 1. Build an `AspxConfig` with `num_env_bits_fixfix = 1` (so
//!    `tmp_num_env` is 2 bits wide → `num_env` up to 8) and
//!    `signals_freq_res() == false` (the multi-envelope writer
//!    precondition — FIXFIX carries only one freq_res entry while SIGNAL
//!    ec_data walks `num_env` envelopes, so the high-res fallback must
//!    apply uniformly).
//! 2. Pack a hand-built `qscf[sbg][atsg]` matrix through the round-292
//!    packer to get per-envelope `AspxEncodedEnvelope` rows.
//! 3. Emit the body, re-parse the framing skeleton, call
//!    `parse_aspx_ec_data` per channel/type, then run
//!    `delta_decode_sig` / `delta_decode_noise` and assert the recovered
//!    `qscf` matrix matches the input exactly.
//!
//! Refs ETSI TS 103 190-1 §4.2.12.3 Table 51, §4.2.12.4 Table 52,
//! §4.2.12.5 Table 54, §4.2.12.8 Table 57, §5.7.6.3.4 Pseudocode 80 / 81.

use oxideav_ac4::aspx::{
    delta_decode_noise, delta_decode_sig, derive_aspx_frequency_tables, num_aspx_timeslots,
    parse_aspx_delta_dir, parse_aspx_ec_data, parse_aspx_framing, parse_aspx_hfgen_iwc_1ch,
    parse_aspx_hfgen_iwc_2ch, AspxConfig, AspxDataType, AspxFreqResMode, AspxIntClass,
    AspxMasterFreqScale, AspxQuantStep, AspxStereoMode,
};
use oxideav_ac4::encoder_acpl3::{
    dpcm_encode_qscf_envelopes, write_aspx_data_1ch_multi_envelope,
    write_aspx_data_2ch_multi_envelope, AspxEncodedEnvelope, AspxMultiEnvelopeChannel,
};
use oxideav_core::bits::{BitReader, BitWriter};

/// FIXFIX config with `num_env_bits_fixfix = 1` so `tmp_num_env` is 2 b
/// (num_env up to 8) and `signals_freq_res() == false` (DurationDependent
/// freq-res mode → no in-band aspx_freq_res bit).
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

/// Reconstruct a `qscf[sbg][atsg]` matrix from the decoder's per-envelope
/// `AspxHuffEnv` output, then assert it equals `expected`.
fn assert_qscf_round_trip(
    decoded: &[oxideav_ac4::aspx::AspxHuffEnv],
    num_sbg: u32,
    delta: i32,
    expected: &[Vec<i32>],
    is_noise: bool,
) {
    let qscf = if is_noise {
        delta_decode_noise(decoded, num_sbg, &[], delta)
    } else {
        delta_decode_sig(decoded, num_sbg, &[], delta)
    };
    assert_eq!(qscf.len(), expected.len(), "sbg count");
    for sbg in 0..expected.len() {
        for atsg in 0..expected[sbg].len() {
            assert_eq!(
                qscf[sbg][atsg], expected[sbg][atsg],
                "mismatch at sbg={sbg} atsg={atsg}"
            );
        }
    }
}

/// 2-ch, `num_env = 2`: hand-built SIGNAL / NOISE qscf matrices for both
/// channels round-trip through the writer → parser → delta-decoder chain.
#[test]
fn write_2ch_multi_envelope_round_trips() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let delta = 1;
    let num_env = 2u32;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    // qscf[sbg][atsg] for ch0 SIGNAL — 2 envelopes; a temporally stable
    // env 1 should pick TIME.
    let mk_sig = |seed: i32| -> Vec<Vec<i32>> {
        (0..num_sbg_sig as i32)
            .map(|sbg| vec![seed + sbg, seed + sbg + 1])
            .collect()
    };
    let mk_noise = |seed: i32| -> Vec<Vec<i32>> {
        (0..num_sbg_noise as i32)
            .map(|sbg| vec![seed + sbg, seed + sbg])
            .collect()
    };

    let ch0_sig = mk_sig(2);
    let ch1_sig = mk_sig(0);
    let ch0_noise = mk_noise(1);
    let ch1_noise = mk_noise(0);

    let ch0_sig_rows = dpcm_encode_qscf_envelopes(&ch0_sig, &[], delta, false);
    let ch1_sig_rows = dpcm_encode_qscf_envelopes(&ch1_sig, &[], delta, false);
    let ch0_noise_rows = dpcm_encode_qscf_envelopes(&ch0_noise, &[], delta, false);
    let ch1_noise_rows = dpcm_encode_qscf_envelopes(&ch1_noise, &[], delta, false);

    assert_eq!(ch0_sig_rows.len(), num_env as usize, "one row per envelope");

    let mut bw = BitWriter::new();
    write_aspx_data_2ch_multi_envelope(
        &mut bw,
        &cfg,
        num_env,
        AspxMultiEnvelopeChannel {
            sig: &ch0_sig_rows,
            noise: &ch0_noise_rows,
        },
        AspxMultiEnvelopeChannel {
            sig: &ch1_sig_rows,
            noise: &ch1_noise_rows,
        },
    )
    .expect("2ch multi-envelope writer succeeds");
    bw.align_to_byte();
    let bytes = bw.finish();

    // Re-parse framing skeleton.
    let mut br = BitReader::new(&bytes);
    assert_eq!(br.read_u32(3).unwrap(), 0, "xover = 0");
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    assert!(matches!(framing.int_class, AspxIntClass::FixFix));
    assert_eq!(framing.num_env, num_env, "num_env = 2");
    assert_eq!(framing.num_noise, 2, "num_noise = 2 when num_env > 1");
    let balance = br.read_bit().unwrap();
    assert!(balance, "aspx_balance = 1");
    let dd0 = parse_aspx_delta_dir(&mut br, &framing).expect("dd0");
    let dd1 = parse_aspx_delta_dir(&mut br, &framing).expect("dd1");
    assert_eq!(dd0.sig_delta_dir.len(), num_env as usize);
    assert_eq!(dd0.noise_delta_dir.len(), 2);
    let _hfgen = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        balance,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen 2ch");

    // num_env > 1 ⇒ qmode = cfg.quant_mode_env (no FIXFIX Fine clamp).
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

    assert_eq!(sig0.len(), num_env as usize);
    assert_qscf_round_trip(&sig0, num_sbg_sig, delta, &ch0_sig, false);
    assert_qscf_round_trip(&sig1, num_sbg_sig, delta, &ch1_sig, false);
    assert_qscf_round_trip(&noise0, num_sbg_noise, delta, &ch0_noise, true);
    assert_qscf_round_trip(&noise1, num_sbg_noise, delta, &ch1_noise, true);
}

/// 1-ch, `num_env = 2`: same round-trip through Table 51 (LEVEL only).
#[test]
fn write_1ch_multi_envelope_round_trips() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let delta = 1;
    let num_env = 2u32;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    let sig: Vec<Vec<i32>> = (0..num_sbg_sig as i32)
        .map(|sbg| vec![3 + sbg, 4 + sbg])
        .collect();
    let noise: Vec<Vec<i32>> = (0..num_sbg_noise as i32)
        .map(|sbg| vec![2 - sbg, 2 - sbg])
        .collect();

    let sig_rows = dpcm_encode_qscf_envelopes(&sig, &[], delta, false);
    let noise_rows = dpcm_encode_qscf_envelopes(&noise, &[], delta, false);

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
    .expect("1ch multi-envelope writer succeeds");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    assert_eq!(br.read_u32(3).unwrap(), 0);
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    assert!(matches!(framing.int_class, AspxIntClass::FixFix));
    assert_eq!(framing.num_env, num_env);
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("delta_dir");
    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen 1ch");

    let qmode = cfg.quant_mode_env;
    let sig_dec = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        qmode,
        AspxStereoMode::Level,
        &dd.sig_delta_dir,
        counts,
    )
    .expect("sig");
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
    .expect("noise");

    assert_qscf_round_trip(&sig_dec, num_sbg_sig, delta, &sig, false);
    assert_qscf_round_trip(&noise_dec, num_sbg_noise, delta, &noise, true);
}

/// A temporally stable second envelope is transmitted TIME by the packer
/// and the writer carries that direction onto the wire — the decoder's
/// `aspx_sig_delta_dir[1]` reads back `true`.
#[test]
fn time_direction_envelope_survives_round_trip() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let delta = 1;
    let num_env = 2u32;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;

    // Env 1 ≈ env 0 (per-sbg jitter only) so TIME beats FREQ.
    let sig: Vec<Vec<i32>> = (0..num_sbg_sig as i32)
        .map(|sbg| {
            let base = 5 + 3 * sbg; // strong FREQ shape
            vec![base, base + 1] // env1 = env0 + 1 (cheap TIME)
        })
        .collect();
    let sig_rows = dpcm_encode_qscf_envelopes(&sig, &[], delta, false);
    assert!(
        sig_rows[1].direction_time,
        "stable env 1 packs as TIME (precondition)"
    );

    // NOISE: single trivial pair.
    let noise = vec![vec![0, 0]];
    let noise_rows = dpcm_encode_qscf_envelopes(&noise, &[], delta, false);

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
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("delta_dir");
    // SIGNAL env 1 direction bit must come back TIME.
    assert!(!dd.sig_delta_dir[0], "env 0 FREQ");
    assert!(dd.sig_delta_dir[1], "env 1 TIME survives the wire");

    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
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
    .expect("sig");
    assert_qscf_round_trip(&sig_dec, num_sbg_sig, delta, &sig, false);
}

/// A short caller slice (fewer envelopes than `num_env`) zero-pads the
/// missing envelopes — they emit all-zero FREQ rows and decode to zero
/// qscf columns.
#[test]
fn short_envelope_slice_zero_pads() {
    let cfg = multi_cfg();
    let frame_len_base: u32 = 2048;
    let delta = 1;
    let num_env = 2u32;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;

    // Only one SIGNAL envelope supplied (env 1 zero-pads). Its FREQ
    // DPCM row is [F0 = 4, DF = 0, …] so it decodes to a constant-4
    // qscf column.
    let mut env0_vals = vec![0i32; num_sbg_sig as usize];
    if !env0_vals.is_empty() {
        env0_vals[0] = 4;
    }
    let env0 = AspxEncodedEnvelope {
        values: env0_vals,
        direction_time: false,
    };
    let sig_rows = vec![env0];
    // No NOISE rows at all (both zero-pad).
    let noise_rows: Vec<AspxEncodedEnvelope> = Vec::new();

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
    .expect("writer tolerates short slices");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    br.read_u32(3).unwrap();
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("delta_dir");
    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
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
    .expect("sig");
    let qscf = delta_decode_sig(&sig_dec, num_sbg_sig, &[], delta);
    // Env 0 recovers the constant-4 FREQ row; env 1 is all-zero.
    for (sbg, col) in qscf.iter().enumerate().take(num_sbg_sig as usize) {
        assert_eq!(col[0], 4, "env0 sbg{sbg}");
        assert_eq!(col[1], 0, "env1 zero-padded sbg{sbg}");
    }
}

/// Preconditions are enforced: signals_freq_res() and non-power-of-two
/// num_env are rejected; num_env exceeding tmp_num_env capacity rejected.
#[test]
fn writer_rejects_invalid_configs() {
    // signals_freq_res() == true rejected.
    let mut cfg_fr = multi_cfg();
    cfg_fr.freq_res_mode = AspxFreqResMode::Signalled;
    assert!(cfg_fr.signals_freq_res());
    let empty: Vec<AspxEncodedEnvelope> = Vec::new();
    let ch = AspxMultiEnvelopeChannel {
        sig: &empty,
        noise: &empty,
    };
    let mut bw = BitWriter::new();
    assert!(write_aspx_data_1ch_multi_envelope(&mut bw, &cfg_fr, 2, ch.clone()).is_err());

    // num_env = 3 (not a power of two) rejected.
    let cfg = multi_cfg();
    let mut bw = BitWriter::new();
    assert!(write_aspx_data_1ch_multi_envelope(&mut bw, &cfg, 3, ch.clone()).is_err());

    // num_env = 0 rejected.
    let mut bw = BitWriter::new();
    assert!(write_aspx_data_1ch_multi_envelope(&mut bw, &cfg, 0, ch.clone()).is_err());

    // num_env = 8 exceeds tmp_num_env capacity (num_env_bits_fixfix = 1 →
    // 2 b → tmp_num_env in 0..3 → max num_env = 8 is tmp_num_env = 3,
    // which is representable; num_env = 16 (tmp_num_env = 4) is not).
    let mut bw = BitWriter::new();
    assert!(write_aspx_data_1ch_multi_envelope(&mut bw, &cfg, 16, ch).is_err());
}

/// Determinism: the same inputs produce byte-identical output.
#[test]
fn writer_is_deterministic() {
    let cfg = multi_cfg();
    let delta = 1;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_sbg_sig = tables.counts.num_sbg_sig_highres;
    let sig: Vec<Vec<i32>> = (0..num_sbg_sig as i32)
        .map(|sbg| vec![1 + sbg, 2 + sbg])
        .collect();
    let sig_rows = dpcm_encode_qscf_envelopes(&sig, &[], delta, false);
    let noise = vec![vec![0, 0]];
    let noise_rows = dpcm_encode_qscf_envelopes(&noise, &[], delta, false);

    let emit = || {
        let mut bw = BitWriter::new();
        write_aspx_data_1ch_multi_envelope(
            &mut bw,
            &cfg,
            2,
            AspxMultiEnvelopeChannel {
                sig: &sig_rows,
                noise: &noise_rows,
            },
        )
        .unwrap();
        bw.align_to_byte();
        bw.finish()
    };
    assert_eq!(emit(), emit(), "multi-envelope writer is deterministic");
}
