//! Round 226 — `write_aspx_data_2ch_real_envelope()` /
//! `write_aspx_data_1ch_real_envelope()` round-trip pinning.
//!
//! Tests the round-226 ASPX envelope builders that take caller-provided
//! per-channel SIGNAL / NOISE F0 + DF quant-index sequences and emit a
//! Table-51 / Table-52 body whose envelope ec_data values exactly
//! match the caller's inputs (subject to the codebook clamp the
//! round-219 value-emitting helpers apply).
//!
//! Verification path:
//!
//! 1. Build an `AspxConfig` whose derived sbg counts are small enough
//!    to enumerate by hand (`start_freq = 0`, `stop_freq = 0`,
//!    `noise_sbg = 0`, etc.).
//! 2. Call the new builder with a deterministic per-channel envelope
//!    quant-index vector.
//! 3. Re-parse the framing skeleton (`aspx_framing` + `aspx_balance` +
//!    `aspx_delta_dir` + `aspx_hfgen_iwc_2ch` / `_1ch`) so the
//!    bitreader is sitting at the first `aspx_ec_data` call.
//! 4. Call `parse_aspx_ec_data` four times (2ch: ch0 SIG, ch1 SIG,
//!    ch0 NOISE, ch1 NOISE — with the stereo_mode rule LEVEL / BALANCE
//!    per Table 52) or twice (1ch: SIG then NOISE — both LEVEL).
//! 5. Assert the decoded envelope values equal the caller's inputs
//!    (after the codebook clamp the round-219 helpers apply).
//!
//! Refs ETSI TS 103 190-1:
//! * §4.2.12.3 Table 51 (`aspx_data_1ch()`).
//! * §4.2.12.4 Table 52 (`aspx_data_2ch()`).
//! * §4.2.12.8 Table 57 (`aspx_ec_data()`).
//! * §4.3.10.4.9 Tables 130.. — codebook selection.
//! * §A.2 Tables A.16..A.33 — eighteen ASPX Huffman codebooks.

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, num_aspx_timeslots, parse_aspx_delta_dir, parse_aspx_ec_data,
    parse_aspx_framing, parse_aspx_hfgen_iwc_1ch, parse_aspx_hfgen_iwc_2ch, AspxConfig,
    AspxDataType, AspxFreqResMode, AspxIntClass, AspxMasterFreqScale, AspxQuantStep,
    AspxStereoMode,
};
use oxideav_ac4::encoder_acpl3::{
    write_aspx_data_1ch_real_envelope, write_aspx_data_2ch_real_envelope, AspxRealEnvelopeChannel,
};
use oxideav_core::bits::{BitReader, BitWriter};

/// A small, well-behaved config: signals_freq_res = false so the
/// SIGNAL band count comes from the high-res template; noise_sbg = 0
/// so `num_noise_sbgroups = 1`; FIXFIX with `num_env_bits_fixfix = 0`
/// so `tmp_num_env` is 1 bit wide → `num_env = 1`.
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

/// 2-ch round-trip: deterministic envelope quant indices for both
/// channels. Decode the body and confirm the recovered SIGNAL /
/// NOISE vectors equal the inputs entry-for-entry.
#[test]
fn write_aspx_data_2ch_real_envelope_round_trips_through_parser() {
    let cfg = small_cfg();
    let frame_len_base: u32 = 2048;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables derived");
    let counts = tables.counts;
    // signals_freq_res() == false ⇒ SIGNAL band count = high-res.
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    // Per-channel envelope quant indices.
    let ch0_sig: Vec<i32> = (0..num_sbg_sig as i32)
        .map(|i| if i == 0 { 5 } else { -i })
        .collect();
    let ch1_sig: Vec<i32> = (0..num_sbg_sig as i32)
        .map(|i| if i == 0 { 3 } else { i })
        .collect();
    let ch0_noise: Vec<i32> = (0..num_sbg_noise as i32).map(|i| 2 + i).collect();
    let ch1_noise: Vec<i32> = (0..num_sbg_noise as i32).map(|i| 1 - i).collect();

    let mut bw = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &ch0_sig,
            noise: &ch0_noise,
        },
        AspxRealEnvelopeChannel {
            sig: &ch1_sig,
            noise: &ch1_noise,
        },
    )
    .expect("writer succeeds for small cfg");
    bw.align_to_byte();
    let bytes = bw.finish();

    // Re-parse the framing skeleton up to the first aspx_ec_data() call.
    let mut br = BitReader::new(&bytes);
    let xover = br.read_u32(3).unwrap();
    assert_eq!(xover, 0);
    let nats = num_aspx_timeslots(frame_len_base);
    let framing_ch0 = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    assert!(matches!(framing_ch0.int_class, AspxIntClass::FixFix));
    assert_eq!(framing_ch0.num_env, 1);
    let balance = br.read_bit().unwrap();
    assert!(balance, "writer emits aspx_balance = 1");
    let dd0 = parse_aspx_delta_dir(&mut br, &framing_ch0).expect("dd0");
    let dd1 = parse_aspx_delta_dir(&mut br, &framing_ch0).expect("dd1");
    let _hfgen = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        balance,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen 2ch");

    // qmode forced Fine on FIXFIX + num_env == 1.
    let qmode = AspxQuantStep::Fine;
    let sig0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing_ch0.num_env,
        &framing_ch0.freq_res,
        qmode,
        AspxStereoMode::Level,
        &dd0.sig_delta_dir,
        counts,
    )
    .expect("ch0 sig");
    let sig1 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing_ch0.num_env,
        &framing_ch0.freq_res,
        qmode,
        AspxStereoMode::Balance,
        &dd1.sig_delta_dir,
        counts,
    )
    .expect("ch1 sig");
    let noise0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Noise,
        framing_ch0.num_noise,
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
        framing_ch0.num_noise,
        &[],
        AspxQuantStep::Fine,
        AspxStereoMode::Balance,
        &dd1.noise_delta_dir,
        counts,
    )
    .expect("ch1 noise");

    assert_eq!(sig0.len(), 1);
    assert_eq!(sig1.len(), 1);
    assert_eq!(noise0.len(), 1);
    assert_eq!(noise1.len(), 1);
    assert_eq!(sig0[0].values, ch0_sig);
    assert_eq!(sig1[0].values, ch1_sig);
    assert_eq!(noise0[0].values, ch0_noise);
    assert_eq!(noise1[0].values, ch1_noise);
}

/// 1-ch round-trip: same shape but only one channel through Table 51.
#[test]
fn write_aspx_data_1ch_real_envelope_round_trips_through_parser() {
    let cfg = small_cfg();
    let frame_len_base: u32 = 2048;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables derived");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    let sig: Vec<i32> = (0..num_sbg_sig as i32)
        .map(|i| if i == 0 { 7 } else { -3 + i })
        .collect();
    let noise: Vec<i32> = (0..num_sbg_noise as i32).map(|i| 4 - i).collect();

    let mut bw = BitWriter::new();
    write_aspx_data_1ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise,
        },
    )
    .expect("writer succeeds for small cfg");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let xover = br.read_u32(3).unwrap();
    assert_eq!(xover, 0);
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    assert!(matches!(framing.int_class, AspxIntClass::FixFix));
    assert_eq!(framing.num_env, 1);
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("delta_dir");
    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen 1ch");

    let qmode = AspxQuantStep::Fine;
    let sigp = parse_aspx_ec_data(
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
    let noisep = parse_aspx_ec_data(
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

    assert_eq!(sigp.len(), 1);
    assert_eq!(noisep.len(), 1);
    assert_eq!(sigp[0].values, sig);
    assert_eq!(noisep[0].values, noise);
}

/// Short input slices: writer zero-pads trailing envelope positions
/// when the caller's slice is shorter than `num_sbg`.
#[test]
fn write_aspx_data_2ch_real_envelope_zero_pads_short_inputs() {
    let cfg = small_cfg();
    let frame_len_base: u32 = 2048;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    assert!(
        num_sbg_sig >= 2,
        "small_cfg must derive ≥ 2 SIGNAL sbg for this assertion"
    );

    // Pass only F0 — DF positions get zero-padded by the writer.
    let ch_sig_short: Vec<i32> = vec![6];
    let ch_noise_short: Vec<i32> = vec![2];

    let mut bw = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &ch_sig_short,
            noise: &ch_noise_short,
        },
        AspxRealEnvelopeChannel {
            sig: &[],
            noise: &[],
        },
    )
    .expect("writer succeeds");
    bw.align_to_byte();
    let bytes = bw.finish();

    // Re-parse and check the SIGNAL / NOISE vectors line up.
    let mut br = BitReader::new(&bytes);
    let _ = br.read_u32(3).unwrap();
    let nats = num_aspx_timeslots(frame_len_base);
    let framing_ch0 = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    let _balance = br.read_bit().unwrap();
    let dd0 = parse_aspx_delta_dir(&mut br, &framing_ch0).expect("dd0");
    let dd1 = parse_aspx_delta_dir(&mut br, &framing_ch0).expect("dd1");
    let _ = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        true,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen");

    let sig0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing_ch0.num_env,
        &framing_ch0.freq_res,
        AspxQuantStep::Fine,
        AspxStereoMode::Level,
        &dd0.sig_delta_dir,
        counts,
    )
    .expect("ch0 sig");
    let sig1 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing_ch0.num_env,
        &framing_ch0.freq_res,
        AspxQuantStep::Fine,
        AspxStereoMode::Balance,
        &dd1.sig_delta_dir,
        counts,
    )
    .expect("ch1 sig");
    let noise0 = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Noise,
        framing_ch0.num_noise,
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
        framing_ch0.num_noise,
        &[],
        AspxQuantStep::Fine,
        AspxStereoMode::Balance,
        &dd1.noise_delta_dir,
        counts,
    )
    .expect("ch1 noise");

    // ch0 SIGNAL: F0 = 6, trailing = 0; ch1 SIGNAL: all zeros.
    assert_eq!(sig0[0].values[0], 6);
    for v in &sig0[0].values[1..] {
        assert_eq!(*v, 0);
    }
    for v in &sig1[0].values {
        assert_eq!(*v, 0);
    }
    // NOISE: ch0 = [2, 0, ...] (or just [2] if num_sbg_noise == 1).
    assert_eq!(noise0[0].values[0], 2);
    for v in &noise0[0].values[1..] {
        assert_eq!(*v, 0);
    }
    for v in &noise1[0].values {
        assert_eq!(*v, 0);
    }
}

/// Determinism: the same input pair produces the same byte stream
/// across repeated invocations.
#[test]
fn write_aspx_data_2ch_real_envelope_is_byte_deterministic() {
    let cfg = small_cfg();
    let ch0_sig: Vec<i32> = vec![4, -2, 1, 0, -1];
    let ch0_noise: Vec<i32> = vec![3];
    let ch1_sig: Vec<i32> = vec![1, 0, 0, 0, 0];
    let ch1_noise: Vec<i32> = vec![-2];

    let run = || -> Vec<u8> {
        let mut bw = BitWriter::new();
        write_aspx_data_2ch_real_envelope(
            &mut bw,
            &cfg,
            AspxRealEnvelopeChannel {
                sig: &ch0_sig,
                noise: &ch0_noise,
            },
            AspxRealEnvelopeChannel {
                sig: &ch1_sig,
                noise: &ch1_noise,
            },
        )
        .expect("writer succeeds");
        bw.align_to_byte();
        bw.finish()
    };
    assert_eq!(run(), run());
}

/// All-zero envelope inputs reproduce the round-219 zero-pick stream
/// the parser's F0 decode resolves to symbol 0, exercising the
/// boundary between the two writer families.
#[test]
fn write_aspx_data_2ch_real_envelope_all_zero_inputs_decode_to_zero() {
    let cfg = small_cfg();
    let frame_len_base: u32 = 2048;
    let mut bw = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &[],
            noise: &[],
        },
        AspxRealEnvelopeChannel {
            sig: &[],
            noise: &[],
        },
    )
    .expect("writer succeeds");
    bw.align_to_byte();
    let bytes = bw.finish();

    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let mut br = BitReader::new(&bytes);
    let _ = br.read_u32(3).unwrap();
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    let _balance = br.read_bit().unwrap();
    let dd0 = parse_aspx_delta_dir(&mut br, &framing).expect("dd0");
    let dd1 = parse_aspx_delta_dir(&mut br, &framing).expect("dd1");
    let _ = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        true,
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
    .expect("ch0 sig");
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
    .expect("ch1 sig");
    for v in &sig0[0].values {
        assert_eq!(*v, 0);
    }
    for v in &sig1[0].values {
        assert_eq!(*v, 0);
    }
}

/// Different inputs produce different bytes (sanity check that the
/// caller's per-band quant indices actually reach the wire).
#[test]
fn write_aspx_data_2ch_real_envelope_different_inputs_diverge_bytes() {
    let cfg = small_cfg();
    let baseline_sig: Vec<i32> = vec![1, 0, 0, 0, 0];
    let baseline_noise: Vec<i32> = vec![1];
    let tweaked_sig: Vec<i32> = vec![1, 5, -3, 0, 0];
    let tweaked_noise: Vec<i32> = vec![4];

    let mut bw_a = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw_a,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &baseline_sig,
            noise: &baseline_noise,
        },
        AspxRealEnvelopeChannel {
            sig: &baseline_sig,
            noise: &baseline_noise,
        },
    )
    .expect("writer a succeeds");
    bw_a.align_to_byte();
    let a = bw_a.finish();

    let mut bw_b = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw_b,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &tweaked_sig,
            noise: &tweaked_noise,
        },
        AspxRealEnvelopeChannel {
            sig: &baseline_sig,
            noise: &baseline_noise,
        },
    )
    .expect("writer b succeeds");
    bw_b.align_to_byte();
    let b = bw_b.finish();

    assert_ne!(a, b);
}

/// 1-ch determinism counterpart.
#[test]
fn write_aspx_data_1ch_real_envelope_is_byte_deterministic() {
    let cfg = small_cfg();
    let sig: Vec<i32> = vec![2, -1, 3, 0, -2];
    let noise: Vec<i32> = vec![1];
    let run = || -> Vec<u8> {
        let mut bw = BitWriter::new();
        write_aspx_data_1ch_real_envelope(
            &mut bw,
            &cfg,
            AspxRealEnvelopeChannel {
                sig: &sig,
                noise: &noise,
            },
        )
        .expect("writer succeeds");
        bw.align_to_byte();
        bw.finish()
    };
    assert_eq!(run(), run());
}

/// DF values outside the codebook's symmetric `±cb_off` range saturate
/// to the codebook edge. With `quant = Fine`, `stereo = Level` the DF
/// codebook is Table A.17 (`codebook_length = 141`, `cb_off = 70`), so
/// an input `+1000` clamps to symbol_index 140 → decoded delta = 70.
#[test]
fn write_aspx_data_2ch_real_envelope_clamps_out_of_range_df() {
    let cfg = small_cfg();
    let frame_len_base: u32 = 2048;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    assert!(num_sbg_sig >= 2, "need ≥ 2 SBGs to exercise a DF slot");

    let mut sig = vec![0i32; num_sbg_sig as usize];
    sig[1] = 1000; // out-of-range delta.
    let noise = vec![0i32];

    let mut bw = BitWriter::new();
    write_aspx_data_2ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise,
        },
        AspxRealEnvelopeChannel {
            sig: &[],
            noise: &[],
        },
    )
    .expect("writer succeeds");
    bw.align_to_byte();
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let _ = br.read_u32(3).unwrap();
    let nats = num_aspx_timeslots(frame_len_base);
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    let _balance = br.read_bit().unwrap();
    let dd0 = parse_aspx_delta_dir(&mut br, &framing).expect("dd0");
    let _dd1 = parse_aspx_delta_dir(&mut br, &framing).expect("dd1");
    let _ = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        true,
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
    .expect("ch0 sig");

    // DF slot 1 should clamp to the codebook's +cb_off edge (= +70 for
    // ASPX_HCB_ENV_LEVEL_15_DF). F0 stayed at 0.
    assert_eq!(sig0[0].values[0], 0);
    assert_eq!(sig0[0].values[1], 70);
}
