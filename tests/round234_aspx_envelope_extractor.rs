//! Round 234 — encoder-side ASPX envelope extractor.
//!
//! Pins the round-234 inverse-of-Pseudocode-{80, 81, 82, 83} primitives
//! in `encoder_acpl3`:
//!
//! * `quantize_sig_scf` — invert `scf = n_subbands · 2^(qscf/a)`
//!   (Pseudocode 82).
//! * `quantize_noise_scf` — invert `scf = 2^(6 − qscf_noise)`
//!   (Pseudocode 83).
//! * `freq_dpcm_encode_qscf` — invert the FREQ-direction DPCM accumulator
//!   `qscf[sbg] = sum(values[0..=sbg])` of Pseudocode 80 / 81.
//! * `extract_aspx_sig_envelope_indices` / `extract_aspx_noise_envelope_indices` —
//!   the per-channel composition `scf[] → qscf[] → [F0, DF₁, …]`.
//! * `build_aspx_real_envelope_channel` — the per-channel
//!   `AspxEnvelopeScfChannel → (sig, noise)` wrapper.
//!
//! The high-level round-trip wires the extractor output into the
//! round-226 `write_aspx_data_2ch_real_envelope()` builder, re-parses
//! the framing skeleton + `aspx_ec_data` envelope payload back to
//! `AspxHuffEnv::values`, runs the decoder's `delta_decode_sig` +
//! `delta_decode_noise` + `dequantize_sig_scf` + `dequantize_noise_scf`,
//! and asserts the recovered `scf` matches the caller's input vector
//! within the per-band rounding of `round(a · log2(scf / 64))` /
//! `round(6 − log2(scf))`.
//!
//! Refs ETSI TS 103 190-1:
//! * §5.7.6.3.4 Pseudocode 80 (signal delta-decode).
//! * §5.7.6.3.4 Pseudocode 81 (noise delta-decode).
//! * §5.7.6.3.5 Pseudocode 82 (signal dequantize).
//! * §5.7.6.3.5 Pseudocode 83 (noise dequantize).
//! * §4.2.12.4 Table 52 (`aspx_data_2ch()`).

use oxideav_ac4::aspx::{
    delta_decode_noise, delta_decode_sig, dequantize_noise_scf, dequantize_sig_scf,
    derive_aspx_frequency_tables, num_aspx_timeslots, parse_aspx_delta_dir, parse_aspx_ec_data,
    parse_aspx_framing, parse_aspx_hfgen_iwc_2ch, AspxConfig, AspxDataType, AspxFreqResMode,
    AspxIntClass, AspxMasterFreqScale, AspxQuantStep, AspxStereoMode,
};
use oxideav_ac4::encoder_acpl3::{
    build_aspx_real_envelope_channel, extract_aspx_noise_envelope_indices,
    extract_aspx_sig_envelope_indices, freq_dpcm_encode_qscf, quantize_noise_scf, quantize_sig_scf,
    write_aspx_data_2ch_real_envelope, AspxEnvelopeScfChannel, AspxRealEnvelopeChannel,
};
use oxideav_core::bits::{BitReader, BitWriter};

/// Forward-inverse identity at sample envelope-scf points where the
/// Pseudocode-82 expression `n · 2^(q/a)` lands on an integer `q` for
/// `a = 2` (Fine).
#[test]
fn quantize_sig_scf_inverts_pseudocode_82_fine() {
    // Pseudocode 82: scf = 64 · 2^(q/2).
    //   q =  0 → scf = 64
    //   q =  2 → scf = 128
    //   q = -2 → scf = 32
    //   q =  4 → scf = 256
    //   q = -4 → scf = 16
    for &q_in in &[-4_i32, -2, 0, 2, 4, 6, -6, 10, -10] {
        let scf = 64.0_f32 * 2_f32.powf(q_in as f32 / 2.0);
        let q_out = quantize_sig_scf(scf, AspxQuantStep::Fine, 64);
        assert_eq!(
            q_out, q_in,
            "Fine extractor at scf={scf}: expected q={q_in}, got {q_out}"
        );
    }
}

/// Forward-inverse identity for the 3 dB Coarse step (`a = 1`).
#[test]
fn quantize_sig_scf_inverts_pseudocode_82_coarse() {
    // Pseudocode 82 (Coarse): scf = 64 · 2^q.
    for &q_in in &[-3_i32, -1, 0, 1, 3, 5] {
        let scf = 64.0_f32 * 2_f32.powf(q_in as f32);
        let q_out = quantize_sig_scf(scf, AspxQuantStep::Coarse, 64);
        assert_eq!(
            q_out, q_in,
            "Coarse extractor at scf={scf}: expected q={q_in}, got {q_out}"
        );
    }
}

/// Non-positive scf (the spec's `scf[0] = scf[1]` carry-through path
/// can leave a negative entry — caller may also pass 0 for a silent
/// band) clamps to a finite quant index instead of producing `-inf`.
#[test]
fn quantize_sig_scf_clamps_non_positive() {
    let q_zero = quantize_sig_scf(0.0, AspxQuantStep::Fine, 64);
    let q_neg = quantize_sig_scf(-1.0e-3, AspxQuantStep::Fine, 64);
    // Non-positive input on a log-based forward stage can land at a
    // very negative integer, but the result must always be a real i32
    // (never NaN / panic).
    assert!(q_zero <= 0);
    assert!(q_neg <= 0);
}

/// Forward-inverse identity for noise: Pseudocode 83 (`scf = 2^(6 − q)`).
#[test]
fn quantize_noise_scf_inverts_pseudocode_83() {
    // scf_noise(q = 0) = 64, scf_noise(q = 6) = 1, scf_noise(q = -2) = 256.
    for &q_in in &[-4_i32, -2, 0, 1, 3, 6, 8] {
        let scf = 2_f32.powi(6 - q_in);
        let q_out = quantize_noise_scf(scf);
        assert_eq!(
            q_out, q_in,
            "Noise extractor at scf={scf}: expected q={q_in}, got {q_out}"
        );
    }
}

/// FREQ-direction DPCM inverse: caller-supplied qscf vector `[a, b, c]`
/// should encode to `[a, b-a, c-b]` so the decoder's accumulator
/// recovers `[a, b, c]`.
#[test]
fn freq_dpcm_encode_qscf_inverts_pseudocode_80_accumulator() {
    let qscf: &[i32] = &[5, 7, 3, -1, 0];
    let dpcm = freq_dpcm_encode_qscf(qscf);
    assert_eq!(dpcm, vec![5, 2, -4, -4, 1]);

    // Walk the decoder's accumulator forward over the DPCM payload —
    // this is exactly `delta_decode_sig` / `delta_decode_noise` with
    // delta = 1 and direction_time = false.
    let mut acc = 0;
    let mut recovered = Vec::new();
    for v in &dpcm {
        acc += v;
        recovered.push(acc);
    }
    assert_eq!(recovered, qscf);
}

#[test]
fn freq_dpcm_encode_qscf_empty_input_round_trips() {
    let dpcm = freq_dpcm_encode_qscf(&[]);
    assert!(dpcm.is_empty());
}

#[test]
fn freq_dpcm_encode_qscf_single_band_passes_through() {
    let dpcm = freq_dpcm_encode_qscf(&[11]);
    assert_eq!(dpcm, vec![11]);
}

/// End-to-end: feed the signal extractor a Pseudocode-82-exact scf
/// vector and confirm the on-wire DPCM payload, when accumulated and
/// dequantized, reproduces the input scf.
#[test]
fn extract_aspx_sig_envelope_indices_round_trips_via_decoder_pipeline() {
    let scf_input: Vec<f32> = [0, 2, -2, 4, -4, 6]
        .iter()
        .map(|&q| 64.0_f32 * 2_f32.powf(q as f32 / 2.0))
        .collect();
    let values = extract_aspx_sig_envelope_indices(&scf_input, AspxQuantStep::Fine, 64);
    assert_eq!(values.len(), scf_input.len());

    // Reconstruct qscf by accumulating the DPCM stream.
    let mut acc = 0;
    let qscf_recovered: Vec<i32> = values
        .iter()
        .map(|v| {
            acc += v;
            acc
        })
        .collect();
    // Dequantize through Pseudocode 82 and confirm it matches.
    let scf_recovered: Vec<f32> = qscf_recovered
        .iter()
        .map(|&q| 64.0_f32 * 2_f32.powf(q as f32 / 2.0))
        .collect();
    for (expected, got) in scf_input.iter().zip(scf_recovered.iter()) {
        let rel = (got - expected).abs() / expected.abs().max(1e-9);
        assert!(
            rel < 1e-5,
            "sig dequantize round-trip mismatch: expected {expected}, got {got}"
        );
    }
}

/// End-to-end on the noise side: scf = 2^(6 − q) maps to integer q for
/// a few representative values; the extractor's DPCM stream accumulated
/// then dequantized should recover the input.
#[test]
fn extract_aspx_noise_envelope_indices_round_trips_via_decoder_pipeline() {
    let scf_input: Vec<f32> = [0, 1, -1, 3, -2, 5]
        .iter()
        .map(|&q| 2_f32.powi(6 - q))
        .collect();
    let values = extract_aspx_noise_envelope_indices(&scf_input);
    assert_eq!(values.len(), scf_input.len());

    let mut acc = 0;
    let qscf_recovered: Vec<i32> = values
        .iter()
        .map(|v| {
            acc += v;
            acc
        })
        .collect();
    let scf_recovered: Vec<f32> = qscf_recovered.iter().map(|&q| 2_f32.powi(6 - q)).collect();
    for (expected, got) in scf_input.iter().zip(scf_recovered.iter()) {
        let rel = (got - expected).abs() / expected.abs().max(1e-9);
        assert!(
            rel < 1e-5,
            "noise dequantize round-trip mismatch: expected {expected}, got {got}"
        );
    }
}

/// Wrapper: `build_aspx_real_envelope_channel` runs both extractors
/// and returns owned `(sig, noise)` Vecs.
#[test]
fn build_aspx_real_envelope_channel_produces_owned_pair() {
    let sig: Vec<f32> = [0, 2, 4]
        .iter()
        .map(|&q: &i32| 64.0_f32 * 2_f32.powf(q as f32 / 2.0))
        .collect();
    let noise: Vec<f32> = [0, 1, 3].iter().map(|&q: &i32| 2_f32.powi(6 - q)).collect();
    let ch = AspxEnvelopeScfChannel {
        sig: &sig,
        noise: &noise,
    };
    let (sig_idx, noise_idx) = build_aspx_real_envelope_channel(&ch, AspxQuantStep::Fine, 64);
    // Direct call should match.
    assert_eq!(
        sig_idx,
        extract_aspx_sig_envelope_indices(&sig, AspxQuantStep::Fine, 64)
    );
    assert_eq!(noise_idx, extract_aspx_noise_envelope_indices(&noise));
}

/// Small AspxConfig matching the round-226 test's `small_cfg`.
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

/// Full encoder→decoder loop: build the round-226 2ch body from caller
/// scf vectors via `build_aspx_real_envelope_channel`, re-parse through
/// `parse_aspx_ec_data` + the decoder's `delta_decode_*` +
/// `dequantize_*_scf`, confirm the recovered `scf` matches the input
/// within the per-band Pseudocode-82 / 83 rounding tolerance.
#[test]
fn aspx_2ch_real_envelope_writer_round_trips_scf_through_decoder() {
    let cfg = small_cfg();
    let frame_len_base: u32 = 2048;
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables derived");
    let counts = tables.counts;
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    // Build a per-channel scf payload exactly on the Pseudocode-82/83
    // integer-quant grid so the rounding step is identity.
    let ch0_sig_scf: Vec<f32> = (0..num_sbg_sig)
        .map(|i| 64.0_f32 * 2_f32.powf((i as i32) as f32 / 2.0))
        .collect();
    let ch1_sig_scf: Vec<f32> = (0..num_sbg_sig)
        .map(|i| 64.0_f32 * 2_f32.powf((1 + i as i32) as f32 / 2.0))
        .collect();
    let ch0_noise_scf: Vec<f32> = (0..num_sbg_noise)
        .map(|i| 2_f32.powi(6 - i as i32))
        .collect();
    let ch1_noise_scf: Vec<f32> = (0..num_sbg_noise)
        .map(|i| 2_f32.powi(6 - (1 + i as i32)))
        .collect();

    let (ch0_sig, ch0_noise) = build_aspx_real_envelope_channel(
        &AspxEnvelopeScfChannel {
            sig: &ch0_sig_scf,
            noise: &ch0_noise_scf,
        },
        cfg.quant_mode_env,
        64,
    );
    let (ch1_sig, ch1_noise) = build_aspx_real_envelope_channel(
        &AspxEnvelopeScfChannel {
            sig: &ch1_sig_scf,
            noise: &ch1_noise_scf,
        },
        cfg.quant_mode_env,
        64,
    );

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
    let bytes = bw.finish();

    // Re-parse framing skeleton up to first aspx_ec_data() call.
    let mut br = BitReader::new(&bytes);
    let _xover = br.read_u32(3).unwrap();
    let nats = num_aspx_timeslots(frame_len_base);
    let framing_ch0 = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    assert!(matches!(framing_ch0.int_class, AspxIntClass::FixFix));
    assert_eq!(framing_ch0.num_env, 1);
    let balance = br.read_bit().unwrap();
    assert!(balance);
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

    // Run the decoder's delta-decode + dequantize, confirm we recover
    // the input scf vectors within the per-band rounding.
    let qscf0_sig = delta_decode_sig(&sig0, num_sbg_sig, &[], 1);
    let qscf1_sig = delta_decode_sig(&sig1, num_sbg_sig, &[], 1);
    let qscf0_noise = delta_decode_noise(&noise0, num_sbg_noise, &[], 1);
    let qscf1_noise = delta_decode_noise(&noise1, num_sbg_noise, &[], 1);

    let dd0_sig_bits: Vec<bool> = dd0.sig_delta_dir.to_vec();
    let dd1_sig_bits: Vec<bool> = dd1.sig_delta_dir.to_vec();
    let scf0_sig = dequantize_sig_scf(&qscf0_sig, qmode, &dd0_sig_bits, 64);
    let scf1_sig = dequantize_sig_scf(&qscf1_sig, qmode, &dd1_sig_bits, 64);
    let scf0_noise = dequantize_noise_scf(&qscf0_noise);
    let scf1_noise = dequantize_noise_scf(&qscf1_noise);

    // scf_*[sbg][atsg=0]: pull the first envelope column out and compare.
    let column = |m: &Vec<Vec<f32>>| -> Vec<f32> { m.iter().map(|row| row[0]).collect() };
    let scf0_sig_col = column(&scf0_sig);
    let scf1_sig_col = column(&scf1_sig);
    let scf0_noise_col = column(&scf0_noise);
    let scf1_noise_col = column(&scf1_noise);

    for (e, g) in ch0_sig_scf.iter().zip(scf0_sig_col.iter()) {
        let rel = (g - e).abs() / e.abs().max(1e-9);
        assert!(rel < 1e-5, "ch0 sig scf mismatch: expected {e}, got {g}");
    }
    for (e, g) in ch1_sig_scf.iter().zip(scf1_sig_col.iter()) {
        let rel = (g - e).abs() / e.abs().max(1e-9);
        assert!(rel < 1e-5, "ch1 sig scf mismatch: expected {e}, got {g}");
    }
    for (e, g) in ch0_noise_scf.iter().zip(scf0_noise_col.iter()) {
        let rel = (g - e).abs() / e.abs().max(1e-9);
        assert!(rel < 1e-5, "ch0 noise scf mismatch: expected {e}, got {g}");
    }
    for (e, g) in ch1_noise_scf.iter().zip(scf1_noise_col.iter()) {
        let rel = (g - e).abs() / e.abs().max(1e-9);
        assert!(rel < 1e-5, "ch1 noise scf mismatch: expected {e}, got {g}");
    }
}

/// Determinism: same input scf vectors produce identical extractor
/// output bytes across repeated invocations.
#[test]
fn extractor_is_byte_deterministic() {
    let sig_scf: Vec<f32> = (0..4)
        .map(|i| 64.0_f32 * 2_f32.powf(i as f32 / 2.0))
        .collect();
    let noise_scf: Vec<f32> = (0..3).map(|i| 2_f32.powi(6 - i)).collect();
    let a = extract_aspx_sig_envelope_indices(&sig_scf, AspxQuantStep::Fine, 64);
    let b = extract_aspx_sig_envelope_indices(&sig_scf, AspxQuantStep::Fine, 64);
    assert_eq!(a, b);
    let c = extract_aspx_noise_envelope_indices(&noise_scf);
    let d = extract_aspx_noise_envelope_indices(&noise_scf);
    assert_eq!(c, d);
}

/// Different input scf vectors produce materially different DPCM
/// payloads — the extractor must not collapse distinct envelopes onto
/// the same on-wire bytes.
#[test]
fn extractor_distinguishes_different_inputs() {
    let scf_a: Vec<f32> = vec![64.0, 128.0, 256.0];
    let scf_b: Vec<f32> = vec![128.0, 64.0, 32.0];
    let a = extract_aspx_sig_envelope_indices(&scf_a, AspxQuantStep::Fine, 64);
    let b = extract_aspx_sig_envelope_indices(&scf_b, AspxQuantStep::Fine, 64);
    assert_ne!(a, b);
}

/// Empty per-channel slices return empty vectors.
#[test]
fn extractor_empty_input_returns_empty() {
    let s = extract_aspx_sig_envelope_indices(&[], AspxQuantStep::Fine, 64);
    let n = extract_aspx_noise_envelope_indices(&[]);
    assert!(s.is_empty());
    assert!(n.is_empty());
}
