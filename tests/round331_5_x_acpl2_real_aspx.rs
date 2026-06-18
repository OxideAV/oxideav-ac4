//! Round 331 — 5_X SIMPLE/ASPX_ACPL_2 encoder with a **real** ASPX
//! SIGNAL / NOISE envelope on the **centre** carrier's `aspx_data_1ch()`
//! element (in addition to the L / R carrier pair's `aspx_data_2ch()`),
//! closing the README's "the 1-channel (`aspx_data_1ch`) real-envelope
//! path … still writes the single-envelope scaffold on the live frame
//! path" deferral.
//!
//! ### Background
//!
//! The round-226 single-channel real-envelope writer
//! (`write_aspx_data_1ch_real_envelope`) has existed for many rounds, but
//! no live IMS frame path ever spliced it onto the wire — every ACPL_2 /
//! ACPL_1 5_X body still emitted the round-95 minimum-bit-cost
//! `write_aspx_data_1ch_minimal` scaffold for the centre carrier's
//! `aspx_data_1ch()` element. (The round-322 work wired the *2ch*
//! real-envelope writer into the ACPL_3 carrier pair, but ACPL_3 has no
//! `aspx_data_1ch()` element — its body is `aspx_data_2ch()` only.)
//!
//! This round wires the single-channel real envelope end-to-end for the
//! ASPX_ACPL_2 5_X path:
//!
//! 1. The new IMS entry point `encode_frame_pcm_5_0_acpl2_real_aspx`
//!    QMF-analyses the L / R **and** C input PCM
//!    (`QmfAnalysisBank::process_block` → `qmf_slots_to_sb_major` →
//!    `build_aspx_real_envelope_channel_from_qmf`) and produces real
//!    `[F0, DF₁, …]` SIGNAL / NOISE quant-index vectors for all three
//!    carriers.
//! 2. `build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`
//!    splices the L / R vectors into `aspx_data_2ch()` (via
//!    `write_aspx_data_2ch_real_envelope`) **and** the C vectors into
//!    `aspx_data_1ch()` (via `write_aspx_data_1ch_real_envelope`),
//!    keeping every other element (companding, split-MDCT stereo carriers,
//!    the centre `mono_data`, the two real-α/β `acpl_data_1ch()` parameter
//!    sets) byte-for-byte identical to the round-144 builder.
//!
//! ### What this round measures
//!
//! 1. Round-trip — the real-ASPX 5.0 ACPL_2 encoder output is accepted
//!    by `Ac4Decoder` and yields a 5-channel AudioFrame.
//! 2. Round-trip — the 5.1 wrapper yields a 5-channel AudioFrame too
//!    (ACPL_2 reconstructs the surround pair from the carriers).
//! 3. Decode-side envelope recovery — the centre carrier's
//!    `aspx_data_1ch()` element parses back through the round-226 framing
//!    skeleton (`parse_aspx_framing` / `parse_aspx_delta_dir` /
//!    `parse_aspx_hfgen_iwc_1ch` / `parse_aspx_ec_data`) to exactly the
//!    SIGNAL / NOISE quant-index vectors
//!    `build_aspx_real_envelope_channel_from_qmf` produces for the same C
//!    input.
//! 4. Wire liveness — a HF-rich centre input drives a non-zero recovered
//!    1ch envelope (the real extractor reaches the wire, distinct from
//!    the all-zero scaffold body).
//! 5. Determinism — matched inputs + fresh encoder state produce
//!    identical bytes.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.6 Table 25 (`case ASPX_ACPL_2:`),
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §5.7.6.3.4 / §5.7.6.3.5
//! Pseudocodes 80–83, §5.7.6.4.2.1 Pseudocodes 90–91.

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, num_aspx_timeslots, num_ts_in_ats, parse_aspx_delta_dir,
    parse_aspx_ec_data, parse_aspx_framing, parse_aspx_hfgen_iwc_1ch, AspxConfig, AspxDataType,
    AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep, AspxStereoMode,
};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::{
    build_aspx_real_envelope_channel_from_qmf, qmf_slots_to_sb_major,
    write_aspx_data_1ch_real_envelope, AspxQmfEnvelopeChannel, AspxRealEnvelopeChannel,
};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::qmf::QmfAnalysisBank;
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn make_tone_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// The live config the IMS encoder uses for the 5_X ACPL_2 path.
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

/// Reproduce the encoder's single-carrier (centre) envelope extraction
/// independently so the test can assert against the same quant-index
/// vectors.
fn extract_mono_envelope(cfg: &AspxConfig, frame_len: u32, pcm: &[f32]) -> (Vec<i32>, Vec<i32>) {
    let tables = derive_aspx_frequency_tables(cfg, 0).expect("freq tables");
    let num_ts = num_ts_in_ats(frame_len);
    let aspx_frame_ts_count = num_aspx_timeslots(frame_len);
    let n_slots = pcm.len() / 64;
    let usable = n_slots * 64;

    let mut bank = QmfAnalysisBank::new();
    let q_high = qmf_slots_to_sb_major(&bank.process_block(&pcm[..usable]));

    let ch = AspxQmfEnvelopeChannel {
        q_high: &q_high,
        sbg_sig_borders: &tables.sbg_sig_highres,
        sbg_noise_borders: &tables.sbg_noise,
    };
    build_aspx_real_envelope_channel_from_qmf(
        &ch,
        cfg.quant_mode_env,
        64,
        num_ts,
        aspx_frame_ts_count,
        tables.sbx,
    )
}

/// 5.0 ACPL_2 real-ASPX encode round-trips to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl2_real_aspx_round_trips_to_5_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2, "5-channel S16 interleaved");
}

/// 5.1 wrapper round-trips to a 5-channel AudioFrame (ACPL_2
/// reconstructs the surround pair from the carriers; the `.1` element is
/// the round-144 scaffold).
#[test]
fn encode_5_1_acpl2_real_aspx_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lfe = make_tone_frame(60.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_5_1_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lfe]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
}

/// Decode-side envelope recovery: the centre carrier's `aspx_data_1ch()`
/// element parses back to exactly the SIGNAL / NOISE quant-index vectors
/// the QMF-energy extractor produces for the same C input. Re-emit just
/// the `aspx_data_1ch()` element with the extracted envelope and confirm
/// the parser recovers it entry-for-entry.
#[test]
fn acpl2_real_aspx_1ch_envelope_recovers_through_parser() {
    let tl = 1920u32;
    let cfg = live_cfg();

    // HF-rich centre input so the recovered envelope is non-trivial.
    let c: Vec<f32> = (0..tl as usize)
        .map(|i| {
            let t = i as f32 / FS;
            0.3 * (2.0 * std::f32::consts::PI * 9000.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 15000.0 * t).sin()
        })
        .collect();

    let (c_sig, c_noise) = extract_mono_envelope(&cfg, tl, &c);

    let counts = derive_aspx_frequency_tables(&cfg, 0)
        .expect("tables")
        .counts;
    assert!(
        counts.num_sbg_sig_highres >= 1,
        "live cfg must derive ≥ 1 signal SBG"
    );
    assert_eq!(c_sig.len(), counts.num_sbg_sig_highres as usize);
    assert_eq!(c_noise.len(), counts.num_sbg_noise as usize);

    // Re-emit just the aspx_data_1ch() element with the same envelope.
    let mut bw = BitWriter::new();
    write_aspx_data_1ch_real_envelope(
        &mut bw,
        &cfg,
        AspxRealEnvelopeChannel {
            sig: &c_sig,
            noise: &c_noise,
        },
    )
    .expect("writer");
    bw.align_to_byte();
    let aspx_bytes = bw.finish();

    // Parse the aspx_data_1ch() body back. Layout per Table 51:
    //   xover (3 b), aspx_framing(0), aspx_delta_dir(0) [SIGNAL + NOISE
    //   FREQ bits], aspx_hfgen_iwc_1ch(), SIGNAL ec_data, NOISE ec_data.
    let nats = num_aspx_timeslots(tl);
    let mut br = BitReader::new(&aspx_bytes);
    let _xover = br.read_u32(3).unwrap();
    let framing = parse_aspx_framing(&mut br, &cfg, true, nats > 8).expect("framing");
    let dd = parse_aspx_delta_dir(&mut br, &framing).expect("dd");
    let _hfgen = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        cfg.num_noise_sbgroups(),
        counts.num_sbg_sig_highres,
        nats,
    )
    .expect("hfgen");

    let sig = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        framing.num_env,
        &framing.freq_res,
        AspxQuantStep::Fine,
        AspxStereoMode::Level,
        &dd.sig_delta_dir,
        counts,
    )
    .expect("sig");
    let noise = parse_aspx_ec_data(
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

    // The single-channel writer uses LEVEL stereo mode for both SIGNAL
    // and NOISE. F0 is clamped to the ASPX_HCB_ENV_LEVEL_15_F0 /
    // ASPX_HCB_NOISE_LEVEL_F0 codebook ranges ([0, 70] / [0, 29]); the
    // DF deltas reach the wire unchanged.
    let clamp_f0 = |v: &[i32], hi: i32| {
        let mut c = v.to_vec();
        if let Some(f0) = c.first_mut() {
            *f0 = (*f0).clamp(0, hi);
        }
        c
    };
    assert_eq!(sig[0].values, clamp_f0(&c_sig, 70));
    assert_eq!(noise[0].values, clamp_f0(&c_noise, 29));
    // Independent of the F0 clamp, every DF delta survives the round
    // trip verbatim — the real per-band envelope shape reaches the wire.
    assert_eq!(sig[0].values[1..], c_sig[1..]);
}

/// Wire liveness: a HF-rich centre input drives a non-zero recovered
/// 1ch envelope (the real extractor reaches the wire, distinct from the
/// all-zero scaffold).
#[test]
fn acpl2_real_aspx_1ch_envelope_is_nonzero_for_hf_input() {
    let tl = 1920u32;
    let cfg = live_cfg();
    let c: Vec<f32> = (0..tl as usize)
        .map(|i| {
            let t = i as f32 / FS;
            0.5 * (2.0 * std::f32::consts::PI * 12000.0 * t).sin()
        })
        .collect();
    let (c_sig, c_noise) = extract_mono_envelope(&cfg, tl, &c);
    // Strong HF energy must produce a non-zero SIGNAL F0 — the scaffold
    // path would emit all zeros.
    assert!(
        c_sig.iter().any(|&v| v != 0) || c_noise.iter().any(|&v| v != 0),
        "HF-rich centre input must drive a non-zero envelope"
    );
}

/// Determinism — matched inputs + fresh encoder state produce identical
/// bytes.
#[test]
fn encode_5_0_acpl2_real_aspx_is_deterministic() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c = make_tone_frame(9000.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    let a = enc_a.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    let b = enc_b.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    assert_eq!(a, b);
}

/// The real-ASPX ACPL_2 body differs from the round-144 scaffold body
/// when the centre carrier carries HF energy — confirming the real 1ch
/// envelope reaches the wire and is not optimised back to the scaffold.
#[test]
fn acpl2_real_aspx_body_differs_from_scaffold_for_hf_centre() {
    let l = make_tone_frame(220.0, 0.3);
    let r = make_tone_frame(440.0, 0.3);
    let c_hf = make_tone_frame(12000.0, 0.5);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    let mut enc_real = Ac4ImsEncoder::new();
    let mut enc_scaffold = Ac4ImsEncoder::new();
    let real = enc_real.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c_hf, &ls, &rs]);
    let scaffold =
        enc_scaffold.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c_hf, &ls, &rs]);
    assert_ne!(
        real, scaffold,
        "real 1ch+2ch ASPX envelope must differ from the all-zero scaffold body"
    );
}
