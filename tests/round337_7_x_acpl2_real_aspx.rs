//! Round 337 — 7_X SIMPLE/ASPX_ACPL_2 encoder with a **real** ASPX
//! SIGNAL / NOISE envelope on all three ASPX carriers: the L/R front pair
//! and the Ls/Rs surround pair (both `aspx_data_2ch()`) plus the centre
//! carrier (`aspx_data_1ch()`).
//!
//! ### Background
//!
//! The round-331 work wired real single- and two-channel ASPX envelopes
//! into the live **5_X** ASPX_ACPL_2 frame path. The 7_X ACPL_2 body
//! carries one extra `aspx_data_2ch()` element (for the Ls/Rs surround
//! pair) on top of the 5_X shape — the decoder's
//! `parse_7x_audio_data_outer` trailer walks
//! `aspx_data_2ch + aspx_data_2ch + aspx_data_1ch`. Until this round the
//! 7_X live frame path (`encode_frame_pcm_7_0_acpl2_real_alpha_beta`,
//! round 202) carried real per-band α + β but still emitted the round-107
//! minimum-bit-cost ASPX envelope scaffolds on all three carriers.
//!
//! This round wires the real envelopes end-to-end for the 7_X ASPX_ACPL_2
//! path via the new IMS entry points
//! `encode_frame_pcm_7_0_acpl2_real_aspx` / `_7_1_` and the builder
//! `build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`.
//!
//! ### What this round measures
//!
//! 1. Round-trip — the real-ASPX 7.0 ACPL_2 encoder output is accepted by
//!    `Ac4Decoder` and yields a 7-channel AudioFrame; the decoder resolves
//!    the ASPX_ACPL_2 7_X mode + the A-CPL pair.
//! 2. Round-trip — the 7.1 wrapper yields an 8-channel AudioFrame.
//! 3. Decode-side envelope recovery — the centre carrier's
//!    `aspx_data_1ch()` element parses back to exactly the SIGNAL / NOISE
//!    quant-index vectors the QMF-energy extractor produces for the same C
//!    input.
//! 4. Wire liveness — the real-ASPX body differs from the round-202
//!    scaffold body when the carriers carry HF energy.
//! 5. Determinism — matched inputs + fresh encoder state produce identical
//!    bytes.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
//! §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
//! (`aspx_data_2ch()`), §5.7.6.3.4 / §5.7.6.3.5 Pseudocodes 80–83,
//! §5.7.6.4.2.1 Pseudocodes 90–91.

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

fn make_tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// The live config the IMS encoder uses for the 7_X ACPL_2 path.
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

/// 7.0 ACPL_2 real-ASPX encode round-trips to a 7-channel AudioFrame and
/// resolves the ASPX_ACPL_2 7_X mode + the A-CPL pair.
#[test]
fn encode_7_0_acpl2_real_aspx_round_trips_to_7_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.2);
    let rs = make_tone(1100.0, 0.2);
    let lb = make_tone(1320.0, 0.2);
    let rb = make_tone(1540.0, 0.2);
    let frame_bytes = enc.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");

    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::AspxAcpl2)
    );
    assert!(sub.tools.acpl_data_1ch_pair[0].is_some());
    assert!(sub.tools.acpl_data_1ch_pair[1].is_some());
}

/// 7.1 wrapper round-trips to an 8-channel AudioFrame (the `.1` element is
/// the leading LFE `mono_data(1)`).
#[test]
fn encode_7_1_acpl2_real_aspx_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.2);
    let rs = make_tone(1100.0, 0.2);
    let lb = make_tone(1320.0, 0.2);
    let rb = make_tone(1540.0, 0.2);
    let lfe = make_tone(60.0, 0.2);
    let frame_bytes =
        enc.encode_frame_pcm_7_1_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb, &lfe]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 8 * 2, "8-channel S16 interleaved");
}

/// Decode-side envelope recovery: the centre carrier's `aspx_data_1ch()`
/// element parses back to exactly the SIGNAL / NOISE quant-index vectors
/// the QMF-energy extractor produces for the same C input.
#[test]
fn acpl2_real_aspx_7x_1ch_envelope_recovers_through_parser() {
    let tl = 1920u32;
    let cfg = live_cfg();

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
    assert!(counts.num_sbg_sig_highres >= 1);
    assert_eq!(c_sig.len(), counts.num_sbg_sig_highres as usize);
    assert_eq!(c_noise.len(), counts.num_sbg_noise as usize);

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

    let clamp_f0 = |v: &[i32], hi: i32| {
        let mut c = v.to_vec();
        if let Some(f0) = c.first_mut() {
            *f0 = (*f0).clamp(0, hi);
        }
        c
    };
    assert_eq!(sig[0].values, clamp_f0(&c_sig, 70));
    assert_eq!(noise[0].values, clamp_f0(&c_noise, 29));
    assert_eq!(sig[0].values[1..], c_sig[1..]);
}

/// The real-ASPX 7_X ACPL_2 body differs from the round-202 scaffold body
/// when the carriers carry HF energy — confirming the real envelopes reach
/// the wire and are not optimised back to the all-zero scaffold.
#[test]
fn acpl2_real_aspx_7x_body_differs_from_scaffold_for_hf_input() {
    let l = make_tone(9000.0, 0.4);
    let r = make_tone(11000.0, 0.4);
    let c = make_tone(12000.0, 0.5);
    let ls = make_tone(8500.0, 0.3);
    let rs = make_tone(10500.0, 0.3);
    let lb = make_tone(1320.0, 0.2);
    let rb = make_tone(1540.0, 0.2);

    let mut enc_real = Ac4ImsEncoder::new();
    let mut enc_scaffold = Ac4ImsEncoder::new();
    let real = enc_real.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let scaffold =
        enc_scaffold.encode_frame_pcm_7_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    // Substream bodies are tightly sized (audio_size = exact body
    // length), so the two frames need not share a length; the byte-stream
    // inequality below is the real check.
    assert_ne!(
        real, scaffold,
        "real L/R + Ls/Rs + centre ASPX envelopes must differ from the all-zero scaffold body"
    );
}

/// Determinism — matched inputs + fresh encoder state produce identical
/// bytes.
#[test]
fn encode_7_0_acpl2_real_aspx_is_deterministic() {
    let l = make_tone(220.0, 0.3);
    let r = make_tone(440.0, 0.3);
    let c = make_tone(9000.0, 0.3);
    let ls = make_tone(880.0, 0.2);
    let rs = make_tone(1100.0, 0.2);
    let lb = make_tone(1320.0, 0.2);
    let rb = make_tone(1540.0, 0.2);

    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    let a = enc_a.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let b = enc_b.encode_frame_pcm_7_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert_eq!(a, b);
}
