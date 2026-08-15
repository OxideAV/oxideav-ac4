//! Round 419 — encoder ICE synthesis parity, part 1: the ASPX_SCPL
//! immersive-channel-element encode route (TS 103 190-2 §6.2.4.1,
//! `immersive_codec_mode = ASPX_SCPL`) from PCM.
//!
//! The encoder derives the eleven SMP tracks A..K as the exact inverse
//! of the decoder's §5.3.3.1 Table 23 S-CPL full-decoding matrix
//! composed with the §4.8.3.11.3 Table 10 output gains, forward-MDCTs
//! them on persistent TDAC states, and emits real SIGNAL / NOISE
//! A-SPX envelopes (+ `aspx_tna_mode` / `aspx_add_harmonic`) per
//! Table 8 channel group, extracted from the decoupled channels the
//! decoder extends.
//!
//! Measured here:
//! 1. Settled waveform round-trip on frame-periodic low-band content —
//!    circularly aligned per-channel relative RMS error against the
//!    input frame.
//! 2. Real-envelope HF synthesis — a channel with HF content decodes
//!    with matching HF band energy (within 3 dB), while HF-silent
//!    channels stay HF-silent (the floored-noise scaffold could not do
//!    this: it either floored or full-scaled every channel alike).
//! 3. 7.1.4: the LFE arm decodes on the leading output slot.
//! 4. Parse-exactness — the emitted bitstream re-reads to exactly the
//!    envelope quant rows the encoder's extractor produces.
//! 5. Determinism — fresh encoder states produce identical bytes.

use oxideav_ac4::aspx::ASPX_QMF_PCM_SCALE;
use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::qmf_slots_to_sb_major;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::ice::{IceAspxElement, IceCodecMode};
use oxideav_ac4::qmf::QmfAnalysisBank;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

/// Frame-periodic tone: `cycles` full cycles per 1920-sample frame
/// (f = cycles · 25 Hz), so consecutive identical frames form a
/// continuous periodic signal and any whole-sample pipeline delay is a
/// circular rotation of the frame.
fn periodic_tone(cycles: u32, amp: f32, phase: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            amp * (2.0 * std::f32::consts::PI * cycles as f32 * t + phase).sin()
        })
        .collect()
}

fn decode_frame(dec: &mut Ac4Decoder, bytes: Vec<u8>, channels: usize) -> Vec<Vec<f32>> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let Frame::Audio(af) = frame else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, N as u32, "frame length");
    let buf = &af.data[0];
    assert_eq!(buf.len(), N * channels * 2, "interleaved buffer size");
    let mut out = vec![Vec::with_capacity(N); channels];
    for i in 0..N {
        for (c, ch) in out.iter_mut().enumerate() {
            let off = (i * channels + c) * 2;
            let s = i16::from_le_bytes([buf[off], buf[off + 1]]);
            ch.push(s as f32 / 32768.0);
        }
    }
    out
}

/// Best circular lag (maximising cross-correlation) of `dec` against
/// `reference` — the settled decode of a frame-periodic input is a
/// circular rotation of the input frame by the whole-sample pipeline
/// delay.
fn best_circular_lag(reference: &[f32], dec: &[f32]) -> usize {
    let n = reference.len();
    let mut best = (0usize, f64::MIN);
    for lag in 0..n {
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += reference[i] as f64 * dec[(i + lag) % n] as f64;
        }
        if acc > best.1 {
            best = (lag, acc);
        }
    }
    best.0
}

/// Relative RMS error of `dec` (circularly rotated by `lag`) against
/// `reference`: `sqrt(Σ(dec−ref)² / Σref²)`.
fn rel_rms_err(reference: &[f32], dec: &[f32], lag: usize) -> f64 {
    let n = reference.len();
    let (mut err, mut sig) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = reference[i] as f64;
        let d = dec[(i + lag) % n] as f64;
        err += (d - r) * (d - r);
        sig += r * r;
    }
    (err / sig.max(1e-30)).sqrt()
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// The live ICE A-SPX config (mirrors the encoder's internal choice).
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

/// HF-band energy of a PCM frame: energy above the A-SPX crossover
/// frequency (sbx · 375 Hz), measured by projection onto DFT bins.
fn hf_energy(pcm: &[f32], lo_hz: f32) -> f64 {
    let n = pcm.len();
    let lo_bin = (lo_hz * n as f32 / FS).ceil() as usize;
    let mut acc = 0.0f64;
    for k in lo_bin..(n / 2) {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        for (i, &s) in pcm.iter().enumerate() {
            let ph = w * i as f64;
            re += s as f64 * ph.cos();
            im -= s as f64 * ph.sin();
        }
        acc += (re * re + im * im) * 2.0 / (n as f64 * n as f64);
    }
    acc
}

/// Eleven distinct frame-periodic low-band tones (cycles → 25·c Hz),
/// all below the live config's crossover.
fn low_band_input() -> Vec<Vec<f32>> {
    let cycles = [12u32, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]; // 300..1300 Hz
    cycles
        .iter()
        .enumerate()
        .map(|(ch, &c)| periodic_tone(c, 0.35, 0.3 * ch as f32))
        .collect()
}

#[test]
fn ice_aspx_scpl_7_0_4_settled_low_band_round_trip() {
    let input = low_band_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    // Per-channel circular alignment: a single tone only defines the
    // pipeline delay modulo its own period, so each channel aligns on
    // its own best lag (all consistent with one true delay).
    let mut worst = 0.0f64;
    for ch in 0..11 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-419 ASPX_SCPL settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.10,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-419 ASPX_SCPL worst settled relative RMS err: {worst:.4}");
}

#[test]
fn ice_aspx_scpl_real_envelopes_synthesise_hf() {
    // L carries a strong HF component; R stays HF-silent. The real
    // per-channel envelopes must regenerate L's HF at the right level
    // while leaving R's high band quiet — the floored/minimal payload
    // could not distinguish the two.
    let tables = derive_aspx_frequency_tables(&live_cfg(), 0).expect("freq tables");
    let sbx_hz = tables.sbx as f32 * 375.0;
    let hf_cycles = ((sbx_hz + 800.0) / 25.0).ceil() as u32;
    let mut input = low_band_input();
    let hf = periodic_tone(hf_cycles, 0.3, 0.0);
    for (a, b) in input[0].iter_mut().zip(hf.iter()) {
        *a += *b;
    }
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    let e_in_l = hf_energy(&input[0], sbx_hz);
    let e_out_l = hf_energy(&last[0], sbx_hz);
    let e_out_r = hf_energy(&last[1], sbx_hz);
    let ratio = e_out_l / e_in_l.max(1e-30);
    eprintln!(
        "ROUND-419 ASPX_SCPL HF energy: in L {e_in_l:.6}, out L {e_out_l:.6} (ratio {ratio:.3}), out R {e_out_r:.6}"
    );
    assert!(
        (0.5..=2.0).contains(&ratio),
        "L HF band energy within 3 dB of the input ({ratio:.3})"
    );
    assert!(
        e_out_r < e_out_l / 20.0,
        "HF-silent R stays at least 13 dB below L's regenerated band ({e_out_r} vs {e_out_l})"
    );
}

#[test]
fn ice_aspx_scpl_7_1_4_lfe_round_trip() {
    let named = low_band_input();
    let lfe = periodic_tone(3, 0.4, 0.0); // 75 Hz
    let mut all: Vec<&[f32]> = named.iter().map(|v| v.as_slice()).collect();
    all.push(&lfe);
    let refs: [&[f32]; 12] = std::array::from_fn(|i| all[i]);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_1_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 12);
    }
    // Output order: [LFE, L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    let lag = best_circular_lag(&lfe, &last[0]);
    let e_lfe = rel_rms_err(&lfe, &last[0], lag);
    eprintln!("ROUND-419 ASPX_SCPL 7.1.4 LFE settled relative RMS err: {e_lfe:.4}");
    assert!(e_lfe < 0.05, "LFE settled error too high: {e_lfe:.4}");
    for ch in 0..11 {
        let lag = best_circular_lag(&named[ch], &last[ch + 1]);
        let e = rel_rms_err(&named[ch], &last[ch + 1], lag);
        assert!(
            e < 0.10,
            "7.1.4 named channel {ch} settled relative RMS error too high: {e:.4}"
        );
    }
}

/// Analyse one channel's PCM exactly like the encoder does on its
/// first frame (fresh streaming bank, integer-PCM scale), so the
/// parse-exactness assertions below are anchored to an independent
/// derivation of the same wire rows via the promoted row builders.
fn analyse(pcm: &[f32]) -> Vec<Vec<(f32, f32)>> {
    let n_slots = pcm.len() / 64;
    let usable = n_slots * 64;
    let scaled: Vec<f32> = pcm[..usable]
        .iter()
        .map(|&v| v * ASPX_QMF_PCM_SCALE)
        .collect();
    let mut bank = QmfAnalysisBank::new();
    qmf_slots_to_sb_major(&bank.process_block(&scaled))
}

#[test]
fn ice_aspx_scpl_bitstream_re_reads_to_extracted_rows() {
    let input = low_band_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let bytes = enc.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
    let _ = decode_frame(&mut dec, bytes, 11);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let ice = sub.tools.ice.as_deref().expect("ice element parsed");
    assert_eq!(ice.mode, IceCodecMode::AspxScpl);
    assert_eq!(ice.aspx_elements.len(), 6, "ASPX_SCPL payload roster");
    // The encoder's config (preflat decided per frame — recompute the
    // parse side's view from the sticky slot).
    let cfg = sub.tools.aspx_config.expect("sticky aspx_config");
    // Payload 0 extends the decoupled (Ls, Lb) / √2 — the V1.3.1
    // Table 8 roster leads with the surround pairs. First frame: both
    // the encoder's and this reproduction's banks are fresh, so the
    // rows match exactly.
    let isq2 = std::f32::consts::FRAC_1_SQRT_2;
    let dec_ls: Vec<f32> = input[3].iter().map(|&v| v * isq2).collect();
    let dec_lb: Vec<f32> = input[5].iter().map(|&v| v * isq2).collect();
    let rows0 = Ac4ImsEncoder::ice_2ch_rows_from_matrices(
        &cfg,
        N as u32,
        &analyse(&dec_ls),
        &analyse(&dec_lb),
    );
    // §5.7.6.3.5 joint coding: the 2ch writer converts each pair's
    // (L, R) LEVEL rows to the (sum, pan) wire pair (Pseudocode 84
    // inverse), so the parsed primary carries the sum rows.
    use oxideav_ac4::encoder_acpl3::{
        balance_encode_noise_rows, balance_encode_sig_rows, freq_dpcm_encode_qscf,
        qscf_row_from_freq_dpcm_extended,
    };
    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        oxideav_ac4::aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };
    let expect_sum = |ch0_sig: &[i32], ch1_sig: &[i32], ch0_noise: &[i32], ch1_noise: &[i32]| {
        let n_sig = ch0_sig.len().max(ch1_sig.len());
        let n_noise = ch0_noise.len().max(ch1_noise.len());
        let (sum_sig, _) = balance_encode_sig_rows(
            &qscf_row_from_freq_dpcm_extended(ch0_sig, n_sig),
            &qscf_row_from_freq_dpcm_extended(ch1_sig, n_sig),
            qmode_sig,
        );
        let (sum_noise, _) = balance_encode_noise_rows(
            &qscf_row_from_freq_dpcm_extended(ch0_noise, n_noise),
            &qscf_row_from_freq_dpcm_extended(ch1_noise, n_noise),
        );
        (
            freq_dpcm_encode_qscf(&sum_sig),
            freq_dpcm_encode_qscf(&sum_noise),
        )
    };
    let IceAspxElement::TwoCh(Some(t0)) = &ice.aspx_elements[0] else {
        panic!("payload 0 must be a parsed 2ch element");
    };
    assert!(!t0.primary.data_sig.is_empty(), "payload 0 sig envelope");
    let (exp_sig0, exp_noise0) = expect_sum(
        &rows0.ch0.sig,
        &rows0.ch1.sig,
        &rows0.ch0.noise,
        &rows0.ch1.noise,
    );
    assert_eq!(
        t0.primary.data_sig[0].values, exp_sig0,
        "payload 0 primary SIGNAL row re-reads to the converted (sum) extractor output"
    );
    assert_eq!(
        t0.primary.data_noise[0].values, exp_noise0,
        "payload 0 primary NOISE row re-reads to the converted (sum) extractor output"
    );
    // The 1ch payload (roster position 2) carries the decoupled C.
    let dec_c: Vec<f32> = input[2].iter().map(|&v| v * 0.5).collect();
    let rows_c = Ac4ImsEncoder::ice_1ch_rows_from_matrix(&cfg, N as u32, &analyse(&dec_c));
    let IceAspxElement::OneCh(Some(t2)) = &ice.aspx_elements[2] else {
        panic!("payload 2 must be a parsed 1ch element");
    };
    assert_eq!(
        t2.primary.data_sig[0].values, rows_c.ch.sig,
        "1ch payload SIGNAL row re-reads to the extractor output"
    );
    assert_eq!(
        t2.primary.data_noise[0].values, rows_c.ch.noise,
        "1ch payload NOISE row re-reads to the extractor output"
    );
    // The front payload (roster 3) extends the decoupled (L, R) =
    // inputs / 2 — the same signals as SMP tracks A / B.
    let dec_l: Vec<f32> = input[0].iter().map(|&v| v * 0.5).collect();
    let dec_r: Vec<f32> = input[1].iter().map(|&v| v * 0.5).collect();
    let rows1 = Ac4ImsEncoder::ice_2ch_rows_from_matrices(
        &cfg,
        N as u32,
        &analyse(&dec_l),
        &analyse(&dec_r),
    );
    let IceAspxElement::TwoCh(Some(t1)) = &ice.aspx_elements[3] else {
        panic!("payload 3 must be a parsed 2ch element");
    };
    let (exp_sig1, exp_noise1) = expect_sum(
        &rows1.ch0.sig,
        &rows1.ch1.sig,
        &rows1.ch0.noise,
        &rows1.ch1.noise,
    );
    assert_eq!(t1.primary.data_sig[0].values, exp_sig1);
    assert_eq!(t1.primary.data_noise[0].values, exp_noise1);
}

#[test]
fn ice_aspx_scpl_encode_is_deterministic() {
    let input = low_band_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
        let b = enc_b.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
}

#[test]
fn ice_aspx_scpl_energy_reaches_every_output_slot() {
    // Distinct tones per input channel must land on their own output
    // slots (the matrix inverse and the payload/track wiring line up).
    let input = low_band_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    for (ch, out) in last.iter().enumerate() {
        let e_out = energy(out);
        let e_in = energy(&input[ch]);
        let ratio = e_out / e_in.max(1e-30);
        assert!(
            (0.5..=2.0).contains(&ratio),
            "channel {ch} settled energy within 3 dB of input (ratio {ratio:.3})"
        );
    }
}
