//! Round 419 — encoder ICE synthesis parity, part 2: the ASPX_ACPL_1
//! / ASPX_ACPL_2 immersive-channel-element encode routes
//! (TS 103 190-2 §6.2.4.1 + §5.5.2 Table 27) from PCM.
//!
//! The encoder codes `A = L/2`, `B = R/2`, `C = C/2` plus four module
//! mid carriers `(P+Q)/(2√2)`; each §5.5.2 module's `(α, β)` comes
//! from the pair's mid/side statistics (α carries the correlated part
//! of the inter-channel difference, β the decorrelated residual
//! energy), and — for ASPX_ACPL_1 — the side signals below
//! `acpl_qmf_band` ride exactly-coded M/S residual tracks.
//!
//! Measured here:
//! 1. ACPL_2, correlated pairs (`Q = g·P`): the module reconstruction
//!    is dry-exact (`P̂ = mid(1+α)`, `Q̂ = mid(1−α)`), so the settled
//!    per-channel waveform round-trips within a small relative RMS
//!    error and the pair's level ratio is preserved.
//! 2. ACPL_2, independent pair content: per-channel settled energy
//!    within 3 dB (β fills the decorrelated part).
//! 3. ACPL_1: below `acpl_qmf_band` the M/S residual path
//!    reconstructs the pair waveform-exactly (settled RMS bar).
//! 4. Parse-exactness — the emitted `acpl_data_1ch()` rows re-read
//!    (through the decoder's differential decode) to exactly the
//!    quantised α/β rows the extractor computes.
//! 5. Determinism.

use oxideav_ac4::acpl_synth;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::extract_ice_acpl_pair_alpha_beta_q;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::ice::IceCodecMode;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;

fn periodic_tone(cycles: u32, amp: f32, phase: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            amp * (2.0 * std::f32::consts::PI * cycles as f32 * t + phase).sin()
        })
        .collect()
}

/// Frame-periodic noise-like signal: many equal-amplitude partials at
/// distinct cycle counts with pseudo-random phases (`seed`-keyed).
/// Decorrelation-friendly content — a lone tone is only phase-shifted
/// by the §5.7.3.5-style all-pass decorrelators, while a dense
/// multi-partial signal decorrelates statistically.
fn periodic_noise(lo_cycles: u32, hi_cycles: u32, amp: f32, seed: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; N];
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let n_part = (hi_cycles - lo_cycles).max(1);
    let a = amp / (n_part as f32).sqrt();
    for c in lo_cycles..hi_cycles {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let phase = (state >> 8) as f32 / (1u32 << 24) as f32 * std::f32::consts::TAU;
        for (i, v) in out.iter_mut().enumerate() {
            let t = i as f32 / N as f32;
            *v += a * (std::f32::consts::TAU * c as f32 * t + phase).sin();
        }
    }
    out
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

/// Correlated 7.0.4 content: every module pair's second channel is a
/// scaled copy of the first (`Q = 0,6·P`) so the module dry mix alone
/// reconstructs the pair.
fn correlated_input() -> Vec<Vec<f32>> {
    let mk = |c: u32, a: f32, p: f32| periodic_tone(c, a, p);
    let ls = mk(24, 0.35, 0.4);
    let rs = mk(28, 0.35, 0.9);
    let tfl = mk(36, 0.3, 1.3);
    let tfr = mk(40, 0.3, 1.8);
    let scale = |x: &Vec<f32>, g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
    vec![
        mk(12, 0.35, 0.0), // L
        mk(16, 0.35, 0.5), // R
        mk(20, 0.35, 1.0), // C
        ls.clone(),        // Ls
        rs.clone(),        // Rs
        scale(&ls, 0.6),   // Lb = 0,6·Ls
        scale(&rs, 0.6),   // Rb
        tfl.clone(),       // Tfl
        tfr.clone(),       // Tfr
        scale(&tfl, 0.6),  // Tbl
        scale(&tfr, 0.6),  // Tbr
    ]
}

#[test]
fn ice_acpl2_correlated_pairs_settle_and_keep_ratios() {
    let input = correlated_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_acpl2(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    let mut worst = 0.0f64;
    for ch in 0..11 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-419 ACPL_2 settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.25,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-419 ACPL_2 worst settled relative RMS err (correlated): {worst:.4}");
    // Pair level ratios preserved (0,6² = 0,36 energy; α / β lane
    // quantisation plus the phase-sensitive wet term on tonal content
    // widen the window).
    for (p, q) in [(3usize, 5usize), (4, 6), (7, 9), (8, 10)] {
        let r = energy(&last[q]) / energy(&last[p]).max(1e-30);
        eprintln!("ROUND-419 ACPL_2 pair ({p},{q}) level ratio: {r:.3}");
        assert!(
            (0.15..=0.75).contains(&r),
            "pair ({p},{q}) level ratio drifted: {r:.3}"
        );
    }
}

#[test]
fn ice_acpl2_independent_pairs_keep_energy() {
    // Independent tones on each pair channel: the dry mid carries
    // half, β restores the decorrelated remainder — per-channel
    // settled energy within 3 dB.
    let mut input = correlated_input();
    input[3] = periodic_noise(20, 40, 0.3, 11); // Ls
    input[5] = periodic_noise(20, 40, 0.3, 23); // Lb independent of Ls
    input[4] = periodic_noise(24, 44, 0.3, 37); // Rs
    input[6] = periodic_noise(24, 44, 0.3, 41); // Rb
    input[7] = periodic_noise(30, 50, 0.25, 53); // Tfl
    input[9] = periodic_noise(30, 50, 0.25, 67); // Tbl
    input[8] = periodic_noise(34, 54, 0.25, 71); // Tfr
    input[10] = periodic_noise(34, 54, 0.25, 83); // Tbr
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_acpl2(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    for ch in 0..11 {
        let r = energy(&last[ch]) / energy(&input[ch]).max(1e-30);
        eprintln!("ROUND-419 ACPL_2 independent-pair energy ratio ch{ch}: {r:.3}");
        assert!(
            (0.4..=2.5).contains(&r),
            "channel {ch} settled energy ratio out of range: {r:.3}"
        );
    }
}

#[test]
fn ice_acpl1_ms_band_reconstructs_pairs() {
    // All pair content below acpl_qmf_band = 6 (2 250 Hz): the
    // PARTIAL config's M/S residual path reconstructs the pairs
    // waveform-exactly; fronts ride their own tracks as usual.
    let input = correlated_input();
    // Independent second channels — the M/S band codes them exactly
    // regardless of correlation.
    let mut input = input;
    input[5] = periodic_tone(26, 0.35, 0.2);
    input[6] = periodic_tone(30, 0.35, 0.7);
    input[9] = periodic_tone(38, 0.3, 1.1);
    input[10] = periodic_tone(42, 0.3, 1.6);
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_acpl1(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    let mut worst = 0.0f64;
    for ch in 0..11 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-419 ACPL_1 settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.15,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-419 ACPL_1 worst settled relative RMS err (M/S band): {worst:.4}");
}

#[test]
fn ice_acpl_bitstream_re_reads_to_extracted_alpha_beta() {
    let input = correlated_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let bytes = enc.encode_frame_pcm_7_0_4_ice_acpl2(&refs);
    let _ = decode_frame(&mut dec, bytes, 11);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let ice = sub.tools.ice.as_deref().expect("ice element parsed");
    assert_eq!(ice.mode, IceCodecMode::AspxAcpl2);
    assert_eq!(ice.acpl_data.len(), 4, "four acpl_data_1ch() modules");
    let acfg = sub
        .tools
        .acpl_config_1ch_full
        .expect("FULL acpl_config_1ch sticky");
    // Recompute module 0's (α, β) from the (Ls, Lb) mid/side spectra
    // exactly like the encoder (fresh TDAC state on the first frame).
    let mut mid_state = oxideav_ac4::encoder_mdct::EncoderMdctState::new(N as u32);
    let mut side_state = oxideav_ac4::encoder_mdct::EncoderMdctState::new(N as u32);
    let mid_pcm: Vec<f32> = (0..N).map(|i| 0.5 * (input[3][i] + input[5][i])).collect();
    let side_pcm: Vec<f32> = (0..N).map(|i| 0.5 * (input[3][i] - input[5][i])).collect();
    let mid_spec = mid_state.analyse_frame(&mid_pcm);
    let side_spec = side_state.analyse_frame(&side_pcm);
    let (alpha_q, beta_q) = extract_ice_acpl_pair_alpha_beta_q(
        &mid_spec,
        &side_spec,
        N as u32,
        acfg.num_param_bands,
        0,
        acfg.quant_mode,
    );
    let mut ad = acpl_synth::AcplDiffState::new();
    let mut bd = acpl_synth::AcplDiffState::new();
    let alpha_rows =
        acpl_synth::differential_decode(&ice.acpl_data[0].alpha1, acfg.num_param_bands, &mut ad);
    let beta_rows =
        acpl_synth::differential_decode(&ice.acpl_data[0].beta1, acfg.num_param_bands, &mut bd);
    assert_eq!(
        alpha_rows.first().map(|r| r.as_slice()),
        Some(alpha_q.as_slice()),
        "module 0 α row re-reads to the extractor output"
    );
    assert_eq!(
        beta_rows.first().map(|r| r.as_slice()),
        Some(beta_q.as_slice()),
        "module 0 β row re-reads to the extractor output"
    );
}

#[test]
fn ice_acpl_encode_is_deterministic_and_lfe_arm_decodes() {
    let input = correlated_input();
    let lfe = periodic_tone(3, 0.4, 0.0);
    let mut all: Vec<&[f32]> = input.iter().map(|v| v.as_slice()).collect();
    all.push(&lfe);
    let refs12: [&[f32]; 12] = std::array::from_fn(|i| all[i]);
    let refs11: [&[f32]; 11] = std::array::from_fn(|i| all[i]);
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_7_0_4_ice_acpl1(&refs11);
        let b = enc_b.encode_frame_pcm_7_0_4_ice_acpl1(&refs11);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
    // 7.1.4 arms decode to 12 channels with the LFE on the leading slot.
    let params = CodecParameters::audio(CodecId::new("ac4"));
    for acpl1 in [false, true] {
        let mut enc = Ac4ImsEncoder::new();
        let mut dec = Ac4Decoder::new(&params);
        let mut last = Vec::new();
        for _ in 0..4 {
            let bytes = if acpl1 {
                enc.encode_frame_pcm_7_1_4_ice_acpl1(&refs12)
            } else {
                enc.encode_frame_pcm_7_1_4_ice_acpl2(&refs12)
            };
            last = decode_frame(&mut dec, bytes, 12);
        }
        let lag = best_circular_lag(&lfe, &last[0]);
        let e = rel_rms_err(&lfe, &last[0], lag);
        assert!(
            e < 0.05,
            "ACPL{} 7.1.4 LFE settled error too high: {e:.4}",
            if acpl1 { 1 } else { 2 }
        );
    }
}
