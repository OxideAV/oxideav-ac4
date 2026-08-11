//! Round 440 — b_5fronts ICE encode, part 2: the 9.0.4 / 9.1.4
//! ASPX_ACPL_1 / ASPX_ACPL_2 immersive-channel-element encode routes
//! (TS 103 190-2 §6.2.4.1, `b_5fronts = 1`).
//!
//! The six §5.5.2 Table 27 modules cover the four surround / top pairs
//! `(Ls, Lb)` / `(Rs, Rb)` / `(Tfl, Tbl)` / `(Tfr, Tbr)` (mid carriers
//! `(P+Q)/(2√2)` on tracks D..G, √2 immersive output scale) plus the
//! two **front modules** `(L, Lscr)` / `(R, Rscr)`, whose mid carriers
//! ride the A / B track positions directly (plain `(P+Q)/2`, no output
//! scale). ASPX_ACPL_1 additionally codes each pair's side signal as an
//! M/S residual below `acpl_qmf_band`: surround sides on F / G, top
//! mids + sides on the S-CPL H / I + J / K, and the front sides on the
//! third b_5fronts S-CPL pair L″ / M″.
//!
//! Measured here:
//! 1. ASPX_ACPL_2 on correlated pairs — settled per-channel RMS with
//!    pair level ratios preserved, all 13 channels.
//! 2. ASPX_ACPL_2 on independent (noise-like) pair content —
//!    per-channel settled energy within the β decorrelator-fill window.
//! 3. ASPX_ACPL_1 with all pair content below the split — the M/S
//!    residual path reconstructs every pair (including the fronts)
//!    waveform-accurately.
//! 4. Parse shape + α/β re-read: 6 `acpl_data_1ch()` modules, and the
//!    front module's rows re-read to the extractor output.
//! 5. Determinism + the 9.1.4 LFE arms.

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

/// Frame-periodic noise-like signal (seed-keyed pseudo-random phases).
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

/// Correlated 9.0.4 content: every module pair's second channel is a
/// scaled copy of the first (`Q = 0,6·P`), including the front pairs
/// `(L, Lscr)` / `(R, Rscr)`.
/// Order: `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
///
/// The front-pair tones sit near QMF subband centres (sb width
/// 375 Hz): unlike the 7.X layout, the b_5fronts front channels run
/// **parametric** (per-parameter-band α gains), and a tone at a QMF
/// band edge splits across two subbands whose differing per-band gains
/// break the filterbank's alias cancellation — a real property of
/// per-subband parametric coding, not an encoder bug.
fn correlated_input() -> Vec<Vec<f32>> {
    let mk = |c: u32, a: f32, p: f32| periodic_tone(c, a, p);
    let l = mk(9, 0.35, 0.0); // 225 Hz — sb0 centre
    let r = mk(22, 0.35, 0.5); // 550 Hz — sb1 centre
    let ls = mk(24, 0.35, 0.4);
    let rs = mk(28, 0.35, 0.9);
    let tfl = mk(36, 0.3, 1.3);
    let tfr = mk(40, 0.3, 1.8);
    let scale = |x: &Vec<f32>, g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
    vec![
        l.clone(),         // L
        r.clone(),         // R
        mk(20, 0.35, 1.0), // C
        scale(&l, 0.6),    // Lscr = 0,6·L
        scale(&r, 0.6),    // Rscr
        ls.clone(),        // Ls
        rs.clone(),        // Rs
        scale(&ls, 0.6),   // Lb
        scale(&rs, 0.6),   // Rb
        tfl.clone(),       // Tfl
        tfr.clone(),       // Tfr
        scale(&tfl, 0.6),  // Tbl
        scale(&tfr, 0.6),  // Tbr
    ]
}

/// The six module (main, sub) output-slot pairs in module order.
const MODULE_SLOTS: [(usize, usize); 6] = [(5, 7), (6, 8), (9, 11), (10, 12), (0, 3), (1, 4)];

#[test]
fn ice_acpl2_9_0_4_correlated_pairs_settle_and_keep_ratios() {
    let input = correlated_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_acpl2(&refs);
        last = decode_frame(&mut dec, bytes, 13);
    }
    let mut worst = 0.0f64;
    for ch in 0..13 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-440 9.0.4 ACPL_2 settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.25,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-440 9.0.4 ACPL_2 worst settled relative RMS err (correlated): {worst:.4}");
    for (p, q) in MODULE_SLOTS {
        let r = energy(&last[q]) / energy(&last[p]).max(1e-30);
        eprintln!("ROUND-440 9.0.4 ACPL_2 pair ({p},{q}) level ratio: {r:.3}");
        assert!(
            (0.15..=0.75).contains(&r),
            "pair ({p},{q}) level ratio drifted: {r:.3}"
        );
    }
}

#[test]
fn ice_acpl2_9_0_4_independent_pairs_keep_energy() {
    let mut input = correlated_input();
    input[0] = periodic_noise(14, 34, 0.3, 5); // L
    input[3] = periodic_noise(14, 34, 0.3, 7); // Lscr independent of L
    input[1] = periodic_noise(18, 38, 0.3, 9); // R
    input[4] = periodic_noise(18, 38, 0.3, 13); // Rscr
    input[5] = periodic_noise(20, 40, 0.3, 11); // Ls
    input[7] = periodic_noise(20, 40, 0.3, 23); // Lb
    input[6] = periodic_noise(24, 44, 0.3, 37); // Rs
    input[8] = periodic_noise(24, 44, 0.3, 41); // Rb
    input[9] = periodic_noise(30, 50, 0.25, 53); // Tfl
    input[11] = periodic_noise(30, 50, 0.25, 67); // Tbl
    input[10] = periodic_noise(34, 54, 0.25, 71); // Tfr
    input[12] = periodic_noise(34, 54, 0.25, 83); // Tbr
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_acpl2(&refs);
        last = decode_frame(&mut dec, bytes, 13);
    }
    for ch in 0..13 {
        let r = energy(&last[ch]) / energy(&input[ch]).max(1e-30);
        eprintln!("ROUND-440 9.0.4 ACPL_2 independent-pair energy ratio ch{ch}: {r:.3}");
        assert!(
            (0.4..=2.5).contains(&r),
            "channel {ch} settled energy ratio out of range: {r:.3}"
        );
    }
}

#[test]
fn ice_acpl1_9_0_4_ms_band_reconstructs_pairs() {
    // All pair content below acpl_qmf_band = 6 (2 250 Hz): the PARTIAL
    // config's M/S residual path reconstructs every pair — including
    // the front (L, Lscr) / (R, Rscr) pairs riding the third S-CPL
    // residual pair — waveform-accurately, regardless of correlation.
    let mut input = correlated_input();
    input[3] = periodic_tone(14, 0.35, 0.3); // Lscr independent of L
    input[4] = periodic_tone(18, 0.35, 0.8); // Rscr
    input[7] = periodic_tone(26, 0.35, 0.2); // Lb
    input[8] = periodic_tone(30, 0.35, 0.7); // Rb
    input[11] = periodic_tone(38, 0.3, 1.1); // Tbl
    input[12] = periodic_tone(42, 0.3, 1.6); // Tbr
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_acpl1(&refs);
        last = decode_frame(&mut dec, bytes, 13);
    }
    let mut worst = 0.0f64;
    for ch in 0..13 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-440 9.0.4 ACPL_1 settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.15,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-440 9.0.4 ACPL_1 worst settled relative RMS err (M/S band): {worst:.4}");
}

#[test]
fn ice_acpl_9_0_4_bitstream_re_reads_to_extracted_alpha_beta() {
    let input = correlated_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let bytes = enc.encode_frame_pcm_9_0_4_ice_acpl2(&refs);
    let _ = decode_frame(&mut dec, bytes, 13);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let ice = sub.tools.ice.as_deref().expect("ice element parsed");
    assert!(ice.b_5fronts, "b_5fronts channel mode");
    assert_eq!(ice.mode, IceCodecMode::AspxAcpl2);
    assert_eq!(ice.acpl_data.len(), 6, "six acpl_data_1ch() modules");
    let acfg = sub
        .tools
        .acpl_config_1ch_full
        .expect("FULL acpl_config_1ch sticky");
    // Recompute the front-left module's (α, β) from the (L, Lscr)
    // mid/side spectra exactly like the encoder (plain (P±Q)/2, no √2
    // scale; fresh TDAC state on the first frame). The front modules
    // are the fifth / sixth acpl_data_1ch() elements.
    let mut mid_state = oxideav_ac4::encoder_mdct::EncoderMdctState::new(N as u32);
    let mut side_state = oxideav_ac4::encoder_mdct::EncoderMdctState::new(N as u32);
    let mid_pcm: Vec<f32> = (0..N).map(|i| 0.5 * (input[0][i] + input[3][i])).collect();
    let side_pcm: Vec<f32> = (0..N).map(|i| 0.5 * (input[0][i] - input[3][i])).collect();
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
        acpl_synth::differential_decode(&ice.acpl_data[4].alpha1, acfg.num_param_bands, &mut ad);
    let beta_rows =
        acpl_synth::differential_decode(&ice.acpl_data[4].beta1, acfg.num_param_bands, &mut bd);
    assert_eq!(
        alpha_rows.first().map(|r| r.as_slice()),
        Some(alpha_q.as_slice()),
        "front-left module α row re-reads to the extractor output"
    );
    assert_eq!(
        beta_rows.first().map(|r| r.as_slice()),
        Some(beta_q.as_slice()),
        "front-left module β row re-reads to the extractor output"
    );
}

#[test]
fn ice_acpl_9_x_4_deterministic_and_lfe_arms_decode() {
    let input = correlated_input();
    let lfe = periodic_tone(3, 0.4, 0.0);
    let mut all: Vec<&[f32]> = input.iter().map(|v| v.as_slice()).collect();
    all.push(&lfe);
    let refs14: [&[f32]; 14] = std::array::from_fn(|i| all[i]);
    let refs13: [&[f32]; 13] = std::array::from_fn(|i| all[i]);
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_9_0_4_ice_acpl1(&refs13);
        let b = enc_b.encode_frame_pcm_9_0_4_ice_acpl1(&refs13);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
    // 9.1.4 arms decode to 14 channels with the LFE on the leading slot.
    let params = CodecParameters::audio(CodecId::new("ac4"));
    for acpl1 in [false, true] {
        let mut enc = Ac4ImsEncoder::new();
        let mut dec = Ac4Decoder::new(&params);
        let mut last = Vec::new();
        for _ in 0..4 {
            let bytes = if acpl1 {
                enc.encode_frame_pcm_9_1_4_ice_acpl1(&refs14)
            } else {
                enc.encode_frame_pcm_9_1_4_ice_acpl2(&refs14)
            };
            last = decode_frame(&mut dec, bytes, 14);
        }
        let lag = best_circular_lag(&lfe, &last[0]);
        let e = rel_rms_err(&lfe, &last[0], lag);
        assert!(
            e < 0.05,
            "ACPL{} 9.1.4 LFE settled error too high: {e:.4}",
            if acpl1 { 1 } else { 2 }
        );
    }
}
