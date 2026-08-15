//! Round 440 — the ICE **SAP encode decisions** (TS 103 190-2
//! §5.2.3.2 steps 3-6): plain-SCPL encode arms for 7.X.4 / 9.X.4 that
//! automatically
//!
//! * M/S + prediction code the `(D, F)` / `(E, G)` quartets
//!   (`b_use_sap_add_ch`, step 3/4 — Pseudocode 59 quartet inverse),
//!   and
//! * predict each S-CPL track from its Table 20 source carrier
//!   (step 5/6 — full-SAP `chparam_info()` gains `a′ = alpha_q · 0,1`,
//!   wire track = prediction residual).
//!
//! Measured here:
//! 1. Round-trip: correlated vertical content decodes back within the
//!    ASF quantisation floor on all channels (both layouts + LFE).
//! 2. Engagement: the parsed element carries `b_use_sap_add_ch = 1`
//!    and full-SAP S-CPL chparam elements whose extracted `a′` rows
//!    match the constructed correlations.
//! 3. Bit savings: the SAP frame is smaller than an identity-SAP
//!    encode of the same content (the predicted tracks code darker).
//! 4. Determinism.

use oxideav_ac4::asf::extract_sap_a_prime;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::encoder_mdct::EncoderMdctState;
use oxideav_ac4::ice::{encode_ice_raw_frame, write_ice_body_scpl, IceCodecMode, IceScplSpectra};
use oxideav_core::bits::BitWriter;
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

fn decode_frame(dec: &mut Ac4Decoder, bytes: Vec<u8>, channels: usize) -> Vec<Vec<f32>> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let Frame::Audio(af) = frame else {
        panic!("expected audio frame");
    };
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

/// Correlated vertical 7.0.4 content: the top channels track their
/// surround carriers (`Tfl = 0,8·Ls`, `Tbl = 0,8·Lb`) and the backs
/// track the surrounds (`Lb = 0,7·Ls`), so both SAP stages engage on
/// the left side; the right side carries independent tones (identity
/// elements). The left-side base is a multi-tone (broadband-ish)
/// signal so the predicted-to-residual tracks carry real bit cost.
/// Order: `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
fn correlated_vertical_input() -> Vec<Vec<f32>> {
    let parts: [(u32, f32, f32); 5] = [
        (24, 0.10, 0.4),
        (29, 0.09, 1.1),
        (33, 0.08, 0.2),
        (37, 0.07, 0.9),
        (41, 0.06, 1.7),
    ];
    let mut ls = vec![0.0f32; N];
    for &(c, a, p) in &parts {
        for (dst, s) in ls.iter_mut().zip(periodic_tone(c, a, p)) {
            *dst += s;
        }
    }
    // Ratio 2/3 everywhere puts every SAP gain exactly on the
    // alpha_q · 0,1 grid: the (D, F) quartet gain and both left-side
    // step-5/6 predictions land on (1 − 2/3)/(1 + 2/3) = 0,2, so the
    // predicted wire tracks are exactly zero.
    const G: f32 = 2.0 / 3.0;
    let scale = |x: &Vec<f32>, g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
    let lb = scale(&ls, G);
    vec![
        periodic_tone(9, 0.35, 0.0),  // L
        periodic_tone(22, 0.35, 0.5), // R
        periodic_tone(20, 0.35, 1.0), // C
        ls.clone(),                   // Ls
        periodic_tone(28, 0.35, 0.9), // Rs
        lb.clone(),                   // Lb
        periodic_tone(32, 0.3, 1.4),  // Rb
        scale(&ls, G),                // Tfl = ⅔·Ls
        periodic_tone(40, 0.3, 1.8),  // Tfr
        scale(&lb, G),                // Tbl = ⅔·Lb
        periodic_tone(44, 0.3, 2.2),  // Tbr
    ]
}

/// 9.0.4 variant: additionally `Lscr = 0,5·L` so the front residual
/// `L″` predicts from the front mid track A.
/// Order: `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
fn correlated_vertical_input_9() -> Vec<Vec<f32>> {
    let base = correlated_vertical_input();
    let scale = |x: &Vec<f32>, g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
    vec![
        base[0].clone(),             // L
        base[1].clone(),             // R
        base[2].clone(),             // C
        scale(&base[0], 0.5),        // Lscr = 0,5·L
        periodic_tone(26, 0.3, 0.7), // Rscr
        base[3].clone(),             // Ls
        base[4].clone(),             // Rs
        base[5].clone(),             // Lb
        base[6].clone(),             // Rb
        base[7].clone(),             // Tfl
        base[8].clone(),             // Tfr
        base[9].clone(),             // Tbl
        base[10].clone(),            // Tbr
    ]
}

#[test]
fn ice_scpl_sap_7_0_4_round_trips_correlated_content() {
    let input = correlated_vertical_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_scpl_sap(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    let mut worst = 0.0f64;
    for ch in 0..11 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-440 SCPL-SAP settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.08,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-440 SCPL-SAP 7.0.4 worst settled relative RMS err: {worst:.4}");
}

#[test]
fn ice_scpl_sap_9_x_4_round_trips_and_lfe() {
    let input = correlated_vertical_input_9();
    let lfe = periodic_tone(3, 0.4, 0.0);
    let mut all: Vec<&[f32]> = input.iter().map(|v| v.as_slice()).collect();
    all.push(&lfe);
    let refs14: [&[f32]; 14] = std::array::from_fn(|i| all[i]);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_9_1_4_ice_scpl_sap(&refs14);
        last = decode_frame(&mut dec, bytes, 14);
    }
    let lag = best_circular_lag(&lfe, &last[0]);
    let e_lfe = rel_rms_err(&lfe, &last[0], lag);
    assert!(e_lfe < 0.05, "9.1.4 LFE settled error too high: {e_lfe:.4}");
    for ch in 0..13 {
        let lag = best_circular_lag(&input[ch], &last[ch + 1]);
        let e = rel_rms_err(&input[ch], &last[ch + 1], lag);
        eprintln!("ROUND-440 SCPL-SAP 9.1.4 settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.08,
            "9.1.4 channel {ch} settled relative RMS error too high: {e:.4}"
        );
    }
}

#[test]
fn ice_scpl_sap_engages_and_gains_match_correlations() {
    let input = correlated_vertical_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let bytes = enc.encode_frame_pcm_7_0_4_ice_scpl_sap(&refs);
    let _ = decode_frame(&mut dec, bytes, 11);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let ice = sub.tools.ice.as_deref().expect("ice element parsed");
    assert_eq!(ice.mode, IceCodecMode::Scpl);
    // Step 3/4: the (D, F) quartet — surround mid D ∝ (5/3)·Ls vs
    // top-front mid F = ⅔·D — is fully correlated, so the pair
    // engages.
    assert_eq!(ice.b_use_sap_add_ch, Some(true), "b_use_sap_add_ch");
    let cp = ice.sap_add_chparam.as_deref().expect("sap add chparam");
    assert_eq!(cp[0].sap_mode, 3, "(D, F) chparam is full SAP");
    // Step 5/6: the (H ← D) element engages. H = (Ls − Lb)/(2√2)
    // ∝ (1/3)·Ls against D ∝ (5/3)·Ls → a′ = 0,2 exactly on the
    // alpha_q · 0,1 grid.
    assert_eq!(ice.scpl_chparam.len(), 4, "four S-CPL chparam elements");
    let h_info = &ice.scpl_chparam[0];
    assert_eq!(h_info.sap_mode, 3, "(H ← D) element is full SAP");
    let a_prime = extract_sap_a_prime(h_info, &[40]);
    let row = a_prime.first().expect("a' row");
    let peak = row.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        (0.1..=0.3).contains(&peak),
        "peak a' on the (H ← D) element ≈ 0,2 (got {peak})"
    );
    // (No claim about the right-side elements: even nominally
    // independent tones share first-frame TDAC window-skirt leakage,
    // and pairs may legitimately engage wherever the quantised gain
    // clears the energy-reduction gate — the round-trip tests pin the
    // decode either way.)
}

#[test]
fn ice_scpl_sap_saves_bits_against_identity_encode() {
    let input = correlated_vertical_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let mut enc = Ac4ImsEncoder::new();
    let sap_frame = enc.encode_frame_pcm_7_0_4_ice_scpl_sap(&refs);

    // Identity encode of the same content: same Table 23 track
    // derivation + MDCT, no SAP stages (write_ice_body_scpl).
    let n = N;
    let half = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| v * 0.5).collect() };
    let q = 0.5 * std::f32::consts::FRAC_1_SQRT_2;
    let mix = |x: &[f32], y: &[f32], sign: f32| -> Vec<f32> {
        (0..n).map(|i| q * (x[i] + sign * y[i])).collect()
    };
    let tracks: Vec<Vec<f32>> = vec![
        half(&input[0]),
        half(&input[1]),
        half(&input[2]),
        mix(&input[3], &input[5], 1.0),
        mix(&input[4], &input[6], 1.0),
        mix(&input[7], &input[9], 1.0),
        mix(&input[8], &input[10], 1.0),
        mix(&input[3], &input[5], -1.0),
        mix(&input[4], &input[6], -1.0),
        mix(&input[7], &input[9], -1.0),
        mix(&input[8], &input[10], -1.0),
    ];
    let coeffs: Vec<Vec<f32>> = tracks
        .iter()
        .map(|pcm| EncoderMdctState::new(N as u32).analyse_frame(pcm))
        .collect();
    let core: [&[f32]; 5] = [&coeffs[0], &coeffs[1], &coeffs[2], &coeffs[3], &coeffs[4]];
    let scpl_pairs: [[&[f32]; 2]; 2] = [[&coeffs[7], &coeffs[8]], [&coeffs[9], &coeffs[10]]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&coeffs[5], &coeffs[6]],
        scpl_pairs: &scpl_pairs,
    };
    let mut body = BitWriter::new();
    write_ice_body_scpl(&mut body, &spectra, None, false, N as u32, 40).expect("identity body");
    let identity_frame = encode_ice_raw_frame(0, false, false, true, body).expect("frame");
    eprintln!(
        "ROUND-440 SCPL-SAP frame {} B vs identity {} B",
        sap_frame.len(),
        identity_frame.len()
    );
    assert!(
        sap_frame.len() < identity_frame.len(),
        "SAP prediction must code the correlated content smaller ({} vs {})",
        sap_frame.len(),
        identity_frame.len()
    );
}

#[test]
fn ice_scpl_sap_encode_is_deterministic() {
    let input = correlated_vertical_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_7_0_4_ice_scpl_sap(&refs);
        let b = enc_b.encode_frame_pcm_7_0_4_ice_scpl_sap(&refs);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
}
