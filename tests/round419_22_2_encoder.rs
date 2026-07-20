//! Round 419 — encoder ICE synthesis parity, part 3: the
//! `22_2_channel_element()` encode arm (TS 103 190-2 §6.2.4.3, both
//! Table 98 codec modes) from PCM, plus the encoder-side §5.7.5
//! companding decision (`select_compand_on_from_qmf`) for the
//! immersive companding routes.
//!
//! Measured here:
//! 1. Simple mode: 24 distinct channels round-trip with a small
//!    settled per-channel relative RMS error (pure per-channel MDCT
//!    path — no parametric stages).
//! 2. A-SPX mode: low-band content settles like Simple, and a channel
//!    with real HF content regenerates its band within 3 dB while an
//!    HF-silent partner stays down.
//! 3. Companding decision: a transient HF envelope trips
//!    `b_compand_on`, a stationary one (and silence) does not — the
//!    per-channel input to `companding_control(5)` on the immersive
//!    routes (the round-417 harness pins the decode-side effect of
//!    the resulting control words).
//! 4. Determinism.

use oxideav_ac4::aspx::{select_compand_on_from_qmf, ASPX_QMF_PCM_SCALE};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::qmf_slots_to_sb_major;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::qmf::QmfAnalysisBank;
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

fn hf_energy(pcm: &[f32], lo_hz: f32) -> f64 {
    let n = pcm.len();
    let lo_bin = (lo_hz * n as f32 / 48_000.0).ceil() as usize;
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

/// 24 distinct low-band frame-periodic tones (Table 21 order + two
/// LFEs last).
fn input_22_2() -> Vec<Vec<f32>> {
    (0..24)
        .map(|ch| {
            if ch >= 22 {
                periodic_tone(3 + (ch as u32 - 22), 0.4, 0.0) // LFEs
            } else {
                periodic_tone(12 + 2 * ch as u32, 0.3, 0.27 * ch as f32)
            }
        })
        .collect()
}

#[test]
fn el_22_2_simple_settles_per_channel() {
    let input = input_22_2();
    let refs: [&[f32]; 24] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..5 {
        let bytes = enc.encode_frame_pcm_22_2_simple(&refs);
        last = decode_frame(&mut dec, bytes, 24);
    }
    // Output order: [LFE, LFE2, then the 22 Table 21 channels].
    let mut worst = 0.0f64;
    for ch in 0..24 {
        let out = if ch >= 22 {
            &last[ch - 22]
        } else {
            &last[ch + 2]
        };
        let lag = best_circular_lag(&input[ch], out);
        let e = rel_rms_err(&input[ch], out, lag);
        assert!(
            e < 0.08,
            "22.2 Simple channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-419 22.2 Simple worst settled relative RMS err: {worst:.4}");
}

#[test]
fn el_22_2_aspx_settles_and_synthesises_hf() {
    let mut input = input_22_2();
    // Real HF on L; R stays HF-silent.
    let hf = periodic_tone(182, 0.3, 0.0); // 4 550 Hz — above the crossover
    for (a, b) in input[0].iter_mut().zip(hf.iter()) {
        *a += *b;
    }
    let refs: [&[f32]; 24] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_22_2_aspx(&refs);
        last = decode_frame(&mut dec, bytes, 24);
    }
    let mut worst = 0.0f64;
    for ch in 1..22 {
        let out = &last[ch + 2];
        let lag = best_circular_lag(&input[ch], out);
        let e = rel_rms_err(&input[ch], out, lag);
        assert!(
            e < 0.10,
            "22.2 A-SPX channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-419 22.2 A-SPX worst settled relative RMS err (low band): {worst:.4}");
    let e_in = hf_energy(&input[0], 3750.0);
    let e_out_l = hf_energy(&last[2], 3750.0);
    let e_out_r = hf_energy(&last[3], 3750.0);
    let ratio = e_out_l / e_in.max(1e-30);
    eprintln!(
        "ROUND-419 22.2 A-SPX HF energy: in {e_in:.6}, out L {e_out_l:.6} (ratio {ratio:.3}), out R {e_out_r:.6}"
    );
    assert!(
        (0.5..=2.0).contains(&ratio),
        "22.2 A-SPX L HF band within 3 dB of the input ({ratio:.3})"
    );
    assert!(
        e_out_r < e_out_l / 20.0,
        "HF-silent R stays at least 13 dB below L's regenerated band"
    );
}

#[test]
fn el_22_2_encode_is_deterministic() {
    let input = input_22_2();
    let refs: [&[f32]; 24] = std::array::from_fn(|i| input[i].as_slice());
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_22_2_aspx(&refs);
        let b = enc_b.encode_frame_pcm_22_2_aspx(&refs);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
}

/// Analyse one PCM frame at the A-SPX integer-PCM scale.
fn analyse(pcm: &[f32]) -> Vec<Vec<(f32, f32)>> {
    let scaled: Vec<f32> = pcm.iter().map(|&v| v * ASPX_QMF_PCM_SCALE).collect();
    let mut bank = QmfAnalysisBank::new();
    qmf_slots_to_sb_major(&bank.process_block(&scaled))
}

#[test]
fn compand_selector_fires_on_transients_only() {
    // Stationary HF: constant tone above the band edge.
    let stationary = periodic_tone(200, 0.3, 0.0);
    // Transient HF: the same tone gated to one eighth of the frame.
    let mut transient = vec![0.0f32; N];
    let burst = periodic_tone(200, 0.6, 0.0);
    transient[..N / 8].copy_from_slice(&burst[..N / 8]);
    let q_st = analyse(&stationary);
    let q_tr = analyse(&transient);
    assert!(
        !select_compand_on_from_qmf(&q_st, 10, 46, 2.5),
        "stationary HF must not trip companding"
    );
    assert!(
        select_compand_on_from_qmf(&q_tr, 10, 46, 2.5),
        "a gated HF burst must trip companding"
    );
    // Silence must not trip either.
    let q_sil = analyse(&vec![0.0f32; N]);
    assert!(
        !select_compand_on_from_qmf(&q_sil, 10, 46, 2.5),
        "silence must not trip companding"
    );
}
