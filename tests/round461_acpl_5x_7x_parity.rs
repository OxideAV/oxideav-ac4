//! Round 461 — 5.X / 7.X A-CPL (ASPX_ACPL_1 / _2 / _3) PCM parity:
//! encode a multi-frame sequence through every live 5.X / 7.X A-CPL
//! entry point, decode it back through `Ac4Decoder`, and measure the
//! settled per-channel RMS against the input.
//!
//! Refs ETSI TS 103 190-1 §5.3.4.3.2 (Table 181), §5.3.4.4.2 / .3
//! (Tables 184 / 185), §5.7.7.6.1 (Pseudocode 117), §5.7.7.6.2
//! (Pseudocode 118), §5.7.7.6.3 (Table 202, Pseudocode 120), §6.2.10
//! Table 213 (A-SPX channel order), §6.2.9 Table 212 (companding).

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const FRAMES: usize = 8;
const SETTLE: usize = 4;

/// Per-channel decorrelated multitone: channel `ch` gets its own
/// frequency set (all below 8 kHz so the A-SPX crossover does not
/// dominate the comparison) at amplitude `amp`.
fn channel_tone(ch: usize, frame: usize, amp: f32) -> Vec<f32> {
    let base = [
        [311.0f32, 1_150.0, 2_900.0, 4_700.0],
        [419.0, 1_370.0, 3_100.0, 5_300.0],
        [523.0, 1_610.0, 3_500.0, 6_100.0],
        [277.0, 990.0, 2_650.0, 4_100.0],
        [367.0, 1_270.0, 2_800.0, 4_900.0],
        [233.0, 870.0, 2_300.0, 3_700.0],
        [349.0, 1_090.0, 2_500.0, 4_300.0],
        [55.0, 80.0, 95.0, 110.0],
    ][ch];
    (0..N)
        .map(|i| {
            let t = (frame * N + i) as f32 / FS;
            let mut v = 0.0f32;
            for &f in &base {
                v += (2.0 * std::f32::consts::PI * f * t).sin();
            }
            amp * v / 4.0
        })
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn decode_one(dec: &mut Ac4Decoder, bytes: Vec<u8>, ch: usize) -> Vec<Vec<f32>> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples as usize, N);
    let raw = &af.data[0];
    assert_eq!(raw.len(), N * ch * 2, "{ch}-ch S16");
    let mut out = vec![Vec::with_capacity(N); ch];
    for (i, c) in raw.chunks_exact(2).enumerate() {
        out[i % ch].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
    }
    out
}

/// Runs `FRAMES` frames through `enc_frame`, decodes them, and returns
/// the per-channel `output_rms / input_rms` over the settled frames.
fn measure<F>(label: &str, names: &[&str], amps: &[f32], mut enc_frame: F) -> Vec<f32>
where
    F: FnMut(&mut Ac4ImsEncoder, &[Vec<f32>]) -> Vec<u8>,
{
    let ch = names.len();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut in_acc = vec![Vec::new(); ch];
    let mut out_acc = vec![Vec::new(); ch];
    for frame in 0..FRAMES {
        let pcm: Vec<Vec<f32>> = (0..ch).map(|c| channel_tone(c, frame, amps[c])).collect();
        let bytes = enc_frame(&mut enc, &pcm);
        let out = decode_one(&mut dec, bytes, ch);
        if frame >= SETTLE {
            for c in 0..ch {
                in_acc[c].extend_from_slice(&pcm[c]);
                out_acc[c].extend_from_slice(&out[c]);
            }
        }
    }
    eprintln!("== {label}");
    let mut ratios = Vec::with_capacity(ch);
    for (c, name) in names.iter().enumerate() {
        let (i, o) = (rms(&in_acc[c]), rms(&out_acc[c]));
        let ratio = if i > 0.0 { o / i } else { 0.0 };
        eprintln!("   {name:>3}: in {i:.4}  out {o:.4}  ratio {ratio:.3}");
        ratios.push(ratio);
    }
    ratios
}

/// Every channel of the decode carries energy — the round-456 finding
/// ("the pair synthesis runs on silence") is closed on the decoder side.
fn assert_non_silent(label: &str, names: &[&str], ratios: &[f32]) {
    for (name, r) in names.iter().zip(ratios) {
        assert!(
            *r > 0.05,
            "{label}: channel {name} decodes to silence (ratio {r:.3})"
        );
    }
}

const N5: [&str; 5] = ["L", "R", "C", "Ls", "Rs"];
const N6: [&str; 6] = ["L", "R", "C", "Ls", "Rs", "LFE"];
const N7: [&str; 7] = ["L", "R", "C", "Ls", "Rs", "Lb", "Rb"];
const N8: [&str; 8] = ["L", "R", "C", "Ls", "Rs", "Lb", "Rb", "LFE"];
const A5: [f32; 5] = [0.5, 0.4, 0.3, 0.25, 0.2];
const A6: [f32; 6] = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15];
const A7: [f32; 7] = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1];
const A8: [f32; 8] = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];

fn a5(p: &[Vec<f32>]) -> [&[f32]; 5] {
    [&p[0], &p[1], &p[2], &p[3], &p[4]]
}
fn a6(p: &[Vec<f32>]) -> [&[f32]; 6] {
    [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]]
}
fn a7(p: &[Vec<f32>]) -> [&[f32]; 7] {
    [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6]]
}
fn a8(p: &[Vec<f32>]) -> [&[f32]; 8] {
    [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7]]
}

#[test]
fn decode_5_0_acpl1_real_alpha_beta_is_not_silent() {
    let label = "5.0 ASPX_ACPL_1 (real alpha/beta)";
    let r = measure(label, &N5, &A5, |e, p| {
        e.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&a5(p))
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_0_acpl1_sap_is_not_silent() {
    let label = "5.0 ASPX_ACPL_1 (SAP)";
    let r = measure(label, &N5, &A5, |e, p| {
        e.encode_frame_pcm_5_0_acpl1_sap(&a5(p))
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_0_acpl2_real_aspx_is_not_silent() {
    let label = "5.0 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N5, &A5, |e, p| {
        e.encode_frame_pcm_5_0_acpl2_real_aspx(&a5(p))
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_0_acpl3_real_aspx_is_not_silent() {
    let label = "5.0 ASPX_ACPL_3 (real aspx)";
    let r = measure(label, &N5, &A5, |e, p| {
        e.encode_frame_pcm_5_0_acpl3_real_aspx(&a5(p), 1.0, 1.0, 1.0, 1.0)
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_1_acpl3_real_aspx_is_not_silent() {
    let label = "5.1 ASPX_ACPL_3 (real aspx)";
    let r = measure(label, &N6, &A6, |e, p| {
        e.encode_frame_pcm_5_1_acpl3_real_aspx(&a6(p), 1.0, 1.0, 1.0, 1.0)
    });
    assert_non_silent(label, &N6, &r);
}

#[test]
fn decode_7_0_acpl1_real_alpha_beta_is_not_silent() {
    let label = "7.0 ASPX_ACPL_1 (real alpha/beta)";
    let r = measure(label, &N7, &A7, |e, p| {
        e.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&a7(p))
    });
    assert_non_silent(label, &N7, &r);
}

#[test]
fn decode_7_1_acpl1_real_alpha_beta_is_not_silent() {
    let label = "7.1 ASPX_ACPL_1 (real alpha/beta)";
    let r = measure(label, &N8, &A8, |e, p| {
        e.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&a8(p))
    });
    assert_non_silent(label, &N8, &r);
}

#[test]
fn decode_7_0_acpl2_real_aspx_is_not_silent() {
    let label = "7.0 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N7, &A7, |e, p| {
        e.encode_frame_pcm_7_0_acpl2_real_aspx(&a7(p))
    });
    assert_non_silent(label, &N7, &r);
}

#[test]
fn decode_7_1_acpl2_real_aspx_is_not_silent() {
    let label = "7.1 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N8, &A8, |e, p| {
        e.encode_frame_pcm_7_1_acpl2_real_aspx(&a8(p))
    });
    assert_non_silent(label, &N8, &r);
}
