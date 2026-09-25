//! Round 461 — 5.X / 7.X A-CPL (ASPX_ACPL_1 / _2 / _3) PCM parity:
//! encode a multi-frame sequence through every live 5.X / 7.X A-CPL
//! entry point, decode it back through `Ac4Decoder`, and pin the
//! settled per-channel RMS against the input.
//!
//! Refs ETSI TS 103 190-1 §5.3.4.3.2 (Table 181), §5.3.4.4.2 / .3
//! (Tables 184 / 185), §5.7.7.6.1 (Pseudocode 117), §5.7.7.6.2
//! (Pseudocode 118), §5.7.7.6.3 (Table 202, Pseudocode 120), §6.2.10
//! Table 213 (A-SPX channel order), §6.2.9 Table 212 (companding),
//! §5.7.1 (QMF round-trip alignment).

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const FRAMES: usize = 8;
const SETTLE: usize = 4;

/// Per-channel decorrelated multitone: channel `ch` gets its own
/// frequency set (all below 8 kHz so the A-SPX crossover does not
/// dominate the comparison) at amplitude `amp`. `LFE_TONE` is the
/// single in-band LFE tone.
const LFE_TONE: usize = 7;

fn channel_tone(ch: usize, frame: usize, amp: f32) -> Vec<f32> {
    let base = [
        [311.0f32, 1_150.0, 2_900.0, 4_700.0],
        [419.0, 1_370.0, 3_100.0, 5_300.0],
        [523.0, 1_610.0, 3_500.0, 6_100.0],
        [277.0, 990.0, 2_650.0, 4_100.0],
        [367.0, 1_270.0, 2_800.0, 4_900.0],
        [233.0, 870.0, 2_300.0, 3_700.0],
        [349.0, 1_090.0, 2_500.0, 4_300.0],
        [62.0, 62.0, 62.0, 62.0],
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

/// FNV-1a over the interleaved S16 bytes of the settled frames.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn decode_one(dec: &mut Ac4Decoder, bytes: Vec<u8>, ch: usize) -> (Vec<Vec<f32>>, Vec<u8>) {
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
    (out, raw.to_vec())
}

struct Run {
    /// Per-channel `output_rms / input_rms` over the settled frames.
    ratios: Vec<f32>,
    /// FNV-1a of the settled frames' interleaved S16 output.
    hash: u64,
}

/// Runs `FRAMES` frames through `enc_frame` (the LFE slot, when the
/// layout has one, gets the `LFE_TONE` set), decodes them, and returns
/// the settled per-channel ratios + the output hash. `gop` = I-frame
/// interval (1 = all I-frames).
fn measure<F>(label: &str, names: &[&str], amps: &[f32], gop: usize, mut enc_frame: F) -> Run
where
    F: FnMut(&mut Ac4ImsEncoder, &[Vec<f32>]) -> Vec<u8>,
{
    let ch = names.len();
    let has_lfe = names[ch - 1] == "LFE";
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut in_acc = vec![Vec::new(); ch];
    let mut out_acc = vec![Vec::new(); ch];
    let mut out_bytes = Vec::new();
    for frame in 0..FRAMES {
        let pcm: Vec<Vec<f32>> = (0..ch)
            .map(|c| {
                let tone = if has_lfe && c == ch - 1 { LFE_TONE } else { c };
                channel_tone(tone, frame, amps[c])
            })
            .collect();
        enc.b_iframe_global = frame % gop == 0;
        let bytes = enc_frame(&mut enc, &pcm);
        let (out, raw) = decode_one(&mut dec, bytes, ch);
        if frame >= SETTLE {
            for c in 0..ch {
                in_acc[c].extend_from_slice(&pcm[c]);
                out_acc[c].extend_from_slice(&out[c]);
            }
            out_bytes.extend_from_slice(&raw);
        }
    }
    eprintln!("== {label} (gop {gop})");
    let mut ratios = Vec::with_capacity(ch);
    for (c, name) in names.iter().enumerate() {
        let (i, o) = (rms(&in_acc[c]), rms(&out_acc[c]));
        let ratio = if i > 0.0 { o / i } else { 0.0 };
        eprintln!("   {name:>3}: in {i:.4}  out {o:.4}  ratio {ratio:.3}");
        ratios.push(ratio);
    }
    let hash = fnv1a(&out_bytes);
    eprintln!("   settled S16 hash {hash:#018x}");
    Run { ratios, hash }
}

/// Every channel of the decode carries energy — the round-456 finding
/// ("the pair synthesis runs on silence") stays closed.
fn assert_non_silent(label: &str, names: &[&str], run: &Run) {
    for (name, r) in names.iter().zip(&run.ratios) {
        assert!(
            *r > 0.05,
            "{label}: channel {name} decodes to silence (ratio {r:.3})"
        );
    }
}

/// Parity pin for the live ACPL_1 / ACPL_2 routes: every main channel
/// within `lo..=hi` of its input RMS, the LFE within ±10 %.
fn assert_parity(label: &str, names: &[&str], run: &Run, lo: f32, hi: f32) {
    for (name, r) in names.iter().zip(&run.ratios) {
        let (l, h) = if *name == "LFE" {
            (0.90, 1.10)
        } else {
            (lo, hi)
        };
        assert!(
            (l..=h).contains(r),
            "{label}: channel {name} ratio {r:.3} outside {l}..={h}"
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

// ---------------------------------------------------------------------
// Parity routes: the Table 181 / 184 carrier model + real A-SPX.
// ---------------------------------------------------------------------

/// The live ACPL_1 / ACPL_2 encode paths reconstruct every channel
/// within this window of its input RMS (parametric surround + A-SPX
/// above the crossover; the waveform band is exact).
const LO: f32 = 0.78;
const HI: f32 = 1.12;

#[test]
fn parity_5_0_acpl2_real_aspx() {
    let label = "5.0 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl2_real_aspx(&a5(p))
    });
    assert_parity(label, &N5, &r, LO, HI);
}

#[test]
fn parity_5_1_acpl2_real_aspx() {
    let label = "5.1 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N6, &A6, 1, |e, p| {
        e.encode_frame_pcm_5_1_acpl2_real_aspx(&a6(p))
    });
    assert_parity(label, &N6, &r, LO, HI);
}

#[test]
fn parity_5_0_acpl1_real_aspx() {
    let label = "5.0 ASPX_ACPL_1 (real aspx)";
    let r = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl1_real_aspx(&a5(p))
    });
    assert_parity(label, &N5, &r, LO, HI);
}

#[test]
fn parity_5_1_acpl1_real_aspx() {
    let label = "5.1 ASPX_ACPL_1 (real aspx)";
    let r = measure(label, &N6, &A6, 1, |e, p| {
        e.encode_frame_pcm_5_1_acpl1_real_aspx(&a6(p))
    });
    assert_parity(label, &N6, &r, LO, HI);
}

#[test]
fn parity_7_0_acpl2_real_aspx() {
    let label = "7.0 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N7, &A7, 1, |e, p| {
        e.encode_frame_pcm_7_0_acpl2_real_aspx(&a7(p))
    });
    assert_parity(label, &N7, &r, LO, HI);
}

#[test]
fn parity_7_1_acpl2_real_aspx() {
    let label = "7.1 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N8, &A8, 1, |e, p| {
        e.encode_frame_pcm_7_1_acpl2_real_aspx(&a8(p))
    });
    assert_parity(label, &N8, &r, LO, HI);
}

#[test]
fn parity_7_0_acpl1_real_alpha_beta() {
    let label = "7.0 ASPX_ACPL_1 (real alpha/beta + real aspx)";
    let r = measure(label, &N7, &A7, 1, |e, p| {
        e.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&a7(p))
    });
    assert_parity(label, &N7, &r, LO, HI);
}

#[test]
fn parity_7_1_acpl1_real_alpha_beta() {
    let label = "7.1 ASPX_ACPL_1 (real alpha/beta + real aspx)";
    let r = measure(label, &N8, &A8, 1, |e, p| {
        e.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&a8(p))
    });
    assert_parity(label, &N8, &r, LO, HI);
}

/// I + 3×P GOPs: the data elements are present on every frame
/// (Tables 25 / 33) and the decoder's sticky configs carry the
/// A-SPX / A-CPL configuration, so the parity holds across P-frames.
#[test]
fn parity_gop_5_0_acpl2_and_7_0_acpl1() {
    let label = "5.0 ASPX_ACPL_2 (real aspx)";
    let r = measure(label, &N5, &A5, 4, |e, p| {
        e.encode_frame_pcm_5_0_acpl2_real_aspx(&a5(p))
    });
    assert_parity(label, &N5, &r, LO, HI);
    let label = "7.0 ASPX_ACPL_1 (real alpha/beta + real aspx)";
    let r = measure(label, &N7, &A7, 4, |e, p| {
        e.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&a7(p))
    });
    assert_parity(label, &N7, &r, LO, HI);
}

/// The encoder-authored probe decodes bit-identically run to run (the
/// hash of the settled S16 output is a stable pin on one platform; it
/// is printed by every `measure` call for the record).
#[test]
fn decode_of_encoder_probe_is_bit_exact_run_to_run() {
    let label = "5.0 ASPX_ACPL_2 (real aspx)";
    let a = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl2_real_aspx(&a5(p))
    });
    let b = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl2_real_aspx(&a5(p))
    });
    assert_eq!(a.hash, b.hash, "5.0 ACPL_2 decode is not deterministic");
    let label = "7.1 ASPX_ACPL_1 (real alpha/beta + real aspx)";
    let a = measure(label, &N8, &A8, 1, |e, p| {
        e.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&a8(p))
    });
    let b = measure(label, &N8, &A8, 1, |e, p| {
        e.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&a8(p))
    });
    assert_eq!(a.hash, b.hash, "7.1 ACPL_1 decode is not deterministic");
}

// ---------------------------------------------------------------------
// Scaffold routes (minimum-bit-cost A-SPX / α-less SAP selector / the
// ASPX_ACPL_3 model): decoded, never silent.
// ---------------------------------------------------------------------

#[test]
fn decode_5_0_acpl1_real_alpha_beta_is_not_silent() {
    let label = "5.0 ASPX_ACPL_1 (real alpha/beta, scaffold aspx)";
    let r = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&a5(p))
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_0_acpl1_sap_is_not_silent() {
    let label = "5.0 ASPX_ACPL_1 (SAP selector, scaffold alpha)";
    let r = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl1_sap(&a5(p))
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_0_acpl3_real_aspx_is_not_silent() {
    let label = "5.0 ASPX_ACPL_3 (real aspx)";
    let r = measure(label, &N5, &A5, 1, |e, p| {
        e.encode_frame_pcm_5_0_acpl3_real_aspx(&a5(p), 1.0, 1.0, 1.0, 1.0)
    });
    assert_non_silent(label, &N5, &r);
}

#[test]
fn decode_5_1_acpl3_real_aspx_is_not_silent() {
    let label = "5.1 ASPX_ACPL_3 (real aspx)";
    let r = measure(label, &N6, &A6, 1, |e, p| {
        e.encode_frame_pcm_5_1_acpl3_real_aspx(&a6(p), 1.0, 1.0, 1.0, 1.0)
    });
    assert_non_silent(label, &N6, &r);
}
