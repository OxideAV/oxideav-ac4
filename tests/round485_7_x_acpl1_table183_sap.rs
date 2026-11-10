//! Round 485 — 7_X ASPX_ACPL_1 Table 183 quartets from the two
//! `chparam_info()` elements (in-repo errata A5, recommended
//! "Reading C"), authored by the encoder and applied by the decoder.
//!
//! Refs ETSI TS 103 190-1 §4.2.6.14 Table 33 (`case ASPX_ACPL_1:`),
//! §4.3.5.13 (`max_sfb_master`), §5.3.2 Pseudocode 59, §5.3.4.4.1
//! Table 183, §5.3.4.4.2, §5.7.7.6.3 Table 202 / Pseudocode 120;
//! `docs/audio/ac4/ac4-errata.md` §A5.
//!
//! What is pinned:
//!
//! * the live 7.0 / 7.1 ASPX_ACPL_1 encoder writes SAP-coded
//!   (`sap_mode == 3`) `chparam_info()` rows when the back channels
//!   are predictable from the surround carriers, and the transmitted
//!   residual shrinks against the identity-SAP writer;
//! * decoded-PCM parity on that content: every channel within the
//!   round-461 RMS window, and — because the content sits below
//!   `acpl_qmf_band`, where ASPX_ACPL_1 is waveform-coded — a
//!   per-channel waveform SNR that only holds if the decoder applies
//!   the transmitted quartets (an identity-reading decoder would hand
//!   the A-CPL module the prediction residual as the side track);
//! * I + 3×P GOPs keep the parity (the residual layer is present on
//!   every frame).

use oxideav_ac4::asf::{
    SEVEN_X_ACPL1_TABLE183_SOURCE, SevenXAcpl1Table183Source, seven_x_acpl1_sap_authoring,
};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::mch::SevenXCodecMode;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const FRAMES: usize = 8;
const SETTLE: usize = 4;

/// Sum of unit sines at `freqs`, scaled to `amp / freqs.len()`.
fn tones(freqs: &[f32], frame: usize, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = (frame * N + i) as f32 / FS;
            let v: f32 = freqs
                .iter()
                .map(|&f| (2.0 * std::f32::consts::PI * f * t).sin())
                .sum();
            amp * v / freqs.len() as f32
        })
        .collect()
}

/// 7.0 (3/4/0) content below the ASPX_ACPL_1 waveform crossover: the
/// back channels are mostly scaled copies of their surrounds (the
/// case SAP prediction exists for), plus a small independent part.
fn content(frame: usize, with_lfe: bool) -> Vec<Vec<f32>> {
    let l = tones(&[180.0, 410.0], frame, 0.4);
    let r = tones(&[230.0, 520.0], frame, 0.35);
    let c = tones(&[300.0, 610.0], frame, 0.3);
    let ls = tones(&[160.0, 370.0, 690.0], frame, 0.3);
    let rs = tones(&[210.0, 450.0, 740.0], frame, 0.28);
    let lb_own = tones(&[270.0], frame, 0.03);
    let rb_own = tones(&[330.0], frame, 0.03);
    let lb: Vec<f32> = ls.iter().zip(&lb_own).map(|(a, b)| 0.6 * a + b).collect();
    let rb: Vec<f32> = rs.iter().zip(&rb_own).map(|(a, b)| -0.4 * a + b).collect();
    let mut out = vec![l, r, c, ls, rs, lb, rb];
    if with_lfe {
        out.push(tones(&[62.0], frame, 0.1));
    }
    out
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
    for (i, c) in raw.as_chunks::<2>().0.iter().enumerate() {
        out[i % ch].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
    }
    out
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Waveform SNR (dB) of `out` against `inp` delayed by `lag` samples.
fn snr_db(inp: &[f32], out: &[f32], lag: usize) -> f32 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for k in lag..out.len().min(inp.len() + lag) {
        let x = inp[k - lag] as f64;
        let y = out[k] as f64;
        sig += x * x;
        err += (y - x) * (y - x);
    }
    (10.0 * (sig / err.max(1e-20)).log10()) as f32
}

/// The decoder's output latency against the input: the lag (in whole
/// samples) maximising the L-channel cross-correlation.
fn best_lag(inp: &[f32], out: &[f32]) -> usize {
    let mut best = (0usize, f64::MIN);
    for lag in 0..2 * N {
        let mut acc = 0.0f64;
        for k in lag..out.len().min(inp.len() + lag) {
            acc += inp[k - lag] as f64 * out[k] as f64;
        }
        if acc > best.1 {
            best = (lag, acc);
        }
    }
    best.0
}

struct Run {
    ratios: Vec<f32>,
    snrs: Vec<f32>,
    sap_modes: [u32; 2],
}

fn run(names: &[&str], gop: usize) -> Run {
    let ch = names.len();
    let with_lfe = ch == 8;
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut inp = vec![Vec::new(); ch];
    let mut out = vec![Vec::new(); ch];
    let mut sap_modes = [0u32; 2];
    for frame in 0..FRAMES {
        let pcm = content(frame, with_lfe);
        enc.b_iframe_global = frame % gop == 0;
        let bytes = if with_lfe {
            let f: [&[f32]; 8] = std::array::from_fn(|c| pcm[c].as_slice());
            enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&f)
        } else {
            let f: [&[f32]; 7] = std::array::from_fn(|c| pcm[c].as_slice());
            enc.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&f)
        };
        let dec_out = decode_one(&mut dec, bytes, ch);
        let tools = &dec.last_substream.as_ref().expect("substream").tools;
        assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::AspxAcpl1));
        for (slot, m) in sap_modes.iter_mut().enumerate() {
            let cp = tools.acpl_1_residual_chparam[slot]
                .as_ref()
                .expect("residual chparam_info parsed");
            *m = (*m).max(cp.sap_mode);
        }
        for c in 0..ch {
            inp[c].extend_from_slice(&pcm[c]);
            out[c].extend_from_slice(&dec_out[c]);
        }
    }
    // Settled window: drop the first SETTLE frames of the output.
    let lag = best_lag(&inp[0], &out[0]);
    let start = SETTLE * N;
    let mut ratios = Vec::with_capacity(ch);
    let mut snrs = Vec::with_capacity(ch);
    eprintln!("== 7.X ASPX_ACPL_1 Table 183 SAP (gop {gop}, lag {lag})");
    for (c, name) in names.iter().enumerate() {
        let o = &out[c][start..];
        let i = &inp[c][start - lag..inp[c].len() - lag];
        let ratio = rms(o) / rms(i);
        let snr = snr_db(i, o, 0);
        eprintln!("   {name:>3}: ratio {ratio:.3}  snr {snr:.1} dB");
        ratios.push(ratio);
        snrs.push(snr);
    }
    Run {
        ratios,
        snrs,
        sap_modes,
    }
}

const N7: [&str; 7] = ["L", "R", "C", "Ls", "Rs", "Lb", "Rb"];
const N8: [&str; 8] = ["L", "R", "C", "Ls", "Rs", "Lb", "Rb", "LFE"];

fn assert_parity(label: &str, names: &[&str], r: &Run) {
    for (k, name) in names.iter().enumerate() {
        let ratio = r.ratios[k];
        assert!(
            (0.9..=1.1).contains(&ratio),
            "{label}: {name} RMS ratio {ratio:.3} outside 0.9..=1.1"
        );
        if *name != "LFE" {
            assert!(
                r.snrs[k] >= 20.0,
                "{label}: {name} waveform SNR {:.1} dB < 20 dB",
                r.snrs[k]
            );
        }
    }
}

#[test]
fn errata_a5_reading_in_force_is_chparam_quartets() {
    assert_eq!(
        SEVEN_X_ACPL1_TABLE183_SOURCE,
        SevenXAcpl1Table183Source::ChparamQuartets
    );
    assert!(seven_x_acpl1_sap_authoring());
}

#[test]
fn seven_0_acpl1_sap_coded_quartets_decode_with_waveform_parity() {
    let r = run(&N7, 1);
    assert_eq!(r.sap_modes, [3, 3], "both residual rows must be SAP-coded");
    assert_parity("7.0 ACPL_1 SAP", &N7, &r);
}

#[test]
fn seven_1_acpl1_sap_coded_quartets_decode_with_waveform_parity() {
    let r = run(&N8, 1);
    assert_eq!(r.sap_modes, [3, 3], "both residual rows must be SAP-coded");
    assert_parity("7.1 ACPL_1 SAP", &N8, &r);
}

#[test]
fn seven_0_acpl1_sap_parity_holds_across_p_frames() {
    let r = run(&N7, 4);
    assert_eq!(r.sap_modes, [3, 3]);
    assert_parity("7.0 ACPL_1 SAP (I + 3xP)", &N7, &r);
}

/// The transmitted side residual shrinks against the identity-SAP
/// writer on the same content (the bit-efficiency reason SAP exists):
/// the authored wire `F` / `G` carry only the unpredictable part.
#[test]
fn sap_wire_side_is_smaller_than_identity_side() {
    use oxideav_ac4::encoder_acpl3::author_7x_acpl1_table183_wire;
    use oxideav_ac4::encoder_mdct::EncoderMdctState;
    let pcm = content(0, false);
    let spec = |x: &[f32]| {
        let mut st = EncoderMdctState::new(N as u32);
        let _ = st.analyse_frame(x);
        st.analyse_frame(x)
    };
    let k = 1.0 / (2.0 * std::f32::consts::SQRT_2);
    let mix = |a: &[f32], b: &[f32], s: f32| -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| k * (x + s * y)).collect()
    };
    let d = spec(&mix(&pcm[3], &pcm[5], 1.0));
    let e = spec(&mix(&pcm[4], &pcm[6], 1.0));
    let f = spec(&mix(&pcm[3], &pcm[5], -1.0));
    let g = spec(&mix(&pcm[4], &pcm[6], -1.0));
    let wire = author_7x_acpl1_table183_wire(N as u32, 20, &d, &e, &f, &g, true);
    assert_eq!(wire.chparam[0].sap_mode, 3);
    assert_eq!(wire.chparam[1].sap_mode, 3);
    let energy = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
    let (ef, eg) = (energy(&wire.f), energy(&wire.g));
    let (ef_id, eg_id) = (energy(&f), energy(&g));
    eprintln!("side energy: F {ef:.4e} vs identity {ef_id:.4e}; G {eg:.4e} vs {eg_id:.4e}");
    assert!(
        ef < 0.5 * ef_id,
        "F residual {ef:.4e} vs identity {ef_id:.4e}"
    );
    assert!(
        eg < 0.5 * eg_id,
        "G residual {eg:.4e} vs identity {eg_id:.4e}"
    );
}
