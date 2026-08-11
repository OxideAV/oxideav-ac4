//! Round 440 — b_5fronts ICE encode, part 1: the 9.0.4 / 9.1.4
//! ASPX_SCPL immersive-channel-element encode route (TS 103 190-2
//! §6.2.4.1, `immersive_codec_mode = ASPX_SCPL`, `b_5fronts = 1`)
//! from PCM.
//!
//! The encoder derives the thirteen SMP tracks A..M as the exact
//! inverse of the decoder's §5.3.3.1 Table 23 S-CPL full-decoding
//! matrix (whose `b_5fronts` front rows carry the fixed ×2 matrix)
//! composed with the §4.8.3.11.3 Table 11 output gains, forward-MDCTs
//! them on persistent TDAC states, and emits real SIGNAL / NOISE
//! A-SPX envelopes (+ `aspx_tna_mode` / `aspx_add_harmonic`) per
//! `b_5fronts` Table 8 channel group.
//!
//! Measured here:
//! 1. Settled waveform round-trip on frame-periodic low-band content —
//!    circularly aligned per-channel relative RMS error against the
//!    input frame, all 13 channels.
//! 2. Real-envelope HF synthesis — a channel with HF content decodes
//!    with matching HF band energy (within 3 dB) while an HF-silent
//!    partner stays quiet.
//! 3. 9.1.4: the LFE arm decodes on the leading output slot.
//! 4. Parse shape — the emitted bitstream re-reads as an ASPX_SCPL
//!    element with the 7-payload `b_5fronts` roster and 3 S-CPL pairs.
//! 5. Determinism — fresh encoder states produce identical bytes.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::ice::{IceAspxElement, IceCodecMode};
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

/// Frame-periodic tone: `cycles` full cycles per 1920-sample frame
/// (f = cycles · 25 Hz).
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

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// HF-band energy of a PCM frame above `lo_hz` (DFT projection).
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

/// Thirteen distinct frame-periodic low-band tones (cycles → 25·c Hz),
/// all below the live config's crossover (sbx = 10 → 3 750 Hz).
fn low_band_input() -> Vec<Vec<f32>> {
    let cycles = [12u32, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]; // 300..1500 Hz
    cycles
        .iter()
        .enumerate()
        .map(|(ch, &c)| periodic_tone(c, 0.35, 0.3 * ch as f32))
        .collect()
}

#[test]
fn ice_aspx_scpl_9_0_4_settled_low_band_round_trip() {
    let input = low_band_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 13);
    }
    let mut worst = 0.0f64;
    for ch in 0..13 {
        let lag = best_circular_lag(&input[ch], &last[ch]);
        let e = rel_rms_err(&input[ch], &last[ch], lag);
        eprintln!("ROUND-440 9.0.4 ASPX_SCPL settled relative RMS err ch{ch}: {e:.4}");
        assert!(
            e < 0.10,
            "channel {ch} settled relative RMS error too high: {e:.4}"
        );
        worst = worst.max(e);
    }
    eprintln!("ROUND-440 9.0.4 ASPX_SCPL worst settled relative RMS err: {worst:.4}");
}

#[test]
fn ice_aspx_scpl_9_0_4_real_envelopes_synthesise_hf() {
    // L carries a strong HF component; the front-pair partner Lscr
    // stays HF-silent. With `b_5fronts` the front A-SPX groups are
    // (L, Lscr) / (R, Rscr), so the real envelopes must regenerate
    // L's HF at level while leaving Lscr's high band quiet.
    let sbx_hz: f32 = 10.0 * 375.0;
    let hf_cycles = ((sbx_hz + 800.0) / 25.0).ceil() as u32;
    let mut input = low_band_input();
    let hf = periodic_tone(hf_cycles, 0.3, 0.0);
    for (a, b) in input[0].iter_mut().zip(hf.iter()) {
        *a += *b;
    }
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 13);
    }
    let e_in_l = hf_energy(&input[0], sbx_hz);
    let e_out_l = hf_energy(&last[0], sbx_hz);
    let e_out_lscr = hf_energy(&last[3], sbx_hz);
    let ratio = e_out_l / e_in_l.max(1e-30);
    eprintln!(
        "ROUND-440 9.0.4 ASPX_SCPL HF energy: in L {e_in_l:.6}, out L {e_out_l:.6} (ratio {ratio:.3}), out Lscr {e_out_lscr:.6}"
    );
    assert!(
        (0.5..=2.0).contains(&ratio),
        "L HF band energy within 3 dB of the input ({ratio:.3})"
    );
    assert!(
        e_out_lscr < e_out_l / 20.0,
        "HF-silent Lscr stays at least 13 dB below L's regenerated band ({e_out_lscr} vs {e_out_l})"
    );
}

#[test]
fn ice_aspx_scpl_9_1_4_lfe_round_trip() {
    let named = low_band_input();
    let lfe = periodic_tone(3, 0.4, 0.0); // 75 Hz
    let mut all: Vec<&[f32]> = named.iter().map(|v| v.as_slice()).collect();
    all.push(&lfe);
    let refs: [&[f32]; 14] = std::array::from_fn(|i| all[i]);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_1_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 14);
    }
    // Output order: [LFE, L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl,
    // Tfr, Tbl, Tbr].
    let lag = best_circular_lag(&lfe, &last[0]);
    let e_lfe = rel_rms_err(&lfe, &last[0], lag);
    eprintln!("ROUND-440 9.1.4 ASPX_SCPL LFE settled relative RMS err: {e_lfe:.4}");
    assert!(e_lfe < 0.05, "LFE settled error too high: {e_lfe:.4}");
    for ch in 0..13 {
        let lag = best_circular_lag(&named[ch], &last[ch + 1]);
        let e = rel_rms_err(&named[ch], &last[ch + 1], lag);
        assert!(
            e < 0.10,
            "9.1.4 named channel {ch} settled relative RMS error too high: {e:.4}"
        );
    }
}

#[test]
fn ice_aspx_scpl_9_0_4_parses_with_5fronts_roster() {
    let input = low_band_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let bytes = enc.encode_frame_pcm_9_0_4_ice_aspx_scpl(&refs);
    let _ = decode_frame(&mut dec, bytes, 13);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    let ice = sub.tools.ice.as_deref().expect("ice element parsed");
    assert!(ice.b_5fronts, "b_5fronts channel mode");
    assert!(!ice.b_lfe, "9.0.4 carries no LFE");
    assert_eq!(ice.mode, IceCodecMode::AspxScpl);
    assert_eq!(
        ice.aspx_elements.len(),
        7,
        "b_5fronts ASPX_SCPL payload roster: 6× 2ch + 1× 1ch"
    );
    let n_two = ice
        .aspx_elements
        .iter()
        .filter(|e| matches!(e, IceAspxElement::TwoCh(_)))
        .count();
    assert_eq!(n_two, 6, "six aspx_data_2ch payloads");
    assert!(
        matches!(&ice.aspx_elements[2], IceAspxElement::OneCh(Some(_))),
        "1ch payload spliced at the third roster position"
    );
    assert_eq!(ice.scpl_pairs.len(), 3, "three S-CPL pairs (H..M)");
    assert_eq!(ice.scpl_chparam.len(), 6, "six S-CPL chparam elements");
    for (i, el) in ice.aspx_elements.iter().enumerate() {
        let parsed = match el {
            IceAspxElement::TwoCh(t) | IceAspxElement::OneCh(t) => t.is_some(),
        };
        assert!(parsed, "payload {i} trailer parsed");
    }
}

#[test]
fn ice_aspx_scpl_9_0_4_encode_is_deterministic() {
    let input = low_band_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_9_0_4_ice_aspx_scpl(&refs);
        let b = enc_b.encode_frame_pcm_9_0_4_ice_aspx_scpl(&refs);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
}

#[test]
fn ice_aspx_scpl_9_0_4_energy_reaches_every_output_slot() {
    let input = low_band_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_aspx_scpl(&refs);
        last = decode_frame(&mut dec, bytes, 13);
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
