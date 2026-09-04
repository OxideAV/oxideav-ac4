//! Round 456 — the `frame_rate_index` matrix through the framework
//! encoder (TS 103 190-1 §4.3.3.2 Tables 83 / 84): every long-frame
//! index (48 kHz 0..=4 and 13 — 1920 / 2048 / 1536 samples — plus the
//! 44,1 kHz index 13) on the waveform layouts and the parametric
//! immersive route. Each configuration must produce frames whose TOC
//! announces the requested indices, decode to exactly the Table 83 /
//! 84 frame length, and land within the waveform parity floor on the
//! settled frame. The short-frame indices 5..=12 (< 1536 samples) are
//! rejected at construction — their `asf_transform_info()` carries the
//! Table 103 `transf_length` code instead of `b_long_frame` (§4.3.6.1),
//! which the body writers do not emit.
//!
//! Before this round the TOC writers emitted one unconditional
//! `frame_rate_multiply_info()` bit and no `frame_rate_fractions_info()`
//! (wrong for indices 5, 6, 7, 8, 9, 13), the immersive / 22.2
//! raw-frame writer hard-coded 48 kHz / index 1, and the immersive LFE
//! `max_sfb` ignored the Table 106 field width.

use oxideav_ac4::toc::{frame_rate_entry, parse_ac4_toc};
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Frame, SampleFormat,
};

const FRAMES: usize = 5;

fn registry() -> CodecRegistry {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    reg
}

fn signal(ch: usize, len: usize, rate: u32, lfe: bool) -> Vec<f32> {
    let (f1, f2) = if lfe {
        (40.0, 75.0)
    } else {
        (110.0 + 37.0 * ch as f32, 700.0 + 61.0 * ch as f32)
    };
    (0..len)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.25 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * f2 * t + 0.3).sin()
        })
        .collect()
}

fn lfe_slots(channels: usize) -> &'static [usize] {
    match channels {
        6 | 8 => &[3],
        12 | 14 => &[0],
        24 => &[0, 1],
        _ => &[],
    }
}

/// Input channel index → decoder output slot.
fn decoder_slot(channels: usize, in_ch: usize) -> usize {
    match channels {
        6 => [0, 1, 2, 5, 3, 4][in_ch],
        8 => [0, 1, 2, 7, 3, 4, 5, 6][in_ch],
        _ => in_ch,
    }
}

/// Run one configuration; returns the worst settled relative RMS
/// error over the channels.
fn run(channels: usize, rate: u32, fri: u32, mode: &str) -> f64 {
    let reg = registry();
    let fs_index = if rate == 48_000 { 1 } else { 0 };
    let (_, n) = frame_rate_entry(fri, fs_index);
    let n = n as usize;
    assert!(n > 0, "index {fri} at {rate} Hz is reserved");
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(rate);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    p.options = CodecOptions::new()
        .set("frame_rate_index", fri.to_string())
        .set("framing", "raw")
        .set("mode", mode);
    let mut enc = reg.first_encoder(&p).expect("encoder");
    let total = n * FRAMES;
    let chans: Vec<Vec<f32>> = (0..channels)
        .map(|c| signal(c, total, rate, lfe_slots(channels).contains(&c)))
        .collect();
    let mut data = Vec::with_capacity(total * channels * 4);
    for i in 0..total {
        for c in &chans {
            data.extend_from_slice(&c[i].to_le_bytes());
        }
    }
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: total as u32,
        pts: Some(0),
        data: vec![data],
    }))
    .unwrap();
    enc.flush().unwrap();
    let mut pkts = Vec::new();
    while let Ok(pk) = enc.receive_packet() {
        pkts.push(pk);
    }
    assert_eq!(
        pkts.len(),
        FRAMES,
        "{channels}ch idx {fri}: one packet per frame"
    );
    let mut dec = reg.first_decoder(&p).expect("decoder");
    let mut out = vec![Vec::new(); channels];
    for pk in &pkts {
        assert_eq!(pk.duration, Some(n as i64));
        let info = parse_ac4_toc(&pk.data).expect("toc");
        assert_eq!(info.fs_index, fs_index);
        assert_eq!(info.frame_rate_index, fri);
        assert_eq!(info.frame_length as usize, n);
        assert_eq!(info.sample_rate, rate);
        dec.send_packet(pk).unwrap();
        let Frame::Audio(af) = dec.receive_frame().expect("decode") else {
            panic!()
        };
        assert_eq!(
            af.samples as usize, n,
            "{channels}ch idx {fri}: frame length"
        );
        let buf = &af.data[0];
        assert_eq!(buf.len(), n * channels * 2);
        for i in 0..n {
            for (c, ch) in out.iter_mut().enumerate() {
                let off = (i * channels + c) * 2;
                ch.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f32 / 32768.0);
            }
        }
    }
    // Settled input frame FRAMES-3 lands on output frame FRAMES-2 plus
    // the route lag (0 on the waveform routes, 577 on the QMF routes).
    let start = n * (FRAMES - 3);
    let mut worst = 0.0f64;
    for c in 0..channels {
        let reference = &chans[c][start..start + n];
        let o = &out[decoder_slot(channels, c)];
        let den: f64 = reference
            .iter()
            .map(|a| (*a as f64).powi(2))
            .sum::<f64>()
            .max(1e-12);
        let mut best = f64::INFINITY;
        for lag in [0usize, 577] {
            let base = start + n + lag;
            if base + n > o.len() {
                continue;
            }
            let num: f64 = (0..n)
                .map(|i| (reference[i] as f64 - o[base + i] as f64).powi(2))
                .sum();
            best = best.min((num / den).sqrt());
        }
        worst = worst.max(best);
    }
    eprintln!(
        "ROUND-456 frame-rate matrix {channels}ch {mode} {rate} Hz idx {fri} ({n} samples): worst settled rel err {worst:.4}, {} bytes/frame",
        pkts[2].data.len()
    );
    worst
}

const LONG_FRAME_INDICES_48K: [u32; 6] = [0, 1, 2, 3, 4, 13];

#[test]
fn every_long_frame_48k_index_on_mono_stereo_5_1() {
    for fri in LONG_FRAME_INDICES_48K {
        for ch in [1usize, 2, 6] {
            let e = run(ch, 48_000, fri, "waveform");
            assert!(e < 0.12, "{ch}ch idx {fri}: settled rel err {e:.4}");
        }
    }
}

#[test]
fn every_long_frame_48k_index_on_the_immersive_waveform_route() {
    for fri in LONG_FRAME_INDICES_48K {
        let e = run(12, 48_000, fri, "waveform");
        assert!(e < 0.12, "12ch idx {fri}: settled rel err {e:.4}");
    }
}

#[test]
fn selected_indices_on_7_1_22_2_and_the_parametric_immersive_route() {
    for fri in [2u32, 3, 13] {
        for (ch, mode) in [(8usize, "waveform"), (24, "waveform"), (11, "parametric")] {
            let e = run(ch, 48_000, fri, mode);
            assert!(e < 0.12, "{ch}ch {mode} idx {fri}: settled rel err {e:.4}");
        }
    }
}

#[test]
fn short_frame_indices_are_rejected_at_construction() {
    let reg = registry();
    for fri in 5..=12u32 {
        let mut p = CodecParameters::audio(CodecId::new("ac4"));
        p.sample_rate = Some(48_000);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::F32);
        p.options = CodecOptions::new().set("frame_rate_index", fri.to_string());
        assert!(
            reg.first_encoder(&p).is_err(),
            "index {fri} must be rejected"
        );
    }
}

#[test]
fn the_44_1_khz_frame_rate_index_13() {
    for (ch, mode) in [
        (1usize, "waveform"),
        (2, "waveform"),
        (6, "waveform"),
        (12, "waveform"),
    ] {
        let e = run(ch, 44_100, 13, mode);
        assert!(
            e < 0.12,
            "{ch}ch {mode} 44.1 kHz idx 13: settled rel err {e:.4}"
        );
    }
}
