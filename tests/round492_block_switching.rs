//! Round 492 — transient-switched block lengths (TS 103 190-1 §5.5.3)
//! through the framework encoder and the decoder's block-switched
//! IMDCT (§5.5.2.2 Pseudocodes 63 / 64).
//!
//! The content is a quiet tone with sharp decaying bursts. With
//! `block_switching` the frames whose span holds a burst are coded as
//! short blocks; the quantisation noise a full block smears ahead of
//! the burst ("far" pre-echo: between a quarter and three quarters of
//! a frame before the onset) must drop by a wide margin, at an
//! unchanged overall signal-to-noise ratio and a lower bitrate. Run on
//! mono, stereo, 5.0 and 5.1 and on every frame length family.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Decoder, Frame, SampleFormat,
};

const FRAMES: usize = 28;

fn signal(ch: usize, len: usize, clicks: &[usize]) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let mut v: Vec<f32> = (0..len)
        .map(|i| 0.02 * (tau * (220.0 + 30.0 * ch as f32) * i as f32 / 48_000.0).sin())
        .collect();
    for &c in clicks {
        for k in 0..600.min(len - c) {
            let t = k as f32 / 48_000.0;
            let env = (-(k as f32) / 120.0).exp();
            v[c + k] += 0.5
                * env
                * ((tau * 1_800.0 * t).sin()
                    + 0.7 * (tau * 4_300.0 * t + ch as f32).sin()
                    + 0.5 * (tau * 7_900.0 * t).sin());
        }
    }
    v
}

struct Outcome {
    short_frames: usize,
    far_pre_echo_db: f64,
    snr_db: f64,
    bytes: usize,
}

fn run(channels: u16, fri: u32, n: usize, block_switching: bool) -> Outcome {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let len = n * FRAMES;
    let clicks: Vec<usize> = (0..12)
        .map(|k| 4 * n + k * (3 * n + 137 * (k + 1)))
        .filter(|&c| c + 2 * n < len)
        .collect();
    assert!(clicks.len() >= 4);
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels);
    p.sample_format = Some(SampleFormat::F32);
    p.options = CodecOptions::new()
        .set("frame_rate_index", fri.to_string())
        .set(
            "block_switching",
            if block_switching { "true" } else { "false" },
        );
    let mut enc = reg.first_encoder(&p).expect("encoder");
    let chans: Vec<Vec<f32>> = (0..channels as usize)
        .map(|c| signal(c, len, &clicks))
        .collect();
    let mut data = Vec::with_capacity(len * channels as usize * 4);
    for i in 0..len {
        for c in &chans {
            data.extend_from_slice(&c[i].to_le_bytes());
        }
    }
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: len as u32,
        pts: Some(0),
        data: vec![data],
    }))
    .unwrap();
    enc.flush().unwrap();
    let mut pkts = Vec::new();
    while let Ok(pk) = enc.receive_packet() {
        pkts.push(pk);
    }
    assert_eq!(pkts.len(), FRAMES, "one packet per frame");
    for (i, pk) in pkts.iter().enumerate() {
        assert_eq!(pk.pts, Some((i * n) as i64), "pts continuity");
    }
    let mut dec = Ac4Decoder::new(&p);
    let mut out0 = Vec::with_capacity(len);
    let mut short_frames = 0;
    for pk in &pkts {
        dec.send_packet(pk).unwrap();
        let Frame::Audio(af) = dec.receive_frame().expect("decode") else {
            panic!("audio frame expected")
        };
        assert_eq!(af.samples as usize, n);
        let sub = dec.last_substream.as_ref().expect("substream walked");
        let ti = sub.tools.transform_info_primary.or(sub
            .tools
            .five_channel_data
            .as_ref()
            .and_then(|d| d.transform_info));
        let ti = ti.expect("transform info");
        if ti.windows(n as u32).expect("blocks tile the frame").len() > 1 {
            short_frames += 1;
        }
        // Channel 0 is L (mono: the channel) on every layout here.
        let buf = &af.data[0];
        for i in 0..n {
            let off = i * channels as usize * 2;
            out0.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f64 / 32768.0);
        }
    }
    // The waveform routes delay by one frame (MDCT overlap).
    let err = |i: usize| out0[i + n] - chans[0][i] as f64;
    let far0 = (n * 3 / 4).min(1400);
    let far1 = (n / 4).max(130).min(far0 - 40);
    let (mut e_far, mut e_click) = (0.0f64, 0.0f64);
    for &c in &clicks {
        e_far += (c - far0..c - far1).map(|i| err(i).powi(2)).sum::<f64>() / (far0 - far1) as f64
            * 120.0;
        e_click += (c..c + 600)
            .map(|i| (chans[0][i] as f64).powi(2))
            .sum::<f64>();
    }
    let e_all: f64 = (4 * n..len - 2 * n).map(|i| err(i).powi(2)).sum();
    let s_all: f64 = chans[0][4 * n..len - 2 * n]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum();
    Outcome {
        short_frames,
        far_pre_echo_db: 10.0 * (e_far / e_click).log10(),
        snr_db: 10.0 * (s_all / e_all).log10(),
        bytes: pkts.iter().map(|p| p.data.len()).sum(),
    }
}

fn check(channels: u16, fri: u32, n: usize) {
    let off = run(channels, fri, n, false);
    let on = run(channels, fri, n, true);
    eprintln!(
        "ROUND-492 block switching {channels}ch idx {fri} ({n} samples): short frames {} / {FRAMES}; \
         far pre-echo {:.1} -> {:.1} dB re burst; SNR {:.1} -> {:.1} dB; {} -> {} bytes",
        on.short_frames,
        off.far_pre_echo_db,
        on.far_pre_echo_db,
        off.snr_db,
        on.snr_db,
        off.bytes,
        on.bytes
    );
    assert_eq!(
        off.short_frames, 0,
        "block_switching = false stays on full blocks"
    );
    assert!(
        on.short_frames >= 4,
        "bursts switch to short blocks: {}",
        on.short_frames
    );
    assert!(
        on.short_frames <= FRAMES / 2,
        "steady stretches stay on full blocks: {}",
        on.short_frames
    );
    assert!(
        on.far_pre_echo_db < off.far_pre_echo_db - 6.0,
        "{channels}ch idx {fri}: far pre-echo {:.1} -> {:.1} dB",
        off.far_pre_echo_db,
        on.far_pre_echo_db
    );
    assert!(
        on.snr_db > 22.0,
        "{channels}ch idx {fri}: SNR {:.1} dB",
        on.snr_db
    );
    assert!(on.bytes < off.bytes, "short blocks compact the bursts");
}

#[test]
fn bursts_switch_blocks_on_every_waveform_layout() {
    for channels in [1u16, 2, 5, 6] {
        check(channels, 1, 1920);
    }
}

#[test]
fn bursts_switch_blocks_at_every_frame_length() {
    for (fri, n) in [
        (2u32, 2048usize),
        (3, 1536),
        (6, 960),
        (7, 1024),
        (9, 768),
        (10, 512),
        (12, 384),
    ] {
        check(1, fri, n);
        check(2, fri, n);
    }
}

#[test]
fn switched_streams_are_deterministic() {
    let a = run(6, 1, 1920, true);
    let b = run(6, 1, 1920, true);
    assert_eq!(a.bytes, b.bytes);
    assert_eq!(a.short_frames, b.short_frames);
    assert_eq!(a.snr_db.to_bits(), b.snr_db.to_bits());
}
