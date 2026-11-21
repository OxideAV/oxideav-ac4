//! Round 492 — decoder delay handling: inter-channel alignment on
//! every route, the TS 103 190-1 §5.6 Table 188 frame-alignment delay,
//! and the three [`LatencyMode`]s.
//!
//! The probe is one short burst per channel at a known position. Its
//! energy centroid in the decoded output, minus the one-frame overlap
//! of the transform, is the route's delay.

use oxideav_ac4::decoder::{ALIGNED_TOOL_DELAY, Ac4Decoder, LatencyMode, frame_alignment_delay};
use oxideav_ac4::qmf::QMF_ROUND_TRIP_DELAY;
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Decoder, Frame, Packet,
    SampleFormat,
};

const N: usize = 1920;
const FRAMES: usize = 12;
const T0: usize = N * 6 + 400;

fn burst(ch: usize, len: usize, lfe: bool) -> Vec<f32> {
    let (f, w) = if lfe {
        (90.0f32, 500.0f32)
    } else {
        (700.0 + 150.0 * (ch % 5) as f32, 160.0)
    };
    (0..len)
        .map(|i| {
            let d = (i as f32 - T0 as f32) / w;
            if d.abs() < 4.0 {
                0.5 * (-d * d).exp()
                    * (std::f32::consts::TAU * f * (i as f32 - T0 as f32) / 48_000.0).cos()
            } else {
                0.0
            }
        })
        .collect()
}

fn centroid(x: &[f32]) -> f64 {
    let (mut m, mut e) = (0.0f64, 0.0f64);
    for (i, v) in x.iter().enumerate() {
        let p = (*v as f64).powi(2);
        m += p * i as f64;
        e += p;
    }
    assert!(e > 1e-6, "channel carries the burst");
    m / e
}

fn lfe_inputs(channels: usize) -> &'static [usize] {
    match channels {
        6 | 8 => &[3],
        12 | 14 => &[0],
        24 => &[0, 1],
        _ => &[],
    }
}

fn encode(channels: usize, mode: &str, acpl: &str) -> (CodecParameters, Vec<Packet>) {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    p.options = CodecOptions::new()
        .set("mode", mode)
        .set("acpl", acpl)
        .set("block_switching", "false");
    let mut enc = reg.first_encoder(&p).expect("encoder");
    let len = N * FRAMES;
    let chans: Vec<Vec<f32>> = (0..channels)
        .map(|c| burst(c, len, lfe_inputs(channels).contains(&c)))
        .collect();
    let mut data = Vec::with_capacity(len * channels * 4);
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
    (p, pkts)
}

/// Decode in `mode`; returns per-channel PCM, the first frame's pts
/// and the decoder (for its latency report).
fn decode(
    p: &CodecParameters,
    pkts: &[Packet],
    channels: usize,
    mode: LatencyMode,
) -> (Vec<Vec<f32>>, Option<i64>, Ac4Decoder) {
    let mut dec = Ac4Decoder::new(p);
    dec.set_latency_mode(mode);
    let mut out = vec![Vec::new(); channels];
    let mut first_pts = None;
    let mut push = |af: AudioFrame, out: &mut Vec<Vec<f32>>| {
        if first_pts.is_none() {
            first_pts = Some(af.pts);
        }
        let buf = &af.data[0];
        for i in 0..af.samples as usize {
            for (c, o) in out.iter_mut().enumerate() {
                let off = (i * channels + c) * 2;
                o.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f32 / 32768.0);
            }
        }
    };
    for pk in pkts {
        dec.send_packet(pk).unwrap();
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => push(af, &mut out),
            Err(e) if e.is_need_more() => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    dec.flush().unwrap();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => push(af, &mut out),
            Err(e) if e.is_eof() => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    (out, first_pts.flatten(), dec)
}

/// Routes the framework encoder reaches, with the tool delay each one
/// has: none in the time domain, one QMF round trip on the immersive
/// and 22.2 A-SPX elements, two on the 5.X / 7.X A-CPL elements.
const ROUTES: [(usize, &str, &str, usize); 11] = [
    (1, "waveform", "acpl_2", 0),
    (2, "waveform", "acpl_2", 0),
    (6, "waveform", "acpl_2", 0),
    (6, "parametric", "acpl_1", 2),
    (6, "parametric", "acpl_2", 2),
    (6, "parametric", "acpl_3", 2),
    (8, "waveform", "acpl_2", 0),
    (8, "parametric", "acpl_2", 2),
    (12, "waveform", "acpl_2", 0),
    (12, "parametric", "acpl_2", 1),
    (24, "parametric", "acpl_2", 1),
];

/// Burst-centroid tolerance: exact on waveform channels, a few
/// hundred samples where a decorrelator spreads a parametric channel.
fn tolerance(mode: &str) -> f64 {
    if mode == "waveform" { 8.0 } else { 400.0 }
}

#[test]
fn every_channel_of_a_route_shares_the_reported_tool_delay() {
    for (channels, mode, acpl, passes) in ROUTES {
        let (p, pkts) = encode(channels, mode, acpl);
        let (out, _, dec) = decode(&p, &pkts, channels, LatencyMode::Raw);
        let want = passes * QMF_ROUND_TRIP_DELAY;
        assert_eq!(
            dec.tool_delay_samples() as usize,
            want,
            "{channels}ch {mode} {acpl}: reported tool delay"
        );
        assert_eq!(dec.latency_samples() as usize, want);
        for (c, o) in out.iter().enumerate() {
            let lag = centroid(o) - T0 as f64 - N as f64;
            eprintln!("ROUND-492 latency {channels}ch {mode} {acpl} slot {c}: {lag:.0} samples");
            assert!(
                (lag - want as f64).abs() < tolerance(mode),
                "{channels}ch {mode} {acpl} slot {c}: lag {lag:.0}, route delay {want}"
            );
        }
    }
}

#[test]
fn aligned_mode_gives_every_route_the_same_latency() {
    let (d_pcm, d_ctrl) = frame_alignment_delay(1).unwrap();
    assert_eq!((d_pcm, d_ctrl), (288, 1));
    let target = d_pcm as usize + ALIGNED_TOOL_DELAY;
    for (channels, mode, acpl, _) in ROUTES {
        let (p, pkts) = encode(channels, mode, acpl);
        let (out, _, dec) = decode(&p, &pkts, channels, LatencyMode::Aligned);
        assert_eq!(
            dec.latency_samples() as usize,
            target,
            "{channels}ch {mode}"
        );
        assert_eq!(dec.frame_alignment_delay_samples(), 288);
        assert_eq!(out[0].len(), N * FRAMES, "one frame per packet");
        for (c, o) in out.iter().enumerate() {
            let lag = centroid(o) - T0 as f64 - N as f64;
            assert!(
                (lag - target as f64).abs() < tolerance(mode),
                "{channels}ch {mode} {acpl} slot {c}: aligned lag {lag:.0} vs {target}"
            );
        }
    }
}

#[test]
fn trimmed_mode_removes_the_tool_delay_and_restamps() {
    for (channels, mode, acpl, passes) in ROUTES {
        let (p, pkts) = encode(channels, mode, acpl);
        let (out, first_pts, dec) = decode(&p, &pkts, channels, LatencyMode::Trimmed);
        assert_eq!(dec.latency_samples(), 0);
        assert_eq!(first_pts, Some(0), "timestamps count from the first packet");
        // The stream is shorter by exactly the removed delay.
        assert_eq!(
            out[0].len(),
            N * FRAMES - passes * QMF_ROUND_TRIP_DELAY,
            "{channels}ch {mode} {acpl}: trimmed length"
        );
        for (c, o) in out.iter().enumerate() {
            let lag = centroid(o) - T0 as f64 - N as f64;
            assert!(
                lag.abs() < tolerance(mode),
                "{channels}ch {mode} {acpl} slot {c}: trimmed lag {lag:.0}"
            );
        }
    }
}

#[test]
fn table_188_and_the_decoder_option() {
    // Table 188, by frame_rate_index (Table 83 order).
    let d: Vec<u32> = (0..=13)
        .map(|i| frame_alignment_delay(i).unwrap().0)
        .collect();
    assert_eq!(
        d,
        vec![
            288, 288, 352, 96, 96, 960, 960, 1056, 672, 672, 1312, 864, 864, 352
        ]
    );
    let c: Vec<u32> = (0..=13)
        .map(|i| frame_alignment_delay(i).unwrap().1)
        .collect();
    assert_eq!(c, vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 1]);
    assert!(frame_alignment_delay(14).is_none());
    // The `latency` decoder option selects the mode.
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.options = CodecOptions::new().set("latency", "trimmed");
    assert_eq!(Ac4Decoder::new(&p).latency_mode(), LatencyMode::Trimmed);
    p.options = CodecOptions::new().set("latency", "aligned");
    assert_eq!(Ac4Decoder::new(&p).latency_mode(), LatencyMode::Aligned);
    assert_eq!(
        Ac4Decoder::new(&CodecParameters::audio(CodecId::new("ac4"))).latency_mode(),
        LatencyMode::Raw
    );
}
