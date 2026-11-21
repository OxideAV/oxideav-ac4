//! Round 492 — rate control of the framework encoder (`bitrate`
//! option): a noise-allocation level searched per frame so the stream
//! averages the target with a one-frame bit reservoir, announced as an
//! average-bit-rate stream through `wait_frames` (TS 103 190-1
//! §4.3.3.2.4 Table 81).
//!
//! Checked per layout and bitrate: the achieved rate, the reservoir
//! bound on every prefix of the stream (a constant-rate channel plus
//! the one frame the decoder waits never runs dry), the TOC
//! announcement, and that quality rises with the rate — at 96 kbit/s
//! a stereo stream reaches the signal-to-noise ratio the constant
//! quantiser needs more than 300 kbit/s for.

use oxideav_ac4::toc::parse_ac4_toc;
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Frame, SampleFormat,
};

const N: usize = 1920;
const FRAMES: usize = 50;

/// Programme-like content: decaying harmonic notes, a noise bed and a
/// percussive hit every 300 ms, mixed differently per channel.
fn signal(ch: usize, len: usize, lfe: bool) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let mut seed = 0x1234_5678u32 ^ (ch as u32).wrapping_mul(0x9E37_79B9);
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let notes = [196.0f32, 246.9, 293.7, 370.0, 440.0, 329.6];
    let mut lp = 0.0f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            if lfe {
                return 0.2 * (tau * 55.0 * t).sin() * (0.5 + 0.5 * (tau * 1.5 * t).sin());
            }
            let note = notes[((t * 2.0) as usize + ch) % notes.len()];
            let phase = (t * 2.0).fract() / 2.0;
            let env = (-phase * 5.0).exp();
            let mut v = 0.0f32;
            for h in 1..=10 {
                v += env * 0.2 / h as f32 * (tau * note * h as f32 * t + (ch * h) as f32).sin();
            }
            lp += 0.15 * (noise() - lp);
            v += 0.03 * lp + 0.004 * noise();
            let hit = (t * 3.33 + 0.1 * ch as f32).fract() / 3.33;
            if hit < 0.02 {
                v += 0.35 * (-hit * 300.0).exp() * (tau * 3_000.0 * hit).sin();
            }
            v
        })
        .collect()
}

struct Outcome {
    sizes: Vec<usize>,
    /// Waveform SNR of the first full-band channel (dB).
    snr_db: f64,
    wait_frames: Vec<Option<u32>>,
}

fn run(channels: usize, bitrate: u32, mode: &str) -> Outcome {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    p.options = CodecOptions::new()
        .set("bitrate", bitrate.to_string())
        .set("mode", mode)
        .set("framing", "raw");
    let lfe_in: &[usize] = match channels {
        6 => &[3],
        12 => &[0],
        _ => &[],
    };
    let len = N * FRAMES;
    let chans: Vec<Vec<f32>> = (0..channels)
        .map(|c| signal(c, len, lfe_in.contains(&c)))
        .collect();
    let mut enc = reg.first_encoder(&p).expect("encoder");
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
    assert_eq!(pkts.len(), FRAMES);
    let wait_frames = pkts
        .iter()
        .map(|pk| parse_ac4_toc(&pk.data).expect("toc").wait_frames)
        .collect();
    let mut dec = reg.first_decoder(&p).expect("decoder");
    // First full-band channel: input 1 on the LFE-first immersive
    // layout, input 0 elsewhere; same slot on the decoder side.
    let probe = if channels == 12 { 1 } else { 0 };
    let mut out = Vec::with_capacity(len);
    for pk in &pkts {
        dec.send_packet(pk).unwrap();
        let Frame::Audio(af) = dec.receive_frame().expect("decode") else {
            panic!("audio frame expected")
        };
        let buf = &af.data[0];
        for i in 0..af.samples as usize {
            let off = (i * channels + probe) * 2;
            out.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f64 / 32768.0);
        }
    }
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for i in 4 * N..len - 2 * N {
        let x = chans[probe][i] as f64;
        sig += x * x;
        err += (out[i + N] - x).powi(2);
    }
    Outcome {
        sizes: pkts.iter().map(|p| p.data.len()).collect(),
        snr_db: 10.0 * (sig / err.max(1e-30)).log10(),
        wait_frames,
    }
}

/// Achieved rate and the reservoir bound of a rate-controlled run.
fn check_rate(o: &Outcome, bitrate: u32, what: &str) -> f64 {
    let budget = bitrate as f64 * N as f64 / 48_000.0;
    let total: usize = o.sizes.iter().sum();
    let actual = total as f64 * 8.0 / (FRAMES as f64 * N as f64 / 48_000.0);
    eprintln!(
        "ROUND-492 rate control {what}: target {bitrate} bit/s, actual {actual:.0} bit/s, \
         largest frame {:.0} bits (budget {budget:.0}), SNR {:.1} dB",
        o.sizes.iter().copied().max().unwrap() as f64 * 8.0,
        o.snr_db
    );
    assert!(
        (actual / bitrate as f64 - 1.0).abs() < 0.03,
        "{what}: {actual:.0} bit/s for a {bitrate} bit/s target"
    );
    // Reservoir: after any number of frames the stream has used at
    // most its budget so far plus nothing (the reservoir only holds
    // savings), and no frame exceeds its budget plus one frame's
    // worth of savings.
    let mut used = 0.0f64;
    for (k, &s) in o.sizes.iter().enumerate() {
        used += s as f64 * 8.0;
        assert!(
            used <= (k + 1) as f64 * budget + 1.0,
            "{what}: frame {k} overdraws the budget ({used:.0} bits)"
        );
        assert!(
            s as f64 * 8.0 <= 2.0 * budget + 1.0,
            "{what}: frame {k} of {} bits exceeds budget + reservoir",
            s * 8
        );
    }
    // Announced as average bit rate, decoder waits one frame.
    assert!(
        o.wait_frames.iter().all(|w| *w == Some(2)),
        "{what}: wait_frames announcement"
    );
    actual
}

#[test]
fn stereo_meets_its_targets_and_quality_follows_the_rate() {
    let mut last = f64::MIN;
    let mut at_96k = 0.0;
    for bitrate in [48_000u32, 64_000, 96_000, 128_000, 192_000] {
        let o = run(2, bitrate, "waveform");
        check_rate(&o, bitrate, "stereo");
        assert!(
            o.snr_db > last + 0.5,
            "stereo {bitrate}: SNR {:.1} dB after {last:.1} dB",
            o.snr_db
        );
        last = o.snr_db;
        if bitrate == 96_000 {
            at_96k = o.snr_db;
        }
    }
    // The constant quantiser (no rate control) for comparison.
    let free = run(2, 0, "waveform");
    let free_rate =
        free.sizes.iter().sum::<usize>() as f64 * 8.0 / (FRAMES as f64 * N as f64 / 48_000.0);
    eprintln!(
        "ROUND-492 rate control stereo constant quantiser: {free_rate:.0} bit/s, SNR {:.1} dB",
        free.snr_db
    );
    assert!(free.wait_frames.iter().all(|w| w.is_none()));
    assert!(free_rate > 250_000.0);
    assert!(
        at_96k > free.snr_db - 1.5,
        "96 kbit/s rate-controlled ({at_96k:.1} dB) vs constant quantiser at \
         {free_rate:.0} bit/s ({:.1} dB)",
        free.snr_db
    );
}

#[test]
fn multichannel_layouts_meet_their_targets() {
    for (channels, bitrate, floor_db) in [(6usize, 256_000u32, 22.0), (12, 512_000, 20.0)] {
        let o = run(channels, bitrate, "waveform");
        check_rate(&o, bitrate, &format!("{channels}ch waveform"));
        assert!(
            o.snr_db > floor_db,
            "{channels}ch at {bitrate}: SNR {:.1} dB",
            o.snr_db
        );
    }
}

#[test]
fn parametric_routes_meet_their_targets() {
    for (channels, bitrate) in [(6usize, 160_000u32), (12, 320_000)] {
        let o = run(channels, bitrate, "parametric");
        check_rate(&o, bitrate, &format!("{channels}ch parametric"));
    }
}

#[test]
fn rate_controlled_streams_are_deterministic() {
    let a = run(2, 96_000, "waveform");
    let b = run(2, 96_000, "waveform");
    assert_eq!(a.sizes, b.sizes);
}
