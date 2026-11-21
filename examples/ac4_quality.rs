//! Rate / quality table of the AC-4 encoder: achieved bitrate,
//! signal-to-noise ratio and a noise-to-mask ratio per layout and
//! target bitrate, measured through this crate's decoder.
//!
//! ```text
//! cargo run --release --example ac4_quality [seconds]
//! ```
//!
//! The noise-to-mask ratio compares, per scale-factor band of a
//! 1920-line MDCT, the coding-noise energy with a masking threshold:
//! the signal's band energies spread along the Bark scale (27 dB/Bark
//! towards lower, 12 dB/Bark towards higher frequencies) and lowered
//! by 12 dB. Reported is the mean ratio over bands, frames and
//! channels in dB — below 0 dB the noise sits under the modelled
//! threshold on average.

use oxideav_ac4::encoder_mdct::EncoderMdctState;
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Frame, SampleFormat,
};

const N: usize = 1920;

/// Programme-like content: a decaying-harmonic chord sequence, a
/// noise bed and a percussive hit every 300 ms, differently mixed per
/// channel.
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

fn decoder_slot(channels: usize, in_ch: usize) -> usize {
    match channels {
        6 => [0, 1, 2, 5, 3, 4][in_ch],
        8 => [0, 1, 2, 7, 3, 4, 5, 6][in_ch],
        _ => in_ch,
    }
}

fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0) * (f / 7500.0)).atan()
}

/// `(snr_db, nmr_db)` of `out` against `inp` (already aligned).
fn measure(inp: &[f32], out: &[f32]) -> (f64, f64) {
    let sfbo = oxideav_ac4::sfb_offset::sfb_offset_48(N as u32).unwrap();
    let bands = sfbo.len() - 1;
    let z: Vec<f64> = (0..bands)
        .map(|b| bark((sfbo[b] as f64 + sfbo[b + 1] as f64) * 0.5 * 24_000.0 / N as f64))
        .collect();
    let mut a = EncoderMdctState::new(N as u32);
    let mut e = EncoderMdctState::new(N as u32);
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    let (mut ratio_sum, mut ratio_n) = (0.0f64, 0usize);
    let frames = inp.len().min(out.len()) / N;
    for f in 0..frames {
        let x = &inp[f * N..(f + 1) * N];
        let d: Vec<f32> = x
            .iter()
            .zip(&out[f * N..(f + 1) * N])
            .map(|(a, b)| b - a)
            .collect();
        sig += x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        err += d.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        let xs = a.analyse_frame(x);
        let ds = e.analyse_frame(&d);
        let energy = |s: &[f32], b: usize| -> f64 {
            s[sfbo[b] as usize..sfbo[b + 1] as usize]
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum()
        };
        let es: Vec<f64> = (0..bands).map(|b| energy(&xs, b)).collect();
        let total: f64 = es.iter().sum();
        if f < 2 || total < 1e-3 {
            continue;
        }
        for b in 0..bands {
            let mut mask = 0.0f64;
            for (m, &em) in es.iter().enumerate() {
                let dz = z[b] - z[m];
                let slope = if dz < 0.0 { 27.0 } else { 12.0 };
                mask += em * 10f64.powf(-slope * dz.abs() / 10.0);
            }
            mask *= 10f64.powf(-1.2);
            if mask > 1e-9 {
                ratio_sum += energy(&ds, b) / mask;
                ratio_n += 1;
            }
        }
    }
    (
        10.0 * (sig / err.max(1e-30)).log10(),
        10.0 * (ratio_sum / ratio_n.max(1) as f64).max(1e-30).log10(),
    )
}

fn run(reg: &CodecRegistry, channels: usize, len: usize, bitrate: u32, extra: &[(&str, &str)]) {
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    let mut o = CodecOptions::new().set("bitrate", bitrate.to_string());
    for (k, v) in extra {
        o = o.set(*k, *v);
    }
    p.options = o;
    let lfe_in: &[usize] = match channels {
        6 | 8 => &[3],
        12 => &[0],
        _ => &[],
    };
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
    let sizes: Vec<usize> = pkts.iter().map(|p| p.data.len()).collect();
    let bytes: usize = sizes.iter().sum();
    let mut dec = reg.first_decoder(&p).expect("decoder");
    let mut out = vec![Vec::new(); channels];
    for pk in &pkts {
        dec.send_packet(pk).unwrap();
        if let Ok(Frame::Audio(af)) = dec.receive_frame() {
            let buf = &af.data[0];
            for i in 0..af.samples as usize {
                for (c, o) in out.iter_mut().enumerate() {
                    let off = (i * channels + c) * 2;
                    o.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f32 / 32768.0);
                }
            }
        }
    }
    let (mut snr, mut nmr, mut n) = (0.0f64, 0.0f64, 0usize);
    for c in 0..channels {
        if lfe_in.contains(&c) {
            continue;
        }
        let o = &out[decoder_slot(channels, c)];
        // One frame of transform overlap on the waveform routes.
        let (s, m) = measure(&chans[c][..len - N], &o[N..]);
        snr += s;
        nmr += m;
        n += 1;
    }
    let dur = len as f64 / 48_000.0;
    let peak = sizes.iter().copied().max().unwrap_or(0);
    println!(
        "{channels:2} ch  target {:>7}  actual {:7.1} kbit/s  peak frame {:6.1} kbit/s  SNR {:5.1} dB  NMR {:6.1} dB",
        if bitrate == 0 {
            "-".to_string()
        } else {
            format!("{}k", bitrate / 1000)
        },
        bytes as f64 * 8.0 / dur / 1000.0,
        peak as f64 * 8.0 / (N as f64 / 48_000.0) / 1000.0,
        snr / n as f64,
        nmr / n as f64,
    );
}

fn main() {
    let seconds: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);
    let len = ((seconds * 48_000.0) as usize / N).max(8) * N;
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    println!("constant quantiser (no rate control), block switching off / on:");
    for channels in [2usize, 6, 12] {
        run(&reg, channels, len, 0, &[("block_switching", "false")]);
        run(&reg, channels, len, 0, &[]);
    }
    println!("rate controlled:");
    for (channels, rates) in [
        (2usize, &[64_000u32, 96_000, 128_000, 192_000][..]),
        (6, &[192_000, 256_000, 384_000][..]),
        (12, &[384_000, 512_000, 768_000][..]),
    ] {
        for &r in rates {
            run(&reg, channels, len, r, &[]);
        }
    }
    println!("rate controlled, parametric (waveform SNR / NMR do not apply to regenerated bands):");
    for (channels, r) in [
        (6usize, 128_000u32),
        (6, 192_000),
        (12, 256_000),
        (12, 384_000),
    ] {
        run(&reg, channels, len, r, &[("mode", "parametric")]);
    }
}
