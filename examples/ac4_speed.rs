//! Encode / decode throughput of the AC-4 codec in multiples of
//! realtime, per layout and coding-tool family.
//!
//! ```text
//! cargo run --release --example ac4_speed [seconds] [direct]
//! ```
//!
//! `direct` routes the MDCT / IMDCT through the O(N²) reference
//! summations (the pre-FFT behaviour) and reports whether the decoded
//! S16 output of both transform forms is byte-identical.

use std::time::Instant;

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Frame, Packet, SampleFormat,
};

fn signal(ch: usize, len: usize) -> Vec<f32> {
    // Harmonic stack + a click train + a little noise: busy enough to
    // exercise every scale-factor band and the entropy coder.
    let mut seed = 0x9E37_79B9u32 ^ (ch as u32).wrapping_mul(0x85EB_CA6B);
    (0..len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            let mut v = 0.0f32;
            for h in 1..=12 {
                let f = (110.0 + 23.0 * ch as f32) * h as f32;
                v += 0.25 / h as f32 * (std::f32::consts::TAU * f * t + h as f32).sin();
            }
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v += ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02;
            if i % 12_000 == 700 * (ch + 1) {
                v += 0.6;
            }
            v
        })
        .collect()
}

fn params(channels: u16, opts: CodecOptions) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels);
    p.sample_format = Some(SampleFormat::F32);
    p.options = opts;
    p
}

fn encode(reg: &CodecRegistry, chans: &[Vec<f32>], mode: &str) -> (Vec<Packet>, f64) {
    let p = params(chans.len() as u16, CodecOptions::new().set("mode", mode));
    let mut enc = reg.first_encoder(&p).expect("encoder");
    let total = chans[0].len();
    let mut pkts = Vec::new();
    let t0 = Instant::now();
    let mut pos = 0;
    while pos < total {
        let n = 1920.min(total - pos);
        let mut data = Vec::with_capacity(n * chans.len() * 4);
        for i in pos..pos + n {
            for c in chans {
                data.extend_from_slice(&c[i].to_le_bytes());
            }
        }
        enc.send_frame(&Frame::Audio(AudioFrame {
            samples: n as u32,
            pts: Some(pos as i64),
            data: vec![data],
        }))
        .expect("send_frame");
        pos += n;
        while let Ok(p) = enc.receive_packet() {
            pkts.push(p);
        }
    }
    enc.flush().expect("flush");
    while let Ok(p) = enc.receive_packet() {
        pkts.push(p);
    }
    (pkts, t0.elapsed().as_secs_f64())
}

fn decode(reg: &CodecRegistry, channels: u16, pkts: &[Packet]) -> (Vec<u8>, f64) {
    let p = params(channels, CodecOptions::new());
    let mut dec = reg.first_decoder(&p).expect("decoder");
    let mut out = Vec::new();
    let t0 = Instant::now();
    for p in pkts {
        dec.send_packet(p).expect("send_packet");
        if let Ok(Frame::Audio(af)) = dec.receive_frame() {
            out.extend_from_slice(&af.data[0]);
        }
    }
    (out, t0.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seconds: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let direct = args.iter().any(|a| a == "direct");
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let len = ((seconds * 48_000.0) as usize / 1920).max(4) * 1920;
    let dur = len as f64 / 48_000.0;
    println!("layout mode        kbit/s  enc xRT  dec xRT  identical  pcm hash");
    for &(name, nch) in &[("stereo", 2u16), ("5.1", 6), ("7.1.4", 12)] {
        let chans: Vec<Vec<f32>> = (0..nch as usize).map(|c| signal(c, len)).collect();
        for mode in ["waveform", "parametric"] {
            oxideav_ac4::mdct::set_direct_transforms(direct);
            let (pkts, te) = encode(&reg, &chans, mode);
            let (pcm, td) = decode(&reg, nch, &pkts);
            // Cross-check the other transform form on the same packets.
            oxideav_ac4::mdct::set_direct_transforms(!direct);
            let (pcm_other, _) = decode(&reg, nch, &pkts);
            oxideav_ac4::mdct::set_direct_transforms(false);
            let bytes: usize = pkts.iter().map(|p| p.data.len()).sum();
            // FNV-1a over the decoded S16 bytes: equal hashes across
            // builds mean bit-identical decoder output.
            let hash = pcm.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
            });
            println!(
                "{name:6} {mode:10} {:8.1} {:8.2} {:8.2}  {:9}  {hash:016x}",
                bytes as f64 * 8.0 / dur / 1000.0,
                dur / te,
                dur / td,
                pcm == pcm_other
            );
        }
    }
}
