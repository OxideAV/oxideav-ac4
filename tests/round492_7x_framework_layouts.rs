//! Round 492 — the 7.0 / 7.1 speaker layouts of TS 103 190-1 Table 88
//! through the framework encoder's `layout_7x` option: 3/4/0 (`back`,
//! Lb / Rb), 5/2/0 (`wide`, Lw / Rw) and 3/2/2 (`top`, Tfl / Tfr), the
//! last two with either `add_ch_base` (`*_surround`). Every stream's
//! TOC must announce the layout, and every channel — the additional
//! pair included — must come back at its own level on the waveform
//! route and on both A-CPL routes.

use oxideav_ac4::toc::{SevenXLayout, parse_ac4_toc};
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Frame, SampleFormat,
};

const N: usize = 1920;
const FRAMES: usize = 8;

/// Distinct low-band multitone per channel (below the ASPX_ACPL_1
/// waveform crossover), each at its own level.
fn channel(ch: usize, len: usize, lfe: bool) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    (0..len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            if lfe {
                0.1 * (tau * 62.0 * t).sin()
            } else {
                let f0 = 150.0 + 47.0 * ch as f32;
                let a = 0.4 - 0.03 * ch as f32;
                a * 0.5 * ((tau * f0 * t).sin() + (tau * (2.3 * f0 + 31.0) * t).sin())
            }
        })
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Encode / decode one configuration; returns the per-input-channel
/// `(input rms, output rms)` over the settled frames.
fn run(channels: usize, layout: &str, mode: &str, acpl: &str) -> (SevenXLayout, Vec<(f32, f32)>) {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    p.options = CodecOptions::new()
        .set("layout_7x", layout)
        .set("mode", mode)
        .set("acpl", acpl)
        .set("framing", "raw");
    let mut enc = reg.first_encoder(&p).expect("encoder");
    let len = N * FRAMES;
    // Framework input order: L R C [LFE] Ls Rs X1 X2.
    let lfe_in = (channels == 8).then_some(3usize);
    let chans: Vec<Vec<f32>> = (0..channels)
        .map(|c| channel(c, len, lfe_in == Some(c)))
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
    assert_eq!(pkts.len(), FRAMES);
    let announced = parse_ac4_toc(&pkts[0].data)
        .expect("toc")
        .seven_x_layout
        .expect("7.X layout in the TOC");
    let mut dec = reg.first_decoder(&p).expect("decoder");
    let mut out = vec![Vec::new(); channels];
    for pk in &pkts {
        dec.send_packet(pk).unwrap();
        let Frame::Audio(af) = dec.receive_frame().expect("decode") else {
            panic!("audio frame expected")
        };
        let buf = &af.data[0];
        for i in 0..af.samples as usize {
            for (c, o) in out.iter_mut().enumerate() {
                let off = (i * channels + c) * 2;
                o.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f32 / 32768.0);
            }
        }
    }
    // Decoder order for 8 channels: L R C Ls Rs X1 X2 LFE.
    let decoder_slot = |in_ch: usize| match (channels, in_ch) {
        (8, 3) => 7,
        (8, c) if c > 3 => c - 1,
        (_, c) => c,
    };
    let settled = N * 4..N * 7;
    let levels = (0..channels)
        .map(|c| {
            (
                rms(&chans[c][settled.clone()]),
                rms(&out[decoder_slot(c)][settled.clone()]),
            )
        })
        .collect();
    (announced, levels)
}

const LAYOUTS: [(&str, SevenXLayout); 5] = [
    ("back", SevenXLayout::BackSurround),
    ("wide", SevenXLayout::Wide { add_ch_base: false }),
    ("wide_surround", SevenXLayout::Wide { add_ch_base: true }),
    ("top", SevenXLayout::TopFront { add_ch_base: false }),
    ("top_surround", SevenXLayout::TopFront { add_ch_base: true }),
];

#[test]
fn every_7x_layout_on_the_waveform_route() {
    for channels in [7usize, 8] {
        for (name, want) in LAYOUTS {
            let (got, levels) = run(channels, name, "waveform", "acpl_2");
            assert_eq!(got, want, "{channels}ch {name}: TOC layout");
            for (c, (i, o)) in levels.iter().enumerate() {
                let r = o / i;
                eprintln!("ROUND-492 7.X {channels}ch {name} waveform ch{c}: {r:.3}x input rms");
                assert!(
                    (0.9..1.1).contains(&r),
                    "{channels}ch {name} waveform ch{c}: {r:.3}x"
                );
            }
        }
    }
}

#[test]
fn every_7x_layout_on_the_acpl_routes() {
    for channels in [7usize, 8] {
        for acpl in ["acpl_1", "acpl_2"] {
            for (name, want) in LAYOUTS {
                let (got, levels) = run(channels, name, "parametric", acpl);
                assert_eq!(got, want, "{channels}ch {name} {acpl}: TOC layout");
                for (c, (i, o)) in levels.iter().enumerate() {
                    let r = o / i;
                    eprintln!("ROUND-492 7.X {channels}ch {name} {acpl} ch{c}: {r:.3}x input rms");
                    assert!(
                        (0.6..1.4).contains(&r),
                        "{channels}ch {name} {acpl} ch{c}: {r:.3}x"
                    );
                }
            }
        }
    }
}

#[test]
fn unknown_layout_names_are_rejected() {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(7);
    p.sample_format = Some(SampleFormat::F32);
    p.options = CodecOptions::new().set("layout_7x", "sideways");
    assert!(reg.first_encoder(&p).is_err());
}
