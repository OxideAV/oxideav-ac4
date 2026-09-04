//! Round 456 — the framework `oxideav_core::Encoder` face of the AC-4
//! encoder (`oxideav_ac4::encoder`): registry-resolved encode → packets
//! → registry-resolved decode, on every supported channel layout in
//! both coding-tool families, with the PCM parity measured per channel
//! against the encoder's own input.
//!
//! Also pinned: FIFO framing with input frames that don't line up with
//! the AC-4 frame length (+ zero-padded flush), pts / duration
//! continuity, the three packet framings (0xAC40 / 0xAC41 / raw), and
//! the S16-interleaved input conversion.

use oxideav_ac4::encoder::{Ac4Encoder, Ac4EncoderOptions, EncodeMode, Framing};
use oxideav_ac4::sync::parse_sync_frame_at_start;
use oxideav_core::{
    AudioFrame, ChannelLayout, CodecId, CodecOptions, CodecParameters, CodecRegistry, Decoder,
    Encoder, Frame, SampleFormat,
};

const N: usize = 1920;
const FRAMES: usize = 6;

fn registry() -> CodecRegistry {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    reg
}

/// Deterministic per-channel test signal: three harmonically unrelated
/// tones per channel (distinct per channel) at moderate level.
fn signal(ch: usize, len: usize) -> Vec<f32> {
    let f1 = 110.0 + 37.0 * ch as f32;
    let f2 = 440.0 + 61.0 * ch as f32;
    let f3 = 1_300.0 + 97.0 * ch as f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.22 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                + 0.14 * (2.0 * std::f32::consts::PI * f2 * t + 0.3).sin()
                + 0.07 * (2.0 * std::f32::consts::PI * f3 * t + 1.1).sin()
        })
        .collect()
}

fn f32_interleaved(chans: &[Vec<f32>], start: usize, n: usize) -> AudioFrame {
    let nch = chans.len();
    let mut data = Vec::with_capacity(n * nch * 4);
    for i in start..start + n {
        for c in chans {
            data.extend_from_slice(&c[i].to_le_bytes());
        }
    }
    AudioFrame {
        samples: n as u32,
        pts: Some(start as i64),
        data: vec![data],
    }
}

fn params(channels: u16, fmt: SampleFormat, opts: CodecOptions) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels);
    p.sample_format = Some(fmt);
    p.options = opts;
    p
}

/// Encode `chans` through the registry encoder in `chunk`-sample input
/// frames, flush, and return the packets.
fn encode_all(
    enc: &mut dyn Encoder,
    chans: &[Vec<f32>],
    chunk: usize,
) -> Vec<oxideav_core::Packet> {
    let total = chans[0].len();
    let mut pkts = Vec::new();
    let mut pos = 0;
    while pos < total {
        let n = chunk.min(total - pos);
        enc.send_frame(&Frame::Audio(f32_interleaved(chans, pos, n)))
            .expect("send_frame");
        pos += n;
        while let Ok(p) = enc.receive_packet() {
            pkts.push(p);
        }
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => pkts.push(p),
            Err(e) if e.is_eof() => break,
            Err(e) => panic!("receive_packet after flush: {e:?}"),
        }
    }
    pkts
}

/// Decode every packet; returns the concatenated per-channel output
/// as floats.
fn decode_all(
    dec: &mut dyn Decoder,
    pkts: &[oxideav_core::Packet],
    channels: usize,
) -> Vec<Vec<f32>> {
    let mut out = vec![Vec::new(); channels];
    for p in pkts {
        dec.send_packet(p).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("audio frame expected")
        };
        let n = af.samples as usize;
        let buf = &af.data[0];
        assert_eq!(buf.len(), n * channels * 2, "S16 interleaved size");
        for i in 0..n {
            for (c, ch) in out.iter_mut().enumerate() {
                let off = (i * channels + c) * 2;
                ch.push(i16::from_le_bytes([buf[off], buf[off + 1]]) as f32 / 32768.0);
            }
        }
    }
    out
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

/// Codec latency: the MDCT overlap makes frame `k`'s output carry
/// input frame `k − 1`, and the QMF-domain routes add a few hundred
/// samples on top (577 on the immersive A-SPX routes). `settled_err`
/// compares input frame `SETTLED` against the decoder output one frame
/// later at the best lag in `0..=MAX_ROUTE_LAG`.
const MAX_ROUTE_LAG: usize = N;
const SETTLED: usize = FRAMES - 3;

/// Relative RMS error of a settled output frame against the input
/// frame it carries, minimised over the route lag; returns `(err,
/// lag)`.
fn settled_err(input: &[f32], output: &[f32]) -> (f64, usize) {
    let start = N * SETTLED;
    let reference = &input[start..start + N];
    let denom = rms(reference).max(1e-9);
    let mut best = (f64::INFINITY, 0usize);
    for lag in 0..=MAX_ROUTE_LAG {
        let base = start + N + lag;
        if base + N > output.len() {
            break;
        }
        let e: f64 = (0..N)
            .map(|i| (reference[i] as f64 - output[base + i] as f64).powi(2))
            .sum::<f64>();
        let r = (e / N as f64).sqrt() / denom;
        if r < best.0 {
            best = (r, lag);
        }
    }
    best
}

/// Plain relative RMS error, no lag search.
fn rel_err(reference: &[f32], got: &[f32]) -> f64 {
    let denom = rms(reference).max(1e-9);
    let n = reference.len().min(got.len());
    let e: f64 = (0..n)
        .map(|i| (reference[i] as f64 - got[i] as f64).powi(2))
        .sum::<f64>();
    (e / n.max(1) as f64).sqrt() / denom
}

/// Test signal set for `channels`: independent tones on every
/// fullband channel, low-frequency content on the LFE slots.
fn signals(channels: usize) -> Vec<Vec<f32>> {
    let len = N * FRAMES;
    let mut chans: Vec<Vec<f32>> = (0..channels).map(|c| signal(c, len)).collect();
    // LFE slots carry low-frequency content only (the LFE element codes
    // a handful of scale-factor bands).
    let lfe_slots: &[usize] = match channels {
        6 | 8 => &[3],
        12 | 14 => &[0],
        24 => &[0, 1],
        _ => &[],
    };
    for (k, &slot) in lfe_slots.iter().enumerate() {
        chans[slot] = (0..len)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                0.3 * (2.0 * std::f32::consts::PI * (40.0 + 9.0 * k as f32) * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * (75.0 + 7.0 * k as f32) * t + 0.4).sin()
            })
            .collect();
    }
    chans
}

/// Input channel index → decoder output slot for `channels` (see the
/// `encoder` module docs: core layouts for ≤ 8 channels, decoder slot
/// order above).
fn decoder_slot(channels: usize, in_ch: usize) -> usize {
    match channels {
        6 => [0, 1, 2, 5, 3, 4][in_ch],
        8 => [0, 1, 2, 7, 3, 4, 5, 6][in_ch],
        _ => in_ch,
    }
}

fn run_layout(channels: usize, mode: EncodeMode, chunk: usize) -> Vec<f64> {
    let reg = registry();
    let mode_s = match mode {
        EncodeMode::Waveform => "waveform",
        EncodeMode::Parametric => "parametric",
    };
    let opts = CodecOptions::new().set("mode", mode_s);
    let p = params(channels as u16, SampleFormat::F32, opts);
    let mut enc = reg.first_encoder(&p).expect("registry encoder");
    assert_eq!(enc.output_params().channels, Some(channels as u16));
    let chans = signals(channels);
    let pkts = encode_all(enc.as_mut(), &chans, chunk);
    assert_eq!(pkts.len(), FRAMES, "one packet per AC-4 frame");
    for (k, p) in pkts.iter().enumerate() {
        assert_eq!(p.pts, Some((k * N) as i64));
        assert_eq!(p.duration, Some(N as i64));
        assert!(p.flags.keyframe, "all-I by default");
        let sf = parse_sync_frame_at_start(&p.data).expect("0xAC40 sync frame");
        assert_eq!(
            sf.total_len,
            p.data.len(),
            "packet is exactly one sync frame"
        );
    }
    let mut dec = reg.first_decoder(&p).expect("registry decoder");
    let out = decode_all(dec.as_mut(), &pkts, channels);
    assert_eq!(out[0].len(), N * FRAMES);
    let mut errs = Vec::with_capacity(channels);
    let mut lags = Vec::with_capacity(channels);
    for c in 0..channels {
        let (e, lag) = settled_err(&chans[c], &out[decoder_slot(channels, c)]);
        errs.push(e);
        lags.push(lag);
    }
    let worst = errs.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "ROUND-456 framework {channels}ch {mode_s}: per-channel settled rel RMS err {errs:.3?} \
         (worst {worst:.4}; route lag beyond one frame {lags:?})"
    );
    errs
}

#[test]
fn registry_exposes_encoder_and_schema() {
    let reg = registry();
    let id = CodecId::new("ac4");
    assert!(reg.has_encoder(&id));
    let schema = reg.encoder_options_schema(&id).expect("schema");
    for key in ["frame_rate_index", "framing", "mode", "bandwidth", "gop"] {
        assert!(schema.iter().any(|f| f.name == key), "schema lacks {key}");
    }
}

#[test]
fn waveform_mono_stereo_round_trip() {
    for ch in [1usize, 2] {
        let errs = run_layout(ch, EncodeMode::Waveform, 1000);
        for (c, e) in errs.iter().enumerate() {
            assert!(*e < 0.10, "{ch}ch waveform ch{c}: rel err {e:.4}");
        }
    }
}

#[test]
fn waveform_5_x_7_x_round_trip() {
    for ch in [5usize, 6, 7, 8] {
        let errs = run_layout(ch, EncodeMode::Waveform, 1000);
        for (c, e) in errs.iter().enumerate() {
            assert!(*e < 0.10, "{ch}ch waveform ch{c}: rel err {e:.4}");
        }
    }
}

#[test]
fn waveform_immersive_round_trip() {
    for ch in [11usize, 12, 13, 14] {
        let errs = run_layout(ch, EncodeMode::Waveform, 1920);
        for (c, e) in errs.iter().enumerate() {
            assert!(*e < 0.10, "{ch}ch waveform ch{c}: rel err {e:.4}");
        }
    }
}

#[test]
fn waveform_22_2_round_trip() {
    let errs = run_layout(24, EncodeMode::Waveform, 1920);
    for (c, e) in errs.iter().enumerate() {
        assert!(*e < 0.10, "22.2 waveform ch{c}: rel err {e:.4}");
    }
}

#[test]
fn parametric_5_x_7_x_is_rejected_until_parity_is_pinned() {
    let reg = registry();
    for ch in [5u16, 6, 7, 8] {
        let p = params(
            ch,
            SampleFormat::F32,
            CodecOptions::new().set("mode", "parametric"),
        );
        assert!(
            reg.first_encoder(&p).is_err(),
            "{ch}ch parametric must be rejected"
        );
    }
}

#[test]
fn parametric_immersive_round_trip() {
    for ch in [11usize, 12, 13, 14, 24] {
        let errs = run_layout(ch, EncodeMode::Parametric, 1920);
        // The tonal test content sits below the A-SPX crossover, so
        // the ASPX_SCPL / 22.2-A-SPX routes land near the waveform
        // floor; the LFE bypasses the QMF stage (lag 0 vs 577).
        for (c, e) in errs.iter().enumerate() {
            assert!(*e < 0.10, "{ch}ch parametric ch{c}: rel err {e:.4}");
        }
    }
}

#[test]
fn framing_variants_and_s16_input() {
    let reg = registry();
    let chans: Vec<Vec<f32>> = (0..2).map(|c| signal(c, N * 2)).collect();
    // Raw framing: packets are bare raw_ac4_frame() payloads.
    let p = params(
        2,
        SampleFormat::F32,
        CodecOptions::new().set("framing", "raw"),
    );
    let mut enc = reg.first_encoder(&p).unwrap();
    let raw = encode_all(enc.as_mut(), &chans, N);
    assert!(parse_sync_frame_at_start(&raw[0].data).is_none());
    // CRC framing: 0xAC41 with a valid trailer.
    let p = params(
        2,
        SampleFormat::F32,
        CodecOptions::new().set("framing", "sync_crc"),
    );
    let mut enc = reg.first_encoder(&p).unwrap();
    let crc = encode_all(enc.as_mut(), &chans, N);
    let sf = parse_sync_frame_at_start(&crc[0].data).expect("sync frame");
    assert!(
        sf.crc_protected && sf.crc_valid == Some(true),
        "0xAC41 CRC verified"
    );
    assert_eq!(
        sf.payload,
        &raw[0].data[..],
        "same raw frame under both framings"
    );
    // Both decode identically.
    let mut d = reg.first_decoder(&p).unwrap();
    let a = decode_all(d.as_mut(), &raw, 2);
    let mut d = reg.first_decoder(&p).unwrap();
    let b = decode_all(d.as_mut(), &crc, 2);
    assert_eq!(a, b);

    // S16 interleaved input reaches the same encoder output as F32.
    let p16 = params(
        2,
        SampleFormat::S16,
        CodecOptions::new().set("framing", "raw"),
    );
    let mut enc16 = reg.first_encoder(&p16).unwrap();
    let mut data = Vec::new();
    for i in 0..N * 2 {
        for c in &chans {
            let v = (c[i] * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    enc16
        .send_frame(&Frame::Audio(AudioFrame {
            samples: (N * 2) as u32,
            pts: Some(0),
            data: vec![data],
        }))
        .unwrap();
    enc16.flush().unwrap();
    let p0 = enc16.receive_packet().unwrap();
    let p1 = enc16.receive_packet().unwrap();
    let mut d = reg.first_decoder(&p16).unwrap();
    let s16_out = decode_all(d.as_mut(), &[p0, p1], 2);
    let mut d = reg.first_decoder(&p16).unwrap();
    let f32_out = decode_all(d.as_mut(), &raw[..2], 2);
    for c in 0..2 {
        let e = rel_err(&f32_out[c][N..], &s16_out[c][N..]);
        assert!(e < 0.01, "S16 vs F32 input divergence ch{c}: {e:.4}");
    }
}

#[test]
fn flush_pads_partial_tail_and_reports_eof() {
    let reg = registry();
    let p = params(1, SampleFormat::F32, CodecOptions::new());
    let mut enc = reg.first_encoder(&p).unwrap();
    let chans = vec![signal(0, N + 700)];
    let pkts = encode_all(enc.as_mut(), &chans, 500);
    assert_eq!(pkts.len(), 2, "1 full frame + 1 padded tail frame");
    assert_eq!(pkts[1].pts, Some(N as i64));
    assert!(enc.receive_packet().unwrap_err().is_eof());
    assert!(enc
        .send_frame(&Frame::Audio(f32_interleaved(&chans, 0, 10)))
        .is_err());
}

#[test]
fn typed_constructor_and_output_params() {
    let p = params(6, SampleFormat::F32, CodecOptions::new());
    let enc = Ac4Encoder::with_options(
        &p,
        Ac4EncoderOptions {
            framing: Framing::Raw,
            bandwidth_hz: 8_000,
            ..Ac4EncoderOptions::default()
        },
    )
    .unwrap();
    assert_eq!(enc.frame_len(), N);
    assert!(
        enc.max_sfb() < 55 && enc.max_sfb() > 20,
        "max_sfb {}",
        enc.max_sfb()
    );
    let out = enc.output_params();
    assert_eq!(out.codec_id, CodecId::new("ac4"));
    assert_eq!(out.channel_layout, Some(ChannelLayout::Surround51));
    assert_eq!(out.sample_rate, Some(48_000));
}

#[test]
fn gop_emits_pframes_on_parametric_immersive() {
    let reg = registry();
    let opts = CodecOptions::new()
        .set("mode", "parametric")
        .set("gop", "3")
        .set("framing", "raw");
    let p = params(11, SampleFormat::F32, opts);
    let mut enc = reg.first_encoder(&p).unwrap();
    let chans = signals(11);
    let pkts = encode_all(enc.as_mut(), &chans, N);
    let kf: Vec<bool> = pkts.iter().map(|p| p.flags.keyframe).collect();
    assert_eq!(kf, [true, false, false, true, false, false]);
    // The GOP still decodes to the same settled parity as the all-I
    // stream.
    let mut dec = reg.first_decoder(&p).unwrap();
    let out = decode_all(dec.as_mut(), &pkts, 11);
    for c in 0..11 {
        let (e, lag) = settled_err(&chans[c], &out[c]);
        eprintln!("ROUND-456 gop=3 parametric 7.0.4 ch{c}: settled rel err {e:.4} (lag {lag})");
        assert!(e < 0.10, "gop parity ch{c}: {e:.4}");
    }
}
