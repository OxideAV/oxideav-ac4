//! Round 456 — tight `ac4_substream()` bodies + the closing
//! `metadata()` element on every channel-coded frame the encoder emits.
//!
//! Before this round every body builder announced a fixed
//! `audio_size` "pad budget" (2–32 KiB), zero-filled up to it and
//! truncated anything longer: a mono frame cost 2 048 bytes whatever
//! the content, 5.0 cost 8 192, and no frame carried the mandatory
//! `metadata()` element (TS 103 190-1 §4.2.4.2 Table 16 / TS 103
//! 190-2 §6.2.2.2). Now `audio_size` is the exact byte length of
//! `audio_data()` (§4.3.4.1) and the substream closes with the minimal
//! `metadata(…)` element in the form the TOC's `sus_ver` demands.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder::{Ac4Encoder, Ac4EncoderOptions, EncodeMode, Framing};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::metadata::{parse_metadata, parse_metadata_v2, MetadataContext, MetadataState};
use oxideav_ac4::toc::parse_ac4_toc;
use oxideav_core::bits::BitReader;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Decoder, Encoder, Frame, Packet, SampleFormat, TimeBase,
};

const N: usize = 1920;

fn signal(ch: usize, len: usize, lfe: bool) -> Vec<f32> {
    let (f1, f2) = if lfe {
        (40.0, 75.0)
    } else {
        (110.0 + 37.0 * ch as f32, 700.0 + 61.0 * ch as f32)
    };
    (0..len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
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

/// Encode `frames` AC-4 frames of test content through the framework
/// encoder (raw framing) and return the raw frames.
fn raw_frames(channels: usize, mode: EncodeMode, gop: u32, frames: usize) -> Vec<Vec<u8>> {
    raw_frames_with_range(channels, mode, gop, frames, 60)
}

fn raw_frames_with_range(
    channels: usize,
    mode: EncodeMode,
    gop: u32,
    frames: usize,
    dynamic_range_db: u32,
) -> Vec<Vec<u8>> {
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    let mut enc = Ac4Encoder::with_options(
        &p,
        Ac4EncoderOptions {
            framing: Framing::Raw,
            mode,
            gop,
            dynamic_range_db,
            ..Ac4EncoderOptions::default()
        },
    )
    .expect("encoder");
    let total = N * frames;
    let chans: Vec<Vec<f32>> = (0..channels)
        .map(|c| signal(c, total, lfe_slots(channels).contains(&c)))
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
    let mut out = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.push(p.data);
    }
    out
}

/// Decode `frame` and return `(substream_start, audio_data_offset,
/// audio_size)` from the decoder's parsed substream view.
fn layout_of(dec: &mut Ac4Decoder, frame: &[u8]) -> (usize, usize, usize) {
    let info = parse_ac4_toc(frame).expect("toc");
    let start = (info.toc_size + info.payload_base) as usize;
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 48_000), frame.to_vec()))
        .unwrap();
    let _ = dec.receive_frame().expect("decode");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    (
        start,
        sub.audio_data_offset as usize,
        sub.audio_size as usize,
    )
}

/// The minimal `metadata(…, sus_ver = 1)`: `b_more_basic_metadata`,
/// `b_dialog`, `b_channels_classifier`, `b_event_probability`, 7 + 1
/// bits of tools size, the 1-bit `dialog_enhancement()`,
/// `b_emdf_payloads_substream` — 14 bits → 2 bytes.
const V2_META_BYTES: usize = 2;

#[test]
fn every_layout_announces_the_exact_audio_size_and_closes_with_metadata() {
    let mut report = Vec::new();
    for (channels, mode) in [
        (1usize, EncodeMode::Waveform),
        (2, EncodeMode::Waveform),
        (5, EncodeMode::Waveform),
        (6, EncodeMode::Waveform),
        (7, EncodeMode::Waveform),
        (8, EncodeMode::Waveform),
        (11, EncodeMode::Waveform),
        (12, EncodeMode::Waveform),
        (13, EncodeMode::Parametric),
        (14, EncodeMode::Parametric),
        (24, EncodeMode::Waveform),
        (24, EncodeMode::Parametric),
    ] {
        let frames = raw_frames(channels, mode, 1, 2);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        for frame in &frames {
            let (start, off, size) = layout_of(&mut dec, frame);
            let meta_off = start + off + size;
            assert_eq!(
                meta_off + V2_META_BYTES,
                frame.len(),
                "{channels}ch {mode:?}: frame must end right after audio_data + metadata()"
            );
            let mut br = BitReader::new(&frame[meta_off..]);
            let meta = parse_metadata_v2(&mut br, 0, true, false, None).expect("metadata v2");
            assert!(!meta.basic.more_basic_metadata);
            assert!(!meta.b_dialog);
            assert_eq!(meta.tools_metadata_size, 1);
            assert!(!meta.dialog_enhancement.data_present);
            assert!(meta.emdf_payloads_substream.is_none());
            assert_eq!(br.bit_position(), 14, "minimal v2 metadata is 14 bits");
        }
        report.push((channels, mode, frames[1].len()));
    }
    for (channels, mode, bytes) in &report {
        eprintln!(
            "ROUND-456 tight substream {channels}ch {mode:?}: {bytes} bytes/frame ({:.0} kbit/s at 24 fps)",
            *bytes as f64 * 8.0 * 24.0 / 1000.0
        );
    }
    // The old fixed pad budgets: mono 2 048, stereo 4 096, 5.X / 7.X
    // 8 192+ bytes. With the 60 dB band gate the test tones code an
    // order of magnitude under those budgets.
    let by = |ch: usize, m: EncodeMode| report.iter().find(|r| r.0 == ch && r.1 == m).unwrap().2;
    assert!(by(1, EncodeMode::Waveform) < 256);
    assert!(by(2, EncodeMode::Waveform) < 512);
    assert!(by(5, EncodeMode::Waveform) < 1_024);
    assert!(by(8, EncodeMode::Waveform) < 1_536);
    assert!(by(24, EncodeMode::Waveform) < 4_096);
}

#[test]
fn band_gate_ladder_trades_bytes_for_floor_bands() {
    // dynamic_range 0 = legacy (every band with energy coded at full
    // resolution), 80 / 60 / 40 dB progressively gate the leakage
    // floor. Bytes must fall monotonically; the decoded tones must
    // survive every setting (the gated bands hold only MDCT leakage).
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut sizes = Vec::new();
    for range in [0u32, 80, 60, 40] {
        let frames = raw_frames_with_range(2, EncodeMode::Waveform, 1, 4, range);
        let mut dec = Ac4Decoder::new(&params);
        let mut out = vec![Vec::new(); 2];
        for f in &frames {
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 48_000), f.clone()))
                .unwrap();
            let Frame::Audio(af) = dec.receive_frame().unwrap() else {
                panic!()
            };
            for i in 0..af.samples as usize {
                for (c, ch) in out.iter_mut().enumerate() {
                    let off = (i * 2 + c) * 2;
                    ch.push(
                        i16::from_le_bytes([af.data[0][off], af.data[0][off + 1]]) as f32 / 32768.0,
                    );
                }
            }
        }
        // Settled frame 2 of the input lands on output frame 3 (one
        // frame of MDCT latency).
        for (c, ch) in out.iter().enumerate() {
            let x = signal(c, N * 4, false);
            let reference = &x[2 * N..3 * N];
            let got = &ch[3 * N..4 * N];
            let num: f64 = reference
                .iter()
                .zip(got)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let den: f64 = reference.iter().map(|a| (*a as f64).powi(2)).sum();
            let e = (num / den).sqrt();
            eprintln!("ROUND-456 gate {range} dB: stereo ch{c} settled rel err {e:.4}");
            assert!(e < 0.10, "range {range} ch{c}: {e:.4}");
        }
        sizes.push(frames[2].len());
    }
    eprintln!("ROUND-456 gate ladder (0 / 80 / 60 / 40 dB): {sizes:?} bytes/frame");
    assert!(
        sizes[0] > sizes[1] && sizes[1] > sizes[2] && sizes[2] >= sizes[3],
        "{sizes:?}"
    );
    assert!(
        sizes[0] > 4 * sizes[2],
        "the gate must shed the leakage floor: {sizes:?}"
    );
}

#[test]
fn v0_path_closes_with_the_part_1_metadata_form() {
    let mut enc = Ac4ImsEncoder::new().with_v0();
    let pcm = signal(0, N, false);
    let frame = enc.encode_frame_pcm(&pcm);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let (start, off, size) = layout_of(&mut dec, &frame);
    let meta_off = start + off + size;
    // dialnorm_bits (7) + b_more_basic (1) + b_channels_classifier (1) +
    // b_event_probability (1) + tools size (7 + 1) + drc_frame (1) +
    // dialog_enhancement (1) + b_emdf_payloads_substream (1) = 21 bits
    // → 3 bytes.
    assert_eq!(meta_off + 3, frame.len());
    let mut br = BitReader::new(&frame[meta_off..]);
    let ctx = MetadataContext {
        channel_mode: 0,
        b_iframe: true,
        b_associated: false,
        b_dialog: false,
        frame_length: N as u32,
    };
    let meta = parse_metadata(&mut br, ctx, &MetadataState::default()).expect("metadata v0");
    assert_eq!(meta.basic.dialnorm_bits, Ac4ImsEncoder::DIALNORM_DEFAULT);
    assert!(!meta.drc.b_drc_present);
    assert!(!meta.dialog_enhancement.data_present);
    assert_eq!(meta.tools_metadata_size, 2);
    assert_eq!(meta.tools_metadata_trailing_bits, 0);
    assert_eq!(br.bit_position(), 21);
}

#[test]
fn pframes_are_smaller_than_their_iframe_anchor() {
    // With real (unpadded) sizes the P-frame saving is finally visible
    // at the packet level: the sticky aspx_config / xover are omitted.
    // Compare each frame against the all-I encode of the same content
    // so content variation between frames cannot mask the saving.
    let all_i = raw_frames(11, EncodeMode::Parametric, 1, 6);
    let gop3 = raw_frames(11, EncodeMode::Parametric, 3, 6);
    assert_eq!(all_i.len(), 6);
    assert_eq!(gop3.len(), 6);
    for k in 0..6 {
        if k % 3 == 0 {
            assert_eq!(gop3[k].len(), all_i[k].len(), "I-frame {k} identical size");
        } else {
            assert!(
                gop3[k].len() < all_i[k].len(),
                "P-frame {k} ({} B) must be smaller than the same content as an I-frame ({} B)",
                gop3[k].len(),
                all_i[k].len()
            );
        }
    }
}

#[test]
fn oversized_bodies_reopen_the_header_with_the_variable_bits_escape() {
    // Drive the 15-bit escape directly: a synthetic body past 32 767
    // bytes must announce its size through `b_more_bits +
    // variable_bits(7)` and still parse back to the same length.
    let mut bw = oxideav_core::bits::BitWriter::new();
    bw.write_u32(0, 15);
    bw.write_bit(false);
    bw.align_to_byte();
    let body_len = 40_000usize;
    for i in 0..body_len {
        bw.write_u32((i & 0xFF) as u32, 8);
    }
    let bytes = oxideav_ac4::encoder_asf::finish_substream_body(bw);
    let mut br = BitReader::new(&bytes);
    let mut audio_size = br.read_u32(15).unwrap();
    assert!(br.read_bit().unwrap(), "b_more_bits set");
    audio_size += oxideav_ac4::toc::variable_bits(&mut br, 7).unwrap() << 15;
    br.align_to_byte();
    assert_eq!(audio_size as usize, body_len);
    assert_eq!(bytes.len() - br.byte_position(), body_len);
}
