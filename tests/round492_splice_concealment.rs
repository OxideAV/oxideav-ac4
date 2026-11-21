//! Round 492 — `sequence_counter` handling (TS 103 190-1 §4.3.3.2.2)
//! and opt-in error concealment.
//!
//! The encoder counts 1..=1020 and wraps to 1 (0 is the splicer's
//! marker). The decoder reports a source change when a frame's counter
//! is none of: previous + 1, the 1 020 → 1 wrap, or any non-zero value
//! after a 0. With concealment enabled a damaged frame (0xAC41 CRC
//! mismatch) or the frames between a source change and the next
//! I-frame come out as silence instead of an error / a prediction from
//! the wrong side of the gap.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::toc::{
    SEQUENCE_COUNTER_MAX, next_sequence_counter, parse_ac4_toc, sequence_counter_continues,
};
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Decoder, Frame, Packet,
    SampleFormat, TimeBase,
};

const N: usize = 1920;

fn tone(ch: usize, start: usize, len: usize) -> Vec<f32> {
    (start..start + len)
        .map(|i| {
            0.3 * (std::f32::consts::TAU * (330.0 + 110.0 * ch as f32) * i as f32 / 48_000.0).sin()
        })
        .collect()
}

fn encode(channels: usize, frames: usize, opts: CodecOptions) -> (CodecParameters, Vec<Packet>) {
    let mut reg = CodecRegistry::new();
    oxideav_ac4::register_codecs(&mut reg);
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(48_000);
    p.channels = Some(channels as u16);
    p.sample_format = Some(SampleFormat::F32);
    p.options = opts;
    let mut enc = reg.first_encoder(&p).expect("encoder");
    let len = N * frames;
    let chans: Vec<Vec<f32>> = (0..channels).map(|c| tone(c, 0, len)).collect();
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
    assert_eq!(pkts.len(), frames);
    (p, pkts)
}

fn energy(f: &Frame) -> i64 {
    let Frame::Audio(af) = f else { panic!("audio") };
    af.data[0]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| (i16::from_le_bytes(*b) as i64).pow(2))
        .sum()
}

#[test]
fn counter_rules() {
    assert_eq!(next_sequence_counter(0), 1);
    assert_eq!(next_sequence_counter(1019), 1020);
    assert_eq!(next_sequence_counter(SEQUENCE_COUNTER_MAX), 1);
    assert!(sequence_counter_continues(5, 6));
    assert!(sequence_counter_continues(1020, 1));
    assert!(sequence_counter_continues(0, 713));
    assert!(!sequence_counter_continues(5, 5));
    assert!(!sequence_counter_continues(5, 7));
    assert!(!sequence_counter_continues(1020, 2));
    assert!(!sequence_counter_continues(0, 0));
    assert!(!sequence_counter_continues(9, 0));
}

#[test]
fn the_encoder_never_writes_a_counter_above_1020() {
    let mut enc = Ac4ImsEncoder::new();
    enc.sequence_counter = 1018;
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut seen = Vec::new();
    for f in 0..5 {
        let x = tone(0, f * N, N);
        let bytes = enc.encode_frame_pcm_stereo(&x, &x);
        seen.push(parse_ac4_toc(&bytes).unwrap().sequence_counter);
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 48_000), bytes))
            .unwrap();
        dec.receive_frame().unwrap();
    }
    assert_eq!(seen, vec![1018, 1019, 1020, 1, 2]);
    assert_eq!(dec.source_changes(), 0, "the wrap is normal operation");
}

#[test]
fn a_gap_is_reported_as_a_source_change() {
    let (p, pkts) = encode(2, 10, CodecOptions::new());
    let mut dec = Ac4Decoder::new(&p);
    for (i, pk) in pkts.iter().enumerate() {
        if i == 4 || i == 5 {
            continue;
        }
        dec.send_packet(pk).unwrap();
        dec.receive_frame().unwrap();
        assert_eq!(dec.last_frame_was_source_change(), i == 6, "frame {i}");
    }
    assert_eq!(dec.source_changes(), 1);
    assert_eq!(dec.concealed_frames(), 0, "concealment is off by default");
}

#[test]
fn a_crc_damaged_frame_is_an_error_or_a_concealed_frame() {
    let (p, mut pkts) = encode(2, 8, CodecOptions::new().set("framing", "sync_crc"));
    // Flip a payload bit of frame 3: the 0xAC41 CRC no longer matches.
    let mid = pkts[3].data.len() / 2;
    pkts[3].data[mid] ^= 0x10;

    let mut strict = Ac4Decoder::new(&p);
    for (i, pk) in pkts.iter().enumerate() {
        strict.send_packet(pk).unwrap();
        assert_eq!(strict.receive_frame().is_err(), i == 3, "frame {i}");
    }

    let mut lenient = Ac4Decoder::new(&p);
    lenient.set_error_concealment(true);
    for (i, pk) in pkts.iter().enumerate() {
        lenient.send_packet(pk).unwrap();
        let f = lenient.receive_frame().expect("no error with concealment");
        let Frame::Audio(af) = &f else {
            panic!("audio")
        };
        assert_eq!(af.samples as usize, N);
        assert_eq!(af.data[0].len(), N * 2 * 2);
        if i == 3 {
            assert_eq!(energy(&f), 0, "the damaged frame is silent");
        } else if i >= 5 {
            assert!(energy(&f) > 0, "frame {i} decodes again");
        }
    }
    assert_eq!(lenient.concealed_frames(), 1);
    // The decoder option does the same.
    let mut q = p.clone();
    q.options = CodecOptions::new().set("concealment", "on");
    let mut by_option = Ac4Decoder::new(&q);
    for pk in &pkts {
        by_option.send_packet(pk).unwrap();
        by_option
            .receive_frame()
            .expect("no error with concealment");
    }
    assert_eq!(by_option.concealed_frames(), 1);
}

#[test]
fn after_a_source_change_output_is_concealed_until_the_next_i_frame() {
    // Parametric 5.1 with an I-frame every 4 frames: frames 0, 4, 8, …
    let (p, pkts) = encode(
        6,
        12,
        CodecOptions::new()
            .set("mode", "parametric")
            .set("gop", "4"),
    );
    let keyframes: Vec<bool> = pkts.iter().map(|pk| pk.flags.keyframe).collect();
    assert!(keyframes[0] && keyframes[4] && keyframes[8] && !keyframes[5]);
    let mut dec = Ac4Decoder::new(&p);
    dec.set_error_concealment(true);
    let mut silent = Vec::new();
    for (i, pk) in pkts.iter().enumerate() {
        // Lose frame 5: frame 6 arrives with a counter jump.
        if i == 5 {
            continue;
        }
        dec.send_packet(pk).unwrap();
        let f = dec.receive_frame().expect("decodes or conceals");
        if energy(&f) == 0 {
            silent.push(i);
        }
    }
    assert_eq!(dec.source_changes(), 1);
    // P-frames 6 and 7 cannot be decoded without frame 5's state;
    // frame 8 is the next I-frame.
    assert_eq!(silent, vec![6, 7]);
    assert_eq!(dec.concealed_frames(), 2);
}
