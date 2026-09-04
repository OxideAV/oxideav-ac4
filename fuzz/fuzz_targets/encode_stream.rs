//! Framework-encoder round trip: the leading bytes pick the layout,
//! tool family, packet framing, frame-rate index, bandwidth, band gate
//! and GOP; the rest is S16 PCM fed through `Ac4Encoder` (FIFO framing
//! + flush) and every produced packet is decoded again with
//! `Ac4Decoder`. Construction errors and decode errors are fine; a
//! panic anywhere on the writer / reader path is a finding.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder::{Ac4Encoder, Ac4EncoderOptions, EncodeMode, Framing};
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Decoder, Encoder, Frame, SampleFormat,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 || data.len() > 1 << 15 {
        return;
    }
    let ctl = &data[..6];
    let pcm = &data[6..];
    // Layouts: mostly the cheap ones, the immersive / 22.2 shapes on a
    // narrow selector so the naive MDCT keeps the iteration fast.
    let channels: u16 = match ctl[0] % 16 {
        0..=3 => 1,
        4..=6 => 2,
        7 => 5,
        8 => 6,
        9 => 7,
        10 => 8,
        11 => 11,
        12 => 12,
        13 => 13,
        14 => 14,
        _ => 24,
    };
    let mode = if ctl[1] & 1 == 0 {
        EncodeMode::Waveform
    } else {
        EncodeMode::Parametric
    };
    let framing = match (ctl[1] >> 1) % 3 {
        0 => Framing::Sync,
        1 => Framing::SyncCrc,
        _ => Framing::Raw,
    };
    let (sample_rate, frame_rate_index) = match ctl[2] % 8 {
        0 => (48_000, 0),
        1 => (48_000, 1),
        2 => (48_000, 2),
        3 => (48_000, 3),
        4 => (48_000, 4),
        5 => (48_000, 13),
        6 => (44_100, 13),
        _ => (48_000, u32::from(ctl[2] >> 3) % 16), // incl. rejected ones
    };
    let opts = Ac4EncoderOptions {
        frame_rate_index,
        framing,
        mode,
        bandwidth_hz: 400 + 100 * u32::from(ctl[3]),
        dynamic_range_db: u32::from(ctl[4] % 100),
        gop: 1 + u32::from(ctl[5] % 4),
    };
    let mut p = CodecParameters::audio(CodecId::new("ac4"));
    p.sample_rate = Some(sample_rate);
    p.channels = Some(channels);
    p.sample_format = Some(SampleFormat::S16);
    let Ok(mut enc) = Ac4Encoder::with_options(&p, opts) else {
        return;
    };
    // At most ~2,5 frames of the largest layout so one iteration stays
    // bounded; the FIFO / flush padding gets exercised by the ragged
    // tail.
    let frame_len = enc.frame_len();
    let max_samples = frame_len * 5 / 2;
    let nch = usize::from(channels);
    let samples = (pcm.len() / (2 * nch)).min(max_samples);
    if samples == 0 {
        return;
    }
    let frame = AudioFrame {
        samples: samples as u32,
        pts: Some(0),
        data: vec![pcm[..samples * nch * 2].to_vec()],
    };
    if enc.send_frame(&Frame::Audio(frame)).is_err() {
        return;
    }
    let _ = enc.flush();
    let mut dec = Ac4Decoder::new(&p);
    while let Ok(pkt) = enc.receive_packet() {
        if dec.send_packet(&pkt).is_ok() {
            let _ = dec.receive_frame();
        }
    }
});
