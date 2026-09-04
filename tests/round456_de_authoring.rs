//! Round 456 — encoder-side dialogue-enhancement authoring
//! (`de_author`, TS 103 190-1 §4.3.14 / §5.7.8.7 channel-independent
//! method) carried in the closing `metadata()` element and validated
//! through the decoder's own DE application tool on the immersive
//! route:
//!
//! * a 7.0.4 SCPL stream whose centre channel is entirely dialogue
//!   decodes with `G_DE = 6 dB` at +6 dB on C and unchanged L / R;
//! * a centre whose dialogue stem is half the mix amplitude lands at
//!   `20·log10(1 + g·0,5)` ≈ +3,5 dB;
//! * the emitted `dialog_enhancement()` parses back to the authored
//!   `de_par` rows and the announced `tools_metadata_size` is exact.

use oxideav_ac4::de::{DeConfig, DeMethod};
use oxideav_ac4::de_author::DeAuthor;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::metadata::parse_metadata_v2;
use oxideav_ac4::toc::parse_ac4_toc;
use oxideav_core::bits::BitReader;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FRAMES: usize = 6;

fn tone(freq: f32, amp: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin())
        .collect()
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

/// Encode a 7.0.4 stream (centre = `mix_c`, dialogue stem `dlg_c`, the
/// other channels independent tones) with authored DE metadata, decode
/// under `gain_db`, and return the settled per-channel RMS of L / R /
/// C.
fn run(mix_c: &[f32], dlg_c: &[f32], gain_db: f32) -> ([f64; 3], Vec<u8>) {
    let len = N * FRAMES;
    let mut chans: Vec<Vec<f32>> = (0..11)
        .map(|c| tone(150.0 + 45.0 * c as f32, 0.15, len))
        .collect();
    chans[2] = mix_c.to_vec();
    let cfg = DeConfig {
        method: DeMethod::ChannelIndependent,
        max_gain: 3, // Gmax = 12 dB
        channel_config: 0b001,
    };
    let mut enc = Ac4ImsEncoder::new();
    let mut author = DeAuthor::new();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    dec.set_dialogue_enhancement_gain_db(gain_db);
    let mut out = vec![Vec::new(); 11];
    let mut last_frame = Vec::new();
    for k in 0..FRAMES {
        let r = k * N..(k + 1) * N;
        let data = author
            .author_channel_independent(
                &cfg,
                [&[], &[], &chans[2][r.clone()]],
                [&[], &[], &dlg_c[r.clone()]],
            )
            .expect("authored");
        enc.dialogue_enhancement = Some((cfg, data));
        let refs: [&[f32]; 11] = std::array::from_fn(|c| &chans[c][r.clone()]);
        let bytes = enc.encode_frame_pcm_7_0_4_ice_scpl_sap(&refs);
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 48_000), bytes.clone()))
            .unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!()
        };
        for i in 0..af.samples as usize {
            for (c, ch) in out.iter_mut().enumerate() {
                let off = (i * 11 + c) * 2;
                ch.push(
                    i16::from_le_bytes([af.data[0][off], af.data[0][off + 1]]) as f32 / 32768.0,
                );
            }
        }
        last_frame = bytes;
    }
    let settled = (FRAMES - 1) * N..FRAMES * N;
    (
        [
            rms(&out[0][settled.clone()]),
            rms(&out[1][settled.clone()]),
            rms(&out[2][settled]),
        ],
        last_frame,
    )
}

#[test]
fn full_dialogue_centre_is_boosted_by_the_user_gain() {
    let len = N * FRAMES;
    let c = tone(440.0, 0.2, len);
    let (base, _) = run(&c, &c, 0.0);
    let (boosted, _) = run(&c, &c, 6.0);
    let db = |a: f64, b: f64| 20.0 * (a / b).log10();
    let d_c = db(boosted[2], base[2]);
    let d_l = db(boosted[0], base[0]);
    let d_r = db(boosted[1], base[1]);
    eprintln!("ROUND-456 DE full dialogue: C {d_c:+.2} dB, L {d_l:+.2} dB, R {d_r:+.2} dB");
    assert!(
        (d_c - 6.0).abs() < 0.5,
        "centre must gain ≈ 6 dB, got {d_c:+.2}"
    );
    assert!(d_l.abs() < 0.2 && d_r.abs() < 0.2, "L / R untouched");
}

#[test]
fn half_dialogue_centre_is_boosted_proportionally() {
    let len = N * FRAMES;
    let c = tone(440.0, 0.2, len);
    let half: Vec<f32> = c.iter().map(|v| v * 0.5).collect();
    let (base, _) = run(&c, &half, 0.0);
    let (boosted, _) = run(&c, &half, 6.0);
    let d_c = 20.0 * (boosted[2] / base[2]).log10();
    // Y = m + g·p·m with p = 0,5 and g = 10^(6/20) − 1 → +3,5 dB.
    let expect = 20.0 * (1.0 + (10f64.powf(0.3) - 1.0) * 0.5).log10();
    eprintln!("ROUND-456 DE half dialogue: C {d_c:+.2} dB (expected {expect:+.2})");
    assert!(
        (d_c - expect).abs() < 0.5,
        "got {d_c:+.2}, expected {expect:+.2}"
    );
}

#[test]
fn user_gain_is_clamped_to_the_authored_gmax() {
    let len = N * FRAMES;
    let c = tone(440.0, 0.2, len);
    let (base, _) = run(&c, &c, 0.0);
    let (boosted, _) = run(&c, &c, 30.0);
    let d_c = 20.0 * (boosted[2] / base[2]).log10();
    eprintln!("ROUND-456 DE 30 dB request under Gmax 12 dB: C {d_c:+.2} dB");
    assert!(
        (d_c - 12.0).abs() < 0.5,
        "clamped to Gmax = 12 dB, got {d_c:+.2}"
    );
}

#[test]
fn emitted_dialog_enhancement_parses_back() {
    let len = N * FRAMES;
    let c = tone(440.0, 0.2, len);
    let (_, frame) = run(&c, &c, 0.0);
    let info = parse_ac4_toc(&frame).unwrap();
    let start = (info.toc_size + info.payload_base) as usize;
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 48_000), frame.clone()))
        .unwrap();
    let _ = dec.receive_frame().unwrap();
    let sub = dec.last_substream.as_ref().unwrap();
    let meta_off = start + (sub.audio_data_offset + sub.audio_size) as usize;
    let mut br = BitReader::new(&frame[meta_off..]);
    let meta = parse_metadata_v2(&mut br, 11, true, false, None).expect("metadata v2");
    let de = &meta.dialog_enhancement;
    assert!(de.data_present);
    let cfg = de.config.expect("I-frame de_config");
    assert_eq!(cfg.method, DeMethod::ChannelIndependent);
    assert_eq!(cfg.channel_config, 0b001);
    let data = de.data.as_ref().expect("de_data");
    assert_eq!(data.de_par.len(), 1);
    assert_eq!(
        data.de_par[0][1], 10,
        "full dialogue in the 440 Hz band → p = 1,0"
    );
    assert_eq!(
        meta.tools_metadata_trailing_bits, 0,
        "tools_metadata_size is exact"
    );
    assert_eq!(
        meta_off + br.bit_position().div_ceil(8) as usize,
        frame.len(),
        "frame ends right after metadata()"
    );
}
