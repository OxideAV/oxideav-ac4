//! Round 340 (part 3) — the 7_X Table-202 **back-pair (Lb/Rb)** channel
//! ASPX envelopes: wire a live pure-ASPX (`7_X_codec_mode = ASPX`) 7.0
//! frame path whose body carries an independent `aspx_data_2ch()` real
//! envelope for the back pair Lb/Rb.
//!
//! ### Background
//!
//! Per ETSI TS 103 190-1 §4.2.6.14 Table 33, the pure-ASPX 7_X body carries
//! four ASPX trailers: `aspx_data_2ch()` (L/R front) + `aspx_data_2ch()`
//! (Ls/Rs surround) + `aspx_data_1ch()` (centre) — the
//! `7_X_codec_mode != SIMPLE` block — plus an **extra** `aspx_data_2ch()`
//! for the additional-channel pair — the `7_X_codec_mode == ASPX` block.
//! Per Table 202 (7_X_channel_element A-CPL channel mapping) the 3/4/0.x
//! additional pair is the back pair **Lb/Rb** (A-CPL variables x3/x4); in
//! pure-ASPX mode (no A-CPL coupling) it is carried as an independent
//! `two_channel_data()` carrier with its own HF-reconstruction envelope.
//!
//! The decoder's `parse_7x_audio_data_outer` already walks this fourth
//! `aspx_data_2ch()` for `mode == ASPX`; the encoder side previously only
//! emitted SIMPLE 7_X bodies (no ASPX trailers). This round adds
//! `encode_frame_pcm_7_0_aspx_real_aspx`, which QMF-analyses all seven
//! carriers and emits real SIGNAL/NOISE envelopes on each of the four ASPX
//! elements, closing the back-pair ASPX deferral.
//!
//! ### What this round measures
//!
//! 1. Round-trip — the pure-ASPX 7.0 encoder output is accepted by
//!    `Ac4Decoder` and yields a 7-channel AudioFrame.
//! 2. Back-pair liveness — an HF transient on the back pair Lb/Rb reaches
//!    the wire distinct from a flat-back-pair frame (the fourth
//!    `aspx_data_2ch()` is genuinely carrying the back-pair envelope).
//! 3. Determinism — matched inputs + fresh encoder state produce identical
//!    bytes.
//!
//! Refs ETSI TS 103 190-1: §4.2.6.14 Table 33 (`case ASPX:`), Table 202,
//! §4.2.12.3 Table 51, §4.2.12.4 Table 52.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn make_tone_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// 7.0 pure-ASPX encode round-trips to a 7-channel AudioFrame.
#[test]
fn encode_7_0_aspx_real_aspx_round_trips_to_7_channel_audio() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lb = make_tone_frame(1300.0, 0.2);
    let rb = make_tone_frame(1500.0, 0.2);

    let frame_bytes = enc.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 7 * 2, "7-channel S16 interleaved");
}

/// Back-pair liveness: the fourth `aspx_data_2ch()` genuinely carries the
/// Lb/Rb envelope — changing only the back pair's HF content changes the
/// wire bytes.
#[test]
fn back_pair_aspx_envelope_is_live() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);

    // Frame A: HF-rich back pair.
    let lb_a = make_tone_frame(15000.0, 0.6);
    let rb_a = make_tone_frame(14000.0, 0.6);
    let mut enc_a = Ac4ImsEncoder::new();
    let a = enc_a.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb_a, &rb_a]);

    // Frame B: near-silent back pair (different HF envelope).
    let lb_b = make_tone_frame(15000.0, 0.001);
    let rb_b = make_tone_frame(14000.0, 0.001);
    let mut enc_b = Ac4ImsEncoder::new();
    let b = enc_b.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb_b, &rb_b]);

    assert_eq!(a.len(), b.len(), "same padded substream length");
    assert_ne!(
        a, b,
        "the back-pair aspx_data_2ch() must reach the wire — changing only \
         Lb/Rb HF content must change the frame bytes"
    );
}

/// Determinism: matched inputs + fresh encoder state produce identical
/// bytes across repeated invocations.
#[test]
fn encode_7_0_aspx_real_aspx_is_byte_deterministic() {
    let l = make_tone_frame(440.0, 0.5);
    let r = make_tone_frame(550.0, 0.4);
    let c = make_tone_frame(660.0, 0.3);
    let ls = make_tone_frame(880.0, 0.2);
    let rs = make_tone_frame(1100.0, 0.2);
    let lb = make_tone_frame(1300.0, 0.2);
    let rb = make_tone_frame(1500.0, 0.2);
    let run = || -> Vec<u8> {
        let mut enc = Ac4ImsEncoder::new();
        enc.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb])
    };
    assert_eq!(run(), run());
}
