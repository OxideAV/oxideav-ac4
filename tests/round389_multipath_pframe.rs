//! Round 389 — P-frame (`b_iframe = 0`) coverage for the **remaining
//! live A-SPX paths**: 5_X / 7_X ASPX_ACPL_2, 7_X ASPX_ACPL_1,
//! 5_X ASPX_ACPL_1 (SAP), and 7.0 pure-ASPX.
//!
//! Per ETSI TS 103 190-1 §4.2.6.6 Table 25 / §4.2.6.14 Table 33 the
//! `aspx_data_*()` and `acpl_data_1ch()` data elements are present on
//! every frame; only the configs and the per-element 3-bit
//! `aspx_xover_subband_offset` are I-frame-gated. Each live body
//! builder now emits the correct P-frame shape, and the decoder's
//! sticky-config state parses it. These tests drive every path through
//! an I,P sequence and assert the P-frame's data layer parsed —
//! and does *not* parse on a cold decoder that never saw the I-frame.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

fn multitone(amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            let mut v = 0.0f32;
            for &f in &[400.0f32, 1200.0, 3000.0, 9000.0, 13_000.0] {
                v += (2.0 * std::f32::consts::PI * f * t).sin();
            }
            amp * v / 5.0
        })
        .collect()
}

fn decode_one(dec: &mut Ac4Decoder, bytes: Vec<u8>, expect_ch: usize) -> Vec<i16> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    let raw = &af.data[0];
    assert_eq!(raw.len(), 1920 * expect_ch * 2, "{expect_ch}-ch S16");
    raw.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Shared I,P harness: encode two frames with `enc_frame`, decode both
/// warm, decode the P-frame cold, and return
/// `(warm_p_tools_has_aspx, warm_p_tools_has_acpl_pair, cold_has_aspx)`.
fn run_i_p<F: FnMut(&mut Ac4ImsEncoder) -> Vec<u8>>(
    mut enc_frame: F,
    expect_ch: usize,
) -> (bool, bool, bool) {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    enc.b_iframe_global = true;
    let f_i = enc_frame(&mut enc);
    enc.b_iframe_global = false;
    let f_p = enc_frame(&mut enc);

    let mut dec = Ac4Decoder::new(&params);
    let _ = decode_one(&mut dec, f_i, expect_ch);
    let pcm_p = decode_one(&mut dec, f_p.clone(), expect_ch);
    assert!(
        !dec.last_info.as_ref().unwrap().b_iframe_global,
        "P-frame must be signalled"
    );
    let e: i64 = pcm_p.iter().map(|&s| (s as i64) * (s as i64)).sum();
    assert!(e > 0, "P-frame must decode to nonsilent PCM");
    let tools = &dec.last_substream.as_ref().unwrap().tools;
    let has_aspx = tools.aspx_data_sig_primary.is_some();
    let has_acpl_pair = tools.acpl_data_1ch_pair[0].is_some();

    let mut dec_cold = Ac4Decoder::new(&params);
    let _ = decode_one(&mut dec_cold, f_p, expect_ch);
    let cold_has_aspx = dec_cold
        .last_substream
        .as_ref()
        .map(|s| s.tools.aspx_data_sig_primary.is_some())
        .unwrap_or(false);
    (has_aspx, has_acpl_pair, cold_has_aspx)
}

#[test]
fn acpl2_5_0_p_frame_parses_aspx_and_acpl_pair() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let (aspx, acpl, cold) = run_i_p(
        |enc| enc.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]),
        5,
    );
    assert!(aspx, "5_X ACPL_2 P-frame aspx_data must parse");
    assert!(acpl, "5_X ACPL_2 P-frame acpl_data_1ch pair must parse");
    assert!(!cold, "cold P-frame decode must have no A-SPX layer");
}

#[test]
fn acpl2_7_0_p_frame_parses_aspx_and_acpl_pair() {
    let chans: Vec<Vec<f32>> = (0..7)
        .map(|i| {
            if i < 2 {
                multitone(0.3 - 0.02 * i as f32)
            } else {
                tone(500.0 + 200.0 * i as f32, 0.2)
            }
        })
        .collect();
    let refs: [&[f32]; 7] = std::array::from_fn(|i| chans[i].as_slice());
    let (aspx, acpl, cold) = run_i_p(|enc| enc.encode_frame_pcm_7_0_acpl2_real_aspx(&refs), 7);
    assert!(aspx, "7_X ACPL_2 P-frame aspx_data must parse");
    assert!(acpl, "7_X ACPL_2 P-frame acpl_data_1ch pair must parse");
    assert!(!cold, "cold P-frame decode must have no A-SPX layer");
}

#[test]
fn acpl1_7_0_p_frame_parses_aspx_and_acpl_pair() {
    let chans: Vec<Vec<f32>> = (0..7)
        .map(|i| {
            if i < 2 {
                multitone(0.3 - 0.02 * i as f32)
            } else {
                tone(500.0 + 200.0 * i as f32, 0.2)
            }
        })
        .collect();
    let refs: [&[f32]; 7] = std::array::from_fn(|i| chans[i].as_slice());
    let (aspx, acpl, cold) = run_i_p(
        |enc| enc.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&refs),
        7,
    );
    assert!(aspx, "7_X ACPL_1 P-frame aspx_data must parse");
    assert!(acpl, "7_X ACPL_1 P-frame acpl_data_1ch pair must parse");
    assert!(!cold, "cold P-frame decode must have no A-SPX layer");
}

#[test]
fn aspx_7_0_pure_p_frame_parses_all_trailers() {
    let chans: Vec<Vec<f32>> = (0..7).map(|i| multitone(0.3 - 0.02 * i as f32)).collect();
    let refs: [&[f32]; 7] = std::array::from_fn(|i| chans[i].as_slice());
    let (aspx, _acpl, cold) = run_i_p(|enc| enc.encode_frame_pcm_7_0_aspx_real_aspx(&refs), 7);
    assert!(aspx, "7.0 pure-ASPX P-frame aspx trailers must parse");
    assert!(!cold, "cold P-frame decode must have no A-SPX layer");
}

#[test]
fn acpl1_5_0_sap_p_frame_parses_aspx() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let (aspx, acpl, cold) = run_i_p(
        |enc| enc.encode_frame_pcm_5_0_acpl1_sap(&[&l, &r, &c, &ls, &rs]),
        5,
    );
    assert!(aspx, "5_X ACPL_1 SAP P-frame aspx_data must parse");
    assert!(acpl, "5_X ACPL_1 SAP P-frame acpl_data_1ch pair must parse");
    assert!(!cold, "cold P-frame decode must have no A-SPX layer");
}

/// Deterministic emission across all multipath P-frames.
#[test]
fn multipath_p_frames_deterministic() {
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let run = || {
        let mut enc = Ac4ImsEncoder::new();
        enc.b_iframe_global = true;
        let a = enc.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
        enc.b_iframe_global = false;
        let b = enc.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
        (a, b)
    };
    let (a1, b1) = run();
    let (a2, b2) = run();
    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_ne!(a1, b1);
}
