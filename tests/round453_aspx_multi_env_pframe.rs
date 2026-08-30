//! Round 453 — **multi-envelope (`num_env > 1`) P-frame bodies with
//! TIME-direction envelopes** on the live 5_X ASPX_ACPL_3 path (ETSI
//! TS 103 190-1 §5.7.6.3.4 Pseudocodes 80 / 81): the encoder now tracks
//! the last envelope of a multi-envelope frame as the next frame's
//! `qscf_*_prev` reference (sum / pan wire-step domain) instead of
//! clearing it, so a P-frame after a multi-envelope frame — itself
//! multi-envelope or single-envelope — may delta-code its leading
//! envelope in TIME direction against it, and the decoder's Pseudocode
//! 80 reconstruction lands on the same absolute rows.

use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const FS: f32 = 48_000.0;
const N: usize = 1920;

fn live_cfg() -> AspxConfig {
    AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        start_freq: 0,
        stop_freq: 0,
        master_freq_scale: AspxMasterFreqScale::LowRes,
        interpolation: false,
        preflat: false,
        limiter: false,
        noise_sbg: 0,
        num_env_bits_fixfix: 0,
        freq_res_mode: AspxFreqResMode::DurationDependent,
    }
}

fn tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / FS).sin())
        .collect()
}

/// HF tone gated to one half of the frame (`loud_second_half` picks
/// which) — the transient the encoder's `num_env` driver promotes to
/// a 2-envelope FIXFIX frame.
fn hf_transient(freq: f32, amp: f32, loud_second_half: bool) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let second = i >= N / 2;
            let gate = if second == loud_second_half {
                1.0
            } else {
                0.02
            };
            gate * amp * (2.0 * std::f32::consts::PI * freq * i as f32 / FS).sin()
        })
        .collect()
}

fn decode_one(dec: &mut Ac4Decoder, bytes: Vec<u8>) -> Vec<f64> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).unwrap();
    let Frame::Audio(af) = dec.receive_frame().unwrap() else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, N as u32);
    let buf = &af.data[0];
    let mut e = vec![0.0f64; 5];
    for i in 0..N {
        for (c, slot) in e.iter_mut().enumerate() {
            let off = (i * 5 + c) * 2;
            let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as f64;
            *slot += s * s;
        }
    }
    e
}

#[test]
fn multi_env_p_frame_codes_leading_envelope_in_time_direction() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    // I-frame: quiet → loud; P-frame: loud → quiet. Through the
    // encoder's per-frame QMF analysis (whose prototype-window delay
    // shifts the frame by ten slots) the I-frame's last envelope and
    // the P-frame's leading envelope both read the loud level — the
    // TIME reference the encoder must have kept from the
    // multi-envelope I-frame makes the P-frame's leading envelope a
    // near-zero TIME row.
    let l_i = hf_transient(14_000.0, 0.6, true);
    let r_i = hf_transient(11_000.0, 0.5, true);
    let l_p = hf_transient(14_000.0, 0.6, false);
    let r_p = hf_transient(11_000.0, 0.5, false);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let chans_i: [&[f32]; 5] = [&l_i, &r_i, &c, &ls, &rs];
    let chans_p: [&[f32]; 5] = [&l_p, &r_p, &c, &ls, &rs];

    enc.b_iframe_global = true;
    let f_i = enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(&chans_i, 0.5, 0.1, 1.0, 1.0);
    let _ = decode_one(&mut dec, f_i);
    let sig_i = dec
        .last_substream
        .as_ref()
        .unwrap()
        .tools
        .aspx_data_sig_primary
        .clone()
        .expect("I-frame sig data");
    assert_eq!(
        sig_i.len(),
        2,
        "transient I-frame must carry two SIGNAL envelopes"
    );
    assert!(sig_i.iter().all(|e| !e.direction_time || e == &sig_i[1]));
    assert!(!sig_i[0].direction_time, "I-frame leading envelope is FREQ");

    // Mirrored transient on the P-frame: multi-envelope again, with
    // the leading envelope TIME-coded against the I-frame's last.
    enc.b_iframe_global = false;
    let f_p = enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(&chans_p, 0.5, 0.1, 1.0, 1.0);
    let _ = decode_one(&mut dec, f_p);
    let tools_p = &dec.last_substream.as_ref().unwrap().tools;
    let dd_p = tools_p
        .aspx_delta_dir_primary
        .as_ref()
        .expect("P-frame delta dir");
    assert_eq!(
        dd_p.sig_delta_dir.len(),
        2,
        "P-frame keeps two SIGNAL envelopes"
    );
    assert!(
        dd_p.sig_delta_dir[0],
        "P-frame leading envelope must be TIME-coded against the multi-envelope reference"
    );
    let sig_p = tools_p
        .aspx_data_sig_primary
        .clone()
        .expect("P-frame sig data");
    assert!(sig_p[0].direction_time);

    // Decoder-side Pseudocode 80: with the I-frame's LAST envelope as
    // prev, the P-frame's leading envelope reconstructs to that same
    // (loud) level within a couple of quantizer steps — the TIME row
    // landed on the reference the encoder tracked.
    let cfg = live_cfg();
    let tables = oxideav_ac4::aspx::derive_aspx_frequency_tables(&cfg, 0).expect("tables");
    let qscf_i = oxideav_ac4::aspx::delta_decode_sig_p80(
        &sig_i,
        &[],
        &tables.sbg_sig_highres,
        &tables.sbg_sig_lowres,
        &[],
        1,
    );
    let prev_last: Vec<i32> = qscf_i.iter().map(|row| *row.last().unwrap()).collect();
    let qscf_p = oxideav_ac4::aspx::delta_decode_sig_p80(
        &sig_p,
        &[],
        &tables.sbg_sig_highres,
        &tables.sbg_sig_lowres,
        &prev_last,
        1,
    );
    assert_eq!(qscf_i.len(), qscf_p.len());
    for (sbg, (ri, rp)) in qscf_i.iter().zip(qscf_p.iter()).enumerate() {
        assert_eq!(ri.len(), 2);
        assert_eq!(rp.len(), 2);
        assert!(
            (rp[0] - ri[1]).abs() <= 2,
            "sbg {sbg}: P leading row {rp:?} must sit on the I last row {ri:?}"
        );
    }
}

/// A stationary single-envelope P-frame after a multi-envelope frame
/// decodes against the tracked last-envelope reference (no stale-row
/// FREQ fallback needed): the GOP stays non-silent and finite on every
/// channel.
#[test]
fn single_env_p_frame_after_multi_env_frame_decodes() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let l_t = hf_transient(14_000.0, 0.6, true);
    let r_t = hf_transient(11_000.0, 0.5, true);
    // HF-rich but stationary (no temporal gate): the num_env driver
    // sees a flat HF energy profile and keeps one envelope.
    let l_s: Vec<f32> = tone(9_000.0, 0.3)
        .iter()
        .zip(tone(15_000.0, 0.2))
        .map(|(a, b)| a + b)
        .collect();
    let r_s: Vec<f32> = tone(6_000.0, 0.25)
        .iter()
        .zip(tone(18_000.0, 0.3))
        .map(|(a, b)| a + b)
        .collect();
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);

    enc.b_iframe_global = true;
    let f0 = enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l_t, &r_t, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let _ = decode_one(&mut dec, f0);
    assert_eq!(
        dec.last_substream
            .as_ref()
            .unwrap()
            .tools
            .aspx_data_sig_primary
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    enc.b_iframe_global = false;
    let f1 = enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l_s, &r_s, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let e1 = decode_one(&mut dec, f1);
    let sig_1 = dec
        .last_substream
        .as_ref()
        .unwrap()
        .tools
        .aspx_data_sig_primary
        .clone()
        .unwrap();
    assert_eq!(
        sig_1.len(),
        1,
        "stationary P-frame falls back to one envelope"
    );
    let f2 = enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l_s, &r_s, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let e2 = decode_one(&mut dec, f2);
    for (ch, (a, b)) in e1.iter().zip(e2.iter()).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "channel {ch} finite");
        assert!(
            *b > 1e4,
            "channel {ch} silent on the settled P-frame: {e2:?}"
        );
    }
}
