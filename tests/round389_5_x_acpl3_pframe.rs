//! Round 389 — **P-frame (`b_iframe = 0`) support for the live 5_X
//! SIMPLE/ASPX_ACPL_3 path**, encoder and decoder.
//!
//! Per ETSI TS 103 190-1 §4.2.6.6 Table 25 only the *configuration*
//! elements (`aspx_config()`, `acpl_config_2ch()`) and — per Table 52 —
//! the 3-bit `aspx_xover_subband_offset` are transmitted inside
//! `if (b_iframe) { … }` blocks. The `aspx_data_2ch()` and
//! `acpl_data_2ch()` **data** elements are present on every frame; a
//! non-I-frame simply reuses the configs from the most recent I-frame.
//!
//! Historically both sides of this crate treated the entire A-SPX /
//! A-CPL layer as I-frame-only: the encoder omitted the data elements
//! when `b_iframe == 0` and the decoder skipped parsing them. This
//! round fixes both:
//!
//! * the encoder's live ACPL_3 body builders emit `aspx_data_2ch()` +
//!   `acpl_data_2ch()` on every frame, gating only the configs and the
//!   xover offset on `b_iframe`;
//! * the decoder carries an [`oxideav_ac4::asf::StickyConfig`] across
//!   frames — harvested from each I-frame, seeded into each P-frame
//!   walk — so the P-frame data layer parses and drives the same
//!   §5.7.6 / §5.7.7 synthesis as on I-frames.
//!
//! Refs ETSI TS 103 190-1 §4.2.6.6 Table 25, §4.2.12.3/4 Tables 51/52,
//! §4.3.3.7.8 (`b_iframe`), §4.3.3.2.7 (`b_iframe_global`).

use oxideav_ac4::asf::{walk_ac4_substream_sticky, StickyConfig};
use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

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
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Multi-partial carrier so the A-SPX layer has real HF content.
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

/// Build the live ACPL_3 substream body directly, I- or P-frame form.
fn build_body(b_iframe: bool) -> Vec<u8> {
    let cfg = live_cfg();
    let counts = oxideav_ac4::aspx::derive_aspx_frequency_tables(&cfg, 0)
        .expect("freq tables")
        .counts;
    let sig: Vec<i32> = (0..counts.num_sbg_sig_highres as i32)
        .map(|i| 4 - (i % 3))
        .collect();
    let noise: Vec<i32> = vec![2; counts.num_sbg_noise as usize];
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna(
        1920,
        40,
        None,
        b_iframe,
        &l,
        &r,
        Some(&c),
        Some(&ls),
        Some(&rs),
        None,
        &cfg,
        &sig,
        &noise,
        &sig,
        &noise,
        &[],
        &[],
        &[],
        3,
        oxideav_ac4::acpl::AcplQuantMode::Fine,
        oxideav_ac4::acpl::AcplQuantMode::Fine,
        0.5,
        0.1,
        1.0,
        1.0,
        8192,
    )
}

fn decode_one(dec: &mut Ac4Decoder, bytes: Vec<u8>, _pts: i64) -> Vec<i16> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    let raw = &af.data[0];
    raw.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// An I,P,P sequence from the live 5.0 ACPL_3 real-ASPX entry point
/// decodes end-to-end: the P frames are signalled as non-I-frames in
/// the TOC and still produce nonsilent 5-channel PCM.
#[test]
fn i_p_sequence_decodes_five_channels() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let chans: [&[f32]; 5] = [&l, &r, &c, &ls, &rs];

    // Frame 0: I-frame.
    enc.b_iframe_global = true;
    let f0 = enc.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
    let pcm0 = decode_one(&mut dec, f0, 0);
    assert!(dec.last_info.as_ref().unwrap().b_iframe_global);

    // Frames 1..3: P-frames.
    enc.b_iframe_global = false;
    for n in 1..3i64 {
        let fp = enc.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
        let pcm = decode_one(&mut dec, fp, n * 1920);
        let info = dec.last_info.as_ref().unwrap();
        assert!(!info.b_iframe_global, "P-frame must clear b_iframe_global");
        assert_eq!(info.channels, 5);
        let energy: i64 = pcm.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(energy > 0, "P-frame {n} must decode to nonsilent PCM");
        // The P-frame's A-SPX + A-CPL layer parsed: the substream walk
        // must surface both data elements.
        let sub = dec.last_substream.as_ref().expect("substream walked");
        assert!(
            sub.tools.aspx_data_sig_primary.is_some(),
            "P-frame aspx_data_2ch() must parse via the sticky config"
        );
        assert!(
            sub.tools.acpl_data_2ch.is_some(),
            "P-frame acpl_data_2ch() must parse via the sticky config"
        );
        assert!(
            sub.tools.acpl_config_2ch.is_some(),
            "P-frame tools must carry the seeded sticky acpl_config_2ch"
        );
    }
    let e0: i64 = pcm0.iter().map(|&s| (s as i64) * (s as i64)).sum();
    assert!(e0 > 0);
}

/// The P-frame body differs from the I-frame body (configs + xover are
/// omitted) while both decode; a P-frame walked *without* the sticky
/// state has no A-SPX / A-CPL layer.
#[test]
fn p_frame_body_walks_only_with_sticky_state() {
    let body_i = build_body(true);
    let body_p = build_body(false);
    assert_ne!(body_i, body_p, "P-frame body must differ from I-frame");

    // I-frame walk harvests the sticky configs.
    let mut sticky = StickyConfig::default();
    let info_i = walk_ac4_substream_sticky(&body_i, 5, true, 1920, None, Some(&mut sticky))
        .expect("I-frame walk");
    assert!(info_i.tools.aspx_data_sig_primary.is_some());
    assert!(info_i.tools.acpl_data_2ch.is_some());
    assert!(sticky.aspx_config.is_some(), "sticky harvested aspx_config");
    assert_eq!(sticky.aspx_xover, Some(0), "sticky harvested xover");
    assert!(sticky.acpl_config_2ch.is_some());

    // P-frame walk with the sticky state recovers the identical
    // envelope + A-CPL payloads (same inputs on both frames).
    let info_p = walk_ac4_substream_sticky(&body_p, 5, false, 1920, None, Some(&mut sticky))
        .expect("P-frame walk");
    assert_eq!(
        info_i.tools.aspx_framing_primary,
        info_p.tools.aspx_framing_primary
    );
    assert_eq!(
        info_i.tools.aspx_data_sig_primary,
        info_p.tools.aspx_data_sig_primary
    );
    assert_eq!(
        info_i.tools.aspx_data_noise_primary,
        info_p.tools.aspx_data_noise_primary
    );
    let acpl_i = info_i.tools.acpl_data_2ch.as_ref().expect("I acpl");
    let acpl_p = info_p.tools.acpl_data_2ch.as_ref().expect("P acpl");
    assert_eq!(acpl_i.alpha1, acpl_p.alpha1);
    assert_eq!(acpl_i.gamma5, acpl_p.gamma5);

    // Without the sticky state the P-frame has no config in scope —
    // the A-SPX / A-CPL layer must stay unparsed (clean bail).
    let info_bare =
        walk_ac4_substream_sticky(&body_p, 5, false, 1920, None, None).expect("bare walk");
    assert!(info_bare.tools.aspx_data_sig_primary.is_none());
    assert!(info_bare.tools.acpl_data_2ch.is_none());
}

/// Decoding a P-frame *after* its I-frame produces different (richer)
/// PCM than decoding the same P-frame cold on a fresh decoder — the
/// sticky configs change the decoded output, not just the parse.
#[test]
fn sticky_state_changes_p_frame_pcm() {
    let params = CodecParameters::audio(CodecId::new("ac4"));

    let mut enc = Ac4ImsEncoder::new();
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let chans: [&[f32]; 5] = [&l, &r, &c, &ls, &rs];
    enc.b_iframe_global = true;
    let f_i = enc.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
    enc.b_iframe_global = false;
    let f_p = enc.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);

    // Warm decode: I then P.
    let mut dec_warm = Ac4Decoder::new(&params);
    let _ = decode_one(&mut dec_warm, f_i, 0);
    let pcm_warm = decode_one(&mut dec_warm, f_p.clone(), 1920);

    // Cold decode: the P-frame alone on a fresh decoder (no sticky).
    let mut dec_cold = Ac4Decoder::new(&params);
    let pcm_cold = decode_one(&mut dec_cold, f_p, 1920);

    assert_ne!(
        pcm_warm, pcm_cold,
        "the I-frame's sticky configs must influence the P-frame's decoded PCM"
    );
}

/// P-frame emission is deterministic across fresh encoder instances.
#[test]
fn p_frame_emission_is_deterministic() {
    let run = || {
        let mut enc = Ac4ImsEncoder::new();
        let l = multitone(0.3);
        let r = multitone(0.25);
        let c = tone(660.0, 0.2);
        let ls = tone(880.0, 0.2);
        let rs = tone(1100.0, 0.2);
        let chans: [&[f32]; 5] = [&l, &r, &c, &ls, &rs];
        enc.b_iframe_global = true;
        let f_i = enc.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
        enc.b_iframe_global = false;
        let f_p = enc.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
        (f_i, f_p)
    };
    let (i1, p1) = run();
    let (i2, p2) = run();
    assert_eq!(i1, i2);
    assert_eq!(p1, p2);
    assert_ne!(i1, p1, "I and P frames must differ on the wire");
}
