//! Round 91 — 7.1 (3/4/0.1) SIMPLE/ASF Cfg3Five multichannel forward
//! analysis tests.
//!
//! Exercises
//! [`oxideav_ac4::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_7_1`]
//! end-to-end through [`oxideav_ac4::decoder::Ac4Decoder`]. The encoder
//! emits an IMS bitstream_version = 2 TOC with channel_mode prefix
//! `0b1111001` (7 b — Table 88 channel_mode 6 → 8 channels, layout
//! `7.1: 3/4/0.1 (L, C, R, Ls, Rs, Lb, Rb, LFE)`) followed by a
//! `7_X_channel_element(b_has_lfe = 1)` substream body for
//! `7_X_codec_mode = SIMPLE, coding_config = 3 (Cfg3Five)` per ETSI
//! TS 103 190-1 §4.2.6.14 Table 33 + §4.2.7.5 Table 29 + §4.2.7.4
//! Table 26 (additional-channel `two_channel_data()`) + §4.2.8 Table 35
//! (LFE `sf_info_lfe()`).
//!
//! Per the round-91 brief the SNR target is ≥ 20 dB spectral SNR per
//! non-LFE channel (matching the round-74 5.0 / round-80 5.1 SNR floor —
//! the encoder reuses the same per-channel forward pipeline). LFE is
//! bounded to `max_sfb_lfe ≤ 7` at `tl = 1920` per Table 106 column 4
//! so its coverage is narrower than the non-LFE channels.
//!
//! Note on slot order: the decoder's `dispatch_5x_cfg3_simple_aspx`
//! (round 39) lays the inner `five_channel_data()` 5 SCEs into slots
//! 0..4 as `[L, R, C, Ls, Rs]` per Table 180 — that's the internal
//! coding order; the surface Table 88 ordering of channel_mode 6 is
//! `(L, C, R, Ls, Rs, Lb, Rb, LFE)`. The encoder's input slice order
//! follows the decoder slot convention: `[L, R, C, Ls, Rs, Lb, Rb,
//! LFE]`. The decoder's `dispatch_7x_additional_channel_pair` then
//! routes the additional `two_channel_data()` pair Lb/Rb directly into
//! slots 5/6 (identity-SAP path with `b_use_sap_add_ch = 0`); the LFE
//! `mono_data(1)` lands in slot 7 via the shared 5.1/7.1 LFE render in
//! `receive_frame`.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const CHANNELS: usize = 8;

/// Build a pure-tone PCM frame at `freq` Hz starting at sample index
/// `start` with amplitude 0.3.
fn make_tone_frame(freq: f32, start: usize) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = (start + i) as f32 / FS;
            0.3 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Helper: encode + decode 7.1 frames through the full pipeline,
/// returning per-frame deinterleaved S16 PCM (one Vec<i16> per output
/// channel).
fn encode_decode_7_1_frames(frames_per_channel: &[Vec<Vec<f32>>; 8]) -> Vec<[Vec<i16>; 8]> {
    let n_frames = frames_per_channel[0].len();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let mut out: Vec<[Vec<i16>; 8]> = Vec::with_capacity(n_frames);
    let frame_tuples = (0..n_frames).map(|i| {
        [
            frames_per_channel[0][i].as_slice(),
            frames_per_channel[1][i].as_slice(),
            frames_per_channel[2][i].as_slice(),
            frames_per_channel[3][i].as_slice(),
            frames_per_channel[4][i].as_slice(),
            frames_per_channel[5][i].as_slice(),
            frames_per_channel[6][i].as_slice(),
            frames_per_channel[7][i].as_slice(),
        ]
    });
    for channel_slices in frame_tuples {
        let bytes = enc.encode_frame_pcm_7_1(&channel_slices);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, 1920);
        assert_eq!(af.data.len(), 1);
        // 7.1 S16 interleaved: 1920 samples × 8 ch × 2 bytes.
        assert_eq!(af.data[0].len(), 1920 * CHANNELS * 2);
        let buf = &af.data[0];
        let mut per_ch: [Vec<i16>; 8] = std::array::from_fn(|_| Vec::with_capacity(N));
        for j in 0..N {
            for (ch, ch_vec) in per_ch.iter_mut().enumerate() {
                let off = (j * CHANNELS + ch) * 2;
                ch_vec.push(i16::from_le_bytes([buf[off], buf[off + 1]]));
            }
        }
        out.push(per_ch);
    }
    out
}

#[test]
fn round91_7_1_encoder_produces_8channel_layout_pcm() {
    let frames_per_channel: [Vec<Vec<f32>>; 8] =
        std::array::from_fn(|_| (0..2).map(|_| vec![0.0_f32; N]).collect());
    let decoded = encode_decode_7_1_frames(&frames_per_channel);
    for f in &decoded {
        for ch_vec in f.iter() {
            assert_eq!(ch_vec.len(), N);
        }
    }
}

#[test]
fn round91_7_1_encoder_toc_declares_8_channels() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0_f32; N];
    let slices: [&[f32]; 8] = [&z, &z, &z, &z, &z, &z, &z, &z];
    let bytes = enc.encode_frame_pcm_7_1(&slices);
    let info = oxideav_ac4::toc::parse_ac4_toc(&bytes).expect("parse_ac4_toc");
    assert_eq!(info.channels, 8, "TOC must declare 8 channels for 7.1");
    assert_eq!(info.bitstream_version, 2);
    assert!(info.b_iframe_global);
}

#[test]
fn round91_7_1_encoder_bumps_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    assert_eq!(enc.sequence_counter, 0);
    let z = vec![0.0_f32; N];
    let slices: [&[f32]; 8] = [&z, &z, &z, &z, &z, &z, &z, &z];
    let _ = enc.encode_frame_pcm_7_1(&slices);
    assert_eq!(enc.sequence_counter, 1);
    let _ = enc.encode_frame_pcm_7_1(&slices);
    assert_eq!(enc.sequence_counter, 2);
}

#[test]
fn round91_7_1_with_builder_sets_channel_mode_prefix_for_7_1() {
    // Smoke-check `with_7_1()`: encoder default frame must declare 8
    // channels even via the v0 path. The encoder uses the
    // `0b1111001` 7-bit prefix (Table 88 channel_mode 6).
    let mut enc = Ac4ImsEncoder::new().with_v0().with_7_1();
    let frame = enc.encode_frame(64);
    let info = oxideav_ac4::toc::parse_ac4_toc(&frame).expect("v0 7.1 TOC must parse");
    assert_eq!(info.channels, 8, "with_7_1() must declare 8 channels");
}

#[test]
fn round91_7_1_substream_walker_sees_b_has_lfe_and_additional_pair() {
    // Encode a 7.1 frame and run the substream bytes through
    // `walk_ac4_substream` directly. The 7_X walker should:
    //   * set `seven_x_b_has_lfe = true` (b_has_lfe gate fires for
    //     channel_mode 6 → 8 channels),
    //   * populate `lfe_mono_data` with a `b_lfe = true` shell and a
    //     non-empty `scaled_spec`,
    //   * populate `five_channel_data` (Cfg3Five) with five SCEs each
    //     carrying a non-empty `scaled_spec`,
    //   * record `seven_x_b_use_sap_add_ch = false` and populate
    //     `seven_x_additional_channel_data` with two `scaled_spec`
    //     entries for the Lb/Rb pair.
    let mut enc = Ac4ImsEncoder::new();
    let frames: [Vec<f32>; 8] = std::array::from_fn(|ch| {
        let freq = 100.0 + 150.0 * ch as f32;
        make_tone_frame(freq, 0)
    });
    let slices: [&[f32]; 8] = [
        &frames[0], &frames[1], &frames[2], &frames[3], &frames[4], &frames[5], &frames[6],
        &frames[7],
    ];
    let bytes = enc.encode_frame_pcm_7_1(&slices);
    let info = oxideav_ac4::toc::parse_ac4_toc(&bytes).expect("parse_ac4_toc");
    assert_eq!(info.channels, 8);
    let sub_start = (info.toc_size + info.payload_base) as usize;
    let sub_bytes = &bytes[sub_start..];
    let sub_info = oxideav_ac4::asf::walk_ac4_substream(sub_bytes, 8, true, info.frame_length)
        .expect("walk_ac4_substream");
    assert!(
        sub_info.tools.seven_x_b_has_lfe,
        "7_X walker must set b_has_lfe = true for 7.1"
    );
    assert!(matches!(
        sub_info.tools.seven_x_mode,
        Some(oxideav_ac4::mch::SevenXCodecMode::Simple)
    ));
    assert!(matches!(
        sub_info.tools.seven_x_coding_config,
        Some(oxideav_ac4::mch::FiveXCodingConfig::Cfg3Five)
    ));
    // LFE walked.
    let lfe = sub_info
        .tools
        .lfe_mono_data
        .as_ref()
        .expect("LFE mono_data(1) must be populated for 7.1");
    assert!(lfe.b_lfe, "lfe_mono_data.b_lfe must be true");
    assert!(
        lfe.scaled_spec.is_some(),
        "LFE sf_data(ASF) body must decode into scaled_spec"
    );
    // Inner five_channel_data walked.
    let five = sub_info
        .tools
        .five_channel_data
        .as_ref()
        .expect("five_channel_data populated");
    for (ch, scaled) in five.scaled_spec_per_channel.iter().enumerate() {
        assert!(
            scaled.is_some(),
            "non-LFE channel {ch} scaled spectrum missing"
        );
    }
    // Additional pair walked, identity-SAP path.
    assert_eq!(
        sub_info.tools.seven_x_b_use_sap_add_ch,
        Some(false),
        "encoder emits identity SAP (b_use_sap_add_ch = 0)"
    );
    let add = sub_info
        .tools
        .seven_x_additional_channel_data
        .as_ref()
        .expect("additional two_channel_data populated");
    assert_eq!(add.scaled_spec_per_channel.len(), 2);
    for (ch, scaled) in add.scaled_spec_per_channel.iter().enumerate() {
        assert!(
            scaled.is_some(),
            "additional-pair channel {ch} (Lb/Rb) scaled spectrum missing"
        );
    }
}

#[test]
fn round91_7_1_independent_tones_per_channel_round_trip() {
    // Drive each of the seven non-LFE channels with a distinct pure
    // tone. LFE gets a low (60 Hz) tone within its narrow band budget.
    // The SIMPLE path has no joint-MDCT mixing (identity SAP on both
    // five_channel_data and the additional two_channel_data) so each
    // output channel should reflect only its input channel's content
    // (no cross-channel bleed).
    let freqs: [f32; 8] = [220.0, 440.0, 660.0, 880.0, 1100.0, 1320.0, 1540.0, 60.0];
    let frames_per_channel: [Vec<Vec<f32>>; 8] =
        std::array::from_fn(|ch| (0..3).map(|i| make_tone_frame(freqs[ch], i * N)).collect());
    let decoded = encode_decode_7_1_frames(&frames_per_channel);
    let f = &decoded[2];
    // Non-LFE channels must all carry audible reconstructed PCM.
    for (ch, ch_vec) in f.iter().enumerate().take(7) {
        let nz = ch_vec.iter().filter(|&&s| s != 0).count();
        let peak = ch_vec.iter().map(|&s| s.abs()).max().unwrap_or(0);
        assert!(
            nz > 100,
            "non-LFE channel {ch} too few non-zero samples: {nz}"
        );
        assert!(peak > 1000, "non-LFE channel {ch} peak too low: {peak}");
    }
    // LFE: at tl=1920 with max_sfb_lfe=7, sfb 0..7 covers bins 0..28
    // (≈0..350 Hz). 60 Hz lives in sfb 0; the reconstructed LFE
    // channel should be non-silent.
    let lfe = &f[7];
    let lfe_nz = lfe.iter().filter(|&&s| s != 0).count();
    let lfe_peak = lfe.iter().map(|&s| s.abs()).max().unwrap_or(0);
    assert!(
        lfe_nz > 50,
        "LFE channel too few non-zero samples: {lfe_nz}"
    );
    assert!(lfe_peak > 100, "LFE channel peak too low: {lfe_peak}");
    // L (220 Hz) vs R (440 Hz) should differ on many samples.
    let differs_l_r = f[0]
        .iter()
        .zip(f[1].iter())
        .filter(|(a, b)| (a.saturating_sub(**b)).abs() > 100)
        .count();
    assert!(
        differs_l_r > 100,
        "L (220 Hz) and R (440 Hz) should differ ({differs_l_r} samples differ by >100)"
    );
    // Lb (1320 Hz) vs Rb (1540 Hz) should differ on many samples too.
    let differs_lb_rb = f[5]
        .iter()
        .zip(f[6].iter())
        .filter(|(a, b)| (a.saturating_sub(**b)).abs() > 100)
        .count();
    assert!(
        differs_lb_rb > 100,
        "Lb (1320 Hz) and Rb (1540 Hz) should differ ({differs_lb_rb} samples differ by >100)"
    );
}

#[test]
fn round91_7_1_silence_round_trips_to_silence() {
    let z = vec![0.0_f32; N];
    let frames_per_channel: [Vec<Vec<f32>>; 8] =
        std::array::from_fn(|_| (0..3).map(|_| z.clone()).collect());
    let decoded = encode_decode_7_1_frames(&frames_per_channel);
    let f = &decoded[2];
    for (ch, ch_vec) in f.iter().enumerate() {
        let peak = ch_vec.iter().map(|&s| s.abs()).max().unwrap_or(0);
        assert!(
            peak < 50,
            "channel {ch} silent reconstruction failed, got peak amplitude {peak}"
        );
    }
}

#[test]
fn round91_7_1_per_channel_spectral_snr_non_lfe_exceeds_20db() {
    use oxideav_ac4::encoder_mdct::EncoderMdctState;

    // Mirror the encoder's MDCT on each input channel and compare to
    // the decoder's reconstructed scaled spectra — same convention as
    // the round-74 5.0 / round-80 5.1 SNR tests. The first five
    // channels' spectra come from `five_channel_data.scaled_spec_per_channel`;
    // the additional pair (Lb, Rb) come from
    // `seven_x_additional_channel_data.scaled_spec_per_channel`.
    let freqs: [f32; 8] = [220.0, 440.0, 660.0, 880.0, 1100.0, 1320.0, 1540.0, 60.0];
    let frames_per_channel: [Vec<Vec<f32>>; 8] =
        std::array::from_fn(|ch| (0..3).map(|i| make_tone_frame(freqs[ch], i * N)).collect());
    let mut input_mdcts: [EncoderMdctState; 8] =
        std::array::from_fn(|_| EncoderMdctState::new(N as u32));
    let mut last_input_spec: [Option<Vec<f32>>; 8] = std::array::from_fn(|_| None);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let mut last_recon_spec: [Option<Vec<f32>>; 7] = std::array::from_fn(|_| None);
    let n_frames = frames_per_channel[0].len();
    let frame_tuples = (0..n_frames).map(|i| {
        [
            frames_per_channel[0][i].as_slice(),
            frames_per_channel[1][i].as_slice(),
            frames_per_channel[2][i].as_slice(),
            frames_per_channel[3][i].as_slice(),
            frames_per_channel[4][i].as_slice(),
            frames_per_channel[5][i].as_slice(),
            frames_per_channel[6][i].as_slice(),
            frames_per_channel[7][i].as_slice(),
        ]
    });
    for channel_slices in frame_tuples {
        for ((mdct, slice), input_spec_slot) in input_mdcts
            .iter_mut()
            .zip(channel_slices.iter())
            .zip(last_input_spec.iter_mut())
        {
            *input_spec_slot = Some(mdct.analyse_frame(slice));
        }
        let bytes = enc.encode_frame_pcm_7_1(&channel_slices);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let _ = dec.receive_frame().expect("receive_frame");
        let sub = dec.last_substream.as_ref().expect("substream parsed");
        let five = sub
            .tools
            .five_channel_data
            .as_ref()
            .expect("five_channel_data populated for SIMPLE/Cfg3Five");
        for (i, recon_slot) in last_recon_spec.iter_mut().enumerate().take(5) {
            *recon_slot = five.scaled_spec_per_channel[i].clone();
        }
        let add = sub
            .tools
            .seven_x_additional_channel_data
            .as_ref()
            .expect("additional channel pair populated for SIMPLE/Cfg3Five");
        last_recon_spec[5] = add.scaled_spec_per_channel[0].clone();
        last_recon_spec[6] = add.scaled_spec_per_channel[1].clone();
    }
    let snr = |orig: &[f32], recon: &[f32]| -> f64 {
        let mut sig_e = 0.0_f64;
        let mut err_e = 0.0_f64;
        let n_compare = orig.len().min(recon.len());
        for k in 0..n_compare {
            let o = orig[k] as f64;
            let r = recon[k] as f64;
            sig_e += o * o;
            err_e += (o - r) * (o - r);
        }
        10.0 * (sig_e / err_e.max(1e-30)).log10()
    };
    let snrs: [f64; 7] = std::array::from_fn(|ch| {
        let orig = last_input_spec[ch].as_ref().unwrap();
        let recon = last_recon_spec[ch].as_ref().expect("recon spec missing");
        snr(orig, recon)
    });
    eprintln!(
        "ROUND-91 7.1 non-LFE spectral SNR (220/440/660/880/1100/1320/1540 Hz): \
         L={:.1} R={:.1} C={:.1} Ls={:.1} Rs={:.1} Lb={:.1} Rb={:.1} dB",
        snrs[0], snrs[1], snrs[2], snrs[3], snrs[4], snrs[5], snrs[6]
    );
    for (ch, &snr_val) in snrs.iter().enumerate() {
        assert!(
            snr_val > 20.0,
            "non-LFE channel {ch} spectral SNR too low: {snr_val:.1} dB (expected > 20 dB)"
        );
    }
}
