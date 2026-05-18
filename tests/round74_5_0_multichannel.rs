//! Round 74 — 5.0 SIMPLE/ASF Cfg3Five multichannel forward analysis tests.
//!
//! These tests exercise the new
//! [`oxideav_ac4::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0`] entry
//! point end-to-end through [`oxideav_ac4::decoder::Ac4Decoder`]. The
//! encoder emits an IMS bitstream_version = 2 TOC with channel_mode prefix
//! `0b1101` (4 b — channel_mode 3 → 5.0 surround) followed by a
//! `5_X_channel_element` substream body for
//! `5_X_codec_mode = SIMPLE, coding_config = 3 (Cfg3Five)` per ETSI
//! TS 103 190-1 §4.2.6.6 Table 25 + §4.2.7.5 Table 29.
//!
//! Per the round-74 brief the SNR target is ≥ 20 dB spectral SNR per channel
//! on tone fixtures (matching the round-51 stereo Path A SNR floor — the
//! encoder uses the same per-channel forward pipeline so the per-channel SNR
//! is bounded by the same quantiser/codebook constraints).

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::encoder_mdct::EncoderMdctState;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const CHANNELS: usize = 5;

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

/// Helper: encode + decode 5.0 frames through the full pipeline, returning
/// per-frame deinterleaved S16 PCM (one Vec<i16> per output channel).
fn encode_decode_5_0_frames(frames_per_channel: &[Vec<Vec<f32>>; 5]) -> Vec<[Vec<i16>; 5]> {
    let n_frames = frames_per_channel[0].len();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let mut out: Vec<[Vec<i16>; 5]> = Vec::with_capacity(n_frames);
    // Iterate frame-major over the per-channel frame vectors by zipping all
    // five channels together. Each iteration yields one (L, R, C, Ls, Rs)
    // tuple of frame-slices to feed to the encoder.
    let frame_tuples = (0..n_frames).map(|i| {
        (
            frames_per_channel[0][i].as_slice(),
            frames_per_channel[1][i].as_slice(),
            frames_per_channel[2][i].as_slice(),
            frames_per_channel[3][i].as_slice(),
            frames_per_channel[4][i].as_slice(),
        )
    });
    for (f0, f1, f2, f3, f4) in frame_tuples {
        let bytes = enc.encode_frame_pcm_5_0(&[f0, f1, f2, f3, f4]);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, 1920);
        assert_eq!(af.data.len(), 1);
        // 5.0 S16 interleaved: 1920 samples × 5 ch × 2 bytes.
        assert_eq!(af.data[0].len(), 1920 * CHANNELS * 2);
        let buf = &af.data[0];
        let mut per_ch: [Vec<i16>; 5] = std::array::from_fn(|_| Vec::with_capacity(N));
        for i in 0..N {
            for (ch, ch_vec) in per_ch.iter_mut().enumerate() {
                let off = (i * CHANNELS + ch) * 2;
                ch_vec.push(i16::from_le_bytes([buf[off], buf[off + 1]]));
            }
        }
        out.push(per_ch);
    }
    out
}

#[test]
fn round74_5_0_encoder_produces_5channel_layout_pcm() {
    let frames_per_channel: [Vec<Vec<f32>>; 5] =
        std::array::from_fn(|_| (0..2).map(|_| vec![0.0_f32; N]).collect());
    let decoded = encode_decode_5_0_frames(&frames_per_channel);
    for f in &decoded {
        for ch_vec in f.iter() {
            assert_eq!(ch_vec.len(), N);
        }
    }
}

#[test]
fn round74_5_0_encoder_bumps_sequence_counter() {
    let mut enc = Ac4ImsEncoder::new();
    assert_eq!(enc.sequence_counter, 0);
    let z = vec![0.0_f32; N];
    let _ = enc.encode_frame_pcm_5_0(&[
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
    ]);
    assert_eq!(enc.sequence_counter, 1);
    let _ = enc.encode_frame_pcm_5_0(&[
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
    ]);
    assert_eq!(enc.sequence_counter, 2);
}

#[test]
fn round74_5_0_encoder_toc_declares_5_channels() {
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0_f32; N];
    let bytes = enc.encode_frame_pcm_5_0(&[
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
        z.as_slice(),
    ]);
    let info = oxideav_ac4::toc::parse_ac4_toc(&bytes).expect("parse_ac4_toc");
    assert_eq!(info.channels, 5, "TOC must declare 5 channels for 5.0");
    assert_eq!(info.bitstream_version, 2);
    assert!(info.b_iframe_global);
}

#[test]
fn round74_5_0_independent_tones_per_channel_round_trip_with_distinct_audio() {
    // Drive each channel with a different pure tone so we can sanity-check
    // that the decoder reconstructs each independently. The 5.0 SIMPLE
    // path has no joint-MDCT mixing, so each output channel should reflect
    // only its input channel's content (no cross-channel bleed).
    let freqs: [f32; 5] = [220.0, 440.0, 660.0, 880.0, 1100.0];
    let frames_per_channel: [Vec<Vec<f32>>; 5] =
        std::array::from_fn(|ch| (0..3).map(|i| make_tone_frame(freqs[ch], i * N)).collect());
    let decoded = encode_decode_5_0_frames(&frames_per_channel);
    // Steady-state frame: index 2.
    let f = &decoded[2];
    for (ch, ch_vec) in f.iter().enumerate() {
        let nz = ch_vec.iter().filter(|&&s| s != 0).count();
        let peak = ch_vec.iter().map(|&s| s.abs()).max().unwrap_or(0);
        assert!(nz > 100, "channel {ch} too few non-zero samples: {nz}");
        assert!(peak > 1000, "channel {ch} peak too low: {peak}");
    }
    // Distinct-content sanity: any two channels should differ on a
    // substantial number of samples (independent tones produce different
    // waveforms in the time domain).
    let differs_l_r = f[0]
        .iter()
        .zip(f[1].iter())
        .filter(|(a, b)| (a.saturating_sub(**b)).abs() > 100)
        .count();
    assert!(
        differs_l_r > 100,
        "L (220 Hz) and R (440 Hz) reconstructed PCM should differ ({differs_l_r} samples differ by >100)"
    );
}

#[test]
fn round74_5_0_per_channel_spectral_snr_exceeds_20db() {
    // Match the round-51 stereo Path A SNR target — the encoder runs the
    // same per-channel forward pipeline (KBD-windowed MDCT + per-band
    // scalefactor + DP-optimal section partition + HCB1..11 codebook
    // selection + SNF emission) independently per channel, so per-channel
    // spectral SNR should reach ≥ 20 dB on a steady-state pure-tone fixture
    // with amplitude 0.3 — bounded by the same quantiser/codebook
    // constraints as the stereo Path A baseline.
    //
    // The five channels are driven with independent pure tones (220 / 440 /
    // 660 / 880 / 1100 Hz). Spectral SNR is measured by mirroring the
    // encoder's forward MDCT over the input PCM and comparing the input
    // MDCT spectrum bin-for-bin against the decoder's reconstructed
    // `scaled_spec_per_channel[i]` — same convention as the round-49
    // white-noise test and the round-51 stereo SNR tests.
    let freqs: [f32; 5] = [220.0, 440.0, 660.0, 880.0, 1100.0];
    let frames_per_channel: [Vec<Vec<f32>>; 5] =
        std::array::from_fn(|ch| (0..3).map(|i| make_tone_frame(freqs[ch], i * N)).collect());
    // Mirror the encoder's MDCT on each input channel.
    let mut input_mdcts: [EncoderMdctState; 5] =
        std::array::from_fn(|_| EncoderMdctState::new(N as u32));
    let mut last_input_spec: [Option<Vec<f32>>; 5] = std::array::from_fn(|_| None);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let mut last_recon_spec: [Option<Vec<f32>>; 5] = std::array::from_fn(|_| None);
    let n_frames = frames_per_channel[0].len();
    let frame_tuples = (0..n_frames).map(|i| {
        [
            frames_per_channel[0][i].as_slice(),
            frames_per_channel[1][i].as_slice(),
            frames_per_channel[2][i].as_slice(),
            frames_per_channel[3][i].as_slice(),
            frames_per_channel[4][i].as_slice(),
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
        let bytes = enc.encode_frame_pcm_5_0(&channel_slices);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let _ = dec.receive_frame().expect("receive_frame");
        let sub = dec.last_substream.as_ref().expect("substream parsed");
        let five = sub
            .tools
            .five_channel_data
            .as_ref()
            .expect("five_channel_data must be populated for SIMPLE/Cfg3Five");
        for (recon_slot, parsed) in last_recon_spec
            .iter_mut()
            .zip(five.scaled_spec_per_channel.iter())
        {
            *recon_slot = parsed.clone();
        }
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
    let snrs: [f64; 5] = std::array::from_fn(|ch| {
        let orig = last_input_spec[ch].as_ref().unwrap();
        let recon = last_recon_spec[ch].as_ref().expect("recon spec missing");
        snr(orig, recon)
    });
    eprintln!(
        "ROUND-74 5.0 per-channel spectral SNR (220/440/660/880/1100 Hz): \
         L={:.1} R={:.1} C={:.1} Ls={:.1} Rs={:.1} dB",
        snrs[0], snrs[1], snrs[2], snrs[3], snrs[4]
    );
    for (ch, &snr_val) in snrs.iter().enumerate() {
        assert!(
            snr_val > 20.0,
            "channel {ch} spectral SNR too low: {snr_val:.1} dB (expected > 20 dB)"
        );
    }
}

#[test]
fn round74_5_0_substream_parses_via_walk_ac4_substream() {
    // Encode a 5.0 frame, peel off the TOC byte-length, and run the
    // substream bytes through `walk_ac4_substream` directly. The walker
    // should route to `parse_5x_audio_data_outer` (channels == 5,
    // b_has_lfe = false) and populate `tools.five_x_mode` /
    // `tools.five_x_coding_config` / `tools.five_channel_data`.
    let mut enc = Ac4ImsEncoder::new();
    let frames: [Vec<f32>; 5] = std::array::from_fn(|ch| {
        let freq = 100.0 + 200.0 * ch as f32;
        make_tone_frame(freq, 0)
    });
    let bytes = enc.encode_frame_pcm_5_0(&[
        frames[0].as_slice(),
        frames[1].as_slice(),
        frames[2].as_slice(),
        frames[3].as_slice(),
        frames[4].as_slice(),
    ]);
    let info = oxideav_ac4::toc::parse_ac4_toc(&bytes).expect("parse_ac4_toc");
    assert_eq!(info.channels, 5);
    // Substream starts at `toc_size + payload_base` per the
    // `Ac4FrameInfo` doc comment on `toc_size`.
    let sub_start = (info.toc_size + info.payload_base) as usize;
    let sub_bytes = &bytes[sub_start..];
    let sub_info = oxideav_ac4::asf::walk_ac4_substream(sub_bytes, 5, true, info.frame_length)
        .expect("walk_ac4_substream");
    assert!(matches!(
        sub_info.tools.five_x_mode,
        Some(oxideav_ac4::mch::FiveXCodecMode::Simple)
    ));
    assert!(matches!(
        sub_info.tools.five_x_coding_config,
        Some(oxideav_ac4::mch::FiveXCodingConfig::Cfg3Five)
    ));
    let five = sub_info
        .tools
        .five_channel_data
        .as_ref()
        .expect("five_channel_data populated");
    // All five per-channel scaled spectra should be populated (long-frame
    // single-window-group case — every channel's sf_data(ASF) body parsed
    // cleanly).
    for (ch, scaled) in five.scaled_spec_per_channel.iter().enumerate() {
        assert!(
            scaled.is_some(),
            "channel {ch} scaled spectrum missing — sf_data(ASF) body did not parse"
        );
    }
}

#[test]
fn round74_5_0_silence_round_trips_to_silence() {
    let z = vec![0.0_f32; N];
    let frames_per_channel: [Vec<Vec<f32>>; 5] =
        std::array::from_fn(|_| (0..3).map(|_| z.clone()).collect());
    let decoded = encode_decode_5_0_frames(&frames_per_channel);
    // Steady-state frame must be effectively silent on every channel.
    let f = &decoded[2];
    for (ch, ch_vec) in f.iter().enumerate() {
        let peak = ch_vec.iter().map(|&s| s.abs()).max().unwrap_or(0);
        // i16 peak < 50 = -56 dBFS; comfortably below any audible threshold.
        assert!(
            peak < 50,
            "channel {ch} silent reconstruction failed, got peak amplitude {peak}"
        );
    }
}
