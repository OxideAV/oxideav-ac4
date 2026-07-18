//! Round 414 — the two remaining A-JOC downmix forms driven to object
//! PCM (TS 103 190-2 §6.2.3.4):
//!
//! * **`b_static_dmx == 1`** — the 5-signal 5.X core
//!   (`audio_data_chan(5.0/5.1)`, SIMPLE / Cfg3Five) feeds the
//!   §5.7.3.6 spatial reconstruction in `[L, R, C, Ls, Rs]` order.
//! * **A-SPX `var_channel_element`** (`var_codec_mode == 1`) — each
//!   downmix channel is bandwidth-extended in the QMF domain from its
//!   captured `aspx_data_2ch()` payload before the reconstruction,
//!   with the I-frame-sticky `aspx_config` / xover carried across
//!   P-frames.

use oxideav_ac4::ajoc::{AjocCtrlInfo, AjocDataPointInfo, AjocQuantMode};
use oxideav_ac4::ajoc_data::new_ajoc_diff_state;
use oxideav_ac4::ajoc_substream::{
    encode_ajoc_raw_frame_aspx, encode_ajoc_raw_frame_static, AjocBodyParams,
};
use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ajoc::AjocQuantMatrices;
use oxideav_ac4::oamd::ObjType;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const MAX_SFB: u32 = 20;

fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
    let sfbo = oxideav_ac4::sfb_offset::sfb_offset_48(N as u32).unwrap();
    let end = sfbo[MAX_SFB as usize] as usize;
    let mut v = vec![0.0f32; end];
    v[bin.min(end - 1)] = amp;
    v
}

/// Selector control info: object `o` reconstructs from downmix
/// channel `o % num_dmx` with unit dry gain and zero wet gain.
fn selector_setup(
    num_dmx: usize,
    num_umx: usize,
    num_decorr: usize,
) -> (AjocCtrlInfo, AjocQuantMatrices) {
    let ctrl = AjocCtrlInfo {
        decorr_enable: vec![true; num_decorr],
        object_present: vec![true; num_umx],
        data_point_info: AjocDataPointInfo {
            num_dpoints: 1,
            start_pos: vec![0],
            ramp_len: vec![16],
        },
        num_bands_code: vec![7; num_umx],
        num_bands: vec![1; num_umx],
        quant_select: vec![AjocQuantMode::Fine; num_umx],
        sparse_select: vec![false; num_umx],
        mix_mtx_dry_present: vec![vec![true; num_dmx]; num_umx],
        mix_mtx_wet_present: vec![vec![true; num_decorr]; num_umx],
    };
    let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
        .map(|o| {
            vec![(0..num_dmx)
                .map(|ch| vec![if ch == o % num_dmx { 1.0 } else { 0.0 }])
                .collect()]
        })
        .collect();
    let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
        .map(|_| vec![vec![vec![0.0]; num_decorr]])
        .collect();
    let qmats = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);
    (ctrl, qmats)
}

fn per_object_energy(buf: &[u8], num_ch: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; num_ch];
    for i in 0..N {
        for (c, e) in out.iter_mut().enumerate() {
            let off = (i * num_ch + c) * 2;
            let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as f64;
            *e += s * s;
        }
    }
    out
}

/// Static-downmix A-JOC frames decode end-to-end: the 5.X SIMPLE core
/// feeds the spatial reconstruction and the LFE of the 5.1 core lands
/// on the leading output slot.
#[test]
fn decoder_ajoc_static_dmx_core_to_object_pcm() {
    let num_dmx = 5usize;
    let num_umx = 3usize;
    let num_decorr = 1usize;
    let params = AjocBodyParams {
        b_lfe: true,
        b_static_dmx: true,
        n_fullband_dmx_signals: 5,
        n_fullband_upmix_signals: num_umx as u32,
        obj_type_dmx: vec![ObjType::Dyn; num_dmx + 1],
        obj_type_umx: vec![ObjType::Dyn; num_umx + 1],
    };
    let (ctrl, qmats) = selector_setup(num_dmx, num_umx, num_decorr);
    // Core [L, R, C, Ls, Rs] tones — objects 0/1/2 select L/R/C.
    let chan: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(12 + 8 * i, 40.0)).collect();
    let chan_refs: [&[f32]; 5] = [&chan[0], &chan[1], &chan[2], &chan[3], &chan[4]];
    let lfe = tone_spectrum(2, 25.0);

    let mut enc_state = new_ajoc_diff_state(num_umx, num_dmx, 7);
    let dec_params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&dec_params);
    let channels_out = num_umx + 1; // LFE first.
    let mut last = vec![0.0f64; channels_out];
    for seq in 0..3u32 {
        let frame = encode_ajoc_raw_frame_static(
            seq,
            &params,
            &chan_refs,
            Some((&lfe, 4)),
            MAX_SFB,
            num_decorr as u32,
            &ctrl,
            &qmats,
            true,
            &mut enc_state,
        )
        .unwrap();
        if seq == 0 {
            // The TOC descriptor round-trips the static form.
            let info = oxideav_ac4::toc::parse_ac4_toc(&frame).unwrap();
            let desc = &info.ajoc_substreams[0];
            assert!(desc.b_static_dmx, "descriptor signals b_static_dmx");
            assert_eq!(desc.n_fullband_dmx_signals, 5);
            assert!(desc.b_lfe);
        }
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected an audio frame");
        };
        assert_eq!(af.samples, N as u32);
        let buf = &af.data[0];
        assert_eq!(buf.len(), N * channels_out * 2);
        last = per_object_energy(buf, channels_out);
    }
    // LFE slot + every reconstructed object carry settled signal.
    assert!(last[0] > 1e4, "LFE slot silent: {last:?}");
    for (o, &e) in last.iter().enumerate().skip(1) {
        assert!(e > 1e4, "object {} silent: {last:?}", o - 1);
    }
    // Objects 0/1 select equal-amplitude core channels (L/R): the
    // settled energies match closely.
    let ratio = last[1] / last[2];
    assert!(
        (0.8..=1.25).contains(&ratio),
        "L/R-selecting objects diverge: ratio {ratio}"
    );
}

fn test_aspx_config() -> AspxConfig {
    AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        start_freq: 7,
        stop_freq: 0,
        master_freq_scale: AspxMasterFreqScale::LowRes,
        interpolation: false,
        preflat: false,
        limiter: false,
        noise_sbg: 0,
        num_env_bits_fixfix: 0,
        freq_res_mode: AspxFreqResMode::Low,
    }
}

/// A-SPX-downmix A-JOC frames decode end-to-end over an I + P + P GOP:
/// the QMF-domain extension runs per channel (I-frame configs sticky
/// across the P-frames) and the reconstruction stays stable.
#[test]
fn decoder_ajoc_aspx_downmix_to_object_pcm() {
    let num_dmx = 2usize;
    let num_umx = 2usize;
    let num_decorr = 1usize;
    let params = AjocBodyParams {
        b_lfe: false,
        b_static_dmx: false,
        n_fullband_dmx_signals: num_dmx as u32,
        n_fullband_upmix_signals: num_umx as u32,
        obj_type_dmx: vec![ObjType::Dyn; num_dmx],
        obj_type_umx: vec![ObjType::Dyn; num_umx],
    };
    let (ctrl, qmats) = selector_setup(num_dmx, num_umx, num_decorr);
    let s0 = tone_spectrum(24, 40.0);
    let s1 = tone_spectrum(60, 40.0);
    let spectra: Vec<&[f32]> = vec![&s0, &s1];
    let cfg = test_aspx_config();

    let mut enc_state = new_ajoc_diff_state(num_umx, num_dmx, 7);
    let dec_params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&dec_params);
    let mut energies: Vec<Vec<f64>> = Vec::new();
    for (seq, iframe) in [(0u32, true), (1, false), (2, false)] {
        let frame = encode_ajoc_raw_frame_aspx(
            seq,
            &params,
            &spectra,
            &cfg,
            None,
            MAX_SFB,
            num_decorr as u32,
            &ctrl,
            &qmats,
            iframe,
            &mut enc_state,
            None,
        )
        .unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected an audio frame");
        };
        assert_eq!(af.samples, N as u32);
        let buf = &af.data[0];
        assert_eq!(buf.len(), N * num_umx * 2);
        energies.push(per_object_energy(buf, num_umx));
    }
    // Settled frames: every object reconstructs its selected channel.
    for (fi, e) in energies.iter().enumerate().skip(1) {
        for (o, &v) in e.iter().enumerate() {
            assert!(v > 1e4, "frame {fi} object {o} silent: {v}");
        }
    }
    // P-frames (sticky aspx_config + xover) stay within 3 dB of the
    // preceding settled frame.
    for (o, (&e2, &e1)) in energies[2].iter().zip(energies[1].iter()).enumerate() {
        let ratio = e2 / e1.max(1.0);
        assert!(
            (0.5..=2.0).contains(&ratio),
            "object {o} P-frame energy unstable: ratio {ratio}"
        );
    }
}
