//! Round 453 — dialogue enhancement on **A-JOC object substreams**
//! (TS 103 190-2 §4.8.3.15 Table 14 / §5.8.2.3 full decoding /
//! §5.8.2.4 core decoding): `ajoc_dmx_de_data()` flags the
//! main-dialogue objects and carries their downmix coefficients; a
//! user gain G_DE boosts the dialogue object (full decoding —
//! Pseudocode 22 scales its dry / wet rows) or the downmix channels
//! the dialogue maps onto (core decoding — `y = H_M·H_A·x + x`),
//! sticky across P-frames (`b_dmx_de_cfg = 0`, `b_keep_dmx_de_coeffs`).

use oxideav_ac4::ajoc::{AjocCtrlInfo, AjocDataPointInfo, AjocDmxDeData, AjocQuantMode};
use oxideav_ac4::ajoc_data::new_ajoc_diff_state;
use oxideav_ac4::ajoc_substream::{encode_ajoc_raw_frame_with_dmx_de, AjocBodyParams};
use oxideav_ac4::decoder::{Ac4Decoder, DecodingMode};
use oxideav_ac4::encoder_ajoc::AjocQuantMatrices;
use oxideav_ac4::oamd::ObjType;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const MAX_SFB: u32 = 20;
const NUM_DMX: usize = 2;
const NUM_UMX: usize = 3;

fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
    let sfbo = oxideav_ac4::sfb_offset::sfb_offset_48(N as u32).unwrap();
    let end = sfbo[MAX_SFB as usize] as usize;
    let mut v = vec![0.0f32; end];
    v[bin.min(end - 1)] = amp;
    v
}

/// Objects 0 / 2 select downmix channel 0, object 1 selects channel 1.
fn selector_setup(num_decorr: usize) -> (AjocCtrlInfo, AjocQuantMatrices) {
    let ctrl = AjocCtrlInfo {
        decorr_enable: vec![true; num_decorr],
        object_present: vec![true; NUM_UMX],
        data_point_info: AjocDataPointInfo {
            num_dpoints: 1,
            start_pos: vec![0],
            ramp_len: vec![16],
        },
        num_bands_code: vec![7; NUM_UMX],
        num_bands: vec![1; NUM_UMX],
        quant_select: vec![AjocQuantMode::Fine; NUM_UMX],
        sparse_select: vec![false; NUM_UMX],
        mix_mtx_dry_present: vec![vec![true; NUM_DMX]; NUM_UMX],
        mix_mtx_wet_present: vec![vec![true; num_decorr]; NUM_UMX],
    };
    let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..NUM_UMX)
        .map(|o| {
            vec![(0..NUM_DMX)
                .map(|ch| vec![if ch == o % NUM_DMX { 1.0 } else { 0.0 }])
                .collect()]
        })
        .collect();
    let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..NUM_UMX)
        .map(|_| vec![vec![vec![0.0]; num_decorr]])
        .collect();
    let qmats = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);
    (ctrl, qmats)
}

/// I-frame `ajoc_dmx_de_data()`: object 1 is the main dialogue
/// (mask bit `num_umx − 1 − obj`), Gmax = 12 dB, its downmix
/// coefficients put it entirely on downmix channel 1.
fn de_iframe() -> AjocDmxDeData {
    AjocDmxDeData {
        dmx_de_cfg: true,
        keep_dmx_de_coeffs: false,
        de_max_gain: 3,
        de_main_dlg_mask: 0b010,
        de_dlg_dmx_coeff: vec![vec![0.0, 1.0]],
    }
}

fn de_pframe() -> AjocDmxDeData {
    AjocDmxDeData {
        dmx_de_cfg: false,
        keep_dmx_de_coeffs: true,
        de_max_gain: 0,
        de_main_dlg_mask: 0,
        de_dlg_dmx_coeff: Vec::new(),
    }
}

fn gop_energies(mode: DecodingMode, gain_db: f32) -> (Vec<Vec<f64>>, usize) {
    let num_decorr = 1usize;
    let params = AjocBodyParams {
        b_lfe: false,
        b_static_dmx: false,
        n_fullband_dmx_signals: NUM_DMX as u32,
        n_fullband_upmix_signals: NUM_UMX as u32,
        obj_type_dmx: vec![ObjType::Dyn; NUM_DMX],
        obj_type_umx: vec![ObjType::Dyn; NUM_UMX],
    };
    let (ctrl, qmats) = selector_setup(num_decorr);
    let s0 = tone_spectrum(24, 40.0);
    let s1 = tone_spectrum(60, 40.0);
    let spectra: Vec<&[f32]> = vec![&s0, &s1];
    let mut enc_state = new_ajoc_diff_state(NUM_UMX, NUM_DMX, 7);
    let dec_params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&dec_params);
    dec.set_decoding_mode(mode);
    dec.set_dialogue_enhancement_gain_db(gain_db);
    let channels_out = if mode == DecodingMode::Core {
        NUM_DMX
    } else {
        NUM_UMX
    };
    let mut out = Vec::new();
    for (seq, iframe) in [(0u32, true), (1, false), (2, false), (3, false)] {
        let de = if iframe { de_iframe() } else { de_pframe() };
        let frame = encode_ajoc_raw_frame_with_dmx_de(
            seq,
            &params,
            &spectra,
            None,
            MAX_SFB,
            num_decorr as u32,
            &ctrl,
            &qmats,
            iframe,
            &mut enc_state,
            Some(&de),
        )
        .unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected an audio frame");
        };
        assert_eq!(af.samples, N as u32);
        let buf = &af.data[0];
        assert_eq!(buf.len(), N * channels_out * 2, "{mode:?} shape");
        let mut e = vec![0.0f64; channels_out];
        for i in 0..N {
            for (c, slot) in e.iter_mut().enumerate() {
                let off = (i * channels_out + c) * 2;
                let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as f64;
                *slot += s * s;
            }
        }
        out.push(e);
    }
    (out, channels_out)
}

fn expect_ratio(ratio: f64, expect: f64, what: &str) {
    assert!(
        (expect * 0.8..=expect * 1.25).contains(&ratio),
        "{what}: ratio {ratio}, expected ≈ {expect}"
    );
}

/// Full decoding: G_DE = 6 dB boosts the dialogue object (object 1)
/// by (10^(6/20))² in energy and leaves objects 0 / 2 alone; the
/// configuration sticks across the P-frames.
#[test]
fn ajoc_full_decoding_de_boosts_dialogue_object_only() {
    let (base, _) = gop_energies(DecodingMode::Full, 0.0);
    let (boosted, _) = gop_energies(DecodingMode::Full, 6.0);
    let g = 10f64.powf(6.0 / 20.0);
    for f in 1..4 {
        expect_ratio(boosted[f][1] / base[f][1], g * g, "dialogue object, frame");
        expect_ratio(boosted[f][0] / base[f][0], 1.0, "object 0 pass-through");
        expect_ratio(boosted[f][2] / base[f][2], 1.0, "object 2 pass-through");
    }
}

/// Full decoding clamps to the bitstream's Gmax (12 dB).
#[test]
fn ajoc_full_decoding_de_clamps_to_gmax() {
    let (base, _) = gop_energies(DecodingMode::Full, 0.0);
    let (boosted, _) = gop_energies(DecodingMode::Full, 40.0);
    let g = 10f64.powf(12.0 / 20.0);
    expect_ratio(boosted[3][1] / base[3][1], g * g, "dialogue object at Gmax");
}

/// Core decoding: the dialogue (object 1 ← downmix channel 1, mapped
/// back onto channel 1 by `de_dlg_dmx_coeff`) adds `(10^(6/20) − 1)·x1`
/// onto channel 1 — a (10^(6/20))² energy boost — while channel 0
/// (coefficient 0) passes through.
#[test]
fn ajoc_core_decoding_de_boosts_dialogue_downmix_channel() {
    let (base, _) = gop_energies(DecodingMode::Core, 0.0);
    let (boosted, _) = gop_energies(DecodingMode::Core, 6.0);
    let g = 10f64.powf(6.0 / 20.0);
    for f in 2..4 {
        expect_ratio(
            boosted[f][1] / base[f][1],
            g * g,
            "core dialogue channel, frame",
        );
        expect_ratio(
            boosted[f][0] / base[f][0],
            1.0,
            "core channel 0 pass-through",
        );
    }
}
