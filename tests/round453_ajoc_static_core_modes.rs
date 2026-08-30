//! Round 453 — the A-JOC `b_static_dmx` core beyond SIMPLE: a 5.X core
//! coded in the part-1 **ASPX_ACPL_3** codec mode (ETSI TS 103 190-1
//! §4.2.6.6 Table 25 nested in TS 103 190-2 §6.2.3.4
//! `audio_data_ajoc`) renders through the shared 5_X carrier pipeline
//! (A-SPX extension + companding + A-CPL centre / surround synthesis)
//! and feeds the §5.7.3.6 object reconstruction — so object-based
//! presentations over such a core decode to PCM in both §4.7 decoding
//! modes, over an I + P + P GOP (Table 25's I-frame-gated configs are
//! sticky across the P-frames).

use oxideav_ac4::ajoc::{AjocCtrlInfo, AjocDataPointInfo, AjocQuantMode};
use oxideav_ac4::ajoc_data::new_ajoc_diff_state;
use oxideav_ac4::ajoc_substream::{
    encode_ajoc_raw_frame_static_acpl3, AjocBodyParams, AjocStaticAcpl3Core,
};
use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::decoder::{Ac4Decoder, DecodingMode};
use oxideav_ac4::encoder_acpl3::{Acpl3ParamPrevRows, AspxEncodedEnvelope};
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

fn identity_setup(num_umx: usize, num_decorr: usize) -> (AjocCtrlInfo, AjocQuantMatrices) {
    let num_dmx = 5usize;
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

fn per_channel_energy(buf: &[u8], num_ch: usize) -> Vec<f64> {
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

/// Encode an I + P + P GOP of static-downmix A-JOC frames over an
/// ASPX_ACPL_3 core (identity dry matrix: object o = core signal o)
/// and decode it in `mode`; returns the per-frame output energies.
fn gop_energies(mode: DecodingMode, b_lfe: bool) -> (Vec<Vec<f64>>, usize) {
    let num_umx = 5usize;
    let num_decorr = 1usize;
    let params = AjocBodyParams {
        b_lfe,
        b_static_dmx: true,
        n_fullband_dmx_signals: 5,
        n_fullband_upmix_signals: num_umx as u32,
        obj_type_dmx: vec![ObjType::Dyn; 5 + usize::from(b_lfe)],
        obj_type_umx: vec![ObjType::Dyn; num_umx + usize::from(b_lfe)],
    };
    let (ctrl, qmats) = identity_setup(num_umx, num_decorr);
    // L / R carriers plus C / Ls / Rs references that the A-CPL γ
    // extraction can fit from the carriers (Ls = ½(L + R), Rs = ½(L −
    // R), C = ½(L + R)) so the synthesised surround / centre carry
    // real energy.
    // Carrier inputs follow the crate's ASPX_ACPL_3 builder
    // convention (PCM-domain tones, as in the round-322 tests): the
    // α / β / γ extractors analyse them as PCM while the split-MDCT
    // writer codes them as the L / R carrier coefficients. Distinct
    // tones decorrelate the pair; the C / Ls / Rs references are
    // carrier mixes so the γ least-squares fit lands real dry-upmix
    // weights (a zero γ matrix makes the whole Pseudocode 118 output
    // collapse to silence).
    let tone = |freq: f32, amp: f32| -> Vec<f32> {
        (0..N)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin())
            .collect()
    };
    let l = tone(220.0, 0.4);
    let r = tone(440.0, 0.3);
    let mix = |a: f32, b: f32| -> Vec<f32> {
        l.iter()
            .zip(r.iter())
            .map(|(x, y)| 0.5 * (a * x + b * y))
            .collect()
    };
    let c = mix(1.0, 1.0);
    let ls = mix(1.0, 1.0);
    let rs = mix(1.0, -1.0);
    let lfe = tone_spectrum(2, 25.0);
    let cfg = live_cfg();
    let zero = AspxEncodedEnvelope {
        values: Vec::new(),
        direction_time: false,
    };
    let core = AjocStaticAcpl3Core {
        transform_length: N as u32,
        max_sfb: MAX_SFB,
        coeffs_l: &l,
        coeffs_r: &r,
        coeffs_c: Some(&c),
        coeffs_ls: Some(&ls),
        coeffs_rs: Some(&rs),
        lfe: b_lfe.then_some((lfe.as_slice(), 4)),
        aspx_cfg: &cfg,
        aspx_l_sig: &zero,
        aspx_l_noise: &zero,
        aspx_r_sig: &zero,
        aspx_r_noise: &zero,
        aspx_tna_mode: &[],
        aspx_l_ah: &[],
        aspx_r_ah: &[],
        acpl_num_param_bands_id: 3,
        acpl_qm0: oxideav_ac4::acpl::AcplQuantMode::Fine,
        acpl_qm1: oxideav_ac4::acpl::AcplQuantMode::Fine,
        alpha_scale: 0.5,
        beta_scale: 0.1,
        gamma_scale: 1.0,
        beta3_scale: 1.0,
    };
    let mut enc_state = new_ajoc_diff_state(num_umx, 5, 7);
    let mut acpl_prev = Acpl3ParamPrevRows::default();
    let dec_params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&dec_params);
    dec.set_decoding_mode(mode);
    let channels_out = 5 + usize::from(b_lfe);
    let mut energies = Vec::new();
    for (seq, iframe) in [(0u32, true), (1, false), (2, false), (3, false)] {
        let frame = encode_ajoc_raw_frame_static_acpl3(
            seq,
            &params,
            &core,
            num_decorr as u32,
            &ctrl,
            &qmats,
            iframe,
            &mut enc_state,
            Some(&mut acpl_prev),
        )
        .unwrap();
        if seq == 0 {
            let info = oxideav_ac4::toc::parse_ac4_toc(&frame).unwrap();
            assert!(info.ajoc_substreams[0].b_static_dmx);
        }
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected an audio frame");
        };
        assert_eq!(af.samples, N as u32);
        let buf = &af.data[0];
        assert_eq!(buf.len(), N * channels_out * 2, "{mode:?} output shape");
        energies.push(per_channel_energy(buf, channels_out));
    }
    (energies, channels_out)
}

/// Full decoding: the five objects (identity dry matrix over the
/// ASPX_ACPL_3 core) all carry settled signal — the L / R carriers
/// directly, C / Ls / Rs through the A-CPL synthesis — and the
/// P-frames keep decoding against the sticky I-frame configuration.
#[test]
fn static_acpl3_core_full_decoding_reconstructs_all_five_objects() {
    let (energies, channels_out) = gop_energies(DecodingMode::Full, false);
    assert_eq!(channels_out, 5);
    let last = energies.last().unwrap();
    for (o, &e) in last.iter().enumerate() {
        assert!(e > 1e4, "object {o} silent on the P-frame tail: {last:?}");
    }
}

/// Core decoding of the same GOP: the output is the rendered 5.X core
/// signal set (LFE first when signalled) without the spatial
/// reconstruction — every slot carries signal over the I + P + P GOP.
#[test]
fn static_acpl3_core_core_decoding_emits_rendered_downmix() {
    let (energies, channels_out) = gop_energies(DecodingMode::Core, true);
    assert_eq!(channels_out, 6);
    // Frame 0 is filterbank / interpolation warm-up (the smooth A-CPL
    // parameter ramp starts from the zero reference); the settled
    // frames must carry signal on every slot.
    for (f, e) in energies.iter().enumerate().skip(1) {
        for (c, &v) in e.iter().enumerate() {
            assert!(v > 1e4, "frame {f} slot {c} silent: {e:?}");
        }
    }
}
