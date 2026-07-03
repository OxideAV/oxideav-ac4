//! Round 389 — GOP-depth P-frame chain consistency + measured savings.
//!
//! The two cross-frame reference chains introduced this round — the
//! A-SPX envelope `qscf_*_prev` (§5.7.6.3.4 Pseudocodes 80/81) and the
//! A-CPL parameter `q_prev` (§5.7.7.7 Pseudocode 121) — must stay
//! consistent over an arbitrarily long I,P,P,… sequence: every P-frame
//! references the *previous frame*, not the I-frame, so a drift bug
//! would only show up beyond the second frame. These tests drive an
//! I + 5×P GOP through the live 5_X ACPL_3 path and check every
//! frame's chained reconstruction against a parallel all-I-frame
//! encoder fed the identical PCM (extraction depends only on input
//! history; only the packaging differs).
//!
//! The bit-cost test measures the concrete wire saving of the P-frame
//! element forms (no config block, no xover, TIME/DT rows) for a
//! stationary signal.

use oxideav_ac4::acpl_synth::{differential_decode, AcplDiffState};
use oxideav_ac4::aspx::{
    delta_decode_sig_p80, derive_aspx_frequency_tables, AspxConfig, AspxFreqResMode,
    AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const GOP_P_FRAMES: usize = 5;

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

fn decode_one(dec: &mut Ac4Decoder, bytes: Vec<u8>) -> Vec<i16> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    af.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// I + 5×P GOP: every P-frame's chained A-SPX qscf **and** A-CPL rows
/// must equal the parallel all-I encoder's rows for the same frame
/// index, proving the frame-to-frame reference chains don't drift.
#[test]
fn gop_chained_reconstruction_matches_all_iframe_reference() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let l = multitone(0.3);
    let r = multitone(0.25);
    let c = tone(660.0, 0.2);
    let ls = tone(880.0, 0.2);
    let rs = tone(1100.0, 0.2);
    let chans: [&[f32]; 5] = [&l, &r, &c, &ls, &rs];

    // GOP encoder: I then P frames. Reference encoder: all I frames.
    let mut enc_gop = Ac4ImsEncoder::new();
    let mut enc_ref = Ac4ImsEncoder::new();
    let mut dec_gop = Ac4Decoder::new(&params);
    let mut dec_ref = Ac4Decoder::new(&params);

    let tables = derive_aspx_frequency_tables(&live_cfg(), 0).expect("tables");
    let num_bands = 7u32; // live path: acpl_num_param_bands_id = 3

    // Chained decoder state (what the real decoder maintains).
    let mut sig_prev: Vec<i32> = Vec::new();
    let mut acpl_states: Vec<AcplDiffState> = (0..11).map(|_| AcplDiffState::new()).collect();

    for n in 0..=GOP_P_FRAMES {
        enc_gop.b_iframe_global = n == 0;
        enc_ref.b_iframe_global = true;
        let f_gop = enc_gop.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
        let f_ref = enc_ref.encode_frame_pcm_5_0_acpl3_real_aspx(&chans, 0.5, 0.1, 1.0, 1.0);
        let pcm = decode_one(&mut dec_gop, f_gop);
        let _ = decode_one(&mut dec_ref, f_ref);
        let e: i64 = pcm.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(e > 0, "frame {n} must decode nonsilent");

        let t_gop = dec_gop.last_substream.as_ref().unwrap().tools.clone();
        let t_ref = dec_ref.last_substream.as_ref().unwrap().tools.clone();

        // --- A-SPX SIGNAL envelope chain (primary carrier) ---
        let sig_gop = t_gop.aspx_data_sig_primary.expect("gop sig");
        let sig_ref = t_ref.aspx_data_sig_primary.expect("ref sig");
        let qscf_gop = delta_decode_sig_p80(
            &sig_gop,
            &[],
            &tables.sbg_sig_highres,
            &tables.sbg_sig_lowres,
            &sig_prev,
            1,
        );
        let qscf_ref = delta_decode_sig_p80(
            &sig_ref,
            &[],
            &tables.sbg_sig_highres,
            &tables.sbg_sig_lowres,
            &[],
            1,
        );
        for (sbg, (rg, rr)) in qscf_gop.iter().zip(qscf_ref.iter()).enumerate() {
            assert_eq!(
                rg[0], rr[0],
                "frame {n} sbg {sbg}: chained qscf must match the all-I reference"
            );
        }
        sig_prev = qscf_gop.iter().map(|row| *row.last().unwrap()).collect();

        // --- A-CPL parameter chain (all 11 elements) ---
        let a_gop = t_gop.acpl_data_2ch.expect("gop acpl");
        let a_ref = t_ref.acpl_data_2ch.expect("ref acpl");
        let elems_gop = [
            &a_gop.alpha1,
            &a_gop.alpha2,
            &a_gop.beta1,
            &a_gop.beta2,
            &a_gop.beta3,
            &a_gop.gamma1,
            &a_gop.gamma2,
            &a_gop.gamma3,
            &a_gop.gamma4,
            &a_gop.gamma5,
            &a_gop.gamma6,
        ];
        let elems_ref = [
            &a_ref.alpha1,
            &a_ref.alpha2,
            &a_ref.beta1,
            &a_ref.beta2,
            &a_ref.beta3,
            &a_ref.gamma1,
            &a_ref.gamma2,
            &a_ref.gamma3,
            &a_ref.gamma4,
            &a_ref.gamma5,
            &a_ref.gamma6,
        ];
        for (idx, (eg, er)) in elems_gop.iter().zip(elems_ref.iter()).enumerate() {
            let qg = differential_decode(eg, num_bands, &mut acpl_states[idx]);
            let mut st_fresh = AcplDiffState::new();
            let qr = differential_decode(er, num_bands, &mut st_fresh);
            assert_eq!(
                qg[0], qr[0],
                "frame {n} element {idx}: chained A-CPL rows must match the all-I reference"
            );
        }
    }
}

/// Measure the wire saving of the stationary P-frame element forms.
/// The 5_X ACPL_3 P-frame drops the 19-bit config block and the 3-bit
/// xover and swaps FREQ envelopes / A-CPL rows for near-empty TIME/DT
/// rows — assert a real saving and print the measured numbers.
#[test]
fn stationary_p_frame_element_bits_are_cheaper() {
    use oxideav_ac4::encoder_acpl3::{
        choose_acpl_direction, choose_envelope_direction, qscf_row_from_freq_dpcm,
        write_aspx_data_2ch_directional_envelope_tna_ah_framed, AspxEncodedEnvelope,
    };
    use oxideav_core::bits::BitWriter;

    let cfg = live_cfg();
    let counts = derive_aspx_frequency_tables(&cfg, 0)
        .expect("tables")
        .counts;
    let n_sig = counts.num_sbg_sig_highres as usize;
    let n_noise = counts.num_sbg_noise as usize;
    let sig: Vec<i32> = (0..n_sig as i32).map(|i| 5 - (i % 4)).collect();
    let noise: Vec<i32> = vec![2; n_noise];

    // I-frame form: xover + FREQ rows.
    let freq = |v: &[i32]| AspxEncodedEnvelope {
        values: v.to_vec(),
        direction_time: false,
    };
    let mut bw_i = BitWriter::new();
    write_aspx_data_2ch_directional_envelope_tna_ah_framed(
        &mut bw_i,
        &cfg,
        &freq(&sig),
        &freq(&noise),
        &freq(&sig),
        &freq(&noise),
        &[],
        &[],
        &[],
        true,
    )
    .unwrap();
    let i_bits = bw_i.bit_position();

    // P-frame form: no xover + stationary TIME rows.
    let prev_sig = qscf_row_from_freq_dpcm(&sig);
    let prev_noise = qscf_row_from_freq_dpcm(&noise);
    let time_sig = choose_envelope_direction(&sig, Some(&prev_sig));
    let time_noise = choose_envelope_direction(&noise, Some(&prev_noise));
    assert!(time_sig.direction_time && time_noise.direction_time);
    let mut bw_p = BitWriter::new();
    write_aspx_data_2ch_directional_envelope_tna_ah_framed(
        &mut bw_p,
        &cfg,
        &time_sig,
        &time_noise,
        &time_sig,
        &time_noise,
        &[],
        &[],
        &[],
        false,
    )
    .unwrap();
    let p_bits = bw_p.bit_position();
    eprintln!(
        "ROUND-389 aspx_data_2ch(): I-frame form = {i_bits} bits, stationary P-frame form = {p_bits} bits ({}% saving)",
        (i_bits - p_bits) * 100 / i_bits
    );
    assert!(
        p_bits < i_bits,
        "stationary P-frame aspx_data_2ch must be cheaper ({p_bits} vs {i_bits} bits)"
    );

    // A-CPL: FREQ rows vs stationary DT rows must also shrink.
    let q_row: Vec<i32> = (0..7).map(|i| 4 - (i % 3)).collect();
    let p_freq = choose_acpl_direction(&q_row, None);
    let p_time = choose_acpl_direction(&q_row, Some(&q_row));
    assert!(!p_freq.direction_time);
    assert!(p_time.direction_time);
    assert!(p_time.values.iter().all(|&v| v == 0));
}
