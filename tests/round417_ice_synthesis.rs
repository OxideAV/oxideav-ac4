//! Round 417 — immersive_channel_element synthesis remainders
//! (TS 103 190-2 §5.2.3.2 SAP mixing, §5.3.3.1 ASPX_SCPL full
//! decoding, §5.5.2 ASPX_ACPL_1 / ASPX_ACPL_2 full decoding) through
//! [`oxideav_ac4::decoder::Ac4Decoder`], from complete v2
//! `raw_ac4_frame()` packets.

use oxideav_ac4::asf::ChparamInfo;
use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::ice::{
    encode_ice_raw_frame, write_ice_body_aspx_scpl, write_ice_body_scpl_with_sap, IceScplSpectra,
};
use oxideav_core::bits::BitWriter;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const TL: u32 = 1920;
const N: usize = 1920;
const MAX_SFB: u32 = 20;

fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
    let sfbo = oxideav_ac4::sfb_offset::sfb_offset_48(TL).unwrap();
    let end = sfbo[MAX_SFB as usize] as usize;
    let mut v = vec![0.0f32; end];
    v[bin.min(end - 1)] = amp;
    v
}

fn decode_frame(dec: &mut Ac4Decoder, bytes: Vec<u8>, channels: usize) -> Vec<Vec<i16>> {
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let Frame::Audio(af) = frame else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, N as u32, "frame length");
    let buf = &af.data[0];
    assert_eq!(buf.len(), N * channels * 2, "interleaved buffer size");
    let mut out = vec![Vec::with_capacity(N); channels];
    for i in 0..N {
        for (c, ch) in out.iter_mut().enumerate() {
            let off = (i * channels + c) * 2;
            ch.push(i16::from_le_bytes([buf[off], buf[off + 1]]));
        }
    }
    out
}

fn energy(pcm: &[i16]) -> f64 {
    pcm.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

/// SCPL 7.0.4 with a `b_use_sap_add_ch` M/S-all chparam pair: step 4
/// turns (D, F) into sum/difference tracks BEFORE the S-CPL matrix, so
/// with a silent F the decoded Lb = m(D' − F') cancels (D' = F' = D)
/// while a no-SAP decode of the same spectra splits the D tone evenly
/// across Ls and Lb.
#[test]
fn ice_scpl_sap_step4_msall_steers_surround_pair() {
    let a = tone_spectrum(10, 25.0);
    let b = tone_spectrum(14, 25.0);
    let c = tone_spectrum(18, 25.0);
    let d = tone_spectrum(22, 25.0);
    let e = tone_spectrum(26, 25.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let silent = vec![0.0f32; d.len()];
    let g = tone_spectrum(30, 25.0);
    let hi = [tone_spectrum(40, 20.0), tone_spectrum(44, 20.0)];
    let jk = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
    let scpl_pairs: [[&[f32]; 2]; 2] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&silent, &g], // F silent
        scpl_pairs: &scpl_pairs,
    };
    let msall = ChparamInfo {
        sap_mode: 2,
        ..ChparamInfo::default()
    };
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let run = |with_sap: bool| {
        let mut dec = Ac4Decoder::new(&params);
        let build = |seq: u32| {
            let mut body = BitWriter::new();
            let pair = [msall.clone(), msall.clone()];
            write_ice_body_scpl_with_sap(
                &mut body,
                &spectra,
                None,
                false,
                TL,
                MAX_SFB,
                if with_sap { Some(&pair) } else { None },
                &[],
            )
            .unwrap();
            encode_ice_raw_frame(seq, false, false, true, body).unwrap()
        };
        let _ = decode_frame(&mut dec, build(0), 11);
        decode_frame(&mut dec, build(1), 11)
    };
    let plain = run(false);
    let sap = run(true);
    // Output order: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    let (p_ls, p_lb) = (energy(&plain[3]), energy(&plain[5]));
    let (s_ls, s_lb) = (energy(&sap[3]), energy(&sap[5]));
    // No SAP, F silent: Ls = m·D and Lb = m·D carry equal energy.
    assert!(p_ls > 1e6 && p_lb > 1e6, "plain Ls/Lb live ({p_ls}/{p_lb})");
    let bal = p_ls / p_lb;
    assert!(
        (0.8..=1.25).contains(&bal),
        "plain decode splits D evenly ({bal})"
    );
    // M/S-all SAP: D' = D + F = D, F' = D − F = D → Ls = m·2D, Lb = 0.
    assert!(
        s_lb < s_ls / 1e3,
        "SAP decode cancels Lb ({s_lb} vs {s_ls})"
    );
    let gain = s_ls / p_ls;
    assert!(
        (3.0..=5.0).contains(&gain),
        "SAP decode doubles the Ls amplitude (energy x4, got x{gain})"
    );
    // L / R / C and the top rows are unaffected by step 4.
    let l_ratio = energy(&sap[0]) / energy(&plain[0]).max(1.0);
    assert!(
        (0.95..=1.05).contains(&l_ratio),
        "L unchanged by SAP ({l_ratio})"
    );
}

/// SCPL 7.0.4 with a full-SAP (`sap_mode == 3`) first S-CPL-section
/// chparam element: step 6 adds `a'_0 · D'` into track H, which the
/// Table 23 rows spread onto Tfl/Tbl — the D tone appears on the top
/// row while a plain decode keeps it out.
#[test]
fn ice_scpl_sap_step6_full_sap_predicts_top_from_surround() {
    let a = tone_spectrum(10, 25.0);
    let b = tone_spectrum(14, 25.0);
    let c = tone_spectrum(18, 25.0);
    let d = tone_spectrum(22, 25.0);
    let e = tone_spectrum(26, 25.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let fg = [tone_spectrum(30, 20.0), tone_spectrum(34, 20.0)];
    // Silent S-CPL section: any decoded top-row energy comes from the
    // step-6 prediction alone.
    let silent = vec![0.0f32; a.len()];
    let scpl_pairs: [[&[f32]; 2]; 2] = [[&silent, &silent], [&silent, &silent]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&fg[0], &fg[1]],
        scpl_pairs: &scpl_pairs,
    };
    let m = MAX_SFB as usize;
    let mut dpcm = vec![0i32; m];
    dpcm[0] = 10; // alpha_q = 10 everywhere -> a'_0 = 1,0
    let full_sap = ChparamInfo {
        sap_mode: 3,
        sap_data: Some(oxideav_ac4::asf::SapData {
            sap_coeff_all: true,
            sap_coeff_used: vec![vec![true; m]],
            delta_code_time: false,
            dpcm_alpha_q: vec![dpcm],
        }),
        ..ChparamInfo::default()
    };
    let identity = ChparamInfo::default();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let run = |with_sap: bool| {
        let mut dec = Ac4Decoder::new(&params);
        let build = |seq: u32| {
            let mut body = BitWriter::new();
            let chparams = [
                full_sap.clone(),
                identity.clone(),
                identity.clone(),
                identity.clone(),
            ];
            write_ice_body_scpl_with_sap(
                &mut body,
                &spectra,
                None,
                false,
                TL,
                MAX_SFB,
                None,
                if with_sap { &chparams } else { &[] },
            )
            .unwrap();
            encode_ice_raw_frame(seq, false, false, true, body).unwrap()
        };
        let _ = decode_frame(&mut dec, build(0), 11);
        decode_frame(&mut dec, build(1), 11)
    };
    let plain = run(false);
    let sap = run(true);
    // Output order: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    let p_tfl = energy(&plain[7]);
    let s_tfl = energy(&sap[7]);
    let s_tbl = energy(&sap[9]);
    let s_tfr = energy(&sap[8]);
    assert!(p_tfl < 1e3, "plain decode keeps the top row silent");
    // H'' = 1,0 · D' → Tfl = Tbl = m·H'' both carry the D tone.
    assert!(s_tfl > 1e6, "SAP decode predicts Tfl from D' ({s_tfl})");
    let tb = s_tfl / s_tbl.max(1.0);
    assert!(
        (0.8..=1.25).contains(&tb),
        "Tfl/Tbl split evenly from the H mid ({tb})"
    );
    // The a'_1 element is identity → the right top row stays silent.
    assert!(s_tfr < s_tfl / 1e3, "Tfr stays silent ({s_tfr})");
}

fn test_aspx_config() -> AspxConfig {
    AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        // A high crossover keeps the minimal payload's regenerated
        // band well above the test tones.
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

/// Power of the projection onto the test tone's frequency (MDCT bin
/// `k` ↔ `(k + 0,5) · fs / (2N)` Hz) — isolates the coherent tone
/// from the A-SPX-regenerated high band.
fn tone_power(pcm: &[i16], bin: usize) -> f64 {
    let omega = std::f64::consts::PI * (bin as f64 + 0.5) / N as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &s) in pcm.iter().enumerate() {
        let ph = omega * i as f64;
        re += s as f64 * ph.cos();
        im -= s as f64 * ph.sin();
    }
    re * re + im * im
}

/// ASPX_SCPL 7.0.4 full decoding: the c_gain = m_gain = 1 S-CPL matrix
/// followed by the per-channel A-SPX extension and the Table 10 output
/// gains must reproduce the SCPL decode of the same track spectra in
/// the coded band (the two gain ladders compose to the same totals),
/// while the extension regenerates a high band SCPL does not have.
#[test]
fn ice_aspx_scpl_7_0_4_matches_scpl_low_band_and_extends() {
    // Distinct low-band tones per track group.
    let a = tone_spectrum(6, 200.0);
    let b = tone_spectrum(10, 200.0);
    let c = tone_spectrum(14, 200.0);
    let d = tone_spectrum(18, 200.0);
    let e = tone_spectrum(22, 200.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let fg = [tone_spectrum(26, 150.0), tone_spectrum(30, 150.0)];
    let hi = [tone_spectrum(34, 150.0), tone_spectrum(38, 150.0)];
    let jk = [tone_spectrum(42, 150.0), tone_spectrum(46, 150.0)];
    let scpl_pairs: [[&[f32]; 2]; 2] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&fg[0], &fg[1]],
        scpl_pairs: &scpl_pairs,
    };
    let cfg = test_aspx_config();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    // SCPL reference decode.
    let mut dec_ref = Ac4Decoder::new(&params);
    let build_scpl = |seq: u32| {
        let mut body = BitWriter::new();
        write_ice_body_scpl_with_sap(&mut body, &spectra, None, false, TL, MAX_SFB, None, &[])
            .unwrap();
        encode_ice_raw_frame(seq, false, false, true, body).unwrap()
    };
    let _ = decode_frame(&mut dec_ref, build_scpl(0), 11);
    let scpl = decode_frame(&mut dec_ref, build_scpl(1), 11);
    // ASPX_SCPL decode of the same spectra.
    let mut dec = Ac4Decoder::new(&params);
    let build_aspx = |seq: u32| {
        let mut body = BitWriter::new();
        write_ice_body_aspx_scpl(&mut body, &spectra, None, false, &cfg, true, TL, MAX_SFB)
            .unwrap();
        encode_ice_raw_frame(seq, false, false, true, body).unwrap()
    };
    let _ = decode_frame(&mut dec, build_aspx(0), 11);
    let aspx = decode_frame(&mut dec, build_aspx(1), 11);
    // Per-channel tone bins in the output slot order
    // [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]: L/R/C carry the
    // A/B/C tones; the mixing rows carry both source tones.
    let checks: [(usize, usize); 7] = [
        (0, 6),  // L   <- A
        (1, 10), // R   <- B
        (2, 14), // C   <- C
        (3, 18), // Ls  <- D (+F)
        (4, 22), // Rs  <- E (+G)
        (7, 34), // Tfl <- H (+J)
        (9, 34), // Tbl <- H (−J)
    ];
    for (slot, bin) in checks {
        let p_ref = tone_power(&scpl[slot], bin);
        let p_aspx = tone_power(&aspx[slot], bin);
        assert!(p_ref > 1e12, "SCPL slot {slot} tone live ({p_ref})");
        let ratio = p_aspx / p_ref;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "slot {slot}: ASPX_SCPL low band matches SCPL within 3 dB (ratio {ratio})"
        );
    }
    // The composed gain ladders (S-CPL c/m = 1 + Table 10 output
    // gains vs SCPL c = 2 / m = √2) keep the overall level: total
    // energy across all channels matches within 3 dB (the scaffold
    // payloads keep the regenerated band at the floored noise level).
    let tot_ref: f64 = scpl.iter().map(|ch| energy(ch)).sum();
    let tot_aspx: f64 = aspx.iter().map(|ch| energy(ch)).sum();
    let tot_ratio = tot_aspx / tot_ref;
    assert!(
        (0.5..=2.0).contains(&tot_ratio),
        "ASPX_SCPL total level matches SCPL within 3 dB (ratio {tot_ratio})"
    );
}

/// ASPX_SCPL 9.1.4: 13 named channels + LFE decode with the b_5fronts
/// payload roster (7 elements) and Table 11 gains.
#[test]
fn ice_aspx_scpl_9_1_4_full_decode() {
    let a = tone_spectrum(6, 200.0);
    let b = tone_spectrum(10, 200.0);
    let c = tone_spectrum(14, 200.0);
    let d = tone_spectrum(18, 200.0);
    let e = tone_spectrum(22, 200.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let fg = [tone_spectrum(26, 150.0), tone_spectrum(30, 150.0)];
    let hi = [tone_spectrum(34, 150.0), tone_spectrum(38, 150.0)];
    let jk = [tone_spectrum(42, 150.0), tone_spectrum(46, 150.0)];
    // L'' == A -> Lscr = A − L'' cancels at the A tone; Lw reinforces.
    let lm = [a.clone(), tone_spectrum(50, 150.0)];
    let scpl_pairs: [[&[f32]; 2]; 3] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]], [&lm[0], &lm[1]]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&fg[0], &fg[1]],
        scpl_pairs: &scpl_pairs,
    };
    let lfe = tone_spectrum(2, 100.0);
    let cfg = test_aspx_config();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let build = |seq: u32| {
        let mut body = BitWriter::new();
        write_ice_body_aspx_scpl(
            &mut body,
            &spectra,
            Some((&lfe, 4)),
            true,
            &cfg,
            true,
            TL,
            MAX_SFB,
        )
        .unwrap();
        encode_ice_raw_frame(seq, true, true, true, body).unwrap()
    };
    let _ = decode_frame(&mut dec, build(0), 14);
    let chans = decode_frame(&mut dec, build(1), 14);
    // Output order: [LFE, Lw, Rw, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl,
    // Tfr, Tbl, Tbr].
    let e_lfe = tone_power(&chans[0], 2);
    let p_lw = tone_power(&chans[1], 6);
    let p_lscr = tone_power(&chans[4], 6);
    let p_c = tone_power(&chans[3], 14);
    let p_ls = tone_power(&chans[6], 18);
    assert!(e_lfe > 1e10, "LFE decoded ({e_lfe})");
    assert!(p_lw > 1e12, "Lw carries the A tone ({p_lw})");
    assert!(
        p_lscr < p_lw / 1e3,
        "Lscr = A − L'' cancels ({p_lscr} vs {p_lw})"
    );
    assert!(p_c > 1e12, "C carries its tone ({p_c})");
    assert!(p_ls > 1e12, "Ls carries the D tone ({p_ls})");
}

/// Build one ASPX_ACPL_1 / ASPX_ACPL_2 frame with per-module constant
/// alpha_q rows (beta_q = 0).
#[allow(clippy::too_many_arguments)]
fn build_acpl_frame(
    seq: u32,
    mode: oxideav_ac4::ice::IceCodecMode,
    core: &[&[f32]; 5],
    add_pair: [&[f32]; 2],
    scpl_pairs: &[[&[f32]; 2]],
    alpha_q_per_module: &[i32],
    qmf_band: u8,
    b_5fronts: bool,
    b_iframe: bool,
) -> Vec<u8> {
    let num_bands = oxideav_ac4::acpl::num_param_bands_from_id(2) as usize;
    let rows: Vec<(Vec<i32>, Vec<i32>)> = alpha_q_per_module
        .iter()
        .map(|&a| (vec![a; num_bands], vec![0i32; num_bands]))
        .collect();
    let modules: Vec<(&[i32], &[i32])> = rows
        .iter()
        .map(|(a, b)| (a.as_slice(), b.as_slice()))
        .collect();
    let acpl = oxideav_ac4::ice::IceAcplParams {
        num_param_bands_id: 2,
        quant_mode: oxideav_ac4::acpl::AcplQuantMode::Fine,
        qmf_band,
        modules: &modules,
    };
    let cfg = test_aspx_config();
    let mut body = BitWriter::new();
    oxideav_ac4::ice::write_ice_body_acpl(
        &mut body, mode, core, add_pair, scpl_pairs, &acpl, &cfg, None, b_5fronts, b_iframe, TL,
        MAX_SFB,
    )
    .unwrap();
    encode_ice_raw_frame(seq, false, b_5fronts, b_iframe, body).unwrap()
}

/// ASPX_ACPL_2 7.0.4 full decoding (§5.5.2 Table 27): with alpha =
/// beta = 0 every module splits its carrier evenly onto both outputs,
/// the coded F / G tracks drive the top rows (the Table 27 ACPL_2
/// branch reads x9 / x10), and the L / R / C passthroughs carry the
/// A / B / C tones.
#[test]
fn ice_acpl2_7_0_4_full_decode_routes_carriers() {
    use oxideav_ac4::ice::IceCodecMode;
    let a = tone_spectrum(6, 200.0);
    let b = tone_spectrum(10, 200.0);
    let c = tone_spectrum(14, 200.0);
    let d = tone_spectrum(18, 200.0);
    let e = tone_spectrum(22, 200.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let f = tone_spectrum(26, 150.0);
    let g = tone_spectrum(30, 150.0);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let build = |seq: u32| {
        build_acpl_frame(
            seq,
            IceCodecMode::AspxAcpl2,
            &core,
            [&f, &g],
            &[],
            &[0, 0, 0, 0],
            0,
            false,
            true,
        )
    };
    let _ = decode_frame(&mut dec, build(0), 11);
    let chans = decode_frame(&mut dec, build(1), 11);
    // Output order: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    let p_l = tone_power(&chans[0], 6);
    let p_r = tone_power(&chans[1], 10);
    let p_c = tone_power(&chans[2], 14);
    let p_ls = tone_power(&chans[3], 18);
    let p_lb = tone_power(&chans[5], 18);
    let p_tfl = tone_power(&chans[7], 26);
    let p_tbl = tone_power(&chans[9], 26);
    let p_tfr = tone_power(&chans[8], 30);
    assert!(p_l > 1e12, "L carries the A tone ({p_l})");
    assert!(p_r > 1e12, "R carries the B tone ({p_r})");
    assert!(p_c > 1e12, "C carries the C tone ({p_c})");
    assert!(p_ls > 1e11, "Ls carries the D carrier ({p_ls})");
    let ls_lb = p_ls / p_lb.max(1.0);
    assert!(
        (0.7..=1.4).contains(&ls_lb),
        "alpha = 0 splits D evenly across Ls / Lb ({ls_lb})"
    );
    assert!(p_tfl > 1e11, "Tfl carries the F carrier ({p_tfl})");
    let tf = p_tfl / p_tbl.max(1.0);
    assert!(
        (0.7..=1.4).contains(&tf),
        "alpha = 0 splits F evenly across Tfl / Tbl ({tf})"
    );
    assert!(p_tfr > 1e11, "Tfr carries the G carrier ({p_tfr})");
    // Cross-talk: the D carrier stays out of the top row.
    let leak = tone_power(&chans[7], 18) / p_ls.max(1.0);
    assert!(leak < 0.02, "D carrier leaks into Tfl ({leak})");
}

/// ASPX_ACPL_2 alpha mutation: alpha_q = 8 dequantises to 1,0, so the
/// first module's sub output (Lb) collapses to the (zero-beta)
/// decorrelator term while Ls doubles — the parameters demonstrably
/// steer the decoded PCM.
#[test]
fn ice_acpl2_alpha_steers_surround_pair() {
    use oxideav_ac4::ice::IceCodecMode;
    let a = tone_spectrum(6, 200.0);
    let b = tone_spectrum(10, 200.0);
    let c = tone_spectrum(14, 200.0);
    let d = tone_spectrum(18, 200.0);
    let e = tone_spectrum(22, 200.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let f = tone_spectrum(26, 150.0);
    let g = tone_spectrum(30, 150.0);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let run = |alpha0: i32| {
        let mut dec = Ac4Decoder::new(&params);
        let build = |seq: u32| {
            build_acpl_frame(
                seq,
                IceCodecMode::AspxAcpl2,
                &core,
                [&f, &g],
                &[],
                &[alpha0, 0, 0, 0],
                0,
                false,
                true,
            )
        };
        let _ = decode_frame(&mut dec, build(0), 11);
        decode_frame(&mut dec, build(1), 11)
    };
    let flat = run(0);
    let steered = run(8);
    let flat_ls = tone_power(&flat[3], 18);
    let flat_lb = tone_power(&flat[5], 18);
    let st_ls = tone_power(&steered[3], 18);
    let st_lb = tone_power(&steered[5], 18);
    assert!(
        st_lb < flat_lb / 1e3,
        "alpha = 1 cancels the Lb dry path ({st_lb} vs {flat_lb})"
    );
    let gain = st_ls / flat_ls.max(1.0);
    assert!(
        (3.0..=5.0).contains(&gain),
        "alpha = 1 doubles the Ls amplitude (energy x4, got x{gain})"
    );
    // The other modules are untouched.
    let rs_ratio = tone_power(&steered[4], 22) / tone_power(&flat[4], 22).max(1.0);
    assert!(
        (0.8..=1.25).contains(&rs_ratio),
        "Rs unaffected by the module-0 mutation ({rs_ratio})"
    );
}

/// ASPX_ACPL_1 7.0.4 full decoding across an I + P GOP: with
/// acpl_qmf_band = 8 every test tone sits in the M/S-coded band, so
/// each module reconstructs (main + residual) / (main − residual) —
/// identical D / F tracks cancel in Lb, and a silent J residual splits
/// H evenly across Tfl / Tbl. The P-frame reuses the sticky
/// aspx_config + acpl_config.
#[test]
fn ice_acpl1_7_0_4_ms_band_and_p_frame() {
    use oxideav_ac4::ice::IceCodecMode;
    let a = tone_spectrum(6, 200.0);
    let b = tone_spectrum(10, 200.0);
    let c = tone_spectrum(14, 200.0);
    let d = tone_spectrum(18, 200.0);
    let e = tone_spectrum(22, 200.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let f = d.clone(); // F == D → Lb = (D − F) cancels
    let g = tone_spectrum(30, 150.0);
    let h = tone_spectrum(34, 150.0);
    let silent = vec![0.0f32; a.len()];
    let jk = [tone_spectrum(42, 150.0), tone_spectrum(46, 150.0)];
    let scpl_pairs: [[&[f32]; 2]; 2] = [[&h, &silent], [&silent, &jk[1]]];
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let build = |seq: u32, iframe: bool| {
        build_acpl_frame(
            seq,
            IceCodecMode::AspxAcpl1,
            &core,
            [&f, &g],
            &scpl_pairs,
            &[0, 0, 0, 0],
            8,
            false,
            iframe,
        )
    };
    let _ = decode_frame(&mut dec, build(0, true), 11);
    let ifr = decode_frame(&mut dec, build(1, false), 11);
    let pfr = decode_frame(&mut dec, build(2, false), 11);
    // Output order: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    for (tag, chans) in [("settled", &ifr), ("p-frame", &pfr)] {
        let p_l = tone_power(&chans[0], 6);
        let p_ls = tone_power(&chans[3], 18);
        let p_lb = tone_power(&chans[5], 18);
        let p_tfl = tone_power(&chans[7], 34);
        let p_tbl = tone_power(&chans[9], 34);
        assert!(p_l > 1e12, "{tag}: L carries the A tone ({p_l})");
        assert!(p_ls > 1e11, "{tag}: Ls = D + F reinforces ({p_ls})");
        assert!(
            p_lb < p_ls / 1e3,
            "{tag}: Lb = D − F cancels ({p_lb} vs {p_ls})"
        );
        assert!(p_tfl > 1e11, "{tag}: Tfl = H + J carries H ({p_tfl})");
        let t = p_tfl / p_tbl.max(1.0);
        assert!(
            (0.7..=1.4).contains(&t),
            "{tag}: silent J splits H evenly across Tfl / Tbl ({t})"
        );
    }
}

/// ASPX_ACPL_2 9.1.4: the b_5fronts modules 5 / 6 reconstruct
/// (L, Lscr) / (R, Rscr) from the A / B tracks (decorrelator-only
/// second input); 13 named channels decode.
#[test]
fn ice_acpl2_9_0_4_front_modules() {
    use oxideav_ac4::ice::IceCodecMode;
    let a = tone_spectrum(6, 200.0);
    let b = tone_spectrum(10, 200.0);
    let c = tone_spectrum(14, 200.0);
    let d = tone_spectrum(18, 200.0);
    let e = tone_spectrum(22, 200.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let f = tone_spectrum(26, 150.0);
    let g = tone_spectrum(30, 150.0);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let build = |seq: u32| {
        build_acpl_frame(
            seq,
            IceCodecMode::AspxAcpl2,
            &core,
            [&f, &g],
            &[],
            &[0, 0, 0, 0, 0, 0],
            0,
            true,
            true,
        )
    };
    let _ = decode_frame(&mut dec, build(0), 13);
    let chans = decode_frame(&mut dec, build(1), 13);
    // Output order: [L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr,
    // Tbl, Tbr].
    let p_l = tone_power(&chans[0], 6);
    let p_lscr = tone_power(&chans[3], 6);
    let p_r = tone_power(&chans[1], 10);
    let p_c = tone_power(&chans[2], 14);
    let p_ls = tone_power(&chans[5], 18);
    let p_tfl = tone_power(&chans[9], 26);
    assert!(p_l > 1e11, "L carries the A tone ({p_l})");
    let scr = p_l / p_lscr.max(1.0);
    assert!(
        (0.7..=1.4).contains(&scr),
        "alpha = 0 splits A evenly across L / Lscr ({scr})"
    );
    assert!(p_r > 1e11, "R carries the B tone ({p_r})");
    assert!(p_c > 1e12, "C carries its tone ({p_c})");
    assert!(p_ls > 1e11, "Ls carries the D carrier ({p_ls})");
    assert!(p_tfl > 1e11, "Tfl carries the F carrier ({p_tfl})");
}
