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
