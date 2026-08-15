//! Round 414 — immersive_channel_element decode routing
//! (TS 103 190-2 §6.2.4.1) through [`oxideav_ac4::decoder::Ac4Decoder`].
//!
//! Two synthesis chains are pinned end-to-end from complete v2
//! `raw_ac4_frame()` packets built by
//! [`oxideav_ac4::ice::encode_ice_raw_frame`]:
//!
//! * **SCPL full decoding** (§5.3.3.1 Table 23) — the S-CPL matrix is
//!   verified structurally: identical D/F track spectra must cancel in
//!   the `Lb = m_gain × (D − F)` row while reinforcing in
//!   `Ls = m_gain × (D + F)`, and (for 9.1.4) identical A / L'' tracks
//!   cancel in `Lscr` while reinforcing in `Lw`.
//! * **ASPX_AJCC full decoding** (§5.6.3.5.2) — a tone driven on core
//!   track A with known dequantized A-JCC parameters must reconstruct
//!   on the left-module outputs (L and Tfl) and stay out of the
//!   right-module outputs, across an I + P (DIFF_TIME) GOP.

use oxideav_ac4::acpl::AcplQuantMode;
use oxideav_ac4::ajcc::{encode_ajcc_deltas_freq, AjccData, AjccFramingData};
use oxideav_ac4::ajoc::AjocDiffType;
use oxideav_ac4::aspx::{AspxConfig, AspxFreqResMode, AspxMasterFreqScale, AspxQuantStep};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::ice::{
    encode_ice_raw_frame, write_ice_body_ajcc, write_ice_body_scpl, IceScplSpectra,
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

/// SCPL 7.0.4: Table 23 sum/difference structure over the SMP tracks.
#[test]
fn ice_scpl_7_0_4_full_decode_follows_table_23() {
    // Core A..E tones; D-track content deliberately duplicated on the
    // H track (the first S-CPL pair's first channel) so Lb = m(D − H)
    // cancels while Ls = m(D + H) reinforces (the V1.3.1 Table 23
    // fold pairs (D, H) / (E, I) / (F, J) / (G, K)).
    let a = tone_spectrum(10, 25.0);
    let b = tone_spectrum(14, 25.0);
    let c = tone_spectrum(18, 25.0);
    let d = tone_spectrum(22, 25.0);
    let e = tone_spectrum(26, 25.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    // Additional pair (F, G) = the top-front mids.
    let f = tone_spectrum(30, 25.0);
    let g = tone_spectrum(34, 25.0);
    let hi = [d.clone(), tone_spectrum(44, 20.0)]; // H == D → Lb cancels.
    let jk = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
    let scpl_pairs: [[&[f32]; 2]; 2] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&f, &g],
        scpl_pairs: &scpl_pairs,
    };
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    // Two identical I-frames: the first is the IMDCT overlap ramp-in,
    // the second carries settled PCM.
    let mut last = Vec::new();
    for seq in 0..2u32 {
        let mut body = BitWriter::new();
        write_ice_body_scpl(&mut body, &spectra, None, false, TL, MAX_SFB).unwrap();
        let frame = encode_ice_raw_frame(seq, false, false, true, body).unwrap();
        last = frame;
        if seq == 0 {
            let _ = decode_frame(&mut dec, last.clone(), 11);
        }
    }
    let chans = decode_frame(&mut dec, last, 11);
    // Output order: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    let e_l = energy(&chans[0]);
    let e_ls = energy(&chans[3]);
    let e_lb = energy(&chans[5]);
    let e_rs = energy(&chans[4]);
    let e_rb = energy(&chans[6]);
    let e_tfl = energy(&chans[7]);
    assert!(e_l > 1e6, "L carries the core-A tone (got {e_l})");
    assert!(e_ls > 1e6, "Ls = m(D + H) reinforces (got {e_ls})");
    assert!(
        e_lb < e_ls / 1e3,
        "Lb = m(D − H) cancels for identical D/H ({e_lb} vs {e_ls})"
    );
    // E and I differ → both Rs and Rb carry energy.
    assert!(e_rs > 1e6 && e_rb > 1e6, "Rs/Rb both live ({e_rs}/{e_rb})");
    assert!(e_tfl > 1e6, "Tfl = m(F + J) live ({e_tfl})");
}

/// SCPL 9.1.4: 13 named channels + LFE; the Table 23 b_5fronts front
/// rows (Lw/Lscr from A and L'').
#[test]
fn ice_scpl_9_1_4_full_decode_front_rows_and_lfe() {
    let a = tone_spectrum(10, 25.0);
    let b = tone_spectrum(14, 25.0);
    let c = tone_spectrum(18, 25.0);
    let d = tone_spectrum(22, 25.0);
    let e = tone_spectrum(26, 25.0);
    let core: [&[f32]; 5] = [&a, &b, &c, &d, &e];
    let fg = [tone_spectrum(30, 20.0), tone_spectrum(34, 20.0)];
    let hi = [tone_spectrum(40, 20.0), tone_spectrum(44, 20.0)];
    let jk = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
    // L'' == A → Lscr = A − L'' cancels, Lw = A + L'' reinforces.
    let lm = [a.clone(), tone_spectrum(64, 20.0)];
    let scpl_pairs: [[&[f32]; 2]; 3] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]], [&lm[0], &lm[1]]];
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&fg[0], &fg[1]],
        scpl_pairs: &scpl_pairs,
    };
    let lfe = tone_spectrum(2, 20.0);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let build = |seq: u32| {
        let mut body = BitWriter::new();
        write_ice_body_scpl(&mut body, &spectra, Some((&lfe, 4)), true, TL, MAX_SFB).unwrap();
        encode_ice_raw_frame(seq, true, true, true, body).unwrap()
    };
    let _ = decode_frame(&mut dec, build(0), 14);
    let chans = decode_frame(&mut dec, build(1), 14);
    // Output order: [LFE, Lw, Rw, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl,
    // Tfr, Tbl, Tbr] (LFE first; Lw/Rw on the L/R slots).
    let e_lfe = energy(&chans[0]);
    let e_lw = energy(&chans[1]);
    let e_lscr = energy(&chans[4]);
    let e_rw = energy(&chans[2]);
    let e_rscr = energy(&chans[5]);
    assert!(e_lfe > 1e5, "LFE decoded (got {e_lfe})");
    assert!(e_lw > 1e6, "Lw = A + L'' reinforces (got {e_lw})");
    assert!(
        e_lscr < e_lw / 1e3,
        "Lscr = A − L'' cancels for identical tracks ({e_lscr} vs {e_lw})"
    );
    // B and M'' differ → both Rw and Rscr live.
    assert!(e_rw > 1e6 && e_rscr > 1e6, "Rw/Rscr live ({e_rw}/{e_rscr})");
}

fn minimal_ajcc_data(iframe: bool) -> AjccData {
    let nb = 9usize;
    let fine = AcplQuantMode::Fine;
    let freq_row = |q: i32| vec![(AjocDiffType::Freq, encode_ajcc_deltas_freq(&vec![q; nb]))];
    let time_row = || vec![(AjocDiffType::Time, vec![0i32; nb])];
    let row = |q: i32| if iframe { freq_row(q) } else { time_row() };
    AjccData {
        b_5fronts: false,
        b_no_dt: iframe,
        num_param_bands_id: 2,
        num_bands: nb as u32,
        core_mode: false,
        qm_f: fine,
        qm_b: fine,
        qm_ab: fine,
        qm_dw: fine,
        framing: vec![
            AjccFramingData {
                steep: false,
                num_param_sets: 1,
                param_timeslot: vec![],
            };
            2
        ],
        alpha: vec![row(16), row(16)],
        beta: vec![row(0), row(0)],
        dry: vec![row(8); 4],
        wet: vec![row(20); 6],
    }
}

fn test_aspx_config() -> AspxConfig {
    AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        // A high crossover keeps the minimal payload's regenerated
        // band well above the test lowpass corner.
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
/// from the A-SPX-regenerated high band (the minimal payload's
/// all-zero envelope rows decode to a full-scale HF target on every
/// channel, whose clipping residue is broadband but spectrally thin).
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

/// ASPX_AJCC 7.0.4: bitstream → immersive channel element → A-SPX →
/// A-JCC full decode → 11-channel PCM, over an I + P + P GOP.
#[test]
fn ice_ajcc_7_0_4_full_decode_reconstructs_left_module() {
    // Low-band tone on core track A only; silence elsewhere. With
    // alpha = 0, dry = 0,2 and wet = 0 the left A-JCC module routes
    // x0 to z0 (L) and z9 (Tfl); the right module (fed by B / E) and
    // the centre passthrough carry no low-band content. The energy
    // pins run through a lowpass because the minimal A-SPX payloads
    // regenerate a noise floor above the crossover on every channel.
    let a = tone_spectrum(6, 200.0);
    let silent = vec![0.0f32; a.len()];
    let core: [&[f32]; 5] = [&a, &silent, &silent, &silent, &silent];
    let cfg = test_aspx_config();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut lp: Vec<Vec<f64>> = Vec::new();
    for (seq, iframe) in [(0u32, true), (1, false), (2, false)] {
        let ajcc = minimal_ajcc_data(iframe);
        let mut body = BitWriter::new();
        write_ice_body_ajcc(&mut body, &core, &ajcc, &cfg, None, iframe, TL, MAX_SFB).unwrap();
        let frame = encode_ice_raw_frame(seq, false, false, iframe, body).unwrap();
        let chans = decode_frame(&mut dec, frame, 11);
        lp.push(chans.iter().map(|c| tone_power(c, 6)).collect());
    }
    // Frame 1+ (settled interpolation + IMDCT overlap): output order
    // [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr].
    for (fi, e) in lp.iter().enumerate().skip(1) {
        let e_l = e[0];
        let e_r = e[1];
        let e_c = e[2];
        let e_tfl = e[7];
        let e_tfr = e[8];
        assert!(
            e_l > 2e12,
            "frame {fi}: L reconstructs the core-A tone ({e_l})"
        );
        assert!(
            e_tfl > 2e11,
            "frame {fi}: Tfl carries the wet-free d3 path ({e_tfl})"
        );
        assert!(
            e_r < e_l / 50.0,
            "frame {fi}: right module tone-frequency power stays down (R {e_r} vs L {e_l})"
        );
        assert!(
            e_tfr < e_tfl.max(1.0) / 20.0,
            "frame {fi}: Tfr tone-frequency power stays down ({e_tfr} vs {e_tfl})"
        );
        assert!(
            e_c < e_l / 20.0,
            "frame {fi}: centre passthrough carries no coherent tone ({e_c})"
        );
    }
    // The P-frames (DIFF_TIME all-zero deltas) must not collapse the
    // reconstruction: tone power stays within 3 dB of the
    // settled first P-frame.
    let ratio = lp[2][0] / lp[1][0].max(1.0);
    assert!(
        (0.5..=2.0).contains(&ratio),
        "P-frame L energy within 3 dB of settled predecessor (ratio {ratio})"
    );
}
