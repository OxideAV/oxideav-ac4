//! Round 443 — the immersive **core decoding mode** (ETSI TS
//! 103 190-2 §4.7.3): [`Ac4Decoder::set_decoding_mode`] +
//! [`DecodingMode::Core`] decode an `immersive_channel_element`
//! substream to the seven-channel core operating point
//! `[L, R, C, Ls, Rs, Tsl, Tsr]` (+ LFE), per §5.3.3.2 (S-CPL
//! Table 24), §4.8.3.11.2 (ASPX_SCPL: Table 8 core roster + §5.4
//! postprocessing + `g = 2`), §4.8.3.14 (ACPL modes: carriers with
//! `g = 2`, no A-CPL) and §5.6.3.5.3 (A-JCC core decoding).
//!
//! Validation is PCM-level against the crate's established
//! full-decode gates on the *same* streams, using the relationships
//! the spec defines between the two modes:
//!
//! * `L / R / C` decode identically in both modes;
//! * core `Ls = (full Ls + full Lb)/√2` (the Table 24 first-seven
//!   fold against the Table 23 mid/side reconstruction), same for
//!   `Rs` and the top pairs;
//! * with `b_5fronts`, core `L = full L + full Lscr`;
//! * rendering the core output to 5.X.2 (§5.10.2.6 Table 45)
//!   reproduces the full-decode output folded to 5.X.2 with the same
//!   customized downmix gains.

use oxideav_ac4::core_render::{render_core_to_5_x_2, CoreRenderGains};
use oxideav_ac4::decoder::{Ac4Decoder, DecodingMode};
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::ice::{encode_ice_raw_frame, write_ice_body_scpl, IceScplSpectra};
use oxideav_core::bits::BitWriter;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const TL: u32 = 1920;
const N: usize = 1920;
const MAX_SFB: u32 = 20;
const SQRT_2: f32 = std::f32::consts::SQRT_2;

fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
    let sfbo = oxideav_ac4::sfb_offset::sfb_offset_48(TL).unwrap();
    let end = sfbo[MAX_SFB as usize] as usize;
    let mut v = vec![0.0f32; end];
    v[bin.min(end - 1)] = amp;
    v
}

fn decode_frame(dec: &mut Ac4Decoder, bytes: Vec<u8>, channels: usize) -> Vec<Vec<f32>> {
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
            let s = i16::from_le_bytes([buf[off], buf[off + 1]]);
            ch.push(s as f32 / 32768.0);
        }
    }
    out
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// Relative RMS error between a reference and a decoded channel.
fn rel_rms_err(reference: &[f32], dec: &[f32]) -> f64 {
    let (mut err, mut sig) = (0.0f64, 0.0f64);
    for (&r, &d) in reference.iter().zip(dec) {
        err += (d as f64 - r as f64) * (d as f64 - r as f64);
        sig += r as f64 * r as f64;
    }
    (err / sig.max(1e-30)).sqrt()
}

fn fold(a: &[f32], b: &[f32], g: f32) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| (x + y) * g).collect()
}

/// Build the round-414-style SCPL spectra set: distinct tones on
/// every SMP track (core A..E, additional pair F/G = top-front mids,
/// S-CPL pairs H/I + J/K = the pair sides).
struct ScplTones {
    core: Vec<Vec<f32>>,
    fg: [Vec<f32>; 2],
    hi: [Vec<f32>; 2],
    jk: [Vec<f32>; 2],
}

fn scpl_tones() -> ScplTones {
    ScplTones {
        core: vec![
            tone_spectrum(10, 25.0),
            tone_spectrum(14, 25.0),
            tone_spectrum(18, 25.0),
            tone_spectrum(22, 25.0),
            tone_spectrum(26, 25.0),
        ],
        fg: [tone_spectrum(30, 22.0), tone_spectrum(34, 22.0)],
        hi: [tone_spectrum(40, 20.0), tone_spectrum(44, 20.0)],
        jk: [tone_spectrum(50, 18.0), tone_spectrum(54, 18.0)],
    }
}

fn build_scpl_frame(seq: u32, t: &ScplTones, lfe: Option<&[f32]>, b_5fronts: bool) -> Vec<u8> {
    let core: [&[f32]; 5] = std::array::from_fn(|i| t.core[i].as_slice());
    let lm = [tone_spectrum(60, 16.0), tone_spectrum(64, 16.0)];
    let mut scpl_pairs: Vec<[&[f32]; 2]> = vec![
        [t.hi[0].as_slice(), t.hi[1].as_slice()],
        [t.jk[0].as_slice(), t.jk[1].as_slice()],
    ];
    if b_5fronts {
        scpl_pairs.push([lm[0].as_slice(), lm[1].as_slice()]);
    }
    let spectra = IceScplSpectra {
        core: &core,
        add_pair: [&t.fg[0], &t.fg[1]],
        scpl_pairs: &scpl_pairs,
    };
    let mut body = BitWriter::new();
    write_ice_body_scpl(
        &mut body,
        &spectra,
        lfe.map(|l| (l, 4u32)),
        b_5fronts,
        TL,
        MAX_SFB,
    )
    .unwrap();
    encode_ice_raw_frame(seq, lfe.is_some(), b_5fronts, true, body).unwrap()
}

/// SCPL 7.0.4: the §5.3.3.2 Table 24 core decode against the
/// §5.3.3.1 Table 23 full decode of the *same* frames — L/R/C are
/// identical, each core surround / top channel is the mid of the
/// corresponding full-decode pair (÷√2 of the pair sum).
#[test]
fn core_scpl_7_0_4_matches_full_decode_fold() {
    let t = scpl_tones();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    assert_eq!(dec_core.decoding_mode(), DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for seq in 0..2u32 {
        let frame = build_scpl_frame(seq, &t, None, false);
        full = decode_frame(&mut dec_full, frame.clone(), 11);
        core = decode_frame(&mut dec_core, frame, 7);
    }
    // Full order: [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr];
    // core order: [L, R, C, Ls, Rs, Tsl, Tsr].
    for slot in 0..3 {
        let e = rel_rms_err(&full[slot], &core[slot]);
        assert!(e < 0.02, "L/R/C identical in both modes (slot {slot}: {e})");
    }
    let checks: [(usize, usize, usize); 4] = [
        (3, 3, 5),  // core Ls = (Ls + Lb)/√2
        (4, 4, 6),  // core Rs = (Rs + Rb)/√2
        (5, 7, 9),  // core Tsl = (Tfl + Tbl)/√2
        (6, 8, 10), // core Tsr = (Tfr + Tbr)/√2
    ];
    for (c_slot, f_a, f_b) in checks {
        let reference = fold(&full[f_a], &full[f_b], 1.0 / SQRT_2);
        assert!(energy(&core[c_slot]) > 1e-4, "core slot {c_slot} live");
        let e = rel_rms_err(&reference, &core[c_slot]);
        assert!(
            e < 0.05,
            "core slot {c_slot} folds the full-decode pair (err {e})"
        );
    }
}

/// SCPL 9.1.4 (`b_5fronts`): core L / R fold the screen channels
/// (`core L = 2·A'' = full L + full Lscr`), and the LFE decodes
/// identically (unity in both modes) on the leading slot.
#[test]
fn core_scpl_9_1_4_folds_screens_and_lfe() {
    let t = scpl_tones();
    let lfe = tone_spectrum(2, 20.0);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for seq in 0..2u32 {
        let frame = build_scpl_frame(seq, &t, Some(&lfe), true);
        full = decode_frame(&mut dec_full, frame.clone(), 14);
        core = decode_frame(&mut dec_core, frame, 8);
    }
    // Full order: [LFE, L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl,
    // Tfr, Tbl, Tbr]; core order: [LFE, L, R, C, Ls, Rs, Tsl, Tsr].
    let e_lfe = rel_rms_err(&full[0], &core[0]);
    assert!(energy(&core[0]) > 1e-4, "core LFE live");
    assert!(e_lfe < 0.02, "LFE identical in both modes ({e_lfe})");
    // core L = full L + full Lscr (the Table 23 b_5fronts front rows
    // are ×2·½ so the sum is exact), same for R.
    for (c_slot, f_a, f_b) in [(1usize, 1usize, 4usize), (2, 2, 5)] {
        let reference = fold(&full[f_a], &full[f_b], 1.0);
        let e = rel_rms_err(&reference, &core[c_slot]);
        assert!(
            e < 0.05,
            "core front slot {c_slot} folds the screens (err {e})"
        );
    }
    // core C identical; surround / top folds as in the 7.0.4 case.
    assert!(rel_rms_err(&full[3], &core[3]) < 0.02, "C identical");
    let reference = fold(&full[6], &full[8], 1.0 / SQRT_2);
    let e = rel_rms_err(&reference, &core[4]);
    assert!(e < 0.05, "core Ls folds (Ls, Lb) with b_5fronts (err {e})");
}

/// ASPX_SCPL 7.0.4 through the real encoder arm: in the coded band
/// L / R / C decode identically in both modes (the pair payload is
/// jointly decoded and C carries its own 1ch payload — no §5.4
/// postprocessing on them, and `g = 2` matches the Table 10 gains),
/// while the core surround / top channels track the full-decode pair
/// folds.
#[test]
fn core_aspx_scpl_7_0_4_relations() {
    let cycles = [12u32, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52];
    let input: Vec<Vec<f32>> = cycles
        .iter()
        .enumerate()
        .map(|(ch, &c)| {
            (0..N)
                .map(|i| {
                    let t = i as f32 / N as f32;
                    0.35 * (2.0 * std::f32::consts::PI * c as f32 * t + 0.3 * ch as f32).sin()
                })
                .collect()
        })
        .collect();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_aspx_scpl(&refs);
        full = decode_frame(&mut dec_full, bytes.clone(), 11);
        core = decode_frame(&mut dec_core, bytes, 7);
    }
    for slot in 0..3 {
        let e = rel_rms_err(&full[slot], &core[slot]);
        assert!(
            e < 0.05,
            "ASPX_SCPL L/R/C identical in both modes (slot {slot}: {e})"
        );
    }
    for (c_slot, f_a, f_b) in [(3usize, 3usize, 5usize), (4, 4, 6), (5, 7, 9), (6, 8, 10)] {
        let reference = fold(&full[f_a], &full[f_b], 1.0 / SQRT_2);
        assert!(energy(&core[c_slot]) > 1e-4, "core slot {c_slot} live");
        let e = rel_rms_err(&reference, &core[c_slot]);
        assert!(
            e < 0.2,
            "core slot {c_slot} tracks the full-decode fold (err {e})"
        );
    }
}

/// ASPX_ACPL_2 7.0.4 through the real encoder arm (§4.8.3.14): core
/// mode skips the A-CPL modules and emits `2×` the A-SPX-extended
/// carriers — `L / R / C` match the full decode exactly (full mode's
/// z0 = 2·x0 rows are the same operation), and each core surround /
/// top channel is the ÷√2 fold of its full-decode module pair (the
/// module's dry outputs reconstruct `mid ± α·mid`-style splits whose
/// sum returns the carrier; correlated content keeps the wet part
/// small).
#[test]
fn core_acpl2_7_0_4_carrier_relations() {
    // Correlated pairs: each surround / top partner tracks its lead.
    let lead = |c: u32, p: f32| -> Vec<f32> {
        (0..N)
            .map(|i| {
                let t = i as f32 / N as f32;
                0.3 * (2.0 * std::f32::consts::PI * c as f32 * t + p).sin()
            })
            .collect()
    };
    let scale = |x: &[f32], g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
    let ls = lead(24, 0.4);
    let rs = lead(28, 0.9);
    let tfl = lead(36, 1.3);
    let tfr = lead(40, 1.8);
    let input: Vec<Vec<f32>> = vec![
        lead(9, 0.0),      // L
        lead(22, 0.5),     // R
        lead(20, 1.0),     // C
        ls.clone(),        // Ls
        rs.clone(),        // Rs
        scale(&ls, 0.8),   // Lb
        scale(&rs, 0.8),   // Rb
        tfl.clone(),       // Tfl
        tfr.clone(),       // Tfr
        scale(&tfl, 0.75), // Tbl
        scale(&tfr, 0.75), // Tbr
    ];
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_acpl2(&refs);
        full = decode_frame(&mut dec_full, bytes.clone(), 11);
        core = decode_frame(&mut dec_core, bytes, 7);
    }
    for slot in 0..3 {
        let e = rel_rms_err(&full[slot], &core[slot]);
        assert!(
            e < 0.05,
            "ACPL_2 L/R/C identical in both modes (slot {slot}: {e})"
        );
    }
    for (c_slot, f_a, f_b) in [(3usize, 3usize, 5usize), (4, 4, 6), (5, 7, 9), (6, 8, 10)] {
        let reference = fold(&full[f_a], &full[f_b], 1.0 / SQRT_2);
        assert!(energy(&core[c_slot]) > 1e-4, "core slot {c_slot} live");
        let e = rel_rms_err(&reference, &core[c_slot]);
        assert!(
            e < 0.35,
            "core slot {c_slot} tracks the full-decode module fold (err {e})"
        );
    }
}

/// ASPX_ACPL_1 7.0.4: identical relations — the core mode drops the
/// S-CPL-section residual tracks and the M/S band entirely, keeping
/// only the `2×` carriers.
#[test]
fn core_acpl1_7_0_4_carrier_relations() {
    let lead = |c: u32, a: f32, p: f32| -> Vec<f32> {
        (0..N)
            .map(|i| {
                let t = i as f32 / N as f32;
                a * (2.0 * std::f32::consts::PI * c as f32 * t + p).sin()
            })
            .collect()
    };
    let scale = |x: &[f32], g: f32| -> Vec<f32> { x.iter().map(|&v| v * g).collect() };
    let ls = lead(24, 0.3, 0.4);
    let rs = lead(28, 0.3, 0.9);
    let tfl = lead(36, 0.3, 1.3);
    let tfr = lead(40, 0.3, 1.8);
    let input: Vec<Vec<f32>> = vec![
        lead(9, 0.3, 0.0),
        lead(22, 0.3, 0.5),
        lead(20, 0.3, 1.0),
        ls.clone(),
        rs.clone(),
        scale(&ls, 0.8),
        scale(&rs, 0.8),
        tfl.clone(),
        tfr.clone(),
        scale(&tfl, 0.75),
        scale(&tfr, 0.75),
    ];
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_acpl1(&refs);
        full = decode_frame(&mut dec_full, bytes.clone(), 11);
        core = decode_frame(&mut dec_core, bytes, 7);
    }
    for slot in 0..3 {
        let e = rel_rms_err(&full[slot], &core[slot]);
        assert!(
            e < 0.05,
            "ACPL_1 L/R/C identical in both modes (slot {slot}: {e})"
        );
    }
    for (c_slot, f_a, f_b) in [(3usize, 3usize, 5usize), (5, 7, 9)] {
        let reference = fold(&full[f_a], &full[f_b], 1.0 / SQRT_2);
        assert!(energy(&core[c_slot]) > 1e-4, "core slot {c_slot} live");
        let e = rel_rms_err(&reference, &core[c_slot]);
        assert!(
            e < 0.35,
            "core slot {c_slot} tracks the full-decode module fold (err {e})"
        );
    }
}

/// ASPX_AJCC 7.0.4 through the real encoder arm: §5.6.3.5.3 core
/// decoding reconstructs per-module content on the correct side —
/// the centre passes through (`z2 = x2in`), left-module content
/// stays off the right outputs, and the top outputs are live.
#[test]
fn core_ajcc_7_0_4_reconstructs_and_separates() {
    let sb_tone = |sb: u32, amp: f32, phase: f32| -> Vec<f32> {
        let cycles = 15 * sb + 7;
        (0..N)
            .map(|i| {
                let t = i as f32 / N as f32;
                amp * (2.0 * std::f32::consts::PI * cycles as f32 * t + phase).sin()
            })
            .collect()
    };
    let tone_power = |pcm: &[f32], sb: u32| -> f64 {
        let cycles = (15 * sb + 7) as f64;
        let omega = 2.0 * std::f64::consts::PI * cycles / N as f64;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &s) in pcm.iter().enumerate() {
            let ph = omega * i as f64;
            re += s as f64 * ph.cos();
            im -= s as f64 * ph.sin();
        }
        re * re + im * im
    };
    // Round-440 core-layout content: left module (L sb1, Tfl sb4,
    // Ls sb2, Lb sb5, Tbl sb7), right module (R sb3, Tfr sb6, Rs sb0,
    // Rb sb8, Tbr sb9), C sb5.
    let input: Vec<Vec<f32>> = vec![
        sb_tone(1, 0.35, 0.0),
        sb_tone(3, 0.35, 0.5),
        sb_tone(5, 0.35, 1.0),
        sb_tone(2, 0.35, 0.4),
        sb_tone(0, 0.35, 0.9),
        sb_tone(5, 0.30, 0.2),
        sb_tone(8, 0.30, 0.7),
        sb_tone(4, 0.30, 1.3),
        sb_tone(6, 0.30, 1.8),
        sb_tone(7, 0.28, 0.6),
        sb_tone(9, 0.28, 1.1),
    ];
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    dec.set_decoding_mode(DecodingMode::Core);
    let mut core = Vec::new();
    for _ in 0..4 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_ajcc(&refs);
        core = decode_frame(&mut dec, bytes, 7);
    }
    // Centre: z2 = x2in restores the coded core C = input C.
    let settle = N / 2;
    let e_c_in = energy(&input[2][settle..]);
    let e_c_out = energy(&core[2][settle..]);
    let ratio = e_c_out / e_c_in;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "core C within 3 dB of the input C ({ratio})"
    );
    // Left-module content stays off the right outputs: the R tone
    // (sb3) must not land on core L.
    let l_own = tone_power(&core[0][settle..], 1);
    let l_leak = tone_power(&core[0][settle..], 3);
    assert!(l_own > 1e-2, "core L carries the front-left tone");
    assert!(
        l_leak < l_own * 0.05,
        "right-module content stays off core L ({l_leak} vs {l_own})"
    );
    // Surround + top outputs are live.
    for slot in [3usize, 4, 5, 6] {
        assert!(
            energy(&core[slot][settle..]) > 1e-4,
            "core slot {slot} live"
        );
    }
}

/// The §5.10.2.6 rendering fold: rendering the core output to 5.X.2
/// (Table 45, default Table 130 gains) reproduces the full decode of
/// the same SCPL stream folded to 5.X.2 with the same gains
/// (`Ls_out = gain_b·(Ls + Lb)`, `Tsl_out = gain_t1·(Tfl + Tbl)` —
/// the +3 dB static terms exactly cancel the core fold's ÷√2).
#[test]
fn core_render_to_5_x_2_matches_full_decode_fold() {
    let t = scpl_tones();
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for seq in 0..2u32 {
        let frame = build_scpl_frame(seq, &t, None, false);
        full = decode_frame(&mut dec_full, frame.clone(), 11);
        core = decode_frame(&mut dec_core, frame, 7);
    }
    let gains = CoreRenderGains::default();
    let rendered = render_core_to_5_x_2(&core, 3, true, &gains);
    // Full-decode 5.X.2 fold with the same customized gains.
    let want: Vec<Vec<f32>> = vec![
        full[0].clone(),
        full[1].clone(),
        full[2].clone(),
        fold(&full[3], &full[5], gains.gain_b),
        fold(&full[4], &full[6], gains.gain_b),
        fold(&full[7], &full[9], gains.gain_t1),
        fold(&full[8], &full[10], gains.gain_t1),
    ];
    for (slot, (w, r)) in want.iter().zip(rendered.iter()).enumerate() {
        assert!(energy(w) > 1e-4, "rendered slot {slot} live");
        let e = rel_rms_err(w, r);
        assert!(
            e < 0.05,
            "core render to 5.X.2 matches the full-decode fold (slot {slot}: {e})"
        );
    }
}

/// The 7.1.4 LFE decodes identically in core mode (leading slot,
/// unity gain) and the core frame carries exactly 8 channels.
#[test]
fn core_mode_7_1_4_lfe_and_channel_count() {
    let t = scpl_tones();
    let lfe = tone_spectrum(2, 20.0);
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec_full = Ac4Decoder::new(&params);
    let mut dec_core = Ac4Decoder::new(&params);
    dec_core.set_decoding_mode(DecodingMode::Core);
    let (mut full, mut core) = (Vec::new(), Vec::new());
    for seq in 0..2u32 {
        let frame = build_scpl_frame(seq, &t, Some(&lfe), false);
        full = decode_frame(&mut dec_full, frame.clone(), 12);
        core = decode_frame(&mut dec_core, frame, 8);
    }
    assert!(energy(&core[0]) > 1e-4, "core LFE live");
    let e = rel_rms_err(&full[0], &core[0]);
    assert!(e < 0.02, "LFE identical in both modes ({e})");
}
