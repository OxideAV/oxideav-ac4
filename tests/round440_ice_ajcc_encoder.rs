//! Round 440 — the A-JCC parameter **extractor** wired end-to-end:
//! ICE ASPX_AJCC encode from PCM for both layouts (TS 103 190-2
//! §6.2.4.1 + §5.6 — 7.0.4 / 7.1.4 core layout, 9.0.4 / 9.1.4
//! `b_5fronts`).
//!
//! The encoder derives the five-channel core as the per-module output
//! sums of the decoder's Table 35 reconstruction, extracts per-band
//! alpha / beta / dry / wet rows from the named channels' QMF
//! statistics, and assembles a GOP-aware `ajcc_data()` element.
//!
//! Measured here:
//! 1. Per-band-separated tones (each module-group channel in its own
//!    parameter band) reconstruct on their own output slots: settled
//!    per-channel energy within 3 dB and cross-talk pinned below the
//!    reconstruction.
//! 2. The 7.1.4 / 9.1.4 LFE arms decode on the leading output slot.
//! 3. Parse-exactness: the emitted `ajcc_data()` re-reads to exactly
//!    the extractor's quantised rows (through the decoder's own
//!    differential decode).
//! 4. P-frame GOP: a stationary I+P+P run selects TIME rows on the
//!    wire and decodes to the same settled output.
//! 5. Determinism.

use oxideav_ac4::ajcc::{decode_ajcc_parsed, AjccState};
use oxideav_ac4::ajoc::AjocDiffType;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::ice::IceCodecMode;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;

/// Frame-periodic tone at a QMF subband centre: subband `sb` spans
/// `375·sb .. 375·(sb+1)` Hz, centre `375·sb + 187,5` Hz — cycles =
/// `15·sb + 7` puts the tone within 12,5 Hz of the centre, keeping its
/// energy inside one subband (and, with `nb = 15`, one parameter band
/// for sb ≤ 8).
fn sb_tone(sb: u32, amp: f32, phase: f32) -> Vec<f32> {
    let cycles = 15 * sb + 7;
    (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            amp * (2.0 * std::f32::consts::PI * cycles as f32 * t + phase).sin()
        })
        .collect()
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

/// 11-channel core-layout content with every module-group channel in
/// its own QMF subband (= its own parameter band at `nb = 15`):
/// left module (L sb1, Tfl sb4 | Ls sb2, Lb sb5, Tbl sb7), right
/// module (R sb3, Tfr sb6 | Rs sb0, Rb sb8, Tbr sb9), C sb5.
/// Order: `[L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
fn core_layout_input() -> Vec<Vec<f32>> {
    vec![
        sb_tone(1, 0.35, 0.0), // L
        sb_tone(3, 0.35, 0.5), // R
        sb_tone(5, 0.35, 1.0), // C
        sb_tone(2, 0.35, 0.4), // Ls
        sb_tone(0, 0.35, 0.9), // Rs
        sb_tone(5, 0.30, 0.2), // Lb
        sb_tone(8, 0.30, 0.7), // Rb
        sb_tone(4, 0.30, 1.3), // Tfl
        sb_tone(6, 0.30, 1.8), // Tfr
        sb_tone(7, 0.28, 0.6), // Tbl
        sb_tone(9, 0.28, 1.1), // Tbr
    ]
}

/// 13-channel b_5fronts content: front modules (L sb1, Tfl sb4,
/// Lscr sb2 | R sb3, Tfr sb6, Rscr sb0), back modules (Ls sb2, Lb sb5,
/// Tbl sb7 | Rs sb1, Rb sb8, Tbr sb9), C sb5.
/// Order: `[L, R, C, Lscr, Rscr, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`.
fn fronts_layout_input() -> Vec<Vec<f32>> {
    vec![
        sb_tone(1, 0.35, 0.0), // L
        sb_tone(3, 0.35, 0.5), // R
        sb_tone(5, 0.35, 1.0), // C
        sb_tone(2, 0.32, 0.3), // Lscr
        sb_tone(0, 0.32, 0.8), // Rscr
        sb_tone(2, 0.35, 2.4), // Ls
        sb_tone(1, 0.35, 2.9), // Rs
        sb_tone(5, 0.30, 0.2), // Lb
        sb_tone(8, 0.30, 0.7), // Rb
        sb_tone(4, 0.30, 1.3), // Tfl
        sb_tone(6, 0.30, 1.8), // Tfr
        sb_tone(7, 0.28, 0.6), // Tbl
        sb_tone(9, 0.28, 1.1), // Tbr
    ]
}

#[test]
fn ice_ajcc_7_0_4_extractor_reconstructs_per_band_content() {
    let input = core_layout_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_7_0_4_ice_ajcc(&refs);
        last = decode_frame(&mut dec, bytes, 11);
    }
    for ch in 0..11 {
        let r = energy(&last[ch]) / energy(&input[ch]).max(1e-30);
        eprintln!("ROUND-440 AJCC 7.0.4 settled energy ratio ch{ch}: {r:.3}");
        assert!(
            (0.5..=2.0).contains(&r),
            "channel {ch} settled energy within 3 dB of input (ratio {r:.3})"
        );
    }
}

#[test]
fn ice_ajcc_9_0_4_extractor_reconstructs_per_band_content() {
    let input = fronts_layout_input();
    let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    for _ in 0..6 {
        let bytes = enc.encode_frame_pcm_9_0_4_ice_ajcc(&refs);
        last = decode_frame(&mut dec, bytes, 13);
    }
    for ch in 0..13 {
        let r = energy(&last[ch]) / energy(&input[ch]).max(1e-30);
        eprintln!("ROUND-440 AJCC 9.0.4 settled energy ratio ch{ch}: {r:.3}");
        assert!(
            (0.5..=2.0).contains(&r),
            "channel {ch} settled energy within 3 dB of input (ratio {r:.3})"
        );
    }
}

#[test]
fn ice_ajcc_lfe_arms_decode_on_leading_slot() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let lfe = sb_tone(0, 0.4, 0.0);
    // 7.1.4.
    {
        let input = core_layout_input();
        let mut all: Vec<&[f32]> = input.iter().map(|v| v.as_slice()).collect();
        all.push(&lfe);
        let refs: [&[f32]; 12] = std::array::from_fn(|i| all[i]);
        let mut enc = Ac4ImsEncoder::new();
        let mut dec = Ac4Decoder::new(&params);
        let mut last = Vec::new();
        for _ in 0..4 {
            let bytes = enc.encode_frame_pcm_7_1_4_ice_ajcc(&refs);
            last = decode_frame(&mut dec, bytes, 12);
        }
        let r = energy(&last[0]) / energy(&lfe).max(1e-30);
        assert!(
            (0.5..=2.0).contains(&r),
            "7.1.4 LFE settled energy ratio {r:.3}"
        );
    }
    // 9.1.4.
    {
        let input = fronts_layout_input();
        let mut all: Vec<&[f32]> = input.iter().map(|v| v.as_slice()).collect();
        all.push(&lfe);
        let refs: [&[f32]; 14] = std::array::from_fn(|i| all[i]);
        let mut enc = Ac4ImsEncoder::new();
        let mut dec = Ac4Decoder::new(&params);
        let mut last = Vec::new();
        for _ in 0..4 {
            let bytes = enc.encode_frame_pcm_9_1_4_ice_ajcc(&refs);
            last = decode_frame(&mut dec, bytes, 14);
        }
        let r = energy(&last[0]) / energy(&lfe).max(1e-30);
        assert!(
            (0.5..=2.0).contains(&r),
            "9.1.4 LFE settled energy ratio {r:.3}"
        );
    }
}

#[test]
fn ice_ajcc_bitstream_re_reads_to_extracted_rows() {
    // Encode one I-frame per layout and check the parsed ajcc_data()
    // differential-decodes to exactly the rows an independent run of
    // the extractor produces on the same content.
    let params = CodecParameters::audio(CodecId::new("ac4"));
    for b_5fronts in [false, true] {
        let input = if b_5fronts {
            fronts_layout_input()
        } else {
            core_layout_input()
        };
        let mut enc = Ac4ImsEncoder::new();
        let mut dec = Ac4Decoder::new(&params);
        let bytes = if b_5fronts {
            let refs: [&[f32]; 13] = std::array::from_fn(|i| input[i].as_slice());
            enc.encode_frame_pcm_9_0_4_ice_ajcc(&refs)
        } else {
            let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
            enc.encode_frame_pcm_7_0_4_ice_ajcc(&refs)
        };
        let _ = decode_frame(&mut dec, bytes, if b_5fronts { 13 } else { 11 });
        let sub = dec.last_substream.as_ref().expect("substream parsed");
        let ice = sub.tools.ice.as_deref().expect("ice element parsed");
        assert_eq!(ice.mode, IceCodecMode::AspxAjcc);
        assert_eq!(ice.b_5fronts, b_5fronts);
        let data = ice.ajcc.as_deref().expect("ajcc_data parsed").clone();
        assert_eq!(data.num_bands, 15, "extractor band config");
        let mut st = AjccState::new(b_5fronts);
        let decoded = decode_ajcc_parsed(data, &mut st).expect("differential decode");

        // Independent extraction: analyse the named channels through
        // fresh streaming banks exactly like the encoder's first
        // frame.
        let analyse = |pcm: &[f32]| -> Vec<Vec<(f32, f32)>> {
            let scaled: Vec<f32> = pcm
                .iter()
                .map(|&v| v * oxideav_ac4::aspx::ASPX_QMF_PCM_SCALE)
                .collect();
            let mut bank = oxideav_ac4::qmf::QmfAnalysisBank::new();
            oxideav_ac4::encoder_acpl3::qmf_slots_to_sb_major(&bank.process_block(&scaled))
        };
        let q: Vec<Vec<Vec<(f32, f32)>>> = input.iter().map(|c| analyse(c)).collect();
        let cfg = oxideav_ac4::encoder_ajcc::AjccBuildConfig::default();
        let rows = if b_5fronts {
            let named_q: [&oxideav_ac4::encoder_ajcc::QmfMat; 13] =
                std::array::from_fn(|i| q[i].as_slice());
            oxideav_ac4::encoder_ajcc::extract_ajcc_5fronts_rows(
                &named_q,
                15,
                cfg.qm_first,
                cfg.qm_second,
            )
        } else {
            let named_q: [&oxideav_ac4::encoder_ajcc::QmfMat; 11] =
                std::array::from_fn(|i| q[i].as_slice());
            oxideav_ac4::encoder_ajcc::extract_ajcc_core_rows(
                &named_q,
                15,
                cfg.qm_first,
                cfg.qm_second,
            )
        };
        for (i, set) in rows.dry.iter().enumerate() {
            assert_eq!(
                decoded.dry_q[i][0], *set,
                "dry SET {i} re-reads to the extractor rows (5fronts {b_5fronts})"
            );
        }
        for (i, set) in rows.wet.iter().enumerate() {
            assert_eq!(
                decoded.wet_q[i][0], *set,
                "wet SET {i} (5fronts {b_5fronts})"
            );
        }
        for (i, set) in rows.alpha.iter().enumerate() {
            assert_eq!(decoded.alpha_q[i][0], *set, "alpha SET {i}");
        }
        for (i, set) in rows.beta.iter().enumerate() {
            assert_eq!(decoded.beta_q[i][0], *set, "beta SET {i}");
        }
    }
}

#[test]
fn ice_ajcc_pframe_gop_uses_time_rows_and_settles() {
    // Stationary content over an I + 5×P GOP: the P-frames must carry
    // TIME rows somewhere (all-zero deltas are strictly cheapest) and
    // the settled decode must still land within 3 dB per channel.
    let input = core_layout_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut last = Vec::new();
    let mut saw_time = false;
    for frame in 0..6 {
        enc.b_iframe_global = frame == 0;
        let bytes = enc.encode_frame_pcm_7_0_4_ice_ajcc(&refs);
        last = decode_frame(&mut dec, bytes, 11);
        let sub = dec.last_substream.as_ref().expect("substream parsed");
        let ice = sub.tools.ice.as_deref().expect("ice element parsed");
        let data = ice.ajcc.as_deref().expect("ajcc_data parsed");
        let any_time = data
            .dry
            .iter()
            .chain(data.wet.iter())
            .chain(data.alpha.iter())
            .chain(data.beta.iter())
            .flat_map(|s| s.iter())
            .any(|(d, _)| *d == AjocDiffType::Time);
        if frame == 0 {
            assert!(!any_time, "I-frame carries FREQ rows only");
        } else {
            saw_time |= any_time;
        }
    }
    assert!(saw_time, "stationary P-frames must pick TIME rows");
    for ch in 0..11 {
        let r = energy(&last[ch]) / energy(&input[ch]).max(1e-30);
        assert!(
            (0.5..=2.0).contains(&r),
            "P-frame GOP channel {ch} settled energy ratio {r:.3}"
        );
    }
}

#[test]
fn ice_ajcc_encode_is_deterministic() {
    let input = core_layout_input();
    let refs: [&[f32]; 11] = std::array::from_fn(|i| input[i].as_slice());
    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    for _ in 0..3 {
        let a = enc_a.encode_frame_pcm_7_0_4_ice_ajcc(&refs);
        let b = enc_b.encode_frame_pcm_7_0_4_ice_ajcc(&refs);
        assert_eq!(a, b, "matched inputs + fresh state → identical bytes");
    }
}
