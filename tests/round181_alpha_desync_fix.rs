//! Round 181 — close the r128 ALPHA `alpha_q` desync followup at two
//! distinct layers.
//!
//! ### Layer 1 — §5.7.7.7 Pseudocode 121 parser indexing
//!
//! ETSI TS 103 190-1 §4.2.13.7 Table 65 indexes the `a_huff_data[i]`
//! array by parameter band `i` from `start_band` to `data_bands`
//! exclusive. §5.7.7.7 Pseudocode 121 then walks `i in 0..num_bands`
//! over the same `acpl_<SET>[ps]` array — implicitly treating
//! `acpl_<SET>[ps][0..start_band]` as zero. Pre-r181 the parser packed
//! the `(num_bands - start_band)` Huffman values into a vector starting
//! at index 0, which silently shifted the §5.7.7.7 DIFF_FREQ
//! accumulation by `start_band` parameter bands. For the 5_X
//! SIMPLE/ASPX_ACPL_1 PARTIAL path (`acpl_qmf_band > 0`,
//! `start_band > 0`) this produced an off-by-`start_band` drift in the
//! recovered `alpha_q[pb]` row that the round-128 / 132 / 135 / 139 /
//! 144 PCM-path encoders' on-wire α/β indices visibly diverged from.
//!
//! ### Layer 2 — §4.2.12.4 Table 52 `aspx_data_2ch()` SIGNAL band count
//!
//! Per ETSI TS 103 190-1 §4.3.10.4.9 (Table 124 NOTE 3) `aspx_ec_data()`
//! reads `num_sbg_sig_lowres` SIGNAL bands when the corresponding
//! `aspx_freq_res[env]` bit was emitted as 0, and `num_sbg_sig_highres`
//! when it was 1 or absent (no in-band bit because
//! `freq_res_mode != Signalled`). Pre-r181 the encoder's
//! `write_aspx_data_2ch_minimal` hard-coded `num_sbg_sig_lowres`
//! regardless — so for the encoder's default `freq_res_mode =
//! DurationDependent` configuration (no in-band `aspx_freq_res` bit),
//! the parser's high-res fallback read 20 SIGNAL codewords per channel
//! while the writer emitted only 10. The 20-vs-10 mismatch buried every
//! subsequent `acpl_data_1ch()` α / β codeword in trailing zero-padding
//! and silently produced all-zero recovered indices — the upstream
//! bit-position drift the user's followup observed sitting on top of
//! the Layer-1 issue.
//!
//! ### Tests
//!
//! Layer 1 is pinned by the two standalone writer→parser round-trip
//! tests below. Layer 2 is pinned by the end-to-end 5.0 ASPX_ACPL_2
//! encode→decode flow — the ACPL_2 path is the cleanest reproducer
//! because it has no joint-MDCT residual layer between the ASPX
//! trailers and the `acpl_data_1ch()` pair (so once the ASPX walker is
//! aligned the `acpl_data` is the next read). The ACPL_1 path retains
//! a separate residual-layer alignment issue that's tracked as the
//! remaining follow-up — fixing it would require sf_data scale-factor
//! round-trip verification through the joint-MDCT residual writer, which
//! is out of scope for this round.

use oxideav_ac4::acpl::{self, AcplData1ch, AcplQuantMode};
use oxideav_ac4::acpl_synth::{self, AcplDiffState};
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::mch::FiveXCodecMode;
use oxideav_core::bits::BitReader;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn make_tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Standalone writer → parser → differential-decode round trip, with
/// `start_band == 2` exercising the PARTIAL `acpl_config_1ch` path.
#[test]
fn standalone_alpha_writer_round_trips_through_parser() {
    let qm = AcplQuantMode::Fine;
    let num_bands: u32 = 7;
    let start_band: u32 = 2;
    let alpha_q_in: Vec<i32> = vec![0, 0, 1, -2, 3, -1, 0];
    let beta_q_in: Vec<i32> = vec![0; num_bands as usize];

    let bytes = encoder_acpl3::write_acpl_data_1ch_real_alpha_beta_bytes(
        num_bands,
        start_band,
        qm,
        &alpha_q_in,
        &beta_q_in,
    );

    let mut br = BitReader::new(&bytes);
    let data = acpl::parse_acpl_data_1ch(&mut br, num_bands, start_band, qm)
        .expect("parser must accept the writer's output");

    let mut state = AcplDiffState::default();
    let alpha_rows = acpl_synth::differential_decode(&data.alpha1, num_bands, &mut state);
    assert_eq!(alpha_rows.len(), 1, "one parameter set expected");
    assert_eq!(
        alpha_rows[0], alpha_q_in,
        "recovered alpha_q must match the writer's input at spec param-band indexing"
    );
}

/// `parse_acpl_huff_data`'s recovered values are now spec-aligned per
/// §5.7.7.7 Pseudocode 121: positions `[0..start_band)` are zero, the
/// F0 codeword's signed `alpha_q[start_band]` lands at
/// `values[start_band]`, and DF deltas occupy
/// `[start_band+1..num_bands)`.
#[test]
fn parser_values_are_indexed_by_full_param_band_number() {
    let qm = AcplQuantMode::Coarse;
    let num_bands: u32 = 9; // table_143 id = 2
    let start_band: u32 = 3;
    let alpha_q_in: Vec<i32> = vec![0, 0, 0, 1, -1, 2, -2, 3, -3];
    let beta_q_in: Vec<i32> = vec![0; num_bands as usize];

    let bytes = encoder_acpl3::write_acpl_data_1ch_real_alpha_beta_bytes(
        num_bands,
        start_band,
        qm,
        &alpha_q_in,
        &beta_q_in,
    );

    let mut br = BitReader::new(&bytes);
    let data = acpl::parse_acpl_data_1ch(&mut br, num_bands, start_band, qm).expect("parse");

    assert_eq!(data.alpha1.len(), 1, "one parameter set");
    let values = &data.alpha1[0].values;
    assert_eq!(
        values.len(),
        num_bands as usize,
        "parser must return one entry per param band (full num_bands array)"
    );
    // Bands below start_band are zero (the encoder did not transmit
    // them — Pseudocode 121 carries them through as no-ops).
    for (pb, &v) in values.iter().enumerate().take(start_band as usize) {
        assert_eq!(v, 0, "band {pb} (below start_band) must be zero");
    }
    // Band at start_band is the F0 signed `alpha_q`.
    assert_eq!(
        values[start_band as usize], alpha_q_in[start_band as usize],
        "F0 lands at param band start_band"
    );
    // Bands above are DF deltas — the cumulative sum from F0 should
    // reproduce the original signed `alpha_q[pb]`.
    let mut acc = values[start_band as usize];
    for pb in (start_band as usize + 1)..num_bands as usize {
        acc += values[pb];
        assert_eq!(
            acc, alpha_q_in[pb],
            "Pseudocode 121 DIFF_FREQ accumulation at band {pb}"
        );
    }
}

/// Layer 2 end-to-end fix: encode a 5.0 ASPX_ACPL_2 frame with
/// asymmetric L/Ls, decode it, and verify the decoder recovers a
/// **non-zero** per-band `alpha_q` row on both `acpl_data_1ch_pair`
/// slots. Pre-r181 the `aspx_data_2ch()` 20-vs-10 SIGNAL band-count
/// mismatch drove the parser into trailing zero-padding before
/// `parse_acpl_data_1ch` was called, leaving `pair0.alpha1` and
/// `pair1.alpha1` both reading length-7 all-zero rows (the user's
/// "alpha_q desync" symptom).
#[test]
fn end_to_end_acpl2_asymmetric_surround_recovers_nonzero_alpha() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    // Asymmetric: Ls / Rs energy ≪ L / R energy → analytic α drifts
    // off the symmetric peak and quantises to a non-zero lane in
    // multiple parameter bands.
    let l = make_tone(220.0, 0.5);
    let r = make_tone(440.0, 0.5);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.05);
    let rs = make_tone(1100.0, 0.05);

    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&l, &r, &c, &ls, &rs]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");

    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(FiveXCodecMode::AspxAcpl2),
        "decoder must resolve ASPX_ACPL_2 mode"
    );
    let pair0: &AcplData1ch = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("D0 ACplModule pair must be populated");
    let pair1: &AcplData1ch = sub.tools.acpl_data_1ch_pair[1]
        .as_ref()
        .expect("D1 ACplModule pair must be populated");

    // FULL acpl_config_1ch carries no qmf_band → start_band = 0, every
    // parameter band participates in α / β coding.
    let cfg = sub
        .tools
        .acpl_config_1ch_full
        .as_ref()
        .expect("FULL acpl_config_1ch must be parsed");
    assert_eq!(cfg.qmf_band, 0, "FULL config: qmf_band = 0");

    assert_eq!(
        pair0.alpha1[0].values.len(),
        cfg.num_param_bands as usize,
        "parser must return one entry per param band"
    );

    // Both D0 (L→Ls) and D1 (R→Rs) ACplModules must carry non-zero
    // per-band α — pre-r181 the upstream ASPX desync left these
    // length-`num_param_bands` all-zero rows.
    let any_pair0_nonzero = pair0.alpha1[0].values.iter().any(|&v| v != 0);
    let any_pair1_nonzero = pair1.alpha1[0].values.iter().any(|&v| v != 0);
    assert!(
        any_pair0_nonzero,
        "asymmetric L/Ls must drive the recovered D0 alpha row off all-zero; \
         got values = {:?}",
        pair0.alpha1[0].values
    );
    assert!(
        any_pair1_nonzero,
        "asymmetric R/Rs must drive the recovered D1 alpha row off all-zero; \
         got values = {:?}",
        pair1.alpha1[0].values
    );
}

/// Silence input still round-trips through the ACPL_2 path post-r181 —
/// the Layer 2 fix changed the encoder's SIGNAL band-count
/// derivation, so the byte-stream is structurally different from r144;
/// the decoder must still resolve `FiveXCodecMode::AspxAcpl2` and
/// recover all-zero α / β rows for a fully-silent input.
#[test]
fn end_to_end_acpl2_silence_still_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let z = vec![0.0f32; N];
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl2_real_alpha_beta(&[&z, &z, &z, &z, &z]);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2);
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(sub.tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl2));
    let pair0 = sub.tools.acpl_data_1ch_pair[0].as_ref().expect("pair0");
    let all_zero = pair0.alpha1[0].values.iter().all(|&v| v == 0)
        && pair0.beta1[0].values.iter().all(|&v| v == 0);
    assert!(
        all_zero,
        "silence input must produce all-zero α / β rows; got alpha={:?} beta={:?}",
        pair0.alpha1[0].values, pair0.beta1[0].values
    );
}
