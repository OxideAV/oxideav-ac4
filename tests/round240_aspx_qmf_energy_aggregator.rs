//! Round 240 — encoder-side HF QMF energy aggregator.
//!
//! Pins the new `aggregate_qmf_to_sbg_atsg` / per-channel scf extractors
//! and the QMF-driven envelope builder shortcut, which together complete
//! the encoder's `q_high → on-wire bytes` chain for real ASPX envelope
//! coding. Refs ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocodes 90 + 91 +
//! §5.7.6.3.5 Pseudocodes 82 + 83.

use oxideav_ac4::aspx::AspxQuantStep;
use oxideav_ac4::encoder_acpl3::{
    aggregate_qmf_to_sbg_atsg, build_aspx_real_envelope_channel,
    build_aspx_real_envelope_channel_from_qmf, extract_aspx_noise_envelope_scf_from_qmf,
    extract_aspx_sig_envelope_scf_from_qmf, AspxEnvelopeScfChannel, AspxQmfEnvelopeChannel,
};

/// Build a synthetic HF QMF matrix where every (sb, ts) cell has
/// complex magnitude squared equal to `f(sb, ts)`. Returns a matrix
/// shaped `[sb][ts]` of length `num_sb × num_ts`. Each cell is split
/// evenly between real and imaginary so `re² + im² == f(sb, ts)`.
fn synthetic_qmf<F: Fn(usize, usize) -> f32>(
    num_sb: usize,
    num_ts: usize,
    f: F,
) -> Vec<Vec<(f32, f32)>> {
    (0..num_sb)
        .map(|sb| {
            (0..num_ts)
                .map(|ts| {
                    let mag2 = f(sb, ts).max(0.0);
                    let half = (mag2 / 2.0).sqrt();
                    (half, half)
                })
                .collect()
        })
        .collect()
}

fn assert_close(actual: f32, expected: f32, tol: f32, ctx: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{ctx}: got {actual}, expected {expected} (tol {tol})"
    );
}

#[test]
fn aggregate_constant_energy_matches_average() {
    // 4 QMF subbands, 8 time slots, every cell carries energy 4.0.
    let q = synthetic_qmf(4, 8, |_, _| 4.0);
    let sbg = [0u32, 2, 4];
    let atsg = [0u32, 2];
    let agg = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 0);
    assert_eq!(agg.len(), 2);
    assert_eq!(agg[0].len(), 1);
    assert_eq!(agg[1].len(), 1);
    // Per-(sbg, atsg) average squared magnitude = 4.0 (constant field).
    assert_close(agg[0][0], 4.0, 1e-5, "sbg=0 atsg=0 constant");
    assert_close(agg[1][0], 4.0, 1e-5, "sbg=1 atsg=0 constant");
}

#[test]
fn aggregate_partitions_correctly_across_atsg() {
    // 2 subbands, 8 time slots. First half (ts < 4) has energy 1.0, second
    // half has energy 9.0. With atsg borders [0, 1, 2] and num_ts_in_ats = 4,
    // we expect [1.0, 9.0] across atsg.
    let q = synthetic_qmf(2, 8, |_, ts| if ts < 4 { 1.0 } else { 9.0 });
    let sbg = [0u32, 2];
    let atsg = [0u32, 1, 2];
    let agg = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 0);
    assert_eq!(agg.len(), 1);
    assert_eq!(agg[0].len(), 2);
    assert_close(agg[0][0], 1.0, 1e-5, "atsg=0 low-energy half");
    assert_close(agg[0][1], 9.0, 1e-5, "atsg=1 high-energy half");
}

#[test]
fn aggregate_partitions_correctly_across_sbg() {
    // 4 subbands, 4 time slots. SB 0..2 carries energy 1.0, SB 2..4 carries
    // energy 16.0. SBG split [0, 2, 4] should give [1.0, 16.0].
    let q = synthetic_qmf(4, 4, |sb, _| if sb < 2 { 1.0 } else { 16.0 });
    let sbg = [0u32, 2, 4];
    let atsg = [0u32, 1];
    let agg = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 0);
    assert_close(agg[0][0], 1.0, 1e-5, "sbg=0 low-energy bands");
    assert_close(agg[1][0], 16.0, 1e-5, "sbg=1 high-energy bands");
}

#[test]
fn aggregate_respects_sbx_clamp() {
    // sbg_borders below sbx should clamp upward to sbx — caller passes
    // spec-shaped absolute borders even if sbx > 0.
    let q = synthetic_qmf(8, 4, |sb, _| (sb + 1) as f32);
    let sbg = [0u32, 4, 8];
    let atsg = [0u32, 1];
    let result_no_sbx = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 0);
    let result_sbx2 = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 2);
    // With sbx = 2, the first sbg shrinks to bands [2, 4) which averages
    // ((3.0 + 4.0) / 2) = 3.5; the second sbg is unaffected.
    assert_close(result_sbx2[0][0], 3.5, 1e-5, "sbg=0 clamped to sbx=2");
    // Second SBG unaffected by clamp.
    assert_close(
        result_sbx2[1][0],
        result_no_sbx[1][0],
        1e-5,
        "sbg=1 sbx-invariant",
    );
}

#[test]
fn aggregate_empty_borders_returns_empty() {
    let q = synthetic_qmf(2, 2, |_, _| 1.0);
    let zero_sbg: [u32; 0] = [];
    let atsg = [0u32, 1];
    let agg = aggregate_qmf_to_sbg_atsg(&q, &zero_sbg, &atsg, 2, 0);
    assert!(agg.is_empty(), "empty sbg borders returns empty");

    let sbg = [0u32, 2];
    let zero_atsg: [u32; 0] = [];
    let agg2 = aggregate_qmf_to_sbg_atsg(&q, &sbg, &zero_atsg, 2, 0);
    assert_eq!(agg2.len(), 1);
    assert!(
        agg2[0].is_empty(),
        "empty atsg borders returns one row of length 0"
    );
}

#[test]
fn aggregate_zero_span_atsg_returns_zero() {
    // atsg with zero-span (atsg_borders[1] == atsg_borders[0]) returns 0.0
    // for the affected atsg cell.
    let q = synthetic_qmf(2, 4, |_, _| 5.0);
    let sbg = [0u32, 2];
    // Two atsgs: first is zero-span [0, 0], second spans [0, 1] → ts
    // range [0, num_ts_in_ats=4) → energy 5.0 per cell.
    let atsg = [0u32, 0, 1];
    let agg = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 0);
    assert_eq!(agg[0].len(), 2);
    assert_close(agg[0][0], 0.0, 1e-7, "zero-span atsg returns 0");
    assert_close(agg[0][1], 5.0, 1e-5, "non-zero-span atsg returns mean");
}

#[test]
fn extract_sig_scf_from_qmf_emits_per_sbg_vector() {
    let q = synthetic_qmf(4, 4, |sb, _| (sb + 1) as f32);
    let sbg = [0u32, 2, 4];
    let scf = extract_aspx_sig_envelope_scf_from_qmf(&q, &sbg, 4, 1, 0);
    assert_eq!(scf.len(), 2, "one entry per sbg");
    // sbg=0 bands [0, 2) → mean(1, 2) = 1.5
    assert_close(scf[0], 1.5, 1e-5, "sbg=0 mean");
    // sbg=1 bands [2, 4) → mean(3, 4) = 3.5
    assert_close(scf[1], 3.5, 1e-5, "sbg=1 mean");
}

#[test]
fn extract_noise_scf_from_qmf_emits_per_sbg_vector() {
    let q = synthetic_qmf(4, 4, |_, _| 64.0);
    let sbg = [0u32, 4];
    let scf = extract_aspx_noise_envelope_scf_from_qmf(&q, &sbg, 4, 1, 0);
    assert_eq!(scf.len(), 1);
    assert_close(scf[0], 64.0, 1e-4, "single-band mean");
}

#[test]
fn extract_sig_scf_empty_borders_returns_empty() {
    let q = synthetic_qmf(2, 2, |_, _| 1.0);
    let empty: [u32; 0] = [];
    assert!(extract_aspx_sig_envelope_scf_from_qmf(&q, &empty, 2, 1, 0).is_empty());
    let single = [0u32];
    assert!(extract_aspx_sig_envelope_scf_from_qmf(&q, &single, 2, 1, 0).is_empty());
}

#[test]
fn build_from_qmf_matches_two_step_path() {
    // Verify that the QMF-driven convenience builder matches the manual
    // chain `extract_*_scf_from_qmf → AspxEnvelopeScfChannel →
    // build_aspx_real_envelope_channel`.
    let q = synthetic_qmf(4, 4, |sb, ts| {
        // Gentle gradient so the chosen sbg borders produce distinct scf values.
        ((sb + 1) * (ts + 1)) as f32
    });
    let sbg_sig = [0u32, 2, 4];
    let sbg_noise = [0u32, 4];
    let ch = AspxQmfEnvelopeChannel {
        q_high: &q,
        sbg_sig_borders: &sbg_sig,
        sbg_noise_borders: &sbg_noise,
    };
    let (sig_q, noise_q) =
        build_aspx_real_envelope_channel_from_qmf(&ch, AspxQuantStep::Fine, 64, 4, 1, 0);

    // Manual chain.
    let sig_scf = extract_aspx_sig_envelope_scf_from_qmf(&q, &sbg_sig, 4, 1, 0);
    let noise_scf = extract_aspx_noise_envelope_scf_from_qmf(&q, &sbg_noise, 4, 1, 0);
    let scf_ch = AspxEnvelopeScfChannel {
        sig: &sig_scf,
        noise: &noise_scf,
    };
    let (sig_m, noise_m) = build_aspx_real_envelope_channel(&scf_ch, AspxQuantStep::Fine, 64);
    assert_eq!(sig_q, sig_m, "sig DPCM payload matches");
    assert_eq!(noise_q, noise_m, "noise DPCM payload matches");
}

#[test]
fn qmf_chain_hits_integer_quant_grid_points() {
    // QMF → aggregator → extractor pipeline. Pick scf grid points that
    // sit on integer qscf so the DPCM payload is predictable:
    //   Fine sig: scf = 64 · 2^(qscf/2); qscf = 0 → 64, qscf = 2 → 128.
    // Two SBGs, each holding one QMF subband; constant energy per band
    // makes the aggregator average == the per-cell value.
    let half_64 = (64.0_f32 / 2.0).sqrt();
    let half_128 = (128.0_f32 / 2.0).sqrt();
    let q = vec![vec![(half_64, half_64); 4], vec![(half_128, half_128); 4]];
    let sbg_sig = [0u32, 1, 2];
    // Per-sbg scf agg.
    let scf = extract_aspx_sig_envelope_scf_from_qmf(&q, &sbg_sig, 4, 1, 0);
    assert_close(scf[0], 64.0, 1e-3, "sig sbg=0 hits 64");
    assert_close(scf[1], 128.0, 1e-3, "sig sbg=1 hits 128");
    let sig_dpcm = oxideav_ac4::encoder_acpl3::extract_aspx_sig_envelope_indices(
        &scf,
        AspxQuantStep::Fine,
        64,
    );
    // F0 = qscf[0] = 0; DF₁ = qscf[1] − qscf[0] = 2 − 0 = 2.
    assert_eq!(sig_dpcm, vec![0, 2], "DPCM payload matches integer quant");
}

#[test]
fn aggregate_handles_short_qmf_rows() {
    // QMF rows shorter than tsz contribute zero past their end; the
    // aggregator must not panic and must still divide by the full
    // ts_span (matching the decoder's bounds-checked Pseudocode 90 path).
    let mut q = vec![vec![(1.0_f32, 1.0_f32); 4]; 2];
    q[1].truncate(2); // sb 1 only has 2 time slots.
    let sbg = [0u32, 2];
    let atsg = [0u32, 1];
    let agg = aggregate_qmf_to_sbg_atsg(&q, &sbg, &atsg, 4, 0);
    // sb 0 contributes 4 × (1² + 1²) = 8; sb 1 contributes 2 × 2 = 4; total
    // 12; band_span = 2; ts_span = 4; mean = 12 / (2 × 4) = 1.5.
    assert_close(agg[0][0], 1.5, 1e-5, "short row contributes partial energy");
}

#[test]
fn build_from_qmf_is_deterministic() {
    let q = synthetic_qmf(8, 8, |sb, ts| ((sb * 3 + ts) % 7 + 1) as f32);
    let sbg_sig = [0u32, 2, 5, 8];
    let sbg_noise = [0u32, 4, 8];
    let ch = AspxQmfEnvelopeChannel {
        q_high: &q,
        sbg_sig_borders: &sbg_sig,
        sbg_noise_borders: &sbg_noise,
    };
    let a = build_aspx_real_envelope_channel_from_qmf(&ch, AspxQuantStep::Coarse, 64, 4, 2, 0);
    let b = build_aspx_real_envelope_channel_from_qmf(&ch, AspxQuantStep::Coarse, 64, 4, 2, 0);
    assert_eq!(a, b, "deterministic across repeated calls");
}

#[test]
fn different_qmf_inputs_produce_different_dpcm_payloads() {
    let q1 = synthetic_qmf(4, 4, |_, _| 1.0);
    let q2 = synthetic_qmf(4, 4, |sb, _| ((sb + 1) * 2) as f32);
    let sbg_sig = [0u32, 2, 4];
    let sbg_noise = [0u32, 4];
    let ch_a = AspxQmfEnvelopeChannel {
        q_high: &q1,
        sbg_sig_borders: &sbg_sig,
        sbg_noise_borders: &sbg_noise,
    };
    let ch_b = AspxQmfEnvelopeChannel {
        q_high: &q2,
        sbg_sig_borders: &sbg_sig,
        sbg_noise_borders: &sbg_noise,
    };
    let (sig_a, _) =
        build_aspx_real_envelope_channel_from_qmf(&ch_a, AspxQuantStep::Fine, 64, 4, 1, 0);
    let (sig_b, _) =
        build_aspx_real_envelope_channel_from_qmf(&ch_b, AspxQuantStep::Fine, 64, 4, 1, 0);
    assert_ne!(
        sig_a, sig_b,
        "distinct QMF inputs produce distinct payloads"
    );
}
