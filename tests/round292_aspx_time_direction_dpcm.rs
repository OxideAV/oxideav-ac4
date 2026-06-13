//! Round 292 — encoder-side TIME-direction ASPX envelope DPCM packing.
//!
//! The round-219/226/234/240 envelope-coding chain only ever emitted the
//! **FREQ** transmission direction (`freq_dpcm_encode_qscf`): first-
//! difference of `qscf` across subband groups, one envelope at a time.
//! The decoder, however, accepts a per-envelope direction flag and walks
//! a **TIME** branch when `direction_time == true`
//! ([`oxideav_ac4::aspx::delta_decode_sig`] / `delta_decode_noise`):
//!
//! ```text
//!   qscf[sbg][atsg] = prev[sbg] + delta · values[sbg]
//! ```
//!
//! where `prev` is the previous envelope's row (or `qscf_prev_last` for
//! the first envelope). This round adds the encoder dual —
//! [`oxideav_ac4::encoder_acpl3::time_dpcm_encode_qscf`] for one envelope
//! and [`oxideav_ac4::encoder_acpl3::dpcm_encode_qscf_envelopes`] for a
//! full multi-envelope `qscf[sbg][atsg]` matrix with per-envelope
//! min-L1 direction selection — and pins the bit-exact round-trip
//! against the decoder's `delta_decode_*`.
//!
//! Refs ETSI TS 103 190-1 §5.7.6.3.4 Pseudocode 80 / 81.

use oxideav_ac4::aspx::{delta_decode_noise, delta_decode_sig, AspxHuffEnv};
use oxideav_ac4::encoder_acpl3::{
    dpcm_encode_qscf_envelopes, freq_dpcm_encode_qscf, time_dpcm_encode_qscf, AspxEncodedEnvelope,
};

/// `time_dpcm_encode_qscf` is the exact inverse of the decoder's TIME
/// branch: encoding `qscf` against `prev` with step `delta`, then
/// decoding the deltas back, recovers `qscf`.
#[test]
fn time_dpcm_round_trips_through_decoder_time_branch() {
    let qscf = vec![4, 3, 5, 5];
    let prev = vec![3, 3, 4, 6];
    let delta = 1;

    let values = time_dpcm_encode_qscf(&qscf, &prev, delta);
    // Each transmitted DT = qscf - prev (delta = 1).
    assert_eq!(values, vec![1, 0, 1, -1]);

    // Decode the single TIME envelope back; prev becomes qscf_prev_last.
    let deltas = vec![AspxHuffEnv {
        values: values.clone(),
        direction_time: true,
    }];
    let decoded = delta_decode_sig(&deltas, qscf.len() as u32, &prev, delta);
    let recovered: Vec<i32> = (0..qscf.len()).map(|sbg| decoded[sbg][0]).collect();
    assert_eq!(recovered, qscf);
}

/// A `delta = -1` step is handled (the division by the step sign keeps
/// the encode/decode pair consistent).
#[test]
fn time_dpcm_handles_negative_delta_step() {
    let qscf = vec![2, -1, 4];
    let prev = vec![0, 0, 0];
    let delta = -1;

    let values = time_dpcm_encode_qscf(&qscf, &prev, delta);
    // values = (qscf - prev) / -1.
    assert_eq!(values, vec![-2, 1, -4]);

    let deltas = vec![AspxHuffEnv {
        values,
        direction_time: true,
    }];
    let decoded = delta_decode_sig(&deltas, qscf.len() as u32, &prev, delta);
    let recovered: Vec<i32> = (0..qscf.len()).map(|sbg| decoded[sbg][0]).collect();
    assert_eq!(recovered, qscf);
}

/// A length-short `prev` zero-extends, mirroring the decoder's
/// `qscf_prev_last.get(sbg).unwrap_or(0)`.
#[test]
fn time_dpcm_zero_extends_short_prev() {
    let qscf = vec![5, 7, 9];
    let prev = vec![5]; // only sbg 0 has history.
    let delta = 1;

    let values = time_dpcm_encode_qscf(&qscf, &prev, delta);
    assert_eq!(values, vec![0, 7, 9]);

    let deltas = vec![AspxHuffEnv {
        values,
        direction_time: true,
    }];
    let decoded = delta_decode_sig(&deltas, qscf.len() as u32, &prev, delta);
    let recovered: Vec<i32> = (0..qscf.len()).map(|sbg| decoded[sbg][0]).collect();
    assert_eq!(recovered, qscf);
}

/// Empty input returns empty output.
#[test]
fn time_dpcm_empty_input_is_empty() {
    assert!(time_dpcm_encode_qscf(&[], &[], 1).is_empty());
    assert!(time_dpcm_encode_qscf(&[], &[1, 2], 1).is_empty());
}

/// A `delta = 0` (never produced by a conformant decoder) is treated as
/// `delta = 1` so the encoder stays total / panic-free.
#[test]
fn time_dpcm_delta_zero_is_total() {
    let qscf = vec![3, 4];
    let prev = vec![1, 1];
    let values = time_dpcm_encode_qscf(&qscf, &prev, 0);
    assert_eq!(values, vec![2, 3]);
}

/// The multi-envelope packer reproduces the decoder's input matrix
/// exactly when its `(direction_time, values)` rows are fed back through
/// `delta_decode_sig` with the same `delta` + `qscf_prev_last`.
#[test]
fn multi_envelope_packer_round_trips_through_decoder() {
    // qscf[sbg][atsg] — 4 subband groups, 3 envelopes.
    let qscf = vec![
        vec![4, 5, 5], // sbg 0
        vec![3, 4, 4], // sbg 1
        vec![5, 6, 7], // sbg 2
        vec![5, 6, 6], // sbg 3
    ];
    let prev_last = vec![3, 3, 4, 6];
    let delta = 1;

    let rows = dpcm_encode_qscf_envelopes(&qscf, &prev_last, delta, false);
    assert_eq!(rows.len(), 3, "one row per envelope");

    let deltas: Vec<AspxHuffEnv> = rows
        .iter()
        .map(|r| AspxHuffEnv {
            values: r.values.clone(),
            direction_time: r.direction_time,
        })
        .collect();
    let decoded = delta_decode_sig(&deltas, qscf.len() as u32, &prev_last, delta);
    for sbg in 0..qscf.len() {
        for atsg in 0..qscf[0].len() {
            assert_eq!(
                decoded[sbg][atsg], qscf[sbg][atsg],
                "mismatch at sbg={sbg} atsg={atsg}"
            );
        }
    }
}

/// The packer is direction-agnostic to the consumer: it round-trips the
/// noise decoder too (same delta semantics).
#[test]
fn multi_envelope_packer_round_trips_through_noise_decoder() {
    let qscf = vec![vec![6, 6], vec![4, 5]];
    let prev_last: Vec<i32> = Vec::new();
    let delta = 1;

    let rows = dpcm_encode_qscf_envelopes(&qscf, &prev_last, delta, false);
    let deltas: Vec<AspxHuffEnv> = rows
        .iter()
        .map(|r| AspxHuffEnv {
            values: r.values.clone(),
            direction_time: r.direction_time,
        })
        .collect();
    let decoded = delta_decode_noise(&deltas, qscf.len() as u32, &prev_last, delta);
    for sbg in 0..qscf.len() {
        for atsg in 0..qscf[0].len() {
            assert_eq!(decoded[sbg][atsg], qscf[sbg][atsg]);
        }
    }
}

/// When a later envelope is nearly flat against its predecessor, the
/// min-L1 policy picks TIME for it (cheaper than re-coding the absolute
/// shape in FREQ).
#[test]
fn packer_picks_time_for_temporally_stable_envelope() {
    // Envelope 0 has a non-trivial FREQ shape; envelope 1 is env 0 + a
    // tiny per-sbg jitter, so TIME (Σ|Δ| = 2) beats FREQ (Σ|first-diff|).
    let qscf = vec![
        vec![10, 11], // sbg 0
        vec![2, 2],   // sbg 1
        vec![8, 9],   // sbg 2
        vec![1, 1],   // sbg 3
    ];
    let prev_last: Vec<i32> = Vec::new();
    let delta = 1;

    let rows = dpcm_encode_qscf_envelopes(&qscf, &prev_last, delta, false);
    assert_eq!(rows.len(), 2);
    // Envelope 1 should pick TIME: deltas vs env 0 are [1,0,1,0]
    // (Σ = 2), far cheaper than FREQ on the [11,2,9,1] absolute column
    // (first-difference Σ = |11| + |-9| + |7| + |-8| = 35).
    assert!(rows[1].direction_time, "stable envelope 1 codes TIME");
    assert_eq!(rows[1].values, vec![1, 0, 1, 0]);
    // Whatever direction envelope 0 took, the pair still reconstructs the
    // exact input matrix through the decoder.
    let deltas: Vec<AspxHuffEnv> = rows
        .iter()
        .map(|r| AspxHuffEnv {
            values: r.values.clone(),
            direction_time: r.direction_time,
        })
        .collect();
    let decoded = delta_decode_sig(&deltas, qscf.len() as u32, &prev_last, delta);
    for sbg in 0..qscf.len() {
        for atsg in 0..qscf[0].len() {
            assert_eq!(decoded[sbg][atsg], qscf[sbg][atsg]);
        }
    }
}

/// FREQ wins ties — an envelope whose TIME and FREQ costs are equal is
/// transmitted FREQ (no cross-envelope state needed), matching the
/// decoder's independent-envelope default.
#[test]
fn packer_prefers_freq_on_cost_tie() {
    // Single envelope, no history: TIME reference is all-zero so the TIME
    // deltas equal the raw qscf [3, 3] (Σ = 6); FREQ deltas are
    // [3, 0] (Σ = 3) — FREQ strictly cheaper here. Use a flat column to
    // force the tie: qscf [0, 0] → both directions Σ = 0.
    let qscf = vec![vec![0], vec![0]];
    let prev_last: Vec<i32> = Vec::new();
    let rows = dpcm_encode_qscf_envelopes(&qscf, &prev_last, 1, false);
    assert_eq!(rows.len(), 1);
    assert!(!rows[0].direction_time, "tie → FREQ");
}

/// `force_freq = true` codes every envelope FREQ and matches
/// `freq_dpcm_encode_qscf` column-for-column — the legacy scaffold path.
#[test]
fn packer_force_freq_matches_freq_only_encoder() {
    let qscf = vec![vec![10, 11], vec![2, 2], vec![8, 9]];
    let prev_last = vec![5, 5, 5];

    let rows = dpcm_encode_qscf_envelopes(&qscf, &prev_last, 1, true);
    assert_eq!(rows.len(), 2);
    for (atsg, row) in rows.iter().enumerate() {
        assert!(!row.direction_time, "force_freq → all FREQ");
        let col: Vec<i32> = (0..qscf.len()).map(|sbg| qscf[sbg][atsg]).collect();
        assert_eq!(row.values, freq_dpcm_encode_qscf(&col));
    }
}

/// Empty matrix (no subband groups) returns no rows.
#[test]
fn packer_empty_matrix_is_empty() {
    let qscf: Vec<Vec<i32>> = Vec::new();
    assert!(dpcm_encode_qscf_envelopes(&qscf, &[], 1, false).is_empty());
}

/// The encoded-envelope struct is the encoder-side mirror of
/// `AspxHuffEnv` and constructs cleanly.
#[test]
fn encoded_envelope_struct_default_and_fields() {
    let e = AspxEncodedEnvelope::default();
    assert!(e.values.is_empty());
    assert!(!e.direction_time);

    let e2 = AspxEncodedEnvelope {
        values: vec![1, -2, 3],
        direction_time: true,
    };
    assert_eq!(e2.values, vec![1, -2, 3]);
    assert!(e2.direction_time);
}
