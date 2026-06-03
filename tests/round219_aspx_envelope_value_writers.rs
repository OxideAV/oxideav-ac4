//! Round 219 — ASPX envelope value-emitting helpers
//! (`write_aspx_sig_f0_value` / `write_aspx_sig_df_value` /
//! `write_aspx_sig_dt_value` / `write_aspx_noise_f0_value` /
//! `write_aspx_noise_df_value` / `write_aspx_noise_dt_value`).
//!
//! ### Background
//!
//! The round-95 ASPX scaffold in `encoder_acpl3.rs` emits each
//! `aspx_data_*()` body with the **minimum-bit-cost** codeword from
//! every Huffman codebook (`pick_min_len_cw` for F0, `pick_zero_delta_cw`
//! for DF/DT). This keeps the body well-formed for the decoder's
//! walker and lets the trailing ACPL parameter pair land where the
//! walker expects it — but the F0 codewords decode to whichever
//! symbol happens to carry the shortest codelength in each table,
//! not to envelope value 0. For `ASPX_HCB_ENV_LEVEL_15_F0` that
//! symbol is index 30 (LEN = 4); for `ASPX_HCB_ENV_LEVEL_30_F0` it's
//! also non-zero. The decoder's Pseudocode 80 then DPCM-accumulates
//! these into `qscf[sbg][atsg]`, which Pseudocode 82 turns into
//! `n_subbands · 2^(qscf/a)` — i.e. a loud HF replica.
//!
//! Closing the README's long-standing "real ASPX envelope coding"
//! deferral starts with the encoder being able to write an
//! **arbitrary** envelope quant index, not just the shortest one. The
//! round-219 value-emitting helpers cover that: each accepts an integer
//! index (`v` for F0, `delta_q` for DF/DT) and writes the matching
//! `(cw, len)` from the codebook selected by `(quant_mode,
//! stereo_mode)` for SIGNAL paths or `stereo_mode` alone for NOISE
//! paths.
//!
//! ### What this round measures
//!
//! 1. Round-trip — every `(quant_mode, stereo_mode, value)` the encoder
//!    can write is recovered exactly by the decoder's
//!    `parse_aspx_huff_data()` path (which composes `decode_delta()`
//!    over the matching codebook).
//! 2. F0 cb_off — F0 codebooks decode `symbol_index == v` directly
//!    (cb_off = 0). DF / DT codebooks decode
//!    `symbol_index - cb_off == delta_q`.
//! 3. Range clamping — values outside the codebook's addressable
//!    `[0, codebook_length)` (F0) or `[-cb_off, +cb_off]` (DF/DT)
//!    saturate to the codebook's extreme entries rather than
//!    panicking, matching the decoder's clamp semantics.
//! 4. Determinism — repeating a write produces an identical bit
//!    sequence.
//!
//! ### What this round does **not** change
//!
//! * No existing `write_aspx_*()` call site is touched. The
//!   `write_aspx_data_2ch_minimal` / `write_aspx_data_1ch_minimal`
//!   scaffolds still emit minimum-bit-cost codewords, so every
//!   existing test that pins the round-95 zero-delta scaffold byte-
//!   stream stays valid.
//! * No top-level encoder entry point is added.

use oxideav_ac4::aspx::{
    parse_aspx_huff_data, AspxDataType, AspxHcbType, AspxQuantStep, AspxSbgCounts, AspxStereoMode,
};
use oxideav_ac4::encoder_acpl3::{
    write_aspx_noise_df_value, write_aspx_noise_dt_value, write_aspx_noise_f0_value,
    write_aspx_sig_df_value, write_aspx_sig_dt_value, write_aspx_sig_f0_value,
};
use oxideav_core::bits::{BitReader, BitWriter};

/// Encode a single SIGNAL F0 codeword for `v` and parse it back via the
/// decoder's `parse_aspx_huff_data()` FREQ-direction path. With
/// `num_sbg = 1` the parser walks exactly one F0 codeword and returns
/// the recovered symbol as `values[0]`.
fn round_trip_sig_f0(quant: AspxQuantStep, stereo: AspxStereoMode, v: i32) -> i32 {
    let mut bw = BitWriter::new();
    write_aspx_sig_f0_value(&mut bw, quant, stereo, v);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let env = parse_aspx_huff_data(
        &mut br,
        AspxDataType::Signal,
        1, // num_sbg
        quant,
        stereo,
        false, // FREQ direction
    )
    .expect("decoder parsed F0");
    assert_eq!(env.values.len(), 1);
    env.values[0]
}

/// Encode a SIGNAL F0 + one DF codeword pair for `(v_f0, delta_q_df)`
/// and parse it back with `num_sbg = 2`.
fn round_trip_sig_f0_df(
    quant: AspxQuantStep,
    stereo: AspxStereoMode,
    v_f0: i32,
    delta_q_df: i32,
) -> (i32, i32) {
    let mut bw = BitWriter::new();
    write_aspx_sig_f0_value(&mut bw, quant, stereo, v_f0);
    write_aspx_sig_df_value(&mut bw, quant, stereo, delta_q_df);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let env = parse_aspx_huff_data(&mut br, AspxDataType::Signal, 2, quant, stereo, false)
        .expect("decoder parsed F0 + DF");
    assert_eq!(env.values.len(), 2);
    (env.values[0], env.values[1])
}

/// Encode a SIGNAL DT codeword for `delta_q` and parse it back via the
/// decoder's TIME-direction path. With `num_sbg = 1` and
/// `direction = true` the parser uses the DT codebook directly.
fn round_trip_sig_dt(quant: AspxQuantStep, stereo: AspxStereoMode, delta_q: i32) -> i32 {
    let mut bw = BitWriter::new();
    write_aspx_sig_dt_value(&mut bw, quant, stereo, delta_q);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let env = parse_aspx_huff_data(
        &mut br,
        AspxDataType::Signal,
        1,
        quant,
        stereo,
        true, // TIME direction
    )
    .expect("decoder parsed DT");
    assert_eq!(env.values.len(), 1);
    env.values[0]
}

/// Equivalent of `round_trip_sig_f0` for the NOISE codebooks. NOISE
/// codebooks share the F0 / DF / DT shape but have a single
/// `quant_mode` setting — the parser ignores `quant_mode` for NOISE
/// rows (per the §A.2 NOTE), so the encoder can also pass either
/// value and the round-trip still holds.
fn round_trip_noise_f0(stereo: AspxStereoMode, v: i32) -> i32 {
    let mut bw = BitWriter::new();
    write_aspx_noise_f0_value(&mut bw, stereo, v);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let env = parse_aspx_huff_data(
        &mut br,
        AspxDataType::Noise,
        1,
        // NOISE codebooks ignore quant_mode (§A.2 NOTE). Pass Fine to
        // exercise the decoder path that hits the codebook table the
        // encoder targeted.
        AspxQuantStep::Fine,
        stereo,
        false,
    )
    .expect("decoder parsed NOISE F0");
    assert_eq!(env.values.len(), 1);
    env.values[0]
}

/// SIGNAL F0 + DF round-trip across the full Cartesian product of
/// `(quant_mode, stereo_mode)` × a representative spread of envelope
/// indices. All four codebooks (`LEVEL_15_F0`, `BALANCE_15_F0`,
/// `LEVEL_30_F0`, `BALANCE_30_F0`) have different codebook lengths so
/// the test picks per-codebook in-range values rather than a single
/// fixed set.
#[test]
fn sig_f0_round_trips_across_quant_and_stereo() {
    // (codebook_length per pair below — index range is `0..len`):
    //   (Fine,  Level)   71 entries → values 0..70
    //   (Fine,  Balance) 25 entries → values 0..24
    //   (Coarse, Level)  36 entries → values 0..35
    //   (Coarse, Balance) 13 entries → values 0..12
    for (quant, stereo, hi) in [
        (AspxQuantStep::Fine, AspxStereoMode::Level, 70),
        (AspxQuantStep::Fine, AspxStereoMode::Balance, 24),
        (AspxQuantStep::Coarse, AspxStereoMode::Level, 35),
        (AspxQuantStep::Coarse, AspxStereoMode::Balance, 12),
    ] {
        for v in [0, 1, hi / 2, hi - 1, hi] {
            let recovered = round_trip_sig_f0(quant, stereo, v);
            assert_eq!(
                recovered, v,
                "F0 round-trip mismatch for ({quant:?}, {stereo:?}, v={v})"
            );
        }
    }
}

/// SIGNAL F0 + DF round-trip — covers the full symmetric DF range for
/// each codebook. The DF codebook's `cb_off` is the half-width:
/// (Fine, Level) 70; (Fine, Balance) 24; (Coarse, Level) 35;
/// (Coarse, Balance) 12.
#[test]
fn sig_df_round_trips_across_quant_and_stereo() {
    // Pick an interior F0 value so we don't waste the second sbg on a
    // boundary clamp.
    let v_f0 = 5;
    for (quant, stereo, cb_off) in [
        (AspxQuantStep::Fine, AspxStereoMode::Level, 70),
        (AspxQuantStep::Fine, AspxStereoMode::Balance, 24),
        (AspxQuantStep::Coarse, AspxStereoMode::Level, 35),
        (AspxQuantStep::Coarse, AspxStereoMode::Balance, 12),
    ] {
        for delta in [-cb_off, -1, 0, 1, cb_off] {
            let (f0, df) = round_trip_sig_f0_df(quant, stereo, v_f0, delta);
            assert_eq!(f0, v_f0);
            assert_eq!(
                df, delta,
                "DF round-trip mismatch for ({quant:?}, {stereo:?}, delta={delta})"
            );
        }
    }
}

/// SIGNAL DT (time-delta) round-trip — covers the symmetric DT range
/// per codebook (same `cb_off` as the DF tables per §A.2). The
/// time-direction path is exercised when an envelope's
/// `aspx_sig_delta_dir[env]` bit signals TIME (Pseudocode 80 TIME
/// branch).
#[test]
fn sig_dt_round_trips_across_quant_and_stereo() {
    for (quant, stereo, cb_off) in [
        (AspxQuantStep::Fine, AspxStereoMode::Level, 70),
        (AspxQuantStep::Fine, AspxStereoMode::Balance, 24),
        (AspxQuantStep::Coarse, AspxStereoMode::Level, 35),
        (AspxQuantStep::Coarse, AspxStereoMode::Balance, 12),
    ] {
        for delta in [-cb_off, -2, 0, 2, cb_off] {
            let recovered = round_trip_sig_dt(quant, stereo, delta);
            assert_eq!(
                recovered, delta,
                "DT round-trip mismatch for ({quant:?}, {stereo:?}, delta={delta})"
            );
        }
    }
}

/// NOISE F0 round-trip across both stereo modes — the NOISE codebooks
/// are 30 entries (Level, values 0..29) and 13 entries (Balance,
/// values 0..12).
#[test]
fn noise_f0_round_trips_across_stereo() {
    for (stereo, hi) in [(AspxStereoMode::Level, 29), (AspxStereoMode::Balance, 12)] {
        for v in [0, 1, hi / 2, hi - 1, hi] {
            let recovered = round_trip_noise_f0(stereo, v);
            assert_eq!(
                recovered, v,
                "NOISE F0 round-trip mismatch for ({stereo:?}, v={v})"
            );
        }
    }
}

/// NOISE DF round-trip — symmetric range `(-cb_off..=cb_off)` per
/// stereo mode. The DF codebooks: Level cb_off = 29 (59 entries);
/// Balance cb_off = 12 (25 entries).
#[test]
fn noise_df_round_trips_across_stereo() {
    for (stereo, cb_off) in [(AspxStereoMode::Level, 29), (AspxStereoMode::Balance, 12)] {
        for delta in [-cb_off, -3, 0, 3, cb_off] {
            let mut bw = BitWriter::new();
            // num_sbg = 2 → F0 + DF on the FREQ path. Use an interior
            // F0 value so it can't be confused with the codebook's
            // edge clamps.
            write_aspx_noise_f0_value(&mut bw, stereo, 5);
            write_aspx_noise_df_value(&mut bw, stereo, delta);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let env = parse_aspx_huff_data(
                &mut br,
                AspxDataType::Noise,
                2,
                AspxQuantStep::Fine,
                stereo,
                false,
            )
            .expect("decoder parsed NOISE F0 + DF");
            assert_eq!(env.values.len(), 2);
            assert_eq!(env.values[0], 5);
            assert_eq!(
                env.values[1], delta,
                "NOISE DF round-trip mismatch for ({stereo:?}, delta={delta})"
            );
        }
    }
}

/// NOISE DT round-trip — TIME direction (`direction = true`).
#[test]
fn noise_dt_round_trips_across_stereo() {
    for (stereo, cb_off) in [(AspxStereoMode::Level, 29), (AspxStereoMode::Balance, 12)] {
        for delta in [-cb_off, -2, 0, 2, cb_off] {
            let mut bw = BitWriter::new();
            write_aspx_noise_dt_value(&mut bw, stereo, delta);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let env = parse_aspx_huff_data(
                &mut br,
                AspxDataType::Noise,
                1,
                AspxQuantStep::Fine,
                stereo,
                true,
            )
            .expect("decoder parsed NOISE DT");
            assert_eq!(env.values.len(), 1);
            assert_eq!(
                env.values[0], delta,
                "NOISE DT round-trip mismatch for ({stereo:?}, delta={delta})"
            );
        }
    }
}

/// Out-of-range F0 values clamp to the codebook's last entry rather
/// than panicking. `LEVEL_15_F0` has 71 entries (values 0..70), so a
/// caller-supplied value of 100 should saturate to symbol index 70.
#[test]
fn sig_f0_value_clamps_above_range() {
    let recovered = round_trip_sig_f0(AspxQuantStep::Fine, AspxStereoMode::Level, 100);
    assert_eq!(recovered, 70);
}

/// Negative F0 values clamp to symbol 0 (since F0 codebooks have
/// `cb_off = 0`).
#[test]
fn sig_f0_value_clamps_below_range() {
    let recovered = round_trip_sig_f0(AspxQuantStep::Fine, AspxStereoMode::Balance, -5);
    assert_eq!(recovered, 0);
}

/// Out-of-range DF deltas clamp to the codebook's symmetric edge.
/// `LEVEL_15_DF` covers `-70..=70` (141 entries, cb_off = 70); a
/// caller-supplied delta of -200 should saturate to -70 (symbol 0).
#[test]
fn sig_df_value_clamps_below_range() {
    let (_, df) = round_trip_sig_f0_df(AspxQuantStep::Fine, AspxStereoMode::Level, 5, -200);
    assert_eq!(df, -70);
}

#[test]
fn sig_df_value_clamps_above_range() {
    let (_, df) = round_trip_sig_f0_df(AspxQuantStep::Fine, AspxStereoMode::Level, 5, 200);
    assert_eq!(df, 70);
}

/// Repeating a value-emitting write produces byte-identical output —
/// the helpers are pure functions of `(codebook, value)` and don't
/// carry any hidden state.
#[test]
fn sig_f0_value_is_deterministic() {
    let mut bw0 = BitWriter::new();
    let mut bw1 = BitWriter::new();
    write_aspx_sig_f0_value(&mut bw0, AspxQuantStep::Fine, AspxStereoMode::Level, 42);
    write_aspx_sig_f0_value(&mut bw1, AspxQuantStep::Fine, AspxStereoMode::Level, 42);
    bw0.align_to_byte();
    bw1.align_to_byte();
    assert_eq!(bw0.finish(), bw1.finish());
}

/// SbgCounts smoke test — the helpers don't consume an `AspxSbgCounts`
/// directly (that's the parser's territory), but the decoder side
/// composes the F0 + DF walk via `AspxSbgCounts` in
/// `parse_aspx_ec_data`. Verify that an encoder-built F0 + DF stream
/// with `num_sbg_sig_highres = 2` round-trips one level under the
/// proper ec_data() entry point so future builder work has a fixture
/// to point at.
#[test]
fn sig_f0_df_round_trips_via_ec_data_entry() {
    use oxideav_ac4::aspx::parse_aspx_ec_data;

    let quant = AspxQuantStep::Fine;
    let stereo = AspxStereoMode::Level;

    let mut bw = BitWriter::new();
    write_aspx_sig_f0_value(&mut bw, quant, stereo, 7);
    write_aspx_sig_df_value(&mut bw, quant, stereo, -3);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);

    let envs = parse_aspx_ec_data(
        &mut br,
        AspxDataType::Signal,
        1,       // num_env
        &[true], // freq_res[0] = true → highres
        quant,
        stereo,
        &[false], // FREQ direction
        AspxSbgCounts {
            num_sbg_sig_highres: 2,
            num_sbg_sig_lowres: 0,
            num_sbg_noise: 0,
        },
    )
    .expect("decoder parsed ec_data(SIGNAL, num_env=1, freq_res=highres, num_sbg=2)");
    assert_eq!(envs.len(), 1);
    assert_eq!(envs[0].values, vec![7, -3]);
    assert!(!envs[0].direction_time);

    // Pseudocode 80 accumulates these as qscf[0] = 2·7 = 14 (Fine
    // delta = 2), qscf[1] = 14 + 2·(-3) = 8. This is exactly the
    // "encoder emits a real-ish envelope" pattern the README's "real
    // ASPX envelope coding" deferral targets.
    let _ = AspxHcbType::F0; // ensure the public enum import is exercised
}
