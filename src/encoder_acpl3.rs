//! 5_X ASPX_ACPL_3 multichannel encoder per ETSI TS 103 190-1 §4.2.6.6
//! Table 25 row `case ASPX_ACPL_3:` (round 95).
//!
//! Symmetric counterpart to the decoder's r34 [`crate::mch::parse_5x_audio_data_outer`]
//! ASPX_ACPL_3 walker. Emits a structurally-valid `5_X_channel_element()`
//! whose `5_X_codec_mode == 4` (ASPX_ACPL_3) body lays out:
//!
//! ```text
//!   5_X_codec_mode = 4               // 3 b
//!   if (b_iframe) {
//!       aspx_config();                  // 15 b — §4.2.12.1 Table 50
//!       acpl_config_2ch();              //  4 b — §4.2.13.2 Table 60
//!   }
//!   if (b_has_lfe) mono_data(1);        // LFE — Table 21
//!   companding_control(2);             // §4.2.11 Table 49 — sync=1, off, no avg
//!   stereo_data();                      // §4.2.6.3 Table 22 — split MDCT
//!   if (b_iframe) {
//!       aspx_data_2ch();               // §4.2.12.4 Table 52 — minimum-valid
//!       acpl_data_2ch();               // §4.2.13.4 Table 62 — minimum-valid
//!   }
//! ```
//!
//! The encoder targets a "structural" round-trip: it forward-MDCTs the
//! caller's L+R PCM into the stereo carrier spectra (using the same KBD
//! window the decoder reverses), and emits minimum-bit-cost ASPX / A-CPL
//! Huffman codewords (zero-delta DF/DT plus a near-zero F0 index for
//! each codebook). The decoder walks the full Table 25 ASPX_ACPL_3 body
//! and produces 5-channel `[L, R, C, Ls, Rs]` PCM via
//! [`crate::acpl_synth::run_acpl_5x_mch_pcm`] (Pseudocode 118). With
//! all-zero ACPL parameter deltas the surround pair Ls/Rs collapses to
//! ducker-driven reconstruction from the L/R carriers — non-silent in
//! the general case, exactly silent when all parameters are at their
//! zero-codebook indices and the carriers are silent.
//!
//! Future rounds will replace the zero-delta ACPL parameter writer with
//! a real QMF-domain parameter extractor that estimates `(alpha, beta,
//! gamma)` per parameter band from the L/R/Ls/Rs source PCM. The ASPX
//! envelope coder side will likewise grow from "structural zero-delta"
//! to "real envelope extraction" in subsequent rounds.

use oxideav_core::bits::BitWriter;

use crate::acpl_huffman;
use crate::asf_data::AsfSections;
use crate::aspx;
use crate::aspx_huffman;
use crate::encoder_asf::{
    build_band_codebook_cost_table, build_sections_from_dp, compute_snf_dpcm_for_zero_quant_bands,
    dp_optimise_sections, finish_substream_body, pick_best_codebook_for_band, write_scalefac_data,
    write_section_data, write_snf_data, write_spectral_data_sections,
};

// ====================================================================
// Minimum-cost Huffman codeword pickers
// ====================================================================

/// Pick the entry from a Huffman LEN/CW pair with the smallest LEN and
/// return `(cw, len)`. Used by the ASPX_ACPL_3 encoder to write
/// minimum-bit-cost codewords at every entropy-coded position.
///
/// Ties are broken by the lowest table index — the spec doesn't promise
/// uniqueness here, but the test invariants in [`acpl_huffman`] /
/// [`aspx_huffman`] do ensure each codebook has a single minimum-length
/// entry.
fn pick_min_len_cw(len: &[u8], cw: &[u32]) -> (u32, u32) {
    debug_assert_eq!(len.len(), cw.len());
    let (idx, min_len) = len
        .iter()
        .enumerate()
        .min_by_key(|(_, &l)| l)
        .map(|(i, &l)| (i, l))
        .expect("hcb table must be non-empty");
    (cw[idx], min_len as u32)
}

/// Pick the entry from a Huffman LEN/CW pair with `index == cb_off`
/// (i.e. the zero-delta entry for DF/DT codebooks). Returns `(cw, len)`.
///
/// For DF / DT codebooks the per-band recovered value is
/// `symbol_index - cb_off`, so this picks the codeword that decodes to
/// delta = 0. The chosen entry typically also has the **shortest** code
/// in the table (the Huffman tree is built around the zero-delta peak).
fn pick_zero_delta_cw(len: &[u8], cw: &[u32], cb_off: usize) -> (u32, u32) {
    debug_assert_eq!(len.len(), cw.len());
    debug_assert!(cb_off < len.len());
    (cw[cb_off], len[cb_off] as u32)
}

// ====================================================================
// ASPX HCB minimum-cost codeword helpers
// ====================================================================

/// Write the ASPX SIGNAL F0 codeword that picks an arbitrary (low-bit)
/// envelope value. Per Pseudocode 79 (`get_aspx_hcb`) the SIGNAL F0
/// codebook is selected by `(quant_mode, stereo_mode)` — we use the four
/// (Fine|Coarse, Level|Balance) combinations explicitly.
fn write_aspx_sig_f0(bw: &mut BitWriter, quant: aspx::AspxQuantStep, stereo: aspx::AspxStereoMode) {
    let (cw, len) = match (quant, stereo) {
        (aspx::AspxQuantStep::Fine, aspx::AspxStereoMode::Level) => pick_min_len_cw(
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_CW,
        ),
        (aspx::AspxQuantStep::Fine, aspx::AspxStereoMode::Balance) => pick_min_len_cw(
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_F0_CW,
        ),
        (aspx::AspxQuantStep::Coarse, aspx::AspxStereoMode::Level) => pick_min_len_cw(
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_F0_CW,
        ),
        (aspx::AspxQuantStep::Coarse, aspx::AspxStereoMode::Balance) => pick_min_len_cw(
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_F0_CW,
        ),
    };
    bw.write_u32(cw, len);
}

/// Write the ASPX SIGNAL DF zero-delta codeword (decoded value
/// `symbol_index - cb_off == 0`).
fn write_aspx_sig_df_zero(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
) {
    let (cw, len) = match (quant, stereo) {
        (aspx::AspxQuantStep::Fine, aspx::AspxStereoMode::Level) => pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_CW,
            70,
        ),
        (aspx::AspxQuantStep::Fine, aspx::AspxStereoMode::Balance) => pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DF_CW,
            24,
        ),
        (aspx::AspxQuantStep::Coarse, aspx::AspxStereoMode::Level) => pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DF_CW,
            35,
        ),
        (aspx::AspxQuantStep::Coarse, aspx::AspxStereoMode::Balance) => pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DF_CW,
            12,
        ),
    };
    bw.write_u32(cw, len);
}

/// Write the ASPX NOISE F0 codeword (minimum-bit-cost).
fn write_aspx_noise_f0(bw: &mut BitWriter, stereo: aspx::AspxStereoMode) {
    let (cw, len) = match stereo {
        aspx::AspxStereoMode::Level => pick_min_len_cw(
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_F0_LEN,
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_F0_CW,
        ),
        aspx::AspxStereoMode::Balance => pick_min_len_cw(
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_F0_LEN,
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_F0_CW,
        ),
    };
    bw.write_u32(cw, len);
}

/// Write the ASPX NOISE DF zero-delta codeword.
fn write_aspx_noise_df_zero(bw: &mut BitWriter, stereo: aspx::AspxStereoMode) {
    let (cw, len) = match stereo {
        aspx::AspxStereoMode::Level => pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_DF_LEN,
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_DF_CW,
            29,
        ),
        aspx::AspxStereoMode::Balance => pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_DF_LEN,
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_DF_CW,
            12,
        ),
    };
    bw.write_u32(cw, len);
}

// ====================================================================
// ASPX HCB value-emitting helpers (round 219)
// ====================================================================
//
// The `write_aspx_*_f0` / `write_aspx_*_df_zero` writers above pick the
// **minimum-bit-cost** codeword (the shortest entry in the Huffman
// table, breaking ties on the lowest table index). They are the right
// choice for the round-95 "zero-delta scaffold" — the goal there is to
// emit a structurally-valid `aspx_data_*()` body with the smallest
// possible bit footprint so the surrounding `ac4_substream()` walker
// stays well-formed and the trailing ACPL parameter pair lands where the
// decoder expects it.
//
// They are **not** the right choice for real envelope coding. The
// minimum-bit F0 codeword for `ASPX_HCB_ENV_LEVEL_15_F0` lands on symbol
// index 30 (LEN = 4 bits, the first occurrence of the table-wide
// minimum), which the decoder dequantises through Pseudocode 80
// (`acc += delta * d`) plus Pseudocode 82 (`scf = n_subbands *
// 2^(qscf/a)`) into an envelope ~64·2^15 — an unintentionally loud
// HF replica. Real envelope coding needs the encoder to (a) compute
// per-`(sbg, atsg)` quantised envelope indices from input band energies,
// (b) FREQ-direction-delta-decode them across `sbg` against an
// accumulator that starts at 0 (Pseudocode 80), so the on-wire stream is
// F0 followed by `(num_sbg - 1)` × DF DPCM deltas, and (c) write each
// codebook value through the matching Huffman table.
//
// These value-emitting helpers are the encoder-side primitives for that
// real envelope coding path: each takes an integer index `v` (F0) or
// `delta_q` (DF) and writes the matching `(cw, len)` from the codebook
// selected by `(quant_mode, stereo_mode)` (SIGNAL) or `stereo_mode`
// alone (NOISE). The helpers clamp their inputs to the codebook's
// addressable range (`0..codebook_length` for F0, `-cb_off..=+cb_off`
// for DF — both inclusive on both ends since the F0 max symbol index is
// `codebook_length - 1` and the DF table is symmetric about `cb_off`).
//
// Round 219 ships the helpers + unit tests; the existing minimum-bit
// writers in `write_aspx_data_2ch_minimal` / `write_aspx_data_1ch_minimal`
// stay in place. A subsequent round will route them through a new
// `write_aspx_data_2ch_real_envelope()` builder that consumes per-sbg
// envelope indices computed from the input MDCT spectra (the inverse
// of Pseudocode 82's `n_subbands * 2^(q/a)` form).

/// Codebook addressing parameters: `(len, cw, cb_off)` triple. For
/// `*_F0` tables `cb_off` is 0 (symbol index == decoded value); for
/// `*_DF` / `*_DT` tables it's the per-table cb_off the decoder uses
/// to recover `delta = symbol_index - cb_off`.
type AspxHcbArrays = (&'static [u8], &'static [u32], i32);

/// Return the `(LEN, CW, cb_off)` triple for the ASPX SIGNAL Huffman
/// codebook keyed by `(quant_mode, stereo_mode, hcb_type)`. The cb_off
/// values mirror the Annex A.2 Tables A.16..=A.27 headers (and match
/// the decoder constants in `aspx::ASPX_HCB_ENV_*`).
fn aspx_sig_hcb_arrays(
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    hcb: aspx::AspxHcbType,
) -> AspxHcbArrays {
    use aspx::AspxHcbType::*;
    use aspx::AspxQuantStep::*;
    use aspx::AspxStereoMode::*;
    match (quant, stereo, hcb) {
        // 15 / Fine (1.5 dB step)
        (Fine, Level, F0) => (
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_CW,
            0,
        ),
        (Fine, Level, Df) => (
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_CW,
            70,
        ),
        (Fine, Level, Dt) => (
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DT_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DT_CW,
            70,
        ),
        (Fine, Balance, F0) => (
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_F0_CW,
            0,
        ),
        (Fine, Balance, Df) => (
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DF_CW,
            24,
        ),
        (Fine, Balance, Dt) => (
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DT_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DT_CW,
            24,
        ),
        // 30 / Coarse (3 dB step)
        (Coarse, Level, F0) => (
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_F0_CW,
            0,
        ),
        (Coarse, Level, Df) => (
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DF_CW,
            35,
        ),
        (Coarse, Level, Dt) => (
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DT_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DT_CW,
            35,
        ),
        (Coarse, Balance, F0) => (
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_F0_CW,
            0,
        ),
        (Coarse, Balance, Df) => (
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DF_CW,
            12,
        ),
        (Coarse, Balance, Dt) => (
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DT_LEN,
            aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DT_CW,
            12,
        ),
    }
}

/// Return the `(LEN, CW, cb_off)` triple for the ASPX NOISE Huffman
/// codebook keyed by `(stereo_mode, hcb_type)`. NOISE codebooks don't
/// depend on `quant_mode` (per ETSI TS 103 190-1 §A.2 NOTE the NOISE
/// envelope is always coded at the Fine step).
fn aspx_noise_hcb_arrays(stereo: aspx::AspxStereoMode, hcb: aspx::AspxHcbType) -> AspxHcbArrays {
    use aspx::AspxHcbType::*;
    use aspx::AspxStereoMode::*;
    match (stereo, hcb) {
        (Level, F0) => (
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_F0_LEN,
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_F0_CW,
            0,
        ),
        (Level, Df) => (
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_DF_LEN,
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_DF_CW,
            29,
        ),
        (Level, Dt) => (
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_DT_LEN,
            aspx_huffman::ASPX_HCB_NOISE_LEVEL_DT_CW,
            29,
        ),
        (Balance, F0) => (
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_F0_LEN,
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_F0_CW,
            0,
        ),
        (Balance, Df) => (
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_DF_LEN,
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_DF_CW,
            12,
        ),
        (Balance, Dt) => (
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_DT_LEN,
            aspx_huffman::ASPX_HCB_NOISE_BALANCE_DT_CW,
            12,
        ),
    }
}

/// Write an ASPX SIGNAL F0 codeword that decodes to the unsigned
/// integer index `v` per ETSI TS 103 190-1 §A.2 Tables A.16 / A.19 /
/// A.22 / A.25. F0 codebooks have `cb_off = 0`, so the decoder
/// recovers `symbol_index == v` directly via
/// [`aspx::AspxHcb::decode_delta`].
///
/// `v` is clamped to the addressable range `0..codebook_length` (i.e.
/// `0..71` for Fine/Level, `0..25` for Fine/Balance, `0..36` for
/// Coarse/Level, `0..13` for Coarse/Balance). Values outside the range
/// silently saturate to the codebook's last entry. This matches the
/// decoder's clamp semantics in `parse_aspx_huff_data()` and keeps the
/// encoder safe against an over-aggressive envelope-energy quantiser.
pub fn write_aspx_sig_f0_value(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    v: i32,
) {
    let (len, cw, _cb_off) = aspx_sig_hcb_arrays(quant, stereo, aspx::AspxHcbType::F0);
    let idx = v.clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write an ASPX SIGNAL DF codeword that decodes to the signed delta
/// `delta_q` per ETSI TS 103 190-1 §A.2 Tables A.17 / A.20 / A.23 /
/// A.26. The decoder recovers `delta = symbol_index - cb_off`, so the
/// encoder writes `cw[delta_q + cb_off]` with the matching `len[]`.
///
/// `delta_q` is clamped to the codebook's symmetric range
/// `-cb_off..=cb_off` (which exactly matches `0..codebook_length`
/// after the offset is applied), so values outside the range saturate
/// to the extreme entries. The DF codebooks all happen to be symmetric
/// `2·cb_off + 1` wide.
pub fn write_aspx_sig_df_value(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    delta_q: i32,
) {
    let (len, cw, cb_off) = aspx_sig_hcb_arrays(quant, stereo, aspx::AspxHcbType::Df);
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write an ASPX SIGNAL DT codeword that decodes to the signed
/// time-delta `delta_q` per ETSI TS 103 190-1 §A.2 Tables A.18 / A.21 /
/// A.24 / A.27. Same shape as [`write_aspx_sig_df_value`] but routed
/// through the `Dt` codebook variants — used when the caller has
/// signalled `aspx_sig_delta_dir[env] == TIME` (Pseudocode 80's TIME
/// branch).
pub fn write_aspx_sig_dt_value(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    delta_q: i32,
) {
    let (len, cw, cb_off) = aspx_sig_hcb_arrays(quant, stereo, aspx::AspxHcbType::Dt);
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write an ASPX NOISE F0 codeword that decodes to the unsigned
/// integer index `v` per ETSI TS 103 190-1 §A.2 Tables A.28 / A.31.
/// NOISE codebooks share the F0 / Df / Dt shape of the SIGNAL paths;
/// the only differences are the codebook contents and the absence of
/// a per-envelope `quant_mode` selector (NOISE is always 1.5 dB step).
pub fn write_aspx_noise_f0_value(bw: &mut BitWriter, stereo: aspx::AspxStereoMode, v: i32) {
    let (len, cw, _cb_off) = aspx_noise_hcb_arrays(stereo, aspx::AspxHcbType::F0);
    let idx = v.clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write an ASPX NOISE DF codeword that decodes to the signed delta
/// `delta_q` per ETSI TS 103 190-1 §A.2 Tables A.29 / A.32.
pub fn write_aspx_noise_df_value(bw: &mut BitWriter, stereo: aspx::AspxStereoMode, delta_q: i32) {
    let (len, cw, cb_off) = aspx_noise_hcb_arrays(stereo, aspx::AspxHcbType::Df);
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write an ASPX NOISE DT codeword that decodes to the signed
/// time-delta `delta_q` per ETSI TS 103 190-1 §A.2 Tables A.30 / A.33.
pub fn write_aspx_noise_dt_value(bw: &mut BitWriter, stereo: aspx::AspxStereoMode, delta_q: i32) {
    let (len, cw, cb_off) = aspx_noise_hcb_arrays(stereo, aspx::AspxHcbType::Dt);
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

// ====================================================================
// ACPL HCB minimum-cost codeword helpers
// ====================================================================

/// Map a (`data_type`, `quant_mode`, `hcb_type`) tuple to the
/// matching ACPL Huffman codebook LEN/CW arrays and `cb_off`.
///
/// Mirrors [`crate::acpl::get_acpl_hcb`] but returns the raw table
/// references rather than an `AcplHcb` handle. Used by the encoder to
/// pick the minimum-cost codeword for each parameter band.
fn acpl_hcb_arrays(
    dt: crate::acpl::AcplDataType,
    qm: crate::acpl::AcplQuantMode,
    ht: crate::acpl::AcplHcbType,
) -> (&'static [u8], &'static [u32], i32) {
    use crate::acpl::AcplDataType::*;
    use crate::acpl::AcplHcbType::*;
    use crate::acpl::AcplQuantMode::*;
    use acpl_huffman::*;
    match (dt, qm, ht) {
        // ALPHA — F0 codebooks are symmetric (Coarse 17 entries / Fine 33
        // entries) so the signed `alpha_q ∈ [-N/2, +N/2]` lives at
        // `symbol_index = alpha_q + cb_off` with `cb_off = N/2`. Must
        // match [`crate::acpl::get_acpl_hcb`] for round-trip parity.
        (Alpha, Coarse, F0) => (ACPL_HCB_ALPHA_COARSE_F0_LEN, ACPL_HCB_ALPHA_COARSE_F0_CW, 8),
        (Alpha, Fine, F0) => (ACPL_HCB_ALPHA_FINE_F0_LEN, ACPL_HCB_ALPHA_FINE_F0_CW, 16),
        (Alpha, Coarse, Df) => (
            ACPL_HCB_ALPHA_COARSE_DF_LEN,
            ACPL_HCB_ALPHA_COARSE_DF_CW,
            16,
        ),
        (Alpha, Fine, Df) => (ACPL_HCB_ALPHA_FINE_DF_LEN, ACPL_HCB_ALPHA_FINE_DF_CW, 32),
        (Alpha, Coarse, Dt) => (
            ACPL_HCB_ALPHA_COARSE_DT_LEN,
            ACPL_HCB_ALPHA_COARSE_DT_CW,
            16,
        ),
        (Alpha, Fine, Dt) => (ACPL_HCB_ALPHA_FINE_DT_LEN, ACPL_HCB_ALPHA_FINE_DT_CW, 32),
        // BETA
        (Beta, Coarse, F0) => (ACPL_HCB_BETA_COARSE_F0_LEN, ACPL_HCB_BETA_COARSE_F0_CW, 0),
        (Beta, Fine, F0) => (ACPL_HCB_BETA_FINE_F0_LEN, ACPL_HCB_BETA_FINE_F0_CW, 0),
        (Beta, Coarse, Df) => (ACPL_HCB_BETA_COARSE_DF_LEN, ACPL_HCB_BETA_COARSE_DF_CW, 4),
        (Beta, Fine, Df) => (ACPL_HCB_BETA_FINE_DF_LEN, ACPL_HCB_BETA_FINE_DF_CW, 8),
        (Beta, Coarse, Dt) => (ACPL_HCB_BETA_COARSE_DT_LEN, ACPL_HCB_BETA_COARSE_DT_CW, 4),
        (Beta, Fine, Dt) => (ACPL_HCB_BETA_FINE_DT_LEN, ACPL_HCB_BETA_FINE_DT_CW, 8),
        // BETA3 — F0 codebooks are symmetric (Coarse 9 / Fine 17) so the
        // signed `beta3_q ∈ [-N/2, +N/2]` lives at `symbol_index =
        // beta3_q + cb_off` with `cb_off = N/2`. Must match
        // [`crate::acpl::get_acpl_hcb`] for round-trip parity.
        (Beta3, Coarse, F0) => (ACPL_HCB_BETA3_COARSE_F0_LEN, ACPL_HCB_BETA3_COARSE_F0_CW, 4),
        (Beta3, Fine, F0) => (ACPL_HCB_BETA3_FINE_F0_LEN, ACPL_HCB_BETA3_FINE_F0_CW, 8),
        (Beta3, Coarse, Df) => (ACPL_HCB_BETA3_COARSE_DF_LEN, ACPL_HCB_BETA3_COARSE_DF_CW, 8),
        (Beta3, Fine, Df) => (ACPL_HCB_BETA3_FINE_DF_LEN, ACPL_HCB_BETA3_FINE_DF_CW, 16),
        (Beta3, Coarse, Dt) => (ACPL_HCB_BETA3_COARSE_DT_LEN, ACPL_HCB_BETA3_COARSE_DT_CW, 8),
        (Beta3, Fine, Dt) => (ACPL_HCB_BETA3_FINE_DT_LEN, ACPL_HCB_BETA3_FINE_DT_CW, 16),
        // GAMMA
        (Gamma, Coarse, F0) => (
            ACPL_HCB_GAMMA_COARSE_F0_LEN,
            ACPL_HCB_GAMMA_COARSE_F0_CW,
            10,
        ),
        (Gamma, Fine, F0) => (ACPL_HCB_GAMMA_FINE_F0_LEN, ACPL_HCB_GAMMA_FINE_F0_CW, 20),
        (Gamma, Coarse, Df) => (
            ACPL_HCB_GAMMA_COARSE_DF_LEN,
            ACPL_HCB_GAMMA_COARSE_DF_CW,
            20,
        ),
        (Gamma, Fine, Df) => (ACPL_HCB_GAMMA_FINE_DF_LEN, ACPL_HCB_GAMMA_FINE_DF_CW, 40),
        (Gamma, Coarse, Dt) => (
            ACPL_HCB_GAMMA_COARSE_DT_LEN,
            ACPL_HCB_GAMMA_COARSE_DT_CW,
            20,
        ),
        (Gamma, Fine, Dt) => (ACPL_HCB_GAMMA_FINE_DT_LEN, ACPL_HCB_GAMMA_FINE_DT_CW, 40),
    }
}

/// Write the ACPL F0 codeword that picks the recovered value `0`
/// (`symbol_index == cb_off` decodes to `symbol_index - cb_off == 0`).
fn write_acpl_f0_zero(
    bw: &mut BitWriter,
    dt: crate::acpl::AcplDataType,
    qm: crate::acpl::AcplQuantMode,
) {
    let (len, cw, cb_off) = acpl_hcb_arrays(dt, qm, crate::acpl::AcplHcbType::F0);
    let idx = cb_off as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL DF codeword for `symbol_index == cb_off` (zero delta).
fn write_acpl_df_zero(
    bw: &mut BitWriter,
    dt: crate::acpl::AcplDataType,
    qm: crate::acpl::AcplQuantMode,
) {
    let (len, cw, cb_off) = acpl_hcb_arrays(dt, qm, crate::acpl::AcplHcbType::Df);
    let idx = cb_off as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

// ====================================================================
// aspx_config emitter — §4.2.12.1 Table 50 (15 bits)
// ====================================================================

/// Emit an `aspx_config()` element (15 bits) per ETSI TS 103 190-1
/// Table 50 with caller-chosen settings. The wire-bit-order matches the
/// parser's `parse_aspx_config`.
pub fn write_aspx_config(bw: &mut BitWriter, cfg: &aspx::AspxConfig) {
    let qmode_bit = match cfg.quant_mode_env {
        aspx::AspxQuantStep::Fine => 0,
        aspx::AspxQuantStep::Coarse => 1,
    };
    let scale_bit = match cfg.master_freq_scale {
        aspx::AspxMasterFreqScale::LowRes => 0,
        aspx::AspxMasterFreqScale::HighRes => 1,
    };
    let freq_res_bits = match cfg.freq_res_mode {
        aspx::AspxFreqResMode::Signalled => 0u32,
        aspx::AspxFreqResMode::Low => 1,
        aspx::AspxFreqResMode::DurationDependent => 2,
        aspx::AspxFreqResMode::High => 3,
    };
    bw.write_u32(qmode_bit, 1);
    bw.write_u32(cfg.start_freq as u32, 3);
    bw.write_u32(cfg.stop_freq as u32, 2);
    bw.write_u32(scale_bit, 1);
    bw.write_bit(cfg.interpolation);
    bw.write_bit(cfg.preflat);
    bw.write_bit(cfg.limiter);
    bw.write_u32(cfg.noise_sbg as u32, 2);
    bw.write_u32(cfg.num_env_bits_fixfix as u32, 1);
    bw.write_u32(freq_res_bits, 2);
}

// ====================================================================
// acpl_config_2ch emitter — §4.2.13.2 Table 60 (4 bits)
// ====================================================================

/// Emit an `acpl_config_2ch()` element (4 bits) per §4.2.13.2 Table 60:
/// 2-bit `num_param_bands_id` + 1-bit `quant_mode_0` + 1-bit
/// `quant_mode_1`. The decoder's
/// [`crate::acpl::parse_acpl_config_2ch`] reads exactly the same
/// ordering.
pub fn write_acpl_config_2ch(
    bw: &mut BitWriter,
    num_param_bands_id: u8,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
) {
    bw.write_u32(num_param_bands_id as u32 & 0b11, 2);
    let qm0_bit = matches!(quant_mode_0, crate::acpl::AcplQuantMode::Coarse);
    let qm1_bit = matches!(quant_mode_1, crate::acpl::AcplQuantMode::Coarse);
    bw.write_bit(qm0_bit);
    bw.write_bit(qm1_bit);
}

// ====================================================================
// companding_control(2) emitter — §4.2.11 Table 49
// ====================================================================

/// Emit a `companding_control(2)` element with sync_flag = 1 and
/// `b_compand_on = 1` (companding ON, no `compand_avg`). Total: 2 bits.
///
/// Per §4.2.11 Table 49 the field order is:
/// `sync_flag` (1 b, only when `num_chan > 1`) +
/// `b_compand_on[0..nc]` (nc = 1 when sync_flag = 1, else num_chan) +
/// `b_compand_avg` (1 b, only when at least one channel is OFF).
pub fn write_companding_control_2ch_sync_on(bw: &mut BitWriter) {
    bw.write_bit(true); // sync_flag = 1 → single b_compand_on follows
    bw.write_bit(true); // b_compand_on[0] = 1 → no avg follow-on
}

// ====================================================================
// stereo_data() split-MDCT emitter — §4.2.6.3 Table 22
// ====================================================================

/// Per-channel forward-analysis result used by [`build_stereo_split_data`].
type StereoChannelAnalysis = (Vec<i32>, Vec<i32>, Vec<u32>, AsfSections, Option<Vec<i32>>);

fn prepare_stereo_channel(coeffs: &[f32], sfbo: &[u16], max_sfb: u32) -> StereoChannelAnalysis {
    let local_end = sfbo[max_sfb as usize] as usize;
    let mut qspec = vec![0i32; local_end];
    let mut sf_per_band = vec![100i32; max_sfb as usize];
    let mut max_quant_idx = vec![0u32; max_sfb as usize];
    let mut natural_q_per_band: Vec<Vec<i32>> = Vec::with_capacity(max_sfb as usize);
    for sfb in 0..max_sfb as usize {
        let a = sfbo[sfb] as usize;
        let b = sfbo[sfb + 1] as usize;
        let band = &coeffs[a..b.min(coeffs.len())];
        let (_cb_picked, sf, q, _cost) = pick_best_codebook_for_band(band);
        sf_per_band[sfb] = sf;
        let mut max_q: u32 = 0;
        for (i, &qi) in q.iter().enumerate() {
            qspec[a + i] = qi;
            max_q = max_q.max(qi.unsigned_abs());
        }
        max_quant_idx[sfb] = max_q;
        natural_q_per_band.push(q);
    }
    let cost_table = build_band_codebook_cost_table(&natural_q_per_band);
    let dp_sections = dp_optimise_sections(&cost_table, 16);
    let sections = build_sections_from_dp(&dp_sections, max_sfb);
    let snf = compute_snf_dpcm_for_zero_quant_bands(
        coeffs,
        sfbo,
        max_sfb,
        &sections.sfb_cb,
        &max_quant_idx,
    );
    (qspec, sf_per_band, max_quant_idx, sections, snf)
}

/// Emit a `stereo_data()` body with `b_enable_mdct_stereo_proc = 0`
/// (split MDCT path) per §4.2.6.3 Table 22:
///
/// ```text
///   b_enable_mdct_stereo_proc = 0       // 1 b
///   spec_frontend_l = 0 (ASF)           // 1 b
///   sf_info(ASF, 0, 0)                  // asf_transform_info + asf_psy_info
///   spec_frontend_r = 0 (ASF)           // 1 b
///   sf_info(ASF, 0, 0)                  // asf_transform_info + asf_psy_info
///   sf_data(spec_frontend_l)            // L spectrum
///   sf_data(spec_frontend_r)            // R spectrum
/// ```
///
/// Reuses the round-50 forward-MDCT + DP-section + HCB1..11 + SNF
/// pipeline per channel.
fn write_stereo_split_data(
    bw: &mut BitWriter,
    transform_length: u32,
    max_sfb: u32,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
) {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let (n_msfb_bits, _, _) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    let analysis_l = prepare_stereo_channel(coeffs_l, sfbo, max_sfb);
    let analysis_r = prepare_stereo_channel(coeffs_r, sfbo, max_sfb);

    // b_enable_mdct_stereo_proc = 0 → split-MDCT path.
    bw.write_bit(false);
    // L channel: spec_frontend_l = 0 (ASF) + asf_transform_info + asf_psy_info.
    bw.write_bit(false);
    bw.write_bit(true); // asf_transform_info: b_long_frame = 1
    bw.write_u32(max_sfb, n_msfb_bits); // asf_psy_info: max_sfb[0] in n_msfb_bits
                                        // R channel: spec_frontend_r = 0 (ASF) + asf_transform_info + asf_psy_info.
    bw.write_bit(false);
    bw.write_bit(true);
    bw.write_u32(max_sfb, n_msfb_bits);

    // L sf_data(ASF).
    let (qspec_l, sf_l, max_q_l, sections_l, snf_l) = &analysis_l;
    write_section_data(bw, sections_l);
    write_spectral_data_sections(bw, qspec_l, sfbo, sections_l);
    write_scalefac_data(bw, sf_l, &sections_l.sfb_cb, max_q_l, max_sfb);
    write_snf_data(bw, snf_l.as_deref(), &sections_l.sfb_cb, max_q_l, max_sfb);

    // R sf_data(ASF).
    let (qspec_r, sf_r, max_q_r, sections_r, snf_r) = &analysis_r;
    write_section_data(bw, sections_r);
    write_spectral_data_sections(bw, qspec_r, sfbo, sections_r);
    write_scalefac_data(bw, sf_r, &sections_r.sfb_cb, max_q_r, max_sfb);
    write_snf_data(bw, snf_r.as_deref(), &sections_r.sfb_cb, max_q_r, max_sfb);
}

// ====================================================================
// aspx_data_2ch() emitter — §4.2.12.4 Table 52 (FIXFIX num_env=1 path)
// ====================================================================

/// Emit a minimum-viable `aspx_data_2ch()` body per Table 52 with:
/// * `xover_subband_offset = 0` (3 b)
/// * Channel-0 `aspx_framing()`: `int_class = FIXFIX` (prefix `0`, 1 b
///   per Table 126), `tmp_num_env = 0` (1 or 2 b per `num_env_bits_fixfix`),
///   `aspx_freq_res[0] = 0` (1 b, only when `freq_res_mode == Signalled`).
/// * `aspx_balance = 1` (1 b) — channel-1 reuses channel-0's framing.
/// * `aspx_delta_dir(0)`: 1 SIGNAL-env-direction bit + 1 NOISE bit (FREQ).
/// * `aspx_delta_dir(1)`: same shape.
/// * `aspx_hfgen_iwc_2ch(balance=1)`: `num_sbg_noise` × 2 b tna_mode +
///   `ah_left/right/fic_present/tic_present = 0` (4 × 1 b).
/// * 4× `aspx_ec_data()`: ch0/ch1 SIGNAL + ch0/ch1 NOISE, each
///   `num_env=1` envelope's worth of Huffman codewords (F0 + `(num_sbg-1)` × DF).
///
/// The frequency-table derivation runs the existing
/// [`aspx::derive_aspx_frequency_tables`] internally so the emitted bit
/// counts line up with whatever the decoder rederives.
fn write_aspx_data_2ch_minimal(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
) -> Result<(), &'static str> {
    write_aspx_data_2ch_minimal_framed(bw, cfg, true)
}

/// [`write_aspx_data_2ch_minimal`] with an explicit `b_iframe` — Tables 51/52 gate the
/// leading 3-bit xover offset on I-frames.
pub(crate) fn write_aspx_data_2ch_minimal_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
) -> Result<(), &'static str> {
    write_aspx_data_2ch_minimal_framed_xover(bw, cfg, b_iframe, 0)
}

/// [`write_aspx_data_2ch_minimal_framed`] with an explicit
/// `aspx_xover_subband_offset` (3 bits on I-frames; the frequency
/// tables — and so the envelope band counts — follow it).
pub(crate) fn write_aspx_data_2ch_minimal_framed_xover(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
    xover: u32,
) -> Result<(), &'static str> {
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    // Channel-0 aspx_framing: FIXFIX, num_env = 1 (tmp_num_env = 0).
    // int_class bits per AspxIntClass::read: prefix '0' for FIXFIX
    // (Table 126), 1 b.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits); // tmp_num_env = 0 → num_env = 1
    if cfg.signals_freq_res() {
        bw.write_bit(false); // aspx_freq_res[0] = 0 (low-res)
    }
    // aspx_balance = 1 → channel-1 reuses channel-0's framing.
    bw.write_bit(true);
    // aspx_delta_dir(0): num_env=1 SIGNAL direction bit + num_noise=1 NOISE bit.
    bw.write_bit(false); // sig_delta_dir[0] = false (FREQ)
    bw.write_bit(false); // noise_delta_dir[0] = false (FREQ)
                         // aspx_delta_dir(1): same shape.
    bw.write_bit(false);
    bw.write_bit(false);

    // Derive frequency tables so we know the per-channel SBG counts.
    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_2ch(balance=1):
    //   tna_mode[0][..num_sbg_noise]: 2 b each → 0 (TNA off).
    for _ in 0..counts.num_sbg_noise {
        bw.write_u32(0, 2);
    }
    // tna_mode[1][..] is implicit when balance = 1 (mirrors channel 0).
    // ah_left = 0, ah_right = 0 (no add-harmonic vectors).
    bw.write_bit(false);
    bw.write_bit(false);
    // fic_present = 0 (no frequency-interleaved-coding).
    bw.write_bit(false);
    // tic_present = 0 (no time-interleaved-coding).
    bw.write_bit(false);

    // SIGNAL ec_data band count — per ETSI TS 103 190-1 §4.3.10.4.9
    // (Table 124 NOTE 3) the SIGNAL ec_data walks `num_sbg_sig_highres`
    // bands when the `aspx_freq_res[env]` bit is absent or set to 1,
    // and `num_sbg_sig_lowres` only when an explicit
    // `aspx_freq_res = 0` was emitted (the parser's
    // `freq_res.get(env).copied().unwrap_or(true)` fallback selects
    // the high-resolution count). The 2ch emitter above writes
    // `aspx_freq_res[0] = 0` only when `cfg.signals_freq_res()` is
    // true — so the SIGNAL ec_data band count follows that gate.
    //
    // Pre-r181 the 2ch emitter hard-coded `num_sbg_sig_lowres`
    // regardless of `signals_freq_res()`, which for the encoder's
    // default `DurationDependent` config caused a walker desync that
    // buried every subsequent ACPL_1 / ACPL_2 `acpl_data_1ch()` α / β
    // codeword in trailing zero-padding and silently produced
    // all-zero recovered indices (the issue the user's "alpha_q
    // desync" follow-up tracked).
    let num_sbg_sig = if cfg.signals_freq_res() {
        // freq_res bit emitted as 0 above → low-res selection on both channels.
        counts.num_sbg_sig_lowres
    } else {
        // No freq_res bit emitted → parser defaults to high-res.
        counts.num_sbg_sig_highres
    };
    let num_sbg_noise = counts.num_sbg_noise;

    // ch0 SIGNAL: FREQ direction → F0 + (num_sbg_sig - 1) × DF.
    // stereo_mode = LEVEL per Table 52.
    let qmode_ch0 = if cfg.fixfix_tmp_num_env_bits() == 1 {
        // Per Table 52: FIXFIX + num_env == 1 → qmode forced to Fine.
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };
    if num_sbg_sig >= 1 {
        write_aspx_sig_f0(bw, qmode_ch0, aspx::AspxStereoMode::Level);
    }
    for _ in 1..num_sbg_sig {
        write_aspx_sig_df_zero(bw, qmode_ch0, aspx::AspxStereoMode::Level);
    }
    // ch1 SIGNAL: stereo_mode = BALANCE (aspx_balance = 1). The
    // scaffold pans neutrally: per §5.7.6.3.5 Pseudocode 84 the
    // neutral balance is qscf_b = a · PAN_OFFSET, i.e. a wire F0 of
    // 6a under the decoder's delta = 2 accumulation — both channels
    // then dequantize to the Pseudocode 82 value of the sum row.
    let qmode_ch1 = qmode_ch0; // shared framing
    let sig_neutral = match qmode_ch1 {
        aspx::AspxQuantStep::Fine => 12,
        aspx::AspxQuantStep::Coarse => 6,
    };
    if num_sbg_sig >= 1 {
        write_aspx_sig_f0_value(bw, qmode_ch1, aspx::AspxStereoMode::Balance, sig_neutral);
    }
    for _ in 1..num_sbg_sig {
        write_aspx_sig_df_zero(bw, qmode_ch1, aspx::AspxStereoMode::Balance);
    }

    // ch0 NOISE: FREQ direction → F0 + (num_sbg_noise - 1) × DF.
    // Per Table 52 NOISE qmode = 0 (Fine).
    if num_sbg_noise >= 1 {
        write_aspx_noise_f0(bw, aspx::AspxStereoMode::Level);
    }
    for _ in 1..num_sbg_noise {
        write_aspx_noise_df_zero(bw, aspx::AspxStereoMode::Level);
    }
    // ch1 NOISE: stereo_mode = BALANCE, neutral pan (wire F0 = 6 ⇒
    // qscf_b = PAN_OFFSET).
    if num_sbg_noise >= 1 {
        write_aspx_noise_f0_value(bw, aspx::AspxStereoMode::Balance, 6);
    }
    for _ in 1..num_sbg_noise {
        write_aspx_noise_df_zero(bw, aspx::AspxStereoMode::Balance);
    }
    Ok(())
}

// ====================================================================
// acpl_data_2ch() emitter — §4.2.13.4 Table 62 (1 param-set path)
// ====================================================================

/// Emit a minimum-viable `acpl_data_2ch()` body per Table 62 with:
/// * `acpl_framing_data()`: `interpolation_type = Smooth` (1 b),
///   `num_param_sets_cod = 0` (1 b) → `num_param_sets = 1`.
/// * 11 × `acpl_huff_data()` calls: alpha1, alpha2, beta1, beta2,
///   beta3, gamma1..gamma6. Each emits `diff_type = 0` (DIFF_FREQ)
///   then one F0 codeword + `(num_bands - 1)` DF zero-delta codewords.
fn write_acpl_data_2ch_minimal(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);
    // num_param_sets = 1 — single parameter set per frame.

    // helper to emit one acpl_huff_data() FREQ-mode block: diff_type=0 +
    // F0 + (num_bands - 1) × DF.
    let emit_one =
        |bw: &mut BitWriter, dt: crate::acpl::AcplDataType, qm: crate::acpl::AcplQuantMode| {
            bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
            if num_bands >= 1 {
                write_acpl_f0_zero(bw, dt, qm);
            }
            for _ in 1..num_bands {
                write_acpl_df_zero(bw, dt, qm);
            }
        };

    // alpha1, alpha2 — ALPHA codebook family, quant_mode_0.
    emit_one(bw, crate::acpl::AcplDataType::Alpha, quant_mode_0);
    emit_one(bw, crate::acpl::AcplDataType::Alpha, quant_mode_0);
    // beta1, beta2 — BETA codebook family, quant_mode_0.
    emit_one(bw, crate::acpl::AcplDataType::Beta, quant_mode_0);
    emit_one(bw, crate::acpl::AcplDataType::Beta, quant_mode_0);
    // beta3 — BETA3 codebook family, quant_mode_0.
    emit_one(bw, crate::acpl::AcplDataType::Beta3, quant_mode_0);
    // gamma1..6 — GAMMA codebook family, quant_mode_1.
    for _ in 0..6 {
        emit_one(bw, crate::acpl::AcplDataType::Gamma, quant_mode_1);
    }
}

/// Emit an `acpl_data_2ch()` body per §4.2.13.4 Table 62 with the
/// `acpl_beta_1_dq` / `acpl_beta_2_dq` entropy layers carrying real
/// per-parameter-band magnitudes; the remaining nine parameter sets
/// (`alpha_1`, `alpha_2`, `beta_3`, `gamma_1..6`) keep the zero-delta
/// scaffold from [`write_acpl_data_2ch_minimal`].
///
/// Each β layer is coded as `diff_type = 0` (DIFF_FREQ) + one F0
/// codeword + `(num_bands − 1)` DF codewords. Per [`acpl_hcb_arrays`]
/// the BETA F0 codebook is addressed by `symbol_index = beta_q` (cb_off
/// = 0) so the F0 codeword carries the non-negative magnitude directly.
/// The DF codebook uses `symbol_index = delta_q + cb_off` and supports
/// signed band-to-band deltas which the decoder reverses via
/// [`crate::acpl_synth::differential_decode`]'s DIFF_FREQ branch.
///
/// `beta1_q_per_band` / `beta2_q_per_band` must each contain at least
/// `num_bands` entries; trailing positions outside the slice are coded
/// as `0`. The α / β3 / γ slots remain at zero-delta to preserve the
/// round-95 wire-bit layout invariants.
fn write_acpl_data_2ch_real_beta(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
    beta1_q_per_band: &[i32],
    beta2_q_per_band: &[i32],
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);

    // Helper: emit one zero-delta `acpl_huff_data()` FREQ-mode block
    // (matches `write_acpl_data_2ch_minimal`'s inner closure).
    let emit_zero =
        |bw: &mut BitWriter, dt: crate::acpl::AcplDataType, qm: crate::acpl::AcplQuantMode| {
            bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
            if num_bands >= 1 {
                write_acpl_f0_zero(bw, dt, qm);
            }
            for _ in 1..num_bands {
                write_acpl_df_zero(bw, dt, qm);
            }
        };

    // Helper: emit one real-β `acpl_huff_data()` FREQ-mode block. F0
    // carries `beta_q[0]`; DFs carry `delta_q[pb] = beta_q[pb] − beta_q[pb-1]`.
    let emit_real_beta = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let b_q = beta_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta_f0_value(bw, qm, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta_df_value(bw, qm, delta);
            }
            prev_q = b_q;
        }
    };

    // alpha1, alpha2 — zero-delta (ALPHA codebook family, quant_mode_0).
    emit_zero(bw, crate::acpl::AcplDataType::Alpha, quant_mode_0);
    emit_zero(bw, crate::acpl::AcplDataType::Alpha, quant_mode_0);
    // beta1, beta2 — REAL per-band, BETA codebook family, quant_mode_0.
    emit_real_beta(bw, quant_mode_0, beta1_q_per_band);
    emit_real_beta(bw, quant_mode_0, beta2_q_per_band);
    // beta3 — zero-delta (BETA3 codebook family, quant_mode_0).
    emit_zero(bw, crate::acpl::AcplDataType::Beta3, quant_mode_0);
    // gamma1..6 — zero-delta (GAMMA codebook family, quant_mode_1).
    for _ in 0..6 {
        emit_zero(bw, crate::acpl::AcplDataType::Gamma, quant_mode_1);
    }
}

/// Emit an `acpl_data_2ch()` body per §4.2.13.4 Table 62 with the
/// `acpl_alpha_1_dq` / `acpl_alpha_2_dq` entropy layers carrying real
/// per-parameter-band magnitudes *in addition* to the r193 real β1 / β2
/// layers; β3 / γ1..γ6 keep the zero-delta scaffold from
/// [`write_acpl_data_2ch_minimal`].
///
/// Each α layer is coded as `diff_type = 0` (DIFF_FREQ) + one F0
/// codeword + `(num_bands − 1)` DF codewords. Per [`acpl_hcb_arrays`]
/// the ALPHA F0 codebook is symmetric around `cb_off` (8 Coarse / 16
/// Fine) so the F0 codeword carries the signed `alpha_q` directly. The
/// DF codebook addresses `symbol_index = delta_q + cb_off` and supports
/// signed band-to-band deltas which the decoder reverses via
/// [`crate::acpl_synth::differential_decode`]'s DIFF_FREQ branch.
///
/// `alpha1_q_per_band` / `alpha2_q_per_band` / `beta1_q_per_band` /
/// `beta2_q_per_band` must each contain at least `num_bands` entries;
/// trailing positions outside the slice are coded as `0`.
#[allow(clippy::too_many_arguments)]
fn write_acpl_data_2ch_real_alpha_beta(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
    alpha1_q_per_band: &[i32],
    alpha2_q_per_band: &[i32],
    beta1_q_per_band: &[i32],
    beta2_q_per_band: &[i32],
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);

    // Helper: emit one zero-delta `acpl_huff_data()` FREQ-mode block
    // (matches `write_acpl_data_2ch_minimal`'s inner closure).
    let emit_zero =
        |bw: &mut BitWriter, dt: crate::acpl::AcplDataType, qm: crate::acpl::AcplQuantMode| {
            bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
            if num_bands >= 1 {
                write_acpl_f0_zero(bw, dt, qm);
            }
            for _ in 1..num_bands {
                write_acpl_df_zero(bw, dt, qm);
            }
        };

    // Helper: emit one real-α `acpl_huff_data()` FREQ-mode block. F0
    // carries `alpha_q[0]`; DFs carry `delta_q[pb] = alpha_q[pb] − alpha_q[pb-1]`.
    let emit_real_alpha = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, alpha_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let a_q = alpha_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_alpha_f0_value(bw, qm, a_q);
                first = false;
            } else {
                let delta = a_q - prev_q;
                write_acpl_alpha_df_value(bw, qm, delta);
            }
            prev_q = a_q;
        }
    };

    // Helper: emit one real-β `acpl_huff_data()` FREQ-mode block. F0
    // carries `beta_q[0]`; DFs carry `delta_q[pb] = beta_q[pb] − beta_q[pb-1]`.
    let emit_real_beta = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let b_q = beta_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta_f0_value(bw, qm, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta_df_value(bw, qm, delta);
            }
            prev_q = b_q;
        }
    };

    // alpha1, alpha2 — REAL per-band (ALPHA codebook family, quant_mode_0).
    emit_real_alpha(bw, quant_mode_0, alpha1_q_per_band);
    emit_real_alpha(bw, quant_mode_0, alpha2_q_per_band);
    // beta1, beta2 — REAL per-band (BETA codebook family, quant_mode_0).
    emit_real_beta(bw, quant_mode_0, beta1_q_per_band);
    emit_real_beta(bw, quant_mode_0, beta2_q_per_band);
    // beta3 — zero-delta (BETA3 codebook family, quant_mode_0).
    emit_zero(bw, crate::acpl::AcplDataType::Beta3, quant_mode_0);
    // gamma1..6 — zero-delta (GAMMA codebook family, quant_mode_1).
    for _ in 0..6 {
        emit_zero(bw, crate::acpl::AcplDataType::Gamma, quant_mode_1);
    }
}

/// Emit an `acpl_data_2ch()` body per §4.2.13.4 Table 62 with the
/// `acpl_alpha_1_dq` / `acpl_alpha_2_dq` / `acpl_beta_1_dq` /
/// `acpl_beta_2_dq` / `acpl_g_5_dq` / `acpl_g_6_dq` entropy layers all
/// carrying real per-parameter-band magnitudes; β3 / γ1..γ4 still emit
/// the round-95 zero-delta scaffold (those parameter sets only enter the
/// Pseudocode 119 (L, R, Ls, Rs) sub-pipeline plus the ACplModule3
/// cross-residual — neither of which has a per-side surround reference
/// available at encode time for the 5.0 / 5.1 PCM input layouts the
/// real-γ entry point targets).
///
/// γ5 / γ6 drive the centre-channel reconstruction (Pseudocode 118 step
/// 7: `z4 = 0.5 · (γ5 · x0in + γ6 · x1in)` followed by `z4 *= √2` in
/// step 11) — see
/// [`extract_gamma_5_6_q_per_band_centre_least_squares`] for the
/// per-band least-squares extractor that produces the codeword inputs.
///
/// `alpha1_q_per_band` / `alpha2_q_per_band` / `beta1_q_per_band` /
/// `beta2_q_per_band` / `gamma5_q_per_band` / `gamma6_q_per_band` must
/// each contain at least `num_bands` entries; trailing positions outside
/// the slice are coded as `0`. The α / β codebook families use
/// `quant_mode_0`; the γ family uses `quant_mode_1` per Table 62.
#[allow(clippy::too_many_arguments)]
fn write_acpl_data_2ch_real_alpha_beta_gamma(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
    alpha1_q_per_band: &[i32],
    alpha2_q_per_band: &[i32],
    beta1_q_per_band: &[i32],
    beta2_q_per_band: &[i32],
    gamma5_q_per_band: &[i32],
    gamma6_q_per_band: &[i32],
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);

    // Helper: emit one zero-delta `acpl_huff_data()` FREQ-mode block.
    let emit_zero =
        |bw: &mut BitWriter, dt: crate::acpl::AcplDataType, qm: crate::acpl::AcplQuantMode| {
            bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
            if num_bands >= 1 {
                write_acpl_f0_zero(bw, dt, qm);
            }
            for _ in 1..num_bands {
                write_acpl_df_zero(bw, dt, qm);
            }
        };

    // Helper: emit one real-α `acpl_huff_data()` FREQ-mode block.
    let emit_real_alpha = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, alpha_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let a_q = alpha_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_alpha_f0_value(bw, qm, a_q);
                first = false;
            } else {
                let delta = a_q - prev_q;
                write_acpl_alpha_df_value(bw, qm, delta);
            }
            prev_q = a_q;
        }
    };

    // Helper: emit one real-β `acpl_huff_data()` FREQ-mode block.
    let emit_real_beta = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let b_q = beta_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta_f0_value(bw, qm, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta_df_value(bw, qm, delta);
            }
            prev_q = b_q;
        }
    };

    // Helper: emit one real-γ `acpl_huff_data()` FREQ-mode block. The γ
    // F0 codebook is symmetric around `cb_off` (10 Coarse / 20 Fine) so
    // the F0 codeword carries the signed `gamma_q` directly; the DF
    // codebook addresses `symbol_index = delta_q + cb_off` and supports
    // signed band-to-band deltas which the decoder reverses via
    // `differential_decode`'s DIFF_FREQ branch.
    let emit_real_gamma = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, gamma_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let g_q = gamma_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_gamma_f0_value(bw, qm, g_q);
                first = false;
            } else {
                let delta = g_q - prev_q;
                write_acpl_gamma_df_value(bw, qm, delta);
            }
            prev_q = g_q;
        }
    };

    // alpha1, alpha2 — REAL (ALPHA family, quant_mode_0).
    emit_real_alpha(bw, quant_mode_0, alpha1_q_per_band);
    emit_real_alpha(bw, quant_mode_0, alpha2_q_per_band);
    // beta1, beta2 — REAL (BETA family, quant_mode_0).
    emit_real_beta(bw, quant_mode_0, beta1_q_per_band);
    emit_real_beta(bw, quant_mode_0, beta2_q_per_band);
    // beta3 — zero-delta (BETA3 family, quant_mode_0).
    emit_zero(bw, crate::acpl::AcplDataType::Beta3, quant_mode_0);
    // gamma1..4 — zero-delta (GAMMA family, quant_mode_1).
    for _ in 0..4 {
        emit_zero(bw, crate::acpl::AcplDataType::Gamma, quant_mode_1);
    }
    // gamma5, gamma6 — REAL (GAMMA family, quant_mode_1).
    emit_real_gamma(bw, quant_mode_1, gamma5_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma6_q_per_band);
}

/// Body of `acpl_data_2ch()` (Pseudocode 117 / §5.7.7.6.2 + Table 62)
/// for the 5_X SIMPLE/ASPX_ACPL_3 path with **all** of α₁ / α₂ / β₁ /
/// β₂ / γ₁ / γ₂ / γ₃ / γ₄ / γ₅ / γ₆ emitted as REAL per-parameter-band
/// codewords. β₃ stays at the zero-delta scaffold (its analytic
/// extraction requires per-side surround references plus a model for
/// the third decorrelator output `y₂`, neither of which is observable
/// at encode time for the 5.x PCM input layout).
///
/// The γ₁ / γ₂ (L, Ls) and γ₃ / γ₄ (R, Rs) per-band magnitudes come
/// from
/// [`extract_gamma_1_2_q_per_band_surround_least_squares`] and
/// [`extract_gamma_3_4_q_per_band_surround_least_squares`]; γ₅ / γ₆
/// (centre) come from
/// [`extract_gamma_5_6_q_per_band_centre_least_squares`].
///
/// Each `*_q_per_band` slice must contain at least `num_bands` entries;
/// trailing positions outside the slice are coded as `0`.
#[allow(clippy::too_many_arguments)]
fn write_acpl_data_2ch_real_alpha_beta_full_gamma(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
    alpha1_q_per_band: &[i32],
    alpha2_q_per_band: &[i32],
    beta1_q_per_band: &[i32],
    beta2_q_per_band: &[i32],
    gamma1_q_per_band: &[i32],
    gamma2_q_per_band: &[i32],
    gamma3_q_per_band: &[i32],
    gamma4_q_per_band: &[i32],
    gamma5_q_per_band: &[i32],
    gamma6_q_per_band: &[i32],
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);

    // Helper: emit one zero-delta `acpl_huff_data()` FREQ-mode block.
    let emit_zero =
        |bw: &mut BitWriter, dt: crate::acpl::AcplDataType, qm: crate::acpl::AcplQuantMode| {
            bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
            if num_bands >= 1 {
                write_acpl_f0_zero(bw, dt, qm);
            }
            for _ in 1..num_bands {
                write_acpl_df_zero(bw, dt, qm);
            }
        };

    // Helper: emit one real-α `acpl_huff_data()` FREQ-mode block.
    let emit_real_alpha = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, alpha_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let a_q = alpha_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_alpha_f0_value(bw, qm, a_q);
                first = false;
            } else {
                let delta = a_q - prev_q;
                write_acpl_alpha_df_value(bw, qm, delta);
            }
            prev_q = a_q;
        }
    };

    // Helper: emit one real-β `acpl_huff_data()` FREQ-mode block.
    let emit_real_beta = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let b_q = beta_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta_f0_value(bw, qm, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta_df_value(bw, qm, delta);
            }
            prev_q = b_q;
        }
    };

    // Helper: emit one real-γ `acpl_huff_data()` FREQ-mode block.
    let emit_real_gamma = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, gamma_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let g_q = gamma_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_gamma_f0_value(bw, qm, g_q);
                first = false;
            } else {
                let delta = g_q - prev_q;
                write_acpl_gamma_df_value(bw, qm, delta);
            }
            prev_q = g_q;
        }
    };

    // alpha1, alpha2 — REAL (ALPHA family, quant_mode_0).
    emit_real_alpha(bw, quant_mode_0, alpha1_q_per_band);
    emit_real_alpha(bw, quant_mode_0, alpha2_q_per_band);
    // beta1, beta2 — REAL (BETA family, quant_mode_0).
    emit_real_beta(bw, quant_mode_0, beta1_q_per_band);
    emit_real_beta(bw, quant_mode_0, beta2_q_per_band);
    // beta3 — zero-delta (BETA3 family, quant_mode_0).
    emit_zero(bw, crate::acpl::AcplDataType::Beta3, quant_mode_0);
    // gamma1..gamma4 — REAL (GAMMA family, quant_mode_1).
    emit_real_gamma(bw, quant_mode_1, gamma1_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma2_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma3_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma4_q_per_band);
    // gamma5, gamma6 — REAL (GAMMA family, quant_mode_1).
    emit_real_gamma(bw, quant_mode_1, gamma5_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma6_q_per_band);
}

/// Emit a full `acpl_data_2ch()` body per §4.2.13.4 Table 62 with the
/// α₁ / α₂ / β₁ / β₂ / β₃ / γ₁..γ₆ entropy layers ALL carrying real
/// per-parameter-band values — the round-285 β₃-real extension of
/// [`write_acpl_data_2ch_real_alpha_beta_full_gamma`]. The β₃ layer
/// (BETA3 codebook family, `quant_mode_0`) emits one F0 codeword at
/// band 0 followed by DF deltas, same FREQ-direction DPCM shape as the
/// α / β / γ layers. An all-zero `beta3_q_per_band` is byte-identical
/// to the zero-delta scaffold emission (the F0 value 0 and DF delta 0
/// codewords are exactly the `write_acpl_f0_zero` / `write_acpl_df_zero`
/// picks).
#[allow(clippy::too_many_arguments)]
/// Write one ACPL DT codeword (`symbol_index = delta_q + cb_off`) for
/// the given data-type family — the DIFF_TIME dual of the per-family
/// F0 / DF value writers.
fn write_acpl_dt_value(
    bw: &mut BitWriter,
    data_type: crate::acpl::AcplDataType,
    qm: crate::acpl::AcplQuantMode,
    delta_q: i32,
) {
    let (len, cw, cb_off) = acpl_hcb_arrays(data_type, qm, crate::acpl::AcplHcbType::Dt);
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Emit one ACPL F0 codeword for the given family (dispatch over the
/// per-family value writers).
fn write_acpl_f0_value_for(
    bw: &mut BitWriter,
    data_type: crate::acpl::AcplDataType,
    qm: crate::acpl::AcplQuantMode,
    q: i32,
) {
    match data_type {
        crate::acpl::AcplDataType::Alpha => write_acpl_alpha_f0_value(bw, qm, q),
        crate::acpl::AcplDataType::Beta => write_acpl_beta_f0_value(bw, qm, q),
        crate::acpl::AcplDataType::Beta3 => write_acpl_beta3_f0_value(bw, qm, q),
        crate::acpl::AcplDataType::Gamma => write_acpl_gamma_f0_value(bw, qm, q),
    }
}

/// Emit one ACPL DF codeword for the given family.
fn write_acpl_df_value_for(
    bw: &mut BitWriter,
    data_type: crate::acpl::AcplDataType,
    qm: crate::acpl::AcplQuantMode,
    delta: i32,
) {
    match data_type {
        crate::acpl::AcplDataType::Alpha => write_acpl_alpha_df_value(bw, qm, delta),
        crate::acpl::AcplDataType::Beta => write_acpl_beta_df_value(bw, qm, delta),
        crate::acpl::AcplDataType::Beta3 => write_acpl_beta3_df_value(bw, qm, delta),
        crate::acpl::AcplDataType::Gamma => write_acpl_gamma_df_value(bw, qm, delta),
    }
}

/// Table 62 element order → (data type, quant-mode selector) map.
/// `false` selects `quant_mode_0`, `true` selects `quant_mode_1`.
const ACPL_2CH_ELEMENT_TYPES: [(crate::acpl::AcplDataType, bool); 11] = [
    (crate::acpl::AcplDataType::Alpha, false),
    (crate::acpl::AcplDataType::Alpha, false),
    (crate::acpl::AcplDataType::Beta, false),
    (crate::acpl::AcplDataType::Beta, false),
    (crate::acpl::AcplDataType::Beta3, false),
    (crate::acpl::AcplDataType::Gamma, true),
    (crate::acpl::AcplDataType::Gamma, true),
    (crate::acpl::AcplDataType::Gamma, true),
    (crate::acpl::AcplDataType::Gamma, true),
    (crate::acpl::AcplDataType::Gamma, true),
    (crate::acpl::AcplDataType::Gamma, true),
];

/// `acpl_data_2ch()` writer with **per-element transmission
/// directions** (Table 65 `diff_type`) — the P-frame generalisation of
/// [`write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3`]. `params`
/// holds the 11 elements in Table 62 order (α₁, α₂, β₁, β₂, β₃,
/// γ₁..γ₆); all-FREQ rows reproduce the legacy writer byte-for-byte.
fn write_acpl_data_2ch_directional(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
    params: &[AcplEncodedParam; 11],
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);
    for (elem, (data_type, use_qm1)) in params.iter().zip(ACPL_2CH_ELEMENT_TYPES.iter()) {
        let qm = if *use_qm1 { quant_mode_1 } else { quant_mode_0 };
        bw.write_bit(elem.direction_time); // diff_type
        if elem.direction_time {
            // DIFF_TIME: one DT codeword per band.
            for pb in 0..num_bands {
                let d = elem.values.get(pb as usize).copied().unwrap_or(0);
                write_acpl_dt_value(bw, *data_type, qm, d);
            }
        } else {
            // DIFF_FREQ: F0 + band-to-band DF deltas from the absolute row.
            let mut prev_q: i32 = 0;
            for pb in 0..num_bands {
                let q = elem.values.get(pb as usize).copied().unwrap_or(0);
                if pb == 0 {
                    write_acpl_f0_value_for(bw, *data_type, qm, q);
                } else {
                    write_acpl_df_value_for(bw, *data_type, qm, q - prev_q);
                }
                prev_q = q;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3(
    bw: &mut BitWriter,
    num_bands: u32,
    quant_mode_0: crate::acpl::AcplQuantMode,
    quant_mode_1: crate::acpl::AcplQuantMode,
    alpha1_q_per_band: &[i32],
    alpha2_q_per_band: &[i32],
    beta1_q_per_band: &[i32],
    beta2_q_per_band: &[i32],
    beta3_q_per_band: &[i32],
    gamma1_q_per_band: &[i32],
    gamma2_q_per_band: &[i32],
    gamma3_q_per_band: &[i32],
    gamma4_q_per_band: &[i32],
    gamma5_q_per_band: &[i32],
    gamma6_q_per_band: &[i32],
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);

    // Helper: emit one real-α `acpl_huff_data()` FREQ-mode block.
    let emit_real_alpha = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, alpha_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let a_q = alpha_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_alpha_f0_value(bw, qm, a_q);
                first = false;
            } else {
                let delta = a_q - prev_q;
                write_acpl_alpha_df_value(bw, qm, delta);
            }
            prev_q = a_q;
        }
    };

    // Helper: emit one real-β `acpl_huff_data()` FREQ-mode block.
    let emit_real_beta = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let b_q = beta_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta_f0_value(bw, qm, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta_df_value(bw, qm, delta);
            }
            prev_q = b_q;
        }
    };

    // Helper: emit one real-β₃ `acpl_huff_data()` FREQ-mode block.
    let emit_real_beta3 = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta3_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let b_q = beta3_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta3_f0_value(bw, qm, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta3_df_value(bw, qm, delta);
            }
            prev_q = b_q;
        }
    };

    // Helper: emit one real-γ `acpl_huff_data()` FREQ-mode block.
    let emit_real_gamma = |bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, gamma_q: &[i32]| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in 0..num_bands {
            let g_q = gamma_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_gamma_f0_value(bw, qm, g_q);
                first = false;
            } else {
                let delta = g_q - prev_q;
                write_acpl_gamma_df_value(bw, qm, delta);
            }
            prev_q = g_q;
        }
    };

    // alpha1, alpha2 — REAL (ALPHA family, quant_mode_0).
    emit_real_alpha(bw, quant_mode_0, alpha1_q_per_band);
    emit_real_alpha(bw, quant_mode_0, alpha2_q_per_band);
    // beta1, beta2 — REAL (BETA family, quant_mode_0).
    emit_real_beta(bw, quant_mode_0, beta1_q_per_band);
    emit_real_beta(bw, quant_mode_0, beta2_q_per_band);
    // beta3 — REAL (BETA3 family, quant_mode_0).
    emit_real_beta3(bw, quant_mode_0, beta3_q_per_band);
    // gamma1..gamma6 — REAL (GAMMA family, quant_mode_1).
    emit_real_gamma(bw, quant_mode_1, gamma1_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma2_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma3_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma4_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma5_q_per_band);
    emit_real_gamma(bw, quant_mode_1, gamma6_q_per_band);
}

// ====================================================================
// Top-level body builder: `5_X_channel_element` ASPX_ACPL_3
// ====================================================================

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body that the decoder's
/// [`crate::mch::parse_5x_audio_data_outer`] (with `mode = AspxAcpl3`)
/// walks end-to-end and synthesises 5-channel `[L, R, C, Ls, Rs]` PCM
/// via [`crate::acpl_synth::run_acpl_5x_mch_pcm`].
///
/// `coeffs_per_channel` holds the forward-MDCT carrier spectra. For
/// 5.0 the layout is `[L_carrier, R_carrier, C_carrier]` (length 3); for
/// 5.1 it is `[L_carrier, R_carrier, C_carrier, LFE]` (length 4). The
/// centre carrier is unused on the ASPX_ACPL_3 spec path (the centre is
/// reconstructed from `cfg0_centre_mono.scaled_spec` elsewhere or zero-
/// filled when that's missing — see `Ac4Decoder::receive_frame`).
///
/// `pad_target_bytes` sizes the trailing zero-pad so the substream body
/// fits the caller's frame-rate / bit-rate budget. The audio-size header
/// is set to `pad_target_bytes`.
///
/// Returns the substream bytes (`audio_size` header + audio data + zero
/// padding) sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // I-frame: aspx_data_2ch() + acpl_data_2ch().
    if b_iframe {
        // aspx_data_2ch() is a Result internally; we treat its failure as
        // a panic at encoding time since the cfg comes from the caller.
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_2ch_minimal(&mut bw, acpl_num_bands, acpl_qm0, acpl_qm1);
    }

    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body identical to
/// [`build_5_x_acpl3_body_from_pcm_spectra`] but with the β1 / β2
/// entropy layers carrying real per-parameter-band magnitudes derived
/// from the L / R carrier energies. α1 / α2 / β3 / γ1..γ6 stay at the
/// round-95 zero-delta scaffold.
///
/// `beta_scale` controls the encoder's wet/dry balance for the surround
/// reconstruction (`β = β_scale · √E[x²]`); see
/// [`extract_beta_q_per_band_carrier_energy`] for the rationale and
/// recommended range. `beta_scale = 0.0` reproduces the round-95
/// zero-delta scaffold byte-for-byte at the β1 / β2 positions.
///
/// The decoder walks the same Table 25 ASPX_ACPL_3 body and applies the
/// recovered β1 / β2 to the ACplModule2 mix (Pseudocode 119). With α1 =
/// α2 = 0 and β3 = 0 the synthesis at parameter band `pb` reduces to:
///
/// ```text
///   z0[ts][sb] = 0.5 · ( x0[ts][sb]·g1 + x1[ts][sb]·g2 + y0[ts][sb]·β1 )
///   z1[ts][sb] = 0.5 · ( x0[ts][sb]·g1 + x1[ts][sb]·g2 − y0[ts][sb]·β1 )
/// ```
///
/// (and analogously for `(z2, z3)` with β2 driving the second
/// ACplModule2 instance). Non-zero β1 / β2 therefore drive the
/// decorrelator injection that gives the Ls / Rs outputs their
/// decorrelated spaciousness.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_beta(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    beta_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    // Extract per-band β_q from the L and R carrier energy distributions.
    // start_pb = 0 for ACPL_3 because the ACPL_3 path codes all
    // parameter bands across the QMF range (no PARTIAL `acpl_qmf_band`
    // cutoff — that's ACPL_1 only).
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // I-frame: aspx_data_2ch() + acpl_data_2ch() with real β1 / β2.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_2ch_real_beta(
            &mut bw,
            acpl_num_bands,
            acpl_qm0,
            acpl_qm1,
            &beta1_q,
            &beta2_q,
        );
    }

    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body identical to
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_beta`] but with the
/// α1 / α2 entropy layers ALSO carrying real per-parameter-band
/// magnitudes derived from the L↔R carrier cross-correlation (in
/// addition to the r193 real β1 / β2). β3 / γ1..γ6 stay at the
/// round-95 zero-delta scaffold.
///
/// `alpha_scale` controls the encoder's front/back balance policy for
/// the dry-mix split — see
/// [`extract_alpha_q_per_band_carrier_correlation`] for the per-band
/// mapping rationale. `alpha_scale = 0.0` reproduces the
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_beta`] output
/// byte-for-byte (all-zero α layers).
///
/// `beta_scale` retains the r193 meaning and is forwarded unchanged to
/// [`extract_beta_q_per_band_carrier_energy`]. With
/// `alpha_scale = beta_scale = 0.0` this entry point reproduces the
/// round-95 zero-delta scaffold
/// ([`build_5_x_acpl3_body_from_pcm_spectra`]) byte-for-byte.
///
/// The decoder walks the same Table 25 ASPX_ACPL_3 body and applies the
/// recovered α1 / α2 (in addition to β1 / β2) to the two ACplModule2
/// instances (Pseudocode 119). With γ1..γ6 still at the zero-delta
/// mid-range and β3 = 0 the synthesis at parameter band `pb` becomes:
///
/// ```text
///   z0[ts][sb] = 0.5 · ( (1+α1)·(g·x0 + g·x1) + y0·β1 )
///   z1[ts][sb] = 0.5 · ( (1−α1)·(g·x0 + g·x1) − y0·β1 )
///   z2[ts][sb] = 0.5 · ( (1+α2)·(g·x0 + g·x1) + y1·β2 )
///   z3[ts][sb] = 0.5 · ( (1−α2)·(g·x0 + g·x1) − y1·β2 )
/// ```
///
/// where `g` is the zero-delta γ mid-range gain shared by all six γ
/// slots. Non-zero α modulates the front/back energy ratio: higher α →
/// more energy in the front pair, less in the surround pair (the
/// surround pair then leans on the β-driven decorrelator injection).
/// Highly-correlated L/R bands (mono-like content) therefore bias more
/// of the dry mix toward the front, and decorrelated bands keep the
/// front/back energy balanced.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    // α₁ and α₂ both receive the same L↔R-correlation policy: the two
    // ACplModule2 instances in ACPL_3 share the (L, R) carrier pair as
    // their (x0, x1) input, so without a per-side surround reference at
    // encode time the natural choice is one extractor driving both α
    // layers. The asymmetry between the (L, Ls) and (R, Rs) outputs is
    // already carried by β1 vs β2 (driven from E[L²] vs E[R²]) and by
    // the two independent decorrelator outputs y0 vs y1.
    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        coeffs_l,
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        alpha_scale,
        acpl_qm0,
    );
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // I-frame: aspx_data_2ch() + acpl_data_2ch() with real α₁ / α₂ / β₁ / β₂.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_2ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            acpl_qm0,
            acpl_qm1,
            &alpha_q,
            &alpha_q,
            &beta1_q,
            &beta2_q,
        );
    }

    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body identical to
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta`] but with
/// the γ5 / γ6 entropy layers ALSO carrying real per-parameter-band
/// magnitudes derived from a 2×2 per-band least-squares fit
/// `C ≈ K · (γ5·L + γ6·R)` (Pseudocode 118 step 7 + step 11 with
/// `K = √2 · (1 + √2) / 2`). β3 / γ1..γ4 stay at the round-95
/// zero-delta scaffold.
///
/// `gamma_scale` controls the magnitude of the recovered γ pair:
/// `gamma_scale = 1.0` reproduces the analytic least-squares solution
/// (clamped to the Table-208 ±2.0 magnitude bound);
/// `gamma_scale = 0.0` reproduces the round-95 zero-delta scaffold
/// byte-for-byte at the γ5 / γ6 positions.
///
/// With `alpha_scale = beta_scale = gamma_scale = 0.0` this entry point
/// reproduces the round-95 zero-delta scaffold
/// ([`build_5_x_acpl3_body_from_pcm_spectra`]) byte-for-byte.
///
/// `coeffs_c` is the centre-channel MDCT spectrum used by the γ5 / γ6
/// least-squares extractor; if `None` the centre layer falls back to the
/// zero-delta scaffold (matching
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta`] byte-for-
/// byte when `gamma_scale = 0.0`).
///
/// The decoder walks the same Table 25 ASPX_ACPL_3 body and applies the
/// recovered γ5 / γ6 to the third ACplModule2 instance (Pseudocode 119
/// with `a = 1`, `b = 0`, `y = 0`); with γ1..γ4 still at zero-delta
/// and β3 = 0, only the centre (`z4`) reconstruction sees a non-trivial
/// γ-driven mix:
///
/// ```text
///   z4 = 0.5 · (γ5·x0in + γ6·x1in)
///   C  = √2 · z4 = √2 · 0.5 · (γ5·x0in + γ6·x1in)
/// ```
///
/// (with `x0in / x1in = (1 + √2) · L / R` from Pseudocode 118 step 1).
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_gamma(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        coeffs_l,
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        alpha_scale,
        acpl_qm0,
    );
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let (g5_q, g6_q) = if let Some(coeffs_c_buf) = coeffs_c {
        extract_gamma_5_6_q_per_band_centre_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // I-frame: aspx_data_2ch() + acpl_data_2ch() with real α/β/γ5/γ6.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_2ch_real_alpha_beta_gamma(
            &mut bw,
            acpl_num_bands,
            acpl_qm0,
            acpl_qm1,
            &alpha_q,
            &alpha_q,
            &beta1_q,
            &beta2_q,
            &g5_q,
            &g6_q,
        );
    }

    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body identical to
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_gamma`] but
/// with the γ₁ / γ₂ / γ₃ / γ₄ entropy layers ALSO carrying real
/// per-parameter-band magnitudes derived from per-band 2×2
/// least-squares fits of the (L, Ls) and (R, Rs) output channel pairs
/// onto the (L, R) carrier pair (Pseudocode 118 step 5 / step 6 +
/// step 11).
///
/// Per [`extract_gamma_1_2_q_per_band_surround_least_squares`] the
/// `(L + Ls/√2)` sum is independent of the (α₁, β₁) decorrelator
/// contribution and equals `(1 + √2) · (γ₁·L + γ₂·R)`; by symmetry
/// the `(R + Rs/√2)` sum equals `(1 + √2) · (γ₃·L + γ₄·R)`. Each pair
/// solves the same 2×2 normal-equations system as the round-208
/// γ₅ / γ₆ centre fit but with the per-side surround pair as the
/// target.
///
/// β₃ stays at the round-95 zero-delta scaffold — its analytic
/// extraction requires per-side surround references plus a model for
/// the third decorrelator output `y₂`, neither of which is observable
/// at encode time for the 5.x PCM input layout.
///
/// `gamma_scale` controls the magnitude of the recovered γ values
/// (applied uniformly to γ₁..γ₆): `gamma_scale = 1.0` reproduces the
/// analytic least-squares solution (clamped to the Table-208 ±2.0
/// magnitude bound); `gamma_scale = 0.0` reproduces the round-95
/// zero-delta scaffold byte-for-byte at the γ₁..γ₆ positions.
///
/// With `alpha_scale = beta_scale = gamma_scale = 0.0` this entry point
/// reproduces the round-95 zero-delta scaffold
/// ([`build_5_x_acpl3_body_from_pcm_spectra`]) byte-for-byte.
///
/// `coeffs_c` / `coeffs_ls` / `coeffs_rs` are the centre / surround-
/// left / surround-right MDCT spectra used by the γ extractors; if
/// `None` the corresponding γ layer falls back to the zero-delta
/// scaffold.
///
/// The decoder walks the same Table 25 ASPX_ACPL_3 body and applies
/// the recovered γ matrices through all three ACplModule2 invocations
/// (Pseudocode 118 steps 5 / 6 / 7), so the (L, Ls), (R, Rs) and
/// centre output channels can now all carry input-derived γ-shaped
/// dry mix in place of the previous-round zero-γ silence.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        coeffs_l,
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        alpha_scale,
        acpl_qm0,
    );
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let (g1_q, g2_q) = if let Some(coeffs_ls_buf) = coeffs_ls {
        extract_gamma_1_2_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_ls_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g3_q, g4_q) = if let Some(coeffs_rs_buf) = coeffs_rs {
        extract_gamma_3_4_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_rs_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g5_q, g6_q) = if let Some(coeffs_c_buf) = coeffs_c {
        extract_gamma_5_6_q_per_band_centre_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // I-frame: aspx_data_2ch() + acpl_data_2ch() with real α/β/γ1..6.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_2ch_real_alpha_beta_full_gamma(
            &mut bw,
            acpl_num_bands,
            acpl_qm0,
            acpl_qm1,
            &alpha_q,
            &alpha_q,
            &beta1_q,
            &beta2_q,
            &g1_q,
            &g2_q,
            &g3_q,
            &g4_q,
            &g5_q,
            &g6_q,
        );
    }

    finish_substream_body(bw)
}

/// Like [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma`]
/// but with the β₃ entropy layer ALSO carrying real per-parameter-band
/// values derived by energy-matching the centre-channel reconstruction
/// residual against the third-decorrelator drive (Pseudocode 118 steps
/// 2 / 7 / 10 / 11 — see
/// [`extract_beta3_q_per_band_centre_residual`]). This closes the
/// round-215 "β₃ stays at the round-95 zero-delta scaffold" deferral:
/// the unobservable decoder-side decorrelator output `y₂` is modelled
/// by its drive energy `E[v₃²]` (the decorrelator + ducker chain is
/// energy-preserving in steady state), which IS observable at encode
/// time from the carrier spectra and the quantised γ matrix.
///
/// `beta3_scale` controls the magnitude of the recovered β₃ values:
/// `beta3_scale = 1.0` applies the full energy-matching solution
/// (clamped to the Table-207 ±1.0 magnitude bound); `beta3_scale = 0.0`
/// reproduces the round-215
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma`]
/// output byte-for-byte (an all-zero β₃ row emits exactly the
/// zero-delta scaffold codewords).
///
/// With `alpha_scale = beta_scale = gamma_scale = beta3_scale = 0.0`
/// this entry point reproduces the round-95 zero-delta scaffold
/// ([`build_5_x_acpl3_body_from_pcm_spectra`]) byte-for-byte.
///
/// β₃ extraction requires the centre spectrum (the residual target)
/// and fires only when `coeffs_c` is `Some`; with `coeffs_c = None`
/// the β₃ layer falls back to the zero-delta scaffold.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    beta3_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        coeffs_l,
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        alpha_scale,
        acpl_qm0,
    );
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let (g1_q, g2_q) = if let Some(coeffs_ls_buf) = coeffs_ls {
        extract_gamma_1_2_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_ls_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g3_q, g4_q) = if let Some(coeffs_rs_buf) = coeffs_rs {
        extract_gamma_3_4_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_rs_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g5_q, g6_q) = if let Some(coeffs_c_buf) = coeffs_c {
        extract_gamma_5_6_q_per_band_centre_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let beta3_q = if let Some(coeffs_c_buf) = coeffs_c {
        extract_beta3_q_per_band_centre_residual(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            &g1_q,
            &g2_q,
            &g3_q,
            &g4_q,
            &g5_q,
            &g6_q,
            transform_length,
            acpl_num_bands,
            0,
            beta3_scale,
            acpl_qm1,
            acpl_qm0,
        )
    } else {
        vec![0i32; acpl_num_bands as usize]
    };

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // I-frame: aspx_data_2ch() + acpl_data_2ch() with real α/β/β₃/γ1..6.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3(
            &mut bw,
            acpl_num_bands,
            acpl_qm0,
            acpl_qm1,
            &alpha_q,
            &alpha_q,
            &beta1_q,
            &beta2_q,
            &beta3_q,
            &g1_q,
            &g2_q,
            &g3_q,
            &g4_q,
            &g5_q,
            &g6_q,
        );
    }

    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body identical to
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3`]
/// but with the I-frame `aspx_data_2ch()` element carrying **real**
/// per-subband-group SIGNAL / NOISE envelope quant indices for the L / R
/// carriers (via [`write_aspx_data_2ch_real_envelope`]) instead of the
/// round-95 minimum-bit-cost ASPX scaffold (`write_aspx_data_2ch_minimal`).
///
/// This is the frame-path wiring of the round-226 real-envelope ASPX
/// writers: the caller passes the `[F0, DF₁, …]` SIGNAL + NOISE quant
/// index vectors for channel 0 (L carrier) and channel 1 (R carrier) —
/// the same shape [`build_aspx_real_envelope_channel_from_qmf`] produces
/// from an HF QMF matrix — and this builder splices them into the live
/// substream body in place of the scaffold. Every other element (LFE,
/// companding, split-MDCT stereo, the full real α / β / β₃ / γ₁..γ₆
/// A-CPL layers) is byte-for-byte identical to the round-285 builder.
///
/// The decoder walks the same Table 25 ASPX_ACPL_3 body; the
/// real-envelope `aspx_data_2ch()` is a strict framing superset of the
/// minimal one (FIXFIX, `num_env = 1`, `aspx_balance = 1`, all-FREQ
/// `aspx_delta_dir`, zero `aspx_hfgen_iwc_2ch`), so the substream still
/// parses end-to-end and synthesises 5-channel PCM. The difference is
/// that the SIGNAL / NOISE envelope ec_data now carries the L / R HF
/// energy distribution rather than an all-zero scaffold.
///
/// Passing all-zero / empty envelope vectors for both channels
/// reproduces the round-285 scaffold body byte-for-byte at the
/// `aspx_data_2ch()` position (the round-226 writer's all-zero path
/// resolves to the same minimum-bit-cost F0 / DF codewords the minimal
/// writer emits).
///
/// Refs ETSI TS 103 190-1 §4.2.6.6 Table 25, §4.2.12.4 Table 52,
/// §5.7.6.3.4 / §5.7.6.3.5 Pseudocodes 80–83.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    aspx_l_sig: &[i32],
    aspx_l_noise: &[i32],
    aspx_r_sig: &[i32],
    aspx_r_noise: &[i32],
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    beta3_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    // Delegate with an empty tna_mode → all-zero `aspx_hfgen_iwc_2ch`,
    // reproducing the historical (round-285/322) body byte-for-byte.
    build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna(
        transform_length,
        max_sfb,
        max_sfb_lfe,
        b_iframe,
        coeffs_l,
        coeffs_r,
        coeffs_c,
        coeffs_ls,
        coeffs_rs,
        coeffs_lfe,
        aspx_cfg,
        aspx_l_sig,
        aspx_l_noise,
        aspx_r_sig,
        aspx_r_noise,
        &[],
        &[],
        &[],
        acpl_num_param_bands_id,
        acpl_qm0,
        acpl_qm1,
        alpha_scale,
        beta_scale,
        gamma_scale,
        beta3_scale,
        pad_target_bytes,
    )
}

/// `aspx_tna_mode`-carrying variant of
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx`].
///
/// Identical except the I-frame `aspx_data_2ch()` element is emitted via
/// [`write_aspx_data_2ch_real_envelope_tna`] with the caller-supplied
/// per-noise-subband-group `tna_mode` vector (signalled once on channel 0
/// and mirrored to channel 1 by `aspx_balance = 1`). An empty / all-zero
/// `tna_mode` reproduces the non-tna builder byte-for-byte.
///
/// The `tna_mode` is the encoder's §4.3.10.6.1 inverse-filtering decision
/// — typically produced by [`crate::aspx_tna_select::select_tna_mode`]
/// from the L-carrier QMF low band. `aspx_l_ah` / `aspx_r_ah` are the
/// per-channel `aspx_add_harmonic` decisions (§4.2.12.6) — typically from
/// [`crate::aspx_ah_select::select_add_harmonic`] over each carrier's HF
/// QMF band; empty slices reproduce the all-zero `add_harmonic` scaffold.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    aspx_l_sig: &[i32],
    aspx_l_noise: &[i32],
    aspx_r_sig: &[i32],
    aspx_r_noise: &[i32],
    aspx_tna_mode: &[u8],
    aspx_l_ah: &[bool],
    aspx_r_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    beta3_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let freq = |v: &[i32]| AspxEncodedEnvelope {
        values: v.to_vec(),
        direction_time: false,
    };
    build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna_directional(
        transform_length,
        max_sfb,
        max_sfb_lfe,
        b_iframe,
        coeffs_l,
        coeffs_r,
        coeffs_c,
        coeffs_ls,
        coeffs_rs,
        coeffs_lfe,
        aspx_cfg,
        &freq(aspx_l_sig),
        &freq(aspx_l_noise),
        &freq(aspx_r_sig),
        &freq(aspx_r_noise),
        aspx_tna_mode,
        aspx_l_ah,
        aspx_r_ah,
        acpl_num_param_bands_id,
        acpl_qm0,
        acpl_qm1,
        alpha_scale,
        beta_scale,
        gamma_scale,
        beta3_scale,
        pad_target_bytes,
        None,
    )
}

/// `audio_data_chan(5.X)` body of
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna_directional`]
/// written into the caller's `BitWriter` — from the 3-bit
/// `5_X_codec_mode` through the trailing `acpl_data_2ch()`, without
/// the `ac4_substream()` size header, alignment or padding. Lets the
/// same ASPX_ACPL_3 core be embedded where Table 25 is nested, e.g.
/// the TS 103 190-2 §6.2.3.4 A-JOC `b_static_dmx` downmix.
#[allow(clippy::too_many_arguments)]
pub fn write_5_x_acpl3_audio_data_directional(
    bw: &mut BitWriter,
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    aspx_l_sig: &AspxEncodedEnvelope,
    aspx_l_noise: &AspxEncodedEnvelope,
    aspx_r_sig: &AspxEncodedEnvelope,
    aspx_r_noise: &AspxEncodedEnvelope,
    aspx_tna_mode: &[u8],
    aspx_l_ah: &[bool],
    aspx_r_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    beta3_scale: f32,
    acpl_prev: Option<&mut Acpl3ParamPrevRows>,
) {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        coeffs_l,
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        alpha_scale,
        acpl_qm0,
    );
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let (g1_q, g2_q) = if let Some(coeffs_ls_buf) = coeffs_ls {
        extract_gamma_1_2_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_ls_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g3_q, g4_q) = if let Some(coeffs_rs_buf) = coeffs_rs {
        extract_gamma_3_4_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_rs_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g5_q, g6_q) = if let Some(coeffs_c_buf) = coeffs_c {
        extract_gamma_5_6_q_per_band_centre_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let beta3_q = if let Some(coeffs_c_buf) = coeffs_c {
        extract_beta3_q_per_band_centre_residual(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            &g1_q,
            &g2_q,
            &g3_q,
            &g4_q,
            &g5_q,
            &g6_q,
            transform_length,
            acpl_num_bands,
            0,
            beta3_scale,
            acpl_qm1,
            acpl_qm0,
        )
    } else {
        vec![0i32; acpl_num_bands as usize]
    };

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(bw, aspx_cfg);
        write_acpl_config_2ch(bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // Real-envelope aspx_data_2ch() + acpl_data_2ch() with real
    // α / β / β₃ / γ₁..γ₆. Per Table 25 both data elements are present
    // on every frame; only the configs above and the 3-bit xover offset
    // inside aspx_data_2ch() are I-frame-gated.
    {
        write_aspx_data_2ch_directional_envelope_tna_ah_framed(
            bw,
            aspx_cfg,
            aspx_l_sig,
            aspx_l_noise,
            aspx_r_sig,
            aspx_r_noise,
            aspx_tna_mode,
            aspx_l_ah,
            aspx_r_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        // A-CPL parameter layer: on P-frames with a primed previous-row
        // state each Table 62 element may switch to DIFF_TIME (Table 65)
        // against the previous frame's row — the decoder's per-element
        // AcplDiffState accumulates it per Pseudocode 121. I-frames and
        // unprimed states emit DIFF_FREQ rows (byte-identical to the
        // legacy writer).
        let cur_rows: [&[i32]; 11] = [
            &alpha_q, &alpha_q, &beta1_q, &beta2_q, &beta3_q, &g1_q, &g2_q, &g3_q, &g4_q, &g5_q,
            &g6_q,
        ];
        let use_prev = !b_iframe && acpl_prev.as_ref().map(|st| st.primed).unwrap_or(false);
        let params: [AcplEncodedParam; 11] = std::array::from_fn(|i| {
            let prev = if use_prev {
                acpl_prev.as_ref().map(|st| st.rows[i].as_slice())
            } else {
                None
            };
            choose_acpl_direction(cur_rows[i], prev)
        });
        write_acpl_data_2ch_directional(bw, acpl_num_bands, acpl_qm0, acpl_qm1, &params);
        // The transmitted absolute rows become the next frame's
        // Pseudocode 121 q_prev reference.
        if let Some(st) = acpl_prev {
            for (slot, row) in st.rows.iter_mut().zip(cur_rows.iter()) {
                slot.clear();
                slot.extend_from_slice(row);
            }
            st.primed = true;
        }
    }
}

/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna`]
/// with **per-channel, per-data-type A-SPX envelope transmission
/// directions** ([`AspxEncodedEnvelope`] rows instead of raw FREQ-DPCM
/// slices) — the P-frame builder: TIME-direction rows delta-code
/// against the previous frame's last envelope (§5.7.6.3.4 Pseudocodes
/// 80 / 81 `qscf_*_prev`), which the decoder reconstructs from its
/// per-channel [`crate::aspx::AspxEnvPrev`] state. All-FREQ rows
/// reproduce the plain `_real_aspx_tna` builder byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_tna_directional(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    aspx_l_sig: &AspxEncodedEnvelope,
    aspx_l_noise: &AspxEncodedEnvelope,
    aspx_r_sig: &AspxEncodedEnvelope,
    aspx_r_noise: &AspxEncodedEnvelope,
    aspx_tna_mode: &[u8],
    aspx_l_ah: &[bool],
    aspx_r_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    beta3_scale: f32,
    pad_target_bytes: usize,
    acpl_prev: Option<&mut Acpl3ParamPrevRows>,
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    write_5_x_acpl3_audio_data_directional(
        &mut bw,
        transform_length,
        max_sfb,
        max_sfb_lfe,
        b_iframe,
        coeffs_l,
        coeffs_r,
        coeffs_c,
        coeffs_ls,
        coeffs_rs,
        coeffs_lfe,
        aspx_cfg,
        aspx_l_sig,
        aspx_l_noise,
        aspx_r_sig,
        aspx_r_noise,
        aspx_tna_mode,
        aspx_l_ah,
        aspx_r_ah,
        acpl_num_param_bands_id,
        acpl_qm0,
        acpl_qm1,
        alpha_scale,
        beta_scale,
        gamma_scale,
        beta3_scale,
        acpl_prev,
    );
    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_3 substream body identical to
/// [`build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx`]
/// but with the I-frame `aspx_data_2ch()` element carrying a **real
/// multi-envelope** SIGNAL / NOISE payload (`num_env > 1`) for the L / R
/// carriers via [`write_aspx_data_2ch_multi_envelope`] instead of the
/// single-envelope [`write_aspx_data_2ch_real_envelope`].
///
/// This is the frame-path wiring of the round-299 multi-envelope
/// `aspx_data_2ch()` writer and the round-316
/// [`build_aspx_multi_envelope_2ch_from_qmf`] QMF→rows builder: the caller
/// passes the per-channel [`AspxMultiEnvelopeChannel`] rows (one
/// [`AspxEncodedEnvelope`] per envelope, each carrying its chosen FREQ /
/// TIME DPCM direction) together with the frame's chosen FIXFIX `num_env`
/// (a power of two in `2..=1 << fixfix_tmp_num_env_bits()`). Every other
/// element (LFE, companding, split-MDCT stereo, the full real
/// α / β / β₃ / γ₁..γ₆ A-CPL layers) is byte-for-byte identical to the
/// round-322 single-envelope builder.
///
/// The decoder walks the same Table 25 ASPX_ACPL_3 body; the
/// multi-envelope `aspx_data_2ch()` signals `tmp_num_env = log2(num_env)`
/// in the FIXFIX `aspx_framing()` so the decoder derives `num_env`
/// SIGNAL envelopes and `num_noise = 2` NOISE envelopes, then walks the
/// per-envelope SIGNAL / NOISE ec_data. The body parses end-to-end
/// through [`crate::asf::parse_aspx_data_2ch_body`] and synthesises
/// 5-channel PCM. Per Table 52, with `num_env > 1` the FIXFIX
/// `num_env == 1` → Fine quant clamp does **not** apply, so `cfg`'s
/// `quant_mode_env` carries through unchanged.
///
/// Returns an empty `Vec` when [`write_aspx_data_2ch_multi_envelope`]
/// rejects the config / `num_env` combination (caller should fall back to
/// the single-envelope path); the substream is otherwise structurally
/// identical to the round-322 builder.
///
/// Refs ETSI TS 103 190-1 §4.2.6.6 Table 25, §4.2.12.4 Table 52,
/// §4.3.10.1.9 Table 123, §5.7.6.3.4 / §5.7.6.3.5 Pseudocodes 80–83.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_multi_env(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: Option<&[f32]>,
    coeffs_ls: Option<&[f32]>,
    coeffs_rs: Option<&[f32]>,
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    aspx_num_env: u32,
    aspx_rows: &AspxMultiEnvelope2chRows,
    aspx_l_ah: &[bool],
    aspx_r_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_qm0: crate::acpl::AcplQuantMode,
    acpl_qm1: crate::acpl::AcplQuantMode,
    alpha_scale: f32,
    beta_scale: f32,
    gamma_scale: f32,
    beta3_scale: f32,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);

    let alpha_q = extract_alpha_q_per_band_carrier_correlation(
        coeffs_l,
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        alpha_scale,
        acpl_qm0,
    );
    let beta1_q = extract_beta_q_per_band_carrier_energy(
        coeffs_l,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let beta2_q = extract_beta_q_per_band_carrier_energy(
        coeffs_r,
        transform_length,
        acpl_num_bands,
        0,
        beta_scale,
        acpl_qm0,
    );
    let (g1_q, g2_q) = if let Some(coeffs_ls_buf) = coeffs_ls {
        extract_gamma_1_2_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_ls_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g3_q, g4_q) = if let Some(coeffs_rs_buf) = coeffs_rs {
        extract_gamma_3_4_q_per_band_surround_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_rs_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let (g5_q, g6_q) = if let Some(coeffs_c_buf) = coeffs_c {
        extract_gamma_5_6_q_per_band_centre_least_squares(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            transform_length,
            acpl_num_bands,
            0,
            gamma_scale,
            acpl_qm1,
        )
    } else {
        (
            vec![0i32; acpl_num_bands as usize],
            vec![0i32; acpl_num_bands as usize],
        )
    };
    let beta3_q = if let Some(coeffs_c_buf) = coeffs_c {
        extract_beta3_q_per_band_centre_residual(
            coeffs_l,
            coeffs_r,
            coeffs_c_buf,
            &g1_q,
            &g2_q,
            &g3_q,
            &g4_q,
            &g5_q,
            &g6_q,
            transform_length,
            acpl_num_bands,
            0,
            beta3_scale,
            acpl_qm1,
            acpl_qm0,
        )
    } else {
        vec![0i32; acpl_num_bands as usize]
    };

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_3 (4) — 3 bits.
    bw.write_u32(4, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_2ch() (4 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_2ch(&mut bw, acpl_num_param_bands_id, acpl_qm0, acpl_qm1);
    }

    // LFE: mono_data(b_lfe=1) when present.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(2): sync=1, on=1, no avg.
    write_companding_control_2ch_sync_on(&mut bw);

    // stereo_data(): split-MDCT L/R carriers.
    write_stereo_split_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // Multi-envelope real aspx_data_2ch() + acpl_data_2ch() with real
    // α / β / β₃ / γ₁..γ₆. Present on every frame per Table 25; only
    // the 3-bit xover offset inside aspx_data_2ch() is I-frame-gated.
    {
        if write_aspx_data_2ch_multi_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            aspx_num_env,
            AspxMultiEnvelopeChannel {
                sig: &aspx_rows.ch0_sig,
                noise: &aspx_rows.ch0_noise,
            },
            AspxMultiEnvelopeChannel {
                sig: &aspx_rows.ch1_sig,
                noise: &aspx_rows.ch1_noise,
            },
            &[],
            aspx_l_ah,
            aspx_r_ah,
            b_iframe,
        )
        .is_err()
        {
            // Config / num_env rejected — signal the caller to fall back.
            return Vec::new();
        }
        write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3(
            &mut bw,
            acpl_num_bands,
            acpl_qm0,
            acpl_qm1,
            &alpha_q,
            &alpha_q,
            &beta1_q,
            &beta2_q,
            &beta3_q,
            &g1_q,
            &g2_q,
            &g3_q,
            &g4_q,
            &g5_q,
            &g6_q,
        );
    }

    finish_substream_body(bw)
}

/// Transpose a `[ts][sb]` QMF analysis matrix (the row-per-time-slot
/// shape [`crate::qmf::QmfAnalysisBank::process_block`] returns) into the
/// `[absolute_sb][ts]` column-per-subband shape the round-240 energy
/// aggregator ([`aggregate_qmf_to_sbg_atsg`] and its callers) consumes
/// on `q_high`.
///
/// `slots[ts][sb]` becomes `out[sb][ts]` for all `64` QMF subbands. The
/// result has `64` rows (one per absolute QMF subband) each of length
/// `slots.len()` (one per time slot). An empty input returns an empty
/// matrix.
pub fn qmf_slots_to_sb_major(slots: &[[(f32, f32); 64]]) -> Vec<Vec<(f32, f32)>> {
    if slots.is_empty() {
        return Vec::new();
    }
    let n_ts = slots.len();
    let mut out: Vec<Vec<(f32, f32)>> = vec![vec![(0.0, 0.0); n_ts]; 64];
    for (ts, slot) in slots.iter().enumerate() {
        for (sb, &val) in slot.iter().enumerate() {
            out[sb][ts] = val;
        }
    }
    out
}

/// Emit a `mono_data(b_lfe = 1)` element per Table 21. No leading
/// `spec_frontend` bit; `sf_info_lfe()` writes `max_sfb[0]` in
/// `n_msfbl_bits` bits (Table 106 column 4). Then a full `sf_data(ASF)`
/// body for the LFE channel.
fn write_lfe_mono_data(
    bw: &mut BitWriter,
    transform_length: u32,
    max_sfb_lfe: u32,
    coeffs_lfe: &[f32],
) {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let (_n_msfb_bits, _, n_msfbl_bits) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    assert!(
        n_msfbl_bits > 0,
        "encoder: LFE not permitted at transform_length = {transform_length}"
    );
    let n_msfbl_cap = (1u32 << n_msfbl_bits) - 1;
    let max_sfb_lfe_clamped = max_sfb_lfe.min(n_msfbl_cap);

    // asf_transform_info(): b_long_frame = 1 (LFE is always long-frame).
    bw.write_bit(true);
    // sf_info_lfe(): max_sfb[0] in n_msfbl_bits.
    bw.write_u32(max_sfb_lfe_clamped, n_msfbl_bits);
    // LFE sf_data(ASF): section + spectral + scalefac + snf.
    let (qspec, sf, max_q, sections, snf) =
        prepare_stereo_channel(coeffs_lfe, sfbo, max_sfb_lfe_clamped);
    write_section_data(bw, &sections);
    write_spectral_data_sections(bw, &qspec, sfbo, &sections);
    write_scalefac_data(bw, &sf, &sections.sfb_cb, &max_q, max_sfb_lfe_clamped);
    write_snf_data(
        bw,
        snf.as_deref(),
        &sections.sfb_cb,
        &max_q,
        max_sfb_lfe_clamped,
    );
}

// ====================================================================
// ASPX_ACPL_2 emitters — §4.2.6.6 Table 25 row `case ASPX_ACPL_2:`
// (round 100)
// ====================================================================

/// Emit an `acpl_config_1ch()` element in FULL mode per §4.2.13.1
/// Table 59: 2-bit `acpl_num_param_bands_id` + 1-bit `acpl_quant_mode`.
/// FULL mode carries no `acpl_qmf_band_minus1` field (that 3-bit field
/// is PARTIAL-only — used by ASPX_ACPL_1). Total: 3 bits. The decoder's
/// [`crate::acpl::parse_acpl_config_1ch`] with
/// [`crate::acpl::Acpl1chMode::Full`] reads exactly this ordering.
pub fn write_acpl_config_1ch_full(
    bw: &mut BitWriter,
    num_param_bands_id: u8,
    quant_mode: crate::acpl::AcplQuantMode,
) {
    bw.write_u32(num_param_bands_id as u32 & 0b11, 2);
    bw.write_bit(matches!(quant_mode, crate::acpl::AcplQuantMode::Coarse));
}

/// Emit a `two_channel_data()` body per §4.2.7.4 Table 26 for the
/// long-frame, single-window-group, identity-SAP case:
///
/// ```text
///   asf_transform_info(): b_long_frame = 1            // 1 b
///   asf_psy_info(ASF,0,0): max_sfb[0] in n_msfb_bits  // shared
///   chparam_info(): sap_mode = 0                       // 2 b
///   sf_data(ASF) ch0                                   // L carrier
///   sf_data(ASF) ch1                                   // R carrier
/// ```
///
/// Unlike `stereo_data()` (split-MDCT — a *per-channel* transform_info /
/// psy_info each), `two_channel_data()` shares one `sf_info(ASF)` header
/// across both channels then runs two `sf_data(ASF)` bodies. Mirrors
/// [`crate::mch::parse_two_channel_data`].
fn write_two_channel_data(
    bw: &mut BitWriter,
    transform_length: u32,
    max_sfb: u32,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
) {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let (n_msfb_bits, _, _) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    // Shared sf_info(ASF, 0, 0).
    bw.write_bit(true); // asf_transform_info: b_long_frame = 1
    bw.write_u32(max_sfb, n_msfb_bits); // asf_psy_info: max_sfb[0]
                                        // chparam_info(): sap_mode = 0 (identity SAP, no ms_used / sap_data).
    bw.write_u32(0, 2);

    // Two sf_data(ASF) bodies, one per channel.
    for coeffs in [coeffs_l, coeffs_r] {
        let (qspec, sf, max_q, sections, snf) = prepare_stereo_channel(coeffs, sfbo, max_sfb);
        write_section_data(bw, &sections);
        write_spectral_data_sections(bw, &qspec, sfbo, &sections);
        write_scalefac_data(bw, &sf, &sections.sfb_cb, &max_q, max_sfb);
        write_snf_data(bw, snf.as_deref(), &sections.sfb_cb, &max_q, max_sfb);
    }
}

/// Emit a non-LFE `mono_data(0)` element per Table 21:
///
/// ```text
///   spec_frontend = 0 (ASF)                           // 1 b
///   asf_transform_info(): b_long_frame = 1            // 1 b
///   asf_psy_info(ASF,0,0): max_sfb[0] in n_msfb_bits
///   sf_data(ASF)                                       // mono spectrum
/// ```
///
/// Mirrors [`crate::mch::parse_mono_data`] with `b_lfe = false`.
fn write_mono_data_centre(bw: &mut BitWriter, transform_length: u32, max_sfb: u32, coeffs: &[f32]) {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let (n_msfb_bits, _, _) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    // spec_frontend = 0 (ASF).
    bw.write_bit(false);
    // asf_transform_info(): b_long_frame = 1.
    bw.write_bit(true);
    // asf_psy_info(ASF, 0, 0): max_sfb[0].
    bw.write_u32(max_sfb, n_msfb_bits);
    // sf_data(ASF).
    let (qspec, sf, max_q, sections, snf) = prepare_stereo_channel(coeffs, sfbo, max_sfb);
    write_section_data(bw, &sections);
    write_spectral_data_sections(bw, &qspec, sfbo, &sections);
    write_scalefac_data(bw, &sf, &sections.sfb_cb, &max_q, max_sfb);
    write_snf_data(bw, snf.as_deref(), &sections.sfb_cb, &max_q, max_sfb);
}

// ====================================================================
// ASPX HF-generation / interleaved-waveform-coding writers
// (encoder duals of parse_aspx_hfgen_iwc_1ch / _2ch)
// ====================================================================

/// Caller-provided `aspx_hfgen_iwc_1ch()` payload — the real
/// inverse-filtering (`aspx_tna_mode`), additive-harmonic
/// (`aspx_add_harmonic`), frequency-interleaved-coding
/// (`aspx_fic_used_in_sfb`) and time-interleaved-coding
/// (`aspx_tic_used_in_slot`) decisions for a single channel.
///
/// Every slice is interpreted positionally against the derived
/// subband-group / time-slot counts:
///
/// * `tna_mode` — `num_sbg_noise` entries, each a 2-bit inverse-
///   filtering mode in `0..=3`. Missing entries default to `0`; values
///   above `3` are masked to their low 2 bits (matching the decoder's
///   `read_u32(2)`).
/// * `add_harmonic` — `num_sbg_sig_highres` booleans. The presence bit
///   `aspx_ah_present` is emitted as `1` iff any entry is `true`; only
///   then is the per-SBG vector written. An all-`false` slice (or an
///   empty one) emits `aspx_ah_present = 0` and no per-SBG bits, which
///   the decoder reconstructs as an all-`false` vector.
/// * `fic_used_in_sfb` — `num_sbg_sig_highres` booleans, gated by
///   `aspx_fic_present` with the same any-true rule.
/// * `tic_used_in_slot` — `num_aspx_timeslots` booleans, gated by
///   `aspx_tic_present` with the same any-true rule.
#[derive(Debug, Clone, Default)]
pub struct AspxHfgenIwc1ChPayload<'a> {
    /// Per-noise-SBG 2-bit inverse-filtering modes (`0..=3`).
    pub tna_mode: &'a [u8],
    /// Per-high-res-signal-SBG additive-harmonic flags.
    pub add_harmonic: &'a [bool],
    /// Per-high-res-signal-SBG frequency-interleaved-coding flags.
    pub fic_used_in_sfb: &'a [bool],
    /// Per-A-SPX-time-slot time-interleaved-coding flags.
    pub tic_used_in_slot: &'a [bool],
}

/// Emit a positional run of `n` booleans, reading from `src` and
/// zero-(false-)padding past its end.
fn write_bool_run(bw: &mut BitWriter, src: &[bool], n: u32) {
    for i in 0..n as usize {
        bw.write_bit(src.get(i).copied().unwrap_or(false));
    }
}

/// Returns `true` iff any of the first `n` entries of `src` is set
/// (entries past `src.len()` count as `false`). Drives the encoder's
/// `*_present` gate decision so a payload with no active flags emits the
/// compact `present = 0` form.
fn any_set(src: &[bool], n: u32) -> bool {
    src.iter().take(n as usize).any(|&b| b)
}

/// Write `aspx_hfgen_iwc_1ch()` per ETSI TS 103 190-1 §4.2.12.6
/// (Table 55) — the exact encoder dual of
/// [`crate::aspx::parse_aspx_hfgen_iwc_1ch`].
///
/// Bit layout (matching the parser):
///
/// ```text
///   for n in 0..num_sbg_noise:        aspx_tna_mode[n]        // 2 b
///   aspx_ah_present                                           // 1 b
///   if aspx_ah_present:
///     for n in 0..num_sbg_sig_highres: aspx_add_harmonic[n]   // 1 b
///   aspx_fic_present                                          // 1 b
///   if aspx_fic_present:
///     for n in 0..num_sbg_sig_highres: aspx_fic_used_in_sfb[n] // 1 b
///   aspx_tic_present                                          // 1 b
///   if aspx_tic_present:
///     for n in 0..num_aspx_timeslots: aspx_tic_used_in_slot[n] // 1 b
/// ```
///
/// The three `*_present` gates are set automatically from the payload:
/// each is `1` iff the corresponding slice has at least one active flag
/// in range. Re-parsing the emitted bits with
/// `parse_aspx_hfgen_iwc_1ch` recovers an [`crate::aspx::AspxHfgenIwc1Ch`]
/// whose `tna_mode` equals the (masked, padded) input and whose
/// `add_harmonic` / `fic_used_in_sfb` / `tic_used_in_slot` vectors equal
/// the (padded) inputs.
pub fn write_aspx_hfgen_iwc_1ch(
    bw: &mut BitWriter,
    payload: &AspxHfgenIwc1ChPayload<'_>,
    num_sbg_noise: u32,
    num_sbg_sig_highres: u32,
    num_aspx_timeslots: u32,
) {
    // aspx_tna_mode[0..num_sbg_noise] — 2 b each, low 2 bits.
    for i in 0..num_sbg_noise as usize {
        let mode = payload.tna_mode.get(i).copied().unwrap_or(0);
        bw.write_u32((mode & 0x3) as u32, 2);
    }
    // aspx_ah_present + optional per-SBG add_harmonic vector.
    let ah_present = any_set(payload.add_harmonic, num_sbg_sig_highres);
    bw.write_bit(ah_present);
    if ah_present {
        write_bool_run(bw, payload.add_harmonic, num_sbg_sig_highres);
    }
    // aspx_fic_present + optional per-SBG fic_used_in_sfb vector.
    let fic_present = any_set(payload.fic_used_in_sfb, num_sbg_sig_highres);
    bw.write_bit(fic_present);
    if fic_present {
        write_bool_run(bw, payload.fic_used_in_sfb, num_sbg_sig_highres);
    }
    // aspx_tic_present + optional per-timeslot tic_used_in_slot vector.
    let tic_present = any_set(payload.tic_used_in_slot, num_aspx_timeslots);
    bw.write_bit(tic_present);
    if tic_present {
        write_bool_run(bw, payload.tic_used_in_slot, num_aspx_timeslots);
    }
}

/// Caller-provided `aspx_hfgen_iwc_2ch()` payload — the stereo dual of
/// [`AspxHfgenIwc1ChPayload`]. Index `[0]` is the left channel, `[1]`
/// the right.
///
/// `aspx_balance` selects how the right channel's inverse-filtering is
/// signalled:
///
/// * `aspx_balance == false` — both `tna_mode[0]` and `tna_mode[1]` are
///   written explicitly (2 b per SBG per channel).
/// * `aspx_balance == true` — only `tna_mode[0]` is written; the decoder
///   mirrors it into channel 1. The caller's `tna_mode[1]` is ignored in
///   that case (the round-trip yields `tna_mode[1] == tna_mode[0]`).
///
/// `add_harmonic` is gated per channel (`aspx_ah_left` / `aspx_ah_right`,
/// each auto-set from the channel's slice). `fic_used_in_sfb` is gated by
/// an outer `aspx_fic_present` plus per-channel `aspx_fic_left` /
/// `aspx_fic_right`. `tic_used_in_slot` is gated by `aspx_tic_present`;
/// when both channels carry the identical pattern the writer emits the
/// compact `aspx_tic_copy = 1` form, otherwise per-channel
/// `aspx_tic_left` / `aspx_tic_right` gates.
#[derive(Debug, Clone, Default)]
pub struct AspxHfgenIwc2ChPayload<'a> {
    /// Per-channel per-noise-SBG 2-bit inverse-filtering modes.
    pub tna_mode: [&'a [u8]; 2],
    /// Per-channel per-high-res-signal-SBG additive-harmonic flags.
    pub add_harmonic: [&'a [bool]; 2],
    /// Per-channel per-high-res-signal-SBG frequency-interleaved-coding
    /// flags.
    pub fic_used_in_sfb: [&'a [bool]; 2],
    /// Per-channel per-A-SPX-time-slot time-interleaved-coding flags.
    pub tic_used_in_slot: [&'a [bool]; 2],
}

/// Write `aspx_hfgen_iwc_2ch(aspx_balance)` per ETSI TS 103 190-1
/// §4.2.12.7 (Table 56) — the exact encoder dual of
/// [`crate::aspx::parse_aspx_hfgen_iwc_2ch`].
///
/// The per-channel `add_harmonic` gates (`aspx_ah_left` /
/// `aspx_ah_right`), the outer `aspx_fic_present` + per-channel
/// `aspx_fic_left` / `aspx_fic_right` gates, and the `aspx_tic_present` +
/// `aspx_tic_copy` / `aspx_tic_left` / `aspx_tic_right` gates are all set
/// automatically from the payload (a gate is `1` iff its slice has an
/// active flag in range). When both channels' `tic_used_in_slot`
/// patterns are equal **and** at least one slot is active, the compact
/// `aspx_tic_copy = 1` encoding is used.
///
/// Re-parsing the emitted bits with `parse_aspx_hfgen_iwc_2ch` (same
/// `aspx_balance`, same counts) recovers an
/// [`crate::aspx::AspxHfgenIwc2Ch`] whose `tna_mode` /
/// `add_harmonic` / `fic_used_in_sfb` / `tic_used_in_slot` reproduce the
/// (masked, padded) inputs.
pub fn write_aspx_hfgen_iwc_2ch(
    bw: &mut BitWriter,
    payload: &AspxHfgenIwc2ChPayload<'_>,
    aspx_balance: bool,
    num_sbg_noise: u32,
    num_sbg_sig_highres: u32,
    num_aspx_timeslots: u32,
) {
    // aspx_tna_mode[0][..] always written.
    for i in 0..num_sbg_noise as usize {
        let mode = payload.tna_mode[0].get(i).copied().unwrap_or(0);
        bw.write_u32((mode & 0x3) as u32, 2);
    }
    // aspx_tna_mode[1][..] only when balance == 0; else mirrors ch0.
    if !aspx_balance {
        for i in 0..num_sbg_noise as usize {
            let mode = payload.tna_mode[1].get(i).copied().unwrap_or(0);
            bw.write_u32((mode & 0x3) as u32, 2);
        }
    }
    // Per-channel additive-harmonic gates + vectors.
    let ah_left = any_set(payload.add_harmonic[0], num_sbg_sig_highres);
    bw.write_bit(ah_left);
    if ah_left {
        write_bool_run(bw, payload.add_harmonic[0], num_sbg_sig_highres);
    }
    let ah_right = any_set(payload.add_harmonic[1], num_sbg_sig_highres);
    bw.write_bit(ah_right);
    if ah_right {
        write_bool_run(bw, payload.add_harmonic[1], num_sbg_sig_highres);
    }
    // Frequency-interleaved-coding: outer present gate, then per-channel
    // left/right gates.
    let fic_left = any_set(payload.fic_used_in_sfb[0], num_sbg_sig_highres);
    let fic_right = any_set(payload.fic_used_in_sfb[1], num_sbg_sig_highres);
    let fic_present = fic_left || fic_right;
    bw.write_bit(fic_present);
    if fic_present {
        bw.write_bit(fic_left);
        if fic_left {
            write_bool_run(bw, payload.fic_used_in_sfb[0], num_sbg_sig_highres);
        }
        bw.write_bit(fic_right);
        if fic_right {
            write_bool_run(bw, payload.fic_used_in_sfb[1], num_sbg_sig_highres);
        }
    }
    // Time-interleaved-coding: outer present gate, then copy / per-channel
    // gates.
    let tic_left_active = any_set(payload.tic_used_in_slot[0], num_aspx_timeslots);
    let tic_right_active = any_set(payload.tic_used_in_slot[1], num_aspx_timeslots);
    let tic_present = tic_left_active || tic_right_active;
    bw.write_bit(tic_present);
    if tic_present {
        // Use the compact copy form when both channels carry the same
        // active pattern.
        let patterns_equal = (0..num_aspx_timeslots as usize).all(|i| {
            payload.tic_used_in_slot[0].get(i).copied().unwrap_or(false)
                == payload.tic_used_in_slot[1].get(i).copied().unwrap_or(false)
        });
        let tic_copy = patterns_equal && tic_left_active;
        bw.write_bit(tic_copy);
        if !tic_copy {
            bw.write_bit(tic_left_active);
            bw.write_bit(tic_right_active);
        }
        if tic_copy || tic_left_active {
            write_bool_run(bw, payload.tic_used_in_slot[0], num_aspx_timeslots);
        }
        if !tic_copy && tic_right_active {
            write_bool_run(bw, payload.tic_used_in_slot[1], num_aspx_timeslots);
        }
    }
}

/// Emit a minimum-viable `aspx_data_1ch()` body per §4.2.12.3 Table 51
/// with the FIXFIX + num_env = 1 path:
///
/// ```text
///   aspx_xover_subband_offset = 0                      // 3 b
///   aspx_framing(0): FIXFIX, tmp_num_env = 0           // 2 + envbits b
///   aspx_delta_dir(0): 1 SIGNAL + 1 NOISE bit (FREQ)   // 2 b
///   aspx_hfgen_iwc_1ch(): num_sbg_noise × 2 b tna_mode + 3 × present=0
///   aspx_ec_data(SIGNAL): F0 + (num_sbg_sig - 1) × DF
///   aspx_ec_data(NOISE):  F0 + (num_sbg_noise - 1) × DF
/// ```
///
/// All stereo_mode = LEVEL (mono — there is no balance dimension). The
/// SIGNAL band count is derived per [`parse_aspx_ec_data`]: when the
/// `aspx_config` does not signal per-envelope frequency resolution
/// (`freq_res_mode != Signalled`), the framing carries no
/// `aspx_freq_res[0]` bit, so the parser's `freq_res` vector is empty
/// and the SIGNAL ec_data falls back to the **high-res** subband count.
/// We therefore drive the writer with `num_sbg_sig_highres`.
fn write_aspx_data_1ch_minimal(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
) -> Result<(), &'static str> {
    write_aspx_data_1ch_minimal_framed(bw, cfg, true)
}

/// [`write_aspx_data_1ch_minimal`] with an explicit `b_iframe` — Tables 51/52 gate the
/// leading 3-bit xover offset on I-frames.
pub(crate) fn write_aspx_data_1ch_minimal_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
) -> Result<(), &'static str> {
    write_aspx_data_1ch_minimal_framed_xover(bw, cfg, b_iframe, 0)
}

/// [`write_aspx_data_1ch_minimal_framed`] with an explicit
/// `aspx_xover_subband_offset`.
pub(crate) fn write_aspx_data_1ch_minimal_framed_xover(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
    xover: u32,
) -> Result<(), &'static str> {
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    // aspx_framing(0): FIXFIX (int_class prefix '0', 1 b per Table 126),
    // tmp_num_env = 0.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits); // tmp_num_env = 0 → num_env = 1
    if cfg.signals_freq_res() {
        bw.write_bit(false); // aspx_freq_res[0] = 0
    }
    // aspx_delta_dir(0): num_env = 1 SIGNAL bit + num_noise = 1 NOISE bit.
    bw.write_bit(false); // sig_delta_dir[0] = false (FREQ)
    bw.write_bit(false); // noise_delta_dir[0] = false (FREQ)

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_1ch(): all-zero payload — tna_mode[0..num_sbg_noise]
    // = 0 (2 b each) + ah_present / fic_present / tic_present = 0 (3 × 1 b).
    // Routed through the real writer with a default (empty) payload so the
    // gates auto-collapse to the compact form. `num_sbg_sig_highres` /
    // `num_aspx_timeslots` are irrelevant when all gates are 0.
    write_aspx_hfgen_iwc_1ch(
        bw,
        &AspxHfgenIwc1ChPayload::default(),
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    );

    // num_env = 1 with empty freq_res → SIGNAL ec_data reads the high-res
    // subband count (see doc comment).
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;

    // SIGNAL ec_data (LEVEL). FIXFIX + num_env == 1 → qmode forced Fine.
    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };
    if num_sbg_sig >= 1 {
        write_aspx_sig_f0(bw, qmode_sig, aspx::AspxStereoMode::Level);
    }
    for _ in 1..num_sbg_sig {
        write_aspx_sig_df_zero(bw, qmode_sig, aspx::AspxStereoMode::Level);
    }
    // NOISE ec_data (LEVEL, qmode Fine per Table 51).
    if num_sbg_noise >= 1 {
        write_aspx_noise_f0(bw, aspx::AspxStereoMode::Level);
    }
    for _ in 1..num_sbg_noise {
        write_aspx_noise_df_zero(bw, aspx::AspxStereoMode::Level);
    }
    Ok(())
}

// ====================================================================
// ASPX real-envelope writers (round 226)
// ====================================================================

/// Per-channel envelope quant-index payload consumed by
/// [`write_aspx_data_2ch_real_envelope`] / [`write_aspx_data_1ch_real_envelope`].
///
/// Each slice carries the F0 codeword for parameter band 0 followed by
/// `(num_sbg − 1)` signed DF deltas. The decoder's
/// [`crate::aspx::parse_aspx_ec_data`] FREQ branch recovers the same
/// vector — F0 directly via `decode_delta()` on the F0 codebook and the
/// trailing entries as signed `symbol_index − cb_off` deltas on the DF
/// codebook.
///
/// `sig` is the SIGNAL envelope (`aspx_ec_data(SIGNAL, …)`); `noise` is
/// the NOISE envelope (`aspx_ec_data(NOISE, …)`). Slice lengths
/// shorter than the SBG count are zero-padded; entries past the SBG
/// count are ignored. F0 values outside `[0, codebook_length)` clamp
/// to the codebook's extreme entries; DF values outside `[-cb_off,
/// +cb_off]` saturate to the symmetric edge.
#[derive(Debug, Clone, Copy, Default)]
pub struct AspxRealEnvelopeChannel<'a> {
    /// SIGNAL envelope quant indices for this channel — `[F0, DF1, DF2, …]`
    /// (length should match `num_sbg_sig` for the active framing).
    pub sig: &'a [i32],
    /// NOISE envelope quant indices for this channel — `[F0, DF1, DF2, …]`
    /// (length should match `num_sbg_noise`).
    pub noise: &'a [i32],
}

/// Emit an `aspx_data_2ch()` body per ETSI TS 103 190-1 §4.2.12.4
/// Table 52 with caller-provided real envelope quant indices, mirroring
/// the [`write_aspx_data_2ch_minimal`] framing skeleton:
///
/// * `aspx_xover_subband_offset = 0` (3 b).
/// * `aspx_framing(0)`: FIXFIX prefix `0` (1 b), `tmp_num_env = 0` →
///   `num_env = 1`, optional `aspx_freq_res[0] = 0` (low-res selection
///   only when `cfg.signals_freq_res()`).
/// * `aspx_balance = 1` (1 b) — channel 1 reuses channel 0's framing.
/// * `aspx_delta_dir(0)` + `aspx_delta_dir(1)`: SIGNAL + NOISE
///   directions both 0 (FREQ).
/// * `aspx_hfgen_iwc_2ch(balance = 1)`: per-SBG `tna_mode = 0` (2 b
///   each), `ah_left = ah_right = fic_present = tic_present = 0`.
/// * SIGNAL ec_data per channel (Level for ch0, Balance for ch1):
///   one F0 codeword + `(num_sbg_sig − 1)` DF codewords routed
///   through [`write_aspx_sig_f0_value`] / [`write_aspx_sig_df_value`].
/// * NOISE ec_data per channel (Level for ch0, Balance for ch1):
///   one F0 codeword + `(num_sbg_noise − 1)` DF codewords routed
///   through [`write_aspx_noise_f0_value`] / [`write_aspx_noise_df_value`].
///
/// The body round-trips through [`crate::aspx::parse_aspx_ec_data`] —
/// each per-channel SIGNAL / NOISE envelope decodes to a vector whose
/// position-0 entry equals the caller's F0 input (clamped to the
/// codebook range) and whose trailing entries equal the caller's DF
/// inputs (clamped to the codebook's symmetric ±cb_off range).
///
/// The existing [`write_aspx_data_2ch_minimal`] writer is preserved
/// unchanged; this real-envelope variant is a strict superset.
pub fn write_aspx_data_2ch_real_envelope(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch0: AspxRealEnvelopeChannel<'_>,
    ch1: AspxRealEnvelopeChannel<'_>,
) -> Result<(), &'static str> {
    let xover: u32 = 0;
    bw.write_u32(xover, 3);

    // aspx_framing(0): FIXFIX prefix `0` (1 b per Table 126),
    // tmp_num_env = 0 → num_env = 1.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits);
    if cfg.signals_freq_res() {
        bw.write_bit(false); // aspx_freq_res[0] = 0 (low-res)
    }
    // aspx_balance = 1 → channel-1 reuses channel-0's framing.
    bw.write_bit(true);
    // aspx_delta_dir(0) + aspx_delta_dir(1): SIGNAL + NOISE both FREQ.
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_2ch(balance = 1): tna_mode[0][..num_sbg_noise]
    // = 0 (2 b each); ah_left = ah_right = fic_present = tic_present = 0.
    for _ in 0..counts.num_sbg_noise {
        bw.write_u32(0, 2);
    }
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);

    // SIGNAL band count: low-res when freq_res was emitted as 0,
    // else high-res — matches parse_aspx_ec_data's
    // `freq_res.get(env).copied().unwrap_or(true)` fallback.
    let num_sbg_sig = if cfg.signals_freq_res() {
        counts.num_sbg_sig_lowres
    } else {
        counts.num_sbg_sig_highres
    };
    let num_sbg_noise = counts.num_sbg_noise;

    // Per Table 52: FIXFIX + num_env == 1 → qmode forced to Fine.
    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };

    // §5.7.6.3.5: with aspx_balance = 1 the pair is jointly coded —
    // convert the caller's per-channel (L, R) LEVEL rows into the
    // (sum, pan) wire pair (the Pseudocode 84 inverse; the decoder's
    // joint dequantization recovers each channel's level).
    let (ch0_sig_w, ch1_sig_w, ch0_noise_w, ch1_noise_w) = balance_convert_packed_rows(
        ch0.sig,
        ch1.sig,
        ch0.noise,
        ch1.noise,
        qmode_sig,
        num_sbg_sig as usize,
        num_sbg_noise as usize,
    );
    // ch0 SIGNAL (sum): stereo_mode = LEVEL.
    write_aspx_sig_envelope_values(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        &ch0_sig_w,
        num_sbg_sig,
    );
    // ch1 SIGNAL (pan): stereo_mode = BALANCE (aspx_balance = 1).
    write_aspx_sig_envelope_values(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Balance,
        &ch1_sig_w,
        num_sbg_sig,
    );

    // ch0 NOISE (sum): stereo_mode = LEVEL. Per Table 52 NOISE qmode = Fine.
    write_aspx_noise_envelope_values(bw, aspx::AspxStereoMode::Level, &ch0_noise_w, num_sbg_noise);
    // ch1 NOISE (pan): stereo_mode = BALANCE.
    write_aspx_noise_envelope_values(
        bw,
        aspx::AspxStereoMode::Balance,
        &ch1_noise_w,
        num_sbg_noise,
    );
    Ok(())
}

/// Shared §5.7.6.3.5-inverse conversion for the single-envelope
/// `aspx_data_2ch()` FREQ writers: unpack both channels' packed
/// `[F0, DF₁, …]` LEVEL rows to absolute quant rows (constant-tail
/// extension to the band count), apply the Pseudocode 84 inverse
/// ([`balance_encode_sig_rows`] / [`balance_encode_noise_rows`]), and
/// repack. Returns `(ch0_sig, ch1_sig, ch0_noise, ch1_noise)` wire
/// rows — sum on channel 0 (LEVEL codebooks), pan wire steps on
/// channel 1 (BALANCE codebooks, `delta = 2` accumulation on the
/// decoder side).
pub(crate) fn balance_convert_packed_rows(
    ch0_sig: &[i32],
    ch1_sig: &[i32],
    ch0_noise: &[i32],
    ch1_noise: &[i32],
    qmode_sig: aspx::AspxQuantStep,
    num_sbg_sig: usize,
    num_sbg_noise: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let l_sig = qscf_row_from_freq_dpcm_extended(ch0_sig, num_sbg_sig);
    let r_sig = qscf_row_from_freq_dpcm_extended(ch1_sig, num_sbg_sig);
    let (sum_sig, pan_sig) = balance_encode_sig_rows(&l_sig, &r_sig, qmode_sig);
    let l_noise = qscf_row_from_freq_dpcm_extended(ch0_noise, num_sbg_noise);
    let r_noise = qscf_row_from_freq_dpcm_extended(ch1_noise, num_sbg_noise);
    let (sum_noise, pan_noise) = balance_encode_noise_rows(&l_noise, &r_noise);
    (
        freq_dpcm_encode_qscf(&sum_sig),
        freq_dpcm_encode_qscf(&pan_sig),
        freq_dpcm_encode_qscf(&sum_noise),
        freq_dpcm_encode_qscf(&pan_noise),
    )
}

/// `aspx_data_2ch()` real-envelope writer that additionally carries a
/// real channel-0 `aspx_tna_mode` inverse-filtering vector.
///
/// Identical to [`write_aspx_data_2ch_real_envelope`] except the
/// `aspx_hfgen_iwc_2ch(balance = 1)` element is routed through
/// [`write_aspx_hfgen_iwc_2ch`] with the caller-supplied `tna_mode`
/// payload on channel 0. Because `aspx_balance = 1` is used (channel 1
/// mirrors channel 0's framing), the decoder copies channel 0's
/// `tna_mode` into channel 1, so only one vector is signalled. The
/// `add_harmonic` / `fic_used_in_sfb` / `tic_used_in_slot` gates stay 0.
///
/// Passing an all-zero (or empty) `tna_mode` slice reproduces the
/// [`write_aspx_data_2ch_real_envelope`] bytes exactly.
pub fn write_aspx_data_2ch_real_envelope_tna(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch0: AspxRealEnvelopeChannel<'_>,
    ch1: AspxRealEnvelopeChannel<'_>,
    tna_mode: &[u8],
) -> Result<(), &'static str> {
    write_aspx_data_2ch_real_envelope_tna_ah(bw, cfg, ch0, ch1, tna_mode, &[], &[])
}

/// `aspx_data_2ch()` real-envelope + `aspx_tna_mode` writer that
/// additionally carries a real per-channel `aspx_add_harmonic` vector.
///
/// Identical to [`write_aspx_data_2ch_real_envelope_tna`] except the
/// `aspx_hfgen_iwc_2ch(balance = 1)` element's per-channel
/// `add_harmonic[0]` / `add_harmonic[1]` slices are populated from
/// `ah_left` / `ah_right` (each a per-high-res-signal-subband-group
/// boolean vector, typically from [`crate::aspx_ah_select::select_add_harmonic`]).
/// The writer's `aspx_ah_left` / `aspx_ah_right` gates auto-collapse to 0
/// when a slice has no active flag, so passing two empty slices reproduces
/// the [`write_aspx_data_2ch_real_envelope_tna`] bytes exactly.
///
/// `fic_used_in_sfb` / `tic_used_in_slot` stay at the all-zero scaffold.
#[allow(clippy::too_many_arguments)]
pub fn write_aspx_data_2ch_real_envelope_tna_ah(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch0: AspxRealEnvelopeChannel<'_>,
    ch1: AspxRealEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah_left: &[bool],
    ah_right: &[bool],
) -> Result<(), &'static str> {
    write_aspx_data_2ch_real_envelope_tna_ah_framed(
        bw, cfg, ch0, ch1, tna_mode, ah_left, ah_right, true,
    )
}

/// [`write_aspx_data_2ch_real_envelope_tna_ah`] with an explicit
/// `b_iframe` — per ETSI TS 103 190-1 Table 52 the leading 3-bit
/// `aspx_xover_subband_offset` is only present on I-frames; P-frame
/// (`b_iframe = false`) bodies start directly at `aspx_framing(0)` and
/// the decoder reuses the I-frame-sticky xover (always 0 for this
/// writer family).
#[allow(clippy::too_many_arguments)]
pub fn write_aspx_data_2ch_real_envelope_tna_ah_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch0: AspxRealEnvelopeChannel<'_>,
    ch1: AspxRealEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah_left: &[bool],
    ah_right: &[bool],
    b_iframe: bool,
) -> Result<(), &'static str> {
    let xover: u32 = 0;
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits);
    if cfg.signals_freq_res() {
        bw.write_bit(false);
    }
    // aspx_balance = 1 → channel-1 reuses channel-0's framing + tna_mode.
    bw.write_bit(true);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_2ch(balance = 1): real tna_mode[0] (mirrored to [1])
    // plus per-channel add_harmonic vectors.
    let empty: &[u8] = &[];
    let payload = AspxHfgenIwc2ChPayload {
        tna_mode: [tna_mode, empty],
        add_harmonic: [ah_left, ah_right],
        ..AspxHfgenIwc2ChPayload::default()
    };
    write_aspx_hfgen_iwc_2ch(
        bw,
        &payload,
        true, // aspx_balance = 1
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    );

    let num_sbg_sig = if cfg.signals_freq_res() {
        counts.num_sbg_sig_lowres
    } else {
        counts.num_sbg_sig_highres
    };
    let num_sbg_noise = counts.num_sbg_noise;

    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };

    // §5.7.6.3.5 joint (sum, pan) conversion — see
    // [`balance_convert_packed_rows`].
    let (ch0_sig_w, ch1_sig_w, ch0_noise_w, ch1_noise_w) = balance_convert_packed_rows(
        ch0.sig,
        ch1.sig,
        ch0.noise,
        ch1.noise,
        qmode_sig,
        num_sbg_sig as usize,
        num_sbg_noise as usize,
    );
    write_aspx_sig_envelope_values(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        &ch0_sig_w,
        num_sbg_sig,
    );
    write_aspx_sig_envelope_values(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Balance,
        &ch1_sig_w,
        num_sbg_sig,
    );
    write_aspx_noise_envelope_values(bw, aspx::AspxStereoMode::Level, &ch0_noise_w, num_sbg_noise);
    write_aspx_noise_envelope_values(
        bw,
        aspx::AspxStereoMode::Balance,
        &ch1_noise_w,
        num_sbg_noise,
    );
    Ok(())
}

/// Single-envelope `aspx_data_2ch()` writer with **per-channel,
/// per-data-type transmission directions** (ETSI TS 103 190-1
/// §4.2.12.5 Table 54 `aspx_delta_dir()` + §4.2.12.9 Table 58) — the
/// P-frame generalisation of
/// [`write_aspx_data_2ch_real_envelope_tna_ah_framed`].
///
/// Each of the four envelope payloads (`ch0_sig`, `ch0_noise`,
/// `ch1_sig`, `ch1_noise`) is an [`AspxEncodedEnvelope`] carrying its
/// chosen FREQ / TIME direction:
///
/// * FREQ rows hold `[F0, DF₁, …]` (frequency-DPCM, decoded by the
///   Pseudocode 80/81 FREQ branch);
/// * TIME rows hold per-subband-group `DT` deltas against the
///   **previous frame's last envelope** (`qscf_prev` — the decoder's
///   [`crate::aspx::AspxEnvPrev`] carries it across frames).
///
/// With all-FREQ rows this reproduces
/// [`write_aspx_data_2ch_real_envelope_tna_ah_framed`] byte-for-byte.
/// Framing is FIXFIX `num_env = 1`, `aspx_balance = 1` (ch1 BALANCE
/// coded, framing + tna mirrored from ch0), matching the live 5_X
/// ACPL_3 path.
#[allow(clippy::too_many_arguments)]
pub fn write_aspx_data_2ch_directional_envelope_tna_ah_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch0_sig: &AspxEncodedEnvelope,
    ch0_noise: &AspxEncodedEnvelope,
    ch1_sig: &AspxEncodedEnvelope,
    ch1_noise: &AspxEncodedEnvelope,
    tna_mode: &[u8],
    ah_left: &[bool],
    ah_right: &[bool],
    b_iframe: bool,
) -> Result<(), &'static str> {
    let xover: u32 = 0;
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits);
    if cfg.signals_freq_res() {
        bw.write_bit(false);
    }
    // aspx_balance = 1 → channel-1 reuses channel-0's framing + tna_mode.
    bw.write_bit(true);
    // aspx_delta_dir(0) + aspx_delta_dir(1): 1 sig + 1 noise bit each
    // (num_env = 1, num_noise = 1) per Table 54.
    bw.write_bit(ch0_sig.direction_time);
    bw.write_bit(ch0_noise.direction_time);
    bw.write_bit(ch1_sig.direction_time);
    bw.write_bit(ch1_noise.direction_time);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    let empty: &[u8] = &[];
    let payload = AspxHfgenIwc2ChPayload {
        tna_mode: [tna_mode, empty],
        add_harmonic: [ah_left, ah_right],
        ..AspxHfgenIwc2ChPayload::default()
    };
    write_aspx_hfgen_iwc_2ch(
        bw,
        &payload,
        true, // aspx_balance = 1
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    );

    let num_sbg_sig = if cfg.signals_freq_res() {
        counts.num_sbg_sig_lowres
    } else {
        counts.num_sbg_sig_highres
    };
    let num_sbg_noise = counts.num_sbg_noise;

    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };

    write_aspx_sig_envelope_directional(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        ch0_sig,
        num_sbg_sig,
    );
    write_aspx_sig_envelope_directional(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Balance,
        ch1_sig,
        num_sbg_sig,
    );
    write_aspx_noise_envelope_directional(
        bw,
        aspx::AspxStereoMode::Level,
        ch0_noise,
        num_sbg_noise,
    );
    write_aspx_noise_envelope_directional(
        bw,
        aspx::AspxStereoMode::Balance,
        ch1_noise,
        num_sbg_noise,
    );
    Ok(())
}

/// Emit an `aspx_data_1ch()` body per ETSI TS 103 190-1 §4.2.12.3
/// Table 51 with caller-provided real envelope quant indices, mirroring
/// the [`write_aspx_data_1ch_minimal`] framing skeleton:
///
/// * `aspx_xover_subband_offset = 0` (3 b).
/// * `aspx_framing(0)`: FIXFIX prefix `0` (1 b), `tmp_num_env = 0` →
///   `num_env = 1`, optional `aspx_freq_res[0] = 0`.
/// * `aspx_delta_dir(0)`: SIGNAL + NOISE both FREQ (2 × 1 b).
/// * `aspx_hfgen_iwc_1ch()`: per-SBG `tna_mode = 0` (2 b each),
///   `ah_present = fic_present = tic_present = 0`.
/// * SIGNAL ec_data (Level): F0 + `(num_sbg_sig_highres − 1)` DF.
/// * NOISE ec_data (Level): F0 + `(num_sbg_noise − 1)` DF.
///
/// The single-channel path uses the high-res SIGNAL band count (no
/// in-band `aspx_freq_res` bit when `signals_freq_res()` is false —
/// the parser's fallback selects high-res). When
/// `cfg.signals_freq_res()` is true the writer emits an
/// `aspx_freq_res[0] = 0` bit and uses the low-res count.
pub fn write_aspx_data_1ch_real_envelope(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch: AspxRealEnvelopeChannel<'_>,
) -> Result<(), &'static str> {
    let xover: u32 = 0;
    bw.write_u32(xover, 3);

    // aspx_framing(0): FIXFIX, num_env = 1.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits);
    if cfg.signals_freq_res() {
        bw.write_bit(false);
    }
    // aspx_delta_dir(0): SIGNAL + NOISE both FREQ.
    bw.write_bit(false);
    bw.write_bit(false);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_1ch(): tna_mode×num_sbg_noise + 3×1 b trailer.
    for _ in 0..counts.num_sbg_noise {
        bw.write_u32(0, 2);
    }
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);

    let num_sbg_sig = if cfg.signals_freq_res() {
        counts.num_sbg_sig_lowres
    } else {
        counts.num_sbg_sig_highres
    };
    let num_sbg_noise = counts.num_sbg_noise;

    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };

    // SIGNAL ec_data (LEVEL).
    write_aspx_sig_envelope_values(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        ch.sig,
        num_sbg_sig,
    );
    // NOISE ec_data (LEVEL, qmode Fine per Table 51).
    write_aspx_noise_envelope_values(bw, aspx::AspxStereoMode::Level, ch.noise, num_sbg_noise);
    Ok(())
}

/// `aspx_data_1ch()` real-envelope writer that additionally carries a
/// real per-noise-subband-group `aspx_tna_mode` inverse-filtering vector.
///
/// Identical to [`write_aspx_data_1ch_real_envelope`] except the
/// `aspx_hfgen_iwc_1ch()` element is routed through
/// [`write_aspx_hfgen_iwc_1ch`] with the caller-supplied `tna_mode`
/// payload instead of the all-zero scaffold. The `add_harmonic` /
/// `fic_used_in_sfb` / `tic_used_in_slot` flags remain absent (the
/// `*_present` gates auto-collapse to 0 for the default empty slices), so
/// the only wire difference from the scaffold is the 2-bit-per-SBG
/// `tna_mode` field — which the decoder's `parse_aspx_hfgen_iwc_1ch`
/// recovers verbatim and feeds into the §5.7.6.4.1.3 chirp pipeline.
///
/// Passing an all-zero (or empty) `tna_mode` slice reproduces the
/// [`write_aspx_data_1ch_real_envelope`] bytes exactly.
pub fn write_aspx_data_1ch_real_envelope_tna(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch: AspxRealEnvelopeChannel<'_>,
    tna_mode: &[u8],
) -> Result<(), &'static str> {
    write_aspx_data_1ch_real_envelope_tna_ah(bw, cfg, ch, tna_mode, &[])
}

/// `aspx_data_1ch()` real-envelope + `aspx_tna_mode` writer that
/// additionally carries a real `aspx_add_harmonic` vector.
///
/// Identical to [`write_aspx_data_1ch_real_envelope_tna`] except the
/// `aspx_hfgen_iwc_1ch()` element's `add_harmonic` slice is populated from
/// `ah` (a per-high-res-signal-subband-group boolean vector, typically
/// from [`crate::aspx_ah_select::select_add_harmonic`]). The writer's
/// `aspx_ah_present` gate auto-collapses to 0 when the slice has no active
/// flag, so passing an empty slice reproduces the
/// [`write_aspx_data_1ch_real_envelope_tna`] bytes exactly.
///
/// `fic_used_in_sfb` / `tic_used_in_slot` stay at the all-zero scaffold.
pub fn write_aspx_data_1ch_real_envelope_tna_ah(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch: AspxRealEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah: &[bool],
) -> Result<(), &'static str> {
    write_aspx_data_1ch_real_envelope_tna_ah_framed(bw, cfg, ch, tna_mode, ah, true)
}

/// [`write_aspx_data_1ch_real_envelope_tna_ah`] with an explicit
/// `b_iframe` — per ETSI TS 103 190-1 Table 51 the leading 3-bit
/// `aspx_xover_subband_offset` is only present on I-frames; P-frame
/// bodies start directly at `aspx_framing(0)` and the decoder reuses
/// the I-frame-sticky xover (always 0 for this writer family).
pub fn write_aspx_data_1ch_real_envelope_tna_ah_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    ch: AspxRealEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah: &[bool],
    b_iframe: bool,
) -> Result<(), &'static str> {
    let xover: u32 = 0;
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    // aspx_framing(0): FIXFIX, num_env = 1.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(0, envbits);
    if cfg.signals_freq_res() {
        bw.write_bit(false);
    }
    // aspx_delta_dir(0): SIGNAL + NOISE both FREQ.
    bw.write_bit(false);
    bw.write_bit(false);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_1ch(): real tna_mode + real add_harmonic; fic/tic
    // gates stay 0.
    let payload = AspxHfgenIwc1ChPayload {
        tna_mode,
        add_harmonic: ah,
        ..AspxHfgenIwc1ChPayload::default()
    };
    write_aspx_hfgen_iwc_1ch(
        bw,
        &payload,
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    );

    let num_sbg_sig = if cfg.signals_freq_res() {
        counts.num_sbg_sig_lowres
    } else {
        counts.num_sbg_sig_highres
    };
    let num_sbg_noise = counts.num_sbg_noise;

    let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };

    write_aspx_sig_envelope_values(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        ch.sig,
        num_sbg_sig,
    );
    write_aspx_noise_envelope_values(bw, aspx::AspxStereoMode::Level, ch.noise, num_sbg_noise);
    Ok(())
}

/// Write a single ASPX SIGNAL envelope (one F0 + `(num_sbg − 1)` DF
/// codewords) using the value-emitting helpers. Caller-supplied values
/// shorter than `num_sbg` zero-pad the trailing entries.
fn write_aspx_sig_envelope_values(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    values: &[i32],
    num_sbg: u32,
) {
    if num_sbg == 0 {
        return;
    }
    let f0 = values.first().copied().unwrap_or(0);
    write_aspx_sig_f0_value(bw, quant, stereo, f0);
    for i in 1..num_sbg as usize {
        let delta = values.get(i).copied().unwrap_or(0);
        write_aspx_sig_df_value(bw, quant, stereo, delta);
    }
}

/// Write a single ASPX NOISE envelope (one F0 + `(num_sbg − 1)` DF
/// codewords) using the value-emitting helpers.
fn write_aspx_noise_envelope_values(
    bw: &mut BitWriter,
    stereo: aspx::AspxStereoMode,
    values: &[i32],
    num_sbg: u32,
) {
    if num_sbg == 0 {
        return;
    }
    let f0 = values.first().copied().unwrap_or(0);
    write_aspx_noise_f0_value(bw, stereo, f0);
    for i in 1..num_sbg as usize {
        let delta = values.get(i).copied().unwrap_or(0);
        write_aspx_noise_df_value(bw, stereo, delta);
    }
}

/// Write one ASPX SIGNAL envelope honouring its transmission direction
/// per ETSI TS 103 190-1 §4.2.12.9 Table 58 (`aspx_huff_data()`):
///
/// * FREQ (`direction_time == false`): one `*_F0` codeword followed by
///   `(num_sbg − 1)` `*_DF` codewords (matching the decoder's FREQ
///   branch in [`crate::aspx::parse_aspx_huff_data`]).
/// * TIME (`direction_time == true`): `num_sbg` `*_DT` codewords (every
///   subband group carries a delta-time value).
///
/// Caller-supplied `values` shorter than `num_sbg` zero-pad the trailing
/// entries (matching the decoder's `unwrap_or(0)` clamp surface).
fn write_aspx_sig_envelope_directional(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    env: &AspxEncodedEnvelope,
    num_sbg: u32,
) {
    if num_sbg == 0 {
        return;
    }
    if env.direction_time {
        for i in 0..num_sbg as usize {
            let d = env.values.get(i).copied().unwrap_or(0);
            write_aspx_sig_dt_value(bw, quant, stereo, d);
        }
    } else {
        let f0 = env.values.first().copied().unwrap_or(0);
        write_aspx_sig_f0_value(bw, quant, stereo, f0);
        for i in 1..num_sbg as usize {
            let d = env.values.get(i).copied().unwrap_or(0);
            write_aspx_sig_df_value(bw, quant, stereo, d);
        }
    }
}

/// Write one ASPX NOISE envelope honouring its transmission direction
/// per ETSI TS 103 190-1 §4.2.12.9 Table 58. Same FREQ / TIME shape as
/// [`write_aspx_sig_envelope_directional`] but routed through the NOISE
/// value writers (NOISE quant step is always Fine / 1.5 dB).
fn write_aspx_noise_envelope_directional(
    bw: &mut BitWriter,
    stereo: aspx::AspxStereoMode,
    env: &AspxEncodedEnvelope,
    num_sbg: u32,
) {
    if num_sbg == 0 {
        return;
    }
    if env.direction_time {
        for i in 0..num_sbg as usize {
            let d = env.values.get(i).copied().unwrap_or(0);
            write_aspx_noise_dt_value(bw, stereo, d);
        }
    } else {
        let f0 = env.values.first().copied().unwrap_or(0);
        write_aspx_noise_f0_value(bw, stereo, f0);
        for i in 1..num_sbg as usize {
            let d = env.values.get(i).copied().unwrap_or(0);
            write_aspx_noise_df_value(bw, stereo, d);
        }
    }
}

/// A multi-envelope ASPX channel payload — per-envelope SIGNAL + NOISE
/// DPCM rows (the round-292 [`AspxEncodedEnvelope`] shape, carrying each
/// envelope's chosen FREQ / TIME direction).
///
/// `sig` carries `num_env` SIGNAL envelopes; `noise` carries
/// `num_noise` NOISE envelopes (the decoder derives `num_noise = 2` when
/// `num_env > 1`, else `1` — see [`crate::aspx::AspxFraming`]). Slices
/// shorter than the framing's envelope count zero-pad the missing
/// envelopes (each missing envelope emits an all-zero FREQ row).
#[derive(Debug, Clone, Default)]
pub struct AspxMultiEnvelopeChannel<'a> {
    /// Per-envelope SIGNAL DPCM rows (length `num_env`).
    pub sig: &'a [AspxEncodedEnvelope],
    /// Per-envelope NOISE DPCM rows (length `num_noise`).
    pub noise: &'a [AspxEncodedEnvelope],
}

/// Default all-zero FREQ envelope used to zero-pad short caller slices.
fn zero_freq_env() -> AspxEncodedEnvelope {
    AspxEncodedEnvelope {
        values: Vec::new(),
        direction_time: false,
    }
}

/// Emit a **multi-envelope** `aspx_data_2ch()` body per ETSI TS 103
/// 190-1 §4.2.12.4 Table 52 — the `num_env > 1` generalisation of
/// [`write_aspx_data_2ch_real_envelope`].
///
/// The framing is still FIXFIX, but `tmp_num_env` is set so the decoder
/// derives `num_env = 1 << tmp_num_env` (Table 123 / Table 126):
///
/// * `aspx_xover_subband_offset = 0` (3 b).
/// * `aspx_framing(0)`: FIXFIX prefix `0` (1 b), `tmp_num_env` (the
///   config's `fixfix_tmp_num_env_bits()` width) = `log2(num_env)`.
/// * `aspx_balance = 1` (1 b) — channel 1 reuses channel 0's framing.
/// * `aspx_delta_dir(0)` + `aspx_delta_dir(1)`: per-envelope SIGNAL
///   direction bits (`num_env` each) followed by per-envelope NOISE
///   direction bits (`num_noise` each), taken from the
///   [`AspxEncodedEnvelope::direction_time`] flags.
/// * `aspx_hfgen_iwc_2ch(balance = 1)`: per-SBG `tna_mode = 0` (2 b
///   each) + the four trailer gate bits all zero.
/// * Four `aspx_ec_data()` calls (ch0 / ch1 SIGNAL, ch0 / ch1 NOISE) —
///   each walks `num_env` (SIGNAL) / `num_noise` (NOISE) envelopes,
///   honouring per-envelope FREQ / TIME directions via
///   [`write_aspx_sig_envelope_directional`] /
///   [`write_aspx_noise_envelope_directional`].
///
/// Per Table 52 the SIGNAL quant step is `cfg.quant_mode_env` for
/// `num_env > 1` (the FIXFIX + `num_env == 1` → Fine clamp does not
/// apply); NOISE is always Fine. The body round-trips through
/// [`crate::asf::parse_aspx_data_2ch_body`] → `delta_decode_sig` /
/// `delta_decode_noise` to recover the caller's per-`[sbg][atsg]`
/// `qscf` matrices.
///
/// `num_env` must be a power of two in `2..=1 << fixfix_tmp_num_env_bits()`
/// and the config must not signal a per-envelope `aspx_freq_res` bit
/// (`!cfg.signals_freq_res()`) — FIXFIX carries only one freq_res entry
/// while the SIGNAL ec_data walks `num_env` envelopes, so the high-res
/// fallback must apply uniformly. The writer returns an error otherwise.
pub fn write_aspx_data_2ch_multi_envelope(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    num_env: u32,
    ch0: AspxMultiEnvelopeChannel<'_>,
    ch1: AspxMultiEnvelopeChannel<'_>,
) -> Result<(), &'static str> {
    write_aspx_data_2ch_multi_envelope_tna_ah(bw, cfg, num_env, ch0, ch1, &[], &[], &[])
}

/// `aspx_data_2ch()` multi-envelope writer that additionally carries a
/// real channel-0 `aspx_tna_mode` vector + per-channel `aspx_add_harmonic`
/// vectors.
///
/// Identical framing to [`write_aspx_data_2ch_multi_envelope`] except the
/// `aspx_hfgen_iwc_2ch(balance = 1)` element is routed through
/// [`write_aspx_hfgen_iwc_2ch`] with `tna_mode` (channel 0, mirrored to
/// channel 1 by `aspx_balance = 1`) + per-channel `ah_left` / `ah_right`
/// `aspx_add_harmonic` slices. Passing empty `tna_mode` / `ah_left` /
/// `ah_right` reproduces [`write_aspx_data_2ch_multi_envelope`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn write_aspx_data_2ch_multi_envelope_tna_ah(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    num_env: u32,
    ch0: AspxMultiEnvelopeChannel<'_>,
    ch1: AspxMultiEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah_left: &[bool],
    ah_right: &[bool],
) -> Result<(), &'static str> {
    write_aspx_data_2ch_multi_envelope_tna_ah_framed(
        bw, cfg, num_env, ch0, ch1, tna_mode, ah_left, ah_right, true,
    )
}

/// [`write_aspx_data_2ch_multi_envelope_tna_ah`] with an explicit
/// `b_iframe` — Table 52 gates the leading 3-bit xover offset on
/// I-frames (see
/// [`write_aspx_data_2ch_real_envelope_tna_ah_framed`]).
#[allow(clippy::too_many_arguments)]
pub fn write_aspx_data_2ch_multi_envelope_tna_ah_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    num_env: u32,
    ch0: AspxMultiEnvelopeChannel<'_>,
    ch1: AspxMultiEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah_left: &[bool],
    ah_right: &[bool],
    b_iframe: bool,
) -> Result<(), &'static str> {
    let tmp_num_env = check_multi_env_cfg(cfg, num_env)?;
    let num_noise = if num_env > 1 { 2 } else { 1 };

    let xover: u32 = 0;
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    // aspx_framing(0): FIXFIX prefix `0`, tmp_num_env → num_env.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(tmp_num_env, envbits);
    // signals_freq_res() is guaranteed false by check_multi_env_cfg.

    // aspx_balance = 1 → channel-1 reuses channel-0's framing.
    bw.write_bit(true);

    // aspx_delta_dir(0): num_env SIGNAL bits + num_noise NOISE bits.
    write_delta_dir_bits(bw, num_env, num_noise, &ch0);
    // aspx_delta_dir(1): same shape for channel 1.
    write_delta_dir_bits(bw, num_env, num_noise, &ch1);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_2ch(balance = 1): real tna_mode[0] (mirrored to [1])
    // + per-channel add_harmonic vectors.
    let empty: &[u8] = &[];
    let payload = AspxHfgenIwc2ChPayload {
        tna_mode: [tna_mode, empty],
        add_harmonic: [ah_left, ah_right],
        ..AspxHfgenIwc2ChPayload::default()
    };
    write_aspx_hfgen_iwc_2ch(
        bw,
        &payload,
        true,
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    );

    // SIGNAL band count: high-res (no in-band freq_res bit).
    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;
    // num_env > 1 ⇒ no FIXFIX Fine clamp; use cfg.quant_mode_env.
    let qmode_sig = cfg.quant_mode_env;

    // ch0 SIGNAL (LEVEL) — num_env envelopes.
    write_sig_envelopes(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        ch0.sig,
        num_env,
        num_sbg_sig,
    );
    // ch1 SIGNAL (BALANCE).
    write_sig_envelopes(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Balance,
        ch1.sig,
        num_env,
        num_sbg_sig,
    );
    // ch0 NOISE (LEVEL) — num_noise envelopes.
    write_noise_envelopes(
        bw,
        aspx::AspxStereoMode::Level,
        ch0.noise,
        num_noise,
        num_sbg_noise,
    );
    // ch1 NOISE (BALANCE).
    write_noise_envelopes(
        bw,
        aspx::AspxStereoMode::Balance,
        ch1.noise,
        num_noise,
        num_sbg_noise,
    );
    Ok(())
}

/// Emit a **multi-envelope** `aspx_data_1ch()` body per ETSI TS 103
/// 190-1 §4.2.12.3 Table 51 — the `num_env > 1` generalisation of
/// [`write_aspx_data_1ch_real_envelope`]. Same framing rules as
/// [`write_aspx_data_2ch_multi_envelope`] minus the `aspx_balance` bit
/// and the second channel; the lone channel uses the LEVEL stereo mode
/// throughout. Round-trips through [`crate::asf::parse_aspx_data_1ch_body`].
pub fn write_aspx_data_1ch_multi_envelope(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    num_env: u32,
    ch: AspxMultiEnvelopeChannel<'_>,
) -> Result<(), &'static str> {
    write_aspx_data_1ch_multi_envelope_tna_ah(bw, cfg, num_env, ch, &[], &[])
}

/// `aspx_data_1ch()` multi-envelope writer that additionally carries a real
/// `aspx_tna_mode` vector + an `aspx_add_harmonic` vector.
///
/// Identical framing to [`write_aspx_data_1ch_multi_envelope`] except the
/// `aspx_hfgen_iwc_1ch()` element is routed through
/// [`write_aspx_hfgen_iwc_1ch`] with the caller's `tna_mode` + `ah`
/// slices. Passing empty slices reproduces
/// [`write_aspx_data_1ch_multi_envelope`] exactly.
pub fn write_aspx_data_1ch_multi_envelope_tna_ah(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    num_env: u32,
    ch: AspxMultiEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah: &[bool],
) -> Result<(), &'static str> {
    write_aspx_data_1ch_multi_envelope_tna_ah_framed(bw, cfg, num_env, ch, tna_mode, ah, true)
}

/// [`write_aspx_data_1ch_multi_envelope_tna_ah`] with an explicit
/// `b_iframe` — Table 51 gates the leading 3-bit xover offset on
/// I-frames (see [`write_aspx_data_1ch_real_envelope_tna_ah_framed`]).
pub fn write_aspx_data_1ch_multi_envelope_tna_ah_framed(
    bw: &mut BitWriter,
    cfg: &aspx::AspxConfig,
    num_env: u32,
    ch: AspxMultiEnvelopeChannel<'_>,
    tna_mode: &[u8],
    ah: &[bool],
    b_iframe: bool,
) -> Result<(), &'static str> {
    let tmp_num_env = check_multi_env_cfg(cfg, num_env)?;
    let num_noise = if num_env > 1 { 2 } else { 1 };

    let xover: u32 = 0;
    if b_iframe {
        bw.write_u32(xover, 3);
    }

    // aspx_framing(0): FIXFIX, tmp_num_env → num_env.
    bw.write_bit(false);
    let envbits = cfg.fixfix_tmp_num_env_bits();
    bw.write_u32(tmp_num_env, envbits);

    // aspx_delta_dir(0): num_env SIGNAL + num_noise NOISE direction bits.
    write_delta_dir_bits(bw, num_env, num_noise, &ch);

    let tables = aspx::derive_aspx_frequency_tables(cfg, xover)
        .map_err(|_| "encoder: aspx frequency-tables derivation failed")?;
    let counts = tables.counts;

    // aspx_hfgen_iwc_1ch(): real tna_mode + real add_harmonic; fic/tic 0.
    let payload = AspxHfgenIwc1ChPayload {
        tna_mode,
        add_harmonic: ah,
        ..AspxHfgenIwc1ChPayload::default()
    };
    write_aspx_hfgen_iwc_1ch(
        bw,
        &payload,
        counts.num_sbg_noise,
        counts.num_sbg_sig_highres,
        0,
    );

    let num_sbg_sig = counts.num_sbg_sig_highres;
    let num_sbg_noise = counts.num_sbg_noise;
    let qmode_sig = cfg.quant_mode_env;

    write_sig_envelopes(
        bw,
        qmode_sig,
        aspx::AspxStereoMode::Level,
        ch.sig,
        num_env,
        num_sbg_sig,
    );
    write_noise_envelopes(
        bw,
        aspx::AspxStereoMode::Level,
        ch.noise,
        num_noise,
        num_sbg_noise,
    );
    Ok(())
}

/// Validate the multi-envelope writer preconditions and return the
/// `tmp_num_env` field value (`log2(num_env)`).
fn check_multi_env_cfg(cfg: &aspx::AspxConfig, num_env: u32) -> Result<u32, &'static str> {
    if cfg.signals_freq_res() {
        return Err("encoder: multi-envelope writer requires !signals_freq_res()");
    }
    if num_env == 0 || !num_env.is_power_of_two() {
        return Err("encoder: num_env must be a positive power of two");
    }
    let tmp_num_env = num_env.trailing_zeros();
    let envbits = cfg.fixfix_tmp_num_env_bits();
    if tmp_num_env >= (1u32 << envbits) {
        return Err("encoder: num_env exceeds fixfix_tmp_num_env_bits() capacity");
    }
    Ok(tmp_num_env)
}

/// Emit one channel's `aspx_delta_dir(ch)` bits: `num_env` SIGNAL
/// direction flags then `num_noise` NOISE direction flags (Table 54).
fn write_delta_dir_bits(
    bw: &mut BitWriter,
    num_env: u32,
    num_noise: u32,
    ch: &AspxMultiEnvelopeChannel<'_>,
) {
    for e in 0..num_env as usize {
        let dir = ch.sig.get(e).map(|r| r.direction_time).unwrap_or(false);
        bw.write_bit(dir);
    }
    for e in 0..num_noise as usize {
        let dir = ch.noise.get(e).map(|r| r.direction_time).unwrap_or(false);
        bw.write_bit(dir);
    }
}

/// Emit `num_env` SIGNAL envelopes for one channel.
fn write_sig_envelopes(
    bw: &mut BitWriter,
    quant: aspx::AspxQuantStep,
    stereo: aspx::AspxStereoMode,
    rows: &[AspxEncodedEnvelope],
    num_env: u32,
    num_sbg: u32,
) {
    let pad = zero_freq_env();
    for e in 0..num_env as usize {
        let env = rows.get(e).unwrap_or(&pad);
        write_aspx_sig_envelope_directional(bw, quant, stereo, env, num_sbg);
    }
}

/// Emit `num_noise` NOISE envelopes for one channel.
fn write_noise_envelopes(
    bw: &mut BitWriter,
    stereo: aspx::AspxStereoMode,
    rows: &[AspxEncodedEnvelope],
    num_noise: u32,
    num_sbg: u32,
) {
    let pad = zero_freq_env();
    for e in 0..num_noise as usize {
        let env = rows.get(e).unwrap_or(&pad);
        write_aspx_noise_envelope_directional(bw, stereo, env, num_sbg);
    }
}

// ====================================================================
// ASPX envelope-extractor (round 234)
// ====================================================================
//
// The round-219 value-emitting helpers + the round-226 builder pair
// (`write_aspx_data_{1,2}ch_real_envelope`) consume per-channel
// `AspxRealEnvelopeChannel { sig: &[i32], noise: &[i32] }` payloads
// whose entries are the FREQ-direction DPCM sequence
// `[F0, DF₁, DF₂, …]` for one envelope (`atsg = 0`) — exactly the
// per-`(sbg, atsg)` quant indices the decoder's
// `delta_decode_sig` / `delta_decode_noise` would reconstruct.
//
// To close the loop and let the new builders be chained with input
// MDCT spectra, the encoder needs the inverse of Pseudocodes 80, 81,
// 82 and 83:
//
//   * Pseudocode 82 (signal, non-balance): `scf = n_subbands · 2^(qscf/a)`
//     with `n_subbands = 64`, `a = 2` (Fine) / `1` (Coarse). Inverse:
//     `qscf = round(a · log2(scf / n_subbands))`.
//   * Pseudocode 83 (noise): `scf_noise = 2^(6 - qscf_noise)`. Inverse:
//     `qscf_noise = round(6 - log2(scf_noise))`.
//   * Pseudocode 80 / 81 (FREQ direction, `delta = 1`): forward
//     `qscf[sbg] = sum(values[0..=sbg])` ⇒ inverse
//     `values[0] = qscf[0]`, `values[sbg≥1] = qscf[sbg] − qscf[sbg−1]`.
//
// The DPCM step always emits the F0 entry as `qscf[0]` (no cb_off
// applied — the F0 codebooks have `cb_off = 0`); the DF entries are
// the signed first-difference of `qscf`. The round-219 helpers'
// `[-cb_off, +cb_off]` symmetric clamp on the DF side and
// `[0, codebook_length)` clamp on the F0 side are then applied by the
// writer, so the extractor is allowed to produce values outside those
// ranges — the writer saturates them at the codebook edge.

/// Quantize a single signal-envelope `scf` value into the per-`(sbg,
/// atsg)` `qscf` integer used by Pseudocode 80, inverting Pseudocode 82
/// (`scf = n_subbands · 2^(qscf/a)`).
///
/// `qmode_env = Fine` ⇒ `a = 2` (1.5 dB step), `Coarse` ⇒ `a = 1`
/// (3 dB step). `num_qmf_subbands` mirrors the decoder's constant
/// `64` (the dequantizer's `n_subbands` parameter). Non-positive `scf`
/// — including the spec's `scf[0][atsg] = scf[1][atsg]` carry-through
/// path when `scf[1]` is negative — is clamped to a minimum positive
/// value before the log so the result stays finite.
///
/// Refs ETSI TS 103 190-1 §5.7.6.3.5 Pseudocode 82.
pub fn quantize_sig_scf(scf: f32, qmode_env: aspx::AspxQuantStep, num_qmf_subbands: u32) -> i32 {
    let a: f32 = match qmode_env {
        aspx::AspxQuantStep::Fine => 2.0,
        aspx::AspxQuantStep::Coarse => 1.0,
    };
    let n_subbands = num_qmf_subbands as f32;
    // The dequantizer's range is (0, ∞); a non-positive scf cannot be
    // represented by the formula. Clamp on the encoder side to a tiny
    // positive value so the log stays defined.
    let scf_clamped = scf.max(f32::MIN_POSITIVE);
    let ratio = (scf_clamped / n_subbands).max(f32::MIN_POSITIVE);
    let q = a * ratio.log2();
    q.round() as i32
}

/// Quantize a single noise-envelope `scf` value into the per-`(sbg,
/// atsg)` `qscf` integer used by Pseudocode 81, inverting Pseudocode 83
/// (`scf_noise = 2^(6 - qscf_noise)`).
///
/// Refs ETSI TS 103 190-1 §5.7.6.3.5 Pseudocode 83.
pub fn quantize_noise_scf(scf: f32) -> i32 {
    const NOISE_FLOOR_OFFSET: i32 = 6;
    let scf_clamped = scf.max(f32::MIN_POSITIVE);
    let q = (NOISE_FLOOR_OFFSET as f32) - scf_clamped.log2();
    q.round() as i32
}

/// Convert a per-`sbg` `qscf` vector for a single envelope (`atsg = 0`)
/// into the FREQ-direction DPCM sequence `[F0, DF₁, DF₂, …]` that the
/// round-219 value-emitting helpers + round-226 builder pair consume.
///
/// Per Pseudocode 80 / 81 FREQ branch with `delta = 1`:
/// `qscf[sbg] = sum(values[0..=sbg])` ⇒ `values[0] = qscf[0]`,
/// `values[sbg ≥ 1] = qscf[sbg] − qscf[sbg − 1]`.
///
/// `qscf.len()` is the number of subband groups; the returned vector
/// has the same length. Empty input returns an empty vector.
pub fn freq_dpcm_encode_qscf(qscf: &[i32]) -> Vec<i32> {
    if qscf.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(qscf.len());
    out.push(qscf[0]);
    for sbg in 1..qscf.len() {
        out.push(qscf[sbg] - qscf[sbg - 1]);
    }
    out
}

/// Encode one envelope's per-`sbg` `qscf` row into the **TIME-direction**
/// DPCM delta sequence — the exact dual of the `direction_time == true`
/// branch of the decoder's [`crate::aspx::delta_decode_sig`] /
/// [`crate::aspx::delta_decode_noise`].
///
/// On the decode side the TIME branch reconstructs, per subband group,
///
/// ```text
///   qscf[sbg][atsg] = prev[sbg] + delta · values[sbg]
/// ```
///
/// where `prev[sbg]` is the same subband group's `qscf` in the
/// *previous* envelope (`qscf[sbg][atsg-1]`), or — for the first
/// envelope of the frame (`atsg == 0`) — the carried-over
/// `qscf_prev_last[sbg]` from the previous frame (Pseudocode 80 / 81
/// `qscf[sbg][-1]`). Inverting for the transmitted DT value:
///
/// ```text
///   values[sbg] = (qscf[sbg] − prev[sbg]) / delta
/// ```
///
/// `delta` is the per-frame DPCM step sign (`±1` for A-SPX — the only
/// values the bitstream carries); the division is therefore exact. A
/// `delta == 0` (never produced by a conformant decoder configuration)
/// is treated as `delta == 1` so the encoder stays total.
///
/// `qscf` is this envelope's per-`sbg` quant indices; `prev` is the
/// reference row the decoder will subtract against (length-mismatched
/// `prev` zero-extends, matching the decoder's
/// `qscf_prev_last.get(sbg).unwrap_or(0)`). The returned vector has the
/// same length as `qscf`. Empty input returns an empty vector.
///
/// Refs ETSI TS 103 190-1 §5.7.6.3.4 Pseudocode 80 (TIME branch) /
/// Pseudocode 81 (NOISE TIME branch).
pub fn time_dpcm_encode_qscf(qscf: &[i32], prev: &[i32], delta: i32) -> Vec<i32> {
    if qscf.is_empty() {
        return Vec::new();
    }
    let step = if delta == 0 { 1 } else { delta };
    qscf.iter()
        .enumerate()
        .map(|(sbg, &q)| {
            let p = prev.get(sbg).copied().unwrap_or(0);
            (q - p) / step
        })
        .collect()
}

/// One encoded ASPX envelope: the transmission direction the encoder
/// chose plus the per-`sbg` DPCM delta sequence for that direction.
/// `direction_time == false` ⇒ FREQ (`[F0, DF₁, …]`, from
/// [`freq_dpcm_encode_qscf`]); `direction_time == true` ⇒ TIME
/// (all-`DT`, from [`time_dpcm_encode_qscf`]). Mirrors the decoder's
/// [`crate::aspx::AspxHuffEnv`] before Huffman coding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxEncodedEnvelope {
    /// Per-subband-group DPCM delta values for the chosen direction.
    pub values: Vec<i32>,
    /// Transmission direction (false = FREQ, true = TIME).
    pub direction_time: bool,
}

/// Reconstruct the absolute quantized scale-factor row from a packed
/// FREQ-DPCM vector `[F0, DF₁, …]` — the running sum the decoder's
/// Pseudocode 80/81 FREQ branch accumulates (`delta = 1` domain).
pub fn qscf_row_from_freq_dpcm(packed: &[i32]) -> Vec<i32> {
    let mut acc = 0i32;
    packed
        .iter()
        .map(|&d| {
            acc += d;
            acc
        })
        .collect()
}

/// [`qscf_row_from_freq_dpcm`] extended to exactly `n` subband groups
/// by repeating the last accumulated value — the same "missing deltas
/// read 0 ⇒ constant tail" semantics the envelope writers apply when a
/// packed row is shorter than the band count. An empty row extends to
/// all-zero.
pub fn qscf_row_from_freq_dpcm_extended(packed: &[i32], n: usize) -> Vec<i32> {
    let mut row = qscf_row_from_freq_dpcm(packed);
    let last = row.last().copied().unwrap_or(0);
    row.resize(n, last);
    row
}

// ====================================================================
// §5.7.6.3.5 Pseudocode 84 inverse — (L, R) LEVEL rows → (sum, pan)
// wire rows for aspx_balance = 1 pairs.
// ====================================================================

/// Convert a per-channel pair of **absolute** SIGNAL envelope quant
/// rows (LEVEL domain — `scf = 64 · 2^(q/a)`) into the jointly coded
/// `(sum, pan)` pair the `aspx_balance == 1` bitstream carries, i.e.
/// the exact inverse of ETSI TS 103 190-1 §5.7.6.3.5 Pseudocode 84
/// (signal loop) on the quantization grid:
///
/// * sum channel A: `q_a = a · log2(2^(q_l/a) + 2^(q_r/a)) − a`
///   (so the joint `nom = 2 · 64 · 2^(q_a/a)` equals
///   `scf_l + scf_r`), rounded to the LEVEL grid;
/// * balance channel B **wire steps**: `p = (a · PAN_OFFSET + q_l −
///   q_r) / 2`, rounded and clamped to the BALANCE codebook range
///   `[0, a · PAN_OFFSET]`. The decoder's Pseudocode 80 `delta = 2`
///   accumulation turns the transmitted steps back into
///   `qscf_b = 2 · p`, and `qscf_b/a − PAN_OFFSET` recovers the pan
///   `(q_l − q_r)/a` exactly (up to the balance grid's 2-step
///   quantization).
///
/// Both outputs are absolute rows in the transmitted-symbol
/// accumulation domain (`delta = 1` for A, wire steps for B), ready
/// for [`freq_dpcm_encode_qscf`] / [`time_dpcm_encode_qscf`] packing —
/// the same domain the cross-frame [`Acpl3EnvPrevRows`] bookkeeping
/// and the decoder's [`crate::aspx::AspxEnvPrev`] use. Length-
/// mismatched inputs zero-extend the shorter row.
/// Single-cell inverse of the Pseudocode 84 signal loop: `(q_l, q_r)`
/// LEVEL quant values → `(sum, pan-wire-step)`. See
/// [`balance_encode_sig_rows`] for the derivation.
pub fn balance_encode_sig_cell(ql: i32, qr: i32, qmode: crate::aspx::AspxQuantStep) -> (i32, i32) {
    let a = match qmode {
        crate::aspx::AspxQuantStep::Fine => 2i32,
        crate::aspx::AspxQuantStep::Coarse => 1i32,
    };
    let af = a as f64;
    let pan_max = a * crate::aspx::ASPX_PAN_OFFSET;
    // LEVEL F0 codebook upper bound (Annex A.2: 71 entries Fine, 36
    // Coarse) — keeps the sum row inside the range every LEVEL
    // codeword path can carry.
    let sum_max = match qmode {
        crate::aspx::AspxQuantStep::Fine => 70i32,
        crate::aspx::AspxQuantStep::Coarse => 35i32,
    };
    // q_a = a·log2(2^(ql/a) + 2^(qr/a)) − a, computed around the
    // larger level for numeric stability.
    let hi = ql.max(qr);
    let lo = ql.min(qr);
    let qa = af * (1.0 + 2f64.powf((lo - hi) as f64 / af)).log2() + hi as f64 - af;
    let sum = (qa.round() as i32).clamp(0, sum_max);
    // p = (a·PAN_OFFSET + ql − qr) / 2, round-half-up.
    let num = a * crate::aspx::ASPX_PAN_OFFSET + ql - qr;
    let p = ((num as f64 / 2.0).round() as i32).clamp(0, pan_max);
    (sum, p)
}

pub fn balance_encode_sig_rows(
    q_l: &[i32],
    q_r: &[i32],
    qmode: crate::aspx::AspxQuantStep,
) -> (Vec<i32>, Vec<i32>) {
    let n = q_l.len().max(q_r.len());
    let mut sum = Vec::with_capacity(n);
    let mut pan = Vec::with_capacity(n);
    for i in 0..n {
        let ql = q_l.get(i).copied().unwrap_or(0);
        let qr = q_r.get(i).copied().unwrap_or(0);
        let (s, p) = balance_encode_sig_cell(ql, qr, qmode);
        sum.push(s);
        pan.push(p);
    }
    (sum, pan)
}

/// NOISE counterpart of [`balance_encode_sig_rows`] — inverse of the
/// Pseudocode 84 noise loop. Inputs are absolute LEVEL-domain noise
/// quant rows (`scf = 2^(6 − q)`); outputs are the sum channel row
/// (`q_a = 7 − log2(2^(6−q_l) + 2^(6−q_r))`, LEVEL noise grid) and the
/// balance channel **wire steps** `p = (PAN_OFFSET + q_r − q_l) / 2`
/// clamped to `[0, PAN_OFFSET]` (note the sign flip versus the signal
/// pan: noise levels quantize inverted, `scf_a/scf_b =
/// 2^(qscf_b − PAN_OFFSET)`).
/// Single-cell inverse of the Pseudocode 84 noise loop: `(q_l, q_r)`
/// LEVEL noise quant values → `(sum, pan-wire-step)`. See
/// [`balance_encode_noise_rows`].
pub fn balance_encode_noise_cell(ql: i32, qr: i32) -> (i32, i32) {
    // NOISE_LEVEL_F0 codebook range: 30 entries (Annex A.2).
    const SUM_MAX: i32 = 29;
    // q_a = 7 − log2(2^(6−ql) + 2^(6−qr)), computed around the
    // louder (numerically larger scf ⇒ smaller q) channel.
    let lo_q = ql.min(qr); // louder channel
    let hi_q = ql.max(qr);
    let qa = 7.0 - ((1.0 + 2f64.powi(lo_q - hi_q)).log2() + (6 - lo_q) as f64);
    let sum = (qa.round() as i32).clamp(0, SUM_MAX);
    let num = crate::aspx::ASPX_PAN_OFFSET + qr - ql;
    let p = ((num as f64 / 2.0).round() as i32).clamp(0, crate::aspx::ASPX_PAN_OFFSET);
    (sum, p)
}

pub fn balance_encode_noise_rows(q_l: &[i32], q_r: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let n = q_l.len().max(q_r.len());
    let mut sum = Vec::with_capacity(n);
    let mut pan = Vec::with_capacity(n);
    for i in 0..n {
        let ql = q_l.get(i).copied().unwrap_or(0);
        let qr = q_r.get(i).copied().unwrap_or(0);
        let (s, p) = balance_encode_noise_cell(ql, qr);
        sum.push(s);
        pan.push(p);
    }
    (sum, pan)
}

/// Matrix (`[sbg][atsg]`) forms of [`balance_encode_sig_cell`] /
/// [`balance_encode_noise_cell`] for the multi-envelope builder: apply
/// the Pseudocode 84 inverse cell-wise across a channel pair's quant
/// matrices. Missing / short partner cells read 0.
pub fn balance_encode_sig_matrix(
    q_l: &[Vec<i32>],
    q_r: &[Vec<i32>],
    qmode: crate::aspx::AspxQuantStep,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut sum = q_l.to_vec();
    let mut pan = q_l.to_vec();
    for (sbg, row) in q_l.iter().enumerate() {
        for (atsg, &ql) in row.iter().enumerate() {
            let qr = q_r.get(sbg).and_then(|r| r.get(atsg)).copied().unwrap_or(0);
            let (s, p) = balance_encode_sig_cell(ql, qr, qmode);
            sum[sbg][atsg] = s;
            pan[sbg][atsg] = p;
        }
    }
    (sum, pan)
}

/// See [`balance_encode_sig_matrix`].
pub fn balance_encode_noise_matrix(
    q_l: &[Vec<i32>],
    q_r: &[Vec<i32>],
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut sum = q_l.to_vec();
    let mut pan = q_l.to_vec();
    for (sbg, row) in q_l.iter().enumerate() {
        for (atsg, &ql) in row.iter().enumerate() {
            let qr = q_r.get(sbg).and_then(|r| r.get(atsg)).copied().unwrap_or(0);
            let (s, p) = balance_encode_noise_cell(ql, qr);
            sum[sbg][atsg] = s;
            pan[sbg][atsg] = p;
        }
    }
    (sum, pan)
}

/// P-frame envelope direction decision for a single FIXFIX envelope
/// (ETSI TS 103 190-1 §4.2.12.5 / §5.7.6.3.4): given the current
/// envelope's packed FREQ-DPCM vector and the previous frame's last
/// envelope row (`qscf_prev`, same `delta = 1` domain), transmit in the
/// TIME direction when its delta magnitudes are strictly cheaper
/// (`Σ|qscf_cur − qscf_prev| < Σ|freq values|`; FREQ wins ties so
/// I-frame-shaped output is preferred). `None` / absent previous rows
/// (first frame, I-frame, path switch) always select FREQ.
pub fn choose_envelope_direction(
    packed_freq: &[i32],
    prev_row: Option<&[i32]>,
) -> AspxEncodedEnvelope {
    if let Some(prev) = prev_row {
        let cur = qscf_row_from_freq_dpcm(packed_freq);
        let dt: Vec<i32> = cur
            .iter()
            .enumerate()
            .map(|(i, &c)| c - prev.get(i).copied().unwrap_or(0))
            .collect();
        let time_cost: i64 = dt.iter().map(|&d| (d as i64).abs()).sum();
        let freq_cost: i64 = packed_freq.iter().map(|&d| (d as i64).abs()).sum();
        if time_cost < freq_cost {
            return AspxEncodedEnvelope {
                values: dt,
                direction_time: true,
            };
        }
    }
    AspxEncodedEnvelope {
        values: packed_freq.to_vec(),
        direction_time: false,
    }
}

/// Previous-frame absolute envelope rows for the live 5_X ACPL_3
/// L / R carrier pair — the encoder-side mirror of the decoder's
/// per-channel [`crate::aspx::AspxEnvPrev`], used to drive
/// [`choose_envelope_direction`] on P-frames.
#[derive(Debug, Clone, Default)]
pub struct Acpl3EnvPrevRows {
    /// Channel-0 SIGNAL row — the §5.7.6.3.5 **sum** row (LEVEL
    /// domain, `delta = 1` accumulation).
    pub l_sig: Vec<i32>,
    /// Channel-0 NOISE row (sum, LEVEL noise domain).
    pub l_noise: Vec<i32>,
    /// Channel-1 SIGNAL row — the balance **pan wire steps** (the
    /// decoder's `delta = 2` accumulation recovers `qscf_b = 2 · pan`).
    pub r_sig: Vec<i32>,
    /// Channel-1 NOISE row (pan wire steps).
    pub r_noise: Vec<i32>,
}

/// One A-CPL parameter element's encoded row + transmission direction
/// (ETSI TS 103 190-1 §4.2.13.7 Table 65 `acpl_huff_data()`):
///
/// * `direction_time == false` (DIFF_FREQ): `values` holds the
///   **absolute** per-band quantized indices; the writer derives the
///   F0 + DF codewords.
/// * `direction_time == true` (DIFF_TIME): `values` holds the per-band
///   DT deltas against the previous frame's row (the decoder's
///   Pseudocode 121 `q_prev`, carried in
///   [`crate::acpl_synth::AcplDiffState`]).
#[derive(Debug, Clone, Default)]
pub struct AcplEncodedParam {
    /// Per-parameter-band values (see direction semantics above).
    pub values: Vec<i32>,
    /// `false = DIFF_FREQ`, `true = DIFF_TIME`.
    pub direction_time: bool,
}

/// P-frame A-CPL parameter direction decision (Table 65): transmit
/// DIFF_TIME when its delta magnitudes against the previous frame's
/// row are strictly cheaper than the DIFF_FREQ band-to-band DPCM
/// (`|q0| + Σ|q_i − q_{i−1}|`). FREQ wins ties; `None` previous rows
/// (I-frame, first frame, path switch) always select FREQ.
pub fn choose_acpl_direction(q_cur: &[i32], prev_row: Option<&[i32]>) -> AcplEncodedParam {
    if let Some(prev) = prev_row {
        let dt: Vec<i32> = q_cur
            .iter()
            .enumerate()
            .map(|(i, &c)| c - prev.get(i).copied().unwrap_or(0))
            .collect();
        let time_cost: i64 = dt.iter().map(|&d| (d as i64).abs()).sum();
        let mut freq_cost: i64 = 0;
        let mut prev_q = 0i32;
        for (i, &q) in q_cur.iter().enumerate() {
            freq_cost += if i == 0 {
                (q as i64).abs()
            } else {
                ((q - prev_q) as i64).abs()
            };
            prev_q = q;
        }
        if time_cost < freq_cost {
            return AcplEncodedParam {
                values: dt,
                direction_time: true,
            };
        }
    }
    AcplEncodedParam {
        values: q_cur.to_vec(),
        direction_time: false,
    }
}

/// Previous-frame absolute A-CPL quantized parameter rows for the live
/// 5_X ACPL_3 path — the encoder-side mirror of the decoder's
/// per-element [`crate::acpl_synth::AcplDiffState`] `q_prev` rows
/// (element order: α₁, α₂, β₁, β₂, β₃, γ₁..γ₆ per Table 62).
#[derive(Debug, Clone, Default)]
pub struct Acpl3ParamPrevRows {
    /// The 11 per-element rows in Table 62 order.
    pub rows: [Vec<i32>; 11],
    /// True once a frame has primed the rows (a fresh / cleared state
    /// never drives a DIFF_TIME decision).
    pub primed: bool,
}

/// Encode a full multi-envelope `qscf[sbg][atsg]` matrix into a sequence
/// of [`AspxEncodedEnvelope`] rows — one per envelope (`atsg`) — choosing
/// the cheaper transmission direction per envelope.
///
/// This is the encoder-side inverse of [`crate::aspx::delta_decode_sig`]
/// / [`crate::aspx::delta_decode_noise`]: for each envelope `atsg` the
/// encoder can transmit either
///
/// * **FREQ** — first-difference across subband groups within the
///   envelope ([`freq_dpcm_encode_qscf`]), independent of any other
///   envelope; or
/// * **TIME** — difference against the *previous* envelope's row
///   ([`time_dpcm_encode_qscf`]); for `atsg == 0` the reference is
///   `qscf_prev_last` (the previous frame's last envelope, Pseudocode
///   80 / 81 `qscf[sbg][-1]`).
///
/// The direction policy minimises the L1 norm `Σ|values[sbg]|` of the
/// transmitted deltas per envelope (a proxy for Huffman cost: the
/// `*_DF` / `*_DT` codebooks both peak at the zero-delta lane, so the
/// row with the smaller absolute deltas is the cheaper one). FREQ wins
/// ties — it needs no cross-envelope state, matching the decoder's
/// independent-envelope default. When `force_freq` is `true` every
/// envelope is coded FREQ regardless (reproduces the single-direction
/// scaffold path).
///
/// Round-trip guarantee: feeding the returned `(direction_time, values)`
/// rows back through `delta_decode_sig` / `delta_decode_noise` with the
/// same `delta` and `qscf_prev_last` reconstructs the input `qscf`
/// matrix exactly.
///
/// `qscf` is indexed `[sbg][atsg]` (the decoder's output shape).
/// `qscf_prev_last` is the previous frame's last-envelope row (empty for
/// an I-frame / first envelope with no history). `delta` is the per-frame
/// DPCM step sign (`±1`). An empty `qscf` (no subband groups) returns an
/// empty row list.
///
/// Refs ETSI TS 103 190-1 §5.7.6.3.4 Pseudocode 80 / 81.
pub fn dpcm_encode_qscf_envelopes(
    qscf: &[Vec<i32>],
    qscf_prev_last: &[i32],
    delta: i32,
    force_freq: bool,
) -> Vec<AspxEncodedEnvelope> {
    let num_sbg = qscf.len();
    if num_sbg == 0 {
        return Vec::new();
    }
    let num_env = qscf[0].len();
    let mut out: Vec<AspxEncodedEnvelope> = Vec::with_capacity(num_env);
    for atsg in 0..num_env {
        // Gather this envelope's per-sbg qscf column.
        let col: Vec<i32> = (0..num_sbg)
            .map(|sbg| qscf[sbg].get(atsg).copied().unwrap_or(0))
            .collect();
        let freq = freq_dpcm_encode_qscf(&col);
        if force_freq {
            out.push(AspxEncodedEnvelope {
                values: freq,
                direction_time: false,
            });
            continue;
        }
        // TIME reference row: previous envelope, or carried-over history
        // for the first envelope.
        let prev: Vec<i32> = if atsg == 0 {
            qscf_prev_last.to_vec()
        } else {
            (0..num_sbg)
                .map(|sbg| qscf[sbg].get(atsg - 1).copied().unwrap_or(0))
                .collect()
        };
        let time = time_dpcm_encode_qscf(&col, &prev, delta);
        let cost_freq: i64 = freq.iter().map(|&v| (v as i64).abs()).sum();
        let cost_time: i64 = time.iter().map(|&v| (v as i64).abs()).sum();
        // FREQ wins ties (no cross-envelope state needed).
        if cost_time < cost_freq {
            out.push(AspxEncodedEnvelope {
                values: time,
                direction_time: true,
            });
        } else {
            out.push(AspxEncodedEnvelope {
                values: freq,
                direction_time: false,
            });
        }
    }
    out
}

/// Extract per-channel ASPX SIGNAL envelope quant indices from an input
/// `scf_sig` vector (the per-`sbg` signal envelope-energy scale factors
/// the decoder produces from Pseudocode 82) and pack them into the
/// FREQ-direction DPCM `[F0, DF₁, …]` form the round-219 value-emitting
/// helpers + round-226 builder accept on `AspxRealEnvelopeChannel::sig`.
///
/// `qmode_env` selects the 1.5 dB / 3 dB step on the inverse of
/// Pseudocode 82; `num_qmf_subbands` mirrors the decoder's constant
/// `64`.
///
/// Roundtrip property: feeding the output through the round-219
/// helpers and round-226 builder, then parsing the body back through
/// `parse_aspx_ec_data` + `delta_decode_sig` + `dequantize_sig_scf`,
/// recovers `scf_sig` exactly up to the per-band rounding of
/// `qscf = round(a · log2(scf / 64))`.
///
/// Refs ETSI TS 103 190-1 §5.7.6.3.4 Pseudocode 80, §5.7.6.3.5
/// Pseudocode 82.
pub fn extract_aspx_sig_envelope_indices(
    scf_sig: &[f32],
    qmode_env: aspx::AspxQuantStep,
    num_qmf_subbands: u32,
) -> Vec<i32> {
    let qscf: Vec<i32> = scf_sig
        .iter()
        .map(|&v| quantize_sig_scf(v, qmode_env, num_qmf_subbands))
        .collect();
    freq_dpcm_encode_qscf(&qscf)
}

/// Extract per-channel ASPX NOISE envelope quant indices from an input
/// `scf_noise` vector (the per-`sbg` noise envelope-energy scale
/// factors the decoder produces from Pseudocode 83) and pack them into
/// the FREQ-direction DPCM `[F0, DF₁, …]` form the round-226 builder
/// accepts on `AspxRealEnvelopeChannel::noise`.
///
/// Roundtrip property: feeding the output through the round-219
/// helpers + round-226 builder, then parsing the body back through
/// `parse_aspx_ec_data` + `delta_decode_noise` + `dequantize_noise_scf`,
/// recovers `scf_noise` exactly up to the per-band rounding of
/// `qscf_noise = round(6 - log2(scf_noise))`.
///
/// Refs ETSI TS 103 190-1 §5.7.6.3.4 Pseudocode 81, §5.7.6.3.5
/// Pseudocode 83.
pub fn extract_aspx_noise_envelope_indices(scf_noise: &[f32]) -> Vec<i32> {
    let qscf: Vec<i32> = scf_noise.iter().map(|&v| quantize_noise_scf(v)).collect();
    freq_dpcm_encode_qscf(&qscf)
}

/// Per-channel envelope-energy bundle consumed by
/// [`build_aspx_real_envelope_channel`]: caller-supplied SIGNAL + NOISE
/// `scf` slices (the dequantizer outputs at the decoder's
/// Pseudocode-82 / 83 stage, one value per subband group).
///
/// Lengths shorter than the active SBG count zero-pad on the encoder
/// side — the writer further zero-pads anything still missing past the
/// extracted vector.
#[derive(Debug, Clone, Default)]
pub struct AspxEnvelopeScfChannel<'a> {
    /// Signal envelope scale factors (one per signal subband group).
    pub sig: &'a [f32],
    /// Noise envelope scale factors (one per noise subband group).
    pub noise: &'a [f32],
}

/// Build an [`AspxRealEnvelopeChannel`]-shaped quant-index payload from
/// caller-supplied per-channel envelope `scf` slices by running the
/// signal + noise extractors in sequence. Returns the owned `(sig,
/// noise)` `Vec<i32>` pair; callers wire it into the round-226 builders
/// by taking slice references:
///
/// ```text
///   let (sig0, noise0) = build_aspx_real_envelope_channel(&ch0, qmode);
///   write_aspx_data_1ch_real_envelope(
///       &mut bw,
///       &cfg,
///       AspxRealEnvelopeChannel { sig: &sig0, noise: &noise0 },
///   )?;
/// ```
///
/// `num_qmf_subbands` is the dequantizer's `n_subbands` constant,
/// `64` for AC-4. `qmode_env` selects the 1.5 dB / 3 dB step on the
/// signal side; noise is always Pseudocode-83 (`2^(6 − qscf)`).
pub fn build_aspx_real_envelope_channel(
    ch: &AspxEnvelopeScfChannel<'_>,
    qmode_env: aspx::AspxQuantStep,
    num_qmf_subbands: u32,
) -> (Vec<i32>, Vec<i32>) {
    let sig = extract_aspx_sig_envelope_indices(ch.sig, qmode_env, num_qmf_subbands);
    let noise = extract_aspx_noise_envelope_indices(ch.noise);
    (sig, noise)
}

// ====================================================================
// ASPX QMF-energy aggregator (round 240) — encoder-side
// ====================================================================
//
// The round-234 envelope extractor consumes per-`sbg` `scf_sig` /
// `scf_noise` slices the caller supplies. The natural source of those
// `scf` vectors is the HF QMF matrix `q_high` that the encoder's QMF
// analysis bank produces, aggregated over the same `(sbg, atsg)`
// borders the decoder uses on the inverse path.
//
// The decoder's [`crate::aspx::estimate_envelope_energy`] reduces
// `q_high` to a per-QMF-subband `est_sig_sb[sb][atsg]` matrix. The
// encoder needs the dual: aggregate `q_high` into the per-`sbg` /
// per-`atsg` matrix that [`crate::aspx::dequantize_sig_scf`] would
// produce on its inverse path — i.e. the average squared magnitude
// across the QMF subbands that fall inside `[sbg_borders[sbg],
// sbg_borders[sbg+1])` (absolute subband indices, NOT relative to
// `sbx`) and across the time slots in `[atsg_borders[atsg] *
// num_ts_in_ats, atsg_borders[atsg+1] * num_ts_in_ats)`. The result
// is the SBG-aggregated counterpart of Pseudocode 90's `est_sig_sb`,
// keyed `[sbg][atsg]`.
//
// The result is exactly the shape `extract_aspx_sig_envelope_indices`
// / `extract_aspx_noise_envelope_indices` accept for a single envelope
// (`atsg = 0`): the encoder picks `result[sbg][0]` per subband group
// (or runs every `atsg` for multi-envelope frames; the round-219 /
// round-226 builders only consume the leading envelope today).
//
// Refs ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocode 90 (per-QMF-subband
// version) inverted with the `sbg_*` border list of Pseudocode 91.

/// Aggregate an HF QMF matrix into per-`sbg`, per-`atsg` envelope
/// energies that mirror the SBG-grouped scale factors a decoder would
/// dequantise into. Returns `result[sbg][atsg]` keyed by SIGNAL or
/// NOISE subband-group borders.
///
/// * `q_high` — HF QMF matrix, `[absolute_sb][ts]` complex
///   (`(re, im)` pairs). Entries outside the matrix's bounds
///   contribute zero energy.
/// * `sbg_borders` — `[lo₀, lo₁, …, hi]` absolute subband borders
///   (length `num_sbg + 1`). Pseudocode 91's `sbg_sig` / `sbg_noise`.
/// * `atsg_borders` — `[a₀, a₁, …, aₙ]` ATS indices for the envelope
///   borders (length `num_atsg + 1`). Pseudocode 90's `atsg_sig` /
///   `atsg_noise`.
/// * `num_ts_in_ats` — ATS span in QMF time slots (Pseudocode 90).
/// * `sbx` — first QMF subband index covered by A-SPX (Pseudocode 90
///   reference). The function uses `sbg_borders` as absolute indices
///   but tolerates `sbg_borders[i] < sbx` by clamping to `sbx`, so
///   callers can pass spec-shaped border lists verbatim.
///
/// Empty borders, zero-span groups and zero-span ATS intervals return
/// `0.0` for the affected `(sbg, atsg)` cell — matching the decoder's
/// no-op path on those edges.
///
/// Refs ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocodes 90 + 91.
pub fn aggregate_qmf_to_sbg_atsg(
    q_high: &[Vec<(f32, f32)>],
    sbg_borders: &[u32],
    atsg_borders: &[u32],
    num_ts_in_ats: u32,
    sbx: u32,
) -> Vec<Vec<f32>> {
    let num_sbg = sbg_borders.len().saturating_sub(1);
    let num_atsg = atsg_borders.len().saturating_sub(1);
    let mut result: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sbg];
    if num_sbg == 0 || num_atsg == 0 {
        return result;
    }
    for atsg in 0..num_atsg {
        let tsa = atsg_borders[atsg] * num_ts_in_ats;
        let tsz = atsg_borders[atsg + 1] * num_ts_in_ats;
        let ts_span = tsz.saturating_sub(tsa) as f64;
        if ts_span <= 0.0 {
            continue;
        }
        for sbg in 0..num_sbg {
            let lo = sbg_borders[sbg].max(sbx) as usize;
            let hi = sbg_borders[sbg + 1].max(sbx) as usize;
            if hi <= lo {
                continue;
            }
            let band_span = (hi - lo) as f64;
            let mut acc: f64 = 0.0;
            for sb_abs in lo..hi {
                if sb_abs >= q_high.len() {
                    continue;
                }
                let row = &q_high[sb_abs];
                for ts in tsa..tsz {
                    let ts = ts as usize;
                    if ts < row.len() {
                        let (re, im) = row[ts];
                        acc += (re as f64) * (re as f64) + (im as f64) * (im as f64);
                    }
                }
            }
            // Pseudocode 90 (non-interpolation branch): per-(sb, atsg) division
            // by ts_span; SBG aggregation then divides by band_span. Combined,
            // a per-(sbg, atsg) average squared magnitude.
            result[sbg][atsg] = (acc / (band_span * ts_span)) as f32;
        }
    }
    result
}

/// Extract a single-envelope (`atsg = 0`) `scf_sig` `Vec<f32>` from an
/// HF QMF matrix by aggregating energies across the SIGNAL subband-
/// group borders. The returned vector has length `num_sbg_sig` and is
/// ready to feed [`extract_aspx_sig_envelope_indices`] or the
/// `AspxEnvelopeScfChannel::sig` slot.
///
/// `num_ts_in_ats` mirrors the decoder's constant; `sbx` is the first
/// A-SPX QMF subband index. Empty `sbg_sig_borders` returns an empty
/// vector.
pub fn extract_aspx_sig_envelope_scf_from_qmf(
    q_high: &[Vec<(f32, f32)>],
    sbg_sig_borders: &[u32],
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
) -> Vec<f32> {
    if sbg_sig_borders.len() < 2 || aspx_frame_ts_count == 0 || num_ts_in_ats == 0 {
        return Vec::new();
    }
    let atsg_borders = [0u32, aspx_frame_ts_count];
    let agg = aggregate_qmf_to_sbg_atsg(q_high, sbg_sig_borders, &atsg_borders, num_ts_in_ats, sbx);
    // One column for atsg = 0.
    agg.into_iter()
        .map(|row| row.first().copied().unwrap_or(0.0))
        .collect()
}

/// Extract a single-envelope (`atsg = 0`) `scf_noise` `Vec<f32>` from
/// an HF QMF matrix by aggregating energies across the NOISE
/// subband-group borders. Returned shape matches the SIGNAL helper.
pub fn extract_aspx_noise_envelope_scf_from_qmf(
    q_high: &[Vec<(f32, f32)>],
    sbg_noise_borders: &[u32],
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
) -> Vec<f32> {
    if sbg_noise_borders.len() < 2 || aspx_frame_ts_count == 0 || num_ts_in_ats == 0 {
        return Vec::new();
    }
    let atsg_borders = [0u32, aspx_frame_ts_count];
    let agg =
        aggregate_qmf_to_sbg_atsg(q_high, sbg_noise_borders, &atsg_borders, num_ts_in_ats, sbx);
    agg.into_iter()
        .map(|row| row.first().copied().unwrap_or(0.0))
        .collect()
}

/// Extract a single-envelope per-noise-SBG **ratio** `scf_noise`
/// vector from an HF QMF matrix — the tonal-to-noise semantics the
/// decoder's §5.7.6.4.2.1 Pseudocode 94 consumes (`noise_lev² =
/// scf_sig · noise/(1 + noise)`, i.e. `scf_noise` splits the SIGNAL
/// envelope's energy between the patched/tonal part and injected
/// noise), rather than an absolute band energy.
///
/// Per noise subband group the noise fraction is estimated from the
/// group's spectral peakiness: with `e_peak` the loudest subband's
/// mean energy and `e_total` the group total,
/// `noise_frac = 1 − e_peak / e_total` — a single dominant partial
/// (tonal content the HF generator can transpose) drives the fraction
/// to 0, while energy spread evenly across the group's `n` subbands
/// drives it to `1 − 1/n`. The coded ratio is
/// `noise_frac / (1 − noise_frac)`, which the decoder's
/// `noise/(1 + noise)` maps back onto exactly `noise_frac`. A
/// zero-energy group returns ratio 0 (the noise row then quantises to
/// the dark end of Pseudocode 83).
///
/// The estimator itself is an encoder policy (the spec leaves noise
/// estimation informative); the mapping onto the wire is the exact
/// Pseudocode 83/94 inverse.
pub fn extract_aspx_noise_ratio_scf_from_qmf(
    q_high: &[Vec<(f32, f32)>],
    sbg_noise_borders: &[u32],
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
) -> Vec<f32> {
    if sbg_noise_borders.len() < 2 || aspx_frame_ts_count == 0 || num_ts_in_ats == 0 {
        return Vec::new();
    }
    let tsz = (aspx_frame_ts_count * num_ts_in_ats) as usize;
    let num_sbg = sbg_noise_borders.len() - 1;
    let mut out = Vec::with_capacity(num_sbg);
    for sbg in 0..num_sbg {
        let lo = sbg_noise_borders[sbg].max(sbx) as usize;
        let hi = sbg_noise_borders[sbg + 1].max(sbx) as usize;
        let mut e_total = 0.0f64;
        let mut e_peak = 0.0f64;
        for row in q_high.iter().take(hi).skip(lo) {
            let mut e_sb = 0.0f64;
            for &(re, im) in row.iter().take(tsz) {
                e_sb += (re as f64) * (re as f64) + (im as f64) * (im as f64);
            }
            e_total += e_sb;
            if e_sb > e_peak {
                e_peak = e_sb;
            }
        }
        let ratio = if e_total > 0.0 && e_peak < e_total {
            let noise_frac = 1.0 - e_peak / e_total;
            (noise_frac / (1.0 - noise_frac)).max(0.0)
        } else {
            0.0
        };
        out.push(ratio as f32);
    }
    out
}

/// Predict, per NOISE subband group, the fraction of the target
/// envelope energy the decoder's HF generator can deliver from its
/// patch tiles — the encoder-side mirror of the §5.7.6.4.1.4
/// Pseudocode 89 tile mapping composed with the §5.7.6.4.2.2
/// Pseudocode 95 gain (`patched_out = est/(1+est) · scf/(1+noise)`;
/// with an (near-)empty patch source `est ≪ 1` the compensatory gain
/// cannot resurrect energy that is not there, and the envelope's
/// energy must instead be carried by injected noise).
///
/// For each regenerated subband the tile source subband is derived
/// from the same Pseudocode 71 patch tables the decoder uses, its
/// per-sample energy `est` is measured on the encoder's own low band
/// (which is what the decoder will patch from, up to core coding
/// error), and the deliverable fraction `est/(1+est)` is averaged
/// across each noise group weighted by the target subband energies.
/// Groups with a silent target return fraction 1 (nothing needs
/// compensating).
#[allow(clippy::too_many_arguments)]
pub fn predict_aspx_patch_delivery_fraction_from_qmf(
    q: &[Vec<(f32, f32)>],
    tables: &crate::aspx::AspxFrequencyTables,
    master_freq_scale_highres: bool,
    sbg_noise_borders: &[u32],
    tna_mode: &[u8],
    preflat: bool,
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
) -> Vec<f32> {
    let num_groups = sbg_noise_borders.len().saturating_sub(1);
    let mut out = vec![1.0f32; num_groups];
    let patches = crate::aspx::derive_patch_tables(
        &tables.sbg_master,
        tables.num_sbg_master,
        tables.sba,
        tables.sbx,
        tables.num_sb_aspx,
        true,
        master_freq_scale_highres,
    );
    if patches.num_sbg_patches == 0 || num_groups == 0 || tables.sba == 0 {
        return out;
    }
    let mean_energy = |sb: usize| -> f64 {
        let Some(row) = q.get(sb) else { return 0.0 };
        if row.is_empty() {
            return 0.0;
        }
        row.iter()
            .map(|&(re, im)| (re as f64) * (re as f64) + (im as f64) * (im as f64))
            .sum::<f64>()
            / row.len() as f64
    };
    // Mirror the tile-generation front half on the encoder's own low
    // band: Pseudocode 86-88 covariance → alphas → steady-state chirp
    // per group (TAB_NEW_CHIRP[t][t] — the smoothing fixed point for a
    // stationary tna decision), then the Pseudocode 89 inverse filter
    // + optional inverse pre-flatten gain, measuring the post-filter
    // (residual) source energy per subband.
    let sba = tables.sba as usize;
    let q_low: Vec<Vec<(f32, f32)>> = q.iter().take(sba).map(|row| row.to_vec()).collect();
    let q_low_ext = crate::aspx_tns::build_q_low_ext(&q_low, &[], tables.sba);
    let cov = crate::aspx_tns::compute_covariance(&q_low_ext, tables.sba);
    let (alpha0, alpha1) = crate::aspx_tns::compute_alphas(&cov);
    let preflat_gains = if preflat && aspx_frame_ts_count > 0 {
        let atsg_sig = [0u32, aspx_frame_ts_count];
        Some(crate::aspx_tns::compute_preflat_gains(
            &q_low,
            tables.sbx,
            &atsg_sig,
            num_ts_in_ats,
        ))
    } else {
        None
    };
    let resid_energy = |p: usize, c: f32| -> f64 {
        let Some(row) = q_low_ext.get(p) else {
            return 0.0;
        };
        if row.len() < 5 {
            return mean_energy(p);
        }
        let a0 = alpha0.get(p).copied().unwrap_or((0.0, 0.0));
        let a1 = alpha1.get(p).copied().unwrap_or((0.0, 0.0));
        let c2 = c * c;
        let mut acc = 0.0f64;
        let mut n_cnt = 0usize;
        for n in 4..row.len() {
            let x = row[n];
            let mut s = x;
            if c > 0.0 {
                let z = row[n - 2];
                s.0 += c * (a0.0 * z.0 - a0.1 * z.1);
                s.1 += c * (a0.0 * z.1 + a0.1 * z.0);
                let z = row[n - 4];
                s.0 += c2 * (a1.0 * z.0 - a1.1 * z.1);
                s.1 += c2 * (a1.0 * z.1 + a1.1 * z.0);
            }
            let mut e = (s.0 as f64) * (s.0 as f64) + (s.1 as f64) * (s.1 as f64);
            if let Some(gv) = &preflat_gains {
                if let Some(&g_p) = gv.get(p) {
                    if g_p != 0.0 {
                        e /= (g_p as f64) * (g_p as f64);
                    }
                }
            }
            acc += e;
            n_cnt += 1;
        }
        if n_cnt == 0 {
            0.0
        } else {
            acc / n_cnt as f64
        }
    };
    // Per regenerated subband: target energy + deliverable fraction.
    let sbz = tables.sbz as usize;
    let mut acc_num = vec![0.0f64; num_groups];
    let mut acc_den = vec![0.0f64; num_groups];
    let mut sum_sb_patches = 0u32;
    for i in 0..(patches.num_sbg_patches as usize) {
        for sb_off in 0..patches.sbg_patch_num_sb[i] {
            let sb_high = (tables.sbx + sum_sb_patches + sb_off) as usize;
            if sb_high >= sbz {
                continue;
            }
            let g = sbg_noise_borders
                .iter()
                .take(num_groups)
                .rposition(|&b| (b as usize) <= sb_high)
                .unwrap_or(0);
            let t = tna_mode.get(g).copied().unwrap_or(0).min(3) as usize;
            let c = crate::aspx_tns::TAB_NEW_CHIRP[t][t];
            let src = (patches.sbg_patch_start_sb[i] + sb_off) as usize;
            let est = resid_energy(src, c);
            let f = est / (1.0 + est);
            let target = mean_energy(sb_high);
            acc_num[g] += target * f;
            acc_den[g] += target;
        }
        sum_sb_patches += patches.sbg_patch_num_sb[i];
    }
    for g in 0..num_groups {
        if acc_den[g] > 0.0 {
            out[g] = (acc_num[g] / acc_den[g]) as f32;
        }
    }
    out
}

/// Per-channel HF QMF + SBG-border bundle consumed by
/// [`build_aspx_real_envelope_channel_from_qmf`].
///
/// Each channel's HF QMF matrix is the encoder's QMF-analysis output
/// over the high-frequency range; the SIGNAL / NOISE border lists are
/// the absolute (`sbx`-rooted) subband borders that the decoder will
/// consume on the inverse path (Pseudocode 91 `sbg_sig` /
/// `sbg_noise`).
#[derive(Debug, Clone)]
pub struct AspxQmfEnvelopeChannel<'a> {
    /// HF QMF matrix `[absolute_sb][ts]`.
    pub q_high: &'a [Vec<(f32, f32)>],
    /// SIGNAL subband-group borders (`sbg_sig`).
    pub sbg_sig_borders: &'a [u32],
    /// NOISE subband-group borders (`sbg_noise`).
    pub sbg_noise_borders: &'a [u32],
}

/// Build an [`AspxRealEnvelopeChannel`]-shaped quant-index payload
/// straight from an HF QMF matrix by chaining
/// [`extract_aspx_sig_envelope_scf_from_qmf`] +
/// [`extract_aspx_noise_envelope_scf_from_qmf`] into
/// [`build_aspx_real_envelope_channel`].
///
/// Returns the owned `(sig, noise)` `Vec<i32>` pair the round-226
/// builder pair accepts directly via slice references.
///
/// `num_ts_in_ats` mirrors the decoder's constant (Pseudocode 90);
/// `aspx_frame_ts_count` is the number of ATSs covered by a single
/// envelope (1 for the round-226 single-envelope path);
/// `num_qmf_subbands` is the dequantizer's `n_subbands` constant
/// (`64` for AC-4); `qmode_env` selects the 1.5 dB / 3 dB step on the
/// signal side.
pub fn build_aspx_real_envelope_channel_from_qmf(
    ch: &AspxQmfEnvelopeChannel<'_>,
    qmode_env: aspx::AspxQuantStep,
    num_qmf_subbands: u32,
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
) -> (Vec<i32>, Vec<i32>) {
    let sig_scf = extract_aspx_sig_envelope_scf_from_qmf(
        ch.q_high,
        ch.sbg_sig_borders,
        num_ts_in_ats,
        aspx_frame_ts_count,
        sbx,
    );
    let noise_scf = extract_aspx_noise_envelope_scf_from_qmf(
        ch.q_high,
        ch.sbg_noise_borders,
        num_ts_in_ats,
        aspx_frame_ts_count,
        sbx,
    );
    let scf_ch = AspxEnvelopeScfChannel {
        sig: &sig_scf,
        noise: &noise_scf,
    };
    build_aspx_real_envelope_channel(&scf_ch, qmode_env, num_qmf_subbands)
}

// ====================================================================
// ASPX QMF → multi-envelope builder + envelope-count selection (round 310)
// ====================================================================
//
// The round-240 QMF aggregator (`aggregate_qmf_to_sbg_atsg`) already
// reduces an HF QMF matrix to a per-`(sbg, atsg)` energy matrix, and the
// round-292 packer (`dpcm_encode_qscf_envelopes`) already turns a
// per-`(sbg, atsg)` quant matrix into the per-envelope `AspxEncodedEnvelope`
// rows the round-299 multi-envelope writers consume. What was missing was
// the glue between them: nothing built `[sbg][atsg]` *quant* matrices from
// the QMF energy across a multi-envelope FIXFIX framing, and nothing chose
// the per-frame envelope count `num_env` from the input energy.
//
// In a FIXFIX interval the signal envelopes are uniformly spaced in time
// (§4.3.10.4.1): an `aspx_num_env`-envelope frame splits the
// `aspx_frame_ts_count` ATSs into `aspx_num_env` equal envelope spans, so
// the atsg-border list passed to `aggregate_qmf_to_sbg_atsg` is the
// uniform partition `[0, span, 2·span, …, aspx_frame_ts_count]` with
// `span = aspx_frame_ts_count / aspx_num_env`. This module derives those
// borders, quantises each cell with the round-234 Pseudocode-82 / 83
// inverses, and packs the result with the round-292 packer.
//
// Refs ETSI TS 103 190-1 §4.3.10.4.1 (FIXFIX uniform envelope spacing),
// §4.3.10.4.11 (aspx_num_env / aspx_num_noise), §5.7.6.3.5 Pseudocodes
// 82 / 83, §5.7.6.3.4 Pseudocodes 80 / 81, §5.7.6.4.2.1 Pseudocodes
// 90 / 91.

/// Build the uniform FIXFIX atsg-border list for `num_env` signal
/// envelopes over `aspx_frame_ts_count` A-SPX time slot groups.
///
/// Per §4.3.10.4.1 a FIXFIX interval spaces its `aspx_num_env` signal
/// envelopes uniformly in time, so the per-envelope ATS spans partition
/// `[0, aspx_frame_ts_count)` into `num_env` equal pieces. Any remainder
/// (when `aspx_frame_ts_count` is not divisible by `num_env`) is folded
/// into the final envelope so the borders always end exactly at
/// `aspx_frame_ts_count` and cover every ATS.
///
/// Returns a length-`num_env + 1` border list, or an empty vector when
/// `num_env == 0`. The result is the `atsg_borders` argument
/// [`aggregate_qmf_to_sbg_atsg`] expects (ATS indices, scaled internally
/// by `num_ts_in_ats`).
pub fn fixfix_uniform_atsg_borders(aspx_frame_ts_count: u32, num_env: u32) -> Vec<u32> {
    if num_env == 0 {
        return Vec::new();
    }
    let span = aspx_frame_ts_count / num_env;
    let mut borders = Vec::with_capacity(num_env as usize + 1);
    for e in 0..num_env {
        borders.push(e * span);
    }
    // Final border closes at the exact frame length so the trailing
    // envelope absorbs any non-divisible remainder.
    borders.push(aspx_frame_ts_count);
    borders
}

/// Aggregate an HF QMF matrix into a per-`(sbg, atsg)` energy matrix over
/// a uniform FIXFIX `num_env`-envelope partition of the frame.
///
/// Thin wrapper over [`aggregate_qmf_to_sbg_atsg`] that derives the
/// atsg-border list with [`fixfix_uniform_atsg_borders`], so callers
/// holding the QMF matrix + a chosen `num_env` get the
/// `result[sbg][atsg]` matrix the multi-envelope packer consumes without
/// hand-rolling the uniform partition.
///
/// `num_env == 0` returns an empty matrix.
///
/// Refs ETSI TS 103 190-1 §4.3.10.4.1, §5.7.6.4.2.1 Pseudocodes 90 / 91.
pub fn aggregate_qmf_to_sbg_atsg_uniform(
    q_high: &[Vec<(f32, f32)>],
    sbg_borders: &[u32],
    num_env: u32,
    aspx_frame_ts_count: u32,
    num_ts_in_ats: u32,
    sbx: u32,
) -> Vec<Vec<f32>> {
    let atsg_borders = fixfix_uniform_atsg_borders(aspx_frame_ts_count, num_env);
    aggregate_qmf_to_sbg_atsg(q_high, sbg_borders, &atsg_borders, num_ts_in_ats, sbx)
}

/// Quantise a per-`(sbg, atsg)` SIGNAL energy matrix into the
/// `qscf[sbg][atsg]` integer matrix the round-292 packer consumes,
/// inverting Pseudocode 82 cell-by-cell with [`quantize_sig_scf`].
///
/// The input shape is the `aggregate_qmf_to_sbg_atsg*` output
/// (`[sbg][atsg]`); the output preserves that shape. Empty rows pass
/// through as empty rows.
pub fn quantize_sig_energy_matrix(
    energy: &[Vec<f32>],
    qmode_env: aspx::AspxQuantStep,
    num_qmf_subbands: u32,
) -> Vec<Vec<i32>> {
    energy
        .iter()
        .map(|row| {
            row.iter()
                .map(|&e| quantize_sig_scf(e, qmode_env, num_qmf_subbands))
                .collect()
        })
        .collect()
}

/// Quantise a per-`(sbg, atsg)` NOISE energy matrix into the
/// `qscf[sbg][atsg]` integer matrix the round-292 packer consumes,
/// inverting Pseudocode 83 cell-by-cell with [`quantize_noise_scf`].
pub fn quantize_noise_energy_matrix(energy: &[Vec<f32>]) -> Vec<Vec<i32>> {
    energy
        .iter()
        .map(|row| row.iter().map(|&e| quantize_noise_scf(e)).collect())
        .collect()
}

/// Per-channel HF QMF + SBG-border bundle consumed by
/// [`build_aspx_multi_envelope_channel_from_qmf`] and
/// [`select_aspx_num_env_from_qmf`] — the multi-envelope generalisation
/// of [`AspxQmfEnvelopeChannel`].
#[derive(Debug, Clone)]
pub struct AspxQmfMultiEnvelopeChannel<'a> {
    /// HF QMF matrix `[absolute_sb][ts]` complex `(re, im)` pairs.
    pub q_high: &'a [Vec<(f32, f32)>],
    /// SIGNAL subband-group borders (`sbg_sig`, absolute / `sbx`-rooted).
    pub sbg_sig_borders: &'a [u32],
    /// NOISE subband-group borders (`sbg_noise`).
    pub sbg_noise_borders: &'a [u32],
}

/// Build the per-channel multi-envelope SIGNAL + NOISE
/// [`AspxEncodedEnvelope`] rows straight from an HF QMF matrix for a
/// chosen FIXFIX `num_env`.
///
/// Chains the full encoder envelope pipeline end-to-end:
///
/// 1. [`aggregate_qmf_to_sbg_atsg_uniform`] → per-`(sbg, atsg)` SIGNAL
///    and NOISE energy matrices over the uniform `num_env`-envelope
///    FIXFIX partition (SIGNAL keyed by `sbg_sig_borders`; NOISE keyed
///    by `sbg_noise_borders`, collapsed to `num_noise` envelopes).
/// 2. [`quantize_sig_energy_matrix`] / [`quantize_noise_energy_matrix`]
///    → `qscf[sbg][atsg]` matrices (Pseudocode 82 / 83 inverses).
/// 3. [`dpcm_encode_qscf_envelopes`] → per-envelope `AspxEncodedEnvelope`
///    rows with per-envelope FREQ / TIME direction selection
///    (Pseudocode 80 / 81 inverses).
///
/// The NOISE side uses `num_noise = if num_env > 1 { 2 } else { 1 }`
/// envelopes (§4.3.10.4.11 / Table 128). `sig_prev_last` /
/// `noise_prev_last` are the previous frame's last-envelope `qscf` rows
/// (empty for an I-frame); they feed the TIME-direction reference of the
/// leading envelope. `delta` is the per-frame DPCM step sign (`±1`).
/// `force_freq` reproduces the single-direction scaffold (every envelope
/// coded FREQ).
///
/// Returns the `(sig_rows, noise_rows)` pair ready to drop into an
/// [`AspxMultiEnvelopeChannel`] for
/// [`write_aspx_data_1ch_multi_envelope`] /
/// [`write_aspx_data_2ch_multi_envelope`].
///
/// Refs ETSI TS 103 190-1 §4.3.10.4.1, §4.3.10.4.11, §5.7.6.3.4
/// Pseudocodes 80 / 81, §5.7.6.3.5 Pseudocodes 82 / 83, §5.7.6.4.2.1
/// Pseudocodes 90 / 91.
#[allow(clippy::too_many_arguments)]
pub fn build_aspx_multi_envelope_channel_from_qmf(
    ch: &AspxQmfMultiEnvelopeChannel<'_>,
    num_env: u32,
    qmode_env: aspx::AspxQuantStep,
    num_qmf_subbands: u32,
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
    sig_prev_last: &[i32],
    noise_prev_last: &[i32],
    delta: i32,
    force_freq: bool,
) -> (Vec<AspxEncodedEnvelope>, Vec<AspxEncodedEnvelope>) {
    let num_noise = if num_env > 1 { 2 } else { 1 };

    let sig_energy = aggregate_qmf_to_sbg_atsg_uniform(
        ch.q_high,
        ch.sbg_sig_borders,
        num_env,
        aspx_frame_ts_count,
        num_ts_in_ats,
        sbx,
    );
    let noise_energy = aggregate_qmf_to_sbg_atsg_uniform(
        ch.q_high,
        ch.sbg_noise_borders,
        num_noise,
        aspx_frame_ts_count,
        num_ts_in_ats,
        sbx,
    );

    let sig_qscf = quantize_sig_energy_matrix(&sig_energy, qmode_env, num_qmf_subbands);
    let noise_qscf = quantize_noise_energy_matrix(&noise_energy);

    let sig_rows = dpcm_encode_qscf_envelopes(&sig_qscf, sig_prev_last, delta, force_freq);
    let noise_rows = dpcm_encode_qscf_envelopes(&noise_qscf, noise_prev_last, delta, force_freq);
    (sig_rows, noise_rows)
}

/// Previous-frame last-envelope `qscf` reference rows for one channel's
/// SIGNAL + NOISE leading-envelope TIME direction, threaded into
/// [`build_aspx_multi_envelope_2ch_from_qmf`]. Both fields are empty for
/// an I-frame (no inter-frame prediction).
#[derive(Debug, Clone, Copy, Default)]
pub struct AspxMultiEnvelopePrevLast<'a> {
    /// Previous frame's last SIGNAL envelope `qscf` row (one entry per
    /// `sbg_sig`), or empty for an I-frame.
    pub sig: &'a [i32],
    /// Previous frame's last NOISE envelope `qscf` row (one entry per
    /// `sbg_noise`), or empty for an I-frame.
    pub noise: &'a [i32],
}

/// Owned per-channel multi-envelope SIGNAL + NOISE rows produced by
/// [`build_aspx_multi_envelope_2ch_from_qmf`]. The caller borrows these
/// into the [`AspxMultiEnvelopeChannel`] pair the round-299
/// [`write_aspx_data_2ch_multi_envelope`] writer consumes.
#[derive(Debug, Clone, Default)]
pub struct AspxMultiEnvelope2chRows {
    /// Channel-0 (LEVEL) SIGNAL envelope rows.
    pub ch0_sig: Vec<AspxEncodedEnvelope>,
    /// Channel-0 (LEVEL) NOISE envelope rows.
    pub ch0_noise: Vec<AspxEncodedEnvelope>,
    /// Channel-1 (BALANCE) SIGNAL envelope rows.
    pub ch1_sig: Vec<AspxEncodedEnvelope>,
    /// Channel-1 (BALANCE) NOISE envelope rows.
    pub ch1_noise: Vec<AspxEncodedEnvelope>,
    /// Absolute `qscf` row of the frame's **last** SIGNAL envelope on
    /// channel 0 (sum domain) — what the decoder's §5.7.6.3.4
    /// `qscf_sig_prev` holds after this frame, i.e. the next frame's
    /// leading-envelope TIME reference.
    pub ch0_last_sig: Vec<i32>,
    /// Last NOISE envelope row, channel 0 (sum domain).
    pub ch0_last_noise: Vec<i32>,
    /// Last SIGNAL envelope row, channel 1 (pan wire-step domain).
    pub ch1_last_sig: Vec<i32>,
    /// Last NOISE envelope row, channel 1 (pan wire-step domain).
    pub ch1_last_noise: Vec<i32>,
}

/// Build both channels' multi-envelope SIGNAL + NOISE
/// [`AspxEncodedEnvelope`] rows straight from a stereo pair of HF QMF
/// matrices for a chosen FIXFIX `num_env` — the two-channel dual of
/// [`build_aspx_multi_envelope_channel_from_qmf`].
///
/// The coupled `aspx_data_2ch()` body (Table 52, `aspx_balance = 1`)
/// carries channel 0 in the LEVEL stereo mode and channel 1 in BALANCE,
/// but the LEVEL / BALANCE distinction is purely a codebook-family
/// selection applied by the [`write_aspx_data_2ch_multi_envelope`] writer
/// when it emits each envelope — the per-`(sbg, atsg)` `qscf` quantisation
/// (Pseudocode 82 / 83) and the FREQ / TIME DPCM packing (Pseudocode
/// 80 / 81) are identical for both channels. This builder therefore runs
/// [`build_aspx_multi_envelope_channel_from_qmf`] independently per
/// channel with the same `num_env`, quant mode, and partition, and the
/// writer assigns the stereo modes.
///
/// Both channels share the frame's single FIXFIX framing
/// (`num_env` / `num_noise` / atsg partition), so they must be built with
/// the same `num_env`; pick it once per frame via
/// [`select_aspx_num_env_from_qmf`] on (typically) the LEVEL channel.
///
/// `prev0` / `prev1` carry each channel's previous-frame last-envelope
/// `qscf` rows for the leading-envelope TIME reference (empty for an
/// I-frame). `delta` is the per-frame DPCM step sign (`±1`); `force_freq`
/// reproduces the single-direction scaffold.
///
/// Refs ETSI TS 103 190-1 §4.2.12.4 Table 52 (LEVEL / BALANCE coupled
/// SIGNAL coding), §4.3.10.4.1, §4.3.10.4.11, §5.7.6.3.4 Pseudocodes
/// 80 / 81, §5.7.6.3.5 Pseudocodes 82 / 83, §5.7.6.4.2.1 Pseudocodes
/// 90 / 91.
#[allow(clippy::too_many_arguments)]
pub fn build_aspx_multi_envelope_2ch_from_qmf(
    ch0: &AspxQmfMultiEnvelopeChannel<'_>,
    ch1: &AspxQmfMultiEnvelopeChannel<'_>,
    num_env: u32,
    qmode_env: aspx::AspxQuantStep,
    num_qmf_subbands: u32,
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
    prev0: AspxMultiEnvelopePrevLast<'_>,
    prev1: AspxMultiEnvelopePrevLast<'_>,
    delta: i32,
    force_freq: bool,
) -> AspxMultiEnvelope2chRows {
    let num_noise = if num_env > 1 { 2 } else { 1 };
    // Per-channel §5.7.6.4.2.1 energy aggregation + Pseudocode 82 / 83
    // inverse quantization on the shared FIXFIX partition.
    let quantize_channel = |ch: &AspxQmfMultiEnvelopeChannel<'_>| {
        let sig_energy = aggregate_qmf_to_sbg_atsg_uniform(
            ch.q_high,
            ch.sbg_sig_borders,
            num_env,
            aspx_frame_ts_count,
            num_ts_in_ats,
            sbx,
        );
        let noise_energy = aggregate_qmf_to_sbg_atsg_uniform(
            ch.q_high,
            ch.sbg_noise_borders,
            num_noise,
            aspx_frame_ts_count,
            num_ts_in_ats,
            sbx,
        );
        (
            quantize_sig_energy_matrix(&sig_energy, qmode_env, num_qmf_subbands),
            quantize_noise_energy_matrix(&noise_energy),
        )
    };
    let (l_sig_q, l_noise_q) = quantize_channel(ch0);
    let (r_sig_q, r_noise_q) = quantize_channel(ch1);
    // §5.7.6.3.5: the aspx_balance = 1 pair transmits (sum, pan) —
    // apply the Pseudocode 84 inverse cell-wise, then DPCM-pack both
    // channels in the transmitted-symbol domain (the decoder's
    // delta = 2 accumulation on the balance channel recovers
    // qscf_b = 2 · pan). `prev0` / `prev1` carry previous-frame
    // last-envelope rows in the same (sum / pan-wire-step) domain.
    let (sum_sig_q, pan_sig_q) = balance_encode_sig_matrix(&l_sig_q, &r_sig_q, qmode_env);
    let (sum_noise_q, pan_noise_q) = balance_encode_noise_matrix(&l_noise_q, &r_noise_q);
    let ch0_sig = dpcm_encode_qscf_envelopes(&sum_sig_q, prev0.sig, delta, force_freq);
    let ch0_noise = dpcm_encode_qscf_envelopes(&sum_noise_q, prev0.noise, delta, force_freq);
    let ch1_sig = dpcm_encode_qscf_envelopes(&pan_sig_q, prev1.sig, delta, force_freq);
    let ch1_noise = dpcm_encode_qscf_envelopes(&pan_noise_q, prev1.noise, delta, force_freq);
    // The last atsg column of each transmitted matrix is the decoder's
    // post-frame `qscf_*_prev` reference (Pseudocodes 80 / 81).
    let last_col = |m: &[Vec<i32>]| -> Vec<i32> {
        m.iter()
            .map(|row| row.last().copied().unwrap_or(0))
            .collect()
    };
    AspxMultiEnvelope2chRows {
        ch0_last_sig: last_col(&sum_sig_q),
        ch0_last_noise: last_col(&sum_noise_q),
        ch1_last_sig: last_col(&pan_sig_q),
        ch1_last_noise: last_col(&pan_noise_q),
        ch0_sig,
        ch0_noise,
        ch1_sig,
        ch1_noise,
    }
}

/// Choose the per-frame FIXFIX signal-envelope count `num_env` (a power
/// of two) from an HF QMF matrix's temporal energy variation.
///
/// In FIXFIX framing the encoder is free to split the frame into 1, 2, 4
/// (or more, up to the `aspx_num_env_bits_fixfix` capacity) uniformly
/// spaced signal envelopes; more envelopes track fast temporal energy
/// change (transients) at the cost of bits, fewer envelopes are cheaper
/// for stationary content. This driver evaluates each candidate power of
/// two and picks the smallest `num_env` whose finer time partition does
/// **not** materially reduce the per-envelope energy variance versus the
/// next-coarser candidate — i.e. it stops splitting once the extra
/// envelopes stop carrying new temporal detail.
///
/// Concretely, for each candidate it computes the broadband
/// (SBG-summed) per-envelope energy vector and its coefficient of
/// variation `cv = stddev / mean`. The chosen count is the largest power
/// of two `≤ max_num_env` whose `cv` first exceeds `transient_threshold`
/// (a transient is present), falling back to `1` when no candidate clears
/// the threshold (stationary frame). `max_num_env` is clamped to the
/// config's representable maximum `1 << (1 << num_env_bits_fixfix - 1)`
/// region by the caller via [`AspxConfig::fixfix_tmp_num_env_bits`]; this
/// driver additionally bounds it to powers of two ≥ 1.
///
/// A frame with fewer than 2 ATSs, or all-zero energy, returns `1`.
///
/// Refs ETSI TS 103 190-1 §4.3.10.4.1, §4.3.10.1.9 (Table 123).
pub fn select_aspx_num_env_from_qmf(
    q_high: &[Vec<(f32, f32)>],
    sbg_sig_borders: &[u32],
    num_ts_in_ats: u32,
    aspx_frame_ts_count: u32,
    sbx: u32,
    max_num_env: u32,
    transient_threshold: f32,
) -> u32 {
    if aspx_frame_ts_count < 2 || max_num_env < 2 || sbg_sig_borders.len() < 2 {
        return 1;
    }
    // Candidate envelope counts: powers of two in 2..=max_num_env (the
    // coarsest, num_env = 1, is the fallback).
    let mut best = 1u32;
    let mut cand = 2u32;
    while cand <= max_num_env && cand <= aspx_frame_ts_count {
        if !cand.is_power_of_two() {
            cand <<= 1;
            continue;
        }
        let energy = aggregate_qmf_to_sbg_atsg_uniform(
            q_high,
            sbg_sig_borders,
            cand,
            aspx_frame_ts_count,
            num_ts_in_ats,
            sbx,
        );
        // Broadband per-envelope energy: sum across subband groups.
        let num_env = cand as usize;
        let mut per_env = vec![0.0_f64; num_env];
        for row in &energy {
            for (e, &v) in row.iter().enumerate().take(num_env) {
                per_env[e] += v as f64;
            }
        }
        let n = per_env.len() as f64;
        let mean = per_env.iter().sum::<f64>() / n;
        if mean <= 0.0 {
            // No energy in this candidate's partition — splitting cannot
            // reveal a transient; stop.
            break;
        }
        let var = per_env
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / n;
        let cv = var.sqrt() / mean;
        if cv as f32 >= transient_threshold {
            // This partition exposes a transient — keep the finer count
            // and keep probing for an even finer one.
            best = cand;
        }
        cand <<= 1;
    }
    best
}

/// Emit a minimum-viable `acpl_data_1ch()` body per §4.2.13.3 Table 61:
///
/// ```text
///   acpl_framing_data(): smooth interp + num_param_sets = 1   // 2 b
///   acpl_ec_data(ALPHA): 1 param set × acpl_huff_data()
///   acpl_ec_data(BETA):  1 param set × acpl_huff_data()
/// ```
///
/// Each `acpl_huff_data()` emits `diff_type = 0` (DIFF_FREQ) then one
/// F0 codeword + `(num_bands - start_band - 1)` DF zero-delta codewords.
/// The recovered `(alpha, beta)` per-band deltas are all 0 — the
/// minimal-cost scaffold matching the round-95 ACPL_3 emitter. Mirrors
/// [`crate::acpl::parse_acpl_data_1ch`].
fn write_acpl_data_1ch_minimal(
    bw: &mut BitWriter,
    num_bands: u32,
    start_band: u32,
    quant_mode: crate::acpl::AcplQuantMode,
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);
    // num_param_sets = 1 → each acpl_ec_data() runs one acpl_huff_data().

    let emit_one = |bw: &mut BitWriter, dt: crate::acpl::AcplDataType| {
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        if num_bands > start_band {
            write_acpl_f0_zero(bw, dt, quant_mode);
        }
        for _ in (start_band + 1)..num_bands {
            write_acpl_df_zero(bw, dt, quant_mode);
        }
    };

    // alpha1 — ALPHA codebook family.
    emit_one(bw, crate::acpl::AcplDataType::Alpha);
    // beta1 — BETA codebook family.
    emit_one(bw, crate::acpl::AcplDataType::Beta);
}

/// Build a 5_X SIMPLE/ASPX_ACPL_2 substream body per §4.2.6.6 Table 25
/// row `case ASPX_ACPL_2:` that the decoder's
/// [`crate::mch::parse_5x_audio_data_outer`] (with `mode = AspxAcpl2`)
/// walks end-to-end and synthesises 5-channel `[L, R, C, Ls, Rs]` PCM
/// via [`crate::acpl_synth::run_acpl_5x_pair_pcm`] (Pseudocode 117).
///
/// Body layout (Table 25, `coding_config = 0` — the AcplLite2 / two-
/// channel false-branch):
///
/// ```text
///   5_X_codec_mode = ASPX_ACPL_2 (3)        // 3 b
///   if (b_iframe) {
///       aspx_config();                       // 15 b — Table 50
///       acpl_config_1ch(FULL);               //  3 b — Table 59
///   }
///   companding_control(3);                   // sync = 1, on = 1 — Table 49
///   coding_config = 0;                        //  1 b
///   two_channel_data();                       // L/R carriers — Table 26
///   // (ASPX_ACPL_1 joint-MDCT residual layer is SKIPPED for ACPL_2)
///   mono_data(0);                             // centre (Cfg0 only) — Table 21
///   if (b_iframe) {
///       aspx_data_2ch();                     // Table 52
///       aspx_data_1ch();                     // Table 51
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[0]
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[1]
///   }
/// ```
///
/// `coeffs_l` / `coeffs_r` are the forward-MDCT L/R carrier spectra;
/// `coeffs_c` is the centre carrier coded via the Cfg0 `mono_data(0)`.
/// ASPX_ACPL_2 has no surround carriers — the Ls/Rs PCM is reconstructed
/// from the L/R carriers + the two `acpl_data_1ch()` parameter sets.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl2_body_from_pcm_spectra(
    transform_length: u32,
    max_sfb: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_2 (3) — 3 bits.
    bw.write_u32(3, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(FULL) (3 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }

    // companding_control(3): sync = 1, on = 1, no avg (same wire shape as
    // the 2-channel sync-on case).
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (1 b) — false → AcplLite2 / two_channel_data path.
    bw.write_bit(false);

    // two_channel_data(): L/R carriers.
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // (No ASPX_ACPL_1 residual layer for ACPL_2.)

    // Cfg0 (coding_config == 0): mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        // acpl_config_1ch(FULL) has no qmf_band → start_band = 0.
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, 0, acpl_quant_mode);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, 0, acpl_quant_mode);
    }

    finish_substream_body(bw)
}

/// Build a 5_X SIMPLE/ASPX_ACPL_2 substream body identical on the wire
/// schedule to [`build_5_x_acpl2_body_from_pcm_spectra`] but with **real
/// per-parameter-band α + β** carried by the two trailing
/// `acpl_data_1ch()` elements per ETSI TS 103 190-1 §5.7.7.5 Pseudocode
/// 116 / §5.7.7.6.1 Pseudocode 117 (round 144 — the ACPL_2 5.0 counterpart
/// to the round-132 ACPL_1 5.0 real α+β extractor).
///
/// ACPL_2 does **not** transmit the surround pair Ls/Rs on the wire (the
/// decoder reconstructs them from the L/R carriers + the two
/// `acpl_data_1ch()` parameter sets), so this builder accepts `coeffs_ls`
/// / `coeffs_rs` purely to drive the analytic α + β extractors — the
/// emitted body still carries only L/R/C and the two parameter-set
/// elements. Caller passes Ls/Rs spectra equal to the forward-MDCT of the
/// caller's desired surround signals; the encoder picks α + β per band so
/// the decoder's Pseudocode-116 reconstruction matches the surround
/// energy + cross-correlation against the L/R carriers.
///
/// Per §5.7.7.5 Pseudocode 116 with `y` ⊥ `x0` and `E[y²] ≈ E[x0²]`:
///
/// ```text
///   α   = 1 − 2·√2 · ⟨x_carrier, x_surround⟩ / ⟨x_carrier, x_carrier⟩
///   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
///   ⇒  β = √max(0, 2·E[Ls²]/E[L²] − (1 − α_dq)²)
/// ```
///
/// The decoder's [`crate::acpl_synth::differential_decode`] reverses the
/// DIFF_FREQ chain and [`crate::acpl_synth::dequantize_alpha_index`] /
/// `dequantize_beta_index` recover the per-band magnitudes. The
/// `acpl_config_1ch(FULL)` carries no `qmf_band` → `start_band = 0` so
/// every parameter band participates.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
    transform_length: u32,
    max_sfb: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    // acpl_config_1ch(FULL) carries no qmf_band → start_band = 0 (every
    // parameter band participates, in contrast to the PARTIAL ACPL_1 path
    // whose qmf_band masks the low bands).
    let start_band = 0u32;

    // α extraction — identical to the round-128 / 132 ACPL_1 helper, run
    // independently for the (L, Ls) and (R, Rs) decorrelator legs.
    let (num_l, den_l) = compute_per_band_correlations(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (num_r, den_r) = compute_per_band_correlations(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_real = analytic_alpha_per_band(&num_l, &den_l, acpl_quant_mode);
    let alpha_r_real = analytic_alpha_per_band(&num_r, &den_r, acpl_quant_mode);
    let alpha_l_q: Vec<i32> = alpha_l_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();
    let alpha_r_q: Vec<i32> = alpha_r_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();

    // β — energy residual after α removes the level-only component.
    let (e_c_l, e_s_l) = compute_per_band_energies(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (e_c_r, e_s_r) = compute_per_band_energies(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_dq: Vec<f32> = alpha_l_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let alpha_r_dq: Vec<f32> = alpha_r_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let beta_l_real = analytic_beta_per_band(&e_c_l, &e_s_l, &alpha_l_dq, acpl_quant_mode);
    let beta_r_real = analytic_beta_per_band(&e_c_r, &e_s_r, &alpha_r_dq, acpl_quant_mode);
    let beta_l_q: Vec<i32> = beta_l_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();
    let beta_r_q: Vec<i32> = beta_r_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_2 (3) — 3 bits.
    bw.write_u32(3, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }
    write_companding_control_2ch_sync_on(&mut bw);
    bw.write_bit(false); // coding_config = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// 5_X SIMPLE/ASPX_ACPL_2 body with **real** ASPX SIGNAL / NOISE
/// envelopes on both the L / R carrier pair (`aspx_data_2ch()`) and the
/// centre carrier (`aspx_data_1ch()`), plus the real per-band α / β
/// A-CPL parameters.
///
/// Identical framing to [`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta`]
/// except the I-frame's `aspx_data_2ch()` + `aspx_data_1ch()` elements
/// carry caller-provided envelope quant-index vectors (via
/// [`write_aspx_data_2ch_real_envelope`] / [`write_aspx_data_1ch_real_envelope`])
/// instead of the round-95 minimum-bit-cost
/// [`write_aspx_data_2ch_minimal`] / [`write_aspx_data_1ch_minimal`]
/// scaffolds. The L / R / C SIGNAL + NOISE rows come from
/// [`build_aspx_real_envelope_channel_from_qmf`] run over the input PCM.
///
/// `l_sig` / `l_noise` / `r_sig` / `r_noise` feed the two-channel
/// element (channel 0 = L carrier, channel 1 = R carrier); `c_sig` /
/// `c_noise` feed the single-channel centre element. Every other element
/// (companding, split-MDCT stereo carriers, the centre `mono_data`, the
/// two real-α/β `acpl_data_1ch()` parameter sets) stays byte-for-byte
/// identical to the round-144 builder, so the round-trip continues to
/// parse through [`crate::asf::parse_aspx_data_2ch_body`] +
/// [`crate::asf::parse_aspx_data_1ch_body`] and synthesise 5-channel PCM.
///
/// Refs ETSI TS 103 190-1 §4.2.6.6 Table 25 (`case ASPX_ACPL_2:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
/// (`aspx_data_2ch()`).
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx(
    transform_length: u32,
    max_sfb: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    l_sig: &[i32],
    l_noise: &[i32],
    r_sig: &[i32],
    r_noise: &[i32],
    c_sig: &[i32],
    c_noise: &[i32],
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let start_band = 0u32;

    // α / β extraction — identical to the round-144 real-α/β builder.
    let (num_l, den_l) = compute_per_band_correlations(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (num_r, den_r) = compute_per_band_correlations(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_real = analytic_alpha_per_band(&num_l, &den_l, acpl_quant_mode);
    let alpha_r_real = analytic_alpha_per_band(&num_r, &den_r, acpl_quant_mode);
    let alpha_l_q: Vec<i32> = alpha_l_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();
    let alpha_r_q: Vec<i32> = alpha_r_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();

    let (e_c_l, e_s_l) = compute_per_band_energies(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (e_c_r, e_s_r) = compute_per_band_energies(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_dq: Vec<f32> = alpha_l_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let alpha_r_dq: Vec<f32> = alpha_r_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let beta_l_real = analytic_beta_per_band(&e_c_l, &e_s_l, &alpha_l_dq, acpl_quant_mode);
    let beta_r_real = analytic_beta_per_band(&e_c_r, &e_s_r, &alpha_r_dq, acpl_quant_mode);
    let beta_l_q: Vec<i32> = beta_l_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();
    let beta_r_q: Vec<i32> = beta_r_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_2 (3) — 3 bits.
    bw.write_u32(3, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }
    write_companding_control_2ch_sync_on(&mut bw);
    bw.write_bit(false); // coding_config = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    if b_iframe {
        write_aspx_data_2ch_real_envelope(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: l_sig,
                noise: l_noise,
            },
            AspxRealEnvelopeChannel {
                sig: r_sig,
                noise: r_noise,
            },
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_1ch_real_envelope(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: c_sig,
                noise: c_noise,
            },
        )
        .expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// `aspx_tna_mode`-carrying variant of
/// [`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`].
///
/// Identical to the base ACPL_2 real-ASPX builder except all three A-SPX
/// trailers carry caller-supplied real per-noise-subband-group
/// `aspx_tna_mode` inverse-filtering vectors instead of the all-zero
/// scaffold: the L / R carrier-pair `aspx_data_2ch()` is routed through
/// [`write_aspx_data_2ch_real_envelope_tna`] (channel 0 = `lr_tna_mode`,
/// mirrored to channel 1 under `aspx_balance = 1`), and the centre carrier
/// `aspx_data_1ch()` through [`write_aspx_data_1ch_real_envelope_tna`] with
/// `c_tna_mode`. Each carrier's `tna_mode` is derived independently by the
/// encoder from that carrier's own QMF low band, so the centre vector need
/// not match the front-pair vector.
///
/// Passing empty (or all-zero) `lr_tna_mode` / `c_tna_mode` slices
/// reproduces the
/// [`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`]
/// bytes exactly — the only wire difference is the 2-bit-per-noise-SBG
/// `aspx_tna_mode` field the decoder's `parse_aspx_hfgen_iwc_{1,2}ch`
/// recovers and feeds into the §5.7.6.4.1.3 chirp / order-2 LPC inverse
/// filtering.
///
/// Refs ETSI TS 103 190-1 §4.2.6.6 Table 25 (`case ASPX_ACPL_2:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
/// (`aspx_data_2ch()`), §4.3.10.6.1 Table 131 (`aspx_tna_mode`).
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
    transform_length: u32,
    max_sfb: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    l_sig: &[i32],
    l_noise: &[i32],
    r_sig: &[i32],
    r_noise: &[i32],
    c_sig: &[i32],
    c_noise: &[i32],
    lr_tna_mode: &[u8],
    c_tna_mode: &[u8],
    l_ah: &[bool],
    r_ah: &[bool],
    c_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let start_band = 0u32;

    // α / β extraction — identical to the base real-α/β/real-ASPX builder.
    let (num_l, den_l) = compute_per_band_correlations(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (num_r, den_r) = compute_per_band_correlations(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_real = analytic_alpha_per_band(&num_l, &den_l, acpl_quant_mode);
    let alpha_r_real = analytic_alpha_per_band(&num_r, &den_r, acpl_quant_mode);
    let alpha_l_q: Vec<i32> = alpha_l_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();
    let alpha_r_q: Vec<i32> = alpha_r_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();

    let (e_c_l, e_s_l) = compute_per_band_energies(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (e_c_r, e_s_r) = compute_per_band_energies(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_dq: Vec<f32> = alpha_l_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let alpha_r_dq: Vec<f32> = alpha_r_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let beta_l_real = analytic_beta_per_band(&e_c_l, &e_s_l, &alpha_l_dq, acpl_quant_mode);
    let beta_r_real = analytic_beta_per_band(&e_c_r, &e_s_r, &alpha_r_dq, acpl_quant_mode);
    let beta_l_q: Vec<i32> = beta_l_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();
    let beta_r_q: Vec<i32> = beta_r_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_2 (3) — 3 bits.
    bw.write_u32(3, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }
    write_companding_control_2ch_sync_on(&mut bw);
    bw.write_bit(false); // coding_config = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // Data elements are present on every frame per Tables 25/33;
    // only the configs and the per-element 3-bit xover offset are
    // I-frame-gated (framed writers skip the xover when
    // b_iframe == 0; the decoder reuses the sticky value).
    {
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: l_sig,
                noise: l_noise,
            },
            AspxRealEnvelopeChannel {
                sig: r_sig,
                noise: r_noise,
            },
            lr_tna_mode,
            l_ah,
            r_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_1ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: c_sig,
                noise: c_noise,
            },
            c_tna_mode,
            c_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// Multi-envelope **centre-carrier** variant of
/// [`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`].
///
/// The 5_X ASPX_ACPL_2 body carries three A-SPX trailers — `aspx_data_2ch()`
/// for the L / R carrier pair and `aspx_data_1ch()` for the centre carrier.
/// The single-envelope builder emits a `num_env = 1` FIXFIX envelope on the
/// centre via [`write_aspx_data_1ch_real_envelope`]. This variant instead
/// emits a **multi-envelope** `aspx_data_1ch()` (`num_env > 1`) for the
/// centre via [`write_aspx_data_1ch_multi_envelope`] when the caller's
/// transient probe selected `num_env > 1` — the 1-channel dual of the
/// round-299/331 5_X ACPL_3 multi-envelope `aspx_data_2ch()` path. The
/// L / R carrier pair retains its single-envelope `aspx_data_2ch()` (the
/// decoder reads `num_env` independently per A-SPX element, so a
/// multi-envelope centre coexists with a single-envelope front pair).
///
/// `c_num_env` must be a power of two within the config's FIXFIX
/// `tmp_num_env` capacity, and `centre` carries the per-envelope SIGNAL /
/// NOISE DPCM rows from [`build_aspx_multi_envelope_channel_from_qmf`].
/// Returns an **empty `Vec`** when [`write_aspx_data_1ch_multi_envelope`]
/// rejects the config / `num_env` combination (so the caller can fall back
/// to the single-envelope builder), exactly like the round-299 5_X ACPL_3
/// multi-env builder.
///
/// Every other element (α / β extraction, companding, the two-channel
/// carriers, the centre `mono_data`, the L / R `aspx_data_2ch()`, the two
/// real-α/β `acpl_data_1ch()` parameter sets) is byte-for-byte identical to
/// the single-envelope builder.
///
/// Refs ETSI TS 103 190-1 §4.2.6.6 Table 25 (`case ASPX_ACPL_2:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.3.10.4.11 (`aspx_num_env`).
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env(
    transform_length: u32,
    max_sfb: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    l_sig: &[i32],
    l_noise: &[i32],
    r_sig: &[i32],
    r_noise: &[i32],
    c_num_env: u32,
    centre: AspxMultiEnvelopeChannel<'_>,
    l_ah: &[bool],
    r_ah: &[bool],
    c_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let start_band = 0u32;

    // α / β extraction — identical to the single-envelope builder.
    let (num_l, den_l) = compute_per_band_correlations(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (num_r, den_r) = compute_per_band_correlations(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_real = analytic_alpha_per_band(&num_l, &den_l, acpl_quant_mode);
    let alpha_r_real = analytic_alpha_per_band(&num_r, &den_r, acpl_quant_mode);
    let alpha_l_q: Vec<i32> = alpha_l_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();
    let alpha_r_q: Vec<i32> = alpha_r_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();

    let (e_c_l, e_s_l) = compute_per_band_energies(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (e_c_r, e_s_r) = compute_per_band_energies(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_dq: Vec<f32> = alpha_l_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let alpha_r_dq: Vec<f32> = alpha_r_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let beta_l_real = analytic_beta_per_band(&e_c_l, &e_s_l, &alpha_l_dq, acpl_quant_mode);
    let beta_r_real = analytic_beta_per_band(&e_c_r, &e_s_r, &alpha_r_dq, acpl_quant_mode);
    let beta_l_q: Vec<i32> = beta_l_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();
    let beta_r_q: Vec<i32> = beta_r_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_2 (3) — 3 bits.
    bw.write_u32(3, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }
    write_companding_control_2ch_sync_on(&mut bw);
    bw.write_bit(false); // coding_config = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // Data elements are present on every frame per Tables 25/33;
    // only the configs and the per-element 3-bit xover offset are
    // I-frame-gated (framed writers skip the xover when
    // b_iframe == 0; the decoder reuses the sticky value).
    {
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: l_sig,
                noise: l_noise,
            },
            AspxRealEnvelopeChannel {
                sig: r_sig,
                noise: r_noise,
            },
            &[],
            l_ah,
            r_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        // Multi-envelope centre `aspx_data_1ch()`. A config / num_env
        // rejection signals the caller to fall back to the single-envelope
        // builder (mirrors the round-299 5_X ACPL_3 multi-env path).
        if write_aspx_data_1ch_multi_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            c_num_env,
            centre,
            &[],
            c_ah,
            b_iframe,
        )
        .is_err()
        {
            return Vec::new();
        }
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

// ====================================================================
// ASPX_ACPL_1 emitters — §4.2.6.6 Table 25 row `case ASPX_ACPL_1:`
// (round 103)
// ====================================================================

/// Emit an `acpl_config_1ch()` element in PARTIAL mode per §4.2.13.1
/// Table 59: 2-bit `acpl_num_param_bands_id` + 1-bit `acpl_quant_mode` +
/// 3-bit `acpl_qmf_band_minus1`. PARTIAL mode carries the extra
/// `acpl_qmf_band_minus1` field that FULL mode omits — that 3-bit field
/// is the structural difference between the ASPX_ACPL_1 (PARTIAL) and
/// ASPX_ACPL_2 (FULL) `acpl_config_1ch()` calls. Total: 6 bits. The
/// decoder's [`crate::acpl::parse_acpl_config_1ch`] with
/// [`crate::acpl::Acpl1chMode::Partial`] reads exactly this ordering and
/// resolves `qmf_band = acpl_qmf_band_minus1 + 1`.
pub fn write_acpl_config_1ch_partial(
    bw: &mut BitWriter,
    num_param_bands_id: u8,
    quant_mode: crate::acpl::AcplQuantMode,
    qmf_band_minus1: u8,
) {
    bw.write_u32(num_param_bands_id as u32 & 0b11, 2);
    bw.write_bit(matches!(quant_mode, crate::acpl::AcplQuantMode::Coarse));
    bw.write_u32(qmf_band_minus1 as u32 & 0b111, 3);
}

/// Emit the ASPX_ACPL_1-only joint-MDCT residual layer per §4.2.6.6
/// Table 25 (`case ASPX_ACPL_1:` arm, after the channel data):
///
/// ```text
///   max_sfb_master              // n_side bits — Table 106 column
///   chparam_info(): sap_mode=0  // 2 b — residual ch0
///   chparam_info(): sap_mode=0  // 2 b — residual ch1
///   sf_data(ASF)                // residual ch0 spectrum (sSMP,3)
///   sf_data(ASF)                // residual ch1 spectrum (sSMP,4)
/// ```
///
/// The two residual `sf_data(ASF)` bodies carry the joint-MDCT residual
/// spectra the decoder IMDCTs into the Ls/Rs surround PCM carriers
/// (Table 181 sSMP,3 / sSMP,4). Both bodies share the same long-frame
/// transform length and the explicit `max_sfb_master` band bound.
/// Mirrors the decoder's residual-layer walk in
/// [`crate::mch::parse_aspx_acpl_1_2_inner_body`].
///
/// `max_sfb_master` is clamped to the band budget at `transform_length`
/// and to the `n_side`-bit field width. Returns the clamped value the
/// decoder will recover.
fn write_acpl_1_residual_layer(
    bw: &mut BitWriter,
    transform_length: u32,
    max_sfb_master: u32,
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
) -> u32 {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let (_n_msfb, n_side, _n_msfbl) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    let num_sfb_cap = crate::tables::num_sfb_48(transform_length).expect("encoder: bad tl");
    let n_side_cap = (1u32 << n_side) - 1;
    // The decoder bails on max_sfb_master == 0; keep at least 1 band.
    let max_sfb_master = max_sfb_master.clamp(1, num_sfb_cap.min(n_side_cap));

    // max_sfb_master in n_side bits.
    bw.write_u32(max_sfb_master, n_side);
    // Two chparam_info() calls — one per residual channel, sap_mode = 0.
    bw.write_u32(0, 2);
    bw.write_u32(0, 2);
    // Two sf_data(ASF) bodies bounded by max_sfb_master.
    for coeffs in [coeffs_ls, coeffs_rs] {
        let (qspec, sf, max_q, sections, snf) =
            prepare_stereo_channel(coeffs, sfbo, max_sfb_master);
        write_section_data(bw, &sections);
        write_spectral_data_sections(bw, &qspec, sfbo, &sections);
        write_scalefac_data(bw, &sf, &sections.sfb_cb, &max_q, max_sfb_master);
        write_snf_data(bw, snf.as_deref(), &sections.sfb_cb, &max_q, max_sfb_master);
    }
    max_sfb_master
}

/// SAP-aware ASPX_ACPL_1 joint-MDCT residual-layer writer per §4.2.6.6
/// Table 25 (`case ASPX_ACPL_1:` arm). Generalises
/// [`write_acpl_1_residual_layer`] from the hard-coded identity
/// `sap_mode = 0` case to the full Table-47 / Table-181 SAP family
/// (identity / M/S / SAP-coded `alpha_q`) by:
///
/// 1. Emitting the caller-supplied `chparam_info()` pair via
///    [`crate::encoder_asf::write_chparam_info`] — bit-for-bit equal to
///    what the decoder's [`crate::asf::parse_chparam_info`] then walks.
/// 2. Recovering the joint-MDCT preliminary residual spectra
///    `(sSMP,3, sSMP,4)` from the desired `(L, R, Ls, Rs)` preliminary
///    set via [`crate::asf::invert_sap_table_181`]. Inside the SAP-coded
///    extent (`sfb < max_sfb_master`) the inverse uses the closed-form
///    2x2 inverse driven by the chparam_info pair's `(a, b, c, d)`;
///    outside it the inverse passes `s3 = s4 = 0` mirroring the forward
///    path's surround-silent convention.
/// 3. Writing the two sf_data(ASF) bodies for the recovered s3 / s4
///    spectra bounded by `max_sfb_master`.
///
/// When `chparam_pair` is `None` or all-identity (`sap_mode == 0` on
/// both rows) the emitted body is bit-equivalent to
/// [`write_acpl_1_residual_layer`] fed with `coeffs_ls` / `coeffs_rs`
/// directly — the inverse for the identity row reduces to
/// `s3 = ls, s4 = rs` (proven in
/// `asf::invert_sap_table_181_identity_passthrough`).
///
/// The two preliminary L/R carrier spectra are needed for the inverse
/// even when only Ls/Rs are being expressed as residual: the M/S and
/// SAP-coded rows mix L into s3 and R into s4. Callers driving the
/// existing identity-SAP path can pass dummy L/R slices and a `None`
/// chparam pair; the new ergonomic path is documented in
/// [`build_5_x_acpl1_body_from_pcm_spectra_sap`] which forwards the
/// caller's full `(L, R, Ls, Rs)` preliminaries.
///
/// `max_sfb_master` is clamped the same way as
/// [`write_acpl_1_residual_layer`]. Returns the clamped value.
#[allow(clippy::too_many_arguments)]
fn write_acpl_1_residual_layer_sap(
    bw: &mut BitWriter,
    transform_length: u32,
    max_sfb_master: u32,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    chparam_pair: Option<&[crate::asf::ChparamInfo; 2]>,
) -> u32 {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .expect("encoder: unsupported transform_length");
    let (_n_msfb, n_side, _n_msfbl) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    let num_sfb_cap = crate::tables::num_sfb_48(transform_length).expect("encoder: bad tl");
    let n_side_cap = (1u32 << n_side) - 1;
    let max_sfb_master = max_sfb_master.clamp(1, num_sfb_cap.min(n_side_cap));

    // max_sfb_master in n_side bits.
    bw.write_u32(max_sfb_master, n_side);

    // Resolve the chparam pair to emit + drive the inverse. When the
    // caller passes `None` we fall back to two identity rows — same as
    // `write_acpl_1_residual_layer`'s `sap_mode = 0` × 2 emission.
    let identity_pair = [
        crate::asf::ChparamInfo {
            sap_mode: 0,
            ms_used: vec![],
            sap_data: None,
        },
        crate::asf::ChparamInfo {
            sap_mode: 0,
            ms_used: vec![],
            sap_data: None,
        },
    ];
    let pair: &[crate::asf::ChparamInfo; 2] = chparam_pair.unwrap_or(&identity_pair);

    // Emit chparam_info() × 2 — the decoder's residual-layer walker
    // parses these via `parse_chparam_info` with the same
    // `max_sfb_per_group = [max_sfb_master]` we hand the writer.
    let max_sfb_per_group = [max_sfb_master];
    crate::encoder_asf::write_chparam_info(bw, &pair[0], &max_sfb_per_group);
    crate::encoder_asf::write_chparam_info(bw, &pair[1], &max_sfb_per_group);

    // Recover the residual `(sSMP,3, sSMP,4)` from `(L, R, Ls, Rs)` via
    // the Table-181 inverse. The inverse needs the *full* tl-length
    // spectra; we feed it the long-frame coefficient slices padded to
    // `tl` if they're shorter so the call stays total.
    let n = transform_length as usize;
    let pad = |src: &[f32]| -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        let take = src.len().min(n);
        v[..take].copy_from_slice(&src[..take]);
        v
    };
    let l_pad = pad(coeffs_l);
    let r_pad = pad(coeffs_r);
    let ls_pad = pad(coeffs_ls);
    let rs_pad = pad(coeffs_rs);
    let (s3, s4) = match crate::asf::invert_sap_table_181(
        &l_pad,
        &r_pad,
        &ls_pad,
        &rs_pad,
        pair,
        max_sfb_master,
        transform_length,
    ) {
        Some((_a, _b, s3, s4)) => (s3, s4),
        // Inverse refused (e.g. unsupported tl) — fall back to the
        // identity convention (s3 = ls, s4 = rs) so the writer stays
        // total and the body still parses.
        None => (ls_pad.clone(), rs_pad.clone()),
    };

    // Two sf_data(ASF) bodies bounded by max_sfb_master — same shape as
    // `write_acpl_1_residual_layer`.
    for coeffs in [&s3, &s4] {
        let (qspec, sf, max_q, sections, snf) =
            prepare_stereo_channel(coeffs, sfbo, max_sfb_master);
        write_section_data(bw, &sections);
        write_spectral_data_sections(bw, &qspec, sfbo, &sections);
        write_scalefac_data(bw, &sf, &sections.sfb_cb, &max_q, max_sfb_master);
        write_snf_data(bw, snf.as_deref(), &sections.sfb_cb, &max_q, max_sfb_master);
    }
    max_sfb_master
}

/// Build a 5_X SIMPLE/ASPX_ACPL_1 substream body per §4.2.6.6 Table 25
/// row `case ASPX_ACPL_1:` that the decoder's
/// [`crate::mch::parse_5x_audio_data_outer`] (with `mode = AspxAcpl1`)
/// walks end-to-end and synthesises 5-channel `[L, R, C, Ls, Rs]` PCM
/// via [`crate::acpl_synth::run_acpl_5x_pair_pcm`] (Pseudocode 117).
///
/// Body layout (Table 25, `coding_config = 0` — the AcplLite2 / two-
/// channel false-branch):
///
/// ```text
///   5_X_codec_mode = ASPX_ACPL_1 (2)        // 3 b
///   if (b_iframe) {
///       aspx_config();                       // 15 b — Table 50
///       acpl_config_1ch(PARTIAL);            //  6 b — Table 59
///   }
///   companding_control(3);                   // sync = 1, on = 1 — Table 49
///   coding_config = 0;                        //  1 b
///   two_channel_data();                       // L/R carriers — Table 26
///   max_sfb_master;                           // joint-MDCT residual layer
///   chparam_info(); chparam_info();           // residual ch0 / ch1
///   sf_data(ASF); sf_data(ASF);               // residual sSMP,3 / sSMP,4
///   mono_data(0);                             // centre (Cfg0 only) — Table 21
///   if (b_iframe) {
///       aspx_data_2ch();                     // Table 52
///       aspx_data_1ch();                     // Table 51
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[0]
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[1]
///   }
/// ```
///
/// `coeffs_l` / `coeffs_r` are the forward-MDCT L/R carrier spectra;
/// `coeffs_c` is the centre carrier coded via the Cfg0 `mono_data(0)`;
/// `coeffs_ls` / `coeffs_rs` are the Ls/Rs surround spectra coded as the
/// joint-MDCT residual pair (sSMP,3 / sSMP,4). Unlike ASPX_ACPL_2 (which
/// reconstructs Ls/Rs purely from the L/R carriers + the `acpl_data_1ch`
/// parameter pair), ASPX_ACPL_1 transmits the surround residual
/// explicitly, so it accepts a full 5-channel `[L, R, C, Ls, Rs]` input.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl1_body_from_pcm_spectra(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_1 (2) — 3 bits.
    bw.write_u32(2, 3);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(PARTIAL) (6 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }

    // companding_control(3): sync = 1, on = 1.
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (1 b) — false → AcplLite2 / two_channel_data path.
    bw.write_bit(false);

    // two_channel_data(): L/R carriers.
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // ASPX_ACPL_1 joint-MDCT residual layer: Ls/Rs surround residual.
    write_acpl_1_residual_layer(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_ls,
        coeffs_rs,
    );

    // Cfg0 (coding_config == 0): mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        // PARTIAL acpl_config_1ch carries a qmf_band → resolve start_band.
        let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
        let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
    }

    finish_substream_body(bw)
}

/// SAP-aware variant of [`build_5_x_acpl1_body_from_pcm_spectra`] —
/// emits the same Table-25 `case ASPX_ACPL_1:` body but drives the
/// joint-MDCT residual layer through [`write_acpl_1_residual_layer_sap`]
/// with the caller's `chparam_info()` pair so the resulting body
/// expresses the desired `(L, R, Ls, Rs)` preliminary set after the
/// decoder's Table-181 forward mixing fires (round 257).
///
/// The carrier `two_channel_data()` payload is still driven by the
/// L/R preliminary spectra `coeffs_l` / `coeffs_r` — the Table-181
/// inverse only reshapes the residual pair `(s3, s4)` so that the
/// decoder's `apply_sap_table_181` reproduces the requested
/// `(L_pre, R_pre, Ls_pre, Rs_pre)`. When `chparam_pair = None` (or
/// both rows carry `sap_mode = 0`) the emitted body is bit-equivalent
/// to [`build_5_x_acpl1_body_from_pcm_spectra`] called with the same
/// spectra (identity SAP — `s3 = ls`, `s4 = rs`, surround silent past
/// `max_sfb_master`).
///
/// The decoder's `parse_5x_audio_data_outer` ASPX_ACPL_1 walker
/// recovers the chparam_info pair into `tools.acpl_1_residual_chparam`
/// (round 41) and the residual spectra into
/// `tools.acpl_1_residual_pair[0..1]`, both of which the round-30
/// per-frame dispatcher pipes into `apply_sap_table_181` so the
/// surround spectra hit IMDCT with the correctly mixed contents.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl1_body_from_pcm_spectra_sap(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    chparam_pair: Option<&[crate::asf::ChparamInfo; 2]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_1 (2) — 3 bits.
    bw.write_u32(2, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }

    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (false → AcplLite2 / two_channel_data path).
    bw.write_bit(false);

    // two_channel_data(): L/R carriers (identity SAP — header chparam
    // emitted by `write_two_channel_data` is sap_mode = 0).
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);

    // ASPX_ACPL_1 joint-MDCT residual layer — SAP-aware path.
    write_acpl_1_residual_layer_sap(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_l,
        coeffs_r,
        coeffs_ls,
        coeffs_rs,
        chparam_pair,
    );

    // Cfg0 (coding_config == 0): mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
        let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
    }

    finish_substream_body(bw)
}

/// Decision-driven `chparam_info()` pair selector for the ASPX_ACPL_1
/// joint-MDCT residual layer (round 279) — wires the round-271
/// [`crate::asf::select_alpha_q_for_pair`] decision driver into the
/// round-257 SAP-aware residual writer.
///
/// Per §5.3.4.3.2 / Table 181 the residual layer's two `chparam_info()`
/// payloads drive two independent 2x2 SAP systems: `chparam_pair[0]`
/// maps the transmitted `(sSMP_A, sSMP_3)` tracks to the preliminary
/// `(L, Ls)` pair and `chparam_pair[1]` maps `(sSMP_B, sSMP_4)` to
/// `(R, Rs)`. The §5.3.2 Pseudocode 59 SAP-coded arm reconstructs each
/// output pair via `(a, b, c, d) = (1 + g, 1, 1 - g, -1)` with
/// `g = alpha_q · 0.1`, so the per-pair decision input for
/// [`crate::asf::select_alpha_q_for_pair`] is the *target output pair*
/// — `(L, Ls)` for row 0 and `(R, Rs)` for row 1 — and the transmitted
/// residual is the least-squares side prediction error
/// `s3 = S − g·M` (`M = (L + Ls) / 2`, `S = (L − Ls) / 2`).
///
/// Per pair the selector:
///
/// 1. Runs [`crate::asf::select_alpha_q_for_pair`] over the
///    single-window-group residual layout (`max_sfb_per_group =
///    [max_sfb_master]`, sfb offsets at `transform_length` — the
///    residual layer runs on the dominant transform length per the
///    §4.2.6.6 NOTE).
/// 2. Clamps the picked `alpha_q` to `[-30, +30]` so the pair-major
///    DPCM deltas Pseudocode 59 accumulates stay within the
///    HCB_SCALEFAC-codable `[-60, +60]` range even on a worst-case
///    sign flip between adjacent pairs.
/// 3. Builds the `SapMode::SapData` row via
///    [`crate::asf::build_chparam_info_sap_data_from_alpha_q`]
///    (`delta_code_time = false` — single group) when at least one
///    band raised `sap_coeff_used`; otherwise falls back to the
///    header-only [`crate::asf::build_chparam_info_none`] row so no
///    `sap_data()` body bits are spent where prediction offers no
///    benefit.
///
/// `max_sfb_master` must be the post-clamp residual band budget (the
/// caller mirrors [`write_acpl_1_residual_layer_sap`]'s clamp) so the
/// produced rows match the bound the writer hands
/// [`crate::encoder_asf::write_chparam_info`]. Returns two identity
/// (`SapMode::None`) rows when `transform_length` has no SFB table.
pub fn select_acpl1_residual_chparam_pair(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    max_sfb_master: u32,
    transform_length: u32,
) -> [crate::asf::ChparamInfo; 2] {
    let Some(sfbo) = crate::sfb_offset::sfb_offset_48(transform_length) else {
        return [
            crate::asf::build_chparam_info_none(),
            crate::asf::build_chparam_info_none(),
        ];
    };
    let max_sfb_per_group = [max_sfb_master];
    let build_row = |front: &[f32], surround: &[f32]| -> crate::asf::ChparamInfo {
        let (mut alpha_q, used) = crate::asf::select_alpha_q_for_pair(
            &[front.to_vec()],
            &[surround.to_vec()],
            sfbo,
            &max_sfb_per_group,
        );
        // Clamp to ±30: limits the worst-case pair-to-pair DPCM delta
        // to 60, the HCB_SCALEFAC codebook bound (see doc above).
        for row in alpha_q.iter_mut() {
            for v in row.iter_mut() {
                *v = (*v).clamp(-30, 30);
            }
        }
        let any_used = used.iter().any(|row| row.iter().any(|&b| b));
        if any_used {
            crate::asf::build_chparam_info_sap_data_from_alpha_q(
                &alpha_q,
                &used,
                false,
                &max_sfb_per_group,
            )
        } else {
            crate::asf::build_chparam_info_none()
        }
    };
    [
        build_row(coeffs_l, coeffs_ls),
        build_row(coeffs_r, coeffs_rs),
    ]
}

/// Fully automatic SAP-coded variant of
/// [`build_5_x_acpl1_body_from_pcm_spectra`] (round 279) — same
/// argument surface as the identity-SAP builder, but the joint-MDCT
/// residual layer's `chparam_info()` pair is *derived* from the input
/// spectra via [`select_acpl1_residual_chparam_pair`] and — unlike the
/// round-257 [`build_5_x_acpl1_body_from_pcm_spectra_sap`], which
/// still transmitted the raw L/R preliminaries as carriers — the
/// `two_channel_data()` payload carries the Table-181 **matrix-input**
/// carriers `(sSMP_A, sSMP_B)` recovered through
/// [`crate::asf::invert_sap_table_181`]. On a SAP-coded band the
/// transmitted pair is therefore `(M, S − g·M)` (mid + side prediction
/// residual) and the decoder's `apply_sap_table_181` forward mix
/// reproduces the requested `(L, R, Ls, Rs)` preliminaries exactly (up
/// to sf_data quantisation):
///
/// ```text
///   L  = (1 + g)·A + s3            A  = (L + Ls) / 2 = M
///   Ls = (1 − g)·A − s3    with    s3 = S − g·M,  S = (L − Ls) / 2
/// ```
///
/// When the selector picks no SAP band on either pair (e.g. `Ls = L`,
/// `Rs = R` — zero side energy ⇒ `g* = 0`) both rows fall back to
/// `SapMode::None`, the inverse degenerates to `A = L, B = R,
/// s3 = Ls, s4 = Rs`, and the emitted body is bit-for-bit identical to
/// [`build_5_x_acpl1_body_from_pcm_spectra`] — the strict-superset
/// invariant pinned by `build_5_x_acpl1_body_sap_auto_identity_matches_legacy`.
///
/// For a surround pair correlated with its front carrier
/// (`Ls = κ·L`) the optimal projection `g* = (1 − κ) / (1 + κ)` drives
/// the transmitted residual `s3 = S − g*·M` to zero, so the residual
/// sf_data quantises to (near-)silence and the surround content rides
/// the carrier + `alpha_q` for free — the measurable bit-efficiency
/// win of SAP coding over the identity path (which spends a full
/// sf_data body on the raw `Ls` spectrum).
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl1_body_from_pcm_spectra_sap_auto(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    // Mirror `write_acpl_1_residual_layer_sap`'s max_sfb_master clamp so
    // the selector's rows and the carrier inverse share the writer's
    // effective band budget.
    let (_n_msfb, n_side, _n_msfbl) =
        crate::tables::n_msfb_bits_48(transform_length).expect("encoder: bad tl");
    let num_sfb_cap = crate::tables::num_sfb_48(transform_length).expect("encoder: bad tl");
    let n_side_cap = (1u32 << n_side) - 1;
    let max_sfb_master = max_sfb_master.clamp(1, num_sfb_cap.min(n_side_cap));

    // Decision pass: derive the residual chparam_info() pair from the
    // target (L, Ls) / (R, Rs) preliminary pairs.
    let chparam_pair = select_acpl1_residual_chparam_pair(
        coeffs_l,
        coeffs_r,
        coeffs_ls,
        coeffs_rs,
        max_sfb_master,
        transform_length,
    );

    // Carrier pass: recover the Table-181 matrix-input carriers
    // (sSMP_A, sSMP_B) so the decoder's forward mix lands on the
    // requested preliminaries. The inverse needs tl-length spectra.
    let n = transform_length as usize;
    let pad = |src: &[f32]| -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        let take = src.len().min(n);
        v[..take].copy_from_slice(&src[..take]);
        v
    };
    let l_pad = pad(coeffs_l);
    let r_pad = pad(coeffs_r);
    let ls_pad = pad(coeffs_ls);
    let rs_pad = pad(coeffs_rs);
    let (carrier_a, carrier_b) = match crate::asf::invert_sap_table_181(
        &l_pad,
        &r_pad,
        &ls_pad,
        &rs_pad,
        &chparam_pair,
        max_sfb_master,
        transform_length,
    ) {
        Some((a, b, _s3, _s4)) => (a, b),
        // Inverse refused — fall back to raw L/R carriers (identity
        // convention) so the writer stays total.
        None => (l_pad.clone(), r_pad.clone()),
    };

    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_1 (2) — 3 bits.
    bw.write_u32(2, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }

    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (false → AcplLite2 / two_channel_data path).
    bw.write_bit(false);

    // two_channel_data(): Table-181 matrix-input carriers (sSMP_A,
    // sSMP_B) — NOT the raw L/R preliminaries (round-279 carrier fix).
    write_two_channel_data(&mut bw, transform_length, max_sfb, &carrier_a, &carrier_b);

    // ASPX_ACPL_1 joint-MDCT residual layer — SAP-aware path with the
    // derived chparam pair (recomputes the same inverse internally for
    // the (s3, s4) residual tracks).
    write_acpl_1_residual_layer_sap(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_l,
        coeffs_r,
        coeffs_ls,
        coeffs_rs,
        Some(&chparam_pair),
    );

    // Cfg0 (coding_config == 0): mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // Data elements are present on every frame per Tables 25/33;
    // only the configs and the per-element 3-bit xover offset are
    // I-frame-gated (framed writers skip the xover when
    // b_iframe == 0; the decoder reuses the sticky value).
    {
        write_aspx_data_2ch_minimal_framed(&mut bw, aspx_cfg, b_iframe)
            .expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal_framed(&mut bw, aspx_cfg, b_iframe)
            .expect("encoder: aspx config invalid");
        let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
        let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
    }

    finish_substream_body(bw)
}

// ====================================================================
// Real per-band α extraction — ETSI TS 103 190-1 §5.7.7 (round 128)
// ====================================================================
//
// Estimates A-CPL alpha per `acpl_num_param_bands` parameter band from the
// MDCT-domain carrier vs. surround band-energy ratio, then writes the
// matching per-band F0 + DT codewords.
//
// Approach (β = 0 simplification — spec-defensible "level-only" coding):
//
// For pair-1 (D0, L-side ACplModule, `acpl_data_1ch_pair[0]`):
//   * `x0 = L_carrier`, `x1 = Ls_carrier` (PARTIAL mode only).
//   * Per §5.7.7.5 / Pseudocode 116 above `acpl_qmf_band`:
//
//         z0 = 0.5 · (x0·(1+α) + y·β)    → reconstructed L
//         z1 = 0.5 · (x0·(1-α) - y·β)    → reconstructed Ls (then ×√2 in
//                                          Pseudocode 117)
//
//   * With β = 0:
//
//         Ls_recon · √2 = 0.5 · L · (1 - α)
//         ⇒  α  =  1 − 2·√2 · ⟨Ls, L⟩ / ⟨L, L⟩
//
//   * The 〈·,·〉 inner product is computed per parameter band over the
//     MDCT bins that fall in that band (mapped via the
//     QMF-subband-frequency-aligned partition of the spectrum).
//
// For pair-2 (D1, R-side ACplModule, `acpl_data_1ch_pair[1]`): same shape
// with `x0 = R_carrier`, `x1 = Rs_carrier`.
//
// The recovered α is quantised by nearest-neighbour to the spec's
// `ALPHA_DQ_FINE` (Table 203) / `ALPHA_DQ_COARSE` (Table 205) tables and
// the matching α_q index (range `-N/2..=+N/2` with `N + 1 == table_len`)
// is written via the ACPL ALPHA F0 + DT codebooks.
//
// β stays at the zero-codebook index (current scaffold). The decoder
// recovers β = 0 ⇒ no decorrelator contribution and the ducker output is
// suppressed in that band. This preserves the level-only correctness
// established by α.
//
// Limitations (deferred to a future round):
//   * BETA, BETA3, GAMMA stay at zero-delta (the GAMMA / BETA3 paths only
//     fire in ASPX_ACPL_3 anyway).
//   * Only `AcplQuantMode::Fine` is exercised — the coarse table just
//     re-indexes the same algorithm so adding it is mechanical.
//   * Only DIFF_FREQ (DF) coding is used (no DIFF_TIME) — DT requires
//     carrying state across frames.
//   * Only the first parameter set is emitted (the framing emits
//     `num_param_sets = 1`).

use crate::acpl_synth::{ALPHA_DQ_COARSE, ALPHA_DQ_FINE};

/// Map an MDCT bin index `bin` (range `0..transform_length`) to the
/// matching A-CPL parameter band `pb` (range `0..num_param_bands`).
///
/// The MDCT lives at ~`fs/(2·transform_length)` bin width while the QMF
/// subbands the §5.7.7.2 Table 197 mapping is defined on live at
/// `fs/(2·64) = fs/128`. So the MDCT-to-QMF subband mapping is
/// `sb = bin · 64 / transform_length` (integer) — the §5.7.7.2 table is
/// then walked with `sb`.
///
/// Returns `pb` clamped to `num_param_bands - 1`.
fn mdct_bin_to_param_band(bin: u32, transform_length: u32, num_param_bands: u32) -> u32 {
    let sb = (bin * 64) / transform_length.max(1);
    let sb = sb.min(63);
    crate::acpl::sb_to_pb(sb, num_param_bands)
}

/// Compute per-parameter-band cross-energy ratios
/// `(num = Σ x_carrier · x_surround, den = Σ x_carrier²)` for one MDCT
/// frame across the configured A-CPL parameter band layout.
///
/// `coeffs_carrier` is the L (or R) MDCT spectrum; `coeffs_surround` is
/// the Ls (or Rs) MDCT spectrum.
///
/// Bands strictly below `start_pb` (the parameter-band index that the
/// PARTIAL-mode `acpl_qmf_band` resolves to via [`crate::acpl::sb_to_pb`])
/// are not estimated — the synth M/S split below `acpl_qmf_band` carries
/// those bins directly and α has no effect there.
///
/// Returned vectors are length `num_param_bands`; entries below
/// `start_pb` are `(0.0, 0.0)`.
fn compute_per_band_correlations(
    coeffs_carrier: &[f32],
    coeffs_surround: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_param_bands as usize;
    let mut num = vec![0.0f32; n];
    let mut den = vec![0.0f32; n];
    let len = coeffs_carrier.len().min(coeffs_surround.len());
    for bin in 0..len {
        let pb = mdct_bin_to_param_band(bin as u32, transform_length, num_param_bands) as usize;
        if (pb as u32) < start_pb {
            continue;
        }
        let xc = coeffs_carrier[bin];
        let xs = coeffs_surround[bin];
        num[pb] += xc * xs;
        den[pb] += xc * xc;
    }
    (num, den)
}

/// Compute the analytic per-band α value
/// `α = 1 − 2·√2 · ⟨carrier, surround⟩ / ⟨carrier, carrier⟩` and clamp it
/// to the spec dequantisation range.
///
/// Returns one α per parameter band; bands with `den[pb] == 0` or below
/// `start_pb` are returned as `0.0` (which quantises to the zero-codebook
/// alpha index, identical to the round-95 scaffold).
fn analytic_alpha_per_band(num: &[f32], den: &[f32], qm: crate::acpl::AcplQuantMode) -> Vec<f32> {
    let max_abs: f32 = match qm {
        crate::acpl::AcplQuantMode::Fine => 2.0, // ALPHA_DQ_FINE bounds: ±2.0
        crate::acpl::AcplQuantMode::Coarse => 2.0, // ALPHA_DQ_COARSE bounds: ±2.0
    };
    let sqrt2 = (2.0f32).sqrt();
    let mut out = Vec::with_capacity(num.len());
    for i in 0..num.len() {
        let d = den[i];
        if d <= 0.0 || !d.is_finite() {
            out.push(0.0);
            continue;
        }
        let ratio = num[i] / d;
        let mut a = 1.0 - 2.0 * sqrt2 * ratio;
        if !a.is_finite() {
            a = 0.0;
        }
        out.push(a.clamp(-max_abs, max_abs));
    }
    out
}

/// Quantise an analytic α to the spec's nearest `alpha_q` index in the
/// signed range `-N/2..=+N/2` (where `N + 1 == ALPHA_DQ_*.len()`), per the
/// dequantisation tables [`ALPHA_DQ_FINE`] (Table 203) /
/// [`ALPHA_DQ_COARSE`] (Table 205).
pub(crate) fn quantise_alpha(alpha: f32, qm: crate::acpl::AcplQuantMode) -> i32 {
    let (table, cb_off): (&[f32], i32) = match qm {
        crate::acpl::AcplQuantMode::Fine => (&ALPHA_DQ_FINE, 16),
        crate::acpl::AcplQuantMode::Coarse => (&ALPHA_DQ_COARSE, 8),
    };
    let mut best_lane = 0usize;
    let mut best_err = f32::INFINITY;
    for (lane, &v) in table.iter().enumerate() {
        let err = (v - alpha).abs();
        if err < best_err {
            best_err = err;
            best_lane = lane;
        }
    }
    (best_lane as i32) - cb_off
}

/// Emit a standalone `acpl_data_1ch()` element (§4.2.13.3 Table 61) with
/// real per-band α and β indices, returning the byte buffer. Intended
/// for round-trip validation against [`crate::acpl::parse_acpl_data_1ch`]
/// — the production encoder embeds this element inside the full substream
/// body via [`build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta`].
///
/// `alpha_q_per_band` / `beta_q_per_band` are indexed by parameter band;
/// entries below `start_band` are ignored (the element only codes bands
/// `start_band..num_bands`).
pub fn write_acpl_data_1ch_real_alpha_beta_bytes(
    num_bands: u32,
    start_band: u32,
    quant_mode: crate::acpl::AcplQuantMode,
    alpha_q_per_band: &[i32],
    beta_q_per_band: &[i32],
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    write_acpl_data_1ch_real_alpha_beta(
        &mut bw,
        num_bands,
        start_band,
        quant_mode,
        alpha_q_per_band,
        Some(beta_q_per_band),
    );
    bw.finish()
}

/// Public entry point that lets callers extract the per-parameter-band
/// β magnitudes the encoder would emit for a given (carrier, surround)
/// MDCT pair plus the already-quantised α values. Intended for tests
/// and validators that need to inspect the extractor's intermediate
/// state — the production encoder calls the internals directly through
/// [`build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta`].
///
/// Returns one β_q per parameter band; entries below `start_pb` or
/// where the carrier energy is zero are 0.
pub fn extract_beta_q_per_band(
    coeffs_carrier: &[f32],
    coeffs_surround: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    alpha_q: &[i32],
    qm: crate::acpl::AcplQuantMode,
) -> Vec<i32> {
    let (e_c, e_s) = compute_per_band_energies(
        coeffs_carrier,
        coeffs_surround,
        transform_length,
        num_param_bands,
        start_pb,
    );
    let alpha_dq: Vec<f32> = alpha_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(qm, q).0)
        .collect();
    let beta = analytic_beta_per_band(&e_c, &e_s, &alpha_dq, qm);
    beta.iter()
        .map(|&b| quantise_beta_magnitude(b, qm))
        .collect()
}

/// Extract the per-parameter-band `(alpha_q, beta_q)` rows for one
/// **immersive** A-CPL module (TS 103 190-2 §5.5.2 Table 27 driving
/// the part-1 §5.7.7.5 Pseudocode 115/116 pair synthesis with the √2
/// immersive output scale).
///
/// The decoder reconstructs the channel pair `(P, Q)` from the coded
/// mid carrier `x = (P+Q)/(2√2)` as
///
/// ```text
///   P̂ = √2·(x·(1+α) + w)      Q̂ = √2·(x·(1−α) − w)
/// ```
///
/// (`w` = β-scaled ducked decorrelator output with
/// `E[w²]·2 ≈ β²·E[mid²]`). With `mid = (P+Q)/2`, `side = (P−Q)/2`
/// the sum is reconstructed exactly and the optimal parameters per
/// band are
///
/// ```text
///   α* = Σ(mid·side) / Σ(mid²)
///   β² = max(0, Σ(side²)/Σ(mid²) − α_dq²)
/// ```
///
/// — α carries the correlated part of the inter-channel difference,
/// β fills the residual (decorrelated) energy. Both quantise through
/// the Tables 203-206 codebook lanes.
///
/// `mid_coeffs` / `side_coeffs` are the MDCT spectra of `(P+Q)/2`
/// and `(P−Q)/2`; bands below `start_pb` return 0 (the PARTIAL
/// config codes them as M/S residuals instead).
pub fn extract_ice_acpl_pair_alpha_beta_q(
    mid_coeffs: &[f32],
    side_coeffs: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    qm: crate::acpl::AcplQuantMode,
) -> (Vec<i32>, Vec<i32>) {
    let (num, den) = compute_per_band_correlations(
        mid_coeffs,
        side_coeffs,
        transform_length,
        num_param_bands,
        start_pb,
    );
    let (e_mid, e_side) = compute_per_band_energies(
        mid_coeffs,
        side_coeffs,
        transform_length,
        num_param_bands,
        start_pb,
    );
    let n = num_param_bands as usize;
    let mut alpha_q = Vec::with_capacity(n);
    let mut beta_q = Vec::with_capacity(n);
    #[allow(clippy::needless_range_loop)] // per-band lockstep over four arrays
    for pb in 0..n {
        let d = den.get(pb).copied().unwrap_or(0.0);
        if d <= 0.0 || !d.is_finite() {
            alpha_q.push(0);
            beta_q.push(0);
            continue;
        }
        let alpha = (num[pb] / d).clamp(-2.0, 2.0);
        let a_q = quantise_alpha(alpha, qm);
        let a_dq = crate::acpl_synth::dequantize_alpha_index(qm, a_q).0;
        let em = e_mid.get(pb).copied().unwrap_or(0.0);
        let es = e_side.get(pb).copied().unwrap_or(0.0);
        let beta_sq = if em > 0.0 { es / em - a_dq * a_dq } else { 0.0 };
        let beta = if beta_sq > 0.0 && beta_sq.is_finite() {
            beta_sq.sqrt().clamp(0.0, 4.0)
        } else {
            0.0
        };
        alpha_q.push(a_q);
        beta_q.push(quantise_beta_magnitude(beta, qm));
    }
    (alpha_q, beta_q)
}

/// Public entry point that returns the per-parameter-band α_q the
/// encoder would emit for a given (carrier, surround) MDCT pair.
pub fn extract_alpha_q_per_band(
    coeffs_carrier: &[f32],
    coeffs_surround: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    qm: crate::acpl::AcplQuantMode,
) -> Vec<i32> {
    let (num, den) = compute_per_band_correlations(
        coeffs_carrier,
        coeffs_surround,
        transform_length,
        num_param_bands,
        start_pb,
    );
    let alpha = analytic_alpha_per_band(&num, &den, qm);
    alpha.iter().map(|&a| quantise_alpha(a, qm)).collect()
}

/// Extract a per-parameter-band β_q sequence from a single carrier's
/// MDCT energy distribution, suitable for the ASPX_ACPL_3 path where
/// only the L / R / C carriers are available at encode time and no
/// surround reference exists for the analytic `β² = max(0, 2·E[Ls²]/
/// E[L²] − (1−α)²)` extractor above.
///
/// The β parameter in ACplModule2 (§5.7.7.6.2 Pseudocode 119) is the
/// gain applied to the decorrelator output `y`. The decoder writes
///
/// ```text
///   z0 = 0.5·(x0·g1 + x1·g2 + y·β)
///   z1 = 0.5·(x0·g1 + x1·g2 − y·β)
/// ```
///
/// with `E[y²] ≈ E[x0²]` after the upstream `Transform()` call. Setting
/// β proportional to the per-band carrier RMS keeps the surround
/// reconstruction's wet/dry balance bounded — a band carrying more
/// energy gets a proportionally louder decorrelator injection so the
/// surround channels remain perceptually consistent with the dry
/// front-channel mix.
///
/// The scale chosen here — `β = scale · √E[x0²]` clipped to the β
/// codebook's column-0 magnitude range — is an encoder choice (not a
/// spec mandate); the decoder reverses the BETA codebook lookup and
/// applies whatever magnitude was written. With `scale = 0.0` this
/// returns all-zero β_q (matching the round-95 zero-delta scaffold);
/// `scale = 1.0` saturates a band carrying unit-RMS energy to the
/// codebook's mid-range lane (~1.4 fine / 1.4 coarse).
///
/// `scale` should typically be small (≤ 0.5) to keep β within the
/// quantiser's perceptually-useful lower half. Returns one β_q per
/// parameter band; entries below `start_pb` are 0.
pub fn extract_beta_q_per_band_carrier_energy(
    coeffs_carrier: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    scale: f32,
    qm: crate::acpl::AcplQuantMode,
) -> Vec<i32> {
    let (e_c, _e_zero) = compute_per_band_energies(
        coeffs_carrier,
        coeffs_carrier,
        transform_length,
        num_param_bands,
        start_pb,
    );
    e_c.iter()
        .map(|&e| {
            if e <= 0.0 || !e.is_finite() {
                0
            } else {
                let rms = e.sqrt();
                let beta_mag = (scale * rms).max(0.0);
                quantise_beta_magnitude(beta_mag, qm)
            }
        })
        .collect()
}

/// Extract a per-parameter-band α_q sequence from the L↔R carrier
/// normalised cross-correlation, suitable for the ASPX_ACPL_3 encoder
/// path where the two ACplModule2 instances share the (L, R) carrier
/// pair as their (x0, x1) input and no per-side surround reference
/// exists at encode time.
///
/// The α parameter in ACplModule2 (§5.7.7.6.2 Pseudocode 119) modulates
/// the front/back balance of the dry mix:
///
/// ```text
///   z0 = 0.5·(x0·(g1+g1·α) + x1·(g2+g2·α) + y·β)
///      = 0.5·(1+α)·(g1·x0 + g2·x1) + 0.5·y·β
///   z1 = 0.5·(1−α)·(g1·x0 + g2·x1) − 0.5·y·β
/// ```
///
/// so α = +1 collapses the surround output to pure decorrelator (front-
/// only), α = −1 collapses the front output to pure decorrelator (back-
/// only), and α = 0 splits the dry mix evenly between front and back.
/// The encoder's choice of α is therefore a policy: how much of the dry
/// (γ·x0 + γ·x1) energy should land in the surround pair?
///
/// This extractor's policy: drive α from the per-band normalised L↔R
/// cross-correlation `ρ(L, R) = E[L·R] / √(E[L²]·E[R²])`:
///
/// ```text
///   α[pb] = α_scale · ρ(L, R)[pb]
/// ```
///
/// clamped to the ALPHA_DQ table magnitude bound (±2.0 Fine / ±2.0
/// Coarse). With `α_scale = 0.0` this returns all-zero α_q (matching
/// the round-95 zero-delta scaffold byte-for-byte). With
/// `α_scale = 1.0` highly-correlated (mono-like) bands push α toward
/// +1.0 — biasing more dry energy toward the front pair; decorrelated
/// bands stay near α = 0 — splitting the dry mix evenly; rare
/// anti-correlated bands push α toward −1.0 — biasing the dry mix
/// toward the back pair.
///
/// Returns one α_q per parameter band; entries below `start_pb` or
/// where either L or R carrier per-band energy is zero are 0.
pub fn extract_alpha_q_per_band_carrier_correlation(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    alpha_scale: f32,
    qm: crate::acpl::AcplQuantMode,
) -> Vec<i32> {
    let n = num_param_bands as usize;
    let mut e_lr = vec![0.0f32; n];
    let mut e_ll = vec![0.0f32; n];
    let mut e_rr = vec![0.0f32; n];
    let len = coeffs_l.len().min(coeffs_r.len());
    for bin in 0..len {
        let pb = mdct_bin_to_param_band(bin as u32, transform_length, num_param_bands) as usize;
        if (pb as u32) < start_pb {
            continue;
        }
        let xl = coeffs_l[bin];
        let xr = coeffs_r[bin];
        e_lr[pb] += xl * xr;
        e_ll[pb] += xl * xl;
        e_rr[pb] += xr * xr;
    }
    let max_abs: f32 = match qm {
        crate::acpl::AcplQuantMode::Fine => 2.0,
        crate::acpl::AcplQuantMode::Coarse => 2.0,
    };
    (0..n)
        .map(|pb| {
            let denom_sq = e_ll[pb] * e_rr[pb];
            if denom_sq <= 0.0 || !denom_sq.is_finite() {
                return 0;
            }
            let denom = denom_sq.sqrt();
            let rho = e_lr[pb] / denom;
            let mut a = alpha_scale * rho;
            if !a.is_finite() {
                a = 0.0;
            }
            quantise_alpha(a.clamp(-max_abs, max_abs), qm)
        })
        .collect()
}

/// Quantise an analytic γ magnitude (signed) to the spec's nearest
/// `gamma_q` index in the signed range `-cb_off..=+cb_off` (where
/// `cb_off = 10` Coarse / `20` Fine), per §5.7.7.7 Table 208. The
/// γ dequantisation is the simple linear map
/// `gamma_dq = gamma_q · gamma_delta(qm)` so we recover `gamma_q` as the
/// nearest-integer multiple of `1 / gamma_delta`.
///
/// Fine:   `gamma_delta = 1638 / 16384 ≈ 0.0999756`,
///         table magnitude bound `≈ 20 · 0.1 = 2.0`.
/// Coarse: `gamma_delta = 3276 / 16384 ≈ 0.1999512`,
///         table magnitude bound `≈ 10 · 0.2 = 2.0`.
fn quantise_gamma(gamma: f32, qm: crate::acpl::AcplQuantMode) -> i32 {
    let delta = crate::acpl_synth::gamma_delta(qm);
    let cb_off: i32 = match qm {
        crate::acpl::AcplQuantMode::Fine => 20,
        crate::acpl::AcplQuantMode::Coarse => 10,
    };
    // gamma magnitude bound: cb_off · delta (= 2.0 for both modes).
    let max_abs = (cb_off as f32) * delta;
    let g = gamma.clamp(-max_abs, max_abs);
    let raw = (g / delta).round() as i32;
    raw.clamp(-cb_off, cb_off)
}

/// Quantise an analytic β₃ magnitude (signed) to the spec's nearest
/// `beta3_q` index in the signed range `-cb_off..=+cb_off` (where
/// `cb_off = 4` Coarse / `8` Fine — half the BETA3 F0 codebook length
/// per the staged ETSI table file §A.3 Tables A.46 / A.47), per
/// §5.7.7.7 Table 207. The β₃ dequantisation is the simple linear map
/// `acpl_beta_3_dq = beta3_q · beta3_delta(qm)` so we recover
/// `beta3_q` as the nearest-integer multiple of `1 / beta3_delta`.
///
/// Fine:   `beta3_delta = 0.125`, table magnitude bound `8 · 0.125 = 1.0`.
/// Coarse: `beta3_delta = 0.25`,  table magnitude bound `4 · 0.25 = 1.0`.
fn quantise_beta3(beta3: f32, qm: crate::acpl::AcplQuantMode) -> i32 {
    let delta = crate::acpl_synth::beta3_delta(qm);
    let cb_off: i32 = match qm {
        crate::acpl::AcplQuantMode::Fine => 8,
        crate::acpl::AcplQuantMode::Coarse => 4,
    };
    // β₃ magnitude bound: cb_off · delta (= 1.0 for both modes).
    let max_abs = (cb_off as f32) * delta;
    let b = beta3.clamp(-max_abs, max_abs);
    let raw = (b / delta).round() as i32;
    raw.clamp(-cb_off, cb_off)
}

/// Compute the per-parameter-band gamma pair `(γ5, γ6)` that minimises
/// the centre-channel reconstruction error for ASPX_ACPL_3 step 7 of
/// §5.7.7.6.2 Pseudocode 118:
///
/// ```text
///   z4 = 0.5 · (g5 · x0in + g6 · x1in)
///   C  = √2 · z4         (Pseudocode 118 step 11 scales z4 by √2)
///   x0in = (1 + √2) · L  (Pseudocode 118 step 1 input scaling)
///   x1in = (1 + √2) · R
/// ```
///
/// Combining, the centre-channel reconstruction (β3 = 0, no ACplModule3
/// correction, ducker = 1) is:
///
/// ```text
///   C ≈ K · (γ5 · L + γ6 · R)        K = √2 · (1 + √2) / 2 = 1 + √(1/2)
/// ```
///
/// — i.e. `C / K ≈ γ5 · L + γ6 · R`. The per-band least-squares fit
/// solves
///
/// ```text
///   min_{γ5, γ6}  Σ_bin ( C[bin] / K - γ5 · L[bin] - γ6 · R[bin] )²
/// ```
///
/// via the 2×2 normal equations
///
/// ```text
///   [ <L,L>  <L,R> ] [γ5]   [ <L,C/K> ]
///   [ <L,R>  <R,R> ] [γ6] = [ <R,C/K> ]
/// ```
///
/// where the inner products are summed over the MDCT bins that
/// [`mdct_bin_to_param_band`] maps to parameter band `pb`. Bands where
/// the 2×2 system is singular (zero L or R energy, or perfectly
/// collinear L = ±R within numerical tolerance) return `(0, 0)`.
///
/// Returns one `(γ5_q, γ6_q)` pair per parameter band; entries below
/// `start_pb` are `(0, 0)`. The non-empty bands are scaled by
/// `gamma_scale` (typically `1.0` for full strength; `0.0` reproduces
/// the round-95 zero-delta scaffold byte-for-byte). The quantiser uses
/// the Table-208 linear `gamma_q = round(γ / gamma_delta)` mapping with
/// the symmetric `±cb_off` clamp.
#[allow(clippy::too_many_arguments)]
pub fn extract_gamma_5_6_q_per_band_centre_least_squares(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    gamma_scale: f32,
    qm: crate::acpl::AcplQuantMode,
) -> (Vec<i32>, Vec<i32>) {
    let n = num_param_bands as usize;
    let mut e_ll = vec![0.0f32; n];
    let mut e_rr = vec![0.0f32; n];
    let mut e_lr = vec![0.0f32; n];
    let mut e_lc = vec![0.0f32; n];
    let mut e_rc = vec![0.0f32; n];
    let len = coeffs_l.len().min(coeffs_r.len()).min(coeffs_c.len());
    for bin in 0..len {
        let pb = mdct_bin_to_param_band(bin as u32, transform_length, num_param_bands) as usize;
        if (pb as u32) < start_pb {
            continue;
        }
        let xl = coeffs_l[bin];
        let xr = coeffs_r[bin];
        let xc = coeffs_c[bin];
        e_ll[pb] += xl * xl;
        e_rr[pb] += xr * xr;
        e_lr[pb] += xl * xr;
        e_lc[pb] += xl * xc;
        e_rc[pb] += xr * xc;
    }
    // K = √2 · (1 + √2) / 2 = 1 + √(1/2).
    let k = 1.0 + (0.5f32).sqrt();
    let inv_k = 1.0 / k;
    let mut g5_q = vec![0i32; n];
    let mut g6_q = vec![0i32; n];
    for pb in 0..n {
        if (pb as u32) < start_pb {
            continue;
        }
        let a = e_ll[pb];
        let b = e_lr[pb];
        let c = e_rr[pb];
        let det = a * c - b * b;
        if !det.is_finite() || det.abs() <= f32::EPSILON * (a.abs() + c.abs() + 1.0) {
            continue;
        }
        // Right-hand side is <L, C/K> = <L, C> / K and similarly for R.
        let rhs0 = e_lc[pb] * inv_k;
        let rhs1 = e_rc[pb] * inv_k;
        // Inverse of [[a, b], [b, c]] is (1/det) · [[c, -b], [-b, a]].
        let g5_raw = (c * rhs0 - b * rhs1) / det;
        let g6_raw = (-b * rhs0 + a * rhs1) / det;
        let g5 = gamma_scale * g5_raw;
        let g6 = gamma_scale * g6_raw;
        if !g5.is_finite() || !g6.is_finite() {
            continue;
        }
        g5_q[pb] = quantise_gamma(g5, qm);
        g6_q[pb] = quantise_gamma(g6, qm);
    }
    (g5_q, g6_q)
}

/// Shared per-band 2×2 least-squares fit solver for the γ pair that
/// reproduces an output channel pair `(out_front, out_back)` from the
/// (L, R) carrier pair through Pseudocode 118 step 5 / step 6 / step 11.
///
/// Given the Pseudocode 119 module-2 outputs at step 5 / 6 with
/// `(a = α, b = β, y)`:
///
/// ```text
///   z0 = 0.5·(1+α)·(g·x0in + g'·x1in) + 0.5·y·β
///   z1 = 0.5·(1−α)·(g·x0in + g'·x1in) − 0.5·y·β
/// ```
///
/// step 11 scales `z1 *= √2` before QMF synthesis. Forming the sum
/// `(front + back/√2)` cancels the decorrelator term `y·β` entirely
/// (the `+0.5·y·β` and `−0.5·y·β` contributions add to 0):
///
/// ```text
///   front + back/√2 = (g·x0in + g'·x1in)
/// ```
///
/// and step 1 expands `x0in / x1in = (1 + √2) · L / R`, giving
///
/// ```text
///   front + back/√2 = (1 + √2) · (g · L + g' · R)
/// ```
///
/// — i.e. `(front + back/√2) / (1 + √2) ≈ g · L + g' · R`, independent
/// of α and β. The per-band least-squares fit solves
///
/// ```text
///   min_{g, g'}  Σ_bin ( T[bin] − g · L[bin] − g' · R[bin] )²
/// ```
///
/// with `T = (front + back/√2) / (1 + √2)`, via the 2×2 normal equations
///
/// ```text
///   [ <L,L>  <L,R> ] [g ]   [ <L,T> ]
///   [ <L,R>  <R,R> ] [g'] = [ <R,T> ]
/// ```
///
/// where the inner products are summed over the MDCT bins that
/// [`mdct_bin_to_param_band`] maps to parameter band `pb`. Bands where
/// the 2×2 system is singular (zero L or R energy, or perfectly
/// collinear L = ±R within numerical tolerance) return `(0, 0)`.
///
/// Returns one `(g_q, g'_q)` pair per parameter band. Entries below
/// `start_pb` are `(0, 0)`. The non-empty bands are scaled by
/// `gamma_scale` and quantised through [`quantise_gamma`] (Table-208
/// linear, symmetric `±cb_off` clamp).
#[allow(clippy::too_many_arguments)]
fn extract_gamma_pair_q_per_band_surround_least_squares(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_front: &[f32],
    coeffs_back: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    gamma_scale: f32,
    qm: crate::acpl::AcplQuantMode,
) -> (Vec<i32>, Vec<i32>) {
    let n = num_param_bands as usize;
    let mut e_ll = vec![0.0f32; n];
    let mut e_rr = vec![0.0f32; n];
    let mut e_lr = vec![0.0f32; n];
    let mut e_lt = vec![0.0f32; n];
    let mut e_rt = vec![0.0f32; n];
    // K = 1 + √2 from Pseudocode 118 step 1 carrier rescaling.
    let k = 1.0 + (2.0f32).sqrt();
    let inv_k = 1.0 / k;
    // T = (front + back / √2) · inv_k.
    let inv_sqrt2 = 1.0 / (2.0f32).sqrt();
    let len = coeffs_l
        .len()
        .min(coeffs_r.len())
        .min(coeffs_front.len())
        .min(coeffs_back.len());
    for bin in 0..len {
        let pb = mdct_bin_to_param_band(bin as u32, transform_length, num_param_bands) as usize;
        if (pb as u32) < start_pb {
            continue;
        }
        let xl = coeffs_l[bin];
        let xr = coeffs_r[bin];
        let t = (coeffs_front[bin] + coeffs_back[bin] * inv_sqrt2) * inv_k;
        e_ll[pb] += xl * xl;
        e_rr[pb] += xr * xr;
        e_lr[pb] += xl * xr;
        e_lt[pb] += xl * t;
        e_rt[pb] += xr * t;
    }
    let mut g_q = vec![0i32; n];
    let mut gp_q = vec![0i32; n];
    for pb in 0..n {
        if (pb as u32) < start_pb {
            continue;
        }
        let a = e_ll[pb];
        let b = e_lr[pb];
        let c = e_rr[pb];
        let det = a * c - b * b;
        if !det.is_finite() || det.abs() <= f32::EPSILON * (a.abs() + c.abs() + 1.0) {
            continue;
        }
        // Inverse of [[a, b], [b, c]] is (1/det) · [[c, -b], [-b, a]].
        let g_raw = (c * e_lt[pb] - b * e_rt[pb]) / det;
        let gp_raw = (-b * e_lt[pb] + a * e_rt[pb]) / det;
        let g = gamma_scale * g_raw;
        let gp = gamma_scale * gp_raw;
        if !g.is_finite() || !gp.is_finite() {
            continue;
        }
        g_q[pb] = quantise_gamma(g, qm);
        gp_q[pb] = quantise_gamma(gp, qm);
    }
    (g_q, gp_q)
}

/// Compute the per-parameter-band gamma pair `(γ1, γ2)` that minimises
/// the (L, Ls) output-pair reconstruction error for ASPX_ACPL_3 step 5
/// of §5.7.7.6.2 Pseudocode 118 + Pseudocode 119:
///
/// ```text
///   z0 = 0.5·(1+α₁)·(γ₁·x0in + γ₂·x1in) + 0.5·y₀·β₁         → L
///   z1 = 0.5·(1−α₁)·(γ₁·x0in + γ₂·x1in) − 0.5·y₀·β₁
///   Ls = √2 · z1                                              (step 11)
///   x0in = (1 + √2)·L_orig, x1in = (1 + √2)·R_orig            (step 1)
/// ```
///
/// Forming `(L + Ls/√2)` cancels the decorrelator-driven `y₀·β₁`
/// contributions exactly, leaving
///
/// ```text
///   L + Ls/√2 = (γ₁·x0in + γ₂·x1in) = (1 + √2) · (γ₁·L_orig + γ₂·R_orig)
/// ```
///
/// independent of α₁ and β₁. The per-band least-squares fit returns
/// the `(γ₁, γ₂)` pair that minimises the MDCT-bin-wise residual.
///
/// `coeffs_ls` is the surround-left MDCT spectrum; `coeffs_l` / `coeffs_r`
/// are the carrier MDCT spectra (the L / R inputs the encoder is also
/// emitting on the `two_channel_data()` body). Entries below `start_pb`
/// are `(0, 0)`. Bands with a degenerate Gram matrix (no L or R energy,
/// or perfectly collinear L = ±R within numerical tolerance) return
/// `(0, 0)`.
///
/// `gamma_scale = 1.0` reproduces the analytic least-squares solution
/// (clamped to the Table-208 ±2.0 magnitude bound); `gamma_scale = 0.0`
/// returns all-zero `(γ₁_q, γ₂_q)`.
#[allow(clippy::too_many_arguments)]
pub fn extract_gamma_1_2_q_per_band_surround_least_squares(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    gamma_scale: f32,
    qm: crate::acpl::AcplQuantMode,
) -> (Vec<i32>, Vec<i32>) {
    extract_gamma_pair_q_per_band_surround_least_squares(
        coeffs_l,
        coeffs_r,
        coeffs_l,
        coeffs_ls,
        transform_length,
        num_param_bands,
        start_pb,
        gamma_scale,
        qm,
    )
}

/// Compute the per-parameter-band gamma pair `(γ3, γ4)` that minimises
/// the (R, Rs) output-pair reconstruction error for ASPX_ACPL_3 step 6
/// of §5.7.7.6.2 Pseudocode 118 + Pseudocode 119. By symmetry with the
/// γ1 / γ2 derivation above (substituting Ls → Rs and z0 → z2 / z1 → z3
/// per Pseudocode 118 step 6):
///
/// ```text
///   R + Rs/√2 = (γ₃·x0in + γ₄·x1in) = (1 + √2) · (γ₃·L_orig + γ₄·R_orig)
/// ```
///
/// independent of α₂ and β₂.
///
/// `coeffs_rs` is the surround-right MDCT spectrum; `coeffs_l` /
/// `coeffs_r` are the carrier MDCT spectra. Returns one `(γ₃_q, γ₄_q)`
/// pair per parameter band with the same fallback / scaling /
/// quantisation behaviour as
/// [`extract_gamma_1_2_q_per_band_surround_least_squares`].
#[allow(clippy::too_many_arguments)]
pub fn extract_gamma_3_4_q_per_band_surround_least_squares(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_rs: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    gamma_scale: f32,
    qm: crate::acpl::AcplQuantMode,
) -> (Vec<i32>, Vec<i32>) {
    extract_gamma_pair_q_per_band_surround_least_squares(
        coeffs_l,
        coeffs_r,
        coeffs_r,
        coeffs_rs,
        transform_length,
        num_param_bands,
        start_pb,
        gamma_scale,
        qm,
    )
}

/// Compute the per-parameter-band β₃ index that energy-matches the
/// centre-channel reconstruction residual left over after the
/// round-208 γ₅ / γ₆ dry-mix fit, for ASPX_ACPL_3 steps 7 / 10 / 11 of
/// §5.7.7.6.2 Pseudocode 118.
///
/// β₃ is the gain on the third decorrelator output `y₂` (Pseudocode
/// 118 steps 8–10, `ACplModule3`). The decoder's centre channel is
///
/// ```text
///   z4  = 0.5 · (γ₅·x0in + γ₆·x1in)                    (step 7, dry)
///   z4 += 0.25 · y₂ · (−β₃ − β₃·1) = −0.5 · β₃ · y₂    (step 10, wet)
///   C   = √2 · z4                                       (step 11)
/// ```
///
/// so the wet (decorrelated) centre contribution carries energy
/// `(√2 · 0.5 · β₃)² · E[y₂²] = 0.5 · β₃² · E[y₂²]` per band. The
/// decorrelator input is `v₃ = (γ₁+γ₃+γ₅)·x0in + (γ₂+γ₄+γ₆)·x1in`
/// (Pseudocode 118 step 2, third `Transform()` call) and the
/// decorrelator + ducker chain is energy-preserving in steady state,
/// so the encoder estimates `E[y₂²] ≈ E[v₃²]` from the carrier
/// spectra and the already-quantised γ indices:
///
/// ```text
///   E[v₃²] = (1+√2)² · (G₁²·<L,L> + 2·G₁·G₂·<L,R> + G₂²·<R,R>)
///   G₁ = (γ₁+γ₃+γ₅)_dq,  G₂ = (γ₂+γ₄+γ₆)_dq
/// ```
///
/// The dry-fit residual the wet path must cover is the per-band
/// least-squares remainder of the round-208 centre fit `C ≈ K·(γ₅·L +
/// γ₆·R)` with `K = 1 + √(1/2)` (using the *quantised* γ₅ / γ₆ the
/// decoder will actually apply):
///
/// ```text
///   E_res = <C,C> − 2K·(γ₅·<L,C> + γ₆·<R,C>)
///         + K²·(γ₅²·<L,L> + 2·γ₅·γ₆·<L,R> + γ₆²·<R,R>)
/// ```
///
/// Energy matching `0.5 · β₃² · E[v₃²] = E_res` gives the encoder
/// decision `β₃ = √(2 · E_res / E[v₃²])` — a non-negative magnitude
/// (decorrelated noise carries no usable sign), scaled by
/// `beta3_scale` and quantised through [`quantise_beta3`] (Table-207
/// linear, symmetric `±cb_off` clamp). Bands with no decorrelator
/// drive (`E[v₃²] ≈ 0`, e.g. all-zero γ) or no residual return 0.
///
/// `gamma*_q` are the per-band quantised γ indices the encoder is
/// emitting (dequantised internally per Table 208 with `qm_gamma` —
/// the `acpl_config_2ch` `quant_mode_1`); the returned `beta3_q` is
/// quantised per Table 207 with `qm_beta3` (`quant_mode_0`). Entries
/// below `start_pb` are 0. `beta3_scale = 0.0` returns all-zero
/// `beta3_q` (matching the round-95 zero-delta scaffold).
#[allow(clippy::too_many_arguments)]
pub fn extract_beta3_q_per_band_centre_residual(
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    gamma1_q: &[i32],
    gamma2_q: &[i32],
    gamma3_q: &[i32],
    gamma4_q: &[i32],
    gamma5_q: &[i32],
    gamma6_q: &[i32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
    beta3_scale: f32,
    qm_gamma: crate::acpl::AcplQuantMode,
    qm_beta3: crate::acpl::AcplQuantMode,
) -> Vec<i32> {
    let n = num_param_bands as usize;
    let mut e_ll = vec![0.0f32; n];
    let mut e_rr = vec![0.0f32; n];
    let mut e_lr = vec![0.0f32; n];
    let mut e_lc = vec![0.0f32; n];
    let mut e_rc = vec![0.0f32; n];
    let mut e_cc = vec![0.0f32; n];
    let len = coeffs_l.len().min(coeffs_r.len()).min(coeffs_c.len());
    for bin in 0..len {
        let pb = mdct_bin_to_param_band(bin as u32, transform_length, num_param_bands) as usize;
        if (pb as u32) < start_pb {
            continue;
        }
        let xl = coeffs_l[bin];
        let xr = coeffs_r[bin];
        let xc = coeffs_c[bin];
        e_ll[pb] += xl * xl;
        e_rr[pb] += xr * xr;
        e_lr[pb] += xl * xr;
        e_lc[pb] += xl * xc;
        e_rc[pb] += xr * xc;
        e_cc[pb] += xc * xc;
    }
    let gd = crate::acpl_synth::gamma_delta(qm_gamma);
    // K = √2 · (1 + √2) / 2 = 1 + √(1/2) — step-1 carrier rescale folded
    // with the step-7 0.5 and the step-11 √2 (same constant as the
    // round-208 γ₅ / γ₆ centre fit).
    let k = 1.0 + (0.5f32).sqrt();
    // (1 + √2)² — step-1 carrier rescale entering the Transform() input.
    let s2 = {
        let s = 1.0 + (2.0f32).sqrt();
        s * s
    };
    let mut beta3_q = vec![0i32; n];
    for pb in 0..n {
        if (pb as u32) < start_pb {
            continue;
        }
        let g_at = |g: &[i32]| g.get(pb).copied().unwrap_or(0) as f32 * gd;
        let g1 = g_at(gamma1_q);
        let g2 = g_at(gamma2_q);
        let g3 = g_at(gamma3_q);
        let g4 = g_at(gamma4_q);
        let g5 = g_at(gamma5_q);
        let g6 = g_at(gamma6_q);
        // Dry-fit residual energy (floored at 0 — the quantised γ pair
        // can over/undershoot the analytic optimum slightly and float
        // rounding may drive the closed form fractionally negative).
        let e_res = (e_cc[pb] - 2.0 * k * (g5 * e_lc[pb] + g6 * e_rc[pb])
            + k * k * (g5 * g5 * e_ll[pb] + 2.0 * g5 * g6 * e_lr[pb] + g6 * g6 * e_rr[pb]))
            .max(0.0);
        // Third-decorrelator drive energy E[v₃²].
        let big_g1 = g1 + g3 + g5;
        let big_g2 = g2 + g4 + g6;
        let e_v3 = s2
            * (big_g1 * big_g1 * e_ll[pb]
                + 2.0 * big_g1 * big_g2 * e_lr[pb]
                + big_g2 * big_g2 * e_rr[pb]);
        if !e_res.is_finite() || !e_v3.is_finite() || e_res <= 0.0 || e_v3 <= f32::EPSILON {
            continue;
        }
        let b3 = beta3_scale * (2.0 * e_res / e_v3).sqrt();
        if !b3.is_finite() {
            continue;
        }
        beta3_q[pb] = quantise_beta3(b3, qm_beta3);
    }
    beta3_q
}

/// Compute per-parameter-band carrier and surround energies
/// `(E_c = Σ x_carrier², E_s = Σ x_surround²)` across one MDCT frame.
///
/// Used by the β extractor to estimate the energy of the decorrelated
/// residual that `acpl_beta_dq` must reconstruct. Entries below
/// `start_pb` are returned as `(0.0, 0.0)`.
fn compute_per_band_energies(
    coeffs_carrier: &[f32],
    coeffs_surround: &[f32],
    transform_length: u32,
    num_param_bands: u32,
    start_pb: u32,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_param_bands as usize;
    let mut e_c = vec![0.0f32; n];
    let mut e_s = vec![0.0f32; n];
    let len = coeffs_carrier.len().min(coeffs_surround.len());
    for bin in 0..len {
        let pb = mdct_bin_to_param_band(bin as u32, transform_length, num_param_bands) as usize;
        if (pb as u32) < start_pb {
            continue;
        }
        let xc = coeffs_carrier[bin];
        let xs = coeffs_surround[bin];
        e_c[pb] += xc * xc;
        e_s[pb] += xs * xs;
    }
    (e_c, e_s)
}

/// Compute the analytic per-band β magnitude that closes the
/// energy-balance for the Pseudocode 116 surround reconstruction.
///
/// Given `x0 = L` (carrier) and `y` = decorrelated(L) (energy-preserving
/// and ⊥ x0 in the long-term sense), the Pseudocode 116 surround output is
///
/// ```text
///   z1 = 0.5 · (x0·(1 − α) − y·β)
/// ```
///
/// Per Pseudocode 117 the Ls reconstruction at the decoder is
/// `Ls_recon = √2 · z1`. Squaring and taking expectations under
/// `E[x0 · y] = 0` and `E[y²] ≈ E[x0²]`:
///
/// ```text
///   E[Ls²] = 0.25 · 2 · ( E[x0²] · (1-α)² + E[y²] · β² )
///          = 0.5 · E[x0²] · ( (1-α)² + β² )
/// ```
///
/// Solving for β² and clamping to `[0, BETA_DQ_MAX²]`:
///
/// ```text
///   β² = max(0, 2·E[Ls²]/E[x0²] − (1 − α)²)
/// ```
///
/// Returns the **non-negative** β per band (entries below `start_pb`
/// or with zero carrier energy → 0.0). The encoder writes the F0
/// magnitude into the ACPL BETA F0 codebook (unsigned `cb_off = 0`)
/// and chains sign-less DF deltas thereafter — the round-132 entry
/// point therefore produces β_q ∈ {0..=max_q} for every band, leaving
/// the decoder's `dequantize_beta_index` magnitude-only mapping
/// (§5.7.7.7 Table 204 / 206) producing the matching positive β.
fn analytic_beta_per_band(
    energy_c: &[f32],
    energy_s: &[f32],
    alpha_dq: &[f32],
    qm: crate::acpl::AcplQuantMode,
) -> Vec<f32> {
    // β table magnitude bounds: column-0 (ibeta = 0) goes up to row-N.
    // Fine: 4.0 (row 8). Coarse: 4.0 (row 4). Per Table 204 / 206.
    let max_abs: f32 = match qm {
        crate::acpl::AcplQuantMode::Fine => 4.0,
        crate::acpl::AcplQuantMode::Coarse => 4.0,
    };
    let n = energy_c.len().min(energy_s.len()).min(alpha_dq.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ec = energy_c[i];
        let es = energy_s[i];
        if ec <= 0.0 || !ec.is_finite() {
            out.push(0.0);
            continue;
        }
        let one_minus_a = 1.0 - alpha_dq[i];
        let beta_sq = 2.0 * es / ec - one_minus_a * one_minus_a;
        let beta = if beta_sq <= 0.0 || !beta_sq.is_finite() {
            0.0
        } else {
            beta_sq.sqrt()
        };
        out.push(beta.clamp(0.0, max_abs));
    }
    out
}

/// Quantise a non-negative analytic β magnitude to the spec's nearest
/// `beta_q` index (`0..=max_q`). The β dequantisation table is
/// 2-dimensional (`[beta_q][ibeta]` per Table 204 / 206) — for the
/// round-132 F0-only encoder we pick `ibeta = 0` (the column matched
/// by the `IBETA_*` row at `alpha_q = 0`, i.e. column 0 carries the
/// maximum magnitudes 0.0 / 0.2375 / 0.55 / 0.9375 / 1.4 / 1.9375 /
/// 2.55 / 3.2375 / 4.0 fine, or 0.0 / 0.55 / 1.4 / 2.55 / 4.0 coarse).
/// This keeps the F0-only path's quantisation deterministic and matches
/// the decoder's column-0 lookup when no α delta-table mismatch occurs.
pub(crate) fn quantise_beta_magnitude(beta_mag: f32, qm: crate::acpl::AcplQuantMode) -> i32 {
    let table: &[f32] = match qm {
        crate::acpl::AcplQuantMode::Fine => {
            &[0.0, 0.2375, 0.55, 0.9375, 1.4, 1.9375, 2.55, 3.2375, 4.0]
        }
        crate::acpl::AcplQuantMode::Coarse => &[0.0, 0.55, 1.4, 2.55, 4.0],
    };
    let mut best_lane = 0usize;
    let mut best_err = f32::INFINITY;
    for (lane, &v) in table.iter().enumerate() {
        let err = (v - beta_mag).abs();
        if err < best_err {
            best_err = err;
            best_lane = lane;
        }
    }
    best_lane as i32
}

/// Write the ACPL BETA F0 codeword for a recovered non-negative
/// `beta_q` index per §A.3 Table A.41 (Fine) / Table A.40 (Coarse).
/// The F0 codebook is addressed by `symbol_index = beta_q + cb_off`
/// with `cb_off = 0` (per [`acpl_hcb_arrays`]) so the wire index is
/// directly `beta_q`.
fn write_acpl_beta_f0_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Beta,
        qm,
        crate::acpl::AcplHcbType::F0,
    );
    let idx = (beta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL BETA DF codeword for a recovered band-to-band delta
/// `delta_q = beta_q[pb] - beta_q[pb-1]`. Per Table A.41 / A.40 the DF
/// codebook is addressed by `symbol_index = delta_q + cb_off`.
fn write_acpl_beta_df_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, delta_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Beta,
        qm,
        crate::acpl::AcplHcbType::Df,
    );
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL ALPHA F0 codeword for a recovered `alpha_q` index per
/// §A.3 Table A.35 (Fine) / Table A.34 (Coarse). The Huffman table is
/// addressed by `symbol_index = alpha_q + cb_off` (cb_off = 8 Coarse /
/// 16 Fine for the ALPHA F0 codebooks per [`acpl_hcb_arrays`] — the
/// codebooks are symmetric around the centre index so `alpha_q`
/// carries its sign).
fn write_acpl_alpha_f0_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, alpha_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Alpha,
        qm,
        crate::acpl::AcplHcbType::F0,
    );
    let idx = (alpha_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL ALPHA DF codeword for a recovered band-to-band delta
/// `delta_q = alpha_q[pb] - alpha_q[pb-1]`. Per Table A.35 / A.34 the DF
/// codebook is addressed by `symbol_index = delta_q + cb_off`.
fn write_acpl_alpha_df_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, delta_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Alpha,
        qm,
        crate::acpl::AcplHcbType::Df,
    );
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL GAMMA F0 codeword for a signed `gamma_q` index per
/// §A.3 Tables A.52 (Fine) / A.53 (Coarse). The Huffman table is
/// addressed by `symbol_index = gamma_q + cb_off` with `cb_off = 10`
/// Coarse / `20` Fine — the codebook is symmetric around the centre so
/// `gamma_q` carries its sign.
fn write_acpl_gamma_f0_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, gamma_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Gamma,
        qm,
        crate::acpl::AcplHcbType::F0,
    );
    let idx = (gamma_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL GAMMA DF codeword for a band-to-band delta
/// `delta_q = gamma_q[pb] - gamma_q[pb-1]`. Per Tables A.54 / A.55 the
/// DF codebook is addressed by `symbol_index = delta_q + cb_off` with
/// `cb_off = 20` Coarse / `40` Fine.
fn write_acpl_gamma_df_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, delta_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Gamma,
        qm,
        crate::acpl::AcplHcbType::Df,
    );
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL BETA3 F0 codeword for a signed `beta3_q` index per
/// the staged ETSI table file §A.3 Tables A.46 (Coarse) / A.47 (Fine).
/// The Huffman table is addressed by `symbol_index = beta3_q + cb_off`
/// with `cb_off = 4` Coarse / `8` Fine — the codebook is symmetric
/// around the centre so `beta3_q` carries its sign.
fn write_acpl_beta3_f0_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, beta3_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Beta3,
        qm,
        crate::acpl::AcplHcbType::F0,
    );
    let idx = (beta3_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Write the ACPL BETA3 DF codeword for a band-to-band delta
/// `delta_q = beta3_q[pb] - beta3_q[pb-1]`. Per the staged ETSI table
/// file §A.3 Tables A.48 / A.49 the DF codebook is addressed by
/// `symbol_index = delta_q + cb_off` with `cb_off = 8` Coarse / `16`
/// Fine.
fn write_acpl_beta3_df_value(bw: &mut BitWriter, qm: crate::acpl::AcplQuantMode, delta_q: i32) {
    let (len, cw, cb_off) = acpl_hcb_arrays(
        crate::acpl::AcplDataType::Beta3,
        qm,
        crate::acpl::AcplHcbType::Df,
    );
    let idx = (delta_q + cb_off).clamp(0, (len.len() as i32) - 1) as usize;
    bw.write_u32(cw[idx], len[idx] as u32);
}

/// Emit a real-α `acpl_data_1ch()` body per §4.2.13.3 Table 61 with the
/// β / β3 / γ entries kept at the round-95 zero-delta scaffold.
///
/// Body layout (matches [`write_acpl_data_1ch_minimal`] but with α
/// carrying real per-band values):
///
/// ```text
///   acpl_framing_data(): smooth interp + num_param_sets = 1
///   acpl_ec_data(ALPHA):
///     diff_type = 0 (DIFF_FREQ)
///     F0 codeword (alpha_q[start_band])
///     DF codewords (delta_q[pb] = alpha_q[pb] - alpha_q[pb-1])
///   acpl_ec_data(BETA): zero-delta F0 + DF (β = 0 everywhere)
/// ```
///
/// `alpha_q_per_band` must be length ≥ `num_bands`; entries below
/// `start_band` are ignored. The decoder's `parse_acpl_huff_data` walks
/// `(num_bands - start_band)` codewords per `acpl_ec_data`.
fn write_acpl_data_1ch_real_alpha(
    bw: &mut BitWriter,
    num_bands: u32,
    start_band: u32,
    quant_mode: crate::acpl::AcplQuantMode,
    alpha_q_per_band: &[i32],
) {
    write_acpl_data_1ch_real_alpha_beta(
        bw,
        num_bands,
        start_band,
        quant_mode,
        alpha_q_per_band,
        None,
    );
}

/// Emit an `acpl_data_1ch()` body per §4.2.13.3 Table 61 with real per-
/// band α and (optionally) real per-band β. When `beta_q_per_band` is
/// `None` the β layer falls back to the zero-delta scaffold (identical
/// behaviour to [`write_acpl_data_1ch_real_alpha`]).
///
/// The β codebook (Tables A.40 / A.41) addresses F0 by `symbol_index =
/// beta_q` (cb_off = 0) so the F0 codeword carries the non-negative
/// magnitude directly. The DF codebook supports signed deltas; here we
/// chain a forward `delta_q = beta_q[pb] − beta_q[pb-1]` which the
/// decoder reverses via `acpl_synth::differential_decode`'s DIFF_FREQ
/// branch.
pub(crate) fn write_acpl_data_1ch_real_alpha_beta(
    bw: &mut BitWriter,
    num_bands: u32,
    start_band: u32,
    quant_mode: crate::acpl::AcplQuantMode,
    alpha_q_per_band: &[i32],
    beta_q_per_band: Option<&[i32]>,
) {
    // acpl_framing_data(): smooth interp (1 b) + num_param_sets_cod = 0 (1 b).
    bw.write_bit(false);
    bw.write_bit(false);

    // acpl_ec_data(ALPHA): diff_type = 0, then F0 + (n - 1) × DF.
    bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
    let mut prev_q: i32 = 0;
    let mut first = true;
    for pb in start_band..num_bands {
        let a_q = alpha_q_per_band.get(pb as usize).copied().unwrap_or(0);
        if first {
            write_acpl_alpha_f0_value(bw, quant_mode, a_q);
            first = false;
        } else {
            let delta = a_q - prev_q;
            write_acpl_alpha_df_value(bw, quant_mode, delta);
        }
        prev_q = a_q;
    }

    // acpl_ec_data(BETA): diff_type = 0, then F0 + (n - 1) × DF.
    bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
    if let Some(beta_q) = beta_q_per_band {
        let mut prev_q: i32 = 0;
        let mut first = true;
        for pb in start_band..num_bands {
            let b_q = beta_q.get(pb as usize).copied().unwrap_or(0);
            if first {
                write_acpl_beta_f0_value(bw, quant_mode, b_q);
                first = false;
            } else {
                let delta = b_q - prev_q;
                write_acpl_beta_df_value(bw, quant_mode, delta);
            }
            prev_q = b_q;
        }
    } else {
        if num_bands > start_band {
            write_acpl_f0_zero(bw, crate::acpl::AcplDataType::Beta, quant_mode);
        }
        for _ in (start_band + 1)..num_bands {
            write_acpl_df_zero(bw, crate::acpl::AcplDataType::Beta, quant_mode);
        }
    }
}

/// Build a 5_X SIMPLE/ASPX_ACPL_1 substream body identical to
/// [`build_5_x_acpl1_body_from_pcm_spectra`] but with real per-parameter-
/// band α coefficients extracted from the (L, Ls) and (R, Rs) MDCT energy
/// ratios. β / β3 / γ stay at the zero-delta scaffold (round 95 / 100 /
/// 103). The decoder's [`crate::acpl_synth::run_acpl_5x_pair_pcm`] applies
/// the recovered α to the §5.7.7.5 Pseudocode-116 mix:
///
/// ```text
///   z1 (= Ls_recon)  =  (1/√2) · 0.5 · (x0·(1-α) - y·β)
/// ```
///
/// With β = 0 the Ls / Rs reconstruction is a pure level-only image:
///
/// ```text
///   Ls_recon  =  0.5/√2 · L · (1 − α_1)
///   Rs_recon  =  0.5/√2 · R · (1 − α_2)
/// ```
///
/// — the encoder's α picks the value that minimises
/// `(L · (1 − α)/(2√2) − Ls)²` per parameter band.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl1_body_from_pcm_spectra_real_alpha(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
    let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);

    // Per-band α extraction for the two D0 / D1 ACplModule's.
    let (num_l, den_l) = compute_per_band_correlations(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (num_r, den_r) = compute_per_band_correlations(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_real = analytic_alpha_per_band(&num_l, &den_l, acpl_quant_mode);
    let alpha_r_real = analytic_alpha_per_band(&num_r, &den_r, acpl_quant_mode);
    let alpha_l_q: Vec<i32> = alpha_l_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();
    let alpha_r_q: Vec<i32> = alpha_r_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_1 (2) — 3 bits.
    bw.write_u32(2, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }
    write_companding_control_2ch_sync_on(&mut bw);
    bw.write_bit(false); // coding_config = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_acpl_1_residual_layer(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_ls,
        coeffs_rs,
    );
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
        );
        write_acpl_data_1ch_real_alpha(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
        );
    }

    finish_substream_body(bw)
}

// ====================================================================
// Round 132 — real per-parameter-band α + β extractor (ASPX_ACPL_1)
// ====================================================================

/// Build a 5_X SIMPLE/ASPX_ACPL_1 substream body identical to
/// [`build_5_x_acpl1_body_from_pcm_spectra_real_alpha`] but with real
/// per-parameter-band β magnitudes additionally extracted from the
/// surround-vs-carrier energy residual after α removes the level-only
/// component.
///
/// Per Pseudocode 116 with `y` ⊥ `x0` and `E[y²] ≈ E[x0²]`:
///
/// ```text
///   E[Ls²] = 0.5 · E[x0²] · ( (1 − α)² + β² )
///   ⇒  β = √max(0, 2·E[Ls²]/E[x0²] − (1 − α)²)
/// ```
///
/// The encoder emits the resulting non-negative β_q via the BETA F0 +
/// DF codebooks (Tables A.40 / A.41 — F0 cb_off = 0 carries the
/// magnitude directly). The decoder reverses this via
/// `acpl_synth::differential_decode` (DIFF_FREQ) and
/// `dequantize_beta_index` (column-0 of the Table 204 / 206 grid for
/// the magnitude lookup).
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_c: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
    let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);

    // α — same MDCT-energy correlation step as the round-128 path.
    let (num_l, den_l) = compute_per_band_correlations(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (num_r, den_r) = compute_per_band_correlations(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let alpha_l_real = analytic_alpha_per_band(&num_l, &den_l, acpl_quant_mode);
    let alpha_r_real = analytic_alpha_per_band(&num_r, &den_r, acpl_quant_mode);
    let alpha_l_q: Vec<i32> = alpha_l_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();
    let alpha_r_q: Vec<i32> = alpha_r_real
        .iter()
        .map(|&a| quantise_alpha(a, acpl_quant_mode))
        .collect();

    // β — energy residual after α removes the level-only component.
    let (e_c_l, e_s_l) = compute_per_band_energies(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    let (e_c_r, e_s_r) = compute_per_band_energies(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
    );
    // Use the *dequantised* α (the value the decoder will see) so that
    // β closes the energy balance against the actually-reconstructed
    // (1 − α_dq), not the analytic α.
    let alpha_l_dq: Vec<f32> = alpha_l_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let alpha_r_dq: Vec<f32> = alpha_r_q
        .iter()
        .map(|&q| crate::acpl_synth::dequantize_alpha_index(acpl_quant_mode, q).0)
        .collect();
    let beta_l_real = analytic_beta_per_band(&e_c_l, &e_s_l, &alpha_l_dq, acpl_quant_mode);
    let beta_r_real = analytic_beta_per_band(&e_c_r, &e_s_r, &alpha_r_dq, acpl_quant_mode);
    let beta_l_q: Vec<i32> = beta_l_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();
    let beta_r_q: Vec<i32> = beta_r_real
        .iter()
        .map(|&b| quantise_beta_magnitude(b, acpl_quant_mode))
        .collect();

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 5_X_codec_mode = ASPX_ACPL_1 (2) — 3 bits.
    bw.write_u32(2, 3);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }
    write_companding_control_2ch_sync_on(&mut bw);
    bw.write_bit(false); // coding_config = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_acpl_1_residual_layer(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_ls,
        coeffs_rs,
    );
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

// ====================================================================
// 7_X ASPX_ACPL_2 emitter — §4.2.6.14 Table 33 row `case ASPX_ACPL_2:`
// (round 107)
// ====================================================================

/// Build a 7.0 SIMPLE/ASPX_ACPL_2 substream body per §4.2.6.14 Table 33
/// row `case ASPX_ACPL_2:` that the decoder's
/// [`crate::mch::parse_7x_audio_data_outer`] (with `mode = AspxAcpl2`)
/// walks end-to-end. The 7_X immersive channel element shares the same
/// 1ch ACPL / ASPX parameter shape as the round-100 5_X ASPX_ACPL_2 path
/// (Pseudocode 117) but differs in five structural places versus the
/// 5_X walker:
///
/// 1. `7_X_codec_mode` is **2 bits** (vs the 5_X 3-bit `5_X_codec_mode`).
///    ASPX_ACPL_2 = value 3.
/// 2. `companding_control(5)` sits **before** `coding_config` (the 5_X
///    ASPX_ACPL_{1,2} walker reads `companding_control(3)` first too, but
///    the 7_X `num_chan` argument is 5 — with `sync_flag = 1` the wire
///    shape is identical: 1 sync bit + 1 on bit).
/// 3. `coding_config` is **2 bits** (Table 33 4-way selector) — Cfg0 = 0.
/// 4. Cfg0 body is `b_2ch_mode + two_channel_data + two_channel_data`
///    (two stereo pairs — L/R then Ls/Rs), and the centre `mono_data(0)`
///    moves **out** of the coding_config switch to a single trailing
///    element after the (skipped-for-ACPL_2) additional-channel block.
/// 5. The I-frame ASPX trailer is `aspx_data_2ch() + aspx_data_2ch() +
///    aspx_data_1ch()` (two 2ch + one 1ch — the 5_X ACPL_2 path emits a
///    single `aspx_data_2ch() + aspx_data_1ch()`). The extra 2ch envelope
///    covers the second stereo pair.
///
/// Body layout (Table 33, `coding_config = 0`):
///
/// ```text
///   7_X_codec_mode = ASPX_ACPL_2 (3)        // 2 b
///   if (b_iframe) {
///       aspx_config();                       // 15 b — Table 50
///       acpl_config_1ch(FULL);               //  3 b — Table 59
///   }
///   if (b_has_lfe) mono_data(1);             // LFE (7.1) — Table 21
///   companding_control(5);                   // sync = 1, on = 1 — Table 49
///   coding_config = 0;                        //  2 b
///   b_2ch_mode;                               //  1 b
///   two_channel_data();                       // L/R carriers — Table 26
///   two_channel_data();                       // Ls/Rs carriers — Table 26
///   // (additional-channel block SKIPPED for ASPX_ACPL_2)
///   // (ASPX_ACPL_1 joint-MDCT residual layer SKIPPED for ACPL_2)
///   mono_data(0);                             // centre (Cfg0) — Table 21
///   if (b_iframe) {
///       aspx_data_2ch();                     // Table 52 — L/R envelope
///       aspx_data_2ch();                     // Table 52 — Ls/Rs envelope
///       aspx_data_1ch();                     // Table 51 — centre envelope
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[0]
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[1]
///   }
/// ```
///
/// `coeffs_l` / `coeffs_r` are the forward-MDCT L/R carrier spectra
/// (first `two_channel_data`); `coeffs_ls` / `coeffs_rs` are the surround
/// carriers (second `two_channel_data`); `coeffs_c` is the centre coded
/// via the trailing Cfg0 `mono_data(0)`. The decoder's 7_X ACPL_2
/// dispatch reconstructs the Ls/Rs PCM from the L/R carriers + the two
/// `acpl_data_1ch()` parameter sets — the second `two_channel_data()`
/// keeps the body well-formed for the walker.
///
/// When `coeffs_lfe` + `max_sfb_lfe` are both `Some`, an LFE
/// `mono_data(b_lfe = 1)` element (Table 21 + `sf_info_lfe()` Table 35)
/// is emitted between the I-frame config block and `companding_control(5)`
/// — exactly where the decoder's
/// [`crate::mch::parse_7x_audio_data_outer`] reads `if (b_has_lfe)
/// mono_data(1);` (§4.2.6.14 Table 33). This is the 7.1 (3/4/0.1) path:
/// the caller must drive the TOC channel_mode prefix to 8 channels so the
/// decoder dispatches `channels == 8` through
/// `parse_7x_audio_data_outer(b_has_lfe = true)`. With both `None` the
/// body is the round-107 7.0 (3/4/0) ASPX_ACPL_2 form.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl2_body_from_pcm_spectra(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_2 (3) — 2 bits.
    bw.write_u32(3, 2);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(FULL) (3 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }

    // LFE: mono_data(b_lfe = 1) when present (7.1 / channel_mode 6). The
    // decoder's parse_7x_audio_data_outer reads this immediately after the
    // I-frame config block and before companding_control(5) — §4.2.6.14
    // Table 33 `if (b_has_lfe) mono_data(1);`.
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(5): sync = 1, on = 1 — same 2-bit wire shape as
    // the 5_X companding_control(2/3) sync-on case (the `num_chan`
    // argument only changes how many `b_compand_on` bits follow when
    // `sync_flag == 0`; here sync_flag == 1 so exactly one bit follows).
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (2 b) — Cfg0.
    bw.write_u32(0, 2);

    // Cfg0: b_2ch_mode (1 b) + two_channel_data (L/R) + two_channel_data
    // (Ls/Rs).
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    // (Additional-channel block SKIPPED for ASPX_ACPL_2.)
    // (ASPX_ACPL_1 residual layer SKIPPED for ACPL_2.)

    // Trailing Cfg0 mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers: aspx_data_2ch + aspx_data_2ch +
    // aspx_data_1ch + acpl_data_1ch × 2.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        // acpl_config_1ch(FULL) has no qmf_band → start_band = 0.
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, 0, acpl_quant_mode);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, 0, acpl_quant_mode);
    }

    finish_substream_body(bw)
}

// ====================================================================
// 7_X ASPX_ACPL_2 emitter — real per-band α + β (round 202)
// ====================================================================

/// Build a 7.0 / 7.1 SIMPLE/ASPX_ACPL_2 substream body per §4.2.6.14
/// Table 33 row `case ASPX_ACPL_2:` with **real per-parameter-band α + β**
/// extracted from the (L, Ls) and (R, Rs) MDCT pairs. The 7_X (immersive)
/// counterpart to the round-144 5_X path
/// ([`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta`]) and the
/// real-α-β upgrade of the round-107 / 114 zero-delta 7_X ACPL_2 builder
/// ([`build_7_x_acpl2_body_from_pcm_spectra`]).
///
/// ACPL_2 does **not** transmit the Ls/Rs surround pair on the wire — the
/// decoder reconstructs the surround from the L/R carriers + the two
/// `acpl_data_1ch()` parameter sets per ETSI TS 103 190-1 §5.7.7.5
/// Pseudocode 116 + §5.7.7.6.1 Pseudocode 117:
///
/// ```text
///   z0 = 0.5 · (x0·(1+α) + y·β)        // recovers L  carrier
///   z1 = 0.5 · (x0·(1−α) − y·β)        // recovers Ls (then ·√2)
/// ```
///
/// The encoder accepts the caller's full L / R / C / Ls / Rs (+ optional
/// LFE) spectra; the Ls / Rs spectra feed the α + β extractors only and
/// are not emitted on the ACPL_2 wire. D0 module models (L → Ls); D1
/// module models (R → Rs). `acpl_config_1ch(FULL)` carries no `qmf_band`
/// → `start_band = 0` so every parameter band participates (in contrast
/// to the ACPL_1 PARTIAL mode whose `acpl_qmf_band` masks the low bands).
///
/// All five structural differences versus the round-118 7_X ACPL_1
/// walker (2-bit `7_X_codec_mode = 3`, optional LFE `mono_data(1)`, two
/// `two_channel_data()` pairs, **no** joint-MDCT residual layer, trailing
/// centre `mono_data`) are unchanged from
/// [`build_7_x_acpl2_body_from_pcm_spectra`]. The only difference: each
/// trailing `acpl_data_1ch()` now carries the analytic α + β indices
/// rather than the zero-delta scaffold.
///
/// When `coeffs_lfe` + `max_sfb_lfe` are both `Some`, an LFE
/// `mono_data(b_lfe = 1)` element (Table 21 + `sf_info_lfe()` Table 35)
/// is emitted between the I-frame config block and `companding_control(5)`,
/// exactly where the decoder's `parse_7x_audio_data_outer(b_has_lfe =
/// true)` reads `if (b_has_lfe) mono_data(1);` (§4.2.6.14 Table 33). This
/// is the 7.1 (3/4/0.1) path; with both `None` the body is the 7.0
/// (3/4/0) form.
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    // acpl_config_1ch(FULL) carries no qmf_band → start_band = 0.
    let start_band = 0u32;

    // α + β extraction — identical primitives to the round-144 5_X ACPL_2
    // path. D0 module models (L → Ls); D1 module models (R → Rs).
    let alpha_l_q = extract_alpha_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let alpha_r_q = extract_alpha_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let beta_l_q = extract_beta_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_l_q,
        acpl_quant_mode,
    );
    let beta_r_q = extract_beta_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_r_q,
        acpl_quant_mode,
    );

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_2 (3) — 2 bits.
    bw.write_u32(3, 2);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(FULL) (3 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }

    // LFE: mono_data(b_lfe = 1) when present (7.1 / channel_mode 6).
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(5): sync = 1, on = 1.
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (2 b) — Cfg0.
    bw.write_u32(0, 2);

    // Cfg0: b_2ch_mode (1 b) + two_channel_data (L/R) + two_channel_data (Ls/Rs).
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    // (Additional-channel block SKIPPED for ASPX_ACPL_2.)
    // (ASPX_ACPL_1 residual layer SKIPPED for ACPL_2.)

    // Trailing Cfg0 mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers: aspx_data_2ch + aspx_data_2ch +
    // aspx_data_1ch + acpl_data_1ch × 2 (now carrying real α + β).
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// Build a 7.0 / 7.1 SIMPLE/ASPX_ACPL_2 substream body identical in
/// framing to [`build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta`]
/// except the I-frame's **three** ASPX envelope elements
/// (`aspx_data_2ch()` for the L/R front pair, `aspx_data_2ch()` for the
/// Ls/Rs surround pair, `aspx_data_1ch()` for the centre carrier) carry
/// caller-provided **real** SIGNAL / NOISE envelope quant-index vectors
/// via [`write_aspx_data_2ch_real_envelope`] /
/// [`write_aspx_data_1ch_real_envelope`] instead of the round-107
/// minimum-bit-cost [`write_aspx_data_2ch_minimal`] /
/// [`write_aspx_data_1ch_minimal`] scaffolds.
///
/// This is the 7_X (immersive) counterpart of the round-331 5_X
/// [`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`].
/// Where the 5_X ACPL_2 body has a single `aspx_data_2ch()` (the L/R
/// carrier pair) plus the centre `aspx_data_1ch()`, the 7_X ACPL_2 body
/// carries an additional `aspx_data_2ch()` for the Ls/Rs pair — matching
/// the decoder's `parse_7x_audio_data_outer` trailer walk
/// (`aspx_data_2ch + aspx_data_2ch + aspx_data_1ch`).
///
/// `l_sig` / `l_noise` / `r_sig` / `r_noise` feed the first 2ch element
/// (channel 0 = L carrier, channel 1 = R carrier); `ls_sig` / `ls_noise`
/// / `rs_sig` / `rs_noise` feed the second 2ch element (channel 0 = Ls,
/// channel 1 = Rs); `c_sig` / `c_noise` feed the centre `aspx_data_1ch()`.
/// Every other element (companding, the two `two_channel_data()` carrier
/// pairs, the optional LFE `mono_data(1)`, the centre `mono_data(0)`, the
/// two real-α/β `acpl_data_1ch()` parameter sets) stays byte-for-byte
/// identical to [`build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta`],
/// so the round-trip continues to parse through
/// [`crate::asf::parse_aspx_data_2ch_body`] +
/// [`crate::asf::parse_aspx_data_1ch_body`] and synthesise the 7-channel
/// PCM.
///
/// Refs ETSI TS 103 190-1 §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
/// (`aspx_data_2ch()`).
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    l_sig: &[i32],
    l_noise: &[i32],
    r_sig: &[i32],
    r_noise: &[i32],
    ls_sig: &[i32],
    ls_noise: &[i32],
    rs_sig: &[i32],
    rs_noise: &[i32],
    c_sig: &[i32],
    c_noise: &[i32],
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    // acpl_config_1ch(FULL) carries no qmf_band → start_band = 0.
    let start_band = 0u32;

    // α + β extraction — identical primitives to the round-202 7_X ACPL_2
    // real-α/β path. D0 module models (L → Ls); D1 module models (R → Rs).
    let alpha_l_q = extract_alpha_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let alpha_r_q = extract_alpha_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let beta_l_q = extract_beta_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_l_q,
        acpl_quant_mode,
    );
    let beta_r_q = extract_beta_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_r_q,
        acpl_quant_mode,
    );

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_2 (3) — 2 bits.
    bw.write_u32(3, 2);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(FULL) (3 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }

    // LFE: mono_data(b_lfe = 1) when present (7.1 / channel_mode 6).
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(5): sync = 1, on = 1.
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (2 b) — Cfg0.
    bw.write_u32(0, 2);

    // Cfg0: b_2ch_mode (1 b) + two_channel_data (L/R) + two_channel_data (Ls/Rs).
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    // Trailing Cfg0 mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers: aspx_data_2ch (L/R real envelope) +
    // aspx_data_2ch (Ls/Rs real envelope) + aspx_data_1ch (centre real
    // envelope) + acpl_data_1ch × 2 (real α + β).
    if b_iframe {
        write_aspx_data_2ch_real_envelope(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: l_sig,
                noise: l_noise,
            },
            AspxRealEnvelopeChannel {
                sig: r_sig,
                noise: r_noise,
            },
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_2ch_real_envelope(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: ls_sig,
                noise: ls_noise,
            },
            AspxRealEnvelopeChannel {
                sig: rs_sig,
                noise: rs_noise,
            },
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_1ch_real_envelope(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: c_sig,
                noise: c_noise,
            },
        )
        .expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// `aspx_tna_mode`-carrying variant of
/// [`build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`].
///
/// Identical to the base 7_X ACPL_2 real-ASPX builder except all three
/// A-SPX trailers carry caller-supplied real per-noise-subband-group
/// `aspx_tna_mode` inverse-filtering vectors instead of the all-zero
/// scaffold: the L / R front-pair `aspx_data_2ch()` carries `front_tna_mode`
/// (channel 0, mirrored to channel 1 under `aspx_balance = 1`), the Ls / Rs
/// surround-pair `aspx_data_2ch()` carries `surround_tna_mode`, and the
/// centre `aspx_data_1ch()` carries `c_tna_mode`. Each carrier's tna_mode is
/// derived independently by the encoder from that carrier's own QMF low
/// band, so the three vectors need not match.
///
/// Passing empty (or all-zero) tna_mode slices reproduces the
/// [`build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`] bytes
/// exactly — the only wire difference is the 2-bit-per-noise-SBG
/// `aspx_tna_mode` field the decoder's `parse_aspx_hfgen_iwc_{1,2}ch`
/// recovers and feeds into the §5.7.6.4.1.3 chirp / order-2 LPC inverse
/// filtering.
///
/// Refs ETSI TS 103 190-1 §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
/// (`aspx_data_2ch()`), §4.3.10.6.1 Table 131 (`aspx_tna_mode`).
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    l_sig: &[i32],
    l_noise: &[i32],
    r_sig: &[i32],
    r_noise: &[i32],
    ls_sig: &[i32],
    ls_noise: &[i32],
    rs_sig: &[i32],
    rs_noise: &[i32],
    c_sig: &[i32],
    c_noise: &[i32],
    front_tna_mode: &[u8],
    surround_tna_mode: &[u8],
    c_tna_mode: &[u8],
    l_ah: &[bool],
    r_ah: &[bool],
    ls_ah: &[bool],
    rs_ah: &[bool],
    c_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let start_band = 0u32;

    // α + β extraction — identical to the base 7_X ACPL_2 real-ASPX builder.
    let alpha_l_q = extract_alpha_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let alpha_r_q = extract_alpha_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let beta_l_q = extract_beta_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_l_q,
        acpl_quant_mode,
    );
    let beta_r_q = extract_beta_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_r_q,
        acpl_quant_mode,
    );

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_2 (3) — 2 bits.
    bw.write_u32(3, 2);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }

    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    write_companding_control_2ch_sync_on(&mut bw);

    bw.write_u32(0, 2); // coding_config = 0 — Cfg0
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // Data elements are present on every frame per Tables 25/33;
    // only the configs and the per-element 3-bit xover offset are
    // I-frame-gated (framed writers skip the xover when
    // b_iframe == 0; the decoder reuses the sticky value).
    {
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: l_sig,
                noise: l_noise,
            },
            AspxRealEnvelopeChannel {
                sig: r_sig,
                noise: r_noise,
            },
            front_tna_mode,
            l_ah,
            r_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: ls_sig,
                noise: ls_noise,
            },
            AspxRealEnvelopeChannel {
                sig: rs_sig,
                noise: rs_noise,
            },
            surround_tna_mode,
            ls_ah,
            rs_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_1ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: c_sig,
                noise: c_noise,
            },
            c_tna_mode,
            c_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// Multi-envelope **centre-carrier** variant of
/// [`build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`] — the
/// 7_X dual of
/// [`build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env`].
///
/// The 7_X ASPX_ACPL_2 body carries three A-SPX trailers — `aspx_data_2ch()`
/// for the L / R front pair, `aspx_data_2ch()` for the Ls / Rs surround
/// pair, and `aspx_data_1ch()` for the centre carrier. This variant emits a
/// **multi-envelope** `aspx_data_1ch()` (`num_env > 1`) for the centre via
/// [`write_aspx_data_1ch_multi_envelope`] when the caller's transient probe
/// selected `num_env > 1`; both carrier pairs retain their single-envelope
/// `aspx_data_2ch()` (the decoder reads `num_env` independently per A-SPX
/// element).
///
/// `c_num_env` must be a power of two within the config's FIXFIX capacity,
/// and `centre` carries the per-envelope SIGNAL / NOISE DPCM rows from
/// [`build_aspx_multi_envelope_channel_from_qmf`]. Returns an **empty
/// `Vec`** when [`write_aspx_data_1ch_multi_envelope`] rejects the config /
/// `num_env`, so the caller can fall back to the single-envelope builder.
/// Every other element is byte-for-byte identical to the single-envelope
/// 7_X builder.
///
/// Refs ETSI TS 103 190-1 §4.2.6.14 Table 33 (`case ASPX_ACPL_2:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.3.10.4.11 (`aspx_num_env`).
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    l_sig: &[i32],
    l_noise: &[i32],
    r_sig: &[i32],
    r_noise: &[i32],
    ls_sig: &[i32],
    ls_noise: &[i32],
    rs_sig: &[i32],
    rs_noise: &[i32],
    c_num_env: u32,
    centre: AspxMultiEnvelopeChannel<'_>,
    l_ah: &[bool],
    r_ah: &[bool],
    ls_ah: &[bool],
    rs_ah: &[bool],
    c_ah: &[bool],
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let start_band = 0u32;

    let alpha_l_q = extract_alpha_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let alpha_r_q = extract_alpha_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let beta_l_q = extract_beta_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_l_q,
        acpl_quant_mode,
    );
    let beta_r_q = extract_beta_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_r_q,
        acpl_quant_mode,
    );

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_2 (3) — 2 bits.
    bw.write_u32(3, 2);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_full(&mut bw, acpl_num_param_bands_id, acpl_quant_mode);
    }

    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    write_companding_control_2ch_sync_on(&mut bw);

    bw.write_u32(0, 2); // coding_config = 0
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // Data elements are present on every frame per Tables 25/33;
    // only the configs and the per-element 3-bit xover offset are
    // I-frame-gated (framed writers skip the xover when
    // b_iframe == 0; the decoder reuses the sticky value).
    {
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: l_sig,
                noise: l_noise,
            },
            AspxRealEnvelopeChannel {
                sig: r_sig,
                noise: r_noise,
            },
            &[],
            l_ah,
            r_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            AspxRealEnvelopeChannel {
                sig: ls_sig,
                noise: ls_noise,
            },
            AspxRealEnvelopeChannel {
                sig: rs_sig,
                noise: rs_noise,
            },
            &[],
            ls_ah,
            rs_ah,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        // Multi-envelope centre `aspx_data_1ch()`. A rejection signals the
        // caller to fall back to the single-envelope 7_X builder.
        if write_aspx_data_1ch_multi_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            c_num_env,
            centre,
            &[],
            c_ah,
            b_iframe,
        )
        .is_err()
        {
            return Vec::new();
        }
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

// ====================================================================
// 7_X ASPX_ACPL_1 emitter — §4.2.6.14 Table 33 row `case ASPX_ACPL_1:`
// (round 118)
// ====================================================================

/// Build a 7.0 / 7.1 SIMPLE/ASPX_ACPL_1 substream body per §4.2.6.14
/// Table 33 row `case ASPX_ACPL_1:` that the decoder's
/// [`crate::mch::parse_7x_audio_data_outer`] (with `mode = AspxAcpl1`)
/// walks end-to-end. The 7_X (immersive) counterpart to the round-103
/// 5_X ASPX_ACPL_1 path and the encoder side of the decoder's round-27
/// `parse_7x_audio_data_outer` ASPX_ACPL_1 branch (which already reads
/// the joint-MDCT residual layer at §4.2.6.14 Table 33).
///
/// Structurally this is the round-107/114 7_X ASPX_ACPL_2 body with three
/// differences (the same three that separate the 5_X ACPL_1 path from the
/// 5_X ACPL_2 path):
///
/// 1. `7_X_codec_mode` is **2** (ASPX_ACPL_1) rather than 3 (ASPX_ACPL_2).
/// 2. `acpl_config_1ch` is **PARTIAL** (Table 59, 6 b — carries the 3-bit
///    `acpl_qmf_band_minus1` field FULL omits), so the `acpl_data_1ch()`
///    start_band resolves from `qmf_band` via [`crate::acpl::sb_to_pb`].
/// 3. The body carries an explicit **joint-MDCT residual layer**
///    (`max_sfb_master + 2× chparam_info + 2× sf_data(ASF)`) transmitting
///    the Ls/Rs surround pair (sSMP,3 / sSMP,4 per Table 181) after the two
///    `two_channel_data()` pairs and before the trailing Cfg0 centre
///    `mono_data(0)` — exactly where the decoder's residual-layer walk
///    sits (`if (mode == ASPX_ACPL_1) { … }`).
///
/// Body layout (Table 33, `coding_config = 0`):
///
/// ```text
///   7_X_codec_mode = ASPX_ACPL_1 (2)        // 2 b
///   if (b_iframe) {
///       aspx_config();                       // 15 b — Table 50
///       acpl_config_1ch(PARTIAL);            //  6 b — Table 59
///   }
///   if (b_has_lfe) mono_data(1);             // LFE (7.1) — Table 21
///   companding_control(5);                   // sync = 1, on = 1 — Table 49
///   coding_config = 0;                        //  2 b
///   b_2ch_mode;                               //  1 b
///   two_channel_data();                       // L/R carriers — Table 26
///   two_channel_data();                       // Ls/Rs carriers — Table 26
///   // (additional-channel block SKIPPED for ASPX_ACPL_1)
///   max_sfb_master;                           // joint-MDCT residual layer
///   chparam_info(); chparam_info();           // residual ch0 / ch1
///   sf_data(ASF); sf_data(ASF);               // residual sSMP,3 / sSMP,4
///   mono_data(0);                             // centre (Cfg0) — Table 21
///   if (b_iframe) {
///       aspx_data_2ch();                     // Table 52 — L/R envelope
///       aspx_data_2ch();                     // Table 52 — Ls/Rs envelope
///       aspx_data_1ch();                     // Table 51 — centre envelope
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[0]
///       acpl_data_1ch();                     // -> acpl_data_1ch_pair[1]
///   }
/// ```
///
/// `coeffs_l` / `coeffs_r` are the forward-MDCT L/R carrier spectra
/// (first `two_channel_data`); `coeffs_ls` / `coeffs_rs` are the surround
/// carriers — carried *both* by the second `two_channel_data()` (keeps the
/// walker well-formed) *and* by the joint-MDCT residual pair (the
/// surround content the decoder reconstructs). `coeffs_c` is the centre
/// coded via the trailing Cfg0 `mono_data(0)`.
///
/// When `coeffs_lfe` + `max_sfb_lfe` are both `Some`, an LFE
/// `mono_data(b_lfe = 1)` element (Table 21 + `sf_info_lfe()` Table 35) is
/// emitted between the I-frame config block and `companding_control(5)` —
/// exactly where the decoder's `parse_7x_audio_data_outer(b_has_lfe =
/// true)` reads `if (b_has_lfe) mono_data(1);`. This is the 7.1 (3/4/0.1)
/// path: the caller must drive the TOC channel_mode prefix to 8 channels
/// so the decoder dispatches `channels == 8`. With both `None` the body is
/// the 7.0 (3/4/0) form (`channels == 7`).
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl1_body_from_pcm_spectra(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_1 (2) — 2 bits.
    bw.write_u32(2, 2);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(PARTIAL) (6 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }

    // LFE: mono_data(b_lfe = 1) when present (7.1 / channel_mode 6) — read
    // by parse_7x_audio_data_outer immediately after the I-frame config
    // block and before companding_control(5).
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(5): sync = 1, on = 1 — same 2-bit wire shape as
    // the 5_X / 7_X ACPL_2 sync-on case.
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (2 b) — Cfg0.
    bw.write_u32(0, 2);

    // Cfg0: b_2ch_mode (1 b) + two_channel_data (L/R) + two_channel_data
    // (Ls/Rs).
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    // (Additional-channel block SKIPPED for ASPX_ACPL_1 — the decoder only
    // walks it for SIMPLE / Aspx modes.)

    // ASPX_ACPL_1-only joint-MDCT residual layer: Ls/Rs surround residual.
    // The decoder derives n_side from the largest signalled transform
    // length across the channel data — which is `transform_length` (the
    // long-frame two_channel_data() pairs all signal transform_length_0 ==
    // transform_length), so we pass the same value.
    write_acpl_1_residual_layer(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_ls,
        coeffs_rs,
    );

    // Trailing Cfg0 mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers: aspx_data_2ch + aspx_data_2ch +
    // aspx_data_1ch + acpl_data_1ch × 2.
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        // PARTIAL acpl_config_1ch carries a qmf_band → resolve start_band.
        let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
        let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
        write_acpl_data_1ch_minimal(&mut bw, acpl_num_bands, start_band, acpl_quant_mode);
    }

    finish_substream_body(bw)
}

// ====================================================================
// Round 135 — real per-parameter-band α + β extractor (7_X ASPX_ACPL_1)
// ====================================================================

/// Build a 7.0 / 7.1 SIMPLE/ASPX_ACPL_1 substream body identical in wire
/// layout to [`build_7_x_acpl1_body_from_pcm_spectra`] but with **real
/// per-parameter-band α + β** carried by the two trailing
/// `acpl_data_1ch()` parameter sets, exactly as the round-132 5_X path
/// ([`build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta`]) does for
/// the 5.0 immersive element.
///
/// This is the round-132 followup: the 7_X immersive ASPX_ACPL_1 path
/// previously emitted both `acpl_data_1ch()` sets at the round-118
/// zero-delta scaffold ([`write_acpl_data_1ch_minimal`]); here each set
/// instead carries the analytic α (from the L/Ls and R/Rs MDCT-energy
/// correlation, §5.7.7.5 Pseudocode 116) and the analytic β magnitude
/// that closes the surround/carrier energy balance after α removes the
/// level-only component (§5.7.7.6.1 Pseudocode 117):
///
/// ```text
///   E[Ls²] = 0.5 · E[L²] · ( (1 − α)² + β² )
///   ⇒  β = √max(0, 2·E[Ls²]/E[L²] − (1 − α)²)
/// ```
///
/// The decoder's [`crate::mch::parse_7x_audio_data_outer`] (with `mode =
/// AspxAcpl1`) walks the same body; the recovered α/β feed the §5.7.7.6.1
/// `ACplModule(alpha, beta, …)` reconstruction of the Ls/Rs surround pair.
/// β / β3 / γ for non-ACPL_1 paths stay at their respective scaffolds.
///
/// All five structural differences versus the 7_X ACPL_2 walker (2-bit
/// `7_X_codec_mode`, optional LFE `mono_data(1)`, two `two_channel_data()`
/// pairs, the joint-MDCT residual layer, the trailing centre `mono_data`)
/// are unchanged from [`build_7_x_acpl1_body_from_pcm_spectra`].
///
/// Returns the substream bytes sized to `pad_target_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
    let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);

    // α + β extraction — identical primitives to the round-128 / 132 5_X
    // path. D0 module models (L → Ls); D1 module models (R → Rs).
    let alpha_l_q = extract_alpha_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let alpha_r_q = extract_alpha_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let beta_l_q = extract_beta_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_l_q,
        acpl_quant_mode,
    );
    let beta_r_q = extract_beta_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_r_q,
        acpl_quant_mode,
    );

    let mut bw = BitWriter::new();
    // ac4_substream() per §5.7.1: audio_size_value (15 b) + b_more_bits (1 b).
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_1 (2) — 2 bits.
    bw.write_u32(2, 2);

    // I-frame block: aspx_config() (15 b) + acpl_config_1ch(PARTIAL) (6 b).
    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }

    // LFE: mono_data(b_lfe = 1) when present (7.1 / channel_mode 6).
    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    // companding_control(5): sync = 1, on = 1.
    write_companding_control_2ch_sync_on(&mut bw);

    // coding_config = 0 (2 b) — Cfg0.
    bw.write_u32(0, 2);

    // Cfg0: b_2ch_mode (1 b) + two_channel_data (L/R) + two_channel_data (Ls/Rs).
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    // ASPX_ACPL_1-only joint-MDCT residual layer: Ls/Rs surround residual.
    write_acpl_1_residual_layer(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_ls,
        coeffs_rs,
    );

    // Trailing Cfg0 mono_data(0) — centre carrier.
    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // I-frame ASPX + A-CPL trailers: aspx_data_2ch + aspx_data_2ch +
    // aspx_data_1ch + acpl_data_1ch × 2 (now carrying real α + β).
    if b_iframe {
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_2ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_aspx_data_1ch_minimal(&mut bw, aspx_cfg).expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

/// Real-ASPX + `aspx_tna_mode` variant of
/// [`build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta`].
///
/// Identical to that builder — same α + β A-CPL extraction, same Cfg0
/// `two_channel_data()` / residual-layer / centre `mono_data()` body —
/// except the three I-frame ASPX trailers carry **real** per-sbg SIGNAL /
/// NOISE envelopes ([`write_aspx_data_2ch_real_envelope_tna`] /
/// [`write_aspx_data_1ch_real_envelope_tna`]) plus a real per-noise-
/// subband-group `aspx_tna_mode` inverse-filtering vector, instead of the
/// round-118 minimum-bit-cost ASPX scaffold (`write_aspx_data_2ch_minimal`
/// / `write_aspx_data_1ch_minimal`):
///
/// * the L / R front pair `aspx_data_2ch()` carries `front` envelopes +
///   `front_tna_mode`,
/// * the Ls / Rs surround pair `aspx_data_2ch()` carries `surround`
///   envelopes + `surround_tna_mode`,
/// * the centre `aspx_data_1ch()` carries `c` envelopes + `c_tna_mode`.
///
/// Each carrier's envelopes come from
/// [`build_aspx_real_envelope_channel_from_qmf`] over its own input PCM,
/// and its tna_mode from [`crate::aspx_tna_select::select_tna_mode`] over
/// the same carrier's QMF low band. Under `aspx_balance = 1` channel 1 of
/// each pair mirrors channel 0's framing + tna_mode, so a single per-pair
/// vector suffices.
///
/// Passing empty (or all-zero) envelopes + tna slices reproduces the
/// `write_aspx_data_*_minimal` framing exactly, so the decoder still
/// round-trips; the difference is the real envelope values + the 2-bit-
/// per-noise-subband-group `aspx_tna_mode` field the decoder feeds into
/// the §5.7.6.4.1.2-4 chirp / order-2 LPC inverse filter.
///
/// `front_ah` / `surround_ah` / `c_ah` carry the per-channel
/// `aspx_add_harmonic` (§4.2.12.6) decisions (front/surround as `(ch0,
/// ch1)` tuples since it is signalled independently per channel, the
/// centre as a single vector) — typically from
/// [`crate::aspx_ah_select::select_add_harmonic`] over each carrier's HF
/// QMF band. Empty slices reproduce the all-zero `add_harmonic` scaffold.
///
/// Refs ETSI TS 103 190-1 §4.2.6.14 Table 33 (`case ASPX_ACPL_1:`),
/// §4.2.12.3 Table 51 (`aspx_data_1ch()`), §4.2.12.4 Table 52
/// (`aspx_data_2ch()`), §4.3.10.6.1 Table 131 (`aspx_tna_mode`),
/// §4.2.12.6 (`aspx_hfgen_iwc`).
#[allow(clippy::too_many_arguments)]
pub fn build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna(
    transform_length: u32,
    max_sfb: u32,
    max_sfb_master: u32,
    max_sfb_lfe: Option<u32>,
    b_iframe: bool,
    coeffs_l: &[f32],
    coeffs_r: &[f32],
    coeffs_ls: &[f32],
    coeffs_rs: &[f32],
    coeffs_c: &[f32],
    coeffs_lfe: Option<&[f32]>,
    aspx_cfg: &aspx::AspxConfig,
    acpl_num_param_bands_id: u8,
    acpl_quant_mode: crate::acpl::AcplQuantMode,
    acpl_qmf_band_minus1: u8,
    front: (AspxRealEnvelopeChannel<'_>, AspxRealEnvelopeChannel<'_>),
    surround: (AspxRealEnvelopeChannel<'_>, AspxRealEnvelopeChannel<'_>),
    c: AspxRealEnvelopeChannel<'_>,
    front_tna_mode: &[u8],
    surround_tna_mode: &[u8],
    c_tna_mode: &[u8],
    front_ah: (&[bool], &[bool]),
    surround_ah: (&[bool], &[bool]),
    c_ah: &[bool],
    pad_target_bytes: usize,
) -> Vec<u8> {
    let acpl_num_bands = crate::acpl::num_param_bands_from_id(acpl_num_param_bands_id as u32);
    let qmf_band = (acpl_qmf_band_minus1 as u32 & 0b111) + 1;
    let start_band = crate::acpl::sb_to_pb(qmf_band, acpl_num_bands);

    // α + β extraction — identical primitives to the real_alpha_beta path.
    let alpha_l_q = extract_alpha_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let alpha_r_q = extract_alpha_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        acpl_quant_mode,
    );
    let beta_l_q = extract_beta_q_per_band(
        coeffs_l,
        coeffs_ls,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_l_q,
        acpl_quant_mode,
    );
    let beta_r_q = extract_beta_q_per_band(
        coeffs_r,
        coeffs_rs,
        transform_length,
        acpl_num_bands,
        start_band,
        &alpha_r_q,
        acpl_quant_mode,
    );

    let mut bw = BitWriter::new();
    let audio_size = pad_target_bytes as u32;
    bw.write_u32(audio_size & 0x7FFF, 15);
    bw.write_bit(false);
    bw.align_to_byte();

    // 7_X_codec_mode = ASPX_ACPL_1 (2) — 2 bits.
    bw.write_u32(2, 2);

    if b_iframe {
        write_aspx_config(&mut bw, aspx_cfg);
        write_acpl_config_1ch_partial(
            &mut bw,
            acpl_num_param_bands_id,
            acpl_quant_mode,
            acpl_qmf_band_minus1,
        );
    }

    if let (Some(lfe), Some(m_lfe)) = (coeffs_lfe, max_sfb_lfe) {
        write_lfe_mono_data(&mut bw, transform_length, m_lfe, lfe);
    }

    write_companding_control_2ch_sync_on(&mut bw);

    bw.write_u32(0, 2); // coding_config = 0 (Cfg0)
    bw.write_bit(false); // b_2ch_mode = 0
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_l, coeffs_r);
    write_two_channel_data(&mut bw, transform_length, max_sfb, coeffs_ls, coeffs_rs);

    write_acpl_1_residual_layer(
        &mut bw,
        transform_length,
        max_sfb_master,
        coeffs_ls,
        coeffs_rs,
    );

    write_mono_data_centre(&mut bw, transform_length, max_sfb, coeffs_c);

    // Data elements are present on every frame per Tables 25/33;
    // only the configs and the per-element 3-bit xover offset are
    // I-frame-gated (framed writers skip the xover when
    // b_iframe == 0; the decoder reuses the sticky value).
    {
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            front.0,
            front.1,
            front_tna_mode,
            front_ah.0,
            front_ah.1,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_2ch_real_envelope_tna_ah_framed(
            &mut bw,
            aspx_cfg,
            surround.0,
            surround.1,
            surround_tna_mode,
            surround_ah.0,
            surround_ah.1,
            b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_aspx_data_1ch_real_envelope_tna_ah_framed(
            &mut bw, aspx_cfg, c, c_tna_mode, c_ah, b_iframe,
        )
        .expect("encoder: aspx config invalid");
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_l_q,
            Some(&beta_l_q),
        );
        write_acpl_data_1ch_real_alpha_beta(
            &mut bw,
            acpl_num_bands,
            start_band,
            acpl_quant_mode,
            &alpha_r_q,
            Some(&beta_r_q),
        );
    }

    finish_substream_body(bw)
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitReader;

    fn r389_test_cfg() -> aspx::AspxConfig {
        aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        }
    }

    #[test]
    fn balance_encode_sig_rows_round_trips_through_p84() {
        // (L, R) LEVEL rows → (sum, pan) → Pseudocode 84 joint
        // dequantization must recover each channel's LEVEL scf within
        // the (sum, pan) grid's quantization error: half a sum step
        // (0,75 dB Fine) plus half a pan wire step (1,5 dB Fine) per
        // channel ⇒ comfortably under 3 dB.
        for qmode in [aspx::AspxQuantStep::Fine, aspx::AspxQuantStep::Coarse] {
            let a = match qmode {
                aspx::AspxQuantStep::Fine => 2.0f32,
                aspx::AspxQuantStep::Coarse => 1.0f32,
            };
            // Pan differences stay inside the balance codebook range
            // (|q_l − q_r| ≤ a · PAN_OFFSET); beyond it the encoder
            // clamps at the codebook edge — the spec's own range limit.
            let q_l = [8, 12, 5, 20, 0, 16];
            let q_r = [8, 6, 15, 20, 4, 6];
            let (sum, pan) = balance_encode_sig_rows(&q_l, &q_r, qmode);
            assert_eq!(sum.len(), 6);
            assert_eq!(pan.len(), 6);
            // Decoder side: qscf_b = 2 · pan (delta = 2 accumulation).
            let qscf_a: Vec<Vec<i32>> = sum.iter().map(|&q| vec![q]).collect();
            let qscf_b: Vec<Vec<i32>> = pan.iter().map(|&p| vec![2 * p]).collect();
            let (scf_a, scf_b) =
                crate::aspx::dequantize_sig_scf_balance(&qscf_a, &qscf_b, qmode, 64);
            for i in 0..6 {
                let want_l = 64.0 * 2f32.powf(q_l[i] as f32 / a);
                let want_r = 64.0 * 2f32.powf(q_r[i] as f32 / a);
                let err_db_l = 10.0 * (scf_a[i][0] / want_l).log10().abs();
                let err_db_r = 10.0 * (scf_b[i][0] / want_r).log10().abs();
                assert!(
                    err_db_l < 3.0 && err_db_r < 3.0,
                    "{qmode:?} sbg {i}: L err {err_db_l} dB, R err {err_db_r} dB"
                );
            }
        }
    }

    #[test]
    fn balance_encode_sig_rows_equal_channels_are_exact() {
        // Equal L/R levels sit exactly on the (sum, pan) grid: sum
        // row == the common level, pan == neutral, and the joint
        // decode is bit-exact against Pseudocode 82.
        let q = [0, 4, 9, 31];
        let (sum, pan) = balance_encode_sig_rows(&q, &q, aspx::AspxQuantStep::Fine);
        assert_eq!(sum, q.to_vec());
        // Fine-mode neutral: qscf_b = a · PAN_OFFSET = 24 ⇒ 12 wire steps.
        assert!(pan.iter().all(|&p| p == 12));
        let (sum_c, pan_c) = balance_encode_sig_rows(&q, &q, aspx::AspxQuantStep::Coarse);
        assert_eq!(sum_c, q.to_vec());
        assert!(pan_c.iter().all(|&p| p == 6));
    }

    #[test]
    fn balance_encode_clamps_match_annex_a2_codebook_ranges() {
        // The (sum, pan) converters clamp to the Annex A.2 F0
        // codebook ranges (ETSI TS 103 190-1 electronic annex, staged:
        // docs/audio/ac4/ts_10319001v010401p0-tables.c). Tie the clamp
        // constants to the transcribed table lengths so a codebook
        // change can never silently desynchronise the encoder range.
        use crate::aspx_huffman as h;
        // Sum channel (LEVEL): F0 range = codebook_length − 1.
        assert_eq!(h::ASPX_HCB_ENV_LEVEL_15_F0_LEN.len() - 1, 70);
        assert_eq!(h::ASPX_HCB_ENV_LEVEL_30_F0_LEN.len() - 1, 35);
        assert_eq!(h::ASPX_HCB_NOISE_LEVEL_F0_LEN.len() - 1, 29);
        // Pan channel (BALANCE): wire-step range = codebook_length − 1
        // = a · PAN_OFFSET (delta = 2 ⇒ qscf_b range [0, 2·a·PAN_OFFSET]).
        assert_eq!(
            h::ASPX_HCB_ENV_BALANCE_15_F0_LEN.len() as i32 - 1,
            2 * crate::aspx::ASPX_PAN_OFFSET
        );
        assert_eq!(
            h::ASPX_HCB_ENV_BALANCE_30_F0_LEN.len() as i32 - 1,
            crate::aspx::ASPX_PAN_OFFSET
        );
        assert_eq!(
            h::ASPX_HCB_NOISE_BALANCE_F0_LEN.len() as i32 - 1,
            crate::aspx::ASPX_PAN_OFFSET
        );
        // And the converters saturate exactly at those edges.
        let (_, pan_hi) = balance_encode_sig_cell(100, 0, crate::aspx::AspxQuantStep::Fine);
        let (_, pan_lo) = balance_encode_sig_cell(0, 100, crate::aspx::AspxQuantStep::Fine);
        assert_eq!(pan_hi, 24);
        assert_eq!(pan_lo, 0);
        let (sum_hi, _) = balance_encode_sig_cell(80, 80, crate::aspx::AspxQuantStep::Fine);
        assert_eq!(sum_hi, 70);
        let (_, npan_hi) = balance_encode_noise_cell(0, 100);
        let (_, npan_lo) = balance_encode_noise_cell(100, 0);
        assert_eq!(npan_hi, 12);
        assert_eq!(npan_lo, 0);
    }

    #[test]
    fn balance_encode_noise_rows_round_trips_through_p84() {
        let q_l = [0, 3, 10, 6];
        let q_r = [0, 9, 2, 6];
        let (sum, pan) = balance_encode_noise_rows(&q_l, &q_r);
        // Equal channels: exact sum + neutral pan.
        assert_eq!(sum[0], 0);
        assert_eq!(pan[0], 6); // wire steps: qscf_b = 12 = PAN_OFFSET.
        assert_eq!(sum[3], 6);
        let qscf_a: Vec<Vec<i32>> = sum.iter().map(|&q| vec![q]).collect();
        let qscf_b: Vec<Vec<i32>> = pan.iter().map(|&p| vec![2 * p]).collect();
        let (scf_a, scf_b) = crate::aspx::dequantize_noise_scf_balance(&qscf_a, &qscf_b);
        for i in 0..4 {
            let want_l = 2f32.powi(6 - q_l[i]);
            let want_r = 2f32.powi(6 - q_r[i]);
            let err_db_l = 10.0 * (scf_a[i][0] / want_l).log10().abs();
            let err_db_r = 10.0 * (scf_b[i][0] / want_r).log10().abs();
            assert!(
                err_db_l < 4.0 && err_db_r < 4.0,
                "sbg {i}: L err {err_db_l} dB, R err {err_db_r} dB"
            );
        }
    }

    #[test]
    fn choose_acpl_direction_partition() {
        // Stationary rows → all-zero DT, strictly cheaper.
        let q = [3, 3, 4, -2];
        let env = choose_acpl_direction(&q, Some(&q));
        assert!(env.direction_time);
        assert_eq!(env.values, vec![0, 0, 0, 0]);
        // No prev → FREQ absolute row verbatim.
        let env_i = choose_acpl_direction(&q, None);
        assert!(!env_i.direction_time);
        assert_eq!(env_i.values, q.to_vec());
        // Distant prev → FREQ.
        let env_f = choose_acpl_direction(&q, Some(&[50, 50, 50, 50]));
        assert!(!env_f.direction_time);
        // Tie (all zero) → FREQ.
        let env_t = choose_acpl_direction(&[0, 0], Some(&[0, 0]));
        assert!(!env_t.direction_time);
    }

    #[test]
    fn directional_acpl_writer_all_freq_matches_legacy() {
        let num_bands = 12u32;
        let qm = crate::acpl::AcplQuantMode::Fine;
        let rows: Vec<Vec<i32>> = (0i32..11)
            .map(|e| (0..num_bands as i32).map(|b| (e + b) % 5 - 2).collect())
            .collect();
        let mut bw_a = BitWriter::new();
        write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3(
            &mut bw_a, num_bands, qm, qm, &rows[0], &rows[1], &rows[2], &rows[3], &rows[4],
            &rows[5], &rows[6], &rows[7], &rows[8], &rows[9], &rows[10],
        );
        let params: [AcplEncodedParam; 11] = std::array::from_fn(|i| AcplEncodedParam {
            values: rows[i].clone(),
            direction_time: false,
        });
        let mut bw_b = BitWriter::new();
        write_acpl_data_2ch_directional(&mut bw_b, num_bands, qm, qm, &params);
        assert_eq!(
            bw_a.into_bytes(),
            bw_b.into_bytes(),
            "all-FREQ directional ACPL writer must match the legacy writer"
        );
    }

    #[test]
    fn directional_acpl_dt_rows_chain_through_decoder_diff_state() {
        // Frame 1: FREQ rows. Frame 2: stationary → DT zero rows. The
        // decoder's Pseudocode-121 chain (differential_decode +
        // AcplDiffState) must recover the same absolute rows on both
        // frames.
        let num_bands = 12u32;
        let qm = crate::acpl::AcplQuantMode::Fine;
        let rows: Vec<Vec<i32>> = (0i32..11)
            .map(|e| (0..num_bands as i32).map(|b| (e * 2 + b) % 4 - 1).collect())
            .collect();
        // Frame 1: all FREQ.
        let params_f: [AcplEncodedParam; 11] = std::array::from_fn(|i| AcplEncodedParam {
            values: rows[i].clone(),
            direction_time: false,
        });
        let mut bw1 = BitWriter::new();
        write_acpl_data_2ch_directional(&mut bw1, num_bands, qm, qm, &params_f);
        bw1.align_to_byte();
        let f1 = bw1.into_bytes();
        // Frame 2: all TIME with zero deltas (stationary).
        let params_t: [AcplEncodedParam; 11] = std::array::from_fn(|_| AcplEncodedParam {
            values: vec![0; num_bands as usize],
            direction_time: true,
        });
        let mut bw2 = BitWriter::new();
        write_acpl_data_2ch_directional(&mut bw2, num_bands, qm, qm, &params_t);
        bw2.align_to_byte();
        let f2 = bw2.into_bytes();

        let mut br1 = BitReader::new(&f1);
        let d1 = crate::acpl::parse_acpl_data_2ch(&mut br1, num_bands, 0, qm, qm)
            .expect("frame-1 parse");
        let mut br2 = BitReader::new(&f2);
        let d2 = crate::acpl::parse_acpl_data_2ch(&mut br2, num_bands, 0, qm, qm)
            .expect("frame-2 parse");
        assert!(!d1.alpha1[0].direction_time);
        assert!(d2.alpha1[0].direction_time);

        // Chain alpha1 through the decoder's diff state.
        let mut st = crate::acpl_synth::AcplDiffState::new();
        let q1 = crate::acpl_synth::differential_decode(&d1.alpha1, num_bands, &mut st);
        let q2 = crate::acpl_synth::differential_decode(&d2.alpha1, num_bands, &mut st);
        assert_eq!(q1[0], rows[0], "frame-1 FREQ decode");
        assert_eq!(
            q2[0], rows[0],
            "frame-2 DT decode must inherit frame-1 rows"
        );
    }

    #[test]
    fn qscf_row_from_freq_dpcm_running_sum() {
        assert_eq!(qscf_row_from_freq_dpcm(&[4, -1, 2, 0]), vec![4, 3, 5, 5]);
        assert!(qscf_row_from_freq_dpcm(&[]).is_empty());
    }

    #[test]
    fn choose_envelope_direction_picks_time_when_stationary() {
        // Current == previous → all-zero DT row, strictly cheaper than
        // the non-trivial FREQ row.
        let packed = [4, -1, 2, 0];
        let prev = qscf_row_from_freq_dpcm(&packed);
        let env = choose_envelope_direction(&packed, Some(&prev));
        assert!(env.direction_time);
        assert_eq!(env.values, vec![0, 0, 0, 0]);
        // No previous rows (I-frame / first frame) → FREQ verbatim.
        let env_i = choose_envelope_direction(&packed, None);
        assert!(!env_i.direction_time);
        assert_eq!(env_i.values, packed.to_vec());
        // A distant previous row keeps FREQ (time_cost >= freq_cost).
        let far = vec![100, 100, 100, 100];
        let env_f = choose_envelope_direction(&packed, Some(&far));
        assert!(!env_f.direction_time);
        // Zero-vs-zero tie resolves to FREQ.
        let env_t = choose_envelope_direction(&[0, 0], Some(&[0, 0]));
        assert!(!env_t.direction_time);
    }

    #[test]
    fn directional_writer_all_freq_matches_tna_ah_framed_writer() {
        // Contract split by the §5.7.6.3.5 joint (sum, balance)
        // coding: the framed FREQ writer takes per-channel (L, R)
        // LEVEL rows and converts internally (Pseudocode 84 inverse),
        // while the directional writer takes the already-converted
        // wire rows verbatim. Feeding the directional writer the
        // conversion of the same (L, R) rows must therefore be
        // byte-identical to the framed writer, on both I- and P-frame
        // forms.
        let cfg = r389_test_cfg();
        let counts = aspx::derive_aspx_frequency_tables(&cfg, 0)
            .expect("tables")
            .counts;
        let sig: Vec<i32> = (0..counts.num_sbg_sig_highres as i32)
            .map(|i| i % 3 - 1)
            .collect();
        let noise: Vec<i32> = vec![1; counts.num_sbg_noise as usize];
        let tna: Vec<u8> = vec![2; counts.num_sbg_noise as usize];
        // The writers select low-res SIGNAL bands when the config
        // signals a freq_res bit; mirror that band count here.
        let num_sbg_sig = if cfg.signals_freq_res() {
            counts.num_sbg_sig_lowres
        } else {
            counts.num_sbg_sig_highres
        };
        let qmode_sig = if cfg.fixfix_tmp_num_env_bits() == 1 {
            aspx::AspxQuantStep::Fine
        } else {
            cfg.quant_mode_env
        };
        let (sum_sig, pan_sig, sum_noise, pan_noise) = balance_convert_packed_rows(
            &sig,
            &sig,
            &noise,
            &noise,
            qmode_sig,
            num_sbg_sig as usize,
            counts.num_sbg_noise as usize,
        );
        for b_iframe in [true, false] {
            let mut bw_a = BitWriter::new();
            write_aspx_data_2ch_real_envelope_tna_ah_framed(
                &mut bw_a,
                &cfg,
                AspxRealEnvelopeChannel {
                    sig: &sig,
                    noise: &noise,
                },
                AspxRealEnvelopeChannel {
                    sig: &sig,
                    noise: &noise,
                },
                &tna,
                &[],
                &[],
                b_iframe,
            )
            .unwrap();
            let mut bw_b = BitWriter::new();
            let f = |v: &[i32]| AspxEncodedEnvelope {
                values: v.to_vec(),
                direction_time: false,
            };
            write_aspx_data_2ch_directional_envelope_tna_ah_framed(
                &mut bw_b,
                &cfg,
                &f(&sum_sig),
                &f(&sum_noise),
                &f(&pan_sig),
                &f(&pan_noise),
                &tna,
                &[],
                &[],
                b_iframe,
            )
            .unwrap();
            assert_eq!(
                bw_a.into_bytes(),
                bw_b.into_bytes(),
                "all-FREQ directional writer must match (b_iframe={b_iframe})"
            );
        }
    }

    #[test]
    fn directional_writer_time_rows_round_trip_through_parser() {
        // A P-frame body with TIME-direction SIGNAL rows parses through
        // parse_aspx_data_2ch_body and the recovered deltas +
        // directions match what was written.
        let cfg = r389_test_cfg();
        let counts = aspx::derive_aspx_frequency_tables(&cfg, 0)
            .expect("tables")
            .counts;
        let n_sig = counts.num_sbg_sig_highres as usize;
        let n_noise = counts.num_sbg_noise as usize;
        let dt_sig: Vec<i32> = (0..n_sig as i32).map(|i| (i % 3) - 1).collect();
        let noise_freq: Vec<i32> = vec![2; n_noise];
        let mut bw = BitWriter::new();
        write_aspx_data_2ch_directional_envelope_tna_ah_framed(
            &mut bw,
            &cfg,
            &AspxEncodedEnvelope {
                values: dt_sig.clone(),
                direction_time: true,
            },
            &AspxEncodedEnvelope {
                values: noise_freq.clone(),
                direction_time: false,
            },
            &AspxEncodedEnvelope {
                values: vec![0; n_sig],
                direction_time: true,
            },
            &AspxEncodedEnvelope {
                values: noise_freq.clone(),
                direction_time: false,
            },
            &[],
            &[],
            &[],
            false, // P-frame form: no xover
        )
        .unwrap();
        bw.align_to_byte();
        let bytes = bw.into_bytes();
        let mut tools = crate::asf::SubstreamTools {
            aspx_xover_subband_offset: Some(0),
            ..Default::default()
        };
        let mut br = BitReader::new(&bytes);
        crate::asf::parse_aspx_data_2ch_body(&mut br, &mut tools, &cfg, false, 1920)
            .expect("P-frame parse");
        let dd = tools.aspx_delta_dir_primary.as_ref().expect("delta dir");
        assert_eq!(dd.sig_delta_dir, vec![true]);
        assert_eq!(dd.noise_delta_dir, vec![false]);
        let sig = tools.aspx_data_sig_primary.as_ref().expect("sig data");
        assert_eq!(sig.len(), 1);
        assert!(sig[0].direction_time);
        assert_eq!(sig[0].values, dt_sig);
        let dd1 = tools
            .aspx_delta_dir_secondary
            .as_ref()
            .expect("ch1 delta dir");
        assert_eq!(dd1.sig_delta_dir, vec![true]);
    }

    /// `pick_min_len_cw` returns the smallest-length entry. Verified
    /// against ASPX_HCB_ENV_LEVEL_15_F0 whose min-length entry is index
    /// 30 (length 4, codeword 0x00000).
    #[test]
    fn pick_min_len_cw_finds_smallest_length() {
        let (cw, len) = pick_min_len_cw(
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_CW,
        );
        assert_eq!(len, 4);
        assert_eq!(cw, 0x00000);
    }

    /// `pick_zero_delta_cw` returns the codeword at `index == cb_off`,
    /// which by spec invariant is the shortest entry. ASPX_HCB_ENV_LEVEL_15_DF:
    /// cb_off = 70 → length 2, codeword 0x00000.
    #[test]
    fn pick_zero_delta_cw_returns_cb_off_entry() {
        let (cw, len) = pick_zero_delta_cw(
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_LEN,
            aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_CW,
            70,
        );
        assert_eq!(len, 2);
        assert_eq!(cw, 0x00000);
    }

    /// `write_aspx_config` round-trips through `parse_aspx_config`
    /// for an arbitrary configuration — verifies the bit-order matches
    /// Table 50.
    #[test]
    fn write_aspx_config_round_trips_through_parser() {
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Coarse,
            start_freq: 5,
            stop_freq: 2,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: true,
            preflat: false,
            limiter: true,
            noise_sbg: 3,
            num_env_bits_fixfix: 1,
            freq_res_mode: aspx::AspxFreqResMode::Low,
        };
        let mut bw = BitWriter::new();
        write_aspx_config(&mut bw, &cfg);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = aspx::parse_aspx_config(&mut br).unwrap();
        assert_eq!(parsed, cfg);
    }

    /// `write_acpl_config_2ch` round-trips through `parse_acpl_config_2ch`.
    #[test]
    fn write_acpl_config_2ch_round_trips_through_parser() {
        let mut bw = BitWriter::new();
        // num_param_bands_id = 3 → 7 param bands; qm0 = Fine, qm1 = Coarse.
        write_acpl_config_2ch(
            &mut bw,
            3,
            crate::acpl::AcplQuantMode::Fine,
            crate::acpl::AcplQuantMode::Coarse,
        );
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = crate::acpl::parse_acpl_config_2ch(&mut br).unwrap();
        assert_eq!(parsed.num_param_bands_id, 3);
        assert_eq!(parsed.num_param_bands, 7);
        assert!(matches!(
            parsed.quant_mode_0,
            crate::acpl::AcplQuantMode::Fine
        ));
        assert!(matches!(
            parsed.quant_mode_1,
            crate::acpl::AcplQuantMode::Coarse
        ));
    }

    /// `write_companding_control_2ch_sync_on` emits exactly two bits and
    /// round-trips through `parse_companding_control(2)`.
    #[test]
    fn companding_control_2ch_sync_on_round_trips() {
        let mut bw = BitWriter::new();
        write_companding_control_2ch_sync_on(&mut bw);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = aspx::parse_companding_control(&mut br, 2).unwrap();
        assert_eq!(cc.sync_flag, Some(true));
        assert_eq!(cc.compand_on, vec![true]);
        assert!(cc.compand_avg.is_none());
    }

    /// `write_acpl_data_2ch_minimal` produces a body that round-trips
    /// through `parse_acpl_data_2ch` with all-zero recovered values.
    #[test]
    fn acpl_data_2ch_minimal_round_trips() {
        let num_bands: u32 = 7; // num_param_bands_id = 3
        let qm0 = crate::acpl::AcplQuantMode::Fine;
        let qm1 = crate::acpl::AcplQuantMode::Fine;
        let mut bw = BitWriter::new();
        write_acpl_data_2ch_minimal(&mut bw, num_bands, qm0, qm1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = crate::acpl::parse_acpl_data_2ch(&mut br, num_bands, 0, qm0, qm1).unwrap();
        assert_eq!(parsed.framing.num_param_sets, 1);
        assert_eq!(parsed.alpha1.len(), 1);
        assert_eq!(parsed.alpha1[0].values.len(), num_bands as usize);
        // First value is F0; remaining are DF (zero deltas).
        for v in &parsed.alpha1[0].values[1..] {
            assert_eq!(*v, 0);
        }
        for v in &parsed.gamma1[0].values[1..] {
            assert_eq!(*v, 0);
        }
    }

    /// `write_aspx_data_2ch_minimal` produces a body that round-trips
    /// through `parse_aspx_data_2ch_body` without erroring out. Uses
    /// a small `AspxConfig` so the per-channel SBG counts are small.
    #[test]
    fn aspx_data_2ch_minimal_round_trips_through_parser() {
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0, // num_noise_sbgroups = 1
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let mut bw = BitWriter::new();
        write_aspx_data_2ch_minimal(&mut bw, &cfg).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let _ = bytes;
        // Note: parse_aspx_data_2ch_body is pub(crate) and takes
        // SubstreamTools; full round-trip is exercised via the integration
        // test in tests/round95_5_x_acpl3_encoder.rs.
    }

    // ----------------------------------------------------------------
    // Round 100 — ASPX_ACPL_2 emitter tests
    // ----------------------------------------------------------------

    /// `write_acpl_config_1ch_full` round-trips through
    /// `parse_acpl_config_1ch(FULL)` and emits exactly 3 bits (2-bit id +
    /// 1-bit quant_mode, no qmf_band).
    #[test]
    fn write_acpl_config_1ch_full_round_trips_through_parser() {
        let mut bw = BitWriter::new();
        write_acpl_config_1ch_full(&mut bw, 3, crate::acpl::AcplQuantMode::Fine);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed =
            crate::acpl::parse_acpl_config_1ch(&mut br, crate::acpl::Acpl1chMode::Full).unwrap();
        assert_eq!(parsed.num_param_bands_id, 3);
        assert_eq!(parsed.num_param_bands, 7);
        assert!(matches!(
            parsed.quant_mode,
            crate::acpl::AcplQuantMode::Fine
        ));
        // FULL mode → qmf_band is 0 (no acpl_qmf_band_minus1 read).
        assert_eq!(parsed.qmf_band, 0);
    }

    /// `write_two_channel_data` produces a body that round-trips through
    /// `parse_two_channel_data` for the long-frame identity-SAP case.
    #[test]
    fn write_two_channel_data_round_trips_through_parser() {
        let tl = 1920u32;
        let max_sfb = 8u32;
        let coeffs_l = vec![0.0f32; tl as usize / 2];
        let coeffs_r = vec![0.0f32; tl as usize / 2];
        let mut bw = BitWriter::new();
        write_two_channel_data(&mut bw, tl, max_sfb, &coeffs_l, &coeffs_r);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d = crate::mch::parse_two_channel_data(&mut br, tl).unwrap();
        assert_eq!(d.transform_info.as_ref().unwrap().transform_length_0, tl);
        assert_eq!(d.psy_info.as_ref().unwrap().max_sfb_0, max_sfb);
        assert_eq!(d.chparam.as_ref().unwrap().sap_mode, 0);
        assert_eq!(d.scaled_spec_per_channel.len(), 2);
        assert!(d.scaled_spec_per_channel.iter().all(|c| c.is_some()));
    }

    /// `write_mono_data_centre` produces a non-LFE `mono_data(0)` body
    /// that round-trips through `parse_mono_data(b_lfe = false)`.
    #[test]
    fn write_mono_data_centre_round_trips_through_parser() {
        let tl = 1920u32;
        let max_sfb = 6u32;
        let coeffs = vec![0.0f32; tl as usize / 2];
        let mut bw = BitWriter::new();
        write_mono_data_centre(&mut bw, tl, max_sfb, &coeffs);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let m = crate::mch::parse_mono_data(&mut br, false, tl).unwrap();
        assert!(!m.b_lfe);
        assert_eq!(m.spec_frontend_bit, 0);
        assert_eq!(m.psy_info.as_ref().unwrap().max_sfb_0, max_sfb);
        assert!(m.scaled_spec.is_some());
    }

    /// `write_acpl_data_1ch_minimal` produces a body that round-trips
    /// through `parse_acpl_data_1ch` with all-zero recovered deltas.
    #[test]
    fn acpl_data_1ch_minimal_round_trips() {
        let num_bands: u32 = 7; // num_param_bands_id = 3
        let qm = crate::acpl::AcplQuantMode::Fine;
        let mut bw = BitWriter::new();
        write_acpl_data_1ch_minimal(&mut bw, num_bands, 0, qm);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = crate::acpl::parse_acpl_data_1ch(&mut br, num_bands, 0, qm).unwrap();
        assert_eq!(parsed.framing.num_param_sets, 1);
        assert_eq!(parsed.alpha1.len(), 1);
        assert_eq!(parsed.alpha1[0].values.len(), num_bands as usize);
        assert_eq!(parsed.beta1[0].values.len(), num_bands as usize);
        // F0 + DF zero-delta → all subsequent values are 0.
        for v in &parsed.alpha1[0].values[1..] {
            assert_eq!(*v, 0);
        }
        for v in &parsed.beta1[0].values[1..] {
            assert_eq!(*v, 0);
        }
    }

    /// `write_aspx_data_1ch_minimal` emits without erroring for the small
    /// `AspxConfig` used by the ASPX_ACPL_2 encoder path. Full round-trip
    /// is exercised via the integration test
    /// `tests/round100_5_x_acpl2_encoder.rs`.
    #[test]
    fn aspx_data_1ch_minimal_emits_without_error() {
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let mut bw = BitWriter::new();
        write_aspx_data_1ch_minimal(&mut bw, &cfg).unwrap();
        bw.align_to_byte();
        assert!(!bw.finish().is_empty());
    }

    // ----------------------------------------------------------------
    // Round 103 — ASPX_ACPL_1 emitter tests
    // ----------------------------------------------------------------

    /// `write_acpl_config_1ch_partial` round-trips through
    /// `parse_acpl_config_1ch(PARTIAL)` and emits exactly 6 bits (2-bit
    /// id, 1-bit quant_mode, 3-bit acpl_qmf_band_minus1). The recovered
    /// `qmf_band` equals `acpl_qmf_band_minus1 + 1`.
    #[test]
    fn write_acpl_config_1ch_partial_round_trips_through_parser() {
        let mut bw = BitWriter::new();
        write_acpl_config_1ch_partial(&mut bw, 3, crate::acpl::AcplQuantMode::Fine, 2);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed =
            crate::acpl::parse_acpl_config_1ch(&mut br, crate::acpl::Acpl1chMode::Partial).unwrap();
        assert_eq!(parsed.num_param_bands_id, 3);
        assert_eq!(parsed.num_param_bands, 7);
        assert!(matches!(
            parsed.quant_mode,
            crate::acpl::AcplQuantMode::Fine
        ));
        // PARTIAL mode → qmf_band = qmf_band_minus1 + 1 = 3.
        assert_eq!(parsed.qmf_band, 3);
    }

    /// `write_acpl_1_residual_layer` produces a body whose
    /// `max_sfb_master` + 2× chparam_info + 2× sf_data(ASF) round-trip
    /// through the decoder's residual-layer walk. We exercise the parse
    /// directly via `parse_chparam_info` + `decode_asf_long_mono_body_*`
    /// mirroring `parse_aspx_acpl_1_2_inner_body`; here we just confirm the
    /// writer returns the clamped band budget and emits a non-empty body.
    #[test]
    fn write_acpl_1_residual_layer_clamps_and_emits() {
        let tl = 1920u32;
        let coeffs_ls = vec![0.0f32; tl as usize / 2];
        let coeffs_rs = vec![0.0f32; tl as usize / 2];
        let mut bw = BitWriter::new();
        // Request a band budget above the n_side cap (31 @ tl=1920) →
        // clamped to 31.
        let used = write_acpl_1_residual_layer(&mut bw, tl, 40, &coeffs_ls, &coeffs_rs);
        bw.align_to_byte();
        assert_eq!(used, 31, "max_sfb_master clamped to n_side cap (5 b → 31)");
        assert!(!bw.finish().is_empty());

        // A zero request clamps up to 1 (the decoder bails on 0).
        let mut bw2 = BitWriter::new();
        let used2 = write_acpl_1_residual_layer(&mut bw2, tl, 0, &coeffs_ls, &coeffs_rs);
        assert_eq!(
            used2, 1,
            "max_sfb_master clamped up to 1 (decoder bails on 0)"
        );
    }

    // ----------------------------------------------------------------
    // Round 257 — SAP-aware ASPX_ACPL_1 residual-layer emitter tests
    // ----------------------------------------------------------------

    /// `write_acpl_1_residual_layer_sap` with `chparam_pair = None` is
    /// bit-equivalent to the legacy `write_acpl_1_residual_layer` —
    /// the SAP-aware path defaults to two identity rows, and the
    /// Table-181 inverse for the identity row reduces to
    /// `s3 = ls, s4 = rs` (per
    /// `asf::invert_sap_table_181_identity_passthrough`).
    #[test]
    fn write_acpl_1_residual_layer_sap_none_matches_legacy() {
        let tl = 1920u32;
        let n = tl as usize;
        let coeffs_ls: Vec<f32> = (0..n).map(|i| 0.10 + 1e-4 * i as f32).collect();
        let coeffs_rs: Vec<f32> = (0..n).map(|i| -0.05 + 2e-4 * i as f32).collect();
        let coeffs_l = vec![0.0f32; n];
        let coeffs_r = vec![0.0f32; n];

        let mut bw_legacy = BitWriter::new();
        let used_legacy =
            write_acpl_1_residual_layer(&mut bw_legacy, tl, 8, &coeffs_ls, &coeffs_rs);
        bw_legacy.align_to_byte();
        let bytes_legacy = bw_legacy.finish();

        let mut bw_sap = BitWriter::new();
        let used_sap = write_acpl_1_residual_layer_sap(
            &mut bw_sap,
            tl,
            8,
            &coeffs_l,
            &coeffs_r,
            &coeffs_ls,
            &coeffs_rs,
            None,
        );
        bw_sap.align_to_byte();
        let bytes_sap = bw_sap.finish();

        assert_eq!(used_legacy, used_sap);
        assert_eq!(bytes_legacy, bytes_sap,
            "SAP-aware writer with chparam_pair = None must be bit-equal to the legacy identity-SAP writer");
    }

    /// SAP-aware residual layer with a non-identity (M/S) chparam pair
    /// emits chparam_info `sap_mode = 1` payload that the decoder's
    /// `parse_chparam_info` recovers exactly, and produces residual
    /// sf_data bodies whose subsequent `apply_sap_table_181` re-mix
    /// reconstructs the requested `(L, R, Ls, Rs)` preliminary set
    /// inside the SAP-coded extent.
    #[test]
    fn write_acpl_1_residual_layer_sap_ms_row_roundtrips_through_decoder() {
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 2u32;
        // Choose tight L/R/Ls/Rs preliminary spectra. The M/S inverse
        // produces `(A, s3) = ((L + Ls)/2, (L - Ls)/2)`; feeding those
        // back through the forward path with the same M/S row
        // reproduces (L, Ls).
        let l_spec = vec![4.0f32; n];
        let r_spec = vec![6.0f32; n];
        let ls_spec = vec![-2.0f32; n];
        let rs_spec = vec![-2.0f32; n];

        let cp_ms = crate::asf::ChparamInfo {
            sap_mode: 1,
            ms_used: vec![vec![true, true]],
            sap_data: None,
        };
        let pair = [cp_ms.clone(), cp_ms];

        let mut bw = BitWriter::new();
        let used = write_acpl_1_residual_layer_sap(
            &mut bw,
            tl,
            max_sfb_master,
            &l_spec,
            &r_spec,
            &ls_spec,
            &rs_spec,
            Some(&pair),
        );
        bw.align_to_byte();
        let bytes = bw.finish();
        assert_eq!(used, max_sfb_master);

        // Decode the body the same way `parse_aspx_acpl_1_2_inner_body`
        // does on the wire: max_sfb_master (n_side bits) + two
        // chparam_info()s with `max_sfb_per_group = [max_sfb_master]`.
        let mut br = BitReader::new(&bytes);
        let (_, n_side, _) = crate::tables::n_msfb_bits_48(tl).unwrap();
        let parsed_max_sfb_master = br.read_u32(n_side).unwrap();
        assert_eq!(parsed_max_sfb_master, max_sfb_master);

        let cp0 = crate::asf::parse_chparam_info(&mut br, &[max_sfb_master]).unwrap();
        let cp1 = crate::asf::parse_chparam_info(&mut br, &[max_sfb_master]).unwrap();
        assert_eq!(cp0.sap_mode, 1);
        assert_eq!(cp1.sap_mode, 1);
        assert_eq!(cp0.ms_used, vec![vec![true, true]]);
        assert_eq!(cp1.ms_used, vec![vec![true, true]]);
    }

    /// SAP-aware residual layer with an explicit identity chparam pair
    /// is bit-equivalent to passing `None`. Pins the
    /// "identity-pair-explicit == identity-pair-default" contract.
    #[test]
    fn write_acpl_1_residual_layer_sap_identity_explicit_matches_default() {
        let tl = 1920u32;
        let n = tl as usize;
        let coeffs_ls = vec![0.25f32; n];
        let coeffs_rs = vec![-0.25f32; n];
        let coeffs_l = vec![0.0f32; n];
        let coeffs_r = vec![0.0f32; n];

        let cp_id = crate::asf::ChparamInfo {
            sap_mode: 0,
            ms_used: vec![],
            sap_data: None,
        };
        let pair = [cp_id.clone(), cp_id];

        let mut bw_default = BitWriter::new();
        write_acpl_1_residual_layer_sap(
            &mut bw_default,
            tl,
            8,
            &coeffs_l,
            &coeffs_r,
            &coeffs_ls,
            &coeffs_rs,
            None,
        );
        bw_default.align_to_byte();

        let mut bw_explicit = BitWriter::new();
        write_acpl_1_residual_layer_sap(
            &mut bw_explicit,
            tl,
            8,
            &coeffs_l,
            &coeffs_r,
            &coeffs_ls,
            &coeffs_rs,
            Some(&pair),
        );
        bw_explicit.align_to_byte();

        assert_eq!(bw_default.finish(), bw_explicit.finish());
    }

    /// `build_5_x_acpl1_body_from_pcm_spectra_sap` with `chparam_pair =
    /// None` produces a body bit-equivalent to the legacy
    /// `build_5_x_acpl1_body_from_pcm_spectra` for the same inputs —
    /// the identity-SAP path is byte-for-byte unchanged.
    #[test]
    fn build_5_x_acpl1_body_sap_none_matches_legacy() {
        let tl = 1920u32;
        let half = tl as usize / 2;
        let zeros = vec![0.0f32; half];
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let body_legacy = build_5_x_acpl1_body_from_pcm_spectra(
            tl,
            16,
            8,
            true,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            4096,
        );
        let body_sap = build_5_x_acpl1_body_from_pcm_spectra_sap(
            tl,
            16,
            8,
            true,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            None,
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            4096,
        );
        assert_eq!(body_legacy, body_sap);
    }

    /// The SAP-aware body builder with an explicit non-identity (M/S)
    /// chparam pair produces a body the decoder's
    /// `parse_5x_audio_data_outer` walks to `FiveXCodecMode::AspxAcpl1`,
    /// recovers the chparam pair into `tools.acpl_1_residual_chparam`
    /// with `sap_mode = 1` on both rows, and persists the residual
    /// `(sSMP,3, sSMP,4)` spectra at the requested `max_sfb_master`.
    #[test]
    fn build_5_x_acpl1_body_sap_ms_decoder_recovers_chparam() {
        let tl = 1920u32;
        let half = tl as usize / 2;
        let zeros = vec![0.0f32; half];
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let max_sfb_master = 8u32;
        // Need `max_sfb_master` flags per row to satisfy the parser at
        // sap_mode = 1; build a per-band MsUsed pattern.
        let ms_row = (0..max_sfb_master).map(|i| i % 2 == 0).collect::<Vec<_>>();
        let cp_ms = crate::asf::ChparamInfo {
            sap_mode: 1,
            ms_used: vec![ms_row.clone()],
            sap_data: None,
        };
        let pair = [cp_ms.clone(), cp_ms];

        let body = build_5_x_acpl1_body_from_pcm_spectra_sap(
            tl,
            16,
            max_sfb_master,
            true,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            Some(&pair),
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            4096,
        );
        let mut br = BitReader::new(&body[2..]);
        let mut tools = crate::asf::SubstreamTools::default();
        crate::mch::parse_5x_audio_data_outer(&mut br, &mut tools, false, true, tl).unwrap();
        assert_eq!(
            tools.five_x_mode,
            Some(crate::mch::FiveXCodecMode::AspxAcpl1)
        );
        assert_eq!(tools.acpl_1_residual_max_sfb_master, Some(max_sfb_master));
        let cp0 = tools.acpl_1_residual_chparam[0]
            .as_ref()
            .expect("residual chparam[0] parsed");
        let cp1 = tools.acpl_1_residual_chparam[1]
            .as_ref()
            .expect("residual chparam[1] parsed");
        assert_eq!(cp0.sap_mode, 1);
        assert_eq!(cp1.sap_mode, 1);
        assert_eq!(cp0.ms_used, vec![ms_row.clone()]);
        assert_eq!(cp1.ms_used, vec![ms_row]);
        assert!(tools.acpl_1_residual_pair[0].is_some());
        assert!(tools.acpl_1_residual_pair[1].is_some());
    }

    // ------------------------------------------------------------------
    // Round 279 — decision-driven SAP residual selector + auto builder
    // ------------------------------------------------------------------

    /// Build a synthetic spectrum that is `amp` over the bins of sfbs
    /// `[0, n_sfb)` at tl = 1920 and zero elsewhere.
    fn const_spectrum_1920(amp: f32, n_sfb: usize) -> Vec<f32> {
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let hi = sfbo[n_sfb] as usize;
        let mut v = vec![0.0f32; 1920];
        for s in v.iter_mut().take(hi) {
            *s = amp;
        }
        v
    }

    /// `Ls = κ·L` (κ = 0.2) over the first 4 sfbs: the least-squares
    /// projection is `g* = (1 − κ) / (1 + κ) = 2/3` → `alpha_q = 7` →
    /// the SAP-coded row extracts to `(1.7, 1, 0.3, −1)` per band.
    /// The silent (R, Rs) pair falls back to the `SapMode::None` row.
    #[test]
    fn select_acpl1_residual_chparam_correlated_pair_picks_sap_row() {
        let l = const_spectrum_1920(1.0, 4);
        let ls = const_spectrum_1920(0.2, 4);
        let zeros = vec![0.0f32; 1920];
        let max_sfb_master = 4u32;
        let pair =
            select_acpl1_residual_chparam_pair(&l, &zeros, &ls, &zeros, max_sfb_master, 1920);
        assert_eq!(pair[0].sap_mode, 3, "correlated pair must be SAP-coded");
        assert_eq!(pair[1].sap_mode, 0, "silent pair must fall back to None");
        let coeffs = crate::asf::extract_sap_abcd(&pair[0], &[max_sfb_master]);
        for sfb in 0..max_sfb_master as usize {
            let (a, b, c, d) = coeffs.abcd[0][sfb];
            assert!(
                (a - 1.7).abs() < 1e-6 && (b - 1.0).abs() < 1e-6,
                "sfb {sfb}: expected a = 1.7, b = 1, got ({a}, {b})"
            );
            assert!(
                (c - 0.3).abs() < 1e-6 && (d + 1.0).abs() < 1e-6,
                "sfb {sfb}: expected c = 0.3, d = -1, got ({c}, {d})"
            );
        }
    }

    /// `Ls = L` exactly ⇒ zero side energy ⇒ `g* = 0` ⇒ no band raises
    /// `sap_coeff_used` and both rows fall back to `SapMode::None`.
    #[test]
    fn select_acpl1_residual_chparam_equal_pair_falls_back_to_none() {
        let l = const_spectrum_1920(0.7, 4);
        let r = const_spectrum_1920(0.4, 4);
        let pair = select_acpl1_residual_chparam_pair(&l, &r, &l, &r, 4, 1920);
        assert_eq!(pair[0].sap_mode, 0);
        assert_eq!(pair[1].sap_mode, 0);
        assert!(pair[0].sap_data.is_none());
        assert!(pair[1].sap_data.is_none());
    }

    /// A near-anti-correlated pair (`Ls = −0.9·L`) drives `g*` far past
    /// the codable range; the selector clamps `alpha_q` to ±30 (g = 3.0
    /// → a = 4.0) so pair-major DPCM deltas stay HCB_SCALEFAC-codable.
    #[test]
    fn select_acpl1_residual_chparam_clamps_alpha_q_to_30() {
        let l = const_spectrum_1920(1.0, 4);
        let ls: Vec<f32> = l.iter().map(|v| v * -0.9).collect();
        let zeros = vec![0.0f32; 1920];
        let pair = select_acpl1_residual_chparam_pair(&l, &zeros, &ls, &zeros, 4, 1920);
        assert_eq!(pair[0].sap_mode, 3);
        let coeffs = crate::asf::extract_sap_abcd(&pair[0], &[4]);
        let (a, _b, c, _d) = coeffs.abcd[0][0];
        assert!(
            (a - 4.0).abs() < 1e-6 && (c + 2.0).abs() < 1e-6,
            "alpha_q must clamp to +30 (g = 3.0): got a = {a}, c = {c}"
        );
    }

    /// Strict-superset invariant: when the selector picks no SAP band
    /// (`Ls = L`, `Rs = R`) the auto builder's output is bit-for-bit
    /// identical to the round-103 identity builder on the same input.
    #[test]
    fn build_5_x_acpl1_body_sap_auto_identity_matches_legacy() {
        let tl = 1920u32;
        let l = const_spectrum_1920(0.6, 6);
        let r = const_spectrum_1920(0.3, 6);
        let c = const_spectrum_1920(0.2, 6);
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let legacy = build_5_x_acpl1_body_from_pcm_spectra(
            tl,
            16,
            8,
            true,
            &l,
            &r,
            &c,
            &l,
            &r,
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            4096,
        );
        let auto = build_5_x_acpl1_body_from_pcm_spectra_sap_auto(
            tl,
            16,
            8,
            true,
            &l,
            &r,
            &c,
            &l,
            &r,
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            4096,
        );
        assert_eq!(
            legacy, auto,
            "no-SAP-band input must produce a bit-identical body"
        );
    }

    /// End-to-end bit-stream round trip of the auto builder on a
    /// correlated surround pair: the decoder walks the body, recovers
    /// the SAP-coded chparam rows, and `apply_sap_table_181` on the
    /// parsed carrier + residual spectra reproduces the requested
    /// `(L, Ls)` / `(R, Rs)` preliminaries (up to sf_data quantisation).
    /// The transmitted residual energy collapses versus the raw
    /// surround energy — the measurable SAP win.
    #[test]
    fn build_5_x_acpl1_body_sap_auto_round_trips_and_shrinks_residual() {
        let tl = 1920u32;
        let max_sfb_master = 8u32;
        let l = const_spectrum_1920(1.0, 4);
        let ls = const_spectrum_1920(0.5, 4); // κ = 0.5 → g* = 1/3 → alpha_q = 3
        let r = const_spectrum_1920(0.8, 4);
        let rs = const_spectrum_1920(0.2, 4); // κ = 0.25 → g* = 0.6 → alpha_q = 6
        let c = vec![0.0f32; 1920];
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let body = build_5_x_acpl1_body_from_pcm_spectra_sap_auto(
            tl,
            16,
            max_sfb_master,
            true,
            &l,
            &r,
            &c,
            &ls,
            &rs,
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            8192,
        );
        let mut br = BitReader::new(&body[2..]);
        let mut tools = crate::asf::SubstreamTools::default();
        crate::mch::parse_5x_audio_data_outer(&mut br, &mut tools, false, true, tl).unwrap();
        assert_eq!(
            tools.five_x_mode,
            Some(crate::mch::FiveXCodecMode::AspxAcpl1)
        );
        assert_eq!(tools.acpl_1_residual_max_sfb_master, Some(max_sfb_master));
        let cp0 = tools.acpl_1_residual_chparam[0]
            .as_ref()
            .expect("residual chparam[0] parsed");
        let cp1 = tools.acpl_1_residual_chparam[1]
            .as_ref()
            .expect("residual chparam[1] parsed");
        assert_eq!(cp0.sap_mode, 3, "pair 0 must be SAP-coded");
        assert_eq!(cp1.sap_mode, 3, "pair 1 must be SAP-coded");

        // Transmitted residual energy collapses vs the raw surround
        // energy (the identity path would code the full Ls spectrum).
        let (_tl3, s3) = tools.acpl_1_residual_pair[0]
            .as_ref()
            .expect("residual pair[0] parsed");
        let e_s3: f64 = s3.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let e_ls: f64 = ls.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        assert!(
            e_s3 < 0.05 * e_ls,
            "SAP residual energy must collapse: e_s3 = {e_s3}, e_ls = {e_ls}"
        );

        // Forward Table-181 mix on the parsed spectra reproduces the
        // requested preliminaries.
        let tcd = tools
            .two_channel_data
            .first()
            .expect("two_channel_data parsed");
        let a_spec = tcd.scaled_spec_per_channel[0]
            .as_ref()
            .expect("carrier A spectrum");
        let b_spec = tcd.scaled_spec_per_channel[1]
            .as_ref()
            .expect("carrier B spectrum");
        let (_tl4, s4) = tools.acpl_1_residual_pair[1]
            .as_ref()
            .expect("residual pair[1] parsed");
        let pad = |src: &[f32]| -> Vec<f32> {
            let mut v = vec![0.0f32; tl as usize];
            let take = src.len().min(tl as usize);
            v[..take].copy_from_slice(&src[..take]);
            v
        };
        let (l_out, r_out, ls_out, rs_out) = crate::asf::apply_sap_table_181(
            &pad(a_spec),
            &pad(b_spec),
            &pad(s3),
            &pad(s4),
            &[cp0.clone(), cp1.clone()],
            max_sfb_master,
            tl,
        )
        .expect("forward SAP mix");
        let rel_err = |got: &[f32], want: &[f32]| -> f64 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (g, w) in got.iter().zip(want.iter()) {
                num += ((*g - *w) as f64).powi(2);
                den += (*w as f64).powi(2);
            }
            if den == 0.0 {
                num.sqrt()
            } else {
                (num / den).sqrt()
            }
        };
        assert!(rel_err(&l_out, &l) < 0.2, "L: {}", rel_err(&l_out, &l));
        assert!(rel_err(&r_out, &r) < 0.2, "R: {}", rel_err(&r_out, &r));
        assert!(rel_err(&ls_out, &ls) < 0.2, "Ls: {}", rel_err(&ls_out, &ls));
        assert!(rel_err(&rs_out, &rs) < 0.2, "Rs: {}", rel_err(&rs_out, &rs));
    }

    /// The full ASPX_ACPL_1 body builder produces output the decoder walks
    /// to `FiveXCodecMode::AspxAcpl1` with the PARTIAL config, the residual
    /// pair, the centre mono, and both acpl_data_1ch parameter sets.
    #[test]
    fn build_5_x_acpl1_body_decoder_resolves_full_body() {
        let tl = 1920u32;
        let half = tl as usize / 2;
        let zeros = vec![0.0f32; half];
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let body = build_5_x_acpl1_body_from_pcm_spectra(
            tl,
            16,
            8,
            true,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            0,
            4096,
        );
        // Skip the 2-byte ac4_substream() audio_size header the builder
        // prepends (15 b size + 1 b more_bits, byte-aligned).
        let mut br = BitReader::new(&body[2..]);
        let mut tools = crate::asf::SubstreamTools::default();
        crate::mch::parse_5x_audio_data_outer(&mut br, &mut tools, false, true, tl).unwrap();
        assert_eq!(
            tools.five_x_mode,
            Some(crate::mch::FiveXCodecMode::AspxAcpl1)
        );
        let cfg_partial = tools
            .acpl_config_1ch_partial
            .expect("PARTIAL config parsed");
        assert_eq!(cfg_partial.qmf_band, 1); // qmf_band_minus1 = 0 → 1
        assert_eq!(tools.two_channel_data.len(), 1);
        assert!(tools.cfg0_centre_mono.is_some());
        assert_eq!(tools.acpl_1_residual_max_sfb_master, Some(8));
        assert!(tools.acpl_1_residual_pair[0].is_some());
        assert!(tools.acpl_1_residual_pair[1].is_some());
        assert!(tools.acpl_data_1ch_pair[0].is_some());
        assert!(tools.acpl_data_1ch_pair[1].is_some());
    }

    /// The 7_X ASPX_ACPL_2 body builder produces output the decoder's
    /// `parse_7x_audio_data_outer` walks to `SevenXCodecMode::AspxAcpl2`
    /// with the FULL acpl config, both stereo pairs (L/R + Ls/Rs), the
    /// trailing Cfg0 centre mono, and both `acpl_data_1ch` parameter sets.
    #[test]
    fn build_7_x_acpl2_body_decoder_resolves_full_body() {
        let tl = 1920u32;
        let half = tl as usize / 2;
        let zeros = vec![0.0f32; half];
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let body = build_7_x_acpl2_body_from_pcm_spectra(
            tl,
            16,
            None, // 7.0 — no LFE
            true,
            &zeros, // L
            &zeros, // R
            &zeros, // Ls
            &zeros, // Rs
            &zeros, // C
            None,   // 7.0 — no LFE
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            4096,
        );
        // Skip the 2-byte ac4_substream() audio_size header.
        let mut br = BitReader::new(&body[2..]);
        let mut tools = crate::asf::SubstreamTools::default();
        crate::mch::parse_7x_audio_data_outer(&mut br, &mut tools, false, true, tl).unwrap();
        assert_eq!(
            tools.seven_x_mode,
            Some(crate::mch::SevenXCodecMode::AspxAcpl2)
        );
        assert!(tools.acpl_config_1ch_full.is_some());
        // Cfg0 in the 7_X walker carries two two_channel_data pairs.
        assert_eq!(tools.two_channel_data.len(), 2);
        assert!(tools.cfg0_centre_mono.is_some());
        assert!(tools.acpl_data_1ch_pair[0].is_some());
        assert!(tools.acpl_data_1ch_pair[1].is_some());
        // ASPX_ACPL_2 has no joint-MDCT residual layer.
        assert!(tools.acpl_1_residual_pair[0].is_none());
        assert!(tools.acpl_1_residual_pair[1].is_none());
        // No LFE for the 7.0 path.
        assert!(tools.lfe_mono_data.is_none());
    }

    /// The 7.1 (3/4/0.1) ASPX_ACPL_2 body builder — with
    /// `coeffs_lfe`/`max_sfb_lfe` set — emits a leading `mono_data(1)`
    /// element the decoder's `parse_7x_audio_data_outer(b_has_lfe = true)`
    /// resolves into `tools.lfe_mono_data`, in addition to the full
    /// round-107 7.0 body (both stereo pairs, centre mono, ACPL pair).
    #[test]
    fn build_7_x_acpl2_body_with_lfe_decoder_resolves_lfe() {
        let tl = 1920u32;
        let half = tl as usize / 2;
        let zeros = vec![0.0f32; half];
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let body = build_7_x_acpl2_body_from_pcm_spectra(
            tl,
            16,
            Some(7), // LFE max_sfb (n_msfbl_bits = 3 cap at tl = 1920)
            true,
            &zeros,       // L
            &zeros,       // R
            &zeros,       // Ls
            &zeros,       // Rs
            &zeros,       // C
            Some(&zeros), // LFE coeffs
            &cfg,
            3,
            crate::acpl::AcplQuantMode::Fine,
            4096,
        );
        let mut br = BitReader::new(&body[2..]);
        let mut tools = crate::asf::SubstreamTools::default();
        // b_has_lfe = true mirrors the channels == 8 dispatch path.
        crate::mch::parse_7x_audio_data_outer(&mut br, &mut tools, true, true, tl).unwrap();
        assert_eq!(
            tools.seven_x_mode,
            Some(crate::mch::SevenXCodecMode::AspxAcpl2)
        );
        assert!(tools.seven_x_b_has_lfe);
        // LFE element resolved.
        assert!(tools.lfe_mono_data.is_some());
        // ...followed by the full 7.0 ACPL_2 body.
        assert!(tools.acpl_config_1ch_full.is_some());
        assert_eq!(tools.two_channel_data.len(), 2);
        assert!(tools.cfg0_centre_mono.is_some());
        assert!(tools.acpl_data_1ch_pair[0].is_some());
        assert!(tools.acpl_data_1ch_pair[1].is_some());
        assert!(tools.acpl_1_residual_pair[0].is_none());
    }

    /// Direct unit test of `analytic_beta_per_band` + `quantise_beta_magnitude`:
    /// when carrier energy is non-zero and the surround energy exceeds
    /// `0.5·E[carrier²]·(1-α)²`, the analytic β must be positive and
    /// the quantised index non-zero.
    #[test]
    fn analytic_beta_positive_when_surround_energy_exceeds_alpha_model() {
        let qm = crate::acpl::AcplQuantMode::Fine;
        // 7 param bands. Bands 0..3 are "noise"; band 4 carries data.
        let e_c = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let e_s = vec![0.0f32, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0]; // 2·E_s/E_c = 5.0
        let alpha_dq = vec![0.0f32; 7]; // (1 − α)² = 1
        let beta = analytic_beta_per_band(&e_c, &e_s, &alpha_dq, qm);
        // β² = 5.0 − 1.0 = 4.0 → β = 2.0.
        assert!(
            (beta[4] - 2.0).abs() < 1e-5,
            "expected β=2.0 at band 4, got {}",
            beta[4]
        );
        let q = quantise_beta_magnitude(beta[4], qm);
        // BETA_DQ_FINE column-0 lane closest to 2.0 is row 5 (1.9375)
        // or row 6 (2.55) — 5 is closer.
        assert_eq!(q, 5, "expected beta_q lane 5 (=1.9375), got {q}");
    }

    /// When the surround energy exactly matches `0.5·E[c²]·(1-α)²`,
    /// the residual is zero → β = 0 → quantised index is 0.
    #[test]
    fn analytic_beta_zero_when_alpha_fully_explains_surround() {
        let qm = crate::acpl::AcplQuantMode::Fine;
        let e_c = vec![1.0f32; 1];
        // For α = 0.5: (1-α)² = 0.25, so E_s = 0.5·1·0.25 = 0.125
        // gives exact match → β = 0.
        let alpha_dq = vec![0.5f32; 1];
        let e_s = vec![0.125f32; 1];
        let beta = analytic_beta_per_band(&e_c, &e_s, &alpha_dq, qm);
        assert!((beta[0]).abs() < 1e-5, "expected β=0, got {}", beta[0]);
        assert_eq!(quantise_beta_magnitude(beta[0], qm), 0);
    }

    /// Zero carrier energy short-circuits to β = 0 (we can't extract
    /// β where the carrier doesn't exist).
    #[test]
    fn analytic_beta_zero_when_carrier_silent() {
        let qm = crate::acpl::AcplQuantMode::Fine;
        let e_c = vec![0.0f32; 1];
        let e_s = vec![1.0f32; 1];
        let alpha_dq = vec![0.0f32; 1];
        let beta = analytic_beta_per_band(&e_c, &e_s, &alpha_dq, qm);
        assert_eq!(beta[0], 0.0);
    }

    /// β F0 + DF round-trip through the ACPL BETA codebooks: a per-band
    /// sequence `[5, 5, 3, 0, 0, 0]` writes to F0(5) + DF(0,-2,-3,0,0)
    /// and the parser's `decode_delta` returns the same values.
    #[test]
    fn write_beta_f0_df_round_trips() {
        use crate::acpl::{get_acpl_hcb, AcplDataType, AcplHcbType, AcplQuantMode};
        let qm = AcplQuantMode::Fine;
        let beta_seq = [5i32, 5, 3, 0, 0, 0];
        let mut bw = BitWriter::new();
        let mut prev = 0i32;
        for (i, &b) in beta_seq.iter().enumerate() {
            if i == 0 {
                write_acpl_beta_f0_value(&mut bw, qm, b);
            } else {
                write_acpl_beta_df_value(&mut bw, qm, b - prev);
            }
            prev = b;
        }
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let hcb_f0 = get_acpl_hcb(AcplDataType::Beta, qm, AcplHcbType::F0);
        let hcb_df = get_acpl_hcb(AcplDataType::Beta, qm, AcplHcbType::Df);
        let mut got = vec![hcb_f0.decode_delta(&mut br).unwrap()];
        for _ in 1..beta_seq.len() {
            got.push(hcb_df.decode_delta(&mut br).unwrap());
        }
        // Differential decode: cumulative sum.
        let mut absvals = Vec::with_capacity(got.len());
        let mut acc = 0;
        for (i, &v) in got.iter().enumerate() {
            if i == 0 {
                acc = v;
            } else {
                acc += v;
            }
            absvals.push(acc);
        }
        assert_eq!(absvals, beta_seq);
    }

    // ================================================================
    // Round 285 — real β₃ (ACPL_3 third-decorrelator gain)
    // ================================================================

    /// `quantise_beta3` follows the Table-207 linear map on the quant
    /// grid for both modes and clamps at the BETA3 F0 codebook's
    /// symmetric `±cb_off` edge (±8 Fine / ±4 Coarse, magnitude bound
    /// 1.0 in both modes).
    #[test]
    fn quantise_beta3_grid_and_clamp() {
        use crate::acpl::AcplQuantMode::{Coarse, Fine};
        // Fine: delta = 0.125 → 0.25 ⇒ 2; -0.5 ⇒ -4; saturation at ±8.
        assert_eq!(quantise_beta3(0.0, Fine), 0);
        assert_eq!(quantise_beta3(0.25, Fine), 2);
        assert_eq!(quantise_beta3(-0.5, Fine), -4);
        assert_eq!(quantise_beta3(1.0, Fine), 8);
        assert_eq!(quantise_beta3(7.5, Fine), 8);
        assert_eq!(quantise_beta3(-99.0, Fine), -8);
        // Coarse: delta = 0.25 → 0.5 ⇒ 2; saturation at ±4.
        assert_eq!(quantise_beta3(0.5, Coarse), 2);
        assert_eq!(quantise_beta3(1.0, Coarse), 4);
        assert_eq!(quantise_beta3(2.0, Coarse), 4);
        assert_eq!(quantise_beta3(-2.0, Coarse), -4);
    }

    /// The BETA3 F0 + DF value writers round-trip through the decoder's
    /// `acpl_huff_data()` parser (DIFF_FREQ direction) and Pseudocode
    /// 121 accumulation for a representative signed sequence.
    #[test]
    fn write_beta3_f0_df_round_trips_through_parse_acpl_huff_data() {
        use crate::acpl::{parse_acpl_huff_data, AcplDataType, AcplQuantMode};
        let qm = AcplQuantMode::Fine;
        let beta3_seq = [3i32, 3, -2, 0, 8, -8];
        let mut bw = BitWriter::new();
        bw.write_bit(false); // diff_type = 0 (DIFF_FREQ)
        let mut prev = 0i32;
        for (i, &b) in beta3_seq.iter().enumerate() {
            if i == 0 {
                write_acpl_beta3_f0_value(&mut bw, qm, b);
            } else {
                write_acpl_beta3_df_value(&mut bw, qm, b - prev);
            }
            prev = b;
        }
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let param =
            parse_acpl_huff_data(&mut br, AcplDataType::Beta3, beta3_seq.len() as u32, 0, qm)
                .expect("parse");
        assert!(!param.direction_time);
        let mut state = crate::acpl_synth::AcplDiffState::new();
        let rows = crate::acpl_synth::differential_decode(
            std::slice::from_ref(&param),
            beta3_seq.len() as u32,
            &mut state,
        );
        assert_eq!(rows[0], beta3_seq);
    }

    /// A centre channel that the γ₅ / γ₆ dry mix reproduces exactly (γ
    /// on the Table-208 quant grid, residual = 0) yields β₃_q = 0 in
    /// every band; a centre with content the dry mix cannot capture
    /// yields β₃_q > 0 in the affected band.
    #[test]
    fn extract_beta3_zero_residual_vs_uncaptured_centre() {
        use crate::acpl::AcplQuantMode::Fine;
        let tl = 1920u32;
        let nb = 12u32;
        let gd = crate::acpl_synth::gamma_delta(Fine);
        let k = 1.0 + (0.5f32).sqrt();
        // L active everywhere; R quiet-but-distinct so the gamma Gram
        // matrix stays non-singular.
        let mut l = vec![0.0f32; tl as usize];
        let mut r = vec![0.0f32; tl as usize];
        for (i, v) in l.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 1.0 } else { 0.5 };
        }
        for (i, v) in r.iter_mut().enumerate() {
            *v = if i % 3 == 0 { 0.8 } else { -0.4 };
        }
        // Exactly-representable centre: C = K · (10·gd · L) → γ₅_q = 10,
        // γ₆_q = 0, residual 0.
        let c_exact: Vec<f32> = l.iter().map(|&x| k * 10.0 * gd * x).collect();
        let (g5_q, g6_q) = extract_gamma_5_6_q_per_band_centre_least_squares(
            &l, &r, &c_exact, tl, nb, 0, 1.0, Fine,
        );
        assert!(g5_q.iter().all(|&q| q == 10), "γ₅_q = 10: {g5_q:?}");
        let zeros = vec![0i32; nb as usize];
        let b3_exact = extract_beta3_q_per_band_centre_residual(
            &l, &r, &c_exact, &zeros, &zeros, &zeros, &zeros, &g5_q, &g6_q, tl, nb, 0, 1.0, Fine,
            Fine,
        );
        assert!(
            b3_exact.iter().all(|&q| q == 0),
            "exact dry fit ⇒ β₃_q = 0 everywhere: {b3_exact:?}"
        );

        // Uncaptured centre: alternate bins orthogonal to L and R within
        // each band (C lives on bins where the dry mix has independent
        // content) → non-zero residual → β₃_q > 0 somewhere.
        let c_orth: Vec<f32> = (0..tl as usize)
            .map(|i| if i % 5 == 1 { 2.0 } else { -1.0 })
            .collect();
        let (g5o, g6o) = extract_gamma_5_6_q_per_band_centre_least_squares(
            &l, &r, &c_orth, tl, nb, 0, 1.0, Fine,
        );
        let b3_orth = extract_beta3_q_per_band_centre_residual(
            &l, &r, &c_orth, &zeros, &zeros, &zeros, &zeros, &g5o, &g6o, tl, nb, 0, 1.0, Fine, Fine,
        );
        assert!(
            b3_orth.iter().any(|&q| q > 0),
            "uncaptured centre ⇒ β₃_q > 0 in ≥ 1 band: {b3_orth:?}"
        );
        // β₃ is a non-negative magnitude decision.
        assert!(b3_orth.iter().all(|&q| q >= 0));
    }

    /// `beta3_scale = 0.0` reproduces the round-215 full-γ builder
    /// byte-for-byte (the all-zero β₃ row emits exactly the zero-delta
    /// scaffold codewords).
    #[test]
    fn build_acpl3_beta3_zero_scale_matches_round215_full_gamma_builder() {
        let tl = 1920u32;
        let n = tl as usize;
        let l: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let r: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
        let c: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let ls: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.04).collect();
        let rs: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.03).collect();
        let aspx_cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::DurationDependent,
        };
        let qm = crate::acpl::AcplQuantMode::Fine;
        let legacy = build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma(
            tl,
            40,
            None,
            true,
            &l,
            &r,
            Some(&c),
            Some(&ls),
            Some(&rs),
            None,
            &aspx_cfg,
            3,
            qm,
            qm,
            0.5,
            0.1,
            1.0,
            8192,
        );
        let with_beta3_off = build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3(
            tl,
            40,
            None,
            true,
            &l,
            &r,
            Some(&c),
            Some(&ls),
            Some(&rs),
            None,
            &aspx_cfg,
            3,
            qm,
            qm,
            0.5,
            0.1,
            1.0,
            0.0,
            8192,
        );
        assert_eq!(legacy, with_beta3_off);
    }

    /// A representative high-res FIXFIX `aspx_config` with several noise
    /// subband groups, used by the `_tna` round-trip tests.
    fn tna_test_cfg() -> aspx::AspxConfig {
        aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 3, // larger noise-SBG count so tna_mode has width
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::High,
        }
    }

    /// `write_aspx_data_1ch_real_envelope_tna` with an all-zero tna_mode
    /// reproduces the non-tna writer byte-for-byte.
    #[test]
    fn write_1ch_tna_all_zero_matches_base_writer() {
        let cfg = tna_test_cfg();
        let sig = [5_i32, 1, -1, 0, 2];
        let noise = [3_i32, -1, 0, 1];
        let ch = AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise,
        };
        let mut bw_base = BitWriter::new();
        write_aspx_data_1ch_real_envelope(&mut bw_base, &cfg, ch).unwrap();
        bw_base.align_to_byte();
        let base = bw_base.finish();

        let zeros = vec![0u8; 8];
        let mut bw_tna = BitWriter::new();
        write_aspx_data_1ch_real_envelope_tna(&mut bw_tna, &cfg, ch, &zeros).unwrap();
        bw_tna.align_to_byte();
        let tna = bw_tna.finish();
        assert_eq!(base, tna, "all-zero tna must match the base writer");
    }

    /// A non-zero tna_mode produces different bytes than the all-zero
    /// scaffold and round-trips through `parse_aspx_hfgen_iwc_1ch`.
    #[test]
    fn write_1ch_tna_round_trips_through_parser() {
        let cfg = tna_test_cfg();
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let counts = tables.counts;
        let n_noise = counts.num_sbg_noise as usize;
        assert!(n_noise >= 2, "need multiple noise SBGs to exercise tna");

        // Distinct per-SBG modes (cycling 0..=3) so position matters.
        let tna_mode: Vec<u8> = (0..n_noise).map(|i| (i % 4) as u8).collect();
        let sig = [5_i32, 1, -1, 0, 2, 0, 1];
        let noise = [3_i32, -1, 0, 1, 0];
        let ch = AspxRealEnvelopeChannel {
            sig: &sig,
            noise: &noise,
        };

        let mut bw = BitWriter::new();
        write_aspx_data_1ch_real_envelope_tna(&mut bw, &cfg, ch, &tna_mode).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();

        // Re-parse the leading header to reach the hfgen element, then the
        // hfgen, and confirm tna_mode round-trips.
        let mut br = BitReader::new(&bytes);
        br.read_u32(3).unwrap(); // aspx_xover_subband_offset
        br.read_bit().unwrap(); // FIXFIX prefix
        let envbits = cfg.fixfix_tmp_num_env_bits();
        if envbits > 0 {
            br.read_u32(envbits).unwrap();
        }
        if cfg.signals_freq_res() {
            br.read_bit().unwrap(); // aspx_freq_res[0]
        }
        br.read_bit().unwrap(); // sig delta_dir
        br.read_bit().unwrap(); // noise delta_dir
        let hfgen = aspx::parse_aspx_hfgen_iwc_1ch(
            &mut br,
            counts.num_sbg_noise,
            counts.num_sbg_sig_highres,
            0,
        )
        .unwrap();
        assert_eq!(hfgen.tna_mode, tna_mode);
        assert!(!hfgen.ah_present);
        assert!(!hfgen.fic_present);
        assert!(!hfgen.tic_present);

        // The tna body must differ from the all-zero scaffold (some mode
        // is non-zero).
        let mut bw_zero = BitWriter::new();
        write_aspx_data_1ch_real_envelope(&mut bw_zero, &cfg, ch).unwrap();
        bw_zero.align_to_byte();
        assert_ne!(bytes, bw_zero.finish());
    }

    /// `write_aspx_data_2ch_real_envelope_tna` with all-zero tna matches
    /// the base 2ch writer; a non-zero tna round-trips (balance = 1 →
    /// channel 0's modes mirror to channel 1).
    #[test]
    fn write_2ch_tna_matches_base_and_round_trips() {
        let cfg = tna_test_cfg();
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let counts = tables.counts;
        let n_noise = counts.num_sbg_noise as usize;

        let s0 = [4_i32, 1, 0, -1, 2, 0, 1];
        let n0 = [2_i32, 0, -1, 1, 0];
        let s1 = [3_i32, -1, 1, 0, 0, 1, 0];
        let n1 = [1_i32, 1, 0, -1, 0];
        let ch0 = AspxRealEnvelopeChannel {
            sig: &s0,
            noise: &n0,
        };
        let ch1 = AspxRealEnvelopeChannel {
            sig: &s1,
            noise: &n1,
        };

        // All-zero tna == base writer.
        let zeros = vec![0u8; n_noise];
        let mut bw_base = BitWriter::new();
        write_aspx_data_2ch_real_envelope(&mut bw_base, &cfg, ch0, ch1).unwrap();
        bw_base.align_to_byte();
        let base = bw_base.finish();
        let mut bw_tz = BitWriter::new();
        write_aspx_data_2ch_real_envelope_tna(&mut bw_tz, &cfg, ch0, ch1, &zeros).unwrap();
        bw_tz.align_to_byte();
        assert_eq!(base, bw_tz.finish());

        // Non-zero tna round-trips (balance = 1 mirrors ch0 → ch1).
        let tna_mode: Vec<u8> = (0..n_noise).map(|i| ((i + 1) % 4) as u8).collect();
        let mut bw = BitWriter::new();
        write_aspx_data_2ch_real_envelope_tna(&mut bw, &cfg, ch0, ch1, &tna_mode).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        assert_ne!(bytes, base);

        let mut br = BitReader::new(&bytes);
        br.read_u32(3).unwrap(); // xover
        br.read_bit().unwrap(); // FIXFIX prefix
        let envbits = cfg.fixfix_tmp_num_env_bits();
        if envbits > 0 {
            br.read_u32(envbits).unwrap();
        }
        if cfg.signals_freq_res() {
            br.read_bit().unwrap();
        }
        let balance = br.read_bit().unwrap();
        assert!(balance, "writer uses aspx_balance = 1");
        br.read_bit().unwrap(); // ch0 sig delta_dir
        br.read_bit().unwrap(); // ch0 noise delta_dir
        br.read_bit().unwrap(); // ch1 sig delta_dir
        br.read_bit().unwrap(); // ch1 noise delta_dir
        let hfgen = aspx::parse_aspx_hfgen_iwc_2ch(
            &mut br,
            balance,
            counts.num_sbg_noise,
            counts.num_sbg_sig_highres,
            0,
        )
        .unwrap();
        assert_eq!(hfgen.tna_mode[0], tna_mode);
        // balance = 1 → channel 1 mirrors channel 0.
        assert_eq!(hfgen.tna_mode[1], tna_mode);
    }
}
