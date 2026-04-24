//! AC-4 Advanced Spectral Extension (A-SPX) parameter parsing.
//!
//! A-SPX is AC-4's spectral-bandwidth-extension tool (ETSI TS 103 190-1
//! §4.2.12 / §4.3.10 / §5.7.6): the core codec carries the low band and
//! A-SPX reconstructs the high band in the QMF domain from a compact
//! envelope-energy sidecar and a small set of control parameters.
//!
//! This module implements:
//!
//! * **`aspx_config()`** (Table 50, §4.2.12.1) — the 15-bit I-frame
//!   configuration block.
//! * **`companding_control()`** (Table 49, §4.2.11) — the per-channel
//!   companding sidecar.
//! * **`aspx_framing()`** (Table 53, §4.2.12.4) — the per-channel
//!   envelope framing. This block is variable-width: its size depends
//!   on the `aspx_int_class` (a 1/2/3-bit prefix code selecting one of
//!   FIXFIX / FIXVAR / VARFIX / VARVAR), on the config fields
//!   `aspx_num_env_bits_fixfix` and `aspx_freq_res_mode`, on whether
//!   the frame is an I-frame (only I-frames carry the leading
//!   `aspx_var_bord_left` for VARFIX / VARVAR), and on a tool-level
//!   `num_aspx_timeslots` flag that selects between 1- and 2-bit fields
//!   for `aspx_num_rel_*` / `aspx_rel_bord_*` (Note 1 in Table 53).
//!   Together these fully determine the signal- and noise-envelope
//!   count (`aspx_num_env`, `aspx_num_noise`) for each channel — the
//!   inputs the envelope Huffman payload will be parameterised by.
//!
//! * **`aspx_delta_dir()`** (Table 54, §4.2.12.5) — one bit per signal
//!   envelope plus one per noise envelope.
//! * **`aspx_hfgen_iwc_1ch()`** (Table 55, §4.2.12.6) — 1-channel
//!   HF-generation control: `tna_mode[0..num_sbg_noise]`,
//!   `aspx_ah_present` + optional `add_harmonic[0..num_sbg_sig_highres]`,
//!   `aspx_fic_present` + optional `fic_used_in_sfb[0..]`, and
//!   `aspx_tic_present` + optional
//!   `tic_used_in_slot[0..num_aspx_timeslots]`. The three sbg / ats
//!   counts come from the A-SPX freq-scale derivation and are passed
//!   in from the caller.
//! * **`aspx_hfgen_iwc_2ch()`** (Table 56, §4.2.12.7) — 2-channel
//!   analogue with explicit per-channel `aspx_ah_left` / `_right` and
//!   `aspx_fic_left` / `_right` gates, plus `aspx_tic_copy` which
//!   lets the encoder mirror the left-channel TIC pattern into the
//!   right. Mirrors the pseudocode in Table 56 byte-for-byte.
//!
//! We deliberately stop short of:
//!
//! * `aspx_ec_data()` and the A-SPX Huffman tables in Annex A.2 —
//!   envelope and noise scale factors are Huffman-coded with dedicated
//!   codebooks (different ones per delta direction, quantization step,
//!   signal vs noise etc.) and would double the crate's line count.
//!   See the module-level roadmap in `lib.rs`.
//!
//! Even so, `parse_aspx_config()` is useful on its own: it unblocks the
//! `mono_codec_mode == ASPX` I-frame path in the outer `audio_data()`
//! walker, so the substream walker no longer bails out silently on
//! ASPX substreams. It just stops one syntax element earlier — after
//! the config — instead of at the `mono_codec_mode` flag.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::aspx_huffman;

/// Metadata + bit-level decoder for one A-SPX Huffman codebook per
/// Annex A.2 (Tables A.16 .. A.33).
///
/// The eighteen A-SPX envelope / noise codebooks come in three flavours
/// each:
///
/// * `*_F0` — base-frequency (raw index). Read directly.
/// * `*_DF` — delta-frequency. Stored value = `index - cb_off`.
/// * `*_DT` — delta-time. Stored value = `index - cb_off`.
///
/// Accordingly each codebook has a `len[]` table, a `cw[]` table, and
/// (for DF/DT) a `cb_off`. The metadata captured here matches Annex
/// A.2's per-table header fields verbatim — `codebook_length` and
/// (where present) `cb_off`. Unlike the 11 spectral ASF codebooks in
/// [`crate::huffman`] there is no `dim` / `cb_mod` / `unsigned` —
/// A-SPX codebooks are always `dim = 1`.
pub struct AspxHcb {
    /// Name (e.g. `"ASPX_HCB_ENV_BALANCE_15_DF"`) — for diagnostics.
    pub name: &'static str,
    /// Codeword lengths in bits, indexed by symbol.
    pub len: &'static [u8],
    /// Codewords packed MSB-first, indexed by symbol.
    pub cw: &'static [u32],
    /// `cb_off` per Annex A.2 — the stored-symbol-to-delta offset.
    /// For `*_F0` tables this is 0 (no offset in the spec). For
    /// `*_DF` / `*_DT` tables it's the table's `cb_off` header, so
    /// the caller can recover the delta as `symbol_index - cb_off`.
    pub cb_off: i32,
}

impl AspxHcb {
    /// Decode one symbol from `br` and return the recovered delta
    /// (symbol index minus `cb_off`). Uses the same "try one entry at
    /// a time, widest-match-wins" algorithm as the ASF Huffman
    /// decoder in [`crate::huffman::huff_decode`] — suitable for
    /// correctness tests; not optimised.
    pub fn decode_delta(&self, br: &mut BitReader<'_>) -> Result<i32> {
        debug_assert_eq!(self.len.len(), self.cw.len());
        let mut code: u32 = 0;
        let mut width: u8 = 0;
        while width < 32 {
            let b = br.read_u32(1)?;
            code = (code << 1) | b;
            width += 1;
            for (i, &l) in self.len.iter().enumerate() {
                if l == width && self.cw[i] == code {
                    return Ok(i as i32 - self.cb_off);
                }
            }
        }
        Err(Error::invalid("ac4: no matching A-SPX Huffman codeword"))
    }
}

/// Metadata for all eighteen A-SPX Huffman codebooks (Annex A.2,
/// Tables A.16..=A.33). `len` / `cw` are `None` where the codeword
/// arrays haven't been transcribed yet — those tables live in the
/// spec's normative accompaniment file `ts_103190_tables.c` inside
/// `ts_10319001v010401p0.zip`. `cb_off` and `codebook_length` are
/// carried for every table straight off the PDF headers so sizing
/// assertions stay checkable today.
#[derive(Debug, Clone, Copy)]
pub struct AspxHcbMeta {
    pub name: &'static str,
    pub codebook_length: u32,
    pub cb_off: i32,
}

/// Table A.16: `ASPX_HCB_ENV_LEVEL_15_F0` (codebook_length = 71).
pub const ASPX_HCB_ENV_LEVEL_15_F0_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_LEVEL_15_F0",
    codebook_length: 71,
    cb_off: 0,
};
/// Table A.17: `ASPX_HCB_ENV_LEVEL_15_DF` (codebook_length = 141,
/// cb_off = 70).
pub const ASPX_HCB_ENV_LEVEL_15_DF_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_LEVEL_15_DF",
    codebook_length: 141,
    cb_off: 70,
};
/// Table A.18: `ASPX_HCB_ENV_LEVEL_15_DT` (codebook_length = 141,
/// cb_off = 70).
pub const ASPX_HCB_ENV_LEVEL_15_DT_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_LEVEL_15_DT",
    codebook_length: 141,
    cb_off: 70,
};
/// Table A.19: `ASPX_HCB_ENV_BALANCE_15_F0` (codebook_length = 25).
pub const ASPX_HCB_ENV_BALANCE_15_F0_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_BALANCE_15_F0",
    codebook_length: 25,
    cb_off: 0,
};
/// Table A.20: `ASPX_HCB_ENV_BALANCE_15_DF` (codebook_length = 49,
/// cb_off = 24).
pub const ASPX_HCB_ENV_BALANCE_15_DF_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_BALANCE_15_DF",
    codebook_length: 49,
    cb_off: 24,
};
/// Table A.21: `ASPX_HCB_ENV_BALANCE_15_DT` (codebook_length = 49,
/// cb_off = 24).
pub const ASPX_HCB_ENV_BALANCE_15_DT_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_BALANCE_15_DT",
    codebook_length: 49,
    cb_off: 24,
};
/// Table A.22: `ASPX_HCB_ENV_LEVEL_30_F0` (codebook_length = 36).
pub const ASPX_HCB_ENV_LEVEL_30_F0_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_LEVEL_30_F0",
    codebook_length: 36,
    cb_off: 0,
};
/// Table A.23: `ASPX_HCB_ENV_LEVEL_30_DF` (codebook_length = 71,
/// cb_off = 35).
pub const ASPX_HCB_ENV_LEVEL_30_DF_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_LEVEL_30_DF",
    codebook_length: 71,
    cb_off: 35,
};
/// Table A.24: `ASPX_HCB_ENV_LEVEL_30_DT` (codebook_length = 71,
/// cb_off = 35).
pub const ASPX_HCB_ENV_LEVEL_30_DT_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_LEVEL_30_DT",
    codebook_length: 71,
    cb_off: 35,
};
/// Table A.25: `ASPX_HCB_ENV_BALANCE_30_F0` (codebook_length = 13).
pub const ASPX_HCB_ENV_BALANCE_30_F0_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_BALANCE_30_F0",
    codebook_length: 13,
    cb_off: 0,
};
/// Table A.26: `ASPX_HCB_ENV_BALANCE_30_DF` (codebook_length = 25,
/// cb_off = 12).
pub const ASPX_HCB_ENV_BALANCE_30_DF_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_BALANCE_30_DF",
    codebook_length: 25,
    cb_off: 12,
};
/// Table A.27: `ASPX_HCB_ENV_BALANCE_30_DT` (codebook_length = 25,
/// cb_off = 12).
pub const ASPX_HCB_ENV_BALANCE_30_DT_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_ENV_BALANCE_30_DT",
    codebook_length: 25,
    cb_off: 12,
};
/// Table A.28: `ASPX_HCB_NOISE_LEVEL_F0` (codebook_length = 30).
pub const ASPX_HCB_NOISE_LEVEL_F0_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_NOISE_LEVEL_F0",
    codebook_length: 30,
    cb_off: 0,
};
/// Table A.29: `ASPX_HCB_NOISE_LEVEL_DF` (codebook_length = 59,
/// cb_off = 29).
pub const ASPX_HCB_NOISE_LEVEL_DF_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_NOISE_LEVEL_DF",
    codebook_length: 59,
    cb_off: 29,
};
/// Table A.30: `ASPX_HCB_NOISE_LEVEL_DT` (codebook_length = 59,
/// cb_off = 29).
pub const ASPX_HCB_NOISE_LEVEL_DT_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_NOISE_LEVEL_DT",
    codebook_length: 59,
    cb_off: 29,
};
/// Table A.31: `ASPX_HCB_NOISE_BALANCE_F0` (codebook_length = 13).
pub const ASPX_HCB_NOISE_BALANCE_F0_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_NOISE_BALANCE_F0",
    codebook_length: 13,
    cb_off: 0,
};
/// Table A.32: `ASPX_HCB_NOISE_BALANCE_DF` (codebook_length = 25,
/// cb_off = 12).
pub const ASPX_HCB_NOISE_BALANCE_DF_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_NOISE_BALANCE_DF",
    codebook_length: 25,
    cb_off: 12,
};
/// Table A.33: `ASPX_HCB_NOISE_BALANCE_DT` (codebook_length = 25,
/// cb_off = 12).
pub const ASPX_HCB_NOISE_BALANCE_DT_META: AspxHcbMeta = AspxHcbMeta {
    name: "ASPX_HCB_NOISE_BALANCE_DT",
    codebook_length: 25,
    cb_off: 12,
};

/// Identifier for one of the eighteen A-SPX Huffman codebooks in
/// Annex A.2 (Tables A.16..=A.33).
///
/// The encoding choice in `aspx_ec_data()` is driven by the triple
/// `(signal/noise, quant-step, delta direction)` — here split across
/// the `EnvLevel15` / `EnvLevel30` / `EnvBalance15` / `EnvBalance30` /
/// `Noise*` prefixes (signal-vs-noise + level-vs-balance + 15 dB vs
/// 30 dB step) and the `F0` / `DF` / `DT` suffixes (raw value,
/// delta-frequency, delta-time). See §4.3.10.5 / Tables 130.. for the
/// selection rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanCodebookId {
    EnvLevel15F0,
    EnvLevel15Df,
    EnvLevel15Dt,
    EnvBalance15F0,
    EnvBalance15Df,
    EnvBalance15Dt,
    EnvLevel30F0,
    EnvLevel30Df,
    EnvLevel30Dt,
    EnvBalance30F0,
    EnvBalance30Df,
    EnvBalance30Dt,
    NoiseLevelF0,
    NoiseLevelDf,
    NoiseLevelDt,
    NoiseBalanceF0,
    NoiseBalanceDf,
    NoiseBalanceDt,
}

/// Table A.16 — `ASPX_HCB_ENV_LEVEL_15_F0` decoder.
pub static ASPX_HCB_ENV_LEVEL_15_F0: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_LEVEL_15_F0",
    len: aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_LEVEL_15_F0_CW,
    cb_off: 0,
};
/// Table A.17 — `ASPX_HCB_ENV_LEVEL_15_DF` decoder.
pub static ASPX_HCB_ENV_LEVEL_15_DF: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_LEVEL_15_DF",
    len: aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DF_CW,
    cb_off: 70,
};
/// Table A.18 — `ASPX_HCB_ENV_LEVEL_15_DT` decoder.
pub static ASPX_HCB_ENV_LEVEL_15_DT: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_LEVEL_15_DT",
    len: aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DT_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_LEVEL_15_DT_CW,
    cb_off: 70,
};
/// Table A.19 — `ASPX_HCB_ENV_BALANCE_15_F0` decoder.
pub static ASPX_HCB_ENV_BALANCE_15_F0: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_BALANCE_15_F0",
    len: aspx_huffman::ASPX_HCB_ENV_BALANCE_15_F0_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_BALANCE_15_F0_CW,
    cb_off: 0,
};
/// Table A.20 — `ASPX_HCB_ENV_BALANCE_15_DF` decoder.
pub static ASPX_HCB_ENV_BALANCE_15_DF: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_BALANCE_15_DF",
    len: aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DF_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DF_CW,
    cb_off: 24,
};
/// Table A.21 — `ASPX_HCB_ENV_BALANCE_15_DT` decoder.
pub static ASPX_HCB_ENV_BALANCE_15_DT: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_BALANCE_15_DT",
    len: aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DT_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_BALANCE_15_DT_CW,
    cb_off: 24,
};
/// Table A.22 — `ASPX_HCB_ENV_LEVEL_30_F0` decoder.
pub static ASPX_HCB_ENV_LEVEL_30_F0: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_LEVEL_30_F0",
    len: aspx_huffman::ASPX_HCB_ENV_LEVEL_30_F0_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_LEVEL_30_F0_CW,
    cb_off: 0,
};
/// Table A.23 — `ASPX_HCB_ENV_LEVEL_30_DF` decoder.
pub static ASPX_HCB_ENV_LEVEL_30_DF: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_LEVEL_30_DF",
    len: aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DF_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DF_CW,
    cb_off: 35,
};
/// Table A.24 — `ASPX_HCB_ENV_LEVEL_30_DT` decoder.
pub static ASPX_HCB_ENV_LEVEL_30_DT: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_LEVEL_30_DT",
    len: aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DT_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_LEVEL_30_DT_CW,
    cb_off: 35,
};
/// Table A.25 — `ASPX_HCB_ENV_BALANCE_30_F0` decoder.
pub static ASPX_HCB_ENV_BALANCE_30_F0: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_BALANCE_30_F0",
    len: aspx_huffman::ASPX_HCB_ENV_BALANCE_30_F0_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_BALANCE_30_F0_CW,
    cb_off: 0,
};
/// Table A.26 — `ASPX_HCB_ENV_BALANCE_30_DF` decoder.
pub static ASPX_HCB_ENV_BALANCE_30_DF: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_BALANCE_30_DF",
    len: aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DF_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DF_CW,
    cb_off: 12,
};
/// Table A.27 — `ASPX_HCB_ENV_BALANCE_30_DT` decoder.
pub static ASPX_HCB_ENV_BALANCE_30_DT: AspxHcb = AspxHcb {
    name: "ASPX_HCB_ENV_BALANCE_30_DT",
    len: aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DT_LEN,
    cw: aspx_huffman::ASPX_HCB_ENV_BALANCE_30_DT_CW,
    cb_off: 12,
};
/// Table A.28 — `ASPX_HCB_NOISE_LEVEL_F0` decoder.
pub static ASPX_HCB_NOISE_LEVEL_F0: AspxHcb = AspxHcb {
    name: "ASPX_HCB_NOISE_LEVEL_F0",
    len: aspx_huffman::ASPX_HCB_NOISE_LEVEL_F0_LEN,
    cw: aspx_huffman::ASPX_HCB_NOISE_LEVEL_F0_CW,
    cb_off: 0,
};
/// Table A.29 — `ASPX_HCB_NOISE_LEVEL_DF` decoder.
pub static ASPX_HCB_NOISE_LEVEL_DF: AspxHcb = AspxHcb {
    name: "ASPX_HCB_NOISE_LEVEL_DF",
    len: aspx_huffman::ASPX_HCB_NOISE_LEVEL_DF_LEN,
    cw: aspx_huffman::ASPX_HCB_NOISE_LEVEL_DF_CW,
    cb_off: 29,
};
/// Table A.30 — `ASPX_HCB_NOISE_LEVEL_DT` decoder.
pub static ASPX_HCB_NOISE_LEVEL_DT: AspxHcb = AspxHcb {
    name: "ASPX_HCB_NOISE_LEVEL_DT",
    len: aspx_huffman::ASPX_HCB_NOISE_LEVEL_DT_LEN,
    cw: aspx_huffman::ASPX_HCB_NOISE_LEVEL_DT_CW,
    cb_off: 29,
};
/// Table A.31 — `ASPX_HCB_NOISE_BALANCE_F0` decoder.
pub static ASPX_HCB_NOISE_BALANCE_F0: AspxHcb = AspxHcb {
    name: "ASPX_HCB_NOISE_BALANCE_F0",
    len: aspx_huffman::ASPX_HCB_NOISE_BALANCE_F0_LEN,
    cw: aspx_huffman::ASPX_HCB_NOISE_BALANCE_F0_CW,
    cb_off: 0,
};
/// Table A.32 — `ASPX_HCB_NOISE_BALANCE_DF` decoder.
pub static ASPX_HCB_NOISE_BALANCE_DF: AspxHcb = AspxHcb {
    name: "ASPX_HCB_NOISE_BALANCE_DF",
    len: aspx_huffman::ASPX_HCB_NOISE_BALANCE_DF_LEN,
    cw: aspx_huffman::ASPX_HCB_NOISE_BALANCE_DF_CW,
    cb_off: 12,
};
/// Table A.33 — `ASPX_HCB_NOISE_BALANCE_DT` decoder.
pub static ASPX_HCB_NOISE_BALANCE_DT: AspxHcb = AspxHcb {
    name: "ASPX_HCB_NOISE_BALANCE_DT",
    len: aspx_huffman::ASPX_HCB_NOISE_BALANCE_DT_LEN,
    cw: aspx_huffman::ASPX_HCB_NOISE_BALANCE_DT_CW,
    cb_off: 12,
};

/// Resolve an [`HuffmanCodebookId`] into the matching [`AspxHcb`]
/// decoder instance. All 18 Annex A.2 codebooks are covered.
pub fn lookup_aspx_hcb(id: HuffmanCodebookId) -> &'static AspxHcb {
    match id {
        HuffmanCodebookId::EnvLevel15F0 => &ASPX_HCB_ENV_LEVEL_15_F0,
        HuffmanCodebookId::EnvLevel15Df => &ASPX_HCB_ENV_LEVEL_15_DF,
        HuffmanCodebookId::EnvLevel15Dt => &ASPX_HCB_ENV_LEVEL_15_DT,
        HuffmanCodebookId::EnvBalance15F0 => &ASPX_HCB_ENV_BALANCE_15_F0,
        HuffmanCodebookId::EnvBalance15Df => &ASPX_HCB_ENV_BALANCE_15_DF,
        HuffmanCodebookId::EnvBalance15Dt => &ASPX_HCB_ENV_BALANCE_15_DT,
        HuffmanCodebookId::EnvLevel30F0 => &ASPX_HCB_ENV_LEVEL_30_F0,
        HuffmanCodebookId::EnvLevel30Df => &ASPX_HCB_ENV_LEVEL_30_DF,
        HuffmanCodebookId::EnvLevel30Dt => &ASPX_HCB_ENV_LEVEL_30_DT,
        HuffmanCodebookId::EnvBalance30F0 => &ASPX_HCB_ENV_BALANCE_30_F0,
        HuffmanCodebookId::EnvBalance30Df => &ASPX_HCB_ENV_BALANCE_30_DF,
        HuffmanCodebookId::EnvBalance30Dt => &ASPX_HCB_ENV_BALANCE_30_DT,
        HuffmanCodebookId::NoiseLevelF0 => &ASPX_HCB_NOISE_LEVEL_F0,
        HuffmanCodebookId::NoiseLevelDf => &ASPX_HCB_NOISE_LEVEL_DF,
        HuffmanCodebookId::NoiseLevelDt => &ASPX_HCB_NOISE_LEVEL_DT,
        HuffmanCodebookId::NoiseBalanceF0 => &ASPX_HCB_NOISE_BALANCE_F0,
        HuffmanCodebookId::NoiseBalanceDf => &ASPX_HCB_NOISE_BALANCE_DF,
        HuffmanCodebookId::NoiseBalanceDt => &ASPX_HCB_NOISE_BALANCE_DT,
    }
}

/// Every codebook shipped in this module — used for table-wide
/// correctness tests + diagnostics.
pub static ASPX_HCB_ALL: &[(HuffmanCodebookId, &AspxHcb)] = &[
    (HuffmanCodebookId::EnvLevel15F0, &ASPX_HCB_ENV_LEVEL_15_F0),
    (HuffmanCodebookId::EnvLevel15Df, &ASPX_HCB_ENV_LEVEL_15_DF),
    (HuffmanCodebookId::EnvLevel15Dt, &ASPX_HCB_ENV_LEVEL_15_DT),
    (
        HuffmanCodebookId::EnvBalance15F0,
        &ASPX_HCB_ENV_BALANCE_15_F0,
    ),
    (
        HuffmanCodebookId::EnvBalance15Df,
        &ASPX_HCB_ENV_BALANCE_15_DF,
    ),
    (
        HuffmanCodebookId::EnvBalance15Dt,
        &ASPX_HCB_ENV_BALANCE_15_DT,
    ),
    (HuffmanCodebookId::EnvLevel30F0, &ASPX_HCB_ENV_LEVEL_30_F0),
    (HuffmanCodebookId::EnvLevel30Df, &ASPX_HCB_ENV_LEVEL_30_DF),
    (HuffmanCodebookId::EnvLevel30Dt, &ASPX_HCB_ENV_LEVEL_30_DT),
    (
        HuffmanCodebookId::EnvBalance30F0,
        &ASPX_HCB_ENV_BALANCE_30_F0,
    ),
    (
        HuffmanCodebookId::EnvBalance30Df,
        &ASPX_HCB_ENV_BALANCE_30_DF,
    ),
    (
        HuffmanCodebookId::EnvBalance30Dt,
        &ASPX_HCB_ENV_BALANCE_30_DT,
    ),
    (HuffmanCodebookId::NoiseLevelF0, &ASPX_HCB_NOISE_LEVEL_F0),
    (HuffmanCodebookId::NoiseLevelDf, &ASPX_HCB_NOISE_LEVEL_DF),
    (HuffmanCodebookId::NoiseLevelDt, &ASPX_HCB_NOISE_LEVEL_DT),
    (
        HuffmanCodebookId::NoiseBalanceF0,
        &ASPX_HCB_NOISE_BALANCE_F0,
    ),
    (
        HuffmanCodebookId::NoiseBalanceDf,
        &ASPX_HCB_NOISE_BALANCE_DF,
    ),
    (
        HuffmanCodebookId::NoiseBalanceDt,
        &ASPX_HCB_NOISE_BALANCE_DT,
    ),
];

/// A-SPX frequency-resolution transmission mode (Table 124).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxFreqResMode {
    /// `aspx_freq_res` is signalled explicitly in `aspx_framing()`.
    Signalled,
    /// Defaults to low resolution.
    Low,
    /// Defaults to values dependent on signal envelope duration.
    DurationDependent,
    /// Defaults to high resolution.
    High,
}

impl AspxFreqResMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b11 {
            0 => Self::Signalled,
            1 => Self::Low,
            2 => Self::DurationDependent,
            _ => Self::High,
        }
    }
}

/// A-SPX master frequency-table scale (Table 119). `false` == low-bit-
/// rate scale-factor table (`sbg_template_lowres`), `true` == high-bit-
/// rate table (`sbg_template_highres`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxMasterFreqScale {
    LowRes,
    HighRes,
}

impl AspxMasterFreqScale {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::LowRes
        } else {
            Self::HighRes
        }
    }
}

/// A-SPX initial envelope quantization step (Table 118). Used as the
/// seed for `aspx_qmode_env[ch]`; re-initialised by `aspx_data_1ch()` /
/// `aspx_data_2ch()` when the interval class is FIXFIX with exactly one
/// envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxQuantStep {
    /// 1.5 dB step size.
    Fine,
    /// 3.0 dB step size.
    Coarse,
}

impl AspxQuantStep {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::Fine
        } else {
            Self::Coarse
        }
    }
}

/// Parsed `aspx_config()` (ETSI TS 103 190-1 §4.2.12.1, Table 50).
///
/// Total wire size is 15 bits: a single I-frame-sticky block that
/// configures the A-SPX processor for the substream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AspxConfig {
    /// `aspx_quant_mode_env` — 1 bit, Table 118.
    pub quant_mode_env: AspxQuantStep,
    /// `aspx_start_freq` — 3 bits, index into scale-factor subband group
    /// table counting up in 2-subband steps from the first subband
    /// (§4.3.10.1.2).
    pub start_freq: u8,
    /// `aspx_stop_freq` — 2 bits, index counting down in 2-subband
    /// steps from the last subband (§4.3.10.1.3).
    pub stop_freq: u8,
    /// `aspx_master_freq_scale` — 1 bit, Table 119.
    pub master_freq_scale: AspxMasterFreqScale,
    /// `aspx_interpolation` — 1 bit, Table 120.
    pub interpolation: bool,
    /// `aspx_preflat` — 1 bit, Table 121.
    pub preflat: bool,
    /// `aspx_limiter` — 1 bit, Table 122.
    pub limiter: bool,
    /// `aspx_noise_sbg` — 2 bits. Input to
    /// `numNoiseSbgroups = aspx_noise_sbg + 1` (§5.7.6.3.1.3).
    pub noise_sbg: u8,
    /// `aspx_num_env_bits_fixfix` — 1 bit, Table 123. When 0 the FIXFIX
    /// `tmp_num_env` field is 1 bit wide (1 or 2 envelopes); when 1
    /// it's 2 bits wide (1, 2 or 4 envelopes).
    pub num_env_bits_fixfix: u8,
    /// `aspx_freq_res_mode` — 2 bits, Table 124.
    pub freq_res_mode: AspxFreqResMode,
}

impl AspxConfig {
    /// Bit width of this config element on the wire.
    pub const BITS: u32 = 15;

    /// Returns `numNoiseSbgroups` per §5.7.6.3.1.3:
    /// `numNoiseSbgroups = aspx_noise_sbg + 1` (so one of 1..=4).
    pub fn num_noise_sbgroups(&self) -> u32 {
        self.noise_sbg as u32 + 1
    }

    /// Returns how many bits the `tmp_num_env` field in a FIXFIX
    /// `aspx_framing()` will take (1 or 2) per §4.3.10.1.9.
    pub fn fixfix_tmp_num_env_bits(&self) -> u32 {
        self.num_env_bits_fixfix as u32 + 1
    }

    /// Returns `true` when the configuration calls for explicit
    /// `aspx_freq_res` bits in `aspx_framing()` (Table 124 row 0).
    pub fn signals_freq_res(&self) -> bool {
        matches!(self.freq_res_mode, AspxFreqResMode::Signalled)
    }
}

/// Parse `aspx_config()` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 50 (§4.2.12.1). Consumes exactly 15 bits.
pub fn parse_aspx_config(br: &mut BitReader<'_>) -> Result<AspxConfig> {
    // Field order straight from Table 50.
    let quant_mode_env = AspxQuantStep::from_bit(br.read_u32(1)?);
    let start_freq = br.read_u32(3)? as u8;
    let stop_freq = br.read_u32(2)? as u8;
    let master_freq_scale = AspxMasterFreqScale::from_bit(br.read_u32(1)?);
    let interpolation = br.read_bit()?;
    let preflat = br.read_bit()?;
    let limiter = br.read_bit()?;
    let noise_sbg = br.read_u32(2)? as u8;
    let num_env_bits_fixfix = br.read_u32(1)? as u8;
    let freq_res_mode = AspxFreqResMode::from_u32(br.read_u32(2)?);
    Ok(AspxConfig {
        quant_mode_env,
        start_freq,
        stop_freq,
        master_freq_scale,
        interpolation,
        preflat,
        limiter,
        noise_sbg,
        num_env_bits_fixfix,
        freq_res_mode,
    })
}

/// Parsed `companding_control()` (ETSI TS 103 190-1 §4.2.11, Table 49).
///
/// Companding is a per-channel transient-response tool that toggles
/// short-time gain compression/expansion. This struct captures the
/// flags exactly as the bitstream carries them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompandingControl {
    /// Present only for `num_chan > 1`; absent for mono.
    pub sync_flag: Option<bool>,
    /// `b_compand_on[ch]` — length is `1` when `sync_flag == true`, else
    /// `num_chan`.
    pub compand_on: Vec<bool>,
    /// `b_compand_avg` — only present when at least one channel had
    /// companding off.
    pub compand_avg: Option<bool>,
}

/// Parse `companding_control(num_chan)` at the current bit-reader
/// position. `num_chan` matches the caller's companding-grouping arg:
/// 1 for single-channel calls, 2 for stereo, 3 / 5 for multi-channel.
pub fn parse_companding_control(
    br: &mut BitReader<'_>,
    num_chan: u32,
) -> Result<CompandingControl> {
    let sync_flag = if num_chan > 1 {
        Some(br.read_bit()?)
    } else {
        None
    };
    let nc = match sync_flag {
        Some(true) => 1,
        _ => num_chan,
    };
    let mut compand_on = Vec::with_capacity(nc as usize);
    let mut b_need_avg = false;
    for _ in 0..nc {
        let bit = br.read_bit()?;
        if !bit {
            b_need_avg = true;
        }
        compand_on.push(bit);
    }
    let compand_avg = if b_need_avg {
        Some(br.read_bit()?)
    } else {
        None
    };
    Ok(CompandingControl {
        sync_flag,
        compand_on,
        compand_avg,
    })
}

/// A-SPX interval class (ETSI TS 103 190-1 §4.3.10.4.1, Table 126).
///
/// Encoded on the wire as a 1/2/3-bit variable-length prefix code:
/// `0b0` → FIXFIX, `0b10` → FIXVAR, `0b110` → VARFIX, `0b111` → VARVAR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxIntClass {
    FixFix,
    FixVar,
    VarFix,
    VarVar,
}

impl AspxIntClass {
    /// Read `aspx_int_class` from the bit-reader using the prefix code
    /// in Table 126. Consumes between 1 and 3 bits.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        if !br.read_bit()? {
            return Ok(Self::FixFix); // 0
        }
        if !br.read_bit()? {
            return Ok(Self::FixVar); // 10
        }
        if !br.read_bit()? {
            Ok(Self::VarFix) // 110
        } else {
            Ok(Self::VarVar) // 111
        }
    }

    /// Number of bits this class occupies on the wire.
    pub fn bits(self) -> u32 {
        match self {
            Self::FixFix => 1,
            Self::FixVar => 2,
            Self::VarFix | Self::VarVar => 3,
        }
    }
}

/// Parsed per-channel `aspx_framing()` output (ETSI TS 103 190-1
/// §4.2.12.4 Table 53 / §4.3.10.4).
///
/// This holds the framing elements for a single A-SPX channel: the
/// interval class, the signal- and noise-envelope counts, the freq-res
/// vector (empty when `aspx_freq_res_mode != Signalled`), and the raw
/// variable-border fields that feed later A-SPX stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspxFraming {
    /// `aspx_int_class[ch]`.
    pub int_class: AspxIntClass,
    /// `aspx_num_env[ch]` — number of signal envelopes, derived from
    /// `tmp_num_env` (FIXFIX) or from `num_rel_left + num_rel_right + 1`
    /// (otherwise). Bounded by Table 128: 4 for FIXFIX, 5 otherwise.
    pub num_env: u32,
    /// `aspx_num_noise[ch]` — 1 if `num_env == 1`, else 2
    /// (§4.3.10.4.11).
    pub num_noise: u32,
    /// `aspx_freq_res[ch][env]`. Populated only when
    /// `aspx_freq_res_mode == Signalled`; empty otherwise.
    pub freq_res: Vec<bool>,
    /// `aspx_var_bord_left[ch]` — present for VARFIX / VARVAR and only
    /// in an I-frame (per Table 53).
    pub var_bord_left: Option<u8>,
    /// `aspx_var_bord_right[ch]` — present for FIXVAR / VARVAR.
    pub var_bord_right: Option<u8>,
    /// `aspx_num_rel_left[ch]`. Zero for FIXFIX / FIXVAR.
    pub num_rel_left: u8,
    /// `aspx_num_rel_right[ch]`. Zero for FIXFIX / VARFIX.
    pub num_rel_right: u8,
    /// `aspx_rel_bord_left[ch][rel]`. Empty when `num_rel_left == 0`.
    pub rel_bord_left: Vec<u8>,
    /// `aspx_rel_bord_right[ch][rel]`. Empty when `num_rel_right == 0`.
    pub rel_bord_right: Vec<u8>,
    /// `aspx_tsg_ptr[ch]` — transient-pointer, present for every class
    /// except FIXFIX. `ptr_bits = ceil(log2(num_env + 2))`.
    pub tsg_ptr: Option<u8>,
}

/// Parse `aspx_framing(ch)` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 53 (§4.2.12.4).
///
/// * `cfg` — the previously parsed `aspx_config()`, which drives the
///   FIXFIX envelope-count field width and the freq-res signalling.
/// * `b_iframe` — from `ac4_substream_info()`; gates `aspx_var_bord_left`
///   for VARFIX / VARVAR.
/// * `num_aspx_timeslots_over_8` — `true` when `num_aspx_timeslots > 8`;
///   selects 2-bit vs 1-bit fields for `aspx_num_rel_*` and
///   `aspx_rel_bord_*` (Note 1 in Table 53).
pub fn parse_aspx_framing(
    br: &mut BitReader<'_>,
    cfg: &AspxConfig,
    b_iframe: bool,
    num_aspx_timeslots_over_8: bool,
) -> Result<AspxFraming> {
    // Width of the Note-1 fields: aspx_num_rel_*, aspx_rel_bord_*.
    let note1_bits: u32 = if num_aspx_timeslots_over_8 { 2 } else { 1 };

    let int_class = AspxIntClass::read(br)?;
    let mut num_rel_left: u8 = 0;
    let mut num_rel_right: u8 = 0;
    let mut var_bord_left: Option<u8> = None;
    let mut var_bord_right: Option<u8> = None;
    let mut rel_bord_left: Vec<u8> = Vec::new();
    let mut rel_bord_right: Vec<u8> = Vec::new();
    // Signal-envelope count. FIXFIX sets this directly from tmp_num_env;
    // the other classes compute it after the branch.
    let num_env: u32;
    // Freq-res vector. Signalled in-band only when the config opted in.
    let mut freq_res: Vec<bool> = Vec::new();

    match int_class {
        AspxIntClass::FixFix => {
            // envbits = aspx_num_env_bits_fixfix + 1
            let envbits = cfg.fixfix_tmp_num_env_bits();
            let tmp_num_env = br.read_u32(envbits)?;
            // aspx_num_env = 1 << tmp_num_env
            num_env = 1u32 << tmp_num_env;
            if cfg.signals_freq_res() {
                freq_res.push(br.read_bit()?);
            }
        }
        AspxIntClass::FixVar => {
            var_bord_right = Some(br.read_u32(2)? as u8);
            num_rel_right = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_right {
                rel_bord_right.push(br.read_u32(note1_bits)? as u8);
            }
            num_env = u32::from(num_rel_left) + u32::from(num_rel_right) + 1;
        }
        AspxIntClass::VarVar => {
            if b_iframe {
                var_bord_left = Some(br.read_u32(2)? as u8);
            }
            num_rel_left = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_left {
                rel_bord_left.push(br.read_u32(note1_bits)? as u8);
            }
            var_bord_right = Some(br.read_u32(2)? as u8);
            num_rel_right = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_right {
                rel_bord_right.push(br.read_u32(note1_bits)? as u8);
            }
            num_env = u32::from(num_rel_left) + u32::from(num_rel_right) + 1;
        }
        AspxIntClass::VarFix => {
            if b_iframe {
                var_bord_left = Some(br.read_u32(2)? as u8);
            }
            num_rel_left = br.read_u32(note1_bits)? as u8;
            for _ in 0..num_rel_left {
                rel_bord_left.push(br.read_u32(note1_bits)? as u8);
            }
            num_env = u32::from(num_rel_left) + u32::from(num_rel_right) + 1;
        }
    }

    let mut tsg_ptr: Option<u8> = None;
    if !matches!(int_class, AspxIntClass::FixFix) {
        // ptr_bits = ceil(log2(num_env + 2)). "log" here is a float log
        // without rounding, so result = bits needed to represent
        // num_env + 2 - 1 = num_env + 1 (for powers of two). We use
        // u32::next_power_of_two / leading_zeros to evaluate ceil_log2.
        let ptr_bits = ceil_log2(num_env + 2);
        tsg_ptr = Some(br.read_u32(ptr_bits)? as u8);
        if cfg.signals_freq_res() {
            freq_res.reserve(num_env as usize);
            for _ in 0..num_env {
                freq_res.push(br.read_bit()?);
            }
        }
    }

    let num_noise = if num_env > 1 { 2 } else { 1 };

    Ok(AspxFraming {
        int_class,
        num_env,
        num_noise,
        freq_res,
        var_bord_left,
        var_bord_right,
        num_rel_left,
        num_rel_right,
        rel_bord_left,
        rel_bord_right,
        tsg_ptr,
    })
}

/// Parsed `aspx_delta_dir(ch)` (ETSI TS 103 190-1 §4.2.12.5, Table 54).
///
/// Two bit-arrays gating how the matching `aspx_ec_data()` Huffman
/// codebook interprets its deltas:
///
/// * `aspx_sig_delta_dir[ch][env]` (length `aspx_num_env[ch]`): per
///   signal-envelope direction flag.
/// * `aspx_noise_delta_dir[ch][env]` (length `aspx_num_noise[ch]`):
///   per noise-envelope direction flag.
///
/// Convention (per §4.3.10.5 / Table 130): `false` means DF
/// (delta-frequency / F0 depending on position), `true` means DT
/// (delta-time).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxDeltaDir {
    pub sig_delta_dir: Vec<bool>,
    pub noise_delta_dir: Vec<bool>,
}

/// Parse `aspx_delta_dir(ch)` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 54 (§4.2.12.5). Reads
/// `aspx_num_env[ch] + aspx_num_noise[ch]` bits in total.
pub fn parse_aspx_delta_dir(br: &mut BitReader<'_>, framing: &AspxFraming) -> Result<AspxDeltaDir> {
    let mut sig = Vec::with_capacity(framing.num_env as usize);
    for _ in 0..framing.num_env {
        sig.push(br.read_bit()?);
    }
    let mut noise = Vec::with_capacity(framing.num_noise as usize);
    for _ in 0..framing.num_noise {
        noise.push(br.read_bit()?);
    }
    Ok(AspxDeltaDir {
        sig_delta_dir: sig,
        noise_delta_dir: noise,
    })
}

/// Parsed `aspx_hfgen_iwc_1ch()` (ETSI TS 103 190-1 §4.2.12.6,
/// Table 55) — the 1-channel HF-generation + interleaved-waveform
/// coding element carried after `aspx_delta_dir(0)` on the
/// `aspx_data_1ch()` path.
///
/// Field semantics (§4.3.10.6):
///
/// * `tna_mode[n]` (2 bits) — subband-wise tonal-to-noise adjustment
///   selector for each of `num_sbg_noise` noise subband groups.
/// * `add_harmonic[n]` (1 bit, optional) — per-highres-subband add-
///   harmonics flag; gated by `aspx_ah_present`.
/// * `fic_used_in_sfb[n]` (1 bit, optional) — per-highres-subband
///   frequency-interleaved-coding flag; gated by `aspx_fic_present`.
/// * `tic_used_in_slot[n]` (1 bit, optional) — per-A-SPX-timeslot
///   time-interleaved-coding flag; gated by `aspx_tic_present`.
///
/// When a gate bit is 0 the corresponding vector stays all-zero per
/// the pseudocode's explicit initialisation loops.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxHfgenIwc1Ch {
    pub tna_mode: Vec<u8>,
    pub ah_present: bool,
    pub add_harmonic: Vec<bool>,
    pub fic_present: bool,
    pub fic_used_in_sfb: Vec<bool>,
    pub tic_present: bool,
    pub tic_used_in_slot: Vec<bool>,
}

/// Parse `aspx_hfgen_iwc_1ch()` at the current bit-reader position per
/// ETSI TS 103 190-1 Table 55 (§4.2.12.6).
///
/// * `num_sbg_noise` — number of noise subband groups (§5.7.6.3.1.3).
/// * `num_sbg_sig_highres` — number of high-resolution signal subband
///   groups (§5.7.6.3.1.2).
/// * `num_aspx_timeslots` — A-SPX time slots per frame (§5.7.6.3.3.0,
///   Pseudocode 75a).
///
/// All three come from the A-SPX freq-scale derivation which is
/// driven by `aspx_config` + the frame's sample-rate family; this
/// parser takes them as ready-to-go counts so it stays decoupled from
/// that future derivation.
pub fn parse_aspx_hfgen_iwc_1ch(
    br: &mut BitReader<'_>,
    num_sbg_noise: u32,
    num_sbg_sig_highres: u32,
    num_aspx_timeslots: u32,
) -> Result<AspxHfgenIwc1Ch> {
    let mut tna_mode = Vec::with_capacity(num_sbg_noise as usize);
    for _ in 0..num_sbg_noise {
        tna_mode.push(br.read_u32(2)? as u8);
    }
    // aspx_ah_present + optional per-sbg add_harmonic flags.
    let ah_present = br.read_bit()?;
    let mut add_harmonic = vec![false; num_sbg_sig_highres as usize];
    if ah_present {
        for ah in add_harmonic.iter_mut() {
            *ah = br.read_bit()?;
        }
    }
    // aspx_fic_present + optional per-sbg fic_used_in_sfb flags.
    let fic_present = br.read_bit()?;
    let mut fic_used_in_sfb = vec![false; num_sbg_sig_highres as usize];
    if fic_present {
        for f in fic_used_in_sfb.iter_mut() {
            *f = br.read_bit()?;
        }
    }
    // aspx_tic_present + optional per-timeslot tic_used_in_slot flags.
    let tic_present = br.read_bit()?;
    let mut tic_used_in_slot = vec![false; num_aspx_timeslots as usize];
    if tic_present {
        for t in tic_used_in_slot.iter_mut() {
            *t = br.read_bit()?;
        }
    }
    Ok(AspxHfgenIwc1Ch {
        tna_mode,
        ah_present,
        add_harmonic,
        fic_present,
        fic_used_in_sfb,
        tic_present,
        tic_used_in_slot,
    })
}

/// Parsed `aspx_hfgen_iwc_2ch()` (ETSI TS 103 190-1 §4.2.12.7,
/// Table 56). Stereo variant of [`AspxHfgenIwc1Ch`] with the
/// additional per-channel gating introduced by the 2-channel tool
/// (Table 56):
///
/// * `tna_mode` is 2-dim `[ch][sbg]`. When `aspx_balance == 1` the
///   encoder signals only channel 0 and the decoder mirrors channel 0
///   into channel 1.
/// * `ah_left` / `ah_right` gate the per-channel `add_harmonic[ch][]`
///   independently.
/// * `fic_present` gates both channels; when present, `fic_left` and
///   `fic_right` gate each channel's `fic_used_in_sfb[ch][]` vector
///   independently.
/// * `tic_present` gates both channels; when present, `tic_copy`
///   first decides whether the right channel's pattern is copied
///   from the left. When `tic_copy == 0`, `tic_left` / `tic_right`
///   gate each channel's `tic_used_in_slot[ch][]` vector.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxHfgenIwc2Ch {
    pub tna_mode: [Vec<u8>; 2],
    pub ah_left: bool,
    pub ah_right: bool,
    pub add_harmonic: [Vec<bool>; 2],
    pub fic_present: bool,
    pub fic_left: bool,
    pub fic_right: bool,
    pub fic_used_in_sfb: [Vec<bool>; 2],
    pub tic_present: bool,
    pub tic_copy: bool,
    pub tic_left: bool,
    pub tic_right: bool,
    pub tic_used_in_slot: [Vec<bool>; 2],
}

/// Parse `aspx_hfgen_iwc_2ch(aspx_balance)` at the current bit-reader
/// position per ETSI TS 103 190-1 Table 56 (§4.2.12.7).
pub fn parse_aspx_hfgen_iwc_2ch(
    br: &mut BitReader<'_>,
    aspx_balance: bool,
    num_sbg_noise: u32,
    num_sbg_sig_highres: u32,
    num_aspx_timeslots: u32,
) -> Result<AspxHfgenIwc2Ch> {
    // tna_mode[0][..] always present.
    let mut tna0 = Vec::with_capacity(num_sbg_noise as usize);
    for _ in 0..num_sbg_noise {
        tna0.push(br.read_u32(2)? as u8);
    }
    // tna_mode[1][..] present only when aspx_balance == 0; otherwise
    // mirrors channel 0.
    let tna1 = if !aspx_balance {
        let mut v = Vec::with_capacity(num_sbg_noise as usize);
        for _ in 0..num_sbg_noise {
            v.push(br.read_u32(2)? as u8);
        }
        v
    } else {
        tna0.clone()
    };
    // Per-channel add-harmonic gates and vectors.
    let ah_left = br.read_bit()?;
    let mut ah0 = vec![false; num_sbg_sig_highres as usize];
    if ah_left {
        for a in ah0.iter_mut() {
            *a = br.read_bit()?;
        }
    }
    let ah_right = br.read_bit()?;
    let mut ah1 = vec![false; num_sbg_sig_highres as usize];
    if ah_right {
        for a in ah1.iter_mut() {
            *a = br.read_bit()?;
        }
    }
    // Frequency-interleaved-coding — outer `fic_present` gate, then
    // per-channel `fic_left` / `fic_right` gates.
    let mut fic0 = vec![false; num_sbg_sig_highres as usize];
    let mut fic1 = vec![false; num_sbg_sig_highres as usize];
    let fic_present = br.read_bit()?;
    let mut fic_left = false;
    let mut fic_right = false;
    if fic_present {
        fic_left = br.read_bit()?;
        if fic_left {
            for f in fic0.iter_mut() {
                *f = br.read_bit()?;
            }
        }
        fic_right = br.read_bit()?;
        if fic_right {
            for f in fic1.iter_mut() {
                *f = br.read_bit()?;
            }
        }
    }
    // Time-interleaved-coding — outer `tic_present` gate, then
    // `tic_copy` (mirror L into R) or per-channel gates.
    let mut tic0 = vec![false; num_aspx_timeslots as usize];
    let mut tic1 = vec![false; num_aspx_timeslots as usize];
    let mut tic_copy = false;
    let mut tic_left = false;
    let mut tic_right = false;
    let tic_present = br.read_bit()?;
    if tic_present {
        tic_copy = br.read_bit()?;
        if !tic_copy {
            tic_left = br.read_bit()?;
            tic_right = br.read_bit()?;
        }
        if tic_copy || tic_left {
            for t in tic0.iter_mut() {
                *t = br.read_bit()?;
            }
        }
        if tic_right {
            for t in tic1.iter_mut() {
                *t = br.read_bit()?;
            }
        }
        if tic_copy {
            tic1 = tic0.clone();
        }
    }
    Ok(AspxHfgenIwc2Ch {
        tna_mode: [tna0, tna1],
        ah_left,
        ah_right,
        add_harmonic: [ah0, ah1],
        fic_present,
        fic_left,
        fic_right,
        fic_used_in_sfb: [fic0, fic1],
        tic_present,
        tic_copy,
        tic_left,
        tic_right,
        tic_used_in_slot: [tic0, tic1],
    })
}

/// Returns `ceil(log2(n))` for `n >= 1`. For `n == 1` returns 0, i.e.
/// "zero bits needed". Used to size `aspx_tsg_ptr` (`ptr_bits`).
fn ceil_log2(n: u32) -> u32 {
    debug_assert!(n >= 1);
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Return `num_qmf_timeslots = frame_length / num_qmf_subbands` per
/// ETSI TS 103 190-1 §5.7.3.2 (Table 189). `num_qmf_subbands` is fixed
/// at 64 for AC-4. Valid inputs are the eight base-rate
/// `frame_length` values: 2048 / 1920 / 1536 / 1024 / 960 / 768 / 512
/// / 384 — for which Table 189 gives 32 / 30 / 24 / 16 / 15 / 12 / 8 /
/// 6. We compute the quotient directly (which matches Table 189 for
/// any of those inputs) rather than hard-coding the table.
pub fn num_qmf_timeslots(frame_len_base: u32) -> u32 {
    frame_len_base / 64
}

/// Return `num_ts_in_ats` — how many QMF time slots make up one A-SPX
/// time slot — per ETSI TS 103 190-1 §5.7.6.3.3.0 (Table 192). Two for
/// `frame_length >= 1536`, one for shorter frames.
pub fn num_ts_in_ats(frame_len_base: u32) -> u32 {
    if frame_len_base >= 1536 {
        2
    } else {
        1
    }
}

/// Return `num_aspx_timeslots = num_qmf_timeslots / num_ts_in_ats` per
/// ETSI TS 103 190-1 §5.7.6.3.3.0 Pseudocode 75a. Valid only for the
/// eight base-rate `frame_length` values in Table 189.
pub fn num_aspx_timeslots(frame_len_base: u32) -> u32 {
    num_qmf_timeslots(frame_len_base) / num_ts_in_ats(frame_len_base)
}

/// Data-type selector for `aspx_ec_data()` (ETSI TS 103 190-1 Table 57).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxDataType {
    Signal,
    Noise,
}

/// Stereo-mode selector for `aspx_ec_data()` (ETSI TS 103 190-1 Table 57).
/// Maps to `LEVEL` / `BALANCE` in `get_aspx_hcb()` (Pseudocode 79).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxStereoMode {
    Level,
    Balance,
}

/// `get_aspx_hcb()` per ETSI TS 103 190-1 §5.7.6.3.4 (Pseudocode 79) —
/// pick the correct Annex A.2 Huffman codebook from the four-tuple
/// `(data_type, quant_mode, stereo_mode, hcb_type)`.
///
/// * `data_type` — SIGNAL or NOISE.
/// * `quant_mode` — [`AspxQuantStep::Fine`] (15, 1.5 dB) or
///   [`AspxQuantStep::Coarse`] (30, 3 dB). Note: NOISE tables don't
///   carry a quant-mode dimension in Annex A.2; the caller passes `qm = 0`
///   per Tables 51 / 52, so for NOISE we ignore the quant step.
/// * `stereo_mode` — LEVEL or BALANCE.
/// * `hcb_type` — `F0`, `DF`, or `DT` (selected by the encoder's
///   choice of base-frequency / delta-frequency / delta-time coding).
pub fn get_aspx_hcb(
    data_type: AspxDataType,
    quant_mode: AspxQuantStep,
    stereo_mode: AspxStereoMode,
    hcb_type: AspxHcbType,
) -> HuffmanCodebookId {
    use AspxDataType::*;
    use AspxHcbType::*;
    use AspxQuantStep::*;
    use AspxStereoMode::*;
    match (data_type, quant_mode, stereo_mode, hcb_type) {
        (Signal, Fine, Level, F0) => HuffmanCodebookId::EnvLevel15F0,
        (Signal, Fine, Level, Df) => HuffmanCodebookId::EnvLevel15Df,
        (Signal, Fine, Level, Dt) => HuffmanCodebookId::EnvLevel15Dt,
        (Signal, Fine, Balance, F0) => HuffmanCodebookId::EnvBalance15F0,
        (Signal, Fine, Balance, Df) => HuffmanCodebookId::EnvBalance15Df,
        (Signal, Fine, Balance, Dt) => HuffmanCodebookId::EnvBalance15Dt,
        (Signal, Coarse, Level, F0) => HuffmanCodebookId::EnvLevel30F0,
        (Signal, Coarse, Level, Df) => HuffmanCodebookId::EnvLevel30Df,
        (Signal, Coarse, Level, Dt) => HuffmanCodebookId::EnvLevel30Dt,
        (Signal, Coarse, Balance, F0) => HuffmanCodebookId::EnvBalance30F0,
        (Signal, Coarse, Balance, Df) => HuffmanCodebookId::EnvBalance30Df,
        (Signal, Coarse, Balance, Dt) => HuffmanCodebookId::EnvBalance30Dt,
        (Noise, _, Level, F0) => HuffmanCodebookId::NoiseLevelF0,
        (Noise, _, Level, Df) => HuffmanCodebookId::NoiseLevelDf,
        (Noise, _, Level, Dt) => HuffmanCodebookId::NoiseLevelDt,
        (Noise, _, Balance, F0) => HuffmanCodebookId::NoiseBalanceF0,
        (Noise, _, Balance, Df) => HuffmanCodebookId::NoiseBalanceDf,
        (Noise, _, Balance, Dt) => HuffmanCodebookId::NoiseBalanceDt,
    }
}

/// Codebook flavour per Annex A.2 — the last suffix in the codebook
/// name (`F0` / `DF` / `DT`). Driven by whether the symbol is an
/// envelope's base-frequency value, a delta along frequency, or a
/// delta along time (§4.2.12.9 Table 58).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspxHcbType {
    F0,
    Df,
    Dt,
}

/// One envelope's worth of Huffman-decoded A-SPX data — the output
/// of `aspx_huff_data()` (ETSI TS 103 190-1 Table 58). Holds the
/// per-subband-group quantised delta values in the order they were
/// read from the stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AspxHuffEnv {
    /// Per-subband-group decoded delta values. For `FREQ` direction
    /// the first entry is an F0 value and the rest are DF deltas;
    /// for `TIME` direction every entry is a DT delta.
    pub values: Vec<i32>,
    /// Transmission direction this envelope used (false = FREQ,
    /// true = TIME) — tracks `aspx_sig_delta_dir[env]` /
    /// `aspx_noise_delta_dir[env]`.
    pub direction_time: bool,
}

/// `aspx_huff_data()` per ETSI TS 103 190-1 §4.2.12.9 (Table 58) —
/// decode one envelope's `num_sbg` Huffman codewords.
///
/// The `direction` flag selects between the two branches of the
/// syntax:
///
/// * `direction == false` (FREQ): the first value is read with the
///   matching `*_F0` codebook, the remaining `num_sbg - 1` with the
///   `*_DF` codebook.
/// * `direction == true` (TIME): every value is read with the `*_DT`
///   codebook.
///
/// `data_type`, `quant_mode`, `stereo_mode` drive `get_aspx_hcb()`
/// selection.
pub fn parse_aspx_huff_data(
    br: &mut BitReader<'_>,
    data_type: AspxDataType,
    num_sbg: u32,
    quant_mode: AspxQuantStep,
    stereo_mode: AspxStereoMode,
    direction: bool,
) -> Result<AspxHuffEnv> {
    let mut out = Vec::with_capacity(num_sbg as usize);
    if !direction {
        // FREQ — F0 for index 0, DF for the rest.
        let hcb_f0 = lookup_aspx_hcb(get_aspx_hcb(
            data_type,
            quant_mode,
            stereo_mode,
            AspxHcbType::F0,
        ));
        if num_sbg >= 1 {
            out.push(hcb_f0.decode_delta(br)?);
        }
        if num_sbg >= 2 {
            let hcb_df = lookup_aspx_hcb(get_aspx_hcb(
                data_type,
                quant_mode,
                stereo_mode,
                AspxHcbType::Df,
            ));
            for _ in 1..num_sbg {
                out.push(hcb_df.decode_delta(br)?);
            }
        }
    } else {
        // TIME — all-DT.
        let hcb_dt = lookup_aspx_hcb(get_aspx_hcb(
            data_type,
            quant_mode,
            stereo_mode,
            AspxHcbType::Dt,
        ));
        for _ in 0..num_sbg {
            out.push(hcb_dt.decode_delta(br)?);
        }
    }
    Ok(AspxHuffEnv {
        values: out,
        direction_time: direction,
    })
}

/// A-SPX subband-group count context for `aspx_ec_data()` — the three
/// `num_sbg_sig_highres` / `num_sbg_sig_lowres` / `num_sbg_noise`
/// counts derived in §5.7.6.3 from the master frequency scale.
///
/// The spec's Note on Table 57 reads:
///
/// > Variables num_sbg_sig_highres and num_sbg_sig_lowres are derived
/// > in clause 5.7.6.3.1.2 and num_sbg_noise is derived according to
/// > clause 5.7.6.3.1.3.
///
/// Since we don't evaluate the full master-freq-scale derivation yet,
/// the caller passes the derived counts in explicitly. For testing /
/// hand-built fixtures this is straightforward; for real AC-4
/// bitstreams this needs the QMF-subband layout wiring in a later
/// round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AspxSbgCounts {
    pub num_sbg_sig_highres: u32,
    pub num_sbg_sig_lowres: u32,
    pub num_sbg_noise: u32,
}

/// `aspx_ec_data()` per ETSI TS 103 190-1 §4.2.12.8 (Table 57) —
/// decode `num_env` envelopes' worth of Huffman codewords for either
/// SIGNAL or NOISE data.
///
/// Per envelope:
///
/// * SIGNAL: `num_sbg = num_sbg_sig_highres` when `freq_res[env] == 1`,
///   else `num_sbg_sig_lowres`.
/// * NOISE: `num_sbg = num_sbg_noise` regardless of `freq_res` (the
///   caller passes `freq_res = &[]` for NOISE paths — see the
///   `aspx_data_1ch()` / `aspx_data_2ch()` call sites in Tables 51
///   and 52 where `freq_res` is literally `0`).
///
/// Each envelope's direction is taken from `direction[env]`
/// (`aspx_sig_delta_dir` / `aspx_noise_delta_dir` from Table 54).
/// Returns a per-envelope vector of Huffman-decoded symbol streams.
pub fn parse_aspx_ec_data(
    br: &mut BitReader<'_>,
    data_type: AspxDataType,
    num_env: u32,
    freq_res: &[bool],
    quant_mode: AspxQuantStep,
    stereo_mode: AspxStereoMode,
    direction: &[bool],
    sbg: AspxSbgCounts,
) -> Result<Vec<AspxHuffEnv>> {
    if direction.len() < num_env as usize {
        return Err(Error::invalid(
            "ac4: aspx_ec_data direction vector shorter than num_env",
        ));
    }
    let mut out = Vec::with_capacity(num_env as usize);
    for env in 0..num_env as usize {
        let num_sbg = match data_type {
            AspxDataType::Signal => {
                // freq_res may be empty when the caller doesn't have
                // per-envelope resolution data — fall back to the
                // high-res count (Table 57 Note / §4.3.10.4.9).
                let use_highres = freq_res.get(env).copied().unwrap_or(true);
                if use_highres {
                    sbg.num_sbg_sig_highres
                } else {
                    sbg.num_sbg_sig_lowres
                }
            }
            AspxDataType::Noise => sbg.num_sbg_noise,
        };
        let envdata = parse_aspx_huff_data(
            br,
            data_type,
            num_sbg,
            quant_mode,
            stereo_mode,
            direction[env],
        )?;
        out.push(envdata);
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// §5.7.6.3.1 Subband-group derivation
// ---------------------------------------------------------------------

/// Static high-resolution subband-group template, `sbg_template_highres`
/// (ETSI TS 103 190-1 §5.7.6.3.1.1). 23 entries.
pub const ASPX_SBG_TEMPLATE_HIGHRES: [u32; 23] = [
    18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47, 50, 53, 56, 59, 62,
];

/// Static low-resolution subband-group template, `sbg_template_lowres`
/// (ETSI TS 103 190-1 §5.7.6.3.1.1). 21 entries.
pub const ASPX_SBG_TEMPLATE_LOWRES: [u32; 21] = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 35, 38, 42, 46,
];

/// Derived A-SPX frequency tables from §5.7.6.3.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspxFrequencyTables {
    /// Master subband-group table (Pseudocode 67). Length is
    /// `num_sbg_master + 1` (borders).
    pub sbg_master: Vec<u32>,
    /// Number of master subband groups.
    pub num_sbg_master: u32,
    /// Lower border of the first subband group = `sbg_master[0]`.
    pub sba: u32,
    /// Upper border of the last subband group = `sbg_master[num_sbg_master]`.
    pub sbz: u32,
    /// High-resolution signal-envelope subband-group table
    /// (Pseudocode 68).
    pub sbg_sig_highres: Vec<u32>,
    /// Cross-over subband = `sbg_sig_highres[0]`.
    pub sbx: u32,
    /// Number of subbands spanned by the A-SPX tool (`sbz - sbx`).
    pub num_sb_aspx: u32,
    /// Low-resolution signal-envelope subband-group table
    /// (Pseudocode 69).
    pub sbg_sig_lowres: Vec<u32>,
    /// Noise-envelope subband-group table (Pseudocode 70).
    pub sbg_noise: Vec<u32>,
    /// Aggregate counts suitable for passing into
    /// [`parse_aspx_ec_data`] / [`parse_aspx_hfgen_iwc_1ch`] /
    /// [`parse_aspx_hfgen_iwc_2ch`].
    pub counts: AspxSbgCounts,
}

/// Derive the A-SPX master subband-group table (Pseudocode 67) from an
/// `aspx_config`.
///
/// Returns `(sbg_master, num_sbg_master, sba, sbz)`.
pub fn derive_master_sbg_table(cfg: &AspxConfig) -> (Vec<u32>, u32, u32, u32) {
    let start = cfg.start_freq as u32;
    let stop = cfg.stop_freq as u32;
    let (template, base_count): (&[u32], u32) = match cfg.master_freq_scale {
        AspxMasterFreqScale::HighRes => (&ASPX_SBG_TEMPLATE_HIGHRES, 22),
        AspxMasterFreqScale::LowRes => (&ASPX_SBG_TEMPLATE_LOWRES, 20),
    };
    let num_sbg_master = base_count - 2 * start - 2 * stop;
    let mut sbg_master = Vec::with_capacity((num_sbg_master + 1) as usize);
    for sbg in 0..=num_sbg_master {
        sbg_master.push(template[(2 * start + sbg) as usize]);
    }
    let sba = sbg_master[0];
    let sbz = sbg_master[num_sbg_master as usize];
    (sbg_master, num_sbg_master, sba, sbz)
}

/// Result bundle for [`derive_sig_sbg_tables`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspxSigSbgTables {
    pub sbg_sig_highres: Vec<u32>,
    pub sbg_sig_lowres: Vec<u32>,
    pub sbx: u32,
    pub num_sb_aspx: u32,
    pub num_sbg_sig_highres: u32,
    pub num_sbg_sig_lowres: u32,
}

/// Derive the high/low-resolution signal-envelope subband-group tables
/// (§5.7.6.3.1.2, Pseudocodes 68 and 69) from the master table.
pub fn derive_sig_sbg_tables(
    sbg_master: &[u32],
    num_sbg_master: u32,
    xover_offset: u32,
) -> Result<AspxSigSbgTables> {
    if xover_offset > num_sbg_master {
        return Err(Error::invalid(
            "ac4: aspx_xover_subband_offset exceeds num_sbg_master",
        ));
    }
    let num_sbg_sig_highres = num_sbg_master - xover_offset;
    let mut sbg_sig_highres = Vec::with_capacity((num_sbg_sig_highres + 1) as usize);
    for sbg in 0..=num_sbg_sig_highres {
        sbg_sig_highres.push(sbg_master[(sbg + xover_offset) as usize]);
    }
    let sbx = sbg_sig_highres[0];
    let num_sb_aspx = sbg_sig_highres[num_sbg_sig_highres as usize].saturating_sub(sbx);

    // Pseudocode 69: low-res table is a decimation of the high-res table
    // by 2. Even branch keeps 0, 2, 4, ...; odd branch keeps 0, 1, 3, 5, ...
    let num_sbg_sig_lowres = num_sbg_sig_highres - num_sbg_sig_highres / 2;
    let mut sbg_sig_lowres = Vec::with_capacity((num_sbg_sig_lowres + 1) as usize);
    sbg_sig_lowres.push(sbg_sig_highres[0]);
    if num_sbg_sig_highres % 2 == 0 {
        for sbg in 1..=num_sbg_sig_lowres {
            sbg_sig_lowres.push(sbg_sig_highres[(2 * sbg) as usize]);
        }
    } else {
        for sbg in 1..=num_sbg_sig_lowres {
            sbg_sig_lowres.push(sbg_sig_highres[(2 * sbg - 1) as usize]);
        }
    }
    Ok(AspxSigSbgTables {
        sbg_sig_highres,
        sbg_sig_lowres,
        sbx,
        num_sb_aspx,
        num_sbg_sig_highres,
        num_sbg_sig_lowres,
    })
}

/// Derive the noise-envelope subband-group table (§5.7.6.3.1.3,
/// Pseudocode 70).
///
/// `aspx_noise_sbg` here is the **raw** 2-bit field from
/// [`AspxConfig`].
pub fn derive_noise_sbg_table(
    aspx_noise_sbg: u32,
    sbz: u32,
    sbx: u32,
    sbg_sig_lowres: &[u32],
    num_sbg_sig_lowres: u32,
) -> Result<Vec<u32>> {
    if sbx == 0 {
        return Err(Error::invalid(
            "ac4: sbx must be > 0 for noise sbg derivation",
        ));
    }
    if sbz <= sbx {
        return Err(Error::invalid(
            "ac4: sbz must exceed sbx for noise sbg derivation",
        ));
    }
    let ratio = (sbz as f64) / (sbx as f64);
    let log2_ratio = ratio.log2();
    let raw = (aspx_noise_sbg as f64) * log2_ratio + 0.5;
    let mut num_sbg_noise = raw.floor().max(1.0) as u32;
    if num_sbg_noise > 5 {
        num_sbg_noise = 5;
    }
    if num_sbg_noise > num_sbg_sig_lowres {
        num_sbg_noise = num_sbg_sig_lowres.max(1);
    }
    let mut idx = vec![0u32; (num_sbg_noise + 1) as usize];
    let mut sbg_noise = Vec::with_capacity((num_sbg_noise + 1) as usize);
    sbg_noise.push(sbg_sig_lowres[0]);
    for sbg in 1..=num_sbg_noise {
        idx[sbg as usize] = idx[(sbg - 1) as usize];
        let remaining = num_sbg_sig_lowres - idx[(sbg - 1) as usize];
        let divisor = num_sbg_noise + 1 - sbg;
        idx[sbg as usize] += remaining / divisor;
        sbg_noise.push(sbg_sig_lowres[idx[sbg as usize] as usize]);
    }
    Ok(sbg_noise)
}

/// Derive the full set of A-SPX frequency tables (master, high-res
/// signal, low-res signal, noise) per §5.7.6.3.1 from `aspx_config()`
/// plus the per-frame `aspx_xover_subband_offset`.
pub fn derive_aspx_frequency_tables(
    cfg: &AspxConfig,
    xover_offset: u32,
) -> Result<AspxFrequencyTables> {
    let (sbg_master, num_sbg_master, sba, sbz) = derive_master_sbg_table(cfg);
    let sig = derive_sig_sbg_tables(&sbg_master, num_sbg_master, xover_offset)?;
    let AspxSigSbgTables {
        sbg_sig_highres,
        sbg_sig_lowres,
        sbx,
        num_sb_aspx,
        num_sbg_sig_highres,
        num_sbg_sig_lowres,
    } = sig;
    let sbg_noise = derive_noise_sbg_table(
        cfg.noise_sbg as u32,
        sbz,
        sbx,
        &sbg_sig_lowres,
        num_sbg_sig_lowres,
    )?;
    let num_sbg_noise = (sbg_noise.len() as u32).saturating_sub(1);
    Ok(AspxFrequencyTables {
        sbg_master,
        num_sbg_master,
        sba,
        sbz,
        sbg_sig_highres,
        sbx,
        num_sb_aspx,
        sbg_sig_lowres,
        sbg_noise,
        counts: AspxSbgCounts {
            num_sbg_sig_highres,
            num_sbg_sig_lowres,
            num_sbg_noise,
        },
    })
}

// ---------------------------------------------------------------------
// §5.7.6.3.1.4 Patch subband group table (Pseudocode 71)
// ---------------------------------------------------------------------

/// Result of patch subband-group derivation (§5.7.6.3.1.4,
/// Pseudocode 71): the `sbg_patches` border table plus the per-patch
/// metadata arrays used by HF signal creation (§5.7.6.4.1.4,
/// Pseudocode 89).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AspxPatchTables {
    /// `sbg_patches[]` — QMF subband border table for the patches.
    /// Length = `num_sbg_patches + 1`. Starts at `sbx`.
    pub sbg_patches: Vec<u32>,
    /// Number of patches (≤ 5 per the spec).
    pub num_sbg_patches: u32,
    /// `sbg_patch_num_sb[i]` — number of QMF subbands in patch `i`.
    pub sbg_patch_num_sb: Vec<u32>,
    /// `sbg_patch_start_sb[i]` — QMF subband in the low band that is
    /// the source start for patch `i`'s tile copy.
    pub sbg_patch_start_sb: Vec<u32>,
}

/// Derive the patch subband-group tables from the master table plus
/// the A-SPX crossover/upper borders. Implements ETSI TS 103 190-1
/// §5.7.6.3.1.4 Pseudocode 71.
///
/// `base_samp_freq_is_48` is `true` for the 48 kHz family, `false`
/// for the 44.1 kHz family. `master_freq_scale_highres` is `true`
/// when `aspx_master_freq_scale` selected the high-resolution
/// template.
pub fn derive_patch_tables(
    sbg_master: &[u32],
    num_sbg_master: u32,
    sba: u32,
    sbx: u32,
    num_sb_aspx: u32,
    base_samp_freq_is_48: bool,
    master_freq_scale_highres: bool,
) -> AspxPatchTables {
    let goal_sb: u32 = if base_samp_freq_is_48 { 43 } else { 46 };
    let source_band_low: u32 = if master_freq_scale_highres { 4 } else { 2 };
    let mut sbg = if goal_sb < sbx + num_sb_aspx {
        // Find the smallest i such that sbg_master[i] >= goal_sb.
        let mut s = 0u32;
        for i in 0..sbg_master.len() {
            if sbg_master[i] < goal_sb {
                s = (i + 1) as u32;
            } else {
                break;
            }
        }
        s
    } else {
        num_sbg_master
    };
    let mut msb = sba;
    let mut usb = sbx;
    let mut sbg_patch_num_sb: Vec<u32> = Vec::new();
    let mut sbg_patch_start_sb: Vec<u32> = Vec::new();
    // Target stopping condition: sb == (sbx + num_sb_aspx).
    let target = sbx + num_sb_aspx;
    // Safety guard: bound the outer do-while loop to avoid runaway.
    for _ in 0..32 {
        // Inner while loop searches j downward.
        let mut j = sbg as usize;
        if j >= sbg_master.len() {
            break;
        }
        let mut sb = sbg_master[j];
        // `odd = (sb - 2 + sba) % 2` — spec uses signed subtraction but
        // sb >= 2 in practice.
        // Inner while loop: sb > (sba - source_band_low + msb - odd)
        // where odd is recomputed each iteration.
        loop {
            let odd = ((sb as i64 - 2 + sba as i64).rem_euclid(2)) as i64;
            let rhs = sba as i64 - source_band_low as i64 + msb as i64 - odd;
            if !((sb as i64) > rhs) {
                break;
            }
            if j == 0 {
                break;
            }
            j -= 1;
            sb = sbg_master[j];
        }
        let num_sb = sb.saturating_sub(usb);
        let odd_final = ((sb as i64 - 2 + sba as i64).rem_euclid(2)) as u32;
        let patch_start = sba.saturating_sub(odd_final).saturating_sub(num_sb);
        if num_sb > 0 {
            sbg_patch_num_sb.push(num_sb);
            sbg_patch_start_sb.push(patch_start);
            usb = sb;
            msb = sb;
        } else {
            msb = sbx;
        }
        if (sbg as usize) < sbg_master.len() && sbg_master[sbg as usize].saturating_sub(sb) < 3 {
            sbg = num_sbg_master;
        }
        if sb == target {
            break;
        }
    }
    let mut num_sbg_patches = sbg_patch_num_sb.len() as u32;
    // Tail trim: if last patch has <3 subbands and there are multiple
    // patches, drop it.
    if num_sbg_patches > 1 && *sbg_patch_num_sb.last().unwrap_or(&0) < 3 {
        sbg_patch_num_sb.pop();
        sbg_patch_start_sb.pop();
        num_sbg_patches -= 1;
    }
    // Build sbg_patches[] borders starting at sbx.
    let mut sbg_patches = Vec::with_capacity((num_sbg_patches + 1) as usize);
    sbg_patches.push(sbx);
    for i in 0..num_sbg_patches {
        let next = sbg_patches[i as usize] + sbg_patch_num_sb[i as usize];
        sbg_patches.push(next);
    }
    AspxPatchTables {
        sbg_patches,
        num_sbg_patches,
        sbg_patch_num_sb,
        sbg_patch_start_sb,
    }
}

// ---------------------------------------------------------------------
// §5.7.6.4.1.4 HF signal creation — simplified tile-copy path
// ---------------------------------------------------------------------

/// Simplified HF signal creation: copy low-band QMF samples into the
/// A-SPX range via the patch table.
///
/// This skips the full Pseudocode 89 chirp / alpha0 / alpha1 tonal
/// adjustment (which is gated by `aspx_preflat` and complex-valued
/// linear prediction over a covariance matrix from §5.7.6.4.1.2) and
/// instead lays down a clean tile copy — enough to produce
/// high-frequency content in the output PCM when the rest of the A-SPX
/// pipeline isn't yet producing valid alpha0/alpha1 coefficients.
///
/// `q_low[sb]` is a time-series of complex QMF samples for subband
/// `sb` (0..64). For `sb < sbx`, `q_low[sb]` has the analysis output
/// of the low band. For `sb >= sbx`, `q_low[sb]` is zero on entry.
/// On return, `q_high[sb][ts] = q_low[patch_src][ts]` where
/// `patch_src = sbg_patch_start_sb[i] + (sb_high - sbx - sum_prev)`.
pub fn hf_tile_copy(
    q_low: &[Vec<(f32, f32)>],
    patches: &AspxPatchTables,
    sbx: u32,
    num_qmf_subbands: u32,
) -> Vec<Vec<(f32, f32)>> {
    // Determine number of timeslots from any populated column of q_low.
    let n_ts = q_low.iter().map(|row| row.len()).max().unwrap_or(0);
    let mut q_high: Vec<Vec<(f32, f32)>> = (0..num_qmf_subbands)
        .map(|_| vec![(0.0f32, 0.0f32); n_ts])
        .collect();
    let mut sum_sb_patches: u32 = 0;
    for i in 0..patches.num_sbg_patches as usize {
        let n = patches.sbg_patch_num_sb[i];
        let start = patches.sbg_patch_start_sb[i];
        for sb_off in 0..n {
            let sb_high = sbx + sum_sb_patches + sb_off;
            let sb_src = start + sb_off;
            if sb_high >= num_qmf_subbands || (sb_src as usize) >= q_low.len() {
                continue;
            }
            // Copy the time series.
            let src = &q_low[sb_src as usize];
            let dst = &mut q_high[sb_high as usize];
            for ts in 0..n_ts.min(src.len()) {
                dst[ts] = src[ts];
            }
        }
        sum_sb_patches += n;
    }
    q_high
}

/// Apply a flat envelope gain to the A-SPX range of a QMF time-frequency
/// matrix in place. `q[sb][ts]` is scaled by `gain` for
/// `sbx <= sb < sbz`.
///
/// This is a simplified stand-in for the full §5.7.6.4.2 HF envelope
/// adjustment tool (per-envelope / per-subband-group gains from
/// Pseudocode 91). Used today as a scaffold so the HF range of the
/// QMF matrix carries non-zero, reasonable-amplitude data to drive
/// the QMF synthesis filter-bank.
pub fn apply_flat_envelope_gain(q: &mut [Vec<(f32, f32)>], sbx: u32, sbz: u32, gain: f32) {
    for sb in sbx..sbz {
        if (sb as usize) >= q.len() {
            break;
        }
        for sample in q[sb as usize].iter_mut() {
            sample.0 *= gain;
            sample.1 *= gain;
        }
    }
}

// ---------------------------------------------------------------------
// §5.7.6.3.3.1 FIXFIX atsg_sig / atsg_noise borders (Table 194)
// ---------------------------------------------------------------------

/// `tab_border[num_aspx_timeslots][num_atsg]` — ETSI TS 103 190-1
/// Table 194. Returns the A-SPX time-slot-group border vector for an
/// `aspx_int_class == FIXFIX` interval. Returned vector has length
/// `num_atsg + 1` and its last entry equals `num_aspx_timeslots`.
///
/// Returns `None` when the `(num_aspx_timeslots, num_atsg)` pair is not
/// one of the 15 combinations listed in Table 194.
pub fn tab_border_fixfix(num_aspx_timeslots: u32, num_atsg: u32) -> Option<Vec<u32>> {
    match (num_aspx_timeslots, num_atsg) {
        (6, 1) => Some(vec![0, 6]),
        (6, 2) => Some(vec![0, 3, 6]),
        (6, 4) => Some(vec![0, 2, 3, 4, 6]),
        (8, 1) => Some(vec![0, 8]),
        (8, 2) => Some(vec![0, 4, 8]),
        (8, 4) => Some(vec![0, 2, 4, 6, 8]),
        (12, 1) => Some(vec![0, 12]),
        (12, 2) => Some(vec![0, 6, 12]),
        (12, 4) => Some(vec![0, 3, 6, 9, 12]),
        (15, 1) => Some(vec![0, 15]),
        (15, 2) => Some(vec![0, 8, 15]),
        (15, 4) => Some(vec![0, 4, 8, 12, 15]),
        (16, 1) => Some(vec![0, 16]),
        (16, 2) => Some(vec![0, 8, 16]),
        (16, 4) => Some(vec![0, 4, 8, 12, 16]),
        _ => None,
    }
}

/// Derive `atsg_sig` / `atsg_noise` per ETSI TS 103 190-1 §5.7.6.3.3.1
/// (Pseudocode 76) for an `aspx_int_class == FIXFIX` interval. Returns
/// `None` if the (num_aspx_timeslots, num_env) or (num_aspx_timeslots,
/// num_noise) pair is not in Table 194.
pub fn derive_fixfix_atsg(
    num_aspx_timeslots: u32,
    num_env: u32,
    num_noise: u32,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let sig = tab_border_fixfix(num_aspx_timeslots, num_env)?;
    let noise = tab_border_fixfix(num_aspx_timeslots, num_noise)?;
    Some((sig, noise))
}

// ---------------------------------------------------------------------
// §5.7.6.3.4 Decoding A-SPX signal / noise envelopes
// Pseudocodes 80 / 81 delta-decode, Pseudocodes 82 / 83 dequantize.
// ---------------------------------------------------------------------

/// Delta-decode the signal envelope scale factors per ETSI TS 103 190-1
/// §5.7.6.3.4 Pseudocode 80.
pub fn delta_decode_sig(
    deltas: &[AspxHuffEnv],
    num_sbg: u32,
    qscf_prev_last: &[i32],
    delta: i32,
) -> Vec<Vec<i32>> {
    let num_env = deltas.len();
    let mut qscf: Vec<Vec<i32>> = vec![vec![0_i32; num_env]; num_sbg as usize];
    for (atsg, env) in deltas.iter().enumerate() {
        if env.direction_time {
            for sbg in 0..(num_sbg as usize) {
                let prev = if atsg == 0 {
                    qscf_prev_last.get(sbg).copied().unwrap_or(0)
                } else {
                    qscf[sbg][atsg - 1]
                };
                let d = env.values.get(sbg).copied().unwrap_or(0);
                qscf[sbg][atsg] = prev + delta * d;
            }
        } else {
            let mut acc: i32 = 0;
            for sbg in 0..(num_sbg as usize) {
                let d = env.values.get(sbg).copied().unwrap_or(0);
                acc += delta * d;
                qscf[sbg][atsg] = acc;
            }
        }
    }
    qscf
}

/// Delta-decode the noise envelope scale factors per ETSI TS 103 190-1
/// §5.7.6.3.4 Pseudocode 81. Same shape / semantics as
/// [`delta_decode_sig`].
pub fn delta_decode_noise(
    deltas: &[AspxHuffEnv],
    num_sbg: u32,
    qscf_prev_last: &[i32],
    delta: i32,
) -> Vec<Vec<i32>> {
    let num_env = deltas.len();
    let mut qscf: Vec<Vec<i32>> = vec![vec![0_i32; num_env]; num_sbg as usize];
    for (atsg, env) in deltas.iter().enumerate() {
        if env.direction_time {
            for sbg in 0..(num_sbg as usize) {
                let prev = if atsg == 0 {
                    qscf_prev_last.get(sbg).copied().unwrap_or(0)
                } else {
                    qscf[sbg][atsg - 1]
                };
                let d = env.values.get(sbg).copied().unwrap_or(0);
                qscf[sbg][atsg] = prev + delta * d;
            }
        } else {
            let mut acc: i32 = 0;
            for sbg in 0..(num_sbg as usize) {
                let d = env.values.get(sbg).copied().unwrap_or(0);
                acc += delta * d;
                qscf[sbg][atsg] = acc;
            }
        }
    }
    qscf
}

/// Dequantize signal envelope scale factors per ETSI TS 103 190-1
/// §5.7.6.3.5 Pseudocode 82 (non-balance path). `a = 2` for Fine /
/// 1.5 dB, `a = 1` for Coarse / 3 dB. `num_qmf_subbands = 64`.
pub fn dequantize_sig_scf(
    qscf: &[Vec<i32>],
    qmode_env: AspxQuantStep,
    delta_dir: &[bool],
    num_qmf_subbands: u32,
) -> Vec<Vec<f32>> {
    let a: f32 = match qmode_env {
        AspxQuantStep::Fine => 2.0,
        AspxQuantStep::Coarse => 1.0,
    };
    let n_subbands = num_qmf_subbands as f32;
    let num_sbg = qscf.len();
    if num_sbg == 0 {
        return Vec::new();
    }
    let num_env = qscf[0].len();
    let mut scf: Vec<Vec<f32>> = vec![vec![0.0_f32; num_env]; num_sbg];
    for atsg in 0..num_env {
        for sbg in 0..num_sbg {
            let q = qscf[sbg][atsg] as f32;
            scf[sbg][atsg] = n_subbands * 2_f32.powf(q / a);
        }
        if num_sbg >= 2
            && !delta_dir.get(atsg).copied().unwrap_or(false)
            && qscf[0][atsg] == 0
            && scf[1][atsg] < 0.0
        {
            scf[0][atsg] = scf[1][atsg];
        }
    }
    scf
}

/// Dequantize noise envelope scale factors per ETSI TS 103 190-1
/// §5.7.6.3.5 Pseudocode 83: `scf_noise_sbg = 2^(6 - qscf_noise_sbg)`.
pub fn dequantize_noise_scf(qscf: &[Vec<i32>]) -> Vec<Vec<f32>> {
    const NOISE_FLOOR_OFFSET: i32 = 6;
    let num_sbg = qscf.len();
    if num_sbg == 0 {
        return Vec::new();
    }
    let num_env = qscf[0].len();
    let mut scf: Vec<Vec<f32>> = vec![vec![0.0_f32; num_env]; num_sbg];
    for atsg in 0..num_env {
        for sbg in 0..num_sbg {
            let q = qscf[sbg][atsg];
            scf[sbg][atsg] = 2_f32.powi(NOISE_FLOOR_OFFSET - q);
        }
    }
    scf
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.1 Pseudocode 90 — envelope-energy estimation.
// ---------------------------------------------------------------------

/// Estimate per-envelope, per-subband energy of the HF QMF matrix
/// `q_high` per ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocode 90. Returns
/// `est_sig_sb` indexed `[sb_relative][atsg_sig]`.
pub fn estimate_envelope_energy(
    q_high: &[Vec<(f32, f32)>],
    sbg_sig: &[u32],
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
    num_sb_aspx: u32,
    sbx: u32,
    aspx_interpolation: bool,
) -> Vec<Vec<f32>> {
    let num_atsg_sig = atsg_sig.len().saturating_sub(1);
    let mut est: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg_sig]; num_sb_aspx as usize];
    if num_atsg_sig == 0 || sbg_sig.len() < 2 {
        return est;
    }
    for atsg in 0..num_atsg_sig {
        let tsa = atsg_sig[atsg] * num_ts_in_ats;
        let tsz = atsg_sig[atsg + 1] * num_ts_in_ats;
        let ts_span = tsz.saturating_sub(tsa) as f32;
        if ts_span <= 0.0 {
            continue;
        }
        let mut sbg = 0_usize;
        for sb in 0..(num_sb_aspx as usize) {
            while sbg + 1 < sbg_sig.len().saturating_sub(1) && (sb as u32 + sbx) >= sbg_sig[sbg + 1]
            {
                sbg += 1;
            }
            let mut est_sig: f64 = 0.0;
            for ts in tsa..tsz {
                let ts = ts as usize;
                if aspx_interpolation {
                    let sb_abs = sb + sbx as usize;
                    if sb_abs < q_high.len() && ts < q_high[sb_abs].len() {
                        let (re, im) = q_high[sb_abs][ts];
                        est_sig += (re as f64) * (re as f64) + (im as f64) * (im as f64);
                    }
                } else {
                    let j_lo = sbg_sig[sbg] as usize;
                    let j_hi = sbg_sig[sbg + 1] as usize;
                    for j in j_lo..j_hi {
                        if j < q_high.len() && ts < q_high[j].len() {
                            let (re, im) = q_high[j][ts];
                            est_sig += (re as f64) * (re as f64) + (im as f64) * (im as f64);
                        }
                    }
                }
            }
            if aspx_interpolation {
                est_sig /= ts_span as f64;
            } else {
                let band_span = (sbg_sig[sbg + 1] - sbg_sig[sbg]) as f64;
                if band_span > 0.0 {
                    est_sig /= band_span;
                }
                est_sig /= ts_span as f64;
            }
            est[sb][atsg] = est_sig as f32;
        }
    }
    est
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.1 Pseudocode 91 — map scf_sig_sbg / scf_noise_sbg to QMF
// subbands.
// ---------------------------------------------------------------------

/// Output of [`map_scf_to_qmf_subbands`].
#[derive(Debug, Clone, Default)]
pub struct ScfByQmfSubband {
    /// `scf_sig_sb[sb][atsg]`, shape `num_sb_aspx × num_atsg_sig`.
    pub scf_sig_sb: Vec<Vec<f32>>,
    /// `scf_noise_sb[sb][atsg]`, keyed by signal envelope.
    pub scf_noise_sb: Vec<Vec<f32>>,
}

/// Map `scf_sig_sbg` / `scf_noise_sbg` to per-QMF-subband matrices per
/// ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocode 91.
#[allow(clippy::too_many_arguments)]
pub fn map_scf_to_qmf_subbands(
    scf_sig_sbg: &[Vec<f32>],
    scf_noise_sbg: &[Vec<f32>],
    sbg_sig: &[u32],
    sbg_noise: &[u32],
    atsg_sig: &[u32],
    atsg_noise: &[u32],
    num_sb_aspx: u32,
    sbx: u32,
) -> ScfByQmfSubband {
    let num_atsg_sig = atsg_sig.len().saturating_sub(1);
    let mut scf_sig_sb: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg_sig]; num_sb_aspx as usize];
    let mut scf_noise_sb: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg_sig]; num_sb_aspx as usize];
    let num_sbg_sig = sbg_sig.len().saturating_sub(1);
    let num_sbg_noise = sbg_noise.len().saturating_sub(1);
    let mut atsg_noise_idx: usize = 0;
    for atsg in 0..num_atsg_sig {
        for sbg in 0..num_sbg_sig {
            let lo = sbg_sig[sbg].saturating_sub(sbx) as usize;
            let hi = sbg_sig[sbg + 1].saturating_sub(sbx) as usize;
            let val = scf_sig_sbg
                .get(sbg)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            for sb in lo..hi.min(num_sb_aspx as usize) {
                scf_sig_sb[sb][atsg] = val;
            }
        }
        if atsg_noise_idx + 1 < atsg_noise.len().saturating_sub(1)
            && atsg_sig.get(atsg).copied().unwrap_or(0)
                == atsg_noise
                    .get(atsg_noise_idx + 1)
                    .copied()
                    .unwrap_or(u32::MAX)
        {
            atsg_noise_idx += 1;
        }
        for sbg in 0..num_sbg_noise {
            let lo = sbg_noise[sbg].saturating_sub(sbx) as usize;
            let hi = sbg_noise[sbg + 1].saturating_sub(sbx) as usize;
            let val = scf_noise_sbg
                .get(sbg)
                .and_then(|row| row.get(atsg_noise_idx))
                .copied()
                .unwrap_or(0.0);
            for sb in lo..hi.min(num_sb_aspx as usize) {
                scf_noise_sb[sb][atsg] = val;
            }
        }
    }
    ScfByQmfSubband {
        scf_sig_sb,
        scf_noise_sb,
    }
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.1 Pseudocode 92 — sine_idx_sb derivation
//               Pseudocode 93 — sine_area_sb derivation
//               Pseudocode 94 — sine_lev_sb / noise_lev_sb derivation
// ---------------------------------------------------------------------

/// Derive the `sine_idx_sb[sb][atsg]` binary matrix per ETSI TS 103 190-1
/// §5.7.6.4.2.1 Pseudocode 92.
///
/// * `sbg_sig_highres` — high-resolution signal subband-group borders
///   (absolute QMF subbands). Length = `num_sbg_sig_highres + 1`.
/// * `add_harmonic[sbg]` — per-highres-subband `aspx_add_harmonic` flag
///   from [`AspxHfgenIwc1Ch`] / [`AspxHfgenIwc2Ch`]. Length must equal
///   `num_sbg_sig_highres`.
/// * `num_atsg_sig` — number of signal envelopes in the current interval.
/// * `sbx` / `num_sb_aspx` — crossover + A-SPX range width.
/// * `aspx_tsg_ptr` — transient-pointer (FIXFIX: 0; other classes:
///   parsed value from `aspx_framing`).
/// * `prev` — state from the previous interval: `(aspx_tsg_ptr_prev,
///   num_atsg_sig_prev, sine_idx_sb_prev)`. `None` on the first frame
///   (master_reset == 1). When provided, `sine_idx_sb_prev` has shape
///   `num_sb_aspx_prev × num_atsg_sig_prev`; if the current A-SPX range
///   is larger, the extra subbands are treated as zero.
///
/// Returns a `num_sb_aspx × num_atsg_sig` binary matrix (values 0/1).
pub fn derive_sine_idx_sb(
    sbg_sig_highres: &[u32],
    add_harmonic: &[bool],
    num_atsg_sig: u32,
    sbx: u32,
    num_sb_aspx: u32,
    aspx_tsg_ptr: u32,
    prev: Option<(u32, u32, &[Vec<u8>])>,
) -> Vec<Vec<u8>> {
    let num_sbg = sbg_sig_highres.len().saturating_sub(1);
    let num_atsg = num_atsg_sig as usize;
    if num_sbg == 0 || num_atsg == 0 {
        return vec![vec![0_u8; num_atsg]; num_sb_aspx as usize];
    }
    // p_sine_at_end: 0 if the previous interval ended with a sinusoid
    // present (aspx_tsg_ptr_prev == num_atsg_sig_prev), else -1 (we use
    // Option<usize> to represent "atsg to match later on" or sentinel).
    let p_sine_at_end_is_0 = match prev {
        Some((prev_tsg, prev_num_atsg, _)) => prev_tsg == prev_num_atsg,
        None => false, // first frame — spec uses the -1 branch
    };
    let mut sine_idx_sb: Vec<Vec<u8>> = vec![vec![0_u8; num_atsg]; num_sb_aspx as usize];
    for atsg in 0..num_atsg {
        for sbg in 0..num_sbg {
            let sba = sbg_sig_highres[sbg].saturating_sub(sbx) as usize;
            let sbz_local = sbg_sig_highres[sbg + 1].saturating_sub(sbx) as usize;
            if sba >= num_sb_aspx as usize {
                continue;
            }
            let hi = sbz_local.min(num_sb_aspx as usize);
            // "sb_mid = (int) 0.5 * (sbz + sba)" — integer truncation toward zero.
            let sb_mid = (sba + sbz_local) / 2;
            for sb in sba..hi {
                let ah = add_harmonic.get(sbg).copied().unwrap_or(false);
                let prev_idx_at_last_env = prev.and_then(|(_, prev_num, prev_mat)| {
                    if prev_num == 0 {
                        return None;
                    }
                    prev_mat
                        .get(sb)
                        .and_then(|row| row.get((prev_num - 1) as usize).copied())
                });
                let prev_was_set = matches!(prev_idx_at_last_env, Some(v) if v != 0);
                let condition = sb == sb_mid
                    && ((atsg as u32 >= aspx_tsg_ptr) || p_sine_at_end_is_0 || prev_was_set);
                sine_idx_sb[sb][atsg] = if condition && ah { 1 } else { 0 };
            }
        }
    }
    sine_idx_sb
}

/// Derive the `sine_area_sb[sb][atsg]` binary matrix per ETSI TS 103
/// 190-1 §5.7.6.4.2.1 Pseudocode 93.
///
/// The signal subband-group table is per-envelope (`sbg_sig[atsg]`):
/// the frequency resolution varies between high-res and low-res on a
/// per-envelope basis (§5.7.6.3.3.1 Pseudocode 77).
///
/// * `sine_idx_sb` — from [`derive_sine_idx_sb`].
/// * `sbg_sig_per_env` — per-envelope signal subband-group borders,
///   shape `num_atsg_sig × (num_sbg_sig[atsg] + 1)`.
/// * `sbx` / `num_sb_aspx` — crossover + A-SPX range width.
pub fn derive_sine_area_sb(
    sine_idx_sb: &[Vec<u8>],
    sbg_sig_per_env: &[Vec<u32>],
    sbx: u32,
    num_sb_aspx: u32,
) -> Vec<Vec<u8>> {
    let num_atsg = sine_idx_sb.first().map(|r| r.len()).unwrap_or(0);
    let mut sine_area_sb: Vec<Vec<u8>> = vec![vec![0_u8; num_atsg]; num_sb_aspx as usize];
    for (atsg, sbg_sig) in sbg_sig_per_env.iter().enumerate() {
        if atsg >= num_atsg {
            break;
        }
        let num_sbg = sbg_sig.len().saturating_sub(1);
        for sbg in 0..num_sbg {
            let sba = sbg_sig[sbg].saturating_sub(sbx) as usize;
            let sbz_local = sbg_sig[sbg + 1].saturating_sub(sbx) as usize;
            if sba >= num_sb_aspx as usize {
                continue;
            }
            let hi = sbz_local.min(num_sb_aspx as usize);
            let mut b_sine_present = 0_u8;
            for sb in sba..hi {
                if sine_idx_sb
                    .get(sb)
                    .and_then(|row| row.get(atsg))
                    .copied()
                    .unwrap_or(0)
                    == 1
                {
                    b_sine_present = 1;
                    break;
                }
            }
            for sb in sba..hi {
                sine_area_sb[sb][atsg] = b_sine_present;
            }
        }
    }
    sine_area_sb
}

/// Compute `sine_lev_sb[sb][atsg]` and `noise_lev_sb[sb][atsg]` per
/// ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocode 94.
///
/// ```text
/// sig_noise_fact  = scf_sig_sb[sb][atsg] / (1 + scf_noise_sb[sb][atsg])
/// sine_lev_sb     = sqrt(sig_noise_fact * sine_idx_sb[sb][atsg])
/// noise_lev_sb    = sqrt(sig_noise_fact * scf_noise_sb[sb][atsg])
/// ```
///
/// Shape of both outputs is `num_sb_aspx × num_atsg_sig`. These feed the
/// tone generator (§5.7.6.4.4) and noise generator (§5.7.6.4.3)
/// respectively.
pub fn derive_sine_noise_levels(
    scf_sig_sb: &[Vec<f32>],
    scf_noise_sb: &[Vec<f32>],
    sine_idx_sb: &[Vec<u8>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let num_sb = scf_sig_sb.len();
    if num_sb == 0 {
        return (Vec::new(), Vec::new());
    }
    let num_atsg = scf_sig_sb[0].len();
    let mut sine_lev_sb: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    let mut noise_lev_sb: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    for sb in 0..num_sb {
        for atsg in 0..num_atsg {
            let sig = scf_sig_sb[sb][atsg];
            let noise = scf_noise_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let sidx = sine_idx_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0) as f32;
            let denom = 1.0 + noise;
            let sig_noise_fact = if denom > 0.0 { sig / denom } else { 0.0 };
            sine_lev_sb[sb][atsg] = (sig_noise_fact * sidx).max(0.0).sqrt();
            noise_lev_sb[sb][atsg] = (sig_noise_fact * noise).max(0.0).sqrt();
        }
    }
    (sine_lev_sb, noise_lev_sb)
}

// ---------------------------------------------------------------------
// §5.7.6.4.2.2 Pseudocode 95 — compensatory gains (non-harmonic path).
// ---------------------------------------------------------------------

/// Compute the per-subband, per-envelope compensatory signal gain per
/// ETSI TS 103 190-1 §5.7.6.4.2.2 Pseudocode 95 (non-harmonic branch
/// `sine_area_sb == 0`).
pub fn compute_sig_gains(
    est_sig_sb: &[Vec<f32>],
    scf_sig_sb: &[Vec<f32>],
    scf_noise_sb: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    const EPSILON: f32 = 1.0;
    let num_sb = est_sig_sb.len();
    if num_sb == 0 {
        return Vec::new();
    }
    let num_atsg = est_sig_sb[0].len();
    let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; num_atsg]; num_sb];
    for sb in 0..num_sb {
        for atsg in 0..num_atsg {
            let est = est_sig_sb[sb][atsg];
            let sig = scf_sig_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let noise = scf_noise_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let denom = (EPSILON + est) * (1.0 + noise);
            let ratio = if denom > 0.0 { sig / denom } else { 0.0 };
            out[sb][atsg] = ratio.max(0.0).sqrt();
        }
    }
    out
}

/// Apply `sig_gain_sb[sb][atsg]` per-envelope, per-subband to the QMF
/// matrix `q` in place (Pseudocode 106 simplified — the non-harmonic,
/// non-limited, no-`Y_prev` path).
pub fn apply_envelope_gains(
    q: &mut [Vec<(f32, f32)>],
    sig_gain_sb: &[Vec<f32>],
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
    sbx: u32,
    num_sb_aspx: u32,
) {
    let num_atsg = atsg_sig.len().saturating_sub(1);
    if num_atsg == 0 {
        return;
    }
    for atsg in 0..num_atsg {
        let tsa = (atsg_sig[atsg] * num_ts_in_ats) as usize;
        let tsz = (atsg_sig[atsg + 1] * num_ts_in_ats) as usize;
        for sb in 0..(num_sb_aspx as usize) {
            let sb_abs = sb + sbx as usize;
            if sb_abs >= q.len() {
                break;
            }
            let g = sig_gain_sb
                .get(sb)
                .and_then(|row| row.get(atsg))
                .copied()
                .unwrap_or(0.0);
            let row = &mut q[sb_abs];
            let ts_hi = tsz.min(row.len());
            for slot in row[tsa..ts_hi].iter_mut() {
                slot.0 *= g;
                slot.1 *= g;
            }
        }
    }
}

/// Bundled A-SPX envelope-adjustment payload derived from bitstream
/// deltas (Pseudocodes 80 / 81 / 82 / 83 / 90 / 91 / 95).
#[derive(Debug, Clone)]
pub struct AspxEnvelopeAdjuster {
    pub atsg_sig: Vec<u32>,
    pub num_ts_in_ats: u32,
    pub sbx: u32,
    pub num_sb_aspx: u32,
    pub sig_gain_sb: Vec<Vec<f32>>,
    pub scf_sig_sb: Vec<Vec<f32>>,
    pub scf_noise_sb: Vec<Vec<f32>>,
}

impl AspxEnvelopeAdjuster {
    /// Build the full envelope-adjuster payload: delta decode (P80/P81)
    /// → dequantize (P82/P83) → estimate (P90) → map (P91) →
    /// compensate (P95).
    #[allow(clippy::too_many_arguments)]
    pub fn from_deltas(
        q_high: &[Vec<(f32, f32)>],
        tables: &AspxFrequencyTables,
        sig_deltas: &[AspxHuffEnv],
        noise_deltas: &[AspxHuffEnv],
        qmode_env: AspxQuantStep,
        delta_dir_sig: &[bool],
        atsg_sig: &[u32],
        atsg_noise: &[u32],
        num_ts_in_ats: u32,
        aspx_interpolation: bool,
    ) -> Self {
        let sbg_sig = &tables.sbg_sig_highres;
        let sbg_noise = &tables.sbg_noise;
        let num_sbg_sig = (sbg_sig.len() as u32).saturating_sub(1);
        let num_sbg_noise = (sbg_noise.len() as u32).saturating_sub(1);
        let qscf_sig = delta_decode_sig(sig_deltas, num_sbg_sig, &[], 1);
        let qscf_noise = delta_decode_noise(noise_deltas, num_sbg_noise, &[], 1);
        let scf_sig_sbg = dequantize_sig_scf(&qscf_sig, qmode_env, delta_dir_sig, 64);
        let scf_noise_sbg = dequantize_noise_scf(&qscf_noise);
        let est_sig_sb = estimate_envelope_energy(
            q_high,
            sbg_sig,
            atsg_sig,
            num_ts_in_ats,
            tables.num_sb_aspx,
            tables.sbx,
            aspx_interpolation,
        );
        let mapped = map_scf_to_qmf_subbands(
            &scf_sig_sbg,
            &scf_noise_sbg,
            sbg_sig,
            sbg_noise,
            atsg_sig,
            atsg_noise,
            tables.num_sb_aspx,
            tables.sbx,
        );
        let sig_gain_sb = compute_sig_gains(&est_sig_sb, &mapped.scf_sig_sb, &mapped.scf_noise_sb);
        Self {
            atsg_sig: atsg_sig.to_vec(),
            num_ts_in_ats,
            sbx: tables.sbx,
            num_sb_aspx: tables.num_sb_aspx,
            sig_gain_sb,
            scf_sig_sb: mapped.scf_sig_sb,
            scf_noise_sb: mapped.scf_noise_sb,
        }
    }

    /// Apply the gains to a QMF matrix in place.
    pub fn apply(&self, q: &mut [Vec<(f32, f32)>]) {
        apply_envelope_gains(
            q,
            &self.sig_gain_sb,
            &self.atsg_sig,
            self.num_ts_in_ats,
            self.sbx,
            self.num_sb_aspx,
        );
    }
}

// ---------------------------------------------------------------------
// §5.7.6.4.3 noise + §5.7.6.4.4 tone injection glue — ties Pseudocodes
// 92 (sine_idx_sb), 94 (sine_lev_sb / noise_lev_sb), 102 (noise gen),
// 104 (tone gen), 107 + 108 (HF assembler) together.
// ---------------------------------------------------------------------

/// Per-channel persistent state for the A-SPX HF regeneration pipeline.
///
/// Carries the index state of the noise generator (Pseudocode 103),
/// the tone generator (Pseudocode 105), and the `sine_idx_sb_prev`
/// matrix that Pseudocode 92 consults at the start of each interval.
/// The `tsg_ptr_prev` / `num_atsg_sig_prev` fields feed Pseudocode 92's
/// `p_sine_at_end` branch.
#[derive(Debug, Clone, Default)]
pub struct AspxChannelExtState {
    /// Noise generator `noise_idx_prev` state.
    pub noise: crate::aspx_noise::NoiseGenState,
    /// Tone generator `sine_idx_prev` state.
    pub tone: crate::aspx_tone::ToneGenState,
    /// `sine_idx_sb` from the previous interval (Pseudocode 92 P_prev
    /// input). `None` on the first frame or after `master_reset`.
    pub sine_idx_sb_prev: Option<Vec<Vec<u8>>>,
    /// `aspx_tsg_ptr_prev` — the previous interval's transient-pointer
    /// (0 for FIXFIX). Used in Pseudocode 92's p_sine_at_end test.
    pub tsg_ptr_prev: u32,
    /// `num_atsg_sig_prev` — the previous interval's signal-envelope
    /// count. Used in Pseudocode 92's p_sine_at_end test.
    pub num_atsg_sig_prev: u32,
}

impl AspxChannelExtState {
    /// Fresh state (first-frame, `master_reset == 1` semantics).
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to first-frame behaviour.
    pub fn reset(&mut self) {
        self.noise.reset();
        self.tone.reset();
        self.sine_idx_sb_prev = None;
        self.tsg_ptr_prev = 0;
        self.num_atsg_sig_prev = 0;
    }
}

/// Inject the A-SPX noise floor and tonal components into the
/// envelope-adjusted high-band QMF matrix `q`.
///
/// Pipeline (§5.7.6.4.2.1 / §5.7.6.4.3 / §5.7.6.4.4 / §5.7.6.4.5):
///
/// 1. Pseudocode 92 — derive `sine_idx_sb[sb][atsg]` from
///    `add_harmonic[sbg]` flags (input from `aspx_hfgen_iwc_1ch/2ch`).
/// 2. Pseudocode 94 — derive `sine_lev_sb[sb][atsg]` and
///    `noise_lev_sb[sb][atsg]` from the envelope adjuster's
///    `scf_sig_sb` / `scf_noise_sb` + the `sine_idx_sb` mask.
/// 3. Pseudocode 102 — noise generator produces `qmf_noise`.
/// 4. Pseudocode 104 — tone generator produces `qmf_sine`.
/// 5. Pseudocodes 107 / 108 — add both into `q` in place.
///
/// Persistent index state and `sine_idx_sb_prev` are advanced on
/// `state` so the next call picks up where this one left off.
///
/// `add_harmonic` must have length `num_sbg_sig_highres`. `atsg_sig_ptr`
/// is the transient-pointer for the current interval (0 for FIXFIX).
#[allow(clippy::too_many_arguments)]
pub fn inject_noise_and_tone(
    q: &mut [Vec<(f32, f32)>],
    adjuster: &AspxEnvelopeAdjuster,
    tables: &AspxFrequencyTables,
    atsg_noise: &[u32],
    add_harmonic: &[bool],
    aspx_tsg_ptr: u32,
    state: &mut AspxChannelExtState,
) {
    const NUM_QMF_SUBBANDS: u32 = crate::qmf::NUM_QMF_SUBBANDS as u32;
    let num_atsg_sig = adjuster.atsg_sig.len().saturating_sub(1) as u32;
    if num_atsg_sig == 0 {
        return;
    }
    // Pseudocode 92 — sine_idx_sb.
    let prev_ref: Option<(u32, u32, &[Vec<u8>])> = state
        .sine_idx_sb_prev
        .as_ref()
        .map(|m| (state.tsg_ptr_prev, state.num_atsg_sig_prev, m.as_slice()));
    let sine_idx_sb = derive_sine_idx_sb(
        &tables.sbg_sig_highres,
        add_harmonic,
        num_atsg_sig,
        tables.sbx,
        tables.num_sb_aspx,
        aspx_tsg_ptr,
        prev_ref,
    );
    // Pseudocode 94 — sine_lev_sb / noise_lev_sb.
    let (sine_lev_sb, noise_lev_sb) =
        derive_sine_noise_levels(&adjuster.scf_sig_sb, &adjuster.scf_noise_sb, &sine_idx_sb);
    // Pseudocode 102 — noise generator.
    let qmf_noise = crate::aspx_noise::generate_qmf_noise(
        &noise_lev_sb,
        &adjuster.atsg_sig,
        atsg_noise,
        adjuster.num_ts_in_ats,
        NUM_QMF_SUBBANDS,
        tables.sbx,
        tables.num_sb_aspx,
        &mut state.noise,
    );
    // Pseudocode 104 — tone generator.
    let qmf_sine = crate::aspx_tone::generate_qmf_sine(
        &sine_lev_sb,
        &adjuster.atsg_sig,
        adjuster.num_ts_in_ats,
        NUM_QMF_SUBBANDS,
        tables.sbx,
        tables.num_sb_aspx,
        &mut state.tone,
    );
    // Pseudocodes 107 + 108 — add noise + tone into the HF matrix.
    crate::aspx_tone::hf_assemble(
        q,
        &qmf_noise,
        &qmf_sine,
        &adjuster.atsg_sig,
        adjuster.num_ts_in_ats,
        tables.sbx,
        tables.sbz,
    );
    // Advance persistent state for the next interval.
    state.sine_idx_sb_prev = Some(sine_idx_sb);
    state.tsg_ptr_prev = aspx_tsg_ptr;
    state.num_atsg_sig_prev = num_atsg_sig;
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[allow(clippy::too_many_arguments)]
    fn build_aspx_config_bits(
        qmode: u32,
        start: u32,
        stop: u32,
        scale: u32,
        interp: bool,
        preflat: bool,
        limiter: bool,
        noise_sbg: u32,
        num_env_bits_fixfix: u32,
        freq_res_mode: u32,
    ) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write_u32(qmode, 1);
        bw.write_u32(start, 3);
        bw.write_u32(stop, 2);
        bw.write_u32(scale, 1);
        bw.write_bit(interp);
        bw.write_bit(preflat);
        bw.write_bit(limiter);
        bw.write_u32(noise_sbg, 2);
        bw.write_u32(num_env_bits_fixfix, 1);
        bw.write_u32(freq_res_mode, 2);
        bw.align_to_byte();
        bw.finish()
    }

    #[test]
    fn aspx_config_bit_order_matches_table_50() {
        // quant_mode_env=1 (coarse/3dB), start_freq=5, stop_freq=2,
        // master_freq_scale=1 (highres), interpolation=1, preflat=0,
        // limiter=1, noise_sbg=3, num_env_bits_fixfix=1,
        // freq_res_mode=2 (duration-dependent default).
        let bytes = build_aspx_config_bits(1, 5, 2, 1, true, false, true, 3, 1, 2);
        let mut br = BitReader::new(&bytes);
        let cfg = parse_aspx_config(&mut br).unwrap();
        assert_eq!(cfg.quant_mode_env, AspxQuantStep::Coarse);
        assert_eq!(cfg.start_freq, 5);
        assert_eq!(cfg.stop_freq, 2);
        assert_eq!(cfg.master_freq_scale, AspxMasterFreqScale::HighRes);
        assert!(cfg.interpolation);
        assert!(!cfg.preflat);
        assert!(cfg.limiter);
        assert_eq!(cfg.noise_sbg, 3);
        assert_eq!(cfg.num_env_bits_fixfix, 1);
        assert_eq!(cfg.freq_res_mode, AspxFreqResMode::DurationDependent);
    }

    #[test]
    fn aspx_config_exactly_15_bits() {
        // Pack a config plus a known sentinel bit right after; verify
        // the parser left the cursor at the 16th bit (index 15).
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1);
        bw.write_u32(0, 3);
        bw.write_u32(0, 2);
        bw.write_u32(0, 1);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_u32(0, 2);
        bw.write_u32(0, 1);
        bw.write_u32(0, 2);
        // Sentinel bit.
        bw.write_bit(true);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let _ = parse_aspx_config(&mut br).unwrap();
        assert_eq!(br.bit_position(), 15);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_config_helpers() {
        let cfg = AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::Signalled,
        };
        assert_eq!(cfg.num_noise_sbgroups(), 1);
        assert_eq!(cfg.fixfix_tmp_num_env_bits(), 1);
        assert!(cfg.signals_freq_res());
        let cfg2 = AspxConfig {
            noise_sbg: 3,
            num_env_bits_fixfix: 1,
            freq_res_mode: AspxFreqResMode::High,
            ..cfg
        };
        assert_eq!(cfg2.num_noise_sbgroups(), 4);
        assert_eq!(cfg2.fixfix_tmp_num_env_bits(), 2);
        assert!(!cfg2.signals_freq_res());
    }

    #[test]
    fn companding_control_mono() {
        // num_chan = 1: no sync_flag; one b_compand_on; no avg when on.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_compand_on[0] = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 1).unwrap();
        assert!(cc.sync_flag.is_none());
        assert_eq!(cc.compand_on, vec![true]);
        assert!(cc.compand_avg.is_none());
    }

    #[test]
    fn companding_control_stereo_sync_needs_avg() {
        // num_chan = 2, sync_flag = 1 -> single compand_on; set it 0
        // (companding off) -> needs avg.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // sync_flag
        bw.write_bit(false); // compand_on[0] = 0
        bw.write_bit(true); // compand_avg = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 2).unwrap();
        assert_eq!(cc.sync_flag, Some(true));
        assert_eq!(cc.compand_on, vec![false]);
        assert_eq!(cc.compand_avg, Some(true));
    }

    #[test]
    fn companding_control_stereo_no_sync() {
        // num_chan = 2, sync_flag = 0 -> per-channel flags.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // sync_flag
        bw.write_bit(true); // compand_on[0] = 1
        bw.write_bit(true); // compand_on[1] = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 2).unwrap();
        assert_eq!(cc.sync_flag, Some(false));
        assert_eq!(cc.compand_on, vec![true, true]);
        assert!(cc.compand_avg.is_none());
    }

    #[test]
    fn companding_control_5ch_one_off() {
        // num_chan = 5, sync_flag = 0, channels = [1,1,0,1,1] -> avg
        // required.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // sync
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // ch2 off
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // compand_avg = 0
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cc = parse_companding_control(&mut br, 5).unwrap();
        assert_eq!(cc.sync_flag, Some(false));
        assert_eq!(cc.compand_on, vec![true, true, false, true, true]);
        assert_eq!(cc.compand_avg, Some(false));
    }

    // --- aspx_framing tests ------------------------------------------

    /// Build a default `AspxConfig` parameterised on the two fields
    /// `parse_aspx_framing` actually consults.
    fn test_config(num_env_bits_fixfix: u8, freq_res_mode: AspxFreqResMode) -> AspxConfig {
        AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix,
            freq_res_mode,
        }
    }

    #[test]
    fn aspx_int_class_prefix_code_matches_table_126() {
        // 0b0 -> FIXFIX
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(false);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::FixFix);
        assert_eq!(br.bit_position(), 1);

        // 0b10 -> FIXVAR
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(true);
            bw.write_bit(false);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::FixVar);
        assert_eq!(br.bit_position(), 2);

        // 0b110 -> VARFIX
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(true);
            bw.write_bit(true);
            bw.write_bit(false);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::VarFix);
        assert_eq!(br.bit_position(), 3);

        // 0b111 -> VARVAR
        let bytes = {
            let mut bw = BitWriter::new();
            bw.write_bit(true);
            bw.write_bit(true);
            bw.write_bit(true);
            bw.align_to_byte();
            bw.finish()
        };
        let mut br = BitReader::new(&bytes);
        assert_eq!(AspxIntClass::read(&mut br).unwrap(), AspxIntClass::VarVar);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn aspx_framing_fixfix_signalled_freq_res() {
        // aspx_num_env_bits_fixfix = 1 (envbits = 2), freq_res_mode =
        // Signalled. Bits: int_class=0 (FIXFIX, 1 bit); tmp_num_env=10
        // (2 bits, value 2 -> num_env = 1 << 2 = 4); aspx_freq_res[0]=1
        // (1 bit). Total = 4 bits. Sentinel follows.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // FIXFIX
        bw.write_u32(2, 2); // tmp_num_env = 2
        bw.write_bit(true); // freq_res[0]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(1, AspxFreqResMode::Signalled);
        let f = parse_aspx_framing(&mut br, &cfg, true, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::FixFix);
        assert_eq!(f.num_env, 4);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.freq_res, vec![true]);
        assert_eq!(f.var_bord_left, None);
        assert_eq!(f.var_bord_right, None);
        assert_eq!(f.num_rel_left, 0);
        assert_eq!(f.num_rel_right, 0);
        assert!(f.rel_bord_left.is_empty());
        assert!(f.rel_bord_right.is_empty());
        assert_eq!(f.tsg_ptr, None);
        // Cursor lands on the sentinel bit.
        assert_eq!(br.bit_position(), 4);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_fixfix_narrow_envbits_no_freq_res() {
        // aspx_num_env_bits_fixfix = 0 (envbits = 1), freq_res_mode =
        // High (NOT signalled). Bits: FIXFIX=0 (1 bit), tmp_num_env=1
        // (1 bit, num_env = 2). Total = 2 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // FIXFIX
        bw.write_bit(true); // tmp_num_env = 1 -> num_env = 2
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::High);
        let f = parse_aspx_framing(&mut br, &cfg, false, true).unwrap();
        assert_eq!(f.int_class, AspxIntClass::FixFix);
        assert_eq!(f.num_env, 2);
        assert_eq!(f.num_noise, 2);
        assert!(f.freq_res.is_empty());
        assert_eq!(br.bit_position(), 2);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_fixvar_ts_over_8_with_two_rel() {
        // FIXVAR prefix = 0b10 (2 bits). num_aspx_timeslots > 8 so
        // note-1 fields are 2 bits. var_bord_right = 0b11 (2 bits),
        // num_rel_right = 2 (2 bits), rel_bord_right = [0b01, 0b10]
        // (2 bits each). Then branch ends -> num_env = 0 + 2 + 1 = 3,
        // ptr_bits = ceil(log2(5)) = 3. Then since freq_res_mode is
        // Signalled, read 3 freq_res bits. tsg_ptr = 0b101, freq_res =
        // [1,0,1]. Total = 2+2+2+2+2+3+3 = 16 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(false); // FIXVAR
        bw.write_u32(0b11, 2); // var_bord_right
        bw.write_u32(2, 2); // num_rel_right
        bw.write_u32(0b01, 2); // rel_bord_right[0]
        bw.write_u32(0b10, 2); // rel_bord_right[1]
        bw.write_u32(0b101, 3); // tsg_ptr
        bw.write_bit(true); // freq_res[0]
        bw.write_bit(false); // freq_res[1]
        bw.write_bit(true); // freq_res[2]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Signalled);
        let f = parse_aspx_framing(&mut br, &cfg, true, true).unwrap();
        assert_eq!(f.int_class, AspxIntClass::FixVar);
        assert_eq!(f.num_env, 3);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.var_bord_right, Some(0b11));
        assert_eq!(f.num_rel_right, 2);
        assert_eq!(f.rel_bord_right, vec![0b01, 0b10]);
        assert_eq!(f.num_rel_left, 0);
        assert!(f.rel_bord_left.is_empty());
        assert_eq!(f.var_bord_left, None);
        assert_eq!(f.tsg_ptr, Some(0b101));
        assert_eq!(f.freq_res, vec![true, false, true]);
        assert_eq!(br.bit_position(), 16);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_varfix_iframe_ts_short() {
        // VARFIX prefix = 0b110 (3 bits). b_iframe = true so
        // var_bord_left is read (2 bits). num_aspx_timeslots <= 8 so
        // note-1 fields are 1 bit. num_rel_left = 1 (1 bit),
        // rel_bord_left = [1] (1 bit). Branch ends -> num_env = 2,
        // ptr_bits = ceil(log2(4)) = 2. freq_res_mode = Low (not
        // signalled) so no freq_res read. Total bits = 3+2+1+1+2 = 9.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // VARFIX
        bw.write_u32(0b10, 2); // var_bord_left
        bw.write_u32(1, 1); // num_rel_left
        bw.write_u32(1, 1); // rel_bord_left[0]
        bw.write_u32(0b11, 2); // tsg_ptr
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Low);
        let f = parse_aspx_framing(&mut br, &cfg, true, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::VarFix);
        assert_eq!(f.num_env, 2);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.var_bord_left, Some(0b10));
        assert_eq!(f.num_rel_left, 1);
        assert_eq!(f.rel_bord_left, vec![1]);
        assert_eq!(f.num_rel_right, 0);
        assert!(f.rel_bord_right.is_empty());
        assert_eq!(f.var_bord_right, None);
        assert_eq!(f.tsg_ptr, Some(0b11));
        assert!(f.freq_res.is_empty());
        assert_eq!(br.bit_position(), 9);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_varfix_non_iframe_omits_var_bord_left() {
        // Same as above but b_iframe = false -> var_bord_left is NOT
        // present in the bitstream. Bits: VARFIX=0b110 (3),
        // num_rel_left=0 (1), tsg_ptr on num_env=1 -> ptr_bits =
        // ceil(log2(3)) = 2. Total = 3+1+2 = 6 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false); // VARFIX
        bw.write_u32(0, 1); // num_rel_left = 0
        bw.write_u32(0b10, 2); // tsg_ptr
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Low);
        let f = parse_aspx_framing(&mut br, &cfg, false, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::VarFix);
        assert_eq!(f.num_env, 1);
        assert_eq!(f.num_noise, 1);
        assert_eq!(f.var_bord_left, None);
        assert_eq!(f.num_rel_left, 0);
        assert_eq!(f.tsg_ptr, Some(0b10));
        assert_eq!(br.bit_position(), 6);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_framing_varvar_iframe_symmetric() {
        // VARVAR prefix = 0b111 (3 bits). b_iframe = true.
        // num_aspx_timeslots <= 8 (1-bit note-1 fields).
        //   var_bord_left = 0b01 (2)
        //   num_rel_left = 1 (1) -> rel_bord_left = [0] (1)
        //   var_bord_right = 0b11 (2)
        //   num_rel_right = 1 (1) -> rel_bord_right = [1] (1)
        // num_env = 1+1+1 = 3, ptr_bits = ceil(log2(5)) = 3, tsg_ptr =
        // 0b100. freq_res_mode = Signalled -> 3 freq_res bits.
        // Total = 3 + 2 + 1 + 1 + 2 + 1 + 1 + 3 + 3 = 17 bits.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(true); // VARVAR
        bw.write_u32(0b01, 2); // var_bord_left
        bw.write_u32(1, 1); // num_rel_left
        bw.write_u32(0, 1); // rel_bord_left[0]
        bw.write_u32(0b11, 2); // var_bord_right
        bw.write_u32(1, 1); // num_rel_right
        bw.write_u32(1, 1); // rel_bord_right[0]
        bw.write_u32(0b100, 3); // tsg_ptr
        bw.write_bit(false); // freq_res[0]
        bw.write_bit(true); // freq_res[1]
        bw.write_bit(false); // freq_res[2]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = test_config(0, AspxFreqResMode::Signalled);
        let f = parse_aspx_framing(&mut br, &cfg, true, false).unwrap();
        assert_eq!(f.int_class, AspxIntClass::VarVar);
        assert_eq!(f.num_env, 3);
        assert_eq!(f.num_noise, 2);
        assert_eq!(f.var_bord_left, Some(0b01));
        assert_eq!(f.num_rel_left, 1);
        assert_eq!(f.rel_bord_left, vec![0]);
        assert_eq!(f.var_bord_right, Some(0b11));
        assert_eq!(f.num_rel_right, 1);
        assert_eq!(f.rel_bord_right, vec![1]);
        assert_eq!(f.tsg_ptr, Some(0b100));
        assert_eq!(f.freq_res, vec![false, true, false]);
        assert_eq!(br.bit_position(), 17);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn ceil_log2_matches_spec_ptr_bits_formula() {
        // Spec: ptr_bits = ceil(log(num_env + 2) / log(2)).
        // num_env = 1 -> log2(3) = ~1.58 -> 2
        // num_env = 2 -> log2(4) = 2    -> 2
        // num_env = 3 -> log2(5) = ~2.32 -> 3
        // num_env = 4 -> log2(6) = ~2.58 -> 3
        // num_env = 5 -> log2(7) = ~2.81 -> 3
        assert_eq!(ceil_log2(1 + 2), 2);
        assert_eq!(ceil_log2(2 + 2), 2);
        assert_eq!(ceil_log2(3 + 2), 3);
        assert_eq!(ceil_log2(4 + 2), 3);
        assert_eq!(ceil_log2(5 + 2), 3);
    }

    #[test]
    fn num_qmf_timeslots_matches_table_189() {
        assert_eq!(num_qmf_timeslots(2048), 32);
        assert_eq!(num_qmf_timeslots(1920), 30);
        assert_eq!(num_qmf_timeslots(1536), 24);
        assert_eq!(num_qmf_timeslots(1024), 16);
        assert_eq!(num_qmf_timeslots(960), 15);
        assert_eq!(num_qmf_timeslots(768), 12);
        assert_eq!(num_qmf_timeslots(512), 8);
        assert_eq!(num_qmf_timeslots(384), 6);
    }

    #[test]
    fn num_ts_in_ats_matches_table_192() {
        // >= 1536 -> 2, else 1.
        assert_eq!(num_ts_in_ats(2048), 2);
        assert_eq!(num_ts_in_ats(1920), 2);
        assert_eq!(num_ts_in_ats(1536), 2);
        assert_eq!(num_ts_in_ats(1024), 1);
        assert_eq!(num_ts_in_ats(960), 1);
        assert_eq!(num_ts_in_ats(768), 1);
        assert_eq!(num_ts_in_ats(512), 1);
        assert_eq!(num_ts_in_ats(384), 1);
    }

    #[test]
    fn aspx_delta_dir_reads_num_env_plus_num_noise_bits() {
        // Framing with num_env=3, num_noise=2 -> 5 bits total.
        let framing = AspxFraming {
            int_class: AspxIntClass::VarVar,
            num_env: 3,
            num_noise: 2,
            freq_res: Vec::new(),
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: Vec::new(),
            rel_bord_right: Vec::new(),
            tsg_ptr: None,
        };
        // Write: sig = [1,0,1], noise = [0,1], sentinel.
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let dd = parse_aspx_delta_dir(&mut br, &framing).unwrap();
        assert_eq!(dd.sig_delta_dir, vec![true, false, true]);
        assert_eq!(dd.noise_delta_dir, vec![false, true]);
        assert_eq!(br.bit_position(), 5);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_delta_dir_single_env_single_noise() {
        // Minimal case: num_env=1, num_noise=1 -> 2 bits.
        let framing = AspxFraming {
            int_class: AspxIntClass::FixFix,
            num_env: 1,
            num_noise: 1,
            freq_res: Vec::new(),
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: Vec::new(),
            rel_bord_right: Vec::new(),
            tsg_ptr: None,
        };
        let mut bw = BitWriter::new();
        bw.write_bit(false); // sig[0] = 0
        bw.write_bit(true); // noise[0] = 1
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let dd = parse_aspx_delta_dir(&mut br, &framing).unwrap();
        assert_eq!(dd.sig_delta_dir, vec![false]);
        assert_eq!(dd.noise_delta_dir, vec![true]);
    }

    #[test]
    fn aspx_hfgen_iwc_1ch_all_gates_off() {
        // num_sbg_noise=2 -> 4 bits (tna_mode[0..=1], 2 bits each).
        // ah_present=0, fic_present=0, tic_present=0 — 3 more bits.
        // Total = 7 bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0b01, 2); // tna_mode[0] = 1
        bw.write_u32(0b10, 2); // tna_mode[1] = 2
        bw.write_bit(false); // ah_present
        bw.write_bit(false); // fic_present
        bw.write_bit(false); // tic_present
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_1ch(&mut br, 2, 3, 16).unwrap();
        assert_eq!(h.tna_mode, vec![1, 2]);
        assert!(!h.ah_present);
        assert!(!h.fic_present);
        assert!(!h.tic_present);
        // Gated-off vectors stay all-zero at their declared length.
        assert_eq!(h.add_harmonic, vec![false; 3]);
        assert_eq!(h.fic_used_in_sfb, vec![false; 3]);
        assert_eq!(h.tic_used_in_slot, vec![false; 16]);
        assert_eq!(br.bit_position(), 7);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_hfgen_iwc_1ch_all_gates_on() {
        // num_sbg_noise=1, num_sbg_sig_highres=2, num_aspx_timeslots=3.
        // tna_mode[0] = 3 (2b)
        // ah_present=1 + add_harmonic=[1,0] (3b)
        // fic_present=1 + fic_used_in_sfb=[0,1] (3b)
        // tic_present=1 + tic_used_in_slot=[1,0,1] (4b)
        // Total = 2 + 3 + 3 + 4 = 12 bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0b11, 2);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_1ch(&mut br, 1, 2, 3).unwrap();
        assert_eq!(h.tna_mode, vec![3]);
        assert!(h.ah_present);
        assert_eq!(h.add_harmonic, vec![true, false]);
        assert!(h.fic_present);
        assert_eq!(h.fic_used_in_sfb, vec![false, true]);
        assert!(h.tic_present);
        assert_eq!(h.tic_used_in_slot, vec![true, false, true]);
        assert_eq!(br.bit_position(), 12);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_hfgen_iwc_2ch_balance_on_mirrors_tna_and_all_gates_off() {
        // aspx_balance = 1 -> tna_mode[1] mirrors tna_mode[0].
        // num_sbg_noise=2 -> 4 bits for tna0 (no tna1 in bitstream).
        // ah_left=0, ah_right=0, fic_present=0, tic_present=0 -> 4 bits.
        // Total = 4 + 4 = 8 bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0b01, 2); // tna0[0] = 1
        bw.write_u32(0b10, 2); // tna0[1] = 2
        bw.write_bit(false); // ah_left
        bw.write_bit(false); // ah_right
        bw.write_bit(false); // fic_present
        bw.write_bit(false); // tic_present
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_2ch(&mut br, true, 2, 3, 16).unwrap();
        assert_eq!(h.tna_mode[0], vec![1, 2]);
        // Channel 1 mirrors channel 0 verbatim.
        assert_eq!(h.tna_mode[1], vec![1, 2]);
        assert!(!h.ah_left && !h.ah_right);
        assert!(!h.fic_present && !h.tic_present);
        assert_eq!(h.add_harmonic[0], vec![false; 3]);
        assert_eq!(h.add_harmonic[1], vec![false; 3]);
        assert_eq!(h.fic_used_in_sfb[0], vec![false; 3]);
        assert_eq!(h.fic_used_in_sfb[1], vec![false; 3]);
        assert_eq!(h.tic_used_in_slot[0], vec![false; 16]);
        assert_eq!(h.tic_used_in_slot[1], vec![false; 16]);
        assert_eq!(br.bit_position(), 8);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_hfgen_iwc_2ch_balance_off_reads_both_tnas() {
        // aspx_balance = 0 -> tna1 is read from the bitstream.
        // num_sbg_noise=1: tna0 (2b) + tna1 (2b).
        // ah_left=1 + ah_left_vector (sbg_highres=2 -> 2 bits); ah_right=0.
        // fic_present=1, fic_left=0, fic_right=1 + right vector (2 bits).
        // tic_present=1, tic_copy=0, tic_left=1, tic_right=0, + left vec (ats=2 -> 2 bits).
        // Bits:
        // tna0=3, tna1=2 -> 11, 10
        // ah_left=1, ah0=[1,0]
        // ah_right=0
        // fic_present=1, fic_left=0, fic_right=1, fic1=[1,1]
        // tic_present=1, tic_copy=0, tic_left=1, tic_right=0, tic0=[0,1]
        let mut bw = BitWriter::new();
        bw.write_u32(0b11, 2); // tna0[0]
        bw.write_u32(0b10, 2); // tna1[0]
        bw.write_bit(true); // ah_left
        bw.write_bit(true);
        bw.write_bit(false); // ah0 = [1,0]
        bw.write_bit(false); // ah_right
        bw.write_bit(true); // fic_present
        bw.write_bit(false); // fic_left
        bw.write_bit(true); // fic_right
        bw.write_bit(true);
        bw.write_bit(true); // fic1 = [1,1]
        bw.write_bit(true); // tic_present
        bw.write_bit(false); // tic_copy
        bw.write_bit(true); // tic_left
        bw.write_bit(false); // tic_right
        bw.write_bit(false);
        bw.write_bit(true); // tic0 = [0,1]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_2ch(&mut br, false, 1, 2, 2).unwrap();
        assert_eq!(h.tna_mode[0], vec![3]);
        assert_eq!(h.tna_mode[1], vec![2]);
        assert!(h.ah_left);
        assert!(!h.ah_right);
        assert_eq!(h.add_harmonic[0], vec![true, false]);
        assert_eq!(h.add_harmonic[1], vec![false, false]);
        assert!(h.fic_present);
        assert!(!h.fic_left);
        assert!(h.fic_right);
        assert_eq!(h.fic_used_in_sfb[0], vec![false, false]);
        assert_eq!(h.fic_used_in_sfb[1], vec![true, true]);
        assert!(h.tic_present);
        assert!(!h.tic_copy);
        assert!(h.tic_left);
        assert!(!h.tic_right);
        assert_eq!(h.tic_used_in_slot[0], vec![false, true]);
        assert_eq!(h.tic_used_in_slot[1], vec![false, false]);
        // Total bits: 2+2+1+2+1+1+1+1+2+1+1+1+1+2 = 19
        assert_eq!(br.bit_position(), 19);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_hfgen_iwc_2ch_tic_copy_mirrors_left_into_right() {
        // num_sbg_noise=1 (tna0 only, balance=1).
        // ah_left=0, ah_right=0.
        // fic_present=0.
        // tic_present=1, tic_copy=1, tic0=[1,0,1] (ats=3 -> 3 bits).
        // Because tic_copy == 1, tic1 should equal tic0 without extra
        // bits being read (and no tic_left/tic_right signalled).
        let mut bw = BitWriter::new();
        bw.write_u32(0b00, 2); // tna0[0] = 0
        bw.write_bit(false); // ah_left
        bw.write_bit(false); // ah_right
        bw.write_bit(false); // fic_present
        bw.write_bit(true); // tic_present
        bw.write_bit(true); // tic_copy
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true); // tic0 = [1,0,1]
        bw.write_bit(true); // sentinel
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let h = parse_aspx_hfgen_iwc_2ch(&mut br, true, 1, 2, 3).unwrap();
        assert_eq!(h.tna_mode[0], vec![0]);
        assert_eq!(h.tna_mode[1], vec![0]); // balance mirror
        assert!(h.tic_present);
        assert!(h.tic_copy);
        assert_eq!(h.tic_used_in_slot[0], vec![true, false, true]);
        assert_eq!(h.tic_used_in_slot[1], vec![true, false, true]);
        // Total: 2 + 1+1 + 1 + 1+1+3 = 10
        assert_eq!(br.bit_position(), 10);
        assert!(br.read_bit().unwrap());
    }

    #[test]
    fn aspx_hcb_decode_delta_walks_len_cw_arrays() {
        // Construct a synthetic micro-codebook:
        //   symbol 0: "0"    (len=1)
        //   symbol 1: "10"   (len=2)
        //   symbol 2: "110"  (len=3)
        //   symbol 3: "111"  (len=3)
        // cb_off = 2 so decoded deltas are {-2, -1, 0, 1}.
        static LENS: &[u8] = &[1, 2, 3, 3];
        static CWS: &[u32] = &[0b0, 0b10, 0b110, 0b111];
        let hcb = AspxHcb {
            name: "SYNTHETIC",
            len: LENS,
            cw: CWS,
            cb_off: 2,
        };
        // Pack the codeword for symbol 3 then symbol 1 then symbol 0.
        // Expected deltas: 1, -1, -2.
        let mut bw = BitWriter::new();
        bw.write_u32(0b111, 3);
        bw.write_u32(0b10, 2);
        bw.write_u32(0b0, 1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(hcb.decode_delta(&mut br).unwrap(), 1);
        assert_eq!(hcb.decode_delta(&mut br).unwrap(), -1);
        assert_eq!(hcb.decode_delta(&mut br).unwrap(), -2);
    }

    #[test]
    fn aspx_hcb_annex_a2_metadata_matches_pdf_headers() {
        // Codebook lengths and cb_off values straight off the Annex
        // A.2 table headers in ETSI TS 103 190-1 V1.4.1 (2025-07).
        assert_eq!(ASPX_HCB_ENV_LEVEL_15_F0_META.codebook_length, 71);
        assert_eq!(ASPX_HCB_ENV_LEVEL_15_F0_META.cb_off, 0);
        assert_eq!(ASPX_HCB_ENV_LEVEL_15_DF_META.codebook_length, 141);
        assert_eq!(ASPX_HCB_ENV_LEVEL_15_DF_META.cb_off, 70);
        assert_eq!(ASPX_HCB_ENV_LEVEL_15_DT_META.codebook_length, 141);
        assert_eq!(ASPX_HCB_ENV_LEVEL_15_DT_META.cb_off, 70);
        assert_eq!(ASPX_HCB_ENV_BALANCE_15_F0_META.codebook_length, 25);
        assert_eq!(ASPX_HCB_ENV_BALANCE_15_DF_META.codebook_length, 49);
        assert_eq!(ASPX_HCB_ENV_BALANCE_15_DF_META.cb_off, 24);
        assert_eq!(ASPX_HCB_ENV_BALANCE_15_DT_META.codebook_length, 49);
        assert_eq!(ASPX_HCB_ENV_BALANCE_15_DT_META.cb_off, 24);
        assert_eq!(ASPX_HCB_ENV_LEVEL_30_F0_META.codebook_length, 36);
        assert_eq!(ASPX_HCB_ENV_LEVEL_30_DF_META.codebook_length, 71);
        assert_eq!(ASPX_HCB_ENV_LEVEL_30_DF_META.cb_off, 35);
        assert_eq!(ASPX_HCB_ENV_LEVEL_30_DT_META.codebook_length, 71);
        assert_eq!(ASPX_HCB_ENV_LEVEL_30_DT_META.cb_off, 35);
        assert_eq!(ASPX_HCB_ENV_BALANCE_30_F0_META.codebook_length, 13);
        assert_eq!(ASPX_HCB_ENV_BALANCE_30_DF_META.codebook_length, 25);
        assert_eq!(ASPX_HCB_ENV_BALANCE_30_DF_META.cb_off, 12);
        assert_eq!(ASPX_HCB_ENV_BALANCE_30_DT_META.codebook_length, 25);
        assert_eq!(ASPX_HCB_ENV_BALANCE_30_DT_META.cb_off, 12);
    }

    #[test]
    fn num_aspx_timeslots_matches_pseudocode_75a() {
        // num_aspx_timeslots = num_qmf_timeslots / num_ts_in_ats
        assert_eq!(num_aspx_timeslots(2048), 16); // 32 / 2
        assert_eq!(num_aspx_timeslots(1920), 15); // 30 / 2
        assert_eq!(num_aspx_timeslots(1536), 12); // 24 / 2
        assert_eq!(num_aspx_timeslots(1024), 16); // 16 / 1
        assert_eq!(num_aspx_timeslots(960), 15); // 15 / 1
        assert_eq!(num_aspx_timeslots(768), 12); // 12 / 1
        assert_eq!(num_aspx_timeslots(512), 8); // 8 / 1
        assert_eq!(num_aspx_timeslots(384), 6); // 6 / 1
    }

    // --- A-SPX Huffman codebook transcription tests -------------------

    /// Pack `value` as an `len`-bit MSB-first code into `bw`.
    fn push_hcb_code(bw: &mut BitWriter, len: u8, value: u32) {
        bw.write_u32(value, len as u32);
    }

    /// Round-trip every entry of one codebook: for each symbol
    /// `i`, feed `cw[i]` (packed as `len[i]` MSB-first bits) through
    /// `AspxHcb::decode_delta()` and check the recovered delta equals
    /// `i - cb_off`. This is the canonical Huffman contract — if any
    /// transcription slipped this test catches it.
    fn round_trip_codebook(hcb: &AspxHcb) {
        assert_eq!(
            hcb.len.len(),
            hcb.cw.len(),
            "{}: len[] and cw[] length mismatch",
            hcb.name
        );
        for (i, (&l, &cw)) in hcb.len.iter().zip(hcb.cw.iter()).enumerate() {
            assert!(l > 0, "{}: len[{}] = 0", hcb.name, i);
            assert!(l <= 32, "{}: len[{}] = {} (> 32)", hcb.name, i, l);
            let mut bw = BitWriter::new();
            push_hcb_code(&mut bw, l, cw);
            // Pad to byte boundary with zeros (safe: any residual
            // bits after the codeword are irrelevant for this
            // decode since decode_delta stops at the first match).
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let delta = hcb.decode_delta(&mut br).unwrap_or_else(|e| {
                panic!(
                    "{}: decode_delta failed on symbol {} (cw={:#x}, len={}): {:?}",
                    hcb.name, i, cw, l, e
                );
            });
            let expected = i as i32 - hcb.cb_off;
            assert_eq!(
                delta, expected,
                "{}: symbol {} decoded to delta {} (expected {})",
                hcb.name, i, delta, expected
            );
            assert_eq!(
                br.bit_position(),
                l as u64,
                "{}: consumed {} bits, expected {}",
                hcb.name,
                br.bit_position(),
                l
            );
        }
    }

    #[test]
    fn aspx_hcb_env_level_15_f0_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_LEVEL_15_F0);
    }

    #[test]
    fn aspx_hcb_env_level_15_df_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_LEVEL_15_DF);
    }

    #[test]
    fn aspx_hcb_env_level_15_dt_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_LEVEL_15_DT);
    }

    #[test]
    fn aspx_hcb_env_balance_15_f0_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_BALANCE_15_F0);
    }

    #[test]
    fn aspx_hcb_env_balance_15_df_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_BALANCE_15_DF);
    }

    #[test]
    fn aspx_hcb_env_balance_15_dt_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_BALANCE_15_DT);
    }

    #[test]
    fn aspx_hcb_env_level_30_f0_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_LEVEL_30_F0);
    }

    #[test]
    fn aspx_hcb_env_level_30_df_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_LEVEL_30_DF);
    }

    #[test]
    fn aspx_hcb_env_level_30_dt_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_LEVEL_30_DT);
    }

    #[test]
    fn aspx_hcb_env_balance_30_f0_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_BALANCE_30_F0);
    }

    #[test]
    fn aspx_hcb_env_balance_30_df_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_BALANCE_30_DF);
    }

    #[test]
    fn aspx_hcb_env_balance_30_dt_round_trip() {
        round_trip_codebook(&ASPX_HCB_ENV_BALANCE_30_DT);
    }

    #[test]
    fn aspx_hcb_noise_level_f0_round_trip() {
        round_trip_codebook(&ASPX_HCB_NOISE_LEVEL_F0);
    }

    #[test]
    fn aspx_hcb_noise_level_df_round_trip() {
        round_trip_codebook(&ASPX_HCB_NOISE_LEVEL_DF);
    }

    #[test]
    fn aspx_hcb_noise_level_dt_round_trip() {
        round_trip_codebook(&ASPX_HCB_NOISE_LEVEL_DT);
    }

    #[test]
    fn aspx_hcb_noise_balance_f0_round_trip() {
        round_trip_codebook(&ASPX_HCB_NOISE_BALANCE_F0);
    }

    #[test]
    fn aspx_hcb_noise_balance_df_round_trip() {
        round_trip_codebook(&ASPX_HCB_NOISE_BALANCE_DF);
    }

    #[test]
    fn aspx_hcb_noise_balance_dt_round_trip() {
        round_trip_codebook(&ASPX_HCB_NOISE_BALANCE_DT);
    }

    /// Sanity: ASPX_HCB_ALL lines up with the lookup + meta consts —
    /// guards against the table list and `lookup_aspx_hcb()` drifting
    /// apart, and checks every `(codebook_length, cb_off)` matches the
    /// corresponding `AspxHcbMeta` header.
    #[test]
    fn aspx_hcb_all_matches_lookup_and_meta() {
        let metas: &[AspxHcbMeta] = &[
            ASPX_HCB_ENV_LEVEL_15_F0_META,
            ASPX_HCB_ENV_LEVEL_15_DF_META,
            ASPX_HCB_ENV_LEVEL_15_DT_META,
            ASPX_HCB_ENV_BALANCE_15_F0_META,
            ASPX_HCB_ENV_BALANCE_15_DF_META,
            ASPX_HCB_ENV_BALANCE_15_DT_META,
            ASPX_HCB_ENV_LEVEL_30_F0_META,
            ASPX_HCB_ENV_LEVEL_30_DF_META,
            ASPX_HCB_ENV_LEVEL_30_DT_META,
            ASPX_HCB_ENV_BALANCE_30_F0_META,
            ASPX_HCB_ENV_BALANCE_30_DF_META,
            ASPX_HCB_ENV_BALANCE_30_DT_META,
            ASPX_HCB_NOISE_LEVEL_F0_META,
            ASPX_HCB_NOISE_LEVEL_DF_META,
            ASPX_HCB_NOISE_LEVEL_DT_META,
            ASPX_HCB_NOISE_BALANCE_F0_META,
            ASPX_HCB_NOISE_BALANCE_DF_META,
            ASPX_HCB_NOISE_BALANCE_DT_META,
        ];
        assert_eq!(ASPX_HCB_ALL.len(), 18);
        assert_eq!(ASPX_HCB_ALL.len(), metas.len());
        for ((id, hcb), meta) in ASPX_HCB_ALL.iter().zip(metas.iter()) {
            assert_eq!(hcb.name, meta.name);
            assert_eq!(hcb.len.len(), meta.codebook_length as usize);
            assert_eq!(hcb.cw.len(), meta.codebook_length as usize);
            assert_eq!(hcb.cb_off, meta.cb_off);
            let via_lookup = lookup_aspx_hcb(*id);
            assert_eq!(via_lookup.name, hcb.name);
        }
    }

    #[test]
    fn get_aspx_hcb_matches_pseudocode_79() {
        // Sample the full 3x2x2x3 = 36 SIGNAL+NOISE permutations.
        // SIGNAL:
        assert_eq!(
            get_aspx_hcb(
                AspxDataType::Signal,
                AspxQuantStep::Fine,
                AspxStereoMode::Level,
                AspxHcbType::Df
            ),
            HuffmanCodebookId::EnvLevel15Df
        );
        assert_eq!(
            get_aspx_hcb(
                AspxDataType::Signal,
                AspxQuantStep::Coarse,
                AspxStereoMode::Balance,
                AspxHcbType::F0
            ),
            HuffmanCodebookId::EnvBalance30F0
        );
        assert_eq!(
            get_aspx_hcb(
                AspxDataType::Signal,
                AspxQuantStep::Coarse,
                AspxStereoMode::Level,
                AspxHcbType::Dt
            ),
            HuffmanCodebookId::EnvLevel30Dt
        );
        // NOISE ignores quant-mode — every step-mode maps the same way.
        for qm in [AspxQuantStep::Fine, AspxQuantStep::Coarse] {
            assert_eq!(
                get_aspx_hcb(
                    AspxDataType::Noise,
                    qm,
                    AspxStereoMode::Level,
                    AspxHcbType::F0
                ),
                HuffmanCodebookId::NoiseLevelF0
            );
            assert_eq!(
                get_aspx_hcb(
                    AspxDataType::Noise,
                    qm,
                    AspxStereoMode::Balance,
                    AspxHcbType::Dt
                ),
                HuffmanCodebookId::NoiseBalanceDt
            );
        }
    }

    // --- aspx_ec_data end-to-end -------------------------------------

    /// Hand-build a minimal `aspx_ec_data()` payload for one SIGNAL
    /// envelope with FREQ direction and 3 subband groups, then confirm
    /// the walker extracts the three expected deltas and sits at the
    /// right bit position. Uses `ASPX_HCB_ENV_LEVEL_15_{F0,DF}` so the
    /// test exercises the Table 57 / 58 branch covering AC-4's most
    /// common config (1.5 dB step, SIGNAL, LEVEL stereo, FREQ dir).
    #[test]
    fn aspx_ec_data_one_env_freq_dir_signal() {
        // Target deltas:
        //   sbg 0 (F0):      symbol index 40 — from ASPX_HCB_ENV_LEVEL_15_F0
        //                    (cb_off = 0, so decoded delta = 40).
        //   sbg 1 (DF):      symbol index 70 — the zero-delta — from
        //                    ASPX_HCB_ENV_LEVEL_15_DF (cb_off = 70, so
        //                    decoded delta = 0).
        //   sbg 2 (DF):      symbol index 71 — delta +1.
        let f0_cb = &ASPX_HCB_ENV_LEVEL_15_F0;
        let df_cb = &ASPX_HCB_ENV_LEVEL_15_DF;
        let f0_idx = 40usize;
        let df_zero_idx = 70usize;
        let df_plus1_idx = 71usize;
        let mut bw = BitWriter::new();
        bw.write_u32(f0_cb.cw[f0_idx], f0_cb.len[f0_idx] as u32);
        bw.write_u32(df_cb.cw[df_zero_idx], df_cb.len[df_zero_idx] as u32);
        bw.write_u32(df_cb.cw[df_plus1_idx], df_cb.len[df_plus1_idx] as u32);
        let total_bits = f0_cb.len[f0_idx] as u64
            + df_cb.len[df_zero_idx] as u64
            + df_cb.len[df_plus1_idx] as u64;
        bw.align_to_byte();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let out = parse_aspx_ec_data(
            &mut br,
            AspxDataType::Signal,
            1,       // num_env
            &[true], // freq_res[0] = high-res
            AspxQuantStep::Fine,
            AspxStereoMode::Level,
            &[false], // direction[0] = FREQ
            AspxSbgCounts {
                num_sbg_sig_highres: 3,
                num_sbg_sig_lowres: 2,
                num_sbg_noise: 2,
            },
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        let env = &out[0];
        assert!(!env.direction_time);
        assert_eq!(env.values, vec![40, 0, 1]);
        assert_eq!(br.bit_position(), total_bits);
    }

    /// Same as above but for the NOISE path with TIME direction —
    /// exercises the `*_DT` codebook and confirms that NOISE ignores
    /// the caller-passed quant_mode (per Pseudocode 79's NOISE
    /// branch).
    #[test]
    fn aspx_ec_data_two_env_time_dir_noise() {
        let dt_cb = &ASPX_HCB_NOISE_LEVEL_DT;
        // Pick a couple of readily-identifiable symbols.
        //   env 0 sbg 0 -> symbol 29 (DT zero — cb_off = 29, so
        //                   decoded delta = 0).
        //   env 0 sbg 1 -> symbol 30 -> delta +1.
        //   env 1 sbg 0 -> symbol 28 -> delta -1.
        //   env 1 sbg 1 -> symbol 31 -> delta +2.
        let symbols = [29usize, 30usize, 28usize, 31usize];
        let mut bw = BitWriter::new();
        let mut total_bits: u64 = 0;
        for &s in &symbols {
            bw.write_u32(dt_cb.cw[s], dt_cb.len[s] as u32);
            total_bits += dt_cb.len[s] as u64;
        }
        bw.align_to_byte();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let out = parse_aspx_ec_data(
            &mut br,
            AspxDataType::Noise,
            2,   // num_env
            &[], // freq_res ignored for NOISE
            // Flip quant_mode to confirm it's ignored.
            AspxQuantStep::Coarse,
            AspxStereoMode::Level,
            &[true, true], // TIME for both envelopes
            AspxSbgCounts {
                num_sbg_sig_highres: 3,
                num_sbg_sig_lowres: 2,
                num_sbg_noise: 2,
            },
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].direction_time);
        assert_eq!(out[0].values, vec![0, 1]);
        assert!(out[1].direction_time);
        assert_eq!(out[1].values, vec![-1, 2]);
        assert_eq!(br.bit_position(), total_bits);
    }

    // --- §5.7.6.3.1 master-freq-scale derivation ----------------------

    fn cfg_for(scale: AspxMasterFreqScale, start: u8, stop: u8, noise_sbg: u8) -> AspxConfig {
        AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: start,
            stop_freq: stop,
            master_freq_scale: scale,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::Signalled,
        }
    }

    #[test]
    fn master_sbg_table_highres_full_span() {
        let cfg = cfg_for(AspxMasterFreqScale::HighRes, 0, 0, 0);
        let (master, num_master, sba, sbz) = derive_master_sbg_table(&cfg);
        assert_eq!(num_master, 22);
        assert_eq!(sba, 18);
        assert_eq!(sbz, 62);
        assert_eq!(master.len(), 23);
        assert_eq!(master[0], 18);
        assert_eq!(master[22], 62);
        assert_eq!(master[10], 32);
    }

    #[test]
    fn master_sbg_table_lowres_full_span() {
        let cfg = cfg_for(AspxMasterFreqScale::LowRes, 0, 0, 0);
        let (master, num_master, sba, sbz) = derive_master_sbg_table(&cfg);
        assert_eq!(num_master, 20);
        assert_eq!(sba, 10);
        assert_eq!(sbz, 46);
        assert_eq!(master.first(), Some(&10));
        assert_eq!(master.last(), Some(&46));
    }

    #[test]
    fn master_sbg_table_highres_trimmed() {
        let cfg = cfg_for(AspxMasterFreqScale::HighRes, 3, 3, 0);
        let (master, num_master, sba, sbz) = derive_master_sbg_table(&cfg);
        assert_eq!(num_master, 10);
        assert_eq!(sba, 24);
        assert_eq!(sbz, 44);
        assert_eq!(master, vec![24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]);
    }

    #[test]
    fn sig_sbg_tables_xover_zero_on_highres() {
        let cfg = cfg_for(AspxMasterFreqScale::HighRes, 0, 0, 0);
        let (master, num_master, _sba, _sbz) = derive_master_sbg_table(&cfg);
        let sig = derive_sig_sbg_tables(&master, num_master, 0).unwrap();
        assert_eq!(sig.num_sbg_sig_highres, 22);
        assert_eq!(sig.sbx, 18);
        assert_eq!(sig.num_sb_aspx, 62 - 18);
        assert_eq!(sig.sbg_sig_highres, master);
        assert_eq!(sig.num_sbg_sig_lowres, 11);
        assert_eq!(sig.sbg_sig_lowres.len(), 12);
        assert_eq!(sig.sbg_sig_lowres[0], 18);
        for k in 1..=sig.num_sbg_sig_lowres as usize {
            assert_eq!(sig.sbg_sig_lowres[k], sig.sbg_sig_highres[2 * k]);
        }
    }

    #[test]
    fn sig_sbg_tables_xover_odd_branch() {
        let cfg = cfg_for(AspxMasterFreqScale::HighRes, 3, 3, 0);
        let (master, num_master, _sba, _sbz) = derive_master_sbg_table(&cfg);
        assert_eq!(num_master, 10);
        let sig = derive_sig_sbg_tables(&master, num_master, 2).unwrap();
        assert_eq!(sig.num_sbg_sig_highres, 8);
        assert_eq!(sig.sbx, master[2]);
        assert_eq!(sig.num_sb_aspx, master[num_master as usize] - sig.sbx);
        assert_eq!(sig.num_sbg_sig_lowres, 4);
        for k in 1..=sig.num_sbg_sig_lowres as usize {
            assert_eq!(sig.sbg_sig_lowres[k], sig.sbg_sig_highres[2 * k]);
        }

        let sig_odd = derive_sig_sbg_tables(&master, num_master, 3).unwrap();
        assert_eq!(sig_odd.num_sbg_sig_highres, 7);
        assert_eq!(sig_odd.num_sbg_sig_lowres, 4);
        assert_eq!(sig_odd.sbg_sig_lowres[0], sig_odd.sbg_sig_highres[0]);
        for k in 1..=sig_odd.num_sbg_sig_lowres as usize {
            assert_eq!(
                sig_odd.sbg_sig_lowres[k],
                sig_odd.sbg_sig_highres[2 * k - 1]
            );
        }
    }

    #[test]
    fn sig_sbg_tables_xover_exceeds_master_errors() {
        let cfg = cfg_for(AspxMasterFreqScale::HighRes, 3, 3, 0);
        let (master, num_master, _sba, _sbz) = derive_master_sbg_table(&cfg);
        assert!(derive_sig_sbg_tables(&master, num_master, num_master + 1).is_err());
    }

    #[test]
    fn noise_sbg_table_derivation_matches_pseudocode_70() {
        let cfg = cfg_for(AspxMasterFreqScale::HighRes, 0, 0, 3);
        let tables = derive_aspx_frequency_tables(&cfg, 0).unwrap();
        assert_eq!(tables.sbx, 18);
        assert_eq!(tables.sbz, 62);
        assert!(tables.counts.num_sbg_noise <= 5);
        assert_eq!(tables.counts.num_sbg_noise, 5);
        assert_eq!(tables.sbg_noise.len(), 6);
        assert_eq!(tables.sbg_noise[0], tables.sbg_sig_lowres[0]);
    }

    #[test]
    fn noise_sbg_clamps_to_minimum_of_one() {
        let cfg = cfg_for(AspxMasterFreqScale::LowRes, 0, 0, 0);
        let tables = derive_aspx_frequency_tables(&cfg, 0).unwrap();
        assert_eq!(tables.counts.num_sbg_noise, 1);
        assert_eq!(tables.sbg_noise.len(), 2);
        assert_eq!(tables.sbg_noise[0], tables.sbg_sig_lowres[0]);
    }

    /// Exercise a multi-envelope, mixed-resolution SIGNAL ec_data stream.
    ///
    /// Three envelopes with `freq_res = [true, false, true]` force the
    /// walker to pick high-res counts for envs 0 and 2, low-res for env 1.
    /// Verifies the bit cursor advances by exactly the sum of emitted
    /// code lengths.
    #[test]
    fn aspx_ec_data_three_env_mixed_freq_res_bit_accounting() {
        let f0_cb = &ASPX_HCB_ENV_LEVEL_15_F0;
        let df_cb = &ASPX_HCB_ENV_LEVEL_15_DF;
        let counts = AspxSbgCounts {
            num_sbg_sig_highres: 3,
            num_sbg_sig_lowres: 2,
            num_sbg_noise: 1,
        };
        // env 0 (highres, FREQ): F0(30), DF(70), DF(71)
        // env 1 (lowres,  FREQ): F0(31), DF(70)
        // env 2 (highres, FREQ): F0(32), DF(70), DF(70)
        let seq: &[&[(bool, usize)]] = &[
            &[(true, 30), (false, 70), (false, 71)],
            &[(true, 31), (false, 70)],
            &[(true, 32), (false, 70), (false, 70)],
        ];
        let mut bw = BitWriter::new();
        let mut total_bits: u64 = 0;
        for sbgs in seq {
            for &(is_f0, idx) in *sbgs {
                let cb = if is_f0 { f0_cb } else { df_cb };
                bw.write_u32(cb.cw[idx], cb.len[idx] as u32);
                total_bits += cb.len[idx] as u64;
            }
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let out = parse_aspx_ec_data(
            &mut br,
            AspxDataType::Signal,
            3,
            &[true, false, true],
            AspxQuantStep::Fine,
            AspxStereoMode::Level,
            &[false, false, false],
            counts,
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].values, vec![30, 0, 1]);
        assert_eq!(out[1].values, vec![31, 0]);
        assert_eq!(out[2].values, vec![32, 0, 0]);
        assert_eq!(br.bit_position(), total_bits);
    }

    #[test]
    fn aspx_frequency_tables_produce_counts_compatible_with_ec_data() {
        let cfg = cfg_for(AspxMasterFreqScale::LowRes, 2, 1, 1);
        let tables = derive_aspx_frequency_tables(&cfg, 1).unwrap();
        // num_sbg_master = 20 - 4 - 2 = 14.
        assert_eq!(tables.num_sbg_master, 14);
        assert_eq!(tables.counts.num_sbg_sig_highres, 13);
        let f0_cb = &ASPX_HCB_ENV_LEVEL_15_F0;
        let df_cb = &ASPX_HCB_ENV_LEVEL_15_DF;
        let f0_idx = 50usize;
        let df_zero_idx = 70usize;
        let mut bw = BitWriter::new();
        bw.write_u32(f0_cb.cw[f0_idx], f0_cb.len[f0_idx] as u32);
        let mut total_bits = f0_cb.len[f0_idx] as u64;
        for _ in 1..tables.counts.num_sbg_sig_highres {
            bw.write_u32(df_cb.cw[df_zero_idx], df_cb.len[df_zero_idx] as u32);
            total_bits += df_cb.len[df_zero_idx] as u64;
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let out = parse_aspx_ec_data(
            &mut br,
            AspxDataType::Signal,
            1,
            &[true],
            AspxQuantStep::Fine,
            AspxStereoMode::Level,
            &[false],
            tables.counts,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].values.len(),
            tables.counts.num_sbg_sig_highres as usize
        );
        assert_eq!(out[0].values[0], 50);
        for v in &out[0].values[1..] {
            assert_eq!(*v, 0);
        }
        assert_eq!(br.bit_position(), total_bits);
    }

    #[test]
    fn patch_derivation_midrange_highres() {
        // Use a plausible high-res configuration: master_freq_scale
        // highres, sba = 18 (first template entry), sbx mid-range,
        // num_sb_aspx spanning to a reasonable upper band.
        let sbg_master: Vec<u32> = ASPX_SBG_TEMPLATE_HIGHRES.to_vec();
        let num_sbg_master = (sbg_master.len() - 1) as u32;
        let sba = sbg_master[0];
        let sbx = 24;
        // sbz is the last master entry (template tail).
        let sbz = *sbg_master.last().unwrap();
        let num_sb_aspx = sbz - sbx;
        let patches = derive_patch_tables(
            &sbg_master,
            num_sbg_master,
            sba,
            sbx,
            num_sb_aspx,
            true, // 48 kHz
            true, // high-res
        );
        assert!(patches.num_sbg_patches <= 5);
        // sbg_patches must start at sbx and end at sbz.
        assert_eq!(patches.sbg_patches[0], sbx);
        // Not every config reaches sbz exactly (algorithm may trim the
        // tail); but the last border is <= sbz.
        assert!(*patches.sbg_patches.last().unwrap() <= sbz);
    }

    #[test]
    fn hf_tile_copy_copies_patch_source() {
        // Minimal patch config with 1 patch: copy subbands 4..8 from
        // source start 2 into high bands 8..12.
        let patches = AspxPatchTables {
            sbg_patches: vec![8, 12],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![4],
            sbg_patch_start_sb: vec![2],
        };
        let n_ts = 4usize;
        let mut q_low: Vec<Vec<(f32, f32)>> = (0..64).map(|_| vec![(0.0, 0.0); n_ts]).collect();
        // Populate sources 2..6 with distinctive values.
        for src_sb in 2..6 {
            for ts in 0..n_ts {
                q_low[src_sb][ts] = (src_sb as f32, ts as f32);
            }
        }
        let q_high = hf_tile_copy(&q_low, &patches, 8, 64);
        // Expect q_high[8..12] to mirror q_low[2..6].
        for (high_off, &src_sb) in (2..6).collect::<Vec<_>>().iter().enumerate() {
            let hsb = 8 + high_off;
            for ts in 0..n_ts {
                assert_eq!(
                    q_high[hsb][ts], q_low[src_sb as usize][ts],
                    "mismatch at high_sb={hsb} ts={ts}"
                );
            }
        }
        // Below sbx and above patch end should be zeros.
        for sb in 0..8 {
            for ts in 0..n_ts {
                assert_eq!(
                    q_high[sb][ts],
                    (0.0, 0.0),
                    "low-band sb={sb} should be zero"
                );
            }
        }
        for sb in 12..64 {
            for ts in 0..n_ts {
                assert_eq!(
                    q_high[sb][ts],
                    (0.0, 0.0),
                    "high-band sb={sb} should be zero"
                );
            }
        }
    }

    #[test]
    fn hf_regen_produces_non_silent_high_band() {
        // End-to-end sanity: feed PCM through forward QMF, truncate at
        // sbx, run HF tile copy into the high band, and check that the
        // high-band QMF matrix is non-zero. This is the key "ASPX
        // substream should produce actual audio" building block.
        use crate::qmf::{QmfAnalysisBank, NUM_QMF_SUBBANDS};
        let n_slots = 32usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        // Narrow-band signal in the LF range (1 kHz sine at 48 kHz).
        let f = 1000.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let mut ana = QmfAnalysisBank::new();
        let slots = ana.process_block(&pcm);
        // Re-layout: q_low[sb][ts] instead of slot[sb].
        let mut q_low: Vec<Vec<(f32, f32)>> = (0..NUM_QMF_SUBBANDS as u32)
            .map(|_| vec![(0.0, 0.0); n_slots])
            .collect();
        for (ts, slot) in slots.iter().enumerate() {
            for (sb, s) in slot.iter().enumerate() {
                q_low[sb][ts] = *s;
            }
        }
        // Fake patch config: sbx=16, num_sb_aspx=16, one patch of 16.
        let patches = AspxPatchTables {
            sbg_patches: vec![16, 32],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![16],
            sbg_patch_start_sb: vec![0],
        };
        // Truncate the high band first.
        for sb in 16..NUM_QMF_SUBBANDS {
            for s in q_low[sb].iter_mut() {
                *s = (0.0, 0.0);
            }
        }
        let q_high = hf_tile_copy(&q_low, &patches, 16, NUM_QMF_SUBBANDS as u32);
        // Check that q_high[16..32] is non-zero.
        let mut energy = 0.0f64;
        for sb in 16..32 {
            for &(re, im) in q_high[sb].iter() {
                energy += (re as f64) * (re as f64) + (im as f64) * (im as f64);
            }
        }
        assert!(energy > 0.0, "HF regen high band has no energy");
    }

    #[test]
    fn aspx_end_to_end_pipeline_produces_non_silent_pcm() {
        // Full forward + HF regen + inverse pipeline: start from a
        // LOW-band-only signal (1 kHz sine, well below any A-SPX
        // sbx), run it through the analysis bank, truncate the high
        // band, apply tile-copy HF regen via a synthetic patch, then
        // synthesise back to PCM. The output PCM must be non-silent —
        // this is the end-to-end check that the whole bandwidth
        // extension pipeline delivers audio rather than zeros.
        use crate::qmf::{QmfAnalysisBank, QmfSynthesisBank, NUM_QMF_SUBBANDS};
        let n_slots = 60usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let f = 1000.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let mut ana = QmfAnalysisBank::new();
        let slots = ana.process_block(&pcm);
        // Re-layout: q[sb][ts].
        let mut q: Vec<Vec<(f32, f32)>> = (0..NUM_QMF_SUBBANDS as u32)
            .map(|_| vec![(0.0, 0.0); n_slots])
            .collect();
        for (ts, slot) in slots.iter().enumerate() {
            for (sb, s) in slot.iter().enumerate() {
                q[sb][ts] = *s;
            }
        }
        let sbx = 16u32;
        let num_sb_aspx = 16u32;
        let sbz = sbx + num_sb_aspx;
        // Truncate the high band (simulates the core decoder not
        // carrying high-band spectral data past sbx).
        for sb in sbx..NUM_QMF_SUBBANDS as u32 {
            for s in q[sb as usize].iter_mut() {
                *s = (0.0, 0.0);
            }
        }
        // HF regen via tile copy: source band 0..16 → high band 16..32.
        let patches = AspxPatchTables {
            sbg_patches: vec![sbx, sbz],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![num_sb_aspx],
            sbg_patch_start_sb: vec![0],
        };
        let q_high = hf_tile_copy(&q, &patches, sbx, NUM_QMF_SUBBANDS as u32);
        // Merge into q: write the high band back on top of the truncated q.
        for sb in sbx..sbz {
            q[sb as usize] = q_high[sb as usize].clone();
        }
        // Flat envelope gain on the HF range (scaffold for §5.7.6.4.2).
        apply_flat_envelope_gain(&mut q, sbx, sbz, 0.5);
        // Inverse synthesis.
        let mut syn = QmfSynthesisBank::new();
        let mut out = Vec::with_capacity(n);
        for ts in 0..n_slots {
            let mut slot = [(0.0f32, 0.0f32); 64];
            for sb in 0..64 {
                slot[sb] = q[sb][ts];
            }
            let pcm_row = syn.process_slot(&slot);
            out.extend_from_slice(&pcm_row);
        }
        // Output must be non-silent. Steady-state region only.
        let start = 1100usize;
        let end = n - 10;
        let mut energy = 0.0f64;
        let mut nonzero = 0usize;
        for i in start..end {
            let s = out[i] as f64;
            energy += s * s;
            if s != 0.0 {
                nonzero += 1;
            }
        }
        assert!(
            energy > 1e-3,
            "ASPX pipeline produced silent PCM (energy={energy})"
        );
        assert!(
            nonzero > (end - start) / 2,
            "too many zero samples: {nonzero}/{}",
            end - start
        );
    }

    #[test]
    fn flat_envelope_gain_scales_range() {
        let mut q: Vec<Vec<(f32, f32)>> = (0..16).map(|_| vec![(1.0, 2.0); 4]).collect();
        apply_flat_envelope_gain(&mut q, 4, 12, 3.0);
        for sb in 0..4 {
            for ts in 0..4 {
                assert_eq!(q[sb][ts], (1.0, 2.0));
            }
        }
        for sb in 4..12 {
            for ts in 0..4 {
                assert_eq!(q[sb][ts], (3.0, 6.0));
            }
        }
        for sb in 12..16 {
            for ts in 0..4 {
                assert_eq!(q[sb][ts], (1.0, 2.0));
            }
        }
    }

    // ---- §5.7.6.4.2 envelope adjustment tests ----

    #[test]
    fn tab_border_fixfix_matches_table_194() {
        assert_eq!(tab_border_fixfix(16, 1), Some(vec![0, 16]));
        assert_eq!(tab_border_fixfix(16, 2), Some(vec![0, 8, 16]));
        assert_eq!(tab_border_fixfix(16, 4), Some(vec![0, 4, 8, 12, 16]));
        assert_eq!(tab_border_fixfix(12, 4), Some(vec![0, 3, 6, 9, 12]));
        // Non-power-of-two-envelope combinations don't exist.
        assert_eq!(tab_border_fixfix(16, 3), None);
        // Out-of-range num_aspx_timeslots.
        assert_eq!(tab_border_fixfix(10, 2), None);
    }

    #[test]
    fn delta_decode_sig_freq_then_time() {
        // Envelope 0: FREQ direction, deltas [4, -1, 2, 0] →
        // qscf[0..=3] = [4, 3, 5, 5]. Envelope 1: TIME direction,
        // deltas [1, 1, 1, 1] → qscf += 1 per sbg relative to env 0.
        let deltas = vec![
            AspxHuffEnv {
                values: vec![4, -1, 2, 0],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![1, 1, 1, 1],
                direction_time: true,
            },
        ];
        let qscf = delta_decode_sig(&deltas, 4, &[], 1);
        assert_eq!(qscf[0][0], 4);
        assert_eq!(qscf[1][0], 3);
        assert_eq!(qscf[2][0], 5);
        assert_eq!(qscf[3][0], 5);
        assert_eq!(qscf[0][1], 5);
        assert_eq!(qscf[1][1], 4);
        assert_eq!(qscf[2][1], 6);
        assert_eq!(qscf[3][1], 6);
    }

    #[test]
    fn dequantize_sig_scf_fine_3db_matches_formula() {
        // qscf = [[0,2]], qmode=Fine => a=2, num_qmf_subbands=64
        // scf = 64 * 2^(q/2) -> q=0: 64, q=2: 128.
        let qscf = vec![vec![0, 2]];
        let scf = dequantize_sig_scf(&qscf, AspxQuantStep::Fine, &[false, false], 64);
        assert!((scf[0][0] - 64.0).abs() < 1e-4);
        assert!((scf[0][1] - 128.0).abs() < 1e-4);
    }

    #[test]
    fn dequantize_noise_scf_offset_matches_formula() {
        // qscf=[[0,6,12]] -> scf = 2^(6-q) = [64, 1, 1/64]
        let qscf = vec![vec![0, 6, 12]];
        let scf = dequantize_noise_scf(&qscf);
        assert!((scf[0][0] - 64.0).abs() < 1e-4);
        assert!((scf[0][1] - 1.0).abs() < 1e-4);
        assert!((scf[0][2] - (1.0 / 64.0)).abs() < 1e-6);
    }

    #[test]
    fn estimate_envelope_energy_matches_direct_average() {
        // Build a 2-envelope, 1-subband-group test case. q_high is a
        // 4-subband × 4-timeslot matrix (sbx=2, num_sb_aspx=2). Env 0
        // spans ts [0,2), env 1 spans ts [2,4). Subband-group table
        // covers sbx..sbz in a single group.
        let num_sb_aspx = 2u32;
        let sbx = 2u32;
        let sbg_sig = vec![sbx, sbx + num_sb_aspx];
        let atsg_sig = vec![0u32, 1, 2];
        let num_ts_in_ats = 2u32; // -> ts ranges [0,2) and [2,4).
        let mut q_high: Vec<Vec<(f32, f32)>> = (0..4).map(|_| vec![(0.0, 0.0); 4]).collect();
        // Env 0: each of sb 2..4 and ts 0..2 = (1,0) -> |^2 = 1.
        for sb in 2..4 {
            for ts in 0..2 {
                q_high[sb][ts] = (1.0, 0.0);
            }
        }
        // Env 1: each of sb 2..4 and ts 2..4 = (2,0) -> |^2 = 4.
        for sb in 2..4 {
            for ts in 2..4 {
                q_high[sb][ts] = (2.0, 0.0);
            }
        }
        // aspx_interpolation = false -> average over group & time.
        let est = estimate_envelope_energy(
            &q_high,
            &sbg_sig,
            &atsg_sig,
            num_ts_in_ats,
            num_sb_aspx,
            sbx,
            false,
        );
        // Each sb, env 0: sum of 4 contributions / (2 band × 2 ts) = 1
        // Each sb, env 1: sum of 16 / 4 = 4
        for sb in 0..2 {
            assert!((est[sb][0] - 1.0).abs() < 1e-5, "env0 sb{sb} mismatch");
            assert!((est[sb][1] - 4.0).abs() < 1e-5, "env1 sb{sb} mismatch");
        }
    }

    #[test]
    fn compute_sig_gains_equals_sqrt_ratio() {
        // est = 3, scf_sig = 16, scf_noise = 0 -> gain = sqrt(16/(1+3))
        //       = sqrt(4) = 2.
        let est = vec![vec![3.0]];
        let sig = vec![vec![16.0]];
        let noise = vec![vec![0.0]];
        let g = compute_sig_gains(&est, &sig, &noise);
        assert!((g[0][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn envelope_adjustment_follows_per_envelope_gain_profile() {
        // Two-envelope FIXFIX test: build a q_high with a known
        // constant amplitude in the A-SPX range. Provide signal
        // deltas that produce monotone-increasing scf across
        // envelopes. Assert that the per-envelope output energy after
        // AspxEnvelopeAdjuster::apply is higher for the second
        // envelope than the first, matching the expected curve.
        let cfg = AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::High,
        };
        let tables = derive_aspx_frequency_tables(&cfg, 0).unwrap();
        // Layout: num_aspx_timeslots = 16 (matches FIXFIX num_atsg_sig=2),
        // num_ts_in_ats = 1 -> QMF ts=0..16. Build q_high with flat amp.
        let num_aspx_ts = 16u32;
        let num_ts_in_ats = 1u32;
        let n_qmf_ts = (num_aspx_ts * num_ts_in_ats) as usize;
        let mut q: Vec<Vec<(f32, f32)>> = (0..64).map(|_| vec![(0.0, 0.0); n_qmf_ts]).collect();
        // Fill the A-SPX range with a flat unit-amplitude signal.
        for sb in (tables.sbx as usize)..(tables.sbz as usize) {
            for ts in 0..n_qmf_ts {
                q[sb][ts] = (1.0, 0.0);
            }
        }
        // Simulate aspx_data_sig deltas: envelope 0 FREQ with a small
        // base value (q=2), envelope 1 TIME delta +4 on every sbg
        // (q=6). Because dequant is 64 * 2^(q/2), env 1 scf should be
        // 8x env 0 -> gain(env1)^2 ≈ 8x gain(env0)^2. With flat
        // estimates the dominant factor is the sqrt of the scf ratio,
        // i.e. env1 gain ≈ sqrt(8) * env0 gain.
        let num_sbg_sig = tables.counts.num_sbg_sig_highres;
        let sig_deltas = vec![
            AspxHuffEnv {
                values: vec![2; num_sbg_sig as usize],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![4; num_sbg_sig as usize],
                direction_time: true,
            },
        ];
        // Noise floor zero (q=6 gives noise scf=1, i.e. unit).
        let num_sbg_noise = (tables.sbg_noise.len() as u32).saturating_sub(1);
        let noise_deltas = vec![
            AspxHuffEnv {
                values: vec![0; num_sbg_noise as usize],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![0; num_sbg_noise as usize],
                direction_time: true,
            },
        ];
        // Envelope borders from Table 194 (num_aspx_timeslots=16, num_env=2).
        let (atsg_sig, atsg_noise) = derive_fixfix_atsg(num_aspx_ts, 2, 2).unwrap();
        let adjuster = AspxEnvelopeAdjuster::from_deltas(
            &q,
            &tables,
            &sig_deltas,
            &noise_deltas,
            AspxQuantStep::Fine,
            &[false, true],
            &atsg_sig,
            &atsg_noise,
            num_ts_in_ats,
            false,
        );
        adjuster.apply(&mut q);
        // Per-envelope energy in the A-SPX range.
        let (t0_a, t0_b) = (atsg_sig[0] as usize, atsg_sig[1] as usize);
        let (t1_a, t1_b) = (atsg_sig[1] as usize, atsg_sig[2] as usize);
        let mut e0 = 0.0_f64;
        let mut e1 = 0.0_f64;
        for sb in (tables.sbx as usize)..(tables.sbz as usize) {
            for ts in t0_a..t0_b {
                let (re, im) = q[sb][ts];
                e0 += (re as f64) * (re as f64) + (im as f64) * (im as f64);
            }
            for ts in t1_a..t1_b {
                let (re, im) = q[sb][ts];
                e1 += (re as f64) * (re as f64) + (im as f64) * (im as f64);
            }
        }
        // Env 1 should carry *more* energy than env 0 (scf grew 64 ->
        // 64*2^4 across envs). The ratio is approximately 2^(Δq/a)
        // times the same baseline — here Δq = 4, a = 2, so the linear
        // scf ratio is 2^2 = 4; with the sqrt taken by compute_sig_gains
        // the *amplitude* ratio is 2, and the energy ratio is 4.
        assert!(e0 > 0.0, "env 0 energy must be non-zero");
        assert!(e1 > 0.0, "env 1 energy must be non-zero");
        assert!(
            e1 > 3.0 * e0,
            "env 1 energy ({e1}) should be ~4x env 0 energy ({e0})"
        );
    }

    #[test]
    fn apply_envelope_gains_respects_per_envelope_boundaries() {
        // Two envelopes, two subbands (sbx=2, num_sb_aspx=2), flat
        // input (1,0). Gains: env 0 -> [1, 2], env 1 -> [3, 4].
        // Expect per-ts amplitudes mirror those factors.
        let sbx = 2u32;
        let num_sb_aspx = 2u32;
        let atsg_sig = vec![0u32, 2, 4]; // 2 envelopes, 2 A-SPX ts each.
        let num_ts_in_ats = 1u32;
        let mut q: Vec<Vec<(f32, f32)>> = (0..4).map(|_| vec![(1.0, 0.0); 4]).collect();
        let sig_gain_sb = vec![vec![1.0_f32, 3.0], vec![2.0_f32, 4.0]];
        apply_envelope_gains(
            &mut q,
            &sig_gain_sb,
            &atsg_sig,
            num_ts_in_ats,
            sbx,
            num_sb_aspx,
        );
        // sb 2 (= sb 0 relative), env 0 (ts 0..2): gain 1
        for ts in 0..2 {
            assert!((q[2][ts].0 - 1.0).abs() < 1e-6);
        }
        // sb 2, env 1 (ts 2..4): gain 3
        for ts in 2..4 {
            assert!((q[2][ts].0 - 3.0).abs() < 1e-6);
        }
        // sb 3 (= sb 1 relative), env 0: gain 2; env 1: gain 4
        for ts in 0..2 {
            assert!((q[3][ts].0 - 2.0).abs() < 1e-6);
        }
        for ts in 2..4 {
            assert!((q[3][ts].0 - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn map_scf_to_qmf_subbands_broadcasts_within_group() {
        // Two subband groups: [sbx..sbx+2, sbx+2..sbx+4].
        // scf_sig_sbg = [[a_e0, a_e1], [b_e0, b_e1]].
        // Expect scf_sig_sb[0][0]=a_e0, [0][1]=a_e1, [1][0]=a_e0,
        // [1][1]=a_e1, [2][0]=b_e0, [2][1]=b_e1, [3][0]=b_e0,
        // [3][1]=b_e1.
        let sbx = 2u32;
        let sbg_sig = vec![sbx, sbx + 2, sbx + 4];
        let sbg_noise = vec![sbx, sbx + 4];
        let atsg_sig = vec![0u32, 1, 2];
        let atsg_noise = vec![0u32, 1, 2];
        let scf_sig_sbg = vec![vec![10.0_f32, 20.0], vec![30.0, 40.0]];
        let scf_noise_sbg = vec![vec![1.0_f32, 2.0]];
        let out = map_scf_to_qmf_subbands(
            &scf_sig_sbg,
            &scf_noise_sbg,
            &sbg_sig,
            &sbg_noise,
            &atsg_sig,
            &atsg_noise,
            4,
            sbx,
        );
        assert_eq!(out.scf_sig_sb[0][0], 10.0);
        assert_eq!(out.scf_sig_sb[0][1], 20.0);
        assert_eq!(out.scf_sig_sb[1][0], 10.0);
        assert_eq!(out.scf_sig_sb[2][0], 30.0);
        assert_eq!(out.scf_sig_sb[3][0], 30.0);
        assert_eq!(out.scf_noise_sb[0][0], 1.0);
        assert_eq!(out.scf_noise_sb[0][1], 2.0);
        assert_eq!(out.scf_noise_sb[3][1], 2.0);
    }

    // ---- §5.7.6.4.3 + §5.7.6.4.4 noise + tone injection integration ----

    /// End-to-end: run the full A-SPX HF regen pipeline — envelope
    /// adjustment → Pseudocode 92 sine_idx_sb → Pseudocode 94 levels →
    /// Pseudocode 102 noise gen → Pseudocode 104 tone gen → Pseudocodes
    /// 107/108 HF assembler — on a synthetic 1 kHz sine source, and
    /// FFT-probe the output for (a) a tone at the expected HF bin
    /// (dictated by the add_harmonic flag + sb_mid pick) and (b) a
    /// measurable noise floor in the HF range.
    ///
    /// This is the integration checkpoint round 9 was supposed to
    /// land: round 8 produced the envelope-adjusted HF; this test
    /// wires `inject_noise_and_tone` on top and confirms the FFT
    /// signature of the added harmonic + noise floor.
    #[test]
    fn inject_noise_and_tone_adds_tone_and_noise_floor() {
        use crate::qmf::{QmfAnalysisBank, QmfSynthesisBank, NUM_QMF_SUBBANDS};
        // 48 kHz, 1 kHz sine source.
        let fs = 48_000.0_f32;
        let f0 = 1_000.0_f32;
        // 16 A-SPX timeslots × 2 QMF slots each == 32 QMF slots ==
        // 32 * 64 = 2048 PCM samples. Frame length 1920 would put
        // num_ts_in_ats at 2 as well; pick 2048 for cleaner divisions.
        let num_aspx_ts = 16u32;
        let num_ts_in_ats = 2u32;
        let n_slots = (num_aspx_ts * num_ts_in_ats) as usize;
        let n = n_slots * NUM_QMF_SUBBANDS;
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / fs).sin() * 0.5)
            .collect();
        // Forward QMF.
        let mut ana = QmfAnalysisBank::new();
        let slots = ana.process_block(&pcm);
        let mut q: Vec<Vec<(f32, f32)>> = (0..NUM_QMF_SUBBANDS as u32)
            .map(|_| vec![(0.0f32, 0.0f32); n_slots])
            .collect();
        for (ts, slot) in slots.iter().enumerate() {
            for (sb, s) in slot.iter().enumerate() {
                q[sb][ts] = *s;
            }
        }
        // A-SPX frequency tables with HighRes master scale; yields
        // sbx ~16, num_sb_aspx on the order of 20+. xover_offset=0.
        let cfg = AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::High,
        };
        let tables = derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let sbx = tables.sbx as usize;
        let sbz = tables.sbz as usize;
        // Truncate the HF band (core carries only LF past sbx).
        for sb in sbx..NUM_QMF_SUBBANDS {
            for s in q[sb].iter_mut() {
                *s = (0.0, 0.0);
            }
        }
        // Tile-copy HF regeneration using the patch tables.
        let patches = derive_patch_tables(
            &tables.sbg_master,
            tables.num_sbg_master,
            tables.sba,
            tables.sbx,
            tables.num_sb_aspx,
            true,
            true,
        );
        let q_high = hf_tile_copy(&q, &patches, tables.sbx, NUM_QMF_SUBBANDS as u32);
        for sb in sbx..sbz {
            q[sb] = q_high[sb].clone();
        }
        // Build envelope deltas: two signal envelopes, two noise
        // envelopes. Signal scf -> constant, noise scf -> q=6 gives
        // unit scf (2^(6-6) = 1) -> moderate noise floor. Delta_dir
        // all-FREQ (false) so delta_decode_sig path is flat.
        let num_sbg_sig = tables.counts.num_sbg_sig_highres;
        let num_sbg_noise = (tables.sbg_noise.len() as u32).saturating_sub(1);
        let sig_deltas = vec![
            AspxHuffEnv {
                values: vec![4; num_sbg_sig as usize],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![0; num_sbg_sig as usize],
                direction_time: true,
            },
        ];
        let noise_deltas = vec![
            AspxHuffEnv {
                values: vec![6; num_sbg_noise as usize],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![0; num_sbg_noise as usize],
                direction_time: true,
            },
        ];
        let (atsg_sig, atsg_noise) = derive_fixfix_atsg(num_aspx_ts, 2, 2).unwrap();
        let adjuster = AspxEnvelopeAdjuster::from_deltas(
            &q,
            &tables,
            &sig_deltas,
            &noise_deltas,
            AspxQuantStep::Fine,
            &[false; 2],
            &atsg_sig,
            &atsg_noise,
            num_ts_in_ats,
            false,
        );
        adjuster.apply(&mut q);
        // add_harmonic flag: set on the first signal subband group
        // only. Pseudocode 92 places a tone at sb_mid of that group.
        let mut add_harmonic = vec![false; num_sbg_sig as usize];
        if !add_harmonic.is_empty() {
            add_harmonic[0] = true;
        }
        let mut state = AspxChannelExtState::new();
        inject_noise_and_tone(
            &mut q,
            &adjuster,
            &tables,
            &atsg_noise,
            &add_harmonic,
            0,
            &mut state,
        );
        // HF QMF energy must be substantial.
        let mut hf_qmf_energy = 0.0_f64;
        for sb in sbx..sbz {
            for &(re, im) in q[sb].iter() {
                hf_qmf_energy += (re as f64).powi(2) + (im as f64).powi(2);
            }
        }
        assert!(
            hf_qmf_energy > 1.0,
            "HF QMF energy too low after injection: {hf_qmf_energy}"
        );
        // Persistent state must advance.
        assert!(state.sine_idx_sb_prev.is_some());
        assert_eq!(state.num_atsg_sig_prev, 2);
        assert!(state.noise.prev.is_some());
        assert!(state.tone.prev.is_some());
        // Inverse QMF + FFT probe.
        let mut syn = QmfSynthesisBank::new();
        let mut out = Vec::with_capacity(n);
        for ts in 0..n_slots {
            let mut slot = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
            for (sb, s) in slot.iter_mut().enumerate() {
                *s = q[sb][ts];
            }
            let row = syn.process_slot(&slot);
            out.extend_from_slice(&row);
        }
        // Skip QMF group delay.
        let skip = 384_usize.min(out.len().saturating_sub(64));
        let tail = &out[skip..];
        // Compute single-bin DFT at a probe frequency.
        let dft_mag_at = |pcm: &[f32], probe_hz: f64| -> f64 {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (i, &s) in pcm.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * probe_hz * (i as f64) / (fs as f64);
                re += (s as f64) * angle.cos();
                im += (s as f64) * angle.sin();
            }
            (re * re + im * im).sqrt() / (pcm.len() as f64)
        };
        // HF region should have non-trivial broadband energy (noise
        // floor). Probe a few HF points away from the tone and check
        // all are above a threshold, confirming the noise generator
        // is in.
        let f_per_sb = fs / (2.0 * NUM_QMF_SUBBANDS as f32);
        let hf_probe_mid = (sbx as f32 + (sbz - sbx) as f32 / 2.0 - 2.0) * f_per_sb;
        let hf_probe_hi = (sbx as f32 + (sbz - sbx) as f32 / 2.0 + 2.0) * f_per_sb;
        let mag_mid = dft_mag_at(tail, hf_probe_mid as f64);
        let mag_hi = dft_mag_at(tail, hf_probe_hi as f64);
        // And the baseline 1 kHz sine — this shouldn't have
        // significant HF leakage at these bins, so mag_mid/mag_hi
        // being non-trivial confirms the injected contribution.
        let mag_1khz_in = dft_mag_at(&pcm[skip..], 1_000.0);
        // Output RMS must exceed a low-energy threshold.
        let rms =
            (tail.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(
            rms > 0.01,
            "output RMS too low: {rms} (hf_qmf_energy={hf_qmf_energy})"
        );
        assert!(
            mag_mid > 1e-4 && mag_hi > 1e-4,
            "HF noise-floor bins too quiet: mid={mag_mid} hi={mag_hi} (1kHz_in={mag_1khz_in})"
        );
    }

    /// State carries across successive calls: a second run with the
    /// same add_harmonic flags must produce a matrix whose values
    /// differ at selected subbands because the noise-index seed and
    /// sine-index progression have advanced.
    #[test]
    fn inject_noise_and_tone_state_advances_between_calls() {
        let cfg = AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::High,
        };
        let tables = derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let num_aspx_ts = 16u32;
        let num_ts_in_ats = 1u32;
        let n_slots = (num_aspx_ts * num_ts_in_ats) as usize;
        let mut q: Vec<Vec<(f32, f32)>> =
            (0..64).map(|_| vec![(1.0_f32, 0.0_f32); n_slots]).collect();
        let num_sbg_sig = tables.counts.num_sbg_sig_highres;
        let num_sbg_noise = (tables.sbg_noise.len() as u32).saturating_sub(1);
        let sig_deltas = vec![
            AspxHuffEnv {
                values: vec![2; num_sbg_sig as usize],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![0; num_sbg_sig as usize],
                direction_time: true,
            },
        ];
        let noise_deltas = vec![
            AspxHuffEnv {
                values: vec![6; num_sbg_noise as usize],
                direction_time: false,
            },
            AspxHuffEnv {
                values: vec![0; num_sbg_noise as usize],
                direction_time: true,
            },
        ];
        let (atsg_sig, atsg_noise) = derive_fixfix_atsg(num_aspx_ts, 2, 2).unwrap();
        let adjuster = AspxEnvelopeAdjuster::from_deltas(
            &q,
            &tables,
            &sig_deltas,
            &noise_deltas,
            AspxQuantStep::Fine,
            &[false; 2],
            &atsg_sig,
            &atsg_noise,
            num_ts_in_ats,
            false,
        );
        adjuster.apply(&mut q);
        let add_harmonic = vec![true; num_sbg_sig as usize];
        let mut q_a = q.clone();
        let mut q_b = q.clone();
        // Fresh state run.
        let mut state = AspxChannelExtState::new();
        inject_noise_and_tone(
            &mut q_a,
            &adjuster,
            &tables,
            &atsg_noise,
            &add_harmonic,
            0,
            &mut state,
        );
        // Second run — state has advanced; noise_idx_prev and
        // sine_idx_prev non-zero so values at a given (sb, ts) must
        // differ from the fresh-state call.
        inject_noise_and_tone(
            &mut q_b,
            &adjuster,
            &tables,
            &atsg_noise,
            &add_harmonic,
            0,
            &mut state,
        );
        let mut diff_count = 0;
        for sb in (tables.sbx as usize)..(tables.sbz as usize) {
            for ts in 0..n_slots {
                if (q_a[sb][ts].0 - q_b[sb][ts].0).abs() > 1e-6
                    || (q_a[sb][ts].1 - q_b[sb][ts].1).abs() > 1e-6
                {
                    diff_count += 1;
                }
            }
        }
        assert!(
            diff_count > 10,
            "noise/tone state should diverge between calls, got {diff_count} diffs"
        );
    }
}
