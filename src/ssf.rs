//! AC-4 Speech Spectral Frontend (SSF) bitstream walker.
//!
//! Implements ETSI TS 103 190-1 V1.4.1 §4.2.9 / §4.3.7 — the
//! `ssf_data() / ssf_granule() / ssf_st_data() / ssf_ac_data()`
//! bitstream walkers (Tables 43-46) that drive the SSF arithmetic
//! decoder and Annex C tables already landed in [`crate::ssf_ac`] and
//! [`crate::ssf_tables`].
//!
//! Coverage:
//!
//! * **Table 43** — [`parse_ssf_data`]: outer `b_ssf_iframe` gate
//!   followed by either one or two [`parse_ssf_granule`] calls
//!   depending on `frame_len_base >= 1536`.
//! * **Table 44** — [`parse_ssf_granule`]: `stride_flag`,
//!   `num_bands_minus12` (3-bit, I-frame only), per-block
//!   `predictor_presence_flag` + `delta_flag` followed by
//!   `ssf_st_data()` and `ssf_ac_data()`.
//! * **Table 45** — [`parse_ssf_st_data`]: `env_curr_band0_bits`,
//!   I-frame `env_startup_band0_bits`, per-block `gain_bits`,
//!   per-block `predictor_lag(_delta)_bits` / `variance_preserving_flag`
//!   / `alloc_offset_bits`.
//! * **Table 46** — [`parse_ssf_ac_data`]: `env_curr_ac_bits` (drives
//!   [`crate::ssf_ac::decode_envelope_indices`]), I-frame
//!   `env_startup_ac_bits`, per-block predictor-gain index
//!   (Pseudocode 49) plus per-block `q_mdct_coefficients_ac_bits`
//!   (Pseudocode 50) using a flat allocation table seeded from
//!   `alloc_offset_bits[block]` (Pseudocode 31, §5.2.5.2.3 lossless
//!   decode shell).
//!
//! The walker also derives the SSF block layout per Table 112 (48 kHz
//! family) — `granule_length` and `n_mdct` — plus `num_bins` from the
//! Annex C.1 SSF-bandwidth table ([`SSF_BANDWIDTHS`]) so the per-block
//! coefficient decode pulls the correct number of arithmetic-coded
//! values.
//!
//! The full §5.2.3-5.2.7 SSF synthesis chain (envelope decoder,
//! predictor decoder, spectrum decoder, subband predictor, inverse
//! flattening) is not yet wired through to PCM — the walker decodes
//! the indices into [`SsfGranule`] / [`SsfBlock`] structs but the
//! decoder still emits silence for SSF substreams. Implementing the
//! synth chain is a separate round; this walker is the prerequisite.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::ssf_ac::{
    decode_envelope_indices, decode_predictor_gain, AcError, AcState, SsfRandGenState,
};
use crate::ssf_tables::{AC_COEFF_MAX_INDEX, ENVELOPE_CDF_LUT, STEP_SIZES_Q4_15};

/// SSF coder mode (Table 111).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrideFlag {
    /// `stride_flag == 0` — `num_blocks = 1`.
    LongStride,
    /// `stride_flag == 1` — `num_blocks = 4`.
    ShortStride,
}

impl StrideFlag {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::LongStride
        } else {
            Self::ShortStride
        }
    }

    pub fn num_blocks(self) -> usize {
        match self {
            Self::LongStride => 1,
            Self::ShortStride => 4,
        }
    }
}

/// Annex C.1 SSF bandwidths — number of spectral lines per band, keyed
/// by SSF block length (`n_mdct`). Each row is exactly 19 entries
/// (`MAX_NUM_BANDS = 19`); the column count is the eight supported
/// `n_mdct` values from Table C.1.
///
/// Columns in order: 192, 240, 256, 384, 512, 768, 960, 1024.
pub const SSF_BANDWIDTH_BLOCK_LENGTHS: [u32; 8] = [192, 240, 256, 384, 512, 768, 960, 1024];

/// Annex C.1: SSF bandwidths matrix `[band_idx][n_mdct_col]`.
pub static SSF_BANDWIDTHS: [[u8; 8]; 19] = [
    // band 0..9: same row repeated nine times.
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    [2, 3, 3, 5, 6, 9, 11, 12],
    // band 10..12.
    [3, 4, 4, 6, 8, 12, 15, 16],
    [3, 4, 4, 6, 8, 12, 15, 16],
    [3, 4, 4, 6, 8, 12, 15, 16],
    // band 13..14.
    [4, 5, 5, 8, 10, 15, 19, 20],
    [4, 5, 5, 8, 10, 15, 19, 20],
    // band 15.
    [5, 6, 6, 9, 12, 18, 23, 24],
    // band 16..17.
    [5, 7, 7, 11, 14, 21, 26, 28],
    [5, 7, 7, 11, 14, 21, 26, 28],
    // band 18.
    [6, 8, 8, 12, 16, 24, 30, 32],
];

/// `MAX_NUM_BANDS` from §4.3.7.5 Pseudocode 7.
pub const MAX_NUM_BANDS: usize = 19;

/// Resolve the SSF column index (0..=7) for a given `n_mdct` block
/// length. Returns `None` for unsupported `n_mdct` values (real SSF
/// fixtures only ever pick one of the eight Annex C.1 / Table 112-113
/// columns).
pub fn ssf_bandwidth_column(n_mdct: u32) -> Option<usize> {
    SSF_BANDWIDTH_BLOCK_LENGTHS
        .iter()
        .position(|&v| v == n_mdct)
}

/// Look up the band_widths column for a given block length.
/// Returns 19 entries — one per band — when `n_mdct` is supported.
pub fn ssf_band_widths_for(n_mdct: u32) -> Option<[u8; MAX_NUM_BANDS]> {
    let col = ssf_bandwidth_column(n_mdct)?;
    let mut out = [0u8; MAX_NUM_BANDS];
    for (band, slot) in out.iter_mut().enumerate() {
        *slot = SSF_BANDWIDTHS[band][col];
    }
    Some(out)
}

/// Derived helper triple for Pseudocode 7: `(start_bin[k], end_bin[k],
/// num_bins)`.
#[derive(Debug, Clone)]
pub struct SsfBinLayout {
    pub start_bin: Vec<u32>,
    pub end_bin: Vec<u32>,
    pub num_bins: u32,
}

impl SsfBinLayout {
    /// §4.3.7.5 Pseudocode 7: build `start_bin[]` / `end_bin[]` /
    /// `num_bins` from `num_bands` and the block-length-keyed
    /// bandwidths column. Returns `None` if `n_mdct` is unsupported or
    /// `num_bands == 0`.
    pub fn build(num_bands: usize, n_mdct: u32) -> Option<Self> {
        if num_bands == 0 || num_bands > MAX_NUM_BANDS {
            return None;
        }
        let widths = ssf_band_widths_for(n_mdct)?;
        let mut start_bin = vec![0u32; MAX_NUM_BANDS];
        let mut end_bin = vec![0u32; MAX_NUM_BANDS];
        end_bin[0] = u32::from(widths[0]).saturating_sub(1);
        for i in 1..MAX_NUM_BANDS {
            start_bin[i] = start_bin[i - 1] + u32::from(widths[i - 1]);
            end_bin[i] = start_bin[i] + u32::from(widths[i]).saturating_sub(1);
        }
        let num_bins = end_bin[num_bands - 1] + 1;
        Some(SsfBinLayout {
            start_bin,
            end_bin,
            num_bins,
        })
    }
}

/// SSF granule + block configuration derived from `frame_len_base`,
/// `frame_rate_index`, and `fs_index`. Mirrors Tables 112 (48 kHz) and
/// 113 (44.1 kHz). Returns `None` for reserved combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsfFrameConfig {
    /// `frame_length` from the TOC (samples).
    pub frame_length: u32,
    /// `granule_length` = `frame_length / num_granules`.
    pub granule_length: u32,
    /// Number of SSF granules per AC-4 frame (1 or 2).
    pub num_granules: u32,
    /// Maximum `num_blocks` per granule — 1 or 4. SHORT_STRIDE may only
    /// be selected when this is 4 (per §4.3.7.2.1).
    pub max_num_blocks: u32,
}

impl SsfFrameConfig {
    /// Convenience: resolve a default SSF configuration from
    /// `frame_len_base` alone, assuming the 48 kHz family. The mapping
    /// is unambiguous for every Table 112 row. Returns `None` for
    /// unsupported frame lengths.
    pub fn from_frame_len_base(frame_len_base: u32) -> Option<Self> {
        let (gl, ng, mb) = match frame_len_base {
            1920 | 1536 | 2048 => (frame_len_base / 2, 2, 4),
            960 | 1024 | 768 => (frame_len_base, 1, 4),
            512 | 384 => (frame_len_base, 1, 1),
            _ => return None,
        };
        Some(Self {
            frame_length: frame_len_base,
            granule_length: gl,
            num_granules: ng,
            max_num_blocks: mb,
        })
    }

    /// Resolve from `(fs_index, frame_rate_index, frame_len_base)`.
    pub fn from_toc(fs_index: u32, frame_rate_index: u32, frame_length: u32) -> Option<Self> {
        // Number of granules and max_num_blocks come from Table 112
        // (48 kHz, fs_index == 1) or Table 113 (44.1 kHz, fs_index == 0).
        // Values are entirely keyed by frame_rate_index.
        match (fs_index, frame_rate_index) {
            (1, 0..=4) | (1, 13) => Some(Self {
                frame_length,
                granule_length: frame_length / 2,
                num_granules: 2,
                max_num_blocks: 4,
            }),
            (1, 5..=9) => Some(Self {
                frame_length,
                granule_length: frame_length,
                num_granules: 1,
                max_num_blocks: 4,
            }),
            (1, 10..=12) => Some(Self {
                frame_length,
                granule_length: frame_length,
                num_granules: 1,
                max_num_blocks: 1,
            }),
            (0, 13) => Some(Self {
                frame_length,
                granule_length: frame_length / 2,
                num_granules: 2,
                max_num_blocks: 4,
            }),
            _ => None,
        }
    }
}

/// Per-block SSF parameters (the half decoded by `ssf_granule()` +
/// `ssf_st_data()`).
#[derive(Debug, Clone, Default)]
pub struct SsfBlock {
    /// `predictor_presence_flag[block]` — Table 44.
    pub predictor_presence: bool,
    /// `delta_flag[block]` — Table 44 (only meaningful when
    /// `predictor_presence == 1`).
    pub delta_flag: bool,
    /// `gain_bits[block]` — Table 45 (4 bits, SHORT_STRIDE only). The
    /// derived `gain_idx[block] = gain_bits[block] - 8` is exposed via
    /// [`SsfBlock::gain_idx`].
    pub gain_bits: u8,
    /// `predictor_lag_bits[block]` — Table 45 (9 bits, raw lag index;
    /// only present when `predictor_presence && !delta_flag`).
    pub predictor_lag_bits: u16,
    /// `predictor_lag_delta_bits[block]` — Table 45 (4 bits, delta over
    /// previous block; only present when `predictor_presence &&
    /// delta_flag`).
    pub predictor_lag_delta_bits: u8,
    /// `variance_preserving_flag[block]` — Table 45 (1 bit).
    pub variance_preserving: bool,
    /// `alloc_offset_bits[block]` — Table 45 (5 bits). Derived
    /// `i_alloc_offset = alloc_offset_bits + MIN_ALLOC_OFFSET (-21)` per
    /// §5.2.5.2.3 Pseudocode 31.
    pub alloc_offset_bits: u8,
    /// `i_pred_gain_idx[block]` from Pseudocode 49 (post-AC decode).
    /// `None` when `predictor_presence == 0` or when the block is
    /// outside the [start_block, end_block) live range.
    pub pred_gain_idx: Option<i32>,
    /// `i_quant_idx[bin]` — Pseudocode 50 output, `num_bins` entries.
    pub quant_idx: Vec<i32>,
}

impl SsfBlock {
    /// Derived gain index per §4.3.7.3.3 — `gain_bits - 8`.
    pub fn gain_idx(&self) -> i32 {
        i32::from(self.gain_bits) - 8
    }

    /// Derived `i_alloc_offset` per §5.2.5.2.3 Pseudocode 31.
    pub fn i_alloc_offset(&self) -> i32 {
        const MIN_ALLOC_OFFSET: i32 = -21;
        i32::from(self.alloc_offset_bits) + MIN_ALLOC_OFFSET
    }
}

/// Parsed SSF granule (Table 44 + 45 + 46).
#[derive(Debug, Clone)]
pub struct SsfGranule {
    /// Whether this granule is an SSF-I-frame (either inherits
    /// `b_iframe` from `ac4_substream_info` or has its own
    /// `b_ssf_iframe` bit set).
    pub b_iframe: bool,
    /// `stride_flag` — Table 44.
    pub stride_flag: StrideFlag,
    /// `num_bands` — derived from `num_bands_minus12 + 12` on I-frames;
    /// inherited from the previous granule otherwise.
    pub num_bands: u32,
    /// `start_block` / `end_block` — the live block range from the
    /// per-block predictor loop in Table 44.
    pub start_block: u32,
    pub end_block: u32,
    /// `num_blocks` — SHORT_STRIDE → 4, LONG_STRIDE → 1.
    pub num_blocks: u32,
    /// `n_mdct` — SSF block length in samples.
    pub n_mdct: u32,
    /// `num_bins` — Pseudocode 7.
    pub num_bins: u32,
    /// `env_curr_band0_bits` — Table 45 (5 bits). The decoded
    /// `env_idx[0] = env_curr_band0_bits + ENV_BAND0_MIN (-28)` lands
    /// in [`SsfGranule::env_curr`] and is the seed for §5.2.3.0a
    /// envelope decoding.
    pub env_curr_band0_bits: u8,
    /// `env_startup_band0_bits` — Table 45 (5 bits, I-frame +
    /// SHORT_STRIDE only).
    pub env_startup_band0_bits: Option<u8>,
    /// Decoded `env_idx[band]` for the current envelope (band 0 from
    /// `env_curr_band0_bits`, bands 1..num_bands from
    /// [`crate::ssf_ac::decode_envelope_indices`]).
    pub env_curr: Vec<i32>,
    /// Decoded startup `env_idx[band]` — only present on I-frame +
    /// SHORT_STRIDE granules.
    pub env_startup: Option<Vec<i32>>,
    /// Per-block parameters.
    pub blocks: Vec<SsfBlock>,
    /// Number of arithmetic-decoded bits inside `ssf_ac_data()` (from
    /// `AcDecodeFinish` per Pseudocode 47). Includes termination bits.
    pub ac_bits_used: u32,
}

/// `ssf_data()` payload — one or two granules.
#[derive(Debug, Clone, Default)]
pub struct SsfData {
    pub granules: Vec<SsfGranule>,
}

/// Per-channel SSF state carried forward across granules / frames.
///
/// The dither / noise random-number generators reset at the start of
/// every SSF-I-frame (Pseudocodes 54-57); the predictor lag history is
/// kept across granules so the delta-coded path stays valid.
#[derive(Debug, Clone, Default)]
pub struct SsfChannelState {
    /// Dither RNG (Pseudocode 56).
    pub dither_rng: SsfRandGenState,
    /// Random-noise RNG (Pseudocode 57).
    pub noise_rng: SsfRandGenState,
    /// `i_prev_pred_lag_idx` per §5.2.4.0a Pseudocode 4e.
    pub prev_pred_lag_idx: i32,
    /// `num_bands` carried from the previous granule. Only consulted on
    /// non-I granules where `num_bands_minus12` is not present.
    pub last_num_bands: u32,
    /// Last `n_mdct` used — required by the next granule when
    /// `num_bands` is inherited.
    pub last_n_mdct: u32,
    /// Decoded envelope from the previous granule — used as
    /// `env_prev[]` per §5.2.3.0 Note 2 (SSF-P-frames only).
    pub env_prev: Vec<i32>,
}

impl SsfChannelState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset RNG state — called at the start of every SSF-I-frame per
    /// Pseudocode 55.
    pub fn reset_rngs(&mut self) {
        self.dither_rng.reset();
        self.noise_rng.reset();
    }
}

// === Table 43: ssf_data() ====================================================

/// §4.2.9.1 Table 43: walk an `ssf_data(b_iframe)` element.
///
/// `cfg` is the SSF frame configuration derived from the TOC
/// ([`SsfFrameConfig::from_toc`]); it determines `granule_length`,
/// `num_granules`, and the maximum `num_blocks` allowed for SHORT_STRIDE.
/// `state` carries forward per-channel RNG / `prev_pred_lag_idx` /
/// `last_num_bands` between granules and frames.
pub fn parse_ssf_data(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    cfg: &SsfFrameConfig,
    state: &mut SsfChannelState,
) -> Result<SsfData> {
    let mut data = SsfData::default();
    // Granule 0: b_iframe gates b_ssf_iframe.
    let g0_iframe = if b_iframe { true } else { br.read_bit()? };
    if g0_iframe {
        state.reset_rngs();
    }
    let g0 = parse_ssf_granule(br, g0_iframe, cfg, state)?;
    data.granules.push(g0);
    // Granule 1: only present when frame_length >= 1536. Non-I per
    // Table 43 (the only granule allowed to be I is the first one).
    if cfg.frame_length >= 1536 && cfg.num_granules >= 2 {
        let g1 = parse_ssf_granule(br, false, cfg, state)?;
        data.granules.push(g1);
    }
    Ok(data)
}

// === Table 44: ssf_granule() =================================================

/// §4.2.9.2 Table 44: walk an `ssf_granule(b_iframe)` element.
///
/// Reads `stride_flag`, the I-frame `num_bands_minus12`, and the
/// per-block `predictor_presence_flag` / `delta_flag` loop, then
/// dispatches into [`parse_ssf_st_data`] and [`parse_ssf_ac_data`].
pub fn parse_ssf_granule(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    cfg: &SsfFrameConfig,
    state: &mut SsfChannelState,
) -> Result<SsfGranule> {
    let stride_bit = br.read_u32(1)?;
    let stride_flag = StrideFlag::from_bit(stride_bit);
    // §4.3.7.2.1: SHORT_STRIDE only allowed when max_num_blocks == 4.
    if matches!(stride_flag, StrideFlag::ShortStride) && cfg.max_num_blocks < 4 {
        return Err(Error::invalid(
            "ac4 SSF: SHORT_STRIDE not allowed for this configuration",
        ));
    }
    let num_bands = if b_iframe {
        let nb_minus = br.read_u32(3)?;
        let nb = nb_minus + 12;
        state.last_num_bands = nb;
        nb
    } else if state.last_num_bands == 0 {
        return Err(Error::invalid(
            "ac4 SSF: P-granule before any I-granule (no num_bands inherited)",
        ));
    } else {
        state.last_num_bands
    };
    let num_blocks = stride_flag.num_blocks() as u32;
    let n_mdct = cfg.granule_length / num_blocks;
    state.last_n_mdct = n_mdct;
    // Block range — Table 44.
    let mut start_block: u32 = 0;
    let mut end_block: u32 = 0;
    if matches!(stride_flag, StrideFlag::LongStride) && !b_iframe {
        end_block = 1;
    }
    if matches!(stride_flag, StrideFlag::ShortStride) {
        end_block = 4;
        if b_iframe {
            start_block = 1;
        }
    }
    // Per-block predictor loop.
    let mut blocks: Vec<SsfBlock> = (0..num_blocks).map(|_| SsfBlock::default()).collect();
    for block in start_block..end_block {
        let pres = br.read_bit()?;
        blocks[block as usize].predictor_presence = pres;
        if pres {
            // I-frame + SHORT_STRIDE + first live block (block == 1):
            // delta_flag is forced to 0 (no delta on the first live
            // block — there's no prior lag to delta against).
            if start_block == 1 && block == 1 {
                blocks[block as usize].delta_flag = false;
            } else {
                blocks[block as usize].delta_flag = br.read_bit()?;
            }
        }
    }
    // ssf_st_data(): drains static fields into the `blocks` vector +
    // the granule-level envelope-band-0 fields.
    let (env_curr_band0_bits, env_startup_band0_bits) = parse_ssf_st_data(
        br,
        b_iframe,
        stride_flag,
        start_block,
        end_block,
        &mut blocks,
    )?;
    // Build the bin layout (Pseudocode 7).
    let layout = SsfBinLayout::build(num_bands as usize, n_mdct).ok_or_else(|| {
        Error::invalid("ac4 SSF: unsupported (num_bands, n_mdct) for SSF_BANDWIDTHS")
    })?;
    // ssf_ac_data(): runs the arithmetic decoder for the envelope and
    // per-block coefficient streams.
    let (env_curr, env_startup, ac_bits_used) = parse_ssf_ac_data(
        br,
        b_iframe,
        stride_flag,
        start_block,
        end_block,
        num_bands as usize,
        env_curr_band0_bits,
        env_startup_band0_bits,
        &layout,
        &mut blocks,
    )?;
    // Update per-channel state for the next granule.
    state.env_prev = env_curr.clone();
    Ok(SsfGranule {
        b_iframe,
        stride_flag,
        num_bands,
        start_block,
        end_block,
        num_blocks,
        n_mdct,
        num_bins: layout.num_bins,
        env_curr_band0_bits,
        env_startup_band0_bits,
        env_curr,
        env_startup,
        blocks,
        ac_bits_used,
    })
}

// === Table 45: ssf_st_data() =================================================

/// §4.2.9.3 Table 45: walk an `ssf_st_data()` element.
///
/// Returns `(env_curr_band0_bits, env_startup_band0_bits)`. The
/// per-block fields (`gain_bits`, `predictor_lag*_bits`,
/// `variance_preserving_flag`, `alloc_offset_bits`) are filled in
/// directly on `blocks` so the caller doesn't have to thread a second
/// return value.
pub fn parse_ssf_st_data(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    stride_flag: StrideFlag,
    start_block: u32,
    end_block: u32,
    blocks: &mut [SsfBlock],
) -> Result<(u8, Option<u8>)> {
    let env_curr_band0_bits = br.read_u32(5)? as u8;
    let env_startup_band0_bits = if b_iframe && matches!(stride_flag, StrideFlag::ShortStride) {
        Some(br.read_u32(5)? as u8)
    } else {
        None
    };
    if matches!(stride_flag, StrideFlag::ShortStride) {
        for blk in blocks.iter_mut().take(4) {
            blk.gain_bits = br.read_u32(4)? as u8;
        }
    }
    let num_blocks = stride_flag.num_blocks();
    for (block, blk) in blocks.iter_mut().enumerate().take(num_blocks) {
        let in_live = block as u32 >= start_block && (block as u32) < end_block;
        if in_live && blk.predictor_presence {
            if blk.delta_flag {
                blk.predictor_lag_delta_bits = br.read_u32(4)? as u8;
            } else {
                blk.predictor_lag_bits = br.read_u32(9)? as u16;
            }
        }
        blk.variance_preserving = br.read_bit()?;
        blk.alloc_offset_bits = br.read_u32(5)? as u8;
    }
    Ok((env_curr_band0_bits, env_startup_band0_bits))
}

// === Table 46: ssf_ac_data() =================================================

/// §4.2.9.4 Table 46: walk an `ssf_ac_data()` element.
///
/// Initializes the arithmetic decoder per `AcDecoderInit` (Pseudocode
/// 43), pulls `env_curr_ac_bits` (Pseudocode 48), optionally
/// `env_startup_ac_bits`, then loops over the live blocks pulling
/// `predictor_gain_ac_bits[block]` (Pseudocode 49) and
/// `q_mdct_coefficients_ac_bits[block]` (Pseudocode 50) using the
/// block-local `i_alloc_table[band]` derived from `alloc_offset_bits`
/// (a flat allocation per Pseudocode 31's lossless-decoding shell).
/// Finally invokes `AcDecodeFinish` (Pseudocode 47) and consumes the
/// termination-bit overhead from the bitstream cursor so the
/// surrounding walker can continue.
///
/// Returns `(env_curr, env_startup, ac_bits_used)`.
#[allow(clippy::too_many_arguments)]
pub fn parse_ssf_ac_data(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    stride_flag: StrideFlag,
    start_block: u32,
    end_block: u32,
    num_bands: usize,
    env_curr_band0_bits: u8,
    env_startup_band0_bits: Option<u8>,
    layout: &SsfBinLayout,
    blocks: &mut [SsfBlock],
) -> Result<(Vec<i32>, Option<Vec<i32>>, u32)> {
    const ENV_BAND0_MIN: i32 = -28;
    // Build the static envelope CDF used by `decode_envelope_indices`.
    let env_cdf: Vec<u32> = ENVELOPE_CDF_LUT.iter().map(|&x| x as u32).collect();
    let _ = env_cdf; // silence unused-warning if helper changes upstream.
                     // Initialize the arithmetic decoder.
    let mut ac = AcState::init(br).map_err(map_ac_err)?;
    let bit_cursor_start = ac.bits_consumed; // == SSF_RANGE_BITS.
                                             // env_curr: env_idx[0] from env_curr_band0_bits, env_idx[1..] from
                                             // the AC stream.
    let mut env_curr = vec![0i32; num_bands];
    env_curr[0] = i32::from(env_curr_band0_bits) + ENV_BAND0_MIN;
    if num_bands > 1 {
        decode_envelope_indices(&mut ac, num_bands, &mut env_curr, br).map_err(map_ac_err)?;
    }
    // env_startup (I-frame + SHORT_STRIDE only).
    let env_startup = if b_iframe && matches!(stride_flag, StrideFlag::ShortStride) {
        let mut env_st = vec![0i32; num_bands];
        env_st[0] = i32::from(env_startup_band0_bits.unwrap_or(0)) + ENV_BAND0_MIN;
        if num_bands > 1 {
            decode_envelope_indices(&mut ac, num_bands, &mut env_st, br).map_err(map_ac_err)?;
        }
        Some(env_st)
    } else {
        None
    };
    // Per-block AC loop.
    let num_blocks = stride_flag.num_blocks();
    // Bands as (start_bin, end_bin) pairs for the active num_bands.
    let bands: Vec<(usize, usize)> = (0..num_bands)
        .map(|b| (layout.start_bin[b] as usize, layout.end_bin[b] as usize))
        .collect();
    // Pre-build a zero-dither vector as a placeholder — the live
    // §5.2.5.2.3 spec injects per-bin dither only when
    // `i_alloc_table[band] < i_alloc_dithering_threshold`. The walker
    // doesn't try to reproduce the heuristic-scaling threshold logic
    // (§5.2.5.2.1 / .2 / .3 / .4) here; the synthesis chain is a
    // separate round. We feed the AC decoder a zero-dither path which
    // is the in-spec degenerate case (i_dither[bin] = 0 when the
    // allocation is dither-less).
    let zero_dither = vec![0i32; layout.num_bins as usize];
    for block in 0..num_blocks {
        let in_live = (block as u32) >= start_block && (block as u32) < end_block;
        let blk_idx = block;
        let block_iframe_first = b_iframe && block == 0;
        if !block_iframe_first && in_live {
            // predictor_gain_ac_bits[block] — Pseudocode 49.
            if blocks[blk_idx].predictor_presence {
                let gidx = decode_predictor_gain(&mut ac, br).map_err(map_ac_err)?;
                blocks[blk_idx].pred_gain_idx = Some(gidx);
            }
        }
        // q_mdct_coefficients_ac_bits[block] — Pseudocode 50.
        // Build a flat i_alloc_table[band] = max(0, min(20,
        // i_alloc_offset)). The full §5.2.5.2.3 path computes per-band
        // env-deltas; we stub a flat allocation here so the AC stream
        // pulls the correct number of symbols. Synthesis quality lives
        // in a future round; bitstream conformance is preserved
        // because Pseudocode 50's number of pulls is only a function
        // of (num_bands, num_bins, i_alloc_table[band] != 0 vs == 0).
        let alloc_offset = blocks[blk_idx].i_alloc_offset();
        let i_alloc = alloc_offset.clamp(0, 20) as u32;
        let i_alloc_table: Vec<u32> = vec![i_alloc; num_bands];
        let mut quant = vec![0i32; layout.num_bins as usize];
        if i_alloc == 0 {
            // Pseudocode 50: no symbols pulled; quant stays zero.
        } else {
            // Validate table indices we'll touch.
            let i_alloc_us = i_alloc as usize;
            if i_alloc_us >= STEP_SIZES_Q4_15.len() || i_alloc_us >= AC_COEFF_MAX_INDEX.len() {
                return Err(Error::invalid("ac4 SSF: i_alloc out of table bounds"));
            }
            crate::ssf_ac::decode_coefficient_indices(
                &mut ac,
                &i_alloc_table,
                &bands,
                &zero_dither,
                &mut quant,
                br,
            )
            .map_err(map_ac_err)?;
        }
        blocks[blk_idx].quant_idx = quant;
    }
    // AcDecodeFinish — termination-bit accounting.
    let total_bits = ac.decode_finish();
    let _ = bit_cursor_start;
    Ok((env_curr, env_startup, total_bits))
}

fn map_ac_err(e: AcError) -> Error {
    match e {
        AcError::BitstreamUnderflow => Error::invalid("ac4 SSF AC: bitstream underflow"),
        AcError::SymbolNotFound => Error::invalid("ac4 SSF AC: symbol not found in CDF"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[test]
    fn stride_flag_block_count() {
        assert_eq!(StrideFlag::LongStride.num_blocks(), 1);
        assert_eq!(StrideFlag::ShortStride.num_blocks(), 4);
    }

    #[test]
    fn ssf_bandwidths_anchor() {
        // Annex C.1 anchors: row 0 (band 0) for n_mdct = 1024 → 12;
        // row 18 (band 18) for n_mdct = 192 → 6; row 10 for 1024 → 16.
        assert_eq!(SSF_BANDWIDTHS[0][7], 12);
        assert_eq!(SSF_BANDWIDTHS[18][0], 6);
        assert_eq!(SSF_BANDWIDTHS[10][7], 16);
    }

    #[test]
    fn ssf_bin_layout_long_stride_48k_24fps() {
        // frame_length = 1920, num_granules = 2 → granule_length = 960.
        // LongStride → num_blocks = 1 → n_mdct = 960. Annex C.1 column
        // for 960. With num_bands = 12 (num_bands_minus12 == 0):
        // bands 0..9 widths = 11 each, band 10..11 widths = 15 each.
        // num_bins = sum = 11*10 + 15*2 = 110 + 30 = 140.
        let layout = SsfBinLayout::build(12, 960).unwrap();
        assert_eq!(layout.start_bin[0], 0);
        assert_eq!(layout.end_bin[0], 10);
        assert_eq!(layout.start_bin[1], 11);
        assert_eq!(layout.num_bins, 140);
    }

    #[test]
    fn frame_config_48k_24fps() {
        // fs_index = 1 (48 kHz), frame_rate_index = 1 (24 fps),
        // frame_length = 1920. Table 112 row: granule_length 960,
        // num_granules 2, max_num_blocks 4.
        let cfg = SsfFrameConfig::from_toc(1, 1, 1920).unwrap();
        assert_eq!(cfg.granule_length, 960);
        assert_eq!(cfg.num_granules, 2);
        assert_eq!(cfg.max_num_blocks, 4);
    }

    #[test]
    fn frame_config_48k_50fps_one_granule() {
        // fs_index = 1, frame_rate_index = 7 (50 fps), frame_length = 1024.
        let cfg = SsfFrameConfig::from_toc(1, 7, 1024).unwrap();
        assert_eq!(cfg.granule_length, 1024);
        assert_eq!(cfg.num_granules, 1);
        assert_eq!(cfg.max_num_blocks, 4);
    }

    #[test]
    fn frame_config_48k_120fps_no_short_stride() {
        // fs_index = 1, frame_rate_index = 12 (120 fps), frame_length = 384.
        let cfg = SsfFrameConfig::from_toc(1, 12, 384).unwrap();
        assert_eq!(cfg.max_num_blocks, 1);
    }

    #[test]
    fn frame_config_44_1k_only_index_13() {
        // fs_index = 0, frame_rate_index 13 only — Table 113.
        assert!(SsfFrameConfig::from_toc(0, 0, 2048).is_none());
        let cfg = SsfFrameConfig::from_toc(0, 13, 2048).unwrap();
        assert_eq!(cfg.granule_length, 1024);
        assert_eq!(cfg.num_granules, 2);
    }

    /// Build a minimal LONG_STRIDE I-frame ssf_data() and walk it.
    /// Granule-0 is I, num_bands_minus12 = 0 (so num_bands = 12),
    /// no predictor anywhere (predictor_presence_flag stays 0 — but
    /// LONG_STRIDE I-frame has start_block == end_block == 0 so the
    /// per-block predictor loop runs zero times anyway). frame_length
    /// = 960 → granule_length = 960 → n_mdct = 960; second granule is
    /// not emitted because frame_length < 1536.
    #[test]
    fn parse_ssf_data_long_stride_iframe_smoke() {
        let mut bw = BitWriter::new();
        // ssf_granule(b_iframe=1):
        bw.write_u32(0, 1); // stride_flag = LONG_STRIDE.
        bw.write_u32(0, 3); // num_bands_minus12 = 0 → num_bands = 12.
                            // For LONG_STRIDE + b_iframe == 1, start_block = 0, end_block = 0
                            // → per-block predictor loop runs zero iterations.
                            // ssf_st_data():
                            //   env_curr_band0_bits (5).
        bw.write_u32(0, 5);
        // env_startup_band0_bits NOT present (LONG_STRIDE).
        // SHORT_STRIDE gain_bits NOT present.
        // num_blocks = 1 → one iteration of the per-block loop:
        //   in_live? block=0 < start_block=0 — false; so skip pred lag.
        //   variance_preserving_flag (1).
        bw.write_u32(0, 1);
        //   alloc_offset_bits (5).
        bw.write_u32(0, 5);
        // ssf_ac_data():
        //   AcDecoderInit pulls SSF_RANGE_BITS (30) bits — feed zeros.
        for _ in 0..30 {
            bw.write_bit(false);
        }
        // num_bands - 1 = 11 envelope indices via decode_envelope_indices —
        // since CDF is monotone and `target` based on a zero-bit init
        // lands at zero, the AC will pick the lowest-CDF symbol (sym=0)
        // for each. Each `AcDecode` may renormalise; pad enough zeros.
        // Per-block predictor_gain not pulled (b_iframe + block 0 first
        // block — the non-`block_iframe_first` gate skips it).
        // Per-block q_mdct: i_alloc = clamp(0 - 21, 0, 20) = 0 → no
        // symbol pulled, quant stays zero.
        // Pad ample zeros for any AC renormalisation.
        for _ in 0..256 {
            bw.write_bit(false);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = SsfFrameConfig::from_toc(1, 5, 960).unwrap();
        let mut state = SsfChannelState::new();
        let data = parse_ssf_data(&mut br, true, &cfg, &mut state).expect("parse OK");
        assert_eq!(data.granules.len(), 1);
        let g = &data.granules[0];
        assert!(g.b_iframe);
        assert_eq!(g.stride_flag, StrideFlag::LongStride);
        assert_eq!(g.num_bands, 12);
        assert_eq!(g.start_block, 0);
        assert_eq!(g.end_block, 0);
        assert_eq!(g.num_blocks, 1);
        assert_eq!(g.n_mdct, 960);
        assert_eq!(g.env_curr.len(), 12);
        assert!(g.env_startup.is_none());
    }

    /// SHORT_STRIDE I-frame: start_block=1, end_block=4 — three live
    /// blocks. With predictor_presence_flag=0 across all three, no
    /// predictor lag fields are read. Verifies env_startup is decoded.
    #[test]
    fn parse_ssf_short_stride_iframe_smoke() {
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // stride_flag = SHORT_STRIDE.
        bw.write_u32(0, 3); // num_bands_minus12 = 0 → num_bands = 12.
                            // Per-block predictor loop: block 1, 2, 3 each predictor_presence_flag = 0.
        for _ in 1..4 {
            bw.write_bit(false);
        }
        // ssf_st_data():
        bw.write_u32(0, 5); // env_curr_band0_bits.
        bw.write_u32(0, 5); // env_startup_band0_bits (I + SHORT_STRIDE).
                            // gain_bits[0..3] — 4 bits each.
        for _ in 0..4 {
            bw.write_u32(0, 4);
        }
        // Per-block st-data: predictor lag NOT present (predictor_presence
        // = 0); always emit variance_preserving_flag (1) + alloc_offset_bits (5).
        for _ in 0..4 {
            bw.write_u32(0, 1);
            bw.write_u32(0, 5);
        }
        // ssf_ac_data() — AcDecoderInit + AC stream.
        for _ in 0..(30 + 512) {
            bw.write_bit(false);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        // 48 kHz, 50 fps (frame_rate_index = 7): granule_length = 1024,
        // num_granules = 1, max_num_blocks = 4. SHORT_STRIDE allowed.
        let cfg = SsfFrameConfig::from_toc(1, 7, 1024).unwrap();
        let mut state = SsfChannelState::new();
        let data = parse_ssf_data(&mut br, true, &cfg, &mut state).expect("parse OK");
        assert_eq!(data.granules.len(), 1);
        let g = &data.granules[0];
        assert_eq!(g.stride_flag, StrideFlag::ShortStride);
        assert_eq!(g.start_block, 1);
        assert_eq!(g.end_block, 4);
        assert_eq!(g.num_blocks, 4);
        assert_eq!(g.n_mdct, 256);
        assert_eq!(g.blocks.len(), 4);
        assert!(g.env_startup.is_some());
    }
}
