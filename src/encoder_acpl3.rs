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
    dp_optimise_sections, pick_best_codebook_for_band, write_scalefac_data, write_section_data,
    write_snf_data, write_spectral_data_sections,
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
    let xover: u32 = 0;
    bw.write_u32(xover, 3);

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
    // ch1 SIGNAL: stereo_mode = BALANCE when balance = 1 else LEVEL.
    let qmode_ch1 = qmode_ch0; // shared framing
    if num_sbg_sig >= 1 {
        write_aspx_sig_f0(bw, qmode_ch1, aspx::AspxStereoMode::Balance);
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
    // ch1 NOISE: stereo_mode = BALANCE.
    if num_sbg_noise >= 1 {
        write_aspx_noise_f0(bw, aspx::AspxStereoMode::Balance);
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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
    let xover: u32 = 0;
    bw.write_u32(xover, 3);

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

    // aspx_hfgen_iwc_1ch(): tna_mode[0..num_sbg_noise] = 0 (2 b each) +
    // ah_present / fic_present / tic_present = 0 (3 × 1 b).
    for _ in 0..counts.num_sbg_noise {
        bw.write_u32(0, 2);
    }
    bw.write_bit(false); // ah_present = 0
    bw.write_bit(false); // fic_present = 0
    bw.write_bit(false); // tic_present = 0

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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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
fn quantise_alpha(alpha: f32, qm: crate::acpl::AcplQuantMode) -> i32 {
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
fn quantise_beta_magnitude(beta_mag: f32, qm: crate::acpl::AcplQuantMode) -> i32 {
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
fn write_acpl_data_1ch_real_alpha_beta(
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
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

    bw.align_to_byte();
    while bw.byte_len() < pad_target_bytes {
        bw.write_u32(0, 8);
    }
    let mut bytes = bw.finish();
    if bytes.len() > pad_target_bytes {
        bytes.truncate(pad_target_bytes);
    }
    bytes
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitReader;

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
}
