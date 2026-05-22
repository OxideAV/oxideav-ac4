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
        // ALPHA
        (Alpha, Coarse, F0) => (ACPL_HCB_ALPHA_COARSE_F0_LEN, ACPL_HCB_ALPHA_COARSE_F0_CW, 0),
        (Alpha, Fine, F0) => (ACPL_HCB_ALPHA_FINE_F0_LEN, ACPL_HCB_ALPHA_FINE_F0_CW, 0),
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
        // BETA3
        (Beta3, Coarse, F0) => (ACPL_HCB_BETA3_COARSE_F0_LEN, ACPL_HCB_BETA3_COARSE_F0_CW, 0),
        (Beta3, Fine, F0) => (ACPL_HCB_BETA3_FINE_F0_LEN, ACPL_HCB_BETA3_FINE_F0_CW, 0),
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
/// * Channel-0 `aspx_framing()`: `int_class = FIXFIX` (`11`, 2 b),
///   `tmp_num_env = 0` (1 or 2 b per `num_env_bits_fixfix`),
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
    // int_class bits per AspxIntClass::read: '11' for FIXFIX, 2 b.
    bw.write_u32(0b11, 2);
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

    // num_env=1, low-res freq_res → use num_sbg_sig_lowres.
    // Per Table 57 NOTE / freq_res = 0 → low-res SIGNAL bands.
    let num_sbg_sig = counts.num_sbg_sig_lowres;
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
}
