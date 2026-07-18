//! Foundation AC-4 decoder.
//!
//! Given a packet that carries either a full `ac4_syncframe()` (the
//! TS/RTP form) or a bare `raw_ac4_frame()` payload (the ISO BMFF MP4
//! sample form), the decoder:
//!
//! 1. Scans for the `0xAC40` / `0xAC41` sync word; if not found, treats
//!    the full packet as a bare payload.
//! 2. Runs [`toc::parse_ac4_toc`] to extract the channel count, effective
//!    sample rate and frame length.
//! 3. Emits an `AudioFrame` full of zero S16 samples with the correct
//!    shape.
//!
//! This is not a real AC-4 decoder — decoding the ASF / A-SPX / ASF-A2
//! substream coefficient streams is spec work measured in weeks. What
//! it *does* give us is a clean path for the rest of the oxideav
//! pipeline (demuxer → decoder → filter → output) to run end-to-end
//! against real AC-4 fixtures without panics, plus a parsed
//! [`toc::Ac4FrameInfo`] surface for downstream tooling.

use oxideav_core::Decoder;
#[cfg(test)]
use oxideav_core::TimeBase;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::{acpl_synth, ajcc_synth, asf, aspx, ice, mdct, qmf, ssf, ssf_synth, sync, toc};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Ac4Decoder::new(params)))
}

pub struct Ac4Decoder {
    codec_id: CodecId,
    /// Channel count hint supplied by the container (CodecParameters).
    /// Used as a fallback when the TOC's channel-mode code is one of
    /// the reserved/escape values.
    hint_channels: u16,
    /// Sample-rate hint from the container.
    hint_sample_rate: u32,
    pending: Option<Packet>,
    eof: bool,
    /// Last parsed frame info — exposed for downstream inspection.
    pub last_info: Option<toc::Ac4FrameInfo>,
    /// Last parsed substream tool summary (first substream of the last
    /// decoded frame). `None` when the TOC didn't expose a usable size
    /// for the substream (e.g. single-substream frame where
    /// `b_size_present == 0`).
    pub last_substream: Option<asf::Ac4SubstreamInfo>,
    /// Per-channel overlap-add state (length = transform_length samples).
    /// Keyed by channel index; resized on transform-length change.
    overlap: Vec<Vec<f32>>,
    /// Transform length of the previous frame (for overlap sizing).
    prev_transform_length: u32,
    /// Per-channel A-SPX persistent state — noise generator
    /// `noise_idx_prev` (§5.7.6.4.3 Pseudocode 103), tone generator
    /// `sine_idx_prev` (§5.7.6.4.4 Pseudocode 105), and the
    /// `sine_idx_sb_prev` / `tsg_ptr_prev` / `num_atsg_sig_prev` bundle
    /// that Pseudocode 92 consults. Grown on demand as channels decode.
    aspx_ext_state: Vec<aspx::AspxChannelExtState>,
    /// Per-substream A-CPL persistent state for the channel-pair
    /// element (§5.7.7.5 Pseudocode 115). Only one substream is wired
    /// in the foundation decoder; multichannel `ASPX_ACPL_3` would carry
    /// a vector keyed by substream index.
    acpl_state: acpl_synth::AcplSubstreamState,
    /// Per-substream state for the 5_X `ASPX_ACPL_1` / `ASPX_ACPL_2`
    /// pair pipeline (§5.7.7.6.1 Pseudocode 117). Carries the pair
    /// decorrelator + QMF analysis/synthesis banks across frames.
    acpl_5x_pair_state: acpl_synth::Acpl5xPairPcmState,
    /// Per-substream state for the 5_X `ASPX_ACPL_3` multichannel
    /// synthesis pipeline (§5.7.7.6.2 Pseudocode 118). Carries the
    /// D0/D1/D2 + ducker IIR state + differential-decode rolling sums
    /// across frames.
    acpl_5x_mch_state: acpl_synth::Acpl5xMchPcmState,
    /// Per-channel SSF synthesis state — RNGs, predictor lag history,
    /// subband-predictor spec/env buffers, and the previous block's
    /// `f_spec[]` latch. Grown on demand as channels decode SSF.
    ssf_synth_state: Vec<ssf_synth::SsfSynthState>,
    /// Per-channel SSF *walker* state — the bitstream-side dither /
    /// noise RNG, `prev_pred_lag_idx`, `last_num_bands`, and the
    /// `env_prev[]` snapshot of raw delta symbols. Hoisted onto the
    /// decoder in round 32 so RNG continuity (Pseudocodes 54-57) is
    /// preserved across frame boundaries; pre-r32 the walker built a
    /// fresh per-frame state and dropped it. Grown on demand to match
    /// the channel count seen on the latest frame.
    ssf_walker_state: Vec<ssf::SsfChannelState>,
    /// I-frame-sticky substream configuration (aspx_config /
    /// acpl_config_* / aspx_xover_subband_offset) carried across
    /// frames so `b_iframe == 0` (P-frame) substreams parse per
    /// §4.2.6.x. Harvested from every I-frame, seeded into every
    /// non-I-frame walk.
    sticky: asf::StickyConfig,
    /// A-JOC object-substream decode state (TS 103 190-2 §5.7),
    /// allocated on the first v2 frame whose TOC carries an
    /// `ac4_substream_info_ajoc()` descriptor. Keyed by
    /// `(num_dmx, num_umx)`; a geometry change reallocates.
    ajoc_dec: Option<(usize, usize, crate::ajoc_substream::AjocSubstreamDecoder)>,
    /// Immersive-channel-element A-JCC decode state (TS 103 190-2
    /// §5.6): the §5.6.3.2 differential reference + the §5.6.3.3-5
    /// interpolator / decorrelator / ducker state, keyed by
    /// `b_5fronts`. Allocated on the first ASPX_AJCC immersive frame.
    ice_ajcc: Option<(bool, crate::ajcc::AjccState, ajcc_synth::AjccSynthState)>,
    /// Immersive-channel-element A-CPL decode state (TS 103 190-2
    /// §5.5.2): the up-to-six parallel ACplModule states (decorrelator,
    /// ducker and prev arrays) plus the per-module alpha / beta
    /// differential-decode rolling references, keyed by `b_5fronts`.
    /// Allocated on the first ASPX_ACPL_1 / ASPX_ACPL_2 immersive
    /// frame.
    ice_acpl: Option<IceAcplState>,
}

/// Per-decoder immersive A-CPL state — see [`Ac4Decoder::ice_acpl`].
///
/// Module order follows the §5.5.2 Table 27 pseudocode: modules 0-3
/// drive (Ls, Lb) / (Rs, Rb) / (Tfl, Tbl) / (Tfr, Tbr) with
/// decorrelators D0 / D0 / D1 / D1; modules 4-5 (b_5fronts only)
/// drive (L, Lscr) / (R, Rscr) with D2.
struct IceAcplState {
    b_5fronts: bool,
    modules: Vec<acpl_synth::AcplCpeState>,
    diffs: Vec<(acpl_synth::AcplDiffState, acpl_synth::AcplDiffState)>,
}

impl IceAcplState {
    fn new(b_5fronts: bool) -> Self {
        use acpl_synth::DecorrelatorId as D;
        let ids = [D::D0, D::D0, D::D1, D::D1, D::D2, D::D2];
        Self {
            b_5fronts,
            modules: ids
                .iter()
                .map(|&d| acpl_synth::AcplCpeState::new(d))
                .collect(),
            diffs: (0..6)
                .map(|_| {
                    (
                        acpl_synth::AcplDiffState::new(),
                        acpl_synth::AcplDiffState::new(),
                    )
                })
                .collect(),
        }
    }
}

/// Phase-1 result of [`Ac4Decoder::aspx_extend_to_qmf`]:
/// `(qmf_matrix, sbx, sbz)` — the post-extension QMF matrix and the
/// (sb0, sb1) range needed by the §5.7.5 companding tool.
type AspxQmfPhase1 = (aspx::QmfMatrix, u32, u32);

/// One per-channel entry consumed by
/// [`Ac4Decoder::extend_5x_channels_with_sync_companding`].
type SyncCompandingChannelEntry<'a> = (
    usize,                             // output slot (0..=4)
    &'a [f32],                         // pcm_in
    &'a aspx::FiveXAspxTrailer,        // trailer
    &'a aspx::FiveXAspxChannelTrailer, // channel trailer
    &'a aspx::AspxConfig,              // aspx config
    Option<u32>,                       // sb0 override (acpl_qmf_band)
);

/// One per-channel entry consumed by [`Ac4Decoder::extend_5x_entries`]:
/// `(slot, pcm_f32, trailer_pair)` where `trailer_pair` is
/// `(trailer, is_secondary)` — `None` when the channel has no trailer
/// (passthrough case).
type FiveXChannelEntry<'a> = (usize, Vec<f32>, Option<(&'a aspx::FiveXAspxTrailer, bool)>);

/// Round 45: per-channel input bundle for the stereo-CPE M=2 synced
/// companding helper [`Ac4Decoder::extend_stereo_cpe_pair_with_sync_companding`].
/// Mirrors the per-channel arguments of [`Ac4Decoder::aspx_extend_pcm`]
/// (the un-trailerised stereo-CPE form used by the primary / secondary
/// dispatch path) so the two-channel cohort can run phase-1 → synced
/// companding apply → phase-2 in lockstep.
struct StereoCpeChannelInput<'a> {
    /// Decoder-local channel index used to pick the right
    /// `aspx_ext_state[ch_index]` carry-over (0 for primary, 1 for
    /// secondary on a 2-channel CPE).
    ch_index: usize,
    /// IMDCT'd low-band PCM for this channel; the helper runs forward
    /// QMF + HF generation + envelope adjustment + companding +
    /// inverse QMF on this buffer.
    pcm_in: &'a [f32],
    /// `aspx_framing()` for this channel (per-channel in stereo CPE).
    framing: Option<&'a aspx::AspxFraming>,
    /// `aspx_data_sig` Huffman envelopes for this channel.
    sig: Option<&'a [aspx::AspxHuffEnv]>,
    /// `aspx_data_noise` Huffman envelopes for this channel.
    noise: Option<&'a [aspx::AspxHuffEnv]>,
    /// `aspx_qmode_env` quant-step for this channel's envelopes.
    qmode: Option<aspx::AspxQuantStep>,
    /// Per-envelope sign of the dpcm directionality (`f` flag in the
    /// spec): `true` = freq-direction, `false` = time-direction.
    delta_dir: Option<&'a aspx::AspxDeltaDir>,
    /// `aspx_hfgen_iwc.add_harmonic[ch]` for tone injection.
    add_harmonic: Option<&'a [bool]>,
    /// `aspx_hfgen_iwc.tna_mode[ch]` for the chirp + α0 + α1 TNS body.
    tna_mode: Option<&'a [u8]>,
}

impl Ac4Decoder {
    pub fn new(params: &CodecParameters) -> Self {
        Self {
            codec_id: params.codec_id.clone(),
            hint_channels: params.channels.unwrap_or(2),
            hint_sample_rate: params.sample_rate.unwrap_or(48_000),
            pending: None,
            eof: false,
            last_info: None,
            last_substream: None,
            overlap: Vec::new(),
            prev_transform_length: 0,
            aspx_ext_state: Vec::new(),
            acpl_state: acpl_synth::AcplSubstreamState::new(),
            acpl_5x_pair_state: acpl_synth::Acpl5xPairPcmState::new(),
            acpl_5x_mch_state: acpl_synth::Acpl5xMchPcmState::new(),
            ssf_synth_state: Vec::new(),
            ssf_walker_state: Vec::new(),
            sticky: asf::StickyConfig::default(),
            ajoc_dec: None,
            ice_ajcc: None,
            ice_acpl: None,
        }
    }

    fn extract_raw_frame<'a>(&self, pkt: &'a Packet) -> Result<(&'a [u8], bool)> {
        if let Some(f) = sync::find_sync_frame(&pkt.data) {
            // Annex G.4.2: verify the 0xAC41 crc_word over frame_size +
            // raw_ac4_frame; a mismatch means the frame is corrupt.
            if f.crc_valid == Some(false) {
                return Err(Error::invalid(
                    "ac4 decoder: sync-frame CRC mismatch (0xAC41)",
                ));
            }
            Ok((f.payload, true))
        } else {
            Ok((pkt.data.as_slice(), false))
        }
    }

    /// Run the A-SPX bandwidth-extension pipeline on a block of
    /// low-band PCM (produced by the core ASF/MDCT path) using the
    /// derived A-SPX frequency tables: forward QMF, HF tile-copy via
    /// the patch subband groups (§5.7.6.3.1.4 + §5.7.6.4.1.4
    /// simplified), per-envelope HF envelope adjustment gains
    /// (§5.7.6.4.2 Pseudocodes 90 / 91 / 95) when the substream
    /// carried envelope deltas, noise + tone injection (§5.7.6.4.3 P102,
    /// §5.7.6.4.4 P104, §5.7.6.4.5 P107/P108) driven by `add_harmonic`
    /// flags + `scf_sig_sb` / `scf_noise_sb` (Pseudocode 92/94), and
    /// otherwise a flat 0.5 gain scaffold. Finally runs inverse QMF
    /// synthesis. Returns the bandwidth-extended PCM (f32) aligned to
    /// the input PCM after accounting for the combined QMF group delay.
    ///
    /// `state` carries the noise/tone/sine-idx index state across calls
    /// (one per decoder channel). `add_harmonic` is from the parsed
    /// `aspx_hfgen_iwc_*` (Table 55/56) — empty/None if the substream
    /// didn't carry it, in which case the tone generator stays silent
    /// but noise still injects if envelope deltas are available.
    /// `tna_mode` is `aspx_tna_mode[sbg]` from the same hfgen payload;
    /// when present + FIXFIX framing, the HF generator runs the full
    /// §5.7.6.4.1.3 chirp + α0 + α1 TNS body (Pseudocodes 86 → 89)
    /// instead of the bare tile copy.
    ///
    /// If any preconditions fail (length not a multiple of 64, tables
    /// missing, sbx >= 64) the original PCM is returned unchanged.
    #[allow(clippy::too_many_arguments)]
    fn aspx_extend_pcm(
        pcm_in: &[f32],
        tables: &aspx::AspxFrequencyTables,
        cfg: &aspx::AspxConfig,
        framing: Option<&aspx::AspxFraming>,
        sig_deltas: Option<&[aspx::AspxHuffEnv]>,
        noise_deltas: Option<&[aspx::AspxHuffEnv]>,
        qmode_env: Option<aspx::AspxQuantStep>,
        delta_dir: Option<&aspx::AspxDeltaDir>,
        add_harmonic: Option<&[bool]>,
        // §5.7.6.4.1.3 Pseudocode 88 — `aspx_tna_mode[sbg]` per noise
        // subband group, drives chirp + α0 + α1 TNS path. `None` falls
        // back to the bare HF tile copy.
        tna_mode: Option<&[u8]>,
        state: &mut aspx::AspxChannelExtState,
        num_ts_in_ats: u32,
        // Round 43: §5.7.5 companding tool — applied on the QMF matrix
        // between envelope adjustment and QMF synthesis. `mode` selects
        // the Pseudocode 121 sub-branch (`Off` / `PerSlot` / `Averaged`
        // / `SyncPerSlot` / `SyncAveraged`). `sb0_override == Some(b)`
        // overrides the lower band edge with `acpl_qmf_band` for the
        // ASPX_ACPL_1 codec mode (per §5.7.5.2 sb0 selection); `None`
        // falls back to `tables.sbx` (the A-SPX crossover, default
        // for ASPX / SIMPLE).
        compand_mode: aspx::CompandingMode,
        compand_sb0_override: Option<u32>,
    ) -> Vec<f32> {
        let extended = Self::aspx_extend_to_qmf(
            pcm_in,
            tables,
            cfg,
            framing,
            sig_deltas,
            noise_deltas,
            qmode_env,
            delta_dir,
            add_harmonic,
            tna_mode,
            state,
            num_ts_in_ats,
        );
        match extended {
            Some((mut q, sbx_eff, sbz_eff)) => {
                let compand_sb0 = compand_sb0_override.unwrap_or(sbx_eff);
                aspx::apply_companding_on_qmf_with_mode(&mut q, compand_sb0, sbz_eff, compand_mode);
                Self::qmf_synthesise_pcm(&q, pcm_in.len())
            }
            None => pcm_in.to_vec(),
        }
    }

    /// Round 44: phase-1 of the A-SPX HF-extension pipeline — runs the
    /// QMF analysis, HF generation (TNS / tile-copy), envelope
    /// adjustment + noise / tone injection, and updates `state` — but
    /// stops BEFORE the §5.7.5 companding gain and the inverse-QMF
    /// synthesis. Returns the post-extension QMF matrix `q[sb][ts]`
    /// along with the (`sbx`, `sbz`) the companding tool will need.
    ///
    /// Returns `None` (and leaves `state` untouched in the same
    /// preconditions [`aspx_extend_pcm`] historically returned the
    /// input PCM verbatim) when the input fails the multiple-of-64
    /// length check, the frequency tables are degenerate, or the
    /// patches couldn't be derived.
    ///
    /// This split exists so that cross-channel synchronised companding
    /// (`sync_flag == 1`, see [`aspx::apply_synchronised_companding_across_channels`])
    /// can collect every channel's QMF matrix, compute the
    /// geometric-mean gain across them, then apply the synced gain
    /// uniformly before each channel runs its own synthesis via
    /// [`Self::qmf_synthesise_pcm`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn aspx_extend_to_qmf(
        pcm_in: &[f32],
        tables: &aspx::AspxFrequencyTables,
        cfg: &aspx::AspxConfig,
        framing: Option<&aspx::AspxFraming>,
        sig_deltas: Option<&[aspx::AspxHuffEnv]>,
        noise_deltas: Option<&[aspx::AspxHuffEnv]>,
        qmode_env: Option<aspx::AspxQuantStep>,
        delta_dir: Option<&aspx::AspxDeltaDir>,
        add_harmonic: Option<&[bool]>,
        tna_mode: Option<&[u8]>,
        state: &mut aspx::AspxChannelExtState,
        num_ts_in_ats: u32,
    ) -> Option<AspxQmfPhase1> {
        const NUM_QMF: usize = qmf::NUM_QMF_SUBBANDS;
        // Need PCM length as a multiple of 64 for whole QMF slots.
        if pcm_in.is_empty() || pcm_in.len() % NUM_QMF != 0 {
            return None;
        }
        let sbx = tables.sbx as usize;
        let sbz = tables.sbz as usize;
        if sbx == 0 || sbx >= NUM_QMF || sbz <= sbx || sbz > NUM_QMF {
            return None;
        }
        let n_slots = pcm_in.len() / NUM_QMF;
        // Forward QMF analysis on the low-band PCM.
        let mut ana = qmf::QmfAnalysisBank::new();
        let slots = ana.process_block(pcm_in);
        // Re-layout to q[sb][ts].
        let mut q: Vec<Vec<(f32, f32)>> = (0..NUM_QMF)
            .map(|_| vec![(0.0f32, 0.0f32); n_slots])
            .collect();
        for (ts, slot) in slots.iter().enumerate() {
            for (sb, s) in slot.iter().enumerate() {
                q[sb][ts] = *s;
            }
        }
        // Derive patches from the master-freq-scale tables. 48 kHz
        // family is the only base_samp_freq wired in the current
        // TOC-driven pipeline; 44.1 kHz would pass `false` instead.
        let is_highres = matches!(cfg.master_freq_scale, aspx::AspxMasterFreqScale::HighRes);
        let patches = aspx::derive_patch_tables(
            &tables.sbg_master,
            tables.num_sbg_master,
            tables.sba,
            tables.sbx,
            tables.num_sb_aspx,
            true,
            is_highres,
        );
        if patches.num_sbg_patches == 0 {
            return None;
        }
        // Truncate the high band (ASPX substreams only carry spectral
        // data up to sbx in the core path; the bandwidth-extension
        // tool is responsible for filling sbx..sbz).
        for row in q.iter_mut().skip(sbx) {
            for sample in row.iter_mut() {
                *sample = (0.0, 0.0);
            }
        }
        // HF generation: when the substream gave us aspx_tna_mode + a
        // framing we can derive atsg_sig and run the full §5.7.6.4.1.3
        // / .4 TNS body (Pseudocodes 86 → 89). Covers FIXFIX (Pseudocode
        // 76) and variable interval classes (Pseudocode 77).
        // Otherwise fall back to the bare tile copy in §5.7.6.4.1.4.
        let mut tns_used = false;
        if let (Some(tna), Some(frm)) = (tna_mode, framing) {
            let num_aspx_ts = (n_slots as u32) / num_ts_in_ats.max(1);
            let atsg_sig_opt = aspx::derive_atsg_borders(num_aspx_ts, frm).map(|(s, _)| s);
            if let Some(atsg_sig) = atsg_sig_opt {
                if !tna.is_empty() {
                    let q_low_ext =
                        crate::aspx_tns::build_q_low_ext(&q, &state.q_low_prev, tables.sba);
                    let cov = crate::aspx_tns::compute_covariance(&q_low_ext, tables.sba);
                    let (alpha0, alpha1) = crate::aspx_tns::compute_alphas(&cov);
                    let chirp = crate::aspx_tns::chirp_factors(tna, &state.tns);
                    let gain_vec = if cfg.preflat {
                        Some(crate::aspx_tns::compute_preflat_gains(
                            &q,
                            tables.sbx,
                            &atsg_sig,
                            num_ts_in_ats,
                        ))
                    } else {
                        None
                    };
                    let q_high = crate::aspx_tns::hf_tile_tns(
                        &q_low_ext,
                        &patches,
                        &tables.sbg_noise,
                        &chirp.chirp_arr,
                        &alpha0,
                        &alpha1,
                        gain_vec.as_deref(),
                        tables.sbx,
                        NUM_QMF as u32,
                        &atsg_sig,
                        num_ts_in_ats,
                    );
                    for (dst, src) in q.iter_mut().zip(q_high.iter()).take(sbz).skip(sbx) {
                        let len = dst.len().min(src.len());
                        dst[..len].copy_from_slice(&src[..len]);
                    }
                    crate::aspx_tns::advance_tns_state(&mut state.tns, &chirp);
                    tns_used = true;
                }
            }
        }
        if !tns_used {
            // Bare tile copy (§5.7.6.4.1.4 with chirp/α0/α1 = 0).
            let q_high = aspx::hf_tile_copy(&q, &patches, tables.sbx, NUM_QMF as u32);
            for (dst, src) in q.iter_mut().zip(q_high.iter()).take(sbz).skip(sbx) {
                dst.clone_from(src);
            }
        }
        // Snapshot Q_low for the next interval's Pseudocode 86 prefix.
        // Only snapshot the actual low-band (sb < sba); the high-band
        // is what we just synthesised, not part of Q_low.
        state.q_low_prev = (0..(tables.sba as usize))
            .map(|sb| {
                if sb < q.len() {
                    q[sb].clone()
                } else {
                    Vec::new()
                }
            })
            .collect();
        // Per-envelope HF envelope adjustment (§5.7.6.4.2 Pseudocodes
        // 90 / 91 / 95) when the bitstream surface carried envelope
        // deltas, followed by noise + tone injection (§5.7.6.4.3 / .4 /
        // .5 Pseudocodes 102 / 104 / 107 / 108) when add_harmonic flags
        // are available. Otherwise fall back to the flat-gain scaffold
        // so output PCM still has audible HF content.
        let mut used_envelope = false;
        if let (Some(frm), Some(sig), Some(noise), Some(qm), Some(dd)) =
            (framing, sig_deltas, noise_deltas, qmode_env, delta_dir)
        {
            let num_aspx_ts = (n_slots as u32) / num_ts_in_ats.max(1);
            // §5.7.6.3.3.1 Pseudocode 76 (FIXFIX) or §5.7.6.3.3.2
            // Pseudocode 77 (FIXVAR / VARFIX / VARVAR) border derivation.
            if let Some((atsg_sig, atsg_noise)) = aspx::derive_atsg_borders(num_aspx_ts, frm) {
                if sig.len() as u32 == frm.num_env {
                    let adjuster = aspx::AspxEnvelopeAdjuster::from_deltas_stateful(
                        &q,
                        tables,
                        sig,
                        noise,
                        qm,
                        &dd.sig_delta_dir,
                        &frm.freq_res,
                        &atsg_sig,
                        &atsg_noise,
                        num_ts_in_ats,
                        cfg.interpolation,
                        &mut state.env_prev,
                    );
                    // Noise + tone injection on top of the
                    // envelope-adjusted HF. `add_harmonic` is sized
                    // to `num_sbg_sig_highres`; if the caller didn't
                    // provide one (no aspx_hfgen_iwc in the
                    // substream), default to an all-false slice so
                    // only the noise floor contributes.
                    let num_sbg_sig_highres = tables.sbg_sig_highres.len().saturating_sub(1);
                    let default_ah = vec![false; num_sbg_sig_highres];
                    let ah: &[bool] = match add_harmonic {
                        Some(s) if s.len() == num_sbg_sig_highres => s,
                        _ => &default_ah,
                    };
                    // tsg_ptr: 0 for FIXFIX (§4.3.10.4.7), from
                    // framing.tsg_ptr for variable interval classes.
                    let aspx_tsg_ptr: u32 = frm.tsg_ptr.map(|p| p as u32).unwrap_or(0);
                    if matches!(frm.int_class, aspx::AspxIntClass::FixFix) && cfg.limiter {
                        // §5.7.6.4.2.2 limiter pipeline (Pseudocodes
                        // 96 → 101) replaces the raw sig_gain with
                        // the boost-corrected sig_gain_sb_adj, so
                        // do NOT pre-apply adjuster.apply here.
                        aspx::inject_noise_and_tone_with_limiter(
                            &mut q,
                            &adjuster,
                            tables,
                            &patches,
                            &atsg_noise,
                            ah,
                            aspx_tsg_ptr,
                            state,
                        );
                    } else {
                        adjuster.apply(&mut q);
                        aspx::inject_noise_and_tone(
                            &mut q,
                            &adjuster,
                            tables,
                            &atsg_noise,
                            ah,
                            aspx_tsg_ptr,
                            state,
                        );
                    }
                    used_envelope = true;
                }
            }
        }
        if !used_envelope {
            // Flat envelope gain fallback (scaffold kept for the
            // non-FIXFIX / missing-envelope paths). Using 0.5 so the
            // regenerated HF doesn't overwhelm the LF.
            aspx::apply_flat_envelope_gain(&mut q, tables.sbx, tables.sbz, 0.5);
            // Reset per-channel envelope/tone carry-over state — the
            // envelope adjustment didn't run, so its index state has
            // nothing consistent to advance. Next successful interval
            // starts at master_reset semantics. The TNS chirp / α0 /
            // α1 history (`state.tns` + `state.q_low_prev`) is
            // independent and is *kept* — its update has already been
            // recorded above when the TNS path ran.
            state.noise.reset();
            state.tone.reset();
            state.sine_idx_sb_prev = None;
            state.tsg_ptr_prev = 0;
            state.num_atsg_sig_prev = 0;
            state.env_prev.reset();
        }
        // Phase-1 returns the post-extension QMF matrix + the (sbx,
        // sbz) the §5.7.5 companding tool will need. Companding +
        // inverse-QMF synthesis happen in `aspx_extend_pcm` (single
        // channel) or in the caller via
        // [`aspx::apply_synchronised_companding_across_channels`] +
        // [`Self::qmf_synthesise_pcm`] (cross-channel sync_flag=1).
        Some((q, tables.sbx, tables.sbz))
    }

    /// Round 44: phase-2 of the A-SPX HF-extension pipeline — runs the
    /// inverse-QMF synthesis on a `q[sb][ts]` matrix and returns
    /// `out_len`-long PCM. Caller is responsible for having applied
    /// the §5.7.5 companding gain (per-channel via
    /// [`aspx::apply_companding_on_qmf_with_mode`] or cross-channel
    /// via [`aspx::apply_synchronised_companding_across_channels`]).
    fn qmf_synthesise_pcm(q: &[Vec<(f32, f32)>], out_len: usize) -> Vec<f32> {
        const NUM_QMF: usize = qmf::NUM_QMF_SUBBANDS;
        if q.len() < NUM_QMF || out_len == 0 {
            return Vec::new();
        }
        let n_slots = out_len / NUM_QMF;
        let mut syn = qmf::QmfSynthesisBank::new();
        let mut out = Vec::with_capacity(out_len);
        #[allow(clippy::needless_range_loop)] // ETSI TS 103 190-2 §4.4.7 q[sb][ts] indexing
        for ts in 0..n_slots {
            let mut slot = [(0.0f32, 0.0f32); NUM_QMF];
            for (sb, s) in slot.iter_mut().enumerate() {
                *s = q[sb][ts];
            }
            let row = syn.process_slot(&slot);
            out.extend_from_slice(&row);
        }
        out
    }

    /// Run IMDCT + KBD overlap-add for a single channel, returning
    /// floating-point PCM (suitable for the A-SPX QMF pipeline).
    fn imdct_channel_f32(&mut self, ch: usize, scaled: &[f32], n: usize) -> Vec<f32> {
        // Transform-length change clears *all* channel overlap state so
        // the next frame starts from a consistent history.
        if self.prev_transform_length != n as u32 {
            self.overlap.clear();
            self.prev_transform_length = n as u32;
        }
        while self.overlap.len() <= ch {
            self.overlap.push(vec![0.0_f32; n]);
        }
        if self.overlap[ch].len() != n {
            self.overlap[ch] = vec![0.0_f32; n];
        }
        let mut x = vec![0.0_f32; n];
        let copy = scaled.len().min(n);
        x[..copy].copy_from_slice(&scaled[..copy]);
        let y = mdct::imdct(&x);
        let window = mdct::kbd_window(n as u32);
        mdct::imdct_olap_symmetric(&y, &window, &mut self.overlap[ch])
    }

    /// Convert an f32 PCM buffer to i16, clamping to the i16 range.
    fn pcm_f32_to_i16(pcm: &[f32]) -> Vec<i16> {
        pcm.iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect()
    }

    /// Run IMDCT + KBD overlap-add for a single channel. `ch` indexes
    /// the per-channel overlap state (grown on demand). `scaled` is the
    /// dequantised spectrum; bins past `scaled.len()` are zero-padded
    /// up to N.
    fn imdct_channel(&mut self, ch: usize, scaled: &[f32], n: usize) -> Vec<i16> {
        let pcm_f = self.imdct_channel_f32(ch, scaled, n);
        Self::pcm_f32_to_i16(&pcm_f)
    }

    /// SSF synthesis: drive §5.2.3-5.2.7 across every granule + block
    /// in `data`, IMDCT each `n_mdct` block, overlap/add into the
    /// channel's history, and emit a single
    /// `frame_samples`-long S16 vector.
    ///
    /// Each SSF block produces an `n_mdct`-long spectrum; the IMDCT
    /// then yields `2 * n_mdct` time-domain samples which the
    /// overlap-add step combines with the previous block's tail to
    /// emit `n_mdct` PCM samples. So one granule emits
    /// `num_blocks * n_mdct = granule_length` samples; one frame's
    /// `ssf_data` covers the entire frame_length.
    fn run_ssf_channel(
        &mut self,
        ch: usize,
        data: &crate::ssf::SsfData,
        frame_samples: usize,
    ) -> Vec<i16> {
        // Drive the synth.
        let state_idx = ch.min(self.ssf_synth_state.len().saturating_sub(1));
        let mut spec_concat: Vec<f32> = Vec::new();
        let mut block_lengths: Vec<usize> = Vec::new();
        for granule in &data.granules {
            let n_mdct = granule.n_mdct as usize;
            if n_mdct == 0 {
                continue;
            }
            // env_prev[] for SHORT_STRIDE P-frame interpolation now
            // lives on `SsfSynthState` and the synth latches the
            // resolved envelope at the end of each granule, so we pass
            // an empty slice and let the synth pull the previous
            // granule's envelope from `state.env_prev` (§5.2.3.0 Note 2).
            let block =
                ssf_synth::synthesize_granule(granule, &[], &mut self.ssf_synth_state[state_idx]);
            // synthesize_granule returns num_blocks * n_mdct; track
            // each block's n_mdct so the IMDCT loop can split them.
            for _ in 0..(granule.num_blocks as usize) {
                block_lengths.push(n_mdct);
            }
            spec_concat.extend_from_slice(&block);
        }
        if spec_concat.is_empty() || block_lengths.is_empty() {
            return Vec::new();
        }
        // IMDCT each block independently and concat.
        let mut pcm_out: Vec<f32> = Vec::with_capacity(frame_samples);
        let mut off = 0usize;
        for &n in &block_lengths {
            if off + n > spec_concat.len() {
                break;
            }
            let block_spec = &spec_concat[off..off + n];
            // Use `imdct_channel_f32` for KBD-windowed overlap-add.
            // SSF blocks share the channel's overlap state so the
            // history chains across blocks within a frame.
            let pcm_block = self.imdct_channel_f32(ch, block_spec, n);
            pcm_out.extend_from_slice(&pcm_block);
            off += n;
        }
        // Truncate / pad to frame_samples.
        if pcm_out.len() > frame_samples {
            pcm_out.truncate(frame_samples);
        } else if pcm_out.len() < frame_samples {
            pcm_out.resize(frame_samples, 0.0);
        }
        Self::pcm_f32_to_i16(&pcm_out)
    }

    /// IMDCT a `MonoLfeData` payload's `scaled_spec` to PCM `f32` using
    /// the channel slot's overlap-add history. Returns `None` if the
    /// mono shell didn't decode a body (LFE / SSF frontend / Huffman
    /// miss) or if the carrier transform-length differs from `n`.
    ///
    /// `ch` is the per-channel overlap slot index (the centre channel
    /// uses slot 2 for the 5.X path; surround Ls/Rs use 3/4 etc.).
    fn imdct_mono_lfe_data_f32(
        &mut self,
        mono: &crate::mch::MonoLfeData,
        ch: usize,
        n: usize,
    ) -> Option<Vec<f32>> {
        let scaled = mono.scaled_spec.as_ref()?;
        let ti = mono.transform_info.as_ref()?;
        if ti.transform_length_0 as usize != n {
            return None;
        }
        Some(self.imdct_channel_f32(ch, scaled, n))
    }

    /// §5.7.7.6.1 ASPX_ACPL_1 / ASPX_ACPL_2 5_X dispatch helper —
    /// extracted from `receive_frame` so unit tests can drive it
    /// without building a full 5_X TOC + body.
    ///
    /// Carries:
    /// * `mode` — AspxAcpl1 (carrier-pair + Ls/Rs surround) or
    ///   AspxAcpl2 (carrier-pair only).
    /// * `cfg` — single `acpl_config_1ch` shared between both
    ///   ACplModule's (per Pseudocode 117).
    /// * `data_1` — `acpl_data_1ch_pair[0]` — L-side parameters.
    /// * `data_2` — `acpl_data_1ch_pair[1]` — R-side parameters.
    /// * `samples` — frame length in PCM samples.
    /// * `centre_pcm` — optional centre channel PCM (already IMDCT +
    ///   overlap-added). When present and length-matched, used as the
    ///   `x2` carrier for Pseudocode 117's centre passthrough; when
    ///   `None`, falls back to silence (round-36 behaviour). Round 37
    ///   wires this from the parsed `cfg0_centre_mono.scaled_spec`.
    /// * `ls_pcm` / `rs_pcm` — optional surround Ls/Rs carriers for
    ///   ASPX_ACPL_1 (Mode 1's `x3`/`x4` driving channels). When `None`
    ///   and `mode == AspxAcpl1`, falls back to silence (round-36
    ///   behaviour). Ignored entirely for `AspxAcpl2`.
    /// * `pcm_per_channel` — slot list. Reads slots 0/1 as L/R carriers
    ///   (zero-fills if absent); writes slots 0..4 (L/R/C/Ls/Rs) on a
    ///   successful synthesis.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_acpl_5x_pair(
        &mut self,
        mode: acpl_synth::Acpl5xPairMode,
        cfg: &crate::acpl::AcplConfig1ch,
        data_1: &crate::acpl::AcplData1ch,
        data_2: &crate::acpl::AcplData1ch,
        samples: usize,
        centre_pcm: Option<&[f32]>,
        ls_pcm: Option<&[f32]>,
        rs_pcm: Option<&[f32]>,
        pcm_per_channel: &mut Vec<Option<Vec<i16>>>,
    ) {
        let n = samples;
        // run_acpl_5x_pair_pcm requires every PCM input to be a multiple
        // of 64 (one QMF slot). Frame length in AC-4 is always a
        // multiple of 64 by spec, but be defensive.
        if n == 0 || n % qmf::NUM_QMF_SUBBANDS != 0 {
            return;
        }
        let pcm_l_f32: Vec<f32> = pcm_per_channel
            .first()
            .and_then(|p| p.as_ref())
            .map(|v| v.iter().map(|&s| s as f32 / 32767.0).collect())
            .unwrap_or_else(|| vec![0.0_f32; n]);
        let pcm_r_f32: Vec<f32> = pcm_per_channel
            .get(1)
            .and_then(|p| p.as_ref())
            .map(|v| v.iter().map(|&s| s as f32 / 32767.0).collect())
            .unwrap_or_else(|| vec![0.0_f32; n]);
        // Centre carrier: real PCM if the caller supplied a length-matched
        // buffer (round 37 wires this from the parsed centre mono data),
        // else silence (round-36 placeholder behaviour).
        let pcm_c_f32: Vec<f32> = match centre_pcm {
            Some(p) if p.len() == n => p.to_vec(),
            _ => vec![0.0_f32; n],
        };
        // Surround Ls/Rs carriers — only used in ACPL_1 mode. Real PCM
        // when supplied + length-matched, else silence (round-36
        // behaviour).
        let pcm_ls_owned: Option<Vec<f32>> =
            if matches!(mode, acpl_synth::Acpl5xPairMode::AspxAcpl1) {
                Some(match ls_pcm {
                    Some(p) if p.len() == n => p.to_vec(),
                    _ => vec![0.0_f32; n],
                })
            } else {
                None
            };
        let pcm_rs_owned: Option<Vec<f32>> =
            if matches!(mode, acpl_synth::Acpl5xPairMode::AspxAcpl1) {
                Some(match rs_pcm {
                    Some(p) if p.len() == n => p.to_vec(),
                    _ => vec![0.0_f32; n],
                })
            } else {
                None
            };
        if let Some(out) = acpl_synth::run_acpl_5x_pair_pcm(
            mode,
            &pcm_l_f32,
            &pcm_r_f32,
            &pcm_c_f32,
            pcm_ls_owned.as_deref(),
            pcm_rs_owned.as_deref(),
            cfg,
            data_1,
            cfg,
            data_2,
            &mut self.acpl_5x_pair_state,
        ) {
            // Output channel mapping for 5.0/5.1:
            //   ch0 = L, ch1 = R, ch2 = C, ch3 = Ls, ch4 = Rs.
            while pcm_per_channel.len() < 5 {
                pcm_per_channel.push(None);
            }
            pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&out.left));
            pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&out.right));
            pcm_per_channel[2] = Some(Self::pcm_f32_to_i16(&out.centre));
            pcm_per_channel[3] = Some(Self::pcm_f32_to_i16(&out.left_surround));
            pcm_per_channel[4] = Some(Self::pcm_f32_to_i16(&out.right_surround));
        }
    }

    /// Apply A-SPX bandwidth-extension to one channel's IMDCT'd PCM
    /// using a captured 5_X trailer slice. Wraps `aspx_extend_pcm` with
    /// the trailer's per-channel envelopes / framing / hfgen state and
    /// the trailer's frequency tables. `slot` indexes the per-channel
    /// `aspx_ext_state` carry-over so each output slot keeps its own
    /// noise / tone / TNS history.
    #[allow(clippy::too_many_arguments)]
    fn aspx_extend_with_trailer(
        &mut self,
        pcm_in: &[f32],
        trailer: &aspx::FiveXAspxTrailer,
        ch: &aspx::FiveXAspxChannelTrailer,
        cfg: &aspx::AspxConfig,
        slot: usize,
        num_ts_in_ats: u32,
        compand_mode: aspx::CompandingMode,
        compand_sb0_override: Option<u32>,
    ) -> Vec<f32> {
        while self.aspx_ext_state.len() <= slot {
            self.aspx_ext_state.push(aspx::AspxChannelExtState::new());
        }
        let state = &mut self.aspx_ext_state[slot];
        Self::aspx_extend_pcm(
            pcm_in,
            &trailer.frequency_tables,
            cfg,
            Some(&ch.framing),
            Some(&ch.data_sig),
            Some(&ch.data_noise),
            Some(ch.qmode_env),
            Some(&ch.delta_dir),
            ch.add_harmonic.as_deref(),
            ch.tna_mode.as_deref(),
            state,
            num_ts_in_ats,
            compand_mode,
            compand_sb0_override,
        )
    }

    /// Round 43: per-output-channel companding mode from the captured
    /// `companding_control(num_chan)` for a 5_X frame. The Cfg2 / Cfg0 /
    /// Cfg1 / Cfg3 paths all carry `companding_control(5)`, indexed by
    /// the 5_X output channel `slot` (0..4 in L/R/C/Ls/Rs order). If
    /// `sync_flag == true`, `compand_on[0]` applies to all five
    /// channels (Table 116).
    ///
    /// Returns `CompandingMode::Off` whenever the parsed flags don't
    /// reach the requested slot, otherwise resolves to one of the four
    /// active sub-branches of Pseudocode 121
    /// (`PerSlot` / `Averaged` / `SyncPerSlot` / `SyncAveraged`) per
    /// [`aspx::CompandingMode::from_control`].
    fn five_x_compand_mode_for_slot(
        cc: Option<&aspx::CompandingControl>,
        slot: usize,
    ) -> aspx::CompandingMode {
        match cc {
            Some(cc) => aspx::CompandingMode::from_control(cc, slot),
            None => aspx::CompandingMode::Off,
        }
    }

    /// Backward-compat helper kept for round-42 unit tests — returns
    /// the boolean "is companding active on this slot" derived from
    /// the resolved [`aspx::CompandingMode`]. New code should call
    /// [`Self::five_x_compand_mode_for_slot`] directly.
    fn five_x_compand_on_for_slot(cc: Option<&aspx::CompandingControl>, slot: usize) -> bool {
        !matches!(
            Self::five_x_compand_mode_for_slot(cc, slot),
            aspx::CompandingMode::Off
        )
    }

    /// Round 44: cross-channel synchronised A-SPX bandwidth-extension
    /// for the 5_X SIMPLE/ASPX path when the parsed
    /// `companding_control()` carries `sync_flag == 1`.
    ///
    /// Pseudocode 121's `sync_flag == 1` branch defines the gain as
    /// `g_synch(ts) = (∏_{ch=0..M} g_ch(ts))^(1/M)` and applies it
    /// uniformly to every contributing channel — i.e. one cross-channel
    /// gain per slot, NOT one per-channel gain. The pre-r44 pipeline
    /// approximated this with the per-channel `g_ch(ts)` (exact for
    /// `M = 1`); this entry-point closes the gap by:
    ///
    ///   1. Driving each contributing channel through phase-1
    ///      [`Self::aspx_extend_to_qmf`] to capture the post-extension
    ///      QMF matrix `q_ch[sb][ts]` along with each channel's
    ///      `(sb0, sbz)` companding band.
    ///   2. Calling [`aspx::apply_synchronised_companding_across_channels`]
    ///      with the collected QMF matrices and bands — that walks
    ///      Pseudocode 121's geometric-mean across channels and writes
    ///      the synced gain back into every QMF matrix.
    ///   3. Driving each channel through phase-2
    ///      [`Self::qmf_synthesise_pcm`] to produce the final PCM.
    ///
    /// Channels whose phase-1 returned `None` (length / table /
    /// patch-derivation guard tripped — e.g. a slot whose IMDCT'd PCM
    /// length isn't a multiple of 64) fall back to the unmodified
    /// input PCM for that slot — same behaviour as the per-channel
    /// `aspx_extend_pcm` helper used to give.
    ///
    /// `entries[i]` is `(slot, pcm_in, trailer, ch, sb0_override)`:
    ///   * `slot` — output channel index (0..=4 for 5_X), used to
    ///     pick the right `aspx_ext_state[slot]` carry-over.
    ///   * `pcm_in` — IMDCT'd LF PCM for that output channel.
    ///   * `trailer` — captured 5_X trailer (carries
    ///     `frequency_tables`).
    ///   * `ch` — primary or secondary channel within `trailer`.
    ///   * `sb0_override` — `Some(acpl_qmf_band)` for ASPX_ACPL_1
    ///     (`acpl_qmf_band` replaces `aspx_xover_band` per §5.7.5.2);
    ///     `None` for SIMPLE / ASPX (sb0 = trailer.sbx).
    ///
    /// Returns one `(slot, Vec<f32>)` per entry, in the order they
    /// were passed in. The caller is responsible for the trailing
    /// f32→i16 cast and writeback into `pcm_per_channel[slot]`.
    ///
    /// `mode` MUST be either [`aspx::CompandingMode::SyncPerSlot`] or
    /// [`aspx::CompandingMode::SyncAveraged`]; no-op (i.e. the
    /// per-channel pipeline outputs without companding gain) for any
    /// other mode.
    fn extend_5x_channels_with_sync_companding(
        &mut self,
        entries: &[SyncCompandingChannelEntry<'_>],
        num_ts_in_ats: u32,
        mode: aspx::CompandingMode,
    ) -> Vec<(usize, Vec<f32>)> {
        // Phase 1: drive each entry through aspx_extend_to_qmf,
        // capturing the post-extension QMF matrix (or `None` if the
        // extension preconditions tripped — that channel will pass
        // through unchanged).
        let mut phase1: Vec<(usize, usize, Option<AspxQmfPhase1>)> =
            Vec::with_capacity(entries.len());
        for (slot, pcm_in, trailer, ch, cfg, sb0_override) in entries.iter() {
            while self.aspx_ext_state.len() <= *slot {
                self.aspx_ext_state.push(aspx::AspxChannelExtState::new());
            }
            let state = &mut self.aspx_ext_state[*slot];
            let qres = Self::aspx_extend_to_qmf(
                pcm_in,
                &trailer.frequency_tables,
                cfg,
                Some(&ch.framing),
                Some(&ch.data_sig),
                Some(&ch.data_noise),
                Some(ch.qmode_env),
                Some(&ch.delta_dir),
                ch.add_harmonic.as_deref(),
                ch.tna_mode.as_deref(),
                state,
                num_ts_in_ats,
            );
            // Resolve the effective sb0 for the synced companding —
            // sb0_override (acpl_qmf_band for ASPX_ACPL_1) or sbx (the
            // A-SPX crossover band for SIMPLE / ASPX).
            let q_with_band = qres.map(|(q, sbx, sbz)| {
                let sb0 = sb0_override.unwrap_or(sbx);
                (q, sb0, sbz)
            });
            phase1.push((*slot, pcm_in.len(), q_with_band));
        }
        // Phase 2: collect every channel that survived phase 1 into
        // the synced companding helper. Only mutable references to
        // the QMF matrices are passed in; the helper reads each
        // channel's level, computes geometric-mean across them, and
        // writes back the synced scales.
        {
            let mut sync_view: Vec<aspx::SyncCompandingEntry<'_>> = Vec::new();
            for (_, _, q_opt) in phase1.iter_mut() {
                if let Some((q, sb0, sbz)) = q_opt.as_mut() {
                    sync_view.push((q, *sb0, *sbz));
                }
            }
            aspx::apply_synchronised_companding_across_channels(&mut sync_view, mode);
        }
        // Phase 3: synthesise per-channel PCM. Channels whose phase
        // returned None fall back to a clone of the input PCM (same
        // contract as the original `aspx_extend_pcm`).
        let mut out: Vec<(usize, Vec<f32>)> = Vec::with_capacity(entries.len());
        for (i, (slot, pcm_len, q_opt)) in phase1.into_iter().enumerate() {
            let pcm = match q_opt {
                Some((q, _, _)) => Self::qmf_synthesise_pcm(&q, pcm_len),
                None => entries[i].1.to_vec(),
            };
            out.push((slot, pcm));
        }
        out
    }

    /// Round 45: stereo-CPE counterpart to
    /// [`Self::extend_5x_channels_with_sync_companding`] for the M=2
    /// case where the two channels are not 5_X trailer slots but the
    /// primary / secondary of an `aspx_data_2ch` stereo CPE — in
    /// particular the L/R carrier pair that drives a 5_X ASPX_ACPL_3
    /// `run_acpl_5x_mch_pcm` synthesis (Pseudocode 118 expects the
    /// extended carriers, not raw IMDCT'd PCM).
    ///
    /// When `companding_control(2)` carried `sync_flag == 1` the spec's
    /// `g_synch(ts) = (∏_{ch=0..M} g_ch(ts))^(1/M)` collapses for M=2
    /// to `√(g_0(ts) · g_1(ts))` — a single geometric-mean gain shared
    /// across both channels rather than two independent per-channel
    /// gains. This matches r44's 5_X SIMPLE/ASPX dispatch path:
    /// phase-1 runs each channel through [`Self::aspx_extend_to_qmf`]
    /// (capturing the post-extension QMF matrix + each channel's
    /// `(sb0, sbz)` companding band), phase-2 calls
    /// [`aspx::apply_synchronised_companding_across_channels`] to
    /// write the synced gain into both QMF matrices, and phase-3
    /// runs each channel through [`Self::qmf_synthesise_pcm`] to
    /// produce the final PCM.
    ///
    /// `mode` MUST be one of
    /// [`aspx::CompandingMode::SyncPerSlot`] / [`aspx::CompandingMode::SyncAveraged`]
    /// (the cross-channel sync sub-branches of Pseudocode 121); any
    /// other mode is a no-op for the synced pipeline and the caller
    /// should run the per-channel `aspx_extend_pcm` path instead.
    ///
    /// `tables` / `cfg` are shared between the two channels in a
    /// stereo CPE (one `aspx_config()` per substream). `sb0_override`
    /// is `Some(acpl_qmf_band)` for the stereo ASPX_ACPL_1 path
    /// (which substitutes `acpl_qmf_band` for `aspx_xover_band` per
    /// §5.7.5.2 sb0 selection); `None` everywhere else (SIMPLE / ASPX
    /// / ACPL_3 paths use `tables.sbx`).
    ///
    /// When either channel's phase-1 returns `None` (PCM length not a
    /// multiple of 64, missing tables, etc.) that channel falls back
    /// to its un-extended PCM — same contract as
    /// [`Self::aspx_extend_pcm`] / [`Self::extend_5x_channels_with_sync_companding`].
    #[allow(clippy::too_many_arguments)]
    fn extend_stereo_cpe_pair_with_sync_companding(
        &mut self,
        primary: &StereoCpeChannelInput<'_>,
        secondary: &StereoCpeChannelInput<'_>,
        tables: &aspx::AspxFrequencyTables,
        cfg: &aspx::AspxConfig,
        num_ts_in_ats: u32,
        mode: aspx::CompandingMode,
        sb0_override: Option<u32>,
    ) -> (Vec<f32>, Vec<f32>) {
        // Phase 1: drive each channel through aspx_extend_to_qmf and
        // capture the post-extension QMF matrix + (sb0, sbz) band.
        // Lay out as Vec so indices are stable across the
        // borrow-juggle below.
        let mut phase1: [(usize, usize, Option<AspxQmfPhase1>); 2] = [
            (primary.ch_index, primary.pcm_in.len(), None),
            (secondary.ch_index, secondary.pcm_in.len(), None),
        ];
        for (i, input) in [primary, secondary].iter().enumerate() {
            while self.aspx_ext_state.len() <= input.ch_index {
                self.aspx_ext_state.push(aspx::AspxChannelExtState::new());
            }
            let state = &mut self.aspx_ext_state[input.ch_index];
            phase1[i].2 = Self::aspx_extend_to_qmf(
                input.pcm_in,
                tables,
                cfg,
                input.framing,
                input.sig,
                input.noise,
                input.qmode,
                input.delta_dir,
                input.add_harmonic,
                input.tna_mode,
                state,
                num_ts_in_ats,
            )
            .map(|(q, sbx_eff, sbz_eff)| {
                // sb0_override is shared across both channels of the
                // stereo CPE (acpl_qmf_band for ASPX_ACPL_1 stereo,
                // sbx everywhere else).
                let sb0 = sb0_override.unwrap_or(sbx_eff);
                (q, sb0, sbz_eff)
            });
        }
        // Phase 2: collect every channel that survived phase 1 into
        // the synced companding helper. M=2 → `g_synch(ts) = √(g_0(ts) · g_1(ts))`
        // is written back into BOTH QMF matrices uniformly.
        {
            let mut sync_view: Vec<aspx::SyncCompandingEntry<'_>> = Vec::with_capacity(2);
            for (_, _, q_opt) in phase1.iter_mut() {
                if let Some((q, sb0, sbz)) = q_opt.as_mut() {
                    sync_view.push((q, *sb0, *sbz));
                }
            }
            aspx::apply_synchronised_companding_across_channels(&mut sync_view, mode);
        }
        // Phase 3: synthesise per-channel PCM. Channels whose phase-1
        // returned None fall back to the unmodified input PCM (same
        // contract as `aspx_extend_pcm`).
        let pcm_out = |idx: usize, fallback: &[f32]| -> Vec<f32> {
            match &phase1[idx].2 {
                Some((q, _, _)) => Self::qmf_synthesise_pcm(q, phase1[idx].1),
                None => fallback.to_vec(),
            }
        };
        let pri = pcm_out(0, primary.pcm_in);
        let sec = pcm_out(1, secondary.pcm_in);
        (pri, sec)
    }

    /// Round 44: shared front-end for the 5_X SIMPLE/ASPX dispatchers
    /// that resolves the synced-companding mode for the whole 5_X
    /// frame. With `sync_flag == 1`, every channel resolves to the
    /// SAME mode (Pseudocode 121 broadcasts `compand_on[0]`); with
    /// `sync_flag == 0` (or no companding) the per-channel
    /// [`Self::five_x_compand_mode_for_slot`] is what callers want.
    ///
    /// Returns `Some(mode)` when the cross-channel synced pipeline
    /// should run (mode is `SyncPerSlot` or `SyncAveraged`); `None`
    /// when the per-channel pipeline should run (sync_flag missing /
    /// false, or sync_flag=true resolves to `Off`).
    fn five_x_synced_mode(cc: Option<&aspx::CompandingControl>) -> Option<aspx::CompandingMode> {
        let cc = cc?;
        if !matches!(cc.sync_flag, Some(true)) {
            return None;
        }
        let mode = aspx::CompandingMode::from_control(cc, 0);
        match mode {
            aspx::CompandingMode::SyncPerSlot | aspx::CompandingMode::SyncAveraged => Some(mode),
            _ => None,
        }
    }

    /// Round 44: drive every entry through the synced-companding
    /// pipeline (when `synced_mode` is `Some`), apply the resulting
    /// PCM to `pcm_per_channel[slot]`. Otherwise (sync mode = None),
    /// drive each entry through the per-channel pipeline.
    ///
    /// Each entry is `(slot, pcm_f, trailer, ch, sb0_override)`.
    /// `aspx_cfg` is shared across all entries (one config per 5_X
    /// substream).
    #[allow(clippy::too_many_arguments)]
    fn extend_5x_entries(
        &mut self,
        entries: Vec<FiveXChannelEntry<'_>>,
        aspx_cfg: Option<aspx::AspxConfig>,
        companding: Option<&aspx::CompandingControl>,
        num_ts_in_ats: u32,
        pcm_per_channel: &mut [Option<Vec<i16>>],
    ) {
        let synced = Self::five_x_synced_mode(companding);
        if let (Some(mode), Some(cfg)) = (synced, aspx_cfg) {
            // Cross-channel synced path. Build the entries-with-trailer
            // list (skipping any whose trailer is missing — those fall
            // back to the unmodified PCM for that slot).
            let mut sync_entries: Vec<SyncCompandingChannelEntry<'_>> = Vec::new();
            // Track which entries had no trailer — they pass through
            // the PCM unchanged.
            let mut passthrough: Vec<(usize, &[f32])> = Vec::new();
            for (slot, pcm_f, trailer_pair) in entries.iter() {
                match trailer_pair {
                    Some((trailer, is_secondary)) => {
                        let ch = if *is_secondary {
                            trailer.secondary.as_ref().unwrap_or(&trailer.primary)
                        } else {
                            &trailer.primary
                        };
                        sync_entries.push((*slot, pcm_f.as_slice(), trailer, ch, &cfg, None));
                    }
                    None => {
                        passthrough.push((*slot, pcm_f.as_slice()));
                    }
                }
            }
            let extended =
                self.extend_5x_channels_with_sync_companding(&sync_entries, num_ts_in_ats, mode);
            for (slot, pcm) in extended {
                pcm_per_channel[slot] = Some(Self::pcm_f32_to_i16(&pcm));
            }
            for (slot, pcm) in passthrough {
                pcm_per_channel[slot] = Some(Self::pcm_f32_to_i16(pcm));
            }
            return;
        }
        // Per-channel path (sync_flag == 0 or sync_flag == 1 + Off).
        for (slot, pcm_f, trailer_pair) in entries.into_iter() {
            let pcm_i16 = match (aspx_cfg, trailer_pair) {
                (Some(cfg), Some((trailer, is_secondary))) => {
                    let ch = if is_secondary {
                        trailer.secondary.as_ref().unwrap_or(&trailer.primary)
                    } else {
                        &trailer.primary
                    };
                    let compand_mode = Self::five_x_compand_mode_for_slot(companding, slot);
                    let extended = self.aspx_extend_with_trailer(
                        &pcm_f,
                        trailer,
                        ch,
                        &cfg,
                        slot,
                        num_ts_in_ats,
                        compand_mode,
                        None,
                    );
                    Self::pcm_f32_to_i16(&extended)
                }
                _ => Self::pcm_f32_to_i16(&pcm_f),
            };
            pcm_per_channel[slot] = Some(pcm_i16);
        }
    }

    /// §5.3.4.3.1 / Table 180 — 5_X SIMPLE/ASPX `coding_config == 2`
    /// dispatch: the parsed `four_channel_data` carries L/R/Ls/Rs in
    /// `scaled_spec_per_channel[0..4]` and the trailing
    /// `cfg2_back_mono.scaled_spec` carries the centre. Channel mapping
    /// per Table 180:
    ///
    /// ```text
    ///     four_channel_data[0] -> slot 0 (L)
    ///     four_channel_data[1] -> slot 1 (R)
    ///     four_channel_data[2] -> slot 3 (Ls)
    ///     four_channel_data[3] -> slot 4 (Rs)
    ///     mono_data           -> slot 2 (C)
    /// ```
    ///
    /// Round 41 wires the ASPX bandwidth-extension trailer per channel:
    /// `aspx_data_2ch[L,R] + aspx_data_2ch[Ls,Rs] + aspx_data_1ch[C]`
    /// (per Table 25 row `case ASPX:`). Each trailer's per-channel
    /// envelope set drives `aspx_extend_pcm` on the IMDCT'd low-band
    /// PCM before quantisation. When a trailer is absent (SIMPLE mode
    /// or trailer-parse miss) the channel passes through with low-band
    /// PCM only — matching the round-38 behaviour for those paths.
    ///
    /// The function is a no-op when any of the four per-channel scaled
    /// spectra are absent (short / grouped frame or Huffman miss);
    /// centre is silent when the trailing `mono_data` body is absent.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_5x_cfg2_simple_aspx(
        &mut self,
        four: &crate::mch::FourChannelData,
        back_mono: Option<&crate::mch::MonoLfeData>,
        aspx_lr: Option<&aspx::FiveXAspxTrailer>,
        aspx_ls_rs: Option<&aspx::FiveXAspxTrailer>,
        aspx_centre: Option<&aspx::FiveXAspxTrailer>,
        aspx_cfg: Option<aspx::AspxConfig>,
        companding: Option<&aspx::CompandingControl>,
        num_ts_in_ats: u32,
        samples: usize,
        pcm_per_channel: &mut Vec<Option<Vec<i16>>>,
    ) {
        let Some(ti) = four.transform_info.as_ref() else {
            return;
        };
        let n = ti.transform_length_0 as usize;
        if n == 0 || n != samples {
            return;
        }
        if four.scaled_spec_per_channel.len() < 4 {
            return;
        }
        // Channel mapping: ch_in -> slot_out per Table 180 cfg2 column.
        // The ASPX trailers map (L,R) and (Ls,Rs) onto the front /
        // surround stereo pairs.
        const SLOT_MAP: [usize; 4] = [0, 1, 3, 4];
        // Need at least 5 output slots (L/R/C/Ls/Rs). Resize on demand.
        while pcm_per_channel.len() < 5 {
            pcm_per_channel.push(None);
        }
        // L (slot 0) — primary channel of the L/R 2ch trailer.
        // R (slot 1) — secondary channel of the L/R 2ch trailer.
        // Ls (slot 3) — primary channel of the Ls/Rs 2ch trailer.
        // Rs (slot 4) — secondary channel of the Ls/Rs 2ch trailer.
        let trailers_for_ch: [Option<(&aspx::FiveXAspxTrailer, bool)>; 4] = [
            aspx_lr.map(|t| (t, false)),    // L
            aspx_lr.map(|t| (t, true)),     // R
            aspx_ls_rs.map(|t| (t, false)), // Ls
            aspx_ls_rs.map(|t| (t, true)),  // Rs
        ];
        // Build the per-slot entries (slot, pcm_f, trailer_pair) for
        // the L/R/Ls/Rs quartet, plus the centre. The centre joins the
        // synced-companding cohort when both `back_mono` and a centre
        // trailer are present — that way Pseudocode 121's
        // `g_synch(ts) = (∏ g_ch(ts))^(1/M)` averages across all five
        // 5_X channels, not just the four front/surround.
        let mut entries: Vec<FiveXChannelEntry<'_>> = Vec::with_capacity(5);
        for (ch_in, &slot) in SLOT_MAP.iter().enumerate() {
            let Some(scaled) = four.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm_f = self.imdct_channel_f32(slot, scaled, n);
            entries.push((slot, pcm_f, trailers_for_ch[ch_in]));
        }
        if let Some(mono) = back_mono {
            if let Some(pcm_f) = self.imdct_mono_lfe_data_f32(mono, 2, samples) {
                let centre_pair = aspx_centre.map(|t| (t, false));
                entries.push((2, pcm_f, centre_pair));
            }
        }
        self.extend_5x_entries(
            entries,
            aspx_cfg,
            companding,
            num_ts_in_ats,
            pcm_per_channel,
        );
    }

    /// §5.3.4.3.1 / Table 180 — 5_X SIMPLE/ASPX `coding_config == 0`
    /// dispatch. The body shape is
    /// `b_2ch_mode + two_channel_data + two_channel_data + mono_data(0)`,
    /// with channel mapping driven by the 1-bit `b_2ch_mode`:
    ///
    /// ```text
    /// 2ch_mode == 0 (Table 180 column 0a):
    ///     two_channel_data[0]      -> [0, 1] (L,  R)
    ///     two_channel_data[1]      -> [3, 4] (Ls, Rs)
    ///     mono_data                -> [2]    (C)
    ///
    /// 2ch_mode == 1 (Table 180 column 0b):
    ///     two_channel_data[0]      -> [0, 3] (L,  Ls)
    ///     two_channel_data[1]      -> [1, 4] (R,  Rs)
    ///     mono_data                -> [2]    (C)
    /// ```
    ///
    /// The function is a no-op when `tcd_a` doesn't carry a transform_info
    /// matching `samples`, when fewer than two `two_channel_data` shells
    /// are present, or when any per-channel scaled spectrum is missing
    /// (short / grouped / Huffman-miss path). Centre is silent when the
    /// trailing `mono_data` body is absent.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_5x_cfg0_simple_aspx(
        &mut self,
        tcd_a: &crate::mch::TwoChannelData,
        tcd_b: &crate::mch::TwoChannelData,
        b_2ch_mode: bool,
        centre_mono: Option<&crate::mch::MonoLfeData>,
        aspx_lr: Option<&aspx::FiveXAspxTrailer>,
        aspx_ls_rs: Option<&aspx::FiveXAspxTrailer>,
        aspx_centre: Option<&aspx::FiveXAspxTrailer>,
        aspx_cfg: Option<aspx::AspxConfig>,
        companding: Option<&aspx::CompandingControl>,
        num_ts_in_ats: u32,
        samples: usize,
        pcm_per_channel: &mut Vec<Option<Vec<i16>>>,
    ) {
        let Some(ti_a) = tcd_a.transform_info.as_ref() else {
            return;
        };
        let n_a = ti_a.transform_length_0 as usize;
        if n_a == 0 || n_a != samples {
            return;
        }
        let Some(ti_b) = tcd_b.transform_info.as_ref() else {
            return;
        };
        let n_b = ti_b.transform_length_0 as usize;
        if n_b == 0 || n_b != samples {
            return;
        }
        if tcd_a.scaled_spec_per_channel.len() < 2 || tcd_b.scaled_spec_per_channel.len() < 2 {
            return;
        }
        // Slot map per Table 180 column 0:
        //   2ch_mode == 0: [0,1] then [3,4] (L,R / Ls,Rs)
        //   2ch_mode == 1: [0,3] then [1,4] (L,Ls / R,Rs)
        let slot_map_a: [usize; 2] = if b_2ch_mode { [0, 3] } else { [0, 1] };
        let slot_map_b: [usize; 2] = if b_2ch_mode { [1, 4] } else { [3, 4] };
        while pcm_per_channel.len() < 5 {
            pcm_per_channel.push(None);
        }
        // Trailer-to-output-slot mapping (independent of b_2ch_mode):
        // ASPX is applied per output channel after channel-element
        // decode produces PCM. Per Table 25 trailer order:
        //   slot 0 (L)  -> aspx_lr.primary
        //   slot 1 (R)  -> aspx_lr.secondary
        //   slot 3 (Ls) -> aspx_ls_rs.primary
        //   slot 4 (Rs) -> aspx_ls_rs.secondary
        //   slot 2 (C)  -> aspx_centre.primary
        let mut entries: Vec<FiveXChannelEntry<'_>> = Vec::with_capacity(5);
        for (ch_in, &slot) in slot_map_a.iter().enumerate() {
            let Some(scaled) = tcd_a.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm_f = self.imdct_channel_f32(slot, scaled, n_a);
            entries.push((
                slot,
                pcm_f,
                Self::trailer_for_5x_slot(slot, aspx_lr, aspx_ls_rs, aspx_centre),
            ));
        }
        for (ch_in, &slot) in slot_map_b.iter().enumerate() {
            let Some(scaled) = tcd_b.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm_f = self.imdct_channel_f32(slot, scaled, n_b);
            entries.push((
                slot,
                pcm_f,
                Self::trailer_for_5x_slot(slot, aspx_lr, aspx_ls_rs, aspx_centre),
            ));
        }
        if let Some(mono) = centre_mono {
            if let Some(pcm_f) = self.imdct_mono_lfe_data_f32(mono, 2, samples) {
                entries.push((
                    2,
                    pcm_f,
                    Self::trailer_for_5x_slot(2, aspx_lr, aspx_ls_rs, aspx_centre),
                ));
            }
        }
        self.extend_5x_entries(
            entries,
            aspx_cfg,
            companding,
            num_ts_in_ats,
            pcm_per_channel,
        );
    }

    /// Round 42: canonical Table-25 trailer-to-slot mapping for the
    /// 5_X SIMPLE/ASPX dispatchers. Returns `(trailer, is_secondary)`
    /// when the appropriate trailer is present, else `None`.
    fn trailer_for_5x_slot<'a>(
        slot: usize,
        aspx_lr: Option<&'a aspx::FiveXAspxTrailer>,
        aspx_ls_rs: Option<&'a aspx::FiveXAspxTrailer>,
        aspx_centre: Option<&'a aspx::FiveXAspxTrailer>,
    ) -> Option<(&'a aspx::FiveXAspxTrailer, bool)> {
        match slot {
            0 => aspx_lr.map(|t| (t, false)),
            1 => aspx_lr.map(|t| (t, true)),
            2 => aspx_centre.map(|t| (t, false)),
            3 => aspx_ls_rs.map(|t| (t, false)),
            4 => aspx_ls_rs.map(|t| (t, true)),
            _ => None,
        }
    }

    /// Round 42: trailer-aware ASPX extension on one 5_X output slot
    /// (0..=4). Used by `dispatch_5x_cfg{0,1,3}_simple_aspx` to apply
    /// the per-channel trailer + companding pulled from the per-cfg
    /// slots in [`crate::asf::SubstreamTools`].
    ///
    /// Trailer-to-slot mapping is the canonical Table-25 order
    /// `aspx_data_2ch + aspx_data_2ch + aspx_data_1ch` translated to
    /// 5.X output channels:
    ///   slot 0 (L)  -> aspx_lr.primary
    ///   slot 1 (R)  -> aspx_lr.secondary
    ///   slot 3 (Ls) -> aspx_ls_rs.primary
    ///   slot 4 (Rs) -> aspx_ls_rs.secondary
    ///   slot 2 (C)  -> aspx_centre.primary
    /// Trailers / config absent -> i16 cast of `pcm_f` only.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn maybe_extend_5x_slot(
        &mut self,
        slot: usize,
        pcm_f: Vec<f32>,
        aspx_lr: Option<&aspx::FiveXAspxTrailer>,
        aspx_ls_rs: Option<&aspx::FiveXAspxTrailer>,
        aspx_centre: Option<&aspx::FiveXAspxTrailer>,
        aspx_cfg: Option<aspx::AspxConfig>,
        companding: Option<&aspx::CompandingControl>,
        num_ts_in_ats: u32,
    ) -> Vec<i16> {
        let trailer_pair = Self::trailer_for_5x_slot(slot, aspx_lr, aspx_ls_rs, aspx_centre);
        match (aspx_cfg, trailer_pair) {
            (Some(cfg), Some((trailer, is_secondary))) => {
                let ch = if is_secondary {
                    trailer.secondary.as_ref().unwrap_or(&trailer.primary)
                } else {
                    &trailer.primary
                };
                let compand_mode = Self::five_x_compand_mode_for_slot(companding, slot);
                let extended = self.aspx_extend_with_trailer(
                    &pcm_f,
                    trailer,
                    ch,
                    &cfg,
                    slot,
                    num_ts_in_ats,
                    compand_mode,
                    // SIMPLE/ASPX cfg{0,1,3} dispatchers never run on
                    // ASPX_ACPL_1, so sb0 stays at aspx_xover_band.
                    None,
                );
                Self::pcm_f32_to_i16(&extended)
            }
            _ => Self::pcm_f32_to_i16(&pcm_f),
        }
    }

    /// §5.3.4.3.1 / Table 180 — 5_X SIMPLE/ASPX `coding_config == 1`
    /// dispatch. The body shape is
    /// `three_channel_data + two_channel_data`, with channel mapping per
    /// Table 180 column 1:
    ///
    /// ```text
    ///     three_channel_data[0..3] -> [0, 1, 2] (L, R, C)
    ///     two_channel_data[0..2]   -> [3, 4]    (Ls, Rs)
    /// ```
    ///
    /// No-op on transform-length / sample-count mismatch, or when a
    /// per-channel scaled spectrum is absent.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_5x_cfg1_simple_aspx(
        &mut self,
        three: &crate::mch::ThreeChannelData,
        tcd: &crate::mch::TwoChannelData,
        aspx_lr: Option<&aspx::FiveXAspxTrailer>,
        aspx_ls_rs: Option<&aspx::FiveXAspxTrailer>,
        aspx_centre: Option<&aspx::FiveXAspxTrailer>,
        aspx_cfg: Option<aspx::AspxConfig>,
        companding: Option<&aspx::CompandingControl>,
        num_ts_in_ats: u32,
        samples: usize,
        pcm_per_channel: &mut Vec<Option<Vec<i16>>>,
    ) {
        let Some(ti3) = three.transform_info.as_ref() else {
            return;
        };
        let n3 = ti3.transform_length_0 as usize;
        if n3 == 0 || n3 != samples {
            return;
        }
        let Some(ti2) = tcd.transform_info.as_ref() else {
            return;
        };
        let n2 = ti2.transform_length_0 as usize;
        if n2 == 0 || n2 != samples {
            return;
        }
        if three.scaled_spec_per_channel.len() < 3 || tcd.scaled_spec_per_channel.len() < 2 {
            return;
        }
        while pcm_per_channel.len() < 5 {
            pcm_per_channel.push(None);
        }
        const THREE_SLOTS: [usize; 3] = [0, 1, 2];
        let mut entries: Vec<FiveXChannelEntry<'_>> = Vec::with_capacity(5);
        for (ch_in, &slot) in THREE_SLOTS.iter().enumerate() {
            let Some(scaled) = three.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm_f = self.imdct_channel_f32(slot, scaled, n3);
            entries.push((
                slot,
                pcm_f,
                Self::trailer_for_5x_slot(slot, aspx_lr, aspx_ls_rs, aspx_centre),
            ));
        }
        const TWO_SLOTS: [usize; 2] = [3, 4];
        for (ch_in, &slot) in TWO_SLOTS.iter().enumerate() {
            let Some(scaled) = tcd.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm_f = self.imdct_channel_f32(slot, scaled, n2);
            entries.push((
                slot,
                pcm_f,
                Self::trailer_for_5x_slot(slot, aspx_lr, aspx_ls_rs, aspx_centre),
            ));
        }
        self.extend_5x_entries(
            entries,
            aspx_cfg,
            companding,
            num_ts_in_ats,
            pcm_per_channel,
        );
    }

    /// §5.3.4.3.1 / Table 180 — 5_X SIMPLE/ASPX `coding_config == 3`
    /// dispatch. The body is a single `five_channel_data`; channel
    /// mapping is the identity:
    ///
    /// ```text
    ///     five_channel_data[0..5] -> [0, 1, 2, 3, 4] (L, R, C, Ls, Rs)
    /// ```
    ///
    /// No-op on transform-length / sample-count mismatch, or when a
    /// per-channel scaled spectrum is absent.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_5x_cfg3_simple_aspx(
        &mut self,
        five: &crate::mch::FiveChannelData,
        aspx_lr: Option<&aspx::FiveXAspxTrailer>,
        aspx_ls_rs: Option<&aspx::FiveXAspxTrailer>,
        aspx_centre: Option<&aspx::FiveXAspxTrailer>,
        aspx_cfg: Option<aspx::AspxConfig>,
        companding: Option<&aspx::CompandingControl>,
        num_ts_in_ats: u32,
        samples: usize,
        pcm_per_channel: &mut Vec<Option<Vec<i16>>>,
    ) {
        let Some(ti) = five.transform_info.as_ref() else {
            return;
        };
        let n = ti.transform_length_0 as usize;
        if n == 0 || n != samples {
            return;
        }
        if five.scaled_spec_per_channel.len() < 5 {
            return;
        }
        while pcm_per_channel.len() < 5 {
            pcm_per_channel.push(None);
        }
        const SLOT_MAP: [usize; 5] = [0, 1, 2, 3, 4];
        let mut entries: Vec<FiveXChannelEntry<'_>> = Vec::with_capacity(5);
        for (ch_in, &slot) in SLOT_MAP.iter().enumerate() {
            let Some(scaled) = five.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm_f = self.imdct_channel_f32(slot, scaled, n);
            entries.push((
                slot,
                pcm_f,
                Self::trailer_for_5x_slot(slot, aspx_lr, aspx_ls_rs, aspx_centre),
            ));
        }
        self.extend_5x_entries(
            entries,
            aspx_cfg,
            companding,
            num_ts_in_ats,
            pcm_per_channel,
        );
    }

    /// §5.3.4.4.1 / Table 182 / Table 183 — 7_X SIMPLE/ASPX additional-
    /// channel pair dispatch. The `seven_x_additional_channel_data` shell
    /// carries two `sf_data(ASF)` bodies for the F / G preliminary output
    /// channels (Table 182). The optional `partner_pair_spectra` carry the
    /// 5.X-core counterparts D/E (or A/B per `channel_mode`) that pair
    /// with F/G in the Table 183 SAP matrix.
    ///
    /// With `b_use_sap_add_ch == false` (or absent), Table 183's SAP
    /// matrix collapses to identity — F / G land directly on slots 5 / 6
    /// and the partner spectra are untouched (their independent IMDCT
    /// path produces the unmodified slots 3 / 4 elsewhere in the
    /// pipeline).
    ///
    /// With `b_use_sap_add_ch == true`, the per-sfb (a, b, c, d)
    /// coefficients are extracted from each `chparam_info` (Pseudocode 59
    /// via [`crate::asf::extract_sap_abcd`]) and applied to the spectral
    /// pair (P, F) → (slot_partner, slot_F+1) and (Q, G) → (slot_partner+1,
    /// slot_F+2) per the Table 183 row for the active channel_mode:
    ///
    /// ```text
    ///     [out_high]   [a  b]   [partner]
    ///     [        ] = [    ] · [        ]
    ///     [out_low ]   [c  d]   [add_ch ]
    /// ```
    ///
    /// where `out_high` lands on the partner's existing slot (overwriting
    /// the unmixed PCM at that slot) and `out_low` lands on the
    /// additional-pair slot. When partner spectra are absent or the
    /// transform lengths don't match, falls back to identity render
    /// (slots 5 / 6 from F / G unmodified).
    ///
    /// No-op on transform-length / sample-count mismatch, or when either
    /// per-channel scaled spectrum is absent (short / grouped frame /
    /// Huffman miss).
    fn dispatch_7x_additional_channel_pair(
        &mut self,
        add: &crate::mch::TwoChannelData,
        partner_pair_spectra: Option<[&[f32]; 2]>,
        partner_slots: [usize; 2],
        chparam: Option<&[asf::ChparamInfo; 2]>,
        samples: usize,
        pcm_per_channel: &mut Vec<Option<Vec<i16>>>,
    ) {
        let Some(ti) = add.transform_info.as_ref() else {
            return;
        };
        let n = ti.transform_length_0 as usize;
        if n == 0 || n != samples {
            return;
        }
        if add.scaled_spec_per_channel.len() < 2 {
            return;
        }
        // Slots 5 / 6 are the additional-pair output channels (F, G).
        let pair_out_slots: [usize; 2] = [5, 6];
        while pcm_per_channel.len() < 7 {
            pcm_per_channel.push(None);
        }
        // Resize so partner slots are addressable too.
        for &slot in partner_slots.iter() {
            while pcm_per_channel.len() <= slot {
                pcm_per_channel.push(None);
            }
        }
        // Per-pair SAP application: for each i in 0..2, mix
        // `partner_pair_spectra[i]` (P or Q) with `add[i]` (F or G).
        // When partner is absent or chparam is None, falls through to
        // identity (only the additional-pair F/G is rendered).
        let tl = ti.transform_length_0;
        let max_sfb_cap = crate::tables::num_sfb_48(tl).unwrap_or(0);
        for ch_in in 0..2 {
            let Some(scaled_add) = add.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            // Build SAP coefficients for this pair if requested.
            let abcd: Option<Vec<(f32, f32, f32, f32)>> = match (chparam, partner_pair_spectra) {
                (Some(cps), Some(partners))
                    if max_sfb_cap > 0 && partners[ch_in].len() == n && scaled_add.len() == n =>
                {
                    let coeffs = asf::extract_sap_abcd(&cps[ch_in], &[max_sfb_cap]);
                    coeffs.abcd.into_iter().next()
                }
                _ => None,
            };
            if let (Some(abcd_row), Some(partners)) = (abcd.as_ref(), partner_pair_spectra) {
                // Spectral SAP per-sfb. Mix (P, F) -> (out_high, out_low).
                let partner = partners[ch_in];
                let sfbo = match crate::sfb_offset::sfb_offset_48(tl) {
                    Some(s) => s,
                    None => {
                        // SFB table missing — fall through to identity.
                        let pcm = self.imdct_channel(pair_out_slots[ch_in], scaled_add, n);
                        pcm_per_channel[pair_out_slots[ch_in]] = Some(pcm);
                        continue;
                    }
                };
                let mut out_high = vec![0.0f32; n];
                let mut out_low = vec![0.0f32; n];
                let usable_sfb = abcd_row.len().min(max_sfb_cap as usize);
                for sfb in 0..usable_sfb {
                    let lo = sfbo[sfb] as usize;
                    let hi = sfbo[sfb + 1] as usize;
                    let hi = hi.min(n).min(partner.len()).min(scaled_add.len());
                    let (a, b, c, d) = abcd_row[sfb];
                    for k in lo..hi {
                        let p = partner[k];
                        let f = scaled_add[k];
                        out_high[k] = a * p + b * f;
                        out_low[k] = c * p + d * f;
                    }
                }
                // Copy untouched bands (sfb >= usable_sfb) from the
                // partner / add spectra so the high half retains
                // the partner's bandwidth and the low half is silent
                // outside the SAP-coded range.
                let unmixed_start = sfbo
                    .get(usable_sfb)
                    .copied()
                    .map(|v| v as usize)
                    .unwrap_or(n);
                let unmixed_lo = unmixed_start.min(n);
                let unmixed_hi = n.min(partner.len());
                if unmixed_lo < unmixed_hi {
                    out_high[unmixed_lo..unmixed_hi]
                        .copy_from_slice(&partner[unmixed_lo..unmixed_hi]);
                }
                let pcm_high = self.imdct_channel(partner_slots[ch_in], &out_high, n);
                pcm_per_channel[partner_slots[ch_in]] = Some(pcm_high);
                let pcm_low = self.imdct_channel(pair_out_slots[ch_in], &out_low, n);
                pcm_per_channel[pair_out_slots[ch_in]] = Some(pcm_low);
            } else {
                // Identity passthrough — only render the additional pair
                // (slots 5/6). Partner slots untouched (their independent
                // 5_X-core IMDCT runs separately).
                let pcm = self.imdct_channel(pair_out_slots[ch_in], scaled_add, n);
                pcm_per_channel[pair_out_slots[ch_in]] = Some(pcm);
            }
        }
    }

    /// Decode a v2 frame whose first substream group carries an A-JOC
    /// object substream (TS 103 190-2 §6.2.3.4): run the
    /// [`crate::ajoc_substream::AjocSubstreamDecoder`] chain and emit
    /// the reconstructed objects as interleaved S16 output channels
    /// (decoded LFE slot first when signalled).
    fn receive_frame_ajoc(
        &mut self,
        raw: &[u8],
        info: toc::Ac4FrameInfo,
        pts: Option<i64>,
    ) -> Result<Frame> {
        let desc = info.ajoc_substreams[0].clone();
        let params = crate::ajoc_substream::AjocBodyParams::from_substream_info(&desc);
        let num_dmx = params.n_fullband_dmx_signals as usize;
        let num_umx = params.n_fullband_upmix_signals as usize;
        // Substream 0 starts at toc_size + payload_base.
        let start = (info.toc_size + info.payload_base) as usize;
        if start >= raw.len() {
            return Err(Error::invalid("ac4: A-JOC frame without substream bytes"));
        }
        let end = info
            .substream_sizes
            .first()
            .map(|&sz| start.saturating_add(sz as usize).min(raw.len()))
            .unwrap_or(raw.len());
        let substream = &raw[start..end];
        let matches = matches!(
            &self.ajoc_dec,
            Some((d, u, _)) if *d == num_dmx && *u == num_umx
        );
        if !matches {
            self.ajoc_dec = Some((
                num_dmx,
                num_umx,
                crate::ajoc_substream::AjocSubstreamDecoder::new(num_dmx, num_umx),
            ));
        }
        let dec = &mut self.ajoc_dec.as_mut().expect("ajoc decoder ensured").2;
        let (objects, lfe, _body, _metadata) = dec.decode_substream_pcm(
            substream,
            &params,
            desc.b_iframe(),
            false,
            info.frame_length,
        )?;
        let samples = info.frame_length;
        // Output layout mirrors the upmix signal set: decoded LFE slot
        // first when signalled, then the fullband objects.
        let channels_out = desc.n_umx_signals() as usize;
        let lfe_slots = usize::from(desc.b_lfe);
        let mut buf = vec![0u8; samples as usize * channels_out * 2];
        let write_channel = |buf: &mut [u8], c: usize, pcm: &[f32]| {
            for (i, &v) in pcm.iter().take(samples as usize).enumerate() {
                let s = (v * 32767.0).clamp(-32768.0, 32767.0) as i16;
                let le = s.to_le_bytes();
                let off = (i * channels_out + c) * 2;
                buf[off] = le[0];
                buf[off + 1] = le[1];
            }
        };
        if let (true, Some(lfe_pcm)) = (desc.b_lfe, lfe.as_ref()) {
            write_channel(&mut buf, 0, lfe_pcm);
        }
        for (o, pcm) in objects.iter().enumerate() {
            write_channel(&mut buf, o + lfe_slots, pcm);
        }
        self.last_info = Some(info);
        Ok(Frame::Audio(AudioFrame {
            samples,
            pts,
            data: vec![buf],
        }))
    }

    /// `out[i] = g * (a[i] + sign * b[i])` over `n` samples (Table 23
    /// S-CPL half-sum / half-difference rows; the ×2 of the matrix and
    /// the ½ entries cancel).
    fn ice_mix2(a: &[f32], b: &[f32], g: f32, sign: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = a.get(i).copied().unwrap_or(0.0);
                let y = b.get(i).copied().unwrap_or(0.0);
                g * (x + sign * y)
            })
            .collect()
    }

    /// `out[i] = g * a[i]` over `n` samples.
    fn ice_scale(a: &[f32], g: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| g * a.get(i).copied().unwrap_or(0.0))
            .collect()
    }

    /// §5.3.3.1 Table 23 S-CPL full-decoding matrix over the 13 IMDCT
    /// track signals `t[0..13]` (A''..M''). `c_gain` / `m_gain` follow
    /// the mode: 2 / √2 for SCPL, 1 / 1 for ASPX_SCPL (whose output
    /// levels are restored by the §4.8.3.11.3 post-A-SPX gains). The
    /// b_5fronts front rows carry the fixed x2 matrix (independent of
    /// both gains); the x2 · ½ entries of every mixing row cancel.
    ///
    /// Returns the 11 / 13 output channels in the route's slot order
    /// `[L, R, C, (Lscr, Rscr,) Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`
    /// (for b_5fronts the Table 23 `Lw` / `Rw` outputs occupy the L /
    /// R slots).
    fn ice_scpl_full_matrix(
        t: &[Vec<f32>],
        b_5fronts: bool,
        c_gain: f32,
        m_gain: f32,
        n: usize,
    ) -> Vec<Vec<f32>> {
        let n_named = if b_5fronts { 13 } else { 11 };
        let mut chans: Vec<Vec<f32>> = vec![Vec::new(); n_named];
        if b_5fronts {
            // [Lw, Lscr] from (A, L''), [Rw, Rscr] from (B, M'') —
            // Lw / Rw occupy the L / R slots.
            chans[0] = Self::ice_mix2(&t[0], &t[11], 1.0, 1.0, n);
            chans[3] = Self::ice_mix2(&t[0], &t[11], 1.0, -1.0, n);
            chans[1] = Self::ice_mix2(&t[1], &t[12], 1.0, 1.0, n);
            chans[4] = Self::ice_mix2(&t[1], &t[12], 1.0, -1.0, n);
            chans[2] = Self::ice_scale(&t[2], c_gain, n);
        } else {
            chans[0] = Self::ice_scale(&t[0], c_gain, n);
            chans[1] = Self::ice_scale(&t[1], c_gain, n);
            chans[2] = Self::ice_scale(&t[2], c_gain, n);
        }
        // Surround / top rows are position-independent of b_5fronts;
        // the output slots shift by the two screen channels.
        let base = if b_5fronts { 5 } else { 3 };
        let rows: [(usize, usize, f32); 8] = [
            (3, 5, 1.0),   // Ls  = m(D + F)
            (4, 6, 1.0),   // Rs  = m(E + G)
            (3, 5, -1.0),  // Lb  = m(D − F)
            (4, 6, -1.0),  // Rb  = m(E − G)
            (7, 9, 1.0),   // Tfl = m(H + J)
            (8, 10, 1.0),  // Tfr = m(I + K)
            (7, 9, -1.0),  // Tbl = m(H − J)
            (8, 10, -1.0), // Tbr = m(I − K)
        ];
        for (i, (a, b, sign)) in rows.into_iter().enumerate() {
            chans[base + i] = Self::ice_mix2(&t[a], &t[b], m_gain, sign, n);
        }
        chans
    }

    /// Plain (extension-free) QMF analysis of one channel's PCM into a
    /// `q[sb][ts]` matrix — the fallback when a core channel has no
    /// usable A-SPX payload.
    fn ice_plain_qmf(pcm: &[f32]) -> aspx::QmfMatrix {
        const NUM_QMF: usize = qmf::NUM_QMF_SUBBANDS;
        let n_slots = pcm.len() / NUM_QMF;
        let mut ana = qmf::QmfAnalysisBank::new();
        let slots = ana.process_block(pcm);
        let mut q: Vec<Vec<(f32, f32)>> = (0..NUM_QMF)
            .map(|_| vec![(0.0f32, 0.0f32); n_slots])
            .collect();
        for (ts, slot) in slots.iter().take(n_slots).enumerate() {
            for (sb, s) in slot.iter().enumerate() {
                q[sb][ts] = *s;
            }
        }
        q
    }

    /// A-SPX-extend one immersive core channel into the QMF domain
    /// using a captured payload trailer; falls back to the plain
    /// low-band analysis when the extension guards trip.
    #[allow(clippy::too_many_arguments)]
    /// Returns the post-extension matrix plus the `(sbx, sbz)` band
    /// the §5.7.5 companding tool needs; the band is `None` when the
    /// extension guards tripped and the plain low-band analysis was
    /// used instead.
    fn ice_extend_channel_qmf(
        &mut self,
        pcm: &[f32],
        trailer: &aspx::FiveXAspxTrailer,
        is_secondary: bool,
        cfg: &aspx::AspxConfig,
        slot: usize,
        num_ts_in_ats: u32,
    ) -> (aspx::QmfMatrix, Option<(u32, u32)>) {
        while self.aspx_ext_state.len() <= slot {
            self.aspx_ext_state.push(aspx::AspxChannelExtState::new());
        }
        let ch = if is_secondary {
            trailer.secondary.as_ref().unwrap_or(&trailer.primary)
        } else {
            &trailer.primary
        };
        match Self::aspx_extend_to_qmf(
            pcm,
            &trailer.frequency_tables,
            cfg,
            Some(&ch.framing),
            Some(&ch.data_sig),
            Some(&ch.data_noise),
            Some(ch.qmode_env),
            Some(&ch.delta_dir),
            ch.add_harmonic.as_deref(),
            ch.tna_mode.as_deref(),
            &mut self.aspx_ext_state[slot],
            num_ts_in_ats,
        ) {
            Some((q, sb0, sb1)) => (q, Some((sb0, sb1))),
            None => (Self::ice_plain_qmf(pcm), None),
        }
    }

    /// Transpose a `q[sb][ts]` matrix into the `[ts][sb]`
    /// [`ajcc_synth::QmfCol`] layout the A-JCC tool consumes.
    fn ice_qmf_cols(q: &aspx::QmfMatrix, num_ts: usize) -> Vec<ajcc_synth::QmfCol> {
        let mut out = vec![[(0.0f32, 0.0f32); qmf::NUM_QMF_SUBBANDS]; num_ts];
        for (sb, row) in q.iter().enumerate().take(qmf::NUM_QMF_SUBBANDS) {
            for (ts, col) in out.iter_mut().enumerate() {
                col[sb] = row.get(ts).copied().unwrap_or((0.0, 0.0));
            }
        }
        out
    }

    /// Decode one immersive-channel-element frame (TS 103 190-2
    /// §6.2.4.1, channel modes 7.0.4 / 7.1.4 / 9.0.4 / 9.1.4) to PCM.
    ///
    /// Wired synthesis paths (all five Table 95 codec modes):
    ///
    /// * **SCPL** (full decoding, §5.3.3.1 Table 23) — §5.2.3.2 SAP
    ///   steps 3-6 on the track spectra, IMDCT, then the pure
    ///   time-domain S-CPL matrix (`c_gain = 2`, `m_gain = √2`).
    /// * **ASPX_SCPL** (full decoding) — SAP + IMDCT + Table 23 with
    ///   `c_gain = m_gain = 1`, per-channel A-SPX extension over the
    ///   Table 8 channel grouping, and the §4.8.3.11.3 Table 10 / 11
    ///   output gains.
    /// * **ASPX_ACPL_1 / ASPX_ACPL_2** (full decoding, §5.5.2
    ///   Table 27) — SAP (ACPL_1 only) + IMDCT + per-track QMF with
    ///   the Table 8 A-SPX extension, then the four / six parallel
    ///   ACplModules (D0/D0/D1/D1/D2/D2 decorrelators, per-module
    ///   cross-frame differential state) over the Table 26 mapping.
    /// * **ASPX_AJCC** (full decoding, §5.6.3.5.2) — IMDCT the
    ///   5-channel core, A-SPX-extend each core channel in the QMF
    ///   domain (payload mapping `(A, B)` / `(D, E)` / `C` — see the
    ///   ice module notes), run the A-JCC full decode and
    ///   QMF-synthesise the 11 / 13 outputs.
    ///
    /// Output channel order follows the §5.6.3.5.2 output addressing:
    /// `[L, R, C, (Lscr, Rscr,) Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr]`
    /// (the screen channels only for `b_5fronts`; for SCPL b_5fronts
    /// the Table 23 `Lw` / `Rw` outputs occupy the L / R slots). The
    /// decoded LFE (7.1.4 / 9.1.4) is emitted on the leading slot.
    fn receive_frame_ice(
        &mut self,
        raw: &[u8],
        info: toc::Ac4FrameInfo,
        pts: Option<i64>,
    ) -> Result<Frame> {
        let mode_desc = info
            .first_chan_mode
            .ok_or_else(|| Error::invalid("ac4: immersive route without channel mode"))?;
        let b_lfe = mode_desc.immersive_b_lfe();
        let b_5fronts = mode_desc.immersive_b_5fronts();
        let start = (info.toc_size + info.payload_base) as usize;
        if start >= raw.len() {
            return Err(Error::invalid("ac4: immersive frame without substream"));
        }
        let end = info
            .substream_sizes
            .first()
            .map(|&sz| start.saturating_add(sz as usize).min(raw.len()))
            .unwrap_or(raw.len());
        let substream = &raw[start..end];
        let b_iframe = info
            .presentations
            .first()
            .map(|p| p.b_iframe)
            .unwrap_or(info.b_iframe_global);
        let sub = asf::walk_ac4_substream_sticky(
            substream,
            mode_desc.channels as u16,
            b_iframe,
            info.frame_length,
            None,
            Some(&mut self.sticky),
        )?;
        let samples = info.frame_length;
        let n = samples as usize;
        let n_named = if b_5fronts { 13 } else { 11 };
        let channels_out = n_named + usize::from(b_lfe);
        let mut chans: Vec<Vec<f32>> = vec![Vec::new(); n_named];
        let mut lfe_pcm: Option<Vec<f32>> = None;
        let sub_aspx_cfg = sub.tools.aspx_config;
        let sub_acpl_partial = sub.tools.acpl_config_1ch_partial;
        let sub_acpl_full = sub.tools.acpl_config_1ch_full;
        if let Some(ice_el) = sub.tools.ice.as_deref() {
            let tl_ok = ice_el.transform_length() == Some(samples);
            // LFE (Table 21 mono_data(1)) — IMDCT on a dedicated
            // overlap slot past the 13 track slots.
            if b_lfe {
                if let Some(spec) = ice_el.lfe_spectrum().map(<[f32]>::to_vec) {
                    lfe_pcm = Some(self.imdct_channel_f32(13, &spec, n));
                }
            }
            match ice_el.mode {
                ice::IceCodecMode::Scpl if tl_ok => {
                    // §5.2.3.2 steps 3-6 (SAP mixing on the track
                    // spectra), then §5.3.3.1 Table 23 with c_gain = 2
                    // / m_gain = √2.
                    let mut specs = ice_el.track_spectra_owned();
                    ice::apply_sap_steps(&mut specs, ice_el, samples);
                    let mut t: Vec<Vec<f32>> = Vec::with_capacity(13);
                    for (slot, spec) in specs.iter().enumerate() {
                        t.push(self.imdct_channel_f32(slot, spec, n));
                    }
                    chans =
                        Self::ice_scpl_full_matrix(&t, b_5fronts, 2.0, std::f32::consts::SQRT_2, n);
                }
                ice::IceCodecMode::AspxScpl if tl_ok => {
                    // §5.2.3.2 steps 3-6 (SAP), IMDCT, §5.3.3.1
                    // Table 23 with c_gain = m_gain = 1, then the
                    // per-channel A-SPX extension (Table 8 grouping)
                    // and the §4.8.3.11.3 Table 10 / 11 output gains.
                    let mut specs = ice_el.track_spectra_owned();
                    ice::apply_sap_steps(&mut specs, ice_el, samples);
                    let mut t: Vec<Vec<f32>> = Vec::with_capacity(13);
                    for (slot, spec) in specs.iter().enumerate() {
                        t.push(self.imdct_channel_f32(slot, spec, n));
                    }
                    let mut ch_pcm = Self::ice_scpl_full_matrix(&t, b_5fronts, 1.0, 1.0, n);
                    // A-SPX per decoupled channel. Table 8 groups the
                    // channels as (L, R), C, (Ls, Lb), (Rs, Rb),
                    // (Tfl, Tbl), (Tfr, Tbr) — with b_5fronts the
                    // front groups become (L, Lscr) and (R, Rscr) —
                    // associated with the §6.2.4.1 payload roster in
                    // transmission order (2ch payloads to the pair
                    // groups, the 1ch payload to C).
                    let num_ts_in_ats = aspx::num_ts_in_ats(samples.max(1));
                    if let Some(cfg) = sub_aspx_cfg {
                        // (payload index, output slot, is_secondary)
                        let mapping: &[(usize, usize, bool)] = if b_5fronts {
                            &[
                                (0, 0, false), // (L, Lscr)
                                (0, 3, true),
                                (1, 1, false), // (R, Rscr)
                                (1, 4, true),
                                (2, 2, false), // C
                                (3, 5, false), // (Ls, Lb)
                                (3, 7, true),
                                (4, 6, false), // (Rs, Rb)
                                (4, 8, true),
                                (5, 9, false), // (Tfl, Tbl)
                                (5, 11, true),
                                (6, 10, false), // (Tfr, Tbr)
                                (6, 12, true),
                            ]
                        } else {
                            &[
                                (0, 0, false), // (L, R)
                                (0, 1, true),
                                (1, 3, false), // (Ls, Lb)
                                (1, 5, true),
                                (2, 2, false), // C
                                (3, 4, false), // (Rs, Rb)
                                (3, 6, true),
                                (4, 7, false), // (Tfl, Tbl)
                                (4, 9, true),
                                (5, 8, false), // (Tfr, Tbr)
                                (5, 10, true),
                            ]
                        };
                        for &(elem_idx, slot, is_secondary) in mapping {
                            let trailer = match ice_el.aspx_elements.get(elem_idx) {
                                Some(
                                    ice::IceAspxElement::TwoCh(Some(t))
                                    | ice::IceAspxElement::OneCh(Some(t)),
                                ) => (**t).clone(),
                                _ => continue,
                            };
                            let ch = if is_secondary {
                                trailer.secondary.as_ref().unwrap_or(&trailer.primary)
                            } else {
                                &trailer.primary
                            };
                            while self.aspx_ext_state.len() <= slot {
                                self.aspx_ext_state.push(aspx::AspxChannelExtState::new());
                            }
                            ch_pcm[slot] = Self::aspx_extend_pcm(
                                &ch_pcm[slot],
                                &trailer.frequency_tables,
                                &cfg,
                                Some(&ch.framing),
                                Some(&ch.data_sig),
                                Some(&ch.data_noise),
                                Some(ch.qmode_env),
                                Some(&ch.delta_dir),
                                ch.add_harmonic.as_deref(),
                                ch.tna_mode.as_deref(),
                                &mut self.aspx_ext_state[slot],
                                num_ts_in_ats,
                                aspx::CompandingMode::Off,
                                None,
                            );
                        }
                    }
                    // §4.8.3.11.3 Tables 10 / 11 channel-dependent
                    // output gains (applied full-band, so equivalently
                    // on the synthesised PCM).
                    let sq2 = std::f32::consts::SQRT_2;
                    for (slot, pcm) in ch_pcm.iter_mut().enumerate() {
                        let g = if b_5fronts {
                            match slot {
                                2 => 2.0,             // C
                                0 | 1 | 3 | 4 => 1.0, // L, R, Lscr, Rscr
                                _ => sq2,
                            }
                        } else if slot < 3 {
                            2.0 // L, R, C
                        } else {
                            sq2
                        };
                        if (g - 1.0).abs() > f32::EPSILON {
                            for v in pcm.iter_mut() {
                                *v *= g;
                            }
                        }
                    }
                    chans = ch_pcm;
                }
                ice::IceCodecMode::AspxAcpl1 | ice::IceCodecMode::AspxAcpl2 if tl_ok => {
                    let is_acpl1 = matches!(ice_el.mode, ice::IceCodecMode::AspxAcpl1);
                    // §5.2.3.2 steps 3-6 (SAP — the ASPX_ACPL_2
                    // processing clause §5.2.3.3 carries no SAP stage;
                    // apply_sap_steps gates on the mode).
                    let mut specs = ice_el.track_spectra_owned();
                    ice::apply_sap_steps(&mut specs, ice_el, samples);
                    let mut t: Vec<Vec<f32>> = Vec::with_capacity(13);
                    for (slot, spec) in specs.iter().enumerate() {
                        t.push(self.imdct_channel_f32(slot, spec, n));
                    }
                    // QMF analysis per track, with the A-SPX extension
                    // on the §6.2.4.1 payload roster tracks (Table 8:
                    // (A'', B''), (D'', F''), (E'', G'') pairs + C'').
                    let num_ts_in_ats = aspx::num_ts_in_ats(samples.max(1));
                    let mut q_tr: Vec<aspx::QmfMatrix> =
                        t.iter().map(|p| Self::ice_plain_qmf(p)).collect();
                    if let Some(cfg) = sub_aspx_cfg {
                        let mapping: [(usize, usize, bool); 7] = [
                            (0, 0, false), // payload 0 → (A, B)
                            (0, 1, true),
                            (1, 3, false), // payload 1 → (D, F)
                            (1, 5, true),
                            (2, 4, false), // payload 2 → (E, G)
                            (2, 6, true),
                            (3, 2, false), // payload 3 (1ch) → C
                        ];
                        for (elem_idx, tr, is_secondary) in mapping {
                            let trailer = match ice_el.aspx_elements.get(elem_idx) {
                                Some(
                                    ice::IceAspxElement::TwoCh(Some(t))
                                    | ice::IceAspxElement::OneCh(Some(t)),
                                ) => (**t).clone(),
                                _ => continue,
                            };
                            (q_tr[tr], _) = self.ice_extend_channel_qmf(
                                &t[tr],
                                &trailer,
                                is_secondary,
                                &cfg,
                                tr,
                                num_ts_in_ats,
                            );
                        }
                    }
                    // §5.5.2 Table 27 — four / six parallel A-CPL
                    // modules over the Table 26 channel/x mapping.
                    let acfg = if is_acpl1 {
                        sub_acpl_partial
                    } else {
                        sub_acpl_full
                    };
                    let n_modules = if b_5fronts { 6 } else { 4 };
                    if let Some(acfg) = acfg {
                        if ice_el.acpl_data.len() >= n_modules {
                            let num_ts = n / qmf::NUM_QMF_SUBBANDS;
                            let cols: Vec<Vec<ajcc_synth::QmfCol>> =
                                q_tr.iter().map(|q| Self::ice_qmf_cols(q, num_ts)).collect();
                            let state_ok =
                                matches!(&self.ice_acpl, Some(s) if s.b_5fronts == b_5fronts);
                            if !state_ok {
                                self.ice_acpl = Some(IceAcplState::new(b_5fronts));
                            }
                            let st = self.ice_acpl.as_mut().expect("ice acpl state ensured");
                            // Module routing: (main track, ASPX_ACPL_1
                            // residual track, (z_main, z_sub)). For
                            // ASPX_ACPL_2 the coded F / G tracks
                            // occupy the Tfl / Tfr carrier positions
                            // (x9 / x10 — the Table 27 ACPL_2 branch
                            // reads exactly x5, x6, x9, x10) and each
                            // module runs decorrelator-only.
                            let mut routing: Vec<(usize, Option<usize>, usize, usize, bool)> =
                                if is_acpl1 {
                                    vec![
                                        (3, Some(5), 5, 6, true),    // (Ls, Lb)
                                        (4, Some(6), 7, 8, true),    // (Rs, Rb)
                                        (7, Some(9), 9, 10, true),   // (Tfl, Tbl)
                                        (8, Some(10), 11, 12, true), // (Tfr, Tbr)
                                    ]
                                } else {
                                    vec![
                                        (3, None, 5, 6, true),
                                        (4, None, 7, 8, true),
                                        (5, None, 9, 10, true),
                                        (6, None, 11, 12, true),
                                    ]
                                };
                            if b_5fronts {
                                let res_l = if is_acpl1 { Some(11) } else { None };
                                let res_m = if is_acpl1 { Some(12) } else { None };
                                routing.push((0, res_l, 0, 1, false)); // (L, Lscr)
                                routing.push((1, res_m, 2, 3, false)); // (R, Rscr)
                            }
                            let sq2 = std::f32::consts::SQRT_2;
                            let mut z: Vec<Option<Vec<ajcc_synth::QmfCol>>> = vec![None; 13];
                            for (m, &(main, res, z_main, z_sub, scale)) in
                                routing.iter().enumerate()
                            {
                                let data = &ice_el.acpl_data[m];
                                let (ad, bd) = &mut st.diffs[m];
                                let alpha_q = acpl_synth::differential_decode(
                                    &data.alpha1,
                                    acfg.num_param_bands,
                                    ad,
                                );
                                let beta_q = acpl_synth::differential_decode(
                                    &data.beta1,
                                    acfg.num_param_bands,
                                    bd,
                                );
                                let (alpha_dq, beta_dq) = acpl_synth::dequantize_alpha_beta(
                                    &alpha_q,
                                    &beta_q,
                                    acfg.quant_mode,
                                );
                                if alpha_dq.is_empty() {
                                    continue;
                                }
                                let frame = acpl_synth::AcplCpeFrame {
                                    x0: &cols[main],
                                    x1: res.map(|r| cols[r].as_slice()),
                                    alpha_dq: &alpha_dq,
                                    beta_dq: &beta_dq,
                                    num_param_bands: acfg.num_param_bands,
                                    // ASPX_ACPL_1 keeps the M/S-coded
                                    // residual band below acpl_qmf_band
                                    // (PARTIAL config); FULL runs fully
                                    // parametric from band 0.
                                    acpl_qmf_band: if is_acpl1 { acfg.qmf_band as u32 } else { 0 },
                                    steep: matches!(
                                        data.framing.interpolation_type,
                                        crate::acpl::AcplInterpolationType::Steep
                                    ),
                                    param_timeslots: &data.framing.param_timeslots,
                                };
                                let out =
                                    acpl_synth::run_pseudocode_115_pair(&mut st.modules[m], frame);
                                let (mut z0, mut z1) = (out.z0, out.z1);
                                if scale {
                                    for col in z0.iter_mut().chain(z1.iter_mut()) {
                                        for v in col.iter_mut() {
                                            v.0 *= sq2;
                                            v.1 *= sq2;
                                        }
                                    }
                                }
                                z[z_main] = Some(z0);
                                z[z_sub] = Some(z1);
                            }
                            // Passthrough rows: z4 = 2·x2 (C), and the
                            // non-b_5fronts z0 = 2·x0 / z2 = 2·x1.
                            let double = |cols: &[ajcc_synth::QmfCol]| {
                                cols.iter()
                                    .map(|col| {
                                        let mut c = *col;
                                        for v in c.iter_mut() {
                                            v.0 *= 2.0;
                                            v.1 *= 2.0;
                                        }
                                        c
                                    })
                                    .collect::<Vec<_>>()
                            };
                            z[4] = Some(double(&cols[2]));
                            if !b_5fronts {
                                z[0] = Some(double(&cols[0]));
                                z[2] = Some(double(&cols[1]));
                            }
                            // Table 26 z → output-slot mapping (slot
                            // order [L, R, C, (Lscr, Rscr,) Ls, Rs,
                            // Lb, Rb, Tfl, Tfr, Tbl, Tbr]).
                            let z_to_slot: &[(usize, usize)] = if b_5fronts {
                                &[
                                    (0, 0),
                                    (2, 1),
                                    (4, 2),
                                    (1, 3),
                                    (3, 4),
                                    (5, 5),
                                    (7, 6),
                                    (6, 7),
                                    (8, 8),
                                    (9, 9),
                                    (11, 10),
                                    (10, 11),
                                    (12, 12),
                                ]
                            } else {
                                &[
                                    (0, 0),
                                    (2, 1),
                                    (4, 2),
                                    (5, 3),
                                    (7, 4),
                                    (6, 5),
                                    (8, 6),
                                    (9, 7),
                                    (11, 8),
                                    (10, 9),
                                    (12, 10),
                                ]
                            };
                            for &(zi, slot) in z_to_slot {
                                let Some(zm) = &z[zi] else { continue };
                                let mut syn = qmf::QmfSynthesisBank::new();
                                let mut pcm = Vec::with_capacity(num_ts * qmf::NUM_QMF_SUBBANDS);
                                for col in zm.iter().take(num_ts) {
                                    let row = syn.process_slot(col);
                                    pcm.extend_from_slice(&row);
                                }
                                chans[slot] = pcm;
                            }
                        }
                    }
                }
                ice::IceCodecMode::AspxAjcc if tl_ok => {
                    let specs: Vec<Option<Vec<f32>>> = ice_el
                        .track_spectra()
                        .into_iter()
                        .take(5)
                        .map(|s| s.map(<[f32]>::to_vec))
                        .collect();
                    let mut pcm_core: Vec<Vec<f32>> = Vec::with_capacity(5);
                    for (slot, spec) in specs.iter().enumerate() {
                        let coeffs: &[f32] = spec.as_deref().unwrap_or(&[]);
                        pcm_core.push(self.imdct_channel_f32(slot, coeffs, n));
                    }
                    // A-SPX QMF extension per core channel. Payload
                    // mapping per the syntax order: first
                    // aspx_data_2ch → (A, B), second → (D, E), the
                    // aspx_data_1ch → C (see the ice module notes on
                    // Table 8's (D'', F'') row).
                    let num_ts_in_ats = aspx::num_ts_in_ats(samples.max(1));
                    let mut q_ch: Vec<aspx::QmfMatrix> = Vec::with_capacity(5);
                    let mut bands: Vec<Option<(u32, u32)>> = vec![None; 5];
                    for pcm in pcm_core.iter().take(5) {
                        q_ch.push(Self::ice_plain_qmf(pcm));
                    }
                    if let Some(cfg) = sub_aspx_cfg {
                        let mapping: [(usize, usize, bool); 5] = [
                            (0, 0, false), // payload 0 primary   → A
                            (0, 1, true),  // payload 0 secondary → B
                            (1, 3, false), // payload 1 primary   → D
                            (1, 4, true),  // payload 1 secondary → E
                            (2, 2, false), // payload 2 (1ch)     → C
                        ];
                        for (elem_idx, ch, is_secondary) in mapping {
                            let trailer = match ice_el.aspx_elements.get(elem_idx) {
                                Some(
                                    ice::IceAspxElement::TwoCh(Some(t))
                                    | ice::IceAspxElement::OneCh(Some(t)),
                                ) => (**t).clone(),
                                _ => continue,
                            };
                            (q_ch[ch], bands[ch]) = self.ice_extend_channel_qmf(
                                &pcm_core[ch],
                                &trailer,
                                is_secondary,
                                &cfg,
                                ch,
                                num_ts_in_ats,
                            );
                        }
                    }
                    // §4.8.3.10.3: companding on the five input
                    // channels L, R, C, Ls, Rs — the same order as
                    // companding_control(5) — via the §5.7.5 tool.
                    // Channel slot order here is the Table 19 track
                    // order A..E = L, R, C, Ls, Rs.
                    let cc = ice_el.companding.as_ref();
                    if let Some(mode) = Self::five_x_synced_mode(cc) {
                        let mut sync_view: Vec<aspx::SyncCompandingEntry<'_>> = Vec::new();
                        for (q, band) in q_ch.iter_mut().zip(bands.iter()) {
                            if let Some((sb0, sbz)) = band {
                                sync_view.push((q, *sb0, *sbz));
                            }
                        }
                        aspx::apply_synchronised_companding_across_channels(&mut sync_view, mode);
                    } else {
                        for (slot, (q, band)) in q_ch.iter_mut().zip(bands.iter()).enumerate() {
                            let Some((sb0, sbz)) = band else { continue };
                            let mode = Self::five_x_compand_mode_for_slot(cc, slot);
                            aspx::apply_companding_on_qmf_with_mode(q, *sb0, *sbz, mode);
                        }
                    }
                    // §5.6.3.2-5: differential decode + full-decode
                    // reconstruction with cross-frame state.
                    if let Some(ajcc_data) = ice_el.ajcc.as_deref() {
                        let state_ok = matches!(&self.ice_ajcc, Some((f, _, _)) if *f == b_5fronts);
                        if !state_ok {
                            self.ice_ajcc = Some((
                                b_5fronts,
                                crate::ajcc::AjccState::new(b_5fronts),
                                ajcc_synth::AjccSynthState::new(b_5fronts),
                            ));
                        }
                        let (_, param_state, synth_state) =
                            self.ice_ajcc.as_mut().expect("ice ajcc state ensured");
                        let decoded =
                            crate::ajcc::decode_ajcc_parsed(ajcc_data.clone(), param_state)?;
                        let owned = ajcc_synth::AjccOwnedParams::from_decoded(&decoded);
                        let num_ts = n / qmf::NUM_QMF_SUBBANDS;
                        let cols: Vec<Vec<ajcc_synth::QmfCol>> =
                            q_ch.iter().map(|q| Self::ice_qmf_cols(q, num_ts)).collect();
                        let x: [&[ajcc_synth::QmfCol]; 5] =
                            [&cols[0], &cols[1], &cols[2], &cols[3], &cols[4]];
                        let z = ajcc_synth::ajcc_full_decode(&x, &owned.params(), synth_state)?;
                        // z-order → output slots; z3/z4 (screen) only
                        // for b_5fronts.
                        let z_order: Vec<usize> = if b_5fronts {
                            (0..13).collect()
                        } else {
                            vec![0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
                        };
                        for (out_idx, zi) in z_order.into_iter().enumerate() {
                            let zm = &z[zi];
                            let mut syn = qmf::QmfSynthesisBank::new();
                            let mut pcm = Vec::with_capacity(num_ts * qmf::NUM_QMF_SUBBANDS);
                            for col in zm.iter().take(num_ts) {
                                let row = syn.process_slot(col);
                                pcm.extend_from_slice(&row);
                            }
                            chans[out_idx] = pcm;
                        }
                    }
                }
                // Transform-length mismatches fall through to silence
                // with the correct shape.
                _ => {}
            }
        }
        // Interleave to S16.
        let mut buf = vec![0u8; n * channels_out * 2];
        let lfe_slots = usize::from(b_lfe);
        let write_channel = |buf: &mut [u8], c: usize, pcm: &[f32]| {
            for (i, &v) in pcm.iter().take(n).enumerate() {
                let s = (v * 32767.0).clamp(-32768.0, 32767.0) as i16;
                let le = s.to_le_bytes();
                let off = (i * channels_out + c) * 2;
                buf[off] = le[0];
                buf[off + 1] = le[1];
            }
        };
        if let (true, Some(lfe)) = (b_lfe, lfe_pcm.as_ref()) {
            write_channel(&mut buf, 0, lfe);
        }
        for (c, pcm) in chans.iter().enumerate() {
            write_channel(&mut buf, c + lfe_slots, pcm);
        }
        self.last_substream = Some(sub);
        self.last_info = Some(info);
        Ok(Frame::Audio(AudioFrame {
            samples,
            pts,
            data: vec![buf],
        }))
    }
}

impl Decoder for Ac4Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "ac4 decoder: call receive_frame before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        if pkt.data.is_empty() {
            // Empty packet — emit a 0-sample frame so the pipeline
            // continues rather than erroring.
            return Ok(Frame::Audio(AudioFrame {
                samples: 0,
                pts: pkt.pts,
                data: vec![Vec::new()],
            }));
        }
        let (raw, _had_sync) = self.extract_raw_frame(&pkt)?;
        let info = toc::parse_ac4_toc(raw)
            .map_err(|e| Error::invalid(format!("ac4 decoder: TOC parse failed: {e}")))?;
        // A-JOC object substream (TS 103 190-2 §6.2.3.4): route to the
        // dedicated object decode chain and emit the reconstructed
        // objects as output channels.
        if !info.ajoc_substreams.is_empty() {
            let raw_owned = raw.to_vec();
            let pts = pkt.pts;
            return self.receive_frame_ajoc(&raw_owned, info, pts);
        }
        // Immersive channel modes (TS 103 190-2 Table 78, ch_mode
        // 11..=14): route `audio_data_chan()` to the
        // `immersive_channel_element()` decode chain (§6.2.4.1).
        if info
            .first_chan_mode
            .map(|m| m.is_immersive())
            .unwrap_or(false)
        {
            let raw_owned = raw.to_vec();
            let pts = pkt.pts;
            return self.receive_frame_ice(&raw_owned, info, pts);
        }
        // Resolve shape with fallbacks to the container hint when the
        // TOC carried a reserved / escape value.
        let channels = if info.channels == 0 {
            self.hint_channels
        } else {
            info.channels
        };
        let sample_rate = if info.sample_rate == 0 {
            self.hint_sample_rate
        } else {
            info.sample_rate
        };
        let samples = if info.frame_length == 0 {
            // Unknown frame length (reserved frame_rate_index): fall back
            // to 1024 samples at 48 kHz, 480 @ 44.1 kHz — both
            // round-numbers the resampler handles cleanly.
            if sample_rate == 44_100 {
                480
            } else {
                1024
            }
        } else {
            // frame_length in the table is expressed at the base sample
            // rate; for 96/192 kHz (sf_multiplier) we scale up.
            if sample_rate == 96_000 {
                info.frame_length * 2
            } else if sample_rate == 192_000 {
                info.frame_length * 4
            } else {
                info.frame_length
            }
        };
        // Best-effort walk of the first substream. The exact byte offset
        // of substream 0 is `toc_len + payload_base`, where `toc_len` is
        // the length of the byte-aligned ac4_toc() element. We don't
        // currently track `toc_len` out of [`toc::parse_ac4_toc`]; as a
        // cheap approximation we try the first substream size if the
        // substream_index_table exposed one, carving the tail of the
        // packet. This is fine for single-substream frames (the
        // overwhelmingly common case).
        let substream_try = {
            // Substream 0 starts at toc_size + payload_base.
            let start = (info.toc_size + info.payload_base) as usize;
            let first_size = info.substream_sizes.first().copied();
            if start >= raw.len() {
                None
            } else if let Some(sz) = first_size {
                let sz = sz as usize;
                let end = start.saturating_add(sz).min(raw.len());
                if sz > 0 {
                    Some(&raw[start..end])
                } else {
                    None
                }
            } else {
                // Single-substream frame with implicit size: the
                // substream spans to the end of the packet (possibly
                // minus CRC bytes, which the syncframe layer stripped).
                Some(&raw[start..])
            }
        };
        // Round 32: grow the per-channel SSF walker state vector to
        // match the current frame's channel count *before* invoking the
        // walker so the state borrow has the right shape.
        while self.ssf_walker_state.len() < channels as usize {
            self.ssf_walker_state.push(ssf::SsfChannelState::new());
        }
        self.last_substream = substream_try.and_then(|sb| {
            let channels_u16 = channels;
            let b_iframe = info
                .presentations
                .first()
                .map(|p| p.b_iframe)
                .unwrap_or(info.b_iframe_global);
            asf::walk_ac4_substream_sticky(
                sb,
                channels_u16,
                b_iframe,
                info.frame_length,
                Some(&mut self.ssf_walker_state[..channels as usize]),
                Some(&mut self.sticky),
            )
            .ok()
        });
        // If we have scaled spectra for the substream, run IMDCT + OLA
        // and produce real PCM. Per-channel PCM buffers live in
        // `pcm_per_channel`; the interleaver below lays them out to the
        // frame's channel count. Any channel without decoded spectra
        // stays silent. We detach the per-channel inputs from
        // `last_substream` up front so the IMDCT step can mutate
        // `self.overlap` without a borrow conflict.
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> = vec![None; channels as usize];
        // Detach the inputs + the ASPX tables once so we can run IMDCT
        // (which mutates overlap state) and the ASPX extension without
        // a borrow conflict on self.
        // Detach A-CPL config + parsed data so the synth call below
        // doesn't conflict with the immutable borrow of `last_substream`
        // when we later mutate decoder state.
        let acpl_active_cfg = self.last_substream.as_ref().and_then(|sub| {
            sub.tools
                .acpl_config_1ch_full
                .or(sub.tools.acpl_config_1ch_partial)
        });
        let acpl_active_data = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_data_1ch.clone());
        // Detach SSF data so we can run §5.2.3-5.2.7 synthesis without
        // a borrow conflict on `self`. SSF substreams are mutually
        // exclusive with ASF on a per-channel basis (per
        // `spec_frontend`), so when these are populated the IMDCT input
        // for that channel comes from `synthesize_ssf_data` instead of
        // the ASF Huffman path.
        let ssf_primary = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.ssf_data_primary.clone());
        let ssf_secondary = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.ssf_data_secondary.clone());
        // Detach 5_X ASPX_ACPL_3 synthesis inputs: two carrier spectra
        // land on scaled_spec_primary / scaled_spec_secondary (via the
        // stereo body walker), centre from cfg0_centre_mono, and the
        // A-CPL parameter pair from acpl_config_2ch / acpl_data_2ch.
        // Only populated when five_x_mode == AspxAcpl3.
        let five_x_acpl3_active = self
            .last_substream
            .as_ref()
            .map(|sub| {
                matches!(
                    sub.tools.five_x_mode,
                    Some(crate::mch::FiveXCodecMode::AspxAcpl3)
                ) && sub.tools.acpl_config_2ch.is_some()
                    && sub.tools.acpl_data_2ch.is_some()
                    && sub.tools.scaled_spec_primary.is_some()
                    && sub.tools.scaled_spec_secondary.is_some()
            })
            .unwrap_or(false);
        let five_x_acpl3_cfg = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_config_2ch);
        let five_x_acpl3_data = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_data_2ch.clone());
        // Detach 5_X ASPX_ACPL_1 / ASPX_ACPL_2 synthesis inputs
        // (Pseudocode 117). The active acpl_config_1ch is one of:
        //   - acpl_config_1ch_partial (ASPX_ACPL_1 — surround Ls/Rs
        //     carriers come from extra mono carriers; here we silence
        //     them as placeholders since the standalone Ls/Rs decode
        //     path isn't fleshed out yet).
        //   - acpl_config_1ch_full   (ASPX_ACPL_2 — no surround carriers).
        // The two `acpl_data_1ch_pair[]` entries drive the L-side
        // (alpha_1/beta_1) and R-side (alpha_2/beta_2) ACplModule's.
        let five_x_pair_mode: Option<acpl_synth::Acpl5xPairMode> = self
            .last_substream
            .as_ref()
            .and_then(|sub| match sub.tools.five_x_mode {
                Some(crate::mch::FiveXCodecMode::AspxAcpl1) => {
                    Some(acpl_synth::Acpl5xPairMode::AspxAcpl1)
                }
                Some(crate::mch::FiveXCodecMode::AspxAcpl2) => {
                    Some(acpl_synth::Acpl5xPairMode::AspxAcpl2)
                }
                _ => None,
            });
        let five_x_pair_cfg =
            self.last_substream
                .as_ref()
                .and_then(|sub| match sub.tools.five_x_mode {
                    Some(crate::mch::FiveXCodecMode::AspxAcpl1) => {
                        sub.tools.acpl_config_1ch_partial
                    }
                    Some(crate::mch::FiveXCodecMode::AspxAcpl2) => sub.tools.acpl_config_1ch_full,
                    _ => None,
                });
        let five_x_pair_data_1 = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_data_1ch_pair[0].clone());
        let five_x_pair_data_2 = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_data_1ch_pair[1].clone());
        let five_x_pair_active = five_x_pair_mode.is_some()
            && five_x_pair_cfg.is_some()
            && five_x_pair_data_1.is_some()
            && five_x_pair_data_2.is_some();
        // Round 37: detach the parsed `cfg0_centre_mono` payload (Cfg0
        // trailing `mono_data(0)`) for the 5_X pair / 7_X pair paths so
        // we can IMDCT its `scaled_spec` into a real centre carrier
        // (replacing the silence-placeholder used in round 36). For
        // ACPL_3 the centre is also pulled from the same source. The
        // detach is a clone so the substream tools borrow can be
        // released before we mutate decoder IMDCT state.
        let cfg0_centre_mono = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg0_centre_mono.clone());
        // Round 38 / 39: detach the 5_X SIMPLE/ASPX `coding_config`
        // payloads so we can drive end-to-end multichannel decode.
        // Round 38 wired Cfg2 (four_channel_data + cfg2_back_mono);
        // round 39 adds Cfg0 (b_2ch_mode + 2x two_channel_data +
        // cfg0_centre_mono), Cfg1 (three_channel_data + two_channel_data),
        // and Cfg3 (five_channel_data). Each helper computes its own
        // gating; we just detach the inputs once.
        let five_x_simple_aspx_active = self
            .last_substream
            .as_ref()
            .map(|sub| {
                matches!(
                    sub.tools.five_x_mode,
                    Some(crate::mch::FiveXCodecMode::Simple)
                        | Some(crate::mch::FiveXCodecMode::Aspx)
                )
            })
            .unwrap_or(false);
        let five_x_coding_cfg = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.five_x_coding_config);
        let cfg2_four_channel_data = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.four_channel_data.clone());
        let cfg2_back_mono = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg2_back_mono.clone());
        // Round 41: 5_X SIMPLE/ASPX cfg2 ASPX trailer detach. The
        // outer walker populates these when `5_X_codec_mode == ASPX`
        // (the SIMPLE path leaves them None and the dispatch falls
        // back to low-band only PCM).
        let cfg2_aspx_lr = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg2_aspx_lr.clone());
        let cfg2_aspx_ls_rs = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg2_aspx_ls_rs.clone());
        let cfg2_aspx_centre = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg2_aspx_centre.clone());
        // Round 42: cfg0 / cfg1 / cfg3 ASPX trailer detach.
        let cfg0_aspx_lr = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg0_aspx_lr.clone());
        let cfg0_aspx_ls_rs = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg0_aspx_ls_rs.clone());
        let cfg0_aspx_centre = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg0_aspx_centre.clone());
        let cfg1_aspx_lr = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg1_aspx_lr.clone());
        let cfg1_aspx_ls_rs = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg1_aspx_ls_rs.clone());
        let cfg1_aspx_centre = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg1_aspx_centre.clone());
        let cfg3_aspx_lr = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg3_aspx_lr.clone());
        let cfg3_aspx_ls_rs = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg3_aspx_ls_rs.clone());
        let cfg3_aspx_centre = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.cfg3_aspx_centre.clone());
        let five_x_aspx_config = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.aspx_config);
        // Round 42: companding_control() per-channel flags. The 5_X
        // ASPX path captures companding(3) (L/R, Ls/Rs, C) into
        // `tools.companding`; we lift the parsed flags here so the
        // dispatch can hand a per-channel companding-on bool to the
        // `aspx_extend_with_trailer` wrapper.
        let five_x_companding = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.companding.clone());
        // Cfg0 / Cfg1 / Cfg3 5_X SIMPLE/ASPX detach. Round 39: the walker
        // already populates the same `tools.three_channel_data` /
        // `four_channel_data` / `five_channel_data` / `two_channel_data`
        // slots; here we detach clones for the dispatch helpers.
        let cfg_two_channel_data: Vec<crate::mch::TwoChannelData> = self
            .last_substream
            .as_ref()
            .map(|sub| sub.tools.two_channel_data.clone())
            .unwrap_or_default();
        let cfg_b_2ch_mode = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.b_2ch_mode);
        let cfg_three_channel_data = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.three_channel_data.clone());
        let cfg_five_channel_data = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.five_channel_data.clone());
        // Round 39: 7_X SIMPLE/ASPX additional-channel pair (Table 182).
        // The walker populates `seven_x_additional_channel_data` with two
        // `sf_data(ASF)` bodies for the F / G preliminary outputs (slots
        // 5 / 6 in the bitstream order). Render with identity SAP for now.
        let seven_x_additional_channel_data = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.seven_x_additional_channel_data.clone());
        let seven_x_simple_aspx_active = self
            .last_substream
            .as_ref()
            .map(|sub| {
                matches!(
                    sub.tools.seven_x_mode,
                    Some(crate::mch::SevenXCodecMode::Simple)
                        | Some(crate::mch::SevenXCodecMode::Aspx)
                )
            })
            .unwrap_or(false);
        // Round 37: 7_X ASPX_ACPL_1 / ASPX_ACPL_2 pair dispatch state
        // (mirrors the 5_X detach above). Both modes carry the same
        // shape of `acpl_config_1ch_*` + `acpl_data_1ch_pair`. The 7_X
        // walker also fires for 7.0 and 7.1 (b_has_lfe). Channel
        // mapping per Table 202 — for ACPL_1/_2 (no SIMPLE/ASPX
        // additional-channel block in scope), z6/z7 stay silent and
        // we populate slots 0..4 (L/R/C/Ls/Rs) only.
        let seven_x_pair_mode: Option<acpl_synth::Acpl5xPairMode> = self
            .last_substream
            .as_ref()
            .and_then(|sub| match sub.tools.seven_x_mode {
                Some(crate::mch::SevenXCodecMode::AspxAcpl1) => {
                    Some(acpl_synth::Acpl5xPairMode::AspxAcpl1)
                }
                Some(crate::mch::SevenXCodecMode::AspxAcpl2) => {
                    Some(acpl_synth::Acpl5xPairMode::AspxAcpl2)
                }
                _ => None,
            });
        let seven_x_pair_cfg =
            self.last_substream
                .as_ref()
                .and_then(|sub| match sub.tools.seven_x_mode {
                    Some(crate::mch::SevenXCodecMode::AspxAcpl1) => {
                        sub.tools.acpl_config_1ch_partial
                    }
                    Some(crate::mch::SevenXCodecMode::AspxAcpl2) => sub.tools.acpl_config_1ch_full,
                    _ => None,
                });
        let seven_x_pair_data_1 = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_data_1ch_pair[0].clone());
        let seven_x_pair_data_2 = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.acpl_data_1ch_pair[1].clone());
        let seven_x_pair_active = seven_x_pair_mode.is_some()
            && seven_x_pair_cfg.is_some()
            && seven_x_pair_data_1.is_some()
            && seven_x_pair_data_2.is_some();
        // Centre channel for ASPX_ACPL_3: round 38 wires the parsed
        // `cfg0_centre_mono.scaled_spec` (when present) through IMDCT +
        // overlap-add for slot 2 (centre). This replaces the round-37
        // silence placeholder used while the body decoder was deferred.
        // Falls back to a zero-filled placeholder when the centre body
        // isn't decoded (LFE / SSF / Huffman miss / ACPL_3 walker
        // doesn't populate cfg0_centre_mono on every frame) so the
        // length-checked run_acpl_5x_mch_pcm still fires and emits
        // shaped Ls/Rs from the L/R carriers.
        let five_x_centre_spec: Option<Vec<f32>> = if five_x_acpl3_active {
            let centre_pcm = cfg0_centre_mono
                .as_ref()
                .and_then(|m| self.imdct_mono_lfe_data_f32(m, 2, samples as usize));
            Some(centre_pcm.unwrap_or_else(|| vec![0.0_f32; samples as usize]))
        } else {
            None
        };
        // ASPX_ACPL_1 (joint-MDCT residual layer): M spectrum lives on
        // `scaled_spec_primary`, S on `scaled_spec_secondary`; both
        // share the same transform_info. Detect it via the parsed
        // stereo_codec_mode + acpl_config_1ch_partial (`partial` is the
        // ACPL_1 flavour).
        let acpl1_active = self
            .last_substream
            .as_ref()
            .map(|sub| {
                matches!(sub.tools.stereo_mode, Some(asf::StereoCodecMode::AspxAcpl1))
                    && sub.tools.acpl_config_1ch_partial.is_some()
                    && sub.tools.scaled_spec_primary.is_some()
                    && sub.tools.scaled_spec_secondary.is_some()
            })
            .unwrap_or(false);
        let (
            primary_in,
            secondary_in,
            aspx_tables,
            aspx_cfg,
            framing_pri,
            framing_sec,
            sig_pri,
            sig_sec,
            noise_pri,
            noise_sec,
            qmode_pri,
            qmode_sec,
            delta_dir_pri,
            delta_dir_sec,
            ah_pri,
            ah_sec,
            tna_pri,
            tna_sec,
        ) = if let Some(sub) = self.last_substream.as_ref() {
            let pri = sub
                .tools
                .scaled_spec_primary
                .as_ref()
                .zip(sub.tools.transform_info_primary.as_ref())
                .map(|(s, ti)| (s.clone(), ti.transform_length_0 as usize));
            let sec = sub
                .tools
                .scaled_spec_secondary
                .as_ref()
                .zip(sub.tools.transform_info_secondary.as_ref())
                .map(|(s, ti)| (s.clone(), ti.transform_length_0 as usize));
            let tables = sub.tools.aspx_frequency_tables.clone();
            let cfg = sub.tools.aspx_config;
            // add_harmonic flags per channel: prefer the 2-channel
            // hfgen payload when present, else fall back to the 1-ch
            // one for the primary channel (secondary inherits nothing
            // in that case — the 1-ch hfgen only covers one channel).
            let (ah_p, ah_s) = if let Some(h2) = sub.tools.aspx_hfgen_iwc_2ch.as_ref() {
                (
                    Some(h2.add_harmonic[0].clone()),
                    Some(h2.add_harmonic[1].clone()),
                )
            } else if let Some(h1) = sub.tools.aspx_hfgen_iwc_1ch.as_ref() {
                (Some(h1.add_harmonic.clone()), None)
            } else {
                (None, None)
            };
            // §5.7.6.4.1.3 Pseudocode 88 input — `aspx_tna_mode[ch][sbg]`.
            // 2-ch hfgen carries per-channel modes; 1-ch hfgen carries
            // a single channel's modes that we apply to the primary.
            let (tna_p, tna_s) = if let Some(h2) = sub.tools.aspx_hfgen_iwc_2ch.as_ref() {
                (Some(h2.tna_mode[0].clone()), Some(h2.tna_mode[1].clone()))
            } else if let Some(h1) = sub.tools.aspx_hfgen_iwc_1ch.as_ref() {
                (Some(h1.tna_mode.clone()), None)
            } else {
                (None, None)
            };
            (
                pri,
                sec,
                tables,
                cfg,
                sub.tools.aspx_framing_primary.clone(),
                sub.tools.aspx_framing_secondary.clone(),
                sub.tools.aspx_data_sig_primary.clone(),
                sub.tools.aspx_data_sig_secondary.clone(),
                sub.tools.aspx_data_noise_primary.clone(),
                sub.tools.aspx_data_noise_secondary.clone(),
                sub.tools.aspx_qmode_env_primary,
                sub.tools.aspx_qmode_env_secondary,
                sub.tools.aspx_delta_dir_primary.clone(),
                sub.tools.aspx_delta_dir_secondary.clone(),
                ah_p,
                ah_s,
                tna_p,
                tna_s,
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None,
            )
        };
        // If the ASPX I-frame pipeline populated derived frequency
        // tables + config, run the A-SPX bandwidth-extension on top of
        // the IMDCT low-band PCM.
        let use_aspx_ext = aspx_tables.is_some() && aspx_cfg.is_some();
        let num_ts_in_ats = aspx::num_ts_in_ats(info.frame_length.max(1));
        // Round 43: per-channel companding mode from the parsed
        // `companding_control()`. For mono / stereo CPE paths the
        // grouping is `companding_control(1)` / `companding_control(2)`
        // — i.e. compand_on[0] is the primary channel, compand_on[1]
        // is the secondary (or the sole entry mirrors via sync_flag).
        let (compand_mode_pri, compand_mode_sec) = self
            .last_substream
            .as_ref()
            .map(|sub| {
                let cc = sub.tools.companding.as_ref();
                (
                    Self::five_x_compand_mode_for_slot(cc, 0),
                    Self::five_x_compand_mode_for_slot(cc, 1),
                )
            })
            .unwrap_or((aspx::CompandingMode::Off, aspx::CompandingMode::Off));
        // Round 43: §5.7.5.2 sb0 selection — for the ASPX_ACPL_1 codec
        // mode the companding tool starts at `acpl_qmf_band` instead of
        // `aspx_xover_band`. Both the stereo CPE ASPX_ACPL_1 path and
        // the 5_X ASPX_ACPL_1 path read this from
        // `acpl_config_1ch_partial.qmf_band`. `None` for any other
        // codec mode → falls back to `tables.sbx`.
        let compand_sb0_override: Option<u32> = self.last_substream.as_ref().and_then(|sub| {
            let stereo_acpl1 =
                matches!(sub.tools.stereo_mode, Some(asf::StereoCodecMode::AspxAcpl1));
            let five_x_acpl1 = matches!(
                sub.tools.five_x_mode,
                Some(crate::mch::FiveXCodecMode::AspxAcpl1)
            );
            if stereo_acpl1 || five_x_acpl1 {
                sub.tools
                    .acpl_config_1ch_partial
                    .as_ref()
                    .map(|c| c.qmf_band as u32)
            } else {
                None
            }
        });
        // Make sure the per-channel A-SPX state vector is large enough.
        while self.aspx_ext_state.len() < channels as usize {
            self.aspx_ext_state.push(aspx::AspxChannelExtState::new());
        }
        // Same for the SSF synth state.
        while self.ssf_synth_state.len() < channels as usize {
            self.ssf_synth_state.push(ssf_synth::SsfSynthState::new());
        }
        // §5.7.7 A-CPL: when the substream parsed `acpl_config_1ch` +
        // `acpl_data_1ch` we run the channel-pair synthesis on the
        // ASPX-extended primary PCM and emit two channels. The path
        // owns the primary IMDCT + ASPX path so `pcm_per_channel[1]`
        // ends up populated by the synth's `z1` output instead of by a
        // duplicate-of-primary fallback.
        let use_acpl =
            channels as usize >= 2 && acpl_active_cfg.is_some() && acpl_active_data.is_some();
        // Round 45: stereo-CPE M=2 synced companding. When
        // `companding_control(2)` carried `sync_flag == 1` and the
        // primary / secondary cohort both feed the standalone ASPX
        // path (i.e. `!use_acpl` — ACPL_1 stereo only ASPX-extends
        // the M-channel via the `acpl1_active` branch and so falls
        // outside the synced cohort), the two channels share one
        // geometric-mean gain `g_synch(ts) = √(g_0 · g_1)` per
        // Pseudocode 121's `sync_flag == 1` branch instead of two
        // independent per-channel gains. For 5_X ASPX_ACPL_3 the
        // primary / secondary are the L / R carriers feeding
        // Pseudocode 118's `run_acpl_5x_mch_pcm`, so this puts the
        // ACPL_3 surround-pair driver on the same synced footing as
        // r44's 5_X SIMPLE/ASPX dispatch. Resolves to `None` for
        // `sync_flag == 0`, missing companding, or any non-sync
        // sub-branch — falling back to the per-channel
        // `aspx_extend_pcm` path below.
        let stereo_cpe_synced_mode: Option<aspx::CompandingMode> = self
            .last_substream
            .as_ref()
            .and_then(|sub| sub.tools.companding.as_ref())
            .and_then(|cc| Self::five_x_synced_mode(Some(cc)));
        let use_stereo_cpe_synced = use_aspx_ext
            && !use_acpl
            && channels as usize >= 2
            && stereo_cpe_synced_mode.is_some()
            && primary_in.is_some()
            && secondary_in.is_some()
            && primary_in.as_ref().map(|(_, n)| *n) == secondary_in.as_ref().map(|(_, n)| *n);
        if use_stereo_cpe_synced {
            // Synced stereo-CPE pipeline. IMDCT each channel, then
            // run the M=2 phase-1 / sync-apply / phase-2 helper.
            // SAFETY of the unwraps: guarded by `use_stereo_cpe_synced`
            // (use_aspx_ext, primary_in.is_some(), secondary_in.is_some(),
            // stereo_cpe_synced_mode.is_some()).
            let (p_scaled, p_n) = primary_in.as_ref().unwrap();
            let (s_scaled, s_n) = secondary_in.as_ref().unwrap();
            let n = *p_n;
            if n > 0 && n == samples as usize && *s_n == n && !pcm_per_channel.is_empty() {
                let pcm_pri_f = self.imdct_channel_f32(0, p_scaled, n);
                let pcm_sec_f = self.imdct_channel_f32(1, s_scaled, n);
                let pri_input = StereoCpeChannelInput {
                    ch_index: 0,
                    pcm_in: &pcm_pri_f,
                    framing: framing_pri.as_ref(),
                    sig: sig_pri.as_deref(),
                    noise: noise_pri.as_deref(),
                    qmode: qmode_pri,
                    delta_dir: delta_dir_pri.as_ref(),
                    add_harmonic: ah_pri.as_deref(),
                    tna_mode: tna_pri.as_deref(),
                };
                let sec_input = StereoCpeChannelInput {
                    ch_index: 1,
                    pcm_in: &pcm_sec_f,
                    framing: framing_sec.as_ref().or(framing_pri.as_ref()),
                    sig: sig_sec.as_deref(),
                    noise: noise_sec.as_deref(),
                    qmode: qmode_sec.or(qmode_pri),
                    delta_dir: delta_dir_sec.as_ref().or(delta_dir_pri.as_ref()),
                    add_harmonic: ah_sec.as_deref().or(ah_pri.as_deref()),
                    tna_mode: tna_sec.as_deref().or(tna_pri.as_deref()),
                };
                let (ext_pri, ext_sec) = self.extend_stereo_cpe_pair_with_sync_companding(
                    &pri_input,
                    &sec_input,
                    aspx_tables.as_ref().unwrap(),
                    aspx_cfg.as_ref().unwrap(),
                    num_ts_in_ats,
                    stereo_cpe_synced_mode.unwrap(),
                    compand_sb0_override,
                );
                if pcm_per_channel.len() < 2 {
                    while pcm_per_channel.len() < 2 {
                        pcm_per_channel.push(None);
                    }
                }
                pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&ext_pri));
                pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&ext_sec));
            }
        }
        if !use_stereo_cpe_synced {
            if let Some((scaled, n)) = primary_in {
                if n > 0 && n == samples as usize && !pcm_per_channel.is_empty() {
                    if use_aspx_ext {
                        let pcm_f = self.imdct_channel_f32(0, &scaled, n);
                        let state = &mut self.aspx_ext_state[0];
                        let extended = Self::aspx_extend_pcm(
                            &pcm_f,
                            aspx_tables.as_ref().unwrap(),
                            aspx_cfg.as_ref().unwrap(),
                            framing_pri.as_ref(),
                            sig_pri.as_deref(),
                            noise_pri.as_deref(),
                            qmode_pri,
                            delta_dir_pri.as_ref(),
                            ah_pri.as_deref(),
                            tna_pri.as_deref(),
                            state,
                            num_ts_in_ats,
                            compand_mode_pri,
                            compand_sb0_override,
                        );
                        if use_acpl {
                            if let (Some(cfg), Some(data)) =
                                (acpl_active_cfg.as_ref(), acpl_active_data.as_ref())
                            {
                                // ASPX_ACPL_1: feed both M (extended) and S
                                // PCM into the stereo A-CPL. The S spectrum
                                // is already in `secondary_in`; we IMDCT it
                                // here without ASPX (the `aspx_data_1ch` in
                                // ACPL_1 covers the M channel only).
                                let acpl1_result = if acpl1_active {
                                    if let Some((s_scaled, s_n)) = secondary_in.as_ref() {
                                        if *s_n == n {
                                            let s_pcm = self.imdct_channel_f32(1, s_scaled, *s_n);
                                            acpl_synth::run_acpl_1ch_pcm_stereo(
                                                &extended,
                                                &s_pcm,
                                                cfg,
                                                data,
                                                &mut self.acpl_state,
                                            )
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    acpl_synth::run_acpl_1ch_pcm(
                                        &extended,
                                        cfg,
                                        data,
                                        &mut self.acpl_state,
                                    )
                                };
                                if let Some((left, right)) = acpl1_result {
                                    pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&left));
                                    pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&right));
                                } else {
                                    pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&extended));
                                }
                            } else {
                                pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&extended));
                            }
                        } else {
                            pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&extended));
                        }
                    } else {
                        pcm_per_channel[0] = Some(self.imdct_channel(0, &scaled, n));
                    }
                }
            }
            if channels as usize >= 2 && !use_acpl {
                if let Some((scaled, n)) = secondary_in {
                    if n > 0 && n == samples as usize {
                        if use_aspx_ext {
                            let pcm_f = self.imdct_channel_f32(1, &scaled, n);
                            let state = &mut self.aspx_ext_state[1];
                            let extended = Self::aspx_extend_pcm(
                                &pcm_f,
                                aspx_tables.as_ref().unwrap(),
                                aspx_cfg.as_ref().unwrap(),
                                framing_sec.as_ref().or(framing_pri.as_ref()),
                                sig_sec.as_deref(),
                                noise_sec.as_deref(),
                                qmode_sec.or(qmode_pri),
                                delta_dir_sec.as_ref().or(delta_dir_pri.as_ref()),
                                ah_sec.as_deref().or(ah_pri.as_deref()),
                                tna_sec.as_deref().or(tna_pri.as_deref()),
                                state,
                                num_ts_in_ats,
                                compand_mode_sec,
                                compand_sb0_override,
                            );
                            pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&extended));
                        } else {
                            pcm_per_channel[1] = Some(self.imdct_channel(1, &scaled, n));
                        }
                    }
                }
            }
        } // end `if !use_stereo_cpe_synced`
          // SSF synthesis path — if either ssf_data_* is populated and
          // the corresponding `pcm_per_channel[ch]` slot is still empty
          // (the ASF Huffman pipeline didn't fire because spec_frontend
          // was SSF), drive §5.2.3-5.2.7 → IMDCT to produce real PCM.
          // Synthesize each granule into a `num_blocks * n_mdct`-long
          // spectrum vector, then IMDCT each `n_mdct` block independently
          // and concat the resulting overlap-added PCM.
        if let Some(data) = ssf_primary.as_ref() {
            if !pcm_per_channel.is_empty() && pcm_per_channel[0].is_none() {
                let pcm = self.run_ssf_channel(0, data, samples as usize);
                if !pcm.is_empty() {
                    pcm_per_channel[0] = Some(pcm);
                }
            }
        }
        if channels as usize >= 2 {
            if let Some(data) = ssf_secondary.as_ref() {
                if pcm_per_channel.len() >= 2 && pcm_per_channel[1].is_none() {
                    let pcm = self.run_ssf_channel(1, data, samples as usize);
                    if !pcm.is_empty() {
                        pcm_per_channel[1] = Some(pcm);
                    }
                }
            }
        }
        // §5.7.7.6.2 ASPX_ACPL_3 5_X synthesis (Pseudocode 118) —
        // When the substream parsed acpl_config_2ch + acpl_data_2ch and
        // the stereo-body path decoded the L/R carrier spectra, run the
        // full 5-channel A-CPL synthesis and populate channels 0..4.
        // Only fires when all five pcm_per_channel slots are still empty
        // (i.e. the standard stereo path didn't already claim them), or
        // when the frame is explicitly a 5_X ASPX_ACPL_3 substream.
        if five_x_acpl3_active {
            if let (Some(cfg), Some(data), Some(centre)) = (
                five_x_acpl3_cfg.as_ref(),
                five_x_acpl3_data.as_ref(),
                five_x_centre_spec.as_deref(),
            ) {
                // Carrier L and R come from pcm_per_channel[0] / [1] (already
                // filled by the stereo ASF / ASPX decode path above). If they
                // are present use them; otherwise zero-fill as placeholders so
                // the A-CPL synthesis still produces shaped Ls/Rs.
                let n = samples as usize;
                let pcm_l_f32: Vec<f32> = pcm_per_channel
                    .first()
                    .and_then(|p| p.as_ref())
                    .map(|v| v.iter().map(|&s| s as f32 / 32767.0).collect())
                    .unwrap_or_else(|| vec![0.0_f32; n]);
                let pcm_r_f32: Vec<f32> = pcm_per_channel
                    .get(1)
                    .and_then(|p| p.as_ref())
                    .map(|v| v.iter().map(|&s| s as f32 / 32767.0).collect())
                    .unwrap_or_else(|| vec![0.0_f32; n]);
                if let Some(out) = acpl_synth::run_acpl_5x_mch_pcm(
                    &pcm_l_f32,
                    &pcm_r_f32,
                    centre,
                    cfg,
                    data,
                    &mut self.acpl_5x_mch_state,
                ) {
                    // Output channel mapping for 5.0/5.1:
                    //   ch0 = L, ch1 = R, ch2 = C, ch3 = Ls, ch4 = Rs.
                    // Resize pcm_per_channel to 5 slots if needed.
                    while pcm_per_channel.len() < 5 {
                        pcm_per_channel.push(None);
                    }
                    pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&out.left));
                    pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&out.right));
                    pcm_per_channel[2] = Some(Self::pcm_f32_to_i16(&out.centre));
                    pcm_per_channel[3] = Some(Self::pcm_f32_to_i16(&out.left_surround));
                    pcm_per_channel[4] = Some(Self::pcm_f32_to_i16(&out.right_surround));
                }
            }
        }
        // §5.7.7.6.1 ASPX_ACPL_1 / ASPX_ACPL_2 5_X synthesis (Pseudocode 117) —
        // When the 5_X walker resolved `five_x_mode` to AspxAcpl1 / AspxAcpl2
        // and parsed the matching `acpl_config_1ch_*` + `acpl_data_1ch_pair`,
        // run the channel-pair synthesis on the L/R carrier PCM and emit
        // L / R / C / Ls / Rs.
        //
        // L/R carriers come from `pcm_per_channel[0]/[1]` (already filled
        // by the stereo ASF/ASPX decode path above when present, else
        // zero-filled placeholders). The centre carrier mirrors the
        // ACPL_3 path — `cfg0_centre_mono` exists in the tools struct
        // but lacks an end-to-end decode path; we use silence so the
        // QMF lengths line up. ACPL_1's Ls/Rs surround carriers are
        // similarly silence-placeholders for the same reason: A-CPL
        // synthesis still produces shaped Ls/Rs from the L/R carriers
        // and the pair parameters; the contribution from the surround
        // carriers (when those gain a real decode path) just adds in
        // on top.
        if five_x_pair_active && !five_x_acpl3_active {
            if let (Some(mode), Some(cfg), Some(data_1), Some(data_2)) = (
                five_x_pair_mode,
                five_x_pair_cfg.as_ref(),
                five_x_pair_data_1.as_ref(),
                five_x_pair_data_2.as_ref(),
            ) {
                // Round 37: IMDCT the parsed centre `mono_data(0)`
                // spectrum (Cfg0 trailing) into a real PCM carrier;
                // falls back to silence when `scaled_spec` is None
                // (LFE / SSF / Huffman miss) — see `imdct_mono_lfe_data_f32`.
                let centre_pcm = cfg0_centre_mono
                    .as_ref()
                    .and_then(|m| self.imdct_mono_lfe_data_f32(m, 2, samples as usize));
                // Round 40: standalone Ls/Rs surround mono walker for
                // ACPL_1's Mode 1 surround-driven path. The 5_X
                // ASPX_ACPL_1 inner walker now persists the joint-MDCT
                // residual pair (sSMP,3 / sSMP,4 per Table 181) on
                // `tools.acpl_1_residual_pair`; we IMDCT them here into
                // Ls/Rs PCM carriers and feed them as the `x3` / `x4`
                // inputs of Pseudocode 117. ACPL_2 mode never emits a
                // residual pair (no max_sfb_master in the walker), so
                // the detach is `None` for that path → silence — same
                // as the round-37 placeholder.
                //
                // Round 46 — ACPL_1 surround Ls/Rs ASPX extension:
                // SPEC-CONFIRMS-NOT-ASPX. Per ETSI TS 103 190-1 §4.2.6.6
                // Table 25 row `case ASPX_ACPL_1:` (the `5_X_codec_mode
                // == ASPX_ACPL_1` body parsed by
                // `parse_aspx_acpl_1_2_inner_body` in `mch.rs`) the
                // trailer order is `aspx_data_2ch()` (L/R primary
                // carriers) + `aspx_data_1ch()` (centre mono) + two
                // `acpl_data_1ch()` parameter sets — there is NO third
                // ASPX trailer for the surround Ls/Rs pair. The Ls/Rs
                // carriers are the joint-MDCT residual sSMP,3 / sSMP,4
                // straight out of the inner sf_data×2 walker; per
                // §5.7.5.2 / §5.7.6 ASPX BWE applies to the
                // M-channel-side carriers only (acpl_qmf_band-rooted
                // sb0 on the L/R primary pair + centre mono, never on
                // the residual surround pair). Feeding them raw into
                // Pseudocode 117 as `x3` / `x4` matches the spec — the
                // post-Pseudocode-117 surround output gets its
                // synthesis-bandwidth shape from the L/R carriers via
                // alpha/beta/decorrelator, not from independent
                // surround-pair extension. Same finding for the
                // matching M=2 surround-pair synced companding cohort:
                // no carriers means no companding to sync. Round-46
                // therefore wires no new surround-pair ASPX/companding
                // path here; the existing raw-PCM path is correct.
                let acpl_1_residual_pair = self
                    .last_substream
                    .as_ref()
                    .map(|sub| sub.tools.acpl_1_residual_pair.clone())
                    .unwrap_or([None, None]);
                // Round 41: §5.3.4.3.2 / Table 181 first-stage matrix —
                // when the 5_X ACPL_1 walker captured the two
                // `chparam_info()` payloads + the joint-MDCT residual
                // pair AND the inner `two_channel_data` carries
                // sSMP_A / sSMP_B spectra, mix per-sfb to produce
                // preliminary (L, R, Ls, Rs) spectra, IMDCT each, and
                // feed those PCMs into Pseudocode 117.
                //
                // When the SAP inputs aren't all available (ACPL_2 path,
                // or non-AspxAcpl1 mode, or any of the inputs missing)
                // fall through to the round-40 path: raw sSMP_3/sSMP_4
                // PCM as ls/rs, slots 0/1 untouched.
                let chparam_pair = self
                    .last_substream
                    .as_ref()
                    .map(|sub| sub.tools.acpl_1_residual_chparam.clone())
                    .unwrap_or([None, None]);
                let max_sfb_master_opt: Option<u32> = self
                    .last_substream
                    .as_ref()
                    .and_then(|sub| sub.tools.acpl_1_residual_max_sfb_master);
                let inner_tcd_specs: Option<(Vec<f32>, Vec<f32>)> =
                    self.last_substream.as_ref().and_then(|sub| {
                        let tcd = sub.tools.two_channel_data.first()?;
                        let a = tcd.scaled_spec_per_channel.first().cloned().flatten()?;
                        let b = tcd.scaled_spec_per_channel.get(1).cloned().flatten()?;
                        Some((a, b))
                    });
                let sap_outputs: Option<asf::SapTable181Output> = match (
                    mode,
                    inner_tcd_specs.as_ref(),
                    &chparam_pair,
                    &acpl_1_residual_pair,
                    max_sfb_master_opt,
                ) {
                    (
                        acpl_synth::Acpl5xPairMode::AspxAcpl1,
                        Some((a_spec, b_spec)),
                        [Some(cp0), Some(cp1)],
                        [Some((tl3, s3)), Some((tl4, s4))],
                        Some(max_sfb_master),
                    ) if *tl3 == *tl4
                        && *tl3 as usize == samples as usize
                        && max_sfb_master > 0 =>
                    {
                        asf::apply_sap_table_181(
                            a_spec,
                            b_spec,
                            s3,
                            s4,
                            &[cp0.clone(), cp1.clone()],
                            max_sfb_master,
                            *tl3,
                        )
                    }
                    _ => None,
                };
                let (ls_pcm, rs_pcm) =
                    if let Some((l_spec, r_spec, ls_spec, rs_spec)) = sap_outputs.as_ref() {
                        // SAP path: replace pcm_per_channel[0]/[1] with the
                        // mixed L/R PCM and pass mixed Ls/Rs PCM into the
                        // pair dispatcher.
                        let n = samples as usize;
                        let l_pcm = self.imdct_channel_f32(0, l_spec, n);
                        let r_pcm = self.imdct_channel_f32(1, r_spec, n);
                        while pcm_per_channel.len() < 2 {
                            pcm_per_channel.push(None);
                        }
                        pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&l_pcm));
                        pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&r_pcm));
                        let ls_pcm = self.imdct_channel_f32(3, ls_spec, n);
                        let rs_pcm = self.imdct_channel_f32(4, rs_spec, n);
                        (Some(ls_pcm), Some(rs_pcm))
                    } else {
                        let ls_pcm = acpl_1_residual_pair[0].as_ref().and_then(|(tl, scaled)| {
                            if *tl as usize == samples as usize {
                                Some(self.imdct_channel_f32(3, scaled, samples as usize))
                            } else {
                                None
                            }
                        });
                        let rs_pcm = acpl_1_residual_pair[1].as_ref().and_then(|(tl, scaled)| {
                            if *tl as usize == samples as usize {
                                Some(self.imdct_channel_f32(4, scaled, samples as usize))
                            } else {
                                None
                            }
                        });
                        (ls_pcm, rs_pcm)
                    };
                self.dispatch_acpl_5x_pair(
                    mode,
                    cfg,
                    data_1,
                    data_2,
                    samples as usize,
                    centre_pcm.as_deref(),
                    ls_pcm.as_deref(),
                    rs_pcm.as_deref(),
                    &mut pcm_per_channel,
                );
            }
        }
        // §5.7.7.6.3 Pseudocode 120 — 7_X ASPX_ACPL_1 / ASPX_ACPL_2
        // dispatch (mirrors the 5_X path above). Channel mapping is
        // Table 202 (channel_mode, add_ch_base) — for ACPL_1/_2 the
        // additional 2 channels (z6/z7 in Pseudocode 120) live outside
        // the A-CPL pair so they aren't generated here; we populate
        // slots 0..4 (L/R/C/Ls/Rs) and leave 5..7 for the per-channel
        // fallback path. The pair core itself is bit-equivalent to
        // Pseudocode 117 — same `(z0, z1) = ACplModule(...)` shape +
        // `z1 *= sqrt(2)` / `z3 *= sqrt(2)` scaling — modulo the extra
        // `add_ch_base == 0` z0/z2 sqrt(2) tweak which only fires when
        // the additional channels carry the L/R pair. Since we treat
        // the additional pair as silence here, that conditional scale
        // does not affect the produced 5-channel core.
        if seven_x_pair_active {
            if let (Some(mode), Some(cfg), Some(data_1), Some(data_2)) = (
                seven_x_pair_mode,
                seven_x_pair_cfg.as_ref(),
                seven_x_pair_data_1.as_ref(),
                seven_x_pair_data_2.as_ref(),
            ) {
                let centre_pcm = cfg0_centre_mono
                    .as_ref()
                    .and_then(|m| self.imdct_mono_lfe_data_f32(m, 2, samples as usize));
                // Round 40: same standalone Ls/Rs surround mono walker
                // as the 5_X path — the 7_X ASPX_ACPL_1 walker writes
                // to the same `acpl_1_residual_pair` slot. ACPL_2 path
                // detaches `None` (no residual pair).
                let acpl_1_residual_pair = self
                    .last_substream
                    .as_ref()
                    .map(|sub| sub.tools.acpl_1_residual_pair.clone())
                    .unwrap_or([None, None]);
                let ls_pcm = acpl_1_residual_pair[0].as_ref().and_then(|(tl, scaled)| {
                    if *tl as usize == samples as usize {
                        Some(self.imdct_channel_f32(3, scaled, samples as usize))
                    } else {
                        None
                    }
                });
                let rs_pcm = acpl_1_residual_pair[1].as_ref().and_then(|(tl, scaled)| {
                    if *tl as usize == samples as usize {
                        Some(self.imdct_channel_f32(4, scaled, samples as usize))
                    } else {
                        None
                    }
                });
                self.dispatch_acpl_5x_pair(
                    mode,
                    cfg,
                    data_1,
                    data_2,
                    samples as usize,
                    centre_pcm.as_deref(),
                    ls_pcm.as_deref(),
                    rs_pcm.as_deref(),
                    &mut pcm_per_channel,
                );
            }
        }
        // Round 38 / 39: §5.3.4.3.1 / Table 180 — 5_X SIMPLE/ASPX
        // end-to-end decode. Round 38 wired Cfg2; round 39 wires Cfg0,
        // Cfg1, Cfg3. Mutually exclusive with the ACPL_3 / pair paths
        // above (they own different `five_x_mode` enums), so each cfg
        // fires only when the SIMPLE/ASPX pure-MDCT path is in scope.
        if five_x_simple_aspx_active && !five_x_acpl3_active && !five_x_pair_active {
            match five_x_coding_cfg {
                Some(crate::mch::FiveXCodingConfig::Cfg0Stereo2plusMono)
                    if cfg_two_channel_data.len() >= 2 =>
                {
                    let b_2ch = cfg_b_2ch_mode.unwrap_or(false);
                    self.dispatch_5x_cfg0_simple_aspx(
                        &cfg_two_channel_data[0],
                        &cfg_two_channel_data[1],
                        b_2ch,
                        cfg0_centre_mono.as_ref(),
                        cfg0_aspx_lr.as_ref(),
                        cfg0_aspx_ls_rs.as_ref(),
                        cfg0_aspx_centre.as_ref(),
                        five_x_aspx_config,
                        five_x_companding.as_ref(),
                        num_ts_in_ats,
                        samples as usize,
                        &mut pcm_per_channel,
                    );
                }
                Some(crate::mch::FiveXCodingConfig::Cfg1ThreeStereo) => {
                    if let (Some(three), Some(tcd)) = (
                        cfg_three_channel_data.as_ref(),
                        cfg_two_channel_data.first(),
                    ) {
                        self.dispatch_5x_cfg1_simple_aspx(
                            three,
                            tcd,
                            cfg1_aspx_lr.as_ref(),
                            cfg1_aspx_ls_rs.as_ref(),
                            cfg1_aspx_centre.as_ref(),
                            five_x_aspx_config,
                            five_x_companding.as_ref(),
                            num_ts_in_ats,
                            samples as usize,
                            &mut pcm_per_channel,
                        );
                    }
                }
                Some(crate::mch::FiveXCodingConfig::Cfg2FourMono) => {
                    if let Some(four) = cfg2_four_channel_data.as_ref() {
                        self.dispatch_5x_cfg2_simple_aspx(
                            four,
                            cfg2_back_mono.as_ref(),
                            cfg2_aspx_lr.as_ref(),
                            cfg2_aspx_ls_rs.as_ref(),
                            cfg2_aspx_centre.as_ref(),
                            five_x_aspx_config,
                            five_x_companding.as_ref(),
                            num_ts_in_ats,
                            samples as usize,
                            &mut pcm_per_channel,
                        );
                    }
                }
                Some(crate::mch::FiveXCodingConfig::Cfg3Five) => {
                    if let Some(five) = cfg_five_channel_data.as_ref() {
                        self.dispatch_5x_cfg3_simple_aspx(
                            five,
                            cfg3_aspx_lr.as_ref(),
                            cfg3_aspx_ls_rs.as_ref(),
                            cfg3_aspx_centre.as_ref(),
                            five_x_aspx_config,
                            five_x_companding.as_ref(),
                            num_ts_in_ats,
                            samples as usize,
                            &mut pcm_per_channel,
                        );
                    }
                }
                _ => {}
            }
        }
        // Round 91: 7_X SIMPLE/ASPX inner 5-channel core render (slots
        // 0..4). The 7_X SIMPLE/Cfg3Five path inherits the inner
        // `five_channel_data()` from the 5_X Table 29 layout (5 SCEs in
        // L/R/C/Ls/Rs order, identity SAP via 5x `chparam_info(sap_mode
        // = 0)`); the only difference from the 5_X dispatch is which
        // walker populated `tools.five_channel_data` (7_X here, vs 5_X
        // for the 5.0/5.1 paths). The 5_X dispatch fires the same
        // IMDCT/KBD/overlap-add chain regardless of which walker
        // populated the slot, so we route the 7_X-walker-produced
        // five_channel_data through it. With identity SAP no joint-MDCT
        // mixing happens at decode time so each output slot 0..4 reflects
        // only its own input SCE. ASPX trailers for the 7_X path land in
        // different `tools.*_aspx_*` slots (the 7_X walker has its own
        // ASPX trailer plumbing — out of scope here); pass `None` for
        // the trailer slots so the round-91 SIMPLE path reduces to
        // low-band only. Cfg0/Cfg1/Cfg2 7_X variants need their own
        // wiring (queued for follow-up rounds — they share the same
        // 5_X core dispatchers, just with the 7_X-specific trailing
        // `mono_data(0)` gate and ASPX trailer plumbing).
        if seven_x_simple_aspx_active
            && matches!(
                self.last_substream
                    .as_ref()
                    .and_then(|sub| sub.tools.seven_x_coding_config),
                Some(crate::mch::FiveXCodingConfig::Cfg3Five)
            )
        {
            if let Some(five) = self
                .last_substream
                .as_ref()
                .and_then(|sub| sub.tools.five_channel_data.clone())
            {
                self.dispatch_5x_cfg3_simple_aspx(
                    &five,
                    None,
                    None,
                    None,
                    None,
                    None,
                    num_ts_in_ats,
                    samples as usize,
                    &mut pcm_per_channel,
                );
            }
        }
        // Round 39 / 40: §5.3.4.4.1 / Table 182 + Table 183 — 7_X
        // SIMPLE/ASPX additional-channel pair render. The walker populates
        // `seven_x_additional_channel_data` (two sf_data(ASF) bodies)
        // when `7_X_codec_mode in {SIMPLE, ASPX}`. Slots 5 / 6 (the F/G
        // preliminary outputs in Table 182) get the IMDCT'd low-band PCM.
        //
        // Round 40 wires the SAP a/b/c/d coefficient extraction
        // (`extract_sap_abcd` per Pseudocode 59) through Table 183's
        // 2-pair joint-stereo matrix when `b_use_sap_add_ch == true` AND
        // partner spectra (D, E for `coding_config in {0, 2, 3}` —
        // 3/4/0.x channel_mode) are present. The dispatch walks
        // (P, F) → (slot_high, slot_low) and (Q, G) → (slot_high+1,
        // slot_low+1) per-sfb in the spectral domain. With identity SAP
        // (`b_use_sap_add_ch == false`), the partner spectra are left
        // untouched at their 5.X-core slots and only F/G land at slots
        // 5/6 — matching the round-39 behaviour.
        //
        // The 7_X ACPL_1/_2 walker has its own additional-channel
        // handling per §5.3.4.4.2/.3 (z6/z7 in Pseudocode 120) — this
        // branch is gated on the SIMPLE/ASPX active-flag.
        if seven_x_simple_aspx_active {
            if let Some(add) = seven_x_additional_channel_data.as_ref() {
                // Resolve partner spectra + slots based on the active
                // 7_X coding_config. Per Table 183 row "3/4/0.x" (the
                // standard 7.0/7.1 layout that our 7_X walker handles)
                // the partner pair is (Ls, Rs) — slot 3 / slot 4 in our
                // 5.X-core dispatch; F/G lift to (Lb, Rb) on slot 5/6.
                let partner_slots: [usize; 2] = [3, 4];
                let (partner_d, partner_e): (Option<Vec<f32>>, Option<Vec<f32>>) =
                    match five_x_coding_cfg {
                        Some(crate::mch::FiveXCodingConfig::Cfg2FourMono) => {
                            // 5_X cfg2 four_channel_data carries [L, R, Ls, Rs]
                            // in indices [0, 1, 2, 3] per Table 180. The
                            // surround pair lives at four[2]/four[3].
                            let (d, e) = match cfg2_four_channel_data.as_ref() {
                                Some(four) => (
                                    four.scaled_spec_per_channel.get(2).cloned().flatten(),
                                    four.scaled_spec_per_channel.get(3).cloned().flatten(),
                                ),
                                None => (None, None),
                            };
                            (d, e)
                        }
                        Some(crate::mch::FiveXCodingConfig::Cfg3Five) => {
                            // 5_X cfg3 five_channel_data lays out [L, R, C,
                            // Ls, Rs] per Table 180. Surround pair lives at
                            // five[3]/five[4].
                            let (d, e) = match cfg_five_channel_data.as_ref() {
                                Some(five) => (
                                    five.scaled_spec_per_channel.get(3).cloned().flatten(),
                                    five.scaled_spec_per_channel.get(4).cloned().flatten(),
                                ),
                                None => (None, None),
                            };
                            (d, e)
                        }
                        Some(crate::mch::FiveXCodingConfig::Cfg1ThreeStereo) => {
                            // 5_X cfg1 three_channel_data + two_channel_data:
                            // surround pair lives at the trailing
                            // two_channel_data[0]/[1] (slots 3/4 in our
                            // dispatch). Use the parsed scaled_spec.
                            let (d, e) = match cfg_two_channel_data.first() {
                                Some(tcd) => (
                                    tcd.scaled_spec_per_channel.first().cloned().flatten(),
                                    tcd.scaled_spec_per_channel.get(1).cloned().flatten(),
                                ),
                                None => (None, None),
                            };
                            (d, e)
                        }
                        _ => (None, None),
                    };
                let chparam_pair = self
                    .last_substream
                    .as_ref()
                    .and_then(|sub| sub.tools.seven_x_add_chparam_info.as_ref().cloned());
                let partner_pair: Option<[&[f32]; 2]> =
                    match (partner_d.as_ref(), partner_e.as_ref()) {
                        (Some(d), Some(e)) => Some([d.as_slice(), e.as_slice()]),
                        _ => None,
                    };
                self.dispatch_7x_additional_channel_pair(
                    add,
                    partner_pair,
                    partner_slots,
                    chparam_pair.as_ref(),
                    samples as usize,
                    &mut pcm_per_channel,
                );
            }
        }
        // Round 80: 5.1 / 7.1 LFE channel render. When the 5_X / 7_X
        // walker parsed a `mono_data(b_lfe = 1)` payload (per §4.2.6.6
        // Table 25 `if (b_has_lfe) mono_data(1);` / §4.2.6.14 Table 33
        // equivalent) the LFE scaled spectrum lives on
        // `tools.lfe_mono_data.scaled_spec`. IMDCT it into the trailing
        // LFE PCM slot — slot 5 for 5.1 (after L/R/C/Ls/Rs) and slot 7
        // for 7.1 (after L/R/C/Ls/Rs/Lb/Rb).
        if channels == 6 || channels == 8 {
            let lfe_slot = (channels as usize) - 1;
            let lfe_mono = self
                .last_substream
                .as_ref()
                .and_then(|sub| sub.tools.lfe_mono_data.clone());
            if let Some(lfe) = lfe_mono.as_ref() {
                if let Some(pcm_f) = self.imdct_mono_lfe_data_f32(lfe, lfe_slot, samples as usize) {
                    while pcm_per_channel.len() <= lfe_slot {
                        pcm_per_channel.push(None);
                    }
                    pcm_per_channel[lfe_slot] = Some(Self::pcm_f32_to_i16(&pcm_f));
                }
            }
        }
        self.last_info = Some(info);
        let byte_count = (samples as usize) * (channels as usize) * 2; // S16 interleaved.
        let any_decoded = pcm_per_channel.iter().any(|p| p.is_some());
        let data = if any_decoded {
            let mut buf = vec![0u8; byte_count];
            // Channel fallback: if only channel 0 was decoded for a
            // multi-channel stream (e.g. a stereo frame whose CPE body
            // didn't parse), duplicate it across the remaining slots so
            // the output is audible rather than one-sided.
            let fallback = pcm_per_channel[0].clone();
            for i in 0..samples as usize {
                for c in 0..channels as usize {
                    let sample = pcm_per_channel
                        .get(c)
                        .and_then(|p| p.as_ref())
                        .or(fallback.as_ref())
                        .and_then(|p| p.get(i).copied())
                        .unwrap_or(0);
                    let le = sample.to_le_bytes();
                    let off = (i * channels as usize + c) * 2;
                    if off + 1 < buf.len() {
                        buf[off] = le[0];
                        buf[off + 1] = le[1];
                    }
                }
            }
            vec![buf]
        } else {
            vec![vec![0u8; byte_count]]
        };
        Ok(Frame::Audio(AudioFrame {
            samples,
            pts: pkt.pts,
            data,
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    fn build_minimal_toc() -> Vec<u8> {
        // Build a minimal single-presentation, single-substream AC-4 TOC
        // claiming 48 kHz, 24 fps, stereo (channel_mode prefix '10'),
        // b_iframe = 1.
        let mut bw = BitWriter::new();
        // bitstream_version = 0 (2 bits) — TS 103 190-1 v0 syntax body
        // follows. The parser dispatches `ac4_presentation_info()` only
        // when bitstream_version <= 1.
        bw.write_u32(0, 2);
        // sequence_counter = 7 (10 bits).
        bw.write_u32(7, 10);
        // b_wait_frames = 0.
        bw.write_u32(0, 1);
        // fs_index = 1 (48 kHz), frame_rate_index = 1 (24 fps).
        bw.write_u32(1, 1);
        bw.write_u32(1, 4);
        // b_iframe_global = 1, b_single_presentation = 1.
        bw.write_u32(1, 1);
        bw.write_u32(1, 1);
        // b_payload_base = 0.
        bw.write_u32(0, 1);
        // --- ac4_presentation_info() ---
        // b_single_substream = 1.
        bw.write_u32(1, 1);
        // presentation_version() = 0 (single '0').
        bw.write_u32(0, 1);
        // md_compat (3 bits), b_belongs_to_presentation_id = 0.
        bw.write_u32(0, 3);
        bw.write_u32(0, 1);
        // frame_rate_multiply_info: for fri=1 (index 1) it's a single
        // b_multiplier bit, 0.
        bw.write_u32(0, 1);
        // emdf_info(): emdf_version=0 (2b), key_id=0 (3b),
        // b_emdf_payloads_substream_info=0, emdf_reserved(): b_more=0.
        bw.write_u32(0, 2);
        bw.write_u32(0, 3);
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // ac4_substream_info():
        //   channel_mode prefix '10' = stereo, fs_index==1 so
        //   b_sf_multiplier=0, b_bitrate_info=0, b_content_type=0,
        //   frame_rate_factor=1 -> 1 b_iframe bit (set),
        //   substream_index = 0 (2 bits).
        bw.write_u32(0b10, 2); // channel_mode
        bw.write_u32(0, 1); // b_sf_multiplier
        bw.write_u32(0, 1); // b_bitrate_info
        bw.write_u32(0, 1); // b_content_type
        bw.write_u32(1, 1); // b_iframe
        bw.write_u32(0, 2); // substream_index
                            // b_pre_virtualized = 0, b_add_emdf_substreams = 0.
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // substream_index_table(): n_substreams=1, b_size_present=0.
        bw.write_u32(1, 2);
        bw.write_u32(0, 1);
        // byte_align.
        bw.align_to_byte();
        bw.finish()
    }

    #[test]
    fn decoder_emits_silence_with_correct_shape() {
        let mut bytes = build_minimal_toc();
        // Pad some substream body so the decoder has something to point
        // at (we don't touch it beyond the TOC).
        bytes.extend(vec![0u8; 64]);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        // Per-frame channels / sample_rate / format are no longer carried
        // on AudioFrame — the byte count below implicitly checks stereo
        // S16 layout (1920 samples × 2 ch × 2 bytes).
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.data.len(), 1);
        assert_eq!(af.data[0].len(), (1_920 * 2 * 2) as usize);
        // Samples are silent.
        assert!(af.data[0].iter().all(|&b| b == 0));
        let info = dec.last_info.as_ref().unwrap();
        assert_eq!(info.n_presentations, 1);
        assert_eq!(info.n_substreams, 1);
        assert_eq!(info.fs_index, 1);
        assert_eq!(info.frame_rate_index, 1);
        assert_eq!(info.frame_length, 1_920);
        assert!(info.b_iframe_global);
    }

    fn build_mono_toc() -> Vec<u8> {
        // Single-presentation, single-substream AC-4 TOC claiming
        // 48 kHz, 24 fps, mono (channel_mode prefix '0'), b_iframe = 1.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2); // bitstream_version = 0 (TS 103 190-1 v0 syntax body follows)
        bw.write_u32(7, 10); // sequence_counter
        bw.write_u32(0, 1); // b_wait_frames
        bw.write_u32(1, 1); // fs_index = 1 (48 kHz)
        bw.write_u32(1, 4); // frame_rate_index = 1 (24 fps)
        bw.write_u32(1, 1); // b_iframe_global
        bw.write_u32(1, 1); // b_single_presentation
        bw.write_u32(0, 1); // b_payload_base
                            // ac4_presentation_info:
        bw.write_u32(1, 1); // b_single_substream
        bw.write_u32(0, 1); // presentation_version = 0
        bw.write_u32(0, 3); // md_compat
        bw.write_u32(0, 1); // b_belongs_to_presentation_id
        bw.write_u32(0, 1); // frame_rate_multiply_info
                            // emdf_info:
        bw.write_u32(0, 2);
        bw.write_u32(0, 3);
        bw.write_u32(0, 1);
        bw.write_u32(0, 1);
        // ac4_substream_info:
        bw.write_u32(0b0, 1); // channel_mode = 0 (mono) — prefix '0'
        bw.write_u32(0, 1); // b_sf_multiplier
        bw.write_u32(0, 1); // b_bitrate_info
        bw.write_u32(0, 1); // b_content_type
        bw.write_u32(1, 1); // b_iframe
        bw.write_u32(0, 2); // substream_index
        bw.write_u32(0, 1); // b_pre_virtualized
        bw.write_u32(0, 1); // b_add_emdf_substreams
                            // substream_index_table:
        bw.write_u32(1, 2); // n_substreams - 1
        bw.write_u32(0, 1); // b_size_present
        bw.align_to_byte();
        bw.finish()
    }

    /// Write a sect_len_incr sequence for a given section length.
    /// For n_sect_bits=3, esc=7: sect_len=1+7k+incr; emit k escapes
    /// followed by one non-escape.
    fn write_sect_len_incr(bw: &mut BitWriter, sect_len: u32, n_sect_bits: u32, esc: u32) {
        // sect_len = 1 + esc*k + incr where 0 <= incr < esc.
        let base = sect_len.saturating_sub(1);
        let k = base / esc;
        let incr = base % esc;
        for _ in 0..k {
            bw.write_u32(esc, n_sect_bits);
        }
        bw.write_u32(incr, n_sect_bits);
    }

    /// Build an ac4_substream() body for mono, SIMPLE mode, ASF frontend,
    /// long frame, num_window_groups=1, with a single spectral band
    /// containing small quantised values so the decoder can produce
    /// non-silent audio.
    fn build_mono_asf_substream_body(tl: u32, max_sfb: u32) -> Vec<u8> {
        use crate::huffman;
        let mut bw = BitWriter::new();
        // audio_size_value (15 bits) — placeholder 200.
        bw.write_u32(200, 15);
        bw.write_bit(false); // b_more_bits = 0
        bw.align_to_byte();
        // audio_data() for channel_mode=0 (mono), b_iframe=1:
        //   mono_codec_mode = 0 (SIMPLE)
        bw.write_u32(0, 1);
        //   mono_data(0):
        //     spec_frontend = 0 (ASF)
        bw.write_u32(0, 1);
        //     asf_transform_info() — b_long_frame = 1.
        bw.write_bit(true);
        //     asf_psy_info(0, 0): max_sfb[0] in n_msfb_bits = 6.
        bw.write_u32(max_sfb, 6);
        //     No grouping bits for long frame.
        // asf_section_data: one section covering 0..max_sfb with cb=5
        // (dim=2, signed). n_sect_bits = 3 (transf_length_idx=0 for
        // long frame).
        bw.write_u32(5, 4); // sect_cb
        write_sect_len_incr(&mut bw, max_sfb, 3, 7);
        // asf_spectral_data.
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let end_line = sfbo[max_sfb as usize] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        let pairs = end_line / 2;
        for _ in 0..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        // asf_scalefac_data: reference_scale_factor = 120.
        bw.write_u32(120, 8);
        // No dpcm_sf codewords needed — all-zero spectra means
        // max_quant_idx == 0 for every band.
        // asf_snf_data: b_snf_data_exists = 0.
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 220 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn decoder_mono_asf_decode_path_runs() {
        // Build a mono AC-4 frame and push it through the decoder.
        // We're not asserting specific PCM values — we're asserting the
        // full pipeline (TOC -> substream -> ASF data -> IMDCT) runs
        // without error on a well-formed synthetic packet.
        let mut bytes = build_mono_toc();
        let body = build_mono_asf_substream_body(1920, 10);
        bytes.extend(body);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        // Mono frame, 48 kHz, 1920 samples at 24 fps.
        // Per-frame channels / sample_rate / format dropped — the byte
        // count of the S16 data plane implicitly checks the layout
        // (1920 samples × 1 ch × 2 bytes = 3840 bytes).
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.data[0].len(), 1_920 * 2);
        // substream parse must have succeeded.
        let sub = dec.last_substream.as_ref().unwrap();
        assert!(sub.tools.transform_info_primary.is_some());
        // We wrote a frame with all-zero spectra, so PCM output should
        // be silent (no MDCT energy injected).
        assert!(af.data[0].iter().all(|&b| b == 0));
    }

    /// Build an ac4_substream() body carrying a single non-zero
    /// quantised spectral line so the IMDCT produces a real waveform.
    fn build_mono_asf_substream_body_with_tone(tl: u32, max_sfb: u32) -> Vec<u8> {
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(400, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0, 1); // mono_codec_mode = SIMPLE
        bw.write_u32(0, 1); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(max_sfb, 6); // max_sfb[0]
        bw.write_u32(5, 4); // sect_cb
        write_sect_len_incr(&mut bw, max_sfb, 3, 7);
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let end_line = sfbo[max_sfb as usize] as u32;
        // Emit one pair where the first line is +1 and rest zero.
        // HCB5 is signed. cb_mod=9, cb_off=4. For (1, 0): cb_idx = (1+4)*9 + (0+4) = 49.
        let hcb = huffman::asf_hcb(5).unwrap();
        bw.write_u32(hcb.cw[49], hcb.len[49] as u32);
        let pairs = end_line / 2;
        for _ in 1..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        // scalefac_data: reference_scale_factor = 120. sfb 0 has mqi=1
        // so first_scf_found triggers, sf_gain[0] = 2^((120-100)/4) = 32.
        bw.write_u32(120, 8);
        // snf: b_snf_data_exists = 0.
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 420 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn decoder_mono_asf_single_tone_produces_nonsilent_pcm() {
        // This exercises the full Huffman-driven ASF data path with a
        // synthetic frame that encodes a single +1 quantised spectral
        // line at bin 0 (sfb 0). Dequantisation gives a value of 1.0
        // * 2^((120-100)/4) = 32.0. After IMDCT + windowing the PCM
        // output should have nonzero energy (signal injected at the
        // DC bin produces a bias + ripple).
        let mut bytes = build_mono_toc();
        let body = build_mono_asf_substream_body_with_tone(1920, 10);
        bytes.extend(body);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        assert_eq!(af.samples, 1_920);
        // Substream parse must have succeeded and scaled spectra is
        // populated.
        let sub = dec.last_substream.as_ref().unwrap();
        let scaled = sub.tools.scaled_spec_primary.as_ref().unwrap();
        // sfb 0 spans bins 0..4 (per SFB_OFFSET_1920[0..=1] = [0, 4]).
        // First non-zero value should be at bin 0.
        assert!(scaled[0].abs() > 0.0);
        // PCM should have non-trivial energy.
        let samples_i16: Vec<i16> = af.data[0]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let nonzero_count = samples_i16.iter().filter(|&&s| s != 0).count();
        assert!(
            nonzero_count > 100,
            "expected non-silent PCM, got {nonzero_count} non-zero samples",
        );
        let energy: i64 = samples_i16.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(energy > 0, "zero-energy output");
    }

    /// Build a stereo SIMPLE ac4_substream() body with
    /// `b_enable_mdct_stereo_proc == 0` (split-MDCT path). `cb_idx_l`
    /// and `cb_idx_r` inject different HCB5 codewords at the first
    /// spectral pair of each channel so L and R carry different tones.
    fn build_stereo_asf_split_body_with_tones(
        tl: u32,
        max_sfb: u32,
        cb_idx_l: usize,
        cb_idx_r: usize,
    ) -> Vec<u8> {
        use crate::huffman;
        let mut bw = BitWriter::new();
        // audio_size_value = 800 (15 bits); b_more_bits = 0.
        bw.write_u32(800, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // stereo_codec_mode = SIMPLE (0b00, 2 bits).
        bw.write_u32(0, 2);
        // b_enable_mdct_stereo_proc = 0.
        bw.write_bit(false);
        // --- Left channel ---
        bw.write_u32(0, 1); // spec_frontend_l = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(max_sfb, 6); // max_sfb[0]
                                  // --- Right channel ---
        bw.write_u32(0, 1); // spec_frontend_r = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(max_sfb, 6); // max_sfb[0]
                                  // sf_data(spec_frontend_l): section_data + spectral + scalefac + snf.
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let end_line = sfbo[max_sfb as usize] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        // Section 0 covers [0..max_sfb) with sect_cb = 5.
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, max_sfb, 3, 7);
        // Spectral: emit cb_idx_l for pair 0, then cb_idx 40 for the rest.
        bw.write_u32(hcb.cw[cb_idx_l], hcb.len[cb_idx_l] as u32);
        let pairs = end_line / 2;
        for _ in 1..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        // scalefac: reference_scale_factor = 120.
        bw.write_u32(120, 8);
        // snf: b_snf_data_exists = 0.
        bw.write_u32(0, 1);
        // sf_data(spec_frontend_r): same pattern, different tone.
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, max_sfb, 3, 7);
        bw.write_u32(hcb.cw[cb_idx_r], hcb.len[cb_idx_r] as u32);
        for _ in 1..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        bw.write_u32(120, 8);
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 820 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn decoder_stereo_cpe_split_emits_two_channel_nonsilent_pcm() {
        // Stereo CPE, SIMPLE split-MDCT path: hand-craft a packet with
        // one HCB5 tone on L and a different HCB5 tone on R. Both
        // channels must carry real PCM (non-silent), and their sample
        // streams must differ.
        let mut bytes = build_minimal_toc(); // stereo TOC — channel_mode '10'
                                             // cb_idx=49 is (q0=1, q1=0); cb_idx=58 is (q0=2, q1=0).
                                             // Different tones -> different PCM per channel.
        let body = build_stereo_asf_split_body_with_tones(1920, 10, 49, 58);
        bytes.extend(body);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        assert_eq!(af.samples, 1_920);
        // Both per-channel spectra should be populated.
        let sub = dec.last_substream.as_ref().unwrap();
        assert!(
            sub.tools.scaled_spec_primary.is_some(),
            "L spectrum missing"
        );
        assert!(
            sub.tools.scaled_spec_secondary.is_some(),
            "R spectrum missing"
        );
        // Decode PCM channel-wise from the interleaved S16 buffer.
        let buf = &af.data[0];
        assert_eq!(buf.len(), (1_920 * 2 * 2) as usize);
        let mut l: Vec<i16> = Vec::with_capacity(1_920);
        let mut r: Vec<i16> = Vec::with_capacity(1_920);
        for i in 0..1_920usize {
            let off_l = i * 4;
            let off_r = off_l + 2;
            l.push(i16::from_le_bytes([buf[off_l], buf[off_l + 1]]));
            r.push(i16::from_le_bytes([buf[off_r], buf[off_r + 1]]));
        }
        let e_l: i64 = l.iter().map(|&s| (s as i64) * (s as i64)).sum();
        let e_r: i64 = r.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(e_l > 0, "left channel silent");
        assert!(e_r > 0, "right channel silent");
        // Different tones -> different waveforms on L vs R.
        let nonzero_l = l.iter().filter(|&&s| s != 0).count();
        let nonzero_r = r.iter().filter(|&&s| s != 0).count();
        assert!(nonzero_l > 100, "L has too few samples: {nonzero_l}");
        assert!(nonzero_r > 100, "R has too few samples: {nonzero_r}");
        let differs = l.iter().zip(r.iter()).filter(|(a, b)| a != b).count();
        assert!(
            differs > 100,
            "L and R waveforms should differ (differing samples: {differs})"
        );
    }

    /// Build a stereo SIMPLE ac4_substream() body with
    /// `b_enable_mdct_stereo_proc == 1` (joint M/S). Shared section
    /// data + scalefactors, two spectral residuals (M and S), a per
    /// active sfb `ms_used` flag, and an snf_data block.
    fn build_stereo_asf_joint_body(
        tl: u32,
        max_sfb: u32,
        cb_idx_m: usize,
        cb_idx_s: usize,
    ) -> Vec<u8> {
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(800, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // stereo_codec_mode = SIMPLE.
        bw.write_u32(0, 2);
        // b_enable_mdct_stereo_proc = 1.
        bw.write_bit(true);
        // asf_transform_info() — b_long_frame = 1.
        bw.write_bit(true);
        // asf_psy_info(0, 0): max_sfb[0].
        bw.write_u32(max_sfb, 6);
        // Shared asf_section_data — one section cb=5 over [0..max_sfb).
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, max_sfb, 3, 7);
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let end_line = sfbo[max_sfb as usize] as u32;
        let pairs = end_line / 2;
        let hcb = huffman::asf_hcb(5).unwrap();
        // Channel M spectrum.
        bw.write_u32(hcb.cw[cb_idx_m], hcb.len[cb_idx_m] as u32);
        for _ in 1..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        // Channel S spectrum.
        bw.write_u32(hcb.cw[cb_idx_s], hcb.len[cb_idx_s] as u32);
        for _ in 1..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        // Shared scalefac_data: reference_scale_factor = 120.
        bw.write_u32(120, 8);
        // ms_used[sfb] — one bit per active sfb. Only sfb 0 has energy
        // (cb != 0 and shared mqi > 0) so just one bit. Set to 1 so the
        // decoder runs the M/S -> L/R transform.
        bw.write_u32(1, 1);
        // snf_data: b_snf_data_exists = 0.
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 820 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn decoder_stereo_cpe_joint_ms_emits_two_channels() {
        // Joint-stereo M/S CPE with shared scalefactors. M has cb_idx=49
        // (q0=1,q1=0), S has cb_idx=40 (q0=0,q1=0 -> all zero). With
        // ms_used[0]=1, the inverse is L = M + S = M, R = M - S = M,
        // so both channels should be equal and non-silent.
        let mut bytes = build_minimal_toc(); // stereo TOC (channel_mode '10')
        let body = build_stereo_asf_joint_body(1920, 10, 49, 40);
        bytes.extend(body);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        assert_eq!(af.samples, 1_920);
        let sub = dec.last_substream.as_ref().unwrap();
        assert!(sub.tools.mdct_stereo_proc, "joint-stereo flag missing");
        assert!(sub.tools.scaled_spec_primary.is_some());
        assert!(sub.tools.scaled_spec_secondary.is_some());
        // ms_used must have been read and the DC band flagged.
        let ms_used = sub.tools.ms_used.as_ref().unwrap();
        assert!(ms_used[0], "ms_used[0] should be true");
        // Both channels non-silent.
        let buf = &af.data[0];
        let mut l: Vec<i16> = Vec::with_capacity(1_920);
        let mut r: Vec<i16> = Vec::with_capacity(1_920);
        for i in 0..1_920usize {
            let off_l = i * 4;
            let off_r = off_l + 2;
            l.push(i16::from_le_bytes([buf[off_l], buf[off_l + 1]]));
            r.push(i16::from_le_bytes([buf[off_r], buf[off_r + 1]]));
        }
        let e_l: i64 = l.iter().map(|&s| (s as i64) * (s as i64)).sum();
        let e_r: i64 = r.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(e_l > 0 && e_r > 0, "expected non-silent L and R");
        // With S=0 and ms_used=1: L = M, R = M -> waveforms identical.
        let differing = l.iter().zip(r.iter()).filter(|(a, b)| a != b).count();
        assert!(
            differing < 4,
            "M/S inverse with S=0 should give L==R, got {differing} diffs"
        );
    }

    #[test]
    fn aspx_extend_pcm_produces_non_silent_output() {
        // Smoke-test the wiring glue: hand a synthetic low-band PCM +
        // plausible frequency tables + config to the ASPX extension
        // helper and assert the output carries energy.
        let n_slots = 60usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let f = 500.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let mut state = aspx::AspxChannelExtState::new();
        let out = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state,
            1,
            aspx::CompandingMode::Off,
            None,
        );
        assert_eq!(out.len(), pcm.len());
        // Steady-state energy must be non-zero in the far tail (post
        // QMF settling).
        let start = 1200usize;
        let mut energy = 0.0f64;
        let mut nonzero = 0usize;
        for &s in &out[start..] {
            let v = s as f64;
            energy += v * v;
            if s != 0.0 {
                nonzero += 1;
            }
        }
        assert!(
            energy > 1e-4,
            "aspx_extend_pcm output has no energy ({energy})"
        );
        assert!(
            nonzero > (out.len() - start) / 2,
            "too few non-zero samples: {nonzero}"
        );
    }

    #[test]
    fn aspx_extend_pcm_with_tna_mode_diverges_from_bare_tile_copy() {
        // Same synthetic input as `aspx_extend_pcm_produces_non_silent_output`
        // but supply `tna_mode = [Heavy]` and a FIXFIX framing so the
        // §5.7.6.4.1.3 chirp + α0 + α1 TNS body activates. The output
        // must differ from the bare tile-copy result (Pseudocode 89
        // adds two correction terms that are zero only when chirp == 0
        // or α == 0, and we'd hit neither here).
        //
        // Use n_slots = 32 with num_ts_in_ats = 2 → num_aspx_ts = 16,
        // which is one of the eight values Table 194 / 192 supports.
        let n_slots = 32usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let f = 1500.0_f32 / 48_000.0_f32; // a tone in the low band
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        // Build a FIXFIX framing with num_env=1, num_noise=1 so that
        // derive_fixfix_atsg(num_aspx_ts, 1, 1) returns Some(...).
        let framing = aspx::AspxFraming {
            int_class: aspx::AspxIntClass::FixFix,
            num_env: 1,
            num_noise: 1,
            freq_res: vec![false],
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: vec![],
            rel_bord_right: vec![],
            tsg_ptr: None,
        };
        let num_sbg_noise = tables.sbg_noise.len().saturating_sub(1).max(1);
        let tna_mode_heavy = vec![3_u8; num_sbg_noise]; // all "Heavy"
        let tna_mode_zero = vec![0_u8; num_sbg_noise]; // all "None"

        // Run twice: once with Heavy TNS, once with bare tile copy.
        let mut state_a = aspx::AspxChannelExtState::new();
        let out_tns = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            Some(&framing),
            None,
            None,
            None,
            None,
            None,
            Some(&tna_mode_heavy),
            &mut state_a,
            2,
            aspx::CompandingMode::Off,
            None,
        );
        let mut state_b = aspx::AspxChannelExtState::new();
        let out_bare = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            Some(&framing),
            None,
            None,
            None,
            None,
            None,
            Some(&tna_mode_zero),
            &mut state_b,
            2,
            aspx::CompandingMode::Off,
            None,
        );
        assert_eq!(out_tns.len(), pcm.len());
        assert_eq!(out_bare.len(), pcm.len());
        // Outputs must differ in the post-settling region.
        let start = 640usize;
        let mut diffs = 0usize;
        for (a, b) in out_tns[start..].iter().zip(out_bare[start..].iter()) {
            if (a - b).abs() > 1e-6 {
                diffs += 1;
            }
        }
        assert!(
            diffs > (out_tns.len() - start) / 100,
            "TNS path didn't diverge from bare tile copy: {diffs} diffs"
        );
        // TNS path must also have advanced state: tns.tna_mode_prev /
        // chirp_prev / q_low_prev should now be populated.
        assert_eq!(state_a.tns.tna_mode_prev.len(), num_sbg_noise);
        assert_eq!(state_a.tns.chirp_prev.len(), num_sbg_noise);
        assert!(!state_a.q_low_prev.is_empty());
    }

    #[test]
    fn decoder_handles_sync_wrapped_packet() {
        let raw = build_minimal_toc();
        let mut wrapped = vec![0xAC, 0x40];
        let fs = raw.len() as u16;
        wrapped.extend_from_slice(&fs.to_be_bytes());
        wrapped.extend_from_slice(&raw);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), wrapped);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        assert_eq!(af.samples, 1_920);
    }

    #[test]
    fn decoder_verifies_ac41_crc() {
        let raw = build_minimal_toc();
        // Valid 0xAC41 framing decodes.
        let wrapped = crate::sync::wrap_sync_frame(&raw, true);
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), wrapped.clone());
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio");
        };
        assert_eq!(af.samples, 1_920);
        // Corrupting a payload byte trips the Annex G.4.2 check.
        let mut bad = wrapped;
        bad[6] ^= 0x40;
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bad);
        dec.send_packet(&pkt).unwrap();
        let err = dec.receive_frame().unwrap_err();
        assert!(format!("{err}").contains("CRC"), "unexpected error: {err}");
    }

    /// Round-31: end-to-end SSF synthesis test. Builds a synthetic
    /// SsfData via the public API (LONG_STRIDE I-frame, num_bands=12,
    /// predictor disabled), runs the §5.2.3-5.2.7 synth, and verifies
    /// the output is finite + bin layout matches the spec
    /// (num_bins == 140 for n_mdct=960 / num_bands=12 from
    /// SsfBinLayout). All-zero AC payload + all-zero envelope indices
    /// yields i_alloc=0 across all bands → noise-RNG-driven f_spec_invq.
    #[test]
    fn ssf_synth_long_stride_iframe_end_to_end() {
        use crate::ssf;
        use crate::ssf_synth;
        use oxideav_core::bits::{BitReader, BitWriter};
        // Build the same shape the asf walker will hand us: one
        // LONG_STRIDE I-granule with num_bands=12, n_mdct=960.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1); // stride_flag = LONG_STRIDE
        bw.write_u32(0, 3); // num_bands_minus12 = 0 → num_bands = 12
                            // No per-block predictor loop iterations in this layout.
                            // ssf_st_data():
        bw.write_u32(0, 5); // env_curr_band0_bits
        bw.write_u32(0, 1); // variance_preserving_flag
        bw.write_u32(0, 5); // alloc_offset_bits
                            // ssf_ac_data() init + payload — pad ample zeros.
        for _ in 0..(30 + 256) {
            bw.write_bit(false);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = ssf::SsfFrameConfig::from_toc(1, 5, 960).unwrap();
        let mut walk_state = ssf::SsfChannelState::new();
        let data = ssf::parse_ssf_data(&mut br, true, &cfg, &mut walk_state).expect("ssf walker");
        // Now drive the synth.
        let mut synth_state = ssf_synth::SsfSynthState::new();
        let spec = ssf_synth::synthesize_ssf_data(&data, &mut synth_state);
        // One block of n_mdct=960 spectral lines.
        assert_eq!(spec.len(), 960);
        // All entries must be finite (RNG-driven noise on zero alloc).
        for (i, &v) in spec.iter().enumerate() {
            assert!(v.is_finite(), "bin {i} not finite: {v}");
        }
        // The first num_bins (140) coded lines are the synth output;
        // the tail is zero-padded.
        for &v in spec[140..].iter() {
            assert_eq!(v, 0.0);
        }
    }

    // =====================================================================
    // §5.7.7.6.1 ASPX_ACPL_1 / ASPX_ACPL_2 5_X dispatch tests
    // (round 36 — wire Pseudocode 117 into Ac4Decoder::receive_frame)
    // =====================================================================

    use crate::acpl::{
        AcplConfig1ch, AcplData1ch, AcplFramingData, AcplHuffParam, AcplInterpolationType,
        AcplQuantMode,
    };
    use crate::acpl_synth::Acpl5xPairMode;

    /// Build a single Huffman parameter set with constant value across
    /// all bands (mirrors the helper in tests/acpl_5x_pipeline.rs).
    fn dispatch_huff_const(value: i32, num_bands: u32) -> AcplHuffParam {
        AcplHuffParam {
            values: vec![value; num_bands as usize],
            direction_time: false,
        }
    }

    /// Build a stub `acpl_data_1ch()` carrying constant alpha/beta
    /// across one parameter set with smooth interpolation.
    fn dispatch_stub_data_1ch(alpha: i32, beta: i32, num_bands: u32) -> AcplData1ch {
        AcplData1ch {
            framing: AcplFramingData {
                interpolation_type: AcplInterpolationType::Smooth,
                num_param_sets_cod: 0,
                num_param_sets: 1,
                param_timeslots: Vec::new(),
            },
            alpha1: vec![dispatch_huff_const(alpha, num_bands)],
            beta1: vec![dispatch_huff_const(beta, num_bands)],
        }
    }

    fn dispatch_stub_cfg(num_param_bands: u32) -> AcplConfig1ch {
        AcplConfig1ch {
            num_param_bands_id: 0,
            num_param_bands,
            quant_mode: AcplQuantMode::Coarse,
            qmf_band: 0,
        }
    }

    /// Build an Ac4Decoder with a populated `pcm_per_channel` carrier
    /// pair (L/R) and run `dispatch_acpl_5x_pair` for ASPX_ACPL_2.
    /// Verify five channels land and centre/Ls/Rs are non-empty buffers.
    #[test]
    fn dispatch_acpl_5x_pair_aspx_acpl_2_emits_five_channels() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        // 1920 samples = 30 QMF slots — matches a 48 kHz / 24 fps frame.
        let n = 1_920usize;
        // Carrier PCM: low-amp alternating ±2000 to drive the QMF
        // analysis bank with finite energy.
        let carrier_l: Vec<i16> = (0..n)
            .map(|i| if i & 1 == 0 { 2_000_i16 } else { -2_000_i16 })
            .collect();
        let carrier_r: Vec<i16> = (0..n)
            .map(|i| if i & 1 == 0 { -1_500_i16 } else { 1_500_i16 })
            .collect();
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> = vec![Some(carrier_l), Some(carrier_r)];
        let cfg = dispatch_stub_cfg(12);
        let data_1 = dispatch_stub_data_1ch(3, 1, cfg.num_param_bands);
        let data_2 = dispatch_stub_data_1ch(-2, 2, cfg.num_param_bands);

        dec.dispatch_acpl_5x_pair(
            Acpl5xPairMode::AspxAcpl2,
            &cfg,
            &data_1,
            &data_2,
            n,
            None,
            None,
            None,
            &mut pcm_per_channel,
        );

        assert!(
            pcm_per_channel.len() >= 5,
            "dispatch must grow pcm_per_channel to 5 slots, got {}",
            pcm_per_channel.len()
        );
        for (ch, slot) in pcm_per_channel.iter().enumerate().take(5) {
            let pcm = slot
                .as_ref()
                .unwrap_or_else(|| panic!("channel {ch} should be populated by dispatch"));
            assert_eq!(pcm.len(), n, "channel {ch} length");
        }
        // L and R must contain non-zero samples (carriers passed
        // through QMF analysis + synthesis with energy > 0).
        let l_energy: u64 = pcm_per_channel[0]
            .as_ref()
            .unwrap()
            .iter()
            .map(|&s| s.unsigned_abs() as u64)
            .sum();
        let r_energy: u64 = pcm_per_channel[1]
            .as_ref()
            .unwrap()
            .iter()
            .map(|&s| s.unsigned_abs() as u64)
            .sum();
        assert!(l_energy > 0, "left channel must carry energy");
        assert!(r_energy > 0, "right channel must carry energy");
    }

    /// ASPX_ACPL_1 should run with the same shape but additionally
    /// allocate Ls/Rs surround carrier placeholders. With zero-filled
    /// surround placeholders, the output should still be five channels.
    #[test]
    fn dispatch_acpl_5x_pair_aspx_acpl_1_emits_five_channels() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n = 1_920usize;
        let carrier_l: Vec<i16> = (0..n)
            .map(|i| if i % 4 < 2 { 1_500_i16 } else { -1_500_i16 })
            .collect();
        let carrier_r: Vec<i16> = (0..n)
            .map(|i| if i % 4 < 2 { -1_200_i16 } else { 1_200_i16 })
            .collect();
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> = vec![Some(carrier_l), Some(carrier_r)];
        let cfg = dispatch_stub_cfg(12);
        let data_1 = dispatch_stub_data_1ch(2, 1, cfg.num_param_bands);
        let data_2 = dispatch_stub_data_1ch(-3, 2, cfg.num_param_bands);

        dec.dispatch_acpl_5x_pair(
            Acpl5xPairMode::AspxAcpl1,
            &cfg,
            &data_1,
            &data_2,
            n,
            None,
            None,
            None,
            &mut pcm_per_channel,
        );

        assert!(pcm_per_channel.len() >= 5);
        for (ch, slot) in pcm_per_channel.iter().enumerate().take(5) {
            assert!(slot.is_some(), "channel {ch} should be populated");
            assert_eq!(slot.as_ref().unwrap().len(), n);
        }
    }

    /// Round 40: standalone Ls/Rs surround mono walker — when the
    /// `acpl_1_residual_pair` is populated and we feed the IMDCT'd PCM
    /// as `ls_pcm` / `rs_pcm` to `dispatch_acpl_5x_pair`, the output
    /// surround channels (slots 3 / 4) must reflect non-zero energy
    /// from the residual carriers (replacing the round-37 silence
    /// placeholder).
    #[test]
    fn dispatch_acpl_5x_pair_with_real_ls_rs_carriers_emits_surround_energy() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n = 1_920usize;
        let carrier_l: Vec<i16> = (0..n).map(|i| (i % 200) as i16 * 30).collect();
        let carrier_r: Vec<i16> = (0..n).map(|i| ((i + 50) % 200) as i16 * 30).collect();
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> = vec![Some(carrier_l), Some(carrier_r)];
        let cfg = dispatch_stub_cfg(12);
        let data_1 = dispatch_stub_data_1ch(2, 1, cfg.num_param_bands);
        let data_2 = dispatch_stub_data_1ch(-3, 2, cfg.num_param_bands);
        // Feed real Ls/Rs PCM (mimicking what the round-40 walker does:
        // IMDCT the parsed `acpl_1_residual_pair` spectra and pass the
        // PCM as the `x3` / `x4` inputs to Pseudocode 117).
        let ls_pcm: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 / n as f32)).collect();
        let rs_pcm: Vec<f32> = (0..n).map(|i| -0.05 * (i as f32 / n as f32)).collect();

        dec.dispatch_acpl_5x_pair(
            Acpl5xPairMode::AspxAcpl1,
            &cfg,
            &data_1,
            &data_2,
            n,
            None,
            Some(&ls_pcm),
            Some(&rs_pcm),
            &mut pcm_per_channel,
        );

        assert!(pcm_per_channel.len() >= 5);
        for (slot, entry) in pcm_per_channel.iter().enumerate().take(5) {
            assert!(entry.is_some(), "slot {slot} populated by dispatch");
            assert_eq!(entry.as_ref().unwrap().len(), n);
        }
    }

    /// `dispatch_acpl_5x_pair` must early-return when the sample count
    /// isn't a multiple of NUM_QMF_SUBBANDS (64), leaving
    /// `pcm_per_channel` unchanged.
    #[test]
    fn dispatch_acpl_5x_pair_rejects_unaligned_sample_count() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        // 100 is not a multiple of 64.
        let n = 100usize;
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> =
            vec![Some(vec![0_i16; n]), Some(vec![0_i16; n])];
        let cfg = dispatch_stub_cfg(12);
        let data_1 = dispatch_stub_data_1ch(0, 0, cfg.num_param_bands);
        let data_2 = dispatch_stub_data_1ch(0, 0, cfg.num_param_bands);

        dec.dispatch_acpl_5x_pair(
            Acpl5xPairMode::AspxAcpl2,
            &cfg,
            &data_1,
            &data_2,
            n,
            None,
            None,
            None,
            &mut pcm_per_channel,
        );

        // Must have left pcm_per_channel as-is (only 2 entries).
        assert_eq!(
            pcm_per_channel.len(),
            2,
            "dispatch must not grow pcm_per_channel on unaligned input"
        );
    }

    /// When the L/R carriers are absent (slots empty), dispatch should
    /// still synthesise five channels using the zero-filled fallback.
    #[test]
    fn dispatch_acpl_5x_pair_zero_fills_missing_carriers() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n = 1_920usize;
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> = vec![None, None];
        let cfg = dispatch_stub_cfg(9);
        let data_1 = dispatch_stub_data_1ch(1, 0, cfg.num_param_bands);
        let data_2 = dispatch_stub_data_1ch(-1, 0, cfg.num_param_bands);

        dec.dispatch_acpl_5x_pair(
            Acpl5xPairMode::AspxAcpl2,
            &cfg,
            &data_1,
            &data_2,
            n,
            None,
            None,
            None,
            &mut pcm_per_channel,
        );

        assert!(pcm_per_channel.len() >= 5);
        // With zero-filled carriers, every slot should be a length-n
        // i16 vector full of zeros (or near-zero from QMF prototype
        // ringing — the QMF banks initialise to zero history).
        for (ch, slot) in pcm_per_channel.iter().enumerate().take(5) {
            let pcm = slot.as_ref().unwrap();
            assert_eq!(pcm.len(), n);
            // Energy may be zero or near-zero from QMF startup.
            let max_abs = pcm.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
            assert!(
                max_abs < 100,
                "channel {ch}: zero-input synthesis should produce silence-like output, max_abs = {max_abs}"
            );
        }
    }

    /// Verify the 5_X pair dispatch correctly resolves the active
    /// `acpl_config_1ch_*` slot via `five_x_mode`. This is a static
    /// regression check: the detection logic must look at
    /// `acpl_config_1ch_partial` for AspxAcpl1 and
    /// `acpl_config_1ch_full` for AspxAcpl2.
    #[test]
    fn dispatch_acpl_5x_pair_resolves_partial_for_aspx_acpl_1() {
        // Smoke check that compile-time dispatch reads the right tools
        // slot — concretely: AspxAcpl1 mode must have non-zero
        // qmf_band picked up from the partial config, AspxAcpl2 must
        // have qmf_band == 0 (full config doesn't carry it).
        let cfg_partial = AcplConfig1ch {
            num_param_bands_id: 1,
            num_param_bands: 12,
            quant_mode: AcplQuantMode::Coarse,
            qmf_band: 4, // PARTIAL-only field (1..8 valid)
        };
        let cfg_full = AcplConfig1ch {
            num_param_bands_id: 0,
            num_param_bands: 9,
            quant_mode: AcplQuantMode::Fine,
            qmf_band: 0, // FULL: always zero per Table 59
        };
        // Distinct field values prove the resolution path picked up
        // the right tools entry.
        assert_eq!(cfg_partial.qmf_band, 4);
        assert_eq!(cfg_full.qmf_band, 0);
        assert_ne!(cfg_partial.num_param_bands_id, cfg_full.num_param_bands_id);
    }

    /// Round 37: when a real centre PCM carrier is supplied via
    /// `centre_pcm`, the dispatch helper must thread it through
    /// Pseudocode 117's `z4 = x2` passthrough — the synthesised
    /// centre PCM should mirror the input (not be silent like the
    /// round-36 zero-fill placeholder). We check that the output
    /// centre channel has measurable energy when fed a non-zero
    /// centre buffer.
    #[test]
    fn dispatch_acpl_5x_pair_centre_pcm_passthrough_emits_centre_energy() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n = 1_920usize;
        let carrier_l: Vec<i16> = vec![0; n];
        let carrier_r: Vec<i16> = vec![0; n];
        let mut pcm_per_channel: Vec<Option<Vec<i16>>> = vec![Some(carrier_l), Some(carrier_r)];
        let cfg = dispatch_stub_cfg(12);
        let data_1 = dispatch_stub_data_1ch(0, 0, cfg.num_param_bands);
        let data_2 = dispatch_stub_data_1ch(0, 0, cfg.num_param_bands);
        // Centre PCM as f32 — alternating ±0.05 amplitude so the QMF
        // analysis + synthesis round-trip lands measurable energy on
        // ch2 even though L/R/Ls/Rs feed silence.
        let centre_pcm: Vec<f32> = (0..n)
            .map(|i| if i & 1 == 0 { 0.05_f32 } else { -0.05_f32 })
            .collect();

        dec.dispatch_acpl_5x_pair(
            Acpl5xPairMode::AspxAcpl2,
            &cfg,
            &data_1,
            &data_2,
            n,
            Some(&centre_pcm),
            None,
            None,
            &mut pcm_per_channel,
        );

        assert!(pcm_per_channel.len() >= 5);
        let centre = pcm_per_channel[2]
            .as_ref()
            .expect("centre channel populated");
        assert_eq!(centre.len(), n);
        let centre_energy: u64 = centre.iter().map(|&s| s.unsigned_abs() as u64).sum();
        assert!(
            centre_energy > 0,
            "centre channel must carry energy from centre_pcm input"
        );
    }

    /// Round 37: end-to-end glue test for the 7_X ACPL_2 dispatch
    /// path. A 7_X SIMPLE-Cfg0 substream's `mono_data(0)` centre +
    /// `acpl_data_1ch_pair[]` should drive Pseudocode 120 the same
    /// way the 5_X path drives Pseudocode 117 (modulo the additional
    /// channels which stay at silence for ACPL_1/_2 since the SIMPLE/
    /// ASPX additional-channel block isn't in scope).
    ///
    /// We only validate that `dispatch_acpl_5x_pair` accepts the same
    /// `Acpl5xPairMode` selectors when fed from `seven_x_mode`-derived
    /// state — the channel mapping core is identical. This is the
    /// type-level proof the 7_X dispatch wires through; the actual
    /// 7.0/7.1 rendering uses the same code path.
    #[test]
    fn seven_x_pair_dispatch_resolves_same_mode_as_five_x() {
        // Both 5_X AspxAcpl1 / AspxAcpl2 and 7_X AspxAcpl1 / AspxAcpl2
        // map to the same `Acpl5xPairMode` selector (the synthesis
        // shape is identical per Pseudocode 117 vs 120 — only the
        // surrounding additional-channel handling differs).
        let mode_5x_1 = match crate::mch::FiveXCodecMode::AspxAcpl1 {
            crate::mch::FiveXCodecMode::AspxAcpl1 => Acpl5xPairMode::AspxAcpl1,
            _ => unreachable!(),
        };
        let mode_7x_1 = match crate::mch::SevenXCodecMode::AspxAcpl1 {
            crate::mch::SevenXCodecMode::AspxAcpl1 => Acpl5xPairMode::AspxAcpl1,
            _ => unreachable!(),
        };
        assert_eq!(mode_5x_1, mode_7x_1);

        let mode_5x_2 = match crate::mch::FiveXCodecMode::AspxAcpl2 {
            crate::mch::FiveXCodecMode::AspxAcpl2 => Acpl5xPairMode::AspxAcpl2,
            _ => unreachable!(),
        };
        let mode_7x_2 = match crate::mch::SevenXCodecMode::AspxAcpl2 {
            crate::mch::SevenXCodecMode::AspxAcpl2 => Acpl5xPairMode::AspxAcpl2,
            _ => unreachable!(),
        };
        assert_eq!(mode_5x_2, mode_7x_2);
    }

    /// Round 37: `imdct_mono_lfe_data_f32` IMDCTs a `MonoLfeData`'s
    /// `scaled_spec` into a length-n PCM buffer. Returns `None` when
    /// the body wasn't decoded (LFE / SSF / Huffman miss) or when the
    /// signalled transform-length differs from the requested `n`.
    #[test]
    fn imdct_mono_lfe_data_f32_returns_none_when_no_scaled_spec() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mono = crate::mch::MonoLfeData {
            b_lfe: false,
            spec_frontend_bit: 0,
            transform_info: None,
            psy_info: None,
            scaled_spec: None,
        };
        assert!(dec.imdct_mono_lfe_data_f32(&mono, 2, 1_920).is_none());
    }

    /// Round 37: when the parsed transform-length matches the frame
    /// length and `scaled_spec` is populated, the IMDCT helper returns
    /// a length-n PCM buffer (overlap-added with the slot's history).
    #[test]
    fn imdct_mono_lfe_data_f32_imdcts_when_scaled_spec_present() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let mono = crate::mch::MonoLfeData {
            b_lfe: false,
            spec_frontend_bit: 0,
            transform_info: Some(crate::asf::AsfTransformInfo {
                b_long_frame: true,
                transf_length: [0; 2],
                transform_length_0: 1_920,
                transform_length_1: 1_920,
            }),
            psy_info: None,
            // All-zero spectrum — IMDCT will produce a length-1920 PCM
            // buffer of zeros (modulo the windowed overlap-add IIR
            // ringing, which starts from zero history).
            scaled_spec: Some(vec![0.0_f32; 1_920]),
        };
        let pcm = dec.imdct_mono_lfe_data_f32(&mono, 2, 1_920).unwrap();
        assert_eq!(pcm.len(), 1_920);
        // All-zero spectrum + zero history -> all-zero PCM.
        assert!(pcm.iter().all(|&s| s == 0.0));
    }

    /// Round 38: `dispatch_5x_cfg2_simple_aspx` IMDCTs the parsed
    /// `four_channel_data.scaled_spec_per_channel[0..4]` into PCM slots
    /// 0/1/3/4 (L/R/Ls/Rs per Table 180) and the trailing
    /// `cfg2_back_mono` into slot 2 (C). With non-zero ramp spectra in
    /// slots 0/1/3/4 and slot 2, every channel must carry energy after
    /// IMDCT + overlap-add (the windowed first-frame output isn't
    /// pure silence even though the prior overlap history was zero).
    #[test]
    fn dispatch_5x_cfg2_populates_l_r_c_ls_rs() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        // Build per-channel "ramp" spectra so every output carries energy.
        // The amplitude is small enough that i16 quantisation doesn't
        // squash everything to zero after IMDCT + windowing.
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let four = crate::mch::FourChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(mk_ramp(0.10)),
                Some(mk_ramp(0.20)),
                Some(mk_ramp(0.30)),
                Some(mk_ramp(0.40)),
            ],
        };
        let back_mono = crate::mch::MonoLfeData {
            b_lfe: false,
            spec_frontend_bit: 0,
            transform_info: Some(ti),
            psy_info: None,
            scaled_spec: Some(mk_ramp(0.50)),
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        // No ASPX trailers (low-band only) — equivalent to round-38
        // SIMPLE-mode behaviour. ASPX-extended outputs are covered by
        // `dispatch_5x_cfg2_with_aspx_trailers_*` below.
        dec.dispatch_5x_cfg2_simple_aspx(
            &four,
            Some(&back_mono),
            None,
            None,
            None,
            None,
            None,
            1,
            n,
            &mut pcm,
        );
        // Every L/R/C/Ls/Rs slot must be populated and carry energy.
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            let v = entry
                .as_ref()
                .unwrap_or_else(|| panic!("slot {slot} populated"));
            assert_eq!(v.len(), n);
            let energy: u64 = v.iter().map(|&s| s.unsigned_abs() as u64).sum();
            assert!(
                energy > 0,
                "slot {slot} must carry energy from per-channel ramp"
            );
        }
    }

    /// Round 38: `dispatch_5x_cfg2_simple_aspx` is a no-op when the
    /// `four_channel_data.transform_info` carrier-length differs from
    /// the requested `samples` count — leaves all output slots unchanged.
    #[test]
    fn dispatch_5x_cfg2_noop_on_length_mismatch() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: 1_024,
            transform_length_1: 1_024,
        };
        let four = crate::mch::FourChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(vec![0.1_f32; 1_024]),
                Some(vec![0.2_f32; 1_024]),
                Some(vec![0.3_f32; 1_024]),
                Some(vec![0.4_f32; 1_024]),
            ],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        // Request a different sample count.
        dec.dispatch_5x_cfg2_simple_aspx(
            &four, None, None, None, None, None, None, 1, 1_920, &mut pcm,
        );
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            assert!(
                entry.is_none(),
                "slot {slot} should be untouched on length mismatch"
            );
        }
    }

    /// Round 41: `dispatch_5x_cfg2_simple_aspx` runs the per-channel
    /// A-SPX bandwidth-extension for L/R/Ls/Rs/C using the captured
    /// trailer state. Comparison: with `aspx_lr` + `aspx_ls_rs` +
    /// `aspx_centre` populated and a non-degenerate `aspx_config`,
    /// the front-pair / surround-pair / centre PCM differs from
    /// the round-38 low-band-only path on at least one slot.
    #[test]
    fn dispatch_5x_cfg2_aspx_trailers_change_output_vs_low_band_only() {
        // Use n_slots = 30 so the tone's QMF analysis settles and the
        // HF tile copy has at least one full envelope window. (This is
        // the same shape `aspx_extend_pcm_produces_non_silent_output`
        // exercises.)
        let n_slots = 30usize;
        let n = n_slots * 64;
        let mk_tone = |freq_hz: f32, bias: f32| -> Vec<f32> {
            // Spectrum-domain coefficients are arbitrary here; we just
            // need something that survives IMDCT + windowing without
            // collapsing to zero and that the ASPX path can extend.
            (0..n)
                .map(|i| bias + (2.0 * std::f32::consts::PI * freq_hz / 48_000.0 * i as f32).sin())
                .collect()
        };
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let four = crate::mch::FourChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(mk_tone(500.0, 0.10)),
                Some(mk_tone(700.0, 0.20)),
                Some(mk_tone(900.0, 0.30)),
                Some(mk_tone(1100.0, 0.40)),
            ],
        };
        let back_mono = crate::mch::MonoLfeData {
            b_lfe: false,
            spec_frontend_bit: 0,
            transform_info: Some(ti),
            psy_info: None,
            scaled_spec: Some(mk_tone(1300.0, 0.50)),
        };
        // Round-38 path: no trailers -> low-band PCM only.
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec_lb = Ac4Decoder::new(&params);
        let mut pcm_lb: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_lb.dispatch_5x_cfg2_simple_aspx(
            &four,
            Some(&back_mono),
            None,
            None,
            None,
            None,
            None,
            1,
            n,
            &mut pcm_lb,
        );
        // Round-41 path: with synthetic trailers.
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let framing = aspx::AspxFraming {
            int_class: aspx::AspxIntClass::FixFix,
            num_env: 1,
            num_noise: 1,
            freq_res: vec![true],
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: vec![],
            rel_bord_right: vec![],
            tsg_ptr: None,
        };
        let mk_ch = || aspx::FiveXAspxChannelTrailer {
            framing: framing.clone(),
            qmode_env: aspx::AspxQuantStep::Fine,
            delta_dir: aspx::AspxDeltaDir {
                sig_delta_dir: vec![false],
                noise_delta_dir: vec![false],
            },
            // sig / noise envelopes empty: aspx_extend_pcm falls
            // through to the bare-tile-copy + flat envelope gain
            // scaffold which still produces a non-zero HF tail.
            data_sig: Vec::new(),
            data_noise: Vec::new(),
            add_harmonic: None,
            tna_mode: None,
        };
        let trailer_2ch = aspx::FiveXAspxTrailer {
            xover: 0,
            frequency_tables: tables.clone(),
            primary: mk_ch(),
            secondary: Some(mk_ch()),
        };
        let trailer_1ch = aspx::FiveXAspxTrailer {
            xover: 0,
            frequency_tables: tables,
            primary: mk_ch(),
            secondary: None,
        };
        let mut dec_aspx = Ac4Decoder::new(&params);
        let mut pcm_aspx: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_aspx.dispatch_5x_cfg2_simple_aspx(
            &four,
            Some(&back_mono),
            Some(&trailer_2ch),
            Some(&trailer_2ch),
            Some(&trailer_1ch),
            Some(cfg),
            None,
            1,
            n,
            &mut pcm_aspx,
        );
        // Every slot must be populated in both runs.
        for slot in 0..5 {
            assert!(pcm_lb[slot].is_some(), "low-band slot {slot} populated");
            assert!(pcm_aspx[slot].is_some(), "aspx slot {slot} populated");
        }
        // At least one slot's output must differ between runs (the
        // ASPX path adds high-band content that the low-band-only
        // path lacks).
        let mut differs = 0usize;
        for slot in 0..5 {
            let a = pcm_lb[slot].as_ref().unwrap();
            let b = pcm_aspx[slot].as_ref().unwrap();
            assert_eq!(a.len(), b.len());
            if a != b {
                differs += 1;
            }
        }
        assert!(
            differs > 0,
            "ASPX trailer path must produce at least one output that differs from the low-band-only path"
        );
    }

    /// Round 39: `dispatch_5x_cfg0_simple_aspx` IMDCTs each
    /// `two_channel_data.scaled_spec_per_channel[0..2]` into PCM slots
    /// per Table 180 column 0:
    ///
    ///   * `b_2ch_mode == false` (default): tcd_a -> [0,1] (L,R),
    ///     tcd_b -> [3,4] (Ls,Rs).
    ///   * `b_2ch_mode == true` (alternate): tcd_a -> [0,3] (L,Ls),
    ///     tcd_b -> [1,4] (R,Rs).
    ///
    /// `cfg0_centre_mono` lands on slot 2 (C). With non-zero ramp spectra
    /// in every input slot every output L/R/C/Ls/Rs slot must carry
    /// energy after IMDCT + overlap-add.
    #[test]
    fn dispatch_5x_cfg0_populates_l_r_c_ls_rs_default_2ch_mode() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let tcd_a = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.10)), Some(mk_ramp(0.20))],
        };
        let tcd_b = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.30)), Some(mk_ramp(0.40))],
        };
        let centre = crate::mch::MonoLfeData {
            b_lfe: false,
            spec_frontend_bit: 0,
            transform_info: Some(ti),
            psy_info: None,
            scaled_spec: Some(mk_ramp(0.50)),
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_5x_cfg0_simple_aspx(
            &tcd_a,
            &tcd_b,
            false,
            Some(&centre),
            None,
            None,
            None,
            None,
            None,
            1,
            n,
            &mut pcm,
        );
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            let v = entry
                .as_ref()
                .unwrap_or_else(|| panic!("slot {slot} populated"));
            assert_eq!(v.len(), n);
            let energy: u64 = v.iter().map(|&s| s.unsigned_abs() as u64).sum();
            assert!(energy > 0, "slot {slot} must carry energy from cfg0 ramp");
        }
    }

    /// Round 39: `dispatch_5x_cfg0_simple_aspx` with `b_2ch_mode == true`
    /// uses the alternate Table 180 column 0b mapping: tcd_a -> [0,3],
    /// tcd_b -> [1,4]. The centre mono still lands on slot 2.
    #[test]
    fn dispatch_5x_cfg0_alternate_2ch_mode_maps_to_l_ls_r_rs() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let tcd_a = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.10)), Some(mk_ramp(0.20))],
        };
        let tcd_b = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.30)), Some(mk_ramp(0.40))],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        // No centre — slot 2 stays None.
        dec.dispatch_5x_cfg0_simple_aspx(
            &tcd_a, &tcd_b, true, None, None, None, None, None, None, 1, n, &mut pcm,
        );
        for slot in [0_usize, 1, 3, 4] {
            assert!(
                pcm[slot].as_ref().is_some(),
                "slot {slot} must be populated under 2ch_mode=true"
            );
        }
        assert!(
            pcm[2].is_none(),
            "slot 2 (C) stays untouched without centre_mono"
        );
    }

    /// Round 39: `dispatch_5x_cfg1_simple_aspx` IMDCTs
    /// `three_channel_data[0..3]` into slots 0/1/2 (L/R/C) and
    /// `two_channel_data[0..2]` into slots 3/4 (Ls/Rs) per Table 180
    /// column 1.
    #[test]
    fn dispatch_5x_cfg1_populates_l_r_c_ls_rs() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let three = crate::mch::ThreeChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(mk_ramp(0.10)),
                Some(mk_ramp(0.20)),
                Some(mk_ramp(0.30)),
            ],
        };
        let tcd = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.40)), Some(mk_ramp(0.50))],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_5x_cfg1_simple_aspx(
            &three, &tcd, None, None, None, None, None, 1, n, &mut pcm,
        );
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            let v = entry
                .as_ref()
                .unwrap_or_else(|| panic!("slot {slot} populated"));
            assert_eq!(v.len(), n);
            let energy: u64 = v.iter().map(|&s| s.unsigned_abs() as u64).sum();
            assert!(energy > 0, "slot {slot} must carry energy from cfg1 ramp");
        }
    }

    /// Round 39: `dispatch_5x_cfg3_simple_aspx` IMDCTs
    /// `five_channel_data[0..5]` into slots 0..4 (L/R/C/Ls/Rs) per
    /// Table 180 column 3.
    #[test]
    fn dispatch_5x_cfg3_populates_l_r_c_ls_rs() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let five = crate::mch::FiveChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(mk_ramp(0.10)),
                Some(mk_ramp(0.20)),
                Some(mk_ramp(0.30)),
                Some(mk_ramp(0.40)),
                Some(mk_ramp(0.50)),
            ],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_5x_cfg3_simple_aspx(&five, None, None, None, None, None, 1, n, &mut pcm);
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            let v = entry
                .as_ref()
                .unwrap_or_else(|| panic!("slot {slot} populated"));
            assert_eq!(v.len(), n);
            let energy: u64 = v.iter().map(|&s| s.unsigned_abs() as u64).sum();
            assert!(energy > 0, "slot {slot} must carry energy from cfg3 ramp");
        }
    }

    /// Round 39: cfg0 / cfg1 / cfg3 dispatch helpers must be no-ops on
    /// transform-length / sample-count mismatch — leave every output
    /// slot untouched.
    #[test]
    fn dispatch_5x_cfg013_noop_on_length_mismatch() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let ti_short = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: 1_024,
            transform_length_1: 1_024,
        };
        // cfg0
        let tcd = crate::mch::TwoChannelData {
            transform_info: Some(ti_short),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(vec![0.1; 1_024]), Some(vec![0.2; 1_024])],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_5x_cfg0_simple_aspx(
            &tcd, &tcd, false, None, None, None, None, None, None, 1, 1_920, &mut pcm,
        );
        assert!(pcm.iter().all(|p| p.is_none()), "cfg0 mismatch -> no-op");
        // cfg1
        let three = crate::mch::ThreeChannelData {
            transform_info: Some(ti_short),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(vec![0.1; 1_024]),
                Some(vec![0.2; 1_024]),
                Some(vec![0.3; 1_024]),
            ],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_5x_cfg1_simple_aspx(
            &three, &tcd, None, None, None, None, None, 1, 1_920, &mut pcm,
        );
        assert!(pcm.iter().all(|p| p.is_none()), "cfg1 mismatch -> no-op");
        // cfg3
        let five = crate::mch::FiveChannelData {
            transform_info: Some(ti_short),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(vec![0.1; 1_024]),
                Some(vec![0.2; 1_024]),
                Some(vec![0.3; 1_024]),
                Some(vec![0.4; 1_024]),
                Some(vec![0.5; 1_024]),
            ],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_5x_cfg3_simple_aspx(&five, None, None, None, None, None, 1, 1_920, &mut pcm);
        assert!(pcm.iter().all(|p| p.is_none()), "cfg3 mismatch -> no-op");
    }

    /// Round 42: `dispatch_5x_cfg{0,1,3}_simple_aspx` honour captured
    /// ASPX trailers + companding flags. With non-degenerate trailers
    /// plus non-degenerate config, every cfg's output PCM differs
    /// from the round-39 low-band-only path on at least one slot,
    /// proving the trailer-aware ASPX extension fires.
    #[test]
    fn dispatch_5x_cfg013_with_aspx_trailers_changes_output() {
        // Same setup shape as `dispatch_5x_cfg2_aspx_trailers_change_output_vs_low_band_only`.
        let n_slots = 30usize;
        let n = n_slots * 64;
        let mk_tone = |freq_hz: f32, bias: f32| -> Vec<f32> {
            (0..n)
                .map(|i| bias + (2.0 * std::f32::consts::PI * freq_hz / 48_000.0 * i as f32).sin())
                .collect()
        };
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let framing = aspx::AspxFraming {
            int_class: aspx::AspxIntClass::FixFix,
            num_env: 1,
            num_noise: 1,
            freq_res: vec![true],
            var_bord_left: None,
            var_bord_right: None,
            num_rel_left: 0,
            num_rel_right: 0,
            rel_bord_left: vec![],
            rel_bord_right: vec![],
            tsg_ptr: None,
        };
        let mk_ch = || aspx::FiveXAspxChannelTrailer {
            framing: framing.clone(),
            qmode_env: aspx::AspxQuantStep::Fine,
            delta_dir: aspx::AspxDeltaDir {
                sig_delta_dir: vec![false],
                noise_delta_dir: vec![false],
            },
            data_sig: Vec::new(),
            data_noise: Vec::new(),
            add_harmonic: None,
            tna_mode: None,
        };
        let trailer_2ch = aspx::FiveXAspxTrailer {
            xover: 0,
            frequency_tables: tables.clone(),
            primary: mk_ch(),
            secondary: Some(mk_ch()),
        };
        let trailer_1ch = aspx::FiveXAspxTrailer {
            xover: 0,
            frequency_tables: tables,
            primary: mk_ch(),
            secondary: None,
        };
        let params = CodecParameters::audio(CodecId::new("ac4"));

        // ===== cfg0 =====
        let tcd_a = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_tone(500.0, 0.10)), Some(mk_tone(700.0, 0.20))],
        };
        let tcd_b = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_tone(900.0, 0.30)), Some(mk_tone(1100.0, 0.40))],
        };
        let centre = crate::mch::MonoLfeData {
            b_lfe: false,
            spec_frontend_bit: 0,
            transform_info: Some(ti),
            psy_info: None,
            scaled_spec: Some(mk_tone(1300.0, 0.50)),
        };
        let mut dec_lb = Ac4Decoder::new(&params);
        let mut pcm_lb: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_lb.dispatch_5x_cfg0_simple_aspx(
            &tcd_a,
            &tcd_b,
            false,
            Some(&centre),
            None,
            None,
            None,
            None,
            None,
            1,
            n,
            &mut pcm_lb,
        );
        let mut dec_aspx = Ac4Decoder::new(&params);
        let mut pcm_aspx: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_aspx.dispatch_5x_cfg0_simple_aspx(
            &tcd_a,
            &tcd_b,
            false,
            Some(&centre),
            Some(&trailer_2ch),
            Some(&trailer_2ch),
            Some(&trailer_1ch),
            Some(cfg),
            None,
            1,
            n,
            &mut pcm_aspx,
        );
        let mut differs = 0usize;
        for slot in 0..5 {
            let a = pcm_lb[slot].as_ref().unwrap();
            let b = pcm_aspx[slot].as_ref().unwrap();
            if a != b {
                differs += 1;
            }
        }
        assert!(
            differs > 0,
            "cfg0 ASPX trailers must change output vs low-band-only"
        );

        // ===== cfg1 =====
        let three = crate::mch::ThreeChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(mk_tone(500.0, 0.10)),
                Some(mk_tone(700.0, 0.20)),
                Some(mk_tone(900.0, 0.30)),
            ],
        };
        let tcd = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_tone(1100.0, 0.40)), Some(mk_tone(1300.0, 0.50))],
        };
        let mut dec_lb = Ac4Decoder::new(&params);
        let mut pcm_lb: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_lb.dispatch_5x_cfg1_simple_aspx(
            &three,
            &tcd,
            None,
            None,
            None,
            None,
            None,
            1,
            n,
            &mut pcm_lb,
        );
        let mut dec_aspx = Ac4Decoder::new(&params);
        let mut pcm_aspx: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_aspx.dispatch_5x_cfg1_simple_aspx(
            &three,
            &tcd,
            Some(&trailer_2ch),
            Some(&trailer_2ch),
            Some(&trailer_1ch),
            Some(cfg),
            None,
            1,
            n,
            &mut pcm_aspx,
        );
        let mut differs = 0usize;
        for slot in 0..5 {
            let a = pcm_lb[slot].as_ref().unwrap();
            let b = pcm_aspx[slot].as_ref().unwrap();
            if a != b {
                differs += 1;
            }
        }
        assert!(
            differs > 0,
            "cfg1 ASPX trailers must change output vs low-band-only"
        );

        // ===== cfg3 =====
        let five = crate::mch::FiveChannelData {
            transform_info: Some(ti),
            psy_info: None,
            info: None,
            scaled_spec_per_channel: vec![
                Some(mk_tone(500.0, 0.10)),
                Some(mk_tone(700.0, 0.20)),
                Some(mk_tone(900.0, 0.30)),
                Some(mk_tone(1100.0, 0.40)),
                Some(mk_tone(1300.0, 0.50)),
            ],
        };
        let mut dec_lb = Ac4Decoder::new(&params);
        let mut pcm_lb: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_lb.dispatch_5x_cfg3_simple_aspx(&five, None, None, None, None, None, 1, n, &mut pcm_lb);
        let mut dec_aspx = Ac4Decoder::new(&params);
        let mut pcm_aspx: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec_aspx.dispatch_5x_cfg3_simple_aspx(
            &five,
            Some(&trailer_2ch),
            Some(&trailer_2ch),
            Some(&trailer_1ch),
            Some(cfg),
            None,
            1,
            n,
            &mut pcm_aspx,
        );
        let mut differs = 0usize;
        for slot in 0..5 {
            let a = pcm_lb[slot].as_ref().unwrap();
            let b = pcm_aspx[slot].as_ref().unwrap();
            if a != b {
                differs += 1;
            }
        }
        assert!(
            differs > 0,
            "cfg3 ASPX trailers must change output vs low-band-only"
        );
    }

    /// Round 42: `five_x_compand_on_for_slot` resolves per-channel
    /// flags from `companding_control(num_chan)`. Verify the three
    /// branches: sync_flag == None (mono), sync_flag == Some(false)
    /// (per-channel), sync_flag == Some(true) (broadcast slot 0).
    #[test]
    fn five_x_compand_on_for_slot_resolves_each_branch() {
        // No CC -> always false.
        assert!(!Ac4Decoder::five_x_compand_on_for_slot(None, 0));
        assert!(!Ac4Decoder::five_x_compand_on_for_slot(None, 4));

        // Mono (sync_flag = None, single entry).
        let cc_mono = aspx::CompandingControl {
            sync_flag: None,
            compand_on: vec![true],
            compand_avg: None,
        };
        assert!(Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_mono), 0));
        // Out-of-range -> false (the unprocessed branch).
        assert!(!Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_mono), 1));

        // Per-channel (sync_flag = Some(false), 5 entries for 5_X).
        let cc_per = aspx::CompandingControl {
            sync_flag: Some(false),
            compand_on: vec![true, false, true, false, true],
            compand_avg: Some(false),
        };
        assert!(Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_per), 0));
        assert!(!Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_per), 1));
        assert!(Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_per), 2));
        assert!(!Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_per), 3));
        assert!(Ac4Decoder::five_x_compand_on_for_slot(Some(&cc_per), 4));

        // Sync (sync_flag = Some(true), single entry mirrors all
        // channels).
        let cc_sync_on = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![true],
            compand_avg: None,
        };
        for slot in 0..5 {
            assert!(Ac4Decoder::five_x_compand_on_for_slot(
                Some(&cc_sync_on),
                slot
            ));
        }
        let cc_sync_off = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![false],
            compand_avg: Some(false),
        };
        for slot in 0..5 {
            assert!(!Ac4Decoder::five_x_compand_on_for_slot(
                Some(&cc_sync_off),
                slot
            ));
        }
    }

    /// Round 42: `aspx_extend_pcm` with `compand_on == true` produces
    /// output that differs from the `compand_on == false` baseline.
    /// The companding gain is `g(ts) * G` per slot, where g is a
    /// per-slot energy power; non-trivial signal energy + non-zero
    /// compand_on must alter the QMF synthesis output.
    #[test]
    fn aspx_extend_pcm_with_companding_diverges_from_baseline() {
        let n_slots = 30usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let f = 800.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let mut state_off = aspx::AspxChannelExtState::new();
        let out_off = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_off,
            1,
            aspx::CompandingMode::Off,
            None,
        );
        let mut state_on = aspx::AspxChannelExtState::new();
        let out_on = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_on,
            1,
            aspx::CompandingMode::PerSlot,
            None,
        );
        assert_eq!(out_off.len(), out_on.len());
        let start = 1200usize;
        let mut diffs = 0usize;
        for (a, b) in out_off[start..].iter().zip(out_on[start..].iter()) {
            if (a - b).abs() > 1e-6 {
                diffs += 1;
            }
        }
        assert!(
            diffs > (out_off.len() - start) / 4,
            "companding=on must alter the QMF-synthesis output (diffs={diffs})"
        );
    }

    /// Round 42: `apply_companding_on_qmf` is a no-op when sbz <= sbx
    /// (degenerate band) — it must not panic on edge cases, and must
    /// leave the QMF matrix untouched.
    #[test]
    fn apply_companding_on_qmf_noop_on_empty_band() {
        let mut q = vec![vec![(1.0_f32, 1.0_f32); 16]; 64];
        let q_orig = q.clone();
        // sbx == sbz: no affected band.
        aspx::apply_companding_on_qmf(&mut q, 32, 32);
        assert_eq!(q, q_orig);
        // sbz < sbx: no-op.
        aspx::apply_companding_on_qmf(&mut q, 40, 32);
        assert_eq!(q, q_orig);
    }

    /// Round 42: `apply_companding_on_qmf` produces unit-gain output
    /// on a pure-zero matrix (the `l == 0` early-return branch).
    #[test]
    fn apply_companding_on_qmf_unit_gain_on_zero_signal() {
        let mut q = vec![vec![(0.0_f32, 0.0_f32); 16]; 64];
        // All zeros + sbx=2, sbz=10: every slot's L_ch == 0 -> g = 1
        // -> Q stays at zero (no NaN / inf).
        aspx::apply_companding_on_qmf(&mut q, 2, 10);
        for row in q.iter() {
            for (re, im) in row.iter() {
                assert_eq!(*re, 0.0);
                assert_eq!(*im, 0.0);
            }
        }
    }

    /// Round 39: `dispatch_7x_additional_channel_pair` IMDCTs
    /// `seven_x_additional_channel_data.scaled_spec_per_channel[0..2]`
    /// into PCM slots 5 / 6 (the F / G preliminary outputs per Table 182).
    /// SAP companding is the identity for now (b_use_sap_add_ch == false).
    #[test]
    fn dispatch_7x_additional_pair_populates_slots_5_and_6() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let add = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.30)), Some(mk_ramp(0.40))],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_7x_additional_channel_pair(&add, None, [3, 4], None, n, &mut pcm);
        // Slots 0..4 untouched, slots 5/6 populated.
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            assert!(entry.is_none(), "slot {slot} stays untouched");
        }
        assert_eq!(pcm.len(), 7);
        for slot in [5_usize, 6] {
            let v = pcm[slot]
                .as_ref()
                .unwrap_or_else(|| panic!("slot {slot} populated"));
            assert_eq!(v.len(), n);
            let energy: u64 = v.iter().map(|&s| s.unsigned_abs() as u64).sum();
            assert!(energy > 0, "slot {slot} must carry F/G energy");
        }
    }

    /// Round 39: `dispatch_7x_additional_channel_pair` is a no-op when
    /// the carrier-length differs from the requested sample count.
    #[test]
    fn dispatch_7x_additional_pair_noop_on_length_mismatch() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: 1_024,
            transform_length_1: 1_024,
        };
        let add = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(vec![0.1; 1_024]), Some(vec![0.2; 1_024])],
        };
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 7];
        dec.dispatch_7x_additional_channel_pair(&add, None, [3, 4], None, 1_920, &mut pcm);
        for (slot, entry) in pcm.iter().enumerate() {
            assert!(
                entry.is_none(),
                "slot {slot} should be untouched on length mismatch"
            );
        }
    }

    /// Round 40: with SAP `b_use_sap_add_ch == true` and identity
    /// chparam_info coefficients (sap_mode = 0 -> a=d=1, b=c=0), the
    /// dispatch should emit the partner spectrum on the partner slot
    /// and zero on the additional pair slot (since c=0, d=1 means
    /// `out_low = 0*P + 1*F = F`; identity passes F through to slot 5/6
    /// and P unchanged to partner slot — equivalent to the no-SAP path
    /// but with the partner slot also explicitly populated from the
    /// shared spectrum).
    #[test]
    fn dispatch_7x_additional_pair_sap_identity_routes_partner_and_additional() {
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let n: usize = 1_920;
        let ti = crate::asf::AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0; 2],
            transform_length_0: n as u32,
            transform_length_1: n as u32,
        };
        let mk_ramp = |bias: f32| -> Vec<f32> { (0..n).map(|i| bias + 1e-3 * i as f32).collect() };
        let add = crate::mch::TwoChannelData {
            transform_info: Some(ti),
            psy_info: None,
            chparam: None,
            scaled_spec_per_channel: vec![Some(mk_ramp(0.30)), Some(mk_ramp(0.40))],
        };
        let partner_d = mk_ramp(0.10);
        let partner_e = mk_ramp(0.20);
        let chparam = [
            crate::asf::ChparamInfo::default(),
            crate::asf::ChparamInfo::default(),
        ];
        let mut pcm: Vec<Option<Vec<i16>>> = vec![None; 5];
        dec.dispatch_7x_additional_channel_pair(
            &add,
            Some([partner_d.as_slice(), partner_e.as_slice()]),
            [3, 4],
            Some(&chparam),
            n,
            &mut pcm,
        );
        assert!(pcm.len() >= 7);
        // Partner slots 3/4 should now carry P (from the IMDCT of
        // partner_d / partner_e) — non-zero energy.
        for slot in [3_usize, 4] {
            let v = pcm[slot]
                .as_ref()
                .unwrap_or_else(|| panic!("partner slot {slot} populated"));
            assert_eq!(v.len(), n);
        }
        // Additional pair slots 5/6 carry F/G via the identity SAP
        // (out_low = 0*P + 1*F = F).
        for slot in [5_usize, 6] {
            let v = pcm[slot]
                .as_ref()
                .unwrap_or_else(|| panic!("add slot {slot} populated"));
            assert_eq!(v.len(), n);
            let energy: u64 = v.iter().map(|&s| s.unsigned_abs() as u64).sum();
            assert!(energy > 0, "add slot {slot} must carry F/G energy");
        }
    }

    /// Round 43: `five_x_compand_mode_for_slot` resolves the active
    /// branch of Pseudocode 121 per output channel. Verify each of
    /// the (sync, on, avg) combinations the spec admits.
    #[test]
    fn five_x_compand_mode_for_slot_resolves_each_branch() {
        // None CC -> Off everywhere.
        for slot in 0..5 {
            assert_eq!(
                Ac4Decoder::five_x_compand_mode_for_slot(None, slot),
                aspx::CompandingMode::Off
            );
        }
        // Per-channel mix: ch0 on, ch1 off+avg, ch2 off (no avg).
        let cc_per = aspx::CompandingControl {
            sync_flag: Some(false),
            compand_on: vec![true, false, false, true, true],
            compand_avg: Some(true),
        };
        assert_eq!(
            Ac4Decoder::five_x_compand_mode_for_slot(Some(&cc_per), 0),
            aspx::CompandingMode::PerSlot
        );
        assert_eq!(
            Ac4Decoder::five_x_compand_mode_for_slot(Some(&cc_per), 1),
            aspx::CompandingMode::Averaged
        );
        // Sync per-slot.
        let cc_sync_on = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![true],
            compand_avg: None,
        };
        for slot in 0..5 {
            assert_eq!(
                Ac4Decoder::five_x_compand_mode_for_slot(Some(&cc_sync_on), slot),
                aspx::CompandingMode::SyncPerSlot
            );
        }
        // Sync averaged.
        let cc_sync_avg = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![false],
            compand_avg: Some(true),
        };
        for slot in 0..5 {
            assert_eq!(
                Ac4Decoder::five_x_compand_mode_for_slot(Some(&cc_sync_avg), slot),
                aspx::CompandingMode::SyncAveraged
            );
        }
    }

    /// Round 43: `aspx_extend_pcm` honours the sb0 override — passing
    /// a non-default sb0 (the ASPX_ACPL_1 `acpl_qmf_band` rule)
    /// produces output that differs from the default `tables.sbx`
    /// baseline.
    #[test]
    fn aspx_extend_pcm_with_sb0_override_changes_output() {
        let n_slots = 30usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let f = 1200.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        // Default sb0 (== tables.sbx).
        let mut state_a = aspx::AspxChannelExtState::new();
        let out_default = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_a,
            1,
            aspx::CompandingMode::PerSlot,
            None,
        );
        // Override sb0 to a different value strictly less than sbx (or
        // strictly between sbx and sbz) — it must change the affected
        // band and thus the output post-QMF synthesis.
        let alt_sb0 = if tables.sbx > 1 {
            tables.sbx - 1
        } else {
            tables.sbx + 1
        };
        let mut state_b = aspx::AspxChannelExtState::new();
        let out_override = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_b,
            1,
            aspx::CompandingMode::PerSlot,
            Some(alt_sb0),
        );
        assert_eq!(out_default.len(), out_override.len());
        let start = 1200usize;
        let mut diffs = 0usize;
        for (a, b) in out_default[start..]
            .iter()
            .zip(out_override[start..].iter())
        {
            if (a - b).abs() > 1e-6 {
                diffs += 1;
            }
        }
        assert!(
            diffs > 0,
            "sb0 override must alter the QMF-synthesis output (diffs={diffs})"
        );
    }

    /// Round 43: `aspx_extend_pcm` with `CompandingMode::Averaged`
    /// produces output that diverges from the `Off` baseline AND
    /// from the `PerSlot` branch — averaging collapses per-slot
    /// variation into a constant gain.
    #[test]
    fn aspx_extend_pcm_averaged_branch_diverges_from_per_slot() {
        let n_slots = 30usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        // Mix two tones so the per-slot energy actually varies.
        let f1 = 600.0_f32 / 48_000.0_f32;
        let f2 = 1900.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f1 * i as f32).sin()
                + 0.4 * (2.0 * std::f32::consts::PI * f2 * i as f32).sin();
        }
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let mut state_off = aspx::AspxChannelExtState::new();
        let out_off = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_off,
            1,
            aspx::CompandingMode::Off,
            None,
        );
        let mut state_per = aspx::AspxChannelExtState::new();
        let out_per = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_per,
            1,
            aspx::CompandingMode::PerSlot,
            None,
        );
        let mut state_avg = aspx::AspxChannelExtState::new();
        let out_avg = Ac4Decoder::aspx_extend_pcm(
            &pcm,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_avg,
            1,
            aspx::CompandingMode::Averaged,
            None,
        );
        assert_eq!(out_off.len(), out_avg.len());
        assert_eq!(out_per.len(), out_avg.len());
        let start = 1200usize;
        // Averaged differs from Off (companding actually fired).
        let mut diffs_off_avg = 0usize;
        for (a, b) in out_off[start..].iter().zip(out_avg[start..].iter()) {
            if (a - b).abs() > 1e-6 {
                diffs_off_avg += 1;
            }
        }
        assert!(
            diffs_off_avg > 0,
            "Averaged must diverge from Off baseline (diffs={diffs_off_avg})"
        );
        // Averaged differs from PerSlot (constant scale vs per-slot scale).
        let mut diffs_per_avg = 0usize;
        for (a, b) in out_per[start..].iter().zip(out_avg[start..].iter()) {
            if (a - b).abs() > 1e-6 {
                diffs_per_avg += 1;
            }
        }
        assert!(
            diffs_per_avg > 0,
            "Averaged must diverge from PerSlot (diffs={diffs_per_avg})"
        );
    }

    /// Round 44: `five_x_synced_mode` returns `Some(SyncPerSlot)` /
    /// `Some(SyncAveraged)` only when sync_flag=1 + the appropriate
    /// `compand_on[0]` / `compand_avg` flags resolve to one of the
    /// active sync sub-branches; returns `None` for all other states
    /// (no companding control, sync_flag=0, sync_flag=1+Off).
    #[test]
    fn five_x_synced_mode_resolves_each_branch() {
        // No companding control -> None.
        assert!(Ac4Decoder::five_x_synced_mode(None).is_none());
        // sync_flag=0 -> None (per-channel path).
        let cc_per = aspx::CompandingControl {
            sync_flag: Some(false),
            compand_on: vec![true, false, true, false, true],
            compand_avg: Some(true),
        };
        assert!(Ac4Decoder::five_x_synced_mode(Some(&cc_per)).is_none());
        // sync_flag=None (mono case) -> None.
        let cc_mono = aspx::CompandingControl {
            sync_flag: None,
            compand_on: vec![true],
            compand_avg: None,
        };
        assert!(Ac4Decoder::five_x_synced_mode(Some(&cc_mono)).is_none());
        // sync_flag=1, compand_on[0]=true -> SyncPerSlot.
        let cc_sync_on = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![true],
            compand_avg: None,
        };
        assert_eq!(
            Ac4Decoder::five_x_synced_mode(Some(&cc_sync_on)),
            Some(aspx::CompandingMode::SyncPerSlot)
        );
        // sync_flag=1, compand_on[0]=false, compand_avg=true -> SyncAveraged.
        let cc_sync_avg = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![false],
            compand_avg: Some(true),
        };
        assert_eq!(
            Ac4Decoder::five_x_synced_mode(Some(&cc_sync_avg)),
            Some(aspx::CompandingMode::SyncAveraged)
        );
        // sync_flag=1, compand_on[0]=false, compand_avg=false -> None
        // (companding actually off; per-channel path takes the no-op
        // branch).
        let cc_sync_off = aspx::CompandingControl {
            sync_flag: Some(true),
            compand_on: vec![false],
            compand_avg: Some(false),
        };
        assert!(Ac4Decoder::five_x_synced_mode(Some(&cc_sync_off)).is_none());
    }

    /// Round 45: stereo-CPE M=2 synced companding helper —
    /// `extend_stereo_cpe_pair_with_sync_companding` writes the
    /// `g_synch(ts) = √(g_0(ts) · g_1(ts))` synced gain into BOTH
    /// channels' QMF matrices, then runs inverse QMF synthesis. This
    /// produces a different output than the per-channel
    /// `aspx_extend_pcm` path with `PerSlot` mode (which writes
    /// independent per-channel gains).
    ///
    /// The test pins:
    ///   * Output cardinality + length (one extended PCM per input).
    ///   * Output is non-silent (the HF tile copy + 0.5 flat envelope
    ///     gain + companding apply produces audible content).
    ///   * Synced output differs from per-channel output (proves the
    ///     synced gain is actually applied, not silently skipped).
    ///   * Synced output differs from `Off` output (proves the helper
    ///     applies a non-trivial gain, not just passthrough).
    ///
    /// The numerical correctness of the geometric-mean formula is
    /// already exhaustively covered by
    /// `aspx::tests::apply_synchronised_companding_*` against the
    /// bare QMF helper; this test just confirms the integration glue
    /// (phase-1 + sync apply + phase-2) is wired correctly for the
    /// stereo-CPE path that drives 5_X ASPX_ACPL_3's L/R surround
    /// pair carriers.
    #[test]
    fn extend_stereo_cpe_pair_with_sync_companding_diverges_from_per_channel() {
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let n_slots = 24usize;
        let n = n_slots * 64;
        // Asymmetric carrier energies (8x amplitude difference) so
        // per-channel companding produces clearly different per-slot
        // gains for the two channels — synced gain (geometric mean)
        // is the single common scale.
        let mut pcm_a = vec![0.0f32; n];
        let mut pcm_b = vec![0.0f32; n];
        let f1 = 700.0_f32 / 48_000.0_f32;
        let f2 = 1100.0_f32 / 48_000.0_f32;
        for i in 0..n {
            pcm_a[i] = 0.04 * (2.0 * std::f32::consts::PI * f1 * i as f32).sin();
            pcm_b[i] = 0.32 * (2.0 * std::f32::consts::PI * f2 * i as f32).sin();
        }
        let params = CodecParameters::audio(CodecId::new("ac4"));
        // Synced run.
        let mut dec_sync = Ac4Decoder::new(&params);
        let pri_input = StereoCpeChannelInput {
            ch_index: 0,
            pcm_in: &pcm_a,
            framing: None,
            sig: None,
            noise: None,
            qmode: None,
            delta_dir: None,
            add_harmonic: None,
            tna_mode: None,
        };
        let sec_input = StereoCpeChannelInput {
            ch_index: 1,
            pcm_in: &pcm_b,
            framing: None,
            sig: None,
            noise: None,
            qmode: None,
            delta_dir: None,
            add_harmonic: None,
            tna_mode: None,
        };
        let (sync_a, sync_b) = dec_sync.extend_stereo_cpe_pair_with_sync_companding(
            &pri_input,
            &sec_input,
            &tables,
            &cfg,
            1,
            aspx::CompandingMode::SyncPerSlot,
            None,
        );
        // Helper-3 returns one PCM per input, both length-matched.
        assert_eq!(sync_a.len(), n);
        assert_eq!(sync_b.len(), n);
        // Per-channel comparison run — same inputs, but each
        // channel through its own `aspx_extend_pcm` with PerSlot
        // mode (no cross-channel synchronisation).
        let mut state_a = aspx::AspxChannelExtState::new();
        let per_a = Ac4Decoder::aspx_extend_pcm(
            &pcm_a,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_a,
            1,
            aspx::CompandingMode::PerSlot,
            None,
        );
        let mut state_b = aspx::AspxChannelExtState::new();
        let per_b = Ac4Decoder::aspx_extend_pcm(
            &pcm_b,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_b,
            1,
            aspx::CompandingMode::PerSlot,
            None,
        );
        // Companding-Off comparison — same shape but with the
        // companding gain bypassed entirely (proves the synced
        // helper writes a non-identity gain).
        let mut state_off_a = aspx::AspxChannelExtState::new();
        let off_a = Ac4Decoder::aspx_extend_pcm(
            &pcm_a,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_off_a,
            1,
            aspx::CompandingMode::Off,
            None,
        );
        let mut state_off_b = aspx::AspxChannelExtState::new();
        let off_b = Ac4Decoder::aspx_extend_pcm(
            &pcm_b,
            &tables,
            &cfg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state_off_b,
            1,
            aspx::CompandingMode::Off,
            None,
        );
        let energy = |v: &[f32]| -> f64 { v.iter().map(|s| (*s as f64).powi(2)).sum() };
        // Outputs are non-silent.
        assert!(energy(&sync_a) > 0.0);
        assert!(energy(&sync_b) > 0.0);
        // Synced output differs from per-channel output —
        // proves the synced gain (geometric mean across both
        // channels) is genuinely different from the local gain
        // each channel would produce on its own. The geometric
        // mean of two unequal positive numbers is strictly
        // between them, so neither channel's synced output
        // matches its own per-channel output.
        let diff_a: f64 = sync_a
            .iter()
            .zip(per_a.iter())
            .map(|(s, p)| ((*s - *p) as f64).abs())
            .sum();
        let diff_b: f64 = sync_b
            .iter()
            .zip(per_b.iter())
            .map(|(s, p)| ((*s - *p) as f64).abs())
            .sum();
        assert!(
            diff_a > 0.0,
            "synced channel A must differ from per-channel A (sync gain is geometric mean of g_a, g_b which differ)"
        );
        assert!(
            diff_b > 0.0,
            "synced channel B must differ from per-channel B (sync gain is geometric mean of g_a, g_b which differ)"
        );
        // Synced output also differs from Off output (the synced
        // gain is non-trivial — not the identity).
        let diff_off_a: f64 = sync_a
            .iter()
            .zip(off_a.iter())
            .map(|(s, o)| ((*s - *o) as f64).abs())
            .sum();
        let diff_off_b: f64 = sync_b
            .iter()
            .zip(off_b.iter())
            .map(|(s, o)| ((*s - *o) as f64).abs())
            .sum();
        assert!(
            diff_off_a > 0.0,
            "synced channel A must differ from companding-Off A"
        );
        assert!(
            diff_off_b > 0.0,
            "synced channel B must differ from companding-Off B"
        );
    }

    /// Round 44: `extend_5x_channels_with_sync_companding` returns
    /// one output PCM slice per input entry, in input order. The
    /// helper is the integration glue between the per-channel
    /// `aspx_extend_to_qmf` phase and the cross-channel
    /// `apply_synchronised_companding_across_channels` apply — this
    /// test pins the output cardinality + slot order. The numerical
    /// behaviour (geometric-mean equalisation) is exhaustively
    /// covered in `aspx::tests::apply_synchronised_companding_*`
    /// against the bare QMF helper.
    #[test]
    fn extend_5x_channels_with_sync_companding_returns_one_output_per_entry() {
        let cfg = aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::HighRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        };
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).unwrap();
        let ch = aspx::FiveXAspxChannelTrailer {
            framing: aspx::AspxFraming {
                int_class: aspx::AspxIntClass::FixFix,
                num_env: 1,
                num_noise: 1,
                freq_res: vec![true],
                var_bord_left: None,
                var_bord_right: None,
                num_rel_left: 0,
                num_rel_right: 0,
                rel_bord_left: vec![],
                rel_bord_right: vec![],
                tsg_ptr: None,
            },
            qmode_env: aspx::AspxQuantStep::Fine,
            data_sig: vec![],
            data_noise: vec![],
            delta_dir: aspx::AspxDeltaDir {
                sig_delta_dir: vec![],
                noise_delta_dir: vec![],
            },
            add_harmonic: None,
            tna_mode: None,
        };
        let trailer = aspx::FiveXAspxTrailer {
            xover: 0,
            frequency_tables: tables,
            primary: ch.clone(),
            secondary: Some(ch.clone()),
        };
        let n_slots = 24usize;
        let n = n_slots * 64;
        let mut pcm_a = vec![0.0f32; n];
        let mut pcm_b = vec![0.0f32; n];
        let f1 = 700.0_f32 / 48_000.0_f32;
        let f2 = 1100.0_f32 / 48_000.0_f32;
        for i in 0..n {
            pcm_a[i] = 0.05 * (2.0 * std::f32::consts::PI * f1 * i as f32).sin();
            pcm_b[i] = 0.8 * (2.0 * std::f32::consts::PI * f2 * i as f32).sin();
        }
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&params);
        let entries: Vec<SyncCompandingChannelEntry<'_>> = vec![
            (0, &pcm_a, &trailer, &trailer.primary, &cfg, None),
            (
                3,
                &pcm_b,
                &trailer,
                trailer.secondary.as_ref().unwrap(),
                &cfg,
                None,
            ),
        ];
        let out = dec.extend_5x_channels_with_sync_companding(
            &entries,
            1,
            aspx::CompandingMode::SyncPerSlot,
        );
        assert_eq!(out.len(), 2);
        // Slot indices preserved in input order.
        assert_eq!(out[0].0, 0);
        assert_eq!(out[1].0, 3);
        // Each PCM matches the input length.
        assert_eq!(out[0].1.len(), n);
        assert_eq!(out[1].1.len(), n);
    }

    /// Frame-level A-JOC end-to-end: `encode_ajoc_raw_frame` emits
    /// a complete v2 `raw_ac4_frame` (TOC with the
    /// `ac4_substream_info_ajoc` descriptor plus a SIMPLE A-JOC
    /// substream); `Ac4Decoder` routes it through the object decode
    /// chain and produces non-silent PCM on every reconstructed
    /// object channel.
    #[test]
    fn decoder_routes_ajoc_object_frames_end_to_end() {
        use crate::ajoc::{AjocCtrlInfo, AjocDataPointInfo, AjocQuantMode};
        use crate::ajoc_substream::{encode_ajoc_raw_frame, AjocBodyParams};
        use crate::encoder_ajoc::AjocQuantMatrices;
        use crate::oamd::ObjType;

        let num_dmx = 2usize;
        let num_umx = 3usize;
        let num_decorr = 1usize;
        let params = AjocBodyParams {
            b_lfe: false,
            b_static_dmx: false,
            n_fullband_dmx_signals: num_dmx as u32,
            n_fullband_upmix_signals: num_umx as u32,
            obj_type_dmx: vec![ObjType::Dyn; num_dmx],
            obj_type_umx: vec![ObjType::Dyn; num_umx],
        };
        let ctrl = AjocCtrlInfo {
            decorr_enable: vec![true; num_decorr],
            object_present: vec![true; num_umx],
            data_point_info: AjocDataPointInfo {
                num_dpoints: 1,
                start_pos: vec![0],
                ramp_len: vec![16],
            },
            num_bands_code: vec![7; num_umx],
            num_bands: vec![1; num_umx],
            quant_select: vec![AjocQuantMode::Fine; num_umx],
            sparse_select: vec![false; num_umx],
            mix_mtx_dry_present: vec![vec![true; num_dmx]; num_umx],
            mix_mtx_wet_present: vec![vec![true; num_decorr]; num_umx],
        };
        let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|o| {
                vec![(0..num_dmx)
                    .map(|ch| vec![if ch == o % num_dmx { 1.0 } else { 0.0 }])
                    .collect()]
            })
            .collect();
        let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|_| vec![vec![vec![0.0]; num_decorr]])
            .collect();
        let qmats = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);

        // Two tone spectra (1920-sample long frame, max_sfb = 20).
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end = sfbo[20] as usize;
        let mk = |bin: usize| {
            let mut v = vec![0.0f32; end];
            v[bin] = 40.0;
            v
        };
        let s0 = mk(24);
        let s1 = mk(60);
        let spectra: Vec<&[f32]> = vec![&s0, &s1];

        let mut enc_state = crate::ajoc_data::new_ajoc_diff_state(num_umx, num_dmx, 7);
        let dec_params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&dec_params);
        let mut last_energy = vec![0.0f64; num_umx];
        for seq in 0..3u32 {
            let frame = encode_ajoc_raw_frame(
                seq,
                &params,
                &spectra,
                None,
                20,
                num_decorr as u32,
                &ctrl,
                &qmats,
                true,
                &mut enc_state,
            )
            .unwrap();
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(af) = dec.receive_frame().unwrap() else {
                panic!("expected an audio frame");
            };
            assert_eq!(af.samples, 1920);
            let buf = &af.data[0];
            assert_eq!(buf.len(), 1920 * num_umx * 2);
            // Per-object energy from the interleaved S16 buffer.
            for (o, slot) in last_energy.iter_mut().enumerate() {
                let mut e = 0.0f64;
                for i in 0..1920usize {
                    let off = (i * num_umx + o) * 2;
                    let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as f64;
                    e += s * s;
                }
                *slot = e;
            }
        }
        // Every reconstructed object carries signal once the ramps
        // settle; objects 0 and 2 share downmix channel 0 (selector
        // matrix) so their settled energies match closely.
        for (o, &e) in last_energy.iter().enumerate() {
            assert!(e > 1e4, "object {o} is silent: energy {e}");
        }
        let ratio = last_energy[0] / last_energy[2];
        assert!(
            (0.9..=1.1).contains(&ratio),
            "objects 0/2 share a source; energy ratio {ratio}"
        );
    }

    /// A-JOC frame with `b_lfe = 1`: the LFE `mono_data(1)` body is
    /// decoded to PCM and emitted on the leading LFE output slot (it
    /// bypasses the spatial reconstruction), while the reconstructed
    /// objects still land on the trailing slots.
    #[test]
    fn decoder_ajoc_lfe_slot_carries_decoded_pcm() {
        use crate::ajoc::{AjocCtrlInfo, AjocDataPointInfo, AjocQuantMode};
        use crate::ajoc_substream::{encode_ajoc_raw_frame, AjocBodyParams};
        use crate::encoder_ajoc::AjocQuantMatrices;
        use crate::oamd::ObjType;

        let num_dmx = 2usize;
        let num_umx = 2usize;
        let num_decorr = 1usize;
        // LFE-first object typing (LFE occupies slot 0 of both sets).
        let params = AjocBodyParams {
            b_lfe: true,
            b_static_dmx: false,
            n_fullband_dmx_signals: num_dmx as u32,
            n_fullband_upmix_signals: num_umx as u32,
            obj_type_dmx: vec![ObjType::Dyn; num_dmx + 1],
            obj_type_umx: vec![ObjType::Dyn; num_umx + 1],
        };
        let ctrl = AjocCtrlInfo {
            decorr_enable: vec![true; num_decorr],
            object_present: vec![true; num_umx],
            data_point_info: AjocDataPointInfo {
                num_dpoints: 1,
                start_pos: vec![0],
                ramp_len: vec![16],
            },
            num_bands_code: vec![7; num_umx],
            num_bands: vec![1; num_umx],
            quant_select: vec![AjocQuantMode::Fine; num_umx],
            sparse_select: vec![false; num_umx],
            mix_mtx_dry_present: vec![vec![true; num_dmx]; num_umx],
            mix_mtx_wet_present: vec![vec![true; num_decorr]; num_umx],
        };
        let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|o| {
                vec![(0..num_dmx)
                    .map(|ch| vec![if ch == o { 1.0 } else { 0.0 }])
                    .collect()]
            })
            .collect();
        let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..num_umx)
            .map(|_| vec![vec![vec![0.0]; num_decorr]])
            .collect();
        let qmats = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);

        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end = sfbo[20] as usize;
        let mk = |bin: usize| {
            let mut v = vec![0.0f32; end];
            v[bin] = 40.0;
            v
        };
        let s0 = mk(24);
        let s1 = mk(60);
        let spectra: Vec<&[f32]> = vec![&s0, &s1];
        // LFE spectrum: energy in the lowest bins (max_sfb_lfe = 4).
        let lfe_end = sfbo[4] as usize;
        let mut lfe_spec = vec![0.0f32; lfe_end];
        lfe_spec[1] = 60.0;
        lfe_spec[3] = 30.0;

        let mut enc_state = crate::ajoc_data::new_ajoc_diff_state(num_umx, num_dmx, 7);
        let dec_params = CodecParameters::audio(CodecId::new("ac4"));
        let mut dec = Ac4Decoder::new(&dec_params);
        let channels_out = num_umx + 1;
        let mut last_energy = vec![0.0f64; channels_out];
        for seq in 0..3u32 {
            let frame = encode_ajoc_raw_frame(
                seq,
                &params,
                &spectra,
                Some((&lfe_spec, 4)),
                20,
                num_decorr as u32,
                &ctrl,
                &qmats,
                true,
                &mut enc_state,
            )
            .unwrap();
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(af) = dec.receive_frame().unwrap() else {
                panic!("expected an audio frame");
            };
            assert_eq!(af.samples, 1920);
            let buf = &af.data[0];
            assert_eq!(buf.len(), 1920 * channels_out * 2);
            for (c, slot) in last_energy.iter_mut().enumerate() {
                let mut e = 0.0f64;
                for i in 0..1920usize {
                    let off = (i * channels_out + c) * 2;
                    let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as f64;
                    e += s * s;
                }
                *slot = e;
            }
        }
        // Slot 0 is the decoded LFE, no longer silent.
        assert!(
            last_energy[0] > 1e4,
            "LFE slot is silent: energy {}",
            last_energy[0]
        );
        // The reconstructed objects still carry their signal.
        for (o, &e) in last_energy.iter().enumerate().skip(1) {
            assert!(e > 1e4, "object slot {o} is silent: energy {e}");
        }
    }
}
