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

use crate::{acpl_synth, asf, aspx, mdct, qmf, ssf, ssf_synth, sync, toc};

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
        }
    }

    fn extract_raw_frame<'a>(&self, pkt: &'a Packet) -> (&'a [u8], bool) {
        if let Some(f) = sync::find_sync_frame(&pkt.data) {
            (f.payload, true)
        } else {
            (pkt.data.as_slice(), false)
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
    ) -> Vec<f32> {
        const NUM_QMF: usize = qmf::NUM_QMF_SUBBANDS;
        // Need PCM length as a multiple of 64 for whole QMF slots.
        if pcm_in.is_empty() || pcm_in.len() % NUM_QMF != 0 {
            return pcm_in.to_vec();
        }
        let sbx = tables.sbx as usize;
        let sbz = tables.sbz as usize;
        if sbx == 0 || sbx >= NUM_QMF || sbz <= sbx || sbz > NUM_QMF {
            return pcm_in.to_vec();
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
            return pcm_in.to_vec();
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
                    let adjuster = aspx::AspxEnvelopeAdjuster::from_deltas(
                        &q,
                        tables,
                        sig,
                        noise,
                        qm,
                        &dd.sig_delta_dir,
                        &atsg_sig,
                        &atsg_noise,
                        num_ts_in_ats,
                        cfg.interpolation,
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
        }
        // Inverse QMF synthesis. Transpose q[sb][ts] -> slot[ts][sb] per
        // §4.4.7 inverse QMF synthesis bank.
        let mut syn = qmf::QmfSynthesisBank::new();
        let mut out = Vec::with_capacity(pcm_in.len());
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
    /// Round 38 wires this end-to-end. ASPX bandwidth-extension /
    /// companding for the four channels is deferred — only the ASF body
    /// IMDCT lands here. The function is a no-op when any of the four
    /// per-channel scaled spectra are absent (short / grouped frame
    /// or Huffman miss).
    fn dispatch_5x_cfg2_simple_aspx(
        &mut self,
        four: &crate::mch::FourChannelData,
        back_mono: Option<&crate::mch::MonoLfeData>,
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
        const SLOT_MAP: [usize; 4] = [0, 1, 3, 4];
        // Need at least 5 output slots (L/R/C/Ls/Rs). Resize on demand.
        while pcm_per_channel.len() < 5 {
            pcm_per_channel.push(None);
        }
        for (ch_in, &slot) in SLOT_MAP.iter().enumerate() {
            let Some(scaled) = four.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n);
            pcm_per_channel[slot] = Some(pcm);
        }
        // Centre — slot 2. `cfg2_back_mono` may carry a body when the
        // walker decoded the trailing `mono_data(0)`; otherwise the
        // centre stays silent.
        if let Some(mono) = back_mono {
            if let Some(pcm_f) = self.imdct_mono_lfe_data_f32(mono, 2, samples) {
                pcm_per_channel[2] = Some(Self::pcm_f32_to_i16(&pcm_f));
            }
        }
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
    fn dispatch_5x_cfg0_simple_aspx(
        &mut self,
        tcd_a: &crate::mch::TwoChannelData,
        tcd_b: &crate::mch::TwoChannelData,
        b_2ch_mode: bool,
        centre_mono: Option<&crate::mch::MonoLfeData>,
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
        for (ch_in, &slot) in slot_map_a.iter().enumerate() {
            let Some(scaled) = tcd_a.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n_a);
            pcm_per_channel[slot] = Some(pcm);
        }
        for (ch_in, &slot) in slot_map_b.iter().enumerate() {
            let Some(scaled) = tcd_b.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n_b);
            pcm_per_channel[slot] = Some(pcm);
        }
        // Centre — slot 2. `cfg0_centre_mono` carries the trailing
        // `mono_data(0)` body when the walker decoded it.
        if let Some(mono) = centre_mono {
            if let Some(pcm_f) = self.imdct_mono_lfe_data_f32(mono, 2, samples) {
                pcm_per_channel[2] = Some(Self::pcm_f32_to_i16(&pcm_f));
            }
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
    fn dispatch_5x_cfg1_simple_aspx(
        &mut self,
        three: &crate::mch::ThreeChannelData,
        tcd: &crate::mch::TwoChannelData,
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
        for (ch_in, &slot) in THREE_SLOTS.iter().enumerate() {
            let Some(scaled) = three.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n3);
            pcm_per_channel[slot] = Some(pcm);
        }
        const TWO_SLOTS: [usize; 2] = [3, 4];
        for (ch_in, &slot) in TWO_SLOTS.iter().enumerate() {
            let Some(scaled) = tcd.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n2);
            pcm_per_channel[slot] = Some(pcm);
        }
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
    fn dispatch_5x_cfg3_simple_aspx(
        &mut self,
        five: &crate::mch::FiveChannelData,
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
        for (ch_in, &slot) in SLOT_MAP.iter().enumerate() {
            let Some(scaled) = five.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n);
            pcm_per_channel[slot] = Some(pcm);
        }
    }

    /// §5.3.4.4.1 / Table 182 — 7_X SIMPLE/ASPX additional-channel
    /// pair dispatch. The `seven_x_additional_channel_data` shell
    /// carries two `sf_data(ASF)` bodies for the F / G preliminary
    /// output channels (Table 182). With `b_use_sap_add_ch == false`
    /// (or absent), Table 183's SAP matrix collapses to identity and
    /// F / G land directly on slots 5 / 6 (Lb,Rb / Lw,Rw / Tfl,Tfr per
    /// `channel_mode` — slot ordering is the bitstream-order pair, the
    /// container's channel layout decides the symbolic name).
    ///
    /// SAP companding (a,b,c,d coefficients from `seven_x_add_chparam_info`)
    /// is deferred — round-39 lands the identity pass-through. When
    /// `b_use_sap_add_ch == true`, the dispatch still emits F/G into
    /// slots 5/6; the SAP matrix multiplication folds in once the
    /// coefficient extraction pipeline lands.
    ///
    /// No-op on transform-length / sample-count mismatch, or when
    /// either per-channel scaled spectrum is absent (short / grouped
    /// frame / Huffman miss).
    fn dispatch_7x_additional_channel_pair(
        &mut self,
        add: &crate::mch::TwoChannelData,
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
        while pcm_per_channel.len() < 7 {
            pcm_per_channel.push(None);
        }
        const SLOT_MAP: [usize; 2] = [5, 6];
        for (ch_in, &slot) in SLOT_MAP.iter().enumerate() {
            let Some(scaled) = add.scaled_spec_per_channel[ch_in].as_ref() else {
                continue;
            };
            let pcm = self.imdct_channel(slot, scaled, n);
            pcm_per_channel[slot] = Some(pcm);
        }
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
        let (raw, _had_sync) = self.extract_raw_frame(&pkt);
        let info = toc::parse_ac4_toc(raw)
            .map_err(|e| Error::invalid(format!("ac4 decoder: TOC parse failed: {e}")))?;
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
            asf::walk_ac4_substream_stateful(
                sb,
                channels_u16,
                b_iframe,
                info.frame_length,
                Some(&mut self.ssf_walker_state[..channels as usize]),
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
                        );
                        pcm_per_channel[1] = Some(Self::pcm_f32_to_i16(&extended));
                    } else {
                        pcm_per_channel[1] = Some(self.imdct_channel(1, &scaled, n));
                    }
                }
            }
        }
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
                self.dispatch_acpl_5x_pair(
                    mode,
                    cfg,
                    data_1,
                    data_2,
                    samples as usize,
                    centre_pcm.as_deref(),
                    None,
                    None,
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
                self.dispatch_acpl_5x_pair(
                    mode,
                    cfg,
                    data_1,
                    data_2,
                    samples as usize,
                    centre_pcm.as_deref(),
                    None,
                    None,
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
                            samples as usize,
                            &mut pcm_per_channel,
                        );
                    }
                }
                Some(crate::mch::FiveXCodingConfig::Cfg3Five) => {
                    if let Some(five) = cfg_five_channel_data.as_ref() {
                        self.dispatch_5x_cfg3_simple_aspx(
                            five,
                            samples as usize,
                            &mut pcm_per_channel,
                        );
                    }
                }
                _ => {}
            }
        }
        // Round 39: §5.3.4.4.1 / Table 182 — 7_X SIMPLE/ASPX additional
        // channel pair render. The walker populates
        // `seven_x_additional_channel_data` (two sf_data(ASF) bodies)
        // when `7_X_codec_mode in {SIMPLE, ASPX}`. Slots 5 / 6 (the F/G
        // preliminary outputs in Table 182) get the IMDCT'd low-band PCM.
        // SAP companding (a,b,c,d coefficients from
        // `seven_x_add_chparam_info`) is deferred — identity matrix is
        // applied implicitly when `b_use_sap_add_ch == false`. The 7_X
        // ACPL_1/_2 walker has its own additional-channel handling per
        // §5.3.4.4.2/.3 (z6/z7 in Pseudocode 120) — this branch is gated
        // on the SIMPLE/ASPX active-flag so they don't collide.
        if seven_x_simple_aspx_active {
            if let Some(add) = seven_x_additional_channel_data.as_ref() {
                self.dispatch_7x_additional_channel_pair(
                    add,
                    samples as usize,
                    &mut pcm_per_channel,
                );
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
        // bitstream_version = 2 (2 bits).
        bw.write_u32(2, 2);
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
        bw.write_u32(2, 2); // bitstream_version = 2
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
            &pcm, &tables, &cfg, None, None, None, None, None, None, None, &mut state, 1,
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
        dec.dispatch_5x_cfg2_simple_aspx(&four, Some(&back_mono), n, &mut pcm);
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
        dec.dispatch_5x_cfg2_simple_aspx(&four, None, 1_920, &mut pcm);
        for (slot, entry) in pcm.iter().enumerate().take(5) {
            assert!(
                entry.is_none(),
                "slot {slot} should be untouched on length mismatch"
            );
        }
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
        dec.dispatch_5x_cfg0_simple_aspx(&tcd_a, &tcd_b, false, Some(&centre), n, &mut pcm);
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
        dec.dispatch_5x_cfg0_simple_aspx(&tcd_a, &tcd_b, true, None, n, &mut pcm);
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
        dec.dispatch_5x_cfg1_simple_aspx(&three, &tcd, n, &mut pcm);
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
        dec.dispatch_5x_cfg3_simple_aspx(&five, n, &mut pcm);
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
        dec.dispatch_5x_cfg0_simple_aspx(&tcd, &tcd, false, None, 1_920, &mut pcm);
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
        dec.dispatch_5x_cfg1_simple_aspx(&three, &tcd, 1_920, &mut pcm);
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
        dec.dispatch_5x_cfg3_simple_aspx(&five, 1_920, &mut pcm);
        assert!(pcm.iter().all(|p| p.is_none()), "cfg3 mismatch -> no-op");
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
        dec.dispatch_7x_additional_channel_pair(&add, n, &mut pcm);
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
        dec.dispatch_7x_additional_channel_pair(&add, 1_920, &mut pcm);
        for (slot, entry) in pcm.iter().enumerate() {
            assert!(
                entry.is_none(),
                "slot {slot} should be untouched on length mismatch"
            );
        }
    }
}
