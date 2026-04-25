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
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::{asf, aspx, mdct, qmf, sync, toc};

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
        // FIXFIX framing pair we can derive atsg_sig and run the full
        // §5.7.6.4.1.3 / .4 TNS body (Pseudocodes 86 → 89). Otherwise
        // fall back to the bare tile copy in §5.7.6.4.1.4 minus the
        // chirp/α0/α1 terms.
        let mut tns_used = false;
        if let (Some(tna), Some(frm)) = (tna_mode, framing) {
            let num_aspx_ts = (n_slots as u32) / num_ts_in_ats.max(1);
            let atsg_sig_opt = if matches!(frm.int_class, aspx::AspxIntClass::FixFix) {
                aspx::derive_fixfix_atsg(num_aspx_ts, frm.num_env, frm.num_noise).map(|(s, _)| s)
            } else {
                None
            };
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
            if matches!(frm.int_class, aspx::AspxIntClass::FixFix) {
                if let Some((atsg_sig, atsg_noise)) =
                    aspx::derive_fixfix_atsg(num_aspx_ts, frm.num_env, frm.num_noise)
                {
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
                        // FIXFIX has no transient pointer (§4.3.10.4.7).
                        let aspx_tsg_ptr: u32 = 0;
                        if cfg.limiter {
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
                format: SampleFormat::S16,
                channels: self.hint_channels,
                sample_rate: self.hint_sample_rate,
                samples: 0,
                pts: pkt.pts,
                time_base: TimeBase::new(1, self.hint_sample_rate as i64),
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
        self.last_substream = substream_try.and_then(|sb| {
            let channels_u16 = channels;
            let b_iframe = info
                .presentations
                .first()
                .map(|p| p.b_iframe)
                .unwrap_or(info.b_iframe_global);
            asf::walk_ac4_substream(sb, channels_u16, b_iframe, info.frame_length).ok()
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
                    pcm_per_channel[0] = Some(Self::pcm_f32_to_i16(&extended));
                } else {
                    pcm_per_channel[0] = Some(self.imdct_channel(0, &scaled, n));
                }
            }
        }
        if channels as usize >= 2 {
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
            format: SampleFormat::S16,
            channels,
            sample_rate,
            samples,
            pts: pkt.pts,
            time_base: TimeBase::new(1, sample_rate as i64),
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
        assert_eq!(af.channels, 2);
        assert_eq!(af.sample_rate, 48_000);
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.format, SampleFormat::S16);
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
        assert_eq!(af.channels, 1);
        assert_eq!(af.sample_rate, 48_000);
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.format, SampleFormat::S16);
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
        assert_eq!(af.channels, 1);
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
        assert_eq!(af.channels, 2);
        assert_eq!(af.sample_rate, 48_000);
        assert_eq!(af.samples, 1_920);
        assert_eq!(af.format, SampleFormat::S16);
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
        assert_eq!(af.channels, 2);
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
        assert_eq!(af.channels, 2);
        assert_eq!(af.samples, 1_920);
    }
}
