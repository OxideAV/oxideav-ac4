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

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::{asf, mdct, sync, toc};

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
        }
    }

    fn extract_raw_frame<'a>(&self, pkt: &'a Packet) -> (&'a [u8], bool) {
        if let Some(f) = sync::find_sync_frame(&pkt.data) {
            (f.payload, true)
        } else {
            (pkt.data.as_slice(), false)
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
        // If we have scaled spectra for channel 0, run IMDCT + OLA
        // and produce real PCM. Otherwise fall back to silence.
        let mut pcm_samples: Option<Vec<i16>> = None;
        if let Some(sub) = self.last_substream.as_ref() {
            if let Some(scaled) = sub.tools.scaled_spec_primary.as_ref() {
                if let Some(ti) = sub.tools.transform_info_primary.as_ref() {
                    let n = ti.transform_length_0 as usize;
                    if n > 0 && n == samples as usize {
                        // Zero-pad the scaled coefficients up to N.
                        let mut x = vec![0.0_f32; n];
                        let copy = scaled.len().min(n);
                        x[..copy].copy_from_slice(&scaled[..copy]);
                        // Resize overlap buffer if transform length
                        // changed.
                        if self.prev_transform_length != n as u32 {
                            self.overlap.clear();
                            self.overlap.push(vec![0.0_f32; n]);
                            self.prev_transform_length = n as u32;
                        } else if self.overlap.is_empty() {
                            self.overlap.push(vec![0.0_f32; n]);
                        }
                        let y = mdct::imdct(&x);
                        let window = mdct::kbd_window(n as u32);
                        let pcm_f = mdct::imdct_olap_symmetric(
                            &y,
                            &window,
                            &mut self.overlap[0],
                        );
                        // Convert to S16, clamped.
                        let mut out = Vec::with_capacity(pcm_f.len());
                        for s in pcm_f {
                            let scaled = (s * 32767.0).clamp(-32768.0, 32767.0);
                            out.push(scaled as i16);
                        }
                        pcm_samples = Some(out);
                    }
                }
            }
        }
        self.last_info = Some(info);
        let byte_count = (samples as usize) * (channels as usize) * 2; // S16 interleaved.
        let data = if let Some(pcm) = pcm_samples {
            // Interleave mono into channels slots (dup for stereo).
            let mut buf = vec![0u8; byte_count];
            for (i, &s) in pcm.iter().enumerate() {
                let le = s.to_le_bytes();
                for c in 0..channels as usize {
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
