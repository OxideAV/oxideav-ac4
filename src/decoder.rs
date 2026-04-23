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

use crate::{sync, toc};

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
        self.last_info = Some(info);
        let byte_count = (samples as usize) * (channels as usize) * 2; // S16 interleaved.
        let data = vec![vec![0u8; byte_count]];
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
