//! Pure-Rust Dolby AC-4 audio decoder foundation.
//!
//! AC-4 is a complex, hierarchical codec — multiple presentations,
//! nested substream descriptors, ASF/ASF-A2/A-SPX coefficient streams
//! driven by huffman-coded scalefactor data, plus an EMDF metadata
//! sidecar carrying DRC / downmix / dialog-norm info. Full decode is
//! weeks of work.
//!
//! What this crate lands today (per ETSI TS 103 190-1 V1.4.1):
//!
//! * **Sync framing** — `ac4_syncframe()` from Annex G: `0xAC40` plain
//!   and `0xAC41` CRC-protected, 16-bit `frame_size()` with 24-bit
//!   escape, plus a standalone CRC-16 helper. See [`sync`].
//! * **Table of contents** — full `ac4_toc()` walker in [`toc`]:
//!   bitstream_version, sequence_counter, fs_index, frame_rate_index,
//!   b_iframe_global, payload_base, per-presentation
//!   `ac4_presentation_info()` (single / multi-substream, configs 0..=5
//!   + extension escape, HSF extension, pre-virtualised flag, extra
//!   EMDF substreams), per-substream `ac4_substream_info()`
//!   (channel_mode prefix decoder, sf_multiplier, bitrate_indicator,
//!   content_type w/ language tag, b_iframe),
//!   `substream_index_table()` byte sizes, and the `variable_bits(n)`
//!   codec.
//! * **Decoder** — [`decoder::Ac4Decoder`] accepts either a sync-wrapped
//!   packet (`0xAC40` / `0xAC41` prefix) or a bare MP4-style
//!   `raw_ac4_frame` payload, parses the TOC, and emits a silent S16
//!   `AudioFrame` with the correct channel count, sample rate, and
//!   samples-per-frame for the stream configuration.
//!
//! Known gaps (Unsupported or stubbed):
//!
//! * ASF / ASF-A2 / A-SPX substream coefficient decoding — the
//!   generated PCM is silence.
//! * Per-substream `metadata()` payload parsing (DRC, dialog norm,
//!   downmix coefficients) — bits are skipped via `substream_size`.
//! * TS 103 190-2 IFM (immersive / object) decoding.
//! * Encoder.
//!
//! Downstream consumers should therefore treat this crate as a
//! container / framing aid today and expect the audio to be silent
//! until the coefficient pipeline lands.

#![allow(dead_code)]

pub mod asf;
pub mod decoder;
pub mod sync;
pub mod toc;

use oxideav_codec::{CodecInfo, CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag, Result};

/// Canonical codec id.
pub const CODEC_ID_STR: &str = "ac4";

/// Register the AC-4 decoder in a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("ac4_sw")
        .with_lossy(true)
        .with_intra_only(false)
        // TS 103 190-1 supports up to 24-channel immersive configs via
        // IFM; the foundation path returns whatever the TOC declares.
        .with_max_channels(24)
        .with_max_sample_rate(192_000);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            // ISO BMFF sample entry fourcc per ETSI TS 103 190-2 Annex E.
            .tag(CodecTag::fourcc(b"ac-4")),
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_installs_decoder() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn iso_bmff_tag_resolves() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let hits: Vec<_> = reg.all_tag_registrations().collect();
        assert!(hits
            .iter()
            .any(|(t, id)| matches!(t, CodecTag::Fourcc(v) if v == b"AC-4")
                && id.as_str() == CODEC_ID_STR));
    }
}
