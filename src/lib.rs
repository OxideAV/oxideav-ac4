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
//! * **ASF substream walker** — [`asf::walk_ac4_substream`] reads
//!   `ac4_substream()` (audio_size + variable_bits extension), the
//!   mono/stereo outer `audio_data()` layers (mono_codec_mode /
//!   stereo_codec_mode, spec_frontend, b_enable_mdct_stereo_proc),
//!   `asf_transform_info()` (Tables 99 / 100 / 103) and
//!   `asf_psy_info()` (Table 106 n_msfb_bits + Tables 109/110
//!   n_grp_bits). Surfaces the result through
//!   [`decoder::Ac4Decoder::last_substream`] so downstream tooling can
//!   see the frame's tool mix and MDCT window grouping without touching
//!   Huffman state.
//!
//! * **Coefficient pipeline** — [`huffman`] carries the normative
//!   ASF_HCB_SCALEFAC / ASF_HCB_SNF / ASF_HCB_1..11 tables from Annex
//!   A (plus CB_DIM / UNSIGNED_CB); [`sfb_offset`] carries the
//!   Annex B.4-B.7 scale-factor-band offset vectors for the 48 kHz
//!   family. [`asf_data`] walks `asf_section_data()`,
//!   `asf_spectral_data()` (with Pseudocode-19 dim=2/dim=4 split and
//!   Pseudocode-20 codebook-11 extension code),
//!   `asf_scalefac_data()` (dpcm-over-reference scale factors with
//!   `sf_gain = 2^((sf-100)/4)`), and `asf_snf_data()`.
//!   `dequantise_and_scale()` applies `rec_spec = sign(q)|q|^(4/3)`
//!   then multiplies by the band gain.
//! * **MDCT** — [`mdct`] implements the reference AC-4 IMDCT (§5.5.2
//!   pseudocodes 60-64, naive O(N^2) complex DFT) plus the KBD
//!   window family (§5.5.3, alphas from Table 186) with overlap-add.
//!
//! Known gaps (Unsupported or stubbed):
//!
//! * Short / grouped frames (`num_window_groups > 1`) — coefficient
//!   path only exercises the long-frame path today.
//! * A-SPX (`aspx_config`, `aspx_data_*`) and A-CPL tools.
//! * Speech Spectral Frontend (SSF) arithmetic-coded path.
//! * Spectral noise fill synthesis — `asf_snf_data()` parses the
//!   Huffman-coded indices but doesn't inject shaped noise into
//!   zero bands yet.
//! * Per-substream `metadata()` payload parsing (DRC, dialog norm,
//!   downmix coefficients) — bits are skipped via `substream_size`.
//! * TS 103 190-2 IFM (immersive / object) decoding.
//! * Encoder.
//!
//! The decoder emits real PCM for long-frame, single-window-group
//! mono SIMPLE/ASF streams. Everything else falls back to silence
//! with a correctly-shaped AudioFrame.

#![allow(dead_code)]

pub mod asf;
pub mod asf_data;
pub mod decoder;
pub mod huffman;
pub mod mdct;
pub mod sfb_offset;
pub mod sync;
pub mod tables;
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
