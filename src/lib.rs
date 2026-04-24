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
//! * **A-SPX configuration** — [`aspx::parse_aspx_config`] implements
//!   the 15-bit `aspx_config()` element (Table 50, §4.2.12.1) and
//!   [`aspx::parse_companding_control`] the `companding_control()`
//!   element (Table 49, §4.2.11). The outer `audio_data()` walker in
//!   [`asf`] now consumes these for the mono ASPX, stereo ASPX, and
//!   stereo ASPX_ACPL_{1,2} I-frame paths (for ACPL it stops before
//!   `acpl_config_1ch`, which isn't parsed yet). Exposes the parsed
//!   `AspxConfig` through
//!   [`decoder::Ac4Decoder::last_substream`]`.tools.aspx_config`.
//! * **A-SPX framing** — [`aspx::parse_aspx_framing`] implements
//!   `aspx_framing()` (Table 53, §4.2.12.4) end-to-end for all four
//!   interval classes (FIXFIX / FIXVAR / VARFIX / VARVAR) including
//!   the 1/2/3-bit `aspx_int_class` prefix code, the envelope-count
//!   derivation for FIXFIX (`1 << tmp_num_env` with
//!   `envbits = aspx_num_env_bits_fixfix + 1`), Note-1 1-vs-2-bit
//!   field widths driven by `num_aspx_timeslots`, the
//!   `aspx_tsg_ptr` sizing via `ceil(log2(num_env + 2))`, and the
//!   I-frame gate on `aspx_var_bord_left` for VARFIX / VARVAR.
//!   Returns the full [`aspx::AspxFraming`] (int_class, num_env,
//!   num_noise, freq_res vector, border fields, tsg_ptr). Wired into
//!   `asf::walk_ac4_substream` for both the mono ASPX and stereo
//!   ASPX I-frame paths: after `companding_control()` and the
//!   `mono_data()` / `stereo_data()` body the walker reads
//!   `aspx_xover_subband_offset` (3 bits) and then `aspx_framing(0)`
//!   (and, for stereo, `aspx_balance` + conditional
//!   `aspx_framing(1)`). `num_aspx_timeslots` for the Note-1 field
//!   width comes from the TOC's `frame_length` via
//!   [`aspx::num_aspx_timeslots`] (Table 189 × Table 192).
//! * **A-SPX delta direction** — [`aspx::parse_aspx_delta_dir`]
//!   implements `aspx_delta_dir(ch)` (Table 54, §4.2.12.5): one bit
//!   per signal envelope plus one bit per noise envelope. The per-
//!   channel `AspxDeltaDir` drives which `ASPX_HCB_*_{F0,DF,DT}`
//!   codebook the matching `aspx_ec_data()` path will pull from.
//! * **A-SPX HF generation / interleaved-waveform coding (mono)** —
//!   [`aspx::parse_aspx_hfgen_iwc_1ch`] implements Table 55
//!   (§4.2.12.6): per-subband-group `tna_mode`, `ah_present` +
//!   conditional `add_harmonic[]`, `fic_present` + conditional
//!   `fic_used_in_sfb[]`, and `tic_present` + conditional
//!   `tic_used_in_slot[]`. Takes `num_sbg_noise`,
//!   `num_sbg_sig_highres`, `num_aspx_timeslots` from the caller.
//! * **A-SPX HF generation / interleaved-waveform coding (stereo)** —
//!   [`aspx::parse_aspx_hfgen_iwc_2ch`] implements Table 56
//!   (§4.2.12.7). Adds per-channel `tna_mode[ch][]` (with
//!   `aspx_balance == 1` mirroring channel 0 into channel 1),
//!   per-channel `aspx_ah_left` / `_right` gates, `aspx_fic_present`
//!   + per-channel `fic_left` / `fic_right` gates, and the
//!   `aspx_tic_copy` / `aspx_tic_left` / `aspx_tic_right` TIC
//!   gating (including mirroring left-channel TIC into right when
//!   `tic_copy` is set).
//!
//! * **A-SPX Huffman infrastructure** — [`aspx::AspxHcb`] is a
//!   `(len[], cw[], cb_off)` codebook helper: the symbol decoder walks
//!   one bit at a time until a `(len == width, cw == code)` match
//!   lands, then returns `symbol_index - cb_off` as the delta. All 18
//!   Annex A.2 codebooks (Tables A.16..=A.33) are transcribed in
//!   [`aspx_huffman`] — six `(F0, DF, DT)` triples covering
//!   envelope-LEVEL / envelope-BALANCE @ 1.5 dB / 3 dB plus
//!   noise-LEVEL / noise-BALANCE. A [`aspx::HuffmanCodebookId`] enum
//!   + [`aspx::lookup_aspx_hcb`] resolve the
//!   `get_aspx_hcb(data_type, quant_mode, stereo_mode, hcb_type)`
//!   tuple from §5.7.6.3.4 Pseudocode 79.
//! * **A-SPX entropy coded data** — [`aspx::parse_aspx_ec_data`]
//!   implements `aspx_ec_data()` (Table 57, §4.2.12.8) on top of
//!   [`aspx::parse_aspx_huff_data`] (Table 58) — per-envelope loop
//!   that picks F0/DF/DT codebook per direction and returns a vector
//!   of [`aspx::AspxHuffEnv`]s. Caller supplies `num_sbg_*` counts via
//!   [`aspx::AspxSbgCounts`] pending the §5.7.6.3.1 master-freq-scale
//!   derivation. Plumbed into the substream walker through the ASPX
//!   framing + `aspx_delta_dir(ch)` read: `asf::walk_ac4_substream`
//!   now reads both `aspx_delta_dir(0)` (mono + stereo ASPX I-frames)
//!   and `aspx_delta_dir(1)` (stereo ASPX I-frames) on top of the
//!   existing `aspx_framing` consumption, alongside the effective
//!   per-channel `aspx_qmode_env` post FIXFIX + num_env=1 clamp.
//!
//! Known gaps (Unsupported or stubbed):
//!
//! * Short / grouped frames (`num_window_groups > 1`) — coefficient
//!   path only exercises the long-frame path today.
//! * A-SPX subband-group derivation (§5.7.6.3.1 `num_sbg_*`), QMF
//!   analysis / synthesis, HF regeneration, and envelope application —
//!   `aspx_ec_data()` is parseable but its outputs don't feed an
//!   actual bandwidth-extension pipeline yet.
//! * A-CPL (`acpl_config_*`, `acpl_data_*`).
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
//! mono and stereo SIMPLE/ASF streams. The stereo path covers both
//! the split-MDCT layout (two independent ASF spectra) and the joint
//! `b_enable_mdct_stereo_proc == 1` mode (shared sections +
//! scalefactors, two residuals, per-sfb `ms_used[]` inverse
//! L = M + S / R = M - S per §7.5). Everything else falls back to
//! silence with a correctly-shaped AudioFrame.

#![allow(dead_code)]

pub mod asf;
pub mod asf_data;
pub mod aspx;
pub mod aspx_huffman;
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
