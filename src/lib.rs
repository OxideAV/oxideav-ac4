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
//!   plus extension escape, HSF extension, pre-virtualised flag, extra
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
//!   stereo ASPX_ACPL_{1,2} I-frame paths. For ASPX_ACPL_{1,2} it now
//!   also reads the trailing `acpl_config_1ch(PARTIAL)` /
//!   `acpl_config_1ch(FULL)` element (§4.2.13.1 Table 59) via
//!   [`acpl::parse_acpl_config_1ch`]. Exposes the parsed
//!   `AspxConfig` through
//!   [`decoder::Ac4Decoder::last_substream`]`.tools.aspx_config` and
//!   the A-CPL configs through `acpl_config_1ch_partial` /
//!   `acpl_config_1ch_full` on [`asf::SubstreamTools`].
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
//!   plus per-channel `fic_left` / `fic_right` gates, and the
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
//!   plus [`aspx::lookup_aspx_hcb`] resolve the
//!   `get_aspx_hcb(data_type, quant_mode, stereo_mode, hcb_type)`
//!   tuple from §5.7.6.3.4 Pseudocode 79.
//! * **A-SPX entropy coded data** — [`aspx::parse_aspx_ec_data`]
//!   implements `aspx_ec_data()` (Table 57, §4.2.12.8) on top of
//!   [`aspx::parse_aspx_huff_data`] (Table 58) — per-envelope loop
//!   that picks F0/DF/DT codebook per direction and returns a vector
//!   of [`aspx::AspxHuffEnv`]s.
//! * **A-SPX master freq-scale derivation** —
//!   [`aspx::derive_aspx_frequency_tables`] implements §5.7.6.3.1
//!   Pseudocodes 67, 68, 69 and 70: picks between
//!   [`aspx::ASPX_SBG_TEMPLATE_HIGHRES`] and
//!   [`aspx::ASPX_SBG_TEMPLATE_LOWRES`] by `aspx_master_freq_scale`,
//!   trims with `aspx_start_freq` / `aspx_stop_freq` into the master
//!   subband-group table, then applies `aspx_xover_subband_offset` to
//!   produce the high-res / low-res signal tables. The noise
//!   subband-group table follows Pseudocode 70's `max(1,
//!   floor(aspx_noise_sbg * log2(sbz/sbx) + 0.5))` count rule and is
//!   clamped to `num_sbg_noise <= 5`. Returns
//!   [`aspx::AspxFrequencyTables`] containing master / high-res /
//!   low-res / noise border tables plus `sba`, `sbz`, `sbx`,
//!   `num_sb_aspx` and an [`aspx::AspxSbgCounts`] ready to feed
//!   [`aspx::parse_aspx_ec_data`].
//! * **Full A-SPX data-path wiring** — `asf::walk_ac4_substream` now
//!   runs the whole `aspx_data_1ch()` / `aspx_data_2ch()` body on
//!   I-frame ASPX substreams: `aspx_xover_subband_offset` →
//!   `aspx_framing` (+ stereo `aspx_balance` / second framing) →
//!   `aspx_delta_dir` → derived [`aspx::AspxFrequencyTables`] →
//!   `aspx_hfgen_iwc_1ch()` / `aspx_hfgen_iwc_2ch()` →
//!   `aspx_ec_data()` SIGNAL and NOISE per channel. All parsed data
//!   lands on [`asf::SubstreamTools`] alongside the existing framing /
//!   delta-dir / qmode fields.
//!
//! * **QMF analysis + synthesis filter bank** — [`qmf::QWIN`] carries
//!   the 640-coefficient QMF prototype window from Annex D.3.
//!   [`qmf::qmf_analysis_slot`] + [`qmf::QmfAnalysisBank`] implement
//!   the §5.7.3.2 Pseudocode 65 forward transform (windowing +
//!   5-fold time-fold to vector u + 64-point complex modulation);
//!   [`qmf::qmf_synthesis_slot`] + [`qmf::QmfSynthesisBank`] implement
//!   the matching §5.7.4.2 Pseudocode 66 inverse transform (shifted
//!   1 280-sample `qsyn_filt` delay line + 128-point modulation +
//!   folded-tap 64-way tap sum). The analysis/synthesis pair achieves
//!   ~80 dB PSNR end-to-end roundtrip on sine and noise test signals
//!   (unit tests in [`qmf`]).
//! * **A-SPX HF regeneration scaffold** — [`aspx::derive_patch_tables`]
//!   implements §5.7.6.3.1.4 Pseudocode 71 (patch subband-group table
//!   derivation). [`aspx::hf_tile_copy`] implements a simplified
//!   §5.7.6.4.1.4 Pseudocode 89 high-band tile copy via the patch
//!   table (no chirp/alpha0/alpha1 tonal adjust). The full TNS body
//!   lives in the dedicated [`aspx_tns`] module — see below.
//!   [`aspx::apply_flat_envelope_gain`] is a one-gain scaffold kept as
//!   a fallback for the §5.7.6.4.2 HF envelope adjustment tool.
//!   Together with the QMF bank these form an end-to-end bandwidth-
//!   extension pipeline: PCM → QMF analysis → low-band truncate → HF
//!   tile-copy → QMF synthesis → non-silent PCM.
//! * **A-SPX TNS (chirp + α0 + α1)** — [`aspx_tns`] implements the
//!   full §5.7.6.4.1.2 / .1.3 / .1.4 complex-covariance Temporal
//!   Noise Shaping path: pre-flatten gain vector
//!   ([`aspx_tns::compute_preflat_gains`], Pseudocode 85), complex
//!   covariance matrix over `Q_low_ext`
//!   ([`aspx_tns::compute_covariance`], Pseudocode 86), α0 / α1 LPC
//!   coefficients with the EPSILON_INV slack and the |α|≥4 fallback
//!   ([`aspx_tns::compute_alphas`], Pseudocode 87), per-noise-subband-
//!   group chirp factors via the Table 195 `tabNewChirp` lookup with
//!   attack / decay smoothing ([`aspx_tns::chirp_factors`],
//!   Pseudocode 88), and the full HF signal creation that adds
//!   `chirp * α0 * Q_low[n-2]` + `chirp² * α1 * Q_low[n-4]` plus the
//!   optional pre-flatten divide ([`aspx_tns::hf_tile_tns`],
//!   Pseudocode 89). Per-channel state ([`aspx_tns::AspxTnsState`])
//!   carries `aspx_tna_mode_prev[]` / `prev_chirp_array[]` plus the
//!   tail of the previous interval's `Q_low` for the
//!   `ts_offset_hfadj = 4` look-back. The decoder auto-selects the
//!   TNS path when the parsed `aspx_hfgen_iwc_*` provides
//!   `aspx_tna_mode[ch][]` and the framing is FIXFIX; otherwise it
//!   falls back to the bare tile copy.
//! * **A-SPX HF envelope adjustment (per-envelope gain)** —
//!   [`aspx::AspxEnvelopeAdjuster`] implements §5.7.6.4.2 Pseudocodes
//!   90 + 91 + 95 (non-harmonic, non-limited path): delta-decode
//!   `aspx_data_sig` / `aspx_data_noise` (Pseudocodes 80 / 81) via
//!   [`aspx::delta_decode_sig`] / [`aspx::delta_decode_noise`],
//!   dequantize to `scf_sig_sbg` / `scf_noise_sbg` (Pseudocodes 82 / 83)
//!   via [`aspx::dequantize_sig_scf`] / [`aspx::dequantize_noise_scf`],
//!   estimate the actual HF envelope energy
//!   ([`aspx::estimate_envelope_energy`]), map subband-group scale
//!   factors onto QMF subbands ([`aspx::map_scf_to_qmf_subbands`]),
//!   then compute per-subband compensatory gains
//!   ([`aspx::compute_sig_gains`]) = `sqrt(scf_sig / ((1 + est) *
//!   (1 + scf_noise)))`. The per-envelope gains are applied via
//!   [`aspx::apply_envelope_gains`] using the FIXFIX Table-194
//!   `atsg_sig` / `atsg_noise` borders derived by
//!   [`aspx::derive_fixfix_atsg`]. The decoder auto-selects the
//!   per-envelope path when the substream parsed FIXFIX framing
//!   plus matching envelope deltas, and falls back to
//!   `apply_flat_envelope_gain(0.5)` otherwise.
//! * **ASPX decoder wiring** — [`decoder::Ac4Decoder`] now routes
//!   ASPX substreams through the extension pipeline: when an I-frame
//!   ASPX substream produces derived `aspx_frequency_tables`, the
//!   decoder takes the IMDCT low-band PCM, runs QMF analysis →
//!   tile-copy HF regen → per-envelope gain (or flat-gain fallback)
//!   → QMF synthesis, and emits bandwidth-extended PCM instead of
//!   silence.
//!
//! * **DRC metadata parser** — [`drc::parse_drc_frame`] walks the
//!   `drc_frame()` element (§4.2.14.5 Table 70) end-to-end:
//!   `b_drc_present` gate, optional I-frame `drc_config()` (§4.2.14.6
//!   Table 71) including up to eight `drc_decoder_mode_config()` blocks
//!   with all three branches (`drc_repeat_profile_flag`,
//!   `drc_default_profile_flag`, explicit `drc_compression_curve()`),
//!   the full `drc_compression_curve()` with optional boost / cut
//!   sections and per-mode time-constant block (§4.2.14.8 Table 73),
//!   and the per-frame `drc_data()` payload (§4.2.14.9 Table 74) that
//!   pulls one `drc_gains()` entry per gainset mode.
//!   [`drc::parse_drc_gains`] decodes the seven-bit `drc_gain_val`
//!   seed plus all `(ch, band, sf)` deltas through the Annex A.5
//!   `DRC_HCB` Huffman codebook (`huff_decode_diff` per §4.3.10.8.3),
//!   with the per-band / per-channel `ref_drc_gain` reset semantics
//!   from Table 75 honoured. The Annex A.5 codebook itself
//!   ([`drc_huffman::DRC_HCB_LEN`] / [`drc_huffman::DRC_HCB_CW`]) is
//!   transcribed verbatim from the ETSI accompaniment file
//!   `ts_10319001v010401p0-tables.c` and is verified by a Kraft-sum
//!   = 1 unit test (complete prefix code) plus an explicit prefix-
//!   code check.
//!
//! Known gaps (Unsupported or stubbed):
//!
//! * Short / grouped frames (`num_window_groups > 1`) — coefficient
//!   path only exercises the long-frame path today.
//! * Remaining §5.7.6.4 A-SPX HF regeneration — non-FIXFIX interval
//!   classes (FIXVAR / VARFIX / VARVAR) still fall back to the
//!   flat-gain scaffold; the limiter (§5.7.6.4.2.2) and TNS
//!   (§5.7.6.4.1) paths only run on FIXFIX framing today. The TNS
//!   pipeline is fully implemented in [`aspx_tns`] but per-channel
//!   `master_reset` semantics (resetting `prev_chirp_array` / the
//!   `Q_low_prev` history) when a new substream starts mid-stream
//!   isn't surfaced through the decoder API yet.
//! * A-CPL data-path wiring — the [`acpl`] module fully implements
//!   `acpl_config_1ch` / `acpl_config_2ch` / `acpl_framing_data` /
//!   `acpl_huff_data` / `acpl_ec_data` / `acpl_data_1ch` /
//!   `acpl_data_2ch` (§4.2.13 Tables 59..65) over all 24 §A.3
//!   Huffman codebooks ([`acpl_huffman`], Tables A.34..A.57), and the
//!   outer `audio_data()` walker now reads `acpl_config_1ch` for the
//!   stereo ASPX_ACPL_{1,2} paths. Wiring the per-frame
//!   `acpl_data_1ch()` / `acpl_data_2ch()` payloads into the QMF
//!   bandwidth-extension pipeline (and the `sb_to_pb` Table 197
//!   mapping that drives `start_band`) still needs to be done.
//! * Speech Spectral Frontend (SSF) arithmetic-coded path.
//! * Spectral noise fill synthesis — `asf_snf_data()` parses the
//!   Huffman-coded indices but doesn't inject shaped noise into
//!   zero bands yet.
//! * Per-substream `metadata()` payload parsing — the outer
//!   `metadata()` walker (§4.2.14.1) plus `basic_metadata()`,
//!   `extended_metadata()`, `further_loudness_info()`, dialog
//!   enhancement, and `emdf_payloads_substream()` are still skipped
//!   en bloc via the substream byte size. The `drc_frame()` parser
//!   ([`drc`]) is in place and can be invoked directly once the outer
//!   walker is wired.
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

pub mod acpl;
pub mod acpl_huffman;
pub mod asf;
pub mod asf_data;
pub mod aspx;
pub mod aspx_huffman;
pub mod aspx_limiter;
pub mod aspx_noise;
pub mod aspx_tns;
pub mod aspx_tone;
pub mod decoder;
pub mod drc;
pub mod drc_huffman;
pub mod huffman;
pub mod mdct;
pub mod qmf;
pub mod sfb_offset;
pub mod sync;
pub mod tables;
pub mod toc;

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag, Result};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder};

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
