# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A-CPL decoder wiring (round 17)**: `ASPX_ACPL_2` substreams now go
  through the §5.7.7 channel-pair synthesis end-to-end. The asf walker
  parses `aspx_data_1ch()` (Table 51) and `acpl_data_1ch()` (Table 61)
  for the ASPX_ACPL_2 path; `Ac4Decoder` runs `acpl_synth::run_acpl_1ch_pcm`
  (mono PCM → QMF analysis → §5.7.7.5 channel-pair → QMF synthesis × 2)
  to emit a real stereo signal in place of the duplicate-of-primary
  fallback. ASPX_ACPL_1's joint-MDCT body is still gated.

## [0.0.2](https://github.com/OxideAV/oxideav-ac4/compare/v0.0.1...v0.0.2) - 2026-04-25

### Other

- fix clippy 1.95 lints
- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- wire §5.7.6.4.3 noise + §5.7.6.4.4 tone into aspx_extend_pcm (round-9)
- end-to-end FFT probe for A-SPX noise + tone HF injection
- wire §5.7.6.4.2 per-envelope HF envelope adjustment (P90+P91+P95)
- land §5.7.6.4.3 noise + §5.7.6.4.4 tone generators
- land §5.7.4 QMF synthesis + A-SPX HF regen pipeline (round-7)
- add §5.7.3 QMF analysis scaffold — QWIN + single-slot transform
- derive §5.7.6.3.1 master-freq-scale and wire aspx_ec_data()
- wire aspx_delta_dir + effective qmode into substream walker
- transcribe all 18 A-SPX Huffman tables + aspx_ec_data() walker
- A-SPX Huffman scaffolding + Annex A.2 table metadata
- implement aspx_hfgen_iwc_2ch() per Table 56
- implement aspx_hfgen_iwc_1ch() per Table 55
- parse aspx_delta_dir() per-channel delta-direction bits
- wire aspx_framing into the ASF substream walker
- parse aspx_framing() end-to-end for all four interval classes
- parse aspx_config + companding_control sidecar
- stereo joint M/S test + refresh lib-level doc
- stereo CPE decoder test — different tones on L and R
- stereo CPE body decode (split + joint M/S) and per-channel IMDCT
- refresh lib-level doc for new coefficient pipeline
- wire ASF data path into decoder — real mono PCM output
- Huffman-driven ASF data parsers and dequantisation
- implement IMDCT + KBD window + overlap-add
- add sfb_offset tables for 48 kHz family (Annex B.4-B.7)
- transcribe ASF Huffman codebooks and add decoder helpers
- document sfb_offset tables (B.4/B.5/B.6) as next-up work
- parse asf_psy_info + Annex B num_sfb / Table 106 n_msfb_bits
- land ASF substream walker (ac4_substream + audio_data outer layers)
- switch workflows to master branch
