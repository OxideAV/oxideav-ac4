# oxideav-ac4

[![CI](https://github.com/OxideAV/oxideav-ac4/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-ac4/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-ac4.svg)](https://crates.io/crates/oxideav-ac4) [![docs.rs](https://docs.rs/oxideav-ac4/badge.svg)](https://docs.rs/oxideav-ac4) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **Dolby AC-4** audio codec — decoder and encoder per ETSI
TS 103 190-1 V1.4.1. Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

AC-4 is a complex, hierarchical codec — multiple presentations, nested
substream descriptors, ASF / A-SPX / SSF coefficient streams, A-CPL
channel-pair coupling, and an EMDF metadata sidecar. This crate parses
the full framing and decodes channel-based streams to PCM; an encoder
covers the channel-based layouts.

### Bitstream framing and TOC

- **Sync framing** (`sync`) — `0xAC40` plain / `0xAC41` CRC-protected,
  16-bit `frame_size()` with 24-bit escape. The Annex G.4.2 `crc_word`
  is verified on every 0xAC41 frame (protected payload = `frame_size` +
  `raw_ac4_frame`); the decoder rejects mismatching frames, and
  `wrap_sync_frame` emits both framing forms.
- **Table of contents** (`toc`) — the full `ac4_toc()` walker:
  bitstream_version, sequence_counter, fs_index, frame_rate_index,
  `b_iframe_global`, payload_base, per-presentation
  `ac4_presentation_info()` (single / multi-substream, configs 0..=5
  plus extension escape, HSF extension, pre-virtualised flag, extra
  EMDF substreams), per-substream `ac4_substream_info()` (channel-mode
  prefix decoder, sf_multiplier, bitrate_indicator, content_type with
  language tag, b_iframe), the substream index table, and the
  `variable_bits(n)` codec. Surfaced on a parsed `Ac4FrameInfo`
  (including the part-2 A-JOC / object / OAMD substream descriptors).
  The v2 `b_pres_ndot` / `b_audio_ndot` / `b_oamd_ndot` flags follow
  the §6.3.2.11.2 "no dependency over time" polarity — ndot **is** the
  I-frame flag (a long-standing inversion on both the parse and write
  sides was fixed in r411).
- **Presentation substream** (`pres_data`) — the complete part-2
  `ac4_presentation_substream()` (§6.2.2.3): alternative-presentation
  names + targets with per-substream activation maps, the
  additional-data envelope, presentation dialnorm +
  `further_loudness_info(1, 1)`, the `drc_metadata_size` envelope
  around `drc_frame(b_pres_ndot)`, substream-group gains, the
  associated-audio scale block, and the §6.2.9 presentation data:
  `custom_dmx_data()` (bs_ch_config decision tree, `cdmx_parameters()`
  with all six `tool_*()` elements) and `loud_corr()` (the full gate
  ladder incl. the object corrections) — every parser with an exact
  writer inverse.
- **Stereo downmix coefficients** (`dmx_coeff`) — `stereo_dmx_coeff()`
  (the factored-out `custom_dmx_data()` block, also invoked from
  `bed_render_info()`) with the TS 103 190-1 §4.3.12.2 code → gain
  mappings (Tables 149/149a quarter-power-of-two steps, the LFE
  `5,5 − code` dB rule, dmx loudness corrections, Table 150 preferred
  method).
- **OAMD substream** — the standalone `oamd_substream()` (§6.2.2.4):
  optional `oamd_common_data()` / `oamd_timing_data()` and the
  `b_alternative == 0` `oamd_dyndata_multi()` walk, with
  `oamd_substream_info()` surfaced from the TOC.

### Decoder

`Ac4Decoder` accepts a sync-wrapped packet or a bare MP4-style
`raw_ac4_frame` payload, parses the TOC, and decodes the substreams to
an S16 `AudioFrame`:

- **ASF coefficient pipeline** — `asf_section_data()`,
  `asf_spectral_data()` (HCB 1..11 + the codebook-11 escape),
  `asf_scalefac_data()`, and `asf_snf_data()` for mono, stereo (split +
  joint M/S MDCT), and the multichannel layouts (Tables 26–29), with
  long-frame and grouped short-frame window handling. Dequantise +
  scale (`rec_spec = sign(q)·|q|^(4/3)`) feeds the reference KBD-window
  IMDCT (`mdct`) with overlap-add.
- **A-SPX** bandwidth extension — `aspx_config()` / `companding_control`
  parse, the FIXFIX / FIXVAR / VARFIX / VARVAR ATSG border derivation,
  envelope / noise / tone payload decode, QMF analysis/synthesis, the
  **§5.7.6.3.5 balance stereo joint decode** (an `aspx_data_2ch()`
  pair with `aspx_balance = 1` decodes as the spec's jointly coded
  sum/pan pair — Pseudocode 84 with `PAN_OFFSET = 12` plus the
  Pseudocode 80/81 `delta = 2` accumulation on the balance channel,
  wired on every 2ch consumer: stereo CPE, the 5_X trailers, the ICE
  payload rosters, the 22.2 pairs, and the A-JOC A-SPX downmix; every
  live `aspx_balance = 1` writer emits the exact Pseudocode 84
  inverse, so encoded pans survive the joint decode — measured
  ≈ 20 dB preserved on a hard-panned ACPL_3 chain), and the
  §5.7.6.4.1.2–4 temporal-noise-shaping chirp / order-2 LPC inverse
  filtering driven by `aspx_tna_mode`. The encoder now **selects** a real
  per-noise-subband-group `aspx_tna_mode` (§4.3.10.6.1 None / Light /
  Moderate / Heavy) from the carrier's QMF low band — a level-independent
  predictor-strength measure (`|alpha0|² + |alpha1|²` from the decoder's
  own `compute_covariance` / `compute_alphas`, aggregated per noise group
  via the Pseudocode-89 high-band walk) — and wires it into the live 5_X
  ASPX_ACPL_3 frame path (`aspx_tna_select` + the `_tna` body writers),
  replacing the all-zero inverse-filtering scaffold.
- **A-CPL** channel-pair coupling — ASPX_ACPL_1 / _2 (Pseudocode 117)
  and ASPX_ACPL_3 (Pseudocode 118) synthesis producing 5-channel
  L/R/C/Ls/Rs PCM, plus the 7.X (7.0 / 7.1) walker.
- **A-JOC** (Advanced Joint Object Coding, TS 103 190-2 §5.7 + §6.2.5 /
  §6.3.6) — the `ajoc` module lands the bit-exact parameter-processing
  core: the §5.7.3.1 Table 42 QMF-subband → parameter-band mapping
  (`sb_to_pb`, all eight `ajoc_num_bands` configs) + Table 100 band-code
  resolution, the §5.7.3.2 Table 43 differential decoder (DIFF_FREQ /
  DIFF_TIME / sparse-absent), the §5.7.3.3 Tables 44-47 uniform
  dequantizers (validated against the documented dry ±5,0048828 / wet
  ±2,001953125 endpoints), the §5.7.3.4 Table 48 linear-ramp time
  interpolator, and the §5.7.3.6 Table 49 dry + wet matrix
  reconstruction (`reconstruct`). The **full §5.7.3.6 spatial
  reconstruction** (`ajoc_reconstruct`) closes the A-JOC decode chain
  end-to-end: it accumulates the §5.7.3.6.2 decorrelation-input
  pre-matrix `D[de][ch] = Σ_o |wet[o][de]|·dry[o][ch]`
  (`pre_matrix_param`), walks the `(ts, sb)` QMF grid interpolating the
  dry / wet / pre tracks (per-track interpolators carried across frames
  in `AjocReconState`), forms the decorrelator inputs `u = pre · x`,
  decorrelates them with the §5.7.3.5 cyclic-0,2,1 decorrelator bank
  (reusing the part-1 §5.7.7.4.2 `InputSignalModifier`), and sums the dry
  (`x · mtx_dry`) + wet (`y · mtx_wet`) contributions into the
  reconstructed output objects `z[ts][sb][o]`. The Huffman-independent
  §6.2.5 config-layer parsers (`ajoc_ctrl_info`, `ajoc_data_point_info`,
  `ajoc_bed_info`, `ajoc_dmx_de_data` with the Table 106
  `de_dlg_dmx_coeff` prefix code) walk the side-information. The
  **§6.2.5.5 `ajoc_huff_data()` codeword layer** (`ajoc_huffman`)
  decodes over the twelve Annex A.1.1 `AJOC_HCB_*` codebooks
  (transcribed from the ETSI Part 2 electronic-attachment table file,
  verified in-tree as complete prefix codes) with the §6.3.6.5.2
  Table 104 `get_ajoc_hcb()` selection, and `ajoc_data::decode_ajoc()`
  joins the halves: a complete §6.2.5.1 `ajoc()` element goes ctrl-info
  → Huffman rows → Table 43 differential decode (cross-frame
  `mtx_*_q_prev` state) → dequantized matrices, straight into
  `ajoc_reconstruct`. The **A-JOC parameter encoder** (`encoder_ajoc`)
  quantises real-valued dry / wet matrices (Tables 44-47 inverses),
  prices every matrix row with the real codeword lengths to pick
  DIFF_FREQ vs DIFF_TIME, and emits GOP-chained `ajoc()` elements whose
  decode is exact on the quantised grid (verified over I/P/P/P GOPs
  with measured P-frame row-bit savings). The chain is now **wired
  end-to-end into the frame decoder**: the v2 TOC parses
  `ac4_substream_info_ajoc()` / `ac4_substream_info_obj()`
  (§6.2.1.9-11, `bed_dyn_obj_assignment` with all five
  position-signalling forms and Table 83-86 static-object counts), the
  `ajoc_substream` module walks `audio_data_ajoc()` (§6.2.3.4) with its
  `var_channel_element()` downmix (§6.2.4.4 — SIMPLE + A-SPX modes,
  all odd-count tails) and OAMD side information, and
  `AjocSubstreamDecoder` drives IMDCT → QMF analysis → Table 49
  reconstruction → per-object QMF synthesis with all cross-frame state
  persistent. `Ac4Decoder` routes v2 A-JOC frames to this chain
  automatically, and `encode_ajoc_raw_frame` emits complete matching
  frames (decoder-level packet-to-PCM tests pin per-object energy and
  a < 0,5 %-of-peak settled reconstruction error). The **LFE channel**
  (coded directly in the downmix as a Table 21 `mono_data(1)` body,
  bypassing the spatial reconstruction) is decoded to PCM and emitted
  on the leading LFE output slot, and the §6.2.2.2 post-audio
  `metadata(…, sus_ver = 1)` element is parsed on the route (with
  `de_config()` carried across frames; per §6.2.7.2 an object
  substream opens no channel-gated `basic_metadata` field and carries
  no stereo-dmx block — the bed/custom downmix data is authoritative).
  All three §6.2.3.4 downmix forms reach object PCM: the dynamic
  SIMPLE form, the dynamic **A-SPX** form (each downmix channel is
  bandwidth-extended in the QMF domain from its captured
  `aspx_data_2ch()` / `aspx_data_1ch()` payload before the spatial
  reconstruction, with the I-frame `aspx_config` + xover sticky
  across P-frames), and the **`b_static_dmx`** 5.X core (the
  `audio_data_chan(5.0/5.1)` SIMPLE coding configs feed the
  reconstruction in the Table 180 `[L, R, C, Ls, Rs]` order; the 5.1
  core's LFE lands on the leading output slot). The write side gains
  the matching `write_audio_data_ajoc_static` /
  `write_audio_data_ajoc_aspx` bodies and
  `encode_ajoc_raw_frame_static` / `encode_ajoc_raw_frame_aspx`
  full-frame writers.
- **Immersive channel element** (TS 103 190-2 §6.2.4.1-2 — channel
  modes 7.0.4 / 7.1.4 / 9.0.4 / 9.1.4, Table 78) — the `ice` module
  parses the full `immersive_channel_element()` family: the Table 95
  `immersive_codec_mode` prefix code, `immers_cfg()` on the shared
  I-frame-sticky config slots, all four Table 97 core groupings
  (1+2+2 incl. both `2ch_mode` forms / 3+2 / 1+4 / 5) with the
  §5.2.3.2 Table 19 track assignment, the 7CH_STATIC
  `b_use_sap_add_ch` chparam pair + additional pair, the
  mode-dependent A-SPX payload roster (captured per element),
  `ajcc_data(b_5fronts)`, the S-CPL pair/chparam section, and the
  4/6-element `acpl_data_1ch()` section — with P-frame sticky-config
  support. `Ac4Decoder` routes the immersive channel modes and
  synthesises **all five Table 95 codec modes** to PCM in full
  decoding: **SCPL** (§5.3.3.1 Table 23 time-domain matrix),
  **ASPX_SCPL** (Table 23 with `c_gain = m_gain = 1`, per-channel
  A-SPX over the Table 8 channel grouping, §4.8.3.11.3 Table 10/11
  output gains), **ASPX_ACPL_1 / ASPX_ACPL_2** (§5.5.2 Table 27 —
  four/six parallel ACplModules with D0/D0/D1/D1/D2/D2 decorrelators
  over the Table 26 mapping; ACPL_1 runs the PARTIAL config's
  `acpl_qmf_band` M/S split on the S-CPL-section residual tracks,
  ACPL_2 runs fully parametric with the coded F/G tracks on the
  Tfl/Tfr carrier positions; per-module cross-frame differential
  state), and **ASPX_AJCC** (§5.6.3.5.2: core IMDCT → per-channel QMF
  A-SPX extension → §4.8.3.10.3 companding on L/R/C/Ls/Rs → A-JCC
  full decode → 11/13 outputs; LFE first). The §5.2.3.2 SAP mixing
  (steps 3-6) runs on the track spectra for the SCPL / ASPX_SCPL /
  ASPX_ACPL_1 modes — the `b_use_sap_add_ch` quartets mix (D, F) /
  (E, G) and the S-CPL-section full-SAP `a'_j` gains drive the
  Table 20 additive rows. Writers emit complete v2 frames for all
  five codec modes (`write_ice_body_scpl[_with_sap]` /
  `write_ice_body_aspx_scpl` / `write_ice_body_acpl` /
  `write_ice_body_ajcc[_with_companding]` / `encode_ice_raw_frame`).
- **22.2 channel element** (TS 103 190-2 §6.2.4.3 + §5.2.4 — Table 78
  channel mode 15, 24 channels) — `22_2_channel_element()` parses and
  decodes in both Table 98 codec modes: the two LFE `mono_data(1)`
  bodies route directly through the IMDCT, the eleven
  `two_channel_data()` pairs map to channels per Table 21, and the
  A-SPX mode bandwidth-extends every pair from its `aspx_data_2ch()`
  payload (I-frame-sticky config). Output is 24-channel PCM (both
  LFEs first, then the Table 21 order); `write_22_2_body` +
  `encode_22_2_raw_frame` emit complete frames for both modes.
- **OAMD** (object audio metadata, TS 103 190-2 §6.2.8 + §6.3.9) —
  the `oamd` module parses and re-emits (exact writer inverses)
  `oamd_timing_data()`, `object_info_block()` with basic / render info
  and the full property groups, `add_per_object_md()` /
  `ext_prec_pos()` extended-precision refinements,
  `oamd_dyndata_single()` (incl. the `b_alternative` data-set tail) /
  `oamd_dyndata_multi()`, and `oamd_common_data()` (trim,
  bed-render-info, tool elements) with all size-announced envelopes
  reconciled.
- **A-JCC** (Advanced Joint Channel Coding, TS 103 190-2 §5.6 + §6.2.6 /
  §6.3.7) — the complete parameter + synthesis chain for both layouts
  (`b_5fronts` and the 5-channel core layout): the twelve Annex A.1.2
  `AJCC_HCB_*` codebooks plus the §6.3.7.3.2 Table 116 `get_ajcc_hcb()`
  selection (ALPHA / BETA delegate to the Part 1 A-CPL books),
  `ajcc_huff_data()` / `ajcc_framing_data()` / `ajced()` / `ajcc_data()`
  parsers with exact writer inverses, the §5.6.3.2 Table 29 differential
  decode (plain running sums, `ajcc_<SET>_q_prev` chained across
  parameter sets and frames) and Table 30/31 dry (`q·Δ−0,6`) / wet
  (`q·Δ−2,0`) dequantizers with alpha / beta through the Part 1
  Tables 202-205 `ibeta`-coupled machinery. `ajcc_synth` lands the
  §5.6.3.3 Table 32 smooth / steep interpolator (Table 33 tails), the
  Table 36 core-mode crossfade, all four reconstruction modules
  (Tables 37/38 full-mode, Tables 40/41 core-mode) and both drivers:
  `ajcc_full_decode()` (Table 35 — 13/11 output channels, per-instance
  D0/D1/D2 decorrelators + Part 1 transient ducking, √2 output gains)
  and `ajcc_core_decode()` (Table 39 — 7 channels), with a
  bitstream-to-QMF end-to-end GOP test through `AjccOwnedParams`.
- **P-frames (`b_iframe = 0`)** — full inter-frame decode per §4.2.6.x:
  a per-substream sticky-config state (`asf::StickyConfig`) carries the
  I-frame-gated `aspx_config()` / `acpl_config_*()` elements and the
  Tables 51/52 `aspx_xover_subband_offset` across frames, so non-I-frame
  substreams parse and synthesise their full A-SPX + A-CPL layer on the
  mono / stereo / 5_X / 7_X paths. The envelope delta decode is the
  **full** §5.7.6.3.4 Pseudocode 80/81 — per-envelope frequency
  resolution with the `high2low` / `low2high` subband-group index maps
  and the cross-interval `qscf_*_prev` reference carried per channel
  (`aspx::AspxEnvPrev`); the A-CPL DIFF_TIME chain accumulates across
  frames via the per-element Pseudocode-121 `AcplDiffState` rows.
- **SSF** front-end — the §5.2.8 arithmetic decoder + Annex C scalar
  inventory + 37 prediction-coefficient matrices, the bitstream walker,
  the §5.2.3–5.2.7 PCM synthesis chain, and §5.2.5.2.2 heuristic
  scaling, with envelope / dither / noise RNG state threaded across
  granules.
- **Metadata** — the `metadata()` walker, the EMDF payloads substream
  (`emdf_payloads_substream()` Table 18 + `emdf_payload_config()`
  Table 79, capturing each payload's bytes verbatim), DRC gain
  application (`drc_raw_to_linear` + dialnorm correction applied to
  planar PCM), and the DE (dialogue enhancement) walker.
- **Metadata write-side (encoder symmetry)** — every metadata parser now
  has a bit-exact inverse, so a decoded `Metadata` round-trips back to a
  parse-equivalent bitstream. `write_metadata` (Table 66) drives
  `write_basic_metadata` + `write_further_loudness_info` (Table 67/68,
  incl. the `prgmbndy` unary code and the loudness-version escape),
  `write_extended_metadata` (Table 69, with an explicit
  `b_channels_classifier` flag for layouts that carry no classifiable
  channels), `write_drc_frame` / `write_drc_data` / `write_drc_gains`
  (Table 70/74/75, re-deriving the DRC_HCB gain deltas via
  `write_drc_huff_diff`) and `write_drc_config` / `write_drc_compression_curve`
  (Table 71/72/73), `write_dialog_enhancement` / `write_de_config` /
  `write_de_data` (Table 76/77/78, re-encoding `de_par` through the
  Annex A.4 `write_de_abs_huffman` / `write_de_diff_huffman` helpers),
  and `write_emdf_payloads_substream` / `write_emdf_payload_config`
  (Table 18/79), all over the canonical `write_variable_bits` codec
  (proven bit-exact against the §4.2.2 decoder for every `u32`).

### Encoder

`Ac4ImsEncoder` emits IMS v2 frames for the channel-based layouts:

- Mono / stereo (SIMPLE/ASF split-MDCT and joint M/S CPE).
- 5.0 / 5.1 and 7.0 / 7.1 SIMPLE/Cfg3Five (per-channel forward MDCT +
  DP-optimal sectioning + HCB selection + SNF, with an LFE element for
  the `.1` layouts).
- 5.X / 7.X ASPX_ACPL_1 / _2 / _3 paths with real per-parameter-band
  α / β extraction from the input channels' MDCT energy / correlation.
- **P-frames (`b_iframe = 0`)** on every live A-SPX path (5_X ACPL_3
  single + multi-envelope, 5_X / 7_X ACPL_2, 7_X / 5_X-SAP ACPL_1, 7.0
  pure-ASPX): setting `b_iframe_global = false` emits the correct
  Table 25 / Table 33 P-frame body — data elements present, configs +
  per-element xover omitted — and signals it through the v0/v1/v2 TOC
  (`b_iframe` / `b_pres_ndot` / `b_audio_ndot`). On the flagship 5_X
  ACPL_3 path the encoder additionally keeps the previous frame's
  envelope + parameter rows and switches each of the four A-SPX
  envelopes (L/R × SIGNAL/NOISE) to **TIME-direction DPCM**
  (Pseudocodes 80/81) and each of the 11 A-CPL Table 62 elements to
  **DIFF_TIME** (Table 65) whenever strictly cheaper — a stationary
  `aspx_data_2ch()` element shrinks 302 → 55 bits (−81%). Chain
  consistency over I+5×P GOPs is pinned against an all-I reference.
- 5.X ASPX_ACPL_3 with a **real ASPX SIGNAL / NOISE envelope** on the
  L / R carriers (`encode_frame_pcm_5_{0,1}_acpl3_real_aspx`): the
  encoder QMF-analyses the input PCM, aggregates the HF energy across the
  A-SPX subband-group borders (Pseudocodes 90/91), quantises + FREQ-DPCM
  packs it (Pseudocodes 80–83), and emits a real-envelope
  `aspx_data_2ch()` instead of the minimum-bit-cost scaffold.
- **Immersive channel element (ICE) synthesis routes** (TS 103 190-2
  §6.2.4.1, r419): `encode_frame_pcm_7_{0,1}_4_ice_aspx_scpl` (exact
  Table 23 + §4.8.3.11.3 matrix inverse to the eleven SMP tracks +
  real per-Table-8-group A-SPX synthesis; ≤ 6,2 % settled relative
  RMS on all 11 channels, LFE 3,6 %, regenerated HF within 3 dB) and
  `encode_frame_pcm_7_{0,1}_4_ice_acpl{1,2}` (§5.5.2 Table 27 module
  mid carriers with per-band `(α, β)` from the pair mid/side
  statistics; ACPL_1 codes the sides below `acpl_qmf_band` as exact
  M/S residual tracks — ≤ 5,2 % settled RMS; ACPL_2 dry-exact on
  correlated pairs with β decorrelator fill on independent content).
  The A-SPX synthesis stack behind them: streaming §5.7.6.2/§5.7.6.5
  QMF banks carried across frames, integer-PCM scale anchors
  (`ASPX_QMF_PCM_SCALE` — the Pseudocode 82/95 absolute anchors sit
  at their intended magnitudes, so HF-silence is representable),
  ratio-coded NOISE envelopes (Pseudocode 94 semantics) and a
  patch-delivery model (Pseudocode 71 tile map + Pseudocode 86-89
  TNS whitening replicated on the encoder's own low band) driving
  both the noise ratio and an inverse-delivery SIGNAL boost.
- **22.2 encode** (`encode_frame_pcm_22_2_{simple,aspx}`, §6.2.4.3):
  both Table 98 codec modes from PCM — Simple ≤ 6,0 % settled RMS on
  all 24 channels, A-SPX with real per-pair synthesis rows.
- **Encoder companding decision** (`select_compand_on_from_qmf`):
  §5.7.5.2 level-crest transient detection feeding the immersive
  `companding_control(5)` writers.
- **9.0.4 / 9.1.4 (`b_5fronts`) ICE encode arms** (r440): the full
  encode-side Table 23 `b_5fronts` matrix inverse
  (`A = (L + Lscr)/2`, `L″ = (L − Lscr)/2` on the fixed ×2 front rows
  plus the shared half-sum/half-difference surround / top rows) drives
  `encode_frame_pcm_9_{0,1}_4_ice_aspx_scpl` (6× `aspx_data_2ch()` +
  1× `aspx_data_1ch()` real-synthesis payloads on the `b_5fronts`
  Table 8 groups, three S-CPL pairs; ≤ 6,3 % settled relative RMS on
  all 13 channels, LFE 3,6 %) and
  `encode_frame_pcm_9_{0,1}_4_ice_acpl{1,2}` (the six-module §5.5.2
  Table 27 roster — the two front modules `(L, Lscr)` / `(R, Rscr)`
  ride the A / B track positions plain, ACPL_1 codes the front sides
  on the third S-CPL residual pair `L″ / M″`; ACPL_1 M/S band ≤ 7,1 %
  settled RMS on all 13 channels).
- **A-JCC parameter extractor + ASPX_AJCC encode from PCM** (r440,
  `encoder_ajcc` + `encode_frame_pcm_{7,9}_{0,1}_4_ice_ajcc`): exact
  Table 30/31 dry / wet quantiser inverses plus alpha (raw F0 lane) /
  beta lanes; per-parameter-band least-squares dry projections
  `⟨z, x⟩/⟨x, x⟩` against the exact per-module output sums (the
  Table 35/37/38 dry gains sum to 1 and the wet rows cancel, so
  `x = Σ outputs` is the natural core), wet gains filling the
  projection residual through the decorrelator model (`wet3 = 0` —
  an informative encoder choice), and alpha / beta from the pair
  mid/side statistics. `build_ajcc_data()` assembles smooth-framing
  single-set elements — FREQ rows on I-frames, per-SET FREQ-vs-TIME
  rows priced by the real Annex A.1.2 codeword lengths on P-frames —
  in decoder lockstep via an encoder-held `AjccState` mirror. The
  encode arms derive the five-channel core as the per-module output
  sums ÷ (2 + 1/√2) for **both** layouts; per-band-separated content
  reconstructs at 0,90..1,08 settled energy ratios on all 11 / 13
  channels and the emitted `ajcc_data()` differential-decodes to
  exactly the extractor's quantised grid across I+P GOPs.
- **SAP encode decisions** (r440, §5.2.3.2 steps 3-6):
  `encode_frame_pcm_{7,9}_{0,1}_4_ice_scpl_sap` make both immersive
  SAP decisions automatically — the step-3/4 `b_use_sap_add_ch`
  quartets M/S + prediction code `(D, F)` / `(E, G)` per sfb pair
  (`wire = (mid, side − g·mid)`, the exact Pseudocode 59 quartet
  inverse), and the step-5/6 full-SAP `chparam_info()` elements
  predict each S-CPL track from its Table 20 source carrier with
  per-pair least-squares gains on the `alpha_q · 0,1` grid (the wire
  track carries the residual). Correlated vertical content codes
  measurably smaller than an identity encode and decodes back within
  the ASF quantisation floor.

## Not yet supported

- **Immersive remainders** — core decoding mode (the reduced
  7-channel operating point of §5.3.3.2 / §5.6.3.5.3 / §4.8.3.11.2;
  the A-JCC core-decode reconstruction itself is implemented, but no
  decoder API selects the mode yet); the A-SPX / A-CPL codec modes of
  the A-JOC `b_static_dmx` core parse but their carrier synthesis
  into the object path is pending (needs the 5_X carrier pipeline
  shared into the object decoder). Table 8's ASPX_ACPL_1 rows list
  more A-SPX groups than the §6.2.4.1 syntax carries payloads for —
  the extension covers the four transmitted payloads and the
  S-CPL-section tracks pass through unextended (see the `ice` module
  notes).
- Remaining TS 103 190-2 multi-stream / immersive / object-based (IFM)
  extensions beyond the parsed presentation / OAMD / object substream
  surfaces.
- P-frame refinements: the sticky state carries **one** xover offset per
  substream, so P-frames assume the I-frame used a single
  `aspx_xover_subband_offset` across all A-SPX elements of the element
  (always true for this encoder; per-element sticky xovers would need a
  per-trailer table). Multi-envelope (`num_env > 1`) P-frame bodies emit
  FREQ-direction envelopes only (the encoder clears its cross-frame rows
  after a multi-envelope frame rather than tracking the last envelope).
  Cross-frame TIME/DIFF_TIME emission is wired on the 5_X ACPL_3 path;
  the other live paths emit correct P-frame bodies with FREQ rows.
- Per-`emdf_payload_id` semantic interpretation of EMDF payload bodies
  (captured as raw bytes).
- Some advanced A-CPL parameters (β3 / γ on certain encoder paths)
  remain scaffolded at minimum-bit-cost defaults.
- **A-SPX `aspx_hfgen_iwc` sub-fields:** the live 5_X ASPX_ACPL_3, the
  5_X / 7_X ASPX_ACPL_2, the **7.0 pure-ASPX**, and the **7_X
  ASPX_ACPL_1** paths now emit a real `aspx_tna_mode` (inverse filtering)
  on every A-SPX carrier — each body derives an independent
  `aspx_tna_mode` per carrier from that carrier's own QMF low band (front
  pair from L, surround pair from Ls, centre from C, and the 7.0
  pure-ASPX back pair from Lb). The 7_X ASPX_ACPL_1 path additionally now
  emits **real** per-sbg SIGNAL/NOISE ASPX envelopes on all three carriers
  (replacing the round-118 `write_aspx_data_*_minimal` scaffold). Every
  live A-SPX path now also emits a **real `aspx_add_harmonic`** decision:
  the `aspx_ah_select` module measures each carrier's per-high-res-signal-
  subband-group HF QMF **spectral crest** (the group's loudest subband
  energy ÷ its mean per-subband energy) and requests a restored missing
  harmonic (§4.2.12.6) where a dominant tonal partial is present (the
  decoder places the §5.7.6.4.2.1 Pseudocode 92 sinusoid at the group's
  `sb_mid`). This is wired per-channel into the live 5_X ASPX_ACPL_3
  (single- **and** multi-envelope), 5_X / 7_X ASPX_ACPL_2 (single- and
  centre-multi-envelope), 7.0 pure-ASPX, and 7_X ASPX_ACPL_1 paths via new
  `write_aspx_data_{1,2}ch_real_envelope_tna_ah` +
  `write_aspx_data_{1,2}ch_multi_envelope_tna_ah` writers and an
  `extract_aspx_add_harmonic` per-carrier analysis. The decoder fully
  consumes `aspx_add_harmonic` (§5.7.6.4.4 tone generator → HF QMF
  injection), so the decision changes the **decoded PCM**, not just the
  wire bytes. Every live A-SPX path additionally emits a **real
  `aspx_preflat`** decision (Table 121): the `aspx_preflat_select` module
  reuses the decoder's own §5.7.6.4.1.2 Pseudocode 85 gain fit
  (`compute_preflat_gains`) over the carrier's QMF low band — the
  HF-generation source range — and signals spectral pre-flattening when the
  fitted-slope dB dynamic range (`20·log10(max gain ÷ min gain)`, a
  level-independent measure of the source range's overall tilt) clears a
  threshold. A spectrally flat source range yields ~unity gains and is left
  alone; a steeply tilted one flips the per-`aspx_config` flag so the
  decoder applies the §5.7.6.4.1.4 Pseudocode 89 inverse pre-flatten gain to
  the patched tile (re-shaping the spectrum within each subband group while
  the SIGNAL envelope, restored *after* pre-flattening, pins each group's
  energy). Wired into every live path (5_X ASPX_ACPL_3 single + multi-env,
  5_X / 7_X ASPX_ACPL_2, 7.0 pure-ASPX, 7_X ASPX_ACPL_1) via an
  `extract_aspx_preflat` per-carrier analysis. Still pending:
  `fic_used_in_sfb` / `tic_used_in_slot` remain
  at the all-zero scaffold on every live path — they are parsed but not yet
  driven through the decoder's HF synthesis, so an encoder decision for
  them would be informative-only (a docs gap on their §5.7.6.4 synthesis
  semantics blocks a real round-trip). The 7.X ASPX_ACPL_3 path does not
  yet exist. The `aspx_tna_mode` / `aspx_add_harmonic` threshold mappings
  are encoder tuning choices (the spec leaves the selection informative);
  they are calibrated to the live QMF pipeline but not yet tuned against a
  perceptual reference.
- The live 5_X ASPX_ACPL_3 real-ASPX frame path now selects between a
  single FIXFIX envelope and a `num_env = 2` multi-envelope body per
  frame (`encode_frame_pcm_5_{0,1}_acpl3_real_aspx_multi_env` — the
  encoder probes the L/R HF QMF energy for a transient and emits the
  multi-envelope `aspx_data_2ch()` with per-envelope FREQ/TIME DPCM when
  one is present, else falls back to the single-envelope path). The
  ASPX_ACPL_2 5.X live frame path now also emits a **real single-envelope
  `aspx_data_1ch()`** for the centre carrier
  (`encode_frame_pcm_5_{0,1}_acpl2_real_aspx`: QMF-analyses L/R **and** C,
  emitting real SIGNAL/NOISE envelopes on all three carriers via
  `write_aspx_data_1ch_real_envelope` + `write_aspx_data_2ch_real_envelope`).
  The 7.X ASPX_ACPL_2 live frame path now also emits real single-envelope
  ASPX on all three carriers — both carrier-pair `aspx_data_2ch()` elements
  (L/R front, Ls/Rs surround) **and** the centre `aspx_data_1ch()`
  (`encode_frame_pcm_7_{0,1}_acpl2_real_aspx` →
  `build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx`). The
  7.0 pure-ASPX path (`encode_frame_pcm_7_0_aspx_real_aspx` →
  `build_7_0_aspx_asf_body_from_pcm_spectra_real_aspx_tna`) and the 7_X
  ASPX_ACPL_1 path (`encode_frame_pcm_7_{0,1}_acpl1_real_alpha_beta` →
  `build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta_real_aspx_tna`)
  now also emit real single-envelope ASPX + real `aspx_tna_mode` on every
  carrier. The live `aspx_data_1ch()` path remains single-envelope
  (`num_env = 1`), and multi-envelope (`num_env > 1`) is wired only on the
  5.X ASPX_ACPL_3 live path; `num_env > 2` (requiring a wider
  `num_env_bits_fixfix`) is not yet selected.

## Specs

- ETSI TS 103 190-1 — channel-based coding + bitstream syntax.
- ETSI TS 103 190-2 — multi-stream / immersive / object-based (IFM).

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-ac4 = "0.0"
```

## Codec id

`"ac4"`. Also registers the ISO BMFF fourcc `ac-4` so MP4 tracks tagged
with the AC-4 sample entry resolve cleanly.

## License

MIT — see [LICENSE](LICENSE).
