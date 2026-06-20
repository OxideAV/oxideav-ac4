# oxideav-ac4

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
  16-bit `frame_size()` with 24-bit escape, plus a CRC-16 helper.
- **Table of contents** (`toc`) — the full `ac4_toc()` walker:
  bitstream_version, sequence_counter, fs_index, frame_rate_index,
  `b_iframe_global`, payload_base, per-presentation
  `ac4_presentation_info()` (single / multi-substream, configs 0..=5
  plus extension escape, HSF extension, pre-virtualised flag, extra
  EMDF substreams), per-substream `ac4_substream_info()` (channel-mode
  prefix decoder, sf_multiplier, bitrate_indicator, content_type with
  language tag, b_iframe), the substream index table, and the
  `variable_bits(n)` codec. Surfaced on a parsed `Ac4FrameInfo`.

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
  envelope / noise / tone payload decode, QMF analysis/synthesis, and the
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
  per-codeword `ajoc_huff_data()` decode that feeds the dequantized
  matrices into `ajoc_reconstruct` is blocked on the missing
  `AJOC_HCB_*` codebook arrays (see "Not yet supported").
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

### Encoder

`Ac4ImsEncoder` emits IMS v2 frames for the channel-based layouts:

- Mono / stereo (SIMPLE/ASF split-MDCT and joint M/S CPE).
- 5.0 / 5.1 and 7.0 / 7.1 SIMPLE/Cfg3Five (per-channel forward MDCT +
  DP-optimal sectioning + HCB selection + SNF, with an LFE element for
  the `.1` layouts).
- 5.X / 7.X ASPX_ACPL_1 / _2 / _3 paths with real per-parameter-band
  α / β extraction from the input channels' MDCT energy / correlation.
- 5.X ASPX_ACPL_3 with a **real ASPX SIGNAL / NOISE envelope** on the
  L / R carriers (`encode_frame_pcm_5_{0,1}_acpl3_real_aspx`): the
  encoder QMF-analyses the input PCM, aggregates the HF energy across the
  A-SPX subband-group borders (Pseudocodes 90/91), quantises + FREQ-DPCM
  packs it (Pseudocodes 80–83), and emits a real-envelope
  `aspx_data_2ch()` instead of the minimum-bit-cost scaffold.

## Not yet supported

- **A-JOC per-codeword Huffman decode** (`ajoc_huff_data()`, §6.2.5.5) —
  blocked on a docs gap: the twelve `AJOC_HCB_*` Huffman `_LEN` / `_CW`
  arrays (Annex A.1.1 Tables A.1-A.12) are named in the spec with their
  `codebook_length` / `cb_off` metadata, but the actual codeword and
  length values are not listed in the TS 103 190-2 PDF and are not in
  the part-1 accompaniment table file (which carries only the part-1
  ASF / DRC / DE codebooks). The `ajoc` module's differential decoder
  consumes those deltas directly the moment the arrays are supplied; the
  full end-to-end A-JOC object decode also needs the surrounding
  immersive / OAMD substream machinery (`audio_data_ajoc()`,
  `oamd_dyndata_single()`, `var_channel_element()`).
- TS 103 190-2 multi-stream / immersive / object-based (IFM) extensions.
- Per-`emdf_payload_id` semantic interpretation of EMDF payload bodies
  (captured as raw bytes).
- Some advanced A-CPL parameters (β3 / γ on certain encoder paths)
  remain scaffolded at minimum-bit-cost defaults.
- **A-SPX `aspx_hfgen_iwc` sub-fields:** the live 5_X ASPX_ACPL_3 path now
  emits a real `aspx_tna_mode` (inverse filtering), but `add_harmonic` /
  `fic_used_in_sfb` / `tic_used_in_slot` remain at the all-zero scaffold
  on every live path, and the real-`aspx_tna_mode` selection is wired only
  into the 5_X ASPX_ACPL_3 frame path so far (the `_tna` body / writer
  variants exist for the other paths but aren't yet driven by the live
  encoders). The `aspx_tna_mode` threshold mapping is an encoder tuning
  choice (the spec leaves the selection informative); it is calibrated to
  the live QMF pipeline but not yet tuned against a perceptual reference.
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
  remaining 7.X paths (ASPX_ACPL_1 / ASPX_ACPL_3 / pure-ASPX) still write
  the single-envelope scaffold on the live frame path, the live
  `aspx_data_1ch()` path remains single-envelope (`num_env = 1`), and
  multi-envelope (`num_env > 1`) is wired only on the 5.X ASPX_ACPL_3 live
  path; `num_env > 2` (requiring a wider `num_env_bits_fixfix`) is not yet
  selected.

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
