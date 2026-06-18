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
  envelope / noise / tone payload decode, and QMF analysis/synthesis.
- **A-CPL** channel-pair coupling — ASPX_ACPL_1 / _2 (Pseudocode 117)
  and ASPX_ACPL_3 (Pseudocode 118) synthesis producing 5-channel
  L/R/C/Ls/Rs PCM, plus the 7.X (7.0 / 7.1) walker.
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

- TS 103 190-2 multi-stream / immersive / object-based (IFM) extensions.
- Per-`emdf_payload_id` semantic interpretation of EMDF payload bodies
  (captured as raw bytes).
- Some advanced A-CPL parameters (β3 / γ on certain encoder paths)
  remain scaffolded at minimum-bit-cost defaults.
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
  The 7.X layouts still write the single-envelope scaffold on the live
  frame path, the live `aspx_data_1ch()` path remains single-envelope
  (`num_env = 1`), and `num_env > 2` (requiring a wider
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
