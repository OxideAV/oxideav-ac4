# oxideav-ac4

Pure-Rust **Dolby AC-4** audio decoder foundation — sync / TOC / presentation
/ substream parsing, plus a stub decode path that emits silence at the
correct channel count and sample rate so container fixtures can round-trip
without crashing. Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

> **Status**: Foundation. AC-4 is a complex codec. This crate parses the
> bitstream framing, the table of contents (`ac4_toc()`), presentations
> and substream descriptors per ETSI TS 103 190-1 V1.4.1, and exposes a
> decoder that emits PCM with the right shape. Mono ASF, stereo CPE
> (split + joint MDCT), full A-SPX front-end, A-CPL channel-pair
> synthesis (ASPX_ACPL_1 / ASPX_ACPL_2), DRC + DE + outer metadata
> walker are all implemented. Round 20 unblocks the ETSI Huffman-table
> audit (60 codebooks validated byte-for-byte against the canonical
> ETSI accompaniment file in `tests/etsi_table_validation.rs`) and
> wires the 5.X channel-element walker family's Cfg0 / Cfg1 / Cfg2
> outer shells plus a Table-21-correct `sf_info_lfe()` parser. Round 21
> lands the §5.7.7.6.2 ASPX_ACPL_3 transform-matrix synthesis math
> (Pseudocodes 118/119 — `Transform()`, `ACplModule2()`, `ACplModule3()`
> and the full 5-channel pipeline `run_pseudocode_118_5x()`). Round 22
> lands §5.7.7.6.1 ASPX_ACPL_1 / ASPX_ACPL_2 multichannel wrappers
> (Pseudocode 117 — `run_pseudocode_117_5x()`: two parallel ACplModule's
> with D0/D1 decorrelators) plus the 5_X-walker glue: PCM-level helpers
> `run_acpl_5x_pair_pcm()` (ASPX_ACPL_1/2) and `run_acpl_5x_mch_pcm()`
> (ASPX_ACPL_3) consume the parsed `acpl_config_*` + `acpl_data_*` to
> produce 5-channel L/R/C/Ls/Rs PCM end-to-end via QMF
> analysis → A-CPL → QMF synthesis. Round 23 wires the per-channel
> `sf_data(ASF)` Huffman bodies for the multichannel layouts (Tables
> 26 / 27 / 28 / 29): `parse_two_channel_data` / `parse_three_channel_data`
> / `parse_four_channel_data` / `parse_five_channel_data` now also walk
> the trailing 2 / 3 / 4 / 5 `sf_data(ASF)` calls through
> `decode_mch_sf_data_channels()` and deposit the per-channel scaled
> MDCT spectra on each `*ChannelData::scaled_spec_per_channel` for the
> long-frame, single-window-group case. The Huffman codebook IDs reused
> are `HCB_1`..`HCB_11` (spectral lines), `HCB_SCALEFAC` (scale-factor
> DPCM) and `HCB_SNF` (spectral noise fill) — Annex A.1 shares the
> codebooks across mono / stereo / multichannel; there is no separate
> "MCH" codebook set. Round 24 closes the two r23 follow-ups: (1) the
> grouped / short-frame multichannel `sf_data(ASF)` walker
> (`num_window_groups > 1`) is now driven by
> `decode_asf_grouped_mono_body_with_max_sfb()` — each per-channel
> spectrum is the concatenation of `num_window_groups` independent
> `(section + spectral + scalefac + snf)` chains, group-major; (2) the
> ASPX_ACPL_3 inner body walker is now wired in
> `parse_5x_audio_data_outer` — on an I-frame the walker chains
> `stereo_data() + aspx_data_2ch() + acpl_data_2ch()` and the parsed
> `tools.acpl_data_2ch` flows straight into the §5.7.7.6.2
> Pseudocode-118 5_X synthesis pipeline. The Table-52 `aspx_data_2ch()`
> body parser was factored out of the stereo CPE ASPX path into a
> shared `parse_aspx_data_2ch_body()` helper — both the stereo CPE
> mode and the 5_X ASPX_ACPL_3 mode use the same parser. Round 25 wires
> the **ASPX_ACPL_1 / ASPX_ACPL_2 inner body walker** in
> `parse_5x_audio_data_outer` per §4.2.6.6 Table 25
> (`case ASPX_ACPL_1: case ASPX_ACPL_2:`): a new
> `parse_aspx_acpl_1_2_inner_body()` helper walks
> `two_channel_data() / three_channel_data()` (selected by the 1-bit
> `coding_config`), the ASPX_ACPL_1-only joint-MDCT residual layer
> (`max_sfb_master + 2x chparam_info + 2x sf_data(ASF)` over the
> dominant transform length signalled by the upstream channel data —
> `n_side_bits` is derived per the §4.2.6.6 NOTE), the optional Cfg0
> trailer `mono_data(0)`, then `aspx_data_2ch()` + `aspx_data_1ch()` and
> finally the **two parallel `acpl_data_1ch()` calls** per Pseudocode 117.
> The pair lands in `tools.acpl_data_1ch_pair[0/1]` (D0 / D1
> ACplModule). The walker is try-and-bail: any inner Huffman / parse
> miss leaves the already-populated `tools.*` slots intact and returns
> silently. Round 27 lands the **7_X channel-element walker**
> (`parse_7x_audio_data_outer`) per §4.2.6.14 Table 33 — immersive 7.0
> and 7.1 streams now parse end-to-end. The 7.X walker mirrors the 5_X
> SIMPLE/ASPX path's `coding_config` selector but has its own quirks:
> 2-bit `7_X_codec_mode` (no ASPX_ACPL_3 in 7.X), `companding_control(5)`
> only on ASPX_ACPL_{1,2}, the centre/back monos move out of the
> coding_config switch into a single trailing `mono_data(0)` gated on
> `coding_config in {0, 2}`, and a SIMPLE/ASPX-only additional-channel
> block (`b_use_sap_add_ch + optional chparam_info×2 +
> two_channel_data`) carries the front-extension / back-surround pair
> beyond the 5.X core. `walk_ac4_substream` now dispatches
> `channels == 7/8` (7.0/7.1) into the new walker. **416 tests** (405
> lib + 5 + 6 integration). Pending: ASF short-frame `sf_data` walk
> for the mono / stereo paths (the grouped walker added in r24 covers
> the multichannel layouts only).

## Specs

- ETSI TS 103 190-1 — Channel-based coding + bitstream syntax.
- ETSI TS 103 190-2 — Multi-stream / Immersive / Object-based (IFM).

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-ac4 = "0.0"
```

## What's parsed (TS 103 190-1 clause 4)

- **Sync frame** (`ac4_syncframe()`, Annex G) — `0xAC40` plain or `0xAC41`
  CRC-protected, plus the two-tier `frame_size()` (16-bit, `0xFFFF`
  escape to 24-bit).
- **Raw frame** (`raw_ac4_frame()`).
- **Table of contents** (`ac4_toc()`): bitstream_version (with
  `variable_bits(2)` escape for version == 3), sequence_counter,
  wait_frames, `fs_index` -> 44.1 / 48 kHz, `frame_rate_index` -> 24…120
  fps + 23.44 (Table 83 / 84), `b_iframe_global`, payload_base.
- **Presentations**: per-presentation `ac4_presentation_info()` walking
  both the `presentation_v1` (default) and `presentation_v0` forms.
  Handles `presentation_config` 0..=5 (M+E+D, Main+DE, Main+Assoc,
  M+E+D+Assoc, Main+DE+Assoc, Main+HSF) plus the
  `presentation_config_ext_info` escape, `b_hsf_ext`, `b_pre_virtualized`
  and additional EMDF substreams.
- **Substream info**: `ac4_substream_info()` channel mode (1/2/4/7-bit
  with `variable_bits(2)` escape), sample-frequency multiplier,
  bitrate_indicator, content_type + language tag, per-frame-rate-factor
  `b_iframe` flags.
- **Substream index table**: per-substream `substream_size` with the
  `b_more_bits` / `variable_bits(2)` extension.
- **Bit-rate indicator / content classifier / frame_rate_factor /
  sf_multiplier** all surfaced on the parsed `Ac4FrameInfo` struct.

## What's not parsed yet

- ASF / ASF-A2 / A-SPX audio coefficient coding (the heart of the
  codec). The A-SPX `aspx_config()` header and `companding_control()`
  element **are** parsed (ETSI §4.2.11 / §4.2.12.1); the Huffman-coded
  envelope / noise payload (`aspx_framing`, `aspx_ec_data`, etc.) is
  not.
- Metadata payloads inside substreams (DRC, dialog normalization,
  downmix params) — the spec's `metadata()` tree is skipped by size,
  not parsed.
- TS 103 190-2 IFM (immersive / object) extensions.

## Decode path

`make_decoder` builds an `Ac4Decoder` that:

1. Scans the packet for a sync word.
2. Parses the full TOC + presentation + substream descriptors, and
   therefore knows the channel count, sample rate (44.1 / 48 kHz
   scaled by `sf_multiplier`), and frame length in samples.
3. Emits a silence `AudioFrame` (S16 zeros) with the correct
   `channels`, `sample_rate`, `samples` and `pts`.

This is enough to keep a container/demuxer pipeline running against an
AC-4 track without crashing, and to exercise the TOC parser against
real fixtures.

## Codec id

`"ac4"`. Also registers the ISO BMFF fourcc `ac-4` so MP4 tracks tagged
with the AC-4 sample entry resolve cleanly.

## License

MIT — see [LICENSE](LICENSE).
