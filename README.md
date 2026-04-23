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
> decoder that returns PCM silence with the right shape. Full audio
> decoding (ASF / ASF-A2 / A-SPX subband coding) is not implemented yet.

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
  codec).
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
