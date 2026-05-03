# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/OxideAV/oxideav-ac4/compare/v0.0.2...v0.0.3) - 2026-05-03

### Other

- skip etsi_table_validation when docs sibling absent
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- round 28 — mono / stereo short-frame sf_data(ASF) walker
- round 27 — 7_X channel-element walker (immersive 7.0 / 7.1)
- round 26 — add per-codebook decode roundtrip sweeps
- round 25 — ASPX_ACPL_1 / ASPX_ACPL_2 inner body walker
- round 24 — grouped multichannel sf_data(ASF) walker + ASPX_ACPL_3 inner body walker
- round 23 — multichannel sf_data(ASF) Huffman codebook table walk
- round 22 — ASPX_ACPL_1/2 multichannel wrapper (Pseudocode 117) + 5_X-walker glue
- round 21 — ASPX_ACPL_3 transform synthesis (Pseudocodes 118/119)
- round 20 — ETSI Huffman table audit + 5.X cfg0/1/2 + sf_info_lfe
- round 19 — design 5_X channel-element walker family
- round 18 — wire ASPX_ACPL_1 joint-MDCT residual layer
- round 17 — wire A-CPL synthesis into Ac4Decoder
- adopt slim AudioFrame shape
- land §5.7.7 A-CPL QMF synthesis math (round-16)
- outer §4.2.14.1 metadata() walker + §5.7.7.2 sb_to_pb
- A.4 Huffman codebooks + dialog_enhancement parser
- A.3 Huffman codebooks + acpl_data_*ch parser
- A.5 Huffman codebook + drc_frame parser
- implement complex-covariance TNS (chirp + α0 + α1) — round-11
- land §5.7.6.4.2.2 A-SPX limiter (P72 + P96..101) — round-10
- pin release-plz to patch-only bumps

### Added

- **Round 28 — mono / stereo short-frame `sf_data(ASF)` walker** (TS 103
  190-1 §4.2.8.3-6 Tables 39-42, §4.3.6.2.6 Pseudocodes 2/3/5):
  - New spec-correct `_grouped` payload parsers in `asf_data.rs` —
    `parse_asf_section_data_grouped()`,
    `parse_asf_spectral_data_grouped()`,
    `parse_asf_scalefac_data_grouped()`,
    `parse_asf_snf_data_grouped()` — each takes per-group transform-
    length and `max_sfb` arrays and walks the spec's outer
    `for (g = 0; g < num_window_groups; g++)` loop. Critically:
    `asf_scalefac_data()` consumes a *single* 8-bit
    `reference_scale_factor` at the head with `first_scf_found` shared
    across groups (DPCM state is continuous over the whole frame), and
    `asf_snf_data()` consumes a *single* 1-bit `b_snf_data_exists`
    gate at the head. This matches Tables 41 / 42 verbatim.
  - New helpers in `asf.rs`:
    `derive_per_group()` / `derive_per_group_with_max_sfb()` resolve
    per-group `(transf_length_idx, transform_length, max_sfb)` from
    `(AsfTransformInfo, AsfPsyInfo)` per Pseudocodes 2 (`get_transf_length`)
    and 5 (`get_max_sfb`), including the `b_different_framing`
    half-frame split (Pseudocode 3's grouping-bit shift +
    `num_windows_0 - 1` boundary injection).
  - New body decoders:
    - `decode_asf_grouped_mono_body[_with_max_sfb]()` — wraps the four
      `_grouped` payload parsers; returns the per-group dequantised
      spectra concatenated group-major.
    - `decode_asf_grouped_stereo_joint_body()` — joint-MDCT residual
      layer with shared section, two independent spectral bodies (L/M
      then R/S), shared scalefactors (band-wise max_quant_idx over
      both channels), per-group `ms_used[g][sfb]` flag arrays, then
      snf. Inverse M/S applied per-group: L = M+S, R = M-S for bands
      with ms_used set.
    - `decode_asf_mono_body_dispatch()` /
      `decode_asf_mono_body_for_max_sfb()` — long-frame vs grouped
      dispatch wrappers used by all per-channel call sites.
  - Wired into the four mono / stereo call sites:
    - `parse_mono_audio_data_outer()` — mono SIMPLE / ASPX path.
    - `parse_aspx_acpl2_mdct_body()` — single-channel ASPX_ACPL_2
      MDCT residual.
    - `parse_aspx_acpl1_mdct_body()` joint + split — ASPX_ACPL_1
      joint-MDCT residual layer (two independent mono bodies with
      `max_sfb_0` / `max_sfb_side_0`) and the split case.
    - `parse_stereo_data_body()` joint + split — stereo CPE body
      with both joint MDCT (shared section + ms_used) and split MDCT
      (two independent mono bodies).
  - Real Dolby AC-4 mono / stereo streams that include short-frame
    `sf_data(ASF)` (i.e. the encoder picks short-window sub-frames)
    now decode end-to-end without bailing at the
    `num_window_groups != 1` guard. The grouped multichannel walker
    in `mch.rs` from r24 (per-group interleaved
    `section + spectral + scalefac + snf`) is left untouched — its
    pinned tests continue to pass.
  - 9 new tests: 4 in `asf_data.rs` (grouped section / scalefac
    reference-once / scalefac DPCM-state-carries / snf gate-once) and
    5 in `asf.rs` (decode_asf_grouped_mono two-group + truncated;
    parse_mono_audio_data_outer SIMPLE short-frame; parse_stereo_data_body
    split + joint short-frame). **425 tests** (414 lib + 5 + 6
    integration), up from 416.

- **Round 27 — 7_X channel-element walker (immersive 7.0 / 7.1)**
  (TS 103 190-1 §4.2.6.14 Table 33 + §4.3.5.7 Table 98):
  - New `parse_7x_audio_data_outer()` walker in `mch.rs` plus a
    `SevenXCodecMode` enum (Table 98 — 2 bits, 4 codepoints: SIMPLE /
    ASPX / ASPX_ACPL_1 / ASPX_ACPL_2; **no** ASPX_ACPL_3 in 7.X). The
    walker mirrors the 5_X SIMPLE/ASPX path's coding_config selector
    but with the 7.X-specific shape:
    - 2-bit `7_X_codec_mode` (vs 3-bit for 5_X — no Reserved values).
    - LFE `mono_data(1)` gated on `channel_mode == "7.1"` (mapped from
      the parent substream's channel count: 7 → 7.0, 8 → 7.1).
    - `companding_control(5)` for ASPX_ACPL_{1,2} only — SIMPLE/ASPX in
      7.X have no leading companding (different from 5_X where ASPX
      gets `companding_control(5)`).
    - Cfg0 body: `2ch_mode + two_channel_data + two_channel_data` (no
      centre mono inside the switch).
    - Cfg2 body: `four_channel_data` only (no surround mono inside the
      switch). Both centre / surround monos move out to a single
      trailing `mono_data(0)` call gated on `coding_config in {0, 2}`,
      placed after the additional-channel block.
    - SIMPLE/ASPX-only additional-channel block: 1-bit
      `b_use_sap_add_ch` gating optional `chparam_info()×2`, then a
      mandatory `two_channel_data()` for the front-extension /
      back-surround pair beyond the 5.X core. Lands in new
      `tools.seven_x_b_use_sap_add_ch`,
      `tools.seven_x_add_chparam_info` and
      `tools.seven_x_additional_channel_data` slots.
    - ASPX_ACPL_1-only joint-MDCT residual layer (max_sfb_master +
      chparam_info×2 + sf_data×2) — same shape as the 5_X path,
      `n_side_bits` derived per the Table 33 NOTE from the largest
      signalled transform length across all preceding
      `two_channel_data` / `three_channel_data` / `four_channel_data` /
      `five_channel_data` (including the additional-channel one when
      it's the largest).
    - Trailers: `aspx_data_2ch×2 + aspx_data_1ch` for any non-SIMPLE,
      plus an extra `aspx_data_2ch` for ASPX (covering the additional
      pair); `acpl_data_1ch×2` for ASPX_ACPL_{1,2} landing in
      `tools.acpl_data_1ch_pair[0/1]` (shared with the 5_X
      §5.7.7.6.1 pair walker).
  - `walk_ac4_substream` now dispatches `channels == 7` (7.0) and
    `channels == 8` (7.1) into the new walker. Previously these
    channel counts fell through to the catch-all that just records
    `channel_mode_channels` and bails — real Dolby AC-4 streams using
    a `7_X_channel_element` now parse end-to-end without hitting the
    catch-all.
  - Walker is **try-and-bail** with the same contract as the 5_X
    walker: any inner Huffman / parse miss surfaces `Ok(())` to the
    caller, leaving already-populated `tools.*` slots intact. The
    deeper `aspx_data` / `acpl_data` trailers are gated on
    `b_iframe && tools.aspx_config.is_some()`.
  - 11 new lib tests (394 → 405 total): SIMPLE Cfg3 (no SAP), 7.1
    SIMPLE LFE walk, SIMPLE Cfg0 (two pairs + trailing centre mono),
    SIMPLE Cfg2 (four-channel + back surround mono), SIMPLE Cfg1 (no
    trailer), SIMPLE with `b_use_sap_add_ch == 1` (chparam pair
    populated), ASPX_ACPL_2 non-iframe Cfg1 (no additional-channel
    block), ASPX_ACPL_1 I-frame Cfg0 (residual layer + Cfg0 trailer),
    ASPX_ACPL_1 zero `max_sfb_master` bails silently, truncated
    SIMPLE five_channel_data bails silently, and
    `SevenXCodecMode::from_u32` round-trip.

- **Round 25 — ASPX_ACPL_1 / ASPX_ACPL_2 inner body walker**
  (TS 103 190-1 §4.2.6.6 Table 25 + §5.7.7.6.1 Pseudocode 117):
  - New `parse_aspx_acpl_1_2_inner_body()` helper in `mch.rs` walks the
    bits past the existing `companding_control(3) + 1-bit
    coding_config` selector for the 5_X `ASPX_ACPL_1` / `ASPX_ACPL_2`
    paths. The body shape (Table 25):
    `two_channel_data()` OR `three_channel_data()` →
    [ASPX_ACPL_1 only] `max_sfb_master (n_side_bits) +
     chparam_info()×2 + sf_data(ASF)×2` joint-MDCT residual layer →
    [coding_config==0 only] `mono_data(0)` centre/surround trailer →
    `aspx_data_2ch() + aspx_data_1ch() + acpl_data_1ch()×2`.
    The two acpl_data_1ch payloads land in
    `tools.acpl_data_1ch_pair[0]` (D0-side) and
    `tools.acpl_data_1ch_pair[1]` (D1-side) per Pseudocode 117 — the
    same pair the §5.7.7.6.1 `run_acpl_5x_pair_pcm()` PCM driver
    consumes.
  - `n_side_bits` is derived per the §4.2.6.6 NOTE: largest signalled
    transform length from the preceding `two_channel_data()` /
    `three_channel_data()` (look up `tables::n_msfb_bits_48` Table 106
    column 2). The joint-MDCT residual sf_data bodies reuse
    `decode_asf_long_mono_body_with_max_sfb()` against a synthesised
    long-frame `AsfTransformInfo` at the dominant transform length.
  - The walker is **try-and-bail**: every step returns `Ok(())` to the
    outer walker on any inner Huffman / parse miss, leaving the
    already-populated `tools.*` slots intact (matching the round-24
    ASPX_ACPL_3 walker contract). Deeper aspx_data / acpl_data steps
    are gated on `b_iframe && tools.aspx_config.is_some()` —
    non-iframe paths simply consume what they can of the upstream
    channel data and stop.
  - Active `acpl_config_1ch` for the pair-extraction step is selected
    by codec mode: `acpl_config_1ch_partial` for ASPX_ACPL_1 (with
    `start_band` derived from `qmf_band` via `acpl::sb_to_pb()`),
    `acpl_config_1ch_full` for ASPX_ACPL_2 (start_band always 0).
  - 6 new lib tests + 2 new `tests/acpl_5x_pipeline.rs` integration
    tests (387 → 395 total): non-iframe ASPX_ACPL_2 `coding_config==0`
    path lands `two_channel_data` + `cfg0_centre_mono` and leaves the
    ACPL pair `None` (gated); non-iframe ASPX_ACPL_1
    `coding_config==1` path lands `three_channel_data` and walks the
    joint-MDCT residual layer; I-frame ASPX_ACPL_2 with
    `three_channel_data` parses `aspx_config + acpl_config_1ch_full`
    out of the bitstream and walks the channel data; I-frame
    ASPX_ACPL_1 with Cfg0 walks the residual layer + Cfg0 mono trailer
    end-to-end; truncated `three_channel_data` mid-bitstream bails
    silently with `Ok(())`; `max_sfb_master == 0` in the residual
    layer bails silently. Two integration tests assert the
    walker → synthesis glue: a real Table-27 `three_channel_data()`
    body now flows into `tools.three_channel_data` (which the r22
    walker treated as opaque), and the staged ACPL pair drives
    `run_acpl_5x_pair_pcm()` end-to-end for both ASPX_ACPL_1 and
    ASPX_ACPL_2 modes.

- **Round 24 — Grouped multichannel `sf_data(ASF)` walker + ASPX_ACPL_3
  inner body walker** (TS 103 190-1 §4.2.6.6 + §5.4.4.4 + Table 52 / 62):
  - `decode_mch_sf_data_channels()` in `mch.rs` now also handles the
    grouped / short-frame case (`num_window_groups > 1`). A new
    `decode_asf_grouped_mono_body_with_max_sfb()` helper walks
    `num_window_groups` independent `(asf_section_data +
    asf_spectral_data + asf_scalefac_data + asf_snf_data)` chains per
    body and concatenates the per-group dequantised spectra
    group-major into a single `Vec<f32>` of length
    `num_window_groups * sfb_offset[max_sfb]`. With `b_dual_maxsfb = 0`
    every group shares the same `max_sfb_0`, matching Pseudocode 5
    `get_max_sfb(g)` for the non-side-channel multichannel path.
    `parse_two_channel_data` / `parse_three_channel_data` /
    `parse_four_channel_data` / `parse_five_channel_data` now populate
    `scaled_spec_per_channel` for **both** the long-frame /
    single-window-group (r23) and the grouped / multi-window-group
    paths.
  - `parse_5x_audio_data_outer` for `5_X_codec_mode == ASPX_ACPL_3`
    now walks the inner body (Table 25 row ASPX_ACPL_3:
    `stereo_data() + aspx_data_2ch() + acpl_data_2ch()`). The flow is:
    `parse_stereo_data_body()` → on success + I-frame +
    `tools.aspx_config.is_some()`, `parse_aspx_data_2ch_body()` →
    `parse_acpl_data_2ch(num_param_bands, 0, qm0, qm1)`. The parsed
    `tools.acpl_data_2ch` slot now flows straight into the
    §5.7.7.6.2 Pseudocode-118 5_X synthesis pipeline (closing the
    contract the round-22 staging tests stubbed by hand).
  - **Refactor**: factored the Table-52 `aspx_data_2ch()` body parser
    out of the stereo CPE ASPX path (`parse_stereo_audio_data_outer`)
    into a shared `pub(crate) parse_aspx_data_2ch_body()` helper in
    `asf.rs`. Both the stereo CPE `StereoCodecMode::Aspx` mode and
    the new 5_X `ASPX_ACPL_3` mode now drive this single parser —
    one definition of `aspx_xover_subband_offset + aspx_framing(0) +
    aspx_balance + [aspx_framing(1)] + aspx_delta_dir(0/1) +
    aspx_hfgen_iwc_2ch + 4x aspx_ec_data` instead of two divergent
    inline copies.
  - 7 new tests (380 → 387 total): grouped two-group two-channel walk
    with all-zero spectra (length matches `2 * sfb_offset[max_sfb]`);
    three-group one-channel walk pinning the linear `num_window_groups`
    scale; `parse_three_channel_data` grouped-short-frame end-to-end
    walk through `parse_5x_audio_data_outer`; `parse_two_channel_data`
    grouped-short-frame walk for Cfg0 / Cfg1; truncated grouped input
    yields `None` without panicking; ASPX_ACPL_3 non-iframe leaves
    `tools.acpl_data_2ch == None`; ASPX_ACPL_3 I-frame parses the
    `aspx_config + acpl_config_2ch` configs out of the bitstream and
    surfaces them on tools (the inner body walker bails silently
    downstream of `stereo_data()` on a degenerate aspx_config — that
    bail is part of the try-and-bail contract).

- **Round 23 — Multichannel `sf_data(ASF)` Huffman codebook table walk**
  (TS 103 190-1 §4.2.6.7-10 Tables 26 / 27 / 28 / 29):
  - New `decode_mch_sf_data_channels()` helper in `mch.rs` walks
    `n_channels` consecutive `sf_data(ASF)` bodies sharing the head
    `(transform_info, psy_info)` pair. Each body decodes
    `asf_section_data` then `asf_spectral_data` then `asf_scalefac_data`
    then `asf_snf_data` per §4.2.8.3-6, producing one dequantised + scaled
    MDCT spectrum per channel of length `sfb_offset[max_sfb]`.
  - `parse_two_channel_data()` / `parse_three_channel_data()` /
    `parse_four_channel_data()` / `parse_five_channel_data()` now walk
    the trailing 2 / 3 / 4 / 5 `sf_data(ASF)` calls and store the
    per-channel scaled spectra on each `*ChannelData::scaled_spec_per_channel`
    (new `Vec<Option<Vec<f32>>>` field). For the long-frame,
    single-window-group case every slot is populated; short / grouped
    frames push `Some(...)` for none of the slots and let the outer
    shell still parse cleanly.
  - Huffman codebook IDs wired (per Annex A.1, all reused from the
    mono / stereo paths — there is no separate "MCH" codebook set):
    `HCB_1` (`ASF_HCB_1_LEN/CW`, 81 entries) through `HCB_11`
    (`ASF_HCB_11_LEN/CW`, 289 entries) for spectral lines,
    `HCB_SCALEFAC` (`ASF_HCB_SCALEFAC_LEN/CW`, 121 entries) for
    scale-factor DPCM, and `HCB_SNF` (`ASF_HCB_SNF_LEN/CW`, 22 entries)
    for spectral noise fill. Round 22's
    `decode_asf_long_mono_body_with_max_sfb` was raised from `fn` to
    `pub(crate) fn` so `mch.rs` can drive one body per channel from the
    shared `sf_info` block.
  - Removed the previous "scaffold values" comments / TODOs from
    `mch.rs` for the per-channel `sf_data(ASF)` paths — the per-channel
    spectra now flow through the validated ASF Huffman codebook suite
    (audited byte-for-byte in r20's `etsi_table_validation.rs` against
    `docs/audio/ac4/ts_10319001v010401p0-tables.c`).
  - 6 new tests (374 → 380 total): all-zero two-channel sf_data round
    trip with sfb-offset length pin, short-frame guard returns all-`None`,
    `parse_three_channel_data` decodes 3 bodies with all-zero spectra
    pin, `parse_four_channel_data` + `parse_five_channel_data`
    per-channel-count pin, truncated `sf_data` graceful partial decode,
    `parse_two_channel_data` per-channel length-matches-sfb-offset pin.
    The pre-existing 5_X outer-walker tests (`parse_5x_outer_simple_*`)
    were extended to feed valid all-zero `sf_data(ASF)` trailers so they
    still exercise the outer dispatch end-to-end.

- **Round 22 — ASPX_ACPL_1/2 multichannel wrapper (Pseudocode 117) +
  5_X-walker glue**:
  - New §5.7.7.6.1 multichannel pipeline in `acpl_synth.rs`:
    `run_pseudocode_117_5x()` wraps two parallel
    `run_pseudocode_115_pair()` passes (D0 decorrelator on the L-side
    ACplModule, D1 on the R-side) and forms the five 5.X output
    channels from the L/R/C carriers (plus optional Ls/Rs carriers in
    `ASPX_ACPL_1` mode). Centre channel is a passthrough (`z4 = x2`);
    surround pair (`z1`/`z3`) gets the spec's final `sqrt(2)` scale.
  - New `Acpl5xPairState` (left/right `AcplCpeState` + alpha/beta
    differential-decode rolling state for two `acpl_data_1ch` rows),
    `Acpl5xPairFrame` (5 carrier slots + two `(alpha_dq, beta_dq)`
    matrices + interpolation control), `Acpl5xPairMode` selector
    (`AspxAcpl1` vs `AspxAcpl2`), and `Acpl5xPairOutput` (z0/z1/z2/z3/z4).
  - PCM-level helpers wire the parsed 5_X bitstream straight through
    QMF analysis → A-CPL → QMF synthesis:
    - `run_acpl_5x_pair_pcm()` — drives Pseudocode 117 from
      `(pcm_l, pcm_r, pcm_c[, pcm_ls, pcm_rs], cfg, data_1, data_2)`.
    - `run_acpl_5x_mch_pcm()` — drives Pseudocode 118 from
      `(pcm_l, pcm_r, pcm_c, acpl_config_2ch, acpl_data_2ch)`.
    Both return `Acpl5xPcmOutput { left, right, centre, left_surround,
    right_surround }` PCM buffers and bundle the QMF banks + ACPL state
    in `Acpl5xPairPcmState` / `Acpl5xMchPcmState`.
  - New SubstreamTools fields: `acpl_data_2ch` (parsed
    `acpl_data_2ch()` per Table 62, for ASPX_ACPL_3) and
    `acpl_data_1ch_pair: [Option<...>; 2]` (one `acpl_data_1ch()` per
    parallel ACplModule, for the 5_X ASPX_ACPL_1/2 paths).
  - 8 new lib tests + 3 new `tests/acpl_5x_pipeline.rs` integration
    tests (363 → 374 total): D0/D1 decorrelator-id init,
    Pseudocode 117 ASPX_ACPL_2 centre passthrough + finite-output,
    ASPX_ACPL_1 low-band M/S split spot-check, prev-state carry across
    frames, ASPX_ACPL_2 equivalence to two parallel
    `run_pseudocode_115_pair()` passes, PCM-level input rejection
    (misaligned / surround-presence vs. mode), end-to-end 5-channel
    PCM emission for both `ASPX_ACPL_2` and `ASPX_ACPL_3`, and the
    walker-→-synthesis glue for all three multichannel modes (the
    walker hands back `acpl_config_*` slots, the test stages
    `acpl_data_*` and asserts the synthesis pipeline consumes the
    pair without further glue).

- **Round 21 — ASPX_ACPL_3 transform synthesis (Pseudocodes 118/119)**:
  - New §5.7.7.6.2 multichannel pipeline in `acpl_synth.rs`:
    `transform()` (Pseudocode 119) linearly mixes the two A-CPL
    carriers `(x0, x1)` by interpolated gamma matrices `g1, g2`;
    `acpl_module2()` (Pseudocode 119) builds the `(z0, z1)` channel
    pair from `g1+g1*a`, `g2+g2*a` and the beta-weighted decorrelator
    output; `acpl_module3()` (Pseudocode 119) adds the beta3-driven
    cross-residual term `0.25*y2*(b3 ± b3*a)` to an existing pair.
  - New `run_pseudocode_118_5x()` runs the full 5-channel synthesis
    end-to-end: x0/x1 input scaling by `(1 + 2*sqrt(0.5))`, three
    parallel `Transform()` outputs into the D0/D1/D2 decorrelators
    + transient duckers (one persistent state per path), three
    `ACplModule2()` channel-pair builds (L/Ls, R/Rs, C with `a=1, b=0`),
    three `ACplModule3()` cross-residual corrections, and the final
    `sqrt(2)` channel scaling for `z1`, `z3`, `z4`.
  - New `AcplMchState` (D0/D1/D2 + 3x ducker + per-pset prev gammas),
    `AcplMchFrame` (5 input channels + 6 gammas + 5 alpha/beta arrays
    + interpolation control), `AcplMchOutput` (z0/z1/z2/z3/z4) and
    `AcplQmfMatrix` type alias.
  - 11 new lib tests (352 → 363 total): unit-gamma `Transform()`,
    mixed-gamma combinator, `ACplModule2` zero-coupling, half-x0
    passthrough, `ACplModule3` residual + no-op cases, full
    `run_pseudocode_118_5x()` 5-channel smoke test (finite + non-zero
    on all five outputs), zero-alpha-beta degenerate path, `pb_matrix_*`
    helpers, scaling-factor invariant `1 + 2*sqrt(0.5) == 1 + sqrt(2)`,
    `AcplMchState::new()` zero-init.

- **Round 20 — ETSI Huffman table audit + 5.X coding-config wiring**:
  - New `tests/etsi_table_validation.rs` integration suite parses the
    canonical ETSI accompaniment file
    `docs/audio/ac4/ts_10319001v010401p0-tables.c` at runtime via a tiny
    C-array tokeniser and validates every Huffman codebook this crate
    ships (`huffman_tables.rs` ASF, `aspx_huffman.rs` A-SPX,
    `acpl_huffman.rs` A-CPL, `de_huffman.rs` DE, `drc_huffman.rs` DRC)
    byte-for-byte against it. 60 codebooks, 120 arrays, 0 divergences
    found.
  - `mch::parse_two_channel_data()` lands the Table 26 outer shell
    (sf_info + chparam_info). The 5.X walker now wires Cfg0
    (2ch_mode + two_channel_data ×2 + mono_data(0)), Cfg1
    (three_channel_data + two_channel_data) and Cfg2 (four_channel_data
    + mono_data(0)) — previously gated as r20 TODO behind round-19's
    Cfg3-only path. New `SubstreamTools` fields: `b_2ch_mode`,
    `two_channel_data: Vec<TwoChannelData>`, `cfg0_centre_mono`,
    `cfg2_back_mono`.
  - `asf::parse_asf_psy_info_lfe()` splits the LFE `sf_info_lfe()`
    parser from the regular `parse_asf_psy_info()`. Table 106 column 4
    `n_msfbl_bits` (3 bits @ 1920, 2 bits @ 512, etc.) is now used for
    `max_sfb[0]` instead of the regular `n_msfb_bits`, and
    `parse_mono_data(b_lfe=true)` dispatches to it. The function
    rejects transform lengths whose `n_msfbl_bits == 0` (Table 21
    permits long-frame transforms only on LFE).
  - 5 new lib tests + 6 new integration tests (337 → 352 total).

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
