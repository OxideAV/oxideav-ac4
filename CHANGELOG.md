# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Round 36 — 5_X ASPX_ACPL_1 / ASPX_ACPL_2 decoder dispatch wiring**
  (TS 103 190-1 §5.7.7.6.1, Pseudocode 117):
  - `Ac4Decoder::receive_frame` now dispatches the §5.7.7.6.1
    Pseudocode 117 channel-pair multichannel synthesis when the 5_X
    walker resolved `five_x_mode` to `AspxAcpl1` or `AspxAcpl2` and
    populated the matching `acpl_config_1ch_*` + `acpl_data_1ch_pair`
    tools slots. The dispatch:
    - Reads L/R carrier PCM from `pcm_per_channel[0]`/`[1]` (already
      filled by the stereo ASF / ASPX decode path), zero-filling on
      absence to keep QMF analysis history consistent.
    - Resolves the active `acpl_config_1ch` per mode:
      `acpl_config_1ch_partial` for `AspxAcpl1`,
      `acpl_config_1ch_full` for `AspxAcpl2` (per Tables 25 + 59).
    - Builds zero-filled centre / Ls / Rs carrier placeholders
      (centre matches the existing ACPL_3 wiring; Ls/Rs only for
      ACPL_1 mode per `Acpl5xPairMode` mode-vs-surround consistency
      check). The carrier-decode paths gain real signal when
      `cfg0_centre_mono` and the surround mono carriers acquire
      end-to-end decoders in a future round.
    - Calls `acpl_synth::run_acpl_5x_pair_pcm` and writes the five
      output channels (L, R, C, Ls, Rs) into `pcm_per_channel[0..5]`,
      growing the slot vector as needed.
    - Skipped when the substream is already an `AspxAcpl3` 5_X frame
      (the two pipelines are mutually exclusive per Table 97).
  - The dispatch logic is extracted into a private
    `Ac4Decoder::dispatch_acpl_5x_pair` helper so the path can be
    unit-tested without building a full 5_X TOC + body.
  - Five new unit tests in `decoder::tests` cover the synthesis
    arithmetic + dispatch behaviour:
    - `dispatch_acpl_5x_pair_aspx_acpl_2_emits_five_channels` —
      ACPL_2 dispatch with ±2000 carrier-PCM tones produces five
      channels with non-zero L / R energy.
    - `dispatch_acpl_5x_pair_aspx_acpl_1_emits_five_channels` —
      ACPL_1 dispatch with zero-filled Ls/Rs surround placeholders
      still emits five-channel output.
    - `dispatch_acpl_5x_pair_rejects_unaligned_sample_count` —
      dispatch is a no-op when sample count isn't a multiple of
      `NUM_QMF_SUBBANDS` (64).
    - `dispatch_acpl_5x_pair_zero_fills_missing_carriers` — when L/R
      slots are `None`, dispatch synthesises silence-grade output
      from zero-filled fallbacks.
    - `dispatch_acpl_5x_pair_resolves_partial_for_aspx_acpl_1` —
      regression check that `qmf_band` differs between `partial`
      (1..8) and `full` (always 0) configs per Table 59.
  - Test count: 535 → 540 (+5).
  - The §5.7.7.6.2 ACPL_3 dispatch (round 34 — Pseudocode 118)
    remains unchanged; the new ACPL_1 / ACPL_2 path takes its place
    when those modes are signalled.

- **Round 35 — ETSI float-table validation suite** (TS 103 190-1
  Annex C.1, C.3, C.11 + Annex D.2, D.3):
  - New `parse_c_float_arrays()` parser in `tests/etsi_table_validation.rs`
    extracts every `const float` / `const float32` array (1-D and 2-D) from
    the ETSI accompaniment file `docs/audio/ac4/ts_10319001v010401p0-tables.c`.
    The existing integer parser (round 20) bailed on the `float` keyword, so
    the five float-typed reference tables had been silently un-validated.
  - New `assert_float_table()` / `assert_complex_float_table()` helpers
    compare Rust `&[f32]` (and `&[(f32, f32)]`) constants against the
    parsed reference under a 1 ppm absolute / 10 ppm relative epsilon —
    tight enough to catch a single-digit transcription typo in the visible
    decimal prefix, loose enough that f32 literal-rounding noise is
    invisible.
  - Four new tests:
    - `etsi_source_parses_floats` — sanity-check that the 5 expected float
      arrays are parsed with the right lengths (20, 32, 256, 640, 1024).
    - `validate_ssf_float_tables` — `POST_GAIN_LUT[20]`,
      `PRED_GAIN_QUANT_TAB[32]`, `RANDOM_NOISE_TABLE[256]` against the
      Annex C.1 / C.3 / C.11 reference data.
    - `validate_qmf_window` — the 640-entry Annex D.3 `QWIN` prototype
      window against the reference.
    - `validate_aspx_noise_table` — the Annex D.2 `ASPX_NOISE[512][2]`
      complex-noise table against the reference (flattened row-major to
      compare `&[(f32, f32)]` against the parsed flat `Vec<f32>`).
  - Outcome: all 1,860 published float reference values
    (20 + 32 + 256 + 640 + 1,024) cleared the epsilon test on first run,
    proving the existing transcriptions in `ssf_tables.rs`, `qmf.rs`, and
    `aspx_noise.rs` are byte-correct against the ETSI accompaniment data.
    Future regressions caught at compile-time of the test target rather
    than at decode-time of a fixture.

- **Round 34 — FIXVAR / VARFIX / VARVAR atsg border derivation + SNF injection + 5_X ASPX_ACPL_3 synthesis** (TS 103 190-1 §5.7.6.3.3.2 + §5.1.4 + §5.7.7.6.2):
  - New `derive_fixvar_atsg()` (§5.7.6.3.3.2 Pseudocode 77, FIXVAR arm): builds
    the signal-envelope border vector right-to-left from `T - var_bord_right` using
    `rel_bord_right[]` deltas, then reverses to ascending order.
  - New `derive_varfix_atsg()` (Pseudocode 77, VARFIX arm): builds left-to-right
    from `var_bord_left` using `rel_bord_left[]` deltas, appends T.
  - New `derive_varvar_atsg()` (Pseudocode 77, VARVAR arm): left-side anchors +
    right-side internal anchors + T, totalling `num_env + 1` entries.
  - New `derive_atsg_borders()` dispatcher: routes FIXFIX / FIXVAR / VARFIX /
    VARVAR framing to the matching derivation and computes noise borders
    (`[0, T]` for 1 noise envelope, `[0, mid, T]` for 2).
  - Both the TNS path and envelope-adjustment path in `decoder.rs` now call
    `derive_atsg_borders` instead of the FIXFIX-only `derive_fixfix_atsg`, enabling
    A-SPX bandwidth extension for all four interval classes.
  - New `inject_snf_noise()` in `asf_data.rs` (§5.1.4 SNF): injects
    shaped noise (`gain = 2^((idx * 1.5 - 84) / 4)`) into zero-energy MDCT
    bins using a 16-bit LCG (multiplier 69069, addend 1). Previously the
    `parse_asf_snf_data()` result was discarded; now it is consumed by the
    long-mono ASF decode path.
  - 5_X `ASPX_ACPL_3` synthesis wired into `Ac4Decoder::receive_frame`:
    `Ac4Decoder` gains two new persistent fields (`acpl_5x_pair_state`,
    `acpl_5x_mch_state`); when the walker populates `acpl_config_2ch` +
    `acpl_data_2ch` + stereo carrier spectra, `run_acpl_5x_mch_pcm`
    (Pseudocode 118) runs and fills `pcm_per_channel[0..5]` with L/R/C/Ls/Rs.
  - Unit tests: `derive_fixvar_atsg` (2 cases + reject), `derive_varfix_atsg`
    (2 cases + reject), `derive_varvar_atsg` (2 cases), `derive_atsg_borders`
    dispatch (FIXFIX + FIXVAR), `inject_snf_noise` (fill, gain formula, LCG).

### Changed

- **`register` entry point unified on `RuntimeContext`** (task #502).
  The legacy `pub fn register(reg: &mut CodecRegistry)` is renamed to
  `register_codecs` and a new `pub fn register(ctx: &mut
  oxideav_core::RuntimeContext)` calls it internally. Breaking change
  for direct callers passing a `CodecRegistry`; switch to either the
  new `RuntimeContext` entry or the explicit `register_codecs` name.

## [0.0.4](https://github.com/OxideAV/oxideav-ac4/compare/v0.0.3...v0.0.4) - 2026-05-05

### Other

- land §5.2.5.2.2 Heuristic Scaling (round 33)
- env_prev tracking + walker state hoisting (round 32)
- land §5.2.3-5.2.7 PCM synthesis chain (round 31)
- land Tables 43-46 bitstream walker (round 30)
- land Annex C tables + arithmetic decoder core (round 29)

### Added

- **Round 33 — §5.2.5.2.2 Heuristic Scaling (Pseudocodes 27/28/29/30)**
  (TS 103 190-1 §5.2.5.2.0 selector + §5.2.5.2.2):
  - New `map_db_to_lin_q10()` (Pseudocode 29) and `map_lin_to_db_q10()`
    (Pseudocode 30) — Q.10 fixed-point dB↔linear converters using the
    Annex C.14 `SLOPES_DB_TO_LIN` / `OFFSETS_DB_TO_LIN` /
    `SLOPES_LIN_TO_DB` / `OFFSETS_LIN_TO_DB` LUTs (already shipped in
    `ssf_tables`). Out-of-range inputs clamp to the spec's `100 << 10`
    (dB→lin) and `40 << 10` (lin→dB) ceilings.
  - New `heuristic_scaling()` (Pseudocode 28) implements the full
    `HeuristicScaling(iRfu, env_in, ...) -> int_weights_dB[]` chain:
    dynamic-range compression on `env_in[]` when the spread exceeds
    the 40 Q.10 threshold, sort-descending of `env_local[]`,
    `Map_dB_to_Lin` per band, weighted sum scaled by `iRfu²`, reverse
    water-filling to find `iTCurrLev`, and a final per-band
    `Map_Lin_to_dB(iTCurrLev) - env_local[band]` weight that's clamped
    to `[0, 15 << 10]`.
  - New `apply_heuristic_scaling()` (Pseudocode 27) wraps Pseudocode 28
    with the `env_in = 3 * env_alloc` pre-multiply, the LF-boost
    threshold (`i_w_dB[0]` knocked down by 3), and the
    `env_alloc_mod = (env_alloc - i_w_dB).clamp(ENV_MIN, ENV_MAX)` +
    `f_gain_q = pow(10, 1.5 / 20 * f_w_dB)` post-processing. Returns
    `(env_alloc_mod[band], f_gain_q[band])`.
  - `synthesize_granule()` now dispatches the §5.2.5.2.0 selector:
    when `f_rfu > 0 && !variance_preserving` AND the SSF bandwidths
    table is available, the heuristic-scaling branch fires and
    `inverse_heuristic_scale()` consumes the resulting `f_gain_q[]`
    instead of the previous all-1 stub. The `variance_preserving`
    block also correctly skips the inverse-scale call per
    §5.2.5.2.0 step 5. Pre-r33 the synth crashed (well, silently
    bailed) on `f_pred_gain != 0` blocks; now they decode all the way
    through.
  - 11 new lib unit tests (470 → 481):
    `map_db_to_lin_zero_input` / `map_db_to_lin_out_of_range_clamps` /
    `map_db_to_lin_monotone_within_table` /
    `map_lin_to_db_zero_input` / `map_lin_to_db_out_of_range_clamps` /
    `heuristic_scaling_zero_envelope_yields_zero_weights` /
    `heuristic_scaling_clamps_to_max` /
    `apply_heuristic_scaling_short_circuits_on_empty` /
    `apply_heuristic_scaling_clamps_env_alloc_mod` /
    `synthesize_granule_runs_with_heuristic_scaling_branch` /
    `synthesize_granule_variance_preserving_skips_heuristic`.

- **Round 32 — SSF SHORT_STRIDE `env_prev` tracking + walker state
  hoisting** (TS 103 190-1 §5.2.3.0 Note 2, §5.2.3.0b Pseudocode 4b,
  §4.3.7.4.2 Pseudocodes 54-57):
  - `SsfSynthState` gains a new `env_prev: Vec<i32>` field that
    `synthesize_granule()` latches at the end of each granule with the
    *resolved* envelope (post-`decode_envelope` δ-chain), not the raw
    delta symbols. SHORT_STRIDE P-granules now use this latch as the
    `env_prev[]` interpolation input when the caller doesn't supply
    one — `interpolate_envelope` no longer degrades to a flat-zero
    envelope across frame boundaries on real P-frame streams.
    `Ac4Decoder::run_ssf_channel` drops its zero-vector
    `state_idx_env_prev` stub and passes an empty slice; the synth
    pulls from `state.env_prev` automatically.
  - `Ac4Decoder` adopts `Vec<SsfChannelState>` (one per channel,
    grown on demand) keyed `ssf_walker_state`. New
    `walk_ac4_substream_stateful()` and the matching
    `_stateful` variants of `parse_mono_audio_data_outer`,
    `parse_stereo_audio_data_outer`, `parse_stereo_data_body`, and
    `parse_aspx_acpl1_mdct_body` thread an
    `Option<&mut [SsfChannelState]>` through the SSF body parses so
    the walker's dither / noise RNGs (Pseudocodes 54-57) and
    `prev_pred_lag_idx` / `last_num_bands` / `env_prev` (raw symbol
    snapshot) persist across frames. The original public functions
    keep their pre-r32 signatures and delegate with `None` so the
    sibling repos / test fixtures stay binary-compatible.
  - 4 new lib unit tests (466 → 470):
    `synthesize_granule_latches_env_prev` (verifies `state.env_prev`
    holds the post-`decode_envelope` chain after each granule),
    `short_stride_p_frame_uses_state_env_prev` (proves a P-granule
    interpolates against the latched I-granule envelope, not zero),
    `synthesize_ssf_data_chains_env_prev_across_granules` (two-granule
    end-to-end), and
    `walk_ac4_substream_stateful_persists_ssf_walker_state`
    (round-trip through the substream walker leaves the channel-0
    state's `last_num_bands` / `last_n_mdct` / `env_prev` populated).

- **Round 31 — Speech Spectral Frontend (SSF) PCM synthesis chain** (TS
  103 190-1 §5.2.3 / §5.2.4 / §5.2.5 / §5.2.6 / §5.2.7 +
  §5.2.8.1 — Pseudocodes 4a / 4b / 4c / 4d / 4e / 26 / 31 / 32 / 33 /
  34 / 35 / 36 / 37 / 38 / 39):
  - New `ssf_synth` module turning the per-block indices on
    `crate::ssf::SsfData` into `n_mdct` spectral lines per block.
    Functions: `decode_envelope` (Pseudocode 4a — δ-decode chain over
    `env_curr[]`), `interpolate_envelope` (Pseudocode 4b — SHORT_STRIDE
    fixed-point linear interpolation between `env_prev[]` and the
    current granule's `env[]`), `decode_gains` (Pseudocode 4c —
    `pow(10, gain_idx * 0.1)` per block, LONG_STRIDE clamp to 1.0),
    `refine_envelope` (Pseudocode 4d — band-≥2 gain application + the
    `round(2 * gain_idx / 3)` allocation tweak with [-64, 63] clamp).
  - `decode_predictor` (Pseudocode 4e) reconstructs `f_pred_gain` from
    `PRED_GAIN_QUANT_TAB` and `f_pred_lag = 640 * 2^((idx - 509)/170)`,
    with `i_prev_pred_lag_idx` carried forward on `SsfSynthState`.
  - `compute_helpers` (Pseudocode 26 — `f_rfu`,
    `i_alloc_dithering_threshold`, adaptive noise gains) +
    `build_alloc_table` (Pseudocode 31 — no-rfu path:
    `env_alloc_mod = env_alloc`).
  - `inverse_quantize_block` (Pseudocode 32) implements all three
    branches: `i_alloc == 0` noise-RNG path with the
    variance-preserving `band > 1` branch, dithered branch via
    `Idx2Reconstruction` + `POST_GAIN_LUT[i_alloc - 1]` + the
    `f_post_gain_var_pres = sqrt(post_gain) * f_adaptive_noise_gain_var_pres`
    rule, and the no-dither MMSE branch via `mmse_laplace`
    (Pseudocode 33).
  - `inverse_heuristic_scale` (Pseudocode 34) — currently a no-op
    because the no-rfu path leaves `f_gain_q == 1`.
  - `build_c_matrix` (Pseudocode 39) reconstructs the per-`tab_idx`
    `(2*Rf+1, 65, Rt)` prediction-coefficient matrix from the
    quantized bytes in `crate::ssf_pred_coeff` using the
    `1.1787855 * (q - 146) / 128` reconstruction formula and the
    spec's `s = (-1)^(k+1)` mirror rule for negative-η.
  - `SubbandPredictorState::run` (Pseudocodes 35 / 36 / 37) maintains
    `f_spec_buffer[NUM_SPEC_BUF=5]` + `f_env_buffer[NUM_ENV_BUF=4]`
    histories, runs the model-based extractor (`f_period`, `k_s`,
    `tab_idx`, `Z`-matrix even-reflection, the per-bin
    `Σ_{ν,k} s * C[ν][f][k] * Z[bin+ν][k]` summation), then applies
    Pseudocode 37's per-band shaper (`f_envelope * f_pred_gain`)
    with the I-frame `integer_lag = 0` clamp.
  - `inverse_flatten` (Pseudocode 38) sums `f_spec_res + f_spec_pred`
    and multiplies by the per-band signal envelope.
  - `synthesize_granule()` runs the chain across every block in one
    granule; `synthesize_ssf_data()` runs it across both granules of
    one frame, threading `env_prev[]` between them.
  - `Ac4Decoder` adopts `Vec<SsfSynthState>` (one per channel) and
    consumes `tools.ssf_data_primary` / `tools.ssf_data_secondary`
    after the existing ASF/A-CPL pipeline: each granule's
    `num_blocks * n_mdct` spectrum is split per-block, fed into the
    per-channel KBD-windowed IMDCT + overlap-add, then truncated /
    padded to the frame's sample count. SSF substreams now emit real
    PCM instead of silence.
  - 16 new lib unit tests (450 → 466) covering: empty/empty-tail
    envelope decode, low-band-no-gain refinement, allocation-table
    clamping (min + max), zero-RFU + unit-window helpers, predictor
    presence/absence + delta-lag carry, full C-matrix dimensions
    across all 37 `tab_idx` values, the negative-η mirror rule, the
    subband-predictor zero-gain pass-through and finite-output smoke
    tests, the per-band envelope-gain inverse-flattening test, and a
    LONG_STRIDE I-granule synthesis end-to-end smoke (`synthesize_granule`
    on a synthetic granule). Plus one decoder-level integration test
    (`ssf_synth_long_stride_iframe_end_to_end`) that walks a synthetic
    SSF bitstream through `parse_ssf_data` then `synthesize_ssf_data`
    and verifies finiteness + zero-padding past `num_bins`.
  - Note: §5.2.5.2.2 Pseudocodes 27 / 28 / 29 / 30 (full Heuristic
    Scaling) are deferred — when `f_pred_gain == 0` the spec
    short-circuits to `env_alloc_mod = env_alloc` + `f_gain_q = 1`,
    which is the no-rfu path landed here. Synthesis of streams that
    enable the predictor across many bands at once with
    `variance_preserving == 0` will lose the heuristic envelope
    spreading until a follow-up round.

- **Round 30 — Speech Spectral Frontend (SSF) bitstream walker** (TS 103
  190-1 §4.2.9 / §4.3.7 + §4.3.7.5 + Tables 43-46 / 111-113):
  - New `ssf` module with the four-table walker family:
    `parse_ssf_data` (Table 43 — `b_ssf_iframe` gate plus 1 / 2
    granules per `frame_length >= 1536`), `parse_ssf_granule`
    (Table 44 — `stride_flag`, I-frame `num_bands_minus12`, per-block
    `predictor_presence_flag` / `delta_flag` loop), `parse_ssf_st_data`
    (Table 45 — `env_curr_band0_bits`, I-frame
    `env_startup_band0_bits`, per-block `gain_bits` /
    `predictor_lag(_delta)_bits` / `variance_preserving_flag` /
    `alloc_offset_bits`), and `parse_ssf_ac_data` (Table 46 — drives
    `decode_envelope_indices` Pseudocode 48 + `decode_predictor_gain`
    Pseudocode 49 + `decode_coefficient_indices` Pseudocode 50, then
    `AcDecodeFinish` Pseudocode 47 termination-bit accounting).
  - New `SsfFrameConfig` derives `(granule_length, num_granules,
    max_num_blocks)` per Tables 112-113 with both
    `from_toc(fs_index, frame_rate_index, frame_length)` and a
    `from_frame_len_base()` 48 kHz convenience overload. SHORT_STRIDE
    is rejected when `max_num_blocks < 1`.
  - New `SSF_BANDWIDTHS` matrix transcribes Annex C.1 verbatim
    (19 bands × 8 block-length columns); `SsfBinLayout::build()`
    implements §4.3.7.5 Pseudocode 7 to derive `start_bin[]` /
    `end_bin[]` / `num_bins` from `(num_bands, n_mdct)`.
  - New `SsfChannelState` carries forward dither / noise RNG state
    (reset per SSF-I-frame per Pseudocode 55), `prev_pred_lag_idx`
    (§5.2.4.0a), `last_num_bands` / `last_n_mdct` (inheritance for
    P-frame granules), and `env_prev[]` (§5.2.3.0).
  - Wired into `asf::walk_ac4_substream` for three call sites:
    `parse_mono_audio_data_outer` (mono SIMPLE / ASPX path —
    `spec_frontend == SSF` no longer returns `Unsupported`),
    the split-MDCT stereo `parse_stereo_data_body` (per-L/R SSF
    selection), and `parse_aspx_acpl1_mdct_body` split case (per-M/S
    SSF selection on the ACPL_1 residual layer). Parsed payload lands
    on `SubstreamTools::ssf_data_primary` /
    `ssf_data_secondary` slots.
  - **Bug fix**: `decode_envelope_indices` and `decode_predictor_gain`
    in `ssf_ac` previously capped at symbol 31 (`AcDecodeSymbolExtCdf
    (cdf, 0, 31)`); the spec's Pseudocodes 48 / 49 use `(cdf, 0, 32)`.
    The 33-entry CDF tables (`ENVELOPE_CDF_LUT`,
    `PREDICTOR_GAIN_CDF_LUT`) supply 32 symbol slots — the previous
    cap clipped the highest-probability tail symbol.
  - 6 new lib unit tests + 1 new integration test in `asf::tests`
    (449 → 463 total): stride-flag block count, Annex C.1 anchors,
    `SsfBinLayout::build` for 48 kHz / 24 fps LongStride, frame-config
    resolution for all five Table 112-113 row classes, end-to-end
    LongStride I-frame walk, end-to-end ShortStride I-frame walk
    (3 live blocks, env_startup populated), and a substream-level
    integration test (`mono_ssf_substream_walker_populates_ssf_data`)
    that builds a synthetic SSF substream + walks it through the
    public `walk_ac4_substream` API.

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

- **Round 29 — Speech Spectral Frontend (SSF) tables + arithmetic
  decoder core** (TS 103 190-1 §5.2.8 + Annex C):
  - New `ssf_tables` module — verbatim transcription of every Annex C
    scalar lookup table from the ETSI accompaniment file
    `docs/audio/ac4/ts_10319001v010401p0-tables.c`:
    `POST_GAIN_LUT` (C.1, 20 floats), `PRED_GAIN_QUANT_TAB` (C.3, 32),
    `PRED_RFS_TABLE` (C.4, 37 u8), `PRED_RTS_TABLE` (C.5, 37 u8), the
    full 705-entry `CDF_TABLE` (C.7), `PREDICTOR_GAIN_CDF_LUT` (C.8,
    33), `ENVELOPE_CDF_LUT` (C.9, 33), the 256-entry `DITHER_TABLE`
    (C.10, Q0.15) and `RANDOM_NOISE_TABLE` (C.11, float),
    `STEP_SIZES_Q4_15` (C.12, 21), `AC_COEFF_MAX_INDEX` (C.13, 21),
    and the four C.14 dB↔linear conversion LUTs (`SLOPES_DB_TO_LIN`,
    `OFFSETS_DB_TO_LIN`, `SLOPES_LIN_TO_DB`, `OFFSETS_LIN_TO_DB`)
    plus the `lin_to_db` / `db_to_lin` piecewise-linear helpers.
  - New `ssf_pred_coeff` module — all 37 SSF prediction-coefficient
    matrices from Annex C.6 (`SSF_PRED_COEFF_MAT0..36`, ~22 KB total
    addressable via `ssf_pred_coeff_mat(i)` with `SSF_PRED_MAT_DIMS`
    giving each matrix's `(rows, cols)` shape).
  - New `ssf_ac` module — full §5.2.8 binary arithmetic decoder
    (`AcState` from Pseudocode 42 with `init` / `decode_target` /
    `decode` / `decode_symbol_ext_cdf` / `decode_symbol_calc_cdf` /
    `decode_finish` mapping to Pseudocodes 43-47), the
    `Idx2Reconstruction` + `CdfEst` computed-CDF transform-coefficient
    path (Pseudocodes 51-53), three convenience entry points
    `decode_envelope_indices` / `decode_predictor_gain` /
    `decode_coefficient_indices` (Pseudocodes 48-50), and the
    `SsfRandGenState` random-number generator (Pseudocodes 54-57)
    backing both the dither sequence (`dither_value`) and the noise
    sequence (`random_noise_value`).
  - 26 new lib unit tests cover every table length + spec anchor +
    monotonicity / range invariants, plus AC-decoder smoke tests
    (init pulls 30 bits, renormalisation does not loop, `CdfEst` is
    monotone, `Idx2Reconstruction` is monotone in the index, etc.).
  - 2 new `etsi_table_validation` integration tests
    (`validate_ssf_scalar_tables` + `validate_ssf_pred_coeff_matrices`)
    assert byte-for-byte equality between the Rust constants and the
    canonical C accompaniment file.
  - The §5.2.8 SSF arithmetic-coded `ssf_ac_data()` bitstream walker
    (Tables 43-46) is the next round's scope — all building blocks
    are now in place.

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
