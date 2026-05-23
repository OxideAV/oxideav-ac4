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
> `channels == 7/8` (7.0/7.1) into the new walker. Round 28 lands the
> mono / stereo **short-frame `sf_data(ASF)` walker** per ETSI TS 103
> 190-1 §4.2.8.3-6 Tables 39-42: new spec-correct `_grouped` payload
> parsers in `asf_data.rs` (each with its own outer
> `for (g = 0; g < num_window_groups; g++)` loop, a *single* 8-bit
> `reference_scale_factor` at the head of `asf_scalefac_data()` with
> `first_scf_found` carrying across groups, and a *single* 1-bit
> `b_snf_data_exists` gate at the head of `asf_snf_data()`), plus
> `derive_per_group()` helpers that resolve per-group
> `(transf_length_idx, transform_length, max_sfb)` from `(ti, psy)`
> per Pseudocodes 2 / 3 / 5 (handling the `b_different_framing`
> half-frame split). New body decoders
> `decode_asf_grouped_mono_body[_with_max_sfb]()` and
> `decode_asf_grouped_stereo_joint_body()` (shared section, per-group
> `ms_used[g][sfb]`, inverse M/S) are wired into all four mono /
> stereo call sites: `parse_mono_audio_data_outer`,
> `parse_aspx_acpl2_mdct_body`, `parse_aspx_acpl1_mdct_body` (joint +
> split) and `parse_stereo_data_body` (joint + split). Real Dolby
> AC-4 mono / stereo streams using short-window sub-frames now
> decode end-to-end without bailing at the previous
> `num_window_groups != 1` guard. Round 29 lands the full **§5.2.8
> SSF arithmetic decoder + Annex C scalar inventory + 37
> prediction-coefficient matrices** — 705-entry `CDF_TABLE`,
> `PREDICTOR_GAIN_CDF_LUT`, `ENVELOPE_CDF_LUT`, `DITHER_TABLE` /
> `RANDOM_NOISE_TABLE`, `STEP_SIZES_Q4_15`, `AC_COEFF_MAX_INDEX`,
> the four C.14 dB↔linear LUTs, plus `AcState` (`init` /
> `decode_target` / `decode_symbol_ext_cdf` /
> `decode_symbol_calc_cdf` / `decode_finish` per Pseudocodes 41-47),
> the `Idx2Reconstruction + CdfEst` computed-CDF path (Pseudocodes
> 51-53), envelope / predictor-gain / coefficient convenience
> entry points (Pseudocodes 48-50), and the `SsfRandGenState`
> dither + noise RNG (Pseudocodes 54-57). Round 30 lands the
> **SSF bitstream walker** (`ssf::parse_ssf_data` /
> `parse_ssf_granule` / `parse_ssf_st_data` / `parse_ssf_ac_data`
> per Tables 43-46), the Annex C.1 SSF-bandwidths matrix
> (`SSF_BANDWIDTHS`, 19 bands × 8 block-length columns),
> `SsfBinLayout::build()` (Pseudocode 7 — `start_bin[]` /
> `end_bin[]` / `num_bins`), `SsfFrameConfig` (Tables 112-113), and
> the `SsfChannelState` carrying RNG / `prev_pred_lag_idx` /
> `last_num_bands` / `env_prev[]` across granules. Wired into
> `walk_ac4_substream` for mono SIMPLE/ASPX, split-MDCT stereo, and
> the ASPX_ACPL_1 split residual layer — `spec_frontend == SSF` no
> longer falls through silently. Round 31 lands the
> **§5.2.3-5.2.7 SSF PCM synthesis chain** in a new `ssf_synth`
> module: envelope decoder + predictor + helpers + lossless decode +
> inverse-quant + subband predictor + inverse-flattening
> (Pseudocodes 4a / 4b / 4c / 4d / 4e / 26 / 31 / 32 / 33 / 34 / 35 /
> 36 / 37 / 38) plus the C-matrix reconstruction (Pseudocode 39) for
> all 37 `tab_idx` values. `synthesize_ssf_data()` threads
> `env_prev[]` between granules. `Ac4Decoder` now carries a
> per-channel `Vec<SsfSynthState>` and consumes
> `tools.ssf_data_primary` / `tools.ssf_data_secondary` after the
> ASF/A-CPL pipeline: each granule's `num_blocks * n_mdct` spectrum
> is split per-block and IMDCT'd through the existing KBD
> overlap-add path. SSF substreams now emit real PCM in place of
> silence. §5.2.5.2.2 Heuristic Scaling (Pseudocodes 27 / 28 / 29 /
> 30) is deferred — the spec's `f_rfu == 0` short-circuit covers any
> block with the predictor disabled, which the current synth
> supports. Round 32 closes the SHORT_STRIDE P-frame correctness gap
> by adding `env_prev: Vec<i32>` to `SsfSynthState`:
> `synthesize_granule()` latches the *resolved* envelope (post-
> `decode_envelope` δ-chain) at the end of each granule so that a
> SHORT_STRIDE P-granule with no caller-supplied `env_prev[]`
> interpolates against the previous frame's envelope rather than a
> zero fallback (§5.2.3.0 Note 2). The walker side gets a parallel
> hoist: `Ac4Decoder` now owns a `Vec<SsfChannelState>`
> (`ssf_walker_state`) and a new `walk_ac4_substream_stateful()`
> threads it through the SSF body parses so dither / noise RNGs
> (Pseudocodes 54-57) and `prev_pred_lag_idx` / `last_num_bands`
> persist across frames — pre-r32 the walker built a fresh state
> per frame and dropped it. Round 33 lands **§5.2.5.2.2 Heuristic
> Scaling (Pseudocodes 27/28/29/30)** — the predictor-enabled
> spectrum-decoding branch the spec's `f_rfu == 0` short-circuit
> previously skipped. New `map_db_to_lin_q10()` / `map_lin_to_db_q10()`
> Q.10 fixed-point converters use the Annex C.14 LUTs;
> `heuristic_scaling()` runs the full Pseudocode 28 chain
> (dynamic-range compression of `env_in[]`, sorted-descending
> `Map_dB_to_Lin`, `iRfu²`-weighted reverse water-filling,
> `Map_Lin_to_dB`-driven per-band weight); `apply_heuristic_scaling()`
> wraps it with the Pseudocode 27 `env_in = 3 * env_alloc` pre-multiply,
> LF-boost, and `(env_alloc_mod, f_gain_q)` post-processing.
> `synthesize_granule()` dispatches the §5.2.5.2.0 selector — when
> `f_rfu > 0 && !variance_preserving` the heuristic-scaling branch
> fires and `inverse_heuristic_scale()` consumes the resulting
> `f_gain_q[]` instead of the all-1 stub; `variance_preserving` blocks
> correctly skip the inverse-scale call per §5.2.5.2.0 step 5. Round 34
> lands **FIXVAR / VARFIX / VARVAR atsg border derivation** (§5.7.6.3.3.2
> Pseudocode 77) — new `derive_fixvar_atsg()`, `derive_varfix_atsg()`,
> `derive_varvar_atsg()` and a unified `derive_atsg_borders()` dispatcher
> cover all four `aspx_int_class` values; the decoder's TNS and
> envelope-adjustment paths now route through `derive_atsg_borders`
> instead of the FIXFIX-only path, enabling A-SPX bandwidth extension
> for FIXVAR / VARFIX / VARVAR substreams. **§5.1.4 SNF injection**:
> `inject_snf_noise()` fills zero-energy MDCT bins using a 16-bit LCG
> (multiplier 69069, addend 1) and gain formula `2^((idx×1.5−84)/4)`;
> the long-mono ASF decode path now consumes `parse_asf_snf_data()` output
> instead of discarding it. **5_X ASPX_ACPL_3 wired in `Ac4Decoder`**:
> two new persistent state fields (`acpl_5x_mch_state` /
> `acpl_5x_pair_state`) are added; when `acpl_config_2ch` +
> `acpl_data_2ch` + stereo carrier spectra are present,
> `run_acpl_5x_mch_pcm()` (Pseudocode 118) fires and fills
> `pcm_per_channel[0..5]` with L/R/C/Ls/Rs surround PCM.
> Round 35 lands the **§4.2.4.4 EMDF payloads substream parser**
> (Table 18) plus the §4.2.14.14 `emdf_payload_config()` (Table 79) in a
> new `emdf` module — `parse_emdf_payloads_substream()` walks the
> while-loop until the `emdf_payload_id == 0` terminator, handles the
> `id == 31 → variable_bits(5)` extension, decodes the full
> `EmdfPayloadConfig` (sample-offset / duration / group-id / codecdata /
> discard / frame-aligned + create-/remove-duplicate / priority /
> proc_allowed gates per the Table 79 conditional tree), and captures
> each payload's `emdf_payload_byte[]` verbatim. Defensive caps
> (`MAX_EMDF_PAYLOADS = 64`, `MAX_EMDF_PAYLOAD_BYTES = 65 536`) bound
> malformed input. The outer `metadata::parse_metadata` walker now
> consumes the substream when `b_emdf_payloads_substream == 1` and
> surfaces it through `Metadata::emdf_payloads_substream` instead of
> erroring out with "not yet implemented" — real-bitstream metadata can
> now fully round-trip through the walker. Round 35 also lands the
> **§5.7.9.3.3 PCM gain application path**: `drc::drc_raw_to_linear()`
> maps a 7-bit `drc_gain[ch][sf][band]` value to its linear multiplier
> via `2^((raw-64)/6)`, `dialnorm_correction_linear()` resolves the
> `2^((Lout-Lin)/6)` dialnorm correction, and
> `drc::apply_drc_gains_to_pcm()` applies a parsed `DrcGains` (per
> channel-group, per subframe — multi-band averaged in the linear
> domain) to a planar `&mut [Vec<f32>]` PCM buffer with a
> `DrcChannelMap` (helpers for the `[L, R, C, LFE?, Ls, Rs]` 5_X
> layout and the wideband single-group mono/stereo case). DE walker
> hardened with three new edge-case tests covering EOF on truncation,
> non-I-frame without `prev_config`, and the `nr_channels == 0`
> degenerate case. Round 51 lands **stereo SIMPLE/ASF split-MDCT
> (Path A: 2× SCE) encoding** per ETSI TS 103 190-1 §5.3 + §4.2.6.3 —
> `Ac4ImsEncoder::encode_frame_pcm_stereo` accepts L+R PCM frames,
> runs the existing forward MDCT + scalefactor + DP-section + HCB1..11
> codebook-selection + SNF emission pipeline independently per
> channel, and emits a `b_enable_mdct_stereo_proc == 0` stereo CPE
> body the decoder reconstructs at ≥24.8 dB spectral SNR for both
> 440 Hz L+R (matched) and 440 Hz L + 660 Hz R (independent) tone
> fixtures. Round 52 lands the matching **joint M/S CPE (Path B,
> `b_enable_mdct_stereo_proc == 1`) encoder** per §5.3 + §4.2.6.3
> + §7.5: `encode_frame_pcm_stereo` now dispatches between Path A
> and Path B based on an energy-weighted per-SFB correlation rising
> above the 0.7 threshold. Path B emits one shared `asf_section_data`
> + two `asf_spectral_data` (M/S or L/R per band based on bit-cost
> comparison) + shared `asf_scalefac_data` + per-active-sfb `ms_used`
> flags + shared `asf_snf_data`. A frame-level "matched-channels"
> gate (S/M energy ratio < 0.15) bumps the M-channel q_target up to
> 16 to spend the bits saved on the silent / near-silent S residual
> on finer M quantisation. End-to-end SNR on the 440 Hz L=R matched
> fixture rises from round-51's 24.8 dB to 34.5 dB; the half-
> correlated 440 Hz amplitude-imbalance fixture clears ≥ 26 dB;
> independent 440 L + 660 R correctly routes via Path A and
> preserves the round-51 SNR floor. Round 74 lands the first
> **multichannel forward-analysis encoder**: a 5.0
> SIMPLE/Cfg3Five path (5 SCE, no LFE, no joint coding) per
> §4.2.6.6 Table 25 row `case SIMPLE: coding_config == 3` +
> §4.2.7.5 Table 29 (`five_channel_data()`). New
> `Ac4ImsEncoder::with_5_0()` flips the TOC channel_mode prefix
> to `0b1101` (4 b — Table 85 channel_mode 3), and
> `encode_frame_pcm_5_0(&[L, R, C, Ls, Rs])` runs the round-50
> forward pipeline (KBD-MDCT + DP-optimal sectioning + HCB1..11
> + SNF) independently per channel into a shared `sf_info(ASF,
> 0, 0)` header followed by `five_channel_info()` (identity SAP:
> `chel_matsel = 0` + 5x `chparam_info` with `sap_mode = 0`)
> and 5x `sf_data(ASF)` bodies. Decoder's existing
> `dispatch_5x_cfg3_simple_aspx` (round 39) consumes the body
> and emits 5-channel interleaved S16 PCM. End-to-end per-
> channel spectral SNR ≥ 20 dB on the independent-tone fixture
> (220/440/660/880/1100 Hz on L/R/C/Ls/Rs — measured
> L=24.5 / R=24.8 / C=25.0 / Ls=23.4 / Rs=27.4 dB), matching
> the round-51 stereo Path A SNR floor. Round 80 extends the
> multichannel encoder to **5.1 (Cfg3Five + LFE)** per §4.2.6.6
> Table 25 (`if (b_has_lfe) mono_data(1);`) + §4.2.8 (`sf_info_lfe()`
> Table 35 / Table 106 column 4 `n_msfbl_bits`):
> `Ac4ImsEncoder::encode_frame_pcm_5_1(&[L, R, C, Ls, Rs, LFE])`
> flips the TOC channel_mode prefix to `0b1110` (4 b — Table 85
> channel_mode 4) and `build_5_1_simple_asf_body_from_pcm_spectra`
> prepends an LFE `mono_data(1)` element
> (`b_long_frame = 1` + `sf_info_lfe()` with `max_sfb_lfe` capped to
> `n_msfbl_bits = 3` → 7 sfb / ≈350 Hz at tl = 1920) before the
> Cfg3Five `five_channel_data()` body. The decoder side gains a
> matching LFE PCM render in `Ac4Decoder::receive_frame`: when
> `channels == 6` (5.1) or `channels == 8` (7.1) and
> `tools.lfe_mono_data.scaled_spec` is present, the LFE spectrum is
> IMDCT'd into the trailing PCM slot (slot 5 for 5.1, slot 7 for
> 7.1). Non-LFE per-channel SNR matches the 5.0 numbers
> (L=24.5 / R=24.8 / C=25.0 / Ls=23.4 / Rs=27.4 dB ≥ 20 dB floor); a
> 60 Hz LFE tone round-trips to a non-silent reconstructed LFE
> channel. Round 91 extends the multichannel encoder to **7.1
> (3/4/0.1) SIMPLE/Cfg3Five + LFE** per §4.2.6.14 Table 33 + §4.2.7.4
> Table 26 (additional-channel `two_channel_data()`) + Table 88
> channel_mode 6 (`0b1111001`, 7 b):
> `Ac4ImsEncoder::with_7_1()` +
> `encode_frame_pcm_7_1(&[L, R, C, Ls, Rs, Lb, Rb, LFE])` emit a 7_X
> SIMPLE/Cfg3Five body whose inner `five_channel_data()` reuses the
> round-80 5.1 forward pipeline, followed by `b_use_sap_add_ch = 0`
> identity-SAP + `two_channel_data()` for the immersive Lb/Rb pair.
> The decoder side gains the inner 5-channel core render for the
> 7_X SIMPLE/Cfg3Five path (it previously parsed but never IMDCT'd
> slots 0..4 — only the round-39 additional-pair slots 5/6 and the
> round-80 LFE slot 7 were touched); slots 0..4 now route through
> `dispatch_5x_cfg3_simple_aspx` (the inner SCE shape is identical
> to the 5_X Cfg3Five case). Per-channel spectral SNR on the
> 220/440/660/880/1100/1320/1540 Hz independent-tone 7.1 fixture:
> L=24.5 / R=24.8 / C=25.0 / Ls=23.4 / Rs=27.4 / Lb=25.4 / Rb=26.0 dB
> — all above the ≥ 20 dB floor. Round 95 lands the **5_X
> SIMPLE/ASPX_ACPL_3 multichannel encoder path** per §4.2.6.6
> Table 25 row `case ASPX_ACPL_3:` — symmetric counterpart to the
> round-34 decoder ACPL_3 walker (5a58f6a).
> `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3(&[L, R, C])` and
> `encode_frame_pcm_5_1_acpl3(&[L, R, C, LFE])` emit IMS v2 frames
> with `5_X_codec_mode = 4` (ASPX_ACPL_3). The new `encoder_acpl3`
> module ships bit-exact emitters for `aspx_config()` (Table 50,
> 15 b), `acpl_config_2ch()` (Table 60, 4 b), `companding_control(2)`
> (Table 49, 2 b), plus minimum-bit-cost zero-delta Huffman writers
> covering all 18 ASPX HCBs (Annex A.2 Tables A.16-A.33) and all 24
> ACPL HCBs (Annex A.3 Tables A.34-A.57) — `pick_zero_delta_cw`
> picks the entry at `index == cb_off` (zero delta for DF/DT) and
> `pick_min_len_cw` picks the smallest-length entry (used for F0
> seeds). The body layout is `5_X_codec_mode = 4 (3 b)` +
> I-frame `aspx_config() + acpl_config_2ch()` + optional LFE
> `mono_data(b_lfe = 1)` + `companding_control(2)` + `stereo_data()`
> (split-MDCT path, two carrier channels) + `aspx_data_2ch()`
> (FIXFIX num_env=1, balance=1, all-FREQ deltas) + `acpl_data_2ch()`
> (1 param set, 7 param bands, 11 EC streams with `diff_type = 0`
> and zero-delta DF). Decoder round-trip: 5.0 ACPL_3 → 5-channel
> S16 interleaved PCM (1920 samples × 5 ch × 2 bytes); 5.1
> ACPL_3 → 6-channel S16 (with LFE slot 5). The decoder walks
> the full Table 25 body and resolves `five_x_mode == AspxAcpl3`
> + `acpl_config_2ch.is_some() && acpl_data_2ch.is_some()`. The
> 5-channel `[L, R, C, Ls, Rs]` synthesis runs via
> `acpl_synth::run_acpl_5x_mch_pcm` (Pseudocode 118) — with
> all-zero ACPL parameter deltas Ls/Rs collapses to ducker-driven
> reconstruction from the L/R carriers (non-silent in the general
> case). Total tests 691 (was 680). Real per-band
> `(alpha, beta, gamma)` parameter extraction (replacing the
> zero-delta scaffold), real ASPX envelope coding, and matching
> encoder paths for `5_X_codec_mode in {ASPX_ACPL_1, ASPX_ACPL_2}`
> (Pseudocode 117) remain deferred. The 7_X paths inherit the
> same `aspx_data` / `acpl_data` shape as 5_X so they're queued
> behind the 5_X work continuing to harden. Round 100 lands the
> **5_X SIMPLE/ASPX_ACPL_2 multichannel encoder path** per §4.2.6.6
> Table 25 row `case ASPX_ACPL_2:` — the symmetric counterpart to the
> round-25 decoder `parse_aspx_acpl_1_2_inner_body` walker (Pseudocode
> 117). `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl2(&[L, R, C])` emits an
> IMS v2 frame with `5_X_codec_mode = ASPX_ACPL_2 (3)` whose body is
> `aspx_config() + acpl_config_1ch(FULL) + companding_control(3) +
> coding_config = 0 + two_channel_data() (L/R carriers) +
> mono_data(0) (centre) + aspx_data_2ch() + aspx_data_1ch() + two
> acpl_data_1ch()` parameter sets. The ASPX_ACPL_1-only joint-MDCT
> residual layer (`max_sfb_master + 2× chparam_info + 2× sf_data`) is
> skipped for ACPL_2 — that's the structural difference that makes the
> ACPL_2 path the cleanest encoder win. New `encoder_acpl3` emitters:
> `write_acpl_config_1ch_full` (Table 59, 3 b), `write_two_channel_data`
> (Table 26 — shared `sf_info(ASF)` + identity-SAP `chparam_info` +
> 2× `sf_data`), `write_mono_data_centre` (Table 21 non-LFE),
> `write_aspx_data_1ch_minimal` (Table 51 FIXFIX num_env=1) and
> `write_acpl_data_1ch_minimal` (Table 61). The 1ch ASPX SIGNAL band
> count uses `num_sbg_sig_highres` to match `parse_aspx_ec_data`'s
> empty-`freq_res` fallback. Decoder round-trip: 5.0 ACPL_2 →
> 5-channel S16 interleaved PCM; the decoder resolves
> `five_x_mode == AspxAcpl2`, walks `acpl_config_1ch_full`,
> `two_channel_data`, the Cfg0 centre `mono_data(0)`, and both
> `acpl_data_1ch_pair` entries, then synthesises `[L, R, C, Ls, Rs]`
> via `acpl_synth::run_acpl_5x_pair_pcm`. Total tests 700 (was 691).
> The ASPX_ACPL_1 encoder path (joint-MDCT residual + PARTIAL-mode
> `acpl_config_1ch` with `acpl_qmf_band`), real `(alpha, beta)`
> parameter extraction, and the 7_X ASPX_ACPL_{1,2} encoder paths
> remain deferred. Round 103 lands the **5_X SIMPLE/ASPX_ACPL_1
> multichannel encoder path** per §4.2.6.6 Table 25 row
> `case ASPX_ACPL_1:` — the round-100 follow-up and the symmetric
> counterpart to the decoder's round-25 `parse_aspx_acpl_1_2_inner_body`
> ASPX_ACPL_1 branch (Pseudocode 117).
> `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1(&[L, R, C, Ls, Rs])` emits
> an IMS v2 frame with `5_X_codec_mode = ASPX_ACPL_1 (2)`. The body
> differs from the ACPL_2 path in two structural places:
> (1) `acpl_config_1ch` is PARTIAL — `write_acpl_config_1ch_partial`
> (Table 59, 6 b: id + quant_mode + `acpl_qmf_band_minus1`), so the
> `acpl_data_1ch()` start_band resolves from `qmf_band` via `sb_to_pb`;
> (2) the body carries an explicit joint-MDCT residual layer —
> `write_acpl_1_residual_layer` emits `max_sfb_master` (n_side bits) +
> 2× identity-SAP `chparam_info` + 2× `sf_data(ASF)` for the Ls/Rs
> surround pair (sSMP,3 / sSMP,4 per Table 181), so the encoder takes a
> full 5-channel input instead of reconstructing Ls/Rs from the L/R
> carriers. New `build_5_x_acpl1_body_from_pcm_spectra` lays out
> `5_X_codec_mode = 2 + aspx_config() + acpl_config_1ch(PARTIAL) +
> companding_control(3) + coding_config = 0 + two_channel_data() +
> residual-layer + mono_data(0) + aspx_data_2ch() + aspx_data_1ch() +
> 2× acpl_data_1ch()`. Decoder round-trip: 5.0 ACPL_1 → 5-channel S16
> interleaved PCM; the decoder resolves `five_x_mode == AspxAcpl1`,
> the PARTIAL config, the persisted residual pair, the Cfg0 centre, and
> both `acpl_data_1ch_pair` entries, then synthesises `[L, R, C, Ls, Rs]`
> via `acpl_synth::run_acpl_5x_pair_pcm`. Total tests 708 (was 700).
> Real per-band `(alpha, beta)` extraction, real ASPX envelope coding,
> real Table-181 SAP-derived residual content (the residual sf_data
> currently codes the raw Ls/Rs spectra), and the 7_X ASPX_ACPL_{1,2}
> encoder paths remain deferred.

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
- EMDF payload bodies — the outer `emdf_payloads_substream()` walker
  (Table 18) and `emdf_payload_config()` (Table 79) are parsed but
  the per-payload `emdf_payload_byte[]` opaque sequence is captured
  as raw bytes; per-`emdf_payload_id` semantic interpretation lives
  in the AC-4 EMDF datatype registry [i.14] and is out of scope for
  the present document.

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
