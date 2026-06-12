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
> encoder paths remain deferred. Round 107 lands the first of those
> deferred 7_X encoder paths: the **7.0 SIMPLE/ASPX_ACPL_2 multichannel
> encoder** per §4.2.6.14 Table 33 row `case ASPX_ACPL_2:` — the 7_X
> (immersive) counterpart to the round-100 5_X ASPX_ACPL_2 path and the
> encoder side of the decoder's round-27 `parse_7x_audio_data_outer`
> ASPX_ACPL_2 branch. `Ac4ImsEncoder::encode_frame_pcm_7_0_acpl2(&[L, R,
> C, Ls, Rs, Lb, Rb])` emits an IMS v2 frame with `7_X_codec_mode =
> ASPX_ACPL_2 (3)` and channel_mode prefix `0b1111000` (7 b — Table 85
> channel_mode 5, 7.0 (3/4/0)). The new
> `encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra` reuses the same
> 1ch ACPL / ASPX emitters as the 5_X ACPL_2 path but lays out the 7_X
> channel element's distinct framing: 2-bit `7_X_codec_mode` (vs the 5_X
> 3-bit field), `companding_control(5)` (sync-on 2-bit wire shape), 2-bit
> `coding_config = 0` (Cfg0), `b_2ch_mode + two_channel_data() (L/R) +
> two_channel_data() (Ls/Rs)`, a trailing centre `mono_data(0)` moved out
> of the coding_config switch, and an `aspx_data_2ch() + aspx_data_2ch() +
> aspx_data_1ch()` envelope trailer before the two `acpl_data_1ch()`
> parameter sets (Pseudocode 117 D0/D1). The ASPX_ACPL_1-only joint-MDCT
> residual layer and the SIMPLE/ASPX additional-channel block are skipped
> for ACPL_2. Decoder round-trip: 7.0 ACPL_2 → 7-channel S16 interleaved
> PCM (1920 samples × 7 ch × 2 bytes); the decoder resolves
> `seven_x_mode == AspxAcpl2`, both `two_channel_data` pairs, the Cfg0
> centre, and both `acpl_data_1ch_pair` entries, then synthesises
> `[L, R, C, Ls, Rs]` (slots 0..4) via `acpl_synth::run_acpl_5x_pair_pcm`
> (the back pair Lb/Rb stays silent per the Table 202 ACPL_2 mapping).
> Total tests 714 (was 708). The 7.1 (LFE) ASPX_ACPL_2 path, the 7_X
> ASPX_ACPL_1 path (PARTIAL config + joint-MDCT residual), real per-band
> `(alpha, beta)` extraction, real ASPX envelope coding, and back-pair
> Lb/Rb carriage remain deferred. Round 114 closes the first of those
> deferred items — the **7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_2 multichannel
> encoder path** per §4.2.6.14 Table 33 row `case ASPX_ACPL_2:` with
> `b_has_lfe = 1` + §4.2.6.5 Table 21 (`mono_data(b_lfe)`) + §4.2.8
> Table 35 (`sf_info_lfe()`). `Ac4ImsEncoder::encode_frame_pcm_7_1_acpl2(&[L,
> R, C, Ls, Rs, Lb, Rb, LFE])` emits the round-107 7.0 ASPX_ACPL_2 body
> plus a leading `mono_data(b_lfe = 1)` element between the I-frame config
> block and `companding_control(5)` — exactly where the decoder's
> `parse_7x_audio_data_outer(b_has_lfe = true)` reads
> `if (b_has_lfe) mono_data(1);`. The channel_mode prefix is forced to
> `0b1111001` (7 b — Table 88 channel_mode 6) so the decoder dispatches
> `channels == 8`. `build_7_x_acpl2_body_from_pcm_spectra` gained
> `max_sfb_lfe: Option<u32>` + `coeffs_lfe: Option<&[f32]>` and reuses the
> shared round-80 `write_lfe_mono_data` emitter (`max_sfb_lfe` capped to
> `n_msfbl_bits = 3` → 7 sfb at `tl = 1920`). Decoder round-trip: 7.1
> ACPL_2 → 8-channel S16 interleaved PCM (1920 samples × 8 ch × 2 bytes);
> the LFE spectrum IMDCT's into slot 7 via the round-80 LFE render, the
> `[L, R, C, Ls, Rs]` slots 0..4 synthesis is unchanged, and the back pair
> Lb/Rb (slots 5/6) stays silent per the Table 202 ACPL_2 mapping. A 60 Hz
> LFE tone round-trips to a non-silent reconstructed LFE channel. Total
> tests 721 (was 714). The 7_X ASPX_ACPL_1 path (PARTIAL config +
> joint-MDCT residual), real per-band `(alpha, beta)` extraction, real
> ASPX envelope coding, and back-pair Lb/Rb carriage remain deferred.
> Round 118 closes the first of those — the **7.0 / 7.1 (3/4/0(.1))
> SIMPLE/ASPX_ACPL_1 multichannel encoder path** per §4.2.6.14 Table 33 row
> `case ASPX_ACPL_1:` (the 7_X analogue of the round-103 5_X ACPL_1 path).
> `Ac4ImsEncoder::encode_frame_pcm_7_0_acpl1(&[L, R, C, Ls, Rs, Lb, Rb])`
> and `encode_frame_pcm_7_1_acpl1(&[.., LFE])` emit IMS v2 frames with
> `7_X_codec_mode = ASPX_ACPL_1 (2)`. New
> `encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra` is the round-107/114
> 7_X ACPL_2 body with the three ACPL_1 differences: `7_X_codec_mode = 2`
> (not 3), `acpl_config_1ch` PARTIAL (`write_acpl_config_1ch_partial`,
> carries `acpl_qmf_band_minus1` → `acpl_data_1ch()` start_band via
> `sb_to_pb`), and an explicit joint-MDCT residual layer
> (`write_acpl_1_residual_layer`: `max_sfb_master + 2× chparam_info +
> 2× sf_data(ASF)` for the Ls/Rs surround pair sSMP,3 / sSMP,4 per Table
> 181) after the two `two_channel_data()` pairs and before the trailing
> Cfg0 centre `mono_data(0)`. The 7.1 form prepends the round-80
> `write_lfe_mono_data` LFE element. Decoder round-trip: 7.0 ACPL_1 →
> 7-channel S16 PCM; 7.1 ACPL_1 → 8-channel S16 (LFE IMDCT'd into slot 7);
> `[L, R, C, Ls, Rs]` synthesise via `acpl_synth::run_acpl_5x_pair_pcm`,
> Lb/Rb (slots 5/6) silent per Table 202. Total tests 729 (was 721). Real
> per-band `(alpha, beta)` extraction, real ASPX envelope coding, real
> Table-181 SAP-derived residual content, and back-pair Lb/Rb carriage
> remain deferred. Round 125 lands the **7.0 (3/4/0) SIMPLE/Cfg3Five
> multichannel encoder path** per ETSI TS 103 190-1 §4.2.6.14 Table 33 +
> §4.2.7.5 Table 29 (`five_channel_data()`) + §4.2.7.4 Table 26
> (additional-channel `two_channel_data()`) — the non-LFE immersive
> counterpart of round-91's 7.1 encoder (the 7_X analogue of round 74's
> 5.0 vs round 80's 5.1). `Ac4ImsEncoder::with_7_0()` flips the TOC
> channel_mode prefix to `0b1111000` (7 b — Table 85 channel_mode 5,
> 7.0 (3/4/0) → 7 channels), and
> `encode_frame_pcm_7_0(&[L, R, C, Ls, Rs, Lb, Rb])` (+
> `..._with_max_sfb(.., max_sfb, max_sfb_add)`) emits a SIMPLE/Cfg3Five
> 7_X channel-element body whose inner `five_channel_data()` reuses the
> round-80 5.1 forward pipeline for the L/R/C/Ls/Rs front/surround pair
> and whose trailing identity-SAP `two_channel_data()` (`b_use_sap_add_ch
> = 0`) carries the immersive Lb/Rb pair. The body is structurally the
> round-91 7.1 body with the leading `mono_data(b_lfe = 1)` element
> omitted (the walker's `if (b_has_lfe) mono_data(1);` branch is gated
> off for channel_mode 5). New `encoder_asf::build_7_0_simple_asf_body_from_pcm_spectra`
> emits the body bytes; decoder round-trip: 7.0 → 7-channel S16
> interleaved PCM (1920 samples × 7 ch × 2 bytes). The 7.0 walker
> resolves `seven_x_mode == Simple`, `seven_x_b_has_lfe == false`,
> `five_channel_data` populated, identity-SAP additional-channel pair
> populated (slots 5/6 = Lb/Rb routed via
> `dispatch_7x_additional_channel_pair`). Per-channel spectral SNR on
> the 220/440/660/880/1100/1320/1540 Hz independent-tone 7.0 fixture:
> L=24.5 / R=24.8 / C=25.0 / Ls=23.4 / Rs=27.4 / Lb=25.4 / Rb=26.0 dB —
> all above the ≥ 20 dB floor. Total tests 737 (was 729). Round 128
> lands the first **real per-parameter-band α extraction** in the
> ACPL_1 5.0 encoder per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 +
> §5.7.7.6.1 Pseudocode 117 — replaces the round-103 zero-delta α
> scaffold (β / β3 / γ stay at the round-95 / 100 / 103 scaffold; β3 /
> γ only fire in ASPX_ACPL_3 anyway). With β = 0 the surround
> reconstruction is `Ls_recon = 0.5/√2 · L · (1 − α)`; solving for α
> per parameter band gives the closed form `α = 1 − 2·√2 · ⟨L, Ls⟩ /
> ⟨L, L⟩`. New `encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_real_alpha`
> + helper chain (`compute_per_band_correlations` mapping MDCT bins →
> QMF subbands → A-CPL parameter bands via §5.7.7.2 Table 197,
> `analytic_alpha_per_band` + `quantise_alpha` against Tables 203 /
> 205, then `write_acpl_alpha_f0_value` / `write_acpl_alpha_df_value`
> emit the ALPHA F0 + DF codewords per Tables A.35 / A.34); new
> `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_real_alpha` entry point
> alongside the round-103 zero-delta variant. The on-wire body
> structure is unchanged — decoder resolves
> `FiveXCodecMode::AspxAcpl1`, both `acpl_data_1ch_pair[0/1]`
> populated, joint-MDCT residual layer walked,
> `[L, R, C, Ls, Rs]` synthesised via
> `acpl_synth::run_acpl_5x_pair_pcm`. Total tests 743 (was 737).
> Round 132 adds the first **real per-parameter-band β extraction** in
> the ACPL_1 5.0 encoder per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 +
> §5.7.7.6.1 Pseudocode 117 — replaces the round-95/100/103/128 zero-β
> scaffold for the ACPL_1 path. With the decorrelator output `y` ⊥ `x0`
> and `E[y²] ≈ E[x0²]`, the surround energy balance is
> `E[Ls²] = 0.5·E[x0²]·((1−α)² + β²)`, so the per-band β magnitude is
> `β = √max(0, 2·E[Ls²]/E[x0²] − (1−α_dq)²)`. New
> `encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta`
> + helper chain (`compute_per_band_energies`, `analytic_beta_per_band`,
> `quantise_beta_magnitude` against Tables 204/206, `write_acpl_beta_f0_value`
> / `write_acpl_beta_df_value` per Tables A.40/A.41) and the new
> `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_real_alpha_beta` entry
> point. The on-wire body structure is unchanged; the β coding contract
> round-trips byte-exact through `acpl::parse_acpl_data_1ch`. Total tests
> 755 (was 743). Real β extraction for the 7_X / ACPL_2 / ACPL_3 paths,
> real γ extraction, and the round-128 ALPHA-writer negative-`alpha_q`
> desync fix remain deferred. Round 135 extends the **real per-band α + β
> extraction to the 7_X (7.0 immersive) ASPX_ACPL_1 path** — the round-132
> followup. New `encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta`
> (the real-α+β upgrade of the round-118 zero-delta 7_X builder, reusing the
> `extract_alpha_q_per_band` / `extract_beta_q_per_band` primitives) and the
> `Ac4ImsEncoder::encode_frame_pcm_7_0_acpl1_real_alpha_beta` entry point;
> both trailing `acpl_data_1ch()` sets now carry real α + β. The on-wire
> body structure is unchanged — decoder resolves `SevenXCodecMode::AspxAcpl1`
> (`b_has_lfe = false`), both `acpl_data_1ch_pair[0/1]` populated, joint-MDCT
> residual layer walked. Total tests 760 (was 755). Round 139 extends the
> **real per-band α + β extraction to the 7.1-with-LFE (3/4/0.1)
> ASPX_ACPL_1 path** — the round-135 LFE follow-up. New
> `Ac4ImsEncoder::encode_frame_pcm_7_1_acpl1_real_alpha_beta` (and the
> `_with_max_sfb` form) reuses the round-135
> `build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta` builder with the
> LFE `coeffs_lfe` + `max_sfb_lfe` slots populated, emitting a leading
> `mono_data(b_lfe = 1)` element (Table 21 + `sf_info_lfe()` Table 35)
> between the I-frame config block and `companding_control(5)` exactly
> where the decoder's `parse_7x_audio_data_outer(b_has_lfe = true)` reads
> `if (b_has_lfe) mono_data(1);`. The on-wire body structure matches the
> existing round-118 7.1 ACPL_1 path — decoder resolves
> `SevenXCodecMode::AspxAcpl1` with `b_has_lfe = true`, both
> `acpl_data_1ch_pair[0/1]` populated (now carrying real α + β),
> joint-MDCT residual layer walked, LFE IMDCT'd into slot 7. A 60 Hz
> LFE tone round-trips to a non-silent reconstructed LFE channel. Total
> tests 766 (was 760). Real β extraction for the ACPL_2 / ACPL_3 paths
> and the round-128 ALPHA-writer negative-`alpha_q` desync fix remain
> deferred. Round 144 closes the ACPL_2 5.0 half of the deferred list
> with the **5_X SIMPLE/ASPX_ACPL_2 encoder with real per-parameter-band
> α + β extraction** per §4.2.6.6 Table 25 row `case ASPX_ACPL_2:` +
> §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode 117. New
> `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl2_real_alpha_beta` (and the
> `_with_max_sfb` form) accepts a 5-channel `[L, R, C, Ls, Rs]` input and
> produces a 5_X ASPX_ACPL_2 frame whose two trailing `acpl_data_1ch()`
> elements carry real per-band α + β indices extracted from the (L, Ls)
> and (R, Rs) MDCT energy ratios via the round-128 / 132 shared analytic
> primitives. The on-wire body layout matches the round-100
> `build_5_x_acpl2_body_from_pcm_spectra` schedule (no joint-MDCT
> residual layer — ACPL_2 reconstructs the surround from L/R + the two
> `acpl_data_1ch()` parameter sets at decode time); the Ls/Rs spectra
> are consumed only by the α + β extractors and are not transmitted.
> `acpl_config_1ch(FULL)` carries no `qmf_band` → `start_band = 0` so
> every parameter band participates in the α + β coding (in contrast to
> the ACPL_1 PARTIAL mode whose `acpl_qmf_band` masks the low bands).
> Total tests 773 (was 766). Real β extraction for the ACPL_3 paths and
> the round-128 ALPHA-writer negative-`alpha_q` desync fix (which
> currently obscures per-band on-wire α/β recovery through the full
> PCM→MDCT→writer→parser→synth chain when the analytic α quantises to a
> non-center lane) remain deferred. Round 174 closes the desync fix at
> the codebook contract level: ALPHA / BETA3 **F0** `cb_off` corrected
> per §A.3 Tables A.34 / A.35 / A.46 / A.47. Pre-fix `cb_off = 0`
> conflicted with the §5.7.7.7 Pseudocode 121 differential-decoder /
> [`acpl_synth::dequantize_alpha_index`] signed-lane contract; the
> ALPHA F0 codebooks (17-entry Coarse / 33-entry Fine) are symmetric
> around the centre with a 1-bit peak at the `alpha_q = 0` lane, so
> `cb_off = N/2` (8 Coarse / 16 Fine) is the right offset to read
> back signed `alpha_q ∈ [-N/2, +N/2]` directly from `decode_delta`.
> The fix lands in both [`acpl::get_acpl_hcb`] (decoder side) and the
> encoder's local [`encoder_acpl3::acpl_hcb_arrays`] mirror, plus the
> symmetric BETA3 F0 codebooks (`cb_off = 4` Coarse / `8` Fine — the
> §5.7.7.7 `dequantize_beta3` multiplies the signed lane by
> `beta3_delta(qm)` directly so they share ALPHA's signed convention).
> BETA F0 stays at `cb_off = 0` (unsigned magnitude — Table 204 / 206
> stores positive entries only and `dequantize_beta_index` takes
> `unsigned_abs` then re-applies the sign carried in by the differential
> accumulator). Three new unit tests
> ([`alpha_f0_signed_lanes_round_trip_fine_and_coarse`],
> [`beta3_f0_signed_lanes_round_trip_fine_and_coarse`],
> [`alpha_f0_zero_alpha_picks_one_bit_peak`]) sweep every signed lane
> through encode → `decode_delta` and confirm the writer now picks the
> 1-bit symmetric peak for `alpha_q = 0` (down from 10 / 12 bits
> pre-fix). The round-128 family
> (`encode_5_0_acpl1_real_alpha_emits_nonzero_alpha_when_surround_differs`,
> `..._symmetric_scaling_yields_matching_alpha`) is re-shaped to assert
> on encoder byte-stream divergence rather than the decoder's recovered
> `alpha_q` — bit-position drift through the full 5_X SIMPLE/ASPX_ACPL_1
> walker on non-silence input is independent of the F0 cb_off bug and
> still pending separate investigation (it manifests as a misalignment
> upstream of `parse_acpl_data_1ch`, not in the ACPL F0 codeword
> itself). Total tests 776 (was 773). Round 181 closes the deferred
> "alpha_q desync" follow-up at two distinct spec-alignment layers.
> Layer 1: [`acpl::parse_acpl_huff_data`] now returns a
> spec-indexed length-`num_param_bands` vector matching §5.7.7.7
> Pseudocode 121's `acpl_<SET>[ps][i]` shape (positions
> `[0..start_band)` are zero, F0 lands at `values[start_band]`,
> DF deltas occupy `[start_band+1..num_param_bands)`); pre-r181's
> packed `(num_bands - start_band)`-length layout silently shifted
> the DIFF_FREQ accumulation for the PARTIAL `acpl_config_1ch` path.
> Layer 2: [`encoder_acpl3::write_aspx_data_2ch_minimal`] now keys
> the SIGNAL ec_data band count off `cfg.signals_freq_res()` per
> §4.3.10.4.9 (Table 124 NOTE 3) — when the encoder doesn't emit an
> in-band `aspx_freq_res` bit, the parser's high-res fallback
> selects `num_sbg_sig_highres` and the writer must match
> (pre-r181 it hard-coded `num_sbg_sig_lowres`, causing a 20-vs-10
> SIGNAL desync that buried every subsequent `acpl_data_1ch()`
> α/β codeword in trailing zero-padding). End-to-end 5.0 ASPX_ACPL_2
> asymmetric L/Ls input now recovers a non-zero per-band `alpha_q`
> row on `acpl_data_1ch_pair[0/1]` through the full PCM →
> MDCT → encode → AC-4 walker → `differential_decode` chain. Total
> tests 780 (was 776). The ASPX_ACPL_1 path retains a separate
> joint-MDCT residual-layer alignment issue between
> [`encoder_acpl3::write_acpl_1_residual_layer`] and the decoder's
> `parse_aspx_acpl_1_2_inner_body` residual-pair walker — tracked
> as the remaining follow-up. Round 187 pins that follow-up with four
> end-to-end characterisation tests (silence / L-only / Ls-only /
> combined) that sweep
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_real_alpha_beta`]
> and assert each pair slot's recovered
> `acpl_data_1ch_pair[0/1].framing.num_param_sets` so the next round
> can iterate on the residual / α-β writers without regressing the
> aligned silence / L-only / Ls-only paths. The diagnostic narrative
> in `tests/round187_acpl1_residual_desync_characterization.rs`
> triangulates the drift surface: writer→parser pairs for
> [`encoder_acpl3::write_acpl_data_1ch_real_alpha_beta_bytes`] ↔
> [`acpl::parse_acpl_data_1ch`] are bit-exact in isolation (already
> pinned by `round181_alpha_desync_fix::standalone_*`), so the
> remaining drift sits upstream of pair0 — either in
> `write_acpl_1_residual_layer` vs the inline residual walk inside
> `parse_aspx_acpl_1_2_inner_body`'s ASPX_ACPL_1 branch, or in
> `write_two_channel_data` vs `parse_two_channel_data` — when L and
> Ls are simultaneously non-trivial. Total tests 784 (was 780).
> Round 190 closes the desync at the root cause: the two minimal
> A-SPX writers
> ([`encoder_acpl3::write_aspx_data_2ch_minimal`] and
> [`encoder_acpl3::write_aspx_data_1ch_minimal`]) emitted
> `aspx_int_class = FIXFIX` as the wrong prefix code: `0b11` (2 bits)
> instead of `0b0` (1 bit) per ETSI TS 103 190-1 Table 126. The
> decoder's [`aspx::AspxIntClass::read`] walks the prefix correctly
> (`0` → FixFix, `10` → FixVar, `110` / `111` → VarFix / VarVar), so
> the writer's `11` start drove the parser into the VarFix branch
> with `b_iframe = 1` and Note-1 2-bit width
> (`num_aspx_timeslots = 15 > 8`): `var_bord_left` (2 b) +
> `num_rel_left` (2 b) + `tsg_ptr` (2 b) — parser consumed **9 bits**
> in framing where the writer emitted only **3**. The 6-bit drift was
> masked in the silence / L-only / Ls-only paths (α / β quantised to
> 0 ⇒ constant minimum-cost `acpl_data_1ch` bodies whose
> `num_param_sets_cod` bit positions sampled `0` on both sides), but
> with non-zero α / β the codewords shifted and the pair-1
> `num_param_sets_cod` bit position landed on a `1` (the r187 pinned
> failure mode). Fix is one-line per writer: emit
> `bw.write_bit(false)` for the FIXFIX prefix. The r187 pinned-broken
> test (`acpl1_combined_l_and_ls_pair1_currently_misaligns`) is now
> `acpl1_full_round_trips_with_aligned_pair_lengths` and asserts
> `pair1.num_param_sets = 1`. Total tests 784 (unchanged from r187 —
> r190 fixed the third pin in place rather than adding new ones).
> Round 193 lifts the round-95 5_X ASPX_ACPL_3 encoder's β1 / β2
> parameter sets out of the zero-delta scaffold: a new
> [`encoder_acpl3::extract_beta_q_per_band_carrier_energy`] extracts
> per-parameter-band β_q from a single carrier's MDCT energy
> distribution (β proportional to `√E[x²]` keeps the wet/dry balance
> bounded), and a sibling
> [`encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_beta`]
> drops it into the existing `acpl_data_2ch()` body in place of the
> two zero-delta `acpl_huff_data()` codewords (α1 / α2 / β3 / γ1..γ6
> stay at the round-95 minimum-bit-cost defaults). Caller-facing
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_beta`]
> / `encode_frame_pcm_5_1_acpl3_real_beta` wrap the new builder with
> the same channel-mode forcing and TOC framing as their round-95
> counterparts. With α1 = α2 = 0 and β3 = 0 the §5.7.7.6.2
> Pseudocode 119 `ACplModule2` for the first parameter set reduces to
> `z0 = 0.5·(x0·g1 + x1·g2 + y0·β1)`,
> `z1 = 0.5·(x0·g1 + x1·g2 − y0·β1)` (and analogously `(z2, z3)`
> with β2), so non-zero β1 / β2 injects the decorrelator output that
> gives the Ls / Rs outputs their decorrelated spaciousness. Seven
> integration tests in `tests/round193_5_x_acpl3_real_beta.rs` pin:
> round-trip to 5- / 6-channel `AudioFrame` for 5.0 / 5.1; silent
> input → all-zero β_q indices; tonal carrier + non-zero
> `beta_scale` → at least one non-zero β_q lane; `beta_scale = 0.0`
> is byte-for-byte identical to the round-95 scaffold (strict-superset
> invariant); silent inputs at any `beta_scale` are scaffold-identical;
> non-silent tonal inputs at `beta_scale > 0` diverge from the
> scaffold (different β1 / β2 codeword bit-positions) while keeping
> the padded substream length identical. Total tests 791 (was 784).
> Round 196 layers real per-band α1 / α2 on top of r193: a new
> [`encoder_acpl3::extract_alpha_q_per_band_carrier_correlation`]
> drives α from the per-band L↔R normalised cross-correlation
> `ρ(L, R) = E[L·R] / √(E[L²]·E[R²])` (`α[pb] = α_scale · ρ`),
> clamped to ALPHA_DQ ±2.0. The two ACplModule2 instances in
> ACPL_3 share the (L, R) carrier pair as their (x0, x1) input, so
> without a per-side surround reference α₁ and α₂ both receive the
> same correlation-policy output; the (L, Ls) ↔ (R, Rs) asymmetry
> stays carried by β1 ≠ β2 (from `√E[L²]` vs `√E[R²]`) and the two
> independent decorrelator outputs y0 / y1. A sibling
> [`encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta`]
> drops both the α and β extractors into the `acpl_data_2ch()` body;
> β3 / γ1..γ6 stay zero-delta. Caller-facing
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta`]
> / `encode_frame_pcm_5_1_acpl3_real_alpha_beta` wrap the new builder
> with the same channel-mode forcing and TOC framing. Mono-like
> (highly-correlated) bands push α toward +1, biasing more dry energy
> to the front pair; decorrelated bands stay near α = 0 and split the
> dry mix evenly. Four integration tests in
> `tests/round196_5_x_acpl3_real_alpha_beta.rs` pin: round-trip to a
> 5-channel `AudioFrame`; perfect L = R correlation quantises to
> `α_q = +8` (lane 24 = 1.0) at α_scale = 1; perfect anti-correlation
> L = -R quantises to `α_q = −8`; `α_scale = β_scale = 0.0` is
> byte-for-byte identical to the round-95 scaffold. Total tests 795
> (was 791). Round 202 closes the ACPL_2 half of the deferred 7_X list
> with the **7.0 / 7.1 (3/4/0(.1)) SIMPLE/ASPX_ACPL_2 multichannel
> encoder with real per-parameter-band α + β extraction** per ETSI TS
> 103 190-1 §4.2.6.14 Table 33 row `case ASPX_ACPL_2:` + §5.7.7.5
> Pseudocode 116 + §5.7.7.6.1 Pseudocode 117 — the 7_X (immersive)
> counterpart to the round-144 5.0 ACPL_2 real-α-β path and the
> real-α-β upgrade of the round-107 / 114 zero-delta 7_X ACPL_2
> encoder. New
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_7_0_acpl2_real_alpha_beta`]
> and [`encode_frame_pcm_7_1_acpl2_real_alpha_beta`] (plus the
> `_with_max_sfb` forms) accept a 7- / 8-channel `[L, R, C, Ls, Rs,
> Lb, Rb (, LFE)]` input and produce a 7_X ASPX_ACPL_2 frame whose two
> trailing `acpl_data_1ch()` elements carry the analytic α + β indices
> extracted from the (L, Ls) and (R, Rs) MDCT pairs via the shared
> [`encoder_acpl3::extract_alpha_q_per_band`] /
> [`extract_beta_q_per_band`] primitives. New
> [`encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta`]
> mirrors the round-107 zero-delta `build_7_x_acpl2_body_from_pcm_spectra`
> body schedule (2-bit `7_X_codec_mode = 3`, optional LFE
> `mono_data(b_lfe = 1)`, two `two_channel_data()` pairs, **no**
> joint-MDCT residual layer, trailing centre `mono_data(0)`,
> `aspx_data_2ch + aspx_data_2ch + aspx_data_1ch` envelope trailer)
> with the two trailing `acpl_data_1ch_minimal` writers replaced by
> `write_acpl_data_1ch_real_alpha_beta`. `acpl_config_1ch(FULL)`
> carries no `qmf_band` → `start_band = 0` so every parameter band
> participates in α + β coding (in contrast to the ACPL_1 PARTIAL
> mode whose `acpl_qmf_band` masks the low bands). D0 module models
> (L → Ls); D1 module models (R → Rs); the Ls / Rs spectra feed the
> α + β extractors only and are not emitted on the ACPL_2 wire. The
> back pair Lb / Rb is accepted for layout completeness but not
> carried by the ASPX_ACPL_2 body (the decoder's 7_X ACPL_2 dispatch
> populates slots 0..4 + the LFE slot 7 — slots 5/6 stay silent),
> matching the round-107 documented Table 202 channel mapping plus
> the round-80 LFE PCM render at decode time. Decoder round-trip: 7.0
> ACPL_2 → 7-channel S16 interleaved PCM (1920 samples × 7 ch × 2
> bytes); 7.1 ACPL_2 → 8-channel S16 with LFE IMDCT'd into slot 7.
> Ten integration tests in
> `tests/round202_7_x_acpl2_real_alpha_beta.rs` pin: 7.0 / 7.1
> round-trip to 7- / 8-channel `AudioFrame`; decoder resolves
> `SevenXCodecMode::AspxAcpl2` with both `acpl_data_1ch_pair[0/1]`
> populated; loud-surround vs silence-surround inputs produce
> materially different bytes (the round-107 / 114 zero-delta scaffold
> would emit identical α / β codewords regardless of surround input);
> silence input round-trips with β_q = 0 in every band; encoder is
> bit-deterministic for matched inputs and fresh state; direct
> body-builder probe diverges from the round-107 zero-delta scaffold
> byte stream when the caller's Ls/Rs spectra are non-trivial. Total
> tests 805 (was 795). Round 208 lands the **5_X SIMPLE/ASPX_ACPL_3
> encoder's real per-band γ5 / γ6 extraction** — closing the centre-
> channel reconstruction half of the long-standing "γ stays at the
> zero-delta scaffold" deferral. In §5.7.7.6.2 Pseudocode 118 step 7
> the centre output `z4` is built by the third `ACplModule2`
> invocation with `(a = 1, b = 0, y = 0)`:
> `z4 = 0.5 · (γ5·x0in + γ6·x1in)`. Step 11 scales `z4 *= √2` before
> QMF synthesis; step 1 rescales the carriers
> `x0in = (1 + √2)·L`, `x1in = (1 + √2)·R`. The centre reconstruction
> (β3 = 0, ducker = 1) is therefore `C ≈ K · (γ5·L + γ6·R)` with
> `K = √2·(1+√2)/2 = 1+√(1/2)`. New
> [`encoder_acpl3::extract_gamma_5_6_q_per_band_centre_least_squares`]
> solves the 2×2 normal equations per parameter band
> (`[<L,L>, <L,R>; <L,R>, <R,R>]·[γ5; γ6] = [<L,C>/K; <R,C>/K]`)
> for the (γ5, γ6) pair that minimises the MDCT-bin-wise residual
> `Σ (C/K − γ5·L − γ6·R)²`. Bands with a degenerate Gram matrix (no
> L or R energy, or perfectly collinear L = ±R within numerical
> tolerance) keep γ5 = γ6 = 0. The quantiser uses the Table-208
> linear `gamma_q = round(γ / gamma_delta)` mapping with the
> symmetric `±cb_off` clamp (`cb_off = 20` Fine / `10` Coarse, table
> magnitude bound ±2.0). γ1..γ4 + β3 stay at the round-95 scaffold —
> those parameter sets drive the (L, R, Ls, Rs) sub-pipeline plus
> the ACplModule3 cross-residual, and neither has a per-side
> surround reference at encode time for the 5.0 / 5.1 PCM input
> layouts. New
> [`encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_gamma`]
> is a drop-in replacement for the round-196 real-α-β builder with
> additional `coeffs_c: Option<&[f32]>` + `gamma_scale: f32`
> parameters. New
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma`]
> + `_5_1_` high-level entry points accept `[L, R, C]` (5.0) or
> `[L, R, C, LFE]` (5.1) PCM. Eight integration tests in
> `tests/round208_5_x_acpl3_real_gamma.rs` pin: 5.0 round-trip to a
> 5-channel `AudioFrame`; 5.1 round-trip to a 6-channel
> `AudioFrame`; silent-centre input produces γ5_q = γ6_q = 0 in
> every band; `C = (L + R) / 2` produces non-zero γ_q in ≥1
> tonally-active band (verifies the least-squares extractor selects
> non-trivial γ); loud-centre vs silent-centre inputs produce
> materially different bytes (the round-196 path would emit
> identical γ codewords regardless of centre input);
> `α/β/γ_scale = 0.0` matches the round-95 scaffold byte-for-byte;
> `γ_scale = 0.0` reproduces the round-196 real-α-β bytes exactly;
> encoder is bit-deterministic for matched inputs and fresh state.
> Total tests 813 (was 805). Real γ1..γ4 extraction (the
> (L,R,Ls,Rs) sub-pipeline mix parameters) requires per-side
> surround references which the 5.0 / 5.1 PCM input layout does not
> carry — these stay at the round-95 zero-delta scaffold pending a
> 5.1+Ls+Rs PCM input layout. Real β extraction for the 7_X ACPL_3
> paths, real ASPX envelope coding, real Table-181 SAP-derived
> residual content (for the ACPL_1 paths), and back-pair Lb/Rb
> carriage remain deferred. Round 215 closes the round-208 γ1..γ4
> deferral by adding the **5_X SIMPLE/ASPX_ACPL_3 encoder's real
> per-band γ1 / γ2 / γ3 / γ4 extraction** — the (L, Ls) and (R, Rs)
> output-pair gammas — driven by a 5-channel `[L, R, C, Ls, Rs]`
> (5.0) or 6-channel `[L, R, C, Ls, Rs, LFE]` (5.1) PCM input
> layout. In §5.7.7.6.2 Pseudocode 118 step 5 the (L, Ls) pair is
> built by the first `ACplModule2` invocation with `(a = α₁,
> b = β₁, y = y₀)`: `z0 = 0.5·(1+α₁)·(γ₁·x0in + γ₂·x1in) +
> 0.5·y₀·β₁` and `z1 = 0.5·(1−α₁)·(γ₁·x0in + γ₂·x1in) − 0.5·y₀·β₁`,
> with step 11 scaling `Ls = √2·z1`. Forming `(L + Ls/√2)` cancels
> the `y₀·β₁` decorrelator contribution exactly, leaving
> `L + Ls/√2 = (γ₁·x0in + γ₂·x1in) = (1+√2)·(γ₁·L + γ₂·R)` via the
> step-1 carrier rescaling `x0in / x1in = (1+√2)·L / R` —
> independent of α₁ and β₁. By symmetry with step 6, the same fit
> shape gives `(γ₃, γ₄)` from `(R + Rs/√2)/(1+√2)`. New
> [`encoder_acpl3::extract_gamma_1_2_q_per_band_surround_least_squares`]
> and [`extract_gamma_3_4_q_per_band_surround_least_squares`]
> solve the 2×2 normal equations
> `[<L,L> <L,R>; <L,R> <R,R>]·[γ; γ'] = [<L,T>; <R,T>]` per
> parameter band (the same shape as the round-208 γ5 / γ6 centre
> fit, just with a different per-band target). Bands with a
> degenerate Gram matrix keep `γ = γ' = 0`. The quantiser reuses
> the Table-208 linear `gamma_q = round(γ / gamma_delta)` mapping
> with the symmetric `±cb_off` clamp. New
> [`encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma`]
> is a drop-in replacement for the round-208 real-α-β-γ5-γ6 builder
> with additional `coeffs_ls: Option<&[f32]>` +
> `coeffs_rs: Option<&[f32]>` parameters and a
> `write_acpl_data_2ch_real_alpha_beta_full_gamma` helper that
> emits all six γ entropy layers (γ1..γ6) as REAL codewords (β3
> stays zero-delta). New
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma`]
> and `_5_1_` high-level entry points accept `[L, R, C, Ls, Rs]`
> (5.0) or `[L, R, C, Ls, Rs, LFE]` (5.1) PCM. Nine integration
> tests in `tests/round215_5_x_acpl3_real_full_gamma.rs` pin: 5.0
> round-trip to a 5-channel `AudioFrame`; 5.1 round-trip to a
> 6-channel `AudioFrame`; silent Ls (Ls = 0) → γ2_q = 0 in every
> band when probed directly; silent Rs (Rs = 0) → γ3_q = 0 in every
> band; `α/β/γ_scale = 0.0` matches the round-95 zero-delta
> scaffold byte-for-byte; `γ_scale = 0.0` reproduces the round-196
> real-α-β bytes exactly; loud-surround vs silent-surround inputs
> produce materially different bytes (the round-208 path would emit
> identical γ1..γ4 codewords regardless of surround input);
> deterministic for matched inputs and fresh state. Total tests
> 822 (was 813). β3 extraction (requires modelling the
> unobservable third decorrelator output `y₂`), real ASPX envelope
> coding, real Table-181 SAP-derived residual content (for the
> ACPL_1 paths), and back-pair Lb/Rb carriage remain deferred.
> Round 219 begins closing the "real ASPX envelope coding" deferral
> by landing the encoder's value-emitting ASPX-Huffman primitives —
> the six [`encoder_acpl3::write_aspx_sig_f0_value`] /
> `write_aspx_sig_df_value` / `write_aspx_sig_dt_value` /
> `write_aspx_noise_f0_value` / `write_aspx_noise_df_value` /
> `write_aspx_noise_dt_value` helpers — that the existing scaffold's
> `pick_min_len_cw` / `pick_zero_delta_cw` writers can be swapped
> for in a follow-up round. Each takes an integer index `v` (F0) or
> signed `delta_q` (DF / DT) and writes the matching `(cw, len)`
> from the codebook selected by `(quant_mode, stereo_mode)` for
> SIGNAL paths or `stereo_mode` alone for NOISE paths; values
> outside `[0, codebook_length)` (F0) or `[-cb_off, +cb_off]`
> (DF / DT) clamp to the codebook's extreme entries rather than
> panicking, matching the decoder's parser semantics. The four
> SIGNAL codebooks (`LEVEL_15` / `BALANCE_15` / `LEVEL_30` /
> `BALANCE_30`) and the two NOISE codebooks (`LEVEL` / `BALANCE`)
> are all dispatched via a new `aspx_sig_hcb_arrays()` /
> `aspx_noise_hcb_arrays()` `(LEN, CW, cb_off)` triple lookup,
> mirroring the existing `acpl_hcb_arrays()` shape. Twelve
> integration tests in `tests/round219_aspx_envelope_value_writers.rs`
> pin: SIGNAL F0 / DF / DT round-trip against `parse_aspx_huff_data()`
> for every `(quant_mode, stereo_mode)` × representative-value
> combination; NOISE F0 / DF / DT round-trip across both stereo
> modes; out-of-range F0 and DF values clamp to the codebook edge;
> repeated writes are byte-deterministic; and a full F0 + DF
> envelope round-trips through the higher-level `parse_aspx_ec_data()`
> entry point with `num_sbg = 2` and `freq_res = highres`. Total
> tests 834 (was 822). The existing minimum-bit-cost
> `write_aspx_sig_f0` / `write_aspx_sig_df_zero` /
> `write_aspx_noise_f0` / `write_aspx_noise_df_zero` writers stay
> in place; no `write_aspx_data_*_minimal()` call site is touched.
> A subsequent round will route the new helpers through a
> `write_aspx_data_2ch_real_envelope()` builder that consumes per-
> `(sbg, atsg)` envelope quant indices computed from the input MDCT
> spectra (inverting Pseudocode 82's
> `scf = n_subbands · 2^(qscf/a)` form).
> Round 226 lands that builder pair. Two new public emitters in
> [`encoder_acpl3`] — `write_aspx_data_2ch_real_envelope()` (Table 52)
> and `write_aspx_data_1ch_real_envelope()` (Table 51) — accept a
> per-channel [`encoder_acpl3::AspxRealEnvelopeChannel`] payload
> (`sig: &[i32]` + `noise: &[i32]`) carrying caller-supplied F0 +
> signed DF quant indices and route them through the round-219
> value-emitting helpers `write_aspx_sig_f0_value` /
> `write_aspx_sig_df_value` / `write_aspx_noise_f0_value` /
> `write_aspx_noise_df_value`. The framing skeleton mirrors the
> existing `_minimal` writers — FIXFIX prefix `0`, `tmp_num_env = 0`
> (→ `num_env = 1`), `aspx_balance = 1` for the 2ch variant (shared
> channel-0 framing), SIGNAL + NOISE delta-direction bits = FREQ,
> `aspx_hfgen_iwc_2ch`/`_1ch` trailer all zeros — and the SIGNAL
> band count keys off `cfg.signals_freq_res()` (low-res when the
> in-band `aspx_freq_res = 0` bit is emitted, otherwise the parser's
> high-res fallback). Per-channel stereo-mode follows Table 52: ch0
> = LEVEL, ch1 = BALANCE; the 1ch path uses LEVEL throughout. Caller
> slices shorter than the derived SBG count zero-pad the trailing
> envelope positions; F0 values outside `[0, codebook_length)` clamp
> to the codebook edge; DF values outside `[-cb_off, +cb_off]`
> saturate to the symmetric edge — matching the round-219 helper
> semantics and the decoder's `decode_delta()` clamp surface. Eight
> integration tests in `tests/round226_aspx_real_envelope_writers.rs`
> pin: a 2ch deterministic F0 + DF envelope round-trips through
> `parse_aspx_ec_data` to recover the caller's input per-channel; a
> 1ch envelope round-trips through the same path with LEVEL-only
> stereo_mode; short input slices zero-pad in place; the 2ch and 1ch
> writers are byte-deterministic across repeated invocations; all-
> zero inputs decode to all-zero envelopes; different per-channel
> inputs produce different bytes; out-of-range DF saturates at the
> codebook's `+cb_off` edge (Fine/Level DF cb_off = 70). Total tests
> 842 (was 834). The minimum-bit-cost `write_aspx_data_*_minimal()`
> family stays in place; existing call sites are untouched. A
> follow-up round can chain the new builders with a per-(sbg, env)
> envelope-extractor that quantises the input MDCT spectra into the
> F0 + DF indices the new emitters accept (the inverse of
> Pseudocode 82's `scf = n_subbands · 2^(qscf/a)` reconstruction).
> Round 234 closes that follow-up by adding the **encoder-side ASPX
> envelope extractor** — the inverse of ETSI TS 103 190-1 §5.7.6.3.4
> Pseudocodes 80 / 81 (FREQ-direction DPCM accumulator) plus §5.7.6.3.5
> Pseudocodes 82 (`scf = n_subbands · 2^(qscf/a)`) and 83
> (`scf_noise = 2^(6 − qscf_noise)`). Five new public primitives in
> [`encoder_acpl3`] —
> [`encoder_acpl3::quantize_sig_scf`] (`Fine` ⇒ `a = 2`, `Coarse` ⇒
> `a = 1`, `num_qmf_subbands = 64`),
> [`encoder_acpl3::quantize_noise_scf`],
> [`encoder_acpl3::freq_dpcm_encode_qscf`] (returns `[F0, DF₁, …]`
> with `F0 = qscf[0]` and `DF[sbg ≥ 1] = qscf[sbg] − qscf[sbg − 1]`),
> [`encoder_acpl3::extract_aspx_sig_envelope_indices`] and
> [`encoder_acpl3::extract_aspx_noise_envelope_indices`] — together
> compose the per-channel chain `scf[] → qscf[] → [F0, DF₁, …]` whose
> output is exactly the `Vec<i32>` slice the round-226 builder pair
> accepts on `AspxRealEnvelopeChannel::{sig, noise}`. A new public
> type [`encoder_acpl3::AspxEnvelopeScfChannel`] (`{ sig: &[f32],
> noise: &[f32] }`) plus
> [`encoder_acpl3::build_aspx_real_envelope_channel`] runs both
> extractors and returns owned `(Vec<i32>, Vec<i32>)` for direct
> wiring. Non-positive `scf` clamps on the encoder side so the spec's
> `scf[0] = scf[1]` carry-through path (Pseudocode 82 line) and
> callers passing 0 for silent bands stay well-defined; the round-219
> writers further saturate any quant index outside `[0,
> codebook_length)` (F0) or `[-cb_off, +cb_off]` (DF) at the
> codebook edge. The round-trip property is: feeding caller `scf`
> slices through the extractor, then the round-226 builder, then
> re-parsing the body through `parse_aspx_ec_data` plus the decoder's
> `delta_decode_sig` / `delta_decode_noise` plus `dequantize_sig_scf`
> / `dequantize_noise_scf`, recovers the input `scf` vectors within
> the per-band rounding of `round(a · log2(scf / 64))` /
> `round(6 − log2(scf))`. Fourteen integration tests in
> `tests/round234_aspx_envelope_extractor.rs` pin: forward-inverse
> identity at integer-quant grid points for both Fine and Coarse
> signal step sizes; forward-inverse identity for Pseudocode 83 on
> the noise side; non-positive `scf` clamps to a finite quant index;
> FREQ-DPCM encoder produces `[5, 2, −4, −4, 1]` for `qscf = [5, 7,
> 3, −1, 0]` with the decoder's accumulator recovering the input
> exactly; empty / single-band inputs pass through; end-to-end
> accumulator + Pseudocode-82 / 83 round-trip from caller `scf`
> through extractor through Pseudocode-{82, 83};
> `build_aspx_real_envelope_channel` matches direct calls
> entry-for-entry; full encoder→decoder loop wiring
> `build_aspx_real_envelope_channel` into
> `write_aspx_data_2ch_real_envelope` recovers the per-channel
> SIGNAL / NOISE `scf` vectors through the decoder's full pipeline;
> determinism across repeated invocations; different inputs produce
> materially different DPCM payloads; empty per-channel slices return
> empty vectors. Total tests 856 (was 842). The encoder now has the
> complete `scf[] → on-wire bytes` chain for real ASPX envelope
> coding; remaining envelope-coding work is the energy estimator
> that turns input MDCT spectra into the per-`sbg` `scf` vectors the
> extractor consumes (the inverse of Pseudocodes 90 + 91), plus
> driving the new extractor + builder pair from the existing
> high-level encode entry points. Round 240 closes the first half of
> that follow-up by adding the **encoder-side HF QMF energy
> aggregator** — the dual of ETSI TS 103 190-1 §5.7.6.4.2.1
> Pseudocodes 90 + 91 that the decoder uses on the inverse path. The
> aggregator [`encoder_acpl3::aggregate_qmf_to_sbg_atsg`] takes an HF
> QMF matrix `q_high` shaped `[absolute_sb][ts]`, the SIGNAL or NOISE
> subband-group borders (`sbg_sig` / `sbg_noise` per Pseudocode 91),
> the ATS-envelope borders (`atsg_sig` / `atsg_noise` per Pseudocode
> 90), the `num_ts_in_ats` ATS span and the A-SPX start subband
> `sbx`, and returns the per-`[sbg][atsg]` matrix of average squared
> magnitudes — i.e. the SBG-aggregated counterpart of the decoder's
> per-QMF-subband `est_sig_sb`. Two thin per-side helpers
> [`encoder_acpl3::extract_aspx_sig_envelope_scf_from_qmf`] and
> [`encoder_acpl3::extract_aspx_noise_envelope_scf_from_qmf`] pick the
> leading envelope (`atsg = 0`) column for single-envelope frames,
> producing a per-`sbg` `Vec<f32>` ready to feed the round-234
> [`encoder_acpl3::extract_aspx_sig_envelope_indices`] /
> [`encoder_acpl3::extract_aspx_noise_envelope_indices`] extractors.
> A new public type [`encoder_acpl3::AspxQmfEnvelopeChannel`] (`{
> q_high: &[Vec<(f32, f32)>], sbg_sig_borders: &[u32],
> sbg_noise_borders: &[u32] }`) plus
> [`encoder_acpl3::build_aspx_real_envelope_channel_from_qmf`] runs
> the aggregator + extractor pair end-to-end and returns owned `(sig,
> noise) Vec<i32>` ready to drop into the round-226
> `AspxRealEnvelopeChannel { sig: &[i32], noise: &[i32] }` slot.
> Edge handling tracks the decoder's bounds-checked Pseudocode 90:
> entries past the QMF matrix bounds contribute zero energy, empty /
> single-entry borders return empty vectors, zero-span ATS or band
> groups return `0.0` for the affected cell, and `sbg_borders[i] <
> sbx` clamps upward to `sbx` so callers can pass spec-shaped
> absolute borders verbatim. Fourteen integration tests in
> `tests/round240_aspx_qmf_energy_aggregator.rs` pin: constant-energy
> aggregation matches the per-cell mean; per-ATSG partitioning
> recovers a [1.0, 9.0] split across two envelopes; per-SBG
> partitioning recovers a [1.0, 16.0] split across two bands;
> sub-`sbx` borders clamp upward; empty SBG / ATSG borders return
> empty matrices; zero-span ATSG cells return 0.0; the SIGNAL +
> NOISE per-side helpers emit per-`sbg` vectors mirroring the
> aggregator; the QMF-driven convenience builder matches the manual
> aggregator + extractor + builder chain entry-for-entry; an integer-
> quant-grid input (`scf = 64` and `128` for Fine signal) hits the
> expected `[F0 = 0, DF₁ = 2]` DPCM payload; QMF rows shorter than
> `tsz` contribute partial energy without panicking; the QMF-driven
> builder is deterministic across repeated invocations; different
> QMF inputs produce different DPCM payloads. Total tests 870 (was
> 856). The encoder now has the complete `q_high → scf → qscf →
> DPCM → on-wire bytes` chain for real ASPX envelope coding;
> remaining envelope-coding work is driving the new
> aggregator + extractor + builder chain from the existing
> high-level encode entry points (currently scaffolds emit
> minimum-cost zero-delta envelopes). Round 243 lands the
> **encoder-side `chparam_info()` / `sap_data()` builders** —
> dual of `parse_chparam_info` / `parse_sap_data` per ETSI TS
> 103 190-1 §4.2.10.1 Table 47 + §4.2.10.2 Table 48. Before
> this round the encoder open-coded `bw.write_u32(0, 2)` at
> six sites in `encoder_asf.rs` for identity-SAP (`sap_mode =
> 0`). [`encoder_asf::write_chparam_info`] now covers every
> legal `sap_mode in {0, 1, 2, 3}`: `0` is the existing
> identity emission, `1` walks per-`(group, sfb)` `ms_used[g
> ][sfb]` bits in group-major order, `2` (reserved) emits the
> 2-bit header on its own, mirroring the parser's
> accept-and-skip behaviour, and `3` dispatches into
> [`encoder_asf::write_sap_data`] which emits the
> `sap_coeff_all` bit, the per-pair flag array when
> `sap_coeff_all = 0`, the conditional `delta_code_time` bit
> when `num_window_groups != 1`, and the per-pair
> HCB_SCALEFAC-coded `dpcm_alpha_q` deltas (same `delta + 60
> → HCB_SCALEFAC index` map the round-49 `write_scalefac_data`
> uses, with the same `[0, 120]` clamp policy). Half-built
> `ChparamInfo` inputs (rows shorter than `max_sfb_per_group`)
> zero-fill the missing entries so the writer stays total; a
> `sap_mode = 3` input with `sap_data = None` emits a
> `SapData::default()` body that the parser walks as a
> `sap_coeff_all = 0` all-false row. Thirteen integration
> tests in `tests/round243_chparam_info_writer.rs` pin every
> `sap_mode` code: header-only emissions (`sap_mode in {0,
> 2}` produce exactly 2 bits); single- and multi-group
> `ms_used` payloads recover entry-for-entry; missing
> `ms_used` rows zero-fill on the wire; `sap_coeff_all = 1`
> single-group bodies recover the DPCM deltas at even-sfb
> pair starts; `sap_coeff_all = 0` partial-pair bodies
> recover both the per-pair flag array and the selectively-
> emitted DPCM entries; multi-group bodies with
> `delta_code_time = 1` recover across two groups; the
> `sap_data = None` degenerate input emits a default body
> the parser walks; out-of-range DPCM deltas clamp to ±60;
> a full sweep of every legal delta in `[-60, +60]`
> round-trips exactly; `sap_mode = 0` drops a populated
> payload on emission; and in-memory `sap_mode` values with
> high bits set are masked to the on-wire 2-bit field.
> Total tests 883 (was 870). The encoder now has a single
> reusable chparam-emission helper for all four `sap_mode`
> codes, ready for §4.2.10 SAP-mode decisioning (M/S vs.
> independent vs. joint-MDCT) to feed real per-band
> `ms_used[]` / per-pair DPCM arrays into the existing 5_X
> / 7_X channel-element walkers in place of today's hard-coded
> identity-SAP literals. Round 246 lands the **encoder-side
> Table-181 SAP residual extractor** — a closed-form
> 2x2-per-sfb inverse of the §5.3.4.3.2 / Table 181
> first-stage SAP matrix recovering joint-MDCT preliminary
> spectra `(sSMP_A, sSMP_B, sSMP_3, sSMP_4)` from a target
> preliminary set `(L, R, Ls, Rs)` plus a `chparam_info()`
> pair. The forward path implemented in round 41
> ([`asf::apply_sap_table_181`]) mixes the front-pair
> carriers `(sSMP_A, sSMP_B)` with the joint-MDCT residual
> `(sSMP_3, sSMP_4)` per Table 181's two independent 2x2
> sub-systems; the new inverse
> ([`asf::invert_sap_table_181`]) reverses each sub-system
> using `det = a*d - b*c` and the closed-form
> `[[d, -b], [-c, a]] / det`. The three SAP coefficient
> families produced by [`asf::extract_sap_abcd`] all admit
> non-singular determinants — identity `(1, 0, 0, 1)` gives
> `det = 1`, M/S `(1, 1, 1, -1)` gives `det = -2`, and the
> SAP-coded `(1 + g, 1, 1 - g, -1)` with `g = alpha_q * 0.1`
> also gives `det = -2`. A future spec extension with a
> singular row emits silence for that band instead of
> panicking (matching the forward path's
> graceful-degradation convention). Outside the SAP-coded
> extent the forward pass leaves `(L, R) = (A, B)` and zeros
> the surround pair; the inverse mirrors with
> `A = L, B = R, s3 = s4 = 0` so the round-trip is symmetric
> at the band boundary. Returns `None` when
> `transform_length` lacks an entry in `sfb_offset_48`, same
> failure mode as the forward path. Five new unit tests in
> `src/asf.rs` pin: identity-row inverse recovers `(A = L,
> B = R, s3 = Ls, s4 = Rs)` inside the SAP extent with
> passthrough + zero-surround outside; M/S-row inverse
> recovers the classic sum-and-difference `A = (L + Ls)/2,
> s3 = (L - Ls)/2` over the SAP extent; forward-then-inverse
> round-trip is bit-stable on the identity row and tight to
> `1e-5` on the M/S row at f32; the unsupported-tl path
> returns `None`. Total lib tests 662 (was 657); integration
> tests unchanged (8 round91 7.X tests + 4 round95 5_X ACPL_3
> tests still green). The IMS encoder's residual-layer
> builder ([`encoder_acpl3::write_acpl_1_residual_layer`])
> currently emits raw `sSMP,3 = Ls`, `sSMP,4 = Rs` matching
> the identity `sap_mode = 0` it writes; the new inverse
> opens the door to non-identity SAP modes producing the
> correct residual spectra for real psychoacoustic-driven
> joint-stereo decisions in subsequent rounds. Round 257
> wires that door open: a new SAP-aware residual-layer
> writer ([`encoder_acpl3::write_acpl_1_residual_layer_sap`])
> pairs the round-246 inverse with the round-243
> [`encoder_asf::write_chparam_info`] emitter so the IMS
> encoder's §4.2.6.6 Table-25 `case ASPX_ACPL_1:` residual
> layer can now express any of the three SAP coefficient
> families produced by [`asf::extract_sap_abcd`] — identity,
> M/S and SAP-coded `alpha_q` — driven by a caller-supplied
> `chparam_info()` pair. The new path takes
> `(coeffs_l, coeffs_r, coeffs_ls, coeffs_rs)` preliminary
> spectra and an `Option<&[ChparamInfo; 2]>`: it emits the
> chparam pair via `write_chparam_info` with
> `max_sfb_per_group = [max_sfb_master]`, recovers the
> residual `(sSMP,3, sSMP,4)` via `invert_sap_table_181`, and
> writes the two `sf_data(ASF)` bodies. When
> `chparam_pair = None` (or both rows carry `sap_mode = 0`)
> the body is bit-equivalent to the legacy round-103
> [`encoder_acpl3::write_acpl_1_residual_layer`] — the
> identity-row inverse reduces to `s3 = ls, s4 = rs`. A new
> public body builder
> ([`encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_sap`])
> wraps the SAP-aware path with the same shape as the legacy
> [`encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra`]
> plus the extra `chparam_pair` slot between the surround
> spectra and the ASPX config. Five new tests in
> `src/encoder_acpl3.rs` pin: bit-equivalence of the
> SAP-aware writer with `chparam_pair = None` against the
> legacy emitter; explicit-identity-rows == default `None`;
> M/S-row body round-trips through `parse_chparam_info`
> with the expected per-band `ms_used` recovered;
> body-builder bit-equivalence with `chparam_pair = None`;
> full body fed through `parse_5x_audio_data_outer`
> recovers the chparam pair into
> `tools.acpl_1_residual_chparam[0..1]` with `sap_mode = 1`
> and the original per-band flags on both rows. Total lib
> tests 667 (was 662); integration suites unchanged. The
> decoder's round-30 pipeline already consumes
> `tools.acpl_1_residual_chparam` through
> `apply_sap_table_181` to re-mix the surround spectra
> before IMDCT, so an encoder that drives the SAP-aware
> path now produces a stream that round-trips end-to-end
> through the existing decoder without further changes.
> Round 260 adds the **encoder-side `ChparamInfo` builders**
> ([`asf::build_chparam_info_ms_used`] and
> [`asf::build_chparam_info_sap_data_from_alpha_q`]) — the
> two non-trivial-arm duals of [`asf::extract_sap_abcd`]
> (§5.3.4.3.2 / Pseudocode 59). The MsUsed builder wraps a
> per-(group, sfb) `ms_used` flag matrix into a
> `ChparamInfo` with `sap_mode = 1`; the SapData builder
> takes the desired per-(g, sfb) `alpha_q` indices in
> `[-60, +60]` plus per-pair `sap_coeff_used` flags and
> computes the pair-major DPCM `dpcm_alpha_q[g][sfb]`
> deltas Pseudocode 59 accumulates back into `alpha_q` —
> odd sfbs leave the dpcm slot at zero (decoder inherits
> from the pair-mate); even sfbs compute `cur - prev` with
> the same `code_delta` policy as the decoder
> (`code_delta == 1` requires `g > 0`,
> `max_sfb_per_group[g] == max_sfb_per_group[g-1]` and
> caller-supplied `delta_code_time` set, with reference
> `alpha_q[g-1][sfb]`; otherwise `alpha_q[g][sfb-2]` for
> `sfb > 0` and zero for `sfb == 0`). A fully-uniform "all
> set" matrix raises `sap_coeff_all` so the per-pair flag
> array elides; `delta_code_time` is normalised to `false`
> on single-group payloads (Table 48 doesn't transmit the
> bit there). Five new unit tests in `src/asf.rs` pin:
> `extract_sap_abcd` reproduces the original `alpha_q` row
> on set bands and identity on cleared bands; the
> cross-group `delta_code_time` path delivers the
> expected `dpcm_alpha_q` deltas; the single-group
> `delta_code_time = true` input is dropped to `false` on
> emit; and `write_chparam_info` →
> `parse_chparam_info` recovers the same SAP body which
> extracts to the original `alpha_q`. Total lib tests 674
> (was 667); integration suites unchanged. Slots into the
> round-257 SAP-aware residual-layer writer — an IMS
> encoder that runs a psychoacoustic decision per
> parameter-band (M/S vs alpha-driven SAP joint stereo)
> can now materialise the `ChparamInfo` pair from its
> decision matrix instead of hand-crafting the inner
> `SapData` body. Round 263 completes the
> `build_chparam_info_*` family with the trivial third arm
> ([`asf::build_chparam_info_none`] — header-only
> `SapMode::None`; `extract_sap_abcd` reproduces identity
> per-sfb across any per-group bound) and adds a
> per-(group, sfb) M/S-vs-L/R **decision driver**
> ([`asf::select_ms_used_for_pair`]) that picks
> `ms_used[g][sfb]` per band using the standard
> joint-stereo *concentration* criterion: pick M/S when
> `min(E_M', E_S') < min(E_L, E_R)` over the per-band
> MDCT bins, with `M' = (L + R) / 2, S' = (L - R) / 2`
> (matching the per-sfb `(1, 1, 1, -1)` matrix the decoder's
> `SapMode::MsUsed` arm applies). For a correlated pair
> M' carries the signal and S' vanishes
> (`min_ms = 0 < min_lr`); for an uncorrelated or
> anti-correlated pair both sit near `(E_L + E_R) / 4`.
> Ties (zero-energy bands, no concentration benefit)
> resolve to `false` so the encoder doesn't spend a
> `ms_used` bit when joint coding offers no concentration.
> The returned `Vec<Vec<bool>>` plugs directly into
> `build_chparam_info_ms_used` and the result round-trips
> through `extract_sap_abcd` to the per-sfb `(1, 1, 1, -1)`
> matrix on picked bands and identity on the rest. Five
> new unit tests in `src/asf.rs` cover `SapMode::None`
> builder extract + bit-stream round-trip; per-band
> correlated / anti-correlated / one-sided / zero-energy
> decision discrimination; round-trip through
> `build_chparam_info_ms_used` + `extract_sap_abcd`;
> respect of the per-group `max_sfb` bound; multi-group
> independence. Total lib tests 679 (was 674); integration
> suites unchanged. Together with round 260 this closes the
> encoder path for the `SapMode::None` and `SapMode::MsUsed`
> arms — an IMS encoder can now go directly from per-group
> L/R MDCT spectra to a fully-populated `ChparamInfo`
> without hand-crafting the SAP matrix. Round 271 closes the
> last of the three non-reserved arms with the **SAP-coded
> `alpha_q` decision driver** ([`asf::select_alpha_q_for_pair`])
> — the `SapMode::SapData` analogue of round-263's
> `select_ms_used_for_pair`. Given target stereo MDCT spectra
> `(L, R)` it picks per-(group, sfb) `alpha_q[g][sfb]` indices
> + `sap_coeff_used[g][sfb]` flags per §5.3.2 Pseudocode 59 +
> §5.3.3.2. The decoder reconstructs the output pair from the
> transmitted tracks via the SAP matrix
> `(a, b, c, d) = (1 + g, 1, 1 - g, -1)` with `g = alpha_q · 0.1`;
> inverting (`det = -2`) shows the encoder must transmit
> `I_0 = M = (L + R) / 2` and `I_1 = S − g·M` with
> `S = (L − R) / 2`, so SAP coding is a one-tap prediction of
> the side track from the mid. The `g` minimising the
> transmitted residual energy `Σ (S[k] − g·M[k])²` per
> parameter band is the least-squares projection
> `g* = ⟨S, M⟩ / ⟨M, M⟩`, quantised by `alpha_q = round(10·g*)`
> and clamped to the HCB_SCALEFAC range `[-60, +60]`.
> `sap_coeff_used` is raised only when the quantised index is
> non-zero (pure-mid `⟨S, M⟩ == 0` and zero-mid-energy bands
> clear the flag — no SAP bit where prediction offers nothing).
> The even (pair-leading) sfb drives each `(sfb, sfb+1)` pair
> and the odd partner inherits, matching Pseudocode 59's
> pair-major copy. New public type alias
> [`asf::SapAlphaDecision`] for the `(alpha_q, sap_coeff_used)`
> pair. The returned matrices plug directly into the round-260
> [`asf::build_chparam_info_sap_data_from_alpha_q`] builder,
> closing the encoder path from per-group L/R MDCT spectra to a
> fully-populated `SapMode::SapData` `ChparamInfo`. Five new
> unit tests in `src/asf.rs` pin the least-squares projection
> (`S = M → +10`, `S = -M → -10`, odd-partner inheritance),
> flag-clearing on pure-mid / zero-energy bands, round-trip
> through the SapData builder + `extract_sap_abcd` to the
> `(2, 1, 0, -1)` matrix on picked bands, `alpha_q = 60`
> saturation, and multi-group independence. Total lib tests
> 684 (was 679); integration suites unchanged. Round 279 wires
> the decision drivers into the encoder proper: the
> **decision-driven SAP-coded ASPX_ACPL_1 residual layer**
> (§5.3.4.3.2 Table 181 + §5.3.2 Pseudocode 59). New
> [`encoder_acpl3::select_acpl1_residual_chparam_pair`] runs the
> round-271 `select_alpha_q_for_pair` least-squares decision per
> target `(L, Ls)` / `(R, Rs)` pair over the residual layer's
> single-group `[max_sfb_master]` layout (clamping `alpha_q` to
> ±30 so pair-major DPCM deltas stay HCB_SCALEFAC-codable) and
> materialises the `chparam_info()` rows via the round-260
> `SapData` builder — falling back to the header-only
> `SapMode::None` row when no band benefits. New
> [`encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_sap_auto`]
> + caller-facing [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_sap`]
> additionally fix the round-257 deferred carrier side: the
> `two_channel_data()` payload now carries the Table-181
> matrix-input carriers `(sSMP_A, sSMP_B) = (M, ·)` recovered via
> `invert_sap_table_181` (not the raw L/R preliminaries), so the
> decoder's `apply_sap_table_181` forward mix reproduces the
> requested `(L, R, Ls, Rs)` exactly (up to sf_data quantisation).
> For `Ls = κ·L` the optimal projection `g* = (1−κ)/(1+κ)`
> collapses the transmitted residual `S − g·M` to near-silence —
> measured end-to-end: SAP residual energy < 5 % (unit) / < 10 %
> (full PCM→decoder integration) of the identity path's raw-`Ls`
> residual on correlated tone fixtures, while a no-benefit input
> (`Ls = L` ⇒ `g* = 0`) encodes bit-for-bit identical to the
> round-103 identity path (strict-superset invariant). Five new
> unit tests + 4 integration tests
> (`tests/round279_5_x_acpl1_sap_auto.rs`) pin the selector's
> per-band `(1.7, 1, 0.3, −1)` extraction, the ±30 clamp, the
> identity fallback byte-equality, the bit-stream round trip
> (decoder recovers `sap_mode = 3` rows + forward mix matches all
> four preliminaries within 20 % relative L2), and the residual
> collapse. Total lib tests 689 (was 684); integration suites +4.
> Round 285 closes the round-215 "β₃ stays at the round-95 zero-delta
> scaffold" deferral with **real per-parameter-band β₃ extraction**
> for the 5_X SIMPLE/ASPX_ACPL_3 encoder — the last of the eleven
> `acpl_data_2ch()` parameter layers (α₁ α₂ β₁ β₂ β₃ γ₁..γ₆) to go
> real. Per §5.7.7.6.2 Pseudocode 118, β₃ gains the third decorrelator
> output `y₂` into all three output pairs (steps 8-10); on the centre
> channel step 10 + step 11 give the wet contribution
> `C_wet = −√2·0.5·β₃·y₂` with energy `0.5·β₃²·E[y₂²]`. `y₂` itself is
> decoder-side decorrelator state — unobservable at encode time — but
> its energy is observable: the decorrelator + ducker chain is
> energy-preserving in steady state so `E[y₂²] ≈ E[v₃²]`, and the
> step-2 third-Transform drive
> `v₃ = (γ₁+γ₃+γ₅)·x0in + (γ₂+γ₄+γ₆)·x1in` is fully determined by the
> carrier spectra and the quantised γ matrix the encoder is already
> emitting. New
> [`encoder_acpl3::extract_beta3_q_per_band_centre_residual`]
> energy-matches the wet centre contribution against the per-band
> least-squares remainder of the round-208 dry fit
> `E_res = Σ (C − K·(γ₅·L + γ₆·R))²` (with the *quantised* γ₅ / γ₆ the
> decoder will apply), giving `β₃ = √(2·E_res / E[v₃²])` — quantised
> per Table 207 (`beta3_q = round(β₃ / beta3_delta)`, delta 0.125
> Fine / 0.25 Coarse, symmetric clamp ±8 / ±4 per the staged ETSI
> table file's BETA3 F0 codebooks). New BETA3 value writers
> (`write_acpl_beta3_f0_value` / `_df_value`), a full
> `acpl_data_2ch()` emitter with the β₃ layer live
> (`write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3`), the
> drop-in builder
> [`encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3`]
> (extra `beta3_scale` knob) and caller-facing
> [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3`]
> / `_5_1_` entry points. `beta3_scale = 0.0` reproduces the round-215
> full-γ bytes exactly. Four unit + six integration tests
> (`tests/round285_5_x_acpl3_real_beta3.rs`) pin the Table-207 quant
> grid + clamp, BETA3 F0/DF round-trip through `parse_acpl_huff_data`
> + Pseudocode-121 accumulation, zero-residual ⇒ β₃ = 0 vs
> uncaptured-centre ⇒ β₃ > 0 decisioning, 5.0 → 5-ch and 5.1 → 6-ch
> decoder round-trips, decode-side recovery of the exact per-band
> `beta3_q` row through `parse_5x_audio_data_outer` +
> `differential_decode`, byte-equality at `beta3_scale = 0`,
> wire-liveness, and determinism. Total tests 929 (was 919). Remaining
> ACPL deferral: the 7_X back-pair Lb/Rb — §5.7.7.6.3 Table 202 maps
> the 7_X ASPX_ACPL_{1,2} A-CPL pair onto (Ls → Lb) / (Rs → Rb) with
> L / R / C as passthrough (`z6 = x6`, `z7 = x7`, `z4 = x2` in
> Pseudocode 120), whereas the current decoder render + encoder
> builders reuse the 5_X (L → Ls) / (R → Rs) mapping and leave the
> back-pair slots silent; lifting both sides onto the Table-202
> mapping is the next 7_X round.

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
