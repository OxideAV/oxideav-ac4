# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

- ac4 round 340 (part 3) — the **7_X Table-202 back-pair (Lb/Rb)** channel ASPX envelopes: wire a live pure-ASPX (7_X_codec_mode = ASPX) 7.0 frame path (encode_frame_pcm_7_0_aspx_real_aspx) whose body carries the four ASPX trailers per §4.2.6.14 Table 33 — aspx_data_2ch() (L/R front) + aspx_data_2ch() (Ls/Rs surround) + aspx_data_1ch() (centre) + the extra aspx_data_2ch() for the back pair Lb/Rb (the 7_X_codec_mode == ASPX block). Per Table 202 the 3/4/0.x additional pair is Lb/Rb (A-CPL variables x3/x4); in pure-ASPX mode (no A-CPL coupling) it is carried as an independent two_channel_data() carrier with its own real HF-reconstruction envelope, replacing the SIMPLE-only 7_X body that emitted no ASPX trailers. The encoder QMF-analyses all seven carriers and emits real SIGNAL/NOISE envelopes on each of the four ASPX elements. Adds build_7_0_aspx_asf_body_from_pcm_spectra_real_aspx. Three integration tests (tests/round340_7_x_aspx_back_pair.rs) pin the 7.0 → 7-channel decoder round-trip, back-pair envelope wire liveness (HF-rich vs near-silent back pair → different bytes), and determinism
- ac4 round 340 (part 2) — extend the mono multi-envelope live A-SPX centre path to the **7_X** ASPX_ACPL_2 body (encode_frame_pcm_7_{0,1}_acpl2_real_aspx_centre_multi_env): the 7_X dual of part 1. Same transient-probe → num_env = 2 multi-envelope aspx_data_1ch() centre emission; both carrier pairs (L/R front, Ls/Rs surround) keep their single-envelope aspx_data_2ch(). Per Table 202 (7_X_channel_element A-CPL mapping) the back pair Lb/Rb is reconstructed at decode time from the A-CPL coupling (z1/z3 decorrelator outputs), so it carries no independent carrier under ASPX_ACPL_2. Adds build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env; stationary centre carriers fall back to the round-337 single-envelope 7_X path. Five integration tests (tests/round340_7_x_acpl2_centre_multi_env.rs) pin 7.0→7-channel and 7.1→8-channel decoder round-trips, multi-env-differs-from-scaffold liveness, byte-for-byte stationary fallback, and determinism
- ac4 round 340 — wire the **mono multi-envelope** (num_env > 1) real ASPX payload into the live 5_X ASPX_ACPL_2 centre carrier (encode_frame_pcm_5_0_acpl2_real_aspx_centre_multi_env): the encoder QMF-analyses the **centre** carrier, probes its HF energy for a temporal transient (select_aspx_num_env_from_qmf), and — when present — splits the frame into num_env = 2 uniformly spaced FIXFIX signal envelopes, emitting the round-299 multi-envelope aspx_data_1ch() body (per-envelope FREQ/TIME DPCM) for the centre while the L/R front pair keeps its single-envelope aspx_data_2ch(); the decoder reads aspx_num_env independently per A-SPX element, so the multi-envelope centre coexists with the single-envelope front pair. First live frame path to exercise the round-299 write_aspx_data_1ch_multi_envelope writer (the round-327 5_X ACPL_3 path drives the 2ch writer only). Adds build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx_centre_multi_env + extract_aspx_mono_multi_env; stationary centre carriers fall back byte-for-byte to the round-331 single-envelope path. Five integration tests (tests/round340_5_x_acpl2_centre_multi_env.rs)
- ac4 round 337 — wire the real single- and two-channel ASPX SIGNAL/NOISE envelopes into the live 7_X ASPX_ACPL_2 frame path (encode_frame_pcm_7_{0,1}_acpl2_real_aspx): the 7_X counterpart of round 331. The 7_X ACPL_2 body carries one extra aspx_data_2ch() element (the Ls/Rs surround pair) beyond the 5_X shape, so the encoder QMF-analyses the L/R **and** Ls/Rs **and** centre input PCM and emits real [F0, DF₁, …] envelopes on both carrier-pair aspx_data_2ch() elements (write_aspx_data_2ch_real_envelope) **and** the centre aspx_data_1ch() element (write_aspx_data_1ch_real_envelope), replacing the round-107 minimum-bit-cost scaffolds; real per-band α/β unchanged from round 202. Adds build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx. Five integration tests (tests/round337_7_x_acpl2_real_aspx.rs) pin 7.0→7-channel and 7.1→8-channel decoder round-trips (ASPX_ACPL_2 mode + A-CPL pair resolved), centre 1ch envelope recovery through the round-226 framing skeleton, body-differs-from-scaffold wire liveness on HF input, and bit-determinism
- ac4 round 331 — wire the real single-envelope aspx_data_1ch() into the live 5_X ASPX_ACPL_2 frame path (encode_frame_pcm_5_{0,1}_acpl2_real_aspx): QMF-analyse the L/R **and** centre input PCM and emit real SIGNAL/NOISE envelopes on all three carriers via write_aspx_data_1ch_real_envelope + write_aspx_data_2ch_real_envelope, replacing the round-95 minimum-bit-cost scaffolds; adds build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta_real_aspx. First live frame path to exercise the round-226 aspx_data_1ch() real-envelope writer (ASPX_ACPL_3 has no 1ch element)
- ac4 round 327 — wire the multi-envelope (num_env > 1) real ASPX payload into the live 5_X ASPX_ACPL_3 frame path (encode_frame_pcm_5_{0,1}_acpl3_real_aspx_multi_env): the encoder probes the L/R carrier HF energy for a temporal transient (select_aspx_num_env_from_qmf) and, when present, splits the frame into num_env = 2 uniformly spaced FIXFIX signal envelopes, emitting the round-299 multi-envelope aspx_data_2ch() body (per-envelope FREQ/TIME DPCM) via build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx_multi_env; stationary frames fall back to the round-322 single-envelope path
- ac4 round 322 — wire the real ASPX SIGNAL/NOISE envelope into the live 5_X ASPX_ACPL_3 frame path (encode_frame_pcm_5_{0,1}_acpl3_real_aspx): QMF-analyse the L/R input, aggregate HF energy across the A-SPX subband-group borders, and emit a real-envelope aspx_data_2ch() in place of the minimum-bit-cost scaffold; adds qmf_slots_to_sb_major + build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3_real_aspx
- ac4 round 316 — stereo QMF → multi-envelope ASPX builder (build_aspx_multi_envelope_2ch_from_qmf), the two-channel dual of the round-310 builder feeding the round-299 coupled aspx_data_2ch() writer

## [0.0.7](https://github.com/OxideAV/oxideav-ac4/compare/v0.0.6...v0.0.7) - 2026-06-14

### Other

- ac4 round 306 — encoder-side aspx_hfgen_iwc_1ch/_2ch writers
- ac4 r299: multi-envelope ASPX body writers (num_env > 1) consuming the r292 packer
- ac4 round 292 — encoder-side TIME-direction ASPX envelope DPCM packing
- ac4 round 285 — real per-band β₃ for the 5_X ASPX_ACPL_3 encoder
- ac4 round 279 — decision-driven SAP-coded ASPX_ACPL_1 residual layer
- ac4 round 271 — SAP-coded alpha_q decision driver (select_alpha_q_for_pair)
- ac4 r263: build_chparam_info_none + select_ms_used_for_pair
- ac4 r260: encoder-side ChparamInfo builders — duals of extract_sap_abcd
- ac4 r257: SAP-aware ASPX_ACPL_1 residual-layer writer
- drop release-plz.toml — use release-plz defaults across the workspace
- ac4 r246: encoder-side Table-181 SAP residual extractor
- ac4 r243: encoder-side chparam_info() / sap_data() builders
- ac4 r240: encoder-side HF QMF energy aggregator (dual of Pseudocodes 90 + 91)
- ac4 r234: encoder-side ASPX envelope extractor (inverse of P82/83 + P80/81 DPCM)
- ac4 r226: write_aspx_data_{1,2}ch_real_envelope() builders
- ac4 r219: ASPX envelope value-emitting helpers (sig/noise F0/DF/DT)
- ac4 r215: real per-band γ₁ / γ₂ / γ₃ / γ₄ extraction in 5_X ASPX_ACPL_3 encoder
- ac4 r208: real per-band γ5 / γ6 extraction in 5_X ASPX_ACPL_3 encoder
- ac4 r202: real per-band α + β extraction in 7.0/7.1 ASPX_ACPL_2 encoder
- ac4 r196: real per-band α1/α2 extraction in 5_X ASPX_ACPL_3 encoder

### Added

- **Round 306 — encoder-side `aspx_hfgen_iwc_1ch()` /
  `aspx_hfgen_iwc_2ch()` writers (`crate::encoder_acpl3`).** The exact
  duals of the decoder's `aspx::parse_aspx_hfgen_iwc_1ch` /
  `parse_aspx_hfgen_iwc_2ch` (ETSI TS 103 190-1 §4.2.12.6 / §4.2.12.7,
  Tables 55 / 56). Until now every encoder body writer emitted this
  HF-generation / interleaved-waveform-coding element as the all-zero
  compact form (`aspx_tna_mode[*] = 0`, all three presence bits 0),
  even though the decoder fully parses real inverse-filtering modes,
  additive harmonics (`add_harmonic`), frequency-interleaved coding
  (`fic_used_in_sfb`) and time-interleaved coding (`tic_used_in_slot`).
  New `encoder_acpl3::write_aspx_hfgen_iwc_1ch` /
  `write_aspx_hfgen_iwc_2ch` take real per-SBG `tna_mode` (2 b, masked
  to `0..=3`) plus per-SBG / per-timeslot flag vectors via the public
  `encoder_acpl3::AspxHfgenIwc1ChPayload` /
  `encoder_acpl3::AspxHfgenIwc2ChPayload` payloads, and auto-derive
  every gate from the payload (`*_present` / `*_left` / `*_right` set
  iff the slice has an active flag in range; the 2ch TIC path uses the
  compact `aspx_tic_copy = 1` form when both channels carry the same
  active pattern). Under `aspx_balance = 1` only channel-0 `tna_mode`
  is written (decoder mirrors it); short caller slices zero-pad. The
  existing `write_aspx_data_1ch_minimal` HFGEN block is refactored to
  route through the new 1ch writer with a default payload — output
  stays byte-identical. Eight integration tests in
  `tests/round306_aspx_hfgen_iwc_writers.rs` pin the bit-exact
  round-trip through the decoder parsers (all-zero compact form, real
  flags, padding + masking, balance-mirror, distinct-tna, TIC-copy,
  TIC-right-only, full multi-field stress).
- **Round 292 — encoder-side TIME-direction ASPX envelope DPCM packing
  (`crate::encoder_acpl3`).** The dual of the `direction_time == true`
  branch of the decoder's `aspx::delta_decode_sig` /
  `aspx::delta_decode_noise` (ETSI TS 103 190-1 §5.7.6.3.4 Pseudocode
  80 / 81). The prior round-219/226/234/240 envelope-coding chain only
  emitted the FREQ direction (`freq_dpcm_encode_qscf`); the decoder also
  accepts a per-envelope direction flag and walks a TIME branch
  reconstructing `qscf[sbg][atsg] = prev[sbg] + delta·values[sbg]`
  (with `prev` the previous envelope's row, or `qscf_prev_last` for the
  first envelope). New `encoder_acpl3::time_dpcm_encode_qscf` inverts it
  exactly (`values[sbg] = (qscf[sbg] − prev[sbg]) / delta`), with
  zero-extend-short-`prev` and `±1`-step semantics matching the decoder
  (`delta = 0` treated as `1` for totality). New
  `encoder_acpl3::dpcm_encode_qscf_envelopes` packs a full
  `qscf[sbg][atsg]` matrix into per-envelope
  `encoder_acpl3::AspxEncodedEnvelope { values, direction_time }` rows,
  selecting the cheaper direction per envelope by minimising
  `Σ|values[sbg]|` (FREQ wins ties; `force_freq` reproduces the legacy
  single-direction scaffold). Twelve integration tests
  (`tests/round292_aspx_time_direction_dpcm.rs`) pin the bit-exact
  round-trip through both `delta_decode_sig` and `delta_decode_noise`,
  step/totality edges, short-`prev` zero-extension, the min-L1 policy,
  `force_freq` parity with `freq_dpcm_encode_qscf`, and empty inputs.
  Total tests 941 (was 929).
- **Round 285 — real per-parameter-band β₃ extraction for the 5_X
  SIMPLE/ASPX_ACPL_3 encoder (`crate::encoder_acpl3` +
  `crate::encoder_ims`).** Closes the round-215 "β₃ stays at the
  round-95 zero-delta scaffold" deferral. Per ETSI TS 103 190-1
  §5.7.7.6.2 Pseudocode 118 steps 8-10, β₃ is the gain on the third
  decorrelator output `y₂`; step 10 + step 11 give the centre channel
  a wet contribution `C_wet = −√2 · 0.5 · β₃ · y₂` carrying energy
  `0.5 · β₃² · E[y₂²]`. `y₂` is decoder-side decorrelator state and
  unobservable at encode time, but its energy is not: the
  decorrelator + ducker chain is energy-preserving in steady state,
  so `E[y₂²] ≈ E[v₃²]` with the third-Transform drive
  `v₃ = (γ₁+γ₃+γ₅)·x0in + (γ₂+γ₄+γ₆)·x1in` (Pseudocode 118 step 2)
  fully determined by the carrier spectra and the quantised γ matrix
  the encoder is already emitting. New
  `encoder_acpl3::extract_beta3_q_per_band_centre_residual` energy-
  matches that wet contribution against the per-band least-squares
  remainder of the round-208 centre dry fit
  `E_res = Σ (C − K·(γ₅·L + γ₆·R))²` (`K = 1 + √(1/2)`, using the
  quantised γ₅ / γ₆ the decoder will apply), giving the encoder
  decision `β₃ = √(2 · E_res / E[v₃²])` — a non-negative magnitude,
  quantised per §5.7.7.7 Table 207 (`beta3_q = round(β₃ / beta3_delta)`
  with `beta3_delta = 0.125` Fine / `0.25` Coarse and the symmetric
  `±cb_off` clamp at `±8` / `±4` — half the BETA3 F0 codebook length
  per the staged ETSI table file §A.3 Tables A.46 / A.47). New BETA3
  value writers `write_acpl_beta3_f0_value` / `write_acpl_beta3_df_value`
  mirror the round-208 γ writers (`symbol_index = q + cb_off`
  addressing); a new full `acpl_data_2ch()` emitter
  `write_acpl_data_2ch_real_alpha_beta_full_gamma_beta3` lifts the β₃
  entropy layer from zero-delta scaffold to real FREQ-direction DPCM
  codewords. New public builder
  `encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma_beta3`
  is a drop-in over the round-215 full-γ builder with an extra
  `beta3_scale` decision knob, and new caller-facing entry points
  `encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma_beta3`
  / `_5_1_` accept `[L, R, C, Ls, Rs]` / `[L, R, C, Ls, Rs, LFE]` PCM.
  `beta3_scale = 0.0` reproduces the round-215 byte stream exactly
  (the all-zero β₃ row emits exactly the zero-delta scaffold
  codewords). Four new unit tests pin the Table-207 quant grid +
  clamp, the BETA3 F0/DF writer round-trip through
  `parse_acpl_huff_data` + Pseudocode-121 accumulation, the
  zero-residual ⇒ β₃ = 0 / uncaptured-centre ⇒ β₃ > 0 decision
  split, and builder byte-equality at `beta3_scale = 0`. Six
  integration tests (`tests/round285_5_x_acpl3_real_beta3.rs`) pin
  5.0 → 5-channel and 5.1 → 6-channel decoder round-trips, the
  decode-side recovery of the exact per-band `beta3_q` row through
  `parse_5x_audio_data_outer` + `differential_decode`, IMS
  byte-equality with the round-215 entry at `beta3_scale = 0`,
  wire-liveness of the β₃ layer for an uncaptured centre, and
  bit-determinism. Total tests 919 → 929.

- **Round 279 — decision-driven SAP-coded ASPX_ACPL_1 residual layer
  (`crate::encoder_acpl3` + `crate::encoder_ims`).** Wires the
  round-271 `select_alpha_q_for_pair` decision driver into the encoder
  proper, per ETSI TS 103 190-1 §5.3.4.3.2 / Table 181 + §5.3.2
  Pseudocode 59. New
  `encoder_acpl3::select_acpl1_residual_chparam_pair` runs the
  least-squares `alpha_q` decision per target `(L, Ls)` / `(R, Rs)`
  pair over the residual layer's single-window-group
  `[max_sfb_master]` layout — the residual layer's two
  `chparam_info()` payloads drive two independent 2x2 SAP systems
  mapping the transmitted `(sSMP_A, sSMP_3)` / `(sSMP_B, sSMP_4)`
  tracks to the preliminary front/surround pairs — and materialises
  the rows via the round-260
  `build_chparam_info_sap_data_from_alpha_q` builder, falling back to
  the header-only `SapMode::None` row when no band raises
  `sap_coeff_used`. The picked `alpha_q` is clamped to `[-30, +30]` so
  the pair-major DPCM deltas Pseudocode 59 accumulates stay within the
  HCB_SCALEFAC-codable `[-60, +60]` range on a worst-case sign flip.
  New `encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_sap_auto`
  (+ caller-facing
  `encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_sap` /
  `_with_max_sfb`) additionally closes the round-257 deferred carrier
  side: the `two_channel_data()` payload now carries the Table-181
  **matrix-input** carriers `(sSMP_A, sSMP_B)` recovered through
  `invert_sap_table_181` — on a SAP-coded band the transmitted pair is
  `(M, S − g·M)` (mid + side prediction residual) rather than the raw
  L/R preliminaries the round-257 builder still emitted, so the
  decoder's `apply_sap_table_181` forward mix reproduces the requested
  `(L, R, Ls, Rs)` preliminaries exactly (up to sf_data quantisation).
  Measured: for `Ls = κ·L` correlated surround the optimal projection
  `g* = (1 − κ) / (1 + κ)` collapses the transmitted residual to
  near-silence — SAP residual energy < 5 % (unit, synthetic spectra) /
  < 10 % (full PCM → MDCT → encode → decode integration) of the
  identity path's raw-`Ls` residual — while a no-benefit input
  (`Ls = L` ⇒ zero side energy ⇒ `g* = 0`) encodes **bit-for-bit
  identical** to the round-103 identity path (strict-superset
  invariant). Five new unit tests in `src/encoder_acpl3.rs` pin the
  selector's per-band `(1.7, 1, 0.3, −1)` extraction for `κ = 0.2`,
  the `SapMode::None` fallback on equal pairs, the ±30 clamp on a
  near-anti-correlated pair, the identity byte-equality, and the
  bit-stream round trip (decoder walker recovers `sap_mode = 3` rows;
  forward Table-181 mix matches all four preliminaries within 20 %
  relative L2; residual energy < 5 % of raw surround energy). Four new
  integration tests in `tests/round279_5_x_acpl1_sap_auto.rs` cover
  the 5-channel AudioFrame shape, the recovered SAP rows + residual
  collapse vs the identity encoder on the same tone fixture, the
  no-benefit byte-equality through the full encoder entry point, and
  sequence-counter advancement. Total lib tests 689 (was 684);
  integration suites +4.

- **Round 271 — `alpha_q` decision driver `select_alpha_q_for_pair`
  (`crate::asf`).** The SAP-coded (`SapMode::SapData`) analogue of the
  round-263 `select_ms_used_for_pair` — completes the encoder decision
  surface for the third non-reserved `chparam_info()` arm. Given the
  target stereo MDCT spectra `(L, R)` it picks the per-(group, sfb)
  `alpha_q[g][sfb]` index and the matching `sap_coeff_used[g][sfb]`
  flag per ETSI TS 103 190-1 §5.3.2 Pseudocode 59 + §5.3.3.2. The
  decoder reconstructs the output pair from the transmitted tracks via
  the SAP matrix `(a, b, c, d) = (1 + g, 1, 1 - g, -1)`, `g = alpha_q
  · 0.1`; inverting (`det = -2`) gives the tracks the encoder must
  transmit: `I_0 = M = (L + R) / 2` and `I_1 = S − g·M` with `S = (L −
  R) / 2`. SAP coding is therefore a one-tap prediction of the side
  track from the mid; the `g` that minimises the transmitted residual
  energy `Σ (S[k] − g·M[k])²` per parameter band is the least-squares
  projection `g* = ⟨S, M⟩ / ⟨M, M⟩`, quantised by `alpha_q = round(10
  · g*)` and clamped to the HCB_SCALEFAC-codable range `[-60, +60]`
  (the offset of 60 is applied by `write_sap_data`, not the driver).
  `sap_coeff_used` is raised only when the quantised index is non-zero
  (a pure-mid band, `⟨S, M⟩ == 0`, and a zero-mid-energy band both
  clear the flag so no SAP bit is spent where prediction offers no
  benefit). The decision is taken on the even (pair-leading) sfb of
  each `(sfb, sfb+1)` pair and copied to the odd partner, matching the
  pair-major flag-copy semantics of Pseudocode 59 and
  `build_chparam_info_sap_data_from_alpha_q`. New public type alias
  `crate::asf::SapAlphaDecision` for the `(alpha_q, sap_coeff_used)`
  matrix pair. Five new unit tests in `src/asf.rs` pin: the
  least-squares projection (`S = M → alpha_q = +10`, `S = -M →
  alpha_q = -10`, with odd-partner inheritance); pure-mid and
  zero-energy bands clear the flag; round-trip through
  `build_chparam_info_sap_data_from_alpha_q` + `extract_sap_abcd`
  reproduces the `(2, 1, 0, -1)` matrix on picked bands and identity
  on cleared bands; saturation to `alpha_q = 60` for `g* ≫ 6`;
  multi-group independence. The returned matrices plug directly into
  the round-260 `build_chparam_info_sap_data_from_alpha_q` builder,
  closing the encoder path from per-group L/R MDCT spectra to a
  fully-populated `SapMode::SapData` `ChparamInfo`. Total lib tests
  684 (was 679); integration suites unchanged.

- **Round 263 — `build_chparam_info_none` + `select_ms_used_for_pair`
  encoder helpers (`crate::asf`).** Completes the
  `build_chparam_info_*` family with the trivial third arm
  (`SapMode::None`, header-only emission whose `extract_sap_abcd`
  reproduces identity per-sfb across any per-group bound), plus a
  per-(group, sfb) M/S-vs-L/R decision driver
  (`select_ms_used_for_pair`) that picks `ms_used[g][sfb]` per band
  using the standard joint-stereo *concentration* criterion:
  `min(E_M', E_S') < min(E_L, E_R)` over the per-band MDCT bins
  `[sfb_offset[sfb], sfb_offset[sfb+1])`. For a correlated pair, M'
  carries the signal and S' vanishes (`min_ms = 0`); for an
  uncorrelated or anti-correlated pair, M' and S' both sit near
  `(E_L + E_R) / 4`. Ties (zero-energy bands, no concentration
  benefit) resolve to `false` so the encoder doesn't spend a
  `ms_used` bit when joint coding offers no concentration. The
  returned `Vec<Vec<bool>>` plugs directly into
  `build_chparam_info_ms_used` and the result round-trips through
  `extract_sap_abcd` to the per-sfb `(1, 1, 1, -1)` matrix on picked
  bands and identity on the rest. Five new unit tests in
  `src/asf.rs` cover: `SapMode::None` builder extract + bit-stream
  round-trip; per-band correlated / anti-correlated / one-sided /
  zero-energy decision discrimination; round-trip through
  `build_chparam_info_ms_used` + `extract_sap_abcd`; respect of the
  per-group `max_sfb` bound; multi-group independence. Total lib
  tests 679 (was 674); integration suites unchanged.

- **Round 260 — encoder-side `ChparamInfo` builders
  (`crate::asf::build_chparam_info_ms_used` +
  `crate::asf::build_chparam_info_sap_data_from_alpha_q`).** Encoder-
  side duals of [`crate::asf::extract_sap_abcd`] (§5.3.4.3.2 /
  Pseudocode 59) for the two non-trivial `SapMode` arms.
  - `build_chparam_info_ms_used` wraps a per-(group, sfb) `ms_used`
    flag matrix into a `ChparamInfo` with `sap_mode = 1`; feeding
    the result into `extract_sap_abcd` reproduces the per-sfb
    `(1, 1, 1, -1)` vs identity `(1, 0, 0, 1)` mix the input
    describes, and a `write_chparam_info` → `parse_chparam_info`
    round-trip recovers the same row.
  - `build_chparam_info_sap_data_from_alpha_q` is the real
    workhorse: starting from per-(group, sfb) `alpha_q` indices
    (range `[-60, +60]` — the HCB_SCALEFAC raw-symbol offset of 60
    is applied by the writer, not the builder) plus per-pair
    `sap_coeff_used` flags, it computes the pair-major DPCM
    `dpcm_alpha_q[g][sfb]` deltas Pseudocode 59 accumulates back
    into `alpha_q[g][sfb]`. Odd sfbs leave the dpcm slot at zero
    (decoder inherits from the pair-mate); even sfbs compute
    `cur - prev` with the `code_delta` policy mirrored exactly
    from `extract_sap_abcd` — `code_delta == 1` when `g > 0`,
    `max_sfb_per_group[g] == max_sfb_per_group[g-1]`, and the
    caller-supplied `delta_code_time` is set, with the reference
    being `alpha_q[g-1][sfb]`; otherwise the reference is
    `alpha_q[g][sfb-2]` for `sfb > 0` and zero for `sfb == 0`. The
    fully-uniform "all set" matrix is detected and `sap_coeff_all`
    is raised so the bitstream elides the per-pair flag array.
    `delta_code_time` is normalised to `false` on single-group
    payloads (Table 48 doesn't transmit the bit there).
  - Round-trip guarantees pinned by five new unit tests in
    `src/asf.rs`: `extract_sap_abcd` reproduces the original
    `alpha_q` row on set bands and identity on cleared bands
    (`build_chparam_info_sap_data_pair_major_round_trip` +
    `..._unused_bands_pass_through`); the cross-group
    `delta_code_time` path delivers the expected `dpcm_alpha_q`
    deltas (`..._delta_code_time_cross_group`); single-group
    `delta_code_time = true` input is normalised to `false` on
    emit (`..._single_group_drops_delta_code_time`); and
    `write_chparam_info` → `parse_chparam_info` recovers the same
    SAP body which extracts to the original `alpha_q`
    (`..._round_trips_through_bitstream`).
  - Slots into the round-257 SAP-aware residual-layer writer: an
    IMS encoder that runs a psychoacoustic decision per
    parameter-band (M/S vs. alpha-driven SAP joint stereo) can now
    materialise the `ChparamInfo` pair from its decision matrix
    instead of hand-crafting the inner `SapData` body — the same
    bytes the decoder's `parse_chparam_info` walks back into the
    `apply_sap_table_181` pipeline. Total lib tests 674 (was 667);
    integration suites unchanged.

- **Round 257 — SAP-aware ASPX_ACPL_1 residual-layer writer
  (`write_acpl_1_residual_layer_sap` + body-builder wrapper
  `build_5_x_acpl1_body_from_pcm_spectra_sap`).** Pairs the round-246
  Table-181 inverse with the existing round-243
  [`crate::encoder_asf::write_chparam_info`] emitter so the IMS
  encoder's §4.2.6.6 Table-25 `case ASPX_ACPL_1:` residual layer can
  now express any of the three SAP coefficient families produced by
  [`crate::asf::extract_sap_abcd`] — identity (`sap_mode = 0`), M/S
  (`sap_mode = 1`) and SAP-coded `alpha_q` (`sap_mode = 3`) — rather
  than being hard-pinned to the identity row by the round-103
  [`write_acpl_1_residual_layer`].
  - The new private helper `write_acpl_1_residual_layer_sap` takes
    `(coeffs_l, coeffs_r, coeffs_ls, coeffs_rs)` *preliminary*
    spectra plus an `Option<&[ChparamInfo; 2]>` and (1) emits the
    `chparam_info()` pair via `write_chparam_info` with
    `max_sfb_per_group = [max_sfb_master]`, (2) recovers the
    joint-MDCT residual `(sSMP,3, sSMP,4)` via
    [`crate::asf::invert_sap_table_181`] driven by the same chparam
    pair, and (3) writes the two `sf_data(ASF)` bodies for the
    recovered residual spectra bounded by `max_sfb_master`. When
    `chparam_pair = None` (or both rows carry `sap_mode = 0`) the
    body is bit-for-bit equivalent to
    `write_acpl_1_residual_layer(... coeffs_ls, coeffs_rs)` — the
    identity-row inverse reduces to `s3 = ls, s4 = rs`. The inverse's
    surround-silent convention past `max_sfb_master` is preserved.
  - The new public body builder
    `build_5_x_acpl1_body_from_pcm_spectra_sap` mirrors the round-103
    `build_5_x_acpl1_body_from_pcm_spectra` API with the extra
    `chparam_pair: Option<&[ChparamInfo; 2]>` slot wedged in between
    the surround spectra and the ASPX config. The legacy
    identity-only builder is unchanged.
  - Five new tests in `encoder_acpl3::tests`:
    `write_acpl_1_residual_layer_sap_none_matches_legacy` pins the
    bit-equivalence of the SAP-aware path with `chparam_pair = None`
    against the legacy emitter on identical Ls/Rs preliminaries;
    `write_acpl_1_residual_layer_sap_identity_explicit_matches_default`
    pins explicit identity rows == default `None`;
    `write_acpl_1_residual_layer_sap_ms_row_roundtrips_through_decoder`
    feeds the body through `parse_chparam_info` and asserts the
    decoder recovers `sap_mode = 1` with the right `ms_used` rows
    on both chparam slots; `build_5_x_acpl1_body_sap_none_matches_legacy`
    is the body-builder analogue of the bit-equivalence test;
    `build_5_x_acpl1_body_sap_ms_decoder_recovers_chparam` feeds the
    full body through `parse_5x_audio_data_outer` and asserts
    `tools.acpl_1_residual_chparam[0..1]` recover the original
    chparam pair with all `max_sfb_master` ms_used bands present.
    Total lib tests 667 (was 662); existing integration suites
    remain green.
  - The downstream decoder pipeline that consumes this is already
    wired up: round-30 `decoder.rs` (lines 2661-2705) reads the
    persisted `tools.acpl_1_residual_chparam` and feeds it through
    `apply_sap_table_181` to re-mix the L/R/Ls/Rs preliminary
    spectra before IMDCT, so an encoder building a body via the
    new SAP-aware path produces a stream that round-trips through
    the existing decoder without further changes.

- **Round 246 — encoder-side Table-181 SAP residual extractor
  (`invert_sap_table_181`, dual of `apply_sap_table_181`).** An IMS
  encoder that wants to populate the §4.2.6.6 ASPX_ACPL_1 residual
  layer (Table 25 row `case ASPX_ACPL_1:`, two trailing
  `sf_data(ASF)` bodies carrying `sSMP,3` / `sSMP,4`) now has a
  closed-form 2x2-per-sfb inverse of the §5.3.4.3.2 / Table 181
  first-stage SAP matrix that recovers the joint-MDCT preliminary
  spectra `(sSMP_A, sSMP_B, sSMP_3, sSMP_4)` from a target
  `(L, R, Ls, Rs)` preliminary set and a `chparam_info()` pair.
  - [`crate::asf::invert_sap_table_181`] / new public type alias
    [`crate::asf::SapTable181EncodeOutput`]. Inversion splits the
    Table-181 5x5 matrix into the two independent 2x2 sub-systems
    `(L, Ls)` ↔ `(A, s3)` driven by `chparam_pair[0]` and
    `(R, Rs)` ↔ `(B, s4)` driven by `chparam_pair[1]`. Per sfb the
    inverse uses `det = a*d - b*c` and the closed-form
    `[[d, -b], [-c, a]] / det`.
  - For the three SAP coefficient families produced by
    [`crate::asf::extract_sap_abcd`] the determinant is always
    non-singular: identity row gives `det = 1`, M/S row
    `(1, 1, 1, -1)` gives `det = -2`, and the SAP-coded row
    `(1 + g, 1, 1 - g, -1)` with `g = alpha_q * 0.1` also gives
    `det = -2`. The implementation tolerates a hypothetical
    `det == 0` band (e.g. a future spec extension) by emitting
    silence for that band instead of panicking, mirroring the
    forward path's graceful-degradation convention.
  - Outside the SAP-coded extent (bins past
    `sfb_offset[max_sfb_master]`) the forward pass leaves the front
    pair at `(L, R) = (A, B)` and zeros the surround pair; the
    inverse mirrors this — `A = L`, `B = R`, `s3 = s4 = 0` — so
    the round-trip is symmetric at the band boundary. Returns
    `None` when the transform_length has no entry in
    `sfb_offset_48`, matching the forward path's failure mode.
  - Five new unit tests in `src/asf.rs` cover identity-row
    inverse, M/S-row inverse, forward-then-inverse round-trip on
    both the identity and M/S rows, and the unsupported-tl `None`
    return. All 5 pass; the existing crate test suite remains
    green (662 lib tests pass at this commit).

- **Round 243 — encoder-side `chparam_info()` / `sap_data()` builders
  (dual of `parse_chparam_info` / `parse_sap_data`, Table 47 / 48).**
  Adds a reusable encoder helper covering all four `sap_mode` codes —
  the parser's complement for §4.2.10.1 Table 47 (`chparam_info()`)
  and §4.2.10.2 Table 48 (`sap_data()`). Until this round the
  encoder's six chparam-emission sites in `encoder_asf.rs` open-coded
  `bw.write_u32(0, 2)` for `sap_mode = 0` (identity SAP); now there
  is a single builder that handles `sap_mode = 0` (header-only),
  `sap_mode = 1` (header + per-`(group, sfb)` `ms_used[g][sfb]` bit
  array), `sap_mode = 2` (reserved; header-only, mirroring the
  parser's accept-and-skip behaviour) and `sap_mode = 3` (full
  `sap_data()` body — `sap_coeff_all` bit, per-pair flag array when
  `sap_coeff_all = 0`, `delta_code_time` when `num_window_groups != 1`,
  per-pair HCB_SCALEFAC-coded `dpcm_alpha_q` deltas).
  - [`crate::encoder_asf::write_chparam_info`] — emits the 2-bit
    `sap_mode` selector and dispatches to the matching payload
    branch. Half-built `ChparamInfo` inputs (rows shorter than
    `max_sfb_per_group`) zero-fill the missing entries so the writer
    stays total. A `sap_mode = 3` input with `sap_data = None`
    emits a `SapData::default()` body that the parser walks
    successfully.
  - [`crate::encoder_asf::write_sap_data`] — emits the `sap_coeff_all`
    bit, the per-pair flag array (skipped when `sap_coeff_all = 1`),
    the conditional `delta_code_time` bit and the per-pair DPCM
    deltas. The DPCM map is the same `delta + 60 → HCB_SCALEFAC
    index` the round-49 [`crate::encoder_asf::write_scalefac_data`]
    uses, with the same `[0, 120]` clamp policy.
  - Round-trip is bit-exact with [`crate::asf::parse_chparam_info`]
    and [`crate::asf::parse_sap_data`] across `sap_mode in {0, 1, 2,
    3}`, including: single- and multi-group `ms_used` payloads;
    `sap_coeff_all = 1` single-group and `sap_coeff_all = 0`
    partial-pair multi-group bodies with `delta_code_time = 1`;
    the parser's pair-flag copy semantic (one bit drives both halves
    of `(sfb, sfb+1)`); asymmetric pair-flag input rows; and the
    full `[-60, +60]` DPCM delta range. Out-of-range deltas clamp at
    the codebook boundary (±60), matching the existing scale-factor
    writer's policy.
  - Thirteen integration tests in
    `tests/round243_chparam_info_writer.rs` pin: `sap_mode = 0` emits
    exactly 2 bits as a header-only element; `sap_mode = 2` (reserved)
    is round-trip stable as a header-only emission; `sap_mode = 1`
    single-group `ms_used` recovers entry-for-entry; `sap_mode = 1`
    multi-group with 3 groups of (3, 4, 1) bands recovers the full
    matrix; missing `ms_used` rows zero-fill on the wire; `sap_mode
    = 3` `sap_coeff_all` body recovers the DPCM deltas at even-sfb
    pair starts; `sap_mode = 3` partial-pair body with
    `sap_coeff_all = 0` recovers both the flag array and the
    selectively-emitted DPCM entries; `sap_mode = 3` multi-group
    body with `delta_code_time = 1` recovers across two groups;
    `sap_mode = 3` with `sap_data = None` emits a default body that
    parses as a `sap_coeff_all = 0` all-false row; out-of-range DPCM
    deltas clamp to ±60; a full sweep of every legal delta in `[-60,
    +60]` round-trips exactly; `sap_mode = 0` drops a populated
    `ms_used` / `sap_data` payload on emission; in-memory `sap_mode`
    values with high bits set are masked to the on-wire 2-bit field.
  - Total tests 883 (was 870). The encoder now has a single reusable
    chparam-emission helper covering every legal `sap_mode`, ready
    for the §4.2.10 SAP-mode decisioning work (M/S vs. independent
    vs. joint-MDCT) to feed real per-band `ms_used[]` / per-pair
    DPCM arrays into the existing 5_X / 7_X channel-element walkers
    in place of today's hard-coded identity-SAP literals.

- **Round 240 — encoder-side HF QMF energy aggregator (dual of
  Pseudocodes 90 + 91).** Closes the first half of the round-234
  remaining-work note by landing the per-`(sbg, atsg)` energy
  aggregator that converts an HF QMF matrix into the per-`sbg`
  `scf` vector the round-234 envelope-index extractor consumes —
  completing the encoder's `q_high → scf → qscf → DPCM →
  on-wire bytes` chain for real ASPX envelope coding.
  - [`crate::encoder_acpl3::aggregate_qmf_to_sbg_atsg`] — aggregate
    an HF QMF matrix `q_high` (shape `[absolute_sb][ts]`) into a
    `[sbg][atsg]` matrix of average squared magnitudes per
    Pseudocode 90's per-subband energy reduction grouped by
    Pseudocode 91's SBG borders. Tolerates QMF rows shorter than
    `tsz` (entries past the bounds contribute zero), zero-span ATS
    intervals and zero-span band groups (return `0.0`), and
    `sbg_borders[i] < sbx` (clamps upward to `sbx` so callers can
    pass spec-shaped absolute borders verbatim).
  - [`crate::encoder_acpl3::extract_aspx_sig_envelope_scf_from_qmf`]
    / [`crate::encoder_acpl3::extract_aspx_noise_envelope_scf_from_qmf`]
    — per-side helpers that pick the leading envelope (`atsg = 0`)
    column of the aggregator output, producing a per-`sbg` `Vec<f32>`
    ready to feed the round-234 envelope-index extractor.
  - New public type
    [`crate::encoder_acpl3::AspxQmfEnvelopeChannel`] — `{ q_high:
    &[Vec<(f32, f32)>], sbg_sig_borders: &[u32],
    sbg_noise_borders: &[u32] }` per-channel bundle consumed by the
    QMF-driven envelope builder.
  - [`crate::encoder_acpl3::build_aspx_real_envelope_channel_from_qmf`]
    — convenience builder that runs the QMF aggregator + the round-234
    `extract_aspx_*_envelope_indices` extractors + the round-234
    `build_aspx_real_envelope_channel` builder end-to-end and returns
    owned `(sig, noise) Vec<i32>` ready to drop into the round-226
    `AspxRealEnvelopeChannel { sig: &[i32], noise: &[i32] }` slot.
  - Fourteen integration tests in
    `tests/round240_aspx_qmf_energy_aggregator.rs` pin: constant-
    energy aggregation matches the per-cell mean; per-ATSG
    partitioning recovers a [1.0, 9.0] split; per-SBG partitioning
    recovers a [1.0, 16.0] split; sub-`sbx` borders clamp upward;
    empty SBG / ATSG borders return empty matrices; zero-span ATSG
    cells return 0.0; the per-side helpers emit per-`sbg` vectors
    mirroring the aggregator; the QMF-driven convenience builder
    matches the manual aggregator + extractor + builder chain
    entry-for-entry; an integer-quant-grid input (`scf = 64` and
    `128` for Fine signal) hits the expected `[F0 = 0, DF₁ = 2]`
    DPCM payload; short QMF rows contribute partial energy without
    panicking; the QMF-driven builder is deterministic across
    repeated invocations; different QMF inputs produce different
    DPCM payloads. Total tests 870 (was 856).
  - Refs ETSI TS 103 190-1 §5.7.6.4.2.1 Pseudocodes 90 + 91.

- **Round 234 — encoder-side ASPX envelope extractor (inverse of
  Pseudocodes 80, 81, 82, 83).** Closes the round-226 deferral by
  landing the per-`(sbg, env)` envelope-index extractor that inverts
  Pseudocode 82's `scf = n_subbands · 2^(qscf/a)` reconstruction and
  Pseudocode 83's `scf_noise = 2^(6 − qscf_noise)` reconstruction so
  the round-226 `write_aspx_data_{1,2}ch_real_envelope` builders can
  be chained with caller-supplied envelope-energy scale factors.
  - [`crate::encoder_acpl3::quantize_sig_scf`] — `scf → qscf` for one
    signal-envelope band per Pseudocode 82. `qmode_env = Fine` ⇒
    `a = 2` (1.5 dB step), `Coarse` ⇒ `a = 1` (3 dB step);
    `num_qmf_subbands` mirrors the dequantizer's `64`. Non-positive
    `scf` clamps to a finite quant index instead of producing
    `-inf` so the spec's `scf[0] = scf[1]` carry-through path and
    callers passing 0 for silent bands stay well-defined.
  - [`crate::encoder_acpl3::quantize_noise_scf`] — `scf → qscf` for
    one noise-envelope band per Pseudocode 83 (`qscf = round(6 −
    log2(scf))`).
  - [`crate::encoder_acpl3::freq_dpcm_encode_qscf`] — invert the
    FREQ-direction DPCM accumulator `qscf[sbg] = sum(values[0..=sbg])`
    of Pseudocode 80 / 81. Returns `[F0, DF₁, DF₂, …]` where
    `F0 = qscf[0]`, `DF[sbg ≥ 1] = qscf[sbg] − qscf[sbg − 1]`. Empty
    input returns an empty vector.
  - [`crate::encoder_acpl3::extract_aspx_sig_envelope_indices`] /
    [`crate::encoder_acpl3::extract_aspx_noise_envelope_indices`] —
    per-channel compositions `scf[] → qscf[] → [F0, DF₁, …]` ready
    for the round-219 value-emitting helpers + the round-226 builder
    pair.
  - New public type
    [`crate::encoder_acpl3::AspxEnvelopeScfChannel`] — `{ sig:
    &[f32], noise: &[f32] }` per-channel envelope-energy payload.
  - [`crate::encoder_acpl3::build_aspx_real_envelope_channel`] —
    convenience wrapper that runs both extractors and returns owned
    `(sig, noise)` `Vec<i32>` pairs callers wire into
    `AspxRealEnvelopeChannel` by slice reference.
  - Round-trip property: feeding caller `scf` slices through the
    extractor, then the round-226 builder, then re-parsing the body
    through `parse_aspx_ec_data` + the decoder's `delta_decode_sig` /
    `delta_decode_noise` + `dequantize_sig_scf` /
    `dequantize_noise_scf`, recovers the input `scf` vector within
    the per-band rounding of `round(a · log2(scf / 64))` /
    `round(6 − log2(scf))`.
  - Fourteen integration tests in
    `tests/round234_aspx_envelope_extractor.rs` cover: forward-inverse
    identity at integer-quant grid points for both Fine and Coarse
    signal step sizes; forward-inverse identity for Pseudocode 83
    on the noise side; non-positive `scf` clamps to a finite quant
    index; FREQ-DPCM encoder produces `[5, 2, −4, −4, 1]` for
    `qscf = [5, 7, 3, −1, 0]` with the decoder's accumulator
    recovering the input; empty / single-band inputs pass through;
    end-to-end accumulator + Pseudocode-82 / 83 round-trip from
    caller `scf` through extractor through Pseudocode-{82, 83};
    `build_aspx_real_envelope_channel` matches direct calls; full
    encoder→decoder loop wiring `build_aspx_real_envelope_channel`
    into `write_aspx_data_2ch_real_envelope` recovers the input
    `scf` vectors through the decoder's full pipeline; determinism
    across repeated invocations; different inputs produce materially
    different DPCM payloads; empty per-channel slices return empty
    vectors.
  - Total tests 856 (was 842). With this round the encoder now has
    the complete `scf[] → on-wire bytes` chain for real ASPX
    envelope coding; remaining envelope-coding work is the energy
    estimator that turns input MDCT spectra into the per-`sbg`
    `scf` vectors the extractor consumes (the inverse of Pseudocodes
    90 + 91), plus driving the new extractor + builder pair from
    the existing high-level encode entry points. β₃ extraction in
    the 5_X ACPL_3 path and real Table-181 SAP-derived residual
    content for the ACPL_1 paths remain deferred.

- **Round 226 — `write_aspx_data_2ch_real_envelope()` and
  `write_aspx_data_1ch_real_envelope()` builders.** Closes the second
  step of the README's "real ASPX envelope coding" deferral. The
  round-219 value-emitting ASPX-Huffman primitives
  (`write_aspx_sig_f0_value` / `write_aspx_sig_df_value` /
  `write_aspx_noise_f0_value` / `write_aspx_noise_df_value`) are now
  driven by per-channel envelope builders that emit a full
  ETSI TS 103 190-1 §4.2.12.4 Table 52 (`aspx_data_2ch()`) or
  §4.2.12.3 Table 51 (`aspx_data_1ch()`) body with caller-supplied
  F0 + signed DF quant indices.
  - New public type
    [`crate::encoder_acpl3::AspxRealEnvelopeChannel`] — `{ sig:
    &[i32], noise: &[i32] }` per-channel envelope payload.
  - [`crate::encoder_acpl3::write_aspx_data_2ch_real_envelope`] —
    accepts `(cfg, ch0, ch1)` and writes the Table-52 body with
    `aspx_xover_subband_offset = 0`, FIXFIX framing (`num_env = 1`,
    optional `aspx_freq_res = 0`), `aspx_balance = 1` (shared
    channel-0 framing), SIGNAL + NOISE delta-direction bits = FREQ,
    `aspx_hfgen_iwc_2ch` all-zero trailer, then four `aspx_ec_data`
    calls (ch0 SIGNAL LEVEL, ch1 SIGNAL BALANCE, ch0 NOISE LEVEL,
    ch1 NOISE BALANCE). qmode is forced Fine on FIXFIX + `num_env
    == 1` per Table 52.
  - [`crate::encoder_acpl3::write_aspx_data_1ch_real_envelope`] —
    accepts `(cfg, ch)` and writes the Table-51 body with two
    `aspx_ec_data` calls (SIGNAL + NOISE, both LEVEL).
  - The SIGNAL band count keys off `cfg.signals_freq_res()`: low-res
    when the in-band `aspx_freq_res = 0` bit is emitted (Signalled
    mode), otherwise the parser's `freq_res.get(env)
    .copied().unwrap_or(true)` fallback selects high-res (matching
    the r181 fix in `write_aspx_data_2ch_minimal`).
  - Caller slices shorter than the derived SBG count zero-pad the
    trailing envelope positions; F0 values outside `[0,
    codebook_length)` clamp to the codebook edge; DF values outside
    `[-cb_off, +cb_off]` saturate to the symmetric edge — matching
    the round-219 helper semantics.
  - Eight integration tests in
    `tests/round226_aspx_real_envelope_writers.rs` cover:
    deterministic 2ch envelope round-trips through
    `parse_aspx_ec_data` recovering caller inputs per channel;
    1ch envelope round-trip with LEVEL-only stereo_mode; short
    input slices zero-pad in place; 2ch / 1ch byte determinism;
    all-zero inputs decode to all-zero envelopes; different
    per-channel inputs produce different bytes; out-of-range DF
    saturates at the codebook's `+cb_off` edge (Fine/Level DF
    cb_off = 70).
  - The minimum-bit-cost `write_aspx_data_2ch_minimal` /
    `write_aspx_data_1ch_minimal` writers stay in place; no
    existing call site is touched, so every previous round's
    byte-stream expectations remain valid.
  - Total tests 842 (was 834). The remaining ASPX envelope-coding
    work is the per-(sbg, env) envelope-index extractor that
    inverts Pseudocode 82's `scf = n_subbands · 2^(qscf/a)`
    reconstruction so the new builders can be chained with input
    MDCT spectra. β3 extraction in the 5_X ACPL_3 path and real
    Table-181 SAP-derived residual content for the ACPL_1 paths
    remain deferred.

- **Round 215 — real per-parameter-band γ₁ / γ₂ / γ₃ / γ₄ extraction
  in the 5_X SIMPLE/ASPX_ACPL_3 encoder.** Layered on top of the
  round-208 real γ₅ / γ₆ (centre) + the round-196 real α₁ / α₂ +
  real β₁ / β₂ path: the γ₁..γ₄ entropy layers — previously emitted
  as the round-95 zero-delta scaffold codewords — now carry per-band
  magnitudes derived from per-band 2×2 least-squares fits of the
  (L, Ls) and (R, Rs) output channel pairs onto the (L, R) carrier
  pair. In §5.7.7.6.2 Pseudocode 118 step 5 the (L, Ls) pair is
  built by the first `ACplModule2` invocation with `(a = α₁,
  b = β₁, y = y₀)`, and step 11 scales `Ls = √2·z1`. Forming
  `(L + Ls/√2)` cancels the `y₀·β₁` decorrelator contribution
  exactly, leaving `L + Ls/√2 = (γ₁·x0in + γ₂·x1in)` which expands
  to `(1 + √2)·(γ₁·L + γ₂·R)` via the step-1 carrier rescaling. By
  symmetry with step 6 the same fit shape gives `(γ₃, γ₄)` from
  `(R + Rs/√2)/(1 + √2)`. New
  [`encoder_acpl3::extract_gamma_1_2_q_per_band_surround_least_squares`]
  and [`extract_gamma_3_4_q_per_band_surround_least_squares`]
  solve the 2×2 normal equations
  `[<L,L> <L,R>; <L,R> <R,R>]·[γ; γ'] = [<L,T>; <R,T>]` per
  parameter band. Bands with a degenerate Gram matrix (no L or R
  energy, or perfectly collinear L = ±R within numerical tolerance)
  keep γ = γ' = 0. New
  [`encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_full_gamma`]
  drops all six γ extractors into the `acpl_data_2ch()` body
  alongside the round-208 γ₅ / γ₆ extractor; β₃ stays zero-delta.
  Caller-facing
  [`encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta_full_gamma`]
  / `encode_frame_pcm_5_1_acpl3_real_alpha_beta_full_gamma` wrap
  the new builder, accepting a 5- / 6-channel
  `[L, R, C, Ls, Rs (, LFE)]` input (vs the round-208 3- / 4-channel
  `[L, R, C (, LFE)]` input that could only drive the centre γ
  layer). Nine integration tests in
  `tests/round215_5_x_acpl3_real_full_gamma.rs` pin: 5.0 round-trip
  to a 5-channel `AudioFrame`; 5.1 round-trip to a 6-channel
  `AudioFrame`; silent surround (Ls = 0) yields γ₂_q = 0 in every
  band when probed directly; silent surround (Rs = 0) yields γ₃_q
  = 0 in every band; `α/β/γ_scale = 0.0` reproduces the round-95
  zero-delta scaffold byte-for-byte; `γ_scale = 0.0` reproduces
  the round-196 real-α-β bytes byte-for-byte; loud-surround vs
  silent-surround inputs produce materially different bytes (the
  round-208 path would emit identical γ₁..γ₄ codewords regardless
  of surround input); the encoder is bit-deterministic for matched
  inputs and fresh state. Total tests 822 (was 813). β₃ extraction
  (requires modelling the unobservable decorrelator output `y₂`)
  and real ASPX envelope coding remain deferred.

- **Round 208 — real per-parameter-band γ5 / γ6 extraction in the
  5_X SIMPLE/ASPX_ACPL_3 encoder.** Layered on top of the round-196
  real α₁ / α₂ + real β₁ / β₂ path: the γ5 / γ6 entropy layers now
  carry per-band magnitudes derived from a 2×2 per-band
  least-squares fit of the centre channel. In §5.7.7.6.2
  Pseudocode 118 step 7 the centre output `z4` is built by the
  third `ACplModule2` invocation with `(a = 1, b = 0, y = 0)`:
  `z4 = 0.5 · (γ5·x0in + γ6·x1in)`. Step 11 scales `z4 *= √2`
  before QMF synthesis; step 1 rescales the carriers
  `x0in = (1 + √2)·L`, `x1in = (1 + √2)·R`. The centre
  reconstruction (β3 = 0, ducker = 1) is therefore
  `C ≈ K · (γ5·L + γ6·R)` with `K = √2·(1+√2)/2 = 1+√(1/2)`. The
  round-208 extractor solves the 2×2 normal equations per
  parameter band:
  ```text
    [ <L,L>  <L,R> ] [γ5]   [ <L,C>/K ]
    [ <L,R>  <R,R> ] [γ6] = [ <R,C>/K ]
  ```
  for `(γ5, γ6)` that minimise the MDCT-bin-wise residual
  `Σ (C/K − γ5·L − γ6·R)²`. Bands with a degenerate Gram matrix
  (no L or R energy, or perfectly collinear L = ±R within
  numerical tolerance) keep γ5 = γ6 = 0. The quantiser uses the
  Table-208 linear `gamma_q = round(γ / gamma_delta)` mapping with
  the symmetric `±cb_off` clamp (`cb_off = 20` Fine / `10` Coarse,
  table magnitude bound ±2.0). γ1..γ4 + β3 stay at the round-95
  scaffold (those parameter sets drive the (L, R, Ls, Rs)
  sub-pipeline plus the ACplModule3 cross-residual — neither of
  which has a per-side surround reference at encode time for the
  5.0 / 5.1 PCM input layouts the real-γ entry point targets).
  - [`crate::encoder_acpl3::extract_gamma_5_6_q_per_band_centre_least_squares`]
    — per-parameter-band 2×2 least-squares γ5 / γ6 extractor.
  - [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta_gamma`]
    — drop-in replacement for the round-196
    `build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta` with
    additional `coeffs_c: Option<&[f32]>` + `gamma_scale: f32`
    parameters. `gamma_scale = 0.0` reproduces the round-196 byte
    stream exactly; `alpha_scale = beta_scale = gamma_scale = 0.0`
    reproduces the round-95 zero-delta scaffold.
  - [`crate::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta_gamma`]
    + `..._5_1_...` — high-level entry points accepting `[L, R, C]`
    (5.0) or `[L, R, C, LFE]` (5.1) PCM.
  - `tests/round208_5_x_acpl3_real_gamma.rs` (8 tests) pins: 5.0
    round-trip to 5-channel `AudioFrame`; 5.1 round-trip to
    6-channel `AudioFrame`; silent-centre input produces
    γ5_q = γ6_q = 0 in every band; `C = (L + R) / 2` produces
    non-zero γ_q in ≥1 tonally-active band; loud-centre vs
    silent-centre inputs produce materially different bytes (the
    round-196 path would emit identical γ codewords regardless of
    centre input); `α/β/γ_scale = 0.0` matches the round-95
    scaffold byte-for-byte; `γ_scale = 0.0` reproduces the
    round-196 real-α-β bytes exactly; encoder is bit-deterministic
    for matched inputs and fresh state.
  - Real γ1..γ4 extraction (the (L,R,Ls,Rs) sub-pipeline mix
    parameters) requires per-side surround references which the
    5.0 / 5.1 PCM input layout does not carry — these stay at the
    round-95 zero-delta scaffold pending a 5.1+Ls+Rs PCM input
    layout. Real β extraction for the 7_X ACPL_3 paths, real ASPX
    envelope coding, and real Table-181 SAP-derived residual
    content (for the ACPL_1 paths) remain deferred.

- **Round 202 — real per-parameter-band α + β extraction in the
  7.0 / 7.1 SIMPLE/ASPX_ACPL_2 multichannel encoder.** The 7_X
  (immersive) counterpart to the round-144 5.0 ACPL_2 real-α-β
  path and the real-α-β upgrade of the round-107 / 114 zero-delta
  7_X ACPL_2 encoder. ACPL_2 does not transmit the Ls/Rs surround
  pair on the wire — the decoder reconstructs the surround from
  the L/R carriers + the two `acpl_data_1ch()` parameter sets per
  §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode 117:
  `z0 = 0.5·(x0·(1+α) + y·β)`, `z1 = 0.5·(x0·(1−α) − y·β)`.
  - [`crate::encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra_real_alpha_beta`]
    — real-α-β upgrade of
    [`build_7_x_acpl2_body_from_pcm_spectra`]; identical body
    schedule (2-bit `7_X_codec_mode = 3`, optional LFE
    `mono_data(b_lfe = 1)`, two `two_channel_data()` pairs, no
    joint-MDCT residual layer, trailing centre `mono_data(0)`,
    `aspx_data_2ch + aspx_data_2ch + aspx_data_1ch` envelope
    trailer) with the two trailing `acpl_data_1ch_minimal` writers
    replaced by `write_acpl_data_1ch_real_alpha_beta`. D0 module
    models (L → Ls); D1 module models (R → Rs). `acpl_config_1ch
    (FULL)` carries no `qmf_band` → `start_band = 0` so every
    parameter band participates.
  - [`crate::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_7_0_acpl2_real_alpha_beta`]
    + `_with_max_sfb` — accepts `[L, R, C, Ls, Rs, Lb, Rb]`,
    forces the 7.0 channel_mode prefix (`0b1111000`, 7 b — Table
    85 channel_mode 5), emits a 7-channel S16 PCM round-trip.
  - [`crate::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_7_1_acpl2_real_alpha_beta`]
    + `_with_max_sfb` — accepts `[L, R, C, Ls, Rs, Lb, Rb, LFE]`,
    forces the 7.1 channel_mode prefix (`0b1111001`, 7 b — Table
    88 channel_mode 6), emits an 8-channel S16 PCM round-trip
    with the LFE element written via the round-80
    `write_lfe_mono_data` shared emitter.
  - `tests/round202_7_x_acpl2_real_alpha_beta.rs` (10 tests) pins:
    7.0 / 7.1 `AudioFrame` round-trip; decoder resolves
    `SevenXCodecMode::AspxAcpl2` with both
    `acpl_data_1ch_pair[0/1]` populated; loud-surround vs
    silence-surround inputs produce materially different bytes;
    silence input round-trips with β_q = 0 in every band; encoder
    is bit-deterministic for matched inputs and fresh state;
    direct body-builder probe diverges from the round-107
    zero-delta scaffold byte stream when the caller's Ls/Rs
    spectra are non-trivial.
  - The back pair Lb / Rb is accepted for layout completeness but
    not carried by the ASPX_ACPL_2 body (the decoder's 7_X ACPL_2
    dispatch populates slots 0..4 + the LFE slot 7 — slots 5/6
    stay silent), matching the round-107 documented Table 202
    channel mapping plus the round-80 LFE PCM render at decode
    time.
  - Real β extraction for the 7_X ACPL_3 paths, real γ extraction,
    real ASPX envelope coding, real Table-181 SAP-derived residual
    content (for the ACPL_1 paths), and back-pair Lb/Rb carriage
    remain deferred.

- **Round 196 — real per-parameter-band α1 / α2 extraction in the
  5_X SIMPLE/ASPX_ACPL_3 encoder.** Layered on top of the round-193
  real β1 / β2 path: the two ACplModule2 instances in ACPL_3 share
  the (L, R) carrier pair as their (x0, x1) input, so without a
  per-side surround reference at encode time α₁ / α₂ are driven by
  the same L↔R cross-correlation extractor — `α[pb] = α_scale ·
  ρ(L, R)[pb]` with `ρ = E[L·R] / √(E[L²]·E[R²])` — clamped to the
  ALPHA_DQ table magnitude bound (±2.0 Fine / ±2.0 Coarse).
  - [`crate::encoder_acpl3::extract_alpha_q_per_band_carrier_correlation`]
    — extracts per-band α_q from the L / R MDCT spectra. The α
    parameter modulates the front/back dry-mix balance in
    ACplModule2 (Pseudocode 119): higher α → more dry energy on the
    front pair, lower α → more on the surround pair. Mono-like
    (highly-correlated) bands push α toward +1; decorrelated bands
    stay near α = 0.
  - [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_alpha_beta`]
    — drop-in replacement for the r193
    `build_5_x_acpl3_body_from_pcm_spectra_real_beta` that lifts
    α1 / α2 from the zero-delta scaffold in addition to β1 / β2.
    β3 / γ1..γ6 still zero-delta. With `α_scale = β_scale = 0.0` the
    output is byte-for-byte identical to the round-95
    `build_5_x_acpl3_body_from_pcm_spectra` scaffold.
  - [`crate::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_alpha_beta`]
    and `encode_frame_pcm_5_1_acpl3_real_alpha_beta` — caller-facing
    entry points wrapping the new builder with the same channel-mode
    forcing / MDCT analysis / TOC framing as the existing real-β
    paths.
  - `tests/round196_5_x_acpl3_real_alpha_beta.rs` (4 tests) pins:
    round-trip to 5-channel `AudioFrame`; perfect L = R correlation
    (ρ = +1) quantises to `α_q = +8` (lane 24 = 1.0); perfect
    anti-correlation L = -R (ρ = -1) quantises to `α_q = -8`;
    `α_scale = β_scale = 0.0` is byte-for-byte identical to the
    round-95 scaffold.
  - Total tests 795 (+4 over r193).

- **Round 193 — real per-parameter-band β1 / β2 extraction in the
  5_X SIMPLE/ASPX_ACPL_3 encoder.** The round-95 ASPX_ACPL_3 path
  emits all 11 ACPL parameter sets as zero-delta Huffman codewords:
  with α = β = β3 = γ = 0 the §5.7.7.6.2 Pseudocode 118 / 119
  synthesis collapses to a trivial mix that produces silent surround
  outputs from non-silent carrier inputs (structurally correct but
  perceptually inert). This round adds three encoder surface entries
  that lift β1 / β2 out of the zero-delta scaffold while keeping
  α1 / α2 / β3 / γ1..γ6 at the round-95 minimum-bit-cost defaults.
  - [`crate::encoder_acpl3::extract_beta_q_per_band_carrier_energy`]
    — extracts per-parameter-band β_q from a single carrier's MDCT
    energy distribution. The β parameter in ACplModule2 is the gain
    applied to the decorrelator output; setting it proportional to
    `√E[x²]` keeps the wet/dry balance bounded so the surround
    reconstruction tracks the carrier RMS per band.
  - [`crate::encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra_real_beta`]
    — drop-in replacement for `build_5_x_acpl3_body_from_pcm_spectra`
    that runs the carrier-energy extractor over the L / R inputs and
    emits real β1 / β2 codewords. Mirrors the round-95 wire layout
    everywhere else so the decoder walks the same Table 25 body.
  - [`crate::encoder_ims::Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3_real_beta`]
    and `encode_frame_pcm_5_1_acpl3_real_beta` — caller-facing entry
    points that wrap the new builder with the same channel-mode
    forcing, MDCT analysis and TOC framing the existing
    `encode_frame_pcm_5_0_acpl3` / `encode_frame_pcm_5_1_acpl3` use.
  - With α1 = α2 = 0 and β3 = 0 the ACplModule2 synthesis at
    parameter band `pb` reduces to
    `z0 = 0.5·(x0·g1 + x1·g2 + y0·β1)`,
    `z1 = 0.5·(x0·g1 + x1·g2 − y0·β1)`,
    and analogously `(z2, z3)` with β2 driving the second
    ACplModule2. Non-zero β1 / β2 inject the decorrelator output `y`
    that gives the Ls / Rs outputs their decorrelated spaciousness.
  - `tests/round193_5_x_acpl3_real_beta.rs` (7 tests) pins: round-trip
    to 5- / 6-channel `AudioFrame` for 5.0 / 5.1; silent input →
    all-zero β_q indices; tonal carrier + non-zero `beta_scale` →
    at least one non-zero β_q lane; `beta_scale = 0.0` is
    byte-for-byte identical to the round-95 scaffold; silent inputs
    at any `beta_scale` are scaffold-identical (carrier-energy
    extractor short-circuits to 0); non-silent tonal inputs at
    `beta_scale > 0` diverge from the round-95 scaffold (different
    β1 / β2 codeword bit-positions) while keeping the padded
    substream length identical.
  - Total tests 791 (+7 over r190).

## [0.0.6](https://github.com/OxideAV/oxideav-ac4/compare/v0.0.5...v0.0.6) - 2026-05-30

### Other

- ac4 r190: close ASPX_ACPL_1 desync — fix aspx_framing() FIXFIX prefix
- ac4 round 187 — pin ACPL_1 residual / α-β desync follow-up
- round 181 — close r128 alpha_q desync at parser indexing + aspx_data_2ch SIGNAL band count
- round 174 — fix ALPHA / BETA3 F0 cb_off (latent #1121 desync)
- round 144 — real per-band α + β extraction for 5_X ASPX_ACPL_2
- round 139 — 7.1-with-LFE ACPL_1 real per-band α + β
- round 135 — real per-band α + β extraction for 7_X ASPX_ACPL_1
- round 132 — real per-band β extraction in ACPL_1 5.0 encoder
- round 128 — real per-band α extraction in ACPL_1 5.0 encoder
- round 125 — 7.0 (3/4/0) SIMPLE/Cfg3Five multichannel encoder
- round 118 — 7.0/7.1 SIMPLE/ASPX_ACPL_1 multichannel encoder
- Round 114: 7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_2 multichannel encoder (LFE)
- Round 107: 7.0 SIMPLE/ASPX_ACPL_2 multichannel encoder
- round 103 — 5_X SIMPLE/ASPX_ACPL_1 multichannel encoder path
- round 100 — 5_X SIMPLE/ASPX_ACPL_2 multichannel encoder path
- ac4 round 95: 5_X SIMPLE/ASPX_ACPL_3 multichannel encoder path
- ac4 round 91: 7.1 (3/4/0.1) SIMPLE/Cfg3Five encoder (7 SCE + LFE)
- ac4 round 80: 5.1 SIMPLE/Cfg3Five encoder (5 SCE + LFE) + decoder LFE PCM render
- ac4 round 74: 5.0 SIMPLE/Cfg3Five multichannel forward analysis (5 SCE)
- ac4 round 52: joint M/S CPE (Path B, b_enable_mdct_stereo_proc=1)
- ac4 round 51: stereo SIMPLE/ASF split-MDCT (Path A: 2x SCE) encoder

### Fixed

- **Round 190 — close the 5_X ASPX_ACPL_1 desync the r187 tests
  pinned.** Two minimal A-SPX writers
  ([`crate::encoder_acpl3::write_aspx_data_2ch_minimal`] and
  [`write_aspx_data_1ch_minimal`]) emitted `aspx_int_class = FIXFIX`
  as the wrong prefix code: `0b11` (2 bits) instead of `0b0` (1 bit)
  per ETSI TS 103 190-1 Table 126. The decoder's
  [`crate::aspx::AspxIntClass::read`] correctly walks the prefix —
  `0` → FixFix, `10` → FixVar, `110` / `111` → VarFix / VarVar — so
  the writer's `11` start signalled the parser to read the VarFix /
  VarVar branch instead. For our config that put the parser in the
  VarFix branch with `b_iframe = 1`: it then read
  `var_bord_left` (2 b), `num_rel_left` (2 b — `num_aspx_timeslots
  = 15 > 8` makes Note-1 fields 2-bit wide), and `tsg_ptr` (2 b).
  Net: parser consumed **9 bits** in the framing where the writer
  only emitted **3 bits**, a 6-bit upstream drift that the
  silence / L-only / Ls-only test paths masked (α / β quantised to 0
  ⇒ the `acpl_data_1ch` body shape was constant minimum-cost on each
  side and the `num_param_sets_cod` bit positions on both sides
  sampled `0` within the long run of zero codewords). With non-zero
  α / β the codewords shift, the pair-1 `num_param_sets_cod` bit
  position lands on a `1`, and pair1 reads `num_param_sets = 2` —
  the r187 symptom. Fix is one line per writer: emit
  `bw.write_bit(false)` for the FIXFIX prefix, matching Table 126.
  - The r187 test #4 (`acpl1_combined_l_and_ls_pair1_currently_misaligns`)
    was renamed to `acpl1_full_round_trips_with_aligned_pair_lengths`
    and its assertion flipped from `assert_eq!(n1, 2)` (pinned
    misalignment) to `assert_eq!(n1, 1)` (post-fix). All four
    combinations now round-trip with
    `pair0.num_param_sets = pair1.num_param_sets = 1`.
  - Total tests 784 (unchanged from r187 — r190 fixed the third pin
    in place rather than adding new ones; the bit-level diagnosis is
    carried in the test file's module doc-comment instead).

### Added

- **Round 187 — characterisation tests pinning the remaining
  5_X ASPX_ACPL_1 residual / α-β-writer desync the r181 follow-up
  flagged.** Four end-to-end pinning tests in
  `tests/round187_acpl1_residual_desync_characterization.rs` sweep
  the encoder's `encode_frame_pcm_5_0_acpl1_real_alpha_beta` across
  four input combinations and assert the decoder's recovered
  `acpl_data_1ch_pair[0/1].framing.num_param_sets` so the next round
  can iterate on the residual-layer / α-β writers without regressing
  the aligned silence / L-only / Ls-only paths.
  - Silence (all-zero PCM) → both pair slots resolve
    `num_param_sets = 1`.
  - L-carrier-only (`Ls = Rs = 0`) → both pair slots resolve
    `num_param_sets = 1`; the `write_two_channel_data` carrier writer
    is exercised non-trivially while α / β stay quantised to 0
    (correlation `Σ L · Ls = 0` ⇒ α extractor returns 0; surround
    energy `E[Ls²] = 0` ⇒ β extractor returns 0).
  - Ls-residual-only (`L = R = 0`) → both pair slots still resolve
    `num_param_sets = 1`; the `write_acpl_1_residual_layer` joint-MDCT
    residual writer is exercised non-trivially with `max_sfb_master`
    non-zero band budget but α / β stay 0 because carrier energy is 0.
  - Combined L-carrier + Ls-residual (`L = 0.5`, `Ls = 0.05`) →
    pair0 still resolves `num_param_sets = 1`, but pair1 drifts to
    `num_param_sets = 2`. The pin captures this as the **currently
    expected** behaviour so the next round's residual-layer fix can
    flip the assertion back to 1 once aligned.
  - Diagnostic narrative in the test file's module doc-comment
    triangulates the bug surface: the writer→parser pairs for
    `write_acpl_data_1ch_real_alpha_beta` ↔ `parse_acpl_data_1ch`
    are bit-exact in isolation (pinned by
    `round181_alpha_desync_fix::standalone_*`); back-to-back
    invocations into the same `BitWriter` without byte alignment
    between them also round-trip cleanly. The drift therefore sits
    upstream of pair0 — either in the joint-MDCT residual writer
    (`write_acpl_1_residual_layer`) vs the inline residual walk
    inside `parse_aspx_acpl_1_2_inner_body`'s ASPX_ACPL_1 branch, or
    in the `two_channel_data()` L/R carrier writer vs
    `parse_two_channel_data` — when L and Ls are simultaneously
    non-trivial. Total tests 784 (was 780).

### Fixed

- **Round 181 — A-CPL `acpl_huff_data()` Pseudocode-121 indexing +
  `aspx_data_2ch()` SIGNAL band count.** Closes the user's "alpha_q
  desync" follow-up the round-174 ALPHA / BETA3 F0 `cb_off` fix
  deferred. Two distinct layers were involved.
  - **Layer 1 — §4.2.13.7 Table 65 / §5.7.7.7 Pseudocode 121 parser
    indexing.** Pre-r181 [`crate::acpl::parse_acpl_huff_data`] packed
    the `(num_param_bands - start_band)` Huffman-decoded values into
    a vector starting at index 0. Per the spec the same array is the
    input to Pseudocode 121's `for (i = 0; i < num_bands; i++)`
    accumulation — so positions `[0..start_band)` are zero (the
    encoder did not transmit those bands) and the F0 codeword lands at
    `values[start_band]`. The packed-from-0 layout silently shifted
    the §5.7.7.7 DIFF_FREQ accumulation by `start_band` parameter
    bands for the 5_X SIMPLE/ASPX_ACPL_1 PARTIAL path
    (`acpl_qmf_band > 0`, `start_band > 0`). The r181 fix rewrites
    [`crate::acpl::parse_acpl_huff_data`] to return a length-
    `num_param_bands` vector indexed by full param-band number — the
    spec-aligned `acpl_<SET>[ps][i]` shape Pseudocode 121 reads. The
    `[`AcplHuffParam`] doc comment now spells out the new indexing
    contract.
  - **Layer 2 — §4.2.12.4 Table 52 `aspx_data_2ch()` SIGNAL band
    count.** Per ETSI TS 103 190-1 §4.3.10.4.9 (Table 124 NOTE 3)
    `aspx_ec_data(SIGNAL, …)` reads `num_sbg_sig_lowres` SIGNAL bands
    when the corresponding `aspx_freq_res[env]` bit was emitted as 0
    and `num_sbg_sig_highres` when it was 1 or absent (when
    `freq_res_mode != Signalled` the encoder writes no in-band
    `aspx_freq_res` bit and the decoder's
    `freq_res.get(env).copied().unwrap_or(true)` fallback selects
    high-res). Pre-r181 [`crate::encoder_acpl3::write_aspx_data_2ch_minimal`]
    hard-coded `num_sbg_sig_lowres` regardless — so for the
    encoder's default `freq_res_mode = DurationDependent` config
    (20-band high-res vs 10-band low-res, no in-band freq_res bit)
    the writer emitted 10 SIGNAL F0+DF codewords per channel while
    the parser read 20. The 20-vs-10 mismatch buried every
    subsequent `acpl_data_1ch()` α / β codeword in trailing
    zero-padding, recovered as length-`num_param_bands` all-zero
    rows. r181 keys the SIGNAL band count off `cfg.signals_freq_res()`
    matching the writer's own freq_res-bit gate.
  - 4 new round-181 unit / integration tests in
    `tests/round181_alpha_desync_fix.rs`:
    [`standalone_alpha_writer_round_trips_through_parser`] (Layer 1,
    standalone writer→parser→`differential_decode`),
    [`parser_values_are_indexed_by_full_param_band_number`] (Layer 1,
    structural — confirms `values[0..start_band) == 0` and
    `values[start_band..]` carries the F0 + DF accumulation),
    [`end_to_end_acpl2_asymmetric_surround_recovers_nonzero_alpha`]
    (Layer 2 — encode 5.0 ASPX_ACPL_2 with asymmetric L/Ls energy,
    decode, assert `acpl_data_1ch_pair[0/1].alpha1[0].values`
    carries non-zero entries),
    [`end_to_end_acpl2_silence_still_round_trips`] (Layer 2 regression
    guard — silence input still yields all-zero α/β). Total tests
    780 (was 776).
  - The r132 `acpl_data_1ch_real_alpha_beta_round_trips_byte_exact`
    test is re-shaped to apply the spec-aligned Pseudocode 121 DIFF_FREQ
    accumulation directly on the parser's length-`num_param_bands`
    output (instead of cumulating the pre-r181 packed
    `(num_bands - start_band)`-length slice).
  - The 5_X SIMPLE/ASPX_ACPL_1 PARTIAL end-to-end path retains a
    separate joint-MDCT residual-layer alignment issue (the residual
    sf_data writer and the decoder's `decode_asf_long_mono_body_with_max_sfb`
    appear to read different total bit counts on non-trivial inputs)
    that the r181 ACPL_2 end-to-end tests do not exercise. Tracking
    as the remaining "alpha_q desync" follow-up — the structural
    Layer 1 + Layer 2 fixes are independent of it and land here.

- **Round 174 — ALPHA / BETA3 F0 codebook `cb_off` corrected** per ETSI
  TS 103 190-1 §A.3 Tables A.34 / A.35 (ALPHA) and A.46 / A.47 (BETA3).
  - Pre-fix `cb_off = 0` for ALPHA / BETA3 F0 conflicted with the §5.7.7.7
    Pseudocode 121 differential-decoder contract — `dequantize_alpha_index`
    re-adds the signed-lane offset (`+8` Coarse / `+16` Fine for ALPHA),
    expecting the Huffman pipeline to deliver the signed quantised lane
    `alpha_q ∈ [-N/2, +N/2]`. With `cb_off = 0` the F0 codeword
    `symbol_index` was returned raw (unsigned 0..N), shifting every
    dequant lookup by `N/2` lanes.
  - Symptom: the round-128 / 132 / 135 / 139 / 144 real-α / α+β encoder
    paths populated `alpha_q_per_band` correctly via `quantise_alpha`
    (which already returned signed lanes via its own `cb_off = 8 / 16`)
    but `write_acpl_alpha_f0_value` then wrote `symbol_index = alpha_q`
    instead of `symbol_index = alpha_q + 8 / 16`. For `alpha_q = 0` the
    writer picked the **most expensive** lane (10 / 12 bits) instead of
    the 1-bit symmetric peak; for negative `alpha_q` the writer clamped
    to lane 0; positive `alpha_q` round-tripped accidentally for
    sufficiently small magnitudes via the raw `symbol_index` lookup. The
    decoder dequant lane shifted by `cb_off`, producing wrong
    dequantised α magnitudes.
  - Fix: set `cb_off = 8` (Coarse) / `16` (Fine) for ALPHA F0 in both
    [`crate::acpl::get_acpl_hcb`] (decoder) and the matching encoder-
    local `acpl_hcb_arrays` in [`crate::encoder_acpl3`]. Same shape
    applied to BETA3 F0 (`cb_off = 4` Coarse / `8` Fine — `dequantize_beta3`
    multiplies the signed lane by `beta3_delta` directly). BETA F0 stays
    at `cb_off = 0` (unsigned magnitude — `dequantize_beta_index` takes
    `unsigned_abs` and re-applies the sign from the differential
    accumulator). Companion comment edits document the asymmetry.
  - 3 new round-174 unit tests:
    [`alpha_f0_signed_lanes_round_trip_fine_and_coarse`] sweeps every
    signed lane in both ALPHA F0 codebooks through encode → decode →
    decode value; [`beta3_f0_signed_lanes_round_trip_fine_and_coarse`]
    does the same for BETA3 F0; [`alpha_f0_zero_alpha_picks_one_bit_peak`]
    confirms the writer now picks the 1-bit symmetric peak for
    `alpha_q = 0` (down from the pre-fix 10 / 12 bits). Total tests
    776 (was 773).
  - Round-128 family tests
    (`encode_5_0_acpl1_real_alpha_emits_nonzero_alpha_when_surround_differs`
    + `..._symmetric_scaling_yields_matching_alpha`) re-shaped to assert
    on encoder byte-stream divergence rather than the decoder's
    recovered `alpha_q` — bit-position drift through the full 5_X
    SIMPLE/ASPX_ACPL_1 walker on non-silence input is independent of
    the F0 cb_off bug and still pending separate investigation (the
    user's "alpha_q desync" followup tracks it).

### Added

- **Round 144 — 5_X SIMPLE/ASPX_ACPL_2 encoder with real per-parameter-
  band α + β extraction** per ETSI TS 103 190-1 §4.2.6.6 Table 25 row
  `case ASPX_ACPL_2:` + §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode
  117. The ACPL_2 counterpart to the round-132 5_X ACPL_1 real α+β path.
  - New `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl2_real_alpha_beta`
    (+ `..._with_max_sfb`) accepts a 5-channel `[L, R, C, Ls, Rs]` input
    and produces a 5_X ASPX_ACPL_2 frame whose two trailing
    `acpl_data_1ch()` elements carry per-parameter-band α + β indices
    extracted from the (L, Ls) and (R, Rs) MDCT energy ratios.
  - The on-wire body layout is the round-100
    `build_5_x_acpl2_body_from_pcm_spectra` layout (no joint-MDCT
    residual layer — ACPL_2 reconstructs the surround from L/R + the two
    `acpl_data_1ch()` parameter sets at decode time); the Ls/Rs spectra
    are consumed only by the α + β extractors and are not transmitted.
  - New `encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra_real_alpha_beta`
    builder reuses the round-128 / 132 shared α + β analytic primitives
    (`compute_per_band_correlations` / `analytic_alpha_per_band` /
    `compute_per_band_energies` / `analytic_beta_per_band` /
    `quantise_alpha` / `quantise_beta_magnitude`) and the
    `write_acpl_data_1ch_real_alpha_beta` writer with `start_band = 0`
    (acpl_config_1ch(FULL) carries no qmf_band) so every parameter band
    participates in the α + β coding.
  - β analytic derivation per Pseudocode 116 with `y` ⊥ `x0` and
    `E[y²] ≈ E[x0²]`: `E[Ls²] = 0.5 · E[L²] · ((1 − α)² + β²)` ⇒
    `β = √max(0, 2·E[Ls²]/E[L²] − (1 − α_dq)²)`.
  - Total tests 773 (was 766): 7 new round-144 tests covering 5-channel
    AudioFrame round-trip, decoder mode resolution, on-wire body
    divergence from the round-100 scaffold for non-trivial surround,
    direct `extract_beta_q_per_band` non-zero gate, silence round-trip,
    encoder determinism, and structural pair0/pair1 population.
  - Deferred: real β extraction for ACPL_3 paths; real ASPX envelope
    coding; the round-128 ALPHA F0 writer-side `alpha_q` desync
    (deferred since r132) which currently obscures per-band on-wire α/β
    recovery through the full PCM→MDCT→writer→parser→synth chain when
    the analytic α quantises to a non-center lane.

- **Round 139 — 7.1-with-LFE (3/4/0.1) SIMPLE/ASPX_ACPL_1 encoder
  with real per-parameter-band α + β extraction** per ETSI TS 103 190-1
  §4.2.6.14 Table 33 row `case ASPX_ACPL_1:` with `b_has_lfe = 1` +
  §5.7.7.5 Pseudocode 116 + §5.7.7.6.1 Pseudocode 117. The LFE
  counterpart of the round-135 7.0 immersive real-α+β path.
  - New `Ac4ImsEncoder::encode_frame_pcm_7_1_acpl1_real_alpha_beta`
    (+ `..._with_max_sfb`) reuses the round-135
    `encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra_real_alpha_beta`
    builder with the LFE `coeffs_lfe` + `max_sfb_lfe` slots populated,
    emitting a leading `mono_data(b_lfe = 1)` element (Table 21 +
    `sf_info_lfe()` Table 35) between the I-frame config block and
    `companding_control(5)` — exactly where the decoder's
    `parse_7x_audio_data_outer(b_has_lfe = true)` reads
    `if (b_has_lfe) mono_data(1);`.
  - The on-wire body structure matches the existing round-118 7.1
    ACPL_1 path. Decoder resolves `SevenXCodecMode::AspxAcpl1` with
    `b_has_lfe = true`, both `acpl_data_1ch_pair[0/1]` populated (now
    carrying real α + β), joint-MDCT residual layer walked, LFE
    IMDCT'd into slot 7. A 60 Hz LFE tone round-trips to a non-silent
    reconstructed LFE channel.
  - +6 tests (total 766, was 760).

- **Round 132 — 5.0 SIMPLE/ASPX_ACPL_1 encoder with real per-parameter-
  band β extraction** per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 +
  §5.7.7.6.1 Pseudocode 117. Extends the round-128 real-α path: β was
  pinned to 0 (pure level-only surround image); round 132 derives a real
  per-band β magnitude from the surround/carrier energy residual that
  remains after α removes the level component.
  - Per Pseudocode 116 with the decorrelator output `y` ⊥ `x0` and
    `E[y²] ≈ E[x0²]`: `E[Ls²] = 0.5·E[x0²]·((1-α)² + β²)`. Solving for
    the magnitude gives `β = √max(0, 2·E[Ls²]/E[x0²] − (1-α)²)`, where
    `α` is the *dequantised* value the decoder reconstructs (so β closes
    the balance against the actual `(1 − α_dq)`).
  - New `encoder_acpl3` helpers: `compute_per_band_energies` (per-band
    `Σ x²` for carrier + surround), `analytic_beta_per_band` (the
    energy-residual β estimator), `quantise_beta_magnitude` (nearest
    `beta_q` index against the Table 204 / 206 column-0 grid),
    `write_acpl_beta_f0_value` / `write_acpl_beta_df_value` (ACPL BETA
    F0 + DF codebook emitters, Tables A.40 / A.41), and the optional-β
    `write_acpl_data_1ch_real_alpha_beta` body writer.
  - New build function
    `build_5_x_acpl1_body_from_pcm_spectra_real_alpha_beta` + encoder
    entry points `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_real_alpha_beta`
    (+ `..._with_max_sfb`). The on-wire body structure is unchanged from
    the round-128 path — the decoder resolves `FiveXCodecMode::AspxAcpl1`,
    both `acpl_data_1ch_pair[0/1]` populated, and the β layer now carries
    real magnitudes.
  - Public extractor/validator entry points `extract_alpha_q_per_band`,
    `extract_beta_q_per_band`, and `write_acpl_data_1ch_real_alpha_beta_bytes`
    for round-trip testing.
  - β / β3 / γ otherwise stay at the round-95 / 100 / 103 / 128 scaffold
    for non-ACPL_1 paths. Total tests 755 (was 743).
  - **Followup (round 128 latent bug, not introduced here):** the ACPL
    ALPHA F0/DF writer (`write_acpl_alpha_f0_value`/`_df_value`) clamps a
    negative `alpha_q` to lane 0, writing the wrong codeword and
    desyncing the rest of the `acpl_data_1ch` element. It only round-trips
    correctly for `alpha_q ≥ 0`. The round-132 β coding contract is
    verified byte-exact via the isolated `acpl_data_1ch` round-trip test;
    the full-substream PCM path inherits the round-128 α-writer's
    in-range limitation. Both the α-writer sign/offset fix and real β
    extraction for the 7_X / ACPL_2 / ACPL_3 paths remain deferred.

- **Round 128 — 5.0 SIMPLE/ASPX_ACPL_1 encoder with real per-parameter-
  band α extraction** per ETSI TS 103 190-1 §5.7.7.5 Pseudocode 116 +
  §5.7.7.6.1 Pseudocode 117. Replaces the round-103 zero-delta scaffold
  for the α coefficient family in the ACPL_1 path (β / β3 / γ stay at
  the round-95 / 100 / 103 zero-delta scaffold — β3 / γ only fire in
  ASPX_ACPL_3 anyway).
  - Per Pseudocode 116, above `acpl_qmf_band`: `z0 = 0.5·(x0·(1+α) +
    y·β)`, `z1 = 0.5·(x0·(1-α) - y·β)` (then `z1 *= √2` per
    Pseudocode 117). With β = 0 the surround reconstruction is a pure
    level-only image: `Ls_recon = 0.5/√2 · L · (1 − α)`. Solving for α
    that minimises `(Ls − 0.5/√2·L·(1−α))²` per parameter band gives
    `α = 1 − 2·√2 · ⟨L, Ls⟩ / ⟨L, L⟩`.
  - New `encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra_real_alpha`
    — mirrors `build_5_x_acpl1_body_from_pcm_spectra` (round 103) but
    with real α emitted via the ACPL ALPHA F0 + DF codebooks (Tables
    A.35 / A.34). Helper functions:
    - `mdct_bin_to_param_band` — maps MDCT bin → QMF subband `sb = bin
      · 64 / transform_length` → parameter band via
      `acpl::sb_to_pb` (§5.7.7.2 Table 197).
    - `compute_per_band_correlations` — computes `(Σ x·y, Σ x²)` per
      parameter band over the MDCT carrier vs. surround spectra,
      skipping bands below `start_pb` (the PARTIAL `acpl_qmf_band`
      maps to a `start_pb > 0`; bands below are M/S-recovered by the
      synth and α has no effect there).
    - `analytic_alpha_per_band` — closed-form α with clamp to ±2.0.
    - `quantise_alpha` — nearest-neighbour to `ALPHA_DQ_FINE` (Table
      203) / `ALPHA_DQ_COARSE` (Table 205); returns the signed
      `alpha_q` in `-N/2..=+N/2`.
    - `write_acpl_alpha_f0_value` / `write_acpl_alpha_df_value` —
      emit ALPHA F0 (first band) + ALPHA DF (subsequent bands using
      `delta_q = alpha_q[pb] - alpha_q[pb-1]`) codewords per the
      `acpl_hcb_arrays` table family.
    - `write_acpl_data_1ch_real_alpha` — full `acpl_data_1ch()` body
      with real α + zero-delta β (β / β3 / γ fall back to
      `write_acpl_*_zero` from round 95).
  - New encoder entry points:
    - `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_real_alpha(&[L, R, C, Ls, Rs])`
    - `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1_real_alpha_with_max_sfb(.., max_sfb, max_sfb_master)`
  - The on-wire body structure is identical to the round-103 path: the
    decoder resolves `FiveXCodecMode::AspxAcpl1`, parses both
    `acpl_data_1ch_pair[0/1]` slots, walks the joint-MDCT residual
    layer, and synthesises `[L, R, C, Ls, Rs]` via
    `acpl_synth::run_acpl_5x_pair_pcm`. The only difference is that
    the α huffman values now carry per-band non-zero deltas chosen by
    the encoder rather than the structural zero scaffold.
  - 6 new integration tests in
    `tests/round128_5_x_acpl1_real_alpha.rs`: end-to-end round-trip,
    decoder mode resolution, non-zero α emission when surround
    differs from carrier, silence round-trip, symmetric scaling
    yields a positive α in both pairs, encoder determinism.
  - Total test count: 743 (was 737) — 0 ignored, 0 failed.
  - Follow-ups (deferred): real per-band β / β3 / γ extraction
    (β = 0 simplification is spec-defensible for "level-only"
    encoding); real ASPX envelope coding; real Table-181 SAP-derived
    residual content; same real-α uplift for the ACPL_2 / ACPL_3 5_X
    paths and the 7_X ACPL_1 / ACPL_2 paths; back-pair Lb/Rb
    carriage on the ACPL paths; DT-mode (DIFF_TIME) coding using
    cross-frame state.

- **Round 125 — 7.0 (3/4/0) SIMPLE/Cfg3Five multichannel encoder
  path** per ETSI TS 103 190-1 §4.2.6.14 Table 33 + §4.2.7.5 Table 29
  (`five_channel_data()`) + §4.2.7.4 Table 26 (additional-channel
  `two_channel_data()`). The non-LFE immersive counterpart of
  round-91's 7.1 SIMPLE encoder (the 7_X analogue of round 74's 5.0 vs
  round 80's 5.1).
  - `Ac4ImsEncoder::with_7_0()` — flips the TOC channel_mode prefix to
    `0b1111000` (7 b — Table 85 channel_mode 5, 7.0 (3/4/0) → 7
    channels). The decoder's `walk_ac4_substream` then dispatches
    `channels == 7` through `parse_7x_audio_data_outer(b_has_lfe =
    false)`.
  - `Ac4ImsEncoder::encode_frame_pcm_7_0(&[L, R, C, Ls, Rs, Lb, Rb])`
    + `..._with_max_sfb(&[..], max_sfb, max_sfb_add)` — emit IMS v2
    frames in `7_X_codec_mode = SIMPLE (0)` + `coding_config =
    Cfg3Five (3)`. The five front/surround channels share the
    Cfg3Five `five_channel_data()` body (the same shape as round-74
    5.0 / round-80 5.1 / round-91 7.1); the immersive back pair
    Lb/Rb rides a trailing identity-SAP `two_channel_data()`
    (`b_use_sap_add_ch = 0`, `sap_mode = 0` on the shared
    `chparam_info`) so the decoder's
    `dispatch_7x_additional_channel_pair` routes Lb/Rb directly into
    output slots 5/6 (Table 183 row "3/4/0.x" identity path).
    `max_sfb` defaults to 40; `max_sfb_add` defaults to 40.
  - New `encoder_asf::build_7_0_simple_asf_body_from_pcm_spectra` —
    emits the substream body bytes. Body layout: `7_X_codec_mode =
    SIMPLE (0)` (2 b) + `coding_config = 3` (2 b) +
    `five_channel_data()` (shared sf_info + 5x sf_data per Table 29)
    + `b_use_sap_add_ch = 0` (1 b) + `two_channel_data()` (Lb/Rb per
    Table 26). The body is structurally the round-91 7.1 body with
    the leading `mono_data(b_lfe = 1)` element omitted (the walker's
    `if (b_has_lfe) mono_data(1);` branch is gated off for
    channel_mode 5). No companding (SIMPLE), no ASPX trailers
    (SIMPLE), no ACPL pair (SIMPLE), no trailing `mono_data(0)`
    (Cfg3Five — the 7.X trailing-mono gate is `coding_config in
    {0, 2}` only).
  - Decoder round-trip verified: 7.0 → 7-channel S16 interleaved PCM
    (1920 × 7 × 2). The 7.0 walker resolves `seven_x_mode == Simple`,
    `seven_x_b_has_lfe == false`, `lfe_mono_data == None`,
    `five_channel_data` populated with five non-empty
    `scaled_spec_per_channel` entries, identity-SAP
    `seven_x_additional_channel_data` populated with two non-empty
    `scaled_spec_per_channel` entries for the Lb/Rb pair. Slots 0..4
    synthesise via `dispatch_5x_cfg3_simple_aspx` (round 39); slots
    5/6 via `dispatch_7x_additional_channel_pair` (round 39/40
    identity-SAP path).
  - Per-channel spectral SNR on the 220/440/660/880/1100/1320/1540 Hz
    independent-tone 7.0 fixture: L=24.5 / R=24.8 / C=25.0 / Ls=23.4 /
    Rs=27.4 / Lb=25.4 / Rb=26.0 dB — all above the ≥ 20 dB floor,
    matching the round-74 / 80 / 91 SNR numbers exactly (the encoder
    reuses the same per-channel forward pipeline).
  - 8 new integration tests in `tests/round125_7_0_multichannel.rs`
    (7-channel layout, TOC declares 7 channels, sequence-counter roll,
    `with_7_0()` builder smoke-test, substream-walker confirms
    `b_has_lfe = false` + no LFE + additional pair populated,
    independent-tones round-trip, silence round-trip, per-channel
    spectral SNR ≥ 20 dB). The test helper wraps the encoder's
    `raw_ac4_frame()` payload in an Annex G `0xAC40 + frame_size`
    sync header so the decoder's `find_sync_frame` latches onto the
    genuine sync word rather than an incidental `0xAC40` byte pair in
    the body (a hazard round 91's fixture happened to dodge by data
    luck).
  - Total test count: 737 (was 729) — 0 ignored, 0 failed.
  - Follow-ups (deferred, unchanged from round 118): real per-band
    `(alpha, beta)` extraction replacing the zero-delta scaffold;
    real ASPX envelope coding; real Table-181 SAP-derived residual
    content; back-pair Lb/Rb carriage on the ACPL paths (currently
    silent on the 7_X ACPL_1 / ACPL_2 paths since those modes carry
    no SIMPLE/ASPX additional-channel block).

- **Round 118 — 7.0 / 7.1 (3/4/0(.1)) SIMPLE/ASPX_ACPL_1 multichannel
  encoder path** per ETSI TS 103 190-1 §4.2.6.14 Table 33 row
  `case ASPX_ACPL_1:` (+ `b_has_lfe = 1` for 7.1 — §4.2.6.5 Table 21
  `mono_data(b_lfe)` + §4.2.8 Table 35 `sf_info_lfe()`). The 7_X
  (immersive) counterpart to the round-103 5_X ASPX_ACPL_1 encoder and the
  encoder side of the decoder's round-27 `parse_7x_audio_data_outer`
  ASPX_ACPL_1 branch (which already reads the joint-MDCT residual layer).
  Closes the first deferred follow-up from round 114.
  - `Ac4ImsEncoder::encode_frame_pcm_7_0_acpl1(&[L, R, C, Ls, Rs, Lb, Rb])`
    + `..._with_max_sfb(&[..], max_sfb, max_sfb_master)` — emit IMS v2
    frames in `7_X_codec_mode = ASPX_ACPL_1 (2)`. Channel_mode prefix
    forced to `0b1111000` (7 b — Table 85 channel_mode 5, 7.0 (3/4/0)) so
    the decoder dispatches `channels == 7` through
    `parse_7x_audio_data_outer(b_has_lfe = false)`. `max_sfb` defaults to
    40; `max_sfb_master` (residual band bound) defaults to 20.
  - `Ac4ImsEncoder::encode_frame_pcm_7_1_acpl1(&[L, R, C, Ls, Rs, Lb, Rb,
    LFE])` + `..._with_max_sfb(&[..], max_sfb, max_sfb_master, max_sfb_lfe)`
    — the LFE counterpart: identical body plus a leading
    `mono_data(b_lfe = 1)` between the I-frame config block and
    `companding_control(5)`, exactly where
    `parse_7x_audio_data_outer(b_has_lfe = true)` reads
    `if (b_has_lfe) mono_data(1);`. Channel_mode prefix forced to
    `0b1111001` (7 b — Table 88 channel_mode 6, 7.1) → `channels == 8`.
    `max_sfb_lfe` defaults to 7 (LFE-spec cap at `tl = 1920`,
    `n_msfbl_bits = 3`).
  - New `encoder_acpl3::build_7_x_acpl1_body_from_pcm_spectra` — the 7_X
    ASPX_ACPL_2 body (round 107/114) with three structural differences,
    the same three that separate the 5_X ACPL_1 path from the 5_X ACPL_2
    path: (1) `7_X_codec_mode = 2` (vs 3); (2) `acpl_config_1ch` is
    PARTIAL via the shared `write_acpl_config_1ch_partial` emitter (6 b —
    carries `acpl_qmf_band_minus1`, so `acpl_data_1ch()` start_band
    resolves from `qmf_band` via `sb_to_pb`); (3) an explicit joint-MDCT
    residual layer via the shared `write_acpl_1_residual_layer` emitter
    (`max_sfb_master + 2× chparam_info + 2× sf_data(ASF)`) carrying the
    Ls/Rs surround pair (sSMP,3 / sSMP,4 per Table 181) after the two
    `two_channel_data()` pairs and before the trailing Cfg0 centre
    `mono_data(0)`. The SIMPLE/ASPX additional-channel block is skipped
    (the decoder only walks it for SIMPLE/Aspx modes). Reuses the
    round-80 `write_lfe_mono_data` emitter for the 7.1 LFE element.
  - Decoder round-trip verified: 7.0 ACPL_1 → 7-channel S16 interleaved
    PCM (1920 × 7 × 2); 7.1 ACPL_1 → 8-channel S16 (with LFE slot 7). The
    decoder resolves `seven_x_mode == AspxAcpl1`, the PARTIAL config
    (non-zero `qmf_band`), both `two_channel_data` pairs, the residual
    pair + `max_sfb_master`, the Cfg0 centre, and both
    `acpl_data_1ch_pair[0/1]`. The LFE spectrum IMDCT's into slot 7 (round
    80 render); `[L, R, C, Ls, Rs]` slots 0..4 synthesise via
    `acpl_synth::run_acpl_5x_pair_pcm`; the back pair Lb/Rb (slots 5/6)
    stays silent per the Table 202 mapping.
  - 8 new integration tests in `tests/round118_7_x_acpl1_encoder.rs`
    (7-channel + 8-channel layout, sequence-counter roll, 7.0 + 7.1
    full-body decoder resolution, LFE-slot-non-silent for a 60 Hz LFE
    tone, silence round-trip, small-residual-budget round-trip recovering
    the clamped `max_sfb_master`).
  - Total test count: 729 (was 721) — 0 ignored, 0 failed.
  - Follow-ups (deferred): real per-band `(alpha, beta)` extraction
    replacing the zero-delta scaffold; real ASPX envelope coding; real
    Table-181 SAP-derived residual content (the residual `sf_data`
    currently codes the raw Ls/Rs spectra); back-pair Lb/Rb carriage
    (currently silent on the ACPL paths).

- **Round 114 — 7.1 (3/4/0.1) SIMPLE/ASPX_ACPL_2 multichannel encoder
  path** per ETSI TS 103 190-1 §4.2.6.14 Table 33 row `case ASPX_ACPL_2:`
  with `b_has_lfe = 1` + §4.2.6.5 Table 21 (`mono_data(b_lfe)`) + §4.2.8
  Table 35 (`sf_info_lfe()`) + Table 106 column 4 (`n_msfbl_bits`). The
  LFE counterpart of the round-107 7.0 ASPX_ACPL_2 encoder — the body is
  identical except a leading `mono_data(b_lfe = 1)` element is emitted
  between the I-frame config block and `companding_control(5)`, exactly
  where the decoder's `parse_7x_audio_data_outer(b_has_lfe = true)` reads
  `if (b_has_lfe) mono_data(1);` (§4.2.6.14 Table 33). Closes the first
  deferred follow-up from round 107.
  - `Ac4ImsEncoder::encode_frame_pcm_7_1_acpl2(&[L, R, C, Ls, Rs, Lb, Rb,
    LFE])` + `..._with_max_sfb(&[..], max_sfb, max_sfb_lfe)` — emit IMS v2
    frames in `7_X_codec_mode = ASPX_ACPL_2 (3)` with the LFE element.
    Channel_mode prefix forced to `0b1111001` (7 b — Table 88 channel_mode
    6, 7.1 (3/4/0.1)) so the decoder dispatches `channels == 8` through
    `parse_7x_audio_data_outer(b_has_lfe = true)`. `max_sfb` defaults to
    40; `max_sfb_lfe` defaults to 7 (the LFE-spec cap at `tl = 1920`,
    `n_msfbl_bits = 3`).
  - `encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra` gained two
    parameters — `max_sfb_lfe: Option<u32>` and `coeffs_lfe:
    Option<&[f32]>`. When both are `Some` the shared
    `write_lfe_mono_data` emitter (round 80) prepends the LFE
    `mono_data(b_lfe = 1)` element at the spec-correct position; with both
    `None` the body is the unchanged round-107 7.0 form (the round-107
    7.0 caller passes `None`/`None`).
  - Decoder round-trip verified: 7.1 ACPL_2 → 8-channel S16 interleaved
    PCM (1920 samples × 8 ch × 2 bytes). The decoder walks the full
    Table 33 ASPX_ACPL_2 + LFE body and resolves `seven_x_mode ==
    AspxAcpl2`, `seven_x_b_has_lfe == true`, `lfe_mono_data.is_some()`,
    `acpl_config_1ch_full.is_some()`, `two_channel_data.len() == 2`,
    `cfg0_centre_mono.is_some()`, and both `acpl_data_1ch_pair[0/1]`. The
    LFE spectrum is IMDCT'd into slot 7 via the existing round-80 LFE
    render (`Ac4Decoder::receive_frame`, `channels == 8`); the round-107
    `[L, R, C, Ls, Rs]` slots 0..4 synthesis (via
    `acpl_synth::run_acpl_5x_pair_pcm`) and the silent back pair Lb/Rb
    (slots 5/6, Table 202 ACPL_2 mapping) are unchanged.
  - 6 new integration tests in `tests/round114_7_1_acpl2_encoder.rs`
    (8-channel layout, sequence-counter roll, full-body + LFE decoder
    resolution, LFE-slot-non-silent for a 60 Hz LFE tone, silence
    round-trip, wide-max_sfb round-trip) + 1 new unit test in
    `encoder_acpl3::tests` (`build_7_x_acpl2_body_with_lfe_decoder_resolves_lfe`).
  - Total test count: 721 (was 714) — 0 ignored, 0 failed.
  - Follow-ups (deferred): the 7_X ASPX_ACPL_1 path (PARTIAL config +
    joint-MDCT residual layer, the 7_X analogue of round 103); real
    per-band `(alpha, beta)` extraction replacing the zero-delta scaffold;
    real ASPX envelope coding; back-pair Lb/Rb carriage (currently silent
    on the ACPL_2 path).

- **Round 107 — 7.0 SIMPLE/ASPX_ACPL_2 multichannel encoder path** per
  ETSI TS 103 190-1 §4.2.6.14 Table 33 row `case ASPX_ACPL_2:` +
  §4.2.12.1 Table 50 (aspx_config) + §4.2.13.1 Table 59
  (acpl_config_1ch FULL) + §4.2.7.4 Table 26 (two_channel_data) +
  §4.2.6.5 Table 21 (mono_data) + §4.2.12.4 Table 52 (aspx_data_2ch) +
  §4.2.12.3 Table 51 (aspx_data_1ch) + §4.2.13.3 Table 61
  (acpl_data_1ch) + §4.2.11 Table 49 (companding_control). The 7_X
  (immersive) symmetric counterpart to the round-100 5_X ASPX_ACPL_2
  encoder and the encoder side of the decoder's round-27
  `parse_7x_audio_data_outer` ASPX_ACPL_2 branch. Reuses the same 1ch
  ACPL / ASPX parameter shape (Pseudocode 117) as the 5_X path but emits
  the 7_X channel element's distinct framing.
  - `Ac4ImsEncoder::encode_frame_pcm_7_0_acpl2(&[L, R, C, Ls, Rs, Lb, Rb])`
    + `..._with_max_sfb(&[..], max_sfb)` — emit IMS v2 frames in
    `7_X_codec_mode = ASPX_ACPL_2 (3)`. Channel_mode prefix forced to
    `0b1111000` (7 b — Table 85 channel_mode 5, 7.0 (3/4/0)) so the
    decoder dispatches `channels == 7` through
    `parse_7x_audio_data_outer(b_has_lfe = false)`.
  - `encoder_acpl3::build_7_x_acpl2_body_from_pcm_spectra` — shared body
    builder. Layout: `7_X_codec_mode = 3` (**2 b**, vs the 5_X 3-bit
    field) + I-frame block (`aspx_config()` 15 b + `acpl_config_1ch(FULL)`
    3 b) + `companding_control(5)` (sync=1, on=1 — the 2-bit sync-on wire
    shape is identical to companding_control(2/3)) + `coding_config = 0`
    (**2 b**, Cfg0) + `b_2ch_mode` + `two_channel_data()` (L/R carriers) +
    `two_channel_data()` (Ls/Rs carriers) + trailing Cfg0 `mono_data(0)`
    (centre) + I-frame `aspx_data_2ch() + aspx_data_2ch() +
    aspx_data_1ch()` envelope trailer + two `acpl_data_1ch()` parameter
    sets (Pseudocode 117 D0 / D1). Structural differences from the 5_X
    ACPL_2 path: 2-bit codec_mode, 2-bit coding_config, the centre
    `mono_data(0)` moves out of the coding_config switch to a single
    trailing element, the body carries two stereo pairs (the surround pair
    rides the second `two_channel_data`), and the ASPX trailer carries an
    extra `aspx_data_2ch()`. The ASPX_ACPL_1-only joint-MDCT residual
    layer and the SIMPLE/ASPX additional-channel block are both skipped
    for ASPX_ACPL_2.
  - Decoder round-trip verified: 7.0 ACPL_2 → 7-channel S16 interleaved
    PCM (1920 samples × 7 ch × 2 bytes). The decoder walks the full
    Table 33 ASPX_ACPL_2 body and resolves
    `seven_x_mode == AspxAcpl2`, `acpl_config_1ch_full.is_some()`,
    `two_channel_data.len() == 2`, `cfg0_centre_mono.is_some()`, and both
    `acpl_data_1ch_pair[0/1].is_some()` — the existing round-37/40 7_X
    pair dispatch synthesises `[L, R, C, Ls, Rs]` (slots 0..4) via
    `acpl_synth::run_acpl_5x_pair_pcm`; the back pair Lb/Rb (slots 5/6)
    stays silent per the documented Table 202 ACPL_2 channel mapping.
  - 5 new integration tests in `tests/round107_7_x_acpl2_encoder.rs`
    (7-channel layout, sequence-counter roll, full-body decoder resolution
    including the two-pair Cfg0 + no-residual assertion, silence
    round-trip, wide-max_sfb round-trip) + 1 new unit test in
    `encoder_acpl3::tests` (full-body decoder resolution via
    `parse_7x_audio_data_outer`).
  - Total test count: 714 (was 708) — 0 ignored, 0 failed.
  - Follow-ups (deferred): the 7.1 (LFE) ASPX_ACPL_2 path
    (`b_has_lfe = true` leading `mono_data(1)`); the 7_X ASPX_ACPL_1 path
    (PARTIAL config + joint-MDCT residual layer, the 7_X analogue of
    round 103); real per-band `(alpha, beta)` extraction replacing the
    zero-delta scaffold; real ASPX envelope coding; back-pair Lb/Rb
    carriage (currently silent on the ACPL_2 path).

- **Round 103 — 5_X SIMPLE/ASPX_ACPL_1 multichannel encoder path** per
  ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_1:` +
  §4.2.12.1 Table 50 (aspx_config) + §4.2.13.1 Table 59
  (acpl_config_1ch PARTIAL) + §4.2.7.4 Table 26 (two_channel_data) +
  §4.2.10.1 Table 47 (chparam_info) + §4.2.6.5 Table 21 (mono_data) +
  §4.2.12.4 Table 52 (aspx_data_2ch) + §4.2.12.3 Table 51
  (aspx_data_1ch) + §4.2.13.3 Table 61 (acpl_data_1ch) + §4.2.11
  Table 49 (companding_control). Completes the round-100 follow-up: the
  symmetric encoder for the decoder's round-25
  `parse_aspx_acpl_1_2_inner_body` ASPX_ACPL_1 branch (Pseudocode 117),
  including the joint-MDCT residual layer that ASPX_ACPL_2 omits.
  Extends the `encoder_acpl3` module:
  - `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl1(&[L, R, C, Ls, Rs])` +
    `..._with_max_sfb(&[L, R, C, Ls, Rs], max_sfb, max_sfb_master)` —
    emit IMS v2 frames in `5_X_codec_mode = ASPX_ACPL_1 (2)`.
    Channel_mode prefix forced to `0b1101` (5 ch) per Table 85 so the
    decoder dispatches `channels == 5` through
    `parse_5x_audio_data_outer(b_has_lfe = false)`.
  - `encoder_acpl3::build_5_x_acpl1_body_from_pcm_spectra` — shared body
    builder. Layout: `5_X_codec_mode = 2` (3 b) + I-frame block
    (`aspx_config()` 15 b + `acpl_config_1ch(PARTIAL)` 6 b) +
    `companding_control(3)` (sync=1, on=1) + `coding_config = 0` (1 b) +
    `two_channel_data()` (L/R carriers) + **ASPX_ACPL_1 joint-MDCT
    residual layer** (`max_sfb_master` in n_side bits + 2× `chparam_info`
    + 2× `sf_data(ASF)` for the Ls/Rs surround pair sSMP,3 / sSMP,4 per
    Table 181) + Cfg0 `mono_data(0)` (centre) + I-frame `aspx_data_2ch()`
    + `aspx_data_1ch()` + two `acpl_data_1ch()` parameter sets
    (Pseudocode 117 D0 / D1). The residual layer + PARTIAL config are the
    two structural differences from the round-100 ACPL_2 path: ASPX_ACPL_1
    transmits the surround residual explicitly (so the encoder takes a
    full 5-channel input) rather than reconstructing Ls/Rs purely from the
    L/R carriers.
  - New bit-exact emitters: `write_acpl_config_1ch_partial` (Table 59,
    2-bit id + 1-bit quant_mode + 3-bit acpl_qmf_band_minus1) and
    `write_acpl_1_residual_layer` (max_sfb_master clamped to the n_side
    band budget + identity-SAP chparam pair + two `sf_data(ASF)` bodies
    via the shared `prepare_stereo_channel` forward pipeline). The
    `acpl_data_1ch()` start_band is resolved from the PARTIAL config's
    `qmf_band` via `crate::acpl::sb_to_pb` (vs ACPL_2's FULL config →
    start_band 0).
  - Decoder round-trip verified: 5.0 ACPL_1 → 5-channel S16 interleaved
    PCM (1920 samples × 5 ch × 2 bytes). The decoder walks the full
    Table 25 ASPX_ACPL_1 body and resolves `five_x_mode == AspxAcpl1`,
    `acpl_config_1ch_partial.is_some()` (with non-zero `qmf_band`),
    `acpl_1_residual_max_sfb_master.is_some()`, both
    `acpl_1_residual_pair[0/1].is_some()` (sSMP,3 / sSMP,4),
    `cfg0_centre_mono.is_some()`, and both `acpl_data_1ch_pair[0/1]` —
    the 5-channel `[L, R, C, Ls, Rs]` synthesis runs via
    `acpl_synth::run_acpl_5x_pair_pcm` (Pseudocode 117).
  - 5 new integration tests in `tests/round103_5_x_acpl1_encoder.rs`
    (5-channel layout, sequence-counter roll, full-body decoder
    resolution including the residual layer, silence round-trip,
    small-residual-budget round-trip) + 3 new unit tests in
    `encoder_acpl3::tests` (acpl_config_1ch_partial round-trip,
    residual-layer clamp behaviour, full-body decoder resolution).
  - Total test count: 708 (was 700) — 0 ignored, 0 failed.
  - Follow-ups (deferred): real per-band `(alpha, beta)` parameter
    extraction replacing the zero-delta scaffold; real ASPX envelope
    coding; real joint-MDCT residual content (the residual sf_data
    currently codes the raw Ls/Rs spectra — a future round should derive
    the proper sSMP,3 / sSMP,4 residual from the Table-181 SAP first
    stage); matching 7_X ASPX_ACPL_{1,2} encoder paths (the 7_X walker
    shares this 1ch acpl/aspx shape).

- **Round 100 — 5_X SIMPLE/ASPX_ACPL_2 multichannel encoder path** per
  ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_2:` +
  §4.2.12.1 Table 50 (aspx_config) + §4.2.13.1 Table 59
  (acpl_config_1ch FULL) + §4.2.7.4 Table 26 (two_channel_data) +
  §4.2.6.5 Table 21 (mono_data) + §4.2.12.4 Table 52 (aspx_data_2ch)
  + §4.2.12.3 Table 51 (aspx_data_1ch) + §4.2.13.3 Table 61
  (acpl_data_1ch) + §4.2.11 Table 49 (companding_control). Symmetric
  counterpart to the decoder's round-25 `parse_aspx_acpl_1_2_inner_body`
  walker (Pseudocode 117). Extends the `encoder_acpl3` module:
  - `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl2(&[L, R, C])` +
    `..._with_max_sfb(&[L, R, C], max_sfb)` — emit IMS v2 frames in
    `5_X_codec_mode = ASPX_ACPL_2 (3)`. Channel_mode prefix forced to
    `0b1101` (5 ch) per Table 85 so the decoder dispatches `channels
    == 5` through `parse_5x_audio_data_outer(b_has_lfe = false)`.
  - `encoder_acpl3::build_5_x_acpl2_body_from_pcm_spectra` — shared
    body builder. Layout: `5_X_codec_mode = 3` (3 b) + I-frame block
    (`aspx_config()` 15 b + `acpl_config_1ch(FULL)` 3 b) +
    `companding_control(3)` (sync=1, on=1) + `coding_config = 0` (1 b,
    AcplLite2 / two-channel false-branch) + `two_channel_data()` (L/R
    carriers) + Cfg0 `mono_data(0)` (centre carrier) + I-frame
    `aspx_data_2ch()` + `aspx_data_1ch()` + two `acpl_data_1ch()`
    parameter sets (Pseudocode 117 D0 / D1). The ASPX_ACPL_1-only
    joint-MDCT residual layer (`max_sfb_master + 2× chparam_info +
    2× sf_data`) is **skipped** for ACPL_2 — that's the structural
    difference that makes the ACPL_2 path the cleanest encoder win.
  - New bit-exact emitters: `write_acpl_config_1ch_full` (Table 59,
    2-bit id + 1-bit quant_mode, no qmf_band), `write_two_channel_data`
    (Table 26 shared `sf_info(ASF)` + identity-SAP `chparam_info` +
    2× `sf_data`), `write_mono_data_centre` (Table 21 non-LFE:
    `spec_frontend = 0` + `sf_info(ASF)` + `sf_data`),
    `write_aspx_data_1ch_minimal` (Table 51 FIXFIX num_env=1 path),
    `write_acpl_data_1ch_minimal` (Table 61: `acpl_framing_data` +
    `acpl_ec_data(ALPHA)` + `acpl_ec_data(BETA)`, 1 param set, DF
    zero-delta). The 1ch ASPX SIGNAL band count uses
    `num_sbg_sig_highres` (matching `parse_aspx_ec_data`'s empty-
    `freq_res` fallback when `freq_res_mode != Signalled`).
  - Decoder round-trip verified: 5.0 ACPL_2 → 5-channel S16
    interleaved PCM (1920 samples × 5 ch × 2 bytes). The decoder
    walks the full Table 25 ASPX_ACPL_2 body and resolves
    `five_x_mode == AspxAcpl2`, `acpl_config_1ch_full.is_some()`,
    `two_channel_data.len() == 1`, `cfg0_centre_mono.is_some()`, and
    both `acpl_data_1ch_pair[0/1].is_some()` — the 5-channel
    `[L, R, C, Ls, Rs]` synthesis runs via
    `acpl_synth::run_acpl_5x_pair_pcm` (Pseudocode 117). With all-zero
    ACPL parameter deltas Ls/Rs collapses to ducker-driven
    reconstruction from the L/R carriers.
  - 4 new integration tests in `tests/round100_5_x_acpl2_encoder.rs`
    (5-channel layout, sequence-counter roll, full-body decoder
    resolution, silence round-trip) + 5 new unit tests in
    `encoder_acpl3::tests` (bit-order round-trips for
    acpl_config_1ch_full / two_channel_data / mono_data(0) /
    acpl_data_1ch + aspx_data_1ch emit).
  - Total test count: 700 (was 691) — 0 ignored, 0 failed.
  - Follow-ups (deferred): the ASPX_ACPL_1 encoder path (adds the
    joint-MDCT residual layer + PARTIAL-mode `acpl_config_1ch` with
    `acpl_qmf_band`); real per-band `(alpha, beta)` extraction
    replacing the zero-delta scaffold; matching 7_X ASPX_ACPL_{1,2}
    encoder paths (the 7_X walker shares this 1ch acpl/aspx shape).

- **Round 95 — 5_X SIMPLE/ASPX_ACPL_3 multichannel encoder path** per
  ETSI TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_3:` +
  §4.2.12.1 Table 50 (aspx_config) + §4.2.13.2 Table 60 (acpl_config_2ch)
  + §4.2.13.4 Table 62 (acpl_data_2ch) + §4.2.12.4 Table 52
  (aspx_data_2ch) + §4.2.11 Table 49 (companding_control) + §4.2.6.3
  Table 22 (stereo_data). Symmetric counterpart to the decoder's round-34
  `parse_5x_audio_data_outer` ASPX_ACPL_3 walker (5a58f6a). New
  `encoder_acpl3` module:
  - `Ac4ImsEncoder::encode_frame_pcm_5_0_acpl3(&[L, R, C])` and
    `encode_frame_pcm_5_1_acpl3(&[L, R, C, LFE])` — emit IMS v2 frames
    in 5_X_codec_mode = ASPX_ACPL_3 (4). Channel_mode prefix forced to
    `0b1101` (5 ch) / `0b1110` (6 ch) per Table 85.
  - `encoder_acpl3::build_5_x_acpl3_body_from_pcm_spectra` — shared
    body builder. Layout: `5_X_codec_mode = 4` (3 b) + I-frame block
    (`aspx_config()` 15 b + `acpl_config_2ch()` 4 b) + optional LFE
    `mono_data(b_lfe = 1)` + `companding_control(2)` (sync=1, on=1) +
    `stereo_data()` (split-MDCT path) + `aspx_data_2ch()` + `acpl_data_2ch()`.
  - `encoder_acpl3::write_aspx_config` / `write_acpl_config_2ch` /
    `write_companding_control_2ch_sync_on` — bit-exact emitters for
    the small configuration elements. Round-trip-verified against the
    matching parsers via unit tests.
  - ASPX/A-CPL parameter bits emitted as minimum-bit-cost zero-delta
    Huffman codewords: `pick_zero_delta_cw(len, cw, cb_off)` picks the
    entry at `index == cb_off` (zero delta for DF/DT) and
    `pick_min_len_cw` picks the smallest-length entry (used for F0
    seeds). Covers all 18 ASPX HCBs (Annex A.2 Tables A.16-A.33) and
    all 24 ACPL HCBs (Annex A.3 Tables A.34-A.57).
  - `write_aspx_data_2ch_minimal` emits the FIXFIX + num_env=1 path
    with `aspx_balance = 1`, all-FREQ delta directions, and per-channel
    SBG counts derived from `aspx::derive_aspx_frequency_tables`.
  - `write_acpl_data_2ch_minimal` emits `acpl_framing_data()` (smooth
    interp, num_param_sets = 1) + 11 × `acpl_huff_data()` (alpha1/2,
    beta1/2/3, gamma1..6) with `diff_type = 0` (FREQ) and zero-delta
    DF codewords.
  - Decoder round-trip verified: 5.0 ACPL_3 → 5-channel S16
    interleaved PCM (1920 samples × 5 ch × 2 bytes); 5.1 ACPL_3 →
    6-channel S16. The decoder walks the full Table 25 body
    (`parse_stereo_data_body` + `parse_aspx_data_2ch_body` +
    `parse_acpl_data_2ch`) and resolves `five_x_mode == AspxAcpl3` +
    `acpl_config_2ch.is_some() && acpl_data_2ch.is_some()`. The 5-channel
    `[L, R, C, Ls, Rs]` synthesis runs via `acpl_synth::run_acpl_5x_mch_pcm`
    (Pseudocode 118) with all-zero ACPL parameter deltas — Ls/Rs
    collapses to ducker-driven reconstruction from the L/R carriers.
  - 4 new integration tests in `tests/round95_5_x_acpl3_encoder.rs`:
    encode_5_0_acpl3_produces_5_channel_audio_frame,
    encode_5_1_acpl3_produces_6_channel_audio_frame,
    encode_5_0_acpl3_advances_sequence_counter,
    encode_5_0_acpl3_decoder_resolves_aspx_acpl_3_mode.
  - 7 new unit tests in `encoder_acpl3::tests` covering bit-order
    round-trips for aspx_config / acpl_config_2ch / companding_control /
    acpl_data_2ch + minimum-cost / zero-delta HCB picker invariants.
  - Total test count: 691 (was 680) — 0 ignored, 0 failed.
  - Follow-ups (deferred to subsequent rounds): replace zero-delta
    ACPL parameter writer with real QMF-domain `(alpha, beta, gamma)`
    parameter extractor (per-band downmix correction estimated from
    L/R/Ls/Rs source PCM); replace zero-delta ASPX envelope coder
    with real envelope extraction; matching encoder paths for
    5_X_codec_mode in `{ASPX_ACPL_1, ASPX_ACPL_2}` (Pseudocode 117).

- **Round 91 — 7.1 (3/4/0.1) SIMPLE/ASF Cfg3Five multichannel forward
  analysis (7 SCE + LFE) encoder + decoder 7_X SIMPLE/Cfg3Five core
  render** per ETSI TS 103 190-1 §4.2.6.14 Table 33 + §4.2.7.5 Table 29
  + §4.2.7.4 Table 26 + §4.2.8 Table 35 + §4.3.3.7.1 Table 88
  (channel_mode 6 = 7.1 (3/4/0.1)):
  - `encoder_asf::build_7_1_simple_asf_body_from_pcm_spectra(transform_length,
    max_sfb, max_sfb_add, max_sfb_lfe, &[&[f32]; 8], pad_target_bytes)` —
    emits the full 7.1 multichannel `audio_data` body for `7_X_codec_mode
    = SIMPLE`, `b_has_lfe = 1`, `coding_config = 3` (Cfg3Five). Differs
    from the round-80 5.1 builder in three places per Table 33: (1)
    `7_X_codec_mode` is 2 bits (vs 3 for `5_X_codec_mode` per Table 25);
    (2) SIMPLE skips the leading `companding_control(5)` (5_X SIMPLE
    skips it too, but the 7_X-only ASPX path also skips it whereas 5_X
    ASPX would emit it); (3) after the inner `five_channel_data()` for
    L/R/C/Ls/Rs the SIMPLE/ASPX additional-channel block emits
    `b_use_sap_add_ch (1 b) = 0 + two_channel_data()` carrying the
    immersive pair Lb/Rb (identity SAP with `sap_mode = 0` on its
    `chparam_info`) per Table 26. No trailing `mono_data(0)` for Cfg3
    (gated on `coding_config in {0, 2}` only); no ASPX trailers / ACPL
    data pair for SIMPLE.
  - `Ac4ImsEncoder::with_7_1()` — channel-mode prefix `0b1111001` (7 b
    — Table 88 channel_mode 6) for the 7.1 (3/4/0.1) layout. Builder
    method parity with `with_5_0()` / `with_5_1()`.
  - `Ac4ImsEncoder::encode_frame_pcm_7_1(&[L, R, C, Ls, Rs, Lb, Rb, LFE])`
    and `encode_frame_pcm_7_1_with_max_sfb(..., max_sfb, max_sfb_add,
    max_sfb_lfe)` — force the 7.1 channel_mode prefix so the decoder's
    `walk_ac4_substream` dispatches `channels == 8` through
    `parse_7x_audio_data_outer(b_has_lfe = true)`, run the round-50
    forward MDCT pipeline (KBD-windowed MDCT + DP-optimal sectioning +
    HCB1..11 + SNF) independently per channel, and wrap the body in
    the v2 IMS TOC.
  - **Decoder 7_X SIMPLE/Cfg3Five core render** in
    `Ac4Decoder::receive_frame`: when `seven_x_mode in {SIMPLE, ASPX}`
    and `seven_x_coding_config == Cfg3Five`, drive
    `dispatch_5x_cfg3_simple_aspx` on the inner `five_channel_data` to
    IMDCT slots 0..4 (L/R/C/Ls/Rs). The 7_X walker had been populating
    `tools.five_channel_data` for ~50+ rounds but the inner 5-channel
    PCM was never rendered — only slots 5/6 (the additional-pair F/G)
    and slot 7 (LFE, round 80) were touched. This change inherits the
    5_X core IMDCT/KBD/overlap-add chain unchanged (the per-channel
    body shape is identical to the 5_X Cfg3Five case). ASPX trailer
    plumbing (cfg3_aspx_lr / cfg3_aspx_ls_rs / cfg3_aspx_centre) is
    passed as `None` — the 7_X walker has its own ASPX trailer slots
    that need separate wiring (deferred).
  - **Spectral SNR** on the 220 / 440 / 660 / 880 / 1100 / 1320 /
    1540 Hz independent-tone 7.1 fixture: L=24.5 / R=24.8 / C=25.0 /
    Ls=23.4 / Rs=27.4 / Lb=25.4 / Rb=26.0 dB — all above the ≥ 20 dB
    floor, identical to round-80 5.1 for the L/R/C/Ls/Rs channels (same
    forward pipeline) with Lb/Rb tracking the same SNR-bandwidth
    relationship. The 60 Hz LFE tone round-trips to a non-silent
    reconstructed LFE channel via the shared round-80 LFE render in
    `receive_frame`.
  - **New test suite** `tests/round91_7_1_multichannel.rs` (8 tests):
    layout (8-channel S16 interleaved PCM), TOC channels=8, sequence
    counter roll, `with_7_1()` builder TOC contract, walker contract
    (`b_has_lfe == true`, populated `lfe_mono_data.scaled_spec`,
    populated `five_channel_data` + `seven_x_additional_channel_data`
    with `b_use_sap_add_ch = false`), silence round-trip, independent-
    tone round-trip with audible PCM in every non-LFE slot + audible
    LFE + verified Lb/Rb separation, and per-channel SNR ≥ 20 dB.
  - ACPL_3 multichannel ASPX / A-CPL encoder remains deferred (the
    5_X ASPX_ACPL_3 / 5_X ASPX_ACPL_{1,2} encoder paths haven't
    landed; 7_X ASPX modes are gated on those).

- **Round 80 — 5.1 SIMPLE/ASF Cfg3Five multichannel forward analysis (5 SCE
  + LFE) encoder** per ETSI TS 103 190-1 §4.2.6.6 Table 25 (`if (b_has_lfe)
  mono_data(1);`) + §4.2.7.5 Table 29 + §4.2.8 (`sf_info_lfe()` Table 35 /
  Table 106 column 4 `n_msfbl_bits`):
  - `encoder_asf::build_5_1_simple_asf_body_from_pcm_spectra(transform_length,
    max_sfb, max_sfb_lfe, &[&[f32]; 6], pad_target_bytes)` — emits the full
    5.1 multichannel `audio_data` body for `5_X_codec_mode = SIMPLE`,
    `b_has_lfe = 1`, `coding_config = 3` (Cfg3Five): the round-74 5.0
    `five_channel_data()` payload is preceded by an LFE `mono_data(1)`
    element (no leading `spec_frontend` bit per Table 21,
    `b_long_frame = 1`, `sf_info_lfe()` with `max_sfb_lfe` in
    `n_msfbl_bits` bits, then a single
    `(section + spectral + scalefac + snf)` ASF body capped to the LFE
    band budget). At `tl = 1920` `n_msfbl_bits = 3`, so the LFE channel
    spans at most 7 scalefactor bands (≈0–350 Hz) — comfortably more than
    the 120 Hz LFE crossover and the 60 Hz tone used by the new tests.
  - `Ac4ImsEncoder::encode_frame_pcm_5_1(&[L, R, C, Ls, Rs, LFE])` and
    `encode_frame_pcm_5_1_with_max_sfb(..., max_sfb, max_sfb_lfe)` —
    forces the 5.1 channel_mode prefix (`0b1110`, 4 b — Table 85
    channel_mode 4) so the decoder's `walk_ac4_substream` dispatches
    `channels == 6` through `parse_5x_audio_data_outer(b_has_lfe = true)`,
    runs the round-74 forward MDCT pipeline (KBD-windowed MDCT +
    DP-optimal sectioning + HCB1..11 + SNF) per channel, and wraps the
    body in the v2 IMS TOC with `bitstream_version = 2`.
  - **Decoder LFE PCM render** in `Ac4Decoder::receive_frame`: when
    `channels == 6` (5.1) or `channels == 8` (7.1) and the 5_X / 7_X
    walker populated `tools.lfe_mono_data.scaled_spec`, the LFE
    spectrum is IMDCT'd into the trailing PCM slot (slot 5 for 5.1,
    slot 7 for 7.1) using the per-channel overlap-add history.
    Pre-r80 the LFE block was parsed but its PCM was silently dropped.
  - **Spectral SNR** on the 220 / 440 / 660 / 880 / 1100 Hz independent-tone
    fixture matches the round-74 5.0 numbers
    (L=24.5 / R=24.8 / C=25.0 / Ls=23.4 / Rs=27.4 dB) and clears the ≥ 20 dB
    floor; the 60 Hz LFE tone round-trips to a non-silent reconstructed
    LFE channel. 7.0 / 7.1 (immersive add-channel pair) and the ASPX /
    A-CPL multichannel modes remain deferred.
  - **New test suite** `tests/round80_5_1_multichannel.rs` (7 tests) covers
    the layout, sequence-counter rolling, walker contract
    (`b_has_lfe == true` + populated `lfe_mono_data.scaled_spec`),
    silence round-trip, independent-tone round-trip with audible LFE, and
    per-channel SNR ≥ 20 dB.

- **Round 74 — 5.0 SIMPLE/ASF Cfg3Five multichannel forward analysis (5 SCE)
  encoder** per ETSI TS 103 190-1 §4.2.6.6 Table 25 row
  `case SIMPLE: coding_config == 3` + §4.2.7.5 Table 29 (`five_channel_data()`)
  + §4.2.10.1 Table 47 (`chparam_info()`):
  - `encoder_asf::build_5_0_simple_asf_body_from_pcm_spectra(transform_length,
    max_sfb, &[&[f32]; 5], pad_target_bytes)` — emits the full 5.0
    multichannel audio_data body for `5_X_codec_mode = SIMPLE` /
    `coding_config = 3` (Cfg3Five): `audio_size_value (15 b)` +
    `b_more_bits (1 b)` + byte_align + `5_X_codec_mode = SIMPLE (3 b)` +
    `coding_config = 3 (2 b)` + `five_channel_data()` (shared
    `asf_transform_info` + shared `asf_psy_info` + `five_channel_info` with
    `chel_matsel = 0` + 5x `chparam_info` with `sap_mode = 0` for identity
    SAP + 5x `sf_data(ASF)` bodies). No joint-MDCT mixing happens at decode
    time — every output channel comes straight from its own `sf_data(ASF)`
    body. SIMPLE has no `aspx_config()` / `acpl_config_*()` (those are
    I-frame I/O for ASPX modes only), no companding, no LFE
    `mono_data(b_lfe=1)` (5.0 → `b_has_lfe = false`).
  - `Ac4ImsEncoder::with_5_0()` — channel-mode prefix `0b1101` (4 b — Table
    85 channel_mode 3) for the 5.0 surround layout (`L, R, C, Ls, Rs`)
    without LFE.
  - `Ac4ImsEncoder::encode_frame_pcm_5_0(&[&[f32]; 5])` +
    `encode_frame_pcm_5_0_with_max_sfb()` — accept paired L/R/C/Ls/Rs float
    PCM frames at the encoder's configured frame_len (1920 samples for the
    default 48 kHz / 24 fps), run the round-50 forward pipeline (KBD-
    windowed MDCT + per-band scalefactor + DP-optimal sectioning +
    HCB1..11 codebook selection + SNF emission) independently per channel,
    then emit a `bitstream_version = 2` IMS TOC (channel_mode prefix
    `'1101'`) followed by the 5.0 SIMPLE/Cfg3Five body. The encoder uses
    one [`encoder_mdct::EncoderMdctState`] per channel (new field
    `mdct_states_multi: Vec<EncoderMdctState>`) so 50% TDAC overlap
    continuity is preserved per channel.
  - The decoder's existing `dispatch_5x_cfg3_simple_aspx` path (round 39)
    consumes the body, IMDCTs each per-channel spectrum into output slots
    0..4 (L/R/C/Ls/Rs per Table 180 row `coding_config == 3`), and emits
    5-channel interleaved S16 PCM at the declared sample rate.
  - Round-trip SNR target met: ≥ 20 dB spectral SNR per channel on the
    independent-tone fixture (220 / 440 / 660 / 880 / 1100 Hz on L/R/C/Ls/Rs).
    Measured: L=24.5, R=24.8, C=25.0, Ls=23.4, Rs=27.4 dB — comfortably above
    the 20 dB floor and in the same band as the round-51 stereo Path A SNR
    (24.8 dB on 440 Hz). The decoder's 5-channel S16 interleaved layout
    (1920 × 5 × 2 = 19,200 bytes) round-trips cleanly through
    `Ac4Decoder::receive_frame`.
  - Seven new tests in `tests/round74_5_0_multichannel.rs`
    (`round74_5_0_encoder_produces_5channel_layout_pcm`,
    `round74_5_0_encoder_bumps_sequence_counter`,
    `round74_5_0_encoder_toc_declares_5_channels`,
    `round74_5_0_independent_tones_per_channel_round_trip_with_distinct_audio`,
    `round74_5_0_per_channel_spectral_snr_exceeds_20db`,
    `round74_5_0_substream_parses_via_walk_ac4_substream`,
    `round74_5_0_silence_round_trips_to_silence`).
  - 5.1 (channel_mode `0b1110` — adds LFE), 7.0/7.1 (channel_mode
    `0b11110000`/`0b11110001` — adds front-extension / back-surround pair),
    and the ASPX/A-CPL multichannel modes (`5_X_codec_mode in
    {ASPX, ASPX_ACPL_1, ASPX_ACPL_2, ASPX_ACPL_3}`) are deferred. 5.0
    SIMPLE Cfg3Five is the spec-mandated minimum for 5-channel AC-4
    streams and unblocks the encoder's path to LFE / immersive layouts.

- **Round 52 — Joint M/S CPE (Path B: `b_enable_mdct_stereo_proc == 1`) encoder**
  per ETSI TS 103 190-1 §5.3 (channel_count > 1) + §4.2.6.3 Table 22
  (`stereo_data()` with `b_enable_mdct_stereo_proc == 1`) + §7.5
  (Pseudocode 77 joint stereo, inverse M/S = `L = M+S, R = M-S`):
  - `encoder_asf::average_per_sfb_correlation(transform_length, max_sfb,
    coeffs_l, coeffs_r)` — energy-weighted per-SFB Pearson correlation
    between the L and R MDCT spectra. Bands are weighted by their
    geometric-mean energy `sqrt(s_ll * s_rr)` so spectrally-disjoint
    tones don't contaminate the metric. Returns a value in `[-1.0, 1.0]`.
  - `encoder_asf::build_stereo_simple_asf_joint_body_from_pcm_spectra(
    transform_length, max_sfb, coeffs_l, coeffs_r, pad_target_bytes)`
    — emits the full joint stereo audio_data body:
    `audio_size_value (15 b)` + `b_more_bits (1 b)` + byte_align +
    `stereo_codec_mode = SIMPLE (2 b)` + `b_enable_mdct_stereo_proc = 1
    (1 b)` + `b_long_frame (1 b)` + shared `max_sfb (n_msfb_bits b)`
    + shared `asf_section_data()` + per-channel `asf_spectral_data()`
    for M and S residuals + shared `asf_scalefac_data()` + per-active-
    sfb `ms_used[sfb]` (1 b each) + shared `asf_snf_data()`.
  - Per-SFB M/S vs L/R decision: bit-cost compare between (M, S) and
    (L, R) at the baseline q_target=12 picks the cheaper representation
    band-by-band. The `ms_used[sfb]` flag on each active band tells the
    decoder whether to apply the inverse-M/S transform.
  - Frame-level "matched-channels" q_target bump: when the frame's
    total S energy is below 15% of M (e.g. for matched / near-matched
    stereo content), the M-channel anchor scalefactor is re-picked at a
    bumped peak-quant target (up to 16, smoothly tapering down to 12
    at the 15% threshold), spending the bits saved on the silent /
    near-silent S residual on a finer M quantisation. Per-band bumps
    were tried first but interact destructively with the shared joint-
    section cost table — when some bands are bumped and others aren't
    the joint scalefactor sequence misaligns the decoder's
    dequantisation. The frame-level gate keeps the bump self-
    consistent across the section partition.
  - `Ac4ImsEncoder::encode_frame_pcm_stereo` now dispatches between
    Path A (round 51, split-MDCT) and Path B (this round, joint M/S)
    based on the energy-weighted per-SFB correlation rising above
    `Ac4ImsEncoder::STEREO_JOINT_MS_CORRELATION_THRESHOLD` (0.7). New
    `encode_frame_pcm_stereo_split_with_max_sfb` /
    `encode_frame_pcm_stereo_joint_with_max_sfb` force a specific path
    regardless of correlation — used by tests / fixtures that need a
    deterministic on-wire layout.
  - Round-trip SNR targets met:
    * Matched 440 Hz L=R: ≥ 34.5 dB spectral SNR (vs round-51's
      24.8 dB on this fixture) — the q_target bump tightens the M
      quantisation step by ~half, S quantises to all-zero, and the
      decoder reconstructs L = M+0, R = M-0 with the bumped precision.
    * Independent 440 Hz L + 660 Hz R: routed via Path A (correlation
      0.0 by design), preserving round-51's ≥ 24.8 dB SNR floor.
    * Half-correlated stereo (amplitude-imbalanced 440 Hz at
      0.30 / 0.36): routed via Path B, frame-level S/M ratio ≈ 0.003
      triggers the q_target bump, output ≥ 26.4 dB / 28.0 dB per
      channel — between the pure-matched and fully-independent
      regimes.
  - Six new tests in `tests/round52_joint_ms_stereo.rs`
    (`round52_matched_stereo_joint_ms_snr_exceeds_28db`,
    `round52_independent_stereo_routes_via_split_path_a`,
    `round52_half_correlated_stereo_joint_ms_snr_exceeds_26db`,
    `round52_joint_ms_full_pcm_roundtrip_through_ac4decoder`) plus two
    in-module correlation sanity tests
    (`round52_correlation_identical_channels_is_one`,
    `round52_correlation_independent_tones_below_threshold`).

- **Round 51 — Stereo SIMPLE/ASF split-MDCT (Path A: 2× SCE) encoder**
  per ETSI TS 103 190-1 §5.3 (channel_count > 1) + §4.2.6.3 Table 22
  (`stereo_data()` with `b_enable_mdct_stereo_proc == 0`):
  - `Ac4ImsEncoder::encode_frame_pcm_stereo(frame_l, frame_r)` and
    `encode_frame_pcm_stereo_with_max_sfb()` — accept paired L+R float
    PCM frames at the encoder's configured frame_len (1920 samples for
    the default 48 kHz / 24 fps), run the round-50 forward pipeline
    (KBD-windowed MDCT + per-band scalefactor + HCB1..11 codebook
    selection + DP-optimal section boundaries + SNF emission)
    independently per channel, then emit a `bitstream_version = 2` IMS
    TOC (channel_mode prefix `'10'`) followed by the split-MDCT stereo
    CPE body. The encoder uses separate `EncoderMdctState` per channel
    (new field `mdct_state_r: Option<EncoderMdctState>`) so 50% TDAC
    overlap continuity is preserved per channel.
  - `encoder_asf::build_stereo_simple_asf_split_body_from_pcm_spectra(
    transform_length, max_sfb, coeffs_l, coeffs_r, pad_target_bytes)`
    — emits the full stereo audio_data body: `audio_size_value (15 b)`
    + `b_more_bits (1 b)` + byte_align + `stereo_codec_mode = SIMPLE
    (2 b)` + `b_enable_mdct_stereo_proc = 0 (1 b)` + per-channel
    `spec_frontend (1 b) + b_long_frame (1 b) + max_sfb (n_msfb_bits
    b)` headers + per-channel `sf_data(ASF)` payloads (sections +
    spectral + scalefac + snf).
  - Round-trip SNR target met: ≥24.8 dB spectral SNR on both L and R
    for the 440 Hz matched-tone fixture and the 440 Hz L + 660 Hz R
    independent-tone fixture (PCM amplitude 0.3, 3 frames to reach
    steady state, comparison done in MDCT-spectrum domain to isolate
    the encoder quantisation contribution from IMDCT/KBD reconstruction
    phase shift). PCM peak ~10 400 i16 (= 0.317 amplitude, matching
    input).
  - Three new SNR / non-silence tests
    (`encode_frame_pcm_stereo_440hz_both_channels_snr_exceeds_20db`,
    `encode_frame_pcm_stereo_440l_660r_independent_channels_snr_exceeds_20db`,
    `encode_frame_pcm_stereo_440hz_steady_state_nonsilent_both_channels`)
    plus three structural smoke tests
    (`encode_frame_pcm_stereo_bumps_sequence_counter`,
    `encode_frame_pcm_stereo_produces_stereo_layout_pcm`,
    `encode_frame_pcm_stereo_substream_parses`).
  - Joint M/S coding (Path B — `b_enable_mdct_stereo_proc == 1`) and
    multichannel SAP/M-S decisioning are deferred. SIMPLE 2× SCE is
    the spec-mandated minimum for stereo AC-4 streams and unblocks the
    encoder's path to multichannel.

## [0.0.5](https://github.com/OxideAV/oxideav-ac4/compare/v0.0.4...v0.0.5) - 2026-05-09

### Other

- ac4 round 50: section-boundary DP optimiser + SNF emission
- ac4 round 49: HCB1..11 codebook-selection optimiser + parameterised max_sfb
- ac4 round 48: forward MDCT + ASF entropy encoder for arbitrary PCM
- ac4 round 47: IMS bitstream_version=2 TOC parser + mono SIMPLE/ASF tone encoder
- round 46: AC-4 IMS encoder scaffold + ACPL_1 surround Ls/Rs spec audit
- ac4 round 45: stereo-CPE M=2 synced companding for ACPL_3 surround pair
- ac4 round 44: companding sync_flag=1 cross-channel exact synchronisation
- ac4 round 43: companding sync_flag=1/avg branches + ACPL_1 sb0 hookup
- ac4 round 42: cfg0/cfg1/cfg3 trailer-aware ASPX + §5.7.5 companding
- ac4 round 41: 5_X cfg2 ASPX trailers + Table 181 SAP for ACPL_1
- ac4 round 40: SAP a/b/c/d (Pseudocode 59) + Table 183 + Ls/Rs walker
- ac4 round 39: 5_X cfg0/cfg1/cfg3 dispatch + 7_X additional-channel pair
- ac4 round 38: LFE body decoder + cfg2_back_mono end-to-end + ACPL_3 centre
- ac4 round 37: wire 7_X ACPL_1/_2 dispatch + cfg0 centre end-to-end decode
- ac4 round 36: wire 5_X ASPX_ACPL_1 / ACPL_2 Pseudocode 117 into decoder
- Round 35: extend ETSI validation suite to float reference tables
- drop dead `linkme` dep
- round 35 — EMDF payloads_substream parser + DRC PCM gain application
- cargo fmt pass after round 34
- update round 34 status (SNF + FIXVAR/VARFIX/VARVAR atsg + ACPL_3)
- ac4 round 34: FIXVAR/VARFIX/VARVAR atsg + SNF inject + 5_X ACPL_3 wiring
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-ac4/pull/502))

### Added

- **Round 50 — Section-boundary DP optimiser + Spectral Noise Fill
  (SNF) emission** (TS 103 190-1 §5.7.4 section_data + §5.7.6 SNF +
  Pseudocodes 100 + 105 + Table 39/42 + Table SCFB):
  - `encoder_asf::dp_optimise_sections(cost_band_cb, max_sections)` —
    dynamic-program over scale-factor bands that finds the globally
    cheapest sequence of `(start, end, cb)` sections, paying the
    per-section header cost (`4 + 3 * (floor((L-1)/7)+1)` bits per
    Table 39) against each band's per-codebook bit cost. Section
    count capped at 16 per ETSI Table SCFB. Supersedes the round-49
    greedy run-length codebook-merge optimiser.
  - `encoder_asf::section_overhead_bits(len)` — closed-form per-section
    overhead in bits matching the spec's `n_sect_bits=3 / esc=7`
    long-frame layout: 7 bits for `L ∈ 1..=7`, 10 bits for `L ∈ 8..=14`,
    13 bits for `L ∈ 15..=21`, etc.
  - `encoder_asf::build_band_codebook_cost_table(natural_q_per_band)`
    — precomputed per-band per-codebook bit cost (rows of length 12;
    cb=0 cost 0 only for all-zero bands; HCB1..11 costs from
    `bit_cost_for_band`; `u32::MAX` for codebooks that can't represent
    the band's natural quant magnitudes). Drives the DP via O(1)
    prefix sums where every band is feasible for the codebook.
  - `encoder_asf::build_sections_from_dp(sections, max_sfb)` — lowers
    the DP-derived `(start, end, cb)` triples into an `AsfSections`
    suitable for the existing `write_section_data` /
    `write_spectral_data_sections` emitters.
  - `encoder_asf::compute_snf_dpcm_for_zero_quant_bands(coeffs,
    sfb_offset, max_sfb, sfb_cb, max_quant_idx)` — for each band that
    quantises to all-zero (cb == 0 || mqi == 0), estimates the band's
    RMS energy from the original MDCT coefficients and picks the
    HCB_SNF index whose `snf_gain = 2^((idx*1.5 - 84)/4)` best matches.
    Returns `Some(per_band_idx)` when at least one zero-quant band
    has measurable energy; `None` for fully-silent input.
  - `encoder_asf::write_snf_data(bw, snf, sfb_cb, max_quant_idx, max_sfb)`
    — emits `b_snf_data_exists` (1 bit) plus per-zero-quant-band
    HCB_SNF Huffman codewords per Table 42 / Pseudocode 105. Round-trips
    cleanly through the existing `parse_asf_snf_data` decoder path
    (round 36+).
  - `encoder_asf::measure_greedy_vs_dp_bits(transform_length, max_sfb,
    coeffs)` — diagnostic helper returning `(greedy_bits, dp_bits)`
    for any input spectrum so callers can quantify the section-boundary
    optimiser's contribution to total frame size.
  - `Ac4ImsEncoder::encode_frame_pcm_with_max_sfb` now drives the DP
    optimiser + SNF emission internally; existing call sites get the
    new path automatically with no API change. White-noise spectral
    SNR holds at 27.5 dB (round-49 baseline) with section overhead
    reduced and SNF emission turned on for high-frequency zero-quant
    content.

- **Round 49 — Codebook-selection optimiser (HCB1..11) + parameterised
  `max_sfb`** (TS 103 190-1 §5.7 + Pseudocodes 17 + 19 + 20 + Annex A.0
  huff_codes + Table SCFB):
  - `encoder_asf::pick_best_codebook_for_band` — per-band codebook
    optimiser sweeping HCB1..11 and choosing the lowest-bit-cost
    codebook whose `q_max` covers the band's natural quantised range.
    Anchor scalefactor targets peak quant ≈ 12 (HCB9/10's q_max → 3×
    more quantisation levels per band than the round-48 HCB5-only
    baseline). HCB11 always qualifies via its Pseudocode 20 escape so
    very-high-energy bands don't clip.
  - `encoder_asf::bit_cost_for_band` — precise bit-counter modelling
    HCB1..11 codeword length + sign bits for unsigned codebooks +
    per-Pseudocode-20 `n_ext` extension bits for HCB11 escapes.
    Mirrors the encoder emitter's exact bit shape (inline magnitude
    saturates at 16 for HCB11, sign bit per non-zero post-saturation
    line for unsigned codebooks).
  - `encoder_asf::build_sections_from_per_band_cb` — collapses runs
    of consecutive same-codebook bands into a single `AsfSections`
    entry so the emitted `asf_section_data()` honours the spec's
    grouping pseudocode without spurious cb-switch overhead.
  - `encoder_asf::write_section_data` + `write_spectral_data_sections`
    — multi-section asf_section_data + asf_spectral_data emitters.
    Per-section emission walks `sect_start..sect_end` bins with the
    section's codebook, handles `cb == 0` silent bands, and writes
    Pseudocode 20 escape bits for HCB11 outliers.
  - `Ac4ImsEncoder::encode_frame_pcm_with_max_sfb(frame, max_sfb)` —
    new public entry point parameterising `max_sfb` (round-48 default
    was hard-coded to 40 → ~6.4 kHz at tl=1920 / 48 kHz). Pad target
    scales with max_sfb (2KB / 4KB / 8KB tiers). `encode_frame_pcm`
    keeps the round-48 default of `max_sfb=40` for backwards
    compatibility.
  - White-noise round-trip SNR jumps from **13.6 dB** (round-48 HCB5-only
    baseline) to **27.5 dB** (round-49 HCB1..11 optimiser, q_target=12,
    max_sfb=50) — measured spectrally against the encoder's own MDCT
    coefficients pre/post quantisation. 1 kHz tone reconstruction at
    `max_sfb=55` preserves >100% of input energy (vs ~40% at the
    round-48 max_sfb=40 default).

- **Round 48 — Forward MDCT analysis + ASF entropy encoder for arbitrary
  PCM input** (TS 103 190-1 §5.5 MDCT + §5.7 SIMPLE + §5.8 ASF +
  Pseudocodes 17-19 + Annex A.0 huff_codes):
  - `encoder_mdct::mdct_naive` + `EncoderMdctState` — forward MDCT
    direction complementing the decoder's `mdct::imdct`. Naive
    O(N²) direct-summation cosine basis (correctness-first; encoder
    isn't on a hot path). Sign convention + scaling matched against
    the decoder's IMDCT through a Princen-Bradley TDAC round-trip
    test (constant-signal recovery in steady-state middle frame
    within 1% error). `EncoderMdctState` carries the previous-frame
    `N` PCM samples for cross-frame 50% TDAC overlap.
  - `encoder_asf::quantise_coeff` + `pick_scalefactor_for_band` +
    `encode_pair` + `write_sect_len_incr` + `write_scalefac_data` +
    `build_mono_simple_asf_body_from_pcm_spectrum` — closed-form
    forward ASF entropy encoder for the long-frame, single-window-
    group, mono SIMPLE channel case. Per-band scalefactor selection
    via the closed-form solve `sf_min = ceil(100 + 4*log2(max_abs/q_max^(4/3)))`
    keeping every quantised line within HCB5's ±4 magnitude bound
    after `q = round(sign(c)*|c/sf_gain|^(3/4))`. Single-section
    HCB5 emission across `0..max_sfb`; reference scalefactor
    + DPCM-coded per-band deltas via HCB_SCALEFAC; `b_snf_data_exists = 0`.
  - `Ac4ImsEncoder::encode_frame_pcm(input: &[f32])` — public entry
    point taking arbitrary float PCM input (range `[-1.0, 1.0]`)
    and emitting a structurally-valid IMS v2 frame end-to-end:
    forward MDCT analysis → per-band scalefactor + quantisation →
    HCB5 entropy coding → wrap in v2 IMS TOC + audio_size header.
    Lazily initialises a per-encoder `EncoderMdctState` on first
    call; bumps `sequence_counter` modulo 1024. Mono / 48 kHz / 24 fps
    by default (frame_len = 1920 samples, max_sfb = 40 covering
    bins 0..600 ≈ 7.5 kHz).
  - 13 new unit tests across the three modules cover: forward MDCT
    zero-in-zero-out + linearity + Princen-Bradley constant-signal
    + sine-wave SNR > 40 dB; quantise/dequantise round-trip;
    pick_scalefactor q_max bound; encode_pair signed/unsigned
    round-trip via `huff_decode` + `split_qspec`;
    `build_mono_simple_asf_body_from_pcm_spectrum` end-to-end parse
    via the existing ASF decoder; full encode → decode round-trip
    for 1 kHz tone (peak amplitude > 1000 i16), multi-tone
    250+500+1000 Hz (SNR > 10 dB on steady-state frame), and
    silence (peak < 50 i16); `encode_frame_pcm` bumps
    `sequence_counter` per call.
  - Codebook-selection optimiser (try HCB1..11 per section, pick
    min-bits), section-boundary optimiser (split bands by
    codebook), spectral noise fill, and stereo / multichannel
    forward analysis remain deferred for round 49+.

- **Round 46 — AC-4 IMS encoder scaffold + ACPL_1 surround Ls/Rs
  ASPX-extension spec audit** (TS 103 190-2 §6.2.1.1 / §6.3.2.5,
  TS 103 190-1 §4.2.6.6 Table 25):
  - `encoder_ims::Ac4ImsEncoder` — Auditor-mode AC-4 IMS encoder
    skeleton. Emits a structurally-valid `raw_ac4_frame()` payload
    with the IMS-flavoured `ac4_toc()` (`bitstream_version = 2` +
    `ac4_presentation_v1_info()` + `ac4_substream_group_info()`) per
    TS 103 190-2 §6.2.1.1. Public API: `Ac4ImsEncoder::new()` (defaults
    to mono 48 kHz 24 fps single-presentation single-substream-group
    iframe), `with_v0()` / `with_stereo()` / `with_5_1()` builders,
    `encode_frame(body_padding_bytes)` (bumps the 10-bit
    sequence_counter modulo 1024), and `encode_frame_v0(...)` that
    forces the TS 103 190-1 v0 layout for round-trip-with-decoder
    tests. Audio body is zero-byte placeholder bits — full encoder
    pipeline (MDCT analysis, scalefactor selection, entropy coding,
    A-SPX envelope coding, A-CPL parameter extraction) deferred to
    future rounds. Eight new unit tests cover sequence_counter wrap,
    `parse_ac4_toc` round-trip for mono / stereo / 5.1, full
    `Ac4Decoder` round-trip emitting silent audio, and the leading
    `bitstream_version` bit-pattern invariant for both v0 (`0b00`)
    and v2 (`0b10`) frames.
  - `decoder.rs::dispatch_acpl_5x_pair`-driving block: spec-confirms
    NOT-ASPX finding for the ACPL_1 surround Ls/Rs carriers per ETSI
    TS 103 190-1 §4.2.6.6 Table 25 row `case ASPX_ACPL_1:`. The
    trailer order is `aspx_data_2ch()` (L/R primary pair) +
    `aspx_data_1ch()` (centre mono) + two `acpl_data_1ch()` parameter
    sets — there is NO third ASPX trailer for the surround pair. The
    Ls/Rs sSMP,3 / sSMP,4 carriers are joint-MDCT residuals that feed
    Pseudocode 117 raw, with the post-Pseudocode-117 surround-output
    bandwidth shape coming from the L/R-carrier ACPL synthesis
    (alpha/beta/decorrelator), not from independent surround-pair
    extension. Same finding for the M=2 surround-pair synced
    companding cohort: with no surround carriers, no companding to
    sync. The existing raw-PCM dispatch path is correct per spec —
    no behavioural change in this round, only a documentation note
    closing out the round-45 follow-up flagged in `dispatch_acpl_5x_pair`.

- **Round 44 — companding `sync_flag == 1` cross-channel synchronisation
  (Pseudocode 121's exact `g_synch(ts) = (∏ g_ch)^(1/M)`)** (TS 103
  190-1 §5.7.5.2 + Pseudocode 121):
  - `aspx::compute_companding_levels(q, sb0, sb1)` exposes the
    per-slot energy `L_ch(ts) = 0.9105 * mean E_ch(sb,ts)` without
    applying gain — the building block the synced path collects
    across channels.
  - `aspx::levels_to_scales_per_slot(levels)` and
    `aspx::levels_to_scale_averaged(levels)` produce the per-slot
    array (PerSlot / SyncPerSlot) and single constant (Averaged /
    SyncAveraged) scales `g(ts) * G` from a level vector.
  - `aspx::apply_companding_scales_on_qmf(q, sb0, sb1, scales)`
    applies pre-computed scales — used by both the single-channel
    legacy path and the synced multi-channel path.
  - `aspx::apply_synchronised_companding_across_channels(channels, mode)`
    implements Pseudocode 121's `sync_flag == 1` branch directly:
    collects every channel's `L_ch(ts)`, computes `g_synch(ts)` as
    the geometric mean across `M` channels (`SyncPerSlot`) or
    `g_avg,synch` from per-channel averaged levels (`SyncAveraged`),
    then applies the synced gain `g_synch(ts) * G` UNIFORMLY to
    every contributing channel. Geometric mean is computed via
    log-sum / exp for numerical stability across many channels.
  - `Ac4Decoder::aspx_extend_to_qmf(...)` is the new phase-1 split:
    runs the QMF analysis + HF generation + envelope adjustment +
    noise / tone injection, returns `(qmf_matrix, sbx, sbz)` BEFORE
    companding + synthesis. Returns `None` when the QMF
    preconditions trip (length / table / patch derivation).
  - `Ac4Decoder::qmf_synthesise_pcm(q, out_len)` is the phase-2
    split: runs the inverse QMF synthesis and returns PCM. Caller
    is responsible for having applied any §5.7.5 gain.
  - `Ac4Decoder::extend_5x_channels_with_sync_companding(...)` is
    the integration glue: drives every entry through phase-1,
    collects QMF matrices, calls the synced apply, drives every
    entry through phase-2. Channels whose phase-1 returned `None`
    fall through to the unmodified PCM (matching the per-channel
    `aspx_extend_pcm` contract).
  - `Ac4Decoder::extend_5x_entries(...)` is the shared front-end
    used by `dispatch_5x_cfg{0,1,2,3}_simple_aspx`: routes through
    the synced cross-channel path when `companding.sync_flag == 1`,
    otherwise through the per-channel path.
  - `Ac4Decoder::five_x_synced_mode(cc)` resolves the synced mode
    once for the whole 5_X frame (Pseudocode 121 broadcasts
    `compand_on[0]` to every channel under `sync_flag == 1`).
  - All four `dispatch_5x_cfg{0,1,2,3}_simple_aspx` routes
    refactored to build a per-channel entries list then route
    through `extend_5x_entries`. Cfg2 now also folds the centre
    channel into the synced cohort when `back_mono` is present, so
    `M == 5` for the geometric mean across all five 5_X channels
    (was `M == 4` in the round-43 approximation).
  - The round-43 single-channel approximation (`SyncPerSlot` /
    `SyncAveraged` collapsing to per-channel `g_ch` / `g_avg,ch`)
    is RETIRED — every 5_X SIMPLE/ASPX dispatcher now applies the
    exact synchronised gain across all contributing channels.

- **Round 43 — companding `b_compand_avg` + `sync_flag == 1` branches +
  ASPX_ACPL_1 sb0 = `acpl_qmf_band` hookup** (TS 103 190-1 §5.7.5.2 +
  Pseudocode 121):
  - `aspx::CompandingMode` enum captures the (`sync_flag`,
    `b_compand_on[ch]`, `b_compand_avg`) product per Pseudocode 121:
    `Off`, `PerSlot`, `Averaged`, `SyncPerSlot`, `SyncAveraged`.
  - `aspx::CompandingMode::from_control(cc, slot)` resolves the
    active branch for a single channel from a parsed
    `CompandingControl`, honouring `sync_flag == true` (slot 0
    broadcasts) and the b_compand_avg presence-rule.
  - `aspx::apply_companding_on_qmf_with_mode(q, sb0, sb1, mode)`
    extends the round-42 single-channel path to all four active
    sub-branches: `PerSlot` / `SyncPerSlot` apply `g_ch(ts) * G`
    per timeslot; `Averaged` / `SyncAveraged` average `L_ch(ts)`
    over the full A-SPX interval, derive a single constant
    `g_avg,ch * G` and broadcast it across all timeslots. (`Off`
    is a strict no-op.) `apply_companding_on_qmf` retained as a
    `PerSlot` thin wrapper for backward compatibility.
  - `Ac4Decoder::aspx_extend_pcm` signature now takes
    `(compand_mode: CompandingMode, compand_sb0_override: Option<u32>)`
    instead of a raw `compand_on: bool`. The override implements
    §5.7.5.2's sb0 selection rule: for the `ASPX_ACPL_1` codec mode
    sb0 = `acpl_qmf_band` (from `acpl_config_1ch_partial.qmf_band`);
    otherwise sb0 falls back to `tables.sbx` (= `aspx_xover_band`).
  - Stereo CPE path (`receive_frame`) lifts the override from
    `tools.acpl_config_1ch_partial.qmf_band` whenever the active
    stereo or 5_X mode is `AspxAcpl1`. Cfg0/Cfg1/Cfg2/Cfg3 SIMPLE/ASPX
    dispatchers pass `None` (those paths never run on ACPL_1).
  - `Ac4Decoder::five_x_compand_mode_for_slot` resolves a
    `CompandingMode` per output slot for the 5_X dispatchers; the
    legacy `five_x_compand_on_for_slot` is retained as a thin
    `mode != Off` wrapper for round-42 unit-test compatibility.
  - sync_flag == 1 cross-channel synchronisation: the per-channel
    pipeline computes `g_ch(ts)` per channel; for `M = 1` (the
    dominant case in our pipeline) the geometric mean across
    channels reduces to the local gain (exact). For `M > 1` channels
    processed independently the synchronisation is approximated —
    documented as a known limitation in
    `apply_companding_on_qmf_with_mode`.
  - 5_X ACPL_3 path companding wiring: already in place from round
    42 via the stereo CPE path's `compand_mode_pri` /
    `compand_mode_sec` (the ACPL_3 walker populates `tools.companding`
    from `companding_control(2)`; the L/R carriers go through the
    standard stereo CPE primary/secondary path which now lifts
    `CompandingMode` and applies it before the §5.7.7.6.2
    Pseudocode 118 multichannel synthesis).
  - 7 new tests cover: `CompandingMode::from_control` resolves all
    six sync/on/avg product states; `apply_companding_on_qmf_with_mode`
    `Off` strict no-op + `Averaged`/`SyncAveraged` constant-scale
    invariant + sb0-override band shift; `five_x_compand_mode_for_slot`
    branch resolution; `aspx_extend_pcm` with sb0 override + with
    `Averaged` mode diverges from baseline.

- **Round 42 — 5_X SIMPLE/ASPX cfg0/cfg1/cfg3 trailer-aware ASPX
  dispatch + §5.7.5 companding tool** (TS 103 190-1 §4.2.6.6 / §5.7.5):
  - `asf::SubstreamTools` gained `cfg0_aspx_{lr,ls_rs,centre}`,
    `cfg1_aspx_{lr,ls_rs,centre}`, `cfg3_aspx_{lr,ls_rs,centre}` —
    mirrors of the round-41 cfg2 trailer slots. The 5_X SIMPLE/ASPX
    outer walker now stores the captured Table-25 `case ASPX:`
    trailer triplet (`aspx_data_2ch + aspx_data_2ch + aspx_data_1ch`)
    into the slot matching the active `coding_config` instead of
    discarding round-41's cfg0/1/3 captures.
  - `Ac4Decoder::dispatch_5x_cfg{0,1,3}_simple_aspx` now apply the
    A-SPX bandwidth-extension per output channel using the captured
    trailers, with the canonical Table-25 trailer-to-slot mapping
    (1st-2ch -> L/R, 2nd-2ch -> Ls/Rs, 1ch -> C). Independent of
    cfg0's `b_2ch_mode` (ASPX is applied after the channel-element
    decode produces PCM, so the L,Ls / R,Rs inner stereo coding
    doesn't shuffle the trailer assignment).
  - `Ac4Decoder::maybe_extend_5x_slot` — internal helper that picks
    the right `(trailer, primary/secondary)` for an output slot and
    runs `aspx_extend_with_trailer` with the resolved companding-on
    flag for that slot.
  - `aspx::apply_companding_on_qmf(q, sbx, sbz)` — §5.7.5.2
    companding tool decoder side, applied per-channel on the QMF
    matrix in `[sbx, sbz)` for all timeslots. Implements the
    `sync_flag == 0`, `b_compand_on == true` branch (the dominant
    case): per-slot mean-absolute level `L_ch(ts)` per Pseudocode
    equations, then `g_ch(ts) = L^((1-α)/α)` with α = 0.65 and
    `Q_out = g * G * Q_in` with `G = 2^α`. Zero-signal slot uses
    unit gain (avoids `0^negative_exp = inf`). The b_compand_avg
    averaging and the `sync_flag == 1` per-channel-product synched
    gain branches are scaffolded for a later round.
  - `Ac4Decoder::aspx_extend_pcm` extended with a `compand_on: bool`
    parameter — applies `apply_companding_on_qmf` between envelope
    adjustment and inverse QMF synthesis when set. All call sites
    updated: stereo CPE primary/secondary (uses
    `tools.companding[0]` / `[1]`), and every 5_X cfg0/1/2/3 slot
    (uses the `companding_control(5)` slot lookup via
    `five_x_compand_on_for_slot`).
  - `Ac4Decoder::five_x_compand_on_for_slot(cc, slot)` — resolves
    `b_compand_on[slot]` from a `CompandingControl`, honouring
    `sync_flag == true` (slot 0 broadcasts), per-channel, and absent
    flags (returns `false`).
  - 5 new tests cover: per-cfg trailer-aware dispatch produces
    output that diverges from the round-39 low-band-only path;
    companding flag resolver across mono / per-channel / sync
    branches; `aspx_extend_pcm` with companding diverges from
    baseline; and edge-case no-ops for `apply_companding_on_qmf`
    (degenerate band + zero signal).

- **Round 41 — 5_X SIMPLE/ASPX cfg2 ASPX bandwidth-extension trailers +
  Table 181 first-stage SAP matrix for 5_X / 7_X `ASPX_ACPL_1`** (TS
  103 190-1 §4.2.6.6 / §5.3.4.3.2):
  - `aspx::FiveXAspxTrailer` + `aspx::FiveXAspxChannelTrailer` —
    captured per-trailer bitstream state (xover, frequency tables,
    framing, qmode, delta-dir, sig/noise envelopes, hfgen
    add_harmonic / tna_mode) for one `aspx_data_2ch()` or
    `aspx_data_1ch()` payload. Wraps everything `aspx_extend_pcm`
    needs for one or two channels.
  - `asf::capture_aspx_data_2ch_trailer` /
    `asf::capture_aspx_data_1ch_trailer` — wrap `parse_aspx_data_*_body`
    with snapshot/restore over the per-substream ASPX-trailer slots
    so multiple sequential trailers can be parsed without corrupting
    each other's state.
  - `parse_5x_audio_data_outer` SIMPLE/ASPX branch now walks the
    Table 25 row `case ASPX:` trailer triplet (`aspx_data_2ch +
    aspx_data_2ch + aspx_data_1ch`) and stores the captured trailers
    on `tools.cfg2_aspx_lr / cfg2_aspx_ls_rs / cfg2_aspx_centre`.
    Cfg0/Cfg1/Cfg3 still parse the bits (so the bitreader lands at
    the right offset) but don't yet wire to dispatch.
  - `Ac4Decoder::dispatch_5x_cfg2_simple_aspx` extended to apply
    A-SPX bandwidth-extension per-channel: L/R use `aspx_lr` (primary
    / secondary); Ls/Rs use `aspx_ls_rs`; centre uses `aspx_centre`.
    SIMPLE-mode (no trailers) and trailer-parse-miss paths fall
    through to round-38 low-band-only PCM.
  - `asf::apply_sap_table_181(a, b, s3, s4, chparam_pair, max_sfb,
    tl) -> (l, r, ls, rs)` — Table 181 first-stage matrix for
    `5_X_codec_mode == ASPX_ACPL_1`. Mixes (sSMP_A, sSMP_B) with
    (sSMP_3, sSMP_4) per-sfb using the (a, b, c, d) coefficients
    extracted from each `chparam_info()` payload (Pseudocode 59) into
    preliminary (L, R, Ls, Rs) spectra. Bands past `max_sfb_master`
    pull L/R from A/B and zero Ls/Rs.
  - `parse_aspx_acpl_1_2_inner_body` (5_X) +
    `parse_7x_audio_data_outer` (7_X ASPX_ACPL_1 branch) now persist
    the parsed `chparam_info()` pair on
    `tools.acpl_1_residual_chparam` plus `max_sfb_master` on
    `tools.acpl_1_residual_max_sfb_master` — round 40 already
    persisted the residual spectra; round 41 closes the loop with
    the SAP coefficients + bound.
  - `Ac4Decoder::receive_frame` 5_X ACPL_1 dispatch now applies
    Table 181's SAP matrix in the spectral domain when all four
    inputs (sSMP_A, sSMP_B, sSMP_3, sSMP_4 + chparam pair +
    max_sfb_master) are available, IMDCTs the resulting (L, R, Ls,
    Rs) preliminary spectra, and feeds them into Pseudocode 117
    (`run_acpl_5x_pair_pcm`) as the already-mixed PCM inputs the
    spec expects. ACPL_2 mode (no residual pair) falls through to
    round-40 silence placeholders.

- **Round 40 — SAP a/b/c/d coefficient extraction (Pseudocode 59) +
  Table 183 7_X SIMPLE/ASPX final channel mapping + standalone Ls/Rs
  surround mono walker for ACPL_1 Mode 1** (TS 103 190-1 §5.3.2 /
  §5.3.4.4.1 / §5.7.7.6.1):
  - `asf::SapCoeffs` + `asf::extract_sap_abcd(info, max_sfb_per_group)`
    — Pseudocode 59 implementation. Walks `chparam_info.sap_mode` ∈
    {0, 1, 3} and emits per-(g, sfb) `(a, b, c, d)` quartets:
    * `sap_mode == 0` → identity (a=d=1, b=c=0).
    * `sap_mode == 1, ms_used` → M/S inverse (a=b=c=1, d=-1) per-sfb;
      identity where ms_used == false.
    * `sap_mode == 3` → alpha-driven SAP. Pair-major DPCM differential
      decode of `dpcm_alpha_q` → `alpha_q[g][sfb]` (odd sfbs inherit
      the even pair-mate; cross-group `delta_code_time` folds in when
      `g != 0` and `max_sfb_g == max_sfb_prev`). `sap_gain = alpha_q
      * 0.1` drives `(a, b, c, d) = (1 + sap_gain, 1, 1 - sap_gain, -1)`
      for SAP-coded bands; `(1, 0, 0, 1)` for skipped bands.
  - `Ac4Decoder::dispatch_7x_additional_channel_pair` extended:
    accepts `partner_pair_spectra` + `partner_slots` + `chparam`. With
    `b_use_sap_add_ch == true` AND partner spectra of matching
    transform length, applies Table 183's SAP matrix per-sfb in the
    spectral domain — `[out_high; out_low] = [a b; c d] · [partner;
    add_ch]` — then IMDCTs both halves independently. With identity
    SAP the partner slot is left untouched (its independent 5_X-core
    IMDCT renders elsewhere) and only F/G land at slots 5/6 per Table
    182 — round-39 behaviour preserved.
  - `Ac4Decoder::receive_frame` 7_X branch resolves the partner pair
    from the active `five_x_coding_config`: cfg2 picks
    `four_channel_data[2,3]` (Ls/Rs); cfg3 picks
    `five_channel_data[3,4]`; cfg1 picks the trailing
    `two_channel_data[0,1]`. cfg0 has no natural surround partner
    inside the 5_X core — falls through to identity passthrough.
  - **Standalone Ls/Rs surround mono walker for ACPL_1's Mode 1
    surround-driven path**: `parse_aspx_acpl_1_2_inner_body` (5_X)
    and `parse_7x_audio_data_outer` (7_X) now persist the joint-MDCT
    residual pair (sSMP,3 / sSMP,4 per Table 181) on
    `tools.acpl_1_residual_pair: [Option<(u32, Vec<f32>)>; 2]`. The
    decoder's 5_X / 7_X ACPL_1 dispatch IMDCTs them into Ls / Rs PCM
    carriers and feeds them as the `x3` / `x4` inputs of Pseudocode
    117 — replacing the round-37 silence placeholder for slots 3 / 4.
    ASPX_ACPL_2 mode never emits a residual pair (no max_sfb_master
    in its walker), so the detach is `None` and surround stays at
    silence as before.
  - New tests (+8): `extract_sap_abcd_mode_zero_returns_identity`,
    `extract_sap_abcd_mode_one_swaps_per_sfb_on_ms_used`,
    `extract_sap_abcd_mode_three_pair_dpcm_decode`,
    `extract_sap_abcd_mode_three_unused_bands_pass_through`,
    `extract_sap_abcd_mode_three_delta_code_time_cross_group`,
    `sap_coeffs_identity_helper`,
    `dispatch_7x_additional_pair_sap_identity_routes_partner_and_additional`,
    `dispatch_acpl_5x_pair_with_real_ls_rs_carriers_emits_surround_energy`.
    The existing
    `parse_5x_aspx_acpl_1_non_iframe_walks_three_channel_data` grew an
    assertion that `tools.acpl_1_residual_pair[0/1].is_some()` after
    the inner walker runs. 568 tests total (was 560).

- **Round 39 — 5_X SIMPLE/ASPX cfg0/cfg1/cfg3 dispatch helpers +
  7_X SIMPLE/ASPX additional-channel pair render** (TS 103 190-1
  §5.3.4.3.1 / Table 180 columns 0/1/3 + §5.3.4.4.1 / Table 182):
  - `Ac4Decoder::dispatch_5x_cfg0_simple_aspx` — IMDCTs each
    `two_channel_data.scaled_spec_per_channel[0..2]` into PCM slots
    per the 1-bit `b_2ch_mode`: `false` -> tcd_a→[0,1] (L,R) /
    tcd_b→[3,4] (Ls,Rs); `true` -> tcd_a→[0,3] (L,Ls) /
    tcd_b→[1,4] (R,Rs). `cfg0_centre_mono.scaled_spec` (the trailing
    `mono_data(0)`) lands on slot 2 (C).
  - `Ac4Decoder::dispatch_5x_cfg1_simple_aspx` — IMDCTs
    `three_channel_data[0..3]` into slots 0/1/2 (L/R/C) and
    `two_channel_data[0..2]` into slots 3/4 (Ls/Rs).
  - `Ac4Decoder::dispatch_5x_cfg3_simple_aspx` — IMDCTs
    `five_channel_data[0..5]` straight into slots 0..4 (L/R/C/Ls/Rs).
  - `Ac4Decoder::dispatch_7x_additional_channel_pair` — IMDCTs the
    `seven_x_additional_channel_data.scaled_spec_per_channel[0..2]`
    into PCM slots 5 / 6 (the F / G preliminary outputs in Table 182).
    SAP companding (Table 183 a,b,c,d) is the identity for now —
    `b_use_sap_add_ch == false` collapses the matrix; the explicit
    SAP coefficient extraction lands in a future round.
  - `Ac4Decoder::receive_frame` switch: round 38's cfg2-only branch
    now selects across `Cfg0/Cfg1/Cfg2/Cfg3` based on the parsed
    `five_x_coding_config`. Mutually exclusive with the ACPL_3 / pair
    paths via the existing `five_x_simple_aspx_active` gate. The
    7_X SIMPLE/ASPX path additionally runs the additional-channel
    pair dispatch on top of the core 5-channel mapping.
  - New tests (+9): `dispatch_5x_cfg0_populates_l_r_c_ls_rs_default_2ch_mode`,
    `dispatch_5x_cfg0_alternate_2ch_mode_maps_to_l_ls_r_rs`,
    `dispatch_5x_cfg1_populates_l_r_c_ls_rs`,
    `dispatch_5x_cfg3_populates_l_r_c_ls_rs`,
    `dispatch_5x_cfg013_noop_on_length_mismatch`,
    `dispatch_7x_additional_pair_populates_slots_5_and_6`,
    `dispatch_7x_additional_pair_noop_on_length_mismatch`,
    plus integration tests
    `five_x_simple_cfg0_walker_populates_two_two_plus_centre`,
    `five_x_simple_cfg3_walker_populates_five_channel_data`. Total:
    551 -> 560.

- **Round 38 — LFE body decoder + cfg2_back_mono end-to-end decode +
  ACPL_3 centre channel via IMDCT** (TS 103 190-1 §4.2.7.2 / §4.2.6.6
  Cfg2 / §5.7.7.6.2):
  - `parse_mono_data(b_lfe=true)` now walks the trailing `sf_data(ASF)`
    body via the new `decode_asf_long_lfe_body_with_max_sfb_lfe`
    helper. `sf_info_lfe()` (Table 35) forces long-frame / single
    window group, so the LFE body shape is the regular ASF long-frame
    quartet (`asf_section_data + asf_spectral_data + asf_scalefac_data
    + asf_snf_data`) — only the `max_sfb[0]` bit-width changes
    (`n_msfbl_bits` from Table 106 column 4 instead of `n_msfb_bits`).
    The dequantised + scaled spectrum lands on `MonoLfeData::scaled_spec`,
    matching the round-37 non-LFE behaviour.
  - `Ac4Decoder::dispatch_5x_cfg2_simple_aspx` — new helper that runs
    end-to-end IMDCT + overlap-add for the 5_X SIMPLE/ASPX
    `coding_config == 2` channel layout. Walks the parsed
    `four_channel_data.scaled_spec_per_channel[0..4]` into output
    slots 0/1/3/4 (L/R/Ls/Rs per Table 180) and the trailing
    `cfg2_back_mono.scaled_spec` into slot 2 (C). Fires when
    `five_x_mode in {Simple, Aspx}` and `five_x_coding_config ==
    Cfg2FourMono`. Cfg0 / Cfg1 / Cfg3 dispatch helpers remain deferred.
  - `Ac4Decoder::receive_frame` ACPL_3 path: replaces the round-37
    silence-placeholder for the centre carrier with an IMDCT of
    `cfg0_centre_mono.scaled_spec` via `imdct_mono_lfe_data_f32`. The
    Pseudocode 118 multichannel synthesis now emits a non-silent
    centre when the trailing `mono_data(0)` body decoded successfully;
    falls back to silence when the centre body is absent (LFE / SSF /
    Huffman miss / non-Cfg0 frame).
  - New unit + integration tests:
    - `parse_mono_data_lfe_walks_sf_data_body` — replaces the round-37
      "LFE body deferred" test; verifies the LFE body now decodes into
      `scaled_spec` for an all-zero stream.
    - `dispatch_5x_cfg2_populates_l_r_c_ls_rs` — cfg2 dispatch IMDCTs
      every channel into the right output slot with non-zero energy
      from per-channel ramp spectra.
    - `dispatch_5x_cfg2_noop_on_length_mismatch` — cfg2 dispatch
      leaves output slots untouched when the carrier transform length
      differs from the requested sample count.
    - `five_x_simple_cfg2_walker_populates_four_plus_back_mono` —
      integration test threading `parse_5x_audio_data_outer` → cfg2
      tools layout, asserting the four per-channel scaled spectra +
      back-mono body all populate cleanly.
- **Round 37 — 7_X ASPX_ACPL_1 / ASPX_ACPL_2 dispatch + cfg0 centre
  end-to-end decode** (TS 103 190-1 §5.7.7.6.3 Pseudocode 120 +
  §5.7.7.6.1 Pseudocode 117 cfg0 centre-channel wiring):
  - `Ac4Decoder::receive_frame` now dispatches the §5.7.7.6.3
    Pseudocode 120 channel-pair multichannel synthesis when the 7_X
    walker resolved `seven_x_mode` to `AspxAcpl1` or `AspxAcpl2`. The
    dispatch reuses `dispatch_acpl_5x_pair` (the Pseudocode 117
    channel-pair core that Pseudocode 120 wraps for 7.X) — for
    `ASPX_ACPL_{1,2}` the additional 2 channels (z6/z7 in Pseudocode
    120) live outside the A-CPL pair so the 5-channel core (L/R/C/Ls/Rs)
    is identical between 5_X and 7_X. Slots 5..7 stay at silence until
    the additional-channel decode path lands.
  - `MonoLfeData` (`mch::parse_mono_data`) now decodes the trailing
    `sf_data(ASF)` body for non-LFE, ASF-frontend, long-frame /
    single-window-group mono channels — replacing the round-36
    silence-placeholder for the centre channel in the 5_X / 7_X
    Pseudocode 117 / 120 dispatches with a real IMDCT / overlap-add
    centre carrier. The new `MonoLfeData::scaled_spec` field carries
    the dequantised + scaled MDCT spectrum; LFE / SSF-frontend / short
    / grouped / Huffman-error cases still leave this `None` and
    preserve the prior outer-shell-only behaviour.
  - `Ac4Decoder::dispatch_acpl_5x_pair` now accepts optional
    `centre_pcm` / `ls_pcm` / `rs_pcm` carrier overrides — `None`
    falls back to the round-36 silence-placeholder; `Some(...)` threads
    real per-channel PCM through Pseudocode 117's `z4 = x2` centre
    passthrough (and the `x3in = 2*x3` / `x4in = 2*x4` Ls/Rs pre-
    multiplications for ACPL_1 mode). The Cfg0 centre is wired from
    the parsed `cfg0_centre_mono.scaled_spec` via a new
    `imdct_mono_lfe_data_f32` helper.
  - New unit + integration tests:
    - `parse_mono_data_non_lfe_walks_sf_data_body` — non-LFE long-frame
      ASF-frontend mono walks the trailing body into `scaled_spec`.
    - `parse_mono_data_lfe_skips_body_walk` — LFE body decoder remains
      deferred (round-36 behaviour preserved).
    - `parse_mono_data_non_lfe_ssf_frontend_skips_body_walk` —
      SSF-frontend mono skips the ASF body walk.
    - `dispatch_acpl_5x_pair_centre_pcm_passthrough_emits_centre_energy`
      — supplying a real centre PCM produces non-silent ch2 output.
    - `seven_x_pair_dispatch_resolves_same_mode_as_five_x` — the 7_X
      dispatch maps `SevenXCodecMode::AspxAcpl{1,2}` to the same
      `Acpl5xPairMode` selectors as the 5_X path.
    - `imdct_mono_lfe_data_f32_returns_none_when_no_scaled_spec` /
      `imdct_mono_lfe_data_f32_imdcts_when_scaled_spec_present` —
      cover the new IMDCT helper.
    - `seven_x_aspx_acpl_2_walker_to_synthesis_glue` — integration
      test threading `parse_7x_audio_data_outer` →
      `acpl_data_1ch_pair` → `run_acpl_5x_pair_pcm` for a non-iframe
      7_X ACPL_2 substream.

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
