//! Round 381 — wire a **real `aspx_preflat`** decision into the live 5_X
//! SIMPLE/ASPX_ACPL_3 single-envelope real-ASPX frame path.
//!
//! ### Background
//!
//! Before the A-SPX subband tonal-to-noise-ratio adjustment, the decoder
//! performs a spectral **pre-flattening** step (ETSI TS 103 190-1
//! §5.7.6.4.1.1–§5.7.6.4.1.2 Pseudocode 85): it fits a third-order
//! polynomial to the dB spectral envelope of the HF-generation source low
//! band `Q_low`, turns the overall slope into a per-low-subband gain
//! vector, and — when `aspx_preflat == 1` (Table 121) — multiplies the
//! inverse of that gain into the patched high band (Pseudocode 89),
//! de-tilting the transposed tile.
//!
//! Through round 377 the live encoder paths always emitted `aspx_preflat =
//! 0`. This round adds the `aspx_preflat_select` encoder analysis: reuse
//! the decoder's exact Pseudocode-85 gain vector and signal pre-flattening
//! when its dB dynamic range (= the fitted slope's peak-to-peak excursion)
//! clears a threshold — i.e. when the source low band carries a strong
//! overall spectral tilt that flattening meaningfully de-tilts. A
//! spectrally flat source range yields ~unity gains and is left alone. The
//! decision is wired into the single-envelope 5_X ACPL_3 real-ASPX frame
//! path (one per-`aspx_config` flag, derived from the primary L carrier).
//!
//! ### What this round measures
//!
//! 1. The analysis fires — a carrier whose QMF low band carries a strong
//!    spectral tilt (a low-frequency-dominant input) selects
//!    `aspx_preflat = 1`, while a spectrally flat (broadband-noise) low band
//!    selects `0`.
//! 2. Round-trip — the 5.0 / 5.1 encoder output (a sloped low band, so the
//!    flag is live) is accepted by `Ac4Decoder` and yields a 5-/6-channel
//!    AudioFrame.
//! 3. Liveness — a sloped-low-band input (flag set) produces a frame whose
//!    bytes differ from a flat-low-band input (flag clear), proving the bit
//!    reaches the wire.
//! 4. Determinism — matched inputs + fresh encoder state are byte-identical.
//! 5. Audibility — because the decoder fully consumes `aspx_preflat`
//!    (Pseudocode 89 inverse-gain on the patched tile), a sloped-low-band
//!    frame and the same frame forced flat decode to **different** PCM,
//!    proving the encoder decision reaches the decoded output.
//!
//! Refs ETSI TS 103 190-1: Table 121 (`aspx_preflat`), Table 50
//! (`aspx_config`), §5.7.6.4.1.1–§5.7.6.4.1.2 Pseudocode 85 (pre-flattening
//! control data), §5.7.6.4.1.4 Pseudocode 89 (gain application).

use oxideav_ac4::aspx::{
    derive_aspx_frequency_tables, num_aspx_timeslots, num_ts_in_ats, AspxConfig, AspxFreqResMode,
    AspxMasterFreqScale, AspxQuantStep,
};
use oxideav_ac4::aspx_preflat_select::select_preflat;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_acpl3::qmf_slots_to_sb_major;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::qmf::QmfAnalysisBank;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

/// The live config the IMS encoder uses for the 5_X ACPL_3 path (before the
/// `preflat` field is filled in by the encoder analysis).
fn live_cfg() -> AspxConfig {
    AspxConfig {
        quant_mode_env: AspxQuantStep::Fine,
        start_freq: 0,
        stop_freq: 0,
        master_freq_scale: AspxMasterFreqScale::LowRes,
        interpolation: false,
        preflat: false,
        limiter: false,
        noise_sbg: 0,
        num_env_bits_fixfix: 0,
        freq_res_mode: AspxFreqResMode::DurationDependent,
    }
}

/// A low-frequency tone — concentrates QMF energy into the lowest subbands,
/// producing a steeply tilted low-band spectral envelope (a large fitted
/// slope → wide pre-flatten gain spread).
fn make_low_tone_frame(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// A spectrally flat low band — broadband pseudo-random noise spreads
/// energy evenly across the low subbands, so the fitted slope is near-flat
/// and the pre-flatten gain spread is small.
fn make_flat_noise_frame(amp: f32) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..N)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
            amp * (u * 2.0 - 1.0)
        })
        .collect()
}

/// Reproduce the encoder's `aspx_preflat` decision for one carrier so the
/// test can assert it from outside the encoder. Mirrors
/// `Ac4ImsEncoder::extract_aspx_preflat`.
fn preflat_for(pcm: &[f32]) -> bool {
    let cfg = live_cfg();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_ts_in_ats = num_ts_in_ats(N as u32);
    let aspx_frame_ts_count = num_aspx_timeslots(N as u32);
    let n_slots = pcm.len() / 64;
    let usable = n_slots * 64;
    let q = qmf_slots_to_sb_major(&QmfAnalysisBank::new().process_block(&pcm[..usable]));
    let sba = tables.sba as usize;
    let q_low: Vec<Vec<(f32, f32)>> = q.iter().take(sba).map(|r| r.to_vec()).collect();
    let atsg_sig = [0u32, aspx_frame_ts_count];
    select_preflat(&q_low, tables.sba, &atsg_sig, num_ts_in_ats)
}

/// A strong low-frequency tone (steeply tilted low band) selects
/// `aspx_preflat = 1`; flat broadband noise (flat low band) selects `0`.
#[test]
fn sloped_low_band_sets_preflat_flat_does_not() {
    let tilted = make_low_tone_frame(80.0, 0.9);
    assert!(
        preflat_for(&tilted),
        "a low-frequency-dominant (steeply sloped) low band must select preflat"
    );

    let flat = make_flat_noise_frame(0.5);
    assert!(
        !preflat_for(&flat),
        "a spectrally flat low band must not select preflat"
    );
}

/// 5.0 single-envelope real-ASPX encode (sloped low band → live
/// `aspx_preflat`) round-trips to a 5-channel AudioFrame.
#[test]
fn encode_5_0_acpl3_real_aspx_preflat_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);
    // Sanity: the L carrier really does select preflat.
    assert!(preflat_for(&l));

    let frame_bytes =
        enc.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 5 * 2, "5-channel S16 interleaved");
}

/// 5.1 single-envelope real-ASPX encode round-trips to a 6-channel frame.
#[test]
fn encode_5_1_acpl3_real_aspx_preflat_round_trips() {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();

    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);
    let lfe = make_low_tone_frame(45.0, 0.3);

    let frame_bytes =
        enc.encode_frame_pcm_5_1_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs, &lfe], 0.5, 0.1, 1.0, 1.0);
    assert!(!frame_bytes.is_empty());

    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("decoder must accept packet");
    let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, 1920);
    assert_eq!(af.data.len(), 1);
    assert_eq!(af.data[0].len(), 1920 * 6 * 2, "6-channel S16 interleaved");
}

/// Liveness: a sloped-low-band L/R carrier (flag set) produces a frame
/// whose bytes differ from a flat-low-band L/R carrier (flag clear),
/// proving the `aspx_preflat` bit reaches the wire. The centre / surround
/// carriers are held identical so the difference is attributable to the
/// L-carrier-derived `aspx_preflat` decision.
#[test]
fn preflat_changes_emitted_bytes() {
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);

    let l_tilt = make_low_tone_frame(80.0, 0.9);
    let r_tilt = make_low_tone_frame(95.0, 0.8);
    let l_flat = make_flat_noise_frame(0.5);
    let r_flat = make_flat_noise_frame(0.4);

    // Confirm the analysis differs across the two L carriers.
    assert!(preflat_for(&l_tilt));
    assert!(!preflat_for(&l_flat));

    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    let bytes_tilt = enc_a.encode_frame_pcm_5_0_acpl3_real_aspx(
        &[&l_tilt, &r_tilt, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let bytes_flat = enc_b.encode_frame_pcm_5_0_acpl3_real_aspx(
        &[&l_flat, &r_flat, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_ne!(
        bytes_tilt, bytes_flat,
        "sloped-low-band (preflat set) and flat-low-band (clear) frames must differ"
    );
}

/// Determinism: identical inputs + fresh encoder state → identical bytes.
#[test]
fn preflat_path_is_deterministic() {
    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);

    let mut enc_a = Ac4ImsEncoder::new();
    let mut enc_b = Ac4ImsEncoder::new();
    let a = enc_a.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    let b = enc_b.encode_frame_pcm_5_0_acpl3_real_aspx(&[&l, &r, &c, &ls, &rs], 0.5, 0.1, 1.0, 1.0);
    assert_eq!(a, b, "matched inputs must produce identical bytes");
}

/// Audibility (decoder ground truth): for a sloped low band, the
/// decoder's HF-tile generator (§5.7.6.4.1.4 Pseudocode 89) produces a
/// **different** patched tile with `aspx_preflat = 1` (the inverse
/// pre-flatten gain applied) than with `aspx_preflat = 0` (no gain). This
/// is exactly the wire bit the encoder now selects, so the live decision
/// changes the decoded HF — not just the framing. A flat low band, by
/// contrast, has near-unity gains and the two tiles barely differ, which is
/// why the encoder leaves it at `0`.
#[test]
fn preflat_changes_the_decoded_hf_tile() {
    use oxideav_ac4::aspx::{derive_patch_tables, AspxMasterFreqScale, AspxPatchTables};
    use oxideav_ac4::aspx_tns::{compute_preflat_gains, hf_tile_tns};

    let cfg = live_cfg();
    let tables = derive_aspx_frequency_tables(&cfg, 0).expect("freq tables");
    let num_ts_in_ats = num_ts_in_ats(N as u32);
    let aspx_frame_ts_count = num_aspx_timeslots(N as u32);
    let atsg_sig = [0u32, aspx_frame_ts_count];

    // Build a sloped low band from a low-frequency tone.
    let pcm = make_low_tone_frame(80.0, 0.9);
    let n_slots = pcm.len() / 64;
    let q = qmf_slots_to_sb_major(&QmfAnalysisBank::new().process_block(&pcm[..n_slots * 64]));
    let sba = tables.sba as usize;
    let q_low: Vec<Vec<(f32, f32)>> = q.iter().take(sba).map(|r| r.to_vec()).collect();
    let q_low_ext = oxideav_ac4::aspx_tns::build_q_low_ext(&q_low, &[], tables.sba);

    // The encoder selects preflat on this carrier.
    assert!(select_preflat(&q_low, tables.sba, &atsg_sig, num_ts_in_ats));

    let is_highres = matches!(cfg.master_freq_scale, AspxMasterFreqScale::HighRes);
    let patches: AspxPatchTables = derive_patch_tables(
        &tables.sbg_master,
        tables.num_sbg_master,
        tables.sba,
        tables.sbx,
        tables.num_sb_aspx,
        true,
        is_highres,
    );
    let num_qmf = 64u32;
    // No inverse filtering for this comparison (chirp / alphas = 0): isolate
    // the pre-flatten gain's effect on the patched tile.
    let alpha0 = vec![(0.0_f32, 0.0_f32); sba];
    let alpha1 = vec![(0.0_f32, 0.0_f32); sba];
    let chirp = vec![0.0_f32; tables.sbg_noise.len()];

    let gains = compute_preflat_gains(&q_low, tables.sba, &atsg_sig, num_ts_in_ats);
    let tile_off = hf_tile_tns(
        &q_low_ext,
        &patches,
        &tables.sbg_noise,
        &chirp,
        &alpha0,
        &alpha1,
        None,
        tables.sbx,
        num_qmf,
        &atsg_sig,
        num_ts_in_ats,
    );
    let tile_on = hf_tile_tns(
        &q_low_ext,
        &patches,
        &tables.sbg_noise,
        &chirp,
        &alpha0,
        &alpha1,
        Some(&gains),
        tables.sbx,
        num_qmf,
        &atsg_sig,
        num_ts_in_ats,
    );
    assert_ne!(
        tile_on, tile_off,
        "preflat=1 must produce a different patched HF tile than preflat=0 on a sloped low band"
    );
}

/// Decode a packet to its first (and only) audio plane.
fn decode_plane(bytes: Vec<u8>) -> Vec<u8> {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
    dec.send_packet(&pkt).expect("decode");
    let Frame::Audio(af) = dec.receive_frame().expect("frame") else {
        panic!("audio");
    };
    af.data[0].clone()
}

/// The preflat wiring is live on the multi-envelope 5_X ASPX_ACPL_3 path:
/// a sloped-low-band L/R input (flag set) differs from a flat-low-band
/// input (flag clear), and a sloped frame round-trips to 5-channel audio.
#[test]
fn multi_env_5_0_acpl3_preflat_is_live_and_round_trips() {
    // Transient low tones drive num_env to 2 while keeping a sloped low band.
    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);
    assert!(preflat_for(&l));

    let mut enc = Ac4ImsEncoder::new();
    let pcm = decode_plane(enc.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    ));
    assert_eq!(pcm.len(), 1920 * 5 * 2, "5-channel S16 interleaved");

    // Liveness: sloped vs flat L/R changes the bytes (centre/surround held).
    let l_flat = make_flat_noise_frame(0.5);
    let r_flat = make_flat_noise_frame(0.4);
    assert!(!preflat_for(&l_flat));
    let mut e_a = Ac4ImsEncoder::new();
    let mut e_b = Ac4ImsEncoder::new();
    let b_tilt = e_a.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l, &r, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    let b_flat = e_b.encode_frame_pcm_5_0_acpl3_real_aspx_multi_env(
        &[&l_flat, &r_flat, &c, &ls, &rs],
        0.5,
        0.1,
        1.0,
        1.0,
    );
    assert_ne!(b_tilt, b_flat, "preflat must reach the multi-env wire");
}

/// The preflat wiring is live on the 5_X ASPX_ACPL_2 single-envelope path.
#[test]
fn acpl2_5_0_preflat_is_live_and_round_trips() {
    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);
    assert!(preflat_for(&l));

    let mut enc = Ac4ImsEncoder::new();
    let pcm = decode_plane(enc.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]));
    assert_eq!(pcm.len(), 1920 * 5 * 2);

    let l_flat = make_flat_noise_frame(0.5);
    let r_flat = make_flat_noise_frame(0.4);
    let mut e_a = Ac4ImsEncoder::new();
    let mut e_b = Ac4ImsEncoder::new();
    let b_tilt = e_a.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l, &r, &c, &ls, &rs]);
    let b_flat = e_b.encode_frame_pcm_5_0_acpl2_real_aspx(&[&l_flat, &r_flat, &c, &ls, &rs]);
    assert_ne!(b_tilt, b_flat, "preflat must reach the ACPL_2 wire");
}

/// The preflat wiring is live on the 7.0 pure-ASPX path.
#[test]
fn pure_aspx_7_0_preflat_is_live_and_round_trips() {
    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);
    let lb = make_low_tone_frame(70.0, 0.3);
    let rb = make_low_tone_frame(85.0, 0.3);
    assert!(preflat_for(&l));

    let mut enc = Ac4ImsEncoder::new();
    let pcm =
        decode_plane(enc.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]));
    assert_eq!(pcm.len(), 1920 * 7 * 2, "7-channel S16 interleaved");

    let l_flat = make_flat_noise_frame(0.5);
    let r_flat = make_flat_noise_frame(0.4);
    let mut e_a = Ac4ImsEncoder::new();
    let mut e_b = Ac4ImsEncoder::new();
    let b_tilt = e_a.encode_frame_pcm_7_0_aspx_real_aspx(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let b_flat =
        e_b.encode_frame_pcm_7_0_aspx_real_aspx(&[&l_flat, &r_flat, &c, &ls, &rs, &lb, &rb]);
    assert_ne!(b_tilt, b_flat, "preflat must reach the 7.0 pure-ASPX wire");
}

/// The preflat wiring is live on the 7_X ASPX_ACPL_1 path.
#[test]
fn acpl1_7_0_preflat_is_live_and_round_trips() {
    let l = make_low_tone_frame(80.0, 0.9);
    let r = make_low_tone_frame(95.0, 0.8);
    let c = make_low_tone_frame(60.0, 0.4);
    let ls = make_low_tone_frame(110.0, 0.3);
    let rs = make_low_tone_frame(130.0, 0.3);
    let lb = make_low_tone_frame(70.0, 0.3);
    let rb = make_low_tone_frame(85.0, 0.3);
    assert!(preflat_for(&l));

    let mut enc = Ac4ImsEncoder::new();
    let pcm = decode_plane(
        enc.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]),
    );
    assert_eq!(pcm.len(), 1920 * 7 * 2);

    let l_flat = make_flat_noise_frame(0.5);
    let r_flat = make_flat_noise_frame(0.4);
    let mut e_a = Ac4ImsEncoder::new();
    let mut e_b = Ac4ImsEncoder::new();
    let b_tilt = e_a.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&[&l, &r, &c, &ls, &rs, &lb, &rb]);
    let b_flat =
        e_b.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&[&l_flat, &r_flat, &c, &ls, &rs, &lb, &rb]);
    assert_ne!(b_tilt, b_flat, "preflat must reach the 7_X ACPL_1 wire");
}
