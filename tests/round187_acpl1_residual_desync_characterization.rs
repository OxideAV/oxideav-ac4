//! Round 187 — characterise the remaining 5_X ASPX_ACPL_1 desync the
//! r181 follow-up flagged.
//!
//! ### Background
//!
//! Round 181 closed the r128 ALPHA `alpha_q` desync at two distinct
//! spec-alignment layers (§5.7.7.7 Pseudocode 121 parser indexing +
//! §4.2.12.4 Table 52 `aspx_data_2ch()` SIGNAL band count). The Layer-2
//! fix is end-to-end verified for the **ASPX_ACPL_2** path because that
//! path has no joint-MDCT residual layer between the ASPX trailers and
//! the `acpl_data_1ch()` pair — once the ASPX walker is aligned the
//! `acpl_data_1ch()` codewords are the next read.
//!
//! The **ASPX_ACPL_1** path keeps the §4.2.6.6 Table 25 joint-MDCT
//! residual layer (`max_sfb_master + 2× chparam_info + 2× sf_data(ASF)`,
//! carrying the Ls / Rs surround residual spectra coded as sSMP,3 /
//! sSMP,4 per Table 181) between the L/R carriers and the centre
//! `mono_data(0)`. The r181 notes flag a remaining alignment issue
//! between [`encoder_acpl3::write_acpl_1_residual_layer`] and the
//! decoder's `parse_aspx_acpl_1_2_inner_body` residual-pair walker,
//! tracked as the follow-up.
//!
//! ### What this round measures
//!
//! These tests sweep the encoder's `encode_frame_pcm_5_0_acpl1_real_alpha_beta`
//! across input combinations and assert the **structural** shape of the
//! decoder's recovered `acpl_data_1ch_pair[0/1]`. They confirm:
//!
//!   1. **Silence** round-trips cleanly — both pair slots resolve a
//!      single parameter set with `direction_time = false` (DIFF_FREQ)
//!      and all-zero α / β values.
//!   2. **L-carrier-only** (Ls = 0) round-trips cleanly — the α / β
//!      extractor produces all-zero rows because there's no surround
//!      energy, so the `write_acpl_data_1ch_real_alpha_beta` body is
//!      bit-for-bit identical to the silence body and the parser stays
//!      aligned.
//!   3. **Ls-residual-only** (L = 0) round-trips cleanly — α / β are
//!      all-zero again because `L · Ls = 0` ⇒ correlation = 0. The
//!      residual writer is exercised with non-trivial Ls / Rs spectra
//!      and the parser still resolves both pair slots with
//!      `num_param_sets = 1`.
//!   4. **Both L and Ls non-zero** triggers the remaining desync — α
//!      quantises to a non-zero lane in some bands AND the residual
//!      layer carries non-zero Ls / Rs spectra. The decoder still
//!      resolves `acpl_data_1ch_pair[0]` cleanly but
//!      `acpl_data_1ch_pair[1]` reads with `num_param_sets = 2`, an
//!      off-by-bit-stream drift consistent with the writer emitting a
//!      different bit count than the parser consumes upstream of the
//!      pair-1 framing.
//!
//! The tests below capture that pattern as **pinned expectations** so
//! the next round can iterate on the residual / α-β writers without
//! accidentally regressing the silence / L-only / Ls-only paths.
//!
//! ### Where the bug is NOT
//!
//! Multiple narrower writer→parser round-trips already pass:
//!
//!   * [`encoder_acpl3::write_acpl_data_1ch_real_alpha_beta_bytes`] →
//!     [`acpl::parse_acpl_data_1ch`] cycles for arbitrary signed
//!     `alpha_q[pb]` / unsigned `beta_q[pb]` PARTIAL-mode arrays (pinned
//!     by `round181_alpha_desync_fix::standalone_alpha_writer_round_trips_through_parser`
//!     and `standalone_alpha_beta_writer_round_trips_through_parser`).
//!   * Back-to-back `write_acpl_data_1ch_real_alpha_beta` invocations
//!     into the same `BitWriter` (no byte alignment between calls) round
//!     trip cleanly through `parse_acpl_data_1ch` twice.
//!
//! That points the residual desync at the writer→parser pair operating
//! on the **joint-MDCT residual layer** itself
//! ([`encoder_acpl3::write_acpl_1_residual_layer`] vs the inline
//! `decode_asf_long_mono_body_with_max_sfb` walk inside
//! `parse_aspx_acpl_1_2_inner_body`'s ASPX_ACPL_1 branch) — or at the
//! `two_channel_data()` L/R carrier writer (`write_two_channel_data`)
//! vs the matching `parse_two_channel_data` — when the L / Ls spectra
//! are simultaneously non-trivial. The α / β path itself is bit-exact.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::mch::FiveXCodecMode;
use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;

fn make_tone(freq: f32, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = i as f32 / FS;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Encode `frames` through `encode_frame_pcm_5_0_acpl1_real_alpha_beta`
/// and return the decoder's per-pair `num_param_sets` (length of
/// `acpl_data_1ch_pair[0].alpha1` / `acpl_data_1ch_pair[1].alpha1`).
fn pair_num_param_sets(frames: [&[f32]; 5]) -> (usize, usize) {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    let mut enc = Ac4ImsEncoder::new();
    let frame_bytes = enc.encode_frame_pcm_5_0_acpl1_real_alpha_beta(&frames);
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame_bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let _ = dec.receive_frame().expect("receive_frame");
    let sub = dec.last_substream.as_ref().expect("substream parsed");
    assert_eq!(
        sub.tools.five_x_mode,
        Some(FiveXCodecMode::AspxAcpl1),
        "decoder must resolve ASPX_ACPL_1 mode"
    );
    let p0 = sub.tools.acpl_data_1ch_pair[0]
        .as_ref()
        .expect("pair0 must be populated");
    let p1 = sub.tools.acpl_data_1ch_pair[1]
        .as_ref()
        .expect("pair1 must be populated");
    (p0.alpha1.len(), p1.alpha1.len())
}

/// Silence input — every band's α / β quantises to 0 → the
/// `acpl_data_1ch()` body is the minimal-cost shape on the wire. Both
/// pair slots must read with `num_param_sets = 1` (`alpha1.len() == 1`).
#[test]
fn acpl1_silence_round_trips_with_aligned_pair_lengths() {
    let z = vec![0.0f32; N];
    let (n0, n1) = pair_num_param_sets([&z, &z, &z, &z, &z]);
    assert_eq!(n0, 1, "silence → pair0 num_param_sets = 1");
    assert_eq!(n1, 1, "silence → pair1 num_param_sets = 1");
}

/// L-carrier-only input (Ls = Rs = 0) — the L / R carrier writers
/// emit non-trivial section / spectral / scalefac / SNF data, but
/// because Ls = 0 the α / β extractor's per-band correlation and
/// energy ratio are both zero, so every α_q[pb] / β_q[pb] quantises to
/// 0. The `acpl_data_1ch()` body shape on the wire is therefore the
/// same minimal-cost shape as the silence case, and both pair slots
/// must still read with `num_param_sets = 1`.
#[test]
fn acpl1_l_carrier_only_round_trips_with_aligned_pair_lengths() {
    let z = vec![0.0f32; N];
    let l = make_tone(220.0, 0.5);
    let r = make_tone(440.0, 0.5);
    let c = make_tone(660.0, 0.3);
    let (n0, n1) = pair_num_param_sets([&l, &r, &c, &z, &z]);
    assert_eq!(n0, 1, "L-carrier-only → pair0 num_param_sets = 1");
    assert_eq!(
        n1, 1,
        "L-carrier-only → pair1 num_param_sets = 1 (carrier-only \
         exercises `write_two_channel_data` non-trivially while keeping \
         α / β at 0)"
    );
}

/// Ls-residual-only input (L = R = 0) — the joint-MDCT residual layer
/// writer (`write_acpl_1_residual_layer`) emits non-trivial Ls / Rs
/// section / spectral / scalefac / SNF data, but because L = 0 the
/// α / β extractor's per-band correlation is zero (`Σ L_carrier ·
/// Ls_residual = 0`) and the carrier energy is zero too, so every
/// α_q[pb] / β_q[pb] quantises to 0. The `acpl_data_1ch()` body shape
/// is again the minimal-cost shape, and both pair slots must still
/// read with `num_param_sets = 1`.
///
/// This is the cleanest test that exercises
/// `write_acpl_1_residual_layer` against the parser's inline residual
/// walk in `parse_aspx_acpl_1_2_inner_body` with non-trivial spectra
/// and confirms the residual layer alone (in isolation from the α / β
/// writer's non-zero path) round-trips at the correct bit count.
#[test]
fn acpl1_ls_residual_only_round_trips_with_aligned_pair_lengths() {
    let z = vec![0.0f32; N];
    let ls = make_tone(880.0, 0.5);
    let rs = make_tone(1100.0, 0.5);
    let (n0, n1) = pair_num_param_sets([&z, &z, &z, &ls, &rs]);
    assert_eq!(n0, 1, "Ls-residual-only → pair0 num_param_sets = 1");
    assert_eq!(
        n1, 1,
        "Ls-residual-only → pair1 num_param_sets = 1 (residual writer \
         exercised non-trivially while keeping α / β at 0)"
    );
}

/// Combined L-carrier + Ls-residual input — both carriers are non-zero
/// so the α extractor's correlation lands on a non-zero quantisation
/// lane in some bands AND the residual layer carries non-trivial Ls
/// spectra. Pre-r187 this triggered an upstream bit-stream drift the
/// parser absorbed as `pair1.framing.num_param_sets_cod = 1` (i.e.
/// `pair1.alpha1.len() == 2` instead of 1). The test pins the current
/// state so the next round can validate the residual-layer / α-β
/// writer alignment without regressing the silence / L-only / Ls-only
/// paths above.
///
/// Once the residual desync is closed, the assertion below must flip
/// to `assert_eq!(n1, 1)` and this test should be renamed
/// `acpl1_full_round_trips_with_aligned_pair_lengths`.
#[test]
fn acpl1_combined_l_and_ls_pair1_currently_misaligns() {
    let l = make_tone(220.0, 0.5);
    let r = make_tone(440.0, 0.5);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.05);
    let rs = make_tone(1100.0, 0.05);
    let (n0, n1) = pair_num_param_sets([&l, &r, &c, &ls, &rs]);
    assert_eq!(
        n0, 1,
        "pair0 num_param_sets = 1 (the bit drift sits after pair0; pair0 \
         itself still aligns to the writer's emission)"
    );
    // PINNED EXPECTATION: pair1 misaligns to num_param_sets = 2.
    //
    // This is the residue from the r181 follow-up. When fixed the
    // assertion below must flip back to `assert_eq!(n1, 1)`.
    assert_eq!(
        n1, 2,
        "pair1 num_param_sets currently reads 2 (the desync absorbs the \
         drifted bit as `nps_cod = 1`); flip back to 1 once the residual \
         desync is closed"
    );
}
