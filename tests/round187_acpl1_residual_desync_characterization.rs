//! Round 187 (characterisation) + Round 190 (root-cause fix) — close
//! the remaining 5_X ASPX_ACPL_1 desync the r181 follow-up flagged.
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
//! `mono_data(0)`. Round 187 *characterised* the remaining issue by
//! triangulating around it: silence, L-carrier-only and Ls-residual-only
//! all round-tripped cleanly with `pair1.num_param_sets = 1`, but the
//! "both L and Ls non-zero" combination drifted into
//! `pair1.num_param_sets = 2`.
//!
//! ### Round 190 — root cause: `aspx_framing()` FIXFIX prefix
//!
//! The drift is **not** in the residual layer at all. It's in
//! `write_aspx_data_2ch_minimal` / `write_aspx_data_1ch_minimal`, which
//! emitted `aspx_int_class = FIXFIX` as the wrong prefix code: `0b11`
//! (2 bits) instead of `0b0` (1 bit) per ETSI TS 103 190-1 Table 126.
//! The parser's `AspxIntClass::read` correctly treats the first `1` as
//! "not-FixFix", reads another `1` ("not-FixVar"), then walks the
//! following bit as `VarFix` / `VarVar` selection — landing on
//! `VarFix` with `b_iframe = 1`, which then reads `var_bord_left` (2
//! bits), `num_rel_left` (2 bits per Note 1 since `num_aspx_timeslots
//! = 15 > 8`), and `tsg_ptr` (2 bits). Net: parser consumes **9 bits**
//! in the framing where the writer only emitted **3 bits**.
//!
//! In the silence / L-only / Ls-only paths the encoder happened to
//! quantise α / β to zero, so the `acpl_data_1ch` body shape on the wire
//! was the constant minimum-cost shape and the 6-bit upstream drift
//! was masked by a long run of zero codewords on each side — the
//! `num_param_sets_cod` bit positions on both sides ended up sampling
//! `0`. With non-zero α / β the codewords shift in time, the pair-1
//! `num_param_sets_cod` position lands on a `1`, and the symptom in
//! r187 appeared.
//!
//! Fix is two writers, one line each: write `bw.write_bit(false)`
//! (1 bit) for the FIXFIX `aspx_int_class` prefix in both
//! `write_aspx_data_2ch_minimal` and `write_aspx_data_1ch_minimal`,
//! matching the prefix code in `AspxIntClass::read`.
//!
//! ### What this file pins
//!
//! All four combinations round-trip cleanly with both pair slots reading
//! `num_param_sets = 1`. The "both" case (test #4) was the r187 pinned
//! misalignment — r190 flipped its assertion to the post-fix expectation
//! and renamed it.

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
/// spectra. Pre-r190 this triggered an upstream bit-stream drift the
/// parser absorbed as `pair1.framing.num_param_sets_cod = 1` (i.e.
/// `pair1.alpha1.len() == 2` instead of 1). r190 fixed the
/// `aspx_framing()` FIXFIX prefix in the two minimal writers (1 bit
/// `0` per Table 126, not 2 bits `11`); the desync is now closed for
/// every combination tested here.
#[test]
fn acpl1_full_round_trips_with_aligned_pair_lengths() {
    let l = make_tone(220.0, 0.5);
    let r = make_tone(440.0, 0.5);
    let c = make_tone(660.0, 0.3);
    let ls = make_tone(880.0, 0.05);
    let rs = make_tone(1100.0, 0.05);
    let (n0, n1) = pair_num_param_sets([&l, &r, &c, &ls, &rs]);
    assert_eq!(n0, 1, "L+Ls full case → pair0 num_param_sets = 1");
    assert_eq!(
        n1, 1,
        "L+Ls full case → pair1 num_param_sets = 1 (r190 fixed the \
         aspx_framing() FIXFIX prefix; the upstream desync that drove \
         pair1 to nps_cod = 1 is closed)"
    );
}
