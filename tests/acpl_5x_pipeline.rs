//! Round-22 integration: 5_X-walker → A-CPL multichannel synthesis glue.
//!
//! These tests assert the wiring contract between the §4.2.6.6
//! `5_X_channel_element` walker (`crate::mch::parse_5x_audio_data_outer`)
//! and the §5.7.7.6 multichannel synthesis pipelines:
//!
//! * **Pseudocode 117** — `ASPX_ACPL_1` / `ASPX_ACPL_2` 5_X mode:
//!   `parse_5x_audio_data_outer` populates `tools.acpl_config_1ch_*` and
//!   the per-frame `acpl_data_1ch` pair lives in
//!   `tools.acpl_data_1ch_pair[..]`. Downstream callers feed these into
//!   `acpl_synth::run_acpl_5x_pair_pcm` along with the L/R/C carrier
//!   PCM (and optional Ls/Rs for ASPX_ACPL_1) to produce 5-channel PCM.
//!
//! * **Pseudocode 118** — `ASPX_ACPL_3` 5_X mode:
//!   `parse_5x_audio_data_outer` populates `tools.acpl_config_2ch` and
//!   the per-frame parameters live in `tools.acpl_data_2ch`. Downstream
//!   callers feed these into `acpl_synth::run_acpl_5x_mch_pcm` along
//!   with the L/R/C carrier PCM.
//!
//! Because the inner `stereo_data() + aspx_data_2ch() + acpl_data_2ch()`
//! body walker still belongs to a future round, these tests stage the
//! parsed-data side by hand-constructing the data structures the body
//! walker will eventually populate. The wiring asserted here is purely
//! the type contract: the walker hands back the right config slot, the
//! decoder fills in the matching data slot, and the synthesis pipeline
//! consumes the pair without further glue.

use oxideav_ac4::acpl::{
    AcplConfig1ch, AcplConfig2ch, AcplData1ch, AcplData2ch, AcplFramingData, AcplHuffParam,
    AcplInterpolationType, AcplQuantMode,
};
use oxideav_ac4::acpl_synth::{
    run_acpl_5x_mch_pcm, run_acpl_5x_pair_pcm, Acpl5xMchPcmState, Acpl5xPairMode,
    Acpl5xPairPcmState,
};
use oxideav_ac4::asf::SubstreamTools;
use oxideav_ac4::mch::{parse_5x_audio_data_outer, FiveXCodecMode};
use oxideav_ac4::qmf::NUM_QMF_SUBBANDS;
use oxideav_core::bits::{BitReader, BitWriter};

/// Build a synthetic Huffman parameter set for a single param-set row.
fn huff_const(value: i32, num_bands: u32) -> AcplHuffParam {
    AcplHuffParam {
        values: vec![value; num_bands as usize],
        direction_time: false,
    }
}

/// Build a stub `acpl_data_1ch` with constant alpha/beta across one
/// param set.
fn stub_data_1ch(alpha: i32, beta: i32, num_bands: u32) -> AcplData1ch {
    AcplData1ch {
        framing: AcplFramingData {
            interpolation_type: AcplInterpolationType::Smooth,
            num_param_sets_cod: 0,
            num_param_sets: 1,
            param_timeslots: vec![],
        },
        alpha1: vec![huff_const(alpha, num_bands)],
        beta1: vec![huff_const(beta, num_bands)],
    }
}

/// Build a stub `acpl_data_2ch` with constant Huffman rows for every
/// parameter set.
fn stub_data_2ch(num_bands: u32) -> AcplData2ch {
    AcplData2ch {
        framing: AcplFramingData {
            interpolation_type: AcplInterpolationType::Smooth,
            num_param_sets_cod: 0,
            num_param_sets: 1,
            param_timeslots: vec![],
        },
        alpha1: vec![huff_const(4, num_bands)],
        alpha2: vec![huff_const(-3, num_bands)],
        beta1: vec![huff_const(2, num_bands)],
        beta2: vec![huff_const(1, num_bands)],
        beta3: vec![huff_const(0, num_bands)],
        gamma1: vec![huff_const(2, num_bands)],
        gamma2: vec![huff_const(0, num_bands)],
        gamma3: vec![huff_const(0, num_bands)],
        gamma4: vec![huff_const(2, num_bands)],
        gamma5: vec![huff_const(1, num_bands)],
        gamma6: vec![huff_const(1, num_bands)],
    }
}

fn sine_pcm(n: usize, freq_hz: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq_hz / 48_000.0 * i as f32).sin())
        .collect()
}

/// 5_X ASPX_ACPL_3 walker → Pseudocode 118 pipeline glue.
///
/// Walks an I-frame ASPX_ACPL_3 outer header, asserts that
/// `tools.acpl_config_2ch` was populated by the walker, then drives the
/// pipeline end-to-end after stuffing a hand-built `acpl_data_2ch` into
/// the matching tools slot. Verifies five output PCM channels carry
/// energy and stay finite.
#[test]
fn five_x_aspx_acpl_3_walker_to_synthesis_glue() {
    // Build an outer ASPX_ACPL_3 header on a non-iframe so we don't
    // need to feed an aspx_config payload — that exercise belongs to a
    // future round once `parse_aspx_config` has a stable round-trip
    // helper. For non-iframe the walker still populates
    // `five_x_mode == AspxAcpl3` and consumes companding_control(2),
    // which is exactly what the downstream pipeline keys on.
    let mut bw = BitWriter::new();
    bw.write_u32(4, 3); // 5_X_codec_mode = ASPX_ACPL_3
                        // companding_control(2) = b_compand_avg + 2x b_compand_on
    bw.write_bit(false); // b_compand_avg
    bw.write_bit(false); // b_compand_on[0]
    bw.write_bit(false); // b_compand_on[1]
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl3));
    // On a non-iframe acpl_config_2ch isn't carried in the bitstream —
    // it's inherited from the prior I-frame. Stage one explicitly so
    // the downstream pipeline has a config to consume (this is exactly
    // the contract the eventual config-state plumbing will satisfy).
    let cfg = AcplConfig2ch {
        num_param_bands_id: 0,
        num_param_bands: 15,
        quant_mode_0: AcplQuantMode::Fine,
        quant_mode_1: AcplQuantMode::Fine,
    };
    tools.acpl_config_2ch = Some(cfg);
    // Stage `acpl_data_2ch` — this is what a future body walker will
    // populate from the bitstream.
    let data = stub_data_2ch(cfg.num_param_bands);
    tools.acpl_data_2ch = Some(data);

    // Drive the synthesis pipeline straight off the tools.
    let n_slots = 64usize;
    let n = n_slots * NUM_QMF_SUBBANDS;
    let pcm_l = sine_pcm(n, 440.0, 1.0);
    let pcm_r = sine_pcm(n, 220.0, 0.7);
    let pcm_c = sine_pcm(n, 660.0, 0.3);
    let mut state = Acpl5xMchPcmState::new();
    let cfg_used = tools.acpl_config_2ch.as_ref().unwrap();
    let data_used = tools.acpl_data_2ch.as_ref().unwrap();
    let out = run_acpl_5x_mch_pcm(&pcm_l, &pcm_r, &pcm_c, cfg_used, data_used, &mut state)
        .expect("synth runs");
    assert_eq!(out.left.len(), n);
    assert_eq!(out.right.len(), n);
    assert_eq!(out.centre.len(), n);
    assert_eq!(out.left_surround.len(), n);
    assert_eq!(out.right_surround.len(), n);
    let start = 2048usize;
    let energy = |b: &[f32]| -> f64 { b[start..].iter().map(|&s| (s as f64).powi(2)).sum::<f64>() };
    for buf in [
        &out.left,
        &out.right,
        &out.centre,
        &out.left_surround,
        &out.right_surround,
    ] {
        for &s in buf.iter() {
            assert!(s.is_finite(), "non-finite sample in 5.X output");
        }
    }
    assert!(energy(&out.left) > 1e-6, "L silent");
    assert!(energy(&out.right) > 1e-6, "R silent");
    assert!(energy(&out.centre) > 1e-6, "C silent");
}

/// 5_X ASPX_ACPL_2 walker → Pseudocode 117 pipeline glue (mono carrier
/// per A-CPL module — no Ls/Rs surrounds at the QMF input).
#[test]
fn five_x_aspx_acpl_2_walker_to_synthesis_glue() {
    // The walker only consumes companding_control + 1-bit coding_config
    // when `five_x_mode == AspxAcpl{1,2}` and we skip the I-frame
    // config block to side-step the aspx_config opaque exercise.
    let mut bw = BitWriter::new();
    bw.write_u32(3, 3); // 5_X_codec_mode = ASPX_ACPL_2
                        // companding_control(3): sync_flag=1, 1-bit compand_on=1
                        // (no need for compand_avg).
    bw.write_bit(true); // sync_flag
    bw.write_bit(true); // compand_on (sync=1 → only 1 channel-bit)
    bw.write_bit(true); // coding_config = 1 (Cfg1ThreeStereo)
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl2));
    // Stage acpl_config_1ch_full + the two acpl_data_1ch parameter
    // sets (one per parallel ACplModule).
    let cfg = AcplConfig1ch {
        num_param_bands_id: 0,
        num_param_bands: 15,
        quant_mode: AcplQuantMode::Fine,
        qmf_band: 0,
    };
    tools.acpl_config_1ch_full = Some(cfg);
    tools.acpl_data_1ch_pair[0] = Some(stub_data_1ch(4, 2, cfg.num_param_bands));
    tools.acpl_data_1ch_pair[1] = Some(stub_data_1ch(-3, 1, cfg.num_param_bands));

    let n_slots = 64usize;
    let n = n_slots * NUM_QMF_SUBBANDS;
    let pcm_l = sine_pcm(n, 440.0, 1.0);
    let pcm_r = sine_pcm(n, 220.0, 0.7);
    let pcm_c = sine_pcm(n, 660.0, 0.3);
    let cfg_active = tools.acpl_config_1ch_full.as_ref().unwrap();
    let data_1 = tools.acpl_data_1ch_pair[0].as_ref().unwrap();
    let data_2 = tools.acpl_data_1ch_pair[1].as_ref().unwrap();
    let mut state = Acpl5xPairPcmState::new();
    let out = run_acpl_5x_pair_pcm(
        Acpl5xPairMode::AspxAcpl2,
        &pcm_l,
        &pcm_r,
        &pcm_c,
        None,
        None,
        cfg_active,
        data_1,
        cfg_active,
        data_2,
        &mut state,
    )
    .expect("synth runs");
    assert_eq!(out.left.len(), n);
    assert_eq!(out.right.len(), n);
    assert_eq!(out.centre.len(), n);
    assert_eq!(out.left_surround.len(), n);
    assert_eq!(out.right_surround.len(), n);
    let start = 2048usize;
    let energy = |b: &[f32]| -> f64 { b[start..].iter().map(|&s| (s as f64).powi(2)).sum::<f64>() };
    for buf in [
        &out.left,
        &out.right,
        &out.centre,
        &out.left_surround,
        &out.right_surround,
    ] {
        for &s in buf.iter() {
            assert!(s.is_finite(), "non-finite sample in 5.X pair output");
        }
    }
    assert!(energy(&out.left) > 1e-6, "L silent");
    assert!(energy(&out.right) > 1e-6, "R silent");
    assert!(energy(&out.centre) > 1e-6, "C silent");
}

/// 5_X ASPX_ACPL_1 walker → Pseudocode 117 pipeline glue (4 carriers:
/// L/R + Ls/Rs feed two parallel ACplModule's).
#[test]
fn five_x_aspx_acpl_1_walker_to_synthesis_glue() {
    let mut bw = BitWriter::new();
    bw.write_u32(2, 3); // 5_X_codec_mode = ASPX_ACPL_1
                        // companding_control(3): sync_flag=1, 1-bit compand_on=1.
    bw.write_bit(true); // sync_flag
    bw.write_bit(true); // compand_on
    bw.write_bit(true); // coding_config = 1 (Cfg1ThreeStereo)
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl1));
    // PARTIAL 1ch config is what ASPX_ACPL_1 carries.
    let cfg = AcplConfig1ch {
        num_param_bands_id: 0,
        num_param_bands: 15,
        quant_mode: AcplQuantMode::Fine,
        qmf_band: 4, // PARTIAL non-zero
    };
    tools.acpl_config_1ch_partial = Some(cfg);
    tools.acpl_data_1ch_pair[0] = Some(stub_data_1ch(3, 1, cfg.num_param_bands));
    tools.acpl_data_1ch_pair[1] = Some(stub_data_1ch(-2, 2, cfg.num_param_bands));

    let n_slots = 64usize;
    let n = n_slots * NUM_QMF_SUBBANDS;
    let pcm_l = sine_pcm(n, 440.0, 1.0);
    let pcm_r = sine_pcm(n, 220.0, 0.7);
    let pcm_c = sine_pcm(n, 660.0, 0.3);
    let pcm_ls = sine_pcm(n, 110.0, 0.5);
    let pcm_rs = sine_pcm(n, 880.0, 0.4);
    let cfg_active = tools.acpl_config_1ch_partial.as_ref().unwrap();
    let data_1 = tools.acpl_data_1ch_pair[0].as_ref().unwrap();
    let data_2 = tools.acpl_data_1ch_pair[1].as_ref().unwrap();
    let mut state = Acpl5xPairPcmState::new();
    let out = run_acpl_5x_pair_pcm(
        Acpl5xPairMode::AspxAcpl1,
        &pcm_l,
        &pcm_r,
        &pcm_c,
        Some(&pcm_ls),
        Some(&pcm_rs),
        cfg_active,
        data_1,
        cfg_active,
        data_2,
        &mut state,
    )
    .expect("synth runs");
    assert_eq!(out.left.len(), n);
    assert_eq!(out.right.len(), n);
    assert_eq!(out.centre.len(), n);
    assert_eq!(out.left_surround.len(), n);
    assert_eq!(out.right_surround.len(), n);
    for buf in [
        &out.left,
        &out.right,
        &out.centre,
        &out.left_surround,
        &out.right_surround,
    ] {
        for &s in buf.iter() {
            assert!(s.is_finite(), "non-finite sample in 5.X ACPL_1 output");
        }
    }
    let start = 2048usize;
    let e = |b: &[f32]| -> f64 { b[start..].iter().map(|&s| (s as f64).powi(2)).sum::<f64>() };
    // All 5 outputs carry energy in ACPL_1 (Ls/Rs are mixed).
    assert!(e(&out.left) > 1e-6, "L silent");
    assert!(e(&out.right) > 1e-6, "R silent");
    assert!(e(&out.centre) > 1e-6, "C silent");
    assert!(e(&out.left_surround) > 1e-6, "Ls silent");
    assert!(e(&out.right_surround) > 1e-6, "Rs silent");
}

/// Round-25 wiring contract: with the inner-body walker for
/// `ASPX_ACPL_1/2` now plumbed (`mch::parse_aspx_acpl_1_2_inner_body`
/// gated behind `parse_5x_audio_data_outer`), a non-iframe ASPX_ACPL_2
/// header that includes a real `three_channel_data()` outer shell now
/// flows through the walker into `tools.three_channel_data` (which the
/// r22 walker treated as opaque). The downstream A-CPL pair entries
/// stay `None` on a non-iframe (no aspx_config in scope), so we stage
/// them by hand to drive the synthesis pipeline. Verifies the
/// type-contract end-to-end: walker populates the upstream channel
/// data, downstream synthesis still consumes a staged data pair.
#[test]
fn five_x_aspx_acpl_2_walker_inner_body_populates_three_channel_data() {
    use oxideav_ac4::asf::SubstreamTools;
    use oxideav_ac4::mch::FiveXCodingConfig;

    let mut bw = BitWriter::new();
    bw.write_u32(3, 3); // 5_X_codec_mode = ASPX_ACPL_2
                        // companding_control(3): sync_flag=1, compand_on=1.
    bw.write_bit(true);
    bw.write_bit(true);
    bw.write_bit(true); // coding_config = 1 -> three_channel_data
                        // three_channel_data outer:
    bw.write_bit(true); // b_long_frame
    bw.write_u32(10, 6); // max_sfb[0]
    bw.write_u32(0, 4); // chel_matsel
    bw.write_u32(0, 2); // chparam_info #0
    bw.write_u32(0, 2); // chparam_info #1
                        // 3x sf_data(ASF) bodies — each one-section, all-zero spectra:
                        // sect_cb=0 (4 bits), sect_len_incr (3 bits, 7-escape over max_sfb=10):
                        //   max_sfb-1 = 9, esc=7 -> write 7 then 2.
                        // scalefac reference (8 bits) = 120.
                        // snf b_data_exists = 0 (1 bit).
    for _ in 0..3 {
        bw.write_u32(0, 4); // sect_cb=0
        bw.write_u32(7, 3); // sect_len_incr (esc)
        bw.write_u32(2, 3); // sect_len_incr remainder
        bw.write_u32(120, 8); // scalefac reference
        bw.write_bit(false); // b_snf_data_exists=0
    }
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl2));
    assert_eq!(
        tools.five_x_coding_config,
        Some(FiveXCodingConfig::Cfg1ThreeStereo)
    );
    // The new r25 walker populates this slot from the bitstream — r22
    // would have left it None.
    let three = tools
        .three_channel_data
        .as_ref()
        .expect("r25 walker populates three_channel_data on the ASPX_ACPL_{1,2} body");
    assert_eq!(three.psy_info.as_ref().unwrap().max_sfb_0, 10);

    // Downstream: stage acpl_config + pair (non-iframe path), then
    // drive the synthesis pipeline as in the r22 contract test.
    let cfg = AcplConfig1ch {
        num_param_bands_id: 0,
        num_param_bands: 15,
        quant_mode: AcplQuantMode::Fine,
        qmf_band: 0,
    };
    tools.acpl_config_1ch_full = Some(cfg);
    tools.acpl_data_1ch_pair[0] = Some(stub_data_1ch(2, 1, cfg.num_param_bands));
    tools.acpl_data_1ch_pair[1] = Some(stub_data_1ch(-1, 2, cfg.num_param_bands));

    let n_slots = 64usize;
    let n = n_slots * NUM_QMF_SUBBANDS;
    let pcm_l = sine_pcm(n, 440.0, 1.0);
    let pcm_r = sine_pcm(n, 220.0, 0.7);
    let pcm_c = sine_pcm(n, 660.0, 0.3);
    let cfg_active = tools.acpl_config_1ch_full.as_ref().unwrap();
    let data_1 = tools.acpl_data_1ch_pair[0].as_ref().unwrap();
    let data_2 = tools.acpl_data_1ch_pair[1].as_ref().unwrap();
    let mut state = Acpl5xPairPcmState::new();
    let out = run_acpl_5x_pair_pcm(
        Acpl5xPairMode::AspxAcpl2,
        &pcm_l,
        &pcm_r,
        &pcm_c,
        None,
        None,
        cfg_active,
        data_1,
        cfg_active,
        data_2,
        &mut state,
    )
    .expect("synth runs");
    assert_eq!(out.left.len(), n);
    assert_eq!(out.centre.len(), n);
    for s in out.left.iter().chain(&out.right).chain(&out.centre) {
        assert!(s.is_finite());
    }
}

/// Round-25 wiring contract for the ASPX_ACPL_1 path: the inner-body
/// walker now also walks the joint-MDCT residual layer
/// (`max_sfb_master + 2x chparam_info + 2x sf_data(ASF)`) on top of
/// the upstream `two_channel_data()` / `three_channel_data()`. We feed
/// a Cfg0 (two_channel_data branch) frame on the non-iframe path,
/// assert the upstream channel data is parsed, and that the trailing
/// `mono_data(0)` for the centre channel ends up in
/// `tools.cfg0_centre_mono`. Synthesis is driven from staged ACPL data
/// (matching the r22 contract).
#[test]
fn five_x_aspx_acpl_1_walker_inner_body_populates_two_channel_and_centre() {
    use oxideav_ac4::asf::SubstreamTools;
    use oxideav_ac4::mch::FiveXCodingConfig;

    let mut bw = BitWriter::new();
    bw.write_u32(2, 3); // 5_X_codec_mode = ASPX_ACPL_1
                        // companding_control(3): sync_flag=1, compand_on=1.
    bw.write_bit(true);
    bw.write_bit(true);
    bw.write_bit(false); // coding_config = 0 -> two_channel_data + Cfg0 mono
                         // two_channel_data outer (Table 26):
    bw.write_bit(true); // b_long_frame
    bw.write_u32(8, 6); // max_sfb[0]
    bw.write_u32(0, 2); // chparam sap_mode = 0
                        // 2x sf_data(ASF) bodies (max_sfb=8, transf_length_idx=0):
                        //   sect_cb=0 (4) + sect_len_incr=7 (3) + sect_len_incr=0 (3)
                        //   + scalefac ref (8) + snf b_data_exists (1).
    for _ in 0..2 {
        bw.write_u32(0, 4);
        bw.write_u32(7, 3);
        bw.write_u32(0, 3);
        bw.write_u32(120, 8);
        bw.write_bit(false);
    }
    // Joint-MDCT residual layer: max_sfb_master=4 (n_side_bits=5 @1920),
    // 2x chparam (sap_mode=0), 2x sf_data with max_sfb=4.
    bw.write_u32(4, 5);
    bw.write_u32(0, 2);
    bw.write_u32(0, 2);
    for _ in 0..2 {
        bw.write_u32(0, 4); // sect_cb=0
        bw.write_u32(3, 3); // sect_len_incr (max_sfb-1 = 3, no escape)
        bw.write_u32(120, 8); // scalefac ref
        bw.write_bit(false); // snf
    }
    // Cfg0 trailer: mono_data(0).
    bw.write_bit(false); // spec_frontend = ASF
    bw.write_bit(true); // b_long_frame
    bw.write_u32(5, 6); // max_sfb[0] for centre mono
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::AspxAcpl1));
    assert_eq!(
        tools.five_x_coding_config,
        Some(FiveXCodingConfig::AcplLite2)
    );
    // r25 walker populates the upstream two_channel_data slot.
    assert_eq!(tools.two_channel_data.len(), 1);
    assert_eq!(
        tools.two_channel_data[0]
            .psy_info
            .as_ref()
            .unwrap()
            .max_sfb_0,
        8
    );
    // r25 walker also lands the Cfg0 centre `mono_data(0)`.
    let centre = tools
        .cfg0_centre_mono
        .as_ref()
        .expect("Cfg0 centre mono walked");
    assert_eq!(centre.psy_info.as_ref().unwrap().max_sfb_0, 5);

    // Stage ACPL pair (PARTIAL config for ASPX_ACPL_1) and drive
    // run_acpl_5x_pair_pcm.
    let cfg = AcplConfig1ch {
        num_param_bands_id: 0,
        num_param_bands: 15,
        quant_mode: AcplQuantMode::Fine,
        qmf_band: 4,
    };
    tools.acpl_config_1ch_partial = Some(cfg);
    tools.acpl_data_1ch_pair[0] = Some(stub_data_1ch(3, 1, cfg.num_param_bands));
    tools.acpl_data_1ch_pair[1] = Some(stub_data_1ch(-2, 2, cfg.num_param_bands));

    let n_slots = 32usize;
    let n = n_slots * NUM_QMF_SUBBANDS;
    let pcm_l = sine_pcm(n, 440.0, 1.0);
    let pcm_r = sine_pcm(n, 220.0, 0.7);
    let pcm_c = sine_pcm(n, 660.0, 0.3);
    let pcm_ls = sine_pcm(n, 110.0, 0.5);
    let pcm_rs = sine_pcm(n, 880.0, 0.4);
    let cfg_active = tools.acpl_config_1ch_partial.as_ref().unwrap();
    let data_1 = tools.acpl_data_1ch_pair[0].as_ref().unwrap();
    let data_2 = tools.acpl_data_1ch_pair[1].as_ref().unwrap();
    let mut state = Acpl5xPairPcmState::new();
    let out = run_acpl_5x_pair_pcm(
        Acpl5xPairMode::AspxAcpl1,
        &pcm_l,
        &pcm_r,
        &pcm_c,
        Some(&pcm_ls),
        Some(&pcm_rs),
        cfg_active,
        data_1,
        cfg_active,
        data_2,
        &mut state,
    )
    .expect("synth runs");
    assert_eq!(out.left.len(), n);
    for s in out
        .left
        .iter()
        .chain(&out.right)
        .chain(&out.centre)
        .chain(&out.left_surround)
        .chain(&out.right_surround)
    {
        assert!(s.is_finite());
    }
}

/// Round 37: 7_X ASPX_ACPL_2 walker → Pseudocode 120 pipeline glue.
///
/// Walks a non-iframe 7_X ASPX_ACPL_2 outer header (so we can skip the
/// I-frame `aspx_config()` blob), then drives the same Pseudocode 117
/// `run_acpl_5x_pair_pcm` core that Pseudocode 120 reuses. For
/// `ASPX_ACPL_{1,2}` the additional 2 channels (z6/z7 in Pseudocode
/// 120) live outside the A-CPL pair so this glue test only checks the
/// 5-channel core (L/R/C/Ls/Rs) — same as the 5_X path.
#[test]
fn seven_x_aspx_acpl_2_walker_to_synthesis_glue() {
    use oxideav_ac4::mch::{parse_7x_audio_data_outer, SevenXCodecMode};
    // 7_X_codec_mode = ASPX_ACPL_2 (2 bits, value 3).
    let mut bw = BitWriter::new();
    bw.write_u32(3, 2); // 7_X_codec_mode = ASPX_ACPL_2
                        // No b_iframe path -> no aspx_config / acpl_config_1ch in the
                        // bitstream. companding_control(5) for ACPL_{1,2}:
                        //   sync_flag=1 -> single b_compand_on bit (5 channels share).
    bw.write_bit(true); // sync_flag
    bw.write_bit(true); // compand_on (sync=1 → 1 channel-bit)
                        // coding_config = 0 (Cfg0Stereo2plusMono). Body bails
                        // before the inner 2ch_mode bit due to non-iframe gate
                        // on aspx_data trailers — that's fine; we exit cleanly
                        // and verify the seven_x_mode resolves correctly.
    bw.write_u32(0, 2); // coding_config = 0
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_7x_audio_data_outer(&mut br, &mut tools, false, false, 1920).unwrap();
    assert_eq!(tools.seven_x_mode, Some(SevenXCodecMode::AspxAcpl2));
    // Stage acpl_config_1ch_full + the two acpl_data_1ch parameter
    // sets (one per parallel ACplModule). This is what a future
    // I-frame body walker / state-replay path will populate.
    let cfg = AcplConfig1ch {
        num_param_bands_id: 0,
        num_param_bands: 15,
        quant_mode: AcplQuantMode::Fine,
        qmf_band: 0,
    };
    tools.acpl_config_1ch_full = Some(cfg);
    let data_1 = stub_data_1ch(2, 1, cfg.num_param_bands);
    let data_2 = stub_data_1ch(-1, 2, cfg.num_param_bands);
    tools.acpl_data_1ch_pair[0] = Some(data_1.clone());
    tools.acpl_data_1ch_pair[1] = Some(data_2.clone());

    // Drive the same Pseudocode 117 pipeline (which Pseudocode 120
    // wraps for 7_X) on synthetic L/R/C carrier PCM.
    let n_slots = 64usize;
    let n = n_slots * NUM_QMF_SUBBANDS;
    let pcm_l = sine_pcm(n, 440.0, 1.0);
    let pcm_r = sine_pcm(n, 220.0, 0.7);
    let pcm_c = sine_pcm(n, 660.0, 0.3);
    let mut state = Acpl5xPairPcmState::new();
    let out = run_acpl_5x_pair_pcm(
        Acpl5xPairMode::AspxAcpl2,
        &pcm_l,
        &pcm_r,
        &pcm_c,
        None,
        None,
        &cfg,
        &data_1,
        &cfg,
        &data_2,
        &mut state,
    )
    .expect("synth runs");
    assert_eq!(out.left.len(), n);
    assert_eq!(out.right.len(), n);
    assert_eq!(out.centre.len(), n);
    assert_eq!(out.left_surround.len(), n);
    assert_eq!(out.right_surround.len(), n);
    for s in out
        .left
        .iter()
        .chain(&out.right)
        .chain(&out.centre)
        .chain(&out.left_surround)
        .chain(&out.right_surround)
    {
        assert!(s.is_finite(), "non-finite sample in 7.X output");
    }
}

/// Round 38: 5_X SIMPLE/ASPX `coding_config == 2` walker → tools layout
/// glue. After `parse_5x_audio_data_outer` walks a Cfg2 frame the parsed
/// `four_channel_data.scaled_spec_per_channel[0..4]` carries L/R/Ls/Rs
/// (per Table 180) and `cfg2_back_mono` carries the centre. This test
/// verifies the walker hands back the right slots — the downstream
/// IMDCT dispatch (`Ac4Decoder::dispatch_5x_cfg2_simple_aspx`) is
/// covered by the unit tests in `decoder.rs`.
#[test]
fn five_x_simple_cfg2_walker_populates_four_plus_back_mono() {
    use oxideav_ac4::asf::SubstreamTools;
    use oxideav_ac4::mch::{parse_5x_audio_data_outer, FiveXCodecMode, FiveXCodingConfig};

    // 5_X_codec_mode = SIMPLE (3 bits, value 0). No LFE.
    // coding_config = 2 (Cfg2FourMono).
    // four_channel_data: long-frame, max_sfb=20, 4x chparam_info + 4x
    // all-zero sf_data bodies.
    // mono_data(0): spec_frontend=ASF, long-frame, max_sfb=8, all-zero
    // sf_data body.
    let mut bw = BitWriter::new();
    bw.write_u32(0, 3); // 5_X_codec_mode = SIMPLE
    bw.write_u32(2, 2); // coding_config = 2
    bw.write_bit(true); // b_long_frame
    bw.write_u32(20, 6); // max_sfb[0]
    for _ in 0..4 {
        bw.write_u32(0, 2); // chparam_info #i
    }
    // 4x sf_data(ASF) all-zero bodies.
    let write_zero = |bw: &mut BitWriter, max_sfb: u32| {
        // sect_cb=0 (4 bits), sect_len_incr accumulating to max_sfb-1
        // (3-bit slots with 7-escape).
        bw.write_u32(0, 4);
        let mut remaining = max_sfb.saturating_sub(1);
        while remaining >= 7 {
            bw.write_u32(7, 3);
            remaining -= 7;
        }
        bw.write_u32(remaining, 3);
        bw.write_u32(120, 8); // scalefac reference
        bw.write_bit(false); // b_snf_data_exists=0
    };
    for _ in 0..4 {
        write_zero(&mut bw, 20);
    }
    // Trailing mono_data(0) for the centre.
    bw.write_bit(false); // spec_frontend = ASF
    bw.write_bit(true); // b_long_frame
    bw.write_u32(8, 6); // max_sfb[0]
    write_zero(&mut bw, 8);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::Simple));
    assert_eq!(
        tools.five_x_coding_config,
        Some(FiveXCodingConfig::Cfg2FourMono)
    );
    let four = tools
        .four_channel_data
        .as_ref()
        .expect("four_channel_data populated");
    // All four per-channel scaled spectra must be present and length-
    // matched to sfb_offset[max_sfb=20] for tl=1920.
    assert_eq!(four.scaled_spec_per_channel.len(), 4);
    for (i, ch) in four.scaled_spec_per_channel.iter().enumerate() {
        assert!(
            ch.is_some(),
            "four_channel_data slot {i} must carry scaled_spec"
        );
    }
    let back = tools
        .cfg2_back_mono
        .as_ref()
        .expect("cfg2_back_mono populated");
    assert!(!back.b_lfe);
    assert_eq!(back.psy_info.as_ref().unwrap().max_sfb_0, 8);
    // Round 38: trailing mono_data body now decodes into scaled_spec.
    assert!(
        back.scaled_spec.is_some(),
        "round 38: cfg2 back mono body walks into scaled_spec"
    );
}

/// Round 39: 5_X SIMPLE/ASPX `coding_config == 0` walker → tools layout
/// glue. After `parse_5x_audio_data_outer` walks a Cfg0 frame the parsed
/// `two_channel_data[0/1].scaled_spec_per_channel[0..2]` carries the
/// L/R + Ls/Rs (2ch_mode == 0) or L/Ls + R/Rs (2ch_mode == 1) pairs
/// per Table 180, and `cfg0_centre_mono` carries the centre channel.
#[test]
fn five_x_simple_cfg0_walker_populates_two_two_plus_centre() {
    use oxideav_ac4::asf::SubstreamTools;
    use oxideav_ac4::mch::{parse_5x_audio_data_outer, FiveXCodecMode, FiveXCodingConfig};
    let mut bw = BitWriter::new();
    bw.write_u32(0, 3); // 5_X_codec_mode = SIMPLE
    bw.write_u32(0, 2); // coding_config = 0 (Cfg0Stereo2plusMono)
    bw.write_bit(false); // b_2ch_mode = 0
                         // 2x two_channel_data: long-frame max_sfb=20, chparam_info, 2x sf_data
    let write_zero = |bw: &mut BitWriter, max_sfb: u32| {
        bw.write_u32(0, 4);
        let mut remaining = max_sfb.saturating_sub(1);
        while remaining >= 7 {
            bw.write_u32(7, 3);
            remaining -= 7;
        }
        bw.write_u32(remaining, 3);
        bw.write_u32(120, 8);
        bw.write_bit(false);
    };
    for _ in 0..2 {
        bw.write_bit(true); // b_long_frame
        bw.write_u32(20, 6); // max_sfb[0]
        bw.write_u32(0, 2); // chparam_info
        for _ in 0..2 {
            write_zero(&mut bw, 20);
        }
    }
    // mono_data(0) for centre — spec_frontend = ASF, long-frame, max_sfb=8.
    bw.write_bit(false); // spec_frontend = ASF
    bw.write_bit(true); // b_long_frame
    bw.write_u32(8, 6); // max_sfb[0]
    write_zero(&mut bw, 8);
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::Simple));
    assert_eq!(
        tools.five_x_coding_config,
        Some(FiveXCodingConfig::Cfg0Stereo2plusMono)
    );
    assert_eq!(tools.b_2ch_mode, Some(false));
    assert_eq!(tools.two_channel_data.len(), 2);
    for (idx, tcd) in tools.two_channel_data.iter().enumerate() {
        assert_eq!(tcd.scaled_spec_per_channel.len(), 2);
        for (ch, slot) in tcd.scaled_spec_per_channel.iter().enumerate() {
            assert!(
                slot.is_some(),
                "cfg0 two_channel_data[{idx}] slot {ch} must carry scaled_spec"
            );
        }
    }
    let centre = tools
        .cfg0_centre_mono
        .as_ref()
        .expect("cfg0_centre_mono populated");
    assert!(centre.scaled_spec.is_some());
}

/// Round 39: 5_X SIMPLE/ASPX `coding_config == 3` walker → tools layout
/// glue. After `parse_5x_audio_data_outer` walks a Cfg3 frame the parsed
/// `five_channel_data.scaled_spec_per_channel[0..5]` carries L/R/C/Ls/Rs.
#[test]
fn five_x_simple_cfg3_walker_populates_five_channel_data() {
    use oxideav_ac4::asf::SubstreamTools;
    use oxideav_ac4::mch::{parse_5x_audio_data_outer, FiveXCodecMode, FiveXCodingConfig};
    let mut bw = BitWriter::new();
    bw.write_u32(0, 3); // 5_X_codec_mode = SIMPLE
    bw.write_u32(3, 2); // coding_config = 3 (Cfg3Five)
                        // five_channel_data: long-frame max_sfb=20, chel_matsel(4) +
                        // 5x chparam_info + 5x sf_data(ASF)
    bw.write_bit(true); // b_long_frame
    bw.write_u32(20, 6); // max_sfb[0]
    bw.write_u32(0, 4); // chel_matsel
    for _ in 0..5 {
        bw.write_u32(0, 2); // chparam_info
    }
    let write_zero = |bw: &mut BitWriter, max_sfb: u32| {
        bw.write_u32(0, 4);
        let mut remaining = max_sfb.saturating_sub(1);
        while remaining >= 7 {
            bw.write_u32(7, 3);
            remaining -= 7;
        }
        bw.write_u32(remaining, 3);
        bw.write_u32(120, 8);
        bw.write_bit(false);
    };
    for _ in 0..5 {
        write_zero(&mut bw, 20);
    }
    bw.align_to_byte();
    let bytes = bw.finish();
    let mut br = BitReader::new(&bytes);
    let mut tools = SubstreamTools::default();
    parse_5x_audio_data_outer(&mut br, &mut tools, false, true, 1920).unwrap();
    assert_eq!(tools.five_x_mode, Some(FiveXCodecMode::Simple));
    assert_eq!(
        tools.five_x_coding_config,
        Some(FiveXCodingConfig::Cfg3Five)
    );
    let five = tools
        .five_channel_data
        .as_ref()
        .expect("five_channel_data populated");
    assert_eq!(five.scaled_spec_per_channel.len(), 5);
    for (i, ch) in five.scaled_spec_per_channel.iter().enumerate() {
        assert!(
            ch.is_some(),
            "five_channel_data slot {i} must carry scaled_spec"
        );
    }
}
