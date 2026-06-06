//! Round-trip pinning for the encoder-side `chparam_info()` /
//! `sap_data()` builders introduced in round 243.
//!
//! `oxideav_ac4::encoder_asf::write_chparam_info` is the dual of
//! `oxideav_ac4::asf::parse_chparam_info` per ETSI TS 103 190-1
//! §4.2.10.1 Table 47. The tests below pin the round-trip across all
//! four `sap_mode` codes (None / MsUsed / Reserved / SapData) plus
//! representative edge cases (multi-group, sap_coeff_all, asymmetric
//! pair entry, sub-`max_sfb` rows in the input).

use oxideav_ac4::asf::{parse_chparam_info, ChparamInfo, SapData, SapMode};
use oxideav_ac4::encoder_asf::write_chparam_info;
use oxideav_core::bits::{BitReader, BitWriter};

fn roundtrip(info: &ChparamInfo, max_sfb_per_group: &[u32]) -> ChparamInfo {
    let mut bw = BitWriter::new();
    write_chparam_info(&mut bw, info, max_sfb_per_group);
    let bytes = bw.into_bytes();
    let mut br = BitReader::new(&bytes);
    parse_chparam_info(&mut br, max_sfb_per_group).expect("parse_chparam_info")
}

#[test]
fn sap_mode_zero_header_only_two_bits() {
    let info = ChparamInfo {
        sap_mode: 0,
        ..ChparamInfo::default()
    };
    let mut bw = BitWriter::new();
    write_chparam_info(&mut bw, &info, &[4]);
    assert_eq!(bw.bit_position(), 2);
    let out = roundtrip(&info, &[4]);
    assert_eq!(out.sap_mode, 0);
    assert_eq!(out.mode(), SapMode::None);
    assert!(out.ms_used.is_empty());
    assert!(out.sap_data.is_none());
}

#[test]
fn sap_mode_two_reserved_header_only_two_bits() {
    let info = ChparamInfo {
        sap_mode: 2,
        ..ChparamInfo::default()
    };
    let mut bw = BitWriter::new();
    write_chparam_info(&mut bw, &info, &[6]);
    assert_eq!(bw.bit_position(), 2);
    let out = roundtrip(&info, &[6]);
    assert_eq!(out.sap_mode, 2);
    assert_eq!(out.mode(), SapMode::Reserved);
}

#[test]
fn sap_mode_one_ms_used_single_group_roundtrip() {
    let info = ChparamInfo {
        sap_mode: 1,
        ms_used: vec![vec![true, false, true, false, true]],
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[5]);
    assert_eq!(out.sap_mode, 1);
    assert_eq!(out.ms_used, vec![vec![true, false, true, false, true]]);
    assert!(out.sap_data.is_none());
}

#[test]
fn sap_mode_one_ms_used_multi_group_roundtrip() {
    let info = ChparamInfo {
        sap_mode: 1,
        ms_used: vec![
            vec![true, true, false],
            vec![false, true, false, true],
            vec![true],
        ],
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[3, 4, 1]);
    assert_eq!(out.sap_mode, 1);
    assert_eq!(out.ms_used.len(), 3);
    assert_eq!(out.ms_used[0], vec![true, true, false]);
    assert_eq!(out.ms_used[1], vec![false, true, false, true]);
    assert_eq!(out.ms_used[2], vec![true]);
}

#[test]
fn sap_mode_one_missing_ms_used_row_treated_as_zero() {
    // Half-built ChparamInfo: sap_mode = 1 but ms_used carries fewer
    // rows than max_sfb_per_group. The writer fills missing entries
    // with `false` rather than panicking; the parser then recovers
    // the zero-filled tail.
    let info = ChparamInfo {
        sap_mode: 1,
        ms_used: vec![vec![true, false]],
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[2, 2]);
    assert_eq!(out.ms_used.len(), 2);
    assert_eq!(out.ms_used[0], vec![true, false]);
    assert_eq!(out.ms_used[1], vec![false, false]);
}

#[test]
fn sap_mode_three_sap_coeff_all_single_group_roundtrip() {
    let sd = SapData {
        sap_coeff_all: true,
        sap_coeff_used: vec![vec![true; 4]],
        delta_code_time: false,
        dpcm_alpha_q: vec![vec![3, 0, -2, 0]],
    };
    let info = ChparamInfo {
        sap_mode: 3,
        sap_data: Some(sd),
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[4]);
    assert_eq!(out.sap_mode, 3);
    let parsed = out.sap_data.expect("sap_data present");
    assert!(parsed.sap_coeff_all);
    assert_eq!(parsed.sap_coeff_used, vec![vec![true; 4]]);
    // delta_code_time only transmitted for num_window_groups != 1;
    // single-group default to false.
    assert!(!parsed.delta_code_time);
    // sfb walked in 2-strides, dpcm_alpha_q for even sfb only.
    // Pair (0,1) → delta 3, pair (2,3) → delta -2.
    assert_eq!(parsed.dpcm_alpha_q[0][0], 3);
    assert_eq!(parsed.dpcm_alpha_q[0][2], -2);
}

#[test]
fn sap_mode_three_partial_pairs_single_group_roundtrip() {
    // sap_coeff_all = 0: per-pair flag bits drive which pairs carry
    // DPCM payload. The parser copies each pair-flag into BOTH halves
    // of the pair; the writer reads the even half as the canonical
    // value.
    let sd = SapData {
        sap_coeff_all: false,
        sap_coeff_used: vec![vec![true, true, false, false, true, true]],
        delta_code_time: false,
        dpcm_alpha_q: vec![vec![5, 0, 0, 0, -7, 0]],
    };
    let info = ChparamInfo {
        sap_mode: 3,
        sap_data: Some(sd),
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[6]);
    let parsed = out.sap_data.expect("sap_data present");
    assert!(!parsed.sap_coeff_all);
    assert_eq!(
        parsed.sap_coeff_used[0],
        vec![true, true, false, false, true, true]
    );
    assert_eq!(parsed.dpcm_alpha_q[0][0], 5);
    assert_eq!(parsed.dpcm_alpha_q[0][2], 0); // pair (2,3) flag was false
    assert_eq!(parsed.dpcm_alpha_q[0][4], -7);
}

#[test]
fn sap_mode_three_multi_group_delta_code_time_roundtrip() {
    let sd = SapData {
        sap_coeff_all: true,
        sap_coeff_used: vec![vec![true; 2], vec![true; 4]],
        delta_code_time: true,
        dpcm_alpha_q: vec![vec![2, 0], vec![-3, 0, 4, 0]],
    };
    let info = ChparamInfo {
        sap_mode: 3,
        sap_data: Some(sd),
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[2, 4]);
    let parsed = out.sap_data.expect("sap_data present");
    assert!(parsed.sap_coeff_all);
    assert!(parsed.delta_code_time);
    assert_eq!(parsed.dpcm_alpha_q[0][0], 2);
    assert_eq!(parsed.dpcm_alpha_q[1][0], -3);
    assert_eq!(parsed.dpcm_alpha_q[1][2], 4);
}

#[test]
fn sap_mode_three_missing_sap_data_emits_default_body() {
    // Degenerate input — sap_mode = 3 yet sap_data is None. The
    // writer emits a default body so the parser walks the bytes; the
    // recovered ChparamInfo has sap_mode = 3 with a sap_coeff_all =
    // false / empty sap_coeff_used SapData.
    let info = ChparamInfo {
        sap_mode: 3,
        sap_data: None,
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[4]);
    assert_eq!(out.sap_mode, 3);
    let parsed = out.sap_data.expect("default sap_data emitted");
    // Default SapData has sap_coeff_all = false; parser walks the
    // pair-flag bits and recovers an all-zero pair-flag row.
    assert!(!parsed.sap_coeff_all);
    // Each even-sfb flag was false → entire row copies as false.
    assert_eq!(parsed.sap_coeff_used[0], vec![false; 4]);
    // dpcm_alpha_q for an all-false pair row is all zeros.
    assert_eq!(parsed.dpcm_alpha_q[0], vec![0; 4]);
}

#[test]
fn sap_mode_three_dpcm_delta_clamps_at_codebook_boundary() {
    // HCB_SCALEFAC indexes [0, 120]; raw = delta + 60 → in-range
    // deltas live in [-60, +60]. Out-of-range deltas clamp at the
    // boundary, mirroring write_scalefac_data's policy. The reader
    // recovers the clamped value (+60 or -60), not the original
    // out-of-range delta.
    let sd = SapData {
        sap_coeff_all: true,
        sap_coeff_used: vec![vec![true, true, true, true]],
        delta_code_time: false,
        dpcm_alpha_q: vec![vec![200, 0, -200, 0]],
    };
    let info = ChparamInfo {
        sap_mode: 3,
        sap_data: Some(sd),
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[4]);
    let parsed = out.sap_data.expect("sap_data present");
    assert_eq!(parsed.dpcm_alpha_q[0][0], 60);
    assert_eq!(parsed.dpcm_alpha_q[0][2], -60);
}

#[test]
fn sap_mode_three_dpcm_in_range_full_sweep_roundtrip() {
    // Spot-check every legal delta value across the codebook's full
    // [-60, +60] range. Each value should round-trip exactly.
    for delta in -60i32..=60i32 {
        let sd = SapData {
            sap_coeff_all: true,
            sap_coeff_used: vec![vec![true, true]],
            delta_code_time: false,
            dpcm_alpha_q: vec![vec![delta, 0]],
        };
        let info = ChparamInfo {
            sap_mode: 3,
            sap_data: Some(sd),
            ..ChparamInfo::default()
        };
        let out = roundtrip(&info, &[2]);
        let parsed = out.sap_data.expect("sap_data present");
        assert_eq!(parsed.dpcm_alpha_q[0][0], delta, "delta {delta}");
    }
}

#[test]
fn sap_mode_zero_with_populated_ms_used_drops_payload() {
    // sap_mode = 0 → header-only on the wire, even if ms_used /
    // sap_data happen to be populated on the input struct. Parser
    // recovers mode = None with empty payload.
    let info = ChparamInfo {
        sap_mode: 0,
        ms_used: vec![vec![true, true, true]],
        sap_data: Some(SapData::default()),
    };
    let mut bw = BitWriter::new();
    write_chparam_info(&mut bw, &info, &[3]);
    assert_eq!(bw.bit_position(), 2);
    let out = roundtrip(&info, &[3]);
    assert_eq!(out.sap_mode, 0);
    assert!(out.ms_used.is_empty());
    assert!(out.sap_data.is_none());
}

#[test]
fn sap_mode_high_bits_masked_to_two() {
    // The on-wire sap_mode is a 2-bit field; an in-memory value
    // with extra high bits gets masked to its low 2 bits by the
    // writer, matching the parser's `read_u32(2)` contract.
    let info = ChparamInfo {
        sap_mode: 0b1011, // = 3 mod 4
        sap_data: Some(SapData::default()),
        ..ChparamInfo::default()
    };
    let out = roundtrip(&info, &[2]);
    assert_eq!(out.sap_mode, 3);
}
