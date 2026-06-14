//! Round 306 — encoder-side `aspx_hfgen_iwc_1ch()` / `aspx_hfgen_iwc_2ch()`
//! writers (ETSI TS 103 190-1 §4.2.12.6 / §4.2.12.7, Tables 55 / 56).
//!
//! Until now every encoder body writer emitted the HF-generation /
//! interleaved-waveform-coding element as the all-zero compact form:
//! `aspx_tna_mode[*] = 0`, `aspx_ah_present = aspx_fic_present =
//! aspx_tic_present = 0`. The decoder, however, fully parses real
//! inverse-filtering modes (`tna_mode`), additive harmonics
//! (`add_harmonic`), frequency-interleaved coding (`fic_used_in_sfb`) and
//! time-interleaved coding (`tic_used_in_slot`)
//! ([`oxideav_ac4::aspx::parse_aspx_hfgen_iwc_1ch`] /
//! `parse_aspx_hfgen_iwc_2ch`).
//!
//! This round adds the exact encoder duals
//! ([`oxideav_ac4::encoder_acpl3::write_aspx_hfgen_iwc_1ch`] /
//! `write_aspx_hfgen_iwc_2ch`) and pins the bit-exact round-trip: the
//! writer's emitted bits, re-parsed by the decoder, recover the input
//! `tna_mode` / `add_harmonic` / `fic_used_in_sfb` / `tic_used_in_slot`
//! decisions. The writers auto-derive the `*_present` / `*_left` /
//! `*_right` / `tic_copy` gates from the payload.
//!
//! Refs ETSI TS 103 190-1 §4.2.12.6 Table 55, §4.2.12.7 Table 56.

use oxideav_ac4::aspx::{parse_aspx_hfgen_iwc_1ch, parse_aspx_hfgen_iwc_2ch};
use oxideav_ac4::encoder_acpl3::{
    write_aspx_hfgen_iwc_1ch, write_aspx_hfgen_iwc_2ch, AspxHfgenIwc1ChPayload,
    AspxHfgenIwc2ChPayload,
};
use oxideav_core::bits::{BitReader, BitWriter};

// --------------------------------------------------------------------
// 1-channel
// --------------------------------------------------------------------

/// All-zero / all-absent payload reproduces the historical compact form:
/// tna_mode all 0, all three presence bits 0, all flag vectors false.
#[test]
fn hfgen_1ch_all_zero_round_trips() {
    let num_sbg_noise = 3;
    let num_sbg_sig_highres = 5;
    let num_aspx_timeslots = 4;

    let payload = AspxHfgenIwc1ChPayload::default();
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_1ch(
        &mut bw,
        &payload,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert_eq!(parsed.tna_mode, vec![0u8; num_sbg_noise as usize]);
    assert!(!parsed.ah_present);
    assert_eq!(
        parsed.add_harmonic,
        vec![false; num_sbg_sig_highres as usize]
    );
    assert!(!parsed.fic_present);
    assert_eq!(
        parsed.fic_used_in_sfb,
        vec![false; num_sbg_sig_highres as usize]
    );
    assert!(!parsed.tic_present);
    assert_eq!(
        parsed.tic_used_in_slot,
        vec![false; num_aspx_timeslots as usize]
    );

    // Compact form: 3×2 (tna) + 3×1 (presence) = 9 bits → 2 bytes.
    assert_eq!(bytes.len(), 2);
}

/// Real tna_mode plus active additive-harmonic / FIC / TIC vectors all
/// survive the round-trip; the gates flip to 1 automatically.
#[test]
fn hfgen_1ch_real_flags_round_trip() {
    let num_sbg_noise = 4;
    let num_sbg_sig_highres = 6;
    let num_aspx_timeslots = 5;

    let tna_mode = [1u8, 3, 0, 2];
    let add_harmonic = [false, true, false, false, true, false];
    let fic_used_in_sfb = [true, false, false, true, false, false];
    let tic_used_in_slot = [false, false, true, false, false];

    let payload = AspxHfgenIwc1ChPayload {
        tna_mode: &tna_mode,
        add_harmonic: &add_harmonic,
        fic_used_in_sfb: &fic_used_in_sfb,
        tic_used_in_slot: &tic_used_in_slot,
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_1ch(
        &mut bw,
        &payload,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert_eq!(parsed.tna_mode, tna_mode.to_vec());
    assert!(parsed.ah_present);
    assert_eq!(parsed.add_harmonic, add_harmonic.to_vec());
    assert!(parsed.fic_present);
    assert_eq!(parsed.fic_used_in_sfb, fic_used_in_sfb.to_vec());
    assert!(parsed.tic_present);
    assert_eq!(parsed.tic_used_in_slot, tic_used_in_slot.to_vec());
}

/// Short input slices zero-pad; tna values above 3 mask to low 2 bits.
#[test]
fn hfgen_1ch_padding_and_masking() {
    let num_sbg_noise = 4;
    let num_sbg_sig_highres = 4;
    let num_aspx_timeslots = 3;

    // tna_mode short (2 of 4) and one value (7 → 0b11 = 3) needs masking.
    let tna_mode = [7u8, 2];
    let payload = AspxHfgenIwc1ChPayload {
        tna_mode: &tna_mode,
        ..Default::default()
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_1ch(
        &mut bw,
        &payload,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_1ch(
        &mut br,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    // 7 & 3 = 3, then 2, then zero-padded to 4 entries.
    assert_eq!(parsed.tna_mode, vec![3, 2, 0, 0]);
    assert!(!parsed.ah_present);
}

// --------------------------------------------------------------------
// 2-channel
// --------------------------------------------------------------------

/// balance = 1: only channel-0 tna_mode is written; the decoder mirrors
/// it into channel 1. Active per-channel harmonics survive.
#[test]
fn hfgen_2ch_balance_mirrors_tna() {
    let num_sbg_noise = 3;
    let num_sbg_sig_highres = 4;
    let num_aspx_timeslots = 4;

    let tna0 = [1u8, 2, 3];
    let tna1_ignored = [0u8, 0, 0]; // ignored under balance = 1
    let ah0 = [true, false, false, false];
    let ah1 = [false, false, true, false];

    let payload = AspxHfgenIwc2ChPayload {
        tna_mode: [&tna0, &tna1_ignored],
        add_harmonic: [&ah0, &ah1],
        ..Default::default()
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_2ch(
        &mut bw,
        &payload,
        true, // aspx_balance
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        true,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert_eq!(parsed.tna_mode[0], tna0.to_vec());
    assert_eq!(parsed.tna_mode[1], tna0.to_vec()); // mirrored
    assert!(parsed.ah_left);
    assert!(parsed.ah_right);
    assert_eq!(parsed.add_harmonic[0], ah0.to_vec());
    assert_eq!(parsed.add_harmonic[1], ah1.to_vec());
}

/// balance = 0: both channels' tna_mode written explicitly and distinct.
#[test]
fn hfgen_2ch_no_balance_distinct_tna() {
    let num_sbg_noise = 3;
    let num_sbg_sig_highres = 3;
    let num_aspx_timeslots = 3;

    let tna0 = [0u8, 1, 2];
    let tna1 = [3u8, 2, 1];

    let payload = AspxHfgenIwc2ChPayload {
        tna_mode: [&tna0, &tna1],
        ..Default::default()
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_2ch(
        &mut bw,
        &payload,
        false,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        false,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert_eq!(parsed.tna_mode[0], tna0.to_vec());
    assert_eq!(parsed.tna_mode[1], tna1.to_vec());
}

/// TIC copy path: identical active patterns on both channels emit the
/// compact `aspx_tic_copy = 1` form and decode back identically.
#[test]
fn hfgen_2ch_tic_copy_path() {
    let num_sbg_noise = 2;
    let num_sbg_sig_highres = 2;
    let num_aspx_timeslots = 4;

    let tic = [true, false, true, false];

    let payload = AspxHfgenIwc2ChPayload {
        tic_used_in_slot: [&tic, &tic],
        ..Default::default()
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_2ch(
        &mut bw,
        &payload,
        true,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        true,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert!(parsed.tic_present);
    assert!(parsed.tic_copy);
    assert_eq!(parsed.tic_used_in_slot[0], tic.to_vec());
    assert_eq!(parsed.tic_used_in_slot[1], tic.to_vec());
}

/// TIC distinct path: different patterns flip tic_copy off and gate each
/// channel separately, including the right-only case.
#[test]
fn hfgen_2ch_tic_distinct_and_right_only() {
    let num_sbg_noise = 2;
    let num_sbg_sig_highres = 2;
    let num_aspx_timeslots = 4;

    // Left empty, right active → tic_present=1, tic_copy=0, tic_left=0,
    // tic_right=1, only channel-1 vector written.
    let tic_left: [bool; 4] = [false, false, false, false];
    let tic_right = [false, true, true, false];

    let payload = AspxHfgenIwc2ChPayload {
        tic_used_in_slot: [&tic_left, &tic_right],
        ..Default::default()
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_2ch(
        &mut bw,
        &payload,
        true,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        true,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert!(parsed.tic_present);
    assert!(!parsed.tic_copy);
    assert!(!parsed.tic_left);
    assert!(parsed.tic_right);
    assert_eq!(parsed.tic_used_in_slot[0], vec![false; 4]);
    assert_eq!(parsed.tic_used_in_slot[1], tic_right.to_vec());
}

/// Full 2ch stress: distinct tna (balance=0), per-channel harmonics,
/// per-channel FIC, distinct TIC — every field survives the round-trip.
#[test]
fn hfgen_2ch_full_round_trip() {
    let num_sbg_noise = 4;
    let num_sbg_sig_highres = 5;
    let num_aspx_timeslots = 6;

    let tna0 = [1u8, 0, 3, 2];
    let tna1 = [2u8, 2, 1, 0];
    let ah0 = [true, false, true, false, false];
    let ah1 = [false, false, false, false, true];
    let fic0 = [false, true, false, false, false];
    let fic1 = [true, false, false, true, false];
    let tic0 = [true, false, false, false, false, true];
    let tic1 = [false, false, true, true, false, false];

    let payload = AspxHfgenIwc2ChPayload {
        tna_mode: [&tna0, &tna1],
        add_harmonic: [&ah0, &ah1],
        fic_used_in_sfb: [&fic0, &fic1],
        tic_used_in_slot: [&tic0, &tic1],
    };
    let mut bw = BitWriter::new();
    write_aspx_hfgen_iwc_2ch(
        &mut bw,
        &payload,
        false,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    );
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    let parsed = parse_aspx_hfgen_iwc_2ch(
        &mut br,
        false,
        num_sbg_noise,
        num_sbg_sig_highres,
        num_aspx_timeslots,
    )
    .unwrap();

    assert_eq!(parsed.tna_mode[0], tna0.to_vec());
    assert_eq!(parsed.tna_mode[1], tna1.to_vec());
    assert_eq!(parsed.add_harmonic[0], ah0.to_vec());
    assert_eq!(parsed.add_harmonic[1], ah1.to_vec());
    assert!(parsed.fic_present);
    assert_eq!(parsed.fic_used_in_sfb[0], fic0.to_vec());
    assert_eq!(parsed.fic_used_in_sfb[1], fic1.to_vec());
    assert!(parsed.tic_present);
    assert!(!parsed.tic_copy);
    assert_eq!(parsed.tic_used_in_slot[0], tic0.to_vec());
    assert_eq!(parsed.tic_used_in_slot[1], tic1.to_vec());
}
