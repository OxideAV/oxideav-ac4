//! Round 453 — dialogue-enhancement **application** on the
//! immersive-channel-element route (TS 103 190-2 §4.8.3.15 Tables
//! 13/15 + ETSI TS 103 190-1 §5.7.8): the substream's post-audio
//! `metadata(…, sus_ver = 1)` carries a `dialog_enhancement()`
//! payload, and a user gain G_DE boosts the Table 15 dialogue
//! channels in both §4.7 decoding modes while every other channel
//! passes through.

use oxideav_ac4::de::{DeConfig, DeData, DeMethod, DialogEnhancement, DE_NR_BANDS};
use oxideav_ac4::decoder::{Ac4Decoder, DecodingMode};
use oxideav_ac4::ice::{encode_ice_raw_frame_with_metadata, write_ice_body_scpl, IceScplSpectra};
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const TL: u32 = 1920;
const N: usize = 1920;
const MAX_SFB: u32 = 20;

fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
    let sfbo = oxideav_ac4::sfb_offset::sfb_offset_48(TL).unwrap();
    let end = sfbo[MAX_SFB as usize] as usize;
    let mut v = vec![0.0f32; end];
    v[bin.min(end - 1)] = amp;
    v
}

/// Minimal v2 substream metadata carrying one `dialog_enhancement()`
/// element: `basic_metadata` (b_more = 0), `extended_metadata`
/// (b_dialog / b_channels_classifier / b_event_probability = 0), the
/// exact `tools_metadata_size`, the DE payload, `b_emdf = 0`.
fn de_metadata_bytes(b_iframe: bool, de: &DialogEnhancement) -> Vec<u8> {
    let mut dw = BitWriter::new();
    oxideav_ac4::de::write_dialog_enhancement(&mut dw, de, b_iframe).expect("write DE");
    let de_bits = dw.bit_position() as u32;
    let mut bw = BitWriter::new();
    bw.write_bit(false); // basic_metadata: b_more_basic_metadata
    bw.write_bit(false); // extended_metadata: b_dialog
    bw.write_bit(false); // b_channels_classifier
    bw.write_bit(false); // b_event_probability
    bw.write_u32(de_bits & 0x7F, 7); // tools_metadata_size (low bits)
    if de_bits >= 128 {
        bw.write_bit(true); // b_more_bits
        oxideav_ac4::toc::write_variable_bits(&mut bw, 3, de_bits >> 7);
    } else {
        bw.write_bit(false); // b_more_bits
    }
    let bytes = dw.into_bytes();
    let mut br = BitReader::new(&bytes);
    for _ in 0..de_bits {
        bw.write_bit(br.read_bit().unwrap());
    }
    bw.write_bit(false); // b_emdf_payloads_substream
    bw.align_to_byte();
    bw.into_bytes()
}

fn de_element() -> DialogEnhancement {
    // Channel-independent, L+R+C, p = 1,0 in every band, Gmax = 12 dB.
    let cfg = DeConfig {
        method: DeMethod::ChannelIndependent,
        max_gain: 3,
        channel_config: 0b111,
    };
    let data = DeData {
        keep_pos_flag: false,
        mix_coef1_idx: None,
        mix_coef2_idx: None,
        keep_data_flag: false,
        ms_proc_flag: false,
        de_par: vec![[10i32; DE_NR_BANDS]; 3],
        signal_contribution: None,
    };
    DialogEnhancement {
        data_present: true,
        config_flag: true,
        config: Some(cfg),
        data: Some(data),
    }
}

/// One 7.0.4 SCPL frame (identity SAP, distinct tones on every track)
/// plus the DE metadata.
fn scpl_frame(seq: u32, b_iframe: bool) -> Vec<u8> {
    let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 25.0)).collect();
    let core_refs: [&[f32]; 5] = std::array::from_fn(|i| core[i].as_slice());
    let fg = [tone_spectrum(30, 22.0), tone_spectrum(34, 22.0)];
    let hi = [tone_spectrum(40, 20.0), tone_spectrum(44, 20.0)];
    let jk = [tone_spectrum(50, 18.0), tone_spectrum(54, 18.0)];
    let scpl_pairs: Vec<[&[f32]; 2]> = vec![
        [hi[0].as_slice(), hi[1].as_slice()],
        [jk[0].as_slice(), jk[1].as_slice()],
    ];
    let spectra = IceScplSpectra {
        core: &core_refs,
        add_pair: [&fg[0], &fg[1]],
        scpl_pairs: &scpl_pairs,
    };
    let mut body = BitWriter::new();
    write_ice_body_scpl(&mut body, &spectra, None, false, TL, MAX_SFB).expect("scpl body");
    let meta = de_metadata_bytes(b_iframe, &de_element());
    encode_ice_raw_frame_with_metadata(seq, false, false, b_iframe, body, &meta).expect("frame")
}

fn decode_gop_energies(mode: DecodingMode, gain_db: f32) -> Vec<Vec<f64>> {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    dec.set_decoding_mode(mode);
    dec.set_dialogue_enhancement_gain_db(gain_db);
    let channels = if mode == DecodingMode::Core { 7 } else { 11 };
    let mut out = Vec::new();
    for (seq, iframe) in [(0u32, true), (1, false), (2, false), (3, false)] {
        let frame = scpl_frame(seq, iframe);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        assert_eq!(af.samples, N as u32);
        let buf = &af.data[0];
        assert_eq!(buf.len(), N * channels * 2);
        let mut e = vec![0.0f64; channels];
        for i in 0..N {
            for (c, slot) in e.iter_mut().enumerate() {
                let off = (i * channels + c) * 2;
                let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as f64;
                *slot += s * s;
            }
        }
        out.push(e);
    }
    out
}

/// G_DE = 6 dB on a 7.0.4 SCPL stream boosts L / R / C by
/// (1 + g)² ≈ 3,98× in energy and leaves the other channels alone
/// (full decoding; Table 15 row "5.X, 7.X, 7.X.4").
#[test]
fn ice_full_decoding_de_boosts_front_channels_only() {
    let base = decode_gop_energies(DecodingMode::Full, 0.0);
    let boosted = decode_gop_energies(DecodingMode::Full, 6.0);
    let last_b = boosted.last().unwrap();
    let last_0 = base.last().unwrap();
    let g = 10f64.powf(6.0 / 20.0);
    let expect = g * g;
    for slot in [0usize, 1, 2] {
        let ratio = last_b[slot] / last_0[slot].max(1e-9);
        assert!(
            (expect * 0.8..=expect * 1.25).contains(&ratio),
            "front slot {slot}: boost ratio {ratio}, expected ≈ {expect}"
        );
    }
    for slot in 3..11 {
        let ratio = last_b[slot] / last_0[slot].max(1e-9);
        assert!(
            (0.8..=1.25).contains(&ratio),
            "slot {slot} must pass through: ratio {ratio}"
        );
    }
}

/// The same stream in core decoding mode: the tool applies on the
/// 7-channel core roster's L / R / C.
#[test]
fn ice_core_decoding_de_boosts_front_channels_only() {
    let base = decode_gop_energies(DecodingMode::Core, 0.0);
    let boosted = decode_gop_energies(DecodingMode::Core, 6.0);
    let last_b = boosted.last().unwrap();
    let last_0 = base.last().unwrap();
    let g = 10f64.powf(6.0 / 20.0);
    let expect = g * g;
    for slot in [0usize, 1, 2] {
        let ratio = last_b[slot] / last_0[slot].max(1e-9);
        assert!(
            (expect * 0.8..=expect * 1.25).contains(&ratio),
            "front slot {slot}: boost ratio {ratio}, expected ≈ {expect}"
        );
    }
    for slot in 3..7 {
        let ratio = last_b[slot] / last_0[slot].max(1e-9);
        assert!(
            (0.8..=1.25).contains(&ratio),
            "slot {slot} must pass through: ratio {ratio}"
        );
    }
}

/// The Gmax clamp (§4.3.14.3.2): asking for 40 dB against
/// Gmax = 12 dB caps the boost at (10^(12/20))² energy.
#[test]
fn ice_de_gain_clamps_to_bitstream_gmax() {
    let base = decode_gop_energies(DecodingMode::Full, 0.0);
    let boosted = decode_gop_energies(DecodingMode::Full, 40.0);
    let g = 10f64.powf(12.0 / 20.0);
    let expect = g * g;
    let ratio = boosted.last().unwrap()[2] / base.last().unwrap()[2].max(1e-9);
    assert!(
        (expect * 0.8..=expect * 1.25).contains(&ratio),
        "centre boost ratio {ratio}, expected ≈ {expect} (Gmax clamp)"
    );
}
