//! Whole-packet decode: the first byte selects the §4.7 decoding mode
//! and the user dialogue-enhancement gain, the rest is fed through
//! `Ac4Decoder::send_packet` / `receive_frame` as a sequence of up to
//! four packets split at NUL-free boundaries (exercises the TOC walk,
//! every substream route, the sticky P-frame state and the PCM
//! synthesis). Errors are fine; panics are findings.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_ac4::decoder::{Ac4Decoder, DecodingMode};
use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > 1 << 16 {
        return;
    }
    let ctl = data[0];
    let body = &data[1..];
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    if ctl & 1 != 0 {
        dec.set_decoding_mode(DecodingMode::Core);
    }
    if ctl & 2 != 0 {
        dec.set_dialogue_enhancement_gain_db(f32::from(ctl >> 2));
    }
    // Split into up to four packets so P-frames follow I-frames.
    let n_pkts = 1 + usize::from((ctl >> 4) & 3);
    let chunk = body.len().div_ceil(n_pkts).max(1);
    for (i, piece) in body.chunks(chunk).enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), piece.to_vec());
        if dec.send_packet(&pkt).is_err() {
            break;
        }
        let _ = dec.receive_frame();
        if i >= 3 {
            break;
        }
    }
    let _ = dec.flush();
    let _ = dec.receive_frame();
});
