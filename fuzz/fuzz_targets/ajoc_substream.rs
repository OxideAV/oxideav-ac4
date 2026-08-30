//! A-JOC object substream body (TS 103 190-2 §6.2.2.2 / §6.2.3.4)
//! straight into `AjocSubstreamDecoder` with a fuzz-chosen descriptor
//! (signal counts, LFE, static-downmix form, I/P flag, decoding mode)
//! — the parse, the §5.7.3 reconstruction and both §4.7 modes.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_ac4::ajoc_substream::{AjocBodyParams, AjocSubstreamDecoder};
use oxideav_ac4::oamd::ObjType;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > 1 << 15 {
        return;
    }
    let a = data[0];
    let b = data[1];
    let body = &data[2..];
    let b_static_dmx = a & 1 != 0;
    let b_lfe = a & 2 != 0;
    let num_dmx: u32 = if b_static_dmx { 5 } else { 1 + u32::from((a >> 2) & 7) };
    let num_umx: u32 = 1 + u32::from(b & 15);
    let b_iframe = b & 16 != 0;
    let core = b & 32 != 0;
    let params = AjocBodyParams {
        b_lfe,
        b_static_dmx,
        n_fullband_dmx_signals: num_dmx,
        n_fullband_upmix_signals: num_umx,
        obj_type_dmx: vec![ObjType::Dyn; (num_dmx + u32::from(b_lfe)) as usize],
        obj_type_umx: vec![ObjType::Dyn; (num_umx + u32::from(b_lfe)) as usize],
    };
    let mut dec = AjocSubstreamDecoder::new(num_dmx as usize, num_umx as usize);
    // Two passes: the first primes the sticky / differential state.
    for pass in 0..2 {
        let iframe = b_iframe || pass == 0;
        let _ = if core {
            dec.decode_substream_pcm_core(body, &params, iframe, false, 1920)
        } else {
            dec.decode_substream_pcm(body, &params, iframe, false, 1920)
        };
    }
});
