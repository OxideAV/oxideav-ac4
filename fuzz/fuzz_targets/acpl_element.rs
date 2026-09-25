//! 5.X / 7.X A-CPL element decode under corruption: the leading bytes
//! pick the layout (5.0 / 5.1 / 7.0 / 7.1), the codec mode
//! (ASPX_ACPL_1 / ASPX_ACPL_2, or ASPX_ACPL_3 on 5.X), the I / P
//! framing and up to four byte
//! mutations; the rest is S16 PCM that `Ac4ImsEncoder` turns into two
//! raw frames (I then I or P). The mutations are applied to the coded
//! bytes before `Ac4Decoder` walks them, so the reader lands deep in
//! the Table 181 / 184 track resolution, the ACPL_1 residual / SAP
//! path, the captured A-SPX trailers and the Pseudocode 117 / 120
//! synthesis on *nearly valid* input instead of random bytes. Decode
//! errors are fine; panics are findings.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};

const N: usize = 1920;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let ctl = &data[..10];
    let pcm = &data[10..];
    let channels: usize = match ctl[0] & 3 {
        0 => 5,
        1 => 6,
        2 => 7,
        _ => 8,
    };
    let acpl1 = ctl[0] & 4 != 0;
    let second_is_p = ctl[0] & 8 != 0;
    // ASPX_ACPL_3 (5.X only): one stereo downmix + the eleven Table 62
    // rows, with the Pseudocode 109 previous-set interpolation state.
    let acpl3 = channels <= 6 && ctl[0] & 16 != 0;
    let max_sfb = 8 + u32::from(ctl[1] % 48);
    let max_sfb_master = 1 + u32::from(ctl[2] % 40);
    // Cheap deterministic PCM from the fuzz bytes: each channel is a
    // tone whose frequency / level come from the input, plus the raw
    // S16 samples where the input is long enough.
    let mut chans: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for c in 0..channels {
        let f = 40.0 + 90.0 * f32::from(ctl[3 + (c % 6)]);
        let amp = 0.05 + 0.5 * f32::from(ctl[9]) / 255.0;
        let mut v: Vec<f32> = (0..2 * N)
            .map(|i| amp * (2.0 * std::f32::consts::PI * f * i as f32 / 48_000.0).sin())
            .collect();
        for (i, s) in pcm.chunks_exact(2).enumerate().take(2 * N) {
            let x = i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0;
            v[i] = 0.5 * (v[i] + x);
        }
        chans.push(v);
    }
    let mut enc = Ac4ImsEncoder::new();
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(2);
    for k in 0..2 {
        enc.b_iframe_global = k == 0 || !second_is_p;
        let slice: Vec<&[f32]> = chans.iter().map(|c| &c[k * N..(k + 1) * N]).collect();
        let bytes = match (channels, acpl1) {
            (5, _) if acpl3 => enc.encode_frame_pcm_5_0_acpl3_real_aspx_with_max_sfb(
                &[slice[0], slice[1], slice[2], slice[3], slice[4]],
                max_sfb,
            ),
            (6, _) if acpl3 => enc.encode_frame_pcm_5_1_acpl3_real_aspx_with_max_sfb(
                &[slice[0], slice[1], slice[2], slice[3], slice[4], slice[5]],
                max_sfb,
                7,
            ),
            (5, false) => enc.encode_frame_pcm_5_0_acpl2_real_aspx_with_max_sfb(
                &[slice[0], slice[1], slice[2], slice[3], slice[4]],
                max_sfb,
            ),
            (5, true) => enc.encode_frame_pcm_5_0_acpl1_real_aspx_with_max_sfb(
                &[slice[0], slice[1], slice[2], slice[3], slice[4]],
                max_sfb,
                max_sfb_master,
            ),
            (6, false) => enc.encode_frame_pcm_5_1_acpl2_real_aspx_with_max_sfb(
                &[slice[0], slice[1], slice[2], slice[3], slice[4], slice[5]],
                max_sfb,
                7,
            ),
            (6, true) => enc.encode_frame_pcm_5_1_acpl1_real_aspx_with_max_sfb(
                &[slice[0], slice[1], slice[2], slice[3], slice[4], slice[5]],
                max_sfb,
                max_sfb_master,
                7,
            ),
            (7, false) => enc.encode_frame_pcm_7_0_acpl2_real_aspx_with_max_sfb(
                &[
                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6],
                ],
                max_sfb,
            ),
            (7, true) => enc.encode_frame_pcm_7_0_acpl1_real_alpha_beta_with_max_sfb(
                &[
                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6],
                ],
                max_sfb,
                max_sfb_master,
            ),
            (_, false) => enc.encode_frame_pcm_7_1_acpl2_real_aspx_with_max_sfb(
                &[
                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
                ],
                max_sfb,
                7,
            ),
            (_, true) => enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta_with_max_sfb(
                &[
                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
                ],
                max_sfb,
                max_sfb_master,
                7,
            ),
        };
        frames.push(bytes);
    }
    // Up to four byte mutations per frame, positions / values from the
    // PCM tail so the fuzzer can steer them.
    let tail = &data[data.len().saturating_sub(16)..];
    for (k, frame) in frames.iter_mut().enumerate() {
        if frame.is_empty() {
            continue;
        }
        for m in 0..4usize {
            let sel = tail.get(4 * k + m).copied().unwrap_or(0);
            if sel & 1 == 0 {
                continue;
            }
            let pos_src = tail.get(8 + ((4 * k + m) % 8)).copied().unwrap_or(0);
            let pos = (usize::from(pos_src) * 257 + usize::from(sel)) % frame.len();
            frame[pos] ^= sel.rotate_left(u32::from(m as u8));
        }
    }
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut dec = Ac4Decoder::new(&params);
    for bytes in frames {
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        if dec.send_packet(&pkt).is_err() {
            continue;
        }
        let _ = dec.receive_frame();
    }
    let _ = dec.flush();
    let _ = dec.receive_frame();
});
