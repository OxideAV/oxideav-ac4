//! Round 485 — the LFE (`mono_data(1)`, never A-SPX / A-CPL processed,
//! §6.2.10 / §6.2.11) comes out time-aligned with the main channels on
//! every channel-based 5.1 / 7.1 route: the decoder delays it by the
//! QMF round trips the other channels took (one per A-SPX pass, one per
//! A-CPL pass).
//!
//! Measured with band-limited noise (unambiguous cross-correlation
//! peaks) — the lag of the LFE must match the lag of the centre.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;

/// Moving-average-filtered xorshift noise (`avg` taps), frame `frame`.
fn noise(seed: u32, frame: usize, amp: f32, avg: usize) -> Vec<f32> {
    let total = (frame + 1) * N + avg;
    let mut x = seed.wrapping_mul(2_654_435_761).max(1);
    let raw: Vec<f32> = (0..total)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 2001) as f32 / 1000.0 - 1.0
        })
        .collect();
    (frame * N..(frame + 1) * N)
        .map(|i| amp * raw[i..i + avg].iter().sum::<f32>() / (avg as f32).sqrt())
        .collect()
}

fn lag(inp: &[f32], out: &[f32]) -> usize {
    let mut best = (0usize, f64::MIN);
    for l in 0..3 * N {
        let mut acc = 0.0f64;
        for k in l..out.len().min(inp.len() + l) {
            acc += inp[k - l] as f64 * out[k] as f64;
        }
        if acc > best.1 {
            best = (l, acc);
        }
    }
    best.0
}

/// `(centre lag, LFE lag)` of a 6- or 8-channel route.
fn probe<F>(ch: usize, mut enc_frame: F) -> (usize, usize)
where
    F: FnMut(&mut Ac4ImsEncoder, &[Vec<f32>]) -> Vec<u8>,
{
    let lfe = ch - 1;
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&CodecParameters::audio(CodecId::new("ac4")));
    let mut inp = vec![Vec::new(); ch];
    let mut out = vec![Vec::new(); ch];
    for fr in 0..6 {
        let pcm: Vec<Vec<f32>> = (0..ch)
            .map(|c| {
                if c == lfe {
                    noise(77, fr, 0.006, 256)
                } else {
                    noise(c as u32 + 1, fr, 0.02, 8)
                }
            })
            .collect();
        let bytes = enc_frame(&mut enc, &pcm);
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 48_000), bytes))
            .expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("audio frame expected");
        };
        for (i, c) in af.data[0].as_chunks::<2>().0.iter().enumerate() {
            out[i % ch].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
        }
        for c in 0..ch {
            inp[c].extend_from_slice(&pcm[c]);
        }
    }
    (lag(&inp[2], &out[2]), lag(&inp[lfe], &out[lfe]))
}

fn assert_aligned(label: &str, (c, l): (usize, usize)) {
    eprintln!("{label}: centre lag {c}, LFE lag {l}");
    assert!(c.abs_diff(l) <= 2, "{label}: LFE lag {l} vs centre lag {c}");
}

#[test]
fn lfe_aligned_on_5_1_routes() {
    assert_aligned(
        "5.1 SIMPLE",
        probe(6, |e, p| {
            e.encode_frame_pcm_5_1(&std::array::from_fn(|c| p[c].as_slice()))
        }),
    );
    assert_aligned(
        "5.1 ASPX_ACPL_3",
        probe(6, |e, p| {
            e.encode_frame_pcm_5_1_acpl3_real_aspx(
                &std::array::from_fn(|c| p[c].as_slice()),
                1.0,
                1.0,
                1.0,
                1.0,
            )
        }),
    );
    assert_aligned(
        "5.1 ASPX_ACPL_2",
        probe(6, |e, p| {
            e.encode_frame_pcm_5_1_acpl2_real_aspx(&std::array::from_fn(|c| p[c].as_slice()))
        }),
    );
    assert_aligned(
        "5.1 ASPX_ACPL_1",
        probe(6, |e, p| {
            e.encode_frame_pcm_5_1_acpl1_real_aspx(&std::array::from_fn(|c| p[c].as_slice()))
        }),
    );
}

#[test]
fn lfe_aligned_on_7_1_routes() {
    assert_aligned(
        "7.1 SIMPLE",
        probe(8, |e, p| {
            e.encode_frame_pcm_7_1(&std::array::from_fn(|c| p[c].as_slice()))
        }),
    );
    assert_aligned(
        "7.1 ASPX_ACPL_1",
        probe(8, |e, p| {
            e.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&std::array::from_fn(|c| p[c].as_slice()))
        }),
    );
    assert_aligned(
        "7.1 ASPX_ACPL_2",
        probe(8, |e, p| {
            e.encode_frame_pcm_7_1_acpl2_real_aspx(&std::array::from_fn(|c| p[c].as_slice()))
        }),
    );
}
