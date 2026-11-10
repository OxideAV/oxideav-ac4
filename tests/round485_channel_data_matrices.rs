//! Round 485 — §5.3.3 channel-data element matrices end to end.
//!
//! TS 103 190-1 §5.3.3.2 – §5.3.3.5: the tracks of a
//! `two/three/four/five_channel_data()` element are matrix-transformed
//! by the quartets of its own `chparam_info()` elements (Pseudocode 59)
//! — Table 178 / 179 select the `three_` / `five_channel_data()`
//! matrices by `chel_matsel`. The decoder applies them when it walks
//! the element; the SIMPLE / ASPX 5.X / 7.X writers now code the
//! `(L, R)` and `(Ls, Rs)` pairs of their `five_channel_data()` as M/S
//! (`ms_used`) wherever that compacts the band energy.
//!
//! Pinned: correlated pairs select M/S rows and the decode keeps
//! waveform parity (SNR against the input) on every channel, for
//! correlated and independent content alike.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const FRAMES: usize = 6;
const SETTLE: usize = 2;

fn tones(freqs: &[f32], frame: usize, amp: f32) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = (frame * N + i) as f32 / FS;
            let v: f32 = freqs
                .iter()
                .map(|&f| (2.0 * std::f32::consts::PI * f * t).sin())
                .sum();
            amp * v / freqs.len() as f32
        })
        .collect()
}

/// 5.0 content; `correlated` makes R ≈ 0.9·L and Rs ≈ 0.8·Ls.
fn content(frame: usize, correlated: bool) -> Vec<Vec<f32>> {
    let l = tones(&[220.0, 1_300.0, 3_100.0], frame, 0.4);
    let c = tones(&[330.0, 2_000.0], frame, 0.3);
    let ls = tones(&[180.0, 900.0, 4_200.0], frame, 0.3);
    let (r, rs) = if correlated {
        let rn = tones(&[640.0], frame, 0.01);
        let r: Vec<f32> = l.iter().zip(&rn).map(|(a, b)| 0.9 * a + b).collect();
        let rs: Vec<f32> = ls.iter().map(|a| 0.8 * a).collect();
        (r, rs)
    } else {
        (
            tones(&[260.0, 1_700.0, 3_700.0], frame, 0.35),
            tones(&[200.0, 1_100.0, 4_900.0], frame, 0.28),
        )
    };
    vec![l, r, c, ls, rs]
}

fn snr_db(inp: &[f32], out: &[f32]) -> f32 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (x, y) in inp.iter().zip(out) {
        sig += (*x as f64).powi(2);
        err += (*y as f64 - *x as f64).powi(2);
    }
    (10.0 * (sig / err.max(1e-20)).log10()) as f32
}

/// Encode + decode `FRAMES` 5.0 SIMPLE frames; returns the per-channel
/// settled SNR (output aligned by one frame — the MDCT overlap delay)
/// and the `sap_mode`s of the five `chparam_info()` elements seen.
fn run(correlated: bool) -> (Vec<f32>, [u32; 5]) {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new();
    let mut dec = Ac4Decoder::new(&params);
    let mut inp = vec![Vec::new(); 5];
    let mut out = vec![Vec::new(); 5];
    let mut modes = [0u32; 5];
    for frame in 0..FRAMES {
        let pcm = content(frame, correlated);
        let f: [&[f32]; 5] = std::array::from_fn(|c| pcm[c].as_slice());
        let bytes = enc.encode_frame_pcm_5_0(&f);
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("audio frame expected");
        };
        let five = dec
            .last_substream
            .as_ref()
            .and_then(|s| s.tools.five_channel_data.as_ref())
            .and_then(|f| f.info.as_ref())
            .expect("five_channel_info parsed");
        assert_eq!(five.chel_matsel, 0);
        for (m, cp) in modes.iter_mut().zip(five.chparam.iter()) {
            *m = (*m).max(cp.sap_mode);
        }
        let raw = &af.data[0];
        for (i, c) in raw.as_chunks::<2>().0.iter().enumerate() {
            out[i % 5].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
        }
        for c in 0..5 {
            inp[c].extend_from_slice(&pcm[c]);
        }
    }
    let snrs = (0..5)
        .map(|c| {
            // The MDCT/IMDCT pair delays the output by one frame.
            let o = &out[c][SETTLE * N..];
            let i = &inp[c][(SETTLE - 1) * N..(FRAMES - 1) * N];
            snr_db(i, o)
        })
        .collect();
    (snrs, modes)
}

#[test]
fn correlated_pairs_code_ms_and_decode_with_parity() {
    let (snrs, modes) = run(true);
    eprintln!("correlated: modes {modes:?} snr {snrs:?}");
    assert_ne!(modes[0], 0, "(L, R) pair must carry M/S rows");
    assert_ne!(modes[2], 0, "(Ls, Rs) pair must carry M/S rows");
    assert_eq!([modes[1], modes[3], modes[4]], [0, 0, 0]);
    for (c, s) in snrs.iter().enumerate() {
        assert!(*s > 20.0, "channel {c}: SNR {s:.1} dB");
    }
}

/// Independent pairs: whatever bands the per-band decision codes as
/// M/S (an occasional leakage band can compact), the decode keeps
/// parity and never flips a pair to the all-bands M/S mode.
#[test]
fn independent_pairs_decode_with_parity() {
    let (snrs, modes) = run(false);
    eprintln!("independent: modes {modes:?} snr {snrs:?}");
    for (c, s) in snrs.iter().enumerate() {
        assert!(*s > 20.0, "channel {c}: SNR {s:.1} dB");
    }
    assert!(modes.iter().all(|&m| m <= 1), "no all-bands M/S row");
}
