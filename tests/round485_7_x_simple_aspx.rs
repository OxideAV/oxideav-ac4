//! Round 485 — 7_X SIMPLE / ASPX element render for every channel
//! layout (TS 103 190-1 §5.3.4.4.1 Tables 182 / 183, §6.2.10
//! Table 213).
//!
//! The decoder resolves the core tracks of any coding_config, mixes the
//! additional `two_channel_data()` through Table 183 and — in `ASPX`
//! mode — extends all seven channels with the four captured A-SPX
//! elements in the Table 213 order of the stream's channel mode
//! (`(L, R), (Ls, Rs), C, (Lb, Rb)` / `(L, R), (Lw, Rw), C, (Ls, Rs)` /
//! `(L, R), (Ls, Rs), C, (Tfl, Tfr)`). Pinned: waveform parity of the
//! low band on every channel of every layout, and A-SPX-regenerated
//! high band on the additional channels (whose fourth A-SPX element
//! was previously parsed and dropped).

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::toc::SevenXLayout;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const N: usize = 1920;
const FS: f32 = 48_000.0;
const FRAMES: usize = 8;
const SETTLE: usize = 4;

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

/// Seven channels of low-band tones; `hf` adds a 14 kHz tone to the
/// additional channels (slots 5 / 6) — above any A-SPX crossover.
fn content(frame: usize, hf: bool) -> Vec<Vec<f32>> {
    let sets: [(&[f32], f32); 7] = [
        (&[180.0, 1_410.0], 0.4),
        (&[230.0, 1_520.0], 0.35),
        (&[300.0, 1_610.0], 0.3),
        (&[160.0, 1_370.0], 0.3),
        (&[210.0, 1_450.0], 0.28),
        (&[140.0, 1_330.0], 0.22),
        (&[250.0, 1_470.0], 0.2),
    ];
    (0..7)
        .map(|c| {
            let mut v = tones(sets[c].0, frame, sets[c].1);
            if hf && c >= 5 {
                for (x, h) in v.iter_mut().zip(tones(&[14_000.0], frame, 0.1)) {
                    *x += h;
                }
            }
            v
        })
        .collect()
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

fn best_lag(inp: &[f32], out: &[f32]) -> usize {
    let mut best = (0usize, f64::MIN);
    for lag in 0..2 * N {
        let mut acc = 0.0f64;
        for k in lag..out.len().min(inp.len() + lag) {
            acc += inp[k - lag] as f64 * out[k] as f64;
        }
        if acc > best.1 {
            best = (lag, acc);
        }
    }
    best.0
}

/// Energy of `x` in the 12–16 kHz band (direct DFT over 1920-sample
/// blocks, 25 Hz bins) — the A-SPX-regenerated range.
fn hf_energy(x: &[f32]) -> f64 {
    let mut e = 0.0f64;
    for block in x.as_chunks::<N>().0 {
        for bin in (12_000 / 25)..(16_000 / 25) {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (k, &v) in block.iter().enumerate() {
                let ph = 2.0 * std::f64::consts::PI * (bin * k) as f64 / N as f64;
                re += v as f64 * ph.cos();
                im += v as f64 * ph.sin();
            }
            e += re * re + im * im;
        }
    }
    e
}

fn run(layout: SevenXLayout, aspx: bool, hf: bool) -> (Vec<f32>, Vec<f32>) {
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new().with_seven_x_layout(layout);
    let mut dec = Ac4Decoder::new(&params);
    let mut inp = vec![Vec::new(); 7];
    let mut out = vec![Vec::new(); 7];
    for frame in 0..FRAMES {
        let pcm = content(frame, hf);
        enc.b_iframe_global = frame % 4 == 0;
        let f: [&[f32]; 7] = std::array::from_fn(|c| pcm[c].as_slice());
        let bytes = if aspx {
            enc.encode_frame_pcm_7_0_aspx_real_aspx(&f)
        } else {
            enc.encode_frame_pcm_7_0(&f)
        };
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("audio frame expected");
        };
        for (i, c) in af.data[0].as_chunks::<2>().0.iter().enumerate() {
            out[i % 7].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
        }
        for c in 0..7 {
            inp[c].extend_from_slice(&pcm[c]);
        }
    }
    let lag = best_lag(&inp[2], &out[2]);
    let start = SETTLE * N;
    let mut snrs = Vec::new();
    let mut hf_ratio = Vec::new();
    for c in 0..7 {
        let o = &out[c][start..];
        let i = &inp[c][start - lag..inp[c].len() - lag];
        snrs.push(snr_db(i, o));
        if hf && c >= 5 {
            let ei = hf_energy(i);
            hf_ratio.push((hf_energy(o) / ei.max(1e-30)) as f32);
        } else {
            hf_ratio.push(0.0);
        }
    }
    eprintln!("{layout:?} aspx={aspx} hf={hf} lag {lag}: snr {snrs:.1?} hf {hf_ratio:.2?}");
    (snrs, hf_ratio)
}

const LAYOUTS: [SevenXLayout; 3] = [
    SevenXLayout::BackSurround,
    SevenXLayout::Wide { add_ch_base: false },
    SevenXLayout::TopFront { add_ch_base: true },
];

#[test]
fn simple_every_layout_is_waveform_exact() {
    for layout in LAYOUTS {
        let (snrs, _) = run(layout, false, false);
        for (c, s) in snrs.iter().enumerate() {
            assert!(*s > 20.0, "{layout:?} SIMPLE ch {c}: SNR {s:.1} dB");
        }
    }
}

#[test]
fn aspx_every_layout_keeps_low_band_parity() {
    for layout in LAYOUTS {
        let (snrs, _) = run(layout, true, false);
        for (c, s) in snrs.iter().enumerate() {
            assert!(*s > 15.0, "{layout:?} ASPX ch {c}: SNR {s:.1} dB");
        }
    }
}

/// The additional channels' high band is regenerated by their own
/// A-SPX element (the fourth for 3/4/0 and 3/2/2, the second for
/// 5/2/0 — Table 213).
#[test]
fn aspx_regenerates_the_additional_channels_high_band() {
    for layout in LAYOUTS {
        let (_, hf) = run(layout, true, true);
        for c in [5usize, 6] {
            assert!(
                (0.3..=3.0).contains(&hf[c]),
                "{layout:?} ch {c}: 12-16 kHz energy ratio {:.2}",
                hf[c]
            );
        }
    }
}

/// Additional channels correlated with their Table 183 base pair: the
/// SIMPLE writer codes them M/S (`b_use_sap_add_ch = 1`, `ms_used`
/// rows) and the decoder's Table 183 mix restores every channel — for
/// the (D, E) base of 3/4/0 and the (A, B) base of 5/2/0 / 3/2/2.
#[test]
fn simple_table_183_ms_pairs_decode_with_parity() {
    for layout in LAYOUTS {
        let base: [usize; 2] = if matches!(layout, SevenXLayout::BackSurround) {
            [3, 4]
        } else {
            [0, 1]
        };
        let params = CodecParameters::audio(CodecId::new("ac4"));
        let mut enc = Ac4ImsEncoder::new().with_seven_x_layout(layout);
        let mut dec = Ac4Decoder::new(&params);
        let mut inp = vec![Vec::new(); 7];
        let mut out = vec![Vec::new(); 7];
        let mut sap_seen = false;
        for frame in 0..FRAMES {
            let mut pcm = content(frame, false);
            for i in 0..2 {
                let own = tones(&[700.0 + 90.0 * i as f32], frame, 0.02);
                pcm[5 + i] = pcm[base[i]]
                    .iter()
                    .zip(&own)
                    .map(|(p, o)| 0.8 * p + o)
                    .collect();
            }
            let f: [&[f32]; 7] = std::array::from_fn(|c| pcm[c].as_slice());
            let bytes = enc.encode_frame_pcm_7_0(&f);
            let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
            dec.send_packet(&pkt).expect("send_packet");
            let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
                panic!("audio frame expected");
            };
            let tools = &dec.last_substream.as_ref().expect("substream").tools;
            sap_seen |= tools.seven_x_b_use_sap_add_ch == Some(true)
                && tools
                    .seven_x_add_chparam_info
                    .as_ref()
                    .is_some_and(|cp| cp.iter().all(|c| c.sap_mode != 0));
            for (i, c) in af.data[0].as_chunks::<2>().0.iter().enumerate() {
                out[i % 7].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
            }
            for c in 0..7 {
                inp[c].extend_from_slice(&pcm[c]);
            }
        }
        assert!(sap_seen, "{layout:?}: Table 183 pairs must be M/S-coded");
        let start = SETTLE * N;
        for c in 0..7 {
            let s = snr_db(&inp[c][start - N..inp[c].len() - N], &out[c][start..]);
            eprintln!("{layout:?} SIMPLE M/S ch {c}: SNR {s:.1} dB");
            assert!(s > 20.0, "{layout:?} SIMPLE M/S ch {c}: SNR {s:.1} dB");
        }
    }
}
