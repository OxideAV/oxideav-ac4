//! Round 485 — 7.X channel layouts beyond 3/4/0: 5/2/0.x (Lw / Rw)
//! and 3/2/2.x (Tfl / Tfr) with both `add_ch_base` values, through the
//! live ASPX_ACPL_1 / ASPX_ACPL_2 encoders and `Ac4Decoder`.
//!
//! Refs ETSI TS 103 190-1 Table 88 (`channel_mode`), §4.3.3.7.6
//! (`add_ch_base`), §5.3.4.4 Tables 183 / 185 (final channel mapping),
//! §5.7.7.6.3 Table 202 + Pseudocode 120 (A-CPL variable mapping and
//! output gains), §6.2.10 Table 213 (A-SPX channel order).
//!
//! Output slots are `[L, R, C, Ls, Rs, X1, X2(, LFE)]` with `(X1, X2)`
//! the layout's additional channels.

use oxideav_ac4::decoder::Ac4Decoder;
use oxideav_ac4::encoder_ims::Ac4ImsEncoder;
use oxideav_ac4::toc::{SevenXLayout, parse_ac4_toc};
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

/// Per-channel low-band multitones (below the ASPX_ACPL_1 waveform
/// crossover, so ACPL_1 is waveform-coded on every channel).
fn content(frame: usize, ch: usize) -> Vec<Vec<f32>> {
    let sets: [(&[f32], f32); 8] = [
        (&[180.0, 410.0], 0.4),
        (&[230.0, 520.0], 0.35),
        (&[300.0, 610.0], 0.3),
        (&[160.0, 370.0, 690.0], 0.3),
        (&[210.0, 450.0, 740.0], 0.28),
        (&[140.0, 330.0, 580.0], 0.22),
        (&[250.0, 470.0, 650.0], 0.2),
        (&[62.0], 0.1),
    ];
    (0..ch)
        .map(|c| tones(sets[c].0, frame, sets[c].1))
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
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

#[derive(Clone, Copy, Debug)]
enum Mode {
    Acpl1,
    Acpl2,
}

/// Returns `(ratios, snrs)` per channel over the settled window.
fn run(layout: SevenXLayout, mode: Mode, lfe: bool) -> (Vec<f32>, Vec<f32>) {
    let ch = if lfe { 8 } else { 7 };
    let params = CodecParameters::audio(CodecId::new("ac4"));
    let mut enc = Ac4ImsEncoder::new().with_seven_x_layout(layout);
    let mut dec = Ac4Decoder::new(&params);
    let mut inp = vec![Vec::new(); ch];
    let mut out = vec![Vec::new(); ch];
    for frame in 0..FRAMES {
        let pcm = content(frame, ch);
        enc.b_iframe_global = frame % 4 == 0;
        let bytes = match (mode, lfe) {
            (Mode::Acpl1, false) => {
                let f: [&[f32]; 7] = std::array::from_fn(|c| pcm[c].as_slice());
                enc.encode_frame_pcm_7_0_acpl1_real_alpha_beta(&f)
            }
            (Mode::Acpl1, true) => {
                let f: [&[f32]; 8] = std::array::from_fn(|c| pcm[c].as_slice());
                enc.encode_frame_pcm_7_1_acpl1_real_alpha_beta(&f)
            }
            (Mode::Acpl2, false) => {
                let f: [&[f32]; 7] = std::array::from_fn(|c| pcm[c].as_slice());
                enc.encode_frame_pcm_7_0_acpl2_real_aspx(&f)
            }
            (Mode::Acpl2, true) => {
                let f: [&[f32]; 8] = std::array::from_fn(|c| pcm[c].as_slice());
                enc.encode_frame_pcm_7_1_acpl2_real_aspx(&f)
            }
        };
        let info = parse_ac4_toc(&bytes).expect("toc");
        assert_eq!(info.seven_x_layout, Some(layout), "TOC carries the layout");
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(af) = dec.receive_frame().expect("receive_frame") else {
            panic!("audio frame expected");
        };
        let raw = &af.data[0];
        assert_eq!(raw.len(), N * ch * 2);
        for (i, c) in raw.as_chunks::<2>().0.iter().enumerate() {
            out[i % ch].push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0);
        }
        for c in 0..ch {
            inp[c].extend_from_slice(&pcm[c]);
        }
    }
    let lag = best_lag(&inp[2], &out[2]);
    let start = SETTLE * N;
    let mut ratios = Vec::new();
    let mut snrs = Vec::new();
    for c in 0..ch {
        let o = &out[c][start..];
        let i = &inp[c][start - lag..inp[c].len() - lag];
        ratios.push(rms(o) / rms(i));
        snrs.push(snr_db(i, o));
    }
    eprintln!("{layout:?} {mode:?} lfe={lfe} lag {lag}: ratios {ratios:.3?} snr {snrs:.1?}");
    (ratios, snrs)
}

const LAYOUTS: [SevenXLayout; 5] = [
    SevenXLayout::BackSurround,
    SevenXLayout::Wide { add_ch_base: false },
    SevenXLayout::Wide { add_ch_base: true },
    SevenXLayout::TopFront { add_ch_base: false },
    SevenXLayout::TopFront { add_ch_base: true },
];

/// ASPX_ACPL_1 is waveform-coded below `acpl_qmf_band`: on low-band
/// content every channel of every layout decodes with waveform parity.
#[test]
fn acpl1_every_layout_decodes_with_waveform_parity() {
    for layout in LAYOUTS {
        for lfe in [false, true] {
            let (ratios, snrs) = run(layout, Mode::Acpl1, lfe);
            for (c, (r, s)) in ratios.iter().zip(&snrs).enumerate() {
                assert!(
                    (0.9..=1.1).contains(r),
                    "{layout:?} lfe={lfe} ch {c}: ratio {r:.3}"
                );
                assert!(*s > 20.0, "{layout:?} lfe={lfe} ch {c}: SNR {s:.1} dB");
            }
        }
    }
}

/// ASPX_ACPL_2 reconstructs the additional channels parametrically:
/// every channel of every layout within the round-461 RMS window and
/// the waveform-coded channels (C, the non-coupled pair) with
/// waveform parity.
#[test]
fn acpl2_every_layout_decodes_with_parity() {
    for layout in LAYOUTS {
        let (ratios, snrs) = run(layout, Mode::Acpl2, false);
        for (c, r) in ratios.iter().enumerate() {
            assert!(
                (0.7..=1.15).contains(r),
                "{layout:?} ACPL_2 ch {c}: ratio {r:.3}"
            );
        }
        // Waveform pair per Table 202: L / R for 3/4/0.x and
        // add_ch_base == 1, Ls / Rs for add_ch_base == 0.
        let wave: [usize; 2] = if layout.add_ch_base() == Some(false) {
            [3, 4]
        } else {
            [0, 1]
        };
        for c in [2, wave[0], wave[1]] {
            assert!(
                snrs[c] > 20.0,
                "{layout:?} ACPL_2 waveform ch {c}: SNR {:.1} dB",
                snrs[c]
            );
        }
    }
}

#[test]
fn layout_round_trips_through_ch_mode_and_add_ch_base() {
    for (ch_mode, acb, want) in [
        (5, false, Some(SevenXLayout::BackSurround)),
        (6, true, Some(SevenXLayout::BackSurround)),
        (7, false, Some(SevenXLayout::Wide { add_ch_base: false })),
        (8, true, Some(SevenXLayout::Wide { add_ch_base: true })),
        (9, true, Some(SevenXLayout::TopFront { add_ch_base: true })),
        (
            10,
            false,
            Some(SevenXLayout::TopFront { add_ch_base: false }),
        ),
        (4, false, None),
        (11, false, None),
    ] {
        assert_eq!(SevenXLayout::from_ch_mode(ch_mode, acb), want);
    }
}
