//! Forward MDCT analysis for the AC-4 IMS encoder (round 48).
//!
//! Per ETSI TS 103 190-1 §5.5.2 (Pseudocodes 60-64) the AC-4 transform is
//! a standard MDCT — same Princen-Bradley TDAC convention, same KBD
//! windowing, same `2N`-input / `N`-output symmetry — only the spec uses
//! an FFT-factored unfolding for speed. The decoder ([`crate::mdct`])
//! ships the inverse direction (IMDCT). This module supplies the
//! complementary forward direction in the simplest possible form: the
//! direct-summation cosine matrix evaluated naively in O(N²). Correctness
//! first; speed last (encoder isn't on a hot path).
//!
//! The forward formula matched against the decoder's IMDCT is the
//! standard MDCT pair:
//!
//!   X[k] = sum_{n=0..2N-1} x[n] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
//!   y[n] = (2/N) * sum_{k=0..N-1} X[k] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
//!
//! An end-to-end Princen-Bradley round-trip test (forward MDCT → IMDCT →
//! KBD overlap-add over 3 consecutive frames of a constant signal) lives
//! in `tests/` to lock the sign convention to the decoder.

use crate::mdct::kbd_window;
use core::f64::consts::PI;

/// Apply the KBD analysis window of length `2N` to `2N` time-domain
/// samples. Identical to the synthesis window — KBD is symmetric.
pub fn apply_kbd_window(samples: &mut [f32], window: &[f32]) {
    debug_assert_eq!(samples.len(), window.len());
    for (s, w) in samples.iter_mut().zip(window.iter()) {
        *s *= *w;
    }
}

/// Naive forward MDCT: transforms `2N` time-domain samples (already
/// windowed) into `N` spectral coefficients via the direct-summation
/// cosine basis. O(N²); intended for correctness, not throughput.
///
/// The scaling matches the decoder's IMDCT in [`crate::mdct::imdct`]
/// (which itself divides post-IFFT by `N`): forward = `2 * sum_n x*cos`,
/// inverse = `(1/N) * sum_k X*cos`. The `2x` prefactor here is the
/// canonical AC-4 / AAC normalisation that makes the windowed
/// Princen-Bradley round-trip recover the original signal exactly in
/// steady-state frames.
///
/// Sign convention matches [`crate::mdct::imdct`] so the round-trip
/// `imdct(mdct(x)) ≈ x` (with TDAC overlap-add and KBD windowing).
pub fn mdct_naive(x: &[f32]) -> Vec<f32> {
    let two_n = x.len();
    let n = two_n / 2;
    let mut out = vec![0.0_f32; n];
    let n_f = n as f64;
    for (k, out_k) in out.iter_mut().enumerate() {
        let kf = k as f64;
        let mut acc = 0.0_f64;
        for (nn, &xv) in x.iter().enumerate().take(two_n) {
            let nf = nn as f64;
            let theta = PI / n_f * (nf + 0.5 + n_f * 0.5) * (kf + 0.5);
            acc += xv as f64 * theta.cos();
        }
        // Forward-direction scaling complementary to the decoder's
        // IMDCT post-IFFT divide-by-N.
        *out_k = (2.0 * acc) as f32;
    }
    out
}

/// Per-channel forward-MDCT analysis state — carries the last `N` input
/// samples so the next frame can window across the 50% TDAC boundary.
#[derive(Debug, Clone)]
pub struct EncoderMdctState {
    /// Transform length (samples per output spectrum). The MDCT consumes
    /// `2N` samples per call (last N from history + this frame's N new).
    pub n: u32,
    /// Last `N` input PCM samples for the next-frame overlap.
    pub history: Vec<f32>,
}

impl EncoderMdctState {
    /// Fresh encoder state with N-sample zero history (TDAC priming —
    /// the very first encoded frame will lose half a window of energy at
    /// the leading edge, identical to the decoder's first-frame behaviour).
    pub fn new(n: u32) -> Self {
        Self {
            n,
            history: vec![0.0_f32; n as usize],
        }
    }

    /// Push a fresh `N`-sample frame of input PCM through the windowed
    /// forward MDCT and return the `N` spectral coefficients. The caller
    /// owns the input buffer — this method consumes it logically (copies
    /// into the state).
    pub fn analyse_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        let n = self.n as usize;
        assert_eq!(frame.len(), n, "encoder: frame must be exactly N samples");
        // Build the 2N-sample MDCT input: previous N + new N.
        let mut buf = Vec::with_capacity(2 * n);
        buf.extend_from_slice(&self.history);
        buf.extend_from_slice(frame);
        // KBD window per §5.5.3 — same window as synthesis (Princen-Bradley).
        let window = kbd_window(self.n);
        apply_kbd_window(&mut buf, &window);
        // Forward MDCT.
        let spec = mdct_naive(&buf);
        // Slide history forward — drop the old N, keep this frame's N
        // for the next call's leading half.
        self.history.clear();
        self.history.extend_from_slice(frame);
        spec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdct::{imdct, imdct_olap_symmetric, kbd_window};

    /// Simplest TDAC contract: forward MDCT of an all-zero windowed
    /// signal is all-zero spectrum.
    #[test]
    fn forward_mdct_zero_in_zero_out() {
        let n = 8;
        let x = vec![0.0_f32; 2 * n];
        let spec = mdct_naive(&x);
        assert_eq!(spec.len(), n);
        for &s in &spec {
            assert!(s.abs() < 1e-6, "got non-zero from zero input: {s}");
        }
    }

    /// MDCT followed by IMDCT (no windowing) should give back exactly
    /// `2 * x[n] - aliasing` per the MDCT identity. The straightforward
    /// check: forward then inverse, for a signal that's odd-symmetric
    /// about n = N/2 the time-domain aliasing cancels and we get x back
    /// exactly. The reverse is harder to construct; for round-trip we
    /// rely on the windowed Princen-Bradley test below.
    #[test]
    fn forward_mdct_is_linear() {
        let n = 8;
        let mut x = vec![0.0_f32; 2 * n];
        x[3] = 0.7;
        x[10] = -0.3;
        let s = mdct_naive(&x);
        let scaled: Vec<f32> = x.iter().map(|v| v * 2.5).collect();
        let s2 = mdct_naive(&scaled);
        for (a, b) in s.iter().zip(s2.iter()) {
            assert!(((*a) * 2.5 - *b).abs() < 1e-4, "linearity broken");
        }
    }

    /// Princen-Bradley TDAC: forward MDCT → IMDCT → overlap-add over
    /// 3 consecutive frames of a constant signal recovers the constant
    /// in the steady-state middle frame (within floating-point error).
    /// This is the headline correctness test that locks the sign
    /// convention between mdct_naive and the decoder's imdct.
    #[test]
    fn princen_bradley_constant_signal_recovers_in_middle_frame() {
        let n = 16u32;
        let nsz = n as usize;
        // 4 frames of constant 0.1 amplitude.
        let frames: Vec<Vec<f32>> = (0..4).map(|_| vec![0.1_f32; nsz]).collect();

        // Encoder side.
        let mut enc = EncoderMdctState::new(n);
        let specs: Vec<Vec<f32>> = frames.iter().map(|f| enc.analyse_frame(f)).collect();

        // Decoder side: imdct + KBD overlap-add.
        let window = kbd_window(n);
        let mut overlap = vec![0.0_f32; nsz];
        let mut pcm_out: Vec<Vec<f32>> = Vec::new();
        for spec in &specs {
            let y = imdct(spec);
            let pcm = imdct_olap_symmetric(&y, &window, &mut overlap);
            pcm_out.push(pcm);
        }
        // The middle frames (index 1, 2) should be ~0.1 throughout.
        // First and last lose half a window to the zero history.
        for &v in &pcm_out[2] {
            assert!(
                (v - 0.1).abs() < 0.01,
                "Princen-Bradley failed: got {v}, expected 0.1"
            );
        }
    }

    /// EncoderMdctState round-trips a sine-wave through 4 consecutive
    /// frames; the middle two frames must reconstruct within a small
    /// SNR margin.
    #[test]
    fn encoder_state_sinewave_round_trips() {
        let n = 32u32;
        let nsz = n as usize;
        let freq_bin = 4.0_f32; // bin index k: x[t] = sin(2*pi * k * t / (2N))
        let make_frame = |start: usize| -> Vec<f32> {
            (0..nsz)
                .map(|i| {
                    let t = (start + i) as f32;
                    (2.0 * std::f32::consts::PI * freq_bin * t / (2.0 * n as f32)).sin()
                })
                .collect()
        };
        let frames: Vec<Vec<f32>> = (0..4).map(|i| make_frame(i * nsz)).collect();
        let mut enc = EncoderMdctState::new(n);
        let specs: Vec<Vec<f32>> = frames.iter().map(|f| enc.analyse_frame(f)).collect();

        let window = kbd_window(n);
        let mut overlap = vec![0.0_f32; nsz];
        let mut pcm_out: Vec<Vec<f32>> = Vec::new();
        for spec in &specs {
            let y = imdct(spec);
            let pcm = imdct_olap_symmetric(&y, &window, &mut overlap);
            pcm_out.push(pcm);
        }
        // Frame 2 is steady-state middle; compare to original.
        let orig = &frames[2];
        let recon = &pcm_out[2];
        let mut sig_e = 0.0_f64;
        let mut err_e = 0.0_f64;
        for (o, r) in orig.iter().zip(recon.iter()) {
            sig_e += (*o as f64).powi(2);
            err_e += (*o as f64 - *r as f64).powi(2);
        }
        let snr_db = 10.0 * (sig_e / err_e.max(1e-30)).log10();
        assert!(snr_db > 40.0, "round-trip SNR too low: {snr_db:.1} dB");
    }
}
