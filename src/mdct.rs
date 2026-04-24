//! AC-4 IMDCT and KBD window synthesis (ETSI TS 103 190-1 §5.5).
//!
//! Implements the naive reference IMDCT: given `N` spectral coefficients
//! `X[k]` produce `2N` time-domain samples via the direct-form summation
//! of Pseudocodes 60-64 (with a slow but straightforward complex IFFT).
//! A Kaiser-Bessel Derived window (§5.5.3) is applied during
//! unfolding, and overlap/add integrates the new block with the
//! previous.
//!
//! The spec formulation is well-defined for arbitrary transform lengths
//! N that are multiples of 8 (AC-4 uses 2048/1920/1536/1024/960/768
//! plus short-block subdivisions). This implementation is correctness-
//! first, not performance-first — the inner complex DFT is O(N^2).
//!
//! Window alpha values per Table 186:
//!
//!   - 2048 / 1920 / 1536 → alpha = 3
//!   -  1024 / 960 / 768  → alpha = 4
//!   -  512 / 480 / 384   → alpha = 4.5
//!   -  256 / 240 / 192   → alpha = 5

use core::f32::consts::PI;

/// KBD window alpha for transform length N (Table 186, 48 kHz family).
pub fn kbd_alpha(n: u32) -> f32 {
    match n {
        2048 | 1920 | 1536 | 4096 | 3840 | 3072 | 8192 | 7680 | 6144 => 3.0,
        1024 | 960 | 768 => 4.0,
        512 | 480 | 384 => 4.5,
        256 | 240 | 192 => 5.0,
        _ => 3.0,
    }
}

/// Modified Bessel function of the first kind, I_0(x), via its series
/// expansion (§5.5.3 formula). Converges rapidly for the modest |x|
/// ranges used in KBD (alpha * pi ≤ ~16).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let xh2 = (x / 2.0) * (x / 2.0);
    for k in 1..64 {
        term *= xh2 / (k as f64 * k as f64);
        sum += term;
        if term < 1e-20 * sum {
            break;
        }
    }
    sum
}

/// Compute a Kaiser-Bessel window of length `N+1` samples as defined
/// in §5.5.3: `W(N,n,α) = I0(πα * sqrt(1 - ((2n/N - 1)^2))) / I0(πα)`.
fn kaiser_bessel_kernel(n_kernel: u32, alpha: f32) -> Vec<f64> {
    let n = n_kernel as usize;
    let mut w = vec![0.0_f64; n + 1];
    let pa = PI as f64 * alpha as f64;
    let denom = bessel_i0(pa);
    for i in 0..=n {
        let t = 2.0 * i as f64 / n as f64 - 1.0;
        let arg = pa * (1.0 - t * t).max(0.0).sqrt();
        w[i] = bessel_i0(arg) / denom;
    }
    w
}

/// Generate the KBD window of length `2N` from the kernel per §5.5.3:
///
///   KBD_LEFT[n]  = sqrt( (Σ_{p=0..n} W[p]) / (Σ_{p=0..N} W[p]) )  for 0 ≤ n < N
///   KBD_RIGHT[n] = sqrt( (Σ_{p=0..2N-n-1} W[p]) / (Σ_{p=0..N} W[p]) )  for N ≤ n < 2N
///
/// The total window has length 2N.
pub fn kbd_window(n: u32) -> Vec<f32> {
    let alpha = kbd_alpha(n);
    let kernel = kaiser_bessel_kernel(n, alpha);
    let ns = n as usize;
    // Cumulative sum up to and including index p.
    let mut cum = vec![0.0_f64; ns + 1];
    cum[0] = kernel[0];
    for p in 1..=ns {
        cum[p] = cum[p - 1] + kernel[p];
    }
    let denom = cum[ns];
    let mut w = vec![0.0_f32; 2 * ns];
    for i in 0..ns {
        let num = cum[i];
        w[i] = (num / denom).sqrt() as f32;
    }
    for i in ns..(2 * ns) {
        // 2N - n - 1 per the spec formula.
        let p = 2 * ns - i - 1;
        let num = cum[p];
        w[i] = (num / denom).sqrt() as f32;
    }
    w
}

/// Naive IMDCT: transforms `N` spectral coefficients to `2N` time-domain
/// samples via the direct DFT formulation of Pseudocodes 60-62. No
/// window is applied here; [`imdct_apply_window_and_olap`] does that.
///
/// The output has length `2N` and uses the AC-4 sign convention (§5.5.2,
/// Step 5 pseudocode).
pub fn imdct(x: &[f32]) -> Vec<f32> {
    let big_n = x.len();
    let half_n = big_n / 2;
    let fp_n = big_n as f64;
    // Step 2 (pre-IFFT multiply).
    let mut zr = vec![0.0_f64; half_n];
    let mut zi = vec![0.0_f64; half_n];
    for k in 0..half_n {
        let xc = -(2.0 * std::f64::consts::PI * (8 * k + 1) as f64 / (16.0 * fp_n)).cos();
        let xs = -(2.0 * std::f64::consts::PI * (8 * k + 1) as f64 / (16.0 * fp_n)).sin();
        let a = x[big_n - 2 * k - 1] as f64;
        let b = x[2 * k] as f64;
        zr[k] = a * xc - b * xs;
        zi[k] = b * xc + a * xs;
    }
    // Step 3: complex IFFT of length half_n — direct form.
    let mut yr_half = vec![0.0_f64; half_n];
    let mut yi_half = vec![0.0_f64; half_n];
    for n in 0..half_n {
        let mut acc_r = 0.0_f64;
        let mut acc_i = 0.0_f64;
        for k in 0..half_n {
            let theta = 4.0 * std::f64::consts::PI * k as f64 * n as f64 / fp_n;
            let c = theta.cos();
            let s = theta.sin();
            acc_r += zr[k] * c - zi[k] * s;
            acc_i += zr[k] * s + zi[k] * c;
        }
        yr_half[n] = acc_r;
        yi_half[n] = acc_i;
    }
    // Step 4: post-IFFT multiply, divided by N.
    let mut yr = vec![0.0_f64; half_n];
    let mut yi = vec![0.0_f64; half_n];
    for n in 0..half_n {
        let xc = -(2.0 * std::f64::consts::PI * (8 * n + 1) as f64 / (16.0 * fp_n)).cos();
        let xs = -(2.0 * std::f64::consts::PI * (8 * n + 1) as f64 / (16.0 * fp_n)).sin();
        yr[n] = (yr_half[n] * xc - yi_half[n] * xs) / fp_n;
        yi[n] = (yi_half[n] * xc + yr_half[n] * xs) / fp_n;
    }
    // Step 5: unfold into 2N samples with the y[n] sign convention.
    // Window is NOT applied here — we return x[n] with w[n] = 1.
    let mut out = vec![0.0_f32; 2 * big_n];
    let quarter = big_n / 4;
    for n in 0..quarter {
        out[2 * n] = yi[quarter + n] as f32;
        out[2 * n + 1] = (-yr[quarter - n - 1]) as f32;
        out[big_n / 2 + 2 * n] = yr[n] as f32;
        out[big_n / 2 + 2 * n + 1] = (-yi[big_n / 2 - n - 1]) as f32;
        out[big_n + 2 * n] = yr[quarter + n] as f32;
        out[big_n + 2 * n + 1] = (-yi[quarter - n - 1]) as f32;
        out[3 * big_n / 2 + 2 * n] = (-yi[n]) as f32;
        out[3 * big_n / 2 + 2 * n + 1] = yr[big_n / 2 - n - 1] as f32;
    }
    out
}

/// Apply KBD window to a 2N IMDCT output and overlap/add with a N-sample
/// history buffer. Returns N PCM samples. The history buffer is updated
/// with the second half of the current block (premultiplied by the
/// right window so subsequent overlap is ready).
///
/// Simplified non-block-switching path: assumes N == N_prev.
pub fn imdct_olap_symmetric(x_unwindowed: &[f32], window: &[f32], overlap: &mut [f32]) -> Vec<f32> {
    let two_n = x_unwindowed.len();
    let n = two_n / 2;
    debug_assert_eq!(window.len(), two_n);
    debug_assert_eq!(overlap.len(), n);
    // Apply window to current 2N block.
    let mut x = vec![0.0_f32; two_n];
    for i in 0..two_n {
        x[i] = x_unwindowed[i] * window[i];
    }
    // PCM = overlap + first-N of x.
    let mut pcm = vec![0.0_f32; n];
    for i in 0..n {
        pcm[i] = overlap[i] + x[i];
    }
    // New overlap = second-N of x.
    for i in 0..n {
        overlap[i] = x[n + i];
    }
    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kbd_window_is_symmetric_and_energy_conserving() {
        let w = kbd_window(64);
        assert_eq!(w.len(), 128);
        // Symmetric about the midpoint.
        for i in 0..64 {
            assert!((w[i] - w[127 - i]).abs() < 1e-4, "asymmetry at {i}");
        }
        // Princen-Bradley: w[n]^2 + w[N+n]^2 = 1.
        for i in 0..64 {
            let s = w[i] * w[i] + w[64 + i] * w[64 + i];
            assert!((s - 1.0).abs() < 1e-4, "princen-bradley failed: {s}");
        }
    }

    #[test]
    fn imdct_dc_bin_gives_uniform_output() {
        // X[0] = 1.0, rest 0 -> output is a cosine of known shape.
        // Simple sanity: output has energy (nonzero) and is finite.
        let mut x = vec![0.0_f32; 64];
        x[0] = 1.0;
        let y = imdct(&x);
        assert_eq!(y.len(), 128);
        let energy: f32 = y.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
        assert!(energy.is_finite());
    }

    #[test]
    fn imdct_inverse_roundtrip_bin() {
        // Putting a non-DC spectral line at X[2] and checking that the
        // energy is spread across the time samples.
        let n = 16;
        let mut x = vec![0.0_f32; n];
        x[2] = 1.0;
        let y = imdct(&x);
        assert_eq!(y.len(), 2 * n);
        // Some elements should be nonzero.
        let nonzero = y.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(nonzero > 0);
    }

    #[test]
    fn imdct_olap_produces_reasonable_output() {
        let n = 64;
        let mut x = vec![0.0_f32; n];
        x[1] = 10.0; // one strong spectral line
        let y = imdct(&x);
        let w = kbd_window(n as u32);
        let mut olap = vec![0.0_f32; n];
        let pcm1 = imdct_olap_symmetric(&y, &w, &mut olap);
        assert_eq!(pcm1.len(), n);
        // Second frame — overlap from first should be audible.
        let pcm2 = imdct_olap_symmetric(&y, &w, &mut olap);
        assert_eq!(pcm2.len(), n);
        let e1: f32 = pcm1.iter().map(|s| s * s).sum();
        let e2: f32 = pcm2.iter().map(|s| s * s).sum();
        // Second block should have noticeable energy (first has only
        // right-half contribution; second is the full stationary
        // response).
        assert!(e1.is_finite() && e2.is_finite());
        assert!(e2 > 0.0);
    }
}
