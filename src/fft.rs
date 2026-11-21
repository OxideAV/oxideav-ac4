//! Mixed-radix complex DFT plans shared by the MDCT / IMDCT pair.
//!
//! The AC-4 transform lengths (TS 103 190-1 §5.5.3) are `2^a · 3^b ·
//! 5^c`: the `N/2`-point complex transform the §5.5.2.2 IMDCT needs is
//! therefore built from radix-2 / 3 / 4 / 5 stages (any other prime
//! factor falls back to a direct `p`-point stage, so every length is
//! supported). Arithmetic is `f64` throughout; a plan is computed once
//! per length and cached per thread ([`plan`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A complex DFT of one fixed length.
///
/// `forward` computes `Y[k] = Σ_n y[n]·e^{−2πj·kn/len}`, `inverse` the
/// conjugate kernel (`e^{+2πj·kn/len}`), both unnormalised.
#[derive(Debug)]
pub(crate) struct FftPlan {
    len: usize,
    /// Prime-ish factors, outermost stage first.
    factors: Vec<usize>,
    /// `e^{−2πj·i/len}` for `i` in `0..len`.
    tw_re: Vec<f64>,
    tw_im: Vec<f64>,
}

impl FftPlan {
    fn new(len: usize) -> Self {
        let mut factors = Vec::new();
        let mut rest = len.max(1);
        // Radix 4 first (fewer passes), then 2, 3, 5, then whatever
        // primes remain.
        while rest.is_multiple_of(4) {
            factors.push(4);
            rest /= 4;
        }
        let mut p = 2;
        while rest > 1 {
            if rest.is_multiple_of(p) {
                factors.push(p);
                rest /= p;
            } else {
                p += 1;
            }
        }
        let mut tw_re = Vec::with_capacity(len);
        let mut tw_im = Vec::with_capacity(len);
        for i in 0..len {
            let a = -2.0 * std::f64::consts::PI * i as f64 / len as f64;
            tw_re.push(a.cos());
            tw_im.push(a.sin());
        }
        Self {
            len,
            factors,
            tw_re,
            tw_im,
        }
    }

    /// Transform length.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Forward transform (`e^{−2πj·kn/len}` kernel), in place.
    pub(crate) fn forward(&self, re: &mut [f64], im: &mut [f64], scratch: &mut FftScratch) {
        self.run(re, im, scratch);
    }

    /// Inverse-kernel transform (`e^{+2πj·kn/len}`), unnormalised, in
    /// place: the forward transform with real and imaginary parts
    /// swapped on the way in and out.
    pub(crate) fn inverse(&self, re: &mut [f64], im: &mut [f64], scratch: &mut FftScratch) {
        self.run(im, re, scratch);
    }

    fn run(&self, re: &mut [f64], im: &mut [f64], scratch: &mut FftScratch) {
        let n = self.len;
        debug_assert_eq!(re.len(), n);
        debug_assert_eq!(im.len(), n);
        if n <= 1 {
            return;
        }
        scratch.re.resize(n, 0.0);
        scratch.im.resize(n, 0.0);
        self.stage(re, im, &mut scratch.re, &mut scratch.im, 1, 1, 0);
        re.copy_from_slice(&scratch.re[..n]);
        im.copy_from_slice(&scratch.im[..n]);
    }

    /// Decimation-in-time recursion: transform the `m = len / stride`
    /// input samples `in[k·stride]` into `out[..m]`. `tw_step` is the
    /// stride of this level's root of unity in the twiddle table.
    #[allow(clippy::too_many_arguments)]
    fn stage(
        &self,
        in_re: &[f64],
        in_im: &[f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
        stride: usize,
        tw_step: usize,
        depth: usize,
    ) {
        let p = self.factors[depth];
        let m = out_re.len() / p;
        if m == 1 {
            for q in 0..p {
                out_re[q] = in_re[q * stride];
                out_im[q] = in_im[q * stride];
            }
        } else {
            for q in 0..p {
                self.stage(
                    &in_re[q * stride..],
                    &in_im[q * stride..],
                    &mut out_re[q * m..(q + 1) * m],
                    &mut out_im[q * m..(q + 1) * m],
                    stride * p,
                    tw_step * p,
                    depth + 1,
                );
            }
        }
        // Combine the p sub-transforms of length m:
        // X[k + r·m] = Σ_q w^{q·(k + r·m)} · S_q[k], w = e^{−2πj/(p·m)}.
        let n = self.len;
        let (twr, twi) = (&self.tw_re, &self.tw_im);
        match p {
            2 => {
                let (r0, r1) = out_re.split_at_mut(m);
                let (i0, i1) = out_im.split_at_mut(m);
                for k in 0..m {
                    let (wr, wi) = (twr[k * tw_step], twi[k * tw_step]);
                    let tr = r1[k] * wr - i1[k] * wi;
                    let ti = r1[k] * wi + i1[k] * wr;
                    r1[k] = r0[k] - tr;
                    i1[k] = i0[k] - ti;
                    r0[k] += tr;
                    i0[k] += ti;
                }
            }
            4 => {
                for k in 0..m {
                    let t1 = k * tw_step;
                    let (w1r, w1i) = (twr[t1], twi[t1]);
                    let (w2r, w2i) = (twr[2 * t1], twi[2 * t1]);
                    let (w3r, w3i) = (twr[3 * t1], twi[3 * t1]);
                    let (ar, ai) = (out_re[k], out_im[k]);
                    let (br, bi) = (
                        out_re[k + m] * w1r - out_im[k + m] * w1i,
                        out_re[k + m] * w1i + out_im[k + m] * w1r,
                    );
                    let (cr, ci) = (
                        out_re[k + 2 * m] * w2r - out_im[k + 2 * m] * w2i,
                        out_re[k + 2 * m] * w2i + out_im[k + 2 * m] * w2r,
                    );
                    let (dr, di) = (
                        out_re[k + 3 * m] * w3r - out_im[k + 3 * m] * w3i,
                        out_re[k + 3 * m] * w3i + out_im[k + 3 * m] * w3r,
                    );
                    let (s0r, s0i) = (ar + cr, ai + ci);
                    let (s1r, s1i) = (ar - cr, ai - ci);
                    let (s2r, s2i) = (br + dr, bi + di);
                    let (s3r, s3i) = (br - dr, bi - di);
                    out_re[k] = s0r + s2r;
                    out_im[k] = s0i + s2i;
                    // −j · (b − d)
                    out_re[k + m] = s1r + s3i;
                    out_im[k + m] = s1i - s3r;
                    out_re[k + 2 * m] = s0r - s2r;
                    out_im[k + 2 * m] = s0i - s2i;
                    out_re[k + 3 * m] = s1r - s3i;
                    out_im[k + 3 * m] = s1i + s3r;
                }
            }
            _ => {
                // Generic odd / prime radix: p ≤ 5 for every AC-4
                // length, so the O(p²) butterfly is cheap.
                let mut vr = [0.0f64; 8];
                let mut vi = [0.0f64; 8];
                let mut big_r: Vec<f64> = Vec::new();
                let mut big_i: Vec<f64> = Vec::new();
                let on_stack = p <= 8;
                if !on_stack {
                    big_r = vec![0.0; p];
                    big_i = vec![0.0; p];
                }
                for k in 0..m {
                    {
                        let (sr, si): (&mut [f64], &mut [f64]) = if on_stack {
                            (&mut vr[..p], &mut vi[..p])
                        } else {
                            (&mut big_r[..], &mut big_i[..])
                        };
                        for q in 0..p {
                            sr[q] = out_re[k + q * m];
                            si[q] = out_im[k + q * m];
                        }
                    }
                    let (sr, si): (&[f64], &[f64]) = if on_stack {
                        (&vr[..p], &vi[..p])
                    } else {
                        (&big_r[..], &big_i[..])
                    };
                    for r in 0..p {
                        let idx = k + r * m;
                        let mut accr = sr[0];
                        let mut acci = si[0];
                        for q in 1..p {
                            let t = (q * idx * tw_step) % n;
                            accr += sr[q] * twr[t] - si[q] * twi[t];
                            acci += sr[q] * twi[t] + si[q] * twr[t];
                        }
                        out_re[idx] = accr;
                        out_im[idx] = acci;
                    }
                }
            }
        }
    }
}

/// Reusable work buffers for [`FftPlan`] transforms.
#[derive(Debug, Default)]
pub(crate) struct FftScratch {
    re: Vec<f64>,
    im: Vec<f64>,
}

thread_local! {
    static PLANS: RefCell<HashMap<usize, Rc<FftPlan>>> = RefCell::new(HashMap::new());
}

/// The per-thread shared plan for a `len`-point transform.
pub(crate) fn plan(len: usize) -> Rc<FftPlan> {
    PLANS.with(|p| {
        p.borrow_mut()
            .entry(len)
            .or_insert_with(|| Rc::new(FftPlan::new(len)))
            .clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(re: &[f64], im: &[f64], sign: f64) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let mut or = vec![0.0; n];
        let mut oi = vec![0.0; n];
        for k in 0..n {
            for t in 0..n {
                let a = sign * 2.0 * std::f64::consts::PI * (k * t % n) as f64 / n as f64;
                or[k] += re[t] * a.cos() - im[t] * a.sin();
                oi[k] += re[t] * a.sin() + im[t] * a.cos();
            }
        }
        (or, oi)
    }

    #[test]
    fn plans_match_the_direct_sum_for_every_ac4_length() {
        for &n in &[
            1usize, 2, 3, 4, 5, 7, 8, 12, 15, 48, 60, 64, 96, 120, 128, 192, 240, 256, 384, 480,
            512, 768, 960, 1024, 14,
        ] {
            let mut re: Vec<f64> = (0..n)
                .map(|i| ((i * 37 + 11) % 101) as f64 - 50.0)
                .collect();
            let mut im: Vec<f64> = (0..n).map(|i| ((i * 53 + 7) % 89) as f64 - 44.0).collect();
            let (er, ei) = direct(&re, &im, -1.0);
            let (ir, ii) = direct(&re, &im, 1.0);
            let p = plan(n);
            assert_eq!(p.len(), n);
            let mut s = FftScratch::default();
            let (mut fr, mut fi) = (re.clone(), im.clone());
            p.forward(&mut fr, &mut fi, &mut s);
            p.inverse(&mut re, &mut im, &mut s);
            let tol = 1e-8 * n as f64;
            for k in 0..n {
                assert!(
                    (fr[k] - er[k]).abs() < tol && (fi[k] - ei[k]).abs() < tol,
                    "fwd {n} {k}"
                );
                assert!(
                    (re[k] - ir[k]).abs() < tol && (im[k] - ii[k]).abs() < tol,
                    "inv {n} {k}"
                );
            }
        }
    }
}
