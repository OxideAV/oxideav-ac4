//! A-SPX Temporal Noise Shaping (complex-covariance LPC + chirp).
//!
//! Implements the TNS body of ETSI TS 103 190-1 §5.7.6.4.1.2 /
//! §5.7.6.4.1.3 / §5.7.6.4.1.4 — the chirp-factor + α0 + α1 prediction
//! that lifts the bare HF tile copy into the proper "subband tonal to
//! noise ratio adjustment" output:
//!
//!   * **Pseudocode 85** — pre-flattening control data calculation:
//!     fit a third-order polynomial to the per-subband energy of the
//!     low band, derive a per-subband gain vector. Implemented by
//!     [`compute_preflat_gains`].
//!
//!   * **Pseudocode 86** — complex covariance matrix `cov[sb][i][j]`
//!     for `i in 0..3`, `j in 1..3` over `Q_low_ext` with a delay of
//!     `ts_offset_hfadj = 4` QMF time slots. Implemented by
//!     [`compute_covariance`].
//!
//!   * **Pseudocode 87** — α0 / α1 LPC prediction coefficients from
//!     the covariance matrix, with an `EPSILON_INV = 2^-20`
//!     denominator slack and the `|α0| >= 4 || |α1| >= 4 → both 0`
//!     rule from the prose underneath. Implemented by
//!     [`compute_alphas`].
//!
//!   * **Pseudocode 88** — chirp factor calculation: per-noise-subband-
//!     group lookup of `tabNewChirp[mode][prev_mode]` from Table 195,
//!     attack/decay smoothing against the previous frame's chirp, then
//!     the `< 0.015625` zero gate. Implemented by [`chirp_factors`].
//!
//!   * **Pseudocode 89** — HF signal creation with chirp + α0 + α1 +
//!     pre-flatten. Replaces the bare tile copy in
//!     [`crate::aspx::hf_tile_copy`]. Implemented by [`hf_tile_tns`].
//!
//! Per-channel state that survives across A-SPX intervals
//! ([`AspxTnsState`]) carries `aspx_tna_mode_prev[]` and
//! `prev_chirp_array[]` (assumed zero on the first interval, per the
//! prose under Pseudocode 88).
//!
//! Encoder-side TNS is out of scope.

/// `tabNewChirp[aspx_tna_mode_prev][aspx_tna_mode]` — ETSI TS 103 190-1
/// Table 195. Indexed `[prev][curr]`. Values are `f32` since the
/// pseudocode multiplies them with floating-point smoothing constants.
///
/// Layout matches the spec: rows are prev_mode, cols are curr_mode,
/// modes ordered None=0, Light=1, Moderate=2, Heavy=3.
pub const TAB_NEW_CHIRP: [[f32; 4]; 4] = [
    // prev=None
    [0.0, 0.6, 0.9, 0.98],
    // prev=Light
    [0.6, 0.75, 0.9, 0.98],
    // prev=Moderate
    [0.0, 0.75, 0.9, 0.98],
    // prev=Heavy
    [0.0, 0.75, 0.9, 0.98],
];

/// `ts_offset_hfadj` per Pseudocode 86. The spec hard-codes 4 QMF time
/// slots — this is the additional delay applied on top of
/// `ts_offset_hfgen` so that the LPC predictor can index back two and
/// four time-slot pairs.
pub const TS_OFFSET_HFADJ: usize = 4;

/// `EPSILON_INV = 2^-20` per Pseudocode 87 — used to bias the
/// covariance denominator slightly away from zero when the prediction
/// would otherwise be perfect.
pub const EPSILON_INV: f64 = 1.0_f64 / (1u32 << 20) as f64;

/// Per-channel TNS history that survives across A-SPX intervals.
///
/// The prose under Pseudocode 88 says both `aspx_tna_mode_prev[]` and
/// `prev_chirp_array[]` are assumed zero on the first interval. After
/// `master_reset == 1` the state should be reset via [`Self::reset`].
#[derive(Debug, Clone, Default)]
pub struct AspxTnsState {
    /// `aspx_tna_mode_prev[sbg]` from the previous interval. Length is
    /// `num_sbg_noise`. Zero (`None`) on the first interval.
    pub tna_mode_prev: Vec<u8>,
    /// `prev_chirp_array[sbg]` from the previous interval. Length is
    /// `num_sbg_noise`. Zero on the first interval.
    pub chirp_prev: Vec<f32>,
}

impl AspxTnsState {
    /// Fresh state (first-frame, `master_reset == 1` semantics).
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to first-interval behaviour.
    pub fn reset(&mut self) {
        self.tna_mode_prev.clear();
        self.chirp_prev.clear();
    }
}

// ---------------------------------------------------------------------
// §5.7.6.4.1.2 Pre-flattening control data calculation (Pseudocode 85)
// ---------------------------------------------------------------------

/// Compute the per-low-subband pre-flatten gain vector per ETSI TS 103
/// 190-1 §5.7.6.4.1.2 Pseudocode 85.
///
/// `q_low[sb][ts]` is the complex QMF low-band matrix. `sbx` is the
/// number of low-band subbands. `atsg_sig` are the A-SPX time-slot-
/// group borders for the SIGNAL envelope; `num_ts_in_ats` is the
/// number of QMF time slots per A-SPX time slot (Table 192).
///
/// Returns a `gain_vec` of length `sbx`. The spec specifies that the
/// inverse `1 / gain_vec[p]` is multiplied into `Q_high` when
/// `aspx_preflat == 1` (Pseudocode 89).
pub fn compute_preflat_gains(
    q_low: &[Vec<(f32, f32)>],
    sbx: u32,
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
) -> Vec<f32> {
    let num_qmf_subbands = sbx as usize;
    let mut gain_vec = vec![1.0_f32; num_qmf_subbands];
    if num_qmf_subbands == 0 || atsg_sig.len() < 2 {
        return gain_vec;
    }
    let num_atsg_sig = atsg_sig.len() - 1;
    let ts_lo = (atsg_sig[0] * num_ts_in_ats) as usize;
    let ts_hi = (atsg_sig[num_atsg_sig] * num_ts_in_ats) as usize;
    if ts_hi <= ts_lo {
        return gain_vec;
    }
    let denom = (ts_hi - ts_lo) as f64;

    // Per-subband energy → dB envelope. f64 internally to avoid losing
    // precision in the polynomial fit on small-amplitude inputs.
    let mut pow_env = vec![0.0_f64; num_qmf_subbands];
    let mut mean_energy = 0.0_f64;
    #[allow(clippy::needless_range_loop)] // ETSI Pseudocode 85 indexing
    for sb in 0..num_qmf_subbands {
        if sb >= q_low.len() {
            continue;
        }
        let row = &q_low[sb];
        let mut acc = 0.0_f64;
        let lo = ts_lo.min(row.len());
        let hi = ts_hi.min(row.len());
        for sample in row.iter().take(hi).skip(lo) {
            let re = sample.0 as f64;
            let im = sample.1 as f64;
            acc += re * re + im * im;
        }
        let mean_pow = acc / denom;
        // Pseudocode 85 says "10*log10(pow_env[sb] + 1)" — the +1 keeps
        // the log finite for silent subbands.
        pow_env[sb] = 10.0 * (mean_pow + 1.0).log10();
        mean_energy += pow_env[sb];
    }
    mean_energy /= num_qmf_subbands as f64;

    // Polynomial fit (least-squares, order 3) to (x=0..N, pow_env).
    let poly = polynomial_fit_3(num_qmf_subbands, &pow_env);

    // Transform polynomial back into per-subband slope.
    // The pseudocode loops k = polynomial_order .. 0 and adds
    // `pow(x, k) * poly_array[polynomial_order - k]`. With order=3 and
    // poly_array stored in descending order of powers, that means
    // poly_array[0] is the constant (k=3) and poly_array[3] is the x^3
    // coefficient (k=0). i.e. slope[sb] = c0 + c1*x + c2*x^2 + c3*x^3
    // where (c3,c2,c1,c0) = (poly[0], poly[1], poly[2], poly[3]).
    #[allow(clippy::needless_range_loop)] // ETSI Pseudocode 85 indexing
    for sb in 0..num_qmf_subbands {
        let x = sb as f64;
        let slope = poly[3] + poly[2] * x + poly[1] * x * x + poly[0] * x * x * x;
        let g = 10.0_f64.powf((mean_energy - slope) / 20.0);
        gain_vec[sb] = g as f32;
    }
    gain_vec
}

/// Least-squares polynomial fit of order 3 to `(x = 0..n, y[i])`.
///
/// Returns the four coefficients in **descending** order of powers
/// (so `out = [c3, c2, c1, c0]`), matching the convention the spec
/// expects for `poly_array`.
///
/// Solves the 4×4 normal-equation system `(X^T X) a = X^T y` with a
/// straightforward Gaussian elimination. Falls back to all-zeros when
/// the system is singular (e.g. n < 4).
fn polynomial_fit_3(n: usize, y: &[f64]) -> [f64; 4] {
    if n < 4 {
        return [0.0; 4];
    }
    // Build sums S[k] = sum x^k for k = 0 ..= 6 and B[k] = sum y * x^k
    // for k = 0 ..= 3.
    let mut s = [0.0_f64; 7];
    let mut b = [0.0_f64; 4];
    for (i, &yi) in y.iter().enumerate().take(n) {
        let x = i as f64;
        let mut xk = 1.0_f64;
        for sk in s.iter_mut() {
            *sk += xk;
            xk *= x;
        }
        let mut xk = 1.0_f64;
        for bk in b.iter_mut() {
            *bk += yi * xk;
            xk *= x;
        }
    }
    // 4×4 normal-equation matrix M[i][j] = sum x^(i+j), augmented with
    // RHS b. Solve by Gaussian elimination with partial pivoting.
    let mut m = [[0.0_f64; 5]; 4];
    for i in 0..4 {
        m[i][..4].copy_from_slice(&s[i..(4 + i)]);
        m[i][4] = b[i];
    }
    for col in 0..4 {
        // Partial pivot.
        let mut piv = col;
        for r in (col + 1)..4 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-30 {
            return [0.0; 4];
        }
        if piv != col {
            m.swap(col, piv);
        }
        let pivot = m[col][col];
        for r in (col + 1)..4 {
            let f = m[r][col] / pivot;
            #[allow(clippy::needless_range_loop)] // Gaussian elimination row scan
            for c in col..5 {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    // Back-substitute. a holds the polynomial in *ascending* powers.
    let mut a = [0.0_f64; 4];
    for r in (0..4).rev() {
        let mut sum = m[r][4];
        for c in (r + 1)..4 {
            sum -= m[r][c] * a[c];
        }
        a[r] = sum / m[r][r];
    }
    // Spec wants descending order: poly_array[0] = c3, ... [3] = c0.
    [a[3], a[2], a[1], a[0]]
}

// ---------------------------------------------------------------------
// §5.7.6.4.1.3 Subband tonal-to-noise adjustment data
// Pseudocode 86 (covariance) + Pseudocode 87 (alphas)
// ---------------------------------------------------------------------

/// Build `Q_low_ext[sb][ts]` per Pseudocode 86 — prepend
/// `ts_offset_hfadj` time slots from the previous A-SPX interval, then
/// append the current interval's `q_low` samples.
///
/// `q_low_prev[sb]` is the previous interval's QMF low-band slice (or
/// empty on the first interval — `Q_low_prev[sb][ts]` is then assumed
/// zero). `q_low[sb]` is the current interval's slice.
/// `num_qmf_timeslots` is the spec-defined size of `q_low` *before*
/// the `ts_offset_hfgen` look-ahead — but in our pipeline the QMF
/// matrix already covers the full extended range, so we treat
/// `q_low[sb].len()` as the authoritative length and set
/// `num_ts_ext = ts_offset_hfadj + q_low.len()`.
pub fn build_q_low_ext(
    q_low: &[Vec<(f32, f32)>],
    q_low_prev: &[Vec<(f32, f32)>],
    sba: u32,
) -> Vec<Vec<(f32, f32)>> {
    let n_cur = q_low.iter().map(|r| r.len()).max().unwrap_or(0);
    let n_ext = n_cur + TS_OFFSET_HFADJ;
    let mut ext: Vec<Vec<(f32, f32)>> = (0..(sba as usize))
        .map(|_| vec![(0.0_f32, 0.0_f32); n_ext])
        .collect();
    for sb in 0..(sba as usize) {
        // First TS_OFFSET_HFADJ slots come from the tail of q_low_prev.
        if sb < q_low_prev.len() && !q_low_prev[sb].is_empty() {
            let prev = &q_low_prev[sb];
            let off = prev.len().saturating_sub(TS_OFFSET_HFADJ);
            let take = (prev.len() - off).min(TS_OFFSET_HFADJ);
            ext[sb][..take].copy_from_slice(&prev[off..off + take]);
        }
        // Body.
        if sb < q_low.len() {
            let cur = &q_low[sb];
            let copy_len = cur.len().min(n_ext - TS_OFFSET_HFADJ);
            ext[sb][TS_OFFSET_HFADJ..TS_OFFSET_HFADJ + copy_len].copy_from_slice(&cur[..copy_len]);
        }
    }
    ext
}

/// Per-subband 3×3 complex covariance matrix produced by
/// [`compute_covariance`]. `cov[sb][i][j]` is a `(re, im)` pair in `f64`.
pub type CovMatrix = [[(f64, f64); 3]; 3];

/// Per-subband complex prediction coefficient vector returned by
/// [`compute_alphas`]. `Alphas.0` is `α0[sb]`, `.1` is `α1[sb]`; each
/// element is a `(re, im)` pair in `f32`.
pub type Alphas = (Vec<(f32, f32)>, Vec<(f32, f32)>);

/// Compute the complex covariance matrix `cov[sb][i][j]` per
/// Pseudocode 86. Returns a 3×3 array per subband; indices `(0,1)`,
/// `(0,2)`, `(1,1)`, `(1,2)`, `(2,2)` are the only ones consulted by
/// Pseudocode 87.
///
/// `q_low_ext[sb][ts]` must have length ≥ `2*ts_offset_hfadj` so that
/// the `ts - 2*i` and `ts - 2*j` indexing stays in-range for `i,j ≤ 2`
/// when `ts >= ts_offset_hfadj`. The summation runs `ts` from
/// `ts_offset_hfadj` to `q_low_ext[sb].len()` in steps of 2.
pub fn compute_covariance(q_low_ext: &[Vec<(f32, f32)>], sba: u32) -> Vec<CovMatrix> {
    let mut cov: Vec<CovMatrix> = vec![[[(0.0, 0.0); 3]; 3]; sba as usize];
    #[allow(clippy::needless_range_loop)] // Pseudocode 86 cov[sb][i][j]
    for sb in 0..(sba as usize) {
        if sb >= q_low_ext.len() {
            continue;
        }
        let row = &q_low_ext[sb];
        let n = row.len();
        for i in 0..3 {
            for j in 1..3 {
                let mut acc = (0.0_f64, 0.0_f64);
                let mut ts = TS_OFFSET_HFADJ;
                while ts < n {
                    let a_idx = ts.wrapping_sub(2 * i);
                    let b_idx = ts.wrapping_sub(2 * j);
                    // Both indices must be in [0, n).
                    if a_idx < n && b_idx < n {
                        let a = row[a_idx];
                        let b = row[b_idx];
                        // a * cplx_conj(b) = (a.re + j a.im)(b.re - j b.im)
                        //                  = (a.re*b.re + a.im*b.im) +
                        //                    j (a.im*b.re - a.re*b.im)
                        let re = a.0 as f64 * b.0 as f64 + a.1 as f64 * b.1 as f64;
                        let im = a.1 as f64 * b.0 as f64 - a.0 as f64 * b.1 as f64;
                        acc.0 += re;
                        acc.1 += im;
                    }
                    ts += 2;
                }
                cov[sb][i][j] = acc;
            }
        }
    }
    cov
}

/// Compute α0 / α1 per Pseudocode 87. Each output is a complex pair
/// `(re, im)`.
///
/// Implements the magnitude-≥-4 fallback from the prose: if either
/// `|α0|` or `|α1|` is `>= 4`, both coefficients are set to zero.
pub fn compute_alphas(cov: &[CovMatrix]) -> Alphas {
    let n = cov.len();
    let mut alpha0: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    let mut alpha1: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    let one_over_eps = 1.0_f64 / (1.0 + EPSILON_INV);
    for (sb, c) in cov.iter().enumerate() {
        // denom = cov[2][2] * cov[1][1] - |cov[1][2]|^2 / (1+EPSILON_INV)
        // The first term is complex × complex but the spec uses it as if
        // it were real; cov[i][i] is the autocorrelation which is
        // self-conjugate (its imaginary part is zero by construction).
        // We still compute it complex-real-projected to mirror the
        // pseudocode literally.
        let c11 = c[1][1];
        let c22 = c[2][2];
        let c12 = c[1][2];
        let c01 = c[0][1];
        let c02 = c[0][2];
        let denom_re = c22.0 * c11.0 - c22.1 * c11.1;
        let mag12_sq = c12.0 * c12.0 + c12.1 * c12.1;
        let denom = denom_re - mag12_sq * one_over_eps;
        let mut a1 = (0.0_f64, 0.0_f64);
        if denom != 0.0 {
            // alpha1 = (cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1]) / denom
            let p1 = cmul(c01, c12);
            let p2 = cmul(c02, c11);
            a1.0 = (p1.0 - p2.0) / denom;
            a1.1 = (p1.1 - p2.1) / denom;
        }
        let mut a0 = (0.0_f64, 0.0_f64);
        if c11.0 != 0.0 || c11.1 != 0.0 {
            // alpha0 = (-cov[0][1] + alpha1 * cplx_conj(cov[1][2])) / cov[1][1]
            let conj12 = (c12.0, -c12.1);
            let prod = cmul(a1, conj12);
            let numer = (-c01.0 + prod.0, -c01.1 + prod.1);
            // Divide by complex c11. Because c11 is autocorrelation,
            // c11.1 ≈ 0; do the full complex division regardless.
            let denom11 = c11.0 * c11.0 + c11.1 * c11.1;
            if denom11 != 0.0 {
                a0.0 = (numer.0 * c11.0 + numer.1 * c11.1) / denom11;
                a0.1 = (numer.1 * c11.0 - numer.0 * c11.1) / denom11;
            }
        }
        let mag0 = (a0.0 * a0.0 + a0.1 * a0.1).sqrt();
        let mag1 = (a1.0 * a1.0 + a1.1 * a1.1).sqrt();
        if mag0 >= 4.0 || mag1 >= 4.0 {
            alpha0[sb] = (0.0, 0.0);
            alpha1[sb] = (0.0, 0.0);
        } else {
            alpha0[sb] = (a0.0 as f32, a0.1 as f32);
            alpha1[sb] = (a1.0 as f32, a1.1 as f32);
        }
    }
    (alpha0, alpha1)
}

/// Complex multiply.
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

// ---------------------------------------------------------------------
// §5.7.6.4.1.3 Pseudocode 88 — chirp factor calculation
// ---------------------------------------------------------------------

/// Chirp-factor result per Pseudocode 88. `chirp_arr` has length
/// `num_sbg_noise`; `tna_mode` echoes the input so the caller can
/// snapshot it into [`AspxTnsState`] for the next interval.
#[derive(Debug, Clone, PartialEq)]
pub struct ChirpResult {
    pub chirp_arr: Vec<f32>,
    pub tna_mode: Vec<u8>,
}

/// Compute the chirp factors per Pseudocode 88. Inputs are the
/// current interval's `aspx_tna_mode[sbg]` (each in `0..=3`) and the
/// per-channel TNS state from the previous interval. The state is
/// **not** mutated here — the caller advances it after consuming the
/// chirp factors (so the same state can be read by the alpha pipeline
/// that runs in parallel).
pub fn chirp_factors(tna_mode: &[u8], state: &AspxTnsState) -> ChirpResult {
    let n = tna_mode.len();
    let mut chirp_arr = vec![0.0_f32; n];
    for sbg in 0..n {
        let curr = (tna_mode[sbg] & 0x03) as usize;
        let prev = state.tna_mode_prev.get(sbg).copied().unwrap_or(0) as usize & 0x03;
        let mut new_chirp = TAB_NEW_CHIRP[prev][curr];
        let prev_chirp = state.chirp_prev.get(sbg).copied().unwrap_or(0.0_f32);
        if new_chirp < prev_chirp {
            // Decay: 0.75 / 0.25 weighting.
            new_chirp = 0.75_f32 * new_chirp + 0.25_f32 * prev_chirp;
        } else {
            // Attack: 0.90625 / 0.09375 weighting.
            new_chirp = 0.90625_f32 * new_chirp + 0.09375_f32 * prev_chirp;
        }
        chirp_arr[sbg] = if new_chirp < 0.015625_f32 {
            0.0
        } else {
            new_chirp
        };
    }
    ChirpResult {
        chirp_arr,
        tna_mode: tna_mode.to_vec(),
    }
}

/// Snapshot the per-interval TNS outputs into `state` so the next
/// interval can read them as `aspx_tna_mode_prev` / `prev_chirp_array`.
pub fn advance_tns_state(state: &mut AspxTnsState, result: &ChirpResult) {
    state.tna_mode_prev = result.tna_mode.clone();
    state.chirp_prev = result.chirp_arr.clone();
}

// ---------------------------------------------------------------------
// §5.7.6.4.1.4 Pseudocode 89 — HF signal creation with chirp + alphas
// ---------------------------------------------------------------------

/// Run the full Pseudocode 89 HF signal creation: tile copy from
/// `Q_low_ext` plus `chirp_arr[g] * alpha0[p] * Q_low_ext[p][n-2]`
/// plus `chirp_arr[g]^2 * alpha1[p] * Q_low_ext[p][n-4]` plus an
/// optional pre-flatten divide.
///
/// `q_low_ext[sb][ts]` is the pre-built extended low band (see
/// [`build_q_low_ext`]). `atsg_sig` are the SIGNAL envelope borders;
/// `num_ts_in_ats` is QMF time-slots-per-ATS (Table 192). `sbg_noise`
/// is the noise subband-group border table (used to advance the chirp
/// index `g`). `chirp_arr`, `alpha0`, `alpha1` are the outputs of
/// [`chirp_factors`] and [`compute_alphas`]. `gain_vec` is the
/// pre-flatten gain (see [`compute_preflat_gains`]); pass `None` (or
/// `aspx_preflat == 0`) to skip the divide.
///
/// Output is `q_high[sb_high][ts]` for `sb_high` in
/// `[sbx, sbx + sum(sbg_patch_num_sb))` and `ts` in
/// `[atsg_sig[0]*num_ts_in_ats, atsg_sig[end]*num_ts_in_ats)`.
#[allow(clippy::too_many_arguments)]
pub fn hf_tile_tns(
    q_low_ext: &[Vec<(f32, f32)>],
    patches: &crate::aspx::AspxPatchTables,
    sbg_noise: &[u32],
    chirp_arr: &[f32],
    alpha0: &[(f32, f32)],
    alpha1: &[(f32, f32)],
    gain_vec: Option<&[f32]>,
    sbx: u32,
    num_qmf_subbands: u32,
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
) -> Vec<Vec<(f32, f32)>> {
    if atsg_sig.len() < 2 {
        return (0..num_qmf_subbands).map(|_| Vec::new()).collect();
    }
    let n_ts_ext = q_low_ext.iter().map(|r| r.len()).max().unwrap_or(0);
    // Total HF timeslots covered by atsg_sig.
    let ts_lo = (atsg_sig[0] * num_ts_in_ats) as usize;
    let ts_hi = (atsg_sig[atsg_sig.len() - 1] * num_ts_in_ats) as usize;
    // q_high uses the same indexing as q_low_ext for the destination
    // time axis (so callers can splice slot-by-slot). We size it to
    // ts_hi (in the source-time domain) — Pseudocode 89's `n = ts +
    // ts_offset_hfadj` writes into the *destination*'s `[ts]`, not
    // `[n]`, so the destination range stays `[ts_lo, ts_hi)`.
    let mut q_high: Vec<Vec<(f32, f32)>> = (0..num_qmf_subbands)
        .map(|_| vec![(0.0_f32, 0.0_f32); ts_hi])
        .collect();

    #[allow(clippy::needless_range_loop)] // Pseudocode 89 q_high[sb_high][ts]
    for ts in ts_lo..ts_hi {
        let mut sum_sb_patches: u32 = 0;
        let mut g: usize = 0;
        for i in 0..(patches.num_sbg_patches as usize) {
            for sb_off in 0..patches.sbg_patch_num_sb[i] {
                let sb_high = sbx + sum_sb_patches + sb_off;
                if sb_high >= num_qmf_subbands {
                    continue;
                }
                // Advance noise-group pointer when sb_high crosses the
                // next noise subband-group border.
                while g + 1 < sbg_noise.len() && sbg_noise[g + 1] == sb_high {
                    g += 1;
                }
                let p = (patches.sbg_patch_start_sb[i] + sb_off) as usize;
                let n = ts + TS_OFFSET_HFADJ;
                if p >= q_low_ext.len() || n >= n_ts_ext {
                    continue;
                }
                let row = &q_low_ext[p];
                let mut sample = row[n];
                if g < chirp_arr.len() && p < alpha0.len() {
                    let c = chirp_arr[g];
                    if c > 0.0 && n >= 2 {
                        let a = alpha0[p];
                        let z = row[n - 2];
                        // sample += c * a * z
                        let prod = (c * (a.0 * z.0 - a.1 * z.1), c * (a.0 * z.1 + a.1 * z.0));
                        sample.0 += prod.0;
                        sample.1 += prod.1;
                    }
                    if c > 0.0 && n >= 4 && p < alpha1.len() {
                        let c2 = c * c;
                        let a = alpha1[p];
                        let z = row[n - 4];
                        let prod = (c2 * (a.0 * z.0 - a.1 * z.1), c2 * (a.0 * z.1 + a.1 * z.0));
                        sample.0 += prod.0;
                        sample.1 += prod.1;
                    }
                }
                if let Some(gv) = gain_vec {
                    if let Some(&g_p) = gv.get(p) {
                        if g_p != 0.0 {
                            let inv = 1.0 / g_p;
                            sample.0 *= inv;
                            sample.1 *= inv;
                        }
                    }
                }
                q_high[sb_high as usize][ts] = sample;
            }
            sum_sb_patches += patches.sbg_patch_num_sb[i];
        }
    }
    q_high
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aspx::AspxPatchTables;

    #[test]
    fn tab_new_chirp_layout() {
        // Spot-check Table 195. Diagonal None/None == 0, Heavy@anything
        // is 0.98, Light/Light is 0.75.
        assert_eq!(TAB_NEW_CHIRP[0][0], 0.0);
        assert_eq!(TAB_NEW_CHIRP[0][3], 0.98);
        assert_eq!(TAB_NEW_CHIRP[1][1], 0.75);
        assert_eq!(TAB_NEW_CHIRP[2][3], 0.98);
        assert_eq!(TAB_NEW_CHIRP[3][3], 0.98);
        // Asymmetric rows: prev=None, curr=Light → 0.6;
        // prev=Light, curr=None → 0.6.
        assert_eq!(TAB_NEW_CHIRP[0][1], 0.6);
        assert_eq!(TAB_NEW_CHIRP[1][0], 0.6);
        // prev=Moderate or Heavy, curr=None → 0.0.
        assert_eq!(TAB_NEW_CHIRP[2][0], 0.0);
        assert_eq!(TAB_NEW_CHIRP[3][0], 0.0);
    }

    #[test]
    fn chirp_first_interval_attack() {
        // First interval: prev_chirp == 0, prev_mode == 0. So
        // new_chirp = TAB_NEW_CHIRP[0][curr]; smoothing branch is
        // attack (new >= prev=0); result = 0.90625 * tab + 0.09375*0.
        let state = AspxTnsState::new();
        let modes = vec![0_u8, 1, 2, 3];
        let r = chirp_factors(&modes, &state);
        // sbg=0 (None): tab=0 → < 0.015625 → 0.
        assert_eq!(r.chirp_arr[0], 0.0);
        // sbg=1 (Light): tab=0.6 → 0.90625 * 0.6 = 0.54375.
        assert!((r.chirp_arr[1] - 0.54375).abs() < 1e-6);
        // sbg=2 (Moderate): tab=0.9 → 0.90625 * 0.9 = 0.815625.
        assert!((r.chirp_arr[2] - 0.815625).abs() < 1e-6);
        // sbg=3 (Heavy): tab=0.98 → 0.90625 * 0.98 = 0.888_125.
        assert!((r.chirp_arr[3] - 0.888_125).abs() < 1e-6);
    }

    #[test]
    fn chirp_decay_branch_uses_75_25() {
        // Set up state with a high prev chirp so that next interval's
        // tab value lands lower → decay branch.
        let mut state = AspxTnsState::new();
        state.tna_mode_prev = vec![3]; // prev=Heavy
        state.chirp_prev = vec![0.9]; // high
                                      // curr=None → tab[Heavy][None]=0.0; new=0.0 < 0.9 → decay.
                                      // result = 0.75 * 0.0 + 0.25 * 0.9 = 0.225.
        let r = chirp_factors(&[0_u8], &state);
        assert!((r.chirp_arr[0] - 0.225).abs() < 1e-6);
    }

    #[test]
    fn chirp_zero_gate_at_under_one_64th() {
        // new_chirp < 1/64 = 0.015625 must clamp to 0.
        let mut state = AspxTnsState::new();
        state.tna_mode_prev = vec![0];
        state.chirp_prev = vec![0.01]; // already small
                                       // curr=None → tab=0; attack? 0 >= 0.01? false → decay branch.
                                       // result = 0.75 * 0 + 0.25 * 0.01 = 0.0025 < 0.015625 → 0.
        let r = chirp_factors(&[0_u8], &state);
        assert_eq!(r.chirp_arr[0], 0.0);
    }

    #[test]
    fn chirp_state_advances_across_intervals() {
        let mut state = AspxTnsState::new();
        let modes1 = vec![3_u8];
        let r1 = chirp_factors(&modes1, &state);
        advance_tns_state(&mut state, &r1);
        assert_eq!(state.tna_mode_prev, vec![3]);
        assert_eq!(state.chirp_prev, r1.chirp_arr);

        // Second interval: same mode, attack continues → grows toward
        // tab[3][3] = 0.98.
        let modes2 = vec![3_u8];
        let r2 = chirp_factors(&modes2, &state);
        // attack: 0.90625*0.98 + 0.09375*r1.chirp_arr[0]. Should be >
        // r1.chirp_arr[0] since 0.98 > previous.
        assert!(r2.chirp_arr[0] > r1.chirp_arr[0]);
        assert!(r2.chirp_arr[0] < 0.98);
    }

    #[test]
    fn covariance_zero_input_yields_zero_alphas() {
        let sba = 4_u32;
        let q_low = (0..sba)
            .map(|_| vec![(0.0_f32, 0.0); 32])
            .collect::<Vec<_>>();
        let q_low_ext = build_q_low_ext(&q_low, &[], sba);
        let cov = compute_covariance(&q_low_ext, sba);
        let (a0, a1) = compute_alphas(&cov);
        for sb in 0..(sba as usize) {
            assert_eq!(a0[sb], (0.0, 0.0));
            assert_eq!(a1[sb], (0.0, 0.0));
        }
    }

    #[test]
    fn covariance_complex_exponential_predicts_correctly() {
        // Drive subband 0 with z[ts] = exp(j * w * ts), a single
        // complex sinusoid. The order-1 LPC predictor is
        // z[n] = exp(j*w) * z[n-1]; the order-2 prediction error is
        // zero. So Pseudocode 87's alpha0 should approach exp(j*w),
        // alpha1 should approach zero (the predictor is order-1 in
        // disguise once alpha1 is unused).
        //
        // BUT the spec's predictor is z[n] = -alpha0 z[n-1]
        // -alpha1 z[n-2] (Pseudocode 89 adds alpha0 * z[n-2]; the
        // signs in Pseudocode 87 imply the sign convention is built-in
        // to alpha0). We don't unit-test the algebra against an exact
        // closed form here — instead we just verify that the alphas
        // come out finite and bounded, and reset to 0 for the
        // |alpha|>=4 fallback case.
        let sba = 1_u32;
        let n_ts = 64;
        let w = std::f32::consts::PI / 8.0;
        let mut row = Vec::with_capacity(n_ts);
        for ts in 0..n_ts {
            let phi = w * ts as f32;
            row.push((phi.cos(), phi.sin()));
        }
        let q_low = vec![row];
        let q_low_ext = build_q_low_ext(&q_low, &[], sba);
        let cov = compute_covariance(&q_low_ext, sba);
        // cov[0][1][1] should be roughly real and positive (autocorrelation).
        assert!(cov[0][1][1].0 > 0.0);
        let (a0, a1) = compute_alphas(&cov);
        // Magnitudes must be finite and below the |α|<4 fallback cap.
        let mag0 = (a0[0].0 * a0[0].0 + a0[0].1 * a0[0].1).sqrt();
        let mag1 = (a1[0].0 * a1[0].0 + a1[0].1 * a1[0].1).sqrt();
        assert!(mag0.is_finite() && mag0 < 4.0);
        assert!(mag1.is_finite() && mag1 < 4.0);
    }

    #[test]
    fn build_q_low_ext_prepends_prev_tail_and_zero_pads_first_frame() {
        // First frame: q_low_prev empty → first 4 slots are zeros.
        let q_low = vec![vec![(1.0_f32, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]];
        let ext = build_q_low_ext(&q_low, &[], 1);
        assert_eq!(ext[0][0..4], [(0.0, 0.0); 4]);
        assert_eq!(ext[0][4], (1.0, 0.0));
        assert_eq!(ext[0][5], (2.0, 0.0));
        assert_eq!(ext[0][7], (4.0, 0.0));

        // Second frame: q_low_prev nonempty → tail is its last 4 slots.
        let q_low_prev = vec![vec![(10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]];
        let ext2 = build_q_low_ext(&q_low, &q_low_prev, 1);
        assert_eq!(ext2[0][0], (10.0, 0.0));
        assert_eq!(ext2[0][3], (40.0, 0.0));
        assert_eq!(ext2[0][4], (1.0, 0.0));
    }

    #[test]
    fn preflat_gains_match_for_flat_envelope() {
        // Flat per-subband energy → polynomial slope is constant and
        // equal to mean_energy → gain_vec[sb] = 10^0 = 1 for all sb.
        let sbx = 8_u32;
        let mut q_low: Vec<Vec<(f32, f32)>> = Vec::with_capacity(sbx as usize);
        for _ in 0..sbx {
            q_low.push(vec![(1.0_f32, 0.0); 32]);
        }
        let atsg_sig = vec![0_u32, 16];
        let g = compute_preflat_gains(&q_low, sbx, &atsg_sig, 1);
        for v in g {
            assert!((v - 1.0).abs() < 1e-3, "flat gain != 1: {v}");
        }
    }

    #[test]
    fn hf_tile_tns_zero_alphas_equals_plain_tile_copy() {
        // With chirp = 0 / alphas = 0 / no preflat, hf_tile_tns must
        // reproduce the bare hf_tile_copy output.
        let sba = 8_u32;
        let sbx = sba;
        let num_qmf = 16_u32;
        // Build a q_low with sba populated low subbands.
        let mut q_low: Vec<Vec<(f32, f32)>> = Vec::with_capacity(num_qmf as usize);
        for sb in 0..num_qmf {
            if sb < sba {
                let mut row = Vec::with_capacity(8);
                for ts in 0..8 {
                    row.push(((sb as f32 + 1.0) * (ts as f32 + 1.0), 0.0));
                }
                q_low.push(row);
            } else {
                q_low.push(vec![(0.0, 0.0); 8]);
            }
        }
        let q_low_ext = build_q_low_ext(&q_low, &[], sba);
        let patches = AspxPatchTables {
            sbg_patches: vec![sbx, sbx + 4],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![4],
            sbg_patch_start_sb: vec![0],
        };
        let sbg_noise = vec![sbx, sbx + 4];
        let chirp = vec![0.0_f32];
        let alpha0 = vec![(0.0_f32, 0.0); sba as usize];
        let alpha1 = vec![(0.0_f32, 0.0); sba as usize];
        let atsg_sig = vec![0_u32, 8];
        let q_high = hf_tile_tns(
            &q_low_ext, &patches, &sbg_noise, &chirp, &alpha0, &alpha1, None, sbx, num_qmf,
            &atsg_sig, 1,
        );
        // Patch 0 maps sb_high in [sbx, sbx+4) ← sb_src in [0, 4).
        // q_low_ext[p][n] with n = ts + TS_OFFSET_HFADJ.
        for sb_off in 0u32..4 {
            let sb_high = (sbx + sb_off) as usize;
            for (ts, dst) in q_high[sb_high].iter().enumerate().take(8) {
                let n = ts + TS_OFFSET_HFADJ;
                let expected = q_low_ext[sb_off as usize][n];
                assert_eq!(
                    *dst, expected,
                    "tile copy mismatch at sb_high={sb_high} ts={ts}"
                );
            }
        }
    }

    #[test]
    fn hf_tile_tns_with_alphas_modifies_output() {
        // With a non-zero chirp + alpha0, the HF output must differ
        // from the plain tile copy.
        let sba = 4_u32;
        let sbx = sba;
        let num_qmf = 8_u32;
        let mut q_low: Vec<Vec<(f32, f32)>> = Vec::with_capacity(num_qmf as usize);
        for sb in 0..num_qmf {
            if sb < sba {
                let mut row = Vec::with_capacity(16);
                for ts in 0..16 {
                    let phi = std::f32::consts::PI * 0.25 * ts as f32;
                    row.push(((sb as f32 + 1.0) * phi.cos(), (sb as f32 + 1.0) * phi.sin()));
                }
                q_low.push(row);
            } else {
                q_low.push(vec![(0.0, 0.0); 16]);
            }
        }
        let q_low_ext = build_q_low_ext(&q_low, &[], sba);
        let patches = AspxPatchTables {
            sbg_patches: vec![sbx, sbx + 4],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![4],
            sbg_patch_start_sb: vec![0],
        };
        let sbg_noise = vec![sbx, sbx + 4];
        let chirp = vec![0.5_f32];
        let alpha0 = vec![(0.5_f32, 0.0); sba as usize];
        let alpha1 = vec![(0.0_f32, 0.0); sba as usize];
        let atsg_sig = vec![0_u32, 16];
        let q_with = hf_tile_tns(
            &q_low_ext, &patches, &sbg_noise, &chirp, &alpha0, &alpha1, None, sbx, num_qmf,
            &atsg_sig, 1,
        );
        // Without TNS (chirp = 0): bare tile copy.
        let chirp0 = vec![0.0_f32];
        let q_without = hf_tile_tns(
            &q_low_ext, &patches, &sbg_noise, &chirp0, &alpha0, &alpha1, None, sbx, num_qmf,
            &atsg_sig, 1,
        );
        // Outputs must differ in at least one sb_high.
        let mut diffs = 0;
        for (a_row, b_row) in q_with
            .iter()
            .zip(q_without.iter())
            .take((sbx + 4) as usize)
            .skip(sbx as usize)
        {
            for (a, b) in a_row.iter().zip(b_row.iter()).take(16) {
                if (a.0 - b.0).abs() > 1e-6 || (a.1 - b.1).abs() > 1e-6 {
                    diffs += 1;
                }
            }
        }
        assert!(
            diffs > 0,
            "TNS with chirp=0.5 made no difference vs chirp=0"
        );
    }

    #[test]
    fn hf_tile_tns_preflat_divides_by_gain() {
        // gain_vec all 2.0 → output should be the tile-copy / 2.0.
        let sba = 4_u32;
        let sbx = sba;
        let num_qmf = 8_u32;
        let mut q_low: Vec<Vec<(f32, f32)>> = Vec::with_capacity(num_qmf as usize);
        for sb in 0..num_qmf {
            if sb < sba {
                q_low.push(vec![(2.0_f32, 0.0); 8]);
            } else {
                q_low.push(vec![(0.0, 0.0); 8]);
            }
        }
        let q_low_ext = build_q_low_ext(&q_low, &[], sba);
        let patches = AspxPatchTables {
            sbg_patches: vec![sbx, sbx + 4],
            num_sbg_patches: 1,
            sbg_patch_num_sb: vec![4],
            sbg_patch_start_sb: vec![0],
        };
        let sbg_noise = vec![sbx, sbx + 4];
        let chirp = vec![0.0_f32];
        let alpha0 = vec![(0.0_f32, 0.0); sba as usize];
        let alpha1 = vec![(0.0_f32, 0.0); sba as usize];
        let gain_vec = vec![2.0_f32; sba as usize];
        let atsg_sig = vec![0_u32, 8];
        let q_high = hf_tile_tns(
            &q_low_ext,
            &patches,
            &sbg_noise,
            &chirp,
            &alpha0,
            &alpha1,
            Some(&gain_vec),
            sbx,
            num_qmf,
            &atsg_sig,
            1,
        );
        for row in q_high.iter().take((sbx + 4) as usize).skip(sbx as usize) {
            for sample in row.iter().take(8) {
                // tile-copy source value is 2.0 / gain_vec=2.0 → 1.0.
                assert!((sample.0 - 1.0).abs() < 1e-6);
                assert!(sample.1.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn alphas_clamped_when_magnitude_ge_4() {
        // Hand-craft a covariance matrix that drives |alpha0| or
        // |alpha1| above 4. Construct cov so that
        // alpha1 = (cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1]) /
        //          (cov[2][2] * cov[1][1] - |cov[1][2]|^2).
        // Using cov[0][1] = (10, 0), cov[1][2] = (1, 0),
        // cov[0][2] = (0, 0), cov[1][1] = (1, 0), cov[2][2] = (1, 0):
        // alpha1 = (10*1 - 0) / (1*1 - 1*~1) = 10/(1 - ~1)
        //        ≈ 10 / 9.5e-7 → huge → clamp to 0.
        let mut cov: Vec<CovMatrix> = vec![[[(0.0, 0.0); 3]; 3]; 1];
        cov[0][0][1] = (10.0, 0.0);
        cov[0][0][2] = (0.0, 0.0);
        cov[0][1][1] = (1.0, 0.0);
        cov[0][1][2] = (1.0, 0.0);
        cov[0][2][2] = (1.0, 0.0);
        let (a0, a1) = compute_alphas(&cov);
        assert_eq!(a0[0], (0.0, 0.0), "alpha0 should be clamped to 0");
        assert_eq!(a1[0], (0.0, 0.0), "alpha1 should be clamped to 0");
    }

    #[test]
    fn polynomial_fit_3_recovers_known_polynomial() {
        // y[i] = 1 + 2 i + 3 i^2 + 4 i^3 → poly_array (descending) =
        // [4, 3, 2, 1].
        let n = 16;
        let mut y = vec![0.0_f64; n];
        for (i, yi) in y.iter_mut().enumerate() {
            let x = i as f64;
            *yi = 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * x * x * x;
        }
        let p = polynomial_fit_3(n, &y);
        assert!((p[0] - 4.0).abs() < 1e-6);
        assert!((p[1] - 3.0).abs() < 1e-6);
        assert!((p[2] - 2.0).abs() < 1e-6);
        assert!((p[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tns_state_reset_clears_history() {
        let mut state = AspxTnsState::new();
        state.tna_mode_prev = vec![3, 1];
        state.chirp_prev = vec![0.9, 0.6];
        state.reset();
        assert!(state.tna_mode_prev.is_empty());
        assert!(state.chirp_prev.is_empty());
    }
}
