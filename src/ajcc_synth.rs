//! A-JCC synthesis — ETSI TS 103 190-2 §5.6.3.3 interpolation,
//! §5.6.3.4 decorrelation / transient ducking and §5.6.3.5
//! reconstruction of the output channels (full decoding mode).
//!
//! Everything downstream of the [`crate::ajcc`] parameter decode:
//!
//! * **`interpolate_ajcc()`** (Table 32) — smooth (linear ramps toward
//!   each parameter set) or steep (instantaneous switch at
//!   `ajcc_param_timeslot`) interpolation per QMF `(sb, ts)`, against
//!   the previous frame's tail values `ajcc_param_prev[sb]`
//!   (Table 33 initialisation via [`update_param_prev`], zeros on the
//!   first frame).
//! * **`input_sig_pre_modification()`** (Table 36) — the core-mode
//!   crossfade that blends the two decorrelator feed pairs when
//!   `ajcc_core_mode` flips between frames.
//! * **`ajcc_module_1()`** (Table 37) — the `b_5fronts` 3-output
//!   module: one core channel + two decorrelator feeds → front /
//!   top-front / height triplet.
//! * **`ajcc_module_2()`** (Table 38) — the core-layout 5-output
//!   module with alpha-based mid/side splitting (core mode 0) or
//!   top-channel prediction (core mode 1) plus beta wet terms.
//! * **`ajcc_full_decode()`** (Table 35) — the complete full-decoding
//!   driver: input scaling by `2 + 1/√2`, the per-instance D0/D1/D2
//!   decorrelator assignment, Part 1 §5.7.7.4.3 transient ducking of
//!   every decorrelator output, module dispatch for both layouts,
//!   `z2 = x2in` centre passthrough and the `√2` scaling of the
//!   surround / back / top outputs.
//!
//! The parameter-band ↔ QMF-subband mapping is the Part 1 A-CPL
//! Table 196/197 mapping ([`crate::acpl::sb_to_pb`]) per §5.6.3.1 — the
//! A-JCC band counts (15 / 12 / 9 / 7, Table 108) are exactly the
//! A-CPL set. Decorrelators and ducker come from
//! [`crate::acpl_synth`] unchanged (§5.6.3.4).

use crate::acpl::sb_to_pb;
use crate::acpl_synth::{
    apply_transient_ducker, compute_p_energy, DecorrelatorId, InputSignalModifier, TransientDucker,
};
use crate::ajcc::AjccFramingData;
use crate::qmf::NUM_QMF_SUBBANDS;
use oxideav_core::{Error, Result};

/// One complex QMF column (64 subbands).
pub type QmfCol = [(f32, f32); NUM_QMF_SUBBANDS];

/// `2 + 1/sqrt(2)` — the Table 35 input scaling.
fn input_scale() -> f64 {
    2.0 + 1.0 / std::f64::consts::SQRT_2
}

/// `interpolate_ajcc(ajcc_param, num_pset, sb, ts)` — §5.6.3.3
/// Table 32.
///
/// * `param[ps][pb]` — this frame's dequantized parameter sets.
/// * `prev[sb]` — `ajcc_param_prev`, the previous frame's tail values
///   addressed per QMF subband (Table 33; all zeros on the first
///   frame).
/// * `framing` — interpolation type, parameter-set count and steep
///   switch slots for this parameter's framing group.
/// * `num_bands` — parameter bands (Table 108).
/// * `num_ts` — `num_qmf_timeslots`.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_ajcc(
    param: &[Vec<f64>],
    prev: &[f64],
    framing: &AjccFramingData,
    num_bands: u32,
    sb: usize,
    ts: usize,
    num_ts: usize,
) -> f64 {
    let pb = sb_to_pb(sb as u32, num_bands) as usize;
    let num_pset = framing.num_param_sets as usize;
    let p = |ps: usize| {
        param
            .get(ps)
            .and_then(|r| r.get(pb))
            .copied()
            .unwrap_or(0.0)
    };
    let prev_sb = prev.get(sb).copied().unwrap_or(0.0);
    if !framing.steep {
        // Smooth interpolation.
        if num_pset == 1 {
            let delta = p(0) - prev_sb;
            prev_sb + (ts as f64 + 1.0) * delta / num_ts as f64
        } else {
            let ts_2 = num_ts / 2;
            if ts < ts_2 {
                let delta = p(0) - prev_sb;
                prev_sb + (ts as f64 + 1.0) * delta / ts_2 as f64
            } else {
                let delta = p(1) - p(0);
                p(0) + (ts as f64 - ts_2 as f64 + 1.0) * delta / (num_ts - ts_2) as f64
            }
        }
    } else {
        // Steep interpolation.
        let slot = |i: usize| framing.param_timeslot.get(i).copied().unwrap_or(0) as usize;
        if num_pset == 1 {
            if ts < slot(0) {
                prev_sb
            } else {
                p(0)
            }
        } else if ts < slot(0) {
            prev_sb
        } else if ts < slot(1) {
            p(0)
        } else {
            p(1)
        }
    }
}

/// Table 33: derive the next frame's `ajcc_param_prev[sb]` from this
/// frame's last parameter set.
pub fn update_param_prev(param: &[Vec<f64>], num_bands: u32) -> Vec<f64> {
    let last = param.last();
    (0..NUM_QMF_SUBBANDS)
        .map(|sb| {
            let pb = sb_to_pb(sb as u32, num_bands) as usize;
            last.and_then(|r| r.get(pb)).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Table 36 `input_sig_pre_modification()` — crossfade the two
/// decorrelator feed pairs across a `ajcc_core_mode` flip. The
/// persistent `core_mode_prev` lives in [`AjccSynthState`]; it is
/// initialised to the current `core_mode` on the first frame per the
/// spec note.
pub fn input_sig_pre_modification(
    in1: &[QmfCol],
    in2: &[QmfCol],
    in3: &[QmfCol],
    in4: &[QmfCol],
    core_mode: bool,
    core_mode_prev: bool,
) -> (Vec<QmfCol>, Vec<QmfCol>) {
    let num_ts = in1.len();
    let (mut g, d) = if core_mode == core_mode_prev {
        (if !core_mode { 1.0f64 } else { 0.0 }, 0.0f64)
    } else if !core_mode {
        (0.0, 1.0 / num_ts as f64)
    } else {
        (1.0, -1.0 / num_ts as f64)
    };
    let mut out1 = Vec::with_capacity(num_ts);
    let mut out2 = Vec::with_capacity(num_ts);
    for ts in 0..num_ts {
        g += d;
        let mut c1 = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        let mut c2 = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
        for sb in 0..NUM_QMF_SUBBANDS {
            let blend = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
                (
                    (g * b.0 as f64 + (1.0 - g) * a.0 as f64) as f32,
                    (g * b.1 as f64 + (1.0 - g) * a.1 as f64) as f32,
                )
            };
            c1[sb] = blend(in1[ts][sb], in2[ts][sb]);
            c2[sb] = blend(in3[ts][sb], in4[ts][sb]);
        }
        out1.push(c1);
        out2.push(c2);
    }
    (out1, out2)
}

/// The per-parameter interpolation context of one module invocation:
/// framing group + previous-frame tails for each derived coefficient
/// track.
struct Track<'a> {
    param: Vec<Vec<f64>>,
    prev: &'a mut Vec<f64>,
}

impl Track<'_> {
    fn interp(
        &self,
        framing: &AjccFramingData,
        num_bands: u32,
        sb: usize,
        ts: usize,
        num_ts: usize,
    ) -> f64 {
        interpolate_ajcc(&self.param, self.prev, framing, num_bands, sb, ts, num_ts)
    }
    fn finish(self, num_bands: u32) {
        *self.prev = update_param_prev(&self.param, num_bands);
    }
}

/// Table 37 `ajcc_module_1()` — the `b_5fronts` module. Inputs: one
/// core channel `x` plus two ducked decorrelator outputs `y0` / `y1`;
/// outputs the `(z_a, z_b, z_c)` channel triple.
///
/// `dry1 / dry2 / wet1 / wet2 / wet3` are `[ps][pb]` dequantized
/// parameter sets sharing one framing group; `prev` carries the nine
/// derived coefficient tracks' `ajcc_param_prev` rows across frames
/// (see [`AjccModule1State`]).
#[allow(clippy::too_many_arguments)]
pub fn ajcc_module_1(
    dry1: &[Vec<f64>],
    dry2: &[Vec<f64>],
    wet1: &[Vec<f64>],
    wet2: &[Vec<f64>],
    wet3: &[Vec<f64>],
    framing: &AjccFramingData,
    num_bands: u32,
    x: &[QmfCol],
    y0: &[QmfCol],
    y1: &[QmfCol],
    state: &mut AjccModule1State,
) -> (Vec<QmfCol>, Vec<QmfCol>, Vec<QmfCol>) {
    let num_ts = x.len();
    let num_pset = framing.num_param_sets as usize;
    let nb = num_bands as usize;
    let s = 1.0 / std::f64::consts::SQRT_2;

    // Derived per-(ps, pb) coefficient tracks (Table 37 header loop).
    let derive = |f: &dyn Fn(usize, usize) -> f64| -> Vec<Vec<f64>> {
        (0..num_pset)
            .map(|ps| (0..nb).map(|pb| f(ps, pb)).collect())
            .collect()
    };
    let at = |m: &[Vec<f64>], ps: usize, pb: usize| -> f64 {
        m.get(ps).and_then(|r| r.get(pb)).copied().unwrap_or(0.0)
    };
    let tracks_param: Vec<Vec<Vec<f64>>> = vec![
        derive(&|ps, pb| at(dry1, ps, pb)),
        derive(&|ps, pb| at(dry2, ps, pb)),
        derive(&|ps, pb| 1.0 - at(dry1, ps, pb) - at(dry2, ps, pb)),
        derive(&|ps, pb| s * (at(wet1, ps, pb) + at(wet3, ps, pb))),
        derive(&|ps, pb| s * (at(wet3, ps, pb) + at(wet2, ps, pb))),
        derive(&|ps, pb| -s * at(wet3, ps, pb)),
        derive(&|ps, pb| -s * at(wet2, ps, pb)),
        derive(&|ps, pb| -s * at(wet1, ps, pb)),
        derive(&|ps, pb| -s * at(wet3, ps, pb)),
    ];
    let mut tracks: Vec<Track<'_>> = tracks_param
        .into_iter()
        .zip(state.prev.iter_mut())
        .map(|(param, prev)| Track { param, prev })
        .collect();

    let zero = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
    let mut z0 = zero.clone();
    let mut z1 = zero.clone();
    let mut z2 = zero;
    for sb in 0..NUM_QMF_SUBBANDS {
        for ts in 0..num_ts {
            let iv: Vec<f64> = tracks
                .iter()
                .map(|t| t.interp(framing, num_bands, sb, ts, num_ts))
                .collect();
            let cx = x[ts][sb];
            let c0 = y0[ts][sb];
            let c1 = y1[ts][sb];
            let mix = |d: f64, p_a: f64, p_b: f64| -> (f32, f32) {
                (
                    (d * cx.0 as f64 + p_a * c0.0 as f64 + p_b * c1.0 as f64) as f32,
                    (d * cx.1 as f64 + p_a * c0.1 as f64 + p_b * c1.1 as f64) as f32,
                )
            };
            z0[ts][sb] = mix(iv[0], iv[3], iv[4]);
            z1[ts][sb] = mix(iv[1], iv[5], iv[6]);
            z2[ts][sb] = mix(iv[2], iv[7], iv[8]);
        }
    }
    for t in tracks.drain(..) {
        t.finish(num_bands);
    }
    (z0, z1, z2)
}

/// Table 38 `ajcc_module_2()` — the core-layout module. Inputs: two
/// core channels `x0` / `x1` and three ducked decorrelator outputs;
/// outputs five channels. `core_mode` selects the Table 38 coefficient
/// wiring (mid/side via alpha when 0, top prediction when 1).
#[allow(clippy::too_many_arguments)]
pub fn ajcc_module_2(
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
    dry1: &[Vec<f64>],
    dry2: &[Vec<f64>],
    wet1: &[Vec<f64>],
    wet2: &[Vec<f64>],
    wet3: &[Vec<f64>],
    framing: &AjccFramingData,
    num_bands: u32,
    core_mode: bool,
    x0: &[QmfCol],
    x1: &[QmfCol],
    y0: &[QmfCol],
    y1: &[QmfCol],
    y2: &[QmfCol],
    state: &mut AjccModule2State,
) -> [Vec<QmfCol>; 5] {
    let num_ts = x0.len();
    let num_pset = framing.num_param_sets as usize;
    let nb = num_bands as usize;
    let s = 1.0 / std::f64::consts::SQRT_2;

    let derive = |f: &dyn Fn(usize, usize) -> f64| -> Vec<Vec<f64>> {
        (0..num_pset)
            .map(|ps| (0..nb).map(|pb| f(ps, pb)).collect())
            .collect()
    };
    let at = |m: &[Vec<f64>], ps: usize, pb: usize| -> f64 {
        m.get(ps).and_then(|r| r.get(pb)).copied().unwrap_or(0.0)
    };

    // d0..d9 then w0..w14 in Table 38 order.
    let tracks_param: Vec<Vec<Vec<f64>>> = if !core_mode {
        vec![
            derive(&|ps, pb| (1.0 + at(alpha, ps, pb)) / 2.0), // d0
            derive(&|_, _| 0.0),                               // d1
            derive(&|_, _| 0.0),                               // d2
            derive(&|ps, pb| (1.0 - at(alpha, ps, pb)) / 2.0), // d3
            derive(&|_, _| 0.0),                               // d4
            derive(&|_, _| 0.0),                               // d5
            derive(&|ps, pb| at(dry1, ps, pb)),                // d6
            derive(&|ps, pb| at(dry2, ps, pb)),                // d7
            derive(&|_, _| 0.0),                               // d8
            derive(&|ps, pb| 1.0 - at(dry1, ps, pb) - at(dry2, ps, pb)), // d9
            derive(&|ps, pb| at(beta, ps, pb) / 2.0),          // w0
            derive(&|_, _| 0.0),                               // w1
            derive(&|_, _| 0.0),                               // w2
            derive(&|ps, pb| -at(beta, ps, pb) / 2.0),         // w3
            derive(&|_, _| 0.0),                               // w4
            derive(&|_, _| 0.0),                               // w5
            derive(&|ps, pb| s * (at(wet1, ps, pb) + at(wet3, ps, pb))), // w6
            derive(&|ps, pb| -s * at(wet3, ps, pb)),           // w7
            derive(&|_, _| 0.0),                               // w8
            derive(&|ps, pb| -s * at(wet1, ps, pb)),           // w9
            derive(&|_, _| 0.0),                               // w10
            derive(&|ps, pb| s * (at(wet3, ps, pb) + at(wet2, ps, pb))), // w11
            derive(&|ps, pb| -s * at(wet2, ps, pb)),           // w12
            derive(&|_, _| 0.0),                               // w13
            derive(&|ps, pb| -s * at(wet3, ps, pb)),           // w14
        ]
    } else {
        vec![
            derive(&|ps, pb| at(dry1, ps, pb)),                          // d0
            derive(&|ps, pb| at(dry2, ps, pb)),                          // d1
            derive(&|ps, pb| 1.0 - at(dry1, ps, pb) - at(dry2, ps, pb)), // d2
            derive(&|_, _| 0.0),                                         // d3
            derive(&|_, _| 0.0),                                         // d4
            derive(&|_, _| 0.0),                                         // d5
            derive(&|_, _| 0.0),                                         // d6
            derive(&|_, _| 0.0),                                         // d7
            derive(&|ps, pb| (1.0 + at(alpha, ps, pb)) / 2.0),           // d8
            derive(&|ps, pb| (1.0 - at(alpha, ps, pb)) / 2.0),           // d9
            derive(&|ps, pb| s * (at(wet1, ps, pb) + at(wet3, ps, pb))), // w0
            derive(&|ps, pb| -s * at(wet3, ps, pb)),                     // w1
            derive(&|ps, pb| -s * at(wet1, ps, pb)),                     // w2
            derive(&|_, _| 0.0),                                         // w3
            derive(&|_, _| 0.0),                                         // w4
            derive(&|ps, pb| s * (at(wet3, ps, pb) + at(wet2, ps, pb))), // w5
            derive(&|ps, pb| -s * at(wet2, ps, pb)),                     // w6
            derive(&|ps, pb| -s * at(wet3, ps, pb)),                     // w7
            derive(&|_, _| 0.0),                                         // w8
            derive(&|_, _| 0.0),                                         // w9
            derive(&|_, _| 0.0),                                         // w10
            derive(&|_, _| 0.0),                                         // w11
            derive(&|_, _| 0.0),                                         // w12
            derive(&|ps, pb| at(beta, ps, pb) / 2.0),                    // w13
            derive(&|ps, pb| -at(beta, ps, pb) / 2.0),                   // w14
        ]
    };
    let mut tracks: Vec<Track<'_>> = tracks_param
        .into_iter()
        .zip(state.prev.iter_mut())
        .map(|(param, prev)| Track { param, prev })
        .collect();

    let mut z: [Vec<QmfCol>; 5] =
        std::array::from_fn(|_| vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts]);
    for sb in 0..NUM_QMF_SUBBANDS {
        for ts in 0..num_ts {
            let iv: Vec<f64> = tracks
                .iter()
                .map(|t| t.interp(framing, num_bands, sb, ts, num_ts))
                .collect();
            let c_x0 = x0[ts][sb];
            let c_x1 = x1[ts][sb];
            let c_y0 = y0[ts][sb];
            let c_y1 = y1[ts][sb];
            let c_y2 = y2[ts][sb];
            for o in 0..5 {
                // Table 38 output rows: z_o uses d_o, d_{o+5}, w_o,
                // w_{o+5}, w_{o+10}.
                let re = iv[o] * c_x0.0 as f64
                    + iv[o + 5] * c_x1.0 as f64
                    + iv[10 + o] * c_y0.0 as f64
                    + iv[10 + o + 5] * c_y1.0 as f64
                    + iv[10 + o + 10] * c_y2.0 as f64;
                let im = iv[o] * c_x0.1 as f64
                    + iv[o + 5] * c_x1.1 as f64
                    + iv[10 + o] * c_y0.1 as f64
                    + iv[10 + o + 5] * c_y1.1 as f64
                    + iv[10 + o + 10] * c_y2.1 as f64;
                z[o][ts][sb] = (re as f32, im as f32);
            }
        }
    }
    for t in tracks.drain(..) {
        t.finish(num_bands);
    }
    z
}

/// Interpolation tails for the nine Table 37 coefficient tracks.
#[derive(Clone, Debug)]
pub struct AjccModule1State {
    prev: Vec<Vec<f64>>,
}

impl Default for AjccModule1State {
    fn default() -> Self {
        AjccModule1State {
            prev: vec![vec![0.0; NUM_QMF_SUBBANDS]; 9],
        }
    }
}

/// Interpolation tails for the 25 Table 38 coefficient tracks
/// (d0..d9, w0..w14).
#[derive(Clone, Debug)]
pub struct AjccModule2State {
    prev: Vec<Vec<f64>>,
}

impl Default for AjccModule2State {
    fn default() -> Self {
        AjccModule2State {
            prev: vec![vec![0.0; NUM_QMF_SUBBANDS]; 25],
        }
    }
}

/// Persistent A-JCC synthesis state across AC-4 frames: decorrelator
/// histories, per-instance transient duckers, module interpolation
/// tails and the Table 36 `ajcc_core_mode_prev`.
pub struct AjccSynthState {
    decorr: Vec<InputSignalModifier>,
    duckers: Vec<TransientDucker>,
    mod1: Vec<AjccModule1State>,
    mod2: Vec<AjccModule2State>,
    core_mode_prev: Option<bool>,
    b_5fronts: bool,
}

impl AjccSynthState {
    /// Fresh state for the given layout. The decorrelator instances
    /// follow the Table 35 D0/D1/D2 assignment comments:
    /// `b_5fronts`: `[D0, D2, D0, D2, D1, D2, D1, D2]` (8 instances);
    /// otherwise `[D0, D2, D1, D0, D2, D1]` (6 instances).
    pub fn new(b_5fronts: bool) -> Self {
        let ids: &[DecorrelatorId] = if b_5fronts {
            &[
                DecorrelatorId::D0,
                DecorrelatorId::D2,
                DecorrelatorId::D0,
                DecorrelatorId::D2,
                DecorrelatorId::D1,
                DecorrelatorId::D2,
                DecorrelatorId::D1,
                DecorrelatorId::D2,
            ]
        } else {
            &[
                DecorrelatorId::D0,
                DecorrelatorId::D2,
                DecorrelatorId::D1,
                DecorrelatorId::D0,
                DecorrelatorId::D2,
                DecorrelatorId::D1,
            ]
        };
        AjccSynthState {
            decorr: ids.iter().map(|&d| InputSignalModifier::new(d)).collect(),
            duckers: (0..ids.len()).map(|_| TransientDucker::new()).collect(),
            mod1: vec![AjccModule1State::default(); if b_5fronts { 4 } else { 0 }],
            mod2: vec![AjccModule2State::default(); if b_5fronts { 0 } else { 2 }],
            core_mode_prev: None,
            b_5fronts,
        }
    }
}

/// Dequantized A-JCC parameters for one frame in the roster order of
/// [`crate::ajcc::AjccDecoded`] (`[set][ps][pb]`), plus the framing
/// groups and layout flags — everything Table 35 needs.
pub struct AjccFrameParams<'a> {
    /// `b_5fronts` layout flag.
    pub b_5fronts: bool,
    /// `ajcc_core_mode` (core layout only).
    pub core_mode: bool,
    /// `num_bands` (Table 108).
    pub num_bands: u32,
    /// Framing groups in transmission order (4 or 2).
    pub framing: &'a [AjccFramingData],
    /// Dequantized alpha SETs (2 for the core layout, unused for
    /// 5-fronts).
    pub alpha_dq: &'a [Vec<Vec<f64>>],
    /// Dequantized beta SETs.
    pub beta_dq: &'a [Vec<Vec<f64>>],
    /// Dequantized dry SETs (8 or 4).
    pub dry_dq: &'a [Vec<Vec<f64>>],
    /// Dequantized wet SETs (12 or 6).
    pub wet_dq: &'a [Vec<Vec<f64>>],
}

fn scale_input(x: &[QmfCol]) -> Vec<QmfCol> {
    let k = input_scale();
    x.iter()
        .map(|col| {
            let mut out = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
            for sb in 0..NUM_QMF_SUBBANDS {
                out[sb] = ((k * col[sb].0 as f64) as f32, (k * col[sb].1 as f64) as f32);
            }
            out
        })
        .collect()
}

fn scale_by_sqrt2(z: &mut [QmfCol]) {
    let k = std::f64::consts::SQRT_2;
    for col in z {
        for v in col.iter_mut() {
            v.0 = (k * v.0 as f64) as f32;
            v.1 = (k * v.1 as f64) as f32;
        }
    }
}

/// Run one decorrelator instance over a whole matrix.
fn decorrelate(inst: &mut InputSignalModifier, x: &[QmfCol]) -> Vec<QmfCol> {
    x.iter()
        .map(|col| {
            let mut out = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
            for sb in 0..NUM_QMF_SUBBANDS as u32 {
                out[sb as usize] = inst.process_sample(sb, col[sb as usize]);
            }
            out
        })
        .collect()
}

/// Part 1 §5.7.7.4.3 ducking of one decorrelator output matrix.
fn duck(ducker: &mut TransientDucker, u: &[QmfCol], num_bands: u32) -> Vec<QmfCol> {
    u.iter()
        .map(|col| {
            let e = compute_p_energy(col, num_bands);
            let g = ducker.update(&e);
            apply_transient_ducker(col, &g, num_bands)
        })
        .collect()
}

/// §5.6.3.5.2 full decoding mode (Table 35): reconstruct the output
/// channels from the five core QMF inputs `x[0..5]` and the decoded
/// frame parameters. Returns 13 output matrices for `b_5fronts`
/// (`z0..z12`) or 11 for the core layout (`z3` / `z4` empty).
///
/// Output order matches the spec addressing: `[L, R, C, Lscr, Rscr,
/// Ls, Rs, Lb, Rb, Ltf/Ltm, Rtf/Rtm, Ltb, Rtb]`.
pub fn ajcc_full_decode(
    x: &[&[QmfCol]; 5],
    params: &AjccFrameParams<'_>,
    state: &mut AjccSynthState,
) -> Result<Vec<Vec<QmfCol>>> {
    if state.b_5fronts != params.b_5fronts {
        return Err(Error::invalid("ac4: A-JCC synth state layout mismatch"));
    }
    let num_ts = x[0].len();
    if x.iter().any(|m| m.len() != num_ts) {
        return Err(Error::invalid("ac4: A-JCC input timeslot mismatch"));
    }
    let nb = params.num_bands;

    let x0in = scale_input(x[0]);
    let x1in = scale_input(x[1]);
    let x2in = scale_input(x[2]);
    let x3in = scale_input(x[3]);
    let x4in = scale_input(x[4]);

    let num_z = 13;
    let mut z: Vec<Vec<QmfCol>> = vec![Vec::new(); num_z];

    if params.b_5fronts {
        if params.framing.len() != 4 || params.dry_dq.len() != 8 || params.wet_dq.len() != 12 {
            return Err(Error::invalid("ac4: A-JCC 5-fronts roster mismatch"));
        }
        // u0..u7 per the Table 35 D-assignments.
        let feeds: [&[QmfCol]; 8] = [&x0in, &x0in, &x1in, &x1in, &x3in, &x3in, &x4in, &x4in];
        let mut y: Vec<Vec<QmfCol>> = Vec::with_capacity(8);
        for (i, feed) in feeds.iter().enumerate() {
            let u = decorrelate(&mut state.decorr[i], feed);
            y.push(duck(&mut state.duckers[i], &u, nb));
        }
        // Module dispatch: (z0, z9, z3), (z1, z10, z4), (z5, z7, z11),
        // (z6, z8, z12).
        #[allow(clippy::type_complexity)]
        let calls: [(usize, [usize; 3], usize, &[QmfCol], usize, usize); 4] = [
            (0, [0, 9, 3], 0, &x0in, 0, 1),
            (1, [1, 10, 4], 1, &x1in, 2, 3),
            (2, [5, 7, 11], 2, &x3in, 4, 5),
            (3, [6, 8, 12], 3, &x4in, 6, 7),
        ];
        for (m, zi, fi, xin, ya, yb) in calls {
            let (za, zb, zc) = ajcc_module_1(
                &params.dry_dq[2 * m],
                &params.dry_dq[2 * m + 1],
                &params.wet_dq[3 * m],
                &params.wet_dq[3 * m + 1],
                &params.wet_dq[3 * m + 2],
                &params.framing[fi],
                nb,
                xin,
                &y[ya],
                &y[yb],
                &mut state.mod1[m],
            );
            z[zi[0]] = za;
            z[zi[1]] = zb;
            z[zi[2]] = zc;
        }
    } else {
        if params.framing.len() != 2
            || params.dry_dq.len() != 4
            || params.wet_dq.len() != 6
            || params.alpha_dq.len() != 2
            || params.beta_dq.len() != 2
        {
            return Err(Error::invalid("ac4: A-JCC core-layout roster mismatch"));
        }
        let core_mode_prev = state.core_mode_prev.unwrap_or(params.core_mode);
        let (w1in, w2in) = input_sig_pre_modification(
            &x0in,
            &x3in,
            &x1in,
            &x4in,
            params.core_mode,
            core_mode_prev,
        );
        state.core_mode_prev = Some(params.core_mode);

        let feeds: [&[QmfCol]; 6] = [&x0in, &w1in, &x3in, &x1in, &w2in, &x4in];
        let mut y: Vec<Vec<QmfCol>> = Vec::with_capacity(6);
        for (i, feed) in feeds.iter().enumerate() {
            let u = decorrelate(&mut state.decorr[i], feed);
            y.push(duck(&mut state.duckers[i], &u, nb));
        }
        // (z0, z5, z7, z9, z11) and (z1, z6, z8, z10, z12).
        #[allow(clippy::type_complexity)]
        let calls: [(usize, [usize; 5], usize, &[QmfCol], &[QmfCol], [usize; 3]); 2] = [
            (0, [0, 5, 7, 9, 11], 0, &x0in, &x3in, [0, 1, 2]),
            (1, [1, 6, 8, 10, 12], 1, &x1in, &x4in, [3, 4, 5]),
        ];
        for (m, zi, fi, xa, xb, ys) in calls {
            let zs = ajcc_module_2(
                &params.alpha_dq[m],
                &params.beta_dq[m],
                &params.dry_dq[2 * m],
                &params.dry_dq[2 * m + 1],
                &params.wet_dq[3 * m],
                &params.wet_dq[3 * m + 1],
                &params.wet_dq[3 * m + 2],
                &params.framing[fi],
                nb,
                params.core_mode,
                xa,
                xb,
                &y[ys[0]],
                &y[ys[1]],
                &y[ys[2]],
                &mut state.mod2[m],
            );
            for (k, zm) in zs.into_iter().enumerate() {
                z[zi[k]] = zm;
            }
        }
    }

    // z2 = x2in (centre passthrough); √2 gain on z5..z12.
    z[2] = x2in;
    for zi in z.iter_mut().skip(5) {
        scale_by_sqrt2(zi);
    }
    Ok(z)
}

// ---------------------------------------------------------------------
// §5.6.3.5.3 — A-JCC core decoding mode (Tables 39-41)
// ---------------------------------------------------------------------

/// Interpolation tails for the twelve Table 40 / 41 coefficient
/// tracks (d0..d5, w0..w5).
#[derive(Clone, Debug)]
pub struct AjccModule34State {
    prev: Vec<Vec<f64>>,
}

impl Default for AjccModule34State {
    fn default() -> Self {
        AjccModule34State {
            prev: vec![vec![0.0; NUM_QMF_SUBBANDS]; 12],
        }
    }
}

/// Table 40 `ajcc_module_3()` — the `b_5fronts` core-decoding module:
/// two core channels (front / back) + two ducked decorrelator outputs
/// → three output channels. Tracks d0..d2 / w0..w2 follow the *front*
/// framing group, d3..d5 / w3..w5 the *back* one.
#[allow(clippy::too_many_arguments)]
pub fn ajcc_module_3(
    _dry1f: &[Vec<f64>],
    dry2f: &[Vec<f64>],
    // Table 40's front rows only combine wet2f/wet3f and its back rows
    // only wet1b/wet3b; the full quintuple signature is kept for parity
    // with the spec's parameter listing.
    _wet1f: &[Vec<f64>],
    wet2f: &[Vec<f64>],
    wet3f: &[Vec<f64>],
    dry1b: &[Vec<f64>],
    dry2b: &[Vec<f64>],
    wet1b: &[Vec<f64>],
    _wet2b: &[Vec<f64>],
    wet3b: &[Vec<f64>],
    framing_f: &AjccFramingData,
    framing_b: &AjccFramingData,
    num_bands: u32,
    x0: &[QmfCol],
    x1: &[QmfCol],
    y0: &[QmfCol],
    y1: &[QmfCol],
    state: &mut AjccModule34State,
) -> (Vec<QmfCol>, Vec<QmfCol>, Vec<QmfCol>) {
    let num_ts = x0.len();
    let nb = num_bands as usize;
    let at = |m: &[Vec<f64>], ps: usize, pb: usize| -> f64 {
        m.get(ps).and_then(|r| r.get(pb)).copied().unwrap_or(0.0)
    };
    let derive = |nps: u32, f: &dyn Fn(usize, usize) -> f64| -> Vec<Vec<f64>> {
        (0..nps as usize)
            .map(|ps| (0..nb).map(|pb| f(ps, pb)).collect())
            .collect()
    };
    let wf = |ps: usize, pb: usize| -> f64 {
        let w3 = at(wet3f, ps, pb);
        let w2 = at(wet2f, ps, pb);
        (0.5 * w3 * w3 + 0.5 * w2 * w2).sqrt()
    };
    let wb = |ps: usize, pb: usize| -> f64 {
        let w1 = at(wet1b, ps, pb);
        let w3 = at(wet3b, ps, pb);
        (0.5 * w1 * w1 + 0.5 * w3 * w3).sqrt()
    };
    let nf = framing_f.num_param_sets;
    let nbk = framing_b.num_param_sets;
    // Track order: d0..d5 then w0..w5; front group = indices
    // {0,1,2,6,7,8}, back group = {3,4,5,9,10,11}.
    let tracks_param: Vec<(Vec<Vec<f64>>, &AjccFramingData)> = vec![
        (derive(nf, &|ps, pb| 1.0 - at(dry2f, ps, pb)), framing_f), // d0
        (derive(nf, &|_, _| 0.0), framing_f),                       // d1
        (derive(nf, &|ps, pb| at(dry2f, ps, pb)), framing_f),       // d2
        (derive(nbk, &|_, _| 0.0), framing_b),                      // d3
        (
            derive(nbk, &|ps, pb| at(dry1b, ps, pb) + at(dry2b, ps, pb)),
            framing_b,
        ), // d4
        (
            derive(nbk, &|ps, pb| 1.0 - at(dry1b, ps, pb) - at(dry2b, ps, pb)),
            framing_b,
        ), // d5
        (derive(nf, &|ps, pb| -wf(ps, pb)), framing_f),             // w0
        (derive(nf, &|_, _| 0.0), framing_f),                       // w1
        (derive(nf, &|ps, pb| wf(ps, pb)), framing_f),              // w2
        (derive(nbk, &|_, _| 0.0), framing_b),                      // w3
        (derive(nbk, &|ps, pb| -wb(ps, pb)), framing_b),            // w4
        (derive(nbk, &|ps, pb| wb(ps, pb)), framing_b),             // w5
    ];
    module_34_mix(tracks_param, num_bands, x0, x1, y0, y1, state, num_ts)
}

/// Table 41 `ajcc_module_4()` — the core-layout core-decoding module
/// (alpha mid/side when `core_mode == 0`, top fold-down when 1).
#[allow(clippy::too_many_arguments)]
pub fn ajcc_module_4(
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
    dry1: &[Vec<f64>],
    dry2: &[Vec<f64>],
    wet1: &[Vec<f64>],
    wet2: &[Vec<f64>],
    wet3: &[Vec<f64>],
    framing: &AjccFramingData,
    num_bands: u32,
    core_mode: bool,
    x0: &[QmfCol],
    x1: &[QmfCol],
    y0: &[QmfCol],
    y1: &[QmfCol],
    state: &mut AjccModule34State,
) -> (Vec<QmfCol>, Vec<QmfCol>, Vec<QmfCol>) {
    let num_ts = x0.len();
    let nb = num_bands as usize;
    let nps = framing.num_param_sets;
    let at = |m: &[Vec<f64>], ps: usize, pb: usize| -> f64 {
        m.get(ps).and_then(|r| r.get(pb)).copied().unwrap_or(0.0)
    };
    let derive = |f: &dyn Fn(usize, usize) -> f64| -> Vec<Vec<f64>> {
        (0..nps as usize)
            .map(|ps| (0..nb).map(|pb| f(ps, pb)).collect())
            .collect()
    };
    let rows: Vec<Vec<Vec<f64>>> = if !core_mode {
        let w45 = |ps: usize, pb: usize| -> f64 {
            let w1 = at(wet1, ps, pb);
            let w3 = at(wet3, ps, pb);
            (0.5 * w1 * w1 + 0.5 * w3 * w3).sqrt()
        };
        vec![
            derive(&|ps, pb| (1.0 + at(alpha, ps, pb)) / 2.0), // d0
            derive(&|_, _| 0.0),                               // d1
            derive(&|ps, pb| (1.0 - at(alpha, ps, pb)) / 2.0), // d2
            derive(&|_, _| 0.0),                               // d3
            derive(&|ps, pb| at(dry1, ps, pb) + at(dry2, ps, pb)), // d4
            derive(&|ps, pb| 1.0 - at(dry1, ps, pb) - at(dry2, ps, pb)), // d5
            derive(&|ps, pb| at(beta, ps, pb) / 2.0),          // w0
            derive(&|_, _| 0.0),                               // w1
            derive(&|ps, pb| -at(beta, ps, pb) / 2.0),         // w2
            derive(&|_, _| 0.0),                               // w3
            derive(&|ps, pb| -w45(ps, pb)),                    // w4
            derive(&|ps, pb| w45(ps, pb)),                     // w5
        ]
    } else {
        let w01 = |ps: usize, pb: usize| -> f64 {
            let a = at(wet1, ps, pb) + at(wet3, ps, pb);
            let b = at(wet3, ps, pb) + at(wet2, ps, pb);
            (0.5 * a * a + 0.5 * b * b).sqrt()
        };
        vec![
            derive(&|ps, pb| at(dry1, ps, pb)),       // d0
            derive(&|ps, pb| 1.0 - at(dry1, ps, pb)), // d1
            derive(&|_, _| 0.0),                      // d2
            derive(&|_, _| 0.0),                      // d3
            derive(&|_, _| 0.0),                      // d4
            derive(&|_, _| 1.0),                      // d5
            derive(&|ps, pb| w01(ps, pb)),            // w0
            derive(&|ps, pb| -w01(ps, pb)),           // w1
            derive(&|_, _| 0.0),                      // w2
            derive(&|_, _| 0.0),                      // w3
            derive(&|_, _| 0.0),                      // w4
            derive(&|_, _| 0.0),                      // w5
        ]
    };
    let tracks_param: Vec<(Vec<Vec<f64>>, &AjccFramingData)> =
        rows.into_iter().map(|p| (p, framing)).collect();
    module_34_mix(tracks_param, num_bands, x0, x1, y0, y1, state, num_ts)
}

/// Shared Table 40 / 41 mixing loop: interpolate the twelve tracks per
/// `(sb, ts)` and form the three output rows
/// `z_r = d_r·x0 + d_{r+3}·x1 + w_r·y0 + w_{r+3}·y1`.
#[allow(clippy::too_many_arguments)]
fn module_34_mix(
    tracks_param: Vec<(Vec<Vec<f64>>, &AjccFramingData)>,
    num_bands: u32,
    x0: &[QmfCol],
    x1: &[QmfCol],
    y0: &[QmfCol],
    y1: &[QmfCol],
    state: &mut AjccModule34State,
    num_ts: usize,
) -> (Vec<QmfCol>, Vec<QmfCol>, Vec<QmfCol>) {
    let mut tracks: Vec<(Track<'_>, &AjccFramingData)> = tracks_param
        .into_iter()
        .zip(state.prev.iter_mut())
        .map(|((param, fd), prev)| (Track { param, prev }, fd))
        .collect();

    let zero = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
    let mut z0 = zero.clone();
    let mut z1 = zero.clone();
    let mut z2 = zero;
    for sb in 0..NUM_QMF_SUBBANDS {
        for ts in 0..num_ts {
            let iv: Vec<f64> = tracks
                .iter()
                .map(|(t, fd)| t.interp(fd, num_bands, sb, ts, num_ts))
                .collect();
            let cx0 = x0[ts][sb];
            let cx1 = x1[ts][sb];
            let cy0 = y0[ts][sb];
            let cy1 = y1[ts][sb];
            let mix = |r: usize| -> (f32, f32) {
                (
                    (iv[r] * cx0.0 as f64
                        + iv[r + 3] * cx1.0 as f64
                        + iv[6 + r] * cy0.0 as f64
                        + iv[6 + r + 3] * cy1.0 as f64) as f32,
                    (iv[r] * cx0.1 as f64
                        + iv[r + 3] * cx1.1 as f64
                        + iv[6 + r] * cy0.1 as f64
                        + iv[6 + r + 3] * cy1.1 as f64) as f32,
                )
            };
            z0[ts][sb] = mix(0);
            z1[ts][sb] = mix(1);
            z2[ts][sb] = mix(2);
        }
    }
    for (t, _) in tracks.drain(..) {
        t.finish(num_bands);
    }
    (z0, z1, z2)
}

/// Persistent A-JCC *core decoding mode* synthesis state: the four
/// Table 39 decorrelator instances (`x0in D0, x3in D2, x1in D0,
/// x4in D2`), their duckers and the two module track states.
pub struct AjccCoreSynthState {
    decorr: Vec<InputSignalModifier>,
    duckers: Vec<TransientDucker>,
    modules: Vec<AjccModule34State>,
}

impl AjccCoreSynthState {
    /// Fresh core-decoding state (layout-independent: both layouts use
    /// four instances and two modules).
    pub fn new() -> Self {
        let ids = [
            DecorrelatorId::D0,
            DecorrelatorId::D2,
            DecorrelatorId::D0,
            DecorrelatorId::D2,
        ];
        AjccCoreSynthState {
            decorr: ids.iter().map(|&d| InputSignalModifier::new(d)).collect(),
            duckers: (0..ids.len()).map(|_| TransientDucker::new()).collect(),
            modules: vec![AjccModule34State::default(); 2],
        }
    }
}

impl Default for AjccCoreSynthState {
    fn default() -> Self {
        Self::new()
    }
}

/// §5.6.3.5.3 core decoding mode (Table 39): reconstruct the seven
/// output channels `[L, R, C, Ls, Rs, Lt, Rt]` from the five core QMF
/// inputs. No √2 output scaling in this mode; `z2 = x2in`.
pub fn ajcc_core_decode(
    x: &[&[QmfCol]; 5],
    params: &AjccFrameParams<'_>,
    state: &mut AjccCoreSynthState,
) -> Result<Vec<Vec<QmfCol>>> {
    let num_ts = x[0].len();
    if x.iter().any(|m| m.len() != num_ts) {
        return Err(Error::invalid("ac4: A-JCC input timeslot mismatch"));
    }
    let nb = params.num_bands;

    let x0in = scale_input(x[0]);
    let x1in = scale_input(x[1]);
    let x2in = scale_input(x[2]);
    let x3in = scale_input(x[3]);
    let x4in = scale_input(x[4]);

    // u0..u3 / y0..y3 per Table 39.
    let feeds: [&[QmfCol]; 4] = [&x0in, &x3in, &x1in, &x4in];
    let mut y: Vec<Vec<QmfCol>> = Vec::with_capacity(4);
    for (i, feed) in feeds.iter().enumerate() {
        let u = decorrelate(&mut state.decorr[i], feed);
        y.push(duck(&mut state.duckers[i], &u, nb));
    }

    let mut z: Vec<Vec<QmfCol>> = vec![Vec::new(); 7];
    if params.b_5fronts {
        if params.framing.len() != 4 || params.dry_dq.len() != 8 || params.wet_dq.len() != 12 {
            return Err(Error::invalid("ac4: A-JCC 5-fronts roster mismatch"));
        }
        // (z0, z3, z5) from the left pair, (z1, z4, z6) from the right.
        #[allow(clippy::type_complexity)]
        let calls: [(usize, [usize; 3], usize, usize, usize, usize, usize, usize); 2] = [
            // (module, z-map, dry_f base, wet_f base, framing f, framing b, x pair, y pair)
            (0, [0, 3, 5], 0, 0, 0, 2, 0, 0),
            (1, [1, 4, 6], 2, 3, 1, 3, 1, 2),
        ];
        for (m, zi, dfb, wfb, ff, fb, xp, yp) in calls {
            let (za, zb, zc) = ajcc_module_3(
                &params.dry_dq[dfb],
                &params.dry_dq[dfb + 1],
                &params.wet_dq[wfb],
                &params.wet_dq[wfb + 1],
                &params.wet_dq[wfb + 2],
                &params.dry_dq[dfb + 4],
                &params.dry_dq[dfb + 5],
                &params.wet_dq[wfb + 6],
                &params.wet_dq[wfb + 7],
                &params.wet_dq[wfb + 8],
                &params.framing[ff],
                &params.framing[fb],
                nb,
                if xp == 0 { &x0in } else { &x1in },
                if xp == 0 { &x3in } else { &x4in },
                &y[yp],
                &y[yp + 1],
                &mut state.modules[m],
            );
            z[zi[0]] = za;
            z[zi[1]] = zb;
            z[zi[2]] = zc;
        }
    } else {
        if params.framing.len() != 2
            || params.dry_dq.len() != 4
            || params.wet_dq.len() != 6
            || params.alpha_dq.len() != 2
            || params.beta_dq.len() != 2
        {
            return Err(Error::invalid("ac4: A-JCC core-layout roster mismatch"));
        }
        let calls: [(usize, [usize; 3], usize); 2] = [(0, [0, 3, 5], 0), (1, [1, 4, 6], 1)];
        for (m, zi, fi) in calls {
            let (za, zb, zc) = ajcc_module_4(
                &params.alpha_dq[m],
                &params.beta_dq[m],
                &params.dry_dq[2 * m],
                &params.dry_dq[2 * m + 1],
                &params.wet_dq[3 * m],
                &params.wet_dq[3 * m + 1],
                &params.wet_dq[3 * m + 2],
                &params.framing[fi],
                nb,
                params.core_mode,
                if m == 0 { &x0in } else { &x1in },
                if m == 0 { &x3in } else { &x4in },
                &y[2 * m],
                &y[2 * m + 1],
                &mut state.modules[m],
            );
            z[zi[0]] = za;
            z[zi[1]] = zb;
            z[zi[2]] = zc;
        }
    }
    z[2] = x2in;
    Ok(z)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    fn framing(steep: bool, nps: u32, slots: &[u32]) -> AjccFramingData {
        AjccFramingData {
            steep,
            num_param_sets: nps,
            param_timeslot: slots.to_vec(),
        }
    }

    #[test]
    fn smooth_interpolation_single_set_ramps_to_target() {
        // prev = 0, target = 1: interp at ts is (ts+1)/num_ts.
        let param = vec![vec![1.0; 15]];
        let prev = vec![0.0; NUM_QMF_SUBBANDS];
        let fd = framing(false, 1, &[]);
        for ts in 0..32 {
            let v = interpolate_ajcc(&param, &prev, &fd, 15, 3, ts, 32);
            let want = (ts as f64 + 1.0) / 32.0;
            assert!((v - want).abs() < 1e-12, "ts {ts}: {v} vs {want}");
        }
    }

    #[test]
    fn smooth_interpolation_two_sets_split_frame() {
        // prev = 0, set0 = 1, set1 = 3 over 32 slots: first half ramps
        // 0→1, second half ramps 1→3.
        let param = vec![vec![1.0; 15], vec![3.0; 15]];
        let prev = vec![0.0; NUM_QMF_SUBBANDS];
        let fd = framing(false, 2, &[]);
        let v = interpolate_ajcc(&param, &prev, &fd, 15, 0, 15, 32);
        assert!((v - 1.0).abs() < 1e-12, "end of first ramp {v}");
        let v = interpolate_ajcc(&param, &prev, &fd, 15, 0, 31, 32);
        assert!((v - 3.0).abs() < 1e-12, "end of second ramp {v}");
        let v = interpolate_ajcc(&param, &prev, &fd, 15, 0, 16, 32);
        assert!((v - (1.0 + 2.0 / 16.0)).abs() < 1e-12, "first step {v}");
    }

    #[test]
    fn steep_interpolation_switches_at_timeslots() {
        let param = vec![vec![1.0; 15], vec![2.0; 15]];
        let prev = vec![0.5; NUM_QMF_SUBBANDS];
        let fd = framing(true, 2, &[8, 20]);
        assert_eq!(interpolate_ajcc(&param, &prev, &fd, 15, 0, 7, 32), 0.5);
        assert_eq!(interpolate_ajcc(&param, &prev, &fd, 15, 0, 8, 32), 1.0);
        assert_eq!(interpolate_ajcc(&param, &prev, &fd, 15, 0, 19, 32), 1.0);
        assert_eq!(interpolate_ajcc(&param, &prev, &fd, 15, 0, 20, 32), 2.0);
    }

    #[test]
    fn param_prev_update_ungroups_last_set() {
        // Table 33: prev[sb] = last set's value at sb_to_pb(sb).
        let param = vec![vec![0.0; 7], (0..7).map(|pb| pb as f64).collect()];
        let prev = update_param_prev(&param, 7);
        assert_eq!(prev.len(), NUM_QMF_SUBBANDS);
        for sb in 0..NUM_QMF_SUBBANDS {
            let pb = sb_to_pb(sb as u32, 7) as usize;
            assert_eq!(prev[sb], pb as f64, "sb {sb}");
        }
    }

    #[test]
    fn pre_modification_steady_modes_pick_one_input() {
        let num_ts = 4;
        let a = vec![[(1.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let b = vec![[(2.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        // Steady core_mode 0: g = 1 → out = in2 (b).
        let (o1, o2) = input_sig_pre_modification(&a, &b, &a, &b, false, false);
        assert!((o1[0][0].0 - 2.0).abs() < 1e-6);
        assert!((o2[3][5].0 - 2.0).abs() < 1e-6);
        // Steady core_mode 1: g = 0 → out = in1 (a).
        let (o1, _) = input_sig_pre_modification(&a, &b, &a, &b, true, true);
        assert!((o1[0][0].0 - 1.0).abs() < 1e-6);
        // Flip 1 → 0: ramps from in1 toward in2.
        let (o1, _) = input_sig_pre_modification(&a, &b, &a, &b, false, true);
        assert!(o1[0][0].0 < o1[3][0].0, "crossfade must ramp up");
    }

    /// Constant-parameter helper: one set, all bands the same value.
    fn const_set(v: f64, nb: usize) -> Vec<Vec<f64>> {
        vec![vec![v; nb]]
    }

    #[test]
    fn module_1_dry_only_passthrough_weights() {
        // wet = 0 and dry1 = dry2 = 1/3 → z0 = z1 = 1/3·x and
        // z2 = (1 − 2/3)·x after the ramp from prev = 0 converges.
        let nb = 7usize;
        let num_ts = 32;
        let fd = framing(false, 1, &[]);
        let x = vec![[(3.0f32, -1.5f32); NUM_QMF_SUBBANDS]; num_ts];
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut st = AjccModule1State::default();
        let third = const_set(1.0 / 3.0, nb);
        let zero = const_set(0.0, nb);
        let (z0, z1, z2) = ajcc_module_1(
            &third, &third, &zero, &zero, &zero, &fd, nb as u32, &x, &y, &y, &mut st,
        );
        // Last slot of the ramp: interp = target exactly at ts = num_ts−1.
        let want = 3.0 / 3.0;
        assert!(
            (z0[num_ts - 1][10].0 - want).abs() < 1e-5,
            "{}",
            z0[31][10].0
        );
        assert!((z1[num_ts - 1][10].0 - want).abs() < 1e-5);
        assert!((z2[num_ts - 1][10].0 - want).abs() < 1e-5);
        // Imaginary part follows the same weight.
        assert!((z0[num_ts - 1][10].1 + 0.5).abs() < 1e-5);
        // Energy split: z0 + z1 + z2 = x (dry weights sum to 1).
        let sum = z0[31][10].0 + z1[31][10].0 + z2[31][10].0;
        assert!((sum - 3.0).abs() < 1e-5, "dry weights must sum to 1: {sum}");
    }

    #[test]
    fn module_2_core_mode0_mid_side_alpha() {
        // With alpha = 1, dry = wet = beta = 0 and converged ramps:
        // d0 = 1, d3 = 0 → z0 = x0, z3(row) = 0. With alpha = -1 the
        // opposite. Core mode 0 wiring.
        let nb = 7usize;
        let num_ts = 32;
        let fd = framing(false, 1, &[]);
        let x0 = vec![[(2.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let x1 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut st = AjccModule2State::default();
        let alpha = const_set(1.0, nb);
        let zero = const_set(0.0, nb);
        let z = ajcc_module_2(
            &alpha, &zero, &zero, &zero, &zero, &zero, &zero, &fd, nb as u32, false, &x0, &x1, &y,
            &y, &y, &mut st,
        );
        // z outputs (module order): [z_a, z_b, z_c, z_d, z_e] mapping to
        // rows 0..4; row 0 gets d0·x0 = (1+α)/2·x0 = x0; row 3 gets
        // d3·x0 = (1−α)/2·x0 = 0.
        assert!(
            (z[0][num_ts - 1][4].0 - 2.0).abs() < 1e-5,
            "{}",
            z[0][31][4].0
        );
        assert!(z[3][num_ts - 1][4].0.abs() < 1e-5);
        // Rows 1, 2, 4 have no x0 path (d1 = d2 = d4 = 0).
        assert!(z[1][num_ts - 1][4].0.abs() < 1e-5);
        assert!(z[2][num_ts - 1][4].0.abs() < 1e-5);
        assert!(z[4][num_ts - 1][4].0.abs() < 1e-5);
    }

    #[test]
    fn full_decode_centre_passthrough_and_shapes() {
        // Silence everywhere except the centre channel: z2 must be
        // exactly (2+1/√2)·x2 and every output must have the right
        // shape on both layouts.
        let num_ts = 8;
        let silent = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut centre = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            centre[ts][12] = (1.0, 0.25);
        }
        for b_5fronts in [true, false] {
            let (n_framing, n_dry, n_wet, n_ab) = if b_5fronts {
                (4, 8, 12, 0)
            } else {
                (2, 4, 6, 2)
            };
            let framing_v = vec![framing(false, 1, &[]); n_framing];
            let dry = vec![const_set(0.2, 7); n_dry];
            let wet = vec![const_set(0.1, 7); n_wet];
            let ab = vec![const_set(0.0, 7); n_ab];
            let params = AjccFrameParams {
                b_5fronts,
                core_mode: false,
                num_bands: 7,
                framing: &framing_v,
                alpha_dq: &ab,
                beta_dq: &ab,
                dry_dq: &dry,
                wet_dq: &wet,
            };
            let mut state = AjccSynthState::new(b_5fronts);
            let x: [&[QmfCol]; 5] = [&silent, &silent, &centre, &silent, &silent];
            let z = ajcc_full_decode(&x, &params, &mut state).unwrap();
            assert_eq!(z.len(), 13);
            let k = (2.0 + 1.0 / std::f64::consts::SQRT_2) as f32;
            for ts in 0..num_ts {
                assert!((z[2][ts][12].0 - k).abs() < 1e-5);
                assert!((z[2][ts][12].1 - 0.25 * k).abs() < 1e-5);
            }
            // Silent inputs produce silent (but correctly shaped)
            // outputs elsewhere.
            for (i, zi) in z.iter().enumerate() {
                if b_5fronts || (i != 3 && i != 4) {
                    assert_eq!(zi.len(), num_ts, "z{i} timeslots");
                }
                for col in zi {
                    for sb in 0..NUM_QMF_SUBBANDS {
                        if i != 2 {
                            assert!(col[sb].0.abs() < 1e-6, "z{i} sb{sb} leaked");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn full_decode_dry_path_reaches_outputs() {
        // Drive x0 with a steady tone; with dry1 = 0.5 the module-1
        // z0 output (channel L) must carry energy after the ramp, and
        // the decorrelator/wet path adds energy to the √2-scaled
        // outputs later in the frame.
        let num_ts = 32;
        let silent = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            x0[ts][3] = (1.0, 0.0);
        }
        let framing_v = vec![framing(false, 1, &[]); 4];
        let dry = vec![const_set(0.5, 7); 8];
        let wet = vec![const_set(0.3, 7); 12];
        let params = AjccFrameParams {
            b_5fronts: true,
            core_mode: false,
            num_bands: 7,
            framing: &framing_v,
            alpha_dq: &[],
            beta_dq: &[],
            dry_dq: &dry,
            wet_dq: &wet,
        };
        let mut state = AjccSynthState::new(true);
        let x: [&[QmfCol]; 5] = [&x0, &silent, &silent, &silent, &silent];
        let z = ajcc_full_decode(&x, &params, &mut state).unwrap();
        // L (z0) carries the dry path.
        assert!(z[0][num_ts - 1][3].0.abs() > 0.1, "L dry energy");
        // Ltf (z9, √2-scaled) also fed from module 1 of x0.
        assert!(z[9][num_ts - 1][3].0.abs() > 0.1, "Ltf energy");
        // Channels fed only from other cores stay silent.
        assert!(z[1][num_ts - 1][3].0.abs() < 1e-6, "R silent");
        assert!(z[6][num_ts - 1][3].0.abs() < 1e-6, "Rs silent");
    }

    #[test]
    fn module_3_front_back_dry_split() {
        // Front: dry2f = 0.25 → d0 = 0.75, d2 = 0.25 on x0.
        // Back: dry1b = dry2b = 0.3 → d4 = 0.6, d5 = 0.4 on x1.
        // Wet all zero → pure dry mixing after ramp convergence.
        let nb = 7usize;
        let num_ts = 32;
        let fd = framing(false, 1, &[]);
        let x0 = vec![[(4.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let x1 = vec![[(2.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut st = AjccModule34State::default();
        let zero = const_set(0.0, nb);
        let (z0, z1, z2) = ajcc_module_3(
            &const_set(0.0, nb),
            &const_set(0.25, nb),
            &zero,
            &zero,
            &zero,
            &const_set(0.3, nb),
            &const_set(0.3, nb),
            &zero,
            &zero,
            &zero,
            &fd,
            &fd,
            nb as u32,
            &x0,
            &x1,
            &y,
            &y,
            &mut st,
        );
        let last = num_ts - 1;
        // z0 = d0·x0 = 0.75·4 = 3; z1 = d4·x1 = 0.6·2 = 1.2;
        // z2 = d2·x0 + d5·x1 = 0.25·4 + 0.4·2 = 1.8.
        assert!((z0[last][9].0 - 3.0).abs() < 1e-5, "{}", z0[last][9].0);
        assert!((z1[last][9].0 - 1.2).abs() < 1e-5, "{}", z1[last][9].0);
        assert!((z2[last][9].0 - 1.8).abs() < 1e-5, "{}", z2[last][9].0);
    }

    #[test]
    fn module_4_core_mode1_top_folddown() {
        // Core mode 1: d0 = dry1, d1 = 1 − dry1, d5 = 1 → z0 = dry1·x0,
        // z1 = (1−dry1)·x0, z2 = x1 (wet zero).
        let nb = 7usize;
        let num_ts = 32;
        let fd = framing(false, 1, &[]);
        let x0 = vec![[(1.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let x1 = vec![[(5.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let y = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut st = AjccModule34State::default();
        let zero = const_set(0.0, nb);
        let (z0, z1, z2) = ajcc_module_4(
            &zero,
            &zero,
            &const_set(0.8, nb),
            &zero,
            &zero,
            &zero,
            &zero,
            &fd,
            nb as u32,
            true,
            &x0,
            &x1,
            &y,
            &y,
            &mut st,
        );
        let last = num_ts - 1;
        assert!((z0[last][2].0 - 0.8).abs() < 1e-5);
        assert!((z1[last][2].0 - 0.2).abs() < 1e-5);
        assert!((z2[last][2].0 - 5.0).abs() < 1e-5);
    }

    #[test]
    fn core_decode_centre_passthrough_and_shapes() {
        // Both layouts produce 7 channels; centre is (2+1/√2)·x2 and
        // no √2 gain is applied in core mode.
        let num_ts = 8;
        let silent = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut centre = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            centre[ts][20] = (2.0, -1.0);
        }
        for b_5fronts in [true, false] {
            let (n_framing, n_dry, n_wet, n_ab) = if b_5fronts {
                (4, 8, 12, 0)
            } else {
                (2, 4, 6, 2)
            };
            let framing_v = vec![framing(false, 1, &[]); n_framing];
            let dry = vec![const_set(0.2, 9); n_dry];
            let wet = vec![const_set(0.1, 9); n_wet];
            let ab = vec![const_set(0.0, 9); n_ab];
            let params = AjccFrameParams {
                b_5fronts,
                core_mode: true,
                num_bands: 9,
                framing: &framing_v,
                alpha_dq: &ab,
                beta_dq: &ab,
                dry_dq: &dry,
                wet_dq: &wet,
            };
            let mut state = AjccCoreSynthState::new();
            let x: [&[QmfCol]; 5] = [&silent, &silent, &centre, &silent, &silent];
            let z = ajcc_core_decode(&x, &params, &mut state).unwrap();
            assert_eq!(z.len(), 7);
            let k = (2.0 + 1.0 / std::f64::consts::SQRT_2) as f32;
            for ts in 0..num_ts {
                assert!((z[2][ts][20].0 - 2.0 * k).abs() < 1e-5);
                assert!((z[2][ts][20].1 + k).abs() < 1e-5);
            }
            for (i, zi) in z.iter().enumerate() {
                assert_eq!(zi.len(), num_ts, "z{i} shape (b_5fronts={b_5fronts})");
                if i == 2 {
                    continue;
                }
                for col in zi {
                    for sb in 0..NUM_QMF_SUBBANDS {
                        assert!(col[sb].0.abs() < 1e-6, "z{i} sb{sb} leaked");
                    }
                }
            }
        }
    }

    #[test]
    fn core_decode_routes_front_and_back_energy() {
        // 5-fronts core mode: x0 (front-left core) feeds z0 via d0 and
        // z2 (= Ls slot z3? No: module output c goes to z5) — check the
        // documented z-map: (z0, z3, z5) from the first module call.
        let num_ts = 32;
        let silent = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x0 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        let mut x3 = vec![[(0.0f32, 0.0f32); NUM_QMF_SUBBANDS]; num_ts];
        for ts in 0..num_ts {
            x0[ts][5] = (1.0, 0.0);
            x3[ts][5] = (1.0, 0.0);
        }
        let framing_v = vec![framing(false, 1, &[]); 4];
        let dry = vec![const_set(0.25, 7); 8];
        let wet = vec![const_set(0.0, 7); 12];
        let params = AjccFrameParams {
            b_5fronts: true,
            core_mode: true,
            num_bands: 7,
            framing: &framing_v,
            alpha_dq: &[],
            beta_dq: &[],
            dry_dq: &dry,
            wet_dq: &wet,
        };
        let mut state = AjccCoreSynthState::new();
        let x: [&[QmfCol]; 5] = [&x0, &silent, &silent, &x3, &silent];
        let z = ajcc_core_decode(&x, &params, &mut state).unwrap();
        let last = num_ts - 1;
        let k = 2.0 + 1.0 / std::f64::consts::SQRT_2;
        // z0 = (1 − dry2f)·x0in = 0.75·k.
        assert!((z[0][last][5].0 as f64 - 0.75 * k).abs() < 1e-4);
        // z3 = (dry1b + dry2b)·x3in = 0.5·k.
        assert!((z[3][last][5].0 as f64 - 0.5 * k).abs() < 1e-4);
        // z5 = dry2f·x0in + (1 − dry1b − dry2b)·x3in = 0.25k + 0.5k.
        assert!((z[5][last][5].0 as f64 - 0.75 * k).abs() < 1e-4);
        // Right-side channels stay silent.
        assert!(z[1][last][5].0.abs() < 1e-6);
        assert!(z[4][last][5].0.abs() < 1e-6);
        assert!(z[6][last][5].0.abs() < 1e-6);
    }
}
