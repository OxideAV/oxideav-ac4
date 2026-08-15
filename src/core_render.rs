//! Channel-based renderer for **core decoding mode** (ETSI TS
//! 103 190-2 §5.10.2.6 / §5.10.2.7).
//!
//! A core-mode decode of an immersive channel audio substream yields
//! at most seven fullband channels, addressed `[L, R, C, Ls, Rs, Tsl,
//! Tsr]` (§5.6.3.5.3's core output addressing; the generalized
//! rendering matrix of §5.10.2.2 places the two tops on the Tsl / Tsr
//! input rows — Table 45's `r12,12` / `r13,13` coefficients). This
//! module folds that core output to the §5.10.2.6 Table 44 output
//! channel configurations:
//!
//! * **5.X.2** (Table 45) — pass-through with the back-fold gain on
//!   Ls / Rs and the top-fold gain on Tsl / Tsr;
//! * **5.X.0** (Table 46) — additionally mixes the tops into the
//!   front and/or side channels via `gain_t2a` / `gain_t2b`.
//!
//! The matrix coefficients combine static +3 dB terms with the
//! customized downmix gains of §6.3.10.3.2 (Table 129 code → dB
//! mapping, Table 130 defaults). The +3 dB (×√2) exactly compensates
//! the core fold: a 7.X.4 substream's core `Ls` carries
//! `(Ls + Lb)/√2`, so `gain_b + 3 dB` renders `gain_b · (Ls + Lb)` —
//! the same signal a full decode folded to 5.X.2 produces.
//!
//! The LFE channel is not part of these helpers; per Table 45 NOTE 2
//! its coefficient is 0 dB whenever present (the caller keeps it).

/// Table 129 gain code → dB (`gain_f2` / `gain_b` / `gain_t*`).
/// Code 7 is −∞ dB.
pub const GAIN_CODE_DB: [f32; 8] = [0.0, -1.5, -3.0, -4.5, -6.0, -9.0, -12.0, f32::NEG_INFINITY];

/// Linear gain for a 3-bit Table 129 gain code (code 7 → 0,0).
pub fn gain_code_linear(code: u8) -> f32 {
    let db = GAIN_CODE_DB[(code as usize).min(7)];
    if db == f32::NEG_INFINITY {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

/// Customized downmix gains used by the core-decoding renderer
/// (linear scale). Defaults follow Table 130: `gain_b = gain_t1 =
/// gain_t2b = −3 dB`, `gain_t2a = −∞ dB` (tops to the sides, not the
/// fronts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreRenderGains {
    /// `gain_b` — back-channel fold gain (Table 46 / 45 `r3,3`,
    /// `r4,4` when four back channels are present).
    pub gain_b: f32,
    /// `gain_t1` — four-top fold gain (Table 45 `r12,12` / `r13,13`
    /// when `top_channels_present = 3`).
    pub gain_t1: f32,
    /// `gain_t2a` — top-to-front mix gain (Table 46 `r0,12` /
    /// `r1,13`).
    pub gain_t2a: f32,
    /// `gain_t2b` — top-to-side mix gain (Table 46 `r3,12` /
    /// `r4,13`).
    pub gain_t2b: f32,
}

impl Default for CoreRenderGains {
    fn default() -> Self {
        CoreRenderGains {
            gain_b: gain_code_linear(2),   // −3 dB
            gain_t1: gain_code_linear(2),  // −3 dB
            gain_t2a: gain_code_linear(7), // −∞ dB
            gain_t2b: gain_code_linear(2), // −3 dB
        }
    }
}

impl CoreRenderGains {
    /// Build from raw 3-bit Table 129 codes; `None` keeps the
    /// Table 130 default for that gain.
    pub fn from_codes(
        gain_b_code: Option<u8>,
        gain_t1_code: Option<u8>,
        gain_t2a_code: Option<u8>,
        gain_t2b_code: Option<u8>,
    ) -> Self {
        let d = CoreRenderGains::default();
        CoreRenderGains {
            gain_b: gain_b_code.map(gain_code_linear).unwrap_or(d.gain_b),
            gain_t1: gain_t1_code.map(gain_code_linear).unwrap_or(d.gain_t1),
            gain_t2a: gain_t2a_code.map(gain_code_linear).unwrap_or(d.gain_t2a),
            gain_t2b: gain_t2b_code.map(gain_code_linear).unwrap_or(d.gain_t2b),
        }
    }

    /// Apply a `tool_t2_to_f_s()`-style single-top-pair selection
    /// (§6.2.9.10 / §6.3.10.3.8-9): the `b_top_to_front` form routes
    /// the tops to the fronts (`gain_t2a` = the transmitted code,
    /// `gain_t2b` = −∞), the other form to the sides.
    pub fn with_t2_tool(mut self, tool: crate::oamd::ToolTwoWay) -> Self {
        match tool {
            crate::oamd::ToolTwoWay::Front(code) => {
                self.gain_t2a = gain_code_linear(code);
                self.gain_t2b = 0.0;
            }
            crate::oamd::ToolTwoWay::Other(code) => {
                self.gain_t2a = 0.0;
                self.gain_t2b = gain_code_linear(code);
            }
        }
        self
    }
}

const SQRT_2: f32 = std::f32::consts::SQRT_2;

fn scaled(src: &[f32], g: f32) -> Vec<f32> {
    src.iter().map(|&v| v * g).collect()
}

fn mix_into(dst: &mut [f32], src: &[f32], g: f32) {
    if g == 0.0 {
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += g * s;
    }
}

/// §5.10.2.7 Table 45 — render a 7-channel core-decode output
/// (`[L, R, C, Ls, Rs, Tsl, Tsr]`, immersive 9.X.X / 7.X.X source)
/// to the **5.X.2** output configuration (same slot order).
///
/// `top_channels_present` / `b_4_back_channels_present` are the
/// §6.3.2.7.3-5 substream presence fields (see
/// [`crate::toc::ChannelModeDesc`]); the immersive `X.X.4` channel
/// modes carry four tops (`top_channels_present = 3`) and four backs.
pub fn render_core_to_5_x_2(
    chans: &[Vec<f32>],
    top_channels_present: u8,
    b_4_back_channels_present: bool,
    gains: &CoreRenderGains,
) -> Vec<Vec<f32>> {
    assert_eq!(chans.len(), 7, "core render expects 7 channels");
    // r3,3 = r4,4: gain_b + 3 dB with four backs, +3 dB otherwise.
    let g_side = if b_4_back_channels_present {
        gains.gain_b * SQRT_2
    } else {
        SQRT_2
    };
    // r12,12 = r13,13: gain_t1 + 3 dB with four tops, +3 dB with one
    // or two; no top channels → the default −∞ dB coefficient.
    let g_top = match top_channels_present {
        3 => gains.gain_t1 * SQRT_2,
        1 | 2 => SQRT_2,
        _ => 0.0,
    };
    vec![
        chans[0].clone(),
        chans[1].clone(),
        chans[2].clone(),
        scaled(&chans[3], g_side),
        scaled(&chans[4], g_side),
        scaled(&chans[5], g_top),
        scaled(&chans[6], g_top),
    ]
}

/// §5.10.2.7 Table 46 — render a 7-channel core-decode output
/// (`[L, R, C, Ls, Rs, Tsl, Tsr]`, immersive 9.X.X / 7.X.X source)
/// to the **5.X.0** output configuration (`[L, R, C, Ls, Rs]`).
pub fn render_core_to_5_x_0(
    chans: &[Vec<f32>],
    top_channels_present: u8,
    b_4_back_channels_present: bool,
    gains: &CoreRenderGains,
) -> Vec<Vec<f32>> {
    assert_eq!(chans.len(), 7, "core render expects 7 channels");
    let g_side = if b_4_back_channels_present {
        gains.gain_b * SQRT_2
    } else {
        SQRT_2
    };
    let mut out = vec![
        chans[0].clone(),
        chans[1].clone(),
        chans[2].clone(),
        scaled(&chans[3], g_side),
        scaled(&chans[4], g_side),
    ];
    // r0,12 = r1,13 = gain_t2a + 3 dB and r3,12 = r4,13 =
    // gain_t2b + 3 dB whenever top channels are present.
    if (1..=3).contains(&top_channels_present) {
        let (g_front, g_sides) = (gains.gain_t2a * SQRT_2, gains.gain_t2b * SQRT_2);
        let (front, back) = out.split_at_mut(3);
        mix_into(&mut front[0], &chans[5], g_front);
        mix_into(&mut front[1], &chans[6], g_front);
        mix_into(&mut back[0], &chans[5], g_sides);
        mix_into(&mut back[1], &chans[6], g_sides);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_129_codes_map_to_documented_gains() {
        assert_eq!(gain_code_linear(0), 1.0);
        assert!((gain_code_linear(2) - 10f32.powf(-3.0 / 20.0)).abs() < 1e-6);
        assert_eq!(gain_code_linear(7), 0.0);
    }

    #[test]
    fn default_gains_follow_table_130() {
        let g = CoreRenderGains::default();
        let m3 = 10f32.powf(-3.0 / 20.0);
        assert!((g.gain_b - m3).abs() < 1e-6);
        assert!((g.gain_t1 - m3).abs() < 1e-6);
        assert_eq!(g.gain_t2a, 0.0);
        assert!((g.gain_t2b - m3).abs() < 1e-6);
    }

    #[test]
    fn render_5_x_2_applies_table_45_coefficients() {
        let chans: Vec<Vec<f32>> = (0..7).map(|i| vec![1.0f32 + i as f32]).collect();
        let g = CoreRenderGains::default();
        let out = render_core_to_5_x_2(&chans, 3, true, &g);
        // gain_b + 3 dB with the default −3 dB gain_b ≈ 0 dB.
        assert!((out[3][0] - chans[3][0] * g.gain_b * SQRT_2).abs() < 1e-5);
        assert!((out[3][0] - chans[3][0]).abs() < 1e-2);
        assert!((out[5][0] - chans[5][0] * g.gain_t1 * SQRT_2).abs() < 1e-5);
        // L / R / C pass through.
        assert_eq!(out[0][0], chans[0][0]);
        // Two-top source: +3 dB on the tops instead of gain_t1.
        let out2 = render_core_to_5_x_2(&chans, 2, false, &g);
        assert!((out2[5][0] - chans[5][0] * SQRT_2).abs() < 1e-5);
        assert!((out2[3][0] - chans[3][0] * SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn render_5_x_0_folds_tops_per_table_46() {
        let chans: Vec<Vec<f32>> = (0..7).map(|i| vec![1.0f32 + i as f32]).collect();
        let g = CoreRenderGains::default();
        let out = render_core_to_5_x_0(&chans, 3, true, &g);
        assert_eq!(out.len(), 5);
        // Default gain_t2a = −∞: fronts carry no top mix.
        assert_eq!(out[0][0], chans[0][0]);
        // Sides: gain_b+3dB on Ls plus gain_t2b+3dB · Tsl ≈ Ls + Tsl.
        let want = chans[3][0] * g.gain_b * SQRT_2 + chans[5][0] * g.gain_t2b * SQRT_2;
        assert!((out[3][0] - want).abs() < 1e-5);
        assert!((out[3][0] - (chans[3][0] + chans[5][0])).abs() < 2e-2);
        // Top-to-front tool routes the tops to L / R instead.
        let gf = g.with_t2_tool(crate::oamd::ToolTwoWay::Front(0));
        let out_f = render_core_to_5_x_0(&chans, 3, true, &gf);
        assert!((out_f[0][0] - (chans[0][0] + chans[5][0] * SQRT_2)).abs() < 1e-5);
        // gain_t2b forced to −∞: the sides carry no top mix.
        assert!((out_f[3][0] - chans[3][0] * gf.gain_b * SQRT_2).abs() < 1e-5);
    }
}
