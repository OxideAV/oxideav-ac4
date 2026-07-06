//! A-JOC parameter encoder — quantization (§5.7.3.3 inverse), per-row
//! differential-direction selection by exact Huffman bit cost, and
//! GOP-chained `ajoc()` element emission (ETSI TS 103 190-2 §6.2.5).
//!
//! The encoder mirrors the decode chain in [`crate::ajoc_data`]:
//!
//! 1. [`quantize_dry`] / [`quantize_wet`] map real-valued dry / wet
//!    matrix coefficients onto the uniform Table 44-47 quantizer grids
//!    (nearest index, clamped to the grid).
//! 2. [`choose_row_coding`] prices each matrix row both ways with the
//!    actual Annex A.1.1 codeword lengths — DIFF_FREQ (F0 + DF books)
//!    vs DIFF_TIME (DT book against the previous frame's quantised row,
//!    §5.7.3.2) — and picks the cheaper direction. The time direction
//!    is unavailable on I-frames (no reference), when `b_dfonly` forces
//!    frequency coding (`dp == 0 && ajoc_b_nodt`, §6.2.5.3), or when a
//!    delta falls outside the DT codebook range.
//! 3. [`encode_ajoc_frame`] walks every present object / data point /
//!    row, emits the chosen rows into an [`AjocData`], updates the
//!    encoder-side `mtx_*_q_prev` mirror ([`AjocDiffState`]) exactly
//!    like the decoder does, and [`encode_ajoc`] wraps the result into
//!    a complete `ajoc()` bitstream element via
//!    [`crate::ajoc_data::write_ajoc`].
//!
//! Because both sides run the same Table 43 recurrences on the same
//! quantised values, an encode → decode roundtrip is exact on the
//! quantised grid for arbitrarily long I/P GOPs (verified in the
//! tests, including measured P-frame bit savings from the time
//! direction).

use crate::ajoc::{AjocCtrlInfo, AjocDiffState, AjocDiffType, AjocMatrixKind, AjocQuantMode};
use crate::ajoc_data::{
    encode_ajoc_deltas_freq, encode_ajoc_deltas_time, write_ajoc, AjocData, AjocHuffRow,
};
use crate::ajoc_huffman::{get_ajoc_hcb, AjocHcbType};
use oxideav_core::bits::BitWriter;
use oxideav_core::{Error, Result};

/// Quantize one dry matrix coefficient onto the Table 44 / 45 grid
/// (nearest index, clamped) — the encoder inverse of
/// [`crate::ajoc::dequantize_dry`].
pub fn quantize_dry(v: f64, mode: AjocQuantMode) -> i32 {
    let nquant = mode.nquant(AjocMatrixKind::Dry) as i32;
    let centre = (nquant - 1) / 2;
    // Step is dequantize(centre + 1) - dequantize(centre).
    let step = crate::ajoc::dequantize_dry((centre + 1) as u32, mode);
    ((v / step).round() + centre as f64).clamp(0.0, (nquant - 1) as f64) as i32
}

/// Quantize one wet matrix coefficient onto the Table 46 / 47 grid —
/// the encoder inverse of [`crate::ajoc::dequantize_wet`].
pub fn quantize_wet(v: f64, mode: AjocQuantMode) -> i32 {
    let nquant = mode.nquant(AjocMatrixKind::Wet) as i32;
    let centre = (nquant - 1) / 2;
    let step = crate::ajoc::dequantize_wet((centre + 1) as u32, mode);
    ((v / step).round() + centre as f64).clamp(0.0, (nquant - 1) as f64) as i32
}

/// Bit cost of coding `deltas` with the given codebook (`None` if any
/// symbol is out of the codebook's range).
fn row_bits(
    deltas: &[i32],
    kind: AjocMatrixKind,
    mode: AjocQuantMode,
    hcb_type: AjocHcbType,
) -> Option<u32> {
    let hcb = get_ajoc_hcb(kind, mode, hcb_type);
    let mut bits = 0u32;
    for &d in deltas {
        let idx = d + hcb.cb_off;
        if idx < 0 || idx as usize >= hcb.len.len() {
            return None;
        }
        bits += hcb.len[idx as usize] as u32;
    }
    Some(bits)
}

/// Price a frequency-coded row: F0 codeword for band 0 plus DF
/// codewords for the rest (plus the diff_type bit unless `b_dfonly`).
fn freq_row_bits(
    deltas: &[i32],
    kind: AjocMatrixKind,
    mode: AjocQuantMode,
    b_dfonly: bool,
) -> Option<u32> {
    let (first, rest) = deltas.split_first()?;
    let f0 = row_bits(&[*first], kind, mode, AjocHcbType::F0)?;
    let df = row_bits(rest, kind, mode, AjocHcbType::Df)?;
    Some(f0 + df + u32::from(!b_dfonly))
}

/// Choose the cheaper coding direction for one matrix row.
///
/// * `q` — this frame's absolute quantised row.
/// * `prev` — the previous frame's quantised row (`None` on I-frames /
///   first use, which forbids the time direction).
/// * `b_dfonly` — §6.2.5.3 constraint (`dp == 0 && ajoc_b_nodt`).
///
/// Returns `(diff_type, a_huff_data, bits)`.
pub fn choose_row_coding(
    q: &[i32],
    prev: Option<&[i32]>,
    kind: AjocMatrixKind,
    mode: AjocQuantMode,
    b_dfonly: bool,
) -> Result<(AjocDiffType, Vec<i32>, u32)> {
    let freq_deltas = encode_ajoc_deltas_freq(q, kind, mode);
    let freq_cost = freq_row_bits(&freq_deltas, kind, mode, b_dfonly)
        .ok_or_else(|| Error::invalid("ac4: A-JOC quantised row out of grid"))?;

    if !b_dfonly {
        if let Some(prev) = prev {
            let time_deltas = encode_ajoc_deltas_time(q, prev);
            if let Some(dt_bits) = row_bits(&time_deltas, kind, mode, AjocHcbType::Dt) {
                let time_cost = dt_bits + 1; // diff_type bit
                if time_cost < freq_cost {
                    return Ok((AjocDiffType::Time, time_deltas, time_cost));
                }
            }
        }
    }
    Ok((AjocDiffType::Freq, freq_deltas, freq_cost))
}

/// Absolute quantised A-JOC matrices for one frame:
/// `dry_q[o][dp][ch][pb]`, `wet_q[o][dp][de][pb]` (present objects
/// only need populated rows; band count per object from the ctrl info).
#[derive(Clone, Debug, Default)]
pub struct AjocQuantMatrices {
    /// `mtx_dry_q[o][dp][ch][pb]`.
    pub dry_q: Vec<Vec<Vec<Vec<i32>>>>,
    /// `mtx_wet_q[o][dp][de][pb]`.
    pub wet_q: Vec<Vec<Vec<Vec<i32>>>>,
}

impl AjocQuantMatrices {
    /// Quantize real-valued matrices (`dry[o][dp][ch][pb]`,
    /// `wet[o][dp][de][pb]`) onto the quantizer grid selected per
    /// object by `ctrl.quant_select`.
    pub fn from_real(
        dry: &[Vec<Vec<Vec<f64>>>],
        wet: &[Vec<Vec<Vec<f64>>>],
        ctrl: &AjocCtrlInfo,
    ) -> Self {
        let quant = |o: usize| {
            ctrl.quant_select
                .get(o)
                .copied()
                .unwrap_or(AjocQuantMode::Fine)
        };
        let dry_q = dry
            .iter()
            .enumerate()
            .map(|(o, dps)| {
                dps.iter()
                    .map(|chs| {
                        chs.iter()
                            .map(|row| row.iter().map(|&v| quantize_dry(v, quant(o))).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let wet_q = wet
            .iter()
            .enumerate()
            .map(|(o, dps)| {
                dps.iter()
                    .map(|des| {
                        des.iter()
                            .map(|row| row.iter().map(|&v| quantize_wet(v, quant(o))).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        AjocQuantMatrices { dry_q, wet_q }
    }
}

/// Encode one frame's quantised matrices into an [`AjocData`] payload,
/// choosing the differential direction per row and advancing the
/// encoder-side `mtx_*_q_prev` mirror (`state`) exactly as the decoder
/// will (§5.7.3.2). `b_iframe = true` forbids the time direction for
/// every row (no cross-frame reference) and sets `ajoc_b_nodt`.
///
/// Returns the payload plus the total payload bits spent on Huffman
/// rows (diagnostics for the tests / rate control).
pub fn encode_ajoc_frame(
    ctrl: &AjocCtrlInfo,
    q: &AjocQuantMatrices,
    b_iframe: bool,
    state: &mut AjocDiffState,
) -> Result<(AjocData, u32)> {
    let num_umx = ctrl.object_present.len();
    let num_dp = ctrl.data_point_info.num_dpoints as usize;
    // On I-frames nothing may reference the previous frame; signal
    // ajoc_b_nodt so dp 0 saves the per-row diff_type bit.
    let b_nodt = b_iframe;

    let mut data = AjocData {
        b_nodt,
        mix_mtx_dry: vec![Vec::new(); num_umx],
        mix_mtx_wet: vec![Vec::new(); num_umx],
    };
    let mut total_bits = 0u32;

    for o in 0..num_umx {
        if !ctrl.object_present[o] {
            continue;
        }
        let nb = ctrl.num_bands[o] as usize;
        let mode = ctrl.quant_select[o];
        for dp in 0..num_dp {
            let b_dfonly = dp == 0 && b_nodt;
            let dry_dp = q
                .dry_q
                .get(o)
                .and_then(|d| d.get(dp))
                .ok_or_else(|| Error::invalid("ac4: missing quantised dry data point"))?;
            let mut dry_rows: Vec<AjocHuffRow> = Vec::with_capacity(dry_dp.len());
            for (ch, row) in dry_dp.iter().enumerate() {
                if row.len() != nb {
                    return Err(Error::invalid("ac4: quantised dry row band mismatch"));
                }
                let transmitted = !ctrl.sparse_select[o]
                    || ctrl
                        .mix_mtx_dry_present
                        .get(o)
                        .and_then(|r| r.get(ch))
                        .copied()
                        .unwrap_or(false);
                let prev = &mut state.dry_prev[o][ch];
                if !transmitted {
                    // Decoder resets to the zero-centre index; mirror it.
                    let zero = mode.zero_index(AjocMatrixKind::Dry) as i32;
                    prev[..nb].fill(zero);
                    dry_rows.push(None);
                    continue;
                }
                let reference = if b_iframe { None } else { Some(&prev[..nb]) };
                let (dt, deltas, bits) =
                    choose_row_coding(row, reference, AjocMatrixKind::Dry, mode, b_dfonly)?;
                prev[..nb].copy_from_slice(row);
                total_bits += bits;
                dry_rows.push(Some((dt, deltas)));
            }
            data.mix_mtx_dry[o].push(dry_rows);

            let wet_dp = q
                .wet_q
                .get(o)
                .and_then(|w| w.get(dp))
                .ok_or_else(|| Error::invalid("ac4: missing quantised wet data point"))?;
            let mut wet_rows: Vec<AjocHuffRow> = Vec::with_capacity(wet_dp.len());
            for (de, row) in wet_dp.iter().enumerate() {
                if row.len() != nb {
                    return Err(Error::invalid("ac4: quantised wet row band mismatch"));
                }
                let transmitted = !ctrl.sparse_select[o]
                    || ctrl
                        .mix_mtx_wet_present
                        .get(o)
                        .and_then(|r| r.get(de))
                        .copied()
                        .unwrap_or(false);
                let prev = &mut state.wet_prev[o][de];
                if !transmitted {
                    let zero = mode.zero_index(AjocMatrixKind::Wet) as i32;
                    prev[..nb].fill(zero);
                    wet_rows.push(None);
                    continue;
                }
                let reference = if b_iframe { None } else { Some(&prev[..nb]) };
                let (dt, deltas, bits) =
                    choose_row_coding(row, reference, AjocMatrixKind::Wet, mode, b_dfonly)?;
                prev[..nb].copy_from_slice(row);
                total_bits += bits;
                wet_rows.push(Some((dt, deltas)));
            }
            data.mix_mtx_wet[o].push(wet_rows);
        }
    }

    Ok((data, total_bits))
}

/// Encode a complete `ajoc()` element (§6.2.5.1) from quantised
/// matrices: direction selection + payload assembly + bitstream
/// emission. Returns the payload row bits (see [`encode_ajoc_frame`]).
pub fn encode_ajoc(
    bw: &mut BitWriter,
    num_decorr: u32,
    ctrl: &AjocCtrlInfo,
    q: &AjocQuantMatrices,
    b_iframe: bool,
    state: &mut AjocDiffState,
) -> Result<u32> {
    let (data, bits) = encode_ajoc_frame(ctrl, q, b_iframe, state)?;
    write_ajoc(bw, num_decorr, ctrl, &data)?;
    Ok(bits)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::ajoc::{dequantize_dry, dequantize_wet, AjocDataPointInfo};
    use crate::ajoc_data::{decode_ajoc, new_ajoc_diff_state};
    use oxideav_core::bits::BitReader;

    fn simple_ctrl(
        num_umx: usize,
        num_decorr: usize,
        num_bands_code: u8,
        quant: AjocQuantMode,
    ) -> AjocCtrlInfo {
        let num_bands = crate::ajoc::num_bands_from_code(num_bands_code).unwrap();
        AjocCtrlInfo {
            decorr_enable: vec![true; num_decorr],
            object_present: vec![true; num_umx],
            data_point_info: AjocDataPointInfo {
                num_dpoints: 1,
                start_pos: vec![0],
                ramp_len: vec![8],
            },
            num_bands_code: vec![num_bands_code; num_umx],
            num_bands: vec![num_bands; num_umx],
            quant_select: vec![quant; num_umx],
            sparse_select: vec![false; num_umx],
            mix_mtx_dry_present: vec![Vec::new(); num_umx],
            mix_mtx_wet_present: vec![Vec::new(); num_umx],
        }
    }

    #[test]
    fn quantizers_invert_dequantizers_on_grid() {
        for mode in [AjocQuantMode::Coarse, AjocQuantMode::Fine] {
            for q in 0..mode.nquant(AjocMatrixKind::Dry) {
                assert_eq!(quantize_dry(dequantize_dry(q, mode), mode), q as i32);
            }
            for q in 0..mode.nquant(AjocMatrixKind::Wet) {
                assert_eq!(quantize_wet(dequantize_wet(q, mode), mode), q as i32);
            }
        }
        // Off-grid values snap to the nearest index; out-of-range clamps.
        assert_eq!(quantize_dry(0.0, AjocQuantMode::Coarse), 25);
        assert_eq!(quantize_dry(1e9, AjocQuantMode::Coarse), 50);
        assert_eq!(quantize_dry(-1e9, AjocQuantMode::Coarse), 0);
        assert_eq!(quantize_wet(1e9, AjocQuantMode::Fine), 40);
    }

    #[test]
    fn direction_selection_prefers_cheap_time_deltas() {
        // A row that barely moves: time deltas are all 0/±1 (short DT
        // codes) while freq re-coding pays full F0 + DF codes.
        let mode = AjocQuantMode::Coarse;
        let prev = vec![25, 27, 23, 25, 26, 24, 25];
        let cur = vec![25, 26, 23, 25, 27, 24, 25];
        let (dt, deltas, bits) =
            choose_row_coding(&cur, Some(&prev), AjocMatrixKind::Dry, mode, false).unwrap();
        assert_eq!(dt, AjocDiffType::Time);
        assert_eq!(deltas, vec![0, -1, 0, 0, 1, 0, 0]);
        let (_, _, freq_bits) =
            choose_row_coding(&cur, None, AjocMatrixKind::Dry, mode, false).unwrap();
        assert!(bits < freq_bits, "time {bits} !< freq {freq_bits}");
    }

    #[test]
    fn dfonly_and_missing_reference_force_freq() {
        let mode = AjocQuantMode::Fine;
        let prev = vec![50, 50, 50];
        let cur = vec![50, 50, 50];
        // b_dfonly = true → freq even though time would be free-ish.
        let (dt, _, _) =
            choose_row_coding(&cur, Some(&prev), AjocMatrixKind::Dry, mode, true).unwrap();
        assert_eq!(dt, AjocDiffType::Freq);
        // No reference → freq.
        let (dt, _, _) = choose_row_coding(&cur, None, AjocMatrixKind::Dry, mode, false).unwrap();
        assert_eq!(dt, AjocDiffType::Freq);
    }

    #[test]
    fn gop_encode_decode_roundtrip_is_exact_on_grid() {
        // 4-frame GOP (I P P P), 2 objects, 2 dmx, 1 decorrelator,
        // 9 bands, fine quantization. Matrices evolve smoothly so the
        // P-frames mostly pick the time direction.
        let quant = AjocQuantMode::Fine;
        let ctrl = simple_ctrl(2, 1, 3, quant); // 9 bands
        let nb = 9usize;

        let mut enc_state = new_ajoc_diff_state(2, 2, 1);
        let mut dec_state = new_ajoc_diff_state(2, 2, 1);
        let mut p_bits = Vec::new();

        for frame in 0..4 {
            // Smoothly evolving real-valued matrices.
            let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..2)
                .map(|o| {
                    vec![(0..2)
                        .map(|ch| {
                            (0..nb)
                                .map(|pb| {
                                    0.3 * (o as f64 - 0.5)
                                        + 0.05 * ch as f64
                                        + 0.02 * pb as f64
                                        + 0.01 * frame as f64
                                })
                                .collect()
                        })
                        .collect()]
                })
                .collect();
            let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..2)
                .map(|o| {
                    vec![(0..1)
                        .map(|_| {
                            (0..nb)
                                .map(|pb| 0.1 + 0.01 * pb as f64 - 0.005 * frame as f64 * o as f64)
                                .collect()
                        })
                        .collect()]
                })
                .collect();
            let q = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);

            let mut bw = BitWriter::new();
            let bits = encode_ajoc(&mut bw, 1, &ctrl, &q, frame == 0, &mut enc_state).unwrap();
            let bytes = bw.finish();
            if frame > 0 {
                p_bits.push(bits);
            }

            let mut br = BitReader::new(&bytes);
            let decoded = decode_ajoc(&mut br, 2, 2, &mut dec_state).unwrap();

            // Decoded dequantized matrices equal the quantised encode
            // input mapped through the dequantizer (exact on the grid).
            for o in 0..2 {
                for ch in 0..2 {
                    for pb in 0..nb {
                        let want = dequantize_dry(q.dry_q[o][0][ch][pb] as u32, quant);
                        let got = decoded.matrices.dry_dq[o][0][ch][pb];
                        assert!(
                            (got - want).abs() < 1e-12,
                            "frame {frame} dry o{o} ch{ch} pb{pb}: {got} vs {want}"
                        );
                    }
                }
                for pb in 0..nb {
                    let want = dequantize_wet(q.wet_q[o][0][0][pb] as u32, quant);
                    let got = decoded.matrices.wet_dq[o][0][0][pb];
                    assert!((got - want).abs() < 1e-12, "frame {frame} wet o{o} pb{pb}");
                }
            }
            // Encoder and decoder prev-state mirrors stay in lockstep.
            assert_eq!(enc_state.dry_prev, dec_state.dry_prev);
            assert_eq!(enc_state.wet_prev, dec_state.wet_prev);

            // P-frames actually use the time direction somewhere.
            if frame > 0 {
                let uses_time = decoded
                    .data
                    .mix_mtx_dry
                    .iter()
                    .flatten()
                    .flatten()
                    .chain(decoded.data.mix_mtx_wet.iter().flatten().flatten())
                    .any(|row| matches!(row, Some((AjocDiffType::Time, _))));
                assert!(uses_time, "frame {frame} coded no time-direction rows");
            }
        }

        // Measured savings: re-encode frame 3's matrices I-style (freq
        // only, fresh state) and compare row bits.
        let dry: Vec<Vec<Vec<Vec<f64>>>> = (0..2)
            .map(|o| {
                vec![(0..2)
                    .map(|ch| {
                        (0..nb)
                            .map(|pb| {
                                0.3 * (o as f64 - 0.5) + 0.05 * ch as f64 + 0.02 * pb as f64 + 0.03
                            })
                            .collect()
                    })
                    .collect()]
            })
            .collect();
        let wet: Vec<Vec<Vec<Vec<f64>>>> = (0..2)
            .map(|o| {
                vec![(0..1)
                    .map(|_| {
                        (0..nb)
                            .map(|pb| 0.1 + 0.01 * pb as f64 - 0.015 * o as f64)
                            .collect()
                    })
                    .collect()]
            })
            .collect();
        let q = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);
        let mut fresh = new_ajoc_diff_state(2, 2, 1);
        let (_, i_bits) = encode_ajoc_frame(&ctrl, &q, true, &mut fresh).unwrap();
        let avg_p = p_bits.iter().sum::<u32>() as f64 / p_bits.len() as f64;
        assert!(
            avg_p < i_bits as f64,
            "P-frame rows ({avg_p} bits avg) should undercut I-frame rows ({i_bits} bits)"
        );
    }

    #[test]
    fn sparse_rows_stay_consistent_across_gop() {
        // A sparse object whose absent rows must decode to zero-centre
        // on every frame while transmitted rows chain DIFF_TIME.
        let quant = AjocQuantMode::Coarse;
        let mut ctrl = simple_ctrl(1, 1, 5, quant); // 5 bands
        ctrl.sparse_select[0] = true;
        ctrl.mix_mtx_dry_present[0] = vec![true, false];
        ctrl.mix_mtx_wet_present[0] = vec![true];

        let nb = 5usize;
        let mut enc_state = new_ajoc_diff_state(1, 2, 1);
        let mut dec_state = new_ajoc_diff_state(1, 2, 1);

        for frame in 0..3 {
            let dry = vec![vec![vec![
                (0..nb)
                    .map(|pb| 0.2 * pb as f64 + 0.05 * frame as f64)
                    .collect(),
                vec![0.0; nb], // absent row: content ignored
            ]]];
            let wet = vec![vec![vec![(0..nb)
                .map(|pb| -0.3 + 0.02 * pb as f64 + 0.01 * frame as f64)
                .collect()]]];
            let q = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);

            let mut bw = BitWriter::new();
            encode_ajoc(&mut bw, 1, &ctrl, &q, frame == 0, &mut enc_state).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let decoded = decode_ajoc(&mut br, 2, 1, &mut dec_state).unwrap();

            for pb in 0..nb {
                let want = dequantize_dry(q.dry_q[0][0][0][pb] as u32, quant);
                assert!((decoded.matrices.dry_dq[0][0][0][pb] - want).abs() < 1e-12);
                // Absent dry row pinned at zero-centre (dequantizes to 0).
                assert!(decoded.matrices.dry_dq[0][0][1][pb].abs() < 1e-12);
                let want = dequantize_wet(q.wet_q[0][0][0][pb] as u32, quant);
                assert!((decoded.matrices.wet_dq[0][0][0][pb] - want).abs() < 1e-12);
            }
            assert_eq!(enc_state.dry_prev, dec_state.dry_prev);
            assert_eq!(enc_state.wet_prev, dec_state.wet_prev);
        }
    }

    #[test]
    fn two_data_points_share_the_frame_reference_chain() {
        // Two data points in one frame: dp 0 freq (b_nodt), dp 1 may
        // time-code against dp 0 (Table 43 chains prev across dps).
        let quant = AjocQuantMode::Coarse;
        let mut ctrl = simple_ctrl(1, 0, 5, quant); // 5 bands, no decorr
        ctrl.data_point_info = AjocDataPointInfo {
            num_dpoints: 2,
            start_pos: vec![0, 8],
            ramp_len: vec![4, 4],
        };
        let nb = 5usize;
        let dry = vec![vec![
            vec![(0..nb).map(|pb| 0.2 * pb as f64).collect::<Vec<f64>>()],
            vec![(0..nb)
                .map(|pb| 0.2 * pb as f64 + 0.2)
                .collect::<Vec<f64>>()],
        ]];
        let wet = vec![vec![Vec::<Vec<f64>>::new(), Vec::<Vec<f64>>::new()]];
        let q = AjocQuantMatrices::from_real(&dry, &wet, &ctrl);

        let mut enc_state = new_ajoc_diff_state(1, 1, 0);
        let mut bw = BitWriter::new();
        // I-frame: dp 0 is dfonly, dp 1 references dp 0 within the frame.
        // encode_ajoc_frame forbids time on I-frames entirely (spec-safe:
        // decoder state matches either way), so force a P-frame with a
        // pre-seeded state to exercise the dp chain.
        encode_ajoc(&mut bw, 0, &ctrl, &q, true, &mut enc_state).unwrap();
        let bytes = bw.finish();
        let mut dec_state = new_ajoc_diff_state(1, 1, 0);
        let mut br = BitReader::new(&bytes);
        let decoded = decode_ajoc(&mut br, 1, 1, &mut dec_state).unwrap();
        for dp in 0..2 {
            for pb in 0..nb {
                let want = dequantize_dry(q.dry_q[0][dp][0][pb] as u32, quant);
                assert!((decoded.matrices.dry_dq[0][dp][0][pb] - want).abs() < 1e-12);
            }
        }

        // Now a P-frame on the same state: dp 1 deltas vs dp 0 are the
        // constant +1 index (0.2 ≈ one coarse step), so time direction
        // wins there.
        let mut bw = BitWriter::new();
        encode_ajoc(&mut bw, 0, &ctrl, &q, false, &mut enc_state).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let decoded = decode_ajoc(&mut br, 1, 1, &mut dec_state).unwrap();
        for dp in 0..2 {
            for pb in 0..nb {
                let want = dequantize_dry(q.dry_q[0][dp][0][pb] as u32, quant);
                assert!((decoded.matrices.dry_dq[0][dp][0][pb] - want).abs() < 1e-12);
            }
        }
        assert_eq!(enc_state.dry_prev, dec_state.dry_prev);
    }
}
