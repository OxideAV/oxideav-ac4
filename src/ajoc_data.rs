//! A-JOC parameter payload — ETSI TS 103 190-2 §6.2.5.1 `ajoc()` +
//! §6.2.5.3 `ajoc_data()` and the full frame-level decode driver.
//!
//! This module joins the two halves that already exist in the crate:
//! the [`crate::ajoc_huffman`] codeword layer (§6.2.5.5, Annex A.1.1
//! codebooks) and the [`crate::ajoc`] parameter processing (§5.7.3
//! differential decode / dequantize / interpolate / Table 49 spatial
//! reconstruction). [`decode_ajoc`] takes an `ajoc()` bitstream element
//! plus the persistent [`AjocDiffState`] and returns the per-data-point
//! dequantized dry / wet matrices ready for
//! [`crate::ajoc::ajoc_reconstruct`], which turns downmix QMF signals
//! into reconstructed output objects.
//!
//! The encoder side ([`write_ajoc`], [`write_ajoc_ctrl_info`],
//! [`write_ajoc_data`], [`encode_ajoc_deltas_freq`] /
//! [`encode_ajoc_deltas_time`]) is the exact bitstream inverse and is
//! used for the roundtrip and GOP-consistency tests: an I-frame codes
//! every row frequency-differentially (`ajoc_b_nodt = 1` on the first
//! data point forces DIFF_FREQ), P-frames may switch rows to DIFF_TIME
//! against the previous frame's quantised matrices carried in
//! [`AjocDiffState`].

use crate::ajoc::{
    dequantize_dry, dequantize_wet, differential_decode_dry, differential_decode_wet,
    parse_ajoc_ctrl_info, AjocCtrlInfo, AjocDequantMatrices, AjocDiffState, AjocDiffType,
    AjocMatrixKind, AjocQuantMode,
};
use crate::ajoc_huffman::{ajoc_huff_data, write_ajoc_huff_data};
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

/// Maximum `ajoc_num_bands` (Table 100) — sizing for the persistent
/// differential state.
pub const AJOC_MAX_BANDS: usize = 23;

/// One Huffman-decoded matrix row: the coding direction plus the
/// per-band values (`a_huff_data`, §6.2.5.5). `None` when the row is
/// not transmitted (sparse path with the present bit cleared).
pub type AjocHuffRow = Option<(AjocDiffType, Vec<i32>)>;

/// Huffman-decoded `ajoc_data()` payload for one frame (§6.2.5.3).
#[derive(Clone, Debug, Default)]
pub struct AjocData {
    /// `ajoc_b_nodt` — when set, time-differential coding is forbidden
    /// on data point 0 (`b_dfonly`) (§6.3.6.3.1).
    pub b_nodt: bool,
    /// `mix_mtx_dry[o][dp][ch]` rows (empty vectors for absent objects).
    pub mix_mtx_dry: Vec<Vec<Vec<AjocHuffRow>>>,
    /// `mix_mtx_wet[o][dp][de]` rows.
    pub mix_mtx_wet: Vec<Vec<Vec<AjocHuffRow>>>,
}

/// Is the dry row `(o, ch)` transmitted per the sparse gating of
/// §6.2.5.3? (`sparse_select == 0` ⇒ always.)
fn dry_row_present(ctrl: &AjocCtrlInfo, o: usize, ch: usize) -> bool {
    !ctrl.sparse_select[o]
        || ctrl
            .mix_mtx_dry_present
            .get(o)
            .and_then(|r| r.get(ch))
            .copied()
            .unwrap_or(false)
}

/// Is the wet row `(o, de)` transmitted per the sparse gating of
/// §6.2.5.3? (`sparse_select == 0` ⇒ always.)
fn wet_row_present(ctrl: &AjocCtrlInfo, o: usize, de: usize) -> bool {
    !ctrl.sparse_select[o]
        || ctrl
            .mix_mtx_wet_present
            .get(o)
            .and_then(|r| r.get(de))
            .copied()
            .unwrap_or(false)
}

/// Walk `ajoc_data(num_dmx_signals, num_umx_signals)` (§6.2.5.3),
/// Huffman-decoding every transmitted dry / wet matrix row via
/// `ajoc_huff_data()` (§6.2.5.5).
pub fn parse_ajoc_data(
    br: &mut BitReader<'_>,
    ctrl: &AjocCtrlInfo,
    num_dmx_signals: u32,
    ajoc_num_decorr: u32,
) -> Result<AjocData> {
    let num_umx = ctrl.object_present.len();
    let num_dmx = num_dmx_signals as usize;
    let num_decorr = ajoc_num_decorr as usize;
    let num_dp = ctrl.data_point_info.num_dpoints as usize;

    let b_nodt = br.read_bit()?;
    let mut mix_mtx_dry = vec![Vec::new(); num_umx];
    let mut mix_mtx_wet = vec![Vec::new(); num_umx];

    for o in 0..num_umx {
        if !ctrl.object_present[o] {
            continue;
        }
        let nb = ctrl.num_bands[o] as usize;
        let qs = ctrl.quant_select[o];
        for dp in 0..num_dp {
            let b_dfonly = dp == 0 && b_nodt;
            let mut dry_rows: Vec<AjocHuffRow> = Vec::with_capacity(num_dmx);
            for ch in 0..num_dmx {
                if dry_row_present(ctrl, o, ch) {
                    dry_rows.push(Some(ajoc_huff_data(
                        br,
                        AjocMatrixKind::Dry,
                        nb,
                        qs,
                        b_dfonly,
                    )?));
                } else {
                    dry_rows.push(None);
                }
            }
            let mut wet_rows: Vec<AjocHuffRow> = Vec::with_capacity(num_decorr);
            for de in 0..num_decorr {
                if wet_row_present(ctrl, o, de) {
                    wet_rows.push(Some(ajoc_huff_data(
                        br,
                        AjocMatrixKind::Wet,
                        nb,
                        qs,
                        b_dfonly,
                    )?));
                } else {
                    wet_rows.push(None);
                }
            }
            mix_mtx_dry[o].push(dry_rows);
            mix_mtx_wet[o].push(wet_rows);
        }
    }

    Ok(AjocData {
        b_nodt,
        mix_mtx_dry,
        mix_mtx_wet,
    })
}

/// A fully decoded `ajoc()` element (§6.2.5.1).
#[derive(Clone, Debug)]
pub struct AjocFrame {
    /// `ajoc_num_decorr` (3 bits).
    pub num_decorr: u32,
    /// `ajoc_ctrl_info()` side information.
    pub ctrl: AjocCtrlInfo,
    /// Huffman-decoded `ajoc_data()` payload.
    pub data: AjocData,
    /// Dequantized dry / wet matrices per data point — the direct input
    /// to [`crate::ajoc::ajoc_reconstruct`].
    pub matrices: AjocDequantMatrices,
}

/// Allocate the persistent A-JOC differential state for `decode_ajoc`,
/// sized to the Table 100 maximum band count so any per-frame
/// `ajoc_num_bands` fits.
pub fn new_ajoc_diff_state(num_umx: usize, num_dmx: usize, num_decorr: usize) -> AjocDiffState {
    AjocDiffState::new(num_umx, num_dmx, num_decorr, AJOC_MAX_BANDS)
}

/// Run the §5.7.3.2 Table 43 differential decode + §5.7.3.3 dequantize
/// over a parsed frame, producing the per-data-point dequantized
/// matrices. `state` carries `mtx_*_q_prev` across frames (DIFF_TIME).
pub fn dequantize_ajoc_frame(
    ctrl: &AjocCtrlInfo,
    data: &AjocData,
    num_dmx_signals: u32,
    ajoc_num_decorr: u32,
    state: &mut AjocDiffState,
) -> Result<AjocDequantMatrices> {
    let num_umx = ctrl.object_present.len();
    let num_dmx = num_dmx_signals as usize;
    let num_decorr = ajoc_num_decorr as usize;
    let num_dp = ctrl.data_point_info.num_dpoints as usize;

    if state.dry_prev.len() < num_umx || state.wet_prev.len() < num_umx {
        return Err(Error::invalid("ac4: A-JOC diff state undersized (objects)"));
    }

    let mut matrices = AjocDequantMatrices {
        dry_dq: vec![Vec::new(); num_umx],
        wet_dq: vec![Vec::new(); num_umx],
        num_bands: ctrl.num_bands.clone(),
    };

    for o in 0..num_umx {
        if !ctrl.object_present[o] {
            continue;
        }
        let nb = ctrl.num_bands[o] as usize;
        if nb > AJOC_MAX_BANDS {
            return Err(Error::invalid("ac4: ajoc_num_bands out of range"));
        }
        let qs = ctrl.quant_select[o];
        for dp in 0..num_dp {
            // Dry matrix rows.
            let mut dry_dq = Vec::with_capacity(num_dmx);
            for ch in 0..num_dmx {
                let row = data
                    .mix_mtx_dry
                    .get(o)
                    .and_then(|d| d.get(dp))
                    .and_then(|d| d.get(ch))
                    .ok_or_else(|| Error::invalid("ac4: missing A-JOC dry row"))?;
                let prev = state
                    .dry_prev
                    .get_mut(o)
                    .and_then(|p| p.get_mut(ch))
                    .ok_or_else(|| Error::invalid("ac4: A-JOC diff state undersized (dry)"))?;
                if prev.len() < nb {
                    return Err(Error::invalid("ac4: A-JOC diff state undersized (bands)"));
                }
                let (diff_type, deltas, present) = match row {
                    Some((dt, d)) => (*dt, d.as_slice(), true),
                    None => (AjocDiffType::Freq, &[][..], false),
                };
                let q = differential_decode_dry(deltas, diff_type, qs, present, &mut prev[..nb]);
                let nquant = qs.nquant(AjocMatrixKind::Dry) as i32;
                if q.iter().any(|&v| !(0..nquant).contains(&v)) {
                    return Err(Error::invalid("ac4: A-JOC dry index out of range"));
                }
                dry_dq.push(q.iter().map(|&v| dequantize_dry(v as u32, qs)).collect());
            }
            matrices.dry_dq[o].push(dry_dq);

            // Wet matrix rows.
            let mut wet_dq = Vec::with_capacity(num_decorr);
            for de in 0..num_decorr {
                let row = data
                    .mix_mtx_wet
                    .get(o)
                    .and_then(|w| w.get(dp))
                    .and_then(|w| w.get(de))
                    .ok_or_else(|| Error::invalid("ac4: missing A-JOC wet row"))?;
                let prev = state
                    .wet_prev
                    .get_mut(o)
                    .and_then(|p| p.get_mut(de))
                    .ok_or_else(|| Error::invalid("ac4: A-JOC diff state undersized (wet)"))?;
                if prev.len() < nb {
                    return Err(Error::invalid("ac4: A-JOC diff state undersized (bands)"));
                }
                let (diff_type, deltas, present) = match row {
                    Some((dt, d)) => (*dt, d.as_slice(), true),
                    None => (AjocDiffType::Freq, &[][..], false),
                };
                let q = differential_decode_wet(deltas, diff_type, qs, present, &mut prev[..nb]);
                let nquant = qs.nquant(AjocMatrixKind::Wet) as i32;
                if q.iter().any(|&v| !(0..nquant).contains(&v)) {
                    return Err(Error::invalid("ac4: A-JOC wet index out of range"));
                }
                wet_dq.push(q.iter().map(|&v| dequantize_wet(v as u32, qs)).collect());
            }
            matrices.wet_dq[o].push(wet_dq);
        }
    }

    Ok(matrices)
}

/// Decode a full `ajoc(num_dmx_signals, num_umx_signals)` element
/// (§6.2.5.1): `ajoc_num_decorr`, `ajoc_ctrl_info()`, `ajoc_data()`,
/// then differential-decode + dequantize into reconstruction matrices.
pub fn decode_ajoc(
    br: &mut BitReader<'_>,
    num_dmx_signals: u32,
    num_umx_signals: u32,
    state: &mut AjocDiffState,
) -> Result<AjocFrame> {
    let num_decorr = br.read_u32(3)?;
    let ctrl = parse_ajoc_ctrl_info(br, num_dmx_signals, num_decorr, num_umx_signals)?;
    let data = parse_ajoc_data(br, &ctrl, num_dmx_signals, num_decorr)?;
    let matrices = dequantize_ajoc_frame(&ctrl, &data, num_dmx_signals, num_decorr, state)?;
    Ok(AjocFrame {
        num_decorr,
        ctrl,
        data,
        matrices,
    })
}

// ---------------------------------------------------------------------
// Encoder side — exact bitstream inverses
// ---------------------------------------------------------------------

/// Write `ajoc_ctrl_info()` (§6.2.5.2) — the exact inverse of
/// [`crate::ajoc::parse_ajoc_ctrl_info`].
pub fn write_ajoc_ctrl_info(bw: &mut BitWriter, ctrl: &AjocCtrlInfo) -> Result<()> {
    for &e in &ctrl.decorr_enable {
        bw.write_bit(e);
    }
    for &p in &ctrl.object_present {
        bw.write_bit(p);
    }
    let dpi = &ctrl.data_point_info;
    bw.write_u32(dpi.num_dpoints, 2);
    for dp in 0..dpi.num_dpoints as usize {
        bw.write_u32(dpi.start_pos[dp], 5);
        let ramp = dpi.ramp_len[dp];
        if ramp == 0 || ramp > 64 {
            return Err(Error::invalid("ac4: ajoc_ramp_len out of range"));
        }
        bw.write_u32(ramp - 1, 6);
    }
    if dpi.num_dpoints != 0 {
        for o in 0..ctrl.object_present.len() {
            if !ctrl.object_present[o] {
                continue;
            }
            bw.write_u32(ctrl.num_bands_code[o] as u32, 3);
            bw.write_bit(ctrl.quant_select[o] == AjocQuantMode::Coarse);
            bw.write_bit(ctrl.sparse_select[o]);
            if ctrl.sparse_select[o] {
                for &p in &ctrl.mix_mtx_dry_present[o] {
                    bw.write_bit(p);
                }
                for (d, &p) in ctrl.mix_mtx_wet_present[o].iter().enumerate() {
                    if ctrl.decorr_enable[d] {
                        bw.write_bit(p);
                    } else if p {
                        return Err(Error::invalid(
                            "ac4: wet row present on disabled decorrelator",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Write `ajoc_data()` (§6.2.5.3) from Huffman-decoded rows — the exact
/// inverse of [`parse_ajoc_data`]. Row presence must match the sparse
/// gating in `ctrl`.
pub fn write_ajoc_data(bw: &mut BitWriter, ctrl: &AjocCtrlInfo, data: &AjocData) -> Result<()> {
    bw.write_bit(data.b_nodt);
    let num_dp = ctrl.data_point_info.num_dpoints as usize;
    for o in 0..ctrl.object_present.len() {
        if !ctrl.object_present[o] {
            continue;
        }
        let qs = ctrl.quant_select[o];
        for dp in 0..num_dp {
            let b_dfonly = dp == 0 && data.b_nodt;
            let dry = data
                .mix_mtx_dry
                .get(o)
                .and_then(|d| d.get(dp))
                .ok_or_else(|| Error::invalid("ac4: missing A-JOC dry data point"))?;
            for (ch, row) in dry.iter().enumerate() {
                match (dry_row_present(ctrl, o, ch), row) {
                    (true, Some((dt, vals))) => {
                        write_ajoc_huff_data(bw, AjocMatrixKind::Dry, qs, b_dfonly, *dt, vals)?;
                    }
                    (false, None) => {}
                    _ => {
                        return Err(Error::invalid(
                            "ac4: A-JOC dry row presence mismatches sparse gating",
                        ))
                    }
                }
            }
            let wet = data
                .mix_mtx_wet
                .get(o)
                .and_then(|w| w.get(dp))
                .ok_or_else(|| Error::invalid("ac4: missing A-JOC wet data point"))?;
            for (de, row) in wet.iter().enumerate() {
                match (wet_row_present(ctrl, o, de), row) {
                    (true, Some((dt, vals))) => {
                        write_ajoc_huff_data(bw, AjocMatrixKind::Wet, qs, b_dfonly, *dt, vals)?;
                    }
                    (false, None) => {}
                    _ => {
                        return Err(Error::invalid(
                            "ac4: A-JOC wet row presence mismatches sparse gating",
                        ))
                    }
                }
            }
        }
    }
    Ok(())
}

/// Write a full `ajoc()` element (§6.2.5.1).
pub fn write_ajoc(
    bw: &mut BitWriter,
    num_decorr: u32,
    ctrl: &AjocCtrlInfo,
    data: &AjocData,
) -> Result<()> {
    if num_decorr > 7 {
        return Err(Error::invalid("ac4: ajoc_num_decorr out of range"));
    }
    bw.write_u32(num_decorr, 3);
    write_ajoc_ctrl_info(bw, ctrl)?;
    write_ajoc_data(bw, ctrl, data)
}

// ---------------------------------------------------------------------
// Encoder-side delta derivation (Table 43 inverses)
// ---------------------------------------------------------------------

/// Derive the DIFF_FREQ `a_huff_data` row from absolute quantised
/// indices: band 0 absolute, then `(q[i] - q[i-1]) mod nquant`
/// (the inverse of the Table 43 running modulo sum).
pub fn encode_ajoc_deltas_freq(q: &[i32], kind: AjocMatrixKind, mode: AjocQuantMode) -> Vec<i32> {
    let nquant = mode.nquant(kind) as i32;
    let mut out = Vec::with_capacity(q.len());
    for (i, &v) in q.iter().enumerate() {
        if i == 0 {
            out.push(v);
        } else {
            out.push((v - q[i - 1]).rem_euclid(nquant));
        }
    }
    out
}

/// Derive the DIFF_TIME `a_huff_data` row from absolute quantised
/// indices and the previous frame's row: `q[i] - prev[i]`.
pub fn encode_ajoc_deltas_time(q: &[i32], prev: &[i32]) -> Vec<i32> {
    q.iter().zip(prev.iter()).map(|(&v, &p)| v - p).collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::ajoc::{ajoc_reconstruct, AjocDataPointInfo, AjocGeometry, AjocReconState, Cplx};

    /// Build a simple non-sparse ctrl-info: every object present, one
    /// data point, given band code / quant mode.
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

    /// Frequency-coded AjocData from absolute quantised rows.
    fn freq_data(
        ctrl: &AjocCtrlInfo,
        dry_q: &[Vec<Vec<i32>>],
        wet_q: &[Vec<Vec<i32>>],
        quant: AjocQuantMode,
        b_nodt: bool,
    ) -> AjocData {
        let num_umx = ctrl.object_present.len();
        let mut data = AjocData {
            b_nodt,
            mix_mtx_dry: vec![Vec::new(); num_umx],
            mix_mtx_wet: vec![Vec::new(); num_umx],
        };
        for o in 0..num_umx {
            let dry_rows = dry_q[o]
                .iter()
                .map(|q| {
                    Some((
                        AjocDiffType::Freq,
                        encode_ajoc_deltas_freq(q, AjocMatrixKind::Dry, quant),
                    ))
                })
                .collect();
            let wet_rows = wet_q[o]
                .iter()
                .map(|q| {
                    Some((
                        AjocDiffType::Freq,
                        encode_ajoc_deltas_freq(q, AjocMatrixKind::Wet, quant),
                    ))
                })
                .collect();
            data.mix_mtx_dry[o].push(dry_rows);
            data.mix_mtx_wet[o].push(wet_rows);
        }
        data
    }

    #[test]
    fn ajoc_element_bitstream_roundtrip() {
        // 2 umx objects, 2 dmx channels, 1 decorrelator, 5 bands, coarse.
        let quant = AjocQuantMode::Coarse;
        let ctrl = simple_ctrl(2, 1, 5, quant);
        let dry_q = vec![
            vec![vec![25, 30, 20, 25, 25], vec![10, 10, 10, 50, 0]],
            vec![vec![0, 1, 2, 3, 4], vec![50, 49, 48, 47, 46]],
        ];
        let wet_q = vec![vec![vec![10, 12, 8, 10, 10]], vec![vec![0, 20, 10, 5, 15]]];
        let data = freq_data(&ctrl, &dry_q, &wet_q, quant, true);

        let mut bw = BitWriter::new();
        write_ajoc(&mut bw, 1, &ctrl, &data).unwrap();
        let bytes = bw.finish();

        let mut state = new_ajoc_diff_state(2, 2, 1);
        let mut br = BitReader::new(&bytes);
        let frame = decode_ajoc(&mut br, 2, 2, &mut state).unwrap();

        assert_eq!(frame.num_decorr, 1);
        assert_eq!(frame.ctrl, ctrl);
        assert_eq!(frame.data.b_nodt, data.b_nodt);
        assert_eq!(frame.data.mix_mtx_dry, data.mix_mtx_dry);
        assert_eq!(frame.data.mix_mtx_wet, data.mix_mtx_wet);

        // Dequantized matrices reproduce the absolute quantised rows.
        for o in 0..2 {
            for ch in 0..2 {
                for pb in 0..5 {
                    let want = dequantize_dry(dry_q[o][ch][pb] as u32, quant);
                    let got = frame.matrices.dry_dq[o][0][ch][pb];
                    assert!((got - want).abs() < 1e-12, "dry o{o} ch{ch} pb{pb}");
                }
            }
            for pb in 0..5 {
                let want = dequantize_wet(wet_q[o][0][pb] as u32, quant);
                let got = frame.matrices.wet_dq[o][0][0][pb];
                assert!((got - want).abs() < 1e-12, "wet o{o} pb{pb}");
            }
        }
        // Diff state carries the quantised rows for the next frame.
        assert_eq!(&state.dry_prev[0][0][..5], &dry_q[0][0][..]);
        assert_eq!(&state.wet_prev[1][0][..5], &wet_q[1][0][..]);
    }

    #[test]
    fn sparse_rows_default_to_zero_centre() {
        let quant = AjocQuantMode::Fine;
        let mut ctrl = simple_ctrl(1, 2, 6, quant); // 3 bands
        ctrl.sparse_select[0] = true;
        ctrl.mix_mtx_dry_present[0] = vec![true, false]; // ch1 absent
        ctrl.mix_mtx_wet_present[0] = vec![false, true]; // de0 absent

        let mut data = AjocData {
            b_nodt: true,
            mix_mtx_dry: vec![vec![vec![
                Some((
                    AjocDiffType::Freq,
                    encode_ajoc_deltas_freq(&[50, 60, 40], AjocMatrixKind::Dry, quant),
                )),
                None,
            ]]],
            mix_mtx_wet: vec![vec![vec![
                None,
                Some((
                    AjocDiffType::Freq,
                    encode_ajoc_deltas_freq(&[20, 25, 15], AjocMatrixKind::Wet, quant),
                )),
            ]]],
        };

        let mut bw = BitWriter::new();
        write_ajoc(&mut bw, 2, &ctrl, &data).unwrap();
        let bytes = bw.finish();

        let mut state = new_ajoc_diff_state(1, 2, 2);
        let mut br = BitReader::new(&bytes);
        let frame = decode_ajoc(&mut br, 2, 1, &mut state).unwrap();

        // Transmitted rows decode; absent rows sit at the dequantized
        // zero-centre (0.0 for both dry fine 50 and wet fine 20).
        assert!((frame.matrices.dry_dq[0][0][0][0] - dequantize_dry(50, quant)).abs() < 1e-12);
        for pb in 0..3 {
            assert!(frame.matrices.dry_dq[0][0][1][pb].abs() < 1e-12);
            assert!(frame.matrices.wet_dq[0][0][0][pb].abs() < 1e-12);
        }
        assert!((frame.matrices.wet_dq[0][0][1][1] - dequantize_wet(25, quant)).abs() < 1e-12);

        // Presence mismatch is rejected on the encode side.
        data.mix_mtx_dry[0][0][1] = Some((AjocDiffType::Freq, vec![0, 0, 0]));
        let mut bw = BitWriter::new();
        assert!(write_ajoc(&mut bw, 2, &ctrl, &data).is_err());
    }

    #[test]
    fn p_frame_time_diff_chain_matches_absolute_coding() {
        // GOP consistency: frame 1 codes rows DIFF_FREQ (I-frame style),
        // frame 2 codes the evolved rows DIFF_TIME against frame 1. The
        // decoded quantised chain must equal direct absolute coding.
        let quant = AjocQuantMode::Coarse;
        let ctrl = simple_ctrl(1, 1, 4, quant); // 7 bands
        let q1_dry = vec![25, 27, 23, 25, 26, 24, 25];
        let q1_wet = vec![10, 11, 9, 10, 10, 12, 8];
        let q2_dry = vec![26, 25, 23, 27, 24, 24, 30];
        let q2_wet = vec![9, 11, 10, 10, 13, 12, 5];

        // Frame 1: freq direction (b_nodt = 1 → dfonly on dp 0).
        let data1 = freq_data(
            &ctrl,
            &[vec![q1_dry.clone()]],
            &[vec![q1_wet.clone()]],
            quant,
            true,
        );
        // Frame 2: time direction against frame 1.
        let data2 = AjocData {
            b_nodt: false,
            mix_mtx_dry: vec![vec![vec![Some((
                AjocDiffType::Time,
                encode_ajoc_deltas_time(&q2_dry, &q1_dry),
            ))]]],
            mix_mtx_wet: vec![vec![vec![Some((
                AjocDiffType::Time,
                encode_ajoc_deltas_time(&q2_wet, &q1_wet),
            ))]]],
        };

        let mut state = new_ajoc_diff_state(1, 1, 1);
        for (data, want_dry, want_wet) in [(&data1, &q1_dry, &q1_wet), (&data2, &q2_dry, &q2_wet)] {
            let mut bw = BitWriter::new();
            write_ajoc(&mut bw, 1, &ctrl, data).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let frame = decode_ajoc(&mut br, 1, 1, &mut state).unwrap();
            for pb in 0..7 {
                let want = dequantize_dry(want_dry[pb] as u32, quant);
                assert!((frame.matrices.dry_dq[0][0][0][pb] - want).abs() < 1e-12);
                let want = dequantize_wet(want_wet[pb] as u32, quant);
                assert!((frame.matrices.wet_dq[0][0][0][pb] - want).abs() < 1e-12);
            }
        }
        // The time-coded frame is cheaper than re-coding frame 2
        // absolutely when the deltas are small... at minimum it decodes
        // identically, which is what the assertions above prove.
    }

    #[test]
    fn dfonly_forces_freq_on_first_data_point() {
        // b_nodt = 1 with a time-coded dp-0 row must be rejected by the
        // writer (the bitstream cannot express it).
        let quant = AjocQuantMode::Coarse;
        let ctrl = simple_ctrl(1, 0, 7, quant); // 1 band, no decorrelators
        let data = AjocData {
            b_nodt: true,
            mix_mtx_dry: vec![vec![vec![Some((AjocDiffType::Time, vec![0]))]]],
            mix_mtx_wet: vec![vec![vec![]]],
        };
        let mut bw = BitWriter::new();
        assert!(write_ajoc(&mut bw, 0, &ctrl, &data).is_err());
    }

    #[test]
    fn decoded_frame_drives_table_49_reconstruction() {
        // Full chain: bitstream → decode_ajoc → ajoc_reconstruct.
        // 1 object from 1 downmix channel, no decorrelators, 1 band.
        let quant = AjocQuantMode::Coarse;
        let ctrl = simple_ctrl(1, 0, 7, quant); // 1 band
                                                // Dry index 35 → dequantizes to (35-25)·0.20019531 ≈ 2.0019531.
        let data = freq_data(&ctrl, &[vec![vec![35]]], &[vec![]], quant, true);

        let mut bw = BitWriter::new();
        write_ajoc(&mut bw, 0, &ctrl, &data).unwrap();
        let bytes = bw.finish();

        let mut state = new_ajoc_diff_state(1, 1, 0);
        let mut br = BitReader::new(&bytes);
        let frame = decode_ajoc(&mut br, 1, 1, &mut state).unwrap();

        let geom = AjocGeometry {
            num_dmx: 1,
            num_umx: 1,
            num_decorr: 0,
            num_timeslots: 16,
            num_subbands: 64,
        };
        let mut x = vec![vec![vec![(0.0f64, 0.0f64); 1]; 64]; 16];
        for row in x.iter_mut() {
            row[7][0] = (1.0, -0.5);
        }
        let mut recon = AjocReconState::new(&geom);
        let z: Vec<Vec<Vec<Cplx>>> = ajoc_reconstruct(
            &x,
            &frame.matrices,
            &frame.ctrl.data_point_info,
            &frame.ctrl.object_present,
            &geom,
            &mut recon,
        );
        // After the 8-slot ramp converges, the object is the downmix
        // scaled by the dequantized coefficient times the Table 48
        // plateau factor (1 + 1/ramp_len).
        let coeff = dequantize_dry(35, quant) * (1.0 + 1.0 / 8.0);
        let got = z[15][7][0];
        assert!((got.0 - coeff).abs() < 1e-9, "re {} vs {}", got.0, coeff);
        assert!((got.1 + 0.5 * coeff).abs() < 1e-9);
        // Untouched subbands stay silent.
        assert!(z[15][3][0].0.abs() < 1e-12);
    }

    #[test]
    fn zero_data_points_yields_empty_matrices() {
        // num_dpoints = 0: ctrl carries no per-object fields, ajoc_data
        // carries only b_nodt.
        let ctrl = AjocCtrlInfo {
            decorr_enable: vec![true],
            object_present: vec![true, true],
            data_point_info: AjocDataPointInfo::default(),
            num_bands_code: vec![0; 2],
            num_bands: vec![0; 2],
            quant_select: vec![AjocQuantMode::Fine; 2],
            sparse_select: vec![false; 2],
            mix_mtx_dry_present: vec![Vec::new(); 2],
            mix_mtx_wet_present: vec![Vec::new(); 2],
        };
        let data = AjocData {
            b_nodt: false,
            mix_mtx_dry: vec![Vec::new(); 2],
            mix_mtx_wet: vec![Vec::new(); 2],
        };
        let mut bw = BitWriter::new();
        write_ajoc(&mut bw, 1, &ctrl, &data).unwrap();
        let bytes = bw.finish();
        // 3 bits num_decorr + 1 decorr_enable + 2 present + 2 num_dpoints
        // + 1 b_nodt = 9 bits → 2 bytes.
        assert_eq!(bytes.len(), 2);

        let mut state = new_ajoc_diff_state(2, 2, 1);
        let mut br = BitReader::new(&bytes);
        let frame = decode_ajoc(&mut br, 2, 2, &mut state).unwrap();
        assert!(frame.matrices.dry_dq.iter().all(|d| d.is_empty()));
        assert!(frame.matrices.wet_dq.iter().all(|w| w.is_empty()));
    }
}
