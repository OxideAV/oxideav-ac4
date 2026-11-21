//! ASF window / group layout and the complete `sf_data(ASF)` decode
//! chain for full and partial blocks (ETSI TS 103 190-1 §4.3.6.2.6,
//! §5.1.2 – §5.1.5).
//!
//! An `sf_info(ASF)` element describes one frame of one or more
//! transform blocks ("windows") that Pseudocode 3 collects into window
//! groups. [`AsfLayout`] resolves that description — the helper
//! elements `num_windows`, `window_to_group[]`, `num_win_in_group[]`,
//! `sect_sfb_offset[][]`, `win_offset[]`, `get_transf_length(g)` and
//! `get_max_sfb(g)` — and [`decode_sf_data`] walks one `sf_data(ASF)`
//! body on it: section data (Table 39), spectral data (Table 40),
//! scale factors (Table 41 / Pseudocode 21), spectral noise fill
//! (Table 42 / Pseudocodes 22 – 24) and the ungrouping of Pseudocode 25.
//!
//! Spectra leave this module in **window order** (`spec_reord`): for a
//! frame of one full block that is the plain line order; for a
//! block-switched frame it is the concatenation of every window's
//! lines, `frame_length` values in total, ready for the per-block
//! IMDCT of §5.5.

use std::cell::RefCell;

use oxideav_core::bits::BitReader;

use crate::asf::{AsfPsyInfo, AsfTransformInfo};
use crate::asf_data;
use crate::ssf_ac::SsfRandGenState;

/// `get_transf_length(g)` return value for a long frame (Pseudocode 2).
pub const LONG_FRAME_TL_CODE: u32 = 4;

/// `n_sect_bits` / `sect_esc_val` of `asf_section_data()` for a
/// `get_transf_length(g)` code (Table 39, Pseudocode 6).
pub fn sect_len_coding(tl_code: u32) -> (u32, u32) {
    if tl_code <= 2 { (3, 7) } else { (5, 31) }
}

/// `get_transf_length(g)` for a frame coded as one full block of `tl`
/// lines: 4 for the long frames of `frame_len_base ≥ 1536`; below
/// that Pseudocode 2 returns the `transf_length` field itself — 3 for
/// the 1024 / 960 / 768 frame lengths, 2 for 512 / 384 (Table 103).
pub fn full_block_tl_code(tl: u32) -> u32 {
    if tl >= 1536 {
        LONG_FRAME_TL_CODE
    } else if tl >= 768 {
        3
    } else {
        2
    }
}

/// One window group of an ASF frame.
#[derive(Debug, Clone)]
pub struct AsfGroup {
    /// `get_transf_length(g)` — the `transf_length` index of the
    /// group's windows, or [`LONG_FRAME_TL_CODE`] for a long frame.
    pub tl_code: u32,
    /// Transform length of the group's windows in samples.
    pub tl: u32,
    /// `get_max_sfb(g)`, capped at `num_sfb_48(tl)`.
    pub max_sfb: u32,
    /// Index of the group's first window.
    pub first_win: usize,
    /// `num_win_in_group[g]`.
    pub num_win: usize,
    /// Annex B `sfb_offset[]` for `tl`.
    pub sfbo: &'static [u16],
    /// `sect_sfb_offset[g][sfb] − group_offset` for `sfb` in
    /// `0..=max_sfb` (Pseudocode 4): `sfb_offset[sfb] · num_win`.
    pub sect_offs: Vec<u16>,
    /// `group_offset`: first line of the group in bitstream order.
    pub offset: usize,
}

/// Window / group layout of one `sf_info(ASF)` element.
#[derive(Debug, Clone)]
pub struct AsfLayout {
    /// Samples per frame (the sum of the window lengths).
    pub frame_len: usize,
    /// Transform length of every window, in time order.
    pub windows: Vec<usize>,
    /// `win_offset[win]`: first line of each window in window order.
    pub win_offset: Vec<usize>,
    /// The window groups, in bitstream order.
    pub groups: Vec<AsfGroup>,
}

impl AsfLayout {
    /// Resolve the layout of `ti` / `psy` with `max_sfb[0]` /
    /// `max_sfb[1]` as given (the caller picks `max_sfb` or
    /// `max_sfb_side` per Pseudocode 5). `None` when the combination
    /// has no Annex B table or no coded band.
    pub fn new(
        ti: &AsfTransformInfo,
        psy: &AsfPsyInfo,
        max_sfb_0: u32,
        max_sfb_1: u32,
    ) -> Option<Self> {
        let frame_len_base = if psy.frame_len_base != 0 {
            psy.frame_len_base
        } else if ti.b_long_frame {
            ti.transform_length_0
        } else {
            ti.transform_length_0 * psy.num_windows.max(1)
        };
        let windows = ti.windows(frame_len_base)?;
        let num_windows = windows.len();
        let frame_len: usize = windows.iter().sum();
        let mut win_offset = Vec::with_capacity(num_windows);
        let mut acc = 0usize;
        for &w in &windows {
            win_offset.push(acc);
            acc += w;
        }
        // Pseudocode 3: window_to_group[].
        let different = !ti.b_long_frame
            && frame_len_base >= 1536
            && ti.transf_length[0] != ti.transf_length[1];
        let num_windows_0 = if different {
            1usize << (3 - (ti.transf_length[0] & 3))
        } else {
            num_windows
        };
        let mut grouping: Vec<u8> = psy.scale_factor_grouping.clone();
        if different {
            // Shift the second half's bits by one and force a group
            // boundary between the two transform lengths.
            grouping.insert(num_windows_0 - 1, 0);
        }
        grouping.resize(num_windows.saturating_sub(1), 0);
        let mut window_to_group = vec![0usize; num_windows];
        let mut num_groups = 1usize;
        for i in 0..num_windows.saturating_sub(1) {
            if grouping[i] == 0 {
                num_groups += 1;
            }
            window_to_group[i + 1] = num_groups - 1;
        }
        let split_group = if different {
            window_to_group[num_windows_0]
        } else {
            usize::MAX
        };
        let mut groups = Vec::with_capacity(num_groups);
        let mut offset = 0usize;
        for g in 0..num_groups {
            let first_win = window_to_group.iter().position(|&x| x == g)?;
            let num_win = window_to_group.iter().filter(|&&x| x == g).count();
            let second = g >= split_group;
            let tl = windows[first_win] as u32;
            let tl_code = if ti.b_long_frame {
                LONG_FRAME_TL_CODE
            } else if second {
                ti.transf_length[1]
            } else {
                ti.transf_length[0]
            };
            let cap = crate::tables::num_sfb_48(tl)?;
            let max_sfb = if second { max_sfb_1 } else { max_sfb_0 }.min(cap);
            let sfbo = crate::sfb_offset::sfb_offset_48(tl)?;
            let sect_offs: Vec<u16> = sfbo[..=max_sfb as usize]
                .iter()
                .map(|&o| (o as usize * num_win) as u16)
                .collect();
            let len = *sect_offs.last()? as usize;
            groups.push(AsfGroup {
                tl_code,
                tl,
                max_sfb,
                first_win,
                num_win,
                sfbo,
                sect_offs,
                offset,
            });
            offset += len;
        }
        Some(Self {
            frame_len,
            windows,
            win_offset,
            groups,
        })
    }

    /// Layout of a frame coded as one full block of `tl` lines — a
    /// long frame for `tl ≥ 1536`, the full-length `transf_length`
    /// code below that (see [`full_block_tl_code`]).
    pub fn full_block(tl: u32, max_sfb: u32) -> Option<Self> {
        let code = full_block_tl_code(tl);
        let long = code == LONG_FRAME_TL_CODE;
        let ti = AsfTransformInfo {
            b_long_frame: long,
            transf_length: if long { [0; 2] } else { [code; 2] },
            transform_length_0: tl,
            transform_length_1: tl,
        };
        let psy = AsfPsyInfo {
            num_windows: 1,
            num_window_groups: 1,
            frame_len_base: tl,
            ..AsfPsyInfo::default()
        };
        Self::new(&ti, &psy, max_sfb, max_sfb)
    }

    /// Alias of [`Self::full_block`].
    pub fn long(tl: u32, max_sfb: u32) -> Option<Self> {
        Self::full_block(tl, max_sfb)
    }

    /// `true` when the frame is one full block.
    pub fn is_single_block(&self) -> bool {
        self.windows.len() == 1
    }

    /// `get_max_sfb(g)` for every group — the grid `chparam_info()`
    /// elements are walked on.
    pub fn grid(&self) -> Vec<u32> {
        self.groups.iter().map(|g| g.max_sfb).collect()
    }

    /// Number of coded lines (bitstream order).
    pub fn coded_len(&self) -> usize {
        self.groups
            .last()
            .map(|g| g.offset + *g.sect_offs.last().unwrap_or(&0) as usize)
            .unwrap_or(0)
    }

    /// Length of the window-ordered spectrum: the coded extent for a
    /// single full block, `frame_len` for a block-switched frame.
    pub fn out_len(&self) -> usize {
        if self.is_single_block() {
            self.coded_len()
        } else {
            self.frame_len
        }
    }

    /// Bitstream-order line range of band `sfb` in group `g`.
    pub fn coded_band(&self, g: usize, sfb: usize) -> std::ops::Range<usize> {
        let grp = &self.groups[g];
        grp.offset + grp.sect_offs[sfb] as usize..grp.offset + grp.sect_offs[sfb + 1] as usize
    }

    /// Window-order line ranges of band `sfb` in group `g` (one range
    /// per window of the group).
    pub fn band_ranges(
        &self,
        g: usize,
        sfb: usize,
    ) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let grp = &self.groups[g];
        let (lo, hi) = (grp.sfbo[sfb] as usize, grp.sfbo[sfb + 1] as usize);
        (0..grp.num_win).map(move |w| {
            let base = self.win_offset[grp.first_win + w];
            base + lo..base + hi
        })
    }

    /// Pseudocode 25: reorder a bitstream-order spectrum into window
    /// order.
    pub fn ungroup(&self, coded: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_len()];
        let mut k = 0usize;
        for (g, grp) in self.groups.iter().enumerate() {
            for sfb in 0..grp.max_sfb as usize {
                for r in self.band_ranges(g, sfb) {
                    for o in &mut out[r] {
                        *o = coded.get(k).copied().unwrap_or(0.0);
                        k += 1;
                    }
                }
            }
        }
        out
    }

    /// Inverse of [`Self::ungroup`]: collect a window-order spectrum
    /// into bitstream order.
    pub fn group(&self, reord: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.coded_len());
        for (g, grp) in self.groups.iter().enumerate() {
            for sfb in 0..grp.max_sfb as usize {
                for r in self.band_ranges(g, sfb) {
                    out.extend(r.map(|i| reord.get(i).copied().unwrap_or(0.0)));
                }
            }
        }
        out
    }
}

/// An encoder-side description of one `sf_info(ASF, 0, 0)` element:
/// the block arrangement of a frame, its window grouping and the
/// per-half band counts. [`BlockPlan::write_sf_info`] emits it,
/// [`BlockPlan::layout`] resolves the layout the bodies are coded on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPlan {
    /// `frame_len_base`.
    pub frame_len: u32,
    /// `b_long_frame` (only meaningful for `frame_len ≥ 1536`).
    pub long: bool,
    /// `transf_length[0]` / `transf_length[1]` (equal below 1536).
    pub transf_length: [u32; 2],
    /// The transmitted `scale_factor_grouping` bits (`n_grp_bits` of
    /// them, Tables 109 / 110).
    pub grouping: Vec<u8>,
    /// `max_sfb[0]` / `max_sfb[1]`.
    pub max_sfb: [u32; 2],
}

impl BlockPlan {
    /// One full block of `frame_len` lines.
    pub fn full_block(frame_len: u32, max_sfb: u32) -> Self {
        let code = full_block_tl_code(frame_len);
        Self {
            frame_len,
            long: code == LONG_FRAME_TL_CODE,
            transf_length: if code == LONG_FRAME_TL_CODE {
                [0; 2]
            } else {
                [code; 2]
            },
            grouping: Vec::new(),
            max_sfb: [max_sfb; 2],
        }
    }

    /// The `asf_transform_info()` this plan describes.
    pub fn transform_info(&self) -> AsfTransformInfo {
        let len = |code: u32| {
            if self.long {
                self.frame_len
            } else {
                crate::asf::resolve_transf_length(self.frame_len, false, code)
            }
        };
        AsfTransformInfo {
            b_long_frame: self.long,
            transf_length: self.transf_length,
            transform_length_0: len(self.transf_length[0]),
            transform_length_1: len(self.transf_length[1]),
        }
    }

    /// `true` when the two frame halves use different transform lengths.
    pub fn different_framing(&self) -> bool {
        !self.long && self.frame_len >= 1536 && self.transf_length[0] != self.transf_length[1]
    }

    /// `n_grp_bits` (Tables 109 / 110).
    pub fn n_grp_bits(&self) -> u32 {
        if self.long {
            0
        } else if self.frame_len >= 1536 {
            crate::asf::n_grp_bits_ge_1536(self.transf_length[0], self.transf_length[1])
        } else {
            crate::asf::n_grp_bits_lt_1536(self.frame_len, self.transf_length[0])
        }
    }

    /// The `asf_psy_info()` this plan describes.
    pub fn psy_info(&self) -> AsfPsyInfo {
        let diff = self.different_framing();
        let n_grp = self.n_grp_bits();
        let mut grouping = self.grouping.clone();
        grouping.resize(n_grp as usize, 1);
        let single = self.long || (self.frame_len < 1536 && n_grp == 0);
        AsfPsyInfo {
            b_different_framing: diff,
            max_sfb_0: self.max_sfb[0],
            max_sfb_1: self.max_sfb[1],
            num_windows: if single {
                1
            } else {
                n_grp + 1 + u32::from(diff)
            },
            num_window_groups: if single {
                1
            } else {
                1 + grouping.iter().filter(|&&b| b == 0).count() as u32 + u32::from(diff)
            },
            scale_factor_grouping: grouping,
            frame_len_base: self.frame_len,
            ..AsfPsyInfo::default()
        }
    }

    /// The layout the frame's `sf_data(ASF)` bodies are coded on.
    pub fn layout(&self) -> Option<AsfLayout> {
        AsfLayout::new(
            &self.transform_info(),
            &self.psy_info(),
            self.max_sfb[0],
            self.max_sfb[1],
        )
    }

    /// The block lengths of the frame in time order.
    pub fn windows(&self) -> Vec<usize> {
        self.transform_info()
            .windows(self.frame_len)
            .unwrap_or_else(|| vec![self.frame_len as usize])
    }

    /// Emit `sf_info(ASF, 0, 0)`: `asf_transform_info()` (Table 37)
    /// then `asf_psy_info(0, 0)` (Table 38).
    pub fn write_sf_info(&self, bw: &mut oxideav_core::bits::BitWriter) {
        let ti = self.transform_info();
        if self.frame_len >= 1536 {
            bw.write_bit(self.long);
            if !self.long {
                bw.write_u32(self.transf_length[0], 2);
                bw.write_u32(self.transf_length[1], 2);
            }
        } else {
            bw.write_u32(self.transf_length[0], 2);
        }
        let bits = |tl: u32| crate::tables::n_msfb_bits_48(tl).map(|b| b.0).unwrap_or(6);
        bw.write_u32(self.max_sfb[0], bits(ti.transform_length_0));
        if self.different_framing() {
            bw.write_u32(self.max_sfb[1], bits(ti.transform_length_1));
        }
        let n_grp = self.n_grp_bits() as usize;
        for i in 0..n_grp {
            bw.write_u32(self.grouping.get(i).copied().unwrap_or(1) as u32, 1);
        }
    }
}

thread_local! {
    static SNF_RNG: RefCell<SsfRandGenState> = RefCell::new(snf_state_for(0));
}

/// Pseudocode 24 `ResetRandGenStateSnf()`: the noise-fill generator
/// state for a frame's `sequence_counter`.
pub fn snf_state_for(sequence_counter: u32) -> SsfRandGenState {
    let start = 255u64 * (sequence_counter as u64 % 256);
    let a = start % 255;
    let b = (start / 255) % 256;
    let mut state_idx = (a + 1) * ((a + 2) + 2 * b) / 2 + 255 * b * (255 + b) / 2;
    if start % 130_560 >= 65_280 {
        state_idx += 128;
    }
    let current_idx = a * (a + 1) / 2 + b * 32_386;
    SsfRandGenState {
        offset_a: a as u8,
        offset_b: b as u8,
        state_idx: (state_idx % 256) as u8,
        current_idx: (current_idx % 256) as u8,
    }
}

/// Reset this thread's spectral-noise-fill generator for a frame
/// (§5.1.4.2: "initialized at the beginning of the decoding of an ASF
/// frame, using the sequence_counter value"). The state then runs on
/// through every `sf_data(ASF)` body decoded on the thread until the
/// next reset.
pub fn reset_snf_rng(sequence_counter: u32) {
    SNF_RNG.with(|r| *r.borrow_mut() = snf_state_for(sequence_counter));
}

/// Mean power of a band as `log2` (the `1,44269504 · log(band_rms)` of
/// Pseudocodes 22 / 23), or `None` for an all-zero band.
fn band_level(coded: &[f32]) -> Option<f32> {
    let p: f64 = coded.iter().map(|&v| v as f64 * v as f64).sum();
    (p > 0.0).then(|| (p / coded.len().max(1) as f64).log2() as f32)
}

/// `delta` escape of `dpcm_snf` (index 0): no noise fill, level kept.
pub const SNF_NO_FILL_IDX: i32 = 0;
/// Offset between a `dpcm_snf` codeword index and its level delta.
pub const SNF_DELTA_OFFSET: i32 = 17;

/// §5.1.4.2 spectral noise fill (Pseudocodes 22 + 23) on a
/// bitstream-order spectrum.
pub fn apply_snf(coded: &mut [f32], layout: &AsfLayout, dpcm_snf: &[Vec<i32>]) {
    // Pseudocode 22: the reference level is the first band with a
    // non-zero line.
    let mut previous = -1000.0f32;
    'find: for (g, grp) in layout.groups.iter().enumerate() {
        for sfb in 0..grp.max_sfb as usize {
            if let Some(l) = band_level(&coded[layout.coded_band(g, sfb)]) {
                previous = l;
                break 'find;
            }
        }
    }
    SNF_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for (g, grp) in layout.groups.iter().enumerate() {
            for sfb in 0..grp.max_sfb as usize {
                let r = layout.coded_band(g, sfb);
                if let Some(l) = band_level(&coded[r.clone()]) {
                    previous = l;
                    continue;
                }
                let idx = dpcm_snf
                    .get(g)
                    .and_then(|v| v.get(sfb))
                    .copied()
                    .unwrap_or(SNF_NO_FILL_IDX);
                if idx == SNF_NO_FILL_IDX {
                    continue;
                }
                previous += (idx - SNF_DELTA_OFFSET) as f32;
                let amp = 2.0f32.powf(0.5 * previous);
                for v in &mut coded[r] {
                    *v = rng.random_noise_value() * amp;
                }
            }
        }
    });
}

/// One parsed `sf_data(ASF)` body before reconstruction.
#[derive(Debug, Clone)]
pub struct AsfCoded {
    /// Quantised lines per group (bitstream order within the group).
    pub quant: Vec<Vec<i32>>,
    /// `max_quant_idx[g][sfb]`.
    pub max_quant_idx: Vec<Vec<u32>>,
    /// `sfb_cb[g][sfb]`.
    pub sfb_cb: Vec<Vec<u8>>,
    /// `sf_gain[g][sfb]` (0 for bands without a scale factor).
    pub sf_gain: Vec<Vec<f32>>,
    /// `dpcm_snf[g][sfb]` codeword indices, when `b_snf_data_exists`.
    pub dpcm_snf: Option<Vec<Vec<i32>>>,
}

/// Parse one `sf_data(ASF)` body (Tables 39 – 42) on `layout`.
pub fn parse_sf_data(br: &mut BitReader<'_>, layout: &AsfLayout) -> Option<AsfCoded> {
    let mut sections = Vec::with_capacity(layout.groups.len());
    for g in &layout.groups {
        sections.push(asf_data::parse_asf_section_data(br, g.tl_code, g.tl, g.max_sfb).ok()?);
    }
    let mut quant = Vec::with_capacity(layout.groups.len());
    let mut mqi = Vec::with_capacity(layout.groups.len());
    for (g, s) in layout.groups.iter().zip(&sections) {
        let (q, m) = asf_data::parse_asf_spectral_data(br, s, &g.sect_offs, g.max_sfb).ok()?;
        quant.push(q);
        mqi.push(m);
    }
    let grid = layout.grid();
    let tls: Vec<u32> = layout.groups.iter().map(|g| g.tl).collect();
    let sf_gain =
        asf_data::parse_asf_scalefac_data_grouped(br, &sections, &mqi, &grid, &tls).ok()?;
    let dpcm_snf = asf_data::parse_asf_snf_data_grouped(br, &sections, &mqi, &grid, &tls).ok()?;
    Some(AsfCoded {
        quant,
        max_quant_idx: mqi,
        sfb_cb: sections.into_iter().map(|s| s.sfb_cb).collect(),
        sf_gain,
        dpcm_snf,
    })
}

/// Reconstruct a parsed body: §5.1.3 dequantisation + scaling, §5.1.4
/// noise fill, §5.1.5 ungrouping. Returns the window-order spectrum
/// ([`AsfLayout::out_len`] lines).
pub fn reconstruct(layout: &AsfLayout, coded: &AsfCoded) -> Vec<f32> {
    let mut flat = Vec::with_capacity(layout.coded_len());
    for (g, grp) in layout.groups.iter().enumerate() {
        flat.extend(asf_data::dequantise_and_scale(
            &coded.quant[g],
            &coded.sf_gain[g],
            &grp.sect_offs,
            grp.max_sfb,
        ));
    }
    if let Some(snf) = coded.dpcm_snf.as_ref() {
        apply_snf(&mut flat, layout, snf);
    }
    layout.ungroup(&flat)
}

/// Decode one `sf_data(ASF)` body to its window-order spectrum.
pub fn decode_sf_data(br: &mut BitReader<'_>, layout: &AsfLayout) -> Option<Vec<f32>> {
    let coded = parse_sf_data(br, layout)?;
    Some(reconstruct(layout, &coded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ti(long: bool, t: [u32; 2], l: [u32; 2]) -> AsfTransformInfo {
        AsfTransformInfo {
            b_long_frame: long,
            transf_length: t,
            transform_length_0: l[0],
            transform_length_1: l[1],
        }
    }

    #[test]
    fn long_frame_is_one_window_with_the_five_bit_section_code() {
        let l = AsfLayout::long(1920, 40).unwrap();
        assert!(l.is_single_block());
        assert_eq!(l.groups[0].tl_code, LONG_FRAME_TL_CODE);
        assert_eq!(sect_len_coding(l.groups[0].tl_code), (5, 31));
        assert_eq!(l.out_len(), l.groups[0].sfbo[40] as usize);
    }

    #[test]
    fn different_framing_forces_a_group_boundary() {
        // 1920-sample frame: 2 × 480 then 1 × 960 (the §5.5.3 example,
        // transf_length = [2, 3]); Table 109 gives one grouping bit.
        let t = ti(false, [2, 3], [480, 960]);
        let psy = AsfPsyInfo {
            b_different_framing: true,
            max_sfb_0: 10,
            max_sfb_1: 12,
            scale_factor_grouping: vec![1],
            frame_len_base: 1920,
            ..AsfPsyInfo::default()
        };
        let l = AsfLayout::new(&t, &psy, 10, 12).unwrap();
        assert_eq!(l.windows, vec![480, 480, 960]);
        assert_eq!(l.groups.len(), 2);
        assert_eq!((l.groups[0].num_win, l.groups[1].num_win), (2, 1));
        assert_eq!((l.groups[0].tl_code, l.groups[1].tl_code), (2, 3));
        assert_eq!((l.groups[0].max_sfb, l.groups[1].max_sfb), (10, 12));
        assert_eq!(l.out_len(), 1920);
    }

    #[test]
    fn ungroup_interleaves_the_windows_of_a_group_and_group_inverts_it() {
        // 1024-sample frame of eight 128-line windows, groups of 3 / 5.
        let t = ti(false, [0, 0], [128, 128]);
        let psy = AsfPsyInfo {
            max_sfb_0: 4,
            scale_factor_grouping: vec![1, 1, 0, 1, 1, 1, 1],
            frame_len_base: 1024,
            ..AsfPsyInfo::default()
        };
        let l = AsfLayout::new(&t, &psy, 4, 4).unwrap();
        assert_eq!(l.windows.len(), 8);
        assert_eq!(
            l.groups.iter().map(|g| g.num_win).collect::<Vec<_>>(),
            vec![3, 5]
        );
        let per_win = l.groups[0].sfbo[4] as usize;
        assert_eq!(l.coded_len(), per_win * 8);
        let coded: Vec<f32> = (0..l.coded_len()).map(|i| i as f32 + 1.0).collect();
        let reord = l.ungroup(&coded);
        assert_eq!(reord.len(), 1024);
        // Band 0 of group 0: windows 0, 1, 2 each receive
        // sfb_offset[1] consecutive coded lines.
        let w0 = l.groups[0].sfbo[1] as usize;
        assert_eq!(reord[0], 1.0);
        assert_eq!(reord[128], w0 as f32 + 1.0);
        assert_eq!(reord[256], 2.0 * w0 as f32 + 1.0);
        assert_eq!(l.group(&reord), coded);
    }

    #[test]
    fn noise_fill_follows_the_coded_level_deltas() {
        // Wide bands near the top of a 1920-line frame so the realised
        // level of a band averages many generator draws.
        let l = AsfLayout::long(1920, 50).unwrap();
        let mut flat = vec![0.0f32; l.coded_len()];
        // Band 44 carries a known level; band 45 is empty with delta
        // −2; band 46 empty with the escape; band 47 empty with +1.
        for v in &mut flat[l.coded_band(0, 44)] {
            *v = 8.0;
        }
        let mut row = vec![0i32; 50];
        row[45] = 15;
        row[47] = 18;
        reset_snf_rng(3);
        apply_snf(&mut flat, &l, &[row]);
        let level = |r: std::ops::Range<usize>| band_level(&flat[r]);
        assert_eq!(level(l.coded_band(0, 44)), Some(6.0));
        assert!(level(l.coded_band(0, 46)).is_none());
        assert!(level(l.coded_band(0, 3)).is_none());
        // The generator has unit variance: the realised band level sits
        // within a step of the coded one (the escape leaves the running
        // level at 4, so band 47 lands on 5).
        let b45 = level(l.coded_band(0, 45)).unwrap();
        let b47 = level(l.coded_band(0, 47)).unwrap();
        assert!((b45 - 4.0).abs() < 1.0, "band 45 level {b45}");
        assert!((b47 - 5.0).abs() < 1.0, "band 47 level {b47}");
    }

    #[test]
    fn snf_generator_reset_matches_pseudocode_24() {
        let s = snf_state_for(0);
        assert_eq!(
            (s.offset_a, s.offset_b, s.state_idx, s.current_idx),
            (0, 0, 1, 0)
        );
        // sequence_counter 1: start = 255, A = 0, B = 1,
        // stateIdx = 1·(2 + 2)/2 + 255·1·256/2 = 32 642 → 130,
        // currentIdx = 32 386 → 130.
        let s = snf_state_for(1);
        assert_eq!((s.offset_a, s.offset_b), (0, 1));
        assert_eq!(s.state_idx, (32_642u32 % 256) as u8);
        assert_eq!(s.current_idx, (32_386u32 % 256) as u8);
        // start = 255·(sequence_counter mod 256) wraps with period 256.
        let s = snf_state_for(256 + 3);
        let r = snf_state_for(3);
        assert_eq!((s.state_idx, s.current_idx), (r.state_idx, r.current_idx));
    }

    #[test]
    fn block_plan_sf_info_round_trips_through_the_parser() {
        use oxideav_core::bits::BitWriter;
        // 1920-sample frame: four 240-line blocks (two groups) then one
        // 960-line block.
        let plan = BlockPlan {
            frame_len: 1920,
            long: false,
            transf_length: [1, 3],
            grouping: vec![1, 0, 1],
            max_sfb: [12, 30],
        };
        let mut bw = BitWriter::new();
        plan.write_sf_info(&mut bw);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = crate::asf::parse_asf_transform_info(&mut br, 1920).unwrap();
        let psy = crate::asf::parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        assert_eq!(ti.transf_length, [1, 3]);
        assert_eq!((psy.max_sfb_0, psy.max_sfb_1), (12, 30));
        assert_eq!(psy.num_windows, 5);
        assert_eq!(psy.num_window_groups, 3);
        let a = AsfLayout::new(&ti, &psy, psy.max_sfb_0, psy.max_sfb_1).unwrap();
        let b = plan.layout().unwrap();
        assert_eq!(a.windows, vec![240, 240, 240, 240, 960]);
        assert_eq!(a.windows, b.windows);
        assert_eq!(
            a.groups
                .iter()
                .map(|g| (g.first_win, g.num_win, g.max_sfb))
                .collect::<Vec<_>>(),
            b.groups
                .iter()
                .map(|g| (g.first_win, g.num_win, g.max_sfb))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            a.groups.iter().map(|g| g.num_win).collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        // Below 1536 samples: 960-sample frame as four 240-line blocks.
        let plan = BlockPlan {
            frame_len: 960,
            long: false,
            transf_length: [1, 1],
            grouping: vec![0, 1, 1],
            max_sfb: [10, 10],
        };
        let mut bw = BitWriter::new();
        plan.write_sf_info(&mut bw);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = crate::asf::parse_asf_transform_info(&mut br, 960).unwrap();
        let psy = crate::asf::parse_asf_psy_info(&mut br, &ti, 960, false, false).unwrap();
        let a = AsfLayout::new(&ti, &psy, psy.max_sfb_0, psy.max_sfb_1).unwrap();
        assert_eq!(a.windows, vec![240; 4]);
        assert_eq!(
            a.groups.iter().map(|g| g.num_win).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(a.windows, plan.layout().unwrap().windows);
    }
}
