//! Immersive channel element — ETSI TS 103 190-2 §6.2.4.1
//! `immersive_channel_element()` and §6.2.4.2 `immers_cfg()`.
//!
//! The immersive channel element carries the channel-based immersive
//! layouts (7.0.4 / 7.1.4 / 9.0.4 / 9.1.4 — Table 78 channel modes
//! 11..=14, routed from `audio_data_chan()` per §6.2.3.1). Its body is
//! built from the part-1 channel-data elements:
//!
//! * a **5-channel core** in one of four groupings (Table 97:
//!   1+2+2 / 3+2 / 1+4 / 5) producing SMP tracks A..E per Table 19;
//! * for the `7CH_STATIC` core configuration (every mode except
//!   ASPX_AJCC — Table 96) an **additional `two_channel_data()`**
//!   (tracks F, G) with an optional `chparam_info()` pair
//!   (`b_use_sap_add_ch`, §5.2.3.2 step 3);
//! * a mode-dependent **A-SPX layer** (`aspx_data_2ch()` /
//!   `aspx_data_1ch()`, consumed per Table 8);
//! * **`ajcc_data(b_5fronts)`** for ASPX_AJCC (§6.2.6);
//! * for the S-CPL family (SCPL / ASPX_SCPL / ASPX_ACPL_1) two or
//!   three further `two_channel_data()` elements (tracks H..M) plus
//!   four or six `chparam_info()` elements (§5.2.3.2 step 5);
//! * for ASPX_ACPL_1 / ASPX_ACPL_2 four or six `acpl_data_1ch()`
//!   parameter sets (§5.5.2 Table 27).
//!
//! Interpretation notes (documented deviations / readings where the TS
//! leaves the layout implicit):
//!
//! * The bare `chparam_info()` elements of this syntax carry no
//!   `sf_info()` of their own; their per-group `ms_used` / `sap_data`
//!   walks are parsed with the (single-group) `max_sfb` of the
//!   channel-data element whose tracks they process — the D/E-carrying
//!   core element for the `b_use_sap_add_ch` pair, and the
//!   corresponding S-CPL `two_channel_data()` for the step-5 elements.
//! * The A-SPX payload↔track association follows the V1.3.1 Table 8
//!   roster (its NOTE 3 makes the listed order the bitstream order;
//!   the staged errata notes record the superseded V1.2.1 rows):
//!   `(A'', B'')` / `(D'', E'')` / `(F'', G'')` / `C''` for the whole
//!   ACPL / AJCC family (the third pair only with the 7CH_STATIC core
//!   config), and `(Ls, Lb)`, `(Rs, Rb)`, `C`, `(L[, R| Lscr])`,
//!   `(R, Rscr)?`, `(Tfl, Tbl)`, `(Tfr, Tbr)` for ASPX_SCPL. The
//!   ACPL_1 S-CPL-section residual tracks `H''..K''` (and the
//!   b_5fronts front residuals `L''` / `M''`) are not A-SPX targets.
//! * The 7CH_STATIC track roles follow the V1.3.1 Table 23 fold
//!   pairs `(D, H)` / `(E, I)` / `(F, J)` / `(G, K)`: tracks `A..G`
//!   are the 5.X.2 core (fronts, surround mids `D` / `E`, top-front
//!   mids on the additional pair `F` / `G`) and the S-CPL-section
//!   tracks `H..K` carry the pair sides. In the ACPL modes the same
//!   carriers `D..G` sit at the §5.5.2 Table 25 module positions
//!   x5 / x6 / x9 / x10, with the ACPL_1 residuals `H..K` at
//!   x7 / x8 / x11 / x12.
//! * §5.2.3.2 step 5 extracts a single prediction gain `a'_j` per
//!   element "as specified in [part-1] clause 5.3.2", whose
//!   Pseudocode 59 produces the quartet `(a, b, c, d)`; the additive
//!   Table 20 row `(a'_j, 1)` carries the prediction gain only, read
//!   here as `a' = a − 1` (`= sap_gain`) on full-SAP coded bands and 0
//!   elsewhere (see [`crate::asf::extract_sap_a_prime`]).

use crate::acpl::{
    parse_acpl_config_1ch, parse_acpl_data_1ch, sb_to_pb, Acpl1chMode, AcplConfig1ch, AcplData1ch,
};
use crate::ajcc::{parse_ajcc_data, write_ajcc_data, AjccData};
use crate::asf::{
    capture_aspx_data_1ch_trailer, capture_aspx_data_2ch_trailer, parse_chparam_info, ChparamInfo,
    SubstreamTools,
};
use crate::aspx::{
    parse_aspx_config, parse_companding_control, AspxConfig, CompandingControl, FiveXAspxTrailer,
};
use crate::mch::{
    parse_five_channel_data, parse_four_channel_data, parse_mono_data, parse_three_channel_data,
    parse_two_channel_data, FiveChannelData, FourChannelData, MonoLfeData, ThreeChannelData,
    TwoChannelData,
};
use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

// =====================================================================
// §6.3.5.1 immersive_codec_mode (Table 95) + Table 96 / 97 helpers
// =====================================================================

/// `immersive_codec_mode` per Table 95.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceCodecMode {
    /// SMP + S-CPL (code `0b000`).
    Scpl,
    /// SMP + A-SPX + S-CPL (code `0b001`).
    AspxScpl,
    /// SMP + A-SPX + A-CPL mode 1 (code `0b010`).
    AspxAcpl1,
    /// SMP + A-SPX + A-CPL mode 2 (code `0b011`).
    AspxAcpl2,
    /// SMP + A-SPX + A-JCC (code `0b1`).
    AspxAjcc,
}

impl IceCodecMode {
    /// Parse the `immersive_codec_mode_code` prefix code (§6.3.5.1):
    /// one bit, and two more when the first is 0.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self> {
        if br.read_bit()? {
            return Ok(IceCodecMode::AspxAjcc);
        }
        Ok(match br.read_u32(2)? {
            0b00 => IceCodecMode::Scpl,
            0b01 => IceCodecMode::AspxScpl,
            0b10 => IceCodecMode::AspxAcpl1,
            _ => IceCodecMode::AspxAcpl2,
        })
    }

    /// Exact writer inverse of [`IceCodecMode::parse`].
    pub fn write(self, bw: &mut BitWriter) {
        match self {
            IceCodecMode::AspxAjcc => bw.write_bit(true),
            IceCodecMode::Scpl => {
                bw.write_bit(false);
                bw.write_u32(0b00, 2);
            }
            IceCodecMode::AspxScpl => {
                bw.write_bit(false);
                bw.write_u32(0b01, 2);
            }
            IceCodecMode::AspxAcpl1 => {
                bw.write_bit(false);
                bw.write_u32(0b10, 2);
            }
            IceCodecMode::AspxAcpl2 => {
                bw.write_bit(false);
                bw.write_u32(0b11, 2);
            }
        }
    }

    /// Table 96: `core_channel_config` — every mode except ASPX_AJCC
    /// uses the 7CH_STATIC (5.X.2) core; ASPX_AJCC uses 5CH_DYNAMIC.
    pub fn is_7ch_static(self) -> bool {
        !matches!(self, IceCodecMode::AspxAjcc)
    }

    /// Whether the element carries an A-SPX layer (every mode except
    /// plain SCPL).
    pub fn uses_aspx(self) -> bool {
        !matches!(self, IceCodecMode::Scpl)
    }

    /// Whether the element carries the S-CPL two_channel_data +
    /// chparam_info section (§6.2.4.1: SCPL / ASPX_SCPL / ASPX_ACPL_1).
    pub fn has_scpl_section(self) -> bool {
        matches!(
            self,
            IceCodecMode::Scpl | IceCodecMode::AspxScpl | IceCodecMode::AspxAcpl1
        )
    }

    /// Whether the element carries the trailing `acpl_data_1ch()`
    /// section (ASPX_ACPL_1 / ASPX_ACPL_2).
    pub fn has_acpl_section(self) -> bool {
        matches!(self, IceCodecMode::AspxAcpl1 | IceCodecMode::AspxAcpl2)
    }
}

/// One A-SPX payload of the element, in transmission order.
#[derive(Debug, Clone)]
pub enum IceAspxElement {
    /// `aspx_data_2ch()` — `None` when the trailer capture bailed.
    TwoCh(Option<Box<FiveXAspxTrailer>>),
    /// `aspx_data_1ch()` — `None` when the trailer capture bailed.
    OneCh(Option<Box<FiveXAspxTrailer>>),
}

/// The 5-channel core in one of the Table 97 groupings. Table 19
/// assigns the decoded input tracks to the SMP output tracks A..E.
#[derive(Debug, Clone)]
pub enum IceCoreData {
    /// Grouping 0 (1+2+2): `2ch_mode` + two `two_channel_data()` +
    /// `mono_data(0)`.
    Grouping0 {
        /// `2ch_mode` (§5.2.3.2 Table 19 NOTE 2).
        b_2ch_mode: bool,
        /// First pair — tracks `[A, B]` (`2ch_mode == 0`) or `[A, D]`.
        pair_a: TwoChannelData,
        /// Second pair — tracks `[D, E]` or `[B, E]`.
        pair_b: TwoChannelData,
        /// Centre `mono_data(0)` — track C.
        mono: MonoLfeData,
    },
    /// Grouping 1 (3+2): `three_channel_data()` (tracks A, B, C) +
    /// `two_channel_data()` (tracks D, E).
    Grouping1 {
        /// The leading three-channel element.
        three: ThreeChannelData,
        /// The trailing pair.
        pair: TwoChannelData,
    },
    /// Grouping 2 (1+4): `four_channel_data()` (tracks A, B, D, E) +
    /// `mono_data(0)` (track C).
    Grouping2 {
        /// The four-channel element.
        four: FourChannelData,
        /// Centre mono.
        mono: MonoLfeData,
    },
    /// Grouping 3 (5): `five_channel_data()` (tracks A, B, C, D, E).
    Grouping3 {
        /// The five-channel element.
        five: FiveChannelData,
    },
}

/// Parsed `immersive_channel_element(b_lfe, b_5fronts, b_iframe)`.
#[derive(Debug, Clone)]
pub struct IceElement {
    /// `b_lfe` the element was invoked with (channel modes 12 / 14).
    pub b_lfe: bool,
    /// `b_5fronts` (channel modes 13 / 14).
    pub b_5fronts: bool,
    /// Decoded `immersive_codec_mode`.
    pub mode: IceCodecMode,
    /// LFE `mono_data(1)` when `b_lfe`.
    pub lfe: Option<MonoLfeData>,
    /// `companding_control(5)` (ASPX_AJCC only, §4.8.3.10.3).
    pub companding: Option<CompandingControl>,
    /// 2-bit `core_5ch_grouping` (Table 97).
    pub core_5ch_grouping: u8,
    /// The 5-channel core data.
    pub core: IceCoreData,
    /// `b_use_sap_add_ch` (7CH_STATIC only).
    pub b_use_sap_add_ch: Option<bool>,
    /// The two `chparam_info()` elements driving §5.2.3.2 step 4 when
    /// `b_use_sap_add_ch == 1`.
    pub sap_add_chparam: Option<Box<[ChparamInfo; 2]>>,
    /// The additional `two_channel_data()` (tracks F, G — 7CH_STATIC
    /// only).
    pub add_pair: Option<TwoChannelData>,
    /// A-SPX payloads in transmission order (empty for SCPL).
    pub aspx_elements: Vec<IceAspxElement>,
    /// `ajcc_data(b_5fronts)` (ASPX_AJCC only). Stored parsed-but-not-
    /// differentially-decoded so the caller can run
    /// [`crate::ajcc::decode_ajcc_parsed`] with its cross-frame
    /// [`crate::ajcc::AjccState`].
    pub ajcc: Option<Box<AjccData>>,
    /// S-CPL section pairs — tracks `[H, I]`, `[J, K]` and (b_5fronts)
    /// `[L, M]`.
    pub scpl_pairs: Vec<TwoChannelData>,
    /// S-CPL section `chparam_info()` elements (4, or 6 with
    /// b_5fronts) — §5.2.3.2 step 5.
    pub scpl_chparam: Vec<ChparamInfo>,
    /// `acpl_data_1ch()` parameter sets (4, or 6 with b_5fronts) for
    /// the §5.5.2 Table 27 A-CPL modules.
    pub acpl_data: Vec<AcplData1ch>,
}

/// Number of SMP tracks the element produces (Table 22): 11, or 13
/// with `b_5fronts`.
pub fn num_ice_tracks(b_5fronts: bool) -> usize {
    if b_5fronts {
        13
    } else {
        11
    }
}

impl IceElement {
    /// The decoded fullband MDCT spectra assigned to the SMP internal
    /// tracks A..M per Table 19 (index 0 = A … 12 = M; always 13
    /// entries — tracks the mode leaves uncoded are `None` and are
    /// silence per §5.2.3.3/§5.2.3.4).
    pub fn track_spectra(&self) -> Vec<Option<&[f32]>> {
        fn pair_ch(p: &TwoChannelData, ch: usize) -> Option<&[f32]> {
            p.scaled_spec_per_channel.get(ch).and_then(|s| s.as_deref())
        }
        let mut t: Vec<Option<&[f32]>> = vec![None; 13];
        match &self.core {
            IceCoreData::Grouping0 {
                b_2ch_mode,
                pair_a,
                pair_b,
                mono,
            } => {
                if *b_2ch_mode {
                    // [A, D] / [B, E].
                    t[0] = pair_ch(pair_a, 0);
                    t[3] = pair_ch(pair_a, 1);
                    t[1] = pair_ch(pair_b, 0);
                    t[4] = pair_ch(pair_b, 1);
                } else {
                    // [A, B] / [D, E].
                    t[0] = pair_ch(pair_a, 0);
                    t[1] = pair_ch(pair_a, 1);
                    t[3] = pair_ch(pair_b, 0);
                    t[4] = pair_ch(pair_b, 1);
                }
                t[2] = mono.scaled_spec.as_deref();
            }
            IceCoreData::Grouping1 { three, pair } => {
                for (i, slot) in [0usize, 1, 2].into_iter().enumerate() {
                    t[slot] = three
                        .scaled_spec_per_channel
                        .get(i)
                        .and_then(|s| s.as_deref());
                }
                t[3] = pair_ch(pair, 0);
                t[4] = pair_ch(pair, 1);
            }
            IceCoreData::Grouping2 { four, mono } => {
                // four_channel_data → [A, B, D, E].
                for (i, slot) in [0usize, 1, 3, 4].into_iter().enumerate() {
                    t[slot] = four
                        .scaled_spec_per_channel
                        .get(i)
                        .and_then(|s| s.as_deref());
                }
                t[2] = mono.scaled_spec.as_deref();
            }
            IceCoreData::Grouping3 { five } => {
                for (i, slot) in [0usize, 1, 2, 3, 4].into_iter().enumerate() {
                    t[slot] = five
                        .scaled_spec_per_channel
                        .get(i)
                        .and_then(|s| s.as_deref());
                }
            }
        }
        // Additional pair → F, G.
        if let Some(p) = &self.add_pair {
            t[5] = pair_ch(p, 0);
            t[6] = pair_ch(p, 1);
        }
        // S-CPL pairs → H, I / J, K / L, M.
        for (i, p) in self.scpl_pairs.iter().enumerate() {
            if 7 + 2 * i + 1 < 13 {
                t[7 + 2 * i] = pair_ch(p, 0);
                t[7 + 2 * i + 1] = pair_ch(p, 1);
            }
        }
        t
    }

    /// The core transform length (from the first core channel-data
    /// element's `asf_transform_info`). `None` when the shell parse
    /// did not surface one.
    pub fn transform_length(&self) -> Option<u32> {
        let ti = match &self.core {
            IceCoreData::Grouping0 { pair_a, .. } => pair_a.transform_info.as_ref(),
            IceCoreData::Grouping1 { three, .. } => three.transform_info.as_ref(),
            IceCoreData::Grouping2 { four, .. } => four.transform_info.as_ref(),
            IceCoreData::Grouping3 { five } => five.transform_info.as_ref(),
        };
        ti.map(|t| t.transform_length_0)
    }

    /// The decoded LFE spectrum when present and decodable.
    pub fn lfe_spectrum(&self) -> Option<&[f32]> {
        self.lfe.as_ref().and_then(|m| m.scaled_spec.as_deref())
    }

    /// Owned copy of [`IceElement::track_spectra`] — always 13 dense
    /// rows (uncoded tracks come back as empty vectors, which the SAP
    /// / IMDCT stages treat as silence).
    pub fn track_spectra_owned(&self) -> Vec<Vec<f32>> {
        self.track_spectra()
            .into_iter()
            .map(|s| s.map(<[f32]>::to_vec).unwrap_or_default())
            .collect()
    }
}

/// Apply the TS 103 190-2 §5.2.3.2 SAP mixing (steps 3-6) in place on
/// the 13 dense track spectra (index 0 = A … 12 = M; empty rows are
/// silence).
///
/// * **Steps 3-4** — when `b_use_sap_add_ch == 1`, the two contained
///   `chparam_info()` elements drive the 2x2 quartets `(a_i, b_i, c_i,
///   d_i)` (part-1 Pseudocode 59) mixing `(D, F)` and `(E, G)`:
///   `D' = a_0 D + b_0 F`, `F' = c_0 D + d_0 F` (and likewise `E`/`G`
///   with the second element). Per Table 19 NOTE 2 only D, E, F, G are
///   modified; bins past the SAP-coded extent pass through.
/// * **Steps 5-6** — the S-CPL-section `chparam_info()` elements
///   resolve the additive prediction gains `a'_j`
///   ([`crate::asf::extract_sap_a_prime`] — non-zero only for
///   `sap_mode == 3`) and Table 20 adds the scaled carriers into the
///   S-CPL mid/residual tracks: `H'' = H' + a'_0 D'`,
///   `I'' = I' + a'_1 E'`, `J'' = J' + a'_2 F'`, `K'' = K' + a'_3 G'`,
///   and with `b_5fronts` also `L'' = L' + a'_4 A'`,
///   `M'' = M' + a'_5 B'`.
///
/// The steps only apply to the §5.2.3.2 modes (SCPL / ASPX_SCPL /
/// ASPX_ACPL_1 — [`IceCodecMode::has_scpl_section`]); the ASPX_ACPL_2
/// and ASPX_AJCC processing clauses (§5.2.3.3-4) carry no SAP stage,
/// so the call is a no-op for them. Both stages run on the parse-side
/// single-window-group grid: each `chparam_info()` is keyed against
/// the same `max_sfb` bound it was parsed with, over the
/// `transform_length` sfb offsets.
pub fn apply_sap_steps(specs: &mut [Vec<f32>], ice: &IceElement, transform_length: u32) {
    if !ice.mode.has_scpl_section() || specs.len() < 13 {
        return;
    }
    let Some(sfbo) = crate::sfb_offset::sfb_offset_48(transform_length) else {
        return;
    };
    let n = transform_length as usize;
    // Materialise a lazily-zero row to length `len` so in-place mixing
    // has both operands.
    fn ensure_len(v: &mut Vec<f32>, len: usize) {
        if v.len() < len {
            v.resize(len, 0.0);
        }
    }
    // Steps 3-4: b_use_sap_add_ch quartets on (D, F) and (E, G).
    if ice.b_use_sap_add_ch == Some(true) {
        if let (Some(cp), Ok(ctx)) = (ice.sap_add_chparam.as_deref(), core_de_max_sfb(&ice.core)) {
            for (i, info) in cp.iter().enumerate() {
                let coeffs = crate::asf::extract_sap_abcd(info, &[ctx]);
                let abcd: &[(f32, f32, f32, f32)] =
                    coeffs.abcd.first().map(|v| v.as_slice()).unwrap_or(&[]);
                let usable = abcd.len().min(ctx as usize);
                let hi_bin = sfbo
                    .get(usable)
                    .copied()
                    .map(|v| (v as usize).min(n))
                    .unwrap_or(0);
                let (lo_idx, hi_idx) = (3 + i, 5 + i); // (D, F) / (E, G)
                let (head, tail) = specs.split_at_mut(hi_idx);
                let x_row = &mut head[lo_idx];
                let y_row = &mut tail[0];
                ensure_len(x_row, hi_bin);
                ensure_len(y_row, hi_bin);
                for (sfb, &(a, b, c, d)) in abcd.iter().enumerate().take(usable) {
                    let lo = sfbo[sfb] as usize;
                    let hi = (sfbo[sfb + 1] as usize).min(n);
                    for (x, y) in x_row[lo..hi].iter_mut().zip(y_row[lo..hi].iter_mut()) {
                        let (xv, yv) = (*x, *y);
                        *x = a * xv + b * yv;
                        *y = c * xv + d * yv;
                    }
                }
            }
        }
    }
    // Steps 5-6: additive a'_j prediction into the S-CPL tracks.
    // (target, source) pairs in Table 20 row order.
    const TARGETS: [(usize, usize); 6] = [(7, 3), (8, 4), (9, 5), (10, 6), (11, 0), (12, 1)];
    for (j, info) in ice.scpl_chparam.iter().enumerate() {
        if j >= TARGETS.len() {
            break;
        }
        // Mirror the parse-side context: pair j/2 for the first four
        // elements, the third pair for the b_5fronts extras.
        let pair_idx = if j < 4 { j / 2 } else { 2 };
        let Some(ctx) = ice
            .scpl_pairs
            .get(pair_idx)
            .and_then(|p| p.psy_info.as_ref())
            .map(|p| p.max_sfb_0)
        else {
            continue;
        };
        let a_prime = crate::asf::extract_sap_a_prime(info, &[ctx]);
        let row: &[f32] = a_prime.first().map(|v| v.as_slice()).unwrap_or(&[]);
        if row.iter().all(|&g| g == 0.0) {
            continue;
        }
        let (tgt, src) = TARGETS[j];
        let usable = row.len().min(ctx as usize);
        let hi_bin = sfbo
            .get(usable)
            .copied()
            .map(|v| (v as usize).min(n))
            .unwrap_or(0);
        ensure_len(&mut specs[tgt], hi_bin);
        for (sfb, &g) in row.iter().enumerate().take(usable) {
            if g == 0.0 {
                continue;
            }
            let lo = sfbo[sfb] as usize;
            let hi = (sfbo[sfb + 1] as usize).min(n);
            for k in lo..hi {
                let s = specs[src].get(k).copied().unwrap_or(0.0);
                specs[tgt][k] += g * s;
            }
        }
    }
}

/// Build a `sap_mode = 3` [`ChparamInfo`] from a per-sfb `alpha_q`
/// row (0 on disengaged pairs; both halves of an sfb pair share the
/// even half's value). Returns `None` when no pair engages.
fn chparam_from_alpha_row(alpha_q: &[i32], max_sfb: u32) -> Option<ChparamInfo> {
    let m = max_sfb as usize;
    let mut used = vec![false; m];
    let mut deltas = vec![0i32; m];
    let mut any = false;
    let mut sfb = 0usize;
    while sfb < m {
        let a = alpha_q.get(sfb).copied().unwrap_or(0);
        if a != 0 {
            any = true;
            used[sfb] = true;
            if sfb + 1 < m {
                used[sfb + 1] = true;
            }
            // Parse-side reference: the previous even sfb's recovered
            // alpha (0 when that pair was disengaged or sfb == 0).
            let prev = if sfb >= 2 {
                alpha_q.get(sfb - 2).copied().unwrap_or(0)
            } else {
                0
            };
            deltas[sfb] = a - prev;
        }
        sfb += 2;
    }
    if !any {
        return None;
    }
    Some(ChparamInfo {
        sap_mode: 3,
        ms_used: Vec::new(),
        sap_data: Some(crate::asf::SapData {
            sap_coeff_all: used.iter().all(|&u| u),
            sap_coeff_used: vec![used],
            delta_code_time: false,
            dpcm_alpha_q: vec![deltas],
        }),
    })
}

/// Per-sfb-pair least-squares gain of `target` on `source`, quantised
/// to the Pseudocode 59 `alpha_q · 0,1` grid (|alpha_q| ≤ `max_q`).
fn sap_pair_alpha_row(
    target: &[f32],
    source: &[f32],
    sfbo: &[u16],
    max_sfb: u32,
    n: usize,
    max_q: i32,
) -> Vec<i32> {
    let m = max_sfb as usize;
    let mut alpha_q = vec![0i32; m];
    let mut sfb = 0usize;
    while sfb < m {
        let lo = sfbo.get(sfb).map(|&v| v as usize).unwrap_or(n).min(n);
        let hi_sfb = (sfb + 2).min(m);
        let hi = sfbo.get(hi_sfb).map(|&v| v as usize).unwrap_or(n).min(n);
        let (mut num, mut den, mut e_t) = (0.0f64, 0.0f64, 0.0f64);
        for k in lo..hi {
            let t = target.get(k).copied().unwrap_or(0.0) as f64;
            let s = source.get(k).copied().unwrap_or(0.0) as f64;
            num += t * s;
            den += s * s;
            e_t += t * t;
        }
        if den > 0.0 && num.is_finite() {
            let q = ((num / den) * 10.0).round() as i32;
            let q = q.clamp(-max_q, max_q);
            // Engage only where the quantised gain reduces the
            // target's pair energy by a meaningful fraction:
            // E(t) − E(t − g·s) = g·(2·num − g·den). A bare
            // strictly-positive test lets leakage-level correlations
            // against near-silent source bands engage with clamped
            // gains (the classic least-squares small-denominator
            // blow-up), so require ≥ 10 % of the pair's target
            // energy — an encoder tuning choice.
            let g = q as f64 * 0.1;
            if q != 0 && g * (2.0 * num - g * den) > 0.1 * e_t {
                alpha_q[sfb] = q;
                if sfb + 1 < m {
                    alpha_q[sfb + 1] = q;
                }
            }
        }
        sfb += 2;
    }
    alpha_q
}

/// Extract a §5.2.3.2 **step-5/6** SAP prediction element for one
/// S-CPL track: per sfb-pair least-squares gain of `target` on
/// `source`, quantised to the Pseudocode 59 `alpha_q · 0,1` grid.
/// Returns the full-SAP `chparam_info()` plus the wire residual
/// spectrum (`target − sap_gain·source` on engaged pairs) — the exact
/// encoder inverse of the Table 20 additive rows
/// (`H″ = H′ + a′·D′` etc.), or `None` when no pair engages (emit the
/// identity element and the unmodified spectrum instead).
///
/// The per-pair engage rule (quantised gain strictly reduces the
/// coded pair energy) and the `|alpha_q| ≤ 15` window are encoder
/// tuning choices — the TS specifies only the decoder mapping.
pub fn extract_sap_step56_prediction(
    target: &[f32],
    source: &[f32],
    transform_length: u32,
    max_sfb: u32,
) -> Option<(ChparamInfo, Vec<f32>)> {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)?;
    let n = transform_length as usize;
    let alpha_q = sap_pair_alpha_row(target, source, sfbo, max_sfb, n, 15);
    let info = chparam_from_alpha_row(&alpha_q, max_sfb)?;
    let mut wire: Vec<f32> = target.to_vec();
    let m = max_sfb as usize;
    for (sfb, &q) in alpha_q.iter().enumerate().take(m) {
        if q == 0 {
            continue;
        }
        let lo = sfbo.get(sfb).map(|&v| v as usize).unwrap_or(n).min(n);
        let hi = sfbo.get(sfb + 1).map(|&v| v as usize).unwrap_or(n).min(n);
        let g = q as f32 * 0.1;
        if wire.len() < hi {
            wire.resize(hi, 0.0);
        }
        for (k, w) in wire.iter_mut().enumerate().take(hi).skip(lo) {
            let s = source.get(k).copied().unwrap_or(0.0);
            *w -= g * s;
        }
    }
    Some((info, wire))
}

/// Extract a §5.2.3.2 **step-3/4** `b_use_sap_add_ch` quartet coding
/// decision for one `(D, F)` / `(E, G)` pair: per sfb-pair the wire
/// pair becomes `(mid, side − g·mid)` with `mid = (X+Y)/2`,
/// `side = (X−Y)/2` and the least-squares prediction gain `g` on the
/// `alpha_q · 0,1` grid — the exact inverse of the Pseudocode 59
/// quartet `(1+g, 1, 1−g, −1)` the decoder applies. Disengaged pairs
/// stay identity (wire = the true tracks). Returns `None` when no
/// pair engages.
pub fn extract_sap_step34_pair(
    x: &[f32],
    y: &[f32],
    transform_length: u32,
    max_sfb: u32,
) -> Option<(ChparamInfo, Vec<f32>, Vec<f32>)> {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)?;
    let n = transform_length as usize;
    let m = max_sfb as usize;
    let len = x.len().max(y.len());
    let mid: Vec<f32> = (0..len)
        .map(|k| 0.5 * (x.get(k).copied().unwrap_or(0.0) + y.get(k).copied().unwrap_or(0.0)))
        .collect();
    let side: Vec<f32> = (0..len)
        .map(|k| 0.5 * (x.get(k).copied().unwrap_or(0.0) - y.get(k).copied().unwrap_or(0.0)))
        .collect();
    let alpha_q = sap_pair_alpha_row(&side, &mid, sfbo, max_sfb, n, 15);
    let info = chparam_from_alpha_row(&alpha_q, max_sfb)?;
    let mut wire_x: Vec<f32> = x.to_vec();
    let mut wire_y: Vec<f32> = y.to_vec();
    for (sfb, &q) in alpha_q.iter().enumerate().take(m) {
        if q == 0 {
            continue;
        }
        let lo = sfbo.get(sfb).map(|&v| v as usize).unwrap_or(n).min(n);
        let hi = sfbo.get(sfb + 1).map(|&v| v as usize).unwrap_or(n).min(n);
        let g = q as f32 * 0.1;
        if wire_x.len() < hi {
            wire_x.resize(hi, 0.0);
        }
        if wire_y.len() < hi {
            wire_y.resize(hi, 0.0);
        }
        for k in lo..hi {
            wire_x[k] = mid.get(k).copied().unwrap_or(0.0);
            wire_y[k] =
                side.get(k).copied().unwrap_or(0.0) - g * mid.get(k).copied().unwrap_or(0.0);
        }
    }
    Some((info, wire_x, wire_y))
}

/// The identity (`sap_mode = 0`) chparam element.
pub fn identity_chparam() -> ChparamInfo {
    ChparamInfo {
        sap_mode: 0,
        ms_used: Vec::new(),
        sap_data: None,
    }
}

/// The `max_sfb` context (single window group) of the channel-data
/// element that carries the D / E tracks — used to bound the
/// `b_use_sap_add_ch` `chparam_info()` walks (see the module notes).
fn core_de_max_sfb(core: &IceCoreData) -> Result<u32> {
    let psy = match core {
        IceCoreData::Grouping0 { pair_b, .. } => pair_b.psy_info.as_ref(),
        IceCoreData::Grouping1 { pair, .. } => pair.psy_info.as_ref(),
        IceCoreData::Grouping2 { four, .. } => four.psy_info.as_ref(),
        IceCoreData::Grouping3 { five } => five.psy_info.as_ref(),
    };
    psy.map(|p| p.max_sfb_0)
        .ok_or_else(|| Error::invalid("ac4: ice chparam context without core psy_info"))
}

/// Parse the immersive channel element into `tools` (configs land on
/// the shared I-frame-sticky slots so [`crate::asf::StickyConfig`]
/// carries them across frames; the element body lands on
/// `tools.ice`).
///
/// For `b_iframe == 0` the caller must have pre-seeded `tools` with
/// the sticky `aspx_config` / `acpl_config_1ch_*` /
/// `aspx_xover_subband_offset` from the last I-frame.
pub fn parse_ice_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_lfe: bool,
    b_5fronts: bool,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    let mode = IceCodecMode::parse(br)?;
    // §6.2.4.2 immers_cfg() — I-frames only; P-frames reuse sticky.
    if b_iframe {
        if mode.uses_aspx() {
            tools.aspx_config = Some(parse_aspx_config(br)?);
        }
        match mode {
            IceCodecMode::AspxAcpl1 => {
                tools.acpl_config_1ch_partial =
                    Some(parse_acpl_config_1ch(br, Acpl1chMode::Partial)?);
            }
            IceCodecMode::AspxAcpl2 => {
                tools.acpl_config_1ch_full = Some(parse_acpl_config_1ch(br, Acpl1chMode::Full)?);
            }
            _ => {}
        }
    }
    let lfe = if b_lfe {
        let m = parse_mono_data(br, true, frame_len_base)?;
        tools.lfe_mono_data = Some(m.clone());
        Some(m)
    } else {
        None
    };
    let companding = if matches!(mode, IceCodecMode::AspxAjcc) {
        let c = parse_companding_control(br, 5)?;
        tools.companding = Some(c.clone());
        Some(c)
    } else {
        None
    };
    let core_5ch_grouping = br.read_u32(2)? as u8;
    let core = match core_5ch_grouping {
        0 => {
            let b_2ch_mode = br.read_bit()?;
            let pair_a = parse_two_channel_data(br, frame_len_base)?;
            let pair_b = parse_two_channel_data(br, frame_len_base)?;
            let mono = parse_mono_data(br, false, frame_len_base)?;
            IceCoreData::Grouping0 {
                b_2ch_mode,
                pair_a,
                pair_b,
                mono,
            }
        }
        1 => {
            let three = parse_three_channel_data(br, frame_len_base)?;
            let pair = parse_two_channel_data(br, frame_len_base)?;
            IceCoreData::Grouping1 { three, pair }
        }
        2 => {
            let four = parse_four_channel_data(br, frame_len_base)?;
            let mono = parse_mono_data(br, false, frame_len_base)?;
            IceCoreData::Grouping2 { four, mono }
        }
        _ => {
            let five = parse_five_channel_data(br, frame_len_base)?;
            IceCoreData::Grouping3 { five }
        }
    };
    // 7CH_STATIC extras (Table 96 — every mode but ASPX_AJCC).
    let mut b_use_sap_add_ch = None;
    let mut sap_add_chparam = None;
    let mut add_pair = None;
    if mode.is_7ch_static() {
        let use_sap = br.read_bit()?;
        b_use_sap_add_ch = Some(use_sap);
        if use_sap {
            let ctx = core_de_max_sfb(&core)?;
            let cp0 = parse_chparam_info(br, &[ctx])?;
            let cp1 = parse_chparam_info(br, &[ctx])?;
            sap_add_chparam = Some(Box::new([cp0, cp1]));
        }
        add_pair = Some(parse_two_channel_data(br, frame_len_base)?);
    }
    // A-SPX payloads. `(n_leading_2ch, n_mid_2ch_after_1ch)` per mode:
    // ASPX_SCPL interleaves the 1ch element third; the ACPL / AJCC
    // family carries all 2ch elements first and the 1ch element last.
    let mut aspx_elements = Vec::new();
    if mode.uses_aspx() {
        let cfg = tools.aspx_config.ok_or_else(|| {
            Error::invalid("ac4: non-iframe immersive_channel_element without sticky aspx_config")
        })?;
        let cap_2ch = |br: &mut BitReader<'_>,
                       tools: &mut SubstreamTools,
                       out: &mut Vec<IceAspxElement>|
         -> Result<()> {
            let t = capture_aspx_data_2ch_trailer(br, tools, &cfg, b_iframe, frame_len_base)
                .ok_or_else(|| Error::invalid("ac4: ice aspx_data_2ch parse failed"))?;
            out.push(IceAspxElement::TwoCh(Some(Box::new(t))));
            Ok(())
        };
        let cap_1ch = |br: &mut BitReader<'_>,
                       tools: &mut SubstreamTools,
                       out: &mut Vec<IceAspxElement>|
         -> Result<()> {
            let t = capture_aspx_data_1ch_trailer(br, tools, &cfg, b_iframe, frame_len_base)
                .ok_or_else(|| Error::invalid("ac4: ice aspx_data_1ch parse failed"))?;
            out.push(IceAspxElement::OneCh(Some(Box::new(t))));
            Ok(())
        };
        match mode {
            IceCodecMode::AspxScpl => {
                cap_2ch(br, tools, &mut aspx_elements)?;
                cap_2ch(br, tools, &mut aspx_elements)?;
                cap_1ch(br, tools, &mut aspx_elements)?;
                let n = if b_5fronts { 2 } else { 1 };
                for _ in 0..n {
                    cap_2ch(br, tools, &mut aspx_elements)?;
                }
                cap_2ch(br, tools, &mut aspx_elements)?;
                cap_2ch(br, tools, &mut aspx_elements)?;
            }
            IceCodecMode::AspxAcpl1 | IceCodecMode::AspxAcpl2 | IceCodecMode::AspxAjcc => {
                cap_2ch(br, tools, &mut aspx_elements)?;
                cap_2ch(br, tools, &mut aspx_elements)?;
                if mode.is_7ch_static() {
                    cap_2ch(br, tools, &mut aspx_elements)?;
                }
                cap_1ch(br, tools, &mut aspx_elements)?;
            }
            IceCodecMode::Scpl => unreachable!("uses_aspx() gates SCPL out"),
        }
        // NOTE: `tools` carries a single sticky xover slot (the same
        // one-xover-per-substream convention as the 5_X / 7_X ASPX
        // trailer paths); harvest the first payload's xover for the
        // P-frames that follow this I-frame.
        if b_iframe {
            if let Some(IceAspxElement::TwoCh(Some(t)) | IceAspxElement::OneCh(Some(t))) =
                aspx_elements.first()
            {
                tools.aspx_xover_subband_offset = Some(t.xover);
            }
        }
    }
    // ajcc_data(b_5fronts) — ASPX_AJCC only.
    let ajcc = if matches!(mode, IceCodecMode::AspxAjcc) {
        Some(Box::new(parse_ajcc_data(br, b_5fronts)?))
    } else {
        None
    };
    // S-CPL section (SCPL / ASPX_SCPL / ASPX_ACPL_1).
    let mut scpl_pairs = Vec::new();
    let mut scpl_chparam = Vec::new();
    if mode.has_scpl_section() {
        scpl_pairs.push(parse_two_channel_data(br, frame_len_base)?);
        scpl_pairs.push(parse_two_channel_data(br, frame_len_base)?);
        for j in 0..4usize {
            let ctx = scpl_pairs[j / 2]
                .psy_info
                .as_ref()
                .map(|p| p.max_sfb_0)
                .ok_or_else(|| Error::invalid("ac4: ice scpl chparam context missing"))?;
            scpl_chparam.push(parse_chparam_info(br, &[ctx])?);
        }
        if b_5fronts {
            scpl_pairs.push(parse_two_channel_data(br, frame_len_base)?);
            let ctx = scpl_pairs[2]
                .psy_info
                .as_ref()
                .map(|p| p.max_sfb_0)
                .ok_or_else(|| Error::invalid("ac4: ice scpl chparam context missing"))?;
            scpl_chparam.push(parse_chparam_info(br, &[ctx])?);
            scpl_chparam.push(parse_chparam_info(br, &[ctx])?);
        }
    }
    // A-CPL parameter section (ASPX_ACPL_1 / ASPX_ACPL_2).
    let mut acpl_data = Vec::new();
    if mode.has_acpl_section() {
        let acfg: AcplConfig1ch = match mode {
            IceCodecMode::AspxAcpl1 => tools.acpl_config_1ch_partial,
            _ => tools.acpl_config_1ch_full,
        }
        .ok_or_else(|| {
            Error::invalid("ac4: non-iframe immersive_channel_element without sticky acpl_config")
        })?;
        let start_band = if acfg.qmf_band == 0 {
            0
        } else {
            sb_to_pb(acfg.qmf_band as u32, acfg.num_param_bands)
        };
        let n = if b_5fronts { 6 } else { 4 };
        for _ in 0..n {
            acpl_data.push(parse_acpl_data_1ch(
                br,
                acfg.num_param_bands,
                start_band,
                acfg.quant_mode,
            )?);
        }
    }
    tools.ice = Some(Box::new(IceElement {
        b_lfe,
        b_5fronts,
        mode,
        lfe,
        companding,
        core_5ch_grouping,
        core,
        b_use_sap_add_ch,
        sap_add_chparam,
        add_pair,
        aspx_elements,
        ajcc,
        scpl_pairs,
        scpl_chparam,
        acpl_data,
    }));
    Ok(())
}

// =====================================================================
// §6.2.4.3 22_2_channel_element
// =====================================================================

/// Parsed `22_2_channel_element(b_iframe)` (TS 103 190-2 §6.2.4.3):
/// two LFE `mono_data(1)` bodies, eleven `two_channel_data()` pairs,
/// and (for `22_2_codec_mode == ASPX`, Table 98) one
/// `aspx_data_2ch()` payload per pair. The §5.2.4 Table 21 mapping
/// assigns the pairs to `[L, R]`, `[C, Tc]`, `[Ls, Rs]`, `[Lb, Rb]`,
/// `[Tfl, Tfr]`, `[Tbl, Tbr]`, `[Tsl, Tsr]`, `[Tfc, Tbc]`,
/// `[Bfl, Bfr]`, `[Bfc, Cb]`, `[Lw, Rw]` in transmission order.
#[derive(Debug, Clone)]
pub struct El222 {
    /// `22_2_codec_mode` — false = Simple, true = A-SPX (Table 98).
    pub aspx_mode: bool,
    /// The two LFE `mono_data(1)` bodies (LFE, LFE2).
    pub lfe: [MonoLfeData; 2],
    /// The eleven `two_channel_data()` pairs in transmission order.
    pub pairs: Vec<TwoChannelData>,
    /// One captured `aspx_data_2ch()` trailer per pair (A-SPX mode;
    /// empty for Simple). `None` entries mean the capture bailed.
    pub aspx_elements: Vec<Option<Box<FiveXAspxTrailer>>>,
}

impl El222 {
    /// The 22 fullband spectra in the Table 21 channel order
    /// (`[L, R, C, Tc, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr, Tsl, Tsr,
    /// Tfc, Tbc, Bfl, Bfr, Bfc, Cb, Lw, Rw]`); uncoded tracks are
    /// `None`.
    pub fn fullband_spectra(&self) -> Vec<Option<&[f32]>> {
        let mut out: Vec<Option<&[f32]>> = Vec::with_capacity(22);
        for p in &self.pairs {
            for ch in 0..2 {
                out.push(p.scaled_spec_per_channel.get(ch).and_then(|s| s.as_deref()));
            }
        }
        out.resize(22, None);
        out
    }

    /// The core transform length (from the first pair).
    pub fn transform_length(&self) -> Option<u32> {
        self.pairs
            .first()
            .and_then(|p| p.transform_info.as_ref())
            .map(|t| t.transform_length_0)
    }
}

/// Parse `22_2_channel_element(b_iframe)` into `tools` (the A-SPX
/// config lands on the shared I-frame-sticky slot; the element body
/// lands on `tools.el_22_2`). For `b_iframe == 0` the caller must have
/// pre-seeded `tools` with the sticky `aspx_config` /
/// `aspx_xover_subband_offset` from the last I-frame.
pub fn parse_22_2_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    let aspx_mode = br.read_bit()?;
    if b_iframe && aspx_mode {
        tools.aspx_config = Some(parse_aspx_config(br)?);
    }
    let lfe0 = parse_mono_data(br, true, frame_len_base)?;
    let lfe1 = parse_mono_data(br, true, frame_len_base)?;
    let mut pairs = Vec::with_capacity(11);
    for _ in 0..11 {
        pairs.push(parse_two_channel_data(br, frame_len_base)?);
    }
    let mut aspx_elements = Vec::new();
    if aspx_mode {
        let cfg = tools.aspx_config.ok_or_else(|| {
            Error::invalid("ac4: non-iframe 22_2_channel_element without sticky aspx_config")
        })?;
        for _ in 0..11 {
            let t = capture_aspx_data_2ch_trailer(br, tools, &cfg, b_iframe, frame_len_base)
                .ok_or_else(|| Error::invalid("ac4: 22_2 aspx_data_2ch parse failed"))?;
            aspx_elements.push(Some(Box::new(t)));
        }
        // Same one-xover-per-substream sticky convention as the ICE /
        // 5_X / 7_X trailer paths.
        if b_iframe {
            if let Some(Some(t)) = aspx_elements.first() {
                tools.aspx_xover_subband_offset = Some(t.xover);
            }
        }
    }
    tools.el_22_2 = Some(Box::new(El222 {
        aspx_mode,
        lfe: [lfe0, lfe1],
        pairs,
        aspx_elements,
    }));
    Ok(())
}

// =====================================================================
// Write side
// =====================================================================

/// Write `five_channel_data()` (part-1 Table 29) for the long-frame,
/// single-window-group, identity-SAP case (`chel_matsel = 0`,
/// `sap_mode = 0` on all five `chparam_info()` elements).
pub fn write_five_channel_data_simple(
    bw: &mut BitWriter,
    spectra: &[&[f32]; 5],
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let sfbo = crate::sfb_offset::sfb_offset_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(transform_length)
        .ok_or_else(|| Error::invalid("ac4: unsupported transform_length"))?;
    // sf_info(ASF, 0, 0): b_long_frame = 1 + shared max_sfb.
    bw.write_bit(true);
    bw.write_u32(max_sfb, n_msfb_bits);
    // five_channel_info(): chel_matsel = 0 + 5× chparam_info(sap 0).
    bw.write_u32(0, 4);
    for _ in 0..5 {
        bw.write_u32(0, 2);
    }
    for s in spectra {
        crate::ajoc_substream::write_asf_body_from_spectrum(bw, s, sfbo, max_sfb);
    }
    Ok(())
}

/// Track spectra needed to write one grouping-3 immersive element:
/// core `[A, B, C, D, E]`, the additional `[F, G]` pair, and the
/// S-CPL pairs `[H, I]`, `[J, K]` (+ `[L, M]` for b_5fronts).
#[derive(Debug, Clone, Copy)]
pub struct IceScplSpectra<'a> {
    /// Core tracks A..E.
    pub core: &'a [&'a [f32]; 5],
    /// Additional pair F, G.
    pub add_pair: [&'a [f32]; 2],
    /// S-CPL pairs — 2 entries, or 3 with b_5fronts.
    pub scpl_pairs: &'a [[&'a [f32]; 2]],
}

/// Write a SCPL-mode `immersive_channel_element()` body: grouping 3
/// (`five_channel_data`), `b_use_sap_add_ch = 0`, identity chparam
/// (`sap_mode = 0`) on the S-CPL section, optional leading LFE
/// `mono_data(1)` (`lfe = (coeffs, max_sfb_lfe)`).
///
/// SCPL carries no A-SPX layer, so the body is fully deterministic
/// from the MDCT spectra.
pub fn write_ice_body_scpl(
    bw: &mut BitWriter,
    spectra: &IceScplSpectra<'_>,
    lfe: Option<(&[f32], u32)>,
    b_5fronts: bool,
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    write_ice_body_scpl_with_sap(
        bw,
        spectra,
        lfe,
        b_5fronts,
        transform_length,
        max_sfb,
        None,
        &[],
    )
}

/// [`write_ice_body_scpl`] with explicit SAP payloads: `sap_add_pair`
/// emits `b_use_sap_add_ch = 1` plus the two step-3 `chparam_info()`
/// elements, and `scpl_chparam` replaces the identity (`sap_mode = 0`)
/// S-CPL-section elements — pass the 4 (or 6 with `b_5fronts`)
/// elements whose `a'_j` gains drive the Table 20 step-6 rows; an
/// empty slice keeps the identity elements. All `chparam_info()`
/// bodies are emitted against the shared single-window-group
/// `max_sfb` bound (the same context the parser hands them).
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_scpl_with_sap(
    bw: &mut BitWriter,
    spectra: &IceScplSpectra<'_>,
    lfe: Option<(&[f32], u32)>,
    b_5fronts: bool,
    transform_length: u32,
    max_sfb: u32,
    sap_add_pair: Option<&[ChparamInfo; 2]>,
    scpl_chparam: &[ChparamInfo],
) -> Result<()> {
    let want_pairs = if b_5fronts { 3 } else { 2 };
    if spectra.scpl_pairs.len() != want_pairs {
        return Err(Error::invalid("ac4: ice scpl pair count mismatch"));
    }
    let n_chparam = 2 * want_pairs;
    if !scpl_chparam.is_empty() && scpl_chparam.len() != n_chparam {
        return Err(Error::invalid("ac4: ice scpl chparam count mismatch"));
    }
    IceCodecMode::Scpl.write(bw);
    // immers_cfg(SCPL) is empty (no aspx_config, no acpl_config).
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        crate::ajoc_substream::write_mono_data_lfe_simple(
            bw,
            coeffs,
            transform_length,
            max_sfb_lfe,
        )?;
    }
    // core_5ch_grouping = 3 (five_channel_data).
    bw.write_u32(3, 2);
    write_five_channel_data_simple(bw, spectra.core, transform_length, max_sfb)?;
    // 7CH_STATIC: b_use_sap_add_ch + additional pair (F, G).
    match sap_add_pair {
        Some(pair) => {
            bw.write_bit(true);
            for info in pair.iter() {
                crate::encoder_asf::write_chparam_info(bw, info, &[max_sfb]);
            }
        }
        None => bw.write_bit(false),
    }
    crate::ajoc_substream::write_two_channel_data_simple(
        bw,
        spectra.add_pair[0],
        spectra.add_pair[1],
        transform_length,
        max_sfb,
    )?;
    // S-CPL section: pairs then chparam elements.
    let write_chparam = |bw: &mut BitWriter, j: usize| match scpl_chparam.get(j) {
        Some(info) => crate::encoder_asf::write_chparam_info(bw, info, &[max_sfb]),
        None => bw.write_u32(0, 2),
    };
    for p in &spectra.scpl_pairs[..2] {
        crate::ajoc_substream::write_two_channel_data_simple(
            bw,
            p[0],
            p[1],
            transform_length,
            max_sfb,
        )?;
    }
    for j in 0..4 {
        write_chparam(bw, j);
    }
    if b_5fronts {
        let p = &spectra.scpl_pairs[2];
        crate::ajoc_substream::write_two_channel_data_simple(
            bw,
            p[0],
            p[1],
            transform_length,
            max_sfb,
        )?;
        for j in 4..6 {
            write_chparam(bw, j);
        }
    }
    Ok(())
}

/// Write an ASPX_AJCC-mode `immersive_channel_element()` body:
/// grouping 3 core, `companding_control(5)` with `sync_flag = 1` /
/// `b_compand_on = 0` (no decoder-side gain), single-envelope FIXFIX
/// A-SPX payloads with a **floored noise envelope** (`qnoise = 30` →
/// `2^(6−30)` per Pseudocode 83 — the all-zero minimal scaffold's
/// `qnoise = 0` decodes to a full-scale HF noise floor on every
/// channel), and the given `ajcc_data()` element.
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_ajcc(
    bw: &mut BitWriter,
    core: &[&[f32]; 5],
    ajcc: &AjccData,
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let off = CompandingControl {
        sync_flag: Some(true),
        compand_on: vec![false],
        compand_avg: Some(false),
    };
    write_ice_body_ajcc_with_companding(
        bw,
        core,
        ajcc,
        aspx_cfg,
        lfe,
        b_iframe,
        transform_length,
        max_sfb,
        &off,
        false,
    )
}

/// [`write_ice_body_ajcc`] with an explicit `companding_control(5)`
/// payload (§4.8.3.10.3 — channel order L, R, C, Ls, Rs) and a
/// `loud_hf` switch: when set, the A-SPX payload noise rows stay at
/// the all-zero full-scale floor (`qnoise = 0`) so the §5.7.5
/// companding gains act on a measurable regenerated band.
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_ajcc_with_companding(
    bw: &mut BitWriter,
    core: &[&[f32]; 5],
    ajcc: &AjccData,
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
    companding: &CompandingControl,
    loud_hf: bool,
) -> Result<()> {
    let rows = if loud_hf {
        MinimalAspxRows::derive_loud(aspx_cfg)?
    } else {
        MinimalAspxRows::derive(aspx_cfg)?
    };
    let two_ch = vec![rows.rows_2ch(); 2];
    let one_ch = rows.rows_1ch();
    write_ice_body_ajcc_real(
        bw,
        core,
        ajcc,
        aspx_cfg,
        lfe,
        b_iframe,
        transform_length,
        max_sfb,
        companding,
        &IceAspxRows {
            two_ch: &two_ch,
            one_ch: &one_ch,
        },
    )
}

/// [`write_ice_body_ajcc_with_companding`] with **real**
/// (synthesis-driven) A-SPX payload rows. The 5CH_DYNAMIC ASPX_AJCC
/// roster carries exactly 2× `aspx_data_2ch()` + 1× `aspx_data_1ch()`;
/// per the decoder's association the payloads extend the core tracks
/// `(A, B)`, `(D, E)` and `C`, so `rows.two_ch` (length 2) carries the
/// per-track-pair rows in that order and `rows.one_ch` the C-track
/// rows.
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_ajcc_real(
    bw: &mut BitWriter,
    core: &[&[f32]; 5],
    ajcc: &AjccData,
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
    companding: &CompandingControl,
    rows: &IceAspxRows<'_>,
) -> Result<()> {
    if rows.two_ch.len() != 2 {
        return Err(Error::invalid("ac4: ice ajcc 2ch payload count"));
    }
    IceCodecMode::AspxAjcc.write(bw);
    if b_iframe {
        crate::encoder_acpl3::write_aspx_config(bw, aspx_cfg);
    }
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        crate::ajoc_substream::write_mono_data_lfe_simple(
            bw,
            coeffs,
            transform_length,
            max_sfb_lfe,
        )?;
    }
    // companding_control(5) — exact inverse of
    // `parse_companding_control(_, 5)`.
    match companding.sync_flag {
        Some(sync) => bw.write_bit(sync),
        None => {
            return Err(Error::invalid("ac4: companding_control(5) needs sync_flag"));
        }
    }
    let nc = if companding.sync_flag == Some(true) {
        1
    } else {
        5
    };
    if companding.compand_on.len() != nc {
        return Err(Error::invalid("ac4: companding_control compand_on length"));
    }
    let mut need_avg = false;
    for &b in &companding.compand_on {
        bw.write_bit(b);
        if !b {
            need_avg = true;
        }
    }
    if need_avg {
        bw.write_bit(companding.compand_avg.unwrap_or(false));
    }
    // core_5ch_grouping = 3.
    bw.write_u32(3, 2);
    write_five_channel_data_simple(bw, core, transform_length, max_sfb)?;
    // A-SPX: 2× aspx_data_2ch + 1× aspx_data_1ch (5CH_DYNAMIC).
    for r in rows.two_ch {
        r.write(bw, aspx_cfg, b_iframe)?;
    }
    rows.one_ch.write(bw, aspx_cfg, b_iframe)?;
    write_ajcc_data(bw, ajcc)?;
    Ok(())
}

/// Minimal (floored-noise) A-SPX payload rows shared by the immersive
/// body writers: sig F0 = 0 (codebook minimum), noise F0 = 30
/// (`2^(6−30)` per Pseudocode 83 — an all-zero `qnoise = 0` row
/// decodes to a full-scale HF noise floor). The channel-1 rows go out
/// through the BALANCE codebooks but decode to the same absolute
/// q-values in the in-tree chain, so the secondary noise row is
/// floored as well — the balance noise F0 codebook tops out at 12, and
/// the writer clamps 30 down to it (r414's all-zero ch1 noise row left
/// a full-scale noise floor on every secondary-extended channel).
struct MinimalAspxRows {
    sig_row: Vec<i32>,
    noise_row: Vec<i32>,
    tna: Vec<u8>,
}

impl MinimalAspxRows {
    fn derive(aspx_cfg: &AspxConfig) -> Result<Self> {
        Self::derive_with_noise_floor(aspx_cfg, 30)
    }

    /// Loud-HF variant for the companding round-trip harness: a
    /// mid-codebook SIGNAL F0 (`qsig = 46` → `scf = 64·2²³`, near a
    /// full-scale subband at the [`crate::aspx::ASPX_QMF_PCM_SCALE`]
    /// integer-PCM anchor) plus the full `qnoise = 0` noise ratio, so
    /// the §5.7.5 companding gains act on a measurable regenerated
    /// band.
    fn derive_loud(aspx_cfg: &AspxConfig) -> Result<Self> {
        let mut rows = Self::derive_with_noise_floor(aspx_cfg, 0)?;
        if let Some(f0) = rows.sig_row.first_mut() {
            *f0 = 46;
        }
        Ok(rows)
    }

    fn derive_with_noise_floor(aspx_cfg: &AspxConfig, noise_f0: i32) -> Result<Self> {
        let tables = crate::aspx::derive_aspx_frequency_tables(aspx_cfg, 0)
            .map_err(|_| Error::invalid("ac4: aspx frequency-table derivation failed"))?;
        let counts = tables.counts;
        let num_sbg_sig = if aspx_cfg.signals_freq_res() {
            counts.num_sbg_sig_lowres
        } else {
            counts.num_sbg_sig_highres
        } as usize;
        let num_sbg_noise = counts.num_sbg_noise as usize;
        let sig_row = vec![0i32; num_sbg_sig.max(1)];
        let mut noise_row = vec![0i32; num_sbg_noise.max(1)];
        noise_row[0] = noise_f0;
        Ok(Self {
            sig_row,
            noise_row,
            tna: vec![0u8; num_sbg_noise],
        })
    }

    fn level(&self) -> crate::encoder_acpl3::AspxRealEnvelopeChannel<'_> {
        crate::encoder_acpl3::AspxRealEnvelopeChannel {
            sig: &self.sig_row,
            noise: &self.noise_row,
        }
    }

    fn write_2ch(&self, bw: &mut BitWriter, aspx_cfg: &AspxConfig, b_iframe: bool) -> Result<()> {
        crate::encoder_acpl3::write_aspx_data_2ch_real_envelope_tna_ah_framed(
            bw,
            aspx_cfg,
            self.level(),
            self.level(),
            &self.tna,
            &[],
            &[],
            b_iframe,
        )
        .map_err(Error::invalid)
    }

    fn write_1ch(&self, bw: &mut BitWriter, aspx_cfg: &AspxConfig, b_iframe: bool) -> Result<()> {
        crate::encoder_acpl3::write_aspx_data_1ch_real_envelope_tna_ah_framed(
            bw,
            aspx_cfg,
            self.level(),
            &self.tna,
            &[],
            b_iframe,
        )
        .map_err(Error::invalid)
    }

    fn rows_2ch(&self) -> IceAspx2chRows {
        IceAspx2chRows {
            ch0: IceAspxChannelRows {
                sig: self.sig_row.clone(),
                noise: self.noise_row.clone(),
                ah: Vec::new(),
            },
            ch1: IceAspxChannelRows {
                sig: self.sig_row.clone(),
                noise: self.noise_row.clone(),
                ah: Vec::new(),
            },
            tna: self.tna.clone(),
        }
    }

    fn rows_1ch(&self) -> IceAspx1chRows {
        IceAspx1chRows {
            ch: IceAspxChannelRows {
                sig: self.sig_row.clone(),
                noise: self.noise_row.clone(),
                ah: Vec::new(),
            },
            tna: self.tna.clone(),
        }
    }
}

// =====================================================================
// Real (synthesis-driven) A-SPX payload rows for the ICE body writers
// =====================================================================

/// Real A-SPX quant-index rows for one extended channel of an ICE
/// payload — the encoder-side counterpart of the decoder's
/// Pseudocode 82 / 83 dequantised envelopes. `sig` / `noise` carry the
/// FREQ-DPCM `[F0, DF₁, …]` vectors (see
/// [`crate::encoder_acpl3::build_aspx_real_envelope_channel_from_qmf`]);
/// `ah` the per-high-res-signal-SBG `aspx_add_harmonic` decisions
/// (§4.2.12.6). Empty vectors zero-pad in the underlying writers —
/// note an all-zero NOISE row decodes to a full-scale HF noise floor
/// (Pseudocode 83's `2^(6−0)`), so callers should always supply a real
/// (or floored) noise row.
#[derive(Debug, Clone, Default)]
pub struct IceAspxChannelRows {
    /// SIGNAL envelope FREQ-DPCM quant indices `[F0, DF₁, …]`.
    pub sig: Vec<i32>,
    /// NOISE envelope FREQ-DPCM quant indices `[F0, DF₁, …]`.
    pub noise: Vec<i32>,
    /// Per-high-res-signal-SBG `aspx_add_harmonic` decisions.
    pub ah: Vec<bool>,
}

/// Real A-SPX rows for one `aspx_data_2ch()` ICE payload. The `tna`
/// vector is the primary-derived per-noise-SBG `aspx_tna_mode`
/// (§4.3.10.6.1), mirrored to the secondary channel under the balance
/// coding — matching the 5_X / 7_X live-path writers.
#[derive(Debug, Clone, Default)]
pub struct IceAspx2chRows {
    /// Primary-channel rows.
    pub ch0: IceAspxChannelRows,
    /// Secondary-channel rows.
    pub ch1: IceAspxChannelRows,
    /// Shared per-noise-SBG `aspx_tna_mode` vector.
    pub tna: Vec<u8>,
}

impl IceAspx2chRows {
    fn write(&self, bw: &mut BitWriter, aspx_cfg: &AspxConfig, b_iframe: bool) -> Result<()> {
        crate::encoder_acpl3::write_aspx_data_2ch_real_envelope_tna_ah_framed(
            bw,
            aspx_cfg,
            crate::encoder_acpl3::AspxRealEnvelopeChannel {
                sig: &self.ch0.sig,
                noise: &self.ch0.noise,
            },
            crate::encoder_acpl3::AspxRealEnvelopeChannel {
                sig: &self.ch1.sig,
                noise: &self.ch1.noise,
            },
            &self.tna,
            &self.ch0.ah,
            &self.ch1.ah,
            b_iframe,
        )
        .map_err(Error::invalid)
    }
}

/// Real A-SPX rows for one `aspx_data_1ch()` ICE payload.
#[derive(Debug, Clone, Default)]
pub struct IceAspx1chRows {
    /// The single channel's rows.
    pub ch: IceAspxChannelRows,
    /// Per-noise-SBG `aspx_tna_mode` vector.
    pub tna: Vec<u8>,
}

impl IceAspx1chRows {
    fn write(&self, bw: &mut BitWriter, aspx_cfg: &AspxConfig, b_iframe: bool) -> Result<()> {
        crate::encoder_acpl3::write_aspx_data_1ch_real_envelope_tna_ah_framed(
            bw,
            aspx_cfg,
            crate::encoder_acpl3::AspxRealEnvelopeChannel {
                sig: &self.ch.sig,
                noise: &self.ch.noise,
            },
            &self.tna,
            &self.ch.ah,
            b_iframe,
        )
        .map_err(Error::invalid)
    }
}

/// The full real-A-SPX payload roster for one ICE element body: the
/// `two_ch` payloads in transmission order with the single
/// `aspx_data_1ch()` payload factored out (the writers splice it at
/// the mode's roster position — third for ASPX_SCPL, last for the
/// ACPL / AJCC family).
#[derive(Debug, Clone)]
pub struct IceAspxRows<'a> {
    /// `aspx_data_2ch()` rows in transmission order (5 / 6 for
    /// ASPX_SCPL without / with `b_5fronts`; 3 for the ACPL family).
    pub two_ch: &'a [IceAspx2chRows],
    /// The `aspx_data_1ch()` rows (the centre-carrier payload).
    pub one_ch: &'a IceAspx1chRows,
}

/// Write an ASPX_SCPL-mode `immersive_channel_element()` body:
/// grouping-3 core, `b_use_sap_add_ch = 0`, the §6.2.4.1 ASPX_SCPL
/// payload roster (minimal floored-noise single-envelope FIXFIX
/// payloads — see [`MinimalAspxRows`]), and the identity
/// (`sap_mode = 0`) S-CPL section. `aspx_config()` is emitted on
/// I-frames only (P-frames reuse the sticky config).
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_aspx_scpl(
    bw: &mut BitWriter,
    spectra: &IceScplSpectra<'_>,
    lfe: Option<(&[f32], u32)>,
    b_5fronts: bool,
    aspx_cfg: &AspxConfig,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let rows = MinimalAspxRows::derive(aspx_cfg)?;
    let two_ch = vec![rows.rows_2ch(); if b_5fronts { 6 } else { 5 }];
    let one_ch = rows.rows_1ch();
    write_ice_body_aspx_scpl_real(
        bw,
        spectra,
        lfe,
        b_5fronts,
        aspx_cfg,
        b_iframe,
        transform_length,
        max_sfb,
        &IceAspxRows {
            two_ch: &two_ch,
            one_ch: &one_ch,
        },
    )
}

/// [`write_ice_body_aspx_scpl`] with **real** (synthesis-driven) A-SPX
/// payload rows: `rows.two_ch` carries the 5 (or 6 with `b_5fronts`)
/// `aspx_data_2ch()` payloads in transmission order and `rows.one_ch`
/// the centre `aspx_data_1ch()` payload spliced in at the ASPX_SCPL
/// roster's third position. Per the decoder's Table 8 channel-grouping
/// association the payloads extend the **decoupled** output channels
/// (post-Table 23, pre-§4.8.3.11.3 gain): without `b_5fronts` the
/// `two_ch` order is `(L, R)`, `(Ls, Lb)`, `(Rs, Rb)`, `(Tfl, Tbl)`,
/// `(Tfr, Tbr)` (centre payload = `C`); with `b_5fronts` it is
/// `(L, Lscr)`, `(R, Rscr)`, `(Ls, Lb)`, `(Rs, Rb)`, `(Tfl, Tbl)`,
/// `(Tfr, Tbr)`.
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_aspx_scpl_real(
    bw: &mut BitWriter,
    spectra: &IceScplSpectra<'_>,
    lfe: Option<(&[f32], u32)>,
    b_5fronts: bool,
    aspx_cfg: &AspxConfig,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
    rows: &IceAspxRows<'_>,
) -> Result<()> {
    let want_pairs = if b_5fronts { 3 } else { 2 };
    if spectra.scpl_pairs.len() != want_pairs {
        return Err(Error::invalid("ac4: ice scpl pair count mismatch"));
    }
    let want_2ch = if b_5fronts { 6 } else { 5 };
    if rows.two_ch.len() != want_2ch {
        return Err(Error::invalid("ac4: ice aspx_scpl 2ch payload count"));
    }
    IceCodecMode::AspxScpl.write(bw);
    // immers_cfg(ASPX_SCPL): aspx_config() on I-frames.
    if b_iframe {
        crate::encoder_acpl3::write_aspx_config(bw, aspx_cfg);
    }
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        crate::ajoc_substream::write_mono_data_lfe_simple(
            bw,
            coeffs,
            transform_length,
            max_sfb_lfe,
        )?;
    }
    // core_5ch_grouping = 3 (five_channel_data).
    bw.write_u32(3, 2);
    write_five_channel_data_simple(bw, spectra.core, transform_length, max_sfb)?;
    // 7CH_STATIC: b_use_sap_add_ch = 0 + additional pair (F, G).
    bw.write_bit(false);
    crate::ajoc_substream::write_two_channel_data_simple(
        bw,
        spectra.add_pair[0],
        spectra.add_pair[1],
        transform_length,
        max_sfb,
    )?;
    // ASPX_SCPL roster: 2ch, 2ch, 1ch, (b_5fronts ? 2ch 2ch : 2ch),
    // 2ch, 2ch.
    rows.two_ch[0].write(bw, aspx_cfg, b_iframe)?;
    rows.two_ch[1].write(bw, aspx_cfg, b_iframe)?;
    rows.one_ch.write(bw, aspx_cfg, b_iframe)?;
    for r in &rows.two_ch[2..] {
        r.write(bw, aspx_cfg, b_iframe)?;
    }
    // S-CPL section: pairs then identity chparam elements.
    for p in &spectra.scpl_pairs[..2] {
        crate::ajoc_substream::write_two_channel_data_simple(
            bw,
            p[0],
            p[1],
            transform_length,
            max_sfb,
        )?;
    }
    for _ in 0..4 {
        bw.write_u32(0, 2);
    }
    if b_5fronts {
        let p = &spectra.scpl_pairs[2];
        crate::ajoc_substream::write_two_channel_data_simple(
            bw,
            p[0],
            p[1],
            transform_length,
            max_sfb,
        )?;
        for _ in 0..2 {
            bw.write_u32(0, 2);
        }
    }
    Ok(())
}

/// Per-module quantised A-CPL parameters for the ASPX_ACPL_1 /
/// ASPX_ACPL_2 body writer: one `(alpha_q, beta_q)` row pair per
/// parameter band, one entry per `acpl_data_1ch()` element (4, or 6
/// with `b_5fronts`), in the §6.2.4.1 transmission order (= the §5.5.2
/// module order: (Ls, Lb), (Rs, Rb), (Tfl, Tbl), (Tfr, Tbr)
/// [, (L, Lscr), (R, Rscr)]).
pub struct IceAcplParams<'a> {
    /// `acpl_num_param_bands_id` (2 bits, Table 59).
    pub num_param_bands_id: u8,
    /// ALPHA / BETA quantiser mode.
    pub quant_mode: crate::acpl::AcplQuantMode,
    /// `acpl_qmf_band` for the PARTIAL (ASPX_ACPL_1) config — the
    /// M/S-coded low-band split. Ignored for ASPX_ACPL_2 (FULL).
    pub qmf_band: u8,
    /// Per-module `(alpha_q, beta_q)` per parameter band.
    pub modules: &'a [(&'a [i32], &'a [i32])],
}

/// Write an ASPX_ACPL_1- / ASPX_ACPL_2-mode
/// `immersive_channel_element()` body: grouping-3 core,
/// `b_use_sap_add_ch = 0`, the shared minimal A-SPX roster
/// (3× `aspx_data_2ch` + 1× `aspx_data_1ch` — 7CH_STATIC), for
/// ASPX_ACPL_1 the S-CPL section (`scpl_pairs` carries the residual
/// tracks H..K / H..M with identity `sap_mode = 0` chparam elements),
/// and the trailing `acpl_data_1ch()` parameter section. On I-frames
/// `immers_cfg()` emits `aspx_config()` plus the matching
/// `acpl_config_1ch(PARTIAL / FULL)`.
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_acpl(
    bw: &mut BitWriter,
    mode: IceCodecMode,
    core: &[&[f32]; 5],
    add_pair: [&[f32]; 2],
    scpl_pairs: &[[&[f32]; 2]],
    acpl: &IceAcplParams<'_>,
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    b_5fronts: bool,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let rows = MinimalAspxRows::derive(aspx_cfg)?;
    let two_ch = vec![rows.rows_2ch(); 3];
    let one_ch = rows.rows_1ch();
    write_ice_body_acpl_real(
        bw,
        mode,
        core,
        add_pair,
        scpl_pairs,
        acpl,
        aspx_cfg,
        lfe,
        b_5fronts,
        b_iframe,
        transform_length,
        max_sfb,
        &IceAspxRows {
            two_ch: &two_ch,
            one_ch: &one_ch,
        },
    )
}

/// [`write_ice_body_acpl`] with **real** (synthesis-driven) A-SPX
/// payload rows. The ACPL / AJCC-family roster carries exactly
/// 3× `aspx_data_2ch()` + 1× `aspx_data_1ch()`; per the V1.3.1
/// Table 8 association the payloads extend the **tracks** `(A, B)`,
/// `(D, E)`, `(F, G)` and `C`, so `rows.two_ch` (length 3) carries the
/// per-track-pair rows in that order and `rows.one_ch` the C-track
/// rows.
#[allow(clippy::too_many_arguments)]
pub fn write_ice_body_acpl_real(
    bw: &mut BitWriter,
    mode: IceCodecMode,
    core: &[&[f32]; 5],
    add_pair: [&[f32]; 2],
    scpl_pairs: &[[&[f32]; 2]],
    acpl: &IceAcplParams<'_>,
    aspx_cfg: &AspxConfig,
    lfe: Option<(&[f32], u32)>,
    b_5fronts: bool,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
    rows: &IceAspxRows<'_>,
) -> Result<()> {
    let is_acpl1 = match mode {
        IceCodecMode::AspxAcpl1 => true,
        IceCodecMode::AspxAcpl2 => false,
        _ => return Err(Error::invalid("ac4: write_ice_body_acpl mode")),
    };
    if rows.two_ch.len() != 3 {
        return Err(Error::invalid("ac4: ice acpl 2ch payload count"));
    }
    let want_pairs = if is_acpl1 {
        if b_5fronts {
            3
        } else {
            2
        }
    } else {
        0
    };
    if scpl_pairs.len() != want_pairs {
        return Err(Error::invalid("ac4: ice acpl scpl pair count mismatch"));
    }
    let n_modules = if b_5fronts { 6 } else { 4 };
    if acpl.modules.len() != n_modules {
        return Err(Error::invalid("ac4: ice acpl module count mismatch"));
    }
    mode.write(bw);
    if b_iframe {
        crate::encoder_acpl3::write_aspx_config(bw, aspx_cfg);
        // acpl_config_1ch(PARTIAL / FULL) per Table 59.
        bw.write_u32(acpl.num_param_bands_id as u32 & 0b11, 2);
        bw.write_bit(matches!(
            acpl.quant_mode,
            crate::acpl::AcplQuantMode::Coarse
        ));
        if is_acpl1 {
            let m1 = acpl.qmf_band.saturating_sub(1).min(7);
            bw.write_u32(m1 as u32, 3);
        }
    }
    if let Some((coeffs, max_sfb_lfe)) = lfe {
        crate::ajoc_substream::write_mono_data_lfe_simple(
            bw,
            coeffs,
            transform_length,
            max_sfb_lfe,
        )?;
    }
    // core_5ch_grouping = 3 (five_channel_data).
    bw.write_u32(3, 2);
    write_five_channel_data_simple(bw, core, transform_length, max_sfb)?;
    // 7CH_STATIC: b_use_sap_add_ch = 0 + additional pair (F, G).
    bw.write_bit(false);
    crate::ajoc_substream::write_two_channel_data_simple(
        bw,
        add_pair[0],
        add_pair[1],
        transform_length,
        max_sfb,
    )?;
    // A-SPX roster: 3× 2ch (7CH_STATIC) + 1ch.
    for r in rows.two_ch {
        r.write(bw, aspx_cfg, b_iframe)?;
    }
    rows.one_ch.write(bw, aspx_cfg, b_iframe)?;
    // S-CPL section (ASPX_ACPL_1 only).
    if is_acpl1 {
        for p in &scpl_pairs[..2] {
            crate::ajoc_substream::write_two_channel_data_simple(
                bw,
                p[0],
                p[1],
                transform_length,
                max_sfb,
            )?;
        }
        for _ in 0..4 {
            bw.write_u32(0, 2);
        }
        if b_5fronts {
            let p = &scpl_pairs[2];
            crate::ajoc_substream::write_two_channel_data_simple(
                bw,
                p[0],
                p[1],
                transform_length,
                max_sfb,
            )?;
            for _ in 0..2 {
                bw.write_u32(0, 2);
            }
        }
    }
    // acpl_data_1ch() section.
    let num_bands = crate::acpl::num_param_bands_from_id(acpl.num_param_bands_id as u32);
    let start_band = if is_acpl1 && acpl.qmf_band > 0 {
        sb_to_pb(acpl.qmf_band as u32, num_bands)
    } else {
        0
    };
    for (alpha_q, beta_q) in acpl.modules.iter() {
        crate::encoder_acpl3::write_acpl_data_1ch_real_alpha_beta(
            bw,
            num_bands,
            start_band,
            acpl.quant_mode,
            alpha_q,
            Some(beta_q),
        );
    }
    Ok(())
}

/// Write a `22_2_channel_element()` body (§6.2.4.3): Simple mode when
/// `aspx_cfg` is `None` (pure channel data), A-SPX mode otherwise
/// (I-frame `aspx_config()` + one minimal floored-noise
/// `aspx_data_2ch()` per pair). `lfe` carries the two LFE spectra
/// (`(coeffs, max_sfb_lfe)` each); `pairs` the eleven Table 21 channel
/// pairs in transmission order.
pub fn write_22_2_body(
    bw: &mut BitWriter,
    lfe: [(&[f32], u32); 2],
    pairs: &[[&[f32]; 2]; 11],
    aspx_cfg: Option<&AspxConfig>,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
) -> Result<()> {
    let two_ch = match aspx_cfg {
        Some(cfg) => vec![MinimalAspxRows::derive(cfg)?.rows_2ch(); 11],
        None => Vec::new(),
    };
    write_22_2_body_real(
        bw,
        lfe,
        pairs,
        aspx_cfg,
        b_iframe,
        transform_length,
        max_sfb,
        &two_ch,
    )
}

/// [`write_22_2_body`] with **real** (synthesis-driven) A-SPX payload
/// rows: one [`IceAspx2chRows`] per Table 21 channel pair in
/// transmission order (ignored — pass an empty slice — for the Simple
/// mode).
#[allow(clippy::too_many_arguments)]
pub fn write_22_2_body_real(
    bw: &mut BitWriter,
    lfe: [(&[f32], u32); 2],
    pairs: &[[&[f32]; 2]; 11],
    aspx_cfg: Option<&AspxConfig>,
    b_iframe: bool,
    transform_length: u32,
    max_sfb: u32,
    aspx_rows: &[IceAspx2chRows],
) -> Result<()> {
    if aspx_cfg.is_some() && aspx_rows.len() != 11 {
        return Err(Error::invalid("ac4: 22_2 aspx payload count"));
    }
    bw.write_bit(aspx_cfg.is_some()); // 22_2_codec_mode (Table 98)
    if let (true, Some(cfg)) = (b_iframe, aspx_cfg) {
        crate::encoder_acpl3::write_aspx_config(bw, cfg);
    }
    for (coeffs, max_sfb_lfe) in lfe {
        crate::ajoc_substream::write_mono_data_lfe_simple(
            bw,
            coeffs,
            transform_length,
            max_sfb_lfe,
        )?;
    }
    for p in pairs {
        crate::ajoc_substream::write_two_channel_data_simple(
            bw,
            p[0],
            p[1],
            transform_length,
            max_sfb,
        )?;
    }
    if let Some(cfg) = aspx_cfg {
        for r in aspx_rows {
            r.write(bw, cfg, b_iframe)?;
        }
    }
    Ok(())
}

/// The Table 78 channel-mode prefix code for an immersive layout.
fn ice_channel_mode_code(b_lfe: bool, b_5fronts: bool) -> (u32, u32) {
    match (b_5fronts, b_lfe) {
        (false, false) => (0b11111100, 8), // 7.0.4
        (false, true) => (0b11111101, 8),  // 7.1.4
        (true, false) => (0b111111100, 9), // 9.0.4
        (true, true) => (0b111111101, 9),  // 9.1.4
    }
}

/// Emit a complete v2 `raw_ac4_frame()` carrying one channel-coded
/// substream whose `audio_data_chan()` body is the given (byte-
/// unaligned-safe) immersive-channel-element body bits: `ac4_toc()`
/// (single presentation, single substream group, `b_channel_coded =
/// 1`, the Table 78 immersive channel_mode + presence fields) followed
/// by the `ac4_substream()` audio-size envelope around `body`.
///
/// Fixed shape: `fs_index = 1` (48 kHz), `frame_rate_index = 1`
/// (1920-sample frames), no HSF extension.
pub fn encode_ice_raw_frame(
    sequence_counter: u32,
    b_lfe: bool,
    b_5fronts: bool,
    b_iframe: bool,
    body: BitWriter,
) -> Result<Vec<u8>> {
    let (code, code_bits) = ice_channel_mode_code(b_lfe, b_5fronts);
    encode_chan_raw_frame(
        sequence_counter,
        code,
        code_bits,
        true,
        b_iframe,
        body,
        None,
    )
}

/// [`encode_ice_raw_frame`] with trailing substream `metadata(…)`
/// bytes (TS 103 190-2 §6.2.2.2: the `ac4_substream()` continues past
/// the announced `audio_size` with `metadata(b_alternative, b_ajoc,
/// b_audio_ndot, sus_ver = 1)` — e.g. a `dialog_enhancement()`
/// payload for the §4.8.3.15 tool). The caller provides the
/// byte-aligned metadata element.
pub fn encode_ice_raw_frame_with_metadata(
    sequence_counter: u32,
    b_lfe: bool,
    b_5fronts: bool,
    b_iframe: bool,
    body: BitWriter,
    metadata: &[u8],
) -> Result<Vec<u8>> {
    let (code, code_bits) = ice_channel_mode_code(b_lfe, b_5fronts);
    encode_chan_raw_frame(
        sequence_counter,
        code,
        code_bits,
        true,
        b_iframe,
        body,
        Some(metadata),
    )
}

/// [`encode_ice_raw_frame`] for the 22.2 channel mode (Table 78 code
/// `111111110`, no §6.3.2.7.3-5 presence fields).
pub fn encode_22_2_raw_frame(
    sequence_counter: u32,
    b_iframe: bool,
    body: BitWriter,
) -> Result<Vec<u8>> {
    encode_chan_raw_frame(
        sequence_counter,
        0b111111110,
        9,
        false,
        b_iframe,
        body,
        None,
    )
}

fn encode_chan_raw_frame(
    sequence_counter: u32,
    mode_code: u32,
    mode_code_bits: u32,
    presence_fields: bool,
    b_iframe: bool,
    body: BitWriter,
    metadata: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut bw = BitWriter::new();
    // ---- ac4_toc() (§6.2.1.1) ----
    bw.write_u32(2, 2); // bitstream_version = 2
    bw.write_u32(sequence_counter & 0x3FF, 10);
    bw.write_bit(false); // b_wait_frames
    bw.write_u32(1, 1); // fs_index = 48 kHz
    bw.write_u32(1, 4); // frame_rate_index = 1 (1920 samples)
    bw.write_bit(b_iframe); // b_iframe_global
    bw.write_bit(true); // b_single_presentation
    bw.write_bit(false); // b_payload_base
    bw.write_bit(false); // b_program_id
                         // ---- ac4_presentation_v1_info() (§6.2.1.3) ----
    bw.write_bit(true); // b_single_substream_group
    bw.write_bit(false); // presentation_version() = 0
    bw.write_u32(0, 3); // mdcompat
    bw.write_bit(false); // b_presentation_id
    bw.write_bit(false); // frame_rate_multiply_info: b_multiplier = 0
                         // emdf_info(): version 0, key_id 0, no payloads-substream, no more.
    bw.write_u32(0, 2);
    bw.write_u32(0, 3);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false); // b_presentation_filter
    bw.write_u32(0, 3); // ac4_sgi_specifier(): group_index = 0
    bw.write_bit(false); // b_pre_virtualized
    bw.write_bit(false); // b_add_emdf_substreams
                         // ac4_presentation_substream_info(): b_alternative = 0,
                         // b_pres_ndot = b_iframe (§6.3.2.11.2), substream_index = 0.
    bw.write_bit(false);
    bw.write_bit(b_iframe);
    bw.write_u32(0, 2);
    // ---- ac4_substream_group_info() (§6.2.1.6) ----
    bw.write_bit(true); // b_substreams_present
    bw.write_bit(false); // b_hsf_ext
    bw.write_bit(true); // b_single_substream
    bw.write_bit(true); // b_channel_coded
                        // ac4_substream_info_chan() (§6.2.1.8):
    bw.write_u32(mode_code, mode_code_bits);
    if presence_fields {
        // §6.3.2.7.3-5 presence fields (immersive modes only): back
        // channels + centre carry content, top_channels_present = 0.
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_u32(0, 2);
    }
    bw.write_bit(false); // b_sf_multiplier (fs_index == 1)
    bw.write_bit(false); // b_bitrate_info
    bw.write_bit(b_iframe); // b_audio_ndot = b_iframe (frame_rate_factor = 1)
    bw.write_u32(0, 2); // substream_index
    bw.write_bit(false); // b_content_type (group level, §6.2.1.6)
                         // ---- substream_index_table() ----
    bw.write_u32(1, 2); // n_substreams = 1
    bw.write_bit(false); // b_size_present = 0 (spans to frame end)
    bw.align_to_byte();
    let mut frame = bw.into_bytes();
    // ---- ac4_substream(): audio_size envelope + body ----
    let mut body = body;
    body.align_to_byte();
    let body_bytes = body.into_bytes();
    let audio_size = body_bytes.len() as u32;
    if audio_size >= 1 << 15 {
        return Err(Error::invalid("ac4: ice audio_data too large"));
    }
    let mut sw = BitWriter::new();
    sw.write_u32(audio_size, 15);
    sw.write_bit(false); // b_more_bits
    sw.align_to_byte();
    frame.extend_from_slice(&sw.into_bytes());
    frame.extend_from_slice(&body_bytes);
    // Post-audio metadata(…) — starts right at audio_data_offset +
    // audio_size (the substream spans to the frame end).
    if let Some(meta) = metadata {
        frame.extend_from_slice(meta);
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acpl::AcplQuantMode;
    use crate::ajcc::AjccFramingData;
    use crate::ajoc::AjocDiffType;

    const TL: u32 = 1920;
    const MAX_SFB: u32 = 20;

    fn tone_spectrum(bin: usize, amp: f32) -> Vec<f32> {
        let sfbo = crate::sfb_offset::sfb_offset_48(TL).unwrap();
        let end = sfbo[MAX_SFB as usize] as usize;
        let mut v = vec![0.0f32; end];
        v[bin.min(end - 1)] = amp;
        v
    }

    #[test]
    fn ice_codec_mode_round_trips() {
        for mode in [
            IceCodecMode::Scpl,
            IceCodecMode::AspxScpl,
            IceCodecMode::AspxAcpl1,
            IceCodecMode::AspxAcpl2,
            IceCodecMode::AspxAjcc,
        ] {
            let mut bw = BitWriter::new();
            mode.write(&mut bw);
            bw.write_u32(0, 8);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(IceCodecMode::parse(&mut br).unwrap(), mode);
        }
    }

    /// SCPL body (grouping 3, identity SAP) round-trips through the
    /// walker: 11 tracks A..K populated with the emitted tone bins.
    #[test]
    fn ice_scpl_body_round_trips_7_0_4() {
        let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
        let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
        let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
        let hi = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
        let jk = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
        let scpl_pairs: [[&[f32]; 2]; 2] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]]];
        let spectra = IceScplSpectra {
            core: &core_refs,
            add_pair: [&fg[0], &fg[1]],
            scpl_pairs: &scpl_pairs,
        };
        let mut bw = BitWriter::new();
        write_ice_body_scpl(&mut bw, &spectra, None, false, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 16); // padding
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_ice_audio_data_outer(&mut br, &mut tools, false, false, true, TL).unwrap();
        let ice = tools.ice.as_deref().expect("ice element parsed");
        assert_eq!(ice.mode, IceCodecMode::Scpl);
        assert_eq!(ice.core_5ch_grouping, 3);
        assert_eq!(ice.b_use_sap_add_ch, Some(false));
        assert!(ice.add_pair.is_some());
        assert_eq!(ice.scpl_pairs.len(), 2);
        assert_eq!(ice.scpl_chparam.len(), 4);
        assert!(ice.aspx_elements.is_empty());
        assert!(ice.ajcc.is_none());
        assert!(ice.acpl_data.is_empty());
        assert_eq!(ice.transform_length(), Some(TL));
        let tracks = ice.track_spectra();
        // Tracks A..K (0..=10) decoded; L, M (11, 12) absent.
        for (i, t) in tracks.iter().enumerate().take(11) {
            let spec = t.unwrap_or_else(|| panic!("track {i} missing"));
            assert!(
                spec.iter().any(|&v| v.abs() > 1.0),
                "track {i} unexpectedly silent"
            );
        }
        assert!(tracks[11].is_none() && tracks[12].is_none());
        // Track A carries the bin-10 tone (quantised, so approximate).
        let a = tracks[0].unwrap();
        let peak = a
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
            .unwrap()
            .0;
        assert_eq!(peak, 10, "track A peak bin");
    }

    /// SCPL body with b_5fronts + LFE: 13 tracks + decoded LFE.
    #[test]
    fn ice_scpl_body_round_trips_9_1_4() {
        let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
        let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
        let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
        let p0 = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
        let p1 = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
        let p2 = [tone_spectrum(70, 20.0), tone_spectrum(74, 20.0)];
        let scpl_pairs: [[&[f32]; 2]; 3] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]], [&p2[0], &p2[1]]];
        let spectra = IceScplSpectra {
            core: &core_refs,
            add_pair: [&fg[0], &fg[1]],
            scpl_pairs: &scpl_pairs,
        };
        let lfe = tone_spectrum(2, 15.0);
        let mut bw = BitWriter::new();
        write_ice_body_scpl(&mut bw, &spectra, Some((&lfe, 4)), true, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 16);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_ice_audio_data_outer(&mut br, &mut tools, true, true, true, TL).unwrap();
        let ice = tools.ice.as_deref().expect("ice element parsed");
        assert_eq!(ice.scpl_pairs.len(), 3);
        assert_eq!(ice.scpl_chparam.len(), 6);
        assert!(ice.lfe_spectrum().is_some(), "LFE body decoded");
        let tracks = ice.track_spectra();
        for (i, t) in tracks.iter().enumerate() {
            assert!(t.is_some(), "track {i} missing");
        }
    }

    fn minimal_ajcc_data(b_5fronts: bool, iframe: bool) -> AjccData {
        let nb = 9usize;
        let fine = AcplQuantMode::Fine;
        let freq_row = |q: i32| {
            vec![(
                AjocDiffType::Freq,
                crate::ajcc::encode_ajcc_deltas_freq(&vec![q; nb]),
            )]
        };
        let time_row = || vec![(AjocDiffType::Time, vec![0i32; nb])];
        let row = |q: i32| {
            if iframe {
                freq_row(q)
            } else {
                time_row()
            }
        };
        let framing = vec![
            AjccFramingData {
                steep: false,
                num_param_sets: 1,
                param_timeslot: vec![],
            };
            if b_5fronts { 4 } else { 2 }
        ];
        AjccData {
            b_5fronts,
            b_no_dt: iframe,
            num_param_bands_id: 2,
            num_bands: nb as u32,
            core_mode: false,
            qm_f: fine,
            qm_b: fine,
            qm_ab: fine,
            qm_dw: fine,
            framing,
            alpha: if b_5fronts {
                vec![]
            } else {
                vec![row(16), row(16)]
            },
            beta: if b_5fronts {
                vec![]
            } else {
                vec![row(0), row(0)]
            },
            dry: vec![row(8); if b_5fronts { 8 } else { 4 }],
            wet: vec![row(20); if b_5fronts { 12 } else { 6 }],
        }
    }

    fn test_aspx_config() -> AspxConfig {
        use crate::aspx::*;
        AspxConfig {
            quant_mode_env: AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: AspxFreqResMode::Low,
        }
    }

    /// AJCC body round-trip: companding + core + 3 A-SPX payloads +
    /// `ajcc_data()` all parse back.
    #[test]
    fn ice_ajcc_body_round_trips() {
        let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(8 + 3 * i, 30.0)).collect();
        let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
        let ajcc = minimal_ajcc_data(false, true);
        let cfg = test_aspx_config();
        let mut bw = BitWriter::new();
        write_ice_body_ajcc(&mut bw, &core_refs, &ajcc, &cfg, None, true, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 16);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_ice_audio_data_outer(&mut br, &mut tools, false, false, true, TL).unwrap();
        let ice = tools.ice.as_deref().expect("ice element parsed");
        assert_eq!(ice.mode, IceCodecMode::AspxAjcc);
        assert!(ice.companding.is_some());
        assert_eq!(ice.aspx_elements.len(), 3);
        assert!(matches!(
            &ice.aspx_elements[0],
            IceAspxElement::TwoCh(Some(_))
        ));
        assert!(matches!(
            &ice.aspx_elements[1],
            IceAspxElement::TwoCh(Some(_))
        ));
        assert!(matches!(
            &ice.aspx_elements[2],
            IceAspxElement::OneCh(Some(_))
        ));
        let got = ice.ajcc.as_deref().expect("ajcc data parsed");
        assert_eq!(got, &ajcc);
        // 7CH_STATIC extras absent for the 5CH_DYNAMIC core.
        assert!(ice.b_use_sap_add_ch.is_none());
        assert!(ice.add_pair.is_none());
        assert!(ice.scpl_pairs.is_empty());
        // Core tracks A..E decoded; F..M silent.
        let tracks = ice.track_spectra();
        for t in tracks.iter().take(5) {
            assert!(t.is_some());
        }
        for t in tracks.iter().skip(5) {
            assert!(t.is_none());
        }
        // The I-frame harvested the first payload's xover for sticky.
        assert_eq!(tools.aspx_xover_subband_offset, Some(0));
    }

    /// P-frame AJCC body parses with the sticky aspx_config + xover
    /// pre-seeded (configs omitted from the bitstream per §6.2.4.1 /
    /// Tables 51-52).
    #[test]
    fn ice_ajcc_p_frame_uses_sticky_config() {
        let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(8 + 3 * i, 30.0)).collect();
        let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
        let ajcc = minimal_ajcc_data(false, false);
        let cfg = test_aspx_config();
        let mut bw = BitWriter::new();
        write_ice_body_ajcc(&mut bw, &core_refs, &ajcc, &cfg, None, false, TL, MAX_SFB).unwrap();
        bw.write_u32(0, 16);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools {
            aspx_config: Some(cfg),
            aspx_xover_subband_offset: Some(0),
            ..Default::default()
        };
        parse_ice_audio_data_outer(&mut br, &mut tools, false, false, false, TL).unwrap();
        let ice = tools.ice.as_deref().expect("ice element parsed");
        assert_eq!(ice.aspx_elements.len(), 3);
        assert_eq!(ice.ajcc.as_deref(), Some(&ajcc));
    }

    /// Grouping 0 (1+2+2) core: hand-emitted body exercises the
    /// 2ch_mode-dependent Table 19 track assignment.
    #[test]
    fn ice_grouping0_track_assignment_follows_2ch_mode() {
        for b_2ch_mode in [false, true] {
            let s0 = tone_spectrum(10, 30.0);
            let s1 = tone_spectrum(14, 30.0);
            let s2 = tone_spectrum(18, 30.0);
            let s3 = tone_spectrum(22, 30.0);
            let c = tone_spectrum(26, 30.0);
            let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
            let hi = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
            let jk = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
            let mut bw = BitWriter::new();
            IceCodecMode::Scpl.write(&mut bw);
            bw.write_u32(0, 2); // core_5ch_grouping = 0
            bw.write_bit(b_2ch_mode);
            crate::ajoc_substream::write_two_channel_data_simple(&mut bw, &s0, &s1, TL, MAX_SFB)
                .unwrap();
            crate::ajoc_substream::write_two_channel_data_simple(&mut bw, &s2, &s3, TL, MAX_SFB)
                .unwrap();
            crate::ajoc_substream::write_mono_data_simple(&mut bw, &c, TL, MAX_SFB).unwrap();
            bw.write_bit(false); // b_use_sap_add_ch = 0
            crate::ajoc_substream::write_two_channel_data_simple(
                &mut bw, &fg[0], &fg[1], TL, MAX_SFB,
            )
            .unwrap();
            crate::ajoc_substream::write_two_channel_data_simple(
                &mut bw, &hi[0], &hi[1], TL, MAX_SFB,
            )
            .unwrap();
            crate::ajoc_substream::write_two_channel_data_simple(
                &mut bw, &jk[0], &jk[1], TL, MAX_SFB,
            )
            .unwrap();
            for _ in 0..4 {
                bw.write_u32(0, 2); // chparam sap_mode = 0
            }
            bw.write_u32(0, 16);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let mut tools = SubstreamTools::default();
            parse_ice_audio_data_outer(&mut br, &mut tools, false, false, true, TL).unwrap();
            let ice = tools.ice.as_deref().unwrap();
            assert_eq!(ice.core_5ch_grouping, 0);
            let tracks = ice.track_spectra();
            let peak = |t: Option<&[f32]>| {
                t.unwrap()
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
                    .unwrap()
                    .0
            };
            // Table 19: 2ch_mode 0 → [A,B]+[D,E]; 1 → [A,D]+[B,E].
            assert_eq!(peak(tracks[0]), 10, "A");
            assert_eq!(peak(tracks[2]), 26, "C");
            if b_2ch_mode {
                assert_eq!(peak(tracks[3]), 14, "D (2ch_mode=1)");
                assert_eq!(peak(tracks[1]), 18, "B (2ch_mode=1)");
                assert_eq!(peak(tracks[4]), 22, "E");
            } else {
                assert_eq!(peak(tracks[1]), 14, "B (2ch_mode=0)");
                assert_eq!(peak(tracks[3]), 18, "D (2ch_mode=0)");
                assert_eq!(peak(tracks[4]), 22, "E");
            }
            assert_eq!(peak(tracks[5]), 40, "F");
            assert_eq!(peak(tracks[7]), 50, "H");
        }
    }

    /// Grouping 2 (1+4) core parse: four_channel_data → [A, B, D, E].
    #[test]
    fn ice_grouping2_track_assignment() {
        let s: Vec<Vec<f32>> = (0..4).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
        let c = tone_spectrum(30, 30.0);
        let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
        let hi = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
        let jk = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
        let mut bw = BitWriter::new();
        IceCodecMode::Scpl.write(&mut bw);
        bw.write_u32(2, 2); // core_5ch_grouping = 2
                            // four_channel_data(): sf_info + four_channel_info + 4 bodies.
        let sfbo = crate::sfb_offset::sfb_offset_48(TL).unwrap();
        let (n_msfb_bits, _, _) = crate::tables::n_msfb_bits_48(TL).unwrap();
        bw.write_bit(true); // b_long_frame
        bw.write_u32(MAX_SFB, n_msfb_bits);
        for _ in 0..4 {
            bw.write_u32(0, 2); // four_channel_info: 4× chparam(sap 0)
        }
        for spec in &s {
            crate::ajoc_substream::write_asf_body_from_spectrum(&mut bw, spec, sfbo, MAX_SFB);
        }
        crate::ajoc_substream::write_mono_data_simple(&mut bw, &c, TL, MAX_SFB).unwrap();
        bw.write_bit(false); // b_use_sap_add_ch = 0
        crate::ajoc_substream::write_two_channel_data_simple(&mut bw, &fg[0], &fg[1], TL, MAX_SFB)
            .unwrap();
        crate::ajoc_substream::write_two_channel_data_simple(&mut bw, &hi[0], &hi[1], TL, MAX_SFB)
            .unwrap();
        crate::ajoc_substream::write_two_channel_data_simple(&mut bw, &jk[0], &jk[1], TL, MAX_SFB)
            .unwrap();
        for _ in 0..4 {
            bw.write_u32(0, 2);
        }
        bw.write_u32(0, 16);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_ice_audio_data_outer(&mut br, &mut tools, false, false, true, TL).unwrap();
        let ice = tools.ice.as_deref().unwrap();
        assert_eq!(ice.core_5ch_grouping, 2);
        let tracks = ice.track_spectra();
        let peak = |t: Option<&[f32]>| {
            t.unwrap()
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
                .unwrap()
                .0
        };
        // Table 19 grouping 2: four_channel_data → [A, B, D, E].
        assert_eq!(peak(tracks[0]), 10, "A");
        assert_eq!(peak(tracks[1]), 14, "B");
        assert_eq!(peak(tracks[3]), 18, "D");
        assert_eq!(peak(tracks[4]), 22, "E");
        assert_eq!(peak(tracks[2]), 30, "C");
    }

    /// §5.2.3.2 step 4: a `b_use_sap_add_ch` chparam pair with
    /// `sap_mode == 2` (M/S in all bands) mixes (D, F) / (E, G) into
    /// sum/difference tracks; step-6 identity elements leave the S-CPL
    /// tracks alone.
    #[test]
    fn ice_sap_step4_msall_mixes_d_f_and_e_g() {
        let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
        let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
        // F carries a distinct tone; G is silent.
        let f = tone_spectrum(40, 24.0);
        let g = vec![0.0f32; f.len()];
        let hi = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
        let jk = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
        let scpl_pairs: [[&[f32]; 2]; 2] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]]];
        let spectra = IceScplSpectra {
            core: &core_refs,
            add_pair: [&f, &g],
            scpl_pairs: &scpl_pairs,
        };
        let msall = ChparamInfo {
            sap_mode: 2,
            ..ChparamInfo::default()
        };
        let pair = [msall.clone(), msall];
        let mut bw = BitWriter::new();
        write_ice_body_scpl_with_sap(
            &mut bw,
            &spectra,
            None,
            false,
            TL,
            MAX_SFB,
            Some(&pair),
            &[],
        )
        .unwrap();
        bw.write_u32(0, 16);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_ice_audio_data_outer(&mut br, &mut tools, false, false, true, TL).unwrap();
        let ice = tools.ice.as_deref().unwrap();
        assert_eq!(ice.b_use_sap_add_ch, Some(true));
        assert!(ice.sap_add_chparam.is_some());
        let mut specs = ice.track_spectra_owned();
        let d_before = specs[3].clone();
        let f_before = specs[5].clone();
        apply_sap_steps(&mut specs, ice, TL);
        // D' = D + F: both the bin-22 core-D tone and the bin-40 F
        // tone appear on track D; F' = D − F: the F tone flips sign.
        let d_tone = d_before[22];
        let f_tone = f_before[40];
        assert!(d_tone > 20.0 && f_tone > 20.0, "fixture tones decoded");
        assert!((specs[3][22] - d_tone).abs() < 1.0, "D' keeps the D tone");
        assert!((specs[3][40] - f_tone).abs() < 1.0, "D' gains the F tone");
        assert!((specs[5][22] - d_tone).abs() < 1.0, "F' keeps the D tone");
        assert!((specs[5][40] + f_tone).abs() < 1.0, "F' negates the F tone");
        // Second quartet mixes (E, G): G silent → E' == E, G' == E.
        assert!((specs[4][26] - 30.0).abs() < 3.0, "E' keeps the E tone");
        assert!((specs[6][26] - specs[4][26]).abs() < 1.0, "G' = E - G = E");
        // A, B, C and the S-CPL tracks are untouched by step 4.
        assert!(specs[7][50] > 15.0 && specs[7].get(22).copied().unwrap_or(0.0) == 0.0);
    }

    /// §5.2.3.2 steps 5-6: a full-SAP (`sap_mode == 3`) S-CPL-section
    /// chparam element resolves per-band `a'_0` gains and Table 20
    /// adds `a'_0 · D'` into track H.
    #[test]
    fn ice_sap_step6_full_sap_adds_carrier_into_scpl_mid() {
        let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
        let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
        let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
        let hi = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
        let jk = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
        let scpl_pairs: [[&[f32]; 2]; 2] = [[&hi[0], &hi[1]], [&jk[0], &jk[1]]];
        let spectra = IceScplSpectra {
            core: &core_refs,
            add_pair: [&fg[0], &fg[1]],
            scpl_pairs: &scpl_pairs,
        };
        // alpha_q = 10 on every band -> sap_gain = 1,0 -> a'_0 = 1,0.
        let m = MAX_SFB as usize;
        let mut dpcm = vec![0i32; m];
        dpcm[0] = 10;
        let full_sap = ChparamInfo {
            sap_mode: 3,
            sap_data: Some(crate::asf::SapData {
                sap_coeff_all: true,
                sap_coeff_used: vec![vec![true; m]],
                delta_code_time: false,
                dpcm_alpha_q: vec![dpcm],
            }),
            ..ChparamInfo::default()
        };
        let identity = ChparamInfo::default();
        let chparams = [
            full_sap,
            identity.clone(),
            identity.clone(),
            identity.clone(),
        ];
        let mut bw = BitWriter::new();
        write_ice_body_scpl_with_sap(&mut bw, &spectra, None, false, TL, MAX_SFB, None, &chparams)
            .unwrap();
        bw.write_u32(0, 16);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_ice_audio_data_outer(&mut br, &mut tools, false, false, true, TL).unwrap();
        let ice = tools.ice.as_deref().unwrap();
        assert_eq!(ice.scpl_chparam.len(), 4);
        let mut specs = ice.track_spectra_owned();
        let d_tone = specs[3][22];
        let h_tone = specs[7][50];
        assert!(d_tone > 20.0 && h_tone > 15.0, "fixture tones decoded");
        apply_sap_steps(&mut specs, ice, TL);
        // H'' = H' + 1,0 · D' — the D tone lands on H, the H tone stays.
        assert!((specs[7][22] - d_tone).abs() < 1.0, "H'' gains a'0 * D'");
        assert!((specs[7][50] - h_tone).abs() < 1.0, "H'' keeps its tone");
        // I (a'_1 element is identity-mode) is untouched.
        assert_eq!(specs[8].get(26).copied().unwrap_or(0.0), 0.0);
        // D itself is unmodified by step 6.
        assert!((specs[3][22] - d_tone).abs() < 1.0);
    }

    /// ASPX_SCPL body round-trip: aspx_config on the sticky slot, the
    /// full 6- / 7-element payload roster, and the S-CPL section.
    #[test]
    fn ice_aspx_scpl_body_round_trips() {
        for b_5fronts in [false, true] {
            let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
            let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
            let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
            let p0 = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
            let p1 = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
            let p2 = [tone_spectrum(70, 20.0), tone_spectrum(74, 20.0)];
            let pairs_5f: [[&[f32]; 2]; 3] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]], [&p2[0], &p2[1]]];
            let pairs_4f: [[&[f32]; 2]; 2] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]]];
            let spectra = IceScplSpectra {
                core: &core_refs,
                add_pair: [&fg[0], &fg[1]],
                scpl_pairs: if b_5fronts { &pairs_5f } else { &pairs_4f },
            };
            let cfg = test_aspx_config();
            let mut bw = BitWriter::new();
            write_ice_body_aspx_scpl(&mut bw, &spectra, None, b_5fronts, &cfg, true, TL, MAX_SFB)
                .unwrap();
            bw.write_u32(0, 16);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let mut tools = SubstreamTools::default();
            parse_ice_audio_data_outer(&mut br, &mut tools, false, b_5fronts, true, TL).unwrap();
            let ice = tools.ice.as_deref().expect("ice element parsed");
            assert_eq!(ice.mode, IceCodecMode::AspxScpl);
            assert!(tools.aspx_config.is_some(), "sticky aspx_config landed");
            let want = if b_5fronts { 7 } else { 6 };
            assert_eq!(ice.aspx_elements.len(), want);
            // Roster shape: 2ch, 2ch, 1ch, then 2ch to the end.
            for (i, el) in ice.aspx_elements.iter().enumerate() {
                match (i, el) {
                    (2, IceAspxElement::OneCh(Some(_))) => {}
                    (2, _) => panic!("payload 2 must be the 1ch element"),
                    (_, IceAspxElement::TwoCh(Some(_))) => {}
                    _ => panic!("payload {i} must be a captured 2ch element"),
                }
            }
            assert_eq!(ice.scpl_pairs.len(), if b_5fronts { 3 } else { 2 });
            assert_eq!(ice.scpl_chparam.len(), if b_5fronts { 6 } else { 4 });
            assert!(ice.ajcc.is_none());
            assert!(ice.acpl_data.is_empty());
        }
    }

    /// ASPX_ACPL_1 / ASPX_ACPL_2 body round-trips: configs land on the
    /// sticky slots, the 4-payload roster parses, the S-CPL section is
    /// present exactly for ACPL_1, and the acpl_data section carries
    /// one element per module.
    #[test]
    fn ice_acpl_bodies_round_trip() {
        for (mode, b_5fronts) in [
            (IceCodecMode::AspxAcpl1, false),
            (IceCodecMode::AspxAcpl1, true),
            (IceCodecMode::AspxAcpl2, false),
            (IceCodecMode::AspxAcpl2, true),
        ] {
            let is_acpl1 = matches!(mode, IceCodecMode::AspxAcpl1);
            let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(10 + 4 * i, 30.0)).collect();
            let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
            let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
            let p0 = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
            let p1 = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
            let p2 = [tone_spectrum(70, 20.0), tone_spectrum(74, 20.0)];
            let pairs_5f: [[&[f32]; 2]; 3] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]], [&p2[0], &p2[1]]];
            let pairs_4f: [[&[f32]; 2]; 2] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]]];
            let scpl_pairs: &[[&[f32]; 2]] = if !is_acpl1 {
                &[]
            } else if b_5fronts {
                &pairs_5f
            } else {
                &pairs_4f
            };
            let n_modules = if b_5fronts { 6 } else { 4 };
            let num_bands = crate::acpl::num_param_bands_from_id(2) as usize;
            let alpha = vec![4i32; num_bands];
            let beta = vec![0i32; num_bands];
            let modules: Vec<(&[i32], &[i32])> =
                vec![(alpha.as_slice(), beta.as_slice()); n_modules];
            let acpl = IceAcplParams {
                num_param_bands_id: 2,
                quant_mode: AcplQuantMode::Fine,
                qmf_band: 8,
                modules: &modules,
            };
            let cfg = test_aspx_config();
            let mut bw = BitWriter::new();
            write_ice_body_acpl(
                &mut bw,
                mode,
                &core_refs,
                [&fg[0], &fg[1]],
                scpl_pairs,
                &acpl,
                &cfg,
                None,
                b_5fronts,
                true,
                TL,
                MAX_SFB,
            )
            .unwrap();
            bw.write_u32(0, 16);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let mut tools = SubstreamTools::default();
            parse_ice_audio_data_outer(&mut br, &mut tools, false, b_5fronts, true, TL).unwrap();
            let ice = tools.ice.as_deref().expect("ice element parsed");
            assert_eq!(ice.mode, mode);
            assert!(tools.aspx_config.is_some());
            if is_acpl1 {
                assert!(tools.acpl_config_1ch_partial.is_some());
                assert_eq!(tools.acpl_config_1ch_partial.unwrap().qmf_band, 8);
                assert_eq!(ice.scpl_pairs.len(), if b_5fronts { 3 } else { 2 });
            } else {
                assert!(tools.acpl_config_1ch_full.is_some());
                assert!(ice.scpl_pairs.is_empty());
            }
            assert_eq!(ice.aspx_elements.len(), 4);
            assert!(matches!(
                &ice.aspx_elements[3],
                IceAspxElement::OneCh(Some(_))
            ));
            assert_eq!(ice.acpl_data.len(), n_modules);
        }
    }

    /// 22_2_channel_element bodies round-trip in both Table 98 codec
    /// modes: two LFE bodies, eleven pairs, and (A-SPX) one captured
    /// payload per pair with the config on the sticky slot.
    #[test]
    fn el_22_2_bodies_round_trip() {
        for aspx in [false, true] {
            let pairs_spec: Vec<[Vec<f32>; 2]> = (0..11)
                .map(|p| {
                    [
                        tone_spectrum(10 + 6 * p, 30.0),
                        tone_spectrum(12 + 6 * p, 30.0),
                    ]
                })
                .collect();
            let pairs: [[&[f32]; 2]; 11] =
                std::array::from_fn(|p| [pairs_spec[p][0].as_slice(), pairs_spec[p][1].as_slice()]);
            let lfe0 = tone_spectrum(2, 15.0);
            let lfe1 = tone_spectrum(3, 15.0);
            let cfg = test_aspx_config();
            let mut bw = BitWriter::new();
            write_22_2_body(
                &mut bw,
                [(&lfe0, 4), (&lfe1, 4)],
                &pairs,
                aspx.then_some(&cfg),
                true,
                TL,
                MAX_SFB,
            )
            .unwrap();
            bw.write_u32(0, 16);
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let mut tools = SubstreamTools::default();
            parse_22_2_audio_data_outer(&mut br, &mut tools, true, TL).unwrap();
            let el = tools.el_22_2.as_deref().expect("22.2 element parsed");
            assert_eq!(el.aspx_mode, aspx);
            assert_eq!(el.pairs.len(), 11);
            assert_eq!(el.transform_length(), Some(TL));
            assert!(el.lfe[0].scaled_spec.is_some() && el.lfe[1].scaled_spec.is_some());
            let full = el.fullband_spectra();
            assert_eq!(full.len(), 22);
            for (ch, s) in full.iter().enumerate() {
                let s = s.unwrap_or_else(|| panic!("channel {ch} missing"));
                assert!(s.iter().any(|&v| v.abs() > 1.0), "channel {ch} silent");
            }
            if aspx {
                assert_eq!(el.aspx_elements.len(), 11);
                assert!(el.aspx_elements.iter().all(|t| t.is_some()));
                assert!(tools.aspx_config.is_some());
                assert_eq!(tools.aspx_xover_subband_offset, Some(0));
            } else {
                assert!(el.aspx_elements.is_empty());
            }
        }
    }

    /// The full-frame writer emits a TOC the parser identifies as an
    /// immersive channel substream for all four channel modes.
    #[test]
    fn ice_raw_frame_toc_signals_immersive_mode() {
        for (b_lfe, b_5fronts, want_mode, want_ch) in [
            (false, false, 11u32, 11u32),
            (true, false, 12, 12),
            (false, true, 13, 13),
            (true, true, 14, 14),
        ] {
            let core: Vec<Vec<f32>> = (0..5).map(|i| tone_spectrum(8 + 3 * i, 30.0)).collect();
            let core_refs: [&[f32]; 5] = [&core[0], &core[1], &core[2], &core[3], &core[4]];
            let fg = [tone_spectrum(40, 25.0), tone_spectrum(44, 25.0)];
            let p0 = [tone_spectrum(50, 20.0), tone_spectrum(54, 20.0)];
            let p1 = [tone_spectrum(60, 20.0), tone_spectrum(64, 20.0)];
            let p2 = [tone_spectrum(70, 20.0), tone_spectrum(74, 20.0)];
            let pairs_5f: [[&[f32]; 2]; 3] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]], [&p2[0], &p2[1]]];
            let pairs_4f: [[&[f32]; 2]; 2] = [[&p0[0], &p0[1]], [&p1[0], &p1[1]]];
            let spectra = IceScplSpectra {
                core: &core_refs,
                add_pair: [&fg[0], &fg[1]],
                scpl_pairs: if b_5fronts { &pairs_5f } else { &pairs_4f },
            };
            let lfe_spec = tone_spectrum(2, 15.0);
            let lfe = if b_lfe {
                Some((&lfe_spec[..], 4u32))
            } else {
                None
            };
            let mut body = BitWriter::new();
            write_ice_body_scpl(&mut body, &spectra, lfe, b_5fronts, TL, MAX_SFB).unwrap();
            let frame = encode_ice_raw_frame(7, b_lfe, b_5fronts, true, body).unwrap();
            let info = crate::toc::parse_ac4_toc(&frame).unwrap();
            let mode = info.first_chan_mode.expect("chan mode surfaced");
            assert_eq!(mode.ch_mode, want_mode);
            assert_eq!(mode.channels, want_ch);
            assert!(mode.is_immersive());
            assert_eq!(mode.immersive_b_lfe(), b_lfe);
            assert_eq!(mode.immersive_b_5fronts(), b_5fronts);
            let pres = info.immersive_presence.expect("presence fields");
            assert!(pres.b_4_back_channels_present && pres.b_centre_present);
            assert_eq!(pres.top_channels_present, 0);
            assert_eq!(info.channels, want_ch as u16);
        }
    }
}
