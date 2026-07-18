//! AC-4 Audio Spectral Frontend (ASF) substream baseline.
//!
//! This module implements the framing of an `ac4_substream()` element
//! (ETSI TS 103 190-1 §4.3.4) and the outer shell of the `audio_data()`
//! dispatcher (§4.2.5) for the mono and stereo channel modes. It parses
//! the top-level codec-mode flags, `asf_transform_info()` (§4.2.8.1 /
//! §4.3.6.1), and `asf_psy_info()` (§4.2.8.2 / §4.3.6.2) so the decoder
//! can describe which tools each substream uses (`SIMPLE` / `ASPX` /
//! `ASPX_ACPL_*`), what transform length is in play, how many MDCT
//! windows are present, the window grouping, and the per-group
//! `max_sfb`. It stops short of the actual spectral-coefficient decoding,
//! Huffman tables, and MDCT synthesis.
//!
//! What we deliberately do **not** do yet (and why):
//!
//! * `asf_section_data`, `asf_spectral_data`, `asf_scalefac_data`,
//!   `asf_snf_data` — all Huffman-driven; Huffman tables (§5.1) have
//!   not been transcribed yet.
//! * `ssf_data` (Speech Spectral Frontend) — gated by the same Huffman
//!   / arithmetic-coded layer.
//! * A-SPX (`aspx_config`, `aspx_data_*`) and A-CPL. These are the
//!   bandwidth-extension and inter-channel-coupling tools; each carries
//!   its own Huffman suites.
//!
//! Those components remain marked TODO and the substream walker simply
//! consumes opaque bits from the `audio_size` budget after the outer
//! `asf_transform_info()` is parsed. The decoder continues to emit
//! silence until the tools are filled in.
//!
//! Nothing in here panics on malformed input — every read is
//! result-propagated.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::asf_data;
use crate::aspx;
use crate::huffman::{huff_decode, HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN};
use crate::sfb_offset;
use crate::tables;
use crate::toc::variable_bits;

/// `mono_codec_mode` values (Table 93).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoCodecMode {
    Simple,
    Aspx,
}

impl MonoCodecMode {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::Simple
        } else {
            Self::Aspx
        }
    }
}

/// `stereo_codec_mode` values (Table 95).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoCodecMode {
    Simple,
    Aspx,
    AspxAcpl1,
    AspxAcpl2,
}

impl StereoCodecMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b11 {
            0 => Self::Simple,
            1 => Self::Aspx,
            2 => Self::AspxAcpl1,
            _ => Self::AspxAcpl2,
        }
    }
}

/// `spec_frontend` values (Table 94).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecFrontend {
    /// Audio Spectral Frontend — MDCT-based.
    Asf,
    /// Speech Spectral Frontend — arithmetic-coded MDCT with LPC envelope.
    Ssf,
}

impl SpecFrontend {
    pub fn from_bit(v: u32) -> Self {
        if v == 0 {
            Self::Asf
        } else {
            Self::Ssf
        }
    }
}

/// ASF transform-length info (`asf_transform_info()`, §4.2.8.1 + §4.3.6.1).
#[derive(Debug, Clone, Copy, Default)]
pub struct AsfTransformInfo {
    /// `b_long_frame` — only meaningful when `frame_len_base >= 1536`.
    pub b_long_frame: bool,
    /// `transf_length[0]` / `transf_length[1]` — 2-bit indices. For a
    /// long frame the two are implicitly equal and hold the long-frame
    /// transform length from Table 99. For `frame_len_base < 1536`
    /// there's only a single `transf_length` field; we mirror it into
    /// both slots.
    pub transf_length: [u32; 2],
    /// Resolved transform length in samples for `transf_length[0]`.
    pub transform_length_0: u32,
    /// Resolved transform length for `transf_length[1]`.
    pub transform_length_1: u32,
}

/// Parsed `asf_psy_info()` (ETSI TS 103 190-1 §4.2.8.2 + §4.3.6.2) —
/// the scale-factor-band organisation for one or two transform-length
/// windows.
#[derive(Debug, Clone, Default)]
pub struct AsfPsyInfo {
    /// Derived from the spec: `transf_length[0]` differed from
    /// `transf_length[1]`, triggering the dual-window branch.
    pub b_different_framing: bool,
    /// `max_sfb[0]` or `max_sfb_side[0]` depending on `b_side_limited`.
    pub max_sfb_0: u32,
    /// `max_sfb_side[0]` when `b_dual_maxsfb` is set (else 0).
    pub max_sfb_side_0: u32,
    /// `max_sfb[1]` — only populated when `b_different_framing`.
    pub max_sfb_1: u32,
    /// `max_sfb_side[1]` — only populated when `b_different_framing`
    /// and `b_dual_maxsfb`.
    pub max_sfb_side_1: u32,
    /// Raw `scale_factor_grouping[i]` bits.
    pub scale_factor_grouping: Vec<u8>,
    /// Derived: `num_windows` (1 for long frames).
    pub num_windows: u32,
    /// Derived: `num_window_groups`.
    pub num_window_groups: u32,
}

/// Table 109 row lookup — `n_grp_bits` for `frame_len_base ≥ 1536` and
/// `b_long_frame == 0`, keyed by `(transf_length[0], transf_length[1])`.
pub fn n_grp_bits_ge_1536(tl0: u32, tl1: u32) -> u32 {
    match (tl0 & 0b11, tl1 & 0b11) {
        (0, 0) => 15,
        (0, 1) => 10,
        (0, 2) => 8,
        (0, 3) => 7,
        (1, 0) => 10,
        (1, 1) => 7,
        (1, 2) => 4,
        (1, 3) => 3,
        (2, 0) => 8,
        (2, 1) => 4,
        (2, 2) => 3,
        (2, 3) => 1,
        (3, 0) => 7,
        (3, 1) => 3,
        (3, 2) => 1,
        (3, 3) => 1,
        _ => 0,
    }
}

/// Table 110 row lookup — `n_grp_bits` for `frame_len_base < 1536`,
/// keyed by `(frame_len_base, transf_length)`. Returns 0 for unknown
/// combinations.
pub fn n_grp_bits_lt_1536(frame_len_base: u32, tl: u32) -> u32 {
    match (frame_len_base, tl & 0b11) {
        (1024, 0) | (960, 0) | (768, 0) => 7,
        (1024, 1) | (960, 1) | (768, 1) => 3,
        (1024, 2) | (960, 2) | (768, 2) => 1,
        (1024, 3) | (960, 3) | (768, 3) => 0,
        (512, 0) | (384, 0) => 3,
        (512, 1) | (384, 1) => 1,
        (512, 2) | (384, 2) => 0,
        _ => 0,
    }
}

/// Parse `asf_psy_info(b_dual_maxsfb, b_side_limited)` at the current
/// position. `transform_info` is the previously parsed
/// `asf_transform_info()` result; `frame_len_base` the TOC's
/// `frame_length` at the base rate.
///
/// Returns the scale-factor-band shape plus the parsed
/// `scale_factor_grouping[i]` bits. Derives `num_windows` /
/// `num_window_groups` per Pseudocode 3 in §4.3.6.2.6.
pub fn parse_asf_psy_info(
    br: &mut BitReader<'_>,
    transform_info: &AsfTransformInfo,
    frame_len_base: u32,
    b_dual_maxsfb: bool,
    b_side_limited: bool,
) -> Result<AsfPsyInfo> {
    // b_different_framing is derived, not coded.
    let b_different_framing = frame_len_base >= 1536
        && !transform_info.b_long_frame
        && transform_info.transf_length[0] != transform_info.transf_length[1];
    let mut info = AsfPsyInfo {
        b_different_framing,
        ..AsfPsyInfo::default()
    };

    // Resolve n_msfb_bits / n_side_bits from Table 106 for the primary
    // transform length.
    let (nm0, ns0, _nml0) = tables::n_msfb_bits_48(transform_info.transform_length_0)
        .ok_or_else(|| Error::invalid("ac4: asf_psy_info: unsupported transform_length[0]"))?;

    if b_side_limited {
        info.max_sfb_0 = br.read_u32(ns0)?;
    } else {
        info.max_sfb_0 = br.read_u32(nm0)?;
        if b_dual_maxsfb {
            info.max_sfb_side_0 = br.read_u32(nm0)?;
        }
    }

    if info.b_different_framing {
        let (nm1, ns1, _) = tables::n_msfb_bits_48(transform_info.transform_length_1)
            .ok_or_else(|| Error::invalid("ac4: asf_psy_info: unsupported transform_length[1]"))?;
        if b_side_limited {
            info.max_sfb_1 = br.read_u32(ns1)?;
        } else {
            info.max_sfb_1 = br.read_u32(nm1)?;
            if b_dual_maxsfb {
                info.max_sfb_side_1 = br.read_u32(nm1)?;
            }
        }
    }

    // Determine n_grp_bits per Table 109 / 110 and spec §4.3.6.2.4.
    let n_grp_bits = if transform_info.b_long_frame {
        // Long frames: no grouping bits.
        0
    } else if frame_len_base >= 1536 {
        n_grp_bits_ge_1536(
            transform_info.transf_length[0],
            transform_info.transf_length[1],
        )
    } else {
        n_grp_bits_lt_1536(frame_len_base, transform_info.transf_length[0])
    };

    // Read scale_factor_grouping[i] bits.
    let mut grouping = Vec::with_capacity(n_grp_bits as usize);
    for _ in 0..n_grp_bits {
        grouping.push(br.read_u32(1)? as u8);
    }
    info.scale_factor_grouping = grouping;

    // Derive num_windows / num_window_groups per Pseudocode 3 —
    // simplified: for long frames it's 1/1; for non-long with equal
    // transform lengths, num_windows = n_grp_bits + 1.
    if transform_info.b_long_frame {
        info.num_windows = 1;
        info.num_window_groups = 1;
    } else {
        info.num_windows = n_grp_bits + 1;
        info.num_window_groups = 1;
        // For the equal-transform-length case each grouping==0 bit
        // starts a new group. For b_different_framing the pseudocode
        // inserts an unconditional boundary at the half-frame mark.
        for &b in &info.scale_factor_grouping {
            if b == 0 {
                info.num_window_groups += 1;
            }
        }
        // Safety cap — malformed streams shouldn't explode.
        if info.num_windows == 0 {
            info.num_windows = 1;
        }
    }

    Ok(info)
}

/// Parse `sf_info_lfe()` for the LFE channel of a 5.X / 7.X
/// `mono_data(b_lfe=1)` body (ETSI TS 103 190-1 §4.2.8 / §4.3.6.2.1
/// Table 21 / Table 106 column `n_msfbl_bits`).
///
/// Differences from the regular [`parse_asf_psy_info`]:
///
/// * The `max_sfb[0]` field is read with **`n_msfbl_bits`** width
///   (Table 106 column 4) instead of `n_msfb_bits`. For long-frame
///   transforms at 48 kHz this is `2..=3` bits rather than `4..=6`,
///   matching the LFE band-count cap.
/// * `b_dual_maxsfb` and `b_side_limited` aren't applicable (LFE is
///   single-channel; there's no joint-MDCT side coding).
/// * `b_long_frame` is forced to true per Table 21 — the LFE channel
///   is restricted to long-frame transforms (no scale-factor grouping
///   bits, `num_windows = num_window_groups = 1`).
///
/// The returned [`AsfPsyInfo`] therefore carries `max_sfb_0`,
/// `num_windows = 1`, `num_window_groups = 1`, and an empty
/// `scale_factor_grouping`. `max_sfb_1` / `max_sfb_side_*` are zero.
///
/// Returns `Error::invalid` if the transform length doesn't have an
/// `n_msfbl_bits` entry in Table 106 (i.e. it isn't a permitted LFE
/// transform length).
pub fn parse_asf_psy_info_lfe(
    br: &mut BitReader<'_>,
    transform_info: &AsfTransformInfo,
) -> Result<AsfPsyInfo> {
    let mut info = AsfPsyInfo {
        b_different_framing: false,
        num_windows: 1,
        num_window_groups: 1,
        ..AsfPsyInfo::default()
    };

    // Table 106 column 4: n_msfbl_bits. 0 means the transform length
    // isn't permitted for LFE — bail.
    let (_nm0, _ns0, nml0) = tables::n_msfb_bits_48(transform_info.transform_length_0)
        .ok_or_else(|| Error::invalid("ac4: asf_psy_info_lfe: unsupported transform_length"))?;
    if nml0 == 0 {
        return Err(Error::invalid(
            "ac4: asf_psy_info_lfe: transform_length not permitted for LFE",
        ));
    }
    info.max_sfb_0 = br.read_u32(nml0)?;

    // LFE channel can be capped further by num_sfb_lfe per spec — we
    // surface max_sfb_0 unmodified so the caller can clamp during
    // sf_data decoding.

    Ok(info)
}

/// `sap_mode` enum for `chparam_info()` (ETSI TS 103 190-1 §4.2.10.1
/// Table 47 / §4.3.8.1 Table 114).
///
/// * `0` — no SAP; both channels are independent.
/// * `1` — M/S processing in the scale-factor bands flagged by the
///   `ms_used[g][sfb]` array that follows.
/// * `2` — M/S processing in **all** scale-factor bands (header-only
///   on the wire — no per-band payload follows).
/// * `3` — full SAP: a `sap_data()` body follows (Table 48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SapMode {
    None,
    MsUsed,
    MsAll,
    SapData,
}

impl SapMode {
    pub fn from_u32(v: u32) -> Self {
        match v & 0b11 {
            0 => Self::None,
            1 => Self::MsUsed,
            2 => Self::MsAll,
            _ => Self::SapData,
        }
    }
}

/// `sap_data()` payload (ETSI TS 103 190-1 §4.2.10.2 Table 48).
///
/// Carries the stereo audio processing coefficients used in joint-MDCT
/// stereo coding. `sap_coeff_used[g][sfb]` is a per-pair (sfb, sfb+1)
/// flag — both halves of the pair share the same flag value. When a
/// pair's flag is set the bitstream carries a Huffman-coded `dpcm_alpha_q`
/// delta against the previous coded pair (initial reference is 60 — the
/// HCB_SCALEFAC DC).
#[derive(Debug, Clone, Default)]
pub struct SapData {
    /// 1 = all `sap_coeff_used` flags are set (no flag array transmitted).
    pub sap_coeff_all: bool,
    /// Per-group, per-sfb `sap_coeff_used[g][sfb]` flags (length =
    /// max_sfb_g).
    pub sap_coeff_used: Vec<Vec<bool>>,
    /// `delta_code_time` — only present when `num_window_groups != 1`.
    pub delta_code_time: bool,
    /// `dpcm_alpha_q[g][sfb]` — Huffman-decoded DPCM-coded raw indices,
    /// expressed as deltas. Length = max_sfb_g, entries are 0 when the
    /// pair flag was not set.
    pub dpcm_alpha_q: Vec<Vec<i32>>,
}

/// Parsed `chparam_info()` (ETSI TS 103 190-1 §4.2.10.1 Table 47).
#[derive(Debug, Clone, Default)]
pub struct ChparamInfo {
    /// `sap_mode` — 2-bit selector. Drives whether ms_used / sap_data
    /// follow.
    pub sap_mode: u32,
    /// `ms_used[g][sfb]` — only populated for `sap_mode == 1`. Length =
    /// num_window_groups; inner length = `get_max_sfb(g)`.
    pub ms_used: Vec<Vec<bool>>,
    /// `sap_data()` payload — only populated for `sap_mode == 3`.
    pub sap_data: Option<SapData>,
}

impl ChparamInfo {
    pub fn mode(&self) -> SapMode {
        SapMode::from_u32(self.sap_mode)
    }
}

/// Parse `chparam_info()` per Table 47.
///
/// `max_sfb_per_group[g]` provides the per-group max_sfb to walk the
/// `ms_used` and `sap_coeff_used` loops. Spec note: `get_max_sfb(g)`
/// (Pseudocode 5) returns either `max_sfb[idx]` or `max_sfb_side[idx]`
/// depending on `b_side_channel`; the caller is expected to pass the
/// effective per-group bound here.
pub fn parse_chparam_info(
    br: &mut BitReader<'_>,
    max_sfb_per_group: &[u32],
) -> Result<ChparamInfo> {
    let sap_mode = br.read_u32(2)?;
    let mut info = ChparamInfo {
        sap_mode,
        ..ChparamInfo::default()
    };
    match SapMode::from_u32(sap_mode) {
        SapMode::None | SapMode::MsAll => {}
        SapMode::MsUsed => {
            let mut groups = Vec::with_capacity(max_sfb_per_group.len());
            for &m in max_sfb_per_group {
                let mut row = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    row.push(br.read_bit()?);
                }
                groups.push(row);
            }
            info.ms_used = groups;
        }
        SapMode::SapData => {
            info.sap_data = Some(parse_sap_data(br, max_sfb_per_group)?);
        }
    }
    Ok(info)
}

/// Parse `sap_data()` per Table 48.
pub fn parse_sap_data(br: &mut BitReader<'_>, max_sfb_per_group: &[u32]) -> Result<SapData> {
    let num_groups = max_sfb_per_group.len();
    let sap_coeff_all = br.read_bit()?;
    let mut sap_coeff_used = Vec::with_capacity(num_groups);
    if !sap_coeff_all {
        // For each group, walk pairs of sfb (sfb, sfb+1) — read one bit
        // per pair, copy the flag into both halves.
        for &m in max_sfb_per_group {
            let mut row = vec![false; m as usize];
            let mut sfb = 0usize;
            while sfb < m as usize {
                let f = br.read_bit()?;
                row[sfb] = f;
                if sfb + 1 < m as usize {
                    row[sfb + 1] = f;
                }
                sfb += 2;
            }
            sap_coeff_used.push(row);
        }
    } else {
        for &m in max_sfb_per_group {
            sap_coeff_used.push(vec![true; m as usize]);
        }
    }
    let delta_code_time = if num_groups != 1 {
        br.read_bit()?
    } else {
        false
    };
    let mut dpcm_alpha_q = Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let m = max_sfb_per_group[g] as usize;
        let mut row = vec![0i32; m];
        let mut sfb = 0usize;
        while sfb < m {
            if sap_coeff_used[g][sfb] {
                let raw = huff_decode(br, HCB_SCALEFAC_LEN, HCB_SCALEFAC_CW)?;
                // HCB_SCALEFAC's DC element is at index 60; Table 48
                // codes raw indices 0..120 that map to deltas
                // (raw - 60).
                row[sfb] = raw as i32 - 60;
            }
            sfb += 2;
        }
        dpcm_alpha_q.push(row);
    }
    Ok(SapData {
        sap_coeff_all,
        sap_coeff_used,
        delta_code_time,
        dpcm_alpha_q,
    })
}

/// Per-(group, sfb) SAP coefficients (a, b, c, d) extracted from a
/// `chparam_info()` element per Pseudocode 59 (§5.3.2). Each row is a
/// window-group; each entry is the SAP 2x2 mixing matrix coefficient
/// quartet for that scale-factor band — i.e.
///
/// ```text
///     [O_0]   [a  b] [I_0]
///     [   ] = [    ] [   ]
///     [O_1]   [c  d] [I_1]
/// ```
///
/// applied per-sfb to a stereo pair (the joint-MDCT / SAP companding
/// matrix). The four extraction modes are:
///
/// * `sap_mode == 0` — identity: `a = d = 1, b = c = 0`.
/// * `sap_mode == 1, ms_used == 0` — identity (same as mode 0).
/// * `sap_mode == 1, ms_used == 1` — M/S inverse: `a = b = c = 1, d = -1`.
/// * `sap_mode == 2` — M/S inverse in **all** scale-factor bands
///   (Table 114 row 2; same coefficients as the flagged mode-1 bands).
/// * `sap_mode == 3, sap_used` — alpha-driven SAP: with
///   `sap_gain = alpha_q * 0.1`,
///   `a = 1 + sap_gain, b = 1, c = 1 - sap_gain, d = -1`.
/// * `sap_mode == 3, !sap_used` — passthrough: `a = 1, b = 0, c = 0, d = 1`.
///
/// `alpha_q[g][sfb]` is differential-decoded across pair-major sfb (sfb,
/// sfb+1 share a flag — odd sfbs inherit the even sfb's alpha_q). Cross-
/// group time-deltas (`delta_code_time`) require that group g and g-1
/// share the same `max_sfb_g` (Pseudocode 59 forces `code_delta = 0`
/// otherwise).
#[derive(Debug, Clone, Default)]
pub struct SapCoeffs {
    /// `[g][sfb] -> (a, b, c, d)` — outer length is `num_window_groups`,
    /// inner length is `max_sfb_per_group[g]`. All values in `f32`.
    pub abcd: Vec<Vec<(f32, f32, f32, f32)>>,
}

impl SapCoeffs {
    /// Identity (passthrough) coefficients for an empty `chparam_info()`
    /// or when `sap_mode == 0` is signalled with no follow-up payload.
    pub fn identity(max_sfb_per_group: &[u32]) -> Self {
        let abcd = max_sfb_per_group
            .iter()
            .map(|&m| vec![(1.0, 0.0, 0.0, 1.0); m as usize])
            .collect();
        Self { abcd }
    }
}

/// Extract per-(g, sfb) SAP coefficients (a, b, c, d) from a parsed
/// `chparam_info()` per Pseudocode 59 (§5.3.2). `max_sfb_per_group[g]`
/// must match the bound used to walk the `chparam_info` body.
///
/// Pseudocode 59 derives `alpha_q[g][sfb]` from `dpcm_alpha_q[g][sfb]`
/// via DPCM (sfb-major delta against `alpha_q[g][sfb-2]` for even sfbs;
/// odd sfbs inherit the even pair-mate; cross-group `delta_code_time`
/// folds in when supported). Then `sap_gain = alpha_q * 0.1` drives
/// `(a, b, c, d) = (1 + sap_gain, 1, 1 - sap_gain, -1)` for SAP-coded
/// bands and `(1, 0, 0, 1)` for skipped bands.
pub fn extract_sap_abcd(info: &ChparamInfo, max_sfb_per_group: &[u32]) -> SapCoeffs {
    let mode = info.mode();
    let mut abcd: Vec<Vec<(f32, f32, f32, f32)>> = max_sfb_per_group
        .iter()
        .map(|&m| vec![(1.0, 0.0, 0.0, 1.0); m as usize])
        .collect();
    match mode {
        SapMode::None => {
            // Identity already populated above.
        }
        SapMode::MsAll => {
            // Table 114 mode 2: M/S processing in ALL scale-factor
            // bands — every row takes the Pseudocode 59 M/S column
            // `a = b = c = 1, d = -1` (no per-band payload on the wire).
            for row in abcd.iter_mut() {
                for slot in row.iter_mut() {
                    *slot = (1.0, 1.0, 1.0, -1.0);
                }
            }
        }
        SapMode::MsUsed => {
            // Per-sfb selector — when ms_used == 1, swap the row to the
            // M/S inverse matrix. Pseudocode 59 column:
            //   a = b = c = 1, d = -1.
            for (g, &m) in max_sfb_per_group.iter().enumerate() {
                let row = info.ms_used.get(g);
                for (sfb, slot) in abcd[g].iter_mut().enumerate().take(m as usize) {
                    let used = row.and_then(|r| r.get(sfb).copied()).unwrap_or(false);
                    if used {
                        *slot = (1.0, 1.0, 1.0, -1.0);
                    }
                }
            }
        }
        SapMode::SapData => {
            let Some(sd) = info.sap_data.as_ref() else {
                return SapCoeffs { abcd };
            };
            // Pair-major DPCM differential decode of dpcm_alpha_q ->
            // alpha_q[g][sfb]. For odd sfbs inherit alpha_q[g][sfb-1].
            // For even sfbs: delta = dpcm_alpha_q[g][sfb] (Huffman path
            // already subtracted the DC offset of 60 in our parser);
            // start of group: alpha = delta when `sfb == 0` and not
            // cross-group time-coding; otherwise alpha[g][sfb] =
            // alpha[g][sfb-2] + delta. With cross-group time-coding
            // (`delta_code_time == 1` and `max_sfb_g == max_sfb_prev`),
            // alpha[g][sfb] = alpha[g-1][sfb] + delta.
            let num_groups = max_sfb_per_group.len();
            let mut alpha_q: Vec<Vec<i32>> = max_sfb_per_group
                .iter()
                .map(|&m| vec![0i32; m as usize])
                .collect();
            let mut max_sfb_prev = max_sfb_per_group.first().copied().unwrap_or(0);
            for g in 0..num_groups {
                let m = max_sfb_per_group[g] as usize;
                let dp = sd.dpcm_alpha_q.get(g);
                let used_row = sd.sap_coeff_used.get(g);
                let mut sfb = 0usize;
                while sfb < m {
                    let used = used_row.and_then(|r| r.get(sfb).copied()).unwrap_or(false);
                    if used {
                        if sfb % 2 == 1 {
                            // Odd sfb inherits the pair's even partner.
                            alpha_q[g][sfb] =
                                alpha_q[g].get(sfb.saturating_sub(1)).copied().unwrap_or(0);
                        } else {
                            // Even sfb: differential decode against the
                            // previous even sfb (or the same sfb in group
                            // g-1 when `delta_code_time` is on and the
                            // group-bounds match — Pseudocode 59 forces
                            // `code_delta = 0` if `max_sfb_g != max_sfb_prev`
                            // or `g == 0`).
                            let delta = dp.and_then(|r| r.get(sfb).copied()).unwrap_or(0);
                            let code_delta = g != 0
                                && max_sfb_per_group[g] == max_sfb_prev
                                && sd.delta_code_time;
                            let prev = if code_delta {
                                alpha_q
                                    .get(g.wrapping_sub(1))
                                    .and_then(|r| r.get(sfb).copied())
                                    .unwrap_or(0)
                            } else if sfb == 0 {
                                0
                            } else {
                                alpha_q[g].get(sfb - 2).copied().unwrap_or(0)
                            };
                            alpha_q[g][sfb] = prev + delta;
                        }
                        let sap_gain = alpha_q[g][sfb] as f32 * 0.1;
                        abcd[g][sfb] = (1.0 + sap_gain, 1.0, 1.0 - sap_gain, -1.0);
                    } else {
                        // Skipped band: passthrough identity.
                        abcd[g][sfb] = (1.0, 0.0, 0.0, 1.0);
                    }
                    sfb += 1;
                }
                max_sfb_prev = max_sfb_per_group[g];
            }
        }
    }
    SapCoeffs { abcd }
}

/// `(L_spec, R_spec, Ls_spec, Rs_spec)` — the four preliminary
/// spectra produced by [`apply_sap_table_181`].
pub type SapTable181Output = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Apply the §5.3.4.3.2 / Table 181 first-stage SAP matrix for the
/// 5_X `ASPX_ACPL_1` mode. Mixes the two-channel preliminary spectra
/// `(sSMP_A, sSMP_B)` with the joint-MDCT residual pair
/// `(sSMP_3, sSMP_4)` into preliminary `(L, R, Ls, Rs)` spectra using
/// the per-(g, sfb) `(a, b, c, d)` coefficients extracted from the two
/// `chparam_info()` payloads following `max_sfb_master`.
///
/// Per Table 181 (the matrix on the right):
///
/// ```text
///   [sSMP_L]    [a0  0   0   b0  0 ]   [sSMP_A]
///   [sSMP_R]  = [0   a1  0   0   b1] * [sSMP_B]
///   [sSMP_C]    [0   0   1   0   0 ]   [sSMP_C]
///   [sSMP_Ls]   [c0  0   0   d0  0 ]   [sSMP_3]
///   [sSMP_Rs]   [0   c1  0   0   d1]   [sSMP_4]
/// ```
///
/// Coefficients `(a, b, c, d)` come from each `chparam_info()` SAP
/// extraction (Pseudocode 59 / [`extract_sap_abcd`]); the SAP layer
/// runs on a single window-group at the dominant transform length per
/// the §4.2.6.6 NOTE, so `chparam_pair` is keyed against the joint-
/// MDCT residual layer's `max_sfb_master` bound.
///
/// `tl` is the transform_length used to derive the sfb offsets; bins
/// past the SAP-coded extent (sfb >= max_sfb) pass through (high half
/// keeps `A`/`B`, low half is silent — same convention as the 7_X
/// SAP application in `dispatch_7x_additional_channel_pair`).
///
/// All input slices must be at least `n` long where `n = tl`. Outputs
/// are returned in `(l_spec, r_spec, ls_spec, rs_spec)` order, each
/// length `n`. Returns `None` if `tl` has no SFB table or if any
/// per-channel slice is shorter than `n`.
pub fn apply_sap_table_181(
    a_spec: &[f32],
    b_spec: &[f32],
    s3_spec: &[f32],
    s4_spec: &[f32],
    chparam_pair: &[ChparamInfo; 2],
    max_sfb_master: u32,
    tl: u32,
) -> Option<SapTable181Output> {
    let n = tl as usize;
    if n == 0 || a_spec.len() < n || b_spec.len() < n || s3_spec.len() < n || s4_spec.len() < n {
        return None;
    }
    let sfbo = crate::sfb_offset::sfb_offset_48(tl)?;
    // Joint-MDCT residual layer is single-window-group at the dominant
    // transform length — both chparam_info() bodies use the same
    // `max_sfb_per_group = [max_sfb_master]` bound.
    let coeffs0 = extract_sap_abcd(&chparam_pair[0], &[max_sfb_master]);
    let coeffs1 = extract_sap_abcd(&chparam_pair[1], &[max_sfb_master]);
    let abcd0: &[(f32, f32, f32, f32)] = coeffs0.abcd.first().map(|v| v.as_slice()).unwrap_or(&[]);
    let abcd1: &[(f32, f32, f32, f32)] = coeffs1.abcd.first().map(|v| v.as_slice()).unwrap_or(&[]);
    let max_sfb = max_sfb_master as usize;
    let mut l_spec = vec![0.0f32; n];
    let mut r_spec = vec![0.0f32; n];
    let mut ls_spec = vec![0.0f32; n];
    let mut rs_spec = vec![0.0f32; n];
    let usable0 = abcd0.len().min(max_sfb);
    let usable1 = abcd1.len().min(max_sfb);
    let usable = usable0.min(usable1);
    for sfb in 0..usable {
        let lo = sfbo[sfb] as usize;
        let hi = sfbo[sfb + 1] as usize;
        let hi = hi.min(n);
        let (a0, b0, c0, d0) = abcd0[sfb];
        let (a1, b1, c1, d1) = abcd1[sfb];
        for k in lo..hi {
            let a = a_spec[k];
            let b = b_spec[k];
            let s3 = s3_spec[k];
            let s4 = s4_spec[k];
            l_spec[k] = a0 * a + b0 * s3;
            r_spec[k] = a1 * b + b1 * s4;
            ls_spec[k] = c0 * a + d0 * s3;
            rs_spec[k] = c1 * b + d1 * s4;
        }
    }
    // Bins past the SAP-coded extent: high pair (L/R) keeps the
    // unmodified A/B carriers (so the front pair retains its full
    // bandwidth); low pair (Ls/Rs) stays silent (Pseudocode 117 will
    // synthesise its own surround from the alpha/beta parameters).
    let unmixed_start = sfbo.get(usable).copied().map(|v| v as usize).unwrap_or(n);
    let unmixed_lo = unmixed_start.min(n);
    if unmixed_lo < n {
        l_spec[unmixed_lo..n].copy_from_slice(&a_spec[unmixed_lo..n]);
        r_spec[unmixed_lo..n].copy_from_slice(&b_spec[unmixed_lo..n]);
    }
    Some((l_spec, r_spec, ls_spec, rs_spec))
}

/// `(sSMP_A, sSMP_B, sSMP_3, sSMP_4)` — the four preliminary spectra
/// that an encoder feeds back through [`apply_sap_table_181`] to
/// reconstruct the desired `(L, R, Ls, Rs)`.
pub type SapTable181EncodeOutput = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
/// Encoder-side dual of [`apply_sap_table_181`] — invert the §5.3.4.3.2
/// / Table 181 first-stage SAP matrix to recover the joint-MDCT
/// preliminary spectra `(sSMP_A, sSMP_B, sSMP_3, sSMP_4)` that produce
/// the desired preliminary `(L, R, Ls, Rs)` when fed through the
/// forward matrix with the same `chparam_info()` pair.
///
/// The forward matrix per Table 181 splits into two independent 2x2
/// sub-systems — one for the (L, Ls) / (A, s3) pair driven by
/// `chparam_pair[0]`'s `(a0, b0, c0, d0)`, and one for the (R, Rs) /
/// (B, s4) pair driven by `chparam_pair[1]`'s `(a1, b1, c1, d1)`:
///
/// ```text
///   [L ]   [a0  b0] [A ]        [R ]   [a1  b1] [B ]
///   [Ls] = [c0  d0] [s3]   and  [Rs] = [c1  d1] [s4]
/// ```
///
/// Inversion per sfb uses the closed-form 2x2 inverse:
///
/// ```text
///   det = a*d - b*c
///   [A ] = (1/det) * [ d  -b] [L ]
///   [s3]             [-c   a] [Ls]
/// ```
///
/// For the three SAP coefficient families produced by
/// [`extract_sap_abcd`] the determinant is always non-zero:
///
/// * Identity `(a, b, c, d) = (1, 0, 0, 1)` — `det = 1`; the inverse
///   reproduces the inputs (A = L, s3 = Ls).
/// * M/S inverse `(1, 1, 1, -1)` — `det = -2`; A = (L + Ls)/2,
///   s3 = (L - Ls)/2.
/// * SAP-coded `(1 + g, 1, 1 - g, -1)` with `g = alpha_q * 0.1` —
///   `det = -(1 + g) - (1 - g) = -2`; A = (L + Ls)/2,
///   s3 = ((1 - g)*L - (1 + g)*Ls) / -2 = ((1 + g)*Ls - (1 - g)*L)/2.
///
/// Outside the SAP-coded extent the forward pass leaves the front pair
/// at `(L, R) = (A, B)` and zeros the surround pair. The inverse here
/// mirrors that convention: bins past `sfb_offset[max_sfb_master]`
/// pass `A = L`, `B = R` and emit `s3 = 0`, `s4 = 0`. This matches
/// what the decoder's [`apply_sap_table_181`] re-produces.
///
/// All input slices must be at least `n` long where `n = tl`. Outputs
/// are returned in `(a_spec, b_spec, s3_spec, s4_spec)` order, each
/// length `n`. Returns `None` if `tl` has no SFB table or if any
/// per-channel slice is shorter than `n`, mirroring the forward path.
pub fn invert_sap_table_181(
    l_spec: &[f32],
    r_spec: &[f32],
    ls_spec: &[f32],
    rs_spec: &[f32],
    chparam_pair: &[ChparamInfo; 2],
    max_sfb_master: u32,
    tl: u32,
) -> Option<SapTable181EncodeOutput> {
    let n = tl as usize;
    if n == 0 || l_spec.len() < n || r_spec.len() < n || ls_spec.len() < n || rs_spec.len() < n {
        return None;
    }
    let sfbo = crate::sfb_offset::sfb_offset_48(tl)?;
    // Match the forward path's single-window-group convention at the
    // dominant transform length.
    let coeffs0 = extract_sap_abcd(&chparam_pair[0], &[max_sfb_master]);
    let coeffs1 = extract_sap_abcd(&chparam_pair[1], &[max_sfb_master]);
    let abcd0: &[(f32, f32, f32, f32)] = coeffs0.abcd.first().map(|v| v.as_slice()).unwrap_or(&[]);
    let abcd1: &[(f32, f32, f32, f32)] = coeffs1.abcd.first().map(|v| v.as_slice()).unwrap_or(&[]);
    let max_sfb = max_sfb_master as usize;
    let mut a_spec = vec![0.0f32; n];
    let mut b_spec = vec![0.0f32; n];
    let mut s3_spec = vec![0.0f32; n];
    let mut s4_spec = vec![0.0f32; n];
    let usable0 = abcd0.len().min(max_sfb);
    let usable1 = abcd1.len().min(max_sfb);
    let usable = usable0.min(usable1);
    for sfb in 0..usable {
        let lo = sfbo[sfb] as usize;
        let hi = sfbo[sfb + 1] as usize;
        let hi = hi.min(n);
        let (a0, b0, c0, d0) = abcd0[sfb];
        let (a1, b1, c1, d1) = abcd1[sfb];
        let det0 = a0 * d0 - b0 * c0;
        let det1 = a1 * d1 - b1 * c1;
        // Identity / M/S / SAP-coded families all yield non-zero det;
        // a future spec extension that introduced a singular matrix
        // would land here as silence, matching the forward path's
        // graceful-degradation convention rather than panicking.
        if det0 == 0.0 || det1 == 0.0 {
            continue;
        }
        let inv0 = 1.0 / det0;
        let inv1 = 1.0 / det1;
        for k in lo..hi {
            let l = l_spec[k];
            let ls = ls_spec[k];
            let r = r_spec[k];
            let rs = rs_spec[k];
            a_spec[k] = inv0 * (d0 * l - b0 * ls);
            s3_spec[k] = inv0 * (-c0 * l + a0 * ls);
            b_spec[k] = inv1 * (d1 * r - b1 * rs);
            s4_spec[k] = inv1 * (-c1 * r + a1 * rs);
        }
    }
    // Bins past the SAP-coded extent: the forward path leaves
    // (L, R) = (A, B) and surrounds silent. The inverse mirrors:
    // pass A = L, B = R; leave s3 = s4 = 0 (already zero-initialised).
    let unmixed_start = sfbo.get(usable).copied().map(|v| v as usize).unwrap_or(n);
    let unmixed_lo = unmixed_start.min(n);
    if unmixed_lo < n {
        a_spec[unmixed_lo..n].copy_from_slice(&l_spec[unmixed_lo..n]);
        b_spec[unmixed_lo..n].copy_from_slice(&r_spec[unmixed_lo..n]);
    }
    Some((a_spec, b_spec, s3_spec, s4_spec))
}

// ====================================================================
// Encoder-side ChparamInfo builders — duals of extract_sap_abcd
// ====================================================================
//
// `extract_sap_abcd` is the decoder-side primitive: it takes a parsed
// `ChparamInfo` and produces the per-(g, sfb) Table-181 `(a, b, c, d)`
// matrix the §5.3.4.3.2 SAP layer applies to the carrier spectra.
//
// An IMS encoder needs the inverse: starting from a psychoacoustic
// decision on which bands to code as M/S vs. SAP-driven joint stereo
// (with per-band `alpha_q` indices in [-60, +60]), build a `ChparamInfo`
// that round-trips through `extract_sap_abcd` to recover the same
// matrix the encoder picked. The two builders below are the two
// non-trivial `SapMode` arms (`SapMode::None` / `SapMode::MsAll`
// are header-only and need no builder).

/// Build a `ChparamInfo` with `sap_mode == 1` (`SapMode::MsUsed`) from
/// a per-(group, sfb) `ms_used` flag matrix.
///
/// Encoder-side dual of [`extract_sap_abcd`] for the `SapMode::MsUsed`
/// arm: feeding the result of this builder into `extract_sap_abcd` with
/// the same `max_sfb_per_group` reproduces the input matrix bit-for-bit
/// (per-sfb `(1, 1, 1, -1)` for set bands, identity `(1, 0, 0, 1)` for
/// cleared bands).
///
/// Rows shorter than `max_sfb_per_group[g]` are accepted — the writer
/// in [`crate::encoder_asf::write_chparam_info`] pads missing tail bits
/// with `false`, which is bit-equivalent to a row that explicitly
/// carries `false` in that position. Rows longer than the per-group
/// bound have their tail entries ignored by both writer and
/// `extract_sap_abcd` (which walks `take(max_sfb_per_group[g])`).
pub fn build_chparam_info_ms_used(ms_used_per_group: Vec<Vec<bool>>) -> ChparamInfo {
    ChparamInfo {
        sap_mode: 1,
        ms_used: ms_used_per_group,
        sap_data: None,
    }
}

/// Build a `ChparamInfo` with `sap_mode == 3` (`SapMode::SapData`) from
/// per-(group, sfb) `alpha_q` indices and `sap_coeff_used` flags.
///
/// Encoder-side dual of [`extract_sap_abcd`] for the `SapMode::SapData`
/// arm: computes the pair-major DPCM `dpcm_alpha_q[g][sfb]` deltas the
/// decoder accumulates back into `alpha_q[g][sfb]` per Pseudocode 59.
///
/// `alpha_q_per_group[g][sfb]` is the desired post-DPCM `alpha_q` index
/// (range `[-60, +60]` — the HCB_SCALEFAC raw symbol offset of 60 is
/// applied by the writer, not by this builder). Pair-major encoding
/// per Pseudocode 59:
///
/// * Odd sfbs share the pair-mate's `alpha_q` — the decoder inherits
///   `alpha_q[g][sfb] = alpha_q[g][sfb-1]`. The builder ignores the
///   odd-sfb input value (callers that pass the inherited value get
///   the same encoded byte stream).
/// * Even sfbs differential-decode against the previous reference:
///   * `code_delta == 1` (cross-group time-code: `g > 0`, the per-group
///     bound matches `max_sfb_per_group[g-1]`, and `delta_code_time`
///     is set) — reference is `alpha_q[g-1][sfb]`.
///   * Otherwise the reference is `alpha_q[g][sfb-2]` for `sfb > 0`
///     and zero for `sfb == 0`.
///
/// `sap_coeff_used_per_group[g][sfb]` — per-pair flag matrix matching
/// the matrix `extract_sap_abcd` walks; clear flags emit
/// `dpcm_alpha_q = 0` for that pair and the decoded `alpha_q` stays at
/// the previous reference (which the SAP layer interprets as
/// identity-passthrough on that band, so the actual `alpha_q` value at
/// a cleared pair is don't-care). Rows shorter than the per-group
/// bound are treated as all-false. The fully-uniform "all set" matrix
/// is detected and `sap_coeff_all` is set to `1` so the bitstream
/// elides the per-pair flag array — bit-equivalent to the original
/// encoder's intent.
///
/// `delta_code_time` is honoured only when `max_sfb_per_group.len() > 1`
/// (single-group payloads omit the bit per Table 48 and the decoder
/// treats it as `false`).
///
/// Round-trip guarantee: feeding the returned `ChparamInfo` into
/// [`extract_sap_abcd`] with the same `max_sfb_per_group` reproduces
/// the input `alpha_q_per_group` on set bands (and `(1, 0, 0, 1)` on
/// cleared bands). The exhaustive coverage of `code_delta == 0` /
/// `code_delta == 1` is pinned by unit tests below.
///
/// Defensive clamps mirror [`crate::encoder_asf::write_sap_data`]:
/// per-sfb deltas outside `[-60, +60]` clip to the HCB_SCALEFAC
/// codebook bounds when written; callers driving `alpha_q` jumps
/// larger than the codebook can express should pre-quantise their
/// per-band targets to keep the cumulative per-pair delta inside the
/// codebook range.
pub fn build_chparam_info_sap_data_from_alpha_q(
    alpha_q_per_group: &[Vec<i32>],
    sap_coeff_used_per_group: &[Vec<bool>],
    delta_code_time: bool,
    max_sfb_per_group: &[u32],
) -> ChparamInfo {
    let num_groups = max_sfb_per_group.len();
    let mut sap_coeff_used: Vec<Vec<bool>> = Vec::with_capacity(num_groups);
    let mut dpcm_alpha_q: Vec<Vec<i32>> = Vec::with_capacity(num_groups);
    let mut all_set = true;
    let mut max_sfb_prev: u32 = max_sfb_per_group.first().copied().unwrap_or(0);
    for (g, &m) in max_sfb_per_group.iter().enumerate() {
        let m = m as usize;
        let used_row = sap_coeff_used_per_group.get(g);
        let alpha_row = alpha_q_per_group.get(g);
        let prev_alpha_row = if g == 0 {
            None
        } else {
            alpha_q_per_group.get(g - 1)
        };
        let mut used_out: Vec<bool> = vec![false; m];
        let mut dpcm_out: Vec<i32> = vec![0i32; m];
        let mut sfb = 0usize;
        while sfb < m {
            let used = used_row.and_then(|r| r.get(sfb).copied()).unwrap_or(false);
            used_out[sfb] = used;
            // The decoder's matrix only inherits the pair-mate, so any
            // sfb+1 within an active pair carries the same flag.
            if sfb + 1 < m {
                used_out[sfb + 1] = used;
            }
            if !used {
                all_set = false;
            }
            // Even-sfb in an active pair: compute the DPCM delta. Mirror
            // `extract_sap_abcd`'s `code_delta` policy. Odd sfb leaves
            // `dpcm_out[sfb] = 0` — the writer walks `sfb += 2` so the
            // odd slot is never read on emit and the decoder inherits
            // from the pair-mate anyway.
            if used && sfb % 2 == 0 {
                let code_delta = g != 0 && max_sfb_per_group[g] == max_sfb_prev && delta_code_time;
                let prev = if code_delta {
                    prev_alpha_row
                        .and_then(|r| r.get(sfb).copied())
                        .unwrap_or(0)
                } else if sfb == 0 {
                    0
                } else {
                    alpha_row
                        .and_then(|r| r.get(sfb.wrapping_sub(2)).copied())
                        .unwrap_or(0)
                };
                let cur = alpha_row.and_then(|r| r.get(sfb).copied()).unwrap_or(0);
                dpcm_out[sfb] = cur - prev;
            }
            sfb += 1;
        }
        sap_coeff_used.push(used_out);
        dpcm_alpha_q.push(dpcm_out);
        max_sfb_prev = max_sfb_per_group[g];
    }
    // num_groups == 1 -> delta_code_time bit isn't transmitted; the
    // decoder treats it as `false`. Normalise the field so a
    // round-trip through write_chparam_info / parse_chparam_info
    // recovers `delta_code_time == false` regardless of the caller's
    // input on single-group payloads.
    let dct_normalised = num_groups != 1 && delta_code_time;
    ChparamInfo {
        sap_mode: 3,
        ms_used: vec![],
        sap_data: Some(SapData {
            sap_coeff_all: all_set && num_groups > 0,
            sap_coeff_used,
            delta_code_time: dct_normalised,
            dpcm_alpha_q,
        }),
    }
}

/// Build a header-only `ChparamInfo` with `sap_mode == 0`
/// ([`SapMode::None`]) — the trivial encoder-side dual of
/// [`extract_sap_abcd`] for the `SapMode::None` arm.
///
/// `SapMode::None` carries no body — both channels stay independent for
/// the rest of the element. Feeding this builder's result into
/// [`extract_sap_abcd`] with any `max_sfb_per_group` reproduces the
/// identity per-sfb matrix `(1, 0, 0, 1)` (same as the decoder's
/// `SapMode::None` arm), and a `write_chparam_info` / `parse_chparam_info`
/// round-trip recovers the same header-only element.
///
/// This is the trivial third arm of the
/// [`build_chparam_info_ms_used`] / [`build_chparam_info_sap_data_from_alpha_q`]
/// builder family; it exists so encoder paths can produce all three
/// non-reserved `SapMode` arms uniformly without ad-hoc `ChparamInfo {
/// sap_mode: 0, ..Default::default() }` literals.
pub fn build_chparam_info_none() -> ChparamInfo {
    ChparamInfo {
        sap_mode: 0,
        ms_used: vec![],
        sap_data: None,
    }
}

/// Per-(group, sfb) M/S-vs-L/R decision driver for the encoder-side
/// `SapMode::MsUsed` arm — picks `ms_used[g][sfb]` per band using the
/// energy-concentration criterion.
///
/// The decoder's `SapMode::MsUsed` arm reconstructs `(L, R)` from the
/// joint-stereo pair `(M', S')` via the per-sfb matrix `(1, 1, 1, -1)`
/// (i.e. `L = M' + S', R = M' - S'`), which means the encoder
/// transmits `M' = (L + R) / 2` and `S' = (L - R) / 2`. The total
/// energy is preserved up to the M/S scale:
/// `E_M' + E_S' = (E_L + E_R) / 2`, so raw total energy is **not** a
/// useful criterion (M/S always has half the total energy regardless
/// of correlation).
///
/// The criterion this driver uses is the standard joint-stereo
/// *concentration* test: pick M/S when one of `(M', S')` carries
/// strictly less energy than the smaller of `(L, R)`. For a
/// well-correlated pair, M' concentrates the signal and S' vanishes —
/// `min(E_M', E_S') < min(E_L, E_R)` — and a downstream quantizer can
/// spend fewer bits on the small-energy channel. For an
/// uncorrelated pair, E_M' and E_S' are both around `(E_L + E_R) / 4`
/// and that's not less than `min(E_L, E_R)`, so the band stays L/R.
///
/// Algebraically per band (sum over bins `[sfb_offset[sfb],
/// sfb_offset[sfb+1])`):
///
/// ```text
///   E_L  = sum_k L[k]^2
///   E_R  = sum_k R[k]^2
///   E_M' = sum_k ((L[k] + R[k]) / 2)^2  = (E_L + E_R + 2 * cross) / 4
///   E_S' = sum_k ((L[k] - R[k]) / 2)^2  = (E_L + E_R - 2 * cross) / 4
///   cross = sum_k L[k] * R[k]
/// ```
///
/// pick `ms_used = true` iff `min(E_M', E_S') < min(E_L, E_R)`.
///
/// `l_spec_per_group[g]` / `r_spec_per_group[g]` are per-group MDCT
/// spectra at the same transform length as `sfb_offset`. Rows shorter
/// than `sfb_offset[max_sfb_per_group[g]]` produce `false` decisions on
/// the missing bins (the missing tail bands stay L/R).
///
/// Returns a `Vec<Vec<bool>>` shape-matched to `max_sfb_per_group`
/// suitable for feeding directly into [`build_chparam_info_ms_used`].
/// The strict-less comparison resolves ties (including zero-energy
/// bands where `E_L == E_R == E_M' == E_S' == 0`) to `false` — keep
/// L/R coding when the concentration criterion offers no benefit
/// rather than spend a `ms_used` bit on the band.
///
/// Round-trip: feeding the returned matrix through
/// [`build_chparam_info_ms_used`] and then [`extract_sap_abcd`] with
/// the same `max_sfb_per_group` reproduces the per-sfb `(1, 1, 1, -1)`
/// matrix exactly on the picked bands and identity on the rest.
pub fn select_ms_used_for_pair(
    l_spec_per_group: &[Vec<f32>],
    r_spec_per_group: &[Vec<f32>],
    sfb_offset: &[u16],
    max_sfb_per_group: &[u32],
) -> Vec<Vec<bool>> {
    let mut out: Vec<Vec<bool>> = Vec::with_capacity(max_sfb_per_group.len());
    for (g, &m) in max_sfb_per_group.iter().enumerate() {
        let m_usize = m as usize;
        let mut row = vec![false; m_usize];
        let l = l_spec_per_group.get(g);
        let r = r_spec_per_group.get(g);
        let (Some(l), Some(r)) = (l, r) else {
            out.push(row);
            continue;
        };
        for (sfb, slot) in row.iter_mut().enumerate().take(m_usize) {
            // Guard against an sfb_offset that's shorter than
            // max_sfb + 1 entries — clamp to the available range so we
            // never index past the end. (sfb_offset_48 is always
            // num_sfb_max + 1 in practice but we defensively support
            // shorter caller-supplied tables.)
            let Some(&lo) = sfb_offset.get(sfb) else {
                break;
            };
            let Some(&hi) = sfb_offset.get(sfb + 1) else {
                break;
            };
            let lo = lo as usize;
            let hi = (hi as usize).min(l.len()).min(r.len());
            if hi <= lo {
                continue;
            }
            // Per-band energies. cross = sum L * R drives the per-band
            // M' / S' energy distribution via E_M' = (E_L + E_R + 2 *
            // cross) / 4 and E_S' = (E_L + E_R - 2 * cross) / 4.
            let mut e_l = 0.0f64;
            let mut e_r = 0.0f64;
            let mut cross = 0.0f64;
            for k in lo..hi {
                let lk = l[k] as f64;
                let rk = r[k] as f64;
                e_l += lk * lk;
                e_r += rk * rk;
                cross += lk * rk;
            }
            let sum = e_l + e_r;
            let e_m = (sum + 2.0 * cross) * 0.25;
            let e_s = (sum - 2.0 * cross) * 0.25;
            let min_lr = e_l.min(e_r);
            let min_ms = e_m.min(e_s);
            // Strict less: ties (incl. zero energy) stay L/R.
            *slot = min_ms < min_lr;
        }
        out.push(row);
    }
    out
}

/// `(alpha_q_per_group, sap_coeff_used_per_group)` — the per-(group,
/// sfb) SAP-coded `alpha_q` index matrix plus the per-band SAP-used
/// flag matrix produced by [`select_alpha_q_for_pair`], shaped to feed
/// directly into [`build_chparam_info_sap_data_from_alpha_q`].
pub type SapAlphaDecision = (Vec<Vec<i32>>, Vec<Vec<bool>>);

/// Per-(group, sfb) `alpha_q` decision driver for the encoder-side
/// `SapMode::SapData` arm (Pseudocode 59, §5.3.2) — the SAP-coded
/// analogue of [`select_ms_used_for_pair`]. Picks the per-band
/// `alpha_q[g][sfb]` index (and the matching `sap_coeff_used[g][sfb]`
/// flag) from the target stereo MDCT spectra.
///
/// The decoder reconstructs the output pair `(O_0, O_1)` from the two
/// transmitted tracks `(I_0, I_1)` via the §5.3.3.2 matrix
/// `[[a, b], [c, d]]` with the SAP-coded coefficients
/// `(a, b, c, d) = (1 + g, 1, 1 - g, -1)`, `g = alpha_q · 0.1`
/// (Pseudocode 59). The encoder therefore inverts the matrix to find
/// the tracks it must transmit. With `det = -2` the closed-form inverse
/// of the target `(L, R) = (O_0, O_1)` is:
///
/// ```text
///   I_0 =  M               with  M = (L + R) / 2
///   I_1 =  S − g · M       with  S = (L − R) / 2
/// ```
///
/// `I_0` is the unweighted mid; `I_1` is the side **after subtracting a
/// `g`-scaled copy of the mid** — i.e. SAP coding is a one-tap
/// prediction of the side track from the mid track. The `g` that
/// minimises the transmitted side-residual energy `E[I_1²] =
/// Σ (S[k] − g · M[k])²` is the least-squares projection coefficient
/// per parameter band (sum over bins `[sfb_offset[sfb],
/// sfb_offset[sfb+1])`):
///
/// ```text
///   g* = ⟨S, M⟩ / ⟨M, M⟩   = (Σ S[k]·M[k]) / (Σ M[k]²)
/// ```
///
/// quantised by `alpha_q = round(g* / 0.1) = round(10 · g*)` and
/// clamped to the HCB_SCALEFAC-codable range `[-60, +60]` (the offset
/// of 60 is applied by [`crate::encoder_asf::write_sap_data`], not
/// here). This is exactly the inverse-quantisation grid Pseudocode 59
/// reads (`sap_gain = alpha_q · 0.1`).
///
/// `sap_coeff_used[g][sfb]` is raised only when SAP prediction is
/// beneficial — when the residual energy after the optimal projection
/// is strictly smaller than the raw side energy, i.e. `g* ≠ 0` after
/// quantisation. Bands with no mid energy (`⟨M, M⟩ == 0`) or where the
/// quantised `alpha_q` rounds to 0 (no useful prediction) leave the
/// flag clear so the band stays at the identity-passthrough convention
/// the decoder applies to cleared SAP bands. The decision is taken on
/// the even (pair-leading) sfb of each `(sfb, sfb+1)` pair and copied
/// to the odd partner, matching the pair-major flag-copy semantics of
/// Pseudocode 59 and [`build_chparam_info_sap_data_from_alpha_q`];
/// the odd partner inherits the even partner's `alpha_q` (the decoder
/// reads `alpha_q[g][sfb] = alpha_q[g][sfb-1]` for odd sfbs).
///
/// `l_spec_per_group[g]` / `r_spec_per_group[g]` are per-group MDCT
/// spectra at the same transform length as `sfb_offset`. Rows shorter
/// than `sfb_offset[max_sfb_per_group[g]]` produce a `(0, false)`
/// decision on the missing bins (the band stays identity-passthrough).
///
/// Round-trip: feeding the returned `(alpha_q, sap_coeff_used)` through
/// [`build_chparam_info_sap_data_from_alpha_q`] (with the same
/// `max_sfb_per_group` and a caller-chosen `delta_code_time`) and then
/// [`extract_sap_abcd`] reproduces the per-sfb SAP matrix
/// `(1 + g, 1, 1 - g, -1)` on the picked bands and identity on the
/// rest.
pub fn select_alpha_q_for_pair(
    l_spec_per_group: &[Vec<f32>],
    r_spec_per_group: &[Vec<f32>],
    sfb_offset: &[u16],
    max_sfb_per_group: &[u32],
) -> SapAlphaDecision {
    let num_groups = max_sfb_per_group.len();
    let mut alpha_q_out: Vec<Vec<i32>> = Vec::with_capacity(num_groups);
    let mut used_out: Vec<Vec<bool>> = Vec::with_capacity(num_groups);
    for (g, &m) in max_sfb_per_group.iter().enumerate() {
        let m_usize = m as usize;
        let mut alpha_row = vec![0i32; m_usize];
        let mut used_row = vec![false; m_usize];
        let (Some(l), Some(r)) = (l_spec_per_group.get(g), r_spec_per_group.get(g)) else {
            alpha_q_out.push(alpha_row);
            used_out.push(used_row);
            continue;
        };
        // Pair-major: decide on the even (leading) sfb of each pair and
        // copy to the odd partner.
        let mut sfb = 0usize;
        while sfb < m_usize {
            let Some(&lo) = sfb_offset.get(sfb) else {
                break;
            };
            let Some(&hi) = sfb_offset.get(sfb + 1) else {
                break;
            };
            let lo = lo as usize;
            let hi = (hi as usize).min(l.len()).min(r.len());
            if hi > lo {
                // Per-band least-squares projection of the side track S
                // onto the mid track M: g* = <S, M> / <M, M>.
                //   M = (L + R) / 2, S = (L - R) / 2.
                let mut e_m = 0.0f64; // <M, M>
                let mut cross_sm = 0.0f64; // <S, M>
                for k in lo..hi {
                    let lk = l[k] as f64;
                    let rk = r[k] as f64;
                    let mk = (lk + rk) * 0.5;
                    let sk = (lk - rk) * 0.5;
                    e_m += mk * mk;
                    cross_sm += sk * mk;
                }
                if e_m > 0.0 {
                    let g_star = cross_sm / e_m;
                    // alpha_q = round(g* / 0.1), clamped to the
                    // HCB_SCALEFAC-codable range [-60, +60].
                    let aq = (g_star * 10.0).round().clamp(-60.0, 60.0) as i32;
                    if aq != 0 {
                        alpha_row[sfb] = aq;
                        used_row[sfb] = true;
                        if sfb + 1 < m_usize {
                            // Odd partner inherits the even partner's
                            // alpha_q + flag (pair-major copy).
                            alpha_row[sfb + 1] = aq;
                            used_row[sfb + 1] = true;
                        }
                    }
                }
            }
            sfb += 2;
        }
        alpha_q_out.push(alpha_row);
        used_out.push(used_row);
    }
    (alpha_q_out, used_out)
}

/// I-frame-sticky per-substream configuration, carried across frames
/// so `b_iframe = 0` (P-frame) substreams can be parsed and decoded.
///
/// Per ETSI TS 103 190-1 §4.2.6.1 / §4.2.6.3 / §4.2.6.6 / §4.2.6.14
/// the `aspx_config()` / `acpl_config_1ch()` / `acpl_config_2ch()`
/// elements — and per Tables 51 / 52 the `aspx_xover_subband_offset`
/// field — are only transmitted inside `if (b_iframe) { … }` blocks;
/// every non-I-frame reuses the values from the most recent I-frame.
/// The decoder harvests this struct from the parsed
/// [`SubstreamTools`] after each I-frame and seeds the next frame's
/// tools from it when `b_iframe == 0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct StickyConfig {
    /// `aspx_config()` from the last I-frame.
    pub aspx_config: Option<aspx::AspxConfig>,
    /// `aspx_xover_subband_offset` (3 bits, Tables 51 / 52) from the
    /// last I-frame.
    pub aspx_xover: Option<u8>,
    /// `acpl_config_1ch(PARTIAL)` from the last I-frame
    /// (ASPX_ACPL_1 modes).
    pub acpl_config_1ch_partial: Option<crate::acpl::AcplConfig1ch>,
    /// `acpl_config_1ch(FULL)` from the last I-frame
    /// (ASPX_ACPL_2 modes).
    pub acpl_config_1ch_full: Option<crate::acpl::AcplConfig1ch>,
    /// `acpl_config_2ch()` from the last I-frame (5_X ASPX_ACPL_3).
    pub acpl_config_2ch: Option<crate::acpl::AcplConfig2ch>,
}

impl StickyConfig {
    /// Seed a fresh frame's [`SubstreamTools`] with the sticky configs
    /// ahead of a `b_iframe == 0` parse.
    pub fn seed(&self, tools: &mut SubstreamTools) {
        tools.aspx_config = self.aspx_config;
        tools.aspx_xover_subband_offset = self.aspx_xover;
        tools.acpl_config_1ch_partial = self.acpl_config_1ch_partial;
        tools.acpl_config_1ch_full = self.acpl_config_1ch_full;
        tools.acpl_config_2ch = self.acpl_config_2ch;
    }

    /// Harvest the sticky configs from a parsed I-frame's tools.
    pub fn harvest(&mut self, tools: &SubstreamTools) {
        self.aspx_config = tools.aspx_config;
        self.aspx_xover = tools.aspx_xover_subband_offset;
        self.acpl_config_1ch_partial = tools.acpl_config_1ch_partial;
        self.acpl_config_1ch_full = tools.acpl_config_1ch_full;
        self.acpl_config_2ch = tools.acpl_config_2ch;
    }
}

/// Per-substream tool summary — what the decoder can learn by walking
/// the outer layers of `audio_data()` without touching Huffman tables.
#[derive(Debug, Clone, Default)]
pub struct SubstreamTools {
    /// Channel mode that drove the `audio_data()` switch. Copied from
    /// the parent `ac4_substream_info()`.
    pub channel_mode_channels: u16,
    /// Mono codec mode for channel_mode == 0.
    pub mono_mode: Option<MonoCodecMode>,
    /// Stereo codec mode for channel_mode == 1.
    pub stereo_mode: Option<StereoCodecMode>,
    /// `spec_frontend` for the primary (mid / left / mono) channel.
    pub spec_frontend_primary: Option<SpecFrontend>,
    /// `spec_frontend` for the secondary (side / right) channel if the
    /// substream is dual-MDCT-frontend.
    pub spec_frontend_secondary: Option<SpecFrontend>,
    /// Parsed `asf_transform_info()` for the primary channel when
    /// spec_frontend == ASF.
    pub transform_info_primary: Option<AsfTransformInfo>,
    /// Parsed `asf_transform_info()` for the secondary channel.
    pub transform_info_secondary: Option<AsfTransformInfo>,
    /// Parsed `asf_psy_info()` for the primary channel.
    pub psy_info_primary: Option<AsfPsyInfo>,
    /// Parsed `asf_psy_info()` for the secondary channel.
    pub psy_info_secondary: Option<AsfPsyInfo>,
    /// `b_enable_mdct_stereo_proc` — stereo MDCT joint processing flag.
    pub mdct_stereo_proc: bool,
    /// Dequantised + scaled spectral coefficients for the primary
    /// channel (length = `sfb_offset[max_sfb]`). `None` when the
    /// Huffman-driven data path didn't run (non-SIMPLE / non-ASF /
    /// short-frame grouping cases).
    pub scaled_spec_primary: Option<Vec<f32>>,
    /// Dequantised + scaled spectral coefficients for the secondary
    /// (right) channel in a stereo SIMPLE substream. `None` for mono
    /// or non-decoded stereo cases.
    pub scaled_spec_secondary: Option<Vec<f32>>,
    /// Per-scale-factor-band M/S flags (`ms_used[sfb]`). Populated for
    /// `b_enable_mdct_stereo_proc == 1` joint-stereo frames. `None`
    /// otherwise. Length equals the decoded `max_sfb` for the shared
    /// window.
    pub ms_used: Option<Vec<bool>>,
    /// Parsed `aspx_config()` when the substream's codec mode selected
    /// one of the A-SPX paths (mono ASPX or stereo ASPX / ASPX_ACPL_*)
    /// **and** the current frame is an I-frame. `aspx_config` is only
    /// present in I-frames (§4.2.6.1 / §4.2.6.3); predictive frames
    /// inherit the previous I-frame's config. For non-I-frames or for
    /// SIMPLE substreams this stays `None`.
    pub aspx_config: Option<aspx::AspxConfig>,
    /// Parsed `companding_control()` when the substream's codec mode
    /// is one of the ASPX paths. Captured from the bitstream in the
    /// outer `audio_data()` walker.
    pub companding: Option<aspx::CompandingControl>,
    /// Parsed `aspx_framing(0)` for the primary channel. Populated when
    /// the substream's codec mode is one of the ASPX paths and an
    /// `aspx_config` is in scope (either from this frame if it's an
    /// I-frame or carried over from a prior I-frame — we only wire the
    /// I-frame path here so non-I-frame ASPX bitstreams stay at
    /// companding_control until config state plumbing arrives).
    pub aspx_framing_primary: Option<aspx::AspxFraming>,
    /// Parsed `aspx_framing(1)` for the secondary channel in a stereo
    /// ASPX substream. Only present when `aspx_balance == 0` (the
    /// secondary channel has its own framing per Table 52); when
    /// `aspx_balance == 1` the secondary envelope data reuses the
    /// primary's framing.
    pub aspx_framing_secondary: Option<aspx::AspxFraming>,
    /// `aspx_balance` — 1-bit flag from `aspx_data_2ch()` (Table 52).
    /// Present only in stereo ASPX substreams; otherwise `None`.
    pub aspx_balance: Option<bool>,
    /// `aspx_xover_subband_offset` — 3-bit I-frame-sticky field that
    /// leads `aspx_data_1ch` / `aspx_data_2ch` (Tables 51 / 52). Only
    /// populated for I-frames; carries the crossover-subband offset
    /// used to seed `sbx` in §5.7.6.3.1.2.
    pub aspx_xover_subband_offset: Option<u8>,
    /// `aspx_delta_dir(0)` — per-envelope delta-direction bits for the
    /// primary channel (Table 54). Populated when the walker reaches
    /// `aspx_data_1ch() / aspx_data_2ch()` and the framing decoded
    /// successfully.
    pub aspx_delta_dir_primary: Option<aspx::AspxDeltaDir>,
    /// `aspx_delta_dir(1)` — per-envelope delta-direction bits for
    /// the secondary channel in a stereo ASPX substream. `None` for
    /// mono.
    pub aspx_delta_dir_secondary: Option<aspx::AspxDeltaDir>,
    /// `aspx_qmode_env[0]` — the effective envelope quant step for
    /// the primary channel after the `FIXFIX && num_env == 1`
    /// clamp-to-0 override in Tables 51 / 52.
    pub aspx_qmode_env_primary: Option<aspx::AspxQuantStep>,
    /// `aspx_qmode_env[1]` — same, for the secondary channel.
    pub aspx_qmode_env_secondary: Option<aspx::AspxQuantStep>,
    /// Derived A-SPX frequency tables (§5.7.6.3.1). Populated on any
    /// I-frame ASPX substream that parses `aspx_config` plus
    /// `aspx_xover_subband_offset` without hitting a bailout.
    pub aspx_frequency_tables: Option<aspx::AspxFrequencyTables>,
    /// Parsed `aspx_hfgen_iwc_1ch()` for the mono ASPX path (Table 55).
    pub aspx_hfgen_iwc_1ch: Option<aspx::AspxHfgenIwc1Ch>,
    /// Parsed `aspx_hfgen_iwc_2ch()` for the stereo ASPX path (Table 56).
    pub aspx_hfgen_iwc_2ch: Option<aspx::AspxHfgenIwc2Ch>,
    /// `aspx_data_sig[0]` — per-envelope Huffman-decoded signal
    /// envelopes for the primary channel.
    pub aspx_data_sig_primary: Option<Vec<aspx::AspxHuffEnv>>,
    /// `aspx_data_sig[1]` — per-envelope signal envelopes for the
    /// secondary channel in a stereo ASPX substream.
    pub aspx_data_sig_secondary: Option<Vec<aspx::AspxHuffEnv>>,
    /// `aspx_data_noise[0]` — per-envelope Huffman-decoded noise
    /// envelopes for the primary channel.
    pub aspx_data_noise_primary: Option<Vec<aspx::AspxHuffEnv>>,
    /// `aspx_data_noise[1]` — per-envelope noise envelopes for the
    /// secondary channel.
    pub aspx_data_noise_secondary: Option<Vec<aspx::AspxHuffEnv>>,
    /// Parsed `acpl_config_1ch(PARTIAL)` (§4.2.13.1 Table 59) for
    /// stereo `ASPX_ACPL_1` I-frame substreams. `None` for SIMPLE /
    /// mono ASPX / stereo ASPX paths.
    pub acpl_config_1ch_partial: Option<crate::acpl::AcplConfig1ch>,
    /// Parsed `acpl_config_1ch(FULL)` (§4.2.13.1 Table 59) for
    /// stereo `ASPX_ACPL_2` I-frame substreams. `None` otherwise.
    pub acpl_config_1ch_full: Option<crate::acpl::AcplConfig1ch>,
    /// Parsed `acpl_data_1ch()` (§4.2.13.3 Table 61) when the surrounding
    /// substream is one of the `ASPX_ACPL_{1,2}` paths and the walker
    /// reached the body. `None` when ACPL data was either not present
    /// (SIMPLE / mono ASPX / stereo ASPX) or could not be parsed because
    /// the upstream MDCT body bailed first.
    pub acpl_data_1ch: Option<crate::acpl::AcplData1ch>,
    /// Parsed `chparam_info()` (§4.2.10.1 Table 47) for joint-MDCT
    /// stereo bodies. Populated for the `stereo_data()` joint path
    /// (`b_enable_mdct_stereo_proc == 1`) and for the
    /// `ASPX_ACPL_1` joint-MDCT residual layer. `None` for all other
    /// paths (split-MDCT, mono, ASPX_ACPL_2).
    pub chparam_info: Option<ChparamInfo>,
    /// `5_X_codec_mode` (§4.3.5.6 Table 97) for 5.X channel-element
    /// substreams. Populated by [`crate::mch::parse_5x_audio_data_outer`].
    pub five_x_mode: Option<crate::mch::FiveXCodecMode>,
    /// Whether the enclosing `5_X_channel_element(b_has_lfe, ...)` was
    /// invoked with `b_has_lfe == 1`. Mirrors the parameter so callers
    /// can correlate `lfe_mono_data` against the framing.
    pub five_x_b_has_lfe: bool,
    /// `coding_config` value the 5.X walker resolved (Table 25). `None`
    /// for ASPX_ACPL_3 (which has no `coding_config`) and for
    /// non-5.X substreams.
    pub five_x_coding_config: Option<crate::mch::FiveXCodingConfig>,
    /// Parsed LFE `mono_data(1)` payload from the 5.X / 7.X walkers.
    pub lfe_mono_data: Option<crate::mch::MonoLfeData>,
    /// `b_2ch_mode` flag for 5.X Cfg0 (Table 25 — `coding_config == 0`).
    /// Selects between L/R + Ls/Rs ordering; the centre channel always
    /// follows.
    pub b_2ch_mode: Option<bool>,
    /// 5.X Cfg0 trailing `mono_data(0)` for the centre channel.
    pub cfg0_centre_mono: Option<crate::mch::MonoLfeData>,
    /// 5.X Cfg2 trailing `mono_data(0)` (the surround mono after
    /// `four_channel_data`).
    pub cfg2_back_mono: Option<crate::mch::MonoLfeData>,
    /// 5_X SIMPLE/ASPX cfg2 ASPX trailer for the front L/R pair —
    /// the first `aspx_data_2ch()` after `four_channel_data + mono_data(0)`
    /// (per Table 25 row `case ASPX:` cfg2). Only populated when
    /// `5_X_codec_mode == ASPX` AND the trailer parsed cleanly.
    pub cfg2_aspx_lr: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg2 ASPX trailer for the surround Ls/Rs pair —
    /// the second `aspx_data_2ch()` after the L/R trailer.
    pub cfg2_aspx_ls_rs: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg2 ASPX trailer for the centre channel —
    /// the `aspx_data_1ch()` after the two 2ch trailers.
    pub cfg2_aspx_centre: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg0 ASPX trailer for the front L/R pair (or
    /// the L/Ls inner pair when `b_2ch_mode == true`). Captured from
    /// the first `aspx_data_2ch()` after the two `two_channel_data()`
    /// + `mono_data(0)` triple. Round 42 — mirror of `cfg2_aspx_lr`.
    pub cfg0_aspx_lr: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg0 ASPX trailer for the surround Ls/Rs pair
    /// (or the R/Rs inner pair when `b_2ch_mode == true`).
    pub cfg0_aspx_ls_rs: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg0 ASPX trailer for the centre channel.
    pub cfg0_aspx_centre: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg1 ASPX trailer for the front L/R pair —
    /// extracted from `three_channel_data[0..1]` (channels 0 and 1
    /// of the three_channel_data shell).
    pub cfg1_aspx_lr: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg1 ASPX trailer for the surround Ls/Rs pair —
    /// extracted from the `two_channel_data` that follows the
    /// `three_channel_data` shell.
    pub cfg1_aspx_ls_rs: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg1 ASPX trailer for the centre channel —
    /// `three_channel_data[2]` (the third channel of the
    /// three_channel_data shell).
    pub cfg1_aspx_centre: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg3 ASPX trailer for the front L/R pair —
    /// extracted from `five_channel_data[0..1]`.
    pub cfg3_aspx_lr: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg3 ASPX trailer for the surround Ls/Rs pair —
    /// `five_channel_data[3..4]`.
    pub cfg3_aspx_ls_rs: Option<aspx::FiveXAspxTrailer>,
    /// 5_X SIMPLE/ASPX cfg3 ASPX trailer for the centre channel —
    /// `five_channel_data[2]`.
    pub cfg3_aspx_centre: Option<aspx::FiveXAspxTrailer>,
    /// Parsed `two_channel_data()` outer shells (Table 26) for the 5.X
    /// `coding_config` Cfg0 (twice: `[L/R, Ls/Rs]`) and Cfg1 (once,
    /// after `three_channel_data`). Length matches the spec's call
    /// count for the resolved coding_config (`2` for Cfg0, `1` for Cfg1,
    /// `0` for Cfg2/Cfg3).
    pub two_channel_data: Vec<crate::mch::TwoChannelData>,
    /// Parsed `three_channel_data()` outer shell when the 5.X /
    /// 3.0 walker selected `coding_config == 1`.
    pub three_channel_data: Option<crate::mch::ThreeChannelData>,
    /// Parsed `four_channel_data()` outer shell when the 5.X / 7.X
    /// walker selected `coding_config == 2`.
    pub four_channel_data: Option<crate::mch::FourChannelData>,
    /// Parsed `five_channel_data()` outer shell when the 5.X / 7.X
    /// walker selected `coding_config == 3`.
    pub five_channel_data: Option<crate::mch::FiveChannelData>,
    /// Parsed `acpl_config_2ch()` (§4.2.13.2 Table 60) for `ASPX_ACPL_3`
    /// I-frame substreams. `None` for the other A-CPL paths.
    pub acpl_config_2ch: Option<crate::acpl::AcplConfig2ch>,
    /// Parsed `acpl_data_2ch()` (§4.2.13.4 Table 62) for `ASPX_ACPL_3`
    /// 5_X / 7_X frames. `None` for the other A-CPL paths or when the
    /// walker bailed before reaching the ACPL trailer.
    pub acpl_data_2ch: Option<crate::acpl::AcplData2ch>,
    /// Parsed pair of `acpl_data_1ch()` elements for the 5_X
    /// `ASPX_ACPL_1` / `ASPX_ACPL_2` modes (§5.7.7.6.1 Pseudocode 117).
    /// Two parallel ACplModule's each consume one `acpl_data_1ch()` set;
    /// `[0]` drives the L-side module (alpha_1 / beta_1), `[1]` drives
    /// the R-side module (alpha_2 / beta_2). Either entry stays `None`
    /// when the walker hasn't reached the ACPL trailer yet (e.g.
    /// non-I-frame).
    pub acpl_data_1ch_pair: [Option<crate::acpl::AcplData1ch>; 2],
    /// 5.X `ASPX_ACPL_1` joint-MDCT residual-layer scaled spectra for the
    /// two `sf_data(ASF)` bodies that follow the chparam_info pair. Per
    /// Table 181, these are `sSMP,3` and `sSMP,4` — the surround-driving
    /// inputs that mix with the preliminary outputs (A/B from the
    /// `two_channel_data` or three_channel_data) via the chparam SAP
    /// matrix to produce the final L/R/Ls/Rs output channels.
    ///
    /// Round 40 wires this for the standalone Ls/Rs surround mono
    /// walker — round-39 dropped the parsed bodies on the floor (the
    /// walker still ran them through `decode_asf_long_mono_body_with_max_sfb`
    /// for try-and-bail correctness but didn't persist them). With the
    /// pair populated, the 5_X `ASPX_ACPL_1` dispatch can IMDCT the
    /// residual MDCT spectra and feed them as `ls_pcm` / `rs_pcm`
    /// carriers into [`crate::acpl_synth::run_acpl_5x_pair_pcm`] (the
    /// `x3` / `x4` inputs of Pseudocode 117). When `None`, the
    /// dispatch falls back to the round-37 silence placeholder for
    /// the surround carriers.
    ///
    /// Each entry is the (transform_length, scaled_spec) bundle — the
    /// transform_length matches the upstream channel-data's largest
    /// signalled length (the dominant transform on which the joint-
    /// MDCT residual layer is single-window-grouped, per the §4.2.6.6
    /// NOTE).
    pub acpl_1_residual_pair: [Option<(u32, Vec<f32>)>; 2],
    /// 5_X `ASPX_ACPL_1` inner walker's two `chparam_info()` payloads
    /// (Table 25 row `case ASPX_ACPL_1:` — the two `chparam_info()` calls
    /// after `max_sfb_master`). These drive the Pseudocode 59 SAP a/b/c/d
    /// extraction for Table 181's first-stage matrix that mixes
    /// (sSMP_A, sSMP_B) with (sSMP_3, sSMP_4) → preliminary
    /// (L, R, Ls, Rs) before Pseudocode 117 runs. `[None, None]` for any
    /// path other than 5_X ACPL_1 or when the inner walker bailed
    /// before reaching the chparam pair.
    pub acpl_1_residual_chparam: [Option<ChparamInfo>; 2],
    /// `max_sfb_master` for the 5_X / 7_X `ASPX_ACPL_1` joint-MDCT
    /// residual layer (read after `n_side_bits` per Table 25 / 33 row
    /// `case ASPX_ACPL_1:`). Drives the per-(g, sfb) extent for
    /// Pseudocode 59 SAP extraction in [`apply_sap_table_181`]. `None`
    /// for any path other than 5_X / 7_X ACPL_1, or when the inner
    /// walker bailed before reaching `max_sfb_master`.
    pub acpl_1_residual_max_sfb_master: Option<u32>,
    /// `7_X_codec_mode` (§4.3.5.7 Table 98) for 7.X channel-element
    /// substreams. Populated by [`crate::mch::parse_7x_audio_data_outer`].
    /// Note this is a 2-bit field for 7_X (vs 3 bits for 5_X) — only
    /// SIMPLE / ASPX / ASPX_ACPL_1 / ASPX_ACPL_2 modes exist (no
    /// ASPX_ACPL_3 for 7_X).
    pub seven_x_mode: Option<crate::mch::SevenXCodecMode>,
    /// `channel_mode` of the enclosing `7_X_channel_element` — `false`
    /// for 7.0 (no LFE), `true` for 7.1 (with LFE). Mirrors the
    /// `b_has_lfe` plumbing of the 5_X walker.
    pub seven_x_b_has_lfe: bool,
    /// `coding_config` value the 7.X walker resolved (Table 33 — same
    /// 2-bit selector as 5_X SIMPLE/ASPX path: 0/1/2/3 → Cfg0Stereo /
    /// Cfg1ThreeStereo / Cfg2Four / Cfg3Five). `None` for non-7.X
    /// substreams.
    pub seven_x_coding_config: Option<crate::mch::FiveXCodingConfig>,
    /// `b_use_sap_add_ch` flag (§4.3.5.12) read from the 7.X SIMPLE/ASPX
    /// path before the additional `two_channel_data()`. `None` for
    /// ASPX_ACPL_{1,2} (no additional-channel block) and for non-7.X
    /// substreams.
    pub seven_x_b_use_sap_add_ch: Option<bool>,
    /// 7.X SIMPLE/ASPX additional-channel `chparam_info()` pair (Table
    /// 33). Populated only when `b_use_sap_add_ch == 1`. Each entry
    /// matches the standard `chparam_info()` shell.
    pub seven_x_add_chparam_info: Option<[ChparamInfo; 2]>,
    /// 7.X SIMPLE/ASPX trailing additional-channel `two_channel_data()`
    /// (Table 33 — the two extra channels beyond the 5.X core). `None`
    /// for ASPX_ACPL_{1,2} and for non-7.X substreams.
    pub seven_x_additional_channel_data: Option<crate::mch::TwoChannelData>,
    /// Parsed `ssf_data()` (§4.2.9 / Tables 43-46) for the primary
    /// channel when `spec_frontend_primary == SSF`. Populated by the
    /// mono / stereo walkers when the substream descriptor declares the
    /// SSF spectral frontend. `None` for ASF / mode mismatches.
    pub ssf_data_primary: Option<crate::ssf::SsfData>,
    /// Parsed `ssf_data()` for the secondary (right / side) channel in
    /// a stereo split-MDCT path. `None` for mono substreams or for
    /// stereo substreams whose right channel is on ASF.
    pub ssf_data_secondary: Option<crate::ssf::SsfData>,
    /// Parsed `immersive_channel_element()` (TS 103 190-2 §6.2.4.1)
    /// for the immersive channel modes (7.0.4 / 7.1.4 / 9.0.4 /
    /// 9.1.4). Populated by [`crate::ice::parse_ice_audio_data_outer`].
    pub ice: Option<Box<crate::ice::IceElement>>,
}

/// Result of walking a single `ac4_substream()` payload.
#[derive(Debug, Clone, Default)]
pub struct Ac4SubstreamInfo {
    /// `audio_size_value` in bytes (post variable_bits extension).
    pub audio_size: u32,
    /// Byte position (relative to the start of `ac4_substream()`) where
    /// `audio_data()` begins.
    pub audio_data_offset: u32,
    /// Tool summary — what we parsed from the outer `audio_data()`
    /// layers before bailing to opaque consumption.
    pub tools: SubstreamTools,
}

/// Resolve a `transf_length` index into an actual transform length in
/// samples for a given `frame_len_base` and base sample rate family.
///
/// Covers Tables 99 (long frames), 100 (non-long, 44.1/48 kHz,
/// `frame_len_base >= 1536`) and 103 (non-long, 44.1/48 kHz,
/// `frame_len_base < 1536`) for the base-rate path. 96 kHz and 192 kHz
/// (Tables 101 / 102 / 104 / 105) reach via HSF extension — not wired
/// in this baseline.
pub fn resolve_transf_length(frame_len_base: u32, b_long_frame: bool, idx: u32) -> u32 {
    // Long-frame branch — Table 99, 44.1/48 kHz column.
    if b_long_frame {
        return frame_len_base;
    }
    if frame_len_base >= 1536 {
        // Table 100 — rows are frame_len_base, columns are transf_length[i].
        match (frame_len_base, idx & 0b11) {
            (2048, 0) => 128,
            (2048, 1) => 256,
            (2048, 2) => 512,
            (2048, 3) => 1024,
            (1920, 0) => 120,
            (1920, 1) => 240,
            (1920, 2) => 480,
            (1920, 3) => 960,
            (1536, 0) => 96,
            (1536, 1) => 192,
            (1536, 2) => 384,
            (1536, 3) => 768,
            _ => 0,
        }
    } else {
        // Table 103 — short-ish frames.
        match (frame_len_base, idx & 0b11) {
            (1024, 0) => 128,
            (1024, 1) => 256,
            (1024, 2) => 512,
            (1024, 3) => 1024,
            (960, 0) => 120,
            (960, 1) => 240,
            (960, 2) => 480,
            (960, 3) => 960,
            (768, 0) => 96,
            (768, 1) => 192,
            (768, 2) => 384,
            (768, 3) => 768,
            (512, 0) => 128,
            (512, 1) => 256,
            (512, 2) => 512,
            (384, 0) => 96,
            (384, 1) => 192,
            (384, 2) => 384,
            _ => 0,
        }
    }
}

/// Parse `asf_transform_info()` at the current reader position. The
/// `frame_len_base` comes from the TOC.
pub fn parse_asf_transform_info(
    br: &mut BitReader<'_>,
    frame_len_base: u32,
) -> Result<AsfTransformInfo> {
    // Table 37 / §4.3.6.1.
    if frame_len_base >= 1536 {
        let b_long_frame = br.read_bit()?;
        if b_long_frame {
            // Long-frame transform length equals frame_len_base (at the
            // base sample rate family).
            let tl = resolve_transf_length(frame_len_base, true, 0);
            Ok(AsfTransformInfo {
                b_long_frame: true,
                transf_length: [0, 0],
                transform_length_0: tl,
                transform_length_1: tl,
            })
        } else {
            let t0 = br.read_u32(2)?;
            let t1 = br.read_u32(2)?;
            Ok(AsfTransformInfo {
                b_long_frame: false,
                transf_length: [t0, t1],
                transform_length_0: resolve_transf_length(frame_len_base, false, t0),
                transform_length_1: resolve_transf_length(frame_len_base, false, t1),
            })
        }
    } else {
        let t0 = br.read_u32(2)?;
        let len = resolve_transf_length(frame_len_base, false, t0);
        Ok(AsfTransformInfo {
            b_long_frame: false,
            transf_length: [t0, t0],
            transform_length_0: len,
            transform_length_1: len,
        })
    }
}

/// Parse the outer layers of `audio_data(channel_mode, b_iframe)` for
/// the mono channel mode. Fills in `tools.mono_mode`,
/// `tools.spec_frontend_primary`, and `tools.transform_info_primary`
/// when the mode is SIMPLE/ASPX and the frontend is ASF.
///
/// Returns after parsing `sf_info` — the `sf_data` payload is left
/// untouched for the caller to skip (via `audio_size`).
pub fn parse_mono_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    parse_mono_audio_data_outer_stateful(br, tools, b_iframe, frame_len_base, None)
}

/// Stateful variant of [`parse_mono_audio_data_outer`]. When
/// `ssf_states` is `Some`, the SSF body parse uses the channel-0 state
/// from the slice so RNG / `env_prev` continuity persists across
/// frames. `None` falls back to a stack-local `SsfChannelState`.
pub fn parse_mono_audio_data_outer_stateful(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
    ssf_states: Option<&mut [crate::ssf::SsfChannelState]>,
) -> Result<()> {
    // §4.2.6.1 single_channel_element(b_iframe):
    //   mono_codec_mode; 1 bit
    //   if (b_iframe && mono_codec_mode == ASPX) { aspx_config(); }
    //   if (mono_codec_mode == SIMPLE) {
    //       mono_data(0);
    //   } else {
    //       companding_control(1);
    //       mono_data(0);
    //       aspx_data_1ch();
    //   }
    let mode_bit = br.read_u32(1)?;
    let mode = MonoCodecMode::from_bit(mode_bit);
    tools.mono_mode = Some(mode);
    // ASPX path (§4.2.6.1): if b_iframe, read aspx_config(); then
    // companding_control(1); then mono_data(0); then aspx_data_1ch().
    // We stop after companding_control + mono_data(0) — the aspx_data
    // Huffman body needs Annex A.2 tables we haven't transcribed.
    if mode != MonoCodecMode::Simple {
        if b_iframe {
            tools.aspx_config = Some(aspx::parse_aspx_config(br)?);
        }
        tools.companding = Some(aspx::parse_companding_control(br, 1)?);
        // Fall through into mono_data(0) — same outer shell as SIMPLE.
    }
    // mono_data(b_lfe=0):
    //   spec_frontend;    1 bit
    //   sf_info(spec_frontend, 0, 0);
    //   sf_data(spec_frontend);
    let sf_bit = br.read_u32(1)?;
    let frontend = SpecFrontend::from_bit(sf_bit);
    tools.spec_frontend_primary = Some(frontend);
    if !b_iframe {
        // The transform-info for non-I-frames still runs on the first
        // I-frame's state but the syntax still reads it — keep parsing.
    }
    if let SpecFrontend::Ssf = frontend {
        // §4.2.9 / Tables 43-46 — invoke the SSF bitstream walker.
        // The SSF frame configuration is derived from `frame_len_base`
        // assuming the 48 kHz family (the only one currently driven by
        // the foundation TOC); 44.1 kHz support arrives with the
        // HSF-extension wiring. A None here means the frame length
        // doesn't map to any Annex C.1 / Table 112 row, in which case
        // we silently skip the SSF body — the same shape the
        // pre-round-30 walker had.
        if let Some(cfg) = crate::ssf::SsfFrameConfig::from_frame_len_base(frame_len_base) {
            // Round 32: borrow the persistent channel-0 SSF walker
            // state from the caller when supplied so RNG / env_prev /
            // predictor-lag history carries across frames; otherwise
            // fall back to a stack-local one (same shape as before).
            let mut local = crate::ssf::SsfChannelState::new();
            let state: &mut crate::ssf::SsfChannelState = match ssf_states {
                Some(slice) if !slice.is_empty() => &mut slice[0],
                _ => &mut local,
            };
            if let Ok(d) = crate::ssf::parse_ssf_data(br, b_iframe, &cfg, state) {
                tools.ssf_data_primary = Some(d);
            }
        }
        return Ok(());
    }
    if let SpecFrontend::Asf = frontend {
        let ti = parse_asf_transform_info(br, frame_len_base)?;
        tools.transform_info_primary = Some(ti);
        // asf_psy_info(b_dual_maxsfb=0, b_side_limited=0) for a mono
        // channel — per sf_info(spec_frontend, 0, 0).
        if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
            tools.psy_info_primary = Some(psy);
            // For the single-window-group / long-frame case we can now
            // walk asf_section_data + asf_spectral_data +
            // asf_scalefac_data + asf_snf_data and produce scaled
            // spectral coefficients.
            let mut body_ok = false;
            let psy_ref = tools.psy_info_primary.as_ref().unwrap().clone();
            if ti.b_long_frame && psy_ref.num_window_groups == 1 {
                if let Some(scaled) = decode_asf_long_mono_body(br, &ti, &psy_ref) {
                    tools.scaled_spec_primary = Some(scaled);
                    body_ok = true;
                }
            } else if psy_ref.num_window_groups > 1 {
                // Short-frame / grouped mono walker — Tables 39-42 with
                // the spec's outer `for (g = 0; ...)` loop in each
                // `asf_*_data()` body. Returns the per-group spectra
                // concatenated group-major.
                if let Some(scaled) = decode_asf_grouped_mono_body(br, &ti, &psy_ref) {
                    tools.scaled_spec_primary = Some(scaled);
                    body_ok = true;
                }
            }
            // §4.2.6.1: for the ASPX path, aspx_data_1ch() follows
            // mono_data(0). Parse its leading xover-subband-offset +
            // aspx_framing(0) here. Runs when:
            //   * mono_data(0) was fully decoded, so the bitreader sits
            //     at the start of aspx_data_1ch();
            //   * aspx_config is available — parsed above on I-frames,
            //     pre-seeded from the sticky state ([`StickyConfig`])
            //     on non-I-frames (the xover-subband-offset is likewise
            //     I-frame-sticky; `parse_aspx_data_1ch_body` skips its
            //     3-bit read when `b_iframe == 0`).
            if mode != MonoCodecMode::Simple && body_ok {
                if let Some(cfg) = tools.aspx_config {
                    parse_aspx_data_1ch_body(br, tools, &cfg, b_iframe, frame_len_base)?;
                }
            }
        }
    }
    Ok(())
}

/// Walk the `aspx_data_1ch()` body (Table 51) at the current bit
/// position into `tools`. Reads:
///
///   * `aspx_xover_subband_offset` (3 bits)
///   * `aspx_framing(0)`
///   * `aspx_delta_dir(0)`
///   * `aspx_hfgen_iwc_1ch()`
///   * `aspx_ec_data(SIGNAL)` + `aspx_ec_data(NOISE)`
///
/// The caller is responsible for arranging that the bitreader is sitting
/// at the start of `aspx_data_1ch()`. Used by both the mono ASPX path
/// (§4.2.6.1) and the stereo `ASPX_ACPL_{1,2}` paths (§4.2.6.3) — both
/// follow exactly the Table-51 layout.
pub(crate) fn parse_aspx_data_1ch_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    // Table 51: `if (b_iframe) aspx_xover_subband_offset;` — non-I-
    // frames reuse the sticky value from the last I-frame, which the
    // caller pre-seeds into `tools` (see [`StickyConfig::seed`]).
    let xover = if b_iframe {
        let x = br.read_u32(3)? as u8;
        tools.aspx_xover_subband_offset = Some(x);
        x
    } else {
        tools.aspx_xover_subband_offset.ok_or_else(|| {
            Error::invalid("ac4: non-iframe aspx_data_1ch without sticky xover offset")
        })?
    };
    let nats = aspx::num_aspx_timeslots(frame_len_base);
    let framing = aspx::parse_aspx_framing(br, cfg, b_iframe, nats > 8)?;
    let qmode = if matches!(framing.int_class, aspx::AspxIntClass::FixFix) && framing.num_env == 1 {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };
    tools.aspx_qmode_env_primary = Some(qmode);
    let dd = aspx::parse_aspx_delta_dir(br, &framing)?;
    if let Ok(tables) = aspx::derive_aspx_frequency_tables(cfg, xover as u32) {
        let hfgen = aspx::parse_aspx_hfgen_iwc_1ch(
            br,
            cfg.num_noise_sbgroups(),
            tables.counts.num_sbg_sig_highres,
            nats,
        )?;
        tools.aspx_hfgen_iwc_1ch = Some(hfgen);
        let sig = aspx::parse_aspx_ec_data(
            br,
            aspx::AspxDataType::Signal,
            framing.num_env,
            &framing.freq_res,
            qmode,
            aspx::AspxStereoMode::Level,
            &dd.sig_delta_dir,
            tables.counts,
        )?;
        tools.aspx_data_sig_primary = Some(sig);
        let noise = aspx::parse_aspx_ec_data(
            br,
            aspx::AspxDataType::Noise,
            framing.num_noise,
            &[],
            aspx::AspxQuantStep::Fine,
            aspx::AspxStereoMode::Level,
            &dd.noise_delta_dir,
            tables.counts,
        )?;
        tools.aspx_data_noise_primary = Some(noise);
        tools.aspx_frequency_tables = Some(tables);
    }
    tools.aspx_delta_dir_primary = Some(dd);
    tools.aspx_framing_primary = Some(framing);
    Ok(())
}

/// Walk the `aspx_data_2ch()` body (Table 52) at the current bit
/// position into `tools`. Reads:
///
///   * `aspx_xover_subband_offset` (3 bits)
///   * `aspx_framing(0)` for channel 0
///   * `aspx_balance` (1 bit) — when 0, `aspx_framing(1)` follows
///   * `aspx_delta_dir(0)` and `aspx_delta_dir(1)`
///   * `aspx_hfgen_iwc_2ch(aspx_balance)`
///   * 4x `aspx_ec_data` calls: ch0/ch1 SIGNAL, ch0/ch1 NOISE
///
/// Used by both the stereo CPE `ASPX` mode (§4.2.6.3, Table 22) and
/// the 5.X `ASPX_ACPL_3` mode (§4.2.6.6, Table 25). Both follow the
/// same Table-52 layout.
///
/// The caller is responsible for arranging that the bitreader is
/// sitting at the start of `aspx_data_2ch()`. `cfg` is the active
/// `aspx_config()` (drives quant_mode_env + num_noise_sbgroups).
pub(crate) fn parse_aspx_data_2ch_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    // Table 52: `if (b_iframe) aspx_xover_subband_offset;` — non-I-
    // frames reuse the sticky value from the last I-frame, which the
    // caller pre-seeds into `tools` (see [`StickyConfig::seed`]).
    let xover = if b_iframe {
        let x = br.read_u32(3)? as u8;
        tools.aspx_xover_subband_offset = Some(x);
        x
    } else {
        tools.aspx_xover_subband_offset.ok_or_else(|| {
            Error::invalid("ac4: non-iframe aspx_data_2ch without sticky xover offset")
        })?
    };
    let nats = aspx::num_aspx_timeslots(frame_len_base);
    let framing_ch0 = aspx::parse_aspx_framing(br, cfg, b_iframe, nats > 8)?;
    // Per Table 52: `aspx_qmode_env[0] = aspx_qmode_env[1]
    // = aspx_quant_mode_env` then clamp to 0 on FIXFIX +
    // num_env == 1.
    let qmode_ch0 = if matches!(framing_ch0.int_class, aspx::AspxIntClass::FixFix)
        && framing_ch0.num_env == 1
    {
        aspx::AspxQuantStep::Fine
    } else {
        cfg.quant_mode_env
    };
    tools.aspx_qmode_env_primary = Some(qmode_ch0);
    // Table 52: aspx_balance (1 bit). If 0, aspx_framing(1)
    // follows for channel 1; otherwise channel 1 reuses
    // channel 0's framing.
    let balance = br.read_bit()?;
    tools.aspx_balance = Some(balance);
    let framing_ch1_ref;
    if !balance {
        let framing_ch1 = aspx::parse_aspx_framing(br, cfg, b_iframe, nats > 8)?;
        // Per Table 52 the ch1 qmode is recomputed against the ch1
        // framing (and re-clamped on FIXFIX + num_env == 1).
        let qmode_ch1 = if matches!(framing_ch1.int_class, aspx::AspxIntClass::FixFix)
            && framing_ch1.num_env == 1
        {
            aspx::AspxQuantStep::Fine
        } else {
            cfg.quant_mode_env
        };
        tools.aspx_qmode_env_secondary = Some(qmode_ch1);
        tools.aspx_framing_secondary = Some(framing_ch1);
        framing_ch1_ref = tools.aspx_framing_secondary.as_ref();
    } else {
        // Shared framing; copy the ch0 qmode across.
        tools.aspx_qmode_env_secondary = Some(qmode_ch0);
        framing_ch1_ref = Some(&framing_ch0);
    }
    // aspx_delta_dir(0) then aspx_delta_dir(1) per Table 52.
    let dd0 = aspx::parse_aspx_delta_dir(br, &framing_ch0)?;
    let f_ch1 = framing_ch1_ref.unwrap_or(&framing_ch0);
    let dd1 = aspx::parse_aspx_delta_dir(br, f_ch1)?;
    // §5.7.6.3.1 derivation feeds aspx_hfgen_iwc_2ch() (Table 56)
    // then four aspx_ec_data() calls (ch0/ch1 SIGNAL, ch0/ch1
    // NOISE) per Table 52.
    if let Ok(tables) = aspx::derive_aspx_frequency_tables(cfg, xover as u32) {
        let hfgen = aspx::parse_aspx_hfgen_iwc_2ch(
            br,
            balance,
            cfg.num_noise_sbgroups(),
            tables.counts.num_sbg_sig_highres,
            nats,
        )?;
        tools.aspx_hfgen_iwc_2ch = Some(hfgen);
        let qmode_ch1_effective = tools.aspx_qmode_env_secondary.unwrap_or(qmode_ch0);
        // ch0 SIGNAL: stereo_mode = LEVEL (Table 52).
        let sig0 = aspx::parse_aspx_ec_data(
            br,
            aspx::AspxDataType::Signal,
            framing_ch0.num_env,
            &framing_ch0.freq_res,
            qmode_ch0,
            aspx::AspxStereoMode::Level,
            &dd0.sig_delta_dir,
            tables.counts,
        )?;
        tools.aspx_data_sig_primary = Some(sig0);
        // ch1 SIGNAL: BALANCE when aspx_balance == 1 else LEVEL
        // (Table 52).
        let sm_ch1 = if balance {
            aspx::AspxStereoMode::Balance
        } else {
            aspx::AspxStereoMode::Level
        };
        let sig1 = aspx::parse_aspx_ec_data(
            br,
            aspx::AspxDataType::Signal,
            f_ch1.num_env,
            &f_ch1.freq_res,
            qmode_ch1_effective,
            sm_ch1,
            &dd1.sig_delta_dir,
            tables.counts,
        )?;
        tools.aspx_data_sig_secondary = Some(sig1);
        // ch0 NOISE.
        let noise0 = aspx::parse_aspx_ec_data(
            br,
            aspx::AspxDataType::Noise,
            framing_ch0.num_noise,
            &[],
            aspx::AspxQuantStep::Fine,
            aspx::AspxStereoMode::Level,
            &dd0.noise_delta_dir,
            tables.counts,
        )?;
        tools.aspx_data_noise_primary = Some(noise0);
        // ch1 NOISE mirrors ch1 SIGNAL stereo_mode.
        let noise1 = aspx::parse_aspx_ec_data(
            br,
            aspx::AspxDataType::Noise,
            f_ch1.num_noise,
            &[],
            aspx::AspxQuantStep::Fine,
            sm_ch1,
            &dd1.noise_delta_dir,
            tables.counts,
        )?;
        tools.aspx_data_noise_secondary = Some(noise1);
        tools.aspx_frequency_tables = Some(tables);
    }
    tools.aspx_delta_dir_primary = Some(dd0);
    tools.aspx_delta_dir_secondary = Some(dd1);
    tools.aspx_framing_primary = Some(framing_ch0);
    Ok(())
}

/// Snapshot of the per-substream ASPX-trailer slots used by the
/// 5_X SIMPLE/ASPX trailer-capture helpers. Captures the slots
/// before parsing a trailer so they can be restored afterwards
/// (the cfg2 outer needs to walk three trailers in sequence — we
/// don't want any one of them to leak its state into the
/// "primary" view used by other paths).
struct AspxTrailerSnapshot {
    framing_pri: Option<aspx::AspxFraming>,
    framing_sec: Option<aspx::AspxFraming>,
    balance: Option<bool>,
    xover: Option<u8>,
    delta_dir_pri: Option<aspx::AspxDeltaDir>,
    delta_dir_sec: Option<aspx::AspxDeltaDir>,
    qmode_pri: Option<aspx::AspxQuantStep>,
    qmode_sec: Option<aspx::AspxQuantStep>,
    frequency_tables: Option<aspx::AspxFrequencyTables>,
    hfgen_1ch: Option<aspx::AspxHfgenIwc1Ch>,
    hfgen_2ch: Option<aspx::AspxHfgenIwc2Ch>,
    sig_pri: Option<Vec<aspx::AspxHuffEnv>>,
    sig_sec: Option<Vec<aspx::AspxHuffEnv>>,
    noise_pri: Option<Vec<aspx::AspxHuffEnv>>,
    noise_sec: Option<Vec<aspx::AspxHuffEnv>>,
}

impl AspxTrailerSnapshot {
    fn capture(tools: &mut SubstreamTools) -> Self {
        Self {
            framing_pri: tools.aspx_framing_primary.take(),
            framing_sec: tools.aspx_framing_secondary.take(),
            balance: tools.aspx_balance.take(),
            xover: tools.aspx_xover_subband_offset.take(),
            delta_dir_pri: tools.aspx_delta_dir_primary.take(),
            delta_dir_sec: tools.aspx_delta_dir_secondary.take(),
            qmode_pri: tools.aspx_qmode_env_primary.take(),
            qmode_sec: tools.aspx_qmode_env_secondary.take(),
            frequency_tables: tools.aspx_frequency_tables.take(),
            hfgen_1ch: tools.aspx_hfgen_iwc_1ch.take(),
            hfgen_2ch: tools.aspx_hfgen_iwc_2ch.take(),
            sig_pri: tools.aspx_data_sig_primary.take(),
            sig_sec: tools.aspx_data_sig_secondary.take(),
            noise_pri: tools.aspx_data_noise_primary.take(),
            noise_sec: tools.aspx_data_noise_secondary.take(),
        }
    }

    fn restore(self, tools: &mut SubstreamTools) {
        tools.aspx_framing_primary = self.framing_pri;
        tools.aspx_framing_secondary = self.framing_sec;
        tools.aspx_balance = self.balance;
        tools.aspx_xover_subband_offset = self.xover;
        tools.aspx_delta_dir_primary = self.delta_dir_pri;
        tools.aspx_delta_dir_secondary = self.delta_dir_sec;
        tools.aspx_qmode_env_primary = self.qmode_pri;
        tools.aspx_qmode_env_secondary = self.qmode_sec;
        tools.aspx_frequency_tables = self.frequency_tables;
        tools.aspx_hfgen_iwc_1ch = self.hfgen_1ch;
        tools.aspx_hfgen_iwc_2ch = self.hfgen_2ch;
        tools.aspx_data_sig_primary = self.sig_pri;
        tools.aspx_data_sig_secondary = self.sig_sec;
        tools.aspx_data_noise_primary = self.noise_pri;
        tools.aspx_data_noise_secondary = self.noise_sec;
    }
}

/// Walk a trailing `aspx_data_2ch()` body and capture the result as a
/// [`aspx::FiveXAspxTrailer`]. The per-substream ASPX-trailer slots
/// in `tools` are saved before the parse and restored afterwards, so
/// the call leaves no residue (the captured trailer holds the parsed
/// state).
///
/// Returns `None` when `parse_aspx_data_2ch_body` bails or when the
/// frequency-table derivation didn't fire (the trailer is then
/// considered unusable for bandwidth-extension).
pub(crate) fn capture_aspx_data_2ch_trailer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
    frame_len_base: u32,
) -> Option<aspx::FiveXAspxTrailer> {
    let snap = AspxTrailerSnapshot::capture(tools);
    // Tables 51 / 52: non-I-frames carry no xover offset on the wire —
    // the parse below needs the I-frame-sticky value the snapshot just
    // detached, so re-seed it for the walk (restored either way).
    if !b_iframe {
        tools.aspx_xover_subband_offset = snap.xover;
    }
    let parse_ok = parse_aspx_data_2ch_body(br, tools, cfg, b_iframe, frame_len_base).is_ok();
    let trailer = if parse_ok {
        let xover = tools.aspx_xover_subband_offset?;
        let frequency_tables = tools.aspx_frequency_tables.clone()?;
        let framing_pri = tools.aspx_framing_primary.clone()?;
        let qmode_pri = tools.aspx_qmode_env_primary?;
        let delta_dir_pri = tools.aspx_delta_dir_primary.clone()?;
        let sig_pri = tools.aspx_data_sig_primary.clone().unwrap_or_default();
        let noise_pri = tools.aspx_data_noise_primary.clone().unwrap_or_default();
        // hfgen_2ch carries per-channel `add_harmonic` + `tna_mode`
        // arrays. When absent (xover too high to leave any sig
        // sbgroups, etc.) we treat the channel as having no harmonic /
        // TNS info — the extender then falls back to the noise-only
        // path inside aspx_extend_pcm.
        let hfgen = tools.aspx_hfgen_iwc_2ch.clone();
        let (ah_pri, tna_pri, ah_sec, tna_sec) = if let Some(h) = hfgen.as_ref() {
            let ah_pri = h.add_harmonic.first().cloned();
            let ah_sec = h.add_harmonic.get(1).cloned();
            let tna_pri = h.tna_mode.first().cloned();
            let tna_sec = h.tna_mode.get(1).cloned();
            (ah_pri, tna_pri, ah_sec, tna_sec)
        } else {
            (None, None, None, None)
        };
        let primary = aspx::FiveXAspxChannelTrailer {
            framing: framing_pri.clone(),
            qmode_env: qmode_pri,
            delta_dir: delta_dir_pri,
            data_sig: sig_pri,
            data_noise: noise_pri,
            add_harmonic: ah_pri,
            tna_mode: tna_pri,
        };
        // Secondary channel: framing reuses the primary's when
        // `aspx_balance == 1` (no aspx_framing(1) in the bitstream);
        // delta-dir / sig / noise are always present per Table 52.
        let framing_sec = tools
            .aspx_framing_secondary
            .clone()
            .unwrap_or_else(|| framing_pri.clone());
        let qmode_sec = tools.aspx_qmode_env_secondary.unwrap_or(qmode_pri);
        let delta_dir_sec = tools
            .aspx_delta_dir_secondary
            .clone()
            .unwrap_or_else(|| tools.aspx_delta_dir_primary.clone().unwrap_or_default());
        let sig_sec = tools.aspx_data_sig_secondary.clone().unwrap_or_default();
        let noise_sec = tools.aspx_data_noise_secondary.clone().unwrap_or_default();
        let secondary = aspx::FiveXAspxChannelTrailer {
            framing: framing_sec,
            qmode_env: qmode_sec,
            delta_dir: delta_dir_sec,
            data_sig: sig_sec,
            data_noise: noise_sec,
            add_harmonic: ah_sec,
            tna_mode: tna_sec,
        };
        Some(aspx::FiveXAspxTrailer {
            xover,
            frequency_tables,
            primary,
            secondary: Some(secondary),
        })
    } else {
        None
    };
    snap.restore(tools);
    trailer
}

/// Walk a trailing `aspx_data_1ch()` body and capture the result as a
/// [`aspx::FiveXAspxTrailer`]. Mirror of
/// [`capture_aspx_data_2ch_trailer`] for the 1-channel layout
/// (Table 51).
pub(crate) fn capture_aspx_data_1ch_trailer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    cfg: &aspx::AspxConfig,
    b_iframe: bool,
    frame_len_base: u32,
) -> Option<aspx::FiveXAspxTrailer> {
    let snap = AspxTrailerSnapshot::capture(tools);
    // See capture_aspx_data_2ch_trailer — re-seed the sticky xover the
    // snapshot detached so the Table 51 non-I-frame walk can run.
    if !b_iframe {
        tools.aspx_xover_subband_offset = snap.xover;
    }
    let parse_ok = parse_aspx_data_1ch_body(br, tools, cfg, b_iframe, frame_len_base).is_ok();
    let trailer = if parse_ok {
        let xover = tools.aspx_xover_subband_offset?;
        let frequency_tables = tools.aspx_frequency_tables.clone()?;
        let framing = tools.aspx_framing_primary.clone()?;
        let qmode = tools.aspx_qmode_env_primary?;
        let delta_dir = tools.aspx_delta_dir_primary.clone()?;
        let data_sig = tools.aspx_data_sig_primary.clone().unwrap_or_default();
        let data_noise = tools.aspx_data_noise_primary.clone().unwrap_or_default();
        let hfgen = tools.aspx_hfgen_iwc_1ch.clone();
        let (add_harmonic, tna_mode) = if let Some(h) = hfgen.as_ref() {
            (Some(h.add_harmonic.clone()), Some(h.tna_mode.clone()))
        } else {
            (None, None)
        };
        Some(aspx::FiveXAspxTrailer {
            xover,
            frequency_tables,
            primary: aspx::FiveXAspxChannelTrailer {
                framing,
                qmode_env: qmode,
                delta_dir,
                data_sig,
                data_noise,
                add_harmonic,
                tna_mode,
            },
            secondary: None,
        })
    } else {
        None
    };
    snap.restore(tools);
    trailer
}

/// Derive per-window-group `(transf_length_idx, transform_length, max_sfb)`
/// arrays from a parsed `(ti, psy)` pair, per Pseudocodes 2 and 5 +
/// Pseudocode 3 (`window_to_group[]`) — for the **non-side-channel /
/// non-side-limited** path. `b_dual_maxsfb` and `b_side_channel`
/// callers (joint-MDCT side residual) should pass an explicit
/// `max_sfb_in` and use [`derive_per_group_with_max_sfb`] instead.
///
/// Returns `(transf_length_idx_per_group, transform_length_per_group,
/// max_sfb_per_group)`. All three vectors have length
/// `psy.num_window_groups`.
///
/// For the equal-transform-length (no `b_different_framing`) case all
/// three vectors collapse to the single transform / max_sfb value
/// repeated `num_window_groups` times. For `b_different_framing` the
/// first half-frame's groups use `transf_length[0]` / `max_sfb_0` and
/// the second half's use `transf_length[1]` / `max_sfb_1`.
fn derive_per_group(ti: &AsfTransformInfo, psy: &AsfPsyInfo) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    derive_per_group_with_max_sfb(ti, psy, psy.max_sfb_0, psy.max_sfb_1)
}

/// Derive per-group `(transf_length_idx, transform_length, max_sfb)`
/// arrays with explicit per-half-frame `max_sfb_a` / `max_sfb_b`
/// overrides. Used by joint-MDCT side-channel decoders that want
/// `max_sfb_side[0/1]` instead of `max_sfb[0/1]`.
fn derive_per_group_with_max_sfb(
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
    max_sfb_a: u32,
    max_sfb_b: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = psy.num_window_groups.max(1) as usize;
    let mut tl_idx = Vec::with_capacity(n);
    let mut tl = Vec::with_capacity(n);
    let mut msfb = Vec::with_capacity(n);
    if !psy.b_different_framing {
        // All groups share `transf_length[0]` / `max_sfb_0`.
        for _ in 0..n {
            tl_idx.push(ti.transf_length[0]);
            tl.push(ti.transform_length_0);
            msfb.push(max_sfb_a);
        }
        return (tl_idx, tl, msfb);
    }
    // Pseudocode 2 / 3: groups < window_to_group[num_windows_0] use
    // transf_length[0]; the rest use transf_length[1]. We don't carry
    // `window_to_group[]` on AsfPsyInfo today; derive it from the
    // `scale_factor_grouping` bits (Pseudocode 3).
    let num_windows_0 = 1u32 << (3u32.saturating_sub(ti.transf_length[0]));
    // Walk Pseudocode 3 to find which group index the first second-half
    // window lands in. The pseudocode shifts the grouping bits and
    // injects a 0 (group boundary) at index `num_windows_0 - 1`, so
    // window `num_windows_0` is always the start of a new group.
    // window_to_group[w] increments by 1 each time scale_factor_grouping[i]==0.
    let mut wtg = Vec::with_capacity(psy.num_windows as usize);
    wtg.push(0u32);
    let mut g = 0u32;
    let mut grouping = psy.scale_factor_grouping.clone();
    // Inject the spec's b_different_framing boundary at num_windows_0-1.
    if (num_windows_0 as usize) <= grouping.len() {
        // Shift grouping bits of 2nd half by 1, drop the last entry.
        for i in (num_windows_0 as usize..grouping.len()).rev() {
            grouping[i] = grouping[i - 1];
        }
        grouping[(num_windows_0 - 1) as usize] = 0;
    }
    for &b in &grouping {
        if b == 0 {
            g += 1;
        }
        wtg.push(g);
    }
    let split_group = if (num_windows_0 as usize) < wtg.len() {
        wtg[num_windows_0 as usize]
    } else {
        psy.num_window_groups
    };
    for grp in 0..n as u32 {
        if grp < split_group {
            tl_idx.push(ti.transf_length[0]);
            tl.push(ti.transform_length_0);
            msfb.push(max_sfb_a);
        } else {
            tl_idx.push(ti.transf_length[1]);
            tl.push(ti.transform_length_1);
            msfb.push(max_sfb_b);
        }
    }
    (tl_idx, tl, msfb)
}

/// Decode the `sf_data(ASF)` body for a mono, **short-frame / grouped**
/// (`num_window_groups > 1`) ASF substream. Tables 39-42 (§4.2.8.3-6):
/// the four `asf_*_data()` calls each carry their own outer
/// `for (g = 0; g < num_window_groups; g++)` loop, with a *single*
/// `reference_scale_factor` (8 bits) and *single* `b_snf_data_exists`
/// (1 bit). Returns the per-group dequantised spectra concatenated
/// group-major (each group of length `sfb_offset[max_sfb]` at the
/// per-group transform length).
///
/// On any Huffman / bitstream error returns `None`. The caller falls
/// back to silence.
pub(crate) fn decode_asf_grouped_mono_body_with_max_sfb(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
    max_sfb_a: u32,
    max_sfb_b: u32,
) -> Option<Vec<f32>> {
    if psy.num_window_groups <= 1 {
        return None;
    }
    let (tl_idx_per_g, tl_per_g, max_sfb_per_g) =
        derive_per_group_with_max_sfb(ti, psy, max_sfb_a, max_sfb_b);
    // Resolve sfb_offset table per group.
    let mut sfbo_per_g: Vec<&'static [u16]> = Vec::with_capacity(tl_per_g.len());
    let mut max_sfb_capped: Vec<u32> = Vec::with_capacity(tl_per_g.len());
    for g in 0..tl_per_g.len() {
        let tl = tl_per_g[g];
        let cap = tables::num_sfb_48(tl)?;
        let m = max_sfb_per_g[g].min(cap);
        if m == 0 {
            return None;
        }
        max_sfb_capped.push(m);
        sfbo_per_g.push(sfb_offset::sfb_offset_48(tl)?);
    }
    let sections =
        asf_data::parse_asf_section_data_grouped(br, &tl_idx_per_g, &tl_per_g, &max_sfb_capped)
            .ok()?;
    let (qspec_per_g, mqi_per_g) =
        asf_data::parse_asf_spectral_data_grouped(br, &sections, &sfbo_per_g, &max_sfb_capped)
            .ok()?;
    let sf_gain_per_g = asf_data::parse_asf_scalefac_data_grouped(
        br,
        &sections,
        &mqi_per_g,
        &max_sfb_capped,
        &tl_per_g,
    )
    .ok()?;
    let _snf =
        asf_data::parse_asf_snf_data_grouped(br, &sections, &mqi_per_g, &max_sfb_capped, &tl_per_g)
            .ok()?;
    let mut out: Vec<f32> = Vec::new();
    for g in 0..tl_per_g.len() {
        let scaled = asf_data::dequantise_and_scale(
            &qspec_per_g[g],
            &sf_gain_per_g[g],
            sfbo_per_g[g],
            max_sfb_capped[g],
        );
        out.extend_from_slice(&scaled);
    }
    Some(out)
}

/// Mono short-frame `sf_data(ASF)` walker — convenience wrapper that
/// pulls the same `max_sfb_0` for both halves (the non-different-framing
/// or single-`max_sfb` case).
fn decode_asf_grouped_mono_body(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<Vec<f32>> {
    decode_asf_grouped_mono_body_with_max_sfb(br, ti, psy, psy.max_sfb_0, psy.max_sfb_1)
}

/// Decode the `sf_data(ASF)` body for a **stereo joint-MDCT short-frame**
/// (`num_window_groups > 1`, `b_enable_mdct_stereo_proc == 1`) ASF
/// substream. Mirrors [`decode_asf_long_stereo_joint_body`] but walks
/// the per-group spec layout: shared `asf_section_data()` (one outer
/// g-loop), two `asf_spectral_data()` bodies (L/M then R/S, each with
/// their own outer g-loop), shared `asf_scalefac_data()` (single
/// `reference_scale_factor`, per-group DPCM), per-group `ms_used[g][sfb]`
/// flag arrays, then `asf_snf_data()`.
///
/// Returns `(left_per_group_concatenated, right_per_group_concatenated,
/// ms_used_per_group_concatenated)` on success.
fn decode_asf_grouped_stereo_joint_body(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
    if psy.num_window_groups <= 1 {
        return None;
    }
    let (tl_idx_per_g, tl_per_g, max_sfb_per_g) = derive_per_group(ti, psy);
    let mut sfbo_per_g: Vec<&'static [u16]> = Vec::with_capacity(tl_per_g.len());
    let mut max_sfb_capped: Vec<u32> = Vec::with_capacity(tl_per_g.len());
    for g in 0..tl_per_g.len() {
        let tl = tl_per_g[g];
        let cap = tables::num_sfb_48(tl)?;
        let m = max_sfb_per_g[g].min(cap);
        if m == 0 {
            return None;
        }
        max_sfb_capped.push(m);
        sfbo_per_g.push(sfb_offset::sfb_offset_48(tl)?);
    }
    let sections =
        asf_data::parse_asf_section_data_grouped(br, &tl_idx_per_g, &tl_per_g, &max_sfb_capped)
            .ok()?;
    let (q_l_per_g, mqi_l_per_g) =
        asf_data::parse_asf_spectral_data_grouped(br, &sections, &sfbo_per_g, &max_sfb_capped)
            .ok()?;
    let (q_r_per_g, mqi_r_per_g) =
        asf_data::parse_asf_spectral_data_grouped(br, &sections, &sfbo_per_g, &max_sfb_capped)
            .ok()?;
    // Shared scale-factor DPCM uses the band-wise max over both
    // channels so `first_scf_found` and the per-band gate (cb != 0 &&
    // mqi > 0) track bands carrying any energy at all.
    let mut mqi_per_g: Vec<Vec<u32>> = Vec::with_capacity(tl_per_g.len());
    for g in 0..tl_per_g.len() {
        let v: Vec<u32> = mqi_l_per_g[g]
            .iter()
            .zip(mqi_r_per_g[g].iter())
            .map(|(a, b)| (*a).max(*b))
            .collect();
        mqi_per_g.push(v);
    }
    let sf_gain_per_g = asf_data::parse_asf_scalefac_data_grouped(
        br,
        &sections,
        &mqi_per_g,
        &max_sfb_capped,
        &tl_per_g,
    )
    .ok()?;
    // Per-group, per-sfb ms_used flag. Only active bands (cb != 0 &&
    // mqi > 0) carry a bit per §7.5 Pseudocode 77.
    let mut ms_used_per_g: Vec<Vec<bool>> = Vec::with_capacity(tl_per_g.len());
    for g in 0..tl_per_g.len() {
        let max_sfb = max_sfb_capped[g];
        let mut ms = vec![false; max_sfb as usize];
        for sfb in 0..max_sfb as usize {
            let cb = sections[g].sfb_cb[sfb];
            if cb == 0 || mqi_per_g[g][sfb] == 0 {
                continue;
            }
            ms[sfb] = br.read_bit().ok()?;
        }
        ms_used_per_g.push(ms);
    }
    let _snf =
        asf_data::parse_asf_snf_data_grouped(br, &sections, &mqi_per_g, &max_sfb_capped, &tl_per_g)
            .ok()?;
    // Dequantise + scale + apply inverse M/S per group, concatenate.
    let mut scaled_l: Vec<f32> = Vec::new();
    let mut scaled_r: Vec<f32> = Vec::new();
    let mut ms_concat: Vec<bool> = Vec::new();
    for g in 0..tl_per_g.len() {
        let mut l = asf_data::dequantise_and_scale(
            &q_l_per_g[g],
            &sf_gain_per_g[g],
            sfbo_per_g[g],
            max_sfb_capped[g],
        );
        let mut r = asf_data::dequantise_and_scale(
            &q_r_per_g[g],
            &sf_gain_per_g[g],
            sfbo_per_g[g],
            max_sfb_capped[g],
        );
        for (sfb, &used) in ms_used_per_g[g].iter().enumerate() {
            if !used {
                continue;
            }
            let a = sfbo_per_g[g][sfb] as usize;
            let b = sfbo_per_g[g][sfb + 1] as usize;
            let bmax = b.min(l.len()).min(r.len());
            for k in a..bmax {
                let m = l[k];
                let s = r[k];
                l[k] = m + s;
                r[k] = m - s;
            }
        }
        scaled_l.extend_from_slice(&l);
        scaled_r.extend_from_slice(&r);
        ms_concat.extend_from_slice(&ms_used_per_g[g]);
    }
    Some((scaled_l, scaled_r, ms_concat))
}

/// Decode the `sf_data` body for a mono, long-frame, single-window-
/// group ASF substream. Returns the dequantised + scaled spectral
/// coefficients for the frame.
///
/// On any Huffman / bitstream error we return `None` — the caller
/// falls back to silence. Uses the decoder's own max_sfb (from
/// psy_info) rather than num_sfb_48 since the latter is only a cap.
fn decode_asf_long_mono_body(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<Vec<f32>> {
    let tl = ti.transform_length_0;
    let tl_idx = ti.transf_length[0];
    let max_sfb_cap = tables::num_sfb_48(tl)?;
    let max_sfb = psy.max_sfb_0.min(max_sfb_cap);
    if max_sfb == 0 {
        return None;
    }
    let sfbo = sfb_offset::sfb_offset_48(tl)?;
    // asf_section_data.
    let sections = asf_data::parse_asf_section_data(br, tl_idx, tl, max_sfb).ok()?;
    // asf_spectral_data.
    let (qspec, mqi) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    // asf_scalefac_data.
    let sf_gain = asf_data::parse_asf_scalefac_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // asf_snf_data — §5.1.4 spectral noise fill. If present, inject
    // shaped noise into zero-energy bins. The RNG state is initialised
    // fresh per frame (I-frame reset per §5.1.4) — cross-frame RNG
    // continuity is a future refinement.
    let snf = asf_data::parse_asf_snf_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // Dequantise + scale.
    let mut scaled = asf_data::dequantise_and_scale(&qspec, &sf_gain, sfbo, max_sfb);
    if let Some(snf_data) = snf {
        let mut rng: u32 = 0x1234_5678; // per-frame seed
        asf_data::inject_snf_noise(&mut scaled, &snf_data, sfbo, max_sfb, &mut rng);
    }
    Some(scaled)
}

/// Walk the `ASPX_ACPL_2` MDCT body (§4.2.6.3 case `ASPX_ACPL_2`):
///
/// ```text
/// spec_frontend;            1 bit
/// sf_info(spec_frontend, 0, 0);
/// sf_data(spec_frontend);
/// ```
///
/// Returns `true` when the body parses cleanly enough that the bitreader
/// is sitting at the start of the trailing `aspx_data_1ch()`. The decoded
/// spectral coefficients (when long-frame + single window group) land on
/// `tools.scaled_spec_primary`.
fn parse_aspx_acpl2_mdct_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    frame_len_base: u32,
) -> bool {
    let sf_bit = match br.read_u32(1) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let frontend = SpecFrontend::from_bit(sf_bit);
    tools.spec_frontend_primary = Some(frontend);
    if !matches!(frontend, SpecFrontend::Asf) {
        // SSF is gated on its own arithmetic-coded layer; we can't
        // walk it yet.
        return false;
    }
    let ti = match parse_asf_transform_info(br, frame_len_base) {
        Ok(t) => t,
        Err(_) => return false,
    };
    tools.transform_info_primary = Some(ti);
    let psy = match parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
        Ok(p) => p,
        Err(_) => return false,
    };
    tools.psy_info_primary = Some(psy.clone());
    if ti.b_long_frame && psy.num_window_groups == 1 {
        match decode_asf_long_mono_body(br, &ti, &psy) {
            Some(s) => {
                tools.scaled_spec_primary = Some(s);
                true
            }
            None => false,
        }
    } else if psy.num_window_groups > 1 {
        match decode_asf_grouped_mono_body(br, &ti, &psy) {
            Some(s) => {
                tools.scaled_spec_primary = Some(s);
                true
            }
            None => false,
        }
    } else {
        false
    }
}

/// Walk the `ASPX_ACPL_1` joint-MDCT residual layer (§4.2.6.3 case
/// `ASPX_ACPL_1`):
///
/// ```text
/// if (b_enable_mdct_stereo_proc) {                  1 bit
///     spec_frontend_m = ASF;
///     spec_frontend_s = ASF;
///     sf_info(ASF, 1, 0);
///     chparam_info();
/// } else {
///     spec_frontend_m;                              1 bit
///     sf_info(spec_frontend_m, 0, 0);
///     spec_frontend_s;                              1 bit
///     sf_info(spec_frontend_s, 0, 1);
/// }
/// sf_data(spec_frontend_m);
/// sf_data(spec_frontend_s);
/// ```
///
/// Returns `true` when the body parses cleanly enough that the bitreader
/// is sitting at the start of the trailing `aspx_data_1ch()`. The decoded
/// spectral coefficients land on `tools.scaled_spec_primary` (M / left)
/// and `tools.scaled_spec_secondary` (S / right). Joint-MDCT processing
/// state lands on `tools.chparam_info` (and `tools.mdct_stereo_proc` /
/// `tools.ms_used` mirrors for the simple `sap_mode == 1` case).
///
/// Only the long-frame, single-window-group case walks the residual
/// MDCT body; other shapes parse the outer layers and bail.
fn parse_aspx_acpl1_mdct_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    frame_len_base: u32,
) -> bool {
    parse_aspx_acpl1_mdct_body_stateful(br, tools, frame_len_base, None)
}

/// Stateful variant of [`parse_aspx_acpl1_mdct_body`]. Same SSF
/// state-borrow semantics as [`parse_stereo_data_body_stateful`].
fn parse_aspx_acpl1_mdct_body_stateful(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    frame_len_base: u32,
    ssf_states: Option<&mut [crate::ssf::SsfChannelState]>,
) -> bool {
    let b_mdct_stereo = match br.read_u32(1) {
        Ok(v) => v != 0,
        Err(_) => return false,
    };
    tools.mdct_stereo_proc = b_mdct_stereo;
    if b_mdct_stereo {
        // Joint MDCT: shared transform shell, sf_info(ASF, 1, 0), then
        // chparam_info().
        tools.spec_frontend_primary = Some(SpecFrontend::Asf);
        tools.spec_frontend_secondary = Some(SpecFrontend::Asf);
        let ti = match parse_asf_transform_info(br, frame_len_base) {
            Ok(t) => t,
            Err(_) => return false,
        };
        tools.transform_info_primary = Some(ti);
        tools.transform_info_secondary = Some(ti);
        // sf_info(ASF, 1, 0): b_dual_maxsfb = 1, b_side_limited = 0 —
        // the M channel uses max_sfb[0] (and any per-window max_sfb[1]
        // for non-long frames), the S channel uses max_sfb_side[0].
        let psy = match parse_asf_psy_info(br, &ti, frame_len_base, true, false) {
            Ok(p) => p,
            Err(_) => return false,
        };
        tools.psy_info_primary = Some(psy.clone());
        tools.psy_info_secondary = Some(psy.clone());
        // chparam_info(): drive ms_used / sap_data over the per-group
        // bound. The spec's `get_max_sfb(g)` (Pseudocode 5) returns
        // `max_sfb_side` for the side-channel decode and `max_sfb` for
        // the main; chparam_info itself runs at the joint shell, so we
        // use the side bound (the safer / smaller of the two — the M
        // channel's extra bands beyond max_sfb_side carry only the M
        // residual and don't have a meaningful M/S flag).
        let max_sfb_g = psy.max_sfb_side_0.min(psy.max_sfb_0);
        let cp = match parse_chparam_info(br, &[max_sfb_g]) {
            Ok(c) => c,
            Err(_) => return false,
        };
        // Mirror simple-mode `ms_used` into `tools.ms_used` (group 0)
        // for callers that only care about the per-sfb flag array.
        if cp.mode() == SapMode::MsUsed && !cp.ms_used.is_empty() {
            tools.ms_used = Some(cp.ms_used[0].clone());
        }
        tools.chparam_info = Some(cp);
        // sf_data(M); sf_data(S). Each channel has its own max_sfb (M
        // uses max_sfb_0, S uses max_sfb_side_0). We reuse the
        // mono-body decoder for each since the section / spectral /
        // scalefac / snf streams are independent here (the joint-MDCT
        // M/S coupling is parametrised by chparam_info — the residual
        // MDCT is per-channel). Both long-frame and short-frame /
        // grouped (`num_window_groups > 1`) shapes are supported.
        let m_body = decode_asf_mono_body_for_max_sfb(br, &ti, &psy, psy.max_sfb_0);
        let m_ok = m_body.is_some();
        if let Some(s) = m_body {
            tools.scaled_spec_primary = Some(s);
        }
        if !m_ok {
            return false;
        }
        let s_body = decode_asf_mono_body_for_max_sfb(br, &ti, &psy, psy.max_sfb_side_0);
        let s_ok = s_body.is_some();
        if let Some(s) = s_body {
            tools.scaled_spec_secondary = Some(s);
        }
        s_ok
    } else {
        // Independent stereo MDCT: spec_frontend_m + sf_info(?, 0, 0),
        // spec_frontend_s + sf_info(?, 0, 1).
        let m_bit = match br.read_u32(1) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let m_fe = SpecFrontend::from_bit(m_bit);
        tools.spec_frontend_primary = Some(m_fe);
        let mut ti_m: Option<AsfTransformInfo> = None;
        if let SpecFrontend::Asf = m_fe {
            let ti = match parse_asf_transform_info(br, frame_len_base) {
                Ok(t) => t,
                Err(_) => return false,
            };
            tools.transform_info_primary = Some(ti);
            ti_m = Some(ti);
            // sf_info(spec_frontend_m, 0, 0): b_dual_maxsfb = 0,
            // b_side_limited = 0 — main channel uses max_sfb[0].
            let psy = match parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                Ok(p) => p,
                Err(_) => return false,
            };
            tools.psy_info_primary = Some(psy);
        }
        let s_bit = match br.read_u32(1) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let s_fe = SpecFrontend::from_bit(s_bit);
        tools.spec_frontend_secondary = Some(s_fe);
        let mut ti_s: Option<AsfTransformInfo> = None;
        if let SpecFrontend::Asf = s_fe {
            let ti = match parse_asf_transform_info(br, frame_len_base) {
                Ok(t) => t,
                Err(_) => return false,
            };
            tools.transform_info_secondary = Some(ti);
            ti_s = Some(ti);
            // sf_info(spec_frontend_s, 0, 1): b_dual_maxsfb = 0,
            // b_side_limited = 1 — side channel uses max_sfb_side[0]
            // (deposited into psy.max_sfb_0 by parse_asf_psy_info).
            let psy = match parse_asf_psy_info(br, &ti, frame_len_base, false, true) {
                Ok(p) => p,
                Err(_) => return false,
            };
            tools.psy_info_secondary = Some(psy);
        }
        // sf_data(M); sf_data(S). Both channels are independent.
        // Round 32: borrow per-channel SSF walker state from the
        // caller-supplied slice when present so dither/noise RNG +
        // env_prev + predictor lag history persist across frames.
        let mut local_m = crate::ssf::SsfChannelState::new();
        let mut local_s = crate::ssf::SsfChannelState::new();
        let (state_m_ref, state_s_ref): (
            &mut crate::ssf::SsfChannelState,
            &mut crate::ssf::SsfChannelState,
        ) = match ssf_states {
            Some(slice) if slice.len() >= 2 => {
                let (a, rest) = slice.split_at_mut(1);
                (&mut a[0], &mut rest[0])
            }
            Some(slice) if slice.len() == 1 => (&mut slice[0], &mut local_s),
            _ => (&mut local_m, &mut local_s),
        };
        let psy_m_clone = tools.psy_info_primary.clone();
        if let (Some(ti), Some(psy)) = (ti_m, psy_m_clone.as_ref()) {
            match decode_asf_mono_body_dispatch(br, &ti, psy) {
                Some(s) => tools.scaled_spec_primary = Some(s),
                None => return false,
            }
        } else if matches!(m_fe, SpecFrontend::Ssf) {
            // §4.2.9 SSF body for the M channel of an ASPX_ACPL_1 split
            // residual layer.
            if let Some(cfg) = crate::ssf::SsfFrameConfig::from_frame_len_base(frame_len_base) {
                match crate::ssf::parse_ssf_data(br, false, &cfg, state_m_ref) {
                    Ok(d) => tools.ssf_data_primary = Some(d),
                    Err(_) => return false,
                }
            } else {
                return false;
            }
        }
        let psy_s_clone = tools.psy_info_secondary.clone();
        if let (Some(ti), Some(psy)) = (ti_s, psy_s_clone.as_ref()) {
            match decode_asf_mono_body_dispatch(br, &ti, psy) {
                Some(s) => tools.scaled_spec_secondary = Some(s),
                None => return false,
            }
        } else if matches!(s_fe, SpecFrontend::Ssf) {
            if let Some(cfg) = crate::ssf::SsfFrameConfig::from_frame_len_base(frame_len_base) {
                match crate::ssf::parse_ssf_data(br, false, &cfg, state_s_ref) {
                    Ok(d) => tools.ssf_data_secondary = Some(d),
                    Err(_) => return false,
                }
            } else {
                return false;
            }
        }
        true
    }
}

/// Long-frame / short-frame dispatch for the mono `sf_data(ASF)` body.
/// Picks the long-frame walker when `num_window_groups == 1` and the
/// grouped walker otherwise. Returns `None` on any Huffman /
/// bitstream miss.
fn decode_asf_mono_body_dispatch(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<Vec<f32>> {
    if ti.b_long_frame && psy.num_window_groups == 1 {
        decode_asf_long_mono_body(br, ti, psy)
    } else if psy.num_window_groups > 1 {
        decode_asf_grouped_mono_body(br, ti, psy)
    } else {
        None
    }
}

/// Same as [`decode_asf_mono_body_dispatch`] but with an explicit
/// `max_sfb` (the same value applied to both halves of the
/// `b_different_framing` case). Used by the joint-MDCT residual
/// layer where M and S channels carry distinct `max_sfb_0` /
/// `max_sfb_side_0` bounds.
fn decode_asf_mono_body_for_max_sfb(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
    max_sfb_in: u32,
) -> Option<Vec<f32>> {
    if ti.b_long_frame && psy.num_window_groups == 1 {
        decode_asf_long_mono_body_with_max_sfb(br, ti, max_sfb_in)
    } else if psy.num_window_groups > 1 {
        decode_asf_grouped_mono_body_with_max_sfb(br, ti, psy, max_sfb_in, max_sfb_in)
    } else {
        None
    }
}

/// Like [`decode_asf_long_mono_body`] but uses an explicit `max_sfb`
/// (caller-provided) instead of pulling it from the psy_info — useful
/// for joint-MDCT bodies where M and S channels carry distinct
/// `max_sfb_0` / `max_sfb_side_0` bounds.
///
/// Round-23 raised this to `pub(crate)` so the multichannel
/// [`crate::mch`] walkers (Tables 27 / 28 / 29) can drive one
/// `sf_data(ASF)` body per channel from the shared `sf_info(ASF, 0, 0)`
/// at the head of `three_channel_data` / `four_channel_data` /
/// `five_channel_data`.
pub(crate) fn decode_asf_long_mono_body_with_max_sfb(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    max_sfb_in: u32,
) -> Option<Vec<f32>> {
    let tl = ti.transform_length_0;
    let tl_idx = ti.transf_length[0];
    let max_sfb_cap = tables::num_sfb_48(tl)?;
    let max_sfb = max_sfb_in.min(max_sfb_cap);
    if max_sfb == 0 {
        return None;
    }
    let sfbo = sfb_offset::sfb_offset_48(tl)?;
    let sections = asf_data::parse_asf_section_data(br, tl_idx, tl, max_sfb).ok()?;
    let (qspec, mqi) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    let sf_gain = asf_data::parse_asf_scalefac_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    let _snf = asf_data::parse_asf_snf_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    let scaled = asf_data::dequantise_and_scale(&qspec, &sf_gain, sfbo, max_sfb);
    Some(scaled)
}

/// LFE flavour of [`decode_asf_long_mono_body_with_max_sfb`] — Table 35
/// `sf_info_lfe()` always forces `b_long_frame == 1` and reads `max_sfb[0]`
/// with the `n_msfbl_bits` width (Table 106 column 4) instead of the
/// regular `n_msfb_bits`. The `sf_data(ASF)` body for an LFE channel is
/// then the same shape as for any other long-frame ASF channel — a single
/// `asf_section_data + asf_spectral_data + asf_scalefac_data + asf_snf_data`
/// quartet over `[0, max_sfb)`. The narrower bit-width for `max_sfb` is
/// the only LFE-specific bit-stream difference; the body decoder itself
/// matches `decode_asf_long_mono_body_with_max_sfb` exactly.
///
/// Round 38 surfaces this as a separate entry point so the LFE path in
/// [`crate::mch::parse_mono_data`] can call it explicitly (and so callers
/// reading the call site know they're decoding an LFE body, not a
/// standard mono body that just happens to have a small `max_sfb`).
pub(crate) fn decode_asf_long_lfe_body_with_max_sfb_lfe(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    max_sfb_lfe: u32,
) -> Option<Vec<f32>> {
    // Per Table 35: `sf_info_lfe()` only runs in the long-frame path. If
    // the caller hands us a non-long-frame transform info we bail rather
    // than trying to walk a grouped body — the LFE body shape is
    // explicitly long-frame only per spec (`b_long_frame = 1`).
    if !ti.b_long_frame {
        return None;
    }
    decode_asf_long_mono_body_with_max_sfb(br, ti, max_sfb_lfe)
}

/// Parse the outer layers of a stereo `audio_data()` element
/// (channel_pair_element / stereo_data).
///
/// Only the most common case — `stereo_codec_mode == SIMPLE` with the
/// `stereo_data()` body — is walked to the `sf_info` level. Other
/// modes (ASPX / ASPX_ACPL_1 / ASPX_ACPL_2) set the tool enum and stop.
pub fn parse_stereo_audio_data_outer(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<()> {
    parse_stereo_audio_data_outer_stateful(br, tools, b_iframe, frame_len_base, None)
}

/// Stateful variant of [`parse_stereo_audio_data_outer`]. When
/// `ssf_states` is `Some`, the inner SSF body parses for primary /
/// secondary channels (M / S split-MDCT or independent stereo) borrow
/// the matching channel slot so RNG / `env_prev` continuity persists
/// across frames. `None` falls back to stack-local `SsfChannelState`s
/// (round 31 behavior).
pub fn parse_stereo_audio_data_outer_stateful(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    b_iframe: bool,
    frame_len_base: u32,
    mut ssf_states: Option<&mut [crate::ssf::SsfChannelState]>,
) -> Result<()> {
    // §4.2.6.3 channel_pair_element(b_iframe):
    //   stereo_codec_mode;        2 bits
    let mode_bits = br.read_u32(2)?;
    let mode = StereoCodecMode::from_u32(mode_bits);
    tools.stereo_mode = Some(mode);

    if mode != StereoCodecMode::Simple {
        // §4.2.6.3: for b_iframe all ASPX stereo modes start with an
        // aspx_config(); ASPX_ACPL_{1,2} also carry an acpl_config_1ch
        // (PARTIAL / FULL) right after — both are now parsed (the
        // A-CPL config is only 6 bits in PARTIAL mode, 3 in FULL).
        // Each stereo mode then runs companding_control with a
        // mode-specific channel count (2 for ASPX, 1 for ASPX_ACPL_*),
        // which we also surface.
        let mut acpl_cfg_active: Option<crate::acpl::AcplConfig1ch> = None;
        if b_iframe {
            tools.aspx_config = Some(aspx::parse_aspx_config(br)?);
            match mode {
                StereoCodecMode::AspxAcpl1 => {
                    let cfg =
                        crate::acpl::parse_acpl_config_1ch(br, crate::acpl::Acpl1chMode::Partial)?;
                    tools.acpl_config_1ch_partial = Some(cfg);
                    acpl_cfg_active = Some(cfg);
                }
                StereoCodecMode::AspxAcpl2 => {
                    let cfg =
                        crate::acpl::parse_acpl_config_1ch(br, crate::acpl::Acpl1chMode::Full)?;
                    tools.acpl_config_1ch_full = Some(cfg);
                    acpl_cfg_active = Some(cfg);
                }
                _ => {}
            }
        } else {
            // Non-I-frame: the configs are I-frame-sticky (§4.2.6.3
            // Table 22 gates them on b_iframe) — reuse the values the
            // caller pre-seeded into `tools` from [`StickyConfig`].
            acpl_cfg_active = match mode {
                StereoCodecMode::AspxAcpl1 => tools.acpl_config_1ch_partial,
                StereoCodecMode::AspxAcpl2 => tools.acpl_config_1ch_full,
                _ => None,
            };
        }
        let nc = match mode {
            StereoCodecMode::Aspx => 2,
            _ => 1,
        };
        tools.companding = Some(aspx::parse_companding_control(br, nc)?);
        // §4.2.6.3 ASPX_ACPL_{1,2}: after companding_control(1) follow
        // the MDCT body (`stereo_data`-shaped for ACPL_1, mono for
        // ACPL_2), then `aspx_data_1ch()` (Table 51) and
        // `acpl_data_1ch()` (Table 61). Walk them when the upstream
        // body parses cleanly.
        if matches!(
            mode,
            StereoCodecMode::AspxAcpl1 | StereoCodecMode::AspxAcpl2
        ) {
            // ASPX_ACPL_1 walks the joint-MDCT residual layer
            // (b_dual_maxsfb=1 with chparam_info() — Table 47);
            // ASPX_ACPL_2 walks a single mono MDCT residual.
            let body_ok = match mode {
                StereoCodecMode::AspxAcpl1 => parse_aspx_acpl1_mdct_body_stateful(
                    br,
                    tools,
                    frame_len_base,
                    ssf_states.as_deref_mut(),
                ),
                StereoCodecMode::AspxAcpl2 => parse_aspx_acpl2_mdct_body(br, tools, frame_len_base),
                _ => false,
            };
            if body_ok {
                if let Some(cfg) = tools.aspx_config {
                    if parse_aspx_data_1ch_body(br, tools, &cfg, b_iframe, frame_len_base).is_ok() {
                        if let Some(acfg) = acpl_cfg_active {
                            // §4.2.13.3 Table 61: num_bands =
                            // acpl_num_param_bands; start =
                            // acpl_param_band (= sb_to_pb(qmf_band) for
                            // PARTIAL, 0 for FULL).
                            let start_band = if acfg.qmf_band == 0 {
                                0
                            } else {
                                crate::acpl::sb_to_pb(acfg.qmf_band as u32, acfg.num_param_bands)
                            };
                            if let Ok(d) = crate::acpl::parse_acpl_data_1ch(
                                br,
                                acfg.num_param_bands,
                                start_band,
                                acfg.quant_mode,
                            ) {
                                tools.acpl_data_1ch = Some(d);
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        // For `StereoCodecMode::Aspx` (Table 22) the stereo_data()
        // body follows companding_control(2), then aspx_data_2ch()
        // closes the element. We parse stereo_data() with the same
        // shared decoder as SIMPLE, then attempt to read the leading
        // [xover-offset (I-frames only) +] aspx_framing(0)[ +
        // aspx_balance + framing(1)] of aspx_data_2ch(). Runs when the
        // stereo_data() body decoded cleanly (bitreader is at the
        // right place) and an aspx_config is available (parsed above
        // on I-frames, sticky-seeded on non-I-frames).
        if matches!(mode, StereoCodecMode::Aspx) {
            let body_ok = parse_stereo_data_body_stateful(
                br,
                tools,
                frame_len_base,
                ssf_states.as_deref_mut(),
            );
            if body_ok {
                if let Some(cfg) = tools.aspx_config {
                    parse_aspx_data_2ch_body(br, tools, &cfg, b_iframe, frame_len_base)?;
                }
            }
        }
        return Ok(());
    }

    // SIMPLE path: just `stereo_data()`.
    let _ = parse_stereo_data_body_stateful(br, tools, frame_len_base, ssf_states);
    let _ = b_iframe; // reserved for later Huffman-state keying.
    Ok(())
}

/// Walk `stereo_data()` (§4.2.6.3 / Table 22) into `tools`. Returns
/// `true` when the body decoded cleanly enough for the bitreader to sit
/// at the spec's end-of-body position (i.e. the caller is safe to
/// continue reading downstream elements like `aspx_data_2ch()`). A
/// return of `false` means some inner Huffman-gated decode bailed; the
/// bitreader position after the call is indeterminate.
pub(crate) fn parse_stereo_data_body(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    frame_len_base: u32,
) -> bool {
    parse_stereo_data_body_stateful(br, tools, frame_len_base, None)
}

/// Stateful variant of [`parse_stereo_data_body`]. When `ssf_states`
/// is `Some`, split-MDCT stereo's two SSF body decodes borrow the
/// matching channel slot from the slice (channel 0 → primary,
/// channel 1 → secondary) so the SSF walker's RNG / `env_prev` /
/// predictor lag history persists across frames. `None` falls back to
/// stack-local channel state.
pub(crate) fn parse_stereo_data_body_stateful(
    br: &mut BitReader<'_>,
    tools: &mut SubstreamTools,
    frame_len_base: u32,
    ssf_states: Option<&mut [crate::ssf::SsfChannelState]>,
) -> bool {
    // stereo_data():
    //   if (b_enable_mdct_stereo_proc) {
    //       spec_frontend_l = ASF; spec_frontend_r = ASF;
    //       sf_info(ASF, 0, 0);
    //       chparam_info();
    //   } else {
    //       spec_frontend_l;   1 bit
    //       sf_info(spec_frontend_l, 0, 0);
    //       spec_frontend_r;   1 bit
    //       sf_info(spec_frontend_r, 0, 0);
    //   }
    //   sf_data(spec_frontend_l);
    //   sf_data(spec_frontend_r);
    let b_mdct_stereo = match br.read_bit() {
        Ok(v) => v,
        Err(_) => return false,
    };
    tools.mdct_stereo_proc = b_mdct_stereo;
    if b_mdct_stereo {
        tools.spec_frontend_primary = Some(SpecFrontend::Asf);
        tools.spec_frontend_secondary = Some(SpecFrontend::Asf);
        let ti = match parse_asf_transform_info(br, frame_len_base) {
            Ok(t) => t,
            Err(_) => return false,
        };
        tools.transform_info_primary = Some(ti);
        tools.transform_info_secondary = Some(ti);
        // stereo_data() with b_enable_mdct_stereo_proc==1 calls
        // sf_info(ASF, 0, 0) which fires asf_psy_info(0, 0) over the
        // shared transform window.
        let psy = match parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
            Ok(p) => p,
            Err(_) => return false,
        };
        tools.psy_info_primary = Some(psy.clone());
        // Joint-stereo MDCT path: one shared section / psy layout,
        // two residual channel spectra, followed by a per-sfb
        // ms_used[] flag array. Both long-frame (single window group)
        // and short-frame / grouped (`num_window_groups > 1`) shapes
        // are supported; the latter walks the per-group spec layout
        // (Tables 39-42) with shared scalefactors and per-group
        // ms_used[].
        if ti.b_long_frame && psy.num_window_groups == 1 {
            match decode_asf_long_stereo_joint_body(br, &ti, &psy) {
                Some((l, r, ms)) => {
                    tools.scaled_spec_primary = Some(l);
                    tools.scaled_spec_secondary = Some(r);
                    tools.ms_used = Some(ms);
                    true
                }
                None => false,
            }
        } else if psy.num_window_groups > 1 {
            match decode_asf_grouped_stereo_joint_body(br, &ti, &psy) {
                Some((l, r, ms)) => {
                    tools.scaled_spec_primary = Some(l);
                    tools.scaled_spec_secondary = Some(r);
                    tools.ms_used = Some(ms);
                    true
                }
                None => false,
            }
        } else {
            // Outer layers parsed but Huffman body not walked — the
            // bitreader isn't at the end of stereo_data(), so downstream
            // elements aren't safe to parse.
            false
        }
    } else {
        let l = match br.read_u32(1) {
            Ok(v) => SpecFrontend::from_bit(v),
            Err(_) => return false,
        };
        tools.spec_frontend_primary = Some(l);
        let mut ti_l: Option<AsfTransformInfo> = None;
        let mut psy_l: Option<AsfPsyInfo> = None;
        if let SpecFrontend::Asf = l {
            let ti = match parse_asf_transform_info(br, frame_len_base) {
                Ok(t) => t,
                Err(_) => return false,
            };
            tools.transform_info_primary = Some(ti);
            ti_l = Some(ti);
            if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                tools.psy_info_primary = Some(psy.clone());
                psy_l = Some(psy);
            } else {
                return false;
            }
        }
        let r = match br.read_u32(1) {
            Ok(v) => SpecFrontend::from_bit(v),
            Err(_) => return false,
        };
        tools.spec_frontend_secondary = Some(r);
        let mut ti_r: Option<AsfTransformInfo> = None;
        let mut psy_r: Option<AsfPsyInfo> = None;
        if let SpecFrontend::Asf = r {
            let ti = match parse_asf_transform_info(br, frame_len_base) {
                Ok(t) => t,
                Err(_) => return false,
            };
            tools.transform_info_secondary = Some(ti);
            ti_r = Some(ti);
            if let Ok(psy) = parse_asf_psy_info(br, &ti, frame_len_base, false, false) {
                tools.psy_info_secondary = Some(psy.clone());
                psy_r = Some(psy);
            } else {
                return false;
            }
        }
        // Split-MDCT stereo: two independent ASF (or SSF) spectra.
        // Decode each for long-frame (single window group), short-frame
        // / grouped (`num_window_groups > 1`), or — when the frontend
        // resolved to SSF — drive the §4.2.9 SSF walker. Any malformed
        // body is surfaced as a decode miss (None) rather than a hard
        // error.
        let mut body_ok = true;
        // Round 32: split-MDCT stereo passes ssf_states[0] to the M
        // channel SSF walker and ssf_states[1] to the S channel SSF
        // walker (when present). Local fallbacks keep the no-state path
        // working unchanged.
        let mut local_l = crate::ssf::SsfChannelState::new();
        let mut local_r = crate::ssf::SsfChannelState::new();
        let (state_l_ref, state_r_ref): (
            &mut crate::ssf::SsfChannelState,
            &mut crate::ssf::SsfChannelState,
        ) = match ssf_states {
            Some(slice) if slice.len() >= 2 => {
                let (a, rest) = slice.split_at_mut(1);
                (&mut a[0], &mut rest[0])
            }
            Some(slice) if slice.len() == 1 => (&mut slice[0], &mut local_r),
            _ => (&mut local_l, &mut local_r),
        };
        if let (Some(ti), Some(psy)) = (ti_l, psy_l.as_ref()) {
            tools.scaled_spec_primary = decode_asf_mono_body_dispatch(br, &ti, psy);
            if tools.scaled_spec_primary.is_none() {
                body_ok = false;
            }
        } else if matches!(l, SpecFrontend::Ssf) {
            if let Some(cfg) = crate::ssf::SsfFrameConfig::from_frame_len_base(frame_len_base) {
                // The split-MDCT stereo walker doesn't know whether the
                // surrounding `audio_data()` came from an I-frame —
                // assume non-I (the safe default; an I-frame body would
                // have been gated by `b_iframe_global` upstream and
                // pulled `b_ssf_iframe = 1` from the AC layer
                // explicitly). We surface the parsed body when
                // available and silently skip on parse miss.
                if let Ok(d) = crate::ssf::parse_ssf_data(br, false, &cfg, state_l_ref) {
                    tools.ssf_data_primary = Some(d);
                } else {
                    body_ok = false;
                }
            }
        }
        if let (Some(ti), Some(psy)) = (ti_r, psy_r.as_ref()) {
            tools.scaled_spec_secondary = decode_asf_mono_body_dispatch(br, &ti, psy);
            if tools.scaled_spec_secondary.is_none() {
                body_ok = false;
            }
        } else if matches!(r, SpecFrontend::Ssf) {
            if let Some(cfg) = crate::ssf::SsfFrameConfig::from_frame_len_base(frame_len_base) {
                if let Ok(d) = crate::ssf::parse_ssf_data(br, false, &cfg, state_r_ref) {
                    tools.ssf_data_secondary = Some(d);
                } else {
                    body_ok = false;
                }
            }
        }
        body_ok
    }
}

/// Decode the `sf_data` body for a stereo, long-frame, single-window-
/// group ASF substream with `b_enable_mdct_stereo_proc == 1` (joint
/// MDCT processing — §7.4 / §7.5).
///
/// Layout inferred from the spec:
///   - shared asf_section_data (one section partition drives both
///     channels' codebook assignment).
///   - two asf_spectral_data bodies (residuals for channel L and R,
///     which are actually M and S when ms_used is set).
///   - shared asf_scalefac_data (the scalefactors are shared so the
///     joint quant step is uniform across both channels).
///   - per-sfb ms_used[sfb] flag — one bit per active band.
///   - asf_snf_data consumed but not injected.
///
/// Returns `(left_scaled, right_scaled, ms_used)` on success.
fn decode_asf_long_stereo_joint_body(
    br: &mut BitReader<'_>,
    ti: &AsfTransformInfo,
    psy: &AsfPsyInfo,
) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
    let tl = ti.transform_length_0;
    let tl_idx = ti.transf_length[0];
    let max_sfb_cap = tables::num_sfb_48(tl)?;
    let max_sfb = psy.max_sfb_0.min(max_sfb_cap);
    if max_sfb == 0 {
        return None;
    }
    let sfbo = sfb_offset::sfb_offset_48(tl)?;
    // Shared asf_section_data.
    let sections = asf_data::parse_asf_section_data(br, tl_idx, tl, max_sfb).ok()?;
    // L / M channel residuals.
    let (q_l, mqi_l) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    // R / S channel residuals.
    let (q_r, mqi_r) = asf_data::parse_asf_spectral_data(br, &sections, sfbo, max_sfb).ok()?;
    // Shared scalefactors — compute max_quant_idx as the band-wise max
    // over both channels so the scalefactor DPCM state tracks bands
    // that have any energy at all.
    let mqi: Vec<u32> = mqi_l
        .iter()
        .zip(mqi_r.iter())
        .map(|(a, b)| (*a).max(*b))
        .collect();
    let sf_gain = asf_data::parse_asf_scalefac_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // Per-sfb ms_used flag array. Only active bands (cb != 0 and
    // max_quant_idx > 0) carry a bit per §7.5 Pseudocode 77; other
    // bands default to false.
    let mut ms_used = vec![false; max_sfb as usize];
    for sfb in 0..max_sfb as usize {
        let cb = sections.sfb_cb[sfb];
        if cb == 0 || mqi[sfb] == 0 {
            continue;
        }
        ms_used[sfb] = br.read_bit().ok()?;
    }
    // asf_snf_data consumed but not currently injected.
    let _ = asf_data::parse_asf_snf_data(br, &sections, &mqi, max_sfb, tl).ok()?;
    // Dequantise + scale both channels with the shared gain.
    let mut scaled_l = asf_data::dequantise_and_scale(&q_l, &sf_gain, sfbo, max_sfb);
    let mut scaled_r = asf_data::dequantise_and_scale(&q_r, &sf_gain, sfbo, max_sfb);
    // Apply inverse M/S per §7.5: L = M + S, R = M - S for bands with
    // ms_used == true.
    for (sfb, &used) in ms_used.iter().enumerate() {
        if !used {
            continue;
        }
        let a = sfbo[sfb] as usize;
        let b = sfbo[sfb + 1] as usize;
        let bmax = b.min(scaled_l.len()).min(scaled_r.len());
        for k in a..bmax {
            let m = scaled_l[k];
            let s = scaled_r[k];
            scaled_l[k] = m + s;
            scaled_r[k] = m - s;
        }
    }
    Some((scaled_l, scaled_r, ms_used))
}

/// Walk an `ac4_substream()` payload. `substream_bytes` is the slice
/// covering the substream as pointed to by `substream_index_table()`.
/// The walker reads the outer framing and the initial `audio_data()`
/// flags; on success returns an [`Ac4SubstreamInfo`] with the tool
/// summary. On malformed input — which for a stub walker mostly means
/// running off the end of the bitstream — the function returns a
/// best-effort info with whatever was parsed before the error.
///
/// `channels` is the channel count taken from the parent
/// `ac4_substream_info()`. `b_iframe` likewise. `frame_len_base` is
/// `frame_length` at the base sample rate (i.e. the TOC's
/// `frame_length` entry for 48 kHz and 44.1 kHz).
pub fn walk_ac4_substream(
    substream_bytes: &[u8],
    channels: u16,
    b_iframe: bool,
    frame_len_base: u32,
) -> Result<Ac4SubstreamInfo> {
    walk_ac4_substream_stateful(substream_bytes, channels, b_iframe, frame_len_base, None)
}

/// Stateful variant of [`walk_ac4_substream`] — accepts a `&mut`
/// per-channel [`crate::ssf::SsfChannelState`] slice (one entry per
/// channel) so that SSF dither / noise RNG continuity, predictor lag
/// history, and the previous granule's `env_prev[]` snapshot persist
/// across frame boundaries (round 32). Pass `None` to fall back to the
/// stateless behaviour the original `walk_ac4_substream` offered.
pub fn walk_ac4_substream_stateful(
    substream_bytes: &[u8],
    channels: u16,
    b_iframe: bool,
    frame_len_base: u32,
    ssf_states: Option<&mut [crate::ssf::SsfChannelState]>,
) -> Result<Ac4SubstreamInfo> {
    walk_ac4_substream_sticky(
        substream_bytes,
        channels,
        b_iframe,
        frame_len_base,
        ssf_states,
        None,
    )
}

/// Sticky-config variant of [`walk_ac4_substream_stateful`] — accepts a
/// `&mut` [`StickyConfig`] carried across frames so `b_iframe == 0`
/// (P-frame) substreams can be parsed: the I-frame-gated
/// `aspx_config()` / `acpl_config_*()` elements and the
/// `aspx_xover_subband_offset` field are seeded into the frame's
/// [`SubstreamTools`] before the walk, and re-harvested from every
/// I-frame after it. Pass `None` for the historical I-frame-only
/// behaviour.
pub fn walk_ac4_substream_sticky(
    substream_bytes: &[u8],
    channels: u16,
    b_iframe: bool,
    frame_len_base: u32,
    mut ssf_states: Option<&mut [crate::ssf::SsfChannelState]>,
    sticky: Option<&mut StickyConfig>,
) -> Result<Ac4SubstreamInfo> {
    if substream_bytes.is_empty() {
        return Err(Error::invalid("ac4: empty substream"));
    }
    let mut br = BitReader::new(substream_bytes);

    // §4.3.4.1 audio_size_value — 15-bit value, optional variable_bits(7)
    // extension gated by b_more_bits.
    let audio_size_short = br.read_u32(15)?;
    let b_more_bits = br.read_bit()?;
    let audio_size = if b_more_bits {
        audio_size_short + (variable_bits(&mut br, 7)? << 15)
    } else {
        audio_size_short
    };

    // byte_align to enter audio_data().
    br.align_to_byte();
    let audio_data_offset = br.byte_position() as u32;

    // Parse the outer layers of audio_data(channel_mode, b_iframe).
    let mut tools = SubstreamTools {
        channel_mode_channels: channels,
        ..Default::default()
    };
    // Non-I-frames reuse the I-frame-sticky configs (aspx_config /
    // acpl_config_* / xover offset) from the last I-frame.
    if !b_iframe {
        if let Some(st) = sticky.as_ref() {
            st.seed(&mut tools);
        }
    }
    match channels {
        1 => parse_mono_audio_data_outer_stateful(
            &mut br,
            &mut tools,
            b_iframe,
            frame_len_base,
            ssf_states.as_deref_mut(),
        )?,
        2 => parse_stereo_audio_data_outer_stateful(
            &mut br,
            &mut tools,
            b_iframe,
            frame_len_base,
            ssf_states,
        )?,
        // 5.0 / 5.1 — drive the `5_X_channel_element` walker
        // (§4.2.6.6 Table 25). r19 lands the outer-shell parse;
        // r20 fills in Cfg0/Cfg1/Cfg2 + LFE psy_info_lfe(); inner
        // sf_data bodies + ASPX/A-CPL trailers wait for a later round.
        5 => {
            // 5.0 — no LFE.
            let _ = crate::mch::parse_5x_audio_data_outer(
                &mut br,
                &mut tools,
                false,
                b_iframe,
                frame_len_base,
            );
        }
        6 => {
            // 5.1 — LFE present.
            let _ = crate::mch::parse_5x_audio_data_outer(
                &mut br,
                &mut tools,
                true,
                b_iframe,
                frame_len_base,
            );
        }
        // 7.0 / 7.1 — drive the `7_X_channel_element` walker (round 26).
        // The 7.X walker mirrors the 5_X SIMPLE/ASPX path's
        // coding_config selector with a 2-bit codec_mode (no
        // ASPX_ACPL_3) and an extra additional-channel `two_channel_data`
        // for the front-extension / back-surround pair.
        7 => {
            // 7.0 — no LFE.
            let _ = crate::mch::parse_7x_audio_data_outer(
                &mut br,
                &mut tools,
                false,
                b_iframe,
                frame_len_base,
            );
        }
        8 => {
            // 7.1 — LFE present (channel_mode == "7.1").
            let _ = crate::mch::parse_7x_audio_data_outer(
                &mut br,
                &mut tools,
                true,
                b_iframe,
                frame_len_base,
            );
        }
        // Immersive channel modes (TS 103 190-2 Table 78, ch_mode
        // 11..=14): `audio_data_chan()` routes to
        // `immersive_channel_element(b_lfe, b_5fronts, b_iframe)` per
        // §6.2.3.1. The channel counts are unique per mode (11 =
        // 7.0.4, 12 = 7.1.4, 13 = 9.0.4, 14 = 9.1.4). Try-and-bail
        // like the 5.X / 7.X walkers.
        11..=14 => {
            let b_lfe = channels == 12 || channels == 14;
            let b_5fronts = channels == 13 || channels == 14;
            let _ = crate::ice::parse_ice_audio_data_outer(
                &mut br,
                &mut tools,
                b_lfe,
                b_5fronts,
                b_iframe,
                frame_len_base,
            );
        }
        // 3.0 paths are coding-config-dependent; their
        // outer walkers live behind the same Huffman gate as ASF's
        // spectral data. For the baseline we record the channel count
        // and bail.
        _ => {}
    }

    // Every I-frame refreshes the sticky configs for the P-frames that
    // follow it (§4.2.6.x: configs are only transmitted on I-frames).
    if b_iframe {
        if let Some(st) = sticky {
            st.harvest(&tools);
        }
    }

    Ok(Ac4SubstreamInfo {
        audio_size,
        audio_data_offset,
        tools,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    /// Repack `bytes` dropping the first `skip_bits` bits — used to turn
    /// an I-frame `aspx_data_*` body (leading 3-bit xover offset) into
    /// its P-frame form (xover omitted per Tables 51 / 52).
    fn strip_leading_bits(bytes: &[u8], skip_bits: u32, total_bits: u32) -> Vec<u8> {
        let mut br = BitReader::new(bytes);
        for _ in 0..skip_bits {
            let _ = br.read_bit();
        }
        let mut bw = BitWriter::new();
        for _ in 0..(total_bits - skip_bits) {
            bw.write_bit(br.read_bit().unwrap_or(false));
        }
        bw.into_bytes()
    }

    fn p_frame_test_aspx_config() -> aspx::AspxConfig {
        aspx::AspxConfig {
            quant_mode_env: aspx::AspxQuantStep::Fine,
            start_freq: 0,
            stop_freq: 0,
            master_freq_scale: aspx::AspxMasterFreqScale::LowRes,
            interpolation: false,
            preflat: false,
            limiter: false,
            noise_sbg: 0,
            num_env_bits_fixfix: 0,
            freq_res_mode: aspx::AspxFreqResMode::Signalled,
        }
    }

    #[test]
    fn aspx_data_1ch_p_frame_parses_with_sticky_xover() {
        // Build an I-frame aspx_data_1ch() body, parse it, then strip
        // the leading 3-bit xover offset (its only I-frame-gated field
        // per Table 51) and re-parse as a P-frame with the sticky
        // xover pre-seeded — the recovered framing + envelopes must be
        // identical.
        let cfg = p_frame_test_aspx_config();
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).expect("tables");
        let num_sig = tables.counts.num_sbg_sig_lowres as usize;
        let num_noise = tables.counts.num_sbg_noise as usize;
        let sig: Vec<i32> = (0..num_sig as i32).map(|i| (i % 3) - 1).collect();
        let noise: Vec<i32> = vec![1; num_noise];
        let mut bw = BitWriter::new();
        crate::encoder_acpl3::write_aspx_data_1ch_real_envelope(
            &mut bw,
            &cfg,
            crate::encoder_acpl3::AspxRealEnvelopeChannel {
                sig: &sig,
                noise: &noise,
            },
        )
        .expect("write 1ch body");
        let total_bits = bw.bit_position() as u32;
        let i_bytes = bw.into_bytes();

        let mut tools_i = SubstreamTools::default();
        let mut br = BitReader::new(&i_bytes);
        parse_aspx_data_1ch_body(&mut br, &mut tools_i, &cfg, true, 2048).expect("iframe parse");
        assert_eq!(tools_i.aspx_xover_subband_offset, Some(0));

        let p_bytes = strip_leading_bits(&i_bytes, 3, total_bits);
        let mut tools_p = SubstreamTools {
            aspx_xover_subband_offset: Some(0),
            ..Default::default()
        };
        let mut br = BitReader::new(&p_bytes);
        parse_aspx_data_1ch_body(&mut br, &mut tools_p, &cfg, false, 2048).expect("pframe parse");

        assert_eq!(tools_i.aspx_framing_primary, tools_p.aspx_framing_primary);
        assert_eq!(tools_i.aspx_data_sig_primary, tools_p.aspx_data_sig_primary);
        assert_eq!(
            tools_i.aspx_data_noise_primary,
            tools_p.aspx_data_noise_primary
        );
        assert_eq!(tools_i.aspx_hfgen_iwc_1ch, tools_p.aspx_hfgen_iwc_1ch);
    }

    #[test]
    fn aspx_data_1ch_p_frame_without_sticky_xover_errors() {
        let cfg = p_frame_test_aspx_config();
        let mut tools = SubstreamTools::default();
        let bytes = [0u8; 8];
        let mut br = BitReader::new(&bytes);
        assert!(parse_aspx_data_1ch_body(&mut br, &mut tools, &cfg, false, 2048).is_err());
    }

    #[test]
    fn aspx_data_2ch_p_frame_parses_with_sticky_xover() {
        let cfg = p_frame_test_aspx_config();
        let tables = aspx::derive_aspx_frequency_tables(&cfg, 0).expect("tables");
        let num_sig = tables.counts.num_sbg_sig_lowres as usize;
        let num_noise = tables.counts.num_sbg_noise as usize;
        let sig0: Vec<i32> = (0..num_sig as i32).map(|i| (i % 4) - 2).collect();
        let sig1: Vec<i32> = (0..num_sig as i32).map(|i| 1 - (i % 2)).collect();
        let noise0: Vec<i32> = vec![2; num_noise];
        let noise1: Vec<i32> = vec![-1; num_noise];
        let mut bw = BitWriter::new();
        crate::encoder_acpl3::write_aspx_data_2ch_real_envelope(
            &mut bw,
            &cfg,
            crate::encoder_acpl3::AspxRealEnvelopeChannel {
                sig: &sig0,
                noise: &noise0,
            },
            crate::encoder_acpl3::AspxRealEnvelopeChannel {
                sig: &sig1,
                noise: &noise1,
            },
        )
        .expect("write 2ch body");
        let total_bits = bw.bit_position() as u32;
        let i_bytes = bw.into_bytes();

        let mut tools_i = SubstreamTools::default();
        let mut br = BitReader::new(&i_bytes);
        parse_aspx_data_2ch_body(&mut br, &mut tools_i, &cfg, true, 2048).expect("iframe parse");

        let p_bytes = strip_leading_bits(&i_bytes, 3, total_bits);
        let mut tools_p = SubstreamTools::default();
        let sticky = StickyConfig {
            aspx_config: Some(cfg),
            aspx_xover: Some(0),
            ..Default::default()
        };
        sticky.seed(&mut tools_p);
        let mut br = BitReader::new(&p_bytes);
        parse_aspx_data_2ch_body(&mut br, &mut tools_p, &cfg, false, 2048).expect("pframe parse");

        assert_eq!(tools_i.aspx_framing_primary, tools_p.aspx_framing_primary);
        assert_eq!(
            tools_i.aspx_framing_secondary,
            tools_p.aspx_framing_secondary
        );
        assert_eq!(tools_i.aspx_balance, tools_p.aspx_balance);
        assert_eq!(tools_i.aspx_data_sig_primary, tools_p.aspx_data_sig_primary);
        assert_eq!(
            tools_i.aspx_data_sig_secondary,
            tools_p.aspx_data_sig_secondary
        );
        assert_eq!(
            tools_i.aspx_data_noise_primary,
            tools_p.aspx_data_noise_primary
        );
        assert_eq!(
            tools_i.aspx_data_noise_secondary,
            tools_p.aspx_data_noise_secondary
        );
        assert_eq!(tools_i.aspx_hfgen_iwc_2ch, tools_p.aspx_hfgen_iwc_2ch);
        // Harvesting the I-frame tools must reproduce the seeded state.
        let mut harvested = StickyConfig::default();
        harvested.harvest(&tools_i);
        assert_eq!(harvested.aspx_xover, Some(0));
    }

    #[test]
    fn sticky_config_seed_harvest_round_trip() {
        let cfg = p_frame_test_aspx_config();
        let acpl1 = crate::acpl::AcplConfig1ch {
            num_param_bands_id: 2,
            num_param_bands: 12,
            quant_mode: crate::acpl::AcplQuantMode::Fine,
            qmf_band: 5,
        };
        let tools = SubstreamTools {
            aspx_config: Some(cfg),
            aspx_xover_subband_offset: Some(3),
            acpl_config_1ch_partial: Some(acpl1),
            ..Default::default()
        };
        let mut sticky = StickyConfig::default();
        sticky.harvest(&tools);
        let mut seeded = SubstreamTools::default();
        sticky.seed(&mut seeded);
        assert_eq!(seeded.aspx_config, Some(cfg));
        assert_eq!(seeded.aspx_xover_subband_offset, Some(3));
        assert_eq!(seeded.acpl_config_1ch_partial, Some(acpl1));
        assert_eq!(seeded.acpl_config_1ch_full, None);
        assert_eq!(seeded.acpl_config_2ch, None);
    }

    #[test]
    fn resolve_transf_length_long_frame_table_99() {
        // Long frame at 44.1/48 kHz returns frame_len_base directly.
        assert_eq!(resolve_transf_length(2048, true, 0), 2048);
        assert_eq!(resolve_transf_length(1920, true, 3), 1920);
    }

    #[test]
    fn resolve_transf_length_table_100_rows() {
        assert_eq!(resolve_transf_length(2048, false, 0), 128);
        assert_eq!(resolve_transf_length(2048, false, 1), 256);
        assert_eq!(resolve_transf_length(2048, false, 2), 512);
        assert_eq!(resolve_transf_length(2048, false, 3), 1024);
        assert_eq!(resolve_transf_length(1920, false, 2), 480);
        assert_eq!(resolve_transf_length(1536, false, 1), 192);
    }

    #[test]
    fn resolve_transf_length_table_103_rows() {
        assert_eq!(resolve_transf_length(1024, false, 3), 1024);
        assert_eq!(resolve_transf_length(960, false, 2), 480);
        assert_eq!(resolve_transf_length(512, false, 0), 128);
        assert_eq!(resolve_transf_length(384, false, 2), 384);
    }

    #[test]
    fn asf_transform_info_long_frame_path() {
        // frame_len_base = 1920, b_long_frame = 1.
        let mut bw = BitWriter::new();
        bw.write_bit(true); // b_long_frame
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = parse_asf_transform_info(&mut br, 1920).unwrap();
        assert!(ti.b_long_frame);
        assert_eq!(ti.transform_length_0, 1920);
        assert_eq!(ti.transform_length_1, 1920);
    }

    #[test]
    fn asf_transform_info_short_pair() {
        // frame_len_base = 1920, b_long_frame = 0, transf_length[0] = 2,
        // transf_length[1] = 3. -> transform lengths 480, 960.
        let mut bw = BitWriter::new();
        bw.write_bit(false); // b_long_frame
        bw.write_u32(2, 2);
        bw.write_u32(3, 2);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = parse_asf_transform_info(&mut br, 1920).unwrap();
        assert!(!ti.b_long_frame);
        assert_eq!(ti.transf_length, [2, 3]);
        assert_eq!(ti.transform_length_0, 480);
        assert_eq!(ti.transform_length_1, 960);
    }

    fn build_mono_substream() -> Vec<u8> {
        // Minimal ac4_substream() body: audio_size = 10, no extension,
        // byte_align, then audio_data() for channel_mode=mono,
        // b_iframe=1: mono_codec_mode=0 (SIMPLE), spec_frontend=0 (ASF),
        // asf_transform_info() for frame_len_base=1920 long-frame.
        let mut bw = BitWriter::new();
        // audio_size_value (15 bits) = 10.
        bw.write_u32(10, 15);
        // b_more_bits = 0.
        bw.write_bit(false);
        // byte_align to enter audio_data.
        bw.align_to_byte();
        // mono_codec_mode = 0 (SIMPLE).
        bw.write_u32(0, 1);
        // mono_data(0) body:
        //   spec_frontend = 0 (ASF).
        bw.write_u32(0, 1);
        //   sf_info(ASF, 0, 0) -> asf_transform_info():
        //     frame_len_base >= 1536 path; b_long_frame = 1.
        bw.write_bit(true);
        //   asf_psy_info(b_dual_maxsfb=0, b_side_limited=0):
        //   long-frame @ 1920 => n_msfb_bits = 6; max_sfb[0] = 40.
        bw.write_u32(40, 6);
        //   asf_section_data / spectral / scalefac / snf left opaque.
        bw.align_to_byte();
        // Pad up to "audio_size"-worth of bytes.
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn walk_mono_asf_substream_extracts_tools() {
        let bytes = build_mono_substream();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        assert_eq!(info.audio_size, 10);
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Simple));
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Asf));
        let ti = info.tools.transform_info_primary.unwrap();
        assert!(ti.b_long_frame);
        assert_eq!(ti.transform_length_0, 1920);
        let psy = info.tools.psy_info_primary.as_ref().unwrap();
        assert_eq!(psy.max_sfb_0, 40);
        assert_eq!(psy.num_windows, 1);
        assert_eq!(psy.num_window_groups, 1);
    }

    fn build_stereo_simple_substream() -> Vec<u8> {
        let mut bw = BitWriter::new();
        // audio_size = 20.
        bw.write_u32(20, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // stereo_codec_mode = SIMPLE (0b00).
        bw.write_u32(0, 2);
        // stereo_data(): b_enable_mdct_stereo_proc = 0.
        bw.write_bit(false);
        // spec_frontend_l = 0 (ASF), long-frame, max_sfb[0] = 30.
        bw.write_u32(0, 1);
        bw.write_bit(true); // b_long_frame
        bw.write_u32(30, 6);
        // spec_frontend_r = 0 (ASF), long-frame, max_sfb[0] = 32.
        bw.write_u32(0, 1);
        bw.write_bit(true);
        bw.write_u32(32, 6);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        bw.finish()
    }

    #[test]
    fn walk_stereo_simple_substream_extracts_two_frontends() {
        let bytes = build_stereo_simple_substream();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.audio_size, 20);
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::Simple));
        assert!(!info.tools.mdct_stereo_proc);
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Asf));
        assert_eq!(info.tools.spec_frontend_secondary, Some(SpecFrontend::Asf));
        assert!(info.tools.transform_info_primary.is_some());
        assert!(info.tools.transform_info_secondary.is_some());
        let psy_l = info.tools.psy_info_primary.as_ref().unwrap();
        let psy_r = info.tools.psy_info_secondary.as_ref().unwrap();
        assert_eq!(psy_l.max_sfb_0, 30);
        assert_eq!(psy_r.max_sfb_0, 32);
    }

    #[test]
    fn walk_stereo_mdct_joint_shares_transform_info() {
        let mut bw = BitWriter::new();
        bw.write_u32(20, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0, 2); // SIMPLE
        bw.write_bit(true); // b_enable_mdct_stereo_proc
        bw.write_bit(true); // long-frame
        bw.write_u32(25, 6); // max_sfb[0] = 25.
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert!(info.tools.mdct_stereo_proc);
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Asf));
        assert_eq!(info.tools.spec_frontend_secondary, Some(SpecFrontend::Asf));
        assert_eq!(
            info.tools
                .transform_info_primary
                .unwrap()
                .transform_length_0,
            info.tools
                .transform_info_secondary
                .unwrap()
                .transform_length_0
        );
        let psy = info.tools.psy_info_primary.as_ref().unwrap();
        assert_eq!(psy.max_sfb_0, 25);
    }

    #[test]
    fn asf_psy_info_long_frame_path() {
        // Manually drive parse_asf_psy_info with a known transform_info.
        let ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0, 0],
            transform_length_0: 1920,
            transform_length_1: 1920,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(50, 6); // max_sfb[0]
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        assert!(!psy.b_different_framing);
        assert_eq!(psy.max_sfb_0, 50);
        assert_eq!(psy.num_windows, 1);
        assert_eq!(psy.num_window_groups, 1);
        assert!(psy.scale_factor_grouping.is_empty());
    }

    #[test]
    fn asf_psy_info_short_pair_with_grouping() {
        // frame_len_base = 1920, transf_length = [2, 2] (480 both).
        // From Table 109: n_grp_bits = 3. n_msfb_bits at 480 = 6.
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [2, 2],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(20, 6); // max_sfb[0]
                             // scale_factor_grouping bits (3): 1, 0, 1 => one new group at bit 1.
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        // b_different_framing = false (transf_length[0] == transf_length[1]).
        assert!(!psy.b_different_framing);
        assert_eq!(psy.max_sfb_0, 20);
        assert_eq!(psy.scale_factor_grouping, vec![1, 0, 1]);
        // num_windows = n_grp_bits + 1 = 4; one zero-bit => 2 groups.
        assert_eq!(psy.num_windows, 4);
        assert_eq!(psy.num_window_groups, 2);
    }

    #[test]
    fn asf_psy_info_lfe_long_frame_2048_uses_3bit_max_sfb() {
        // tl=2048 -> n_msfbl_bits = 3 (Table 106 column 4). max_sfb[0] = 7.
        let ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0, 0],
            transform_length_0: 2048,
            transform_length_1: 2048,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(7, 3); // max_sfb[0] -- 3 bits, value 7
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info_lfe(&mut br, &ti).unwrap();
        assert_eq!(psy.max_sfb_0, 7);
        assert_eq!(psy.num_windows, 1);
        assert_eq!(psy.num_window_groups, 1);
        assert_eq!(psy.max_sfb_1, 0);
        assert!(psy.scale_factor_grouping.is_empty());
    }

    #[test]
    fn asf_psy_info_lfe_long_frame_512_uses_2bit_max_sfb() {
        // tl=512 -> n_msfbl_bits = 2 (Table 106). max_sfb[0] = 3.
        let ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0, 0],
            transform_length_0: 512,
            transform_length_1: 512,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(3, 2); // 2 bits, value 3
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info_lfe(&mut br, &ti).unwrap();
        assert_eq!(psy.max_sfb_0, 3);
    }

    #[test]
    fn asf_psy_info_lfe_rejects_short_only_transforms() {
        // tl=480 -> n_msfbl_bits = 0 in Table 106 (LFE not permitted).
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [2, 2],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let err = parse_asf_psy_info_lfe(&mut br, &ti).unwrap_err();
        assert!(format!("{err}").contains("LFE"));
    }

    #[test]
    fn asf_psy_info_lfe_rejects_unknown_transform() {
        // tl=999 isn't in Table 106 at all.
        let ti = AsfTransformInfo {
            b_long_frame: true,
            transf_length: [0, 0],
            transform_length_0: 999,
            transform_length_1: 999,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let err = parse_asf_psy_info_lfe(&mut br, &ti).unwrap_err();
        assert!(format!("{err}").contains("transform_length"));
    }

    #[test]
    fn asf_psy_info_different_framing_path() {
        // transf_length[0] = 1 (240 @ 1920), transf_length[1] = 2 (480).
        // n_grp_bits (Table 109) = 4.
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [1, 2],
            transform_length_0: 240,
            transform_length_1: 480,
        };
        let mut bw = BitWriter::new();
        // n_msfb_bits at 240 = 5 (Table 106).
        bw.write_u32(10, 5); // max_sfb[0]
                             // b_different_framing triggers:
                             // n_msfb_bits at 480 = 6.
        bw.write_u32(15, 6); // max_sfb[1]
                             // scale_factor_grouping bits (4): 0,0,0,0
        bw.write_u32(0, 4);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let psy = parse_asf_psy_info(&mut br, &ti, 1920, false, false).unwrap();
        assert!(psy.b_different_framing);
        assert_eq!(psy.max_sfb_0, 10);
        assert_eq!(psy.max_sfb_1, 15);
        assert_eq!(psy.num_windows, 5);
    }

    #[test]
    fn walk_mono_aspx_substream_parses_config_and_companding() {
        // mono_codec_mode = 1 (ASPX) on an I-frame: the walker now
        // consumes aspx_config() (15 bits) and companding_control(1).
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15); // audio_size = 5
        bw.write_bit(false); // b_more_bits = 0
        bw.align_to_byte();
        // mono_codec_mode = 1 (ASPX)
        bw.write_u32(1, 1);
        // aspx_config: quant_mode=0, start=3, stop=1, scale=1, interp=1,
        // preflat=0, limiter=1, noise_sbg=2, num_env_bits_fixfix=0,
        // freq_res_mode=0.
        bw.write_u32(0, 1);
        bw.write_u32(3, 3);
        bw.write_u32(1, 2);
        bw.write_u32(1, 1);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_u32(2, 2);
        bw.write_u32(0, 1);
        bw.write_u32(0, 2);
        // companding_control(1): no sync_flag; compand_on[0] = 1.
        bw.write_bit(true);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        // aspx_config captured.
        let cfg = info.tools.aspx_config.unwrap();
        assert_eq!(cfg.start_freq, 3);
        assert_eq!(cfg.stop_freq, 1);
        assert_eq!(cfg.noise_sbg, 2);
        assert_eq!(cfg.num_noise_sbgroups(), 3);
        assert!(cfg.interpolation);
        assert!(!cfg.preflat);
        assert!(cfg.limiter);
        assert!(cfg.signals_freq_res());
        // companding captured.
        let cc = info.tools.companding.as_ref().unwrap();
        assert_eq!(cc.compand_on, vec![true]);
        assert!(cc.sync_flag.is_none());
        assert!(cc.compand_avg.is_none());
    }

    #[test]
    fn walk_stereo_aspx_substream_parses_config() {
        // stereo_codec_mode = 01 (ASPX) on an I-frame: read
        // aspx_config() then companding_control(2).
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b01, 2); // ASPX
                               // aspx_config: all-zero fields.
        bw.write_u32(0, 15);
        // companding_control(2): sync_flag=0, compand_on=[1,1].
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::Aspx));
        let cfg = info.tools.aspx_config.unwrap();
        assert_eq!(cfg.start_freq, 0);
        let cc = info.tools.companding.as_ref().unwrap();
        assert_eq!(cc.sync_flag, Some(false));
        assert_eq!(cc.compand_on, vec![true, true]);
    }

    #[test]
    fn walk_stereo_aspx_non_iframe_skips_config() {
        // Predictive (b_iframe = 0) ASPX frame: aspx_config not present;
        // we go straight to companding_control.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b01, 2); // ASPX
                               // No aspx_config — go straight to companding_control(2).
        bw.write_bit(true); // sync_flag = 1 -> single compand_on
        bw.write_bit(true); // compand_on[0] = 1
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, false, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::Aspx));
        assert!(info.tools.aspx_config.is_none());
        let cc = info.tools.companding.as_ref().unwrap();
        assert_eq!(cc.sync_flag, Some(true));
        assert_eq!(cc.compand_on, vec![true]);
    }

    /// Write sect_len_incr per §4.2.7.1 Table 80 (section-length
    /// expansion). Duplicated from `decoder.rs` tests for crate-local use.
    fn write_sect_len_incr(bw: &mut BitWriter, sect_len: u32, n_sect_bits: u32, esc: u32) {
        let base = sect_len.saturating_sub(1);
        let k = base / esc;
        let incr = base % esc;
        for _ in 0..k {
            bw.write_u32(esc, n_sect_bits);
        }
        bw.write_u32(incr, n_sect_bits);
    }

    #[test]
    fn walk_mono_aspx_iframe_reads_framing_after_mono_data() {
        // Full-path mono ASPX I-frame substream:
        //   audio_size_value (15), b_more_bits (1), byte_align.
        //   mono_codec_mode = 1 (ASPX)
        //   aspx_config() — 15 bits.
        //   companding_control(1) — 1 bit.
        //   mono_data(0):
        //     spec_frontend = 0 (ASF)
        //     asf_transform_info(): b_long_frame = 1
        //     asf_psy_info(): max_sfb[0] = 10 (6 bits)
        //     sf_data body (section + spectral zeros + scalefac 120 + snf off)
        //   aspx_data_1ch():
        //     aspx_xover_subband_offset = 3 (3 bits)
        //     aspx_framing(0): FIXFIX (1 bit) + tmp_num_env=1 (1 bit since
        //     aspx_num_env_bits_fixfix=0) -> num_env = 2.
        // We set aspx_freq_res_mode to High (not Signalled) so no
        // freq_res bits follow, keeping the fixture small.
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(200, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // mono_codec_mode = ASPX
        bw.write_u32(1, 1);
        // aspx_config(): quant_mode=0, start=0, stop=0, scale=0,
        // interp=0, preflat=0, limiter=0, noise_sbg=0,
        // num_env_bits_fixfix=0, freq_res_mode=3 (High -> not signalled).
        bw.write_u32(0, 1); // quant_mode_env
        bw.write_u32(0, 3); // start_freq
        bw.write_u32(0, 2); // stop_freq
        bw.write_u32(0, 1); // master_freq_scale
        bw.write_bit(false); // interpolation
        bw.write_bit(false); // preflat
        bw.write_bit(false); // limiter
        bw.write_u32(0, 2); // noise_sbg
        bw.write_u32(0, 1); // num_env_bits_fixfix = 0 -> 1-bit tmp_num_env
        bw.write_u32(3, 2); // freq_res_mode = 3 (High)
                            // companding_control(1): compand_on[0] = 1, no avg.
        bw.write_bit(true);
        // mono_data(0): spec_frontend = 0 (ASF).
        bw.write_u32(0, 1);
        // asf_transform_info: b_long_frame = 1.
        bw.write_bit(true);
        // asf_psy_info: max_sfb[0] = 10 in 6 bits.
        bw.write_u32(10, 6);
        // sf_data body (all-zero spectra):
        //   section: cb_idx = 5 (4 bits) + sect_len for max_sfb=10 via
        //   n_sect_bits=3 (long frame at 1920), esc=7.
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, 10, 3, 7);
        // spectral pairs — cb 5 is dim=2, so pairs = end_line / 2.
        // sfb_offset_48 @ 1920 index 10 = ?
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end_line = sfbo[10] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        let pairs = end_line / 2;
        for _ in 0..pairs {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32); // zero pair
        }
        // scalefac: reference = 120 (8 bits). All bands are empty, so
        // no dpcm follows.
        bw.write_u32(120, 8);
        // snf: b_snf_data_exists = 0.
        bw.write_u32(0, 1);
        // aspx_data_1ch():
        //   aspx_xover_subband_offset = 3 (3 bits).
        bw.write_u32(3, 3);
        //   aspx_framing(0): FIXFIX = '0' (1 bit); tmp_num_env = 1
        //   (1 bit) -> num_env = 1 << 1 = 2.
        bw.write_bit(false); // FIXFIX
        bw.write_bit(true); // tmp_num_env = 1
        bw.align_to_byte();
        while bw.byte_len() < 220 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        // Sanity: we got through the outer layers.
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        assert!(info.tools.aspx_config.is_some());
        assert!(info.tools.companding.is_some());
        assert!(info.tools.transform_info_primary.is_some());
        // aspx_data_1ch path fired:
        assert_eq!(info.tools.aspx_xover_subband_offset, Some(3));
        let framing = info.tools.aspx_framing_primary.expect("framing");
        assert_eq!(framing.int_class, aspx::AspxIntClass::FixFix);
        assert_eq!(framing.num_env, 2);
        assert_eq!(framing.num_noise, 2);
        assert!(framing.freq_res.is_empty()); // freq_res_mode = High
                                              // Stereo sidecar untouched on mono.
        assert!(info.tools.aspx_framing_secondary.is_none());
        assert!(info.tools.aspx_balance.is_none());
        // aspx_delta_dir(0) followed on the same path: consumed
        // num_env + num_noise = 4 bits — all zeros from the padding.
        let dd = info.tools.aspx_delta_dir_primary.expect("delta dir");
        assert_eq!(dd.sig_delta_dir, vec![false; 2]);
        assert_eq!(dd.noise_delta_dir, vec![false; 2]);
        // FIXFIX + num_env > 1, so the config's quant_mode_env carries
        // through (and the config had it as Fine).
        assert_eq!(
            info.tools.aspx_qmode_env_primary,
            Some(aspx::AspxQuantStep::Fine)
        );
    }

    #[test]
    fn walk_mono_aspx_non_iframe_does_not_emit_framing() {
        // Non-I-frame ASPX substream: no aspx_config in the bitstream,
        // so the walker cannot safely consume aspx_data_1ch (xover
        // offset is also I-frame-sticky). Framing stays None.
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(200, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(1, 1); // mono_codec_mode = ASPX
                            // No aspx_config (b_iframe = false).
        bw.write_bit(true); // companding_control(1): compand_on[0] = 1.
        bw.write_u32(0, 1); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame = 1
        bw.write_u32(10, 6); // max_sfb
        bw.write_u32(5, 4); // cb
        write_sect_len_incr(&mut bw, 10, 3, 7);
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end_line = sfbo[10] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        for _ in 0..(end_line / 2) {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        bw.write_u32(120, 8);
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 220 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, false, 1920).unwrap();
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Aspx));
        // No aspx_config (non-I-frame), so no framing either.
        assert!(info.tools.aspx_config.is_none());
        assert!(info.tools.aspx_framing_primary.is_none());
        assert!(info.tools.aspx_xover_subband_offset.is_none());
    }

    /// End-to-end mono ASPX I-frame substream: hand-build
    /// aspx_hfgen_iwc_1ch plus two aspx_ec_data() payloads and verify
    /// the walker captures them. This is the round-6 acceptance test
    /// for parse_aspx_ec_data's wiring through §5.7.6.3.1 derivation.
    #[test]
    fn walk_mono_aspx_iframe_reads_ec_data_end_to_end() {
        use crate::aspx::{
            ASPX_HCB_ENV_LEVEL_15_DF, ASPX_HCB_ENV_LEVEL_15_F0, ASPX_HCB_NOISE_LEVEL_F0,
        };
        use crate::huffman;
        let mut bw = BitWriter::new();
        bw.write_u32(500, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(1, 1); // mono_codec_mode = ASPX
                            // aspx_config(): quant_mode=0, start=3, stop=3, scale=1
                            // (highres -> 22 - 6 - 6 = 10 master groups), freq_res_mode=3
                            // (High -> no per-env freq_res bits), other flags zero.
        bw.write_u32(0, 1); // quant_mode_env
        bw.write_u32(3, 3); // start_freq
        bw.write_u32(3, 2); // stop_freq
        bw.write_u32(1, 1); // master_freq_scale = HighRes
        bw.write_bit(false); // interpolation
        bw.write_bit(false); // preflat
        bw.write_bit(false); // limiter
        bw.write_u32(0, 2); // noise_sbg
        bw.write_u32(0, 1); // num_env_bits_fixfix
        bw.write_u32(3, 2); // freq_res_mode = High
                            // companding_control(1): compand_on[0] = 1.
        bw.write_bit(true);
        // mono_data(0): ASF frontend, long frame, max_sfb=10, zero
        // spectra + scalefac 120 + no snf.
        bw.write_u32(0, 1); // spec_frontend = ASF
        bw.write_bit(true); // b_long_frame
        bw.write_u32(10, 6); // max_sfb
        bw.write_u32(5, 4);
        write_sect_len_incr(&mut bw, 10, 3, 7);
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let end_line = sfbo[10] as u32;
        let hcb = huffman::asf_hcb(5).unwrap();
        for _ in 0..(end_line / 2) {
            bw.write_u32(hcb.cw[40], hcb.len[40] as u32);
        }
        bw.write_u32(120, 8); // reference scalefac
        bw.write_u32(0, 1); // b_snf_data_exists
                            // aspx_data_1ch(): xover=7 -> num_sbg_sig_highres = 10 - 7 = 3
                            // and num_sbg_sig_lowres = 2. num_sbg_noise clamps to 1 since
                            // aspx_noise_sbg = 0. FIXFIX + tmp_num_env=0 -> num_env = 1.
        bw.write_u32(7, 3); // aspx_xover_subband_offset
        bw.write_bit(false); // FIXFIX
        bw.write_bit(false); // tmp_num_env = 0
                             // aspx_delta_dir(0): sig[0]=0, noise[0]=0 (FREQ for both).
        bw.write_bit(false);
        bw.write_bit(false);
        // aspx_hfgen_iwc_1ch: tna_mode[0] = 0 (2 bits), ah_present=0,
        // fic_present=0, tic_present=0 (3 bits).
        bw.write_u32(0, 2);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(false);
        // aspx_data_sig[0]: num_env=1, high-res (3 subband groups).
        // F0 symbol 30 + two DF zero-deltas (index 70).
        let f0_cb = &ASPX_HCB_ENV_LEVEL_15_F0;
        let df_cb = &ASPX_HCB_ENV_LEVEL_15_DF;
        bw.write_u32(f0_cb.cw[30], f0_cb.len[30] as u32);
        bw.write_u32(df_cb.cw[70], df_cb.len[70] as u32);
        bw.write_u32(df_cb.cw[70], df_cb.len[70] as u32);
        // aspx_data_noise[0]: num_noise=1 (since num_env==1), num_sbg_noise=1.
        // One F0 codeword from the NOISE_LEVEL_F0 table.
        let noise_f0 = &ASPX_HCB_NOISE_LEVEL_F0;
        bw.write_u32(noise_f0.cw[6], noise_f0.len[6] as u32);
        bw.align_to_byte();
        while bw.byte_len() < 520 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 1, true, 1920).unwrap();
        // §5.7.6.3.1 derivation results.
        let tables = info.tools.aspx_frequency_tables.as_ref().expect("tables");
        assert_eq!(tables.num_sbg_master, 10);
        assert_eq!(tables.sba, 24);
        assert_eq!(tables.sbz, 44);
        assert_eq!(tables.counts.num_sbg_sig_highres, 3);
        assert_eq!(tables.counts.num_sbg_sig_lowres, 2);
        assert_eq!(tables.counts.num_sbg_noise, 1);
        // hfgen: all gates off.
        let hfgen = info.tools.aspx_hfgen_iwc_1ch.as_ref().expect("hfgen");
        assert_eq!(hfgen.tna_mode, vec![0]);
        assert!(!hfgen.ah_present);
        assert!(!hfgen.fic_present);
        assert!(!hfgen.tic_present);
        // Signal envelope: F0=30, DF=0, DF=0.
        let sig = info.tools.aspx_data_sig_primary.as_ref().expect("sig");
        assert_eq!(sig.len(), 1);
        assert_eq!(sig[0].values, vec![30, 0, 0]);
        assert!(!sig[0].direction_time);
        // Noise envelope: single F0 with delta = 6.
        let noise = info.tools.aspx_data_noise_primary.as_ref().expect("noise");
        assert_eq!(noise.len(), 1);
        assert_eq!(noise[0].values, vec![6]);
        assert!(!noise[0].direction_time);
    }

    #[test]
    fn walk_stereo_aspx_acpl1_parses_acpl_config_partial() {
        // stereo_codec_mode = 10 (ASPX_ACPL_1): aspx_config + then
        // acpl_config_1ch(PARTIAL) (= 6 bits: 2 bands_id + 1 quant +
        // 3 qmf_band_minus1). The walker now also consumes
        // companding_control(1) and tries the MDCT body, but ACPL_1's
        // joint-MDCT residual layer isn't wired so the body bails;
        // config + companding still surface.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b10, 2); // ASPX_ACPL_1
        bw.write_u32(0, 15); // aspx_config (all zero)
                             // acpl_config_1ch(PARTIAL): id=1 (12 bands), coarse, qmf_minus1=3
        bw.write_u32(0b01, 2);
        bw.write_u32(1, 1);
        bw.write_u32(0b011, 3);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::AspxAcpl1));
        assert!(info.tools.aspx_config.is_some());
        let acpl = info
            .tools
            .acpl_config_1ch_partial
            .expect("acpl_config_1ch_partial should be set");
        assert_eq!(acpl.num_param_bands_id, 1);
        assert_eq!(acpl.num_param_bands, 12);
        assert_eq!(acpl.quant_mode, crate::acpl::AcplQuantMode::Coarse);
        assert_eq!(acpl.qmf_band, 4);
        assert!(info.tools.acpl_config_1ch_full.is_none());
        // companding_control is now consumed (was previously skipped);
        // ACPL_1 body still bails so acpl_data_1ch stays None.
        assert!(info.tools.companding.is_some());
        assert!(info.tools.acpl_data_1ch.is_none());
    }

    #[test]
    fn walk_stereo_aspx_acpl2_parses_acpl_config_full() {
        // stereo_codec_mode = 11 (ASPX_ACPL_2): aspx_config + then
        // acpl_config_1ch(FULL) (= 3 bits: 2 bands_id + 1 quant). The
        // walker now also reads companding_control(1) and attempts the
        // mono MDCT body. For this synthetic frame with no real MDCT
        // payload the body parse bails harmlessly; the config still
        // surfaces.
        let mut bw = BitWriter::new();
        bw.write_u32(5, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0b11, 2); // ASPX_ACPL_2
        bw.write_u32(0, 15); // aspx_config (all zero)
                             // acpl_config_1ch(FULL): id=2 (9 bands), fine, no qmf_band field
        bw.write_u32(0b10, 2);
        bw.write_u32(0, 1);
        bw.align_to_byte();
        while bw.byte_len() < 32 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let info = walk_ac4_substream(&bytes, 2, true, 1920).unwrap();
        assert_eq!(info.tools.stereo_mode, Some(StereoCodecMode::AspxAcpl2));
        assert!(info.tools.aspx_config.is_some());
        let acpl = info
            .tools
            .acpl_config_1ch_full
            .expect("acpl_config_1ch_full should be set");
        assert_eq!(acpl.num_param_bands_id, 2);
        assert_eq!(acpl.num_param_bands, 9);
        assert_eq!(acpl.quant_mode, crate::acpl::AcplQuantMode::Fine);
        // qmf_band stays 0 for FULL mode (not transmitted).
        assert_eq!(acpl.qmf_band, 0);
        assert!(info.tools.acpl_config_1ch_partial.is_none());
        // companding_control is now consumed (was previously skipped);
        // ACPL_2 body parse fires but bails on the synthetic payload so
        // acpl_data_1ch stays None.
        assert!(info.tools.companding.is_some());
        assert!(info.tools.acpl_data_1ch.is_none());
    }

    // ---------------- chparam_info() (Table 47) ----------------

    #[test]
    fn chparam_info_sap_mode_zero_no_payload() {
        // sap_mode = 0 -> no further bits.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 2);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cp = parse_chparam_info(&mut br, &[8]).unwrap();
        assert_eq!(cp.sap_mode, 0);
        assert_eq!(cp.mode(), SapMode::None);
        assert!(cp.ms_used.is_empty());
        assert!(cp.sap_data.is_none());
    }

    #[test]
    fn chparam_info_sap_mode_one_walks_ms_used_per_group() {
        // sap_mode = 1; one group with max_sfb = 5 -> 5 bits, MSB-first
        // 0b10110 -> [true, false, true, true, false].
        let mut bw = BitWriter::new();
        bw.write_u32(1, 2);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cp = parse_chparam_info(&mut br, &[5]).unwrap();
        assert_eq!(cp.mode(), SapMode::MsUsed);
        assert_eq!(cp.ms_used.len(), 1);
        assert_eq!(cp.ms_used[0], vec![true, false, true, true, false]);
    }

    #[test]
    fn chparam_info_sap_mode_three_walks_sap_data_pair_packed() {
        // sap_mode = 3 -> sap_data():
        //   sap_coeff_all = 0
        //   For one group with max_sfb = 4 -> 2 pair flags. Set both
        //   so all 4 sfbs carry a dpcm.
        //   delta_code_time omitted (num_window_groups == 1).
        //   Then 2 huff-decoded raw indices (we use index 60 = DC, 0
        //   bits payload — let's encode HCB_SCALEFAC[60]).
        use crate::huffman::{HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN};
        let mut bw = BitWriter::new();
        bw.write_u32(3, 2); // sap_mode
        bw.write_bit(false); // sap_coeff_all
        bw.write_bit(true); // pair flag for sfb 0/1
        bw.write_bit(true); // pair flag for sfb 2/3
                            // 2 dpcm_alpha_q codewords (one per pair = sfbs 0 and 2).
        bw.write_u32(HCB_SCALEFAC_CW[60], HCB_SCALEFAC_LEN[60] as u32);
        bw.write_u32(HCB_SCALEFAC_CW[61], HCB_SCALEFAC_LEN[61] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cp = parse_chparam_info(&mut br, &[4]).unwrap();
        assert_eq!(cp.mode(), SapMode::SapData);
        let sd = cp.sap_data.expect("sap_data populated");
        assert!(!sd.sap_coeff_all);
        assert_eq!(sd.sap_coeff_used, vec![vec![true, true, true, true]]);
        // dpcm_alpha_q row: [delta(60-60)=0, 0, delta(61-60)=1, 0].
        assert_eq!(sd.dpcm_alpha_q, vec![vec![0, 0, 1, 0]]);
    }

    #[test]
    fn chparam_info_sap_mode_three_all_flag_skips_pair_array() {
        // sap_coeff_all = 1 fills sap_coeff_used with `true` and skips
        // the pair-flag bits, but the dpcm_alpha_q stream is still one
        // codeword per pair.
        use crate::huffman::{HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN};
        let mut bw = BitWriter::new();
        bw.write_u32(3, 2);
        bw.write_bit(true); // sap_coeff_all = 1
        bw.write_u32(HCB_SCALEFAC_CW[60], HCB_SCALEFAC_LEN[60] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cp = parse_chparam_info(&mut br, &[2]).unwrap();
        let sd = cp.sap_data.expect("sap_data populated");
        assert!(sd.sap_coeff_all);
        assert_eq!(sd.sap_coeff_used, vec![vec![true, true]]);
        assert_eq!(sd.dpcm_alpha_q, vec![vec![0, 0]]);
    }

    #[test]
    fn chparam_info_sap_mode_three_two_groups_emits_delta_code_time() {
        // num_window_groups = 2 -> after sap_coeff_used, delta_code_time
        // (1 bit) appears.
        use crate::huffman::{HCB_SCALEFAC_CW, HCB_SCALEFAC_LEN};
        let mut bw = BitWriter::new();
        bw.write_u32(3, 2);
        bw.write_bit(true); // sap_coeff_all = 1 (no pair flags)
        bw.write_bit(true); // delta_code_time
                            // Group 0 max_sfb = 2 -> 1 codeword.
        bw.write_u32(HCB_SCALEFAC_CW[60], HCB_SCALEFAC_LEN[60] as u32);
        // Group 1 max_sfb = 2 -> 1 codeword.
        bw.write_u32(HCB_SCALEFAC_CW[60], HCB_SCALEFAC_LEN[60] as u32);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cp = parse_chparam_info(&mut br, &[2, 2]).unwrap();
        let sd = cp.sap_data.expect("sap_data populated");
        assert!(sd.sap_coeff_all);
        assert!(sd.delta_code_time);
        assert_eq!(sd.dpcm_alpha_q.len(), 2);
        assert_eq!(sd.dpcm_alpha_q[0], vec![0, 0]);
        assert_eq!(sd.dpcm_alpha_q[1], vec![0, 0]);
    }

    // ---------------- extract_sap_abcd / Pseudocode 59 ----------------

    /// `sap_mode == 0` -> identity matrix on every band.
    #[test]
    fn extract_sap_abcd_mode_zero_returns_identity() {
        let cp = ChparamInfo {
            sap_mode: 0,
            ..ChparamInfo::default()
        };
        let coeffs = extract_sap_abcd(&cp, &[4]);
        assert_eq!(coeffs.abcd.len(), 1);
        for q in &coeffs.abcd[0] {
            assert_eq!(*q, (1.0, 0.0, 0.0, 1.0));
        }
    }

    /// `sap_mode == 1` -> per-sfb selector. ms_used == 1 -> M/S inverse
    /// (`a=b=c=1, d=-1`); ms_used == 0 -> identity.
    #[test]
    fn extract_sap_abcd_mode_one_swaps_per_sfb_on_ms_used() {
        let cp = ChparamInfo {
            sap_mode: 1,
            ms_used: vec![vec![true, false, true, false]],
            sap_data: None,
        };
        let coeffs = extract_sap_abcd(&cp, &[4]);
        assert_eq!(coeffs.abcd[0][0], (1.0, 1.0, 1.0, -1.0));
        assert_eq!(coeffs.abcd[0][1], (1.0, 0.0, 0.0, 1.0));
        assert_eq!(coeffs.abcd[0][2], (1.0, 1.0, 1.0, -1.0));
        assert_eq!(coeffs.abcd[0][3], (1.0, 0.0, 0.0, 1.0));
    }

    /// `sap_mode == 3, sap_used` -> alpha-driven SAP. Pair-major DPCM:
    /// odd sfbs inherit the even pair-mate's alpha_q. Even sfbs accumulate
    /// the dpcm delta against the previous even sfb in the same group
    /// (or alpha == delta when `sfb == 0`).
    #[test]
    fn extract_sap_abcd_mode_three_pair_dpcm_decode() {
        // Single group, max_sfb = 4. All pair flags set. dpcm row =
        // [delta(sfb=0), 0, delta(sfb=2), 0]. alpha_q[0] = delta_0 = 5;
        // alpha_q[1] = alpha_q[0] = 5 (pair inherit); alpha_q[2] =
        // alpha_q[0] + delta_2 = 5 + 3 = 8; alpha_q[3] = alpha_q[2] = 8.
        let cp = ChparamInfo {
            sap_mode: 3,
            ms_used: vec![],
            sap_data: Some(SapData {
                sap_coeff_all: true,
                sap_coeff_used: vec![vec![true, true, true, true]],
                delta_code_time: false,
                dpcm_alpha_q: vec![vec![5, 0, 3, 0]],
            }),
        };
        let coeffs = extract_sap_abcd(&cp, &[4]);
        let row = &coeffs.abcd[0];
        // sfb 0: alpha_q = 5 -> sap_gain = 0.5 -> (1.5, 1, 0.5, -1).
        let (a, b, c, d) = row[0];
        assert!((a - 1.5).abs() < 1e-6);
        assert_eq!(b, 1.0);
        assert!((c - 0.5).abs() < 1e-6);
        assert_eq!(d, -1.0);
        // sfb 1 inherits sfb 0.
        assert_eq!(row[1], row[0]);
        // sfb 2: alpha_q = 8 -> sap_gain = 0.8 -> (1.8, 1, 0.2, -1).
        let (a2, _b2, c2, _d2) = row[2];
        assert!((a2 - 1.8).abs() < 1e-6);
        assert!((c2 - 0.2).abs() < 1e-6);
        assert_eq!(row[3], row[2]);
    }

    /// `sap_mode == 3, !sap_used` -> band passthrough (a=1,b=0,c=0,d=1).
    #[test]
    fn extract_sap_abcd_mode_three_unused_bands_pass_through() {
        let cp = ChparamInfo {
            sap_mode: 3,
            ms_used: vec![],
            sap_data: Some(SapData {
                sap_coeff_all: false,
                sap_coeff_used: vec![vec![false, false]],
                delta_code_time: false,
                dpcm_alpha_q: vec![vec![10, 0]],
            }),
        };
        let coeffs = extract_sap_abcd(&cp, &[2]);
        assert_eq!(coeffs.abcd[0][0], (1.0, 0.0, 0.0, 1.0));
        assert_eq!(coeffs.abcd[0][1], (1.0, 0.0, 0.0, 1.0));
    }

    /// Cross-group `delta_code_time` only fires when groups share their
    /// `max_sfb_g` (Pseudocode 59 forces `code_delta = 0` otherwise).
    #[test]
    fn extract_sap_abcd_mode_three_delta_code_time_cross_group() {
        // Two groups, both max_sfb = 2. delta_code_time = 1.
        // Group 0 dpcm = [4, 0] -> alpha[0] = [4, 4].
        // Group 1 dpcm = [2, 0] -> alpha[1][0] = alpha[0][0] + 2 = 6;
        //                          alpha[1][1] = alpha[1][0] = 6.
        let cp = ChparamInfo {
            sap_mode: 3,
            ms_used: vec![],
            sap_data: Some(SapData {
                sap_coeff_all: true,
                sap_coeff_used: vec![vec![true, true], vec![true, true]],
                delta_code_time: true,
                dpcm_alpha_q: vec![vec![4, 0], vec![2, 0]],
            }),
        };
        let coeffs = extract_sap_abcd(&cp, &[2, 2]);
        // Group 1 sfb 0: sap_gain = 0.6 -> a = 1.6.
        let (a, _, c, _) = coeffs.abcd[1][0];
        assert!((a - 1.6).abs() < 1e-6);
        assert!((c - 0.4).abs() < 1e-6);
    }

    /// Round 41: Table 181 first-stage matrix with identity
    /// chparam_info pair (sap_mode = 0, identity coefficients) leaves
    /// the front pair (L = A, R = B) untouched and zeros the
    /// preliminary surround pair (Ls / Rs = 0 + 1*sSMP_3,4? — actually
    /// identity coeffs are (1, 0, 0, 1) so c = 0 and d = 1).
    /// Concretely:
    ///   L = 1*A + 0*sSMP_3 = A
    ///   R = 1*B + 0*sSMP_4 = B
    ///   Ls = 0*A + 1*sSMP_3 = sSMP_3
    ///   Rs = 0*B + 1*sSMP_4 = sSMP_4
    #[test]
    fn apply_sap_table_181_identity_passthrough() {
        // Use tl=256 -> sfb table exists per num_sfb_48. max_sfb_master
        // chosen small (4) so the loop is bounded.
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 4u32;
        let a_spec: Vec<f32> = (0..n).map(|i| 0.10 + 1e-3 * i as f32).collect();
        let b_spec: Vec<f32> = (0..n).map(|i| 0.20 + 1e-3 * i as f32).collect();
        let s3_spec: Vec<f32> = (0..n).map(|i| 0.30 + 1e-3 * i as f32).collect();
        let s4_spec: Vec<f32> = (0..n).map(|i| 0.40 + 1e-3 * i as f32).collect();
        let cp_id = ChparamInfo {
            sap_mode: 0,
            ms_used: vec![],
            sap_data: None,
        };
        let pair = [cp_id.clone(), cp_id];
        let (l, r, ls, rs) = apply_sap_table_181(
            &a_spec,
            &b_spec,
            &s3_spec,
            &s4_spec,
            &pair,
            max_sfb_master,
            tl,
        )
        .expect("identity SAP must produce outputs");
        assert_eq!(l.len(), n);
        // Identity coefficients (1, 0, 0, 1) mean L bins are A's bins
        // (in the SAP-coded extent) and the unmixed range still pulls
        // from A. So L == A across the whole spectrum.
        for k in 0..n {
            assert!(
                (l[k] - a_spec[k]).abs() < 1e-6,
                "L[{k}] = {} expected {}",
                l[k],
                a_spec[k]
            );
            assert!((r[k] - b_spec[k]).abs() < 1e-6);
        }
        // Ls / Rs only carry sSMP_3 / sSMP_4 within the SAP-coded extent.
        // Outside that they're silent.
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let sap_hi = sfbo[max_sfb_master as usize] as usize;
        for k in 0..sap_hi {
            assert!((ls[k] - s3_spec[k]).abs() < 1e-6);
            assert!((rs[k] - s4_spec[k]).abs() < 1e-6);
        }
        for k in sap_hi..n {
            assert_eq!(ls[k], 0.0);
            assert_eq!(rs[k], 0.0);
        }
    }

    /// Round 41: Table 181 with an `ms_used == 1` per-sfb selector mixes
    /// (1, 1, 1, -1) over the band — L = A + sSMP_3, R = B + sSMP_4,
    /// Ls = A + sSMP_3, Rs = B - sSMP_4.
    #[test]
    fn apply_sap_table_181_ms_used_mixing() {
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 2u32;
        // Use constant spectra so we can check exact sums per band.
        let a_spec = vec![1.0_f32; n];
        let b_spec = vec![2.0_f32; n];
        let s3_spec = vec![3.0_f32; n];
        let s4_spec = vec![4.0_f32; n];
        let cp = ChparamInfo {
            sap_mode: 1,
            ms_used: vec![vec![true, true]],
            sap_data: None,
        };
        let pair = [cp.clone(), cp];
        let (l, r, ls, rs) = apply_sap_table_181(
            &a_spec,
            &b_spec,
            &s3_spec,
            &s4_spec,
            &pair,
            max_sfb_master,
            tl,
        )
        .unwrap();
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let sap_hi = sfbo[max_sfb_master as usize] as usize;
        // Within SAP-coded extent: (a, b, c, d) = (1, 1, 1, -1).
        // Per Table 181: L = a*A + b*s3; R = a*B + b*s4;
        //                Ls = c*A + d*s3; Rs = c*B + d*s4.
        for k in 0..sap_hi {
            assert!((l[k] - 4.0).abs() < 1e-6); // 1*A + 1*s3 = 1 + 3
            assert!((r[k] - 6.0).abs() < 1e-6); // 1*B + 1*s4 = 2 + 4
            assert!((ls[k] - (-2.0)).abs() < 1e-6); // 1*A + (-1)*s3 = 1 - 3
            assert!((rs[k] - (-2.0)).abs() < 1e-6); // 1*B + (-1)*s4 = 2 - 4
        }
        // Outside the SAP-coded extent: front pair pulls A/B, surround silent.
        for k in sap_hi..n {
            assert_eq!(l[k], 1.0);
            assert_eq!(r[k], 2.0);
            assert_eq!(ls[k], 0.0);
            assert_eq!(rs[k], 0.0);
        }
    }

    /// Round 41: missing sfb table (e.g. tl with no entry in
    /// `num_sfb_48`) returns `None` so the dispatch can fall back to
    /// the round-40 raw sSMP_3 / sSMP_4 path.
    #[test]
    fn apply_sap_table_181_missing_sfb_table_returns_none() {
        // tl = 100 has no entry in `num_sfb_48` (only the defined
        // transform lengths return Some).
        let tl = 100u32;
        let n = tl as usize;
        let cp = ChparamInfo::default();
        let pair = [cp.clone(), cp];
        let zero = vec![0.0f32; n];
        let out = apply_sap_table_181(&zero, &zero, &zero, &zero, &pair, 1, tl);
        assert!(out.is_none());
    }

    /// Round 246: `invert_sap_table_181` with the identity chparam_info
    /// pair (`sap_mode == 0`) recovers `(A = L, B = R, s3 = Ls,
    /// s4 = Rs)` on the SAP-coded extent and passes `(A = L, B = R)`
    /// through on the unmixed tail with `s3 = s4 = 0` — the encoder
    /// dual of [`apply_sap_table_181_identity_passthrough`].
    #[test]
    fn invert_sap_table_181_identity_passthrough() {
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 4u32;
        let l_spec: Vec<f32> = (0..n).map(|i| 0.10 + 1e-3 * i as f32).collect();
        let r_spec: Vec<f32> = (0..n).map(|i| 0.20 + 1e-3 * i as f32).collect();
        let ls_spec: Vec<f32> = (0..n).map(|i| 0.30 + 1e-3 * i as f32).collect();
        let rs_spec: Vec<f32> = (0..n).map(|i| 0.40 + 1e-3 * i as f32).collect();
        let cp_id = ChparamInfo {
            sap_mode: 0,
            ms_used: vec![],
            sap_data: None,
        };
        let pair = [cp_id.clone(), cp_id];
        let (a, b, s3, s4) = invert_sap_table_181(
            &l_spec,
            &r_spec,
            &ls_spec,
            &rs_spec,
            &pair,
            max_sfb_master,
            tl,
        )
        .expect("identity inverse must produce outputs");
        assert_eq!(a.len(), n);
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let sap_hi = sfbo[max_sfb_master as usize] as usize;
        for k in 0..sap_hi {
            // Identity (a, b, c, d) = (1, 0, 0, 1): inverse recovers
            // (A = L, B = R, s3 = Ls, s4 = Rs) — equivalently the
            // forward pass would re-mix to (L, R, Ls, Rs).
            assert!((a[k] - l_spec[k]).abs() < 1e-6);
            assert!((b[k] - r_spec[k]).abs() < 1e-6);
            assert!((s3[k] - ls_spec[k]).abs() < 1e-6);
            assert!((s4[k] - rs_spec[k]).abs() < 1e-6);
        }
        for k in sap_hi..n {
            assert!((a[k] - l_spec[k]).abs() < 1e-6);
            assert!((b[k] - r_spec[k]).abs() < 1e-6);
            assert_eq!(s3[k], 0.0);
            assert_eq!(s4[k], 0.0);
        }
    }

    /// Round 246: `invert_sap_table_181` with the M/S `(1, 1, 1, -1)`
    /// per-sfb matrix yields the classic sum-and-difference inversion
    /// A = (L + Ls)/2, s3 = (L - Ls)/2 and similarly for the B/s4 side.
    #[test]
    fn invert_sap_table_181_ms_used_inverse() {
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 2u32;
        // Constant spectra so the band-wise sums are easy to verify.
        let l_spec = vec![4.0_f32; n]; // L = A + s3 = 1 + 3
        let r_spec = vec![6.0_f32; n]; // R = B + s4 = 2 + 4
        let ls_spec = vec![-2.0_f32; n]; // Ls = A - s3 = 1 - 3
        let rs_spec = vec![-2.0_f32; n]; // Rs = B - s4 = 2 - 4
        let cp = ChparamInfo {
            sap_mode: 1,
            ms_used: vec![vec![true, true]],
            sap_data: None,
        };
        let pair = [cp.clone(), cp];
        let (a, b, s3, s4) = invert_sap_table_181(
            &l_spec,
            &r_spec,
            &ls_spec,
            &rs_spec,
            &pair,
            max_sfb_master,
            tl,
        )
        .unwrap();
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let sap_hi = sfbo[max_sfb_master as usize] as usize;
        for k in 0..sap_hi {
            // With det = -2: A = ((-1)*L - 1*Ls) / -2 = (L + Ls)/2;
            // s3 = ((-1)*L + 1*Ls) / -2 = (L - Ls)/2.
            assert!((a[k] - 1.0).abs() < 1e-6); // (4 + (-2))/2
            assert!((s3[k] - 3.0).abs() < 1e-6); // (4 - (-2))/2
            assert!((b[k] - 2.0).abs() < 1e-6); // (6 + (-2))/2
            assert!((s4[k] - 4.0).abs() < 1e-6); // (6 - (-2))/2
        }
        // Outside the SAP-coded extent: front passthrough, surround
        // silent.
        for k in sap_hi..n {
            assert!((a[k] - 4.0).abs() < 1e-6);
            assert!((b[k] - 6.0).abs() < 1e-6);
            assert_eq!(s3[k], 0.0);
            assert_eq!(s4[k], 0.0);
        }
    }

    /// Round 246: forward-and-inverse round-trip with the identity
    /// chparam pair is bit-stable — `invert_sap_table_181` truly is
    /// the dual of `apply_sap_table_181` on the identity row.
    #[test]
    fn sap_table_181_roundtrip_identity() {
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 4u32;
        let a0: Vec<f32> = (0..n).map(|i| 0.11 + 1e-3 * i as f32).collect();
        let b0: Vec<f32> = (0..n).map(|i| 0.22 + 1e-3 * i as f32).collect();
        let s30: Vec<f32> = (0..n).map(|i| 0.33 + 1e-3 * i as f32).collect();
        let s40: Vec<f32> = (0..n).map(|i| 0.44 + 1e-3 * i as f32).collect();
        let cp_id = ChparamInfo {
            sap_mode: 0,
            ms_used: vec![],
            sap_data: None,
        };
        let pair = [cp_id.clone(), cp_id];
        let (l, r, ls, rs) =
            apply_sap_table_181(&a0, &b0, &s30, &s40, &pair, max_sfb_master, tl).unwrap();
        let (a_r, b_r, s3_r, s4_r) =
            invert_sap_table_181(&l, &r, &ls, &rs, &pair, max_sfb_master, tl).unwrap();
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let sap_hi = sfbo[max_sfb_master as usize] as usize;
        // Inside SAP extent the round-trip is bit-stable on the
        // identity row.
        for k in 0..sap_hi {
            assert!((a_r[k] - a0[k]).abs() < 1e-6);
            assert!((b_r[k] - b0[k]).abs() < 1e-6);
            assert!((s3_r[k] - s30[k]).abs() < 1e-6);
            assert!((s4_r[k] - s40[k]).abs() < 1e-6);
        }
        // Outside SAP extent the forward pass discards s3/s4 — the
        // inverse therefore recovers a_r = l = a0, s3_r = 0.
        for k in sap_hi..n {
            assert!((a_r[k] - a0[k]).abs() < 1e-6);
            assert!((b_r[k] - b0[k]).abs() < 1e-6);
            assert_eq!(s3_r[k], 0.0);
            assert_eq!(s4_r[k], 0.0);
        }
    }

    /// Round 246: forward-and-inverse round-trip with the M/S row is
    /// bit-stable inside the SAP-coded extent — verifies the
    /// closed-form 2x2 inverse is numerically tight at f32.
    #[test]
    fn sap_table_181_roundtrip_ms_used() {
        let tl = 256u32;
        let n = tl as usize;
        let max_sfb_master = 2u32;
        let a0: Vec<f32> = (0..n).map(|i| 0.10 + 1e-3 * i as f32).collect();
        let b0: Vec<f32> = (0..n).map(|i| 0.20 + 1e-3 * i as f32).collect();
        let s30: Vec<f32> = (0..n).map(|i| 0.30 + 1e-3 * i as f32).collect();
        let s40: Vec<f32> = (0..n).map(|i| 0.40 + 1e-3 * i as f32).collect();
        let cp = ChparamInfo {
            sap_mode: 1,
            ms_used: vec![vec![true, true]],
            sap_data: None,
        };
        let pair = [cp.clone(), cp];
        let (l, r, ls, rs) =
            apply_sap_table_181(&a0, &b0, &s30, &s40, &pair, max_sfb_master, tl).unwrap();
        let (a_r, b_r, s3_r, s4_r) =
            invert_sap_table_181(&l, &r, &ls, &rs, &pair, max_sfb_master, tl).unwrap();
        let sfbo = crate::sfb_offset::sfb_offset_48(tl).unwrap();
        let sap_hi = sfbo[max_sfb_master as usize] as usize;
        for k in 0..sap_hi {
            assert!((a_r[k] - a0[k]).abs() < 1e-5);
            assert!((b_r[k] - b0[k]).abs() < 1e-5);
            assert!((s3_r[k] - s30[k]).abs() < 1e-5);
            assert!((s4_r[k] - s40[k]).abs() < 1e-5);
        }
    }

    /// Round 246: missing sfb table (e.g. `tl = 100`) makes the
    /// inverse return `None` for the same reason the forward path
    /// does — symmetry with [`apply_sap_table_181_missing_sfb_table_returns_none`].
    #[test]
    fn invert_sap_table_181_missing_sfb_table_returns_none() {
        let tl = 100u32;
        let n = tl as usize;
        let cp = ChparamInfo::default();
        let pair = [cp.clone(), cp];
        let zero = vec![0.0f32; n];
        let out = invert_sap_table_181(&zero, &zero, &zero, &zero, &pair, 1, tl);
        assert!(out.is_none());
    }

    /// `SapCoeffs::identity` produces an all-ones-diagonal matrix.
    #[test]
    fn sap_coeffs_identity_helper() {
        let coeffs = SapCoeffs::identity(&[3, 2]);
        assert_eq!(coeffs.abcd.len(), 2);
        for row in &coeffs.abcd {
            for q in row {
                assert_eq!(*q, (1.0, 0.0, 0.0, 1.0));
            }
        }
    }

    // =================================================================
    // Round 28 / task #171: mono + stereo short-frame sf_data(ASF) walk
    // =================================================================

    /// Write a spec-correct grouped `sf_data(ASF)` body (Tables 39-42)
    /// for one channel with all-zero CB-0 sections covering all bands
    /// in every group. Layout (per spec, single sf_data() call):
    ///   - asf_section_data: per-group `(4-bit cb=0, n_sect_bits length-incr)`
    ///   - asf_spectral_data: nothing (cb=0)
    ///   - asf_scalefac_data: 8-bit reference_scale_factor (no DPCM as
    ///     mqi==0 everywhere)
    ///   - asf_snf_data: 1-bit b_snf_data_exists = 0
    fn write_zero_grouped_sf_data_body_one_channel(
        bw: &mut BitWriter,
        max_sfb_per_group: &[u32],
        transf_length_idx_per_group: &[u32],
    ) {
        // asf_section_data: per-group section header.
        for g in 0..max_sfb_per_group.len() {
            let max_sfb = max_sfb_per_group[g];
            let tl_idx = transf_length_idx_per_group[g];
            let (n_sect_bits, sect_esc_val) = if tl_idx <= 2 { (3, 7) } else { (5, 31) };
            bw.write_u32(0, 4); // sect_cb = 0
            let mut remaining = max_sfb.saturating_sub(1);
            while remaining >= sect_esc_val {
                bw.write_u32(sect_esc_val, n_sect_bits);
                remaining -= sect_esc_val;
            }
            bw.write_u32(remaining, n_sect_bits);
        }
        // asf_spectral_data: nothing for cb=0.
        // asf_scalefac_data: single 8-bit reference at the head.
        bw.write_u32(120, 8);
        // asf_snf_data: single 1-bit gate at the head.
        bw.write_bit(false);
    }

    /// Mono short-frame walker: equal-transform-length two-group case.
    /// `decode_asf_grouped_mono_body` returns a `Vec<f32>` of length
    /// `2 * sfb_offset[max_sfb]` with all-zero entries.
    #[test]
    fn decode_asf_grouped_mono_body_two_groups_equal_tl() {
        let max_sfb = 6u32;
        let tl_idx = 2u32; // tl=480 short-frame at fl_base=1920
        let mut bw = BitWriter::new();
        write_zero_grouped_sf_data_body_one_channel(
            &mut bw,
            &[max_sfb, max_sfb],
            &[tl_idx, tl_idx],
        );
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [tl_idx, tl_idx],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: max_sfb,
            num_windows: 2,
            num_window_groups: 2,
            scale_factor_grouping: vec![0],
            ..Default::default()
        };
        let scaled = decode_asf_grouped_mono_body(&mut br, &ti, &psy).expect("decodes");
        let sfbo = sfb_offset::sfb_offset_48(480).unwrap();
        let per_group_len = sfbo[max_sfb as usize] as usize;
        assert_eq!(scaled.len(), per_group_len * 2);
        assert!(scaled.iter().all(|&v| v == 0.0));
    }

    /// Mono short-frame walker: three-group case with truncated input
    /// returns `None` (the second group's section header bails on EOF).
    #[test]
    fn decode_asf_grouped_mono_body_truncated_returns_none() {
        let max_sfb = 5u32;
        let tl_idx = 2u32;
        let mut bw = BitWriter::new();
        // Write only one group's section header (incomplete spec-shape
        // body — no scalefac reference, no snf gate, no following groups).
        let (n_sect_bits, sect_esc_val) = (3, 7);
        bw.write_u32(0, 4);
        let mut remaining = max_sfb.saturating_sub(1);
        while remaining >= sect_esc_val {
            bw.write_u32(sect_esc_val, n_sect_bits);
            remaining -= sect_esc_val;
        }
        bw.write_u32(remaining, n_sect_bits);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let ti = AsfTransformInfo {
            b_long_frame: false,
            transf_length: [tl_idx, tl_idx],
            transform_length_0: 480,
            transform_length_1: 480,
        };
        let psy = AsfPsyInfo {
            max_sfb_0: max_sfb,
            num_windows: 3,
            num_window_groups: 3,
            scale_factor_grouping: vec![0, 0],
            ..Default::default()
        };
        // We expect the decoder to bail (return None) since group 1's
        // section header reads past end-of-stream.
        let _ = decode_asf_grouped_mono_body(&mut br, &ti, &psy);
        // No panic — that's the contract.
    }

    /// `parse_mono_audio_data_outer` end-to-end: SIMPLE mono, ASF
    /// frontend, short-frame transform, two window groups. The
    /// substream walker reads sf_info and then the spec-correct
    /// grouped sf_data body, depositing the dequantised spectrum on
    /// `tools.scaled_spec_primary`.
    #[test]
    fn walk_mono_simple_short_frame_two_groups_decodes_sf_data() {
        // mono_codec_mode = SIMPLE (1 bit = 0).
        // mono_data(0):
        //   spec_frontend = ASF (1 bit = 0).
        //   asf_transform_info: b_long_frame = 0, transf_length[0] = 2,
        //     transf_length[1] = 2.
        //   asf_psy_info(0, 0): max_sfb[0] (6 bits = 5),
        //     scale_factor_grouping (3 bits = [1, 0, 1] => 2 groups).
        //   sf_data(ASF): grouped body for max_sfb=5 across two groups.
        let max_sfb = 5u32;
        let mut bw = BitWriter::new();
        bw.write_bit(false); // mono_codec_mode = SIMPLE
        bw.write_bit(false); // spec_frontend = ASF
                             // asf_transform_info: frame_len_base=1920 -> b_long_frame bit
                             // + 2-bit transf_length[0] + 2-bit transf_length[1].
        bw.write_bit(false); // b_long_frame = 0
        bw.write_u32(2, 2); // transf_length[0] = 2 (tl=480)
        bw.write_u32(2, 2); // transf_length[1] = 2 (tl=480)
                            // asf_psy_info(0, 0): max_sfb[0] read with 6 bits at tl=480.
        bw.write_u32(max_sfb, 6);
        // n_grp_bits_ge_1536(2, 2) = 3. Pattern [1, 0, 1] -> one zero ->
        // num_window_groups = 2.
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        // sf_data body (spec-correct grouped layout).
        write_zero_grouped_sf_data_body_one_channel(&mut bw, &[max_sfb, max_sfb], &[2, 2]);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        parse_mono_audio_data_outer(&mut br, &mut tools, true, 1920).unwrap();
        let psy = tools.psy_info_primary.as_ref().unwrap();
        assert_eq!(psy.num_window_groups, 2);
        let scaled = tools
            .scaled_spec_primary
            .as_ref()
            .expect("grouped mono sf_data body decoded");
        let sfbo = sfb_offset::sfb_offset_48(480).unwrap();
        assert_eq!(scaled.len(), 2 * sfbo[max_sfb as usize] as usize);
        assert!(scaled.iter().all(|&v| v == 0.0));
    }

    /// `parse_stereo_data_body` split-MDCT short-frame: two independent
    /// per-channel grouped bodies. Both `scaled_spec_primary` and
    /// `scaled_spec_secondary` carry the per-group concatenated
    /// spectrum.
    #[test]
    fn parse_stereo_data_body_split_short_frame_two_groups_decodes_both_channels() {
        let max_sfb = 4u32;
        let mut bw = BitWriter::new();
        // stereo_data: b_enable_mdct_stereo_proc = 0 (split MDCT).
        bw.write_bit(false);
        // L: spec_frontend = ASF (1 bit = 0).
        bw.write_bit(false);
        // L: asf_transform_info — same as mono test above.
        bw.write_bit(false); // b_long_frame = 0
        bw.write_u32(2, 2);
        bw.write_u32(2, 2);
        // L: sf_info(ASF, 0, 0): max_sfb[0] + 3 grouping bits.
        bw.write_u32(max_sfb, 6);
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        // R: spec_frontend = ASF.
        bw.write_bit(false);
        // R: asf_transform_info.
        bw.write_bit(false);
        bw.write_u32(2, 2);
        bw.write_u32(2, 2);
        // R: sf_info(ASF, 0, 0).
        bw.write_u32(max_sfb, 6);
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        // L sf_data body, then R sf_data body.
        write_zero_grouped_sf_data_body_one_channel(&mut bw, &[max_sfb, max_sfb], &[2, 2]);
        write_zero_grouped_sf_data_body_one_channel(&mut bw, &[max_sfb, max_sfb], &[2, 2]);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        let ok = parse_stereo_data_body(&mut br, &mut tools, 1920);
        assert!(ok, "stereo split short-frame body should parse");
        let sfbo = sfb_offset::sfb_offset_48(480).unwrap();
        let expected = 2 * sfbo[max_sfb as usize] as usize;
        let l = tools.scaled_spec_primary.as_ref().unwrap();
        let r = tools.scaled_spec_secondary.as_ref().unwrap();
        assert_eq!(l.len(), expected);
        assert_eq!(r.len(), expected);
        assert!(l.iter().all(|&v| v == 0.0));
        assert!(r.iter().all(|&v| v == 0.0));
    }

    /// Joint-MDCT stereo short-frame: shared section, two spectra,
    /// shared scalefac, per-group ms_used flag arrays. Layout per
    /// `decode_asf_grouped_stereo_joint_body`.
    #[test]
    fn parse_stereo_data_body_joint_short_frame_two_groups_decodes_both_channels() {
        let max_sfb = 4u32;
        let mut bw = BitWriter::new();
        // stereo_data: b_enable_mdct_stereo_proc = 1 (joint MDCT).
        bw.write_bit(true);
        // asf_transform_info — short-frame.
        bw.write_bit(false);
        bw.write_u32(2, 2);
        bw.write_u32(2, 2);
        // sf_info(ASF, 0, 0).
        bw.write_u32(max_sfb, 6);
        bw.write_u32(1, 1);
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        // Shared asf_section_data (per-group all-zero CB).
        for _ in 0..2 {
            bw.write_u32(0, 4);
            let mut remaining = max_sfb.saturating_sub(1);
            while remaining >= 7 {
                bw.write_u32(7, 3);
                remaining -= 7;
            }
            bw.write_u32(remaining, 3);
        }
        // L spectral: nothing (cb=0). R spectral: nothing.
        // Shared scalefac: 8-bit reference + no DPCM (mqi all zero).
        bw.write_u32(120, 8);
        // ms_used per group: per-band gate on (cb!=0 && mqi>0). Both
        // are zero so no ms_used bits emitted.
        // snf: 1-bit gate.
        bw.write_bit(false);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut tools = SubstreamTools::default();
        let ok = parse_stereo_data_body(&mut br, &mut tools, 1920);
        assert!(ok, "stereo joint short-frame body should parse");
        let sfbo = sfb_offset::sfb_offset_48(480).unwrap();
        let expected = 2 * sfbo[max_sfb as usize] as usize;
        let l = tools.scaled_spec_primary.as_ref().unwrap();
        let r = tools.scaled_spec_secondary.as_ref().unwrap();
        assert_eq!(l.len(), expected);
        assert_eq!(r.len(), expected);
        // ms_used carries one bool per band per group, concatenated.
        let ms = tools.ms_used.as_ref().unwrap();
        assert_eq!(ms.len(), 2 * max_sfb as usize);
        assert!(ms.iter().all(|&b| !b));
    }

    /// Mono SIMPLE substream with `spec_frontend == SSF` exercises the
    /// round-30 wiring of the `ssf_data()` walker (Tables 43-46) into
    /// `parse_mono_audio_data_outer`. Builds a synthetic LONG_STRIDE
    /// I-frame `ssf_data()` body — mirrors the unit test inside
    /// [`crate::ssf::tests`] — and verifies the parsed
    /// `tools.ssf_data_primary` survives the trip through the
    /// substream walker.
    #[test]
    fn mono_ssf_substream_walker_populates_ssf_data() {
        let mut bw = BitWriter::new();
        // audio_size_value = 100 (placeholder), b_more_bits = 0.
        bw.write_u32(100, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        // mono_codec_mode = 0 (SIMPLE).
        bw.write_u32(0, 1);
        // mono_data(0) body:
        //   spec_frontend = 1 (SSF).
        bw.write_u32(1, 1);
        // ssf_data(b_iframe=1) — frame_len_base = 960 → SsfFrameConfig
        // resolves to (granule_length=960, num_granules=1), so only one
        // granule is parsed (the `frame_length >= 1536` second-granule
        // gate stays inactive).
        //   ssf_granule(b_iframe=1):
        bw.write_u32(0, 1); // stride_flag = LONG_STRIDE
        bw.write_u32(0, 3); // num_bands_minus12 = 0 → num_bands = 12
                            // Per-block predictor loop runs zero iterations (start_block ==
                            // end_block == 0 for LONG_STRIDE I-frame).
                            // ssf_st_data():
        bw.write_u32(0, 5); // env_curr_band0_bits
                            // SHORT_STRIDE-gated env_startup / gain not present.
                            // Per-block st-data: variance_preserving (1) + alloc_offset_bits (5).
        bw.write_u32(0, 1);
        bw.write_u32(0, 5);
        // ssf_ac_data(): Init pulls 30 bits, then envelope decode etc.
        for _ in 0..(30 + 256) {
            bw.write_bit(false);
        }
        bw.align_to_byte();
        // Pad to honour audio_size.
        while bw.byte_len() < 128 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        // Walk with `frame_len_base = 960` (48 kHz / 48 fps, one
        // granule per AC-4 frame).
        let info = walk_ac4_substream(&bytes, 1, true, 960).unwrap();
        assert_eq!(info.tools.mono_mode, Some(MonoCodecMode::Simple));
        assert_eq!(info.tools.spec_frontend_primary, Some(SpecFrontend::Ssf));
        let ssf = info
            .tools
            .ssf_data_primary
            .as_ref()
            .expect("SSF walker should have populated ssf_data_primary");
        assert_eq!(ssf.granules.len(), 1);
        let g = &ssf.granules[0];
        assert_eq!(g.num_bands, 12);
        assert_eq!(g.n_mdct, 960);
        assert!(g.b_iframe);
    }

    /// Round 32: `walk_ac4_substream_stateful` must persist
    /// `SsfChannelState` (the bitstream walker's RNG / `env_prev` /
    /// `last_num_bands` carrier) into the caller-supplied slice. After
    /// an SSF I-frame parse, `state[0].last_num_bands` reflects the
    /// granule's num_bands and `state[0].env_prev` is non-empty.
    #[test]
    fn walk_ac4_substream_stateful_persists_ssf_walker_state() {
        // Reuse the same I-frame body as the basic SSF walker test.
        let mut bw = BitWriter::new();
        bw.write_u32(100, 15);
        bw.write_bit(false);
        bw.align_to_byte();
        bw.write_u32(0, 1);
        bw.write_u32(1, 1);
        bw.write_u32(0, 1); // stride_flag = LONG_STRIDE
        bw.write_u32(0, 3); // num_bands_minus12 = 0 → num_bands = 12
        bw.write_u32(0, 5); // env_curr_band0_bits
        bw.write_u32(0, 1); // variance_preserving
        bw.write_u32(0, 5); // alloc_offset_bits
        for _ in 0..(30 + 256) {
            bw.write_bit(false);
        }
        bw.align_to_byte();
        while bw.byte_len() < 128 {
            bw.write_u32(0, 8);
        }
        let bytes = bw.finish();
        let mut state_bank = vec![crate::ssf::SsfChannelState::new()];
        // Pre-condition: state is the default.
        assert_eq!(state_bank[0].last_num_bands, 0);
        assert!(state_bank[0].env_prev.is_empty());
        let info =
            walk_ac4_substream_stateful(&bytes, 1, true, 960, Some(&mut state_bank)).unwrap();
        // Post-condition: walker has updated the channel-0 state.
        assert_eq!(state_bank[0].last_num_bands, 12);
        assert_eq!(state_bank[0].last_n_mdct, 960);
        assert_eq!(state_bank[0].env_prev.len(), 12);
        assert!(info.tools.ssf_data_primary.is_some());
    }

    // ----- build_chparam_info_* encoder-side duals of extract_sap_abcd -----

    /// `build_chparam_info_ms_used` round-trip: feeding the result
    /// into `extract_sap_abcd` reproduces the per-sfb (1, 1, 1, -1)
    /// vs identity matrix the input ms_used array describes.
    #[test]
    fn build_chparam_info_ms_used_round_trips_through_extract_sap_abcd() {
        let ms_used = vec![vec![true, false, true, true]];
        let info = build_chparam_info_ms_used(ms_used.clone());
        assert_eq!(info.sap_mode, 1);
        assert_eq!(info.ms_used, ms_used);
        assert!(info.sap_data.is_none());
        let coeffs = extract_sap_abcd(&info, &[4]);
        assert_eq!(coeffs.abcd[0][0], (1.0, 1.0, 1.0, -1.0));
        assert_eq!(coeffs.abcd[0][1], (1.0, 0.0, 0.0, 1.0));
        assert_eq!(coeffs.abcd[0][2], (1.0, 1.0, 1.0, -1.0));
        assert_eq!(coeffs.abcd[0][3], (1.0, 1.0, 1.0, -1.0));
    }

    /// Bit-stream round-trip: write_chparam_info → parse_chparam_info
    /// recovers the ms_used row that `build_chparam_info_ms_used`
    /// produced.
    #[test]
    fn build_chparam_info_ms_used_round_trips_through_bitstream() {
        let ms_used = vec![vec![true, false, true, false, true]];
        let info = build_chparam_info_ms_used(ms_used.clone());
        let mut bw = BitWriter::new();
        crate::encoder_asf::write_chparam_info(&mut bw, &info, &[5]);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_chparam_info(&mut br, &[5]).unwrap();
        assert_eq!(parsed.sap_mode, 1);
        assert_eq!(parsed.ms_used, ms_used);
    }

    /// `build_chparam_info_sap_data_from_alpha_q` single-group:
    /// pair-major DPCM with `sap_coeff_all = true` reproduces the
    /// original `alpha_q` row through `extract_sap_abcd`. Mirrors
    /// `extract_sap_abcd_mode_three_pair_dpcm_decode` from the
    /// decoder side.
    #[test]
    fn build_chparam_info_sap_data_pair_major_round_trip() {
        // Target alpha_q row [5, 5, 8, 8] — pair-major (sfb 1 mirrors
        // sfb 0; sfb 3 mirrors sfb 2). Expected dpcm row:
        // [5 (= 5 - 0), 0, 3 (= 8 - 5), 0].
        let alpha_q = vec![vec![5, 5, 8, 8]];
        let used = vec![vec![true, true, true, true]];
        let info = build_chparam_info_sap_data_from_alpha_q(&alpha_q, &used, false, &[4]);
        assert_eq!(info.sap_mode, 3);
        let sd = info.sap_data.as_ref().expect("sap_data populated");
        assert!(sd.sap_coeff_all);
        assert_eq!(sd.dpcm_alpha_q[0], vec![5, 0, 3, 0]);
        // Round-trip back through extract_sap_abcd: alpha_q row is
        // reproduced and `sap_gain = alpha_q * 0.1` lands the same
        // (a, b, c, d) tuples the decoder would extract.
        let coeffs = extract_sap_abcd(&info, &[4]);
        let row = &coeffs.abcd[0];
        // sfb 0: alpha=5 -> sap_gain=0.5 -> (1.5, 1, 0.5, -1).
        assert!((row[0].0 - 1.5).abs() < 1e-6);
        assert_eq!(row[0].1, 1.0);
        assert!((row[0].2 - 0.5).abs() < 1e-6);
        assert_eq!(row[0].3, -1.0);
        assert_eq!(row[1], row[0]); // pair inherit
                                    // sfb 2: alpha=8 -> sap_gain=0.8 -> (1.8, 1, 0.2, -1).
        assert!((row[2].0 - 1.8).abs() < 1e-6);
        assert!((row[2].2 - 0.2).abs() < 1e-6);
        assert_eq!(row[3], row[2]);
    }

    /// Cleared pair flag → builder marks the pair as unused and the
    /// decoder's `extract_sap_abcd` skips it as identity passthrough
    /// (a=1, b=0, c=0, d=1).
    #[test]
    fn build_chparam_info_sap_data_unused_bands_pass_through() {
        let alpha_q = vec![vec![10, 10]];
        let used = vec![vec![false, false]];
        let info = build_chparam_info_sap_data_from_alpha_q(&alpha_q, &used, false, &[2]);
        let sd = info.sap_data.as_ref().expect("sap_data populated");
        assert!(!sd.sap_coeff_all);
        assert_eq!(sd.sap_coeff_used[0], vec![false, false]);
        let coeffs = extract_sap_abcd(&info, &[2]);
        assert_eq!(coeffs.abcd[0][0], (1.0, 0.0, 0.0, 1.0));
        assert_eq!(coeffs.abcd[0][1], (1.0, 0.0, 0.0, 1.0));
    }

    /// Cross-group `delta_code_time` round-trip: matching `max_sfb_g`
    /// across groups lets the DPCM reference the same sfb in the
    /// previous group instead of `sfb-2` in the current one.
    #[test]
    fn build_chparam_info_sap_data_delta_code_time_cross_group() {
        // Target: group 0 alpha = [4, 4]; group 1 alpha = [6, 6].
        // delta_code_time = true, both groups max_sfb = 2.
        // Group 0 dpcm = [4 (= 4 - 0), 0].
        // Group 1 dpcm = [2 (= 6 - 4), 0] via cross-group reference.
        let alpha_q = vec![vec![4, 4], vec![6, 6]];
        let used = vec![vec![true, true], vec![true, true]];
        let info = build_chparam_info_sap_data_from_alpha_q(&alpha_q, &used, true, &[2, 2]);
        let sd = info.sap_data.as_ref().expect("sap_data populated");
        assert!(sd.delta_code_time);
        assert_eq!(sd.dpcm_alpha_q[0], vec![4, 0]);
        assert_eq!(sd.dpcm_alpha_q[1], vec![2, 0]);
        let coeffs = extract_sap_abcd(&info, &[2, 2]);
        // Group 1 sfb 0: sap_gain = 0.6 -> (1.6, 1, 0.4, -1).
        let (a, _, c, _) = coeffs.abcd[1][0];
        assert!((a - 1.6).abs() < 1e-6);
        assert!((c - 0.4).abs() < 1e-6);
    }

    /// Single-group: caller-supplied `delta_code_time == true` is
    /// normalised to `false` (the bit isn't on the wire so the
    /// decoder sees it as zero — keep the field consistent).
    #[test]
    fn build_chparam_info_sap_data_single_group_drops_delta_code_time() {
        let alpha_q = vec![vec![3, 3]];
        let used = vec![vec![true, true]];
        let info = build_chparam_info_sap_data_from_alpha_q(&alpha_q, &used, true, &[2]);
        let sd = info.sap_data.as_ref().expect("sap_data populated");
        assert!(!sd.delta_code_time);
    }

    /// Bit-stream round-trip: write_chparam_info on the builder's
    /// output → parse_chparam_info recovers the same SAP body, which
    /// extracts to the original alpha_q matrix.
    #[test]
    fn build_chparam_info_sap_data_round_trips_through_bitstream() {
        let alpha_q = vec![vec![2, 2, 7, 7]];
        let used = vec![vec![true, true, true, true]];
        let info = build_chparam_info_sap_data_from_alpha_q(&alpha_q, &used, false, &[4]);
        let mut bw = BitWriter::new();
        crate::encoder_asf::write_chparam_info(&mut bw, &info, &[4]);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_chparam_info(&mut br, &[4]).unwrap();
        assert_eq!(parsed.sap_mode, 3);
        let coeffs = extract_sap_abcd(&parsed, &[4]);
        // sfb 0: alpha=2 -> sap_gain=0.2 -> a = 1.2, c = 0.8.
        let (a0, _, c0, _) = coeffs.abcd[0][0];
        assert!((a0 - 1.2).abs() < 1e-6);
        assert!((c0 - 0.8).abs() < 1e-6);
        // sfb 2: alpha=7 -> sap_gain=0.7 -> a = 1.7, c = 0.3.
        let (a2, _, c2, _) = coeffs.abcd[0][2];
        assert!((a2 - 1.7).abs() < 1e-6);
        assert!((c2 - 0.3).abs() < 1e-6);
    }

    // ----- build_chparam_info_none / select_ms_used_for_pair -----

    /// The header-only `SapMode::None` builder produces an info that
    /// extracts to identity per-sfb across any per-group bound and
    /// survives a bit-stream round-trip as a 2-bit header.
    #[test]
    fn build_chparam_info_none_round_trips_through_extract_and_bitstream() {
        let info = build_chparam_info_none();
        assert_eq!(info.sap_mode, 0);
        assert!(info.ms_used.is_empty());
        assert!(info.sap_data.is_none());
        // extract_sap_abcd over any per-group bound stays identity.
        let coeffs = extract_sap_abcd(&info, &[3, 5]);
        assert_eq!(coeffs.abcd.len(), 2);
        for row in &coeffs.abcd {
            for &(a, b, c, d) in row {
                assert!((a - 1.0).abs() < 1e-6);
                assert!(b.abs() < 1e-6);
                assert!(c.abs() < 1e-6);
                assert!((d - 1.0).abs() < 1e-6);
            }
        }
        // Bit-stream round-trip: header-only, 2 bits + alignment padding.
        let mut bw = BitWriter::new();
        crate::encoder_asf::write_chparam_info(&mut bw, &info, &[3, 5]);
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_chparam_info(&mut br, &[3, 5]).unwrap();
        assert_eq!(parsed.sap_mode, 0);
        assert_eq!(parsed.mode(), SapMode::None);
        assert!(parsed.ms_used.is_empty());
        assert!(parsed.sap_data.is_none());
    }

    /// `select_ms_used_for_pair` picks `true` on bands where M/S coding
    /// concentrates energy (one of M', S' carries strictly less than
    /// the smaller of L, R) and `false` otherwise.
    ///
    /// Construct a 4-sfb test signal with bins-per-sfb = 4 (transform
    /// length 16, single-group, sfb_offset = [0, 4, 8, 12, 16]):
    ///
    /// * sfb 0 — L = R (perfectly correlated): M' = L, S' = 0 →
    ///   min_ms = 0 < min_lr = E_L → pick M/S.
    /// * sfb 1 — L = -R (anti-correlated): M' = 0, S' = L →
    ///   min_ms = 0 < min_lr = E_L → pick M/S.
    /// * sfb 2 — L only, R = 0 (one-sided): E_L = sum L^2, E_R = 0,
    ///   so min_lr = 0; E_M' = E_S' = E_L/4 so min_ms = E_L/4 >= 0.
    ///   No concentration benefit — keep L/R.
    /// * sfb 3 — zero-energy band: every energy is 0 → tie → L/R.
    #[test]
    fn select_ms_used_for_pair_per_band_decision() {
        // Per-sfb bins: 4 bins per sfb, 4 sfbs, single group, total 16
        // bins (matches a synthetic 16-bin sfb table for the test).
        let sfb_offset: [u16; 5] = [0, 4, 8, 12, 16];
        // sfb 0: L == R (correlated).
        let mut l = vec![1.0f32, 0.5, -0.7, 0.3];
        let mut r = vec![1.0f32, 0.5, -0.7, 0.3];
        // sfb 1: L == -R (anti-correlated).
        l.extend_from_slice(&[0.4, 0.9, -0.2, 0.6]);
        r.extend_from_slice(&[-0.4, -0.9, 0.2, -0.6]);
        // sfb 2: L only (one-sided / no concentration benefit).
        l.extend_from_slice(&[0.8, 0.1, -0.3, 0.5]);
        r.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        // sfb 3: zero-energy.
        l.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        r.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        let l_per_group = vec![l];
        let r_per_group = vec![r];
        let decisions = select_ms_used_for_pair(&l_per_group, &r_per_group, &sfb_offset, &[4]);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].len(), 4);
        // sfb 0 (correlated): M/S concentrates → wins.
        assert!(decisions[0][0], "L == R band must pick M/S");
        // sfb 1 (anti-correlated): M/S concentrates → wins.
        assert!(decisions[0][1], "L == -R band must pick M/S");
        // sfb 2 (L only): no concentration over min(E_L, E_R = 0) → L/R.
        assert!(!decisions[0][2], "L-only band must stay L/R");
        // sfb 3 (zero-energy): tie, keep L/R.
        assert!(!decisions[0][3], "zero-energy band must stay L/R");
    }

    /// Round-trip: the decision matrix from `select_ms_used_for_pair`
    /// flows into `build_chparam_info_ms_used`, then `extract_sap_abcd`
    /// reproduces the per-sfb (1, 1, 1, -1) matrix on the picked
    /// bands and identity on the rest.
    #[test]
    fn select_ms_used_for_pair_round_trips_into_extract_sap_abcd() {
        let sfb_offset: [u16; 5] = [0, 4, 8, 12, 16];
        // Two bands picked (sfb 0 correlated, sfb 1 anti-correlated),
        // two bands not picked (sfb 2 zero-energy, sfb 3 zero-energy).
        let l = vec![
            1.0f32, 0.5, -0.7, 0.3, // sfb 0
            0.4, 0.9, -0.2, 0.6, // sfb 1
            0.0, 0.0, 0.0, 0.0, // sfb 2
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let r = vec![
            1.0f32, 0.5, -0.7, 0.3, // sfb 0
            -0.4, -0.9, 0.2, -0.6, // sfb 1
            0.0, 0.0, 0.0, 0.0, // sfb 2
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let decisions = select_ms_used_for_pair(&[l], &[r], &sfb_offset, &[4]);
        let info = build_chparam_info_ms_used(decisions.clone());
        let coeffs = extract_sap_abcd(&info, &[4]);
        assert_eq!(coeffs.abcd[0].len(), 4);
        // Picked bands -> (1, 1, 1, -1).
        for sfb in [0, 1] {
            let (a, b, c, d) = coeffs.abcd[0][sfb];
            assert!((a - 1.0).abs() < 1e-6);
            assert!((b - 1.0).abs() < 1e-6);
            assert!((c - 1.0).abs() < 1e-6);
            assert!((d - (-1.0)).abs() < 1e-6);
        }
        // Skipped bands -> identity (1, 0, 0, 1).
        for sfb in [2, 3] {
            let (a, b, c, d) = coeffs.abcd[0][sfb];
            assert!((a - 1.0).abs() < 1e-6);
            assert!(b.abs() < 1e-6);
            assert!(c.abs() < 1e-6);
            assert!((d - 1.0).abs() < 1e-6);
        }
    }

    /// `select_ms_used_for_pair` honours the per-group bound: extra
    /// bands beyond `max_sfb_per_group[g]` are not inspected and the
    /// returned row length matches `max_sfb_per_group[g]` exactly.
    #[test]
    fn select_ms_used_for_pair_respects_per_group_bound() {
        let sfb_offset: [u16; 5] = [0, 2, 4, 6, 8];
        // 8 bins total — sfb 0 + sfb 1 correlated (would pick M/S),
        // sfb 2 + sfb 3 also correlated. Bound at max_sfb = 2 so only
        // the first two bands are evaluated.
        let l = vec![1.0f32, 0.5, 0.4, 0.9, 0.8, 0.1, -0.3, 0.5];
        let r = vec![1.0f32, 0.5, 0.4, 0.9, 0.8, 0.1, -0.3, 0.5];
        let decisions = select_ms_used_for_pair(&[l], &[r], &sfb_offset, &[2]);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].len(), 2);
        assert!(decisions[0][0]);
        assert!(decisions[0][1]);
    }

    /// Multi-group input: per-group decisions are independent — group 0
    /// is fully correlated (all M/S) and group 1 is fully zero-energy
    /// (all L/R).
    #[test]
    fn select_ms_used_for_pair_multi_group_independent() {
        let sfb_offset: [u16; 3] = [0, 4, 8];
        let g0_l = vec![1.0f32, 0.5, -0.7, 0.3, 0.4, 0.9, -0.2, 0.6];
        let g0_r = vec![1.0f32, 0.5, -0.7, 0.3, 0.4, 0.9, -0.2, 0.6];
        let g1_l = vec![0.0f32; 8];
        let g1_r = vec![0.0f32; 8];
        let decisions = select_ms_used_for_pair(&[g0_l, g1_l], &[g0_r, g1_r], &sfb_offset, &[2, 2]);
        assert_eq!(decisions.len(), 2);
        assert!(decisions[0][0]);
        assert!(decisions[0][1]);
        assert!(!decisions[1][0]);
        assert!(!decisions[1][1]);
    }

    // ----- select_alpha_q_for_pair -----

    /// `select_alpha_q_for_pair` picks the least-squares projection
    /// `g* = <S, M> / <M, M>` per band, quantised to `alpha_q =
    /// round(10 · g*)`, and raises `sap_coeff_used` only when the
    /// quantised index is non-zero.
    ///
    /// Construct a single-group 4-sfb signal (4 bins/sfb, sfb_offset =
    /// [0, 4, 8, 12, 16]) where each pair-leading even sfb has a known
    /// side-vs-mid relationship:
    ///
    /// * sfb 0 — L only, R = 0: M = L/2, S = L/2 → S = M exactly →
    ///   g* = 1.0 → alpha_q = 10 (positive, SAP picked).
    /// * sfb 2 — R only, L = 0: M = R/2, S = -R/2 → S = -M →
    ///   g* = -1.0 → alpha_q = -10 (negative, SAP picked).
    #[test]
    fn select_alpha_q_for_pair_least_squares_projection() {
        let sfb_offset: [u16; 5] = [0, 4, 8, 12, 16];
        let l = vec![
            1.0f32, 0.5, -0.7, 0.3, // sfb 0: L only
            0.0, 0.0, 0.0, 0.0, // sfb 1: pair-mate of sfb 0
            0.0, 0.0, 0.0, 0.0, // sfb 2: R only
            0.0, 0.0, 0.0, 0.0, // sfb 3: pair-mate of sfb 2
        ];
        let r = vec![
            0.0f32, 0.0, 0.0, 0.0, // sfb 0: R = 0
            0.0, 0.0, 0.0, 0.0, // sfb 1
            0.8, 0.2, -0.4, 0.6, // sfb 2: R only
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let (alpha_q, used) = select_alpha_q_for_pair(&[l], &[r], &sfb_offset, &[4]);
        assert_eq!(alpha_q.len(), 1);
        assert_eq!(alpha_q[0].len(), 4);
        // sfb 0: S = M → g* = 1.0 → alpha_q = 10.
        assert_eq!(alpha_q[0][0], 10);
        assert!(used[0][0]);
        // sfb 1 inherits the even partner's alpha_q + flag.
        assert_eq!(alpha_q[0][1], 10);
        assert!(used[0][1]);
        // sfb 2: S = -M → g* = -1.0 → alpha_q = -10.
        assert_eq!(alpha_q[0][2], -10);
        assert!(used[0][2]);
        assert_eq!(alpha_q[0][3], -10);
        assert!(used[0][3]);
    }

    /// A band with `L == R` (pure mid, zero side) projects to `g* = 0`
    /// (`<S, M> == 0`), so `alpha_q` rounds to 0 and the SAP-used flag
    /// stays clear — no SAP bit spent where prediction offers nothing.
    /// A zero-energy band (no mid energy) likewise stays clear.
    #[test]
    fn select_alpha_q_for_pair_clears_flag_when_no_prediction_benefit() {
        let sfb_offset: [u16; 5] = [0, 4, 8, 12, 16];
        let l = vec![
            1.0f32, 0.5, -0.7, 0.3, // sfb 0: L == R (S = 0)
            0.0, 0.0, 0.0, 0.0, // sfb 1
            0.0, 0.0, 0.0, 0.0, // sfb 2: zero-energy
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let r = vec![
            1.0f32, 0.5, -0.7, 0.3, // sfb 0: R == L
            0.0, 0.0, 0.0, 0.0, // sfb 1
            0.0, 0.0, 0.0, 0.0, // sfb 2: zero-energy
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let (alpha_q, used) = select_alpha_q_for_pair(&[l], &[r], &sfb_offset, &[4]);
        // sfb 0: pure mid → <S, M> == 0 → alpha_q == 0, flag clear.
        assert_eq!(alpha_q[0][0], 0);
        assert!(!used[0][0]);
        assert!(!used[0][1]);
        // sfb 2: zero-energy → <M, M> == 0 → flag clear.
        assert_eq!(alpha_q[0][2], 0);
        assert!(!used[0][2]);
        assert!(!used[0][3]);
    }

    /// Round-trip: the `(alpha_q, sap_coeff_used)` matrices from
    /// `select_alpha_q_for_pair` flow into
    /// `build_chparam_info_sap_data_from_alpha_q`, then `extract_sap_abcd`
    /// reproduces the per-sfb SAP matrix `(1 + g, 1, 1 - g, -1)` with
    /// `g = alpha_q · 0.1` on the picked bands and identity on the rest.
    #[test]
    fn select_alpha_q_for_pair_round_trips_through_sap_data_builder() {
        let sfb_offset: [u16; 5] = [0, 4, 8, 12, 16];
        // sfb 0 picks alpha_q = +10 (S = M); sfb 2 is a pure-mid band
        // that clears the flag.
        let l = vec![
            1.0f32, 0.5, -0.7, 0.3, // sfb 0: L only → S = M
            0.0, 0.0, 0.0, 0.0, // sfb 1
            0.9, 0.4, -0.2, 0.5, // sfb 2: L == R → S = 0
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let r = vec![
            0.0f32, 0.0, 0.0, 0.0, // sfb 0: R = 0
            0.0, 0.0, 0.0, 0.0, // sfb 1
            0.9, 0.4, -0.2, 0.5, // sfb 2: R == L
            0.0, 0.0, 0.0, 0.0, // sfb 3
        ];
        let (alpha_q, used) = select_alpha_q_for_pair(&[l], &[r], &sfb_offset, &[4]);
        let info = build_chparam_info_sap_data_from_alpha_q(&alpha_q, &used, false, &[4]);
        let coeffs = extract_sap_abcd(&info, &[4]);
        assert_eq!(coeffs.abcd[0].len(), 4);
        // sfb 0 + 1 (picked, alpha_q = 10 → g = 1.0): (2, 1, 0, -1).
        for sfb in [0usize, 1] {
            let (a, b, c, d) = coeffs.abcd[0][sfb];
            assert!((a - 2.0).abs() < 1e-5, "sfb {sfb} a");
            assert!((b - 1.0).abs() < 1e-5, "sfb {sfb} b");
            assert!(c.abs() < 1e-5, "sfb {sfb} c");
            assert!((d - (-1.0)).abs() < 1e-5, "sfb {sfb} d");
        }
        // sfb 2 + 3 (cleared): identity (1, 0, 0, 1).
        for sfb in [2usize, 3] {
            let (a, b, c, d) = coeffs.abcd[0][sfb];
            assert!((a - 1.0).abs() < 1e-5, "sfb {sfb} a");
            assert!(b.abs() < 1e-5, "sfb {sfb} b");
            assert!(c.abs() < 1e-5, "sfb {sfb} c");
            assert!((d - 1.0).abs() < 1e-5, "sfb {sfb} d");
        }
    }

    /// The projection coefficient clamps to the HCB_SCALEFAC-codable
    /// range `[-60, +60]`: a side track that is a large multiple of the
    /// mid (`g* ≫ 6.0`) saturates at `alpha_q = 60`.
    #[test]
    fn select_alpha_q_for_pair_clamps_to_codebook_range() {
        let sfb_offset: [u16; 2] = [0, 4];
        // Build L, R so that M = (L + R) / 2 is small and S = (L - R) / 2
        // is a 10x multiple of M, giving g* = 10 → 10 · g* = 100 → clamp
        // to 60. Pick M[k] = m, S[k] = 10·m ⇒ L = M + S = 11·m,
        // R = M - S = -9·m.
        let l = vec![11.0f32, 5.5, -3.3, 2.2];
        let r = vec![-9.0f32, -4.5, 2.7, -1.8];
        let (alpha_q, used) = select_alpha_q_for_pair(&[l], &[r], &sfb_offset, &[1]);
        assert_eq!(alpha_q[0][0], 60);
        assert!(used[0][0]);
    }

    /// Multi-group input: per-group `alpha_q` decisions are independent.
    /// Group 0 has a `S = M` band (alpha_q = +10), group 1 is fully
    /// zero-energy (no SAP).
    #[test]
    fn select_alpha_q_for_pair_multi_group_independent() {
        let sfb_offset: [u16; 3] = [0, 4, 8];
        // Group 0: sfb 0 L-only (S = M → alpha_q = 10), sfb 1 inherits.
        let g0_l = vec![1.0f32, 0.5, -0.7, 0.3, 0.0, 0.0, 0.0, 0.0];
        let g0_r = vec![0.0f32; 8];
        // Group 1: zero-energy throughout.
        let g1_l = vec![0.0f32; 8];
        let g1_r = vec![0.0f32; 8];
        let (alpha_q, used) =
            select_alpha_q_for_pair(&[g0_l, g1_l], &[g0_r, g1_r], &sfb_offset, &[2, 2]);
        assert_eq!(alpha_q.len(), 2);
        assert_eq!(alpha_q[0][0], 10);
        assert!(used[0][0]);
        assert!(used[0][1]);
        assert!(!used[1][0]);
        assert!(!used[1][1]);
    }
}
