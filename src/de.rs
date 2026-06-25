//! Dialogue Enhancement (DE) parser — §4.2.14.11..13 syntax,
//! §4.3.14 semantics. Walks `dialog_enhancement(b_iframe)` (Table 76)
//! end-to-end, including `de_config()` (Table 77, §4.3.14.3) and
//! `de_data()` (Table 78, §4.3.14.4) plus the `de_par[ch][band]`
//! decoding that runs through the Annex A.4 Huffman codebooks
//! ([`crate::de_huffman`]).
//!
//! Out of scope here:
//! * Application of the DE parameters to the QMF subbands (§5.7.10
//!   dialogue-enhancement tool).
//! * Mixing-coefficient resolution from `de_mix_coef1_idx` /
//!   `de_mix_coef2_idx` to actual real-valued coefficients (Table 172
//!   quantization vector); the parser surfaces the raw indices.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::de_huffman::{
    de_abs_huffman, de_diff_huffman, write_de_abs_huffman, write_de_diff_huffman,
};

/// Constant per §4.3.14.5.1: dialogue-enhancement parameter bands count
/// is always 8 (Table 173).
pub const DE_NR_BANDS: usize = 8;

/// Dialogue enhancement method per Table 170 (§4.3.14.3.1).
///
/// The numeric values are the raw 2-bit codepoints in the bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeMethod {
    /// Channel independent dialogue enhancement.
    ChannelIndependent = 0,
    /// Cross-channel dialogue enhancement.
    CrossChannel = 1,
    /// Waveform-parametric hybrid; channel independent enhancement and
    /// waveform channel.
    HybridChannelIndependent = 2,
    /// Waveform-parametric hybrid; cross-channel enhancement and
    /// waveform channel.
    HybridCrossChannel = 3,
}

impl DeMethod {
    fn from_u32(v: u32) -> Self {
        match v & 0x3 {
            0 => DeMethod::ChannelIndependent,
            1 => DeMethod::CrossChannel,
            2 => DeMethod::HybridChannelIndependent,
            _ => DeMethod::HybridCrossChannel,
        }
    }

    /// Raw `de_method` 2-bit value as it appears in the bitstream.
    pub fn raw(self) -> u32 {
        self as u32
    }

    /// `de_method % 2` — the table_idx selector for `de_abs_huffman`
    /// and `de_diff_huffman` per §4.3.14.5.4 / .5.5.
    pub fn huffman_table_idx(self) -> u32 {
        self.raw() & 0x1
    }
}

/// Decoded `de_config()` (§4.2.14.12 Table 77, §4.3.14.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeConfig {
    /// `de_method` (Table 170).
    pub method: DeMethod,
    /// `de_max_gain` raw 2-bit value (§4.3.14.3.2). Interpreted as
    /// `Gmax = (raw + 1) * 3 dB`.
    pub max_gain: u8,
    /// `de_channel_config` raw 3-bit value (Table 171, §4.3.14.3.3).
    pub channel_config: u8,
}

impl DeConfig {
    /// `de_nr_channels` per Table 171 (§4.3.14.3.3) — total number of
    /// channels for which dialogue-enhancement parameters are
    /// transmitted, derived from `de_channel_config`.
    pub fn nr_channels(self) -> u8 {
        match self.channel_config & 0x7 {
            0b000 => 0,                 // No parameters
            0b001 | 0b010 | 0b100 => 1, // Centre / Right / Left
            0b011 | 0b101 | 0b110 => 2, // R+C / L+C / L+R
            0b111 => 3,                 // L+R+C
            _ => unreachable!(),
        }
    }

    /// `Gmax` in dB per §4.3.14.3.2: `(de_max_gain + 1) * 3 dB`.
    pub fn max_gain_db(self) -> f32 {
        (self.max_gain as f32 + 1.0) * 3.0
    }
}

/// Decoded `de_data()` (§4.2.14.13 Table 78, §4.3.14.4).
#[derive(Debug, Clone, PartialEq)]
pub struct DeData {
    /// `de_keep_pos_flag` (§4.3.14.4.1). Defaults to false on I-frames.
    pub keep_pos_flag: bool,
    /// `de_mix_coef1_idx` raw 5-bit value (Table 172) when present.
    pub mix_coef1_idx: Option<u8>,
    /// `de_mix_coef2_idx` raw 5-bit value (Table 172) when present
    /// (only for `de_nr_channels == 3`).
    pub mix_coef2_idx: Option<u8>,
    /// `de_keep_data_flag` (§4.3.14.4.3). Defaults to false on I-frames.
    pub keep_data_flag: bool,
    /// `de_ms_proc_flag` (§4.3.14.4.4). Defaults to 0 unless
    /// `de_method ∈ {0, 2} && de_nr_channels == 2`.
    pub ms_proc_flag: bool,
    /// Decoded parameter matrix `de_par[ch][band]`. Outer length =
    /// `de_nr_channels - de_ms_proc_flag` (channels actually decoded;
    /// the M/S "side" channel inherits from elsewhere). Inner length =
    /// [`DE_NR_BANDS`]. Empty when `de_nr_channels == 0` or when
    /// `keep_data_flag` is true (inherit from previous frame).
    pub de_par: Vec<[i32; DE_NR_BANDS]>,
    /// `de_signal_contribution` raw 5-bit value (§4.3.14.4.6) when
    /// `de_method >= 2`. `αc = raw / 31` per the formula.
    pub signal_contribution: Option<u8>,
}

/// Decoded `dialog_enhancement(b_iframe)` (§4.2.14.11 Table 76).
#[derive(Debug, Clone, PartialEq)]
pub struct DialogEnhancement {
    /// `b_de_data_present` (§4.3.14.1).
    pub data_present: bool,
    /// `de_config_flag` (§4.3.14.2). On I-frames the spec forces a
    /// fresh config so this is always conceptually true; we surface it
    /// faithfully (true on I-frames, the bitstream value otherwise).
    pub config_flag: bool,
    /// Decoded `de_config()` when present (always present on I-frames,
    /// otherwise gated by `config_flag`).
    pub config: Option<DeConfig>,
    /// Decoded `de_data()` when `data_present` is true.
    pub data: Option<DeData>,
}

/// Walk `dialog_enhancement(b_iframe)` per §4.2.14.11 Table 76.
///
/// `prev_config` carries the active `DeConfig` from earlier frames so
/// non-I non-`de_config_flag` frames can be decoded against the
/// running configuration. Pass `None` for the very first call (the
/// caller must guarantee an I-frame seeds the config or the parser
/// returns an `Error::invalid` if a non-I frame tries to read
/// `de_data()` without one).
pub fn parse_dialog_enhancement(
    br: &mut BitReader<'_>,
    b_iframe: bool,
    prev_config: Option<DeConfig>,
) -> Result<DialogEnhancement> {
    let data_present = br.read_u1()? != 0;
    if !data_present {
        return Ok(DialogEnhancement {
            data_present: false,
            config_flag: false,
            config: None,
            data: None,
        });
    }

    let (config_flag, config) = if b_iframe {
        let cfg = parse_de_config(br)?;
        (true, Some(cfg))
    } else {
        let f = br.read_u1()? != 0;
        if f {
            let cfg = parse_de_config(br)?;
            (true, Some(cfg))
        } else {
            (false, prev_config)
        }
    };

    let active_cfg = config.ok_or_else(|| {
        Error::invalid("ac4: dialog_enhancement requires an active de_config (no prev_config)")
    })?;

    let data = parse_de_data(br, active_cfg, b_iframe)?;

    Ok(DialogEnhancement {
        data_present: true,
        config_flag,
        config: Some(active_cfg),
        data: Some(data),
    })
}

/// Walk `de_config()` per §4.2.14.12 Table 77.
pub fn parse_de_config(br: &mut BitReader<'_>) -> Result<DeConfig> {
    let method = DeMethod::from_u32(br.read_u32(2)?);
    let max_gain = br.read_u32(2)? as u8;
    let channel_config = br.read_u32(3)? as u8;
    Ok(DeConfig {
        method,
        max_gain,
        channel_config,
    })
}

/// Write `de_config()` per §4.2.14.12 Table 77, the inverse of
/// [`parse_de_config`].
pub fn write_de_config(bw: &mut oxideav_core::bits::BitWriter, cfg: &DeConfig) -> Result<()> {
    if cfg.max_gain > 3 {
        return Err(Error::invalid("ac4: de_max_gain exceeds 2 bits"));
    }
    if cfg.channel_config > 7 {
        return Err(Error::invalid("ac4: de_channel_config exceeds 3 bits"));
    }
    bw.write_u32(cfg.method.raw(), 2);
    bw.write_u32(cfg.max_gain as u32, 2);
    bw.write_u32(cfg.channel_config as u32, 3);
    Ok(())
}

/// Write `dialog_enhancement(b_iframe)` per §4.2.14.11 Table 76, the
/// inverse of [`parse_dialog_enhancement`].
///
/// On non-I frames the `de_config_flag` is taken from `de.config_flag`;
/// the embedded `de_config()` is only written when that flag is set. The
/// `de_data()` payload is written against the active config.
pub fn write_dialog_enhancement(
    bw: &mut oxideav_core::bits::BitWriter,
    de: &DialogEnhancement,
    b_iframe: bool,
) -> Result<()> {
    bw.write_bit(de.data_present);
    if !de.data_present {
        return Ok(());
    }

    let active_cfg = if b_iframe {
        let cfg = de
            .config
            .ok_or_else(|| Error::invalid("ac4: I-frame dialog_enhancement requires config"))?;
        write_de_config(bw, &cfg)?;
        cfg
    } else {
        bw.write_bit(de.config_flag);
        let cfg = de
            .config
            .ok_or_else(|| Error::invalid("ac4: dialog_enhancement requires active config"))?;
        if de.config_flag {
            write_de_config(bw, &cfg)?;
        }
        cfg
    };

    let data = de
        .data
        .as_ref()
        .ok_or_else(|| Error::invalid("ac4: dialog_enhancement data_present but data is None"))?;
    write_de_data(bw, active_cfg, b_iframe, data)
}

/// Write `de_data(de_method, de_nr_channels, b_iframe)` per §4.2.14.13
/// Table 78, the inverse of [`parse_de_data`].
///
/// The de_par parameter matrix is re-Huffman-encoded: on I-frames the
/// first channel emits a leading `de_abs_huffman` band-0 value followed
/// by `de_diff_huffman` deltas; cross-channel/non-I delta conventions
/// mirror the decoder (the surfaced `de_par` already holds the values the
/// decoder reconstructed, so we re-derive the transmitted deltas here).
pub fn write_de_data(
    bw: &mut oxideav_core::bits::BitWriter,
    cfg: DeConfig,
    b_iframe: bool,
    data: &DeData,
) -> Result<()> {
    let nr_channels = cfg.nr_channels() as usize;
    if nr_channels == 0 {
        return Ok(());
    }

    let raw_method = cfg.method.raw();
    let cross_channel = (raw_method == 1 || raw_method == 3) && nr_channels > 1;
    if cross_channel {
        if !b_iframe {
            bw.write_bit(data.keep_pos_flag);
        }
        if !data.keep_pos_flag {
            bw.write_u32(
                data.mix_coef1_idx
                    .ok_or_else(|| Error::invalid("ac4: mix_coef1_idx required"))?
                    as u32,
                5,
            );
            if nr_channels == 3 {
                bw.write_u32(
                    data.mix_coef2_idx
                        .ok_or_else(|| Error::invalid("ac4: mix_coef2_idx required (3 ch)"))?
                        as u32,
                    5,
                );
            }
        }
    }

    if !b_iframe {
        bw.write_bit(data.keep_data_flag);
    }

    if !data.keep_data_flag {
        let ms_capable = (raw_method == 0 || raw_method == 2) && nr_channels == 2;
        if ms_capable {
            bw.write_bit(data.ms_proc_flag);
        }

        let table_idx = cfg.method.huffman_table_idx();
        let n_decode_channels = nr_channels - data.ms_proc_flag as usize;
        if data.de_par.len() != n_decode_channels {
            return Err(Error::invalid(
                "ac4: de_par channel count mismatches de_nr_channels - ms_proc_flag",
            ));
        }

        let mut ref_val: i32 = 0;
        for (ch, bands) in data.de_par.iter().enumerate() {
            if b_iframe && ch == 0 {
                write_de_abs_huffman(bw, table_idx, bands[0])?;
                ref_val = bands[0];
                for &v in bands.iter().skip(1) {
                    write_de_diff_huffman(bw, table_idx, v - ref_val)?;
                    ref_val = v;
                }
            } else if b_iframe {
                // ch > 0 on I-frame: deltas vs running ref_val.
                for &v in bands.iter() {
                    write_de_diff_huffman(bw, table_idx, v - ref_val)?;
                    ref_val = v;
                }
            } else {
                // Non-I: de_par holds the raw transmitted deltas verbatim.
                for &v in bands.iter() {
                    write_de_diff_huffman(bw, table_idx, v)?;
                }
            }
            if b_iframe {
                ref_val = bands[0];
            }
        }

        if raw_method >= 2 {
            bw.write_u32(
                data.signal_contribution.ok_or_else(|| {
                    Error::invalid("ac4: signal_contribution required (method>=2)")
                })? as u32,
                5,
            );
        }
    }
    Ok(())
}

/// Walk `de_data(de_method, de_nr_channels, b_iframe)` per §4.2.14.13
/// Table 78, including `de_par_code` Huffman decoding via Annex A.4.
pub fn parse_de_data(br: &mut BitReader<'_>, cfg: DeConfig, b_iframe: bool) -> Result<DeData> {
    let nr_channels = cfg.nr_channels() as usize;
    let mut keep_pos_flag = false;
    let mut mix_coef1_idx = None;
    let mut mix_coef2_idx = None;
    let mut keep_data_flag = false;
    let mut ms_proc_flag = false;
    let mut de_par: Vec<[i32; DE_NR_BANDS]> = Vec::new();
    let mut signal_contribution = None;

    if nr_channels > 0 {
        // Cross-channel (de_method 1 or 3): pan info plus optional mixing
        // coefficients. Only present when nr_channels > 1.
        let raw_method = cfg.method.raw();
        let cross_channel = (raw_method == 1 || raw_method == 3) && nr_channels > 1;
        if cross_channel {
            keep_pos_flag = if b_iframe { false } else { br.read_u1()? != 0 };
            if !keep_pos_flag {
                mix_coef1_idx = Some(br.read_u32(5)? as u8);
                if nr_channels == 3 {
                    mix_coef2_idx = Some(br.read_u32(5)? as u8);
                }
            }
        }

        // de_keep_data_flag — forced to 0 on I-frames.
        keep_data_flag = if b_iframe { false } else { br.read_u1()? != 0 };

        if !keep_data_flag {
            // M/S processing flag — only meaningful for method 0/2 with
            // nr_channels == 2.
            ms_proc_flag = if (raw_method == 0 || raw_method == 2) && nr_channels == 2 {
                br.read_u1()? != 0
            } else {
                false
            };

            let table_idx = cfg.method.huffman_table_idx();
            let n_decode_channels = nr_channels - ms_proc_flag as usize;
            let mut ref_val: i32 = 0;
            for ch in 0..n_decode_channels {
                let mut bands = [0i32; DE_NR_BANDS];
                if b_iframe && ch == 0 {
                    // Leading absolute value, then deltas across bands.
                    bands[0] = de_abs_huffman(br, table_idx)?;
                    ref_val = bands[0];
                    for slot in bands.iter_mut().skip(1) {
                        let delta = de_diff_huffman(br, table_idx)?;
                        *slot = ref_val + delta;
                        ref_val = *slot;
                    }
                } else {
                    // For ch > 0 on an I-frame: deltas against the previous
                    // channel's last band ref_val. For non-I-frames: deltas
                    // against the previous frame's per-band parameters.
                    for slot in bands.iter_mut() {
                        let delta = de_diff_huffman(br, table_idx)?;
                        if b_iframe {
                            *slot = ref_val + delta;
                            ref_val = *slot;
                        } else {
                            // Spec: de_par[ch][band] = de_par_prev[ch][band] + delta.
                            // We don't carry de_par_prev across calls (the
                            // caller owns that), so we surface the raw delta
                            // and let the caller reconstruct: the relevant
                            // *delta* is what the bitstream encodes here.
                            // To keep the public type stable, we use the raw
                            // delta as-is and document the convention.
                            *slot = delta;
                        }
                    }
                }
                de_par.push(bands);
                // Per spec: ref_val = de_par[ch][0] for the next channel
                // when iterating across channels in I-frames.
                if b_iframe {
                    if let Some(last) = de_par.last() {
                        ref_val = last[0];
                    }
                }
            }

            if raw_method >= 2 {
                signal_contribution = Some(br.read_u32(5)? as u8);
            }
        }
    }

    Ok(DeData {
        keep_pos_flag,
        mix_coef1_idx,
        mix_coef2_idx,
        keep_data_flag,
        ms_proc_flag,
        de_par,
        signal_contribution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::de_huffman::{
        DE_HCB_ABS_0_CW, DE_HCB_ABS_0_LEN, DE_HCB_ABS_1_CW, DE_HCB_ABS_1_LEN, DE_HCB_DIFF_0_CW,
        DE_HCB_DIFF_0_LEN, DE_HCB_DIFF_1_CW, DE_HCB_DIFF_1_LEN,
    };
    use oxideav_core::bits::BitWriter;

    // ------------------------------------------------------------------------
    // Table 171 channel-config mapping.
    // ------------------------------------------------------------------------

    #[test]
    fn nr_channels_table_171_full_mapping() {
        let cases: &[(u8, u8)] = &[
            (0b000, 0), // No parameters
            (0b001, 1), // Centre
            (0b010, 1), // Right
            (0b011, 2), // R+C
            (0b100, 1), // Left
            (0b101, 2), // L+C
            (0b110, 2), // L+R
            (0b111, 3), // L+R+C
        ];
        for &(cfg_val, expected) in cases {
            let cfg = DeConfig {
                method: DeMethod::ChannelIndependent,
                max_gain: 0,
                channel_config: cfg_val,
            };
            assert_eq!(cfg.nr_channels(), expected, "channel_config={cfg_val:03b}");
        }
    }

    #[test]
    fn de_max_gain_db_formula() {
        for raw in 0u8..=3 {
            let cfg = DeConfig {
                method: DeMethod::ChannelIndependent,
                max_gain: raw,
                channel_config: 0,
            };
            assert_eq!(cfg.max_gain_db(), (raw as f32 + 1.0) * 3.0);
        }
    }

    #[test]
    fn de_method_huffman_table_idx_parity() {
        assert_eq!(DeMethod::ChannelIndependent.huffman_table_idx(), 0);
        assert_eq!(DeMethod::CrossChannel.huffman_table_idx(), 1);
        assert_eq!(DeMethod::HybridChannelIndependent.huffman_table_idx(), 0);
        assert_eq!(DeMethod::HybridCrossChannel.huffman_table_idx(), 1);
    }

    // ------------------------------------------------------------------------
    // Hardening: malformed-input edge cases. Walker must propagate Err
    // and never panic / over-read.
    // ------------------------------------------------------------------------

    #[test]
    fn dialog_enhancement_truncated_after_data_present_flag_errors() {
        // Only a `b_de_data_present = 1` bit then no more *bytes* — the
        // walker should bail out parsing `de_config()` with a clean
        // bit-reader error. We craft a 1-byte buffer with the high bit
        // set and nothing else, then feed only that single byte slice;
        // any read past 8 bits triggers EOF.
        let bytes: [u8; 1] = [0b1000_0000];
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, true, None);
        // de_method (2) + de_max_gain (2) + de_channel_config (3) = 7
        // remaining bits all zero: nr_channels = 0 → no Huffman body.
        // That's actually a *legal* (if degenerate) frame and parses
        // clean. So instead truncate harder: feed an empty slice.
        let _ = r;
        let empty: [u8; 0] = [];
        let mut br2 = BitReader::new(&empty);
        assert!(parse_dialog_enhancement(&mut br2, true, None).is_err());
    }

    #[test]
    fn dialog_enhancement_non_iframe_no_config_no_prev_errors_cleanly() {
        // Non-I-frame, b_de_data_present = 1, de_config_flag = 0,
        // no prev_config supplied → walker must error out (not panic).
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // b_de_data_present
        bw.write_u32(0, 1); // de_config_flag = 0
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, false, None);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("active de_config"));
    }

    #[test]
    fn de_data_with_zero_channels_is_a_clean_no_op() {
        // de_channel_config = 0 -> nr_channels = 0, no Huffman bodies.
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // b_de_data_present
        bw.write_u32(0, 2); // de_method = 0
        bw.write_u32(0, 2); // de_max_gain
        bw.write_u32(0, 3); // de_channel_config = 0 -> nr_channels = 0
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, true, None).unwrap();
        let data = r.data.unwrap();
        assert!(data.de_par.is_empty());
        assert!(!data.keep_pos_flag);
        assert!(!data.keep_data_flag);
        assert!(!data.ms_proc_flag);
        assert!(data.signal_contribution.is_none());
    }

    #[test]
    fn dialog_enhancement_truncated_inside_de_par_huffman_errors() {
        // Set up a valid frame header (b_de_data_present, de_config) but
        // truncate before the Huffman bodies — the walker should bail
        // with an error rather than panic.
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // b_de_data_present
        bw.write_u32(0, 2); // de_method = 0
        bw.write_u32(0, 2); // de_max_gain
        bw.write_u32(1, 3); // de_channel_config = 1 -> Centre, nr_channels=1
                            // No Huffman data follows. parse_de_data should hit EOF.
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, true, None);
        assert!(r.is_err(), "expected truncation error");
    }

    // ------------------------------------------------------------------------
    // de_config / dialog_enhancement gating.
    // ------------------------------------------------------------------------

    #[test]
    fn dialog_enhancement_absent_short_circuits() {
        // b_de_data_present = 0 → no further bits consumed.
        let mut bw = BitWriter::new();
        bw.write_u32(0, 1); // b_de_data_present = 0
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, true, None).unwrap();
        assert!(!r.data_present);
        assert!(r.config.is_none());
        assert!(r.data.is_none());
    }

    #[test]
    fn parse_de_config_round_trip() {
        // method=2 (HybridChannelIndependent), max_gain=1 → 6 dB,
        // channel_config=0b011 → R+C → 2 channels.
        let mut bw = BitWriter::new();
        bw.write_u32(2, 2); // de_method
        bw.write_u32(1, 2); // de_max_gain
        bw.write_u32(0b011, 3); // de_channel_config
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = parse_de_config(&mut br).unwrap();
        assert_eq!(cfg.method, DeMethod::HybridChannelIndependent);
        assert_eq!(cfg.max_gain, 1);
        assert_eq!(cfg.channel_config, 0b011);
        assert_eq!(cfg.nr_channels(), 2);
        assert_eq!(cfg.max_gain_db(), 6.0);
    }

    // ------------------------------------------------------------------------
    // de_data Huffman path — exercises both ABS and DIFF tables.
    // ------------------------------------------------------------------------

    #[test]
    fn iframe_method_0_single_channel_zero_params() {
        // Method 0 → table_idx 0. Channel-independent mode with one
        // channel (Centre, channel_config=0b001). I-frame: no
        // keep_pos_flag / keep_data_flag bits, no ms_proc bit (only
        // method 0 with nr_channels==2 emits it).
        // Encode: ABS(symbol=0 → cb_off=0 → value=0), then 7 DIFFs of 0.
        // ABS_0[0] = (len=3, cw=0x6); DIFF_0[31] = (len=1, cw=0x1).
        let mut bw = BitWriter::new();
        bw.write_u32(DE_HCB_ABS_0_CW[0], DE_HCB_ABS_0_LEN[0] as u32);
        for _ in 1..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_0_CW[31], DE_HCB_DIFF_0_LEN[31] as u32);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);

        let cfg = DeConfig {
            method: DeMethod::ChannelIndependent,
            max_gain: 0,
            channel_config: 0b001, // Centre
        };
        let data = parse_de_data(&mut br, cfg, true).unwrap();
        assert!(!data.keep_pos_flag);
        assert!(!data.keep_data_flag);
        assert!(!data.ms_proc_flag);
        assert!(data.signal_contribution.is_none());
        assert_eq!(data.de_par.len(), 1);
        assert_eq!(data.de_par[0], [0i32; DE_NR_BANDS]);
    }

    #[test]
    fn iframe_method_0_centre_with_ramp_params() {
        // I-frame, single channel (Centre), method 0 → table_idx 0.
        // ABS=index 1 (len=3, cw=0x3, value 1), then 7 DIFFs of +1.
        let mut bw = BitWriter::new();
        bw.write_u32(DE_HCB_ABS_0_CW[1], DE_HCB_ABS_0_LEN[1] as u32);
        // DIFF_0 +1 is symbol_index 32 (cb_off=31).
        for _ in 1..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_0_CW[32], DE_HCB_DIFF_0_LEN[32] as u32);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let cfg = DeConfig {
            method: DeMethod::ChannelIndependent,
            max_gain: 0,
            channel_config: 0b001,
        };
        let data = parse_de_data(&mut br, cfg, true).unwrap();
        assert_eq!(data.de_par.len(), 1);
        let expected: [i32; DE_NR_BANDS] = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(data.de_par[0], expected);
    }

    #[test]
    fn iframe_method_1_stereo_lr_with_pan_and_two_channels() {
        // Method 1 (CrossChannel), channel_config=0b110 → L+R (2 channels).
        // Cross-channel I-frame: keep_pos_flag forced to 0, then 5-bit
        // mix_coef1_idx, no mix_coef2 (only 2 channels). No keep_data
        // bit on I-frame, no ms_proc (method 1 doesn't gate it).
        // Then for ch=0: ABS_1 + 7 DIFF_1; for ch=1: 8 DIFF_1.
        // table_idx = 1 % 2 = 1. ABS_1 cb_off=30 so symbol index 30 →
        // value 0; DIFF_1 cb_off=60 so symbol index 60 → value 0.
        let mix_coef1: u8 = 16; // arbitrary 5-bit value
        let mut bw = BitWriter::new();
        // de_mix_coef1_idx
        bw.write_u32(mix_coef1 as u32, 5);
        // ch=0: ABS_1 value 0
        bw.write_u32(DE_HCB_ABS_1_CW[30], DE_HCB_ABS_1_LEN[30] as u32);
        // ch=0: 7 DIFF_1 value 0
        for _ in 1..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_1_CW[60], DE_HCB_DIFF_1_LEN[60] as u32);
        }
        // ch=1: 8 DIFF_1 value 0
        for _ in 0..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_1_CW[60], DE_HCB_DIFF_1_LEN[60] as u32);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);

        let cfg = DeConfig {
            method: DeMethod::CrossChannel,
            max_gain: 0,
            channel_config: 0b110, // L+R
        };
        let data = parse_de_data(&mut br, cfg, true).unwrap();
        assert!(!data.keep_pos_flag);
        assert_eq!(data.mix_coef1_idx, Some(mix_coef1));
        assert_eq!(data.mix_coef2_idx, None);
        assert!(!data.keep_data_flag);
        assert!(!data.ms_proc_flag);
        assert!(data.signal_contribution.is_none());
        assert_eq!(data.de_par.len(), 2);
        assert_eq!(data.de_par[0], [0i32; DE_NR_BANDS]);
        assert_eq!(data.de_par[1], [0i32; DE_NR_BANDS]);
    }

    #[test]
    fn iframe_method_3_three_channels_emits_signal_contribution() {
        // Method 3 (HybridCrossChannel), channel_config=0b111 → L+R+C
        // (3 channels). Cross-channel I-frame: no keep_pos bit (forced
        // 0), 5-bit mix_coef1_idx, 5-bit mix_coef2_idx. No keep_data
        // bit on I-frame. method 3 doesn't gate ms_proc. table_idx =
        // 3 % 2 = 1. ch=0: ABS_1 + 7 DIFF_1; ch=1, ch=2: 8 DIFF_1
        // each. Then 5-bit signal_contribution at the end (method >= 2).
        let mut bw = BitWriter::new();
        bw.write_u32(7, 5); // mix_coef1_idx = 7
        bw.write_u32(11, 5); // mix_coef2_idx = 11
                             // ch=0: ABS_1 value 0 (sym 30)
        bw.write_u32(DE_HCB_ABS_1_CW[30], DE_HCB_ABS_1_LEN[30] as u32);
        for _ in 1..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_1_CW[60], DE_HCB_DIFF_1_LEN[60] as u32);
        }
        // ch=1: 8 DIFF_1 value 0
        for _ in 0..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_1_CW[60], DE_HCB_DIFF_1_LEN[60] as u32);
        }
        // ch=2: 8 DIFF_1 value 0
        for _ in 0..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_1_CW[60], DE_HCB_DIFF_1_LEN[60] as u32);
        }
        bw.write_u32(13, 5); // signal_contribution = 13
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);

        let cfg = DeConfig {
            method: DeMethod::HybridCrossChannel,
            max_gain: 2,
            channel_config: 0b111,
        };
        let data = parse_de_data(&mut br, cfg, true).unwrap();
        assert_eq!(data.mix_coef1_idx, Some(7));
        assert_eq!(data.mix_coef2_idx, Some(11));
        assert_eq!(data.signal_contribution, Some(13));
        assert_eq!(data.de_par.len(), 3);
    }

    #[test]
    fn iframe_method_2_stereo_emits_ms_proc_flag() {
        // Method 2 (HybridChannelIndependent), nr_channels=2 → ms_proc_flag
        // is read from the bitstream. Setting it to 1 means we only
        // decode `nr_channels - 1 = 1` channel of de_par. table_idx = 0.
        // I-frame: no keep_pos, no keep_data. Stream: ms_proc=1, then
        // ch=0 ABS+7 DIFF, then 5-bit signal_contribution.
        let mut bw = BitWriter::new();
        // method 2, nr_channels==2 → ms_proc bit.
        bw.write_u32(1, 1); // ms_proc = 1
        bw.write_u32(DE_HCB_ABS_0_CW[5], DE_HCB_ABS_0_LEN[5] as u32);
        for _ in 1..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_0_CW[31], DE_HCB_DIFF_0_LEN[31] as u32);
        }
        bw.write_u32(20, 5); // signal_contribution
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);

        let cfg = DeConfig {
            method: DeMethod::HybridChannelIndependent,
            max_gain: 0,
            channel_config: 0b110, // L+R = 2 channels
        };
        let data = parse_de_data(&mut br, cfg, true).unwrap();
        assert!(data.ms_proc_flag);
        assert_eq!(data.de_par.len(), 1, "ms_proc strips one channel");
        assert_eq!(data.de_par[0][0], 5);
        for b in 1..DE_NR_BANDS {
            assert_eq!(data.de_par[0][b], 5, "ramp via 0-deltas");
        }
        assert_eq!(data.signal_contribution, Some(20));
    }

    // ------------------------------------------------------------------------
    // dialog_enhancement() top-level wrapper.
    // ------------------------------------------------------------------------

    #[test]
    fn iframe_dialog_enhancement_full_round_trip() {
        // Top-level: b_de_data_present=1, then de_config, then de_data.
        // method=0 (ChannelIndependent), 1 channel (Centre).
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // b_de_data_present
                            // de_config inline
        bw.write_u32(0, 2); // method 0
        bw.write_u32(0, 2); // max_gain 0
        bw.write_u32(0b001, 3); // channel_config = Centre = 1 ch
                                // de_data: ABS_0 sym 0 then 7 DIFF_0 zeros
        bw.write_u32(DE_HCB_ABS_0_CW[0], DE_HCB_ABS_0_LEN[0] as u32);
        for _ in 1..DE_NR_BANDS {
            bw.write_u32(DE_HCB_DIFF_0_CW[31], DE_HCB_DIFF_0_LEN[31] as u32);
        }
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, true, None).unwrap();
        assert!(r.data_present);
        assert!(r.config_flag);
        let cfg = r.config.expect("config");
        assert_eq!(cfg.method, DeMethod::ChannelIndependent);
        let data = r.data.expect("data");
        assert_eq!(data.de_par.len(), 1);
        assert_eq!(data.de_par[0], [0i32; DE_NR_BANDS]);
    }

    #[test]
    fn non_iframe_de_config_flag_zero_uses_prev_config() {
        // Non-I-frame: b_de_data_present=1, de_config_flag=0 → use
        // prev_config. de_keep_pos_flag default zero (only emitted for
        // cross-channel ≥2ch). de_keep_data_flag=1 → no de_par decode.
        let prev = DeConfig {
            method: DeMethod::ChannelIndependent,
            max_gain: 1,
            channel_config: 0b001, // Centre, 1 ch
        };
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // b_de_data_present
        bw.write_u32(0, 1); // de_config_flag=0
                            // method 0 + nr_channels 1 → no cross-channel pan bits.
        bw.write_u32(1, 1); // de_keep_data_flag=1 → no de_par bits
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let r = parse_dialog_enhancement(&mut br, false, Some(prev)).unwrap();
        assert!(r.data_present);
        assert!(!r.config_flag);
        assert_eq!(r.config, Some(prev));
        let data = r.data.expect("data");
        assert!(data.keep_data_flag);
        assert!(data.de_par.is_empty());
    }

    #[test]
    fn non_iframe_without_prev_config_errors() {
        // Non-I-frame with config_flag=0 and no prev_config → invalid.
        let mut bw = BitWriter::new();
        bw.write_u32(1, 1); // b_de_data_present
        bw.write_u32(0, 1); // de_config_flag=0 → expect prev_config
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let err = parse_dialog_enhancement(&mut br, false, None).unwrap_err();
        assert!(format!("{err}").contains("dialog_enhancement"));
    }

    // ------------------------------------------------------------------
    // Write-side round-trips — de_config / de_data / dialog_enhancement
    // ------------------------------------------------------------------

    #[test]
    fn write_de_config_round_trips_all_fields() {
        for method in [
            DeMethod::ChannelIndependent,
            DeMethod::CrossChannel,
            DeMethod::HybridChannelIndependent,
            DeMethod::HybridCrossChannel,
        ] {
            for max_gain in 0u8..=3 {
                for channel_config in 0u8..=7 {
                    let cfg = DeConfig {
                        method,
                        max_gain,
                        channel_config,
                    };
                    let mut bw = BitWriter::new();
                    write_de_config(&mut bw, &cfg).unwrap();
                    bw.align_to_byte();
                    let bytes = bw.finish();
                    let mut br = BitReader::new(&bytes);
                    let got = parse_de_config(&mut br).unwrap();
                    assert_eq!(
                        got, cfg,
                        "method={method:?} mg={max_gain} cc={channel_config}"
                    );
                }
            }
        }
    }

    /// Helper: encode then decode a `DeData` for a config, asserting
    /// parse(write(parse(x))) is stable.
    fn de_data_parse_write_parse_stable(cfg: DeConfig, b_iframe: bool, raw: &[u8]) {
        let mut br = BitReader::new(raw);
        let first = parse_de_data(&mut br, cfg, b_iframe).unwrap();
        let mut bw = BitWriter::new();
        write_de_data(&mut bw, cfg, b_iframe, &first).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br2 = BitReader::new(&bytes);
        let second = parse_de_data(&mut br2, cfg, b_iframe).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn write_de_data_iframe_single_channel_round_trips() {
        // channel_config 001 (centre, 1 channel), method 0 (no cross-ch,
        // no ms for 1 channel). de_keep_data_flag forced 0 on I-frame.
        // Build de_par via abs(band0) + 7 diffs using the smallest
        // codewords (index == cb_off → value 0 for abs/diff table 0).
        let cfg = DeConfig {
            method: DeMethod::ChannelIndependent,
            max_gain: 1,
            channel_config: 0b001,
        };
        // Emit a leading abs + 7 diff codewords (all value 0 → shortest
        // codes). value 0 maps to symbol cb_off in each table.
        let mut bw = BitWriter::new();
        write_de_abs_huffman(&mut bw, 0, 0).unwrap();
        for _ in 0..7 {
            write_de_diff_huffman(&mut bw, 0, 0).unwrap();
        }
        bw.align_to_byte();
        let raw = bw.finish();
        de_data_parse_write_parse_stable(cfg, true, &raw);
    }

    #[test]
    fn write_de_data_iframe_cross_channel_pair_round_trips() {
        // channel_config 110 (L+R, 2 channels), method 1 (cross-channel).
        // I-frame: keep_pos_flag forced 0 → mix_coef1_idx present.
        let cfg = DeConfig {
            method: DeMethod::CrossChannel,
            max_gain: 2,
            channel_config: 0b110,
        };
        let mut bw = BitWriter::new();
        bw.write_u32(9, 5); // mix_coef1_idx (no mix_coef2 for 2 ch)
                            // ch0: abs + 7 diffs; ch1: 8 diffs (table_idx for method 1 == 1).
        write_de_abs_huffman(&mut bw, 1, 0).unwrap();
        for _ in 0..7 {
            write_de_diff_huffman(&mut bw, 1, 0).unwrap();
        }
        for _ in 0..8 {
            write_de_diff_huffman(&mut bw, 1, 0).unwrap();
        }
        bw.align_to_byte();
        let raw = bw.finish();
        de_data_parse_write_parse_stable(cfg, true, &raw);
    }

    #[test]
    fn write_dialog_enhancement_iframe_round_trips() {
        // Full dialog_enhancement on an I-frame, channel_config 001.
        let cfg = DeConfig {
            method: DeMethod::ChannelIndependent,
            max_gain: 0,
            channel_config: 0b001,
        };
        let mut payload = BitWriter::new();
        payload.write_u32(1, 1); // b_de_data_present
        write_de_config(&mut payload, &cfg).unwrap();
        write_de_abs_huffman(&mut payload, 0, 0).unwrap();
        for _ in 0..7 {
            write_de_diff_huffman(&mut payload, 0, 0).unwrap();
        }
        payload.align_to_byte();
        let raw = payload.finish();

        let mut br = BitReader::new(&raw);
        let first = parse_dialog_enhancement(&mut br, true, None).unwrap();
        let mut bw = BitWriter::new();
        write_dialog_enhancement(&mut bw, &first, true).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br2 = BitReader::new(&bytes);
        let second = parse_dialog_enhancement(&mut br2, true, None).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn write_dialog_enhancement_not_present_round_trips() {
        let de = DialogEnhancement {
            data_present: false,
            config_flag: false,
            config: None,
            data: None,
        };
        let mut bw = BitWriter::new();
        write_dialog_enhancement(&mut bw, &de, true).unwrap();
        bw.align_to_byte();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = parse_dialog_enhancement(&mut br, true, None).unwrap();
        assert_eq!(got, de);
    }
}
