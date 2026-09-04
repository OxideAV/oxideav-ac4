//! Framework encoder — the [`oxideav_core::Encoder`] face of the AC-4
//! encoder, registered through [`crate::register_codecs`] and reachable
//! directly via [`make_encoder`].
//!
//! [`crate::encoder_ims::Ac4ImsEncoder`] exposes one `encode_frame_pcm_*`
//! entry point per channel layout and coding tool; this module turns
//! that surface into a stream encoder:
//!
//! * **Layout dispatch** — `CodecParameters::channels` selects the
//!   layout (mono / stereo / 5.0 / 5.1 / 7.0 / 7.1 / 7.0.4 / 7.1.4 /
//!   9.0.4 / 9.1.4 / 22.2) and the [`Ac4EncoderOptions::mode`] selects
//!   the coding tool family on that layout (see [`EncodeMode`]).
//! * **Framing** — input samples of any [`SampleFormat`] are converted
//!   to float, buffered per channel, and cut into `frame_len`-sample
//!   AC-4 frames (`frame_len` follows the `(fs_index, frame_rate_index)`
//!   pair of TS 103 190-1 Table 83 / 84); `flush` zero-pads the tail.
//! * **Packetisation** — every frame becomes one [`Packet`]: either a
//!   bare `raw_ac4_frame()` (the ISO BMFF sample form) or an Annex G
//!   `ac4_syncframe()` (`0xAC40` plain / `0xAC41` CRC-protected), with
//!   `pts` / `duration` in a `1 / sample_rate` time base and the
//!   keyframe flag mirroring `b_iframe_global`.
//!
//! # Input channel order
//!
//! Layouts that core names carry their [`ChannelLayout`] positions:
//! `Mono`, `Stereo`, `Surround50` (`L R C Ls Rs`), `Surround51`
//! (`L R C LFE Ls Rs`), `Surround70` (`L R C Ls Rs Lb Rb`) and
//! `Surround71` (`L R C LFE Ls Rs Lb Rb`). The immersive counts are
//! `DiscreteN` layouts in the decoder's output slot order — LFE(s)
//! **first** when present, then `L R C [Lscr Rscr] Ls Rs Lb Rb Tfl Tfr
//! Tbl Tbr` (7.X.4 / 9.X.4), or the TS 103 190-2 Table 21 order for
//! 22.2 — so a decode → encode round trip through the framework is an
//! identity mapping on those layouts.

use oxideav_core::options::{
    parse_options, CodecOptionsStruct, OptionField, OptionKind, OptionValue,
};
use oxideav_core::{
    AudioFrame, ChannelLayout, CodecId, CodecParameters, Encoder, Error, Frame, Packet, Result,
    SampleFormat, TimeBase,
};

use crate::encoder_ims::Ac4ImsEncoder;
use crate::sync::wrap_sync_frame;

/// Coding-tool family selected on the chosen layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodeMode {
    /// Waveform coding on every channel: SIMPLE / ASF (mono, stereo
    /// with the automatic joint-M/S decision, 5.X, 7.X, 22.2 Simple)
    /// and the SCPL codec mode with automatic SAP decisions on the
    /// immersive channel element. Every frame is an I-frame (the
    /// waveform bodies carry no I-frame-gated configuration).
    #[default]
    Waveform,
    /// A-SPX bandwidth extension with real per-channel envelope
    /// synthesis: the immersive ASPX_SCPL codec mode (7.X.4 / 9.X.4)
    /// and the 22.2 A-SPX codec mode. Mono / stereo fall back to the
    /// waveform tools (no A-SPX synthesis path exists for them), and
    /// the 5.X / 7.X A-CPL routes are rejected at construction until
    /// their PCM parity is pinned. Honours [`Ac4EncoderOptions::gop`]
    /// (P-frames re-use the I-frame-sticky `aspx_config`).
    Parametric,
}

/// Sync-frame wrapping of the emitted packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Framing {
    /// Annex G `ac4_syncframe()` with the `0xAC40` sync word.
    #[default]
    Sync,
    /// Annex G `ac4_syncframe()` with the `0xAC41` sync word and the
    /// G.4.2 `crc_word` trailer.
    SyncCrc,
    /// Bare `raw_ac4_frame()` payloads (ISO BMFF `ac-4` sample form).
    Raw,
}

/// Typed options for [`make_encoder`] (schema on
/// [`CodecOptionsStruct::SCHEMA`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Ac4EncoderOptions {
    /// `frame_rate_index` (TS 103 190-1 Table 83 for 48 kHz, Table 84
    /// for 44,1 kHz). Sets the AC-4 frame length: at 48 kHz index 1
    /// (24 fps) is 1920 samples, index 2 (25 fps) 2048, index 4
    /// (30 fps) 1536, index 7 (50 fps) 1024, index 9 (60 fps) 768, …
    pub frame_rate_index: u32,
    /// Packet framing (`sync` / `sync_crc` / `raw`).
    pub framing: Framing,
    /// Coding-tool family (`waveform` / `parametric`).
    pub mode: EncodeMode,
    /// Coded audio bandwidth in Hz — the encoder picks the widest
    /// `max_sfb` whose scale-factor-band edge stays at or below this.
    pub bandwidth_hz: u32,
    /// I-frame interval: frame `k` is an I-frame when `k % gop == 0`.
    /// `1` (the default) makes every frame independently decodable.
    /// Only the parametric immersive / 22.2 routes emit P-frames; the
    /// waveform routes carry no I-frame-gated configuration and stay
    /// all-I regardless.
    pub gop: u32,
}

impl Default for Ac4EncoderOptions {
    fn default() -> Self {
        Self {
            frame_rate_index: 1,
            framing: Framing::Sync,
            mode: EncodeMode::Waveform,
            bandwidth_hz: 20_000,
            gop: 1,
        }
    }
}

impl CodecOptionsStruct for Ac4EncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "frame_rate_index",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "TS 103 190-1 Table 83/84 frame_rate_index (1 = 24 fps, 1920 samples at 48 kHz)",
        },
        OptionField {
            name: "framing",
            kind: OptionKind::Enum(&["sync", "sync_crc", "raw"]),
            default: OptionValue::String(String::new()),
            help: "packet framing: Annex G sync frame (0xAC40), CRC-protected sync frame (0xAC41), or bare raw_ac4_frame",
        },
        OptionField {
            name: "mode",
            kind: OptionKind::Enum(&["waveform", "parametric"]),
            default: OptionValue::String(String::new()),
            help: "coding tools: waveform (SIMPLE/ASF + SCPL) or parametric (A-SPX + A-CPL / ASPX_SCPL)",
        },
        OptionField {
            name: "bandwidth",
            kind: OptionKind::U32,
            default: OptionValue::U32(20_000),
            help: "coded audio bandwidth in Hz (selects max_sfb)",
        },
        OptionField {
            name: "gop",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "I-frame interval on the parametric routes (1 = all I-frames)",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> Result<()> {
        match key {
            "frame_rate_index" => self.frame_rate_index = value.as_u32()?,
            "framing" => {
                self.framing = match value.as_str()? {
                    "sync" => Framing::Sync,
                    "sync_crc" => Framing::SyncCrc,
                    "raw" => Framing::Raw,
                    other => return Err(Error::invalid(format!("ac4: unknown framing '{other}'"))),
                }
            }
            "mode" => {
                self.mode = match value.as_str()? {
                    "waveform" => EncodeMode::Waveform,
                    "parametric" => EncodeMode::Parametric,
                    other => return Err(Error::invalid(format!("ac4: unknown mode '{other}'"))),
                }
            }
            "bandwidth" => self.bandwidth_hz = value.as_u32()?,
            "gop" => self.gop = value.as_u32()?,
            _ => return Err(Error::invalid(format!("ac4: unknown option '{key}'"))),
        }
        Ok(())
    }
}

/// Channel layouts the framework encoder dispatches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Mono,
    Stereo,
    S5_0,
    S5_1,
    S7_0,
    S7_1,
    I7_0_4,
    I7_1_4,
    I9_0_4,
    I9_1_4,
    I22_2,
}

impl Layout {
    fn from_channels(n: u16) -> Result<Self> {
        Ok(match n {
            1 => Self::Mono,
            2 => Self::Stereo,
            5 => Self::S5_0,
            6 => Self::S5_1,
            7 => Self::S7_0,
            8 => Self::S7_1,
            11 => Self::I7_0_4,
            12 => Self::I7_1_4,
            13 => Self::I9_0_4,
            14 => Self::I9_1_4,
            24 => Self::I22_2,
            other => {
                return Err(Error::unsupported(format!(
                    "ac4 encoder: no channel layout for {other} channels \
                     (supported: 1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 24)"
                )))
            }
        })
    }

    fn channels(self) -> u16 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::S5_0 => 5,
            Self::S5_1 => 6,
            Self::S7_0 => 7,
            Self::S7_1 => 8,
            Self::I7_0_4 => 11,
            Self::I7_1_4 => 12,
            Self::I9_0_4 => 13,
            Self::I9_1_4 => 14,
            Self::I22_2 => 24,
        }
    }

    fn channel_layout(self) -> ChannelLayout {
        match self {
            Self::Mono => ChannelLayout::Mono,
            Self::Stereo => ChannelLayout::Stereo,
            Self::S5_0 => ChannelLayout::Surround50,
            Self::S5_1 => ChannelLayout::Surround51,
            Self::S7_0 => ChannelLayout::Surround70,
            Self::S7_1 => ChannelLayout::Surround71,
            other => ChannelLayout::DiscreteN(other.channels()),
        }
    }

    /// Does the (layout, mode) pair emit P-frames when `gop > 1`?
    fn supports_pframes(self, mode: EncodeMode) -> bool {
        mode == EncodeMode::Parametric
            && matches!(
                self,
                Self::I7_0_4 | Self::I7_1_4 | Self::I9_0_4 | Self::I9_1_4 | Self::I22_2
            )
    }
}

/// Construct a framework encoder from stream parameters. Validates the
/// sample rate (48 kHz, or 44,1 kHz with `frame_rate_index = 13`), the
/// channel count (see [`Layout`]) and the option bag.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(Ac4Encoder::new(params)?))
}

/// The framework encoder. See the [module docs](self).
pub struct Ac4Encoder {
    codec_id: CodecId,
    out_params: CodecParameters,
    opts: Ac4EncoderOptions,
    layout: Layout,
    input_format: SampleFormat,
    sample_rate: u32,
    frame_len: usize,
    /// Per-channel float FIFO in the encoder's internal order.
    fifo: Vec<Vec<f32>>,
    /// Samples consumed from `fifo` so far (pts of the next frame).
    consumed: i64,
    frame_index: u64,
    inner: Ac4ImsEncoder,
    max_sfb: u32,
    max_sfb_lfe: u32,
    ready: std::collections::VecDeque<Packet>,
    flushed: bool,
}

impl Ac4Encoder {
    /// Typed constructor — the `params.options` bag is parsed into
    /// [`Ac4EncoderOptions`].
    pub fn new(params: &CodecParameters) -> Result<Self> {
        let opts: Ac4EncoderOptions = parse_options(&params.options)?;
        Self::with_options(params, opts)
    }

    /// Constructor taking an already-built options struct (ignores
    /// `params.options`).
    pub fn with_options(params: &CodecParameters, opts: Ac4EncoderOptions) -> Result<Self> {
        let channels = params
            .resolved_channels()
            .ok_or_else(|| Error::invalid("ac4 encoder: channel count not set"))?;
        let layout = Layout::from_channels(channels)?;
        let sample_rate = params
            .sample_rate
            .ok_or_else(|| Error::invalid("ac4 encoder: sample rate not set"))?;
        let fs_index = match sample_rate {
            48_000 => 1u8,
            44_100 => 0u8,
            other => {
                return Err(Error::unsupported(format!(
                    "ac4 encoder: sample rate {other} Hz (48000 or 44100 only)"
                )))
            }
        };
        if opts.frame_rate_index > 15 {
            return Err(Error::invalid("ac4 encoder: frame_rate_index out of range"));
        }
        let (_fps_milli, frame_len) =
            crate::toc::frame_rate_entry(opts.frame_rate_index, fs_index as u32);
        if frame_len == 0 {
            return Err(Error::unsupported(format!(
                "ac4 encoder: frame_rate_index {} is reserved at {sample_rate} Hz",
                opts.frame_rate_index
            )));
        }
        if opts.gop == 0 {
            return Err(Error::invalid("ac4 encoder: gop must be >= 1"));
        }
        if opts.mode == EncodeMode::Parametric
            && matches!(
                layout,
                Layout::S5_0 | Layout::S5_1 | Layout::S7_0 | Layout::S7_1
            )
        {
            return Err(Error::unsupported(
                "ac4 encoder: parametric (A-CPL) coding on 5.X / 7.X is not wired \
                 through the framework encoder yet — use mode=waveform",
            ));
        }
        let input_format = params.sample_format.unwrap_or(SampleFormat::S16);
        if !supported_input_format(input_format) {
            return Err(Error::unsupported(format!(
                "ac4 encoder: input sample format {input_format:?}"
            )));
        }
        let (max_sfb, max_sfb_lfe) = sfb_budget(frame_len, sample_rate, opts.bandwidth_hz)?;

        let mut inner = Ac4ImsEncoder::new();
        inner.fs_index = fs_index;
        inner.frame_rate_index = opts.frame_rate_index as u8;

        let mut out_params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        out_params.sample_rate = Some(sample_rate);
        out_params.channels = Some(channels);
        out_params.channel_layout = Some(layout.channel_layout());
        out_params.sample_format = Some(input_format);

        Ok(Self {
            codec_id: CodecId::new(crate::CODEC_ID_STR),
            out_params,
            opts,
            layout,
            input_format,
            sample_rate,
            frame_len: frame_len as usize,
            fifo: vec![Vec::new(); channels as usize],
            consumed: 0,
            frame_index: 0,
            inner,
            max_sfb,
            max_sfb_lfe,
            ready: std::collections::VecDeque::new(),
            flushed: false,
        })
    }

    /// Samples per AC-4 frame for this configuration.
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// The `max_sfb` the bandwidth option resolved to.
    pub fn max_sfb(&self) -> u32 {
        self.max_sfb
    }

    /// Append one input frame's samples (any supported [`SampleFormat`])
    /// to the per-channel FIFO, remapping from the documented input
    /// order into the inner encoder's slot order.
    fn push_samples(&mut self, af: &AudioFrame) -> Result<()> {
        let nch = self.layout.channels() as usize;
        let n = af.samples as usize;
        let fmt = self.input_format;
        let bps = fmt.bytes_per_sample();
        if fmt.is_planar() {
            if af.data.len() < nch {
                return Err(Error::invalid(format!(
                    "ac4 encoder: planar frame has {} planes, need {nch}",
                    af.data.len()
                )));
            }
            for (in_ch, plane) in af.data.iter().take(nch).enumerate() {
                if plane.len() < n * bps {
                    return Err(Error::invalid("ac4 encoder: short audio plane"));
                }
                let dst = internal_slot(self.layout, in_ch);
                let fifo = &mut self.fifo[dst];
                for i in 0..n {
                    fifo.push(sample_to_f32(fmt, &plane[i * bps..(i + 1) * bps]));
                }
            }
        } else {
            let plane = af
                .data
                .first()
                .ok_or_else(|| Error::invalid("ac4 encoder: audio frame has no data"))?;
            if plane.len() < n * nch * bps {
                return Err(Error::invalid(
                    "ac4 encoder: short interleaved audio buffer",
                ));
            }
            for in_ch in 0..nch {
                let dst = internal_slot(self.layout, in_ch);
                let fifo = &mut self.fifo[dst];
                for i in 0..n {
                    let off = (i * nch + in_ch) * bps;
                    fifo.push(sample_to_f32(fmt, &plane[off..off + bps]));
                }
            }
        }
        Ok(())
    }

    /// Encode as many complete frames as the FIFO holds.
    fn drain_frames(&mut self) {
        while self.fifo[0].len() >= self.frame_len {
            let n = self.frame_len;
            let chans: Vec<Vec<f32>> = self
                .fifo
                .iter_mut()
                .map(|f| f.drain(..n).collect())
                .collect();
            let raw = self.encode_one(&chans);
            let bytes = match self.opts.framing {
                Framing::Sync => wrap_sync_frame(&raw, false),
                Framing::SyncCrc => wrap_sync_frame(&raw, true),
                Framing::Raw => raw,
            };
            let iframe = self.inner.b_iframe_global;
            let pkt = Packet::new(0, TimeBase::new(1, self.sample_rate as i64), bytes)
                .with_pts(self.consumed)
                .with_dts(self.consumed)
                .with_duration(n as i64)
                .with_keyframe(iframe);
            self.ready.push_back(pkt);
            self.consumed += n as i64;
            self.frame_index += 1;
        }
    }

    /// Dispatch one frame (channels in the inner encoder's slot order)
    /// to the matching `Ac4ImsEncoder` path.
    fn encode_one(&mut self, chans: &[Vec<f32>]) -> Vec<u8> {
        let iframe = !self.layout.supports_pframes(self.opts.mode)
            || self.frame_index % self.opts.gop as u64 == 0;
        self.inner.b_iframe_global = iframe;
        let s: Vec<&[f32]> = chans.iter().map(|c| c.as_slice()).collect();
        let sfb = self.max_sfb;
        let lfe = self.max_sfb_lfe;
        let enc = &mut self.inner;
        match (self.layout, self.opts.mode) {
            (Layout::Mono, _) => enc.encode_frame_pcm_with_max_sfb(s[0], sfb),
            (Layout::Stereo, _) => enc.encode_frame_pcm_stereo_with_max_sfb(s[0], s[1], sfb),
            (Layout::S5_0, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_5_0_with_max_sfb(&arr5(&s), sfb)
            }
            (Layout::S5_1, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_5_1_with_max_sfb(&arr6(&s), sfb, lfe)
            }
            (Layout::S7_0, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_7_0_with_max_sfb(&arr7(&s), sfb, sfb)
            }
            (Layout::S7_1, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_7_1_with_max_sfb(&arr8(&s), sfb, sfb, lfe)
            }
            // Rejected at construction (`with_options`).
            (Layout::S5_0 | Layout::S5_1 | Layout::S7_0 | Layout::S7_1, EncodeMode::Parametric) => {
                Vec::new()
            }
            (Layout::I7_0_4, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_7_0_4_ice_scpl_sap_with_max_sfb(&arr11(&s), sfb)
            }
            (Layout::I7_1_4, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_7_1_4_ice_scpl_sap_with_max_sfb(&arr12(&s), sfb)
            }
            (Layout::I9_0_4, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_9_0_4_ice_scpl_sap_with_max_sfb(&arr13(&s), sfb)
            }
            (Layout::I9_1_4, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_9_1_4_ice_scpl_sap_with_max_sfb(&arr14(&s), sfb)
            }
            (Layout::I7_0_4, EncodeMode::Parametric) => {
                enc.encode_frame_pcm_7_0_4_ice_aspx_scpl_with_max_sfb(&arr11(&s), sfb)
            }
            (Layout::I7_1_4, EncodeMode::Parametric) => {
                enc.encode_frame_pcm_7_1_4_ice_aspx_scpl_with_max_sfb(&arr12(&s), sfb)
            }
            (Layout::I9_0_4, EncodeMode::Parametric) => {
                enc.encode_frame_pcm_9_0_4_ice_aspx_scpl_with_max_sfb(&arr13(&s), sfb)
            }
            (Layout::I9_1_4, EncodeMode::Parametric) => {
                enc.encode_frame_pcm_9_1_4_ice_aspx_scpl_with_max_sfb(&arr14(&s), sfb)
            }
            (Layout::I22_2, EncodeMode::Waveform) => {
                enc.encode_frame_pcm_22_2_simple_with_max_sfb(&arr24(&s), sfb)
            }
            (Layout::I22_2, EncodeMode::Parametric) => {
                enc.encode_frame_pcm_22_2_aspx_with_max_sfb(&arr24(&s), sfb)
            }
        }
    }
}

macro_rules! arr_fn {
    ($name:ident, $n:literal) => {
        fn $name<'a>(s: &[&'a [f32]]) -> [&'a [f32]; $n] {
            std::array::from_fn(|i| s[i])
        }
    };
}
arr_fn!(arr5, 5);
arr_fn!(arr6, 6);
arr_fn!(arr7, 7);
arr_fn!(arr8, 8);
arr_fn!(arr11, 11);
arr_fn!(arr12, 12);
arr_fn!(arr13, 13);
arr_fn!(arr14, 14);
arr_fn!(arr24, 24);

/// Map a documented input channel index onto the inner encoder's slot
/// for `layout` (see the module docs for both orders).
fn internal_slot(layout: Layout, in_ch: usize) -> usize {
    match layout {
        // Surround51: L R C LFE Ls Rs → inner L R C Ls Rs LFE.
        Layout::S5_1 => match in_ch {
            0..=2 => in_ch,
            3 => 5,
            4 => 3,
            5 => 4,
            _ => in_ch,
        },
        // Surround71: L R C LFE Ls Rs Lb Rb → inner L R C Ls Rs Lb Rb LFE.
        Layout::S7_1 => match in_ch {
            0..=2 => in_ch,
            3 => 7,
            4..=7 => in_ch - 1,
            _ => in_ch,
        },
        // Immersive with LFE: decoder order is LFE first, the inner
        // paths take it last.
        Layout::I7_1_4 => {
            if in_ch == 0 {
                11
            } else {
                in_ch - 1
            }
        }
        Layout::I9_1_4 => {
            if in_ch == 0 {
                13
            } else {
                in_ch - 1
            }
        }
        // 22.2: decoder order LFE, LFE2, 22 fullband; inner order 22
        // fullband then LFE, LFE2.
        Layout::I22_2 => {
            if in_ch < 2 {
                22 + in_ch
            } else {
                in_ch - 2
            }
        }
        _ => in_ch,
    }
}

/// Resolve the `(max_sfb, max_sfb_lfe)` pair for a bandwidth target:
/// the widest scale-factor band whose upper edge sits at or below
/// `bandwidth_hz`, capped by `num_sfb` for the transform length, and
/// the LFE band count capped by Table 106's `n_msfbl_bits` field width.
fn sfb_budget(frame_len: u32, sample_rate: u32, bandwidth_hz: u32) -> Result<(u32, u32)> {
    let sfbo = crate::sfb_offset::sfb_offset_48(frame_len).ok_or_else(|| {
        Error::unsupported(format!(
            "ac4 encoder: no scale-factor bands for {frame_len}"
        ))
    })?;
    let num_sfb = crate::tables::num_sfb_48(frame_len).unwrap_or(0);
    let (_, _, n_msfbl_bits) = crate::tables::n_msfb_bits_48(frame_len).unwrap_or((6, 5, 3));
    let nyquist = sample_rate as f64 / 2.0;
    let mut max_sfb = 1u32;
    for k in 1..=num_sfb {
        let edge_hz = sfbo[k as usize] as f64 * nyquist / frame_len as f64;
        if edge_hz <= bandwidth_hz as f64 {
            max_sfb = k;
        } else {
            break;
        }
    }
    let lfe_cap = if n_msfbl_bits == 0 {
        0
    } else {
        (1u32 << n_msfbl_bits) - 1
    };
    let max_sfb_lfe = lfe_cap.min(num_sfb).min(max_sfb);
    Ok((max_sfb, max_sfb_lfe))
}

/// Decode one sample of `fmt` from its little-endian bytes to a float
/// in `[-1, 1]`.
fn sample_to_f32(fmt: SampleFormat, b: &[u8]) -> f32 {
    match fmt {
        SampleFormat::U8 | SampleFormat::U8P => (b[0] as f32 - 128.0) / 128.0,
        SampleFormat::S8 => (b[0] as i8) as f32 / 128.0,
        SampleFormat::S16 | SampleFormat::S16P => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
        SampleFormat::S24 => {
            let v = ((b[2] as i32) << 24 | (b[1] as i32) << 16 | (b[0] as i32) << 8) >> 8;
            v as f32 / 8_388_608.0
        }
        SampleFormat::S32 | SampleFormat::S32P => {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
        }
        SampleFormat::F32 | SampleFormat::F32P => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        SampleFormat::F64 | SampleFormat::F64P => {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }
        // Rejected at construction (`supported_input_format`).
        _ => 0.0,
    }
}

/// The input sample formats [`sample_to_f32`] converts.
fn supported_input_format(fmt: SampleFormat) -> bool {
    matches!(
        fmt,
        SampleFormat::U8
            | SampleFormat::U8P
            | SampleFormat::S8
            | SampleFormat::S16
            | SampleFormat::S16P
            | SampleFormat::S24
            | SampleFormat::S32
            | SampleFormat::S32P
            | SampleFormat::F32
            | SampleFormat::F32P
            | SampleFormat::F64
            | SampleFormat::F64P
    )
}

impl Encoder for Ac4Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::invalid("ac4 encoder: send_frame after flush"));
        }
        let Frame::Audio(af) = frame else {
            return Err(Error::invalid("ac4 encoder: expected an audio frame"));
        };
        self.push_samples(af)?;
        self.drain_frames();
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.ready.pop_front() {
            return Ok(p);
        }
        if self.flushed {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        let pending = self.fifo[0].len();
        if pending > 0 {
            let pad = self.frame_len - pending;
            for f in &mut self.fifo {
                f.resize(f.len() + pad, 0.0);
            }
            self.drain_frames();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfb_budget_tracks_bandwidth() {
        let (full, lfe) = sfb_budget(1920, 48_000, 20_000).unwrap();
        // 20 kHz at tl = 1920 is bin 1600: the chosen band ends at or
        // below it and the next one crosses it.
        let sfbo = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        assert!(
            sfbo[full as usize] <= 1600 && sfbo[full as usize + 1] > 1600,
            "full = {full}"
        );
        assert_eq!(lfe, 7);
        let (narrow, _) = sfb_budget(1920, 48_000, 6_400).unwrap();
        assert!(narrow < full && narrow >= 35, "narrow = {narrow}");
        let (tiny, _) = sfb_budget(1920, 48_000, 0).unwrap();
        assert_eq!(tiny, 1);
    }

    #[test]
    fn slot_maps_are_permutations() {
        for layout in [
            Layout::S5_1,
            Layout::S7_1,
            Layout::I7_1_4,
            Layout::I9_1_4,
            Layout::I22_2,
            Layout::S7_0,
        ] {
            let n = layout.channels() as usize;
            let mut seen = vec![false; n];
            for i in 0..n {
                let s = internal_slot(layout, i);
                assert!(!seen[s], "{layout:?}: slot {s} hit twice");
                seen[s] = true;
            }
        }
    }

    #[test]
    fn options_parse_and_reject() {
        let bag = oxideav_core::CodecOptions::new()
            .set("framing", "raw")
            .set("mode", "parametric")
            .set("bandwidth", "12000")
            .set("gop", "4");
        let o: Ac4EncoderOptions = parse_options(&bag).unwrap();
        assert_eq!(o.framing, Framing::Raw);
        assert_eq!(o.mode, EncodeMode::Parametric);
        assert_eq!(o.bandwidth_hz, 12_000);
        assert_eq!(o.gop, 4);
        let bad = oxideav_core::CodecOptions::new().set("framing", "mp4");
        assert!(parse_options::<Ac4EncoderOptions>(&bad).is_err());
    }

    #[test]
    fn rejects_unsupported_shapes() {
        let mut p = CodecParameters::audio(CodecId::new("ac4"));
        p.sample_rate = Some(48_000);
        p.channels = Some(3);
        assert!(make_encoder(&p).is_err());
        p.channels = Some(2);
        p.sample_rate = Some(96_000);
        assert!(make_encoder(&p).is_err());
        p.sample_rate = Some(44_100);
        assert!(
            make_encoder(&p).is_err(),
            "44.1 kHz needs frame_rate_index 13"
        );
        p.options = oxideav_core::CodecOptions::new().set("frame_rate_index", "13");
        assert!(make_encoder(&p).is_ok());
    }
}
