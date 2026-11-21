//! Encoder-side transient detection and block-switching decisions
//! (the informative counterpart of TS 103 190-1 §5.5.3).
//!
//! A frame's partial blocks have their window centres in the span that
//! starts half a frame before the frame's own samples end
//! (Pseudocode 64 places a block of length `n` at `(Nfull − n) / 2` of
//! the composition buffer). The decision for frame `g` therefore looks
//! at the second half of input frame `g − 1` and the first half of
//! input frame `g`; coding frame `f` needs the decisions for `f` (its
//! own blocks) and `f + 1` (the right slope of its last block follows
//! the first block of the next frame), i.e. half a frame of lookahead.
//!
//! [`TransientDetector`] measures the high-pass energy of that span in
//! sixteen sub-blocks and reports an attack where a sub-block rises an
//! order of magnitude above both the recent past and the long-term
//! level. [`plan_for`] turns the attack positions into a
//! [`BlockPlan`]: the half of the frame that holds an attack is coded
//! as four short blocks whose scale factors are grouped before / from
//! the attack; a quiet half stays one half-length block; a quiet frame
//! stays one full block.

use crate::asf_blocks::{BlockPlan, LONG_FRAME_TL_CODE, full_block_tl_code};

/// Sub-blocks per frame span the detector resolves.
pub const SUB_BLOCKS: usize = 16;

/// A sub-block is an attack when its energy exceeds this multiple of
/// the reference level (the larger of the mean of the four sub-blocks
/// before it and a quarter of the long-term mean).
const ATTACK_RATIO: f64 = 10.0;

/// Sub-block energies below this (per sample, high-passed, full scale
/// = 1) never count as attacks: about −80 dBFS.
const ATTACK_FLOOR: f64 = 1e-8;

/// Attack positions of one frame span: per half, the sub-block (0..8)
/// where the first attack sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockChoice {
    /// First attack sub-block of each half of the span.
    pub attack: [Option<u8>; 2],
}

impl BlockChoice {
    /// No attack: the frame is one full block.
    pub fn is_long(&self) -> bool {
        self.attack == [None, None]
    }
}

/// Streaming attack detector over all channels of one element.
#[derive(Debug, Clone)]
pub struct TransientDetector {
    /// Energies of the last four sub-blocks analysed.
    tail: [f64; 4],
    /// Long-term mean sub-block energy.
    slow: f64,
    /// The decision for the frame about to be coded.
    pending: BlockChoice,
}

impl Default for TransientDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TransientDetector {
    /// Fresh detector (silent history, first frame long).
    pub fn new() -> Self {
        Self {
            tail: [0.0; 4],
            slow: 0.0,
            pending: BlockChoice::default(),
        }
    }

    /// Advance by one frame. `frames[ch]` are the `N` samples being
    /// coded, `lookahead[ch]` the first `N / 2` samples of the next
    /// frame (empty when unknown — the second half of the next span is
    /// then assumed quiet). Returns `(this frame's choice, the next
    /// frame's choice)`.
    pub fn advance(
        &mut self,
        frames: &[&[f32]],
        lookahead: &[&[f32]],
    ) -> (BlockChoice, BlockChoice) {
        let n = frames.first().map(|f| f.len()).unwrap_or(0);
        let current = self.pending;
        if n < SUB_BLOCKS * 2 {
            self.pending = BlockChoice::default();
            return (current, self.pending);
        }
        let sub = n / SUB_BLOCKS;
        let half = n / 2;
        let have_la = !lookahead.is_empty() && lookahead.iter().all(|l| l.len() >= half);
        let n_sub = if have_la { SUB_BLOCKS } else { SUB_BLOCKS / 2 };
        let mut energy = vec![0.0f64; n_sub];
        for (ch, f) in frames.iter().enumerate() {
            // First difference: keeps onsets, drops the low-frequency
            // swell of sustained tones.
            let mut prev = f[half - 1];
            let la: &[f32] = if have_la { lookahead[ch] } else { &[] };
            for (i, &x) in f[half..].iter().chain(la.iter().take(half)).enumerate() {
                let d = (x - prev) as f64;
                prev = x;
                if let Some(e) = energy.get_mut(i / sub) {
                    *e += d * d;
                }
            }
        }
        let scale = 1.0 / (sub * frames.len().max(1)) as f64;
        let mut next = BlockChoice::default();
        for (j, e) in energy.iter().enumerate() {
            let e = e * scale;
            let recent = self.tail.iter().sum::<f64>() / self.tail.len() as f64;
            let reference = recent.max(0.25 * self.slow);
            let h = j / (SUB_BLOCKS / 2);
            if e > ATTACK_FLOOR && e > ATTACK_RATIO * reference && next.attack[h].is_none() {
                next.attack[h] = Some((j % (SUB_BLOCKS / 2)) as u8);
            }
            self.tail.rotate_left(1);
            self.tail[3] = e;
            self.slow += (e - self.slow) / 32.0;
        }
        self.pending = next;
        (current, next)
    }
}

/// `max_sfb` for a transform of `tl` lines covering the same bandwidth
/// as `max_sfb_long` bands of the frame's full block: the first band
/// edge at or past the same fraction of the spectrum, capped by the
/// band count and by the `max_sfb` field width (Table 106).
pub fn matched_max_sfb(frame_len: u32, tl: u32, max_sfb_long: u32) -> u32 {
    let (Some(long), Some(short), Some(cap)) = (
        crate::sfb_offset::sfb_offset_48(frame_len),
        crate::sfb_offset::sfb_offset_48(tl),
        crate::tables::num_sfb_48(tl),
    ) else {
        return 1;
    };
    let edge = long[(max_sfb_long as usize).min(long.len() - 1)] as u64;
    let want = (edge * tl as u64).div_ceil(frame_len as u64);
    let mut k = 1u32;
    while k < cap && (short[k as usize] as u64) < want {
        k += 1;
    }
    let field = crate::tables::n_msfb_bits_48(tl).map(|b| b.0).unwrap_or(6);
    k.min(cap).min((1u32 << field) - 1).max(1)
}

/// Grouping bits of four short blocks split before the attack block
/// `a` (0..4): blocks `[0, a)` share scale factors, blocks `[a, 4)`
/// share scale factors.
fn split_bits(a: u8) -> [u8; 3] {
    let mut bits = [1u8; 3];
    if (1..4).contains(&a) {
        bits[a as usize - 1] = 0;
    }
    bits
}

/// The block arrangement for a frame span with attacks at `choice`.
/// `max_sfb_long` is the band count of the full block; the partial
/// blocks get the band counts that cover the same bandwidth.
pub fn plan_for(choice: BlockChoice, frame_len: u32, max_sfb_long: u32) -> BlockPlan {
    if choice.is_long() {
        return BlockPlan::full_block(frame_len, max_sfb_long);
    }
    let sfb = |code: u32| {
        let tl = crate::asf::resolve_transf_length(frame_len, false, code);
        matched_max_sfb(frame_len, tl, max_sfb_long)
    };
    if frame_len >= 1536 {
        // Per half: four blocks of N/8 (code 1) around an attack, one
        // block of N/2 (code 3) otherwise.
        let code = |h: usize| if choice.attack[h].is_some() { 1 } else { 3 };
        let (c0, c1) = (code(0), code(1));
        let mut grouping = Vec::new();
        if let Some(j) = choice.attack[0] {
            grouping.extend(split_bits(j / 2));
        }
        if c0 == c1 {
            // Equal lengths: the half boundary has its own bit — start
            // a new group there.
            grouping.push(0);
        }
        if let Some(j) = choice.attack[1] {
            grouping.extend(split_bits(j / 2));
        }
        BlockPlan {
            frame_len,
            long: false,
            transf_length: [c0, c1],
            grouping,
            max_sfb: [sfb(c0), sfb(c1)],
        }
    } else {
        // One transform length per frame: four blocks of N/4, split
        // before the first attack.
        let full = full_block_tl_code(frame_len);
        debug_assert_ne!(full, LONG_FRAME_TL_CODE);
        let code = full - 2;
        let first = match choice.attack {
            [Some(j), _] => j / 4,
            [None, Some(j)] => 2 + j / 4,
            [None, None] => 0,
        };
        BlockPlan {
            frame_len,
            long: false,
            transf_length: [code; 2],
            grouping: split_bits(first).to_vec(),
            max_sfb: [sfb(code); 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_signals_stay_long_and_a_click_switches() {
        let n = 1920usize;
        let mut det = TransientDetector::new();
        let tone = |f: usize| -> Vec<f32> {
            (0..n + n / 2)
                .map(|i| {
                    let t = (f * n + i) as f32 / 48_000.0;
                    0.3 * (std::f32::consts::TAU * 60.0 * t).sin()
                        + 0.2 * (std::f32::consts::TAU * 2_500.0 * t).sin()
                })
                .collect()
        };
        for f in 0..6 {
            let x = tone(f);
            let (cur, next) = det.advance(&[&x[..n]], &[&x[n..]]);
            // The stream start is an onset from silence; once the
            // history holds signal, a steady mix never switches.
            if f > 1 {
                assert!(
                    cur.is_long() && next.is_long(),
                    "frame {f}: {cur:?} {next:?}"
                );
            }
        }
        // A click 300 samples into the lookahead of frame 6: second half
        // of the next span, sub-block 2.
        let mut x = tone(6);
        for v in &mut x[n + 300..n + 330] {
            *v += 0.8;
        }
        x[n + 300] += 0.9;
        let (_, next) = det.advance(&[&x[..n]], &[&x[n..]]);
        assert_eq!(next.attack[0], None);
        assert_eq!(next.attack[1], Some(2));
        let plan = plan_for(next, 1920, 50);
        assert_eq!(plan.transf_length, [3, 1]);
        assert_eq!(plan.windows(), vec![960, 240, 240, 240, 240]);
        // Attack in block 1 of the short half: groups [0], [1, 2, 3].
        let l = plan.layout().unwrap();
        assert_eq!(
            l.groups.iter().map(|g| g.num_win).collect::<Vec<_>>(),
            vec![1, 1, 3]
        );
    }

    #[test]
    fn plans_cover_every_frame_length() {
        for fl in [2048u32, 1920, 1536, 1024, 960, 768, 512, 384] {
            for choice in [
                BlockChoice {
                    attack: [Some(5), None],
                },
                BlockChoice {
                    attack: [None, Some(0)],
                },
                BlockChoice {
                    attack: [Some(1), Some(7)],
                },
            ] {
                let plan = plan_for(choice, fl, 20);
                let l = plan
                    .layout()
                    .unwrap_or_else(|| panic!("no layout for {fl} {choice:?}"));
                assert_eq!(l.frame_len, fl as usize);
                assert_eq!(
                    plan.grouping.len() as u32,
                    plan.n_grp_bits(),
                    "{fl} {choice:?}"
                );
                assert!(l.windows.len() > 1);
            }
        }
    }

    #[test]
    fn matched_band_counts_keep_the_bandwidth() {
        let long = crate::sfb_offset::sfb_offset_48(1920).unwrap();
        let short = crate::sfb_offset::sfb_offset_48(240).unwrap();
        let k = matched_max_sfb(1920, 240, 40);
        let want = long[40] as f32 / 1920.0;
        let got = short[k as usize] as f32 / 240.0;
        assert!(got >= want - 1e-6 && got < want + 0.1, "{got} vs {want}");
    }
}
