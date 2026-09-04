//! Encoder-side dialogue-enhancement authoring — the inverse of the
//! §5.7.8 application tool for the **parametric channel-independent**
//! method (`de_method = 0`, §5.7.8.7).
//!
//! The decoder multiplies every time-frequency sample of a processed
//! channel `i` by `1 + g · p_i[band]` (`g = 10^(G_DE/20) − 1`, clamped
//! to `Gmax`). The spec leaves `p_i` to the encoder; this module
//! estimates it from a **dialogue stem** and the **final mix** of each
//! processed channel: with `d_i = p_i · m_i` the enhanced output
//! `m_i + g · d_i` boosts exactly the dialogue part, so the natural
//! estimate is the per-band amplitude ratio
//!
//! ```text
//!   p_i[band] = sqrt( E_dialogue,i[band] / E_mix,i[band] )
//! ```
//!
//! over the Table 173 parameter bands of the §5.7.6.2 QMF analysis
//! (streaming banks, so consecutive frames analyse as one signal),
//! quantised to the nearest Table 209 entry. A channel whose band
//! holds only dialogue gets `p = 1` (the full `G_DE` boost); a band
//! without dialogue gets `p = 0` (untouched). Ratios above one — a
//! dialogue stem louder than the mix, e.g. after ducking — are kept up
//! to the table's 9,0 ceiling.
//!
//! The result is a [`DeData`] ready for
//! [`crate::de::write_dialog_enhancement`] under the caller's
//! [`DeConfig`] (which fixes `de_max_gain` / `de_channel_config`). The
//! encoder emits it on every frame with the keep-flags clear, so P
//! frames stay self-contained.

use crate::de::{DeConfig, DeData, DeMethod, DE_NR_BANDS};
use crate::de_apply::{processed_slots, DE_BAND_LAST_SB, DE_PAR_DQ_CI};
use crate::qmf::{QmfAnalysisBank, NUM_QMF_SUBBANDS};

/// Quantise a channel-independent parameter to the nearest Table 209
/// index (`0..=31`; the table spans `0,0..=9,0`).
pub fn quantise_par_ci(p: f32) -> i32 {
    let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &v) in DE_PAR_DQ_CI.iter().enumerate() {
        let d = (v - p).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best as i32
}

/// Energies of the eight Table 173 dialogue-enhancement parameter
/// bands from one block of QMF slots.
pub fn de_band_energies(slots: &[[(f32, f32); NUM_QMF_SUBBANDS]]) -> [f64; DE_NR_BANDS] {
    let mut e = [0.0f64; DE_NR_BANDS];
    let mut first = 0usize;
    for (band, &last) in DE_BAND_LAST_SB.iter().enumerate() {
        for slot in slots {
            for s in &slot[first..=last] {
                e[band] += (s.0 as f64).powi(2) + (s.1 as f64).powi(2);
            }
        }
        first = last + 1;
    }
    e
}

/// Streaming authoring state: one QMF analysis bank per (slot, stem)
/// so the per-band energies come from a continuous analysis.
#[derive(Debug, Default)]
pub struct DeAuthor {
    mix_banks: Vec<QmfAnalysisBank>,
    dlg_banks: Vec<QmfAnalysisBank>,
}

impl DeAuthor {
    /// Fresh state (banks are allocated on first use).
    pub fn new() -> Self {
        Self::default()
    }

    /// Author one frame of channel-independent `de_data()` for `cfg`.
    ///
    /// `mix` / `dialogue` hold the final-mix PCM and the dialogue-stem
    /// PCM of the three dialogue-enhancement slots in (Left, Right,
    /// Centre) order; only the slots `cfg.channel_config` marks as
    /// processed are analysed (Table 171). Slots the layout lacks may
    /// be empty slices. Returns `None` when `cfg` is not a
    /// channel-independent configuration (`de_method ∈ {0, 2}`) or
    /// processes no channel.
    pub fn author_channel_independent(
        &mut self,
        cfg: &DeConfig,
        mix: [&[f32]; 3],
        dialogue: [&[f32]; 3],
    ) -> Option<DeData> {
        if !matches!(
            cfg.method,
            DeMethod::ChannelIndependent | DeMethod::HybridChannelIndependent
        ) || cfg.nr_channels() == 0
        {
            return None;
        }
        while self.mix_banks.len() < 3 {
            self.mix_banks.push(QmfAnalysisBank::new());
            self.dlg_banks.push(QmfAnalysisBank::new());
        }
        let slots = processed_slots(cfg);
        let mut de_par = Vec::with_capacity(cfg.nr_channels() as usize);
        for slot in 0..3 {
            if !slots[slot] {
                continue;
            }
            let n = mix[slot].len().min(dialogue[slot].len());
            let n = n - n % NUM_QMF_SUBBANDS;
            let e_mix = de_band_energies(&self.mix_banks[slot].process_block(&mix[slot][..n]));
            let e_dlg = de_band_energies(&self.dlg_banks[slot].process_block(&dialogue[slot][..n]));
            let mut row = [0i32; DE_NR_BANDS];
            for band in 0..DE_NR_BANDS {
                let p = if e_mix[band] > 0.0 {
                    (e_dlg[band] / e_mix[band]).sqrt() as f32
                } else {
                    0.0
                };
                row[band] = quantise_par_ci(p);
            }
            de_par.push(row);
        }
        Some(DeData {
            keep_pos_flag: false,
            mix_coef1_idx: None,
            mix_coef2_idx: None,
            keep_data_flag: false,
            ms_proc_flag: false,
            de_par,
            signal_contribution: match cfg.method {
                DeMethod::HybridChannelIndependent => Some(0),
                _ => None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiser_hits_table_entries_and_clamps() {
        assert_eq!(quantise_par_ci(0.0), 0);
        assert_eq!(quantise_par_ci(1.0), 10);
        assert_eq!(quantise_par_ci(0.97), 10);
        assert_eq!(quantise_par_ci(1.6), 15); // 1,5 (Δ 0,1) beats 1,75 (Δ 0,15)
        assert_eq!(quantise_par_ci(1.7), 16);
        assert_eq!(quantise_par_ci(100.0), 31);
        assert_eq!(quantise_par_ci(-3.0), 0);
        assert_eq!(quantise_par_ci(f32::NAN), 0);
    }

    #[test]
    fn full_dialogue_channel_authors_unity_parameters() {
        let n = 1920;
        let tone: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let cfg = DeConfig {
            method: DeMethod::ChannelIndependent,
            max_gain: 3,
            channel_config: 0b001, // centre only
        };
        let mut a = DeAuthor::new();
        // Warm the streaming banks, then read the settled frame.
        let _ = a.author_channel_independent(&cfg, [&[], &[], &tone], [&[], &[], &tone]);
        let d = a
            .author_channel_independent(&cfg, [&[], &[], &tone], [&[], &[], &tone])
            .unwrap();
        assert_eq!(d.de_par.len(), 1);
        // 440 Hz sits in QMF subband 1 (375 Hz per band) → parameter
        // band 1; every band with energy reads the dialogue = mix
        // ratio of one, empty bands read zero.
        assert_eq!(d.de_par[0][1], 10, "{:?}", d.de_par[0]);
        // A dialogue stem at half the mix amplitude reads 0,5 once the
        // streaming banks have settled on the new level (the first
        // half-level frame still carries the previous frame's tail in
        // the QMF delay line).
        let half: Vec<f32> = tone.iter().map(|v| v * 0.5).collect();
        let _ = a.author_channel_independent(&cfg, [&[], &[], &tone], [&[], &[], &half]);
        let d = a
            .author_channel_independent(&cfg, [&[], &[], &tone], [&[], &[], &half])
            .unwrap();
        assert_eq!(d.de_par[0][1], 5, "{:?}", d.de_par[0]);
    }

    #[test]
    fn cross_channel_configs_are_refused() {
        let cfg = DeConfig {
            method: DeMethod::CrossChannel,
            max_gain: 0,
            channel_config: 0b111,
        };
        assert!(DeAuthor::new()
            .author_channel_independent(&cfg, [&[], &[], &[]], [&[], &[], &[]])
            .is_none());
    }
}
