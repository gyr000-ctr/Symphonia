// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Packet loss concealment (PLC) engine.
//!
//! When a packet is missing or undecodeable the engine extrapolates a replacement frame from the
//! signal state captured from the most recent good frames:
//!
//! 1. **Pitch (fundamental period)** — normalized square-difference search over the lags relevant
//!    to speech at 48 kHz.
//! 2. **LPC (spectral envelope)** — 10th-order autocorrelation analysis with a white-noise floor,
//!    solved by Levinson-Durbin recursion, which guarantees a minimum-phase synthesis filter.
//! 3. **Gain** — RMS energy of the previous frame; extrapolated energy decays per lost frame and
//!    fades to silence during very long loss bursts.
//!
//! On recovery the first real frame is smoothly cross-faded against the LPC extrapolation so the
//! discontinuity at the concealment boundary is inaudible.
//!
//! # Bounds and allocation
//!
//! All state lives in fixed-size arrays (see the `const` values below). No method on this module
//! allocates, regardless of how many packets are lost in a row. Every emitted sample is checked
//! for finiteness and clamped to the valid `f32` PCM range, so extrapolation can never produce
//! `NaN`, infinities or clipping beyond the representable range.

/// Maximum number of channels handled by one Opus decoder instance (RFC 6716: mono or stereo).
pub const MAX_CHANNELS: usize = 2;

/// Samples of signal history retained per channel (120 ms at 48 kHz).
const HISTORY_LEN: usize = 5_760;

/// Minimum pitch lag searched, in samples (equivalent to 1.5 kHz at 48 kHz).
const MIN_PITCH_LAG: usize = 32;

/// Maximum pitch lag searched, in samples (50 Hz at 48 kHz).
const MAX_PITCH_LAG: usize = 960;

/// Order of the LPC synthesis / analysis filter.
const LPC_ORDER: usize = 10;

/// Window length used for LPC autocorrelation analysis (40 ms).
const LPC_WINDOW: usize = 1_920;

/// Default pitch lag when no periodicity can be established (20 ms).
const DEFAULT_PITCH_LAG: usize = 480;

/// Per-lost-frame gain applied to the extrapolated energy (≈0.2 dB attenuation).
const PER_FRAME_GAIN: f32 = 0.975;

/// Number of consecutive lost frames before the linear fade-to-silence begins.
const LONG_LOSS_FRAMES: u32 = 8;

/// Number of frames over which a long loss burst fades out to silence.
const LONG_LOSS_FADE_FRAMES: u32 = 8;

/// Recovery cross-fade duration in milliseconds.
const XFADE_MS: u32 = 5;

/// Recovery cross-fade length in samples at 48 kHz, capped per frame by half the frame length.
pub const XFADE_SAMPLES: usize = (XFADE_MS as usize) * 48;

/// Largest single Opus frame (120 ms): used to size the in-struct cross-fade scratch region.
const MAX_FRAME_SAMPLES: usize = 5_760;

/// Hard clamp for every sample leaving the engine. PCM f32 is defined on `[-1.0, 1.0]`.
const SAMPLE_LIMIT: f32 = 1.0;

/// Per-channel concealment state.
#[derive(Clone)]
pub struct PlcChannel {
    /// Recent signal samples; the valid region is the tail `hist_len` entries.
    history: [f32; HISTORY_LEN],
    /// Number of valid samples currently stored in `history` (`0..=HISTORY_LEN`).
    hist_len: usize,
    /// LPC predictor coefficients (all-pole synthesis filter), order [`LPC_ORDER`].
    lpc: [f32; LPC_ORDER],
    /// Estimated pitch lag in samples.
    pitch: usize,
    /// RMS energy of the most recent good frame.
    rms: f32,
    /// Number of consecutive lost frames since the last good frame.
    loss_count: u32,
    /// Number of cross-fade samples still to apply after recovery.
    xfade_remaining: usize,
    /// Total cross-fade length of the current recovery (interpolation denominator).
    xfade_total: usize,
    /// Whether at least one analyzable good frame has been observed since reset.
    primed: bool,
}

impl PlcChannel {
    const fn new() -> Self {
        Self {
            history: [0.0; HISTORY_LEN],
            hist_len: 0,
            lpc: [0.0; LPC_ORDER],
            pitch: DEFAULT_PITCH_LAG,
            rms: 0.0,
            loss_count: 0,
            xfade_remaining: 0,
            xfade_total: 0,
            primed: false,
        }
    }

    /// Return all concealment state to its post-construction defaults.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn is_primed(&self) -> bool {
        self.primed
    }

    /// Push a good frame into the history, dropping the oldest samples when full, then refresh the
    /// pitch, LPC and gain analysis.
    ///
    /// Non-finite input samples are replaced with silence in place so a poisoned frame can never
    /// corrupt the predictor state.
    pub fn capture_frame(&mut self, frame: &mut [f32]) {
        // A new good frame ends any loss streak.
        self.loss_count = 0;

        for sample in frame.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }

        let input_len = frame.len();
        if self.hist_len + input_len <= HISTORY_LEN {
            let start = self.hist_len;
            self.history[start..start + input_len].copy_from_slice(frame);
            self.hist_len += input_len;
        }
        else {
            // Drop exactly as many oldest samples as required to fit the new frame.
            let drop = self.hist_len + input_len - HISTORY_LEN;
            self.history.copy_within(drop..self.hist_len, 0);
            self.hist_len -= drop;
            self.history[self.hist_len..self.hist_len + input_len].copy_from_slice(frame);
            self.hist_len += input_len;
        }

        self.rms = rms(frame);

        // Analysis needs at least one maximum-lag span plus one LPC window of context.
        if self.hist_len >= MAX_PITCH_LAG + LPC_ORDER {
            self.pitch = estimate_pitch(&self.history[..self.hist_len]);
        }
        if self.hist_len >= LPC_WINDOW {
            self.lpc = estimate_lpc(&self.history[..self.hist_len]);
        }
        self.primed = self.hist_len > LPC_ORDER;
    }

    /// Extrapolate one lost frame of `out.len()` samples into `out` and append it to the history.
    ///
    /// Before the first good frame (or after a reset) the output is silence, but the timeline is
    /// preserved: exactly `out.len()` samples are always written.
    pub fn extrapolate(&mut self, out: &mut [f32]) {
        extrapolate_frame(self, out, true);
    }

    /// Arm a recovery cross-fade of at most half a frame following one or more losses.
    pub(crate) fn begin_recovery(&mut self, frame_len: usize) {
        debug_assert_eq!(self.xfade_remaining, 0);
        let len = XFADE_SAMPLES.min(frame_len / 2).min(frame_len);
        self.xfade_remaining = len;
        self.xfade_total = len.max(1);
    }

    /// Blend a recovered real frame against the extrapolation at the concealment boundary.
    ///
    /// The blended region ramps from pure extrapolation at sample 0 to the pure real signal at the
    /// end of the ramp. The caller provides the fixed `scratch` region (owned by [`Plc`]) used to
    /// hold the extrapolated reference. Must be called before
    /// [`capture_frame`](Self::capture_frame).
    pub(crate) fn apply_recovery(&mut self, frame: &mut [f32], scratch: &mut [f32]) {
        if self.xfade_remaining == 0 {
            return;
        }

        let total = self.xfade_total;
        let blend_len = self.xfade_remaining.min(frame.len()).min(scratch.len());

        // Generate the "what the PLC would have produced" reference into the caller-provided
        // scratch region so no self-aliasing borrow occurs.
        extrapolate_frame(self, &mut scratch[..blend_len], true);
        // extrapolate_frame() advanced the loss counter and appended predicted samples; undo both
        // since this is a good frame and the blended signal will be captured instead.
        self.loss_count = 0;
        self.rollback_history(blend_len);

        for i in 0..blend_len {
            let real_weight = (total - self.xfade_remaining + i + 1) as f32 / total as f32;
            let plc_weight = 1.0 - real_weight;
            let real = sanitize(frame[i]);
            frame[i] = sanitize(plc_weight * scratch[i] + real_weight * real);
        }

        self.xfade_remaining -= blend_len;
        if self.xfade_remaining == 0 {
            self.xfade_total = 0;
        }
    }

    fn push_history(&mut self, samples: &[f32]) {
        let input_len = samples.len();
        if self.hist_len + input_len <= HISTORY_LEN {
            let start = self.hist_len;
            self.history[start..start + input_len].copy_from_slice(samples);
            self.hist_len += input_len;
        }
        else {
            let drop = self.hist_len + input_len - HISTORY_LEN;
            self.history.copy_within(drop..self.hist_len, 0);
            self.hist_len -= drop;
            let start = self.hist_len;
            self.history[start..start + input_len].copy_from_slice(samples);
            self.hist_len += input_len;
        }
    }

    fn rollback_history(&mut self, n: usize) {
        let removed = n.min(self.hist_len);
        self.hist_len -= removed;
        // Clear the dropped tail so stale data can never re-enter analysis.
        self.history[self.hist_len..self.hist_len + removed].fill(0.0);
    }
}

/// Top-level PLC engine holding one [`PlcChannel`] per supported channel.
pub struct Plc {
    channels: [PlcChannel; MAX_CHANNELS],
    n_channels: usize,
    /// Per-channel fixed scratch regions for the extrapolated half of a recovery cross-fade.
    /// Kept outside [`PlcChannel`] so a channel and its scratch can be borrowed simultaneously.
    scratch: [[f32; MAX_FRAME_SAMPLES]; MAX_CHANNELS],
}

impl Plc {
    /// Create a PLC engine for `n_channels` channels (1 or 2 per RFC 6716).
    pub fn new(n_channels: usize) -> Self {
        assert!(n_channels >= 1 && n_channels <= MAX_CHANNELS);
        Self {
            channels: [PlcChannel::new(), PlcChannel::new()],
            n_channels,
            scratch: [[0.0; MAX_FRAME_SAMPLES], [0.0; MAX_FRAME_SAMPLES]],
        }
    }

    /// Number of channels the engine was configured for.
    pub fn channels(&self) -> usize {
        self.n_channels
    }

    /// Reconfigure the channel count. A channel-count change is a bitstream switch: all concealment
    /// state is discarded so stale cross-channel state can never leak through.
    pub fn set_channels(&mut self, n_channels: usize) {
        assert!(n_channels >= 1 && n_channels <= MAX_CHANNELS);
        if n_channels != self.n_channels {
            self.reset();
            self.n_channels = n_channels;
        }
    }

    /// Full state reset (seek, bitstream switch, explicit decoder reset).
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
    }

    /// Whether every channel has observed at least one good frame.
    pub fn is_primed(&self) -> bool {
        self.channels[..self.n_channels].iter().all(PlcChannel::is_primed)
    }

    /// Record a successfully decoded frame. `planes` gives one planar slice per channel, all of
    /// equal length.
    ///
    /// If the frame follows one or more losses, a recovery cross-fade is applied in place before
    /// the signal is absorbed into the predictor state.
    pub fn good_frame(&mut self, planes: &mut [&mut [f32]]) {
        debug_assert_eq!(planes.len(), self.n_channels);
        let frame_len = planes[0].len();

        let recovering =
            self.channels[..self.n_channels].iter().any(|c| c.loss_count != 0);

        if recovering {
            for ch in &mut self.channels[..self.n_channels] {
                ch.begin_recovery(frame_len);
            }
        }

        // Indexed access keeps each channel borrow disjoint from its scratch and plane borrows.
        for idx in 0..self.n_channels {
            let ch = &mut self.channels[idx];
            let scratch = &mut self.scratch[idx];
            ch.apply_recovery(planes[idx], scratch);
            ch.capture_frame(planes[idx]);
        }
    }

    /// Extrapolate one lost frame into the provided planar output slices. Always writes exactly
    /// `planes[0].len()` samples per channel; silence is emitted when no state is available.
    pub fn lost_frame(&mut self, planes: &mut [&mut [f32]]) {
        debug_assert_eq!(planes.len(), self.n_channels);
        for (ch, plane) in self.channels[..self.n_channels].iter_mut().zip(planes.iter_mut()) {
            ch.extrapolate(plane);
        }
    }

    #[cfg(test)]
    fn channel(&self, idx: usize) -> &PlcChannel {
        &self.channels[idx]
    }
}

/// Clamp and validate a single synthesized sample.
#[inline]
fn sanitize(sample: f32) -> f32 {
    if sample.is_finite() { sample.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT) } else { 0.0 }
}

/// Core LPC/pitch extrapolation routine, factored out as a free function so the recovery path can
/// write into a caller-owned scratch buffer without aliasing `PlcChannel` borrows.
///
/// When `append` is true the synthesized samples are appended to the channel history (normal loss
/// handling and the first recovery reference); the caller is responsible for rolling the history
/// back when a generated reference is discarded.
fn extrapolate_frame(ch: &mut PlcChannel, out: &mut [f32], append: bool) {
    let n = out.len();

    if !ch.primed {
        // First-frame loss: nothing to extrapolate from. Emit silence, keep the timeline.
        out.fill(0.0);
        ch.loss_count = ch.loss_count.saturating_add(1);
        return;
    }

    ch.loss_count = ch.loss_count.saturating_add(1);

    // Long-burst attenuation: geometric decay followed by a linear fade to silence.
    let geometric = PER_FRAME_GAIN.powi(ch.loss_count.min(32) as i32);
    let long_fade = if ch.loss_count <= LONG_LOSS_FRAMES {
        1.0
    }
    else if ch.loss_count >= LONG_LOSS_FRAMES + LONG_LOSS_FADE_FRAMES {
        0.0
    }
    else {
        1.0 - (ch.loss_count - LONG_LOSS_FRAMES) as f32 / LONG_LOSS_FADE_FRAMES as f32
    };
    let gain = geometric * long_fade;

    let pitch = ch.pitch.clamp(MIN_PITCH_LAG, MAX_PITCH_LAG);

    // Build the excitation seed from the residual of the most recent `pitch` samples. Repeating a
    // full period of residual through the LPC filter preserves both periodicity and the spectral
    // envelope captured by the predictor.
    let mut seed = [0.0f32; MAX_PITCH_LAG];
    for (i, slot) in seed.iter_mut().enumerate().take(pitch) {
        let pos = ch.hist_len - pitch + i;
        let mut residual = ch.history[pos];
        for k in 0..LPC_ORDER.min(pos) {
            residual -= ch.lpc[k] * ch.history[pos - 1 - k];
        }
        *slot = if residual.is_finite() { residual } else { 0.0 };
    }

    for i in 0..n {
        let excitation = seed[i % pitch] * gain;
        let mut sample = excitation;
        for k in 0..LPC_ORDER {
            let prev = if i > k {
                // Sample synthesized earlier in this frame (tail of the logical history).
                out[i - 1 - k]
            }
            else {
                // Sample from the captured history.
                let hist_idx = ch.hist_len as isize - (k as isize - i as isize + 1);
                if hist_idx >= 0 {
                    ch.history[hist_idx as usize]
                }
                else {
                    0.0
                }
            };
            sample += ch.lpc[k] * prev;
        }
        out[i] = if sample.is_finite() { sample } else { 0.0 };
    }

    // Energy-preservation step: an all-pole filter can resonate and amplify the excitation beyond
    // the target gain, which would defeat the per-loss attenuation. Re-scale the synthesized frame
    // to the RMS of the last good frame multiplied by the loss gain. The final sanitize() clamp
    // still guarantees no clipping beyond the PCM range.
    let target_rms = ch.rms * gain;
    let current_rms = rms(out);
    if current_rms > 1e-12 && target_rms < current_rms {
        let scale = target_rms / current_rms;
        for sample in out.iter_mut() {
            *sample = sanitize(*sample * scale);
        }
    }
    else {
        for sample in out.iter_mut() {
            *sample = sanitize(*sample);
        }
    }

    if append {
        // Append the synthesized samples so consecutive losses remain continuous.
        ch.push_history(out);
    }
}

/// RMS energy of a frame; zero for empty or non-finite-only input.
fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &s in frame {
        if s.is_finite() {
            sum += (s as f64) * (s as f64);
        }
    }
    (sum / frame.len() as f64).sqrt() as f32
}

/// Pitch estimation via normalized square-difference over [`MIN_PITCH_LAG`..MAX_PITCH_LAG`].
///
/// Returns the lag with the smallest normalized difference; near-silent signals fall back to
/// [`DEFAULT_PITCH_LAG`].
fn estimate_pitch(history: &[f32]) -> usize {
    let n = history.len();
    let search_end = MAX_PITCH_LAG.min(n - 1);
    if search_end <= MIN_PITCH_LAG {
        return DEFAULT_PITCH_LAG;
    }

    // Reference energy over the most recent maximum-lag span; silence => no meaningful pitch.
    let ref_start = n - MAX_PITCH_LAG;
    let mut energy = 0.0f64;
    for &s in &history[ref_start..] {
        energy += (s as f64) * (s as f64);
    }
    if energy < 1e-8 {
        return DEFAULT_PITCH_LAG;
    }

    let mut best_lag = DEFAULT_PITCH_LAG;
    let mut best_score = f64::MAX;

    for lag in MIN_PITCH_LAG..=search_end {
        let start = n - MAX_PITCH_LAG;
        let mut diff = 0.0f64;
        let mut lag_energy = 0.0f64;
        for i in 0..MAX_PITCH_LAG {
            let a = history[start + i] as f64;
            let b = history[start + i - lag] as f64;
            let d = a - b;
            diff += d * d;
            lag_energy += b * b;
        }
        // Normalize so amplitude differences do not bias the search.
        let score = diff / (lag_energy + 1e-10);
        if score < best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    best_lag
}

/// 10th-order LPC coefficient estimation via the Burg method.
///
/// Burg estimates reflection coefficients directly from forward/backward prediction errors rather
/// than from an autocorrelation matrix. Every reflection coefficient is mathematically guaranteed
/// to lie inside the unit circle (and is clamped there defensively), so the resulting all-pole
/// predictor is always minimum-phase (stable), including for highly tonal or short signals where
/// the autocorrelation/Levinson formulation becomes ill-conditioned. All working storage is
/// fixed-size; the function never allocates.
fn estimate_lpc(history: &[f32]) -> [f32; LPC_ORDER] {
    let n = history.len();
    let start = n - LPC_WINDOW;

    // Hamming-window the analysis frame to suppress spectral leakage.
    let mut x = [0.0f64; LPC_WINDOW];
    for i in 0..LPC_WINDOW {
        x[i] = (history[start + i] as f64) * hamming(i);
    }
    let n = LPC_WINDOW;

    // Forward and backward prediction error arrays, initialized to the signal itself.
    let mut ef = x;
    let mut eb = x;

    // Initial error energy.
    let mut e: f64 = x.iter().map(|v| v * v).sum::<f64>() * 2.0 / n as f64;

    let mut coefs = [0.0f64; LPC_ORDER];
    const MAX_REFLECTION: f64 = 0.997;

    for m in 0..LPC_ORDER {
        // Reflection coefficient from the cross-correlation of forward and backward errors.
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in (m + 1)..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }
        let mut k = if den > f64::EPSILON { 2.0 * num / den } else { 0.0 };
        k = k.clamp(-MAX_REFLECTION, MAX_REFLECTION);

        // Update the prediction errors before overwriting coefficients.
        for i in (m + 1..n).rev() {
            let fi = ef[i];
            let bi = eb[i - 1];
            ef[i] = fi - k * bi;
            eb[i] = bi - k * fi;
        }

        // Update the order-m predictor: a'[j] = a[j] - k·a[m-1-j], then set a[m] = k.
        let prev = coefs;
        for j in 0..m {
            coefs[j] = prev[j] - k * prev[m - 1 - j];
        }
        coefs[m] = k;

        e *= 1.0 - k * k;
        if e <= f64::EPSILON {
            break;
        }
    }

    let mut lpc = [0.0f32; LPC_ORDER];
    for (out, coef) in lpc.iter_mut().zip(coefs.iter()) {
        let v = *coef as f32;
        *out = if v.is_finite() { v } else { 0.0 };
    }
    lpc
}

#[inline]
fn hamming(i: usize) -> f64 {
    0.54 - 0.46 * (2.0 * core::f64::consts::PI * i as f64 / (LPC_WINDOW as f64 - 1.0)).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a stationary sinusoid, `frames` blocks of `frame_len` samples, at frequency `freq`.
    fn tone_stream(frame_len: usize, frames: usize, freq: f32) -> Vec<Vec<f32>> {
        (0..frames)
            .map(|f| {
                (0..frame_len)
                    .map(|i| {
                        let t = (f * frame_len + i) as f32 / 48_000.0;
                        0.5 * (2.0 * core::f32::consts::PI * freq * t).sin()
                    })
                    .collect()
            })
            .collect()
    }

    fn feed_good(plc: &mut Plc, frame: &[f32]) {
        let mut buf = frame.to_vec();
        plc.good_frame(&mut [&mut buf]);
    }

    #[test]
    fn first_loss_before_any_good_frame_is_silence_but_timed() {
        let mut plc = Plc::new(1);
        let mut out = vec![0.25f32; 960];
        plc.lost_frame(&mut [&mut out]);
        assert!(out.iter().all(|&s| s == 0.0));
        assert_eq!(out.len(), 960);
    }

    #[test]
    fn single_loss_extrapolates_continuous_finite_samples() {
        let mut plc = Plc::new(1);
        let stream = tone_stream(960, 6, 220.0);
        for f in &stream {
            feed_good(&mut plc, f);
        }

        let mut out = vec![0.0f32; 960];
        plc.lost_frame(&mut [&mut out]);

        assert!(out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        // The boundary sample continues the signal instead of jumping to silence.
        assert!(out[0].abs() > 0.01, "concealment started at near-silence: {}", out[0]);
    }

    #[test]
    fn consecutive_losses_decay_toward_silence() {
        let mut plc = Plc::new(1);
        let stream = tone_stream(960, 6, 300.0);
        for f in &stream {
            feed_good(&mut plc, f);
        }

        let energies: Vec<f32> = (0..(LONG_LOSS_FRAMES + LONG_LOSS_FADE_FRAMES + 2))
            .map(|_| {
                let mut out = vec![0.0f32; 960];
                plc.lost_frame(&mut [&mut out]);
                rms(&out)
            })
            .collect();

        // Early concealment carries energy, the long-burst fade reaches silence.
        assert!(energies[0] > 0.01);
        assert!(*energies.last().unwrap() < 1e-4);
        // The tail is monotonically non-increasing.
        for w in energies.windows(2) {
            assert!(w[1] <= w[0] + 1e-6);
        }
    }

    #[test]
    fn recovery_crossfades_back_to_real_frame() {
        let mut plc = Plc::new(1);
        let stream = tone_stream(960, 6, 180.0);
        for f in &stream {
            feed_good(&mut plc, f);
        }

        let mut lost = vec![0.0f32; 960];
        plc.lost_frame(&mut [&mut lost]);

        // Recover with a deliberately discontinuous real frame.
        let mut recovered = vec![0.9f32; 960];
        plc.good_frame(&mut [&mut recovered]);

        // The first blended sample is pulled away from the raw 0.9 discontinuity.
        assert!(recovered[0].abs() < 0.9);
        // Beyond the ramp the real signal passes through untouched.
        assert_eq!(recovered[959], 0.9);
        assert!(recovered.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }

    #[test]
    fn stereo_channels_are_concealed_independently() {
        let mut plc = Plc::new(2);
        for f in 0..6 {
            let mut left = vec![0.0f32; 960];
            let mut right = vec![0.0f32; 960];
            for i in 0..960 {
                let t = (f * 960 + i) as f32 / 48_000.0;
                left[i] = 0.4 * (2.0 * core::f32::consts::PI * 200.0 * t).sin();
                right[i] = 0.7 * (2.0 * core::f32::consts::PI * 350.0 * t).sin();
            }
            plc.good_frame(&mut [&mut left, &mut right]);
        }

        let mut left = vec![0.0f32; 960];
        let mut right = vec![0.0f32; 960];
        plc.lost_frame(&mut [&mut left, &mut right]);

        assert!(left.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        assert!(right.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        // The louder right channel retains more energy in its concealment.
        assert!(rms(&right) > rms(&left));
    }

    #[test]
    fn different_frame_lengths_are_supported() {
        let mut plc = Plc::new(1);
        let stream = tone_stream(960, 6, 250.0);
        for f in &stream {
            feed_good(&mut plc, f);
        }

        let mut short_out = vec![0.0f32; 120];
        plc.lost_frame(&mut [&mut short_out]);
        assert_eq!(short_out.len(), 120);
        assert!(short_out.iter().all(|s| s.is_finite()));

        let mut long_out = vec![0.0f32; 2880];
        plc.lost_frame(&mut [&mut long_out]);
        assert_eq!(long_out.len(), 2880);
        assert!(long_out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }

    #[test]
    fn reset_clears_all_state() {
        let mut plc = Plc::new(1);
        let stream = tone_stream(960, 6, 220.0);
        for f in &stream {
            feed_good(&mut plc, f);
        }
        assert!(plc.is_primed());
        plc.reset();
        assert!(!plc.is_primed());

        let mut out = vec![0.5f32; 960];
        plc.lost_frame(&mut [&mut out]);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn non_finite_input_cannot_poison_state() {
        let mut plc = Plc::new(1);
        feed_good(&mut plc, &tone_stream(960, 4, 200.0).remove(0));

        let mut poisoned = vec![f32::NAN; 960];
        plc.good_frame(&mut [&mut poisoned]);
        assert!(poisoned.iter().all(|s| s.is_finite()));

        let mut out = vec![0.0f32; 960];
        plc.lost_frame(&mut [&mut out]);
        assert!(out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }

    #[test]
    fn channel_switch_discards_state() {
        let mut plc = Plc::new(1);
        let stream = tone_stream(960, 6, 220.0);
        for f in &stream {
            feed_good(&mut plc, f);
        }
        assert!(plc.is_primed());
        plc.set_channels(2);
        assert!(!plc.is_primed());
        assert_eq!(plc.channels(), 2);
    }

    #[test]
    fn pitch_finds_known_period() {
        // A pure 240 Hz tone at 48 kHz has period 200 samples.
        let history: Vec<f32> = (0..4000)
            .map(|i| (2.0 * core::f32::consts::PI * 240.0 * i as f32 / 48_000.0).sin())
            .collect();
        let lag = estimate_pitch(&history);
        assert!((lag as isize - 200).abs() <= 2, "pitch estimate was {lag}");
    }

    #[test]
    fn lpc_coefficients_are_finite() {
        let history: Vec<f32> = (0..LPC_WINDOW * 2)
            .map(|i| 0.5 * (2.0 * core::f32::consts::PI * 200.0 * i as f32 / 48_000.0).sin())
            .collect();
        let lpc = estimate_lpc(&history);
        assert!(lpc.iter().all(|c| c.is_finite() && c.abs() < 10.0));
    }

    #[test]
    fn state_sizes_are_bounded() {
        // Stereo PLC state must stay within a documented upper bound (256 KiB).
        let plc = Plc::new(2);
        assert!(std::mem::size_of_val(&plc) <= 256 * 1024);
    }

    #[test]
    fn unused_channel_is_isolated() {
        let mut plc = Plc::new(2);
        let stream = tone_stream(960, 6, 220.0);
        for f in &stream {
            let mut left = f.clone();
            let mut right = vec![0.0f32; 960];
            plc.good_frame(&mut [&mut left, &mut right]);
        }
        // Channel 1 observed silence; its RMS must remain zero while channel 0 is active.
        assert!(plc.channel(0).rms > 0.1);
        assert_eq!(plc.channel(1).rms, 0.0);
    }
}
