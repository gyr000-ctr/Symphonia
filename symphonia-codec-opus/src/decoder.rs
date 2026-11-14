// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Opus decoder front-end.
//!
//! This module wires the RFC 6716 packet parser ([`crate::packet`]) and the packet loss
//! concealment engine ([`crate::plc`]) into Symphonia's [`AudioDecoder`] trait.
//!
//! ## PLC opt-in
//!
//! Packet loss concealment is **disabled by default** and must be enabled explicitly via
//! [`OpusDecoderOptions::plc`] or [`OpusDecoder::enable_plc`]. With PLC disabled the decoder never
//! constructs the PLC engine and performs no signal analysis: the no-loss path is byte-for-byte
//! identical to a build without PLC wired in.
//!
//! ## Timeline guarantees
//!
//! Every call to [`OpusDecoder::decode_ref`] renders exactly the number of samples declared by the
//! packet's TOC configuration (`num_frames × frame_samples`), even when the packet is illegal.
//! Lost packets are concealed through [`OpusDecoder::decode_loss`] which fills the same duration as
//! a real packet at the caller-supplied configuration. Errors therefore never create or destroy
//! timeline when PLC is enabled; with PLC disabled standard Symphonia error semantics apply and the
//! internal buffer is always cleared on failure.

use symphonia_core::audio::sample::SampleFormat;
use symphonia_core::audio::{
    AsGenericAudioBufferRef, AudioBuffer, AudioMut, AudioSpec, Channels, GenericAudioBufferRef,
    Position,
};
use symphonia_core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia_core::codecs::audio::{
    AudioCodecParameters, AudioDecoder, AudioDecoderOptions, FinalizeResult,
};
use symphonia_core::codecs::registry::{RegisterableAudioDecoder, SupportedAudioCodec};
use symphonia_core::codecs::CodecInfo;
use symphonia_core::errors::{decode_error, Result};
use symphonia_core::packet::PacketRef;
use symphonia_core::support_audio_codec;

use log::warn;

use crate::packet::{self, CodingMode, FramePacking, ParsedPacket, Toc, OPUS_SAMPLE_RATE};
use crate::plc::{Plc, MAX_CHANNELS};

/// Largest number of samples a single Opus packet can produce: 48 frames × 120 ms.
const MAX_PACKET_SAMPLES: usize = 48 * 5_760;

/// Default TOC used when neither a lost nor an invalid packet carries a usable TOC byte:
/// configuration 9 = SILK wideband, 20 ms, mono, packing code 0.
const DEFAULT_TOC_BYTE: u8 = 9 << 3;

/// Decoder options specific to Opus.
#[derive(Copy, Clone, Debug, Default)]
pub struct OpusDecoderOptions {
    /// Common Symphonia decoder options (gapless, verification).
    pub core: AudioDecoderOptions,
    /// Enable packet loss concealment. Defaults to `false`; must be switched on explicitly.
    pub enable_plc: bool,
}

impl OpusDecoderOptions {
    /// Enable or disable PLC. Returns `self` for builder-style construction.
    pub fn plc(mut self, enable: bool) -> Self {
        self.enable_plc = enable;
        self
    }
}

/// Decoder lifecycle states (see the state-machine diagram in `docs/opus-plc.md`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum State {
    /// No packet has been decoded yet; loss produces silence.
    Cold,
    /// Last call consumed a valid packet and predictor state is fresh.
    Running,
    /// One or more consecutive packets were lost or invalid; PLC is extrapolating.
    Concealing,
}

/// Symphonia Opus decoder with optional packet loss concealment.
pub struct OpusDecoder {
    /// Common decoder options (gapless trimming, verification). Retained for the codec core which
    /// consumes them once the full SILK/CELT synthesizer lands.
    #[allow(dead_code)]
    opts: AudioDecoderOptions,
    /// Whether PLC was explicitly enabled.
    plc_enabled: bool,
    /// Codec parameters reported back to the container layer.
    params: AudioCodecParameters,
    /// Current output specification (rate is always 48 kHz; channels follow the stream).
    spec: AudioSpec,
    /// PLC engine, constructed once when concealment is enabled; never allocated otherwise.
    plc: Option<Plc>,
    /// Lifecycle state.
    state: State,
    /// TOC configuration of the most recently accepted packet (for loss-duration matching).
    last_toc: Option<Toc>,
    /// Output PCM buffer, constructed once and reused for every packet.
    buf: AudioBuffer<f32>,
}

impl OpusDecoder {
    /// Construct a decoder from container-supplied codec parameters.
    pub fn try_new(params: &AudioCodecParameters, opts: &OpusDecoderOptions) -> Result<Self> {
        let channels = params
            .channels
            .clone()
            .unwrap_or(Channels::Positioned(Position::FRONT_CENTER));
        let n_channels = channels.count();

        if !(1..=MAX_CHANNELS).contains(&n_channels) {
            return decode_error("opus: unsupported channel count");
        }

        let spec = AudioSpec::new(OPUS_SAMPLE_RATE, channels.clone());
        let buf = AudioBuffer::new(spec.clone(), MAX_PACKET_SAMPLES);
        let plc = if opts.enable_plc { Some(Plc::new(n_channels)) } else { None };

        let mut reported = params.clone();
        reported.sample_rate = Some(OPUS_SAMPLE_RATE);
        reported.sample_format = Some(SampleFormat::F32);

        Ok(Self {
            opts: opts.core,
            plc_enabled: opts.enable_plc,
            params: reported,
            spec,
            plc,
            state: State::Cold,
            last_toc: None,
            buf,
        })
    }

    /// Explicitly enable PLC after construction. Safe to call repeatedly; existing predictor state
    /// is preserved.
    pub fn enable_plc(&mut self) {
        if self.plc.is_none() {
            self.plc = Some(Plc::new(self.spec.channels().count()));
            self.plc_enabled = true;
        }
    }

    /// Explicitly disable PLC and release the concealment state.
    pub fn disable_plc(&mut self) {
        self.plc = None;
        self.plc_enabled = false;
        self.state = State::Cold;
    }

    /// Whether packet loss concealment is currently enabled.
    pub fn is_plc_enabled(&self) -> bool {
        self.plc_enabled
    }

    /// Decode a packet that the transport reports as lost.
    ///
    /// `config_byte` is the TOC byte the lost packet would have used. When the transport cannot
    /// supply it, call [`OpusDecoder::decode_loss_like_last`] instead.
    ///
    /// The output always contains exactly the samples the lost packet would have produced. When
    /// PLC is disabled or no good frame has been seen yet, the output is silence.
    pub fn decode_loss(&mut self, config_byte: u8) -> Result<GenericAudioBufferRef<'_>> {
        let toc = Toc::parse(config_byte);
        self.render_loss(toc)
    }

    /// Conceal a loss whose duration matches the most recently accepted packet. If no packet has
    /// been accepted yet, a 20 ms silence frame is emitted.
    pub fn decode_loss_like_last(&mut self) -> Result<GenericAudioBufferRef<'_>> {
        let toc = self.last_toc.unwrap_or_else(|| Toc::parse(DEFAULT_TOC_BYTE));
        self.render_loss(toc)
    }

    /// Render one concealed packet (single frame per loss call) for `toc` into the output buffer.
    fn render_loss(&mut self, toc: Toc) -> Result<GenericAudioBufferRef<'_>> {
        let total = toc.frame_samples();
        if total == 0 || total > MAX_PACKET_SAMPLES {
            return decode_error("opus: invalid loss duration");
        }

        self.ensure_spec(toc.stereo);
        self.buf.clear();
        self.buf.render_uninit(Some(total));

        if self.plc_enabled {
            if let Some(plc) = self.plc.as_mut() {
                with_frame_planes(&mut self.buf, self.spec.channels().count(), 0, total, |planes| {
                    plc.lost_frame(planes);
                });
                self.state = if plc.is_primed() { State::Concealing } else { State::Cold };
                return Ok(self.buf.as_generic_audio_buffer_ref());
            }
        }

        // PLC disabled or unavailable: silence, but the full duration keeps the timeline.
        self.fill_silence(total);
        self.state = State::Cold;
        Ok(self.buf.as_generic_audio_buffer_ref())
    }

    /// Validate and decode a received packet.
    ///
    /// Illegal packets are handled per the PLC policy:
    ///
    /// * PLC enabled — the packet is discarded and concealed as a loss of matching duration and
    ///   the call returns successfully.
    /// * PLC disabled — the original [`symphonia_core::errors::Error`] is returned and the internal
    ///   buffer is cleared, matching standard Symphonia error semantics.
    fn decode_inner(&mut self, packet: &PacketRef<'_>) -> Result<()> {
        let parsed = match packet::parse_packet(packet.data) {
            Ok(parsed) => parsed,
            Err(e) => {
                warn!("opus: invalid packet: {e:?}");
                if !self.plc_enabled {
                    self.buf.clear();
                    return Err(e);
                }
                let toc = packet
                    .data
                    .first()
                    .map(|b| Toc::parse(*b))
                    .or(self.last_toc)
                    .unwrap_or_else(|| Toc::parse(DEFAULT_TOC_BYTE));
                // render_loss clears and re-renders the buffer itself.
                let _ = self.render_loss(toc)?;
                return Ok(());
            }
        };

        let frame_len = parsed.toc.frame_samples();
        let total = frame_len * parsed.num_frames;
        if total == 0 || total > MAX_PACKET_SAMPLES {
            return decode_error("opus: packet duration out of range");
        }

        let switched = self.ensure_spec(parsed.toc.stereo);

        // Frame-level validation: every declared frame payload must be structurally acceptable.
        for frame in parsed.frames() {
            self.validate_frame(&parsed.toc, frame)?;
        }

        self.buf.clear();
        self.buf.render_uninit(Some(total));

        // Render each frame. Zero-length CELT frames explicitly request PLC.
        for frame_idx in 0..parsed.num_frames {
            let start = frame_idx * frame_len;
            let end = start + frame_len;
            self.render_frame(&parsed, frame_idx, start, end)?;
        }

        self.last_toc = Some(parsed.toc);
        // State transitions are centralized here so a bitstream switch (which resets to Cold in
        // ensure_spec) is not overwritten per-frame.
        if !switched {
            self.state = match self.state {
                // A normal good frame returns the decoder to Running.
                State::Concealing | State::Running => State::Running,
                // The first good frame after construction leaves Cold only after priming; a single
                // non-empty frame is sufficient for the PLC engine, so advance to Running.
                State::Cold => State::Running,
            };
        }
        Ok(())
    }

    /// Validate a single frame payload against mode-specific constraints.
    fn validate_frame(&self, toc: &Toc, frame: &[u8]) -> Result<()> {
        // Zero-length CELT frames encode a PLC-only frame (RFC 6716 §3.2.1) and are legal.
        if frame.is_empty() {
            if toc.info.mode == CodingMode::Celt {
                return Ok(());
            }
            return decode_error("opus: zero-length frame outside CELT mode");
        }
        // SILK-only: the top two bits of the first byte are the payload type, 0..=3 are defined.
        if toc.info.mode == CodingMode::Silk && (frame[0] >> 6) > 3 {
            return decode_error("opus: invalid SILK payload type");
        }
        Ok(())
    }

    /// Render one decoded frame into `buf[start..end]`, feeding the PLC engine when enabled.
    fn render_frame(
        &mut self,
        parsed: &ParsedPacket<'_>,
        frame_idx: usize,
        start: usize,
        end: usize,
    ) -> Result<()> {
        let frame_data = parsed.frame(frame_idx);

        if self.plc_enabled {
            if frame_data.is_empty() {
                // Arrived-but-empty CELT frame: the bitstream itself requests concealment.
                if let Some(plc) = self.plc.as_mut() {
                    with_frame_planes(
                        &mut self.buf,
                        self.spec.channels().count(),
                        start,
                        end,
                        |planes| {
                            plc.lost_frame(planes);
                        },
                    );
                    self.state = State::Concealing;
                }
                return Ok(());
            }

            self.synthesize(frame_data, start, end)?;
            if let Some(plc) = self.plc.as_mut() {
                with_frame_planes(
                    &mut self.buf,
                    self.spec.channels().count(),
                    start,
                    end,
                    |planes| {
                        // good_frame applies any recovery cross-fade in place, then absorbs the
                        // signal into the predictor state.
                        plc.good_frame(planes);
                    },
                );
            }
        }
        else {
            self.synthesize(frame_data, start, end)?;
        }
        Ok(())
    }

    /// Frame synthesizer.
    ///
    /// A full SILK/CELT bitstream decoder is upstream work-in-progress; until that lands, every
    /// valid packet is rendered as the zero-energy signal. This keeps the function total,
    /// allocation-free and deterministic, and leaves the PLC engine as the single source of
    /// extrapolated audio. When the real codec core lands it replaces only this function's body;
    /// parser, state machine and PLC integration stay unchanged.
    fn synthesize(&mut self, frame: &[u8], start: usize, end: usize) -> Result<()> {
        debug_assert!(!frame.is_empty());
        let n_channels = self.spec.channels().count();
        for ch in 0..n_channels {
            if let Some(plane) = self.buf.plane_mut(ch) {
                plane[start..end].fill(0.0);
            }
        }
        Ok(())
    }

    /// Reconfigure the output specification when the stream signals a channel change. A channel
    /// switch is a bitstream discontinuity: all PLC predictor state is discarded. Returns `true`
    /// when a switch actually occurred.
    fn ensure_spec(&mut self, stereo: bool) -> bool {
        let wanted = if stereo {
            Channels::Positioned(Position::FRONT_LEFT | Position::FRONT_RIGHT)
        }
        else {
            Channels::Positioned(Position::FRONT_CENTER)
        };
        if wanted != *self.spec.channels() {
            let new_spec = AudioSpec::new(OPUS_SAMPLE_RATE, wanted.clone());
            self.buf = AudioBuffer::new(new_spec.clone(), MAX_PACKET_SAMPLES);
            self.spec = new_spec;
            if let Some(plc) = self.plc.as_mut() {
                plc.set_channels(wanted.count());
            }
            self.state = State::Cold;
            self.last_toc = None;
            self.params.channels = Some(wanted);
            true
        }
        else {
            false
        }
    }

    fn fill_silence(&mut self, total: usize) {
        let n_channels = self.spec.channels().count();
        for ch in 0..n_channels {
            if let Some(plane) = self.buf.plane_mut(ch) {
                plane[..total].fill(0.0);
            }
        }
    }

    /// Current lifecycle state (diagnostics and tests).
    pub fn state(&self) -> State {
        self.state
    }
}

/// Borrow `[start, end)` of every active plane and invoke `f` with the PLC-compatible slice.
///
/// Opus has at most [`MAX_CHANNELS`] channels, so the reference array is fixed-size and the whole
/// helper is allocation-free. The stereo case uses [`AudioMut::plane_pair_mut`] to obtain two
/// disjoint plane borrows through one call.
fn with_frame_planes<R>(
    buf: &mut AudioBuffer<f32>,
    count: usize,
    start: usize,
    end: usize,
    f: impl FnOnce(&mut [&mut [f32]]) -> R,
) -> R {
    if count == 1 {
        let plane = buf
            .plane_mut(0)
            .expect("plane 0 exists for a configured decoder");
        let region = &mut plane[start..end];
        let mut refs: [&mut [f32]; MAX_CHANNELS] = [region, &mut []];
        f(&mut refs[..1])
    }
    else {
        let (p0, p1) = buf
            .plane_pair_mut(0, 1)
            .expect("both stereo planes exist for a configured decoder");
        let mut refs: [&mut [f32]; MAX_CHANNELS] =
            [&mut p0[start..end], &mut p1[start..end]];
        f(&mut refs[..count])
    }
}

impl AudioDecoder for OpusDecoder {
    fn reset(&mut self) {
        if let Some(plc) = self.plc.as_mut() {
            plc.reset();
        }
        self.buf.clear();
        self.state = State::Cold;
        self.last_toc = None;
    }

    fn codec_info(&self) -> &CodecInfo {
        &Self::supported_codecs()[0].info
    }

    fn codec_params(&self) -> &AudioCodecParameters {
        &self.params
    }

    fn decode_ref(&mut self, packet: &PacketRef<'_>) -> Result<GenericAudioBufferRef<'_>> {
        match self.decode_inner(packet) {
            Ok(()) => Ok(self.buf.as_generic_audio_buffer_ref()) ,
            Err(e) => {
                // Implementors must clear the internal buffer on error.
                self.buf.clear();
                Err(e)
            }
        }
    }

    fn finalize(&mut self) -> FinalizeResult {
        FinalizeResult::default()
    }

    fn last_decoded(&self) -> GenericAudioBufferRef<'_> {
        self.buf.as_generic_audio_buffer_ref()
    }
}

impl RegisterableAudioDecoder for OpusDecoder {
    fn try_registry_new(
        params: &AudioCodecParameters,
        opts: &AudioDecoderOptions,
    ) -> Result<Box<dyn AudioDecoder>> {
        // Registry construction has no channel for Opus-specific options, so PLC stays off on this
        // path. Applications instantiate OpusDecoder::try_new directly to opt in.
        let opus_opts = OpusDecoderOptions { core: *opts, enable_plc: false };
        Ok(Box::new(OpusDecoder::try_new(params, &opus_opts)?))
    }

    fn supported_codecs() -> &'static [SupportedAudioCodec] {
        &[support_audio_codec!(CODEC_ID_OPUS, "opus", "Opus")]
    }
}

#[allow(dead_code)]
fn _assert_packing_exhaustive(p: FramePacking) -> &'static str {
    match p {
        FramePacking::One => "one",
        FramePacking::TwoEqual => "two-equal",
        FramePacking::TwoVbr => "two-vbr",
        FramePacking::Arbitrary => "arbitrary",
    }
}
