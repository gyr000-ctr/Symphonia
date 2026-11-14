// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Symphonia Opus Decoder
//!
//! A pure-Rust implementation of the decoder-facing parts of [RFC 6716] Opus together with an
//! optional packet loss concealment (PLC) engine for real-time voice and game-voice streams.
//!
//! ## Packet loss concealment
//!
//! Network loss in interactive audio normally surfaces as gaps or clicks because a missing packet
//! either errors out or is replaced by hard silence. When explicitly enabled, this decoder
//! extrapolates every missing or illegal packet from the pitch, LPC spectral envelope and gain
//! state captured from the most recent good frames, then cross-fades smoothly back to the real
//! signal on recovery.
//!
//! PLC is **off by default** and must be opted into via
//! [`decoder::OpusDecoderOptions::plc`] or [`decoder::OpusDecoder::enable_plc`]. With PLC disabled
//! the concealment engine is never allocated and the default decode path is unchanged sample for
//! sample.
//!
//! See `docs/opus-plc.md` (in the repository root) for the state-machine diagram, bounds and the
//! DSP design notes.
//!
//! [RFC 6716]: https://www.rfc-editor.org/rfc/rfc6716

#![forbid(unsafe_code)]
// The following lints are allowed in all Symphonia crates. See clippy.toml for justification.
#![allow(clippy::comparison_chain)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::collapsible_else_if)]

pub mod decoder;
pub mod packet;
pub mod plc;

pub use decoder::{OpusDecoder, OpusDecoderOptions, State};
pub use packet::{
    Bandwidth, CodingMode, Config, FramePacking, ParsedPacket, Toc, MAX_FRAMES_PER_PACKET,
    MAX_FRAME_SAMPLES, OPUS_SAMPLE_RATE,
};
pub use plc::{Plc, MAX_CHANNELS};
