// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use symphonia_core::audio::{AsGenericAudioBufferRef, Audio, AudioBuffer, GenericAudioBufferRef};
use symphonia_core::codecs::CodecInfo;
use symphonia_core::codecs::audio::{
    AudioCodecId, AudioCodecParameters, AudioDecoder, AudioDecoderOptions, FinalizeResult,
};
use symphonia_core::codecs::registry::{RegisterableAudioDecoder, SupportedAudioCodec};
use symphonia_core::errors::{Result, decode_error, unsupported_error};
use symphonia_core::io::FiniteStream;
use symphonia_core::packet::PacketRef;
use symphonia_core::support_audio_codec;

#[cfg(feature = "mp1")]
use symphonia_core::codecs::audio::well_known::CODEC_ID_MP1;
#[cfg(feature = "mp2")]
use symphonia_core::codecs::audio::well_known::CODEC_ID_MP2;
#[cfg(feature = "mp3")]
use symphonia_core::codecs::audio::well_known::CODEC_ID_MP3;

use super::{common::*, header};

#[cfg(feature = "mp1")]
use crate::layer1;
#[cfg(feature = "mp2")]
use crate::layer2;
#[cfg(feature = "mp3")]
use crate::layer3;

enum State {
    #[cfg(feature = "mp1")]
    Layer1(layer1::Layer1),
    #[cfg(feature = "mp2")]
    Layer2(layer2::Layer2),
    #[cfg(feature = "mp3")]
    Layer3(Box<layer3::Layer3>),
}

impl State {
    fn new(codec: AudioCodecId) -> Self {
        match codec {
            #[cfg(feature = "mp1")]
            CODEC_ID_MP1 => State::Layer1(layer1::Layer1::new()),
            #[cfg(feature = "mp2")]
            CODEC_ID_MP2 => State::Layer2(layer2::Layer2::new()),
            #[cfg(feature = "mp3")]
            CODEC_ID_MP3 => State::Layer3(Box::new(layer3::Layer3::new())),
            _ => unreachable!(),
        }
    }
}

/// MPEG1 and MPEG2 audio layer 1, 2, and 3 decoder.
pub struct MpaDecoder {
    opts: AudioDecoderOptions,
    params: AudioCodecParameters,
    state: State,
    buf: AudioBuffer<f32>,
}

impl MpaDecoder {
    pub fn try_new(params: &AudioCodecParameters, opts: &AudioDecoderOptions) -> Result<Self> {
        // This decoder only supports MP1, MP2, and MP3.
        match params.codec {
            #[cfg(feature = "mp1")]
            CODEC_ID_MP1 => (),
            #[cfg(feature = "mp2")]
            CODEC_ID_MP2 => (),
            #[cfg(feature = "mp3")]
            CODEC_ID_MP3 => (),
            _ => return unsupported_error("mpa: invalid codec"),
        }

        // Create decoder state.
        let state = State::new(params.codec);

        Ok(MpaDecoder { opts: *opts, params: params.clone(), state, buf: Default::default() })
    }

    fn decode_inner(&mut self, packet: &PacketRef<'_>) -> Result<()> {
        let mut reader = packet.as_buf_reader();

        let mut header = header::read_frame_header(&mut reader)?;

        if header.is_free_format() {
            // Free bit-rate frames do not encode their frame size or bit-rate in the header. The
            // demuxer derives the frame size from the next frame's sync word and makes the packet
            // exactly that size. The bit-rate is then derived from the frame size since it is
            // required to select the Layer 1 & 2 quantization tables.
            let frame_size = reader.bytes_available() as usize;

            if frame_size == 0 {
                return decode_error("mpa: empty free bit-rate packet");
            }

            header.frame_size = frame_size;
            header.bitrate =
                header.derive_free_format_bitrate(header::MPEG_HEADER_LEN + frame_size);
        }
        // The packet should be the size stated in the header.
        else if header.frame_size != reader.bytes_available() as usize {
            return decode_error("mpa: invalid packet length");
        }

        // The audio buffer can only be created after the first frame is decoded.
        if self.buf.is_unused() {
            self.buf = AudioBuffer::new(header.spec(), 1152);
        }
        else {
            // Ensure the packet contains an audio frame with the same signal specification as the
            // buffer.
            //
            // TODO: Is it worth it to support changing signal specifications?
            if self.buf.spec() != &header.spec() {
                return decode_error("mpa: invalid audio buffer signal spec for packet");
            }
        }

        // Clear the audio buffer.
        self.buf.clear();

        // Choose the decode step based on the MPEG layer and the current codec ID.
        match &mut self.state {
            #[cfg(feature = "mp1")]
            State::Layer1(layer) if header.layer == MpegLayer::Layer1 => {
                layer.decode(&mut reader, &header, &mut self.buf)?;
            }
            #[cfg(feature = "mp2")]
            State::Layer2(layer) if header.layer == MpegLayer::Layer2 => {
                layer.decode(&mut reader, &header, &mut self.buf)?;
            }
            #[cfg(feature = "mp3")]
            State::Layer3(layer) if header.layer == MpegLayer::Layer3 => {
                layer.decode(&mut reader, &header, &mut self.buf)?;
            }
            _ => return decode_error("mpa: invalid mpeg audio layer"),
        }

        // Trim gaps.
        if self.opts.gapless {
            self.buf.trim(packet.trim_start.get() as usize, packet.trim_end.get() as usize);
        }

        Ok(())
    }
}

impl AudioDecoder for MpaDecoder {
    fn codec_info(&self) -> &CodecInfo {
        // Return the codec that's in-use.
        &Self::supported_codecs()
            .iter()
            .find(|desc| desc.id == self.params.codec)
            .expect("codec registered in supported_codecs")
            .info
    }

    fn codec_params(&self) -> &AudioCodecParameters {
        &self.params
    }

    fn reset(&mut self) {
        // Fully reset the decoder state.
        self.state = State::new(self.params.codec);
    }

    fn decode_ref(&mut self, packet: &PacketRef<'_>) -> Result<GenericAudioBufferRef<'_>> {
        if let Err(e) = self.decode_inner(packet) {
            self.buf.clear();
            Err(e)
        }
        else {
            Ok(self.buf.as_generic_audio_buffer_ref())
        }
    }

    fn finalize(&mut self) -> FinalizeResult {
        Default::default()
    }

    fn last_decoded(&self) -> GenericAudioBufferRef<'_> {
        self.buf.as_generic_audio_buffer_ref()
    }
}

impl RegisterableAudioDecoder for MpaDecoder {
    fn try_registry_new(
        params: &AudioCodecParameters,
        opts: &AudioDecoderOptions,
    ) -> Result<Box<dyn AudioDecoder>>
    where
        Self: Sized,
    {
        Ok(Box::new(MpaDecoder::try_new(params, opts)?))
    }

    fn supported_codecs() -> &'static [SupportedAudioCodec] {
        &[
            #[cfg(feature = "mp1")]
            support_audio_codec!(CODEC_ID_MP1, "mp1", "MPEG Audio Layer 1"),
            #[cfg(feature = "mp2")]
            support_audio_codec!(CODEC_ID_MP2, "mp2", "MPEG Audio Layer 2"),
            #[cfg(feature = "mp3")]
            support_audio_codec!(CODEC_ID_MP3, "mp3", "MPEG Audio Layer 3"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia_core::errors::Error;
    use symphonia_core::packet::Packet;

    fn decode_packet(data: Vec<u8>) -> Result<()> {
        let mut params = AudioCodecParameters::new();
        params.for_codec(CODEC_ID_MP3).with_sample_rate(44100);

        let mut decoder = MpaDecoder::try_new(&params, &AudioDecoderOptions::default())?;

        let packet = Packet::new(0, 0i32.into(), 1152u32.into(), data);

        decoder.decode(&packet).map(|_| ())
    }

    #[test]
    fn test_decode_free_format_packet() {
        // A MPEG1 Layer 3, 44.1 kHz, stereo free bit-rate frame. The packet length determines the
        // frame size, and from that, the bit-rate. The zeroed frame body is not valid audio, but
        // it exercises packet sizing and header derivation without a panic.
        let mut data = vec![0u8; 419];
        data[..4].copy_from_slice(&[0xff, 0xfb, 0x00, 0x00]);

        // The frame is not necessarily decodable, but it must not fail packet sizing.
        match decode_packet(data) {
            Ok(()) => (),
            Err(Error::DecodeError(msg)) => {
                assert_ne!(msg, "mpa: invalid packet length");
                assert_ne!(msg, "mpa: empty free bit-rate packet");
            }
            Err(err) => panic!("unexpected error {err:?}"),
        }
    }

    #[test]
    fn test_decode_empty_free_format_packet() {
        // A free bit-rate header with no body must produce an explicit error rather than being
        // treated as a zero-length frame.
        let data = vec![0xff, 0xfb, 0x00, 0x00];

        match decode_packet(data) {
            Err(Error::DecodeError(msg)) => {
                assert_eq!(msg, "mpa: empty free bit-rate packet")
            }
            other => panic!("expected a decode error, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_cbr_packet_length_mismatch() {
        // A regular CBR frame with a truncated body must still be rejected.
        let mut data = vec![0u8; 100];
        data[..4].copy_from_slice(&[0xff, 0xfb, 0x90, 0x00]);

        match decode_packet(data) {
            Err(Error::DecodeError(msg)) => assert_eq!(msg, "mpa: invalid packet length"),
            other => panic!("expected a decode error, got {other:?}"),
        }
    }
}
