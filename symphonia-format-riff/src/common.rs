// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// `PacketInfo` helps to simulate packetization over a number of blocks of data.
/// In case the codec is blockless the block size equals one full audio frame in bytes.
use std::marker::PhantomData;
use std::num::NonZero;

use symphonia_core::audio::Channels;
use symphonia_core::codecs::audio::well_known::{
    CODEC_ID_ADPCM_IMA_WAV, CODEC_ID_ADPCM_MS, CODEC_ID_PCM_F32BE, CODEC_ID_PCM_F32LE,
    CODEC_ID_PCM_F64BE, CODEC_ID_PCM_F64LE,
};
use symphonia_core::codecs::audio::{AudioCodecId, AudioCodecParameters};
use symphonia_core::errors::{Error, Result, decode_error};
use symphonia_core::formats::prelude::*;
use symphonia_core::io::{MediaSourceStream, ReadBytes};

use log::{debug, info};

pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// The on-disk layout of a RIFF-style chunk tree.
///
/// `Riff` is the classic RIFF/RF64 layout: an 8-byte chunk header (4-byte tag, 4-byte little-endian
/// length), chunks padded to a 2-byte boundary. `Wave64` is the Sony Wave64 layout: a 24-byte
/// chunk header (16-byte GUID, 8-byte little-endian length that includes the whole header),
/// chunks padded to an 8-byte boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RiffLayout {
    /// Classic RIFF (and RF64) layout, 8-byte chunk headers, 2-byte chunk alignment.
    Riff,
    /// Sony Wave64 layout, 24-byte (GUID + u64) chunk headers, 8-byte chunk alignment.
    Wave64,
}

impl RiffLayout {
    /// The size of a chunk header in this layout.
    pub fn header_len(&self) -> u64 {
        match self {
            RiffLayout::Riff => 8,
            RiffLayout::Wave64 => 24,
        }
    }

    /// The boundary, in bytes, to which chunk bodies are padded.
    pub fn alignment(&self) -> u64 {
        match self {
            RiffLayout::Riff => 2,
            RiffLayout::Wave64 => 8,
        }
    }
}

/// The maximum number of frames that will be in a packet.
/// Since there are no real packets in AIFF, this is arbitrary, used same value as MP3.
const MAX_FRAMES_PER_PACKET: NonZero<u64> = NonZero::new(1152).unwrap();

/// `ParseChunkTag` implements `parse_tag` to map between the chunk identifier and the enumeration.
///
/// For classic RIFF/RF64 the identifier is the chunk's 4-byte tag. For Sony Wave64 the 16-byte
/// chunk GUID is translated to the equivalent 4-byte tag by the `ChunksReader` before this
/// function is called.
pub trait ParseChunkTag: Sized {
    fn parse_tag(tag: [u8; 4], len: u64) -> Option<Self>;
}

pub enum NullChunks {}

impl ParseChunkTag for NullChunks {
    fn parse_tag(_tag: [u8; 4], _len: u64) -> Option<Self> {
        None
    }
}

/// Resolves the on-disk length of a chunk to its true length.
///
/// This is used by RF64, where chunks that would overflow a 32-bit length field store the
/// placeholder value `0xFFFFFFFF` and record their true 64-bit length in the `ds64` chunk. Sony
/// Wave64 stores 64-bit lengths directly and does not need a resolver.
pub trait ChunkLenResolver {
    /// Resolve the true length of the chunk identified by `tag`.
    ///
    /// `on_disk_len` is the length stored in the chunk header. The returned length excludes the
    /// chunk header itself.
    fn resolve(&mut self, tag: [u8; 4], on_disk_len: u64) -> Result<u64>;
}

/// A `ChunkLenResolver` that accepts every on-disk length unchanged. Used by classic RIFF, AIFF,
/// and Sony Wave64.
pub struct IdentityResolver;

impl ChunkLenResolver for IdentityResolver {
    fn resolve(&mut self, _tag: [u8; 4], on_disk_len: u64) -> Result<u64> {
        Ok(on_disk_len)
    }
}

/// `ChunksReader` reads chunks from a `ByteStream`. It is generic across a type, usually an enum,
/// implementing the `ParseChunkTag` trait. When a new chunk is encountered in the stream,
/// `parse_tag` on T is called to return an object capable of parsing/reading that chunk or `None`.
/// This makes reading the actual chunk data lazy in that the  chunk is not read until the object
/// is consumed.
pub struct ChunksReader<T: ParseChunkTag> {
    len: Option<u64>,
    byte_order: ByteOrder,
    layout: RiffLayout,
    consumed: u64,
    phantom: PhantomData<T>,
}

impl<T: ParseChunkTag> ChunksReader<T> {
    /// Instantiate a `ChunksReader` using the classic RIFF layout. `len` is the length of the
    /// parent chunk's body, excluding its header, or `None` if the parent chunk is open-ended
    /// (streamed).
    pub fn new(len: Option<u32>, byte_order: ByteOrder) -> Self {
        ChunksReader {
            len: len.map(u64::from),
            byte_order,
            layout: RiffLayout::Riff,
            consumed: 0,
            phantom: PhantomData,
        }
    }

    /// Instantiate a `ChunksReader` using the classic RIFF layout with a 64-bit parent length.
    pub fn new_riff(len: Option<u64>, byte_order: ByteOrder) -> Self {
        ChunksReader {
            len,
            byte_order,
            layout: RiffLayout::Riff,
            consumed: 0,
            phantom: PhantomData,
        }
    }

    /// Instantiate a `ChunksReader` for a Sony Wave64 chunk tree. `len` is the length of the
    /// parent chunk's body, excluding the parent's own header, or `None` if the parent is
    /// open-ended.
    pub fn new_wave64(len: Option<u64>) -> Self {
        ChunksReader {
            len,
            byte_order: ByteOrder::LittleEndian,
            layout: RiffLayout::Wave64,
            consumed: 0,
            phantom: PhantomData,
        }
    }

    /// Reads the chunk header and returns the translated 4-byte tag and the on-disk length stored
    /// in the header (excluding the header itself for both layouts).
    fn read_header<B: ReadBytes>(&self, reader: &mut B) -> Result<([u8; 4], u64)> {
        match self.layout {
            RiffLayout::Riff => {
                let tag = reader.read_quad_bytes()?;

                let chunk_len = match self.byte_order {
                    ByteOrder::LittleEndian => reader.read_u32()?,
                    ByteOrder::BigEndian => reader.read_be_u32()?,
                };

                Ok((tag, u64::from(chunk_len)))
            }
            RiffLayout::Wave64 => {
                // A Wave64 chunk header is a 16-byte GUID followed by a 64-bit size that includes
                // the whole 24-byte header (the 16-byte GUID plus this 8-byte size field).
                let guid = reader.read_quad_bytes()?;
                let mut guid_rest = [0u8; 12];
                reader.read_buf_exact(&mut guid_rest)?;

                let size_field = reader.read_u64()?;

                if size_field < 24 {
                    return decode_error("wav: wave64 chunk size is smaller than its header");
                }

                let chunk_len = size_field - 24;

                // Translate well-known Wave64 GUIDs to their equivalent 4-byte RIFF tags. All
                // Wave64 GUIDs share a common 12-byte suffix; only the first 4 bytes differ.
                if guid_rest == WAVE64_GUID_SUFFIX {
                    Ok((guid, chunk_len))
                }
                else {
                    info!("ignoring unknown wave64 chunk: guid={:02x?}, len={}.", guid, chunk_len);
                    Ok((*b"????", chunk_len))
                }
            }
        }
    }

    pub fn next<B: ReadBytes, R: ChunkLenResolver>(
        &mut self,
        reader: &mut B,
        resolver: &mut R,
    ) -> Result<Option<T>> {
        // Loop until a chunk is recognized and returned, or the end of stream is reached.
        loop {
            // Check if at the end of the parent chunk.
            if let Some(len) = self.len {
                if self.consumed >= len {
                    return Ok(None);
                }
            }

            // Align to the layout's chunk boundary if not currently aligned.
            let alignment = self.layout.alignment();

            if self.consumed % alignment != 0 {
                let pad = alignment - (self.consumed % alignment);
                reader.ignore_bytes(pad)?;
                self.consumed += pad;
            }

            // Check if there are enough bytes to read a chunk header. If not, there are no more
            // chunks to be read.
            let header_len = self.layout.header_len();

            if let Some(len) = self.len {
                if self.consumed + header_len > len {
                    return Ok(None);
                }
            }

            // Read chunk tag and length (the chunk header).
            let (tag, on_disk_len) = self.read_header(reader)?;

            self.consumed += header_len;

            // RF64 chunks that overflow 32-bits store a placeholder length that the resolver
            // replaces with the true 64-bit length. For every other layout the resolver is an
            // identity.
            let chunk_len = resolver.resolve(tag, on_disk_len)?;

            // Check if the ChunkReader has enough unread bytes to fully read the chunk body.
            if let Some(len) = self.len {
                // Warning: The formulation of this conditional is critical because chunk_len is an
                // untrusted input, it may overflow when if added to anything.
                if len - self.consumed < chunk_len {
                    debug!(
                        "chunk length of {} exceeds parent (list) chunk length",
                        String::from_utf8_lossy(&tag)
                    );
                    return decode_error("riff: chunk length exceeds parent (list) chunk length");
                }
            }

            // The length of the chunk has been validated, so "consume" the chunk, including its
            // trailing alignment padding.
            let padded_len = chunk_len + ((alignment - (chunk_len % alignment)) % alignment);
            self.consumed = self.consumed.saturating_add(padded_len);

            match T::parse_tag(tag, chunk_len) {
                Some(chunk) => return Ok(Some(chunk)),
                None => {
                    // As per the RIFF spec, unknown chunks are to be ignored. Skip the chunk body
                    // and any trailing alignment padding.
                    info!(
                        "ignoring unknown chunk: tag={}, len={}.",
                        String::from_utf8_lossy(&tag),
                        chunk_len
                    );

                    reader.ignore_bytes(padded_len)?;
                }
            }
        }
    }

    pub fn finish<B: ReadBytes>(&mut self, reader: &mut B) -> Result<()> {
        // If data is remaining in the parent chunk, skip it.
        if let Some(parent_len) = self.len {
            if self.consumed < parent_len {
                let remaining = parent_len - self.consumed;
                reader.ignore_bytes(remaining)?;
                self.consumed += remaining;
            }

            // Pad the chunk to the next 2-byte boundary.
            if parent_len & 0x1 == 1 {
                reader.read_u8()?;
            }
        }

        Ok(())
    }
}

/// The common 12-byte suffix shared by all Sony Wave64 chunk GUIDs, in on-disk byte order.
#[rustfmt::skip]
pub const WAVE64_GUID_SUFFIX: [u8; 12] = [
    0x2c, 0xf3, 0x94, 0x46,
    0x89, 0x9f, 0x29, 0x55,
    0x8b, 0x5e, 0xd5, 0x46,
];

/// Converts a 64-bit chunk length to a 32-bit one for chunks whose body is still constrained to
/// 32-bit lengths. A length exceeding `u32::MAX` is a malformed stream because such chunks must be
/// carried in an RF64 `ds64`/data chunk or a Wave64 data chunk, neither of which use this helper.
pub fn chunk_len_u32(len: u64) -> Result<u32> {
    u32::try_from(len).map_err(|_| Error::DecodeError("riff: chunk length exceeds 32-bit limit"))
}

/// Common trait implemented for all chunks that are parsed by a `ChunkParser`.
pub trait ParseChunk: Sized {
    /// Parse the chunk.
    fn parse<B: ReadBytes>(reader: &mut B, tag: [u8; 4], len: u64) -> Result<Self>;

    /// Parse the chunk and discard any unread data in the chunk.
    fn parse_and_skip_unread<B: ReadBytes>(reader: &mut B, tag: [u8; 4], len: u64) -> Result<Self> {
        let pos = reader.pos();
        let result = Self::parse(reader, tag, len);

        // Only discard unread data if parsing was successful.
        if result.is_ok() {
            let bytes_read = reader.pos() - pos;
            if bytes_read < len {
                reader.ignore_bytes(len - bytes_read)?;
            }
        }

        result
    }
}

/// `ChunkParser` is a utility struct for unifying the parsing of chunks.
pub struct ChunkParser<P: ParseChunk> {
    tag: [u8; 4],
    pub len: u64,
    phantom: PhantomData<P>,
}

impl<P: ParseChunk> ChunkParser<P> {
    pub fn new(tag: [u8; 4], len: u64) -> Self {
        ChunkParser { tag, len, phantom: PhantomData }
    }

    pub fn parse<B: ReadBytes>(&self, reader: &mut B) -> Result<P> {
        P::parse(reader, self.tag, self.len)
    }

    pub fn parse_and_skip_unread<B: ReadBytes>(&self, reader: &mut B) -> Result<P> {
        P::parse_and_skip_unread(reader, self.tag, self.len)
    }
}

pub enum FormatData {
    Pcm(FormatPcm),
    Adpcm(FormatAdpcm),
    IeeeFloat(FormatIeeeFloat),
    Extensible(FormatExtensible),
    ALaw(FormatALaw),
    MuLaw(FormatMuLaw),
}

pub struct FormatPcm {
    /// The number of bits per sample as written.
    pub bits_per_sample: u16,
    /// The number of bits per sample that are valid. Must be <= `bits_per_sample`.
    pub valid_bits_per_sample: u16,
    /// Channel bitmask.
    pub channels: Channels,
    /// Codec ID.
    pub codec: AudioCodecId,
}

impl FormatPcm {
    pub fn make_packet_info(&self) -> Result<PacketInfo> {
        let num_channels = self.channels.count() as u16;
        let frame_len = (self.bits_per_sample / 8) * num_channels;
        PacketInfo::without_blocks(u32::from(frame_len))
    }
}

pub struct FormatAdpcm {
    pub block_align: u16,
    /// The number of bits per sample. At the moment only 4bit is supported.
    pub bits_per_sample: u16,
    /// Channel bitmask.
    pub channels: Channels,
    /// Codec ID.
    pub codec: AudioCodecId,
}

impl FormatAdpcm {
    pub fn make_packet_info(&self) -> Result<PacketInfo> {
        let num_channels = self.channels.count() as u16;
        let bits_per_frame = (self.bits_per_sample * num_channels) as u64;

        match self.codec {
            CODEC_ID_ADPCM_MS => {
                // Don't allow underflow to occur.
                if 7 * num_channels > self.block_align {
                    return decode_error("riff (adpcm_ms): block align too small");
                }
                let frames_per_block =
                    (8 * u64::from(self.block_align - 7 * num_channels)) / bits_per_frame + 2;
                PacketInfo::with_blocks(self.block_align, frames_per_block)
            }
            CODEC_ID_ADPCM_IMA_WAV => {
                // Don't allow underflow to occur.
                if 4 * num_channels > self.block_align {
                    return decode_error("riff (adpcm_ima_wav): block align too small");
                }
                let frames_per_block =
                    (8 * u64::from(self.block_align - 4 * num_channels)) / bits_per_frame + 1;
                PacketInfo::with_blocks(self.block_align, frames_per_block)
            }
            _ => unimplemented!("incomplete new adpcm codec"),
        }
    }
}

pub struct FormatIeeeFloat {
    /// Channel bitmask.
    pub channels: Channels,
    /// Codec ID.
    pub codec: AudioCodecId,
}

impl FormatIeeeFloat {
    pub fn make_packet_info(&self) -> Result<PacketInfo> {
        let num_channels = self.channels.count() as u32;
        let bytes_per_sample = match self.codec {
            CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE => 4,
            CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE => 8,
            _ => unreachable!(),
        };
        let frame_len = num_channels * bytes_per_sample;
        PacketInfo::without_blocks(frame_len)
    }
}

pub struct FormatExtensible {
    /// The number of bits per sample as written. This value is always a multiple of 8-bits.
    pub bits_per_sample: u16,
    /// The number of bits per sample that are valid. This number is always less than the number
    /// of bits per sample.
    pub valid_bits_per_sample: u16,
    /// Channel bitmask.
    pub channels: Channels,
    /// Globally unique identifier of the format.
    pub sub_format_guid: [u8; 16],
    /// Codec ID.
    pub codec: AudioCodecId,
}

impl FormatExtensible {
    pub fn make_packet_info(&self) -> Result<PacketInfo> {
        // NOTE: This is only valid for compressed sub-types.
        let num_channels = self.channels.count() as u16;
        let frame_len = (self.bits_per_sample / 8) * num_channels;
        PacketInfo::without_blocks(u32::from(frame_len))
    }
}

pub struct FormatALaw {
    /// Channel bitmask.
    pub channels: Channels,
    /// Codec ID.
    pub codec: AudioCodecId,
}

impl FormatALaw {
    pub fn make_packet_info(&self) -> Result<PacketInfo> {
        let num_channels = self.channels.count() as u32;
        PacketInfo::without_blocks(num_channels)
    }
}

pub struct FormatMuLaw {
    /// Channel bitmask.
    pub channels: Channels,
    /// Codec ID.
    pub codec: AudioCodecId,
}

impl FormatMuLaw {
    pub fn make_packet_info(&self) -> Result<PacketInfo> {
        let num_channels = self.channels.count() as u32;
        PacketInfo::without_blocks(num_channels)
    }
}

pub struct PacketInfo {
    pub block_size: NonZero<u64>,
    pub frames_per_block: NonZero<u64>,
    pub max_blocks_per_packet: NonZero<u64>,
    pub max_frames_per_packet: NonZero<u64>,
}

impl PacketInfo {
    pub fn with_blocks(block_size: u16, frames_per_block: u64) -> Result<Self> {
        // Frames/block must be non-zero.
        let frames_per_block = match NonZero::new(frames_per_block) {
            Some(value) => value,
            _ => return decode_error("riff: frames per block is 0"),
        };

        // Block size must be non-zero.
        let block_size = match NonZero::new(u64::from(block_size)) {
            Some(value) => value,
            None => return decode_error("riff: block size is 0"),
        };

        // Atleast one block must be present per packet.
        let max_blocks_per_packet = match NonZero::new(
            frames_per_block.max(MAX_FRAMES_PER_PACKET).get() / frames_per_block.get(),
        ) {
            Some(value) => value,
            _ => return decode_error("riff: max blocks per packet is 0"),
        };

        // Pre-compute maximum frames per packet.
        let max_frames_per_packet = match max_blocks_per_packet.checked_mul(frames_per_block) {
            Some(value) => value,
            _ => return decode_error("riff: max frames per packet overflow"),
        };

        Ok(Self { block_size, frames_per_block, max_blocks_per_packet, max_frames_per_packet })
    }

    pub fn without_blocks(frame_len: u32) -> Result<Self> {
        // Block size must be non-zero.
        let block_size = match NonZero::new(u64::from(frame_len)) {
            Some(value) => value,
            None => return decode_error("riff: block size is 0"),
        };

        Ok(Self {
            block_size,
            frames_per_block: NonZero::new(1).expect("1 is non-zero"),
            max_blocks_per_packet: MAX_FRAMES_PER_PACKET,
            max_frames_per_packet: MAX_FRAMES_PER_PACKET,
        })
    }

    pub fn get_num_frames_at(&self, data_pos: u64) -> u64 {
        (data_pos / self.block_size) * self.frames_per_block.get()
    }

    pub fn get_actual_ts(&self, ts: Timestamp) -> Timestamp {
        let max_frames_per_packet = self.max_frames_per_packet.get();
        ts.align_towards_zero(Duration::from(max_frames_per_packet))
            .expect("max_frames_per_packet is non-zero")
    }
}

pub fn next_packet(
    reader: &mut MediaSourceStream<'_>,
    packet_info: &PacketInfo,
    tracks: &[Track],
    data_start_pos: u64,
    data_end_pos: Option<u64>,
) -> Result<Option<Packet>> {
    let pos = reader.pos();
    if tracks.is_empty() {
        return decode_error("riff: no tracks");
    }

    let bounded_data_end = data_end_pos.unwrap_or(u64::MAX);

    // Determine the number of complete blocks remaining in the data chunk.
    let num_blocks_left =
        if pos < bounded_data_end { (bounded_data_end - pos) / packet_info.block_size } else { 0 };

    if num_blocks_left == 0 {
        return Ok(None);
    }

    let blocks_per_packet = num_blocks_left.min(packet_info.max_blocks_per_packet.get());

    // The packet timestamp is the position of the first byte of the first frame in the
    // packet relative to the start of the data chunk divided by the length per frame.
    let pts = match packet_info.get_num_frames_at(pos - data_start_pos).try_into() {
        Ok(pts) => pts,
        Err(_) => return Ok(None),
    };

    let dur = Duration::from(blocks_per_packet * packet_info.frames_per_block.get());
    let pkt_len = blocks_per_packet * packet_info.block_size.get();

    // Copy the frames.
    let packet_buf = reader.read_boxed_slice(pkt_len as usize)?;

    // `read_boxed_slice` is not exact: a short read returns a truncated buffer instead of an
    // error. When the data chunk declares a length, a short read before the declared end means
    // the underlying file was truncated, which must be reported. Open-ended (streamed) data
    // chunks have no declared end, so a short read is their natural end-of-stream.
    if data_end_pos.is_some() && packet_buf.len() < pkt_len as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "riff: data chunk is shorter than its declared length",
        )
        .into());
    }

    Ok(Some(Packet::new(0, pts, dur, packet_buf)))
}

/// TODO: format here refers to format chunk in Wave terminology, but the data being handled here is
/// generic - find a better name, or combine with append_data_params
pub fn append_format_params(
    codec_params: &mut AudioCodecParameters,
    format_data: FormatData,
    sample_rate: u32,
) {
    codec_params.with_sample_rate(sample_rate);

    match format_data {
        FormatData::Pcm(pcm) => {
            codec_params
                .for_codec(pcm.codec)
                .with_bits_per_coded_sample(u32::from(pcm.valid_bits_per_sample))
                .with_bits_per_sample(u32::from(pcm.valid_bits_per_sample))
                .with_channels(pcm.channels);
        }
        FormatData::Adpcm(adpcm) => {
            codec_params.for_codec(adpcm.codec).with_channels(adpcm.channels);
        }
        FormatData::IeeeFloat(ieee) => {
            codec_params.for_codec(ieee.codec).with_channels(ieee.channels);
        }
        FormatData::Extensible(ext) => {
            codec_params
                .for_codec(ext.codec)
                .with_bits_per_coded_sample(u32::from(ext.valid_bits_per_sample))
                .with_bits_per_sample(u32::from(ext.valid_bits_per_sample))
                .with_channels(ext.channels);
        }
        FormatData::ALaw(alaw) => {
            codec_params.for_codec(alaw.codec).with_channels(alaw.channels);
        }
        FormatData::MuLaw(mulaw) => {
            codec_params.for_codec(mulaw.codec).with_channels(mulaw.channels);
        }
    }
}

/// TODO: format here refers to format chunk in Wave terminology, but the data being handled here is
/// generic - find a better name, or combine with append_data_params append_format_params
pub fn append_data_params(track: &mut Track, data_len: u64, packet_info: &PacketInfo) {
    // Prefer the duration in the FACT chunk.
    let num_frames = packet_info.get_num_frames_at(data_len);
    track.with_num_frames(num_frames);
    // Duration equals the number of frames because the timebase is always 1 / sample rate.
    track.with_duration(Duration::from(num_frames));
}
