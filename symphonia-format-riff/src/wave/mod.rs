// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::io::{Seek, SeekFrom};

use symphonia_core::codecs::CodecParameters;
use symphonia_core::codecs::audio::AudioCodecParameters;
use symphonia_core::errors::{Error, Result, SeekErrorKind};
use symphonia_core::errors::{decode_error, seek_error, unsupported_error};
use symphonia_core::formats::prelude::*;
use symphonia_core::formats::probe::{ProbeFormatData, ProbeableFormat, Score, Scoreable};
use symphonia_core::formats::well_known::FORMAT_ID_WAVE;
use symphonia_core::io::*;
use symphonia_core::meta::well_known::METADATA_ID_WAVE;
use symphonia_core::meta::{Metadata, MetadataInfo, MetadataLog};
use symphonia_core::support_format;

use log::{debug, error, info};

use crate::common::{
    ByteOrder, ChunkLenResolver, ChunksReader, IdentityResolver, PacketInfo, WAVE64_GUID_SUFFIX,
    append_data_params, append_format_params, next_packet,
};
mod chunks;
use chunks::*;

/// WAVE is actually a RIFF stream, with a "RIFF" ASCII stream marker.
const WAVE_STREAM_MARKER: [u8; 4] = *b"RIFF";
/// RF64 is a 64-bit extension of RIFF/WAVE, with an "RF64" ASCII stream marker.
const RF64_STREAM_MARKER: [u8; 4] = *b"RF64";
/// BW64 is the EBU Broadcast Wave 2 variant of RF64, with a "BW64" ASCII stream marker.
const BW64_STREAM_MARKER: [u8; 4] = *b"BW64";
/// A Sony Wave64 stream begins with the first 4 bytes of the RIFF GUID (the remaining 12 bytes are
/// the shared Wave64 GUID suffix).
const WAVE64_STREAM_MARKER: [u8; 4] = *b"riff";
/// A possible RIFF form is "wave".
const WAVE_RIFF_FORM: [u8; 4] = *b"WAVE";

/// Identifies the on-disk container variant of a WAVE stream.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WaveVariant {
    /// Classic 32-bit RIFF WAVE.
    Riff,
    /// 64-bit RF64 (or BW64) WAVE with a `ds64` chunk.
    Rf64,
    /// Sony Wave64 with 64-bit GUID-based chunk headers.
    Wave64,
}

const WAVE_FORMAT_INFO: FormatInfo = FormatInfo {
    format: FORMAT_ID_WAVE,
    short_name: "wave",
    long_name: "Waveform Audio File Format",
};

const WAVE_METADATA_INFO: MetadataInfo = MetadataInfo {
    metadata: METADATA_ID_WAVE,
    short_name: "wave",
    long_name: "Waveform Audio File Format",
};

/// Waveform Audio File Format (WAV) format reader.
///
/// `WavReader` implements a demuxer for the WAVE container format.
pub struct WavReader<'s> {
    reader: MediaSourceStream<'s>,
    media_info: MediaInfo,
    tracks: Vec<Track>,
    chapters: Option<ChapterGroup>,
    metadata: MetadataLog,
    packet_info: PacketInfo,
    data_start_pos: u64,
    data_end_pos: Option<u64>,
}

impl<'s> WavReader<'s> {
    pub fn try_new(mut mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        // A Wave file is one large RIFF chunk, with the actual meta and audio data contained in
        // nested chunks. Therefore, the file starts with a top-level chunk header identifying the
        // container variant and its size.
        let marker = mss.read_quad_bytes()?;

        let variant = if marker == WAVE_STREAM_MARKER {
            WaveVariant::Riff
        }
        else if marker == RF64_STREAM_MARKER || marker == BW64_STREAM_MARKER {
            WaveVariant::Rf64
        }
        else if marker == WAVE64_STREAM_MARKER && Self::is_wave64_header(&mut mss)? {
            WaveVariant::Wave64
        }
        else {
            return unsupported_error("wav: missing wave riff stream marker");
        };

        match variant {
            WaveVariant::Riff => Self::try_new_riff(mss, opts),
            WaveVariant::Rf64 => Self::try_new_rf64(mss, opts),
            WaveVariant::Wave64 => Self::try_new_wave64(mss, opts),
        }
    }

    /// Verifies that the 12 bytes following a lowercase "riff" marker complete the RIFF GUID used
    /// by Sony Wave64.
    fn is_wave64_header(mss: &mut MediaSourceStream<'_>) -> Result<bool> {
        let mut suffix = [0u8; 12];
        mss.read_buf_exact(&mut suffix)?;
        Ok(suffix == WAVE64_GUID_SUFFIX)
    }

    /// Reads a classic 32-bit RIFF/WAVE header (the 4-byte size and 4-byte form type follow the
    /// already-consumed "RIFF" marker) and returns the parent body length, if known.
    fn read_riff_header(mss: &mut MediaSourceStream<'_>) -> Result<Option<u64>> {
        let riff_len = mss.read_u32()?;

        if riff_len < 4 {
            return decode_error("wav: invalid riff length");
        }

        Self::read_wave_form(mss)?;

        // When ffmpeg encodes wave to stdout the riff (parent) and data (child) chunk lengths are
        // (2^32)-1 since the size is not known ahead of time.
        Ok(if riff_len < u32::MAX { Some(u64::from(riff_len) - 4) } else { None })
    }

    /// Reads and validates the 4-byte form type that immediately follows a RIFF/RF64 top-level
    /// size field.
    fn read_wave_form(mss: &mut MediaSourceStream<'_>) -> Result<()> {
        let riff_form = mss.read_quad_bytes()?;

        if riff_form != WAVE_RIFF_FORM {
            error!("riff form is not wave ({})", String::from_utf8_lossy(&riff_form));

            return unsupported_error("wav: riff form is not wave");
        }

        Ok(())
    }

    /// Constructs a reader for a classic 32-bit RIFF/WAVE stream. The parsing path is identical to
    /// the legacy implementation.
    fn try_new_riff(mut mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        let riff_data_len = Self::read_riff_header(&mut mss)?;

        let mut riff_chunks =
            ChunksReader::<RiffWaveChunks>::new_riff(riff_data_len, ByteOrder::LittleEndian);

        let mut resolver = IdentityResolver;

        Self::scan_chunks(mss, opts, &mut riff_chunks, &mut resolver)
    }

    /// Constructs a reader for an RF64/BW64 stream. The 32-bit top-level size field must be the
    /// placeholder `0xFFFFFFFF`, and the first child chunk must be `ds64`.
    fn try_new_rf64(mut mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        let riff_len = mss.read_u32()?;

        Self::read_wave_form(&mut mss)?;

        // In RF64 the top-level 32-bit size is always the placeholder; the true size lives in
        // ds64. A concrete size means the file claims RF64 but does not follow the RF64 size
        // convention, which is a distinguishable size conflict.
        if riff_len != RF64_SIZE_PLACEHOLDER {
            return decode_error("wav: rf64 riff size must be the ds64 placeholder value");
        }

        // Parse the mandatory ds64 chunk, which must be the first chunk of the stream. Use an
        // identity resolver here because the ds64 chunk itself always has a concrete 32-bit
        // length.
        let mut header_chunks =
            ChunksReader::<RiffWaveChunks>::new_riff(None, ByteOrder::LittleEndian);
        let mut header_resolver = IdentityResolver;

        let Some(first_chunk) = header_chunks.next(&mut mss, &mut header_resolver)?
        else {
            return decode_error("wav: rf64 stream is missing the ds64 chunk");
        };

        let ds64 = match first_chunk {
            RiffWaveChunks::Ds64(ds64) => ds64.parse(&mut mss)?,
            _ => return decode_error("wav: rf64 stream is missing the ds64 chunk"),
        };

        // The true parent length includes the 4-byte form type, matching the RIFF convention of
        // "size field minus 4".
        if ds64.riff_size < 4 {
            return decode_error("wav: invalid rf64 riff size in ds64");
        }

        let riff_data_len = Some(ds64.riff_size - 4);

        let mut riff_chunks =
            ChunksReader::<RiffWaveChunks>::new_riff(riff_data_len, ByteOrder::LittleEndian);

        let mut resolver = Rf64Resolver::new(&ds64);

        Self::scan_chunks(mss, opts, &mut riff_chunks, &mut resolver)
    }

    /// Constructs a reader for a Sony Wave64 stream. The full 16-byte RIFF GUID has already been
    /// consumed during variant detection; the 64-bit size and WAVE GUID follow.
    fn try_new_wave64(mut mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        // The top-level size includes the whole file, including the 16-byte RIFF GUID and the
        // 8-byte size field itself.
        let riff_size = mss.read_u64()?;

        if riff_size < 40 {
            return decode_error("wav: invalid wave64 riff size");
        }

        // The form type is itself a full 16-byte WAVE GUID.
        let wave_guid = mss.read_quad_bytes()?;
        let mut wave_suffix = [0u8; 12];
        mss.read_buf_exact(&mut wave_suffix)?;

        if wave_guid != WAVE_RIFF_FORM || wave_suffix != WAVE64_GUID_SUFFIX {
            error!("wave64 form is not wave (guid={wave_guid:?})");

            return unsupported_error("wav: wave64 form is not wave");
        }

        // The ChunksReader begins after the 24-byte RIFF header (16-byte GUID + 8-byte size) and
        // the 16-byte WAVE GUID, so the parent body length it observes excludes all 40 bytes of
        // top-level header.
        let riff_data_len = Some(riff_size - 40);

        let mut riff_chunks = ChunksReader::<RiffWaveChunks>::new_wave64(riff_data_len);

        let mut resolver = IdentityResolver;

        Self::scan_chunks(mss, opts, &mut riff_chunks, &mut resolver)
    }

    /// Scans the child chunks of a WAVE stream until the data chunk is reached, building the
    /// single track and positioning the stream at the start of the audio data.
    fn scan_chunks<R: ChunkLenResolver>(
        mut mss: MediaSourceStream<'s>,
        opts: FormatOptions,
        riff_chunks: &mut ChunksReader<RiffWaveChunks>,
        resolver: &mut R,
    ) -> Result<Self> {
        let mut codec_params = AudioCodecParameters::new();
        let mut metadata: MetadataLog = Default::default();
        let mut packet_info = None;
        let mut fact = None;

        loop {
            let chunk = riff_chunks.next(&mut mss, resolver)?;

            // The last chunk should always be a data chunk, if it is not, then the stream is
            // unsupported.
            let Some(chunk) = chunk
            else {
                return unsupported_error("wav: missing data chunk");
            };

            match chunk {
                RiffWaveChunks::Format(fmt) => {
                    let format = fmt.parse(&mut mss)?;

                    // The Format chunk contains the block_align field and possible additional
                    // information to handle packetization and seeking.
                    let info = format.packet_info()?;
                    codec_params
                        .with_max_frames_per_packet(info.max_frames_per_packet.get())
                        .with_frames_per_block(info.frames_per_block.get());

                    // Append Format chunk fields to codec parameters.
                    append_format_params(&mut codec_params, format.format_data, format.sample_rate);

                    packet_info = Some(info);
                }
                RiffWaveChunks::Fact(fct) => {
                    fact = Some(fct.parse(&mut mss)?);
                }
                RiffWaveChunks::List(lst) => {
                    let list = lst.parse(&mut mss)?;

                    // Riff Lists can have many different forms, but WavReader only supports Info
                    // lists.
                    match &list.form {
                        b"INFO" => metadata.push(read_info_chunk(&mut mss, list.len)?),
                        _ => list.skip(&mut mss)?,
                    }
                }
                RiffWaveChunks::Data(dat) => {
                    let data = dat.parse(&mut mss)?;

                    // Record the bounds of the data chunk.
                    let data_start_pos = mss.pos();
                    let data_end_pos = match data.len {
                        Some(len) => data_start_pos
                            .checked_add(len)
                            .ok_or(Error::DecodeError("wav: data chunk length overflow"))?,
                        None => u64::MAX,
                    };

                    // Create the track.
                    let mut track = Track::new(0);

                    track.with_codec_params(CodecParameters::Audio(codec_params));

                    let Some(packet_info) = packet_info
                    else {
                        return decode_error("wav: missing format chunk");
                    };

                    // Append Fact chunk fields to track.
                    if let Some(fact) = &fact {
                        append_fact_params(&mut track, fact);
                    }

                    // Append Data chunk fields to track.
                    if let Some(data_len) = data.len {
                        append_data_params(&mut track, data_len, &packet_info);
                    }

                    // Instantiate the reader.
                    return Ok(WavReader {
                        reader: mss,
                        media_info: MediaInfo::from_track(&track),
                        tracks: vec![track],
                        chapters: opts.external_data.chapters,
                        metadata: opts.external_data.metadata.unwrap_or_default(),
                        packet_info,
                        data_start_pos,
                        data_end_pos: Some(data_end_pos).filter(|&pos| pos != u64::MAX),
                    });
                }
                RiffWaveChunks::Ds64(ds64) => {
                    // A ds64 chunk anywhere other than the first position of an RF64 stream is
                    // ignored; its length has already been accounted for by the chunk reader.
                    info!("ignoring unexpected ds64 chunk");
                    let ds64 = ds64.parse(&mut mss)?;
                    drop(ds64);
                }
            }
        }
    }
}

impl Scoreable for WavReader<'_> {
    fn score(mut src: ScopedStream<&mut MediaSourceStream<'_>>) -> Result<Score> {
        // Perform simple scoring by testing the stream marker and form for classic RIFF WAVE,
        // RF64/BW64, and Sony Wave64.
        let marker = src.read_quad_bytes()?;

        match marker {
            WAVE_STREAM_MARKER | RF64_STREAM_MARKER | BW64_STREAM_MARKER => {
                // Classic RIFF-style layout: 4-byte size, then the 4-byte WAVE form.
                src.ignore_bytes(4)?;
                let riff_form = src.read_quad_bytes()?;

                if riff_form != WAVE_RIFF_FORM {
                    return Ok(Score::Unsupported);
                }
            }
            WAVE64_STREAM_MARKER => {
                // Sony Wave64: a 12-byte GUID suffix, an 8-byte size, then the 16-byte WAVE GUID.
                let mut guid_suffix = [0u8; 12];
                src.read_buf_exact(&mut guid_suffix)?;

                if guid_suffix != WAVE64_GUID_SUFFIX {
                    return Ok(Score::Unsupported);
                }

                src.ignore_bytes(8)?;

                let wave_guid = src.read_quad_bytes()?;
                let mut wave_suffix = [0u8; 12];
                src.read_buf_exact(&mut wave_suffix)?;

                if wave_guid != WAVE_RIFF_FORM || wave_suffix != WAVE64_GUID_SUFFIX {
                    return Ok(Score::Unsupported);
                }
            }
            _ => return Ok(Score::Unsupported),
        }

        Ok(Score::Supported(255))
    }
}

impl ProbeableFormat<'_> for WavReader<'_> {
    fn try_probe_new(
        mss: MediaSourceStream<'_>,
        opts: FormatOptions,
    ) -> Result<Box<dyn FormatReader + '_>> {
        Ok(Box::new(WavReader::try_new(mss, opts)?))
    }

    fn probe_data() -> &'static [ProbeFormatData] {
        &[
            // WAVE RIFF form, including 64-bit RF64 and Sony Wave64 variants.
            support_format!(
                WAVE_FORMAT_INFO,
                &["wav", "wave", "rf64", "bw64", "w64"],
                &["audio/vnd.wave", "audio/x-wav", "audio/wav", "audio/wave"],
                &[b"RIFF", b"RF64", b"BW64", b"riff"]
            ),
        ]
    }
}

impl FormatReader for WavReader<'_> {
    fn format_info(&self) -> &FormatInfo {
        &WAVE_FORMAT_INFO
    }

    fn media_info(&self) -> &MediaInfo {
        &self.media_info
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        next_packet(
            &mut self.reader,
            &self.packet_info,
            &self.tracks,
            self.data_start_pos,
            self.data_end_pos,
        )
    }

    fn metadata(&mut self) -> Metadata<'_> {
        self.metadata.metadata()
    }

    fn chapters(&self) -> Option<&ChapterGroup> {
        self.chapters.as_ref()
    }

    fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    fn seek(&mut self, _mode: SeekMode, to: SeekTo) -> Result<SeekedTo> {
        if self.tracks.is_empty() {
            return seek_error(SeekErrorKind::Unseekable);
        }

        let track = &self.tracks[0];

        let required_ts = match to {
            // Frame timestamp given.
            SeekTo::Timestamp { ts, .. } => ts,
            // Time value given, calculate frame timestamp using the time base.
            SeekTo::Time { time, .. } => {
                // The timebase is required to calculate the timestamp.
                let tb = track.time_base.ok_or(Error::SeekError(SeekErrorKind::Unseekable))?;

                // If the timestamp overflows, the seek if out-of-range.
                tb.calc_timestamp(time).ok_or(Error::SeekError(SeekErrorKind::OutOfRange))?
            }
        };

        // Negative timestamps are not allowed.
        if required_ts.is_negative() {
            return seek_error(SeekErrorKind::OutOfRange);
        }

        // If the total number of frames in the track is known, verify the desired frame timestamp
        // does not exceed it.
        if let Some(num_frames) = track.num_frames {
            if required_ts.get() as u64 > num_frames {
                return seek_error(SeekErrorKind::OutOfRange);
            }
        }

        debug!("seeking to frame_ts={required_ts}");

        // WAVE is not internally packetized for PCM codecs. Packetization is simulated by trying to
        // read a constant number of samples or blocks every call to next_packet. Therefore, a
        // packet begins wherever the data stream is currently positioned. Since timestamps on
        // packets should be determinstic, instead of seeking to the exact timestamp requested and
        // starting the next packet there, seek to a packet boundary. In this way, packets will have
        // the same timestamps regardless if the stream was seeked or not.
        let actual_ts = self.packet_info.get_actual_ts(required_ts);

        // Calculate the absolute byte offset of the desired audio frame.
        let seek_pos =
            self.data_start_pos + (actual_ts.get() as u64 * self.packet_info.block_size.get());

        // If the reader supports seeking we can seek directly to the frame's offset wherever it may
        // be.
        if self.reader.is_seekable() {
            self.reader.seek(SeekFrom::Start(seek_pos))?;
        }
        // If the reader does not support seeking, we can only emulate forward seeks by consuming
        // bytes. If the reader has to seek backwards, return an error.
        else {
            let current_pos = self.reader.pos();
            if seek_pos >= current_pos {
                self.reader.ignore_bytes(seek_pos - current_pos)?;
            }
            else {
                return seek_error(SeekErrorKind::ForwardOnly);
            }
        }

        debug!(
            "seeked to packet_ts={} (delta={})",
            actual_ts,
            actual_ts.saturating_delta(required_ts)
        );

        Ok(SeekedTo { track_id: 0, actual_ts, required_ts })
    }

    fn into_inner<'s>(self: Box<Self>) -> MediaSourceStream<'s>
    where
        Self: 's,
    {
        self.reader
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use symphonia_core::formats::FormatOptions;
    use symphonia_core::io::MediaSourceStream;

    use super::*;
    use crate::common::WAVE64_GUID_SUFFIX;

    /// A minimal mono 16-bit PCM format chunk body (16 bytes).
    const FMT_PCM_BODY: [u8; 16] = [
        0x01, 0x00, // WAVE_FORMAT_PCM
        0x01, 0x00, // 1 channel
        0x40, 0x1f, 0x00, 0x00, // 8000 Hz
        0x80, 0x3e, 0x00, 0x00, // 16000 bytes/sec
        0x02, 0x00, // block align = 2
        0x10, 0x00, // 16 bits/sample
    ];

    /// Interleaved mono 16-bit samples: 0x1122, 0x3344, 0x5566, 0x7788.
    const DATA_BODY: [u8; 8] = [0x22, 0x11, 0x44, 0x33, 0x66, 0x55, 0x88, 0x77];

    fn open(buf: Vec<u8>) -> Result<Box<WavReader<'static>>> {
        let mss = MediaSourceStream::new(Box::new(Cursor::new(buf)), Default::default());
        match WavReader::try_new(mss, FormatOptions::default()) {
            Ok(reader) => Ok(Box::new(reader)),
            Err(err) => Err(err),
        }
    }

    fn assert_decode_error<T>(result: Result<T>, expected: &'static str) {
        match result {
            Err(Error::DecodeError(msg)) => assert_eq!(msg, expected),
            _ => panic!("expected DecodeError({expected:?})"),
        }
    }

    fn assert_unsupported_error<T>(result: Result<T>, expected: &'static str) {
        match result {
            Err(Error::Unsupported(msg)) => assert_eq!(msg, expected),
            _ => panic!("expected Unsupported({expected:?})"),
        }
    }

    fn assert_io_error<T>(result: Result<T>) {
        match result {
            Err(Error::IoError(_)) => (),
            _ => panic!("expected IoError"),
        }
    }

    fn riff_chunk(tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(tag);
        chunk.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chunk.extend_from_slice(body);
        // Classic RIFF pads chunk bodies to a 2-byte boundary.
        if body.len() & 1 == 1 {
            chunk.push(0);
        }
        chunk
    }

    fn wave64_guid(tag: &[u8; 4]) -> Vec<u8> {
        let mut guid = Vec::with_capacity(16);
        guid.extend_from_slice(tag);
        guid.extend_from_slice(&WAVE64_GUID_SUFFIX);
        guid
    }

    fn wave64_chunk(tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut chunk = wave64_guid(tag);
        // The size field includes the whole 24-byte chunk header (16-byte GUID + 8-byte size).
        let size = 24 + body.len() as u64;
        chunk.extend_from_slice(&size.to_le_bytes());
        chunk.extend_from_slice(body);
        // Sony Wave64 pads the whole chunk to an 8-byte boundary. The 24-byte header is already a
        // multiple of 8, so padding the body suffices.
        let pad = (8 - (body.len() as u64 % 8)) % 8;
        chunk.extend(std::iter::repeat(0).take(pad as usize));
        chunk
    }

    fn build_riff_wave() -> Vec<u8> {
        let mut chunks = Vec::new();
        chunks.extend(riff_chunk(b"fmt ", &FMT_PCM_BODY));
        chunks.extend(riff_chunk(b"data", &DATA_BODY));

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        // Top-level size includes the 4-byte WAVE form.
        buf.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(&chunks);
        buf
    }

    fn build_rf64_wave() -> Vec<u8> {
        let mut chunks = Vec::new();

        // ds64: riff_size, data_size, sample_count, table entry count (no extra entries).
        let mut ds64_body = Vec::new();
        let data_size = DATA_BODY.len() as u64;
        // riff_size covers the WAVE form + ds64 chunk + fmt chunk + data chunk; computed after the
        // other chunks are built, so reserve its slot for now.
        ds64_body.extend_from_slice(&0u64.to_le_bytes());
        ds64_body.extend_from_slice(&data_size.to_le_bytes());
        ds64_body.extend_from_slice(&4u64.to_le_bytes());
        ds64_body.extend_from_slice(&0u32.to_le_bytes());

        chunks.extend(riff_chunk(b"ds64", &ds64_body));
        chunks.extend(riff_chunk(b"fmt ", &FMT_PCM_BODY));

        // The data chunk overflows 32-bits in the motivating case, so it always uses the
        // placeholder here to exercise ds64 resolution.
        let mut data_chunk = Vec::new();
        data_chunk.extend_from_slice(b"data");
        data_chunk.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        data_chunk.extend_from_slice(&DATA_BODY);
        chunks.extend(data_chunk);

        let riff_size = (4 + chunks.len()) as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RF64");
        buf.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(&chunks);

        // Backfill the true top-level size into the ds64 body (offset 12: marker(4)+size(4)+
        // form(4)+ds64 header(8)).
        buf[20..28].copy_from_slice(&riff_size.to_le_bytes());
        buf
    }

    fn build_wave64_wave() -> Vec<u8> {
        let mut chunks = Vec::new();
        chunks.extend(wave64_chunk(b"fmt ", &FMT_PCM_BODY));
        chunks.extend(wave64_chunk(b"data", &DATA_BODY));

        let mut buf = Vec::new();
        buf.extend(wave64_guid(b"riff"));
        // Top-level size includes the full file: RIFF GUID (16) + size (8) + WAVE GUID (16) +
        // child chunks.
        let riff_size = 24 + 16 + chunks.len() as u64;
        buf.extend_from_slice(&riff_size.to_le_bytes());
        buf.extend(wave64_guid(b"WAVE"));
        buf.extend_from_slice(&chunks);
        buf
    }

    fn sample_rate_of(reader: &WavReader<'_>) -> u32 {
        reader.tracks()[0]
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .expect("audio codec parameters")
            .sample_rate
            .expect("sample rate")
    }

    fn drain_all_packets(reader: &mut Box<WavReader<'static>>) -> u64 {
        let mut frames = 0;
        while let Some(packet) = reader.next_packet().unwrap() {
            frames += packet.dur.get();
        }
        frames
    }

    #[test]
    fn parses_classic_riff_wave() {
        let mut reader = open(build_riff_wave()).unwrap();

        assert_eq!(reader.tracks().len(), 1);
        let track = &reader.tracks()[0];
        assert_eq!(sample_rate_of(&reader), 8000);
        assert_eq!(track.num_frames, Some(4));

        assert_eq!(drain_all_packets(&mut reader), 4);
    }

    #[test]
    fn parses_rf64_wave_with_ds64() {
        let mut reader = open(build_rf64_wave()).unwrap();

        let track = &reader.tracks()[0];
        assert_eq!(sample_rate_of(&reader), 8000);
        // The data length is the 64-bit value resolved from ds64.
        assert_eq!(track.num_frames, Some(4));

        assert_eq!(drain_all_packets(&mut reader), 4);
    }

    #[test]
    fn parses_sony_wave64_wave() {
        let mut reader = open(build_wave64_wave()).unwrap();

        let track = &reader.tracks()[0];
        assert_eq!(sample_rate_of(&reader), 8000);
        assert_eq!(track.num_frames, Some(4));

        assert_eq!(drain_all_packets(&mut reader), 4);
    }

    #[test]
    fn wave64_skips_unknown_chunks_and_8byte_padding() {
        // A 9-byte unknown chunk body forces 7 bytes of 8-byte alignment padding. Its GUID uses an
        // unknown first 4 bytes but a valid Wave64 suffix, exercising both unknown-GUID skipping
        // and cross-chunk positioning after padding.
        let unknown_body = [0xa5u8; 9];

        let mut chunks = Vec::new();
        chunks.extend(wave64_chunk(b"JUNK", &unknown_body));
        chunks.extend(wave64_chunk(b"fmt ", &FMT_PCM_BODY));
        chunks.extend(wave64_chunk(b"data", &DATA_BODY));

        let mut buf = Vec::new();
        buf.extend(wave64_guid(b"riff"));
        let riff_size = 24 + 16 + chunks.len() as u64;
        buf.extend_from_slice(&riff_size.to_le_bytes());
        buf.extend(wave64_guid(b"WAVE"));
        buf.extend_from_slice(&chunks);

        let mut reader = open(buf).unwrap();
        assert_eq!(drain_all_packets(&mut reader), 4);
    }

    #[test]
    fn riff_skips_unknown_chunks_and_2byte_padding() {
        // A 5-byte unknown chunk body forces 1 byte of 2-byte alignment padding.
        let unknown_body = [0xa5u8; 5];

        let mut chunks = Vec::new();
        chunks.extend(riff_chunk(b"JUNK", &unknown_body));
        chunks.extend(riff_chunk(b"fmt ", &FMT_PCM_BODY));
        chunks.extend(riff_chunk(b"data", &DATA_BODY));

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(&chunks);

        let mut reader = open(buf).unwrap();
        assert_eq!(drain_all_packets(&mut reader), 4);
    }

    #[test]
    fn rf64_missing_ds64_is_distinguishable_error() {
        let mut chunks = Vec::new();
        chunks.extend(riff_chunk(b"fmt ", &FMT_PCM_BODY));
        chunks.extend(riff_chunk(b"data", &DATA_BODY));

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RF64");
        buf.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(&chunks);

        assert_decode_error(open(buf), "wav: rf64 stream is missing the ds64 chunk");
    }

    #[test]
    fn rf64_placeholder_without_ds64_entry_is_distinguishable_error() {
        let mut buf = build_rf64_wave();

        // Rename the data chunk to an unregistered tag while keeping its 0xFFFFFFFF placeholder.
        // The resolver then cannot find a backing ds64 entry and must report a distinguishable
        // error rather than silently accepting the placeholder.
        let marker = buf.windows(4).position(|w| w == b"data").expect("data chunk must be present");
        buf[marker..marker + 4].copy_from_slice(b"JUNK");

        assert_decode_error(open(buf), "wav: rf64 placeholder length missing from ds64");
    }

    #[test]
    fn rf64_ds64_size_conflict_is_distinguishable_error() {
        let mut buf = build_rf64_wave();

        // Locate the data chunk header and replace the placeholder with a concrete 32-bit length
        // that disagrees with the ds64 data_size (8 bytes).
        let marker = buf.windows(4).position(|w| w == b"data").expect("data chunk must be present");
        let len_offset = marker + 4;
        buf[len_offset..len_offset + 4].copy_from_slice(&7u32.to_le_bytes());

        assert_decode_error(open(buf), "wav: ds64 chunk size conflicts with chunk header length");
    }

    #[test]
    fn rf64_concrete_top_level_size_is_conflict_error() {
        let mut buf = build_rf64_wave();

        // Replace the top-level placeholder with a concrete, otherwise-valid size.
        buf[4..8].copy_from_slice(&100u32.to_le_bytes());

        assert_decode_error(open(buf), "wav: rf64 riff size must be the ds64 placeholder value");
    }

    #[test]
    fn truncated_rf64_is_io_error() {
        let mut buf = build_rf64_wave();

        // Drop the final 2 bytes of the data chunk body. The headers remain self-consistent, so
        // opening succeeds; the truncation surfaces as an IO error only when the missing audio
        // bytes are actually read.
        buf.truncate(buf.len() - 2);

        let mut reader = open(buf).unwrap();
        assert_io_error(reader.next_packet());
    }

    #[test]
    fn truncated_wave64_is_io_error() {
        let mut buf = build_wave64_wave();

        // Drop the final 2 bytes of the data chunk body.
        buf.truncate(buf.len() - 2);

        let mut reader = open(buf).unwrap();
        assert_io_error(reader.next_packet());
    }

    #[test]
    fn truncated_classic_riff_is_io_error() {
        let mut buf = build_riff_wave();

        // Drop the final byte of the data chunk body.
        buf.truncate(buf.len() - 1);

        let mut reader = open(buf).unwrap();
        assert_io_error(reader.next_packet());
    }

    #[test]
    fn malformed_ds64_is_decode_error() {
        let mut chunks = Vec::new();
        // ds64 with a body shorter than the 28-byte fixed part.
        chunks.extend(riff_chunk(b"ds64", &[0u8; 8]));
        chunks.extend(riff_chunk(b"fmt ", &FMT_PCM_BODY));
        chunks.extend(riff_chunk(b"data", &DATA_BODY));

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RF64");
        buf.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(&chunks);

        assert_decode_error(open(buf), "wav: malformed ds64 chunk");
    }

    #[test]
    fn wave64_bad_wave_guid_is_unsupported() {
        let mut buf = build_wave64_wave();

        // Corrupt the WAVE form GUID (it starts at offset 24, right after RIFF GUID + size).
        buf[24] = b'X';

        assert_unsupported_error(open(buf), "wav: wave64 form is not wave");
    }

    #[test]
    fn wave64_chunk_size_smaller_than_header_is_decode_error() {
        let mut buf = build_wave64_wave();

        // The first child chunk's size field begins after the 40-byte top-level header. A size of
        // 8 is smaller than the 16-byte chunk header.
        let size_offset = 40 + 16;
        buf[size_offset..size_offset + 8].copy_from_slice(&8u64.to_le_bytes());

        assert_decode_error(open(buf), "wav: wave64 chunk size is smaller than its header");
    }

    #[test]
    fn wave64_child_chunk_exceeding_parent_is_decode_error() {
        let mut buf = build_wave64_wave();

        // Inflate the fmt chunk's size field beyond the parent body. The fmt header starts at
        // offset 40; its size field follows the 16-byte GUID.
        let fmt_size_offset = 40 + 16;
        buf[fmt_size_offset..fmt_size_offset + 8]
            .copy_from_slice(&(16 + 0x0010_0000u64).to_le_bytes());

        assert_decode_error(open(buf), "riff: chunk length exceeds parent (list) chunk length");
    }

    #[test]
    fn rf64_ds64_riff_size_smaller_than_form_is_decode_error() {
        let mut buf = build_rf64_wave();

        // The ds64 riff_size lives at offset 20 (marker(4) + size(4) + form(4) + ds64
        // header(8)). A value below 4 cannot even cover the WAVE form.
        buf[20..28].copy_from_slice(&3u64.to_le_bytes());

        assert_decode_error(open(buf), "wav: invalid rf64 riff size in ds64");
    }

    #[test]
    fn rf64_resolves_non_data_placeholder_from_ds64_table() {
        // A JUNK chunk whose true length does not fit in a 32-bit field uses the placeholder and
        // records its true length in the ds64 table. With only a 1-byte body on disk, the parent
        // boundary check must reject the table-resolved 64-bit length, proving the resolver applied
        // the table entry (an identity resolver would have read 0xFFFFFFFF bytes instead and
        // failed differently).
        let junk_len = 0x1_0000_0001u64;

        let mut ds64_body = Vec::new();
        let data_size = DATA_BODY.len() as u64;
        ds64_body.extend_from_slice(&0u64.to_le_bytes()); // riff_size, backfilled later.
        ds64_body.extend_from_slice(&data_size.to_le_bytes());
        ds64_body.extend_from_slice(&4u64.to_le_bytes()); // sample count.
        ds64_body.extend_from_slice(&1u32.to_le_bytes()); // one table entry.
        ds64_body.extend_from_slice(b"JUNK");
        ds64_body.extend_from_slice(&junk_len.to_le_bytes());

        let mut chunks = Vec::new();
        chunks.extend(riff_chunk(b"ds64", &ds64_body));
        chunks.extend(riff_chunk(b"fmt ", &FMT_PCM_BODY));

        // The JUNK chunk uses the placeholder but its on-disk body is a single byte.
        let mut junk_chunk = Vec::new();
        junk_chunk.extend_from_slice(b"JUNK");
        junk_chunk.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        junk_chunk.push(0xa5);
        chunks.extend(junk_chunk);

        let mut data_chunk = Vec::new();
        data_chunk.extend_from_slice(b"data");
        data_chunk.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        data_chunk.extend_from_slice(&DATA_BODY);
        chunks.extend(data_chunk);

        let riff_size = (4 + chunks.len()) as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RF64");
        buf.extend_from_slice(&RF64_SIZE_PLACEHOLDER.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(&chunks);
        buf[20..28].copy_from_slice(&riff_size.to_le_bytes());

        assert_decode_error(open(buf), "riff: chunk length exceeds parent (list) chunk length");
    }

    #[test]
    fn wave64_seek_locates_frame_across_chunks() {
        let mut reader = open(build_wave64_wave()).unwrap();

        // Request the third frame (timestamp 2). Because packetization aligns to a packet boundary
        // (1152 frames), the aligned timestamp is 0 and the seek targets the data start.
        let requested = Timestamp::from(2);

        let seeked = reader
            .seek(SeekMode::Accurate, SeekTo::Timestamp { track_id: 0, ts: requested })
            .unwrap();

        assert_eq!(seeked.actual_ts, Timestamp::from(0));
        assert_eq!(reader.reader.pos(), reader.data_start_pos);

        let packet = reader.next_packet().unwrap().unwrap();
        assert_eq!(packet.pts, Timestamp::from(0));
        // The full 8-byte body is read from the correctly located data chunk.
        assert_eq!(packet.data.len(), DATA_BODY.len());
        assert_eq!(&packet.data[..2], &[0x22, 0x11]);

        // A later, in-range seek past the available frames is still bounded by the 64-bit data end
        // position rather than wrapping.
        let seeked = reader
            .seek(SeekMode::Accurate, SeekTo::Timestamp { track_id: 0, ts: Timestamp::from(1) })
            .unwrap();
        assert_eq!(seeked.actual_ts, Timestamp::from(0));
    }

    #[test]
    fn scores_all_three_variants() {
        fn score_of(buf: Vec<u8>) -> Score {
            let mut mss = MediaSourceStream::new(Box::new(Cursor::new(buf)), Default::default());
            WavReader::score(ScopedStream::new(&mut mss, 4096)).unwrap()
        }

        for score in [
            score_of(build_riff_wave()),
            score_of(build_rf64_wave()),
            score_of(build_wave64_wave()),
        ] {
            assert!(matches!(score, Score::Supported(255)));
        }
    }
}
