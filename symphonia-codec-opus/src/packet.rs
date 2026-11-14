// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Opus packet parsing and validation.
//!
//! Implements the packet parsing sections of RFC 6716 (sections 3.1 to 3.2.5). The parser is
//! allocation-free and validates every structural constraint a demuxer-agnostic decoder is
//! responsible for: TOC configuration, frame packing code, self-delimiting lengths and total
//! payload length consistency.

use symphonia_core::errors::{decode_error, Result};

/// Opus internal sample rate. All decoder output and PLC DSP runs at this rate.
pub const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Number of samples per millisecond at the Opus internal sample rate.
const SAMPLES_PER_MS: u32 = OPUS_SAMPLE_RATE / 1_000;

/// The longest permitted Opus frame (120 ms at 48 kHz).
pub const MAX_FRAME_SAMPLES: usize = 120 * (SAMPLES_PER_MS as usize);

/// The maximum number of frames a single Opus packet may contain (RFC 6716 §3.2.5).
pub const MAX_FRAMES_PER_PACKET: usize = 48;

/// Audio bandwidth / coding mode of an Opus configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Bandwidth {
    /// SILK narrowband (4 kHz audio bandwidth).
    Narrowband,
    /// SILK mediumband (6 kHz).
    Mediumband,
    /// SILK wideband (8 kHz).
    Wideband,
    /// Hybrid super-wideband (12 kHz).
    SuperWideband,
    /// CELT fullband (20 kHz).
    Fullband,
}

/// Coding layer of an Opus configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CodingMode {
    /// SILK-only mode.
    Silk,
    /// Hybrid SILK/CELT mode.
    Hybrid,
    /// CELT-only mode.
    Celt,
}

/// Static description of one of the 32 Opus TOC configurations (RFC 6716 Table 1).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// Frame duration in milliseconds (2.5, 5, 10, 20, 40 or 60); stored as tenths of a ms.
    pub frame_duration_tenths_ms: u16,
    /// Number of samples in one frame at 48 kHz.
    pub frame_samples: u32,
    /// Coding layer.
    pub mode: CodingMode,
    /// Audio bandwidth.
    pub bandwidth: Bandwidth,
}

const fn cfg(tenths_ms: u16, samples: u32, mode: CodingMode, bw: Bandwidth) -> Config {
    Config {
        frame_duration_tenths_ms: tenths_ms,
        frame_samples: samples,
        mode,
        bandwidth: bw,
    }
}

/// Lookup table for the 32 TOC configuration numbers (RFC 6716, Table 1).
#[rustfmt::skip]
pub const CONFIG_TABLE: [Config; 32] = [
    // SILK-only, NB/MB/WB, 10/20/40/60 ms
    cfg(100,  480, CodingMode::Silk,   Bandwidth::Narrowband),
    cfg(200,  960, CodingMode::Silk,   Bandwidth::Narrowband),
    cfg(400, 1920, CodingMode::Silk,   Bandwidth::Narrowband),
    cfg(600, 2880, CodingMode::Silk,   Bandwidth::Narrowband),
    cfg(100,  480, CodingMode::Silk,   Bandwidth::Mediumband),
    cfg(200,  960, CodingMode::Silk,   Bandwidth::Mediumband),
    cfg(400, 1920, CodingMode::Silk,   Bandwidth::Mediumband),
    cfg(600, 2880, CodingMode::Silk,   Bandwidth::Mediumband),
    cfg(100,  480, CodingMode::Silk,   Bandwidth::Wideband),
    cfg(200,  960, CodingMode::Silk,   Bandwidth::Wideband),
    cfg(400, 1920, CodingMode::Silk,   Bandwidth::Wideband),
    cfg(600, 2880, CodingMode::Silk,   Bandwidth::Wideband),
    // Hybrid, SWB/FB, 10/20 ms
    cfg(100,  480, CodingMode::Hybrid, Bandwidth::SuperWideband),
    cfg(200,  960, CodingMode::Hybrid, Bandwidth::SuperWideband),
    cfg(100,  480, CodingMode::Hybrid, Bandwidth::Fullband),
    cfg(200,  960, CodingMode::Hybrid, Bandwidth::Fullband),
    // CELT-only, NB/WB/SWB/FB, 2.5/5/10/20 ms
    cfg( 25,  120, CodingMode::Celt,   Bandwidth::Narrowband),
    cfg( 50,  240, CodingMode::Celt,   Bandwidth::Narrowband),
    cfg(100,  480, CodingMode::Celt,   Bandwidth::Narrowband),
    cfg(200,  960, CodingMode::Celt,   Bandwidth::Narrowband),
    cfg( 25,  120, CodingMode::Celt,   Bandwidth::Wideband),
    cfg( 50,  240, CodingMode::Celt,   Bandwidth::Wideband),
    cfg(100,  480, CodingMode::Celt,   Bandwidth::Wideband),
    cfg(200,  960, CodingMode::Celt,   Bandwidth::Wideband),
    cfg( 25,  120, CodingMode::Celt,   Bandwidth::SuperWideband),
    cfg( 50,  240, CodingMode::Celt,   Bandwidth::SuperWideband),
    cfg(100,  480, CodingMode::Celt,   Bandwidth::SuperWideband),
    cfg(200,  960, CodingMode::Celt,   Bandwidth::SuperWideband),
    cfg( 25,  120, CodingMode::Celt,   Bandwidth::Fullband),
    cfg( 50,  240, CodingMode::Celt,   Bandwidth::Fullband),
    cfg(100,  480, CodingMode::Celt,   Bandwidth::Fullband),
    cfg(200,  960, CodingMode::Celt,   Bandwidth::Fullband),
];

/// Frame packing code (`c`), the two least-significant bits of the TOC byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FramePacking {
    /// `c = 0`: exactly 1 frame.
    One,
    /// `c = 1`: 2 frames, equal size, one compressed size.
    TwoEqual,
    /// `c = 2`: 2 frames, VBR, one compressed size.
    TwoVbr,
    /// `c = 3`: arbitrary number of frames, VBR or CBR, self-delimiting.
    Arbitrary,
}

/// Parsed table-of-contents of an Opus packet.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Toc {
    /// The raw TOC byte.
    pub byte: u8,
    /// Configuration number (0..32), the 5 most-significant bits.
    pub config: u8,
    /// Stereo flag (`s`).
    pub stereo: bool,
    /// Frame packing code (`c`).
    pub packing: FramePacking,
    /// Static configuration of the configuration number.
    pub info: Config,
}

impl Toc {
    /// Parse the TOC byte of an Opus packet.
    pub fn parse(byte: u8) -> Toc {
        let config = byte >> 3;
        let stereo = (byte & 0x04) != 0;
        let packing = match byte & 0x03 {
            0 => FramePacking::One,
            1 => FramePacking::TwoEqual,
            2 => FramePacking::TwoVbr,
            _ => FramePacking::Arbitrary,
        };
        Toc { byte, config, stereo, packing, info: CONFIG_TABLE[config as usize] }
    }

    /// Number of samples per frame at 48 kHz for this configuration.
    pub fn frame_samples(&self) -> usize {
        self.info.frame_samples as usize
    }
}

/// A validated view of an Opus packet: its TOC and the byte ranges of every contained frame.
///
/// The frame slices borrow from the packet buffer so validation never copies or allocates.
#[derive(Copy, Clone, Debug)]
pub struct ParsedPacket<'a> {
    /// Parsed table of contents.
    pub toc: Toc,
    /// Number of frames in the packet.
    pub num_frames: usize,
    packet: &'a [u8],
    // Start offset and length of each frame; at most [`MAX_FRAMES_PER_PACKET`] entries. A fixed
    // array keeps the parser allocation-free while bounding the working set.
    offsets: [(u16, u16); MAX_FRAMES_PER_PACKET],
}

impl<'a> ParsedPacket<'a> {
    /// Return the payload slice of frame `index` (0-based).
    pub fn frame(&self, index: usize) -> &'a [u8] {
        let (start, len) = self.offsets[index];
        &self.packet[start as usize..start as usize + len as usize]
    }

    /// Iterate over the payload slices of every frame in the packet.
    pub fn frames(&self) -> impl Iterator<Item = &'a [u8]> {
        let packet = self.packet;
        self.offsets[..self.num_frames]
            .iter()
            .map(move |&(start, len)| &packet[start as usize..start as usize + len as usize])
    }
}

/// Read an unsigned LEB128/Opus-style variable-length integer (1 or 2 bytes, RFC 6716 §3.2.5).
///
/// Returns `(value, bytes_consumed)`.
fn read_oplint(buf: &[u8]) -> Result<(u32, usize)> {
    let Some(&first) = buf.first() else {
        return decode_error("opus: truncated frame length");
    };
    if first < 252 {
        return Ok((u32::from(first), 1));
    }
    let Some(&second) = buf.get(1) else {
        return decode_error("opus: truncated second frame length byte");
    };
    Ok((u32::from(first) + u32::from(second) * 4, 2))
}

/// Fully parse and validate an Opus packet per RFC 6716 §3.
///
/// An empty payload, an over-sized frame count, inconsistent self-delimiting lengths or trailing
/// garbage all produce a [`DecodeError`](symphonia_core::errors::Error::DecodeError).
pub fn parse_packet(packet: &[u8]) -> Result<ParsedPacket<'_>> {
    let Some((&toc_byte, payload)) = packet.split_first() else {
        return decode_error("opus: empty packet");
    };

    let toc = Toc::parse(toc_byte);
    let mut parsed = ParsedPacket {
        toc,
        num_frames: 0,
        packet,
        offsets: [(0, 0); MAX_FRAMES_PER_PACKET],
    };

    // Base offset of frame payloads inside the full packet: TOC byte occupies index 0.
    let base: usize = 1;

    match toc.packing {
        FramePacking::One => {
            // The remainder of the packet is the single frame.
            let len = payload.len();
            validate_frame_len(toc, len)?;
            parsed.num_frames = 1;
            parsed.offsets[0] = (base as u16, len as u16);
        }
        FramePacking::TwoEqual => {
            // Two frames of equal compressed size.
            if payload.len() % 2 != 0 {
                return decode_error("opus: odd length for two equal-sized frames");
            }
            let len = payload.len() / 2;
            validate_frame_len(toc, len)?;
            parsed.num_frames = 2;
            parsed.offsets[0] = (base as u16, len as u16);
            parsed.offsets[1] = ((base + len) as u16, len as u16);
        }
        FramePacking::TwoVbr => {
            // First frame size is a 1- or 2-byte oplint, remainder is the second frame.
            let (first_len, int_len) = read_oplint(payload)?;
            let first_len = first_len as usize;
            let second_start = int_len;
            if second_start + first_len > payload.len() {
                return decode_error("opus: first VBR frame exceeds packet length");
            }
            let second_len = payload.len() - second_start - first_len;
            validate_frame_len(toc, first_len)?;
            validate_frame_len(toc, second_len)?;
            parsed.num_frames = 2;
            parsed.offsets[0] = ((base + second_start) as u16, first_len as u16);
            parsed.offsets[1] =
                ((base + second_start + first_len) as u16, second_len as u16);
        }
        FramePacking::Arbitrary => {
            // c = 3, RFC 6716 §3.2.5. Layout:
            //   [TOC][frame count byte]([frame lengths])([padding length])[frame data][padding]
            let Some((&count_byte, rest)) = payload.split_first() else {
                return decode_error("opus: missing frame count byte");
            };
            let is_vbr = (count_byte & 0x80) != 0;
            let has_padding = (count_byte & 0x40) != 0;
            let num_frames = usize::from(count_byte & 0x1f);

            if num_frames == 0 || num_frames > MAX_FRAMES_PER_PACKET {
                return decode_error("opus: frame count out of range");
            }
            parsed.num_frames = num_frames;

            // Walk the length/padding header; `header_end` is the absolute index where frame data
            // begins and `padding_len` is the number of trailing padding bytes.
            let mut cursor: usize = base + 1; // index of the frame-count byte's successor
            let mut header_lengths = [0u16; MAX_FRAMES_PER_PACKET];
            let mut padding_len: usize = 0;

            if is_vbr {
                // VBR: oplint lengths for frames 0..N-2 (the last length is implicit).
                for slot in header_lengths.iter_mut().take(num_frames - 1) {
                    let (len, int_len) = read_oplint(&packet[cursor..])?;
                    validate_frame_len(toc, len as usize)?;
                    *slot = len as u16;
                    cursor += int_len;
                }
            }
            else {
                // CBR: one oplint giving the common frame size.
                let (common_len, int_len) = read_oplint(rest)?;
                validate_frame_len(toc, common_len as usize)?;
                for slot in header_lengths.iter_mut().take(num_frames) {
                    *slot = common_len as u16;
                }
                cursor += int_len;
            }

            if has_padding {
                // One oplint gives the total length of the trailing padding.
                let (p, int_len) = read_oplint(&packet[cursor..])?;
                padding_len = p as usize;
                cursor += int_len;
            }

            let data_start = cursor;
            let data_end = packet
                .len()
                .checked_sub(padding_len)
                .filter(|&end| end >= data_start)
                .ok_or("opus: padding exceeds packet")
                .map_err(|_| {
                    symphonia_core::errors::Error::DecodeError("opus: padding exceeds packet")
                })?;

            if is_vbr {
                // Lay out frames 0..N-2 with their declared lengths; the final frame takes the
                // remainder of the data region.
                let mut off = data_start;
                for i in 0..num_frames - 1 {
                    let len = header_lengths[i] as usize;
                    if off + len > data_end {
                        return decode_error("opus: VBR frame overflows data region");
                    }
                    parsed.offsets[i] = (off as u16, len as u16);
                    off += len;
                }
                let last_len = data_end - off;
                validate_frame_len(toc, last_len)?;
                parsed.offsets[num_frames - 1] = (off as u16, last_len as u16);
            }
            else {
                let common_len = header_lengths[0] as usize;
                if data_start + num_frames * common_len != data_end {
                    return decode_error("opus: CBR packet length mismatch");
                }
                for (i, slot) in
                    parsed.offsets.iter_mut().enumerate().take(num_frames)
                {
                    let start = data_start + i * common_len;
                    *slot = (start as u16, common_len as u16);
                }
            }

            // Every byte of the packet must be accounted for.
            if data_end + padding_len != packet.len() {
                return decode_error("opus: packet length does not match frame layout");
            }

            // Final extent sanity check.
            for &(start, len) in parsed.offsets.iter().take(num_frames) {
                if start as usize + len as usize > packet.len() {
                    return decode_error("opus: frame extent exceeds packet");
                }
            }
        }
    }

    Ok(parsed)
}

/// Validate the compressed size of a single frame against the RFC 6716 maximum (1275 bytes).
fn validate_frame_len(toc: Toc, len: usize) -> Result<()> {
    // RFC 6716 §3.2.1: the largest possible frame length is 1275 bytes.
    const MAX_FRAME_PAYLOAD: usize = 1275;
    // A zero-length frame is only valid for CELT-only modes (DTX / PLC-only frame in CELT).
    if len == 0 && toc.info.mode != CodingMode::Celt {
        return decode_error("opus: zero-length frame outside CELT mode");
    }
    if len > MAX_FRAME_PAYLOAD {
        return decode_error("opus: frame exceeds maximum payload size");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toc(config: u8, stereo: bool, code: u8) -> u8 {
        (config << 3) | ((stereo as u8) << 2) | code
    }

    #[test]
    fn toc_decodes_config_stereo_and_packing() {
        // Config 16 = CELT-only NB, 2.5 ms.
        let t = Toc::parse(toc(16, true, 3));
        assert_eq!(t.config, 16);
        assert!(t.stereo);
        assert_eq!(t.packing, FramePacking::Arbitrary);
        assert_eq!(t.info.mode, CodingMode::Celt);
        assert_eq!(t.info.bandwidth, Bandwidth::Narrowband);
        assert_eq!(t.frame_samples(), 120);
    }

    #[test]
    fn hybrid_fullband_configuration() {
        // Config 14 = hybrid, fullband, 10 ms (RFC 6716 Table 1).
        let t = Toc::parse(toc(14, false, 0));
        assert_eq!(t.info.mode, CodingMode::Hybrid);
        assert_eq!(t.info.bandwidth, Bandwidth::Fullband);
        assert_eq!(t.frame_samples(), 480);
    }

    #[test]
    fn celt_2_5ms_configuration() {
        // Config 28 = CELT fullband, 2.5 ms.
        let t = Toc::parse(toc(28, false, 0));
        assert_eq!(t.info.frame_duration_tenths_ms, 25);
        assert_eq!(t.frame_samples(), 120);
        assert_eq!(t.info.bandwidth, Bandwidth::Fullband);
    }

    #[test]
    fn empty_packet_is_rejected() {
        assert!(parse_packet(&[]).is_err());
    }

    #[test]
    fn single_frame_packet() {
        let data = [toc(0, false, 0), 1, 2, 3, 4];
        let p = parse_packet(&data).unwrap();
        assert_eq!(p.num_frames, 1);
        assert_eq!(p.frame(0), &[1, 2, 3, 4]);
    }

    #[test]
    fn two_equal_frames_must_have_even_length() {
        let ok = [toc(0, false, 1), 1, 2, 3, 4];
        assert!(parse_packet(&ok).is_ok());
        let bad = [toc(0, false, 1), 1, 2, 3];
        assert!(parse_packet(&bad).is_err());
    }

    #[test]
    fn two_vbr_frames() {
        // First frame 3 bytes, second frame 2 bytes.
        let data = [toc(0, false, 2), 3, 10, 11, 12, 20, 21];
        let p = parse_packet(&data).unwrap();
        assert_eq!(p.num_frames, 2);
        assert_eq!(p.frame(0), &[10, 11, 12]);
        assert_eq!(p.frame(1), &[20, 21]);
    }

    #[test]
    fn arbitrary_cbr_packet() {
        // 3 CBR frames of 2 bytes each, no padding:
        // [TOC][M=3][common size = 2][frame0][frame1][frame2]
        let data = [toc(0, false, 3), 0x03, 2, 1, 2, 3, 4, 5, 6];
        let p = parse_packet(&data).unwrap();
        assert_eq!(p.num_frames, 3);
        let frames: Vec<&[u8]> = p.frames().collect();
        assert_eq!(frames, vec![&[1, 2][..], &[3, 4][..], &[5, 6][..]]);
    }

    #[test]
    fn arbitrary_vbr_packet_with_padding() {
        // VBR + padding, 2 frames, first frame 1 byte, second frame 1 byte, 1 padding byte:
        // [TOC][M=2, VBR|padding flags][len0 = 1][padding len = 1][f0=0xaa][f1=0xbb][pad=0x00]
        let data = [
            toc(0, false, 3),
            0x80 | 0x40 | 0x02,
            1,    // length of frame 0
            1,    // total padding length
            0xaa, // frame 0
            0xbb, // frame 1 (implicit length = remaining data bytes)
            0x00, // one padding byte
        ];
        let p = parse_packet(&data).unwrap();
        assert_eq!(p.num_frames, 2);
        assert_eq!(p.frame(0), &[0xaa]);
        assert_eq!(p.frame(1), &[0xbb]);
    }

    #[test]
    fn frame_count_is_bounded() {
        // M = 0 is illegal.
        assert!(parse_packet(&[toc(0, false, 3), 0x00]).is_err());
    }

    #[test]
    fn oversized_frame_is_rejected() {
        // Two equal frames where each half exceeds 1275 bytes.
        let mut data = vec![toc(0, false, 1)];
        data.extend_from_slice(&[0u8; 2 * 1276]);
        assert!(parse_packet(&data).is_err());
    }

    #[test]
    fn trailing_garbage_is_rejected() {
        // CBR packet declaring 2 frames of 1 byte, with one extra trailing byte.
        let data = [toc(0, false, 3), 0x02, 1, 0xaa, 0xbb, 0xcc];
        assert!(parse_packet(&data).is_err());
    }
}
