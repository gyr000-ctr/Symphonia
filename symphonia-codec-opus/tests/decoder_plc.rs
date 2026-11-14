// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Integration tests for the Opus decoder front-end and its PLC policy.
//!
//! These exercise the public API the way a real-time voice stack would: packet validation,
//! explicit loss signalling, illegal-packet handling, channel/sample-rate switching, resets and
//! the default (PLC-off) regression contract.

use symphonia_codec_opus::decoder::{OpusDecoder, OpusDecoderOptions, State};
use symphonia_codec_opus::packet::OPUS_SAMPLE_RATE;
use symphonia_codec_opus::Plc;

use symphonia_core::audio::{Audio, Channels, GenericAudioBufferRef, Position};
use symphonia_core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoder};
use symphonia_core::packet::PacketRef;
use symphonia_core::units::{Duration, Timestamp};

/// Build mono codec parameters as an OGG/ISOBMFF mapper would provide them.
fn mono_params() -> AudioCodecParameters {
    let mut p = AudioCodecParameters::new();
    p.for_codec(CODEC_ID_OPUS)
        .with_sample_rate(OPUS_SAMPLE_RATE)
        .with_channels(Channels::Positioned(Position::FRONT_CENTER));
    p
}

/// Build stereo codec parameters.
fn stereo_params() -> AudioCodecParameters {
    let mut p = AudioCodecParameters::new();
    p.for_codec(CODEC_ID_OPUS)
        .with_sample_rate(OPUS_SAMPLE_RATE)
        .with_channels(Channels::Positioned(
            Position::FRONT_LEFT | Position::FRONT_RIGHT,
        ));
    p
}

/// A minimal legal mono 20 ms SILK packet: TOC config 9, packing 0, plus a non-empty payload.
fn good_packet_20ms() -> Vec<u8> {
    // Config 9 (SILK WB, 20 ms), stereo = 0, code c = 0 (single frame).
    vec![(9u8) << 3, 0x00, 0x11, 0x22, 0x33]
}

/// A 20 ms stereo CELT packet (config 29 = CELT FB 10 ms is avoided; use config 9 with stereo bit).
fn good_stereo_packet_20ms() -> Vec<u8> {
    // Config 9 with the stereo flag set.
    vec![(9u8) << 3 | 0x04, 0x00, 0x11, 0x22]
}

fn packet_ref(data: &[u8]) -> PacketRef<'_> {
    PacketRef::new(0, Timestamp::new(0), Duration::new(960), data)
}

/// Extract the f32 plane slice from a generic buffer reference.
fn plane<'a>(buf: &'a GenericAudioBufferRef<'_>, idx: usize) -> &'a [f32] {
    match buf {
        GenericAudioBufferRef::F32(b) => b.plane(idx).expect("plane exists"),
        _ => panic!("expected f32 buffer"),
    }
}

fn frames_of(buf: &GenericAudioBufferRef<'_>) -> usize {
    match buf {
        GenericAudioBufferRef::F32(b) => b.frames(),
        _ => panic!("expected f32 buffer"),
    }
}

fn plane_rms(buf: &GenericAudioBufferRef<'_>, idx: usize) -> f32 {
    let p = plane(buf, idx);
    let sum: f64 = p.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / p.len() as f64).sqrt() as f32
}

fn plane_is_finite_and_unclipped(buf: &GenericAudioBufferRef<'_>, idx: usize) -> bool {
    plane(buf, idx).iter().all(|s| s.is_finite() && s.abs() <= 1.0)
}

#[test]
fn plc_is_disabled_by_default() {
    let dec = OpusDecoder::try_new(&mono_params(), &OpusDecoderOptions::default()).unwrap();
    assert!(!dec.is_plc_enabled());
}

#[test]
fn default_path_rejects_illegal_packets_with_error() {
    let mut dec =
        OpusDecoder::try_new(&mono_params(), &OpusDecoderOptions::default()).unwrap();

    // Empty payload is structurally illegal and must surface an error when PLC is off.
    let err = dec.decode_ref(&packet_ref(&[]));
    assert!(err.is_err());

    // The internal buffer must be cleared after an error per the AudioDecoder contract.
    assert!(matches!(dec.last_decoded(), GenericAudioBufferRef::F32(b) if b.frames() == 0));
}

#[test]
fn default_path_is_deterministic_across_runs() {
    // Regression guard: two PLC-off decoders fed identical packets must produce byte-identical
    // output, and enabling PLC must not alter the decode of packets that never hit a loss.
    let data = good_packet_20ms();

    let mut a = OpusDecoder::try_new(&mono_params(), &OpusDecoderOptions::default()).unwrap();
    let mut b = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    let out_a = a.decode_ref(&packet_ref(&data)).unwrap();
    let out_b = b.decode_ref(&packet_ref(&data)).unwrap();

    let pa = plane(&out_a, 0);
    let pb = plane(&out_b, 0);
    assert_eq!(pa.len(), pb.len());
    assert!(pa.iter().zip(pb.iter()).all(|(x, y)| x.to_bits() == y.to_bits()));
}

#[test]
fn first_frame_loss_emits_silence_but_keeps_timeline() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    // No good packet has been seen; a 20 ms loss must render 960 silent samples.
    let out = dec.decode_loss(9u8 << 3).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert_eq!(plane_rms(&out, 0), 0.0);
    assert_eq!(dec.state(), State::Cold);
}

#[test]
fn single_loss_after_good_packets_is_concealed() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    // Prime the predictor directly through the PLC engine (the bitstream core renders silence
    // until the upstream SILK/CELT implementation lands; PLC DSP is what is under test here).
    // A sequence of good packets still exercises the full packet-validated front-end.
    for _ in 0..3 {
        let data = good_packet_20ms();
        dec.decode_ref(&packet_ref(&data)).unwrap();
    }
    assert_eq!(dec.state(), State::Running);

    let out = dec.decode_loss(9u8 << 3).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert!(plane_is_finite_and_unclipped(&out, 0));
}

#[test]
fn consecutive_losses_stay_timed_and_eventually_silence() {
    let mut engine = Plc::new(1);

    // Prime with six 20 ms tone frames.
    for f in 0..6 {
        let mut frame = vec![0.0f32; 960];
        for (i, s) in frame.iter_mut().enumerate() {
            let t = (f * 960 + i) as f32 / 48_000.0;
            *s = 0.5 * (2.0 * core::f32::consts::PI * 220.0 * t).sin();
        }
        engine.good_frame(&mut [&mut frame]);
    }

    let mut last_rms = f32::MAX;
    for burst in 0..18 {
        let mut out = vec![0.0f32; 960];
        engine.lost_frame(&mut [&mut out]);
        assert_eq!(out.len(), 960, "loss {burst} changed the timeline");
        assert!(out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        let r = out.iter().map(|s| s * s).sum::<f32>().sqrt() / (960f32).sqrt();
        assert!(r <= last_rms + 1e-6, "energy increased at loss {burst}");
        last_rms = r;
    }
    // Long-burst fade reaches silence.
    assert!(last_rms < 1e-4);
}

#[test]
fn recovery_crossfades_and_returns_to_running() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    for _ in 0..3 {
        dec.decode_ref(&packet_ref(&good_packet_20ms())).unwrap();
    }
    dec.decode_loss(9u8 << 3).unwrap();
    assert_eq!(dec.state(), State::Concealing);

    let recovered = dec.decode_ref(&packet_ref(&good_packet_20ms())).unwrap();
    assert_eq!(frames_of(&recovered), 960);
    assert!(plane_is_finite_and_unclipped(&recovered, 0));
    // The returned buffer borrows `dec`, so observe the state after the buffer's last use.
    assert_eq!(dec.state(), State::Running);
}

#[test]
fn illegal_packet_with_plc_is_concealed_not_errored() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    for _ in 0..3 {
        dec.decode_ref(&packet_ref(&good_packet_20ms())).unwrap();
    }

    // A malformed packet (declares two equal frames but has an odd payload length) must be
    // concealed: success return, full 20 ms duration rendered.
    let bad = vec![(9u8) << 3 | 0x01, 0xAA, 0xBB, 0xCC];
    let out = dec.decode_ref(&packet_ref(&bad)).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert!(plane_is_finite_and_unclipped(&out, 0));
}

#[test]
fn stereo_loss_renders_two_planes() {
    let mut dec = OpusDecoder::try_new(
        &stereo_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    for _ in 0..3 {
        dec.decode_ref(&packet_ref(&good_stereo_packet_20ms())).unwrap();
    }

    let out = dec.decode_loss((9u8) << 3 | 0x04).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert!(plane_is_finite_and_unclipped(&out, 0));
    assert!(plane_is_finite_and_unclipped(&out, 1));
}

#[test]
fn channel_switch_resets_concealment_state() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    for _ in 0..3 {
        dec.decode_ref(&packet_ref(&good_packet_20ms())).unwrap();
    }
    assert_eq!(dec.state(), State::Running);

    // A stereo TOC flips the output specification and discards PLC state.
    let out = dec.decode_ref(&packet_ref(&good_stereo_packet_20ms())).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert_eq!(dec.state(), State::Cold);
    assert_eq!(dec.codec_params().channels, stereo_params().channels);
}

#[test]
fn output_is_always_48khz() {
    // Container parameters may claim another rate; Opus output is defined at 48 kHz regardless.
    let mut params = mono_params();
    params.sample_rate = Some(16_000);
    let dec = OpusDecoder::try_new(&params, &OpusDecoderOptions::default()).unwrap();
    assert_eq!(dec.codec_params().sample_rate, Some(OPUS_SAMPLE_RATE));
}

#[test]
fn reset_returns_decoder_to_cold_and_clears_buffer() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    for _ in 0..3 {
        dec.decode_ref(&packet_ref(&good_packet_20ms())).unwrap();
    }
    dec.reset();
    assert_eq!(dec.state(), State::Cold);
    assert!(matches!(dec.last_decoded(), GenericAudioBufferRef::F32(b) if b.frames() == 0));

    // After reset, loss is silent again.
    let out = dec.decode_loss(9u8 << 3).unwrap();
    assert_eq!(plane_rms(&out, 0), 0.0);
}

#[test]
fn different_frame_lengths_keep_packet_timeline() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    // Config 16 = CELT NB 2.5 ms (120 samples); config 19 = CELT NB 20 ms (960 samples).
    let short = dec.decode_loss(16u8 << 3).unwrap();
    assert_eq!(frames_of(&short), 120);

    let long = dec.decode_loss(19u8 << 3).unwrap();
    assert_eq!(frames_of(&long), 960);
}

#[test]
fn explicit_disable_releases_plc_and_restores_error_semantics() {
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();
    assert!(dec.is_plc_enabled());

    dec.disable_plc();
    assert!(!dec.is_plc_enabled());
    assert!(dec.decode_ref(&packet_ref(&[])).is_err());
}

/// A global allocator that counts allocations only while a thread-local flag is armed, so the rest
/// of the test harness is unaffected.
struct CountingAlloc;

std::thread_local! {
    static COUNTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ALLOC_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOC_COUNT.with(|c| c.set(c.get() + 1));
        }
        // SAFETY: delegates to the system allocator with the unchanged caller-provided layout.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: the pointer and layout originate from a matching `alloc` above.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

#[test]
fn loss_and_recovery_paths_allocate_nothing_per_frame() {
    // Run in a dedicated thread so the global counting flag cannot leak into other tests.
    std::thread::spawn(|| {
        let mut engine = Plc::new(2);

        // Prime with six stereo 20 ms tone frames.
        for f in 0..6 {
            let mut left = vec![0.0f32; 960];
            let mut right = vec![0.0f32; 960];
            for i in 0..960 {
                let t = (f * 960 + i) as f32 / 48_000.0;
                left[i] = 0.4 * (2.0 * core::f32::consts::PI * 200.0 * t).sin();
                right[i] = 0.5 * (2.0 * core::f32::consts::PI * 320.0 * t).sin();
            }
            engine.good_frame(&mut [&mut left, &mut right]);
        }

        // All output buffers are allocated once, outside the counting region, and reused.
        let mut left = vec![0.0f32; 960];
        let mut right = vec![0.0f32; 960];

        COUNTING.with(|c| c.set(true));
        ALLOC_COUNT.with(|c| c.set(0));

        // A long burst of consecutive losses followed by a recovery frame.
        for _ in 0..20 {
            engine.lost_frame(&mut [&mut left, &mut right]);
        }
        left.fill(0.1);
        right.fill(0.1);
        engine.good_frame(&mut [&mut left, &mut right]);

        let observed = ALLOC_COUNT.with(|c| c.get());
        COUNTING.with(|c| c.set(false));
        assert_eq!(observed, 0, "the PLC loss/recovery path performed a heap allocation");
    })
    .join()
    .expect("plc allocation test thread panicked");
}

#[test]
fn bad_packet_before_any_good_frame_is_silent_but_timed() {
    // PLC enabled, but no good frame has primed the engine. An illegal packet must still render a
    // full-duration silent buffer rather than erroring, preserving the timeline.
    let mut dec = OpusDecoder::try_new(
        &mono_params(),
        &OpusDecoderOptions::default().plc(true),
    )
    .unwrap();

    // Two-equal packing (code 1) with an odd payload length is illegal.
    let bad = vec![(9u8) << 3 | 0x01, 0xAA, 0xBB, 0xCC];
    let out = dec.decode_ref(&packet_ref(&bad)).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert_eq!(plane_rms(&out, 0), 0.0);
    assert_eq!(dec.state(), State::Cold);
}

#[test]
fn very_long_loss_burst_reaches_silence_without_changing_timeline() {
    let mut engine = Plc::new(1);

    // Prime with six 20 ms tone frames.
    for f in 0..6 {
        let mut frame = vec![0.0f32; 960];
        for (i, s) in frame.iter_mut().enumerate() {
            let t = (f * 960 + i) as f32 / 48_000.0;
            *s = 0.5 * (2.0 * core::f32::consts::PI * 180.0 * t).sin();
        }
        engine.good_frame(&mut [&mut frame]);
    }

    // 40 consecutive lost frames (~800 ms), well past the fade-out window.
    let mut out = vec![0.0f32; 960];
    for _ in 0..40 {
        engine.lost_frame(&mut [&mut out]);
        assert_eq!(out.len(), 960);
        assert!(out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }
    // The burst must end in exact silence, not NaN, and the frame length is unchanged.
    assert!(out.iter().all(|s| *s == 0.0));
}

#[test]
fn loss_with_plc_disabled_is_silent_but_timed() {
    let mut dec =
        OpusDecoder::try_new(&mono_params(), &OpusDecoderOptions::default()).unwrap();

    // Even without any PLC engine, an explicitly signalled loss renders silence at full length.
    let out = dec.decode_loss(9u8 << 3).unwrap();
    assert_eq!(frames_of(&out), 960);
    assert_eq!(plane_rms(&out, 0), 0.0);
}
