# Opus Packet Loss Concealment (PLC)

This document describes the packet loss concealment engine added to `symphonia-codec-opus`,
its state machine, memory and numerical bounds, and the integration contract used by real-time
voice and game-voice pipelines.

## 1. Goals and non-goals

Interactive audio transports (WebRTC, in-game voice, custom UDP) drop or corrupt packets. A
decoder that returns an error or hard silence on every gap produces audible dropouts and clicks.
The PLC engine instead synthesizes a plausible replacement frame and fades smoothly back to the
real signal when the stream recovers.

Design requirements:

* **Opt-in.** PLC is disabled by default. When disabled, the PLC engine is never allocated and
  the normal decode path is unchanged sample-for-sample.
* **Timeline preserving.** Every call renders exactly the number of samples the packet's TOC
  configuration declares, regardless of whether the packet arrived, was lost, or was illegal.
* **Bounded and allocation-free after construction.** All state is fixed-size; the per-frame loss
  and recovery paths never touch the heap.
* **Numerically safe.** Extrapolation can never produce `NaN`, infinities, or samples outside the
  `f32` PCM range `[-1.0, 1.0]`.
* **Handles edge cases.** First-frame loss, arbitrarily long loss bursts, illegal packets,
  channel/sample-rate switches, and explicit resets.

The full SILK/CELT bitstream synthesizer is upstream work-in-progress. This crate currently
provides the RFC 6716 packet parser/validator, the decoder front-end, state machine and the
complete PLC engine. The frame synthesizer is the single function replaced when the codec core
lands; nothing in the parser or PLC integration changes.

## 2. Module layout

| File | Responsibility |
| --- | --- |
| `src/packet.rs` | RFC 6716 §3 TOC/config table and packet parsing/validation (all four packing codes, oplint lengths, padding). No allocation. |
| `src/plc.rs` | The concealment engine: pitch search, Burg LPC analysis, gain model, LPC extrapolation, recovery cross-fade, per-channel bounded state. |
| `src/decoder.rs` | `OpusDecoder` implementing `AudioDecoder`, PLC opt-in options, lifecycle state, spec switching, timeline guarantees. |
| `tests/decoder_plc.rs` | Integration tests for single/consecutive loss, stereo, recovery, bad packets, switching, reset and the default regression contract. |

## 3. Concealment DSP

Per channel, the engine retains the most recent 120 ms of decoded PCM and three descriptors
refreshed on every good frame:

1. **Pitch (fundamental period).** A normalized square-difference search over lags of 32..960
   samples (1.5 kHz down to 50 Hz at 48 kHz). Near-silent input falls back to a neutral lag.
2. **LPC spectral envelope.** A 10th-order predictor estimated with the **Burg method** over a
   Hamming-windowed 40 ms analysis frame. Burg derives reflection coefficients directly from
   forward/backward prediction errors; each coefficient is guaranteed inside the unit circle
   (and clamped defensively), so the all-pole synthesis filter is always minimum-phase/stable,
   even for highly tonal input where an autocorrelation matrix becomes singular.
3. **Gain.** The RMS energy of the last good frame.

On a lost frame the engine builds an excitation seed from the residual of the last pitch period,
repeats it through the LPC synthesis filter, and applies a gain envelope:

* geometric attenuation of `0.975` per lost frame, then
* a linear fade to silence over frames 9..16 of a continuous burst.

A final energy-preservation step rescales the synthesized frame toward the target RMS so a
resonating filter cannot defeat the attenuation. Every sample passes a finiteness check and is
clamped to `[-1.0, 1.0]`.

On recovery, the first good frame is cross-faded against what the extrapolator would have
produced: a 5 ms linear ramp from pure extrapolation to the pure real signal removes the boundary
click before the good frame is absorbed into the predictor state.

## 4. Decoder state machine

The decoder is always in exactly one of three states.

```text
                         construct / reset
                               |
                               v
                 +-----------------------------+
                 |            Cold             |
                 |  no good frame observed yet  |
                 |  loss  -> silence (timed)    |
                 +-----------------------------+
                    |                       ^
        good packet |                       | reset()
        (PLC primed)|                       | channel/spec switch
                    v                       | (PLC state discarded)
                 +-----------------------------+
        +------->|           Running            |
        |        | last packet good, predictor  |
        |        | state fresh                   |
        |        +-----------------------------+
        |          |                       ^
        |   loss / | good packet           | good packet
        |   illegal| (recovery xfade)      | (normal)
        |   packet v                       |
        |        +-----------------------------+
        |        |         Concealing           |
        +--------| one+ packets lost/invalid;   |
                 | LPC extrapolation active     |
                 +-----------------------------+
```

### Transitions

| Current | Event | Next | Output |
| --- | --- | --- | --- |
| (any) | `reset()` | Cold | buffer cleared |
| Cold | good packet | Running | decoded frame; PLC primed |
| Cold | loss / illegal packet | Cold | silence, full declared duration |
| Running | good packet | Running | decoded frame |
| Running | loss / illegal packet | Concealing | extrapolated frame |
| Concealing | good packet | Running | cross-faded recovery frame |
| Concealing | further loss / illegal packet | Concealing | further extrapolated, decaying frame |
| (any) | channel-count switch (stereo bit flips) | Cold | spec rebuilt; all PLC state dropped |

### Where illegal packets go

* **PLC enabled** — the malformed packet is discarded and concealed as a loss of the same
  duration; `decode_ref` returns successfully with a full-length buffer.
* **PLC disabled (default)** — the original error is returned and the internal buffer is cleared,
  matching standard Symphonia decoder semantics.

A zero-length CELT frame is legal per RFC 6716 §3.2.1 and explicitly requests concealment even
though the packet itself arrived.

## 5. Timeline and configuration handling

* Output is always at the Opus internal rate of **48 kHz**, regardless of the container's
  advertised rate.
* Frame duration comes from the TOC configuration (2.5/5/10/20/40/60 ms) and the frame count from
  the packing code. `decode_loss(toc_byte)` matches the duration the lost packet would have had;
  `decode_loss_like_last()` reuses the previous TOC when the transport supplies none.
* A flipped stereo bit is treated as a bitstream discontinuity: the output `AudioSpec` and buffer
  are rebuilt, the PLC engine's channel state is reset, and the decoder returns to Cold.
* A packet whose declared duration is zero or exceeds the RFC maximum is rejected.

## 6. Memory and allocation bounds

All buffers are fixed-size, per channel:

| State | Size (f32) | Purpose |
| --- | --- | --- |
| `history` | 5 760 (120 ms) | recent PCM for pitch/LPC and the synthesis delay line |
| `lpc` | 10 | predictor coefficients |
| cross-fade scratch (in `Plc`) | 5 760 | extrapolated reference for one recovery frame |
| excitation seed | 960 (max pitch lag) | one residual period |

Stereo PLC state is asserted by test to stay under 256 KiB, and the loss/recovery paths are
asserted by a counting-allocator test to perform **zero** heap allocations per frame. The output
`AudioBuffer` is allocated once at construction and reused.

## 7. Numerical safety

* Non-finite samples arriving in a good frame are replaced with silence before they can enter
  predictor state.
* LPC reflection coefficients are clamped inside the unit circle; coefficients are checked for
  finiteness.
* The long-burst gain envelope and the target-RMS rescale bound extrapolated energy.
* Every emitted sample passes `sanitize()`: non-finite becomes `0.0`, otherwise it is clamped to
  `[-1.0, 1.0]`, so `NaN`, infinity and over-range clipping cannot reach the output.

## 8. Public API summary

```rust
// Explicit opt-in at construction.
let dec = OpusDecoder::try_new(
    &params,
    &OpusDecoderOptions::default().plc(true),
)?;

// Or toggle at runtime.
dec.enable_plc();
dec.disable_plc();

// A packet the transport reports as lost (TOC byte of the missing packet).
let audio = dec.decode_loss(toc_byte)?;

// Or conceal using the last known configuration.
let audio = dec.decode_loss_like_last()?;

// Received packets still flow through the standard trait method; illegal packets are
// concealed automatically when PLC is on.
let audio = dec.decode_ref(&packet)?;
```
