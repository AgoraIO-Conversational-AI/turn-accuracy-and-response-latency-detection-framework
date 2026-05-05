# L2 — Speech Detection Tuning

## Amplitude Detector Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `threshold` | 0.01 | RMS level above which audio is classified as speech |
| `min_silence_ms` | 300 | Minimum silence duration to trigger speech_end |

## How It Works

The VadEngine processes 512 samples at 16kHz (32ms frames) and computes the RMS amplitude of each frame:

1. Buffers incoming audio until 512 samples are available
2. Computes `rms = sqrt(mean(frame ** 2))`
3. If RMS >= threshold and not already in speech state → emit `speech_start`
4. If RMS < threshold while in speech state → accumulate silence samples
5. If accumulated silence >= min_silence_samples → emit `speech_end`

## Why Amplitude Instead of ML-based VAD

The capture path is a clean digital signal: TTS audio from the agent routed through BlackHole virtual audio. There is no:
- Ambient noise
- Background music
- Multiple simultaneous speakers
- Non-speech audio events

In this environment, any audio above a low amplitude floor is speech. An ML model (like Silero VAD) adds unnecessary complexity, download dependencies, and latency for no accuracy benefit.

## Threshold Tuning

- **Lower threshold (0.005)**: Catches quieter agent responses, but may trigger on digital silence floor artifacts
- **Higher threshold (0.02-0.05)**: More robust against any low-level noise, but may miss the very start of soft speech
- **Default 0.01**: Good balance for typical TTS output levels over BlackHole

## Silence Duration

`min_silence_ms = 300` means the detector won't report speech_end until 300ms of continuous silence. This prevents:
- Brief pauses within a response from being treated as end-of-response
- The TurnManager's silence timeout from triggering prematurely

The TurnManager adds its own `response_silence_timeout` (3s) on top.

## Audio-Time vs Wall-Clock

The detector emits sample-accurate timestamps (`time_s = total_samples / 16000`). These represent audio-domain time since the last `reset()` call, which occurs just before playback starts. The TurnManager uses these audio-time values for TTFA calculation, avoiding event-loop scheduling jitter.

## Latency Characteristics

| Stage | Latency |
|-------|---------|
| System audio → BlackHole 16ch | <1ms (kernel) |
| InputStream buffer | ~21ms (1024 frames @ 48kHz) |
| Queue transfer | <1ms |
| Decimate 48→16kHz (3:1) | <0.1ms |
| Accumulate 512 samples @ 16kHz | up to 32ms |
| RMS computation | <0.1ms |
| **Total systematic bias** | **~55ms** |

## Debugging

The capture callback logs RMS every ~1 second — useful for verifying audio is flowing and at what level. The speech detector logs every speech_start/speech_end event with its audio-time offset.
