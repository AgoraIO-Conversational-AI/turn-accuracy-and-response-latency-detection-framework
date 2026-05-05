# L1 — Gotchas

## Audio Routing Feedback Loop

If you use Multi-Output Device for speaker monitoring (instead of MacBook Pro Speakers directly), the turn playback will be captured on BlackHole 16ch and trigger VAD — causing false barge-in detections.

**Fix**: AudioEngine auto-detects "MacBook Pro Speakers" specifically for monitoring output.

## TTFA Systematic Bias

There's a ~50-80ms positive bias in TTFA measurements due to:
- Audio buffer latency (1024 frames @ 48kHz = 21ms)
- VAD chunk processing (512 samples @ 16kHz = 32ms)
- Queue/event loop polling (10-20ms)

This is documented and consistent — useful for relative comparisons between turns/agents.

## Negative TTFA

A negative TTFA means the agent started speaking before the turn playback finished. This is classified as barge-in. It can happen when:
- The agent interrupts during a hesitation gap
- The agent has very low latency and responds to partial input

## BlackHole Channel Mismatch

BlackHole 16ch has 16 channels but agent audio only appears on channel 0. The capture callback always reads `indata[:, 0]` to get mono.

## Turn File Not Found

If `out/turns/` WAVs don't exist, you need to run segmentation first:
```bash
python -m src.harness.segment fixtures/sovereign_place_5.m4a
```

The server will mark turns as "skipped" if WAV files are missing.

## VadEngine Threshold Tuning

The default RMS threshold (0.01) works well for clean digital audio paths. If you get false detections or missed responses, adjust via `VadEngine(threshold=0.02)` in server.py startup.

## macOS Permissions

sounddevice needs microphone permission. If capture fails silently, check System Settings → Privacy & Security → Microphone → Terminal/IDE.
