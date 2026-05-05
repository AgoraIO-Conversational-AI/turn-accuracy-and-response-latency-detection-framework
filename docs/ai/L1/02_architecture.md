# L1 — Architecture

## System Design

```
┌─────────────────────────────────────────────────────────┐
│  Turn Playback Harness (FastAPI on :8000)                │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Server   │◄──►│ TurnManager  │───►│ AudioEngine  │  │
│  │ (WS+REST)│    │ (sequencing) │    │ (playback)   │  │
│  └──────────┘    └──────┬───────┘    └──────────────┘  │
│                          │                               │
│                   ┌──────▼───────┐                      │
│                   │  VadEngine   │                      │
│                   │ (amplitude)  │                      │
│                   └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
         │                    │                │
         ▼                    ▼                ▼
    BlackHole 2ch       BlackHole 16ch     Speakers
    (to browser mic)    (from sys output)  (monitoring)
```

## Data Flow

1. **TurnManager** loads turns from `out/turns_index.json`
2. For each turn, **AudioEngine** plays the WAV to BlackHole 2ch (browser mic) and speakers
3. Simultaneously, AudioEngine captures from BlackHole 16ch (system audio containing agent response)
4. Captured audio is resampled to 16kHz and fed to **VadEngine**
5. VadEngine detects speech start/end events
6. TurnManager computes TTFA (time from playback end to first speech detection)
7. Results are broadcast to the web UI via WebSocket

## Threading Model

- FastAPI runs on the main asyncio event loop
- Audio playback uses dedicated threads (one per output device)
- Audio capture uses a sounddevice callback thread → puts chunks into a thread-safe `queue.Queue`
- TurnManager polls the capture queue from async code, processes amplitude detection inline
- An `asyncio.Queue` bridges TurnManager event callbacks (from worker threads) to the WebSocket event dispatcher task

## State Machine

```
IDLE → PLAYING → WAITING_RESPONSE → RECORDING_RESPONSE → IDLE
                       │
                       └→ (timeout) → no_response → IDLE
```

## Key Design Decisions

- **Dual playback** to separate devices avoids feedback loops (agent hearing its own output)
- **Amplitude detection during playback** enables early barge-in detection with negative TTFA values
- **Simple RMS threshold** instead of ML-based VAD — the capture path is a clean digital signal (TTS over BlackHole) with no ambient noise
- **Thread-safe queue bridge** between audio callbacks and async WebSocket
