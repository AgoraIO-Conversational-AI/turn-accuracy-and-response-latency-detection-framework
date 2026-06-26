# L1 — Architecture

There are two runtime modes that produce the same `TurnResult` records; the choice is about where the audio loop runs, not what it measures.

## Browser Mode (default)

```
Server host (any OS)                     Operator's Mac (Chromium)
┌──────────────────────────────┐         ┌─────────────────────────────────┐
│ FastAPI on :8000             │         │ static/browser_harness.js       │
│  ┌────────┐  ┌─────────────┐ │  HTTP   │  ┌──────────────────────────┐  │
│  │ static │  │ TurnManager │◄│◄──────► │  │ runBrowserTurn()          │  │
│  │ WAV    │  │ ingest_     │ │  /api/  │  │  ├─ <audio>.setSinkId →   │  │
│  │ index  │  │ browser_    │ │  /ws    │  │  │   BlackHole 2ch         │  │
│  └────────┘  │ result()    │ │         │  │  │   (+ optional monitor) │  │
│              └─────────────┘ │         │  │  └─ getUserMedia({input}) │  │
│                              │         │  │     → AudioContext@16k    │  │
│                              │         │  │     → ScriptProcessor 512 │  │
│                              │         │  │     → BrowserVad (RMS)    │  │
│                              │         │  └──────────────────────────┘  │
└──────────────────────────────┘         └─────────────────────────────────┘
                                                       │
                                                       ▼
                                            BlackHole 16ch ◄─ Mac system output
                                            (loopback from agent tab)
```

1. Browser fetches `/api/wav/{source}/{speaker}/{turn}` for the next turn's WAV.
2. `runBrowserTurn()` opens a 16 kHz `AudioContext`, calls `getUserMedia` on the chosen Input device, wires a `ScriptProcessor` (512 samples = 32 ms VAD frame) through a gain-0 sink so nothing leaks to the speakers.
3. Two `HTMLMediaElement`s play the WAV — one to `devices.output` via `setSinkId` (the BlackHole 2ch loopback into the agent tab's mic), one optional monitor.
4. `BrowserVad` (same state machine as Python `VadEngine`, browser threshold RMS ≥ 0.003, 300 ms hangover) detects `speech_start` / `speech_end` events.
5. Poll loop computes TTFA the instant the first `speech_start` fires; emits a `phase` event so the row repaints immediately, then waits for the post-response silence timeout before resolving. A second amplitude-based measurement `TTFA2` runs in parallel: `decodeWavTiming()` decodes the prompt WAV via an `OfflineAudioContext` to find the last-active-RMS sample (`last_output_speech_ms`), and a `playing`-event timestamp (with a 250 ms `play()`-call fallback) anchors it to wall-clock. `ttfa2_ms = firstInputSpeechWall - (playStartWall + last_output_speech_ms)`. Phase A: TTFA stays canonical, TTFA2 is logged + shown as a parallel column for diff comparison; canonical flip happens in a later commit only after enough runs confirm the diff is a stable systematic offset.
6. Result POSTs to `/api/results/submit`; `TurnManager.ingest_browser_result()` stores it and broadcasts the same `turn_done` event server-mode would (with `source: "browser"`). Payload now carries TTFA2 + 11 diagnostic fields alongside the legacy TTFA.

`AbortSignal` is plumbed through Stop / Reset / new-Run-All so an in-flight turn tears down playback + capture cleanly instead of writing into the next run's row.

## Server Mode

The original system — runs entirely on the operator's Mac. The page just shells out to the server-side audio engine over WebSocket.

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

1. **TurnManager** loads turns from the active source's `turns_index.json`
2. For each turn, **AudioEngine** plays the WAV to BlackHole 2ch (browser mic) and speakers
3. Simultaneously, AudioEngine captures from BlackHole 16ch (system audio containing agent response)
4. Captured audio is resampled to 16kHz and fed to **VadEngine**
5. VadEngine detects speech start/end events
6. TurnManager computes TTFA using the turn's expected `duration_ms` as the reference point (not wall-clock playback completion, which has ~200-400ms OS buffer overhead)
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
