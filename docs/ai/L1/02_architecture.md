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
5. Before the poll loop, two anchors are resolved once: the real playback onset (the `<audio>` element's `playing` event — not the `.play()` call) and the prompt's speech-end offset (the index `speech_end_ms`, else a decoded-WAV RMS fallback via `decodeWavTiming()`/`OfflineAudioContext`). The poll loop then computes the single metric the instant the first `speech_start` fires and emits a `phase` event so the row repaints immediately:
   `ttfa_ms = firstInputSpeechWall − (playingWall + speech_end_ms)` — wall-clock latency from when the prompt's speech ends to when the agent begins. The onset wall is frozen at detection. Barge-in is derived from the same timeline (`ttfa_ms < 0`, or onset inside a hesitation gap). A turn whose anchor is not the genuine `playing` event (e.g. the 250 ms fallback fired) is measured but flagged `ttfa_reliable = false` and excluded from the aggregate. Then the loop waits for the post-response silence timeout before resolving.
6. Result POSTs to `/api/results/submit`; `TurnManager.ingest_browser_result()` stores it and broadcasts the same `turn_done` event server-mode would (with `source: "browser"`). Payload carries `ttfa_ms` + `ttfa_reliable` / `ttfa_anchor_source` / `ttfa_error` and the decode/timing diagnostics. `get_summary()` aggregates only eligible rows (done, not barge-in, reliable anchor, finite, ≥ 0) and reports `ttfa_count` + `ttfa_excluded_unreliable`.

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
