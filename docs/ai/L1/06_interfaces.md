# L1 — Interfaces

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/devices` | List audio devices and active assignments (server-side, sounddevice). Browser-mode UI ignores this — see browser harness. |
| GET | `/api/turns?speaker=N` | Get turn list (optional speaker filter) |
| GET | `/api/sources` | Get available audio sources |
| POST | `/api/sources/{key}` | Switch active source |
| GET | `/api/results` | Get current run summary |
| POST | `/api/devices/configure` | Update server-side device assignments |
| GET | `/api/wav/{source}/{speaker}/{turn_id}` | Serve a single turn WAV (audio/wav). Used by the browser harness for in-browser playback; also handy for previewing without an audio engine. |
| POST | `/api/results/submit` | Ingest a browser-measured turn result. Body: `{turn, ttfa_ms, barge_in, barge_in_at_ms, response_duration_ms, status, speaker}`. Server stores it in `TurnManager.run.results` so the existing summary stats + `turn_done` broadcast keep working. |

## WebSocket (`/ws`)

### Server → Client Events

```json
{"type": "init", "devices": {...}, "state": "idle"}
{"type": "turn_start", "turn": 0, "speaker": 0, "text": "...", "duration_ms": 2500}
{"type": "waiting_response", "turn": 0}
{"type": "response_detected", "turn": 0, "ttfa_ms": 342.5}
{"type": "turn_done", "turn": 0, "ttfa_ms": 342.5, "barge_in": false, "status": "done", "summary": {...}, "source": "browser"}
{"type": "run_complete", "total_turns": 74, "completed": 70, ...}
{"type": "reset"}
{"type": "stopped"}
{"type": "source_changed", "sources": [...]}
```

### Client → Server Actions

```json
{"action": "run_all", "speaker": 0}
{"action": "run_single", "turn": 5}
{"action": "stop"}
{"action": "reset"}
{"action": "set_source", "source": "sovereign10"}
{"action": "get_results"}
```

## Key Data Structures

### TurnResult

```python
@dataclass
class TurnResult:
    turn: int
    speaker: int
    text: str
    duration_ms: int
    ttfa_ms: float | None
    barge_in: bool
    barge_in_at_ms: float | None
    response_duration_ms: float | None
    status: str  # pending, playing, done, barge_in, no_response, skipped
```

### turns_index.json

```json
{
  "audio_file": "sovereign_place_5.m4a",
  "total_turns": 148,
  "turns": [
    {
      "turn": 0,
      "speaker": 0,
      "start_ms": 1200,
      "end_ms": 4500,
      "duration_ms": 3300,
      "text": "...",
      "hesitations": [{"at_ms": 2000, "duration_ms": 800}],
      "max_hesitation_ms": 800,
      "category": "pause",
      "expected_complete": false,
      "target_gap_ms": 800,
      "voice_id": "XnKbmWxx8uWjruHkpXmf"
    }
  ]
}
```

TTS-specific fields (`category`, `expected_complete`, `target_gap_ms`, `voice_id`) are present in TTS turn indexes. Segmented audio indexes omit them — the harness handles both formats.

Categories: `normal`, `pause`, `hesitation`, `hesitation2`, `ambiguous`. The `hesitation2` category uses prosody-only pauses (ElevenLabs `[hesitation]` tags) without filler words.

## Browser harness (`static/browser_harness.js`)

ES module imported dynamically by the page. Drives audio entirely in the user's browser via Web Audio API + `HTMLMediaElement.setSinkId`. Replaces the server-side `AudioEngine` / `VadEngine` / per-turn loop in `TurnManager.run_single_turn` so a remote operator's Mac (with BlackHole + mic + speakers) does the actual measurement while the server just hosts the UI and ingests results.

### Exports

```js
primePermission()                     // request mic permission once so enumerateDevices() returns labels
listDevices()                         // {inputs, outputs} from enumerateDevices()
canSelectOutputDevice()               // true if HTMLMediaElement.setSinkId is supported (Chromium only)
runBrowserTurn(turn, sourceKey, devices, baseUrl, onPhase, signal)
                                      // play one turn through devices.output (+ optional devices.monitor),
                                      // capture from devices.input, return a result payload matching POST /api/results/submit.
                                      // signal: AbortSignal — Stop/Reset/new run aborts cleanly.
                                      // onPhase(phaseName, partialResult?) fires on capture_starting, playing,
                                      //   response_detected, barge_in, done — so the UI can paint immediately.
startMeter(inputDeviceId, onLevel)    // continuous RMS meter for an input device; onLevel({rms, peak}) every frame.
submitResult(result, baseUrl)         // POST /api/results/submit
```

### VAD constants (mirror Python `VadEngine`)

```
VAD_RATE = 16000
VAD_FRAME = 512                       // 32 ms @ 16 kHz
VAD_RMS_THRESHOLD = 0.01
VAD_MIN_SILENCE_MS = 300
RESPONSE_SILENCE_TIMEOUT_S = 1.5
BARGE_IN_SILENCE_TIMEOUT_S = 3.0
MAX_WAIT_FOR_RESPONSE_S = 10.0
```

The browser AudioContext is requested at 16 kHz so the ScriptProcessor frames are exactly one VAD frame; Chromium honors this hint, Safari/Firefox silently override — that's why the setup docs require Chromium.
```
