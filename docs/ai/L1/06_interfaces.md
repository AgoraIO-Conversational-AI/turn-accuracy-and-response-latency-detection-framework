# L1 — Interfaces

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/devices` | List audio devices and active assignments |
| GET | `/api/turns?speaker=N` | Get turn list (optional speaker filter) |
| GET | `/api/sources` | Get available audio sources |
| POST | `/api/sources/{key}` | Switch active source |
| GET | `/api/results` | Get current run summary |
| POST | `/api/devices/configure` | Update device assignments |

## WebSocket (`/ws`)

### Server → Client Events

```json
{"type": "init", "devices": {...}, "state": "idle"}
{"type": "turn_start", "turn": 0, "speaker": 0, "text": "...", "duration_ms": 2500}
{"type": "waiting_response", "turn": 0}
{"type": "response_detected", "turn": 0, "ttfa_ms": 342.5}
{"type": "turn_done", "turn": 0, "ttfa_ms": 342.5, "barge_in": false, "status": "done", "summary": {...}}
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
      "max_hesitation_ms": 800
    }
  ]
}
```
