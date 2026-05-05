# L1 — Conventions

## Import Style

- Relative imports within packages: `from .audio_engine import AudioEngine`
- Standard library first, then third-party, then local

## Path Resolution

All modules that need the repo root use:
```python
BASE_DIR = Path(__file__).resolve().parent.parent.parent
```

This navigates: `file → harness/ → src/ → repo_root/`

## Naming

- Files: snake_case (`turn_manager.py`)
- Classes: PascalCase (`TurnManager`, `AudioEngine`)
- Constants: UPPER_SNAKE (`SAMPLE_RATE`, `VAD_RATE`)
- Enums: PascalCase class, UPPER values (`TurnState.IDLE`)

## Audio Standards

- Playback/capture: 48kHz (sounddevice native)
- VAD processing: 16kHz (downsampled from 48kHz capture)
- Output WAVs: 48kHz mono 16-bit PCM
- Block size: 1024 frames

## WebSocket Protocol

Events are JSON objects with a `type` field:
- Server → Client: `init`, `turn_start`, `waiting_response`, `response_detected`, `turn_done`, `run_complete`, `reset`, `stopped`, `source_changed`
- Client → Server: `action` field with values: `run_all`, `run_single`, `stop`, `reset`, `set_source`, `get_results`

## Error Handling

- Audio device not found: logs warning, skips that device
- WAV file missing: turn marked as `skipped`
- API key missing: provider skipped with message
- FFmpeg failure: individual turn skipped, continues with rest
