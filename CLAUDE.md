# Claude Code Guidance

## Project Overview

Turn accuracy and response latency detection framework for Conversational AI agents. Plays pre-segmented conversation turns through virtual audio devices, monitors agent responses via VAD, and measures TTFA/barge-in/no-response metrics.

## Running

```bash
python -m src.harness          # start FastAPI server on :8000
python -m src.harness.segment fixtures/sovereign_place_5.m4a  # segment audio
python -m src.diarization.compare_providers --audio <file> --split
```

## Code Layout

- `src/harness/` — main application (FastAPI server, audio engine, VAD, turn manager)
- `src/diarization/` — standalone provider comparison tool
- `fixtures/` — source audio files (.m4a)
- `out/` — generated turn WAVs and indexes (gitignored)

## Import Conventions

- Within `src/harness/`: use relative imports (`from .audio_engine import ...`)
- `BASE_DIR = Path(__file__).resolve().parent.parent.parent` navigates to repo root

## Key Types

- `TurnResult` — dataclass with turn, speaker, ttfa_ms, barge_in, status
- `TurnState` — enum: idle, playing, waiting_response, recording_response
- `AudioEngine` — manages BlackHole 2ch (output), speakers, BlackHole 16ch (capture)
- `VadEngine` — RMS amplitude-based speech detector
- `TurnManager` — orchestrates playback → VAD → measurement cycle

## Testing

No test suite yet. Verify imports with:
```bash
python -c "from src.harness.server import app"
python -c "from src.diarization.compare_providers import load_env"
```

## Dependencies

System: ffmpeg, BlackHole 2ch + 16ch virtual audio drivers (macOS)
Python: see requirements.txt (fastapi, uvicorn, sounddevice, numpy, scipy, soundfile, requests)
