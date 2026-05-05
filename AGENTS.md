# Agent Orientation

This repo contains a framework for measuring Conversational AI agent response quality — specifically turn accuracy (barge-in detection) and response latency (TTFA).

## Quick Context

- **Language**: Python 3.12+
- **Framework**: FastAPI + WebSocket for real-time control
- **Audio**: sounddevice + BlackHole virtual audio on macOS
- **VAD**: Amplitude-based (RMS threshold) speech detection
- **Entry point**: `python -m src.harness` starts the server on port 8000

## Documentation

Progressive disclosure docs live in `docs/ai/`:

1. Start with [L0_repo_card.md](docs/ai/L0_repo_card.md) for a 30-second overview
2. Dive into [L1/](docs/ai/L1/) for setup, architecture, and code maps
3. See [L2/](docs/ai/L1/L2/) for deep dives on specific subsystems

## Key Patterns

- Relative imports within `src/harness/` (e.g., `from .audio_engine import ...`)
- `BASE_DIR = Path(__file__).resolve().parent.parent.parent` for repo root
- WebSocket events bridge thread-based audio callbacks to async FastAPI
- Turn results track TTFA, barge-in, and no-response states
