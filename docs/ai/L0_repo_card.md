# L0 — Repo Card

## What This Is

A framework for measuring Conversational AI agent quality through automated turn playback and response detection. Plays pre-segmented conversation turns through virtual audio devices into any web-based ConvAI agent, then uses amplitude detection to identify and measure the agent's response.

## Key Metrics Measured

- **TTFA** (Time to First Audio) — latency from end of user turn to agent speech
- **Barge-in** — agent interrupts during mid-turn hesitation gaps
- **No Response** — agent fails to respond within timeout

## Tech Stack

Python 3.12+ | FastAPI + WebSocket | RMS amplitude detection | BlackHole virtual audio (macOS) | Two audio loops: server-side `sounddevice` and browser-side Web Audio (`HTMLMediaElement.setSinkId` + `getUserMedia` + `ScriptProcessor` VAD).

## Entry Points

- `python -m src.harness` — start measurement server on :8000. Hosts the UI + ingests results. Audio loop can run server-side (sounddevice) OR entirely in the operator's browser (default, see `static/browser_harness.js`).
- `python -m src.harness.segment <audio>` — segment audio into turns
- `python -m src.harness.generate_tts` — generate synthetic TTS turns via ElevenLabs
- `python -m src.diarization.compare_providers --audio <file>` — compare diarization providers

## Deeper Docs

- [L1/ Setup](L1/01_setup.md) — environment, dependencies, audio routing
- [L1/ Architecture](L1/02_architecture.md) — system design and data flow
- [L1/ Code Map](L1/03_code_map.md) — file-by-file guide
