# L1 — Setup

## System Requirements

- macOS (required for BlackHole virtual audio drivers)
- Python 3.12+
- ffmpeg (`brew install ffmpeg`)
- BlackHole 2ch + 16ch (`brew install blackhole-2ch blackhole-16ch`, reboot required)

## Python Dependencies

```bash
pip install -r requirements.txt
```

Core packages: fastapi, uvicorn, sounddevice, soundfile, numpy, scipy, requests

## Audio Routing (one-time macOS setup)

1. Install BlackHole drivers and reboot
2. Open Audio MIDI Setup (Applications → Utilities)
3. Create a Multi-Output Device combining: BlackHole 16ch + MacBook Pro Speakers
4. System Settings → Sound → Output → Multi-Output Device
5. In browser: set microphone input to BlackHole 2ch

This routing allows the harness to:
- Play turns into the browser via BlackHole 2ch (appears as mic input)
- Play turns to speakers for monitoring
- Capture agent responses from BlackHole 16ch (system audio output)

## Environment Variables

Copy `.env.example` to `.env` and fill in API keys (only needed for diarization comparison):
- `DEEPGRAM_API_KEY`
- `SONIOX_API_KEY`
- `SPEECHMATICS_API_KEY`

## First Run

```bash
# Segment source audio into turns
python -m src.harness.segment fixtures/sovereign_place_5.m4a

# Start server
python -m src.harness
# Open http://localhost:8000
```
