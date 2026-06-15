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

### 1. Install BlackHole virtual audio drivers

```bash
brew install blackhole-2ch blackhole-16ch
```

Reboot after installation (required for audio drivers to register).

### 2. Create Multi-Output Device in Audio MIDI Setup

1. Open **Audio MIDI Setup** (Finder → Applications → Utilities → Audio MIDI Setup)
2. Click the **+** button at the bottom-left → **Create Multi-Output Device**
3. In the right panel, check these devices:
   - **BlackHole 16ch** (must be checked — this is how the harness captures agent audio)
   - **MacBook Pro Speakers** (check this so you can hear the agent's responses)
4. Optionally rename it (double-click the name) to something like "Harness Output"

### 3. Set macOS system output

- System Settings → Sound → Output → select the **Multi-Output Device** you just created
- This routes all system audio (including the ConvAI agent's voice) through BlackHole 16ch for capture, and through speakers so you can hear it

### 4. Set browser microphone

- In the browser tab running the ConvAI agent, set the microphone input to **BlackHole 2ch**
- The harness plays turn audio into BlackHole 2ch, which the browser picks up as mic input

### How it works

```
Harness ──play──► BlackHole 2ch ──mic──► Browser (ConvAI agent)
                                              │
                                         agent speaks
                                              │
                                              ▼
                     Multi-Output Device (system audio out)
                         ├── BlackHole 16ch ──capture──► Harness (VAD)
                         └── MacBook Pro Speakers (monitoring)
```

**Important**: The harness plays monitoring audio directly to MacBook Pro Speakers (not through Multi-Output). If it played through Multi-Output, the turn playback would appear on BlackHole 16ch and trigger false barge-in detections.

## Environment Variables

Copy `.env.example` to `.env` and fill in API keys:
- `TTS_KEY` — ElevenLabs API key (required for `generate_tts`)
- `DEEPGRAM_API_KEY` — (diarization comparison only)
- `SONIOX_API_KEY` — (diarization comparison only)
- `SPEECHMATICS_API_KEY` — (diarization comparison only)

## First Run

```bash
# Option A: Generate synthetic TTS test turns (recommended)
# Requires TTS_KEY in .env — produces 25 turns across 5 categories
python -m src.harness.generate_tts

# Option B: Segment real recorded audio into turns
python -m src.harness.segment fixtures/sovereign_place_5.m4a

# Start server
python -m src.harness
# Open http://localhost:8000
```

## Running on a Different Machine

The framework requires macOS with BlackHole virtual audio drivers — it cannot run on headless Linux servers. To set up on a new Mac:

1. Clone the repo and install Python dependencies (`pip install -r requirements.txt`)
2. Install system dependencies: `brew install ffmpeg blackhole-2ch blackhole-16ch` and reboot
3. Configure Audio MIDI Setup (see steps above)
4. Copy `.env.example` to `.env` and set `TTS_KEY` if generating TTS turns
5. Generate turns: `python -m src.harness.generate_tts`
6. Start the harness: `python -m src.harness`
7. Open a ConvAI agent in a browser tab (mic set to BlackHole 2ch)
8. Open http://localhost:8000, select a source, and click Run All
