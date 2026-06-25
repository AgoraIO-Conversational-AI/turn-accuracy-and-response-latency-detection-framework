# L1 — Setup

There are two ways to run the harness; **browser mode is the default**. Server mode is kept for headless / scripted use.

## Browser Mode (recommended)

The page in `src/harness/static/` runs the audio loop entirely in the user's Chromium browser via `static/browser_harness.js`. The server only hosts the UI and ingests results. The operator's Mac is where BlackHole + speakers + mic actually live, even if the server is hosted elsewhere (e.g. a cloud VM).

### Operator's Mac

- macOS with **BlackHole 2ch + 16ch**: `brew install blackhole-2ch blackhole-16ch` (reboot required).
- A Chromium-based browser (Chrome / Edge / Opera / Brave). `HTMLMediaElement.setSinkId` is required for output-device selection — Firefox and Safari don't ship it.
- A **Multi-Output Device** combining BlackHole 16ch + your real speakers, set as macOS system output (see "Audio Routing" below).
- The agent under test running in a separate browser tab with its mic set to BlackHole 2ch.

That's all the operator needs. Open the hosted page, click **Grant mic access** in the Audio Devices panel, hit Run All. Three device slots auto-pick from labels (`BlackHole 2ch`, your real speakers, `BlackHole 16ch`) and remember any manual override across reloads.

### Server host

Any machine that can run Python 3.12 + the requirements below — does NOT need audio hardware, BlackHole, or even speakers. Used to serve the UI, the corpus WAVs (`GET /api/wav/...`), and ingest browser-measured results (`POST /api/results/submit`).

## Server Mode (alternative)

The original mode: `python -m src.harness` runs on the operator's Mac and uses Python `sounddevice` to drive BlackHole directly. UI talks to the local `:8000` server. Required if you can't use Chromium or want headless / scripted runs. Setup steps below ("Audio Routing") apply.

## System Requirements

- macOS (required for BlackHole virtual audio drivers in both modes)
- Python 3.12+ (server only)
- BlackHole 2ch + 16ch (operator's Mac, both modes)
- `ffmpeg` ONLY if you're going to run the prep scripts (`audio_prep.py`, `generate_tts.py`, `segment.py`, diarization comparator). The runtime harness doesn't shell out to it.

## Python Dependencies (server)

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
# Requires TTS_KEY in .env — produces the default Benchmark 1 corpus
# (20 turns, 800-1500 ms bit-perfect-zero gaps).
python -m src.harness.generate_tts --benchmark1

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
