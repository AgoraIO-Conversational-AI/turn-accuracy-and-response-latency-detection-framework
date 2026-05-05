# Turn Accuracy and Response Latency Detection Framework

Measure time-to-first-audio (TTFA), barge-in behavior, and no-response rates for Conversational AI agents by playing real conversation turns through virtual audio devices and monitoring agent responses with VAD.

---

[![Demo Video](https://img.shields.io/badge/▶_Demo_Video-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1GK6eZIvzP2JgYRWWjNdNZyQpZlsGuEnh/view?usp=sharing)

---

## Overview

Three stages:

1. **Audio Preparation** — real conversation recordings are sent to a cloud STT provider (Deepgram, Soniox, or Speechmatics) for diarization. The provider returns word-level timestamps with speaker labels, which are assembled into a turn index (timing, speaker, transcript, hesitation gaps).

2. **Turn Playback Harness** — plays the prepared turns into a ConvAI agent via virtual mic, uses amplitude detection to identify when the agent responds, and measures response latency and barge-in behavior.

3. **Diarization Comparison** — optional tool to evaluate multiple STT providers side-by-side on the same audio for speaker separation accuracy.

---

## Key Metrics

| Metric | Description | Thresholds |
|--------|-------------|------------|
| **TTFA** | Time from end of user turn to first agent audio | Green <500ms, Yellow 500–1500ms, Red >1500ms |
| **Barge-in** | Agent speaks during a mid-turn hesitation gap | Detected via negative TTFA or gap overlap |
| **No Response** | Agent fails to respond within timeout | Default 10s timeout |

---

## Quick Start

### Prerequisites

- macOS with [BlackHole](https://existential.audio/blackhole/) virtual audio (2ch + 16ch)
- Python 3.12+
- ffmpeg (`brew install ffmpeg`)

### Install

```bash
git clone https://github.com/AgoraIO-Conversational-AI/turn-accuracy-and-response-latency-detection-framework.git
cd turn-accuracy-and-response-latency-detection-framework
pip install -r requirements.txt
cp .env.example .env  # add API keys if using diarization
```

### Audio Routing Setup (one-time)

1. `brew install blackhole-2ch blackhole-16ch` + reboot
2. Audio MIDI Setup → create Multi-Output Device combining:
   - BlackHole 16ch (for harness to capture agent output)
   - Your speakers (MacBook Pro Speakers, Studio Display, external DAC — whatever you use for audio)
3. System Settings → Sound → Output → Multi-Output Device
4. Browser mic input → BlackHole 2ch

The harness auto-detects "MacBook Pro Speakers" for monitoring playback. If you use different speakers, configure via the API after startup:

```bash
# List available devices
curl http://localhost:8000/api/devices

# Set your speaker device (use the index from the list above)
curl -X POST http://localhost:8000/api/devices/configure \
  -H "Content-Type: application/json" \
  -d '{"speakers": <your_device_index>}'
```

### Prepare Audio (turn generation)

The repo includes pre-generated turn data in `out/` for both sources. To run the harness with the included data, skip to "Run the Harness" below.

To prepare new audio, there are two paths:

**Multi-speaker recordings** — use STT diarization to get speaker-labeled turns:

```bash
# Run diarization via Deepgram (or soniox, speechmatics)
python -m src.diarization.compare_providers --audio fixtures/sovereign_place_5.m4a --providers deepgram --split
```

This produces `diarize_output/comparison.json` with speaker-labeled word timings. The `out/turns_index.json` consumed by the harness was built from this diarization data — currently this is a manual preparation step (the included `out/` was pre-generated from Soniox output).

**Single-speaker recordings** — use local volume-based segmentation:

```bash
# Segment by silence gaps + transcribe via Deepgram for turn text
python -m src.harness.segment fixtures/sovereign_place_10.m4a --transcribe

# Or segment only (no transcript, shows "Turn N" placeholders)
python -m src.harness.segment fixtures/sovereign_place_10.m4a

# Transcribe existing turns without re-segmenting
python -m src.harness.segment --transcribe-only --output-dir out/Sovereign_Place_10
```

Segmentation detects silence gaps in the waveform to split into turns, writes `turns_index.json` + per-turn WAVs. The `--transcribe` flag sends each turn to Deepgram Nova-3 to populate the text field with actual spoken words (requires `DEEPGRAM_API_KEY` in `.env`).

### Run the Harness

```bash
# Start the server
python -m src.harness

# Open http://localhost:8000
```

### Run Diarization Comparison

```bash
# Compare all 3 providers on the same audio
python -m src.diarization.compare_providers --audio fixtures/sovereign_place_5.m4a --split
```

---

## Architecture

```
Harness ──stream 1──→ BlackHole 2ch ──→ Browser mic input (to ConvAI agent)
Harness ──stream 2──→ Speakers      ──→ You hear turns (monitoring)

Browser ──sys output──→ Multi-Output Device (macOS)
                         ├─→ BlackHole 16ch ──→ Harness captures (VAD detects agent speech)
                         └─→ Speakers        ──→ You hear agent response
```

---

## Project Structure

```
├── src/
│   ├── harness/              # Turn playback + measurement server
│   │   ├── server.py         # FastAPI + WebSocket entry point
│   │   ├── audio_engine.py   # Dual playback + capture via sounddevice
│   │   ├── vad_engine.py     # RMS amplitude speech detection (16kHz)
│   │   ├── turn_manager.py   # Turn sequencing, TTFA, barge-in detection
│   │   ├── audio_prep.py     # Extract per-turn WAVs from source
│   │   ├── segment.py        # Volume-based audio segmentation
│   │   └── static/           # Web UI (HTML/JS/CSS)
│   └── diarization/          # Provider comparison tool
│       └── compare_providers.py
├── fixtures/                  # Source audio files
├── docs/                      # Documentation
│   ├── ai/                    # AI-oriented progressive disclosure docs
│   └── diarization_comparison.md
├── requirements.txt
└── .env.example
```

---

## Documentation

- [AI-oriented docs](docs/ai/L0_repo_card.md) — progressive disclosure for AI agents
- [Diarization comparison results](docs/diarization_comparison.md)

---

## License

Proprietary — Agora.io
