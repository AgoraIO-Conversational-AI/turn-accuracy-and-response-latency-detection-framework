# L1 — Code Map

## src/harness/

| File | Purpose | Key Exports |
|------|---------|-------------|
| `server.py` | FastAPI app, WebSocket endpoint, REST API | `app`, `main()` |
| `audio_engine.py` | Dual playback + capture via sounddevice | `AudioEngine`, `list_devices()` |
| `vad_engine.py` | RMS amplitude speech detector | `VadEngine` |
| `turn_manager.py` | Turn sequencing, TTFA measurement, barge-in | `TurnManager`, `TurnResult`, `TurnState` |
| `audio_prep.py` | Extract per-turn WAVs from source using ffmpeg | `extract_turns()` |
| `segment.py` | Volume-based turn segmentation | `segment_audio()` |
| `generate_tts.py` | Synthetic TTS turn generation via ElevenLabs | `main()` |
| `__main__.py` | Module entry point | calls `server.main()` |
| `static/` | Web UI (index.html, app.js, style.css) | — |

## src/diarization/

| File | Purpose | Key Exports |
|------|---------|-------------|
| `compare_providers.py` | Batch diarization across 3 providers | `main()`, `load_env()`, `diarize_deepgram()`, `diarize_soniox()`, `diarize_speechmatics()` |

## fixtures/

| File | Purpose |
|------|---------|
| `sovereign_place_5.m4a` | 10-min 2-speaker conversation (Lucy and Tabby) |
| `sovereign_place_10.m4a` | Single-speaker recording (Ben) |

## Generated at runtime (gitignored)

| Path | Purpose |
|------|---------|
| `out/turns_index.json` | Turn timing and metadata |
| `out/turns/speaker{0,1}/turn_NNN.wav` | Per-turn audio clips |
| `out/TTS_Turns/turns_index.json` | TTS turn metadata with categories |
| `out/TTS_Turns/turns/speaker{0-4}/turn_NNN.wav` | Synthetic TTS turn audio |
| `diarize_output/` | Diarization comparison results |
