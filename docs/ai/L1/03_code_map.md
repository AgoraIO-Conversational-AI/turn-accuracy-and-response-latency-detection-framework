# L1 — Code Map

## src/harness/

| File | Purpose | Key Exports |
|------|---------|-------------|
| `server.py` | FastAPI app, WebSocket endpoint, REST API. Adds `GET /api/wav/...` (serves turn WAVs to the browser harness) and `POST /api/results/submit` (ingests browser-measured results). | `app`, `main()` |
| `audio_engine.py` | Dual playback + capture via sounddevice (server-mode only) | `AudioEngine`, `list_devices()` |
| `vad_engine.py` | RMS amplitude speech detector (server-mode only — mirrored in `static/browser_harness.js` for browser mode) | `VadEngine` |
| `turn_manager.py` | Turn sequencing, TTFA measurement, barge-in. `ingest_browser_result()` stores browser-measured results in the same `run.results` list used by server-mode runs, so summary stats are uniform across both modes. | `TurnManager`, `TurnResult`, `TurnState`, `ingest_browser_result()` |
| `audio_prep.py` | Extract per-turn WAVs from source using ffmpeg | `extract_turns()` |
| `segment.py` | Volume-based turn segmentation | `segment_audio()` |
| `generate_tts.py` | Synthetic TTS turn generation via ElevenLabs | `main()` |
| `__main__.py` | Module entry point | calls `server.main()` |
| `static/index.html` | UI skeleton + Setup modal (3-step browser-mode flow) | — |
| `static/app.js` | UI logic. Device picker reads from `navigator.mediaDevices.enumerateDevices()`, persists `{id, pinned}` selections in `localStorage` (`benchmark.deviceIds.v2`), auto-detects BlackHole on unpinned slots, drives the live RMS meter, and orchestrates `runBrowserTurn` with an `AbortController` for clean Stop / Reset. | — |
| `static/browser_harness.js` | ES module — in-browser playback (`<audio>` + `setSinkId`), capture (`getUserMedia` + 16 kHz `AudioContext` + `ScriptProcessor`), `BrowserVad` (mirror of Python `VadEngine`), `runBrowserTurn()` poll loop, `startMeter()` continuous monitor. Plumbs `AbortSignal` through every async hop. | `runBrowserTurn`, `startMeter`, `primePermission`, `listDevices`, `canSelectOutputDevice`, `submitResult` |
| `static/style.css` | Styling — Setup modal, device picker, amplitude meter, copy buttons, status badges. | — |

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
| `out/Benchmark_1/turns_index.json` | Default 20-turn TTS corpus (bit-perfect zero gaps) |
| `out/Benchmark_1/turns/speaker{0-4}/turn_NNN.wav` | Benchmark 1 turn audio |
| `out/TTS_Turns/turns_index.json` | Legacy TTS turn metadata with categories |
| `out/TTS_Turns/turns/speaker{0-4}/turn_NNN.wav` | Legacy synthetic TTS turn audio |
| `diarize_output/` | Diarization comparison results |
