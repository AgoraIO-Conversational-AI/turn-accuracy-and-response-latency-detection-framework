# Speaker Diarization — Provider Comparison

## Goal
Split a 2-speaker conversation recording into per-speaker audio files.
Evaluate Deepgram, Soniox, and Speechmatics batch diarization.

## Test Audio
- File: `t1.m4a` — 88.6s, stereo, 48kHz AAC
- Content: 2 speakers in casual conversation (mother + daughter discussing Crete holidays)
- 20 speaker turns, ~60/27% talk-time split

## Results Summary

| Metric | Deepgram | Soniox | Speechmatics |
|--------|----------|--------|--------------|
| Processing time | **2.7s** | 9.0s | 5.2s |
| Speakers found | 2 | 2 | 2 |
| Segments | 20 | 20 | 19 |
| Spk 0 time | 52.7s | 49.1s | 53.0s |
| Spk 1 time | 24.2s | 21.8s | 25.3s |
| Spk 0 words | 147 | 148 | 147 |
| Spk 1 words | 75 | 77 | 73 |
| Agreement | — | — | — |

## Agreement
**99% agreement** across all 3 providers (151/153 half-second bins).

Only **2 bins** of disagreement at `0:37.5 - 0:38.0s`:
- Deepgram + Soniox: Speaker 0 (continuation of question)
- Speechmatics: Speaker 1 (starts answer early)

This is at the boundary of "Do you remember whereabouts in Crete?" → "It was near Heraklion" — a natural turn-taking boundary where the exact split point is ambiguous.

## Transcript Quality

| Provider | Punctuation | Accuracy Notes |
|----------|-------------|----------------|
| Deepgram | Full (commas, periods) | "Tabs" (correct name), clean text |
| Soniox | Full (commas, periods) | "Tabs" correct, "Holy" for "Oh yeah" at end |
| Speechmatics | None | "chaps" for "Tabs", "hotbed" for "hot weather", "Previously" hallucinated |

## Observations

1. **All three agree on speaker assignment** — the diarization itself is essentially identical
2. **Deepgram is 2-4x faster** (2.7s vs 5-9s) and is already integrated
3. **Deepgram has best transcript quality** — proper punctuation, correct name recognition
4. **Speechmatics has no punctuation** in batch mode and has more transcription errors
5. **Soniox sub-word tokenization** requires merging logic but diarization is accurate
6. **Speechmatics missed the final "Oh yeah"** — only 19 segments vs 20

## Split Output Files
Stored in `diarize_output/`:
- `t1_{provider}_speaker0.wav` — main speaker
- `t1_{provider}_speaker1.wav` — second speaker
- `raw_{provider}.json` — full API response
- `comparison.json` — segment data for all providers

## Recommendation
**Deepgram** — fastest, best transcript quality, already in the pipeline, diarization accuracy matches the others. No reason to add another provider for this use case.

## Usage

API keys are loaded from `.env` in the project root.

```bash
# run all 3 providers, compare, and split into per-speaker WAVs
python -m src.diarization.compare_providers --audio /path/to/audio.m4a --split

# single provider only
python -m src.diarization.compare_providers --audio /path/to/audio.m4a --providers deepgram --split

# two providers
python -m src.diarization.compare_providers --audio /path/to/audio.m4a --providers deepgram,soniox --split

# custom output directory
python -m src.diarization.compare_providers --audio /path/to/audio.m4a --split --output-dir my_output

# no split — just transcribe and compare
python -m src.diarization.compare_providers --audio /path/to/audio.m4a
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | (required) | Audio file to diarize (m4a, mp3, wav, ogg, flac) |
| `--providers` | `deepgram,soniox,speechmatics` | Comma-separated list of providers |
| `--split` | off | Split audio into per-speaker WAV files |
| `--output-dir` | `diarize_output` | Output directory (relative to script) |
| `--deepgram-key` | from `.env` | Override Deepgram API key |
| `--soniox-key` | from `.env` | Override Soniox API key |
| `--speechmatics-key` | from `.env` | Override Speechmatics API key |

### Dependencies

```bash
pip install requests
```

`ffmpeg` must be installed for `--split` (audio segment extraction).

## API Keys

All stored in `.env`:
- `DEEPGRAM_API_KEY`
- `SONIOX_API_KEY`
- `SPEECHMATICS_API_KEY`
