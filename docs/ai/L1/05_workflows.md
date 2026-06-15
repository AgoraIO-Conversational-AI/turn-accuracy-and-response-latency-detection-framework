# L1 — Workflows

## Workflow 1: Prepare Turn Audio

```bash
# Segment a new audio file into turns
python -m src.harness.segment fixtures/sovereign_place_5.m4a \
    --threshold 0.01 \
    --min-silence 1000 \
    --min-turn 200

# Merge adjacent turns that are part of one utterance
python -m src.harness.segment fixtures/sovereign_place_5.m4a \
    --merge 1,2 --merge 7,8,9
```

Output: `out/<audio_stem>/turns_index.json` + `out/<audio_stem>/turns/speaker0/turn_NNN.wav`

Use `--output-dir out` to write directly to `out/` (matches the default harness source path).

## Workflow 2: Run TTFA Measurement

1. Start ConvAI agent in a browser tab (ensure mic = BlackHole 2ch)
2. Start harness: `python -m src.harness`
3. Open http://localhost:8000
4. Select source and speaker
5. Click "Run All" or play individual turns
6. Review summary stats (avg/median/p95 TTFA, barge-in count)

## Workflow 3: Compare Diarization Providers

```bash
# All providers
python -m src.diarization.compare_providers \
    --audio fixtures/sovereign_place_5.m4a --split

# Single provider
python -m src.diarization.compare_providers \
    --audio fixtures/sovereign_place_5.m4a --providers deepgram --split
```

Output: `diarize_output/raw_*.json`, `diarize_output/*_speaker*.wav`, `diarize_output/comparison.json`

## Workflow 4: Generate TTS Test Turns

```bash
# Generate all 25 turns across 5 speakers (requires TTS_KEY in .env)
python -m src.harness.generate_tts

# Re-generate all WAVs from scratch
python -m src.harness.generate_tts --force

# Generate hesitation2-only subset for isolated testing
python -m src.harness.generate_tts --hesitation2
```

Output: `out/TTS_Turns/turns_index.json` + `out/TTS_Turns/turns/speaker{0-4}/turn_NNN.wav`

Turn categories (25 total, interleaved):
- **normal** (5): semantically and prosodically complete sentences
- **pause** (5): mid-sentence `[pause]` tag producing silence gaps (500-1200ms)
- **hesitation** (5): filler words ("um"/"uh") with `[hesitation]` tags (500-1200ms)
- **hesitation2** (5): prosody-only pauses via `[hesitation]` tags, no fillers (600-1500ms)
- **ambiguous** (5): trailing-off sentences that sound potentially complete

Gap durations are enforced precisely — ElevenLabs produces a natural gap via tags, then the generator stretches/trims to the exact target using comfort noise.

## Workflow 5: Add a New Audio Source

1. Place the audio file in `fixtures/`
2. Segment it: `python -m src.harness.segment fixtures/new_file.m4a`
3. Add entry to `SOURCES` dict in `src/harness/turn_manager.py`
4. Restart server — new source appears in dropdown
