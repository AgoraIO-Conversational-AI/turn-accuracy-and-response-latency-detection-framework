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
# Default — generate the Benchmark 1 corpus (20 turns, bit-perfect zero gaps)
python -m src.harness.generate_tts --benchmark1

# Generate the original 25-turn TTS_Turns corpus (legacy, comfort-noise gaps)
python -m src.harness.generate_tts

# Re-generate all WAVs from scratch
python -m src.harness.generate_tts --benchmark1 --force

# Generate hesitation2-only subset for isolated testing
python -m src.harness.generate_tts --hesitation2
```

Output:
- `--benchmark1`: `out/Benchmark_1/turns_index.json` + `out/Benchmark_1/turns/speaker{0-4}/turn_NNN.wav`
- default (legacy): `out/TTS_Turns/turns_index.json` + `out/TTS_Turns/turns/speaker{0-4}/turn_NNN.wav`

### Benchmark 1 corpus (default source)

20 turns, opening with a billing-call scenario (turns 1-6), then a mix of pause / hesitation / hesitation2 / normal / ambiguous turns kept from the original corpus.

Category mix:
- **normal** (4): including the opener (turn 0, "I'm going to say some random things…")
- **pause** (4): mid-sentence `[pause]` tag producing silence gaps (800-1200ms)
- **hesitation** (4): filler words ("um"/"uh") with `[hesitation]` tags (800-1200ms)
- **hesitation2** (5): prosody-only pauses via `[hesitation]` tags, no fillers (800-1500ms)
- **ambiguous** (3): trailing-off sentences that sound potentially complete

All pause/hesitation/hesitation2 gaps target 800-1500ms and are **bit-perfectly zeroed** (integer-0 samples) with a 3 ms linear fade at each edge to avoid click artifacts. The reported `hesitations.duration_ms` in `turns_index.json` is the exact zero-run length, not an RMS-window estimate — so amplitude-driven EOT detectors see a true zero-amplitude silence window.

Turn 0 keeps its natural prosody and is not zero-filled — it carries the operator instructions ("Please keep your responses to under ten words.").

### Legacy TTS_Turns corpus

25 turns, comfort-noise-filled gaps (500-1500ms), still available as the `tts_turns` source. `tts_turns_silenced` is the same corpus with gap interiors zeroed in post — kept for backwards-compatibility runs. New work should target Benchmark 1.

Gap durations are enforced precisely — ElevenLabs produces a natural gap via tags, then the generator stretches/trims to the exact target. Benchmark 1 uses integer-zero fill; the legacy `tts_turns` source fills with comfort noise.

## Workflow 5: Add a New Audio Source

1. Place the audio file in `fixtures/`
2. Segment it: `python -m src.harness.segment fixtures/new_file.m4a`
3. Add entry to `SOURCES` dict in `src/harness/turn_manager.py`
4. Restart server — new source appears in dropdown
