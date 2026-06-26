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
# Default — generate the Benchmark 1 corpus (21 turns, bit-perfect zero gaps)
python -m src.harness.generate_tts --benchmark1

# Skip turns whose WAV already exists; only synthesize missing ones
python -m src.harness.generate_tts --benchmark1 --skip-existing

# Skip + DO NOT re-run _set_gap_duration on existing keepers (the silenced
# corpus already has correct measured durations; re-running rounds them
# to the RMS-window-quantized target)
python -m src.harness.generate_tts --benchmark1 --skip-existing --no-reprocess-existing

# Force regenerate everything from scratch
python -m src.harness.generate_tts --benchmark1 --force

# Legacy 25-turn TTS_Turns corpus (comfort-noise gaps, not the default)
python -m src.harness.generate_tts

# Hesitation2-only subset for isolated testing
python -m src.harness.generate_tts --hesitation2
```

Output:
- `--benchmark1`: `out/Benchmark_1/turns_index.json` + `out/Benchmark_1/turns/speaker{0-4}/turn_NNN.wav`
- default (legacy): `out/TTS_Turns/turns_index.json` + `out/TTS_Turns/turns/speaker{0-4}/turn_NNN.wav`

### Turn type semantics (rulebook)

The UI shows the same key under "Turn types" beside Audio Devices. Each type encodes a specific test condition:

| Type | Tags | Definition |
|---|---|---|
| `normal` | — | Semantically complete sentence. Baseline TTFA. |
| `pause` | `[pause]` | Semantically incomplete. `[pause]` Semantically complete. The words before the gap clearly do not stand on their own. |
| `hesitation` | `[hesitation] [pause]` or `[hesitation]` × N | Semantically ambiguous. `[hesitation] [pause]` Semantically complete. The words before the gap could plausibly be a complete answer or the start of a longer one. The `[hesitation]` tag adds prosodic lengthening on the **last word** before the gap. |
| `hesitation2` | `[hesitation]` + measured silence | Semantically ambiguous; the engineered silence portion is precisely measured. Same semantics as `hesitation` — kept as a separate category because the original test design produced these via prosody-only `[hesitation]` tags without explicit fillers, post-processed to exact-target zero-filled silence. |
| `ambiguous` | — | Semantically incomplete BUT prosody marks end. Sentence ends with a trailing cue ("…but.", "…so.", "…it's hard to say.") — words don't finish but the TTS renders falling-pitch end-of-sentence prosody. Tests over-cautious EOT detectors that wait when both prosody-driven and semantic-driven cues *disagree*. |

It is taken as given that the agent under test should never interrupt during the silence portion of any gap-bearing turn. That's not stated in the key.

### Benchmark 1 corpus (default source)

21 turns (0-20). Two provenances:

- **6 new utterances** (positions 1-6) — billing-call scenario, defined in `SENTENCES_BENCHMARK1` (`src/harness/generate_tts.py`). Freshly synthesized through ElevenLabs.
- **15 keepers** (positions 0, 7-20) — WAVs + index entries copied verbatim from `out/TTS_Turns_Silenced/`. **Do not regenerate.** Audio + measured silence must stay stable across runs.

Keeper renumber map:

| B1 turn | TTS_Turns_Silenced | Spk | B1 turn | TTS_Turns_Silenced | Spk |
|---|---|---|---|---|---|
| 0 | 0 | 2 | 14 | 16 | 3 |
| 7 | 2 | 2 | 15 | 17 | 2 |
| 8 | 4 | 2 | 16 | 21 | 0 |
| 9 | 5 | 0 | 17 | 22 | 3 |
| 10 | 6 | 3 | 18 | 23 | 4 |
| 11 | 7 | 1 | 19 | 24 | 1 |
| 12 | 11 | 2 | 20 | 9 | 4 |
| 13 | 15 | 1 | | | |

Category mix (current): normal (5), pause (4), hesitation (5), hesitation2 (4), ambiguous (3).

Audio: 48 kHz mono 16-bit PCM. Every pause / hesitation / hesitation2 silence region is **bit-perfectly zero** with a 3 ms linear fade at each edge. Reported `hesitations.duration_ms` is the **actual measured zero-run length** in the WAV — never padded, never rounded to a target.

### Adding a new TTS turn — recipe

1. **Append an entry to `SENTENCES_BENCHMARK1`** in `src/harness/generate_tts.py` with these required fields:

   ```python
   {
       "voice": 0,                     # 0-4 (see VOICES)
       "category": "pause",            # normal / pause / hesitation / hesitation2 / ambiguous
       "tts_text":     "...",          # exact text sent to ElevenLabs
       "display_text": "...",          # MUST equal tts_text byte-for-byte
       "expected_complete": False,     # True for normal / ambiguous
   }
   ```

   Omit `target_gap_ms` — new turns now use whatever silence ElevenLabs produces, then zero it in place. The 800-1500 ms range is a *design guideline* for picking sentences whose tag patterns reliably land in that range, not a runtime target.

2. **Pick a tag pattern that reliably produces the silence you need** (see table below).

3. **Delete the existing WAV** (if regenerating) so the generator re-synthesizes:

   ```bash
   rm out/Benchmark_1/turns/speaker<N>/turn_<NNN>.wav
   ```

4. **Run the generator** with `--skip-existing --no-reprocess-existing` so only the missing WAVs hit the API and keeper WAVs/index entries aren't disturbed:

   ```bash
   python -m src.harness.generate_tts --benchmark1 --skip-existing --no-reprocess-existing
   ```

5. **Re-overlay keeper index entries** from `TTS_Turns_Silenced` (the `--benchmark1` run rewrites the whole `turns_index.json`; keeper entries must be restored from the silenced source). The renumber map above is the source of truth.

6. **Restart pm2** so the index is reloaded into memory:

   ```bash
   pm2 restart benchmark
   ```

7. **Listen** to the new turn via the per-row preview (click the turn number in the UI). Verify the text, gap position, and silence length match expectations.

### Tag patterns and what to expect

| Pattern | Effect | Typical silence |
|---|---|---|
| (no tag) | natural prosody, complete sentence | 0 ms |
| `[hesitation]` | prosodic lengthening on last word, no silence | 0 ms |
| `[pause]` | one engineered silence | **variable** — often 1100-1500 ms, sometimes much less |
| `[hesitation] [pause]` | prosody + silence | unreliable — sometimes collapses to ~200 ms |
| `[hesitation] [hesitation] [pause]` | double prosody + silence | **reliable** — 800-1200 ms |
| `[hesitation] [hesitation] [pause] [pause]` | as above, with longer silence | 1200-1500 ms |
| literal `urr` / `um` / `uh` | audible filler word, no enforced silence | natural breath only |

Match the pattern to the category:

- **`pause`** — `[pause]` alone usually works. If you need a guaranteed 800+ ms gap and a single `[pause]` is producing too little, fall back to a `[hesitation] [hesitation] [pause]` pattern even though semantically it's pause (the prosody on a final-comma sentence end is barely audible if the preceding word doesn't invite lengthening).
- **`hesitation`** — `[hesitation] [hesitation] [pause]` is the workhorse. Add a literal `urr`/`um`/`uh` if you want an audible filler in addition to the silence.
- **`hesitation2`** — legacy category; new turns don't need this. Use plain `hesitation`.
- **`normal`** and **`ambiguous`** — no tags.

### Gotchas

These bit us in practice — not theoretical:

- **`[pause] [pause]` can delete words.** ElevenLabs sometimes places the two tags at *different* sentence positions (one after "for", one after "dollars"). The legacy `_enforce_single_gap()` helper used to merge them by cutting all audio between the first and last gap — wiping out the intervening phrase. The current `_set_gap_duration()` no longer calls `_enforce_single_gap`; instead it picks the **largest** detected silence and operates on that, leaving natural breath-pauses intact. Still: prefer a single `[pause]` to keep the audio shape predictable.

- **Comma before `[hesitation]` kills the prosodic lengthening.** `Michael Turner, [hesitation]` lengthens the comma-stop, not "Turner". Write `Michael Turner [hesitation]`.

- **`[hesitation] [pause]` is unreliable.** ElevenLabs frequently collapses this to a 200 ms breath-pause. Use `[hesitation] [hesitation] [pause]` for a guaranteed 800+ ms gap.

- **`display_text` must equal `tts_text` byte-exactly.** The Text column shows operators what ElevenLabs got. Drift between the two is a bug. Long rows are clipped with CSS ellipsis; full text on hover via the `title` attribute.

- **The zero-fill grows past the true silence by ~400 ms.** `_zero_out_gap` uses `_analyze_speech_boundaries(min_gap_ms=...)` with an RMS threshold of 0.005. Quiet decay tails on either side of the actual silence fall below that threshold and get absorbed into the zeroed region. So a real ElevenLabs 1140 ms silence often ends up as a 1540 ms bit-perfect-zero region in the WAV. That's intentional — the agent sees a cleaner gap — but it means the column reading is "true silence + decay edges", not "true ElevenLabs silence". Lower `RMS_SILENCE_THRESHOLD` if you need tighter mapping.

- **`min_gap_ms` controls which silences are visible.** Default is 450 ms (suits engineered `[pause]` gaps). The natural-gap branch in `generate_benchmark1` passes 150 ms so short thinking pauses (200-400 ms) around `[hesitation]` / filler words still get detected and zeroed.

- **Round numbers in the legacy keepers are coincidence.** Turns 16 and 17 happen to land on 1200.0000 and 1000.0000 ms because their original `[pause]` rendered cleanly within one RMS window of target. The other keepers are 794, 796, 806, 995, 1194, 1494 ms — *not* rounded. Reported values are always actual measurements, never padded.

- **Turn 0 (opener) is special.** Carries operator instructions ("I'm going to say some random things…"). Its mid-sentence pause is **not** zero-filled — the operator must hear the full instruction intact. Don't add `target_gap_ms` to it.

- **Restart pm2 after editing `turns_index.json`.** The service caches the index in memory on source switch. Static JS/HTML/CSS changes are served live by FastAPI; Python and index changes need `pm2 restart benchmark`.

- **Always run with `--skip-existing --no-reprocess-existing`** when adding new turns. Plain `--benchmark1` re-processes existing WAVs through `_set_gap_duration`, which can subtly shift the keeper silence boundaries. The `--no-reprocess-existing` flag is the safe default for incremental updates.

### Legacy TTS_Turns / TTS_Turns_Silenced corpora

25 turns. `tts_turns` has comfort-noise-filled gaps; `tts_turns_silenced` is the same 25 WAVs with gap interiors zeroed in post. Both remain available as alternate sources so old-result comparisons stay reproducible. **New work should target Benchmark 1** — the corpus design is more recent, the semantic-category rulebook is tighter, and the 6 billing-call utterances at positions 1-6 are not present in the legacy corpora.

## Workflow 5: Add a New Audio Source

1. Place the audio file in `fixtures/`
2. Segment it: `python -m src.harness.segment fixtures/new_file.m4a`
3. Add entry to `SOURCES` dict in `src/harness/turn_manager.py`
4. Restart server — new source appears in dropdown
