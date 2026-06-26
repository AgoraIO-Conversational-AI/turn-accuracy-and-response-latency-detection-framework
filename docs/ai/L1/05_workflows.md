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

Four categories. The UI key beside Audio Devices shows the same four:

| Type | Tags | Definition |
|---|---|---|
| `normal` | — | Semantically complete sentence. Baseline TTFA. |
| `pause` | `[pause]` | Semantically incomplete. `[pause]` Semantically complete. The words before the gap clearly do not stand on their own. |
| `hesitation` | `[hesitation]` and/or `[pause]` | Semantically ambiguous. Words before the gap could plausibly be a complete answer or the start of a longer one. The `[hesitation]` tag adds prosodic lengthening on the last word before the gap; `[pause]` opens silence. Pick whichever combination produces silence in the design range (see tag-patterns table below). |
| `ambiguous` | — | Semantically incomplete BUT prosody marks end. Sentence ends with a trailing cue ("…but.", "…so.", "…it's hard to say.") — words don't finish but the TTS renders falling-pitch end-of-sentence prosody. Tests over-cautious EOT detectors that wait when prosody-driven and semantic-driven cues *disagree*. |

The agent under test never interrupts during the silence portion of any gap-bearing turn — that's a given.

Older versions of this corpus also had a `hesitation2` category for prosody-only + precise-silence-injected gaps. We no longer inject silence ourselves on new turns (whatever ElevenLabs renders is what gets zeroed in place), so `hesitation2` was collapsed into `hesitation`.

### Benchmark 1 corpus (default source)

21 turns (0-20). Every turn is **freshly synthesized through ElevenLabs**; no WAVs are copied from the legacy `TTS_Turns_Silenced` corpus anymore. Texts and voice assignments live in `SENTENCES_BENCHMARK1` (`src/harness/generate_tts.py`).

Category mix (current): normal (5), pause (4), hesitation (9), ambiguous (3).

Each gap-bearing turn's silence is **whatever ElevenLabs natural rendering produces**, then bit-perfectly zeroed in place. We pick tag patterns (see below) that land the silence in the 700-1500 ms design range; current corpus distribution is 860, 860, 982, 1000, 1000, 1020, 1040, 1100, 1140, 1220, 1240, 1400, 1500 ms.

Audio: 48 kHz mono 16-bit PCM. Every pause / hesitation silence region is **bit-perfectly zero** with a 3 ms linear fade at each edge. Reported `hesitations.duration_ms` is the **actual measured zero-run length** in the WAV — never padded, never rounded to a target.

### Adding a new TTS turn — recipe

1. **Append an entry to `SENTENCES_BENCHMARK1`** in `src/harness/generate_tts.py` with these fields:

   ```python
   {
       "voice": 0,                     # 0-4 (see VOICES)
       "category": "pause",            # normal / pause / hesitation / ambiguous
       "text": "...",                  # one field — tts and display are the same string
       "expected_complete": False,     # True for normal / ambiguous
   }
   ```

   No `target_gap_ms`. The generator zeros whatever silence ElevenLabs produces and reports the actual length.

2. **Probe the text against ElevenLabs first.** ElevenLabs is very non-deterministic — same text and voice produce different silence lengths each call. Render 2-3 times via `/tmp/tts_probe.py` to see the range before committing:

   ```bash
   python /tmp/tts_probe.py --voice <N> --n 3 --text "your candidate text"
   ```

   The probe renders the text, runs the same `_zero_out_gap(min_gap_ms=150)` post-processing the corpus uses, and reports the measured zero-region length per render. Iterate on tag pattern + sentence shape until at least one render lands in your target band (typically 700-1500 ms).

3. **Delete the existing WAV** if you're replacing a turn:

   ```bash
   rm out/Benchmark_1/turns/speaker<N>/turn_<NNN>.wav
   ```

4. **Generate**:

   ```bash
   python -m src.harness.generate_tts --benchmark1 --skip-existing --no-reprocess-existing
   ```

   Synthesizes only missing WAVs; existing WAVs are re-measured but not re-zeroed.

5. **Verify the silence landed in range.** If not, repeat steps 2-4 with a different tag pattern. ElevenLabs may give you a different value than the probe — non-determinism is real, just keep rolling.

6. **Restart pm2**:

   ```bash
   pm2 restart benchmark
   ```

7. **Listen** to the turn via the per-row preview (click the turn number in the UI). Confirm the text, gap position, and silence length.

### Tag patterns — what to expect (empirical, non-deterministic)

ElevenLabs `eleven_v3` model is *very* non-deterministic. Same text + voice can give silence lengths varying by 2-3× across renders. The table below is the typical center of the distribution; individual renders can fall well outside.

| Pattern | Typical silence range | Notes |
|---|---|---|
| (no tag) | 0 ms | natural prosody, no engineered silence |
| `[hesitation]` only | 0 ms | prosodic lengthening only; no silence opens |
| `[pause]` | 800-1500 ms (varies wildly: 400-2000) | the workhorse; voice-dependent |
| `[pause] [pause]` | 1000-2000+ ms | longer silence; useful when single `[pause]` is too short for a particular voice |
| `[hesitation] [pause]` | unreliable — 200 ms to 1500+ ms | the original "prosody + silence" pattern; very volatile |
| `[hesitation] [hesitation] [pause]` | 800-1200 ms | more reliable than single `[hesitation] [pause]` |
| `[short pause]` | 1500-2000 ms with some voices | seemingly named for the long pause it produces |
| literal `urr` / `um` / `uh` | adds an audible filler word | no silence on its own — combine with `[pause]` |

**Voice quirks observed:**

- **Voice 1** is the most volatile — same text can produce 200 ms one render and 2000 ms the next.
- **Voice 2** trends *long*: even a single `[pause]` often produces 1500-2000 ms. Use shorter sentences when targeting < 1500 ms.
- **Voice 4** ignores `[hesitation]` and (sometimes) `[pause]` if placed late in the sentence. Put the `[pause]` mid-sentence after a content word ("…subscription [pause] was cancelled…") for it to land.
- **Voices 0 and 3** are the most predictable — `[pause]` and `[hesitation] [hesitation] [pause]` patterns behave near the table averages.

### Gotchas

These bit us in practice — not theoretical:

- **ElevenLabs is wildly non-deterministic.** Don't be surprised when "the text that gave 820 ms yesterday" gives 320 ms today. Use the probe to take 2-3 samples before committing. If you cannot get a turn into range after a few iterations, change the tag pattern or sentence, not just retry.

- **`[pause] [pause]` is *not* dangerous anymore.** It used to delete words because the cleanup pass merged multi-gap audio by cutting everything between the first and last gap. `_set_gap_duration` no longer does that — it picks the *largest* silence and ignores natural inter-word breaths. You can use `[pause] [pause]` safely to push silence longer for volatile voices.

- **Comma before `[hesitation]` kills the prosodic lengthening.** `Michael Turner, [hesitation]` lengthens the comma-stop, not "Turner". Write `Michael Turner [hesitation]`.

- **`[hesitation] [pause]` is unreliable.** Most renders collapse to 200-400 ms breath-pauses. Use `[hesitation] [hesitation] [pause]` or just `[pause]` for stronger guarantees.

- **Voice 4 ignores `[hesitation]` tags.** With voice 4, only `[pause]` works, and even then it needs to be placed mid-sentence after a content word — late-sentence `[pause]` gets ignored.

- **`text` is single-field.** No separate `tts_text` / `display_text`. The same string goes to ElevenLabs and to the UI. Long rows clip with CSS `text-overflow: ellipsis`; the full text is on hover via the `title` attribute.

- **The zero-fill grows past the true silence by ~400 ms.** `_zero_out_gap` uses `_analyze_speech_boundaries(min_gap_ms=...)` with an RMS threshold of 0.005. Quiet decay tails on either side of the actual silence fall below that threshold and get absorbed into the zeroed region. So a real ElevenLabs 1140 ms silence often ends up as a 1540 ms bit-perfect-zero region in the WAV. That's intentional — the agent sees a cleaner gap — but it means the column reading is "true silence + decay edges", not "true ElevenLabs silence". Lower `RMS_SILENCE_THRESHOLD` if you need tighter mapping.

- **`min_gap_ms` controls which silences are visible.** Default is 450 ms (suits engineered `[pause]` gaps). The natural-gap branch in `generate_benchmark1` passes 150 ms so short thinking pauses (200-400 ms) around `[hesitation]` / filler words still get detected and zeroed.

- **Turn 0 (opener) is special.** Carries operator instructions ("I'm going to say some random things…"). Its mid-sentence pause is **not** zero-filled — the operator must hear the full instruction intact.

- **Restart pm2 after editing `turns_index.json`.** The service caches the index in memory on source switch. Static JS/HTML/CSS changes are served live by FastAPI; Python and index changes need `pm2 restart benchmark`.

- **Always run with `--skip-existing --no-reprocess-existing`** when iterating on a single turn. Plain `--benchmark1` re-processes existing WAVs through `_set_gap_duration` and can shift silence boundaries.

### Legacy TTS_Turns / TTS_Turns_Silenced corpora

25 turns. `tts_turns` has comfort-noise-filled gaps; `tts_turns_silenced` is the same 25 WAVs with gap interiors zeroed in post. Both remain available as alternate sources so old-result comparisons stay reproducible. **New work should target Benchmark 1** — the corpus design is more recent, the semantic-category rulebook is tighter, and the 6 billing-call utterances at positions 1-6 are not present in the legacy corpora.

## Workflow 5: Add a New Audio Source

1. Place the audio file in `fixtures/`
2. Segment it: `python -m src.harness.segment fixtures/new_file.m4a`
3. Add entry to `SOURCES` dict in `src/harness/turn_manager.py`
4. Restart server — new source appears in dropdown
