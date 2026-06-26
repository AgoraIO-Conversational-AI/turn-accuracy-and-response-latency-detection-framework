# Browser harness TTFA improvements — plan (revised)

**Note:** earlier version of this file de-prioritized the analyser approach in favour of cheaper fixes. That was wrong if accuracy is the top priority. The architecture below is sample-precise and independent of all browser event timing.

## Principle

The most accurate response-latency / barge-in detection comes from continuously sampling **output audio amplitude** and **input audio amplitude** in the **same wall-clock domain**, then computing:

```
TTFA = first_input_speech_wallclock − last_output_speech_wallclock
```

Everything else is approximation. No media-element events (`play`, `playing`, `ended`) and no index-derived predictions (`turn.duration_ms`, `turn.speech_end_ms`) are involved in the primary measurement. They become diagnostic / sanity-check values only.

## Architecture

Two parallel Web Audio analyser chains, both tagged with `performance.now()`:

### Output chain (new)

```
HTMLAudioElement (primary playback)
    ↓ createMediaElementSource
AnalyserNode (output)
    ↓
AudioContext.destination
```

- Dedicated AudioContext, run at native sample rate (typically 48 kHz on BlackHole boxes).
- AudioWorklet (or ScriptProcessor) tick rate of 256 samples → callback every 5.3 ms at 48 kHz.
- Each tick computes RMS over its frame and emits `(t_wallclock, rms)` into a ring buffer.
- Last tick where `rms > OUTPUT_SPEECH_THRESHOLD` becomes `lastOutputSpeechWall`.
- The MediaElement's own sink (set via `setSinkId`) continues to drive the agent-facing output; the Web Audio graph is a parallel tap, so we measure exactly the same sample stream the agent will hear.

### Input chain (already exists, ratify it)

```
getUserMedia (BlackHole 16ch)
    ↓
AudioContext (16 kHz)
    ↓ ScriptProcessor (512 samples)
RMS per frame, tagged with performance.now()
```

- Already amplitude-based. Already in `performance.now()` domain.
- Add a ring buffer of `(t_wallclock, rms)` pairs over the full turn window (currently we only record the first speech moment — keep history for retrospective barge-in analysis).

### Combined math

After each turn:

- `lastOutputSpeechWall = max{ t : output_rms(t) > OUTPUT_SPEECH_THRESHOLD }`
- `firstInputSpeechWall = min{ t : input_rms(t) > INPUT_SPEECH_THRESHOLD ∧ t ≥ armWall }`
- `ttfa_ms = firstInputSpeechWall − lastOutputSpeechWall`

Barge-in (two distinct cases):

- **Hard barge-in** (talked over the agent prompt): `firstInputSpeechWall < lastOutputSpeechWall`
- **Gap barge-in** (replied during an engineered silence): for each hesitation gap in the turn index, compute wall-clock window
  - `gapStartWall = firstOutputSpeechWall + gap.at_ms`
  - `gapEndWall   = gapStartWall + gap.duration_ms`
  - if `gapStartWall ≤ firstInputSpeechWall ≤ gapEndWall` → flag as gap barge-in. Note: in this mode we use `firstOutputSpeechWall` (the moment the prompt actually started, measured) rather than `playStartWall` (the moment we issued `play()`), so latency to the speakers cancels out.

## Why this beats the previous approaches

| Method | Captures | Misses |
|---|---|---|
| `turn.duration_ms` minus `playStartWall` | nothing real | doesn't know when speech ends inside the WAV |
| `turn.speech_end_ms + playStartWall` | better, but assumes WAV-time = wall-clock | output-buffer drain at start of playback |
| `playing` event | actual start of playback | doesn't know when speech *ends* in the prompt; same end-of-output drift |
| `ended` event | end of WAV | trailing silence inflates by 100-300 ms; browser scheduling adds more |
| **Continuous output analyser → last RMS-above-threshold sample** | **the literal last spoken sample** | nothing material; constant by 1 analyser frame (~5 ms) |

The amplitude-based approach is also self-correcting if the WAV changes — no rebuilding the index after every TTS regeneration.

## Implementation order

1. **Add the output analyser chain.** New `OutputAnalyser` class in `browser_harness.js`:
   - Constructs an AudioContext (or reuses the existing playback context if one is suitable).
   - `createMediaElementSource(primaryAudioElement)`, connects through an AnalyserNode to `destination`.
   - AudioWorklet for RMS computation (preferred) or ScriptProcessor as fallback. Both emit `(t, rms)` to a ring buffer of 8192 entries (~40 s at 5.3 ms per frame — more than enough for a single turn).
   - `start(armWall)` to begin recording; `stop()` to freeze the buffer.
   - `lastSpeechAt(threshold)` → max `t` in buffer where `rms > threshold`.

2. **Extend the input analyser to keep history.** Today's input path only remembers `firstSpeechWall`. Add a ring buffer of `(t, rms)` pairs with the same shape so we can:
   - Recompute `firstInputSpeechWall` post-hoc with different thresholds during debugging.
   - Compute gap-barge-in retrospectively against any hesitation window from the index.

3. **Compute amplitude-based TTFA / barge-in in PARALLEL — don't replace yet.** Add the amplitude-based values as new fields alongside the existing `ttfaAudio` and `ttfaWall` calculations. Both methods produce a number per turn. Neither is yet the source of truth in the UI summary — the canonical column still uses the legacy value.

   This is a deliberate two-phase rollout:
   - **Phase A (this step):** both methods compute, both get logged, the legacy value remains canonical so we can ship the change without UI churn.
   - **Phase B (after validation, in a later commit):** flip the canonical TTFA to the amplitude value once we've seen enough runs to be confident in the diff. The drift log then becomes the audit trail.

4. **First-class diff logging.** Per turn, emit a structured timing record covering both methods plus their delta:
   ```
   [harness#turnN] timing {
     armWall, playRequestedWall, playingEventWall, endedEventWall,
     firstOutputSpeechWall, lastOutputSpeechWall,
     firstInputSpeechWall,

     ttfa_amplitude_ms,    // new method (firstInputSpeech - lastOutputSpeech)
     ttfa_index_ms,        // current method (legacy)
     ttfa_event_ms,        // current method's wall-clock fallback (legacy)

     diffs: {
       amp_minus_index = ttfa_amplitude_ms - ttfa_index_ms,
       amp_minus_event = ttfa_amplitude_ms - ttfa_event_ms,
       event_minus_index = ttfa_event_ms - ttfa_index_ms
     },

     // diagnostic offsets — useful for spotting which clock domain disagrees
     buffer_drain_ms = endedEventWall - lastOutputSpeechWall,
     play_to_speech_ms = firstOutputSpeechWall - playRequestedWall,
     trailing_silence_ms = turn.duration_ms - turn.speech_end_ms,
     gap_windows_wallclock = [...]
   }
   ```

5. **First-class summary diff.** In the summary strip footer (or as a separate debug-only line in the same row), surface aggregate diffs across the run:
   ```
   Avg ttfa_amp = 412ms · Avg ttfa_index = 731ms · Avg diff = -319ms · σ(diff) = 18ms
   ```
   - Low σ(diff) across turns → systematic bias removed; we've quantified what the old method was over-reporting by. Flip to amplitude as canonical with confidence.
   - High σ(diff) → one method has per-turn variance the other doesn't; investigate via the per-turn drift log before flipping.

6. **Thresholds and tuning.** Both speech thresholds need calibration:
   - `OUTPUT_SPEECH_THRESHOLD` — the WAV has bit-perfect zero in gap regions and quiet (but nonzero) decay elsewhere. Set initially at 0.005 RMS (same as `_analyze_speech_boundaries` in the corpus generator) — guaranteed to bracket all real speech.
   - `INPUT_SPEECH_THRESHOLD` — keep at the current 0.003 we use for input VAD.
   - Expose both in a single constants block at the top of `browser_harness.js` so future calibration is one edit.

7. **One-frame bias note.** Both analysers are buffered — first detection is one frame late. At 5 ms output frames and 32 ms input frames, the canonical TTFA is biased high by ~37 ms worst case. Either subtract the constant in code or document it in the gotchas doc — it doesn't affect relative comparisons.

8. **(Future commit, after validation) Flip canonical TTFA to amplitude.** Once the diff log shows the amplitude method tracking consistently across a real run, change the summary cell + result-submit payload to use `ttfa_amplitude_ms` as the canonical number. Keep all the diagnostic fields. Keep the diff logging — it doesn't cost anything and becomes the audit trail for any future regression.

## What about the simpler stuff codex called out

Still worth doing alongside the amplitude work — they're cheap and don't conflict:

- Use `playing` event for `playStartWall` (becomes a diagnostic field, not the primary number).
- Treat `ended` as sanity-check only — log `endedWall - lastOutputSpeechWall` to flag tail-silence + buffer-drain.
- Merge `runAll` / `playSingle` into one `runTurnSequence(sequence)` to avoid divergence.
- Remove stale server-mode branches in the UI; remove unused vars (`tag`, `playbackEndWall`, `ttfaAnnounced`).
- Update stale threshold comments (0.01 / 10 s → 0.003 / 8 s).

## Failure modes to watch

1. **CORS / `createMediaElementSource`**: the source element must be on the same origin (or have `crossOrigin="anonymous"` + a CORS-permissive server). We already serve `/api/wav/...` from the same FastAPI origin — fine.
2. **`setSinkId` interaction**: the MediaElement's sink for the user-facing playback is independent of the Web Audio graph the analyser sits in. The analyser sees the same sample stream the sink emits; if the device's playback latency drifts (different output sinks have different buffer depths), the speakers will be a few ms behind the analyser. The agent under test hears what the sink emits, not what the analyser sees — so the "true" TTFA is shifted by the sink's playback latency. This shift is constant per output device, so it's a sub-10 ms calibration concern at worst.
3. **AudioContext freeze on Chrome**: if the browser tab loses focus, the AudioContext can suspend → analyser buffer stops filling. Resume on tab focus or warn the operator to keep the tab active.
4. **Sample-rate mismatch**: AudioContext defaults to system rate (typically 48 kHz, sometimes 44.1 kHz). The math doesn't care as long as we read `audioCtx.sampleRate` for the per-frame ms calculation.

## Reality check

Implementation footprint: ~150-250 lines of new JS, mostly the OutputAnalyser class + its ring buffer + the AudioWorklet processor. Plus ~30-50 lines of refactoring on the input side to keep history. None of it touches the FastAPI server.

The biggest unknown is calibrating `OUTPUT_SPEECH_THRESHOLD` to behave well across all five voices in Benchmark 1 + any future corpus. Cheap to tune since the corpus is a fixed set of WAVs.

## Validation gate before flipping canonical

This is the explicit success criterion for moving from Phase A (both methods, legacy canonical) to Phase B (amplitude canonical, legacy diagnostic):

1. Run the full 21-turn corpus end-to-end at least 3 times against a known agent.
2. Across those runs, compute σ(diff) = stddev of `ttfa_amplitude_ms - ttfa_index_ms` across all gap-bearing turns.
3. **Flip the canonical value if:**
   - σ(diff) < 30 ms (the diff is a systematic offset, not random noise)
   - `mean(diff)` is negative (amplitude method reports a *smaller* TTFA, matching the theoretical prediction that the legacy method over-reports because of trailing silence + buffer drain)
   - No turn has `|amp_minus_index| > 300 ms` (no single-turn pathology)
4. **Investigate before flipping if any of the above fails.** Use the per-turn drift log to localize whether the divergence is on a specific voice, a specific category, or a specific time-of-run (e.g. only first turn — pointing at AudioContext warm-up).

This gate is the codex deliverable's exit criterion for Phase A. Phase B is a separate, smaller commit.

## Review handoff

Codex will implement Phases A and the cleanups (steps 1-7 above + the housekeeping items). I'll review the diff before it lands. Specific things I'll be checking:

- AudioWorklet vs ScriptProcessor: AudioWorklet is preferred for timing precision but more code; ScriptProcessor is fine if the worklet path adds complexity for marginal benefit.
- Whether the output AudioContext is shared with the existing playback context or runs separately. Both options are valid; the choice affects how `setSinkId` interacts with the analyser graph.
- `lastSpeechAt` implementation — must scan the ring buffer backwards from the end (cheaper than full scan) and must handle the case where the entire turn was below threshold (no output speech detected) without throwing.
- That the legacy `ttfaAudio` / `ttfaWall` math is preserved exactly during Phase A — no incidental refactoring that could shift the legacy number, which would invalidate the diff comparison.
- That the drift log fields are stable JSON (or stable JS object) shapes so future log-analysis scripts can grep them reliably.
- That AudioContext lifecycle is correct — created on first armForTurn, suspended/resumed correctly across stop/reset, torn down on page unload to avoid leaks across runs.
