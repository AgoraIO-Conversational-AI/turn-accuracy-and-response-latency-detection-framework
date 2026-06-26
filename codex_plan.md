# Browser TTFA2 Plan

## Goal

Keep the current TTFA calculation unchanged and add a second measurement path, shown as `TTFA2`, so runs can compare old vs new timing side by side before changing the canonical metric.

## Measurement Approach

Use decoded WAV analysis for prompt timing, not a Web Audio tap on the primary playback element. This avoids disturbing `setSinkId` playback routing.

For each turn:

1. Fetch/decode the WAV in the browser, or reuse an `ArrayBuffer` already fetched for cache warming.
2. Compute prompt audio metadata from decoded PCM:
   - `decoded_duration_ms`
   - `first_output_speech_ms`
   - `last_output_speech_ms`
   - optional `output_gap_windows_ms`
3. Use the current input capture wall-clock speech detection for agent response start:
   - `firstInputSpeechWall`
4. Use the media element `playing` event, or the closest available confirmed playback-start wall time, to map decoded WAV time to wall-clock:
   - `lastOutputSpeechWall = playbackStartWall + last_output_speech_ms`
5. Compute:
   - `ttfa2_ms = firstInputSpeechWall - lastOutputSpeechWall`

The existing TTFA remains unchanged and continues to drive the current `TTFA` column and summary stats.

## Barge-In For TTFA2

Keep current barge-in behavior unchanged for now. Add diagnostic TTFA2 barge flags only:

- `ttfa2_hard_barge_in = firstInputSpeechWall < lastOutputSpeechWall`
- For each index gap:
  - `gapStartWall = playbackStartWall + gap.at_ms`
  - `gapEndWall = gapStartWall + gap.duration_ms`
  - `ttfa2_gap_barge_in = firstInputSpeechWall` inside any mapped gap

Do not make these canonical until after comparison runs.

## UI Changes

Add one new table column:

- `TTFA2`

Display rules:

- Show `TTFA2` in ms for non-barge-in responses.
- Leave blank/dash for current canonical barge-ins, matching current TTFA behavior.
- Keep the existing `TTFA` column exactly as-is.
- Do not change summary stats yet.

Optional later: add a compact `Delta` column, but do not add it in the first pass unless the table still fits cleanly.

## Console Logging

Keep logs concise and one structured line per turn:

```js
console.info("[bench timing]", {
  turn: turn.turn,
  category: turn.category,
  ttfa_ms: result.ttfa_ms,
  ttfa2_ms: result.ttfa2_ms,
  delta_ms: result.ttfa2_ms != null && result.ttfa_ms != null
    ? result.ttfa2_ms - result.ttfa_ms
    : null,
  playback_start_source: result.playback_start_source,
  decoded_duration_ms: result.decoded_duration_ms,
  last_output_speech_ms: result.last_output_speech_ms,
  output_tail_ms: result.decoded_duration_ms != null && result.last_output_speech_ms != null
    ? result.decoded_duration_ms - result.last_output_speech_ms
    : null,
  first_input_from_play_ms: result.first_input_from_play_ms,
  ttfa2_hard_barge_in: result.ttfa2_hard_barge_in,
  ttfa2_gap_barge_in: result.ttfa2_gap_barge_in,
});
```

Avoid per-frame analyser logs. Only log aggregate timing per turn.

## Implementation Steps

1. Add a decoded WAV analysis helper in `browser_harness.js`.
   - Decode with `AudioContext.decodeAudioData`.
   - Downmix to mono for analysis.
   - Use RMS windows, initially `20 ms` and threshold `0.005` to match corpus gap detection.
   - Return `first_output_speech_ms`, `last_output_speech_ms`, and `decoded_duration_ms`.

2. Capture a better playback start time.
   - Add a `playing` event timestamp to `startPlayback`.
   - If `playing` does not fire promptly, fall back to current `playController.startWall`.
   - Store `playback_start_source: "playing" | "play_call_fallback"`.

3. Compute TTFA2 in `runBrowserTurn`.
   - Keep existing `ttfa_ms`, `ttfa_audio_ms`, and `ttfa_wall_ms` untouched.
   - Add `ttfa2_ms`, decoded output fields, and diagnostic barge flags to the result payload.

4. Add a `TTFA2` table cell in `app.js`.
   - Render it from local browser result.
   - Include it in server-submitted payload for future analysis.

5. Refactor only if needed.
   - Do not combine `runAll` / `playSingle` in the same change unless the TTFA2 patch becomes awkward.
   - Avoid unrelated UI cleanup in this pass.

6. Update docs/comments.
   - Document that `TTFA` is legacy/current canonical.
   - Document that `TTFA2` is decoded-output-end based and experimental.
   - Fix stale browser threshold comments if touched.

## Validation

Run a full Benchmark 1 pass and inspect:

- `TTFA2` appears for completed non-barge-in turns.
- Existing `TTFA` values and summary stats are unchanged.
- Console logs include one `[bench timing]` object per completed turn.
- `delta_ms` is generally stable enough to reason about.
- No audio routing regression: primary playback still goes to selected `Output 1`.

Do not promote `TTFA2` to canonical until multiple runs show it behaves consistently.
