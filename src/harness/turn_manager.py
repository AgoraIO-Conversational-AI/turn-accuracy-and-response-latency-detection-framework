"""Turn sequencing, TTFA measurement, and barge-in detection."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from .audio_engine import AudioEngine
from .vad_engine import VadEngine

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "out"

# available sources: name -> (turns_index.json path, turns dir)
SOURCES = {
    "sovereign5": {
        "label": "Lucy and Tabby",
        "index": OUT_DIR / "turns_index.json",
        "turns_dir": OUT_DIR / "turns",
    },
    "sovereign10": {
        "label": "Ben",
        "index": OUT_DIR / "Sovereign_Place_10" / "turns_index.json",
        "turns_dir": OUT_DIR / "Sovereign_Place_10" / "turns",
    },
    "tts_turns": {
        "label": "TTS Turn Accuracy",
        "index": OUT_DIR / "TTS_Turns" / "turns_index.json",
        "turns_dir": OUT_DIR / "TTS_Turns" / "turns",
    },
    "tts_hesitation2": {
        "label": "TTS Hesitation 2",
        "index": OUT_DIR / "TTS_Hesitation2" / "turns_index.json",
        "turns_dir": OUT_DIR / "TTS_Hesitation2" / "turns",
    },
}


class TurnState(str, Enum):
    IDLE = "idle"
    PLAYING = "playing"
    WAITING_RESPONSE = "waiting_response"
    RECORDING_RESPONSE = "recording_response"


@dataclass
class TurnResult:
    turn: int
    speaker: int
    text: str
    duration_ms: int
    ttfa_ms: float | None = None
    barge_in: bool = False
    barge_in_at_ms: float | None = None
    response_duration_ms: float | None = None
    status: str = "pending"  # pending, playing, done, barge_in, no_response, skipped


@dataclass
class RunState:
    state: TurnState = TurnState.IDLE
    current_turn: int | None = None
    results: list[TurnResult] = field(default_factory=list)
    running: bool = False
    speaker_filter: int | None = None


class TurnManager:
    """Orchestrates turn playback, VAD monitoring, and result collection."""

    def __init__(
        self,
        audio_engine: AudioEngine,
        vad_engine: VadEngine,
        on_event: Callable | None = None,
        response_silence_timeout: float = 1.5,
        barge_in_silence_timeout: float = 3.0,
        max_wait_for_response: float = 10.0,
        inter_turn_delay: float = 0.5,
    ):
        self.audio = audio_engine
        self.vad = vad_engine
        self.on_event = on_event
        self.response_silence_timeout = response_silence_timeout
        self.barge_in_silence_timeout = barge_in_silence_timeout
        self.max_wait_for_response = max_wait_for_response
        self.inter_turn_delay = inter_turn_delay

        self.run = RunState()
        self._turns_data: list[dict] = []
        self._stop_requested = False
        self._current_source = "tts_turns"
        self._load_turns()

    def _load_turns(self):
        src = SOURCES[self._current_source]
        index_path = src["index"]
        if not index_path.exists():
            log.warning(f"turns index not found: {index_path}")
            self._turns_data = []
            return
        with open(index_path) as f:
            data = json.load(f)
        self._turns_data = data["turns"]

    def set_source(self, source_key: str):
        """Switch to a different audio source."""
        if source_key not in SOURCES:
            raise ValueError(f"Unknown source: {source_key}")
        self._current_source = source_key
        self._load_turns()
        self.run = RunState()
        log.info(f"switched source to {source_key}: {len(self._turns_data)} turns")

    def get_sources(self) -> list[dict]:
        """Return available sources for the UI."""
        return [
            {"key": k, "label": v["label"], "active": k == self._current_source}
            for k, v in SOURCES.items()
        ]

    def get_turns(self, speaker: int | None = None) -> list[dict]:
        turns = self._turns_data
        if speaker is not None:
            turns = [t for t in turns if t["speaker"] == speaker]
        return turns

    def _emit(self, event_type: str, data: dict = None):
        if self.on_event:
            self.on_event(event_type, data or {})

    def _wav_path(self, turn: dict) -> Path:
        src = SOURCES[self._current_source]
        return src["turns_dir"] / f"speaker{turn['speaker']}" / f"turn_{turn['turn']:03d}.wav"

    def _find_turn(self, turn_id: int) -> dict:
        """Find a turn by its ID (not list index)."""
        for t in self._turns_data:
            if t["turn"] == turn_id:
                return t
        raise ValueError(f"Turn {turn_id} not found in current source")

    async def run_single_turn(self, turn_index: int) -> TurnResult:
        """Play a single turn and measure response.

        turn_index is the turn ID (from turns_index.json), not a list position.
        VAD runs continuously from playback start. TTFA is measured from
        playback end — if the agent responds before the turn finishes,
        TTFA will be negative.
        """
        self._stop_requested = False
        turn = self._find_turn(turn_index)
        result = TurnResult(
            turn=turn["turn"],
            speaker=turn["speaker"],
            text=turn["text"],
            duration_ms=turn["duration_ms"],
        )

        wav_path = self._wav_path(turn)
        if not wav_path.exists():
            result.status = "skipped"
            self.run.results = [
                r for r in self.run.results if r.turn != result.turn
            ]
            self.run.results.append(result)
            self._emit("turn_done", {
                "turn": turn["turn"],
                "ttfa_ms": None,
                "barge_in": False,
                "barge_in_at_ms": None,
                "response_duration_ms": None,
                "status": "skipped",
                "summary": self.get_summary(),
            })
            return result

        self.run.state = TurnState.PLAYING
        self.run.current_turn = turn_index
        result.status = "playing"
        self._emit("turn_start", {
            "turn": turn["turn"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "duration_ms": turn["duration_ms"],
        })

        hesitations = turn.get("hesitations", [])
        hes_windows = []
        for hes in hesitations:
            gap_start = hes["at_ms"] - turn["start_ms"]
            hes_windows.append((gap_start, gap_start + hes["duration_ms"]))

        # reset VAD and start capture
        self.vad.reset()
        self.audio.drain_capture_queue()
        self.audio.start_capture()

        # play turn (blocking in thread)
        playback_start = time.perf_counter()
        playback_done = asyncio.Event()
        turn_duration_s = turn["duration_ms"] / 1000.0
        log.info(f"turn {turn['turn']}: playing {turn_duration_s:.1f}s")

        loop = asyncio.get_event_loop()

        async def do_playback():
            await loop.run_in_executor(None, self.audio.play_turn, wav_path)
            playback_done.set()

        playback_task = asyncio.create_task(do_playback())

        # track state across the whole turn
        first_speech_audio_ms = None  # audio-time of first agent speech (from VAD)
        last_speech_time = None  # wall-clock for silence timeout tracking
        playback_end = None  # wall-clock when play_turn() returned

        # TTFA reference: use expected duration from the index, not wall-clock.
        # play_turn() returns ~400ms late due to OS audio buffer drain, but the
        # audio content actually finishes at exactly duration_ms from start.
        expected_end_ms = float(turn["duration_ms"])

        while True:
            if self._stop_requested:
                break

            # check if playback finished BEFORE blocking on capture
            if playback_done.is_set() and playback_end is None:
                playback_end = time.perf_counter()
                actual_play_dur = playback_end - playback_start
                buffer_overhead_ms = (actual_play_dur * 1000) - expected_end_ms
                log.info(f"turn {turn['turn']}: playback func returned, actual={actual_play_dur:.3f}s expected={turn_duration_s:.3f}s (buffer overhead={buffer_overhead_ms:.0f}ms)")
                self.run.state = TurnState.WAITING_RESPONSE
                self._emit("waiting_response", {"turn": turn["turn"]})

            # drain available capture chunks (up to 50 ≈ 1s of audio) to stay real-time
            chunks_processed = 0
            max_chunks = 50
            while chunks_processed < max_chunks:
                chunk = self.audio.get_capture_chunk_16k(
                    timeout=0.02 if chunks_processed == 0 else 0.001,
                )
                if chunk is None:
                    break
                chunks_processed += 1
                events = self.vad.process_chunk(chunk)

                for ev in events:
                    if ev["type"] == "speech_start" and first_speech_audio_ms is None:
                        first_speech_audio_ms = ev["time_s"] * 1000
                        log.info(f"turn {turn['turn']}: speech_start at audio_ms={first_speech_audio_ms:.0f}ms, playback_done={'yes' if playback_done.is_set() else 'no'}")

                        # TTFA = agent speech time - turn audio end time
                        # Both in the same audio-domain clock (capture stream time)
                        ttfa = first_speech_audio_ms - expected_end_ms
                        log.info(f"turn {turn['turn']}: TTFA = {first_speech_audio_ms:.0f}ms - {expected_end_ms:.0f}ms = {ttfa:.0f}ms")

                        # barge-in: agent spoke before turn audio finished
                        if ttfa < 0:
                            result.barge_in = True
                            result.barge_in_at_ms = first_speech_audio_ms
                            log.info(f"turn {turn['turn']}: barge-in (agent spoke {-ttfa:.0f}ms before turn ended)")

                        # also check if barge-in landed in a hesitation gap
                        if not result.barge_in:
                            for gap_start, gap_end in hes_windows:
                                if gap_start <= first_speech_audio_ms <= gap_end:
                                    result.barge_in = True
                                    result.barge_in_at_ms = first_speech_audio_ms
                                    log.info(f"turn {turn['turn']}: barge-in in hesitation gap [{gap_start}-{gap_end}ms]")
                                    break

                        result.ttfa_ms = ttfa
                        if result.barge_in:
                            result.status = "barge_in"
                        log.info(f"turn {turn['turn']}: TTFA={ttfa:.0f}ms barge_in={result.barge_in}")
                        self.run.state = TurnState.RECORDING_RESPONSE
                        self._emit("response_detected", {
                            "turn": turn["turn"],
                            "ttfa_ms": round(ttfa, 1),
                        })

                    if ev["type"] == "speech_end":
                        log.info(f"turn {turn['turn']}: speech_end at audio_ms={ev['time_s'] * 1000:.0f}ms")

                if self.vad.is_speech_active:
                    last_speech_time = time.perf_counter()

            now = time.perf_counter()
            elapsed_s = now - playback_start
            past_expected_end = elapsed_s > turn_duration_s

            # timeout: no response after expected turn end + max_wait
            if past_expected_end and first_speech_audio_ms is None:
                if elapsed_s - turn_duration_s > self.max_wait_for_response:
                    log.info(f"turn {turn['turn']}: no response after {self.max_wait_for_response}s")
                    result.status = "no_response"
                    break

            # end condition: agent responded then went silent
            # don't gate on playback_end — use elapsed time past expected duration
            if first_speech_audio_ms is not None and last_speech_time is not None:
                if not self.vad.is_speech_active and past_expected_end:
                    silence_dur = now - last_speech_time
                    timeout = self.barge_in_silence_timeout if result.barge_in else self.response_silence_timeout
                    # for barge-in, also ensure we've waited at least timeout after expected end
                    if result.barge_in:
                        since_expected_end = elapsed_s - turn_duration_s
                        if since_expected_end < timeout:
                            await asyncio.sleep(0.01)
                            continue
                    if silence_dur >= timeout:
                        result.response_duration_ms = (now - silence_dur - (playback_start + first_speech_audio_ms / 1000)) * 1000
                        if result.response_duration_ms < 0:
                            result.response_duration_ms = 0
                        if not result.barge_in:
                            result.status = "done"
                        else:
                            result.status = "barge_in"
                        log.info(f"turn {turn['turn']}: agent done, response={result.response_duration_ms:.0f}ms, TTFA={result.ttfa_ms:.0f}ms, status={result.status}")
                        break

            await asyncio.sleep(0.01)

        # ensure playback task completes (with timeout to avoid hang)
        try:
            await asyncio.wait_for(playback_task, timeout=2.0)
        except asyncio.TimeoutError:
            log.warning(f"turn {turn['turn']}: playback task hung, forcing stop")
            self.audio.stop_playback()
            try:
                await asyncio.wait_for(playback_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
        self.audio.stop_capture()
        self.run.state = TurnState.IDLE

        # if stopped mid-turn, discard partial result
        if self._stop_requested:
            result.status = "stopped"
            return result

        # store result (replace if re-running same turn)
        self.run.results = [
            r for r in self.run.results if r.turn != result.turn
        ]
        self.run.results.append(result)

        self._emit("turn_done", {
            "turn": turn["turn"],
            "ttfa_ms": round(result.ttfa_ms, 1) if result.ttfa_ms is not None else None,
            "barge_in": result.barge_in,
            "barge_in_at_ms": round(result.barge_in_at_ms, 1) if result.barge_in_at_ms is not None else None,
            "response_duration_ms": round(result.response_duration_ms, 1) if result.response_duration_ms is not None else None,
            "status": result.status,
            "summary": self.get_summary(),
        })
        return result

    async def run_all(self, speaker: int | None = None):
        """Run through all turns sequentially."""
        self._stop_requested = False
        self.run.running = True
        self.run.speaker_filter = speaker
        self.run.results = []

        turns = self.get_turns(speaker)
        self._emit("run_start", {
            "total_turns": len(turns),
            "speaker": speaker,
        })

        for turn in turns:
            if self._stop_requested:
                break

            await self.run_single_turn(turn["turn"])

            if not self._stop_requested:
                await asyncio.sleep(self.inter_turn_delay)

        self.run.running = False
        self.run.state = TurnState.IDLE
        if not self._stop_requested:
            self._emit("run_complete", self.get_summary())

    def stop(self):
        """Request stop of current run and interrupt playback."""
        self._stop_requested = True
        self.audio.stop_playback()
        self.audio.stop_capture()

    def reset(self):
        """Reset all state."""
        self.stop()
        self.run = RunState()

    def get_summary(self) -> dict:
        """Compute summary statistics from results."""
        results = self.run.results
        if not results:
            return {}

        # only include positive TTFAs in stats (negative = barge-in)
        ttfas = [r.ttfa_ms for r in results if r.ttfa_ms is not None and r.ttfa_ms > 0]
        barge_ins = [r for r in results if r.barge_in]
        no_responses = [r for r in results if r.status == "no_response"]

        summary = {
            "total_turns": len(results),
            "completed": len([r for r in results if r.status == "done"]),
            "barge_in_count": len(barge_ins),
            "no_response_count": len(no_responses),
        }

        if ttfas:
            ttfas_sorted = sorted(ttfas)
            n = len(ttfas_sorted)
            summary["ttfa_avg_ms"] = round(sum(ttfas) / n, 1)
            # proper median: average of two middle values for even-length
            if n % 2 == 0:
                median = (ttfas_sorted[n // 2 - 1] + ttfas_sorted[n // 2]) / 2
            else:
                median = ttfas_sorted[n // 2]
            summary["ttfa_median_ms"] = round(median, 1)
            # p95: linear interpolation
            p95_idx = (n - 1) * 0.95
            low = int(p95_idx)
            high = min(low + 1, n - 1)
            frac = p95_idx - low
            p95 = ttfas_sorted[low] + frac * (ttfas_sorted[high] - ttfas_sorted[low])
            summary["ttfa_p95_ms"] = round(p95, 1)
            summary["ttfa_min_ms"] = round(ttfas_sorted[0], 1)
            summary["ttfa_max_ms"] = round(ttfas_sorted[-1], 1)

        summary["results"] = [
            {
                "turn": r.turn,
                "speaker": r.speaker,
                "text": r.text,
                "duration_ms": r.duration_ms,
                "ttfa_ms": round(r.ttfa_ms, 1) if r.ttfa_ms is not None else None,
                "barge_in": r.barge_in,
                "status": r.status,
            }
            for r in results
        ]

        return summary
