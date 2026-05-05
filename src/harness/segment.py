"""Volume-based turn segmentation for audio files.

Detects turns by finding silence gaps (RMS below threshold for >= min_silence_ms).
Produces a turns_index.json-compatible structure and extracts per-turn WAVs.
Optionally transcribes each turn via Deepgram for real text previews.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _transcribe_wav(wav_path: Path, api_key: str) -> str:
    """Transcribe a WAV file via Deepgram Nova-3 and return the transcript text."""
    import urllib.request

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    url = ("https://api.deepgram.com/v1/listen?"
           "model=nova-3&language=en&punctuate=true&smart_format=true")

    req = urllib.request.Request(url, data=audio_data, headers={
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        transcript = (result.get("results", {})
                      .get("channels", [{}])[0]
                      .get("alternatives", [{}])[0]
                      .get("transcript", ""))
        return transcript.strip()
    except Exception as e:
        print(f"  transcription failed for {wav_path.name}: {e}", file=sys.stderr)
        return ""


def _load_deepgram_key() -> str | None:
    """Load Deepgram API key from .env file."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPGRAM_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1]
    return os.environ.get("DEEPGRAM_API_KEY")


def segment_audio(
    audio_path: Path,
    output_dir: Path,
    silence_threshold: float = 0.01,
    min_silence_ms: int = 1000,
    min_turn_ms: int = 200,
    target_sr: int = 48000,
    merge_groups: list[list[int]] | None = None,
    tail_pad_ms: int = 50,
    transcribe: bool = False,
) -> dict:
    """Segment audio into turns based on volume silence detection.

    Args:
        audio_path: Path to source audio file (m4a, wav, etc.)
        output_dir: Directory to write turns_index.json and turn WAVs
        silence_threshold: RMS threshold below which audio is considered silence
        min_silence_ms: Minimum silence duration (ms) to split turns
        min_turn_ms: Minimum turn duration (ms) to keep
        target_sr: Target sample rate for output WAVs

    Returns:
        turns_index dict with same structure as Soniox-generated one
    """
    # decode to WAV in memory via ffmpeg
    tmp_wav = output_dir / "_temp_decode.wav"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ac", "1", "-ar", str(target_sr),
        "-sample_fmt", "s16", "-c:a", "pcm_s16le",
        str(tmp_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"ffmpeg decode failed: {result.stderr[-300:]}", file=sys.stderr)
        return {}

    data, sr = sf.read(str(tmp_wav), dtype="float32")
    tmp_wav.unlink()

    if data.ndim > 1:
        data = data[:, 0]

    # compute RMS in windows
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    hop_samples = window_samples  # no overlap

    rms_values = []
    for i in range(0, len(data) - window_samples, hop_samples):
        window = data[i:i + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)

    # find speech/silence regions
    is_speech = [rms >= silence_threshold for rms in rms_values]

    # find turn boundaries: contiguous speech regions separated by silence >= min_silence_ms
    min_silence_windows = int(min_silence_ms / window_ms)
    min_turn_windows = int(min_turn_ms / window_ms)

    turns = []
    in_turn = False
    turn_start = 0
    silence_count = 0

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_turn:
                in_turn = True
                turn_start = i
                silence_count = 0
            else:
                silence_count = 0
        else:
            if in_turn:
                silence_count += 1
                if silence_count >= min_silence_windows:
                    # end of turn
                    turn_end = i - silence_count
                    if turn_end - turn_start >= min_turn_windows:
                        turns.append((turn_start, turn_end))
                    in_turn = False
                    silence_count = 0

    # handle last turn if still in speech
    if in_turn:
        turn_end = len(is_speech) - 1
        if turn_end - turn_start >= min_turn_windows:
            turns.append((turn_start, turn_end))

    # convert to ms and build index
    turns_data = []
    for idx, (start_w, end_w) in enumerate(turns):
        start_ms = start_w * window_ms
        end_ms = end_w * window_ms
        duration_ms = end_ms - start_ms

        turns_data.append({
            "turn": idx,
            "speaker": 0,  # single speaker for self-recorded audio
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "text": f"Turn {idx} ({duration_ms}ms)",
            "word_count": 0,
            "hesitations": [],
            "max_hesitation_ms": 0,
        })

    # merge groups: combine specified turn indices into single turns
    if merge_groups:
        merged_indices = set()
        for group in merge_groups:
            merged_indices.update(group)

        new_turns = []
        merged_set = set()
        for turn in turns_data:
            idx = turn["turn"]
            if idx in merged_set:
                continue
            # check if this turn starts a merge group
            group = None
            for g in merge_groups:
                if idx == g[0]:
                    group = g
                    break
            if group:
                # merge: take start from first, end from last
                first = turns_data[group[0]]
                last = turns_data[group[-1]]
                duration_ms = last["end_ms"] - first["start_ms"]
                # gaps between sub-turns become hesitations
                hesitations = []
                for i in range(len(group) - 1):
                    prev = turns_data[group[i]]
                    nxt = turns_data[group[i + 1]]
                    gap_ms = nxt["start_ms"] - prev["end_ms"]
                    if gap_ms > 0:
                        hesitations.append({
                            "at_ms": prev["end_ms"],
                            "duration_ms": gap_ms,
                        })
                max_hes = max((h["duration_ms"] for h in hesitations), default=0)
                new_turns.append({
                    **first,
                    "end_ms": last["end_ms"],
                    "duration_ms": duration_ms,
                    "text": f"Turn {len(new_turns)} ({duration_ms}ms)",
                    "hesitations": hesitations,
                    "max_hesitation_ms": max_hes,
                })
                merged_set.update(group)
            elif idx not in merged_indices:
                new_turns.append(turn)

        # re-index
        for i, t in enumerate(new_turns):
            t["turn"] = i
            t["text"] = f"Turn {i} ({t['duration_ms']}ms)"
        turns_data = new_turns
        print(f"after merging: {len(turns_data)} turns")

    index = {
        "audio_file": audio_path.name,
        "provider": "volume",
        "total_turns": len(turns_data),
        "speaker_0_turns": len(turns_data),
        "speaker_1_turns": 0,
        "silence_threshold": silence_threshold,
        "min_silence_ms": min_silence_ms,
        "turns": turns_data,
    }

    # write index
    index_path = output_dir / "turns_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"segmented {len(turns_data)} turns from {audio_path.name}")

    # extract per-turn WAVs
    turn_dir = output_dir / "turns" / "speaker0"
    turn_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for turn in turns_data:
        out_path = turn_dir / f"turn_{turn['turn']:03d}.wav"
        if out_path.exists():
            continue

        start_s = turn["start_ms"] / 1000.0
        duration_s = (turn["duration_ms"] + tail_pad_ms) / 1000.0

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration_s:.3f}",
            "-ac", "1",
            "-ar", str(target_sr),
            "-sample_fmt", "s16",
            "-c:a", "pcm_s16le",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"FAIL turn {turn['turn']}: {r.stderr[-200:]}", file=sys.stderr)
            continue
        created += 1

    print(f"extracted {created} new WAVs ({len(turns_data)} total turns)")

    # transcribe turns via Deepgram if requested
    if transcribe:
        api_key = _load_deepgram_key()
        if not api_key:
            print("WARNING: --transcribe requires DEEPGRAM_API_KEY in .env or environment",
                  file=sys.stderr)
        else:
            print("transcribing turns via Deepgram...")
            for turn in turns_data:
                wav_path = turn_dir / f"turn_{turn['turn']:03d}.wav"
                if not wav_path.exists():
                    continue
                text = _transcribe_wav(wav_path, api_key)
                if text:
                    words = text.split()
                    turn["text"] = text
                    turn["word_count"] = len(words)
                    print(f"  turn {turn['turn']}: \"{text[:60]}...\"" if len(text) > 60
                          else f"  turn {turn['turn']}: \"{text}\"")
                else:
                    turn["text"] = f"Turn {turn['turn']} ({turn['duration_ms']}ms)"

            # re-write index with transcripts
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
            print(f"updated turns_index.json with transcripts")

    return index


def transcribe_existing(output_dir: Path) -> None:
    """Transcribe existing turn WAVs and update turns_index.json with real text."""
    index_path = output_dir / "turns_index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        return

    api_key = _load_deepgram_key()
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY required in .env or environment", file=sys.stderr)
        return

    with open(index_path) as f:
        index = json.load(f)

    turns_data = index["turns"]
    turn_dir = output_dir / "turns" / "speaker0"

    print(f"transcribing {len(turns_data)} turns via Deepgram...")
    for turn in turns_data:
        wav_path = turn_dir / f"turn_{turn['turn']:03d}.wav"
        if not wav_path.exists():
            print(f"  turn {turn['turn']}: WAV missing, skipping", file=sys.stderr)
            continue
        text = _transcribe_wav(wav_path, api_key)
        if text:
            turn["text"] = text
            turn["word_count"] = len(text.split())
            print(f"  turn {turn['turn']}: \"{text[:60]}...\"" if len(text) > 60
                  else f"  turn {turn['turn']}: \"{text}\"")
        else:
            print(f"  turn {turn['turn']}: no transcript returned")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"updated {index_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", help="Path to audio file (not needed with --transcribe-only)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="RMS silence threshold")
    parser.add_argument("--min-silence", type=int, default=1000, help="Min silence gap (ms)")
    parser.add_argument("--min-turn", type=int, default=200, help="Min turn duration (ms)")
    parser.add_argument("--tail-pad", type=int, default=50, help="Extra ms appended to each WAV")
    parser.add_argument("--merge", action="append", help="Merge turns, e.g. --merge 1,2 --merge 7,8,9")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe turns via Deepgram after segmenting")
    parser.add_argument("--transcribe-only", dest="transcribe_only", action="store_true",
                        help="Only transcribe existing turns (requires --output-dir)")
    args = parser.parse_args()

    if args.transcribe_only:
        if not args.output_dir:
            print("ERROR: --transcribe-only requires --output-dir", file=sys.stderr)
            sys.exit(1)
        transcribe_existing(Path(args.output_dir))
    else:
        if not args.audio:
            parser.error("audio path is required (unless using --transcribe-only)")

        merge_groups = None
        if args.merge:
            merge_groups = [[int(x) for x in m.split(",")] for m in args.merge]

        audio_path = Path(args.audio)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = BASE_DIR / "out" / audio_path.stem.replace(" ", "_")

        segment_audio(
            audio_path=audio_path,
            output_dir=output_dir,
            silence_threshold=args.threshold,
            min_silence_ms=args.min_silence,
            min_turn_ms=args.min_turn,
            merge_groups=merge_groups,
            tail_pad_ms=args.tail_pad,
            transcribe=args.transcribe,
        )
