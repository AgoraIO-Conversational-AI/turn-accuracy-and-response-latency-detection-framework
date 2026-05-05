"""Extract per-turn WAV files from the source m4a using ffmpeg."""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TURNS_INDEX = BASE_DIR / "out" / "turns_index.json"
AUDIO_FILE = BASE_DIR / "fixtures" / "sovereign_place_5.m4a"
OUTPUT_DIR = BASE_DIR / "out" / "turns"


def extract_turns(
    index_path: Path = TURNS_INDEX,
    audio_path: Path = AUDIO_FILE,
    output_dir: Path = OUTPUT_DIR,
    speaker: int | None = None,
):
    with open(index_path) as f:
        index = json.load(f)

    turns = index["turns"]
    if speaker is not None:
        turns = [t for t in turns if t["speaker"] == speaker]

    for spk in (0, 1):
        (output_dir / f"speaker{spk}").mkdir(parents=True, exist_ok=True)

    created = 0
    for turn in turns:
        spk = turn["speaker"]
        turn_num = turn["turn"]
        start_s = turn["start_ms"] / 1000.0
        duration_s = turn["duration_ms"] / 1000.0

        out_path = output_dir / f"speaker{spk}" / f"turn_{turn_num:03d}.wav"
        if out_path.exists():
            continue

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration_s:.3f}",
            "-ac", "1",          # mono
            "-ar", "48000",      # 48 kHz
            "-sample_fmt", "s16", # 16-bit
            "-c:a", "pcm_s16le",
            str(out_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"FAIL turn {turn_num}: {result.stderr[-200:]}", file=sys.stderr)
            continue

        created += 1

    total = len(turns)
    print(f"extracted {created} new WAVs ({total} total turns)")
    return total


if __name__ == "__main__":
    extract_turns()
