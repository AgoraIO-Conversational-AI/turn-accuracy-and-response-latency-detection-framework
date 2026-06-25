"""Generate TTS turn-accuracy test audio via ElevenLabs.

Produces WAVs across five categories:
  - Normal (5): semantically and prosodically complete sentences
  - Hesitation (5): sentences with filler words (um/uh) and [short pause] tags
  - Pause (5): mid-sentence [pause] tag producing natural silence
  - Ambiguous (5): trailing-off sentences that sound potentially complete
  - Hesitation2 (5): prosody-only pauses via [hesitation] tags, no fillers

Uses the /with-timestamps endpoint to get character-level alignment data,
then measures actual speech boundaries via RMS amplitude analysis to
populate precise timing metadata in turns_index.json.

After TTS synthesis, silence gaps are measured and stretched to a
per-sentence target duration covering 500-1200ms. This range spans
common VAD endpointing thresholds:
  - LiveKit default min_delay = 500ms
  - Some providers use 640ms
  - Deepgram Flux max_turn_silence up to 5000ms
Each sentence has a fixed target_gap_ms so the test suite covers the
full range, testing whether agents incorrectly take turns at each level.

Usage:
    python -m src.harness.generate_tts [--force] [--voice 0-4] [--skip-existing]
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "out" / "TTS_Turns"

# ── Voice mapping ──────────────────────────────────────────────────────────
VOICES = {
    0: "XnKbmWxx8uWjruHkpXmf",
    1: "maYJAY8nOIBZeB0UYfc5",
    2: "QoC8og5VjCQoTz0caaaO",
    3: "VUGQSU6BSEjkbudnJbOj",
    4: "1AKkSX7KMPHIWuz76m0n",
}

# ── Sentence definitions ───────────────────────────────────────────────────
# 25 total turns: 5 normal + 5 hesitation + 5 pause + 5 ambiguous + 5 hesitation2.
# Each sentence assigned to one voice. Order is deliberately irregular
# so the LLM under test cannot predict the pattern.
# Hesitation/pause sentences have exactly one silence point each.

SENTENCES = [
    # 0 — normal (instruction)
    {
        "voice": 2,
        "category": "normal",
        "tts_text": "I'm going to say some random things to see how you respond. Please keep your responses to under ten words.",
        "display_text": "I'm going to say some random things to see how you respond. Please keep your responses to under ten words.",
        "expected_complete": True,
    },
    # 1 — pause 650ms
    {
        "voice": 1,
        "category": "pause",
        "tts_text": "The order number is [pause] [pause] [short pause] seven four two nine one.",
        "display_text": "The order number is [pause] seven four two nine one.",
        "expected_complete": False,
        "target_gap_ms": 650,
    },
    # 2 — hesitation 800ms
    {
        "voice": 2,
        "category": "hesitation",
        "tts_text": "I need to update my [hesitation] [hesitation] [pause] um, my billing address.",
        "display_text": "I need to update my... um, my billing address.",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 3 — hesitation2 600ms
    {
        "voice": 0,
        "category": "hesitation2",
        "tts_text": "I think the issue is that the [hesitation] [short pause] delivery was supposed to arrive on Monday.",
        "display_text": "I think the issue is that the [hesitation] [short pause] delivery was supposed to arrive on Monday.",
        "expected_complete": False,
        "target_gap_ms": 600,
    },
    # 4 — ambiguous
    {
        "voice": 2,
        "category": "ambiguous",
        "tts_text": "That's not really what I was looking for but.",
        "display_text": "That's not really what I was looking for but.",
        "expected_complete": True,
    },
    # 5 — normal
    {
        "voice": 0,
        "category": "normal",
        "tts_text": "I'd like to book a table for two at seven o'clock tonight please.",
        "display_text": "I'd like to book a table for two at seven o'clock tonight please.",
        "expected_complete": True,
    },
    # 6 — pause 1000ms
    {
        "voice": 3,
        "category": "pause",
        "tts_text": "My account number starts with [pause] [pause] [short pause] eight six three.",
        "display_text": "My account number starts with [pause] eight six three.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 7 — hesitation2 800ms
    {
        "voice": 1,
        "category": "hesitation2",
        "tts_text": "The last time I checked it was [hesitation] [hesitation] [pause] somewhere around forty five dollars.",
        "display_text": "The last time I checked it was [hesitation] [hesitation] [pause] somewhere around forty five dollars.",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 8 — ambiguous
    {
        "voice": 3,
        "category": "ambiguous",
        "tts_text": "I already tried that once before so.",
        "display_text": "I already tried that once before so.",
        "expected_complete": True,
    },
    # 9 — normal
    {
        "voice": 4,
        "category": "normal",
        "tts_text": "Thanks for your help, I appreciate it.",
        "display_text": "Thanks for your help, I appreciate it.",
        "expected_complete": True,
    },
    # 10 — hesitation 500ms
    {
        "voice": 0,
        "category": "hesitation",
        "tts_text": "I was thinking maybe [hesitation] [pause] actually could we do Friday instead?",
        "display_text": "I was thinking maybe... actually could we do Friday instead?",
        "expected_complete": False,
        "target_gap_ms": 500,
    },
    # 11 — hesitation2 1000ms
    {
        "voice": 2,
        "category": "hesitation2",
        "tts_text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "display_text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 12 — pause 500ms
    {
        "voice": 0,
        "category": "pause",
        "tts_text": "I wanted to ask about the [pause] [pause] [short pause] cancellation policy for next week.",
        "display_text": "I wanted to ask about the [pause] cancellation policy for next week.",
        "expected_complete": False,
        "target_gap_ms": 500,
    },
    # 13 — ambiguous
    {
        "voice": 4,
        "category": "ambiguous",
        "tts_text": "I think that might actually work but I'm not sure.",
        "display_text": "I think that might actually work but I'm not sure.",
        "expected_complete": True,
    },
    # 14 — normal
    {
        "voice": 3,
        "category": "normal",
        "tts_text": "No, I don't have any other questions at the moment.",
        "display_text": "No, I don't have any other questions at the moment.",
        "expected_complete": True,
    },
    # 15 — hesitation 1200ms
    {
        "voice": 1,
        "category": "hesitation",
        "tts_text": "We could also try [hesitation] [hesitation] [pause] yeah, the other location might work better.",
        "display_text": "We could also try... yeah, the other location might work better.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 16 — hesitation2 1200ms
    {
        "voice": 3,
        "category": "hesitation2",
        "tts_text": "I was going to renew but then the [hesitation] [hesitation] [pause] [pause] price went up by almost double.",
        "display_text": "I was going to renew but then the [hesitation] [hesitation] [pause] [pause] price went up by almost double.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 17 — pause 800ms
    {
        "voice": 2,
        "category": "pause",
        "tts_text": "Could you transfer me to [pause] [pause] [short pause] the billing department please?",
        "display_text": "Could you transfer me to [pause] the billing department please?",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 18 — hesitation 650ms
    {
        "voice": 1,
        "category": "hesitation",
        "tts_text": "So the total comes to [hesitation] [pause] uh, let me check that again.",
        "display_text": "So the total comes to... uh, let me check that again.",
        "expected_complete": False,
        "target_gap_ms": 650,
    },
    # 19 — ambiguous
    {
        "voice": 0,
        "category": "ambiguous",
        "tts_text": "It's more of a personal thing I suppose, I don't know.",
        "display_text": "It's more of a personal thing I suppose, I don't know.",
        "expected_complete": True,
    },
    # 20 — normal
    {
        "voice": 1,
        "category": "normal",
        "tts_text": "Can you confirm my reservation for Saturday the fourteenth?",
        "display_text": "Can you confirm my reservation for Saturday the fourteenth?",
        "expected_complete": True,
    },
    # 21 — pause 1200ms
    {
        "voice": 0,
        "category": "pause",
        "tts_text": "I'm calling because [pause] [pause] [short pause] I received the wrong item yesterday.",
        "display_text": "I'm calling because [pause] I received the wrong item yesterday.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 22 — hesitation 1000ms
    {
        "voice": 3,
        "category": "hesitation",
        "tts_text": "The appointment was for [hesitation] [hesitation] [pause] uh, I think it was three thirty.",
        "display_text": "The appointment was for... uh, I think it was three thirty.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 23 — hesitation2 1500ms
    {
        "voice": 4,
        "category": "hesitation2",
        "tts_text": "The problem is that my old [hesitation] [hesitation] [pause] [pause] subscription was cancelled without any notice.",
        "display_text": "The problem is that my old [hesitation] [hesitation] [pause] [pause] subscription was cancelled without any notice.",
        "expected_complete": False,
        "target_gap_ms": 1500,
    },
    # 24 — ambiguous
    {
        "voice": 1,
        "category": "ambiguous",
        "tts_text": "I mean I'm not entirely sure about that, it's hard to say.",
        "display_text": "I mean I'm not entirely sure about that, it's hard to say.",
        "expected_complete": True,
    },
]

# ── Hesitation2 sentences ─────────────────────────────────────────────────
# Prosody-only hesitations: no filler words (um/uh) and no "..." in display.
# ElevenLabs uses [hesitation]/[pause] tags to produce prosodic lengthening
# on the last word before the gap, making it sound like the speaker is
# thinking rather than using explicit fillers. Larger gaps (up to 1500ms).

HESITATION2_SENTENCES = [
    # 0 — hesitation2 600ms
    {
        "voice": 0,
        "category": "hesitation2",
        "tts_text": "I think the issue is that the [hesitation] [short pause] delivery was supposed to arrive on Monday.",
        "display_text": "I think the issue is that the [hesitation] [short pause] delivery was supposed to arrive on Monday.",
        "expected_complete": False,
        "target_gap_ms": 600,
    },
    # 1 — hesitation2 800ms
    {
        "voice": 1,
        "category": "hesitation2",
        "tts_text": "The last time I checked it was [hesitation] [hesitation] [pause] somewhere around forty five dollars.",
        "display_text": "The last time I checked it was [hesitation] [hesitation] [pause] somewhere around forty five dollars.",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 2 — hesitation2 1000ms
    {
        "voice": 2,
        "category": "hesitation2",
        "tts_text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "display_text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 3 — hesitation2 1200ms
    {
        "voice": 3,
        "category": "hesitation2",
        "tts_text": "I was going to renew but then the [hesitation] [hesitation] [pause] [pause] price went up by almost double.",
        "display_text": "I was going to renew but then the [hesitation] [hesitation] [pause] [pause] price went up by almost double.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 4 — hesitation2 1500ms
    {
        "voice": 4,
        "category": "hesitation2",
        "tts_text": "The problem is that my old [hesitation] [hesitation] [pause] [pause] subscription was cancelled without any notice.",
        "display_text": "The problem is that my old [hesitation] [hesitation] [pause] [pause] subscription was cancelled without any notice.",
        "expected_complete": False,
        "target_gap_ms": 1500,
    },
]

# ── Benchmark 1 corpus ────────────────────────────────────────────────────
# 20-turn corpus mixing the existing TTS_Turns keepers with a billing-call
# scenario at the top. All non-zero gaps target 800-1500ms and are
# bit-perfectly zeroed during post-processing (no comfort noise), so any
# detected silence in the WAV is a true zero-amplitude region.
#
# Layout:
#   0 — opener / instructions (kept from original corpus)
#   1-6 — billing-call scenario (six new utterances spread across speakers)
#   7-19 — keepers from the original corpus (long-gap pause/hesitation +
#           a few normal/ambiguous turns)
SENTENCES_BENCHMARK1 = [
    # 0 — normal (instruction opener)
    {
        "voice": 2,
        "category": "normal",
        "tts_text": "I'm going to say some random things to see how you respond. Please keep your responses to under ten words.",
        "display_text": "I'm going to say some random things to see how you respond. Please keep your responses to under ten words.",
        "expected_complete": True,
    },
    # 1 — pause 900ms (billing amount)
    {
        "voice": 0,
        "category": "pause",
        "tts_text": "Yeah, I got a bill for like [pause] [pause] [short pause] six hundred and eighty dollars and I can't pay that all today.",
        "display_text": "Yeah, I got a bill for like... six hundred and eighty dollars and I can't pay that all today.",
        "expected_complete": False,
        "target_gap_ms": 900,
    },
    # 2 — hesitation 1000ms (name + date of birth)
    {
        "voice": 1,
        "category": "hesitation",
        "tts_text": "Michael Turner, [hesitation] [hesitation] [pause] um, April fourteenth, nineteen eighty five.",
        "display_text": "Michael Turner... um, April 14, 1985.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 3 — normal (short complete answer)
    {
        "voice": 2,
        "category": "normal",
        "tts_text": "Payment plan.",
        "display_text": "Payment plan.",
        "expected_complete": True,
    },
    # 4 — hesitation2 900ms (prosody-only)
    {
        "voice": 3,
        "category": "hesitation2",
        "tts_text": "Ask them [hesitation] [hesitation] [pause] why insurance didn't cover it.",
        "display_text": "Ask them [hesitation] [hesitation] [pause] why insurance didn't cover it.",
        "expected_complete": False,
        "target_gap_ms": 900,
    },
    # 5 — normal (decision confirmation)
    {
        "voice": 4,
        "category": "normal",
        "tts_text": "That's fine, just do the payment plan.",
        "display_text": "That's fine, just do the payment plan.",
        "expected_complete": True,
    },
    # 6 — ambiguous (one-word reply)
    {
        "voice": 0,
        "category": "ambiguous",
        "tts_text": "Yes.",
        "display_text": "Yes.",
        "expected_complete": True,
    },
    # 7 — hesitation 800ms (kept from original turn 2)
    {
        "voice": 2,
        "category": "hesitation",
        "tts_text": "I need to update my [hesitation] [hesitation] [pause] um, my billing address.",
        "display_text": "I need to update my... um, my billing address.",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 8 — ambiguous (kept from original turn 4)
    {
        "voice": 2,
        "category": "ambiguous",
        "tts_text": "That's not really what I was looking for but.",
        "display_text": "That's not really what I was looking for but.",
        "expected_complete": True,
    },
    # 9 — normal (kept from original turn 5)
    {
        "voice": 0,
        "category": "normal",
        "tts_text": "I'd like to book a table for two at seven o'clock tonight please.",
        "display_text": "I'd like to book a table for two at seven o'clock tonight please.",
        "expected_complete": True,
    },
    # 10 — pause 1000ms (kept from original turn 6)
    {
        "voice": 3,
        "category": "pause",
        "tts_text": "My account number starts with [pause] [pause] [short pause] eight six three.",
        "display_text": "My account number starts with [pause] eight six three.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 11 — hesitation2 800ms (kept from original turn 7)
    {
        "voice": 1,
        "category": "hesitation2",
        "tts_text": "The last time I checked it was [hesitation] [hesitation] [pause] somewhere around forty five dollars.",
        "display_text": "The last time I checked it was [hesitation] [hesitation] [pause] somewhere around forty five dollars.",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 12 — hesitation2 1000ms (kept from original turn 11)
    {
        "voice": 2,
        "category": "hesitation2",
        "tts_text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "display_text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 13 — hesitation 1200ms (kept from original turn 15)
    {
        "voice": 1,
        "category": "hesitation",
        "tts_text": "We could also try [hesitation] [hesitation] [pause] yeah, the other location might work better.",
        "display_text": "We could also try... yeah, the other location might work better.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 14 — hesitation2 1200ms (kept from original turn 16)
    {
        "voice": 3,
        "category": "hesitation2",
        "tts_text": "I was going to renew but then the [hesitation] [hesitation] [pause] [pause] price went up by almost double.",
        "display_text": "I was going to renew but then the [hesitation] [hesitation] [pause] [pause] price went up by almost double.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 15 — pause 800ms (kept from original turn 17)
    {
        "voice": 2,
        "category": "pause",
        "tts_text": "Could you transfer me to [pause] [pause] [short pause] the billing department please?",
        "display_text": "Could you transfer me to [pause] the billing department please?",
        "expected_complete": False,
        "target_gap_ms": 800,
    },
    # 16 — pause 1200ms (kept from original turn 21)
    {
        "voice": 0,
        "category": "pause",
        "tts_text": "I'm calling because [pause] [pause] [short pause] I received the wrong item yesterday.",
        "display_text": "I'm calling because [pause] I received the wrong item yesterday.",
        "expected_complete": False,
        "target_gap_ms": 1200,
    },
    # 17 — hesitation 1000ms (kept from original turn 22)
    {
        "voice": 3,
        "category": "hesitation",
        "tts_text": "The appointment was for [hesitation] [hesitation] [pause] uh, I think it was three thirty.",
        "display_text": "The appointment was for... uh, I think it was three thirty.",
        "expected_complete": False,
        "target_gap_ms": 1000,
    },
    # 18 — hesitation2 1500ms (kept from original turn 23)
    {
        "voice": 4,
        "category": "hesitation2",
        "tts_text": "The problem is that my old [hesitation] [hesitation] [pause] [pause] subscription was cancelled without any notice.",
        "display_text": "The problem is that my old [hesitation] [hesitation] [pause] [pause] subscription was cancelled without any notice.",
        "expected_complete": False,
        "target_gap_ms": 1500,
    },
    # 19 — ambiguous (kept from original turn 24)
    {
        "voice": 1,
        "category": "ambiguous",
        "tts_text": "I mean I'm not entirely sure about that, it's hard to say.",
        "display_text": "I mean I'm not entirely sure about that, it's hard to say.",
        "expected_complete": True,
    },
    # 20 — normal (kept from original turn 9; closes the session)
    {
        "voice": 4,
        "category": "normal",
        "tts_text": "Thanks for your help, I appreciate it.",
        "display_text": "Thanks for your help, I appreciate it.",
        "expected_complete": True,
    },
]

API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
MODEL_ID = "eleven_v3"
TARGET_SR = 48000
RATE_LIMIT_SLEEP = 0.5

# RMS amplitude analysis settings
RMS_WINDOW_MS = 20
RMS_SILENCE_THRESHOLD = 0.005  # below this RMS = silence


def _load_tts_key() -> str | None:
    """Load ElevenLabs API key from .env or environment."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TTS_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1]
    return os.environ.get("TTS_KEY")


def _synthesize_with_timestamps(
    text: str, voice_id: str, api_key: str, out_mp3: Path,
) -> dict | None:
    """Call ElevenLabs /with-timestamps endpoint.

    Returns the full JSON response (with audio_base64 and alignment) on success,
    or None on failure. Also writes the decoded MP3 to out_mp3.
    """
    import urllib.request

    url = f"{API_URL}/{voice_id}/with-timestamps"
    payload = json.dumps({
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.8,
            "speed": 0.95,
        },
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    })

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        audio_bytes = base64.b64decode(data["audio_base64"])
        with open(out_mp3, "wb") as f:
            f.write(audio_bytes)

        return data
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert MP3 to 48kHz mono PCM 16-bit WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ac", "1", "-ar", str(TARGET_SR),
        "-sample_fmt", "s16", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  ffmpeg error: {r.stderr[-200:]}", file=sys.stderr)
        return False
    return True


def _wav_duration_ms(wav_path: Path) -> int:
    """Get WAV duration in milliseconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return 0
    return int(float(r.stdout.strip()) * 1000)


def _analyze_speech_boundaries(wav_path: Path) -> dict:
    """Analyze WAV amplitude to find precise speech start/end and silence gaps.

    Returns dict with:
      - speech_start_ms: first moment RMS exceeds threshold
      - speech_end_ms: last moment RMS exceeds threshold
      - silence_gaps: list of {start_ms, end_ms, duration_ms} for internal gaps
      - rms_peak: peak RMS value across the file
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    window_samples = int(sr * RMS_WINDOW_MS / 1000)
    hop_samples = window_samples

    rms_values = []
    for i in range(0, len(data) - window_samples, hop_samples):
        window = data[i:i + window_samples]
        rms = float(np.sqrt(np.mean(window ** 2)))
        rms_values.append(rms)

    if not rms_values:
        return {
            "speech_start_ms": 0, "speech_end_ms": 0,
            "silence_gaps": [], "rms_peak": 0.0,
        }

    is_speech = [rms >= RMS_SILENCE_THRESHOLD for rms in rms_values]

    # find first and last speech frames
    speech_start_ms = 0
    speech_end_ms = len(rms_values) * RMS_WINDOW_MS
    for i, s in enumerate(is_speech):
        if s:
            speech_start_ms = i * RMS_WINDOW_MS
            break

    for i in range(len(is_speech) - 1, -1, -1):
        if is_speech[i]:
            speech_end_ms = (i + 1) * RMS_WINDOW_MS
            break

    # find internal silence gaps (>= 200ms of silence between speech)
    min_gap_frames = int(450 / RMS_WINDOW_MS)
    silence_gaps = []
    in_silence = False
    gap_start = 0

    start_frame = speech_start_ms // RMS_WINDOW_MS
    end_frame = speech_end_ms // RMS_WINDOW_MS

    for i in range(start_frame, min(end_frame, len(is_speech))):
        if not is_speech[i]:
            if not in_silence:
                in_silence = True
                gap_start = i
        else:
            if in_silence:
                gap_len = i - gap_start
                if gap_len >= min_gap_frames:
                    silence_gaps.append({
                        "start_ms": gap_start * RMS_WINDOW_MS,
                        "end_ms": i * RMS_WINDOW_MS,
                        "duration_ms": gap_len * RMS_WINDOW_MS,
                    })
                in_silence = False

    return {
        "speech_start_ms": speech_start_ms,
        "speech_end_ms": speech_end_ms,
        "silence_gaps": silence_gaps,
        "rms_peak": round(max(rms_values), 4),
    }


def _enforce_single_gap(wav_path: Path) -> None:
    """Ensure a WAV has at most one internal silence gap.

    If multiple gaps are detected, merges them into one by removing the
    audio between the start of the first gap and the end of the last gap,
    replacing it with a single silent region. This preserves speech before
    the first gap and after the last gap.
    """
    boundaries = _analyze_speech_boundaries(wav_path)
    gaps = boundaries["silence_gaps"]

    if len(gaps) <= 1:
        return

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    # merge: cut from start of first gap to end of last gap,
    # replace with silence equal to the largest single gap
    first_start_ms = gaps[0]["start_ms"]
    last_end_ms = gaps[-1]["end_ms"]
    largest_gap_ms = max(g["duration_ms"] for g in gaps)

    first_start_sample = int(sr * first_start_ms / 1000)
    last_end_sample = int(sr * last_end_ms / 1000)
    replacement_samples = int(sr * largest_gap_ms / 1000)

    silence = np.zeros(replacement_samples, dtype="float32")
    data = np.concatenate([data[:first_start_sample], silence, data[last_end_sample:]])

    sf.write(str(wav_path), data, sr, subtype="PCM_16")


def _make_comfort_noise(sr: int, duration_samples: int, reference_data: np.ndarray) -> np.ndarray:
    """Generate comfort noise matching the background level of the reference audio.

    Samples a quiet portion of reference_data to estimate room noise level,
    then generates pink-ish noise at that amplitude. Falls back to low-level
    noise if no quiet region is found.
    """
    # estimate background noise from the quietest 20% of 20ms windows
    window = int(sr * 0.02)
    rms_vals = []
    for i in range(0, len(reference_data) - window, window):
        rms = float(np.sqrt(np.mean(reference_data[i:i + window] ** 2)))
        if rms > 0:
            rms_vals.append(rms)

    if rms_vals:
        rms_vals.sort()
        # use the 20th percentile as the noise floor estimate
        noise_level = rms_vals[max(0, len(rms_vals) // 5)]
        # keep below RMS silence threshold so gap detection still works,
        # but high enough to sound like room presence (not digital silence)
        noise_level = min(max(noise_level, 0.002), RMS_SILENCE_THRESHOLD * 0.6)
    else:
        noise_level = 0.003

    # generate noise with slight low-pass character (pink-ish)
    raw = np.random.randn(duration_samples).astype("float32")
    # simple low-pass: running average over ~2ms
    kernel_size = max(1, int(sr * 0.002))
    kernel = np.ones(kernel_size, dtype="float32") / kernel_size
    filtered = np.convolve(raw, kernel, mode="same")
    # normalize to target level
    current_rms = float(np.sqrt(np.mean(filtered ** 2)))
    if current_rms > 0:
        filtered = filtered * (noise_level / current_rms)

    return filtered


def _zero_out_gap(wav_path: Path, fade_ms: int = 3) -> dict | None:
    """Bit-perfectly zero the largest silence gap, with a short fade at edges.

    Reads the current gap, walks outward from its midpoint until each side
    hits the first non-silent sample, then writes integer zeros in between.
    A short linear fade just outside each edge avoids click artifacts.

    Returns the {start_ms, end_ms, duration_ms} of the zeroed region (sample-
    exact, not RMS-window-quantized), or None if no gap was found.
    """
    boundaries = _analyze_speech_boundaries(wav_path)
    gaps = boundaries["silence_gaps"]
    if not gaps:
        return None

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    gap = max(gaps, key=lambda g: g["duration_ms"])
    start = int(sr * gap["start_ms"] / 1000)
    end = int(sr * gap["end_ms"] / 1000)
    start = max(0, start)
    end = min(len(data), end)
    if end <= start:
        return None

    data[start:end] = 0.0

    fade_samples = max(1, int(sr * fade_ms / 1000))
    pre_start = max(0, start - fade_samples)
    if start - pre_start > 0:
        ramp = np.linspace(1.0, 0.0, start - pre_start).astype("float32")
        data[pre_start:start] *= ramp
    post_end = min(len(data), end + fade_samples)
    if post_end - end > 0:
        ramp = np.linspace(0.0, 1.0, post_end - end).astype("float32")
        data[end:post_end] *= ramp

    sf.write(str(wav_path), data, sr, subtype="PCM_16")

    return {
        "start_ms": int(round(start * 1000 / sr)),
        "end_ms": int(round(end * 1000 / sr)),
        "duration_ms": int(round((end - start) * 1000 / sr)),
    }


def _find_zero_run(wav_path: Path, min_ms: int = 50) -> dict | None:
    """Find the longest contiguous run of bit-exact zero samples in the WAV.

    Returns {start_ms, end_ms, duration_ms} of the longest zero run that is
    at least min_ms long, or None if no such run exists.
    """
    data, sr = sf.read(str(wav_path), dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]

    is_zero = (data == 0).astype(np.int8)
    if not is_zero.any():
        return None

    edges = np.diff(np.concatenate([[0], is_zero, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return None

    min_samples = int(sr * min_ms / 1000)
    best = None
    best_len = 0
    for s, e in zip(starts, ends):
        if (e - s) >= min_samples and (e - s) > best_len:
            best_len = e - s
            best = (s, e)

    if best is None:
        return None

    s, e = best
    return {
        "start_ms": int(round(s * 1000 / sr)),
        "end_ms": int(round(e * 1000 / sr)),
        "duration_ms": int(round((e - s) * 1000 / sr)),
    }


def _set_gap_duration(
    wav_path: Path, target_ms: int, zero_fill: bool = False,
) -> list[dict]:
    """Ensure exactly one silence gap at exactly target_ms duration.

    First enforces a single gap (removes extras), then trims or stretches
    it to hit the target precisely. Stretching inserts comfort noise instead
    of digital silence so the gap sounds like a natural room pause.

    When zero_fill=True the gap is bit-perfectly zeroed after sizing, with a
    short linear fade just outside each edge to avoid click artifacts. Used
    by the Benchmark 1 corpus so amplitude-driven EOT detectors see a true
    zero-amplitude window.
    Returns the final gap list.
    """
    _enforce_single_gap(wav_path)

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    boundaries = _analyze_speech_boundaries(wav_path)
    gaps = boundaries["silence_gaps"]

    def _fill(n_samples: int) -> np.ndarray:
        if zero_fill:
            return np.zeros(n_samples, dtype="float32")
        return _make_comfort_noise(sr, n_samples, data)

    if not gaps:
        # no gap detected — force-insert one at the quietest point
        # scan for the quietest 100ms region in the middle 60% of the audio
        window_samples = int(sr * 0.1)
        search_start = int(len(data) * 0.2)
        search_end = int(len(data) * 0.8) - window_samples
        min_rms = float("inf")
        min_pos = search_start
        for i in range(search_start, search_end, int(sr * 0.01)):
            rms = float(np.sqrt(np.mean(data[i:i + window_samples] ** 2)))
            if rms < min_rms:
                min_rms = rms
                min_pos = i

        target_samples = int(sr * target_ms / 1000)
        filler = _fill(target_samples)
        # apply brief crossfade
        fade = min(int(sr * 0.01), target_samples // 4)
        data[min_pos - fade:min_pos] *= np.linspace(1.0, 0.0, fade).astype("float32")
        data[min_pos:min_pos + fade] *= np.linspace(0.0, 1.0, fade).astype("float32")
        data = np.concatenate([data[:min_pos], filler, data[min_pos:]])
        sf.write(str(wav_path), data, sr, subtype="PCM_16")
        if zero_fill:
            _zero_out_gap(wav_path)
        return _analyze_speech_boundaries(wav_path)["silence_gaps"]

    gap = gaps[0]
    current_ms = gap["duration_ms"]
    # RMS window quantization means measured gap can differ by up to RMS_WINDOW_MS
    # from actual. Accept gaps within one window of target to avoid endless re-adjust.
    if abs(current_ms - target_ms) <= RMS_WINDOW_MS:
        if zero_fill:
            _zero_out_gap(wav_path)
            return _analyze_speech_boundaries(wav_path)["silence_gaps"]
        return gaps

    gap_start_sample = int(sr * gap["start_ms"] / 1000)
    gap_end_sample = int(sr * gap["end_ms"] / 1000)

    if current_ms > target_ms:
        # trim: keep target_ms worth of silence from the start of the gap
        keep_samples = int(sr * target_ms / 1000)
        trim_start = gap_start_sample + keep_samples
        data = np.concatenate([data[:trim_start], data[gap_end_sample:]])
    else:
        # stretch: insert filler at the midpoint
        extra_ms = target_ms - current_ms
        extra_samples = int(sr * extra_ms / 1000)
        gap_mid_sample = int(sr * (gap["start_ms"] + current_ms / 2) / 1000)
        filler = _fill(extra_samples)
        data = np.concatenate([data[:gap_mid_sample], filler, data[gap_mid_sample:]])

    sf.write(str(wav_path), data, sr, subtype="PCM_16")
    if zero_fill:
        _zero_out_gap(wav_path)

    return _analyze_speech_boundaries(wav_path)["silence_gaps"]


def _extract_alignment_info(api_response: dict) -> dict:
    """Extract timing info from the /with-timestamps alignment data."""
    alignment = api_response.get("alignment") or {}
    chars = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])

    if not chars or not starts or not ends:
        return {}

    # find first and last non-space character timestamps
    first_char_s = None
    last_char_s = None
    for i, ch in enumerate(chars):
        if ch.strip() and i < len(starts):
            if first_char_s is None:
                first_char_s = starts[i]
            last_char_s = ends[i] if i < len(ends) else starts[i]

    return {
        "alignment_start_s": first_char_s,
        "alignment_end_s": last_char_s,
        "char_count": len(chars),
    }


def generate(
    api_key: str,
    force: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Generate 15 TTS turns (one per sentence, each with its assigned voice)."""
    turns_dir = OUT_DIR / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    turns_data = []

    for turn_num, sentence in enumerate(SENTENCES):
        speaker = sentence["voice"]
        voice_id = VOICES[speaker]
        category = sentence["category"]
        display_text = sentence["display_text"]
        expected_complete = sentence["expected_complete"]
        target_gap = sentence.get("target_gap_ms", 0)

        speaker_dir = turns_dir / f"speaker{speaker}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_path = speaker_dir / f"turn_{turn_num:03d}.wav"

        turn_entry = {
            "turn": turn_num,
            "speaker": speaker,
            "start_ms": 0,
            "end_ms": 0,
            "duration_ms": 0,
            "text": display_text,
            "word_count": len(display_text.split()),
            "hesitations": [],
            "max_hesitation_ms": 0,
            "category": category,
            "expected_complete": expected_complete,
            "target_gap_ms": target_gap,
            "voice_id": voice_id,
        }

        if wav_path.exists() and not force:
            # stretch gaps on existing files if needed (idempotent)
            if target_gap and category in ("hesitation", "hesitation2", "pause"):
                _set_gap_duration(wav_path, target_gap)

            dur = _wav_duration_ms(wav_path)
            turn_entry["duration_ms"] = dur
            turn_entry["end_ms"] = dur

            boundaries = _analyze_speech_boundaries(wav_path)
            turn_entry["speech_start_ms"] = boundaries["speech_start_ms"]
            turn_entry["speech_end_ms"] = boundaries["speech_end_ms"]
            turn_entry["rms_peak"] = boundaries["rms_peak"]
            if boundaries["silence_gaps"]:
                turn_entry["hesitations"] = [
                    {"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}
                    for g in boundaries["silence_gaps"]
                ]
                turn_entry["max_hesitation_ms"] = max(
                    g["duration_ms"] for g in boundaries["silence_gaps"]
                )

            label = "SKIP" if skip_existing else "EXISTS"
            gaps_info = ""
            if boundaries["silence_gaps"]:
                gaps_info = f" gaps={[g['duration_ms'] for g in boundaries['silence_gaps']]}ms"
            print(f"  [{turn_num:03d}] S{speaker} {category:10s} {label} {dur}ms "
                  f"(speech {boundaries['speech_start_ms']}-{boundaries['speech_end_ms']}ms"
                  f"{gaps_info})")
            turns_data.append(turn_entry)
            continue

        # synthesize via /with-timestamps
        print(f"  [{turn_num:03d}] S{speaker} {category:10s} generating...")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = Path(tmp.name)

        api_resp = _synthesize_with_timestamps(
            sentence["tts_text"], voice_id, api_key, tmp_mp3,
        )

        ok = False
        if api_resp is not None:
            ok = _mp3_to_wav(tmp_mp3, wav_path)

            if ok:
                # extract alignment metadata from API response
                align_info = _extract_alignment_info(api_resp)
                if align_info:
                    turn_entry["alignment_start_s"] = align_info["alignment_start_s"]
                    turn_entry["alignment_end_s"] = align_info["alignment_end_s"]

        tmp_mp3.unlink(missing_ok=True)

        if ok:
            # set silence gap to exact target duration
            if target_gap and category in ("hesitation", "hesitation2", "pause"):
                adjusted = _set_gap_duration(wav_path, target_gap)
                if adjusted:
                    print(f"         set gap to {target_gap}ms")

            # re-measure after any stretching
            dur = _wav_duration_ms(wav_path)
            turn_entry["duration_ms"] = dur
            turn_entry["end_ms"] = dur

            boundaries = _analyze_speech_boundaries(wav_path)
            turn_entry["speech_start_ms"] = boundaries["speech_start_ms"]
            turn_entry["speech_end_ms"] = boundaries["speech_end_ms"]
            turn_entry["rms_peak"] = boundaries["rms_peak"]
            if boundaries["silence_gaps"]:
                turn_entry["hesitations"] = [
                    {"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}
                    for g in boundaries["silence_gaps"]
                ]
                turn_entry["max_hesitation_ms"] = max(
                    g["duration_ms"] for g in boundaries["silence_gaps"]
                )

            gaps_str = ""
            if boundaries["silence_gaps"]:
                gap_durs = [g["duration_ms"] for g in boundaries["silence_gaps"]]
                gaps_str = f" gaps={gap_durs}ms"
                if target_gap and category in ("hesitation", "hesitation2", "pause"):
                    short = [d for d in gap_durs if d < target_gap]
                    if short:
                        gaps_str += f" WARNING: {short}ms < {target_gap}ms target"
            print(f"         -> {dur}ms "
                  f"(speech {boundaries['speech_start_ms']}-{boundaries['speech_end_ms']}ms"
                  f"{gaps_str})")
        else:
            print(f"         -> FAILED", file=sys.stderr)

        turns_data.append(turn_entry)
        time.sleep(RATE_LIMIT_SLEEP)

    # build index
    index = {
        "audio_file": "elevenlabs_tts",
        "provider": "elevenlabs",
        "model": MODEL_ID,
        "total_turns": len(turns_data),
        "voices": {str(k): v for k, v in VOICES.items()},
        "turns": turns_data,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "turns_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {index_path} ({len(turns_data)} turns)")

    # validation summary: check all gaps meet per-sentence targets
    print("\n── VAD threshold validation (target range 500-1500ms) ──")
    issues = []
    ok_count = 0
    for t in turns_data:
        cat = t["category"]
        if cat not in ("hesitation", "hesitation2", "pause"):
            continue
        target = t.get("target_gap_ms", 0)
        if not target:
            continue
        hes = t.get("hesitations", [])
        if not hes:
            issues.append(
                f"  turn {t['turn']:03d} ({cat}, target {target}ms): "
                f"NO silence gaps detected — TTS may not have produced a pause"
            )
        else:
            max_gap = max(h["duration_ms"] for h in hes)
            if max_gap < target:
                issues.append(
                    f"  turn {t['turn']:03d} ({cat}, target {target}ms): "
                    f"max gap {max_gap}ms < target"
                )
            else:
                ok_count += 1

    if issues:
        print(f"WARNING: {len(issues)} turns have gaps below target:")
        for issue in issues:
            print(issue)
    print(f"OK: {ok_count} hesitation/pause turns meet their targets")

    return index


HESITATION2_OUT_DIR = BASE_DIR / "out" / "TTS_Hesitation2"


def generate_hesitation2(
    api_key: str,
    force: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Generate hesitation2 TTS turns — prosody-only pauses, no filler words."""
    turns_dir = HESITATION2_OUT_DIR / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    turns_data = []

    for turn_num, sentence in enumerate(HESITATION2_SENTENCES):
        speaker = sentence["voice"]
        voice_id = VOICES[speaker]
        category = sentence["category"]
        display_text = sentence["display_text"]
        expected_complete = sentence["expected_complete"]
        target_gap = sentence.get("target_gap_ms", 0)

        speaker_dir = turns_dir / f"speaker{speaker}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_path = speaker_dir / f"turn_{turn_num:03d}.wav"

        turn_entry = {
            "turn": turn_num,
            "speaker": speaker,
            "start_ms": 0,
            "end_ms": 0,
            "duration_ms": 0,
            "text": display_text,
            "word_count": len(display_text.split()),
            "hesitations": [],
            "max_hesitation_ms": 0,
            "category": category,
            "expected_complete": expected_complete,
            "target_gap_ms": target_gap,
            "voice_id": voice_id,
        }

        if wav_path.exists() and not force:
            if target_gap:
                _set_gap_duration(wav_path, target_gap)

            dur = _wav_duration_ms(wav_path)
            turn_entry["duration_ms"] = dur
            turn_entry["end_ms"] = dur

            boundaries = _analyze_speech_boundaries(wav_path)
            turn_entry["speech_start_ms"] = boundaries["speech_start_ms"]
            turn_entry["speech_end_ms"] = boundaries["speech_end_ms"]
            turn_entry["rms_peak"] = boundaries["rms_peak"]
            if boundaries["silence_gaps"]:
                turn_entry["hesitations"] = [
                    {"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}
                    for g in boundaries["silence_gaps"]
                ]
                turn_entry["max_hesitation_ms"] = max(
                    g["duration_ms"] for g in boundaries["silence_gaps"]
                )

            label = "SKIP" if skip_existing else "EXISTS"
            gaps_info = ""
            if boundaries["silence_gaps"]:
                gaps_info = f" gaps={[g['duration_ms'] for g in boundaries['silence_gaps']]}ms"
            print(f"  [{turn_num:03d}] S{speaker} {category:12s} {label} {dur}ms "
                  f"(speech {boundaries['speech_start_ms']}-{boundaries['speech_end_ms']}ms"
                  f"{gaps_info})")
            turns_data.append(turn_entry)
            continue

        print(f"  [{turn_num:03d}] S{speaker} {category:12s} generating...")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = Path(tmp.name)

        api_resp = _synthesize_with_timestamps(
            sentence["tts_text"], voice_id, api_key, tmp_mp3,
        )

        ok = False
        if api_resp is not None:
            ok = _mp3_to_wav(tmp_mp3, wav_path)

            if ok:
                align_info = _extract_alignment_info(api_resp)
                if align_info:
                    turn_entry["alignment_start_s"] = align_info["alignment_start_s"]
                    turn_entry["alignment_end_s"] = align_info["alignment_end_s"]

        tmp_mp3.unlink(missing_ok=True)

        if ok:
            if target_gap:
                adjusted = _set_gap_duration(wav_path, target_gap)
                if adjusted:
                    print(f"         set gap to {target_gap}ms")

            dur = _wav_duration_ms(wav_path)
            turn_entry["duration_ms"] = dur
            turn_entry["end_ms"] = dur

            boundaries = _analyze_speech_boundaries(wav_path)
            turn_entry["speech_start_ms"] = boundaries["speech_start_ms"]
            turn_entry["speech_end_ms"] = boundaries["speech_end_ms"]
            turn_entry["rms_peak"] = boundaries["rms_peak"]
            if boundaries["silence_gaps"]:
                turn_entry["hesitations"] = [
                    {"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}
                    for g in boundaries["silence_gaps"]
                ]
                turn_entry["max_hesitation_ms"] = max(
                    g["duration_ms"] for g in boundaries["silence_gaps"]
                )

            gaps_str = ""
            if boundaries["silence_gaps"]:
                gap_durs = [g["duration_ms"] for g in boundaries["silence_gaps"]]
                gaps_str = f" gaps={gap_durs}ms"
            print(f"         -> {dur}ms "
                  f"(speech {boundaries['speech_start_ms']}-{boundaries['speech_end_ms']}ms"
                  f"{gaps_str})")
        else:
            print(f"         -> FAILED", file=sys.stderr)

        turns_data.append(turn_entry)
        time.sleep(RATE_LIMIT_SLEEP)

    # build index
    index = {
        "audio_file": "elevenlabs_tts_hesitation2",
        "provider": "elevenlabs",
        "model": MODEL_ID,
        "total_turns": len(turns_data),
        "voices": {str(k): v for k, v in VOICES.items()},
        "turns": turns_data,
    }

    HESITATION2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = HESITATION2_OUT_DIR / "turns_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {index_path} ({len(turns_data)} turns)")

    # validation
    print("\n── gap validation ──")
    issues = []
    ok_count = 0
    for t in turns_data:
        target = t.get("target_gap_ms", 0)
        if not target:
            continue
        hes = t.get("hesitations", [])
        if not hes:
            issues.append(
                f"  turn {t['turn']:03d} (target {target}ms): "
                f"NO silence gaps detected"
            )
        else:
            max_gap = max(h["duration_ms"] for h in hes)
            if max_gap < target:
                issues.append(
                    f"  turn {t['turn']:03d} (target {target}ms): "
                    f"max gap {max_gap}ms < target"
                )
            else:
                ok_count += 1

    if issues:
        print(f"WARNING: {len(issues)} turns have gaps below target:")
        for issue in issues:
            print(issue)
    print(f"OK: {ok_count}/{len(turns_data)} turns meet their targets")

    return index


BENCHMARK1_OUT_DIR = BASE_DIR / "out" / "Benchmark_1"


def generate_benchmark1(
    api_key: str,
    force: bool = False,
    skip_existing: bool = False,
    reprocess_existing: bool = True,
) -> dict:
    """Generate the Benchmark 1 corpus (21 turns, bit-perfect zero gaps).

    Gaps are sized to 800-1500ms per-turn targets, then bit-perfectly zeroed
    with a short edge fade so amplitude-driven EOT detectors see a true
    zero-amplitude silence window. Turn 0 is the opener and keeps its
    natural prosody (no target gap).

    When reprocess_existing=False, existing WAVs are NOT re-passed through
    _set_gap_duration (which would re-zero the gap and round its measured
    duration up to the RMS-quantized target). Use this when the keeper WAVs
    were copied verbatim from a pre-silenced corpus and the goal is to
    preserve their existing measured zero-run durations.
    """
    turns_dir = BENCHMARK1_OUT_DIR / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    turns_data = []

    for turn_num, sentence in enumerate(SENTENCES_BENCHMARK1):
        speaker = sentence["voice"]
        voice_id = VOICES[speaker]
        category = sentence["category"]
        display_text = sentence["display_text"]
        expected_complete = sentence["expected_complete"]
        target_gap = sentence.get("target_gap_ms", 0)

        speaker_dir = turns_dir / f"speaker{speaker}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_path = speaker_dir / f"turn_{turn_num:03d}.wav"

        turn_entry = {
            "turn": turn_num,
            "speaker": speaker,
            "start_ms": 0,
            "end_ms": 0,
            "duration_ms": 0,
            "text": display_text,
            "word_count": len(display_text.split()),
            "hesitations": [],
            "max_hesitation_ms": 0,
            "category": category,
            "expected_complete": expected_complete,
            "target_gap_ms": target_gap,
            "voice_id": voice_id,
        }

        if wav_path.exists() and not force:
            if (
                reprocess_existing
                and target_gap
                and category in ("hesitation", "hesitation2", "pause")
            ):
                _set_gap_duration(wav_path, target_gap, zero_fill=True)

            dur = _wav_duration_ms(wav_path)
            turn_entry["duration_ms"] = dur
            turn_entry["end_ms"] = dur

            boundaries = _analyze_speech_boundaries(wav_path)
            turn_entry["speech_start_ms"] = boundaries["speech_start_ms"]
            turn_entry["speech_end_ms"] = boundaries["speech_end_ms"]
            turn_entry["rms_peak"] = boundaries["rms_peak"]

            if category in ("hesitation", "hesitation2", "pause"):
                zr = _find_zero_run(wav_path, min_ms=200)
                if zr:
                    turn_entry["hesitations"] = [
                        {"at_ms": zr["start_ms"], "duration_ms": zr["duration_ms"]}
                    ]
                    turn_entry["max_hesitation_ms"] = zr["duration_ms"]
            elif boundaries["silence_gaps"]:
                turn_entry["hesitations"] = [
                    {"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}
                    for g in boundaries["silence_gaps"]
                ]
                turn_entry["max_hesitation_ms"] = max(
                    g["duration_ms"] for g in boundaries["silence_gaps"]
                )

            label = "SKIP" if skip_existing else "EXISTS"
            gaps_info = ""
            if turn_entry["hesitations"]:
                gaps_info = f" gaps={[h['duration_ms'] for h in turn_entry['hesitations']]}ms"
            print(f"  [{turn_num:03d}] S{speaker} {category:12s} {label} {dur}ms "
                  f"(speech {boundaries['speech_start_ms']}-{boundaries['speech_end_ms']}ms"
                  f"{gaps_info})")
            turns_data.append(turn_entry)
            continue

        print(f"  [{turn_num:03d}] S{speaker} {category:12s} generating...")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = Path(tmp.name)

        api_resp = _synthesize_with_timestamps(
            sentence["tts_text"], voice_id, api_key, tmp_mp3,
        )

        ok = False
        if api_resp is not None:
            ok = _mp3_to_wav(tmp_mp3, wav_path)
            if ok:
                align_info = _extract_alignment_info(api_resp)
                if align_info:
                    turn_entry["alignment_start_s"] = align_info["alignment_start_s"]
                    turn_entry["alignment_end_s"] = align_info["alignment_end_s"]

        tmp_mp3.unlink(missing_ok=True)

        if ok:
            if target_gap and category in ("hesitation", "hesitation2", "pause"):
                adjusted = _set_gap_duration(wav_path, target_gap, zero_fill=True)
                if adjusted:
                    print(f"         set gap to {target_gap}ms (zero-fill)")

            dur = _wav_duration_ms(wav_path)
            turn_entry["duration_ms"] = dur
            turn_entry["end_ms"] = dur

            boundaries = _analyze_speech_boundaries(wav_path)
            turn_entry["speech_start_ms"] = boundaries["speech_start_ms"]
            turn_entry["speech_end_ms"] = boundaries["speech_end_ms"]
            turn_entry["rms_peak"] = boundaries["rms_peak"]

            if category in ("hesitation", "hesitation2", "pause"):
                zr = _find_zero_run(wav_path, min_ms=200)
                if zr:
                    turn_entry["hesitations"] = [
                        {"at_ms": zr["start_ms"], "duration_ms": zr["duration_ms"]}
                    ]
                    turn_entry["max_hesitation_ms"] = zr["duration_ms"]
            elif boundaries["silence_gaps"]:
                turn_entry["hesitations"] = [
                    {"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}
                    for g in boundaries["silence_gaps"]
                ]
                turn_entry["max_hesitation_ms"] = max(
                    g["duration_ms"] for g in boundaries["silence_gaps"]
                )

            gaps_str = ""
            if turn_entry["hesitations"]:
                gap_durs = [h["duration_ms"] for h in turn_entry["hesitations"]]
                gaps_str = f" gaps={gap_durs}ms"
                if target_gap and category in ("hesitation", "hesitation2", "pause"):
                    short = [d for d in gap_durs if d < target_gap]
                    if short:
                        gaps_str += f" WARNING: {short}ms < {target_gap}ms target"
            print(f"         -> {dur}ms "
                  f"(speech {boundaries['speech_start_ms']}-{boundaries['speech_end_ms']}ms"
                  f"{gaps_str})")
        else:
            print(f"         -> FAILED", file=sys.stderr)

        turns_data.append(turn_entry)
        time.sleep(RATE_LIMIT_SLEEP)

    index = {
        "audio_file": "elevenlabs_tts_benchmark1",
        "provider": "elevenlabs",
        "model": MODEL_ID,
        "total_turns": len(turns_data),
        "voices": {str(k): v for k, v in VOICES.items()},
        "_note": (
            "Benchmark 1 corpus: 20 turns. Gaps in pause/hesitation/hesitation2 "
            "turns are bit-perfectly zeroed (with a 3 ms linear fade at edges) "
            "so amplitude-driven EOT detectors see a true silent window. "
            "Reported hesitations.duration_ms reflects the measured zero region."
        ),
        "turns": turns_data,
    }

    BENCHMARK1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = BENCHMARK1_OUT_DIR / "turns_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {index_path} ({len(turns_data)} turns)")

    print("\n── gap validation (target range 800-1500ms, zero-fill) ──")
    issues = []
    ok_count = 0
    for t in turns_data:
        cat = t["category"]
        if cat not in ("hesitation", "hesitation2", "pause"):
            continue
        target = t.get("target_gap_ms", 0)
        if not target:
            continue
        hes = t.get("hesitations", [])
        if not hes:
            issues.append(
                f"  turn {t['turn']:03d} ({cat}, target {target}ms): "
                f"NO silence gaps detected"
            )
        else:
            max_gap = max(h["duration_ms"] for h in hes)
            if max_gap < target - RMS_WINDOW_MS:
                issues.append(
                    f"  turn {t['turn']:03d} ({cat}, target {target}ms): "
                    f"max gap {max_gap}ms < target"
                )
            else:
                ok_count += 1

    if issues:
        print(f"WARNING: {len(issues)} turns have gaps below target:")
        for issue in issues:
            print(issue)
    print(f"OK: {ok_count} hesitation/pause turns meet their targets")

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS turn-accuracy test audio via ElevenLabs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate all WAVs even if they exist",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip existing WAVs without re-generating (for resuming)",
    )
    parser.add_argument(
        "--hesitation2", action="store_true",
        help="Generate hesitation2 turns (prosody-only, no fillers, larger gaps)",
    )
    parser.add_argument(
        "--benchmark1", action="store_true",
        help="Generate the Benchmark 1 corpus (21 turns, 800-1500ms zero-fill gaps)",
    )
    parser.add_argument(
        "--no-reprocess-existing", action="store_true",
        help="With --benchmark1 --skip-existing: do not re-zero existing keeper "
             "WAVs (useful when they were copied verbatim from a pre-silenced "
             "source and re-zeroing would round measured durations to the target)",
    )
    args = parser.parse_args()

    api_key = _load_tts_key()
    if not api_key:
        print("ERROR: TTS_KEY not found in .env or environment", file=sys.stderr)
        print("Set TTS_KEY=your_elevenlabs_api_key in .env", file=sys.stderr)
        sys.exit(1)

    if args.benchmark1:
        generate_benchmark1(
            api_key=api_key,
            force=args.force,
            skip_existing=args.skip_existing,
            reprocess_existing=not args.no_reprocess_existing,
        )
    elif args.hesitation2:
        generate_hesitation2(
            api_key=api_key,
            force=args.force,
            skip_existing=args.skip_existing,
        )
    else:
        generate(
            api_key=api_key,
            force=args.force,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
