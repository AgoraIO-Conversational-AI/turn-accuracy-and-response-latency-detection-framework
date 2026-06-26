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
import shutil
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
    4: "BZgkqPqms7Kj9ulSkVzn",
    5: "Nmd04QDxMhcTd5ocBsuE",  # LES profile voice
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
        "text": "I'm going to say some random things to see how you respond. Please keep your responses to under ten words.",
        "expected_complete": True,
    },
    # 1 — pause (billing amount; semantically incomplete after gap).
    #   ONE [pause] tag — two tags get scattered by ElevenLabs to two
    #   different sentence positions, and the cleanup pass used to wipe
    #   the words in between. Whatever silence ElevenLabs renders is
    #   bit-perfectly zeroed in place; no resize.
    {
        "voice": 0,
        "category": "pause",
        "text": "Yeah, I got a bill for [hesitation] [pause] six hundred and eighty dollars and I can't pay it all today.",
        "expected_complete": False,
    },
    # 2 — normal (name + DoB recited cleanly; no engineered gap).
    #   Was a hesitation; converted to normal since the corpus already
    #   has plenty of hesitations and benefits from a longer normal.
    {
        "voice": 1,
        "category": "normal",
        "text": "Michael Turner, born April fourteenth, nineteen eighty five.",
        "expected_complete": True,
    },
    # 3 — normal (short complete answer; LES voice — new speaker in the mix)
    {
        "voice": 5,
        "category": "normal",
        "text": "Payment plan.",
        "expected_complete": True,
    },
    # 4 — hesitation (semantically ambiguous after gap — "Ask them" is a
    #   plausible-sounding fragment in isolation). Audible filler "urr"
    #   replaces the engineered silence: [hesitation] tags lengthen the
    #   prosody of "them", then the speaker says "urr" rather than going
    #   silent. No target_gap_ms / no zero-fill — natural ElevenLabs
    #   audio with the filler is what the agent hears.
    {
        "voice": 3,
        "category": "hesitation",
        "text": "Ask them [hesitation] [hesitation] urr why insurance didn't cover it.",
        "expected_complete": False,
    },
    # 5 — normal (decision confirmation)
    {
        "voice": 4,
        "category": "normal",
        "text": "That's fine, just do the payment plan.",
        "expected_complete": True,
    },
    # 6 — ambiguous (one-word reply)
    {
        "voice": 0,
        "category": "ambiguous",
        "text": "Yes.",
        "expected_complete": True,
    },
    # 7 — hesitation (voice 2; short sentence to bias gap shorter)
    {
        "voice": 2,
        "category": "hesitation",
        "text": "Update my [pause] billing address please.",
        "expected_complete": False,
    },
    # 8 — ambiguous (kept from original turn 4)
    {
        "voice": 2,
        "category": "ambiguous",
        "text": "That's not really what I was looking for but.",
        "expected_complete": True,
    },
    # 9 — normal (kept from original turn 5)
    {
        "voice": 0,
        "category": "normal",
        "text": "I'd like to book a table for two at seven o'clock tonight please.",
        "expected_complete": True,
    },
    # 10 — pause (single [pause] tag; ElevenLabs decides duration)
    {
        "voice": 3,
        "category": "pause",
        "text": "My account number starts with [pause] eight six three.",
        "expected_complete": False,
    },
    # 11 — hesitation (plain [pause]; voice 1 + this sentence is most
    #   reliable when kept simple)
    {
        "voice": 1,
        "category": "hesitation",
        "text": "The last time I checked it was [pause] somewhere around forty five dollars.",
        "expected_complete": False,
    },
    # 12 — hesitation
    {
        "voice": 2,
        "category": "hesitation",
        "text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
        "expected_complete": False,
    },
    # 13 — hesitation (one [hesitation], aiming for shorter range)
    {
        "voice": 1,
        "category": "hesitation",
        "text": "We could also try [hesitation] [pause] yeah, the other location might work better.",
        "expected_complete": False,
    },
    # 14 — hesitation (was hesitation2; collapsed since we no longer
    #   inject silence to hit a precise target. Single [pause] kept.)
    {
        "voice": 3,
        "category": "hesitation",
        "text": "I was going to renew but then the [hesitation] [hesitation] [pause] price went up by almost double.",
        "expected_complete": False,
    },
    # 15 — pause (single [pause] tag)
    {
        "voice": 2,
        "category": "pause",
        "text": "Could you transfer me to [pause] the billing department please?",
        "expected_complete": False,
    },
    # 16 — pause (single [pause] tag)
    {
        "voice": 0,
        "category": "pause",
        "text": "I'm calling because [pause] I received the wrong item yesterday.",
        "expected_complete": False,
    },
    # 17 — hesitation
    {
        "voice": 3,
        "category": "hesitation",
        "text": "The appointment was for [hesitation] [hesitation] [pause] uh, I think it was three thirty.",
        "expected_complete": False,
    },
    # 18 — hesitation (voice 4 ignores [hesitation] tags entirely; only
    #   responds to a bare [pause] placed mid-sentence — at "subscription"
    #   it lands in range 900-1500ms)
    {
        "voice": 4,
        "category": "hesitation",
        "text": "The problem is my old subscription [hesitation] [hesitation] [pause] was cancelled without any notice.",
        "expected_complete": False,
    },
    # 19 — ambiguous (kept from original turn 24)
    {
        "voice": 1,
        "category": "ambiguous",
        "text": "I mean I'm not entirely sure about that, it's hard to say.",
        "expected_complete": True,
    },
    # 20 — normal (kept from original turn 9; closes the session)
    {
        "voice": 4,
        "category": "normal",
        "text": "Thanks for your help, I appreciate it.",
        "expected_complete": True,
    },
]

# ── Benchmark 2 Original ──────────────────────────────────────────────
# Same hesitation texts as Benchmark 1 but all rendered in voice 4
# (the new BZgkqPqms7Kj9ulSkVzn voice that honors [hesitation] tags).
# Target gap range is wider — 1000-2000 ms — for testing agents
# against longer mid-utterance pauses. Original-only: no Zeroed pair.
SENTENCES_BENCHMARK2 = [
    # 0 — normal opener: tells the agent under test what's about to
    #   happen. Same text as Benchmark 1's turn 0 but in voice 4 so the
    #   instruction and the hesitation utterances share a single
    #   speaker throughout.
    {"voice": 4, "category": "normal",
     "text": "I'm going to say some random things to see how you respond. Please keep your responses to under ten words.",
     "expected_complete": True},
    {"voice": 4, "category": "hesitation",
     "text": "Ask them [hesitation] [hesitation] urr why insurance didn't cover it.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "Update my [pause] billing address please.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "The last time I checked it was [pause] somewhere around forty five dollars.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "So what happened was the system [hesitation] [hesitation] [pause] flagged my account for some reason.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "We could also try [hesitation] [pause] yeah, the other location might work better.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "I was going to renew but then the [hesitation] [hesitation] [pause] price went up by almost double.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "The appointment was for [hesitation] [hesitation] [pause] uh, I think it was three thirty.",
     "expected_complete": False},
    {"voice": 4, "category": "hesitation",
     "text": "The problem is my old subscription [hesitation] [hesitation] [pause] [pause] was cancelled without any notice.",
     "expected_complete": False},
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


def _analyze_speech_boundaries(wav_path: Path, min_gap_ms: int = 450) -> dict:
    """Analyze WAV amplitude to find precise speech start/end and silence gaps.

    Returns dict with:
      - speech_start_ms: first moment RMS exceeds threshold
      - speech_end_ms: last moment RMS exceeds threshold
      - silence_gaps: list of {start_ms, end_ms, duration_ms} for internal gaps
      - rms_peak: peak RMS value across the file

    `min_gap_ms` controls which internal silences are reported. The default
    450ms suits engineered [pause] gaps; lower values (e.g. 150-200ms) catch
    natural ElevenLabs thinking pauses around prosody / filler words.
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

    # find internal silence gaps (>= min_gap_ms of silence between speech)
    min_gap_frames = int(min_gap_ms / RMS_WINDOW_MS)
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


def _zero_out_gap(
    wav_path: Path, fade_ms: int = 3, min_gap_ms: int = 450,
) -> dict | None:
    """Bit-perfectly zero the largest silence gap, with a short fade at edges.

    Reads the current gap, walks outward from its midpoint until each side
    hits the first non-silent sample, then writes integer zeros in between.
    A short linear fade just outside each edge avoids click artifacts.

    `min_gap_ms` is forwarded to _analyze_speech_boundaries; lower it (e.g.
    150-200ms) when the source audio's "thinking pause" is shorter than the
    450ms default, as happens with natural ElevenLabs [hesitation]+filler
    output.

    Returns the {start_ms, end_ms, duration_ms} of the zeroed region (sample-
    exact, not RMS-window-quantized), or None if no gap was found.
    """
    boundaries = _analyze_speech_boundaries(wav_path, min_gap_ms=min_gap_ms)
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


def _zero_specified_gap(
    wav_path: Path, start_ms: int, end_ms: int, fade_ms: int = 3,
) -> dict | None:
    """Zero a SPECIFIC [start_ms, end_ms] window in a WAV. No re-detection.

    Used by the paired Original/Zeroed Benchmark 1 pipeline — the test-gap
    window is detected once on the Original WAV, then this helper applies
    the same window to the Zeroed copy. Reuse-only; never re-detect.

    Returns the zeroed region descriptor (using the input start/end_ms,
    not a re-measurement), or None if the window was empty.
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    start = int(sr * start_ms / 1000)
    end = int(sr * end_ms / 1000)
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
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
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
    """Resize the LARGEST silence gap to target_ms.

    Does NOT call _enforce_single_gap — that helper merges multiple gaps
    by deleting all audio between them, which can wipe out entire phrases
    when ElevenLabs renders natural breath-pauses in addition to the
    engineered [pause]/[hesitation] gap (saw this on turn 1: "six hundred
    and eighty dollars" got chopped because there was a natural pause
    between "dollars" and "and"). Instead this function picks the longest
    detected silence region — the engineered gap, by construction — and
    trims/stretches just that one. Natural inter-word pauses are left
    alone.

    When zero_fill=True the resized gap is bit-perfectly zeroed with a
    short linear fade just outside each edge to avoid click artifacts.
    Used by the Benchmark 1 corpus so amplitude-driven EOT detectors see
    a true zero-amplitude window.
    Returns the final gap list.
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    boundaries = _analyze_speech_boundaries(wav_path)
    gaps = boundaries["silence_gaps"]
    if len(gaps) > 1:
        # Use the largest gap, not the first. The engineered [pause]/
        # [hesitation] silence is the largest by construction; smaller
        # gaps are natural inter-word breaths and must stay intact.
        gaps = [max(gaps, key=lambda g: g["duration_ms"])]

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


BENCHMARK1_ORIG_DIR = BASE_DIR / "out" / "Benchmark_1_Original"
BENCHMARK1_ZEROED_DIR = BASE_DIR / "out" / "Benchmark_1_Zeroed"

# Min gap length the detector reports for Benchmark 1 turns. Same value
# is used as the test-gap window in BOTH paired sources — never
# re-detected after zeroing.
BENCHMARK1_MIN_GAP_MS = 150


def _build_turn_entry(sentence: dict, turn_num: int) -> dict:
    """Empty turn-entry skeleton shared by both Original and Zeroed indexes."""
    text = sentence["text"]
    return {
        "turn": turn_num,
        "speaker": sentence["voice"],
        "start_ms": 0,
        "end_ms": 0,
        "duration_ms": 0,
        "text": text,
        "word_count": len(text.split()),
        "hesitations": [],
        "max_hesitation_ms": 0,
        "category": sentence["category"],
        "expected_complete": sentence["expected_complete"],
        "voice_id": VOICES[sentence["voice"]],
    }


def generate_benchmark1(
    api_key: str,
    force: bool = False,
    skip_existing: bool = False,
    reprocess_existing: bool = True,
) -> dict:
    """Generate the paired Benchmark 1 corpora.

    Produces two sources from the SAME generated audio:

    - out/Benchmark_1_Original/  — raw ElevenLabs WAVs (post-transcode,
      pre-zero). Natural low-amplitude content sits inside the gap window.
    - out/Benchmark_1_Zeroed/    — same WAVs with the gap window written
      to bit-perfect zero (3 ms linear edge fade). Same window position
      and duration as Original.

    Key invariant: the test-gap window is detected ONCE on the Original
    WAV via _analyze_speech_boundaries(min_gap_ms=BENCHMARK1_MIN_GAP_MS).
    The detector boundaries (always multiples of RMS_WINDOW_MS = 20 ms)
    are written into both indexes as hesitations.at_ms / duration_ms.
    Never re-detect after zeroing — the Zeroed source's zero-run length
    may differ slightly from the detector window, but the canonical paired
    test gap is the detector window. Zeroed entries also carry
    `zero_run_ms` as a diagnostic for the actual contiguous int16-zero
    run, which has no role in paired comparisons.

    Sources share `voice_id`, `voice` index, `category`, and `text`. The
    only field that differs between paired entries is the on-disk
    duration (a few ms shift can occur if the WAV files are re-encoded
    differently) and the optional `zero_run_ms` field on Zeroed.

    Turn 0 is the operator opener — no test-gap is computed for it
    regardless of detected silences (so the opener is bit-identical
    between Original and Zeroed and operator instructions land intact).
    """
    for d in (BENCHMARK1_ORIG_DIR, BENCHMARK1_ZEROED_DIR):
        (d / "turns").mkdir(parents=True, exist_ok=True)

    turns_orig = []
    turns_zeroed = []

    for turn_num, sentence in enumerate(SENTENCES_BENCHMARK1):
        speaker = sentence["voice"]
        voice_id = VOICES[speaker]
        category = sentence["category"]
        text = sentence["text"]

        for d in (BENCHMARK1_ORIG_DIR, BENCHMARK1_ZEROED_DIR):
            (d / "turns" / f"speaker{speaker}").mkdir(parents=True, exist_ok=True)

        orig_wav = BENCHMARK1_ORIG_DIR / "turns" / f"speaker{speaker}" / f"turn_{turn_num:03d}.wav"
        zeroed_wav = BENCHMARK1_ZEROED_DIR / "turns" / f"speaker{speaker}" / f"turn_{turn_num:03d}.wav"

        entry_orig = _build_turn_entry(sentence, turn_num)
        entry_zeroed = _build_turn_entry(sentence, turn_num)

        need_synth = force or not orig_wav.exists()

        if need_synth:
            print(f"  [{turn_num:03d}] S{speaker} {category:12s} generating (Original)...")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_mp3 = Path(tmp.name)
            api_resp = _synthesize_with_timestamps(text, voice_id, api_key, tmp_mp3)
            ok = False
            if api_resp is not None:
                ok = _mp3_to_wav(tmp_mp3, orig_wav)
                if ok:
                    # Original = whatever ElevenLabs returned, untouched.
                    # If ElevenLabs ends a clip mid-syllable, we preserve
                    # that. Do NOT pad or trim Original.
                    align_info = _extract_alignment_info(api_resp)
                    if align_info:
                        for entry in (entry_orig, entry_zeroed):
                            entry["alignment_start_s"] = align_info["alignment_start_s"]
                            entry["alignment_end_s"] = align_info["alignment_end_s"]
            tmp_mp3.unlink(missing_ok=True)
            if not ok:
                print(f"         FAILED", file=sys.stderr)
                turns_orig.append(entry_orig)
                turns_zeroed.append(entry_zeroed)
                continue
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            label = "SKIP" if skip_existing else "EXISTS"
            print(f"  [{turn_num:03d}] S{speaker} {category:12s} {label} (re-pair from Original)")

        # Always ensure Zeroed copy starts from a fresh Original.
        # Without this, repeated runs would re-zero an already-zeroed
        # WAV (harmless but confusing); with force=True any earlier
        # Zeroed edits are wiped.
        if force or not zeroed_wav.exists():
            shutil.copy2(orig_wav, zeroed_wav)

        # ── Detect once on Original ──
        # Turn 0 is the opener — keep operator instructions intact in
        # both paired sources. No test gap is computed even if the
        # detector finds one.
        test_gap = None
        if turn_num > 0 and category in ("hesitation", "pause"):
            boundaries_raw = _analyze_speech_boundaries(
                orig_wav, min_gap_ms=BENCHMARK1_MIN_GAP_MS
            )
            gaps = boundaries_raw["silence_gaps"]
            if gaps:
                largest = max(gaps, key=lambda g: g["duration_ms"])
                test_gap = {
                    "at_ms": largest["start_ms"],
                    "end_ms": largest["end_ms"],
                    "duration_ms": largest["duration_ms"],
                }

        # ── Apply the SAME window to Zeroed ──
        # Reuse-only: never re-detect. If the original detection found
        # no gap, Zeroed stays identical to Original for this turn.
        zeroed_descriptor = None
        if test_gap is not None and (need_synth or reprocess_existing):
            zeroed_descriptor = _zero_specified_gap(
                zeroed_wav,
                start_ms=test_gap["at_ms"],
                end_ms=test_gap["end_ms"],
                fade_ms=3,
            )

        # ── Populate paired index entries ──
        for entry, wav_path, want_zero_run_ms in (
            (entry_orig, orig_wav, False),
            (entry_zeroed, zeroed_wav, True),
        ):
            dur = _wav_duration_ms(wav_path)
            entry["duration_ms"] = dur
            entry["end_ms"] = dur
            b = _analyze_speech_boundaries(wav_path, min_gap_ms=BENCHMARK1_MIN_GAP_MS)
            entry["speech_start_ms"] = b["speech_start_ms"]
            entry["speech_end_ms"] = b["speech_end_ms"]
            entry["rms_peak"] = b["rms_peak"]
            if test_gap is not None:
                # CANONICAL: shared detector window. Same on both indexes.
                entry["hesitations"] = [{
                    "at_ms": test_gap["at_ms"],
                    "duration_ms": test_gap["duration_ms"],
                }]
                entry["max_hesitation_ms"] = test_gap["duration_ms"]
            if want_zero_run_ms and test_gap is not None:
                # DIAGNOSTIC ONLY: actual contiguous int16-zero run in the
                # zeroed WAV. Not the canonical paired duration.
                zr = _find_zero_run(wav_path, min_ms=100)
                if zr:
                    entry["zero_run_ms"] = zr["duration_ms"]

        gap_str = f"{test_gap['duration_ms']}ms at {test_gap['at_ms']}ms" if test_gap else "—"
        print(f"         test_gap = {gap_str}")
        turns_orig.append(entry_orig)
        turns_zeroed.append(entry_zeroed)

    # ── Write the two paired indexes ──
    pairs = (
        (BENCHMARK1_ORIG_DIR, turns_orig, "Benchmark 1 Original",
         "Raw ElevenLabs WAVs (post-transcode, pre-zero). Natural low-"
         "amplitude content sits inside the hesitations.at_ms / "
         "duration_ms window. Detector window is shared with "
         "Benchmark 1 Zeroed (paired test gap)."),
        (BENCHMARK1_ZEROED_DIR, turns_zeroed, "Benchmark 1 Zeroed",
         "Same WAVs as Benchmark 1 Original, with the hesitations.at_ms "
         "/ duration_ms window written to bit-perfect zero (3 ms linear "
         "edge fade). The canonical paired test gap is hesitations."
         "duration_ms (the detector window); zero_run_ms is the actual "
         "contiguous int16-zero run length and is diagnostic only."),
    )
    for out_dir, turns_data, label, note in pairs:
        index = {
            "audio_file": f"elevenlabs_tts_benchmark1_{out_dir.name.lower()}",
            "provider": "elevenlabs",
            "model": MODEL_ID,
            "total_turns": len(turns_data),
            "voices": {str(k): v for k, v in VOICES.items()},
            "_note": note,
            "_paired_with": (
                "Benchmark_1_Zeroed" if "Original" in label else "Benchmark_1_Original"
            ),
            "turns": turns_data,
        }
        idx_path = out_dir / "turns_index.json"
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"wrote {idx_path} ({len(turns_data)} turns)")

    # ── Pairing sanity check ──
    print("\n── pairing audit ──")
    mismatched = 0
    for o, z in zip(turns_orig, turns_zeroed):
        oh = o["hesitations"][0] if o["hesitations"] else None
        zh = z["hesitations"][0] if z["hesitations"] else None
        if oh != zh:
            print(f"  MISMATCH turn {o['turn']}: orig={oh} zeroed={zh}")
            mismatched += 1
    if mismatched == 0:
        n_paired = sum(1 for o in turns_orig if o["hesitations"])
        print(f"OK: all {n_paired} gap-bearing turns share identical hesitations entries")
    else:
        print(f"WARNING: {mismatched} turns disagree between Original and Zeroed", file=sys.stderr)

    return {"original": turns_orig, "zeroed": turns_zeroed}


BENCHMARK2_ORIG_DIR = BASE_DIR / "out" / "Benchmark_2_Original"
BENCHMARK2_TARGET_BAND = (1000, 2000)  # ms — design band for Benchmark 2 gaps
BENCHMARK2_MAX_ATTEMPTS = 7            # ElevenLabs is non-deterministic; re-roll up to N


def generate_benchmark2(api_key: str, force: bool = False) -> dict:
    """Generate Benchmark 2 Original — voice-4 hesitation corpus.

    Same hesitation TEXTS as Benchmark 1 but every turn rendered in
    voice 4 (BZgkqPqms7Kj9ulSkVzn — the voice that honors [hesitation]
    tags). Target gap band 1000-2000 ms (wider than B1's 500-1500).
    Re-rolls up to BENCHMARK2_MAX_ATTEMPTS per turn until the
    detector-defined gap window lands in band. Original-only — no
    paired Zeroed source.

    Each turn entry shares the same fields as Benchmark 1 Original:
    hesitations.at_ms / duration_ms is the detector window (multiple
    of 20 ms from the RMS scan).
    """
    turns_dir = BENCHMARK2_ORIG_DIR / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    min_ok, max_ok = BENCHMARK2_TARGET_BAND
    target_mid = (min_ok + max_ok) // 2

    turns_data = []
    for turn_num, sentence in enumerate(SENTENCES_BENCHMARK2):
        speaker = sentence["voice"]
        voice_id = VOICES[speaker]
        text = sentence["text"]
        speaker_dir = turns_dir / f"speaker{speaker}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_path = speaker_dir / f"turn_{turn_num:03d}.wav"

        entry = {
            "turn": turn_num, "speaker": speaker,
            "start_ms": 0, "end_ms": 0, "duration_ms": 0,
            "text": text, "word_count": len(text.split()),
            "hesitations": [], "max_hesitation_ms": 0,
            "category": sentence["category"],
            "expected_complete": sentence["expected_complete"],
            "voice_id": voice_id,
        }

        if wav_path.exists() and not force:
            # SKIP — re-measure for the index
            print(f"  [{turn_num:03d}] S{speaker} {sentence['category']:12s} SKIP")
            dur = _wav_duration_ms(wav_path)
            entry["duration_ms"] = dur; entry["end_ms"] = dur
            b = _analyze_speech_boundaries(wav_path, min_gap_ms=BENCHMARK1_MIN_GAP_MS)
            entry["speech_start_ms"] = b["speech_start_ms"]
            entry["speech_end_ms"] = b["speech_end_ms"]
            entry["rms_peak"] = b["rms_peak"]
            if b["silence_gaps"] and sentence["category"] == "hesitation":
                g = max(b["silence_gaps"], key=lambda x: x["duration_ms"])
                entry["hesitations"] = [{"at_ms": g["start_ms"], "duration_ms": g["duration_ms"]}]
                entry["max_hesitation_ms"] = g["duration_ms"]
            turns_data.append(entry)
            continue

        # Normal turns (opener, etc.) — single render, no re-roll
        if sentence["category"] != "hesitation":
            print(f"  [{turn_num:03d}] S{speaker} {sentence['category']:12s} generating...")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_mp3 = Path(tmp.name)
            api_resp = _synthesize_with_timestamps(text, voice_id, api_key, tmp_mp3)
            ok = False
            if api_resp is not None:
                ok = _mp3_to_wav(tmp_mp3, wav_path)
                if ok:
                    align_info = _extract_alignment_info(api_resp)
                    if align_info:
                        entry["alignment_start_s"] = align_info["alignment_start_s"]
                        entry["alignment_end_s"] = align_info["alignment_end_s"]
            tmp_mp3.unlink(missing_ok=True)
            if not ok:
                print(f"         FAILED", file=sys.stderr)
                turns_data.append(entry)
                continue
            dur = _wav_duration_ms(wav_path)
            entry["duration_ms"] = dur; entry["end_ms"] = dur
            bb = _analyze_speech_boundaries(wav_path, min_gap_ms=BENCHMARK1_MIN_GAP_MS)
            entry["speech_start_ms"] = bb["speech_start_ms"]
            entry["speech_end_ms"] = bb["speech_end_ms"]
            entry["rms_peak"] = bb["rms_peak"]
            # normals don't carry hesitations even if the detector spots
            # a long-enough quiet region
            print(f"         -> {dur}ms (speech {bb['speech_start_ms']}-{bb['speech_end_ms']}ms)")
            turns_data.append(entry)
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        print(f"  [{turn_num:03d}] S{speaker} hesitation  rolling (target {min_ok}-{max_ok}ms)...")
        best_dist = None
        best_tmp = None
        best_gap = None
        for attempt in range(1, BENCHMARK2_MAX_ATTEMPTS + 1):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_mp3 = Path(tmp.name)
            api_resp = _synthesize_with_timestamps(text, voice_id, api_key, tmp_mp3)
            if api_resp is None:
                tmp_mp3.unlink(missing_ok=True)
                print(f"         attempt {attempt}: API FAIL")
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                cand_path = Path(tmp_wav.name)
            ok = _mp3_to_wav(tmp_mp3, cand_path)
            tmp_mp3.unlink(missing_ok=True)
            if not ok:
                cand_path.unlink(missing_ok=True)
                continue

            b = _analyze_speech_boundaries(cand_path, min_gap_ms=BENCHMARK1_MIN_GAP_MS)
            gaps = b["silence_gaps"]
            if not gaps:
                print(f"         attempt {attempt}: no gap detected")
                cand_path.unlink(missing_ok=True)
                time.sleep(RATE_LIMIT_SLEEP)
                continue
            g = max(gaps, key=lambda x: x["duration_ms"])
            gms = g["duration_ms"]
            in_band = min_ok <= gms <= max_ok
            tag = "✓" if in_band else ("LOW" if gms < min_ok else "HIGH")
            print(f"         attempt {attempt}: gap = {gms}ms  {tag}")
            if in_band:
                dist = abs(gms - target_mid)
                if best_dist is None or dist < best_dist:
                    if best_tmp is not None:
                        best_tmp.unlink(missing_ok=True)
                    best_dist = dist; best_tmp = cand_path; best_gap = g
                    # Stop early on a very close shot
                    if dist <= 200:
                        break
                else:
                    cand_path.unlink(missing_ok=True)
            else:
                cand_path.unlink(missing_ok=True)
            time.sleep(RATE_LIMIT_SLEEP)

        if best_tmp is None:
            print(f"         GAVE UP — no in-band render after {BENCHMARK2_MAX_ATTEMPTS} attempts")
            turns_data.append(entry)
            continue

        # Promote the best candidate
        shutil.copy2(best_tmp, wav_path)
        best_tmp.unlink(missing_ok=True)
        dur = _wav_duration_ms(wav_path)
        entry["duration_ms"] = dur; entry["end_ms"] = dur
        bb = _analyze_speech_boundaries(wav_path, min_gap_ms=BENCHMARK1_MIN_GAP_MS)
        entry["speech_start_ms"] = bb["speech_start_ms"]
        entry["speech_end_ms"] = bb["speech_end_ms"]
        entry["rms_peak"] = bb["rms_peak"]
        entry["hesitations"] = [{"at_ms": best_gap["start_ms"], "duration_ms": best_gap["duration_ms"]}]
        entry["max_hesitation_ms"] = best_gap["duration_ms"]
        print(f"         picked: gap = {best_gap['duration_ms']}ms at {best_gap['start_ms']}ms")
        turns_data.append(entry)

    # Write index
    index = {
        "audio_file": "elevenlabs_tts_benchmark2_original",
        "provider": "elevenlabs",
        "model": MODEL_ID,
        "total_turns": len(turns_data),
        "voices": {"4": VOICES[4]},
        "_note": (
            "Benchmark 2 Original. 8 hesitation turns sharing the same texts "
            "as Benchmark 1 but all rendered in voice 4 (BZgkqPqms7Kj9ulSkVzn). "
            "Target gap band 1000-2000 ms via re-roll loop. Original-only; "
            "no Zeroed pair."
        ),
        "turns": turns_data,
    }
    idx_path = BENCHMARK2_ORIG_DIR / "turns_index.json"
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {idx_path} ({len(turns_data)} turns)")
    sils = sorted(t["hesitations"][0]["duration_ms"] for t in turns_data if t["hesitations"])
    print(f"gap distribution: {sils}")
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
    parser.add_argument(
        "--benchmark2", action="store_true",
        help="Generate the Benchmark 2 Original corpus (8 hesitation turns in "
             "voice 4, 1000-2000 ms target gap band, re-rolled per turn)",
    )
    args = parser.parse_args()

    api_key = _load_tts_key()
    if not api_key:
        print("ERROR: TTS_KEY not found in .env or environment", file=sys.stderr)
        print("Set TTS_KEY=your_elevenlabs_api_key in .env", file=sys.stderr)
        sys.exit(1)

    if args.benchmark2:
        generate_benchmark2(api_key=api_key, force=args.force)
    elif args.benchmark1:
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
