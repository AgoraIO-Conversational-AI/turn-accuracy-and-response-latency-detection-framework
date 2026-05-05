"""Amplitude-based speech detector for clean digital audio paths.

Detects speech by checking RMS amplitude against a threshold. Suitable for
TTS audio captured over virtual audio devices (no ambient noise).
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

VAD_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz


class VadEngine:
    """Speech detector using RMS amplitude threshold.

    Designed for clean digital audio (e.g. TTS over BlackHole) where any
    audio above a low amplitude floor is speech. No ML model needed.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        min_silence_ms: int = 300,
    ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms

        self._speech_active = False
        self._silence_samples = 0
        self._min_silence_samples = int(VAD_RATE * min_silence_ms / 1000)
        self._total_samples = 0
        self._buffer = np.array([], dtype=np.float32)

    def reset(self):
        """Reset state for a new detection session."""
        self._speech_active = False
        self._silence_samples = 0
        self._total_samples = 0
        self._buffer = np.array([], dtype=np.float32)

    def process_chunk(self, audio_16k: np.ndarray) -> list[dict]:
        """Feed audio chunk (16kHz float32) and return speech events.

        Returns list of dicts with keys:
            - type: 'speech_start' or 'speech_end'
            - sample: sample offset from session start
            - time_s: seconds from session start
        """
        self._buffer = np.concatenate([self._buffer, audio_16k])
        events = []

        while len(self._buffer) >= CHUNK_SAMPLES:
            frame = self._buffer[:CHUNK_SAMPLES]
            self._buffer = self._buffer[CHUNK_SAMPLES:]

            rms = np.sqrt(np.mean(frame ** 2))

            if rms >= self.threshold:
                self._silence_samples = 0
                if not self._speech_active:
                    self._speech_active = True
                    log.info(f"speech_start rms={rms:.4f} at {self._total_samples / VAD_RATE:.2f}s")
                    events.append({
                        "type": "speech_start",
                        "sample": self._total_samples,
                        "time_s": self._total_samples / VAD_RATE,
                    })
            else:
                if self._speech_active:
                    self._silence_samples += CHUNK_SAMPLES
                    if self._silence_samples >= self._min_silence_samples:
                        self._speech_active = False
                        events.append({
                            "type": "speech_end",
                            "sample": self._total_samples,
                            "time_s": self._total_samples / VAD_RATE,
                        })
                        self._silence_samples = 0

            self._total_samples += CHUNK_SAMPLES

        return events

    @property
    def is_speech_active(self) -> bool:
        return self._speech_active
