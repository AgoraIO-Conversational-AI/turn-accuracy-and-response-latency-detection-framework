"""Audio I/O engine using sounddevice for dual playback and agent capture."""

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
VAD_RATE = 16000
BLOCK_SIZE = 1024  # frames per callback


def list_devices():
    """Return all audio devices with index, name, and channel counts."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        result.append({
            "index": i,
            "name": d["name"],
            "max_input_channels": d["max_input_channels"],
            "max_output_channels": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
        })
    return result


def find_device(name_substring: str, kind: str = "output") -> int | None:
    """Find a device index by name substring. kind: 'input' or 'output'."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
        if name_substring.lower() in d["name"].lower() and d[ch_key] > 0:
            return i
    return None


def _auto_detect_devices() -> dict:
    """Try to find the expected devices automatically.

    For monitoring playback, use MacBook Pro Speakers directly — NOT the
    Multi-Output Device, which includes BlackHole 16ch and would cause
    the turn playback to be captured as agent audio by VAD.
    """
    return {
        "blackhole_2ch": find_device("BlackHole 2ch", "output"),
        "speakers": find_device("MacBook Pro Speakers", "output"),
        "blackhole_16ch": find_device("BlackHole 16ch", "input"),
    }


class AudioEngine:
    """Manages dual playback to BlackHole 2ch + speakers, and capture from BlackHole 16ch."""

    def __init__(
        self,
        blackhole_2ch_idx: int | None = None,
        speakers_idx: int | None = None,
        blackhole_16ch_idx: int | None = None,
    ):
        detected = _auto_detect_devices()
        self.bh2_idx = blackhole_2ch_idx or detected["blackhole_2ch"]
        self.spk_idx = speakers_idx or detected["speakers"]
        self.bh16_idx = blackhole_16ch_idx or detected["blackhole_16ch"]

        self._capture_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._capture_stream: sd.InputStream | None = None
        self._playing = False
        self._play_lock = threading.Lock()
        self._playback_end_time: float | None = None
        self._stop_playback = threading.Event()

    @property
    def devices(self) -> dict:
        return {
            "blackhole_2ch": self.bh2_idx,
            "speakers": self.spk_idx,
            "blackhole_16ch": self.bh16_idx,
        }

    def play_turn(self, wav_path: Path, on_done: callable = None):
        """Play a WAV file simultaneously to BlackHole 2ch and speakers.

        Uses explicit OutputStream per device to avoid sd.play() global
        state conflicts. Blocks until playback finishes.
        """
        import soundfile as sf

        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim == 1:
            data = data[:, np.newaxis]  # (samples, 1)

        if sr != SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(SAMPLE_RATE, sr)
            data = resample_poly(data, SAMPLE_RATE // g, sr // g, axis=0)
            sr = SAMPLE_RATE

        self._playing = True
        self._stop_playback.clear()

        def _play_to_device(device_idx, device_data):
            if device_idx is None:
                log.warning("play_to_device: device_idx is None, skipping")
                return
            try:
                dev_info = sd.query_devices(device_idx)
                channels = max(1, dev_info["max_output_channels"])
                log.info(f"play_to_device: device={device_idx} ({dev_info['name']}), channels={channels}, samples={len(device_data)}")
                if device_data.shape[1] < channels:
                    device_data = np.tile(device_data, (1, channels))[:, :channels]

                done_event = threading.Event()
                stop_ref = self._stop_playback
                pos = [0]

                def callback(outdata, frames, time_info, status):
                    if stop_ref.is_set():
                        outdata[:] = 0
                        done_event.set()
                        raise sd.CallbackStop
                    end = pos[0] + frames
                    if end <= len(device_data):
                        outdata[:] = device_data[pos[0]:end]
                    else:
                        remaining = len(device_data) - pos[0]
                        if remaining > 0:
                            outdata[:remaining] = device_data[pos[0]:]
                        outdata[remaining:] = 0
                        done_event.set()
                        raise sd.CallbackStop
                    pos[0] = end

                stream = sd.OutputStream(
                    samplerate=sr,
                    device=device_idx,
                    channels=channels,
                    blocksize=BLOCK_SIZE,
                    dtype="float32",
                    callback=callback,
                )
                with stream:
                    # poll so stop_playback can interrupt
                    while not done_event.is_set():
                        if stop_ref.is_set():
                            break
                        done_event.wait(timeout=0.05)
            except Exception as e:
                log.error(f"playback error on device {device_idx}: {e}")

        t_bh = threading.Thread(
            target=_play_to_device, args=(self.bh2_idx, data.copy()), daemon=True,
        )
        t_spk = threading.Thread(
            target=_play_to_device, args=(self.spk_idx, data.copy()), daemon=True,
        )
        t_bh.start()
        t_spk.start()
        t_bh.join()
        t_spk.join()

        self._playback_end_time = time.perf_counter()
        self._playing = False
        if on_done:
            on_done()

    def stop_playback(self):
        """Signal playback threads to stop immediately."""
        self._stop_playback.set()

    @property
    def playback_end_time(self) -> float | None:
        return self._playback_end_time

    @property
    def is_playing(self) -> bool:
        return self._playing

    def start_capture(self):
        """Start capturing audio from BlackHole 16ch for VAD processing."""
        if self.bh16_idx is None:
            raise RuntimeError("BlackHole 16ch device not found")

        # stop any existing capture first
        self.stop_capture()

        # drain any old data
        while not self._capture_queue.empty():
            try:
                self._capture_queue.get_nowait()
            except queue.Empty:
                break

        dev_info = sd.query_devices(self.bh16_idx)
        channels = min(2, dev_info["max_input_channels"])
        self._capture_chunk_count = 0

        def _capture_callback(indata, frames, time_info, status):
            if status:
                pass  # occasional xruns are normal
            mono = indata[:, 0].copy()
            self._capture_queue.put(mono)
            self._capture_chunk_count += 1
            # log RMS every ~1 second (48000/1024 ≈ 47 chunks/s)
            if self._capture_chunk_count % 47 == 0:
                rms = np.sqrt(np.mean(mono ** 2))
                log.info(f"capture rms={rms:.6f} qsize={self._capture_queue.qsize()}")

        self._capture_stream = sd.InputStream(
            device=self.bh16_idx,
            samplerate=SAMPLE_RATE,
            channels=channels,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=_capture_callback,
        )
        self._capture_stream.start()

    def stop_capture(self):
        """Stop the capture stream."""
        if self._capture_stream is not None:
            self._capture_stream.stop()
            self._capture_stream.close()
            self._capture_stream = None

    def get_capture_chunk_16k(self, timeout: float = 0.05) -> np.ndarray | None:
        """Get the next capture chunk, resampled to 16kHz for VAD.

        Returns None if no data available within timeout.
        """
        try:
            chunk_48k = self._capture_queue.get(timeout=timeout)
        except queue.Empty:
            return None

        # 48kHz → 16kHz is a clean 3:1 decimation
        return chunk_48k[::3]

    def drain_capture_queue(self):
        """Discard all pending capture data."""
        while not self._capture_queue.empty():
            try:
                self._capture_queue.get_nowait()
            except queue.Empty:
                break
