# L2 — Audio Routing

## Device Architecture

The harness uses three audio devices simultaneously:

| Device | Direction | Purpose | sounddevice Type |
|--------|-----------|---------|-----------------|
| BlackHole 2ch | Output | Send turn audio to browser mic | OutputStream |
| MacBook Pro Speakers | Output | Monitor turn playback | OutputStream |
| BlackHole 16ch | Input | Capture agent responses | InputStream |

## Why Not Multi-Output for Monitoring?

The Multi-Output Device combines BlackHole 16ch + Speakers. If we play to Multi-Output, the turn audio appears on BlackHole 16ch and gets captured by VAD — causing immediate false barge-in detection on every turn.

Solution: play directly to "MacBook Pro Speakers" for monitoring, bypassing the Multi-Output entirely.

## Dual Playback Implementation

`AudioEngine.play_turn()` spawns two threads, one per output device:

```python
t_bh = Thread(target=_play_to_device, args=(self.bh2_idx, data.copy()))
t_spk = Thread(target=_play_to_device, args=(self.spk_idx, data.copy()))
t_bh.start(); t_spk.start()
t_bh.join(); t_spk.join()
```

Each thread creates its own `sd.OutputStream` with a callback that feeds audio frames. This avoids the global `sd.play()` state which can only target one device.

## Capture Pipeline

```
BlackHole 16ch → InputStream callback (48kHz, 1024 frames)
    → queue.Queue (thread-safe)
        → get_capture_chunk_16k() (decimate 48→16kHz via 3:1 downsampling)
            → VadEngine.process_chunk()
```

The capture callback runs at ~47 chunks/second (48000 / 1024). Each chunk is 21ms of audio.

## Channel Handling

- BlackHole 16ch has 16 channels but only channel 0 carries the system audio
- Capture callback takes `indata[:, 0]` for mono
- Playback tiles mono data to match device channel count: `np.tile(data, (1, channels))`

## Device Auto-Detection

```python
def _auto_detect_devices():
    return {
        "blackhole_2ch": find_device("BlackHole 2ch", "output"),
        "speakers": find_device("MacBook Pro Speakers", "output"),
        "blackhole_16ch": find_device("BlackHole 16ch", "input"),
    }
```

Case-insensitive substring match. Falls back to None if not found (logged as warning).

## Reconfiguration

Devices can be changed at runtime via:
- REST: `POST /api/devices/configure {"blackhole_2ch": 5, "speakers": 7}`
- Or by setting `audio_engine.bh2_idx`, `.spk_idx`, `.bh16_idx` directly
