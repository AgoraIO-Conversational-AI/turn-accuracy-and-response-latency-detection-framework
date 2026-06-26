"""Probe a tts_text + voice against ElevenLabs and report the silence length
after our standard post-processing. No mutation of source / on-disk WAVs."""
import argparse, base64, json, os, sys, tempfile, urllib.request
from pathlib import Path
import soundfile as sf, numpy as np

VOICES = {
    0: "XnKbmWxx8uWjruHkpXmf",
    1: "maYJAY8nOIBZeB0UYfc5",
    2: "QoC8og5VjCQoTz0caaaO",
    3: "VUGQSU6BSEjkbudnJbOj",
    4: "1AKkSX7KMPHIWuz76m0n",
}

def render(text, voice_id, key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
    payload = json.dumps({
        "text": text, "model_id": "eleven_v3",
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.8, "speed": 0.95},
    }).encode()
    req = urllib.request.Request(url, data=payload,
        headers={"xi-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    mp3.write(base64.b64decode(data["audio_base64"])); mp3.close()
    wav = mp3.name.replace(".mp3", ".wav")
    os.system(f"ffmpeg -y -i {mp3.name} -ac 1 -ar 48000 -sample_fmt s16 -c:a pcm_s16le {wav} 2>/dev/null")
    return wav

def analyse_and_zero(wav, min_gap_ms=150, rms_threshold=0.005):
    """Mirror the natural-gap branch of generate_benchmark1: find the largest
    silence with RMS<threshold, zero it bit-perfectly, return its length."""
    d, sr = sf.read(wav, dtype="float32")
    if d.ndim > 1: d = d[:, 0]
    win = int(sr * 0.02)
    rms = [float(np.sqrt(np.mean(d[i:i+win]**2))) for i in range(0, len(d)-win, win)]
    is_speech = [r >= rms_threshold for r in rms]
    # find runs of NOT-speech inside the speech window
    start = next((i for i,s in enumerate(is_speech) if s), 0)
    end = next((i for i in range(len(is_speech)-1,-1,-1) if is_speech[i]), len(is_speech)-1) + 1
    gaps, in_g, g_s = [], False, 0
    min_frames = int(min_gap_ms / 20)
    for i in range(start, min(end, len(is_speech))):
        if not is_speech[i]:
            if not in_g: in_g = True; g_s = i
        else:
            if in_g:
                if i - g_s >= min_frames:
                    gaps.append((g_s, i, i - g_s))
                in_g = False
    if not gaps:
        # find longest sub-threshold quiet region (above min_gap_ms) regardless
        return {"silence_ms": 0, "wav_ms": int(len(d)*1000/sr), "wav": wav}
    # largest gap
    s_f, e_f, n_f = max(gaps, key=lambda g: g[2])
    s_ms, e_ms = s_f * 20, e_f * 20
    # zero out
    d[int(sr * s_ms/1000):int(sr * e_ms/1000)] = 0.0
    sf.write(wav, d, sr, subtype="PCM_16")
    # remeasure as actual zero-run
    d2, _ = sf.read(wav, dtype="int16")
    if d2.ndim > 1: d2 = d2[:, 0]
    z = (d2 == 0).astype(np.int8)
    edges = np.diff(np.concatenate([[0], z, [0]]))
    starts = np.flatnonzero(edges == 1); ends = np.flatnonzero(edges == -1)
    longest = max((e_-s for s,e_ in zip(starts, ends)), default=0)
    return {"silence_ms": int(round(longest*1000/sr)), "wav_ms": int(len(d)*1000/sr), "wav": wav}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--voice", type=int, required=True)
    p.add_argument("--n", type=int, default=1, help="number of renders")
    args = p.parse_args()
    key = next(l.strip().split("=",1)[1] for l in open(".env") if l.startswith("TTS_KEY="))
    voice_id = VOICES[args.voice]
    print(f"VOICE {args.voice}  N={args.n}")
    print(f"text: {args.text!r}")
    samples = []
    for i in range(args.n):
        wav = render(args.text, voice_id, key)
        r = analyse_and_zero(wav)
        samples.append(r["silence_ms"])
        print(f"  render {i+1}: silence={r['silence_ms']:>4}ms  wav={r['wav_ms']:>5}ms  {r['wav']}")
    if args.n > 1:
        print(f"\nrange: {min(samples)}–{max(samples)}ms  mean={sum(samples)//len(samples)}ms")
