#!/usr/bin/env python3
"""
Speaker diarization comparison — Deepgram vs Soniox vs Speechmatics.
Analyzes who speaks when and splits into per-speaker files.
All providers use batch/async mode for best diarization accuracy.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

# ─── Load .env ───────────────────────────────────────────────────────────

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


# ─── Deepgram ────────────────────────────────────────────────────────────

def diarize_deepgram(audio_path, api_key):
    import urllib.request

    print("[Deepgram] Uploading and transcribing...")
    t0 = time.time()

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    ext = Path(audio_path).suffix.lower()
    mime = {".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav",
            ".ogg": "audio/ogg", ".flac": "audio/flac"}.get(ext, "audio/mp4")

    url = ("https://api.deepgram.com/v1/listen?"
           "model=nova-3&language=en&punctuate=true&smart_format=true"
           "&diarize=true&utterances=true")

    req = urllib.request.Request(url, data=audio_data, headers={
        "Authorization": f"Token {api_key}",
        "Content-Type": mime,
    })

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    elapsed = time.time() - t0
    print(f"[Deepgram] Done in {elapsed:.1f}s")

    words = []
    for w in result["results"]["channels"][0]["alternatives"][0]["words"]:
        words.append({
            "word": w.get("punctuated_word", w["word"]),
            "start": w["start"],
            "end": w["end"],
            "speaker": w.get("speaker", 0),
            "confidence": w.get("speaker_confidence", w.get("confidence", 0)),
        })

    return words, result


# ─── Soniox ──────────────────────────────────────────────────────────────

def diarize_soniox(audio_path, api_key):
    import requests as req_lib

    print("[Soniox] Uploading file...")
    t0 = time.time()
    session = req_lib.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"

    # Upload file
    with open(audio_path, "rb") as f:
        res = session.post("https://api.soniox.com/v1/files",
                           files={"file": f})
    res.raise_for_status()
    file_id = res.json()["id"]
    print(f"[Soniox] File uploaded: {file_id}")

    # Create transcription
    config = {
        "model": "stt-async-v4",
        "language_hints": ["en"],
        "enable_speaker_diarization": True,
        "file_id": file_id,
    }
    res = session.post("https://api.soniox.com/v1/transcriptions", json=config)
    res.raise_for_status()
    txn_id = res.json()["id"]
    print(f"[Soniox] Transcription created: {txn_id}")

    # Poll for completion
    print("[Soniox] Waiting for completion...")
    while True:
        res = session.get(f"https://api.soniox.com/v1/transcriptions/{txn_id}")
        res.raise_for_status()
        status = res.json()["status"]
        if status == "completed":
            break
        if status == "failed":
            print(f"[Soniox] FAILED: {res.json()}")
            return [], res.json()
        time.sleep(1)

    # Get transcript
    res = session.get(f"https://api.soniox.com/v1/transcriptions/{txn_id}/transcript")
    res.raise_for_status()
    result = res.json()

    elapsed = time.time() - t0
    print(f"[Soniox] Done in {elapsed:.1f}s")

    # Parse tokens — Soniox returns sub-word tokens, merge into words.
    # A new word starts when a token has a leading space or speaker changes.
    raw_tokens = result.get("tokens", [])
    words = []
    current_word = ""
    word_start = None
    word_end = None
    word_speaker = None

    for token in raw_tokens:
        text = token.get("text", "")
        if not text:
            continue
        spk = int(token.get("speaker", "1")) - 1  # normalize "1"/"2" → 0/1
        t_start = token.get("start_ms", 0) / 1000.0
        t_end = token.get("end_ms", t_start * 1000) / 1000.0

        # Detect word boundary: leading space or speaker change
        if text.startswith(" ") or word_speaker is None or spk != word_speaker:
            if current_word.strip():
                words.append({
                    "word": current_word.strip(),
                    "start": word_start,
                    "end": word_end,
                    "speaker": word_speaker,
                    "confidence": 0,
                })
            current_word = text
            word_start = t_start
            word_end = t_end
            word_speaker = spk
        else:
            current_word += text
            word_end = t_end

    if current_word.strip():
        words.append({
            "word": current_word.strip(),
            "start": word_start,
            "end": word_end,
            "speaker": word_speaker,
            "confidence": 0,
        })

    return words, result


# ─── Speechmatics ────────────────────────────────────────────────────────

def diarize_speechmatics(audio_path, api_key):
    import requests as req_lib

    print("[Speechmatics] Submitting batch job...")
    t0 = time.time()

    config = json.dumps({
        "type": "transcription",
        "transcription_config": {
            "language": "en",
            "diarization": "speaker",
        }
    })

    with open(audio_path, "rb") as f:
        res = req_lib.post(
            "https://asr.api.speechmatics.com/v2/jobs/",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "data_file": (Path(audio_path).name, f),
                "config": (None, config, "application/json"),
            },
        )
    res.raise_for_status()
    job_id = res.json()["id"]
    print(f"[Speechmatics] Job created: {job_id}")

    # Poll for completion
    print("[Speechmatics] Waiting for completion...")
    while True:
        res = req_lib.get(
            f"https://asr.api.speechmatics.com/v2/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        res.raise_for_status()
        status = res.json()["job"]["status"]
        if status == "done":
            break
        if status in ("rejected", "deleted"):
            print(f"[Speechmatics] FAILED: {res.json()}")
            return [], res.json()
        time.sleep(2)

    # Get transcript
    res = req_lib.get(
        f"https://asr.api.speechmatics.com/v2/jobs/{job_id}/transcript",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"format": "json-v2"},
    )
    res.raise_for_status()
    result = res.json()

    elapsed = time.time() - t0
    print(f"[Speechmatics] Done in {elapsed:.1f}s")

    # Parse results into word list
    words = []
    for item in result.get("results", []):
        if item.get("type") != "word":
            continue
        alt = item["alternatives"][0] if item.get("alternatives") else {}
        speaker_label = alt.get("speaker", "S1")
        # Convert S1/S2 to 0/1
        try:
            speaker = int(speaker_label.replace("S", "")) - 1
        except (ValueError, AttributeError):
            speaker = 0
        words.append({
            "word": alt.get("content", ""),
            "start": item.get("start_time", 0),
            "end": item.get("end_time", 0),
            "speaker": speaker,
            "confidence": alt.get("confidence", 0),
        })

    return words, result


# ─── Analysis ────────────────────────────────────────────────────────────

def build_segments(words):
    """Build speaker segments from word-level data."""
    if not words:
        return []

    segments = []
    current_speaker = words[0]["speaker"]
    seg_start = words[0]["start"]
    seg_words = []

    for w in words:
        if w["speaker"] != current_speaker:
            segments.append({
                "speaker": current_speaker,
                "start": seg_start,
                "end": seg_words[-1]["end"],
                "text": " ".join(x["word"] for x in seg_words),
                "word_count": len(seg_words),
            })
            current_speaker = w["speaker"]
            seg_start = w["start"]
            seg_words = []
        seg_words.append(w)

    if seg_words:
        segments.append({
            "speaker": current_speaker,
            "start": seg_start,
            "end": seg_words[-1]["end"],
            "text": " ".join(x["word"] for x in seg_words),
            "word_count": len(seg_words),
        })

    return segments


def print_timeline(provider, segments):
    speakers = sorted(set(s["speaker"] for s in segments))
    total_dur = segments[-1]["end"] if segments else 0

    print(f"\n{'=' * 90}")
    print(f"  {provider.upper()} — SPEAKER DIARIZATION")
    print(f"  Duration: {total_dur:.1f}s | Speakers: {len(speakers)} | Segments: {len(segments)}")
    print(f"{'=' * 90}\n")

    for spk in speakers:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        total_time = sum(s["end"] - s["start"] for s in spk_segs)
        total_words = sum(s["word_count"] for s in spk_segs)
        print(f"  Speaker {spk}: {len(spk_segs)} turns, "
              f"{total_time:.1f}s ({total_time/total_dur*100:.0f}%), "
              f"{total_words} words")
    print()

    print(f"  {'Time':>8}  {'Dur':>5}  {'Speaker':>10}  Text")
    print(f"  {'-' * 80}")

    for seg in segments:
        dur = seg["end"] - seg["start"]
        t = seg["start"]
        text = seg["text"][:65] + ("..." if len(seg["text"]) > 65 else "")
        print(f"  {int(t//60)}:{t%60:05.2f}  {dur:4.1f}s  Speaker {seg['speaker']:>1}  {text}")

    print()


def compare_providers(all_results):
    """Compare diarization results across providers."""
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON — AGREEMENT & DISAGREEMENT")
    print(f"{'=' * 90}\n")

    providers = list(all_results.keys())

    max_time = max(
        seg[-1]["end"]
        for segs in all_results.values()
        for seg in [segs] if segs
    )

    bin_size = 0.5
    n_bins = int(max_time / bin_size) + 1

    def speaker_at(segments, t):
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                return seg["speaker"]
        return None

    # First pass: build speaker labels per bin per provider
    bins = []
    for i in range(n_bins):
        t = i * bin_size
        row = {}
        for prov, segs in all_results.items():
            s = speaker_at(segs, t)
            if s is not None:
                row[prov] = s
        bins.append((t, row))

    # Normalize labels: use first provider as reference, find best mapping
    # for each other provider by counting co-occurrences
    ref_prov = providers[0]
    label_map = {ref_prov: {0: 0, 1: 1}}  # identity for reference

    for prov in providers[1:]:
        # Count how often prov's label X co-occurs with ref's label Y
        cooccur = {}
        for t, row in bins:
            if ref_prov in row and prov in row:
                r = row[ref_prov]
                p = row[prov]
                cooccur[(p, r)] = cooccur.get((p, r), 0) + 1

        # Find mapping that maximizes agreement
        mapping = {}
        for p_label in set(k[0] for k in cooccur):
            best_r = max(
                (r for (p, r) in cooccur if p == p_label),
                key=lambda r: cooccur.get((p_label, r), 0)
            )
            mapping[p_label] = best_r
        label_map[prov] = mapping

    # Second pass: compare using normalized labels
    agree = 0
    disagree = 0
    disagree_times = []

    for t, row in bins:
        if len(row) < 2:
            continue

        normalized = {}
        for prov in providers:
            if prov in row:
                raw = row[prov]
                normalized[prov] = label_map[prov].get(raw, raw)

        vals = list(normalized.values())
        if len(set(vals)) <= 1:
            agree += 1
        else:
            disagree += 1
            disagree_times.append((t, normalized))

    total = agree + disagree
    if total > 0:
        print(f"  Agreement rate: {agree}/{total} bins ({agree/total*100:.0f}%)")
        print(f"  Disagreement: {disagree}/{total} bins ({disagree/total*100:.0f}%)")

    if disagree_times:
        print(f"\n  Disagreement points:")
        print(f"  {'Time':>8}", end="")
        for p in providers:
            print(f"  {p:>12}", end="")
        print()
        print(f"  {'-' * (8 + 14 * len(providers))}")

        for t, speakers in disagree_times:
            print(f"  {int(t//60)}:{t%60:05.2f}", end="")
            for p in providers:
                s = speakers.get(p, "-")
                print(f"  {'Spk ' + str(s) if s is not None else '    -':>12}", end="")
            print()

    # Per-provider stats (using normalized labels)
    print(f"\n  {'Provider':<15} {'Segments':>8} {'Spk 0 time':>10} {'Spk 1 time':>10} {'Spk 0 words':>11} {'Spk 1 words':>11}")
    print(f"  {'-' * 66}")
    for prov, segs in all_results.items():
        m = label_map[prov]
        s0_time = sum(s["end"] - s["start"] for s in segs if m.get(s["speaker"], s["speaker"]) == 0)
        s1_time = sum(s["end"] - s["start"] for s in segs if m.get(s["speaker"], s["speaker"]) == 1)
        s0_words = sum(s["word_count"] for s in segs if m.get(s["speaker"], s["speaker"]) == 0)
        s1_words = sum(s["word_count"] for s in segs if m.get(s["speaker"], s["speaker"]) == 1)
        print(f"  {prov:<15} {len(segs):>8} {s0_time:>9.1f}s {s1_time:>9.1f}s {s0_words:>11} {s1_words:>11}")

    print()


def split_audio(audio_path, segments, output_dir, provider):
    """Split audio into per-speaker files."""
    speakers = sorted(set(s["speaker"] for s in segments))
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(audio_path).stem

    for spk in speakers:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        if not spk_segs:
            continue

        seg_files = []
        for i, seg in enumerate(spk_segs):
            seg_file = os.path.join(output_dir, f"_tmp_{provider}_spk{spk}_seg{i}.wav")
            start = max(0, seg["start"] - 0.05)
            duration = (seg["end"] - seg["start"]) + 0.1
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(duration),
                "-ar", "16000", "-ac", "1", seg_file
            ], capture_output=True)
            seg_files.append(seg_file)

        concat_list = os.path.join(output_dir, f"_concat_{provider}_spk{spk}.txt")
        with open(concat_list, "w") as f:
            for sf in seg_files:
                f.write(f"file '{sf}'\n")

        out_file = os.path.join(output_dir, f"{stem}_{provider}_speaker{spk}.wav")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy", out_file
        ], capture_output=True)

        total_dur = sum(s["end"] - s["start"] for s in spk_segs)
        print(f"  {provider} Speaker {spk}: {out_file} ({total_dur:.1f}s, {len(spk_segs)} segments)")

        for sf in seg_files:
            os.unlink(sf)
        os.unlink(concat_list)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Speaker diarization comparison")
    parser.add_argument("--audio", required=True, help="Audio file to diarize")
    parser.add_argument("--providers", default="deepgram,soniox,speechmatics",
                        help="Comma-separated providers to test")
    parser.add_argument("--split", action="store_true",
                        help="Split audio into per-speaker files")
    parser.add_argument("--output-dir", default="diarize_output",
                        help="Output directory for results")
    parser.add_argument("--deepgram-key", default=os.environ.get("DEEPGRAM_API_KEY", ""))
    parser.add_argument("--soniox-key", default=os.environ.get("SONIOX_API_KEY", ""))
    parser.add_argument("--speechmatics-key", default=os.environ.get("SPEECHMATICS_API_KEY", ""))
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"File not found: {args.audio}")
        sys.exit(1)

    providers = [p.strip() for p in args.providers.split(",")]
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_segments = {}

    for provider in providers:
        print(f"\n{'─' * 60}")
        try:
            if provider == "deepgram":
                if not args.deepgram_key:
                    print(f"[{provider}] No API key — skipping")
                    continue
                words, raw = diarize_deepgram(args.audio, args.deepgram_key)
            elif provider == "soniox":
                if not args.soniox_key:
                    print(f"[{provider}] No API key — skipping")
                    continue
                words, raw = diarize_soniox(args.audio, args.soniox_key)
            elif provider == "speechmatics":
                if not args.speechmatics_key:
                    print(f"[{provider}] No API key — skipping")
                    continue
                words, raw = diarize_speechmatics(args.audio, args.speechmatics_key)
            else:
                print(f"Unknown provider: {provider}")
                continue

            # Save raw response
            raw_path = os.path.join(output_dir, f"raw_{provider}.json")
            with open(raw_path, "w") as f:
                json.dump(raw, f, indent=2, default=str)

            segments = build_segments(words)
            all_segments[provider] = segments
            print_timeline(provider, segments)

            if args.split:
                split_audio(args.audio, segments, output_dir, provider)

        except Exception as e:
            print(f"[{provider}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Compare if multiple providers ran
    if len(all_segments) > 1:
        compare_providers(all_segments)

    # Save comparison data
    comparison_path = os.path.join(output_dir, "comparison.json")
    export = {}
    for prov, segs in all_segments.items():
        export[prov] = [{
            "speaker": s["speaker"], "start": s["start"],
            "end": s["end"], "text": s["text"]
        } for s in segs]
    with open(comparison_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
