"""Cut the demo clip and measure its waveform envelope.

Window 2326.0-2340.0 s of 2107006::ep005 is chosen because it contains the
whole story the slide tells: speaker 0 finishes, the floor passes to speaker 3
at 2328.82 s, and the clip ends on the exact phrase the judge cited as
evidence for the negative label.

Writes the mp3 and a JSON envelope the slide builder draws as bars.

    python -X utf8 presentation_assets/build_clip_assets.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
import search_core as sc                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "2107006" / "05.mp3"
OUT_DIR = ROOT / "presentation_assets"
CLIP = OUT_DIR / "demo_clip_2107006_ep005.mp3"
ENVELOPE = OUT_DIR / "clip_envelope.json"

START, DURATION = 2326.0, 14.0
BARS = 84


def main() -> int:
    ffmpeg = sc._ffmpeg_exe()
    if not ffmpeg:
        raise SystemExit("no ffmpeg available")

    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(START), "-t", str(DURATION), "-i", str(SOURCE),
         "-ac", "1", "-ar", "44100", "-b:a", "96k", str(CLIP)],
        check=True)
    print(f"clip: {CLIP.name}, {CLIP.stat().st_size/1024:.0f} KB, "
          f"{sc.probe_duration(CLIP):.2f} s")

    # decode to raw mono PCM and take the RMS of each bucket
    raw = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-i", str(CLIP), "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", "8000", "-"],
        check=True, capture_output=True).stdout
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    usable = len(samples) - (len(samples) % BARS)
    buckets = samples[:usable].reshape(BARS, -1)
    rms = np.sqrt((buckets ** 2).mean(axis=1))
    # a mild power curve keeps quiet speech visible without flattening peaks
    shaped = (rms / max(rms.max(), 1e-9)) ** 0.62

    ENVELOPE.write_text(json.dumps({
        "source": "2107006::ep005",
        "start_sec": START,
        "duration_sec": DURATION,
        "bars": [round(float(v), 4) for v in shaped],
    }, indent=1), encoding="utf-8")
    print(f"envelope: {BARS} bars, peak {rms.max():.3f}, "
          f"median {float(np.median(rms)):.3f} -> {ENVELOPE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
