"""Confirm the trained model will run on this machine, before the demo starts.

Run this once on the machine that will present the work. It checks the things
that actually break on a fresh computer, in the order they would break, and
says which of them are fatal and which only remove a feature.

    python -X utf8 demo/check_model_setup.py

It writes nothing and needs no network. A non-zero exit means the live-model
tab will not work; the transcript search and the recorded results still will.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENTIMENT = ROOT / "speaker_sentiment"
RUN = SENTIMENT / "runs" / "banglabert_turn"
MODEL = RUN / "model"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, check: str, detail: str) -> None:
    results.append((status, check, detail))
    print(f"  [{status}] {check}: {detail}", flush=True)


def check_python() -> None:
    version = sys.version_info
    ok = version >= (3, 10)
    record(PASS if ok else FAIL, "Python",
           f"{version.major}.{version.minor}.{version.micro}"
           + ("" if ok else " (3.10 or newer is required)"))


def check_packages() -> None:
    required = {"torch": "runs the model", "transformers": "loads the checkpoint",
                "gradio": "serves the demo", "pandas": "renders the tables"}
    optional = {"normalizer": "Bangla text normalisation the model was trained with",
                "imageio_ffmpeg": "audio clip extraction",
                "sentence_transformers": "semantic search",
                "rapidfuzz": "fuzzy search"}
    import importlib

    for name, why in required.items():
        try:
            module = importlib.import_module(name)
            record(PASS, name, f"{getattr(module, '__version__', 'installed')} ({why})")
        except Exception as exc:                                   # noqa: BLE001
            record(FAIL, name, f"missing, needed to {why} ({type(exc).__name__})")
    for name, why in optional.items():
        try:
            module = importlib.import_module(name)
            record(PASS, name, f"{getattr(module, '__version__', 'installed')} ({why})")
        except Exception:                                          # noqa: BLE001
            record(WARN, name, f"missing; {why} is degraded but the demo still runs")


def check_checkpoint() -> bool:
    if not MODEL.is_dir():
        record(FAIL, "checkpoint", f"not found at {MODEL}")
        return False
    weights = list(MODEL.glob("*.safetensors")) + list(MODEL.glob("*.bin"))
    if not weights:
        record(FAIL, "checkpoint", "config present but no weight file")
        return False
    size = sum(p.stat().st_size for p in MODEL.iterdir() if p.is_file()) / (1024 ** 2)
    record(PASS, "checkpoint", f"{size:.0f} MB in {MODEL.relative_to(ROOT)}")

    metrics = RUN / "metrics.json"
    if metrics.is_file():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        record(PASS, "run settings",
               f"input_mode={data.get('input_mode')} max_len={data.get('max_len')} "
               f"rule={data.get('best_aggregation_rule')}")
    else:
        record(WARN, "run settings", "metrics.json missing; defaults will be used")
    return True


def check_inference() -> bool:
    """Load the model offline and reproduce predictions the run already wrote."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_runner import ModelRunner

    runner = ModelRunner(SENTIMENT)
    start = time.time()
    try:
        runner._ensure_loaded()                                    # noqa: SLF001
    except Exception as exc:                                       # noqa: BLE001
        record(FAIL, "model load", f"{type(exc).__name__}: {exc}")
        return False
    record(PASS, "model load", f"loaded offline in {time.time() - start:.1f} s on CPU")

    recorded_path = RUN / "test_predictions.jsonl"
    turns_path = SENTIMENT / "data" / "turns_test.jsonl"
    if not (recorded_path.is_file() and turns_path.is_file()):
        record(WARN, "prediction check",
               "held-out predictions not found; cannot verify against the recorded run")
        return True

    recorded = {json.loads(line)["turn_id"]: json.loads(line)
                for line in recorded_path.open(encoding="utf-8") if line.strip()}
    turns = [json.loads(line) for line in turns_path.open(encoding="utf-8") if line.strip()]
    sample = turns[:24]
    start = time.time()
    predictions = runner.classify_turns(sample)
    elapsed = time.time() - start
    agree = sum(1 for p in predictions if recorded.get(p.turn_id, {}).get("pred") == p.pred)
    status = PASS if agree == len(sample) else FAIL
    record(status, "prediction check",
           f"reproduced {agree}/{len(sample)} recorded predictions "
           f"in {elapsed:.1f} s ({elapsed / len(sample) * 1000:.0f} ms per turn)")
    return status == PASS


def check_audio() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from search_core import _ffmpeg_exe

    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        record(PASS, "ffmpeg", "found; audio clips can be cut")
    else:
        record(WARN, "ffmpeg", "not found; the demo runs but cannot play audio excerpts")


def main() -> int:
    print("Checking that the trained model can run on this machine.\n")
    check_python()
    check_packages()
    print()
    ready = check_checkpoint()
    if ready:
        ready = check_inference()
    print()
    check_audio()

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    print()
    if failures:
        print(f"NOT READY: {len(failures)} blocking problem(s).")
        for _, check, detail in failures:
            print(f"  - {check}: {detail}")
        return 1
    print("READY. The live-model tab will work.")
    if warnings:
        print(f"{len(warnings)} optional component(s) missing; "
              "the demo runs with those features reduced.")
    print("\nStart the demo with:\n  python -X utf8 demo/app.py --data-root .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
