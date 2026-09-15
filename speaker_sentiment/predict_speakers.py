"""Run a fine-tuned turn-level model over a turns file and emit speaker verdicts.

This is section 8 + section 12 of the notebook as a command-line tool, so a new
episode can be scored without opening Jupyter.

    python -X utf8 speaker_sentiment/build_speaker_corpus.py          # new audio -> turns.jsonl
    python -X utf8 speaker_sentiment/predict_speakers.py \
           --model speaker_sentiment/runs/banglabert_turn/model \
           --turns speaker_sentiment/data/turns.jsonl \
           --out   speaker_sentiment/runs/banglabert_turn/predictions.jsonl

The aggregation rule, input mode and max_len are read from `metrics.json` next to
the model directory, so the CLI cannot silently disagree with the notebook that
trained it. Override any of them on the command line if you want to.

CPU is fine here: inference over the whole 1,989-turn corpus takes a few minutes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TURN_CLASSES = ["negative", "neutral", "positive"]
SPEAKER_CLASSES = ["negative", "mixed", "neutral", "positive"]
ROLE_TOKEN = {"host": "সঞ্চালক", "guest": "অতিথি",
              "minor": "অন্য", "unassigned": "অজানা"}

try:
    from normalizer import normalize as bn_normalize
except ImportError:                                   # pragma: no cover
    sys.stderr.write(
        "csebuetnlp normalizer not installed - falling back to NFC only. "
        "Install it for parity with training:\n"
        "  pip install git+https://github.com/csebuetnlp/normalizer\n")

    def bn_normalize(t, **kw):
        return unicodedata.normalize("NFC", t)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", bn_normalize(str(text))).strip()


def build_input(row: dict, prev_text: str, mode: str):
    """Identical to the notebook's `build_input`. Keep the two in step."""
    text = clean(row["text"])
    role = ROLE_TOKEN.get(row.get("role", "unassigned"), "অজানা")
    if mode == "plain":
        return text, None
    if mode == "role_turn":
        return role, text
    if mode == "context":
        return (clean(prev_text) if prev_text else role), text
    if mode == "role_context":
        head = role if not prev_text else f"{role} {clean(prev_text)}"
        return head, text
    raise ValueError(f"unknown input mode: {mode}")


def net(pos: np.ndarray, neg: np.ndarray, weights: np.ndarray) -> float:
    p, n = float((pos * weights).sum()), float((neg * weights).sum())
    return (p - n) / (p + n) if (p + n) > 0 else 0.0


def topk(pos: np.ndarray, neg: np.ndarray, k: int = 5):
    idx = np.argsort(-np.abs(pos - neg))[:k]
    return pos[idx], neg[idx], np.ones(len(idx))


def to_speaker_label(score: float, polar_share: float,
                     polar_floor: float = 0.15, mixed_band: float = 0.34) -> str:
    if polar_share < polar_floor:
        return "neutral"
    if abs(score) < mixed_band:
        return "mixed"
    return "positive" if score > 0 else "negative"


def aggregate(rows: list[dict], rule: str) -> tuple[float, float]:
    pos = np.array([r["p_positive"] for r in rows])
    neg = np.array([r["p_negative"] for r in rows])
    words = np.array([float(r.get("n_words") or len(str(r["text"]).split()))
                      for r in rows])
    if rule == "word_weighted":
        score = net(pos, neg, words)
    elif rule == "dur_weighted":
        dur = np.array([float(r.get("end_sec", 0)) - float(r.get("start_sec", 0))
                        for r in rows])
        score = net(pos, neg, dur if dur.sum() > 0 else np.ones_like(pos))
    elif rule == "top_k":
        score = net(*topk(pos, neg, k=5))
    elif rule == "hard_vote":
        hp = np.array([r["pred"] == "positive" for r in rows], dtype=float)
        hn = np.array([r["pred"] == "negative" for r in rows], dtype=float)
        score = net(hp, hn, np.ones_like(pos))
    else:                                              # unweighted
        score = net(pos, neg, np.ones_like(pos))
    polar = float(np.mean([r["pred"] != "neutral" for r in rows]))
    return score, polar


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help="directory written by the notebook's section 12")
    ap.add_argument("--turns", required=True, type=Path,
                    help="turns JSONL from build_speaker_corpus.py (or a split file)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write per-speaker verdicts (default: alongside the model)")
    ap.add_argument("--turn-out", type=Path, default=None,
                    help="also write per-turn predictions here")
    ap.add_argument("--rule", default=None,
                    choices=["unweighted", "word_weighted", "dur_weighted",
                             "top_k", "hard_vote"],
                    help="aggregation rule (default: best_aggregation_rule in metrics.json)")
    ap.add_argument("--input-mode", default=None,
                    choices=["plain", "role_turn", "context", "role_context"])
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-turns", type=int, default=3,
                    help="speakers with fewer turns get a verdict but are flagged thin")
    args = ap.parse_args()

    # Training-time settings, so inference cannot drift from the notebook.
    meta_path = args.model.parent / "metrics.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not meta:
        sys.stderr.write(f"no metrics.json at {meta_path}; using CLI defaults\n")
    rule = args.rule or meta.get("best_aggregation_rule", "unweighted")
    mode = args.input_mode or meta.get("input_mode", "role_turn")
    max_len = args.max_len or meta.get("max_len", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    turns = load_jsonl(args.turns)
    turns.sort(key=lambda r: (r.get("global_episode", ""), r.get("turn_index", 0)))
    prev_by_episode: dict[str, str] = {}
    for r in turns:
        ep = r.get("global_episode", "")
        r["_prev"] = prev_by_episode.get(ep, "")
        prev_by_episode[ep] = r["text"]

    print(f"{len(turns)} turns | model={args.model} | mode={mode} "
          f"| max_len={max_len} | rule={rule} | device={device}")

    with torch.no_grad():
        for start in range(0, len(turns), args.batch_size):
            batch = turns[start:start + args.batch_size]
            pairs = [build_input(r, r["_prev"], mode) for r in batch]
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs] if pairs[0][1] is not None else None
            enc = tok(a, b, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            probs = F.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            for r, p in zip(batch, probs):
                for i, c in enumerate(TURN_CLASSES):
                    r[f"p_{c}"] = float(p[i])
                r["pred"] = TURN_CLASSES[int(p.argmax())]
            done = min(start + args.batch_size, len(turns))
            print(f"\r  {done}/{len(turns)}", end="", flush=True)
    print()

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in turns:
        by_speaker[r.get("speaker_key", "unknown")].append(r)

    verdicts = []
    for key, rows in sorted(by_speaker.items()):
        score, polar = aggregate(rows, rule)
        verdicts.append({
            "speaker_key": key,
            "global_episode": rows[0].get("global_episode", ""),
            "role": rows[0].get("role", "unassigned"),
            "n_turns": len(rows),
            "n_words": int(sum(float(r.get("n_words") or 0) for r in rows)),
            "speaker_label": to_speaker_label(score, polar),
            "stance_score": round(score, 4),
            "polar_share": round(polar, 4),
            "aggregation_rule": rule,
            "thin": len(rows) < args.min_turns,
            "turn_pred_counts": {c: sum(r["pred"] == c for r in rows)
                                 for c in TURN_CLASSES},
        })

    out = args.out or (args.model.parent / "speaker_predictions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")

    if args.turn_out:
        args.turn_out.parent.mkdir(parents=True, exist_ok=True)
        with args.turn_out.open("w", encoding="utf-8") as fh:
            for r in turns:
                r.pop("_prev", None)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist: dict[str, int] = defaultdict(int)
    for v in verdicts:
        dist[v["speaker_label"]] += 1
    print(f"\n{len(verdicts)} speaker units -> {out}")
    for c in SPEAKER_CLASSES:
        print(f"  {c:9s} {dist[c]:4d}")
    thin = sum(v["thin"] for v in verdicts)
    if thin:
        print(f"  ({thin} units have < {args.min_turns} turns and are flagged `thin`; "
              f"their verdicts rest on very little evidence)")


if __name__ == "__main__":
    main()
