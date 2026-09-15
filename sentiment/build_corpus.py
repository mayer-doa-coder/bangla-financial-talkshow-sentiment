"""Stage 1: turn the fused speaker-attributed transcripts into a
target-aware sentiment corpus.

Reads every roll's fused/*.json, re-attaches the ASR confidence that the
fusion stage dropped, flags the two transcript defects this corpus actually
has (decoder repetition loops and chunk-boundary truncation), finds
financial targets with the project's own 28-entry lexicon, and writes one
row per (utterance, target) pair.

    python sentiment/build_corpus.py --data-root "D:/ML DATASETS"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

# Bengali dependent signs. A word may not begin with one; when a segment
# does, the chunk stitcher ate its first consonant.
COMBINING = set(
    "\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c7\u09c8"
    "\u09cb\u09cc\u09cd\u09d7\u09bc"
)


def resolve_stage(bundle: Path, stage: str) -> Path | None:
    """Bundles were exported in two shapes: stage folders directly under the
    bundle, or wrapped in data/. Accept both."""
    for candidate in (bundle / stage, bundle / "data" / stage):
        if candidate.is_dir():
            return candidate
    return None


def find_bundles(data_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for roll in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for bundle in sorted(roll.iterdir()):
            if bundle.is_dir() and resolve_stage(bundle, "fused"):
                out.append((roll.name, bundle))
    return out


def load_lexicon(path: Path) -> list[dict[str, Any]]:
    """Expanded target lexicon. The project shipped 28 named entities, which
    matched only 209 of 3,532 utterances because talk-show speech names
    concepts more often than institutions. This file keeps every original
    entry (source=base) and adds corpus-grounded ones (source=expanded)."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "canonical": row["canonical"],
                "target_type": row["target_type"],
                "stems": [s.strip() for s in row["stems"].split("|") if s.strip()],
                "exclude": [s.strip() for s in row["exclude"].split("|") if s.strip()],
                "source": row["source"],
            })
    n_base = sum(1 for r in rows if r["source"] == "base")
    print(f"lexicon: {len(rows)} targets ({n_base} base, {len(rows)-n_base} expanded)")
    return rows


def norm(text: str) -> str:
    """Text survives a Drive round trip in mixed normal forms; compare in NFC."""
    return unicodedata.normalize("NFC", text)


def repetition_score(text: str) -> tuple[int, float]:
    """Longest run of one repeated token, and the unique-token ratio.

    The decoder in this run fell into loops such as 'ডাকার ডাকার ডাকার ...'.
    Both numbers are needed: the run catches a single repeated token, the
    ratio catches an alternating loop."""
    tokens = text.split()
    if not tokens:
        return 0, 1.0
    longest = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        longest = max(longest, run)
    return longest, len(set(tokens)) / len(tokens)


def leading_truncated(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in COMBINING


def asr_confidence_index(asr_dir: Path | None, episode: str) -> list[tuple[float, float, float]]:
    if asr_dir is None:
        return []
    path = asr_dir / f"{episode}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        (float(s["start"]), float(s["end"]), float(s.get("confidence", 0.0)))
        for s in data.get("segments", [])
    ]


def mean_confidence(index, start: float, end: float) -> float | None:
    """Duration-weighted mean of the ASR segment log-probabilities that
    overlap this utterance."""
    total = weight = 0.0
    for seg_start, seg_end, conf in index:
        overlap = min(end, seg_end) - max(start, seg_start)
        if overlap > 0:
            total += conf * overlap
            weight += overlap
    return round(total / weight, 4) if weight else None


def build_matcher(lexicon: list[dict[str, Any]]):
    """Bangla is agglutinative and this ASR emits no punctuation, so a target
    is matched as a *stem prefix* on a whitespace token, not as an exact
    string. One stem 'ব্যাংক' therefore covers ব্যাংক, ব্যাংকের, ব্যাংকে,
    ব্যাংকগুলো and ব্যাংকগুলোর. Multi-word stems are matched as phrases."""
    patterns = []
    for entry in lexicon:
        excludes = [
            re.compile(r"(?:(?<=\s)|^)" + re.escape(norm(x)) + r"\S*(?=\s|$)")
            for x in entry["exclude"]
        ]
        for stem in entry["stems"]:
            stem_n = norm(stem)
            if " " in stem_n:
                body = re.escape(stem_n).replace(r"\ ", r"\s+") + r"\S*"
            else:
                body = re.escape(stem_n) + r"\S*"
            pattern = re.compile(r"(?:(?<=\s)|^)" + body + r"(?=\s|$)")
            patterns.append({
                "pattern": pattern, "canonical": entry["canonical"],
                "target_type": entry["target_type"], "stem": stem_n,
                "excludes": excludes,
            })
    # Longest stem first, so a specific target claims its span before the
    # generic word nested inside it (banking sector before banks).
    patterns.sort(key=lambda p: len(p["stem"]), reverse=True)
    return patterns


def find_targets(text: str, matchers) -> list[dict[str, Any]]:
    """One entry per distinct canonical target, keeping every mention span so
    the model can be told where to look."""
    found: dict[str, dict[str, Any]] = {}
    claimed: list[tuple[int, int]] = []
    for m in matchers:
        for match in m["pattern"].finditer(text):
            span = match.span()
            # A longer stem already covering this span wins.
            if any(span[0] >= c0 and span[1] <= c1 for c0, c1 in claimed):
                continue
            # Guard against stems that prefix an unrelated word
            # (সুদ 'interest' vs সুদান 'Sudan').
            if any(ex.fullmatch(match.group(0)) for ex in m["excludes"]):
                continue
            claimed.append(span)
            slot = found.setdefault(m["canonical"], {
                "canonical": m["canonical"],
                "target_type": m["target_type"],
                "mentions": [],
            })
            slot["mentions"].append({
                "surface": match.group(0), "start": span[0], "end": span[1],
            })
    return list(found.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--lexicon", default="sentiment/lexicon/financial_targets.csv")
    ap.add_argument("--out", default="sentiment/data")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lexicon = load_lexicon(Path(args.lexicon).expanduser().resolve())
    matchers = build_matcher(lexicon)

    rows: list[dict[str, Any]] = []
    utt_records: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []

    for roll, bundle in find_bundles(data_root):
        fused_dir = resolve_stage(bundle, "fused")
        asr_dir = resolve_stage(bundle, "asr")
        for fused_path in sorted(fused_dir.glob("*.json")):
            episode = fused_path.stem
            data = json.loads(fused_path.read_text(encoding="utf-8"))
            conf_index = asr_confidence_index(asr_dir, episode)
            gid = f"{roll}::{episode}"

            n_rep = n_trunc = n_short = 0
            for utt in data.get("utterances", []):
                text = norm(utt.get("text_normalized") or utt.get("text_verbatim") or "").strip()
                if not text:
                    continue
                longest_run, unique_ratio = repetition_score(text)
                truncated = leading_truncated(text)
                words = len(text.split())
                conf = mean_confidence(conf_index, float(utt["start_sec"]), float(utt["end_sec"]))

                # A loop is a run of 4+ identical tokens, or very low lexical
                # variety across a long span. Both thresholds were set by
                # reading the flagged segments.
                is_loop = longest_run >= 4 or (words >= 20 and unique_ratio < 0.35)
                n_rep += is_loop
                n_trunc += truncated
                n_short += words < 4

                record = {
                    "utterance_id": utt["utterance_id"],
                    "global_id": f"{gid}::{utt['utterance_id']}",
                    "roll": roll,
                    "episode": episode,
                    "global_episode": gid,
                    "speaker_id": utt.get("speaker_id"),
                    "start_sec": round(float(utt["start_sec"]), 2),
                    "end_sec": round(float(utt["end_sec"]), 2),
                    "text": text,
                    "n_words": words,
                    "asr_confidence": conf,
                    "longest_token_run": longest_run,
                    "unique_token_ratio": round(unique_ratio, 3),
                    "flag_repetition_loop": bool(is_loop),
                    "flag_leading_truncation": bool(truncated),
                    "flag_too_short": words < 4,
                }
                utt_records.append(record)

                carry = (
                    "global_id", "roll", "episode", "global_episode", "speaker_id",
                    "start_sec", "end_sec", "text", "n_words", "asr_confidence",
                    "flag_repetition_loop", "flag_leading_truncation", "flag_too_short",
                )
                for target in find_targets(text, matchers):
                    rows.append({
                        **{k: record[k] for k in carry},
                        "target": target["canonical"],
                        "target_type": target["target_type"],
                        "mentions": target["mentions"],
                        "pair_id": f"{gid}::{utt['utterance_id']}::{target['canonical']}",
                    })

            episode_stats.append({
                "global_episode": gid, "roll": roll, "episode": episode,
                "utterances": len(data.get("utterances", [])),
                "words": data.get("stats", {}).get("n_words", 0),
                "unassigned_words": data.get("stats", {}).get("unassigned_words", 0),
                "repetition_loops": n_rep,
                "leading_truncations": n_trunc,
                "too_short": n_short,
            })

    with (out_dir / "utterances.jsonl").open("w", encoding="utf-8") as fh:
        for r in utt_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "episode_quality.json").open("w", encoding="utf-8") as fh:
        json.dump(episode_stats, fh, ensure_ascii=False, indent=1)

    by_type = Counter(r["target_type"] for r in rows)
    by_roll = Counter(r["roll"] for r in rows)
    flagged = sum(1 for r in utt_records if r["flag_repetition_loop"])
    trunc = sum(1 for r in utt_records if r["flag_leading_truncation"])
    total = max(len(utt_records), 1)

    print(f"\nepisodes            {len(episode_stats)}")
    print(f"utterances          {len(utt_records):,}")
    print(f"  repetition loops  {flagged:,} ({flagged/total:.1%})")
    print(f"  leading truncated {trunc:,} ({trunc/total:.1%})")
    print(f"target pairs        {len(rows):,}")
    print(f"  distinct targets  {len({r['target'] for r in rows})}")
    print("\nby target type:")
    for k, v in by_type.most_common():
        print(f"  {k:18s} {v:5d}")
    print("\nby roll:")
    for k, v in sorted(by_roll.items()):
        print(f"  {k:18s} {v:5d}")


if __name__ == "__main__":
    main()
