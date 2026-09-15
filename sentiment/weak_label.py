"""Stage 2: attach a silver polarity label to every (utterance, target) pair.

There is no human label anywhere in this project: all 3,023 annotation drafts
carry labels=[] and annotation_status="review_required". So supervision has to
be created before anything can be trained. This module does that with a
transparent rule set instead of a black box, which means every silver label
can be traced back to the exact words that produced it.

The rule that matters for financial text is that direction words are not
polar on their own. "বেড়েছে" (has increased) is good news for reserves and bad
news for inflation. Each target therefore carries an orientation and the
direction cue is multiplied by it. Entity targets have orientation 0, so they
are scored only by evaluative words.

    python sentiment/weak_label.py
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

# A cue inside this many tokens of the target mention is attributed to it.
WINDOW = 12
# Negation particles. Bangla puts these after the verb they negate.
NEGATORS = {"না", "নয়", "নেই", "নাই", "নাহ", "কখনো"}
# Score needed before a pair is called polar rather than non evaluative.
DECISION_THRESHOLD = 0.55


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_orientation(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as fh:
        return {r["canonical"]: int(r["orientation"]) for r in csv.DictReader(fh)}


def load_cues(path: Path) -> list[dict[str, Any]]:
    cues = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cues.append({
                "stem": norm(row["stem"]),
                "kind": row["kind"],
                "polarity": int(row["polarity"]),
                "weight": float(row["weight"]),
                # Only adjectives and verbs can be negated. A polar noun is
                # not flipped by a nearby particle: in "সংকট কাটছে না" the না
                # negates the verb, and the crisis is still bad news.
                "negatable": row["negatable"] == "1",
                "exclude": [x.strip() for x in row["exclude"].split("|") if x.strip()],
                "gloss": row["gloss"],
            })
    return cues


def build_cue_matcher(cues: list[dict[str, Any]]):
    out = []
    for cue in cues:
        # Same stem-prefix rule as target matching, so সংকট also fires on
        # সংকটের, সংকটে and সংকটকে.
        pattern = re.compile("^" + re.escape(cue["stem"]) + r"\S*$")
        blocked = [re.compile("^" + re.escape(norm(x)) + r"\S*$") for x in cue["exclude"]]
        out.append((pattern, blocked, cue))
    return out


def token_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def score_pair(text: str, mentions: list[dict[str, Any]], orientation: int,
               cue_matcher) -> tuple[float, list[dict[str, Any]]]:
    """Signed evidence score for one target inside one utterance."""
    spans = token_spans(text)
    if not spans:
        return 0.0, []

    # Character offset of each mention mapped onto a token index.
    target_idx = []
    for mention in mentions:
        for i, (start, end, _) in enumerate(spans):
            if start <= mention["start"] < end:
                target_idx.append(i)
                break
    if not target_idx:
        return 0.0, []

    score = 0.0
    evidence: list[dict[str, Any]] = []
    for i, (_, _, token) in enumerate(spans):
        distance = min(abs(i - t) for t in target_idx)
        if distance > WINDOW:
            continue
        for pattern, blocked, cue in cue_matcher:
            if not pattern.match(token):
                continue
            # Guard stems that prefix an unrelated word
            # (ভালো 'good' vs ভালোবাসা 'love').
            if any(b.match(token) for b in blocked):
                continue
            polarity = cue["polarity"]
            if cue["kind"] == "direction":
                # A rise is only good or bad relative to what is rising.
                if orientation == 0:
                    continue
                polarity *= orientation
            # Negation flips only what can be negated, and only when the
            # particle is adjacent. Bangla puts it right after its verb.
            negated = False
            if cue["negatable"]:
                # Bangla builds a negated verb complex as
                # ADJ + করতে + পারবেন + না, so the particle can sit three
                # tokens out and still be negating this cue.
                tail = {spans[j][2] for j in range(i + 1, min(i + 4, len(spans)))}
                negated = bool(tail & NEGATORS)
                if negated:
                    polarity *= -1
            # Nearer evidence counts for more.
            decay = 1.0 / (1.0 + distance / 4.0)
            contribution = polarity * cue["weight"] * decay
            score += contribution
            evidence.append({
                "token": token, "gloss": cue["gloss"], "kind": cue["kind"],
                "distance": distance, "negated": negated,
                "contribution": round(contribution, 3),
            })
            break
    return score, evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="sentiment/data/pairs.jsonl")
    ap.add_argument("--targets", default="sentiment/lexicon/financial_targets.csv")
    ap.add_argument("--cues", default="sentiment/lexicon/polarity_cues.csv")
    ap.add_argument("--out", default="sentiment/data/pairs_labelled.jsonl")
    args = ap.parse_args()

    orientation = load_orientation(Path(args.targets))
    cue_matcher = build_cue_matcher(load_cues(Path(args.cues)))

    rows = [json.loads(line) for line in Path(args.pairs).open(encoding="utf-8")]
    out_rows = []
    dropped = Counter()

    for row in rows:
        # A decoder loop has no meaning to read polarity from, and a fragment
        # under four words has no context. Both are excluded from supervision
        # rather than silently labelled.
        if row["flag_repetition_loop"]:
            dropped["repetition_loop"] += 1
            continue
        if row["n_words"] < 6:
            dropped["too_short"] += 1
            continue

        score, evidence = score_pair(
            row["text"], row["mentions"], orientation.get(row["target"], 0), cue_matcher
        )
        if score >= DECISION_THRESHOLD:
            label = "positive"
        elif score <= -DECISION_THRESHOLD:
            label = "negative"
        else:
            label = "non_evaluative"

        out_rows.append({
            **row,
            "silver_label": label,
            "silver_score": round(score, 3),
            "silver_evidence": evidence,
            "annotation_status": "labelled" if label != "non_evaluative" else "non_evaluative",
        })

    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["silver_label"] for r in out_rows)
    print(f"input pairs      {len(rows):,}")
    for k, v in dropped.most_common():
        print(f"  dropped {k:16s} {v:,}")
    print(f"labelled pairs   {len(out_rows):,}\n")
    for k in ("negative", "non_evaluative", "positive"):
        print(f"  {k:16s} {counts[k]:5,d}  ({counts[k]/max(len(out_rows),1):.1%})")

    print("\nby target type:")
    for ttype in sorted({r["target_type"] for r in out_rows}):
        sub = [r for r in out_rows if r["target_type"] == ttype]
        c = Counter(r["silver_label"] for r in sub)
        print(f"  {ttype:18s} n={len(sub):4d}  neg={c['negative']:4d} "
              f"non={c['non_evaluative']:4d} pos={c['positive']:4d}")


if __name__ == "__main__":
    main()
