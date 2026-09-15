"""Stage C: fold the judge's output back onto the corpus, derive speaker labels
by aggregation, and cut episode-disjoint train/val/test splits.

Three jobs:

1. **Merge.** Attach each turn's judge label, confidence and evidence to the turn
   row. With --passes > 1 on the judge, resolve the passes by majority and record
   the agreement rate - that number is the corpus's self-consistency and belongs
   in the report.
2. **Aggregate.** Compute speaker labels from the turn labels five different ways
   and score each against the judge's own holistic verdict. This says which
   aggregation rule the fine-tuned BERT should use at inference time, and it is
   measured rather than assumed.
3. **Split.** By EPISODE, never by row. Two turns from one talk show share the
   panel, the topic, the host's phrasing and the same ASR error signature, so a
   random row split lets the model recognise the episode instead of the stance.

    python speaker_sentiment/make_datasets.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TURN_CLASSES = ["negative", "neutral", "positive"]
SPEAKER_CLASSES = ["negative", "mixed", "neutral", "positive"]
CONF_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}


def majority(labels: list[str]) -> tuple[str, float]:
    """Winning label and the fraction of passes that agreed on it."""
    counts = Counter(labels)
    label, n = counts.most_common(1)[0]
    return label, n / len(labels)


def programme_key(title: str) -> str:
    """Collapse an episode title to the programme that produced it, so the split
    can be checked for host leakage: ATN Business & Finance supplies several
    episodes with the same presenter, and putting two of them either side of the
    split leaks that presenter's voice into test."""
    text = title.lower()
    for name in ("atn business", "business talk", "money talks", "business times",
                 "star biz", "stream talk", "the business review", "30 minutes",
                 "muktobak", "business & finance"):
        if name in text:
            return name
    return re.sub(r"[^a-zঀ-৿]+", "", text)[:24] or "unknown"


# ---------------------------------------------------------------- aggregation

def agg_majority(turns: list[dict[str, Any]]) -> tuple[str, float]:
    """Most common polar label; neutral if polar turns do not dominate."""
    counts = Counter(t["label"] for t in turns)
    pos, neg = counts["positive"], counts["negative"]
    total = max(len(turns), 1)
    if (pos + neg) / total < 0.15:
        return "neutral", 0.0
    if pos and neg and min(pos, neg) / max(pos, neg) >= 0.5:
        return "mixed", (pos - neg) / (pos + neg)
    return ("positive" if pos > neg else "negative"), (pos - neg) / max(pos + neg, 1)


def net_polarity(turns: list[dict[str, Any]], weight_key: str | None) -> float:
    """(positive - negative) / (positive + negative), optionally weighted.

    Weighting by words is the standard move in the earnings-call literature: a
    four-minute answer should not count the same as a one-line interjection."""
    pos = neg = 0.0
    for t in turns:
        w = 1.0
        if weight_key == "words":
            w = t.get("n_words", 1)
        elif weight_key == "confidence":
            w = CONF_WEIGHT.get(t.get("confidence", "medium"), 0.7)
        elif weight_key == "both":
            w = t.get("n_words", 1) * CONF_WEIGHT.get(t.get("confidence", "medium"), 0.7)
        if t["label"] == "positive":
            pos += w
        elif t["label"] == "negative":
            neg += w
    return (pos - neg) / (pos + neg) if (pos + neg) else 0.0


def score_to_label(turns: list[dict[str, Any]], score: float,
                   polar_floor: float = 0.15, mixed_band: float = 0.34) -> str:
    """Turn a continuous stance score into the 4-class speaker label.

    A speaker with almost no polar turns is neutral regardless of score - the
    ratio is unstable when its denominator is two turns. Inside the band, both
    directions are substantially present, which is what `mixed` means."""
    polar = sum(1 for t in turns if t["label"] in ("positive", "negative"))
    if polar / max(len(turns), 1) < polar_floor:
        return "neutral"
    if abs(score) < mixed_band:
        return "mixed"
    return "positive" if score > 0 else "negative"


AGGREGATORS = {
    "majority_vote": lambda ts: agg_majority(ts)[0],
    "net_polarity": lambda ts: score_to_label(ts, net_polarity(ts, None)),
    "word_weighted": lambda ts: score_to_label(ts, net_polarity(ts, "words")),
    "conf_weighted": lambda ts: score_to_label(ts, net_polarity(ts, "confidence")),
    "word_conf_weighted": lambda ts: score_to_label(ts, net_polarity(ts, "both")),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="speaker_sentiment/data")
    ap.add_argument("--judgements", default="speaker_sentiment/data/judgements.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--test-episodes", type=int, default=4)
    ap.add_argument("--val-episodes", type=int, default=3)
    ap.add_argument("--group-by", default="episode", choices=["episode", "programme"],
                    help="unit held out. 'programme' is the stricter choice: it keeps "
                         "every episode of one show (and so one presenter) on one side "
                         "of the split, at the cost of coarser split sizes.")
    args = ap.parse_args()

    data = Path(args.data)
    turns = {t["turn_id"]: t for t in
             (json.loads(l) for l in (data / "turns.jsonl").open(encoding="utf-8"))}
    speakers = {s["speaker_key"]: s for s in
                (json.loads(l) for l in (data / "speakers.jsonl").open(encoding="utf-8"))}

    judge_path = Path(args.judgements)
    if not judge_path.is_file():
        raise SystemExit(f"{judge_path} not found - run llm_judge.py first")
    judgements = [json.loads(l) for l in judge_path.open(encoding="utf-8")]

    # ---- 1. merge, resolving multiple passes ----------------------------------
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for j in judgements:
        by_speaker[j["speaker_key"]].append(j)

    turn_votes: dict[str, list[str]] = defaultdict(list)
    turn_meta: dict[str, dict] = {}
    for key, passes in by_speaker.items():
        for j in passes:
            for item in j["turn_labels"]:
                turn_votes[item["turn_id"]].append(item["label"])
                turn_meta.setdefault(item["turn_id"], item)

    labelled_turns: list[dict[str, Any]] = []
    agreements: list[float] = []
    for turn_id, votes in turn_votes.items():
        label, agreement = majority(votes)
        if len(votes) > 1:
            agreements.append(agreement)
        if label == "illegible":
            continue                       # judged unreadable: not supervision
        base = turns[turn_id]
        meta = turn_meta[turn_id]
        labelled_turns.append({
            **{k: base[k] for k in (
                "turn_id", "roll", "episode", "global_episode", "turn_index",
                "speaker_id", "role", "start_sec", "end_sec", "text", "n_words",
                "asr_confidence", "diarization_provider")},
            "speaker_key": f"{base['global_episode']}::S{base['speaker_id']}",
            "label": label,
            "judge_confidence": meta["confidence"],
            "evidence": meta["evidence"],
            "n_passes": len(votes),
            "pass_agreement": round(agreement, 3),
            # Sample weight for training: a low-confidence label on a turn the
            # ASR was unsure of should not push the gradient as hard as a
            # high-confidence label on a clean one.
            "weight": round(CONF_WEIGHT.get(meta["confidence"], 0.7) * agreement, 3),
        })

    # ---- 2. speaker labels, judged vs aggregated ------------------------------
    turns_by_speaker: dict[str, list[dict]] = defaultdict(list)
    for t in labelled_turns:
        turns_by_speaker[t["speaker_key"]].append(t)

    speaker_rows: list[dict[str, Any]] = []
    for key, passes in by_speaker.items():
        own = sorted(turns_by_speaker.get(key, []), key=lambda r: r["turn_index"])
        if not own:
            continue
        judged, agreement = majority([p["speaker_label"] for p in passes])
        base = speakers[key]
        # A verdict is only meaningful where the speaker id is real. Turns from
        # fallback-diarized episodes and from unassigned buckets are perfectly
        # good turn-level supervision - a turn's stance does not depend on
        # knowing who said it - but their "speaker" is not one person.
        valid = bool(passes[0].get("speaker_level_valid", base.get("judgeable", True)))
        counts = Counter(t["label"] for t in own)
        row = {
            "speaker_key": key,
            "roll": base["roll"],
            "episode": base["episode"],
            "global_episode": base["global_episode"],
            "programme": base["programme"],
            "programme_key": programme_key(base["programme"]),
            "speaker_id": base["speaker_id"],
            "role": base["role"],
            "speaker_level_valid": valid,
            "n_turns": len(own),
            "n_words": sum(t["n_words"] for t in own),
            "judge_label": judged,
            "judge_agreement": round(agreement, 3),
            "judge_stance_score": round(
                sum(p["stance_score"] for p in passes) / len(passes), 3),
            "judge_confidence": passes[0]["speaker_confidence"],
            "transcript_quality": passes[0]["transcript_quality"],
            "dominant_topics": passes[0]["dominant_topics"],
            "summary": passes[0]["speaker_summary"],
            "rationale": passes[0]["speaker_rationale"],
            "n_negative": counts["negative"],
            "n_neutral": counts["neutral"],
            "n_positive": counts["positive"],
            "net_polarity": round(net_polarity(own, None), 3),
            "net_polarity_word_weighted": round(net_polarity(own, "words"), 3),
            "script": "\n\n".join(t["text"] for t in own),
        }
        for name, fn in AGGREGATORS.items():
            row[f"agg_{name}"] = fn(own)
        speaker_rows.append(row)
    speaker_rows.sort(key=lambda r: r["speaker_key"])

    # ---- 3. group-disjoint split ---------------------------------------------
    # Held out by episode (default) or by programme. Grouping by programme is
    # stricter: several rolls recorded multiple episodes of the same show, so an
    # episode split can still put one presenter either side of the boundary.
    group_of_ep = {}
    for r in speaker_rows:
        group_of_ep[r["global_episode"]] = (
            r["programme_key"] if args.group_by == "programme" else r["global_episode"])
    groups = sorted(set(group_of_ep.values()))
    rng = random.Random(args.seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    n_test = args.test_episodes if args.group_by == "episode" else max(2, args.test_episodes // 2)
    n_val = args.val_episodes if args.group_by == "episode" else max(1, args.val_episodes // 2)
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])

    def split_of(ep: str) -> str:
        g = group_of_ep.get(ep, ep)
        return "test" if g in test_groups else "val" if g in val_groups else "train"

    for r in labelled_turns:
        r["split"] = split_of(r["global_episode"])
    for r in speaker_rows:
        r["split"] = split_of(r["global_episode"])

    out = data
    for name in ("train", "val", "test"):
        with (out / f"turns_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in sorted((x for x in labelled_turns if x["split"] == name),
                            key=lambda x: x["turn_id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out / f"speakers_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in (x for x in speaker_rows if x["split"] == name):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "turns_labelled.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(labelled_turns, key=lambda x: x["turn_id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "speakers_labelled.jsonl").open("w", encoding="utf-8") as fh:
        for r in speaker_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report ---------------------------------------------------------------
    print(f"judged speakers      {len(speaker_rows)}")
    print(f"labelled turns       {len(labelled_turns):,} "
          f"(dropped {sum(1 for v in turn_votes.values() if majority(v)[0] == 'illegible')} illegible)")
    if agreements:
        print(f"multi-pass agreement {sum(agreements) / len(agreements):.3f} "
              f"over {len(agreements):,} turns")

    print("\nturn label distribution:")
    counts = Counter(r["label"] for r in labelled_turns)
    for c in TURN_CLASSES:
        print(f"  {c:10s} {counts[c]:5d}  ({counts[c] / max(len(labelled_turns), 1):.1%})")
    print("\n  by role:")
    for role in sorted({r["role"] for r in labelled_turns}):
        sub = Counter(r["label"] for r in labelled_turns if r["role"] == role)
        print(f"    {role:8s} " + "  ".join(f"{c[:3]}={sub[c]:4d}" for c in TURN_CLASSES))

    valid_rows = [r for r in speaker_rows if r["speaker_level_valid"]]
    print(f"\nspeaker label distribution (judge holistic, "
          f"{len(valid_rows)} of {len(speaker_rows)} units usable):")
    sc = Counter(r["judge_label"] for r in valid_rows)
    for c in SPEAKER_CLASSES:
        print(f"  {c:10s} {sc[c]:3d}")

    print("\naggregation rule vs judge holistic verdict:")
    for name in AGGREGATORS:
        hits = sum(1 for r in valid_rows if r[f"agg_{name}"] == r["judge_label"])
        polar = sum(1 for r in valid_rows
                    if (r[f"agg_{name}"] in ("positive", "negative"))
                    == (r["judge_label"] in ("positive", "negative")))
        print(f"  {name:20s} exact={hits / max(len(valid_rows), 1):.1%}  "
              f"polar-direction={polar / max(len(valid_rows), 1):.1%}")

    print(f"\n{'split':7s}{'eps':>5s}{'speakers':>10s}{'turns':>7s}   turn distribution")
    for name in ("train", "val", "test"):
        ts = [r for r in labelled_turns if r["split"] == name]
        ss = [r for r in speaker_rows if r["split"] == name and r["speaker_level_valid"]]
        c = Counter(r["label"] for r in ts)
        dist = " ".join(f"{k[:3]}={c[k]}" for k in TURN_CLASSES)
        print(f"{name:7s}{len({r['global_episode'] for r in ts}):5d}{len(ss):10d}"
              f"{len(ts):7d}   {dist}")

    # A programme that straddles the split leaks a presenter's voice into test.
    train_progs = {r["programme_key"] for r in valid_rows if r["split"] == "train"}
    for name in ("val", "test"):
        shared = train_progs & {r["programme_key"] for r in valid_rows if r["split"] == name}
        if shared:
            print(f"\nWARNING: {name} shares programmes with train: {sorted(shared)}")
            print("  same presenter either side of the split - re-roll --seed or "
                  "split on programme_key instead of episode")
    print(f"\nwrote {out}/turns_{{train,val,test}}.jsonl and speakers_*.jsonl")


if __name__ == "__main__":
    main()
