"""Stage 3: split the labelled pairs into train, validation and test.

The split is by EPISODE, never by row. Two utterances from the same talk show
share the panel, the topic, the host's phrasing and the same ASR error
pattern, so a random row split lets the model recognise the episode instead of
the sentiment and reports a score the model cannot repeat on a new recording.

Test episodes were chosen to carry as much of the hand-read gold set as
possible while keeping the training pool intact, so the final number is
measured against human labels on recordings the model never saw.

    python sentiment/split_data.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Held out entirely. Chosen because they carry 59 of the 180 gold pairs.
TEST_EPISODES = {
    "2107001::ep004",
    "2107015::ep001",
    "2107009::ep004",
    "2107006::ep004",
}
# Used only for early stopping and threshold choice, never for fitting.
VAL_EPISODES = {
    "2107006::ep003",
    "2107004::ep002",
    "2107009::ep002",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="sentiment/data/pairs_labelled.jsonl")
    ap.add_argument("--gold-sample", default="sentiment/data/gold_sample.jsonl")
    ap.add_argument("--gold-labels", default="sentiment/data/gold_labels.json")
    ap.add_argument("--out", default="sentiment/data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    rows = [json.loads(line) for line in Path(args.pairs).open(encoding="utf-8")]

    gold_rows = [json.loads(line) for line in Path(args.gold_sample).open(encoding="utf-8")]
    gold_map = json.load(Path(args.gold_labels).open(encoding="utf-8"))["labels"]
    gold_by_pair = {
        r["pair_id"]: gold_map[str(r["gold_index"])]
        for r in gold_rows if str(r["gold_index"]) in gold_map
    }

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        episode = row["global_episode"]
        name = "test" if episode in TEST_EPISODES else "val" if episode in VAL_EPISODES else "train"
        row = {**row, "gold_label": gold_by_pair.get(row["pair_id"])}
        splits[name].append(row)

    for name, items in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{'split':7s} {'pairs':>6s} {'episodes':>9s} {'gold':>5s}   silver distribution")
    for name, items in splits.items():
        counts = Counter(r["silver_label"] for r in items)
        n_gold = sum(1 for r in items if r["gold_label"])
        eps = len({r["global_episode"] for r in items})
        dist = " ".join(f"{k[:3]}={counts[k]}" for k in ("negative", "non_evaluative", "positive"))
        print(f"{name:7s} {len(items):6d} {eps:9d} {n_gold:5d}   {dist}")

    # A leak check is cheap and catches a bad edit to the episode lists.
    train_eps = {r["global_episode"] for r in splits["train"]}
    for name in ("val", "test"):
        overlap = train_eps & {r["global_episode"] for r in splits[name]}
        assert not overlap, f"{name} shares episodes with train: {overlap}"
    print("\nno episode appears in more than one split")


if __name__ == "__main__":
    main()
