"""Stage 4a: the baseline the neural model has to beat.

Character n-grams rather than words, because this is ASR output. The decoder
writes মূল্যস্ফীতিিতি for মূল্যস্ফীতি and রেমিটেন্যান্স for রেমিট্যান্স, so a
word-level feature space treats every misrecognition as an unseen word.
Character 3-to-5-grams still see the shared substring.

The target is prepended as its own token so the same sentence can be scored
differently for two different targets.

    python sentiment/train_baseline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

LABELS = ["negative", "non_evaluative", "positive"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def encode(row: dict) -> str:
    """Target first, then the utterance. The separator token keeps the two
    fields from blending into one n-gram."""
    surfaces = " ".join(dict.fromkeys(m["surface"] for m in row["mentions"]))
    return f"{surfaces} ‖ {row['text']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sentiment/data")
    ap.add_argument("--out", default="sentiment/runs/baseline")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (load(data / f"{s}.jsonl") for s in ("train", "val", "test"))

    x_train = [encode(r) for r in train]
    y_train = [r["silver_label"] for r in train]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=3, max_features=200_000, sublinear_tf=True,
        )),
        # The corpus is 86 percent non evaluative. Without balancing, the
        # model reaches a high accuracy by never predicting a polarity at all.
        ("clf", LogisticRegression(
            max_iter=2000, C=4.0, class_weight="balanced",
        )),
    ])
    model.fit(x_train, y_train)

    report = {"model": "tfidf_char35_logreg", "n_train": len(train)}
    for name, rows in (("val", val), ("test", test)):
        # Silver labels measure agreement with the rule set, not correctness.
        pred = model.predict([encode(r) for r in rows])
        report[f"{name}_silver_macro_f1"] = round(
            f1_score([r["silver_label"] for r in rows], pred, average="macro", labels=LABELS,
                     zero_division=0), 4)

        gold = [r for r in rows if r["gold_label"]]
        if gold:
            gp = model.predict([encode(r) for r in gold])
            gy = [r["gold_label"] for r in gold]
            report[f"{name}_gold_macro_f1"] = round(
                f1_score(gy, gp, average="macro", labels=LABELS, zero_division=0), 4)
            report[f"{name}_gold_n"] = len(gold)
            if name == "test":
                print("BASELINE on held-out episodes, scored against human labels\n")
                print(classification_report(gy, gp, labels=LABELS, digits=3, zero_division=0))
                report["test_gold_report"] = classification_report(
                    gy, gp, labels=LABELS, digits=3, zero_division=0, output_dict=True)

    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # Predictions on the whole corpus feed the error-propagation analysis.
    allrows = train + val + test
    preds = model.predict([encode(r) for r in allrows])
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r, p in zip(allrows, preds):
            fh.write(json.dumps({
                "pair_id": r["pair_id"], "pred": p,
                "silver_label": r["silver_label"], "gold_label": r["gold_label"],
            }, ensure_ascii=False) + "\n")

    for k, v in report.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
