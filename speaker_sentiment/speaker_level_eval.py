"""Task B evaluation: one sentiment verdict per speaker per whole episode.

Task A asks what a single speaker turn expresses. Task B asks the question the
dataset was actually built for: across everything this person said in this
episode, what position did they hold? The two are not the same problem. A
panellist can grant a point in one turn and still be negative overall, and a
host can sound harsh while holding no position at all.

This script produces the Task B result set that mirrors what Task A already
reports, from artifacts that exist in the repository:

  * the hierarchical speaker model's out-of-fold predictions over all 147
    speaker units (5-fold, episode-disjoint), which is the headline;
  * a majority-class floor and a TF-IDF speaker-script control trained and
    scored under that same fold assignment, so the comparison is like for
    like;
  * confusion matrix, per-class scores, and breakdowns by role, transcript
    quality and how much the speaker actually said.

Run:
    python -X utf8 speaker_sentiment/speaker_level_eval.py

Every score is agreement with the LLM judge's holistic verdict, not with human
ground truth. The same caveat that governs Task A governs this.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OOF_PATH = ROOT / "runs" / "banglabert_turn" / "presentation" / "speaker_level_oof.csv"
OUT_DIR = ROOT / "runs" / "speaker_level"

SPEAKER_CLASSES = ["negative", "mixed", "neutral", "positive"]
N_FOLDS = 5
SEED = 13

# A verdict's polar direction: whether the speaker came down against, for, or
# neither. "mixed" holds both directions at once, so it is its own case and is
# never folded into neutral.
DIRECTION = {"negative": "negative", "positive": "positive",
             "neutral": "neither", "mixed": "both"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def turn_bucket(n_turns: int) -> str:
    if n_turns <= 4:
        return "1-4 turns"
    if n_turns <= 12:
        return "5-12 turns"
    return "13+ turns"


def confusion(y_true: list[str], y_pred: list[str]) -> list[dict]:
    rows = []
    for actual in SPEAKER_CLASSES:
        row: dict[str, object] = {"judge": actual}
        subset = [p for t, p in zip(y_true, y_pred) if t == actual]
        for predicted in SPEAKER_CLASSES:
            row[predicted] = sum(1 for p in subset if p == predicted)
        row["support"] = len(subset)
        rows.append(row)
    return rows


def per_class(y_true: list[str], y_pred: list[str]) -> list[dict]:
    report = classification_report(
        y_true, y_pred, labels=SPEAKER_CLASSES, output_dict=True, zero_division=0
    )
    rows = []
    for name in SPEAKER_CLASSES:
        entry = report[name]
        rows.append(
            {
                "class": name,
                "support": int(entry["support"]),
                "precision": round(entry["precision"], 4),
                "recall": round(entry["recall"], 4),
                "f1": round(entry["f1-score"], 4),
            }
        )
    return rows


def breakdown(units: list[dict], key: str, label: str) -> list[dict]:
    rows = []
    for value in sorted({str(unit[key]) for unit in units}):
        subset = [unit for unit in units if str(unit[key]) == value]
        y_true = [unit["true"] for unit in subset]
        y_pred = [unit["pred"] for unit in subset]
        rows.append(
            {
                label: value,
                "units": len(subset),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "macro_f1": round(macro_f1(y_true, y_pred), 4),
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    speakers = {row["speaker_key"]: row for row in read_jsonl(DATA / "speakers_labelled.jsonl")}
    with OOF_PATH.open(encoding="utf-8-sig", newline="") as handle:
        oof = list(csv.DictReader(handle))

    units: list[dict] = []
    for row in oof:
        meta = speakers.get(row["speaker_key"])
        if meta is None:
            continue
        units.append(
            {
                "speaker_key": row["speaker_key"],
                "episode": row["episode"],
                "true": row["true"],
                "pred": row["pred"],
                "role": meta.get("role", ""),
                "transcript_quality": meta.get("transcript_quality", ""),
                "n_turns": int(meta.get("n_turns") or 0),
                "n_words": int(meta.get("n_words") or 0),
                "stance_score": float(meta.get("judge_stance_score") or 0.0),
                "script": meta.get("script", "") or "",
                "turn_bucket": turn_bucket(int(meta.get("n_turns") or 0)),
            }
        )
    if len(units) != len(oof):
        print(f"warning: {len(oof) - len(units)} out-of-fold rows had no label record")

    y_true = [unit["true"] for unit in units]
    y_pred = [unit["pred"] for unit in units]
    episodes = [unit["episode"] for unit in units]

    # ---------------------------------------------------------- baselines
    # Same fold assignment for every system, so the comparison is like for
    # like. GroupKFold is deterministic given the group sequence.
    splitter = GroupKFold(n_splits=N_FOLDS)
    folds = list(splitter.split(np.zeros(len(units)), y_true, groups=episodes))

    majority_pred = [""] * len(units)
    tfidf_pred = [""] * len(units)
    for train_idx, test_idx in folds:
        train_labels = [y_true[i] for i in train_idx]
        winner = Counter(train_labels).most_common(1)[0][0]
        for i in test_idx:
            majority_pred[i] = winner

        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True
        )
        x_train = vectorizer.fit_transform([units[i]["script"] for i in train_idx])
        x_test = vectorizer.transform([units[i]["script"] for i in test_idx])
        classifier = LogisticRegression(
            C=2.0, max_iter=3000, class_weight="balanced", random_state=SEED
        )
        classifier.fit(x_train, train_labels)
        for position, i in enumerate(test_idx):
            tfidf_pred[i] = classifier.predict(x_test[position])[0]

    # The ceiling for any method that reaches a speaker verdict by combining
    # turn labels. It is handed the judge's own gold turn labels, so no
    # classifier can do better by that route. Whatever it misses is a verdict
    # the turns do not contain, and that gap is the reason Task B is its own
    # task rather than a post-processing step on Task A.
    oracle_pred = [
        str(speakers[unit["speaker_key"]].get("agg_word_weighted", "")) for unit in units
    ]

    systems = {
        "majority_class": majority_pred,
        "tfidf_speaker_script": tfidf_pred,
        "hierarchical_banglabert": y_pred,
        "oracle_aggregated_gold_turns": oracle_pred,
    }

    comparison = []
    for name, predictions in systems.items():
        direction_true = [DIRECTION[label] for label in y_true]
        direction_pred = [DIRECTION[label] for label in predictions]
        comparison.append(
            {
                "system": name,
                "units": len(units),
                "accuracy": round(accuracy_score(y_true, predictions), 4),
                "macro_f1": round(macro_f1(y_true, predictions), 4),
                "direction_accuracy": round(accuracy_score(direction_true, direction_pred), 4),
            }
        )

    # -------------------------------------------------------------- write
    headline = next(row for row in comparison if row["system"] == "hierarchical_banglabert")
    summary = {
        "task": "Task B: one verdict per speaker per whole episode",
        "protocol": f"{N_FOLDS}-fold episode-disjoint cross-validation over all "
                    f"{len(units)} speaker units; no episode appears in both sides of a fold",
        "classes": SPEAKER_CLASSES,
        "class_counts": {name: y_true.count(name) for name in SPEAKER_CLASSES},
        "n_units": len(units),
        "n_episodes": len(set(episodes)),
        "accuracy": headline["accuracy"],
        "macro_f1": headline["macro_f1"],
        "direction_accuracy": headline["direction_accuracy"],
        "systems": comparison,
        "system_notes": {
            "majority_class": "floor: always predict the training fold's most common verdict",
            "tfidf_speaker_script": "control: TF-IDF over the speaker's concatenated script",
            "hierarchical_banglabert": "the Task B model: attention pooling over turn embeddings",
            "oracle_aggregated_gold_turns": "not a model; the judge's own gold turn labels "
                                            "aggregated word-weighted, so it bounds any "
                                            "turn-aggregation route",
        },
        "label_source": "LLM-as-judge (claude-opus-5), no human labels",
        "caveat": "scores measure agreement with the LLM judge's holistic verdict, "
                  "not with human ground truth",
    }

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with (OUT_DIR / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("system_comparison.csv", comparison)
    write_csv("confusion_matrix.csv", confusion(y_true, y_pred))
    write_csv("per_class.csv", per_class(y_true, y_pred))
    write_csv("breakdown_role.csv", breakdown(units, "role", "role"))
    write_csv("breakdown_quality.csv", breakdown(units, "transcript_quality", "transcript_quality"))
    write_csv("breakdown_turns.csv", breakdown(units, "turn_bucket", "turn_bucket"))

    errors = [
        {
            "speaker_key": unit["speaker_key"],
            "role": unit["role"],
            "n_turns": unit["n_turns"],
            "n_words": unit["n_words"],
            "judge": unit["true"],
            "model": unit["pred"],
            "stance_score": unit["stance_score"],
            "transcript_quality": unit["transcript_quality"],
        }
        for unit in units
        if unit["true"] != unit["pred"]
    ]
    errors.sort(key=lambda row: -row["n_words"])
    write_csv("errors.csv", errors)

    # ------------------------------------------------------------- report
    print(f"Task B: {len(units)} speaker units over {len(set(episodes))} episodes")
    print(f"class balance: {summary['class_counts']}")
    print()
    print(f"{'system':<26}{'acc':>8}{'macroF1':>10}{'dir acc':>10}")
    for row in comparison:
        print(f"{row['system']:<26}{row['accuracy']:>8.3f}"
              f"{row['macro_f1']:>10.3f}{row['direction_accuracy']:>10.3f}")
    print()
    print("per class (hierarchical):")
    for row in per_class(y_true, y_pred):
        print(f"  {row['class']:<10} n={row['support']:<4} "
              f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}")
    print()
    print("confusion (judge rows, model columns):")
    print("  " + " " * 10 + "".join(f"{name:>10}" for name in SPEAKER_CLASSES))
    for row in confusion(y_true, y_pred):
        print(f"  {row['judge']:<10}" + "".join(f"{row[name]:>10}" for name in SPEAKER_CLASSES))
    print()
    print(f"{len(errors)} errors written; artifacts in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
