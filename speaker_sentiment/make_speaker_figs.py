"""Render the Task B figures: speaker-level confusion matrix and system ladder.

Kept separate from speaker_level_eval.py so the numbers can be recomputed on a
machine without matplotlib. Reads only what that script wrote.

    python -X utf8 speaker_sentiment/make_speaker_figs.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "speaker_level"
CLASSES = ["negative", "mixed", "neutral", "positive"]


def read_csv(name: str) -> list[dict]:
    with (RUN / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def confusion_figure() -> None:
    rows = read_csv("confusion_matrix.csv")
    counts = np.array([[int(row[name]) for name in CLASSES] for row in rows], dtype=float)
    support = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, support, out=np.zeros_like(counts), where=support > 0)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    fig.subplots_adjust(wspace=0.35)
    for index, (axis, matrix, title, fmt) in enumerate((
        (axes[0], counts, "Counts", "{:.0f}"),
        (axes[1], normalised, "Row-normalised (recall)", "{:.2f}"),
    )):
        image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right", fontsize=9)
        axis.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
        axis.set_xlabel("Model verdict", fontsize=9)
        # Only the left panel carries the y-label; on the right it would sit
        # between the two panels and collide with the left one's cells.
        if index == 0:
            axis.set_ylabel("Judge verdict", fontsize=9)
        axis.set_title(title, fontsize=10)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                axis.text(
                    j, i, fmt.format(matrix[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if normalised[i, j] > 0.55 else "black",
                )
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04)
    out = RUN / "confusion_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def ladder_figure() -> None:
    rows = read_csv("system_comparison.csv")
    labels = {
        "majority_class": "Always predict\nthe common class",
        "tfidf_speaker_script": "TF-IDF over the\nspeaker's script",
        "hierarchical_banglabert": "Hierarchical\nBanglaBERT",
        "oracle_aggregated_gold_turns": "Oracle: aggregate the\njudge's own turn labels",
    }
    order = ["majority_class", "tfidf_speaker_script", "hierarchical_banglabert",
             "oracle_aggregated_gold_turns"]
    rows = sorted(rows, key=lambda row: order.index(row["system"]))
    names = [labels[row["system"]] for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    macro = [float(row["macro_f1"]) for row in rows]

    positions = np.arange(len(rows))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9.0, 3.9))
    bars_a = axis.bar(positions - width / 2, accuracy, width,
                      label="Accuracy", color="#9DB8D6", edgecolor="#3E6FA8")
    bars_b = axis.bar(positions + width / 2, macro, width,
                      label="Macro-F1", color="#A8CBB0", edgecolor="#3C7D52")
    # The oracle is a ceiling, not a competitor; hatch it so it never reads as
    # another model on the ladder.
    bars_a[-1].set_hatch("//")
    bars_b[-1].set_hatch("//")

    for bars in (bars_a, bars_b):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                      f"{bar.get_height():.3f}", ha="center", fontsize=8)

    axis.set_xticks(positions, names, fontsize=8.5)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Score over 147 speaker units", fontsize=9)
    axis.legend(fontsize=9, frameon=False, ncol=2, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    out = RUN / "system_ladder.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    confusion_figure()
    ladder_figure()
    summary = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    print("headline:", summary["accuracy"], summary["macro_f1"])
