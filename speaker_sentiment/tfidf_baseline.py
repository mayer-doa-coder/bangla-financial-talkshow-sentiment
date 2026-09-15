"""TF-IDF + linear-SVM turn-level baseline on the current episode-disjoint split.

This is the "can a bag of words do it?" control for the BanglaBERT baseline. It
uses the same splits, the same three supervised classes and the same metric, so
the two numbers are directly comparable.

    python -X utf8 speaker_sentiment/tfidf_baseline.py

Writes runs/tfidf_baseline/metrics.json next to the BanglaBERT run.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

CLASSES = ["negative", "neutral", "positive"]
HERE = Path(__file__).resolve().parent

try:
    from normalizer import normalize as bn_normalize
except ImportError:
    def bn_normalize(t, **kw):
        return unicodedata.normalize("NFC", t)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", bn_normalize(str(text))).strip()


def load(split: str):
    rows = [json.loads(l) for l in
            (HERE / "data" / f"turns_{split}.jsonl").open(encoding="utf-8")]
    return [clean(r["text"]) for r in rows], [r["label"] for r in rows]


def main() -> int:
    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")

    # Bangla is agglutinative and the corpus is small, so character n-grams
    # carry more signal than word n-grams. Both are tried and chosen on val.
    grids = [
        ("char_wb 3-5", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ("char_wb 2-5", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("word 1-2", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ]
    heads = [("logreg C=%g" % c,
              (lambda c=c: LogisticRegression(C=c, max_iter=2000,
                                              class_weight="balanced")))
             for c in (0.5, 1, 2, 4, 8)]

    best = None
    print("\nselection on validation (test never consulted):")
    for gname, gkw in grids:
        for hname, mk in heads:
            pipe = make_pipeline(TfidfVectorizer(**gkw), mk())
            pipe.fit(Xtr, ytr)
            f1 = f1_score(yva, pipe.predict(Xva), average="macro", labels=CLASSES)
            print(f"  {gname:12s} + {hname:16s}  val macro-F1 {f1:.4f}")
            if best is None or f1 > best[0]:
                best = (f1, gname, hname, gkw, mk)

    val_f1, gname, hname, gkw, mk = best
    print(f"\nselected: {gname} + {hname} (val macro-F1 {val_f1:.4f})")

    # refit on train+val, exactly as the BanglaBERT baseline's final fit would
    pipe = make_pipeline(TfidfVectorizer(**gkw), mk())
    pipe.fit(Xtr + Xva, ytr + yva)
    pred = pipe.predict(Xte)
    test_f1 = f1_score(yte, pred, average="macro", labels=CLASSES)
    acc = sum(p == t for p, t in zip(pred, yte)) / len(yte)

    print(f"\nTEST macro-F1 {test_f1:.4f} | accuracy {acc:.4f}")
    print(classification_report(yte, pred, labels=CLASSES, digits=3, zero_division=0))

    out = HERE / "runs" / "tfidf_baseline"
    out.mkdir(parents=True, exist_ok=True)
    rep = classification_report(yte, pred, labels=CLASSES, digits=4,
                                output_dict=True, zero_division=0)
    (out / "metrics.json").write_text(json.dumps({
        "model": "TF-IDF + " + hname,
        "features": gname,
        "selected_on": "validation split (297 turns)",
        "val_macro_f1": val_f1,
        "test_macro_f1": test_f1,
        "test_accuracy": acc,
        "per_class": rep,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "split": "episode-disjoint, seed 13 (same as BanglaBERT run)",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
