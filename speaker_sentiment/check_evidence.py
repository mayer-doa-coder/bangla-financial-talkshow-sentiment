# -*- coding: utf-8 -*-
"""Are the judge's evidence quotes actually verbatim spans of the turns they cite?

The report claims evidence quotes make fabricated justifications checkable. That is
only true if the quotes really occur in the cited turn, so measure it.
"""
import json, unicodedata, re
from pathlib import Path

D = Path(r"d:/ML DATASETS/speaker_sentiment/data")
turns = {t["turn_id"]: t for t in
         (json.loads(l) for l in (D / "turns.jsonl").open(encoding="utf-8"))}
J = [json.loads(l) for l in (D / "judgements.jsonl").open(encoding="utf-8")]

def norm(s):
    s = unicodedata.normalize("NFC", str(s))
    return re.sub(r"\s+", " ", s).strip()

total = exact = elided = missing_turn = 0
misses = []
for r in J:
    for tl in r["turn_labels"]:
        q = norm(tl.get("evidence", ""))
        if not q:
            continue
        total += 1
        t = turns.get(tl["turn_id"])
        if t is None:
            missing_turn += 1
            continue
        hay = norm(t["text"])
        if q in hay:
            exact += 1
        else:
            # the packets elide the middle of very long turns; a quote may come
            # from the elided region, so check the raw text too
            if q in norm(t.get("text_raw", "")):
                exact += 1
            else:
                elided += 1
                if len(misses) < 8:
                    misses.append((tl["turn_id"], q[:60]))

print(f"evidence quotes            : {total}")
print(f"  verbatim in cited turn   : {exact}  ({exact/total:.1%})")
print(f"  not found                : {elided}  ({elided/total:.1%})")
print(f"  cited turn missing       : {missing_turn}")
if misses:
    print("\nsample of not-found quotes (expected: from the elided middle of long turns):")
    for tid, q in misses:
        print(f"  {tid}  {q}")
