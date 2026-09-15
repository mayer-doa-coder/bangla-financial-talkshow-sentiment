# Presentation data pack — build 2 (45 episodes)

Every number below is either read from a file in this repo or produced by a script in it.
The source is named on each line so a slide can be checked without re-running anything.

---

## 1. Training setup (slide 20) — was amber, now resolved

Read from `Speaker_Sentiment_BanglaBERT.ipynb` (the `CFG` cell) and
`runs/banglabert_turn/metrics.json`.

| Row | Value |
|---|---|
| Backbone | `csebuetnlp/banglabert` (ELECTRA discriminator, 110 M params) |
| Input representation | `role_turn` — sentence pair: role token + turn text |
| Max sequence length | 256 subwords (drops to 128 automatically on CPU) |
| Optimizer | AdamW, 3 parameter groups |
| Learning rate | **2e-5** encoder (decayed params), 2e-5 encoder (no-decay params), **1e-4** classifier head |
| Weight decay | 0.01 on encoder weights; 0.0 on bias and LayerNorm; 0.01 on head |
| Schedule | Linear decay with warmup — `get_linear_schedule_with_warmup` |
| Warmup | **10 % of total steps** (`warmup_ratio = 0.1`) |
| Epochs | 6 (best checkpoint restored, not the last) |
| Batch size | 16 |
| Total optimizer steps | 384 (= ceil(1009/16) = 64 steps × 6 epochs) |
| Warmup steps | 38 |
| Gradient clipping | max-norm 1.0 |
| Loss | Class-weighted cross-entropy with label smoothing, **per-sample weights** = judge confidence × pass agreement |
| Mixed precision | `torch.amp` autocast + GradScaler, enabled on CUDA only |
| Seed | 42 (modelling); 13 (split); 17 (pipeline) |
| Hardware | **NVIDIA Tesla T4** (Google Colab), `torch 2.11.0+cu128`, CUDA |
| **Best epoch** | **epoch 4** — val macro-F1 0.7039, val acc 0.7475 |

**Runtime — one honest caveat.** The notebook does not time its training loop, so there is no
recorded wall-clock for the T4 run and I will not invent one. What *is* measured:
**CPU inference over the 324 test turns took 47 s** (`predict_speakers.py`, no GPU).
If you want a real training figure on the slide, re-run the notebook — I can add wall-clock
logging first so it records itself.

---

## 2. Baselines on the current split (slide 20) — TF-IDF re-run, asterisk can go

Re-run today on the **current** episode-disjoint split by `tfidf_baseline.py`;
result saved to `runs/tfidf_baseline/metrics.json`.

| Model | Val macro-F1 | **Test macro-F1** | Test acc. | κ |
|---|---|---|---|---|
| Majority class | — | 0.236 | 0.549 | 0.000 |
| TF-IDF word 1–2 + LogReg (C=2) | 0.578 | **0.529** | 0.623 | 0.339 |
| **BanglaBERT (`role_turn`)** | **0.704** | **0.670** | **0.701** | **0.496** |

Your slide's 0.526 was essentially right — the re-run gives **0.529** on the new split.
Drop the asterisk and the "earlier split" note. Selection swept 3 feature families
(word 1–2, `char_wb` 2–5, `char_wb` 3–5) × C ∈ {0.5, 1, 2, 4, 8}, chosen on validation only.

TF-IDF per class (test): negative F1 0.686 · neutral 0.632 · **positive 0.269**.
That positive-class collapse is the headline contrast — BanglaBERT gets 0.581 there.

---

## 3. Corpus statistics (slides 4–17) — replaces 25 episodes / 1,340 turns

From `data/turns.jsonl`, `data/speakers.jsonl`, `data/judgements.jsonl`.

| Quantity | Build 1 (old slides) | **Build 2 (current)** |
|---|---|---|
| Episodes | 25 | **45** |
| Rolls contributing | 4 | **6** |
| Audio | ~16 h | **28.6 h** |
| Speaker turns | 1,989 | **2,414** |
| Transcribed words | — | **219,276** |
| Speaker–episode units | 141 | **212** (150 judgeable) |
| Judged units | 72 | **150** |
| Raw turn labels | — | **1,811** |
| **Supervised turns** | 1,340 | **1,630** |
| Illegible (excluded) | — | **181** |

**Turn roles:** guest 1,236 · host 926 · unassigned 233 · minor 19.

**Splits — episode-disjoint, seed 13:**

| Split | Turns | Episodes | negative | neutral | positive |
|---|---|---|---|---|---|
| train | 1,009 | 29 | 530 | 365 | 114 |
| val | 297 | 7 | 162 | 93 | 42 |
| test | 324 | 9 | 178 | 108 | 38 |
| **total** | **1,630** | **45** | **870** | **566** | **194** |

Positive class is **11.9 %** of supervised turns (was 10.2 % in build 1).

**Speaker units by split:** train 94 · val 23 · test 30 (147 valid units total).
**Speaker verdicts:** negative 110 · neutral 18 · positive 11 · mixed 8.

---

## 4. Results (slides 18–22)

**Turn level, test split (324 turns):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.790 | 0.697 | 0.740 | 178 |
| neutral | 0.655 | 0.722 | 0.687 | 108 |
| positive | 0.521 | 0.658 | 0.581 | 38 |
| **macro** | 0.655 | 0.692 | **0.670** | 324 |
| accuracy | | | **0.701** | 324 |

**Confusion matrix (test)** — from `presentation/confusion_matrix.csv`:

| | pred neg | pred neu | pred pos |
|---|---|---|---|
| **true negative** | 124 | **37** | 17 |
| **true neutral** | 24 | 78 | 6 |
| **true positive** | 9 | 4 | 25 |

Dominant error: 37 negative turns read as neutral — understated criticism in panel register.

**Training curve** (`runs/banglabert_turn/history.csv`):

| Epoch | Train loss | Val macro-F1 | Val acc |
|---|---|---|---|
| 1 | 0.7036 | 0.4565 | 0.6229 |
| 2 | 0.6148 | 0.6488 | 0.6801 |
| 3 | 0.4811 | 0.6599 | 0.7273 |
| **4** | **0.4028** | **0.7039** | **0.7475** |
| 5 | 0.3413 | 0.6777 | 0.7340 |
| 6 | 0.3168 | 0.6762 | 0.7239 |

**Speaker level.** Two routes, and the report headlines the second:

- Held-out 30 test units, best rule `hard_vote` (selected on train+val): exact accuracy **0.867**, stance r 0.451. This is a *sanity check* — 30 units is small.
- **Hierarchical model** (frozen turn encoder + attention pooling), out-of-fold over **all 147 valid units**, 5-fold GroupKFold by episode: **accuracy 0.653, macro-F1 0.530** (folds 0.500–0.828). ← quote this one.

**Breakdowns** (`presentation/breakdown_role.csv`, `breakdown_quality.csv`):

| Cut | n | Accuracy | Macro-F1 |
|---|---|---|---|
| guest turns | 184 | 0.696 | 0.610 |
| host turns | 140 | 0.707 | 0.468 |
| ASR Q1 (worst) | 83 | 0.675 | 0.651 |
| ASR Q4 (best) | 80 | 0.738 | 0.729 |

Host macro-F1 is much lower than guest despite similar accuracy — hosts are overwhelmingly
neutral, so the model gets them right for the wrong reason.

---

## 5. Demo (if you show it live)

```
python -X utf8 speaker_sentiment/predict_speakers.py \
    --model speaker_sentiment/runs/banglabert_turn/model \
    --turns speaker_sentiment/data/turns_test.jsonl \
    --out   speaker_sentiment/runs/banglabert_turn/demo_speakers.jsonl
```

Verified: runs on CPU in 47 s, recovers all 30 test speaker units, and reproduces
**all 324 turn predictions identically** to the notebook's saved output.

---

## 6. The one slide claim to be careful with

All scores measure **agreement with the LLM judge, not with human ground truth**.
The annotation drafts (`*/results/annotation_drafts/`, 42 files, 4,886 utterances) all still
carry `human_verified: false`. Human work went into curation — rubric design, diarization
repair, unit screening, out-of-domain exclusion, discarding stale blocks — not per-turn
re-labelling. Report Section 5.5 itemises this with counts.
