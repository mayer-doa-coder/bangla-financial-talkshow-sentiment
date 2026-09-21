# Speaker-level sentiment for Bangla financial talk shows

A pipeline that turns the fused ASR + diarization bundles in this repo into a speaker-level
sentiment corpus, labels it with an LLM judge, and fine-tunes BanglaBERT on the result.

```
build_speaker_corpus.py   fused/*.json  ->  turns.jsonl + speakers.jsonl
llm_judge.py              speakers.jsonl -> judgements.jsonl        (Claude Opus 5)
make_datasets.py          judgements     -> turns_{train,val,test}.jsonl + speakers_*
Speaker_Sentiment_BanglaBERT.ipynb       -> fine-tuned model + out-of-fold speaker predictions
speaker_level_eval.py     out-of-fold preds -> Task B metrics, baselines, breakdowns
make_speaker_figs.py      Task B metrics -> confusion matrix + system ladder figures
make_sample_figure.py     released files -> the worked-example figure (HTML, screenshot it)
predict_speakers.py       turns.jsonl + model -> speaker verdicts   (no notebook needed)
```

### Two tasks, not one

The release supports two supervised tasks, and they are evaluated separately because they are
different problems.

| | Task A | Task B |
|---|---|---|
| Question | what does this turn express? | where does this speaker stand in the whole episode? |
| Unit | one speaker turn | one speaker in one episode |
| Classes | negative / neutral / positive | negative / mixed / neutral / positive, plus a stance score |
| Size | 1,630 supervised turns | 147 speaker units |
| Protocol | episode-disjoint 1,009 / 297 / 324 split | 5-fold `GroupKFold` over all 147 units |
| Headline | macro-F1 **0.670** against a 0.236 floor | macro-F1 **0.530** against a 0.214 floor |

**Task B is not Task A with a summing step.** Hand an oracle the judge's own gold turn labels and
let it aggregate them perfectly: it still only recovers 85.0% of the speaker verdicts. About one
verdict in seven is simply not present in the turn labels. That gap is why the second task exists,
and `speaker_level_eval.py` measures it.

**Status: the corpus is built and labelled.** 45 episodes -> 2,414 turns -> 150 speaker units
judged -> **1,630 supervised turns** split 1,009 / 297 / 324. `data/` holds everything the
notebook needs. **No GPU is required to train**; see §7 for timings and the exact command
sequence. **Read §8 before putting any number in a report.**

| | |
|---|---|
| §1 | what is in the bundles, and the five data defects that shaped everything |
| §2 | every way to assign speaker sentiment (A–J), and 8 ways to train BERT on them |
| §3 | the judge: rubric, how the shipped labels were made, what came out |
| §4 | splits, and the two defects in them you must decide about |
| §5 | backbone choices |
| §6 | the literature this follows |
| §7 | **run it: GPU or not, and the fixed command sequence** |
| §8 | **read before reporting a number** |
| §9 | files |

---

## 1. What is actually in this repo

Six roll-number bundles (`2107001`, `2107004`, `2107006`, `2107009`, `2107010`, `2107015`),
each a run of the same Bangla talk-show pipeline: audio → VAD/hygiene → ASR → diarization →
fusion → annotation drafts.

| | count |
|---|---|
| fused episode files | 46 |
| **unique episodes** (one exact duplicate removed) | **45** |
| episodes with pyannote diarization | **45** (all of them) |
| **merged speaker turns** | **2,414** |
| words | ~219,000 |
| speaker-episode units | 212 |
| **units worth judging** (≥150 clean words) | **150** (45 hosts, 103 guests, 2 minor) |
| **turns judged** | **1,811** |
| turns that survive judging (`illegible` dropped) | **1,630** |
| **speaker verdicts usable** (`speaker_level_valid`) | **147** |
| **human sentiment labels anywhere in the repo** | **0** |

**This is the second build.** The first covered 25 episodes and 1,545 judged turns, with
`2107009` on fallback diarization and excluded from speaker-level supervision. The bundles were
then re-run: twenty more episodes arrived and `2107009` was re-diarized with pyannote. Fifty-three
of the original 150 speaker judgements survived byte-identical - same speaker, same turn ids - and
were carried forward; the other ninety-seven units were judged again from scratch against the new
segmentation. `merge_judgements.py` in the scratchpad enforced that check rather than trusting
speaker keys.

Five findings from profiling the bundles that shape every decision downstream:

**a. `2107010` duplicates `2107004`.** `2107010/Results/fused/ep006.json` is byte-identical in
transcript content to `2107004/results/fused/ep006.json`. Keeping both would put the same panel
on both sides of a train/test split. The corpus builder deduplicates by transcript hash.

**b. `2107009`'s diarization was unusable, and has since been fixed.** In the first build its five
episodes carried `"provider": "fallback"` — energy-based segmentation, not pyannote — with 60–70% of
their words on `speaker_id: -1`, so they trained the turn model but were barred from speaker-level
supervision. They have now been re-diarized with pyannote. The effect is visible in the numbers:
`2107009::ep001` went from 263 unusable fragments to 47 real turns across three identified speakers,
and every one of the 45 episodes now carries a trusted speaker id. This was the highest-value data
fix available and it has been made.

**c. The "utterances" in `fused/*.json` are not turns.** They are fixed ~28-second ASR decode
windows, cut at speaker changes. 1,300+ of the 3,532 have a duration of 27.2–28.0 s. A single
four-minute answer is split across eight rows, and **every window boundary truncates a word** —
a large fraction of windows begin with a Bengali dependent sign (া, ি, ে …), which is
orthographically impossible and means the leading consonant was eaten by the chunker.
`build_speaker_corpus.py` merges consecutive same-speaker windows back into turns and drops the
orphaned fragments.

**d. No supervision exists.** All 3,023 annotation drafts carry `labels: []`,
`annotation_status: "review_required"`, `human_verified: false`, `role_tag: ""`. The 180 labels
in `sentiment/data/gold_labels.json` are single-annotator, target-level (not speaker-level), and
their own header says they need a second annotator before being called gold. This is *why*
LLM-as-judge is the right move here, not merely a convenient one.

**e. ASR quality was never measured.** `submission_gates.json` reports
`wer_measured: SKIPPED — requires gold transcript`, and the same for DER and annotation κ. Every
number this pipeline produces is bounded by an unmeasured WER. See §8.

**Role structure falls out of the data cleanly.** Hosts take many short turns (mean 31 words),
guests take few long ones (mean 97 words). Ranking speakers by *turn share minus word share*
identifies exactly one host in each of the 20 pyannote episodes. Roles are needed because a
host asking "why has the crisis not lifted?" is not expressing a negative stance — it is the
single largest source of label confusion in this genre.

---

## 2. Every way to assign speaker-based sentiment

These are not mutually exclusive. The pipeline here uses B to create the labels and A to apply
them, and the notebook measures F against that combination.

| | Method | How | Fits this data? |
|---|---|---|---|
| **A** | **Bottom-up aggregation** | Label each turn, then combine: majority vote, net polarity `(pos−neg)/(pos+neg)`, word-weighted, duration-weighted, confidence-weighted, or top-k most-opinionated turns | **Yes — the backbone.** Turns 147 speaker labels into 1,630 usable training rows. Notebook §8 scores all six rules. |
| **B** | **Top-down holistic LLM verdict** | Judge reads the speaker's whole script, returns one verdict + a stance score | **Yes — the reference label.** Catches arc and irony that per-turn labelling misses. |
| **C** | **Lexicon / rule scoring** | Polarity cues × target orientation, aggregated per speaker | Already built in `sentiment/weak_label.py`. Useful as a **transparent baseline and as a judge cross-check**, not as the primary label. |
| **D** | **Target-aware (ABSA), then aggregate** | Per (speaker × financial target) stance → a speaker stance *matrix* | **The most useful output** for a financial index. `sentiment/build_corpus.py` already produces the pairs. Combine with this pipeline as a follow-up. |
| **E** | **Role-conditioned** | Model host and guest separately, or condition on role | **Used here** (`role_turn` input mode). Hosts are ~80% neutral; guests carry the evaluation. |
| **F** | **Hierarchical neural** | Turn encoder → attention pooling over the speaker's turns → speaker classifier, trained end-to-end | Implemented in notebook §9, **on a frozen encoder with grouped CV** — 147 units will overfit anything larger. |
| **G** | **Acoustic / prosodic** | wav2vec2 or WavLM on the speaker's audio segments | **Available and unused.** The WAVs are in `results/processed/`. Prosody carries stance that a transcript this noisy destroys. |
| **H** | **Multimodal fusion** | Text branch + audio branch, late or cross-attention fusion | The natural extension of G. Strongest published results on conversational sentiment are multimodal. |
| **I** | **Speaker-turn graph** | Utterances as nodes, edges for sequence and speaker identity (DialogueGCN-style) | Standard in the ERC literature; needs more conversations than 20 to pay off. |
| **J** | **Direct long-document classification** | Feed the whole speaker script to a long-context model | Tempting but wrong here: 147 labelled documents cannot fine-tune a 110M encoder. Viable only as zero/few-shot LLM inference. |

### Ways to train BERT on top of these

1. **Turn-level classification, aggregate at inference** — the notebook's main path.
2. **Sentence-pair `(role, turn)`** — makes the model role-aware without a new vocabulary.
3. **Dialogue-context pair `(previous turn, turn)`** — supplies the question a guest answers.
4. **Target-aware pair `(target surface form, turn)`** — the existing `sentiment/` approach; run
   one utterance once per target so one sentence can be negative on inflation and positive on
   reserves simultaneously.
5. **Hierarchical pooling head** on frozen turn embeddings — notebook §9.
6. **Two-stage: silver then gold** — fine-tune on all judged turns, then a low-LR pass on the
   human-verified subset. Best use of a small gold set.
7. **Noise-robust objectives** — label smoothing, per-sample weights from judge confidence,
   co-teaching, or confident learning to prune suspect labels. The notebook uses the first two.
8. **Domain-adaptive pretraining** — continue masked-LM on unlabelled Bangla financial text
   before fine-tuning. Usually the largest single gain in a low-resource domain shift.

---

## 3. The LLM-as-judge design

One call per speaker, returning **both levels at once**. This is the central design choice:

- A speaker verdict alone gives 147 labelled examples — not enough to fine-tune BanglaBERT.
- Turn labels alone lose the thing we want to predict.
- Labelling turns in isolation and aggregating afterwards produces turn labels that contradict
  the speaker verdict.

Judging both in one call, with the full script in context, gives 1,630 training rows whose labels
were assigned by a judge that could see how the speaker's argument develops.

**`--scope all` mattered more in the first build than it does now.** A turn's stance does not depend
on knowing *who* said it — only the speaker-level task needs correct attribution — so the flag let
the judge label fallback-diarized and unassigned-speaker buckets that were barred from Task B, and
it added 69% more training rows. Now that every episode is pyannote-diarized, 147 of the 150 judged
units are speaker-level valid and the flag adds little. It is still there and still correct; it
simply has less work to do.

**The rubric** (in `llm_judge.py`) is built around the failure modes specific to this genre:

- **Evaluative stance, not emotion.** A calm economist describing a banking collapse is negative.
- **Direction words are not polar.** "বেড়েছে" (increased) is good for reserves, bad for inflation.
  The judge must decide from *what* is rising. This is the rule the rule-based labeller in
  `sentiment/weak_label.py` already encodes via target orientation, restated for the LLM.
- **A pointed question is neutral.** Hosts elicit; they do not assert.
- **A recommendation is neutral; a complaint is negative.** "The government must raise the
  allocation" ≠ "The government failed to raise the allocation".
- **Reported speech is not the speaker's stance.**
- **`illegible` is a first-class label** for turns the ASR destroyed, so the judge declines
  instead of guessing. Those turns are dropped from supervision rather than labelled.
- **`neutral` is a real class, not a dustbin** — most host turns genuinely belong there.

**Outputs per speaker:** verdict (negative/mixed/neutral/positive), continuous `stance_score`
∈ [−1, 1], confidence, a 2–3 sentence summary, a rationale, dominant topics, a transcript-quality
rating, and for each turn a label + confidence + a verbatim evidence quote.

**Bias controls, following current LLM-judge practice:**

- The rubric is a cached system prompt, byte-identical across all calls — no per-call drift, and
  prompt caching makes repeat passes cheap.
- Structured outputs (`output_config.format` with a JSON schema) so parsing never fails silently.
- `--passes N` runs the whole corpus N times; `make_datasets.py` resolves by majority and reports
  the agreement rate. **Run with `--passes 3` before reporting anything** — single-pass
  self-consistency is unmeasured, and the agreement rate is a number your report needs.
- Each turn label carries an evidence quote naming what it is based on. **Measured: 58% of these
  occur verbatim in the turn they cite; the other 42% are close paraphrases of it** (both judging
  rounds behave the same, 59.4% and 56.9%). They localise a label to a specific turn and claim,
  which is what makes spot-checking fast, but they are *not* exact-match provenance and should not
  be used as string keys. `check_evidence.py` in the scratchpad reproduces the measurement.
- A second judge from a different model family is the standard next control if agreement is low.

Cost: ~950k prompt characters for 150 units, roughly $4–6 per pass at Opus 5 list rates. The rubric is a cached prefix, so passes 2 and 3 are cheaper than pass 1.

```bash
export ANTHROPIC_API_KEY=...          # or: ant auth login
python speaker_sentiment/llm_judge.py --dry-run --scope all     # inspect prompts, no spend
python speaker_sentiment/llm_judge.py --scope all --passes 3    # resumable; skips completed work
python speaker_sentiment/make_datasets.py --group-by episode --test-episodes 5 --val-episodes 4 --seed 7
```

### How the shipped `judgements.jsonl` was actually produced

**It was not produced by running `llm_judge.py`.** The file in `data/` was written by Claude Opus 5
reading every transcript directly inside a Claude Code session — the same model, the same rubric,
the same output fields, but one pass, no API key, no cost. `judge_model` is
`claude-opus-5 (in-session, single pass)` on every row, so it is distinguishable from API output.

Two consequences you must carry into any write-up:

- **`pass_agreement` is 1.0 everywhere by construction.** There is no self-consistency measurement
  yet. Running `llm_judge.py --scope all --passes 3` gives you a real agreement rate; `make_datasets.py`
  will merge those passes with the existing rows and compute it.
- The per-turn evidence quotes localise each label to a specific claim, so the labels are spot-checkable even at one pass — with the 58% caveat above.

### What the judge produced

| | |
|---|---|
| speaker units judged | 150 (147 with `speaker_level_valid: true`) |
| turn labels | 1,811 |
|  ↳ negative | 870 (48.0%) |
|  ↳ neutral | 566 (31.3%) |
|  ↳ positive | 194 (10.7%) |
|  ↳ `illegible` (dropped from supervision) | 181 (10.0%) |
| speaker verdicts | negative 110, neutral 18, positive 11, mixed 8 |

On the 1,630 supervised turns: **negative 53.4%, neutral 34.7%, positive 11.9%.**

**The role split confirms the rubric is doing what it was written to do:**

| role | negative | neutral | positive |
|---|---|---|---|
| guest | 602 | 192 | 171 |
| host | 266 | **374** | 23 |
| minor | 2 | 0 | 0 |

Hosts land majority-neutral because the rubric treats a pointed question as elicitation, not
assertion. Guests carry the evaluation. If a future judging pass loses that separation, the rubric
has drifted.

**The corpus is internally consistent.** Rebuilding each speaker verdict from its own turn labels,
without letting the aggregator see the holistic verdict:

| aggregation rule | exact match | polar direction |
|---|---|---|
| `word_weighted` | **85.0%** | 86.4% |
| `word_conf_weighted` | **85.0%** | 86.4% |
| `majority_vote` | 83.7% | **87.1%** |
| `net_polarity` | 83.7% | **87.1%** |
| `conf_weighted` | 83.7% | **87.1%** |

Bottom-up and top-down labelling now agree on between five and six speakers in seven, up from four
in five on the first build — the larger corpus made the judging more internally consistent, not less.
This is the evidence that the two levels were judged coherently, and the number to beat when a
trained model does the same job. The residual 15% is mostly `mixed`: a rule that counts turns cannot
reproduce "credits the government with stabilising the banks but calls reform a lost opportunity".

---

## 4. Splits, and the corpus that came out of them

By **episode**, never by row. Two turns from one talk show share the panel, the topic, the
presenter's phrasing and the same ASR error signature; a random row split lets the model recognise
the episode rather than the stance. `make_datasets.py` asserts episode disjointness.

The shipped split is `--group-by episode --test-episodes 9 --val-episodes 7 --seed 13`:

| split | episodes | speakers | turns | negative | neutral | positive |
|---|---|---|---|---|---|---|
| train | 29 | 94 | **1,009** | 530 | 365 | 114 |
| val | 7 | 23 | 297 | 162 | 93 | 42 |
| test | 9 | 30 | 324 | 178 | 108 | 38 |

Speaker-level, counting only `speaker_level_valid` units:

| split | valid speakers | negative | neutral | positive | mixed |
|---|---|---|---|---|---|
| train | 94 | 70 | 13 | 8 | 3 |
| val | 23 | 16 | 3 | 2 | 2 |
| test | 30 | 24 | 2 | 1 | 3 |

**The seed is not arbitrary.** Seven seeds were compared for class balance before one was chosen;
seed 13 is the only one that puts all four speaker classes and a workable share of positive turns
in every split. On the first build the validation split held six speakers across two classes, which
made speaker-level validation meaningless. That problem is gone.

**Two defects in this split you have to decide about before reporting anything.**

**a. `--group-by programme` does not work on this metadata.** The intent was right — the same
presenter on both sides of a split is leakage even under an episode split — but 17 of the 25
episodes carry the placeholder programme title `RTV Business Talk` in `registry.csv`, across three
different rolls with visibly different shows. Grouping by it collapses the corpus (train = 1
episode, test = 20). The flag is still there and still correct; the *metadata* is wrong. Fixing the
programme titles in the registries is the second-highest-value data fix after re-running pyannote
on `2107009`. Until then, `make_datasets.py` prints
`WARNING: val/test share programmes with train: ['business talk']` and you accept it knowingly.

**b. The speaker level is still thin, though no longer degenerate.** Thirty test speakers across
four classes means one unit moves macro-F1 by about 0.03 - better than the 0.08 of the first build,
but still not a stable measurement. Notebook §9's `GroupKFold` over all 147 valid units remains the
number to quote for anything speaker-level; the held-out split is best read as a sanity check.
`positive` and `mixed` are the classes to watch: eleven and eight units respectively in the whole
corpus.

---

## 5. Model choices

| Backbone | Why consider it |
|---|---|
| **`csebuetnlp/banglabert`** *(default)* | ELECTRA discriminator, 110M, ~27GB Bangla pretraining. Strongest Bangla-specific encoder on published benchmarks; its advantage over XLM-R is largest in low-sample regimes, which is exactly this setting. |
| `csebuetnlp/banglishbert` | Bangla + English pretraining. Worth testing: these transcripts are heavily code-mixed (*inflation*, *reserve*, *GDP* appear inline) and BanglaBERT's tokenizer shatters them into subwords. |
| `xlm-roberta-base` | Multilingual baseline, 270M. Report it so the Bangla-specific gain is quantified. |
| `google/muril-base-cased` | 17 Indian languages, transliteration-aware. |

The notebook's §10 sweeps backbone × input-format. **Select on validation, report on test.**

**Why no `Trainer`:** the loop is hand-written so the per-sample weighting (judge confidence ×
pass agreement) and the class weighting compose cleanly, and so the notebook does not break on
the next `transformers` major version.

---

## 6. Literature this follows

**Bangla sentiment / encoders**
- Bhattacharjee et al., *BanglaBERT* (NAACL 2022 Findings) — the backbone, and the
  BLUB benchmark: [arXiv:2101.00204](https://arxiv.org/abs/2101.00204)
- BLP-2023 Task 2 (Bangla social-media sentiment) system papers — transformer benchmarking and
  ensembling for Bangla sentiment:
  [BanglaNLP](https://arxiv.org/pdf/2310.09238),
  [RSM-NLP](https://arxiv.org/pdf/2310.14261)
- [BABSA](https://www.sciencedirect.com/science/article/pii/S2352340926001575) — 15,860-instance
  Bangla ABSA dataset (2026), and
  [BAN-ABSA](https://link.springer.com/chapter/10.1007/978-981-16-0586-4_31) — relevant if you
  extend to method D (speaker × target)
- [Aspect-based sentiment analysis datasets for Bangla text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11617299/)
- [TigerLLM — a family of Bangla LLMs](https://aclanthology.org/2025.acl-short.69.pdf) (ACL 2025)
  — if you want a Bangla-native judge as a second opinion
- [Negation-aware Bengali sentiment modelling](https://link.springer.com/chapter/10.1007/978-3-032-27448-9_3)
  — Bangla negation is postposed, which both the rule labeller and the judge rubric handle

**Speaker-level and conversational sentiment**
- [Hierarchical Transformer with Speaker Modeling for ERC](https://arxiv.org/pdf/2012.14781) —
  the architecture behind notebook §9
- [Deep Emotion Recognition in Textual Conversations: a survey](https://arxiv.org/pdf/2211.09172)
- [declare-lab ERC reading list](https://github.com/declare-lab/awesome-emotion-recognition-in-conversations)
- [Which Voices Move Markets? Speaker Identity and the Cross-Section of Post-Earnings Returns](https://arxiv.org/html/2604.13260)
  — the direct financial analogue: treating a transcript as one document and averaging over all
  speakers discards the speaker's informational role. Same argument, different genre.
- [Advanced deep learning for earnings call transcripts](https://arxiv.org/pdf/2503.01886) —
  hierarchical discourse-tree encodings of multi-speaker financial talk

**LLM-as-judge and weak supervision**
- [Large Language Models for Data Annotation and Synthesis: a survey](https://arxiv.org/pdf/2402.13446)
- [Time to Impeach LLM-as-a-Judge: Programs are the Future of Evaluation](https://arxiv.org/pdf/2506.10403)
  — the case for distilling judge logic into checkable programs; relevant to reconciling the
  judge with the existing rule labeller
- Standard bias controls: fixed rubric, multi-pass self-consistency, cross-family second judge

**Sentiment on noisy ASR** — the constraint that dominates this project
- [ASR-GLUE: a benchmark for ASR-robust NLU](https://arxiv.org/pdf/2108.13048)
- [Speech Emotion Recognition with ASR Transcripts: WER and fusion techniques](https://arxiv.org/pdf/2406.08353)
  — finds ~12% WER has minimal impact on sentiment; degradation is steep well above that, which
  is why measuring WER here is not optional
- [Confusion2vec](https://arxiv.org/pdf/1904.03576) /
  [Confusion2vec 2.0](https://arxiv.org/pdf/2102.02270) — representations that survive ASR
  confusability
- [ASR-robust SLU with word confusion networks](https://arxiv.org/pdf/2401.02921) — use the ASR
  lattice rather than the 1-best string; your `asr/*.json` files retain per-word probabilities,
  so this is actually available to you

---

## 7. Running it: what needs a GPU and what does not

**Answer: no GPU is strictly required, and the corpus is small enough that a CPU works.** Measured
on this machine (`torch 2.12.1+cpu`, 10 threads, 886 train turns, batch 16):

| | s/step | per epoch | 6 epochs |
|---|---|---|---|
| `max_len=128`, CPU | 5.3 | 4.9 min | **~30 min** |
| `max_len=256`, CPU | 10.7 | 10.0 min | **~60 min** |
| `max_len=256`, Colab T4 *(estimate, not measured here)* | ~0.35 | ~20 s | **~3 min** |

So: the **main path (§§1–8) runs locally in about an hour**, and that is a legitimate way to get
your first model. A GPU stops mattering for one run and starts mattering for everything you have to
run *repeatedly*:

| notebook section | CPU | why |
|---|---|---|
| §§1–7 turn-level fine-tune (one run) | ~60 min | fine |
| §8 aggregation | seconds | fine |
| §9 hierarchical GroupKFold (5 folds) | 5× the embedding pass | tolerable |
| §10 backbone × input ablation (8 configs) | **~8 hours** | **use a GPU** |
| seed-averaging (3–5 seeds — caveat 5 in §8) | **3–5 hours** | **use a GPU** |
| `xlm-roberta-base` (270M) anywhere | ~2.5× BanglaBERT | **use a GPU** |

### Fixed sequence — run exactly this

**Steps 1 and 2 are already done.** `data/judgements.jsonl` and all six split files are written.
Re-run them only if you change the judging or the split; otherwise go straight to step 3.

```bash
# 1. rebuild the corpus from the fused bundles          (CPU, ~20 s)
python -X utf8 speaker_sentiment/build_speaker_corpus.py

# 2. rebuild the splits from the judgements             (CPU, ~5 s)
python -X utf8 speaker_sentiment/make_datasets.py \
       --group-by episode --test-episodes 5 --val-episodes 4 --seed 7

# 3. train
jupyter lab speaker_sentiment/Speaker_Sentiment_BanglaBERT.ipynb

# 4. later: score a new episode without opening Jupyter    (CPU, minutes)
python -X utf8 speaker_sentiment/predict_speakers.py        --model speaker_sentiment/runs/banglabert_turn/model        --turns speaker_sentiment/data/turns.jsonl
```

`predict_speakers.py` reads `input_mode`, `max_len` and the winning aggregation rule out of the
`metrics.json` the notebook wrote, so the CLI cannot silently disagree with the model that
produced it.

On Windows, `python -X utf8` is not optional — without it the console encoder dies on Bangla.

**Locally (CPU):** open the notebook, set `CFG.max_len = 128` in §1, run §§1–8 top to bottom.
Roughly an hour. Leave `CFG.epochs = 6`.

**On Colab (recommended for anything beyond one run):**

1. Runtime → Change runtime type → **T4 GPU**.
2. Upload the whole `speaker_sentiment/` folder to Drive, or just `data/` — the notebook needs
   nothing else.
3. Mount Drive and set `CFG.data_dir = Path("/content/drive/MyDrive/speaker_sentiment/data")` in §1.
   That is the **only** line you change.
4. Run §1 (it pip-installs `transformers`, `torch`, `scikit-learn`, `pandas`, `matplotlib` and
   prints the device — confirm it says `cuda`), then run everything.
5. §12 writes the fine-tuned model to `CFG.out_dir`; download that folder before the runtime recycles.

**What you decide, not the notebook:**

- `CFG.max_len` — measured turn lengths in BanglaBERT subwords: p50 = 46, p75 = 120, p90 = 254,
  p95 = 351, max = 1,265. So **`max_len=256` truncates ~10% of turns and `max_len=128` truncates 23%**.
  128 halves the cost and cuts the tail off long guest answers; stance is usually stated early, so
  the loss is small but real. §4 of the notebook prints this distribution for your own data.
- **Install the csebuetnlp normalizer** (`pip install git+https://github.com/csebuetnlp/normalizer`).
  BanglaBERT was pretrained on normalized text; without it the notebook falls back to bare NFC and
  prints a warning. It is worth a couple of F1 points on its own.
- `CFG.input_mode` — `role_turn` is the default. §10 measures whether `context` or `role_context` beats
  it. Do not pick by intuition; hosts and guests behave differently enough that this matters.
- Class weighting is **on** by default (positive is 10% of the data). Turning it off raises accuracy
  and destroys positive recall, which is the class a financial sentiment index most needs.

---

## 8. Read this before reporting a number

1. **Every score measures agreement with an LLM judge, not with truth.** A test macro-F1 of 0.80
   means the student model reproduces the judge 80% of the time. Notebook §11 draws a stratified
   180-turn sample for hand-labelling, with machine labels withheld from the annotator, and
   computes judge-vs-human alongside model-vs-human. Do this before the write-up.
2. **WER is unmeasured.** Hand-transcribe ~10 minutes per programme and measure it. The
   literature puts the comfortable ceiling near 12% WER for sentiment; this ASR is visibly worse
   than that (invented words, dropped endings, wrong numbers). Report the number either way.
3. **DER is unmeasured**, so speaker attribution error is unknown even on the pyannote episodes.
4. **Single annotator, no κ.** The existing 180-pair gold set says so itself. A second annotator
   on the §11 sample gives you Cohen's κ and tells you whether the label scheme is even reliable.
5. **147 speakers and 1,630 turns is still small.** Report the hierarchical model's grouped-CV score,
   not a single held-out split — one unlucky episode moves it by ten points. At this size,
   report a seed-averaged score (3–5 seeds) for the turn-level model too.
6. **The judge and the student share a blind spot.** If the judge over-reads negativity in
   garbled text, the student learns to do the same, and no evaluation against judge labels will
   show it. Only §11 will.
7. **The shipped labels are a single judging pass.** See §3. Run `--passes 3` before quoting any
   number as final.
8. **One episode is not a financial talk show at all.** `2107015::ep002` is registered as
   `RTV Business Talk` but is an interview with a police DIG about law and order. Its speakers are
   judged almost entirely `illegible` with `OUT OF DOMAIN` in the summary, so it contributes 3 turns
   instead of ~80. Check the other registry titles before trusting any programme-level claim.
9. **The corpus is negative-dominant, and that is partly real and partly selection.** 53% of turns
   and 110 of 150 speaker verdicts are negative. Bangladeshi financial talk shows in this period were
   discussing a banking crisis, a reserve crisis and inflation, so a genuinely negative prior is
   expected — but a judge that sees mostly-negative episodes can also drift. Notebook §11's human
   sample is what separates those two explanations. Report the class prior alongside every F1.

## 9. Files

```
speaker_sentiment/
  build_speaker_corpus.py    dedupe, merge windows into turns, infer roles, quality flags
  llm_judge.py               Claude Opus 5 judge: speaker verdict + per-turn labels
  make_datasets.py           merge, aggregate 5 ways, group-disjoint splits, leakage check
  predict_speakers.py        score a turns file with a trained model -> speaker verdicts (CLI)
  tfidf_baseline.py          Task A bag-of-words control on the same split
  speaker_level_eval.py      Task B: metrics, baseline ladder, confusion, breakdowns
  make_speaker_figs.py       Task B figures (confusion matrix, system ladder)
  make_sample_figure.py      worked-example figure as HTML, for screenshotting
  check_evidence.py          measures how many judge evidence quotes are verbatim
  Speaker_Sentiment_BanglaBERT.ipynb
  data/
    turns.jsonl              2,414 speaker turns
    speakers.jsonl           212 speaker units (150 judgeable)
    judgements.jsonl         150 speaker blocks, 1,811 turn labels   [shipped]
    turns_labelled.jsonl     1,630 supervised turns (illegible dropped)
    speakers_labelled.jsonl  150 units with 5 aggregation rules attached
    turns_{train,val,test}.jsonl      1,009 / 297 / 324
    speakers_{train,val,test}.jsonl      94 /  23 /  30
  runs/
    banglabert_turn/         Task A: metrics.json, history.csv, test_predictions.jsonl, model/
      presentation/          Task A figures and breakdowns, plus speaker_level_oof.csv
    tfidf_baseline/          Task A control
    speaker_level/           Task B: metrics.json, system_comparison.csv, confusion_matrix.csv,
                             per_class.csv, breakdown_*.csv, errors.csv, two PNG figures
    sample/                  sample_figure.html, the worked example
```

### Regenerating the Task B results and the report figures

```bash
python -X utf8 speaker_sentiment/speaker_level_eval.py    # ~15 s, CPU, no model needed
python -X utf8 speaker_sentiment/make_speaker_figs.py     # two PNGs
python -X utf8 speaker_sentiment/make_sample_figure.py    # HTML; screenshot at 2800px wide
```

`speaker_level_eval.py` reads the out-of-fold predictions the notebook saved, so it does not
need a GPU or the checkpoint. The sample figure is HTML rather than a plot because Bengali needs
complex-script shaping that matplotlib does not do; open it in a browser and capture it.
