# অর্থকথা অনুসন্ধান — demonstration app

This application turns the pipeline's fused Bangla transcripts into a
speaker-attributed retrieval demonstration. It scans `datasets/` recursively,
recognises the `datasets/<roll>/<roll>_results/` layout, and presents every
speaker as `<roll>::<episode> / Speaker N`. Speaker IDs remain episode-local.

## What the GUI demonstrates

- Group totals for roll folders, source audio, result bundles, processed
  episodes, hours, utterances, ASR words and word-to-speaker coverage.
- An individual contribution table for every roll folder, including folders
  that contain source audio but do not yet contain completed results.
- A full completion map covering audio inventory, diarization, ASR, fusion,
  annotation drafts, target/sentiment artifacts, aggregation, evaluation,
  error-propagation status, and retrieval.
- Per-episode model provenance, decoding settings, diarization thresholds,
  fusion policy, annotation readiness, and automatic structural diagnostics.
- Exact word/phrase, all-word, any-word, fuzzy, semantic and hybrid search.
- Whole-corpus, roll, episode and episode-local speaker filtering.
- Ranked Bangla transcript cards with speaker, timestamp and context.
- Optional target/polarity badges when future annotation outputs exist.
- Playable audio excerpts generated around a selected result.
- CSV export for every search.
- Episode explorer with speaker airtime, transcript and confidence columns.
- Original waveform reports and a complete artifact inventory.
- Financial target taxonomy and the 28-entry entity/alias lexicon.
- Exploratory weak-labelled profiles, visibly separated from current and
  human-verified results.
- A **Speaker sentiment** tab carrying the finished corpus: per-speaker
  verdicts with stance scores, the BanglaBERT turn classifier's held-out
  scores, its confusion matrix, per-class scores, per-epoch training history
  and prediction examples. These labels come from an LLM judge, so every
  score there is agreement with that judge, not with human ground truth.
- The complete submission-gate table, including honest SKIPPED status for
  WER, DER, target/polarity F1, and error propagation when gold references
  are absent.
- A downloadable ZIP containing the durable data, reports, manifests and
  model provenance for every indexed roll.

Retrieval relevance is a ranking score, not accuracy. Speaker IDs are model
predictions, not real-person identities.

## Six-member consolidation layout

Give each member one roll-number folder. Store that member's source audio in
the roll folder and the exported pipeline bundle in `<roll>_results`. The
scanner uses the roll number as the provenance namespace.

```text
ML_Project/
├── datasets/
│   ├── 2107001/
│   │   ├── 01.mp3 ...
│   │   └── 2107001_results/
│   │       ├── artifacts/run_manifest.json
│   │       ├── data/
│   │       │   ├── raw/registry.csv
│   │       │   ├── diarization/
│   │       │   ├── asr/
│   │       │   ├── fused/
│   │       │   └── annotation_drafts/
│   │       ├── models/manifest.json
│   │       └── reports/figures/
│   ├── 2107004/
│   │   └── 2107004_results/
│   └── ...
└── demo/
```

Both export shapes are recognised, so members do not have to re-export:

```text
<roll>/<roll>_results/data/<stage>/    # canonical
<roll>/results/<stage>/                # stage folders exported directly
```

The folder name may be `results`, `Results` or `<roll>_results`, and the
`annotation_drafts` and `weak_labels` stages are also accepted as `drafts` and
`weak`.

Source audio is located by registry path first, then by a Unicode-insensitive
filename match, then by matching the registry `duration_sec` against the real
duration of the files on disk. The duration match is what allows a member to
rename or re-encode their audio before uploading — a positional `01.mp3` guess
is only used when no duration contradicts it, because the upload order does not
always follow the episode order.

If a member exported `diarization/` and `asr/` but no `fused/`, their episodes
have no transcript and cannot appear. Rebuild the missing stage from their own
durable artifacts:

```powershell
python demo/fuse_stage.py --bundle datasets/<roll>/results
```

`--verify` replays the same rule against a bundle that already has fused files;
it reproduces the existing 2107004 and 2107006 exports exactly.

Episode IDs may repeat across rolls because the application namespaces them.
For example, `2107001::ep001` and `2107006::ep001` remain distinct. A registry
may keep an older path such as `datasets/01.mp3`; the loader resolves the
basename inside the owning roll folder without modifying the stored artifact.

Do not merge speaker IDs across episodes. `2107001::ep001 / Speaker 2` and
`2107006::ep001 / Speaker 2` are unrelated labels.

Before the presentation, audit the consolidated folder and save the report:

```powershell
python demo/audit_demo.py --data-root . --report demo_readiness.json
```

Point `--data-root` at the folder that holds the roll directories -- the
repository root in this checkout. Add `--expected-episodes N` only when you
want the count pinned; the current corpus indexes 46 episodes.

This is a structural readiness check, not a substitute for WER, DER,
target/polarity F1, or human validation. It fails if a transcript, a
diarization output, an annotation draft or the required speaker-attribution
coverage is missing.

Source audio that is absent for a reason already on record is reported as
**GAP** rather than FAIL, so a genuinely new regression still stands out. Those
episodes are listed with their reason in `demo/known_gaps.json`; six episodes
are currently in that state and their transcripts remain fully searchable.

If a roll's bundle is missing the annotation-draft stage, rebuild it from the
fused output -- it is a deterministic projection, not a judgement:

```powershell
python demo/make_missing_drafts.py --data-root .          # report what is missing
python demo/make_missing_drafts.py --data-root . --apply  # write them
```

## Local launch

From the repository root:

```powershell
python -m pip install -r demo/requirements.txt
python demo/app.py --data-root .
```

Open the local URL the app prints. It is normally
`http://127.0.0.1:7860`, but Windows reserves blocks of TCP ports for
Hyper-V/WSL and on some machines one of those blocks covers 7860. When the
requested port cannot be bound the app says so and moves to a free one, so
read the printed URL rather than assuming 7860. To see the reserved blocks:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

To pick a port yourself, or to create a temporary public link:

```powershell
python demo/app.py --data-root . --server-port 8080
python demo/app.py --data-root . --share
```

The default bind address is local-only (`127.0.0.1`). Pass
`--server-name 0.0.0.0` only when the app must accept connections from other
machines on the network.

## Google Colab launch

Run these cells after the project folder is available in Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Project")
assert (PROJECT_ROOT / "demo/app.py").exists(), PROJECT_ROOT
%cd /content/drive/MyDrive/ML_Project
```

```python
!python -m pip install -q -r demo/requirements.txt
```

```python
!python demo/app.py \
    --data-root /content/drive/MyDrive/ML_Project/datasets \
    --cache-dir /content/bangla_financial_search_cache \
    --share
```

The app opens immediately for Smart/Exact/Fuzzy search. Clicking **Prepare
semantic search** downloads the pinned multilingual-E5 checkpoint once and
builds a content-addressed embedding cache. With roughly 30 episodes, use a
GPU runtime for the fastest first build; later searches reuse the cache for
the lifetime of that Colab VM.

No Gemini API key is needed. Exact, fuzzy, semantic, filtering, audio preview,
and CSV export all run locally inside the Colab runtime. A Hugging Face token
is optional and only helps avoid anonymous download rate limits.

## Recommended live demonstration

1. Open **Project overview** and establish the 30-episode, 15–18-hour scale,
   completed pipeline stages, and model provenance.
2. Open **Pipeline results** and briefly show audio, diarization, ASR, fusion,
   and annotation-draft evidence.
3. Open **Financial analysis** and explain the target taxonomy, entity lexicon,
   and why weak-labelled legacy profiles are marked exploratory.
4. In **Search & evidence**, query `ব্যাংক` using Word + Exact across all episodes.
5. Restrict the same query to one episode and one speaker.
6. Query `ব্যাংক খাতের সংকট` with Fuzzy or Hybrid search.
7. Query a question such as `রেমিট্যান্স কেন কমছে?` with Semantic search.
8. Select a result, play its audio excerpt, and point out the episode-local
   speaker ID and timestamp.
9. Open **Evaluation & downloads**, explain unavailable gold-dependent metrics,
   and export both the result CSV and full project-evidence ZIP.

Keep the limitation sentence visible during the demo: transcripts and
speaker IDs are automatic predictions and have not been human-verified.
