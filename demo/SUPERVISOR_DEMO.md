# Supervisor demonstration runbook

## Before the meeting

1. Each member's audio and result bundle sits under their roll folder. Point
   `--data-root` at the folder holding those roll folders (the repository root
   in this checkout); the loader accepts the layout variations the members
   actually used.
2. Episode IDs may repeat across rolls because the demo uses
   `<roll>::<episode>` as the unique key. Never reinterpret an episode-local
   speaker ID as the same person in another episode.
3. Run the readiness audit and keep its JSON beside the consolidated data:
   `python demo/audit_demo.py --data-root . --report demo_readiness.json`.
   Expect `ready: true` with one **GAP** on `audio_availability` -- six
   episodes whose source recording was never uploaded, each listed with its
   reason in `demo/known_gaps.json`. Their transcripts still search normally;
   only playback is unavailable, so pick a playback example from another
   episode.
4. Confirm the trained model will run on this machine:
   `python -X utf8 demo/check_model_setup.py`. It loads the checkpoint offline
   and re-predicts two dozen held-out turns, so a green result means the live
   tab will work in front of an audience. It needs no network.

5. Launch the app once, prepare semantic search, and try every query below.
   Read the local URL the app prints: 7860 is inside a Windows reserved port
   range on some machines, and the app then moves to a free port and says so.
6. Keep a local browser tab open even if you also use a temporary share link.
   Open the **Run the model** tab and press *Load the model now* before the
   audience arrives. The checkpoint is 422 MB and the first load takes a few
   seconds; every prediction after that is quick.

## Eight-minute live sequence

1. **Project overview:** show the scanned roll folders, individual contribution
   table, group totals, speaker-assignment coverage, completion map and model
   provenance.
2. **Pipeline evidence:** open the audio, diarization, ASR, fusion and
   annotation-draft subtabs. Point out that these are real durable artifacts,
   not presentation-only numbers.
3. **Financial design:** show the five target types and financial entity
   lexicon. Explain why target and speaker remain separate.
4. **Legacy experiment:** show the weak-labelled discourse-profile snapshot
   only after pointing to its `llm_weak_exploratory` tier and alignment warning.
5. **Literal evidence:** search `ব্যাংক` with **Word + Exact** over the whole
   corpus. Explain that every result retains roll, episode, predicted
   speaker, timestamp, transcript, and source audio.
6. **Scoped search:** restrict the same query to one episode and one speaker.
   This demonstrates the episode-local identity rule.
7. **ASR-tolerant retrieval:** search `ব্যাংক খাতের সংকট` with **Fuzzy**.
8. **Meaning-based retrieval:** search `রেমিট্যান্স কেন কমছে?` with
   **Concept / question + Hybrid**.
9. **Evidence playback:** select one result and play its audio excerpt.
10. **Speaker sentiment:** open the Speaker sentiment tab. Show the per-speaker
    verdict table, then the model's held-out scores: macro-F1 0.670 against a
    0.236 majority-class floor and a 0.529 TF-IDF control, with the confusion
    matrix and per-class table beside it. Say plainly that the labels come
    from an LLM judge under a fixed rubric, so these are agreement figures,
    not agreement with human truth.
11. **Run the model live.** Open the **Run the model** tab. This is the step
    that separates a recorded result from a working model.

    - Pick a speaker and press **Run the model**. The checkpoint classifies
      every turn that speaker took and aggregates them into one episode
      verdict, on this machine, with no network. The judge's own verdict sits
      beside it, so agreement and disagreement are both visible.
    - Play the clip that appears so the room hears the speaker being scored.
    - Then paste a sentence of your own into **classify any Bangla text**.
      Something the model has never seen is the most convincing test, and it
      answers the obvious question of whether the numbers were merely loaded
      from a file.

    Say plainly what the comparison means: the judge column is a language
    model's opinion, not human ground truth, so the match rate is agreement
    between two automatic systems.

12. **Evaluation status:** show the complete gate table. Explicitly distinguish
    missing gold-dependent metrics from completed automatic processing.
13. **Reproducible output:** download the search CSV and project-evidence ZIP.

## Suggested explanation

“The pipeline first transcribes and diarizes each long-form Bangla episode,
then fuses word timestamps with speaker turns. This application searches that
fused evidence. A label such as `2107006::ep002 / Speaker 1` is an opaque,
episode-local predicted speaker—not a verified identity. Exact search provides
literal evidence; fuzzy search tolerates ASR/spelling differences; semantic
and hybrid search retrieve related meaning. Clicking a result returns to the
audio, so every claim is inspectable.”

## Claims to avoid

- Do not call the speaker ID a person's name or link it across episodes.
- Do not call retrieval relevance “accuracy.”
- Do not claim measured WER, DER, or target/polarity F1: those still have no
  gold reference.
- Do quote the speaker-sentiment macro-F1, but always with what it is measured
  against -- an LLM judge, not human annotation.
- The live tab predicts; it does not re-train. If asked whether the numbers
  could have been fabricated, the honest answer is that the reported scores
  come from the recorded run, and the live tab reproduces that run's
  predictions exactly on the held-out turns, which anyone can check with
  `demo/check_model_setup.py`. No human has re-labelled an
  individual turn; every annotation draft still reads `human_verified: false`.
- Do not say the output is guaranteed correct. Present it as automatic,
  traceable evidence ready for targeted review.
