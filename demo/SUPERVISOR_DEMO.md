# Supervisor demonstration runbook

## Before the meeting

1. Put each member's audio and result bundle below
   `datasets/<roll>/<roll>_results/` using the layout in `README.md`.
2. Episode IDs may repeat across rolls because the demo uses
   `<roll>::<episode>` as the unique key. Never reinterpret an episode-local
   speaker ID as the same person in another episode.
3. Run the readiness audit and keep its JSON beside the consolidated data.
4. Launch the app once, prepare semantic search, and try every query below.
5. Keep a local browser tab open even if you also use a temporary share link.

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
10. **Evaluation status:** show the complete gate table. Explicitly distinguish
    missing gold-dependent metrics from completed automatic processing.
11. **Reproducible output:** download the search CSV and project-evidence ZIP.

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
- Do not claim measured WER, DER, or sentiment F1 when gold annotations are
  unavailable.
- Do not say the output is guaranteed correct. Present it as automatic,
  traceable evidence ready for targeted review.
