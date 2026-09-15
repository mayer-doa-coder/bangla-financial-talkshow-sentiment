"""Stage B: LLM-as-judge. Read each speaker's whole script, label the speaker
and every one of their turns in the same pass.

Why one pass for both levels. A speaker-level verdict alone gives 61 labelled
examples, which is not enough to fine-tune a 110M-parameter encoder. Turn-level
labels alone lose the thing we actually want to predict. Asking the judge for
both in one call, with the full script in context, gets 920 training rows (1,552
with --scope all) whose labels were assigned by a judge that knew how the
speaker's argument develops - so the turn labels and the speaker verdict cannot
contradict each other, which is exactly the inconsistency you get from labelling
turns in isolation and aggregating afterwards.

What the labels mean. This is not emotion. It is the speaker's evaluative
stance on the economic and financial subject matter: is the outlook they express
unfavourable, favourable, or are they describing and prescribing without
judging. Bangladeshi financial talk shows are dominated by the third case, which
is why "neutral" is a first-class class and not a dustbin.

    export ANTHROPIC_API_KEY=...            # or: ant auth login
    python speaker_sentiment/llm_judge.py --limit 3 --dry-run   # inspect prompts, no spend
    python speaker_sentiment/llm_judge.py --scope all           # full run, all turns
    python speaker_sentiment/llm_judge.py --scope all --passes 3  # self-consistency

Results are appended to speaker_sentiment/data/judgements.jsonl and the script
is resumable: a speaker already present for a given pass is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

MODEL = "claude-opus-5"
TURN_LABELS = ["negative", "neutral", "positive", "illegible"]
SPEAKER_LABELS = ["negative", "mixed", "neutral", "positive"]

SYSTEM_PROMPT = """You are annotating Bangla (Bengali) financial television talk shows for a \
speaker-level sentiment corpus. You are a careful, consistent annotator, not an assistant: \
you produce labels, not advice.

## The data

Each item is one speaker from one episode. The text is MACHINE TRANSCRIBED Bangla speech from \
an ASR system with a substantial word error rate, then split by speaker diarization. Expect:

- Misrecognised words, invented words, and dropped word endings.
- English financial terms transcribed inline (inflation, reserve, GDP, budget) or in Bangla.
- Numbers frequently wrong.
- No punctuation at all, so sentence boundaries must be inferred.
- Occasional segments that are simply unreadable.

Read past transcription noise where the intent is still recoverable. Where it is not \
recoverable, say so with the `illegible` label rather than guessing.

## What you are labelling

NOT emotion, tone of voice, or politeness. You are labelling EVALUATIVE STANCE toward the \
economic and financial subject matter: how the speaker judges the state, direction, or \
handling of the economy, a sector, a market, a policy, an institution, or a company.

Turn labels:

- `negative` - the speaker evaluates the economic subject unfavourably: naming a crisis, \
decline, failure, risk, harm, mismanagement, or unmet need; warning that things will worsen; \
criticising a policy, institution, or actor.
- `positive` - the speaker evaluates it favourably: naming improvement, recovery, growth, \
success, resilience, or opportunity; endorsing a policy or decision; expressing confidence \
that things will improve.
- `neutral` - the speaker mentions economic subject matter without judging it. This covers \
descriptive reporting of facts and figures, definitions and explanations, procedural talk, \
questions put to another panellist, introductions and hand-offs, prescriptive recommendations \
stated without blaming anyone ("we should build capacity"), and hedged or genuinely balanced \
statements. Neutral is the correct label for most host turns.
- `illegible` - the transcription is too corrupted to read a stance from, or the turn has no \
economic content at all (greetings, a name, crosstalk).

## Rules that decide the hard cases

1. **Direction words are not polar by themselves.** "বেড়েছে" (has increased) is favourable for \
reserves, exports, or investment and unfavourable for inflation, debt, or unemployment. Decide \
from what is rising or falling, never from the direction word.
2. **A question is neutral even when it is pointed.** Hosts ask "why has the crisis not \
lifted?" to elicit an answer, not to assert a verdict. Label it neutral unless the host \
themself asserts the unfavourable claim.
3. **A recommendation is neutral; a complaint is negative.** "The government must raise the \
allocation" is neutral. "The government failed to raise the allocation" is negative.
4. **Reported speech is not the speaker's stance.** If the speaker cites what CPD, the IMF, or \
a minister said, that is neutral unless the speaker endorses or attacks it.
5. **Sarcasm and rhetorical questions carry their intended stance**, not their literal one, \
when the transcription makes that clear.
6. **Do not let one vivid sentence set the label for a long turn.** Weigh the turn as a whole.
7. **Label each turn on its own content.** Use the full script to understand who the speaker is \
and what they are arguing, but do not carry a previous turn's label forward out of momentum.

## The speaker-level verdict

After labelling the turns, judge the speaker as a whole:

- `negative` - their overall outlook across the episode is unfavourable.
- `positive` - overall favourable.
- `mixed` - they make substantial evaluations in both directions (for example, a guest who \
condemns the banking sector but praises remittance growth). Use this when both are genuinely \
present, not when the speaker is merely uncommitted.
- `neutral` - they largely do not evaluate. Hosts and moderators usually land here.

`stance_score` is a number from -1.0 (uniformly unfavourable) to +1.0 (uniformly favourable), \
with 0.0 for neutral or evenly mixed. It should be consistent with the verdict and with the \
balance of turn labels, but it is your holistic judgement, not an arithmetic mean.

## Output discipline

- Emit exactly one entry per turn, with the turn's index, and in the order given.
- `confidence` is your confidence in that turn's label: `high`, `medium`, or `low`. Use `low` \
freely where the transcription is poor - downstream training weights these.
- `evidence` is a SHORT quote copied verbatim from the turn that carries the stance (empty \
string for neutral or illegible turns). Do not paraphrase and do not translate.
- Judge only what is in the transcript. Do not use outside knowledge of Bangladesh's economy \
to decide what the speaker must have meant."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_summary": {
            "type": "string",
            "description": "Two or three sentences in English: who this speaker appears "
                           "to be, what they argue, and how they evaluate it.",
        },
        "speaker_label": {"type": "string", "enum": SPEAKER_LABELS},
        "stance_score": {"type": "number"},
        "speaker_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "speaker_rationale": {
            "type": "string",
            "description": "One or two sentences justifying the speaker label.",
        },
        "dominant_topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to five economic topics or entities this speaker evaluates, "
                           "in English (e.g. 'banking sector', 'inflation', 'budget').",
        },
        "transcript_quality": {"type": "string", "enum": ["good", "fair", "poor"]},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string", "enum": TURN_LABELS},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence": {"type": "string"},
                },
                "required": ["index", "label", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["speaker_summary", "speaker_label", "stance_score", "speaker_confidence",
                 "speaker_rationale", "dominant_topics", "transcript_quality", "turns"],
    "additionalProperties": False,
}

ROLE_NOTE = {
    "host": "This speaker is the programme host or moderator. Hosts introduce the topic, put "
            "questions to guests, and hand off between panellists. Most of their turns are "
            "neutral; label a host turn polar only where the host asserts an evaluation in "
            "their own voice.",
    "guest": "This speaker is an invited panellist - typically an economist, banker, business "
             "association official, or policy adviser. Guests carry most of the evaluative "
             "content in these programmes.",
    "minor": "This speaker contributes only briefly - a caller, a field reporter, or crosstalk.",
    "unassigned": "WARNING: speaker diarization failed on this episode, so these turns are a "
                  "BAG OF TURNS FROM SEVERAL DIFFERENT PEOPLE, not one person. Label each turn "
                  "on its own content as usual - turn labels are still valid supervision. For "
                  "the speaker-level fields, describe the panel's overall stance and set "
                  "speaker_confidence to `low`; that verdict will be discarded.",
}


def build_user_prompt(speaker: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    """The speaker's whole script, one numbered turn per block."""
    header = [
        f"Programme: {speaker['programme'] or 'unknown'}",
        f"Episode: {speaker['global_episode']}",
        f"Speaker: S{speaker['speaker_id']} ({speaker['role']})",
        f"Speaking time: {speaker['dur_sec']:.0f} s across {len(turns)} turns, "
        f"{speaker['n_clean_words']} words",
        "",
        ROLE_NOTE.get(speaker["role"], ""),
        "",
        f"Label all {len(turns)} turns below, then judge the speaker as a whole.",
        "",
        "--- SPEAKER SCRIPT ---",
    ]
    body = []
    for i, turn in enumerate(turns):
        minutes, seconds = divmod(int(turn["start_sec"]), 60)
        body.append(f"[turn {i}] ({minutes:d}:{seconds:02d})\n{turn['text']}")
    return "\n".join(header) + "\n\n" + "\n\n".join(body) + "\n\n--- END SCRIPT ---"


def judge_one(client, speaker: dict[str, Any], turns: list[dict[str, Any]],
              model: str, effort: str) -> dict[str, Any]:
    """One speaker, one API call. Streamed because a long script plus per-turn
    output can run past the non-streaming HTTP timeout."""
    with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            # The rubric is byte-identical across all 61 calls; cache it.
            "cache_control": {"type": "ephemeral"},
        }],
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
        messages=[{"role": "user", "content": build_user_prompt(speaker, turns)}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused: {getattr(response.stop_details, 'category', None)}")
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    parsed["_usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }
    return parsed


def attach(speaker: dict[str, Any], turns: list[dict[str, Any]], parsed: dict[str, Any],
           pass_id: int, model: str) -> dict[str, Any]:
    """Map the judge's turn indices back onto real turn_ids. An index the judge
    invented or skipped is dropped rather than silently shifting every label
    after it onto the wrong turn."""
    by_index = {}
    for item in parsed.get("turns", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(turns) and idx not in by_index:
            by_index[idx] = item
    labelled = [
        {
            "turn_id": turns[i]["turn_id"],
            "index": i,
            "label": by_index[i]["label"],
            "confidence": by_index[i]["confidence"],
            "evidence": by_index[i]["evidence"],
        }
        for i in range(len(turns)) if i in by_index
    ]
    return {
        "speaker_key": speaker["speaker_key"],
        "global_episode": speaker["global_episode"],
        "roll": speaker["roll"],
        "role": speaker["role"],
        # Only a trusted, real speaker id yields a usable speaker-level verdict.
        # Everything else contributes turn labels only.
        "speaker_level_valid": bool(speaker["judgeable"]),
        "pass_id": pass_id,
        "judge_model": model,
        "speaker_label": parsed["speaker_label"],
        "stance_score": parsed["stance_score"],
        "speaker_confidence": parsed["speaker_confidence"],
        "speaker_summary": parsed["speaker_summary"],
        "speaker_rationale": parsed["speaker_rationale"],
        "dominant_topics": parsed["dominant_topics"],
        "transcript_quality": parsed["transcript_quality"],
        "n_turns_sent": len(turns),
        "n_turns_returned": len(labelled),
        "turn_labels": labelled,
        "usage": parsed.get("_usage", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="speaker_sentiment/data")
    ap.add_argument("--out", default="speaker_sentiment/data/judgements.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--passes", type=int, default=1,
                    help="repeat the whole run N times for self-consistency agreement")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="judge at most N speakers (smoke test)")
    ap.add_argument("--scope", default="trusted", choices=["trusted", "all"],
                    help="trusted: only pyannote episodes with a real speaker id "
                         "(61 speakers / 920 turns) - the speaker-level task needs these. "
                         "all: additionally judge fallback-diarized episodes and unassigned "
                         "buckets (73 / 1,552). Turn-level stance does not depend on correct "
                         "speaker attribution, so `all` buys ~69%% more training data for "
                         "Task A; make_datasets.py keeps those turns out of the "
                         "speaker-level evaluation.")
    ap.add_argument("--min-words", type=int, default=150,
                    help="minimum clean words for a unit to be worth a call")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompts that would be sent and exit")
    args = ap.parse_args()

    data = Path(args.data)
    speakers = [json.loads(l) for l in (data / "speakers.jsonl").open(encoding="utf-8")]
    all_turns = [json.loads(l) for l in (data / "turns.jsonl").open(encoding="utf-8")]
    turns_by_id = {t["turn_id"]: t for t in all_turns}

    if args.scope == "trusted":
        queue = [s for s in speakers if s["judgeable"]]
    else:
        queue = [s for s in speakers if s["n_clean_words"] >= args.min_words]
    queue.sort(key=lambda s: s["speaker_key"])
    if args.limit:
        queue = queue[:args.limit]

    out_path = Path(args.out)
    done: set[tuple[str, int]] = set()
    if out_path.is_file():
        for line in out_path.open(encoding="utf-8"):
            row = json.loads(line)
            done.add((row["speaker_key"], row["pass_id"]))
        print(f"resuming: {len(done)} (speaker, pass) results already on disk")

    jobs = [
        (s, [turns_by_id[tid] for tid in s["turn_ids"]], p)
        for p in range(1, args.passes + 1)
        for s in queue
        if (s["speaker_key"], p) not in done
    ]
    n_valid = sum(1 for s in queue if s["judgeable"])
    n_turns = sum(len(s["turn_ids"]) for s in queue)
    print(f"{len(queue)} units ({n_valid} usable for the speaker-level task, "
          f"{n_turns:,} turns) x {args.passes} pass(es) -> {len(jobs)} calls to make")

    if args.dry_run:
        for speaker, turns, _ in jobs[:2]:
            print("\n" + "=" * 78)
            print(build_user_prompt(speaker, turns)[:4000])
            print(f"... [{len(turns)} turns total]")
        chars = sum(len(build_user_prompt(s, t)) for s, t, _ in jobs)
        print(f"\n{'=' * 78}\ntotal prompt characters: {chars:,}")
        print(f"rough input-token estimate: {chars // 3:,} (Bangla runs ~3 chars/token)")
        return

    if not jobs:
        print("nothing to do")
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("no ANTHROPIC_API_KEY in the environment; falling back to an "
              "`ant auth login` profile if one exists", file=sys.stderr)
    client = anthropic.Anthropic(max_retries=4)

    lock = threading.Lock()
    fh = out_path.open("a", encoding="utf-8")
    counts: Counter = Counter()
    usage = Counter()

    def run(job):
        speaker, turns, pass_id = job
        parsed = judge_one(client, speaker, turns, args.model, args.effort)
        return attach(speaker, turns, parsed, pass_id, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            speaker, turns, pass_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:                      # noqa: BLE001 - one bad speaker
                print(f"[{n}/{len(jobs)}] FAILED {speaker['speaker_key']} p{pass_id}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                counts["failed"] += 1
                continue
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            counts[row["speaker_label"]] += 1
            for k, v in row["usage"].items():
                usage[k] += v
            missing = row["n_turns_sent"] - row["n_turns_returned"]
            note = f"  ({missing} turns unreturned)" if missing else ""
            print(f"[{n}/{len(jobs)}] {row['speaker_key']:<26s} {row['role']:<6s} "
                  f"-> {row['speaker_label']:<8s} {row['stance_score']:+.2f} "
                  f"({row['n_turns_returned']} turns){note}")
    fh.close()

    print("\nspeaker verdicts this run:")
    for label, n in counts.most_common():
        print(f"  {label:10s} {n}")
    cost = usage["input_tokens"] * 5e-6 + usage["output_tokens"] * 25e-6
    print(f"\ntokens: in={usage['input_tokens']:,} "
          f"(cached {usage['cache_read_input_tokens']:,}) out={usage['output_tokens']:,}")
    print(f"approx cost at {args.model} list rates: ${cost:.2f} "
          "(cache reads billed lower, so this is an upper bound)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
