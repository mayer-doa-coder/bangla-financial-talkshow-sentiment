"""Stage A: turn the fused ASR+diarization bundles into a speaker-centred corpus.

The existing sentiment/ pipeline is target-aware: one row per (utterance, financial
target). This module is a different cut of the same bundles - one row per speaker
TURN and one document per (episode, speaker) - because that is the unit a
speaker-level sentiment label attaches to.

Three things the fused files force us to handle:

1. The "utterances" in fused/*.json are not turns. They are ~28 s ASR decode
   windows that happen to be cut at speaker changes, so a single continuous
   answer is split into 5-10 rows. Consecutive same-speaker windows are merged
   back into one turn here.
2. Every window boundary truncates a word. A window whose first character is a
   Bengali dependent sign lost its leading consonant to the chunker, so that
   fragment token is dropped when merging.
3. 2107009 was diarized with the energy-based fallback, not pyannote. Roughly
   60 percent of its words carry speaker_id -1. Those five episodes are kept in
   the corpus but marked diarization_trusted=false and excluded from
   speaker-level supervision by default.

    python speaker_sentiment/build_speaker_corpus.py --data-root "D:/ML DATASETS"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Bengali dependent signs. A word may not begin with one; when a decode window
# does, the chunker ate its first consonant.
COMBINING = set(
    "\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c7\u09c8"
    "\u09cb\u09cc\u09cd\u09d7\u09bc"
)
# Two windows separated by more than this are different turns even for one speaker.
TURN_GAP_SEC = 2.0
UNASSIGNED = -1


def norm(text: str) -> str:
    """Bundles came back from Drive in mixed normal forms; compare in NFC."""
    return unicodedata.normalize("NFC", text)


def resolve_stage(bundle: Path, stage: str) -> Path | None:
    """Bundles were exported in two shapes: stage folders directly under the
    bundle, or wrapped in data/. Accept both."""
    for candidate in (bundle / stage, bundle / "data" / stage):
        if candidate.is_dir():
            return candidate
    return None


def find_bundles(data_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for roll in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for bundle in sorted(roll.iterdir()):
            if bundle.is_dir() and resolve_stage(bundle, "fused"):
                out.append((roll.name, bundle))
    return out


def repetition_score(text: str) -> tuple[int, float]:
    """Longest run of one repeated token, and the unique-token ratio.

    The CTranslate2 decoder in this run fell into loops. The run catches a single
    repeated token; the ratio catches an alternating loop."""
    tokens = text.split()
    if not tokens:
        return 0, 1.0
    longest = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        longest = max(longest, run)
    return longest, len(set(tokens)) / len(tokens)


def is_loop(text: str) -> bool:
    words = len(text.split())
    longest, unique = repetition_score(text)
    return longest >= 4 or (words >= 20 and unique < 0.35)


def collapse_loops(text: str, keep: int = 2) -> str:
    """Leave a repeated token at most `keep` times. A loop carries no extra
    signal and eats the 512-token budget BanglaBERT has to spend."""
    out: list[str] = []
    for token in text.split():
        if len(out) >= keep and all(t == token for t in out[-keep:]):
            continue
        out.append(token)
    return " ".join(out)


def drop_leading_fragment(text: str) -> str:
    """Drop the first token when it opens with a dependent sign: the chunker
    split a word across the window boundary and this is its tail."""
    stripped = text.lstrip()
    if stripped and stripped[0] in COMBINING:
        parts = stripped.split(" ", 1)
        return parts[1] if len(parts) > 1 else ""
    return stripped


def asr_confidence_index(asr_dir: Path | None, episode: str) -> list[tuple[float, float, float]]:
    if asr_dir is None:
        return []
    path = asr_dir / f"{episode}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        (float(s["start"]), float(s["end"]), float(s.get("confidence", 0.0)))
        for s in data.get("segments", [])
    ]


def mean_confidence(index, start: float, end: float) -> float | None:
    """Duration-weighted mean of the ASR segment log-probabilities overlapping
    this span. Used downstream to down-weight turns the ASR was unsure of."""
    total = weight = 0.0
    for seg_start, seg_end, conf in index:
        overlap = min(end, seg_end) - max(start, seg_start)
        if overlap > 0:
            total += conf * overlap
            weight += overlap
    return round(total / weight, 4) if weight else None


def programme_titles(bundle: Path) -> dict[str, str]:
    """Episode titles from registry.csv. The judge reads these: knowing an
    episode is about the national budget rather than the share market changes
    what a bare "it will fall" is evaluating."""
    stage = resolve_stage(bundle, "raw")
    if stage is None:
        return {}
    path = stage / "registry.csv"
    if not path.is_file():
        return {}
    import csv
    out = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            title = (row.get("programme") or "").strip()
            if row.get("episode_id") and title and title != "unknown":
                out[row["episode_id"]] = title
    return out


def diarization_provider(bundle: Path, episode: str) -> str:
    stage = resolve_stage(bundle, "diarization")
    if stage is None:
        return "unknown"
    path = stage / f"{episode}.json"
    if not path.is_file():
        return "unknown"
    return json.loads(path.read_text(encoding="utf-8")).get("provider", "unknown")


def merge_turns(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stitch consecutive same-speaker decode windows back into one turn."""
    turns: list[dict[str, Any]] = []
    for utt in utterances:
        raw = norm(utt.get("text_normalized") or utt.get("text_verbatim") or "").strip()
        if not raw:
            continue
        speaker = utt.get("speaker_id", UNASSIGNED)
        start, end = float(utt["start_sec"]), float(utt["end_sec"])
        piece = drop_leading_fragment(raw)
        if not piece:
            continue
        if (turns and turns[-1]["speaker_id"] == speaker
                and start - turns[-1]["end_sec"] <= TURN_GAP_SEC):
            turns[-1]["pieces"].append(piece)
            turns[-1]["end_sec"] = end
            turns[-1]["window_ids"].append(utt["utterance_id"])
        else:
            turns.append({
                "speaker_id": speaker, "start_sec": start, "end_sec": end,
                "pieces": [piece], "window_ids": [utt["utterance_id"]],
            })
    return turns


def infer_roles(per_speaker: dict[int, dict[str, Any]], first_speaker: int) -> dict[int, str]:
    """A talk-show host takes many short turns; a guest takes few long ones, so
    the host is the speaker whose share of turns most exceeds its share of words.
    Speaking first breaks a tie, because these programmes open on the host."""
    live = {s: v for s, v in per_speaker.items() if s != UNASSIGNED and v["n_turns"] >= 3}
    if not live:
        return {s: "unknown" for s in per_speaker}
    total_turns = sum(v["n_turns"] for v in live.values())
    total_words = sum(v["n_words"] for v in live.values())
    scored = {
        s: (v["n_turns"] / total_turns) - (v["n_words"] / max(total_words, 1))
        + (0.05 if s == first_speaker else 0.0)
        for s, v in live.items()
    }
    host = max(scored, key=scored.get)
    roles = {}
    for s in per_speaker:
        if s == UNASSIGNED:
            roles[s] = "unassigned"
        elif s == host:
            roles[s] = "host"
        elif s in live:
            roles[s] = "guest"
        else:
            roles[s] = "minor"       # a few seconds of crosstalk or a caller
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--out", default="speaker_sentiment/data")
    ap.add_argument("--min-turn-words", type=int, default=5,
                    help="turns shorter than this are kept but flagged too_short")
    ap.add_argument("--min-judge-words", type=int, default=150,
                    help="a speaker needs this much clean speech to be worth judging")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    seen_transcripts: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []

    for roll, bundle in find_bundles(data_root):
        fused_dir = resolve_stage(bundle, "fused")
        asr_dir = resolve_stage(bundle, "asr")
        titles = programme_titles(bundle)
        for fused_path in sorted(fused_dir.glob("*.json")):
            data = json.loads(fused_path.read_text(encoding="utf-8"))
            episode = data.get("episode_id", fused_path.stem)
            gid = f"{roll}::{episode}"
            utterances = data.get("utterances", [])

            # Two rolls submitted byte-identical exports of the same recording.
            # Keeping both would put the same panel in train and test.
            digest = hashlib.md5(
                "".join(u.get("text_verbatim") or "" for u in utterances).encode("utf-8")
            ).hexdigest()
            if digest in seen_transcripts:
                duplicates.append((gid, seen_transcripts[digest]))
                continue
            seen_transcripts[digest] = gid

            provider = diarization_provider(bundle, episode)
            trusted = provider == "pyannote"
            conf_index = asr_confidence_index(asr_dir, episode)
            turns = merge_turns(utterances)

            per_speaker: dict[int, dict[str, Any]] = defaultdict(
                lambda: {"n_turns": 0, "n_words": 0, "dur_sec": 0.0, "n_loop": 0}
            )
            built: list[dict[str, Any]] = []
            for i, turn in enumerate(turns):
                raw = " ".join(turn["pieces"])
                clean = collapse_loops(raw)
                n_words = len(clean.split())
                row = {
                    "turn_id": f"{gid}::turn{i:04d}",
                    "roll": roll,
                    "episode": episode,
                    "global_episode": gid,
                    "turn_index": i,
                    "speaker_id": turn["speaker_id"],
                    "start_sec": round(turn["start_sec"], 2),
                    "end_sec": round(turn["end_sec"], 2),
                    "dur_sec": round(turn["end_sec"] - turn["start_sec"], 2),
                    "text": clean,
                    "text_raw": raw,
                    "n_words": n_words,
                    "n_windows": len(turn["window_ids"]),
                    "asr_confidence": mean_confidence(
                        conf_index, turn["start_sec"], turn["end_sec"]),
                    "diarization_provider": provider,
                    "diarization_trusted": trusted,
                    "flag_repetition_loop": is_loop(raw),
                    "flag_too_short": n_words < args.min_turn_words,
                }
                built.append(row)
                agg = per_speaker[turn["speaker_id"]]
                agg["n_turns"] += 1
                agg["n_words"] += n_words
                agg["dur_sec"] += row["dur_sec"]
                agg["n_loop"] += int(row["flag_repetition_loop"])

            first_speaker = built[0]["speaker_id"] if built else UNASSIGNED
            roles = infer_roles(per_speaker, first_speaker)
            for row in built:
                row["role"] = roles.get(row["speaker_id"], "unknown")
            turn_rows.extend(built)

            for speaker, agg in sorted(per_speaker.items()):
                own = [r for r in built if r["speaker_id"] == speaker]
                usable = [r for r in own
                          if not r["flag_repetition_loop"] and not r["flag_too_short"]]
                confs = [r["asr_confidence"] for r in own if r["asr_confidence"] is not None]
                clean_words = sum(r["n_words"] for r in usable)
                speaker_rows.append({
                    "speaker_key": f"{gid}::S{speaker}",
                    "roll": roll,
                    "episode": episode,
                    "global_episode": gid,
                    "programme": titles.get(episode, ""),
                    "speaker_id": speaker,
                    "role": roles.get(speaker, "unknown"),
                    "n_turns": agg["n_turns"],
                    "n_usable_turns": len(usable),
                    "n_words": agg["n_words"],
                    "n_clean_words": clean_words,
                    "dur_sec": round(agg["dur_sec"], 1),
                    "mean_words_per_turn": round(agg["n_words"] / max(agg["n_turns"], 1), 1),
                    "n_loop_turns": agg["n_loop"],
                    "mean_asr_confidence": round(sum(confs) / len(confs), 4) if confs else None,
                    "diarization_provider": provider,
                    "diarization_trusted": trusted,
                    # The full script is what the LLM judge reads. Loop-collapsed
                    # and fragment-trimmed, turns joined by a blank line so the
                    # judge can see where the speaker was interrupted.
                    "script": "\n\n".join(r["text"] for r in usable),
                    "turn_ids": [r["turn_id"] for r in usable],
                    # Judgeable = enough clean speech that a holistic verdict means
                    # something, and a speaker label we trust.
                    "judgeable": bool(trusted and speaker != UNASSIGNED
                                      and clean_words >= args.min_judge_words),
                })

    turns_path = out_dir / "turns.jsonl"
    speakers_path = out_dir / "speakers.jsonl"
    with turns_path.open("w", encoding="utf-8") as fh:
        for r in turn_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with speakers_path.open("w", encoding="utf-8") as fh:
        for r in speaker_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    episodes = {r["global_episode"] for r in turn_rows}
    trusted_eps = {r["global_episode"] for r in turn_rows if r["diarization_trusted"]}
    judgeable = [r for r in speaker_rows if r["judgeable"]]
    print(f"episodes kept          {len(episodes)}  ({len(trusted_eps)} pyannote, "
          f"{len(episodes) - len(trusted_eps)} fallback)")
    for dup, keeper in duplicates:
        print(f"  dropped duplicate    {dup} == {keeper}")
    print(f"turns                  {len(turn_rows):,}")
    print(f"  repetition loops     {sum(r['flag_repetition_loop'] for r in turn_rows):,}")
    print(f"  under {args.min_turn_words} words        "
          f"{sum(r['flag_too_short'] for r in turn_rows):,}")
    print(f"speaker units          {len(speaker_rows)}  ({len(judgeable)} judgeable)")
    print("\nrole distribution (turns):")
    for role, n in Counter(r["role"] for r in turn_rows).most_common():
        words = sum(r["n_words"] for r in turn_rows if r["role"] == role)
        print(f"  {role:12s} turns={n:5d}  words={words:7,d}")
    print(f"\nwrote {turns_path}")
    print(f"wrote {speakers_path}")


if __name__ == "__main__":
    main()
