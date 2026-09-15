"""Word-to-speaker fusion for bundles whose ``fused/`` export is missing.

Some members exported their diarization and ASR stages but not the fusion
stage, so their episodes carry no searchable transcript and never appear in the
demo. This module re-runs stage 3 from the durable ``diarization/`` and ``asr/``
artifacts that are already in the bundle. It invents nothing: every word,
timestamp and score comes from the stored ASR output, and every speaker
interval comes from the stored diarization output.

The rule implemented here is the one recorded in the project report:

* assign each word by diarization-interval midpoint, falling back to the
  interval with the largest temporal overlap, and keep ``speaker_id = -1``
  for words with no overlap at all rather than dropping them;
* split utterances on speaker change, a pause longer than two seconds,
  sentence-ending punctuation, or the maximum utterance duration.

``--verify`` replays the rule against a bundle that already contains fused
files and reports whether the regenerated output matches byte for byte, which
is how the thresholds below were confirmed against 2107004 and 2107006.

Usage:
    python demo/fuse_stage.py --bundle 2107001/results
    python demo/fuse_stage.py --bundle 2107004/results --verify
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from search_core import resolve_stage_dir

SCHEMA_VERSION = "2.0"
OVERLAP_POLICY = "first-speaker-wins"
ASSIGNMENT = "midpoint_then_max_overlap"

MAX_PAUSE_SEC = 2.0
MAX_UTTERANCE_SEC = 28.0
SENTENCE_ENDINGS = ("।", "?", "!", "؟")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_entity_lexicon(bundle_root: Path) -> list[tuple[str, str, str]]:
    """Read ``raw/financial_entities.csv`` as ``(alias, canonical, target)``.

    Longer aliases are applied first so a specific phrase wins over a prefix of
    itself. A bundle without a lexicon simply restores nothing.
    """

    raw_dir = resolve_stage_dir(bundle_root, "raw")
    candidates = sorted(raw_dir.glob("financial_entities*.csv")) if raw_dir.is_dir() else []
    if not candidates:
        return []
    path = next((p for p in candidates if p.name == "financial_entities.csv"), candidates[0])
    entries: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = (row.get("canonical") or "").strip()
            target_type = (row.get("target_type") or "").strip()
            if not canonical:
                continue
            # The canonical form counts as its own alias, so an entity already
            # written in English is recorded as a restoration too. It is listed
            # first so that on an equal-length tie it is tested before the
            # Bangla alias that would otherwise create it.
            for alias in [canonical, *(row.get("aliases") or "").split("|")]:
                alias = alias.strip()
                if alias:
                    entries.append((alias, canonical, target_type))
    return sorted(entries, key=lambda item: len(item[0]), reverse=True)


def restore_entities(
    text: str, lexicon: list[tuple[str, str, str]]
) -> tuple[str, list[dict[str, str]]]:
    """Replace known Bangla aliases with their canonical financial entity."""

    restorations: list[dict[str, str]] = []
    for alias, canonical, target_type in lexicon:
        # Word-bounded so an alias never fires inside an inflected form:
        # "আমদানি" restores on its own but leaves "আমদানিকৃত" untouched. Bangla
        # vowel signs are not \w characters, so "জ্বালানি খাতে" still matches --
        # which is also why \b cannot be used here: an alias ending in a vowel
        # sign would satisfy \b in the middle of a word.
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        text, count = re.subn(pattern, canonical, text)
        if count:
            restorations.append(
                {"alias": alias, "canonical": canonical, "target_type": target_type}
            )
    return text, restorations


def _asr_words(asr: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in asr.get("segments") or []:
        for word in segment.get("words") or []:
            words.append(
                {
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                    "word": str(word.get("word", "")),
                    "prob": word.get("prob"),
                }
            )
    return words


def assign_speakers(
    words: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach an episode-local speaker ID and the method used to each word."""

    intervals = [
        (float(s.get("start_sec", 0.0)), float(s.get("end_sec", 0.0)), int(s.get("speaker_id", -1)))
        for s in segments
    ]
    assigned: list[dict[str, Any]] = []
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        speaker, method = -1, "unassigned"
        for start, end, spk in intervals:
            if start <= midpoint <= end:
                speaker, method = spk, "midpoint"
                break
        else:
            best_overlap = 0.0
            for start, end, spk in intervals:
                overlap = min(word["end"], end) - max(word["start"], start)
                if overlap > best_overlap:
                    speaker, best_overlap = spk, overlap
            if best_overlap > 0:
                method = "max_overlap"
            else:
                speaker = -1
        assigned.append({**word, "speaker_id": speaker, "assignment_method": method})
    return assigned


def _starts_new_utterance(
    word: dict[str, Any], current: list[dict[str, Any]], utterance_start: float
) -> bool:
    previous = current[-1]
    if word["speaker_id"] != previous["speaker_id"]:
        return True
    if word["start"] - previous["end"] > MAX_PAUSE_SEC:
        return True
    if previous["word"].rstrip().endswith(SENTENCE_ENDINGS):
        return True
    return word["end"] - utterance_start > MAX_UTTERANCE_SEC


def build_utterances(
    words: list[dict[str, Any]],
    episode_id: str,
    lexicon: list[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Group speaker-assigned words into utterances using the recorded rule."""

    utterances: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        # ASR word tokens keep decomposed Bangla forms (য + ়); the exported
        # transcripts are composed (য়). Normalize so both match on search.
        text = unicodedata.normalize("NFC", "".join(item["word"] for item in current)).strip()
        normalized, restorations = restore_entities(text, lexicon or [])
        utterances.append(
            {
                "utterance_id": f"{episode_id}_utt{len(utterances):05d}",
                "episode_id": episode_id,
                "speaker_id": current[0]["speaker_id"],
                "start_sec": round(current[0]["start"], 2),
                "end_sec": round(current[-1]["end"], 2),
                "text_verbatim": text,
                "text_normalized": normalized,
                "words": [dict(item) for item in current],
                "entity_restorations": restorations,
            }
        )
        current.clear()

    for word in words:
        if current and _starts_new_utterance(word, current, current[0]["start"]):
            flush()
        current.append(word)
    flush()
    return utterances


def fuse_episode(bundle_root: Path, episode_id: str) -> dict[str, Any] | None:
    asr_path = resolve_stage_dir(bundle_root, "asr") / f"{episode_id}.json"
    diar_path = resolve_stage_dir(bundle_root, "diarization") / f"{episode_id}.json"
    if not asr_path.is_file() or not diar_path.is_file():
        return None
    asr = _load(asr_path)
    diarization = _load(diar_path)
    words = assign_speakers(_asr_words(asr), diarization.get("segments") or [])
    if not words:
        return None
    utterances = build_utterances(words, episode_id, load_entity_lexicon(bundle_root))
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "diarization_provider": diarization.get("provider", ""),
        "asr_model": asr.get("model", ""),
        "overlap_policy": OVERLAP_POLICY,
        "assignment": ASSIGNMENT,
        "words": words,
        "utterances": utterances,
        "stats": {
            "n_words": len(words),
            "n_utterances": len(utterances),
            "unassigned_words": sum(1 for word in words if word["speaker_id"] < 0),
            "coarse_tokens": 0,
        },
    }


def episode_ids(bundle_root: Path) -> list[str]:
    asr_dir = resolve_stage_dir(bundle_root, "asr")
    diar_dir = resolve_stage_dir(bundle_root, "diarization")
    if not asr_dir.is_dir():
        return []
    return sorted(
        path.stem for path in asr_dir.glob("*.json") if (diar_dir / f"{path.stem}.json").is_file()
    )


def _comparable(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Path to a result bundle")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare against existing fused files instead of writing",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing fused files")
    args = parser.parse_args()

    bundle_root = Path(args.bundle).expanduser().resolve()
    fused_dir = resolve_stage_dir(bundle_root, "fused")
    episodes = episode_ids(bundle_root)
    if not episodes:
        print(f"No episode has both an ASR and a diarization artifact under {bundle_root}")
        return 1

    failures = 0
    for episode_id in episodes:
        payload = fuse_episode(bundle_root, episode_id)
        target = fused_dir / f"{episode_id}.json"
        if payload is None:
            print(f"{episode_id}: no ASR words, skipped")
            continue
        stats = payload["stats"]
        if args.verify:
            if not target.is_file():
                print(f"{episode_id}: no stored fused file to verify against")
                failures += 1
                continue
            stored = _load(target)
            match = _comparable(stored) == _comparable(payload)
            failures += not match
            print(
                f"{episode_id}: {'MATCH' if match else 'DIFFERS'} "
                f"(stored {stored['stats']['n_words']}w/{stored['stats']['n_utterances']}u vs "
                f"rebuilt {stats['n_words']}w/{stats['n_utterances']}u)"
            )
            continue
        if target.exists() and not args.overwrite:
            print(f"{episode_id}: fused file already exists, left untouched")
            continue
        fused_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        coverage = 1 - stats["unassigned_words"] / stats["n_words"]
        print(
            f"{episode_id}: wrote {target.name} — {stats['n_words']:,} words, "
            f"{stats['n_utterances']:,} utterances, {coverage * 100:.2f}% assigned"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
