"""Backfill the annotation-draft stage for result bundles that lack it.

Stage 4 of the pipeline turns each fused episode into a review-ready draft.
One roll's Colab run ended before that stage, so its episodes reach the demo
with zero draft rows and the readiness audit fails. The draft is a pure,
deterministic projection of ``fused/`` -- no model and no judgement -- so it
can be rebuilt here rather than by re-running the GPU pipeline.

The row schema is copied from ``make_annotation_draft`` in the standalone
pipeline notebook so that a backfilled bundle is byte-comparable with one the
notebook produced. In particular ``human_verified`` stays ``False`` and
``annotation_status`` stays ``review_required``: this script creates work
items, never labels.

    python demo/make_missing_drafts.py --data-root .
    python demo/make_missing_drafts.py --data-root . --apply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from search_core import _discover_bundle_roots, resolve_stage_dir

DRAFT_SCHEMA_VERSION = "2.0"


def draft_from_fused(fused: dict) -> dict:
    rows = []
    for utterance in fused.get("utterances", []):
        rows.append(
            {
                "utterance_id": utterance["utterance_id"],
                "episode_id": utterance["episode_id"],
                "speaker_id": utterance["speaker_id"],
                "role_tag": "",
                "start_sec": utterance["start_sec"],
                "end_sec": utterance["end_sec"],
                "text_verbatim": utterance["text_verbatim"],
                "text_normalized": utterance["text_normalized"],
                "labels": [],
                "annotation_status": "review_required",
                "transcript_source": "asr_fused_draft",
                "speaker_source": "diarization_draft",
                "human_verified": False,
                "annotator_id": "",
                "notes": "",
            }
        )
    return {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "episode_id": fused["episode_id"],
        "is_gold_tier": False,
        "overlap_policy": fused.get("overlap_policy", "first-speaker-wins"),
        "utterances": rows,
    }


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def draft_dir_for(bundle_root: Path) -> Path:
    """Where this bundle's drafts belong.

    Rolls disagree on the folder name (``annotation_drafts`` in the notebook,
    ``drafts`` in one hand-made layout). An existing directory wins so a
    backfill never creates a second, competing location; otherwise the
    notebook's name is used.
    """

    existing = resolve_stage_dir(bundle_root, "annotation_drafts")
    if existing.is_dir():
        return existing
    legacy = bundle_root / "drafts"
    if legacy.is_dir():
        return legacy
    fused = resolve_stage_dir(bundle_root, "fused")
    return fused.parent / "annotation_drafts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the drafts; without it the script only reports what is missing",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    written = 0
    for bundle_root in _discover_bundle_roots(data_root):
        fused_dir = resolve_stage_dir(bundle_root, "fused")
        drafts_dir = draft_dir_for(bundle_root)
        for fused_path in sorted(fused_dir.glob("*.json")):
            target = drafts_dir / fused_path.name
            if target.is_file():
                continue
            fused = json.loads(fused_path.read_text(encoding="utf-8"))
            draft = draft_from_fused(fused)
            rel = target.relative_to(data_root) if target.is_relative_to(data_root) else target
            print(f"{'write' if args.apply else 'missing'}: {rel} ({len(draft['utterances'])} rows)")
            if args.apply:
                atomic_json(target, draft)
            written += 1

    if not written:
        print("every fused episode already has an annotation draft")
    elif not args.apply:
        print(f"\n{written} draft(s) missing; re-run with --apply to write them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
