"""Structural readiness audit for the supervisor demonstration corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from search_core import CorpusIndex
from showcase_core import ProjectShowcase


def audit(index: CorpusIndex, expected_episodes: int | None, minimum_assignment: float) -> dict:
    summary = index.corpus_summary()
    showcase = ProjectShowcase(index)
    stage_rows = showcase.stage_rows()
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, value: object, requirement: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "requirement": requirement,
            }
        )

    if expected_episodes is not None:
        add(
            "episode_count",
            summary["episodes"] == expected_episodes,
            summary["episodes"],
            f"exactly {expected_episodes}",
        )
    add(
        "fused_utterances",
        summary["utterances"] > 0,
        summary["utterances"],
        "at least one fused utterance",
    )
    add(
        "speaker_assignment",
        summary["assignment_rate"] >= minimum_assignment,
        round(summary["assignment_rate"], 6),
        f">= {minimum_assignment:.1%}",
    )
    add(
        "audio_availability",
        summary["audio_available"] == summary["episodes"],
        f"{summary['audio_available']}/{summary['episodes']}",
        "audio available for every episode",
    )
    add(
        "diarization_outputs",
        all(row["Diarization"] == "Ready" for row in stage_rows),
        f"{sum(row['Diarization'] == 'Ready' for row in stage_rows)}/{summary['episodes']}",
        "non-empty diarization output for every episode",
    )
    add(
        "asr_outputs",
        all(row["ASR"] == "Ready" for row in stage_rows),
        f"{sum(row['ASR'] == 'Ready' for row in stage_rows)}/{summary['episodes']}",
        "non-empty ASR output for every episode",
    )
    add(
        "fusion_outputs",
        all(row["Fusion"] == "Ready" for row in stage_rows),
        f"{sum(row['Fusion'] == 'Ready' for row in stage_rows)}/{summary['episodes']}",
        "non-empty fused output for every episode",
    )
    add(
        "annotation_drafts",
        all(int(row["Draft rows"]) > 0 for row in stage_rows),
        sum(int(row["Draft rows"]) for row in stage_rows),
        "review-ready draft rows for every episode",
    )
    empty = [record.key for record in index.records if not record.normalized_text]
    add("nonempty_transcripts", not empty, len(empty), "zero empty utterances")
    duplicate_count = len(index.records) - len({record.key for record in index.records})
    add("unique_result_keys", duplicate_count == 0, duplicate_count, "zero duplicate keys")
    unknown_diarization = [
        episode.global_episode_id
        for episode in index.episodes.values()
        if not episode.diarization_provider
    ]
    add(
        "diarization_provenance",
        not unknown_diarization,
        len(unknown_diarization),
        "provider recorded for every episode",
    )
    return {
        "ready": all(item["status"] == "PASS" for item in checks),
        "summary": summary,
        "contributors": index.contributor_rows(),
        "checks": checks,
        "episodes": index.episode_rows(),
        "pipeline_showcase": showcase.stage_summary_rows(),
        "artifact_summary": showcase.artifact_summary_rows(),
        "evaluation_gate_count": len(showcase.gate_rows()),
        "exploratory_profile_rows": len(showcase.profile_rows()),
        "target_taxonomy_rows": len(showcase.target_policy_rows()),
        "financial_entity_rows": len(showcase.entity_rows()),
        "loader_warnings": list(dict.fromkeys(index.warnings)),
        "interpretation": (
            "Structural readiness only. PASS does not establish WER, DER, sentiment F1, "
            "speaker identity, or human-verified correctness."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a roll-aware consolidated demo corpus")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--minimum-assignment", type=float, default=0.95)
    parser.add_argument("--report", help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = CorpusIndex(args.data_root)
    report = audit(index, args.expected_episodes, args.minimum_assignment)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        path = Path(args.report).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        print(f"Report written: {path}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
