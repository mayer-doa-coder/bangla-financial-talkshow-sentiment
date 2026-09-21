"""Read-only view over the finished speaker-sentiment corpus and model run.

The retrieval demo indexes the *pipeline* bundles under each roll folder. The
sentiment work lives elsewhere -- one consolidated corpus in
``speaker_sentiment/data/`` plus a training run in
``speaker_sentiment/runs/`` -- and is keyed by speaker turn rather than by the
fixed ASR decode window the demo calls an utterance. This module loads those
artifacts and exposes them in the demo's own identity terms
(``<roll>::<episode>`` and an episode-local ``speaker_id``).

Nothing here computes a label. It reports what the judge recorded and what the
trained model predicted, and it keeps the two clearly apart, because every
score in the run is agreement with an LLM judge rather than with human truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

TURN_CLASSES = ("negative", "neutral", "positive")
SPEAKER_CLASSES = ("negative", "mixed", "neutral", "positive")

# A turn label is attached to a decode window only when the window sits inside
# the turn. Both come from the same fusion output, so the bounds agree closely;
# the tolerance only absorbs rounding in the turn reconstruction.
BOUNDARY_TOLERANCE_SEC = 0.5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def find_sentiment_root(data_root: Path) -> Path | None:
    """Locate ``speaker_sentiment/`` near the demo's data root.

    The demo is pointed at the folder holding the roll directories, which is
    normally the repository root, but it is also run against a Drive copy one
    level down. Searching a couple of parents keeps both working without a
    second command-line flag.
    """

    seen: set[Path] = set()
    for base in (data_root, *data_root.parents[:3], Path(__file__).resolve().parent.parent):
        candidate = (base / "speaker_sentiment").resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "speakers_labelled.jsonl").is_file():
            return candidate
    return None


@dataclass
class TurnLabel:
    turn_id: str
    global_episode: str
    speaker_id: int
    role: str
    start_sec: float
    end_sec: float
    label: str
    judge_confidence: str
    evidence: str
    split: str
    text: str = ""
    predicted: str = ""
    prediction_confidence: float = 0.0

    def text_preview(self, limit: int = 220) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class SpeakerVerdict:
    speaker_key: str
    global_episode: str
    speaker_id: int
    role: str
    judge_label: str
    stance_score: float
    judge_confidence: str
    transcript_quality: str
    n_turns: int
    n_words: int
    n_negative: int
    n_neutral: int
    n_positive: int
    dominant_topics: list[str] = field(default_factory=list)
    summary: str = ""
    rationale: str = ""
    split: str = ""


class SentimentLayer:
    """Speaker verdicts, turn labels and model predictions, if they exist."""

    def __init__(self, data_root: str | Path):
        self.root = find_sentiment_root(Path(data_root).expanduser().resolve())
        self.warnings: list[str] = []
        self.speakers: dict[tuple[str, int], SpeakerVerdict] = {}
        self.turns_by_episode: dict[str, list[TurnLabel]] = {}
        self.metrics: dict[str, Any] = {}
        self.baseline_metrics: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        if self.root is None:
            self.warnings.append(
                "speaker_sentiment/ not found; the demo shows pipeline evidence only."
            )
            return
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        assert self.root is not None
        data = self.root / "data"

        for row in _read_jsonl(data / "speakers_labelled.jsonl"):
            try:
                speaker_id = int(row["speaker_id"])
            except (KeyError, TypeError, ValueError):
                continue
            episode = str(row.get("global_episode", ""))
            if not episode:
                continue
            self.speakers[(episode, speaker_id)] = SpeakerVerdict(
                speaker_key=str(row.get("speaker_key", "")),
                global_episode=episode,
                speaker_id=speaker_id,
                role=str(row.get("role", "")),
                judge_label=str(row.get("judge_label", "")),
                stance_score=float(row.get("judge_stance_score") or 0.0),
                judge_confidence=str(row.get("judge_confidence", "")),
                transcript_quality=str(row.get("transcript_quality", "")),
                n_turns=int(row.get("n_turns") or 0),
                n_words=int(row.get("n_words") or 0),
                n_negative=int(row.get("n_negative") or 0),
                n_neutral=int(row.get("n_neutral") or 0),
                n_positive=int(row.get("n_positive") or 0),
                dominant_topics=list(row.get("dominant_topics") or []),
                summary=str(row.get("summary", "")),
                rationale=str(row.get("rationale", "")),
                split=str(row.get("split", "")),
            )

        # Model predictions exist for the held-out test turns only. Merging
        # them in by turn_id keeps a prediction beside its judge label without
        # implying the other turns were scored.
        predictions: dict[str, tuple[str, float]] = {}
        for row in _read_jsonl(self.root / "runs" / "banglabert_turn" / "test_predictions.jsonl"):
            turn_id = str(row.get("turn_id", ""))
            predicted = str(row.get("pred", ""))
            if not turn_id or not predicted:
                continue
            confidence = 0.0
            for name in TURN_CLASSES:
                value = row.get(f"p_{name}")
                if value is not None:
                    confidence = max(confidence, float(value))
            predictions[turn_id] = (predicted, confidence)

        for row in _read_jsonl(data / "turns_labelled.jsonl"):
            episode = str(row.get("global_episode", ""))
            turn_id = str(row.get("turn_id", ""))
            if not episode or not turn_id:
                continue
            predicted, confidence = predictions.get(turn_id, ("", 0.0))
            self.turns_by_episode.setdefault(episode, []).append(
                TurnLabel(
                    turn_id=turn_id,
                    global_episode=episode,
                    speaker_id=int(row.get("speaker_id") or -1),
                    role=str(row.get("role", "")),
                    start_sec=float(row.get("start_sec") or 0.0),
                    end_sec=float(row.get("end_sec") or 0.0),
                    label=str(row.get("label", "")),
                    judge_confidence=str(row.get("judge_confidence", "")),
                    evidence=str(row.get("evidence", "")),
                    split=str(row.get("split", "")),
                    text=str(row.get("text", "")),
                    predicted=predicted,
                    prediction_confidence=confidence,
                )
            )
        for turns in self.turns_by_episode.values():
            turns.sort(key=lambda turn: turn.start_sec)

        self.metrics = _read_json(self.root / "runs" / "banglabert_turn" / "metrics.json")
        self.baseline_metrics = _read_json(self.root / "runs" / "tfidf_baseline" / "metrics.json")
        history_path = self.root / "runs" / "banglabert_turn" / "history.csv"
        if history_path.is_file():
            import csv

            with history_path.open(encoding="utf-8-sig", newline="") as handle:
                self.history = list(csv.DictReader(handle))

    # ----------------------------------------------------------------- query

    @property
    def available(self) -> bool:
        return bool(self.speakers)

    @property
    def has_model_run(self) -> bool:
        return bool(self.metrics)

    def verdict(self, global_episode: str, speaker_id: int) -> SpeakerVerdict | None:
        return self.speakers.get((global_episode, int(speaker_id)))

    def turn_for(self, global_episode: str, speaker_id: int, start_sec: float, end_sec: float
                 ) -> TurnLabel | None:
        """The labelled turn containing this decode window, if there is one."""

        midpoint = (float(start_sec) + float(end_sec)) / 2.0
        for turn in self.turns_by_episode.get(global_episode, ()):
            if turn.speaker_id != int(speaker_id):
                continue
            if (turn.start_sec - BOUNDARY_TOLERANCE_SEC
                    <= midpoint
                    <= turn.end_sec + BOUNDARY_TOLERANCE_SEC):
                return turn
        return None

    def labelled_episodes(self) -> set[str]:
        return {verdict.global_episode for verdict in self.speakers.values()}

    def coverage(self, episode_keys: Iterable[str]) -> dict[str, Any]:
        keys = set(episode_keys)
        labelled = self.labelled_episodes() & keys
        turn_labels = sum(len(turns) for turns in self.turns_by_episode.values())
        return {
            "episodes_indexed": len(keys),
            "episodes_with_labels": len(labelled),
            "speaker_units": len(self.speakers),
            "turn_labels": turn_labels,
            "scored_turns": sum(
                1
                for turns in self.turns_by_episode.values()
                for turn in turns
                if turn.predicted
            ),
        }

    def speaker_rows(self) -> list[dict[str, Any]]:
        rows = []
        for verdict in sorted(
            self.speakers.values(), key=lambda item: (item.global_episode, item.speaker_id)
        ):
            rows.append(
                {
                    "Episode": verdict.global_episode,
                    "Speaker": f"Speaker {verdict.speaker_id}",
                    "Role": verdict.role,
                    "Judge verdict": verdict.judge_label,
                    "Stance": round(verdict.stance_score, 3),
                    "Confidence": verdict.judge_confidence,
                    "Turns": verdict.n_turns,
                    "Words": verdict.n_words,
                    "Neg/Neu/Pos": f"{verdict.n_negative}/{verdict.n_neutral}/{verdict.n_positive}",
                    "Transcript quality": verdict.transcript_quality,
                    "Split": verdict.split,
                    "Topics": ", ".join(verdict.dominant_topics[:4]),
                }
            )
        return rows

    def model_rows(self) -> list[dict[str, Any]]:
        """Headline numbers for the trained model and its baselines."""

        if not self.metrics:
            return []
        metrics = self.metrics
        rows = [
            {
                "Measure": "Turn sentiment, macro-F1 (test)",
                "Value": _fmt(metrics.get("test_turn_macro_f1")),
                "Note": "BanglaBERT fine-tuned on role + turn text",
            },
            {
                "Measure": "Turn sentiment, macro-F1 (validation)",
                "Value": _fmt(metrics.get("val_turn_macro_f1")),
                "Note": "used for model selection; never for the headline",
            },
            {
                "Measure": "Majority-class baseline, macro-F1 (test)",
                "Value": _fmt(metrics.get("majority_baseline_test_macro_f1")),
                "Note": "floor: always predict the most frequent class",
            },
        ]
        baseline_f1 = self.baseline_metrics.get("test_macro_f1")
        if baseline_f1 is not None:
            rows.append(
                {
                    "Measure": "TF-IDF + linear baseline, macro-F1 (test)",
                    "Value": _fmt(baseline_f1),
                    "Note": "bag-of-words control on the same split",
                }
            )
        rows.append(
            {
                "Measure": "Best speaker aggregation rule",
                "Value": str(metrics.get("best_aggregation_rule", "n/a")),
                "Note": f"selected on {metrics.get('aggregation_rule_selected_on', 'n/a')}",
            }
        )
        return rows

    def training_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.history]

    def test_turns(self) -> list[TurnLabel]:
        return [
            turn
            for turns in self.turns_by_episode.values()
            for turn in turns
            if turn.predicted
        ]

    def confusion_rows(self) -> list[dict[str, Any]]:
        """Judge label (rows) against model prediction (columns) on the test split."""

        turns = self.test_turns()
        if not turns:
            return []
        rows = []
        for actual in TURN_CLASSES:
            row: dict[str, Any] = {"Judge ↓ / Model →": actual}
            subset = [turn for turn in turns if turn.label == actual]
            for predicted in TURN_CLASSES:
                row[predicted] = sum(1 for turn in subset if turn.predicted == predicted)
            row["total"] = len(subset)
            rows.append(row)
        return rows

    def per_class_rows(self) -> list[dict[str, Any]]:
        turns = self.test_turns()
        if not turns:
            return []
        rows = []
        for name in TURN_CLASSES:
            true_positive = sum(
                1 for turn in turns if turn.label == name and turn.predicted == name
            )
            predicted_count = sum(1 for turn in turns if turn.predicted == name)
            actual_count = sum(1 for turn in turns if turn.label == name)
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / actual_count if actual_count else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append(
                {
                    "Class": name,
                    "Support": actual_count,
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1": round(f1, 3),
                }
            )
        return rows

    def example_rows(self, correct: bool, limit: int = 8) -> list[dict[str, Any]]:
        """Most confident correct predictions, or most confident mistakes."""

        turns = [
            turn
            for turn in self.test_turns()
            if (turn.predicted == turn.label) is correct
        ]
        turns.sort(key=lambda turn: turn.prediction_confidence, reverse=True)
        return [
            {
                "Episode": turn.global_episode,
                "Speaker": f"Speaker {turn.speaker_id}",
                "Role": turn.role,
                "Judge": turn.label,
                "Model": turn.predicted,
                "Model confidence": round(turn.prediction_confidence, 3),
                "Turn text": turn.text_preview(),
            }
            for turn in turns[:limit]
        ]

    def caveat(self) -> str:
        return str(
            self.metrics.get(
                "caveat",
                "Scores measure agreement with the LLM judge, not with human ground truth.",
            )
        )

    def label_source(self) -> str:
        return str(self.metrics.get("label_source", "LLM-as-judge; no human labels"))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"
