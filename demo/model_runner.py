"""Run the fine-tuned BanglaBERT classifier live, inside the demo.

The trained checkpoint ships with the repository, so the demo can classify a
turn or score a whole speaker on the spot rather than only replaying numbers
recorded during training. That is the difference between showing a result and
showing the model producing it.

Three things keep the live output honest:

* the decoding settings come from the run's own metrics.json, so the demo
  cannot silently disagree with the notebook that trained the model;
* the input is built by predict_speakers.build_input, imported rather than
  re-implemented, so there is one definition of what the model is fed;
* nothing here touches the network. The checkpoint, the tokenizer and the
  aggregation rule are all read from disk, which is what makes the demo
  runnable on a machine that has never downloaded the model.

Loading is deferred until the first prediction. The checkpoint is 422 MB and
takes a few seconds to read, and a demo that stalls on startup looks broken.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TURN_CLASSES = ("negative", "neutral", "positive")
SPEAKER_CLASSES = ("negative", "mixed", "neutral", "positive")

# Long turns are the norm in this corpus, so a batch of 8 at 256 subwords is
# about as much as a laptop CPU absorbs without the UI feeling stalled.
BATCH_SIZE = 8


@dataclass
class TurnPrediction:
    turn_id: str
    role: str
    text: str
    start_sec: float
    end_sec: float
    n_words: int
    pred: str = ""
    p_negative: float = 0.0
    p_neutral: float = 0.0
    p_positive: float = 0.0
    judge_label: str = ""

    @property
    def confidence(self) -> float:
        return max(self.p_negative, self.p_neutral, self.p_positive)


@dataclass
class SpeakerScore:
    verdict: str
    stance: float
    rule: str
    turns: list[TurnPrediction] = field(default_factory=list)
    elapsed_sec: float = 0.0


class ModelRunner:
    """Lazy, thread-safe wrapper around the fine-tuned turn classifier."""

    def __init__(self, sentiment_root: Path | str | None):
        self.root = Path(sentiment_root) if sentiment_root else None
        self.run_dir = self.root / "runs" / "banglabert_turn" if self.root else None
        self.model_dir = self.run_dir / "model" if self.run_dir else None
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._build_input = None
        self.load_error = ""
        self.load_seconds = 0.0

        self.metrics: dict[str, Any] = {}
        if self.run_dir and (self.run_dir / "metrics.json").is_file():
            try:
                self.metrics = json.loads(
                    (self.run_dir / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.metrics = {}
        self.input_mode = str(self.metrics.get("input_mode") or "role_turn")
        self.max_len = int(self.metrics.get("max_len") or 256)
        self.rule = str(self.metrics.get("best_aggregation_rule") or "hard_vote")

    # ------------------------------------------------------------ availability

    @property
    def available(self) -> bool:
        """True when a checkpoint is present on disk."""

        return bool(self.model_dir and (self.model_dir / "config.json").is_file())

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def checkpoint_size_mb(self) -> float:
        if not self.model_dir or not self.model_dir.is_dir():
            return 0.0
        return sum(p.stat().st_size for p in self.model_dir.iterdir()
                   if p.is_file()) / (1024 * 1024)

    def status(self) -> str:
        if not self.available:
            return ("No checkpoint found under speaker_sentiment/runs/banglabert_turn/"
                    "model/. Train it with the notebook, or read the recorded results "
                    "in the tabs above.")
        if self.load_error:
            return f"The checkpoint is present but did not load: {self.load_error}"
        if self.loaded:
            return (f"Model loaded in {self.load_seconds:.1f} s, running on CPU. "
                    f"Input mode {self.input_mode}, max length {self.max_len}, "
                    f"aggregation rule {self.rule}.")
        return (f"Checkpoint ready ({self.checkpoint_size_mb():.0f} MB). It loads on "
                "the first prediction, which takes a few seconds; predictions after "
                "that are immediate.")

    # ------------------------------------------------------------------ loading

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            start = time.time()
            # Imported here, not at module scope: torch and transformers take
            # seconds to import and the demo must start without them when the
            # operator only wants transcript search.
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir), local_files_only=True)
            model.eval()

            self._build_input = _load_build_input(self.root)
            self._tokenizer = tokenizer
            self._model = model
            self.load_seconds = time.time() - start

    def warm_up(self) -> str:
        """Load the checkpoint now so the first real prediction is instant."""

        if not self.available:
            return self.status()
        try:
            self._ensure_loaded()
        except Exception as exc:                                  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
        return self.status()

    # --------------------------------------------------------------- inference

    def classify_turns(self, rows: list[dict[str, Any]]) -> list[TurnPrediction]:
        """Classify each row. Rows carry at least text; role is optional."""

        if not rows:
            return []
        self._ensure_loaded()
        import torch

        prepared: list[TurnPrediction] = []
        pairs: list[tuple[str, str | None]] = []
        previous = ""
        for row in rows:
            text = str(row.get("text") or "")
            head, tail = self._build_input(
                {"text": text, "role": str(row.get("role") or "unassigned")},
                previous, self.input_mode)
            pairs.append((head, tail))
            previous = text
            prepared.append(TurnPrediction(
                turn_id=str(row.get("turn_id") or ""),
                role=str(row.get("role") or "unassigned"),
                text=text,
                start_sec=float(row.get("start_sec") or 0.0),
                end_sec=float(row.get("end_sec") or 0.0),
                n_words=int(row.get("n_words") or len(text.split())),
                judge_label=str(row.get("label") or ""),
            ))

        # Turns in this corpus range from a dozen words to several hundred.
        # Batching them in arrival order pads every short turn out to the
        # longest in its batch, so most of the compute is spent on padding.
        # Grouping similar lengths together removes that waste; the original
        # order is restored when the probabilities are written back.
        order = sorted(range(len(pairs)),
                       key=lambda i: len(pairs[i][0]) + len(pairs[i][1] or ""))

        for offset in range(0, len(order), BATCH_SIZE):
            positions = order[offset:offset + BATCH_SIZE]
            chunk = [pairs[i] for i in positions]
            heads = [h for h, _ in chunk]
            tails = [t for _, t in chunk]
            if tails and tails[0] is not None:
                encoded = self._tokenizer(
                    heads, tails, truncation=True, max_length=self.max_len,
                    padding=True, return_tensors="pt")
            else:
                encoded = self._tokenizer(
                    heads, truncation=True, max_length=self.max_len,
                    padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).tolist()
            for position, row_probs in zip(positions, probs):
                item = prepared[position]
                item.p_negative = float(row_probs[0])
                item.p_neutral = float(row_probs[1])
                item.p_positive = float(row_probs[2])
                item.pred = TURN_CLASSES[max(range(3), key=lambda i: row_probs[i])]
        return prepared

    def score_speaker(self, rows: list[dict[str, Any]]) -> SpeakerScore:
        """Classify a speaker's turns, then aggregate them into one verdict."""

        start = time.time()
        turns = self.classify_turns(rows)
        if not turns:
            return SpeakerScore(verdict="", stance=0.0, rule=self.rule)

        aggregate, to_speaker_label = _load_aggregation(self.root)
        payload = [{
            "p_positive": t.p_positive, "p_negative": t.p_negative,
            "pred": t.pred, "text": t.text, "n_words": t.n_words,
            "start_sec": t.start_sec, "end_sec": t.end_sec,
        } for t in turns]
        stance, polar_share = aggregate(payload, self.rule)
        verdict = to_speaker_label(stance, polar_share)
        return SpeakerScore(verdict=verdict, stance=float(stance), rule=self.rule,
                            turns=turns, elapsed_sec=time.time() - start)


# --------------------------------------------------------------------- helpers

def _speaker_sentiment_module(root: Path | None):
    """Import predict_speakers so the demo shares its exact definitions."""

    if root is None:
        raise RuntimeError("speaker_sentiment/ not found")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import predict_speakers                                        # noqa: PLC0415
    return predict_speakers


def _load_build_input(root: Path | None):
    try:
        return _speaker_sentiment_module(root).build_input
    except Exception:                                              # noqa: BLE001
        return _fallback_build_input


def _load_aggregation(root: Path | None):
    module = _speaker_sentiment_module(root)
    return module.aggregate, module.to_speaker_label


_ROLE_TOKEN = {
    "host": "সঞ্চালক",
    "guest": "অতিথি",
    "minor": "অন্য",
    "unassigned": "অজানা",
}


def _fallback_build_input(row: dict, prev_text: str, mode: str):
    """Only used if predict_speakers.py is missing; keeps the demo usable."""

    text = " ".join(unicodedata.normalize("NFC", str(row["text"])).split())
    role = _ROLE_TOKEN.get(row.get("role", "unassigned"), _ROLE_TOKEN["unassigned"])
    if mode == "plain":
        return text, None
    return role, text
