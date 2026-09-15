"""Stage 4b: fine-tune BanglaBERT for target-aware polarity.

Not yet executed on this machine. The run configuration in the project's own
run manifest records run_sentiment_training=false, so no checkpoint has ever
been produced for this dataset. This file is the training code that closes
that gap; the numbers reported in the deck come from the rule labeller and the
character n-gram baseline, which have been run.

Model: csebuetnlp/banglabert (ELECTRA discriminator, 110M parameters), the
checkpoint already named in the project run configuration.

Input is a sentence pair, which is what makes the task target-aware rather
than sentence-level:

    [CLS] <target surface form> [SEP] <utterance> [SEP]

The same utterance is fed once per target, so one sentence can be negative
about inflation and positive about reserves at the same time.

    python sentiment/train_banglabert.py --epochs 4 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["negative", "non_evaluative", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
MODEL_NAME = "csebuetnlp/banglabert"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def target_text(row: dict) -> str:
    """The Bangla surface form actually present in the utterance, not the
    English canonical name. The model has never seen the canonical strings."""
    return " ".join(dict.fromkeys(m["surface"] for m in row["mentions"]))


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int, label_field: str):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_field = label_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        enc = self.tokenizer(
            target_text(row), row["text"],
            truncation=True, max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[row[self.label_field]])
        return item


def class_weights(rows: list[dict], field: str) -> torch.Tensor:
    """The corpus is 86 percent non evaluative. Unweighted, the model reaches
    a high accuracy by predicting that class for everything and scores a
    macro-F1 near 0.31."""
    counts = np.array([sum(1 for r in rows if r[field] == name) for name in LABELS], dtype=float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(model(**batch).logits.argmax(-1).cpu().numpy())
        batch["labels"] = labels
    return np.concatenate(out) if out else np.array([])


def evaluate(model, rows, tokenizer, args, device, field: str):
    subset = [r for r in rows if r.get(field)]
    if not subset:
        return None, None
    loader = DataLoader(PairDataset(subset, tokenizer, args.max_len, field),
                        batch_size=args.eval_batch_size)
    pred = [LABELS[i] for i in predict(model, loader, device)]
    true = [r[field] for r in subset]
    return f1_score(true, pred, average="macro", labels=LABELS, zero_division=0), (true, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sentiment/data")
    ap.add_argument("--out", default="sentiment/runs/banglabert")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128, help="p99 of this corpus is 108 tokens")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    set_seed(args.seed)
    data = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (load(data / f"{s}.jsonl") for s in ("train", "val", "test"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS)).to(device)

    train_loader = DataLoader(
        PairDataset(train, tokenizer, args.max_len, "silver_label"),
        batch_size=args.batch_size, shuffle=True)

    steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * args.warmup_ratio), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train, "silver_label").to(device))

    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            running += loss.item()

        # Model selection uses the silver validation score, because the gold
        # validation slice is only 22 pairs and far too small to choose on.
        val_silver, _ = evaluate(model, val, tokenizer, args, device, "silver_label")
        val_gold, _ = evaluate(model, val, tokenizer, args, device, "gold_label")
        history.append({
            "epoch": epoch, "train_loss": round(running / max(len(train_loader), 1), 4),
            "val_silver_macro_f1": round(val_silver or 0, 4),
            "val_gold_macro_f1": round(val_gold or 0, 4),
        })
        print(history[-1])
        if (val_silver or 0) > best:
            best = val_silver or 0
            model.save_pretrained(out / "best"); tokenizer.save_pretrained(out / "best")

    model = AutoModelForSequenceClassification.from_pretrained(out / "best").to(device)

    # The headline number: human labels, on episodes the model never saw.
    test_gold, pair = evaluate(model, test, tokenizer, args, device, "gold_label")
    test_silver, _ = evaluate(model, test, tokenizer, args, device, "silver_label")
    report = {
        "model": MODEL_NAME, "history": history,
        "test_gold_macro_f1": round(test_gold or 0, 4),
        "test_silver_macro_f1": round(test_silver or 0, 4),
    }
    if pair:
        true, pred = pair
        print("\nBANGLABERT on held-out episodes, scored against human labels\n")
        print(classification_report(true, pred, labels=LABELS, digits=3, zero_division=0))
        report["test_gold_report"] = classification_report(
            true, pred, labels=LABELS, digits=3, zero_division=0, output_dict=True)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
