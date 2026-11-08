"""Fine-tune DistilBERT on the love-chat data (v3).

This is transfer learning: instead of training from nothing, we start from a
transformer that already read billions of words (Wikipedia + books) and just
teach it our 6 labels. That is why it understands sarcasm, word order and
context far better than the from-scratch v1/v2 nets.

Usage:
    python train_bert.py              downloads distilbert-base-uncased
                                      (~250 MB, first run only), fine-tunes,
                                      saves to model_bert/
    python train_bert.py --epochs 5 --lr 2e-5

What it does:
  - Same data (data.tsv) and same 80/20 stratified split (seed 42) as v1/v2,
    so the val accuracy is directly comparable.
  - Tokenizer: WordPiece — splits words into pieces ("stonewalling" ->
    "stone", "##wall", "##ing"), so it never chokes on a word it hasn't seen.
  - Training: AdamW, lr 2e-5 (small! we are nudging a pretrained brain, not
    building one), linear warmup + decay, batch 16, max length 32 tokens.
  - Early stopping on val accuracy, best model saved.

Needs internet the first time (HuggingFace download). On Tobi's laptop: turn
WARP on first, it is a foreign site.
"""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from model import HORSEMEN, LABELS as LABELS_6
from train import load_data, stratified_split

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 32


class ChatDataset(Dataset):
    def __init__(self, rows, tokenizer, labels):
        self.rows = rows
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, label = self.rows[i]
        enc = self.tokenizer(
            text, truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels.index(label), dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, rows, tokenizer, labels, device="cpu"):
    model.eval()
    ds = ChatDataset(rows, tokenizer, labels)
    loader = DataLoader(ds, batch_size=32)
    correct, total = 0, 0
    per_class_correct = {l: 0 for l in labels}
    per_class_total = {l: 0 for l in labels}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=1).cpu()
        targets = batch["labels"].cpu()
        correct += int((preds == targets).sum())
        total += len(targets)
        for p, t in zip(preds.tolist(), targets.tolist()):
            per_class_total[labels[t]] += 1
            if p == t:
                per_class_correct[labels[t]] += 1
    return correct / total, per_class_correct, per_class_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    # v3 data: --data data_v3.tsv --labels affection,bid,criticism,contempt,defensiveness,stonewalling,repair --out model_bert_v3
    parser.add_argument("--data", default="data.tsv")
    parser.add_argument("--labels", default=",".join(LABELS_6))
    parser.add_argument("--out", default="model_bert")
    args = parser.parse_args()
    labels = args.labels.split(",")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cpu"  # laptop has no NVIDIA GPU; DistilBERT still trains fine on CPU

    rows = load_data(args.data)
    print(f"Loaded {len(rows)} examples from {args.data}")
    print(f"Labels ({len(labels)}): {labels}")
    train_rows, val_rows = stratified_split(rows)
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    print(f"Loading {MODEL_NAME} (downloads ~250 MB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels)
    )
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"(v2 BiLSTM was ~0.2M)")

    train_ds = ChatDataset(train_rows, tokenizer, labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_scheduler("linear", opt, int(0.1 * steps), steps)

    best_acc, best_epoch, bad = 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            total_loss += out.loss.item()
        val_acc, _, _ = evaluate(model, val_rows, tokenizer, labels, device)
        print(f"epoch {epoch}  loss {total_loss / len(train_loader):.4f}  "
              f"val acc {val_acc:.2%}  (best {max(best_acc, val_acc):.2%})")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            bad = 0
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            with open(os.path.join(args.out, "labels.json"), "w") as f:
                json.dump(labels, f)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    # Reload the best and report per-class accuracy
    model = AutoModelForSequenceClassification.from_pretrained(args.out)
    val_acc, pc, pt = evaluate(model, val_rows, tokenizer, labels, device)
    print(f"\nBest val accuracy: {val_acc:.2%}  (epoch {best_epoch})")
    print("Per-class accuracy (val):")
    for label in labels:
        n = pt[label]
        print(f"  {label:14s} {pc[label]}/{n} = {pc[label] / n:.0%}")
    print(f"\nSaved {args.out}/ (best-on-val weights + tokenizer + labels.json)")


if __name__ == "__main__":
    main()
