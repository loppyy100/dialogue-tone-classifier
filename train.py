"""Train the love-chat classifier (v2: BiLSTM + attention).

Usage:
    python train.py                    trains on data.tsv, saves model.pt
    python train.py --epochs 80 --hidden-dim 96 --dropout 0.4

What is new in v2 training:
  - Class weights: rare/hard classes count more in the loss, so the net
    can't just ignore stonewalling and contempt (the two weakest in v1).
  - AdamW + weight decay: penalises big weights, fights memorisation.
  - Early stopping: after every epoch we check val accuracy and keep the
    BEST model seen so far. Training stops when val hasn't improved for
    `patience` epochs. The saved model is the best one, not the last one.

Split: stratified 80/20 train/val, fixed seed so numbers are reproducible.
"""

import argparse
import random
from collections import Counter, defaultdict

import torch
import torch.nn as nn

from model import (
    LABELS,
    LoveChatClassifier,
    batch_to_tensors,
    build_vocab,
)


def load_data(path="data.tsv"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            text, label = line.rsplit("\t", 1)
            rows.append((text, label))
    return rows


def stratified_split(rows, val_frac=0.2, seed=42):
    """Split so every class keeps the same share in train and val."""
    by_label = defaultdict(list)
    for text, label in rows:
        by_label[label].append(text)
    rng = random.Random(seed)
    train, val = [], []
    for label, texts in by_label.items():
        texts = texts[:]
        rng.shuffle(texts)
        n_val = max(1, int(len(texts) * val_frac))
        val.extend((t, label) for t in texts[:n_val])
        train.extend((t, label) for t in texts[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def evaluate(model, rows, vocab):
    model.eval()
    texts = [t for t, _ in rows]
    targets = torch.tensor([LABELS.index(l) for _, l in rows], dtype=torch.long)
    with torch.no_grad():
        padded, lengths = batch_to_tensors(texts, vocab)
        logits = model(padded, lengths)
        preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    return acc, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = load_data()
    print(f"Loaded {len(rows)} examples")
    train, val = stratified_split(rows)
    print(f"Train: {len(train)}  Val: {len(val)}")

    vocab = build_vocab([t for t, _ in rows])
    print(f"Vocab size: {len(vocab)}")

    model = LoveChatClassifier(
        len(vocab), embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    )

    # Class weights: a class with fewer examples counts more per example.
    counts = Counter(l for _, l in train)
    weights = torch.tensor(
        [len(train) / (len(LABELS) * counts[l]) for l in LABELS],
        dtype=torch.float,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    train_texts = [t for t, _ in train]
    train_targets = torch.tensor([LABELS.index(l) for _, l in train], dtype=torch.long)
    train_padded, train_lengths = batch_to_tensors(train_texts, vocab)

    best_acc, best_state, best_epoch = 0.0, None, 0
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(train_padded, train_lengths)
        loss = loss_fn(logits, train_targets)
        loss.backward()
        opt.step()

        val_acc, _, _ = evaluate(model, val, vocab)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or bad_epochs == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  "
                  f"val acc {val_acc:.2%}  (best {best_acc:.2%} @ {best_epoch})")
        if bad_epochs >= args.patience:
            print(f"Early stop at epoch {epoch} "
                  f"(no val improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    val_acc, preds, targets = evaluate(model, val, vocab)
    print(f"\nBest val accuracy: {val_acc:.2%}  (epoch {best_epoch})")
    print("\nPer-class accuracy (val):")
    for i, label in enumerate(LABELS):
        mask = targets == i
        n = int(mask.sum())
        if n:
            c = int(((preds == targets) & mask).sum())
            print(f"  {label:14s} {c}/{n} = {c / n:.0%}")

    torch.save(
        {
            "arch": "bilstm-attn",
            "state_dict": best_state,
            "vocab": vocab,
            "labels": LABELS,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "val_accuracy": val_acc,
            "best_epoch": best_epoch,
        },
        "model.pt",
    )
    print("\nSaved model.pt (best-on-val weights)")


if __name__ == "__main__":
    main()
