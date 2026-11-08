"""Use the trained love-chat classifier (v2).

Usage:
    python predict.py "i love you so much"
    python predict.py "fine." "look at this sunset" "you never listen"
    python predict.py            (no args = interactive, type messages, Ctrl+C to quit)

Output: the predicted vibe + the probability of every class.
A low top probability means the model is unsure — take it as a guess,
not a verdict. One message is never a diagnosis of a relationship.

Works with both v1 (bag of words) and v2 (BiLSTM + attention) model.pt files.
"""

import sys

import torch

from model import HORSEMEN, batch_to_tensors, build_model


def load_model(path="model.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    return model, ckpt["vocab"], ckpt["labels"], ckpt.get("arch", "bow")


def predict(model, vocab, labels, texts, arch):
    with torch.no_grad():
        if arch == "bilstm-attn":
            padded, lengths = batch_to_tensors(texts, vocab)
            logits = model(padded, lengths)
        else:
            # v1: EmbeddingBag needs flat indices + offsets
            indices, offsets = [], []
            pos = 0
            for t in texts:
                ids = [vocab.get(w, 0) for w in t.lower().split()]
                indices.extend(ids)
                offsets.append(pos)
                pos += len(ids)
            logits = model(torch.tensor(indices, dtype=torch.long),
                           torch.tensor(offsets, dtype=torch.long))
        return torch.softmax(logits, dim=1)


def show(model, vocab, labels, arch, text):
    probs = predict(model, vocab, labels, [text], arch)[0]
    best = int(probs.argmax())
    tag = "  <-- FOUR HORSEMAN" if labels[best] in HORSEMEN else ""
    print(f'\n"{text}"')
    print(f"  -> {labels[best]} ({probs[best]:.0%}){tag}")
    for i, label in enumerate(labels):
        bar = "#" * int(probs[i] * 30)
        print(f"     {label:14s} {probs[i]:.0%} {bar}")


def main():
    model, vocab, labels, arch = load_model()
    print(f"Model: {arch}")
    texts = sys.argv[1:]
    if texts:
        for t in texts:
            show(model, vocab, labels, arch, t)
        return
    print("Type a chat message, Enter to classify. Ctrl+C to quit.\n")
    try:
        while True:
            text = input("> ").strip()
            if text:
                show(model, vocab, labels, arch, text)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
