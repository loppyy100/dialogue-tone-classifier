"""Use the fine-tuned DistilBERT love-chat classifier (v3).

Usage:
    python predict_bert.py "i love you so much"
    python predict_bert.py "fine." "look at this sunset"
    python predict_bert.py --model model_bert_v3 "i'm sorry, can we try again"
    python predict_bert.py            (no args = interactive, Ctrl+C to quit)

Needs model_bert/ (6 labels) or model_bert_v3/ (7 labels, incl. repair) —
run train_bert.py first if the folder is missing. Labels are read from the
model folder's labels.json when present.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import HORSEMEN, LABELS


def load_model(path="model_bert"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    labels_path = os.path.join(path, "labels.json")
    labels = json.load(open(labels_path)) if os.path.exists(labels_path) else LABELS
    return model, tokenizer, labels


def show(model, tokenizer, labels, text):
    enc = tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    best = int(probs.argmax())
    tag = "  <-- FOUR HORSEMAN" if labels[best] in HORSEMEN else ""
    print(f'\n"{text}"')
    print(f"  -> {labels[best]} ({probs[best]:.0%}){tag}")
    for i, label in enumerate(labels):
        bar = "#" * int(probs[i] * 30)
        print(f"     {label:14s} {probs[i]:.0%} {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_bert")
    parser.add_argument("texts", nargs="*")
    args = parser.parse_args()
    try:
        model, tokenizer, labels = load_model(args.model)
    except Exception:
        print(f"No {args.model}/ found. Run this first:\n"
              "    python train_bert.py")
        return
    print(f"Model: distilbert (fine-tuned), {len(labels)} labels")
    texts = args.texts
    if texts:
        for t in texts:
            show(model, tokenizer, labels, t)
        return
    print("Type a chat message, Enter to classify. Ctrl+C to quit.\n")
    try:
        while True:
            text = input("> ").strip()
            if text:
                show(model, tokenizer, labels, text)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
