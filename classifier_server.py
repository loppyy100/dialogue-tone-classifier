#!/usr/bin/env python3
"""classifier_server.py — the love-chat classifier as a persistent server.

Loads the fine-tuned DistilBERT (model_bert_v3, 7 labels) ONCE, then reads
one chat message per line from stdin and prints one line per message:

    <label> <confidence>
    e.g.  bid 0.87

Blank lines are skipped. A line "__QUIT__" exits cleanly.
Prints "READY" on stdout once the model is loaded; any load error goes
to stderr and the process exits non-zero.

Why a server instead of one-shot predict_bert.py: loading the model takes
seconds, and a chat sends many messages. Load once, classify forever.

Run from this folder, with the venv that has torch + transformers
(on Tobi's laptop that is ~/pytorch-env — install transformers first
if it is missing:  ~/pytorch-env/bin/pip install transformers):

    ~/pytorch-env/bin/python classifier_server.py

Labels (model_bert_v3/labels.json): affection, bid, criticism, contempt,
defensiveness, stonewalling, repair. 86% val on the 7-label set.
"""

import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bert_v3")


def main() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        labels = json.load(open(os.path.join(MODEL_DIR, "labels.json")))
    except Exception as e:
        print(f"ERROR cannot load model: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    model.eval()
    print("READY", flush=True)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "__QUIT__":
            break
        enc = tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1)[0]
        best = int(probs.argmax())
        print(f"{labels[best]} {float(probs[best]):.2f}", flush=True)


if __name__ == "__main__":
    main()
