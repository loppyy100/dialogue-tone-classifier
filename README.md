# Dialogue Tone Classifier 💬🌀

A fine-tuned DistilBERT model that reads **one chat message** and guesses its
emotional tone — built from relationship research (Gottman's Four Horsemen,
bids for connection, and repair attempts).

## Results (v3)

| | |
|---|---|
| Model | DistilBERT (`distilbert-base-uncased`), fine-tuned |
| Data | **405 hand-written examples**, 7 classes |
| Validation accuracy | **86% (68/79)** |
| Weakest class | repair — **50%** (only 30 examples) |

## The 7 classes

| Class | What it means | From the research |
|---|---|---|
| `affection` | warmth, love, gratitude | The 5:1 magic ratio — stable couples have ~5 positive interactions per 1 negative |
| `bid` | bid for connection ("look at this", "guess what") | Couples still together at 6 years turned toward bids **86%** of the time; divorced couples **33%** |
| `criticism` | "you always / you never", attacking character | Four Horsemen #1 |
| `contempt` | sarcasm, mockery, superiority ("nice one genius") | Four Horsemen #2 — **the single strongest predictor of breakups** |
| `defensiveness` | "not my fault", "well you did it too" | Four Horsemen #3 |
| `stonewalling` | shutting down ("fine.", "whatever", "k") | Four Horsemen #4 |
| `repair` | trying to de-escalate and reconnect | Repair attempts are what keep a conflict from spiralling |

All training data is hand-written seed data, **not real private chats**.

## Files

- `data_v3.tsv` — 405 hand-written examples, 7 classes
- `train_bert.py` — stratified train/val split, fine-tunes DistilBERT, early stopping on val accuracy
- `predict_bert.py` — classify messages from the command line or interactively
- `model_bert_v3/` — tokenizer + config for the fine-tuned model (weights below)
- `classifier_server.py` — tiny HTTP server that loads the model and classifies one message per request
- `model.py`, `train.py`, `predict.py`, `data.tsv`, `model.pt` — earlier versions (v1/v2: small PyTorch LSTM, 6 classes)

## Trained weights

The fine-tuned weights (256 MB) are too large for GitHub — download them here:

[model_bert_v3.zip (Google Drive)](https://drive.google.com/file/d/1B53Utf5ngEhOVxGaPTEjjDY4GsURVus5/view?usp=drivesdk)

Unzip into `model_bert_v3/` so it contains `model.safetensors` next to `config.json`, then:

```bash
python predict_bert.py
```

## Retrain from scratch

```bash
pip install torch transformers
python train_bert.py   # downloads distilbert-base-uncased, trains, saves to model_bert_v3/
```
