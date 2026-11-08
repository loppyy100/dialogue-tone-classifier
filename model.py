"""Love-chat classifier — the model (v2).

Reads one chat message and guesses its vibe:
  affection | bid | criticism | contempt | defensiveness | stonewalling

v1 was a bag of words: EmbeddingBag averaged the word vectors, so word order
was invisible ("i love you" and "you love i" looked the same). It worked, but
it memorised the tiny training set and stayed at 57% val accuracy.

v2 is a real sequence model:
  words -> Embedding -> BiLSTM (reads left-to-right AND right-to-left)
        -> attention pooling (the net learns WHICH words matter most)
        -> Dropout -> Linear -> 6 scores

Why each piece:
  - BiLSTM: word order matters now. "you never listen" hits differently from
    "never you listen", and "not my fault" needs the "not".
  - Attention: instead of averaging all words equally, the net puts weight on
    the telling words ("never", "genius", "whatever") and ignores filler.
  - Dropout: randomly switches off neurons while training, so the net can't
    just memorise the 258 examples — it has to learn patterns that generalise.

Gottman grounding:
  - criticism, contempt, defensiveness, stonewalling = the "Four Horsemen".
    Contempt is the single strongest predictor of breakups.
  - bid = a bid for connection. Couples still together at 6 years turned
    toward bids ~86% of the time; divorced couples ~33%.
  - affection = warmth, love, gratitude (the 5:1 magic ratio).
"""

import torch
import torch.nn as nn

LABELS = ["affection", "bid", "criticism", "contempt", "defensiveness", "stonewalling"]
NUM_LABELS = len(LABELS)

# The four horsemen, for reference in the output
HORSEMEN = {"criticism", "contempt", "defensiveness", "stonewalling"}

PAD_IDX = 1  # "<pad>" — the LSTM ignores these positions


def tokenize(text: str):
    """Lowercase, split on whitespace. Simple on purpose."""
    return text.lower().split()


def build_vocab(texts, min_freq=1):
    """Word -> index. Index 0 is <unk> (unknown word), 1 is <pad>."""
    from collections import Counter

    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    vocab = {"<unk>": 0, "<pad>": 1}
    for word, c in counts.most_common():
        if c >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    """Text -> list of word indices (unknown words become 0)."""
    return [vocab.get(w, 0) for w in tokenize(text)]


def batch_to_tensors(texts, vocab):
    """A list of messages -> (padded indices [batch, max_len], lengths).

    Short messages are padded with <pad> (1) to the longest in the batch.
    The model masks padding out, so it never counts as a real word.
    """
    seqs = [encode(t, vocab) for t in texts]
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, lengths


class LoveChatClassifier(nn.Module):
    """v2: BiLSTM + attention pooling."""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64,
                 num_labels=NUM_LABELS, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # bidirectional=True -> hidden is 2x hidden_dim (forward + backward)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        # Attention: one score per word, learned. The net decides what to focus on.
        self.attn_w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, padded, lengths):
        emb = self.embedding(padded)  # [batch, time, embed_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out: [batch, time, hidden*2]

        # Attention scores, with padding masked out so it can't be attended to.
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)  # [batch, time]
        mask = torch.arange(out.size(1), device=out.device)[None, :] >= lengths[:, None].to(out.device)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # [batch, time]
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden*2]

        return self.fc(self.dropout(context))


class LoveChatClassifierV1(nn.Module):
    """v1: EmbeddingBag (bag of words). Kept so old model.pt files still load."""

    def __init__(self, vocab_size, embed_dim=32, num_labels=NUM_LABELS):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, text_indices, offsets):
        return self.fc(self.embedding(text_indices, offsets))


def build_model(ckpt):
    """Build the right model class from a checkpoint dict."""
    arch = ckpt.get("arch", "bow")
    if arch == "bilstm-attn":
        model = LoveChatClassifier(
            len(ckpt["vocab"]),
            embed_dim=ckpt["embed_dim"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt.get("dropout", 0.3),
        )
    else:
        model = LoveChatClassifierV1(len(ckpt["vocab"]), embed_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
