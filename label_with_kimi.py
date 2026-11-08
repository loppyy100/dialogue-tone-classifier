"""Teacher-labelling with Kimi: build a big dataset for distillation.

The idea: our small DistilBERT is limited by only 405 hand-written examples.
Kimi (a trillion-parameter teacher) labels thousands of chat-style messages,
and we fine-tune the small model on those labels. Small specialist + lots
of teacher-labelled data beats big brain + tiny data.

Input: a text file, one chat message per line (see unlabeled_seed.txt).
Output: a .tsv file with "message<TAB>label" lines, ready to use as data.tsv.

Usage:
    python label_with_kimi.py unlabeled_seed.txt --out labeled_kimi.tsv
    python label_with_kimi.py unlabeled_seed.txt --limit 20 --dry-run   # cost check only
    python label_with_kimi.py my_messages.txt --out labeled.tsv --batch 10

Setup: same as predict_kimi.py — MOONSHOT_API_KEY in your environment,
`pip install openai`. Model via MOONSHOT_MODEL (default kimi-k3).

Cost (Oct 31, 2026 snapshot — check platform.kimi.ai before budgeting):
  ~10 messages per API call, ~1,500 input tokens + ~300 output tokens per
  call with kimi-k3 ($3/$15 per 1M) ≈ $0.009 per call, i.e. ~$0.001 per
  message. 5,000 messages ≈ $5. The script prints its own estimate first.

Rate limits: a fresh account (Tier 0) allows ~3 requests/minute, so 5,000
messages at 10/call takes ~3 hours of wall time. Topping up more raises
the tier. The script saves progress after every batch, so you can stop
(Ctrl+C) and resume — already-labelled lines are skipped on the next run.

Privacy: these messages go to Moonshot's servers. Only label public or
synthetic chat lines — NEVER anyone's real private chats.
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI

LABELS = ["affection", "bid", "criticism", "contempt",
          "defensiveness", "stonewalling", "repair"]

BATCH_PROMPT = """You are labelling chat messages for a relationship classifier.
Label each message with exactly one of: affection, bid, criticism,
contempt, defensiveness, stonewalling, repair.

- affection: warmth, love, care ("i love you", "did you eat today")
- bid: invitation to connect ("look at this sunset", "guess what happened")
- criticism: attacks character, "you always/never" ("you never listen")
- contempt: mockery, sarcasm, superiority ("oh brilliant, another genius idea")
- defensiveness: deflecting blame ("it is not my fault", "well you did it first")
- stonewalling: shutting down, one-word or silence ("fine.", "k", "leave me alone")
- repair: trying to fix things after a fight ("i'm sorry, can we try again")

Reply with a JSON array of labels only, one per message, in order:
["affection", "bid", ...]
"""

# Rough token math for the printed estimate (kimi-k3, $3 in / $15 out per 1M)
IN_PRICE = 3.0 / 1_000_000
OUT_PRICE = 15.0 / 1_000_000


def get_client():
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("No MOONSHOT_API_KEY found. See predict_kimi.py for setup.")
        sys.exit(1)
    return OpenAI(api_key=api_key,
                  base_url=os.environ.get("MOONSHOT_BASE_URL",
                                          "https://api.moonshot.ai/v1"))


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f
                if l.strip() and not l.strip().startswith("#")]


def already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    done.add(line.rsplit("\t", 1)[0].strip())
    return done


def label_batch(client, model, batch):
    numbered = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(batch))
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": BATCH_PROMPT},
                          {"role": "user", "content": numbered}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            labels = data if isinstance(data, list) else data.get("labels", [])
            if len(labels) == len(batch) and all(
                    l in LABELS for l in labels):
                return labels, resp.usage
            print(f"  (bad reply, retry {attempt + 1}/3)")
        except Exception as e:
            print(f"  (error: {e}, retry {attempt + 1}/3, waiting 20s)")
            time.sleep(20)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="one message per line")
    parser.add_argument("--out", default="labeled_kimi.tsv")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0,
                        help="label only the first N (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cost estimate and stop")
    parser.add_argument("--sleep", type=float, default=21.0,
                        help="seconds between calls (Tier 0 ≈ 3/min)")
    args = parser.parse_args()

    lines = load_lines(args.input)
    if args.limit:
        lines = lines[:args.limit]
    done = already_done(args.out)
    todo = [l for l in lines if l not in done]
    print(f"{len(lines)} lines in file, {len(done)} already labelled, "
          f"{len(todo)} to do.")

    # Estimate: ~150 tokens prompt overhead + ~15/message in, ~3/message out
    n_calls = (len(todo) + args.batch - 1) // args.batch
    est_in = n_calls * 150 + len(todo) * 15
    est_out = len(todo) * 3
    est_cost = est_in * IN_PRICE + est_out * OUT_PRICE
    print(f"Estimate: ~{est_in:,} input + ~{est_out:,} output tokens "
          f"≈ ${est_cost:.2f} with {os.environ.get('MOONSHOT_MODEL', 'kimi-k3')}")
    print(f"At {args.sleep}s between {n_calls} calls: "
          f"~{n_calls * args.sleep / 60:.0f} minutes wall time.")
    if args.dry_run:
        return

    client = get_client()
    model = os.environ.get("MOONSHOT_MODEL", "kimi-k3")
    total_in = total_out = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            labels, usage = label_batch(client, model, batch)
            if labels is None:
                print(f"  batch {i // args.batch + 1} FAILED, stopping "
                      f"(progress saved, re-run to resume)")
                break
            for text, label in zip(batch, labels):
                f.write(f"{text}\t{label}\n")
            f.flush()
            if usage:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
            spent = total_in * IN_PRICE + total_out * OUT_PRICE
            print(f"  batch {i // args.batch + 1}/{n_calls}: "
                  f"{min(i + args.batch, len(todo))}/{len(todo)} done, "
                  f"spent so far ≈ ${spent:.3f}")
            if i + args.batch < len(todo):
                time.sleep(args.sleep)
    print(f"Done. Wrote {args.out}. Total spend ≈ "
          f"${total_in * IN_PRICE + total_out * OUT_PRICE:.2f}")
    print("Next: copy it to data.tsv (or merge with data_v3.tsv) and run "
          "train_bert.py on it.")


if __name__ == "__main__":
    main()
