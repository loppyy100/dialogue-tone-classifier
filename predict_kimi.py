"""Classify a chat message with Kimi (Moonshot API) — the giant-model teacher.

This is the trillion-parameter side of the project: instead of our small
local net, it asks Kimi (kimi-k3, ~2.8T params) to classify each message.
Use it to (a) see what the big model thinks, and (b) check our small
model's answers against a teacher.

Setup (do this once, on your own laptop):
  1. Sign up at https://platform.kimi.ai (international, USD) or
     https://platform.moonshot.cn (China, RMB) and top up at least $1.
  2. Console -> API Keys -> Create API Key.
  3. NEVER paste the key in chat or save it in a file. Set it as an
     environment variable in your terminal:
         export MOONSHOT_API_KEY="sk-..."
     (put it in ~/.bashrc if you want it to persist across terminals)
  4. pip install openai

Usage:
    python predict_kimi.py "fine." "i love you so much"
    python predict_kimi.py            (no args = interactive, Ctrl+C to quit)

Cost (Oct 31, 2026 snapshot — check platform.kimi.ai before budgeting):
  kimi-k3 is ~$3 per 1M input tokens / $15 per 1M output tokens.
  One classification is ~250 input tokens + ~20 output tokens, so about
  $0.001 per message — roughly 1,000 messages per $1. Labelling 5,000
  messages for distillation costs ~$5. Rates change; the platform is the
  source of truth. Model and endpoint are overridable:
    MOONSHOT_MODEL=kimi-k2  MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
"""

import argparse
import json
import os
import sys

from openai import OpenAI

LABELS = ["affection", "bid", "criticism", "contempt",
          "defensiveness", "stonewalling", "repair"]

SYSTEM_PROMPT = """You are a relationship-message classifier. Classify one chat
message into exactly one label. Reply with JSON only:
{"label": "<one of the labels>", "confidence": <0-100>, "reason": "<one short sentence>"}

Labels:
- affection: warmth, love, care. e.g. "i love you so much", "did you eat today"
- bid: an invitation to connect or share attention. e.g. "look at this sunset", "guess what happened today"
- criticism: attacks character with "you always/never". e.g. "you never listen to me"
- contempt: mockery, sarcasm, superiority. e.g. "oh brilliant, another genius idea"
- defensiveness: deflecting blame, "it's not my fault". e.g. "it is not my fault", "well you did it first"
- stonewalling: shutting down, one-word or silence. e.g. "fine.", "k", "leave me alone"
- repair: trying to fix things after a fight. e.g. "i'm sorry, can we try again", "let's figure this out together"

One message is never a diagnosis — classify the message, not the relationship.
"""


def get_client():
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("No MOONSHOT_API_KEY found in your environment.\n"
              "  1. Get a key at https://platform.kimi.ai (Console -> API Keys)\n"
              "  2. In your terminal run:\n"
              "       export MOONSHOT_API_KEY=\"sk-...\"\n"
              "  Never paste the key in chat or save it in a file.")
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("MOONSHOT_BASE_URL",
                                "https://api.moonshot.ai/v1"),
    )


def classify(client, model, text):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": text}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = resp.usage
    try:
        out = json.loads(resp.choices[0].message.content)
    except Exception:
        return None, usage
    return out, usage


def show(result, usage):
    if result is None:
        print("  -> (could not parse Kimi's reply)")
        return
    label = result.get("label", "?")
    conf = result.get("confidence", "?")
    reason = result.get("reason", "")
    tag = "  <-- FOUR HORSEMAN" if label in (
        "criticism", "contempt", "defensiveness", "stonewalling") else ""
    print(f'  -> {label} ({conf}%){tag}')
    if reason:
        print(f"     {reason}")
    if usage:
        print(f"     [tokens in {usage.prompt_tokens} / out "
              f"{usage.completion_tokens}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="*")
    args = parser.parse_args()
    client = get_client()
    model = os.environ.get("MOONSHOT_MODEL", "kimi-k3")
    print(f"Model: {model} (Kimi API)")
    texts = args.texts
    if texts:
        for t in texts:
            print(f'\n"{t}"')
            result, usage = classify(client, model, t)
            show(result, usage)
        return
    print("Type a chat message, Enter to classify. Ctrl+C to quit.\n")
    try:
        while True:
            text = input("> ").strip()
            if text:
                result, usage = classify(client, model, text)
                show(result, usage)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
