"""Normalize raw sources into instruction/response JSONL for SFT.

Inputs:  data/raw/rodd.jsonl (required), data/raw/reddit.jsonl (optional)
Outputs: data/processed/train.jsonl, data/processed/val.jsonl

Each output row is:
    {"instruction": "<user's message + context>", "response": "<good reply>"}

Filters: dedupe by (instruction, response), drop entries outside [10, 2000] chars,
strip URLs, scrub Reddit usernames /u/ mentions, light profanity gate is NOT applied
here (handled at inference time via system prompt).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"
TRAIN_OUT = OUT_DIR / "train.jsonl"
VAL_OUT = OUT_DIR / "val.jsonl"

MIN_LEN = 10
MAX_LEN = 2000
VAL_FRAC = 0.05
SEED = 42

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"/u/\w+|u/\w+")


def clean(text: str) -> str:
    text = URL_RE.sub("[link]", text)
    text = USER_RE.sub("[user]", text)
    return text.strip()


def in_range(s: str) -> bool:
    return MIN_LEN <= len(s) <= MAX_LEN


def load_rodd() -> list[dict]:
    """RODD schema is inspected at runtime — this is a best-effort adapter.

    The exact field names will be confirmed once download_rodd.py runs; tune
    here if the live schema differs from what's assumed.
    """
    path = RAW_DIR / "rodd.jsonl"
    if not path.exists():
        return []
    pairs: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            # Try common shapes; fall back to skipping.
            instr = row.get("prompt") or row.get("input") or row.get("question") or row.get("post")
            resp = (
                row.get("response") or row.get("output") or row.get("answer") or row.get("comment")
            )
            if instr and resp:
                pairs.append({"instruction": clean(instr), "response": clean(resp)})
    return pairs


def load_reddit() -> list[dict]:
    path = RAW_DIR / "reddit.jsonl"
    if not path.exists():
        return []
    pairs: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            instr = f"{row.get('post_title', '')}\n\n{row.get('post_body', '')}"
            resp = row.get("comment", "")
            pairs.append({"instruction": clean(instr), "response": clean(resp)})
    return pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_rodd() + load_reddit()
    print(f"Raw pair count: {len(pairs)}")

    # Filter
    seen: set[tuple[str, str]] = set()
    filtered: list[dict] = []
    for p in pairs:
        key = (p["instruction"], p["response"])
        if key in seen:
            continue
        if not (in_range(p["instruction"]) and in_range(p["response"])):
            continue
        seen.add(key)
        filtered.append(p)
    print(f"After dedupe + length filter: {len(filtered)}")

    # Split
    random.seed(SEED)
    random.shuffle(filtered)
    n_val = max(1, int(len(filtered) * VAL_FRAC))
    val = filtered[:n_val]
    train = filtered[n_val:]

    with TRAIN_OUT.open("w", encoding="utf-8") as fh:
        for row in train:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with VAL_OUT.open("w", encoding="utf-8") as fh:
        for row in val:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} -> {TRAIN_OUT}")
    print(f"Wrote {len(val)}   -> {VAL_OUT}")


if __name__ == "__main__":
    main()
