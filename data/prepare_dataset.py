"""Normalize raw sources into instruction/response JSONL for SFT.

Inputs:
    data/raw/rodd.jsonl              # posts (from download_rodd.py)
    data/raw/rodd_comments.jsonl     # best top-comment per post (from fetch_rodd_comments.py)
    data/raw/reddit.jsonl            # optional, from scrape_reddit.py

Outputs:
    data/processed/train.jsonl, data/processed/val.jsonl

Each output row is:
    {"instruction": "<post title + body>", "response": "<top comment>"}

Filters: dedupe, length range, strip URLs and /u/ mentions. Safety filtering is
applied at inference time via system prompt, not here.
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
MAX_INSTR_LEN = 4000
MAX_RESP_LEN = 2000
VAL_FRAC = 0.05
SEED = 42

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"/u/\w+|u/\w+")
SUB_RE = re.compile(r"/r/\w+|r/\w+")


def clean(text: str) -> str:
    text = URL_RE.sub("[link]", text)
    text = USER_RE.sub("[user]", text)
    text = SUB_RE.sub("[sub]", text)
    return text.strip()


def in_range(s: str, max_len: int) -> bool:
    return MIN_LEN <= len(s) <= max_len


def load_rodd() -> list[dict]:
    """Join RODD posts with fetched top comments, formatted as SFT pairs."""
    posts_path = RAW_DIR / "rodd.jsonl"
    comments_path = RAW_DIR / "rodd_comments.jsonl"
    if not posts_path.exists():
        print(f"  {posts_path} missing — run download_rodd.py")
        return []
    if not comments_path.exists():
        print(f"  {comments_path} missing — run fetch_rodd_comments.py")
        return []

    # Index comments by post_id (only keep status=ok rows).
    comments: dict[str, str] = {}
    with comments_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("status") == "ok":
                comments[row["post_id"]] = row["comment_body"]

    pairs: list[dict] = []
    with posts_path.open() as fh:
        for line in fh:
            post = json.loads(line)
            comment = comments.get(post["id"])
            if not comment:
                continue
            title = post.get("title") or ""
            body = post.get("selftext") or ""
            instr = f"{title}\n\n{body}".strip() if body else title
            pairs.append({"instruction": clean(instr), "response": clean(comment)})
    print(f"  RODD: {len(pairs):,} joined pairs ({len(comments):,} comments available)")
    return pairs


def load_reddit() -> list[dict]:
    """Optional PRAW scrape from scrape_reddit.py."""
    path = RAW_DIR / "reddit.jsonl"
    if not path.exists():
        return []
    pairs: list[dict] = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            instr = f"{row.get('post_title', '')}\n\n{row.get('post_body', '')}".strip()
            resp = row.get("comment", "")
            pairs.append({"instruction": clean(instr), "response": clean(resp)})
    print(f"  reddit.jsonl: {len(pairs):,} pairs")
    return pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading sources:")
    pairs = load_rodd() + load_reddit()
    print(f"Raw pair count: {len(pairs):,}")
    if not pairs:
        raise SystemExit("No pairs loaded — check that you've run download + fetch.")

    seen: set[tuple[str, str]] = set()
    filtered: list[dict] = []
    n_dup = n_oob = 0
    for p in pairs:
        key = (p["instruction"], p["response"])
        if key in seen:
            n_dup += 1
            continue
        if not (
            in_range(p["instruction"], MAX_INSTR_LEN) and in_range(p["response"], MAX_RESP_LEN)
        ):
            n_oob += 1
            continue
        seen.add(key)
        filtered.append(p)
    print(
        f"After dedupe + length filter: {len(filtered):,} "
        f"(dropped {n_dup:,} dup, {n_oob:,} out-of-range)"
    )

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
    print(f"Wrote {len(train):,} -> {TRAIN_OUT}")
    print(f"Wrote {len(val):,}   -> {VAL_OUT}")


if __name__ == "__main__":
    main()
