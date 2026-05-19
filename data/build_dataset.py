"""Build the CupidGPT SFT dataset from HuggingFaceGECLM/REDDIT_comments.

Replaces the older RODD + PRAW pipeline. All-HF, no auth required.

Approach: stream the r/relationship_advice shards, group comments by post (link_id),
and emit `(parent_comment, top_score_reply)` pairs for each thread. The model
learns to produce Reddit-style relationship-advice replies to advice-shaped prompts.

Usage:
    uv run python data/build_dataset.py                  # full build, ~5-10 min
    uv run python data/build_dataset.py --shards 3       # smoke test on 3/45 shards
    uv run python data/build_dataset.py --target 50000   # stop once N pairs collected
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO = "HuggingFaceGECLM/REDDIT_comments"
SUBREDDIT = "relationship_advice"

OUT_DIR = Path(__file__).parent / "processed"
TRAIN_OUT = OUT_DIR / "train.jsonl"
VAL_OUT = OUT_DIR / "val.jsonl"

# Filter thresholds. These reject low-effort / removed / obvious junk comments
# at the source so we never have to chase them downstream.
MIN_SCORE = 5
MIN_LEN = 40
MAX_LEN = 2000
MAX_PAIRS_PER_THREAD = 3  # don't let one big thread dominate
VAL_FRAC = 0.05
SEED = 42

DEAD_BODIES = {"[deleted]", "[removed]", "[ Removed by Reddit ]"}

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"/u/\w+|u/\w+")
SUB_RE = re.compile(r"/r/\w+|r/\w+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Process only the first N shards (default: all). Useful for smoke testing.",
    )
    p.add_argument(
        "--target",
        type=int,
        default=None,
        help="Stop early once this many pairs are collected.",
    )
    p.add_argument("--min-score", type=int, default=MIN_SCORE)
    return p.parse_args()


def clean(text: str) -> str:
    text = URL_RE.sub("[link]", text)
    text = USER_RE.sub("[user]", text)
    text = SUB_RE.sub("[sub]", text)
    return text.strip()


def good_body(body: str | None) -> bool:
    if not body or body in DEAD_BODIES:
        return False
    return MIN_LEN <= len(body) <= MAX_LEN


def list_shards() -> list[str]:
    files = list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f"/{SUBREDDIT}-" in f and f.endswith(".parquet"))


def pairs_from_shard(path: Path, min_score: int) -> list[tuple[str, str]]:
    """Group this shard's comments by link_id and emit (parent.body, child.body) pairs
    where child.parent_id == parent.id (i.e. child is a direct reply to parent)."""
    pf = pq.ParquetFile(path)
    tbl = pf.read(columns=["id", "parent_id", "link_id", "body", "score"])
    rows = tbl.to_pylist()

    # parse score as int (it's stored as string in this dataset)
    by_link: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            r["score"] = int(r["score"])
        except (TypeError, ValueError):
            continue
        if r["score"] < min_score:
            continue
        if not good_body(r["body"]):
            continue
        by_link[r["link_id"]].append(r)

    pairs: list[tuple[str, str]] = []
    for comments in by_link.values():
        by_id = {c["id"]: c for c in comments}
        # Find (parent, child) where child.parent_id == parent.id, both in this thread.
        candidates: list[tuple[dict, dict]] = []
        for child in comments:
            pid = child["parent_id"]
            if not pid.startswith("t1_"):
                continue
            parent = by_id.get(pid[3:])
            if parent is None:
                continue
            candidates.append((parent, child))
        # Cap per-thread; keep highest combined score
        candidates.sort(key=lambda pc: pc[0]["score"] + pc[1]["score"], reverse=True)
        for parent, child in candidates[:MAX_PAIRS_PER_THREAD]:
            pairs.append((clean(parent["body"]), clean(child["body"])))
    return pairs


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shards = list_shards()
    if args.shards is not None:
        shards = shards[: args.shards]
    print(f"Processing {len(shards)} shards of r/{SUBREDDIT}...")

    all_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for shard_path in tqdm(shards, desc="shards", unit="shard"):
        local = Path(hf_hub_download(REPO, shard_path, repo_type="dataset"))
        new = pairs_from_shard(local, args.min_score)
        for p in new:
            if p in seen:
                continue
            seen.add(p)
            all_pairs.append(p)
        if args.target and len(all_pairs) >= args.target:
            print(f"Hit target {args.target:,} pairs after {len(all_pairs):,}, stopping.")
            break

    print(f"\nCollected {len(all_pairs):,} unique pairs.")
    if not all_pairs:
        raise SystemExit("No pairs collected — check filters.")

    random.seed(SEED)
    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * VAL_FRAC))
    val = all_pairs[:n_val]
    train = all_pairs[n_val:]

    def write_jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for instr, resp in rows:
                fh.write(
                    json.dumps({"instruction": instr, "response": resp}, ensure_ascii=False) + "\n"
                )

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    print(f"Wrote {len(train):,} -> {TRAIN_OUT}")
    print(f"Wrote {len(val):,}   -> {VAL_OUT}")


if __name__ == "__main__":
    main()
