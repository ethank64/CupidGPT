"""Fetch the top comment for each engaged RODD post via PRAW.

RODD ships post bodies only — no replies. This script fills in the missing half:
for each post that passes the engagement filter, it pulls the highest-scoring
top-level comment so prepare_dataset.py can build (post, best_comment) SFT pairs.

The output file is appended to incrementally and the script is resumable: rerun
it after an interruption and it picks up where it left off, skipping post IDs
already present in data/raw/rodd_comments.jsonl.

Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env.

Usage:
    uv run python data/fetch_rodd_comments.py                       # full run
    uv run python data/fetch_rodd_comments.py --limit 500           # smoke test
    uv run python data/fetch_rodd_comments.py --min-score 10        # tighter filter
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import praw
from dotenv import load_dotenv
from tqdm import tqdm

POSTS_PATH = Path(__file__).parent / "raw" / "rodd.jsonl"
OUT_PATH = Path(__file__).parent / "raw" / "rodd_comments.jsonl"
SKIPPED_PATH = Path(__file__).parent / "raw" / "rodd_comments_skipped.jsonl"

# Output rows have a stable "status" field so prepare_dataset.py and reruns can
# distinguish "no usable comment found" from "not fetched yet".
STATUS_OK = "ok"
STATUS_EMPTY = "no_top_comment"  # post had no qualifying top-level comment
STATUS_DELETED = "deleted"  # post is gone / inaccessible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Fetch comments for at most N posts (after engagement filter). For testing.",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=5,
        help="Skip posts with score below this (default: 5).",
    )
    p.add_argument(
        "--min-comments",
        type=int,
        default=5,
        help="Skip posts with num_comments below this (default: 5).",
    )
    p.add_argument(
        "--min-comment-len",
        type=int,
        default=30,
        help="Reject the top comment if it's shorter than this many chars.",
    )
    p.add_argument(
        "--max-comment-len",
        type=int,
        default=4000,
        help="Reject the top comment if it's longer than this many chars.",
    )
    p.add_argument(
        "--min-comment-score",
        type=int,
        default=2,
        help="Reject the top comment if its score is below this.",
    )
    return p.parse_args()


def load_engaged_posts(min_score: int, min_comments: int) -> list[dict]:
    posts: list[dict] = []
    with POSTS_PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["score"] < min_score or row["num_comments"] < min_comments:
                continue
            posts.append(row)
    return posts


def load_done_ids() -> set[str]:
    """All post IDs already written to the output file (any status)."""
    if not OUT_PATH.exists():
        return set()
    ids: set[str] = set()
    with OUT_PATH.open() as fh:
        for line in fh:
            ids.add(json.loads(line)["post_id"])
    return ids


def best_top_comment(submission, args: argparse.Namespace):
    """Return the highest-score top-level comment that passes length/score filters,
    or None if none qualify."""
    submission.comments.replace_more(limit=0)
    candidates = [
        c
        for c in submission.comments
        if c.body
        and args.min_comment_len <= len(c.body) <= args.max_comment_len
        and c.score >= args.min_comment_score
        and not c.stickied
        and c.author is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def main() -> None:
    load_dotenv()
    args = parse_args()

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    reddit.read_only = True

    posts = load_engaged_posts(args.min_score, args.min_comments)
    print(f"Engaged posts (score>={args.min_score}, comments>={args.min_comments}): {len(posts):,}")

    done = load_done_ids()
    todo = [p for p in posts if p["id"] not in done]
    print(f"Already fetched: {len(done):,}. Remaining: {len(todo):,}.")

    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"--limit applied: fetching {len(todo):,} this run.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_empty = n_deleted = n_error = 0

    with OUT_PATH.open("a", encoding="utf-8") as out_fh:
        for post in tqdm(todo, desc="fetching comments", unit="post"):
            try:
                submission = reddit.submission(id=post["id"])
                if getattr(submission, "removed_by_category", None) or submission.author is None:
                    record = {"post_id": post["id"], "status": STATUS_DELETED}
                    n_deleted += 1
                else:
                    top = best_top_comment(submission, args)
                    if top is None:
                        record = {"post_id": post["id"], "status": STATUS_EMPTY}
                        n_empty += 1
                    else:
                        record = {
                            "post_id": post["id"],
                            "status": STATUS_OK,
                            "comment_body": top.body,
                            "comment_score": top.score,
                            "comment_id": top.id,
                        }
                        n_ok += 1
            except Exception as e:  # noqa: BLE001  — log and continue, don't crash the whole run
                n_error += 1
                # Don't record errors permanently — they should retry on the next run.
                SKIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
                with SKIPPED_PATH.open("a", encoding="utf-8") as skip_fh:
                    skip_fh.write(
                        json.dumps({"post_id": post["id"], "error": str(e)}, ensure_ascii=False)
                        + "\n"
                    )
                continue

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()  # so a Ctrl+C doesn't lose the last batch

    print(
        f"\nDone. ok={n_ok:,}, no_top_comment={n_empty:,}, "
        f"deleted={n_deleted:,}, errors={n_error:,}"
    )
    print(f"Output: {OUT_PATH}")
    if n_error:
        print(f"Transient errors logged to {SKIPPED_PATH} — rerun the script to retry them.")


if __name__ == "__main__":
    main()
