"""Optional Reddit supplementation via PRAW.

Pulls top posts + comment trees from dating-related subreddits and writes them to
data/raw/reddit.jsonl. Only run this if RODD turns out to be insufficient (e.g. mostly
profile/matching data rather than actual conversational exchanges).

Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import praw
from dotenv import load_dotenv

SUBREDDITS = [
    "dating",
    "Tinder",
    "datingoverthirty",
    "OnlineDating",
    "Bumble",
    "hingeapp",
]
POSTS_PER_SUB = 500   # cap so we stay tractable
MIN_COMMENT_LEN = 20
MAX_COMMENT_LEN = 1500
OUT_PATH = Path(__file__).parent / "raw" / "reddit.jsonl"


def main() -> None:
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    reddit.read_only = True

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for sub_name in SUBREDDITS:
            print(f"Scraping r/{sub_name}...")
            sub = reddit.subreddit(sub_name)
            for post in sub.top(time_filter="year", limit=POSTS_PER_SUB):
                if not post.selftext or post.over_18:
                    continue
                post.comments.replace_more(limit=0)
                top_comments = sorted(
                    post.comments, key=lambda c: c.score, reverse=True
                )[:5]
                for c in top_comments:
                    body = (c.body or "").strip()
                    if MIN_COMMENT_LEN <= len(body) <= MAX_COMMENT_LEN:
                        fh.write(
                            json.dumps(
                                {
                                    "subreddit": sub_name,
                                    "post_title": post.title,
                                    "post_body": post.selftext,
                                    "comment": body,
                                    "score": c.score,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_written += 1
    print(f"Wrote {n_written} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
