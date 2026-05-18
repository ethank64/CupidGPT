"""One-off exploration of RODD to decide if it's usable as SFT training data.

Run: uv run python data/explore_rodd.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PATH = Path(__file__).parent / "raw" / "rodd.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in PATH.open()]
    print(f"Total rows: {len(rows):,}\n")

    subs = Counter(r["subreddit"] for r in rows)
    print("Top 20 subreddits:")
    for sub, n in subs.most_common(20):
        print(f"  {n:>7,}  r/{sub}")
    print(f"  ({len(subs)} unique subreddits total)\n")

    # selftext stats
    selftext_lens = [len(r.get("selftext") or "") for r in rows]
    empty = sum(1 for n in selftext_lens if n == 0)
    short = sum(1 for n in selftext_lens if 0 < n < 100)
    medium = sum(1 for n in selftext_lens if 100 <= n < 1000)
    long_ = sum(1 for n in selftext_lens if n >= 1000)
    print("selftext lengths:")
    print(f"  empty:  {empty:>7,} ({empty / len(rows):.1%})")
    print(f"  <100:   {short:>7,} ({short / len(rows):.1%})")
    print(f"  100-1k: {medium:>7,} ({medium / len(rows):.1%})")
    print(f"  >=1k:   {long_:>7,} ({long_ / len(rows):.1%})\n")

    # check all keys present in row 0 — verify there really is no "comment" field
    print(f"All available keys: {sorted(rows[0].keys())}\n")

    # Score distribution — are these actually engaged posts?
    scores = sorted([r["score"] for r in rows])
    print(
        f"Score quartiles: min={scores[0]}, "
        f"q1={scores[len(scores) // 4]}, "
        f"median={scores[len(scores) // 2]}, "
        f"q3={scores[3 * len(scores) // 4]}, "
        f"max={scores[-1]}"
    )

    # num_comments — are there at least many engaged posts so PRAW could fetch replies?
    ncs = sorted([r["num_comments"] for r in rows])
    print(
        f"num_comments quartiles: min={ncs[0]}, "
        f"q1={ncs[len(ncs) // 4]}, "
        f"median={ncs[len(ncs) // 2]}, "
        f"q3={ncs[3 * len(ncs) // 4]}, "
        f"max={ncs[-1]}\n"
    )

    # A few sample titles to gauge content
    print("First 5 titles:")
    for r in rows[:5]:
        print(f"  [r/{r['subreddit']}, score={r['score']}, {r['num_comments']} comments]")
        print(f"    {r['title']}")


if __name__ == "__main__":
    main()
