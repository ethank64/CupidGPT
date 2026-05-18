"""Download the Reddit Online Dating Dataset (RODD) from Hugging Face Hub.

Saves the raw rows to data/raw/rodd.jsonl so prepare_dataset.py can ingest it
alongside other sources without re-downloading.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

RODD_REPO = "FabianLeibinger/Reddit-Online-Dating-Dataset-RODD"
OUT_PATH = Path(__file__).parent / "raw" / "rodd.jsonl"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {RODD_REPO} from Hugging Face Hub...")
    ds = load_dataset(RODD_REPO, split="train")

    print(f"Dataset features: {ds.features}")
    print(f"Row count: {len(ds)}")
    print(f"First row: {ds[0]}")

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ds)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
