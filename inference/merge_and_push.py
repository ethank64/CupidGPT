"""Merge the LoRA adapter into the base model and push to Hugging Face Hub.

Usage:
    python inference/merge_and_push.py --hub-id ethank64/CupidGPT [--gguf]

--gguf additionally exports a Q4_K_M GGUF for llama.cpp users.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from unsloth import FastLanguageModel

ADAPTER_DIR = Path("checkpoints/cupidgpt-qwen3-8b/final")
CFG_PATH = Path("training/config.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hub-id", required=True, help="HF Hub repo id, e.g. ethank64/CupidGPT")
    p.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    p.add_argument("--gguf", action="store_true", help="Also push a Q4_K_M GGUF")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = yaml.safe_load(CFG_PATH.read_text())

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env")
    login(token=token)

    print(f"Loading adapter from {args.adapter}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=False,  # we want fp16 merge for the public upload
    )

    print(f"Pushing merged fp16 model to {args.hub_id}...")
    model.push_to_hub_merged(
        args.hub_id,
        tokenizer,
        save_method="merged_16bit",
        token=token,
        private=args.private,
    )

    if args.gguf:
        print(f"Pushing GGUF Q4_K_M to {args.hub_id}-gguf...")
        model.push_to_hub_gguf(
            f"{args.hub_id}-gguf",
            tokenizer,
            quantization_method="q4_k_m",
            token=token,
            private=args.private,
        )

    print("Done.")


if __name__ == "__main__":
    main()
