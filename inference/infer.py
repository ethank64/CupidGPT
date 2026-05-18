"""Local CLI test for the fine-tuned adapter.

Usage:
    python inference/infer.py "She just said 'lol same' — what do I say?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from unsloth import FastLanguageModel

ADAPTER_DIR = Path("checkpoints/cupidgpt-qwen3-8b/final")
CFG_PATH = Path("training/config.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", type=str)
    p.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(CFG_PATH.read_text())
    system_prompt = cfg["data"]["system_prompt"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    out = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        top_p=0.9,
    )
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    print("\n--- CupidGPT ---")
    print(text)


if __name__ == "__main__":
    main()
