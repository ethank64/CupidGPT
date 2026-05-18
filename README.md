# CupidGPT

A fine-tuned LLM that responds like a real Redditor giving dating advice. Built on `Qwen3-8B-Instruct`, fine-tuned with QLoRA on `(post → top comment)` pairs reconstructed from the [RODD](https://huggingface.co/datasets/FabianLeibinger/Reddit-Online-Dating-Dataset-RODD) dataset (218k posts from r/dating_advice, r/dating, r/OnlineDating, etc.) with the missing reply half pulled in via PRAW.

Stock chat models give safe, sanitized relationship advice that sounds like an HR memo. CupidGPT learns from how people actually answer dating questions on Reddit, which is — for better and worse — a lot more direct.

## Hardware

Trained on a single **RTX 4060 Ti (16GB VRAM)** inside WSL2 Ubuntu. The whole pipeline is designed to fit on consumer hardware — no cloud GPUs required.

## Stack

- **Base model:** [`Qwen/Qwen3-8B-Instruct`](https://huggingface.co/Qwen)
- **Training:** [Unsloth](https://github.com/unslothai/unsloth) + QLoRA (4-bit, LoRA adapters)
- **Data:** RODD posts + top comments fetched via PRAW (`data/fetch_rodd_comments.py`)
- **Tracking:** Weights & Biases
- **Demo:** Gradio on Hugging Face Spaces

## Layout

```
data/         download + prepare training data
training/     QLoRA fine-tune script
inference/    local test + HF Hub push
deploy/       Gradio app for HF Spaces
docs/         writeup
```

## Quickstart

Managed with [uv](https://docs.astral.sh/uv/). Install it once: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
# Inside WSL Ubuntu
uv sync                        # creates .venv and installs all deps from uv.lock

cp .env.example .env           # fill in REDDIT_*, HF_TOKEN, (optional) WANDB_API_KEY

uv run python data/download_rodd.py                        # ~219k posts
uv run python data/fetch_rodd_comments.py --limit 500      # smoke-test fetch first
uv run python data/fetch_rodd_comments.py                  # full fetch (resumable, hours-long)
uv run python data/prepare_dataset.py                      # join → train/val JSONL

uv run python training/train_qlora.py --config training/config.yaml
uv run python inference/infer.py "She just said 'lol same' — what do I say?"
uv run python inference/merge_and_push.py --hub-id ethank64/CupidGPT
```

## Development

```bash
uv sync --all-groups           # includes dev deps (ruff, ipython)
uv run ruff format .           # auto-format
uv run ruff check .            # lint
```

CI runs `ruff format --check` and `ruff check` on every PR to `main`.

## Demo

Live demo on Hugging Face Spaces: `ethank64/CupidGPT-demo` (link once deployed).

## License

MIT. Data scraped from Reddit is subject to Reddit's terms; only use this for research / personal projects.
