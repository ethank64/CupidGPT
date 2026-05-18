# CupidGPT

A fine-tuned LLM for dating-app conversations. Built on `Qwen3-8B-Instruct`, fine-tuned with QLoRA on Reddit dating/relationship data (RODD + supplemental PRAW scrapes).

Stock chat models are terrible at dating-app banter — they're too formal, too eager, too safe. CupidGPT learns from how people actually talk about dating to produce replies that sound less like a customer-service bot and more like a friend who's good at this stuff.

## Hardware

Trained on a single **RTX 4060 Ti (16GB VRAM)** inside WSL2 Ubuntu. The whole pipeline is designed to fit on consumer hardware — no cloud GPUs required.

## Stack

- **Base model:** [`Qwen/Qwen3-8B-Instruct`](https://huggingface.co/Qwen)
- **Training:** [Unsloth](https://github.com/unslothai/unsloth) + QLoRA (4-bit, LoRA adapters)
- **Data:** [RODD](https://huggingface.co/datasets) + optional PRAW scrapes of r/dating, r/Tinder, r/datingoverthirty
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

cp .env.example .env           # fill in REDDIT_*, HF_TOKEN, WANDB_API_KEY

uv run python data/download_rodd.py
uv run python data/scrape_reddit.py    # optional
uv run python data/prepare_dataset.py

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
