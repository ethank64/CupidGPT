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

```bash
# Inside WSL Ubuntu
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

cp .env.example .env  # fill in REDDIT_*, HF_TOKEN, WANDB_API_KEY

python data/download_rodd.py
python data/scrape_reddit.py   # optional
python data/prepare_dataset.py

python training/train_qlora.py --config training/config.yaml
python inference/infer.py "She just said 'lol same' — what do I say?"
python inference/merge_and_push.py --hub-id ethank64/CupidGPT
```

## Demo

Live demo on Hugging Face Spaces: `ethank64/CupidGPT-demo` (link once deployed).

## License

MIT. Data scraped from Reddit is subject to Reddit's terms; only use this for research / personal projects.
