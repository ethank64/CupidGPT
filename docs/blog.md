# CupidGPT: Fine-tuning Qwen3-8B for dating-app replies

> Placeholder — fill in after training completes.

## TL;DR

- Took `Qwen3-8B-Instruct`, fine-tuned with QLoRA on Reddit dating/relationship data
- Trained on a single RTX 4060 Ti (16GB VRAM) inside WSL2
- Result: a model that writes less stilted dating-app replies than the stock base
- Live demo: TBD

## Why?

Generic chat models are bad at dating-app banter. They're too formal, too long-winded, and overshoot on enthusiasm. This is a quick experiment to see if a small targeted fine-tune fixes that.

## Data

[RODD](https://huggingface.co/datasets) (Reddit Online Dating Dataset) as the core, optionally supplemented with PRAW scrapes from r/dating, r/Tinder, r/datingoverthirty, r/OnlineDating. After dedupe + length filtering: ~TODO rows.

## Training

| | |
|---|---|
| Base | `unsloth/Qwen3-8B-Instruct-bnb-4bit` |
| Method | QLoRA, r=16, alpha=32 |
| Effective batch size | 16 |
| LR | 2e-4, cosine, 5% warmup |
| Epochs | 2 |
| Hardware | RTX 4060 Ti 16GB, WSL2 |
| Wall time | TODO |
| VRAM peak | TODO |

Training curves: TODO (W&B link).

## Results

### Before

> "She just said 'lol same' — what do I say?"
> Stock Qwen3: TODO

### After

> CupidGPT: TODO

## Turing test

[TBD — link to the guess-the-AI game on the project site.]

## Code

[github.com/ethank64/CupidGPT](https://github.com/ethank64/CupidGPT)
