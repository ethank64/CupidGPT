"""QLoRA fine-tune Qwen3-8B-Instruct on the CupidGPT dataset using Unsloth.

Usage:
    python training/train_qlora.py --config training/config.yaml

Outputs LoRA adapters to <output_dir>; merge separately via inference/merge_and_push.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    return p.parse_args()


def format_row(row: dict, system_prompt: str, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])

    print(f"Loading {cfg['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_rslora=cfg["lora"]["use_rslora"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg["training"]["seed"],
    )

    print("Loading dataset...")
    train_ds = load_dataset("json", data_files=cfg["data"]["train_path"], split="train")
    val_ds = load_dataset("json", data_files=cfg["data"]["val_path"], split="train")
    train_ds = train_ds.map(
        lambda r: format_row(r, cfg["data"]["system_prompt"], tokenizer),
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda r: format_row(r, cfg["data"]["system_prompt"], tokenizer),
        remove_columns=val_ds.column_names,
    )

    sft_cfg = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        bf16=cfg["training"]["bf16"],
        seed=cfg["training"]["seed"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_text_field="text",
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=cfg["wandb"]["run_name"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final adapter to {cfg['training']['output_dir']}/final")
    trainer.save_model(f"{cfg['training']['output_dir']}/final")
    tokenizer.save_pretrained(f"{cfg['training']['output_dir']}/final")


if __name__ == "__main__":
    main()
