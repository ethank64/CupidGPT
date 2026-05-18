"""Gradio chat interface for the CupidGPT Hugging Face Space.

Expects the merged model to be available at HUB_ID (env var) or the default below.
On Spaces with a GPU runtime this loads in fp16; on ZeroGPU it loads on demand.
"""

from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB_ID = os.environ.get("HUB_ID", "ethank64/CupidGPT")
SYSTEM_PROMPT = (
    "You are CupidGPT, a witty, kind, conversational assistant who helps people "
    "write good dating-app replies. Be playful but never creepy or pushy. Match "
    "the casual register of the person you're replying to. Keep responses short "
    "unless asked for more."
)

print(f"Loading {HUB_ID}...")
tokenizer = AutoTokenizer.from_pretrained(HUB_ID)
model = AutoModelForCausalLM.from_pretrained(
    HUB_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()


def respond(message: str, history: list[dict], temperature: float, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="CupidGPT",
    description=(
        "A Qwen3-8B fine-tune trained on Reddit dating/relationship data. "
        "Paste a message you received on a dating app and ask for a reply."
    ),
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(32, 512, value=256, step=32, label="Max new tokens"),
    ],
    examples=[
        ["She just said 'lol same' — what do I say?"],
        ["He opened with 'hey'. How do I keep this going?"],
        ["How do I ask her out without it being awkward?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
