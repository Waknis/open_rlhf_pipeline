"""Gradio chat demo for a trained Open RLHF checkpoint."""

from __future__ import annotations

import argparse
import pathlib
import sys

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.formatting import render_prompt  # noqa: E402
from open_rlhf.runtime import add_runtime_args, load_tokenizer, make_torch_device, resolve_runtime  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Open RLHF Gradio chat demo.")
    parser.add_argument("--model", required=True, help="Path to the trained policy checkpoint.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per user turn.",
    )
    add_runtime_args(parser)
    return parser.parse_args(argv)


def build_chat_fn(model_name: str, runtime, max_new_tokens: int):
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=runtime.torch_dtype,
    )
    device = make_torch_device(runtime)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    def chat(message, _history):
        prompt_text = render_prompt(tokenizer, message)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(device)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=generation_config,
            )
        response_ids = output[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return chat


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = resolve_runtime(args.device, args.dtype)
    chat_fn = build_chat_fn(args.model, runtime, args.max_new_tokens)
    gr.ChatInterface(chat_fn, title="Open RLHF Assistant").launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
