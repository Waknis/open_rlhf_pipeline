"""Supervised fine-tuning for a causal language model."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.formatting import render_prompt, render_prompt_response  # noqa: E402
from open_rlhf.runtime import (  # noqa: E402
    add_runtime_args,
    ensure_tokenizer_has_padding,
    ensure_validated_stack,
    load_tokenizer,
    resolve_runtime,
    seed_everything,
)


@dataclass
class SupervisedDataCollator:
    """Pad variable-length SFT features."""

    tokenizer: any

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            padded_labels = feature["labels"] + [-100] * (max_length - len(feature["labels"]))
            labels.append(padded_labels)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for a causal language model.")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base Hugging Face model ID or local checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/instruct.jsonl"),
        help="Processed instruction data. Canonical rows contain `prompt`, `response`, and `text`.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/sft"),
        help="Directory where the fine-tuned model will be saved.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2,
        help="Maximum training steps.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length after tokenization.",
    )
    add_runtime_args(parser)
    return parser.parse_args(argv)


def _tokenize_structured_example(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    prompt_text = render_prompt(tokenizer, prompt)
    full_text = render_prompt_response(tokenizer, prompt, response)

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    labels = list(full_tokens["input_ids"])
    prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    return {
        "input_ids": list(full_tokens["input_ids"]),
        "attention_mask": list(full_tokens["attention_mask"]),
        "labels": labels,
    }


def _tokenize_legacy_example(tokenizer, text: str, max_length: int) -> dict:
    rendered = text.strip()
    if tokenizer.eos_token and not rendered.endswith(tokenizer.eos_token):
        rendered = f"{rendered}{tokenizer.eos_token}"
    tokenized = tokenizer(
        rendered,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {
        "input_ids": list(tokenized["input_ids"]),
        "attention_mask": list(tokenized["attention_mask"]),
        "labels": list(tokenized["input_ids"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Run SFT training."""
    args = parse_args(argv)
    ensure_validated_stack(require_trl=False)
    runtime = resolve_runtime(args.device, args.dtype)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model, padding_side="right")
    ensure_tokenizer_has_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=runtime.torch_dtype,
    )
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=str(args.data))["train"]

    def tokenize_example(example: dict) -> dict:
        if "prompt" in example and "response" in example:
            return _tokenize_structured_example(
                tokenizer,
                example["prompt"],
                example["response"],
                args.max_length,
            )
        if "text" in example:
            return _tokenize_legacy_example(tokenizer, example["text"], args.max_length)
        raise ValueError(
            "Instruction rows must contain canonical (`prompt`, `response`) fields or legacy `text`."
        )

    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch",
        bf16=runtime.bf16,
        fp16=runtime.fp16,
        use_cpu=runtime.use_cpu,
        use_mps_device=runtime.use_mps_device,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SFT checkpoint saved to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
