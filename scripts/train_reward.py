"""Train a reward model on canonical preference pairs."""

from __future__ import annotations

import argparse
import pathlib
import sys

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.formatting import render_prompt_response  # noqa: E402
from open_rlhf.runtime import (  # noqa: E402
    add_runtime_args,
    ensure_validated_stack,
    load_tokenizer,
    resolve_runtime,
    seed_everything,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a reward model on preference pairs.")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base Hugging Face model ID or local checkpoint used for the reward head.",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/pairs.jsonl"),
        help="Processed preference data with canonical (`prompt`, `chosen`, `rejected`) fields.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/rm"),
        help="Directory where the reward model will be saved.",
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
        default=5e-6,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length used by the default reward data collator.",
    )
    add_runtime_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run reward modeling."""
    args = parse_args(argv)
    ensure_validated_stack(require_trl=True)
    runtime = resolve_runtime(args.device, args.dtype)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from trl import RewardConfig, RewardTrainer

    tokenizer = load_tokenizer(args.model, padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        torch_dtype=runtime.torch_dtype,
    )
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=str(args.data))["train"]
    required_columns = {"prompt", "chosen", "rejected"}
    if not required_columns.issubset(set(dataset.column_names)):
        raise ValueError(
            f"Reward training expects canonical preference rows with {sorted(required_columns)}. "
            f"Found {dataset.column_names}."
        )

    def prepare_example(example: dict) -> dict:
        return {
            "chosen": render_prompt_response(tokenizer, example["prompt"], example["chosen"]),
            "rejected": render_prompt_response(tokenizer, example["prompt"], example["rejected"]),
        }

    prepared_dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
    )

    training_args = RewardConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        save_total_limit=1,
        report_to="none",
        max_length=args.max_length,
        bf16=runtime.bf16,
        fp16=runtime.fp16,
        use_cpu=runtime.use_cpu,
        use_mps_device=runtime.use_mps_device,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Reward model saved to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
