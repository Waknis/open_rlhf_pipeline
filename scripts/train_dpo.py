"""Train a policy model with Direct Preference Optimization."""

from __future__ import annotations

import argparse
import pathlib
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.formatting import render_prompt  # noqa: E402
from open_rlhf.runtime import (  # noqa: E402
    add_runtime_args,
    ensure_validated_stack,
    load_tokenizer,
    resolve_runtime,
    seed_everything,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a policy model using DPO.")
    parser.add_argument(
        "--policy",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base Hugging Face model ID or local checkpoint.",
    )
    parser.add_argument(
        "--pairs",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/pairs.jsonl"),
        help="Processed preference data with canonical (`prompt`, `chosen`, `rejected`) fields.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/dpo"),
        help="Directory where the trained DPO model will be saved.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Maximum optimization steps.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta hyperparameter.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        help="Maximum prompt length used by DPO tokenization.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=256,
        help="Maximum completion length used by DPO tokenization.",
    )
    add_runtime_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run DPO training."""
    args = parse_args(argv)
    ensure_validated_stack(require_trl=True)
    runtime = resolve_runtime(args.device, args.dtype)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from trl import DPOConfig, DPOTrainer

    tokenizer = load_tokenizer(args.policy, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        args.policy,
        torch_dtype=runtime.torch_dtype,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.policy,
        torch_dtype=runtime.torch_dtype,
    )
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None and ref_model.config.pad_token_id is None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=str(args.pairs))["train"]
    required_columns = {"prompt", "chosen", "rejected"}
    if not required_columns.issubset(set(dataset.column_names)):
        raise ValueError(
            f"DPO training expects canonical preference rows with {sorted(required_columns)}. "
            f"Found {dataset.column_names}."
        )

    def prepare_example(example: dict) -> dict:
        return {
            "prompt": render_prompt(tokenizer, example["prompt"]),
            "chosen": example["chosen"].strip(),
            "rejected": example["rejected"].strip(),
        }

    prepared_dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
    )

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.steps),
        save_total_limit=1,
        report_to="none",
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_prompt_length + args.max_completion_length,
        bf16=runtime.bf16,
        fp16=runtime.fp16,
        use_cpu=runtime.use_cpu,
        use_mps_device=runtime.use_mps_device,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=prepared_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"DPO model saved to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
