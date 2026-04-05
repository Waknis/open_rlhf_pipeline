"""Train a policy model with PPO using the validated TRL trainer stack."""

from __future__ import annotations

import argparse
import pathlib
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, DataCollatorWithPadding

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
    parser = argparse.ArgumentParser(description="Train a policy model using PPO.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to the SFT policy checkpoint.",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        required=True,
        help="Path to the trained reward model checkpoint.",
    )
    parser.add_argument(
        "--instruct_data",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/instruct.jsonl"),
        help="Instruction data with canonical `prompt` rows.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/ppo"),
        help="Directory where the trained PPO policy will be saved.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of PPO rollout/update batches to run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Rollout batch size collected before PPO minibatching.",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=1,
        help="Minibatch size used inside each PPO update.",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="Number of PPO epochs per rollout batch.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=256,
        help="Maximum prompt length after tokenization.",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=64,
        help="Maximum number of generated response tokens per rollout.",
    )
    add_runtime_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run PPO training."""
    args = parse_args(argv)
    ensure_validated_stack(require_trl=True)
    runtime = resolve_runtime(args.device, args.dtype)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0 or args.mini_batch_size <= 0:
        raise ValueError("`batch_size` and `mini_batch_size` must both be positive.")
    if args.batch_size % args.mini_batch_size != 0:
        raise ValueError("`batch_size` must be divisible by `mini_batch_size` for PPO.")

    from trl import PPOConfig, PPOTrainer

    tokenizer = load_tokenizer(args.policy, padding_side="left")
    reward_tokenizer = load_tokenizer(args.reward_model_path, padding_side="left")
    if tokenizer.get_vocab() != reward_tokenizer.get_vocab():
        raise ValueError(
            "The reward model tokenizer must match the policy tokenizer. "
            "Train the reward model from the same base tokenizer as the policy."
        )

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy,
        torch_dtype=runtime.torch_dtype,
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.policy,
        torch_dtype=runtime.torch_dtype,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.policy,
        num_labels=1,
        torch_dtype=runtime.torch_dtype,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
        torch_dtype=runtime.torch_dtype,
    )

    for model in (policy_model, ref_policy, value_model, reward_model):
        if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=str(args.instruct_data))["train"]

    def prepare_example(example: dict) -> dict:
        prompt = example.get("prompt", example.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "PPO training expects instruction rows with a canonical `prompt` field "
                "(legacy `text` is accepted as a fallback)."
            )
        prompt_text = render_prompt(tokenizer, prompt)
        tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_prompt_len,
            add_special_tokens=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    prepared_dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
    )
    if len(prepared_dataset) == 0:
        raise ValueError("The PPO prompt dataset is empty.")
    eval_dataset = prepared_dataset.select(range(min(len(prepared_dataset), args.batch_size)))

    ppo_config = PPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=1,
        num_mini_batches=args.batch_size // args.mini_batch_size,
        num_ppo_epochs=args.ppo_epochs,
        total_episodes=args.steps * args.batch_size,
        learning_rate=args.learning_rate,
        response_length=args.max_gen_len,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.steps),
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        bf16=runtime.bf16,
        fp16=runtime.fp16,
        use_cpu=runtime.use_cpu,
        use_mps_device=runtime.use_mps_device,
        seed=args.seed,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        processing_class=tokenizer,
        policy=policy_model,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=prepared_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    policy_model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    value_dir = args.output_dir / "value_model"
    value_model.save_pretrained(str(value_dir))
    tokenizer.save_pretrained(str(value_dir))
    print(f"PPO policy saved to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
