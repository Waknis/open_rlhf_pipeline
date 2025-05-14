"""Train a policy model using Direct Preference Optimization (DPO)."""

import argparse
import pathlib

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer


def main() -> None:
    """Main function to run DPO training."""
    ap = argparse.ArgumentParser(description="Train a policy model using DPO.")
    ap.add_argument(
        "--policy",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Can also be SFT model path
        help="Hugging Face model ID or local path to the base model to be DPO-trained.",
    )
    ap.add_argument(
        "--pairs",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/pairs.jsonl"),
        help="Path to the JSONL file with preference data ('text', 'chosen', 'rejected' fields).",
    )
    ap.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/dpo"),
        help="Directory to save the trained DPO model.",
    )
    ap.add_argument(
        "--steps",  # Renamed from max_steps in user req to be consistent with PPO script arg name
        type=int,
        default=2,
        help="Maximum number of training steps (optimization updates).",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 training if CUDA is available.",
    )
    # DPO-specific HPs, can be exposed more if needed
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta hyperparameter.")
    ap.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="DPO learning rate (often very small).",
    )  # DPO often needs smaller LR
    ap.add_argument(
        "--dpo_batch_size",
        type=int,
        default=1,
        help="DPO per device batch size (pairs per batch).",
    )  # DPO often uses batch_size=1 or 2

    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dtype handling
    model_dtype = torch.float32
    training_bf16 = False
    if args.bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            training_bf16 = True
            print("CUDA and bfloat16 available. Using bfloat16 for model and training.")
        else:
            print(
                "BF16 requested, but CUDA or bfloat16 support not available. Using float32."
            )
    else:
        print("Using float32 for model and training.")

    tokenizer = AutoTokenizer.from_pretrained(args.policy)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # DPO Trainer might also need model.config.pad_token_id if using pad_token for attention masking.
        # print("Set pad_token to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        args.policy,
        torch_dtype=model_dtype,
        # low_cpu_mem_usage=True, # Can be useful for large models
    )
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        # print(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")

    # Load dataset
    # DPOTrainer expects 'prompt', 'chosen', 'rejected' columns.
    # Our pairs.jsonl (from OpenHermes_pairs structure) should have these.
    # UPDATE: pairs.jsonl will now have 'text', 'chosen', 'rejected' due to prepare_data.py change.
    dataset = load_dataset("json", data_files=str(args.pairs))["train"]

    # Rename 'text' to 'prompt' if necessary for DPOTrainer
    if "text" in dataset.column_names and "prompt" not in dataset.column_names:
        dataset = dataset.rename_column("text", "prompt")
        print("Renamed dataset column 'text' to 'prompt' for DPOTrainer.")
    elif "prompt" not in dataset.column_names:
        # This case should ideally not happen if data prep is correct and only one prompt field exists.
        print(
            "Warning: 'prompt' column not found and 'text' column also not found to rename. DPOTrainer might fail."
        )

    # Ensure batch size is not larger than dataset size for toy runs
    # Dataset has 2 lines for the toy example.
    actual_dpo_batch_size = min(args.dpo_batch_size, len(dataset))
    if (
        actual_dpo_batch_size == 0 and len(dataset) > 0
    ):  # handle case where dataset is small but >0
        actual_dpo_batch_size = 1
    elif len(dataset) == 0:
        print("Error: Dataset is empty. Cannot train.")
        return

    # DPO Training Arguments
    # DPOTrainer uses a DPOConfig which subclasses TrainingArguments.
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=actual_dpo_batch_size,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=1,  # Keep it 1 for tiny batch and few steps
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.steps,  # Save at the end of training
        bf16=training_bf16,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False,  # Important as dataset might have other cols from json load
    )

    # Initialize DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        # ref_model=None, # DPOTrainer creates its own reference model if None
        args=training_args,  # Pass TrainingArguments here
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,  # Max sequence length for (prompt + chosen/rejected)
        max_prompt_length=512,  # Max length for prompt part
        # peft_config=None, # Can pass PEFT config for LoRA DPO
    )

    print(f"Starting DPO training for {args.steps} steps...")
    dpo_trainer.train()

    print(f"Saving DPO model to {args.output_dir}...")
    dpo_trainer.save_model(
        str(args.output_dir)
    )  # Saves model and tokenizer (if passed)
    # tokenizer.save_pretrained(str(args.output_dir)) # Not needed if tokenizer passed to DPOTrainer

    print("✓ DPO done")


if __name__ == "__main__":
    main()
