"""Train a reward model on preference pairs."""

import argparse
import pathlib

import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments)
from trl import RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding


def main() -> None:
    """Main function to train the reward model."""
    ap = argparse.ArgumentParser(
        description="Train a reward model on preference pairs."
    )
    ap.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Can also be a path to a local SFT model
        help="Hugging Face model ID or local path to the base model for the reward head.",
    )
    ap.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/pairs.jsonl"),
        help="Path to the JSONL file containing preference pairs ('chosen' and 'rejected' fields).",
    )
    ap.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/rm"),
        help="Directory to save the trained reward model.",
    )
    ap.add_argument(
        "--max_steps", type=int, default=2, help="Maximum number of training steps."
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 training if CUDA is available.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        # Set pad token for models like Llama, needed for padding during tokenization
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure model's config also reflects this if it's used for generation later (not crucial for RM training)
        # model_config = AutoConfig.from_pretrained(args.model)
        # model_config.pad_token_id = tokenizer.pad_token_id

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

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,  # Regression head for reward score
        torch_dtype=model_dtype,
    )
    # If pad_token was added to tokenizer, and model has a config for pad_token_id, update it.
    # This can be important if the model was not initialized with a pad_token_id in its config.
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load and preprocess dataset
    dataset = load_dataset("json", data_files=str(args.data))["train"]

    def preprocess_function(examples):
        """Tokenizes chosen and rejected texts for the RewardTrainer."""
        tokenized_chosen = tokenizer(
            examples["chosen"], truncation=True, padding="max_length", max_length=256
        )
        tokenized_rejected = tokenizer(
            examples["rejected"], truncation=True, padding="max_length", max_length=256
        )
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,  # Remove original 'chosen', 'rejected' text columns
    )
    # RewardTrainer expects columns to be torch tensors
    tokenized_dataset.set_format("torch")

    # Ensure per_device_train_batch_size is not greater than the dataset size
    effective_batch_size = 1
    # if len(tokenized_dataset) >= 2:
    #     effective_batch_size = 2

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=effective_batch_size,  # For RewardTrainer, this is pairs per batch
        gradient_accumulation_steps=1,
        learning_rate=5e-6,  # Typically lower LR for RM than SFT
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.max_steps,
        bf16=training_bf16,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False, # Explicitly set to False for RewardTrainer
    )

    # TRL's RewardTrainer (as of 0.17.0) checks for args.disable_dropout internally.
    # Standard TrainingArguments doesn't have this, so we add it.
    if not hasattr(training_args, 'disable_dropout'):
        training_args.disable_dropout = False
    # TRL's RewardTrainer also checks for args.center_rewards_coefficient.
    if not hasattr(training_args, 'center_rewards_coefficient'):
        training_args.center_rewards_coefficient = None

    # Explicitly create the data collator if RewardTrainer's default expects processing_class implicitly
    # and doesn't take tokenizer directly in the constructor for this version of TRL.
    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator, # Pass the instantiated collator
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))  # Saves model
    tokenizer.save_pretrained(str(args.output_dir)) # Manually save tokenizer as it's not passed to trainer
    print("✓ Reward done")


if __name__ == "__main__":
    main()
