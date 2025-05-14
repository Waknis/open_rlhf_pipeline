"""Supervised fine-tuning script for a causal language model."""

import argparse
import pathlib

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


def main() -> None:
    """Main function to run SFT."""
    ap = argparse.ArgumentParser(
        description="Supervised fine-tuning of a causal language model."
    )
    ap.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model ID or local path to the pre-trained model.",
    )
    ap.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/instruct.jsonl"),
        help="Path to the JSONL file containing instruction data with a 'text' field.",
    )
    ap.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/sft"),
        help="Directory to save the fine-tuned model and tokenizer.",
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
    # Add padding token if it doesn't exist (e.g., for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    )

    # Load and tokenize dataset
    dataset = load_dataset("json", data_files=str(args.data))["train"]

    def tokenize(ex):
        tokens = tokenizer(ex["text"] + tokenizer.eos_token, truncation=True)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(
        tokenize, batched=False, remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format("torch")

    # Ensure per_device_train_batch_size is not greater than the dataset size
    effective_batch_size = 1  # Start with a small batch size for tiny datasets
    if len(tokenized_dataset) >= 2:
        effective_batch_size = 2

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=1,  # For tiny datasets and quick runs, 1 is fine.
        learning_rate=5e-5,  # A common starting point for SFT
        logging_steps=1,  # Log every step for such short runs
        save_strategy="steps",  # Ensure model is saved
        save_steps=args.max_steps,  # Save at the end of training for such short runs
        bf16=training_bf16,
        fp16=False,  # Explicitly disable fp16 if bf16 is not used
        optim="adamw_torch",  # Use a modern optimizer
        # num_train_epochs=1, # max_steps will take precedence
        report_to="none",  # Disable wandb or other reporting for this toy script
        save_total_limit=1,  # Only keep the last checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,  # Pass tokenizer for proper saving
    )

    trainer.train()
    trainer.save_model(
        str(args.output_dir)
    )  # Saves both model and tokenizer if tokenizer is passed to Trainer
    # tokenizer.save_pretrained(str(args.output_dir)) # Not needed if passed to Trainer
    print("✓ SFT done")


if __name__ == "__main__":
    main()
