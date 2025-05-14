"""Train a policy model using Proximal Policy Optimization (PPO)."""

import argparse
import pathlib

# import re # Not used if prompt extraction is direct from "text"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
)
from trl import PPOConfig, PPOTrainer
from tqdm import tqdm


# extract_prompt_from_text might not be needed if "text" field is the direct prompt.
# def extract_prompt_from_text(text_field: str) -> str:
#     """Extracts the prompt part from a formatted text field."""
#     parts = text_field.split("\\nOutput:", 1)
#     return parts[0]


def main() -> None:
    """Main function to run PPO training."""
    ap = argparse.ArgumentParser(description="Train a policy model using PPO.")
    ap.add_argument(
        "--policy",  # This is the SFT model path
        type=str,
        required=True,
        help="Path to the SFT model (policy). E.g., models/sft",
    )
    ap.add_argument(
        "--reward_model_path",
        type=str,
        required=True,
        help="Path to the reward model. E.g., models/rm",
    )
    ap.add_argument(
        "--instruct_data",  # Changed from prompts_data to match SFT script
        type=pathlib.Path,
        default=pathlib.Path("data/processed/instruct.jsonl"),
        help="Path to JSONL file with instruction prompts (expects 'text' field).",
    )
    ap.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("models/ppo"),
        help="Directory to save the trained PPO model.",
    )
    ap.add_argument(
        "--steps", type=int, default=2, help="Number of PPO optimization steps."
    )
    ap.add_argument(  # bf16 flag kept but will be overridden by float32 for this task
        "--bf16",
        action="store_true",
        help="Enable bfloat16 for the policy model if CUDA is available (overridden to float32 for this task).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="PPO batch size (rollout buffer size).",
    )
    ap.add_argument(
        "--mini_batch_size",
        type=int,
        default=2,
        help="PPO mini batch size for gradient updates.",
    )
    ap.add_argument(
        "--ppo_epochs", type=int, default=4, help="PPO epochs over the rollout buffer."
    )
    ap.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for PPO optimizer.",
    )
    ap.add_argument(
        "--max_prompt_len",
        type=int,
        default=128,
        help="Maximum length for prompt tokenization.",
    )
    ap.add_argument(
        "--max_gen_len",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    # args.epochs is not used by this PPO script structure which is step-based.
    # If it were, it would need to be added to argparse.

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # For this task, force float32 for CPU execution
    model_dtype = torch.float32
    print("Forcing float32 for policy and value models for CPU execution.")

    # Load policy tokenizer
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
        # Also update pad_token_id in tokenizer if it was None
        if policy_tokenizer.pad_token_id is None:
            policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

    # 1. Model Setup
    # Policy Model
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy,
        torch_dtype=model_dtype,
    )
    # Value Model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.policy,  # Initialize from SFT model path
        num_labels=1,
        torch_dtype=model_dtype,
    )
    # Reward Model
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    if (
        reward_tokenizer.pad_token is None
    ):  # Ensure reward tokenizer also has a pad token
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        if reward_tokenizer.pad_token_id is None:
            reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,  # Ensure RM is for regression
        torch_dtype=model_dtype,  # Also use float32 for RM for consistency on CPU
    )

    # Ensure model configs have pad_token_id if tokenizer does.
    for model_obj, tokenizer_obj in [
        (policy_model, policy_tokenizer),
        (value_model, policy_tokenizer),  # Value model uses policy_tokenizer
        (reward_model, reward_tokenizer),
    ]:
        if (
            tokenizer_obj.pad_token_id is not None
            and model_obj.config.pad_token_id is None
        ):
            model_obj.config.pad_token_id = tokenizer_obj.pad_token_id

    # 8. Guard for generation_config (for policy_model)
    if not hasattr(policy_model, "generation_config"):
        policy_model.generation_config = GenerationConfig.from_model_config(
            policy_model.config
        )
    # Ensure essential generation params are consistent
    policy_model.generation_config.eos_token_id = policy_tokenizer.eos_token_id
    policy_model.generation_config.pad_token_id = (
        policy_tokenizer.pad_token_id
    )  # Should be eos

    device = torch.device("cpu")  # Force CPU for this task
    print(f"Models will run on: {device}")
    policy_model.to(device)
    value_model.to(device)
    reward_model.to(device)
    reward_model.eval()  # RM is only for inference

    # 2. Dataset
    raw_dataset = load_dataset("json", data_files=str(args.instruct_data))["train"]

    def preprocess_function(examples):
        prompt_text = examples["text"]
        tokenized_query = policy_tokenizer(
            prompt_text, truncation=True, padding=False, max_length=args.max_prompt_len
        )
        return {
            "input_ids": tokenized_query["input_ids"],
            "attention_mask": tokenized_query["attention_mask"],
        }

    dataset = raw_dataset.map(
        preprocess_function,
        batched=False,
        remove_columns=raw_dataset.column_names  # Ensure only new columns remain
    )
    # Convert dataset to torch format for tensors only
    dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    print(f"Dataset prepared with {len(dataset)} samples.")
    if len(dataset) > 0:
        print(f"First sample of dataset (torch format): {dataset[0]}")

    # 3. Data Collator
    collator = DataCollatorWithPadding(tokenizer=policy_tokenizer)

    # 4. PPOTrainer call
    # PPOConfig: remove_unused_columns is a PPOTrainer arg, not PPOConfig.
    # PPOConfig does not take model_name anymore for this init style.
    # learning_rate, batch_size, mini_batch_size, ppo_epochs are main ones here.
    # actual_batch_size and actual_mini_batch_size for PPOConfig
    # Use len(dataset) for dynamic batch sizing, crucial for small datasets

    # Make sure dataset is not empty before trying to use its length
    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError(
            "Dataset is empty after preprocessing. Cannot proceed with PPO training."
        )

    effective_batch_size = min(args.batch_size, dataset_len)
    effective_mini_batch_size = min(
        args.mini_batch_size, effective_batch_size
    )  # mini_batch_size <= batch_size

    ppo_config_params = {
        "learning_rate": args.learning_rate,
        "batch_size": effective_batch_size,
        "mini_batch_size": effective_mini_batch_size,
        "ppo_epochs": args.ppo_epochs,
        # "kl_penalty": "kl", # Default
        # "adap_kl_ctrl": False, # Default
        # "init_kl_coef": 0.2, # Default
        # "target": 6, # Default for target KL
        # "horizon": 10000, # Default
        "gamma": 1,  # Default, but can be tuned
        "lam": 0.95,  # Default, but can be tuned
        # "cliprange": 0.2, # Default
        # "cliprange_value": 0.2, # Default
        # "vf_coef": 0.1, # Default
        # "log_with": None, # Default is None
        "seed": 42,  # For reproducibility
        # `steps` was an invalid PPOConfig key, it's for the outer training loop.
    }
    # Remove keys from PPOConfig that are not valid for TRL 0.17.0 or handled by PPOTrainer directly
    # For TRL 0.17.0, PPOConfig mostly inherits from TrainingArguments, so many specific PPO params might be directly on PPOTrainer
    # or have different names in PPOConfig. The provided spec for PPOConfig is minimal: batch_size, mini_batch_size, ppo_epochs.
    # We will stick to these + learning_rate.

    # Minimal PPOConfig based on user spec
    ppo_config = PPOConfig(
        batch_size=effective_batch_size,
        mini_batch_size=effective_mini_batch_size,
        num_ppo_epochs=args.ppo_epochs, # Explicitly set using PPOConfig's own param
        learning_rate=args.learning_rate, # Explicitly set
        dataloader_drop_last=False, # Explicitly set, though default is False
        # Defaulting other PPO specific hyperparams like kl_coef, cliprange etc.
    )

    ppo_trainer = PPOTrainer(
        ppo_config,                       # 1st pos: PPOConfig object (maps to 'args' param in signature)
        policy_tokenizer,                 # 2nd pos: Tokenizer (maps to 'processing_class' param)
        model=policy_model,
        ref_model=None,                   # PPOTrainer will create this
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        data_collator=collator, # Use the reinstated collator
    )

    print(f"PPOTrainer initialized. Dataloader length: {len(ppo_trainer.dataloader)}")
    # Check base_dataloader length for accurate count
    if hasattr(ppo_trainer.dataloader, "base_dataloader"):
        print("Using base_dataloader directly for more reliable iteration")
        dataloader_iterator = ppo_trainer.dataloader.base_dataloader
        real_dataloader_len = len(dataloader_iterator)
        print(f"Real dataloader length (base_dataloader): {real_dataloader_len}")
        
        # Create a new iterator for preview to avoid consuming the main one
        try:
            preview_iterator = iter(ppo_trainer.dataloader.base_dataloader)
            batch_debug = next(preview_iterator)
            print(f"First batch keys: {batch_debug.keys()}")
        except StopIteration:
            print("Warning: Could not preview batch, iterator is empty")
            
    else:
        print("No base_dataloader found, using standard dataloader")
        dataloader_iterator = ppo_trainer.dataloader
        
    # 5. Training loop
    for step, batch in tqdm(
        enumerate(dataloader_iterator), total=args.steps, desc="PPO Steps"
    ):
        if step >= args.steps:
            break
            
        # Check if batch is None (defensive programming)
        if batch is None:
            print("Warning: Received None batch. Skipping this iteration.")
            continue
            
        query_input_ids = batch["input_ids"].to(device)
        # attention_mask = batch["attention_mask"].to(device) # Not explicitly used by generate/step but good practice
        
        # Decode the query input_ids to get prompt texts
        prompt_texts = policy_tokenizer.batch_decode(
            query_input_ids, skip_special_tokens=True
        )

        # Generation kwargs for ppo_trainer.generate
        # eos_token_id and pad_token_id are critical for controlled generation
        generation_kwargs_loop = {
            "max_new_tokens": args.max_gen_len,
            "eos_token_id": policy_tokenizer.eos_token_id,
            "pad_token_id": policy_tokenizer.pad_token_id,  # Usually eos_token_id
            "do_sample": True,  # Important for exploration in PPO
            "top_k": 0.0,  # Disable top-k
            "top_p": 1.0,  # Disable nucleus
            # "temperature": 0.7, # Can add if desired
        }

        response_tensors = ppo_trainer.generate(
            query_input_ids,  # Should be on device already from dataloader if device is set for trainer
            **generation_kwargs_loop,
        )
        # response_tensors are on device. query_input_ids are also on device.

        # Decode responses (generated part only, as generate should take care of not returning prompt)
        response_texts = policy_tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        # Prepare texts for Reward Model
        combined_texts_for_rm = [p + r for p, r in zip(prompt_texts, response_texts)]

        rm_inputs = reward_tokenizer(
            combined_texts_for_rm,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len
            + args.max_gen_len,  # Ensure RM can see full text
        ).to(device)

        with torch.no_grad():
            # Ensure reward_model is on the correct device (should be already)
            rewards_logits = reward_model(**rm_inputs).logits
            # Rewards should be a list of float scalars or 1D tensor
            rewards = rewards_logits.squeeze(-1).tolist()

        # PPO Step
        # Ensure all inputs to step are on the correct device.
        # query_input_ids from batch (should be on device if dataloader is Accelerate-aware)
        # response_tensors from generate (should be on device)
        # rewards (list of Python floats/tensors)
        stats = ppo_trainer.step(query_input_ids, response_tensors, rewards)

        # Log stats
        log_dict = {
            "step": step,
            "KL": stats.get("objective/kl", 0),
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0,
        }
        if "ppo/learning_rate" in stats:
            log_dict["lr"] = stats["ppo/learning_rate"]
        tqdm.write(f"Stats: {log_dict}")

    print("✓ PPO training loop finished.")

    # Save model and tokenizer
    print(f"Saving final PPO policy model and tokenizer to {args.output_dir}...")
    policy_model.save_pretrained(args.output_dir)
    policy_tokenizer.save_pretrained(args.output_dir)

    # If value model needs to be saved separately (e.g., for inspection or reuse)
    # value_model_path = args.output_dir / "value_model"
    # value_model_path.mkdir(exist_ok=True)
    # value_model.save_pretrained(value_model_path)
    # print(f"✓ PPO value model saved to {value_model_path}")

    print("✓ PPO done")


if __name__ == "__main__":
    main()
