#!/usr/bin/env python
"""Test script to verify PPO dataloader fixes in train_ppo.py"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Create minimal test data
def create_test_data():
    test_dir = Path(tempfile.mkdtemp())
    test_data_path = test_dir / "test_instruct.jsonl"
    
    # Create minimal instruction data (2 samples)
    test_data = [
        {"text": "What is Python programming?"},
        {"text": "Explain machine learning in simple terms."}
    ]
    
    with open(test_data_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    return test_dir, test_data_path

# Create minimal models
def create_test_models():
    # Use a tiny model for testing
    model_name = "sshleifer/tiny-gpt2"
    
    # Create SFT model directory
    sft_dir = Path(tempfile.mkdtemp()) / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and save a tiny model as SFT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer.save_pretrained(sft_dir)
    model.save_pretrained(sft_dir)
    
    # Create RM model directory
    rm_dir = Path(tempfile.mkdtemp()) / "rm"
    rm_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1
    )
    
    reward_model.save_pretrained(rm_dir)
    tokenizer.save_pretrained(rm_dir)
    
    return sft_dir, rm_dir

def main():
    print("Setting up test environment...")
    
    # Create test data and models
    test_dir, test_data_path = create_test_data()
    sft_dir, rm_dir = create_test_models()
    
    # Create output directory
    ppo_output_dir = Path(tempfile.mkdtemp()) / "ppo"
    ppo_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Test data created at: {test_data_path}")
    print(f"SFT model created at: {sft_dir}")
    print(f"RM model created at: {rm_dir}")
    print(f"PPO output will be saved to: {ppo_output_dir}")
    
    # Run the train_ppo.py script with our test data
    cmd = (
        f"python scripts/train_ppo.py "
        f"--policy {sft_dir} "
        f"--reward_model_path {rm_dir} "
        f"--instruct_data {test_data_path} "
        f"--output_dir {ppo_output_dir} "
        f"--steps 2 "
        f"--batch_size 2 "
        f"--mini_batch_size 1 "
        f"--ppo_epochs 1"
    )
    
    print(f"Running command: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n✅ Test passed! PPO training completed successfully.")
        print(f"PPO model saved to: {ppo_output_dir}")
    else:
        print(f"\n❌ Test failed with exit code {exit_code}")
    
    # Clean up (uncomment if you want to remove test directories)
    # shutil.rmtree(test_dir, ignore_errors=True)
    # shutil.rmtree(sft_dir.parent, ignore_errors=True)
    # shutil.rmtree(rm_dir.parent, ignore_errors=True)
    # shutil.rmtree(ppo_output_dir.parent, ignore_errors=True)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 