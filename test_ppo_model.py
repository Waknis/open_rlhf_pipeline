#!/usr/bin/env python
"""Test script to verify the trained PPO model can be loaded and used."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading the PPO model and tokenizer...")
    model_path = "models/ppo"
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure the model is in eval mode
    model.eval()
    
    # Test prompts
    test_prompts = [
        "What is RLHF and why is it useful?",
        "Explain the difference between PPO and DPO in 1 sentence."
    ]
    
    print("Generating responses for test prompts:")
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest prompt {i+1}: {prompt}")
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate a response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
    
    print("\nModel test completed successfully!")

if __name__ == "__main__":
    main() 