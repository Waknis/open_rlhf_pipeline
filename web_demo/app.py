import argparse

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_chat_fn(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)  # Use float32 for CPU

    def chat(message, chat_history):
        inputs = tokenizer(message, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    return chat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    chat_fn = build_chat_fn(args.model)
    gr.ChatInterface(chat_fn, title="Open RLHF Assistant").launch()


if __name__ == "__main__":
    main()
