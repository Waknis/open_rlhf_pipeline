"""Run dummy evaluations on a trained model and generate a markdown report."""

import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DUMMY_TASKS = {
    "MMLU": [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
    ],
    "ARC-Easy": [
        "Which planet is known as the Red Planet?",
        "What force keeps us on the ground?",
    ],
    "TruthfulQA": [
        "What happens if you mix baking soda and vinegar?",
        "Can birds fly to the moon?",
    ],
    "AlpacaEval": [
        "Write a short story about a friendly robot.",
        "Summarize the plot of 'Hamlet' in three sentences.",
    ],
}


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    generation_config: GenerationConfig,
) -> str:
    """Generates a response from the model for a given prompt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # Ensure attention_mask is passed if inputs includes it (tokenizer usually provides it)
    gen_kwargs = {"attention_mask": inputs.get("attention_mask", None)}

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            generation_config=generation_config,
            **gen_kwargs,
        )

    # Decode only the newly generated tokens, not the prompt
    response_text = tokenizer.decode(
        output_sequences[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return response_text.strip()


def main() -> None:
    """Main function to run evaluations."""
    ap = argparse.ArgumentParser(
        description="Run dummy evaluations and generate a report."
    )
    ap.add_argument(
        "--model_path",
        type=pathlib.Path,
        required=True,
        help="Path to the trained Hugging Face model (e.g., models/ppo or models/dpo).",
    )
    ap.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("eval/reports"),
        help="Directory to save the evaluation report.",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Load model in bfloat16 if CUDA is available (for faster inference on compatible GPUs).",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float32
    if args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print("Using bfloat16 for model loading.")
    else:
        print("Using float32 for model loading.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=model_dtype
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:  # Important for generation
            model.config.pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(
        max_new_tokens=100,
        min_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    report_lines = [
        "# Evaluation Report",
        "",
        f"Model: `{args.model_path}`",
        "",
        "| Task         | Question                                              | Model Response                                     |",
        "|--------------|-------------------------------------------------------|----------------------------------------------------|",
    ]

    print(f"Generating responses for model: {args.model_path}")
    for task_name, questions in DUMMY_TASKS.items():
        print(f"  Evaluating task: {task_name}...")
        for q_idx, question in enumerate(questions):
            # For chat models, prompt might need specific formatting. Assuming generic completion for now.
            # E.g. for TinyLlama chat: prompt = f"<|user|>\n{question}<|assistant|>\n"
            # For this general script, we use the question directly as prompt.
            prompt = question
            response = generate_response(
                model, tokenizer, prompt, device, generation_config
            )
            # Truncate long responses for display in table
            display_response = (
                (response[:150] + "...") if len(response) > 150 else response
            )
            display_response = display_response.replace(
                "|", "\|"
            )  # Escape pipe characters for markdown
            display_question = question.replace("|", "\|")
            report_lines.append(
                f"| {task_name:<12} | {display_question:<53} | {display_response:<50} |"
            )
            print(f"    Q{q_idx+1}: {question}")
            print(f"    R{q_idx+1}: {response[:80]}...")

    report_path = args.output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✓ Evaluation report written to {report_path}")


if __name__ == "__main__":
    main()
