"""Run sample-generation smoke evaluation on a trained model."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.formatting import render_prompt  # noqa: E402
from open_rlhf.runtime import (  # noqa: E402
    add_runtime_args,
    ensure_validated_stack,
    load_tokenizer,
    make_torch_device,
    resolve_runtime,
    seed_everything,
)

SAMPLE_TASKS = {
    "factual": [
        "What is the capital of France?",
        "Name one use of supervised fine-tuning.",
    ],
    "reasoning": [
        "If a model overfits, what is one likely symptom?",
        "Why might PPO be less stable than DPO on a tiny dataset?",
    ],
    "helpfulness": [
        "Write a short explanation of RLHF for a junior engineer.",
        "Summarize the difference between a reward model and a policy model in two sentences.",
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run sample-generation smoke evaluation.")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=None,
        help="Path to the trained Hugging Face model checkpoint.",
    )
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        default=None,
        help="Deprecated alias for --model.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("eval/reports"),
        help="Directory where the sample report will be written.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per prompt.",
    )
    add_runtime_args(parser)
    return parser.parse_args(argv)


def _resolve_model_path(args: argparse.Namespace) -> pathlib.Path:
    model_path = args.model or args.model_path
    if model_path is None:
        raise SystemExit("Provide `--model` (or the deprecated alias `--model_path`).")
    return model_path


def _generate_response(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    prompt_text = render_prompt(tokenizer, prompt)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            generation_config=generation_config,
        )
    response_ids = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    """Run sample-generation smoke evaluation and write a report."""
    args = parse_args(argv)
    ensure_validated_stack(require_trl=False)
    runtime = resolve_runtime(args.device, args.dtype)
    seed_everything(args.seed)
    model_path = _resolve_model_path(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(str(model_path), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=runtime.torch_dtype,
    )
    device = make_torch_device(runtime)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    results: list[dict[str, str]] = []
    for task_name, prompts in SAMPLE_TASKS.items():
        for prompt in prompts:
            results.append(
                {
                    "task": task_name,
                    "prompt": prompt,
                    "response": _generate_response(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        args.max_new_tokens,
                    ),
                }
            )

    report_lines = [
        "# Sample Generation Report",
        "",
        "This report is a smoke test of prompt formatting and generation. It is not a benchmark score.",
        "",
        f"Model: `{model_path}`",
        "",
        "| Category | Prompt | Response |",
        "|---|---|---|",
    ]
    for record in results:
        prompt = record["prompt"].replace("|", "\\|")
        response = record["response"].replace("|", "\\|")
        report_lines.append(f"| {record['task']} | {prompt} | {response} |")

    report_path = args.output_dir / "report.md"
    jsonl_path = args.output_dir / "responses.jsonl"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Sample evaluation written to {report_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
