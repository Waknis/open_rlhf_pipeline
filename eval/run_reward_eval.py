"""Evaluate whether a trained reward model ranks preferred answers higher."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
from transformers import AutoModelForSequenceClassification

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.formatting import render_prompt_response  # noqa: E402
from open_rlhf.io import canonicalize_preference_record, load_jsonl  # noqa: E402
from open_rlhf.runtime import (  # noqa: E402
    add_runtime_args,
    ensure_validated_stack,
    load_tokenizer,
    resolve_runtime,
    seed_everything,
)


@torch.inference_mode()
def evaluate_pairs(model, tokenizer, records, *, batch_size, max_length):
    """Score complete transcripts using the same length filter as reward training."""
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive.")
    results = []
    skipped = 0
    for start in range(0, len(records), batch_size):
        encodings = []
        indices = []
        for index, row in enumerate(records[start : start + batch_size], start=start):
            row = canonicalize_preference_record(row)
            pair = [
                tokenizer(render_prompt_response(tokenizer, row["prompt"], row[key]))
                for key in ("chosen", "rejected")
            ]
            if any(len(item["input_ids"]) > max_length for item in pair):
                skipped += 1
                continue
            indices.append(index)
            encodings.extend(pair)
        if not encodings:
            continue
        inputs = tokenizer.pad(encodings, padding=True, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits
        if logits.shape != (len(encodings), 1):
            raise ValueError("Expected one scalar reward per transcript.")
        scores = logits[:, 0].float().tolist()
        for index, chosen, rejected in zip(indices, scores[::2], scores[1::2]):
            if not (math.isfinite(chosen) and math.isfinite(rejected)):
                raise ValueError(f"Non-finite reward for pair at index {index}.")
            results.append(
                {
                    "index": index,
                    "chosen_score": chosen,
                    "rejected_score": rejected,
                    "margin": chosen - rejected,
                }
            )

    if not results:
        raise ValueError(f"No pairs to score ({len(records)} supplied, {skipped} over max_length).")
    wins = sum(row["margin"] > 0 for row in results)
    ties = sum(row["margin"] == 0 for row in results)
    return {
        "total_pairs": len(records),
        "scored_pairs": len(results),
        "skipped_over_length": skipped,
        "wins": wins,
        "ties": ties,
        "losses": len(results) - wins - ties,
        "preference_accuracy": wins / len(results),
        "mean_margin": math.fsum(row["margin"] for row in results) / len(results),
        "pairs": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model_path", required=True, help="Trained reward checkpoint.")
    parser.add_argument("--data", required=True, type=pathlib.Path, help="Held-out preference JSONL.")
    parser.add_argument("--output_file", type=pathlib.Path, default=pathlib.Path("eval/reports/reward.json"))
    parser.add_argument("--batch_size", type=int, default=4, help="Pairs per forward pass.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum tokens per transcript.")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    if args.data.resolve() == args.output_file.resolve():
        parser.error("Choose an output file separate from the input data.")
    ensure_validated_stack(require_trl=False)
    runtime = resolve_runtime(args.device, args.dtype)
    seed_everything(args.seed)
    tokenizer = load_tokenizer(args.model)
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=runtime.torch_dtype, output_loading_info=True
    )
    if model.config.num_labels != 1 or loading_info["missing_keys"]:
        raise ValueError("Supply a saved reward checkpoint with a trained scalar classification head.")
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(runtime.device).eval()
    report = evaluate_pairs(
        model, tokenizer, load_jsonl(args.data), batch_size=args.batch_size, max_length=args.max_length
    )
    report.update(model=args.model, data=str(args.data), max_length=args.max_length)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Reward preference accuracy: {report['wins']}/{report['scored_pairs']} "
        f"({report['preference_accuracy']:.1%}). Report: {args.output_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
