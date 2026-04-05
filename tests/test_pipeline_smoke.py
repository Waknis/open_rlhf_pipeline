"""End-to-end smoke tests for the Open RLHF pipeline scripts."""

from __future__ import annotations

import json
from pathlib import Path

from eval.run_eval import main as run_eval_main
from eval.run_safety import main as run_safety_main
from scripts.train_dpo import main as train_dpo_main
from scripts.train_ppo import main as train_ppo_main

from tests.conftest import assert_model_artifacts


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_sft_masks_prompt_tokens(sft_masking_example):
    labels = sft_masking_example["labels"]
    assert labels[0] == -100
    assert any(label != -100 for label in labels)


def test_dpo_training_smoke(processed_data_dir, sft_checkpoint, tmp_path):
    output_dir = tmp_path / "dpo"
    exit_code = train_dpo_main(
        [
            "--policy",
            str(sft_checkpoint),
            "--pairs",
            str(processed_data_dir / "pairs.jsonl"),
            "--output_dir",
            str(output_dir),
            "--steps",
            "1",
            "--per_device_train_batch_size",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    assert exit_code == 0
    assert_model_artifacts(output_dir)


def test_ppo_training_smoke(processed_data_dir, sft_checkpoint, reward_checkpoint, tmp_path):
    output_dir = tmp_path / "ppo"
    exit_code = train_ppo_main(
        [
            "--policy",
            str(sft_checkpoint),
            "--reward_model_path",
            str(reward_checkpoint),
            "--instruct_data",
            str(processed_data_dir / "instruct.jsonl"),
            "--output_dir",
            str(output_dir),
            "--steps",
            "1",
            "--batch_size",
            "1",
            "--mini_batch_size",
            "1",
            "--ppo_epochs",
            "1",
            "--max_prompt_len",
            "64",
            "--max_gen_len",
            "16",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    assert exit_code == 0
    assert_model_artifacts(output_dir)
    assert_model_artifacts(output_dir / "value_model")


def test_eval_smoke_writes_reports(sft_checkpoint, tmp_path):
    output_dir = tmp_path / "eval"
    exit_code = run_eval_main(
        [
            "--model",
            str(sft_checkpoint),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "report.md").exists()
    responses = _read_jsonl(output_dir / "responses.jsonl")
    assert responses
    assert {"task", "prompt", "response"} <= set(responses[0])


def test_safety_smoke_uses_cli_alias(sft_checkpoint, tmp_path):
    output_file = tmp_path / "safety.jsonl"
    exit_code = run_safety_main(
        [
            "--model_path",
            str(sft_checkpoint),
            "--output_file",
            str(output_file),
            "--scorer",
            "keyword",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    assert exit_code == 0
    rows = _read_jsonl(output_file)
    assert rows
    assert {"prompt", "response", "toxicity_score", "scorer"} <= set(rows[0])
