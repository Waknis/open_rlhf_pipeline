"""Shared pytest fixtures for pipeline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_data import main as prepare_data_main
from scripts.train_reward import main as train_reward_main
from scripts.train_sft import _tokenize_structured_example, main as train_sft_main

MODEL_ID = "sshleifer/tiny-gpt2"


def assert_model_artifacts(model_dir: Path) -> None:
    """Assert that a model directory contains basic HF artifacts."""
    assert model_dir.is_dir()
    assert (model_dir / "config.json").exists()
    assert (model_dir / "tokenizer_config.json").exists()
    assert (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()


@pytest.fixture(scope="session")
def raw_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create tiny raw instruction and preference datasets."""
    root = tmp_path_factory.mktemp("raw_data")
    instruct_path = root / "instruct.jsonl"
    preference_path = root / "preferences.jsonl"

    instruct_rows = [
        {
            "instruction": "Explain what unit tests are.",
            "input": "",
            "output": "Unit tests verify small pieces of code in isolation.",
        },
        {
            "instruction": "What is RLHF?",
            "input": "",
            "output": "RLHF aligns a policy model using preference-derived feedback.",
        },
    ]
    preference_rows = [
        {
            "prompt": "What is RLHF?",
            "chosen": "RLHF trains a policy using preference feedback.",
            "rejected": "RLHF is a kind of image upscaler.",
        },
        {
            "prompt": "Why do we use reward models?",
            "chosen": "Reward models score outputs so policy training can optimize for preferred responses.",
            "rejected": "Reward models are only used for tokenization.",
        },
    ]

    for path, rows in ((instruct_path, instruct_rows), (preference_path, preference_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return root


@pytest.fixture(scope="session")
def processed_data_dir(raw_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run prepare_data once and reuse the canonical processed files."""
    out_dir = tmp_path_factory.mktemp("processed_data")
    exit_code = prepare_data_main(
        [
            "--instruct_data",
            str(raw_data_dir / "instruct.jsonl"),
            "--preference_data",
            str(raw_data_dir / "preferences.jsonl"),
            "--out_dir",
            str(out_dir),
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0
    return out_dir


@pytest.fixture(scope="session")
def sft_checkpoint(processed_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny SFT checkpoint for downstream smoke tests."""
    output_dir = tmp_path_factory.mktemp("sft_model")
    exit_code = train_sft_main(
        [
            "--model",
            MODEL_ID,
            "--data",
            str(processed_data_dir / "instruct.jsonl"),
            "--output_dir",
            str(output_dir),
            "--max_steps",
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
    return output_dir


@pytest.fixture(scope="session")
def reward_checkpoint(
    processed_data_dir: Path,
    sft_checkpoint: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Train a tiny reward model checkpoint for PPO smoke tests."""
    output_dir = tmp_path_factory.mktemp("reward_model")
    exit_code = train_reward_main(
        [
            "--model",
            str(sft_checkpoint),
            "--data",
            str(processed_data_dir / "pairs.jsonl"),
            "--output_dir",
            str(output_dir),
            "--max_steps",
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
    return output_dir


@pytest.fixture(scope="session")
def sft_masking_example() -> dict:
    """Expose one tokenized structured example for label-mask assertions."""
    from open_rlhf.runtime import load_tokenizer

    tokenizer = load_tokenizer(MODEL_ID)
    return _tokenize_structured_example(
        tokenizer,
        "Explain RLHF.",
        "RLHF aligns a policy model with preference feedback.",
        128,
    )
