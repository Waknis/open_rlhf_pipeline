"""Tests for canonical data preparation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_data import main as prepare_data_main


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_prepare_data_is_deterministic(raw_data_dir, tmp_path):
    out_one = tmp_path / "one"
    out_two = tmp_path / "two"

    args = [
        "--instruct_data",
        str(raw_data_dir / "instruct.jsonl"),
        "--preference_data",
        str(raw_data_dir / "preferences.jsonl"),
        "--seed",
        "17",
    ]
    assert prepare_data_main([*args, "--out_dir", str(out_one)]) == 0
    assert prepare_data_main([*args, "--out_dir", str(out_two)]) == 0

    assert _read_jsonl(out_one / "instruct.jsonl") == _read_jsonl(out_two / "instruct.jsonl")
    assert _read_jsonl(out_one / "pairs.jsonl") == _read_jsonl(out_two / "pairs.jsonl")


def test_prepare_data_outputs_canonical_fields(processed_data_dir):
    instruct_rows = _read_jsonl(processed_data_dir / "instruct.jsonl")
    preference_rows = _read_jsonl(processed_data_dir / "pairs.jsonl")

    assert {"prompt", "response", "text"} <= set(instruct_rows[0])
    assert {"prompt", "chosen", "rejected"} <= set(preference_rows[0])

