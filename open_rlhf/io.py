"""JSONL I/O and canonical dataset normalization helpers."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from open_rlhf.formatting import render_prepared_text


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(records: list[dict[str, Any]], path: pathlib.Path) -> None:
    """Save a list of dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _require_text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string field `{key}` in record: {item}")
    return value.strip()


def canonicalize_instruction_record(item: dict[str, Any]) -> dict[str, str]:
    """Convert an instruction example into the canonical prompt/response/text contract."""
    if "prompt" in item and "response" in item:
        prompt = _require_text(item, "prompt")
        response = _require_text(item, "response")
    elif "instruction" in item and "output" in item:
        instruction = _require_text(item, "instruction")
        input_text = item.get("input", "")
        if input_text:
            prompt = f"{instruction}\n\nInput:\n{str(input_text).strip()}"
        else:
            prompt = instruction
        response = _require_text(item, "output")
    else:
        raise ValueError(
            "Instruction data must contain either (`prompt`, `response`) or "
            "(`instruction`, `output`) fields."
        )

    return {
        "prompt": prompt,
        "response": response,
        "text": render_prepared_text(prompt, response),
    }


def canonicalize_preference_record(item: dict[str, Any]) -> dict[str, str]:
    """Convert a preference example into the canonical prompt/chosen/rejected contract."""
    prompt = item.get("prompt", item.get("text"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "Preference data must contain a non-empty `prompt` field "
            "(or legacy `text` field)."
        )

    return {
        "prompt": prompt.strip(),
        "chosen": _require_text(item, "chosen"),
        "rejected": _require_text(item, "rejected"),
    }

