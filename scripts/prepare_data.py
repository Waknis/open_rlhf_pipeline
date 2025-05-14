"""Prepare instruction and preference datasets for RLHF training.

Reads raw JSONL files, processes them, and saves them to the output directory.
Instruction data is formatted into a single 'text' field per example.
Preference data is saved as pairs of 'chosen' and 'rejected' responses.
"""

import argparse
import json
import pathlib
import random


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """Loads a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary is a record from the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save_jsonl(records: list[dict], path: pathlib.Path) -> None:
    """Saves a list of records to a JSONL file.

    Args:
        records: A list of dictionaries to save.
        path: Path to the output JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_instruction(item: dict) -> dict:
    """Formats an item from UltraChat-like data into a single 'text' field."""
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    output = item.get("output", "")

    if inp:
        text = f"Instruction: {instruction}\nInput: {inp}\nOutput: {output}"
    else:
        text = f"Instruction: {instruction}\nOutput: {output}"
    return {"text": text}


def main():
    """Main function to prepare data."""
    ap = argparse.ArgumentParser(
        description="Prepare instruction and preference datasets."
    )
    ap.add_argument(
        "--instruct_data",
        type=pathlib.Path,
        required=True,
        help="Path to the raw instruction JSONL file (e.g., UltraChat format).",
    )
    ap.add_argument(
        "--preference_data",
        type=pathlib.Path,
        required=True,
        help="Path to the raw preference JSONL file (e.g., OpenHermes_pairs format).",
    )
    ap.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("data/processed"),
        help="Directory to save processed data.",
    )
    ap.add_argument(
        "--max_instruct_rows",
        type=int,
        default=5000,
        help="Maximum number of rows for instruction data.",
    )
    ap.add_argument(
        "--max_preference_rows",
        type=int,
        default=5000,
        help="Maximum number of rows for preference data.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Process instruction data
    raw_instruct_data = load_jsonl(args.instruct_data)
    random.shuffle(raw_instruct_data)
    processed_instruct_data = [
        format_instruction(item) for item in raw_instruct_data[: args.max_instruct_rows]
    ]
    instruct_out_path = args.out_dir / "instruct.jsonl"
    save_jsonl(processed_instruct_data, instruct_out_path)

    # Process preference data
    raw_preference_data = load_jsonl(args.preference_data)
    random.shuffle(raw_preference_data)

    # Rename "prompt" to "text" if "prompt" exists and "text" doesn't
    processed_preference_list = []
    for item in raw_preference_data[: args.max_preference_rows]:
        new_item = item.copy()
        if "prompt" in new_item and "text" not in new_item:
            new_item["text"] = new_item.pop("prompt")
        processed_preference_list.append(new_item)

    preference_out_path = args.out_dir / "pairs.jsonl"
    save_jsonl(processed_preference_list, preference_out_path)

    print(f"✓ Data written to {args.out_dir}")


if __name__ == "__main__":
    main()
