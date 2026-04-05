"""Prepare canonical instruction and preference datasets for RLHF training."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rlhf.io import (  # noqa: E402
    canonicalize_instruction_record,
    canonicalize_preference_record,
    load_jsonl,
    save_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare canonical instruction and preference datasets."
    )
    parser.add_argument(
        "--instruct_data",
        type=pathlib.Path,
        required=True,
        help="Path to raw instruction data with (`instruction`, `output`) or (`prompt`, `response`) fields.",
    )
    parser.add_argument(
        "--preference_data",
        type=pathlib.Path,
        required=True,
        help="Path to raw preference data with (`prompt`, `chosen`, `rejected`) fields.",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("data/processed"),
        help="Directory where processed JSONL files will be written.",
    )
    parser.add_argument(
        "--max_instruct_rows",
        type=int,
        default=5000,
        help="Maximum number of instruction records to keep.",
    )
    parser.add_argument(
        "--max_preference_rows",
        type=int,
        default=5000,
        help="Maximum number of preference records to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling input records.",
    )
    return parser.parse_args(argv)


def _sample_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:limit]


def main(argv: list[str] | None = None) -> int:
    """Run data preparation and write the canonical processed files."""
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    instruct_records = _sample_records(
        load_jsonl(args.instruct_data), args.max_instruct_rows, args.seed
    )
    preference_records = _sample_records(
        load_jsonl(args.preference_data), args.max_preference_rows, args.seed
    )

    processed_instruct = [
        canonicalize_instruction_record(record) for record in instruct_records
    ]
    processed_preferences = [
        canonicalize_preference_record(record) for record in preference_records
    ]

    save_jsonl(processed_instruct, args.out_dir / "instruct.jsonl")
    save_jsonl(processed_preferences, args.out_dir / "pairs.jsonl")

    print(
        f"Prepared {len(processed_instruct)} instruction rows and "
        f"{len(processed_preferences)} preference rows in {args.out_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
