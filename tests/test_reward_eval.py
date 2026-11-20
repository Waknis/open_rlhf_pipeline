import json
from types import SimpleNamespace

import pytest
import torch

from eval.run_reward_eval import evaluate_pairs, main
from open_rlhf.runtime import load_tokenizer


class LengthScorer:
    device = torch.device("cpu")

    def __call__(self, attention_mask, **kwargs):
        return SimpleNamespace(logits=attention_mask.sum(dim=1, keepdim=True).float())


def test_preference_metrics_and_length_filter(base_checkpoint):
    tokenizer = load_tokenizer(str(base_checkpoint))
    pairs = [
        {"prompt": "?", "chosen": "a much longer answer", "rejected": "x"},
        {"prompt": "?", "chosen": "x", "rejected": "a much longer answer"},
        {"prompt": "?", "chosen": "x", "rejected": "x"},
        {"prompt": "?", "chosen": "x", "rejected": "xyz " * 100},
    ]
    report = evaluate_pairs(LengthScorer(), tokenizer, pairs, batch_size=3, max_length=64)
    assert report == evaluate_pairs(LengthScorer(), tokenizer, pairs, batch_size=1, max_length=64)
    assert (report["total_pairs"], report["scored_pairs"], report["skipped_over_length"]) == (4, 3, 1)
    assert (report["wins"], report["ties"], report["losses"]) == (1, 1, 1)
    assert report["preference_accuracy"] == pytest.approx(1 / 3)
    assert report["mean_margin"] == 0
    assert [row["index"] for row in report["pairs"]] == [0, 1, 2]
    with pytest.raises(ValueError, match="No pairs to score"):
        evaluate_pairs(LengthScorer(), tokenizer, pairs[-1:], batch_size=1, max_length=64)


def test_saved_reward_checkpoint_on_held_out_pair(reward_checkpoint, tmp_path):
    dataset = tmp_path / "held_out.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "A new question?", "chosen": "same", "rejected": "same"}),
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "reward.json"
    args = [
        "--model", str(reward_checkpoint),
        "--data", str(dataset),
        "--output_file", str(output),
        "--device", "cpu",
    ]
    assert main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scored_pairs"] == report["ties"] == 1
    assert report["pairs"][0]["margin"] == 0


def test_causal_checkpoint_cannot_supply_random_reward_head(base_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="trained scalar classification head"):
        main(["--model", str(base_checkpoint), "--data", str(tmp_path / "unused.jsonl"), "--device", "cpu"])
