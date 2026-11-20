"""Regression tests for causal SFT masking and truncation."""

import pytest

from open_rlhf.formatting import render_prompt
from open_rlhf.runtime import load_tokenizer
from scripts.train_sft import SupervisedDataCollator, _tokenize_structured_example


@pytest.fixture
def tokenizer(base_checkpoint):
    return load_tokenizer(str(base_checkpoint))


@pytest.mark.parametrize(
    ("prompt", "response", "window"),
    [
        ("question " * 100, "yes", 16),
        ("Q", "answer " * 100, 32),
        ("question " * 100, "answer " * 100, 15),
        ("Q", "A", 2),
    ],
)
def test_truncation_keeps_recent_context_and_response_start(tokenizer, prompt, response, window):
    row = _tokenize_structured_example(tokenizer, prompt, response, window)
    prompt_ids = tokenizer.encode(render_prompt(tokenizer, prompt), add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + response.strip(), add_special_tokens=False)
    prompt_budget = min(len(prompt_ids), max(window - len(answer_ids), (window + 1) // 2))
    targets = answer_ids[:window - prompt_budget]

    assert row["input_ids"] == prompt_ids[-prompt_budget:] + targets
    assert row["labels"] == [-100] * prompt_budget + targets
    assert row["attention_mask"] == [1] * window
    assert any(label != -100 for label in row["labels"][1:])


def test_subword_merge_at_chat_boundary_keeps_generation_prefix(tokenizer):
    # The byte-pair tokenizer learned "RLHF" as a single token. Completing "RL"
    # with "HF" must keep the existing prompt token intact for causal training.
    tokenizer.chat_template = (
        "{{ messages[0].content }} RL"
        "{% if messages|length > 1 %}{{ messages[1].content }}{% endif %}"
    )
    prompt_ids = tokenizer.encode("Explain RL", add_special_tokens=False)
    answer_ids = tokenizer.encode("HF", add_special_tokens=False)
    assert tokenizer.encode("Explain RLHF", add_special_tokens=False) != prompt_ids + answer_ids
    row = _tokenize_structured_example(tokenizer, "Explain", "HF", 128)
    assert row["input_ids"] == prompt_ids + answer_ids
    assert row["labels"] == [-100] * len(prompt_ids) + answer_ids


@pytest.mark.parametrize("inference_suffix", ["", "Assistant: I'll answer:"])
def test_template_masks_training_header_even_if_generation_prompt_differs(tokenizer, inference_suffix):
    tokenizer.chat_template = (
        "User: {{ messages[0].content }}\n"
        "{% if add_generation_prompt %}"
        + inference_suffix
        + "{% else %}Assistant: {{ messages[1].content }}{{ eos_token }}{% endif %}"
    )
    row = _tokenize_structured_example(tokenizer, "Why?", "Because.", 128)
    answer_ids = tokenizer.encode(" Because." + tokenizer.eos_token, add_special_tokens=False)
    assert row["labels"] == [-100] * (len(row["input_ids"]) - len(answer_ids)) + answer_ids


def test_padding_masks_only_added_tokens(tokenizer):
    short = _tokenize_structured_example(tokenizer, "Q", "A", 128)
    long = _tokenize_structured_example(tokenizer, "Longer question", "Longer answer", 128)
    batch = SupervisedDataCollator(tokenizer)([short, long])
    padding = len(long["input_ids"]) - len(short["input_ids"])
    assert padding > 0
    assert batch["labels"][0].tolist() == short["labels"] + [-100] * padding
    assert batch["attention_mask"][0].tolist() == short["attention_mask"] + [0] * padding


@pytest.mark.parametrize(("response", "window"), [("answer", 1), ("  ", 32)])
def test_rejects_rows_without_trainable_targets(tokenizer, response, window):
    with pytest.raises(ValueError, match="at least one context token|tokenizable response"):
        _tokenize_structured_example(tokenizer, "question", response, window)
