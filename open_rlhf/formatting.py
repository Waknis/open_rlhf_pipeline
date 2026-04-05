"""Prompt formatting helpers shared across training, eval, and demo flows."""

from __future__ import annotations

from typing import Any


def build_plain_prompt(prompt: str) -> str:
    """Render a deterministic plain-text prompt for base models."""
    return f"User: {prompt.strip()}\nAssistant:"


def build_plain_transcript(prompt: str, response: str) -> str:
    """Render a deterministic plain-text prompt/response transcript."""
    return f"{build_plain_prompt(prompt)} {response.strip()}"


def build_messages(prompt: str, response: str | None = None) -> list[dict[str, str]]:
    """Build a chat-style message list."""
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt.strip()}]
    if response is not None:
        messages.append({"role": "assistant", "content": response.strip()})
    return messages


def tokenizer_uses_chat_template(tokenizer: Any) -> bool:
    """Return True when the tokenizer exposes a usable chat template."""
    return bool(getattr(tokenizer, "chat_template", None))


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render a prompt using the tokenizer chat template when available."""
    if tokenizer_uses_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            build_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
    return build_plain_prompt(prompt)


def render_prompt_response(tokenizer: Any, prompt: str, response: str) -> str:
    """Render a full prompt/response transcript using the tokenizer template when available."""
    if tokenizer_uses_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            build_messages(prompt, response),
            tokenize=False,
            add_generation_prompt=False,
        )
    return build_plain_transcript(prompt, response)


def render_prepared_text(prompt: str, response: str) -> str:
    """Render the plain-text training transcript stored in processed instruction data."""
    return build_plain_transcript(prompt, response)

