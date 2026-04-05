"""Runtime helpers for validated dependency checks and device selection."""

from __future__ import annotations

import importlib.metadata
import random
from dataclasses import dataclass

import numpy as np
import torch
from packaging.version import Version
from transformers import set_seed as transformers_set_seed

VALIDATED_STACK = {
    "transformers": "4.46.3",
    "trl": "0.12.2",
    "accelerate": "1.10.0",
    "datasets": "4.0.0",
}

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime device and precision settings."""

    device: str
    torch_dtype: torch.dtype
    bf16: bool
    fp16: bool
    use_cpu: bool
    use_mps_device: bool


def add_runtime_args(parser) -> None:
    """Add shared device/dtype/seed CLI flags to an argparse parser."""
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Execution device. Defaults to auto-detecting CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="auto",
        help="Model dtype. Defaults to bf16 on supported CUDA, otherwise float32.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset shuffling and training.",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


def _installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_validated_stack(*, require_trl: bool) -> None:
    """Fail fast when the validated trainer stack is not installed."""
    packages = ["transformers", "accelerate", "datasets"]
    if require_trl:
        packages.append("trl")

    problems: list[str] = []
    for package_name in packages:
        expected = VALIDATED_STACK[package_name]
        installed = _installed_version(package_name)
        if installed is None:
            problems.append(f"- {package_name}: missing (expected {expected})")
        elif installed != expected:
            problems.append(f"- {package_name}: found {installed}, expected {expected}")

    if problems:
        message = "\n".join(
            [
                "The validated Open RLHF stack is not installed.",
                *problems,
                "Install the pinned stack with `pip install -r requirements.txt`.",
            ]
        )
        raise SystemExit(message)


def _resolve_device(preferred_device: str) -> str:
    if preferred_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preferred_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this environment.")
    if preferred_device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available in this environment.")
    return preferred_device


def _resolve_dtype(device: str, preferred_dtype: str) -> torch.dtype:
    if preferred_dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = mapping[preferred_dtype]
    if device == "cpu" and torch_dtype == torch.float16:
        raise SystemExit("float16 is not supported for CPU execution; use float32 or bfloat16.")
    if device == "mps" and torch_dtype == torch.bfloat16:
        raise SystemExit("bfloat16 is not supported for MPS execution; use float32.")
    if device == "cuda" and torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 was requested but the current CUDA device does not support it.")
    return torch_dtype


def resolve_runtime(preferred_device: str, preferred_dtype: str) -> RuntimeConfig:
    """Resolve runtime precision and Trainer flags from CLI preferences."""
    device = _resolve_device(preferred_device)
    torch_dtype = _resolve_dtype(device, preferred_dtype)
    return RuntimeConfig(
        device=device,
        torch_dtype=torch_dtype,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        use_cpu=device == "cpu",
        use_mps_device=device == "mps",
    )


def make_torch_device(runtime: RuntimeConfig) -> torch.device:
    """Convert a RuntimeConfig into a torch.device."""
    return torch.device(runtime.device)


def ensure_tokenizer_has_padding(tokenizer, *, padding_side: str = "right") -> None:
    """Ensure tokenizer padding is configured for batching."""
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def load_tokenizer(model_name_or_path: str, *, padding_side: str = "right"):
    """Load a tokenizer with predictable padding defaults."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    ensure_tokenizer_has_padding(tokenizer, padding_side=padding_side)
    return tokenizer


def version_is_at_least(package_name: str, minimum_version: str) -> bool:
    """Check whether an installed package meets a version floor."""
    installed = _installed_version(package_name)
    return installed is not None and Version(installed) >= Version(minimum_version)
