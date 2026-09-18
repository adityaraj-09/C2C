"""Load a real instruct checkpoint for gold / ticket A/B tests.

The default path is the small SmolLM2-Instruct snapshot used in this
environment. Override with ``C2C_INSTRUCT_MODEL``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from rosetta.agent.engine import SharedCausalEngine

DEFAULT_INSTRUCT_MODEL = os.environ.get(
    "C2C_INSTRUCT_MODEL", "/tmp/models/SmolLM2-135M-Instruct"
)


def instruct_model_path(model_dir: Optional[str] = None) -> Path:
    return Path(model_dir or DEFAULT_INSTRUCT_MODEL)


def instruct_model_available(model_dir: Optional[str] = None) -> bool:
    path = instruct_model_path(model_dir)
    return path.is_dir() and (path / "config.json").is_file()


@lru_cache(maxsize=1)
def load_instruct_engine(
    model_dir: Optional[str] = None,
    *,
    block_size: int = 16,
) -> tuple[SharedCausalEngine, object]:
    """Load tokenizer + causal LM. Cached: one set of weights per process."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(instruct_model_path(model_dir))
    if not instruct_model_available(path):
        raise FileNotFoundError(f"instruct model not found at {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    engine = SharedCausalEngine(
        model,
        pad_token_id=int(pad),
        eos_token_id=tokenizer.eos_token_id,
        block_size=block_size,
    )
    return engine, tokenizer
