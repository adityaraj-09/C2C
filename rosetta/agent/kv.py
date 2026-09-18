"""KV-cache operations for same-model agent C2C.

Same-model handoff is copy-on-write of a HuggingFace ``DynamicCache``.
Keys are stored **after RoPE**, so a capsule must be a **prefix of one
timeline** (positions 0..N). Punching a hole in the middle, or concatenating
two diverged branches, would silently mis-align rotary positions.

Gold rule, tested in this repo: greedy tokens from
``prefill(A); continue(B | past=A)`` match ``prefill(A+B)`` exactly.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from rosetta.agent.errors import CapsuleError, TimelineError


SCHEMA_VERSION = 1


def empty_cache() -> DynamicCache:
    return DynamicCache()


def clone_cache(cache: Optional[DynamicCache]) -> Optional[DynamicCache]:
    if cache is None:
        return None
    new = DynamicCache()
    new.key_cache = [k.detach().clone() for k in cache.key_cache]
    new.value_cache = [v.detach().clone() for v in cache.value_cache]
    new._seen_tokens = int(getattr(cache, "_seen_tokens", new.get_seq_length()))
    return new


def cache_seq_len(cache: Optional[DynamicCache]) -> int:
    if cache is None:
        return 0
    return int(cache.get_seq_length())


def crop_prefix(cache: DynamicCache, keep: int) -> DynamicCache:
    """Keep positions ``[0, keep)``. ``keep`` must be <= current length."""
    n = cache_seq_len(cache)
    if keep < 0 or keep > n:
        raise TimelineError(f"crop length {keep} outside [0, {n}]")
    out = clone_cache(cache)
    assert out is not None
    if keep < n:
        out.crop(keep)
    return out


def cache_nbytes(cache: Optional[DynamicCache]) -> int:
    if cache is None:
        return 0
    total = 0
    for k, v in zip(cache.key_cache, cache.value_cache):
        total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
    return total


def model_fingerprint(config: Any) -> Dict[str, Any]:
    n_heads = int(config.num_attention_heads)
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden // n_heads))
    return {
        "model_type": str(getattr(config, "model_type", "unknown")),
        "n_layers": int(config.num_hidden_layers),
        "n_heads": n_heads,
        "n_kv_heads": int(getattr(config, "num_key_value_heads", n_heads)),
        "hidden_size": hidden,
        "head_dim": head_dim,
        "vocab_size": int(config.vocab_size),
    }


def _quantize_int8(x: Tensor) -> Dict[str, Tensor]:
    max_abs = x.detach().abs().amax().clamp(min=1e-8)
    scale = (max_abs / 127.0).to(torch.float32)
    q = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return {"q": q, "scale": scale}


def _dequantize_int8(payload: Dict[str, Tensor]) -> Tensor:
    return payload["q"].float() * payload["scale"]


def serialize_cache(
    cache: DynamicCache,
    *,
    quantize: str = "none",
    fingerprint: Optional[Dict[str, Any]] = None,
) -> bytes:
    if quantize not in ("none", "fp16", "int8"):
        raise CapsuleError(f"unknown quantize mode: {quantize}")
    meta = {
        "schema": SCHEMA_VERSION,
        "quantize": quantize,
        "n_layers": len(cache.key_cache),
        "seq_len": cache_seq_len(cache),
        "seen_tokens": int(getattr(cache, "_seen_tokens", cache_seq_len(cache))),
        "fingerprint": fingerprint or {},
    }
    tensors: Dict[str, Any] = {"meta": json.dumps(meta)}
    if quantize == "int8":
        tensors["k"] = [_quantize_int8(k) for k in cache.key_cache]
        tensors["v"] = [_quantize_int8(v) for v in cache.value_cache]
    else:
        dtype = torch.float16 if quantize == "fp16" else None
        tensors["k"] = [k.detach().cpu().to(dtype=dtype) if dtype else k.detach().cpu() for k in cache.key_cache]
        tensors["v"] = [v.detach().cpu().to(dtype=dtype) if dtype else v.detach().cpu() for v in cache.value_cache]
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return buf.getvalue()


def deserialize_cache(
    blob: bytes,
    *,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[DynamicCache, Dict[str, Any]]:
    try:
        data = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        meta = json.loads(data["meta"])
    except Exception as exc:
        raise CapsuleError(f"capsule decode failed: {exc}") from exc
    if int(meta.get("schema", -1)) != SCHEMA_VERSION:
        raise CapsuleError(f"unsupported capsule schema {meta.get('schema')}")
    quantize = meta.get("quantize", "none")
    cache = DynamicCache()
    keys: List[Tensor] = []
    values: List[Tensor] = []
    if quantize == "int8":
        keys = [_dequantize_int8(k) for k in data["k"]]
        values = [_dequantize_int8(v) for v in data["v"]]
    else:
        keys = list(data["k"])
        values = list(data["v"])
    if dtype is not None:
        keys = [k.to(dtype=dtype) for k in keys]
        values = [v.to(dtype=dtype) for v in values]
    if device is not None:
        keys = [k.to(device=device) for k in keys]
        values = [v.to(device=device) for v in values]
    cache.key_cache = keys
    cache.value_cache = values
    cache._seen_tokens = int(meta.get("seen_tokens", meta.get("seq_len", cache_seq_len(cache))))
    return cache, meta
