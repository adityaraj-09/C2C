"""Optional bridge from Semantic Capsules into RosettaModel / DynamicCache.

The protocol does not require HuggingFace models. When they are present,
these helpers:

* snapshot a model's ``past_key_values`` into a ``KVSlice``
* inject a capsule as a latent prefix (true cross-context C2C)
* build the ``kv_cache_index`` bitmask the existing wrapper expects,
  so positional (same-prompt) fusion still uses trained C2C fusers
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from rosetta.agent.capsule import KVSlice, SemanticCapsule
from rosetta.agent.overlay import OverlayMode, overlay

try:
    import torch
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - optional dependency
    torch = None
    DynamicCache = None


def kvslice_from_cache(cache: Any, token_start: int = 0) -> KVSlice:
    """Convert a HuggingFace DynamicCache (or cache-like) to KVSlice."""
    keys = [k.detach().clone() for k in cache.key_cache]
    values = [v.detach().clone() for v in cache.value_cache]
    seq = 0 if not keys else int(keys[0].shape[2])
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(range(len(keys))),
        token_start=token_start,
        token_end=token_start + seq,
    )


def cache_from_kvslice(kv: KVSlice) -> Any:
    """Build a DynamicCache from a KVSlice (layer order = slice order)."""
    if DynamicCache is None:
        raise ImportError("transformers is required for cache_from_kvslice")
    cache = DynamicCache()
    # Ensure layers are stored at their original indices, padding holes
    # with empty tensors if the slice skipped layers.
    max_layer = max(kv.layer_indices) if kv.layer_indices else -1
    index = kv.layer_map()
    b, h, n, d = kv.shape
    device = kv.keys[0].device if kv.keys else "cpu"
    dtype = kv.keys[0].dtype if kv.keys else torch.float32
    for layer in range(max_layer + 1):
        if layer in index:
            k = kv.keys[index[layer]]
            v = kv.values[index[layer]]
        else:
            k = torch.zeros(b, h, n, d, device=device, dtype=dtype)
            v = torch.zeros_like(k)
        cache.update(k, v, layer)
    return cache


def snapshot_rosetta_sharer(rosetta_model: Any, source_model_idx: int = 1) -> KVSlice:
    """Read the last computed sharer cache out of ``RosettaModel.kv_cache_dict``."""
    base = rosetta_model.base_model_idx
    cache = rosetta_model.kv_cache_dict[base][source_model_idx]
    return kvslice_from_cache(cache)


def apply_prefix_to_past(past_key_values: Any, capsule: SemanticCapsule) -> Any:
    """Prepend a capsule onto an existing past (latent prefix injection)."""
    recv = kvslice_from_cache(past_key_values)
    fused = overlay(recv, capsule, mode=OverlayMode.PREFIX)
    return cache_from_kvslice(fused)


def kv_cache_index_for_prompt(
    seq_len: int,
    sharer_mask: int,
    device=None,
):
    """Build the two-section index the current ``RosettaModel.generate`` uses.

    Section 0 (prompt minus last token): apply C2C with ``sharer_mask``.
    Section 1 (last prompt token / generation): ``-1`` = no projection.

    Agent protocols that want *positional* fusion on a shared prompt can
    keep using this. Prefix/handoff injection goes through
    ``apply_prefix_to_past`` instead, because the capsule tokens do not
    exist in the receiver tokenizer.
    """
    if torch is None:
        raise ImportError("torch is required")
    device = device or "cpu"
    instruction = torch.tensor([sharer_mask, 0], dtype=torch.long, device=device)
    instruction = instruction.repeat(max(seq_len - 1, 1), 1).unsqueeze(0)
    label = torch.tensor([[-1, 0]], dtype=torch.long, device=device).unsqueeze(0)
    return [instruction, label]


def layer_subset_config(
    n_src_layers: int,
    n_dst_layers: int,
    keep: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """Pairs ``(src_layer, dst_layer)`` for selective C2C projector wiring.

    ``keep`` is in destination layer indices. Default: middle third, which
    is where semantic (not lexical, not decision) features usually live.
    """
    if keep is None:
        lo = n_dst_layers // 3
        hi = max(lo + 1, (2 * n_dst_layers) // 3)
        keep = list(range(lo, hi))
    from rosetta.agent.overlay import align_layers

    mapping = align_layers(n_src_layers, n_dst_layers)
    return [(mapping[d], d) for d in keep]
