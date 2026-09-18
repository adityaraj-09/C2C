"""vLLM-style automatic prefix caching, implemented on HuggingFace caches.

This environment is CPU-only and has no vLLM/GPU. The *semantics* are the
same as vLLM APC: hash contiguous token blocks, reuse KV for a matching
prefix, only forward the suffix.

Copy-on-write is handled on ``Agent.cache_cow`` in the runtime: fork
shares the ``DynamicCache`` object until someone appends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from rosetta.agent.kv import cache_seq_len


BlockKey = Tuple[Optional[int], Tuple[int, ...]]


@dataclass
class PrefixStats:
    hits: int = 0
    misses: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "tokens_reused": self.tokens_reused,
            "tokens_computed": self.tokens_computed,
        }


@dataclass
class _Block:
    keys: List[Tensor]
    values: List[Tensor]


class PrefixBlockCache:
    """Immutable KV blocks keyed by (parent_hash, token_ids)."""

    def __init__(self, block_size: int = 16):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = int(block_size)
        self._store: Dict[BlockKey, _Block] = {}
        self.stats = PrefixStats()

    def reset_stats(self) -> None:
        self.stats = PrefixStats()

    def lookup(self, token_ids: Sequence[int]) -> tuple[int, Optional[DynamicCache]]:
        """Return ``(matched_tokens, cache)`` for the longest block prefix."""
        ids = [int(x) for x in token_ids]
        parent: Optional[int] = None
        blocks: List[_Block] = []
        matched = 0
        while matched + self.block_size <= len(ids):
            chunk = tuple(ids[matched : matched + self.block_size])
            key: BlockKey = (parent, chunk)
            block = self._store.get(key)
            if block is None:
                self.stats.misses += 1
                break
            self.stats.hits += 1
            blocks.append(block)
            parent = hash(key)
            matched += self.block_size
        if not blocks:
            return 0, None
        self.stats.tokens_reused += matched
        return matched, _assemble(blocks)

    def remember(self, token_ids: Sequence[int], cache: DynamicCache) -> int:
        """Store completed blocks from a cache that encodes ``token_ids``."""
        ids = [int(x) for x in token_ids]
        n = min(len(ids), cache_seq_len(cache))
        parent: Optional[int] = None
        stored = 0
        pos = 0
        while pos + self.block_size <= n:
            chunk = tuple(ids[pos : pos + self.block_size])
            key: BlockKey = (parent, chunk)
            if key not in self._store:
                self._store[key] = _slice_block(cache, pos, pos + self.block_size)
            parent = hash(key)
            pos += self.block_size
            stored += 1
        return stored


def _slice_block(cache: DynamicCache, start: int, end: int) -> _Block:
    keys = [k[:, :, start:end, :].detach().clone() for k in cache.key_cache]
    values = [v[:, :, start:end, :].detach().clone() for v in cache.value_cache]
    return _Block(keys=keys, values=values)


def _assemble(blocks: List[_Block]) -> DynamicCache:
    cache = DynamicCache()
    n_layers = len(blocks[0].keys)
    for layer in range(n_layers):
        cache.key_cache.append(torch.cat([b.keys[layer] for b in blocks], dim=2))
        cache.value_cache.append(torch.cat([b.values[layer] for b in blocks], dim=2))
    cache._seen_tokens = cache.key_cache[0].shape[2]
    return cache
