"""Shared causal LM engine. One set of weights, many agent KV states.

``generate`` is a local decode loop rather than ``model.generate(past=...)``.
HuggingFace ``generate`` with an inherited cache is version-fragile (we hit
an IndexError on 4.52); the forward-with-past path is stable and is the
path that bit-matches a full prefill.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache

from rosetta.agent.errors import AgentStateError
from rosetta.agent.kv import cache_seq_len, clone_cache, crop_prefix, model_fingerprint
from rosetta.agent.prefix_cache import PrefixBlockCache


@dataclass
class ForwardResult:
    logits: Tensor
    past: DynamicCache
    next_token: Optional[Tensor] = None


class SharedCausalEngine:
    """Thread-safe wrapper around a single ``nn.Module`` causal LM."""

    def __init__(
        self,
        model: nn.Module,
        *,
        pad_token_id: int = 0,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        device: Optional[torch.device] = None,
        block_size: int = 16,
    ):
        self.model = model.eval()
        if device is not None:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.pad_token_id = int(pad_token_id)
        if eos_token_id is None:
            cfg = getattr(model, "config", None)
            eos_token_id = getattr(cfg, "eos_token_id", None) if cfg is not None else None
        if eos_token_id is None:
            self.eos_token_ids: List[int] = []
        elif isinstance(eos_token_id, (list, tuple)):
            self.eos_token_ids = [int(x) for x in eos_token_id]
        else:
            self.eos_token_ids = [int(eos_token_id)]
        self._lock = threading.Lock()
        self.fingerprint = model_fingerprint(model.config)
        self.prefix_cache = PrefixBlockCache(block_size=block_size)

    @property
    def config(self):
        return self.model.config

    def _full_mask(self, past_len: int, new_len: int, batch: int = 1) -> Tensor:
        return torch.ones(batch, past_len + new_len, dtype=torch.long, device=self.device)

    def prefill(
        self,
        input_ids: Tensor,
        past: Optional[DynamicCache] = None,
        *,
        clone_past: bool = True,
    ) -> ForwardResult:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise AgentStateError("engine only serves batch_size=1")
        if input_ids.shape[1] == 0:
            raise AgentStateError("cannot prefill an empty token sequence")
        input_ids = input_ids.to(self.device)
        past_len = cache_seq_len(past)
        mask = self._full_mask(past_len, input_ids.shape[1], input_ids.shape[0])
        work = clone_cache(past) if clone_past else past
        with self._lock, torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                past_key_values=work,
                use_cache=True,
            )
        self.prefix_cache.stats.tokens_computed += int(input_ids.shape[1])
        return ForwardResult(logits=out.logits[:, -1, :].detach(), past=out.past_key_values)

    def prefill_from_empty(self, token_ids: Tensor) -> ForwardResult:
        """Prefill a full sequence, reusing hashed KV blocks when possible."""
        if token_ids.dim() != 2 or token_ids.shape[0] != 1:
            raise AgentStateError("engine only serves batch_size=1")
        ids = token_ids[0].tolist()
        matched, cached = self.prefix_cache.lookup(ids)
        if matched == len(ids) and cached is not None:
            # Exact block cover: re-forward the last token for logits.
            if matched == 1:
                return self.prefill(token_ids, past=None, clone_past=False)
            cropped = crop_prefix(cached, matched - 1)
            return self.prefill(token_ids[:, -1:], past=cropped, clone_past=False)
        suffix = token_ids[:, matched:]
        step = self.prefill(suffix, past=cached, clone_past=False)
        self.prefix_cache.remember(ids, step.past)
        return step

    def decode_one(self, token: Tensor, past: DynamicCache) -> ForwardResult:
        if token.dim() == 1:
            token = token.view(1, 1)
        return self.prefill(token, past=past, clone_past=False)

    def greedy_continue(
        self,
        past: DynamicCache,
        last_logits: Tensor,
        *,
        max_new_tokens: int,
        extra_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, DynamicCache, Tensor]:
        """Sample greedy tokens from an existing timeline.

        If ``extra_ids`` is set they are prefilled first (local scaffold such
        as an assistant header). Then up to ``max_new_tokens`` are decoded.
        """
        if extra_ids is not None and extra_ids.numel() > 0:
            step = self.prefill(extra_ids, past=past, clone_past=False)
            past = step.past
            last_logits = step.logits
        generated: List[Tensor] = []
        logits = last_logits
        for _ in range(max_new_tokens):
            tok = logits.argmax(dim=-1).view(1, 1)
            if self.eos_token_ids and int(tok.item()) in self.eos_token_ids:
                generated.append(tok)
                # still consume so KV/logits stay aligned with the token
                step = self.decode_one(tok, past)
                past, logits = step.past, step.logits
                break
            generated.append(tok)
            step = self.decode_one(tok, past)
            past, logits = step.past, step.logits
        if not generated:
            empty = torch.zeros(1, 0, dtype=torch.long, device=self.device)
            return empty, past, logits
        return torch.cat(generated, dim=1), past, logits

    def mean_nll(self, past: DynamicCache, last_logits: Tensor, target_ids: Tensor) -> float:
        """Average negative log-likelihood of ``target_ids`` under this timeline."""
        if target_ids.dim() == 1:
            target_ids = target_ids.view(1, -1)
        nll = 0.0
        n = int(target_ids.shape[1])
        if n == 0:
            return 0.0
        logits = last_logits
        cur = clone_cache(past)
        for i in range(n):
            tok = target_ids[:, i : i + 1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            nll += float(-logp[0, int(tok.item())])
            step = self.prefill(tok, past=cur, clone_past=False)
            cur, logits = step.past, step.logits
        return nll / n
