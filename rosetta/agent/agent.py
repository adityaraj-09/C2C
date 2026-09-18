"""Per-agent working memory on a shared model.

An agent never sends text to another agent. It only exports a ``Capsule``.
Environment (files, tests) and the user still enter as text, via the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from rosetta.agent.capsule import Capsule, Intent
from rosetta.agent.errors import AgentStateError
from rosetta.agent.kv import cache_seq_len, clone_cache


@dataclass
class Agent:
    name: str
    role: str
    token_ids: Tensor
    cache: Optional[DynamicCache] = None
    last_logits: Optional[Tensor] = None
    extra: dict = field(default_factory=dict)
    messages: List[dict] = field(default_factory=list)
    cache_cow: bool = False

    @property
    def seq_len(self) -> int:
        return cache_seq_len(self.cache)

    def ensure_exclusive(self) -> None:
        """Break copy-on-write before mutating KV."""
        if self.cache_cow:
            self.cache = clone_cache(self.cache)
            self.token_ids = self.token_ids.detach().clone()
            if self.last_logits is not None:
                self.last_logits = self.last_logits.detach().clone()
            self.cache_cow = False

    def clone(self) -> "Agent":
        return Agent(
            name=self.name,
            role=self.role,
            token_ids=self.token_ids.detach().clone(),
            cache=clone_cache(self.cache),
            last_logits=None if self.last_logits is None else self.last_logits.detach().clone(),
            extra=dict(self.extra),
            messages=[dict(m) for m in self.messages],
            cache_cow=False,
        )

    def share(self, name: str, role: str) -> "Agent":
        """O(1) fork: share KV until either side appends."""
        self.cache_cow = True
        return Agent(
            name=name,
            role=role,
            token_ids=self.token_ids,
            cache=self.cache,
            last_logits=self.last_logits,
            extra={"forked_from": self.name},
            messages=[dict(m) for m in self.messages],
            cache_cow=True,
        )

    def export(self, intent: Intent, slot: str, fingerprint: dict) -> Capsule:
        if self.cache is None or self.last_logits is None or self.seq_len == 0:
            raise AgentStateError(f"agent {self.name!r} has no KV to export")
        return Capsule(
            source=self.name,
            intent=intent,
            cache=clone_cache(self.cache),
            token_ids=self.token_ids.detach().clone(),
            last_logits=self.last_logits.detach().clone(),
            fingerprint=dict(fingerprint),
            role=self.role,
            slot=slot,
            extra={"messages": [dict(m) for m in self.messages]},
        )

    def install(self, capsule: Capsule) -> None:
        """Replace this agent's timeline with the capsule (ADOPT)."""
        self.cache = clone_cache(capsule.cache)
        self.token_ids = capsule.token_ids.detach().clone()
        self.last_logits = None if capsule.last_logits is None else capsule.last_logits.detach().clone()
        self.extra["adopted_from"] = capsule.source
        self.messages = [dict(m) for m in capsule.extra.get("messages", [])]
        self.cache_cow = False
