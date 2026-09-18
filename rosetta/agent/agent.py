"""Per-agent working memory on a shared model.

An agent never sends text to another agent. It only exports a ``Capsule``.
Environment (files, tests) and the user still enter as text, via the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

    @property
    def seq_len(self) -> int:
        return cache_seq_len(self.cache)

    def clone(self) -> "Agent":
        return Agent(
            name=self.name,
            role=self.role,
            token_ids=self.token_ids.detach().clone(),
            cache=clone_cache(self.cache),
            last_logits=None if self.last_logits is None else self.last_logits.detach().clone(),
            extra=dict(self.extra),
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
        )

    def install(self, capsule: Capsule) -> None:
        """Replace this agent's timeline with the capsule (ADOPT)."""
        self.cache = clone_cache(capsule.cache)
        self.token_ids = capsule.token_ids.detach().clone()
        self.last_logits = None if capsule.last_logits is None else capsule.last_logits.detach().clone()
        self.extra["adopted_from"] = capsule.source
