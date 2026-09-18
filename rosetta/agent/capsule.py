"""On-the-wire packet for same-model agent C2C.

There is no natural-language ``message`` field. The payload is a KV cache
plus timeline metadata (token ids that *produced* that cache, used for
repetition penalty and length — never re-encoded by the receiver).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from rosetta.agent.errors import CapsuleError, ModelMismatchError, TimelineError
from rosetta.agent.kv import (
    SCHEMA_VERSION,
    cache_seq_len,
    deserialize_cache,
    serialize_cache,
)


class Intent(str, Enum):
    """How the receiver should install the cache.

    FORK
        Child starts from a copy. Sender keeps its own timeline.
    ADOPT
        Receiver *becomes* the sender timeline (join / handoff).
    """

    FORK = "fork"
    ADOPT = "adopt"


@dataclass
class Capsule:
    """Serializable KV state of one agent at one moment."""

    source: str
    intent: Intent
    cache: DynamicCache
    token_ids: Tensor
    last_logits: Optional[Tensor]
    fingerprint: Dict[str, Any]
    role: str = ""
    slot: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = cache_seq_len(self.cache)
        if self.token_ids.dim() != 2 or self.token_ids.shape[0] != 1:
            raise CapsuleError("token_ids must have shape (1, seq)")
        if int(self.token_ids.shape[1]) != n:
            raise TimelineError(
                f"token_ids length {self.token_ids.shape[1]} != cache seq {n}"
            )
        if n == 0:
            raise CapsuleError("refusing to send an empty cache")

    @property
    def seq_len(self) -> int:
        return cache_seq_len(self.cache)

    def to_bytes(self, quantize: str = "none") -> bytes:
        import io
        import json

        payload = {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "intent": self.intent.value,
            "role": self.role,
            "slot": self.slot,
            "fingerprint": self.fingerprint,
            "extra": self.extra,
            "quantize": quantize,
        }
        blob = {
            "header": json.dumps(payload),
            "cache": serialize_cache(self.cache, quantize=quantize, fingerprint=self.fingerprint),
            "token_ids": self.token_ids.detach().cpu().to(torch.long),
            "last_logits": None if self.last_logits is None else self.last_logits.detach().cpu().float(),
        }
        buf = io.BytesIO()
        torch.save(blob, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(
        cls,
        blob: bytes,
        *,
        device=None,
        dtype: Optional[torch.dtype] = None,
        expect_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> "Capsule":
        import io
        import json

        try:
            data = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            header = json.loads(data["header"])
        except Exception as exc:
            raise CapsuleError(f"capsule wrapper decode failed: {exc}") from exc
        cache, _meta = deserialize_cache(data["cache"], device=device, dtype=dtype)
        token_ids = data["token_ids"].to(device=device or "cpu")
        last_logits = data["last_logits"]
        if last_logits is not None:
            last_logits = last_logits.to(device=device or "cpu", dtype=dtype or last_logits.dtype)
        cap = cls(
            source=header["source"],
            intent=Intent(header["intent"]),
            cache=cache,
            token_ids=token_ids,
            last_logits=last_logits,
            fingerprint=dict(header.get("fingerprint") or {}),
            role=header.get("role", ""),
            slot=header.get("slot", ""),
            extra=dict(header.get("extra") or {}),
        )
        if expect_fingerprint is not None and cap.fingerprint != expect_fingerprint:
            raise ModelMismatchError(
                f"capsule fingerprint {cap.fingerprint} != engine {expect_fingerprint}"
            )
        return cap
