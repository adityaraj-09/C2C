"""Semantic Capsules: the unit of agent-to-agent C2C communication.

A capsule is not a generated message. It is a sliced, annotated region of a
KV-cache that another agent can attend to.

Paper C2C assumes both models process the same prompt and fuses KV at the
same token index. Capsules drop that assumption:

* Agent A may have read a file Agent B never saw. Those file tokens live in
  A's cache as a span. Extracting that span and injecting it as a *latent
  prefix* in B is a transfer of understanding, not an explanation.
* Agent A may expose only the layers / heads / tokens relevant to B.
* The same capsule can be quantized and sent over a network.

Tensor convention matches transformers DynamicCache and C2CProjector:
each key/value is ``(batch, n_heads, seq_len, head_dim)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
import io
import json
import time

import torch
from torch import Tensor


class Intent(str, Enum):
    """Why this capsule is being sent. Overlay algebra is intent-specific.

    HANDOFF
        Sender yields working memory. Receiver inherits the slice as its
        new past (prefix or replace). Used for agent succession.
    DELEGATE
        Sender keeps running. Receiver is initialized with a latent task
        spec — the planner's unverbalized intent — as a prefix.
    CONSULT
        Pull-style: a query slice goes out, an answer slice comes back.
        Implemented as remote top-k attention, not a text Q&A.
    FUSE
        Contribute to a shared workspace slot. Multiple writers combine
        with parallel residual fusion (same algebra as multi-sharer C2C).
    STREAM
        Partial / continuous transfer. New KV tokens append to an existing
        slot while the sender is still thinking.
    CRITIQUE
        Overlay a disagreement residual onto a generator's cache instead
        of writing a review.
    SYNC
        Bidirectional state alignment: each side projects into the other.
    """

    HANDOFF = "handoff"
    DELEGATE = "delegate"
    CONSULT = "consult"
    FUSE = "fuse"
    STREAM = "stream"
    CRITIQUE = "critique"
    SYNC = "sync"


@dataclass
class KVSlice:
    """Layer-wise KV tensors, optionally a subset of layers/tokens/heads.

    ``keys[i]`` / ``values[i]`` correspond to ``layer_indices[i]``. Layers
    that were not extracted are simply absent — this is the isolation
    mechanism: an agent can refuse to export early-layer lexical state or
    late-layer decision state.
    """

    keys: List[Tensor]
    values: List[Tensor]
    layer_indices: List[int]
    token_start: int = 0
    token_end: int = 0
    head_indices: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("keys and values must have the same number of layers")
        if len(self.keys) != len(self.layer_indices):
            raise ValueError("layer_indices length must match keys")
        if self.keys:
            n = self.keys[0].shape[2]
            if self.token_end == 0:
                self.token_end = self.token_start + n
            for k, v in zip(self.keys, self.values):
                if k.shape != v.shape:
                    raise ValueError(f"key/value shape mismatch: {k.shape} vs {v.shape}")
                if k.dim() != 4:
                    raise ValueError(f"expected (B, H, N, D), got {tuple(k.shape)}")

    @property
    def n_layers(self) -> int:
        return len(self.keys)

    @property
    def seq_len(self) -> int:
        return 0 if not self.keys else int(self.keys[0].shape[2])

    @property
    def n_heads(self) -> int:
        return 0 if not self.keys else int(self.keys[0].shape[1])

    @property
    def head_dim(self) -> int:
        return 0 if not self.keys else int(self.keys[0].shape[-1])

    @property
    def batch(self) -> int:
        return 0 if not self.keys else int(self.keys[0].shape[0])

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.batch, self.n_heads, self.seq_len, self.head_dim)

    def to(self, device=None, dtype=None) -> "KVSlice":
        keys = [k.to(device=device, dtype=dtype) for k in self.keys]
        values = [v.to(device=device, dtype=dtype) for v in self.values]
        return replace(self, keys=keys, values=values)

    def clone(self) -> "KVSlice":
        return replace(
            self,
            keys=[k.detach().clone() for k in self.keys],
            values=[v.detach().clone() for v in self.values],
            layer_indices=list(self.layer_indices),
            head_indices=None if self.head_indices is None else list(self.head_indices),
        )

    def layer_map(self) -> Dict[int, int]:
        """Map original layer index → position in this slice."""
        return {layer: i for i, layer in enumerate(self.layer_indices)}

    def pooled_value(self, layer: int = -1) -> Tensor:
        """Mean-pool a layer's values over heads and tokens: (B, D)."""
        v = self.values[layer]
        return v.mean(dim=(1, 2))

    def nbytes(self) -> int:
        return sum(k.nelement() * k.element_size() + v.nelement() * v.element_size()
                   for k, v in zip(self.keys, self.values))


def empty_kv(
    n_layers: int,
    batch: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    *,
    device=None,
    dtype=torch.float32,
) -> KVSlice:
    keys = [
        torch.zeros(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        for _ in range(n_layers)
    ]
    values = [torch.zeros_like(k) for k in keys]
    return KVSlice(keys=keys, values=values, layer_indices=list(range(n_layers)),
                   token_start=0, token_end=seq_len)


def cat_seq(left: KVSlice, right: KVSlice) -> KVSlice:
    """Concatenate two slices along the sequence axis (same layers/heads)."""
    if left.n_layers != right.n_layers or left.n_heads != right.n_heads:
        raise ValueError("cat_seq requires matching layer and head counts")
    if left.head_dim != right.head_dim:
        raise ValueError("cat_seq requires matching head_dim")
    keys = [torch.cat([a, b], dim=2) for a, b in zip(left.keys, right.keys)]
    values = [torch.cat([a, b], dim=2) for a, b in zip(left.values, right.values)]
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(left.layer_indices),
        token_start=left.token_start,
        token_end=left.token_start + keys[0].shape[2],
        head_indices=left.head_indices,
    )


def extract_span(kv: KVSlice, start: int, end: int) -> KVSlice:
    """Keep tokens ``[start, end)`` in *slice-local* coordinates."""
    if start < 0 or end > kv.seq_len or start >= end:
        raise ValueError(f"invalid span [{start}, {end}) for seq_len={kv.seq_len}")
    keys = [k[:, :, start:end, :] for k in kv.keys]
    values = [v[:, :, start:end, :] for v in kv.values]
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(kv.layer_indices),
        token_start=kv.token_start + start,
        token_end=kv.token_start + end,
        head_indices=kv.head_indices,
    )


def extract_layers(kv: KVSlice, layers: Sequence[int]) -> KVSlice:
    """Keep a subset of original layer indices (capability isolation)."""
    index = kv.layer_map()
    missing = [L for L in layers if L not in index]
    if missing:
        raise ValueError(f"layers not in slice: {missing}")
    keys = [kv.keys[index[L]] for L in layers]
    values = [kv.values[index[L]] for L in layers]
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(layers),
        token_start=kv.token_start,
        token_end=kv.token_end,
        head_indices=kv.head_indices,
    )


def extract_heads(kv: KVSlice, heads: Sequence[int]) -> KVSlice:
    keys = [k[:, heads, :, :] for k in kv.keys]
    values = [v[:, heads, :, :] for v in kv.values]
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(kv.layer_indices),
        token_start=kv.token_start,
        token_end=kv.token_end,
        head_indices=list(heads),
    )


def pool_tokens(kv: KVSlice, factor: int) -> KVSlice:
    """Average-pool the sequence axis by ``factor`` (lossy compression).

    Remainder tokens are kept as a shorter final group so no tokens are dropped
    silently — they just live at coarser resolution.
    """
    if factor < 1:
        raise ValueError("pool factor must be >= 1")
    if factor == 1:
        return kv.clone()

    def _pool(x: Tensor) -> Tensor:
        b, h, n, d = x.shape
        groups = n // factor
        rem = n % factor
        parts = []
        if groups:
            head = x[:, :, : groups * factor, :].reshape(b, h, groups, factor, d)
            parts.append(head.mean(dim=3))
        if rem:
            parts.append(x[:, :, groups * factor :, :].mean(dim=2, keepdim=True))
        return torch.cat(parts, dim=2) if parts else x[:, :, :0, :]

    keys = [_pool(k) for k in kv.keys]
    values = [_pool(v) for v in kv.values]
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(kv.layer_indices),
        token_start=kv.token_start,
        token_end=kv.token_end,
        head_indices=kv.head_indices,
    )


def quantize_int8(kv: KVSlice) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Per-tensor symmetric int8 quantization of keys and values.

    Returns ``(q_keys, k_scales, q_values, v_scales)``. Scales are scalar
    tensors so dequant is a multiply.
    """

    def _q(x: Tensor) -> Tuple[Tensor, Tensor]:
        max_abs = x.detach().abs().amax().clamp(min=1e-8)
        scale = max_abs / 127.0
        q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return q, scale.to(torch.float32)

    qk, sk, qv, sv = [], [], [], []
    for k, v in zip(kv.keys, kv.values):
        a, b = _q(k)
        c, d = _q(v)
        qk.append(a)
        sk.append(b)
        qv.append(c)
        sv.append(d)
    return qk, sk, qv, sv


def dequantize_int8(
    q_keys: Sequence[Tensor],
    k_scales: Sequence[Tensor],
    q_values: Sequence[Tensor],
    v_scales: Sequence[Tensor],
) -> Tuple[List[Tensor], List[Tensor]]:
    keys = [q.float() * s for q, s in zip(q_keys, k_scales)]
    values = [q.float() * s for q, s in zip(q_values, v_scales)]
    return keys, values


@dataclass
class CapsuleMeta:
    source: str
    slot: str
    role: str = ""
    provenance: str = ""
    created_at: float = field(default_factory=time.time)
    lineage: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCapsule:
    """Addressable KV payload with a communication intent.

    ``slot`` is the latent address (e.g. ``codebase.api``, ``plan.constraints``).
    ``intent`` selects how a receiver should consume the payload.
    """

    source: str
    slot: str
    intent: Intent
    kv: KVSlice
    role: str = ""
    provenance: str = ""
    lineage: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> CapsuleMeta:
        return CapsuleMeta(
            source=self.source,
            slot=self.slot,
            role=self.role,
            provenance=self.provenance,
            created_at=self.created_at,
            lineage=list(self.lineage),
            extra=dict(self.extra),
        )

    def with_intent(self, intent: Intent) -> "SemanticCapsule":
        return replace(self, intent=intent)

    def sliced(
        self,
        *,
        span: Optional[Tuple[int, int]] = None,
        layers: Optional[Sequence[int]] = None,
        heads: Optional[Sequence[int]] = None,
        pool: int = 1,
    ) -> "SemanticCapsule":
        kv = self.kv
        if span is not None:
            kv = extract_span(kv, span[0], span[1])
        if layers is not None:
            kv = extract_layers(kv, layers)
        if heads is not None:
            kv = extract_heads(kv, heads)
        if pool > 1:
            kv = pool_tokens(kv, pool)
        return replace(self, kv=kv)

    def nbytes(self) -> int:
        return self.kv.nbytes()

    def to_bytes(self, quantize: str = "none") -> bytes:
        """Serialize for C2C-over-the-network between physically separate agents."""
        payload: Dict[str, Any] = {
            "source": self.source,
            "slot": self.slot,
            "intent": self.intent.value,
            "role": self.role,
            "provenance": self.provenance,
            "lineage": list(self.lineage),
            "created_at": self.created_at,
            "extra": self.extra,
            "layer_indices": list(self.kv.layer_indices),
            "token_start": self.kv.token_start,
            "token_end": self.kv.token_end,
            "head_indices": self.kv.head_indices,
            "quantize": quantize,
        }
        buffer = io.BytesIO()
        tensors: Dict[str, Any] = {"meta": json.dumps(payload)}
        if quantize == "int8":
            qk, sk, qv, sv = quantize_int8(self.kv)
            tensors.update({
                "q_keys": qk,
                "k_scales": sk,
                "q_values": qv,
                "v_scales": sv,
            })
        elif quantize in ("none", "fp16"):
            dtype = torch.float16 if quantize == "fp16" else None
            kv = self.kv.to(dtype=dtype) if dtype is not None else self.kv
            tensors["keys"] = kv.keys
            tensors["values"] = kv.values
        else:
            raise ValueError(f"unknown quantize mode: {quantize}")
        torch.save(tensors, buffer)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> "SemanticCapsule":
        data = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        payload = json.loads(data["meta"])
        quantize = payload.get("quantize", "none")
        if quantize == "int8":
            keys, values = dequantize_int8(
                data["q_keys"], data["k_scales"], data["q_values"], data["v_scales"]
            )
        else:
            keys = list(data["keys"])
            values = list(data["values"])
            if quantize == "fp16":
                keys = [k.float() for k in keys]
                values = [v.float() for v in values]
        kv = KVSlice(
            keys=keys,
            values=values,
            layer_indices=list(payload["layer_indices"]),
            token_start=int(payload["token_start"]),
            token_end=int(payload["token_end"]),
            head_indices=payload.get("head_indices"),
        )
        return cls(
            source=payload["source"],
            slot=payload["slot"],
            intent=Intent(payload["intent"]),
            kv=kv,
            role=payload.get("role", ""),
            provenance=payload.get("provenance", ""),
            lineage=list(payload.get("lineage", [])),
            created_at=float(payload.get("created_at", 0.0)),
            extra=dict(payload.get("extra", {})),
        )


def compose_lineage(capsules: Sequence[SemanticCapsule], slot: str, source: str) -> SemanticCapsule:
    """Concatenate capsules along the sequence axis, recording lineage.

    Hierarchical handoff: planner → coder → tester. The tester receives one
    capsule whose sequence is ``[planner span | coder span]`` and whose
    lineage lists every ancestor slot.
    """
    if not capsules:
        raise ValueError("compose_lineage requires at least one capsule")
    kv = capsules[0].kv.clone()
    lineage: List[str] = []
    for cap in capsules:
        lineage.extend(cap.lineage)
        lineage.append(f"{cap.source}:{cap.slot}")
    for cap in capsules[1:]:
        kv = cat_seq(kv, cap.kv)
    return SemanticCapsule(
        source=source,
        slot=slot,
        intent=Intent.FUSE,
        kv=kv,
        provenance="compose_lineage",
        lineage=lineage,
    )
