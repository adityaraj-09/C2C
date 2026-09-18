"""How a receiver consumes a capsule.

Four overlay algebras, plus pull-style remote attention:

PREFIX
    Capsule tokens become phantom context *before* the receiver's own
    tokens. This is the cross-context primitive the paper does not have:
    B attends to A's understanding of documents B never tokenized.
POSITIONAL
    Paper-style: fuse at aligned token positions of a shared prompt.
RESIDUAL / CRITIQUE
    Add a projected delta. Critique uses (projected_critic - receiver)
    so the critic writes a disagreement field, not a review.
REPLACE
    Handoff: selected span is overwritten by the projected capsule.
STREAM_APPEND
    Grow the receiver cache with newly thought tokens.

``remote_retrieve`` is the pull counterpart: the receiver sends a query
vector and gets back top-k (K, V) from a remote capsule without copying
the whole cache. That is Cache-RPC.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rosetta.agent.capsule import KVSlice, SemanticCapsule, cat_seq


class OverlayMode(str, Enum):
    PREFIX = "prefix"
    POSITIONAL = "positional"
    RESIDUAL = "residual"
    REPLACE = "replace"
    STREAM_APPEND = "stream"
    CRITIQUE = "critique"
    FUSE_PARALLEL = "fuse_parallel"


def align_layers(n_src: int, n_dst: int) -> List[int]:
    """Map each destination layer to a source layer (heterogeneous depth)."""
    if n_src <= 0 or n_dst <= 0:
        return []
    if n_src == 1:
        return [0] * n_dst
    return [round(i * (n_src - 1) / (n_dst - 1)) for i in range(n_dst)]


class LinearKVAdapter(nn.Module):
    """Learned map between heterogeneous KV geometries.

    Flattens ``(H_s * D_s) → (H_t * D_t)`` independently for keys and values.
    This is the minimum projector that lets a Qwen-shaped cache talk to a
    Llama-shaped cache without the full C2C fuser. When a trained
    ``C2CProjector`` is available, pass it as ``projector`` to ``overlay``
    instead.
    """

    def __init__(
        self,
        source_heads: int,
        source_dim: int,
        target_heads: int,
        target_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.source_heads = source_heads
        self.source_dim = source_dim
        self.target_heads = target_heads
        self.target_dim = target_dim
        in_f = source_heads * source_dim
        out_f = target_heads * target_dim
        self.key_proj = nn.Linear(in_f, out_f, bias=bias)
        self.value_proj = nn.Linear(in_f, out_f, bias=bias)
        # Near-identity when geometries match, so untrained adapters still
        # preserve planted structure in unit tests / probing experiments.
        if in_f == out_f:
            nn.init.eye_(self.key_proj.weight)
            nn.init.eye_(self.value_proj.weight)
            if bias:
                nn.init.zeros_(self.key_proj.bias)
                nn.init.zeros_(self.value_proj.bias)

    def project_pair(self, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        b, h, n, d = key.shape
        k_flat = key.transpose(1, 2).contiguous().view(b, n, h * d)
        v_flat = value.transpose(1, 2).contiguous().view(b, n, h * d)
        k_out = self.key_proj(k_flat).view(b, n, self.target_heads, self.target_dim).transpose(1, 2)
        v_out = self.value_proj(v_flat).view(b, n, self.target_heads, self.target_dim).transpose(1, 2)
        return k_out, v_out

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],
        target_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        sk, sv = source_kv
        pk, pv = self.project_pair(sk, sv)
        if target_kv is None:
            return pk, pv
        tk, tv = target_kv
        # Residual blend in the spirit of C2CProjector (add_self=True,
        # preserve_target_weight=False) when sequence lengths match.
        if tk.shape == pk.shape:
            return tk + pk, tv + pv
        return pk, pv

    def project_slice(self, kv: KVSlice) -> KVSlice:
        keys, values = [], []
        for k, v in zip(kv.keys, kv.values):
            pk, pv = self.project_pair(k, v)
            keys.append(pk)
            values.append(pv)
        return KVSlice(
            keys=keys,
            values=values,
            layer_indices=list(kv.layer_indices),
            token_start=kv.token_start,
            token_end=kv.token_end,
            head_indices=None,
        )


class RoleConditionedAdapter(nn.Module):
    """Same source capsule, different projection per receiver role.

    A researcher publishing ``codebase.api`` should not look the same to a
    coder (who needs call signatures) and a tester (who needs invariants).
    FiLM scale/shift on the projected KV implements that split without a
    separate fuser per pair.
    """

    ROLES = ("planner", "coder", "tester", "critic", "researcher", "generic")

    def __init__(
        self,
        source_heads: int,
        source_dim: int,
        target_heads: int,
        target_dim: int,
        n_roles: int = 6,
    ):
        super().__init__()
        self.base = LinearKVAdapter(source_heads, source_dim, target_heads, target_dim)
        self.role_scale = nn.Embedding(n_roles, target_dim)
        self.role_shift = nn.Embedding(n_roles, target_dim)
        nn.init.zeros_(self.role_scale.weight)
        nn.init.zeros_(self.role_shift.weight)
        self.n_roles = n_roles

    def role_id(self, role: Union[str, int]) -> int:
        if isinstance(role, int):
            return role
        if role in self.ROLES:
            return self.ROLES.index(role)
        return self.ROLES.index("generic")

    def forward(self, kv: KVSlice, role: Union[str, int] = "generic") -> KVSlice:
        projected = self.base.project_slice(kv)
        rid = torch.tensor(self.role_id(role), device=projected.keys[0].device)
        scale = 1.0 + self.role_scale(rid).view(1, 1, 1, -1)
        shift = self.role_shift(rid).view(1, 1, 1, -1)
        keys = [k * scale + shift for k in projected.keys]
        values = [v * scale + shift for v in projected.values]
        return KVSlice(
            keys=keys,
            values=values,
            layer_indices=list(projected.layer_indices),
            token_start=projected.token_start,
            token_end=projected.token_end,
        )


ProjectorLike = Optional[Union[nn.Module, LinearKVAdapter, RoleConditionedAdapter]]


def _project_layer(
    source_k: Tensor,
    source_v: Tensor,
    target_k: Optional[Tensor],
    target_v: Optional[Tensor],
    projector: ProjectorLike,
    role: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    if projector is None:
        if target_k is not None and source_k.shape != target_k.shape:
            raise ValueError(
                f"shape mismatch {tuple(source_k.shape)} vs {tuple(target_k.shape)} "
                "and no projector provided"
            )
        return source_k, source_v
    if isinstance(projector, RoleConditionedAdapter):
        # Role adapters operate on slices; fall back to base pair projection.
        pk, pv = projector.base.project_pair(source_k, source_v)
        rid = projector.role_id(role or "generic")
        device = pk.device
        scale = 1.0 + projector.role_scale(torch.tensor(rid, device=device)).view(1, 1, 1, -1)
        shift = projector.role_shift(torch.tensor(rid, device=device)).view(1, 1, 1, -1)
        return pk * scale + shift, pv * scale + shift
    if isinstance(projector, LinearKVAdapter):
        if target_k is None:
            return projector.project_pair(source_k, source_v)
        return projector.forward((source_k, source_v), (target_k, target_v))
    # Generic C2CProjector-like: forward(source_kv, target_kv) -> (k, v)
    if target_k is None:
        zeros = torch.zeros(
            source_k.shape[0],
            getattr(projector, "target_num_heads", source_k.shape[1]),
            source_k.shape[2],
            getattr(projector, "target_dim", source_k.shape[3]),
            device=source_k.device,
            dtype=source_k.dtype,
        )
        return projector.forward((source_k, source_v), (zeros, torch.zeros_like(zeros)))
    return projector.forward((source_k, source_v), (target_k, target_v))


def _resize_seq(x: Tensor, n: int) -> Tensor:
    if x.shape[2] == n:
        return x
    if x.shape[2] > n:
        return x[:, :, :n, :]
    pad = n - x.shape[2]
    return F.pad(x, (0, 0, 0, pad))


def _match_geometry(src: KVSlice, dst: KVSlice, projector: ProjectorLike, role: Optional[str]) -> KVSlice:
    """Project source layers onto destination layer count / geometry."""
    src_map = align_layers(src.n_layers, dst.n_layers)
    keys, values = [], []
    for dst_i, src_i in enumerate(src_map):
        sk, sv = src.keys[src_i], src.values[src_i]
        tk = dst.keys[dst_i]
        # Sequence length: keep source length (prefix) or match dst later.
        pk, pv = _project_layer(sk, sv, None, None, projector, role)
        if pk.shape[1] != tk.shape[1] or pk.shape[3] != tk.shape[3]:
            # Last-resort interpolate heads/dim by linear adapter on the fly.
            adapter = LinearKVAdapter(pk.shape[1], pk.shape[3], tk.shape[1], tk.shape[3])
            adapter = adapter.to(device=pk.device, dtype=pk.dtype)
            pk, pv = adapter.project_pair(pk, pv)
        keys.append(pk)
        values.append(pv)
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(dst.layer_indices),
        token_start=src.token_start,
        token_end=src.token_end,
    )


def overlay_prefix(receiver: KVSlice, capsule_kv: KVSlice) -> KVSlice:
    """Phantom-token injection: capsule occupies positions before receiver tokens."""
    return cat_seq(capsule_kv, receiver)


def overlay_positional(
    receiver: KVSlice,
    capsule_kv: KVSlice,
    start: int = 0,
    blend: str = "replace",
    scale: float = 1.0,
) -> KVSlice:
    n = capsule_kv.seq_len
    end = start + n
    if end > receiver.seq_len:
        raise ValueError(
            f"positional overlay [{start}, {end}) exceeds receiver seq {receiver.seq_len}"
        )
    keys, values = [], []
    for rk, rv, ck, cv in zip(receiver.keys, receiver.values, capsule_kv.keys, capsule_kv.values):
        ck = _resize_seq(ck, n)
        cv = _resize_seq(cv, n)
        new_k = rk.clone()
        new_v = rv.clone()
        if blend == "replace":
            new_k[:, :, start:end, :] = ck
            new_v[:, :, start:end, :] = cv
        elif blend == "residual":
            new_k[:, :, start:end, :] = rk[:, :, start:end, :] + scale * ck
            new_v[:, :, start:end, :] = rv[:, :, start:end, :] + scale * cv
        elif blend == "average":
            new_k[:, :, start:end, :] = 0.5 * (rk[:, :, start:end, :] + ck)
            new_v[:, :, start:end, :] = 0.5 * (rv[:, :, start:end, :] + cv)
        else:
            raise ValueError(f"unknown blend: {blend}")
        keys.append(new_k)
        values.append(new_v)
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(receiver.layer_indices),
        token_start=receiver.token_start,
        token_end=receiver.token_end,
        head_indices=receiver.head_indices,
    )


def overlay_critique(receiver: KVSlice, critic_kv: KVSlice, gate: float = 1.0) -> KVSlice:
    """Write a disagreement field: receiver += gate * (critic - receiver) on the span."""
    n = min(receiver.seq_len, critic_kv.seq_len)
    keys, values = [], []
    for rk, rv, ck, cv in zip(receiver.keys, receiver.values, critic_kv.keys, critic_kv.values):
        ck = _resize_seq(ck, n)
        cv = _resize_seq(cv, n)
        new_k = rk.clone()
        new_v = rv.clone()
        new_k[:, :, :n, :] = rk[:, :, :n, :] + gate * (ck - rk[:, :, :n, :])
        new_v[:, :, :n, :] = rv[:, :, :n, :] + gate * (cv - rv[:, :, :n, :])
        keys.append(new_k)
        values.append(new_v)
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(receiver.layer_indices),
        token_start=receiver.token_start,
        token_end=receiver.token_end,
        head_indices=receiver.head_indices,
    )


def overlay_fuse_parallel(receiver: KVSlice, sources: Sequence[KVSlice], scale: float = 1.0) -> KVSlice:
    """Paper multi-sharer parallel mode: sum residuals from a clean base."""
    keys = [k.clone() for k in receiver.keys]
    values = [v.clone() for v in receiver.values]
    n_recv = receiver.seq_len
    for src in sources:
        n = min(n_recv, src.seq_len)
        for i, (ck, cv) in enumerate(zip(src.keys, src.values)):
            ck = _resize_seq(ck, n)
            cv = _resize_seq(cv, n)
            base_k = receiver.keys[i][:, :, :n, :]
            base_v = receiver.values[i][:, :, :n, :]
            keys[i][:, :, :n, :] = keys[i][:, :, :n, :] + scale * (ck - base_k)
            values[i][:, :, :n, :] = values[i][:, :, :n, :] + scale * (cv - base_v)
    return KVSlice(
        keys=keys,
        values=values,
        layer_indices=list(receiver.layer_indices),
        token_start=receiver.token_start,
        token_end=receiver.token_end,
        head_indices=receiver.head_indices,
    )


def overlay(
    receiver: KVSlice,
    capsule: Union[SemanticCapsule, KVSlice],
    mode: OverlayMode = OverlayMode.PREFIX,
    projector: ProjectorLike = None,
    role: Optional[str] = None,
    start: int = 0,
    gate: float = 1.0,
    extra_sources: Optional[Sequence[KVSlice]] = None,
) -> KVSlice:
    """Apply a capsule (or raw slice) onto a receiver cache."""
    src_kv = capsule.kv if isinstance(capsule, SemanticCapsule) else capsule
    if isinstance(capsule, SemanticCapsule) and role is None:
        role = capsule.role or None
    matched = _match_geometry(src_kv, receiver, projector, role) if projector is not None or src_kv.n_layers != receiver.n_layers or src_kv.n_heads != receiver.n_heads or src_kv.head_dim != receiver.head_dim else src_kv.clone()

    if mode == OverlayMode.PREFIX:
        # Prefix uses projected source length, not fused with target positions.
        if projector is not None:
            matched = _match_geometry(src_kv, receiver, projector, role)
        return overlay_prefix(receiver.clone(), matched)
    if mode == OverlayMode.POSITIONAL:
        return overlay_positional(receiver.clone(), matched, start=start, blend="replace")
    if mode == OverlayMode.RESIDUAL:
        return overlay_positional(receiver.clone(), matched, start=start, blend="residual", scale=gate)
    if mode == OverlayMode.REPLACE:
        if matched.seq_len != receiver.seq_len:
            # Replace the overlapping prefix, keep the tail of the receiver.
            n = min(matched.seq_len, receiver.seq_len)
            out = overlay_positional(receiver.clone(), 
                                     KVSlice(keys=[k[:, :, :n, :] for k in matched.keys],
                                             values=[v[:, :, :n, :] for v in matched.values],
                                             layer_indices=list(matched.layer_indices),
                                             token_start=matched.token_start,
                                             token_end=matched.token_start + n),
                                     start=0, blend="replace")
            return out
        return matched
    if mode == OverlayMode.STREAM_APPEND:
        return cat_seq(receiver.clone(), matched)
    if mode == OverlayMode.CRITIQUE:
        return overlay_critique(receiver.clone(), matched, gate=gate)
    if mode == OverlayMode.FUSE_PARALLEL:
        sources = [matched]
        if extra_sources:
            sources.extend(extra_sources)
        return overlay_fuse_parallel(receiver.clone(), sources, scale=gate)
    raise ValueError(f"unknown overlay mode: {mode}")


def remote_retrieve(
    query: Tensor,
    capsule: Union[SemanticCapsule, KVSlice],
    *,
    topk: int = 8,
    layer: int = -1,
    head: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pull-style C2C: attend into a remote capsule without copying it all.

    Parameters
    ----------
    query:
        ``(B, H, Q, D)`` or ``(B, Q, D)`` query from the receiver's current
        token(s). This is what would be sent over the network in Cache-RPC.
    capsule:
        Remote agent's published KV.
    topk:
        Only these key/value positions return. Bandwidth scales with Q and
        k, not with the remote context length.
    layer / head:
        Which remote layer (and optional single head) to query.

    Returns
    -------
    values : (B, Q, D)  attended values (mean over heads if query is 3D)
    indices : (B, H, Q, k) selected token indices
    weights : (B, H, Q, k) attention weights over the selected keys
    """
    kv = capsule.kv if isinstance(capsule, SemanticCapsule) else capsule
    key = kv.keys[layer]
    value = kv.values[layer]
    if head is not None:
        key = key[:, head : head + 1]
        value = value[:, head : head + 1]
    if query.dim() == 3:
        query = query.unsqueeze(1).expand(-1, key.shape[1], -1, -1)
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query dim {query.shape[-1]} != key dim {key.shape[-1]}")
    # scores: (B, H, Q, N)
    scale = key.shape[-1] ** 0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) / scale
    k = min(topk, scores.shape[-1])
    weights_full = torch.softmax(scores, dim=-1)
    top_w, top_idx = torch.topk(weights_full, k, dim=-1)
    # Gather values at top-k positions: (B, H, Q, k, D)
    idx_exp = top_idx.unsqueeze(-1).expand(*top_idx.shape, value.shape[-1])
    value_exp = value.unsqueeze(2).expand(-1, -1, query.shape[2], -1, -1)
    gathered = torch.gather(value_exp, dim=3, index=idx_exp)
    mixed = (gathered * top_w.unsqueeze(-1)).sum(dim=3)  # (B, H, Q, D)
    mixed = mixed.mean(dim=1)  # (B, Q, D)
    return mixed, top_idx, top_w


def relevance_token_mask(
    query: Tensor,
    kv: KVSlice,
    *,
    topk: int = 16,
    layer: int = -1,
) -> Tensor:
    """Which source tokens are relevant to a receiver query (selective export).

    Returns a 1D LongTensor of token indices the sender should include in
    a capsule instead of dumping its full cache.
    """
    key = kv.keys[layer]
    if query.dim() == 3:
        query = query.unsqueeze(1)
    scale = key.shape[-1] ** 0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) / scale  # (B, H, Q, N)
    token_score = scores.mean(dim=(0, 1, 2))  # (N,)
    k = min(topk, token_score.numel())
    return torch.topk(token_score, k).indices.sort().values
