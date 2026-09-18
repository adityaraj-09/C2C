"""C2C agent communication protocol.

Verbs, not agents. Each method moves a capsule across the bus with an
intent-specific overlay. Endpoints are thin handles around a current KV
working memory — enough to demonstrate the primitive, not a runtime.

Typical coding-agent wiring::

    bus = LatentBus()
    proto = C2CProtocol(bus)
    planner, coder, tester = (AgentEndpoint(n, role, kv) for ...)

    proto.delegate(planner, coder, slot="plan.intent", span=(sys_len, plan_len))
    proto.handoff(coder, tester, slot="impl.working_memory")
    reply = proto.consult(tester, coder, query)   # remote attention
    proto.critique(tester, coder, slot="test.failures")
    proto.stream(planner, "plan.intent", new_tokens_kv)

A ``C2CChannel`` is a persistent bidirectional pipe with a pair of
projectors, so A→B and B→A are different maps (heterogeneous models,
asymmetric roles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from rosetta.agent.bus import LatentBus
from rosetta.agent.capsule import Intent, KVSlice, SemanticCapsule
from rosetta.agent.overlay import (
    OverlayMode,
    ProjectorLike,
    overlay,
    relevance_token_mask,
    remote_retrieve,
)


@dataclass
class AgentEndpoint:
    """Working memory handle. Not a tool loop, not a framework agent."""

    name: str
    role: str
    kv: KVSlice
    projector: ProjectorLike = None

    def clone(self) -> "AgentEndpoint":
        return AgentEndpoint(self.name, self.role, self.kv.clone(), self.projector)

    def capsule(
        self,
        slot: str,
        intent: Intent,
        *,
        span: Optional[Tuple[int, int]] = None,
        layers: Optional[Sequence[int]] = None,
        heads: Optional[Sequence[int]] = None,
        pool: int = 1,
        provenance: str = "",
        extra: Optional[dict] = None,
    ) -> SemanticCapsule:
        cap = SemanticCapsule(
            source=self.name,
            slot=slot,
            intent=intent,
            kv=self.kv.clone(),
            role=self.role,
            provenance=provenance,
            extra=extra or {},
        )
        return cap.sliced(span=span, layers=layers, heads=heads, pool=pool)

    def absorb(
        self,
        capsule: SemanticCapsule,
        mode: OverlayMode,
        projector: ProjectorLike = None,
        start: int = 0,
        gate: float = 1.0,
    ) -> "AgentEndpoint":
        proj = projector if projector is not None else self.projector
        self.kv = overlay(self.kv, capsule, mode=mode, projector=proj,
                          role=self.role, start=start, gate=gate)
        return self


@dataclass
class ConsultResult:
    values: Tensor
    indices: Tensor
    weights: Tensor
    reply: Optional[SemanticCapsule] = None


class C2CProtocol:
    """Intent-aware capsule transport on a LatentBus."""

    def __init__(self, bus: Optional[LatentBus] = None):
        self.bus = bus if bus is not None else LatentBus()

    def publish(self, endpoint: AgentEndpoint, slot: str, intent: Intent, **slice_kwargs) -> SemanticCapsule:
        cap = endpoint.capsule(slot, intent, **slice_kwargs)
        return self.bus.publish(cap)

    def _overlay_for(self, intent: Intent) -> OverlayMode:
        return {
            Intent.HANDOFF: OverlayMode.PREFIX,
            Intent.DELEGATE: OverlayMode.PREFIX,
            Intent.CONSULT: OverlayMode.PREFIX,
            Intent.FUSE: OverlayMode.FUSE_PARALLEL,
            Intent.STREAM: OverlayMode.STREAM_APPEND,
            Intent.CRITIQUE: OverlayMode.CRITIQUE,
            Intent.SYNC: OverlayMode.RESIDUAL,
        }[intent]

    def handoff(
        self,
        src: AgentEndpoint,
        dst: AgentEndpoint,
        slot: str,
        **slice_kwargs,
    ) -> SemanticCapsule:
        """Yield working memory. Receiver gets it as a latent prefix.

        The interesting case is mid-task: src has not finished, dst continues
        from src's cache rather than from a written briefing. Optional
        ``layers`` / ``span`` restrict what is inherited (capability isolation).
        """
        cap = src.capsule(slot, Intent.HANDOFF, **slice_kwargs)
        self.bus.publish(cap, append=False)
        dst.absorb(cap, OverlayMode.PREFIX)
        return cap

    def delegate(
        self,
        src: AgentEndpoint,
        dst: AgentEndpoint,
        slot: str,
        **slice_kwargs,
    ) -> SemanticCapsule:
        """Send a latent task spec. Sender keeps its own cache."""
        cap = src.capsule(slot, Intent.DELEGATE, **slice_kwargs)
        self.bus.publish(cap)
        dst.absorb(cap, OverlayMode.PREFIX)
        return cap

    def consult(
        self,
        src: AgentEndpoint,
        dst: AgentEndpoint,
        query: Tensor,
        slot: str = "consult.query",
        topk: int = 8,
        layer: int = -1,
        also_prefix: bool = False,
    ) -> ConsultResult:
        """Pull-style: src queries dst's cache via remote top-k attention.

        ``query`` is typically the current-token query of ``src``. Nothing
        textual is exchanged. If ``also_prefix`` is set, the retrieved KV
        positions are also injected as a prefix (hybrid push/pull).
        """
        cap = dst.capsule(slot, Intent.CONSULT, provenance=f"consult from {src.name}")
        self.bus.publish(cap, append=False)
        values, indices, weights = remote_retrieve(query, cap, topk=topk, layer=layer)
        reply = None
        if also_prefix:
            # Materialize only the retrieved positions as a tiny capsule.
            idx = indices[0, 0, 0].tolist()  # first batch/head/query
            from rosetta.agent.capsule import extract_heads  # local; we gather tokens
            gathered_k = []
            gathered_v = []
            for k, v in zip(cap.kv.keys, cap.kv.values):
                gathered_k.append(k[:, :, idx, :])
                gathered_v.append(v[:, :, idx, :])
            reply = SemanticCapsule(
                source=dst.name,
                slot=slot + ".reply",
                intent=Intent.CONSULT,
                kv=KVSlice(
                    keys=gathered_k,
                    values=gathered_v,
                    layer_indices=list(cap.kv.layer_indices),
                    token_start=0,
                    token_end=len(idx),
                ),
                role=dst.role,
                provenance="consult top-k materialize",
            )
            src.absorb(reply, OverlayMode.PREFIX)
            self.bus.publish(reply)
        return ConsultResult(values=values, indices=indices, weights=weights, reply=reply)

    def fuse(
        self,
        dst: AgentEndpoint,
        slot: str,
        projector: ProjectorLike = None,
    ) -> KVSlice:
        dst.kv = self.bus.fuse_slot(slot, dst.kv, projector=projector, role=dst.role)
        return dst.kv

    def stream(
        self,
        src: AgentEndpoint,
        slot: str,
        new_kv: KVSlice,
        dst: Optional[AgentEndpoint] = None,
    ) -> SemanticCapsule:
        """Publish newly thought tokens. Optionally append them onto dst now."""
        cap = SemanticCapsule(
            source=src.name,
            slot=slot,
            intent=Intent.STREAM,
            kv=new_kv,
            role=src.role,
            provenance="stream",
        )
        grown = self.bus.publish(cap)
        if dst is not None:
            live = SemanticCapsule(
                source=src.name, slot=slot, intent=Intent.STREAM, kv=new_kv, role=src.role
            )
            dst.absorb(live, OverlayMode.STREAM_APPEND)
        return grown

    def critique(
        self,
        critic: AgentEndpoint,
        generator: AgentEndpoint,
        slot: str = "critique.residual",
        gate: float = 0.5,
        **slice_kwargs,
    ) -> SemanticCapsule:
        cap = critic.capsule(slot, Intent.CRITIQUE, **slice_kwargs)
        self.bus.publish(cap, append=False)
        generator.absorb(cap, OverlayMode.CRITIQUE, gate=gate)
        return cap

    def sync(
        self,
        a: AgentEndpoint,
        b: AgentEndpoint,
        slot: str = "sync",
        gate: float = 0.5,
    ) -> Tuple[SemanticCapsule, SemanticCapsule]:
        """Bidirectional residual exchange of full working memory."""
        cap_ab = a.capsule(slot + ".a", Intent.SYNC)
        cap_ba = b.capsule(slot + ".b", Intent.SYNC)
        self.bus.publish(cap_ab, append=False)
        self.bus.publish(cap_ba, append=False)
        # Absorb copies so neither side sees a half-updated peer.
        b.absorb(cap_ab, OverlayMode.RESIDUAL, gate=gate)
        a.absorb(cap_ba, OverlayMode.RESIDUAL, gate=gate)
        return cap_ab, cap_ba

    def selective_export(
        self,
        src: AgentEndpoint,
        query: Tensor,
        slot: str,
        topk: int = 16,
        intent: Intent = Intent.DELEGATE,
    ) -> SemanticCapsule:
        """Export only the tokens in src that are relevant to ``query``."""
        idx = relevance_token_mask(query, src.kv, topk=topk)
        keys = [k[:, :, idx, :] for k in src.kv.keys]
        values = [v[:, :, idx, :] for v in src.kv.values]
        cap = SemanticCapsule(
            source=src.name,
            slot=slot,
            intent=intent,
            kv=KVSlice(
                keys=keys,
                values=values,
                layer_indices=list(src.kv.layer_indices),
                token_start=0,
                token_end=int(idx.numel()),
                head_indices=src.kv.head_indices,
            ),
            role=src.role,
            provenance="selective_export",
            extra={"token_indices": idx.tolist()},
        )
        self.bus.publish(cap, append=False)
        return cap


@dataclass
class C2CChannel:
    """Persistent bidirectional latent pipe between two endpoints.

    Projectors may differ in each direction (Qwen→Llama is not Llama→Qwen).
    """

    left: AgentEndpoint
    right: AgentEndpoint
    left_to_right: ProjectorLike = None
    right_to_left: ProjectorLike = None
    bus: LatentBus = field(default_factory=LatentBus)

    def __post_init__(self) -> None:
        self.proto = C2CProtocol(self.bus)

    def send(self, direction: str, slot: str, intent: Intent, **slice_kwargs) -> SemanticCapsule:
        if direction == "lr":
            src, dst, proj = self.left, self.right, self.left_to_right
        elif direction == "rl":
            src, dst, proj = self.right, self.left, self.right_to_left
        else:
            raise ValueError("direction must be 'lr' or 'rl'")
        cap = src.capsule(slot, intent, **slice_kwargs)
        self.bus.publish(cap)
        mode = self.proto._overlay_for(intent)
        dst.absorb(cap, mode, projector=proj)
        return cap

    def ping_pong_consult(self, query_from_left: Tensor, topk: int = 8) -> ConsultResult:
        return self.proto.consult(self.left, self.right, query_from_left, topk=topk)
