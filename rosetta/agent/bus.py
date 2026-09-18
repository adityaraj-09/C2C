"""Named latent address space shared by agents.

Agents do not pass messages through a coordinator. They publish capsules
to slots (``codebase.api``, ``plan.constraints``, ``test.failures``) and
subscribe. A slot with several writers is a shared latent workspace; we
flag *merge conflicts* when two capsules in the same slot disagree in
representation space.

This is closer to a blackboard than to a chat log. The payload is KV,
the address is a slot name, the conflict detector is cosine of pooled
values — not a diff of generated text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from rosetta.agent.capsule import Intent, SemanticCapsule, compose_lineage
from rosetta.agent.overlay import OverlayMode, overlay


@dataclass
class SlotConflict:
    slot: str
    sources: List[str]
    agreement: float
    message: str


@dataclass
class LatentBus:
    """In-process latent bus. Serialize capsules to use it across machines."""

    _slots: Dict[str, List[SemanticCapsule]] = field(default_factory=dict)
    conflict_threshold: float = 0.25

    def publish(self, capsule: SemanticCapsule, *, append: bool = True) -> SemanticCapsule:
        bucket = self._slots.setdefault(capsule.slot, [])
        if capsule.intent == Intent.STREAM and bucket:
            # STREAM grows the latest capsule in this slot along seq dim.
            prev = bucket[-1]
            if prev.source == capsule.source:
                from rosetta.agent.capsule import cat_seq

                grown = SemanticCapsule(
                    source=capsule.source,
                    slot=capsule.slot,
                    intent=Intent.STREAM,
                    kv=cat_seq(prev.kv, capsule.kv),
                    role=capsule.role or prev.role,
                    provenance=prev.provenance,
                    lineage=list(prev.lineage),
                    extra={**prev.extra, **capsule.extra, "stream_steps": prev.extra.get("stream_steps", 1) + 1},
                )
                bucket[-1] = grown
                return grown
        if not append:
            self._slots[capsule.slot] = [capsule]
            return capsule
        bucket.append(capsule)
        return capsule

    def subscribe(self, slot: str) -> List[SemanticCapsule]:
        return list(self._slots.get(slot, []))

    def latest(self, slot: str) -> Optional[SemanticCapsule]:
        bucket = self._slots.get(slot, [])
        return bucket[-1] if bucket else None

    def slots(self) -> List[str]:
        return sorted(self._slots.keys())

    def drop(self, slot: str) -> None:
        self._slots.pop(slot, None)

    def clear(self) -> None:
        self._slots.clear()

    def snapshot(self) -> Dict[str, int]:
        return {slot: len(caps) for slot, caps in self._slots.items()}

    def agreement(self, slot: str) -> Optional[float]:
        """Mean pairwise cosine of pooled values in a slot. 1 = identical."""
        caps = self.subscribe(slot)
        if len(caps) < 2:
            return None
        vecs = []
        for cap in caps:
            v = cap.kv.pooled_value(-1)
            v = torch.nn.functional.normalize(v.flatten().float(), dim=0)
            vecs.append(v)
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(torch.dot(vecs[i], vecs[j])))
        return sum(sims) / len(sims)

    def conflicts(self, slot: Optional[str] = None) -> List[SlotConflict]:
        names = [slot] if slot else self.slots()
        found: List[SlotConflict] = []
        for name in names:
            score = self.agreement(name)
            if score is None:
                continue
            if score < self.conflict_threshold:
                caps = self.subscribe(name)
                found.append(SlotConflict(
                    slot=name,
                    sources=[c.source for c in caps],
                    agreement=score,
                    message=(
                        f"slot '{name}' latent disagreement "
                        f"(cosine={score:.3f} < {self.conflict_threshold})"
                    ),
                ))
        return found

    def fuse_slot(
        self,
        slot: str,
        receiver_kv,
        *,
        mode: OverlayMode = OverlayMode.FUSE_PARALLEL,
        projector=None,
        role: Optional[str] = None,
    ):
        """Fold every capsule in ``slot`` into a receiver cache."""
        caps = self.subscribe(slot)
        if not caps:
            return receiver_kv.clone()
        if mode == OverlayMode.FUSE_PARALLEL and len(caps) > 1:
            first = caps[0]
            extra = [c.kv for c in caps[1:]]
            return overlay(receiver_kv, first, mode=OverlayMode.FUSE_PARALLEL,
                           projector=projector, role=role, extra_sources=extra)
        out = receiver_kv
        for cap in caps:
            out = overlay(out, cap, mode=mode, projector=projector, role=role)
        return out

    def compose(self, slots: Sequence[str], source: str, dest_slot: str) -> SemanticCapsule:
        """Hierarchical chain: concatenate listed slots into one lineage capsule."""
        caps: List[SemanticCapsule] = []
        for name in slots:
            latest = self.latest(name)
            if latest is None:
                raise KeyError(f"empty slot: {name}")
            caps.append(latest)
        composed = compose_lineage(caps, slot=dest_slot, source=source)
        self.publish(composed, append=False)
        return composed


def workspace_view(bus: LatentBus, slots: Iterable[str]) -> Dict[str, Optional[SemanticCapsule]]:
    return {name: bus.latest(name) for name in slots}
