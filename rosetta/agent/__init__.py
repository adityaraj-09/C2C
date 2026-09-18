"""Agent-to-agent communication over Cache-to-Cache.

This package is a communication primitive, not an agent framework.

The paper's C2C fuses two models at the same token positions of the same
prompt. Agents do not share a prompt: a planner has a goal, a coder has
files, a tester has failures. Semantic Capsules let one agent transfer a
sliced KV region — a piece of *understanding* — into another agent's
working memory, or let the receiver *query* a remote cache via attention.

Public surface::

    from rosetta.agent import (
        SemanticCapsule, LatentBus, C2CProtocol, OverlayMode, Intent,
    )
"""

from rosetta.agent.capsule import (
    Intent,
    KVSlice,
    SemanticCapsule,
    compose_lineage,
    empty_kv,
    extract_heads,
    extract_layers,
    extract_span,
    pool_tokens,
)
from rosetta.agent.overlay import (
    OverlayMode,
    LinearKVAdapter,
    RoleConditionedAdapter,
    align_layers,
    overlay,
    remote_retrieve,
)
from rosetta.agent.bus import LatentBus, SlotConflict
from rosetta.agent.protocol import AgentEndpoint, C2CChannel, C2CProtocol

__all__ = [
    "AgentEndpoint",
    "C2CChannel",
    "C2CProtocol",
    "Intent",
    "KVSlice",
    "LatentBus",
    "LinearKVAdapter",
    "OverlayMode",
    "RoleConditionedAdapter",
    "SemanticCapsule",
    "SlotConflict",
    "align_layers",
    "compose_lineage",
    "empty_kv",
    "extract_heads",
    "extract_layers",
    "extract_span",
    "overlay",
    "pool_tokens",
    "remote_retrieve",
]
