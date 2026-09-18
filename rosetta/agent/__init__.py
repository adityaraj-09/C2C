"""Same-model agent-to-agent C2C.

User → agent is text. Agent → agent is a KV capsule. One shared HuggingFace
causal LM, copy-on-write timelines, join by adopting the child's cache.

    from rosetta.agent import CodingRuntime, build_tiny_llama

    engine, tok = build_tiny_llama()
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the auth cache bug in foo.py")
    rt.fork("explorer", role="explore")
    rt.ingest_env("explorer", "class authcache keys on write misses ttl")
    rt.adopt("parent", "explorer")
    print(rt.reply())
"""

from rosetta.agent.agent import Agent
from rosetta.agent.capsule import Capsule, Intent
from rosetta.agent.engine import SharedCausalEngine
from rosetta.agent.errors import (
    AgentC2CError,
    AgentStateError,
    CapsuleError,
    ModelMismatchError,
    TimelineError,
)
from rosetta.agent.instruct import (
    instruct_model_available,
    load_instruct_engine,
)
from rosetta.agent.kv import clone_cache, model_fingerprint
from rosetta.agent.prefix_cache import PrefixBlockCache
from rosetta.agent.runtime import CodingRuntime
from rosetta.agent.tickets import default_tickets, score_ticket, score_ticket_modes
from rosetta.agent.tiny import WordTokenizer, build_tiny_llama

__all__ = [
    "Agent",
    "AgentC2CError",
    "AgentStateError",
    "Capsule",
    "CapsuleError",
    "CodingRuntime",
    "Intent",
    "ModelMismatchError",
    "PrefixBlockCache",
    "SharedCausalEngine",
    "TimelineError",
    "WordTokenizer",
    "build_tiny_llama",
    "clone_cache",
    "default_tickets",
    "instruct_model_available",
    "load_instruct_engine",
    "model_fingerprint",
    "score_ticket",
    "score_ticket_modes",
]
