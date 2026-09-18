"""Production tests for same-model agent C2C.

The invariant that matters: inheriting a KV cache and continuing must
equal prefilling the concatenated token sequence. If that breaks, agent
handoff is wrong regardless of any higher-level protocol.
"""

from __future__ import annotations

import dataclasses

import torch

from rosetta.agent import (
    Capsule,
    CodingRuntime,
    Intent,
    ModelMismatchError,
    TimelineError,
    build_tiny_llama,
)
from rosetta.agent.capsule import Capsule as CapsuleCls
from rosetta.agent.errors import AgentStateError
from rosetta.agent.kv import crop_prefix


def _engine():
    return build_tiny_llama(n_layers=2, n_heads=4, hidden=32, seed=0)


def test_prefill_then_continue_matches_full_prefill():
    engine, _ = _engine()
    torch.manual_seed(1)
    a = torch.randint(12, 40, (1, 9), device=engine.device)
    b = torch.randint(12, 40, (1, 6), device=engine.device)
    full = torch.cat([a, b], dim=1)
    with torch.inference_mode():
        gold = engine.prefill(full)
        first = engine.prefill(a)
        cont = engine.prefill(b, past=first.past)
    assert torch.equal(gold.logits.argmax(-1), cont.logits.argmax(-1))
    assert torch.allclose(gold.logits, cont.logits, atol=1e-5, rtol=1e-5)


def test_greedy_continue_matches_single_timeline():
    engine, _ = _engine()
    torch.manual_seed(2)
    prefix = torch.randint(12, 40, (1, 10), device=engine.device)
    extra = torch.randint(12, 40, (1, 3), device=engine.device)
    gold_pre = engine.prefill(torch.cat([prefix, extra], dim=1))
    gold_ids, _, _ = engine.greedy_continue(gold_pre.past, gold_pre.logits, max_new_tokens=5)

    first = engine.prefill(prefix)
    c2c_ids, _, _ = engine.greedy_continue(
        first.past, first.logits, max_new_tokens=5, extra_ids=extra
    )
    assert torch.equal(gold_ids, c2c_ids)


def test_forked_children_diverge_independently():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok, max_seq_len=256)
    rt.ingest_user("fix the auth cache bug")
    rt.fork("a", role="explore")
    rt.fork("b", role="coder")
    rt.ingest_env("a", "class authcache misses ttl")
    rt.ingest_env("b", "test fails assert stale entry")
    assert rt.agents["a"].seq_len != rt.agents["b"].seq_len or not torch.equal(
        rt.agents["a"].token_ids, rt.agents["b"].token_ids
    )
    # parent timeline unchanged by children
    parent_len = rt.agents["parent"].seq_len
    assert parent_len < rt.agents["a"].seq_len


def test_adopt_equals_single_agent_ingesting_the_same_env():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok, max_seq_len=256)
    rt.ingest_user("fix the auth cache bug in foo.py")
    rt.fork("explorer", role="explore")
    rt.ingest_env("explorer", "class authcache keys on write misses ttl")
    rt.adopt("parent", "explorer")
    reply = rt.reply(max_new_tokens=6)

    gold = CodingRuntime(engine, tok, max_seq_len=256)
    gold.ingest_user("fix the auth cache bug in foo.py")
    gold.ingest_env("parent", "class authcache keys on write misses ttl")
    gold_reply = gold.reply(max_new_tokens=6)
    assert reply == gold_reply
    assert torch.equal(rt.agents["parent"].token_ids, gold.agents["parent"].token_ids)


def test_capsule_has_no_text_message_field():
    names = {f.name for f in dataclasses.fields(CapsuleCls)}
    assert "message" not in names
    assert "text" not in names
    assert "summary" not in names
    assert "cache" in names and "token_ids" in names


def test_capsule_roundtrip_fp32_preserves_greedy():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("read file foo.py")
    rt.fork("explorer")
    rt.ingest_env("explorer", "def get return none")
    cap = rt.export("explorer")
    blob = cap.to_bytes(quantize="none")
    restored = Capsule.from_bytes(
        blob, device=engine.device, dtype=engine.dtype, expect_fingerprint=engine.fingerprint
    )
    rt.adopt("parent", restored)
    # same as adopting live explorer
    live = CodingRuntime(engine, tok)
    live.ingest_user("read file foo.py")
    live.fork("explorer")
    live.ingest_env("explorer", "def get return none")
    live.adopt("parent", "explorer")
    assert torch.equal(rt.agents["parent"].token_ids, live.agents["parent"].token_ids)
    assert rt.reply(4) == live.reply(4)


def test_int8_wire_format_roundtrips_seq_and_fingerprint():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the bug")
    cap = rt.export("parent")
    blob = cap.to_bytes(quantize="int8")
    assert len(blob) > 0
    restored = Capsule.from_bytes(blob, device=engine.device, dtype=engine.dtype)
    assert restored.seq_len == cap.seq_len
    assert restored.fingerprint == cap.fingerprint
    assert restored.intent is Intent.ADOPT


def test_fingerprint_mismatch_rejected():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the bug")
    cap = rt.export("parent")
    blob = cap.to_bytes()
    bad = dict(engine.fingerprint)
    bad["n_layers"] = 99
    try:
        Capsule.from_bytes(blob, expect_fingerprint=bad)
        assert False, "expected ModelMismatchError"
    except ModelMismatchError:
        pass


def test_empty_export_rejected():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok)
    try:
        rt.export("parent")
        assert False, "expected AgentStateError"
    except AgentStateError:
        pass


def test_crop_prefix_is_allowed_hole_punch_is_not():
    engine, _ = _engine()
    ids = torch.randint(12, 40, (1, 12), device=engine.device)
    past = engine.prefill(ids).past
    cropped = crop_prefix(past, 7)
    assert cropped.get_seq_length() == 7
    try:
        crop_prefix(past, 13)
        assert False
    except TimelineError:
        pass


def test_think_does_not_create_a_peer_text_channel():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the auth cache bug")
    rt.fork("explorer")
    n = rt.think("explorer", max_new_tokens=5)
    assert n >= 1
    cap = rt.export("explorer")
    # parent does not receive a decoded thought string — only KV
    assert not hasattr(cap, "thought")
    rt.adopt("parent", cap)
    assert rt.agents["parent"].seq_len == rt.agents["explorer"].seq_len


def test_unknown_agent_and_duplicate_fork():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the bug")
    rt.fork("explorer")
    try:
        rt.fork("explorer")
        assert False
    except AgentStateError:
        pass
    try:
        rt.ingest_env("nope", "file")
        assert False
    except AgentStateError:
        pass


def test_max_seq_len_enforced():
    engine, tok = _engine()
    rt = CodingRuntime(engine, tok, max_seq_len=4)
    try:
        rt.ingest_user("fix the auth cache bug in foo.py")
        assert False
    except AgentStateError:
        pass
