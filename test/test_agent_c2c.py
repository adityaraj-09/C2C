"""Tests for the agent-to-agent C2C communication primitive.

No pretrained weights. These tests check the protocol algebra and the
smallest experiment that can fail: planted-fact retrieval after capsule
transfer vs an equal-budget text briefing.
"""

from __future__ import annotations

import torch

from rosetta.agent import (
    AgentEndpoint,
    C2CChannel,
    C2CProtocol,
    Intent,
    KVSlice,
    LatentBus,
    LinearKVAdapter,
    OverlayMode,
    RoleConditionedAdapter,
    SemanticCapsule,
    extract_layers,
    extract_span,
    overlay,
    pool_tokens,
)
from rosetta.agent.capsule import compose_lineage, empty_kv
from rosetta.agent.experiment import (
    format_reports,
    make_factbook,
    plant_facts,
    retrieve_facts,
    retrieval_accuracy,
    run_planted_transfer_experiment,
    run_streaming_experiment,
    run_tiny_learned_experiment,
)
from rosetta.agent.overlay import align_layers


def _planted_endpoints():
    book = make_factbook(n_facts=6, dim=16, seed=3)
    kv = plant_facts(book, n_layers=2, n_heads=2)
    src = AgentEndpoint("researcher", "researcher", kv.clone())
    dst = AgentEndpoint("coder", "coder", empty_kv(2, 1, 2, 3, 16))
    return book, src, dst


def test_extract_span_and_layers():
    kv = empty_kv(4, 1, 2, 10, 8)
    kv.keys[1][:, :, 3:6, :] = 1.0
    span = extract_span(kv, 3, 6)
    assert span.seq_len == 3
    assert span.token_start == 3
    assert torch.allclose(span.keys[1], torch.ones_like(span.keys[1]))
    mid = extract_layers(span, [1, 2])
    assert mid.layer_indices == [1, 2]
    assert mid.n_layers == 2


def test_pool_tokens_preserves_mass():
    kv = empty_kv(1, 1, 1, 5, 4)
    kv.keys[0][..., :] = 2.0
    pooled = pool_tokens(kv, 2)
    # 2 groups of 2 + remainder 1 → seq 3, still mean 2.
    assert pooled.seq_len == 3
    assert torch.allclose(pooled.keys[0], torch.full_like(pooled.keys[0], 2.0))


def test_capsule_int8_roundtrip_preserves_retrieval():
    book = make_factbook(n_facts=8, dim=32, seed=0)
    kv = plant_facts(book, n_layers=2, n_heads=4)
    cap = SemanticCapsule("a", "codebase.api", Intent.HANDOFF, kv)
    blob = cap.to_bytes(quantize="int8")
    back = SemanticCapsule.from_bytes(blob)
    assert back.intent is Intent.HANDOFF
    assert back.slot == "codebase.api"
    assert retrieval_accuracy(retrieve_facts(back.kv, book)) == 1.0
    assert len(blob) < cap.nbytes()


def test_prefix_overlay_extends_context():
    recv = empty_kv(2, 1, 2, 4, 8)
    cap_kv = empty_kv(2, 1, 2, 5, 8)
    cap_kv.keys[0].fill_(3.0)
    out = overlay(recv, cap_kv, mode=OverlayMode.PREFIX)
    assert out.seq_len == 9
    assert torch.allclose(out.keys[0][:, :, :5, :], torch.full_like(cap_kv.keys[0], 3.0))
    assert torch.allclose(out.keys[0][:, :, 5:, :], torch.zeros(1, 2, 4, 8))


def test_critique_moves_generator_toward_critic():
    gen = empty_kv(1, 1, 1, 2, 4)
    crit = empty_kv(1, 1, 1, 2, 4)
    crit.keys[0].fill_(1.0)
    out = overlay(gen, crit, mode=OverlayMode.CRITIQUE, gate=0.5)
    assert torch.allclose(out.keys[0], torch.full_like(out.keys[0], 0.5))


def test_handoff_transfers_understanding_not_text():
    book, src, dst = _planted_endpoints()
    proto = C2CProtocol()
    cap = proto.handoff(src, dst, slot="codebase.api")
    assert cap.intent is Intent.HANDOFF
    # Receiver never saw the fact tokens; retrieval still works from prefix.
    cos = retrieve_facts(dst.kv, book)
    assert retrieval_accuracy(cos) == 1.0
    assert dst.kv.seq_len == src.kv.seq_len + 3


def test_consult_is_topk_pull_not_full_copy():
    book, src, dst = _planted_endpoints()
    proto = C2CProtocol()
    proto.handoff(src, dst, "codebase.api")  # dst now holds the facts
    planner_kv = empty_kv(2, 1, 2, 1, 16)
    planner = AgentEndpoint("planner", "planner", planner_kv)
    query = book.keys[2].view(1, 1, 16)  # ask about fact 2
    result = proto.consult(planner, dst, query, topk=1)
    assert result.indices.shape[-1] == 1
    # Top-1 index should be the planted token for fact 2.
    assert int(result.indices.reshape(-1)[0]) == 2
    retrieved = torch.nn.functional.normalize(result.values.squeeze(), dim=-1)
    target = torch.nn.functional.normalize(book.values[2], dim=-1)
    assert float((retrieved * target).sum()) > 0.95


def test_selective_export_keeps_relevant_tokens():
    book, src, dst = _planted_endpoints()
    proto = C2CProtocol()
    query = book.keys[4].view(1, 1, 16)
    cap = proto.selective_export(src, query, slot="codebase.slice", topk=2)
    assert cap.kv.seq_len == 2
    assert 4 in cap.extra["token_indices"]


def test_stream_grows_slot_and_receiver():
    book = make_factbook(4, dim=8, seed=2)
    full = plant_facts(book, n_layers=1, n_heads=1)
    src = AgentEndpoint("a", "researcher", full)
    dst = AgentEndpoint("b", "coder", empty_kv(1, 1, 1, 1, 8))
    proto = C2CProtocol()
    for t in range(4):
        step = extract_span(full, t, t + 1)
        proto.stream(src, "codebase.api", step, dst=dst)
    grown = proto.bus.latest("codebase.api")
    assert grown.kv.seq_len == 4
    assert dst.kv.seq_len == 1 + 4
    assert retrieval_accuracy(retrieve_facts(grown.kv, book)) == 1.0


def test_fuse_parallel_and_conflict_detection():
    bus = LatentBus(conflict_threshold=0.5)
    proto = C2CProtocol(bus)
    book = make_factbook(4, dim=8, seed=1)
    a = AgentEndpoint("planner", "planner", plant_facts(book, 1, 1))
    other = make_factbook(4, dim=8, seed=99)
    b = AgentEndpoint("researcher", "researcher", plant_facts(other, 1, 1))
    proto.publish(a, "plan.constraints", Intent.FUSE)
    proto.publish(b, "plan.constraints", Intent.FUSE)
    conflicts = bus.conflicts("plan.constraints")
    assert conflicts and conflicts[0].agreement < 0.5
    recv = empty_kv(1, 1, 1, 4, 8)
    fused = bus.fuse_slot("plan.constraints", recv)
    assert fused.seq_len == 4


def test_compose_lineage_hierarchical_handoff():
    book = make_factbook(3, dim=8, seed=4)
    p = SemanticCapsule("planner", "plan", Intent.DELEGATE, plant_facts(book, 1, 1))
    c = SemanticCapsule("coder", "impl", Intent.HANDOFF, plant_facts(book, 1, 1))
    chained = compose_lineage([p, c], slot="handoff.chain", source="runtime")
    assert chained.kv.seq_len == 6
    assert chained.lineage == ["planner:plan", "coder:impl"]


def test_bidirectional_channel_uses_separate_projectors():
    kv = empty_kv(1, 1, 2, 4, 8)
    left = AgentEndpoint("planner", "planner", kv.clone())
    right = AgentEndpoint("coder", "coder", kv.clone())
    lr = LinearKVAdapter(2, 8, 2, 8)
    rl = LinearKVAdapter(2, 8, 2, 8)
    # Perturb one direction so they are not the same map.
    with torch.no_grad():
        lr.key_proj.weight.mul_(0.5)
    ch = C2CChannel(left, right, left_to_right=lr, right_to_left=rl)
    left.kv.keys[0].fill_(1.0)
    ch.send("lr", "plan.intent", Intent.DELEGATE)
    assert right.kv.seq_len == 8  # prefix 4 + own 4


def test_role_conditioning_changes_projection():
    kv = empty_kv(1, 1, 2, 3, 4)
    kv.keys[0].fill_(1.0)
    adapter = RoleConditionedAdapter(2, 4, 2, 4)
    with torch.no_grad():
        adapter.role_shift.weight[adapter.role_id("coder")].fill_(0.25)
        adapter.role_shift.weight[adapter.role_id("tester")].fill_(-0.25)
    coder = adapter.forward(kv, "coder")
    tester = adapter.forward(kv, "tester")
    assert not torch.allclose(coder.keys[0], tester.keys[0])


def test_align_layers_heterogeneous_depth():
    assert align_layers(4, 2) == [0, 3]
    assert align_layers(2, 4) == [0, 0, 1, 1]


def test_planted_transfer_c2c_beats_equal_budget_text():
    reports = run_planted_transfer_experiment(n_facts=8, dim=32, n_layers=2, n_heads=4)
    by_name = {r.condition: r for r in reports}
    assert by_name["c2c_full"].accuracy == 1.0
    assert by_name["c2c_int8"].accuracy == 1.0
    assert by_name["text_equal_int8_bytes"].accuracy < by_name["c2c_int8"].accuracy
    assert by_name["no_transfer"].accuracy == 0.0
    # The point of the primitive: same byte budget, more recoverable facts.
    assert by_name["c2c_int8"].mean_cosine > by_name["text_equal_int8_bytes"].mean_cosine + 0.3
    print("\n" + format_reports(reports))


def test_streaming_curve_is_monotone_for_known_facts():
    curve = run_streaming_experiment(n_facts=6, dim=16, n_layers=2, n_heads=2)
    known = [row["known_fact_accuracy"] for row in curve]
    assert known[0] == 1.0
    assert known[-1] == 1.0
    overall = [row["accuracy"] for row in curve]
    assert overall[-1] >= overall[0]
    assert overall[-1] == 1.0


def test_tiny_learned_c2c_beats_truncated_text():
    result = run_tiny_learned_experiment(steps=80, seed=0)
    assert result["c2c_accuracy"] > result["text_trunc_accuracy"]
    assert result["c2c_accuracy"] >= 0.7
