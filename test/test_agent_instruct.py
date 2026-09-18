"""Gold + ticket A/B tests on a real instruct checkpoint.

Skipped automatically when ``C2C_INSTRUCT_MODEL`` (default
``/tmp/models/SmolLM2-135M-Instruct``) is not present.
"""

from __future__ import annotations

import pytest
import torch

from rosetta.agent import (
    CodingRuntime,
    instruct_model_available,
    load_instruct_engine,
    score_ticket_modes,
)
from rosetta.agent.templates import env_message, tokenize_messages
from rosetta.agent.tickets import default_tickets

pytestmark = pytest.mark.skipif(
    not instruct_model_available(),
    reason="instruct checkpoint not present (set C2C_INSTRUCT_MODEL)",
)


def _engine():
    return load_instruct_engine()


def test_instruct_chat_template_is_prefix_stable():
    _, tok = _engine()
    first = [{"role": "user", "content": "fix the auth cache bug in foo.py"}]
    second = first + [env_message("file", "TOKEN = 'BUGTOKEN_7f3a'", path="foo.py")]
    a = tokenize_messages(tok, first)
    b = tokenize_messages(tok, second)
    g = tokenize_messages(tok, first, add_generation_prompt=True)
    assert b[: len(a)] == a
    assert g[: len(a)] == a
    assert len(g) > len(a)


def test_instruct_prefill_continue_matches_full_prefill():
    engine, tok = _engine()
    first = [{"role": "user", "content": "fix the auth cache bug in foo.py"}]
    second = first + [env_message("file", "TOKEN = 'BUGTOKEN_7f3a'", path="foo.py")]
    a = tokenize_messages(tok, first)
    b = tokenize_messages(tok, second)
    suffix = b[len(a) :]
    a_t = torch.tensor(a, dtype=torch.long, device=engine.device).view(1, -1)
    s_t = torch.tensor(suffix, dtype=torch.long, device=engine.device).view(1, -1)
    full_t = torch.tensor(b, dtype=torch.long, device=engine.device).view(1, -1)
    gold = engine.prefill(full_t)
    cont = engine.prefill(s_t, past=engine.prefill(a_t).past)
    assert torch.equal(gold.logits.argmax(-1), cont.logits.argmax(-1))
    assert torch.allclose(gold.logits.float(), cont.logits.float(), atol=2e-4, rtol=2e-4)


def test_instruct_adopt_matches_parent_ingesting_the_same_file():
    engine, tok = _engine()
    body = "class AuthCache:\n    TOKEN = 'BUGTOKEN_7f3a'\n"
    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the auth cache bug in foo.py")
    rt.fork("explorer", role="explore")
    rt.ingest_env("explorer", body, kind="file", path="foo.py")
    rt.adopt("parent", "explorer")

    gold = CodingRuntime(engine, tok)
    gold.ingest_user("fix the auth cache bug in foo.py")
    gold.ingest_env("parent", body, kind="file", path="foo.py")
    assert torch.equal(rt.agents["parent"].token_ids, gold.agents["parent"].token_ids)
    assert rt.agents["parent"].messages == gold.agents["parent"].messages
    assert rt.reply(max_new_tokens=8) == gold.reply(max_new_tokens=8)


def test_ticket_ab_c2c_beats_text_recap():
    engine, tok = _engine()
    tickets = [t for t in default_tickets() if t.ticket_id in ("auth-ttl", "hash-salt", "feature-flag")]
    c2c_nll, gold_nll, recap_nll = [], [], []
    for ticket in tickets:
        scores = score_ticket_modes(engine, tok, ticket)
        c2c_nll.append(scores["c2c"])
        gold_nll.append(scores["gold"])
        recap_nll.append(scores["recap"])
        assert scores["c2c"] == pytest.approx(scores["gold"], rel=5e-3, abs=5e-3)
        assert scores["c2c"] < scores["recap"]
    assert sum(c2c_nll) / len(c2c_nll) < sum(recap_nll) / len(recap_nll)
    assert abs(sum(c2c_nll) - sum(gold_nll)) < 0.05 * max(1.0, abs(sum(gold_nll)))
