#!/usr/bin/env python3
"""Gold check: inherited-KV greedy tokens == full-prefill greedy tokens."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rosetta.agent import CodingRuntime, build_tiny_llama


def main() -> None:
    engine, tok = build_tiny_llama(seed=0)
    torch.manual_seed(0)

    a = torch.randint(12, 50, (1, 11), device=engine.device)
    b = torch.randint(12, 50, (1, 7), device=engine.device)
    gold = engine.prefill(torch.cat([a, b], dim=1))
    g_ids, _, _ = engine.greedy_continue(gold.past, gold.logits, max_new_tokens=8)
    first = engine.prefill(a)
    c_ids, _, _ = engine.greedy_continue(first.past, first.logits, max_new_tokens=8, extra_ids=b)
    assert torch.equal(g_ids, c_ids), (g_ids, c_ids)
    print("engine gold: inherited KV matches full prefill")

    rt = CodingRuntime(engine, tok)
    rt.ingest_user("fix the auth cache bug in foo.py")
    rt.fork("explorer")
    rt.ingest_env("explorer", "class authcache keys on write misses ttl")
    rt.adopt("parent", "explorer")
    ans = rt.reply(6)

    gold_rt = CodingRuntime(engine, tok)
    gold_rt.ingest_user("fix the auth cache bug in foo.py")
    gold_rt.ingest_env("parent", "class authcache keys on write misses ttl")
    gold_ans = gold_rt.reply(6)
    assert ans == gold_ans, (ans, gold_ans)
    print("runtime gold: adopt(explorer) matches parent ingest of the same file")
    print("ok")


if __name__ == "__main__":
    main()
