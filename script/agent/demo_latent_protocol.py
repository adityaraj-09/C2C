#!/usr/bin/env python3
"""Narrative demo: planner / researcher / coder / tester over C2C, no text briefings.

The 'codebase' is a planted fact book (stand-in for files a researcher read).
Capsules move across a LatentBus. A coder answers questions it never saw
text for. A tester critiques by writing a residual, not a review.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rosetta.agent import AgentEndpoint, C2CProtocol, empty_kv
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


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    banner("1. Researcher reads a codebase (understanding lives in KV)")
    book = make_factbook(n_facts=8, dim=32, seed=0)
    research_kv = plant_facts(book, n_layers=2, n_heads=4)
    researcher = AgentEndpoint("researcher", "researcher", research_kv)
    coder = AgentEndpoint("coder", "coder", empty_kv(2, 1, 4, 2, 32))
    tester = AgentEndpoint("tester", "tester", empty_kv(2, 1, 4, 2, 32))
    planner = AgentEndpoint("planner", "planner", empty_kv(2, 1, 4, 2, 32))
    proto = C2CProtocol()
    print(f"  facts={book.n}  researcher cache={researcher.kv.seq_len} tokens, "
          f"{researcher.kv.nbytes()} bytes")
    print("  coder has never tokenized those files.")

    banner("2. DELEGATE: planner does not write a spec — it hands a latent prefix")
    # Planner's 'intent' is a slice of what the researcher understood.
    proto.delegate(researcher, planner, slot="codebase.api")
    proto.delegate(planner, coder, slot="plan.intent")
    acc = retrieval_accuracy(retrieve_facts(coder.kv, book))
    print(f"  coder retrieval accuracy after latent delegate: {acc:.2f}")
    print("  no JSON task description was produced.")

    banner("3. CONSULT: coder queries researcher via Cache-RPC (top-1 attention)")
    query = book.keys[5].view(1, 1, 32)
    result = proto.consult(coder, researcher, query, topk=1)
    idx = int(result.indices.reshape(-1)[0])
    print(f"  asked about fact 5; remote attention returned token {idx}")
    print(f"  bytes moved: query + 1 KV column, not the full {researcher.kv.nbytes()}B cache")

    banner("4. HANDOFF: coder yields working memory to tester")
    proto.handoff(coder, tester, slot="impl.working_memory")
    acc_t = retrieval_accuracy(retrieve_facts(tester.kv, book))
    print(f"  tester retrieval accuracy after handoff: {acc_t:.2f}")
    print(f"  lineage slots on the bus: {proto.bus.slots()}")

    banner("5. CRITIQUE: tester writes a disagreement field, not a review")
    # Tester zeros a fact it 'rejects'; residual overlay nudges the coder.
    rejected = tester.kv.clone()
    rejected.keys[0][:, :, :8, :] *= 0.0
    tester.kv = rejected
    proto.critique(tester, coder, slot="test.failures", gate=0.4)
    print("  coder KV was shifted toward the tester residual.")
    print(f"  conflicts on bus: {proto.bus.conflicts()}")

    banner("6. Smallest falsifying experiment: equal-byte transfer")
    reports = run_planted_transfer_experiment()
    print(format_reports(reports))
    by = {r.condition: r for r in reports}
    delta = by["c2c_int8"].accuracy - by["text_equal_int8_bytes"].accuracy
    print(f"\n  C2C int8 accuracy − text@same-bytes accuracy = {delta:+.2f}")

    banner("7. Streaming: coder gets useful before researcher finishes")
    curve = run_streaming_experiment()
    print(f"  {'step':<6} {'known':>8} {'all':>8}")
    for row in curve:
        print(f"  {row['tokens_streamed']:<6} {row['known_fact_accuracy']:>8.2f} "
              f"{row['accuracy']:>8.2f}")

    banner("8. Tiny learned associative memory (CPU, ~few seconds)")
    learned = run_tiny_learned_experiment(steps=120, seed=0)
    print(f"  train_loss={learned['train_loss']:.3f}")
    print(f"  C2C prefix accuracy     : {learned['c2c_accuracy']:.2f}")
    print(f"  truncated-text accuracy : {learned['text_trunc_accuracy']:.2f}")

    banner("Done")
    print("Primitive: Semantic Capsules on a LatentBus, consumed by overlay")
    print("intents (handoff/delegate/consult/stream/critique/fuse/sync).")


if __name__ == "__main__":
    main()
