#!/usr/bin/env python3
"""Parent/explorer/coder on one tiny Llama. User speaks text; agents speak KV."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rosetta.agent import CodingRuntime, build_tiny_llama


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    engine, tok = build_tiny_llama(seed=0)
    rt = CodingRuntime(engine, tok, max_seq_len=256)

    banner("User → parent (text)")
    user = "fix the auth cache bug in foo.py"
    rt.ingest_user(user)
    print(f"  user: {user!r}")
    print(f"  parent seq_len={rt.agents['parent'].seq_len}")

    banner("Parent forks explorer (C2C copy-on-write, no prompt to the child)")
    rt.fork("explorer", role="explore")
    print(f"  snapshot {rt.snapshot()}")

    banner("Environment → explorer (file bytes, still not agent-to-agent)")
    file_body = "class authcache keys on write misses ttl"
    rt.ingest_env("explorer", file_body, kind="file")
    print(f"  ingested file: {file_body!r}")
    print(f"  explorer seq_len={rt.agents['explorer'].seq_len}  parent still {rt.agents['parent'].seq_len}")

    banner("Explorer → parent (ADOPT capsule, no summary string)")
    cap = rt.export("explorer")
    print(f"  capsule fields: source={cap.source} intent={cap.intent.value} seq={cap.seq_len}")
    print(f"  has text message field: {hasattr(cap, 'message')}")
    wire = cap.to_bytes(quantize="fp16")
    print(f"  fp16 wire bytes={len(wire)}")
    rt.adopt("parent", cap)
    print(f"  after adopt parent seq_len={rt.agents['parent'].seq_len}")

    banner("Optional: fork coder from the joined parent, ingest a failing test")
    rt.fork("tester", role="tester")
    rt.ingest_env("tester", "test fails assert stale entry", kind="test")
    rt.adopt("parent", "tester")
    print(f"  snapshot {rt.snapshot()}")

    banner("Parent → user (text)")
    answer = rt.reply(max_new_tokens=12)
    print(f"  reply tokens decoded: {answer!r}")
    print("  (tiny random weights — decode is a sanity print, not a real fix)")


if __name__ == "__main__":
    main()
