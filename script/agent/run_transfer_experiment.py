#!/usr/bin/env python3
"""Run the planted-memory + streaming + tiny-learned C2C experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rosetta.agent.experiment import (
    format_reports,
    run_planted_transfer_experiment,
    run_streaming_experiment,
    run_tiny_learned_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-steps", type=int, default=120)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    reports = run_planted_transfer_experiment()
    print("Planted content-addressable memory")
    print(format_reports(reports))
    print()

    curve = run_streaming_experiment()
    print("Streaming understanding (retrieval vs tokens received)")
    for row in curve:
        print(f"  t={row['tokens_streamed']:<2}  known={row['known_fact_accuracy']:.2f}  "
              f"all={row['accuracy']:.2f}")
    print()

    learned = run_tiny_learned_experiment(steps=args.learned_steps)
    print("Tiny learned encoder/decoder")
    for k, v in learned.items():
        print(f"  {k}: {v:.3f}")

    if args.json_out:
        payload = {
            "planted": [r.__dict__ for r in reports],
            "streaming": curve,
            "learned": learned,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
