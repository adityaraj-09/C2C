#!/usr/bin/env python3
"""Ticket A/B: C2C adopt vs a lossy text recap on an instruct model.

Each ticket hides a unique fact in the file. C2C (child reads the file,
parent adopts KV) should assign lower NLL to that fact than a parent that
only received a recap omitting it. Gold (parent reads the file itself)
should match C2C.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rosetta.agent.instruct import instruct_model_available, load_instruct_engine
from rosetta.agent.tickets import default_tickets, score_ticket_modes


def main() -> None:
    if not instruct_model_available():
        print("instruct checkpoint missing; set C2C_INSTRUCT_MODEL")
        sys.exit(2)
    engine, tok = load_instruct_engine()
    print(f"model fingerprint: {engine.fingerprint}")
    print(f"{'ticket':<14} {'c2c':>8} {'gold':>8} {'recap':>8} {'c2c<recap':>10}")
    wins = 0
    for ticket in default_tickets():
        scores = score_ticket_modes(engine, tok, ticket)
        beat = scores["c2c"] < scores["recap"]
        wins += int(beat)
        print(
            f"{ticket.ticket_id:<14} {scores['c2c']:8.3f} {scores['gold']:8.3f} "
            f"{scores['recap']:8.3f} {str(beat):>10}"
        )
        if abs(scores["c2c"] - scores["gold"]) > 0.05:
            raise SystemExit(
                f"gold mismatch on {ticket.ticket_id}: c2c={scores['c2c']} gold={scores['gold']}"
            )
    print(f"c2c beats recap on {wins}/{len(default_tickets())} tickets")
    if wins < len(default_tickets()):
        raise SystemExit("C2C did not beat text recap on every ticket")
    print("ok")


if __name__ == "__main__":
    main()
