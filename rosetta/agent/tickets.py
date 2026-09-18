"""Synthetic coding tickets for C2C vs text-recap A/B.

Each ticket hides a unique fact in the file that a short recap omits.
C2C should assign higher log-prob to that fact than a lossy briefing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    user: str
    path: str
    file: str
    recap: str
    secret: str
    ask: str


def default_tickets() -> List[Ticket]:
    return [
        Ticket(
            ticket_id="auth-ttl",
            user="Fix the auth cache bug in foo.py. After you inspect the file, name the TOKEN constant value.",
            path="foo.py",
            file=(
                "class AuthCache:\n"
                "    TOKEN = 'BUGTOKEN_7f3a'\n"
                "    def get(self, key):\n"
                "        return self.store.get(key)\n"
                "    def set(self, key, value):\n"
                "        self.store[key] = value  # misses ttl expiry\n"
            ),
            recap="foo.py defines AuthCache with get/set; set() writes to store but TTL expiry is missing.",
            secret="BUGTOKEN_7f3a",
            ask="The TOKEN constant equals",
        ),
        Ticket(
            ticket_id="retry-limit",
            user="The HTTP client retries too often. Find MAX_RETRIES in client.py.",
            path="client.py",
            file=(
                "class HttpClient:\n"
                "    MAX_RETRIES = 17\n"
                "    def send(self, req):\n"
                "        for _ in range(self.MAX_RETRIES):\n"
                "            if self._try(req): return\n"
            ),
            recap="client.py has an HttpClient.send loop that retries failed requests.",
            secret="17",
            ask="MAX_RETRIES is",
        ),
        Ticket(
            ticket_id="port-bind",
            user="The server binds the wrong port. What is LISTEN_PORT in server.py?",
            path="server.py",
            file=(
                "LISTEN_PORT = 48123\n"
                "def main():\n"
                "    bind('0.0.0.0', LISTEN_PORT)\n"
                "    serve_forever()\n"
            ),
            recap="server.py binds on all interfaces and serves forever.",
            secret="48123",
            ask="LISTEN_PORT is",
        ),
        Ticket(
            ticket_id="hash-salt",
            user="Password hashing uses a hardcoded salt. What is SALT in crypto.py?",
            path="crypto.py",
            file=(
                "SALT = 's3edH4shX9'\n"
                "def hash_pw(pw):\n"
                "    return sha256(SALT + pw)\n"
            ),
            recap="crypto.py hashes passwords with sha256 and a constant salt (value not quoted here).",
            secret="s3edH4shX9",
            ask="SALT is",
        ),
        Ticket(
            ticket_id="queue-name",
            user="Jobs go to the wrong queue. What is QUEUE_NAME in worker.py?",
            path="worker.py",
            file=(
                "QUEUE_NAME = 'jobs.prio.omega'\n"
                "def enqueue(job):\n"
                "    redis.lpush(QUEUE_NAME, job)\n"
            ),
            recap="worker.py pushes jobs into Redis via lpush.",
            secret="jobs.prio.omega",
            ask="QUEUE_NAME is",
        ),
        Ticket(
            ticket_id="feature-flag",
            user="A feature flag is stuck. What is FLAG_ID in flags.py?",
            path="flags.py",
            file=(
                "FLAG_ID = 'ff_cache_v4_rollout'\n"
                "def enabled(user):\n"
                "    return user.id % 2 == 0\n"
            ),
            recap="flags.py enables a cache rollout flag for even user ids.",
            secret="ff_cache_v4_rollout",
            ask="FLAG_ID is",
        ),
    ]


def score_ticket(engine, tokenizer, ticket: Ticket, mode: str) -> float:
    """NLL of ``ticket.secret`` under a fresh runtime built for ``mode``."""
    from rosetta.agent.runtime import CodingRuntime

    rt = CodingRuntime(engine, tokenizer)
    rt.apply_ticket(ticket, mode)
    return rt.score_nll("parent", ticket.ask, ticket.secret)


def score_ticket_modes(engine, tokenizer, ticket: Ticket) -> dict:
    return {
        mode: score_ticket(engine, tokenizer, ticket, mode)
        for mode in ("c2c", "gold", "recap")
    }
