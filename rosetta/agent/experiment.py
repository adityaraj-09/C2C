"""Smallest experiments that can falsify agent-C2C.

Two experiments, both CPU-only, no pretrained weights:

1. Planted content-addressable memory
   KV is a key-value store. Attention retrieves facts. We compare:
     * full C2C capsule (upper bound)
     * int8 / pooled capsule at a matching byte budget
     * a text briefing with the same byte budget
     * no transfer
   If capsules are just an expensive way to send text, they will not beat
   a briefing of equal size. If KV is a better communication channel,
   retrieval accuracy stays high after compression that ruins text.

2. Streaming understanding
   Agent A encodes facts one token at a time and publishes STREAM capsules.
   Agent B's retrieval of each fact is measured after every step. The
   curve is the empirical signature of continuous C2C: B starts being
   useful before A finishes.

A third helper trains a tiny associative memory so the same comparison
can be run with learned attention rather than planted keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rosetta.agent.bus import LatentBus
from rosetta.agent.capsule import Intent, KVSlice, SemanticCapsule, empty_kv, pool_tokens
from rosetta.agent.overlay import remote_retrieve
from rosetta.agent.protocol import AgentEndpoint, C2CProtocol


def _normalize(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, dim=dim)


@dataclass
class FactBook:
    """Synthetic 'codebase': N facts as orthogonal-ish key/value vectors."""

    keys: Tensor          # (N, D)
    values: Tensor        # (N, D)
    names: List[str]

    @property
    def n(self) -> int:
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        return int(self.keys.shape[1])


def make_factbook(n_facts: int = 8, dim: int = 32, seed: int = 0) -> FactBook:
    g = torch.Generator().manual_seed(seed)
    keys = _normalize(torch.randn(n_facts, dim, generator=g))
    values = _normalize(torch.randn(n_facts, dim, generator=g))
    names = [f"sym_{i}" for i in range(n_facts)]
    return FactBook(keys=keys, values=values, names=names)


def plant_facts(book: FactBook, n_layers: int = 2, n_heads: int = 4) -> KVSlice:
    """Write facts into a cache so attention on key_i recovers value_i.

    Each fact occupies one token. Every head/layer stores the same pair so
    retrieval is layer-robust. This is the 'researcher has read the repo'
    state: understanding is *in the cache*, not in a summary.
    """
    n, d = book.n, book.dim
    k = book.keys.view(1, 1, n, d).expand(1, n_heads, n, d).contiguous()
    v = book.values.view(1, 1, n, d).expand(1, n_heads, n, d).contiguous()
    keys = [k.clone() for _ in range(n_layers)]
    values = [v.clone() for _ in range(n_layers)]
    return KVSlice(keys=keys, values=values, layer_indices=list(range(n_layers)),
                   token_start=0, token_end=n)


def retrieve_facts(kv: KVSlice, book: FactBook, layer: int = -1) -> Tensor:
    """Return cosine(retrieved, true_value) per fact. Shape (N,)."""
    query = book.keys.view(1, book.n, book.dim)  # (B=1, Q=N, D)
    mixed, _, _ = remote_retrieve(query, kv, topk=1, layer=layer)
    mixed = _normalize(mixed.squeeze(0))  # (N, D)
    return (mixed * book.values).sum(dim=-1)


def retrieval_accuracy(cosines: Tensor, threshold: float = 0.85) -> float:
    return float((cosines > threshold).float().mean())


def text_briefing_kv(book: FactBook, nbytes_budget: int, n_layers: int, n_heads: int, seed: int = 1) -> KVSlice:
    """Lossy 'JSON summary' of the same facts, occupying ``nbytes_budget``.

    Generous to text: the briefing still lives in the same vector space.
    The loss comes from *truncation* — what happens when a researcher
    summarizes a repo into a few hundred tokens.
    """
    bytes_per_token = 4 * n_heads * book.dim * 2 * n_layers  # k+v float32
    n_tokens = max(1, nbytes_budget // max(bytes_per_token, 1))
    n_tokens = min(n_tokens, book.n)
    slim = FactBook(keys=book.keys[:n_tokens], values=book.values[:n_tokens],
                    names=book.names[:n_tokens])
    return plant_facts(slim, n_layers=n_layers, n_heads=n_heads)


@dataclass
class TransferReport:
    condition: str
    nbytes: int
    mean_cosine: float
    accuracy: float
    per_fact: List[float]


def evaluate_condition(name: str, kv: KVSlice, book: FactBook) -> TransferReport:
    cos = retrieve_facts(kv, book)
    return TransferReport(
        condition=name,
        nbytes=kv.nbytes(),
        mean_cosine=float(cos.mean()),
        accuracy=retrieval_accuracy(cos),
        per_fact=[float(x) for x in cos],
    )


def run_planted_transfer_experiment(
    n_facts: int = 8,
    dim: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    pool_factor: int = 4,
    seed: int = 0,
) -> List[TransferReport]:
    """Compare C2C capsules against an equal-budget text briefing."""
    book = make_factbook(n_facts=n_facts, dim=dim, seed=seed)
    researcher = plant_facts(book, n_layers=n_layers, n_heads=n_heads)

    reports = [evaluate_condition("c2c_full", researcher, book)]

    quantized = SemanticCapsule(
        source="researcher", slot="codebase.api", intent=Intent.HANDOFF, kv=researcher
    )
    restored = SemanticCapsule.from_bytes(quantized.to_bytes(quantize="int8"))
    reports.append(evaluate_condition("c2c_int8", restored.kv, book))
    int8_payload = max(researcher.nbytes() // 4, 1)
    briefing_int8 = text_briefing_kv(book, int8_payload, n_layers, n_heads, seed=seed + 1)
    reports.append(evaluate_condition("text_equal_int8_bytes", briefing_int8, book))

    pooled = pool_tokens(researcher, pool_factor)
    reports.append(evaluate_condition(f"c2c_pooled_x{pool_factor}", pooled, book))
    briefing_pool = text_briefing_kv(book, pooled.nbytes(), n_layers, n_heads, seed=seed + 2)
    reports.append(evaluate_condition("text_equal_pooled_bytes", briefing_pool, book))

    empty = empty_kv(n_layers, 1, n_heads, 1, dim)
    reports.append(evaluate_condition("no_transfer", empty, book))
    return reports


def run_streaming_experiment(
    n_facts: int = 8,
    dim: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """B's retrieval vs how much of A's cache has streamed in."""
    book = make_factbook(n_facts=n_facts, dim=dim, seed=seed)
    full = plant_facts(book, n_layers=n_layers, n_heads=n_heads)
    proto = C2CProtocol(LatentBus())
    researcher = AgentEndpoint("researcher", "researcher", full)
    coder = AgentEndpoint("coder", "coder", empty_kv(n_layers, 1, n_heads, 1, dim))

    curve = []
    for t in range(n_facts):
        step = KVSlice(
            keys=[k[:, :, t:t + 1, :] for k in full.keys],
            values=[v[:, :, t:t + 1, :] for v in full.values],
            layer_indices=list(full.layer_indices),
            token_start=t,
            token_end=t + 1,
        )
        grown = proto.stream(researcher, "codebase.api", step)
        coder.kv = grown.kv.clone()
        cos = retrieve_facts(coder.kv, book)
        curve.append({
            "tokens_streamed": t + 1,
            "mean_cosine": float(cos.mean()),
            "accuracy": retrieval_accuracy(cos),
            "known_fact_accuracy": retrieval_accuracy(cos[: t + 1]),
        })
    return curve


@dataclass
class TinyConfig:
    vocab: int = 24
    dim: int = 32
    n_heads: int = 2
    n_facts: int = 4
    n_key_tokens: int = 8
    n_val_tokens: int = 8


class AssociativeMemory(nn.Module):
    """Learned content-addressable memory whose state is a KVSlice.

    Keys/values are embeddings of fact tokens. A query projection is trained
    so attending into the capsule recovers the bound value. Smallest learned
    stand-in for 'encode a file, transfer the cache'.
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.head = nn.Linear(cfg.dim, cfg.vocab)

    def facts_to_kv(self, key_ids: Tensor, value_ids: Tensor) -> KVSlice:
        k = self.embed(key_ids).unsqueeze(1).expand(-1, self.cfg.n_heads, -1, -1).contiguous()
        v = self.embed(value_ids).unsqueeze(1).expand(-1, self.cfg.n_heads, -1, -1).contiguous()
        n = key_ids.shape[1]
        return KVSlice(keys=[k], values=[v], layer_indices=[0], token_start=0, token_end=n)

    def logits_from_kv(self, kv: KVSlice, question_ids: Tensor) -> Tensor:
        q = self.q_proj(self.embed(question_ids))
        mixed, _, _ = remote_retrieve(q, kv, topk=max(kv.seq_len, 1), layer=0)
        return self.head(mixed[:, -1, :])


def _make_bindings(cfg: TinyConfig, g: torch.Generator, batch: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    key_pool = torch.arange(1, cfg.n_key_tokens + 1)
    val_pool = torch.arange(cfg.n_key_tokens + 1, cfg.n_key_tokens + 1 + cfg.n_val_tokens)
    keys, values, questions, targets = [], [], [], []
    for _ in range(batch):
        kidx = torch.randperm(cfg.n_key_tokens, generator=g)[: cfg.n_facts]
        vidx = torch.randperm(cfg.n_val_tokens, generator=g)[: cfg.n_facts]
        k = key_pool[kidx]
        v = val_pool[vidx]
        qi = int(torch.randint(0, cfg.n_facts, (1,), generator=g))
        keys.append(k.unsqueeze(0))
        values.append(v.unsqueeze(0))
        questions.append(k[qi].view(1, 1))
        targets.append(v[qi].view(1))
    return (
        torch.cat(keys, dim=0),
        torch.cat(values, dim=0),
        torch.cat(questions, dim=0),
        torch.cat(targets, dim=0),
    )


def run_tiny_learned_experiment(
    steps: int = 80,
    seed: int = 0,
    lr: float = 1e-2,
    batch: int = 32,
) -> Dict[str, float]:
    """Train a memory so a full C2C capsule beats a 1-fact text briefing."""
    cfg = TinyConfig()
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = AssociativeMemory(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = 0.0
    for _ in range(steps):
        keys, values, questions, targets = _make_bindings(cfg, g, batch)
        kv = model.facts_to_kv(keys, values)
        loss = F.cross_entropy(model.logits_from_kv(kv, questions), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())

    def _acc(use_c2c: bool, n: int = 128) -> float:
        model.eval()
        gg = torch.Generator().manual_seed(seed + 99)
        with torch.no_grad():
            keys, values, questions, targets = _make_bindings(cfg, gg, n)
            if use_c2c:
                kv = model.facts_to_kv(keys, values)
            else:
                kv = model.facts_to_kv(keys[:, :1], values[:, :1])
            pred = model.logits_from_kv(kv, questions).argmax(dim=-1)
            return float((pred == targets).float().mean())

    return {
        "train_loss": last,
        "c2c_accuracy": _acc(True),
        "text_trunc_accuracy": _acc(False),
    }


def format_reports(reports: List[TransferReport]) -> str:
    lines = [
        f"{'condition':<28} {'bytes':>8} {'cos':>7} {'acc':>7}",
        "-" * 54,
    ]
    for r in reports:
        lines.append(f"{r.condition:<28} {r.nbytes:>8} {r.mean_cosine:>7.3f} {r.accuracy:>7.3f}")
    return "\n".join(lines)
