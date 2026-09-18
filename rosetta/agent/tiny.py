"""Tiny in-memory Llama + word tokenizer for tests and the local demo.

Weights are random. That is enough: same-model C2C is an *equality* of two
computations (full prefill vs inherited KV), not a quality benchmark.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from rosetta.agent.engine import SharedCausalEngine


SPECIAL = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<parent>",
    "<explore>",
    "<coder>",
    "<tester>",
    "<user>",
    "<assistant>",
    "<file>",
    "<test>",
    "<unk>",
]

# Small closed vocab so demos stay readable after encode/decode.
WORDS = [
    "fix", "the", "auth", "cache", "bug", "in", "foo.py", "bar.py",
    "function", "invalidates", "keys", "on", "write", "but", "misses",
    "ttl", "expiry", "test", "fails", "assert", "stale", "entry",
    "please", "read", "file", "and", "explain", "patch", "line",
    "user", "session", "token", "expired", "still", "served",
    "from", "memory", "store", "def", "return", "none", "if", "else",
    "class", "authcache", "get", "set", "delete", "now", "ok",
]


class WordTokenizer:
    """Whitespace tokenizer over a closed vocab plus specials."""

    def __init__(self):
        self.itos: List[str] = list(SPECIAL) + WORDS
        # pad vocab to a multiple of 16 for Llama
        while len(self.itos) % 16 != 0:
            self.itos.append(f"<extra_{len(self.itos)}>")
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_token_id = self.stoi["<pad>"]
        self.bos_token_id = self.stoi["<bos>"]
        self.eos_token_id = self.stoi["<eos>"]
        self.unk_token_id = self.stoi["<unk>"]
        self.vocab_size = len(self.itos)

    chat_template = None

    def encode(
        self,
        text: str,
        add_special: bool = False,
        add_special_tokens: bool | None = None,
        **_kwargs,
    ) -> List[int]:
        if add_special_tokens is not None:
            add_special = add_special_tokens
        pieces = text.replace("\n", " \n ").split()
        ids = []
        if add_special:
            ids.append(self.bos_token_id)
        for p in pieces:
            key = p.lower()
            ids.append(self.stoi.get(key, self.unk_token_id))
        return ids

    def decode(
        self,
        ids: Sequence[int],
        skip_special: bool = True,
        skip_special_tokens: bool | None = None,
        **_kwargs,
    ) -> str:
        if skip_special_tokens is not None:
            skip_special = skip_special_tokens
        special = set(range(len(SPECIAL)))
        words = []
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            if 0 <= i < len(self.itos):
                words.append(self.itos[i])
        return " ".join(words)


def build_tiny_llama(
    tokenizer: WordTokenizer | None = None,
    *,
    n_layers: int = 2,
    n_heads: int = 4,
    hidden: int = 32,
    seed: int = 0,
    block_size: int = 4,
) -> tuple[SharedCausalEngine, WordTokenizer]:
    tok = tokenizer or WordTokenizer()
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    model = LlamaForCausalLM(cfg)
    engine = SharedCausalEngine(
        model,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        block_size=block_size,
    )
    return engine, tok
