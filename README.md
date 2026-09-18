<div align="center">
  <img src="resource/logo.png" alt="C2C" width="100"/>

  <h1>Agent C2C</h1>
  <h3>Parent and subagents talk in KV cache, not in recaps</h3>
</div>

This repo is a **same-model agent runtime**. The user speaks **text**. Agents speak **KV cache**. One self-hosted HuggingFace causal LM, many timelines, join by adopting a child’s cache.

That is a different product from the original [Cache-to-Cache paper](https://arxiv.org/abs/2510.03215) (cross-model projectors / fusers). Same idea — skip text between models — but **no projector, no teacher/receiver pair, no trained fuser**. Every agent is the same weights. The child’s keys already live in the parent’s space.

```
user  --text-->  parent
parent.fork(explorer)          # copy-on-write KV
explorer.ingest_env(file)      # files/tests are environment, not peer chat
parent.adopt(explorer)         # parent timeline := explorer cache
parent.reply() --text--> user
```

If you host the weights, you can ship this. Closed APIs (OpenAI, Anthropic, …) cannot: they do not expose KV.

## Why not a text recap

A subagent that read `foo.py` and then *summarized* it for the parent throws away the tokens that actually matter (the constant, the off-by-one, the flag name). C2C hands the parent the explorer’s **working memory**.

Gold rule, tested on a tiny Llama and on SmolLM2-Instruct:

```
greedy(prefill(A+B)) == greedy(prefill(A); continue(B | past=A))
```

`adopt(explorer)` must also match a parent that ingested the same file itself. If that equality fails, do not ship.

Llama/Qwen caches are **post-RoPE**. A capsule is always a prefix of **one** timeline. You `fork`, the child **appends**, you `adopt` one child. You do not concatenate two diverged minds.

## Install

Python 3.10+, PyTorch 2.6, transformers 4.52.

```bash
pip install -e .
```

Optional (original paper training/eval extras, not required for the agent runtime):

```bash
pip install -e ".[training,evaluation]"
```

## Quickstart

Tiny in-memory Llama (random weights — enough to prove the gold equality):

```python
from rosetta.agent import CodingRuntime, build_tiny_llama

engine, tok = build_tiny_llama()
rt = CodingRuntime(engine, tok)

rt.ingest_user("fix the auth cache bug in foo.py")
rt.fork("explorer", role="explore")
rt.ingest_env("explorer", open("foo.py").read(), kind="file", path="foo.py")
rt.think("explorer", max_new_tokens=32)   # stays in explorer KV
rt.adopt("parent", "explorer")            # C2C join — no summary string
print(rt.reply(max_new_tokens=128))
```

Real instruct model (chat template, prefix-stable ingest):

```python
from rosetta.agent import CodingRuntime, load_instruct_engine

engine, tok = load_instruct_engine()  # or a local / Hub path
rt = CodingRuntime(engine, tok)
rt.ingest_user("fix the auth cache bug in foo.py")
```

`load_instruct_engine` reads `C2C_INSTRUCT_MODEL` (default `/tmp/models/SmolLM2-135M-Instruct`).

More detail: [docs/agent_c2c.md](docs/agent_c2c.md).

## Copy-on-write and prefix cache

Both run. They are not alternatives.

| | Copy-on-write | Prefix cache |
|---|---|---|
| Job | Agent ↔ agent | Engine ↔ repeated prompt |
| What | `fork` shares the parent `DynamicCache` in O(1). First append clones. `adopt` replaces the parent timeline. | Hashed KV blocks (`parent_block`, `token_chunk`). Later prefill of the same prefix skips the forward. |
| When | Parent/explorer/tester handoff | Gold vs C2C cells, next ticket with the same system/user prefix |

This tree is CPU / HuggingFace. Prefix cache is **vLLM automatic-prefix-caching semantics**, not vLLM itself (no GPU paged KV here).

Typical ticket: `fork("explorer")` (CoW) → explorer reads the file (clone, then append) → `adopt` → a second runtime on the same user/file prefix can hit the prefix cache.

## What is in the packet

`Capsule` has **no** `message` / `summary` / `text` field. Agents never call each other with strings.

| Field | Why |
|---|---|
| `cache` | The payload. HuggingFace `DynamicCache`. |
| `token_ids` | Timeline metadata (length, repetition penalty). **Never re-encoded.** |
| `last_logits` | Receiver can keep decoding without a dummy token. |
| `fingerprint` | `n_layers / heads / hidden / vocab` — reject a foreign model. |
| `intent` | `fork` (copy) or `adopt` (join). |
| `extra.messages` | Chat-template transcript so adopt stays prefix-stable. |

Wire: `capsule.to_bytes(quantize="fp16"|"int8"|"none")`. In-process adopts clone live tensors. Quantize is for moving a capsule to another worker that loads **the same weights**.

Instruct ingest **appends a message** and tokenizes with `add_generation_prompt=False`, then prefills only the new suffix. `reply()` adds the assistant header as a suffix. That is what keeps RoPE aligned.

## Ticket A/B vs text recap

Each ticket hides a unique constant in a file. Three cells:

* **c2c** — parent forks explorer, explorer reads the file, parent adopts KV
* **gold** — parent reads the same file itself
* **recap** — parent only gets a short briefing that omits the constant

Metric: mean NLL of that constant as the assistant continuation. C2C should **match gold** and **beat recap**. On SmolLM2-Instruct this repo’s tickets do (6/6).

```bash
python script/agent/run_transfer_experiment.py   # gold equality (tiny)
python script/agent/demo_latent_protocol.py      # parent / explorer / tester
python script/agent/run_ticket_ab.py             # C2C vs recap (instruct)
python -m pytest test/test_agent_c2c.py test/test_agent_instruct.py -q -o addopts=
```

## What this is not

* Not a drop-in for OpenAI / Anthropic. Those APIs do not expose KV.
* Not the paper’s cross-model C2C (Qwen planner, Llama coder). That needs a trained fuser. The original code still lives under `rosetta/model/` and `script/train/`.
* Not a merge of two agents that thought in parallel on diverged timelines. Adopt **one** child, or run children sequentially.
* Not vLLM. Prefix-block hashing here is the APC *idea* on HuggingFace `DynamicCache`.

## Layout

```
rosetta/agent/     # product: runtime, capsule, CoW, prefix cache, chat templates
docs/agent_c2c.md  # longer notes
script/agent/      # gold check, demo, ticket A/B
test/              # tiny Llama + SmolLM2-Instruct
rosetta/model/     # original paper (projectors / RosettaModel) — not the agent path
```

## Paper (original C2C)

The research line this runtime takes a *same-model* slice from:

```bibtex
@article{fu2025c2c,
    title={Cache-to-Cache: Direct Semantic Communication Between Large Language Models},
    author={Tianyu Fu and Zihan Min and Hanling Zhang and Jichao Yan and Guohao Dai and Wanli Ouyang and Yu Wang},
    journal={arXiv preprint arXiv:2510.03215},
    year={2025},
}
```

[Project page](https://fuvty.github.io/C2C_Project_Page/) · [arXiv](https://arxiv.org/abs/2510.03215) · [Fuser weights](https://huggingface.co/nics-efc/C2C_Fuser)
