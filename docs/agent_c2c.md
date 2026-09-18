# Same-model agent C2C (production path)

User → agent is **text**. Agent → agent is **KV cache only**.

This is the path you can actually ship if **every agent is the same
self-hosted HuggingFace causal LM** (one process, one set of weights, many
KV timelines). It does not use the paper’s cross-model projector. Same
model means the child’s keys already live in the parent’s space.

```
user  --text-->  parent
parent.fork(explorer)          # copy-on-write DynamicCache
explorer.ingest_env(file)      # files/tests are environment, not peer chat
parent.adopt(explorer)         # parent timeline := explorer cache
parent.reply() --text--> user
```

## Why fork/join, not “concatenate two minds”

Llama/Qwen caches store **post-RoPE** keys. A token encoded at position 17
is useless if you splice it in as position 3. So:

* A capsule is always a **prefix of one timeline** (positions `0..N`).
* `fork` shares that prefix (copy-on-write). The child only **appends**.
* `adopt` replaces the parent timeline with the child’s. No branch-merge.
* Parallel children are allowed; you **adopt one of them**, you do not
  cat their caches.

Gold invariant (tested on the tiny Llama **and** SmolLM2-Instruct):

```
greedy(prefill(A+B)) == greedy(prefill(A); continue(B | past=A))
```

If that equality fails, do not ship.

## Chat templates

Instruct models go through `tokenizer.apply_chat_template`. Ingest **appends
a message** and tokenizes with `add_generation_prompt=False`, then prefills
only the new suffix. The assistant header is a suffix added only in
`reply()`. That keeps the token stream prefix-stable, which is what RoPE
requires.

Tiny demo tokenizers have no chat template; they keep explicit `<user>` /
`<file>` markers. Both paths share the same `fork` / `adopt` / capsule.

## Prefix cache (vLLM APC semantics)

This environment is CPU-only, so there is no vLLM GPU paged cache. The
engine still does **automatic prefix caching**: token blocks are hashed as
`(parent_block, token_chunk)` and a later prefill of the same prefix
reuses KV instead of re-forwarding it. `fork` is O(1) (shared
`DynamicCache` until either side appends).

## What is in the packet

`Capsule` has **no** `message` / `summary` / `text` field.

| Field | Why |
|---|---|
| `cache` | The payload. HuggingFace `DynamicCache`. |
| `token_ids` | Timeline metadata (length, repetition penalty). **Never re-encoded.** |
| `last_logits` | So the receiver can keep decoding without a dummy token. |
| `fingerprint` | `n_layers / heads / hidden / vocab` — reject foreign models. |
| `intent` | `fork` (copy) or `adopt` (join). |
| `extra.messages` | Chat-template transcript so adopt stays prefix-stable. |

Wire: `capsule.to_bytes(quantize="fp16"|"int8"|"none")`. In-process adopts
clone fp32/bf16 tensors; quantize is for moving the capsule to another
worker that loads **the same weights**.

## API

```python
from rosetta.agent import CodingRuntime, build_tiny_llama

engine, tok = build_tiny_llama()          # tests; swap for your model
rt = CodingRuntime(engine, tok)

rt.ingest_user("fix the auth cache bug in foo.py")
rt.fork("explorer", role="explore")
rt.ingest_env("explorer", open("foo.py").read(), kind="file")
rt.think("explorer", max_new_tokens=32)   # stays in explorer KV
rt.adopt("parent", "explorer")            # C2C join
print(rt.reply(max_new_tokens=128))
```

Wrap a real instruct model the same way:

```python
from rosetta.agent import CodingRuntime, load_instruct_engine

engine, tok = load_instruct_engine("HuggingFaceTB/SmolLM2-135M-Instruct")
rt = CodingRuntime(engine, tok)
rt.ingest_user("fix the auth cache bug in foo.py")
```

`load_instruct_engine` reads `C2C_INSTRUCT_MODEL` (default
`/tmp/models/SmolLM2-135M-Instruct`) when no path is passed.

## Ticket A/B vs text recap

`script/agent/run_ticket_ab.py` hides a unique constant in a file. Three
cells:

* **c2c** — parent forks explorer, explorer reads the file, parent adopts KV
* **gold** — parent reads the same file itself
* **recap** — parent only gets a short briefing that omits the constant

Metric: mean NLL of the hidden constant as the assistant continuation.
C2C should match gold and beat recap.

## What this is not

* Not a drop-in for OpenAI/Anthropic. Those APIs do not expose KV.
* Not cross-model C2C (Qwen planner, Llama coder). That needs a trained
  fuser and a different objective than the paper’s same-prompt fusion.
* Not a way to merge two agents that thought in parallel on diverged
  timelines. Adopt one child, or run children sequentially.
* Not vLLM itself. Prefix-block hashing here is the APC *idea* on
  HuggingFace `DynamicCache`.

## Run

```bash
python script/agent/run_transfer_experiment.py   # gold equality (tiny)
python script/agent/demo_latent_protocol.py      # parent/explorer/tester story
python script/agent/run_ticket_ab.py             # C2C vs recap (instruct)
python -m pytest test/test_agent_c2c.py test/test_agent_instruct.py -q -o addopts=
```
