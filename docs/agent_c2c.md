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
* `fork` copies that prefix. The child only **appends** (files, think tokens).
* `adopt` replaces the parent timeline with the child’s. No branch-merge.
* Parallel children are allowed; you **adopt one of them**, you do not
  cat their caches.

Gold invariant (tested):

```
greedy(prefill(A+B)) == greedy(prefill(A); continue(B | past=A))
```

If that equality fails, do not ship.

## What is in the packet

`Capsule` has **no** `message` / `summary` / `text` field.

| Field | Why |
|---|---|
| `cache` | The payload. HuggingFace `DynamicCache`. |
| `token_ids` | Timeline metadata (length, repetition penalty). **Never re-encoded.** |
| `last_logits` | So the receiver can keep decoding without a dummy token. |
| `fingerprint` | `n_layers / heads / hidden / vocab` — reject foreign models. |
| `intent` | `fork` (copy) or `adopt` (join). |

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

Wrap a real model the same way:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rosetta.agent import SharedCausalEngine, CodingRuntime

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
engine = SharedCausalEngine(mdl, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
rt = CodingRuntime(engine, tok)
```

Your tokenizer needs `encode` / `decode`. Chat-template tokenizers work if
`encode(text)` returns a list of ids. Role markers (`<user>`, `<file>`, …)
are optional; the tiny demo tokenizer has them.

## What this is not

* Not a drop-in for OpenAI/Anthropic. Those APIs do not expose KV.
* Not cross-model C2C (Qwen planner, Llama coder). That needs a trained
  fuser and a different objective than the paper’s same-prompt fusion.
* Not a way to merge two agents that thought in parallel on diverged
  timelines. Adopt one child, or run children sequentially.

## Run

```bash
python script/agent/run_transfer_experiment.py   # gold equality
python script/agent/demo_latent_protocol.py      # parent/explorer/tester story
python -m pytest test/test_agent_c2c.py -q -o addopts=
```
