# Agent-to-Agent Cache-to-Cache

The C2C paper teaches two models to share *the same prompt* by projecting
KV-cache tensors into one another. That is an ensemble method. Agents are
not an ensemble. They hold **different contexts**, **different roles**, and
**asynchronous clocks**. The interesting question is therefore not “can we
fuse two answers to MMLU” but:

> If agents no longer had to communicate through language, what
> fundamentally different forms of collaboration become possible?

This note answers that with a new primitive — **Semantic Capsules** on a
**Latent Bus** — implemented in `rosetta/agent/`. It is a communication
layer, not another agent framework.

## The gap the paper leaves open

Paper C2C:

```
prompt ──► sharer KV  ──projector──►  fused into receiver KV at the *same* index
prompt ──► receiver KV ─────────────►  generate
```

Agent communication today:

```
Agent A thinks  ──►  text / JSON / tool result  ──►  Agent B re-encodes the text
```

Two losses happen in the text hop that C2C was invented to avoid:

1. **Serialization loss.** A’s understanding of a 4k-token file is not the
   200-token summary it can afford to write. The discarded state is exactly
   the state C2C knows how to move.
2. **Re-encoding loss.** B tokenizes A’s words from scratch. Associations
   A had already formed (this function is the one the test is failing, this
   type is invariant under the refactor) are not in the string.

Same-prompt fusion does not fix this. A planner’s prompt is a goal. A
coder’s prompt is a diff. A tester’s prompt is a stack trace. Their token
indices do not align. **Agent C2C has to move understanding across
contexts**, not decorate a shared one.

That forces a different unit of communication.

## The primitive: Semantic Capsules

A capsule is a sliced, annotated KV region with an *intent*:

```
SemanticCapsule
  source, slot, intent, role, lineage
  kv: {keys, values} × selected layers × selected tokens × selected heads
```

`slot` is an address (`codebase.api`, `plan.constraints`, `test.failures`).
`intent` selects the overlay algebra (see below). Isolation is physical:
layers/heads/spans that were not extracted are not in the payload. An agent
can refuse to export early-layer lexical state or late-layer decision
state.

Two consumption modes, which is the actual fork from the paper:

| Mode | What moves | When |
|---|---|---|
| **Push / prefix** | Capsule tokens become *phantom context* in front of B’s own tokens | Handoff, delegate, stream |
| **Pull / Cache-RPC** | B sends a query vector; A returns top-k `(K,V)` | Consult, selective export |

Prefix injection is the cross-context primitive. B attends to documents it
never tokenized, in the geometry A already computed. Cache-RPC is the
bandwidth primitive: cost scales with B’s query, not with A’s context.

Intents and their algebra:

| Intent | Overlay | Meaning |
|---|---|---|
| `HANDOFF` | prefix / replace | Yield working memory. Successor continues mid-thought. |
| `DELEGATE` | prefix | Send a latent task spec. Sender keeps running. |
| `CONSULT` | remote top-k attention | Ask a question that has no words. |
| `FUSE` | parallel residual (paper multi-sharer) | Write into a shared slot. |
| `STREAM` | append along seq | Publish tokens while still thinking. |
| `CRITIQUE` | `recv += gate * (critic − recv)` | Disagreement field, not a review. |
| `SYNC` | bidirectional residual | Couple two working memories. |

```
Planner ──delegate(plan.intent)──► Coder ──handoff(impl.*)──► Tester
    │                                 ▲                         │
    │         consult / Cache-RPC      └──── critique residual ─┘
    ▼
Researcher ◄── selective_export(codebase.api)
```

The rest of this note is architectures that *only make sense* once that
primitive exists, plus how to try each one in this repo.

---

## 1. Latent prefix handoff (working-memory fork)

### Concept
When a process `exec`s, the child does not read a letter from the parent.
It inherits the address space. Agent handoff today is the letter. Capsule
handoff is the address space: B’s first generated token attends to A’s
unfinished residual stream.

### Communication mechanism
A publishes a `HANDOFF` capsule: chosen layers × the span of working memory
that should survive (not the chat template, not discarded alternatives).
B injects it as `past_key_values` *before* its own prompt. Those positions
have no tokens in B’s tokenizer. They are phantom tokens.

### Architecture
```
A.prefill(task) → KV_A
capsule = extract(KV_A, span=task_body, layers=middle_third)
B.generate(past = prefix(project(capsule)), prompt = B’s local view)
```
Optional copy-on-write: several children inherit the same capsule and
diverge. That is parallel reasoning without a shared transcript.

### Why C2C
A text briefing is a new document. It cannot carry A’s *partial*
activations — the direction A was about to go, the constraints A was
holding unnamed. KV can. That is the difference between “here is what I
concluded” and “here is my mind, continue”.

### Use case
A planner has walked a repository and formed an intent (“the leak is in
the cache invalidation path, don’t touch the parser”). It dies at a
context limit. The coder starts with that intent as prefix, not with a
markdown spec that forgot the parser warning.

### Prototype
`C2CProtocol.handoff` / `delegate` in `rosetta/agent/protocol.py`.
`bridge.apply_prefix_to_past` wires the same idea into `RosettaModel`’s
`past_key_values`. Use a trained C2C fuser as `projector` when geometries
differ; `LinearKVAdapter` when they do not yet have a fuser.

### Experiment
Planted-fact retrieval after handoff (`test_handoff_transfers_understanding_not_text`).
The receiver never saw the fact tokens and still attends to them. The
equal-byte experiment (`run_planted_transfer_experiment`) then asks
whether int8 capsules beat a text briefing of the same size. They do,
because the briefing has to *drop facts* to hit the budget and the
capsule only has to *quantize* them.

### Extension
True mid-token continuity: also transfer the last residual-stream vector
and the sampler’s RNG, so generation resumes inside a word. That is
process-level live migration of a thought.

---

## 2. Cache-RPC (remote KV attention)

### Concept
Do not copy A’s mind into B. Let B *query* it. Attention is already a
router. Expose A’s cache as an RPC endpoint:

```
B.Q  ──net──►  softmax(Q K_Aᵀ) V_A  ──net──►  B’s residual stream
```

Top-k makes it practical: only k columns return.

### Communication mechanism
`query: (B, H, Q, D)` on the wire. Response: `{indices, weights, V}` or,
cheaper, just the mixed values. No tokens, no summaries, no shared prompt.

### Architecture
```
Coder (generating `def invalidate(...):`)
    Q_t = current token query
    (Ṽ, idx) = researcher.cache_rpc(Q_t, topk=32, layer=mid)
    residual_t += W_o Ṽ
```
The researcher can keep a 128k-token reading of the repo. The coder’s
context window stays small. Relevance is computed, not declared.

### Why C2C
Text consult is: coder writes a question, researcher writes an answer,
coder re-reads it. Three serializations, and the coder had to know what
to ask. Cache-RPC asks with the query the coder was *already computing
to attend to its own past*. The question is implicit in the thought.

### Use case
While writing a patch, the coder’s attention naturally retrieves the
researcher’s encoding of the matching tests and the call sites, without
anyone prompting “please summarize the test file”.

### Prototype
`remote_retrieve` and `C2CProtocol.consult` / `selective_export`.
`relevance_token_mask` is the sender-side dual: A uses B’s query to decide
which of *its* tokens to publish, so even push-mode becomes query-conditioned.

### Experiment
`test_consult_is_topk_pull_not_full_copy` — query for fact *i* returns
token *i*. Next measurement on real models: freeze a researcher on a
corpus, freeze a coder, train only a tiny probe that maps coder Q into
researcher K-space, score retrieval of held-out file facts vs. a RAG
baseline that retrieves *text chunks*.

### Extension
**Distributed cross-attention** as a first-class layer: some heads attend
locally, some heads are RPC heads with a learned destination (researcher,
types, tests). Routing is inside the model, not inside an orchestrator.
This is MoE where the experts are *other agents’ caches*.

---

## 3. Continuous / streaming C2C

### Concept
Agents currently communicate in turns. A finishes, writes, B starts.
Streaming capsules make the bus a live wire: every new KV column A
commits is appended to a slot B is already attending to.

### Communication mechanism
`Intent.STREAM`. The bus concatenates along the sequence axis of the
latest capsule from that source. B can overlay each increment
(`STREAM_APPEND`) or replace its view of the slot with the grown capsule.

### Architecture
```
A: token 1 → publish → token 2 → publish → …
B: generate while attending to slot[0:t] as t grows
```
Coupled decoding is the synchronous variant: at each step both models
exchange columns, then sample. Debate without turns.

### Why C2C
You cannot stream a summary of a thought that has not been verbalized.
You *can* stream the cache column that thought is made of. B starts being
useful at t=3 of a 40-token analysis.

### Use case
A researcher is still reading file 7. The coder has already received
files 1–6 as latent prefix and is generating a scaffold. When file 7’s
capsule arrives, the coder’s later tokens see it. Pipeline parallelism
for cognition.

### Prototype
`C2CProtocol.stream`, `LatentBus.publish` STREAM-merge,
`run_streaming_experiment`. The paper’s `include_response=True` path in
`RosettaModel` is the same-prompt cousin (fuse during decode); streaming
capsules are the cross-context cousin.

### Experiment
`run_streaming_experiment`: known-fact accuracy is 1.0 from the first
committed token; all-fact accuracy rises monotonically and hits 1.0 when
the stream completes. That curve *is* the signature. On real models:
measure coder pass@1 vs. researcher tokens streamed, against a baseline
that waits for a full writeup.

### Extension
Speculative C2C: B generates under a stale capsule, A sends a patch of
KV deltas, B rolls back tokens whose attention mass sat on revised
columns. Like speculative decoding, but the draft is another agent.

---

## 4. Critique as a residual field

### Concept
A critic that writes “this function is wrong because X” forces the
generator to *read a review* and try to comply. A critic that can reach
the generator’s KV can write the *disagreement itself*:

```
KV_gen ← KV_gen + gate ⊙ (project(KV_critic) − KV_gen)
```

That is a vector field over the generator’s positions, not a document.

### Communication mechanism
`Intent.CRITIQUE`, `OverlayMode.CRITIQUE`. Gate can be scalar, per-token,
or per-head (the C2C fuser already knows how to produce those).

### Architecture
```
Generator ──KV──► Critic (reads as prefix, thinks)
Critic ──residual──► Generator (same token positions, or a reply prefix)
        optionally bidirectional SYNC
```
A debate is then two residual fields fighting under a learned gate, not
an N-turn transcript.

### Why C2C
Reviews are slow, coarse, and easy to ignore. A residual on the cache
column of the offending function is *local*. It does not require the
generator to understand the review, only to keep decoding under a
perturbed working memory.

### Use case
A tester’s failure encoding overlays the coder’s implementation span.
The next tokens the coder samples are generated in a cache that already
contains “this assertion is red”. Pair-programming without the pair
speaking.

### Prototype
`C2CProtocol.critique`, `test_critique_moves_generator_toward_critic`.
On `RosettaModel`, this is positional fusion with `add_self=True` and a
negative residual — a different training target than the paper’s
“combine wisdom on the same MCQ”.

### Experiment
Train a critic fuser so that overlaying it *increases* unit-test pass
rate of a frozen generator, vs. concatenating the critic’s text. Control:
shuffle the residual across token positions. If shuffled residuals still
help, you only added noise; if they don’t, the field is actually
addressing the right spans.

### Extension
**Coupled oscillators.** Two models decode in lockstep, each step
exchanging residuals. Stop when both KL(next-token) drop. The
“conversation” is a trajectory in KV space. There is no transcript
unless you decode one for the user at the end.

---

## 5. Shared latent workspace (blackboard, not chat)

### Concept
Stop routing messages A→B. Give the system a set of named slots that any
agent can publish into and fuse from. The workspace *is* the
collaboration. Text chat is a log of the workspace’s shadows.

### Communication mechanism
`LatentBus` slots hold lists of capsules. `fuse_slot` applies the paper’s
*parallel* multi-sharer rule: project every writer from a clean base,
sum residuals. `agreement` is mean pairwise cosine of pooled values.
Below a threshold the slot is in **latent merge conflict**.

### Architecture
```
Agent A ──┐
Agent B ──┼──►  slot "plan.constraints"  ──fuse──► Agent C
Agent C ──┘
```
Conflicts are first-class. Two planners that disagree in representation
space should not be averaged into a mush and silently executed. The
runtime can fork C into two children (one per capsule) or request a
`CONSULT` between the writers.

### Why C2C
A shared Google doc of summaries still serializes. A shared KV slot lets
C attend to A’s and B’s *unverbalized* constraints at once, with the
same fusion the paper trained for multi-sharer ensembles — except the
“sharers” are specialists that saw different evidence.

### Use case
Researcher publishes `codebase.api`, planner publishes `plan.constraints`,
tester publishes `test.failures`. The coder fuses those three slots as
prefix and generates. No orchestrator writes a mega-prompt.

### Prototype
`LatentBus`, `fuse_slot`, `conflicts`, `compose_lineage`.
`test_fuse_parallel_and_conflict_detection` plants two incompatible
fact-books in one slot and flags them.

### Experiment
Three tiny encoders see disjoint fact subsets. A decoder that fuses all
three slots should answer questions that require the union. A decoder
that gets three text summaries of equal byte budget should not, past a
capacity threshold. That is the workspace version of the planted-memory
test.

### Extension
Slot ACLs and *capability isolation*: the tester may subscribe to
`impl.public` but not `planner.discarded_alternatives`. Because isolation
is a slice of tensors, this is closer to an OS memory-protection model
than to prompt hygiene.

---

## 6. Hierarchical, role-conditioned delegation

### Concept
The same researcher capsule should not land in a coder the way it lands
in a tester. Coder needs call signatures; tester needs invariants. A
**role-conditioned projector** (FiLM on the C2C map) is a different
language for each listener, learned, not prompted.

Hierarchy is concatenation along sequence plus lineage metadata:
`[planner span | coder span]` arrives at the tester as one capsule whose
`lineage` records the chain. That is a latent stack trace of *how this
understanding was produced*.

### Communication mechanism
`RoleConditionedAdapter`: `projected' = (1+γ_role) ⊙ projected + β_role`.
`compose_lineage` for the chain. `DELEGATE` vs `HANDOFF` distinguishes
“I’m still here” from “I’m done”.

### Architecture
```
Researcher --role=coder--> Coder --role=tester--> Tester
                 \--role=planner--> Planner ─┘
```
One source cache, N cheap role heads, not N full fusers. Heterogeneous
models still need a geometry adapter (`LinearKVAdapter` or a trained
`C2CProjector`) *under* the FiLM.

### Why C2C
“Write a spec for the coder” and “write a spec for the tester” are two
lossy projections into English. Role-FiLM is a lossless-intent
projection into each agent’s KV geometry. The source does not have to
know how to explain; it has to know how to *aim*.

### Use case
A single researcher pass over a repo fans out, without rewriting, into
a planner’s constraint slot, a coder’s API slot, and a tester’s oracle
slot.

### Prototype
`RoleConditionedAdapter`, `compose_lineage`, `C2CChannel` (asymmetric
projectors per direction). `test_role_conditioning_changes_projection`,
`test_compose_lineage_hierarchical_handoff`.

### Experiment
Train two FiLM roles on the same encoder cache: role `coder` must
predict next-token in a code-completion head; role `tester` must predict
assertion polarity. A shared unconditioned projection should lose on
at least one head. That justifies the extra parameters.

### Extension
**Semantic multicast** over the network: publish once, project per
subscriber at the edge. Combined with Cache-RPC, subscribers don’t even
materialize the full capsule — they query a role-conditioned view.

---

## 7. Selective semantic transfer and compression

### Concept
Dumping a 128k cache at fp16 is not a protocol. The interesting move is
to treat transfer as *compression with a receiver in mind*:

* **Layer subset** — middle third (semantic), drop lexical and decision
  layers. `bridge.layer_subset_config`.
* **Head subset** — export heads whose attention entropy on the receiver
  query is low (they already focused).
* **Token subset** — `relevance_token_mask(query, kv)`.
* **Token pooling** — average-pool by k, a latent analog of summarizing.
* **Quantization** — int8/fp16 on the wire (`SemanticCapsule.to_bytes`).

### Communication mechanism
Bytes on the network are `capsule.to_bytes(quantize="int8")`. The
receiver `from_bytes` and overlays. For pull, the mask is computed on
A’s machine from B’s query, so unused columns never leave A.

### Architecture
```
A.KV --(query-conditioned topk / pool / int8)--> wire --> B.overlay
```
This is also how physically separate agents speak C2C: the bus becomes
a message queue of serialized capsules. No shared GPU, no shared
process, still not text.

### Why C2C
A 200-token summary of a 20k-token cache is a different, worse
compressor than int8 + top-k on the cache itself. The equal-byte
experiment exists to make that claim falsifiable.

### Use case
A cloud researcher and a local coder. The coder’s queries (a few KB)
fetch top-32 columns of the researcher’s cache. The alternative is
shipping the repo summary on every turn.

### Prototype
`to_bytes` / `from_bytes`, `pool_tokens`, `extract_*`,
`selective_export`. Planted experiment compares `c2c_int8` to
`text_equal_int8_bytes`.

### Experiment
The CI test: int8 capsule retrieval stays at accuracy 1.0; a text
briefing with the same payload budget drops most facts. Next: plot
accuracy vs. bits-per-token for `{int8, int4, pool×2, pool×4, text}`.
The interesting region is where C2C is still above text.

### Extension
Learned codec: a small autoencoder on KV columns trained to preserve
*the receiver’s* next-token KL, not reconstruction MSE. Compression
becomes communication-aware, which MSE on tensors is not.

---

## 8. Heterogeneous models, physically apart

### Concept
C2C already handles different depths, head counts, and tokenizers *when
the prompt is shared*. Agents add: different *prompts*, different
*machines*, different *roles*. The adapter stack is:

```
KV_A  →  geometry adapter (C2CProjector / LinearKVAdapter)
      →  role FiLM
      →  quantize
      →  network
      →  overlay mode (prefix | positional | residual | critique)
      →  KV_B
```

`C2CChannel` stores two of those stacks, because A→B is not B→A.

### Communication mechanism
The wire format is the capsule byte blob. Identity of a message is
`(source, slot, intent, lineage)`, not a chat id. Retransmission is
safe: overlay modes are functions of tensors, not of “did you read
this”.

### Architecture
Same as a trained fuser pair in this repo (`qwen3_0.6b+qwen2.5_0.5b_Fuser`)
plus prefix injection. The missing piece in the paper checkpoints is
the *prefix* path: they fuse at aligned indices. For agent use, train
(or freeze-and-adapt) the same projector to land in *new* positions —
i.e. when `target_kv` is zeros of length N_capsule, the fuser should
still produce a usable prefix. That is a different, smaller training
run than the paper’s, and it is the one this primitive needs.

### Why C2C
Tokenizer mismatch already makes text the wrong interlingua (see
`TokenAligner`). KV projection was built for that. Agents just need it
aimed at *new positions* and *new roles*, not at MMLU tokens.

### Use case
Qwen-coder locally, a larger researcher in the cluster, a math-specialist
sharer for a numeric kernel. The local coder never tokenizes the papers
the researcher read.

### Prototype
`LinearKVAdapter`, `align_layers`, `C2CChannel`, `bridge.py`. Plug a
trained `C2CProjector` in as `projector=` on `overlay` / `absorb`.

### Experiment
Take an existing fuser checkpoint. Condition A on documents, B on a
question those documents answer. Compare (i) paper-style same-prompt
fusion of `doc+question`, (ii) text summary of doc into B, (iii) prefix
injection of A’s *document-only* KV into B’s question. (iii) is the
agent setting. If (iii) ≈ (i) ≫ (ii), prefix C2C is doing the job.

### Extension
A directory of fusers is an **interlingua**: any agent that can project
into the directory’s hub geometry can talk to any other. That is a
network protocol whose packets are capsules and whose routing key is
`slot`.

---

## What this is *not*

Not a multi-agent runtime. There is no tool loop, no ReAct, no JSON
schema for “messages”. Those already exist and they all share the same
bottleneck. The bottleneck is the medium.

Not paper C2C with extra wrapping. Same-prompt positional fusion is one
overlay mode (`POSITIONAL` / `FUSE_PARALLEL`). The new modes — prefix,
stream, critique, consult, role FiLM, lineage — are the ones that
change what collaboration can be.

## Mapping onto this repository

| Piece | Where |
|---|---|
| Capsule, slice, quantize | `rosetta/agent/capsule.py` |
| Overlay algebras, adapters, Cache-RPC | `rosetta/agent/overlay.py` |
| Named slots, conflicts, fuse | `rosetta/agent/bus.py` |
| Protocol verbs, bidirectional channel | `rosetta/agent/protocol.py` |
| `RosettaModel` / DynamicCache | `rosetta/agent/bridge.py` |
| Falsifying experiments | `rosetta/agent/experiment.py` |
| Trained fusers (positional, same prompt) | `rosetta/model/projector.py`, `wrapper.py` |

The smallest command that can fail the thesis:

```bash
python script/agent/run_transfer_experiment.py
```

If `c2c_int8` does not beat `text_equal_int8_bytes` on planted retrieval,
capsules are a fancy way to send text and the rest of the architectures
collapse. If it does, the medium is actually different, and the
architectures above are worth training for.
