"""Same-model coding runtime: user speaks text, agents speak KV.

Lifecycle::

    user --text--> parent
    parent.fork("explorer")           # copy-on-write KV
    explorer.ingest_env(file_text)    # environment, not a peer message
    parent.adopt("explorer")          # C2C join: parent timeline := child
    parent.reply() --text--> user

Agents never call each other with strings. ``fork`` / ``export`` / ``adopt``
move ``Capsule`` objects (KV + timeline metadata).

Chat-template tokenizers append messages and prefill only the new suffix so
RoPE stays aligned. Tiny demo tokenizers keep explicit ``<user>`` / ``<file>``
markers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import torch
from torch import Tensor

from rosetta.agent.agent import Agent
from rosetta.agent.capsule import Capsule, Intent
from rosetta.agent.engine import SharedCausalEngine
from rosetta.agent.errors import AgentStateError, CapsuleError, ModelMismatchError
from rosetta.agent.kv import cache_nbytes, crop_prefix
from rosetta.agent.templates import (
    common_prefix_len,
    env_message,
    has_chat_template,
    tokenize_messages,
)

logger = logging.getLogger(__name__)

PARENT_ID = "parent"

ROLE_MARKERS = {
    "parent": "<parent>",
    "explore": "<explore>",
    "explorer": "<explore>",
    "coder": "<coder>",
    "tester": "<tester>",
}

ENV_MARKERS = {
    "file": "<file>",
    "test": "<test>",
}


class CodingRuntime:
    """In-process parent + subagents sharing one causal LM."""

    def __init__(
        self,
        engine: SharedCausalEngine,
        tokenizer,
        *,
        max_seq_len: int = 4096,
        max_capsule_bytes: int = 64 * 1024 * 1024,
        parent_id: str = PARENT_ID,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.max_capsule_bytes = int(max_capsule_bytes)
        self.parent_id = parent_id
        self.agents: Dict[str, Agent] = {
            parent_id: Agent(
                name=parent_id,
                role="parent",
                token_ids=torch.zeros(1, 0, dtype=torch.long, device=engine.device),
            )
        }

    @property
    def chat_template(self) -> bool:
        return has_chat_template(self.tokenizer)

    def _tensor_ids(self, ids: Sequence[int]) -> Tensor:
        if not ids:
            return torch.zeros(1, 0, dtype=torch.long, device=self.engine.device)
        return torch.tensor(list(ids), dtype=torch.long, device=self.engine.device).view(1, -1)

    def _tok(self, text: str) -> Tensor:
        encode = self.tokenizer.encode
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
        if not ids:
            raise AgentStateError("refusing to ingest empty text")
        if hasattr(ids, "ids"):
            ids = ids.ids
        return self._tensor_ids(ids)

    def _detok(self, ids) -> str:
        decode = self.tokenizer.decode
        try:
            return decode(ids, skip_special_tokens=True)
        except TypeError:
            return decode(ids)

    def _require(self, agent_id: str) -> Agent:
        if agent_id not in self.agents:
            raise AgentStateError(f"unknown agent {agent_id!r}")
        return self.agents[agent_id]

    def _append(self, agent: Agent, new_ids: Tensor) -> int:
        if new_ids.shape[1] == 0:
            raise AgentStateError("refusing to ingest empty text")
        if agent.seq_len + new_ids.shape[1] > self.max_seq_len:
            raise AgentStateError(
                f"{agent.name} would exceed max_seq_len={self.max_seq_len}"
            )
        agent.ensure_exclusive()
        if agent.cache is None:
            full = new_ids if agent.token_ids.numel() == 0 else torch.cat(
                [agent.token_ids, new_ids], dim=1
            )
            step = self.engine.prefill_from_empty(full)
        else:
            step = self.engine.prefill(new_ids, past=agent.cache, clone_past=False)
            full = torch.cat([agent.token_ids, new_ids], dim=1)
        agent.cache = step.past
        agent.last_logits = step.logits
        agent.token_ids = full
        return int(new_ids.shape[1])

    def _set_timeline(self, agent: Agent, token_ids: Tensor) -> int:
        """Make ``agent`` encode ``token_ids``, reusing the shared prefix KV."""
        if token_ids.shape[1] > self.max_seq_len:
            raise AgentStateError(
                f"{agent.name} would exceed max_seq_len={self.max_seq_len}"
            )
        old: List[int] = agent.token_ids[0].tolist() if agent.token_ids.numel() else []
        new: List[int] = token_ids[0].tolist()
        keep = common_prefix_len(old, new)
        if (
            keep == len(new)
            and keep == len(old)
            and agent.cache is not None
            and agent.last_logits is not None
        ):
            return 0
        if token_ids.shape[1] == 0:
            raise AgentStateError("refusing to ingest empty text")
        agent.ensure_exclusive()
        if keep == 0 or agent.cache is None:
            step = self.engine.prefill_from_empty(token_ids)
            agent.cache = step.past
            agent.last_logits = step.logits
            agent.token_ids = token_ids
            return int(token_ids.shape[1])
        if keep < len(old):
            agent.cache = crop_prefix(agent.cache, keep)
            agent.token_ids = agent.token_ids[:, :keep]
        suffix = token_ids[:, keep:]
        if suffix.shape[1] == 0:
            if keep <= 1:
                step = self.engine.prefill_from_empty(token_ids)
            else:
                cropped = crop_prefix(agent.cache, keep - 1)
                step = self.engine.prefill(
                    token_ids[:, -1:], past=cropped, clone_past=False
                )
            agent.cache = step.past
            agent.last_logits = step.logits
            agent.token_ids = token_ids
            return 0
        step = self.engine.prefill(suffix, past=agent.cache, clone_past=False)
        agent.cache = step.past
        agent.last_logits = step.logits
        agent.token_ids = token_ids
        return int(suffix.shape[1])

    def _ingest_chat(self, agent: Agent) -> int:
        ids = tokenize_messages(self.tokenizer, agent.messages)
        if not ids:
            raise AgentStateError("refusing to ingest empty text")
        return self._set_timeline(agent, self._tensor_ids(ids))

    # --- user / environment (text) -------------------------------------

    def ingest_user(self, text: str, agent_id: str = PARENT_ID) -> int:
        """User → agent. The only conversational text that enters the system."""
        if not text:
            raise AgentStateError("refusing to ingest empty text")
        agent = self._require(agent_id)
        agent.messages.append({"role": "user", "content": text})
        if self.chat_template:
            n = self._ingest_chat(agent)
            logger.debug("ingest_user %s += %s tokens (chat)", agent_id, n)
            return n
        marker = getattr(self.tokenizer, "stoi", {}).get("<user>")
        chunks = []
        if marker is not None:
            chunks.append(
                torch.tensor([[marker]], dtype=torch.long, device=self.engine.device)
            )
        chunks.append(self._tok(text))
        ids = torch.cat(chunks, dim=1)
        self._append(agent, ids)
        logger.debug("ingest_user %s += %s tokens", agent_id, ids.shape[1])
        return int(ids.shape[1])

    def ingest_env(
        self, agent_id: str, text: str, kind: str = "file", path: str = ""
    ) -> int:
        """Tool/environment → agent (file body, test log). Not agent-to-agent."""
        if not text:
            raise AgentStateError("refusing to ingest empty text")
        agent = self._require(agent_id)
        if self.chat_template:
            agent.messages.append(env_message(kind, text, path=path))
            n = self._ingest_chat(agent)
            logger.debug("ingest_env %s kind=%s += %s tokens (chat)", agent_id, kind, n)
            return n
        marker_name = ENV_MARKERS.get(kind, "<file>")
        marker = getattr(self.tokenizer, "stoi", {}).get(marker_name)
        chunks = []
        if marker is not None:
            chunks.append(
                torch.tensor([[marker]], dtype=torch.long, device=self.engine.device)
            )
        chunks.append(self._tok(text))
        ids = torch.cat(chunks, dim=1)
        self._append(agent, ids)
        logger.debug("ingest_env %s kind=%s += %s tokens", agent_id, kind, ids.shape[1])
        return int(ids.shape[1])

    def ingest_local(self, agent_id: str, text: str) -> int:
        """Local scaffold (role header, assistant prefix). Not a peer message."""
        agent = self._require(agent_id)
        if self.chat_template:
            agent.messages.append({"role": "user", "content": text})
            return self._ingest_chat(agent)
        ids = self._tok(text)
        self._append(agent, ids)
        return int(ids.shape[1])

    def mark_role(self, agent_id: str) -> int:
        """Append this agent's local role marker onto its timeline."""
        if self.chat_template:
            return 0
        agent = self._require(agent_id)
        name = ROLE_MARKERS.get(agent.role, "<parent>")
        marker = getattr(self.tokenizer, "stoi", {}).get(name)
        if marker is None:
            return 0
        ids = torch.tensor([[marker]], dtype=torch.long, device=self.engine.device)
        self._append(agent, ids)
        return 1

    # --- agent ↔ agent (C2C only) -------------------------------------

    def fork(self, child_id: str, *, parent_id: str = PARENT_ID, role: str = "explore") -> Agent:
        """Copy-on-write KV. Child inherits understanding; parent keeps its copy."""
        if child_id in self.agents:
            raise AgentStateError(f"agent {child_id!r} already exists")
        src = self._require(parent_id)
        child = src.share(child_id, role)
        self.agents[child_id] = child
        logger.debug("fork %s -> %s seq=%s cow=1", parent_id, child_id, child.seq_len)
        return child

    def export(self, agent_id: str, intent: Intent = Intent.ADOPT, slot: str = "") -> Capsule:
        agent = self._require(agent_id)
        cap = agent.export(intent, slot=slot or f"{agent_id}.memory", fingerprint=self.engine.fingerprint)
        nbytes = cache_nbytes(cap.cache)
        if nbytes > self.max_capsule_bytes:
            raise CapsuleError(f"capsule {nbytes} bytes exceeds max {self.max_capsule_bytes}")
        return cap

    def adopt(self, dst_id: str, src: str | Capsule) -> Capsule:
        """C2C join: ``dst`` timeline is replaced by ``src``'s cache."""
        dst = self._require(dst_id)
        cap = src if isinstance(src, Capsule) else self.export(src, intent=Intent.ADOPT)
        if cap.fingerprint != self.engine.fingerprint:
            raise ModelMismatchError("capsule is not from this engine's model")
        if cap.intent not in (Intent.ADOPT, Intent.FORK):
            raise AgentStateError(f"cannot adopt intent {cap.intent}")
        dst.install(cap)
        logger.debug("adopt %s <- %s seq=%s", dst_id, cap.source, dst.seq_len)
        return cap

    def think(self, agent_id: str, max_new_tokens: int = 8) -> int:
        """Internal continuation. New tokens stay in this agent's KV, not as a message."""
        agent = self._require(agent_id)
        if agent.cache is None or agent.last_logits is None:
            raise AgentStateError(f"{agent_id} has nothing to think about")
        agent.ensure_exclusive()
        new_ids, past, logits = self.engine.greedy_continue(
            agent.cache, agent.last_logits, max_new_tokens=max_new_tokens
        )
        agent.cache = past
        agent.last_logits = logits
        if new_ids.shape[1]:
            agent.token_ids = torch.cat([agent.token_ids, new_ids.to(agent.token_ids.device)], dim=1)
        return int(new_ids.shape[1])

    # --- agent → user (text) ------------------------------------------

    def reply(self, max_new_tokens: int = 16, agent_id: str = PARENT_ID) -> str:
        """Parent generates the user-visible answer from its (possibly adopted) KV."""
        agent = self._require(agent_id)
        if agent.cache is None or agent.last_logits is None:
            raise AgentStateError("parent has no state to reply from")
        agent.ensure_exclusive()
        extra = None
        if self.chat_template:
            with_gen = tokenize_messages(
                self.tokenizer, agent.messages, add_generation_prompt=True
            )
            current = agent.token_ids[0].tolist()
            extra_list = with_gen[common_prefix_len(current, with_gen) :]
            if extra_list:
                extra = self._tensor_ids(extra_list)
        else:
            marker = getattr(self.tokenizer, "stoi", {}).get("<assistant>")
            if marker is not None:
                extra = torch.tensor(
                    [[marker]], dtype=torch.long, device=self.engine.device
                )
        new_ids, past, logits = self.engine.greedy_continue(
            agent.cache, agent.last_logits, max_new_tokens=max_new_tokens, extra_ids=extra
        )
        agent.cache = past
        agent.last_logits = logits
        if extra is not None:
            agent.token_ids = torch.cat([agent.token_ids, extra], dim=1)
        if new_ids.shape[1]:
            agent.token_ids = torch.cat([agent.token_ids, new_ids], dim=1)
        text = self._detok(new_ids[0].tolist())
        if self.chat_template:
            agent.messages.append({"role": "assistant", "content": text})
        return text

    def apply_ticket(self, ticket, mode: str) -> None:
        """Build the parent timeline for a C2C / gold / recap A/B cell."""
        self.ingest_user(ticket.user)
        if mode == "c2c":
            self.fork("explorer", role="explore")
            self.ingest_env("explorer", ticket.file, kind="file", path=ticket.path)
            self.adopt("parent", "explorer")
        elif mode == "gold":
            self.ingest_env("parent", ticket.file, kind="file", path=ticket.path)
        elif mode == "recap":
            self.ingest_user(ticket.recap)
        else:
            raise AgentStateError(f"unknown ticket mode {mode!r}")

    def score_nll(self, agent_id: str, ask: str, secret: str) -> float:
        """Mean NLL of ``secret`` as the assistant continuation after ``ask``.

        Scoring clones the agent; the live timeline is not mutated.
        """
        agent = self._require(agent_id)
        if agent.cache is None or agent.last_logits is None:
            raise AgentStateError(f"{agent_id} has no state to score")
        work = agent.clone()
        if self.chat_template:
            ask_msg = {"role": "user", "content": ask}
            secret_msg = {"role": "assistant", "content": secret}
            ctx = tokenize_messages(
                self.tokenizer, work.messages + [ask_msg], add_generation_prompt=True
            )
            full = tokenize_messages(
                self.tokenizer,
                work.messages + [ask_msg, secret_msg],
                add_generation_prompt=False,
            )
            keep = common_prefix_len(ctx, full)
            target = full[keep:]
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None:
                trimmed: List[int] = []
                for tok in target:
                    if int(tok) == int(eos):
                        break
                    trimmed.append(int(tok))
                if trimmed:
                    target = trimmed
            if not target:
                raise AgentStateError("secret tokenized to an empty continuation")
            self._set_timeline(work, self._tensor_ids(ctx))
            return self.engine.mean_nll(work.cache, work.last_logits, self._tensor_ids(target))
        self._append(work, self._tok(ask))
        return self.engine.mean_nll(work.cache, work.last_logits, self._tok(secret))

    def snapshot(self) -> Dict[str, int]:
        return {name: ag.seq_len for name, ag in self.agents.items()}

    def prefix_stats(self) -> dict:
        return self.engine.prefix_cache.stats.as_dict()
