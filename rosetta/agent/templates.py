"""Chat-template timelines that stay RoPE-safe.

HF chat templates are prefix-stable if we *append messages* and tokenize
with ``add_generation_prompt=False``. The generation prompt is a suffix
added only when the parent replies. Incremental ingest then prefills
only the new suffix (or crops to the common prefix if a template is
unstable).
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def tokenize_messages(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> List[int]:
    if not messages:
        return []
    ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and int(left[i]) == int(right[i]):
        i += 1
    return i


def env_message(kind: str, text: str, path: str = "") -> Dict[str, str]:
    label = "TEST LOG" if kind == "test" else "FILE"
    header = f"{label} {path}".strip() if path else label
    return {"role": "user", "content": f"{header}:\n{text}"}
