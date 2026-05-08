"""
Tool-result compaction for the ReAct loop.

When accumulated tool-message content exceeds a threshold, clip the bodies of
older tool messages to a small cap. The most recent K messages are kept
verbatim so the model can still reason over fresh evidence.

No LLM call — pure token recovery. Borrowed from deepagents' `_truncate_args`
pattern (langchain-ai/deepagents middleware/summarization.py).

OpenAI tool-call validity is preserved: we only mutate the `content` of `tool`
messages, never the assistant `tool_calls` array, so every tool_call_id still
has a matching tool response.
"""
from typing import Any, Dict, List, Tuple


DEFAULT_TRIGGER_CHARS = 50_000
DEFAULT_KEEP_LAST_N = 6
DEFAULT_CAP_CHARS = 1_500


def compact_tool_results(
    messages: List[Dict[str, Any]],
    *,
    trigger_chars: int = DEFAULT_TRIGGER_CHARS,
    keep_last_n: int = DEFAULT_KEEP_LAST_N,
    cap_chars: int = DEFAULT_CAP_CHARS,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Clip content of older tool messages when total exceeds threshold.

    Returns (new_messages, stats).
    Stats keys: triggered, chars_before, chars_after, calls_compacted.
    """
    chars_before = sum(
        len(m.get("content") or "") for m in messages if m.get("role") == "tool"
    )

    if chars_before < trigger_chars:
        return messages, {
            "triggered": 0,
            "chars_before": chars_before,
            "chars_after": chars_before,
            "calls_compacted": 0,
        }

    cutoff = max(0, len(messages) - keep_last_n)
    new_messages: List[Dict[str, Any]] = list(messages)
    calls_compacted = 0
    chars_after = 0

    for i, m in enumerate(new_messages):
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        if i >= cutoff or len(content) <= cap_chars:
            chars_after += len(content)
            continue
        original_len = len(content)
        truncated = (
            content[:cap_chars]
            + f"\n…(compacted; was {original_len:,} chars, kept first {cap_chars:,})"
        )
        new_messages[i] = {**m, "content": truncated}
        calls_compacted += 1
        chars_after += len(truncated)

    return new_messages, {
        "triggered": 1,
        "chars_before": chars_before,
        "chars_after": chars_after,
        "calls_compacted": calls_compacted,
    }
