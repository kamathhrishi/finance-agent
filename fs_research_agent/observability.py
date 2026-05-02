"""
Logfire integration for fs_research_agent.

Lazy import + no-op fallback so the package keeps working when logfire isn't
installed/configured. Mirrors the pattern already used in `agent/rag/`.

Usage:
    from .observability import logfire, LOGFIRE_AVAILABLE, span

    # Spans (use as context manager):
    with span("fs_research.tool_call", name=tool, args=args) as s:
        result = run_tool()
        if s is not None:
            s.set_attribute("result_chars", len(result))

    # Point events:
    if LOGFIRE_AVAILABLE and logfire:
        logfire.info("fs_research.flow.complete", elapsed_ms=int(elapsed * 1000))

OpenAI calls are already instrumented at app startup via
`logfire.instrument_openai(capture_all=True)`, so each LLM round-trip
produces a span automatically with model, prompts, completions, and tokens.
What this module adds is fs-research-specific observability around those:
the tool-calling loop, scope/memory context, budget exhaustion, citation
extraction, etc.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger("fs_research_agent.observability")

try:
    import logfire as _logfire
    LOGFIRE_AVAILABLE: bool = True
    logfire: Optional[Any] = _logfire
except Exception as e:
    LOGFIRE_AVAILABLE = False
    logfire = None
    logger.debug(f"logfire not available: {e}")


@contextmanager
def span(_event: str, /, **attrs: Any) -> Iterator[Optional[Any]]:
    """Context manager that opens a logfire span (or yields None if unavailable).

    The yielded value is the underlying logfire span object — use it to add
    attributes mid-execution via `s.set_attribute(key, value)`. Always
    None-check before calling set_attribute since logfire may be off.

    NOTE: the event name is positional-only (`/`) so callers can freely
    pass `name=...` in attrs (e.g. tool name) without colliding with the
    parameter name.
    """
    if not (LOGFIRE_AVAILABLE and logfire):
        yield None
        return
    try:
        with logfire.span(_event, **attrs) as s:
            yield s
    except Exception as e:
        # Never let observability break the request.
        logger.debug(f"span {_event!r} failed: {e}")
        yield None


def info(_event: str, /, **attrs: Any) -> None:
    """Fire-and-forget info event. No-op when logfire is unavailable."""
    if not (LOGFIRE_AVAILABLE and logfire):
        return
    try:
        logfire.info(_event, **attrs)
    except Exception as e:
        logger.debug(f"info {_event!r} failed: {e}")


def warn(_event: str, /, **attrs: Any) -> None:
    """Fire-and-forget warn event. No-op when logfire is unavailable."""
    if not (LOGFIRE_AVAILABLE and logfire):
        return
    try:
        logfire.warn(_event, **attrs)
    except Exception as e:
        logger.debug(f"warn {_event!r} failed: {e}")


def error(_event: str, /, **attrs: Any) -> None:
    """Fire-and-forget error event. No-op when logfire is unavailable."""
    if not (LOGFIRE_AVAILABLE and logfire):
        return
    try:
        logfire.error(_event, **attrs)
    except Exception as e:
        logger.debug(f"error {_event!r} failed: {e}")


def truncate(text: Any, n: int = 200) -> str:
    """Helper to keep big payloads (e.g. tool results, prompts) bounded in spans."""
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"
