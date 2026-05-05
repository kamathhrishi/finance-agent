"""
LLM error utilities — `LLMError` and `format_error_for_user`.

Lifted verbatim from the now-deleted `agent/rag/llm_utils.py` because
`app/routers/chat.py` uses these symbols at the top-level error boundary
to translate any raw exception into a user-safe message string.

This module is intentionally tiny. It contains:
  - `LLMError`: a custom exception that carries both a user-safe message
    and an optional internal/technical detail (logged, never surfaced).
  - `format_error_for_user(exc)`: maps an exception to one of a small set
    of friendly strings via keyword pattern-matching on the exception text.

NOTE: this is application boundary code, not LLM-call retry code. The
retry decorator that used to live alongside these in `agent/rag/llm_utils.py`
has been removed because the active agent (`agent.agent`) does
its own per-attempt retry inside the OpenAI call loop.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Friendly messages — keep tight, no jargon, no internal symbol leakage.
# ─────────────────────────────────────────────────────────────────────────────

USER_FRIENDLY_ERRORS = {
    "rate_limit": "The AI service is currently busy. Please try again in a moment.",
    "timeout":    "The request took too long. Please try again.",
    "connection": "Unable to connect to the AI service. Please check your connection and try again.",
    "quota":      "Service quota exceeded. Please try again later.",
    "auth":       "Authentication error. Please contact support.",
    "default":    "We encountered an issue processing your request. Please try again.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Exception with both a user-facing message and a technical detail.

    Top-level error handlers should surface `user_message` to the client and
    log `technical_message` server-side. `retryable=True` is a hint that the
    caller may want to retry the operation that raised this.
    """

    def __init__(
        self,
        user_message: str,
        technical_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.user_message = user_message
        self.technical_message = technical_message or user_message
        self.retryable = retryable
        super().__init__(user_message)

    def __str__(self) -> str:
        return self.user_message


# ─────────────────────────────────────────────────────────────────────────────
# Boundary formatter — used by chat router to never leak internals
# ─────────────────────────────────────────────────────────────────────────────

def _classify(error: Exception) -> str:
    """Return the friendly-message key for an arbitrary exception."""
    msg = str(error).lower()
    if any(w in msg for w in ("rate", "limit", "too_many", "queue", "traffic")):
        return "rate_limit"
    if any(w in msg for w in ("timeout", "timed out")):
        return "timeout"
    if any(w in msg for w in ("connection", "connect")):
        return "connection"
    if any(w in msg for w in ("quota", "billing")):
        return "quota"
    if any(w in msg for w in ("auth", "api_key", "unauthorized")):
        return "auth"
    return "default"


def format_error_for_user(error: Exception) -> str:
    """Translate any exception to a user-safe string. Never returns internals.

    Use at the chat router's top-level error boundary. If the exception is an
    `LLMError`, its pre-baked `user_message` is returned. Otherwise we
    classify by keyword and return one of the canned `USER_FRIENDLY_ERRORS`.
    """
    if isinstance(error, LLMError):
        return error.user_message
    return USER_FRIENDLY_ERRORS[_classify(error)]
