"""
Adapter that exposes FilesystemResearchAgent as a drop-in chat agent.

It implements the chat router's standard agent interface — an async
generator yielding `{type: ...}` events, terminating with a `result` event
of shape `{type: 'result', data: {response: {answer, citations}, timing}}`.
This is the only agent in the platform; the contract used to be shared
with legacy `RAGAgent` / `OrchestratorAgent` implementations, both of which
have since been retired. The shape lives on for the chat router and
frontend that expect it.

Citations are built from the agent's `path:line` cites (via
`agent.citations.extract_citations`) and shaped to match the
existing `ChatCitation` schema (with `source_backend='fs_research'` and
extra `line_start` / `line_end` fields the highlighter endpoint needs).
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from .agent import (
    FilesystemResearchAgent,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    _resolve_default_data_root,
)
from .citations import extract_citations
from .observability import span, info as obs_info, truncate

logger = logging.getLogger("agent.adapter")

# The chat router still passes `max_iterations=3` for historical reasons
# (the retired OrchestratorAgent counted ReAct iterations, where one iter =
# many parallel tool calls). The FS agent counts individual tool calls, so 3
# would strangle it before it can even open a filing. Ignore the kwarg and
# use our own budget. Override with FS_RESEARCH_TOOL_BUDGET env var.
_DEFAULT_FS_BUDGET = int(os.getenv("FS_RESEARCH_TOOL_BUDGET", "30"))

# ─── Model alias map ─────────────────────────────────────────────────────────
#
# The frontend sends a friendly display id (e.g. "gpt-5.4-mini") that matches
# what the user picked in the UI. We resolve to a dated OpenAI model id here.
# Keeping the mapping in one place means the UI can advertise the next model
# the moment the backend swaps the alias — no frontend change required.
#
# Aliases NOT in this map (or set to None) are treated as "not yet wired":
# the request silently falls back to DEFAULT_MODEL. The UI already greys out
# everything except 'gpt-5.4-mini', so this is just defense-in-depth.

MODEL_ALIASES: Dict[str, Optional[str]] = {
    # Currently routed
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
    # Wired for later — flip the value to the real model id when ready
    "gpt-5.5":      "gpt-5.5-2026-04-23",
    "gpt-5.5-pro":  "gpt-5.5-pro-2026-04-23",
    "gpt-5.4-nano": "gpt-5.4-nano-2026-03-17",
    # Not yet known to the backend — None forces a fallback to the default
    "gpt-5.4":      None,
    "gpt-5.4-pro":  None,
    "gpt-5-mini":   None,
    "gpt-5-nano":   None,
}


def resolve_model(display_id: Optional[str], default: str) -> str:
    """Map a UI model id to a real OpenAI model id, or fall back."""
    if not display_id:
        return default
    real = MODEL_ALIASES.get(display_id)
    if real:
        return real
    logger.info(f"model alias '{display_id}' not currently wired; using default {default}")
    return default

# ─── Conversational memory ───────────────────────────────────────────────────
#
# Process-level LRU keyed by `conversation_id` only. Lets follow-up questions
# ("what about FY2024?", "compare with MSFT", "is it deteriorating?") inherit
# the prior topic without the user re-stating context.
#
# Why conversation_id alone is sufficient:
#   - conversation_id is a UUID, globally unique across users. No collision.
#   - For authed users, the chat router validates conversation_id ownership
#     before this cache is touched.
#   - For anonymous users, conversation_id IS the only stable token we have —
#     the user_id (`demo:session_<timestamp>`) gets fabricated fresh per
#     request when the frontend doesn't send a session_id, which made the
#     prior `(user_id, conversation_id)` tuple key NEVER MATCH ITSELF across
#     turns. Result: anon memory was always empty, defeating the point.
#
# The user_id parameter is kept for logging and future hooks but is not part
# of the cache key.
#
# Lifecycle:
#   - Lives in-memory only (not persisted). Lost on restart by design.
#   - LRU-evicts oldest conversations when the cache hits MEM_MAX_CONVERSATIONS.
#   - Per-conversation, only the last MEM_MAX_TURNS user/assistant turns kept.

MEM_MAX_CONVERSATIONS = 200
MEM_MAX_TURNS = 6                # ~3 user + 3 assistant pairs
# Assistant cap is generous on purpose. Analytical answers in this product
# routinely run 6-10 KB (multi-section findings with citations). The prior
# 1800-char cap kept only ~250 words — usually the question restatement and
# the start of section 1 — which made follow-ups like "is the relationship
# deteriorating?" effectively context-free. 8000 chars covers ~95% of
# answers in full, with the tail clipped on the very long ones.
# Cost: at MEM_MAX_TURNS=6 and one user/assistant pair, the worst-case
# preamble is ~50 KB ≈ ~12K input tokens per LLM call. Inside one ReAct
# loop the preamble is byte-identical across calls so prompt caching
# absorbs it after the first hit.
MEM_ASSISTANT_TRUNC_CHARS = 8000
MEM_USER_TRUNC_CHARS = 1500


@dataclass
class _Turn:
    role: str            # "user" or "assistant"
    content: str
    ts: float


class ConversationMemory:
    """Thread-safe LRU of recent conversation turns, keyed by conversation_id."""

    def __init__(self) -> None:
        self._cache: "OrderedDict[str, List[_Turn]]" = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _key(conversation_id: Optional[str]) -> Optional[str]:
        # Keep the staticmethod signature so a future change (e.g. namespace
        # by user) is a one-line edit. Today: conversation_id IS the key.
        if not conversation_id:
            return None
        return conversation_id

    def get(self, user_id: Optional[str], conversation_id: Optional[str]) -> List[_Turn]:
        # user_id retained in signature for backwards compat / call-site logging,
        # but intentionally NOT used in the key. See module docstring.
        del user_id
        key = self._key(conversation_id)
        if key is None:
            return []
        with self._lock:
            turns = self._cache.get(key)
            if turns is None:
                return []
            self._cache.move_to_end(key)
            return list(turns)  # defensive copy

    def append(
        self,
        user_id: Optional[str],
        conversation_id: Optional[str],
        role: str,
        content: str,
    ) -> None:
        # user_id retained in signature for backwards compat. See module docstring.
        del user_id
        key = self._key(conversation_id)
        if key is None or not content:
            return
        cap = MEM_USER_TRUNC_CHARS if role == "user" else MEM_ASSISTANT_TRUNC_CHARS
        if len(content) > cap:
            content = content[: cap - 1].rstrip() + "…"
        turn = _Turn(role=role, content=content, ts=time.time())
        with self._lock:
            turns = self._cache.get(key)
            if turns is None:
                turns = []
                self._cache[key] = turns
            turns.append(turn)
            if len(turns) > MEM_MAX_TURNS:
                drop = len(turns) - MEM_MAX_TURNS
                del turns[:drop]
            self._cache.move_to_end(key)
            while len(self._cache) > MEM_MAX_CONVERSATIONS:
                self._cache.popitem(last=False)


_MEMORY = ConversationMemory()


def _format_scope_preamble(filings: List[Dict[str, Any]]) -> str:
    """Render pinned filings as a scope block prepended to the new question.

    Returns "" when there's nothing pinned. Format is intentionally compact and
    labeled clearly so the model treats it as authoritative scope guidance.
    """
    if not filings:
        return ""
    lines = ["[User has pinned the following filings to this chat — prefer them over auto-discovery unless the question clearly refers to something else:]"]
    for f in filings:
        ticker = (f.get("ticker") or "").strip().upper()
        form = (f.get("form") or "").strip()
        period = (f.get("period_label") or "").strip()
        filing_date = (f.get("filing_date") or "").strip()
        path = (f.get("path") or "").strip()
        if not (ticker and form and path):
            continue
        bits = [f"{ticker} {form}"]
        if period:
            bits.append(period)
        if filing_date:
            bits.append(f"filed {filing_date}")
        lines.append(f"  - {' '.join(bits)} → read from `{path}/`")
    lines.append("[End of pinned scope.]")
    return "\n".join(lines)


def _format_memory_preamble(turns: List[_Turn]) -> str:
    """Render prior turns as a context block prepended to the new question.

    Returns "" when there's nothing to include. The wording is deliberate:
    we instruct the model to RESOLVE follow-up pronouns ("it", "they", "the
    relationship") against the prior turns instead of asking the user to
    restate context. Without this, the model frequently asked clarifying
    questions on obvious follow-ups (e.g. "would you say the relationship
    is deteriorating?" → "which company / which relationship?").
    """
    if not turns:
        return ""
    lines = [
        "[Conversation so far — these are EARLIER turns from this SAME chat with this SAME user.",
        "Treat them as authoritative shared context. The current question is almost certainly a",
        "follow-up. Resolve pronouns and vague references (\"it\", \"they\", \"the relationship\",",
        "\"that company\", \"is it deteriorating\", etc.) using this prior context — DO NOT ask",
        "the user to restate what was just discussed. If the follow-up is genuinely ambiguous",
        "even with this context, then ask; otherwise proceed.]",
        "",
    ]
    for t in turns:
        if t.role == "user":
            lines.append(f"USER (earlier): {t.content}")
        else:
            lines.append(f"ASSISTANT (earlier): {t.content}")
    lines.append("")
    lines.append("[End of prior turns. The CURRENT question follows.]")
    return "\n".join(lines)


class FilesystemResearchOrchestrator:
    """
    Drop-in agent compatible with `app/routers/chat.py`.

    Required surface used by the router:
      - `execute_rag_flow(question, ..., stream=True)` async generator
      - `execute_rag_flow_async(question, **kwargs)` coroutine that returns dict
      - `set_database_connection(db)`            (no-op — FS agent has no DB)
      - `set_user_context(user_id=..., ...)`     (no-op)
    """

    def __init__(
        self,
        data_root: Optional[Path] = None,
        model: str = DEFAULT_MODEL,
        max_tool_calls: int = _DEFAULT_FS_BUDGET,
    ) -> None:
        # Honor FS_RESEARCH_DATA_ROOT env at instantiation time (not just at
        # module import) so a lifespan that sets it after import still works.
        self.data_root = Path(data_root) if data_root else _resolve_default_data_root()
        self.model = model
        self.max_tool_calls = max_tool_calls
        self._agent = FilesystemResearchAgent(
            data_root=self.data_root,
            model=self.model,
            max_tool_calls=self.max_tool_calls,
        )
        logger.info(
            f"FilesystemResearchOrchestrator ready (model={self.model}, "
            f"data_root={self.data_root})"
        )

    # ── Compatibility shims (router calls these, FS agent doesn't need them) ─

    def set_database_connection(self, _db: Any) -> None:  # noqa: D401
        """No-op — the filesystem agent does not touch the DB."""
        return None

    def set_user_context(self, **_kwargs: Any) -> None:  # noqa: D401
        """No-op — no per-user state."""
        return None

    # ── Main execution ───────────────────────────────────────────────────────

    async def execute_rag_flow(
        self,
        question: str,
        show_details: bool = False,                       # noqa: ARG002 (unused)
        comprehensive: bool = True,                       # noqa: ARG002
        stream_callback: Any = None,                      # noqa: ARG002
        max_iterations: Optional[int] = None,
        conversation_id: Optional[str] = None,
        stream: bool = True,                              # noqa: ARG002
        user_id: Optional[str] = None,
        scoped_filings: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **_extra: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Drive the FS agent and yield SSE-shaped events.

        Maps the FS agent's events to chat-router event types:
          fs `progress`   → `{type: progress, message, step}`
          fs `llm_call`   → `{type: reasoning, message, step}`
          fs `tool_start` → `{type: reasoning, message: "📂 ls(...)" / "🔎 grep(...)" / etc, step: '10k_search'}`
          fs `tool_end`   → `{type: reasoning, message: "← N chars", step: 'retrieval_complete'}`
          fs `token`      → `{type: token, content}`
          fs `result`     → `{type: result, data: {response: {answer, citations}, timing}}`
        """
        start = time.time()

        # IGNORE the chat router's `max_iterations` — it ships `3` by default,
        # which is a ReAct-iteration count for OrchestratorAgent (each iter
        # fires N parallel tool calls). For us each tool call counts individually,
        # so 3 is far too low. Stick with the FS agent's own budget
        # (DEFAULT_FS_BUDGET, override via FS_RESEARCH_TOOL_BUDGET).
        _ = max_iterations  # explicitly unused; FS agent uses its own budget

        yield {
            "type": "progress",
            "message": "Analyzing question and planning research...",
            "step": "reasoning",
            "data": {},
        }

        # Top-level observability event for the whole flow. The OpenAI
        # autoinstrumentation handles per-LLM-call spans; agent.run() adds
        # per-tool-call + force-final spans. This event ties them together
        # with caller context (user, conversation, scope, memory).
        obs_info(
            "fs_research.flow.start",
            conversation_id=conversation_id,
            user_id=user_id or "(anon)",
            question=truncate(question, 500),
            question_chars=len(question),
            scoped_filings_count=len(scoped_filings or []),
            scoped_filings=[
                f"{(f.get('ticker') or '').upper()} {f.get('form') or ''} {f.get('period_label') or ''}"
                for f in (scoped_filings or [])
            ][:10],
        )

        # ── Conversational memory: prepend prior turns to the question ──
        # Strict isolation: keyed by (user_id, conversation_id) tuple.
        prior_turns = _MEMORY.get(user_id, conversation_id)
        scope_block = _format_scope_preamble(scoped_filings or [])

        parts: List[str] = []
        if prior_turns:
            parts.append(_format_memory_preamble(prior_turns))
            logger.info(
                f"💬 Using {len(prior_turns)} prior turn(s) for conv={conversation_id} "
                f"user={user_id or '(anon)'}"
            )
        if scope_block:
            parts.append(scope_block)
            logger.info(
                f"📌 Pinned scope: {len(scoped_filings or [])} filing(s) for "
                f"conv={conversation_id} user={user_id or '(anon)'}"
            )
        parts.append(f"USER: {question}" if parts else question)
        agent_question = "\n\n".join(parts)

        raw_answer = ""
        tool_calls = 0
        llm_calls = 0
        elapsed = 0.0
        last_emitted: Optional[str] = None      # last reasoning message we sent the user

        # Heuristic dedupers: the agent often does the same kind of read
        # across many periods (4 metadata.json checks, 4 MD&A reads). Showing
        # all of those is noisy. We collapse repeats of the same "kind" by
        # tracking what's already been emitted.
        emitted_orientation = False              # README/INDEX once is enough
        emitted_metadata_check = False           # "Confirming details…" once is enough

        flow_span_cm = span(
            "fs_research.flow",
            conversation_id=conversation_id,
            user_id=user_id or "(anon)",
            scoped_filings_count=len(scoped_filings or []),
            prior_turns=len(prior_turns),
            model=self.model,
        )
        flow_span = flow_span_cm.__enter__()
        resolved_model = resolve_model(model, self.model)
        try:
            async for ev in self._agent.run(agent_question, model_override=resolved_model):
                et = ev.get("type")

                # ── INTERNAL events: drop entirely (LLM call counts, raw progress) ──
                if et in ("progress", "llm_call"):
                    continue

                # ── tool_start: emit a humanized "what I'm doing" line ──
                if et == "tool_start":
                    name = ev.get("name", "?")
                    args = ev.get("args", {}) or {}
                    label, step = _humanize_tool_call(name, args)

                    # Suppress noisy duplicates
                    if "Getting oriented" in label:
                        if emitted_orientation:
                            continue
                        emitted_orientation = True
                    if "Confirming details" in label or "Checking filing metadata" in label:
                        if emitted_metadata_check:
                            continue
                        emitted_metadata_check = True
                        # collapse to a generic "Confirming filing details…" line
                        # since across many filings the per-filing label is noise
                        label = "Confirming filing details…"
                    if label == last_emitted:
                        # Same exact label as the previous event — drop the repeat
                        continue

                    last_emitted = label
                    yield {
                        "type": "reasoning",
                        "message": label,
                        "step": step,
                        "data": {},
                    }

                # ── tool_end: emit a result summary only when it adds info ──
                elif et == "tool_end":
                    tool_calls = max(tool_calls, ev.get("call_num", tool_calls))
                    name = ev.get("name", "?")
                    chars = ev.get("result_chars", 0)
                    summary = _humanize_tool_result(name, chars)
                    if summary:
                        yield {
                            "type": "reasoning",
                            "message": summary,
                            "step": "retrieval_complete",
                            "data": {},
                        }

                elif et == "token":
                    # Don't forward live tokens — we stream the final answer
                    # AFTER citation rewriting so users see [FS-N] markers,
                    # not raw `path:line` strings.
                    continue

                elif et == "result":
                    raw_answer = ev.get("answer", "") or ""
                    elapsed = float(ev.get("elapsed_s", time.time() - start))

                elif et == "error":
                    obs_info(
                        "fs_research.flow.error",
                        conversation_id=conversation_id,
                        user_id=user_id or "(anon)",
                        message=ev.get("message", "Research error"),
                        tool_calls=tool_calls,
                    )
                    yield {
                        "type": "error",
                        "message": ev.get("message", "Research error"),
                    }
                    flow_span_cm.__exit__(None, None, None)
                    return

        except Exception as e:
            # Log the real exception (with traceback) for the operator + send
            # detailed event to logfire — but show the user a clean, generic
            # message that doesn't leak internal symbols, file paths, stack
            # traces, or library names.
            logger.exception("FilesystemResearchOrchestrator failed")
            obs_info(
                "fs_research.flow.error",
                conversation_id=conversation_id,
                user_id=user_id or "(anon)",
                error_type=type(e).__name__,
                error_message=truncate(str(e), 300),
            )
            yield {
                "type": "error",
                "message": "Sorry — something went wrong on our end. Please try again, or rephrase the question.",
            }
            flow_span_cm.__exit__(None, None, None)
            return

        # ── Post-process: rewrite path:line cites → [FS-N], build citations ──
        rewritten, fs_cites = extract_citations(raw_answer, self.data_root)
        citations = [c.to_chat_citation() for c in fs_cites]

        # ── Persist this turn into the LRU for the next follow-up ──
        # We save the *original* user question (not the preamble-augmented one)
        # and the *rewritten* answer (citation markers stripped via the
        # extractor — the next turn's model gets clean prose, not [FS-N] noise).
        if rewritten and conversation_id:
            _clean_for_memory = re.sub(r"\s*\[(?:FS|10K|10Q|8K)-\d+\]", "", rewritten)
            _MEMORY.append(user_id, conversation_id, "user", question)
            _MEMORY.append(user_id, conversation_id, "assistant", _clean_for_memory)

        # Stream the rewritten answer as tokens (so the chat UI types it out).
        # We do this AFTER rewriting so the user sees [FS-1] markers, not paths.
        if rewritten:
            for token in _word_tokens(rewritten):
                yield {"type": "token", "content": token}

        elapsed_total = time.time() - start

        obs_info(
            "fs_research.flow.complete",
            conversation_id=conversation_id,
            user_id=user_id or "(anon)",
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            citation_count=len(citations),
            scoped_filings_count=len(scoped_filings or []),
            prior_turns=len(prior_turns),
            answer_chars=len(rewritten or ""),
            raw_answer_chars=len(raw_answer or ""),
            agent_elapsed_s=round(elapsed, 2),
            total_elapsed_s=round(elapsed_total, 2),
            elapsed_ms=int(elapsed_total * 1000),
        )
        if flow_span is not None:
            try:
                flow_span.set_attribute("tool_calls", tool_calls)
                flow_span.set_attribute("citation_count", len(citations))
                flow_span.set_attribute("answer_chars", len(rewritten or ""))
            except Exception:
                pass
        flow_span_cm.__exit__(None, None, None)

        yield {
            "type": "result",
            "data": {
                "response": {
                    "answer": rewritten,
                    "citations": citations,
                },
                "timing": {
                    "total_seconds": round(elapsed_total, 2),
                    "agent_seconds": round(elapsed, 2),
                },
                "stats": {
                    "tool_calls": tool_calls,
                    "llm_calls": llm_calls,
                    "citation_count": len(citations),
                },
                "backend": "fs_research",
            },
        }

    async def execute_rag_flow_async(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Collect the stream into a single dict result (for non-streaming callers)."""
        final_result: Dict[str, Any] = {}
        async for ev in self.execute_rag_flow(question, **kwargs):
            if ev.get("type") == "result":
                final_result = ev.get("data", {})
        return final_result


# ─── User-facing reasoning trace helpers ─────────────────────────────────────
#
# All of these translate the agent's internal tool calls into outcome-focused
# messages an analyst would understand — no file paths, no tool names, no
# byte counts. Following the existing platform style, e.g.:
#   "Looking at {ticker}'s annual report..."
#   "Found N relevant sections in the filing"


# Map directory-style section keys to their human names. Keys are derived
# from filename stems (e.g. `item-7-mda` → "Management's Discussion & Analysis").
# These are the SEC's own item titles — generic, not company-specific.
# Short, status-line-friendly section labels (used in the live reasoning trace).
# Citation cards use the longer `_SECTION_LABELS` defined in citations.py.
_SECTION_NAMES: Dict[str, str] = {
    # ── 10-K ──────────────────────────────────────────────────────────────────
    "item-1-business": "Business overview",
    "item-1a-risk-factors": "Risk Factors",
    "item-1b-unresolved-staff-comments": "Unresolved Staff Comments",
    "item-1c-cybersecurity": "Cybersecurity",
    "item-2-properties": "Properties",
    "item-3-legal-proceedings": "Legal Proceedings",
    "item-4-mine-safety": "Mine Safety",
    "item-5-market-for-registrants-equity": "Market for Equity",
    "item-6-selected-financial-data": "Selected Financial Data",
    "item-7-mda": "MD&A",
    "item-7a-quant-qual-disclosures": "Market Risk disclosures",
    "item-8-financial-statements": "Financial Statements",
    "item-9-changes-in-accountants": "Changes in Accountants",
    "item-9a-controls-and-procedures": "Controls & Procedures",
    "item-9b-other-information": "Other Information",
    "item-10-directors-and-officers": "Directors & Officers",
    "item-11-executive-compensation": "Executive Compensation",
    "item-12-security-ownership": "Security Ownership",
    "item-13-related-party-transactions": "Related Party Transactions",
    "item-14-principal-accountant-fees": "Principal Accountant Fees",
    "item-15-exhibits": "Exhibits",
    # ── 10-Q Part I ───────────────────────────────────────────────────────────
    "item-1-financial-statements": "Financial Statements",
    "item-2-mda": "MD&A",
    "item-3-quant-qual-market-risk": "Market Risk disclosures",
    "item-4-controls-and-procedures": "Controls & Procedures",
    # ── 10-Q Part II ──────────────────────────────────────────────────────────
    "item-2-unregistered-equity-sales": "Unregistered Equity Sales",
    "item-3-defaults-on-senior-securities": "Defaults on Senior Securities",
    "item-5-other-information": "Other Information",
    "item-6-exhibits": "Exhibits list",
    # ── 8-K (decimal items) ───────────────────────────────────────────────────
    "item-1-01-material-definitive-agreement": "Material Agreement (1.01)",
    "item-1-02-termination-material-agreement": "Agreement Termination (1.02)",
    "item-2-01-acquisition-disposition": "Acquisition/Disposition (2.01)",
    "item-2-02-results-of-operations": "Results of Operations (2.02)",
    "item-2-05-exit-or-disposal": "Exit/Disposal (2.05)",
    "item-2-06-material-impairments": "Material Impairments (2.06)",
    "item-3-02-unregistered-equity-sales": "Unregistered Equity Sales (3.02)",
    "item-4-02-non-reliance-on-financials": "Restatement (4.02)",
    "item-5-02-officer-departure-election": "Officer Departure (5.02)",
    "item-5-03-bylaw-amendments": "Bylaw Amendments (5.03)",
    "item-5-07-shareholder-vote": "Shareholder Vote (5.07)",
    "item-7-01-regulation-fd": "Regulation FD (7.01)",
    "item-8-01-other-events": "Other Events (8.01)",
    "item-9-01-financial-statements-and-exhibits": "Financial Statements & Exhibits (9.01)",
}


# Path schemas by form:
#   10-K:  filings/<TICKER>/10-K/FY####/...
#   10-Q:  filings/<TICKER>/10-Q/FY####/Q[1-4]/...
#   8-K:   filings/<TICKER>/8-K/YYYY-MM-DD/...
_PATH_RE = re.compile(
    r"filings/(?P<ticker>[A-Z][A-Z0-9._-]{0,9})/(?P<form>10-K|10-Q|8-K)/"
    r"(?P<period>FY?\d{4}(?:/Q[1-4])?|\d{4}-\d{2}-\d{2})"
    r"(?:/(?P<rest>.*))?"
)


def _parse_filing_path(path: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (ticker, form, period_label, section_name_or_None)."""
    if not path:
        return None, None, None, None
    m = _PATH_RE.match(path)
    if not m:
        return None, None, None, None
    ticker = m.group("ticker")
    form = m.group("form")
    period_raw = m.group("period") or ""
    # Pretty period label:
    #   "FY2025"      → "FY2025"
    #   "FY2025/Q3"   → "FY2025 Q3"
    #   "2024-08-15"  → "filed 2024-08-15"
    if "/" in period_raw:           # FY####/Q#
        period_pretty = period_raw.replace("/", " ")
    elif period_raw.startswith("FY"):
        period_pretty = period_raw
    else:
        # Date format used by 8-K
        period_pretty = f"filed {period_raw}"
    rest = m.group("rest") or ""
    section: Optional[str] = None
    if rest.startswith("sections/"):
        stem = Path(rest).stem
        section = _SECTION_NAMES.get(stem, stem.replace("-", " ").title())
    elif rest.startswith("exhibits/"):
        section = f"exhibit {Path(rest).stem}"
    return ticker, form, period_pretty, section


def _humanize_tool_call(name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """
    Translate a tool call into ('user-facing message', 'step name').

    Step names match the platform's existing reasoning-trace step taxonomy
    so the frontend's `_REASONING_TYPES` set picks them up.
    """
    path = args.get("path", "") or ""
    pattern = args.get("pattern", "") or ""
    glob_arg = args.get("glob", "") or ""

    # ── ls ──
    if name == "ls":
        if not path or path in (".", ""):
            return "Browsing the available filings library…", "10k_planning"
        # Per-ticker INDEX shortcut (`filings/NVDA/`)
        rel = path.strip("/")
        if rel.startswith("filings/") and rel.count("/") == 1:
            t = rel.split("/")[1]
            return f"Reviewing {t}'s available filings…", "10k_planning"
        ticker, form, period_pretty, _ = _parse_filing_path(path)
        if ticker and form and period_pretty:
            return f"Looking inside {ticker}'s {form} ({period_pretty})…", "10k_planning"
        if ticker and form:
            return f"Reviewing {ticker}'s available {form} filings…", "10k_planning"
        if ticker:
            return f"Reviewing {ticker}'s available filings…", "10k_planning"
        return "Browsing the filings library…", "10k_planning"

    # ── read_file ──
    if name == "read_file":
        rel = path.strip("/")
        # Top-level orientation reads
        if rel in ("README.md", "INDEX.md"):
            return "Getting oriented in the filings library…", "10k_planning"
        # Per-ticker INDEX
        if rel.startswith("filings/") and rel.endswith("/INDEX.md") and rel.count("/") == 2:
            t = rel.split("/")[1]
            return f"Reading {t}'s filing index…", "10k_planning"
        ticker, form, period_pretty, section = _parse_filing_path(path)
        if rel.endswith("/metadata.json"):
            if ticker and form and period_pretty:
                return f"Confirming details of {ticker}'s {form} ({period_pretty})…", "10k_planning"
            return "Checking filing metadata…", "10k_planning"
        if section and ticker and period_pretty:
            return f"Reading {ticker}'s {section} ({period_pretty})…", "10k_retrieval"
        if rel.endswith("/filing.md") and ticker and form and period_pretty:
            return f"Reading {ticker}'s full {form} ({period_pretty})…", "10k_retrieval"
        return "Reading filing content…", "10k_retrieval"

    # ── grep ──
    if name == "grep":
        snippet = pattern.strip()
        # Drop regex anchors / common ripgrep artifacts for display
        snippet = re.sub(r"^[\^\\]+|[\$\\]+$", "", snippet)
        if len(snippet) > 60:
            snippet = snippet[:57] + "…"
        scope_path = glob_arg or path
        ticker, form, period_pretty, section = _parse_filing_path(scope_path)
        if section and ticker and period_pretty:
            return f"Searching {ticker}'s {section} ({period_pretty}) for “{snippet}”…", "10k_search"
        if ticker and form and period_pretty:
            return f"Searching {ticker}'s {form} ({period_pretty}) for “{snippet}”…", "10k_search"
        if ticker and form:
            return f"Searching {ticker}'s {form} filings for “{snippet}”…", "10k_search"
        if ticker:
            return f"Searching {ticker}'s filings for “{snippet}”…", "10k_search"
        if "**" in scope_path or "*" in scope_path:
            return f"Searching across filings for “{snippet}”…", "10k_search"
        return f"Searching the filings library for “{snippet}”…", "10k_search"

    # ── glob ──
    if name == "glob":
        # Glob is mostly a sanity-check operation; downplay it.
        return "Locating relevant filings…", "10k_planning"

    # Fallback (shouldn't trigger in practice)
    return "Researching the filings…", "10k_search"


def _humanize_tool_result(name: str, chars: int) -> Optional[str]:
    """
    Return a short result message — or None (most of the time) to suppress the event.

    We deliberately surface very little here. The action message ("Searching X
    for Y…") is what the user wants to see; pairing every action with a "Found N
    passages" doubles the trace length without adding signal. Only the
    *informative* result — "no matches" — is worth showing, because it explains
    why the agent will retry with a different pattern.
    """
    if name == "grep" and chars == 0:
        return "No matches — refining search."
    return None


def _word_tokens(text: str):
    """Yield word-sized tokens (preserving spacing) so the UI can stream them."""
    words = text.split(" ")
    for i, w in enumerate(words):
        sep = " " if i < len(words) - 1 else ""
        yield w + sep
