#!/usr/bin/env python3
"""
FilesystemResearchAgent — pure-Python ReAct loop over filesystem tools only.

Driver: gpt-5.4-mini-2026-03-17 via OpenAI function calling.
Tools: ls / read_file / grep / glob (sandboxed to data/).
No LangChain. No reasoning_effort (gpt-5.4 family rejects it with tools on
/v1/chat/completions; we set temperature=0 instead).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from dotenv import load_dotenv

from .prompts import build_system_prompt
from .tools import Sandbox, TOOL_SCHEMAS, make_tool_executor
from .observability import span, info as obs_info, warn as obs_warn, truncate

logger = logging.getLogger("fs_research_agent.agent")


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_MAX_TOOL_CALLS = 25
DEFAULT_MAX_COMPLETION_TOKENS = 16000

PKG_ROOT = Path(__file__).resolve().parent


def _resolve_default_data_root() -> Path:
    """Resolve the corpus root with env override, falling back to the in-repo
    location. This is computed lazily (every import) so a deploy can set
    FS_RESEARCH_DATA_ROOT without any agent-side code change.
    """
    env = os.getenv("FS_RESEARCH_DATA_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return PKG_ROOT / "data"


# Module-level evaluated once on import. Callers that want late binding
# (e.g. tests that patch the env var) should call _resolve_default_data_root().
DEFAULT_DATA_ROOT = _resolve_default_data_root()


def _wait_from_429(e: openai.RateLimitError, attempt: int) -> float:
    """
    Compute how long to sleep before retrying a 429 rate-limit error.

    Order of preference (most accurate → least):
      1. The `retry-after` HTTP header (seconds, integer)
      2. The `Retry-After-Ms` HTTP header (milliseconds)
      3. The "Please try again in 235ms" / "in 1.234s" hint inside the
         error body — OpenAI's TPM limiter writes this when it knows exactly
         when the budget refills.
      4. Exponential backoff (5, 10, 20, 40, 60 seconds).

    Always clamps to [1.0, 60.0] so we never spin tightly OR sleep absurdly.
    """
    # 1 + 2: response headers
    headers = {}
    try:
        headers = dict(getattr(e.response, "headers", {}) or {})
    except Exception:
        headers = {}
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra is not None:
        try:
            v = float(ra)
            if v > 0:
                return max(1.0, min(v, 60.0))
        except (TypeError, ValueError):
            pass
    ra_ms = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
    if ra_ms is not None:
        try:
            v = float(ra_ms) / 1000.0
            if v > 0:
                return max(1.0, min(v, 60.0))
        except (TypeError, ValueError):
            pass

    # 3: parse the error body — OpenAI says "try again in 235ms" or "in 1.234s"
    msg = str(getattr(e, "message", "") or e)
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)\b", msg, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        if m.group(2).lower() == "ms":
            v /= 1000.0
        # Always add a small buffer — the suggested wait is often 100ms short
        # of what's actually needed for the rolling-window cap to refill.
        return max(1.0, min(v + 0.5, 60.0))

    # 4: exponential backoff, starting at 5s (NOT 1s — TPM windows are seconds-scale)
    fallback = min(5.0 * (2 ** attempt), 60.0)
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Event types yielded by the agent
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"type": self.type, **self.data}


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────


class FilesystemResearchAgent:
    """ReAct loop driving gpt-5.4-mini with four filesystem tools."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        model: str = DEFAULT_MODEL,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        openai_api_key: Optional[str] = None,
    ) -> None:
        # Load .env from repo root
        repo_env = PKG_ROOT.parent / ".env"
        if repo_env.exists():
            load_dotenv(dotenv_path=repo_env, override=False)

        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required (set in .env or env)")

        # Re-resolve the env at __init__ time too so that even if the env var
        # was set AFTER this module imported (e.g. by app/lifespan.py before
        # the orchestrator is instantiated), we still pick it up.
        self.data_root = (data_root or _resolve_default_data_root()).resolve()
        # Soft check. We log a warning if the dir is missing or empty at init,
        # but DO NOT raise — on Railway the chat router instantiates this
        # agent at import time, before the S3 bootstrap stage of the lifespan
        # populates the persistent volume. If the dir is genuinely empty when
        # a tool call eventually runs, the Sandbox will return a clean error
        # for that specific call instead of bringing down chat at boot.
        try:
            self.data_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if not self.data_root.is_dir():
            logger.warning(
                f"FilesystemResearchAgent: data root not present at init: {self.data_root}. "
                f"This is OK if the corpus will be populated by an S3 bootstrap or watcher cycle."
            )

        self.sandbox = Sandbox(self.data_root)
        self.execute_tool = make_tool_executor(self.sandbox)

        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_completion_tokens = max_completion_tokens

        self._client = openai.AsyncOpenAI(api_key=self.api_key)
        logger.info(
            f"FilesystemResearchAgent ready (model={self.model}, "
            f"data_root={self.data_root}, budget={self.max_tool_calls})"
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def run(
        self,
        question: str,
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Drive the ReAct loop and yield events.

        Yielded event shapes:
          {type: 'progress', message, ...}
          {type: 'llm_call', call_num}
          {type: 'tool_start', name, args}
          {type: 'tool_end', name, result_preview}
          {type: 'token', content}             — streamed final answer
          {type: 'result', answer, tool_calls, elapsed_s}
          {type: 'error', message}
        """
        start = time.time()

        # Per-request model. Falls back to whatever the agent was constructed
        # with — currently DEFAULT_MODEL (gpt-5.4-mini-2026-03-17). The UI
        # only enables one model so this override is plumbing for the future.
        active_model = model_override or self.model

        # Build the system prompt fresh each request so today's date is current.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ]

        tool_call_count = 0
        llm_call_num = 0
        final_answer = ""
        force_final_triggered = False

        # Top-level span over the whole agent run. Covers all LLM rounds + tool
        # calls + force-final + answer streaming. Sub-spans hang off this one.
        obs_info(
            "fs_research.run.start",
            model=active_model,
            max_tool_calls=self.max_tool_calls,
            question=truncate(question, 500),
            question_chars=len(question),
        )

        yield Event("progress", {"message": "Starting research"}).as_dict()

        while tool_call_count < self.max_tool_calls:
            llm_call_num += 1
            yield Event(
                "llm_call",
                {"call_num": llm_call_num, "messages": len(messages), "tool_calls_so_far": tool_call_count},
            ).as_dict()

            response = None
            with span(
                "fs_research.llm_round",
                call_num=llm_call_num,
                model=active_model,
                message_count=len(messages),
                tool_calls_so_far=tool_call_count,
            ):
                for _attempt in range(5):
                    try:
                        response = await self._client.chat.completions.create(
                            model=active_model,
                            messages=messages,
                            tools=TOOL_SCHEMAS,
                            tool_choice="auto",
                            temperature=0,  # gpt-5.4 family rejects reasoning_effort with tools
                            max_completion_tokens=self.max_completion_tokens,
                        )
                        break
                    except openai.RateLimitError as e:
                        wait = _wait_from_429(e, _attempt)
                        logger.warning(f"OpenAI 429; sleeping {wait:.1f}s (attempt {_attempt+1}/5)")
                        obs_warn(
                            "fs_research.rate_limit",
                            attempt=_attempt + 1,
                            wait_s=round(wait, 2),
                            phase="tool_loop",
                        )
                        await asyncio.sleep(wait)
                        continue
                    except (openai.APITimeoutError, openai.APIConnectionError) as e:
                        wait = min(2 ** _attempt, 15)
                        logger.warning(f"OpenAI transient {type(e).__name__}; sleeping {wait}s")
                        obs_warn(
                            "fs_research.transient_error",
                            error_type=type(e).__name__,
                            attempt=_attempt + 1,
                            wait_s=wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    except Exception as e:
                        logger.exception(f"OpenAI call failed (non-retryable): {e}")
                        obs_info(
                            "fs_research.run.error",
                            phase="llm_call",
                            error_type=type(e).__name__,
                            error_message=truncate(str(e), 300),
                            tool_calls_used=tool_call_count,
                            llm_calls=llm_call_num,
                            elapsed_s=round(time.time() - start, 2),
                        )
                        yield Event("error", {
                            "message": "Sorry — something went wrong contacting the model. Please try again.",
                        }).as_dict()
                        return
            if response is None:
                logger.error("OpenAI rate limit not cleared after 5 retries")
                yield Event("error", {
                    "message": "Sorry — the model is rate-limited right now. Please try again in a moment.",
                }).as_dict()
                return

            choice = response.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls or []

            # Append assistant turn (preserve tool_calls so we can match tool responses)
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            # No more tool calls → done
            if not tool_calls:
                final_answer = msg.content or ""
                break

            # Execute every tool call (parallel within a single turn)
            for tc in tool_calls:
                tool_call_count += 1
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield Event("tool_start", {"name": name, "args": args, "call_num": tool_call_count}).as_dict()

                tool_start = time.time()
                with span(
                    "fs_research.tool_call",
                    name=name,
                    call_num=tool_call_count,
                    path=truncate(args.get("path"), 200),
                    pattern=truncate(args.get("pattern"), 200),
                    glob=truncate(args.get("glob"), 200),
                ) as tool_span:
                    # Execute synchronously — these are local FS ops, fast
                    result = self.execute_tool(name, args)
                    tool_elapsed = time.time() - tool_start
                    truncated = len(result) > 16000
                    if truncated:
                        result = result[:16000] + "\n... (truncated to 16000 chars)"
                    if tool_span is not None:
                        try:
                            tool_span.set_attribute("result_chars", len(result))
                            tool_span.set_attribute("duration_ms", int(tool_elapsed * 1000))
                            tool_span.set_attribute("truncated", truncated)
                            tool_span.set_attribute("hit", len(result) > 0)
                        except Exception:
                            pass

                preview = result[:300].replace("\n", " ⏎ ")
                # Include the FULL result (post-cap) so trace consumers can do
                # forensic analysis later. Live consumers should ignore this
                # field if they don't need it.
                yield Event(
                    "tool_end",
                    {
                        "name": name,
                        "call_num": tool_call_count,
                        "result_chars": len(result),
                        "result_preview": preview,
                        "result_full": result,
                    },
                ).as_dict()

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

                if tool_call_count >= self.max_tool_calls:
                    yield Event(
                        "progress",
                        {"message": f"Tool budget reached ({self.max_tool_calls}). Forcing final answer."},
                    ).as_dict()
                    break

            if tool_call_count >= self.max_tool_calls:
                force_final_triggered = True
                obs_info(
                    "fs_research.force_final.triggered",
                    tool_calls_used=tool_call_count,
                    max_tool_calls=self.max_tool_calls,
                    llm_rounds=llm_call_num,
                )
                # ── CRITICAL: stub any unmatched tool_calls before the force-final call ──
                # If the agent emitted N parallel tool_calls in its last turn but we
                # broke out after executing only K of them (because the budget hit
                # mid-batch), the messages array is malformed: the assistant turn
                # has N tool_call_ids but only K matching tool responses. OpenAI
                # rejects this with: "An assistant message with 'tool_calls' must
                # be followed by tool messages responding to each 'tool_call_id'".
                # Fix: append stub tool responses for every remaining tool_call_id.
                _executed_ids = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
                last_assistant = None
                for m in reversed(messages):
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        last_assistant = m
                        break
                if last_assistant:
                    for tc in last_assistant.get("tool_calls", []):
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        if tc_id and tc_id not in _executed_ids:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "(skipped — agent tool budget exhausted before this call ran)",
                            })

                # Force final-answer call: no tools allowed
                yield Event("llm_call", {"call_num": llm_call_num + 1, "phase": "force_final"}).as_dict()
                final_resp = None
                with span(
                    "fs_research.force_final",
                    model=active_model,
                    message_count=len(messages),
                    tool_calls_used=tool_call_count,
                ):
                  for _attempt in range(5):
                    try:
                        final_resp = await self._client.chat.completions.create(
                            model=active_model,
                            messages=messages
                            + [
                                {
                                    "role": "user",
                                    "content": (
                                        "Tool budget exhausted. Write the best answer you can "
                                        "with the evidence already gathered. Cite every claim with "
                                        "`path:line`. Note any gaps explicitly."
                                    ),
                                }
                            ],
                            temperature=0,
                            max_completion_tokens=self.max_completion_tokens,
                        )
                        break
                    except openai.RateLimitError as e:
                        wait = _wait_from_429(e, _attempt)
                        logger.warning(f"force-final 429; sleeping {wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue
                    except Exception as e:
                        # Log the real error but surface a generic message to the user.
                        logger.exception(f"Force-final call failed: {e}")
                        yield Event("error", {
                            "message": "Sorry — something went wrong while finalizing the answer. Please try again, or rephrase the question more narrowly.",
                        }).as_dict()
                        return
                if final_resp is None:
                    logger.error("Force-final rate-limited after retries")
                    yield Event("error", {
                        "message": "Sorry — the model is rate-limited right now. Please try again in a moment.",
                    }).as_dict()
                    return
                final_answer = final_resp.choices[0].message.content or ""
                break

        elapsed = time.time() - start

        # Stream the final answer in word-sized chunks (so a CLI can pretend it's live)
        if final_answer:
            words = final_answer.split(" ")
            for i, w in enumerate(words):
                sep = " " if i < len(words) - 1 else ""
                yield Event("token", {"content": w + sep}).as_dict()
                if i % 20 == 0:
                    await asyncio.sleep(0)

        obs_info(
            "fs_research.run.complete",
            tool_calls=tool_call_count,
            llm_calls=llm_call_num + (1 if force_final_triggered else 0),
            answer_chars=len(final_answer),
            elapsed_s=round(elapsed, 2),
            elapsed_ms=int(elapsed * 1000),
            force_final=force_final_triggered,
        )

        yield Event(
            "result",
            {
                "answer": final_answer,
                "tool_calls": tool_call_count,
                "llm_calls": llm_call_num + (1 if tool_call_count >= self.max_tool_calls else 0),
                "elapsed_s": round(elapsed, 2),
            },
        ).as_dict()
