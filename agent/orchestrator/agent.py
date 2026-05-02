#!/usr/bin/env python3
"""
OrchestratorAgent — pure Python ReAct loop over OpenAI function calling.

No LangChain. No LangGraph. Full control.

Drop-in replacement for DeepRAGAgent:
  - Same __init__ signature
  - Same execute_rag_flow() async generator and SSE event shapes
  - Same set_database_connection() method
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from dotenv import load_dotenv

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

from agent.rag.config import Config
from agent.rag.database_manager import DatabaseManager
from agent.rag.earnings_transcript_service import EarningsTranscriptService
from agent.rag.question_analyzer import QuestionAnalyzer
from agent.rag.rag_utils import deduplicate_citations_and_chunks
from agent.rag.search_engine import SearchEngine
from agent.rag.sec_filings_service_smart_parallel import (
    SmartParallelSECFilingsService as SECFilingsService,
)
from agent.rag.tavily_service import TavilyService

from .tools import CitationAccumulator, TOOL_SCHEMAS, make_tool_executor

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger("rag_system")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """You are a senior financial research analyst with access to specialized tools. Tools within a single response execute in parallel — always batch as many as possible.

Your tools:
1. **search_sec_filings(ticker, question, fiscal_year?)** — search a company's SEC 10-K annual filing
2. **search_earnings_transcripts(ticker, question, quarters?)** — search earnings call transcripts (pass a `quarters` list like ["2025_q4","2025_q3","2025_q2","2025_q1"] to get trend data)
3. **search_news(query)** — real-time news via Tavily
4. **screen_companies(query, top_n?)** — discover companies matching a theme or thesis
5. **write_research_notes(key, content)** — save intermediate findings to your scratch pad
6. **read_research_notes(key)** — retrieve previously saved notes

{data_availability}

## Research strategy: 1 round, then answer

**Use exactly 1 round of tool calls for most questions, 2 rounds maximum.**

Round 1 — fire everything in parallel. For a growth/performance comparison:
- Call `search_earnings_transcripts` for EACH ticker with **all 4-8 recent quarters** (e.g. 2025_q4 through 2024_q1) to capture the full trend
- Call `search_sec_filings` for EACH ticker for the most recent fiscal year
- One call per ticker per source — all in the same turn

After round 1 results come back, **write your final answer immediately.** Only do a second round if a specific critical number is completely absent. Never search the same ticker/source twice.

- **Comparison (2+ companies)**: all tickers × sources in one turn → answer
- **One company deep-dive**: `search_sec_filings` + `search_earnings_transcripts` (multi-quarter) in one turn → answer
- **Discovery** ("find companies..."): `screen_companies` → answer
- **Greetings / meta-questions** ("hello", "what can you do?"): respond directly, no tools

## What makes a great answer

For any growth/performance question, a great answer covers:
1. **Trend, not just latest**: show the quarter-by-quarter trajectory (is growth accelerating or decelerating?)
2. **Absolute scale**: revenue in dollars, not just percentages
3. **Margin story**: operating margin if available — a high-growth but unprofitable segment tells a different story
4. **Management commentary**: what did executives say about the drivers? (capacity constraints, AI demand, deal sizes, etc.)
5. **Direct comparison**: a table with both companies side by side across multiple quarters

## Citation rules

Cite every fact inline using markers returned by tools: [C-N] for SEC filings, [TC-N] for transcripts, [N-N] for news.

## Response format

Write a **direct, insightful answer**. Lead with the headline conclusion. Then support it with data. Do not hedge with phrases like "based on the gathered output" or "most recent explicit figure captured" — just state the facts and cite them. Do not add a "Takeaway" section that just repeats the table in prose.

End with:

### Suggested follow-up questions
- [question 1]
- [question 2]
- [question 3]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up question parser
# ─────────────────────────────────────────────────────────────────────────────

_FOLLOWUP_RE = re.compile(
    r"###\s*Suggested follow-up questions\s*\n(.*?)(?:\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_follow_up_questions(text: str):
    m = _FOLLOWUP_RE.search(text)
    if not m:
        return text.strip(), []
    answer_body = text[: m.start()].strip()
    questions = [
        line.strip().lstrip("-•*").strip()
        for line in m.group(1).splitlines()
        if line.strip().lstrip("-•*").strip()
    ]
    return answer_body, questions[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Tool call announcement helper
# ─────────────────────────────────────────────────────────────────────────────


def _make_tool_announce(
    tool_name: str, tool_args: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    ticker = str(tool_args.get("ticker", "")).upper()
    question_snippet = str(tool_args.get("question", tool_args.get("query", "")))[:80]
    fiscal_year = tool_args.get("fiscal_year")

    if tool_name == "search_sec_filings":
        fy_str = f" FY{fiscal_year}" if fiscal_year else ""
        return {
            "type": "reasoning",
            "message": f"Looking up {ticker}{fy_str} in SEC annual filings...",
            "step": "10k_planning",
            "data": {"tool": tool_name, "ticker": ticker},
        }
    elif tool_name == "search_earnings_transcripts":
        quarters = tool_args.get("quarters") or []
        q_str = f" ({', '.join(quarters[:2])})" if quarters else ""
        return {
            "type": "reasoning",
            "message": f"Searching {ticker} earnings transcripts{q_str}...",
            "step": "search_planning",
            "data": {"tool": tool_name, "ticker": ticker},
        }
    elif tool_name == "search_news":
        return {
            "type": "reasoning",
            "message": f'Searching news: "{question_snippet}"',
            "step": "news_search",
            "data": {"tool": tool_name, "query": question_snippet},
        }
    elif tool_name == "screen_companies":
        return {
            "type": "reasoning",
            "message": f'Screening companies for: "{question_snippet}"',
            "step": "search",
            "data": {"tool": tool_name, "query": question_snippet},
        }
    elif tool_name == "write_research_notes":
        key = tool_args.get("key", "")
        return {
            "type": "reasoning",
            "message": f"Saving intermediate findings ({key})...",
            "step": "analysis",
            "data": {"tool": tool_name, "key": key},
        }
    elif tool_name == "read_research_notes":
        key = tool_args.get("key", "")
        return {
            "type": "reasoning",
            "message": f"Retrieving saved notes ({key})...",
            "step": "analysis",
            "data": {"tool": tool_name, "key": key},
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorAgent
# ─────────────────────────────────────────────────────────────────────────────


class OrchestratorAgent:
    """
    Pure Python ReAct orchestrator. Drop-in replacement for DeepRAGAgent.

    Uses openai.AsyncOpenAI directly — no LangChain, no LangGraph.
    ReAct loop: call OpenAI (non-streaming) → execute tool calls in parallel
    → repeat until no more tool calls → fake-stream final answer.
    """

    MAX_TOOL_CALLS = 10

    def __init__(self, openai_api_key: Optional[str] = None):
        self.instance_id = f"OrchestratorAgent_{int(time.time() * 1000)}"
        logger.info(f"🚀 Creating OrchestratorAgent: {self.instance_id}")

        self.config = Config()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        try:
            self.config.fetch_available_quarters_from_db()
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch quarters from DB: {e}")

        self.database_manager = DatabaseManager(self.config)
        self.question_analyzer = QuestionAnalyzer(None, self.config, self.database_manager)
        self.search_engine = SearchEngine(self.config, self.database_manager)
        self.tavily_service = TavilyService()
        self.sec_service = SECFilingsService(self.database_manager, self.config)
        self.transcript_service = EarningsTranscriptService(self.search_engine, self.config)

        self._model = os.getenv("DEEP_AGENT_MODEL", "gpt-5.4-nano-2026-03-17")
        self._reasoning_effort = os.getenv("DEEP_AGENT_REASONING_EFFORT", "medium")  # low/medium/high/none
        self._max_completion_tokens = int(os.getenv("DEEP_AGENT_MAX_TOKENS", "32000"))
        self._openai = openai.AsyncOpenAI(api_key=self.openai_api_key)

        # Screener injected after init by lifespan (optional)
        self._qualitative_screener = None

        logger.info(
            f"✅ OrchestratorAgent ready (model={self._model}, "
            f"reasoning_effort={self._reasoning_effort}, "
            f"max_tokens={self._max_completion_tokens})"
        )

    # ── Injection points ──────────────────────────────────────────────────────

    def set_qualitative_screener(self, screener) -> None:
        """Called by lifespan after screener is initialized."""
        self._qualitative_screener = screener
        logger.info("✅ OrchestratorAgent: qualitative screener injected")

    def set_database_connection(self, db_connection) -> None:
        """Pass request-scoped DB connection to ConversationMemory."""
        self.question_analyzer.conversation_memory.set_database_connection(db_connection)

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute_rag_flow(
        self,
        question: str,
        show_details: bool = False,
        comprehensive: bool = True,
        stream_callback=None,
        max_iterations: Optional[int] = None,
        conversation_id: Optional[str] = None,
        stream: bool = True,
        **_extra: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the ReAct flow.
        Yields SSE event dicts identical to RAGAgent / DeepRAGAgent.
        Final event is always {type: 'result', data: {...}} or {type: 'error', ...}.
        """
        start_time = time.time()
        request_id = f"{self.instance_id}_{int(start_time * 1000)}"

        rag_logger.info("=" * 80)
        rag_logger.info("[Orchestrator] ▶ NEW REQUEST")
        rag_logger.info(f"[Orchestrator]   request_id : {request_id}")
        rag_logger.info(f"[Orchestrator]   question   : {question[:120]}")
        rag_logger.info(f"[Orchestrator]   conv_id    : {conversation_id}")
        rag_logger.info("=" * 80)

        if stream:
            yield {"type": "progress", "message": "Starting analysis...", "step": "init", "data": {}}

        # Per-request state
        citation_acc = CitationAccumulator()
        event_queue: asyncio.Queue = asyncio.Queue()
        messages = await self._build_messages(question, conversation_id)
        budget = max_iterations or self.MAX_TOOL_CALLS

        tool_executor = make_tool_executor(
            sec_service=self.sec_service,
            transcript_service=self.transcript_service,
            tavily_service=self.tavily_service,
            search_engine=self.search_engine,
            event_queue=event_queue,
            citation_accumulator=citation_acc,
            request_id=request_id,
            config=self.config,
            qualitative_screener=self._qualitative_screener,
        )

        # Shared mutable state written by background task, read after it finishes
        _state: Dict[str, Any] = {"final_answer": None, "tool_call_count": 0, "llm_call_num": 0}
        sentinel = object()

        # reasoning_effort="none" means treat as plain chat (temperature allowed)
        _use_reasoning = self._reasoning_effort and self._reasoning_effort != "none"

        async def _react_loop():
            try:
                # ── Phase 1: planning call (tool_choice="none") ──────────────
                # Force the LLM to lay out its research plan as text before
                # touching any tools. This prevents multi-round scatter.
                _state["llm_call_num"] += 1
                rag_logger.info(
                    f"[Orchestrator] Planning call (messages={len(messages)})"
                )
                plan_kwargs: Dict[str, Any] = dict(
                    model=self._model,
                    messages=messages + [
                        {
                            "role": "user",
                            "content": (
                                "Before calling any tools, write a specific research plan (2-4 sentences). "
                                "State: which tickers, which sources (10-K / transcripts / news), and which "
                                "years/quarters you'll search. For growth/trend questions, plan to search "
                                "transcripts across ALL available quarters (2025_q4, 2025_q3, 2025_q2, 2025_q1, "
                                "2024_q4, 2024_q3) to capture the full trajectory — not just the latest quarter. "
                                "You will execute ALL of these in parallel in the very next turn."
                            ),
                        }
                    ],
                    tools=TOOL_SCHEMAS,
                    tool_choice="none",
                    max_completion_tokens=1024,
                )
                # gpt-5.4-nano rejects reasoning_effort when tools are present
                # on /v1/chat/completions — fall back to temperature=0.
                plan_kwargs["temperature"] = 0
                try:
                    plan_resp = await self._openai.chat.completions.create(**plan_kwargs)
                    plan_text = plan_resp.choices[0].message.content or ""
                    rag_logger.info(f"[Orchestrator] Plan: {plan_text[:200]}")
                    if plan_text and stream:
                        await event_queue.put({
                            "type": "reasoning",
                            "message": f"Plan: {plan_text}",
                            "step": "planning",
                            "data": {"plan": plan_text},
                        })
                    # Inject the plan into the conversation so the execution
                    # call sees it as context
                    messages.append({"role": "assistant", "content": plan_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Good. Now execute all the searches you described in parallel — "
                            "call every tool at once. For transcripts, pass all the quarters you listed "
                            "(e.g. ['2025_q4','2025_q3','2025_q2','2025_q1','2024_q4','2024_q3']) "
                            "in a single call per ticker — not one call per quarter."
                        ),
                    })
                except Exception as e:
                    rag_logger.warning(f"[Orchestrator] Planning call failed (non-fatal): {e}")

                # ── Phase 2+: ReAct loop (tool_choice="auto") ─────────────────
                while _state["tool_call_count"] < budget:
                    _state["llm_call_num"] += 1
                    rag_logger.info(
                        f"[Orchestrator] LLM call #{_state['llm_call_num']} "
                        f"(messages={len(messages)}, tools_used={_state['tool_call_count']})"
                    )

                    call_kwargs: Dict[str, Any] = dict(
                        model=self._model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto",
                        max_completion_tokens=self._max_completion_tokens,
                    )
                    # gpt-5.4-nano rejects reasoning_effort when tools are present
                    # on /v1/chat/completions — fall back to temperature=0.
                    call_kwargs["temperature"] = 0

                    try:
                        response = await self._openai.chat.completions.create(**call_kwargs)
                    except Exception as e:
                        rag_logger.error(f"[Orchestrator] OpenAI error: {e}")
                        await event_queue.put(("error", str(e)))
                        return

                    choice = response.choices[0]
                    msg = choice.message
                    finish_reason = choice.finish_reason
                    tool_calls = msg.tool_calls or []

                    rag_logger.info(
                        f"[Orchestrator] finish_reason={finish_reason}, "
                        f"tool_calls={len(tool_calls)}"
                    )

                    # Append assistant message to history (plain dict, no LangChain)
                    assistant_dict: Dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content,
                    }
                    if tool_calls:
                        assistant_dict["tool_calls"] = [
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
                    messages.append(assistant_dict)

                    if not tool_calls:
                        # No tool calls → final answer
                        _state["final_answer"] = msg.content or ""
                        rag_logger.info(
                            f"[Orchestrator] Final answer: {len(_state['final_answer'])} chars"
                        )
                        break

                    # Announce and execute tool calls in parallel
                    _state["tool_call_count"] += len(tool_calls)

                    for tc in tool_calls:
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}
                        announce = _make_tool_announce(tc.function.name, fn_args)
                        if announce and stream:
                            await event_queue.put(announce)

                    async def _run_one(tc):
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}

                        fn = tool_executor.get(fn_name)
                        if fn is None:
                            result_text = f"[Unknown tool: {fn_name}]"
                            rag_logger.warning(f"[Orchestrator] Unknown tool: {fn_name}")
                        else:
                            try:
                                rag_logger.info(
                                    f"[Orchestrator] ── Executing {fn_name}"
                                    f"({', '.join(f'{k}={str(v)[:40]}' for k, v in fn_args.items())})"
                                )
                                result_text = await fn(**fn_args)
                            except Exception as e:
                                rag_logger.warning(f"[Orchestrator] Tool {fn_name} error: {e}")
                                result_text = f"[Error in {fn_name}: {e}]"

                        return {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result_text),
                        }

                    tool_results = await asyncio.gather(
                        *[_run_one(tc) for tc in tool_calls]
                    )
                    messages.extend(tool_results)

                else:
                    # Budget exceeded — ask for a final answer with what we have
                    rag_logger.warning(
                        f"[Orchestrator] Tool budget exceeded ({_state['tool_call_count']}), "
                        "forcing final answer"
                    )
                    try:
                        force_msgs = messages + [
                            {
                                "role": "user",
                                "content": (
                                    "Write your final answer now using all the research above. "
                                    "Be comprehensive and detailed — include: "
                                    "(1) a clear headline conclusion, "
                                    "(2) specific revenue figures and YoY growth rates with citations, "
                                    "(3) quarter-by-quarter trend if the data shows it, "
                                    "(4) operating margin or profitability trajectory, "
                                    "(5) key management commentary or strategic drivers, "
                                    "(6) a direct side-by-side comparison. "
                                    "Do not truncate. Do not add a 'Takeaway' section that repeats the table. "
                                    "End with the Suggested follow-up questions section."
                                ),
                            }
                        ]
                        force_call_kwargs: Dict[str, Any] = dict(
                            model=self._model,
                            messages=force_msgs,
                            max_completion_tokens=self._max_completion_tokens,
                        )
                        if _use_reasoning:
                            force_call_kwargs["reasoning_effort"] = self._reasoning_effort
                        else:
                            force_call_kwargs["temperature"] = 0
                        force_resp = await self._openai.chat.completions.create(
                            **force_call_kwargs
                        )
                        _state["final_answer"] = force_resp.choices[0].message.content or ""
                    except Exception as e:
                        rag_logger.error(f"[Orchestrator] Force-final error: {e}")
                        _state["final_answer"] = (
                            "Research complete — see gathered information above."
                        )

            except Exception as e:
                rag_logger.error(f"[Orchestrator] ReAct loop error: {e}", exc_info=True)
                await event_queue.put(("error", str(e)))
            finally:
                await event_queue.put(sentinel)

        react_task = asyncio.create_task(_react_loop())

        # Drain events from the queue while the loop runs
        try:
            while True:
                item = await event_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, tuple) and item[0] == "error":
                    yield {
                        "type": "error",
                        "message": f"Agent error: {item[1]}",
                        "step": "generation",
                        "data": {"error": item[1]},
                    }
                    await react_task
                    return
                if isinstance(item, dict) and stream:
                    yield item
        finally:
            await react_task

        final_answer = _state["final_answer"]
        tool_call_count = _state["tool_call_count"]

        if not final_answer:
            yield {
                "type": "error",
                "message": "Agent did not produce a final answer.",
                "step": "generation",
                "data": {},
            }
            return

        answer_body, follow_up_questions = _extract_follow_up_questions(final_answer)
        rag_logger.info(
            f"[Orchestrator] Answer body: {len(answer_body)} chars | "
            f"follow-up questions: {len(follow_up_questions)}"
        )

        # Fake-stream the final answer token-by-token
        if stream:
            yield {
                "type": "progress",
                "message": "Generating response...",
                "step": "generation",
                "data": {},
            }
            words = answer_body.split(" ")
            for i, word in enumerate(words):
                sep = " " if i < len(words) - 1 else ""
                yield {"type": "token", "content": word + sep, "step": "generation", "data": {}}
                if i % 8 == 0:
                    await asyncio.sleep(0)

        # Deduplicate citations
        all_cits = citation_acc.all_citations()
        unique_citations, _ = deduplicate_citations_and_chunks(all_cits, [], rag_logger)

        total_time = time.time() - start_time

        # Update conversation memory
        if conversation_id:
            self.question_analyzer.add_to_conversation_memory(
                conversation_id, question, "user"
            )
            self.question_analyzer.add_to_conversation_memory(
                conversation_id, answer_body, "assistant"
            )

        rag_logger.info("=" * 80)
        rag_logger.info("[Orchestrator] ✅ COMPLETE")
        rag_logger.info(f"[Orchestrator]   total_time  : {total_time:.2f}s")
        rag_logger.info(f"[Orchestrator]   tool_calls  : {tool_call_count}")
        rag_logger.info(f"[Orchestrator]   citations   : {len(unique_citations)}")
        rag_logger.info(f"[Orchestrator]   answer_len  : {len(answer_body)} chars")
        rag_logger.info("=" * 80)

        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "orchestrator.flow.complete",
                total_time_ms=int(total_time * 1000),
                tool_calls=tool_call_count,
                citations=len(unique_citations),
                answer_length=len(answer_body),
            )

        yield {
            "type": "result",
            "message": "Response generated successfully",
            "step": "complete",
            "data": {
                "success": True,
                "response": {
                    "answer": answer_body,
                    "confidence": 0.9,
                    "citations": unique_citations,
                    "context_chunks": [],
                    "iterations": [],
                    "total_iterations": tool_call_count,
                    "follow_up_questions_asked": follow_up_questions,
                    "accumulated_chunks_count": 0,
                    "answer_mode": "standard",
                },
                "chunks": [],
                "analysis": {},
                "timing": {
                    "analysis": 0.0,
                    "search": 0.0,
                    "generation": total_time,
                    "total": total_time,
                },
            },
        }

    async def execute_rag_flow_async(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming wrapper — returns final result dict."""
        kwargs["stream"] = False
        final_result = None
        async for event in self.execute_rag_flow(question, **kwargs):
            if event.get("type") == "result":
                final_result = event.get("data")
        return final_result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        try:
            data_availability = self.config.get_quarter_context_for_llm()
        except Exception:
            data_availability = "Data availability information not loaded."
        current_date = datetime.now().strftime("%B %d, %Y")
        return _SYSTEM_PROMPT_BASE.format(
            data_availability=f"Today's date: {current_date}\n\n{data_availability}"
        )

    async def _build_messages(
        self, question: str, conversation_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build plain-dict message list (no LangChain objects)."""
        system_prompt = self._build_system_prompt()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        if conversation_id:
            try:
                history_str = await self.question_analyzer.conversation_memory.format_context(
                    conversation_id
                )
                if history_str:
                    for line in history_str.strip().splitlines():
                        if line.startswith("👤 User:"):
                            messages.append(
                                {
                                    "role": "user",
                                    "content": line[len("👤 User:") :].strip(),
                                }
                            )
                        elif line.startswith("🤖 Assistant:"):
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": line[len("🤖 Assistant:") :].strip(),
                                }
                            )
            except Exception as e:
                logger.warning(f"⚠️ Could not load conversation history: {e}")

        messages.append({"role": "user", "content": question})
        rag_logger.info(
            f"[Orchestrator] Built {len(messages)} input messages "
            f"(1 system + {len(messages)-2} history + 1 user)"
        )
        return messages
