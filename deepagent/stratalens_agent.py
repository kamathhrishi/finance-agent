"""
StrataLens Deep Research Agent — powered by LangChain DeepAgents.

Uses the deepagents harness with three domain tools:
  - search_10k:           RAG search over SEC 10-K filings + exhibits
  - search_transcript:    RAG search over earnings call transcripts
  - screen_stocks:        Natural-language financial screener

Streams reasoning events (tool calls, tool results, service-level planning/retrieval
events, and final LLM answer) to stdout.

Usage:
    python stratalens_agent.py
    python stratalens_agent.py --query "Compare NVDA and AMD data center revenue in 2024"
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root so agent imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import BaseModel, Field

from agent.rag.config import Config
from agent.rag.database_manager import DatabaseManager
from agent.rag.earnings_transcript_service import EarningsTranscriptService
from agent.rag.search_engine import SearchEngine
from agent.rag.sec_filings_service_smart_parallel import SmartParallelSECFilingsService


# ─────────────────────────────────────────────
# Retry-aware LLM
# ─────────────────────────────────────────────

class RetryingChatOpenAI(ChatOpenAI):
    """ChatOpenAI with exponential-backoff retry on 429 rate limits."""

    def _generate(self, *args, **kwargs):
        for attempt in range(8):
            try:
                return super()._generate(*args, **kwargs)
            except RateLimitError:
                if attempt == 7:
                    raise
                wait = min(2 ** attempt + random.random(), 60)
                print(f"\n⏳ Rate limited — waiting {wait:.1f}s (retry {attempt + 1}/7)...")
                time.sleep(wait)


# ─────────────────────────────────────────────
# Service singletons (initialized once)
# ─────────────────────────────────────────────

_config: Optional[Config] = None
_db_manager: Optional[DatabaseManager] = None
_search_engine: Optional[SearchEngine] = None
_sec_service: Optional[SmartParallelSECFilingsService] = None
_transcript_service: Optional[EarningsTranscriptService] = None
_screener = None


def _get_services():
    global _config, _db_manager, _search_engine, _sec_service, _transcript_service, _screener
    if _config is None:
        print("🔧 Initializing StrataLens services...")
        _config = Config()
        _db_manager = DatabaseManager(_config)
        _search_engine = SearchEngine(_config, _db_manager)
        _sec_service = SmartParallelSECFilingsService(_db_manager, _config)
        _transcript_service = EarningsTranscriptService(_search_engine, _config)
        # Lazy import to avoid circular import via app.__init__
        from agent.screener_agent import ScreenerAgent
        _screener = ScreenerAgent()
        print("✅ Services ready\n")
    return _sec_service, _transcript_service, _screener


# ─────────────────────────────────────────────
# Async helpers
# ─────────────────────────────────────────────

def _run_async(coro):
    """Run a coroutine from sync context (safe to call from any thread)."""
    # Tools run inside langgraph's ThreadPoolExecutor — always create a fresh loop
    return asyncio.run(coro)


def _collect_events(async_gen) -> List[Dict]:
    """Drain an async generator, printing intermediate events, and return all events."""
    async def _collect():
        events = []
        async for event in async_gen:
            events.append(event)
            _print_service_event(event)
        return events
    return _run_async(_collect())


def _print_service_event(event: Dict):
    """Print a service-level streaming event."""
    t = event.get("type", "")
    msg = event.get("message", "")
    data = event.get("data", {})

    if t == "planning_start":
        print(f"   📋 Planning: {msg}")
    elif t == "planning_complete":
        subs = data.get("sub_questions", [])
        print(f"   📋 Plan ready — {len(subs)} sub-questions")
    elif t == "iteration_start":
        it = data.get("iteration", "?")
        nq = data.get("num_queries", "?")
        print(f"   🔍 Iteration {it}: running {nq} queries in parallel")
    elif t == "retrieval_complete":
        nc = data.get("new_chunks", 0)
        tc = data.get("total_chunks", 0)
        print(f"   📄 Retrieved {nc} new chunks (total {tc})")
    elif t == "evaluation_complete":
        qs = data.get("quality_score", "?")
        print(f"   ✅ Quality score: {qs}")
    elif t == "search_complete":
        iters = data.get("iterations", "?")
        qs = data.get("quality_score", "?")
        print(f"   🏁 Search complete in {iters} iteration(s), quality={qs}")
    elif t in ("screener_start", "screener_complete", "screener_error", "reasoning"):
        if msg:
            print(f"   📊 {msg}")


# ─────────────────────────────────────────────
# Tool: search_10k
# ─────────────────────────────────────────────

class Search10KInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'NVDA'")
    query: str = Field(description="Financial question to answer from SEC 10-K filings")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year to target, e.g. 2024. Omit to search all years.")


def _search_10k(ticker: str, query: str, fiscal_year: Optional[int] = None) -> str:
    """Search SEC 10-K filings + exhibits for a company and return a cited answer."""
    sec_service, _, _ = _get_services()

    async def _run():
        # Get embedding for the query
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(query)

        events = []
        async for event in sec_service.execute_smart_parallel_search_async(
            query=query,
            query_embedding=embedding,
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            max_iterations=5,
            confidence_threshold=0.8,
        ):
            events.append(event)
            _print_service_event(event)
        return events

    events = _run_async(_run())

    # Extract final answer + citations
    final = next((e for e in reversed(events) if e.get("type") == "search_complete"), None)
    if not final:
        return json.dumps({"error": "No answer found in 10-K filings."})

    data = final.get("data", {})
    answer = data.get("answer", "")
    chunks = data.get("chunks", [])
    citations = sec_service.get_10k_citations(chunks)

    return json.dumps({
        "answer": answer,
        "citations": citations,
        "quality_score": data.get("quality_score"),
        "iterations": data.get("iterations"),
    }, default=str)


search_10k_tool = StructuredTool(
    name="search_10k",
    description=(
        "Search NVIDIA's SEC 10-K annual filings and exhibits for financial data, "
        "risk factors, business segments, executive compensation, insider trading policy, etc. "
        "Returns a cited answer with [10K-N] citation markers."
    ),
    func=_search_10k,
    args_schema=Search10KInput,
)


# ─────────────────────────────────────────────
# Tool: search_transcript
# ─────────────────────────────────────────────

class SearchTranscriptInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'NVDA'")
    query: str = Field(description="Question to answer from earnings call transcripts")
    year: Optional[int] = Field(None, description="Target year, e.g. 2024")
    quarter: Optional[int] = Field(None, description="Target quarter 1-4. Omit to search all quarters.")


@dataclass
class TranscriptSearch:
    ticker: str
    quarters: List[str]  # e.g. ["2024_q1", "2024_q2"]


def _search_transcript(ticker: str, query: str, year: Optional[int] = None, quarter: Optional[int] = None) -> str:
    """Search earnings call transcripts for a company and return a cited answer."""
    _, transcript_service, _ = _get_services()
    db_manager = _db_manager

    # Build quarter list
    if year and quarter:
        quarters = [f"{year}_q{quarter}"]
    elif year:
        quarters = [f"{year}_q{q}" for q in range(1, 5)]
    else:
        quarters = db_manager.get_last_n_quarters_for_company(ticker.upper(), n=8)

    transcript_search = TranscriptSearch(ticker=ticker.upper(), quarters=quarters)

    question_analysis = {
        "extracted_tickers": [ticker.upper()],
        "topic": query,
        "time_refs": [str(year)] if year else [],
        "question_type": "single_company",
    }

    async def _run():
        events = []
        async for event in transcript_service.execute_search_async(
            query=query,
            question_analysis=question_analysis,
            transcript_searches=[transcript_search],
        ):
            events.append(event)
            _print_service_event(event)
        return events

    events = _run_async(_run())

    final = next((e for e in reversed(events) if e.get("type") == "search_complete"), None)
    if not final:
        return json.dumps({"error": "No answer found in transcripts."})

    data = final.get("data", {})
    answer = data.get("answer", "")
    chunks = data.get("chunks", [])
    citations = transcript_service.get_citations(chunks)

    return json.dumps({
        "answer": answer,
        "citations": citations,
    }, default=str)


search_transcript_tool = StructuredTool(
    name="search_transcript",
    description=(
        "Search earnings call transcripts for management commentary, guidance, "
        "segment performance, forward-looking statements, and Q&A. "
        "Returns a cited answer with [TC-N] citation markers."
    ),
    func=_search_transcript,
    args_schema=SearchTranscriptInput,
)


# ─────────────────────────────────────────────
# Tool: screen_stocks
# ─────────────────────────────────────────────

class ScreenerInput(BaseModel):
    query: str = Field(description="Natural-language screening question, e.g. 'Find tech companies with revenue > $10B and positive free cash flow'")


def _screen_stocks(query: str) -> str:
    """Run a financial screener query and return matching companies."""
    _, _, screener = _get_services()

    events_log = []

    def stream_cb(event):
        _print_service_event(event)
        events_log.append(event)

    async def _run():
        return await screener.execute_screening_flow(query, stream_callback=stream_cb)

    result = _run_async(_run())

    if not result.get("success"):
        return json.dumps({"error": result.get("answer", "Screener failed.")})

    return json.dumps({
        "answer": result.get("answer", ""),
        "data": result.get("data", []),
        "sql_query": result.get("sql_query", ""),
    }, default=str)


screen_stocks_tool = StructuredTool(
    name="screen_stocks",
    description=(
        "Screen stocks using financial metrics. Use for questions like "
        "'which companies have the highest revenue growth', 'find profitable tech companies', "
        "'compare P/E ratios across sectors', etc."
    ),
    func=_screen_stocks,
    args_schema=ScreenerInput,
)


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are StrataLens, an expert financial research agent with access to:

1. **search_10k** — Search SEC 10-K annual filings and exhibits (risk factors, financials, business segments, insider trading policy, compensation, subsidiaries, etc.)
2. **search_transcript** — Search earnings call transcripts for management commentary, guidance, and Q&A
3. **screen_stocks** — Screen stocks using financial metrics (revenue, profitability, valuation, growth, etc.)

## Research workflow

1. **Plan first** — use write_todos to break multi-part questions into steps
2. **Use the right tool** — 10-K for formal filings/exhibits, transcripts for management tone/guidance, screener for comparative metrics
3. **Cite everything** — answers from search_10k use [10K-N] markers, transcripts use [TC-N] markers. Preserve these in your final answer.
4. **Synthesize** — after gathering from multiple tools, compile into a coherent, structured response

Be precise, data-driven, and always cite the source of every claim.
"""


def build_agent():
    llm = RetryingChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17"),
        reasoning_effort="high",
        use_responses_api=True,
    )
    return create_deep_agent(
        model=llm,
        tools=[search_10k_tool, search_transcript_tool, screen_stocks_tool],
        system_prompt=SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# Streaming runner
# ─────────────────────────────────────────────

def run(query: str):
    _get_services()  # Ensure all services are initialized before agent threads start
    agent = build_agent()
    print(f"\nQuery: {query}\n{'='*60}")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if not update or not isinstance(update, dict):
                continue
            raw = update.get("messages", []) or []
            try:
                messages = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]
            except Exception:
                continue

            for msg in messages:
                # Tool calls
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "?")
                        args = tc.get("args", {})
                        args_preview = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                        print(f"\n🔧 [{name}] {args_preview}")

                # Tool results
                if getattr(msg, "type", None) == "tool":
                    content = msg.content or ""
                    try:
                        parsed = json.loads(content)
                        answer = parsed.get("answer", "")
                        snippet = answer[:200].replace("\n", " ") if answer else str(parsed)[:200]
                    except Exception:
                        snippet = str(content)[:200].replace("\n", " ")
                    print(f"   → {snippet}{'...' if len(snippet) == 200 else ''}")

                # LLM text / reasoning
                if getattr(msg, "type", None) == "ai":
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        print(f"\n💬 {content}")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                print(f"\n💬 {block['text']}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StrataLens Deep Research Agent")
    parser.add_argument(
        "--query",
        default="What were NVDA's key business segments and their revenue in FY2024? Also search recent earnings transcripts for management commentary on data center growth.",
    )
    args = parser.parse_args()
    run(args.query)
