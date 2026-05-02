#!/usr/bin/env python3
"""
OrchestratorAgent tools — pure Python, no LangChain, no LangGraph.

Each tool is a plain async function.
TOOL_SCHEMAS defines the OpenAI function-calling schemas.
make_tool_executor() returns a dict {tool_name -> async callable}.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger("rag_system")


# ─────────────────────────────────────────────────────────────────────────────
# Citation accumulator (same logic as deep_agent_tools.CitationAccumulator)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CitationAccumulator:
    """Collects and globally renumbers citations across all tool calls in a request."""

    sec_citations: List[Dict[str, Any]] = field(default_factory=list)
    transcript_citations: List[Dict[str, Any]] = field(default_factory=list)
    news_citations: List[Dict[str, Any]] = field(default_factory=list)

    _sec_offset: int = 0
    _tc_offset: int = 0

    def add_sec(self, citations: List[Dict[str, Any]]) -> int:
        offset = self._sec_offset
        remapped = []
        for i, c in enumerate(citations):
            c = dict(c)
            c["marker"] = f"[C-{self._sec_offset + i + 1}]"
            remapped.append(c)
        self.sec_citations.extend(remapped)
        self._sec_offset += len(citations)
        return offset

    def add_transcript(self, citations: List[Dict[str, Any]]) -> int:
        offset = self._tc_offset
        remapped = []
        for i, c in enumerate(citations):
            c = dict(c)
            c["marker"] = f"[TC-{self._tc_offset + i + 1}]"
            remapped.append(c)
        self.transcript_citations.extend(remapped)
        self._tc_offset += len(citations)
        return offset

    def add_news(self, articles: List[Dict[str, Any]]) -> None:
        for i, a in enumerate(articles):
            self.news_citations.append(
                {
                    "type": "news",
                    "marker": f"[N{len(self.news_citations) + i + 1}]",
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "published_date": a.get("published_date", ""),
                }
            )

    def all_citations(self) -> List[Dict[str, Any]]:
        return self.sec_citations + self.transcript_citations + self.news_citations


# ─────────────────────────────────────────────────────────────────────────────
# Citation marker remapping helpers
# ─────────────────────────────────────────────────────────────────────────────


def _remap_sec_markers(text: str, offset: int) -> str:
    if offset == 0:
        return text

    def shift(m):
        return f"[C-{int(m.group(1)) + offset}]"

    return re.sub(r"\[C-(\d+)\]", shift, text)


def _remap_tc_markers(text: str, offset: int) -> str:
    if offset == 0:
        return text

    def shift(m):
        return f"[TC-{int(m.group(1)) + offset}]"

    return re.sub(r"\[TC-(\d+)\]", shift, text)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI function-calling schemas
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": (
                "Search a company's SEC 10-K annual filing for financial data, risk factors, "
                "business segments, MD&A, balance sheet, or any annual report information. "
                "Returns analysis with [C-N] citation markers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "question": {
                        "type": "string",
                        "description": "Specific question to answer from the 10-K",
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Fiscal year (e.g. 2024). Omit for most recent.",
                    },
                },
                "required": ["ticker", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_earnings_transcripts",
            "description": (
                "Search a company's earnings call transcripts for management commentary, guidance, "
                "analyst Q&A, quarterly performance, or strategic plans. "
                "Returns analysis with [TC-N] citation markers. "
                "Only pass quarters that exist in the database (listed in your system prompt)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "question": {
                        "type": "string",
                        "description": "Specific question to answer from transcripts",
                    },
                    "quarters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Quarters to search, e.g. ['2024_q4', '2024_q3']. Omit for recent quarters.",
                    },
                },
                "required": ["ticker", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search for the latest news about a company or topic using Tavily. "
                "Use for current events, recent announcements, or to supplement SEC/transcript data. "
                "Returns summaries with [N-N] citation markers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'Apple AI strategy 2025'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_companies",
            "description": (
                "Screen and discover companies matching a qualitative investment thesis or theme. "
                "Use when the user asks to 'find companies', 'screen for', or 'which companies are...'. "
                "Returns a list of relevant companies with evidence from SEC filings and transcripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Investment theme or screening criteria",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top companies to return (default: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_research_notes",
            "description": (
                "Save intermediate research findings for multi-step synthesis. "
                "Use descriptive keys like 'msft_cloud_2024' or 'comparison_margins'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Unique key for these notes",
                    },
                    "content": {
                        "type": "string",
                        "description": "Research notes in markdown",
                    },
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_research_notes",
            "description": "Read previously saved research notes by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key used when saving the notes",
                    },
                },
                "required": ["key"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor factory
# ─────────────────────────────────────────────────────────────────────────────


def make_tool_executor(
    sec_service,
    transcript_service,
    tavily_service,
    search_engine,
    event_queue: asyncio.Queue,
    citation_accumulator: CitationAccumulator,
    request_id: str,
    config=None,
    qualitative_screener=None,
) -> Dict[str, Any]:
    """
    Returns {tool_name -> async callable} for all tools.
    All callables accept keyword arguments matching their schema.
    """
    _scratch: Dict[str, str] = {}

    # ── Low-level SEC search ──────────────────────────────────────────────────

    async def search_sec_filings(
        ticker: str, question: str, fiscal_year: Optional[int] = None
    ) -> str:
        ticker = ticker.upper().strip()
        rag_logger.info(
            f"[Tool] SEC: ticker={ticker} year={fiscal_year} q='{question[:60]}'"
        )

        planned_docs = [
            {"ticker": ticker, "fiscal_year": fiscal_year, "filing_type": "10-K"}
        ]
        await event_queue.put(
            {
                "type": "10k_search",
                "message": f"Searching {ticker} {'FY'+str(fiscal_year) if fiscal_year else 'latest'} 10-K...",
                "step": "10k_search",
                "data": {
                    "chunks_found": 0,
                    "tickers_processed": 1,
                    "companies_found": 0,
                    "documents": planned_docs,
                },
            }
        )

        try:
            query_embedding = await search_engine.encode_query_async(question)
        except Exception as e:
            rag_logger.warning(f"[SEC tool] Embedding failed: {e}")
            return f"[Error: could not encode query for {ticker}: {e}]"

        chunks: List[Dict] = []
        sec_answer = ""

        try:
            async for event in sec_service.execute_smart_parallel_search_async(
                query=question,
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year,
                max_iterations=7,
                confidence_threshold=0.85,
                event_yielder=True,
                embedding_function=search_engine.embedding_model.encode,
            ):
                etype = event.get("type", "")
                edata = event.get("data", {})
                if etype == "search_complete":
                    chunks = edata.get("chunks", [])
                    sec_answer = edata.get("answer", "")
                elif etype == "planning_start":
                    await event_queue.put(
                        {
                            "type": "reasoning",
                            "message": f"Analyzing {ticker}'s annual report...",
                            "step": "10k_planning",
                            "data": {"ticker": ticker},
                        }
                    )
                elif etype == "planning_complete":
                    sub_qs = edata.get("sub_questions", [])
                    if sub_qs:
                        await event_queue.put(
                            {
                                "type": "reasoning",
                                "message": "Breaking down question into sub-queries:\n"
                                + "\n".join(f"- {q}" for q in sub_qs[:4]),
                                "step": "10k_planning",
                                "data": {"sub_questions": sub_qs},
                            }
                        )
                elif etype == "retrieval_complete":
                    new_chunks = edata.get("new_chunks", 0)
                    if new_chunks > 0:
                        await event_queue.put(
                            {
                                "type": "reasoning",
                                "message": f"Found {new_chunks} relevant sections in {ticker}'s filing",
                                "step": "10k_retrieval",
                                "data": edata,
                            }
                        )
        except Exception as e:
            rag_logger.warning(f"[SEC tool] Error for {ticker}: {e}")
            return f"[Error searching {ticker} SEC filings: {e}]"

        if not chunks or not sec_answer.strip():
            await event_queue.put(
                {
                    "type": "10k_search",
                    "message": f"No relevant passages found in {ticker} 10-K",
                    "step": "10k_search",
                    "data": {
                        "chunks_found": 0,
                        "tickers_processed": 1,
                        "companies_found": 0,
                        "documents": [],
                    },
                }
            )
            return f"No relevant information found in {ticker}'s SEC 10-K filing."

        raw_citations = sec_service.get_10k_citations(chunks)
        offset = citation_accumulator.add_sec(raw_citations)
        remapped_answer = _remap_sec_markers(sec_answer, offset)

        _seen: set = set()
        _docs = []
        for c in chunks:
            key = (c.get("ticker", ""), c.get("fiscal_year"), c.get("filing_type", "10-K"))
            if key not in _seen:
                _seen.add(key)
                _docs.append(
                    {
                        "ticker": c.get("ticker", ""),
                        "fiscal_year": c.get("fiscal_year"),
                        "filing_type": c.get("filing_type", "10-K"),
                    }
                )

        await event_queue.put(
            {
                "type": "10k_search",
                "message": f"Found {len(chunks)} relevant passages from {ticker} 10-K",
                "step": "10k_search",
                "data": {
                    "chunks_found": len(chunks),
                    "tickers_processed": 1,
                    "companies_found": 1,
                    "documents": _docs,
                },
            }
        )
        rag_logger.info(
            f"[SEC tool] Done: {len(chunks)} chunks, {len(raw_citations)} citations (offset={offset})"
        )
        return remapped_answer

    # ── Low-level Transcript search ───────────────────────────────────────────

    async def search_earnings_transcripts(
        ticker: str, question: str, quarters: Optional[List[str]] = None
    ) -> str:
        ticker = ticker.upper().strip()
        rag_logger.info(
            f"[Tool] Transcripts: ticker={ticker} quarters={quarters} q='{question[:60]}'"
        )

        planned_docs = []
        if quarters:
            for q_str in quarters[:3]:
                parts = q_str.split("_")
                if len(parts) == 2:
                    planned_docs.append(
                        {
                            "ticker": ticker,
                            "year": parts[0],
                            "quarter": parts[1].replace("q", "").replace("Q", ""),
                        }
                    )

        await event_queue.put(
            {
                "type": "search",
                "message": f"Searching {ticker} earnings transcripts...",
                "step": "search",
                "data": {
                    "chunks_found": 0,
                    "tickers_processed": 0,
                    "documents": planned_docs,
                },
            }
        )

        from agent.rag.search_planner import TranscriptSearch

        if quarters:
            effective_quarters = quarters
        elif config is not None:
            available = config.get("available_quarters", [])
            effective_quarters = available[:4] if available else []
        else:
            effective_quarters = []

        ts = TranscriptSearch(ticker=ticker, quarters=effective_quarters, query=question)
        qa_for_service = {
            "tickers": [ticker],
            "extracted_tickers": [ticker],
            "extracted_ticker": ticker,
        }

        result_chunks: List[Dict] = []
        result_answer = ""

        try:
            async for event in transcript_service.execute_search_async(
                query=question,
                question_analysis=qa_for_service,
                transcript_searches=[ts],
            ):
                etype = event.get("type", "")
                edata = event.get("data", {})
                if etype == "search_complete":
                    result_chunks = edata.get("chunks", [])
                    result_answer = edata.get("answer", "")
                elif etype == "retrieval_complete":
                    new_chunks = edata.get("new_chunks", 0)
                    if new_chunks > 0:
                        await event_queue.put(
                            {
                                "type": "reasoning",
                                "message": f"Found {new_chunks} transcript passages for {ticker}",
                                "step": "search",
                                "data": edata,
                            }
                        )
        except Exception as e:
            rag_logger.warning(f"[Transcript tool] Error for {ticker}: {e}")
            return f"[Error searching {ticker} earnings transcripts: {e}]"

        if not result_chunks or not result_answer.strip():
            await event_queue.put(
                {
                    "type": "search",
                    "message": f"No transcript passages found for {ticker}",
                    "step": "search",
                    "data": {"chunks_found": 0, "tickers_processed": 0, "documents": []},
                }
            )
            return f"No relevant earnings transcript information found for {ticker}."

        raw_citations = transcript_service.get_citations(result_chunks)
        offset = citation_accumulator.add_transcript(raw_citations)
        remapped_answer = _remap_tc_markers(result_answer, offset)

        _seen2: set = set()
        _docs2 = []
        for c in result_chunks:
            key = (c.get("ticker", ""), c.get("year", ""), c.get("quarter", ""))
            if key not in _seen2:
                _seen2.add(key)
                _docs2.append(
                    {
                        "ticker": c.get("ticker", ""),
                        "year": c.get("year", ""),
                        "quarter": c.get("quarter", ""),
                    }
                )

        await event_queue.put(
            {
                "type": "search",
                "message": f"Found {len(result_chunks)} transcript passages from {ticker}",
                "step": "search",
                "data": {
                    "chunks_found": len(result_chunks),
                    "tickers_processed": 1,
                    "documents": _docs2,
                },
            }
        )
        rag_logger.info(
            f"[Transcript tool] Done: {len(result_chunks)} chunks, {len(raw_citations)} citations (offset={offset})"
        )
        return remapped_answer

    # ── News tool ─────────────────────────────────────────────────────────────

    async def search_news(query: str) -> str:
        rag_logger.info(f"[Tool] News: q='{query[:80]}'")
        if not tavily_service or not tavily_service.is_available():
            return "News search unavailable (Tavily not configured)."
        try:
            news_results = await asyncio.to_thread(
                tavily_service.search_news, query, max_results=5, include_answer="advanced"
            )
        except Exception as e:
            rag_logger.warning(f"[News tool] Tavily error: {e}")
            return f"[Error searching news: {e}]"

        articles = news_results.get("results", [])
        if not articles:
            return "No recent news found for this query."

        citation_accumulator.add_news(articles)

        try:
            event_queue.put_nowait(
                {
                    "type": "news_search",
                    "message": f"Found {len(articles)} news articles",
                    "step": "news_search",
                    "data": {
                        "articles": [
                            {
                                "title": a.get("title", ""),
                                "url": a.get("url", ""),
                                "published_date": a.get("published_date", ""),
                            }
                            for a in articles
                            if a.get("url")
                        ]
                    },
                }
            )
        except Exception:
            pass

        lines = []
        tavily_answer = news_results.get("answer", "")
        if tavily_answer:
            lines.append(f"**News summary:** {tavily_answer}\n")
        for i, a in enumerate(articles, 1):
            snippet = a.get("content", a.get("snippet", ""))[:300]
            lines.append(
                f"[N{i}] **{a.get('title','Untitled')}** ({a.get('published_date','')})\n"
                f"{a.get('url','')}\n{snippet}\n"
            )
        return "\n".join(lines)

    # ── screen_companies ──────────────────────────────────────────────────────

    async def screen_companies(query: str, top_n: int = 10) -> str:
        rag_logger.info(f"[Tool] screen_companies: q='{query[:80]}' top_n={top_n}")
        if not qualitative_screener:
            return "Company screening is unavailable."

        await event_queue.put(
            {
                "type": "reasoning",
                "message": f"Screening companies for: {query[:80]}...",
                "step": "search_planning",
                "data": {"query": query},
            }
        )

        rows = []
        try:
            async for event in qualitative_screener.screen_with_streaming(
                query=query,
                top_n=top_n,
                sources=["transcripts", "10k"],
            ):
                etype = event.get("type", "")
                if etype == "partial_result":
                    await event_queue.put(
                        {
                            "type": "reasoning",
                            "message": event.get("message", "Screening companies..."),
                            "step": "search",
                            "data": event.get("data", {}),
                        }
                    )
                elif etype == "result":
                    rows = event.get("data", {}).get("rows", [])
        except Exception as e:
            rag_logger.warning(f"[screen_companies] Screener error: {e}")
            return f"[Error screening companies: {e}]"

        if not rows:
            return "No matching companies found for this screening query."

        lines = [f"Found {len(rows)} companies matching '{query}':\n"]
        lines.append("| Ticker | Company | Relevance | Evidence |")
        lines.append("|--------|---------|-----------|----------|")
        for row in rows[:top_n]:
            ticker = row.get("ticker", "")
            company = row.get("company_name", ticker)
            score = row.get("relevance_score", 0)
            summary = (row.get("evidence_summary", "") or "")[:120].replace("|", "-")
            lines.append(f"| {ticker} | {company} | {score}% | {summary} |")
        return "\n".join(lines)

    # ── Notes tools ───────────────────────────────────────────────────────────

    async def write_research_notes(key: str, content: str) -> str:
        _scratch[key] = content
        rag_logger.info(f"[Notes] write key='{key}' ({len(content)} chars)")
        return f"Notes saved under '{key}'."

    async def read_research_notes(key: str) -> str:
        if key not in _scratch:
            rag_logger.info(
                f"[Notes] read key='{key}' → NOT FOUND | available: {list(_scratch.keys())}"
            )
            return f"No notes found for key '{key}'. Available keys: {list(_scratch.keys()) or 'none'}"
        rag_logger.info(f"[Notes] read key='{key}' → {len(_scratch[key])} chars")
        return _scratch[key]

    return {
        "search_sec_filings": search_sec_filings,
        "search_earnings_transcripts": search_earnings_transcripts,
        "search_news": search_news,
        "screen_companies": screen_companies,
        "write_research_notes": write_research_notes,
        "read_research_notes": read_research_notes,
    }
