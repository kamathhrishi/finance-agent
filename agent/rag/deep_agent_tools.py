#!/usr/bin/env python3
"""
DeepAgent Tool Wrappers

Wraps our three data services (SEC filings, earnings transcripts, Tavily news)
as LangChain tools that the DeepAgent ReAct loop can call.

Each tool:
  - Is a closure over the service instances (injected via make_tools factory)
  - Puts intermediate streaming events into a shared asyncio.Queue
  - Returns a citation-marked text string to LangGraph
  - Registers its citations in a CitationAccumulator for final assembly
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ─────────────────────────────────────────────────────────────────────────────
# Citation accumulator (per-request)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CitationAccumulator:
    """
    Collects citations from all tool calls during a single agent request.

    Citation marker renaming:
      SEC tool returns [C-1], [C-2], … but multiple SEC calls would collide.
      We remap each call's markers globally so the final answer stays consistent.
      Transcript tool returns [TC-1], [TC-2], … — same treatment.
    """
    sec_citations: List[Dict[str, Any]] = field(default_factory=list)
    transcript_citations: List[Dict[str, Any]] = field(default_factory=list)
    news_citations: List[Dict[str, Any]] = field(default_factory=list)

    # Running offsets for renumbering
    _sec_offset: int = 0
    _tc_offset: int = 0

    def add_sec(self, citations: List[Dict[str, Any]]) -> int:
        """Add SEC citations; returns the offset used (for remapping in answer text)."""
        offset = self._sec_offset
        remapped = []
        for i, c in enumerate(citations):
            c = dict(c)
            c['marker'] = f"[C-{self._sec_offset + i + 1}]"
            remapped.append(c)
        self.sec_citations.extend(remapped)
        self._sec_offset += len(citations)
        return offset

    def add_transcript(self, citations: List[Dict[str, Any]]) -> int:
        """Add transcript citations; returns offset used."""
        offset = self._tc_offset
        remapped = []
        for i, c in enumerate(citations):
            c = dict(c)
            c['marker'] = f"[TC-{self._tc_offset + i + 1}]"
            remapped.append(c)
        self.transcript_citations.extend(remapped)
        self._tc_offset += len(citations)
        return offset

    def add_news(self, articles: List[Dict[str, Any]]) -> None:
        for i, a in enumerate(articles):
            self.news_citations.append({
                "type": "news",
                "marker": f"[N{len(self.news_citations) + i + 1}]",
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "published_date": a.get("published_date", ""),
            })

    def all_citations(self) -> List[Dict[str, Any]]:
        return self.sec_citations + self.transcript_citations + self.news_citations


# ─────────────────────────────────────────────────────────────────────────────
# Citation text remapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remap_sec_markers(text: str, offset: int) -> str:
    """Shift [C-N] markers in text by offset → [C-(N+offset)]."""
    if offset == 0:
        return text
    def shift(m):
        n = int(m.group(1))
        return f"[C-{n + offset}]"
    return re.sub(r'\[C-(\d+)\]', shift, text)


def _remap_tc_markers(text: str, offset: int) -> str:
    """Shift [TC-N] markers in text by offset → [TC-(N+offset)]."""
    if offset == 0:
        return text
    def shift(m):
        n = int(m.group(1))
        return f"[TC-{n + offset}]"
    return re.sub(r'\[TC-(\d+)\]', shift, text)


# ─────────────────────────────────────────────────────────────────────────────
# Tool argument schemas (Pydantic) — needed for StructuredTool
# ─────────────────────────────────────────────────────────────────────────────

class SECFilingsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'AAPL' or 'MSFT'")
    question: str = Field(description="Specific question to answer from the 10-K filing")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year (e.g. 2024). Defaults to most recent.")


class TranscriptInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'AAPL' or 'MSFT'")
    question: str = Field(description="Specific question to answer from earnings call transcripts")
    quarters: Optional[List[str]] = Field(None, description="Quarters to search, e.g. ['2024_q4', '2024_q3']. Defaults to recent quarters.")


class NewsInput(BaseModel):
    query: str = Field(description="Search query for news, e.g. 'Apple AI strategy 2025'")


class WriteNotesInput(BaseModel):
    key: str = Field(description="Unique key for these notes, e.g. 'apple_revenue_2023'")
    content: str = Field(description="The research notes to save (markdown)")


class ReadNotesInput(BaseModel):
    key: str = Field(description="Key used when saving the notes")


class ResearchCompanyInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to research, e.g. 'AAPL'")
    question: str = Field(description="Specific question to answer for this company")


class ResearchCompaniesInput(BaseModel):
    tickers: List[str] = Field(description="List of stock ticker symbols to research in parallel, e.g. ['MSFT', 'GOOGL']")
    question: str = Field(description="Research question to answer for each company")


# ─────────────────────────────────────────────────────────────────────────────
# Tool factory
# ─────────────────────────────────────────────────────────────────────────────

def make_tools(
    sec_service,
    transcript_service,
    tavily_service,
    search_engine,
    event_queue: asyncio.Queue,
    citation_accumulator: CitationAccumulator,
    request_id: str,
    question_analysis: Optional[Dict[str, Any]] = None,
    config=None,
) -> List[StructuredTool]:
    """
    Create per-request LangChain StructuredTools.

    All tools share:
      - event_queue: intermediate streaming events go here
      - citation_accumulator: citations collected across all tool calls
      - request_id: used to scope the in-memory scratch pad
    """

    # ── In-memory scratch pad for this request ──
    _scratch: Dict[str, str] = {}

    # ── SEC Filings Tool ──────────────────────────────────────────────────────

    async def _search_sec_filings(ticker: str, question: str, fiscal_year: Optional[int] = None) -> str:
        ticker = ticker.upper().strip()
        rag_logger.info(f"[DeepAgent] SEC tool called: ticker={ticker}, year={fiscal_year}, q='{question[:60]}'")

        # Immediate "searching" event with document chip
        planned_docs = [{'ticker': ticker, 'fiscal_year': fiscal_year, 'filing_type': '10-K'}]
        await event_queue.put({
            'type': '10k_search',
            'message': f"Searching {ticker} {'FY'+str(fiscal_year) if fiscal_year else 'latest'} 10-K filing...",
            'step': '10k_search',
            'data': {'chunks_found': 0, 'tickers_processed': 1, 'companies_found': 0, 'documents': planned_docs},
        })

        rag_logger.info(f"[SEC tool] Encoding query...")
        try:
            query_embedding = await search_engine.encode_query_async(question)
            rag_logger.info(f"[SEC tool] Query encoded ({len(query_embedding)} dims)")
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
                etype = event.get('type', '')
                edata = event.get('data', {})

                if etype == 'search_complete':
                    chunks = edata.get('chunks', [])
                    sec_answer = edata.get('answer', '')
                elif etype == 'planning_start':
                    await event_queue.put({
                        'type': 'reasoning',
                        'message': f"Looking at {ticker}'s annual report...",
                        'step': '10k_planning',
                        'event_name': '10k_planning_start',
                        'data': {'ticker': ticker, 'phase': 'planning'},
                    })
                elif etype == 'planning_complete':
                    sub_qs = edata.get('sub_questions', [])
                    if sub_qs:
                        questions_text = "\n".join([f"- {q}" for q in sub_qs[:4]])
                        await event_queue.put({
                            'type': 'reasoning',
                            'message': f"To answer this, I need to find:\n{questions_text}",
                            'step': '10k_planning',
                            'event_name': '10k_sub_questions',
                            'data': {'sub_questions': sub_qs},
                        })
                elif etype == 'retrieval_complete':
                    new_chunks = edata.get('new_chunks', 0)
                    if new_chunks > 0:
                        await event_queue.put({
                            'type': 'reasoning',
                            'message': f"Found {new_chunks} relevant sections in the filing",
                            'step': '10k_retrieval',
                            'event_name': '10k_retrieval_progress',
                            'data': edata,
                        })
                elif etype == 'evaluation_complete':
                    quality = edata.get('quality_score', 0)
                    missing = edata.get('missing_info', [])
                    if quality >= 0.8:
                        await event_queue.put({
                            'type': 'reasoning',
                            'message': "I have enough information to answer this question",
                            'step': '10k_evaluation',
                            'event_name': '10k_evaluation_complete',
                            'data': edata,
                        })
                    elif missing:
                        await event_queue.put({
                            'type': 'reasoning',
                            'message': f"Still looking for: {missing[0] if missing else 'more details'}...",
                            'step': '10k_evaluation',
                            'event_name': '10k_evaluation_progress',
                            'data': edata,
                        })
        except Exception as e:
            rag_logger.warning(f"[DeepAgent] SEC service error for {ticker}: {e}")
            return f"[Error searching {ticker} SEC filings: {e}]"

        rag_logger.info(f"[SEC tool] Service finished: {len(chunks)} chunks, answer={len(sec_answer)} chars")
        if not chunks or not sec_answer.strip():
            rag_logger.warning(f"[SEC tool] No results for {ticker}")
            await event_queue.put({
                'type': '10k_search',
                'message': f"Found 0 relevant passages from {ticker}",
                'step': '10k_search',
                'data': {'chunks_found': 0, 'tickers_processed': 1, 'companies_found': 0, 'documents': []},
            })
            return f"No relevant information found in {ticker}'s SEC 10-K filing."

        # Remap citation markers globally
        raw_citations = sec_service.get_10k_citations(chunks)
        offset = citation_accumulator.add_sec(raw_citations)
        remapped_answer = _remap_sec_markers(sec_answer, offset)

        # Final chips event
        _seen = set()
        _docs = []
        for c in chunks:
            key = (c.get('ticker', ''), c.get('fiscal_year'), c.get('filing_type', '10-K'))
            if key not in _seen:
                _seen.add(key)
                _docs.append({'ticker': c.get('ticker', ''), 'fiscal_year': c.get('fiscal_year'), 'filing_type': c.get('filing_type', '10-K')})

        await event_queue.put({
            'type': '10k_search',
            'message': f"Found {len(chunks)} relevant passages from {ticker}",
            'step': '10k_search',
            'data': {'chunks_found': len(chunks), 'tickers_processed': 1, 'companies_found': 1, 'documents': _docs},
        })

        rag_logger.info(f"[DeepAgent] SEC tool done: {len(chunks)} chunks, {len(raw_citations)} citations (offset={offset})")
        return remapped_answer

    # ── Earnings Transcript Tool ─────────────────────────────────────────────

    async def _search_earnings_transcripts(
        ticker: str, question: str, quarters: Optional[List[str]] = None
    ) -> str:
        ticker = ticker.upper().strip()
        rag_logger.info(f"[DeepAgent] Transcript tool called: ticker={ticker}, quarters={quarters}, q='{question[:60]}'")

        # Immediate event with document chips
        planned_docs = []
        if quarters:
            for q_str in quarters[:3]:
                parts = q_str.split('_')
                if len(parts) == 2:
                    planned_docs.append({'ticker': ticker, 'year': parts[0], 'quarter': parts[1].replace('q', '').replace('Q', '')})

        await event_queue.put({
            'type': 'search',
            'message': f"Searching {ticker} earnings transcripts...",
            'step': 'search',
            'data': {'chunks_found': 0, 'tickers_processed': 0, 'documents': planned_docs},
        })

        # Build a minimal transcript search spec (same as TranscriptSearch in search_planner)
        # We create a lightweight object matching what execute_search_async expects
        from agent.rag.search_planner import TranscriptSearch

        # Default to the 4 most recent quarters from the live DB config
        if quarters:
            effective_quarters = quarters
        elif config is not None:
            available = config.get('available_quarters', [])
            effective_quarters = available[:4] if available else []
        else:
            effective_quarters = []
        rag_logger.info(f"[Transcript tool] Using quarters: {effective_quarters}")

        ts = TranscriptSearch(
            ticker=ticker,
            quarters=effective_quarters,
            query=question,
        )

        # Minimal question_analysis for the service
        qa = question_analysis or {}
        qa_for_service = {
            'tickers': [ticker],
            'extracted_tickers': [ticker],
            'extracted_ticker': ticker,
            **{k: v for k, v in qa.items() if k not in ('tickers', 'extracted_tickers', 'extracted_ticker')},
        }

        result_chunks: List[Dict] = []
        result_answer = ""

        try:
            async for event in transcript_service.execute_search_async(
                query=question,
                question_analysis=qa_for_service,
                transcript_searches=[ts],
            ):
                etype = event.get('type', '')
                edata = event.get('data', {})

                if etype == 'search_complete':
                    result_chunks = edata.get('chunks', [])
                    result_answer = edata.get('answer', '')
                elif etype == 'planning_start':
                    await event_queue.put({
                        'type': 'reasoning',
                        'message': f"Planning transcript search for {ticker}...",
                        'step': 'search_planning',
                        'data': {},
                    })
                elif etype == 'retrieval_complete':
                    new_chunks = edata.get('new_chunks', 0)
                    if new_chunks > 0:
                        await event_queue.put({
                            'type': 'reasoning',
                            'message': f"Found {new_chunks} transcript passages",
                            'step': 'search',
                            'data': edata,
                        })
        except Exception as e:
            rag_logger.warning(f"[DeepAgent] Transcript service error for {ticker}: {e}")
            return f"[Error searching {ticker} earnings transcripts: {e}]"

        rag_logger.info(f"[Transcript tool] Service finished: {len(result_chunks)} chunks, answer={len(result_answer)} chars")
        if not result_chunks or not result_answer.strip():
            rag_logger.warning(f"[Transcript tool] No results for {ticker}")
            await event_queue.put({
                'type': 'search',
                'message': f"Found 0 transcript passages for {ticker}",
                'step': 'search',
                'data': {'chunks_found': 0, 'tickers_processed': 0, 'documents': []},
            })
            return f"No relevant earnings transcript information found for {ticker}."

        # Remap citation markers globally
        raw_citations = transcript_service.get_citations(result_chunks)
        offset = citation_accumulator.add_transcript(raw_citations)
        remapped_answer = _remap_tc_markers(result_answer, offset)

        # Final chips event
        _seen = set()
        _docs = []
        for c in result_chunks:
            key = (c.get('ticker', ''), c.get('year', ''), c.get('quarter', ''))
            if key not in _seen:
                _seen.add(key)
                _docs.append({'ticker': c.get('ticker', ''), 'year': c.get('year', ''), 'quarter': c.get('quarter', '')})

        await event_queue.put({
            'type': 'search',
            'message': f"Found {len(result_chunks)} transcript passages from {ticker}",
            'step': 'search',
            'data': {'chunks_found': len(result_chunks), 'tickers_processed': 1, 'documents': _docs},
        })

        rag_logger.info(f"[DeepAgent] Transcript tool done: {len(result_chunks)} chunks, {len(raw_citations)} citations (offset={offset})")
        return remapped_answer

    # ── News Tool ────────────────────────────────────────────────────────────

    def _search_news(query: str) -> str:
        rag_logger.info(f"[News tool] Called: q='{query[:80]}'")

        if not tavily_service.is_available():
            return "News search unavailable (Tavily not configured)."

        try:
            news_results = tavily_service.search_news(query, max_results=5, include_answer="advanced")
        except Exception as e:
            rag_logger.warning(f"[DeepAgent] Tavily error: {e}")
            return f"[Error searching news: {e}]"

        articles = news_results.get("results", [])
        if not articles:
            rag_logger.info(f"[News tool] No articles found for: '{query[:60]}'")
            return "No recent news found for this query."

        rag_logger.info(f"[News tool] Found {len(articles)} articles for: '{query[:60]}'")
        # Register citations
        citation_accumulator.add_news(articles)

        # Build article chips — put in queue via run_coroutine_threadsafe OR use sync queue.put_nowait
        _article_chips = [
            {'title': a.get('title', ''), 'url': a.get('url', ''), 'published_date': a.get('published_date', '')}
            for a in articles if a.get('url')
        ]
        # News tool is sync; use put_nowait (queue is unbounded so this is safe)
        try:
            event_queue.put_nowait({
                'type': 'news_search',
                'message': f"Found {len(articles)} news articles",
                'step': 'news_search',
                'data': {'articles': _article_chips},
            })
        except Exception:
            pass  # If queue is full, skip the event — answer still works

        # Format answer text
        lines = []
        for i, a in enumerate(articles, 1):
            title = a.get('title', 'Untitled')
            url = a.get('url', '')
            snippet = a.get('content', a.get('snippet', ''))[:300]
            date = a.get('published_date', '')
            lines.append(f"[N{i}] **{title}** ({date})\n{url}\n{snippet}\n")

        tavily_answer = news_results.get("answer", "")
        if tavily_answer:
            lines.insert(0, f"**News summary:** {tavily_answer}\n")

        return "\n".join(lines)

    # ── Scratch pad Tools ────────────────────────────────────────────────────

    def _write_research_notes(key: str, content: str) -> str:
        _scratch[key] = content
        rag_logger.info(f"[Notes tool] write key='{key}' ({len(content)} chars) | scratch_pad keys now: {list(_scratch.keys())}")
        return f"Notes saved under '{key}'."

    def _read_research_notes(key: str) -> str:
        if key not in _scratch:
            rag_logger.info(f"[Notes tool] read key='{key}' → NOT FOUND | available: {list(_scratch.keys())}")
            return f"No notes found for key '{key}'. Available keys: {list(_scratch.keys()) or 'none'}"
        rag_logger.info(f"[Notes tool] read key='{key}' → {len(_scratch[key])} chars")
        return _scratch[key]

    # ── Assemble StructuredTools ─────────────────────────────────────────────

    sec_tool = StructuredTool.from_function(
        coroutine=_search_sec_filings,
        name="search_sec_filings",
        description=(
            "Search a company's SEC 10-K annual filing for financial data, risk factors, "
            "business segments, MD&A, balance sheet, income statement, or any annual report info. "
            "Returns analysis with [C-N] citation markers you should include in your answer."
        ),
        args_schema=SECFilingsInput,
    )

    transcript_tool = StructuredTool.from_function(
        coroutine=_search_earnings_transcripts,
        name="search_earnings_transcripts",
        description=(
            "Search a company's earnings call transcripts for management commentary, guidance, "
            "forward-looking statements, analyst Q&A, quarterly performance details, or strategic plans. "
            "Returns analysis with [TC-N] citation markers you should include in your answer. "
            "IMPORTANT: Only pass quarters that exist in the database (listed in your system prompt). "
            "Do NOT invent quarter labels — if a quarter is not in the system prompt's available list, it does not exist."
        ),
        args_schema=TranscriptInput,
    )

    news_tool = StructuredTool.from_function(
        func=_search_news,
        name="search_news",
        description=(
            "Search for the latest news about a company or topic using Tavily. "
            "Use this for current events, recent announcements, or anything that happened recently. "
            "Returns article summaries with [N-N] citation markers."
        ),
        args_schema=NewsInput,
    )

    write_notes_tool = StructuredTool.from_function(
        func=_write_research_notes,
        name="write_research_notes",
        description=(
            "Save intermediate research findings so you can reference them later without re-searching. "
            "Use descriptive keys like 'aapl_revenue_2024' or 'comparison_margins'."
        ),
        args_schema=WriteNotesInput,
    )

    read_notes_tool = StructuredTool.from_function(
        func=_read_research_notes,
        name="read_research_notes",
        description="Read previously saved research notes by key.",
        args_schema=ReadNotesInput,
    )

    return [sec_tool, transcript_tool, news_tool, write_notes_tool, read_notes_tool]


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor tool factory — spawns subagents
# ─────────────────────────────────────────────────────────────────────────────

def make_supervisor_tools(
    llm,
    sec_service,
    transcript_service,
    tavily_service,
    search_engine,
    event_queue: asyncio.Queue,
    citation_accumulator: CitationAccumulator,
    request_id: str,
    question_analysis: Optional[Dict[str, Any]] = None,
    config=None,
) -> List[StructuredTool]:
    """
    Create supervisor-level tools.

    The supervisor gets:
      - research_company(ticker, question)    — single focused subagent
      - research_companies(tickers, question) — parallel subagents via asyncio.gather
      - search_news                           — direct Tavily (no subagent needed)
      - write_research_notes / read_research_notes

    Each subagent is a fresh create_react_agent with only search_sec_filings +
    search_earnings_transcripts from a dedicated make_tools() call (so _scratch
    dicts never collide between tickers).

    All subagents share the same event_queue and citation_accumulator so
    citations are globally numbered and events stream to the frontend.
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, SystemMessage

    _scratch: Dict[str, str] = {}

    # ── Subagent runner ───────────────────────────────────────────────────────

    async def _run_subagent(ticker: str, question: str) -> str:
        """Spin a focused ReAct subagent for one company (SEC + transcripts only)."""
        rag_logger.info(f"[Supervisor] Spawning subagent for {ticker}: '{question[:80]}'")

        # Fresh make_tools call per ticker — isolated _scratch, shared citations + queue
        all_tools = make_tools(
            sec_service=sec_service,
            transcript_service=transcript_service,
            tavily_service=tavily_service,
            search_engine=search_engine,
            event_queue=event_queue,
            citation_accumulator=citation_accumulator,
            request_id=f"{request_id}_{ticker}",
            question_analysis=question_analysis,
            config=config,
        )
        subagent_tools = all_tools[:2]  # sec + transcript only

        subagent = create_react_agent(llm, subagent_tools)

        task = (
            f"Research {ticker} to answer the following question:\n\n{question}\n\n"
            "Search SEC filings AND earnings transcripts. "
            "Return a detailed, comprehensive answer with all [C-N] and [TC-N] citation markers. "
            "Do NOT include a follow-up questions section."
        )

        try:
            result = await subagent.ainvoke(
                {"messages": [HumanMessage(content=task)]},
                config={"recursion_limit": 30},
            )
        except Exception as e:
            rag_logger.warning(f"[Supervisor] Subagent error for {ticker}: {e}")
            return f"[Error researching {ticker}: {e}]"

        # Walk messages in reverse to find the final AI text response
        for msg in reversed(result.get("messages", [])):
            content = getattr(msg, "content", None)
            if not content:
                continue
            if isinstance(content, str) and content.strip():
                rag_logger.info(f"[Supervisor] Subagent {ticker} done: {len(content)} chars")
                return content
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                joined = " ".join(parts).strip()
                if joined:
                    rag_logger.info(f"[Supervisor] Subagent {ticker} done: {len(joined)} chars")
                    return joined

        rag_logger.warning(f"[Supervisor] Subagent {ticker} returned no content")
        return f"No relevant information found for {ticker}."

    # ── research_company ──────────────────────────────────────────────────────

    async def _research_company(ticker: str, question: str) -> str:
        ticker = ticker.upper().strip()
        await event_queue.put({
            'type': 'reasoning',
            'message': f"Researching {ticker}...",
            'step': 'search_planning',
            'event_name': 'subagent_start',
            'data': {'ticker': ticker},
        })
        return await _run_subagent(ticker, question)

    # ── research_companies (parallel) ─────────────────────────────────────────

    async def _research_companies(tickers: List[str], question: str) -> str:
        tickers = [t.upper().strip() for t in tickers]
        rag_logger.info(f"[Supervisor] research_companies (parallel): {tickers}")

        await event_queue.put({
            'type': 'reasoning',
            'message': f"Researching {', '.join(tickers)} in parallel...",
            'step': 'search_planning',
            'event_name': 'subagent_parallel_start',
            'data': {'tickers': tickers},
        })

        results = await asyncio.gather(
            *[_run_subagent(t, question) for t in tickers],
            return_exceptions=True,
        )

        parts = []
        for ticker, res in zip(tickers, results):
            if isinstance(res, Exception):
                rag_logger.warning(f"[Supervisor] Subagent {ticker} raised: {res}")
                parts.append(f"## {ticker}\n[Research error: {res}]")
            else:
                parts.append(f"## {ticker}\n\n{res}")

        return "\n\n---\n\n".join(parts)

    # ── News tool (direct — no subagent) ─────────────────────────────────────

    def _search_news(query: str) -> str:
        rag_logger.info(f"[Supervisor] News tool called: q='{query[:80]}'")
        if not tavily_service.is_available():
            return "News search unavailable (Tavily not configured)."
        try:
            news_results = tavily_service.search_news(query, max_results=5, include_answer="advanced")
        except Exception as e:
            rag_logger.warning(f"[Supervisor] Tavily error: {e}")
            return f"[Error searching news: {e}]"

        articles = news_results.get("results", [])
        if not articles:
            return "No recent news found for this query."

        citation_accumulator.add_news(articles)
        try:
            event_queue.put_nowait({
                'type': 'news_search',
                'message': f"Found {len(articles)} news articles",
                'step': 'news_search',
                'data': {'articles': [
                    {'title': a.get('title', ''), 'url': a.get('url', ''), 'published_date': a.get('published_date', '')}
                    for a in articles if a.get('url')
                ]},
            })
        except Exception:
            pass

        lines = []
        for i, a in enumerate(articles, 1):
            snippet = a.get('content', a.get('snippet', ''))[:300]
            lines.append(f"[N{i}] **{a.get('title','Untitled')}** ({a.get('published_date','')})\n{a.get('url','')}\n{snippet}\n")
        tavily_answer = news_results.get("answer", "")
        if tavily_answer:
            lines.insert(0, f"**News summary:** {tavily_answer}\n")
        return "\n".join(lines)

    # ── Notes tools ───────────────────────────────────────────────────────────

    def _write_research_notes(key: str, content: str) -> str:
        _scratch[key] = content
        rag_logger.info(f"[Supervisor] write key='{key}' ({len(content)} chars)")
        return f"Notes saved under '{key}'."

    def _read_research_notes(key: str) -> str:
        if key not in _scratch:
            return f"No notes found for key '{key}'. Available keys: {list(_scratch.keys()) or 'none'}"
        rag_logger.info(f"[Supervisor] read key='{key}' → {len(_scratch[key])} chars")
        return _scratch[key]

    # ── Assemble StructuredTools ──────────────────────────────────────────────

    research_company_tool = StructuredTool.from_function(
        coroutine=_research_company,
        name="research_company",
        description=(
            "Research a single company using its SEC 10-K filings and earnings call transcripts. "
            "Spawns a focused subagent and returns a detailed, cited answer. "
            "Use for single-company deep dives."
        ),
        args_schema=ResearchCompanyInput,
    )

    research_companies_tool = StructuredTool.from_function(
        coroutine=_research_companies,
        name="research_companies",
        description=(
            "Research multiple companies IN PARALLEL using SEC filings and earnings transcripts. "
            "Each company gets its own focused subagent running concurrently. "
            "Always use this (not repeated research_company calls) for multi-company comparisons."
        ),
        args_schema=ResearchCompaniesInput,
    )

    news_tool = StructuredTool.from_function(
        func=_search_news,
        name="search_news",
        description=(
            "Search for the latest news about a company or topic using Tavily. "
            "Use for current events, recent announcements, or to supplement SEC/transcript data. "
            "Returns article summaries with [N-N] citation markers."
        ),
        args_schema=NewsInput,
    )

    write_notes_tool = StructuredTool.from_function(
        func=_write_research_notes,
        name="write_research_notes",
        description=(
            "Save intermediate research findings so you can reference them later. "
            "Use descriptive keys like 'msft_cloud_2024' or 'comparison_margins'."
        ),
        args_schema=WriteNotesInput,
    )

    read_notes_tool = StructuredTool.from_function(
        func=_read_research_notes,
        name="read_research_notes",
        description="Read previously saved research notes by key.",
        args_schema=ReadNotesInput,
    )

    return [research_company_tool, research_companies_tool, news_tool, write_notes_tool, read_notes_tool]
