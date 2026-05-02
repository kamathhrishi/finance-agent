"""
Qualitative Company Screener

Two-stage pipeline for screening companies on qualitative criteria
extracted from earnings transcripts and 10-K filings:

1. Optional financial SQL filter -> narrows ticker universe
2. Bulk cross-company RAG search -> finds qualitative matches with LLM-synthesized evidence
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class QualitativeScreener:
    """Orchestrates qualitative screening across earnings transcripts and 10-K filings."""

    # Minimum company count for a quarter to be considered "well-populated"
    MIN_COMPANIES_FOR_DEFAULT = 200

    def __init__(self, rag_system, financial_analyzer=None):
        self.financial_analyzer = financial_analyzer
        self.search_engine = rag_system.search_engine
        self.database_manager = rag_system.database_manager
        self.config = rag_system.config
        self.openai_client = openai.OpenAI()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _get_best_default_quarter(self) -> tuple[int, int]:
        """Return (year, quarter) for the latest well-populated quarter."""
        quarter_details = self.config.get("quarter_details", {})
        available = self.config.get("available_quarters", [])
        for qid in available:
            details = quarter_details.get(qid, {})
            if details.get("company_count", 0) >= self.MIN_COMPANIES_FOR_DEFAULT:
                return details["year"], details["quarter"]
        # Fallback: latest quarter regardless
        if available and quarter_details:
            d = quarter_details[available[0]]
            return d["year"], d["quarter"]
        return datetime.now().year - 1, 4

    def _make_reasoning_event(self, event_type: str, message: str, step: str = "reasoning") -> Dict:
        return {
            'type': 'reasoning',
            'event': {
                'event_type': event_type,
                'message': message,
                'details': {'step': step},
                'timestamp': datetime.now().isoformat(),
            }
        }

    # Data coverage description used in meta-message responses
    _DATA_COVERAGE = (
        "StrataLens covers 500+ publicly listed tech companies (semiconductors, software, fintech, cloud, AI) with:\n"
        "- Earnings call transcripts: Q1 2022 – Q4 2025 (latest available quarter)\n"
        "- 10-K annual filings: FY 2022 – FY 2024\n"
        "You can ask about a single company (e.g. \"$NVDA margins\", \"$AAPL supply chain risks\") "
        "or screen across all companies (e.g. \"which companies are investing in AI infrastructure\").\n"
        "Quantitative financial screening (P/E, revenue filters) is coming soon once real-time market data is integrated."
    )

    # ------------------------------------------------------------------
    # Stage 1: Intent Split (sync, CPU-bound LLM call)
    # Handles ALL message types: research queries, greetings, capability
    # questions, and off-topic messages in a single LLM call.
    # ------------------------------------------------------------------
    def _split_intent(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        default_year, default_quarter = self._get_best_default_quarter()

        system_context = (
            "You are a router for StrataLens, a financial research tool.\n\n"
            f"Data coverage:\n{self._DATA_COVERAGE}\n\n"
            "STEP 1 — Classify the message type:\n"
            "  A) GREETING: hi, hello, hey, how are you, etc.\n"
            "  B) CAPABILITY: what can you do, what data do you have, how does this work, etc.\n"
            "  C) OFF_TOPIC: weather, cooking, personal advice, generic coding help, etc.\n"
            "  D) RESEARCH: any company, stock market, financial, or investment question\n\n"
            "For types A/B/C return JSON with ONLY these keys:\n"
            '  "message_type": "greeting" | "capability" | "off_topic",\n'
            '  "meta_response": string (friendly reply — plain text, no markdown headers)\n\n'
            "For type D return JSON with ALL these keys:\n"
            '  "message_type": "research",\n'
            '  "meta_response": null,\n'
            '  "mode": "financial_only" | "qualitative_only" | "mixed",\n'
            '  "financial": string or null,\n'
            '  "qualitative": string or null,\n'
            '  "source": "transcript" or "10k",\n'
            '  "time_scope": {"year": int, "quarter": int or null},\n'
            '  "user_specified_time": true | false,\n'
            '  "is_multi_company": true | false,\n'
            '  "needs_synthesis": true | false\n\n'
            "Rules for RESEARCH classification:\n"
            "- financial: metrics, ratios, P/E, revenue, market cap (SQL-answerable)\n"
            "- qualitative: themes, strategy, commentary (requires text search in transcripts/10-K)\n"
            "- is_multi_company: true when discovering companies across a theme; false for specific named companies\n"
            "- needs_synthesis: true when the query asks for ANYTHING beyond just listing companies — e.g. analysis, opportunities, risks, trends, insights, overview, breakdown, study, report, deep dive, or any open-ended research question about the space itself. If the user wants to UNDERSTAND the space (not just find who's in it), set true.\n"
            "- user_specified_time: true ONLY if user explicitly mentioned a year, quarter, or time period\n"
            f"- Default time_scope: year={default_year}, quarter={default_quarter}; use quarter=null for 10-K\n\n"
            "IMPORTANT: Err heavily toward RESEARCH. Only use greeting/capability/off_topic when completely certain.\n"
            "You MUST return valid JSON only. No markdown, no explanation, no code fences.\n\n"
            "Examples:\n"
            f'- "hi there" → {{"message_type":"greeting","meta_response":"Hello!...","mode":null,"financial":null,"qualitative":null,"source":null,"time_scope":null,"user_specified_time":false,"is_multi_company":false,"needs_synthesis":false}}\n'
            f'- "companies discussing AI capex" → {{"message_type":"research","meta_response":null,"mode":"qualitative_only","financial":null,"qualitative":"discussing AI capex","source":"transcript","time_scope":{{"year":{default_year},"quarter":{default_quarter}}},"user_specified_time":false,"is_multi_company":true,"needs_synthesis":false}}\n'
            f'- "find companies building agentic finance and analyze opportunities and risks" → {{"message_type":"research","meta_response":null,"mode":"qualitative_only","financial":null,"qualitative":"building agentic finance","source":"transcript","time_scope":{{"year":{default_year},"quarter":{default_quarter}}},"user_specified_time":false,"is_multi_company":true,"needs_synthesis":true}}\n'
            f'- "what did $NVDA say about AI?" → {{"message_type":"research","meta_response":null,"mode":"qualitative_only","financial":null,"qualitative":"AI commentary","source":"transcript","time_scope":{{"year":{default_year},"quarter":{default_quarter}}},"user_specified_time":false,"is_multi_company":false,"needs_synthesis":false}}\n'
        )

        prompt = f"{system_context}\n\nMessage: {question}\n\nReturn JSON only."
        last_error = None
        content = ""

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-5-nano-2025-08-07",
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content or ""

                # Strip markdown code fences if present
                content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
                content = re.sub(r'\s*```$', '', content.strip())
                content = content.strip()

                parsed = json.loads(content)

                # Validate: meta messages need message_type + meta_response
                # research messages need mode
                msg_type = parsed.get("message_type", "research")
                if msg_type in ("greeting", "capability", "off_topic"):
                    if not parsed.get("meta_response"):
                        raise ValueError(f"Missing 'meta_response' for {msg_type}: {content}")
                elif msg_type == "research":
                    if "mode" not in parsed:
                        raise ValueError(f"Missing 'mode' key for research query: {content}")
                else:
                    raise ValueError(f"Unknown message_type '{msg_type}': {content}")

                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(f"Intent split attempt {attempt + 1}/{max_retries} failed: {e}. Raw: {content!r}")
                if attempt < max_retries - 1:
                    prompt = (
                        f"{system_context}\n\nMessage: {question}\n\n"
                        f"Previous attempt returned invalid JSON: {content!r}\n"
                        f"Error: {e}\n\n"
                        "Return ONLY valid JSON, no other text."
                    )
            except Exception as e:
                last_error = e
                logger.warning(f"Intent split attempt {attempt + 1}/{max_retries} API error: {e}")
                if attempt >= max_retries - 1:
                    break

        # All retries failed — assume research query, default to qualitative-only
        logger.error(f"Intent split failed after {max_retries} attempts: {last_error}")
        return {
            "message_type": "research",
            "meta_response": None,
            "mode": "qualitative_only",
            "financial": None,
            "qualitative": question,
            "source": "transcript",
            "time_scope": {"year": default_year, "quarter": default_quarter},
            "user_specified_time": False,
            "is_multi_company": False,
        }

    # ------------------------------------------------------------------
    # Stage 5: LLM Evidence Summary (sync, called via asyncio.to_thread)
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove null bytes and non-printable control characters that break JSON payloads."""
        return ''.join(ch for ch in text if ch >= ' ' or ch in '\n\t')

    def _summarize_evidence(self, ticker: str, source: str, qualitative_query: str, chunks: List[Dict], max_retries: int = 3) -> Dict[str, Any]:
        top_chunks = chunks[:5]
        numbered_excerpts = "\n\n".join(
            f"[{i+1}] {self._clean_text(c['chunk_text'])[:500]}" for i, c in enumerate(top_chunks)
        )

        base_prompt = (
            f"You are evaluating {ticker}'s {source} for relevance to: '{qualitative_query}'.\n\n"
            f"Source excerpts (numbered for citation):\n{numbered_excerpts}\n\n"
            "Return JSON only with these keys:\n"
            '  "relevance_score": 0-100 (strict — only 70+ for strong matches)\n'
            '  "evidence": 2-3 sentence explanation using inline citation markers like [1], [2] to reference the excerpts above. '
            'Include specific quotes or numbers from the sources. Example: "The company discussed X [1] and committed $Y to Z [2]."\n\n'
            "If it doesn't match well, score 20-40 and explain why.\n"
            "Return ONLY valid JSON, no markdown, no code fences."
        )
        prompt = base_prompt
        last_error = None
        content = ""

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-5-nano-2025-08-07",
                    max_completion_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )

                content = response.choices[0].message.content or ""
                if not content:
                    raise ValueError("Empty response from model")

                # Strip markdown code fences
                content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
                content = re.sub(r'\s*```$', '', content.strip()).strip()

                parsed = json.loads(content)
                return {
                    'relevance_score': parsed.get('relevance_score', 0),
                    'evidence': parsed.get('evidence', content),
                }

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(f"Evidence summary attempt {attempt + 1}/{max_retries} for {ticker} failed: {e}. Raw: {content!r}")
                if attempt < max_retries - 1:
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"Previous attempt returned invalid JSON: {content!r}\n"
                        f"Error: {e}\n"
                        "Return ONLY valid JSON."
                    )
            except Exception as e:
                last_error = e
                logger.error(f"Evidence summary attempt {attempt + 1}/{max_retries} for {ticker} API error: {e}")
                if attempt >= max_retries - 1:
                    break

        logger.error(f"Evidence summary failed for {ticker} after {max_retries} attempts: {last_error}")
        return {'relevance_score': 0, 'evidence': 'Unable to generate summary'}

    # ------------------------------------------------------------------
    # Stage 6: Space Synthesis (async streaming)
    # Reads across all scored companies and produces an analysis report.
    # ------------------------------------------------------------------
    async def _synthesize_space(self, topic: str, source_label: str, scored_companies: List[Dict], marker_prefix: str) -> AsyncGenerator[str, None]:
        """Stream a synthesis analysis of the space based on evidence from scored companies."""
        # Collect top chunks from each company (up to 3 per company, top 10 companies)
        evidence_blocks = []
        citation_offset = 0
        for entry in scored_companies[:10]:
            if entry.get('llm_relevance_score', 0) < self.RELEVANCE_THRESHOLD:
                continue
            ticker = entry['ticker']
            chunks = entry.get('chunks', [])[:3]
            if not chunks:
                continue
            excerpts = []
            for chunk in chunks:
                citation_offset += 1
                excerpts.append(f"[{marker_prefix}-{citation_offset}] {self._clean_text(chunk['chunk_text'])[:400]}")
            evidence_blocks.append(f"**{ticker}**:\n" + "\n".join(excerpts))

        if not evidence_blocks:
            yield "No strong matches found to synthesize an analysis from."
            return

        evidence_text = "\n\n".join(evidence_blocks)

        prompt = (
            f"You are a financial analyst. Based on primary source evidence from companies' {source_label}, "
            f"write a concise but insightful analysis of the theme: \"{topic}\".\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Structure your response with these sections (use markdown headers):\n"
            "## Overview\n"
            "## Key Opportunities\n"
            "## Key Risks\n"
            "## Notable Company Initiatives\n\n"
            "Rules:\n"
            "- Use inline citations like [TC-1] or [10K-1] when referencing specific evidence above.\n"
            "- Be specific — quote numbers, product names, and strategic commitments where available.\n"
            "- Keep each section to 3-5 sentences.\n"
            "- Do not speculate beyond what the evidence supports.\n"
        )

        try:
            stream = self.openai_client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1200,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"Synthesis streaming failed: {e}")
            yield "\n\n*(Analysis generation failed — please try again.)*"

    # ------------------------------------------------------------------
    # Main streaming pipeline (async generator)
    # ------------------------------------------------------------------
    RELEVANCE_THRESHOLD = 60  # 0-100 scale; companies scoring >= this are "strong matches"

    async def screen_with_streaming(self, question: str, top_n: Optional[int] = None, page: int = 1, page_size: Optional[int] = None, _parsed_intent: Optional[Dict] = None) -> AsyncGenerator[Dict, None]:
        """Async generator yielding SSE events. Auto-detects financial vs qualitative vs mixed."""
        start_time = time.time()
        top_n = top_n or 20  # default if caller doesn't specify

        # --- Stage 1: Intent Split (skip if already parsed by caller) ---
        yield self._make_reasoning_event("step_start", "Understanding your question...", "intent_split")
        if _parsed_intent is not None:
            intent = _parsed_intent
        else:
            intent = await asyncio.to_thread(self._split_intent, question)

        mode = intent.get("mode", "financial_only")
        financial_part = intent.get("financial")
        qualitative_part = intent.get("qualitative")
        default_year, default_quarter = self._get_best_default_quarter()
        time_scope = intent.get("time_scope", {})

        source = intent.get("source", "transcript")
        needs_synthesis = bool(intent.get("needs_synthesis", False))
        logger.info(f"🔍 Intent: mode={intent.get('mode')}, is_multi={intent.get('is_multi_company')}, needs_synthesis={needs_synthesis}")
        user_specified_time = intent.get("user_specified_time", False)
        year = time_scope.get("year", default_year) if user_specified_time else default_year
        quarter = time_scope.get("quarter", default_quarter) if user_specified_time else default_quarter

        # --- Pure financial: not supported, inform the user ---
        if mode == "financial_only" or not qualitative_part:
            yield self._make_reasoning_event(
                "step_complete",
                "Detected financial-only query",
                "intent_split",
            )
            yield {
                'type': 'result',
                'data': {
                    'success': False,
                    'columns': [],
                    'friendly_columns': {},
                    'data_rows': [],
                    'message': (
                        "Financial screening (metrics, ratios, market cap, etc.) is not available yet. "
                        "Try a qualitative query instead — for example: "
                        "\"companies investing heavily in AI infrastructure\" or "
                        "\"companies with strong ESG commentary in earnings calls\"."
                    ),
                },
            }
            return

        source_label = "earnings transcripts" if source == "transcript" else "10-K filings"
        # For mixed queries, skip the financial filter and treat as qualitative-only
        if financial_part and qualitative_part:
            yield self._make_reasoning_event(
                "step_complete",
                f"Searching {source_label.lower()} (financial filters not yet available)",
                "intent_split",
            )
        else:
            yield self._make_reasoning_event(
                "step_complete",
                f"Searching {source_label.lower()}",
                "intent_split",
            )

        # --- Stage 2: Financial Filter (disabled) ---
        # Search all companies qualitatively; financial filters not yet available.
        ticker_list: List[str] = []

        # --- Stage 3: Bulk RAG Search ---
        company_scope = f"{len(ticker_list)} companies" if ticker_list else "all companies"
        yield self._make_reasoning_event(
            "step_start",
            f"Searching {source_label.lower()} across {company_scope}...",
            "bulk_search",
        )

        # Encode query embedding (CPU-bound, run in thread)
        query_embedding = await asyncio.to_thread(
            self.search_engine.embedding_model.encode, [qualitative_part]
        )
        query_embedding = query_embedding[0]

        # Async bulk search — fetch 100 chunks per company for cross-encoder re-ranking
        if source == "transcript":
            search_quarter = quarter if quarter else 4
            company_chunks = await self.database_manager.bulk_search_transcripts_async(
                query_embedding=query_embedding,
                tickers=ticker_list,
                year=year,
                quarter=search_quarter,
                chunks_per_company=100,  # Fetch 100 for cross-encoder
            )
        else:
            company_chunks = await self.database_manager.bulk_search_10k_async(
                query_embedding=query_embedding,
                tickers=ticker_list,
                fiscal_year=year,
                chunks_per_company=100,  # Fetch 100 for cross-encoder
            )

        # Cap to top_n * 2 companies by best similarity — no need to process all 500+
        max_candidates = top_n * 2
        if len(company_chunks) > max_candidates:
            best_sim = {t: max(c['similarity'] for c in chunks) for t, chunks in company_chunks.items()}
            keep = sorted(best_sim, key=best_sim.get, reverse=True)[:max_candidates]
            company_chunks = {t: company_chunks[t] for t in keep}

        yield self._make_reasoning_event(
            "step_complete",
            f"Found {len(company_chunks)} candidate companies — analyzing top {max_candidates}",
            "bulk_search",
        )

        if not company_chunks:
            yield self._make_reasoning_event("step_error", "No companies found matching your criteria", "aggregate")
            yield {
                'type': 'result',
                'data': {
                    'success': True,
                    'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                    'friendly_columns': {
                        'symbol': 'Symbol',
                        'relevance_score': 'Relevance',
                        'evidence_summary': 'Evidence',
                        'citations': 'Citations',
                    },
                    'data_rows': [],
                    'message': 'No companies matched the qualitative criteria.',
                },
            }
            return

        # --- Stage 3.5: Cross-Encoder Re-ranking ---
        yield self._make_reasoning_event(
            "step_start",
            f"Refining results for better accuracy...",
            "cross_encoder",
        )

        # Re-rank chunks within each company using cross-encoder
        reranked_chunks = {}
        for ticker, chunks in company_chunks.items():
            if not chunks:
                continue

            # Prepare pairs: (query, chunk_text)
            pairs = [[qualitative_part, chunk['chunk_text']] for chunk in chunks]

            # Get cross-encoder scores (CPU-bound, run in thread)
            ce_scores = await asyncio.to_thread(self.cross_encoder.predict, pairs)

            # Add scores and sort
            for i, chunk in enumerate(chunks):
                chunk['cross_encoder_score'] = float(ce_scores[i])

            # Keep top 20 chunks per company
            chunks_sorted = sorted(chunks, key=lambda x: x['cross_encoder_score'], reverse=True)
            reranked_chunks[ticker] = chunks_sorted[:20]

        company_chunks = reranked_chunks

        yield self._make_reasoning_event(
            "step_complete",
            f"Results refined successfully",
            "cross_encoder",
        )

        # --- Stream Partial Results #1: Initial companies with cross-encoder scores ---
        initial_rows = []
        for ticker, chunks in company_chunks.items():
            if chunks:
                # Don't show cross-encoder scores - they'll be replaced by LLM scores
                initial_rows.append({
                    'symbol': ticker,
                    'relevance_score': None,  # Will be filled by LLM
                    'evidence_summary': '...',  # Placeholder
                    'citations': [],
                })

        # Sort and yield initial partial result (None scores last; higher score first)
        initial_rows.sort(key=lambda x: (x['relevance_score'] is None, -(x['relevance_score'] or 0)))
        yield {
            'type': 'partial_result',
            'data': {
                'success': True,
                'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                'friendly_columns': {
                    'symbol': 'Symbol',
                    'relevance_score': 'Relevance',
                    'evidence_summary': 'Evidence',
                    'citations': 'Citations',
                },
                'data_rows': initial_rows[:top_n],  # Show top N immediately
                'stage': 'cross_encoder_complete',
            },
        }

        # --- Stage 4: Aggregate & Rank ---
        yield self._make_reasoning_event("step_start", "Analyzing and ranking results...", "aggregate")

        scored: List[Dict] = []
        for ticker, chunks in company_chunks.items():
            # Use cross-encoder scores for better ranking
            avg_sim = sum(c.get('cross_encoder_score', c.get('similarity', 0)) for c in chunks) / len(chunks)
            scored.append({
                'ticker': ticker,
                'avg_similarity': avg_sim,
                'top_excerpt': chunks[0]['chunk_text'][:300] if chunks else '',
                'chunks': chunks,
            })

        scored.sort(key=lambda x: x['avg_similarity'], reverse=True)
        top_companies = scored[:top_n]

        yield self._make_reasoning_event(
            "step_complete",
            f"Identified top {len(top_companies)} most relevant companies",
            "aggregate",
        )

        # --- Stage 4b: Deep Hybrid Search (Pass 2) ---
        yield self._make_reasoning_event(
            "step_start",
            f"Performing detailed analysis on top {len(top_companies)} companies...",
            "deep_search",
        )

        target_quarter = f"{year}_q{quarter}" if quarter else f"{year}_q4"

        async def _deep_search_one(entry: Dict) -> Dict:
            """Run hybrid search for one company, return updated entry."""
            ticker = entry['ticker']
            try:
                if source == "transcript":
                    deep_chunks = await self.search_engine.follow_up_search_async(
                        question=qualitative_part,
                        has_tickers=True,
                        is_general_question=False,
                        is_multi_ticker=False,
                        tickers_to_process=[ticker],
                        target_quarter=target_quarter,
                        target_quarters=[target_quarter],
                    )
                else:
                    # For 10-K, use the async 10-K search
                    deep_chunks = await self.database_manager.search_10k_filings_async(
                        query_embedding=query_embedding,
                        ticker=ticker,
                        fiscal_year=year,
                    )

                if deep_chunks:
                    # Recompute similarity from the richer hybrid results
                    avg_sim = sum(c.get('similarity', 1 - c.get('distance', 0.5)) for c in deep_chunks) / len(deep_chunks)
                    best_text = deep_chunks[0].get('chunk_text', '')[:300]
                    return {
                        **entry,
                        'avg_similarity': avg_sim,
                        'top_excerpt': best_text,
                        'chunks': deep_chunks[:5],  # Keep top 5 for summarization
                    }
            except Exception as e:
                logger.warning(f"Deep search failed for {ticker}: {e}")
            return entry  # Fallback to Pass 1 results

        deep_results = await asyncio.gather(
            *[_deep_search_one(entry) for entry in top_companies],
            return_exceptions=True,
        )

        for i, result in enumerate(deep_results):
            if isinstance(result, Exception):
                logger.warning(f"Deep search exception for {top_companies[i]['ticker']}: {result}")
            else:
                top_companies[i] = result

        # Re-sort after deep search (scores may have changed)
        top_companies.sort(key=lambda x: x['avg_similarity'], reverse=True)

        yield self._make_reasoning_event(
            "step_complete",
            f"Detailed analysis complete",
            "deep_search",
        )

        # --- Stream Partial Results #2: Updated scores after deep search ---
        partial_rows_refined = []
        for entry in top_companies:
            partial_rows_refined.append({
                'symbol': entry['ticker'],
                'relevance_score': None,  # Will be filled by LLM scoring
                'evidence_summary': '...',  # Still generating
                'citations': [],
            })

        yield {
            'type': 'partial_result',
            'data': {
                'success': True,
                'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                'friendly_columns': {
                    'symbol': 'Symbol',
                    'relevance_score': 'Relevance',
                    'evidence_summary': 'Evidence',
                    'citations': 'Citations',
                },
                'data_rows': partial_rows_refined,
                'stage': 'deep_search_complete',
            },
        }

        # --- Stage 5: LLM Evidence Summary — parallel with early stopping ---
        yield self._make_reasoning_event(
            "step_start",
            f"Scoring {len(top_companies)} companies for relevance...",
            "summarize",
        )

        async def _summarize_one(entry: Dict) -> tuple:
            result = await asyncio.to_thread(
                self._summarize_evidence,
                entry['ticker'], source_label, qualitative_part, entry['chunks'],
            )
            return entry, result

        futures = [asyncio.ensure_future(_summarize_one(e)) for e in top_companies]
        found_relevant = 0
        completed = 0
        scored_companies: List[Dict] = []

        try:
            for fut in asyncio.as_completed(futures):
                entry, summary = await fut
                completed += 1

                if isinstance(summary, dict):
                    entry['llm_relevance_score'] = summary.get('relevance_score', 0)
                    entry['evidence_summary'] = summary.get('evidence', '')
                else:
                    entry['llm_relevance_score'] = 0
                    entry['evidence_summary'] = ''

                scored_companies.append(entry)

                if entry['llm_relevance_score'] >= self.RELEVANCE_THRESHOLD:
                    found_relevant += 1

                yield self._make_reasoning_event(
                    "step_progress",
                    f"Reviewed {completed}/{len(top_companies)} companies — {found_relevant} strong match{'es' if found_relevant != 1 else ''} so far",
                    "summarize",
                )

                # Early stopping: already have enough strong matches
                if found_relevant >= top_n:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    yield self._make_reasoning_event(
                        "step_complete",
                        f"Found {found_relevant} relevant companies — stopping early",
                        "summarize",
                    )
                    break
            else:
                yield self._make_reasoning_event(
                    "step_complete",
                    f"Analysis complete — {found_relevant} relevant {'company' if found_relevant == 1 else 'companies'} found out of {completed} reviewed",
                    "summarize",
                )
        finally:
            for f in futures:
                if not f.done():
                    f.cancel()

        top_companies = scored_companies
        # Re-sort by LLM relevance score (more accurate than cross-encoder)
        top_companies.sort(key=lambda x: x.get('llm_relevance_score', 0), reverse=True)

        # --- Stage 6: Yield Result ---
        elapsed = time.time() - start_time
        data_rows = []
        for entry in top_companies:
            # Format sources — index matches [1],[2] markers in evidence_summary
            # Use typed markers so frontend scroll/highlight works correctly
            marker_prefix = "TC" if source == "transcript" else "10K"
            sources = []
            for idx, chunk in enumerate(entry.get('chunks', [])[:5], 1):
                sources.append({
                    'index': idx,                        # corresponds to [idx] in evidence_summary
                    'marker': f"[{marker_prefix}-{idx}]",  # e.g. [TC-1], [10K-1]
                    'type': source,
                    'source_type': source,
                    'ticker': entry['ticker'],
                    'chunk_text': chunk.get('chunk_text', ''),
                    'similarity': chunk.get('similarity', chunk.get('cross_encoder_score', 0)),
                    'year': year,
                    'quarter': quarter if source == 'transcript' else None,
                    'fiscal_year': year if source == '10k' else None,
                    'section': chunk.get('section') or chunk.get('sec_section'),
                    'section_title': chunk.get('sec_section_title'),
                })

            data_rows.append({
                'symbol': entry['ticker'],
                'relevance_score': entry.get('llm_relevance_score', 0) / 100,  # Convert 0-100 to 0-1
                'evidence_summary': entry.get('evidence_summary', ''),
                'citations': sources,  # Citations in separate column
            })

        # --- Stage 7: Space Synthesis (optional, streamed as tokens) ---
        synthesis_text = ""
        marker_prefix = "TC" if source == "transcript" else "10K"
        if needs_synthesis and top_companies:
            yield self._make_reasoning_event(
                "step_start",
                "Synthesizing insights across companies...",
                "synthesis",
            )
            tokens = []
            async for token in self._synthesize_space(
                topic=qualitative_part or question,
                source_label=source_label,
                scored_companies=top_companies,
                marker_prefix=marker_prefix,
            ):
                tokens.append(token)
                yield {'type': 'synthesis_token', 'token': token}
            synthesis_text = "".join(tokens)
            yield self._make_reasoning_event("step_complete", "Analysis complete", "synthesis")

        yield {
            'type': 'result',
            'data': {
                'success': True,
                'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                'friendly_columns': {
                    'symbol': 'Symbol',
                    'relevance_score': 'Relevance',
                    'evidence_summary': 'Evidence',
                    'citations': 'Citations',
                },
                'data_rows': data_rows,
                'synthesis': synthesis_text,
                'message': f'Found {len(data_rows)} companies matching qualitative criteria ({elapsed:.1f}s)',
                'execution_time': elapsed,
            },
        }
