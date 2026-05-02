#!/usr/bin/env python3
"""
RAG Agent - Orchestration and response generation for RAG system.

High-level flow (see execute_rag_flow):
  1. Setup          (_stage_setup)                        - Init context, max_iterations, logging
  2. Reasoning      (_stage_combined_reasoning)           - 🆕 Single LLM call: analyze question + explain research approach + extract metadata (preserves temporal refs!)
  2.1 Search Plan   (_stage_search_planning)              - Resolve temporal refs → specific quarters, generate queries for each data source
  3. PARALLEL       (_stage_parallel_multi_source_search) - 🚀 Run news + 10-K + transcripts IN PARALLEL for max performance
     3a. News       (_stage_news_search)                  - Optional Tavily news if search plan includes news
     3b. 10-K       (_stage_10k_search)                   - Optional SEC 10-K search if search plan includes 10-K
     3c. Transcript (_stage_transcript_search)            - Vector + keyword search over earnings transcripts if in plan
  4. Prepare        (_stage_prepare_context)              - Build news/10-K context strings and combined_citations
  5. Improvement    (_stage_run_improvement)              - Build initial answer, then loop: evaluate → follow-up search → re-answer
  6. Finalize       (_stage_finalize)                     - Dedupe citations, build final result, update conversation memory

Key helpers (used by stages):
  - _perform_question_analysis     → question_analysis, target_quarters, early_return
  - _execute_search                → individual_results, all_chunks, search strategy (general vs ticker)
  - _search_with_multiple_questions → delegates to SearchEngine.search_with_queries_async
  - _perform_follow_up_search / _perform_parallel_follow_up_search → delegate to SearchEngine follow-up methods
  - _build_initial_improvement_state → ImprovementState or early-return tuple
  - _improvement_step               → one iteration: evaluate → optional transcript/news search → optional re-generate
  - _run_iterative_improvement     → full loop; returns (best_answer, confidence, citations, ...)
"""

# Standard library
import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
import openai
from dotenv import load_dotenv

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Local: same package (agent.rag)
from .config import Config
from .database_manager import DatabaseManager, DatabaseConnectionError
from .question_analyzer import QuestionAnalyzer
from .reasoning_planner import ReasoningPlanner
from .rag_flow_context import ImprovementState, RAGFlowContext
from .rag_utils import (
    deduplicate_citations_and_chunks,
    generate_user_friendly_limit_message,
    normalize_ticker,
)
from .response_generator import ResponseGenerator
from .search_engine import SearchEngine
from .search_planner import SearchPlanner
from .sec_filings_service_smart_parallel import (
    SmartParallelSECFilingsService as SECFilingsService,
)
from .tavily_service import TavilyService
from .earnings_transcript_service import EarningsTranscriptService

# Local: parent and agent packages
from .. import prompts
from agent.prompts import (
    CONTEXT_AWARE_FOLLOWUP_SYSTEM_PROMPT,
    get_context_aware_followup_prompt,
)

# Load .env file from the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create a more detailed logger for RAG operations
rag_logger = logging.getLogger('rag_agent')
rag_logger.setLevel(logging.INFO)


class RAGAgent:
    """
    RAG Agent - Orchestrates the complete RAG flow including response generation
    and iterative improvement.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the RAG Agent.

        Sets up the RAG Agent by initializing all necessary components for orchestration.

        Args:
            openai_api_key (Optional[str]): OpenAI API key for LLM operations.
                If None, will attempt to load from environment variables.
        """
        self.instance_id = f"RAGAgent_{int(time.time() * 1000)}"
        logger.info(f"🚀 Creating RAG Agent instance: {self.instance_id}")

        self.config = Config()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Fetch available quarters from database
        try:
            self.config.fetch_available_quarters_from_db()
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch quarters from database during initialization: {e}")
            logger.warning("⚠️ RAG Agent will continue with default configuration")
            logger.warning("⚠️ You may need to run the quarter formatting script first")

        # Initialize components
        self.database_manager = DatabaseManager(self.config)
        self.question_analyzer = QuestionAnalyzer(None, self.config, self.database_manager)

        # NEW: Combined reasoning + analysis (replaces separate question analyzer + reasoning)
        self.reasoning_planner = ReasoningPlanner(
            self.config,
            self.database_manager,
            None  # conversation_memory will be set from question_analyzer
        )
        self.reasoning_planner.conversation_memory = self.question_analyzer.conversation_memory

        self.search_planner = SearchPlanner(self.database_manager, self.config)
        self.search_engine = SearchEngine(self.config, self.database_manager)
        self.response_generator = ResponseGenerator(self.config, self.openai_api_key)
        self.tavily_service = TavilyService()
        self.sec_service = SECFilingsService(self.database_manager, self.config)
        self.transcript_service = EarningsTranscriptService(self.search_engine, self.config)

        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Screener (injected after init via set_qualitative_screener)
        self.qualitative_screener = None

        # OpenAI access is via response_generator.client (single shared instance)
        logger.info(f"🚀 RAG Agent initialized successfully (instance: {self.instance_id})")

        # Log hybrid search configuration
        if self.config.get("hybrid_search_enabled", True):
            logger.info(f"🔀 Hybrid search enabled - Vector weight: {self.config.get('vector_weight', 0.7)}, Keyword weight: {self.config.get('keyword_weight', 0.3)}")
        else:
            logger.info("⚠️ Hybrid search disabled - using vector-only search")

    def set_qualitative_screener(self, screener):
        """Inject the qualitative screener so it can be used as a data source."""
        self.qualitative_screener = screener
        logger.info("✅ QualitativeScreener injected into RAGAgent")

    def set_database_connection(self, db_connection):
        """Set the database connection for retrieving conversation history."""
        self.question_analyzer.conversation_memory.set_database_connection(db_connection)

    def __del__(self):
        """Cleanup when instance is destroyed."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
                rag_logger.info("✅ Thread pool shut down")
        except Exception as e:
            rag_logger.warning(f"⚠️ Error during RAG Agent cleanup: {e}")

    def _get_db_connection(self):
        """Get database connection from pool."""
        return self.database_manager._get_db_connection()

    def _return_db_connection(self, conn):
        """Return connection to pool."""
        self.database_manager._return_db_connection(conn)

    def _determine_answer_mode(self, question_analysis: Dict[str, Any], comprehensive_override: Optional[bool] = None) -> 'AnswerMode':
        """
        Determine the answer mode (direct, standard, detailed) based on LLM's analysis.

        The LLM (QuestionAnalyzer) determines the appropriate answer_mode based on question
        complexity and scope. We trust the LLM's judgment rather than using keyword heuristics.

        Args:
            question_analysis: Analysis result from QuestionAnalyzer (includes LLM-determined answer_mode)
            comprehensive_override: Optional override for comprehensive mode (for backward compatibility)

        Returns:
            AnswerMode enum value
        """
        from .config import AnswerMode

        # Check comprehensive override (for backward compatibility with existing code)
        if comprehensive_override is True:
            return AnswerMode.DETAILED
        elif comprehensive_override is False:
            return AnswerMode.DIRECT

        # Trust the LLM's answer_mode determination from question analysis
        user_hints = question_analysis.get('user_hints', {})
        if 'answer_mode' in user_hints:
            mode_hint = user_hints['answer_mode']
            if mode_hint == 'direct':
                return AnswerMode.DIRECT
            elif mode_hint == 'detailed':
                return AnswerMode.DETAILED
            elif mode_hint == 'deep_search':
                return AnswerMode.DEEP_SEARCH
            elif mode_hint == 'standard':
                return AnswerMode.STANDARD

        # Fallback to standard if LLM didn't provide answer_mode (shouldn't happen with updated prompt)
        rag_logger.warning("⚠️ No answer_mode in question analysis - defaulting to STANDARD. LLM should always provide answer_mode.")
        return AnswerMode.STANDARD

    # -------------------------------------------------------------------------
    # Ticker processing (sync, used for multi-ticker parallel execution)
    # -------------------------------------------------------------------------

    def _process_single_ticker_sync(self, ticker: str, question: str, processed_question: str,
                                   is_multi_ticker: bool, target_quarters: List[str]) -> Dict[str, Any]:
        """Process a single ticker: build ticker-specific question, search transcripts, generate answer.
        Used when processing multiple tickers in parallel via _process_tickers_parallel_sync."""
        ticker_start_time = time.time()

        try:
            rag_logger.info("=" * 80)
            rag_logger.info(f"🎯 STARTING TICKER PROCESSING: {ticker}")
            rag_logger.info(f"📅 Target quarters: {target_quarters}")
            rag_logger.info(f"📝 Processed question: {processed_question[:100]}...")
            rag_logger.info("=" * 80)
            print(f"   🔍 Processing {ticker} synchronously...")

            # For multi-ticker, inject ticker into question so search is company-specific
            if is_multi_ticker:
                print(f"   🎯 Creating ticker-specific question for {ticker}...")
                ticker_specific_question = self.question_analyzer.create_ticker_specific_question(processed_question, ticker)
                print(f"   ✅ Ticker-specific question: '{ticker_specific_question}'")
            else:
                ticker_specific_question = processed_question

            # Resolve quarters for this ticker (companies can have different fiscal calendars)
            ticker_quarters = target_quarters
            if is_multi_ticker:
                if len(target_quarters) == 1:
                    # Single quarter - get company-specific latest quarter
                    company_latest = self.database_manager.get_latest_quarter_for_company(ticker)
                    if company_latest:
                        ticker_quarters = [company_latest]
                        rag_logger.info(f"   📅 Using company-specific latest quarter for {ticker}: {company_latest}")
                        print(f"   📅 Using {ticker}'s latest quarter: {company_latest}")
                    else:
                        rag_logger.warning(f"   ⚠️ No company-specific quarter found for {ticker}, using provided quarter: {target_quarters[0]}")
                else:
                    # Multiple quarters - get company-specific last N quarters
                    # This handles "last N quarters" queries where each company may have different latest quarters
                    quarter_count = len(target_quarters)
                    company_quarters = self.database_manager.get_last_n_quarters_for_company(ticker, quarter_count)
                    if company_quarters:
                        ticker_quarters = company_quarters
                        rag_logger.info(f"   📅 Using company-specific last {quarter_count} quarters for {ticker}: {company_quarters}")
                        print(f"   📅 Using {ticker}'s last {quarter_count} quarters: {', '.join(company_quarters)}")
                    else:
                        rag_logger.warning(f"   ⚠️ No company-specific quarters found for {ticker}, using provided quarters")

            # Encode query and run vector search (single quarter or parallel across quarters)
            print(f"   🔍 Searching {ticker} transcripts...")
            search_start = time.time()

            # Safety check: ensure we have quarters to search
            if not ticker_quarters:
                rag_logger.warning(f"⚠️ No quarters available for {ticker}, searching without quarter filter")
                ticker_chunks = []
            elif len(ticker_quarters) == 1:
                query_embedding = self.search_engine.embedding_model.encode([ticker_specific_question])
                ticker_chunks = self.database_manager._search_postgres_with_ticker(
                    query_embedding, ticker, ticker_quarters[0]
                )
            else:
                # Multiple quarters - parallel search
                query_embedding = self.search_engine.embedding_model.encode([ticker_specific_question])

                def search_quarter_sync(quarter):
                    return self.database_manager._search_postgres_with_ticker(
                        query_embedding, ticker, quarter
                    )

                quarter_results = {}
                optimal_workers = min(len(ticker_quarters), os.cpu_count() * 2, 8)

                # Safety check: ensure at least 1 worker
                if optimal_workers < 1:
                    rag_logger.warning(f"⚠️ optimal_workers={optimal_workers} (ticker_quarters={len(ticker_quarters)}), forcing to 1")
                    optimal_workers = 1

                with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    future_to_quarter = {
                        executor.submit(search_quarter_sync, quarter): quarter
                        for quarter in ticker_quarters
                    }

                    for future in as_completed(future_to_quarter):
                        quarter = future_to_quarter[future]
                        try:
                            quarter_chunks = future.result()
                            quarter_results[quarter] = quarter_chunks
                        except Exception as e:
                            rag_logger.error(f"❌ Error searching {ticker} in quarter {quarter}: {e}")
                            quarter_results[quarter] = []

                # Take top chunks from each quarter
                chunks_per_quarter = self.config.get("chunks_per_quarter", 15)
                ticker_chunks = []
                for quarter in ticker_quarters:
                    quarter_chunks = quarter_results.get(quarter, [])
                    quarter_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    top_quarter_chunks = quarter_chunks[:chunks_per_quarter]
                    for chunk in top_quarter_chunks:
                        chunk['source_quarter'] = quarter
                    ticker_chunks.extend(top_quarter_chunks)

                # Final sort by similarity
                ticker_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)

            search_time = time.time() - search_start
            print(f"   ⏱️  Search completed in {search_time:.3f}s")

            if ticker_chunks:
                print(f"   ✅ Found {len(ticker_chunks)} chunks for {ticker}")

                # Generate individual answer for this ticker
                print(f"   🤖 Generating answer for {ticker}...")
                ticker_context = [chunk['chunk_text'] for chunk in ticker_chunks]
                ticker_citations = ticker_chunks

                year = ticker_chunks[0].get('year') if ticker_chunks else None
                quarter = ticker_chunks[0].get('quarter') if ticker_chunks else None

                ticker_answer = self.response_generator.generate_openai_response(
                    ticker_specific_question,
                    ticker_context,
                    ticker_citations,
                    ticker=ticker,
                    year=year,
                    quarter=quarter
                )

                ticker_total_time = time.time() - ticker_start_time
                print(f"   ✅ {ticker} completed in {ticker_total_time:.3f}s")

                return {
                    'ticker': ticker,
                    'answer': ticker_answer,
                    'chunks': ticker_chunks,
                    'context_chunks': ticker_context,
                    'citations': ticker_citations
                }
            else:
                print(f"   ⚠️ No chunks found for {ticker}")
                return {
                    'ticker': ticker,
                    'answer': f"No relevant information found for {ticker}",
                    'chunks': [],
                    'context_chunks': [],
                    'citations': []
                }

        except Exception as e:
            print(f"   ❌ Error processing {ticker}: {e}")
            rag_logger.error(f"❌ Error processing ticker {ticker}: {e}")
            return {
                'ticker': ticker,
                'answer': f"Error processing {ticker}: {str(e)}",
                'chunks': [],
                'context_chunks': [],
                'citations': []
            }

    def _process_tickers_parallel_sync(self, tickers: List[str], question: str, processed_question: str,
                                     is_multi_ticker: bool, target_quarters: List[str]) -> List[Dict[str, Any]]:
        """Process multiple tickers in parallel using ThreadPoolExecutor."""
        try:
            rag_logger.info(f"🚀 Starting parallel processing of {len(tickers)} tickers")

            results = []
            cpu_cores = os.cpu_count() or 4

            # Worker allocation strategy
            if len(tickers) <= 4:
                optimal_workers = min(len(tickers), cpu_cores * 2, 12)
            else:
                optimal_workers = min(len(tickers), cpu_cores * 3, 16)

            rag_logger.info(f"🚀 Using {optimal_workers} workers for {len(tickers)} tickers")

            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._process_single_ticker_sync, ticker, question, processed_question, is_multi_ticker, target_quarters): ticker
                    for ticker in tickers
                }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)
                        rag_logger.info(f"✅ Completed processing for {ticker}")
                    except Exception as e:
                        rag_logger.error(f"❌ Exception processing ticker {ticker}: {e}")
                        results.append({
                            'ticker': ticker,
                            'answer': f"Error processing {ticker}: {str(e)}",
                            'chunks': [],
                            'context_chunks': [],
                            'citations': []
                        })

            rag_logger.info(f"✅ Parallel processing completed: {len(results)} tickers processed")
            return results

        except Exception as e:
            rag_logger.error(f"❌ Error in parallel ticker processing: {e}")
            # Fallback to sequential processing
            rag_logger.info("🔄 Falling back to sequential processing")
            return [self._process_single_ticker_sync(ticker, question, processed_question, is_multi_ticker, target_quarters) for ticker in tickers]

    # -------------------------------------------------------------------------
    # Multi-query search (asyncio): run several questions in parallel, merge & dedupe
    # -------------------------------------------------------------------------

    async def _search_with_multiple_questions(self, rephrased_questions: List[str], target_quarters: List[str], target_quarter: str, ticker: str = None) -> List[Dict[str, Any]]:
        """Delegate to SearchEngine: multi-query parallel search, merge and dedupe by citation."""
        return await self.search_engine.search_with_queries_async(
            rephrased_questions, target_quarters, target_quarter, ticker=ticker
        )

    # -------------------------------------------------------------------------
    # Question analysis: tickers, quarters, validation, early reject/error
    # -------------------------------------------------------------------------

    async def _perform_question_analysis(self, question: str, conversation_id: str) -> tuple:
        """Perform question analysis and determine target quarters.

        Analyzes the user's question to extract tickers, determine question type,
        identify target quarters, and validate data availability. Handles quarter
        limits and provides user-friendly error messages for unavailable data.

        Args:
            question (str): The user's question to analyze.
            conversation_id (str): Unique conversation identifier.
        
        Returns:
            tuple: (question_analysis, target_quarters, error_dict)
                - question_analysis (Dict or None): Analysis results with structure:
                    {
                        "status": str,  # 'success', 'rejected', or 'error'
                        "extracted_ticker": str,  # Primary ticker symbol
                        "extracted_tickers": List[str],  # All ticker symbols
                        "question_type": str,  # Question category
                        "processed_question": str,  # Cleaned question
                        "quarter_context": str,  # Quarter information
                        "quarter_reference": str,  # Specific quarter ref
                        "limits_exceeded": Dict,  # Any limit violations
                        "message": str  # Status message (for rejected)
                    }
                - target_quarters (List[str] or None): Quarters to search (e.g., ['2025_q1', '2025_q2'])
                - error_dict (Dict or None): Error information if validation fails:
                    {
                        "status": str,  # 'error'
                        "message": str,  # User-friendly error message
                        "available_quarters": List[str],  # Available data
                        "requested_quarter": str  # What was requested
                    }
        """
        analysis_start = time.time()
        print(f"\n📋 STEP 1: QUESTION ANALYSIS")
        print(f"{'─'*50}")
        rag_logger.info(f"📋 Step 1: Starting question analysis...")
        
        question_analysis = await self.question_analyzer.process_question(question, conversation_id)
        analysis_time = time.time() - analysis_start

        # Add conversation_id to question_analysis for evaluation context
        question_analysis['conversation_id'] = conversation_id

        # Normalize tickers in question_analysis BEFORE quarter lookup
        # This ensures company-specific quarter resolution works for aliases (e.g., TSMC → TSM)
        raw_tickers = question_analysis.get('extracted_tickers', [])
        if raw_tickers:
            question_analysis['extracted_tickers'] = [normalize_ticker(t) for t in raw_tickers if t]
        raw_ticker = question_analysis.get('extracted_ticker')
        if raw_ticker:
            question_analysis['extracted_ticker'] = normalize_ticker(raw_ticker)

        # Debug output
        rag_logger.info(f"🔍 Question analysis debug:")
        rag_logger.info(f"   quarter_context: {question_analysis.get('quarter_context')}")
        rag_logger.info(f"   quarter_reference: {question_analysis.get('quarter_reference')}")
        rag_logger.info(f"   extracted_ticker: {question_analysis.get('extracted_ticker')}")
        rag_logger.info(f"   extracted_tickers: {question_analysis.get('extracted_tickers')}")

        # Determine target quarters
        target_quarters = self.question_analyzer.get_quarters_to_search(question_analysis)

        # Apply quarter limit
        max_quarters = self.config.get("max_quarters", 12)
        if len(target_quarters) > max_quarters:
            rag_logger.warning(f"⚠️ Quarter limit exceeded: {len(target_quarters)} quarters requested, limiting to {max_quarters}")
            original_count = len(target_quarters)
            skipped_quarters = target_quarters[max_quarters:]
            target_quarters = target_quarters[:max_quarters]
            print(f"⚠️ Too many quarters requested ({original_count}). Limiting to most recent {max_quarters} quarters (3 years).")
            print(f"📅 Processing quarters: {', '.join(target_quarters)}")
            
            question_analysis['limits_exceeded'] = question_analysis.get('limits_exceeded', {})
            question_analysis['limits_exceeded']['quarters'] = {
                'requested': original_count,
                'processed': max_quarters,
                'skipped': skipped_quarters
                }

        # Validate quarters availability
        if not target_quarters:
            quarter_reference = question_analysis.get('quarter_reference')
            if quarter_reference:
                available_quarters_human = []
                for q in self.config.get('available_quarters', []):
                    if '_q' in q:
                        year, quarter = q.split('_q')
                        available_quarters_human.append(f"{year} Q{quarter}")
                    else:
                        available_quarters_human.append(q)
                
                requested_quarter_human = quarter_reference
                if '_q' in quarter_reference:
                    year, quarter = quarter_reference.split('_q')
                    requested_quarter_human = f"{year} Q{quarter}"
                
                return None, None, {
                    "status": "error",
                    "message": f"Sorry, I don't have earnings transcript data for {requested_quarter_human} available. The available quarters are: {', '.join(available_quarters_human)}.",
                    "available_quarters": self.config.get('available_quarters', []),
                    "requested_quarter": quarter_reference
                }
            else:
                return None, None, {
                    "status": "error",
                    "message": "Sorry, I don't have any earnings transcript data available at the moment. Please try again later.",
                    "available_quarters": []
                }
        
        target_quarter = target_quarters[0] if len(target_quarters) == 1 else 'multiple'
        rag_logger.info(f"📅 Target quarters determined: {target_quarters}")

        print(f"⏱️  Analysis completed in {analysis_time:.3f}s")
        print(f"📊 Analysis Results:")
        print(f"   ✅ Status: {question_analysis['status']}")
        print(f"   🎯 Primary ticker: {question_analysis.get('extracted_ticker', 'None')}")
        print(f"   📝 Question type: {question_analysis.get('question_type', 'Unknown')}")
        print(f"   📅 Target quarters: {target_quarters}")
        
        rag_logger.info(f"✅ Question analysis completed in {analysis_time:.3f}s")

        # Handle rejected questions
        if question_analysis['status'] == 'rejected':
            rag_logger.warning(f"❌ Question rejected: {question_analysis.get('message', 'Unknown reason')}")
            return None, None, {
                'success': False,
                'errors': [question_analysis['message']],
                'analysis': question_analysis,
                'timing': {
                    'analysis': analysis_time,
                    'total': analysis_time
                }
            }
        
        return question_analysis, target_quarters, None

    # -------------------------------------------------------------------------
    # Search execution: general vs ticker-specific; single vs multi-ticker
    # -------------------------------------------------------------------------

    async def _execute_search(self, ctx: RAGFlowContext) -> None:
        """Execute search based on question type (general vs ticker-specific).

        Reads from ctx: question, question_analysis, target_quarters.
        Writes to ctx: individual_results, all_chunks, all_citations, search_time,
        is_general_question, is_multi_ticker, tickers_to_process, target_quarter.
        """
        question = ctx.question
        question_analysis = ctx.question_analysis
        target_quarters = ctx.target_quarters or []
        search_start = time.time()

        print(f"\n🔍 STEP 2: SEARCH STRATEGY DETERMINATION")
        print(f"{'─'*50}")
        rag_logger.info(f"🔍 Step 2: Starting vector search...")

        processed_question = question_analysis.get('processed_question', question)
        extracted_tickers = [normalize_ticker(t) for t in question_analysis.get('extracted_tickers', []) if t]
        extracted_ticker = normalize_ticker(question_analysis.get('extracted_ticker') or '')
        if not extracted_ticker:
            extracted_ticker = None
        target_quarter = target_quarters[0] if len(target_quarters) == 1 else 'multiple'

        print(f"🔍 Search Strategy Analysis:")
        print(f"   📝 Processed question: '{processed_question}'")
        print(f"   🎯 Extracted ticker: {extracted_ticker}")
        print(f"   🎯 All tickers: {extracted_tickers}")

        # --- Strategy: general (no ticker) vs single-ticker vs multi-ticker ---
        has_tickers = bool(extracted_ticker or extracted_tickers)
        is_multi_ticker = len(extracted_tickers) > 1
        is_general_question = not has_tickers

        if has_tickers:
            tickers_to_process = extracted_tickers if is_multi_ticker else [extracted_ticker]
            
            # Apply ticker limit
            max_tickers = self.config.get("max_tickers", 4)
            if len(tickers_to_process) > max_tickers:
                rag_logger.warning(f"⚠️ Ticker limit exceeded: {len(tickers_to_process)} tickers requested, limiting to {max_tickers}")
                tickers_to_process = tickers_to_process[:max_tickers]
                print(f"⚠️ Too many tickers requested. Limiting to first {max_tickers} tickers.")
                
                question_analysis['limits_exceeded'] = question_analysis.get('limits_exceeded', {})
                question_analysis['limits_exceeded']['tickers'] = {
                    'requested': len(extracted_tickers),
                    'processed': max_tickers,
                    'skipped': extracted_tickers[max_tickers:]
                }
        else:
            tickers_to_process = []

        print(f"📊 Search Strategy Decision:")
        print(f"   🎯 Has tickers: {has_tickers}")
        print(f"   🔄 Is multi-ticker: {is_multi_ticker}")
        print(f"   🌐 Is general question: {is_general_question}")
        print(f"   📋 Tickers to process: {tickers_to_process}")
        
        # Initial search uses processed_question only (rephrasing disabled for precision)
        print(f"\n🔄 STEP 2.5: QUERY PREPARATION")
        print(f"{'─'*50}")
        # rephrased_questions = self._generate_rephrased_questions(question, question_analysis)
        rephrased_questions = [processed_question]  # Use only the processed question
        print(f"✅ Using direct question (rephrasing disabled): {processed_question}")

        # --- Run search: general (all companies) or per-ticker ---
        print(f"\n🔍 STEP 3: SEARCH EXECUTION")
        print(f"{'─'*50}")
        individual_results = []
        all_chunks = []
        all_citations = []

        if is_general_question:
            # No ticker: search all transcripts, then group by ticker and generate per-ticker answers
            print(f"🌐 Processing general question")

            search_start_time = time.time()
            # Use parallel multi-query search
            general_chunks = await self._search_with_multiple_questions(
                rephrased_questions, target_quarters, target_quarter, ticker=None
            )
            search_time_inner = time.time() - search_start_time

            print(f"⏱️  General search completed in {search_time_inner:.3f}s")
            print(f"📊 Search results: {len(general_chunks)} chunks found")
            
            if general_chunks:
                # Group chunks by ticker and generate answers
                ticker_groups = {}
                for chunk in general_chunks:
                    ticker = chunk.get('ticker', 'Unknown')
                    if ticker not in ticker_groups:
                        ticker_groups[ticker] = []
                    ticker_groups[ticker].append(chunk)
                
                for ticker, ticker_chunks in ticker_groups.items():
                    ticker_context = [chunk['chunk_text'] for chunk in ticker_chunks]
                    ticker_citations = ticker_chunks
                    year = ticker_chunks[0].get('year') if ticker_chunks else None
                    quarter = ticker_chunks[0].get('quarter') if ticker_chunks else None
                    
                    ticker_answer = self.response_generator.generate_openai_response(
                        processed_question, ticker_context, ticker_citations, 
                        ticker=ticker, year=year, quarter=quarter
                    )
                    
                    individual_results.append({
                        'ticker': ticker,
                        'answer': ticker_answer,
                        'chunks': ticker_chunks,
                        'context_chunks': ticker_context,
                        'citations': ticker_citations
                    })
                
                # Convert chunks to proper citation format with sequential markers
                for chunk in general_chunks:
                    orig_idx = chunk['citation']
                    seq = len(all_citations) + 1
                    chunk['_chunk_index_orig'] = orig_idx
                    chunk['citation'] = seq
                    all_chunks.append(chunk)
                    all_citations.append({
                        "type": "transcript",
                        "marker": f"[{seq}]",
                        "ticker": chunk.get('ticker') or '',
                        "year": chunk.get('year'),
                        "quarter": chunk.get('quarter'),
                        "chunk_text": (chunk.get('chunk_text') or '')[:400],
                        "chunk_index": orig_idx,
                        "chunk_id": str(chunk.get('id') or f"tc_{chunk.get('ticker','')}_{chunk.get('year','')}_{orig_idx}"),
                        "char_offset": chunk.get('char_offset'),
                        "chunk_length": chunk.get('chunk_length'),
                    })

        else:
            # One or more tickers: parallel ticker processing or single-ticker multi-query search
            print(f"🎯 Processing ticker-specific question with query expansion")

            if len(tickers_to_process) > 1:
                print(f"🚀 Using parallel processing for {len(tickers_to_process)} tickers")
                individual_results = self._process_tickers_parallel_sync(
                    tickers_to_process, question, processed_question, is_multi_ticker, target_quarters
                )
            else:
                print(f"🎯 Processing single ticker: {tickers_to_process[0]}")
                ticker = tickers_to_process[0]
                
                # Use direct search for single ticker (rephrasing disabled)
                search_start_time = time.time()
                ticker_chunks = await self._search_with_multiple_questions(
                    rephrased_questions, target_quarters, target_quarter, ticker=ticker
                )
                search_time_inner = time.time() - search_start_time
                
                print(f"   ✅ Found {len(ticker_chunks)} chunks for {ticker}")
                
                # Create individual result
                ticker_context = [chunk['chunk_text'] for chunk in ticker_chunks]
                ticker_citations = ticker_chunks
                year = ticker_chunks[0].get('year') if ticker_chunks else None
                quarter = ticker_chunks[0].get('quarter') if ticker_chunks else None
                
                ticker_answer = self.response_generator.generate_openai_response(
                    question, ticker_context, ticker_citations, 
                    ticker=ticker, year=year, quarter=quarter
                )
                
                individual_results = [{
                    'ticker': ticker,
                    'answer': ticker_answer,
                    'chunks': ticker_chunks,
                    'context_chunks': ticker_context,
                    'citations': ticker_citations
                }]
            
            # Collect all chunks and citations with sequential markers
            for result in individual_results:
                for chunk in result.get('chunks', []):
                    orig_idx = chunk['citation']
                    seq = len(all_citations) + 1
                    chunk['_chunk_index_orig'] = orig_idx
                    chunk['citation'] = seq
                    all_chunks.append(chunk)
                    all_citations.append({
                        "type": "transcript",
                        "marker": f"[{seq}]",
                        "ticker": chunk.get('ticker') or '',
                        "year": chunk.get('year'),
                        "quarter": chunk.get('quarter'),
                        "chunk_text": (chunk.get('chunk_text') or '')[:400],
                        "chunk_index": orig_idx,
                        "chunk_id": str(chunk.get('id') or f"tc_{chunk.get('ticker','')}_{chunk.get('year','')}_{orig_idx}"),
                        "char_offset": chunk.get('char_offset'),
                        "chunk_length": chunk.get('chunk_length'),
                    })

        search_time = time.time() - search_start

        print(f"\n⏱️  SEARCH PHASE COMPLETED")
        print(f"   ✅ Search completed in {search_time:.3f}s")
        print(f"   📊 Found {len(all_chunks)} total chunks")

        ctx.individual_results = individual_results
        ctx.all_chunks = all_chunks
        ctx.all_citations = all_citations
        ctx.search_time = search_time
        ctx.is_general_question = is_general_question
        ctx.is_multi_ticker = is_multi_ticker
        ctx.tickers_to_process = tickers_to_process
        ctx.target_quarter = target_quarter

    # -------------------------------------------------------------------------
    # Follow-up search: used during improvement loop to fetch more chunks by new queries
    # -------------------------------------------------------------------------

    async def _perform_parallel_follow_up_search(self, follow_up_questions: List[str], has_tickers: bool,
                                                 is_general_question: bool, is_multi_ticker: bool,
                                                 tickers_to_process: List[str], target_quarter,
                                                 target_quarters: List[str]) -> List[Dict[str, Any]]:
        """Delegate to SearchEngine: parallel follow-up searches, merge and dedupe by citation."""
        return await self.search_engine.follow_up_search_parallel_async(
            follow_up_questions, has_tickers, is_general_question, is_multi_ticker,
            tickers_to_process, target_quarter, target_quarters
        )

    async def _perform_follow_up_search(self, follow_up_question: str, has_tickers: bool, is_general_question: bool,
                                       is_multi_ticker: bool, tickers_to_process: List[str], target_quarter,
                                       target_quarters: List[str]) -> List[Dict[str, Any]]:
        """Delegate to SearchEngine: single follow-up search (hybrid for ticker(s), general otherwise)."""
        return await self.search_engine.follow_up_search_async(
            follow_up_question, has_tickers, is_general_question, is_multi_ticker,
            tickers_to_process, target_quarter, target_quarters
        )

    # -------------------------------------------------------------------------
    # Iterative improvement: initial state, one step (evaluate → search → re-answer), full loop
    # -------------------------------------------------------------------------

    async def _build_initial_improvement_state(
        self,
        ctx: RAGFlowContext,
        sync_retry_callback,
        flush_retry_events,
        conversation_context: str = "",
    ):
        """
        Build initial state for the improvement loop: empty-context early return or generate initial answer.
        Reads from ctx; returns (state, early_return_tuple). If early_return_tuple is not None, caller should return it.
        Otherwise state is the ImprovementState to run the loop on.
        conversation_context: Formatted recent conversation (sliding window) for stateful follow-up questions.
        """
        question = ctx.question
        individual_results = ctx.individual_results
        all_chunks = ctx.all_chunks
        question_analysis = ctx.question_analysis
        is_general_question = ctx.is_general_question
        is_multi_ticker = ctx.is_multi_ticker
        tickers_to_process = ctx.tickers_to_process
        show_details = ctx.show_details
        comprehensive = ctx.comprehensive
        news_context = ctx.news_context_str
        ten_k_context = ctx.ten_k_context_str
        transcript_context = ctx.transcript_context_str
        all_citations = ctx.combined_citations

        # Screener-only fast path: build a clean formatted list directly from screener rows
        if (ctx.screener_data_rows
                and not ctx.transcript_per_ticker_results
                and not ctx.sec_service_results
                and not all_chunks):
            import re as _re_scr
            rag_logger.info(f"🔍 Screener-only path: formatting {len(ctx.screener_data_rows)} results directly")
            rows = ctx.screener_data_rows
            lines = [f"Found **{len(rows)} companies** matching your criteria:\n"]
            global_citations = []
            citation_offset = 0

            for i, row in enumerate(rows, 1):
                symbol = row.get('symbol', '')
                evidence = (row.get('evidence_summary') or '').strip()
                cit_list = row.get('citations', [])

                # Remap local [1],[2] in evidence_summary → global [TC-N] / [10K-N] markers
                remapped = evidence
                for cit in cit_list:
                    local_idx = cit.get('index', 0)
                    if not local_idx:
                        continue
                    global_idx = local_idx + citation_offset
                    src_type = cit.get('source_type', cit.get('type', 'transcript'))
                    prefix = 'TC' if src_type in ('transcript', 'tc') else '10K'
                    global_marker = f"[{prefix}-{global_idx}]"
                    remapped = _re_scr.sub(rf'\[{local_idx}\]', global_marker, remapped)
                    global_citations.append({**cit, 'source_number': global_idx, 'marker': global_marker})

                citation_offset += len(cit_list)
                lines.append(f"{i}. **{symbol}** — {remapped}" if remapped else f"{i}. **{symbol}**")

            screener_answer = "\n".join(lines)
            ctx.combined_citations = global_citations
            state = ImprovementState(
                accumulated_chunks=[],
                accumulated_citations=global_citations,
                best_answer=screener_answer,
                best_confidence=0.9,
                best_citations=global_citations,
                best_context_chunks=[],
                best_chunks=[],
                news_context=news_context,
                ten_k_context=ten_k_context,
                transcript_context=transcript_context,
            )
            ctx.skip_improvement = True
            return (state, None)

        # Fast path: single subagent answer is final — skip response generator entirely
        if getattr(ctx, 'skip_improvement', False) and ctx.transcript_service_answer:
            rag_logger.info("⚡ skip_improvement=True: using single subagent answer directly")
            state = ImprovementState(
                accumulated_chunks=all_chunks.copy(),
                accumulated_citations=all_citations.copy(),
                best_answer=ctx.transcript_service_answer,
                best_confidence=0.9,
                best_citations=all_citations.copy(),
                best_context_chunks=[c.get('chunk_text', '') for c in all_chunks],
                best_chunks=all_chunks.copy(),
                news_context=news_context,
                ten_k_context=ten_k_context,
                transcript_context=transcript_context,
            )
            return (state, None)

        # Multi-agent path: synthesize per-subagent answers (news counted as a subagent)
        has_news = bool(news_context)
        total_subagents = (
            len(ctx.transcript_per_ticker_results)
            + len(ctx.sec_service_results)
            + (1 if has_news else 0)
        )
        if total_subagents > 1 and (ctx.transcript_per_ticker_results or ctx.sec_service_results):
            rag_logger.info(f"🔀 Multi-agent synthesis: {total_subagents} subagents")
            all_subagent_results = [
                {'type': 'transcript', 'ticker': r['ticker'], 'answer': r['answer'], 'citations': r['citations']}
                for r in ctx.transcript_per_ticker_results
            ] + [
                {'type': '10k', 'ticker': r['ticker'], 'answer': r['answer'], 'citations': r['citations'], 'fiscal_year': r.get('fiscal_year')}
                for r in ctx.sec_service_results
            ]
            synthesis = await self.transcript_service.synthesize_subagents(
                ctx.question, all_subagent_results, news_context
            )
            rag_logger.info(f"🔀 Synthesis complete ({len(synthesis)} chars)")
            # Mark skip so improvement loop doesn't overwrite the synthesis with raw per-subagent blocks
            ctx.skip_improvement = True
            state = ImprovementState(
                accumulated_chunks=all_chunks.copy(),
                accumulated_citations=all_citations.copy(),
                best_answer=synthesis,
                best_confidence=0.9,
                best_citations=all_citations.copy(),
                best_context_chunks=[c.get('chunk_text', '') for c in all_chunks],
                best_chunks=all_chunks.copy(),
                news_context=news_context,
                ten_k_context=ten_k_context,
                transcript_context=transcript_context,
            )
            return (state, None)

        # No context at all → return early with empty-context answer (no improvement loop)
        if not individual_results and not news_context and not ten_k_context and not transcript_context:
            rag_logger.warning(
                "⚠️ No transcript chunks found and no 10-K or news context - delegating to response generator with empty context"
            )
            _answer_mode = ctx.answer_mode.value if ctx.answer_mode else None
            empty_answer = self.response_generator.generate_openai_response(
                question=question,
                context_chunks=[],
                chunk_objects=[],
                ticker=tickers_to_process[0] if tickers_to_process else None,
                stream_callback=None,
                news_context=None,
                ten_k_context=None,
                transcript_context=None,
                previous_answer=None,
                conversation_context=conversation_context,
                answer_mode=_answer_mode,
            )
            early_return = (empty_answer, 0.0, [], [], [], [], [], [])
            return (None, early_return)

        # Build state and generate initial answer (multi-ticker vs single-ticker)
        state = ImprovementState(
            accumulated_chunks=all_chunks.copy(),
            accumulated_citations=all_citations.copy(),
            news_context=news_context,
            ten_k_context=ten_k_context,
            transcript_context=transcript_context,
        )
        _answer_mode = ctx.answer_mode.value if ctx.answer_mode else None
        if is_general_question or (is_multi_ticker and len(individual_results) > 1):
            state.best_answer = self.response_generator.generate_multi_ticker_response(
                question, state.accumulated_chunks, individual_results, show_details, comprehensive,
                stream_callback=None, news_context=news_context, ten_k_context=ten_k_context,
                transcript_context=transcript_context,
                conversation_context=conversation_context, retry_callback=sync_retry_callback,
                answer_mode=_answer_mode
            )
        else:
            state.best_answer = self.response_generator.generate_openai_response(
                question, [c['chunk_text'] for c in state.accumulated_chunks], state.accumulated_chunks,
                ticker=tickers_to_process[0] if tickers_to_process else None, stream_callback=None,
                news_context=news_context, ten_k_context=ten_k_context,
                transcript_context=transcript_context,
                conversation_context=conversation_context, retry_callback=sync_retry_callback,
                answer_mode=_answer_mode
            )
        await flush_retry_events()
        state.best_citations = state.accumulated_citations.copy()
        state.best_context_chunks = [c['chunk_text'] for c in state.accumulated_chunks]
        state.best_chunks = state.accumulated_chunks.copy()

        # Prepend user-facing limit message if quarters/tickers were capped
        if question_analysis.get('limits_exceeded'):
            limit_message = generate_user_friendly_limit_message(question_analysis['limits_exceeded'])
            if limit_message:
                state.best_answer = limit_message + "\n\n" + state.best_answer
        return (state, None)

    async def _improvement_step(
        self,
        iteration: int,
        max_iterations: int,
        state: ImprovementState,
        ctx: RAGFlowContext,
        stream_callback=None,
        event_yielder=None,
        sync_retry_callback=None,
        flush_retry_events=None,
        conversation_context: str = "",
    ) -> bool:
        """
        One improvement iteration: evaluate → maybe stop → search (transcript + news + follow-up) → if new chunks, generate.
        Reads from ctx; returns False if the loop should stop, True to continue.
        """
        # Single subagent or synthesized answer is final — skip evaluation loop entirely
        if getattr(ctx, 'skip_improvement', False):
            rag_logger.info("⚡ skip_improvement=True — skipping improvement loop")
            return False

        question = ctx.question
        question_analysis = ctx.question_analysis
        individual_results = ctx.individual_results
        is_general_question = ctx.is_general_question
        is_multi_ticker = ctx.is_multi_ticker
        tickers_to_process = ctx.tickers_to_process
        target_quarter = ctx.target_quarter
        target_quarters = ctx.target_quarters or []
        show_details = ctx.show_details
        comprehensive = ctx.comprehensive

        if LOGFIRE_AVAILABLE and logfire:
            logfire.info("rag.iteration.start", iteration=iteration, max_iterations=max_iterations, current_chunks=len(state.accumulated_chunks), current_confidence=state.best_confidence)
        if event_yielder:
            await event_yielder({
                'type': 'iteration_start',
                'message': 'Refining answer with additional context',
                'step': 'iteration',
                'data': {'iteration': iteration, 'total_iterations': max_iterations, 'current_chunks': len(state.accumulated_chunks), 'current_confidence': state.best_confidence}
            })
            await asyncio.sleep(0.01)

        # 1) Evaluate current answer (LLM returns confidence, should_iterate, follow_up_questions, optional transcript/news queries)
        from .rag_utils import assess_answer_quality
        answer_quality = assess_answer_quality(state.best_answer, len(state.accumulated_chunks))
        conversation_memory = self.question_analyzer.conversation_memory if hasattr(self, 'question_analyzer') else None
        conversation_id_for_eval = question_analysis.get('conversation_id')
        reasoning_context = question_analysis.get('reasoning_statement')
        # Infer data source from search plan
        current_data_source = 'earnings_transcripts'  # Default
        if ctx.search_plan:
            if ctx.search_plan.has_10k() and not ctx.search_plan.has_transcripts():
                current_data_source = '10k'
            elif ctx.search_plan.has_news() and not ctx.search_plan.has_transcripts():
                current_data_source = 'latest_news'
            elif ctx.search_plan.has_10k() and ctx.search_plan.has_transcripts():
                current_data_source = 'hybrid'

        _answer_mode = ctx.answer_mode.value if ctx.answer_mode else None
        try:
            if answer_quality['is_insufficient']:
                evaluation = await self.response_generator.evaluate_answer_quality(
                    question, state.best_answer, [c['chunk_text'] for c in state.accumulated_chunks], state.accumulated_chunks,
                    conversation_memory=conversation_memory, conversation_id=conversation_id_for_eval,
                    follow_up_questions_asked=state.follow_up_questions_asked, evaluation_context=state.evaluation_context,
                    reasoning_context=reasoning_context, data_source=current_data_source,
                    answer_mode=_answer_mode
                )
            else:
                evaluation = await self.response_generator.evaluate_answer_quality(
                    question, state.best_answer, [c['chunk_text'] for c in state.accumulated_chunks],
                    conversation_memory=conversation_memory, conversation_id=conversation_id_for_eval,
                    follow_up_questions_asked=state.follow_up_questions_asked, evaluation_context=state.evaluation_context,
                    reasoning_context=reasoning_context, data_source=current_data_source,
                    answer_mode=_answer_mode
                )
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
            rag_logger.warning(f"⚠️ LLM API error during evaluation at iteration {iteration}: {e}")
            if state.best_answer and event_yielder:
                await event_yielder({'type': 'api_retry', 'message': 'API error during evaluation, returning best answer so far', 'step': 'iteration', 'data': {'error': str(e), 'iteration': iteration, 'graceful_fallback': True}})
            return False

        evaluation_confidence = evaluation.get('overall_confidence', 0.5)
        if evaluation_confidence > state.best_confidence:
            state.best_confidence = evaluation_confidence

        iteration_searches = {'transcript_search_performed': False, 'transcript_search_results': 0, 'news_search_performed': False, 'news_search_results': 0, 'ten_k_search_performed': False, 'ten_k_search_results': 0}
        state.evaluation_context.append({'iteration': iteration, 'evaluation': evaluation, 'confidence': evaluation_confidence, 'searches_performed': iteration_searches})

        follow_up_questions = evaluation.get('follow_up_questions', [])
        should_iterate = evaluation.get('should_iterate', False)
        needs_news_search = evaluation.get('needs_news_search', False)
        news_search_query = evaluation.get('news_search_query', None)
        needs_transcript_search = evaluation.get('needs_transcript_search', False)
        transcript_search_query = evaluation.get('transcript_search_query', None)
        iteration_new_chunks = []

        if event_yielder and should_iterate:
            await event_yielder({
                'type': 'agent_decision',
                'message': 'Analyzing answer quality',
                'step': 'iteration',
                'data': {
                    'iteration': iteration, 'should_iterate': should_iterate, 'confidence': evaluation_confidence,
                    'completeness_score': evaluation.get('completeness_score', 10),
                    'specificity_score': evaluation.get('specificity_score', 10),
                    'accuracy_score': evaluation.get('accuracy_score', 0), 'clarity_score': evaluation.get('clarity_score', 0),
                    'has_follow_up_questions': len(follow_up_questions) > 0, 'follow_up_count': len(follow_up_questions)
                }
            })
            await asyncio.sleep(0.01)

        # 2) Stop if max iterations, high confidence, or evaluator says no more iteration / no follow-ups
        from .config import AnswerMode, ANSWER_MODE_CONFIG
        confidence_threshold = ANSWER_MODE_CONFIG.get(
            ctx.answer_mode, ANSWER_MODE_CONFIG[AnswerMode.STANDARD]
        )["confidence_threshold"] if ctx.answer_mode else 0.9
        should_stop = (
            iteration >= max_iterations
            or evaluation_confidence >= confidence_threshold
            or not should_iterate
            or not follow_up_questions
        )
        if should_stop:
            if event_yielder and iteration < max_iterations:
                await event_yielder({
                    'type': 'iteration_complete',
                    'message': 'Research complete',
                    'step': 'iteration',
                    'data': {'iteration': iteration, 'total_iterations': max_iterations, 'final_confidence': evaluation_confidence, 'reason': 'Answer quality sufficient'}
                })
                await asyncio.sleep(0.01)
            return False

        # 3) Optional: transcript follow-up search (evaluator-provided query); add new chunks to state
        if needs_transcript_search and current_data_source not in ['10k', 'latest_news'] and transcript_search_query and not getattr(ctx, 'skip_transcript_follow_up', False):
            if event_yielder:
                await event_yielder({'type': 'iteration_transcript_search', 'message': 'Searching earnings transcripts to enhance answer', 'step': 'iteration', 'data': {'iteration': iteration, 'query': transcript_search_query}})
                await asyncio.sleep(0.01)
            has_tickers = bool(tickers_to_process)
            transcript_chunks = await self._perform_parallel_follow_up_search(
                [transcript_search_query], has_tickers, is_general_question, is_multi_ticker,
                tickers_to_process, target_quarter, target_quarters
            )
            iteration_searches['transcript_search_performed'] = True
            iteration_searches['transcript_search_results'] = len(transcript_chunks or [])
            existing_orig_indices = {c.get('_chunk_index_orig', c['citation']) for c in state.accumulated_chunks}
            for chunk in transcript_chunks or []:
                orig_idx = chunk['citation']
                if orig_idx not in existing_orig_indices:
                    seq = len(state.accumulated_citations) + 1
                    chunk['_chunk_index_orig'] = orig_idx
                    chunk['citation'] = seq
                    state.accumulated_chunks.append(chunk)
                    iteration_new_chunks.append(chunk)
                    existing_orig_indices.add(orig_idx)
                    state.accumulated_citations.append({
                        "type": "transcript",
                        "marker": f"[{seq}]",
                        "ticker": chunk.get('ticker') or '',
                        "year": chunk.get('year'),
                        "quarter": chunk.get('quarter'),
                        "chunk_text": (chunk.get('chunk_text') or '')[:400],
                        "chunk_index": orig_idx,
                    })

        # 4) Optional: news search (evaluator-provided query); append to news_context and citations
        if needs_news_search and self.tavily_service.is_available() and news_search_query:
            if event_yielder:
                await event_yielder({'type': 'iteration_news_search', 'message': 'Searching for latest news to enhance answer', 'step': 'iteration', 'data': {'iteration': iteration, 'query': news_search_query}})
                await asyncio.sleep(0.01)
            iteration_news_results = self.tavily_service.search_news(news_search_query, max_results=5, include_answer="advanced")
            iteration_searches['news_search_performed'] = True
            iteration_searches['news_search_results'] = len(iteration_news_results.get('results', [])) if iteration_news_results else 0
            if iteration_news_results and iteration_news_results.get("results"):
                new_news_context = self.tavily_service.format_news_context(iteration_news_results)
                state.news_context = f"{state.news_context or ''}\n\n=== ADDITIONAL NEWS (from iteration) ===\n{new_news_context}".strip()
                for news_citation in self.tavily_service.get_news_citations(iteration_news_results):
                    state.accumulated_citations.append({
                        "type": "news", "marker": f"[N{news_citation['index']}]",
                        "title": news_citation["title"], "url": news_citation["url"], "published_date": news_citation.get("published_date", "")
                    })

        # 5) Follow-up questions: from evaluator + optionally context-aware; skip for 10k/latest_news-only
        if answer_quality['is_insufficient']:
            context_aware = self._generate_context_aware_follow_up_questions(question, state.best_answer, state.accumulated_chunks)
            all_follow_up_questions = list(set(follow_up_questions + context_aware))
        else:
            all_follow_up_questions = follow_up_questions

        skip_followup_transcript = current_data_source in ['10k', 'latest_news'] or getattr(ctx, 'skip_transcript_follow_up', False)
        if skip_followup_transcript:
            all_follow_up_questions = []

        if all_follow_up_questions:
            state.follow_up_questions_asked.extend(all_follow_up_questions)
            if event_yielder:
                await event_yielder({'type': 'iteration_followup', 'message': '\n'.join([f'Searching: "{q}"' for q in all_follow_up_questions]), 'step': 'iteration', 'data': {'iteration': iteration, 'followup_question': all_follow_up_questions[0], 'all_questions': all_follow_up_questions}})
                await asyncio.sleep(0.01)
            has_tickers = bool(tickers_to_process)
            refined_chunks = await self._perform_parallel_follow_up_search(
                all_follow_up_questions, has_tickers, is_general_question, is_multi_ticker,
                tickers_to_process, target_quarter, target_quarters
            )
            if refined_chunks:
                existing_orig_indices = {c.get('_chunk_index_orig', c['citation']) for c in state.accumulated_chunks}
                new_chunks = []
                for chunk in refined_chunks:
                    orig_idx = chunk['citation']
                    if orig_idx not in existing_orig_indices:
                        chunk['_chunk_index_orig'] = orig_idx
                        new_chunks.append(chunk)
                        existing_orig_indices.add(orig_idx)
                state.accumulated_chunks.extend(new_chunks)
                iteration_new_chunks.extend(new_chunks)
                # Convert chunks to proper citation format with sequential markers
                for chunk in new_chunks:
                    seq = len(state.accumulated_citations) + 1
                    chunk['citation'] = seq
                    state.accumulated_citations.append({
                        "type": "transcript",
                        "marker": f"[{seq}]",
                        "ticker": chunk.get('ticker') or '',
                        "year": chunk.get('year'),
                        "quarter": chunk.get('quarter'),
                        "chunk_text": (chunk.get('chunk_text') or '')[:400],
                        "chunk_index": chunk['_chunk_index_orig'],
                    })
                if event_yielder:
                    await event_yielder({'type': 'iteration_search', 'message': 'Incorporating additional relevant sources', 'step': 'iteration', 'data': {'iteration': iteration, 'new_chunks_count': len(new_chunks), 'total_chunks': len(state.accumulated_chunks)}})
                    await asyncio.sleep(0.01)

        # 6) If we added new chunks this iteration, re-generate answer (multi-ticker or single) and update state
        if iteration_new_chunks:
            try:
                _am = ctx.answer_mode.value if ctx.answer_mode else None
                if is_general_question or (is_multi_ticker and len(individual_results) > 1):
                    refined_answer = self.response_generator.generate_multi_ticker_response(
                        question, iteration_new_chunks, individual_results, show_details, comprehensive,
                        stream_callback=None, news_context=state.news_context, ten_k_context=state.ten_k_context,
                        transcript_context=state.transcript_context,
                        previous_answer=state.best_answer, conversation_context=conversation_context,
                        retry_callback=sync_retry_callback, answer_mode=_am
                    )
                else:
                    refined_answer = self.response_generator.generate_openai_response(
                        question, [c['chunk_text'] for c in iteration_new_chunks], iteration_new_chunks,
                        ticker=tickers_to_process[0] if tickers_to_process else None, stream_callback=None,
                        news_context=state.news_context, ten_k_context=state.ten_k_context,
                        transcript_context=state.transcript_context,
                        previous_answer=state.best_answer, conversation_context=conversation_context,
                        retry_callback=sync_retry_callback, answer_mode=_am
                    )
                await flush_retry_events()
                state.best_answer = refined_answer
                state.best_citations = state.accumulated_citations.copy()
                state.best_context_chunks = [c['chunk_text'] for c in state.accumulated_chunks]
                state.best_chunks = state.accumulated_chunks.copy()
            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
                rag_logger.warning(f"⚠️ LLM API error during refinement at iteration {iteration}: {e}")
                if event_yielder:
                    await event_yielder({'type': 'api_retry', 'message': 'API error during refinement, keeping previous answer', 'step': 'iteration', 'data': {'error': str(e), 'iteration': iteration, 'graceful_fallback': True}})
                await flush_retry_events()
                return False
        return True

    async def _run_iterative_improvement(
        self,
        ctx: RAGFlowContext,
        stream_callback=None,
        event_yielder=None,
    ) -> tuple:
        """Run iterative improvement loop to refine the answer.

        Reads all inputs from ctx; returns (best_answer, best_confidence, best_citations,
        best_context_chunks, best_chunks, evaluation_context, follow_up_questions_asked,
        accumulated_chunks, generation_time). Caller assigns to ctx.
        """
        generation_start = time.time()
        rag_logger.info("🤖 Starting iterative response generation (loop: build state → iterate → stream final)")

        # Fetch conversation context once for stateful follow-up (sliding window of last N exchanges)
        conversation_context = ""
        if ctx.conversation_id and hasattr(self, "question_analyzer"):
            try:
                conversation_context = await self.question_analyzer.conversation_memory.format_context(ctx.conversation_id)
                if conversation_context:
                    rag_logger.info(f"📜 Using conversation history for answer generation ({len(conversation_context)} chars)")
            except Exception as e:
                rag_logger.warning(f"⚠️ Failed to load conversation context: {e}")

        # Queue for retry/API events so we can flush after sync callbacks
        retry_event_queue = []

        def sync_retry_cb(event):
            retry_event_queue.append(event)
            rag_logger.info(f"🔄 Retry event queued: {event.get('message', 'unknown')}")

        async def flush_retry_events():
            if event_yielder and retry_event_queue:
                for event in retry_event_queue:
                    await event_yielder(event)
                retry_event_queue.clear()

        # Build initial answer (or early-return if no context); then run improvement loop
        state, early_return = await self._build_initial_improvement_state(ctx, sync_retry_cb, flush_retry_events, conversation_context)
        if early_return is not None:
            generation_time = time.time() - generation_start
            return (*early_return[:8], generation_time)

        max_iterations = ctx.max_iterations or 3
        for iteration in range(1, max_iterations + 1):
            # _improvement_step returns False to stop (confidence high, no follow-ups, or error)
            if not await self._improvement_step(
                iteration, max_iterations, state, ctx,
                stream_callback=stream_callback,
                event_yielder=event_yielder,
                sync_retry_callback=sync_retry_cb,
                flush_retry_events=flush_retry_events,
                conversation_context=conversation_context,
            ):
                break

        # After loop: stream final answer. Re-use existing best_answer when present to avoid redundant LLM call (no functionality change).
        if stream_callback:
            if event_yielder:
                await event_yielder({
                    'type': 'iteration_final',
                    'message': 'Preparing comprehensive response',
                    'step': 'iteration',
                    'data': {
                        'total_iterations': max_iterations,
                        'final_chunks': len(state.accumulated_chunks),
                        'final_confidence': state.best_confidence
                    }
                })
                await asyncio.sleep(0.01)
            if state.best_answer:
                # Stream the answer token-by-token instead of sending all at once
                # Split by words and stream with slight delay to simulate token streaming
                words = state.best_answer.split(' ')
                for i, word in enumerate(words):
                    if i < len(words) - 1:
                        stream_callback(word + ' ')
                    else:
                        stream_callback(word)
                    # Small delay every few words to avoid overwhelming the stream
                    if i % 5 == 0:
                        await asyncio.sleep(0.01)
            else:
                try:
                    _am = ctx.answer_mode.value if ctx.answer_mode else None
                    if ctx.is_general_question or (ctx.is_multi_ticker and len(ctx.individual_results) > 1):
                        final_answer = self.response_generator.generate_multi_ticker_response(
                            ctx.question, state.accumulated_chunks, ctx.individual_results, ctx.show_details, ctx.comprehensive,
                            stream_callback=stream_callback, news_context=state.news_context, ten_k_context=state.ten_k_context,
                            transcript_context=state.transcript_context,
                            conversation_context=conversation_context, retry_callback=sync_retry_cb,
                            answer_mode=_am
                        )
                    else:
                        final_answer = self.response_generator.generate_openai_response(
                            ctx.question, [c['chunk_text'] for c in state.accumulated_chunks], state.accumulated_chunks,
                            ticker=ctx.tickers_to_process[0] if ctx.tickers_to_process else None, stream_callback=stream_callback,
                            news_context=state.news_context, ten_k_context=state.ten_k_context,
                            transcript_context=state.transcript_context,
                            conversation_context=conversation_context, retry_callback=sync_retry_cb,
                            answer_mode=_am
                        )
                    await flush_retry_events()
                    state.best_answer = final_answer
                except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
                    rag_logger.warning(f"⚠️ LLM API error during final answer generation: {e}")
                    await flush_retry_events()
                    if state.best_answer and event_yielder:
                        await event_yielder({'type': 'api_retry', 'message': 'API error during final generation, returning best answer', 'step': 'final', 'data': {'error': str(e), 'graceful_fallback': True}})
                    if state.best_answer and stream_callback:
                        # Stream the answer token-by-token instead of sending all at once
                        words = state.best_answer.split(' ')
                        for i, word in enumerate(words):
                            if i < len(words) - 1:
                                stream_callback(word + ' ')
                            else:
                                stream_callback(word)
                            if i % 5 == 0:
                                await asyncio.sleep(0.01)
                    elif not state.best_answer:
                        raise

        generation_time = time.time() - generation_start
        logger.info("⏱️  RAG PIPELINE TIMING SUMMARY: generation=%.3fs, iterations=%s, chunks=%s, confidence=%.3f",
                    generation_time, len(state.evaluation_context), len(state.accumulated_chunks), state.best_confidence)
        return (
            state.best_answer,
            state.best_confidence,
            state.best_citations,
            state.best_context_chunks,
            state.best_chunks,
            state.evaluation_context,
            state.follow_up_questions_asked,
            state.accumulated_chunks,
            generation_time,
        )

    # Used by _improvement_step when answer_quality['is_insufficient'] to suggest extra search queries
    def _generate_context_aware_follow_up_questions(self, original_question: str, current_answer: str, available_chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate follow-up questions based on available context and gaps in the answer.
        
        Analyzes the available chunks and current answer to suggest intelligent follow-up
        questions that would help find missing or additional relevant information.
        Considers what data is already available to avoid redundant searches.
        
        Args:
            original_question (str): The user's original question.
            current_answer (str): The currently generated answer.
            available_chunks (List[Dict[str, Any]]): Chunks available in context with structure:
                [
                    {
                        "chunk_text": str,  # The text content
                        "ticker": str,  # Company ticker
                        "distance": float,  # Similarity score
                        "metadata": {
                            "date": str,  # Date of the data
                            ...
                        }
                    },
                    ...
                ]
        
        Returns:
            List[str]: List of 2-3 context-aware follow-up questions, e.g.:
                ["What specific metrics were mentioned?", "Are there any risks discussed?"]
        
        Raises:
            Exception: If OpenAI is not available.
        """
        if not self.response_generator.openai_available:
            raise Exception("OpenAI not available for context-aware follow-up questions")
        if not available_chunks:
            return []
        
        max_retries = 2  # Keep it small
        retry_delay = 0  # No delays
        
        for attempt in range(max_retries):
            try:
                # Get context-aware follow-up prompt from centralized prompts
                analysis_prompt = get_context_aware_followup_prompt(original_question, current_answer, available_chunks)

                # Detailed LLM stage logging for context-aware follow-up questions
                rag_logger.info(f"🤖 ===== CONTEXT-AWARE FOLLOW-UP QUESTIONS LLM CALL ===== (attempt {attempt + 1}/{max_retries})")
                rag_logger.info(f"🔍 Model: {self.config.get('evaluation_model', 'gpt-4.1-mini-2025-04-14')}")
                rag_logger.info(f"📊 Max tokens: 300")
                rag_logger.info(f"🌡️ Temperature: 0.3")
                rag_logger.info(f"📝 Original question: {original_question}")
                rag_logger.info(f"📊 Current answer length: {len(current_answer)} characters")
                rag_logger.info(f"📊 Available chunks count: {len(available_chunks)}")
                rag_logger.info(f"📋 Analysis prompt length: {len(analysis_prompt)} characters")
                rag_logger.info(f"📋 Analysis prompt preview: {analysis_prompt[:300]}...")
                
                start_time = time.time()
                response_text = self.response_generator.llm.complete(
                    [
                        {"role": "system", "content": CONTEXT_AWARE_FOLLOWUP_SYSTEM_PROMPT},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    model=self.config.get("evaluation_model", "gpt-4.1-mini-2025-04-14"),
                    temperature=0.3,
                    max_tokens=300,
                    stream=False,
                )
                call_time = time.time() - start_time
                
                rag_logger.info(f"✅ ===== CONTEXT-AWARE FOLLOW-UP QUESTIONS LLM RESPONSE ===== (call time: {call_time:.3f}s)")
                
                if not response_text or not response_text.strip():
                    raise ValueError(f"Empty response from LLM on attempt {attempt + 1}")
                
                response_text = response_text.strip()
                rag_logger.info(f"📝 Raw context-aware response: {response_text[:100]}...")
                
                # Clean up the response (remove any markdown formatting)
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                

                try:
                    follow_up_questions = json.loads(response_text)
                    if isinstance(follow_up_questions, list) and len(follow_up_questions) > 0:
                        rag_logger.info(f"✅ ===== CONTEXT-AWARE QUESTIONS PARSING SUCCESSFUL =====")
                        rag_logger.info(f"📊 Questions count: {len(follow_up_questions)}")
                        rag_logger.info(f"📝 Generated questions: {follow_up_questions}")
                        rag_logger.info(f"✅ Successfully parsed context-aware questions on attempt {attempt + 1}")
                        return follow_up_questions
                    else:
                        raise ValueError("Invalid or empty list returned")
                except (json.JSONDecodeError, ValueError) as e:
                    rag_logger.warning(f"⚠️ JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        # Return default questions if all retries fail
                        rag_logger.error(f"❌ All {max_retries} JSON parsing attempts failed, returning default questions")
                        return ["What additional details can you provide?", "Are there any specific metrics mentioned?", "What risks or challenges were discussed?"]
                    else:
                        # Immediate retry for JSON parsing issues
                        rag_logger.info(f"🔄 Immediate retry for JSON parsing (attempt {attempt + 2}/{max_retries})")
                        continue
                        
            except Exception as e:
                rag_logger.warning(f"⚠️ Context-aware question generation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Return default questions if all retries fail
                    rag_logger.error(f"❌ All {max_retries} attempts failed, returning default questions")
                    return ["What additional details can you provide?", "Are there any specific metrics mentioned?", "What risks or challenges were discussed?"]
                else:
                    # Immediate retry for API errors
                    rag_logger.info(f"🔄 Immediate retry for API error (attempt {attempt + 2}/{max_retries})")
                    continue

    # -------------------------------------------------------------------------
    # RAG flow stages (async generators: mutate ctx, yield progress/analysis/search/result events)
    # -------------------------------------------------------------------------

    async def _stage_reflection(self, ctx: RAGFlowContext):
        """
        Reflection stage: review Wave 1 results and decide if Wave 2 per-company research is needed.
        Runs after _stage_parallel_multi_source_search, before _stage_prepare_context.
        """
        # Build a compact Wave 1 summary for the LLM
        summary_parts = []
        if ctx.screener_data_rows:
            tickers = [r.get('symbol', '') for r in ctx.screener_data_rows[:15]]
            summary_parts.append(f"Screener: Found {len(ctx.screener_data_rows)} companies: {', '.join(tickers)}")
        if ctx.transcript_per_ticker_results:
            tickers = [r.get('ticker', '') for r in ctx.transcript_per_ticker_results]
            summary_parts.append(f"Transcripts already searched: {', '.join(tickers)}")
        if ctx.sec_service_results:
            tickers = [r.get('ticker', '') for r in ctx.sec_service_results]
            summary_parts.append(f"10-K already searched: {', '.join(tickers)}")
        if ctx.news_results and ctx.news_results.get('results'):
            summary_parts.append(f"News: {len(ctx.news_results['results'])} articles found")

        if not summary_parts:
            return  # nothing to reflect on

        wave1_summary = "\n".join(summary_parts)
        rag_logger.info(f"🔍 Reflection — Wave 1 summary:\n{wave1_summary}")

        try:
            original_data_sources = ctx.question_analysis.get('data_sources', [])
            reflection = await self.reasoning_planner.reflect(ctx.question, wave1_summary, original_data_sources)
            ctx.reflection_result = reflection

            if reflection.needs_more_research and reflection.threads:
                rag_logger.info(f"🔄 Reflection: spawning Wave 2 for {len(reflection.threads)} companies")
                if ctx.stream:
                    yield {
                        'type': 'reasoning',
                        'message': reflection.reasoning,
                        'step': 'reflection',
                        'event_name': 'wave2_plan',
                        'data': {'threads': [{'ticker': t.ticker, 'reason': t.reason} for t in reflection.threads]}
                    }
            else:
                rag_logger.info("✅ Reflection: Wave 1 sufficient, no Wave 2 needed")
        except Exception as e:
            rag_logger.error(f"❌ Reflection stage failed: {e}")

    async def _stage_wave2_search(self, ctx: RAGFlowContext):
        """
        Wave 2: spawn parallel sub-planners (one per company thread), then run their searches in parallel.
        Merges results into ctx alongside Wave 1 data.
        """
        if not ctx.reflection_result or not ctx.reflection_result.needs_more_research:
            return
        threads = ctx.reflection_result.threads
        if not threads:
            return
        # Cap Wave 2 to max 1 company thread for now
        threads = threads[:1]

        if ctx.stream:
            yield {
                'type': 'progress',
                'message': f'Researching {len(threads)} companies in depth...',
                'step': 'wave2_search',
                'data': {'companies': [t.ticker for t in threads]}
            }

        rag_logger.info(f"🚀 Wave 2: {len(threads)} parallel sub-planners")

        # Step 1: run all sub-planner LLM calls in parallel
        sub_plans = await asyncio.gather(
            *[self.reasoning_planner.plan_for_company(ctx.question, t.ticker, t.reason) for t in threads],
            return_exceptions=True
        )

        # Normalize sub_plans (replace exceptions with defaults)
        normalized_plans = []
        for i, (thread, plan) in enumerate(zip(threads, sub_plans)):
            if isinstance(plan, Exception):
                rag_logger.warning(f"Sub-planner failed for {thread.ticker}: {plan}")
                plan = {"ticker": thread.ticker, "data_sources": ["earnings_transcripts"], "time_refs": ["latest"]}
            normalized_plans.append(plan)

        rag_logger.info(f"✅ Wave 2 sub-plans ready: {[(p['ticker'], p['data_sources']) for p in normalized_plans]}")

        # Step 2: run all company searches in parallel, collecting events
        event_queue = asyncio.Queue()

        async def collect(gen):
            async for event in gen:
                await event_queue.put(event)

        async def research_company(sub_plan):
            ticker = sub_plan["ticker"]
            try:
                from .rag_flow_context import RAGFlowContext as _RFC
                sub_ctx = _RFC(
                    question=ctx.question,
                    stream=ctx.stream,
                    show_details=ctx.show_details,
                    comprehensive=ctx.comprehensive,
                    max_iterations=1,
                )
                sub_ctx.question_analysis = {
                    'tickers': [ticker],
                    'extracted_tickers': [ticker],
                    'extracted_ticker': ticker,
                    'topic': ctx.question_analysis.get('topic', ''),
                    'question_type': 'specific_company',
                    'time_refs': sub_plan.get('time_refs', ['latest']),
                    'data_sources': sub_plan.get('data_sources', ['earnings_transcripts']),
                    'original_question': ctx.question,
                    'reasoning_statement': sub_plan.get('reasoning', ''),
                }
                # Search planning (sets sub_ctx.search_plan)
                async for _ in self._stage_search_planning(sub_ctx):
                    pass  # no events needed from sub-planner planning

                # Run searches in parallel within this company
                data_sources = sub_plan.get('data_sources', ['earnings_transcripts'])
                company_tasks = []
                if 'earnings_transcripts' in data_sources:
                    company_tasks.append(asyncio.create_task(collect(self._stage_transcript_search(sub_ctx))))
                if '10k' in data_sources:
                    company_tasks.append(asyncio.create_task(collect(self._stage_10k_search(sub_ctx))))
                if 'news' in data_sources:
                    company_tasks.append(asyncio.create_task(collect(self._stage_news_search(sub_ctx))))
                if company_tasks:
                    await asyncio.gather(*company_tasks, return_exceptions=True)

                # Merge into main ctx
                ctx.transcript_per_ticker_results.extend(sub_ctx.transcript_per_ticker_results)
                ctx.sec_service_results.extend(sub_ctx.sec_service_results)
                ctx.all_chunks.extend(sub_ctx.all_chunks)
                ctx.all_citations.extend(sub_ctx.all_citations)
                if sub_ctx.news_results and sub_ctx.news_results.get('results'):
                    if ctx.news_results and ctx.news_results.get('results'):
                        ctx.news_results['results'].extend(sub_ctx.news_results['results'])
                    else:
                        ctx.news_results = sub_ctx.news_results

                rag_logger.info(f"✅ Wave 2 research complete for {ticker}: {len(sub_ctx.all_chunks)} chunks")
            except Exception as e:
                rag_logger.error(f"❌ Wave 2 research failed for {ticker}: {e}")

        company_tasks = [asyncio.create_task(research_company(p)) for p in normalized_plans]

        async def wait_and_signal():
            await asyncio.gather(*company_tasks, return_exceptions=True)
            await event_queue.put(None)

        done_task = asyncio.create_task(wait_and_signal())

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event

        await done_task
        rag_logger.info(f"✅ Wave 2 complete — ctx now has {len(ctx.all_chunks)} total chunks")

    async def _stage_screener_search(self, ctx: RAGFlowContext):
        """Run qualitative screener search if 'screener' is in data_sources."""
        data_sources = ctx.question_analysis.get('data_sources', []) if ctx.question_analysis else []
        if 'screener' not in data_sources or not self.qualitative_screener:
            return

        if ctx.stream:
            yield {'type': 'progress', 'message': 'Searching across companies...', 'step': 'screener_search', 'data': {}}

        rag_logger.info("🔍 Running screener search for company discovery query")
        try:
            data_rows = []
            synthesis_text = ""
            async for event in self.qualitative_screener.screen_with_streaming(question=ctx.question):
                if ctx.stream:
                    yield event
                if event.get('type') == 'result':
                    result_data = event.get('data', {})
                    data_rows = result_data.get('data_rows', [])
                    synthesis_text = result_data.get('synthesis', '')

            if data_rows:
                ctx.screener_data_rows = data_rows  # store raw rows for reflection
                lines = [f"=== SCREENER RESULTS ===\nFound {len(data_rows)} companies matching your criteria:\n"]
                for i, row in enumerate(data_rows, 1):
                    symbol = row.get('symbol', '')
                    evidence = row.get('evidence_summary', '')
                    lines.append(f"{i}. **{symbol}** — {evidence}")
                if synthesis_text:
                    lines.append(f"\n=== ANALYSIS ===\n{synthesis_text}")
                ctx.screener_context_str = "\n".join(lines)
                rag_logger.info(f"✅ Screener found {len(data_rows)} companies")
            else:
                rag_logger.info("⚠️ Screener returned no results")
        except Exception as e:
            rag_logger.error(f"❌ Screener search failed: {e}")

    async def _stage_parallel_multi_source_search(self, ctx: RAGFlowContext):
        """
        Run news, 10-K, and transcript searches in parallel for maximum performance.
        Collects and yields events from all three sources as they arrive.
        """
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Searching multiple data sources in parallel...', 'step': 'parallel_search', 'data': {}}

        rag_logger.info("🚀 Starting parallel multi-source search (news + 10-K + transcripts)")
        parallel_start = time.time()

        # Create event queue to collect events from all parallel tasks
        event_queue = asyncio.Queue()

        # Helper to collect events from a stage generator
        async def collect_stage_events(stage_coro, stage_name):
            try:
                async for event in stage_coro:
                    await event_queue.put(event)
            except Exception as e:
                rag_logger.error(f"❌ Error in parallel stage {stage_name}: {e}")
                await event_queue.put({
                    'type': 'error',
                    'message': f'Error in {stage_name}: {str(e)}',
                    'step': stage_name,
                    'data': {'error': str(e)}
                })

        # Create tasks for all search stages
        tasks = []

        # Always create tasks, but they'll skip internally if not needed
        news_task = asyncio.create_task(
            collect_stage_events(self._stage_news_search(ctx), 'news_search')
        )
        tasks.append(news_task)

        tenk_task = asyncio.create_task(
            collect_stage_events(self._stage_10k_search(ctx), '10k_search')
        )
        tasks.append(tenk_task)

        transcript_task = asyncio.create_task(
            collect_stage_events(self._stage_transcript_search(ctx), 'transcript_search')
        )
        tasks.append(transcript_task)

        screener_task = asyncio.create_task(
            collect_stage_events(self._stage_screener_search(ctx), 'screener_search')
        )
        tasks.append(screener_task)

        # Sentinel to mark completion
        async def wait_and_mark_done():
            await asyncio.gather(*tasks, return_exceptions=True)
            await event_queue.put(None)  # Sentinel value

        done_task = asyncio.create_task(wait_and_mark_done())

        # Yield events as they arrive from any source
        while True:
            event = await event_queue.get()
            if event is None:  # Sentinel - all tasks complete
                break
            yield event

        # Wait for done task to complete
        await done_task

        parallel_time = time.time() - parallel_start
        rag_logger.info(f"✅ Parallel multi-source search completed in {parallel_time:.3f}s")

        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "rag.parallel_search.complete",
                parallel_time_ms=int(parallel_time * 1000),
                news_searched=ctx.search_plan.has_news() if ctx.search_plan else False,
                tenk_searched=ctx.search_plan.has_10k() if ctx.search_plan else False,
                transcript_searched=ctx.search_plan.has_transcripts() if ctx.search_plan else False
            )

    async def _stage_setup(self, ctx: RAGFlowContext):
        """Stage 1: Setup. Writes: start_time, max_iterations (from config if None). Yields: progress."""
        ctx.start_time = time.time()
        if ctx.max_iterations is None or ctx.max_iterations <= 0:
            ctx.max_iterations = self.config.get("max_iterations", 3)
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Starting analysis...', 'step': 'init', 'data': {}}
        rag_logger.info(f"🚀 Starting complete RAG flow")
        rag_logger.info(f"📝 Question: '{ctx.question}'")
        rag_logger.info(f"🔄 Max iterations: {ctx.max_iterations}")
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "rag.flow.start",
                question=ctx.question,
                max_iterations=ctx.max_iterations,
                conversation_id=ctx.conversation_id,
                comprehensive=ctx.comprehensive,
                stream=ctx.stream
            )
        print(f"\n{'='*80}")
        print(f"🔍 RAG FLOW DEBUG - DETAILED ANALYSIS")
        print(f"{'='*80}")
        print(f"📝 Question: '{ctx.question}'")
        print(f"🔄 Max iterations: {ctx.max_iterations}")
        print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(ctx.start_time))}")
        print(f"{'='*80}")

    async def _stage_question_analysis(self, ctx: RAGFlowContext):
        """
        DEPRECATED: This stage is no longer used. Replaced by _stage_combined_reasoning().

        Old Stage 2: Question analysis. Reads: question, conversation_id. Writes: question_analysis, target_quarters, early_return, analysis_time. May yield rejected/error and return.
        """
        analysis_start = time.time()
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Analyzing question...', 'step': 'analysis', 'data': {}}
        question_analysis, target_quarters, early_return = await self._perform_question_analysis(ctx.question, ctx.conversation_id)
        ctx.analysis_time = time.time() - analysis_start
        ctx.question_analysis = question_analysis
        ctx.target_quarters = target_quarters
        ctx.early_return = early_return

        # Determine answer mode and cap max_iterations accordingly
        if question_analysis:
            from .config import ANSWER_MODE_CONFIG, AnswerMode
            ctx.answer_mode = self._determine_answer_mode(
                question_analysis,
                comprehensive_override=ctx.comprehensive if ctx.comprehensive is not True else None
            )
            mode_config = ANSWER_MODE_CONFIG[ctx.answer_mode]
            ctx.max_iterations = min(ctx.max_iterations, mode_config["max_iterations"])
            rag_logger.info(f"🎯 Answer mode: {ctx.answer_mode.value} (max_iterations={ctx.max_iterations}, max_tokens={mode_config['max_tokens']})")

        if early_return:
            analysis = early_return.get('analysis', {})
            if analysis.get('status') == 'rejected':
                yield {
                    'type': 'rejected',
                    'message': analysis.get('message', 'I can only help with public company financial data.'),
                    'step': 'complete',
                    'data': {
                        'suggestions': analysis.get('suggestions', []),
                        'original_question': early_return.get('original_question', ctx.question)
                    }
                }
                return
            error_msg = early_return.get('error') or (early_return.get('errors', []) or [None])[0] or 'Unknown error'
            yield {'type': 'error', 'message': error_msg, 'step': 'analysis', 'data': early_return}
            return
        if LOGFIRE_AVAILABLE and logfire and question_analysis:
            logfire.info(
                "rag.question_analysis",
                tickers=question_analysis.get('extracted_tickers', []),
                quarter_context=question_analysis.get('quarter_context', 'latest'),
                target_quarters=target_quarters,
                confidence=question_analysis.get('confidence', 0),
                analysis_time_ms=int(ctx.analysis_time * 1000)
            )
        if ctx.stream:
            tickers = question_analysis.get('extracted_tickers', [])
            reasoning = question_analysis.get('reason', '')
            message = reasoning or (
                f"Analyzing {'and '.join(tickers) if len(tickers) <= 2 else f'{len(tickers)} companies'}"
                if tickers else "General financial question detected"
            )
            yield {
                'type': 'analysis',
                'message': message,
                'step': 'analysis',
                'data': {
                    'tickers': tickers,
                    'target_quarters': target_quarters,
                    'quarter_context': question_analysis.get('quarter_context', 'latest'),
                    'reasoning': reasoning,
                    'question_type': question_analysis.get('question_type', ''),
                    'confidence': question_analysis.get('confidence', 0),
                    'answer_mode': ctx.answer_mode.value if ctx.answer_mode else 'standard'
                }
            }

    async def _stage_combined_reasoning(self, ctx: RAGFlowContext):
        """
        NEW Stage: Combined Reasoning + Analysis (replaces stages 2 & 4).

        Single LLM call that:
        1. Analyzes the question (extracts tickers, topic, time refs)
        2. Explains the research approach (reasoning)
        3. Outputs structured metadata for search planning

        Reads: question, conversation_id
        Writes: reasoning_result, reasoning_statement, question_analysis (compatible format)
        """
        reasoning_start = time.time()
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Analyzing question and planning research...', 'step': 'reasoning', 'data': {}}

        rag_logger.info("🧠 Starting combined reasoning & analysis...")

        # Call new ReasoningPlanner
        try:
            reasoning_result = await self.reasoning_planner.create_reasoning_plan(
                ctx.question,
                ctx.conversation_id
            )

            ctx.reasoning_time = time.time() - reasoning_start
            ctx.reasoning_result = reasoning_result
            ctx.reasoning_statement = reasoning_result.reasoning

            # Check if valid — non-research intents (greetings, capability, off-topic)
            # respond directly as a chat message without spawning any searches
            if not reasoning_result.is_valid:
                answer = reasoning_result.validation_message or "I can help with public company financial research — earnings calls, 10-K filings, news, and company discovery. What would you like to explore?"
                ctx.final_result = {
                    'answer': answer,
                    'confidence': 1.0,
                    'citations': [],
                    'context_chunks': [],
                    'iterations': [],
                    'chunks': [],
                    'timing': {'total': time.time() - ctx.start_time},
                }
                ctx.early_return = True
                return

            # Convert to question_analysis format for backward compatibility
            ctx.question_analysis = {
                'tickers': reasoning_result.tickers,  # Primary field for search planner
                'extracted_tickers': reasoning_result.tickers,
                'extracted_ticker': reasoning_result.tickers[0] if reasoning_result.tickers else None,
                'topic': reasoning_result.topic,
                'question_type': reasoning_result.question_type,
                'time_refs': reasoning_result.time_refs,  # ✅ PRESERVED EXACTLY!
                'user_hints': {
                    'answer_mode': reasoning_result.answer_mode,
                    'data_source': reasoning_result.data_sources[0] if reasoning_result.data_sources else None
                },
                'confidence': reasoning_result.confidence,
                'reasoning_statement': reasoning_result.reasoning,
                'data_sources': reasoning_result.data_sources
            }

            # Determine answer mode
            from .config import AnswerMode
            mode_str = reasoning_result.answer_mode
            if mode_str == 'direct':
                ctx.answer_mode = AnswerMode.DIRECT
            elif mode_str == 'detailed':
                ctx.answer_mode = AnswerMode.DETAILED
            else:
                ctx.answer_mode = AnswerMode.STANDARD

            # Cap max_iterations based on answer mode
            from .config import ANSWER_MODE_CONFIG
            mode_config = ANSWER_MODE_CONFIG[ctx.answer_mode]
            ctx.max_iterations = min(ctx.max_iterations, mode_config["max_iterations"])

            rag_logger.info(f"✅ Combined reasoning completed in {ctx.reasoning_time:.3f}s")
            rag_logger.info(f"🎯 Answer mode: {ctx.answer_mode.value} (max_iterations={ctx.max_iterations})")
            rag_logger.info(f"📊 Tickers: {reasoning_result.tickers}, Time refs: {reasoning_result.time_refs}")

            # Yield reasoning to user
            if ctx.stream:
                yield {
                    'type': 'reasoning',
                    'message': reasoning_result.reasoning,
                    'step': 'reasoning',
                    'event_name': 'research_approach',
                    'data': {
                        'tickers': reasoning_result.tickers,
                        'time_refs': reasoning_result.time_refs,
                        'data_sources': reasoning_result.data_sources,
                        'answer_mode': reasoning_result.answer_mode,
                        'reasoning': reasoning_result.reasoning
                    }
                }

        except Exception as e:
            rag_logger.error(f"❌ Combined reasoning failed: {e}")
            yield {'type': 'error', 'message': f'Failed to analyze question: {str(e)}', 'step': 'reasoning', 'data': {'error': str(e)}}
            ctx.early_return = True
            return

    async def _stage_search_planning(self, ctx: RAGFlowContext):
        """Stage 2.05: Search Planning. Creates declarative search plan based on question analysis. Reads: question_analysis. Writes: search_plan. Yields: reasoning event with search plan."""
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Creating search plan...', 'step': 'search_planning', 'data': {}}

        rag_logger.info("🎯 Creating search plan based on question analysis...")

        try:
            # Add original question to analysis for LLM-based data source routing
            ctx.question_analysis['original_question'] = ctx.question

            # Create search plan using SearchPlanner (now 100% LLM-based, no keywords)
            search_plan = self.search_planner.create_plan(
                question_analysis=ctx.question_analysis,
                user_preferences=None  # Future: pass user preferences
            )

            # Store search plan in context
            ctx.search_plan = search_plan

            # Log search plan creation
            rag_logger.info(f"✅ Search plan created: {search_plan.total_searches()} searches "
                          f"(transcripts={len(search_plan.earnings_transcripts)}, "
                          f"10k={len(search_plan.ten_k)}, "
                          f"news={len(search_plan.news)})")

            # Yield reasoning event with search plan explanation
            if ctx.stream:
                yield {
                    'type': 'reasoning',
                    'message': search_plan.reasoning,
                    'step': 'search_planning',
                    'event_name': 'search_planning',
                    'data': {
                        'reasoning': search_plan.reasoning,
                        'total_searches': search_plan.total_searches(),
                        'has_transcripts': search_plan.has_transcripts(),
                        'has_10k': search_plan.has_10k(),
                        'has_news': search_plan.has_news()
                    }
                }

            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "rag.search_planning.complete",
                    reasoning=search_plan.reasoning,
                    total_searches=search_plan.total_searches(),
                    transcript_searches=len(search_plan.earnings_transcripts),
                    ten_k_searches=len(search_plan.ten_k),
                    news_searches=len(search_plan.news)
                )

        except DatabaseConnectionError as e:
            # Database connection error - propagate to frontend as error event
            rag_logger.error(f"❌ Database connection error during search planning: {e.technical_message}")
            if ctx.stream:
                yield {
                    'type': 'error',
                    'message': e.user_message,
                    'step': 'search_planning',
                    'event_name': 'search_planning_error'
                }
            # Re-raise to stop the flow
            raise
        except Exception as e:
            rag_logger.error(f"❌ Search planning failed: {e}")
            # Create fallback empty search plan
            from .search_planner import SearchPlan
            ctx.search_plan = SearchPlan(
                earnings_transcripts=[],
                ten_k=[],
                news=[],
                reasoning="Using default search strategy due to planning error"
            )
            if ctx.stream:
                yield {
                    'type': 'reasoning',
                    'message': "Using default search strategy",
                    'step': 'search_planning',
                    'event_name': 'search_planning',
                    'data': {'error': str(e)}
                }

    async def _stage_planning(self, ctx: RAGFlowContext):
        """
        DEPRECATED: This stage is no longer used. Replaced by _stage_combined_reasoning().

        Old Stage 2.1: Planning. Reads: question, question_analysis, conversation_id. Writes: reasoning_statement, question_analysis['reasoning_statement']. Yields: reasoning.
        """
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Planning research approach...', 'step': 'planning', 'data': {}}
        conversation_context = ""
        if ctx.conversation_id and hasattr(self, "question_analyzer"):
            try:
                conversation_context = await self.question_analyzer.conversation_memory.format_context(ctx.conversation_id)
                if conversation_context:
                    rag_logger.info(f"📜 Using conversation history for planning ({len(conversation_context)} chars)")
            except Exception as e:
                rag_logger.warning(f"⚠️ Failed to load conversation context for planning: {e}")
        rag_logger.info("🧠 Generating research reasoning...")
        try:
            reasoning_statement = await self.response_generator.plan_question_approach(
                ctx.question, ctx.question_analysis, conversation_context=conversation_context
            )
            ctx.reasoning_statement = reasoning_statement
            ctx.question_analysis['reasoning_statement'] = reasoning_statement
            if ctx.stream:
                yield {
                    'type': 'reasoning',
                    'message': reasoning_statement,
                    'step': 'planning',
                    'data': {'reasoning': reasoning_statement}
                }
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info("rag.reasoning.complete", reasoning=reasoning_statement)
        except Exception as e:
            rag_logger.error(f"❌ Reasoning failed: {e}")
            tickers = [normalize_ticker(t) for t in ctx.question_analysis.get('extracted_tickers', []) if t]
            ticker_text = f" for {', '.join(tickers)}" if tickers else ""
            ctx.reasoning_statement = f"The user is asking about{ticker_text}: {ctx.question}. I will search the available financial data to answer this."
            ctx.question_analysis['reasoning_statement'] = ctx.reasoning_statement

    async def _stage_news_search(self, ctx: RAGFlowContext):
        """Stage 2.5: News search. Runs only if search_plan includes news searches. Writes: news_results. Yields: progress, news_search."""
        # Check if search plan includes news searches
        if not ctx.search_plan or not ctx.search_plan.has_news():
            return

        if not self.tavily_service.is_available():
            if ctx.stream:
                yield {
                    'type': 'news_search',
                    'message': 'News search unavailable - Tavily not configured',
                    'step': 'news_search',
                    'data': {'error': 'Tavily service not available. Install tavily-python and set TAVILY_API_KEY environment variable.'}
                }
            return

        if ctx.stream:
            yield {'type': 'progress', 'message': 'Searching latest news...', 'step': 'news_search', 'data': {}}

        rag_logger.info("📰 Search plan includes news - searching Tavily...")

        # Use query from search plan
        news_search = ctx.search_plan.news[0]  # Take first news search
        news_query = news_search.query

        rag_logger.info(f"📰 News query: {news_query}")

        news_results = self.tavily_service.search_news(news_query, max_results=5, include_answer="advanced")
        ctx.news_results = news_results

        if news_results.get("results"):
            rag_logger.info(f"✅ Found {len(news_results['results'])} news articles")
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info("rag.news_search", query=news_query, articles_found=len(news_results['results']))
            if ctx.stream:
                _articles = [{'title': r.get('title', ''), 'url': r.get('url', ''), 'published_date': r.get('published_date', '')} for r in news_results['results'] if r.get('url')]
                yield {
                    'type': 'news_search',
                    'message': f'Found {len(news_results["results"])} recent news articles',
                    'step': 'news_search',
                    'data': {'articles_count': len(news_results["results"]), 'query': news_query, 'articles': _articles}
                }
        else:
            ctx.news_results = None

    async def _stage_10k_search(self, ctx: RAGFlowContext):
        """Stage 2.6: 10-K search. Runs if search_plan includes 10-K searches. Writes: ten_k_results. Yields: progress, reasoning, 10k_search."""
        # Check if search plan includes 10-K searches
        if not ctx.search_plan or not ctx.search_plan.has_10k():
            return

        if ctx.stream:
            yield {'type': 'progress', 'message': 'Searching 10-K filings...', 'step': '10k_search', 'data': {}}
            # Emit document chips immediately with the planned docs (before search starts)
            _planned_docs = [{'ticker': normalize_ticker(s.ticker), 'fiscal_year': s.year, 'filing_type': '10-K'} for s in ctx.search_plan.ten_k]
            if _planned_docs:
                yield {
                    'type': '10k_search',
                    'message': f"Searching {len(_planned_docs)} 10-K filing{'s' if len(_planned_docs) > 1 else ''}...",
                    'step': '10k_search',
                    'data': {'chunks_found': 0, 'tickers_processed': 0, 'companies_found': 0, 'documents': _planned_docs}
                }

        rag_logger.info(f"📄 Search plan includes 10-K - searching SEC filings...")
        rag_logger.info(f"📄 10-K searches: {len(ctx.search_plan.ten_k)}")

        # Run all 10-K searches in parallel
        async def _run_single_10k_search(ten_k_search):
            ticker = normalize_ticker(ten_k_search.ticker)
            fiscal_year = ten_k_search.year
            query = ten_k_search.query
            rag_logger.info(f"📄 Searching 10-K for {ticker} (year={fiscal_year}, query='{query[:50]}...')")
            query_embedding = await self.search_engine.encode_query_async(query)
            chunks = []
            sec_answer = ''
            stream_events = []
            try:
                async for event in self.sec_service.execute_smart_parallel_search_async(
                    query=query,
                    query_embedding=query_embedding,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    max_iterations=7,
                    confidence_threshold=0.85,
                    event_yielder=ctx.stream,
                    embedding_function=self.search_engine.embedding_model.encode
                ):
                    event_type = event.get('type', '')
                    event_data = event.get('data', {})
                    if event_type == 'search_complete':
                        chunks = event_data.get('chunks', [])
                        sec_answer = event_data.get('answer', '')
                        rag_logger.info(f"✅ Found {len(chunks)} 10-K chunks for {ticker} via parallel search")
                    elif ctx.stream:
                        stream_events.append((ticker, event_type, event_data))
            except Exception as e:
                rag_logger.warning(f"⚠️ Failed to search 10-K for {ticker}: {e}")
            return chunks, sec_answer, stream_events, ticker, fiscal_year

        all_results = await asyncio.gather(*[_run_single_10k_search(s) for s in ctx.search_plan.ten_k])

        sec_citation_offset = 0
        for chunks, sec_answer, stream_events, ticker, fiscal_year in all_results:
            if chunks:
                ctx.ten_k_results.extend(chunks)
            if chunks and sec_answer and sec_answer.strip():
                # Build remapped per-ticker SEC result for unified subagent architecture
                raw_citations = self.sec_service.get_10k_citations(chunks)
                remapped_answer, remapped_cits = SECFilingsService._remap_sec_citations(
                    sec_answer, raw_citations, sec_citation_offset
                )
                ctx.sec_service_results.append({
                    'type': '10k',
                    'ticker': ticker,
                    'fiscal_year': fiscal_year,
                    'answer': remapped_answer,
                    'citations': remapped_cits,
                    'chunks': chunks,
                })
                sec_citation_offset += len(raw_citations)
            for ticker, event_type, event_data in stream_events:
                if event_type == 'planning_start':
                    yield {'type': 'reasoning', 'message': f"Looking at {ticker}'s annual report...", 'step': '10k_planning', 'event_name': '10k_planning_start', 'data': {'ticker': ticker, 'phase': 'planning'}}
                elif event_type == 'planning_complete':
                    sub_questions = event_data.get('sub_questions', [])
                    if sub_questions:
                        questions_text = "\n".join([f"- {q}" for q in sub_questions[:4]])
                        yield {'type': 'reasoning', 'message': f"To answer this, I need to find:\n{questions_text}", 'step': '10k_planning', 'event_name': '10k_sub_questions', 'data': {'sub_questions': sub_questions}}
                elif event_type == 'retrieval_complete':
                    new_chunks = event_data.get('new_chunks', 0)
                    if new_chunks > 0:
                        yield {'type': 'reasoning', 'message': f"Found {new_chunks} relevant sections in the filing", 'step': '10k_retrieval', 'event_name': '10k_retrieval_progress', 'data': event_data}
                elif event_type == 'evaluation_complete':
                    quality = event_data.get('quality_score', 0)
                    missing = event_data.get('missing_info', [])
                    if quality >= 0.8:
                        yield {'type': 'reasoning', 'message': "I have enough information to answer this question", 'step': '10k_evaluation', 'event_name': '10k_evaluation_complete', 'data': event_data}
                    elif missing:
                        yield {'type': 'reasoning', 'message': f"Still looking for: {missing[0] if missing else 'more details'}...", 'step': '10k_evaluation', 'event_name': '10k_evaluation_progress', 'data': event_data}

        # Collect tickers that were searched
        searched_tickers = [normalize_ticker(s.ticker) for s in ctx.search_plan.ten_k]

        if ctx.ten_k_results:
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info("rag.10k_search", tickers=searched_tickers, chunks_found=len(ctx.ten_k_results), companies_found=len(set(c.get('ticker', '') for c in ctx.ten_k_results)))
            if ctx.stream:
                # Build unique document list for frontend linking
                _seen_docs = set()
                _documents = []
                for _c in ctx.ten_k_results:
                    _key = (_c.get('ticker', ''), _c.get('fiscal_year'), _c.get('filing_type', '10-K'))
                    if _key not in _seen_docs:
                        _seen_docs.add(_key)
                        _documents.append({'ticker': _c.get('ticker', ''), 'fiscal_year': _c.get('fiscal_year'), 'filing_type': _c.get('filing_type', '10-K')})
                yield {
                    'type': '10k_search',
                    'message': f'Found {len(ctx.ten_k_results)} relevant passages from {len(set(c.get("ticker", "") for c in ctx.ten_k_results))} companies',
                    'step': '10k_search',
                    'data': {'chunks_found': len(ctx.ten_k_results), 'tickers_processed': len(searched_tickers), 'companies_found': len(set(c.get('ticker', '') for c in ctx.ten_k_results)), 'documents': _documents}
                }
        else:
            if ctx.stream:
                yield {'type': '10k_search', 'message': 'Found 0 relevant passages from 0 companies', 'step': '10k_search', 'data': {'chunks_found': 0, 'tickers_processed': len(searched_tickers), 'companies_found': 0, 'documents': []}}

    async def _stage_transcript_search(self, ctx: RAGFlowContext):
        """Stage 3: Transcript search. Skips if search_plan has no transcript searches. Calls _execute_search; writes individual_results, all_chunks, all_citations, search_time, etc. Yields: progress, search."""
        # Check if search plan includes transcript searches
        if not ctx.search_plan or not ctx.search_plan.has_transcripts():
            rag_logger.info(f"📋 Search plan has no transcript searches - skipping initial transcript search.")
            ctx.skip_initial_transcript_search = True
            ctx.individual_results = []
            ctx.all_chunks = []
            ctx.all_citations = []
            ctx.search_time = 0.0
            ctx.is_general_question = False
            ctx.is_multi_ticker = False
            ctx.tickers_to_process = [normalize_ticker(t) for t in ctx.question_analysis.get('extracted_tickers', []) if t]
            ctx.target_quarter = None

            # Check if we should increase max_iterations for 10-K queries
            if ctx.search_plan and ctx.search_plan.has_10k():
                from .config import AnswerMode
                if ctx.answer_mode == AnswerMode.DETAILED:
                    sec_max = self.config.get("sec_max_iterations", 6)
                    if ctx.max_iterations < sec_max:
                        rag_logger.info(f"📈 SEC/10-K query (detailed mode) - increasing max_iterations from {ctx.max_iterations} to {sec_max}")
                        ctx.max_iterations = sec_max
            return

        ctx.skip_initial_transcript_search = False
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Searching documents...', 'step': 'search', 'data': {}}
            # Emit document chips immediately with planned transcripts (before search starts)
            _planned_trans = []
            for _ts in ctx.search_plan.earnings_transcripts:
                for _q_str in _ts.quarters[:2]:  # show up to 2 quarters per ticker
                    _parts = _q_str.split('_')
                    if len(_parts) == 2:
                        _planned_trans.append({'ticker': normalize_ticker(_ts.ticker), 'year': _parts[0], 'quarter': _parts[1].replace('q', '')})
            if _planned_trans:
                yield {
                    'type': 'search',
                    'message': f"Searching earnings transcripts...",
                    'step': 'search',
                    'data': {'chunks_found': 0, 'tickers_processed': 0, 'documents': _planned_trans}
                }

        # ── One transcript agent per ticker, running in parallel (mirrors _stage_10k_search) ──
        logger.info("=" * 80)
        logger.info("🔍 STARTING TRANSCRIPT SERVICE (per-ticker parallel agents)")
        logger.info("=" * 80)
        search_phase_start = time.time()

        async def _run_one_ticker(ts):
            """Drive transcript_service for a single ticker; return result dict."""
            result = {
                'ticker': normalize_ticker(ts.ticker),
                'answer': '',
                'chunks': [],
                'citations': [],
                'sub_questions': [],
            }
            async for event in self.transcript_service.execute_search_async(
                query=ctx.question,
                question_analysis=ctx.question_analysis,
                transcript_searches=[ts],
            ):
                if event.get('type') == 'search_complete':
                    data = event.get('data', {})
                    result['chunks'] = data.get('chunks', [])
                    result['answer'] = data.get('answer', '')
                    result['sub_questions'] = data.get('sub_questions', [])
                    result['citations'] = self.transcript_service.get_citations(result['chunks'])
            return result

        ticker_results_raw = await asyncio.gather(
            *[_run_one_ticker(ts) for ts in ctx.search_plan.earnings_transcripts]
        )

        # Filter tickers that returned nothing
        valid_results = [r for r in ticker_results_raw if r['answer'] and r['chunks']]

        # ── Remap [TC-N] citations across tickers so numbers are globally unique ──
        all_sub_questions = []
        all_chunks = []
        all_citations = []
        remapped_results = []   # for multi-ticker synthesis
        offset = 0

        for result in valid_results:
            remapped_answer, remapped_cits = self.transcript_service._remap_citations(
                result['answer'], result['citations'], offset
            )
            remapped_results.append({
                'ticker': result['ticker'],
                'answer': remapped_answer,
                'citations': remapped_cits,
                'chunks': result['chunks'],
            })
            all_chunks.extend(result['chunks'])
            all_citations.extend(remapped_cits)
            all_sub_questions.extend(result['sub_questions'])
            offset += len(result['citations'])

        ctx.all_chunks = all_chunks
        ctx.all_citations = all_citations

        # Store per-ticker results; _stage_prepare_context decides single vs multi path
        ctx.transcript_per_ticker_results = remapped_results
        ctx.skip_transcript_follow_up = True

        # Stream planning reasoning to frontend (aggregate across all tickers)
        if ctx.stream and all_sub_questions:
            unique_sub_qs = list(dict.fromkeys(all_sub_questions))[:4]  # dedup, keep order
            questions_text = "\n".join([f"- {q}" for q in unique_sub_qs])
            yield {
                'type': 'reasoning',
                'message': f"To answer this, I need to find:\n{questions_text}",
                'step': 'transcript_planning',
                'data': {'sub_questions': unique_sub_qs},
            }
        if ctx.stream and ctx.all_chunks:
            yield {
                'type': 'reasoning',
                'message': f"Found {len(ctx.all_chunks)} relevant transcript passages across {len(valid_results)} {'company' if len(valid_results) == 1 else 'companies'}",
                'step': 'transcript_retrieval',
                'data': {'chunks_found': len(ctx.all_chunks)},
            }

        ctx.search_time = time.time() - search_phase_start
        logger.info(f"🔍 TRANSCRIPT AGENTS COMPLETED in {ctx.search_time:.3f}s — {len(valid_results)} ticker(s), {len(ctx.all_chunks)} chunks")
        logger.info("=" * 80)

        # Populate fields used by downstream stages
        ctx.individual_results = ctx.all_chunks
        ctx.tickers_to_process = [r['ticker'] for r in valid_results]
        ctx.is_multi_ticker = len(ctx.tickers_to_process) > 1
        ctx.is_general_question = ctx.is_multi_ticker

        if LOGFIRE_AVAILABLE and logfire:
            logfire.info("rag.transcript_search", chunks_found=len(ctx.all_chunks), tickers=ctx.tickers_to_process, target_quarters=ctx.target_quarters, is_multi_ticker=ctx.is_multi_ticker, search_time_ms=int(ctx.search_time * 1000))

        if ctx.stream:
            transcripts = {}
            for chunk in ctx.all_chunks:
                ticker = chunk.get('ticker', 'Unknown')
                year, quarter = chunk.get('year', ''), chunk.get('quarter', '')
                if ticker not in transcripts:
                    transcripts[ticker] = set()
                if year and quarter:
                    transcripts[ticker].add(f"Q{quarter} {year}")
            transcripts_sorted = {t: sorted(list(qs), reverse=True)[:3] for t, qs in transcripts.items()}
            # Build unique transcript document list for frontend linking
            _seen_trans = set()
            _trans_docs = []
            for _c in ctx.all_chunks:
                _t, _y, _q = _c.get('ticker', ''), _c.get('year', ''), _c.get('quarter', '')
                if _t and _y and _q and (_t, _y, _q) not in _seen_trans:
                    _seen_trans.add((_t, _y, _q))
                    _trans_docs.append({'ticker': _t, 'year': _y, 'quarter': _q})
            yield {
                'type': 'search',
                'message': f"Found {len(ctx.all_chunks)} relevant passages from {len(transcripts)} {'company' if len(transcripts) == 1 else 'companies'}",
                'step': 'search',
                'data': {'chunks_found': len(ctx.all_chunks), 'tickers_processed': len(ctx.tickers_to_process) if ctx.tickers_to_process else 0, 'transcripts': transcripts_sorted, 'documents': _trans_docs}
            }

    def _stage_prepare_context(self, ctx: RAGFlowContext):
        """Stage 4: Prepare context. Writes: news_context_str, ten_k_context_str, combined_citations (all_citations + news + 10-K). No yields."""
        if ctx.news_results and ctx.news_results.get("results"):
            ctx.news_context_str = self.tavily_service.format_news_context(ctx.news_results)
            news_citations = self.tavily_service.get_news_citations(ctx.news_results)
            rag_logger.info(f"📰 Initial news search returned {len(news_citations)} citations")
        else:
            news_citations = []
        # ── Unified subagent counting ───────────────────────────────────────
        # Transcript subagents: one per ticker in transcript_per_ticker_results
        # SEC subagents: one per ticker in sec_service_results
        # News (Tavily): counts as a subagent — if present alongside others it triggers synthesis
        has_news = bool(news_citations)
        total_agents = len(ctx.transcript_per_ticker_results) + len(ctx.sec_service_results) + (1 if has_news else 0)

        if total_agents == 1 and ctx.transcript_per_ticker_results:
            # Single transcript subagent — pass as context to main agent so follow-up
            # questions are generated once by the response generator (not the subagent)
            ctx.skip_improvement = False
            result = ctx.transcript_per_ticker_results[0]
            ctx.transcript_service_answer = result['answer']
            ctx.all_citations = result['citations']
            ctx.all_chunks = result['chunks']
            import re as _re
            clean_answer = _re.split(r'\*\*You might also ask', result['answer'], flags=_re.IGNORECASE)[0].rstrip()
            ctx.transcript_context_str = f"=== EARNINGS TRANSCRIPT ANALYSIS ===\n{clean_answer}"
            rag_logger.info(f"📝 Single transcript subagent — passing to response generator ({len(clean_answer)} chars)")
        elif total_agents == 1 and ctx.sec_service_results:
            # SEC single-agent: pass pre-analyzed answer as context to main agent for synthesis
            ctx.skip_improvement = False
            result = ctx.sec_service_results[0]
            ctx.all_citations = result['citations']
            ctx.all_chunks = result['chunks']
            ctx.ten_k_context_str = f"=== SEC 10-K ANALYSIS ===\n{result['answer']}"
            rag_logger.info(f"📄 Single SEC subagent — passing to main agent ({len(result['answer'])} chars)")
        elif total_agents == 1 and has_news:
            # News-only: no pre-generated answer, main agent generates from news context
            ctx.skip_improvement = False
            rag_logger.info("📰 News-only query — main agent generates answer from news context")
        elif total_agents > 1:
            # Multi-agent synthesis: transcript/SEC answers + optional news context
            ctx.skip_improvement = False
            all_subagent_chunks = []
            all_subagent_citations = []
            for r in ctx.transcript_per_ticker_results:
                all_subagent_chunks.extend(r['chunks'])
                all_subagent_citations.extend(r['citations'])
            for r in ctx.sec_service_results:
                all_subagent_chunks.extend(r['chunks'])
                all_subagent_citations.extend(r['citations'])
            ctx.all_chunks = all_subagent_chunks
            ctx.all_citations = all_subagent_citations
            import re as _re2
            def _strip_followup(text):
                return _re2.split(r'\*\*You might also ask', text, flags=_re2.IGNORECASE)[0].rstrip()

            if ctx.transcript_per_ticker_results:
                combined_tc = "\n\n".join(
                    f"=== {r['ticker']} (Earnings Transcripts) ===\n{_strip_followup(r['answer'])}"
                    for r in ctx.transcript_per_ticker_results
                )
                ctx.transcript_context_str = combined_tc
            if ctx.sec_service_results:
                def _10k_label(r):
                    fy = r.get('fiscal_year')
                    fy_str = (' FY' + str(fy)) if fy else ''
                    return f"=== {r['ticker']}{fy_str} (10-K Filing) ===\n{_strip_followup(r['answer'])}"
                combined_10k = "\n\n".join(_10k_label(r) for r in ctx.sec_service_results)
                ctx.ten_k_context_str = combined_10k
            n_tc = len(ctx.transcript_per_ticker_results)
            n_10k = len(ctx.sec_service_results)
            rag_logger.info(f"🔀 Multi-agent synthesis: {n_tc} transcript + {n_10k} SEC + {'news' if has_news else 'no news'}")
        else:
            # No subagents at all (empty context)
            ctx.skip_improvement = False
            # Legacy fallback: raw 10-K chunks if SEC service didn't produce a result
            if ctx.ten_k_results and not ctx.sec_service_results:
                ctx.ten_k_context_str = self.sec_service.format_10k_context(ctx.ten_k_results)
                rag_logger.info(f"📄 Using formatted raw 10-K chunks ({len(ctx.ten_k_context_str)} chars)")

        # NOTE: screener context is handled separately in _build_initial_improvement_state
        # (screener-only queries get a clean fast-path answer, not merged into 10-K context)

        # Build combined citations (subagent citations + news)
        ten_k_citations = []
        if ctx.ten_k_results and not ctx.sec_service_results:
            ten_k_citations = self.sec_service.get_10k_citations(ctx.ten_k_results)

        ctx.combined_citations = ctx.all_citations.copy()
        for c in news_citations:
            ctx.combined_citations.append({"type": "news", "marker": f"[N{c['index']}]", "title": c["title"], "url": c["url"], "published_date": c.get("published_date", "")})
        for c in ten_k_citations:
            ctx.combined_citations.append(c)
        rag_logger.info(f"📎 Final combined citations: {len(ctx.combined_citations)} total (agents={total_agents}, skip_improvement={ctx.skip_improvement})")

    async def _stage_run_improvement(self, ctx: RAGFlowContext):
        """Stage 5: Run improvement. Calls _run_iterative_improvement (build initial answer + loop). Writes: improvement_results, then best_answer, best_confidence, best_citations, etc. Yields: progress, tokens, iteration events."""
        if ctx.stream:
            yield {'type': 'progress', 'message': 'Generating response...', 'step': 'generation', 'data': {}}
        token_queue = asyncio.Queue()
        generation_complete = asyncio.Event()
        loop = asyncio.get_running_loop()

        def token_callback(content: str):
            loop.call_soon_threadsafe(token_queue.put_nowait, content)

        async def iteration_event_yielder(event):
            await iteration_event_queue.put(event)

        iteration_event_queue = asyncio.Queue()
        generation_result = None

        async def run_generation():
            nonlocal generation_result
            try:
                generation_result = await self._run_iterative_improvement(
                    ctx, stream_callback=token_callback, event_yielder=iteration_event_yielder
                )
            except Exception as e:
                logger.error(f"Error in token streaming generation: {e}", exc_info=True)
                generation_result = {'success': False, 'error': str(e)}
            finally:
                generation_complete.set()

        if ctx.stream:
            gen_task = asyncio.create_task(run_generation())
            try:
                while not generation_complete.is_set() or not token_queue.empty() or not iteration_event_queue.empty():
                    try:
                        iteration_event = await asyncio.wait_for(iteration_event_queue.get(), timeout=0.001)
                        yield iteration_event
                        continue
                    except asyncio.TimeoutError:
                        pass
                    try:
                        token = await asyncio.wait_for(token_queue.get(), timeout=0.01)
                        yield {'type': 'token', 'content': token, 'step': 'generation', 'data': {}}
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.001)
                        continue
            except Exception as e:
                logger.error(f"Error streaming tokens and iteration events: {e}", exc_info=True)
            await gen_task
            ctx.improvement_results = generation_result
        else:
            ctx.improvement_results = await self._run_iterative_improvement(ctx, stream_callback=None, event_yielder=None)
        if isinstance(ctx.improvement_results, dict) and not ctx.improvement_results.get('success'):
            ctx.improvement_error = True
            return
        (ctx.best_answer, ctx.best_confidence, ctx.best_citations, ctx.best_context_chunks, ctx.best_chunks,
         ctx.evaluation_context, ctx.follow_up_questions_asked, ctx.accumulated_chunks, ctx.generation_time) = ctx.improvement_results

    def _stage_finalize(self, ctx: RAGFlowContext):
        """Stage 6: Finalize. Applies limit message, dedupes citations/chunks, builds response_data and final_result, updates conversation memory. No yields."""
        total_time = time.time() - ctx.start_time
        logger.info("=" * 80)
        logger.info("⏱️  COMPLETE RAG PIPELINE TIMING BREAKDOWN")
        logger.info("=" * 80)
        logger.info(f"📝 Question: {ctx.question[:80]}...")
        logger.info(f"⏱️  ANALYSIS: {ctx.analysis_time:.3f}s | SEARCH: {ctx.search_time:.3f}s | GENERATION: {ctx.generation_time:.3f}s | TOTAL: {total_time:.3f}s")
        logger.info("=" * 80)
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info("rag.flow.complete", total_time_ms=int(total_time * 1000), analysis_time_ms=int(ctx.analysis_time * 1000), search_time_ms=int(ctx.search_time * 1000), generation_time_ms=int(ctx.generation_time * 1000), max_iterations=ctx.max_iterations, actual_iterations=len(ctx.evaluation_context), chunks_found=len(ctx.all_chunks), confidence=ctx.best_confidence, tickers=ctx.tickers_to_process, data_source=ctx.question_analysis.get('data_source', 'earnings_transcripts'), answer_length=len(ctx.best_answer or ''))
        if ctx.question_analysis.get('limits_exceeded') and ctx.best_answer and not ctx.best_answer.startswith("⚠️"):
            limit_message = generate_user_friendly_limit_message(ctx.question_analysis['limits_exceeded'])
            if limit_message:
                ctx.best_answer = limit_message + ctx.best_answer
        # Include ALL sources (initial + iterative follow-ups) across transcripts, 10-K, and news
        combined = ctx.combined_citations if hasattr(ctx, 'combined_citations') and ctx.combined_citations else []
        best = ctx.best_citations if hasattr(ctx, 'best_citations') and ctx.best_citations else []
        accumulated = ctx.accumulated_citations if hasattr(ctx, 'accumulated_citations') and ctx.accumulated_citations else []
        citations_to_dedupe = combined + best + accumulated

        # Use accumulated_chunks to include ALL chunks that contributed to context, not just best_chunks
        chunks_to_dedupe = ctx.accumulated_chunks if hasattr(ctx, 'accumulated_chunks') and ctx.accumulated_chunks else ctx.best_chunks
        unique_citations, unique_chunks = deduplicate_citations_and_chunks(citations_to_dedupe, chunks_to_dedupe, rag_logger)
        response_data = {
            'answer': ctx.best_answer,
            'confidence': ctx.best_confidence,
            'citations': unique_citations,
            'context_chunks': ctx.best_context_chunks,
            'iterations': ctx.evaluation_context,
            'total_iterations': len(ctx.evaluation_context),
            'follow_up_questions_asked': ctx.follow_up_questions_asked,
            'accumulated_chunks_count': len(ctx.accumulated_chunks),
            'answer_mode': ctx.answer_mode.value if ctx.answer_mode else 'standard'
        }
        if ctx.is_multi_ticker or ctx.is_general_question:
            response_data['individual_results'] = ctx.individual_results
        if ctx.question_analysis.get('limits_exceeded'):
            response_data['limits_exceeded'] = ctx.question_analysis['limits_exceeded']
        if ctx.conversation_id:
            self.question_analyzer.add_to_conversation_memory(ctx.conversation_id, ctx.question, "user")
            # Store full answer; ConversationMemory applies sliding-window truncation per message
            self.question_analyzer.add_to_conversation_memory(ctx.conversation_id, ctx.best_answer or "", "assistant")
        ctx.final_result = {
            'success': True,
            'response': response_data,
            'chunks': unique_chunks,
            'analysis': ctx.question_analysis,
            'timing': {'analysis': ctx.analysis_time, 'search': ctx.search_time, 'generation': ctx.generation_time, 'total': total_time}
        }

    # -------------------------------------------------------------------------
    # Main entry: execute_rag_flow runs stages in order and yields events (or returns final result)
    # -------------------------------------------------------------------------

    async def execute_rag_flow(self, question: str, show_details: bool = False, comprehensive: bool = True, stream_callback=None, max_iterations: int = None, conversation_id: str = None, stream: bool = True, **_extra):
        """
        ╔═══════════════════════════════════════════════════════════════════════╗
        ║  MAIN RAG FLOW EXECUTION - Earnings Transcript Q&A Pipeline          ║
        ╚═══════════════════════════════════════════════════════════════════════╝

        Execute the complete RAG flow with iterative improvement:

        Pipeline Stages:
        ────────────────
        1. Setup & Initialization
        2. Question Analysis (ticker extraction, intent detection)
        2.05. Search Planning (declarative plan: which data sources to use)
        2.1. Question Reasoning (planning approach)
        3. 🚀 PARALLEL Search Execution (news + 10-K + transcripts run concurrently)
        4. Context Preparation
        5. Initial Answer Generation
        6. Iterative Improvement
        7. Final Response Assembly

        Args:
            question: User's question to answer
            show_details: Print debug information (default: False)
            comprehensive: Use comprehensive mode for multi-ticker (default: True)
            stream_callback: Callback for streaming responses
            max_iterations: Max improvement iterations (default: 3)
            conversation_id: Unique conversation ID for memory tracking
            stream: Whether to yield progress events (default: True)

        Yields (if stream=True):
            Event dictionaries with type, message, step, and data

        Returns (if stream=False):
            Complete response object with answer, citations, and metadata
        """

        ctx = RAGFlowContext(
            question=question,
            stream=stream,
            show_details=show_details,
            comprehensive=comprehensive,
            stream_callback=stream_callback,
            max_iterations=max_iterations,
            conversation_id=conversation_id,
        )

        # Pipeline: setup → reasoning → search planning → Wave 1 parallel search
        #           → reflection → Wave 2 parallel sub-planners → prepare context → improvement → finalize
        async for event in self._stage_setup(ctx):
            yield event

        async for event in self._stage_combined_reasoning(ctx):
            yield event
        if ctx.early_return:
            if ctx.final_result:
                yield {
                    'type': 'result',
                    'message': 'Response generated successfully',
                    'step': 'complete',
                    'data': ctx.final_result,
                }
            return

        async for event in self._stage_search_planning(ctx):
            yield event

        # Wave 1: all sources in parallel
        async for event in self._stage_parallel_multi_source_search(ctx):
            yield event

        # Reflection: review Wave 1, decide if Wave 2 needed
        async for event in self._stage_reflection(ctx):
            yield event

        # Wave 2: spawn parallel sub-planners + searches per discovered company
        async for event in self._stage_wave2_search(ctx):
            yield event

        self._stage_prepare_context(ctx)

        async for event in self._stage_run_improvement(ctx):
            yield event
        if ctx.improvement_error:
            err = ctx.improvement_results or {}
            err['timing'] = {
                'analysis': ctx.analysis_time,
                'search': ctx.search_time,
                'total': time.time() - ctx.start_time
            }
            yield {'type': 'error', 'message': err.get('error', 'No results found'), 'step': 'generation', 'data': err}
            return

        self._stage_finalize(ctx)
        yield {
            'type': 'result',
            'message': 'Response generated successfully',
            'step': 'complete',
            'data': ctx.final_result
        }

    async def execute_rag_flow_async(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming wrapper - returns final result only."""
        kwargs['stream'] = False
        final_result = None
        async for event in self.execute_rag_flow(question, **kwargs):
            if event.get('type') == 'result':
                final_result = event.get('data')
        return final_result
    
