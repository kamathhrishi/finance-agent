#!/usr/bin/env python3
"""
SEC Filings Service for RAG System

This service provides access to 10-K SEC filings data for the main RAG agent.
It works as a specialized sub-agent that the main agent calls when annual report
or 10-K data is needed.

Similar to TavilyService for news, this provides a clean interface for the main
agent to access 10-K data without mixing concerns.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')

# SEC 10-K Section Definitions
SEC_10K_SECTIONS = {
    "item_1": {
        "title": "Business",
        "keywords": ["business", "operations", "products", "services", "company overview"]
    },
    "item_1a": {
        "title": "Risk Factors",
        "keywords": ["risk", "risks", "risk factors", "uncertainty", "challenges"]
    },
    "item_1b": {
        "title": "Unresolved Staff Comments",
        "keywords": ["staff comments", "unresolved", "sec comments"]
    },
    "item_2": {
        "title": "Properties",
        "keywords": ["properties", "facilities", "real estate", "locations"]
    },
    "item_3": {
        "title": "Legal Proceedings",
        "keywords": ["legal", "proceedings", "litigation", "lawsuits", "court"]
    },
    "item_5": {
        "title": "Market for Common Equity",
        "keywords": ["equity", "stock", "market", "shareholders", "common stock"]
    },
    "item_7": {
        "title": "Management's Discussion and Analysis (MD&A)",
        "keywords": ["md&a", "management discussion", "analysis", "performance", "results", "revenue", "sales"]
    },
    "item_7a": {
        "title": "Market Risk Disclosures",
        "keywords": ["market risk", "risk disclosure", "hedging", "derivatives"]
    },
    "item_8": {
        "title": "Financial Statements",
        "keywords": ["financial statements", "balance sheet", "income statement", "cash flow", "revenue", "expenses"]
    },
    "item_9a": {
        "title": "Controls and Procedures",
        "keywords": ["controls", "procedures", "internal controls", "compliance"]
    },
    "item_10": {
        "title": "Directors and Officers",
        "keywords": ["directors", "executives", "officers", "governance", "board"]
    },
    "item_11": {
        "title": "Executive Compensation",
        "keywords": ["compensation", "pay", "salaries", "benefits", "stock options", "executive"]
    },
    "item_12": {
        "title": "Security Ownership",
        "keywords": ["ownership", "beneficial owners", "shareholders", "insider"]
    },
    "item_15": {
        "title": "Exhibits and Schedules",
        "keywords": ["exhibits", "schedules", "attachments"]
    }
}


class SECFilingsService:
    """Service for accessing 10-K SEC filings data."""

    def __init__(self, database_manager, config):
        """
        Initialize SEC Filings Service with advanced search capabilities.

        Args:
            database_manager: DatabaseManager instance for database access
            config: Config instance with settings
        """
        self.database_manager = database_manager
        self.config = config

        # Initialize cross-encoder for reranking
        try:
            rag_logger.info("🔧 Loading cross-encoder model for reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.cross_encoder_available = True
            rag_logger.info("✅ Cross-encoder loaded successfully")
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load cross-encoder: {e}")
            rag_logger.warning("⚠️ Reranking will be disabled")
            self.cross_encoder = None
            self.cross_encoder_available = False

        # TF-IDF vectorizer will be initialized on-demand per query
        # (computing TF-IDF matrix for all chunks would be too expensive)

        # Initialize Cerebras client for section routing
        try:
            cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
            if cerebras_api_key:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_api_key)
                self.cerebras_available = True
                rag_logger.info("✅ Cerebras client initialized for section routing")
            else:
                self.cerebras_client = None
                self.cerebras_available = False
                rag_logger.warning("⚠️ CEREBRAS_API_KEY not found - section routing will be disabled")
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load Cerebras client: {e}")
            self.cerebras_client = None
            self.cerebras_available = False

        logger.info("✅ SEC Filings Service initialized with advanced search")

    def search_10k_filings(self, query_embedding: np.ndarray, ticker: str, fiscal_year: int = None, max_results: int = 15) -> List[Dict[str, Any]]:
        """
        Search 10-K filings for relevant information.

        Args:
            query_embedding: Query embedding vector
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year to filter by
            max_results: Maximum number of results to return

        Returns:
            List of relevant chunks from 10-K filings
        """
        rag_logger.info(f"📄 SEC Filings Service: Searching 10-K for {ticker}")

        try:
            # Call database manager's 10-K search function
            chunks = self.database_manager.search_10k_filings(
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year
            )

            # Limit results
            chunks = chunks[:max_results]

            rag_logger.info(f"✅ SEC Filings Service: Found {len(chunks)} chunks from 10-K")
            return chunks

        except Exception as e:
            rag_logger.error(f"❌ SEC Filings Service search failed: {e}")
            return []

    async def search_10k_filings_async(self, query_embedding: np.ndarray, ticker: str, fiscal_year: int = None, max_results: int = 15) -> List[Dict[str, Any]]:
        """
        Search 10-K filings for relevant information (async version).

        Args:
            query_embedding: Query embedding vector
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year to filter by
            max_results: Maximum number of results to return

        Returns:
            List of relevant chunks from 10-K filings
        """
        rag_logger.info(f"📄 SEC Filings Service: Async searching 10-K for {ticker}")

        try:
            # Call database manager's async 10-K search function
            chunks = await self.database_manager.search_10k_filings_async(
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year
            )

            # Limit results
            chunks = chunks[:max_results]

            rag_logger.info(f"✅ SEC Filings Service: Found {len(chunks)} chunks from 10-K")
            return chunks

        except Exception as e:
            rag_logger.error(f"❌ SEC Filings Service async search failed: {e}")
            return []

    async def search_10k_filings_advanced_async(
        self,
        query: str,
        query_embedding: np.ndarray,
        ticker: str,
        fiscal_year: int = None,
        max_results: int = 15,
        initial_k: int = 100,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        use_reranking: bool = True,
        boost_tables: bool = True,
        use_section_routing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with section routing, hybrid search, cross-encoder reranking, and table boosting.

        This implements the sophisticated search from experiments/sec_filings_rag_scratch:
        - Phase 0: Section routing (LLM determines relevant SEC sections)
        - Stage 1: Hybrid search (TF-IDF keyword + semantic)
        - Stage 2: Cross-encoder reranking
        - Stage 3: Table boosting (prioritize financial tables)

        Args:
            query: Search query string (for TF-IDF and cross-encoder)
            query_embedding: Query embedding vector (for semantic search)
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year to filter by
            max_results: Number of final results to return
            initial_k: Number of candidates to retrieve before reranking (default 100)
            keyword_weight: Weight for TF-IDF keyword search (0-1, default 0.3)
            semantic_weight: Weight for semantic search (0-1, default 0.7)
            use_reranking: Whether to use cross-encoder reranking (default True)
            boost_tables: Whether to boost table chunks (default True)
            use_section_routing: Whether to use LLM section routing (default True)

        Returns:
            List of relevant chunks from 10-K filings, ranked by relevance
        """
        rag_logger.info(f"🔍 Advanced 10-K Search for {ticker} (section routing + hybrid + reranking)")

        try:
            # PHASE 0: Section Routing (optional)
            target_sections = []
            routing_result = None
            if use_section_routing:
                routing_result = await self.route_query_to_sections(query)
                target_sections = routing_result.get('target_sections', [])

                if target_sections:
                    rag_logger.info(f"🎯 Targeting sections: {target_sections}")
            else:
                rag_logger.info(f"⏭️ Section routing disabled - searching all sections")
            # Fetch initial candidates from database (semantic search only)
            # We'll do hybrid scoring and reranking in-memory
            chunks = await self.database_manager.search_10k_filings_async(
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year
            )

            # Get more candidates for hybrid search
            chunks = chunks[:initial_k]

            if not chunks:
                rag_logger.warning(f"⚠️ No chunks found for {ticker}")
                return []

            rag_logger.info(f"📊 Retrieved {len(chunks)} initial candidates")

            # Filter by target sections if routing was used
            if target_sections and len(target_sections) > 0:
                filtered_chunks = [
                    chunk for chunk in chunks
                    if chunk.get('sec_section') in target_sections
                ]

                if filtered_chunks:
                    rag_logger.info(f"📊 Filtered: {len(chunks)} → {len(filtered_chunks)} chunks (by SEC sections)")
                    chunks = filtered_chunks
                else:
                    rag_logger.warning(f"⚠️ No chunks in target sections - using all {len(chunks)} chunks")

            rag_logger.info(f"📊 Processing {len(chunks)} candidates for hybrid search")

            # STAGE 1: Hybrid Search (TF-IDF + Semantic)
            rag_logger.info(f"⚖️ Stage 1: Hybrid scoring (keyword: {keyword_weight:.1f}, semantic: {semantic_weight:.1f})")

            # Extract chunk texts for TF-IDF
            chunk_texts = [chunk['chunk_text'] for chunk in chunks]

            # Compute TF-IDF scores on-the-fly
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
            query_tfidf = tfidf_vectorizer.transform([query])
            keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

            # Get semantic scores from database results
            semantic_scores = np.array([chunk.get('similarity', 0.0) for chunk in chunks])

            # Normalize scores to 0-1 range
            semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)
            keyword_scores_norm = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-10)

            # Combine scores with weights
            hybrid_scores = (semantic_weight * semantic_scores_norm) + (keyword_weight * keyword_scores_norm)

            # Add hybrid scores to chunks
            for i, chunk in enumerate(chunks):
                chunk['hybrid_score'] = float(hybrid_scores[i])
                chunk['keyword_score'] = float(keyword_scores[i])
                chunk['semantic_score_orig'] = float(semantic_scores[i])

            rag_logger.info(f"✅ Stage 1: Computed hybrid scores")

            # STAGE 2: Cross-Encoder Reranking (optional)
            if use_reranking and self.cross_encoder_available:
                rag_logger.info(f"🎯 Stage 2: Reranking with cross-encoder...")

                # Prepare query-document pairs
                pairs = [[query, chunk['chunk_text']] for chunk in chunks]

                # Get cross-encoder scores
                cross_scores = self.cross_encoder.predict(pairs)

                # Add cross-encoder scores to chunks
                for i, chunk in enumerate(chunks):
                    chunk['cross_encoder_score'] = float(cross_scores[i])
                    chunk['similarity'] = float(cross_scores[i])  # Use cross-encoder score as final similarity

                # Sort by cross-encoder scores
                chunks.sort(key=lambda x: x['cross_encoder_score'], reverse=True)

                rag_logger.info(f"✅ Stage 2: Reranked with cross-encoder")
            else:
                # No reranking - sort by hybrid score
                chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
                rag_logger.info(f"⏭️ Stage 2: Skipped reranking (using hybrid scores)")

            # STAGE 3: LLM-Based Table Selection (NEW! Replaces table boosting)
            # Instead of searching for tables with embeddings, let LLM intelligently select them
            if boost_tables:
                rag_logger.info(f"📊 Stage 3: LLM-based table selection (prioritizing financial statements)...")

                # Get all available tables for this ticker
                all_tables = await self.get_all_tables_for_ticker(ticker, fiscal_year)

                if all_tables:
                    # Use LLM to select relevant tables
                    selected_tables, selection_reasoning = await self.select_tables_by_llm(
                        question=query,
                        tables=all_tables,
                        max_tables=5  # Max 5 tables
                    )

                    if selected_tables:
                        rag_logger.info(f"✅ LLM selected {len(selected_tables)} tables")
                        rag_logger.info(f"💭 Selection reasoning: {selection_reasoning}")

                        # Convert selected tables to chunk format
                        table_chunks = []
                        for table in selected_tables:
                            table_chunk = {
                                'chunk_text': table.get('content', ''),
                                'ticker': table.get('ticker'),
                                'fiscal_year': table.get('fiscal_year'),
                                'chunk_type': 'table',
                                'sec_section': table.get('sec_section'),
                                'sec_section_title': table.get('sec_section_title'),
                                'path_string': table.get('path_string'),
                                'is_financial_statement': table.get('is_financial_statement', False),
                                'statement_type': table.get('statement_type', ''),
                                'similarity': 1.0,  # All selected tables have equal priority
                                'selection_method': 'llm_selection',
                                'selection_reasoning': selection_reasoning
                            }
                            table_chunks.append(table_chunk)

                        # Filter text-only chunks from current results
                        text_chunks = [c for c in chunks if c.get('chunk_type') != 'table']

                        # Combine: Selected tables first, then text chunks
                        chunks = table_chunks + text_chunks
                        rag_logger.info(f"✅ Combined: {len(table_chunks)} selected tables + {len(text_chunks)} text chunks")
                else:
                    rag_logger.warning(f"⚠️ No tables available in database for {ticker}")

            # OLD TABLE BOOSTING (now replaced by LLM selection above)
            elif False:  # Disabled - using LLM selection instead
                if boost_tables:
                    rag_logger.info(f"📊 Stage 3: Applying table boosting...")

                    # Identify table chunks
                    table_chunks = [chunk for chunk in chunks if chunk.get('chunk_type') == 'table']
                    text_chunks = [chunk for chunk in chunks if chunk.get('chunk_type') != 'table']

                    # Boost financial statement tables to top
                    financial_tables = [
                        chunk for chunk in table_chunks
                        if self._is_financial_statement_table(chunk)
                    ]
                    other_tables = [
                        chunk for chunk in table_chunks
                        if not self._is_financial_statement_table(chunk)
                    ]

                    # Reorder: financial tables → other tables → text chunks
                    # But only if tables are reasonably relevant (e.g., cross_encoder_score > 0.3)
                    boosted_chunks = []

                    # Add top financial tables first (if relevant)
                    for chunk in financial_tables[:3]:  # Max 3 financial tables
                        if chunk.get('cross_encoder_score', chunk.get('hybrid_score', 0)) > 0.3:
                            boosted_chunks.append(chunk)

                    # Then add regular results (mixed tables and text)
                    remaining_chunks = other_tables + text_chunks
                    remaining_chunks.sort(
                        key=lambda x: x.get('cross_encoder_score', x.get('hybrid_score', 0)),
                        reverse=True
                    )
                    boosted_chunks.extend(remaining_chunks)

                    chunks = boosted_chunks

                    rag_logger.info(f"✅ Stage 3: Applied table boosting ({len(financial_tables)} financial tables)")

            # Limit to max_results
            chunks = chunks[:max_results]

            rag_logger.info(f"🎯 Advanced search complete: {len(chunks)} results")
            return chunks

        except Exception as e:
            rag_logger.error(f"❌ Advanced 10-K search failed: {e}")
            import traceback
            rag_logger.error(traceback.format_exc())
            return []

    def _is_financial_statement_table(self, chunk: Dict[str, Any]) -> bool:
        """Check if a chunk is a core financial statement table."""
        # Check metadata first
        if chunk.get('is_financial_statement'):
            return True

        # Check section - Item 8 typically contains financial statements
        sec_section = chunk.get('sec_section', '')
        if sec_section == 'item_8':
            # Look for financial statement keywords in path or content
            path = chunk.get('path_string', '').lower()
            content = chunk.get('chunk_text', '').lower()

            financial_keywords = [
                'balance sheet', 'income statement', 'cash flow',
                'statement of operations', 'statement of financial position',
                'statement of cash flows', 'consolidated balance',
                'consolidated income', 'consolidated cash'
            ]

            for keyword in financial_keywords:
                if keyword in path or keyword in content[:500]:
                    return True

        return False

    async def get_all_tables_for_ticker(self, ticker: str, fiscal_year: int = None) -> List[Dict[str, Any]]:
        """
        Get all available tables for a ticker from the database.

        Args:
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year filter

        Returns:
            List of table metadata (paths, sections, etc.)
        """
        try:
            # Query database for all tables for this ticker
            # This will be used for LLM-based table selection
            tables = await self.database_manager.get_all_tables_for_ticker_async(
                ticker=ticker,
                fiscal_year=fiscal_year
            )

            rag_logger.info(f"📊 Found {len(tables)} tables for {ticker}")
            return tables

        except Exception as e:
            rag_logger.error(f"❌ Failed to get tables for {ticker}: {e}")
            return []

    async def select_tables_by_llm(
        self,
        question: str,
        tables: List[Dict[str, Any]],
        max_tables: int = 10
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Use LLM to intelligently select relevant tables based on question.

        Args:
            question: User's question
            tables: List of available tables with metadata
            max_tables: Maximum number of tables to select

        Returns:
            Tuple of (selected_tables, reasoning)
        """
        rag_logger.info(f"🤖 Using LLM to select tables from {len(tables)} available")

        if not self.cerebras_available:
            rag_logger.warning(f"⚠️ Cerebras not available - cannot use LLM selection")
            return [], "Cerebras not available"

        if not tables:
            return [], "No tables available"

        # Build list of tables for LLM with prioritization
        table_list = []
        financial_statement_tables = []
        other_tables = []

        for idx, table in enumerate(tables, 1):
            path = table.get('path_string', 'Unknown')
            sec_section = table.get('sec_section_title', 'Unknown Section')
            is_financial = table.get('is_financial_statement', False)
            statement_type = table.get('statement_type', '')

            # Prioritize the three core financial statements
            if is_financial and statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
                priority = "🌟 CORE FINANCIAL STATEMENT"
                financial_statement_tables.append((idx, path, sec_section, statement_type, priority))
            else:
                other_tables.append((idx, path, sec_section))

        # Build prioritized list: financial statements first
        for idx, path, sec_section, stmt_type, priority in financial_statement_tables:
            table_list.append(f"{idx}. [{priority}] {path} ({sec_section}) - {stmt_type.replace('_', ' ').title()}")

        for idx, path, sec_section in other_tables:
            table_list.append(f"{idx}. {path} ({sec_section})")

        tables_text = "\n".join(table_list[:50])  # Limit to 50 for token limits

        prompt = f"""You are an expert financial analyst carefully selecting which tables to examine from a 10-K SEC filing.
Your goal is to select ONLY the MOST IMPORTANT and RELEVANT tables - be selective and precise.
Quality over quantity: selecting 2-3 highly relevant tables is far better than selecting 10 loosely related ones.

IMPORTANT: BE PERSISTENT! If the question asks for a specific line item or metric, don't give up if it's not immediately obvious.
Explore all available tables systematically. The information might be in:
- A table with a different name than expected
- A note or supplementary table
- A table that contains the data in a different format
- Multiple related tables that together provide the answer

QUESTION: {question}

AVAILABLE TABLES:
{tables_text}

CRITICAL THINKING PROCESS - Follow these steps methodically:

STEP 1: DEEP QUESTION ANALYSIS
- Break down the question: What EXACTLY is being asked?
- Identify key terms, concepts, and financial metrics mentioned
- Determine the answer type needed: specific number, ratio, trend, comparison, calculation?
- Note any timeframes, periods, or specific contexts (e.g., "FY2024", "Q4", "year-over-year")
- Identify the question category:
  * Revenue/income → Look for Income Statement or Statement of Operations (marked with 🌟)
  * Cash flow → Look for Cash Flow Statement (marked with 🌟)
  * Balance sheet items (assets, liabilities, equity) → Look for Balance Sheet (marked with 🌟)
  * Ratios/metrics → Often requires multiple financial statements
  * Segment/geographic → Segment reporting tables
  * Specific notes → Look for exact note tables (e.g., "NOTE 13. EARNINGS PER SHARE")
  * Stock/equity → Stockholders' equity tables or Balance Sheet

STEP 2: SYSTEMATIC TABLE EVALUATION
- Read through EVERY table name in the list
- For EACH table, ask yourself:
  * Does this table DIRECTLY contain the information needed?
  * Is this table ESSENTIAL to answer the question, or just tangentially related?
  * Would selecting this table add unique value, or is it redundant?
- Create a mental relevance score (1-10) for each table:
  * 9-10: Directly answers the question, essential
  * 7-8: Highly relevant, likely needed
  * 5-6: Moderately relevant, might be useful
  * 3-4: Loosely related, probably not needed
  * 1-2: Barely related, should skip

STEP 3: MAKE SELECTION
- **PRIORITIZE CORE FINANCIAL STATEMENTS** marked with 🌟:
  * Income Statement: For revenue, profit, expenses, margins
  * Balance Sheet: For assets, liabilities, equity, working capital
  * Cash Flow Statement: For cash flows, capex, free cash flow
- Maximum of {max_tables} tables, but prefer fewer highly relevant ones
- If the question asks for financial metrics, ALWAYS include the relevant core financial statement(s)
- Consider table dependencies: Some questions require multiple related tables (e.g., ratio calculations need both numerator and denominator)

EXAMPLES:
- "What was total revenue?" → Select income statement (🌟)
- "What are total assets?" → Select balance sheet (🌟)
- "What was free cash flow?" → Select cash flow statement (🌟)
- "Working capital ratio" → Select balance sheet (🌟) for current assets/liabilities calculation
- "Revenue by segment" → Select both income statement (🌟) and segment reporting tables

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "selected_table_indices": [1, 2, 5],
    "step_by_step_reasoning": "Brief analysis: (1) Question asks for X, (2) Tables Y and Z directly contain this data, (3) Selected because...",
    "reasoning": "Concise explanation of why these specific tables answer the question"
}}"""

        try:
            messages = [
                {"role": "system", "content": "You are a financial analyst. Return only JSON, no markdown."},
                {"role": "user", "content": prompt}
            ]

            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "llama3.1-70b"),
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()

            # Remove markdown if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)
            selected_indices = result.get('selected_table_indices', [])
            reasoning = result.get('reasoning', '')
            step_by_step = result.get('step_by_step_reasoning', '')

            # Convert indices to actual tables
            selected_tables = []
            for idx in selected_indices:
                if 1 <= idx <= len(tables):
                    selected_tables.append(tables[idx - 1])

            # Log detailed reasoning if available
            if step_by_step:
                rag_logger.info(f"   🧠 Step-by-step reasoning: {step_by_step[:200]}...")
            rag_logger.info(f"   ✅ Selected {len(selected_tables)} tables")
            rag_logger.info(f"   💭 Final reasoning: {reasoning}")

            return selected_tables, reasoning

        except Exception as e:
            rag_logger.error(f"❌ Table selection failed: {e}")
            # Fallback: return financial statement tables if available
            financial_tables = [t for t in tables if t.get('is_financial_statement')]
            return financial_tables[:3], f"Error in LLM selection, using financial statements as fallback"

    async def route_query_to_sections(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze query and route to relevant SEC sections.

        This is PHASE 0 of advanced search - determines which SEC sections
        (Item 1, Item 7, Item 8, etc.) are most relevant to the question.

        Args:
            query: User's question

        Returns:
            Dict with:
            - target_sections: List of section keys (e.g., ['item_7', 'item_8'])
            - confidence: Float 0-1
            - reasoning: String explaining the routing decision
        """
        rag_logger.info(f"🔍 PHASE 0: Routing query to SEC sections using LLM...")

        if not self.cerebras_available:
            rag_logger.warning(f"⚠️ Cerebras client not available - skipping section routing")
            return {
                'target_sections': [],
                'confidence': 0.0,
                'reasoning': 'Cerebras client not available'
            }

        # Build section list for prompt
        section_descriptions = []
        for section_key, section_data in SEC_10K_SECTIONS.items():
            section_descriptions.append(
                f"- {section_key}: {section_data['title']} (keywords: {', '.join(section_data['keywords'][:3])})"
            )
        sections_text = "\n".join(section_descriptions)

        prompt = f"""You are an expert at analyzing SEC 10-K filings. Your task is to determine which sections of a 10-K filing are most relevant to answer a user's question.

AVAILABLE SECTIONS:
{sections_text}

USER QUERY: {query}

UNDERSTANDING 10-K STRUCTURE:
SEC 10-K filings are structured documents where information is often distributed across multiple sections:

QUALITATIVE INFORMATION (Narrative, descriptive, strategic):
- Item 1 (Business): Company operations, products/services, customers, revenue sources, business segments, competitive landscape
- Item 1A (Risk Factors): Risks, uncertainties, forward-looking challenges
- Item 7 (MD&A): Management's narrative analysis of financial results, trends, explanations of changes
- Item 7A (Market Risk): Risk management strategies, hedging, market sensitivities

QUANTITATIVE INFORMATION (Numbers, financial data, metrics):
- Item 8 (Financial Statements): Authoritative audited financial statements (income statement, balance sheet, cash flow, footnotes)
- Item 5 (Market Data): 5-year selected financial data, stock performance, quarterly trends
- Item 7 (MD&A): Also contains financial analysis and tables explaining performance
- Item 1 (Business): May contain segment revenue breakdowns, customer concentration percentages, geographic data

KEY PRINCIPLES FOR ROUTING:

1. QUANTITATIVE QUERIES (numbers, ratios, financial metrics):
   - ALWAYS include item_8 (Financial Statements) - the authoritative source
   - ALSO consider item_5 (Market Data) for trend analysis or 5-year comparisons
   - ALSO consider item_7 (MD&A) for management's quantitative analysis
   - ALSO consider item_1 (Business) for segment/customer/geographic breakdowns

2. QUALITATIVE QUERIES (strategy, risks, operations, business model):
   - item_1 (Business) for operational and strategic information
   - item_1a (Risk Factors) for risks and uncertainties
   - item_7 (MD&A) for management's perspective and trend explanations

3. WHEN IN DOUBT, BE INCLUSIVE:
   - It's better to search 3-4 sections than to miss the answer by being too narrow
   - Financial information especially can appear in item_1, item_5, item_7, AND item_8
   - Return multiple sections to ensure comprehensive coverage

EXAMPLES:
- "What were the main risks?" → ["item_1a"] (Qualitative - Risk Factors section)
- "What was the total revenue?" → ["item_1", "item_7", "item_8"] (Quantitative - Business breakdown + MD&A + Financial Statements)
- "Customer concentration" → ["item_1", "item_7", "item_8"] (Can be in Business description, MD&A discussion, or financial footnotes)
- "Working capital ratio" → ["item_5", "item_7", "item_8"] (Calculated from balance sheet data in multiple sections)
- "Who are the executives?" → ["item_10", "item_11"] (Qualitative - Directors + Compensation)
- "What products do they sell?" → ["item_1"] (Qualitative - Business operations)

Return ONLY valid JSON in this exact format:
{{
    "target_sections": ["item_1", "item_7", "item_8"],
    "confidence": 0.9,
    "reasoning": "Financial query requiring both business context and authoritative financial data from multiple sections"
}}"""

        try:
            messages = [
                {"role": "system", "content": "You are a SEC filing expert. Return only JSON, no markdown."},
                {"role": "user", "content": prompt}
            ]

            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "llama3.1-70b"),
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )

            response_text = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)

            rag_logger.info(f"   ✅ Routed to sections: {result.get('target_sections', [])}")
            rag_logger.info(f"   🎯 Confidence: {result.get('confidence', 0.0):.2f}")
            rag_logger.info(f"   💭 Reasoning: {result.get('reasoning', 'N/A')}")

            return result

        except Exception as e:
            rag_logger.error(f"❌ Error in section routing: {e}")
            rag_logger.info(f"   ⚠️ Falling back to no section filtering")
            return {
                'target_sections': [],
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}'
            }

    def format_10k_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format 10-K chunks into context string for the LLM.

        Args:
            chunks: List of 10-K chunks from search

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []
        context_parts.append("=" * 80)
        context_parts.append("10-K SEC FILINGS DATA")
        context_parts.append("=" * 80)
        context_parts.append("")

        for idx, chunk in enumerate(chunks, 1):
            ticker = chunk.get('ticker', 'UNKNOWN')
            fiscal_year = chunk.get('fiscal_year', 'UNKNOWN')
            sec_section = chunk.get('sec_section_title', 'Unknown Section')
            chunk_type = chunk.get('chunk_type', 'text')

            # Citation format: [10K1], [10K2], etc.
            citation = f"[10K{idx}]"

            context_parts.append(f"{citation} {ticker} - FY{fiscal_year} - {sec_section}")
            if chunk_type == 'table':
                context_parts.append(f"Type: Financial Table")
            context_parts.append(f"Content: {chunk['chunk_text']}")
            context_parts.append("")

        context_parts.append("=" * 80)
        return "\n".join(context_parts)

    def get_10k_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get citation information for 10-K chunks.

        Args:
            chunks: List of 10-K chunks

        Returns:
            List of citation dictionaries for frontend display
        """
        citations = []

        for idx, chunk in enumerate(chunks, 1):
            citation = {
                "type": "10-K",
                "marker": f"[10K{idx}]",
                "ticker": chunk.get('ticker', 'UNKNOWN'),
                "fiscal_year": chunk.get('fiscal_year'),
                "section": chunk.get('sec_section_title', 'Unknown Section'),
                "section_id": chunk.get('sec_section', 'unknown'),
                "chunk_type": chunk.get('chunk_type', 'text'),
                "path": chunk.get('path_string', ''),
                "similarity": chunk.get('similarity', 0)
            }
            citations.append(citation)

        return citations

