#!/usr/bin/env python3
"""
Iterative SEC Filings Service for RAG System

This service implements an iterative approach to 10-K SEC filing search where:
- At each iteration, the agent decides whether to retrieve TABLE or TEXT chunks
- Evaluates answer quality after each iteration
- Dynamically switches between sources based on what's missing
- Builds comprehensive answers step by step

This is based on experiments/sec_filings_rag_scratch/agent.py EnhancedSECRAGAgent.

To switch back to the one-pass approach, use sec_filings_service.py instead.
"""

import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
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

# Financial keywords for table-first detection
FINANCIAL_KEYWORDS = [
    'revenue', 'income', 'profit', 'loss', 'earnings', 'sales', 'expenses',
    'assets', 'liabilities', 'equity', 'cash flow', 'ratio', 'margin', 'growth rate',
    'million', 'billion', 'percent', '%', 'dollar', 'amount', 'total', 'figure', 'eps',
    'balance sheet', 'income statement', 'cash flow statement', 'financial'
]


class IterativeSECFilingsService:
    """
    Iterative SEC Filings Service with step-by-step table/text decision making.

    At each iteration:
    1. Decides whether to retrieve TABLE or TEXT chunks (LLM decision)
    2. Retrieves the chosen chunk type
    3. Generates/refines the answer
    4. Evaluates answer quality
    5. Decides next retrieval strategy based on evaluation

    Stops when:
    - Quality score >= 0.9 (90% confidence)
    - Max iterations reached (default 5)
    - Agent decides answer is sufficient
    """

    def __init__(self, database_manager, config):
        """
        Initialize Iterative SEC Filings Service.

        Args:
            database_manager: DatabaseManager instance for database access
            config: Config instance with settings
        """
        self.database_manager = database_manager
        self.config = config
        self.max_iterations = 5  # Hard limit on iterations

        # Initialize cross-encoder for reranking
        try:
            rag_logger.info("🔧 Loading cross-encoder model for reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.cross_encoder_available = True
            rag_logger.info("✅ Cross-encoder loaded successfully")
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load cross-encoder: {e}")
            self.cross_encoder = None
            self.cross_encoder_available = False

        # Initialize Cerebras client for LLM decisions
        try:
            cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
            if cerebras_api_key:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_api_key)
                self.cerebras_available = True
                rag_logger.info("✅ Cerebras client initialized for iterative decisions")
            else:
                self.cerebras_client = None
                self.cerebras_available = False
                rag_logger.warning("⚠️ CEREBRAS_API_KEY not found - LLM decisions will use fallback")
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load Cerebras client: {e}")
            self.cerebras_client = None
            self.cerebras_available = False

        # Session state for tracking iterations
        self.current_session = None

        logger.info("✅ Iterative SEC Filings Service initialized")

    def _reset_session(self, question: str):
        """Reset session state for a new question."""
        self.current_session = {
            'question': question,
            'iteration_history': [],
            'accumulated_chunks': [],
            'accumulated_table_chunks': [],
            'accumulated_text_chunks': [],
            'table_retrieved_count': 0,
            'text_retrieved_count': 0,
            'current_answer': None,
            'evaluation_history': [],
            'retrieval_history': [],
            'session_start': datetime.now().isoformat()
        }

    async def execute_iterative_search_async(
        self,
        query: str,
        query_embedding: np.ndarray,
        ticker: str,
        fiscal_year: int = None,
        max_iterations: int = 5,
        confidence_threshold: float = 0.9,
        event_yielder=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute iterative 10-K search with step-by-step table/text decisions.

        This is the main entry point for the iterative SEC agent.

        At each iteration:
        1. LLM decides: retrieve TABLE or TEXT?
        2. Retrieves the chosen chunk type
        3. Generates/refines the answer
        4. Evaluates answer quality
        5. Decides next strategy based on what's missing

        Args:
            query: User's question
            query_embedding: Query embedding vector
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year filter
            max_iterations: Maximum iterations (default 5, hard capped at 5)
            confidence_threshold: Quality score to stop early (default 0.9)
            event_yielder: Optional callback for streaming events

        Yields:
            Events with iteration progress and final results
        """
        # Cap iterations at 5
        max_iterations = min(max_iterations, self.max_iterations)

        rag_logger.info(f"🔄 Starting ITERATIVE 10-K search for {ticker}")
        rag_logger.info(f"   Max iterations: {max_iterations}, Confidence threshold: {confidence_threshold}")

        # Reset session state
        self._reset_session(query)

        # PHASE 0: Decide initial retrieval type
        initial_type = await self._decide_initial_retrieval_type(query)
        current_retrieval_type = initial_type['retrieval_type']

        if event_yielder:
            yield {
                'type': 'iteration_init',
                'message': f"Starting iterative search - initial type: {current_retrieval_type.upper()}",
                'data': {
                    'initial_type': current_retrieval_type,
                    'reasoning': initial_type['reasoning'],
                    'max_iterations': max_iterations
                }
            }

        rag_logger.info(f"🎯 Initial retrieval type: {current_retrieval_type.upper()}")
        rag_logger.info(f"   Reasoning: {initial_type['reasoning']}")

        # Get all available tables upfront for LLM selection
        all_tables = await self._get_all_tables_for_ticker(ticker, fiscal_year)
        rag_logger.info(f"📊 {len(all_tables)} tables available for selection")

        # ITERATIVE LOOP
        for iteration in range(max_iterations):
            iteration_num = iteration + 1
            rag_logger.info(f"\n{'='*60}")
            rag_logger.info(f"🔄 ITERATION {iteration_num}/{max_iterations}")
            rag_logger.info(f"{'='*60}")

            if event_yielder:
                yield {
                    'type': 'iteration_start',
                    'message': f"Iteration {iteration_num}: Retrieving {current_retrieval_type.upper()} chunks",
                    'data': {
                        'iteration': iteration_num,
                        'total_iterations': max_iterations,
                        'retrieval_type': current_retrieval_type,
                        'accumulated_chunks': len(self.current_session['accumulated_chunks'])
                    }
                }

            # STEP 1: Decide retrieval type (for iterations > 1)
            if iteration > 0:
                retrieval_decision = await self._decide_retrieval_type(
                    question=query,
                    current_answer=self.current_session['current_answer'],
                    iteration=iteration_num,
                    max_iterations=max_iterations
                )
                current_retrieval_type = retrieval_decision['retrieval_type']
                rag_logger.info(f"🎯 Retrieval decision: {current_retrieval_type.upper()}")
                rag_logger.info(f"   Reasoning: {retrieval_decision['reasoning']}")

            # STEP 2: Retrieve chunks based on decision
            if current_retrieval_type == 'table':
                retrieved_chunks = await self._retrieve_table_chunks(
                    question=query,
                    tables=all_tables,
                    iteration=iteration_num
                )
                self.current_session['table_retrieved_count'] += 1
                self.current_session['accumulated_table_chunks'].extend(retrieved_chunks)
            else:
                retrieved_chunks = await self._retrieve_text_chunks(
                    query=query,
                    query_embedding=query_embedding,
                    ticker=ticker,
                    fiscal_year=fiscal_year
                )
                self.current_session['text_retrieved_count'] += 1
                self.current_session['accumulated_text_chunks'].extend(retrieved_chunks)

            # Add to accumulated chunks (avoid duplicates)
            seen_ids = {c.get('id', c.get('chunk_text', '')[:50]) for c in self.current_session['accumulated_chunks']}
            for chunk in retrieved_chunks:
                chunk_id = chunk.get('id', chunk.get('chunk_text', '')[:50])
                if chunk_id not in seen_ids:
                    self.current_session['accumulated_chunks'].append(chunk)
                    seen_ids.add(chunk_id)

            rag_logger.info(f"✅ Retrieved {len(retrieved_chunks)} {current_retrieval_type} chunks")
            rag_logger.info(f"📊 Total accumulated: {len(self.current_session['accumulated_chunks'])} chunks")

            if event_yielder:
                yield {
                    'type': 'iteration_retrieve',
                    'message': f"Retrieved {len(retrieved_chunks)} {current_retrieval_type} chunks",
                    'data': {
                        'iteration': iteration_num,
                        'retrieval_type': current_retrieval_type,
                        'chunks_retrieved': len(retrieved_chunks),
                        'total_accumulated': len(self.current_session['accumulated_chunks']),
                        'table_count': self.current_session['table_retrieved_count'],
                        'text_count': self.current_session['text_retrieved_count']
                    }
                }

            # STEP 3: Generate/Refine answer
            refined_answer = await self._generate_iterative_answer(
                question=query,
                new_chunks=retrieved_chunks,
                accumulated_chunks=self.current_session['accumulated_chunks'],
                previous_answer=self.current_session['current_answer'],
                iteration=iteration_num,
                retrieval_type=current_retrieval_type
            )
            self.current_session['current_answer'] = refined_answer

            rag_logger.info(f"✅ Answer generated ({len(refined_answer)} chars)")

            # STEP 4: Evaluate answer quality
            evaluation = await self._evaluate_answer_quality(
                question=query,
                answer=refined_answer,
                chunks=retrieved_chunks,
                iteration=iteration_num
            )

            quality_score = evaluation.get('quality_score', 0.0)
            self.current_session['evaluation_history'].append({
                'iteration': iteration_num,
                'evaluation': evaluation
            })

            rag_logger.info(f"📊 Quality Score: {quality_score:.2f}/1.0")
            if evaluation.get('issues'):
                rag_logger.info(f"   Issues: {evaluation['issues'][:2]}")

            if event_yielder:
                yield {
                    'type': 'iteration_evaluate',
                    'message': f"Quality score: {quality_score:.2f}",
                    'data': {
                        'iteration': iteration_num,
                        'quality_score': quality_score,
                        'issues': evaluation.get('issues', []),
                        'missing_info': evaluation.get('missing_info', [])
                    }
                }

            # Store iteration record
            self.current_session['iteration_history'].append({
                'iteration': iteration_num,
                'retrieval_type': current_retrieval_type,
                'chunks_retrieved': len(retrieved_chunks),
                'quality_score': quality_score,
                'timestamp': datetime.now().isoformat()
            })

            self.current_session['retrieval_history'].append({
                'iteration': iteration_num,
                'type': current_retrieval_type,
                'num_chunks': len(retrieved_chunks)
            })

            # STEP 5: Check for early termination
            if quality_score >= confidence_threshold:
                rag_logger.info(f"🎉 EARLY TERMINATION: Quality {quality_score:.2f} >= {confidence_threshold}")
                if event_yielder:
                    yield {
                        'type': 'iteration_complete',
                        'message': f"High confidence achieved ({quality_score:.2f}) - stopping early",
                        'data': {
                            'iteration': iteration_num,
                            'reason': 'confidence_threshold',
                            'quality_score': quality_score
                        }
                    }
                break

            # STEP 6: Decide next retrieval strategy (if not last iteration)
            if iteration < max_iterations - 1:
                next_decision = await self._decide_next_retrieval_strategy(
                    question=query,
                    current_answer=refined_answer,
                    evaluation=evaluation,
                    iteration=iteration_num,
                    max_iterations=max_iterations
                )
                current_retrieval_type = next_decision['next_type']
                rag_logger.info(f"🎯 Next iteration will retrieve: {current_retrieval_type.upper()}")

        # FINAL: Return results
        final_chunks = self.current_session['accumulated_chunks']

        rag_logger.info(f"\n{'='*60}")
        rag_logger.info(f"✅ ITERATIVE SEARCH COMPLETE")
        rag_logger.info(f"   Iterations: {len(self.current_session['iteration_history'])}")
        rag_logger.info(f"   Total chunks: {len(final_chunks)}")
        rag_logger.info(f"   Table retrievals: {self.current_session['table_retrieved_count']}")
        rag_logger.info(f"   Text retrievals: {self.current_session['text_retrieved_count']}")
        rag_logger.info(f"{'='*60}")

        if event_yielder:
            yield {
                'type': 'iteration_final',
                'message': 'Iterative search complete',
                'data': {
                    'total_iterations': len(self.current_session['iteration_history']),
                    'total_chunks': len(final_chunks),
                    'table_retrievals': self.current_session['table_retrieved_count'],
                    'text_retrievals': self.current_session['text_retrieved_count'],
                    'final_quality': self.current_session['evaluation_history'][-1]['evaluation']['quality_score'] if self.current_session['evaluation_history'] else 0
                }
            }

        # Yield final result
        yield {
            'type': 'result',
            'chunks': final_chunks,
            'session': self.current_session
        }

    async def _decide_initial_retrieval_type(self, question: str) -> Dict[str, Any]:
        """
        Decide whether to start with TABLE or TEXT retrieval.

        Uses LLM if available, otherwise falls back to keyword detection.
        """
        # Check for financial keywords - prefer tables for numeric queries
        has_financial_keywords = any(kw in question.lower() for kw in FINANCIAL_KEYWORDS)

        if not self.cerebras_available:
            # Fallback: use keyword detection
            initial_type = 'table' if has_financial_keywords else 'text'
            return {
                'retrieval_type': initial_type,
                'reasoning': f'Keyword analysis: {"financial keywords detected" if has_financial_keywords else "no financial keywords"}'
            }

        prompt = f"""You are deciding whether to search TABLES or TEXT first to answer a financial question.

QUESTION: {question}

DECISION CRITERIA:
- TABLES: For numeric data, financial metrics, specific figures, ratios, balance sheets, income statements
- TEXT: For qualitative information, explanations, risk factors, business descriptions, management commentary

Return ONLY valid JSON:
{{"retrieval_type": "table" or "text", "reasoning": "Brief explanation"}}"""

        try:
            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"),
                messages=[
                    {"role": "system", "content": "Return only JSON, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)

            # Enforce table-first rule for numeric queries
            if has_financial_keywords and result['retrieval_type'] == 'text':
                result['retrieval_type'] = 'table'
                result['reasoning'] = 'Financial keywords detected - starting with tables'

            return result

        except Exception as e:
            rag_logger.error(f"❌ Error in initial retrieval decision: {e}")
            return {
                'retrieval_type': 'table' if has_financial_keywords else 'text',
                'reasoning': f'Fallback: {str(e)}'
            }

    async def _decide_retrieval_type(
        self,
        question: str,
        current_answer: str,
        iteration: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        Decide what type of chunks to retrieve next based on current state.

        Uses evaluation feedback and retrieval history to make intelligent decisions.
        """
        table_count = self.current_session['table_retrieved_count']
        text_count = self.current_session['text_retrieved_count']

        # DYNAMIC SWITCHING: Force trying alternative source if one hasn't been tried
        if table_count == 0 and text_count > 0:
            return {
                'retrieval_type': 'table',
                'reasoning': 'Dynamic switch: text tried but tables not yet explored'
            }
        elif text_count == 0 and table_count > 0:
            return {
                'retrieval_type': 'text',
                'reasoning': 'Dynamic switch: tables tried but text not yet explored'
            }

        if not self.cerebras_available:
            # Fallback: alternate between table and text
            next_type = 'text' if table_count > text_count else 'table'
            return {
                'retrieval_type': next_type,
                'reasoning': f'Fallback alternation: {next_type}'
            }

        # Build context from evaluation history
        eval_context = ""
        if self.current_session['evaluation_history']:
            recent = self.current_session['evaluation_history'][-1]['evaluation']
            eval_context = f"Last quality: {recent.get('quality_score', 0):.2f}, Issues: {recent.get('issues', [])[:2]}"

        prompt = f"""You are deciding what to retrieve next to improve an answer.

QUESTION: {question}

CURRENT STATE:
- Iteration: {iteration}/{max_iterations}
- Tables retrieved: {table_count} times
- Text retrieved: {text_count} times
- {eval_context}

CURRENT ANSWER PREVIEW: {current_answer[:500] if current_answer else 'No answer yet'}...

DECISION RULES:
1. If answer lacks NUMBERS/METRICS → retrieve TABLE
2. If answer lacks CONTEXT/EXPLANATION → retrieve TEXT
3. If both have been tried but quality is low → try the less-used source
4. Balance between sources for comprehensive answers

Return ONLY valid JSON:
{{"retrieval_type": "table" or "text", "reasoning": "Brief explanation"}}"""

        try:
            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"),
                messages=[
                    {"role": "system", "content": "Return only JSON, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)

            # Validate
            if result['retrieval_type'] not in ['table', 'text']:
                result['retrieval_type'] = 'text' if table_count > text_count else 'table'

            return result

        except Exception as e:
            rag_logger.error(f"❌ Error in retrieval decision: {e}")
            next_type = 'text' if table_count > text_count else 'table'
            return {
                'retrieval_type': next_type,
                'reasoning': f'Fallback: {str(e)}'
            }

    async def _decide_next_retrieval_strategy(
        self,
        question: str,
        current_answer: str,
        evaluation: Dict[str, Any],
        iteration: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Decide next retrieval strategy based on evaluation feedback."""

        quality_score = evaluation.get('quality_score', 0.0)
        issues = evaluation.get('issues', [])
        missing_info = evaluation.get('missing_info', [])

        table_count = self.current_session['table_retrieved_count']
        text_count = self.current_session['text_retrieved_count']

        # If quality is low after tables, try text (and vice versa)
        last_type = self.current_session['retrieval_history'][-1]['type'] if self.current_session['retrieval_history'] else 'table'

        if quality_score < 0.7:
            if last_type == 'table':
                return {'next_type': 'text', 'reasoning': f'Low quality ({quality_score:.2f}) after tables - trying text'}
            else:
                return {'next_type': 'table', 'reasoning': f'Low quality ({quality_score:.2f}) after text - trying tables'}

        # Check missing info keywords
        missing_text = ' '.join(missing_info).lower()
        if any(kw in missing_text for kw in ['number', 'figure', 'amount', 'revenue', 'metric']):
            return {'next_type': 'table', 'reasoning': 'Missing numerical data - need tables'}
        if any(kw in missing_text for kw in ['context', 'explanation', 'reason', 'why', 'describe']):
            return {'next_type': 'text', 'reasoning': 'Missing context - need text'}

        # Default: alternate
        next_type = 'text' if last_type == 'table' else 'table'
        return {'next_type': next_type, 'reasoning': 'Alternating for coverage'}

    async def _retrieve_table_chunks(
        self,
        question: str,
        tables: List[Dict[str, Any]],
        iteration: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve table chunks using LLM-based selection.

        LLM sees ALL available tables and selects the most relevant one(s).
        """
        if not tables:
            return []

        # Use LLM to select relevant tables
        selected_tables, reasoning = await self._select_tables_by_llm(
            question=question,
            tables=tables,
            iteration=iteration,
            max_tables=2  # Select up to 2 tables per iteration
        )

        # Convert to chunk format
        chunks = []
        for table in selected_tables:
            chunk = {
                'chunk_text': table.get('content', ''),
                'ticker': table.get('ticker'),
                'fiscal_year': table.get('fiscal_year'),
                'chunk_type': 'table',
                'sec_section': table.get('sec_section'),
                'sec_section_title': table.get('sec_section_title'),
                'path_string': table.get('path_string'),
                'is_financial_statement': table.get('is_financial_statement', False),
                'statement_type': table.get('statement_type', ''),
                'similarity': 1.0,
                'selection_method': 'llm_iterative_selection',
                'selection_reasoning': reasoning,
                'iteration': iteration
            }
            chunks.append(chunk)

        return chunks

    async def _retrieve_text_chunks(
        self,
        query: str,
        query_embedding: np.ndarray,
        ticker: str,
        fiscal_year: int = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve text chunks using hybrid search + reranking.
        """
        try:
            # Get initial candidates
            chunks = await self.database_manager.search_10k_filings_async(
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year
            )

            # Filter to text only (no tables)
            text_chunks = [c for c in chunks if c.get('chunk_type') != 'table']
            text_chunks = text_chunks[:50]  # Limit candidates

            if not text_chunks:
                return []

            # Hybrid scoring
            chunk_texts = [c['chunk_text'] for c in text_chunks]
            tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
            query_tfidf = tfidf_vectorizer.transform([query])
            keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

            semantic_scores = np.array([c.get('similarity', 0.0) for c in text_chunks])

            # Normalize and combine
            if semantic_scores.max() - semantic_scores.min() > 0:
                semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
            else:
                semantic_norm = semantic_scores

            if keyword_scores.max() - keyword_scores.min() > 0:
                keyword_norm = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min())
            else:
                keyword_norm = keyword_scores

            hybrid_scores = (0.7 * semantic_norm) + (0.3 * keyword_norm)

            for i, chunk in enumerate(text_chunks):
                chunk['hybrid_score'] = float(hybrid_scores[i])

            # Cross-encoder reranking if available
            if self.cross_encoder_available:
                pairs = [[query, c['chunk_text']] for c in text_chunks]
                cross_scores = self.cross_encoder.predict(pairs)
                for i, chunk in enumerate(text_chunks):
                    chunk['cross_encoder_score'] = float(cross_scores[i])
                    chunk['similarity'] = float(cross_scores[i])
                text_chunks.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            else:
                text_chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)

            return text_chunks[:top_k]

        except Exception as e:
            rag_logger.error(f"❌ Error retrieving text chunks: {e}")
            return []

    async def _get_all_tables_for_ticker(self, ticker: str, fiscal_year: int = None) -> List[Dict[str, Any]]:
        """Get all available tables for a ticker."""
        try:
            tables = await self.database_manager.get_all_tables_for_ticker_async(
                ticker=ticker,
                fiscal_year=fiscal_year
            )
            return tables
        except Exception as e:
            rag_logger.error(f"❌ Failed to get tables: {e}")
            return []

    async def _select_tables_by_llm(
        self,
        question: str,
        tables: List[Dict[str, Any]],
        iteration: int,
        max_tables: int = 2
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Use LLM to select relevant tables.

        Considers what tables have already been selected in previous iterations.
        """
        if not self.cerebras_available or not tables:
            # Fallback: return financial statement tables
            financial = [t for t in tables if t.get('is_financial_statement')]
            return financial[:max_tables], "Fallback selection"

        # Build list of already selected tables
        already_selected = set()
        for chunk in self.current_session['accumulated_table_chunks']:
            path = chunk.get('path_string', '')
            if path:
                already_selected.add(path)

        # Build table list for LLM
        table_list = []
        for idx, table in enumerate(tables, 1):
            path = table.get('path_string', 'Unknown')
            sec_section = table.get('sec_section_title', 'Unknown')
            is_financial = table.get('is_financial_statement', False)
            stmt_type = table.get('statement_type', '')

            prefix = ""
            if is_financial and stmt_type in ['income_statement', 'balance_sheet', 'cash_flow']:
                prefix = "🌟 CORE: "
            if path in already_selected:
                prefix += "[ALREADY SELECTED] "

            table_list.append(f"{idx}. {prefix}{path} ({sec_section})")

        tables_text = "\n".join(table_list[:40])

        prompt = f"""Select the MOST RELEVANT tables to answer this question.
This is iteration {iteration} - avoid selecting tables already marked [ALREADY SELECTED] unless essential.

QUESTION: {question}

AVAILABLE TABLES:
{tables_text}

RULES:
- Prioritize 🌟 CORE financial statements for numeric questions
- Select MAX {max_tables} tables
- Avoid duplicates from previous iterations
- Quality over quantity

Return ONLY valid JSON:
{{"selected_table_indices": [1, 2], "reasoning": "Brief explanation"}}"""

        try:
            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"),
                messages=[
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)
            selected_indices = result.get('selected_table_indices', [])
            reasoning = result.get('reasoning', '')

            selected = []
            for idx in selected_indices:
                if 1 <= idx <= len(tables):
                    selected.append(tables[idx - 1])

            return selected[:max_tables], reasoning

        except Exception as e:
            rag_logger.error(f"❌ Table selection error: {e}")
            financial = [t for t in tables if t.get('is_financial_statement')]
            return financial[:max_tables], f"Fallback: {str(e)}"

    async def _generate_iterative_answer(
        self,
        question: str,
        new_chunks: List[Dict[str, Any]],
        accumulated_chunks: List[Dict[str, Any]],
        previous_answer: str,
        iteration: int,
        retrieval_type: str
    ) -> str:
        """
        Generate or refine answer using new chunks.

        Builds on previous answer with new information.
        """
        if not self.cerebras_available:
            # Fallback: simple concatenation
            context = "\n\n".join([c.get('chunk_text', '')[:500] for c in accumulated_chunks[:10]])
            return f"Based on available data:\n\n{context}"

        # Format context
        context_parts = []
        for i, chunk in enumerate(accumulated_chunks[:15], 1):
            chunk_type = chunk.get('chunk_type', 'text')
            content = chunk.get('chunk_text', '')[:1000]
            sec = chunk.get('sec_section_title', 'Unknown')
            context_parts.append(f"[{i}] ({chunk_type.upper()}) {sec}:\n{content}")

        context = "\n\n".join(context_parts)

        previous_context = ""
        if previous_answer:
            previous_context = f"\nPREVIOUS ANSWER (refine and improve):\n{previous_answer}\n"

        prompt = f"""Answer this question using the provided 10-K SEC filing data.
This is iteration {iteration} - build upon any previous answer with new information.

QUESTION: {question}

{previous_context}

NEW {retrieval_type.upper()} DATA RETRIEVED:
{context}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer
2. Use specific numbers and quotes from the data
3. Cite sources as [1], [2], etc.
4. If previous answer exists, IMPROVE it with new information
5. Be specific and detailed

ANSWER:"""

        try:
            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"),
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Provide accurate, detailed answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            rag_logger.error(f"❌ Answer generation error: {e}")
            return previous_answer or "Error generating answer."

    async def _evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Evaluate answer quality and identify gaps.

        Returns quality score and suggestions for improvement.
        """
        if not self.cerebras_available:
            # Fallback: basic heuristics
            has_numbers = any(c.isdigit() for c in answer)
            is_long_enough = len(answer) > 200
            score = 0.5 + (0.2 if has_numbers else 0) + (0.2 if is_long_enough else 0)
            return {
                'quality_score': score,
                'issues': [] if score > 0.7 else ['May be incomplete'],
                'missing_info': [],
                'suggestions': []
            }

        prompt = f"""Evaluate this answer to a financial question.

QUESTION: {question}

ANSWER: {answer[:1500]}

EVALUATION CRITERIA:
1. COMPLETENESS: Does it fully answer the question?
2. SPECIFICITY: Does it include specific numbers and data?
3. ACCURACY: Is it supported by the context?
4. CLARITY: Is it well-structured?

Return ONLY valid JSON:
{{
    "quality_score": 0.0 to 1.0,
    "issues": ["issue1", "issue2"],
    "missing_info": ["what's missing"],
    "suggestions": ["how to improve"]
}}"""

        try:
            response = self.cerebras_client.chat.completions.create(
                model=self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"),
                messages=[
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            return json.loads(response_text)

        except Exception as e:
            rag_logger.error(f"❌ Evaluation error: {e}")
            return {
                'quality_score': 0.5,
                'issues': [str(e)],
                'missing_info': [],
                'suggestions': []
            }

    def format_10k_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string for the LLM."""
        if not chunks:
            return ""

        context_parts = []
        context_parts.append("=" * 80)
        context_parts.append("10-K SEC FILINGS DATA (ITERATIVE SEARCH)")
        context_parts.append("=" * 80)
        context_parts.append("")

        for idx, chunk in enumerate(chunks, 1):
            ticker = chunk.get('ticker', 'UNKNOWN')
            fiscal_year = chunk.get('fiscal_year', 'UNKNOWN')
            sec_section = chunk.get('sec_section_title', 'Unknown Section')
            chunk_type = chunk.get('chunk_type', 'text')

            citation = f"[10K{idx}]"

            context_parts.append(f"{citation} {ticker} - FY{fiscal_year} - {sec_section}")
            if chunk_type == 'table':
                context_parts.append(f"Type: Financial Table")
            context_parts.append(f"Content: {chunk['chunk_text']}")
            context_parts.append("")

        context_parts.append("=" * 80)
        return "\n".join(context_parts)

    def get_10k_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get citation information for chunks."""
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
                "similarity": chunk.get('similarity', 0),
                "iteration": chunk.get('iteration', 0)
            }
            citations.append(citation)

        return citations
