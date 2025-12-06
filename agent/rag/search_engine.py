#!/usr/bin/env python3
"""
Search Engine for the RAG system.

This module handles all search operations including vector search, keyword search,
and hybrid search functionality for the RAG system.
"""

import json
import logging
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from psycopg2.extras import RealDictCursor

# Import local modules
from .config import Config
from .database_manager import DatabaseManager
from .rag_utils import extract_keywords, combine_search_results, extract_ticker_simple

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & SETUP
# ═════════════════════════════════════════════════════════════════════

class SearchEngine:
    """Handles all search operations for the RAG system."""

    def __init__(self, config: Config, database_manager: DatabaseManager):
        """Initialize the search engine."""
        self.config = config
        self.database_manager = database_manager

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.get("embedding_model"))

        # Create a thread pool executor for CPU-bound embedding operations
        # This prevents blocking the event loop
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

        logger.info("SearchEngine initialized successfully")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: QUERY EMBEDDING GENERATION
    # ═════════════════════════════════════════════════════════════════════

    async def encode_query_async(self, query: str) -> np.ndarray:
        """Generate query embedding asynchronously using thread pool executor.
        
        This prevents blocking the event loop while encoding queries.
        
        Args:
            query (str): Query text to encode
            
        Returns:
            np.ndarray: Query embedding vector
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.embedding_model.encode, [query])

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: HYBRID SEARCH EXECUTION (Vector + Keyword Search)
    # ═════════════════════════════════════════════════════════════════════

    def search_similar_chunks(self, query: str, max_results: int = None, target_quarter: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using hybrid vector + keyword search with optional quarter filtering.
        
        NOTE: This is a synchronous method. For async contexts, use the async version.
        """
        search_engine_start = time.time()
        rag_logger.info(f"🔍 Starting hybrid search for query: '{query}'")
        if max_results is None:
            max_results = self.config.get("chunks_per_quarter", 15)
        rag_logger.info(f"📊 Search parameters: chunks_per_quarter={max_results}, similarity_threshold={self.config.get('similarity_threshold')}, target_quarter={target_quarter}")
        
        # Check if hybrid search is enabled
        if not self.config.get("hybrid_search_enabled", True):
            rag_logger.info("⚠️ Hybrid search disabled")
            return []
        
        # Time embedding generation
        encode_start = time.time()
        rag_logger.info(f"🧠 Generating query embedding using model: {self.config.get('embedding_model')}")
        # NOTE: This is still synchronous - the async version will use async methods
        query_embedding = self.embedding_model.encode([query])
        encode_time = time.time() - encode_start
        rag_logger.info(f"✅ Query embedding generated in {encode_time:.3f}s: shape={query_embedding.shape}")
        
        # Extract ticker from query for targeted search using simple regex
        ticker = extract_ticker_simple(query)
        rag_logger.info(f"📊 Extracted ticker from query: {ticker}")
        
        try:
            # Perform both vector and keyword searches
            vector_results = []
            keyword_results = []
            
            if ticker:
                # Search with ticker filtering for both vector and keyword in parallel
                rag_logger.info(f"🎯 Performing ticker-specific hybrid search for: {ticker}")
                
                # Run vector and keyword searches in parallel for better performance
                parallel_retrieval = self.config.get("parallel_retrieval_enabled", True)
                
                if parallel_retrieval:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        # Submit both search tasks in parallel
                        vector_future = executor.submit(
                            self.database_manager._search_postgres_with_ticker, 
                            query_embedding, 
                            ticker, 
                            target_quarter
                        )
                        keyword_future = executor.submit(
                            self._search_keywords_with_ticker, 
                            query, 
                            ticker, 
                            target_quarter
                        )
                        
                        # Collect results as they complete
                        vector_results = vector_future.result()
                        keyword_results = keyword_future.result()
                else:
                    # Sequential search (fallback for compatibility)
                    vector_results = self.database_manager._search_postgres_with_ticker(query_embedding, ticker, target_quarter)
                    keyword_results = self._search_keywords_with_ticker(query, ticker, target_quarter)
                
                rag_logger.info(f"📊 Vector search results: {len(vector_results)} chunks")
                rag_logger.info(f"📊 Keyword search results: {len(keyword_results)} chunks")
                
                # No fallback - let it fail if no results
                if not vector_results and not keyword_results:
                    rag_logger.warning(f"⚠️ No results from ticker-specific search for {ticker}")
            else:
                # General search without ticker filtering - run searches in parallel
                rag_logger.info("🌐 Performing general hybrid search across all companies")
                
                # Run vector and keyword searches in parallel for better performance
                parallel_retrieval = self.config.get("parallel_retrieval_enabled", True)
                
                if parallel_retrieval:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        # Submit both search tasks in parallel
                        vector_future = executor.submit(
                            self.database_manager._search_postgres_general, 
                            query_embedding, 
                            max_results, 
                            target_quarter
                        )
                        keyword_future = executor.submit(
                            self._search_keywords, 
                            query, 
                            max_results, 
                            target_quarter
                        )
                        
                        # Collect results as they complete
                        vector_results = vector_future.result()
                        keyword_results = keyword_future.result()
                else:
                    # Sequential search (fallback for compatibility)
                    vector_results = self.database_manager._search_postgres_general(query_embedding, max_results, target_quarter)
                    keyword_results = self._search_keywords(query, max_results, target_quarter)
                
                rag_logger.info(f"📊 Vector search results: {len(vector_results)} chunks")
                rag_logger.info(f"📊 Keyword search results: {len(keyword_results)} chunks")
            
            # Combine and rank results
            combine_start = time.time()
            if vector_results or keyword_results:
                combined_results = combine_search_results(
                    vector_results, keyword_results, 
                    self.config.get("vector_weight", 0.7), 
                    self.config.get("keyword_weight", 0.3),
                    self.config.get("similarity_threshold", 0.3)
                )
                combine_time = time.time() - combine_start
                rag_logger.info(f"⏱️  Combined results in {combine_time:.3f}s: {len(combined_results)} chunks")
                
                # Final timing summary
                total_search_time = time.time() - search_engine_start
                rag_logger.info("=" * 80)
                rag_logger.info("⏱️  SEARCH ENGINE TIMING SUMMARY")
                rag_logger.info("=" * 80)
                rag_logger.info(f"🧠 Embedding generation: {encode_time:.3f}s")
                rag_logger.info(f"🔍 Vector search: {len(vector_results)} chunks")
                rag_logger.info(f"📝 Keyword search: {len(keyword_results)} chunks")
                rag_logger.info(f"🔗 Combination: {combine_time:.3f}s")
                rag_logger.info(f"")
                rag_logger.info(f"⏱️  TOTAL SEARCH: {total_search_time:.3f}s")
                rag_logger.info(f"📊 Final results: {len(combined_results)} chunks")
                rag_logger.info("=" * 80)
                
                return combined_results
            else:
                rag_logger.warning("⚠️ No results from either vector or keyword search")
                total_search_time = time.time() - search_engine_start
                rag_logger.info(f"⏱️  Total search time: {total_search_time:.3f}s (no results)")
                return []
                    
        except Exception as e:
            rag_logger.error(f"❌ Hybrid search failed: {e}")
            return []

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 4: KEYWORD SEARCH OPERATIONS
    # ═════════════════════════════════════════════════════════════════════

    def _search_keywords(self, query: str, max_results: int = None, target_quarter: str = None) -> List[Dict[str, Any]]:
        """Search using PostgreSQL full-text search with keywords."""
        if not self.config.get("hybrid_search_enabled", True):
            return []
        
        if max_results is None:
            max_results = self.config.get("keyword_max_results", 10)
        
        try:
            rag_logger.info(f"🔍 Starting keyword search for query: '{query}'")
            keywords = extract_keywords(query)
            rag_logger.info(f"📝 Extracted keywords: {keywords}")
            
            if not keywords:
                rag_logger.warning("⚠️ No keywords extracted from query")
                return []
            
            conn = self.database_manager._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Create search terms for PostgreSQL full-text search
            search_terms = " | ".join(keywords)
            
            # Build query with optional quarter filtering
            if target_quarter:
                year, quarter = target_quarter.split('_')
                quarter_num = quarter[1:]  # Remove 'q' prefix
                
                query_sql = """
                SELECT chunk_text, metadata, year, quarter, ticker, 
                       ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) as keyword_score,
                       chunk_index
                FROM transcript_chunks 
                WHERE year = %s AND quarter = %s
                  AND to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY keyword_score DESC
                LIMIT %s
                """
                
                cursor.execute(query_sql, (search_terms, int(year), int(quarter_num), search_terms, max_results))
            else:
                query_sql = """
                SELECT chunk_text, metadata, year, quarter, ticker, 
                       ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) as keyword_score,
                       chunk_index
                FROM transcript_chunks 
                WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY keyword_score DESC
                LIMIT %s
                """
                
                cursor.execute(query_sql, (search_terms, search_terms, max_results))
            
            results = cursor.fetchall()
            self.database_manager._return_db_connection(conn)
            
            rag_logger.info(f"✅ Keyword search returned {len(results)} results")
            
            # Convert to expected format
            chunks = []
            for row in results:
                keyword_score = float(row['keyword_score'])
                
                # Handle metadata
                metadata = row['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                elif metadata is None:
                    metadata = {}
                
                chunk = {
                    'chunk_text': row['chunk_text'],
                    'similarity': keyword_score,  # Use keyword score as similarity
                    'distance': 1 - keyword_score,  # Convert to distance
                    'metadata': metadata,
                    'citation': row['chunk_index'],
                    'year': row['year'],
                    'quarter': row['quarter'],
                    'ticker': row['ticker'],
                    'search_type': 'keyword'  # Mark as keyword search result
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search keywords: {e}")
            return []
    
    def _search_keywords_with_ticker(self, query: str, ticker: str, target_quarter: str = None) -> List[Dict[str, Any]]:
        """Search using PostgreSQL full-text search with keywords and ticker filtering."""
        if not self.config.get("hybrid_search_enabled", True):
            return []
        
        try:
            rag_logger.info(f"🔍 Starting keyword search for ticker: {ticker}, query: '{query}'")
            keywords = extract_keywords(query)
            rag_logger.info(f"📝 Extracted keywords: {keywords}")
            
            if not keywords:
                rag_logger.warning("⚠️ No keywords extracted from query")
                return []
            
            conn = self.database_manager._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Create search terms for PostgreSQL full-text search
            search_terms = " | ".join(keywords)
            
            # Build query with ticker and optional quarter filtering
            if target_quarter:
                year, quarter = target_quarter.split('_')
                quarter_num = quarter[1:]  # Remove 'q' prefix
                
                query_sql = """
                SELECT chunk_text, metadata, year, quarter, ticker, 
                       ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) as keyword_score,
                       chunk_index
                FROM transcript_chunks 
                WHERE ticker = %s AND year = %s AND quarter = %s
                  AND to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY keyword_score DESC
                LIMIT %s
                """
                
                cursor.execute(query_sql, (search_terms, ticker, int(year), int(quarter_num), search_terms, self.config.get("keyword_max_results", 10)))
            else:
                query_sql = """
                SELECT chunk_text, metadata, year, quarter, ticker, 
                       ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) as keyword_score,
                       chunk_index
                FROM transcript_chunks 
                WHERE ticker = %s
                  AND to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY keyword_score DESC
                LIMIT %s
                """
                
                cursor.execute(query_sql, (search_terms, ticker, search_terms, self.config.get("keyword_max_results", 10)))
            
            results = cursor.fetchall()
            self.database_manager._return_db_connection(conn)
            
            rag_logger.info(f"✅ Keyword search with ticker returned {len(results)} results")
            
            # Convert to expected format
            chunks = []
            for row in results:
                keyword_score = float(row['keyword_score'])
                
                # Handle metadata
                metadata = row['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                elif metadata is None:
                    metadata = {}
                
                chunk = {
                    'chunk_text': row['chunk_text'],
                    'similarity': keyword_score,  # Use keyword score as similarity
                    'distance': 1 - keyword_score,  # Convert to distance
                    'metadata': metadata,
                    'citation': row['chunk_index'],
                    'year': row['year'],
                    'quarter': row['quarter'],
                    'ticker': row['ticker'],
                    'search_type': 'keyword'  # Mark as keyword search result
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search keywords with ticker: {e}")
            return []

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 5: MULTI-QUARTER SEARCH OPERATIONS
    # ═════════════════════════════════════════════════════════════════════

    async def _search_multiple_quarters_async(self, query: str, target_quarters: List[str], chunks_per_quarter: int = None) -> List[Dict[str, Any]]:
        """Async version of multi-quarter search.
        
        REFACTORED: Now properly uses run_in_executor without creating nested event loops.
        This prevents blocking and ensures better concurrency for multiple users.
        """
        rag_logger.info(f"🚀 Starting async parallel search across {len(target_quarters)} quarters")
        
        # Get chunks per quarter from config if not provided
        if chunks_per_quarter is None:
            chunks_per_quarter = self.config.get("chunks_per_quarter", 15)
        
        # Request 2x chunks initially to have more to rank from
        base_results_per_quarter = max(10, chunks_per_quarter * 2)
        
        loop = asyncio.get_event_loop()
        
        async def search_quarter_async(quarter):
            """Async search function for a single quarter"""
            rag_logger.info(f"🔍 Async search in quarter: {quarter}")
            # Use run_in_executor to run sync search without blocking event loop
            return await loop.run_in_executor(None, self.search_similar_chunks, query, base_results_per_quarter, quarter)
        
        # Create all tasks
        tasks = [search_quarter_async(quarter) for quarter in target_quarters]
        
        # Execute all searches in parallel using asyncio.gather
        quarter_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        quarter_results = {}
        for i, quarter in enumerate(target_quarters):
            try:
                quarter_chunks = quarter_results_list[i]
                if isinstance(quarter_chunks, Exception):
                    rag_logger.error(f"❌ Error searching quarter {quarter}: {quarter_chunks}")
                    quarter_results[quarter] = []
                else:
                    quarter_results[quarter] = quarter_chunks
                    rag_logger.info(f"✅ Quarter {quarter}: found {len(quarter_chunks)} chunks")
            except Exception as e:
                rag_logger.error(f"❌ Error processing quarter {quarter}: {e}")
                quarter_results[quarter] = []
        
        # Now rank within each quarter and take top N from each (same number per quarter)
        all_chunks = []
        for i, quarter in enumerate(target_quarters):
            quarter_chunks = quarter_results.get(quarter, [])
            
            # Sort chunks within this quarter by similarity
            quarter_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Take top N chunks from this quarter
            top_quarter_chunks = quarter_chunks[:chunks_per_quarter]
            
            # Add quarter info for debugging
            for chunk in top_quarter_chunks:
                chunk['source_quarter'] = quarter
            
            all_chunks.extend(top_quarter_chunks)
            rag_logger.info(f"📊 Quarter {quarter}: selected top {len(top_quarter_chunks)} chunks (similarity range: {top_quarter_chunks[0].get('similarity', 0):.3f} - {top_quarter_chunks[-1].get('similarity', 0):.3f})" if top_quarter_chunks else f"📊 Quarter {quarter}: no chunks selected")
        
        # Final sort by similarity across all selected chunks
        all_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        rag_logger.info(f"📊 Async multi-quarter search: {len(all_chunks)} total chunks from {len(target_quarters)} quarters")
        rag_logger.info(f"📊 Per-quarter distribution: {chunks_per_quarter} chunks per quarter")
        
        return all_chunks
