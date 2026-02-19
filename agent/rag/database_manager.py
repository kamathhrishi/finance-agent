#!/usr/bin/env python3
"""
Database Manager for the RAG system.

This module handles all database operations including connection management,
quarter queries, and search operations for the RAG system.

Debug Mode:
    Enable debug mode by setting RAG_DEBUG_MODE=true in your environment.
    When enabled, all database queries will log EXPLAIN ANALYZE output to help
    understand query execution plans and optimize performance.
"""

import json
import logging
import asyncio
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncpg
import numpy as np
from typing import List, Dict, Any, Optional

# Import local modules
from .config import Config

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & CONNECTION MANAGEMENT
# ═════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Handles all database operations for the RAG system."""
    
    def __init__(self, config: Config):
        """Initialize the database manager."""
        self.config = config
        
        # Database connections
        self.postgres_connection_string = self.config.get_connection_string()
        self.pgvector_connection_string = self.config.get_pgvector_connection_string()
        
        # Connection pools
        self._connection_pool = None
        self.db_pool = None
        self.pgvector_pool = None
        
        # Debug mode from config
        self.debug_mode = self.config.get("debug_mode", False)
        
        # Initialize connection pool
        self._init_connection_pool()
        
        if self.debug_mode:
            logger.info("🔍 Debug mode enabled - EXPLAIN ANALYZE will be logged for queries")
        logger.info("DatabaseManager initialized successfully")
    
    def set_database_connection(self, db_connection):
        """Set the database connection for retrieving conversation history."""
        # This method is kept for compatibility with the main RAG system
        pass

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: QUARTER QUERIES & RESOLUTION
    # ═════════════════════════════════════════════════════════════════════

    def get_latest_quarter_for_company(self, ticker: str) -> Optional[str]:
        """Get the latest available quarter for a specific company."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Find the latest quarter for this specific company
                query = """
                SELECT year, quarter 
                FROM transcript_chunks 
                WHERE ticker = %s 
                ORDER BY year DESC, quarter DESC 
                LIMIT 1
                """
                
                cursor.execute(query, (ticker.upper(),))
                result = cursor.fetchone()
                self._return_db_connection(conn)
                
                if result:
                    year, quarter = result
                    latest_quarter = f"{year}_q{quarter}"
                    logger.info(f"📅 Latest quarter for {ticker}: {latest_quarter}")
                    return latest_quarter
                else:
                    logger.warning(f"⚠️ No quarters found for company {ticker}")
                    return None
                    
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning(f"⚠️ Database connection error (attempt {attempt + 1}/{max_retries + 1}) finding latest quarter for {ticker}: {e}")
                try:
                    if 'conn' in locals():
                        conn.close()
                except:
                    pass
                if attempt >= max_retries:
                    logger.error(f"❌ Error finding latest quarter for {ticker} after {max_retries + 1} attempts: {e}")
                    return None
                import time
                time.sleep(0.1 * (attempt + 1))
            except Exception as e:
                logger.error(f"❌ Error finding latest quarter for {ticker}: {e}")
                return None
    
    def get_last_n_quarters_for_company(self, ticker: str, n: int) -> List[str]:
        """
        Get the last N quarters for a specific company.
        
        This method finds the latest quarter for the company and returns N quarters
        going backwards chronologically. The SQL query uses LIMIT to return exactly
        N quarters (or fewer if the company doesn't have N quarters available).
        
        Examples:
        - If latest is 2025 Q2 and n=6:
          Returns: [2025_q2, 2025_q1, 2024_q4, 2024_q3, 2024_q2, 2024_q1]
        
        - If latest is 2025 Q2 and n=8:
          Returns: [2025_q2, 2025_q1, 2024_q4, 2024_q3, 2024_q2, 2024_q1, 2023_q4, 2023_q3]
        
        The SQL query orders by year DESC, quarter DESC, which correctly sequences:
        - Latest year: Q4, Q3, Q2, Q1 (if available)
        - Previous year: Q4, Q3, Q2, Q1 (if available)
        - And so on, going back in time...
        
        Args:
            ticker: Company ticker symbol
            n: Number of quarters to retrieve (going back from latest)
            
        Returns:
            List of quarter strings in format ['2025_q2', '2025_q1', '2024_q4', ...]
            (latest first, going backwards chronologically)
            Will return fewer than N quarters if the company doesn't have that many available.
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Find the last N quarters for this specific company
                # ORDER BY year DESC, quarter DESC ensures:
                # - Latest year first (2025 before 2024)
                # - Within each year, latest quarter first (Q4, Q3, Q2, Q1)
                # This gives us: 2025_q2, 2025_q1, 2024_q4, 2024_q3, 2024_q2, 2024_q1, etc.
                query = """
                SELECT DISTINCT year, quarter 
                FROM transcript_chunks 
                WHERE ticker = %s 
                ORDER BY year DESC, quarter DESC 
                LIMIT %s
                """
                
                cursor.execute(query, (ticker.upper(), n))
                results = cursor.fetchall()
                self._return_db_connection(conn)
                
                if results:
                    quarters = [f"{year}_q{quarter}" for year, quarter in results]
                    logger.info(f"📅 Last {n} quarters for {ticker}: {quarters}")
                    return quarters
                else:
                    # ✅ IMPROVED: Better logging for debugging
                    logger.warning(f"⚠️ No quarters found for company {ticker} in database")
                    logger.warning(f"⚠️ This could mean: (1) ticker not in DB, (2) no transcript data, or (3) data ingestion issue")
                    logger.warning(f"⚠️ Check if {ticker} data exists: SELECT COUNT(*) FROM transcript_chunks WHERE ticker='{ticker.upper()}'")
                    return []

            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                # Connection error - retry with fresh connection
                error_msg = str(e)
                logger.warning(f"⚠️ Database connection error (attempt {attempt + 1}/{max_retries + 1}) finding last {n} quarters for {ticker}: {error_msg}")
                
                # Try to clean up the connection
                try:
                    if 'conn' in locals():
                        try:
                            conn.close()
                        except:
                            pass
                except:
                    pass
                
                # If this was the last attempt, log error and return empty
                if attempt >= max_retries:
                    logger.error(f"❌ Database error finding last {n} quarters for {ticker} after {max_retries + 1} attempts: {e}")
                    logger.error(f"❌ Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"❌ Stack trace: {traceback.format_exc()}")
                    return []
                
                # Wait a bit before retrying
                import time
                time.sleep(0.1 * (attempt + 1))
                
            except Exception as e:
                # ✅ IMPROVED: More detailed error logging
                logger.error(f"❌ Database error finding last {n} quarters for {ticker}: {e}")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"❌ Stack trace: {traceback.format_exc()}")
                return []
    
    def get_general_latest_quarter(self) -> Optional[str]:
        """Get the latest available quarter across all companies."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Find the latest quarter across all companies
                query = """
                SELECT year, quarter 
                FROM transcript_chunks 
                ORDER BY year DESC, quarter DESC 
                LIMIT 1
                """
                
                cursor.execute(query)
                result = cursor.fetchone()
                self._return_db_connection(conn)
                
                if result:
                    year, quarter = result
                    latest_quarter = f"{year}_q{quarter}"
                    logger.info(f"📅 General latest quarter: {latest_quarter}")
                    return latest_quarter
                else:
                    logger.warning("⚠️ No quarters found in database")
                    return None
                    
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning(f"⚠️ Database connection error (attempt {attempt + 1}/{max_retries + 1}) finding general latest quarter: {e}")
                try:
                    if 'conn' in locals():
                        conn.close()
                except:
                    pass
                if attempt >= max_retries:
                    logger.error(f"❌ Error finding general latest quarter after {max_retries + 1} attempts: {e}")
                    return None
                import time
                time.sleep(0.1 * (attempt + 1))
            except Exception as e:
                logger.error(f"❌ Error finding general latest quarter: {e}")
                return None
    
    def resolve_latest_quarter_reference(self, quarter_reference: str, ticker: str = None) -> str:
        """
        Resolve a 'latest' quarter reference to an actual quarter.
        
        Args:
            quarter_reference: The quarter reference from question analysis
            ticker: Company ticker (optional, for company-specific latest)
            
        Returns:
            Actual quarter string or original reference if not 'latest'
        """
        if quarter_reference == 'latest':
            if ticker:
                # Get company-specific latest quarter
                company_latest = self.get_latest_quarter_for_company(ticker)
                if company_latest:
                    logger.info(f"📅 Resolved 'latest' to company-specific quarter: {ticker} -> {company_latest}")
                    return company_latest
                else:
                    # Fallback to general latest
                    general_latest = self.get_general_latest_quarter()
                    if general_latest:
                        logger.info(f"📅 Resolved 'latest' to general latest quarter: {general_latest}")
                        return general_latest
                    else:
                        logger.warning("⚠️ No latest quarter found")
                        return "NO_QUARTERS_AVAILABLE"
            else:
                # Get general latest quarter
                general_latest = self.get_general_latest_quarter()
                if general_latest:
                    logger.info(f"📅 Resolved 'latest' to general latest quarter: {general_latest}")
                    return general_latest
                else:
                    logger.warning("⚠️ No latest quarter found")
                    return "NO_QUARTERS_AVAILABLE"
        else:
            # Not a 'latest' reference, return as-is
            return quarter_reference

    def _init_connection_pool(self):
        """Initialize connection pool for database operations."""
        try:
            from psycopg2 import pool
            import os
            
            # Create connection pool with optimal size for parallel processing
            # Increased from 20 to 50 to support 10+ concurrent users (each user may need 2-5 connections)
            pool_size = min(os.cpu_count() * 2, 50)  # Pool size based on CPU cores, max 50
            
            self._connection_pool = pool.ThreadedConnectionPool(
                minconn=2,  # Minimum connections
                maxconn=pool_size,  # Maximum connections
                dsn=self.pgvector_connection_string
            )
            
            logger.info(f"✅ Database connection pool initialized with {pool_size} max connections")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize connection pool: {e}")
            logger.info("Falling back to individual connections per query")
            self._connection_pool = None
    
    def _get_db_connection(self):
        """Get a database connection from the pool or create a new one."""
        if self._connection_pool:
            try:
                conn = self._connection_pool.getconn()
                # Validate connection is still alive
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return conn
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    # Connection is stale, close it and get a new one
                    logger.warning(f"⚠️  Connection from pool is stale: {e}, getting new connection")
                    try:
                        conn.close()
                    except:
                        pass
                    # Try to get another connection from pool
                    try:
                        conn = self._connection_pool.getconn()
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.close()
                        return conn
                    except Exception as e2:
                        logger.warning(f"⚠️  Could not get valid connection from pool: {e2}, creating direct connection")
                        return psycopg2.connect(self.pgvector_connection_string)
            except Exception as e:
                logger.warning(f"⚠️  Could not get connection from pool: {e}")
                # Fallback to direct connection
                return psycopg2.connect(self.pgvector_connection_string)
        else:
            return psycopg2.connect(self.pgvector_connection_string)
    
    def _return_db_connection(self, conn):
        """Return a database connection to the pool or close it."""
        if not conn:
            return
            
        if self._connection_pool:
            try:
                self._connection_pool.putconn(conn)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                # Connection is bad, close it instead of returning to pool
                logger.warning(f"⚠️  Connection is invalid, closing instead of returning to pool: {e}")
                try:
                    conn.close()
                except:
                    pass
            except Exception as e:
                logger.warning(f"⚠️  Could not return connection to pool: {e}")
                try:
                    conn.close()
                except:
                    pass
        else:
            try:
                conn.close()
            except:
                pass
    
    def _explain_analyze_query(self, query: str, params: tuple):
        """Run EXPLAIN ANALYZE on a query for debugging and optimization."""
        # EXPLAIN ANALYZE disabled - too verbose and expensive
        return
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Create EXPLAIN ANALYZE query
            explain_query = f"EXPLAIN ANALYZE {query}"
            
            logger.info("🔍 DEBUG MODE: Running EXPLAIN ANALYZE")
            logger.info(f"📝 Query: {query}")
            # Don't log full parameter arrays (too verbose) - just log summary
            if params and len(params) > 0:
                param_summary = f"({len(params)} params, first param length: {len(str(params[0])) if params[0] else 0})"
                logger.info(f"📊 Parameters: {param_summary}")
            else:
                logger.info(f"📊 Parameters: (none)")
            
            cursor.execute(explain_query, params)
            result = cursor.fetchall()
            
            logger.info("📊 EXPLAIN ANALYZE Results:")
            logger.info("=" * 80)
            for row in result:
                logger.info(row[0])
            logger.info("=" * 80)
            
            cursor.close()
            self._return_db_connection(conn)
            
        except Exception as e:
            logger.error(f"❌ Failed to run EXPLAIN ANALYZE: {e}")
    
    async def _explain_analyze_query_async(self, query: str, params: tuple):
        """Run EXPLAIN ANALYZE on a query for debugging and optimization (async version)."""
        # EXPLAIN ANALYZE disabled - too verbose and expensive
        return
        
        try:
            if not self.pgvector_pool:
                await self._initialize_async_pools()
            
            async with self.pgvector_pool.acquire() as conn:
                # Create EXPLAIN ANALYZE query
                explain_query = f"EXPLAIN ANALYZE {query}"
                
                rag_logger.info("🔍 DEBUG MODE: Running EXPLAIN ANALYZE (async)")
                rag_logger.info(f"📝 Query: {query}")
                # Don't log full parameter arrays (too verbose) - just log summary
                if params and len(params) > 0:
                    param_summary = f"({len(params)} params, first param length: {len(str(params[0])) if params[0] else 0})"
                    rag_logger.info(f"📊 Parameters: {param_summary}")
                else:
                    rag_logger.info(f"📊 Parameters: (none)")
                
                result = await conn.fetch(explain_query, *params)
                
                rag_logger.info("📊 EXPLAIN ANALYZE Results:")
                rag_logger.info("=" * 80)
                for row in result:
                    rag_logger.info(row['QUERY PLAN'])
                rag_logger.info("=" * 80)
                
        except Exception as e:
            rag_logger.error(f"❌ Failed to run EXPLAIN ANALYZE (async): {e}")
    
    def _ensure_pgvector_extension(self):
        """Ensure pgvector extension is enabled in the pgvector database."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Enable pgvector extension if not already enabled
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            
            cursor.close()
            self._return_db_connection(conn)
            logger.info("✅ pgvector extension enabled")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not enable pgvector extension: {e}")
            logger.info("Make sure pgvector is installed on your PostgreSQL server")

    async def _initialize_async_pools(self):
        """Initialize async database connection pools for parallel processing."""
        try:
            # Initialize pgvector pool for parallel searches
            self.pgvector_pool = await asyncpg.create_pool(
                self.pgvector_connection_string,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            rag_logger.info("✅ Async pgvector pool initialized")
            
            # Initialize main database pool if needed
            if self.postgres_connection_string != self.pgvector_connection_string:
                self.db_pool = await asyncpg.create_pool(
                    self.postgres_connection_string,
                    min_size=2,
                    max_size=10,
                    command_timeout=30
                )
                rag_logger.info("✅ Async main database pool initialized")
            else:
                self.db_pool = self.pgvector_pool
                rag_logger.info("✅ Using pgvector pool for main database")
                
        except Exception as e:
            rag_logger.error(f"❌ Failed to initialize async pools: {e}")
            raise
    
    async def _close_async_pools(self):
        """Close async database connection pools."""
        try:
            if self.pgvector_pool:
                await self.pgvector_pool.close()
                rag_logger.info("✅ pgvector pool closed")
            if self.db_pool and self.db_pool != self.pgvector_pool:
                await self.db_pool.close()
                rag_logger.info("✅ main database pool closed")
        except Exception as e:
            rag_logger.error(f"❌ Error closing async pools: {e}")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: VECTOR SEARCH OPERATIONS (Async)
    # ═════════════════════════════════════════════════════════════════════

    async def _search_postgres_with_ticker_async(self, query_embedding: np.ndarray, ticker: str, target_quarter: str = None) -> List[Dict[str, Any]]:
        """Async version of PostgreSQL search with ticker filtering."""
        try:
            ticker = ticker.upper()  # Normalize ticker case to match DB storage
            rag_logger.info(f"🔍 Starting async PostgreSQL search for ticker: {ticker}, quarter: {target_quarter}")

            if not self.pgvector_pool:
                await self._initialize_async_pools()

            async with self.pgvector_pool.acquire() as conn:
                # Build query with optional quarter filtering
                if target_quarter:
                    year, quarter = target_quarter.split('_')
                    quarter_num = quarter[1:]  # Remove 'q' prefix

                    query = """
                    SELECT chunk_text, metadata, year, quarter, ticker, 1 - (embedding <=> $1::vector) as similarity, chunk_index
                    FROM transcript_chunks
                    WHERE UPPER(ticker) = $2 AND year = $3 AND quarter = $4
                    ORDER BY embedding <=> $1::vector
                    LIMIT $5
                    """

                    params = (
                        query_embedding.flatten().tolist(),
                        ticker,
                        int(year),
                        int(quarter_num),
                        self.config.get("chunks_per_quarter")
                    )
                    
                    # Run EXPLAIN ANALYZE in debug mode
                    await self._explain_analyze_query_async(query, params)
                    
                    rows = await conn.fetch(query, *params)
                else:
                    query = """
                    SELECT chunk_text, metadata, year, quarter, ticker, 1 - (embedding <=> $1::vector) as similarity, chunk_index
                    FROM transcript_chunks
                    WHERE UPPER(ticker) = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """

                    params = (
                        query_embedding.flatten().tolist(),
                        ticker,
                        self.config.get("chunks_per_quarter")
                    )

                    # Run EXPLAIN ANALYZE in debug mode
                    await self._explain_analyze_query_async(query, params)

                    rows = await conn.fetch(query, *params)

                rag_logger.info(f"✅ Async PostgreSQL returned {len(rows)} results for ticker {ticker}")
                
                # Convert to expected format
                chunks = []
                threshold = self.config.get("similarity_threshold")
                
                for row in rows:
                    similarity = row['similarity']
                    if similarity >= threshold:
                        # Handle metadata
                        metadata = row['metadata']
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        elif metadata is None:
                            metadata = {}
                        
                        chunk = {
                            'chunk_text': row['chunk_text'],
                            'similarity': float(similarity),
                            'distance': float(1 - similarity),
                            'metadata': metadata,
                            'citation': row['chunk_index'],
                            'year': row['year'],
                            'quarter': row['quarter'],
                            'ticker': row['ticker']
                        }
                        chunks.append(chunk)
                
                rag_logger.info(f"🎯 Async PostgreSQL final results: {len(chunks)} chunks above threshold {threshold}")
                return chunks
                
        except Exception as e:
            rag_logger.error(f"❌ Async PostgreSQL search failed: {e}")
            return []

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 4: VECTOR SEARCH OPERATIONS (Sync)
    # ═════════════════════════════════════════════════════════════════════

    def _search_postgres_with_ticker(self, query_embedding: np.ndarray, ticker: str, target_quarter: str = None) -> List[Dict[str, Any]]:
        """Search using PostgreSQL with pgvector, ticker filtering, and optional quarter filtering."""
        try:
            ticker = ticker.upper()  # Normalize ticker case to match DB storage
            search_start = time.time()
            rag_logger.info(f"🔍 Starting PostgreSQL search for ticker: {ticker}, quarter: {target_quarter}")
            rag_logger.info(f"📊 Query embedding shape: {query_embedding.shape}")

            # Time connection acquisition
            conn_start = time.time()
            rag_logger.info(f"🔗 Getting connection from pool...")
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            conn_time = time.time() - conn_start
            rag_logger.info(f"✅ Connected to PostgreSQL database ({conn_time:.3f}s)")

            # Build query with optional quarter filtering
            # Handle "multiple" as None (defensive check - should be handled upstream)
            if target_quarter == "multiple":
                rag_logger.warning(f"⚠️ Received target_quarter='multiple' - treating as None (search all quarters)")
                target_quarter = None

            if target_quarter:
                # Extract year and quarter from target_quarter (e.g., "2025_q1" -> year=2025, quarter=1)
                if '_' not in target_quarter:
                    rag_logger.error(f"❌ Invalid target_quarter format: {target_quarter} (expected format: YYYY_qN)")
                    # Fallback to searching without quarter filter
                    target_quarter = None
                else:
                    year, quarter = target_quarter.split('_')
                    quarter_num = quarter[1:]  # Remove 'q' prefix

                    query = """
                    SELECT chunk_text, metadata, year, quarter, ticker, 1 - (embedding <=> %s::vector) as similarity, chunk_index
                    FROM transcript_chunks
                    WHERE UPPER(ticker) = %s AND year = %s AND quarter = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """

                    rag_logger.info(f"📝 Executing PostgreSQL query for ticker: {ticker}, year: {year}, quarter: {quarter_num}")

                    # Run EXPLAIN ANALYZE in debug mode
                    params = (query_embedding.flatten().tolist(), ticker, int(year), int(quarter_num), query_embedding.flatten().tolist(), self.config.get("chunks_per_quarter"))
                    self._explain_analyze_query(query, params)

                    # Time query execution
                    query_start = time.time()
                    cursor.execute(query, params)
                    query_time = time.time() - query_start
                    rag_logger.info(f"⏱️  Query executed in {query_time:.3f}s")

            if not target_quarter or target_quarter == "multiple":
                # Search without quarter filtering
                query = """
                SELECT chunk_text, metadata, year, quarter, ticker, 1 - (embedding <=> %s::vector) as similarity, chunk_index
                FROM transcript_chunks
                WHERE UPPER(ticker) = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                
                rag_logger.info(f"📝 Executing PostgreSQL query for ticker: {ticker} (no quarter filter)")
                rag_logger.info(f"🔍 Query: {query.strip()}")
                rag_logger.info(f"📊 Parameters: ticker={ticker}, chunks_per_quarter={self.config.get('chunks_per_quarter')}")
                
                # Run EXPLAIN ANALYZE in debug mode
                params = (query_embedding.flatten().tolist(), ticker, query_embedding.flatten().tolist(), self.config.get("chunks_per_quarter"))
                self._explain_analyze_query(query, params)

                # ✅ FIX: Time query execution (was missing!)
                query_start = time.time()
                cursor.execute(query, params)
                query_time = time.time() - query_start
                rag_logger.info(f"⏱️  Query executed in {query_time:.3f}s")
            
            # Time fetch operation
            fetch_start = time.time()
            results = cursor.fetchall()
            fetch_time = time.time() - fetch_start
            rag_logger.info(f"⏱️  Fetch completed in {fetch_time:.3f}s, returned {len(results)} results")
            
            if results:
                rag_logger.info(f"📊 Sample result metadata type: {type(results[0]['metadata'])}")
            
            # Time connection return
            return_start = time.time()
            self._return_db_connection(conn)
            return_time = time.time() - return_start
            rag_logger.info(f"🔌 Returned connection to pool ({return_time:.3f}s)")
            
            # Convert to expected format
            chunks = []
            threshold = self.config.get("similarity_threshold")
            rag_logger.info(f"🔍 Filtering PostgreSQL results with similarity threshold: {threshold}")
            
            for i, row in enumerate(results):
                similarity = row['similarity']
                rag_logger.info(f"📝 PostgreSQL result {i+1}: similarity={similarity:.3f}, threshold={threshold}, passes={similarity >= threshold}")
                
                if similarity >= threshold:
                    # Handle metadata - it might already be a dict or a JSON string
                    metadata = row['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}
                    
                    chunk = {
                        'chunk_text': row['chunk_text'],
                        'similarity': float(similarity),
                        'distance': float(1 - similarity),
                        'metadata': metadata,
                        'citation': row['chunk_index'],  # Use chunk_index as citation
                        'year': row['year'],
                        'quarter': row['quarter'],
                        'ticker': row['ticker']
                    }
                    chunks.append(chunk)
                    rag_logger.info(f"✅ Added PostgreSQL chunk {i+1} to results (citation: {row['chunk_index']})")
                else:
                    rag_logger.info(f"❌ PostgreSQL chunk {i+1} below threshold, skipping")
            
            rag_logger.info(f"🎯 PostgreSQL final results: {len(chunks)} chunks above threshold {threshold}")
            
            # Log complete timing summary
            total_search_time = time.time() - search_start
            rag_logger.info("=" * 80)
            rag_logger.info(f"⏱️  PGVECTOR SEARCH TIMING SUMMARY ({ticker})")
            rag_logger.info("=" * 80)
            rag_logger.info(f"🔗 Connection: {conn_time:.3f}s")
            rag_logger.info(f"⚡ Query execution: {query_time:.3f}s")
            rag_logger.info(f"📥 Fetch results: {fetch_time:.3f}s")
            rag_logger.info(f"🔌 Return connection: {return_time:.3f}s")
            rag_logger.info(f"")
            rag_logger.info(f"⏱️  TOTAL: {total_search_time:.3f}s")
            rag_logger.info(f"📊 Results: {len(chunks)} chunks passed threshold, {len(results)} total returned")
            rag_logger.info("=" * 80)
            
            return chunks

        except Exception as e:
            logger.error(f"Failed to search PostgreSQL (ticker={ticker}, quarter={target_quarter}): {e}", exc_info=True)
            rag_logger.error(f"❌ PostgreSQL search EXCEPTION for ticker={ticker}, quarter={target_quarter}: {type(e).__name__}: {e}")
            return []

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 5: GENERAL SEARCH OPERATIONS
    # ═════════════════════════════════════════════════════════════════════

    def _search_postgres_general(self, query_embedding: np.ndarray, max_results: int, target_quarter: str = None) -> List[Dict[str, Any]]:
        """Search using PostgreSQL with pgvector without ticker filtering but with optional quarter filtering."""
        try:
            rag_logger.info(f"🔍 Starting general PostgreSQL search, quarter: {target_quarter}")
            rag_logger.info(f"📊 Query embedding shape: {query_embedding.shape}")
            rag_logger.info(f"🔗 Connecting to PostgreSQL database...")
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            rag_logger.info(f"✅ Connected to PostgreSQL database")
            
            # Build query with optional quarter filtering
            if target_quarter:
                # Extract year and quarter from target_quarter (e.g., "2025_q1" -> year=2025, quarter=1)
                year, quarter = target_quarter.split('_')
                quarter_num = quarter[1:]  # Remove 'q' prefix
                
                query = """
                SELECT chunk_text, metadata, year, quarter, ticker, 1 - (embedding <=> %s::vector) as similarity, chunk_index
                FROM transcript_chunks 
                WHERE year = %s AND quarter = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                
                rag_logger.info(f"📝 Executing general PostgreSQL query for year: {year}, quarter: {quarter_num}")
                rag_logger.info(f"🔍 Query: {query.strip()}")
                rag_logger.info(f"📊 Parameters: year={year}, quarter={quarter_num}, max_results={max_results}")
                
                # Run EXPLAIN ANALYZE in debug mode
                params = (query_embedding.flatten().tolist(), int(year), int(quarter_num), query_embedding.flatten().tolist(), max_results)
                self._explain_analyze_query(query, params)
                
                cursor.execute(query, params)
            else:
                # Search without quarter filtering
                query = """
                SELECT chunk_text, metadata, year, quarter, ticker, 1 - (embedding <=> %s::vector) as similarity, chunk_index
                FROM transcript_chunks 
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                
                rag_logger.info(f"📝 Executing general PostgreSQL query (no quarter filter)")
                rag_logger.info(f"🔍 Query: {query.strip()}")
                rag_logger.info(f"📊 Parameters: max_results={max_results}")
                
                # Run EXPLAIN ANALYZE in debug mode
                params = (query_embedding.flatten().tolist(), query_embedding.flatten().tolist(), max_results)
                self._explain_analyze_query(query, params)
                
                cursor.execute(query, params)
            
            results = cursor.fetchall()
            
            rag_logger.info(f"✅ PostgreSQL returned {len(results)} results")
            if results:
                rag_logger.info(f"📊 Sample result metadata type: {type(results[0]['metadata'])}")
                rag_logger.info(f"📊 Sample result metadata preview: {str(results[0]['metadata'])[:100]}...")
            
            self._return_db_connection(conn)
            rag_logger.info(f"🔌 Closed PostgreSQL connection")
            
            # Convert to expected format
            chunks = []
            threshold = self.config.get("similarity_threshold")
            rag_logger.info(f"🔍 Filtering PostgreSQL results with similarity threshold: {threshold}")
            
            for i, row in enumerate(results):
                similarity = row['similarity']
                rag_logger.info(f"📝 PostgreSQL result {i+1}: similarity={similarity:.3f}, threshold={threshold}, passes={similarity >= threshold}")
                
                if similarity >= threshold:
                    # Handle metadata - it might already be a dict or a JSON string
                    metadata = row['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}
                    
                    chunk = {
                        'chunk_text': row['chunk_text'],
                        'similarity': float(similarity),
                        'distance': float(1 - similarity),
                        'metadata': metadata,
                        'citation': row['chunk_index'],  # Use chunk_index as citation
                        'year': row['year'],
                        'quarter': row['quarter'],
                        'ticker': row['ticker']
                    }
                    chunks.append(chunk)
                    rag_logger.info(f"✅ Added PostgreSQL chunk {i+1} to results (citation: {row['chunk_index']})")
                else:
                    rag_logger.info(f"❌ PostgreSQL chunk {i+1} below threshold, skipping")
            
            rag_logger.info(f"🎯 PostgreSQL general search final results: {len(chunks)} chunks above threshold {threshold}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to search PostgreSQL (general search): {e}", exc_info=True)
            rag_logger.error(f"❌ PostgreSQL general search EXCEPTION: {type(e).__name__}: {e}")
            return []
    
    # ═════════════════════════════════════════════════════════════════════
    # STAGE 7: 10-K FILINGS SEARCH OPERATIONS
    # ═════════════════════════════════════════════════════════════════════

    async def get_all_tables_for_ticker_async(self, ticker: str, fiscal_year: int = None) -> List[Dict[str, Any]]:
        """
        Get all tables for a ticker from the 10k_tables table.

        Args:
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year filter

        Returns:
            List of table dictionaries with metadata
        """
        try:
            ticker = ticker.upper()  # Normalize ticker case
            if not self.pgvector_pool:
                await self._initialize_async_pools()

            async with self.pgvector_pool.acquire() as conn:
                if fiscal_year:
                    query = """
                    SELECT table_id, ticker, fiscal_year, content, table_data,
                           path_string, sec_section, sec_section_title,
                           is_financial_statement, statement_type, priority
                    FROM ten_k_tables
                    WHERE UPPER(ticker) = $1 AND fiscal_year = $2
                    ORDER BY priority DESC, sec_section
                    """
                    params = (ticker, fiscal_year)
                else:
                    # Get most recent fiscal year
                    query = """
                    SELECT table_id, ticker, fiscal_year, content, table_data,
                           path_string, sec_section, sec_section_title,
                           is_financial_statement, statement_type, priority
                    FROM ten_k_tables
                    WHERE UPPER(ticker) = $1
                      AND fiscal_year = (SELECT MAX(fiscal_year) FROM ten_k_tables WHERE UPPER(ticker) = $1)
                    ORDER BY priority DESC, sec_section
                    """
                    params = (ticker,)

                rows = await conn.fetch(query, *params)

                tables = []
                for row in rows:
                    table = {
                        'table_id': row['table_id'],
                        'ticker': row['ticker'],
                        'fiscal_year': row['fiscal_year'],
                        'content': row['content'],
                        'table_data': row['table_data'],
                        'path_string': row['path_string'],
                        'sec_section': row['sec_section'],
                        'sec_section_title': row['sec_section_title'],
                        'is_financial_statement': row['is_financial_statement'],
                        'statement_type': row['statement_type'],
                        'priority': row['priority']
                    }
                    tables.append(table)

                rag_logger.info(f"📊 Found {len(tables)} tables for {ticker}")
                return tables

        except Exception as e:
            rag_logger.error(f"❌ Failed to get tables for {ticker}: {e}")
            return []

    async def search_10k_filings_async(self, query_embedding: np.ndarray, ticker: str, fiscal_year: int = None) -> List[Dict[str, Any]]:
        """Search 10-K filings using vector similarity (async)."""
        try:
            ticker = ticker.upper()  # Normalize ticker case to match DB storage
            rag_logger.info(f"📄 Starting async 10-K search for ticker: {ticker}, fiscal_year: {fiscal_year}")

            if not self.pgvector_pool:
                await self._initialize_async_pools()

            async with self.pgvector_pool.acquire() as conn:
                # Format embedding as PostgreSQL array string for asyncpg/pgvector
                # asyncpg needs the vector as a string representation that PostgreSQL can parse
                embedding_list = query_embedding.flatten().tolist()
                embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'

                # Build query with optional fiscal year filtering
                if fiscal_year:
                    query = """
                    SELECT chunk_text, metadata, ticker, fiscal_year, chunk_type,
                           sec_section, sec_section_title, path_string, chunk_index,
                           char_offset,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM ten_k_chunks
                    WHERE UPPER(ticker) = $2 AND fiscal_year = $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4
                    """
                    rows = await conn.fetch(
                        query,
                        embedding_str,
                        ticker,
                        fiscal_year,
                        self.config.get("chunks_per_quarter", 15)
                    )
                else:
                    query = """
                    SELECT chunk_text, metadata, ticker, fiscal_year, chunk_type,
                           sec_section, sec_section_title, path_string, chunk_index,
                           char_offset,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM ten_k_chunks
                    WHERE UPPER(ticker) = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """
                    rows = await conn.fetch(
                        query,
                        embedding_str,
                        ticker,
                        self.config.get("chunks_per_quarter", 15)
                    )
                rag_logger.info(f"✅ Async 10-K search returned {len(rows)} results for ticker {ticker}")

                # Convert to expected format
                chunks = []
                threshold = self.config.get("similarity_threshold", 0.3)

                for row in rows:
                    similarity = row['similarity']
                    if similarity >= threshold:
                        metadata = row['metadata']
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        elif metadata is None:
                            metadata = {}

                        chunk = {
                            'chunk_text': row['chunk_text'],
                            'similarity': float(similarity),
                            'distance': float(1 - similarity),
                            'metadata': metadata,
                            'citation': f"10K_{row['ticker']}_FY{row['fiscal_year']}_{row['chunk_index']}",
                            'ticker': row['ticker'],
                            'fiscal_year': row['fiscal_year'],
                            'chunk_type': row['chunk_type'],
                            'sec_section': row['sec_section'],
                            'sec_section_title': row['sec_section_title'],
                            'path_string': row['path_string'],
                            'char_offset': row['char_offset'],
                            'source_type': '10-K'  # Mark as 10-K source
                        }
                        chunks.append(chunk)

                rag_logger.info(f"🎯 Async 10-K final results: {len(chunks)} chunks above threshold {threshold}")
                return chunks

        except Exception as e:
            rag_logger.error(f"❌ Async 10-K search failed: {e}")
            return []

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 7b: BULK CROSS-COMPANY SEARCH OPERATIONS
    # ═════════════════════════════════════════════════════════════════════

    async def bulk_search_transcripts_async(
        self,
        query_embedding: np.ndarray,
        tickers: List[str],
        year: int,
        quarter: int,
        chunks_per_company: int = 5,
        similarity_threshold: float = 0.3,
    ) -> Dict[str, List[Dict]]:
        """Search transcript chunks across many companies in a single query.

        Uses ROW_NUMBER() OVER (PARTITION BY ticker) to get top-K chunks per company.
        Returns Dict[ticker -> List[chunk_dicts]] grouped by company.
        """
        try:
            if not self.pgvector_pool:
                await self._initialize_async_pools()

            embedding_list = query_embedding.flatten().tolist()
            embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'

            async with self.pgvector_pool.acquire() as conn:
                if tickers:
                    upper_tickers = [t.upper() for t in tickers]
                    query = """
                    WITH ranked AS (
                        SELECT chunk_text, ticker, year, quarter, chunk_index, metadata,
                               1 - (embedding <=> $1::vector) as similarity,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY embedding <=> $1::vector) as rn
                        FROM transcript_chunks
                        WHERE year = $2 AND quarter = $3
                          AND UPPER(ticker) = ANY($4::text[])
                    )
                    SELECT * FROM ranked WHERE rn <= $5 AND similarity >= $6
                    ORDER BY similarity DESC
                    """
                    rows = await conn.fetch(
                        query, embedding_str, year, quarter,
                        upper_tickers, chunks_per_company, similarity_threshold,
                    )
                else:
                    query = """
                    WITH ranked AS (
                        SELECT chunk_text, ticker, year, quarter, chunk_index, metadata,
                               1 - (embedding <=> $1::vector) as similarity,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY embedding <=> $1::vector) as rn
                        FROM transcript_chunks
                        WHERE year = $2 AND quarter = $3
                    )
                    SELECT * FROM ranked WHERE rn <= $4 AND similarity >= $5
                    ORDER BY similarity DESC
                    """
                    rows = await conn.fetch(
                        query, embedding_str, year, quarter,
                        chunks_per_company, similarity_threshold,
                    )

                rag_logger.info(f"✅ Bulk transcript search returned {len(rows)} total chunks")

                # Group by ticker
                results: Dict[str, List[Dict]] = {}
                for row in rows:
                    ticker = row['ticker']
                    metadata = row['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}

                    chunk = {
                        'chunk_text': row['chunk_text'],
                        'similarity': float(row['similarity']),
                        'metadata': metadata,
                        'year': row['year'],
                        'quarter': row['quarter'],
                        'ticker': ticker,
                        'chunk_index': row['chunk_index'],
                    }
                    results.setdefault(ticker, []).append(chunk)

                rag_logger.info(f"🎯 Bulk transcript search: {len(results)} companies with matches")
                return results

        except Exception as e:
            rag_logger.error(f"❌ Bulk transcript search failed: {e}")
            return {}

    async def bulk_search_10k_async(
        self,
        query_embedding: np.ndarray,
        tickers: List[str],
        fiscal_year: int,
        chunks_per_company: int = 5,
        similarity_threshold: float = 0.3,
    ) -> Dict[str, List[Dict]]:
        """Search 10-K filing chunks across many companies in a single query.

        Uses ROW_NUMBER() OVER (PARTITION BY ticker) to get top-K chunks per company.
        Returns Dict[ticker -> List[chunk_dicts]] grouped by company.
        """
        try:
            if not self.pgvector_pool:
                await self._initialize_async_pools()

            embedding_list = query_embedding.flatten().tolist()
            embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'

            async with self.pgvector_pool.acquire() as conn:
                if tickers:
                    upper_tickers = [t.upper() for t in tickers]
                    query = """
                    WITH ranked AS (
                        SELECT chunk_text, ticker, fiscal_year, chunk_index, metadata,
                               sec_section, sec_section_title, path_string, chunk_type,
                               char_offset,
                               1 - (embedding <=> $1::vector) as similarity,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY embedding <=> $1::vector) as rn
                        FROM ten_k_chunks
                        WHERE fiscal_year = $2
                          AND UPPER(ticker) = ANY($3::text[])
                    )
                    SELECT * FROM ranked WHERE rn <= $4 AND similarity >= $5
                    ORDER BY similarity DESC
                    """
                    rows = await conn.fetch(
                        query, embedding_str, fiscal_year,
                        upper_tickers, chunks_per_company, similarity_threshold,
                    )
                else:
                    query = """
                    WITH ranked AS (
                        SELECT chunk_text, ticker, fiscal_year, chunk_index, metadata,
                               sec_section, sec_section_title, path_string, chunk_type,
                               char_offset,
                               1 - (embedding <=> $1::vector) as similarity,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY embedding <=> $1::vector) as rn
                        FROM ten_k_chunks
                        WHERE fiscal_year = $2
                    )
                    SELECT * FROM ranked WHERE rn <= $3 AND similarity >= $4
                    ORDER BY similarity DESC
                    """
                    rows = await conn.fetch(
                        query, embedding_str, fiscal_year,
                        chunks_per_company, similarity_threshold,
                    )

                rag_logger.info(f"✅ Bulk 10-K search returned {len(rows)} total chunks")

                # Group by ticker
                results: Dict[str, List[Dict]] = {}
                for row in rows:
                    ticker = row['ticker']
                    metadata = row['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}

                    chunk = {
                        'chunk_text': row['chunk_text'],
                        'similarity': float(row['similarity']),
                        'metadata': metadata,
                        'ticker': ticker,
                        'fiscal_year': row['fiscal_year'],
                        'chunk_index': row['chunk_index'],
                        'sec_section': row['sec_section'],
                        'sec_section_title': row['sec_section_title'],
                        'path_string': row['path_string'],
                        'chunk_type': row['chunk_type'],
                        'char_offset': row['char_offset'],
                    }
                    results.setdefault(ticker, []).append(chunk)

                rag_logger.info(f"🎯 Bulk 10-K search: {len(results)} companies with matches")
                return results

        except Exception as e:
            rag_logger.error(f"❌ Bulk 10-K search failed: {e}")
            return {}

    def search_10k_filings(self, query_embedding: np.ndarray, ticker: str, fiscal_year: int = None,
                           selected_sections: List[str] = None) -> List[Dict[str, Any]]:
        """Search 10-K filings using vector similarity with optional section filtering (sync)."""
        try:
            ticker = ticker.upper()  # Normalize ticker case to match DB storage
            rag_logger.info(f"📄 Starting 10-K search for ticker: {ticker}, fiscal_year: {fiscal_year}")
            if selected_sections:
                rag_logger.info(f"   🎯 Restricting to sections: {selected_sections}")

            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build query with optional fiscal year and section filtering
            base_select = """
                SELECT chunk_text, metadata, ticker, fiscal_year, chunk_type,
                       sec_section, sec_section_title, path_string, chunk_index,
                       char_offset,
                       1 - (embedding <=> %s::vector) as similarity
                FROM ten_k_chunks
            """

            # Build WHERE clause dynamically
            conditions = ["UPPER(ticker) = %s"]
            params = [query_embedding.flatten().tolist(), ticker]

            if fiscal_year:
                conditions.append("fiscal_year = %s")
                params.append(fiscal_year)

            if selected_sections:
                # Add section filtering
                section_placeholders = ','.join(['%s'] * len(selected_sections))
                conditions.append(f"sec_section IN ({section_placeholders})")
                params.extend(selected_sections)

            where_clause = " AND ".join(conditions)

            # Add ORDER BY and LIMIT
            # Increase limit when filtering by sections to get more relevant chunks
            chunk_limit = self.config.get("chunks_per_quarter", 15)
            if selected_sections:
                chunk_limit = chunk_limit * 2  # Get more chunks when filtering by section

            query = f"""
                {base_select}
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params.append(query_embedding.flatten().tolist())
            params.append(chunk_limit)

            cursor.execute(query, tuple(params))
            results = cursor.fetchall()

            self._return_db_connection(conn)

            rag_logger.info(f"✅ 10-K search returned {len(results)} results for ticker {ticker}")

            # Convert to expected format
            chunks = []
            threshold = self.config.get("similarity_threshold", 0.3)

            for row in results:
                similarity = row['similarity']
                if similarity >= threshold:
                    metadata = row['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}

                    chunk = {
                        'chunk_text': row['chunk_text'],
                        'similarity': float(similarity),
                        'distance': float(1 - similarity),
                        'metadata': metadata,
                        'citation': f"10K_{row['ticker']}_FY{row['fiscal_year']}_{row['chunk_index']}",
                        'ticker': row['ticker'],
                        'fiscal_year': row['fiscal_year'],
                        'chunk_type': row['chunk_type'],
                        'sec_section': row['sec_section'],
                        'sec_section_title': row['sec_section_title'],
                        'path_string': row['path_string'],
                        'char_offset': row['char_offset'],
                        'source_type': '10-K'  # Mark as 10-K source
                    }
                    chunks.append(chunk)

            rag_logger.info(f"🎯 10-K final results: {len(chunks)} chunks above threshold {threshold}")
            return chunks

        except Exception as e:
            rag_logger.error(f"❌ 10-K search failed: {e}")
            return []

    def __del__(self):
        """Cleanup connection pool when DatabaseManager is destroyed."""
        try:
            if hasattr(self, '_connection_pool') and self._connection_pool:
                self._connection_pool.closeall()
                logger.info("✅ Database connection pool closed")
        except Exception as e:
            logger.warning(f"⚠️  Error closing connection pool: {e}")
