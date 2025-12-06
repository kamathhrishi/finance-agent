#!/usr/bin/env python3
"""
10-K SEC Filings Database Ingestion with Sophisticated Processing

This script uses the full sophisticated data_loading.py processing pipeline
and stores results in PostgreSQL database instead of disk cache.

Features:
- Hierarchical content extraction with section identification
- Contextual chunking preserving document structure
- Table extraction and financial statement identification
- Cross-encoder for reranking
- TF-IDF for hybrid search
- Stores in PostgreSQL with pgvector
- Automatically skips already-processed ticker-year combinations

Usage:
    python ingest_10k_to_database.py --ticker AAPL
    python ingest_10k_to_database.py --tickers AAPL MSFT GOOGL
    python ingest_10k_to_database.py --all-financebench
"""

import argparse
import os
import sys
import json
import logging
import time
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import shutil
import tempfile
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial
import signal

# Import finqual for clean financial statements
try:
    import finqual as fq
    FINQUAL_AVAILABLE = True
except ImportError:
    FINQUAL_AVAILABLE = False
    logging.warning("⚠️ finqual not available - will skip fetching clean financial statements")

# Import the sophisticated data loading module
sys.path.insert(0, str(Path(__file__).parent))
from ingest_10k_filings_full import (
    DataProcessor,
    download_and_extract_10k,
    create_embeddings_for_filing_data,
    FINANCEBENCH_COMPANIES
)

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('10k_db_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatabaseIntegration:
    """Handles storing 10-K data in PostgreSQL database"""

    def __init__(self, db_url: str = None):
        # Use PG_VECTOR database (same as RAG system) which has pgvector extension
        # Priority: provided db_url > PG_VECTOR > DATABASE_URL
        if db_url:
            self.db_url = db_url
        else:
            self.db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")

        if not self.db_url:
            raise ValueError("PG_VECTOR or DATABASE_URL environment variable not set")
        self._connection = None

    def get_connection(self):
        """Get or create a database connection with proper error handling"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                self.db_url,
                connect_timeout=30,
                options='-c statement_timeout=300000'  # 5 minute query timeout
            )
        return self._connection

    def close_connection(self):
        """Explicitly close the database connection"""
        if self._connection and not self._connection.closed:
            try:
                self._connection.close()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
            finally:
                self._connection = None

    def ensure_tables(self):
        """Ensure database tables exist with proper schema"""
        logger.info("🔍 Ensuring database tables exist...")
        logger.info(f"   Database URL: {self.db_url[:50]}...")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # First, ensure pgvector extension is enabled
            logger.info("   Enabling pgvector extension...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logger.info("   ✅ pgvector extension enabled")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not enable pgvector extension: {e}")
                # Continue anyway - extension might already be enabled

            # Create ten_k_chunks table with full metadata support
            # Using pgvector database (PG_VECTOR) which has vector extension
            logger.info("   Creating ten_k_chunks table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ten_k_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    ticker VARCHAR(10),
                    fiscal_year INTEGER,
                    filing_type VARCHAR(10) DEFAULT '10-K',
                    chunk_index INTEGER,
                    citation VARCHAR(200),
                    chunk_type VARCHAR(50),
                    sec_section VARCHAR(50),
                    sec_section_title VARCHAR(200),
                    path_string TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("   ✅ ten_k_chunks table created/verified")

            # Check if table has data before creating indexes
            cursor.execute("SELECT COUNT(*) FROM ten_k_chunks")
            row_count = cursor.fetchone()[0]

            if row_count == 0:
                # Table is empty - create all indexes
                logger.info("   Creating indexes (table is empty)...")

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ten_k_ticker_year
                    ON ten_k_chunks(ticker, fiscal_year);
                """)
                conn.commit()
                logger.info("   ✅ Index on (ticker, fiscal_year)")

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ten_k_section
                    ON ten_k_chunks(sec_section);
                """)
                conn.commit()
                logger.info("   ✅ Index on sec_section")

                # Create vector index
                logger.info("   Creating vector index (this may take a minute for large tables)...")
                try:
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_ten_k_embedding
                        ON ten_k_chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    conn.commit()
                    logger.info("   ✅ Vector index created")
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not create vector index: {e}")
                    conn.rollback()
            else:
                # Table has data - skip all index creation (they should already exist)
                logger.info(f"   ⚠️  Table has {row_count:,} rows - skipping ALL index creation")
                logger.info(f"   💡 Indexes should already exist from initial setup")
                logger.info(f"   💡 If indexes are missing, create them manually or clear the table and re-run")

            # Create ten_k_tables table for storing extracted tables
            logger.info("   Creating ten_k_tables table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ten_k_tables (
                    id SERIAL PRIMARY KEY,
                    table_id VARCHAR(100) UNIQUE NOT NULL,
                    ticker VARCHAR(10),
                    fiscal_year INTEGER,
                    content TEXT NOT NULL,
                    table_data JSONB,
                    path_string TEXT,
                    sec_section VARCHAR(50),
                    sec_section_title VARCHAR(200),
                    is_financial_statement BOOLEAN,
                    statement_type VARCHAR(50),
                    priority VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("   ✅ ten_k_tables table created/verified")

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ten_k_tables_ticker_year
                ON ten_k_tables(ticker, fiscal_year);
            """)
            conn.commit()

            logger.info("✅ Database tables ensured successfully")

            # Verify tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('ten_k_chunks', 'ten_k_tables')
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"   Verified tables: {existing_tables}")

        except Exception as e:
            logger.error(f"❌ Failed to ensure tables: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def store_chunks(self, ticker: str, fiscal_year: int, processor: DataProcessor):
        """Store processed chunks in database"""
        chunks = processor.get_chunks()
        embeddings = processor.get_embeddings()

        if not chunks or embeddings is None:
            logger.warning(f"⚠️ No chunks or embeddings to store for {ticker} FY{fiscal_year}")
            return 0

        logger.info(f"💾 Storing {len(chunks)} chunks for {ticker} FY{fiscal_year}")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Delete existing chunks for this ticker and fiscal year
            cursor.execute("""
                DELETE FROM ten_k_chunks
                WHERE ticker = %s AND fiscal_year = %s
            """, (ticker, fiscal_year))
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"  🗑️ Deleted {deleted_count} existing chunks")

            # Prepare chunk data
            chunk_data = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    'ticker': ticker,
                    'fiscal_year': fiscal_year,
                    'chunk_index': idx,
                    'type': chunk.get('type'),
                    'level': chunk.get('level'),
                    'path': chunk.get('path', []),
                    'sec_section': chunk.get('sec_section'),
                    'sec_section_title': chunk.get('sec_section_title')
                }

                citation = f"{ticker}_10K_FY{fiscal_year}_{idx}"

                chunk_data.append((
                    chunk['content'],
                    embedding.tolist(),  # Store as array for vector type
                    json.dumps(metadata),
                    ticker,
                    fiscal_year,
                    '10-K',
                    idx,
                    citation,
                    chunk.get('type', 'unknown'),
                    chunk.get('sec_section', 'unknown'),
                    chunk.get('sec_section_title', 'Unknown'),
                    chunk.get('path_string', '')
                ))

            # Batch insert
            execute_values(
                cursor,
                """
                INSERT INTO ten_k_chunks
                (chunk_text, embedding, metadata, ticker, fiscal_year, filing_type,
                 chunk_index, citation, chunk_type, sec_section, sec_section_title, path_string)
                VALUES %s
                """,
                chunk_data,
                template="(%s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )

            conn.commit()
            logger.info(f"  ✅ Stored {len(chunk_data)} chunks")
            return len(chunk_data)

        except Exception as e:
            logger.error(f"  ❌ Failed to store chunks: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()

    def store_tables(self, ticker: str, fiscal_year: int, processor: DataProcessor):
        """Store extracted tables in database"""
        tables = processor.get_tables()

        if not tables:
            logger.warning(f"⚠️ No tables to store for {ticker} FY{fiscal_year}")
            return 0

        logger.info(f"📊 Storing {len(tables)} tables for {ticker} FY{fiscal_year}")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Delete existing tables for this ticker and fiscal year
            cursor.execute("""
                DELETE FROM ten_k_tables
                WHERE ticker = %s AND fiscal_year = %s
            """, (ticker, fiscal_year))

            # Prepare table data
            table_data = []
            for table_id, table_info in tables.items():
                # Make table_id unique across tickers by prepending ticker_fiscalyear
                unique_table_id = f"{ticker}_FY{fiscal_year}_{table_id}"
                table_data.append((
                    unique_table_id,
                    ticker,
                    fiscal_year,
                    table_info['content'],
                    json.dumps(table_info.get('table_data')),
                    table_info.get('path_string', ''),
                    table_info.get('sec_section', 'unknown'),
                    table_info.get('sec_section_title', 'Unknown'),
                    table_info.get('is_financial_statement', False),
                    table_info.get('statement_type'),
                    table_info.get('priority', 'NORMAL')
                ))

            # Batch insert
            execute_values(
                cursor,
                """
                INSERT INTO ten_k_tables
                (table_id, ticker, fiscal_year, content, table_data, path_string,
                 sec_section, sec_section_title, is_financial_statement, statement_type, priority)
                VALUES %s
                """,
                table_data,
                template="(%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)"
            )

            conn.commit()

            # Count financial statements
            financial_count = sum(1 for t in tables.values() if t.get('is_financial_statement'))
            logger.info(f"  ✅ Stored {len(table_data)} tables ({financial_count} financial statements)")
            return len(table_data)

        except Exception as e:
            logger.error(f"  ❌ Failed to store tables: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()

    def check_data_exists(self, ticker: str, fiscal_year: int) -> bool:
        """
        Check if data already exists for this ticker-year combination

        Returns True if data exists, False otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check if chunks exist for this ticker-year
            cursor.execute("""
                SELECT COUNT(*) FROM ten_k_chunks
                WHERE ticker = %s AND fiscal_year = %s
            """, (ticker, fiscal_year))

            count = cursor.fetchone()[0]
            return count > 0

        except Exception as e:
            logger.debug(f"Error checking if data exists: {e}")
            return False
        finally:
            if cursor:
                cursor.close()

    def fetch_and_store_finqual_statements(self, ticker: str, fiscal_year: int) -> int:
        """
        Fetch clean financial statements from Finqual and store in database

        Returns number of statements stored
        """
        if not FINQUAL_AVAILABLE:
            return 0

        logger.info(f"📊 Fetching Finqual statements for {ticker} FY{fiscal_year}...")

        statements_stored = 0
        # Map Finqual API types to standard statement type names
        statement_configs = [
            ('income', 'Income Statement', 'income_statement'),
            ('balance', 'Balance Sheet', 'balance_sheet'),
            ('cash', 'Cash Flow Statement', 'cash_flow')
        ]

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            for api_type, stmt_name, stmt_type in statement_configs:
                try:
                    # Fetch from Finqual
                    if api_type == 'income':
                        df = fq.Finqual(ticker).income_stmt(fiscal_year)
                    elif api_type == 'balance':
                        df = fq.Finqual(ticker).balance_sheet(fiscal_year)
                    elif api_type == 'cash':
                        df = fq.Finqual(ticker).cash_flow(fiscal_year)
                    else:
                        continue

                    if df is None or df.empty:
                        logger.debug(f"  ⚠️ No {stmt_name} data from Finqual")
                        continue

                    # Convert to string representation
                    content = df.to_string()

                    # Create table ID
                    table_id = f"{ticker}_FY{fiscal_year}_FINQUAL_{api_type.upper()}"

                    # Store in database with CRITICAL priority
                    cursor.execute("""
                        INSERT INTO ten_k_tables
                        (table_id, ticker, fiscal_year, content, table_data, path_string,
                         sec_section, sec_section_title, is_financial_statement, statement_type, priority)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (table_id) DO UPDATE
                        SET content = EXCLUDED.content,
                            table_data = EXCLUDED.table_data
                    """, (
                        table_id,
                        ticker,
                        fiscal_year,
                        content,
                        json.dumps({'source': 'finqual', 'clean': True}),
                        f"Finqual > {stmt_name}",
                        'item_8',
                        'Item 8 - Financial Statements',
                        True,
                        stmt_type,  # Use standard statement type names
                        'CRITICAL'
                    ))

                    statements_stored += 1
                    logger.info(f"  ✅ Stored {stmt_name} from Finqual")

                except Exception as e:
                    logger.debug(f"  ⚠️ Could not fetch {stmt_name} from Finqual: {e}")
                    continue

            conn.commit()

            if statements_stored > 0:
                logger.info(f"  ✅ Stored {statements_stored} Finqual statements for {ticker} FY{fiscal_year}")

            return statements_stored

        except Exception as e:
            logger.error(f"  ❌ Failed to store Finqual statements: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()


def cleanup_temp_files():
    """Clean up temporary files created by datamule"""
    try:
        # Clean up temp directory
        temp_dir = Path(tempfile.gettempdir())
        datamule_patterns = ['datamule*', 'portfolio*', 'sec_*']

        cleaned_count = 0
        for pattern in datamule_patterns:
            for temp_file in temp_dir.glob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    logger.debug(f"Could not delete temp file {temp_file}: {e}")

        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} temporary files")

        # Force garbage collection
        gc.collect()

    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Process timeout")


def ingest_ticker_worker(ticker: str, lookback_years: int, db_url: str, timeout_minutes: int = 30, skip_finqual: bool = False):
    """
    Worker function for parallel processing - must be picklable.
    Each worker creates its own database connection.

    IMPORTANT: This function includes aggressive garbage collection to prevent
    memory accumulation, especially from finqual and datamule libraries.

    Args:
        ticker: Stock ticker symbol
        lookback_years: Number of years to look back
        db_url: Database connection URL
        timeout_minutes: Maximum time allowed for processing (default: 30 minutes)
        skip_finqual: Skip fetching Finqual statements (default: False)
    """
    # Set up timeout handler (Unix-based systems only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)

    db_integration = None
    try:
        # Create database integration for this worker with explicit db_url
        # This ensures workers use the same database as the main process
        db_integration = DatabaseIntegration(db_url=db_url)

        # Call the main ingestion function
        result = ingest_ticker(ticker, lookback_years, db_integration, skip_finqual=skip_finqual)

        # Clean up database connection
        db_integration.close_connection()

        # AGGRESSIVE CLEANUP to prevent memory leaks
        # This is critical for finqual which can accumulate memory
        cleanup_temp_files()

        # Force garbage collection multiple times
        gc.collect()
        gc.collect()
        gc.collect()

        # Try to clear any remaining references
        db_integration = None

        # Cancel the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        return result

    except TimeoutError:
        logger.error(f"⏱️ Worker timeout for {ticker} (exceeded {timeout_minutes} minutes)")
        return {'ticker': ticker, 'error': f'Timeout after {timeout_minutes} minutes'}

    except Exception as e:
        logger.error(f"❌ Worker failed for {ticker}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {'ticker': ticker, 'error': str(e)}

    finally:
        # Cancel the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        # Ensure cleanup happens even on failure
        try:
            if db_integration:
                db_integration.close_connection()
            cleanup_temp_files()
            gc.collect()
        except:
            pass


def ingest_ticker(ticker: str, lookback_years: int, db_integration: DatabaseIntegration, skip_finqual: bool = False):
    """Ingest 10-K filings for a single ticker using sophisticated processing"""
    logger.info(f"📈 Processing {ticker}...")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    # Download and extract 10-K with sophisticated processing
    start_year = start_date.year - 1  # Buffer for fiscal year mismatch
    end_year = end_date.year + 1

    # Calculate minimum fiscal year to process (based on lookback period)
    min_fiscal_year = start_date.year
    logger.info(f"📅 Will only process fiscal years >= {min_fiscal_year}")

    logger.info(f"📥 Downloading 10-K filings from {start_year} to {end_year}")
    filings_by_year = download_and_extract_10k(ticker, start_year, end_year)

    if not filings_by_year:
        logger.warning(f"⚠️ No 10-K filings found for {ticker}")
        return {'ticker': ticker, 'filings_processed': 0, 'chunks_stored': 0, 'tables_stored': 0, 'finqual_stored': 0}

    # Filter to only recent fiscal years based on lookback period
    filtered_filings = {fy: data for fy, data in filings_by_year.items() if fy >= min_fiscal_year}

    if not filtered_filings:
        logger.warning(f"⚠️ No recent 10-K filings found for {ticker} (need FY >= {min_fiscal_year})")
        return {'ticker': ticker, 'filings_processed': 0, 'chunks_stored': 0, 'tables_stored': 0, 'finqual_stored': 0}

    skipped_years = set(filings_by_year.keys()) - set(filtered_filings.keys())
    if skipped_years:
        logger.info(f"⏭️  Skipping older fiscal years: {sorted(skipped_years)} (older than {lookback_years} year(s))")

    logger.info(f"📦 Found {len(filtered_filings)} fiscal year(s) of recent data: {sorted(filtered_filings.keys())}")

    # Process each fiscal year
    total_chunks = 0
    total_tables = 0
    total_finqual = 0
    filings_processed = 0
    filings_skipped = 0

    for fiscal_year, filing_data in filtered_filings.items():
        logger.info(f"\n🔧 Processing FY{fiscal_year}...")

        # Check if data already exists for this ticker-year
        if db_integration.check_data_exists(ticker, fiscal_year):
            logger.info(f"⏭️  Skipping {ticker} FY{fiscal_year} - already processed")
            filings_skipped += 1
            continue

        try:
            # Create processor and prepare chunks with hierarchical structure
            processor = DataProcessor()

            # Use hierarchical chunks (the sophisticated method)
            processor.prepare_chunks(filing_data, use_hierarchical=True, exclude_titles=True)

            # Create embeddings
            processor.create_embeddings()

            # Identify financial statements
            processor.identify_financial_statement_tables()

            # Store in database
            chunks_stored = db_integration.store_chunks(ticker, fiscal_year, processor)
            tables_stored = db_integration.store_tables(ticker, fiscal_year, processor)

            # Fetch and store Finqual statements (clean financial statements)
            if skip_finqual:
                logger.info(f"⏭️  Skipping Finqual fetch (--skip-finqual enabled)")
                finqual_stored = 0
            else:
                finqual_stored = db_integration.fetch_and_store_finqual_statements(ticker, fiscal_year)

            total_chunks += chunks_stored
            total_tables += tables_stored
            total_finqual += finqual_stored
            filings_processed += 1

            logger.info(f"✅ FY{fiscal_year}: {chunks_stored} chunks, {tables_stored} tables, {finqual_stored} Finqual statements")

            # Clean up processor to free memory
            processor = None
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Failed to process FY{fiscal_year}: {e}")
            # Clean up on error too
            try:
                processor = None
                gc.collect()
            except:
                pass
            continue

    # Build result message
    if filings_processed == 0 and filings_skipped == 0:
        result_msg = f"⚠️  {ticker}: No 10-K found"
    elif filings_processed == 0 and filings_skipped > 0:
        result_msg = f"⏭️  {ticker}: All {filings_skipped} filings already processed (skipped)"
    elif filings_skipped > 0:
        result_msg = f"✅ {ticker}: {filings_processed} new, {filings_skipped} skipped, {total_chunks} chunks, {total_tables} tables, {total_finqual} Finqual"
    else:
        result_msg = f"✅ {ticker}: {filings_processed} filings, {total_chunks} chunks, {total_tables} tables, {total_finqual} Finqual"

    logger.info(result_msg)

    # Clean up temporary files after processing this ticker
    cleanup_temp_files()

    return {
        'ticker': ticker,
        'filings_processed': filings_processed,
        'filings_skipped': filings_skipped,
        'chunks_stored': total_chunks,
        'tables_stored': total_tables,
        'finqual_stored': total_finqual
    }


def load_all_tickers(ticker_file: str = None) -> List[str]:
    """Load all US tickers from file"""
    if ticker_file is None:
        ticker_file = Path(__file__).parent / "us_tickers.txt"
    else:
        ticker_file = Path(ticker_file)

    if not ticker_file.exists():
        logger.error(f"❌ Ticker file not found: {ticker_file}")
        return []

    try:
        with open(ticker_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]

        logger.info(f"📋 Loaded {len(tickers)} tickers from {ticker_file}")
        return tickers
    except Exception as e:
        logger.error(f"❌ Error reading ticker file: {e}")
        return []


def main():
    # Fix CUDA multiprocessing issue - use spawn instead of fork
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description='Ingest 10-K SEC filings with sophisticated processing')
    parser.add_argument('--ticker', type=str, help='Single ticker to ingest')
    parser.add_argument('--tickers', type=str, nargs='+', help='Multiple tickers')
    parser.add_argument('--all-tickers', action='store_true', help='Ingest ALL US companies (9500+)')
    parser.add_argument('--all-financebench', action='store_true', help='Ingest all FinanceBench companies')
    parser.add_argument('--ticker-file', type=str, help='Path to ticker file (default: us_tickers.txt)')
    parser.add_argument('--lookback-years', type=int, default=1, help='Years to look back (default: 1)')
    parser.add_argument('--max-tickers', type=int, help='Limit number of tickers to process')
    parser.add_argument('--skip-first', type=int, default=0, help='Skip first N tickers (for resuming)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers (default: 3)')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially (no parallel processing)')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout per ticker in minutes (default: 30)')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed tickers once')

    args = parser.parse_args()

    # Determine tickers
    tickers = []
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all_tickers:
        tickers = load_all_tickers(args.ticker_file)
        if not tickers:
            logger.error("❌ Failed to load tickers")
            sys.exit(1)
    elif args.all_financebench:
        tickers = list(FINANCEBENCH_COMPANIES.values())
    else:
        parser.print_help()
        print("\nError: Must specify --ticker, --tickers, --all-tickers, or --all-financebench")
        sys.exit(1)

    # Apply skip and max limits
    if args.skip_first > 0:
        logger.info(f"⏭️ Skipping first {args.skip_first} tickers")
        tickers = tickers[args.skip_first:]

    if args.max_tickers:
        logger.info(f"🔢 Limiting to {args.max_tickers} tickers")
        tickers = tickers[:args.max_tickers]

    logger.info(f"🚀 10-K Database Ingestion Starting")
    logger.info(f"  Tickers: {len(tickers)} companies")
    logger.info(f"  Lookback: {args.lookback_years} year(s)")
    logger.info(f"  Workers: {args.workers if not args.sequential else 1} (parallel)" if not args.sequential else "  Mode: Sequential")
    logger.info(f"  Processing: Hierarchical extraction + contextual chunking + table identification")

    # Get database URL for workers (must be consistent across all processes)
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("❌ PG_VECTOR or DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info(f"  Database: {db_url[:50]}...")

    # Initialize database (create tables) - use same URL as workers
    db_integration = DatabaseIntegration(db_url=db_url)
    db_integration.ensure_tables()

    # Process tickers
    results = []
    start_time = time.time()

    if args.sequential:
        # Sequential processing (old behavior)
        logger.info("🔄 Running in sequential mode")
        for idx, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 Progress: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%)")
                logger.info(f"{'='*70}")

                result = ingest_ticker(ticker, args.lookback_years, db_integration)
                results.append(result)

                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(tickers) - idx)
                    logger.info(f"⏱️ Processed {idx}/{len(tickers)} in {elapsed/60:.1f}m, "
                              f"~{remaining/60:.1f}m remaining")

            except Exception as e:
                logger.error(f"❌ Failed to ingest {ticker}: {e}")
                results.append({'ticker': ticker, 'error': str(e)})
                continue

    else:
        # Parallel processing with multiple workers
        logger.info(f"🔥 Running with {args.workers} parallel workers")

        # Check Python version for max_tasks_per_child support
        python_version = sys.version_info
        supports_max_tasks = python_version >= (3, 11)

        if supports_max_tasks:
            logger.info("✅ Using max_tasks_per_child=1 to prevent memory accumulation")
        else:
            logger.info("⚠️  Python < 3.11 detected - workers will be reused (may accumulate memory)")
            logger.info("   Consider upgrading to Python 3.11+ for better memory management")

        completed = 0
        # Use max_tasks_per_child=1 to ensure each process handles only ONE ticker
        # This prevents memory accumulation from finqual and other libraries
        executor_kwargs = {'max_workers': args.workers}
        if supports_max_tasks:
            executor_kwargs['max_tasks_per_child'] = 1

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            # Submit all tasks with timeout
            future_to_ticker = {
                executor.submit(ingest_ticker_worker, ticker, args.lookback_years, db_url, args.timeout): ticker
                for ticker in tickers
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_ticker, timeout=None):
                ticker = future_to_ticker[future]
                completed += 1

                try:
                    # Get result with a timeout buffer (add 2 minutes for overhead)
                    result = future.result(timeout=(args.timeout * 60) + 120)
                    results.append(result)

                    # Log progress every 10 completions or on error
                    if completed % 10 == 0 or 'error' in result:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed
                        remaining = avg_time * (len(tickers) - completed)

                        successful = len([r for r in results if r.get('filings_processed', 0) > 0])
                        failed = len([r for r in results if 'error' in r])
                        logger.info(f"⏱️ Progress: {completed}/{len(tickers)} "
                                  f"({completed/len(tickers)*100:.1f}%) | "
                                  f"✅ {successful} successful | ❌ {failed} failed | "
                                  f"Time: {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining")

                except TimeoutError:
                    logger.error(f"⏱️ Future timeout for {ticker} (process didn't respond)")
                    results.append({'ticker': ticker, 'error': 'Future timeout - process hung'})
                except Exception as e:
                    logger.error(f"❌ Worker exception for {ticker}: {e}")
                    results.append({'ticker': ticker, 'error': str(e)})

        logger.info(f"🎉 All workers completed!")

        # Retry failed tickers if requested
        if args.retry_failed:
            failed_tickers = [r['ticker'] for r in results if 'error' in r]
            if failed_tickers:
                logger.info(f"\n🔄 Retrying {len(failed_tickers)} failed tickers...")
                retry_results = []

                with ProcessPoolExecutor(**executor_kwargs) as executor:
                    future_to_ticker = {
                        executor.submit(ingest_ticker_worker, ticker, args.lookback_years, db_url, args.timeout): ticker
                        for ticker in failed_tickers
                    }

                    for future in as_completed(future_to_ticker, timeout=None):
                        ticker = future_to_ticker[future]
                        try:
                            result = future.result(timeout=(args.timeout * 60) + 120)
                            retry_results.append(result)

                            if 'error' not in result:
                                logger.info(f"✅ Retry successful for {ticker}")
                                # Update original result
                                for i, r in enumerate(results):
                                    if r['ticker'] == ticker:
                                        results[i] = result
                                        break
                            else:
                                logger.warning(f"❌ Retry failed for {ticker}: {result.get('error')}")
                        except Exception as e:
                            logger.error(f"❌ Retry exception for {ticker}: {e}")

                successful_retries = len([r for r in retry_results if 'error' not in r and r.get('filings_processed', 0) > 0])
                logger.info(f"🔄 Retry complete: {successful_retries}/{len(failed_tickers)} succeeded")

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 INGESTION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Tickers processed: {len(results)}")

    total_filings = sum(r.get('filings_processed', 0) for r in results)
    total_skipped = sum(r.get('filings_skipped', 0) for r in results)
    total_chunks = sum(r.get('chunks_stored', 0) for r in results)
    total_tables = sum(r.get('tables_stored', 0) for r in results)
    total_finqual = sum(r.get('finqual_stored', 0) for r in results)

    logger.info(f"Total filings processed: {total_filings}")
    logger.info(f"Total filings skipped: {total_skipped}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Total tables: {total_tables}")
    logger.info(f"Total Finqual statements: {total_finqual}")

    # Count successes and failures
    successful = [r for r in results if 'error' not in r and r.get('filings_processed', 0) > 0]
    skipped_all = [r for r in results if 'error' not in r and r.get('filings_processed', 0) == 0 and r.get('filings_skipped', 0) > 0]
    failed = [r for r in results if 'error' in r]
    no_data = [r for r in results if 'error' not in r and r.get('filings_processed', 0) == 0 and r.get('filings_skipped', 0) == 0]

    logger.info(f"\nResults breakdown:")
    logger.info(f"  ✅ Successful: {len(successful)}")
    logger.info(f"  ⏭️  Already processed (skipped): {len(skipped_all)}")
    logger.info(f"  ❌ Failed: {len(failed)}")
    logger.info(f"  ⚠️ No data found: {len(no_data)}")

    # Show sample of successful
    if successful:
        logger.info(f"\nSample successful ingestions:")
        for result in successful[:10]:
            skipped_msg = f", {result.get('filings_skipped', 0)} skipped" if result.get('filings_skipped', 0) > 0 else ""
            logger.info(f"  {result['ticker']}: {result['filings_processed']} filings{skipped_msg}, "
                       f"{result['chunks_stored']} chunks, {result['tables_stored']} tables, "
                       f"{result.get('finqual_stored', 0)} Finqual")
        if len(successful) > 10:
            logger.info(f"  ... and {len(successful) - 10} more")

    # Show companies that were entirely skipped
    if skipped_all:
        logger.info(f"\n⏭️  Companies with all filings already processed:")
        sample_skipped = [r['ticker'] for r in skipped_all[:20]]
        logger.info(f"  {', '.join(sample_skipped)}")
        if len(skipped_all) > 20:
            logger.info(f"  ... and {len(skipped_all) - 20} more")

    # Show all failures
    if failed:
        logger.info(f"\nFailed tickers:")
        for result in failed[:20]:
            logger.info(f"  {result['ticker']}: {result.get('error', 'Unknown error')}")
        if len(failed) > 20:
            logger.info(f"  ... and {len(failed) - 20} more errors")

    logger.info(f"\n✅ Ingestion complete!")

    # Final cleanup
    try:
        db_integration.close_connection()
    except:
        pass
    cleanup_temp_files()


if __name__ == "__main__":
    main()
