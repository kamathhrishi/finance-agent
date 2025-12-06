#!/usr/bin/env python3
"""
10-K SEC Filings Data Ingestion

This script downloads and processes 10-K filings from the last 1 year for specified companies.
It creates embeddings and stores them in the PostgreSQL database alongside earnings transcripts.

Usage:
    python ingest_10k_filings.py --tickers AAPL MSFT GOOGL
    python ingest_10k_filings.py --ticker AAPL
    python ingest_10k_filings.py --all-financebench

Features:
- Downloads 10-K filings using datamule library
- Extracts hierarchical content from SEC filings
- Creates contextual chunks optimized for RAG
- Generates embeddings using sentence-transformers
- Stores in existing PostgreSQL database with pgvector
- Supports batch processing for multiple companies
"""

import argparse
import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Import datamule for SEC filings download
try:
    from datamule import Portfolio
except ImportError:
    print("Error: datamule library not found. Install with: pip install datamule")
    sys.exit(1)

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('10k_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# FinanceBench companies for reference
FINANCEBENCH_COMPANIES = {
    "3M": "MMM", "AES Corporation": "AES", "AMD": "AMD", "Activision Blizzard": "ATVI",
    "Adobe": "ADBE", "Amazon": "AMZN", "Amcor": "AMCR", "American Express": "AXP",
    "American Water Works": "AWK", "Best Buy": "BBY", "Block": "SQ", "Boeing": "BA",
    "CVS Health": "CVS", "Coca-Cola": "KO", "Corning": "GLW", "Costco": "COST",
    "General Mills": "GIS", "JPMorgan": "JPM", "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC", "Lockheed Martin": "LMT", "MGM Resorts": "MGM",
    "Microsoft": "MSFT", "Netflix": "NFLX", "Nike": "NKE", "Paypal": "PYPL",
    "PepsiCo": "PEP", "Pfizer": "PFE", "Ulta Beauty": "ULTA", "Verizon": "VZ",
    "Walmart": "WMT"
}


class TenKIngestionProcessor:
    """Processes 10-K filings: downloads, chunks, creates embeddings, and stores in database."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": "all-MiniLM-L6-v2",
            "pgvector_url": os.getenv("DATABASE_URL", ""),
        }

        # Initialize embedding model
        logger.info(f"🤖 Loading embedding model: {self.config['embedding_model']}")
        self.embedding_model = SentenceTransformer(self.config["embedding_model"])
        logger.info("✅ Embedding model loaded")

    def extract_fiscal_year(self, document_data: Dict) -> Optional[int]:
        """Extract fiscal year from document metadata."""
        try:
            # Try various locations for fiscal year
            if isinstance(document_data, dict):
                # Check direct fields
                if 'fiscal_year' in document_data:
                    return int(document_data['fiscal_year'])

                # Check document wrapper
                doc = document_data.get('document', document_data)
                if 'fiscal_year' in doc:
                    return int(doc['fiscal_year'])

                # Check metadata
                metadata = doc.get('metadata', {})
                if 'fiscal_year' in metadata:
                    return int(metadata['fiscal_year'])

                # Try to extract from filing date
                if 'filed_date' in doc:
                    filed_date = doc['filed_date']
                    if isinstance(filed_date, str):
                        return int(filed_date[:4])

            return None
        except Exception as e:
            logger.warning(f"Could not extract fiscal year: {e}")
            return None

    def extract_text_from_sections(self, document_data: Dict) -> str:
        """Extract text content from document sections."""
        try:
            doc = document_data.get('document', document_data)
            text_parts = []

            # Try to get sections
            if 'sections' in doc:
                for section in doc['sections']:
                    if isinstance(section, dict):
                        if 'content' in section:
                            text_parts.append(section['content'])
                        elif 'text' in section:
                            text_parts.append(section['text'])

            # Fallback to full text
            if not text_parts:
                if 'text' in doc:
                    return doc['text']
                elif 'content' in doc:
                    return doc['content']

            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())

            start = end - chunk_overlap
            if start >= len(text):
                break

        return chunks

    def download_10k_filings(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Download 10-K filings for a ticker within date range."""
        logger.info(f"📥 Downloading 10-K filings for {ticker} from {start_date} to {end_date}")

        try:
            portfolio = Portfolio(ticker)
            portfolio.download_submissions(
                ticker=ticker,
                filing_date=(start_date, end_date),
                submission_type=['10-K']
            )

            filings = []
            for document in portfolio.document_type('10-K'):
                try:
                    document.parse()

                    # Extract fiscal year
                    fiscal_year = self.extract_fiscal_year(document.data)
                    if not fiscal_year:
                        logger.warning(f"⚠️ Could not determine fiscal year for {ticker}, skipping")
                        continue

                    # Extract text
                    text = self.extract_text_from_sections(document.data)
                    if not text:
                        logger.warning(f"⚠️ No text content found for {ticker} FY{fiscal_year}")
                        continue

                    filing_info = {
                        'ticker': ticker.upper(),
                        'fiscal_year': fiscal_year,
                        'text': text,
                        'document_data': document.data,
                        'filing_type': '10-K'
                    }
                    filings.append(filing_info)
                    logger.info(f"✅ Processed 10-K for {ticker} FY{fiscal_year} ({len(text):,} chars)")

                except Exception as e:
                    logger.error(f"Error processing document for {ticker}: {e}")
                    continue

            return filings

        except Exception as e:
            logger.error(f"Failed to download 10-K for {ticker}: {e}")
            return []

    def create_embeddings_batch(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings for text chunks in batches."""
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])

    def ensure_10k_table(self, cursor):
        """Ensure 10k_chunks table exists with proper structure."""
        logger.info("🔍 Ensuring 10k_chunks table exists...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS 10k_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding VECTOR(384),
                metadata JSONB,
                ticker VARCHAR(10),
                fiscal_year INTEGER,
                filing_type VARCHAR(10) DEFAULT '10-K',
                chunk_index INTEGER,
                citation VARCHAR(200),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create index on ticker and fiscal_year for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_10k_ticker_year
            ON 10k_chunks(ticker, fiscal_year);
        """)

        # Create index on embedding for vector search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_10k_embedding
            ON 10k_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        logger.info("✅ 10k_chunks table ensured")

    def store_filing_chunks(self, filing: Dict[str, Any]) -> int:
        """Process and store a single 10-K filing in the database."""
        ticker = filing['ticker']
        fiscal_year = filing['fiscal_year']
        text = filing['text']

        logger.info(f"📊 Processing {ticker} FY{fiscal_year}")

        # Chunk the text
        chunks = self.chunk_text(text)
        logger.info(f"  📝 Created {len(chunks)} chunks")

        if not chunks:
            logger.warning(f"  ⚠️ No chunks created for {ticker} FY{fiscal_year}")
            return 0

        # Create embeddings
        logger.info(f"  🧮 Creating embeddings...")
        embeddings = self.create_embeddings_batch(chunks)
        logger.info(f"  ✅ Created {len(embeddings)} embeddings")

        # Store in database
        if not self.config["pgvector_url"]:
            logger.error("  ❌ DATABASE_URL not set")
            return 0

        try:
            conn = psycopg2.connect(self.config["pgvector_url"])
            cursor = conn.cursor()

            # Ensure table exists
            self.ensure_10k_table(cursor)

            # Delete existing chunks for this ticker and fiscal year
            cursor.execute("""
                DELETE FROM 10k_chunks
                WHERE ticker = %s AND fiscal_year = %s
            """, (ticker, fiscal_year))
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"  🗑️ Deleted {deleted_count} existing chunks for {ticker} FY{fiscal_year}")

            # Prepare data for insertion
            chunk_data = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    'ticker': ticker,
                    'fiscal_year': fiscal_year,
                    'filing_type': '10-K',
                    'chunk_index': idx,
                    'total_chunks': len(chunks)
                }

                citation = f"{ticker}_10K_FY{fiscal_year}_{idx}"

                chunk_data.append((
                    chunk,
                    embedding.tolist(),
                    json.dumps(metadata),
                    ticker,
                    fiscal_year,
                    '10-K',
                    idx,
                    citation
                ))

            # Insert chunks
            execute_values(
                cursor,
                """
                INSERT INTO 10k_chunks
                (chunk_text, embedding, metadata, ticker, fiscal_year, filing_type, chunk_index, citation)
                VALUES %s
                """,
                chunk_data,
                template="(%s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s)"
            )

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"  ✅ Stored {len(chunk_data)} chunks for {ticker} FY{fiscal_year}")
            return len(chunk_data)

        except Exception as e:
            logger.error(f"  ❌ Failed to store chunks: {e}")
            return 0

    def ingest_ticker(self, ticker: str, lookback_years: int = 1) -> Dict[str, Any]:
        """Ingest 10-K filings for a single ticker."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 Starting ingestion for {ticker}")
        logger.info(f"{'='*60}")

        # Calculate date range (last N years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Download filings
        filings = self.download_10k_filings(ticker, start_str, end_str)

        if not filings:
            logger.warning(f"⚠️ No 10-K filings found for {ticker}")
            return {'ticker': ticker, 'filings_processed': 0, 'chunks_stored': 0}

        logger.info(f"📦 Found {len(filings)} 10-K filing(s) for {ticker}")

        # Process each filing
        total_chunks = 0
        for filing in filings:
            chunks_stored = self.store_filing_chunks(filing)
            total_chunks += chunks_stored

        logger.info(f"✅ Completed ingestion for {ticker}: {len(filings)} filings, {total_chunks} total chunks")

        return {
            'ticker': ticker,
            'filings_processed': len(filings),
            'chunks_stored': total_chunks
        }


def main():
    parser = argparse.ArgumentParser(description='Ingest 10-K SEC filings into the database')
    parser.add_argument('--ticker', type=str, help='Single ticker to ingest (e.g., AAPL)')
    parser.add_argument('--tickers', type=str, nargs='+', help='Multiple tickers to ingest (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--all-financebench', action='store_true', help='Ingest all FinanceBench companies')
    parser.add_argument('--lookback-years', type=int, default=1, help='Number of years to look back (default: 1)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for text splitting (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap (default: 200)')

    args = parser.parse_args()

    # Determine which tickers to process
    tickers = []
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all_financebench:
        tickers = list(FINANCEBENCH_COMPANIES.values())
    else:
        parser.print_help()
        print("\nError: Must specify --ticker, --tickers, or --all-financebench")
        sys.exit(1)

    logger.info(f"🚀 10-K Ingestion Starting")
    logger.info(f"  Tickers: {', '.join(tickers)}")
    logger.info(f"  Lookback: {args.lookback_years} year(s)")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Chunk overlap: {args.chunk_overlap}")

    # Initialize processor
    processor = TenKIngestionProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Process each ticker
    results = []
    start_time = time.time()

    for ticker in tickers:
        try:
            result = processor.ingest_ticker(ticker, lookback_years=args.lookback_years)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Failed to ingest {ticker}: {e}")
            results.append({'ticker': ticker, 'error': str(e)})

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 INGESTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Tickers processed: {len(results)}")

    total_filings = sum(r.get('filings_processed', 0) for r in results)
    total_chunks = sum(r.get('chunks_stored', 0) for r in results)

    logger.info(f"Total filings: {total_filings}")
    logger.info(f"Total chunks: {total_chunks}")

    # Detail by ticker
    logger.info(f"\nBy ticker:")
    for result in results:
        if 'error' in result:
            logger.info(f"  {result['ticker']}: ERROR - {result['error']}")
        else:
            logger.info(f"  {result['ticker']}: {result['filings_processed']} filings, {result['chunks_stored']} chunks")

    logger.info(f"\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
