#!/usr/bin/env python3
"""
Backfill exhibit chunks for all existing 10-K ticker-years that lack them.

Strategy:
- Queries the DB for (ticker, fiscal_year) pairs with NO exhibit_type chunks
- For each, downloads the 10-K submission (datamule uses local cache where available)
- Extracts exhibit chunks only (skips main body re-embedding)
- INSERTs exhibit chunks into ten_k_chunks alongside existing main body chunks
- Stores exhibit full-text in complete_sec_filings + S3

Usage:
    python backfill_exhibits.py --workers 4
    python backfill_exhibits.py --workers 4 --max-tickers 20   # test run
    python backfill_exhibits.py --ticker NET                    # single ticker
"""

import argparse
import os
import sys
import json
import logging
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).parent))

project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(dotenv_path=project_root / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backfill_exhibits.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_url() -> str:
    return os.getenv("PG_VECTOR", "")


def get_tickers_needing_exhibits(db_url: str, only_ticker: str = None) -> list:
    """Return list of (ticker, fiscal_year) pairs that have NO exhibit chunks."""
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    if only_ticker:
        cursor.execute("""
            SELECT DISTINCT ticker, fiscal_year
            FROM ten_k_chunks
            WHERE filing_type = '10-K' AND UPPER(ticker) = %s
            AND NOT EXISTS (
                SELECT 1 FROM ten_k_chunks e
                WHERE e.ticker = ten_k_chunks.ticker
                  AND e.fiscal_year = ten_k_chunks.fiscal_year
                  AND e.exhibit_type IS NOT NULL
            )
            ORDER BY ticker, fiscal_year DESC
        """, (only_ticker.upper(),))
    else:
        cursor.execute("""
            SELECT DISTINCT ticker, fiscal_year
            FROM ten_k_chunks
            WHERE filing_type = '10-K'
            AND NOT EXISTS (
                SELECT 1 FROM ten_k_chunks e
                WHERE e.ticker = ten_k_chunks.ticker
                  AND e.fiscal_year = ten_k_chunks.fiscal_year
                  AND e.exhibit_type IS NOT NULL
            )
            ORDER BY ticker, fiscal_year DESC
        """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows  # list of (ticker, fiscal_year)


# ---------------------------------------------------------------------------
# Core: process one (ticker, fiscal_year)
# ---------------------------------------------------------------------------

def process_ticker_year(ticker: str, fiscal_year: int, db_url: str) -> dict:
    """
    Download the 10-K, extract exhibit chunks, embed and insert them.
    Returns a summary dict.
    """
    from ingest_10k_filings_full import download_and_extract_10k, DataProcessor
    from ingest_10k_to_database import DatabaseIntegration, upload_filing_to_bucket

    result = {'ticker': ticker, 'fiscal_year': fiscal_year,
              'exhibit_chunks': 0, 'exhibit_types': [], 'status': 'ok'}
    try:
        # Download (datamule caches locally; repeated calls are fast)
        filings = download_and_extract_10k(ticker, fiscal_year, fiscal_year)
        if fiscal_year not in filings:
            # Try adjacent year in case fiscal year detection is off by one
            for fy, data in filings.items():
                if abs(fy - fiscal_year) <= 1:
                    filings = {fiscal_year: data}
                    break
            else:
                result['status'] = 'no_filing'
                return result

        filing_data = filings[fiscal_year]
        ctx_chunks = filing_data.get('contextual_chunks', [])

        # Filter to exhibit chunks only
        exhibit_ctx = [c for c in ctx_chunks
                       if c.get('exhibit_source') or c.get('metadata', {}).get('exhibit_source')]

        if not exhibit_ctx:
            result['status'] = 'no_exhibits'
            return result

        # Use DataProcessor to create embeddings for exhibit chunks only
        processor = DataProcessor()
        # Temporarily replace contextual_chunks with exhibit-only subset
        exhibit_filing_data = dict(filing_data)
        exhibit_filing_data['contextual_chunks'] = exhibit_ctx
        exhibit_filing_data['ticker'] = ticker

        processor.prepare_chunks(exhibit_filing_data, use_hierarchical=True, exclude_titles=True)
        processor.create_embeddings()
        chunks = processor.get_chunks()
        embeddings = processor.get_embeddings()

        # Keep only chunks that have exhibit_source set
        exhibit_pairs = [(c, e) for c, e in zip(chunks, embeddings)
                         if c.get('exhibit_source')]

        if not exhibit_pairs:
            result['status'] = 'no_exhibit_chunks_after_processing'
            return result

        # Insert into DB
        db = DatabaseIntegration(db_url)
        conn = db.get_connection()
        cursor = conn.cursor()

        filing_type = '10-K'
        chunk_data = []
        for idx, (chunk, embedding) in enumerate(exhibit_pairs):
            exhibit_type = chunk.get('exhibit_source')
            metadata = {
                'ticker': ticker,
                'fiscal_year': fiscal_year,
                'filing_type': filing_type,
                'chunk_index': idx,
                'level': chunk.get('level'),
                'path': chunk.get('path', []),
                'sec_section': chunk.get('sec_section'),
                'sec_section_title': chunk.get('sec_section_title'),
                'exhibit_type': exhibit_type,
            }
            citation = f"{ticker}_10K_FY{fiscal_year}_EX_{idx}"
            chunk_data.append((
                chunk['content'],
                embedding.tolist(),
                json.dumps(metadata),
                ticker,
                fiscal_year,
                filing_type,
                idx,
                citation,
                chunk.get('type', 'text'),
                chunk.get('sec_section', 'unknown'),
                chunk.get('sec_section_title', 'Unknown'),
                chunk.get('path_string', ''),
                exhibit_type,
                chunk.get('char_offset'),
                chunk.get('chunk_length'),
            ))

        execute_values(
            cursor,
            """
            INSERT INTO ten_k_chunks
            (chunk_text, embedding, metadata, ticker, fiscal_year, filing_type,
             chunk_index, citation, chunk_type, sec_section, sec_section_title,
             path_string, exhibit_type, char_offset, chunk_length)
            VALUES %s
            """,
            chunk_data,
            template="(%s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        conn.commit()
        cursor.close()

        exhibit_types_inserted = list({c.get('exhibit_source') for c, _ in exhibit_pairs})

        # Store exhibit full-text documents
        exhibit_texts = filing_data.get('_exhibit_texts', {})
        for ex_type, ex_text in exhibit_texts.items():
            db.store_exhibit_filing(ticker, fiscal_year, ex_type, ex_text)

        result['exhibit_chunks'] = len(exhibit_pairs)
        result['exhibit_types'] = exhibit_types_inserted
        logger.info(f"✅ {ticker} FY{fiscal_year}: {len(exhibit_pairs)} exhibit chunks "
                    f"({exhibit_types_inserted})")

    except Exception as e:
        result['status'] = f'error: {e}'
        logger.error(f"❌ {ticker} FY{fiscal_year}: {e}")
    finally:
        gc.collect()
        time.sleep(0.5)  # brief pause to keep CPU/GPU load low

    return result


def worker(args):
    ticker, fiscal_year, db_url = args
    return process_ticker_year(ticker, fiscal_year, db_url)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backfill exhibit chunks for existing 10-K data")
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-tickers', type=int, default=None,
                        help="Cap number of distinct tickers (for testing)")
    parser.add_argument('--ticker', type=str, default=None,
                        help="Process only this ticker")
    parser.add_argument('--timeout', type=int, default=600,
                        help="Timeout per ticker-year in seconds")
    args = parser.parse_args()

    db_url = get_db_url()
    if not db_url:
        logger.error("PG_VECTOR env var not set")
        sys.exit(1)

    logger.info("🔍 Querying DB for ticker-years missing exhibit chunks...")
    pairs = get_tickers_needing_exhibits(db_url, only_ticker=args.ticker)
    logger.info(f"📋 Found {len(pairs)} ticker-years needing exhibits")

    if args.max_tickers:
        # Cap by distinct tickers
        seen = set()
        filtered = []
        for ticker, fy in pairs:
            seen.add(ticker)
            if len(seen) <= args.max_tickers:
                filtered.append((ticker, fy))
        pairs = filtered
        logger.info(f"🔬 Limited to {len(pairs)} ticker-years ({len(seen)} tickers)")

    if not pairs:
        logger.info("✅ Nothing to do — all ticker-years already have exhibit chunks")
        return

    tasks = [(ticker, fy, db_url) for ticker, fy in pairs]
    start = time.time()

    ok = skipped = errors = total_chunks = 0
    results = []

    logger.info(f"🚀 Starting exhibit backfill with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for future in as_completed(futures):
            t = futures[future]
            try:
                res = future.result(timeout=args.timeout)
                results.append(res)
                if res['status'] == 'ok':
                    ok += 1
                    total_chunks += res['exhibit_chunks']
                elif res['status'] in ('no_filing', 'no_exhibits', 'no_exhibit_chunks_after_processing'):
                    skipped += 1
                else:
                    errors += 1
            except TimeoutError:
                errors += 1
                logger.error(f"⏱️  Timeout: {t[0]} FY{t[1]}")
            except Exception as e:
                errors += 1
                logger.error(f"❌ Worker error {t[0]} FY{t[1]}: {e}")

    elapsed = time.time() - start
    logger.info(f"""
{'='*60}
📊 EXHIBIT BACKFILL SUMMARY
{'='*60}
  Total time:      {elapsed:.0f}s
  Ticker-years:    {len(pairs)}
  ✅ With exhibits: {ok}
  ⏭️  No exhibits:   {skipped}
  ❌ Errors:        {errors}
  Chunks added:    {total_chunks}
{'='*60}
""")


if __name__ == "__main__":
    main()
