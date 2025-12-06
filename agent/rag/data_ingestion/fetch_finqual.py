#!/usr/bin/env python3
"""
Fetch Finqual Financial Statements Separately

This script fetches clean financial statements from Finqual and stores them
in the database. Run this separately after the main ingestion to avoid memory issues.

Usage:
    # Single ticker
    python fetch_finqual.py --ticker AAPL

    # Multiple tickers
    python fetch_finqual.py --tickers AAPL MSFT GOOGL

    # All tickers in database
    python fetch_finqual.py --all

    # First 50 companies
    python fetch_finqual.py --all --max-tickers 50

    # Skip first 50, process next 50
    python fetch_finqual.py --all --skip-first 50 --max-tickers 50
"""

import argparse
import os
import sys
import logging
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Import finqual
try:
    import finqual as fq
    FINQUAL_AVAILABLE = True
except ImportError:
    print("❌ Error: finqual library not found. Install with: pip install finqual")
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
        logging.FileHandler('finqual_fetch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection"""
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")
    if not db_url:
        raise ValueError("PG_VECTOR or DATABASE_URL environment variable not set")
    return psycopg2.connect(db_url)


def get_all_tickers_from_db():
    """Get all unique tickers from ten_k_chunks table"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ticker FROM ten_k_chunks ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return tickers


def get_fiscal_years_for_ticker(ticker):
    """Get all fiscal years for a ticker from ten_k_chunks table"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT fiscal_year FROM ten_k_chunks WHERE ticker = %s ORDER BY fiscal_year",
        (ticker,)
    )
    years = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return years


def check_finqual_exists(ticker, fiscal_year):
    """Check if Finqual data already exists for this ticker-year"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM ten_k_tables
        WHERE ticker = %s
        AND fiscal_year = %s
        AND table_id LIKE '%FINQUAL%'
    """, (ticker, fiscal_year))

    count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return count > 0


def fetch_and_store_finqual(ticker, fiscal_year, force=False):
    """
    Fetch Finqual statements for a ticker-year and store in database

    Returns:
        int: Number of statements stored, or -1 on error
    """
    # Check if already exists
    if not force and check_finqual_exists(ticker, fiscal_year):
        logger.info(f"⏭️  {ticker} FY{fiscal_year}: Finqual data already exists (use --force to overwrite)")
        return 0

    logger.info(f"📊 Fetching Finqual for {ticker} FY{fiscal_year}...")

    statements_stored = 0
    statement_configs = [
        ('income', 'Income Statement', 'income_statement'),
        ('balance', 'Balance Sheet', 'balance_sheet'),
        ('cash', 'Cash Flow Statement', 'cash_flow')
    ]

    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for api_type, stmt_name, stmt_type in statement_configs:
            try:
                # Fetch from Finqual
                logger.debug(f"  Fetching {stmt_name}...")
                if api_type == 'income':
                    df = fq.Finqual(ticker).income_stmt(fiscal_year)
                elif api_type == 'balance':
                    df = fq.Finqual(ticker).balance_sheet(fiscal_year)
                elif api_type == 'cash':
                    df = fq.Finqual(ticker).cash_flow(fiscal_year)
                else:
                    continue

                if df is None or df.empty:
                    logger.debug(f"  ⚠️  No {stmt_name} data from Finqual")
                    continue

                # Convert to string
                content = df.to_string()
                table_id = f"{ticker}_FY{fiscal_year}_FINQUAL_{api_type.upper()}"

                # Delete existing if force mode
                if force:
                    cursor.execute(
                        "DELETE FROM ten_k_tables WHERE table_id = %s",
                        (table_id,)
                    )

                # Store in database
                cursor.execute("""
                    INSERT INTO ten_k_tables
                    (table_id, ticker, fiscal_year, content, table_data, path_string,
                     sec_section, sec_section_title, is_financial_statement, statement_type, priority)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (table_id) DO NOTHING
                """, (
                    table_id,
                    ticker,
                    fiscal_year,
                    content,
                    content,  # Store same content in table_data
                    f'finqual > {stmt_name}',
                    'finqual',
                    stmt_name,
                    True,
                    stmt_type,
                    'CRITICAL'
                ))

                statements_stored += 1
                logger.info(f"  ✅ Stored {stmt_name}")

            except Exception as e:
                logger.warning(f"  ⚠️  Failed to fetch {stmt_name}: {e}")
                continue

        conn.commit()

        if statements_stored > 0:
            logger.info(f"✅ {ticker} FY{fiscal_year}: Stored {statements_stored} Finqual statements")
        else:
            logger.warning(f"⚠️  {ticker} FY{fiscal_year}: No Finqual data available")

        return statements_stored

    except Exception as e:
        logger.error(f"❌ {ticker} FY{fiscal_year}: Failed - {e}")
        if conn:
            conn.rollback()
        return -1

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Finqual financial statements and store in database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single ticker
  python fetch_finqual.py --ticker AAPL

  # Multiple tickers
  python fetch_finqual.py --tickers AAPL MSFT GOOGL

  # All tickers in database
  python fetch_finqual.py --all

  # Skip first 50, process next 50
  python fetch_finqual.py --all --skip-first 50 --max-tickers 50
        """
    )

    parser.add_argument('--ticker', type=str,
                       help='Single ticker to process')
    parser.add_argument('--tickers', type=str, nargs='+',
                       help='Multiple tickers to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all tickers found in database')
    parser.add_argument('--skip-first', type=int, default=0,
                       help='Skip first N tickers (for --all mode)')
    parser.add_argument('--max-tickers', type=int,
                       help='Limit number of tickers to process (for --all mode)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing Finqual data')
    parser.add_argument('--year', type=int,
                       help='Specific fiscal year (only with --ticker)')

    args = parser.parse_args()

    # Determine which tickers to process
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all:
        logger.info("📊 Fetching all tickers from database...")
        tickers = get_all_tickers_from_db()
        logger.info(f"✅ Found {len(tickers)} tickers in database")

        # Apply skip and limit
        if args.skip_first > 0:
            logger.info(f"⏭️  Skipping first {args.skip_first} tickers")
            tickers = tickers[args.skip_first:]

        if args.max_tickers:
            logger.info(f"🔢 Limiting to {args.max_tickers} tickers")
            tickers = tickers[:args.max_tickers]
    else:
        parser.error("Must specify --ticker, --tickers, or --all")

    # Banner
    logger.info("=" * 80)
    logger.info("FINQUAL FINANCIAL STATEMENTS FETCHER")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Tickers to process: {len(tickers)}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info("=" * 80)

    # Process each ticker
    total_statements = 0
    successful_tickers = 0
    failed_tickers = []
    skipped_count = 0

    for idx, ticker in enumerate(tickers, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"📊 Progress: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%)")
        logger.info(f"Processing {ticker}...")
        logger.info(f"{'=' * 70}")

        try:
            # Get fiscal years for this ticker
            if args.year:
                fiscal_years = [args.year]
            else:
                fiscal_years = get_fiscal_years_for_ticker(ticker)

            if not fiscal_years:
                logger.warning(f"⚠️  No fiscal years found for {ticker}")
                continue

            logger.info(f"📅 Found {len(fiscal_years)} fiscal year(s): {fiscal_years}")

            ticker_statements = 0
            ticker_had_error = False

            for fiscal_year in fiscal_years:
                result = fetch_and_store_finqual(ticker, fiscal_year, args.force)

                if result > 0:
                    ticker_statements += result
                elif result == -1:
                    ticker_had_error = True
                elif result == 0 and check_finqual_exists(ticker, fiscal_year):
                    skipped_count += 1

            if ticker_statements > 0:
                total_statements += ticker_statements
                successful_tickers += 1

            if ticker_had_error:
                failed_tickers.append(ticker)

        except Exception as e:
            logger.error(f"❌ Error processing {ticker}: {e}")
            failed_tickers.append(ticker)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Tickers processed: {len(tickers)}")
    logger.info(f"Total Finqual statements stored: {total_statements}")
    logger.info(f"Successful tickers: {successful_tickers}")
    logger.info(f"Already had data (skipped): {skipped_count}")
    logger.info(f"Failed tickers: {len(failed_tickers)}")

    if failed_tickers:
        logger.info(f"\n❌ FAILED TICKERS:")
        logger.info(f"  {', '.join(failed_tickers)}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ FINQUAL FETCH COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
