#!/usr/bin/env python3
"""
Transcript Bucket Ingestion

For each earnings transcript of a tech company (mega/large/mid cap):
  1. Upload full transcript text to the Railway S3 bucket.
  2. Upsert complete_transcripts.bucket_key so the viewer can fetch it.
  3. Compute char_offset + chunk_length for every row in transcript_chunks
     that matches (ticker, year, quarter) so the viewer can jump to exact positions.

Usage:
    python ingest_transcripts_to_bucket.py
    python ingest_transcripts_to_bucket.py --transcript-dir /path/to/transcripts
    python ingest_transcripts_to_bucket.py --tickers AAPL MSFT GOOGL
    python ingest_transcripts_to_bucket.py --market-caps "Large Cap" "Mid Cap"
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import boto3
import psycopg2
import psycopg2.extras
from botocore.config import Config
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(dotenv_path=project_root / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── S3 ─────────────────────────────────────────────────────────────────────────
_s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("RAILWAY_BUCKET_ENDPOINT", "").strip(),
    aws_access_key_id=os.getenv("RAILWAY_BUCKET_ACCESS_KEY_ID", "").strip(),
    aws_secret_access_key=os.getenv("RAILWAY_BUCKET_SECRET_KEY", "").strip(),
    region_name="auto",
    config=Config(signature_version="s3v4"),
)
_BUCKET = os.getenv("RAILWAY_BUCKET_NAME", "").strip()

# ── DB ──────────────────────────────────────────────────────────────────────────
_PG_URL = os.getenv("PG_VECTOR", "").strip() or os.getenv("DATABASE_URL", "").strip()


def get_conn():
    return psycopg2.connect(_PG_URL)


# ── Helpers ────────────────────────────────────────────────────────────────────

def upload_to_bucket(key: str, text: str) -> bool:
    """Upload text to S3, return True on success."""
    try:
        _s3.put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return True
    except Exception as e:
        logger.warning(f"  ⚠️  S3 upload failed for {key}: {e}")
        return False


def find_char_offset(full_text: str, chunk_text: str) -> Optional[int]:
    """Return the char offset of chunk_text in full_text, or None if not found."""
    chunk_text = chunk_text.strip()
    if not chunk_text:
        return None

    # Exact match first
    idx = full_text.find(chunk_text)
    if idx != -1:
        return idx

    # Flexible whitespace match (handles minor normalization differences)
    anchor = chunk_text[:120]
    tokens = re.split(r"\s+", anchor.strip())
    if len(tokens) >= 3:
        pattern = r"\s+".join(re.escape(t) for t in tokens if t)
        m = re.search(pattern, full_text)
        if m:
            return m.start()

    return None


def ensure_columns(conn):
    """Add char_offset / chunk_length to transcript_chunks if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'transcript_chunks'
              AND column_name IN ('char_offset', 'chunk_length')
        """)
        existing = {row[0] for row in cur.fetchall()}
        if "char_offset" not in existing:
            cur.execute("ALTER TABLE transcript_chunks ADD COLUMN char_offset INTEGER")
            logger.info("✅ Added char_offset column to transcript_chunks")
        if "chunk_length" not in existing:
            cur.execute("ALTER TABLE transcript_chunks ADD COLUMN chunk_length INTEGER")
            logger.info("✅ Added chunk_length column to transcript_chunks")

        # Ensure bucket_key exists on complete_transcripts
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'complete_transcripts'
              AND column_name = 'bucket_key'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE complete_transcripts ADD COLUMN bucket_key VARCHAR(500)")
            logger.info("✅ Added bucket_key column to complete_transcripts")

    conn.commit()


def upsert_complete_transcript(conn, ticker: str, year: int, quarter: int,
                                date: str, bucket_key: str, company_name: str = ""):
    """Insert or update complete_transcripts row with bucket_key."""
    transcript_id = f"{ticker.upper()}_{year}_Q{quarter}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO complete_transcripts (transcript_id, ticker, company_name, date, year, quarter, full_transcript, bucket_key, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (transcript_id) DO UPDATE
              SET bucket_key    = EXCLUDED.bucket_key,
                  company_name  = COALESCE(EXCLUDED.company_name, complete_transcripts.company_name),
                  date          = COALESCE(EXCLUDED.date, complete_transcripts.date)
        """, (
            transcript_id, ticker.upper(), company_name, date,
            year, quarter,
            "",          # full_transcript kept empty — text lives in bucket
            bucket_key,
            json.dumps({}),
        ))
    conn.commit()


def update_chunk_offsets(conn, ticker: str, year: int, quarter: int, full_text: str) -> int:
    """Compute and store char_offset + chunk_length for all chunks of this transcript."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, chunk_text, chunk_index
            FROM transcript_chunks
            WHERE UPPER(ticker) = %s AND year = %s AND quarter = %s
            ORDER BY chunk_index
        """, (ticker.upper(), year, quarter))
        chunks = cur.fetchall()

    if not chunks:
        return 0

    updates = []
    for chunk in chunks:
        text = (chunk["chunk_text"] or "").strip()
        offset = find_char_offset(full_text, text)
        length = len(text) if text else None
        updates.append((offset, length, chunk["id"]))

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            "UPDATE transcript_chunks SET char_offset = %s, chunk_length = %s WHERE id = %s",
            updates,
            page_size=200,
        )
    conn.commit()

    found = sum(1 for o, _, _ in updates if o is not None)
    return found


def process_transcript_file(filepath: Path, conn) -> bool:
    """Process one transcript JSON: upload to S3, upsert DB, update offsets."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"  ⚠️  Could not read {filepath.name}: {e}")
        return False

    if data.get("status") == "no_transcript":
        return False

    ticker = (data.get("ticker") or "").upper()
    year = data.get("year")
    quarter = data.get("quarter")
    inner = data.get("data") or {}
    full_text = inner.get("transcript") or ""
    date = inner.get("date") or ""

    if not ticker or not year or not quarter or not full_text.strip():
        return False

    bucket_key = f"transcripts/{ticker}/{year}_Q{quarter}.txt"

    # 1. Upload to S3
    if not upload_to_bucket(bucket_key, full_text):
        return False

    # 2. Upsert complete_transcripts
    upsert_complete_transcript(conn, ticker, year, quarter, date, bucket_key)

    # 3. Update char_offsets in transcript_chunks
    found = update_chunk_offsets(conn, ticker, year, quarter, full_text)

    logger.info(f"  ✅ {ticker} {year} Q{quarter} — {len(full_text):,} chars uploaded, {found} chunks offset")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest transcripts to S3 bucket with char_offset")
    parser.add_argument("--transcript-dir", default=None,
                        help="Directory containing transcript JSON files (default: searches both known locations)")
    parser.add_argument("--tickers-json", default=None,
                        help="Path to US_TECH_CLEANED.json for ticker filtering")
    parser.add_argument("--market-caps", nargs="+",
                        default=["Mega Cap", "Large Cap", "Mid Cap"],
                        help="Market cap tiers to include")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (overrides --tickers-json filter)")
    args = parser.parse_args()

    # Build target ticker set
    script_dir = Path(__file__).parent
    json_path = Path(args.tickers_json) if args.tickers_json else script_dir / "US_TECH_CLEANED.json"
    target_tickers = None  # None = all tickers

    if args.tickers:
        target_tickers = {t.upper() for t in args.tickers}
        logger.info(f"Filtering to {len(target_tickers)} explicit tickers")
    elif json_path.exists():
        with open(json_path) as f:
            jdata = json.load(f)
        target_caps = set(args.market_caps)
        target_tickers = {
            c["ticker"].upper()
            for c in jdata.get("companies", [])
            if c.get("market_cap") in target_caps
        }
        logger.info(f"Loaded {len(target_tickers)} tickers ({', '.join(target_caps)}) from {json_path.name}")

    # Locate transcript directories
    if args.transcript_dir:
        search_dirs = [Path(args.transcript_dir)]
    else:
        search_dirs = [
            script_dir / "earnings_transcripts",
            project_root / "experiments" / "earnings_transcripts",
        ]
    search_dirs = [d for d in search_dirs if d.exists()]
    if not search_dirs:
        logger.error("No transcript directory found. Use --transcript-dir to specify one.")
        sys.exit(1)
    logger.info(f"Searching {len(search_dirs)} director(y/ies): {[str(d) for d in search_dirs]}")

    # Collect transcript files for target tickers
    transcript_files: list[Path] = []
    seen_keys: set[str] = set()  # deduplicate across dirs

    for d in search_dirs:
        for fp in sorted(d.glob("*_transcript_*.json")):
            parts = fp.stem.split("_transcript_")
            if len(parts) != 2:
                continue
            ticker = parts[0].upper()
            if target_tickers is not None and ticker not in target_tickers:
                continue
            key = fp.stem  # e.g. AAPL_transcript_2023_Q2
            if key not in seen_keys:
                seen_keys.add(key)
                transcript_files.append(fp)

    logger.info(f"Found {len(transcript_files)} transcript files to process")

    if not transcript_files:
        logger.warning("Nothing to do.")
        return

    # Connect and migrate
    conn = get_conn()
    ensure_columns(conn)

    # Process
    ok = fail = skip = 0
    for i, fp in enumerate(transcript_files, 1):
        try:
            result = process_transcript_file(fp, conn)
            if result:
                ok += 1
            else:
                skip += 1
        except Exception as e:
            logger.error(f"  ❌ {fp.name}: {e}")
            fail += 1

        if i % 500 == 0:
            logger.info(f"Progress: {i}/{len(transcript_files)} — ✅{ok} ⏭️{skip} ❌{fail}")

    conn.close()
    logger.info(f"\n{'='*60}")
    logger.info(f"Done: {ok} uploaded, {skip} skipped (no data), {fail} errors")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
