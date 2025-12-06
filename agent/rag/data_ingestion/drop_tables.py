#!/usr/bin/env python3
"""
Drop 10-K tables to start fresh
Run this to clear all 10-K data and start over
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

def drop_tables(force=False):
    """Drop the 10-K tables"""

    # Get database URL
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")

    if not db_url:
        print("❌ ERROR: PG_VECTOR or DATABASE_URL environment variable not set!")
        return False

    print("=" * 80)
    print("DROPPING 10-K TABLES")
    print("=" * 80)
    print(f"\n⚠️  WARNING: This will DELETE ALL 10-K data!")
    print(f"Database: {db_url[:50]}...\n")

    # Ask for confirmation unless --force is used
    if not force:
        response = input("Are you sure you want to drop all 10-K tables? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborted - no tables were dropped")
            return False
    else:
        print("⚡ Running with --force flag, skipping confirmation\n")

    try:
        print("\n🔌 Connecting to database...")
        conn = psycopg2.connect(db_url, connect_timeout=30)
        cursor = conn.cursor()
        print("   ✅ Connected\n")

        # Get row counts before dropping
        try:
            cursor.execute("SELECT COUNT(*) FROM ten_k_chunks")
            chunks_count = cursor.fetchone()[0]
            print(f"📊 Current data:")
            print(f"   ten_k_chunks: {chunks_count:,} rows")
        except:
            print(f"📊 ten_k_chunks table doesn't exist yet")
            chunks_count = 0

        try:
            cursor.execute("SELECT COUNT(*) FROM ten_k_tables")
            tables_count = cursor.fetchone()[0]
            print(f"   ten_k_tables: {tables_count:,} rows\n")
        except:
            print(f"   ten_k_tables table doesn't exist yet\n")
            tables_count = 0

        # Drop tables
        print("🗑️  Dropping ten_k_chunks table...")
        cursor.execute("DROP TABLE IF EXISTS ten_k_chunks CASCADE")
        conn.commit()
        print("   ✅ Dropped\n")

        print("🗑️  Dropping ten_k_tables table...")
        cursor.execute("DROP TABLE IF EXISTS ten_k_tables CASCADE")
        conn.commit()
        print("   ✅ Dropped\n")

        cursor.close()
        conn.close()

        print("=" * 80)
        print("✅ SUCCESS! All 10-K tables have been dropped")
        print("=" * 80)
        print("\nYou can now run:")
        print("  python create_tables.py")
        print("  python ingest_sp500_10k.py --workers 5")
        print()

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(f"\nFull traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop 10-K tables from database")
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompt and drop tables immediately')
    args = parser.parse_args()

    success = drop_tables(force=args.force)
    sys.exit(0 if success else 1)
