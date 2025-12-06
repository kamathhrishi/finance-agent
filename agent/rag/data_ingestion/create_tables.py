#!/usr/bin/env python3
"""
Simple script to create database tables for 10-K ingestion
Run this BEFORE running the ingestion scripts

IMPORTANT: These tables are fully compatible with:
- agent/rag/database_manager.py (search_10k_filings, get_all_tables_for_ticker)
- agent/rag/sec_filings_service.py (SEC RAG agent)
- The main RAG system for querying 10-K filings

Schema matches exactly what the RAG system expects.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

def create_tables():
    """Create the required database tables"""

    # Get database URL
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")

    if not db_url:
        print("❌ ERROR: PG_VECTOR or DATABASE_URL environment variable not set!")
        print("   Please set one of these in your .env file")
        return False

    print("=" * 80)
    print("CREATING DATABASE TABLES FOR 10-K INGESTION")
    print("=" * 80)
    print(f"\n📊 Database: {db_url[:50]}...\n")

    try:
        # Connect to database
        print("🔌 Connecting to database...")
        conn = psycopg2.connect(db_url, connect_timeout=30)
        cursor = conn.cursor()
        print("   ✅ Connected successfully\n")

        # Enable pgvector extension
        print("🔧 Enabling pgvector extension...")
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            print("   ✅ pgvector extension enabled\n")
        except Exception as e:
            print(f"   ⚠️  Warning: {e}")
            print("   Continuing anyway...\n")
            conn.rollback()

        # Create ten_k_chunks table
        print("📋 Creating ten_k_chunks table...")
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
        print("   ✅ ten_k_chunks table created\n")

        # Create ten_k_tables table
        print("📋 Creating ten_k_tables table...")
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
        print("   ✅ ten_k_tables table created\n")

        # Create indexes
        print("🔍 Creating indexes...")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ten_k_ticker_year
            ON ten_k_chunks(ticker, fiscal_year);
        """)
        print("   ✅ Index on (ticker, fiscal_year)")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ten_k_section
            ON ten_k_chunks(sec_section);
        """)
        print("   ✅ Index on sec_section")

        # Check if table has data before creating expensive vector index
        cursor.execute("SELECT COUNT(*) FROM ten_k_chunks")
        row_count = cursor.fetchone()[0]

        # Try to create vector index only if table is empty
        if row_count == 0:
            try:
                print("   Creating vector index (this may take a minute)...")
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ten_k_embedding
                    ON ten_k_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                print("   ✅ Vector index on embeddings (IVFFlat)")
            except Exception as e:
                print(f"   ⚠️  Could not create vector index: {e}")
                print("   (Search will still work, just slower)")
                conn.rollback()
        else:
            print(f"   ⚠️  Table has {row_count:,} rows - skipping vector index creation")
            print("   (Vector index is expensive to create on populated tables)")
            print("   (Create it manually later if needed: CREATE INDEX CONCURRENTLY ...)")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ten_k_tables_ticker_year
            ON ten_k_tables(ticker, fiscal_year);
        """)
        print("   ✅ Index on ten_k_tables (ticker, fiscal_year)")

        conn.commit()
        print("\n" + "=" * 80)

        # Verify tables exist
        print("\n✅ VERIFICATION")
        cursor.execute("""
            SELECT table_name,
                   pg_size_pretty(pg_total_relation_size(quote_ident(table_name)::regclass)) as size
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ten_k_chunks', 'ten_k_tables')
            ORDER BY table_name;
        """)

        tables = cursor.fetchall()
        for table_name, size in tables:
            print(f"   {table_name}: {size}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"      Rows: {count:,}")

        cursor.close()
        conn.close()

        print("\n" + "=" * 80)
        print("✅ SUCCESS! Database tables are ready for 10-K ingestion")
        print("=" * 80)
        print("\nYou can now run:")
        print("  python ingest_sp500_10k.py --max-tickers 10 --workers 3")
        print("  python ingest_10k_to_database.py --ticker AAPL")
        print()

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(f"\nFull traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = create_tables()
    sys.exit(0 if success else 1)
