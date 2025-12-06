#!/usr/bin/env python3
"""
Test script to verify database connection and table creation
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ingest_10k_to_database import DatabaseIntegration
except ImportError as e:
    print(f"❌ Failed to import DatabaseIntegration: {e}")
    sys.exit(1)

def test_database_connection():
    """Test database connection and table creation"""
    print("=" * 80)
    print("DATABASE CONNECTION TEST")
    print("=" * 80)

    # Check environment variables
    pg_vector = os.getenv("PG_VECTOR", "")
    database_url = os.getenv("DATABASE_URL", "")

    print(f"\n📋 Environment Variables:")
    print(f"   PG_VECTOR: {'✅ Set' if pg_vector else '❌ Not set'}")
    print(f"   DATABASE_URL: {'✅ Set' if database_url else '❌ Not set'}")

    if not pg_vector and not database_url:
        print("\n❌ ERROR: Neither PG_VECTOR nor DATABASE_URL is set!")
        print("   Please set one of these environment variables in your .env file")
        return False

    # Test connection
    print(f"\n🔌 Testing database connection...")
    try:
        db_url = pg_vector or database_url
        print(f"   Using: {db_url[:50]}...")

        db_integration = DatabaseIntegration(db_url=db_url)
        conn = db_integration.get_connection()

        print("   ✅ Connection successful!")

        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"   PostgreSQL version: {version[:80]}...")

        # Check for pgvector extension
        cursor.execute("""
            SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';
        """)
        has_vector = cursor.fetchone()[0] > 0
        print(f"   pgvector extension: {'✅ Installed' if has_vector else '⚠️  Not installed'}")

        cursor.close()

        # Test table creation
        print(f"\n📊 Testing table creation...")
        db_integration.ensure_tables()

        # Verify tables exist
        conn = db_integration.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ten_k_chunks', 'ten_k_tables')
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cursor.fetchall()]

        print(f"   Tables found: {tables}")

        if 'ten_k_chunks' in tables and 'ten_k_tables' in tables:
            print("   ✅ All tables created successfully!")

            # Check row counts
            cursor.execute("SELECT COUNT(*) FROM ten_k_chunks")
            chunk_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM ten_k_tables")
            table_count = cursor.fetchone()[0]

            print(f"   Current data:")
            print(f"      ten_k_chunks: {chunk_count:,} rows")
            print(f"      ten_k_tables: {table_count:,} rows")
        else:
            print(f"   ❌ Missing tables: {set(['ten_k_chunks', 'ten_k_tables']) - set(tables)}")
            return False

        cursor.close()
        db_integration.close_connection()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_database_connection()
    sys.exit(0 if success else 1)
