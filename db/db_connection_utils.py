"""
Shared database connection utilities for PostgreSQL connections
"""

import os
import psycopg2
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def get_postgres_connection():
    """Get PostgreSQL connection for financial data"""
    # Try POSTGRES_DB_2 first, then DATABASE_URL as fallback
    postgres_url = os.getenv("POSTGRES_DB_2") or os.getenv("DATABASE_URL")
    if not postgres_url:
        raise HTTPException(status_code=503, detail="PostgreSQL connection not configured")
    
    try:
        conn = psycopg2.connect(postgres_url)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")
