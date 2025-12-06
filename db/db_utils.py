"""
Centralized database utilities for the StrataLens API
"""
import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global database pool - will be set by main server
_db_pool: Optional[asyncpg.Pool] = None

def set_db_pool(db_pool: asyncpg.Pool):
    """Set the database pool from the main server"""
    global _db_pool
    _db_pool = db_pool

async def get_db():
    """Get database connection from the pool"""
    if _db_pool is None:
        raise Exception("Database pool not initialized")
    
    async with _db_pool.acquire() as connection:
        yield connection