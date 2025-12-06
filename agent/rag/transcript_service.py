#!/usr/bin/env python3
"""
Transcript Service for handling complete transcript retrieval and viewing.

This module handles all operations related to complete transcript retrieval,
availability checking, and transcript viewing functionality separate from
the core RAG search functionality.
"""

import logging
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & SETUP
# ═════════════════════════════════════════════════════════════════════

class TranscriptService:
    """Service for handling complete transcript operations."""
    
    def __init__(self, database_manager):
        """Initialize the transcript service with database manager."""
        self.database_manager = database_manager
    
    def _get_db_connection(self):
        """Get database connection from the database manager."""
        return self.database_manager._get_db_connection()
    
    def _return_db_connection(self, conn):
        """Return database connection to the database manager."""
        self.database_manager._return_db_connection(conn)

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: TRANSCRIPT RETRIEVAL
    # ═════════════════════════════════════════════════════════════════════

    async def get_complete_transcript_async(self, ticker: str, year: int, quarter: int) -> Dict[str, Any]:
        """
        Asynchronously retrieve complete earnings transcript for a specific company and quarter from database.
        
        Args:
            ticker (str): Company ticker symbol
            year (int): Year of the transcript
            quarter (int): Quarter number (1-4)
            
        Returns:
            Dict containing transcript data or None if not found
        """
        logger.info(f"📄 Retrieving complete transcript (async) for {ticker} {year} Q{quarter}")
        
        try:
            # Initialize async pools if needed
            if not self.database_manager.pgvector_pool:
                await self.database_manager._initialize_async_pools()
            
            async with self.database_manager.pgvector_pool.acquire() as conn:
                # Use asyncpg-style parameterized query
                query = """
                SELECT transcript_id, ticker, company_name, date, year, quarter, 
                       full_transcript, metadata, created_at
                FROM complete_transcripts 
                WHERE ticker = $1 AND year = $2 AND quarter = $3
                LIMIT 1
                """
                
                result = await conn.fetchrow(query, ticker.upper(), year, quarter)
                
                if result:
                    transcript_data = {
                        'transcript_id': result['transcript_id'],
                        'ticker': result['ticker'],
                        'company_name': result['company_name'],
                        'date': result['date'],
                        'year': result['year'],
                        'quarter': result['quarter'],
                        'full_transcript': result['full_transcript'],
                        'metadata': result['metadata'],
                        'created_at': result['created_at']
                    }
                    logger.info(f"✅ Found complete transcript (async) for {ticker} {year} Q{quarter}")
                    return transcript_data
                else:
                    logger.info(f"❌ No complete transcript found (async) for {ticker} {year} Q{quarter}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error retrieving complete transcript (async) for {ticker} {year} Q{quarter}: {e}")
            return None

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: AVAILABILITY CHECKS
    # ═════════════════════════════════════════════════════════════════════

    async def check_chunks_availability_async(self, ticker: str, year: int, quarter: int) -> bool:
        """
        Check if transcript chunks exist in the database for the given ticker, year, and quarter.
        
        Args:
            ticker (str): Company ticker symbol
            year (int): Year of the transcript
            quarter (int): Quarter number (1-4)
            
        Returns:
            bool: True if chunks exist, False otherwise
        """
        logger.info(f"🔍 Checking chunks availability for {ticker} {year} Q{quarter}")
        
        try:
            # Initialize async pools if needed
            if not self.database_manager.pgvector_pool:
                await self.database_manager._initialize_async_pools()
            
            async with self.database_manager.pgvector_pool.acquire() as conn:
                # Use asyncpg-style parameterized query
                query = """
                SELECT COUNT(*) 
                FROM transcript_chunks 
                WHERE ticker = $1 AND year = $2 AND quarter = $3
                """
                
                result = await conn.fetchval(query, ticker.upper(), year, quarter)
                count = result or 0
                
                exists = count > 0
                logger.info(f"📄 Chunks availability for {ticker} {year} Q{quarter}: {exists} ({count} chunks)")
                return exists
                
        except Exception as e:
            logger.error(f"❌ Error checking chunks availability for {ticker} {year} Q{quarter}: {e}")
            return False
