#!/usr/bin/env python3
"""
Agent - Unified agent for financial Q&A

Simple wrapper around RAGAgent for financial data Q&A (earnings transcripts, 10-K filings, news).
"""

import os
import logging
from typing import Optional

from .rag.rag_agent import RAGAgent

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT CLASS - Main entry point for financial Q&A
# =============================================================================

class Agent:
    """
    Agent for financial Q&A - earnings transcripts, 10-K filings, and news.

    This is a thin wrapper around RAGAgent that provides:
    - Clean API for query execution
    - Lazy initialization for efficiency
    - Backward compatibility with old code
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize agent.

        Args:
            openai_api_key: OpenAI API key. If None, loads from environment.
        """
        logger.info("🚀 Initializing Agent")

        # Store API key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Lazy initialization - only create RAGAgent when needed
        self._rag_agent = None

        logger.info("✅ Agent initialized")


    # =========================================================================
    # PROPERTY ACCESSORS - Expose RAGAgent components
    # =========================================================================


    @property
    def rag_core(self):
        """Access RAG core - lazily initialized."""
        if self._rag_agent is None:
            self._rag_agent = RAGAgent(self.openai_api_key)
        return self._rag_agent

    @property
    def question_analyzer(self):
        """Access question analyzer for query parsing."""
        return self.rag_core.question_analyzer

    @property
    def response_generator(self):
        """Access response generator for LLM operations."""
        return self.rag_core.response_generator

    @property
    def search_engine(self):
        """Access search engine for vector/keyword search."""
        return self.rag_core.search_engine

    @property
    def database_manager(self):
        """Access database manager for PostgreSQL operations."""
        return self.rag_core.database_manager

    @property
    def analytics_logger(self):
        """Access analytics logger for tracking queries."""
        return self.rag_core.analytics_logger

    def set_database_connection(self, db_connection):
        """
        Set database connection for RAG operations.

        Args:
            db_connection: PostgreSQL connection object
        """
        if self._rag_agent is None:
            self._rag_agent = RAGAgent(self.openai_api_key)
        self._rag_agent.set_database_connection(db_connection)


    # =========================================================================
    # MAIN EXECUTION METHODS - Query execution entry points
    # =========================================================================


    async def execute_rag_flow(self, question: str, **kwargs):
        """
        *** MAIN ENTRY POINT (STREAMING) ***

        Execute RAG flow for financial Q&A with streaming events.

        Args:
            question: User's question
            **kwargs: Additional arguments:
                - show_details (bool): Print debug info
                - comprehensive (bool): Use comprehensive multi-ticker mode
                - stream_callback (callable): Callback for streaming
                - max_iterations (int): Number of improvement iterations
                - conversation_id (str): Conversation ID for memory
                - stream (bool): Whether to stream events

        Yields:
            Event dictionaries with progress updates, search results, and final answer
        """
        # Initialize RAG agent on first use
        if self._rag_agent is None:
            self._rag_agent = RAGAgent(self.openai_api_key)

        # Delegate to RAG agent for actual execution
        async for event in self._rag_agent.execute_rag_flow(question, **kwargs):
            yield event

    async def execute_flow(self, question: str, **kwargs):
        """Alias for execute_rag_flow (backward compatibility)."""
        async for event in self.execute_rag_flow(question, **kwargs):
            yield event

    async def execute_rag_flow_async(self, question: str, **kwargs):
        """
        *** MAIN ENTRY POINT (NON-STREAMING) ***

        Execute RAG flow and return final result only (no streaming).

        Args:
            question: User's question
            **kwargs: Same as execute_rag_flow

        Returns:
            Final result dict with answer, citations, and metadata
        """
        # Disable streaming
        kwargs['stream'] = False
        final_result = None

        # Collect events and extract final result
        async for event in self.execute_rag_flow(question, **kwargs):
            event_type = event.get('type')
            if event_type == 'result':
                final_result = event.get('data')
            elif event_type == 'final_answer':
                final_result = event

        return final_result

    async def execute_flow_async(self, question: str, **kwargs):
        """Alias for execute_rag_flow_async (backward compatibility)."""
        return await self.execute_rag_flow_async(question, **kwargs)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================



def create_agent(openai_api_key: Optional[str] = None) -> Agent:
    """
    Factory function to create a new Agent instance.

    Args:
        openai_api_key: OpenAI API key. If None, loads from environment.

    Returns:
        Initialized Agent instance
    """
    return Agent(openai_api_key)
