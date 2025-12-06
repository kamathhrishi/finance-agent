#!/usr/bin/env python3
"""
Screener Agent - Financial data screening and filtering using DuckDB

This agent provides stock screening capabilities using the FinancialDataAnalyzer
from the duckdb_module. It can handle queries like:
- Find stocks with specific criteria (P/E ratios, market cap, sector, etc.)
- Screen companies based on financial metrics
- Filter stocks by various fundamental indicators
"""

import os
import logging
from typing import Dict, List, Any, Optional
import asyncio

# Import the DuckDB Financial Data Analyzer
from agent.screener import FinancialDataAnalyzer

logger = logging.getLogger(__name__)


class ScreenerAgent:
    """
    Screener Agent that wraps the DuckDB FinancialDataAnalyzer for stock screening.

    This agent is specialized for handling screening queries like:
    - "Find tech stocks with P/E ratio less than 20"
    - "Show me all healthcare companies with market cap over $10B"
    - "List stocks in the energy sector with positive revenue growth"
    """

    def __init__(self, openai_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
        """Initialize the Screener Agent.

        Args:
            openai_api_key (Optional[str]): OpenAI API key for LLM operations
            groq_api_key (Optional[str]): Groq API key for LLM operations
        """
        logger.info("🔍 Initializing Screener Agent")

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for Screener Agent")

        # Initialize the Financial Data Analyzer
        try:
            self.analyzer = FinancialDataAnalyzer(
                api_key=self.openai_api_key,
                groq_api_key=self.groq_api_key
            )
            logger.info("✅ Screener Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Screener Agent: {e}")
            raise

    def is_screening_query(self, question: str) -> bool:
        """Determine if a question is a screening query.

        Args:
            question (str): The user's question

        Returns:
            bool: True if this is a screening query
        """
        screening_keywords = [
            'screen', 'find', 'list', 'show', 'filter', 'stocks with',
            'companies with', 'which stocks', 'what stocks', 'top stocks',
            'companies that have', 'stocks that have', 'p/e ratio', 'pe ratio',
            'market cap', 'revenue', 'profit', 'debt', 'ratio', 'sector',
            'industry', 'criteria', 'screening'
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in screening_keywords)

    async def screen_stocks(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """Execute a stock screening query.

        Args:
            query (str): The screening query
            stream_callback (callable, optional): Callback for streaming events

        Returns:
            Dict with screening results including:
                - answer: Natural language answer
                - data: Structured data (DataFrame as dict)
                - sql_query: The SQL query used
                - metadata: Additional information
        """
        logger.info(f"🔍 Executing screening query: {query[:100]}...")

        try:
            # Send initial event if streaming
            if stream_callback:
                await stream_callback({
                    'type': 'screener_start',
                    'message': 'Analyzing screening criteria...'
                })

            # Use the analyzer to process the query
            # The analyzer has a query_pipeline method that handles the full flow
            result = await asyncio.to_thread(
                self.analyzer.query_pipeline,
                query
            )

            # Send completion event if streaming
            if stream_callback:
                await stream_callback({
                    'type': 'screener_complete',
                    'message': 'Screening complete'
                })

            logger.info("✅ Screening query completed successfully")

            return {
                'success': True,
                'answer': result.get('answer', ''),
                'data': result.get('data', None),
                'sql_query': result.get('sql_query', ''),
                'metadata': {
                    'row_count': len(result.get('data', [])) if result.get('data') else 0,
                    'execution_time': result.get('execution_time', 0),
                    'reasoning_events': result.get('reasoning_events', [])
                },
                'type': 'screener'
            }

        except Exception as e:
            logger.error(f"❌ Screening query failed: {e}")

            if stream_callback:
                await stream_callback({
                    'type': 'screener_error',
                    'message': f'Screening failed: {str(e)}'
                })

            return {
                'success': False,
                'error': str(e),
                'answer': f"I encountered an error while screening: {str(e)}",
                'type': 'screener'
            }

    async def execute_screening_flow(self, question: str, stream_callback=None, **kwargs) -> Dict[str, Any]:
        """Execute the complete screening flow.

        This is the main entry point that Master Agent calls.

        Args:
            question (str): The user's question
            stream_callback (callable, optional): Callback for streaming events
            **kwargs: Additional arguments

        Returns:
            Dict with screening results
        """
        logger.info(f"🚀 Starting screening flow for: {question[:100]}...")

        # Execute the screening
        result = await self.screen_stocks(question, stream_callback)

        return result

    def get_available_sectors(self) -> List[str]:
        """Get list of available sectors for screening."""
        return self.analyzer.available_sectors

    def get_available_industries(self) -> List[str]:
        """Get list of available industries for screening."""
        return self.analyzer.available_industries


def create_screener_agent(openai_api_key: Optional[str] = None,
                          groq_api_key: Optional[str] = None) -> ScreenerAgent:
    """Create a new Screener Agent instance.

    Args:
        openai_api_key (Optional[str]): OpenAI API key
        groq_api_key (Optional[str]): Groq API key

    Returns:
        ScreenerAgent: Initialized screener agent
    """
    return ScreenerAgent(openai_api_key, groq_api_key)