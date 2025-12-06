#!/usr/bin/env python3
"""
Tavily Service for real-time news search.

This module provides integration with Tavily API to search for latest news
and current events, complementing the earnings transcript RAG system.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load .env file from the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


class TavilyService:
    """Service for searching latest news using Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Tavily service.
        
        Args:
            api_key (Optional[str]): Tavily API key. If None, will attempt to load from environment.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            logger.warning("⚠️ Tavily API key not found. News search will be disabled.")
            self.client = None
            self.available = False
            return
        
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(self.api_key)
            self.available = True
            logger.info("✅ Tavily service initialized successfully")
        except ImportError:
            logger.warning("⚠️ Tavily Python SDK not installed. Run: pip install tavily-python")
            self.client = None
            self.available = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Tavily client: {e}")
            self.client = None
            self.available = False
    
    def search_news(self, query: str, max_results: int = 5, include_answer: str = "advanced") -> Dict[str, Any]:
        """
        Search for latest news using Tavily.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return (default: 5)
            include_answer (str): Whether to include AI-generated answer ("advanced" or "basic" or None)
        
        Returns:
            Dict containing:
                - answer (str): AI-generated summary (if include_answer is set)
                - results (List[Dict]): List of news articles with:
                    - title (str)
                    - url (str)
                    - content (str)
                    - published_date (str)
                    - score (float)
                - query (str): Original query
                - response_time (float): Time taken for the search
        """
        if not self.available or not self.client:
            logger.warning("⚠️ Tavily service not available. Cannot search for news.")
            return {
                "answer": None,
                "results": [],
                "query": query,
                "error": "Tavily service not available"
            }
        
        try:
            import time
            start_time = time.time()
            
            rag_logger.info(f"🔍 Searching Tavily for: '{query}'")
            
            # Perform the search
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_answer=include_answer
            )
            
            response_time = time.time() - start_time
            
            # Extract and format results
            results = []
            if isinstance(response, dict):
                # Extract results from response
                raw_results = response.get("results", [])
                answer = response.get("answer", None)
                
                for result in raw_results:
                    formatted_result = {
                        "title": result.get("title", "No title"),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "published_date": result.get("published_date", ""),
                        "score": result.get("score", 0.0)
                    }
                    results.append(formatted_result)
                
                rag_logger.info(f"✅ Tavily search completed: {len(results)} results in {response_time:.3f}s")
                
                return {
                    "answer": answer,
                    "results": results,
                    "query": query,
                    "response_time": response_time
                }
            else:
                # Handle case where response might be a different format
                rag_logger.warning(f"⚠️ Unexpected Tavily response format: {type(response)}")
                return {
                    "answer": None,
                    "results": [],
                    "query": query,
                    "response_time": response_time,
                    "error": "Unexpected response format"
                }
                
        except Exception as e:
            logger.error(f"❌ Tavily search failed: {e}")
            return {
                "answer": None,
                "results": [],
                "query": query,
                "error": str(e)
            }
    
    def format_news_context(self, news_results: Dict[str, Any]) -> str:
        """
        Format news search results into context string for LLM with citation markers.
        
        Args:
            news_results (Dict): Results from search_news()
        
        Returns:
            Formatted string with news context including citation markers [N1], [N2], etc.
        """
        if not news_results.get("results"):
            return ""
        
        context_parts = ["\n=== LATEST NEWS (from Tavily) ===\n"]
        
        # Add AI-generated answer if available
        if news_results.get("answer"):
            context_parts.append(f"Summary: {news_results['answer']}\n")
        
        # Add individual articles with citation markers
        for i, result in enumerate(news_results.get("results", [])[:5], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            published_date = result.get("published_date", "")
            
            # Use [N1], [N2], etc. as citation markers for news
            citation_marker = f"[N{i}]"
            context_parts.append(f"\n{citation_marker} {title}")
            if published_date:
                context_parts.append(f"   Published: {published_date}")
            if url:
                context_parts.append(f"   Source: {url}")
            if content:
                # Truncate content if too long
                content_preview = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"   {content_preview}")
        
        context_parts.append("\n=== END NEWS ===\n")
        context_parts.append("\nNote: News citations are marked as [N1], [N2], etc. and should be referenced in your response.")
        
        return "\n".join(context_parts)
    
    def get_news_citations(self, news_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citations from news results for inclusion in response.
        
        Args:
            news_results (Dict): Results from search_news()
        
        Returns:
            List of citation dictionaries with format:
            {
                "type": "news",
                "index": int,
                "title": str,
                "url": str,
                "published_date": str
            }
        """
        citations = []
        if not news_results.get("results"):
            return citations
        
        for i, result in enumerate(news_results.get("results", [])[:5], 1):
            citation = {
                "type": "news",
                "index": i,
                "citation_marker": f"N{i}",
                "title": result.get("title", "No title"),
                "url": result.get("url", ""),
                "published_date": result.get("published_date", "")
            }
            citations.append(citation)
        
        return citations
    
    def is_available(self) -> bool:
        """Check if Tavily service is available."""
        return self.available

